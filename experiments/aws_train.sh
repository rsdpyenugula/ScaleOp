#!/usr/bin/env bash
# Runs the 12B->6.9B screen on ONE 8-GPU box as 8 PARALLEL single-GPU jobs.
# No FSDP needed: a 6.9B target with Adafactor (~56GB state) + gradient checkpointing
# fits in a single 80GB A100, so each arm gets its own GPU via CUDA_VISIBLE_DEVICES.
# 9 runs over 8 GPUs = one full wave of 8 + one trailing run.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH"
LOG=${LOG:-$PWD/aws_logs}; mkdir -p "$LOG"
M=experiments/m5_train.py
CFG=${CFG:-configs/m5_12b69b.yaml}
G="--config $CFG --ckpt-every 1000000 --resume"

# one-time: warm the HF cache so 8 processes don't race the same 24GB download
echo "[aws_train] pre-fetching the 12B donor once (avoids 8-way download race)..."
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('EleutherAI/pythia-12b')" \
  > "$LOG/prefetch.log" 2>&1 || { echo "prefetch FAILED — see $LOG/prefetch.log"; exit 1; }

launch() {   # launch <gpu> <tag> <extra args...>
  local gpu=$1 tag=$2; shift 2
  if grep -aq "FINAL full" "$LOG/$tag.log" 2>/dev/null; then echo "[aws_train] SKIP $tag (done)"; return; fi
  echo "[aws_train] GPU$gpu <- $tag"
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run python $M $G "$@" --tag "_$tag" \
    > "$LOG/$tag.log" 2>&1 &
}

# --- wave 1: 8 runs, one per GPU (3 arms x seeds 0,1 + 2 of seed 2) ---
i=0
for s in 0 1; do
  launch $((i++)) b69_sub_s$s    --init subclone_rs                 --seed $s
  launch $((i++)) b69_shrink_s$s --init hybrid_rs --comp-reg shrink --seed $s
  launch $((i++)) b69_ridge_s$s  --init hybrid_rs --comp-reg ridge  --seed $s
done
launch $((i++)) b69_sub_s2    --init subclone_rs                 --seed 2
launch $((i++)) b69_shrink_s2 --init hybrid_rs --comp-reg shrink --seed 2
echo "[aws_train] wave 1: $i jobs launched; waiting..."
wait

# --- wave 2: the trailing run ---
launch 0 b69_ridge_s2 --init hybrid_rs --comp-reg ridge --seed 2
wait

echo "[aws_train] ALL DONE"
for f in "$LOG"/b69_*.log; do
  printf "%-18s %s\n" "$(basename "$f" .log)" \
    "$(grep -a 'FINAL full' "$f" 2>/dev/null | tail -1 | grep -oE 'ppl=[0-9.]+' || echo FAILED)"
done
