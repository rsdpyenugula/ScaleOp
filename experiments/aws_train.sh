#!/usr/bin/env bash
# Runs the 12B->6.9B screen on ONE 8-GPU box as 8 PARALLEL single-GPU jobs.
# No FSDP needed: a 6.9B target with Adafactor (~56GB state) + gradient checkpointing
# fits in a single 80GB A100/H100, so each arm gets its own GPU via CUDA_VISIBLE_DEVICES.
#
# SPOT-SAFE: resumable checkpoints are mirrored to S3, restored on start, and flushed on
# a spot interruption notice -- so a reclaimed instance costs minutes, not the whole run.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH"
LOG=${LOG:-$PWD/aws_logs}; mkdir -p "$LOG"
M=experiments/m5_train.py
CFG=${CFG:-configs/m5_12b69b.yaml}
BUCKET=${BUCKET:-de-aiml-scaleop-662022802750}
S3CK="s3://$BUCKET/12b69b/ckpt"
G="--config $CFG --ckpt-every 2000000 --resume"

sync_up()   { aws s3 sync data/m5/ckpt/ "$S3CK/" --only-show-errors 2>>"$LOG/s3.log"; }
sync_down() { aws s3 sync "$S3CK/" data/m5/ckpt/ --only-show-errors 2>>"$LOG/s3.log"; }

# --- restore any checkpoints from a previous (interrupted) attempt ---
mkdir -p data/m5/ckpt
echo "[aws_train] restoring checkpoints from $S3CK (if any)..."
sync_down; ls -la data/m5/ckpt/ 2>/dev/null | tail -n +2 | awk '{print "  have", $NF, $5}' | head

# --- background: mirror checkpoints every 10 min ---
( while true; do sleep 600; sync_up; done ) & SYNCER=$!

# --- background: spot interruption notice -> final flush, then let the box die ---
( TOK=$(curl -sX PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null)
  while true; do
    code=$(curl -s -o /dev/null -w '%{http_code}' -H "X-aws-ec2-metadata-token: $TOK" \
             http://169.254.169.254/latest/meta-data/spot/instance-action 2>/dev/null)
    if [ "$code" = "200" ]; then
      echo "[aws_train] SPOT INTERRUPTION NOTICE — flushing checkpoints to S3" | tee -a "$LOG/s3.log"
      sync_up; break
    fi
    sleep 5
  done ) & SPOTW=$!

cleanup(){ kill $SYNCER $SPOTW 2>/dev/null; sync_up; }
trap cleanup EXIT

# one-time: warm the HF cache so 8 processes don't race the same 24GB download
echo "[aws_train] pre-fetching the 12B donor once (avoids an 8-way download race)..."
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('EleutherAI/pythia-12b')" \
  > "$LOG/prefetch.log" 2>&1 || { echo "prefetch FAILED — see $LOG/prefetch.log"; exit 1; }

launch2() {  # launch2 <gpu> <tag> <extra args...>  -- 12B->1.4B ablation config
  local gpu=$1 tag=$2; shift 2
  if grep -aq "FINAL full" "$LOG/$tag.log" 2>/dev/null; then echo "[aws_train] SKIP $tag (done)"; return; fi
  echo "[aws_train] GPU$gpu <- $tag (12B->1.4B)"
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run python $M --config configs/m5_12b14b.yaml \
    --ckpt-every 2000000 --resume "$@" --tag "_$tag" >> "$LOG/$tag.log" 2>&1 &
}

launch() {   # launch <gpu> <tag> <extra args...>
  local gpu=$1 tag=$2; shift 2
  if grep -aq "FINAL full" "$LOG/$tag.log" 2>/dev/null; then echo "[aws_train] SKIP $tag (done)"; return; fi
  echo "[aws_train] GPU$gpu <- $tag"
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run python $M $G "$@" --tag "_$tag" \
    >> "$LOG/$tag.log" 2>&1 &
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

# --- 12B->1.4B donor-scale ablation: same donor, 8192^2 solve. Cannot run on the
# --- Spark (unified memory), so it rides along here where host RAM is separate.
echo "[aws_train] 12B->1.4B ablation (3 arms x 3 seeds across the GPUs)"
j=0
for s in 0 1 2; do
  launch2 $((j++)) d12_shrink_s$s --init hybrid_rs --comp-reg shrink --seed $s
  launch2 $((j++)) d12_ridge_s$s  --init hybrid_rs --comp-reg ridge  --seed $s
  launch2 $((j++)) d12_sub_s$s    --init subclone_rs                --seed $s
done
wait

sync_up
echo "[aws_train] ALL DONE"
for f in "$LOG"/b69_*.log; do
  printf "%-18s %s\n" "$(basename "$f" .log)" \
    "$(grep -a 'FINAL full' "$f" 2>/dev/null | tail -1 | grep -oE 'ppl=[0-9.]+' || echo INCOMPLETE)"
done
