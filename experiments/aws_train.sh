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
# Namespace the checkpoint prefix by CONFIG, not a fixed string: otherwise a smoke run
# (tiny 410M->160M) writes to the same object name as the real 12B->6.9B run and a later
# --resume would try to load a 160M checkpoint into a 6.9B model.
EXPT=$(basename "$CFG" .yaml)
S3CK="s3://$BUCKET/$EXPT/ckpt"
# Which work THIS instance owns. Disjoint slices let several boxes run at once without
# ever touching the same run: checkpoints are one S3 object per (arm,seed), written by
# exactly one process. e.g. SEEDS="0 1" on box A, SEEDS="2" on box B.
SEEDS=${SEEDS:-"0 1 2"}
ARMS=${ARMS:-"sub shrink ridge"}
DO_B69=${DO_B69:-1}      # 12B->6.9B screen
DO_D12=${DO_D12:-1}      # 12B->1.4B ablation
G="--config $CFG --ckpt-every 2000000 --resume --s3-ckpt $S3CK"

# m5_train.py pushes/pulls its OWN checkpoint object (--s3-ckpt), so there is no box-wide
# sync to race: each run restores only its own state and writes only its own object.
sync_up() { aws s3 sync data/m5/ckpt/ "$S3CK/" --only-show-errors 2>>"$LOG/s3.log"; }
mkdir -p data/m5/ckpt aws_logs
echo "[aws_train] per-run S3 checkpoints under $S3CK ; seeds='$SEEDS' arms='$ARMS'"
SYNCER=""

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

cleanup(){ kill ${SYNCER:-} $SPOTW 2>/dev/null; sync_up; }   # belt-and-braces final flush
trap cleanup EXIT

# Guard: the shipped tree must be the Paper-2 branch. Master's m5_train.py has none of
# these flags, and every job dies instantly with "unrecognized arguments".
for flag in comp-reg ckpt-every optimizer grad-checkpoint; do
  grep -q -- "--$flag" "$M" || { echo "FATAL: $M lacks --$flag (wrong branch shipped)"; exit 1; }
done
echo "[aws_train] code check OK (Paper-2 flags present)"

# one-time: warm the HF cache so 8 processes don't race the same 24GB download
echo "[aws_train] pre-fetching the 12B donor once (avoids an 8-way download race)..."
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('EleutherAI/pythia-12b')" \
  > "$LOG/prefetch.log" 2>&1 || { echo "prefetch FAILED — see $LOG/prefetch.log"; exit 1; }

# One-time: build the donor's activation cache. residual_selection() ranks which residual
# dims to keep from data/activations/<donor>/mean_pooled.pt; that cache ships with the repo
# for 6.9B and smaller but has never been built for 12B, so every arm dies with
# FileNotFoundError: .../activations/12b/mean_pooled.pt. Build it once, before any run.
DONOR=$(grep -E '^large:' "$CFG" | sed 's/.*"\(.*\)".*/\1/')
if [ ! -s "data/activations/$DONOR/mean_pooled.pt" ]; then
  echo "[aws_train] building activation cache for donor $DONOR (one-time)..."
  uv run python -c "
from lib.activations import capture_and_cache
capture_and_cache('$DONOR', batch_size=8)
print('cache built')
" > "$LOG/actcache.log" 2>&1 || { echo "FATAL: activation cache build failed — see $LOG/actcache.log"; tail -5 "$LOG/actcache.log"; exit 1; }
  echo "[aws_train] cache ready: $(du -sh data/activations/$DONOR 2>/dev/null | cut -f1)"
else
  echo "[aws_train] activation cache for $DONOR already present"
fi

TRAIN_PIDS=()          # only these are waited on; a bare `wait` would also block on the
                       # never-ending spot-interruption watcher and hang the runner forever
wait_for_training() { [ ${#TRAIN_PIDS[@]} -gt 0 ] && wait "${TRAIN_PIDS[@]}" 2>/dev/null; TRAIN_PIDS=(); }

launch2() {  # launch2 <gpu> <tag> <extra args...>  -- 12B->1.4B ablation config
  local gpu=$1 tag=$2; shift 2
  if grep -aq "FINAL full" "$LOG/$tag.log" 2>/dev/null; then echo "[aws_train] SKIP $tag (done)"; return; fi
  echo "[aws_train] GPU$gpu <- $tag (12B->1.4B)"
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run python $M --config configs/m5_12b14b.yaml \
    --ckpt-every 2000000 --resume --s3-ckpt "$S3CK" "$@" --tag "_$tag" >> "$LOG/$tag.log" 2>&1 &
  TRAIN_PIDS+=($!)
}

launch() {   # launch <gpu> <tag> <extra args...>
  local gpu=$1 tag=$2; shift 2
  if grep -aq "FINAL full" "$LOG/$tag.log" 2>/dev/null; then echo "[aws_train] SKIP $tag (done)"; return; fi
  echo "[aws_train] GPU$gpu <- $tag"
  CUDA_VISIBLE_DEVICES=$gpu nohup uv run python $M $G "$@" --tag "_$tag" \
    >> "$LOG/$tag.log" 2>&1 &
  TRAIN_PIDS+=($!)
}

# --- 12B->6.9B screen: only this box's (arm,seed) slice, packed across its GPUs ---
i=0
if [ "$DO_B69" = 1 ]; then
  for s_ in $SEEDS; do
    for a in $ARMS; do
      case $a in
        sub)    launch $((i++)) b69_sub_s$s_    --init subclone_rs                 --seed $s_;;
        shrink) launch $((i++)) b69_shrink_s$s_ --init hybrid_rs --comp-reg shrink --seed $s_;;
        ridge)  launch $((i++)) b69_ridge_s$s_  --init hybrid_rs --comp-reg ridge  --seed $s_;;
      esac
      [ "$i" -ge 8 ] && { echo "[aws_train] 8 GPUs busy; waiting for this wave"; wait_for_training; i=0; }
    done
  done
  wait_for_training
fi

# --- 12B->1.4B donor-scale ablation (same donor, 8192^2 solve) ---
if [ "$DO_D12" = 1 ]; then
  echo "[aws_train] 12B->1.4B ablation for seeds '$SEEDS'"
  j=0
  for s_ in $SEEDS; do
    for a in $ARMS; do
      case $a in
        sub)    launch2 $((j++)) d12_sub_s$s_    --init subclone_rs                 --seed $s_;;
        shrink) launch2 $((j++)) d12_shrink_s$s_ --init hybrid_rs --comp-reg shrink --seed $s_;;
        ridge)  launch2 $((j++)) d12_ridge_s$s_  --init hybrid_rs --comp-reg ridge  --seed $s_;;
      esac
      [ "$j" -ge 8 ] && { wait_for_training; j=0; }
    done
  done
  wait_for_training
fi

sync_up
touch "$LOG/TRAIN_DONE"      # the launcher polls for this instead of holding an ssh session
echo "[aws_train] ALL DONE"
for f in "$LOG"/b69_*.log "$LOG"/d12_*.log; do
  printf "%-18s %s\n" "$(basename "$f" .log)" \
    "$(grep -a 'FINAL full' "$f" 2>/dev/null | tail -1 | sed 's/.*ppl=//; s/ .*//' || echo INCOMPLETE)"
done
