#!/usr/bin/env bash
# GPU capacity (esp. spot p5/p4de) comes and goes. Retry across regions until one is
# obtainable, then run the 12B->6.9B screen. aws_screen.sh owns termination (EXIT trap +
# an on-instance dead-man switch), so a success here can never bill unattended.
#
#   REGIONS="us-east-1 us-west-2" MAX_TRIES=96 INTERVAL=1800 \
#     ./aws_wait_and_run.sh --spot --types "p5.48xlarge p4de.24xlarge"
#
# Full per-attempt detail lands in aws_last_attempt.log (the console view is filtered).
INTERVAL=${INTERVAL:-900}
MAX_TRIES=${MAX_TRIES:-48}
REGIONS=${REGIONS:-us-east-1}
cd "$(dirname "$0")/.." || exit 1

for i in $(seq 1 "$MAX_TRIES"); do
  echo "=== [$(date +%FT%T)] capacity attempt $i/$MAX_TRIES ==="
  LOGF="aws_attempt_${i}.log"; : > "$LOGF"
  for R in $REGIONS; do
    echo "--- region $R ---"
    ./experiments/aws_screen.sh --region "$R" "$@" 2>&1 | tee -a "$LOGF" \
      | grep -E "trying|unavailable|launched i-|FAILED|ALL DONE" | tail -6
    if grep -q "ALL DONE" "$LOGF"; then
      echo "AWS RUN COMPLETED (attempt $i, region $R)"; cp "$LOGF" aws_success.log; exit 0
    fi
    # Anything other than a capacity shortfall is a real error worth stopping for.
    if ! grep -q "FAILED in all AZs" "$LOGF"; then
      echo "stopped for a non-capacity reason — see $LOGF"; exit 1
    fi
    : > "$LOGF"      # reset so the next region's verdict is judged on its own output
  done
  mv -f "$LOGF" aws_last_attempt.log 2>/dev/null
  sleep "$INTERVAL"
done
echo "gave up after $MAX_TRIES attempts (~$((MAX_TRIES*INTERVAL/3600))h): no capacity in $REGIONS"
exit 2
