#!/usr/bin/env bash
# p4de capacity is exhausted right now. Retry until AWS has one, then run the screen.
# Bounded: gives up after MAX_TRIES. aws_screen.sh owns termination (trap + on-instance
# dead-man switch), so a success here can never bill unattended.
INTERVAL=${INTERVAL:-900}      # 15 min
MAX_TRIES=${MAX_TRIES:-48}     # ~12 h
cd "$(dirname "$0")/.." || exit 1
for i in $(seq 1 "$MAX_TRIES"); do
  echo "=== [$(date +%FT%T)] capacity attempt $i/$MAX_TRIES ==="
  if ./experiments/aws_screen.sh "$@" 2>&1 | tee "aws_attempt_${i}.log" | grep -E "trying|unavailable|launched|FAILED|ALL DONE" | tail -8; then
    if grep -q "ALL DONE\|results ->" "aws_attempt_${i}.log"; then echo "AWS RUN COMPLETED on attempt $i"; exit 0; fi
  fi
  grep -q "FAILED in all AZs" "aws_attempt_${i}.log" || { echo "stopped for a non-capacity reason — see aws_attempt_${i}.log"; exit 1; }
  mv -f "aws_attempt_${i}.log" aws_last_attempt.log
  sleep "$INTERVAL"
done
echo "gave up after $MAX_TRIES attempts (~$((MAX_TRIES*INTERVAL/3600))h): no p4de capacity"
exit 2
