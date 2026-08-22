#!/usr/bin/env bash
# GPU capacity (esp. spot p5/p4de) comes and goes. Retry across regions and pricing tiers
# until one is obtainable, then run the 12B->6.9B screen. aws_screen.sh owns termination
# (EXIT trap + an on-instance dead-man switch), so a success can never bill unattended.
#
#   REGIONS="us-east-1 us-west-2" MAX_TRIES=96 INTERVAL=1800 \
#     ./aws_wait_and_run.sh --spot --types "p5.48xlarge p4de.24xlarge"
#
# Tier 1 = whatever you pass (spot). Tier 2 = on-demand p4de, which has capacity priority
# over spot. Tiers run sequentially and aws_screen.sh exits after a success, so at most one
# instance can ever exist. Full per-attempt detail: aws_last_attempt.log.
INTERVAL=${INTERVAL:-900}
MAX_TRIES=${MAX_TRIES:-48}
REGIONS=${REGIONS:-us-east-1}
ONDEMAND=${ONDEMAND:-1}          # set 0 to stay spot-only
cd "$(dirname "$0")/.." || exit 1

TIER1=("$@")                       # arrays keep --types "a b" as ONE argument
TIER2=(--types p4de.24xlarge)      # on-demand fallback (no --spot)

for i in $(seq 1 "$MAX_TRIES"); do
  echo "=== [$(date +%FT%T)] capacity attempt $i/$MAX_TRIES ==="
  LOGF="aws_attempt_${i}.log"
  for R in $REGIONS; do
    for tier in 1 2; do
      [ "$tier" = 2 ] && [ "$ONDEMAND" != 1 ] && continue
      if [ "$tier" = 1 ]; then ARGS=("${TIER1[@]}"); LBL="spot"; else ARGS=("${TIER2[@]}"); LBL="on-demand p4de"; fi
      echo "--- region $R | $LBL ---"
      : > "$LOGF"
      ./experiments/aws_screen.sh --region "$R" "${ARGS[@]}" 2>&1 | tee -a "$LOGF" \
        | grep -E "trying|unavailable|launched i-|FAILED|ALL DONE|training complete" | tail -6
      if grep -qE "ALL DONE|training complete" "$LOGF"; then
        echo "AWS RUN COMPLETED (attempt $i, region $R, $LBL)"; cp "$LOGF" aws_success.log; exit 0
      fi
      # A capacity shortfall is expected; anything else is a real error worth stopping for.
      if ! grep -q "FAILED in all AZs" "$LOGF"; then
        echo "stopped for a non-capacity reason — see $LOGF"; cat "$LOGF" | tail -5; exit 1
      fi
    done
  done
  mv -f "$LOGF" aws_last_attempt.log 2>/dev/null
  sleep "$INTERVAL"
done
echo "gave up after $MAX_TRIES attempts (~$((MAX_TRIES*INTERVAL/3600))h): no capacity in $REGIONS"
exit 2
