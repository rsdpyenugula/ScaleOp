#!/usr/bin/env bash
# M5 watchdog — hourly cron on the Spark. Ensures the two 100M recovery runs
# finish: if nothing is training, relaunch whichever run lacks a FINAL line
# (subclone first, then hybrid). flock prevents overlapping instances; appends
# to the run logs so FINAL detection spans restarts. Remove the crontab line
# when the 100M runs are done.
set -u
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/projects/ScaleOp" || exit 1
exec 9>/tmp/m5_watchdog.lock
flock -n 9 || exit 0                               # another instance is active

# /tmp is cleared on reboot — regenerate the 100M config if missing.
[ -f /tmp/m5x.yaml ] || sed "s/tokens: 30000000/tokens: 100000000/" configs/m5.yaml > /tmp/m5x.yaml

ts() { date "+%F %T"; }
if pgrep -f m5_train >/dev/null; then
  echo "$(ts) training alive"
elif ! grep -q FINAL /tmp/m5x_subclone.log 2>/dev/null; then
  echo "$(ts) relaunching subclone 100M"
  nohup env PYTHONUNBUFFERED=1 uv run python experiments/m5_train.py \
    --config /tmp/m5x.yaml --init subclone >> /tmp/m5x_subclone.log 2>&1 &
elif ! grep -q FINAL /tmp/m5x_hybrid2.log 2>/dev/null; then
  echo "$(ts) relaunching hybrid 100M"
  nohup env PYTHONUNBUFFERED=1 uv run python experiments/m5_train.py \
    --config /tmp/m5x.yaml --init hybrid >> /tmp/m5x_hybrid2.log 2>&1 &
else
  echo "$(ts) all runs complete"
fi
