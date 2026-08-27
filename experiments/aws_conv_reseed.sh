#!/usr/bin/env bash
# 1B-token convergence RESEED (Paper 1, Kumar review): seeds 1,2 x {hybrid_rs, subclone_rs}
# on one multi-GPU box as 4 PARALLEL single-GPU jobs (m5_conv.yaml is sized for a 40GB A100).
# Same flow as the original seed-0 run: on-demand box, master code, detached + polled,
# artifacts pulled back, instance terminated on exit.
#
#   ./aws_conv_reseed.sh [--profile prod_admin] [--dry-run]
set -uo pipefail

PROFILE="${AWS_PROFILE:-prod_admin}"
REGION="us-east-1"
TYPES="${TYPES:-p4d.24xlarge p4de.24xlarge}"   # tried in order; 4 GPUs used either way
KEY_NAME="de-aiml"
SG_ID="${SG_ID:-}"
BUCKET="de-aiml-scaleop-662022802750"
DRYRUN=0
AMI_NAME='Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*'

while [ $# -gt 0 ]; do case "$1" in
  --profile) PROFILE="$2"; shift 2;;
  --region)  REGION="$2";  shift 2;;
  --types)   TYPES="$2"; shift 2;;
  --dry-run) DRYRUN=1; shift;;
  *) echo "unknown arg $1"; exit 1;;
esac; done

AWS=(aws --profile "$PROFILE" --region "$REGION")
PEM="$HOME/.ssh/${KEY_NAME}.pem"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -i "$PEM")

echo "[aws] profile=$PROFILE region=$REGION types='$TYPES'"
"${AWS[@]}" sts get-caller-identity >/dev/null || { echo "not authenticated: aws sso login --profile $PROFILE"; exit 1; }
[ -f "$PEM" ] || { echo "missing private key $PEM"; exit 1; }

SUBNETS="AUTO $("${AWS[@]}" ec2 describe-subnets --filters Name=default-for-az,Values=true \
  --query 'sort_by(Subnets,&AvailabilityZone)[].SubnetId' --output text | tr '\t' ' ')"
if [ -z "$SG_ID" ]; then
  SG_ID=$("${AWS[@]}" ec2 describe-security-groups --filters Name=group-name,Values=de-aiml-ssh \
          --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null)
  [ "$SG_ID" = "None" ] && { echo "SG de-aiml-ssh not found; create it or pin SG_ID"; exit 1; }
fi
MYIP=$(curl -s --max-time 8 https://checkip.amazonaws.com)
if [ -n "$MYIP" ] && ! "${AWS[@]}" ec2 describe-security-groups --group-ids "$SG_ID" \
     --query 'SecurityGroups[0].IpPermissions[].IpRanges[].CidrIp' --output text 2>/dev/null | grep -q "$MYIP/32"; then
  "${AWS[@]}" ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 \
    --cidr "$MYIP/32" >/dev/null 2>&1 && echo "[aws] authorized SSH from $MYIP/32"
fi

AMI=$("${AWS[@]}" ec2 describe-images --owners amazon \
  --filters "Name=name,Values=$AMI_NAME" "Name=state,Values=available" \
  --query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' --output text)
echo "[aws] AMI=$AMI SG=$SG_ID"
[ "$DRYRUN" = 1 ] && { echo "[dry-run] would launch ${TYPES%% *} from $AMI; stopping."; exit 0; }

# Dead-man switch: box halts (and terminates) after MAX_HOURS even if this script dies.
MAX_HOURS="${MAX_HOURS:-16}"
USERDATA=$(printf '#!/bin/bash\nsetsid nohup bash -c "sleep %d; /sbin/shutdown -h now" >/dev/null 2>&1 &\n' $((MAX_HOURS*3600)) | base64)
IID=""
for INSTANCE_TYPE in $TYPES; do
for SUBNET_ID in $SUBNETS; do
  echo "[aws] trying $INSTANCE_TYPE in ${SUBNET_ID} ..."
  PLACE=(--subnet-id "$SUBNET_ID" --security-group-ids "$SG_ID")
  [ "$SUBNET_ID" = AUTO ] && PLACE=(--security-group-ids "$SG_ID")
  IID=$("${AWS[@]}" ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" "${PLACE[@]}" \
    --associate-public-ip-address --count 1 \
    --iam-instance-profile "Name=de-aiml-s3" \
    --instance-initiated-shutdown-behavior terminate \
    --user-data "$USERDATA" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=de-aiml-scaleop-1b-reseed},{Key=project,Value=scaleop}]' \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --query 'Instances[0].InstanceId' --output text 2>/tmp/ri_err) && break
  echo "[aws] $INSTANCE_TYPE/$SUBNET_ID unavailable: $(tr -d '\n' < /tmp/ri_err | tail -c 130)"; IID=""
done
[ -n "$IID" ] && break
done
[ -n "$IID" ] || { echo "run-instances FAILED (no capacity)"; exit 9; }
echo "[aws] launched $IID — will TERMINATE on exit (after pulling artifacts)"
# Results are pulled in EVERY exit path (normal, idle-failure, Ctrl-C) BEFORE terminating.
pull_artifacts(){
  [ -n "${IP:-}" ] || return 0
  echo "[aws] pulling artifacts before terminate..."
  ssh "${SSH_OPTS[@]}" ubuntu@"$IP" "cd scaleop && aws s3 sync data/m5/ s3://$BUCKET/conv_reseed/checkpoints/ --only-show-errors && echo ckpts-archived" 2>/dev/null || true
  mkdir -p project_docs/results_aws aws_logs
  rsync -az -e "ssh ${SSH_OPTS[*]}" ubuntu@"$IP":scaleop/project_docs/results/ project_docs/results_aws/ 2>/dev/null || true
  rsync -az -e "ssh ${SSH_OPTS[*]}" ubuntu@"$IP":scaleop/aws_logs/ aws_logs/ 2>/dev/null || true
  ssh "${SSH_OPTS[@]}" ubuntu@"$IP" 'cat scaleop/aws_conv_console.log' > aws_conv_console.log 2>/dev/null || true
  echo "[aws] artifacts pulled: project_docs/results_aws/ aws_logs/ s3://$BUCKET/conv_reseed/"
}
cleanup(){ pull_artifacts; echo "[aws] terminating $IID"; "${AWS[@]}" ec2 terminate-instances --instance-ids "$IID" >/dev/null && echo "[aws] TERMINATED"; }
trap cleanup EXIT INT TERM

"${AWS[@]}" ec2 wait instance-running --instance-ids "$IID"
IP=$("${AWS[@]}" ec2 describe-instances --instance-ids "$IID" \
      --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "[aws] $IID at $IP; waiting for sshd..."
for i in $(seq 1 60); do ssh "${SSH_OPTS[@]}" ubuntu@"$IP" 'echo up' >/dev/null 2>&1 && break; sleep 15; done
ssh "${SSH_OPTS[@]}" ubuntu@"$IP" 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -2' || { echo "ssh failed"; exit 1; }

# Ship MASTER (Paper-1 code): the conv reseed needs none of the Paper-2 flags.
STAGE=$(mktemp -d)
git archive master | tar -x -C "$STAGE" || { echo "git archive master failed"; exit 1; }
echo "[aws] shipping master ($(git rev-parse --short master))"
rsync -az -e "ssh ${SSH_OPTS[*]}" --exclude data --exclude '*.pyc' \
      --exclude paper --exclude project_docs/results "$STAGE"/ ubuntu@"$IP":scaleop/
rm -rf "$STAGE"
if [ -s data/eval_tokens.pt ]; then
  ssh "${SSH_OPTS[@]}" ubuntu@"$IP" 'mkdir -p scaleop/data'
  rsync -az -e "ssh ${SSH_OPTS[*]}" data/eval_tokens.pt ubuntu@"$IP":scaleop/data/
else
  echo "[aws] WARNING: data/eval_tokens.pt missing locally"
fi
ssh "${SSH_OPTS[@]}" ubuntu@"$IP" 'command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh'

# 4 parallel single-GPU jobs, detached; logs sync to S3 every 3 min while the box lives.
ssh "${SSH_OPTS[@]}" ubuntu@"$IP" "cd scaleop && export PATH=\$HOME/.local/bin:\$PATH && \
  mkdir -p aws_logs && rm -f aws_logs/TRAIN_DONE && setsid nohup bash -c '
  ( while true; do aws s3 sync aws_logs/ s3://$BUCKET/conv_reseed/logs/ --only-show-errors 2>/dev/null; \
      aws s3 sync project_docs/results/ s3://$BUCKET/conv_reseed/results/ --only-show-errors 2>/dev/null; sleep 180; done ) & \
  ( while true; do sleep 900; aws s3 sync data/m5/ s3://$BUCKET/conv_reseed/checkpoints/ --only-show-errors 2>/dev/null; done ) & \
  CUDA_VISIBLE_DEVICES=0 uv run python experiments/m5_train.py --config configs/m5_conv.yaml --init hybrid_rs   --seed 1 --tag _1b_s1 --resume > aws_logs/conv_hyb_s1.log 2>&1 & \
  CUDA_VISIBLE_DEVICES=1 uv run python experiments/m5_train.py --config configs/m5_conv.yaml --init subclone_rs --seed 1 --tag _1b_s1 --resume > aws_logs/conv_sub_s1.log 2>&1 & \
  CUDA_VISIBLE_DEVICES=2 uv run python experiments/m5_train.py --config configs/m5_conv.yaml --init hybrid_rs   --seed 2 --tag _1b_s2 --resume > aws_logs/conv_hyb_s2.log 2>&1 & \
  CUDA_VISIBLE_DEVICES=3 uv run python experiments/m5_train.py --config configs/m5_conv.yaml --init subclone_rs --seed 2 --tag _1b_s2 --resume > aws_logs/conv_sub_s2.log 2>&1 & \
  wait; touch aws_logs/TRAIN_DONE; \
  aws s3 sync aws_logs/ s3://$BUCKET/conv_reseed/logs/ --only-show-errors; \
  aws s3 sync project_docs/results/ s3://$BUCKET/conv_reseed/results/ --only-show-errors' > aws_conv_console.log 2>&1 < /dev/null & echo started"
echo "[aws] detached; polling every 5 min (max ${MAX_HOURS}h)"
IDLE_N=0
for k in $(seq 1 $((MAX_HOURS*12))); do
  sleep 300
  st=$(ssh "${SSH_OPTS[@]}" ubuntu@"$IP" 'ls scaleop/aws_logs/TRAIN_DONE >/dev/null 2>&1 && echo DONE || pgrep -f "[m]5_train.py" >/dev/null && echo RUNNING || echo IDLE' 2>/dev/null)
  case "$st" in
    DONE)  echo "[aws] training complete: ALL DONE"; break;;
    IDLE)  IDLE_N=$((IDLE_N+1))
           echo "[aws] idle poll $IDLE_N/3"
           if [ "$IDLE_N" -ge 3 ]; then
             echo "[aws] idle 3 polls — treating as failure"
             ssh "${SSH_OPTS[@]}" ubuntu@"$IP" 'tail -5 scaleop/aws_logs/*.log 2>/dev/null | tail -30'
             break
           fi;;
    *)     IDLE_N=0; [ $((k % 6)) -eq 0 ] && ssh "${SSH_OPTS[@]}" ubuntu@"$IP" \
             'grep -h "t=" scaleop/aws_logs/conv_*.log 2>/dev/null | tail -4' || true;;
  esac
done

# Normal-path pull (cleanup pulls again on exit; both are idempotent).
pull_artifacts
echo "[aws] done (instance terminates now)"
