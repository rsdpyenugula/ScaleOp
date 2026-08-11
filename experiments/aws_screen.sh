#!/usr/bin/env bash
# 12B->6.9B stability screen on AWS (Paper 2, largest scale: 16384^2 compensation solve).
# Profile-parameterized so anyone with access runs it with their OWN aws profile; never
# handles raw credentials (uses the named profile's session). Auto-terminates on exit.
#
#   ./aws_screen.sh [--profile prod_admin] [--spot] [--dry-run]
#
# Runs 9 arms as 8 PARALLEL single-GPU jobs (Adafactor puts a 6.9B target in one 80GB A100,
# so no FSDP is needed). Checkpoints (~28GB each) are archived to S3; results/curves/logs are
# pulled back to this machine.
set -uo pipefail

PROFILE="${AWS_PROFILE:-prod_admin}"
REGION="us-east-1"
INSTANCE_TYPE="p4de.24xlarge"        # default; --types overrides
TYPES="${TYPES:-}"                   # e.g. "p4de.24xlarge p5.48xlarge" (tried in order)
KEY_NAME="de-aiml"
SUBNETS=""                           # resolved per region below
                                               # an AZ is short); then 1c, then 1b explicitly
SG_ID=""                             # resolved per region below
BUCKET="de-aiml-scaleop-662022802750"
SPOT=0; DRYRUN=0
AMI_NAME='Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*'

while [ $# -gt 0 ]; do case "$1" in
  --profile) PROFILE="$2"; shift 2;;
  --region)  REGION="$2";  shift 2;;
  --type)    INSTANCE_TYPE="$2"; shift 2;;
  --types)   TYPES="$2"; shift 2;;
  --bucket)  BUCKET="$2"; shift 2;;
  --spot)    SPOT=1; shift;;
  --dry-run) DRYRUN=1; shift;;
  *) echo "unknown arg $1"; exit 1;;
esac; done

AWS=(aws --profile "$PROFILE" --region "$REGION")
PEM="$HOME/.ssh/${KEY_NAME}.pem"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -i "$PEM")

echo "[aws] profile=$PROFILE region=$REGION spot=$SPOT bucket=$BUCKET"
echo "[aws] type search: ${TYPES:-$INSTANCE_TYPE}"
"${AWS[@]}" sts get-caller-identity >/dev/null || { echo "not authenticated: aws sso login --profile $PROFILE"; exit 1; }
[ -f "$PEM" ] || { echo "missing private key $PEM"; exit 1; }

# Network ids differ per region; resolve them (and create the SG if missing) so the
# same script works in us-east-1, us-west-2, ... without editing hardcoded ids.
if [ -z "$SUBNETS" ]; then
  SUBNETS="AUTO $("${AWS[@]}" ec2 describe-subnets --filters Name=default-for-az,Values=true \
    --query 'sort_by(Subnets,&AvailabilityZone)[].SubnetId' --output text | tr '\t' ' ')"
fi
if [ -z "$SG_ID" ]; then
  SG_ID=$("${AWS[@]}" ec2 describe-security-groups --filters Name=group-name,Values=de-aiml-ssh \
          --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null)
  if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
    VPC=$("${AWS[@]}" ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
    SG_ID=$("${AWS[@]}" ec2 create-security-group --group-name de-aiml-ssh \
             --description "SSH for ScaleOp 12B to 6.9B screen" --vpc-id "$VPC" --query GroupId --output text)
    MYIP=$(curl -s https://checkip.amazonaws.com)
    "${AWS[@]}" ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 \
      --cidr "${MYIP}/32" >/dev/null 2>&1
    echo "[aws] created SG $SG_ID in $REGION (SSH from ${MYIP}/32)"
  fi
fi
echo "[aws] SG=$SG_ID subnets=$(echo $SUBNETS | wc -w | tr -d ' ')"

AMI=$("${AWS[@]}" ec2 describe-images --owners amazon \
  --filters "Name=name,Values=$AMI_NAME" "Name=state,Values=available" \
  --query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' --output text)
echo "[aws] AMI=$AMI"
[ "$DRYRUN" = 1 ] && { echo "[dry-run] would launch $INSTANCE_TYPE from $AMI; stopping."; exit 0; }

MARKET=(); [ "$SPOT" = 1 ] && MARKET=(--instance-market-options '{"MarketType":"spot"}')
# Independent dead-man switch: even if this script (or the whole session) dies, the
# instance halts itself after MAX_HOURS and shutdown terminates it. Never bills unattended.
MAX_HOURS="${MAX_HOURS:-14}"
USERDATA=$(printf '#!/bin/bash\nsetsid nohup bash -c "sleep %d; /sbin/shutdown -h now" >/dev/null 2>&1 &\n' $((MAX_HOURS*3600)) | base64)
# p4de capacity is AZ-dependent and fluctuates; try each subnet until one succeeds.
IID=""
[ -n "$TYPES" ] || TYPES="$INSTANCE_TYPE"
for INSTANCE_TYPE in $TYPES; do
for SUBNET_ID in $SUBNETS; do
  echo "[aws] trying $INSTANCE_TYPE in ${SUBNET_ID} ..."
  PLACE=(--subnet-id "$SUBNET_ID" --security-group-ids "$SG_ID")
  [ "$SUBNET_ID" = AUTO ] && PLACE=(--security-group-ids "$SG_ID")   # no AZ pin: AWS chooses
  IID=$("${AWS[@]}" ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" "${PLACE[@]}" \
    --associate-public-ip-address ${MARKET[@]+"${MARKET[@]}"} --count 1 \
    --iam-instance-profile "Name=de-aiml-s3" \
    --instance-initiated-shutdown-behavior terminate \
    --user-data "$USERDATA" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=de-aiml-scaleop-12b69b},{Key=project,Value=scaleop}]' \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":1000,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --query 'Instances[0].InstanceId' --output text 2>/tmp/ri_err) && break
  echo "[aws] $INSTANCE_TYPE/$SUBNET_ID unavailable: $(tr -d '\n' < /tmp/ri_err | tail -c 130)"; IID=""
done
[ -n "$IID" ] && break
done
[ -n "$IID" ] || { echo "run-instances FAILED in all AZs (no p4de capacity right now)"; exit 1; }
echo "[aws] launched $IID — will TERMINATE on exit"
cleanup(){ echo "[aws] terminating $IID"; "${AWS[@]}" ec2 terminate-instances --instance-ids "$IID" >/dev/null && echo "[aws] TERMINATED"; }
trap cleanup EXIT INT TERM

"${AWS[@]}" ec2 wait instance-running --instance-ids "$IID"
IP=$("${AWS[@]}" ec2 describe-instances --instance-ids "$IID" \
      --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "[aws] $IID at $IP; waiting for sshd..."
for i in $(seq 1 60); do ssh "${SSH_OPTS[@]}" ubuntu@"$IP" 'echo up' >/dev/null 2>&1 && break; sleep 15; done
ssh "${SSH_OPTS[@]}" ubuntu@"$IP" 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -2' || { echo "ssh failed"; exit 1; }

# --- sync code (no data/, no git) and install uv ---
rsync -az -e "ssh ${SSH_OPTS[*]}" --exclude data --exclude .git --exclude '*.pyc' \
      --exclude paper --exclude paper2 --exclude project_docs/results ./ ubuntu@"$IP":scaleop/
ssh "${SSH_OPTS[@]}" ubuntu@"$IP" 'command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh'

# --- run the 9-arm screen (8 in parallel) ---
ssh "${SSH_OPTS[@]}" ubuntu@"$IP" 'cd scaleop && export PATH=$HOME/.local/bin:$PATH && bash experiments/aws_train.sh' 2>&1 | tee aws_screen_run.log

# --- archive checkpoints to S3, pull results/logs locally ---
echo "[aws] archiving checkpoints to s3://$BUCKET/12b69b/ ..."
ssh "${SSH_OPTS[@]}" ubuntu@"$IP" "cd scaleop && aws s3 sync data/m5/ s3://$BUCKET/12b69b/checkpoints/ --exclude '*' --include '*b69*' --only-show-errors && echo uploaded"
mkdir -p project_docs/results_aws aws_logs
rsync -az -e "ssh ${SSH_OPTS[*]}" ubuntu@"$IP":scaleop/project_docs/results/ project_docs/results_aws/ 2>/dev/null || true
rsync -az -e "ssh ${SSH_OPTS[*]}" ubuntu@"$IP":scaleop/aws_logs/ aws_logs/ 2>/dev/null || true
echo "[aws] results -> project_docs/results_aws/ ; logs -> aws_logs/ ; checkpoints -> s3://$BUCKET/12b69b/"
echo "[aws] done (instance terminates now)"
