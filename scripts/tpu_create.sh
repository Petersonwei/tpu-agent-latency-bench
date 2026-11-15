#!/usr/bin/env bash
# Create, inspect or delete the Cloud TPU v6e-1 VM. Run from your laptop with gcloud authenticated.
#   MODE=flex   bash scripts/tpu_create.sh         # DEFAULT: flex-start, ~half price, auto-deleted after MAX_RUN
#   MODE=ondemand bash scripts/tpu_create.sh       # on-demand (fallback if flex-start never provisions)
#   MODE=spot   bash scripts/tpu_create.sh         # spot (can be preempted mid-sweep; not recommended)
#   bash scripts/tpu_create.sh status | delete
#   DRY_RUN=1 ... prints the gcloud command without running it.
set -euo pipefail
PROJECT=${PROJECT:-$(gcloud config get-value project 2>/dev/null)}
ZONE=${ZONE:-us-east5-a}             # flex-start v6e zones (docs, Nov 2026): us-east5-a, asia-northeast1-b
NAME=${NAME:-agent-bench-v6e}
QR=${QR:-$NAME-qr}
ACCEL=${ACCEL:-v6e-1}
RUNTIME=${RUNTIME:-v2-alpha-tpuv6e}
MODE=${MODE:-flex}
MAX_RUN=${MAX_RUN:-8h}               # flex-start hard cap: VM is deleted after this -> cost ceiling
VALID_FOR=${VALID_FOR:-6h}           # give up if capacity isn't found within this window
run(){ echo "+ $*"; [[ "${DRY_RUN:-0}" == "1" ]] || "$@"; }

case "${1:-create}" in
  status)
    gcloud alpha compute tpus queued-resources list --zone "$ZONE" --project "$PROJECT" 2>/dev/null || true
    gcloud compute tpus tpu-vm list --zone "$ZONE" --project "$PROJECT"
    exit 0;;
  delete)
    run gcloud compute tpus tpu-vm delete "$NAME" --zone "$ZONE" --project "$PROJECT" --quiet || true
    run gcloud alpha compute tpus queued-resources delete "$QR" --zone "$ZONE" --project "$PROJECT" --quiet --force || true
    echo "Check nothing is left:"; gcloud compute tpus tpu-vm list --zone "$ZONE" --project "$PROJECT"
    exit 0;;
  create) ;;
  *) echo "usage: $0 [create|status|delete]"; exit 2;;
esac

# Money gate: no VM without a budget alert on this project (see scripts/budget_guard.sh).
# Fails closed: a gcloud/network error also blocks creation.
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  bash "$(dirname "$0")/budget_guard.sh" check || {
    echo "REFUSING to create a TPU VM: no budget guard on $PROJECT."
    echo "Fix: AMOUNT=<amount in billing currency> bash scripts/budget_guard.sh create   (needs the account owner's OK)"
    exit 3; }
fi

COMMON=(--node-id "$NAME" --project "$PROJECT" --zone "$ZONE" --accelerator-type "$ACCEL" --runtime-version "$RUNTIME")
case "$MODE" in
  flex) run gcloud alpha compute tpus queued-resources create "$QR" "${COMMON[@]}" \
          --provisioning-model=flex-start --max-run-duration="$MAX_RUN" --valid-until-duration="$VALID_FOR" ;;
  spot) run gcloud alpha compute tpus queued-resources create "$QR" "${COMMON[@]}" --spot \
          --valid-until-duration="$VALID_FOR" ;;
  ondemand) run gcloud compute tpus tpu-vm create "$NAME" --project "$PROJECT" --zone "$ZONE" \
          --accelerator-type "$ACCEL" --version "$RUNTIME" ;;
  *) echo "MODE must be flex|spot|ondemand"; exit 2;;
esac
echo
echo "Watch:  bash scripts/tpu_create.sh status        (queued resource goes WAITING_FOR_RESOURCES -> ACTIVE)"
echo "SSH:    gcloud compute tpus tpu-vm ssh $NAME --zone $ZONE --project $PROJECT"
echo "Delete: bash scripts/tpu_create.sh delete         (billing runs while the VM exists)"
