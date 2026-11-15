#!/usr/bin/env bash
# Move code to the TPU VM and results back, without GitHub (repo can stay private until launch).
#   bash scripts/vm_sync.sh push            # git bundle of HEAD -> VM:~/agent-latency-bench
#   bash scripts/vm_sync.sh pull            # VM results/ -> ./results/tpu_<date>/
#   bash scripts/vm_sync.sh ssh '<cmd>'     # run a command on the VM
#   bash scripts/vm_sync.sh watch           # pull every 10 min until Ctrl-C (protects against the flex-start cap)
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT=${PROJECT:-$(gcloud config get-value project 2>/dev/null)}
ZONE=${ZONE:-us-east5-a}
NAME=${NAME:-agent-bench-v6e}
G=(--zone "$ZONE" --project "$PROJECT")
DEST=${DEST:-results/tpu_$(date +%Y%m%d)}

case "${1:-}" in
  push)
    git bundle create /tmp/alb.bundle HEAD >/dev/null
    gcloud compute tpus tpu-vm scp /tmp/alb.bundle "$NAME:~/alb.bundle" "${G[@]}"
    gcloud compute tpus tpu-vm ssh "$NAME" "${G[@]}" --command \
      'if [ -d agent-latency-bench ]; then cd agent-latency-bench && git pull -q ~/alb.bundle HEAD; else git clone -q ~/alb.bundle agent-latency-bench; fi && cd agent-latency-bench && git log --oneline -1'
    ;;
  pull)
    mkdir -p "$DEST"
    gcloud compute tpus tpu-vm ssh "$NAME" "${G[@]}" --command \
      'cd agent-latency-bench && tar czf /tmp/results.tgz results 2>/dev/null && ls -la /tmp/results.tgz'
    gcloud compute tpus tpu-vm scp "$NAME:/tmp/results.tgz" "$DEST/results.tgz" "${G[@]}"
    tar xzf "$DEST/results.tgz" -C "$DEST" && echo "pulled -> $DEST/results"
    ;;
  ssh)
    shift; gcloud compute tpus tpu-vm ssh "$NAME" "${G[@]}" --command "$*"
    ;;
  watch)
    while true; do "$0" pull || echo "pull failed (VM gone?)"; sleep 600; done
    ;;
  *) echo "usage: $0 push|pull|ssh '<cmd>'|watch"; exit 2;;
esac
