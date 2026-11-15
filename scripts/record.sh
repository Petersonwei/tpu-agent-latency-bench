#!/usr/bin/env bash
# Record a VM step for the video WITHOUT screen recording: asciinema captures the terminal on the
# laptop while the command runs on the TPU VM over ssh. The .cast lands on the laptop at once, so it
# survives the flex-start auto-delete. Only terminal text is captured (no desktop, no notifications).
#   bash scripts/record.sh 01_setup  'cd agent-latency-bench && bash scripts/setup_vm.sh'
#   bash scripts/record.sh 02_smoke_cold 'cd agent-latency-bench && source ~/venv-vllm/bin/activate && bash scripts/smoke_tpu.sh'
#   bash scripts/record.sh 03_smoke_warm '<same as 02>'
#   bash scripts/render_casts.sh          # .cast -> .gif + .mp4 for editing
# LOCAL=1 runs the command on the laptop instead of the VM (rehearsal).
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?usage: record.sh NAME 'command'}; shift
CMD=${1:?usage: record.sh NAME 'command'}
OUT=media/casts/$NAME.cast
mkdir -p media/casts
command -v asciinema >/dev/null || { echo "install first: brew install asciinema agg"; exit 2; }
if [[ "${LOCAL:-0}" == "1" ]]; then RUN="bash -c $(printf %q "$CMD")"
else RUN="bash scripts/vm_sync.sh ssh $(printf %q "$CMD")"; fi
# 110x32 keeps text readable in a 1080p video; idle limit trims long silent waits (TPU compile).
asciinema rec --headless --overwrite --window-size 110x32 --idle-time-limit 3 \
  --title "$NAME" -c "$RUN" "$OUT"
# the recorded command's exit code is the last "x" event; propagate it so gates still gate
code=$(tail -1 "$OUT" | python3 -c 'import json,sys; e=json.loads(sys.stdin.read()); print(e[2] if e[1]=="x" else 0)')
echo "recorded -> $OUT (exit $code)"
exit "$code"
