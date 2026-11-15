#!/usr/bin/env bash
# LAB step 2/3: cheapest possible check that serving + tool calling + thinking work on the TPU,
# before spending hours on the sweep. Exit code 0 = gate passed.
set -euo pipefail
cd "$(dirname "$0")/.."
export TOOL_LATENCY=0 CONTEXT_TOKENS=1000
# run_experiment.sh starts the server and runs the full fixed benchmark; for the smoke test we
# start it the same way but run only small workloads.
source serve_config.env
PORT=${PORT:-8000}
export VLLM_XLA_CACHE_PATH=${VLLM_XLA_CACHE_PATH:-$HOME/.cache/vllm_xla}
pkill -f "vllm serve" 2>/dev/null || true; sleep 3
mkdir -p results/logs results/smoke
T0=$(date +%s)
if [[ "${MOCK:-0}" == "1" ]]; then   # zero-cost rehearsal of this gate (laptop)
  python tests/mock_vllm_server.py --port "$PORT" > results/logs/server_smoke.log 2>&1 &
else
vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" --reasoning-parser "$REASONING_PARSER" \
  --enable-prefix-caching > results/logs/server_smoke.log 2>&1 &
fi
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
for i in $(seq 1 180); do curl -sf "localhost:$PORT/health" >/dev/null && break
  kill -0 $PID 2>/dev/null || { echo "SERVER DIED"; tail -40 results/logs/server_smoke.log; exit 1; }; sleep 5; done
echo "server_ready_s: $(( $(date +%s) - T0 ))" | tee results/smoke/server_ready.txt

fail=0
for TH in off on; do
  echo "=== tool workload, thinking=$TH"
  OUT=$(python harness/bench.py --base-url "http://localhost:$PORT/v1" --model "$MODEL" --workload tool \
        --repeats 1 --warmup 1 --concurrency 2 --thinking "$TH" --context-tokens "$CONTEXT_TOKENS" \
        --out "results/smoke/tool_th$TH.jsonl")
  echo "$OUT"
  succ=$(echo "$OUT" | awk -F': ' '$1=="success_rate"{print $2}')
  trunc=$(echo "$OUT" | awk -F': ' '$1=="truncated_calls"{print $2}')
  calls=$(python - "results/smoke/tool_th$TH.jsonl" <<'EOF'
import json,sys; print(sum(json.loads(l).get("tool_calls",0) for l in open(sys.argv[1]).read().splitlines()[1:]))
EOF
)
  echo "gate[$TH]: success=$succ tool_calls=$calls truncated=$trunc"
  python -c "import sys; sys.exit(0 if float('$succ')>=0.8 and int('$calls')>0 else 1)" || { echo "GATE FAILED (thinking=$TH)"; fail=1; }
done
[[ $fail == 0 ]] && echo "SMOKE GATE PASSED" || { echo "SMOKE GATE FAILED — see LAB.md fallbacks; first 5 answers:"; head -6 results/smoke/tool_thoff.jsonl | tail -5 | cut -c1-400; exit 1; }
