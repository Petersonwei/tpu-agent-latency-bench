#!/usr/bin/env bash
# Start vLLM-TorchTPU with serve_config.env, wait for health, run the FIXED benchmark.
# READ-ONLY for the autoserve loop. Prints the bench summary; `score:` is the metric.
set -euo pipefail
cd "$(dirname "$0")/.."
source serve_config.env

PORT=${PORT:-8000}
MIN_SUCCESS=${MIN_SUCCESS:-0.80}       # fixed success floor: score=inf below this
TOOL_LATENCY=${TOOL_LATENCY:-0.5}      # fixed for the search; swept separately in the study
CONTEXT_TOKENS=${CONTEXT_TOKENS:-4000} # shared agent context per session (prefix-cache realism)
THINKING=${THINKING:-off}
export VLLM_XLA_CACHE_PATH=${VLLM_XLA_CACHE_PATH:-$HOME/.cache/vllm_xla}

pkill -f "vllm serve" 2>/dev/null || true; sleep 3

ARGS=(--port "$PORT" --tensor-parallel-size 1
      --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
      --kv-cache-dtype "$KV_CACHE_DTYPE"
      --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")
[[ -n "${REASONING_PARSER:-}" ]] && ARGS+=(--reasoning-parser "$REASONING_PARSER")
[[ "$ENABLE_PREFIX_CACHING" == "1" ]] && ARGS+=(--enable-prefix-caching) || ARGS+=(--no-enable-prefix-caching)
[[ -n "${SPECULATIVE_CONFIG:-}" ]] && ARGS+=(--speculative-config "$SPECULATIVE_CONFIG")
# shellcheck disable=SC2206
ARGS+=($EXTRA_ARGS)

mkdir -p results/logs results/runs
echo "serve: vllm serve $MODEL ${ARGS[*]}"
T_START=$(date +%s)
vllm serve "$MODEL" "${ARGS[@]}" > results/logs/server.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for i in $(seq 1 180); do            # up to 15 min for first-time TPU graph compilation
  curl -sf "http://localhost:$PORT/health" >/dev/null && break
  kill -0 $SERVER_PID 2>/dev/null || { echo "server died"; tail -40 results/logs/server.log; exit 1; }
  sleep 5
done
curl -sf "http://localhost:$PORT/health" >/dev/null || { echo "server not healthy"; exit 1; }
echo "server_ready_s: $(( $(date +%s) - T_START ))"

STAMP=$(date +%Y%m%d_%H%M%S)
python harness/bench.py --base-url "http://localhost:$PORT/v1" --model "$MODEL" \
  --workload all --concurrency 8 --repeats 3 --warmup 4 --thinking "$THINKING" \
  --tool-latency "$TOOL_LATENCY" --context-tokens "$CONTEXT_TOKENS" --min-success "$MIN_SUCCESS" \
  --out "results/runs/${STAMP}_$(git rev-parse --short HEAD 2>/dev/null || echo nogit).jsonl"
