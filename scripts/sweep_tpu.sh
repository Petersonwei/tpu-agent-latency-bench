#!/usr/bin/env bash
# The main study (E1–E3, E5): for each SERVER config, start vLLM once, then sweep CLIENT
# variables (tool latency, concurrency, thinking, context). Writes one JSONL per cell plus
# results/sweep.tsv. Run on the TPU VM inside tmux; ~2–4 h on v6e-1.
#   SIM=1 bash scripts/sweep_tpu.sh   -> zero-cost rehearsal: same script, simulator instead of vLLM
set -euo pipefail
cd "$(dirname "$0")/.."
source serve_config.env
PORT=${PORT:-8000}
PY=${PY:-python}
SIM=${SIM:-0}
SIM_SCALE=${SIM_SCALE:-10}   # SIM only: everything runs SIM_SCALE x faster; all ratios are preserved
OUTDIR=${OUTDIR:-results}
export VLLM_XLA_CACHE_PATH=${VLLM_XLA_CACHE_PATH:-$HOME/.cache/vllm_xla}
mkdir -p "$OUTDIR/sweep" "$OUTDIR/logs"
TSV=$OUTDIR/sweep.tsv
[[ -f $TSV ]] || printf "server_cfg\tthinking\ttool_latency\tconcurrency\tcontext\ttask_p50_s\ttask_p95_s\tttft_p50_s\tttfa_p50_s\tfirst_visible_p50_s\tsuccess_rate\tllm_share\ttool_share\ttok_s\ttool_dist\n" > $TSV

# name|prefix_caching|speculative_config|sim_decode_tps|sim_prefill_tps   (sim_* used only when SIM=1)
SERVER_CFGS=(
  "base|1||60|8000"
  "noprefix|0||60|2500"
  "mtp1|1|{\"method\": \"mtp\", \"num_speculative_tokens\": 1}|85|8000"
  "mtp3|1|{\"method\": \"mtp\", \"num_speculative_tokens\": 3}|110|8000"
)
if [[ "$SIM" == "1" ]]; then DEF_TL=$(python3 -c "print(' '.join(str(x/$SIM_SCALE) for x in (0,0.5,2)))"); else DEF_TL="0 0.5 2"; fi
TOOL_LATS=(${TOOL_LATS:-$DEF_TL})
CONCS=(${CONCS:-1 8})
THINKS=(${THINKS:-off on})
CONTEXTS=(${CONTEXTS:-4000})
# H5 (prefix caching vs shared-context length): a few extra long-context cells, only for the
# configs that differ in prefix caching, thinking off, middle tool latency, every concurrency.
LONG_CTX=${LONG_CTX:-16000}
LONG_CFGS=" ${LONG_CFGS:-base noprefix} "
LONG_TL=${LONG_TL:-${TOOL_LATS[$(( ${#TOOL_LATS[@]} / 2 ))]}}
# Long-tail tool latency (TraceLab: real tool latency is heavy-tailed): same mean as LONG_TL but
# lognormal, for the configs whose speedup we compare, thinking off, 1 session.
TAIL_CFGS=" ${TAIL_CFGS:-base mtp3 ngram3} "
TAIL_DIST=${TAIL_DIST:-lognormal}
# If an MTP config cannot start (MTP on v6e is not in the vllm-torchtpu CI matrix), run one n-gram
# speculative config instead (n-gram passes the vLLM TPU feature matrix) and log the failure.
NGRAM_CFG='ngram3|1|{"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 4}|80|8000'
FAILED_LOG=$OUTDIR/logs/failed_configs.txt
ngram_added=0

run_cell(){  # NAME TH TL C CTX [DIST]
  local NAME=$1 TH=$2 TL=$3 C=$4 CTX=$5 DIST=${6:-fixed}
  local TAG="${NAME}_th${TH}_tl${TL}_c${C}_ctx${CTX}"
  [[ "$DIST" != "fixed" ]] && TAG="${TAG}_d${DIST}"
  if [[ "$SIM" == "1" ]]; then
    local THINK_TOK=0; [[ "$TH" == "on" ]] && THINK_TOK=150
    SRC=(--sim --sim-tps $((SIMTPS*SIM_SCALE)) --sim-prefill-tps $((SIMPF*SIM_SCALE))
         --sim-ttft "$(python3 -c "print(0.15/$SIM_SCALE)")" --sim-think-tokens "$THINK_TOK" --repeats 1 --warmup 0)
  else
    SRC=(--base-url "http://localhost:$PORT/v1" --model "$MODEL" --repeats 3 --warmup 4)
  fi
  OUT=$($PY harness/bench.py "${SRC[@]}" --workload all --concurrency "$C" --thinking "$TH" \
        --tool-latency "$TL" --tool-dist "$DIST" --context-tokens "$CTX" --out "$OUTDIR/sweep/$TAG.jsonl" \
        2>>"$OUTDIR/logs/bench_errors.log") || true
  g(){ echo "$OUT" | awk -F': ' -v k="$1" '$1==k{print $2}'; }
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$NAME" "$TH" "$TL" "$C" "$CTX" \
    "$(g task_p50_s)" "$(g task_p95_s)" "$(g ttft_p50_s)" "$(g ttfa_p50_s)" "$(g first_visible_p50_s)" \
    "$(g success_rate)" "$(g llm_share)" "$(g tool_share)" "$(g output_tok_per_s)" "$DIST" | tee -a $TSV
}

i=0
while (( i < ${#SERVER_CFGS[@]} )); do
  cfg=${SERVER_CFGS[$i]}; i=$((i+1))
  IFS='|' read -r NAME PC SPEC SIMTPS SIMPF <<<"$cfg"
  if [[ "$SIM" == "1" ]]; then PID=""; else
  pkill -f "vllm serve" 2>/dev/null || true; sleep 3
  ARGS=(--port "$PORT" --tensor-parallel-size 1 --max-model-len "$MAX_MODEL_LEN"
        --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" --reasoning-parser "$REASONING_PARSER")
  [[ "$PC" == "1" ]] && ARGS+=(--enable-prefix-caching) || ARGS+=(--no-enable-prefix-caching)
  [[ -n "$SPEC" ]] && ARGS+=(--speculative-config "$SPEC")
  echo "=== server $NAME: ${ARGS[*]}"
  vllm serve "$MODEL" "${ARGS[@]}" > "$OUTDIR/logs/server_$NAME.log" 2>&1 &
  PID=$!
  for _w in $(seq 1 180); do curl -sf "localhost:$PORT/health" >/dev/null && break; sleep 5; done
  if ! curl -sf "localhost:$PORT/health" >/dev/null; then
    echo "server $NAME failed"; tail -30 "$OUTDIR/logs/server_$NAME.log"; kill $PID 2>/dev/null || true
    printf "%s\t%s\n" "$NAME" "$(grep -m1 -iE 'error|exception' "$OUTDIR/logs/server_$NAME.log" | cut -c1-200)" >> "$FAILED_LOG"
    if [[ "$SPEC" == *mtp* && $ngram_added == 0 ]]; then SERVER_CFGS+=("$NGRAM_CFG"); ngram_added=1; fi
    continue
  fi
  fi

  for TH in "${THINKS[@]}"; do for TL in "${TOOL_LATS[@]}"; do for C in "${CONCS[@]}"; do for CTX in "${CONTEXTS[@]}"; do
    run_cell "$NAME" "$TH" "$TL" "$C" "$CTX"
  done; done; done; done
  if [[ "$LONG_CFGS" == *" $NAME "* && -n "$LONG_CTX" ]]; then
    for C in "${CONCS[@]}"; do run_cell "$NAME" off "$LONG_TL" "$C" "$LONG_CTX"; done
  fi
  if [[ "$TAIL_CFGS" == *" $NAME "* && -n "$TAIL_DIST" ]]; then
    run_cell "$NAME" off "$LONG_TL" 1 "${CONTEXTS[0]}" "$TAIL_DIST"
  fi
  if [[ -n "$PID" ]]; then kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true; fi
done
