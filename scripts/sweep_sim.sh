#!/usr/bin/env bash
# Offline "Amdahl" sweep: how much does faster decode shrink agent task time as tool latency grows?
# Runs on a laptop with the simulator. Same sweep runs on TPU with scripts/sweep_tpu.sh (todo).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/sim
printf "sim_tps\ttool_latency\ttask_p50_s\tttft_p50_s\tllm_share\n"
for TL in 0 0.5 2; do
  for TPS in 20 60 180; do
    OUT=$(python3 harness/bench.py --sim --sim-tps "$TPS" --tool-latency "$TL" --workload tool \
          --repeats 1 --concurrency 6 --warmup 0 --out "results/sim/tps${TPS}_tl${TL}.jsonl")
    get(){ echo "$OUT" | awk -F': ' -v k="$1" '$1==k{print $2}'; }
    printf "%s\t%s\t%s\t%s\t%s\n" "$TPS" "$TL" "$(get task_p50_s)" "$(get ttft_p50_s)" "$(get llm_share)"
  done
done
