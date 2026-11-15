# program.md — autonomous serving-config search ("autoserve")

Adapted from karpathy/autoresearch and windmaple/autofinetune (Tunix Academy L200,
"Autonomous Finetuning"). Start your coding agent in this folder on the TPU VM and say:
`follow @program.md`.

## Goal

Find the vLLM-TorchTPU serving configuration that gives the **lowest median end-to-end
agent task time** (`score`, seconds, lower is better) on the fixed benchmark, without lowering
task success.

Research twist (this is the point of the project): every row also logs `ttft_p50_s`.
Afterwards we ask: *did the configs that improved TTFT also improve task time, and by how much?*
So log faithfully even when a run is discarded.

## Files

| File | Agent may edit? | Purpose |
|---|---|---|
| `serve_config.env` | **YES — the only file you edit** | vLLM serve flags |
| `harness/*` | NO | benchmark, workloads, tools, scorer |
| `scripts/run_experiment.sh` | NO | starts the server, runs the fixed benchmark |
| `results/results.tsv` | append only, never commit | experiment log |

## CAN change (in `serve_config.env` only)
- `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, `MAX_MODEL_LEN` (≥ 16384)
- `GPU_MEMORY_UTILIZATION`, `ENABLE_PREFIX_CACHING`, `KV_CACHE_DTYPE` (auto / fp8)
- `SPECULATIVE_CONFIG`: MTP (`{"method": "mtp", "num_speculative_tokens": N}`, N=1–5) or n-gram
- Any other `vllm serve` flag in `EXTRA_ARGS` (e.g. chunked prefill settings)

## CANNOT change
- `MODEL`, `TOOL_PARSER`, `REASONING_PARSER` (fixed for the whole search), weight quantisation
- Thinking mode, context size and tool latency are fixed by `run_experiment.sh` (THINKING=off,
  CONTEXT_TOKENS=4000, TOOL_LATENCY=0.5)
- The benchmark arguments, workloads, tool latency, the scorer, the success floor
- Do not install packages. Do not edit anything outside `serve_config.env`.

## Setup (once)
1. `git checkout -b autoserve/$(date +%Y%m%d)-$(git rev-parse --short HEAD)`
2. Read `README.md`, `harness/bench.py`, `scripts/run_experiment.sh`, `serve_config.env`.
3. Create `results/results.tsv` with this header (tab-separated):
   `commit	score	ttft_p50_s	first_visible_p50_s	task_p95_s	success_rate	tok_s	status	description`
4. Run the unmodified config once as the **baseline**: `bash scripts/run_experiment.sh > run.log 2>&1`

## LOOP FOREVER
1. Look at the git state and the last rows of `results/results.tsv`.
2. Change `serve_config.env` with **one** idea. Write down why you expect it to help.
3. `git commit -am "<idea>"`
4. `bash scripts/run_experiment.sh > run.log 2>&1`  (kill it if it runs > 25 min → status `crash`)
5. `grep -E "^(score|ttft_p50_s|first_visible_p50_s|task_p95_s|success_rate|output_tok_per_s|server_ready_s):" run.log`
   If empty, `tail -50 run.log` to diagnose. Server start failures are usually an invalid flag or out of
   HBM: try one quick fix, otherwise log `crash` and move on.
6. Append a row to `results/results.tsv` (do not commit it). Use `0` for missing numbers.
7. If `score` is lower than the best kept score by more than 2% → `keep` (the branch advances).
   Otherwise → `discard` and `git reset --hard HEAD~1`.
   A score of `inf` means success fell below the floor: always `discard`.

Never pause to ask whether to continue. If you run out of ideas, re-read the README's
"ideas" list, re-check `ttft_p50_s` vs `score` trends, or try combining two kept ideas.

## Notes for the human reviewer
- Read **every** row, not only the kept ones; the noise floor is visible in discarded rows.
- The `description` is the agent's own summary — check the diff for surprising rows.
- After the search, re-run baseline and best config 3× each to confirm the gain beyond noise.
