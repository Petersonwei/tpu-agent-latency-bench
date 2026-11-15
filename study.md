# study.md — the research loop (L3)

Point a coding agent here after a sweep: `follow @study.md`. No TPU needed; it plans the *next* TPU
session so that money is only spent on experiments that answer an open question.

## Goal
Turn `results/findings.json` into verdicts on the hypotheses below, and propose the smallest next
sweep that resolves the most important open one.

## Editable
`research/lab-notebook.md` (append-only entries) and `research/next_sweep.env` (variables for
`sweep_tpu.sh`: `TOOL_LATS`, `CONCS`, `THINKS`, `CONTEXTS`, and a proposed `SERVER_CFGS` list).

## Read-only
Everything else, especially `analysis/analyze.py`, `harness/`, `results/`.

## Hypotheses (update verdicts, never delete)
| ID | Hypothesis | Evidence key(s) | Verdict |
|---|---|---|---|
| H1 | Token-speed (chat benchmark) speedup **over-predicts** agent task speedup; the prefill + decode rule predicts it within its 95% CI at 1 session | `predict.*.within_ci`, `predict.*.decomposed_within_ci`, `predict.max_rel_error*.c1` | open |
| H1a | (accounting check, not a hypothesis) in-sample Amdahl error < 5% — if not, the harness timing is broken | `amdahl.max_relative_error` | open |
| H2 | At 2 s tool latency, even the fastest config gives < 1.3× task speedup | `amdahl.*.tl2.task_speedup` | open |
| H3 | MTP's agent gain shrinks at 8 sessions (AgentSpec; Liu et al. MLSys'26): the c1 rule breaks at c8 and MTP acceptance drops | `predict.mtp*.c8.*`, `predict.max_rel_error_decomposed.c8`, `mech.mtp*.c*.ctx*.spec_accept_rate` | open |
| H4 | Thinking delays the first visible token by > 2× and raises success on tool tasks | `thinking.*` | open |
| H6 | Heavy-tailed tool latency (lognormal, same mean) leaves the MTP agent speedup unchanged: its 95% CI overlaps the fixed-latency CI | `tail.mtp3.c1.tl0.5.task_speedup(_ci_*)` vs `tail.mtp3.c1.tl0.5.fixed_task_speedup(_ci_*)` | open |
| H5 | Prefix caching matters more as shared context grows (4k → 16k) — only if the cache actually hits on hybrid Qwen3.5 (vLLM "align" mode, ~528-token blocks) | `context.ctx*.c*.tl0.5.prefix_speedup(_ci_*)`, `mech.base.c*.ctx*.prefix_hit_rate` | open |

## Loop
1. Run `python analysis/analyze.py --results-dir results` (or the latest results dir).
2. For each hypothesis: write a dated notebook entry: verdict (supported / refuted / inconclusive),
   the exact findings keys and values, and one sentence on why. **A difference counts only if its
   95% CI excludes 1** (or the prediction lies outside the measured CI); otherwise "inconclusive".
   Check `meta.failed_configs` first: a config that never started is a finding, not missing data.
3. If a result breaks Amdahl (H1 refuted), explain with per-task JSONL (queueing? cache eviction
   during tool pauses, as in the Continuum paper?) before proposing anything.
4. Write `research/next_sweep.env` with the **smallest** sweep that resolves the most important
   open hypothesis, plus an estimated VM-hours line. Stop and ask the human owner before any TPU spend.
