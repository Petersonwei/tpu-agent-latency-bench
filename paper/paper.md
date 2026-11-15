# Faster Tokens, Same Wait? When TPU Inference Speed Actually Speeds Up AI Agents

Peter Wei · TorchTPU Sprint 2026 · vLLM-TorchTPU on Cloud TPU v6e

<!-- Every result number below is a {{placeholder}} filled from results/findings.json by
     paper/check_claims.py. Never type a result number by hand in sections 5–6. -->

## Abstract
LLM serving is usually judged by time-to-first-token (TTFT) and tokens per second. Agents are judged
by how fast a *task* finishes, and a task interleaves model calls with tool calls. We serve
Qwen3.5-4B with vLLM-TorchTPU on one Cloud TPU v6e chip and measure single-turn, multi-turn and
tool-calling agent tasks end to end. MTP-3 made token generation
{{predict.mtp3.token_speedup|.2f}}× faster on a chat benchmark, but agent tasks only
{{predict.mtp3.c1.tl0.5.task_speedup|.2f}}× faster (95% CI {{predict.mtp3.c1.tl0.5.task_speedup_ci_lo|.2f}}–{{predict.mtp3.c1.tl0.5.task_speedup_ci_hi|.2f}})
at 0.5 s per tool call. Predicting agent speedup from the token-speed number alone was off by up
to {{predict.max_rel_error.c1|.0%}}; splitting LLM time into prefill and decode first cut the error to
{{predict.max_rel_error_decomposed.c1|.0%}} with one session and {{predict.max_rel_error_decomposed.c8|.0%}} with eight.
Thinking mode delays the first visible token from {{thinking.first_visible_off_s|.2f}} s to
{{thinking.first_visible_on_s|.2f}} s. We release the harness, raw data and an agent-driven tuning loop ("autoserve").

## 1. Introduction
<!-- RQ1–RQ4 from research/literature-review.md §1; the motivating question from the sprint proposal. -->

## 2. Background and related work
<!-- Agent workloads (AgentSysBench, Aries, XPerf, TraceLab); serving levers (Autellix, Continuum,
     PASTE, Sarathi-Serve, AgentSpec); human response time (Miller 1968, Tan et al. CHI'26). -->

## 3. Method
### 3.1 Harness and workloads
<!-- harness/bench.py: 6 single-turn, 3 multi-turn, 6 tool tasks with checkable answers;
     deterministic mock tools with injected latency; 4k-token shared context. -->
### 3.2 Metrics and uncertainty
<!-- task time (p50/p95), TTFT vs time-to-first-visible-token, LLM/tool/overhead shares, prefill vs
     decode time, success; vLLM /metrics counters for prefix-cache hit rate and MTP acceptance.
     Mock tools sleep for a fixed time: the same method as NVIDIA's agentic trace replay. -->
Every speedup is a ratio of summed task times over the same tasks. Its 95% confidence interval comes
from a paired cluster bootstrap that resamples whole tasks (repeats of one task are not independent),
{{boot.n_resamples|d}} resamples over {{predict.n_agent_tasks|d}} agent tasks. We call a difference real only if the interval excludes 1.
### 3.3 The Amdahl lens, tested out of sample
If a fraction f of task time is LLM time and serving makes the LLM s× faster, the task speeds up
by S = 1 / ((1 − f) + f / s). Tool latency lowers f, so it caps S. Measuring s inside the same agent
run makes this an accounting identity, not a test. We instead take s from a plain single-turn chat
benchmark (decode time per token, one session), the number usually quoted. We then predict agent
task speedup and compare it with measurement at one and eight sessions. A second rule splits LLM
time into prefill (TTFT) and decode shares, each scaled by its own chat-benchmark speedup.

## 4. Setup: zero to serving on TPU v6e
<!-- setup_vm.sh steps, versions from results/env_versions.txt, cold vs warm server start,
     the parser/MTP flags, gotchas found. -->

## 5. Results
### 5.1 Where does the time go? (RQ1)
![breakdown](../results/figures/fig1_breakdown.png)
With no tool latency the LLM accounts for {{breakdown.tl0.llm_share|%}} of task time; at 2 s per
tool call it falls to {{breakdown.tl2.llm_share|%}}.

### 5.2 Faster tokens vs faster tasks (RQ2)
![prediction](../results/figures/fig5_prediction.png)
On the chat benchmark MTP-3 generated tokens {{predict.mtp3.token_speedup|.2f}}× faster. Amdahl from that
number predicted {{predict.mtp3.c1.tl0.5.predicted_task_speedup|.2f}}× for agent tasks at 0.5 s tool latency; we
measured {{predict.mtp3.c1.tl0.5.task_speedup|.2f}}× (95% CI {{predict.mtp3.c1.tl0.5.task_speedup_ci_lo|.2f}}–{{predict.mtp3.c1.tl0.5.task_speedup_ci_hi|.2f}})
with one session and {{predict.mtp3.c8.tl0.5.task_speedup|.2f}}× with eight. The prefill + decode rule predicted
{{predict.mtp3.c1.tl0.5.predicted_decomposed|.2f}}×. Turning prefix caching off left token speed unchanged
({{predict.noprefix.token_speedup|.2f}}×) yet changed agent task speed to {{predict.noprefix.c1.tl0.5.task_speedup|.2f}}×:
a token-speed benchmark cannot see it.

![amdahl](../results/figures/fig2_amdahl.png)
Across tool latencies, MTP-3 tasks became {{amdahl.mtp3.tl0.task_speedup|.2f}}× faster with instant tools,
{{amdahl.mtp3.tl0.5.task_speedup|.2f}}× at 0.5 s and {{amdahl.mtp3.tl2.task_speedup|.2f}}× at 2 s (95% CI
{{amdahl.mtp3.tl2.task_speedup_ci_lo|.2f}}–{{amdahl.mtp3.tl2.task_speedup_ci_hi|.2f}}).

Real tool latency is heavy-tailed. With lognormal tool latency of the same mean, MTP-3 sped agent tasks up
{{tail.mtp3.c1.tl0.5.task_speedup|.2f}}× (95% CI {{tail.mtp3.c1.tl0.5.task_speedup_ci_lo|.2f}}–{{tail.mtp3.c1.tl0.5.task_speedup_ci_hi|.2f}}),
against {{tail.mtp3.c1.tl0.5.fixed_task_speedup|.2f}}× with fixed latency.

### 5.3 Thinking mode (RQ3)
![thinking](../results/figures/fig3_thinking.png)
Thinking made tasks {{thinking.task_slowdown|.1f}}× slower (95% CI {{thinking.task_slowdown_ci_lo|.1f}}–{{thinking.task_slowdown_ci_hi|.1f}}); success went from {{thinking.success_off|%}} to
{{thinking.success_on|%}}.

### 5.4 Concurrency, context length and prefix caching (H5)
![concurrency](../results/figures/fig4_concurrency_prefix.png)
![context](../results/figures/fig6_context_prefix.png)
Prefix caching sped agent tasks up {{context.ctx4000.c1.tl0.5.prefix_speedup|.2f}}× with a 4k-token shared context
and {{context.ctx16000.c1.tl0.5.prefix_speedup|.2f}}× with 16k (95% CI {{context.ctx16000.c1.tl0.5.prefix_speedup_ci_lo|.2f}}–{{context.ctx16000.c1.tl0.5.prefix_speedup_ci_hi|.2f}}).
<!-- Qwen3.5 is a hybrid model (24 Gated-DeltaNet + 8 full-attention layers). vLLM's prefix caching
     for it is the experimental "align" mode with ~528-token blocks. Quote mech.base.c1.ctx*.prefix_hit_rate
     (from vLLM /metrics) to show the cache actually hit before attributing any speedup to it. -->

### 5.5 Autoserve
<!-- results/results.tsv: best config vs baseline, and whether the TTFT-best config was task-best. -->

## 6. Discussion
<!-- Rules of thumb for agent builders; when to buy faster inference vs fix tools;
     implications for perceived responsiveness (hypothesis for a user study, not a result). -->

## 7. Limitations
Single v6e chip, one 4B hybrid-attention model (prefix caching for it is experimental in vLLM),
synthetic tools (fixed sleeps, plus one heavy-tailed lognormal check), a small task set ({{predict.n_agent_tasks|d}} agent tasks, so
confidence intervals are wide), greedy decoding, simulator used only for rehearsal. Success checks
are substring matches.

## 8. Reproducibility
Code, raw JSONL, and commands: <repo URL>. Versions: `results/env_versions.txt`.
