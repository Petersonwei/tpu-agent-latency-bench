# Related work

Work that shaped the questions and the design of this benchmark. Each entry gives the source and
the one point we take from it. Most 2026 papers are arXiv preprints; see the originals for details.

## Research questions
- **RQ1 (where does the time go):** how much of end-to-end agent task time is LLM time (TTFT + decode),
  tool time and harness overhead, on vLLM-TorchTPU and Cloud TPU v6e?
- **RQ2 (elasticity):** when inference gets X% faster (speculative decoding, prefix caching, batching),
  how much faster do *tasks* get, and when does it stop mattering (Amdahl's law applied to agents)?
- **RQ3 (correctness):** do speed optimisations change task success?
- **RQ4 (future work):** is a faster agent perceived as better?

## Agent workloads are not chatbot workloads
| Work | Takeaway |
|---|---|
| Park et al. 2026, *Aries* — [arXiv:2607.29069](https://arxiv.org/abs/2607.29069) | Tool execution is 13–48% of end-to-end agent latency on GPU; token metrics (TTFT/TPOT) miss tool and harness bottlenecks. |
| Chang et al. 2026, *From LLM Inference to Agentic Workloads* (AgentSysBench) — [arXiv:2608.15127](https://arxiv.org/abs/2608.15127) | In 5 of 10 agentic applications, most latency is outside the LLM. |
| Zhu et al. 2026, *TraceLab* — [arXiv:2606.30560](https://arxiv.org/abs/2606.30560) | Real coding-agent sessions have long loops, long contexts, short outputs, heavy-tailed tool latency and high but imperfect prefix-cache hits. |
| Wang et al. 2026, *XPerf* — [arXiv:2608.20370](https://arxiv.org/abs/2608.20370) | Trace replay for agent benchmarks; decode dominates when unsaturated, queueing under load. |
| *Not All AI Agents Are Equal* — [arXiv:2609.19947](https://arxiv.org/abs/2609.19947) | The dominant time component varies by task, so results should be reported per workload. |
| Guo et al. 2026, *SAGA* — [arXiv:2605.00528](https://arxiv.org/abs/2605.00528) | Workflow-level scheduling beats request-level scheduling on agent task time: throughput-optimal is not task-optimal. |
| NVIDIA TensorRT-LLM, *Evaluating Agentic Serving with Trace Replay and Job-Level Metrics* — [tech blog](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog27_Evaluating_Agentic_Serving_with_Trace_Replay_and_Job_Level_Metrics.md) | Replays tool calls as timed sleeps (the method our mock tools use) and argues for job-level metrics over token throughput. |
| *LRAgent* — [arXiv:2602.01053](https://arxiv.org/abs/2602.01053) | Splits model latency from end-to-end agent latency, as this benchmark does. |

## Making agent serving faster
| Work | Takeaway |
|---|---|
| Luo et al. 2025, *Autellix* — [arXiv:2502.13965](https://arxiv.org/abs/2502.13965) | Program-aware scheduling; per-request TTFT is not per-task latency under concurrency. |
| Li et al. 2025, *Continuum* — [arXiv:2511.02230](https://arxiv.org/abs/2511.02230) | KV cache can be evicted during tool pauses; motivates sweeping tool latency × prefix caching. |
| Sui et al. 2026, *PASTE* — [arXiv:2603.18897](https://arxiv.org/abs/2603.18897) | Running tools speculatively in parallel with generation shortens tasks: the tool side is the other lever. |
| Lumer et al. 2026, *Don't Break the Cache* — [arXiv:2601.06007](https://arxiv.org/abs/2601.06007) | Prompt caching cuts agent TTFT; keep dynamic content at the end of the prompt. |
| Norgren 2026, *Stateful Inference* — [arXiv:2605.26289](https://arxiv.org/abs/2605.26289) | Most of each agent turn's prompt is unchanged, so session KV reuse pays off. |
| *Bursts That Outlast Themselves* (2026) — [arXiv:2611.13412](https://arxiv.org/abs/2611.13412) | With prefix caching on, a short burst of requests can push the cache-hit rate down for a long time. |
| Liu et al. 2026, *Speculative Decoding: Performance or Illusion?* (MLSys 2026) — [arXiv:2601.11580](https://arxiv.org/abs/2601.11580) | Real speculative-decoding speedups fall short of theory and vary by workload and batch size. |
| *AgentSpec* (2026) — [arXiv:2608.24004](https://arxiv.org/abs/2608.24004) | Speculative-decoding gains shrink as concurrency grows in agent workloads. |
| Agrawal et al. 2024, *Sarathi-Serve* (OSDI) — [arXiv:2403.02310](https://arxiv.org/abs/2403.02310) | Chunked prefill and the throughput–latency trade-off behind `MAX_NUM_BATCHED_TOKENS`. |
| Jiang et al. 2026, *Ragged Paged Attention* — [arXiv:2604.15464](https://arxiv.org/abs/2604.15464) | The TPU attention kernel used by the vLLM TPU backend. |

## Humans and response time
| Work | Takeaway |
|---|---|
| Miller 1968, *Response time in man-computer conversational transactions* (AFIPS FJCC) | Classic thresholds around 0.1 s, 2 s and 10 s; acceptable latency depends on the task. |
| Card, Moran & Newell 1980, *The Keystroke-Level Model* (CACM) | A framing for user task time versus system time. |
| Hoxmeier & DiCesare 2000 (AMCIS) | Satisfaction falls as response time rises. |
| Tan et al. 2026, *Impact of Response Latency and Task Type on Human-LLM Interaction* (CHI 2026) | Faster is not always better: 2 s answers were rated less thoughtful than slower ones. |
| Schlechte et al. 2025 (QoMEX) | End-to-end latency and response length drive waiting-time satisfaction. |

## Task success
| Work | Takeaway |
|---|---|
| Ghoshal & Al-Bustami 2026, *When Do Tools and Planning Help LLMs Think?* — [arXiv:2601.02663](https://arxiv.org/abs/2601.02663) | The accuracy/latency trade-off of agentic reasoning is task-dependent (our thinking on/off question). |

## TPU inference
| Work | Takeaway |
|---|---|
| Kishnani et al. 2026, *Gemma 4 31B on TPU* — [arXiv:2605.25645](https://arxiv.org/abs/2605.25645) | vLLM-TPU on v6e-8 vs 2×H100 with single-turn throughput sweeps; no agent or multi-turn workloads. |
| [vllm-project/vllm-torchtpu](https://github.com/vllm-project/vllm-torchtpu) | The serving engine used here. |
| [Cloud TPU v6e documentation](https://cloud.google.com/tpu/docs/v6e) | Hardware used here. |

## Methods
| Work | Takeaway |
|---|---|
| Amdahl 1967 (AFIPS) | Speeding up one part of a task bounds the overall speedup: S = 1 / ((1 − f) + f / s). |
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch), [windmaple/autofinetune](https://github.com/windmaple/autofinetune) | The agent experiment loop that `program.md` adapts to serving configuration. |

## Gap this project addresses
1. Published agent-serving characterisation is GPU-only; there is no public measurement of agent task
   latency on TPU with vLLM-TorchTPU.
2. Few studies connect inference speed → task time → perceived responsiveness in one reproducible harness.
3. No prior work tunes a serving configuration with an agent loop against task time instead of TTFT.
