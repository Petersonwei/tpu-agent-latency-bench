"""Agent responsiveness benchmark.

Runs single-turn, multi-turn and tool-calling agent tasks against an
OpenAI-compatible endpoint (vLLM-TorchTPU) or an offline simulator, and
decomposes end-to-end task time into: LLM time (TTFT + decode), tool time,
and harness overhead.

    python harness/bench.py --base-url http://localhost:8000/v1 --model Qwen/Qwen3.5-4B --thinking off
    python harness/bench.py --sim --sim-tps 60 --tool-latency 0.5      # offline

The last block of stdout is machine-readable (`key: value`); run_experiment.sh
greps `score:` from it. READ-ONLY for the optimisation loop (see program.md).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools import TOOL_SCHEMAS, call_tool  # noqa: E402
from workloads import SYSTEM, Task, get_tasks, grade  # noqa: E402


# ---------------------------------------------------------------- backends
@dataclass
class CallResult:
    text: str
    tool_calls: list[dict]
    ttft: float            # seconds to first token of any kind (incl. hidden reasoning)
    total: float           # seconds for the whole call
    prompt_tokens: int
    completion_tokens: int
    ttfa: float = 0.0      # seconds to first *visible* answer/tool-call token (what a user sees)
    reasoning_chars: int = 0
    finish_reason: str = ""


class OpenAIBackend:
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", thinking: str = "default",
                 think_budget: int = 1024):
        from openai import AsyncOpenAI  # imported lazily so --sim needs no deps
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=600)
        self.model = model
        self.thinking = thinking
        self.think_budget = think_budget  # extra max_tokens so reasoning can't starve the answer

    async def chat(self, messages, tools, max_tokens, task=None) -> CallResult:
        t0 = time.perf_counter()
        ttft = ttfa = None
        text, calls, usage, reasoning, finish = [], {}, None, 0, ""
        if self.thinking != "off":
            max_tokens += self.think_budget
        kwargs = dict(model=self.model, messages=messages, max_tokens=max_tokens,
                      temperature=0.0, stream=True, stream_options={"include_usage": True})
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        if self.thinking != "default":  # Qwen3/3.5 + Gemma 4 chat templates honour this
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": self.thinking == "on"}}
        stream = await self.client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue
            if chunk.choices[0].finish_reason:
                finish = chunk.choices[0].finish_reason
            d = chunk.choices[0].delta
            r = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
            if (d.content or d.tool_calls or r) and ttft is None:
                ttft = time.perf_counter() - t0
            if (d.content or d.tool_calls) and ttfa is None:
                ttfa = time.perf_counter() - t0
            if r:
                reasoning += len(r)
            if d.content:
                text.append(d.content)
            for tc in d.tool_calls or []:
                c = calls.setdefault(tc.index, {"id": tc.id or f"call_{tc.index}", "name": "", "arguments": ""})
                if tc.id:
                    c["id"] = tc.id
                if tc.function and tc.function.name:
                    c["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    c["arguments"] += tc.function.arguments
        total = time.perf_counter() - t0
        return CallResult("".join(text), [calls[i] for i in sorted(calls)], ttft if ttft is not None else total,
                          total, getattr(usage, "prompt_tokens", 0) or 0,
                          getattr(usage, "completion_tokens", 0) or 0,
                          ttfa if ttfa is not None else total, reasoning, finish)


class SimBackend:
    """Offline model of an LLM server: TTFT = base + prompt/prefill_tps; decode = out/tps.

    Lets you develop the harness and sketch the 'Amdahl curve' on a laptop.
    """

    def __init__(self, ttft_base: float, prefill_tps: float, tps: float, think_tokens: int = 0):
        self.ttft_base, self.prefill_tps, self.tps = ttft_base, prefill_tps, tps
        self.think_tokens = think_tokens

    async def chat(self, messages, tools, max_tokens, task: Task | None = None) -> CallResult:
        prompt_tokens = sum(len(json.dumps(m)) for m in messages) // 4
        # how many tool results since the last user message?
        done = 0
        for m in reversed(messages):
            if m["role"] == "user":
                break
            done += m["role"] == "tool"
        tool_calls, text = [], ""
        if task and task.plan and done < len(task.plan):
            name, args = task.plan[done]
            tool_calls = [{"id": f"sim_{done}", "name": name, "arguments": json.dumps(args)}]
            out_tokens = 25
        else:
            n_user = sum(m["role"] == "user" for m in messages)
            last = task is None or n_user >= len(task.turns)
            text = f"Answer: {task.expect}." if (task and task.expect and last) else "OK."
            out_tokens = min(max_tokens, 60)
        ttft = self.ttft_base + prompt_tokens / self.prefill_tps
        think = self.think_tokens / self.tps
        decode = out_tokens / self.tps
        t0 = time.perf_counter()
        await asyncio.sleep(ttft + think + decode)
        return CallResult(text, tool_calls, ttft, time.perf_counter() - t0, prompt_tokens,
                          out_tokens + self.think_tokens, ttft + think, self.think_tokens * 4)


# ---------------------------------------------------------------- agent loop
@dataclass
class TaskRecord:
    task_id: str
    kind: str
    repeat: int
    success: bool = False
    e2e_s: float = 0.0
    llm_s: float = 0.0
    ttft_s: list[float] = field(default_factory=list)
    ttfa_s: list[float] = field(default_factory=list)
    first_visible_s: float = 0.0   # task start -> first visible token (perceived responsiveness)
    tool_s: float = 0.0
    overhead_s: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prefill_s: float = 0.0        # sum of per-call TTFT (prompt processing + queueing)
    decode_s: float = 0.0         # sum of per-call (total - TTFT): token generation
    error: str = ""
    truncated_calls: int = 0      # finish_reason == "length": answer cut off by max_tokens
    final_answer: str = ""
    tool_latencies: list[float] = field(default_factory=list)  # injected latency per tool call


def context_pad(n_tokens: int) -> str:
    """Deterministic filler that mimics an agent's long, shared context (repo files, docs)."""
    if n_tokens <= 0:
        return ""
    line = "def handler_{i}(req):\n    \"\"\"Return the record for request {i}.\"\"\"\n    return db.get({i})\n"
    out, i = [], 0
    while len(out) < n_tokens // 18:      # ~18 tokens per line
        out.append(line.format(i=i)); i += 1
    return "\n\n# Project context (reference only)\n" + "".join(out)


def tool_latency_for(task_id: str, repeat: int, n: int, mean: float, dist: str = "fixed",
                     sigma: float = 1.0) -> float:
    """Injected latency for the n-th tool call of (task, repeat).

    fixed:     every call waits `mean` seconds.
    lognormal: heavy-tailed, same mean (mu = ln(mean) - sigma^2/2). Real tool latency is
               long-tailed (TraceLab, arXiv 2606.30560). The draw is seeded by (task, repeat, n)
               only, so two server configs see the SAME latencies -> paired comparison stays valid.
    """
    if mean <= 0 or dist == "fixed":
        return max(mean, 0.0)
    rng = random.Random(f"{task_id}|{repeat}|{n}")
    return rng.lognormvariate(math.log(mean) - sigma ** 2 / 2, sigma)


async def run_task(backend, task: Task, repeat: int, tool_latency: float, max_steps: int,
                   pad_tokens: int = 0, tool_dist: str = "fixed", tool_sigma: float = 1.0) -> TaskRecord:
    rec = TaskRecord(task.id, task.kind, repeat)
    messages = [{"role": "system", "content": SYSTEM + context_pad(pad_tokens)}]
    tools = TOOL_SCHEMAS if task.kind == "tool" else None
    t_start = time.perf_counter()
    answer = ""
    try:
        for user_msg in task.turns:
            messages.append({"role": "user", "content": user_msg})
            for _ in range(max_steps):
                r = await backend.chat(messages, tools, task.max_tokens, task=task)
                rec.llm_calls += 1
                rec.llm_s += r.total
                if not rec.ttfa_s:
                    rec.first_visible_s = (time.perf_counter() - t_start) - r.total + r.ttfa
                rec.ttft_s.append(r.ttft)
                rec.prefill_s += r.ttft
                rec.decode_s += max(0.0, r.total - r.ttft)
                rec.ttfa_s.append(r.ttfa)
                rec.prompt_tokens += r.prompt_tokens
                rec.completion_tokens += r.completion_tokens
                rec.truncated_calls += r.finish_reason == "length"
                if r.tool_calls:
                    messages.append({"role": "assistant", "content": r.text or None, "tool_calls": [
                        {"id": c["id"], "type": "function",
                         "function": {"name": c["name"], "arguments": c["arguments"]}} for c in r.tool_calls]})
                    lats = [tool_latency_for(task.id, repeat, rec.tool_calls + j, tool_latency, tool_dist, tool_sigma)
                            for j in range(len(r.tool_calls))]
                    rec.tool_latencies += lats
                    t_tool = time.perf_counter()
                    results = await asyncio.gather(*[call_tool(c["name"], c["arguments"], lat)
                                                     for c, lat in zip(r.tool_calls, lats)])
                    rec.tool_s += time.perf_counter() - t_tool
                    rec.tool_calls += len(r.tool_calls)
                    for c, res in zip(r.tool_calls, results):
                        messages.append({"role": "tool", "tool_call_id": c["id"], "content": res})
                    continue
                answer = r.text
                messages.append({"role": "assistant", "content": answer})
                break
            else:
                rec.error = "max_steps exceeded"
        rec.final_answer = answer[:500]
        rec.success = grade(task, answer, rec.error)
    except Exception as e:  # noqa: BLE001 - record and continue
        rec.error = f"{type(e).__name__}: {e}"[:300]
    rec.e2e_s = time.perf_counter() - t_start
    rec.overhead_s = max(0.0, rec.e2e_s - rec.llm_s - rec.tool_s)
    return rec


# ---------------------------------------------------------------- server metrics (mechanism evidence)
# vLLM Prometheus counters (docs.vllm.ai design/metrics). They explain WHY a config is faster:
# prefix-cache hit rate (does caching actually work for a hybrid GDN model like Qwen3.5?) and the
# speculative-decoding acceptance rate (does MTP still pay off at concurrency 8?).
SERVER_COUNTERS = ("prefix_cache_queries", "prefix_cache_hits",
                   "spec_decode_num_accepted_tokens", "spec_decode_num_draft_tokens")


def parse_prometheus(text: str) -> dict:
    out = {k: 0.0 for k in SERVER_COUNTERS}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        base = name.removeprefix("vllm:").removesuffix("_total")
        if name.startswith("vllm:") and base in out:
            try:
                out[base] += float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return out


def scrape_metrics(base_url: str) -> dict | None:
    import urllib.request
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(root + "/metrics", timeout=5) as r:
            return parse_prometheus(r.read().decode())
    except Exception:  # noqa: BLE001 - metrics are optional evidence, never fail a run for them
        return None


def metrics_delta(before: dict | None, after: dict | None) -> dict:
    if not before or not after:
        return {}
    d = {k: after[k] - before[k] for k in SERVER_COUNTERS}
    d["prefix_hit_rate"] = d["prefix_cache_hits"] / d["prefix_cache_queries"] if d["prefix_cache_queries"] else None
    d["spec_accept_rate"] = (d["spec_decode_num_accepted_tokens"] / d["spec_decode_num_draft_tokens"]
                             if d["spec_decode_num_draft_tokens"] else None)
    return d


# ---------------------------------------------------------------- reporting
def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    return xs[f] if f == c else xs[f] + (xs[c] - xs[f]) * (k - f)


def summarise(recs: list[TaskRecord], wall: float) -> dict:
    out = {}
    for kind in ["single", "multi", "tool", "all"]:
        rs = [r for r in recs if kind == "all" or r.kind == kind]
        if not rs:
            continue
        e2e = [r.e2e_s for r in rs]
        ttft = [t for r in rs for t in r.ttft_s]
        ttfa = [t for r in rs for t in r.ttfa_s]
        tot = sum(e2e) or 1e-9
        out[kind] = {
            "n": len(rs),
            "success_rate": sum(r.success for r in rs) / len(rs),
            "task_p50_s": pct(e2e, 50), "task_p95_s": pct(e2e, 95),
            "ttft_p50_s": pct(ttft, 50), "ttft_p95_s": pct(ttft, 95),
            "ttfa_p50_s": pct(ttfa, 50),
            "first_visible_p50_s": pct([r.first_visible_s for r in rs], 50),
            "llm_share": sum(r.llm_s for r in rs) / tot,
            "tool_share": sum(r.tool_s for r in rs) / tot,
            "overhead_share": sum(r.overhead_s for r in rs) / tot,
            "llm_calls_per_task": statistics.mean(r.llm_calls for r in rs),
            "truncated_calls": sum(r.truncated_calls for r in rs),
        }
    out["all"]["output_tok_per_s"] = sum(r.completion_tokens for r in recs) / max(wall, 1e-9)
    out["all"]["wall_s"] = wall
    return out


async def main_async(a) -> int:
    if a.sim:
        backend = SimBackend(a.sim_ttft, a.sim_prefill_tps, a.sim_tps, a.sim_think_tokens)
    else:
        backend = OpenAIBackend(a.base_url, a.model, thinking=a.thinking, think_budget=a.think_budget)
    tasks = get_tasks(a.workload)

    for i in range(a.warmup):  # exclude TPU graph compilation from measurements
        await run_task(backend, tasks[i % len(tasks)], -1, 0.0, a.max_steps, a.context_tokens)

    sem = asyncio.Semaphore(a.concurrency)
    jobs = [(t, rep) for rep in range(a.repeats) for t in tasks]
    # Stagger the first wave of concurrent sessions (uniform 0..stagger s, seeded) so identical
    # sessions don't stay phase-aligned (NVIDIA agentic trace-replay methodology). Outside e2e_s.
    jitter = random.Random(0)
    delays = [jitter.uniform(0, a.stagger) if (a.concurrency > 1 and i < a.concurrency) else 0.0
              for i in range(len(jobs))]

    async def guarded(t, rep, delay):
        if delay:
            await asyncio.sleep(delay)
        async with sem:
            return await run_task(backend, t, rep, a.tool_latency, a.max_steps, a.context_tokens,
                                  a.tool_dist, a.tool_sigma)

    m_before = None if a.sim else scrape_metrics(a.base_url)
    t0 = time.perf_counter()
    recs = await asyncio.gather(*[guarded(t, rep, d) for (t, rep), d in zip(jobs, delays)])
    wall = time.perf_counter() - t0
    server_metrics = {} if a.sim else metrics_delta(m_before, scrape_metrics(a.base_url))
    s = summarise(list(recs), wall)

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            meta = {k: v for k, v in vars(a).items()}
            f.write(json.dumps({"meta": meta, "summary": s, "server_metrics": server_metrics}) + "\n")
            for r in recs:
                f.write(json.dumps(asdict(r)) + "\n")

    print(f"{'kind':<7}{'n':>4}{'succ':>7}{'task_p50':>10}{'task_p95':>10}{'ttft_p50':>10}"
          f"{'llm%':>7}{'tool%':>7}{'ovh%':>7}")
    for k, v in s.items():
        print(f"{k:<7}{v['n']:>4}{v['success_rate']:>7.0%}{v['task_p50_s']:>10.3f}{v['task_p95_s']:>10.3f}"
              f"{v['ttft_p50_s']:>10.3f}{v['llm_share']:>7.0%}{v['tool_share']:>7.0%}{v['overhead_share']:>7.0%}")
    errs = [r for r in recs if r.error]
    for r in errs[:5]:
        print(f"error {r.task_id}#{r.repeat}: {r.error}", file=sys.stderr)

    al = s["all"]
    valid = al["success_rate"] >= a.min_success and not (len(errs) > len(recs) // 2)
    print("---")
    print(f"score: {al['task_p50_s']:.4f}" if valid else "score: inf")
    for key in ["task_p50_s", "task_p95_s", "ttft_p50_s", "ttfa_p50_s", "first_visible_p50_s",
                "success_rate", "llm_share", "tool_share", "output_tok_per_s", "truncated_calls"]:
        print(f"{key}: {al[key]:.4f}")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen3.5-4B"))
    p.add_argument("--workload", default="all", choices=["all", "single", "multi", "tool"])
    p.add_argument("--concurrency", type=int, default=4, help="simultaneous agent sessions")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--tool-latency", type=float, default=0.0, help="seconds added to every tool call (mean)")
    p.add_argument("--tool-dist", default="fixed", choices=["fixed", "lognormal"],
                   help="fixed: every call waits --tool-latency; lognormal: heavy-tailed with that mean")
    p.add_argument("--tool-sigma", type=float, default=1.0, help="lognormal shape (1.0: p95 ~ 3x mean)")
    p.add_argument("--stagger", type=float, default=1.0,
                   help="first-wave start jitter (s) when concurrency > 1; 0 = all start together")
    p.add_argument("--max-steps", type=int, default=8, help="LLM calls allowed per user turn")
    p.add_argument("--min-success", type=float, default=0.0, help="score is inf below this success rate")
    p.add_argument("--out", default="", help="write per-task JSONL here")
    p.add_argument("--thinking", default="default", choices=["default", "on", "off"],
                   help="per-request enable_thinking via chat_template_kwargs (Qwen3.5 / Gemma 4)")
    p.add_argument("--think-budget", type=int, default=1024,
                   help="extra max_tokens per call when thinking is not off")
    p.add_argument("--context-tokens", type=int, default=0,
                   help="pad the system prompt with ~N tokens of shared agent context (prefix-cache test)")
    p.add_argument("--sim", action="store_true", help="use the offline simulator instead of a server")
    p.add_argument("--sim-ttft", type=float, default=0.15)
    p.add_argument("--sim-prefill-tps", type=float, default=8000.0)
    p.add_argument("--sim-tps", type=float, default=60.0, help="decode tokens/s per stream")
    p.add_argument("--sim-think-tokens", type=int, default=0, help="hidden reasoning tokens per call")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async(parse_args())))
