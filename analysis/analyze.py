"""Turn a sweep (sweep.tsv + sweep/*.jsonl) into figures, a results table and findings.json.

    python analysis/analyze.py --results-dir results            # real TPU data
    python analysis/analyze.py --results-dir results/rehearsal  # simulator rehearsal

findings.json holds every number the paper/blog may quote (see paper/check_claims.py).
Core idea (out-of-sample): predict an agent's task speedup from the number people usually quote
— the token-speed (time-per-output-token) speedup measured on plain single-turn chat — via Amdahl
    S_task = 1 / ((1 - f) + f / s_tok)
where f = LLM share of agent task time in the baseline. Then test that prediction against the
measured agent task speedup (tool + multi-turn tasks), at 1 and 8 concurrent sessions.
(The older `amdahl.*` keys use s measured inside the same agent run; that is an accounting check,
close to true by construction, and must not be presented as a validation.)
Uncertainty: 95% CIs from a paired cluster bootstrap over task ids (repeats of one task are not
independent; resample whole tasks), 2000 resamples, fixed seed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = "base"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))
from workloads import get_tasks, grade  # noqa: E402

TASKS_BY_ID = {t.id: t for t in get_tasks("all")}


def load_tsv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            for k, v in r.items():
                if k not in ("server_cfg", "thinking", "tool_dist"):
                    try:
                        r[k] = float(v)
                    except (TypeError, ValueError):
                        r[k] = float("nan")
            r["tool_dist"] = r.get("tool_dist") or "fixed"   # older sweeps had no column
            rows.append(r)
    return rows


def tag(cfg, th, tl, c, ctx):
    tl_s = f"{tl:g}" if tl != int(tl) else str(int(tl)) if False else f"{tl:g}"
    return f"{cfg}_th{th}_tl{tl_s}_c{int(c)}_ctx{int(ctx)}"


def load_cell(results_dir, row):
    """Per-task records for one sweep cell (None if missing)."""
    cands = [tag(row["server_cfg"], row["thinking"], row["tool_latency"], row["concurrency"], row["context"])]
    # the shell writes the tool latency exactly as passed (e.g. 0.05, 0.0, 2) — try common spellings
    tl = row["tool_latency"]
    for spell in {str(tl), f"{tl:g}", f"{tl:.1f}", str(int(tl)) if tl == int(tl) else str(tl)}:
        cands.append(f"{row['server_cfg']}_th{row['thinking']}_tl{spell}_c{int(row['concurrency'])}"
                     f"_ctx{int(row['context'])}")
    if row.get("tool_dist", "fixed") != "fixed":
        cands = [c + "_d" + row["tool_dist"] for c in cands]
    for c in cands:
        p = os.path.join(results_dir, "sweep", c + ".jsonl")
        if os.path.exists(p):
            with open(p) as f:
                lines = f.read().splitlines()
            return [json.loads(l) for l in lines[1:]], json.loads(lines[0])
    return None, None


TIME_FIELDS = ("e2e_s", "llm_s", "tool_s", "overhead_s", "first_visible_s", "prefill_s", "decode_s")
AGENT = ("tool", "multi")
N_BOOT = 2000


def scaled(recs, k):
    out = []
    for r in recs:
        r = dict(r)
        for f in TIME_FIELDS:
            r[f] = r.get(f, 0.0) * k
        out.append(r)
    return out


def per_task(recs, kinds=None):
    """Sum per task id over repeats (errors excluded): the resampling unit of the cluster bootstrap."""
    agg = defaultdict(lambda: defaultdict(float))
    for r in recs:
        if r["error"] or (kinds and r["kind"] not in kinds):
            continue
        a = agg[r["task_id"]]
        for f in TIME_FIELDS:
            a[f] += r.get(f, 0.0)
        a["tokens"] += r["completion_tokens"]; a["calls"] += r["llm_calls"]; a["n"] += 1
    return agg


def _time(field):
    return lambda a, ids: sum(a[i][field] for i in ids)


def _tpot(a, ids):  # seconds per generated token, decode phase only
    return sum(a[i]["decode_s"] for i in ids) / max(sum(a[i]["tokens"] - a[i]["calls"] for i in ids), 1)


def paired_ratio(num_recs, den_recs, kinds=None, stat=_time("e2e_s"), seed=0):
    """stat(num)/stat(den) over task ids present in both, with a 95% paired cluster-bootstrap CI.
    Returns (estimate, lo, hi, n_tasks) or None."""
    a, b = per_task(num_recs, kinds), per_task(den_recs, kinds)
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    est = stat(a, ids) / max(stat(b, ids), 1e-12)
    rng, bs = random.Random(seed), []
    for _ in range(N_BOOT):
        smp = [rng.choice(ids) for _ in ids]
        bs.append(stat(a, smp) / max(stat(b, smp), 1e-12))
    bs.sort()
    return est, bs[int(0.025 * N_BOOT)], bs[int(0.975 * N_BOOT) - 1], len(ids)


def put(findings, key, res):
    if res:
        findings[key], findings[key + "_ci_lo"], findings[key + "_ci_hi"] = res[0], res[1], res[2]


def means(recs):
    ok = [r for r in recs if not r["error"]]
    return {
        "e2e": statistics.mean(r["e2e_s"] for r in ok),
        "llm": statistics.mean(r["llm_s"] for r in ok),
        "tool": statistics.mean(r["tool_s"] for r in ok),
        "ovh": statistics.mean(r["overhead_s"] for r in ok),
        "first_visible": statistics.mean(r["first_visible_s"] for r in ok),
        "success": sum(r["success"] for r in recs) / len(recs),
    }


def fmt_rate(x):
    return "–" if x is None else f"{x:.0%}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--concurrency", type=float, default=1, help="concurrency used for the Amdahl/thinking figures")
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help="multiply times and tool latencies (use SIM_SCALE for simulator rehearsals)")
    a = ap.parse_args(argv)
    rd = a.results_dir
    figdir = os.path.join(rd, "figures")
    os.makedirs(figdir, exist_ok=True)
    rows = load_tsv(os.path.join(rd, "sweep.tsv"))
    cells, raw, smet = {}, {}, {}
    tail = {}   # (cfg, tl, c, ctx) -> records with heavy-tailed (lognormal) tool latency
    for r in rows:
        recs, head = load_cell(rd, r)
        for rec in recs or []:  # re-grade with current rules: fixes grader bugs without re-running the TPU
            t = TASKS_BY_ID.get(rec["task_id"])
            if t is not None:
                rec["success"] = grade(t, rec.get("final_answer", ""), rec["error"])
        if recs and r["tool_dist"] != "fixed":
            tail[(r["server_cfg"], round(r["tool_latency"] * a.time_scale, 6), r["concurrency"], r["context"])] = \
                scaled(recs, a.time_scale)
            continue
        if recs:
            m = {k: (v if k == "success" else v * a.time_scale) for k, v in means(recs).items()}
            tl = round(r["tool_latency"] * a.time_scale, 6)
            key = (r["server_cfg"], r["thinking"], tl, r["concurrency"], r["context"])
            cells[key], raw[key] = m, scaled(recs, a.time_scale)
            smet[key] = (head or {}).get("server_metrics") or {}
    if not cells:
        raise SystemExit(f"no per-task JSONL found under {rd}/sweep")
    ctx = sorted({k[4] for k in cells})[0]
    C = a.concurrency
    tls = sorted({k[2] for k in cells})
    cfgs = sorted({k[0] for k in cells}, key=lambda x: (x != BASE, x))
    failed = []
    fp = os.path.join(rd, "logs", "failed_configs.txt")
    if os.path.exists(fp):
        failed = [l.split("\t")[0] for l in open(fp).read().splitlines() if l.strip()]
    findings = {"meta": {"results_dir": rd, "concurrency": C, "context": ctx, "n_cells": len(cells),
                         "time_scale": a.time_scale, "simulated": a.time_scale != 1.0,
                         "failed_configs": failed, "n_boot": N_BOOT}, "boot.n_resamples": N_BOOT}

    # ---- Figure 1: where does the time go (baseline, thinking off)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    labels, llm, tool, ovh = [], [], [], []
    for tl in tls:
        m = cells.get((BASE, "off", tl, C, ctx))
        if not m:
            continue
        labels.append(f"{tl:g}s")
        llm.append(m["llm"]); tool.append(m["tool"]); ovh.append(m["ovh"])
        findings[f"breakdown.tl{tl:g}.llm_share"] = m["llm"] / m["e2e"]
        findings[f"breakdown.tl{tl:g}.task_mean_s"] = m["e2e"]
    ax.bar(labels, llm, label="LLM (TTFT + decode)", color="#3b6fb6")
    ax.bar(labels, tool, bottom=llm, label="tool execution", color="#e0a030")
    ax.bar(labels, ovh, bottom=[x + y for x, y in zip(llm, tool)], label="harness overhead", color="#999")
    ax.set_xlabel("injected tool latency per call"); ax.set_ylabel("mean task time (s)")
    ax.set_title("Where agent task time goes (baseline)"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig1_breakdown.png"), dpi=160); plt.close(fig)

    # ---- Figure 2: agent Amdahl curve, measured vs predicted
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#3b6fb6", "#e0a030", "#c0392b", "#2e8b57"]
    worst_err = 0.0
    obs = [cells[(BASE, "off", tl, C, ctx)]["llm"] / m["llm"] for (cfg, th, tl, c, cx), m in cells.items()
           if th == "off" and c == C and cx == ctx and (BASE, "off", tl, C, ctx) in cells]
    x_lo, x_hi = min(0.8, min(obs) * 0.95), max(1.6, max(obs) * 1.25)
    for i, tl in enumerate(tls):
        b = cells.get((BASE, "off", tl, C, ctx))
        if not b:
            continue
        f = b["llm"] / b["e2e"]
        xs = [x_lo + j * (x_hi - x_lo) / 100 for j in range(101)]
        ax.plot(xs, [1 / ((1 - f) + f / s) for s in xs], color=colors[i % 4], lw=1, alpha=.6)
        for cfg in cfgs:
            m = cells.get((cfg, "off", tl, C, ctx))
            if not m or cfg == BASE:
                continue
            s_inf = b["llm"] / m["llm"]
            s_task = b["e2e"] / m["e2e"]
            pred = 1 / ((1 - f) + f / s_inf)
            worst_err = max(worst_err, abs(s_task - pred) / pred)
            ax.scatter([s_inf], [s_task], color=colors[i % 4], s=28, zorder=3)
            if i == 0:  # label configs once to avoid overlap
                ax.annotate(cfg, (s_inf, s_task), fontsize=7, xytext=(4, -10), textcoords="offset points")
            findings[f"amdahl.{cfg}.tl{tl:g}.inference_speedup"] = s_inf
            findings[f"amdahl.{cfg}.tl{tl:g}.task_speedup"] = s_task
            put(findings, f"amdahl.{cfg}.tl{tl:g}.task_speedup",
                paired_ratio(raw[(BASE, "off", tl, C, ctx)], raw[(cfg, "off", tl, C, ctx)]))
            findings[f"amdahl.{cfg}.tl{tl:g}.predicted_task_speedup"] = pred
        ax.plot([], [], color=colors[i % 4], label=f"tool latency {tl:g}s (f={f:.2f})")
    ax.plot([x_lo, x_hi], [x_lo, x_hi], ls=":", color="#666", lw=1, label="ideal (task = inference)")
    ax.set_xlabel("inference speedup (LLM time, baseline / config)")
    ax.set_ylabel("task speedup (task time, baseline / config)")
    ax.set_title("Faster tokens vs faster tasks: measured (dots) vs Amdahl (lines)")
    ax.axhline(1, color="#ccc", lw=.8); ax.axvline(1, color="#ccc", lw=.8)
    ax.legend(frameon=False, fontsize=7, loc="upper left"); ax.set_xlim(x_lo, x_hi); ax.set_ylim(x_lo, x_hi)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig2_amdahl.png"), dpi=160); plt.close(fig)
    findings["amdahl.max_relative_error"] = worst_err

    # ---- Figure 3: thinking on vs off (baseline, middle tool latency)
    tl_mid = tls[len(tls) // 2]
    off, on = cells.get((BASE, "off", tl_mid, C, ctx)), cells.get((BASE, "on", tl_mid, C, ctx))
    if off and on:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        keys = [("first_visible", "first visible token"), ("e2e", "task time")]
        x = range(len(keys))
        ax.bar([i - .2 for i in x], [off[k] for k, _ in keys], .4, label="thinking off", color="#3b6fb6")
        ax.bar([i + .2 for i in x], [on[k] for k, _ in keys], .4, label="thinking on", color="#c0392b")
        ax.set_xticks(list(x)); ax.set_xticklabels([l for _, l in keys]); ax.set_ylabel("seconds (mean)")
        ax.set_title(f"Thinking mode (tool latency {tl_mid:g}s); success {off['success']:.0%} → {on['success']:.0%}")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig3_thinking.png"), dpi=160); plt.close(fig)
        findings.update({
            "thinking.first_visible_off_s": off["first_visible"], "thinking.first_visible_on_s": on["first_visible"],
            "thinking.task_off_s": off["e2e"], "thinking.task_on_s": on["e2e"],
            "thinking.success_off": off["success"], "thinking.success_on": on["success"],
            "thinking.task_slowdown": on["e2e"] / off["e2e"],
        })
        kon, koff = (BASE, "on", tl_mid, C, ctx), (BASE, "off", tl_mid, C, ctx)
        put(findings, "thinking.task_slowdown", paired_ratio(raw[kon], raw[koff]))
        put(findings, "thinking.first_visible_slowdown",
            paired_ratio(raw[kon], raw[koff], stat=_time("first_visible_s")))

    # ---- Figure 4: concurrency × prefix caching (thinking off, middle tool latency)
    concs = sorted({k[3] for k in cells})
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for i, cfg in enumerate([c for c in (BASE, "noprefix") if c in cfgs]):
        ys = [cells[(cfg, "off", tl_mid, c, ctx)]["e2e"] for c in concs if (cfg, "off", tl_mid, c, ctx) in cells]
        ax.plot([int(c) for c in concs[:len(ys)]], ys, marker="o", label=cfg, color=colors[i])
        for c, y in zip(concs, ys):
            findings[f"concurrency.{cfg}.c{int(c)}.task_mean_s"] = y
    ax.set_xlabel("concurrent agent sessions"); ax.set_ylabel("mean task time (s)")
    ax.set_title("Concurrency and prefix caching"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig4_concurrency_prefix.png"), dpi=160); plt.close(fig)

    # ---- Figure 5 (headline): does the token-speed number predict the agent speedup?
    # s_tok: decode time-per-token speedup on single-turn chat, 1 session, thinking off — the number a
    # tokens/s benchmark reports. Prediction is then tested on agent tasks at every concurrency.
    concs = sorted({k[3] for k in cells})
    tl0 = tls[0]
    findings["predict.n_agent_tasks"] = len(per_task(raw.get((BASE, "off", tl0, C, ctx), []), AGENT))
    pred_rows = []  # (cfg, c, s_tok, pred, meas, lo, hi)
    for cfg in cfgs:
        if cfg == BASE or (BASE, "off", tl0, C, ctx) not in raw or (cfg, "off", tl0, C, ctx) not in raw:
            continue
        tok = paired_ratio(raw[(BASE, "off", tl0, C, ctx)], raw[(cfg, "off", tl0, C, ctx)],
                           kinds=("single",), stat=_tpot)
        if not tok:
            continue
        put(findings, f"predict.{cfg}.token_speedup", tok)
        pre = paired_ratio(raw[(BASE, "off", tl0, C, ctx)], raw[(cfg, "off", tl0, C, ctx)],
                           kinds=("single",), stat=_time("prefill_s"))
        put(findings, f"predict.{cfg}.prefill_speedup", pre)
        for c in concs:
            for tl in tls:
                kb, kc = (BASE, "off", tl, c, ctx), (cfg, "off", tl, c, ctx)
                if kb not in raw or kc not in raw:
                    continue
                ab = per_task(raw[kb], AGENT)
                f = sum(v["llm_s"] for v in ab.values()) / max(sum(v["e2e_s"] for v in ab.values()), 1e-12)
                pred = 1 / ((1 - f) + f / tok[0])
                # decomposed rule: prefill and decode shares scale by their own speedups
                e2e = max(sum(v["e2e_s"] for v in ab.values()), 1e-12)
                f_pre = sum(v["prefill_s"] for v in ab.values()) / e2e
                f_dec = sum(v["decode_s"] for v in ab.values()) / e2e
                pred2 = 1 / ((1 - f_pre - f_dec) + f_pre / (pre[0] if pre else 1.0) + f_dec / tok[0])
                meas = paired_ratio(raw[kb], raw[kc], kinds=AGENT)
                if not meas:
                    continue
                k = f"predict.{cfg}.c{int(c)}.tl{tl:g}"
                findings[k + ".prefill_share_base"], findings[k + ".decode_share_base"] = f_pre, f_dec
                findings[k + ".predicted_decomposed"] = pred2
                err2 = abs(meas[0] - pred2) / pred2
                findings[k + ".rel_error_decomposed"] = err2
                findings[k + ".decomposed_within_ci"] = int(meas[1] <= pred2 <= meas[2])
                mk2 = f"predict.max_rel_error_decomposed.c{int(c)}"
                findings[mk2] = max(findings.get(mk2, 0.0), err2)
                findings[k + ".llm_share_base"] = f
                findings[k + ".predicted_task_speedup"] = pred
                put(findings, k + ".task_speedup", meas)
                err = abs(meas[0] - pred) / pred
                findings[k + ".rel_error"] = err
                findings[k + ".within_ci"] = int(meas[1] <= pred <= meas[2])
                mk = f"predict.max_rel_error.c{int(c)}"
                findings[mk] = max(findings.get(mk, 0.0), err)
                if tl == tl_mid:
                    pred_rows.append((cfg, c, tok[0], pred, meas[0], meas[1], meas[2], pred2))
    if pred_rows:
        names = sorted({r[0] for r in pred_rows})
        fig, ax = plt.subplots(figsize=(6.6, 3.8))
        nb = 3 + len(concs)
        w = 0.84 / nb
        for i, cfg in enumerate(names):
            rs = [r for r in pred_rows if r[0] == cfg]
            x0 = i - w * (nb - 1) / 2
            ax.bar(x0, rs[0][2], w, color="#bbb", label="token speedup (chat benchmark)" if i == 0 else None)
            ax.bar(x0 + w, rs[0][3], w, color="#e0a030", label="Amdahl from token speedup" if i == 0 else None)
            ax.bar(x0 + 2 * w, rs[0][7], w, color="#8e6bbf", label="Amdahl, prefill + decode split" if i == 0 else None)
            for j, r in enumerate(sorted(rs, key=lambda r: r[1])):
                x = x0 + (3 + j) * w
                ax.bar(x, r[4], w, color=("#3b6fb6", "#86aee0", "#1f3f73")[j % 3],
                       yerr=[[r[4] - r[5]], [r[6] - r[4]]], capsize=3,
                       label=f"measured agent speedup, {int(r[1])} session(s)" if i == 0 else None)
        ax.axhline(1, color="#666", lw=.8, ls=":")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
        ax.set_ylabel("speedup vs base (×)")
        ax.set_title(f"Token speedup vs agent speedup (tool latency {tl_mid:g}s, 95% CI)")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig5_prediction.png"), dpi=160); plt.close(fig)

    # ---- Figure 6: prefix caching vs shared-context length (H5)
    ctxs = sorted({k[4] for k in cells})
    ctx_rows = []
    for cx in ctxs:
        for c in concs:
            for tl in tls:
                kb, kn = (BASE, "off", tl, c, cx), ("noprefix", "off", tl, c, cx)
                if kb in raw and kn in raw:
                    res = paired_ratio(raw[kn], raw[kb], kinds=AGENT)
                    if res:
                        put(findings, f"context.ctx{int(cx)}.c{int(c)}.tl{tl:g}.prefix_speedup", res)
                        findings[f"context.ctx{int(cx)}.c{int(c)}.tl{tl:g}.base_task_mean_s"] = cells[kb]["e2e"]
                        if tl == tl_mid:
                            ctx_rows.append((cx, c, *res[:3]))
    if ctx_rows:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        for j, c in enumerate(concs):
            rs = sorted([r for r in ctx_rows if r[1] == c])
            ax.errorbar([int(r[0]) for r in rs], [r[2] for r in rs],
                        yerr=[[r[2] - r[3] for r in rs], [r[4] - r[2] for r in rs]],
                        marker="o", capsize=3, color=colors[j % 4], label=f"{int(c)} session(s)")
        ax.axhline(1, color="#666", lw=.8, ls=":")
        ax.set_xlabel("shared context (tokens)"); ax.set_ylabel("agent speedup from prefix caching (×)")
        ax.set_title(f"Prefix caching vs context length (tool latency {tl_mid:g}s, 95% CI)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(figdir, "fig6_context_prefix.png"), dpi=160); plt.close(fig)

    # ---- mechanism evidence from vLLM /metrics (real runs only): why a config helped or not
    for (cfg, th, tl, c, cx), m in smet.items():
        if th != "off" or tl != tl_mid:
            continue
        for mk in ("prefix_hit_rate", "spec_accept_rate"):
            if m.get(mk) is not None:
                findings[f"mech.{cfg}.c{int(c)}.ctx{int(cx)}.{mk}"] = m[mk]

    # ---- long-tail tool latency (TraceLab): same mean, lognormal. Does the speedup survive?
    for (cfg, tl, c, cx), recs in tail.items():
        if cfg == BASE or (BASE, tl, c, cx) not in tail:
            continue
        k = f"tail.{cfg}.c{int(c)}.tl{tl:g}"
        put(findings, k + ".task_speedup", paired_ratio(tail[(BASE, tl, c, cx)], recs, kinds=AGENT))
        fixed = (BASE, "off", tl, c, cx), (cfg, "off", tl, c, cx)
        if all(x in raw for x in fixed):
            put(findings, k + ".fixed_task_speedup", paired_ratio(raw[fixed[0]], raw[fixed[1]], kinds=AGENT))
        lat = [x * a.time_scale for r in tail[(BASE, tl, c, cx)] for x in r.get("tool_latencies", [])]
        if lat:
            lat.sort()
            findings[f"tail.tool_latency_mean_s.tl{tl:g}"] = statistics.mean(lat)
            findings[f"tail.tool_latency_p95_s.tl{tl:g}"] = lat[int(0.95 * (len(lat) - 1))]
        ok = sorted(r["e2e_s"] for r in tail[(BASE, tl, c, cx)] if not r["error"] and r["kind"] in AGENT)
        if ok:
            findings[f"tail.base.c{int(c)}.tl{tl:g}.agent_task_p95_s"] = ok[int(0.95 * (len(ok) - 1))]

    # ---- results table (markdown) for the blog/paper appendix
    with open(os.path.join(rd, "results_table.md"), "w") as f:
        f.write("| config | thinking | tool lat. (s) | conc. | task p50 (s) | task p95 (s) | TTFT p50 (s) | "
                "first visible p50 (s) | success | LLM share | ctx | prefix hit | MTP accept | tool latency dist |\n"
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            ck = (r["server_cfg"], r["thinking"], round(r["tool_latency"] * a.time_scale, 6),
                  r["concurrency"], r["context"])
            if r["tool_dist"] != "fixed":
                ck = None
            sm = smet.get(ck, {}) if ck else {}
            f.write(f"| {r['server_cfg']} | {r['thinking']} | {r['tool_latency']:g} | {int(r['concurrency'])} | "
                    f"{r['task_p50_s']:.3f} | {r['task_p95_s']:.3f} | {r['ttft_p50_s']:.3f} | "
                    f"{r['first_visible_p50_s']:.3f} | {(cells[ck]['success'] if ck in cells else r['success_rate']):.0%} | {r['llm_share']:.0%} | "
                    f"{int(r['context'])} | {fmt_rate(sm.get('prefix_hit_rate'))} | {fmt_rate(sm.get('spec_accept_rate'))} | {r['tool_dist']} |\n")

    with open(os.path.join(rd, "findings.json"), "w") as f:
        json.dump(findings, f, indent=1, sort_keys=True)
    print(f"wrote {figdir}/fig1..fig6, {rd}/results_table.md, {rd}/findings.json ({len(findings)} keys)")
    print(f"Amdahl accounting check (in-sample) max error: {worst_err:.1%}")
    for c in concs:
        if f"predict.max_rel_error.c{int(c)}" in findings:
            print(f"prediction max error at {int(c)} session(s): token-speed only "
                  f"{findings[f'predict.max_rel_error.c{int(c)}']:.1%}, prefill+decode split "
                  f"{findings[f'predict.max_rel_error_decomposed.c{int(c)}']:.1%}")
    if failed:
        print("server configs that failed to start:", ", ".join(failed))
    return findings


if __name__ == "__main__":
    main()
