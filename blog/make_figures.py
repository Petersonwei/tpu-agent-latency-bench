"""Blog figures, drawn only from findings.json (no hand-typed numbers).

    python blog/make_figures.py --findings results/tpu_20261114/results/findings.json

Palette: default categorical slots 1-3 (blue, orange, aqua), validated with the dataviz skill's
validate_palette.js (all checks pass; aqua is below 3:1 contrast, so every bar is value-labelled).
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax, title, ylabel):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def bar(ax, x, h, w, color, lo=None, hi=None, fmt="{:.2f}×"):
    ax.bar(x, h, w, color=color, edgecolor=SURF, linewidth=2)
    if lo is not None:
        ax.errorbar(x, h, yerr=[[h - lo], [hi - h]], fmt="none", ecolor=INK2, capsize=3, lw=1)
    top = hi if hi is not None else h
    ax.text(x, top + 0.04, fmt.format(h), ha="center", va="bottom", fontsize=8, color=INK)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--findings", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
    a = ap.parse_args(argv)
    f = json.load(open(a.findings))
    os.makedirs(a.out, exist_ok=True)
    ci = lambda k: (f[k], f[k + "_ci_lo"], f[k + "_ci_hi"])  # noqa: E731

    # 1. Where agent task time goes, by tool latency (baseline, 1 session)
    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=SURF)
    tls = ["0", "0.5", "2"]
    llm = [f[f"breakdown.tl{t}.llm_share"] for t in tls]
    xs = range(len(tls))
    ax.bar(xs, [100 * v for v in llm], 0.5, color=BLUE, edgecolor=SURF, linewidth=2, label="model (LLM) time")
    ax.bar(xs, [100 * (1 - v) for v in llm], 0.5, bottom=[100 * v for v in llm], color=ORANGE,
           edgecolor=SURF, linewidth=2, label="tools + harness")
    for x, v in zip(xs, llm):
        ax.text(x, 100 * v / 2, f"{v:.0%}", ha="center", va="center", fontsize=9, color="white")
    ax.set_xticks(list(xs)); ax.set_xticklabels([f"{t} s per tool call" for t in tls])
    ax.set_ylim(0, 100)
    style(ax, "Share of agent task time spent in the model", "% of task time")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "blog1_time_share.png"), dpi=180); plt.close(fig)

    # 2. What a tokens/s benchmark says vs what the agent gets (0.5 s tool latency)
    groups = [
        ("MTP-3 speculative decoding",
         (f["predict.mtp3.token_speedup"], f["predict.mtp3.token_speedup_ci_lo"], f["predict.mtp3.token_speedup_ci_hi"]),
         ci("predict.mtp3.c1.tl0.5.task_speedup"), ci("predict.mtp3.c8.tl0.5.task_speedup")),
        # prefix caching ON vs OFF: invert the noprefix/base ratios so >1 means caching helps
        ("Prefix caching (4k context)",
         tuple(1 / v for v in (f["predict.noprefix.token_speedup"], f["predict.noprefix.token_speedup_ci_hi"],
                               f["predict.noprefix.token_speedup_ci_lo"])),
         ci("context.ctx4000.c1.tl0.5.prefix_speedup"), ci("context.ctx4000.c8.tl0.5.prefix_speedup")),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.9), facecolor=SURF)
    w = 0.24
    for i, (_, tok, a1, a8) in enumerate(groups):
        bar(ax, i - w, tok[0], w, BLUE, tok[1], tok[2])
        bar(ax, i, a1[0], w, ORANGE, a1[1], a1[2])
        bar(ax, i + w, a8[0], w, AQUA, a8[1], a8[2])
    ax.axhline(1, color=INK2, lw=0.8, ls=":")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylim(0, max(g[3][2] for g in groups) * 1.18)
    handles = [Patch(color=c, label=l) for l, c in (("tokens/s benchmark", BLUE), ("agent tasks, 1 session", ORANGE),
                                                     ("agent tasks, 8 sessions", AQUA))]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    style(ax, "Speedup a benchmark reports vs speedup an agent gets", "speedup (×), 95% CI")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "blog2_benchmark_vs_agent.png"), dpi=180); plt.close(fig)

    # 3. Prefix caching gain grows with context length and concurrency
    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=SURF)
    ctxs = [4000, 16000]
    for c, col, mk in ((1, ORANGE, "o"), (8, AQUA, "s")):
        pts = [ci(f"context.ctx{x}.c{c}.tl0.5.prefix_speedup") for x in ctxs]
        ax.errorbar(ctxs, [p[0] for p in pts], yerr=[[p[0] - p[1] for p in pts], [p[2] - p[0] for p in pts]],
                    color=col, marker=mk, ms=8, lw=2, capsize=3, label=f"{c} session{'s' if c > 1 else ''}")
        for x, p in zip(ctxs, pts):
            ax.annotate(f"{p[0]:.1f}×", (x, p[0]), xytext=(10, 6), textcoords="offset points", fontsize=8, color=INK)
    ax.axhline(1, color=INK2, lw=0.8, ls=":")
    ax.set_xticks(ctxs); ax.set_xticklabels(["4k tokens", "16k tokens"]); ax.set_xlim(2000, 18500)
    ax.set_xlabel("shared context per agent", color=INK2, fontsize=9)
    style(ax, "Agent speedup from prefix caching", "speedup (×), 95% CI")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "blog3_prefix_context.png"), dpi=180); plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
