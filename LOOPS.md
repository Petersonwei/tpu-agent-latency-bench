# LOOPS.md — how coding agents run this project

"Loop engineering" (Tunix Academy L200, karpathy/autoresearch) means: **one goal, one editable
surface, one machine-checkable score, a fixed evaluator the agent cannot touch, and a log a human
audits.** This project uses five such loops. Each has a contract file the agent is pointed at.

```
            zero cost                          costs money (TPU)                     zero cost
 ┌──────────────────────────┐   ┌────────────────────────────────────────────┐   ┌─────────────────────┐
 │ L0 PREFLIGHT (gate)      │──▶│ L1 LAB RUN  ──▶  L2 AUTOSERVE (optional)   │──▶│ L3 STUDY → L4 PAPER │
 │ tests + sim rehearsal +  │   │ smoke → sweep → analyze │ program.md loop  │   │ findings.json is the│
 │ analysis + claim check   │   │ (LAB.md)                │ on serve_config  │   │ only source of nums │
 └──────────────────────────┘   └────────────────────────────────────────────┘   └─────────────────────┘
           ▲                                                                             │
           └──────────── L5 RELATED WORK feeds hypotheses ───────── ◀────────┘
```

| Loop | Contract | Agent may edit | Fixed evaluator (read-only) | Score / stop rule | Log a human audits |
|---|---|---|---|---|---|
| **L0 Preflight** | `scripts/preflight_local.sh` | harness, analysis, paper code | `tests/`, rehearsal | all tests pass, rehearsal exit 0, claims resolve | test output |
| **L1 Lab run** | `LAB.md` | nothing (runs scripts) | `scripts/*.sh`, harness | smoke gate: success ≥ 0.8, tool calls parsed; then sweep completes | `results/sweep.tsv`, server logs |
| **L2 Autoserve** | `program.md` | `serve_config.env` only | `run_experiment.sh`, harness | `score` = task p50 ↓, success floor | `results/results.tsv` (every row) |
| **L3 Study** | `study.md` | `research/lab-notebook.md`, new sweep *specs* | `analysis/analyze.py` | each hypothesis gets a verdict backed by a findings key | lab notebook |
| **L4 Paper** | `paper/program.md` | `paper/paper.md` | `paper/check_claims.py` + rubric | checker passes; rubric score ↑ | `paper/revisions.tsv` |
| **L5 Related work** | by hand | `research/related-work.md` | — | new works added with a link to the source | — |

## Rules shared by all loops (from the deck's lessons)
1. **The evaluator lives outside the editable file.** The agent never edits the scorer, the
   workloads or the claim checker during a loop. (In the Tunix GRPO case study the agent edited
   reward code it was told not to touch. The fix was moving it out of the editable file.)
2. **One number decides keep/revert.** Anything else is logged, not optimised.
3. **Every reported number has a provenance key** in `findings.json` (or a row in a TSV).
   The paper uses `{{key}}` placeholders, never typed numbers. This is the TPU Agent Suite's "real TPU runs only" principle.
4. **Read the whole log**, including discarded rows. Noise, crashes and rule breaks hide there.
5. **Money gate:** L1/L2 start only after L0 passes, with a VM time budget, and the VM is deleted at the end.

## Which agent runs what
- L0, L3, L4: Claude Code on the laptop (no TPU needed).
- L1, L2: Claude Code / Antigravity / Codex **on the TPU VM** (`follow @LAB.md`, `follow @program.md`).
  Running the same `program.md` in two harnesses and comparing logs is itself a mini-result (deck "exercise 4").
- L5: update `research/related-work.md` by hand.
