# paper/program.md — the writing loop (L4)

`follow @paper/program.md`. Produces a short workshop-style paper / long-form blog from the same
source, with **zero hand-typed result numbers**.

## Editable
`paper/paper.md` only.

## Fixed evaluator
- `python paper/check_claims.py --findings results/findings.json` must exit 0. It
  - resolves every `{{key}}` / `{{key|fmt}}` from `findings.json` into `paper/paper_rendered.md`
  - fails on unknown keys
  - fails on bare decimal numbers in the Results/Discussion sections (numbers must be placeholders)
- The rubric below, scored 0–2 per item by a *fresh* agent/session that has not seen the drafts.

## Rubric (max 16)
1. The research question is stated in the first paragraph and answered in the abstract.
2. Each RQ has one figure or table and one sentence of answer.
3. The Amdahl framing is explained so a developer with no systems background follows it.
4. Limitations are honest (single chip, small model, synthetic tools, simulator for rehearsal).
5. Related work positions the project against agent-serving and latency-perception papers in `research/literature-review.md`.
6. Reproducibility: exact versions (`results/env_versions.txt`), commands, and repo link.
7. The TPU/vLLM-TorchTPU specifics (setup gotchas, compile cache) are concrete and useful.
8. The human-responsiveness implication (Tan et al. CHI'26) is stated carefully as a hypothesis, not a result.

## Loop
1. Read `paper/revisions.tsv` (columns: `commit	rubric	checker	description`).
2. Improve one section toward the weakest rubric item. Commit.
3. Run the checker. If it fails → fix or `git reset --hard HEAD~1`.
4. Score the rubric (fresh context). Keep if the score improved, else revert. Log the row.
Stop at rubric ≥ 14 or after 10 iterations, then hand to a human for review.
