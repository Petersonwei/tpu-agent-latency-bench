#!/usr/bin/env bash
# L0 money gate: everything that can be verified without a TPU. Must pass before creating a VM.
#   bash scripts/preflight_local.sh          (~6 min; uses .venv)
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -x .venv/bin/python ]] || { uv venv --python 3.12 .venv -q && uv pip install -q --python .venv/bin/python -r requirements-dev.txt; }
export PATH="$PWD/.venv/bin:$PATH"
step(){ echo; echo "=== $*"; }

step "1/5 unit + integration tests (real client vs mock vLLM stream)"
python -m pytest -q

step "2/5 shell syntax"
for f in scripts/*.sh; do bash -n "$f"; done; echo ok

step "3/5 smoke gate against mock server"
MOCK=1 PORT=${MOCK_PORT:-8017} bash scripts/smoke_tpu.sh | grep -E "gate|GATE"

step "4/5 full sweep rehearsal (simulator, 48 grid + 4 long-context + 2 long-tail cells) + analysis"
rm -rf results/rehearsal
SIM=1 SIM_SCALE=10 PY=python OUTDIR=results/rehearsal bash scripts/sweep_tpu.sh > results/rehearsal.log 2>&1 \
  || { tail -20 results/rehearsal.log; exit 1; }
rows=$(($(wc -l < results/rehearsal/sweep.tsv) - 1)); echo "sweep rows: $rows"; [[ $rows -eq 54 ]]
python analysis/analyze.py --results-dir results/rehearsal --time-scale 10

step "5/5 paper claims resolve from findings"
python paper/check_claims.py --findings results/rehearsal/findings.json --out results/rehearsal/paper_rendered.md

echo; echo "PREFLIGHT PASSED — safe to spend TPU time (see LAB.md)"
