"""Offline tests: run with  .venv/bin/python -m pytest -q
They prove the harness is correct BEFORE any TPU money is spent."""
from __future__ import annotations

import asyncio
import json
import os
import socket
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import bench  # noqa: E402
from tools import _run  # noqa: E402
from workloads import get_tasks  # noqa: E402


def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def test_tool_plans_produce_expected_answers():
    """Every tool task's ground truth must be derivable from its tools (no impossible tasks)."""
    for t in get_tasks("tool"):
        outputs = json.dumps([_run(name, args) for name, args in t.plan])
        assert t.expect.lower() in outputs.lower().replace(".0", ""), (t.id, outputs)


def test_task_ids_unique_and_expects_set():
    ids = [t.id for t in get_tasks("all")]
    assert len(ids) == len(set(ids))
    assert all(t.expect for t in get_tasks("tool"))


def test_percentile():
    assert bench.pct([1, 2, 3, 4], 50) == 2.5
    assert bench.pct([5], 95) == 5


def _run_bench(argv):
    a = bench.parse_args(argv)
    return asyncio.run(bench.main_async(a))


def test_sim_end_to_end(tmp_path, capsys):
    out = tmp_path / "sim.jsonl"
    _run_bench(["--sim", "--repeats", "1", "--warmup", "0", "--tool-latency", "0.01",
                "--sim-tps", "2000", "--out", str(out)])
    text = capsys.readouterr().out
    assert "success_rate: 1.0000" in text
    lines = out.read_text().splitlines()
    assert len(lines) == 1 + len(get_tasks("all"))


def test_real_client_against_mock_vllm(tmp_path, capsys):
    """Exercises OpenAIBackend: streamed tool-call fragments, reasoning deltas, usage chunk."""
    import mock_vllm_server
    port = _free_port()
    srv = mock_vllm_server.serve(port)
    try:
        out = tmp_path / "mock.jsonl"
        _run_bench(["--base-url", f"http://127.0.0.1:{port}/v1", "--model", "mock", "--repeats", "1",
                    "--warmup", "0", "--thinking", "on", "--tool-latency", "0.01",
                    "--context-tokens", "300", "--min-success", "1.0", "--out", str(out)])
        text = capsys.readouterr().out
        assert "success_rate: 1.0000" in text and "score: inf" not in text
        recs = [json.loads(l) for l in out.read_text().splitlines()[1:]]
        tool_recs = [r for r in recs if r["kind"] == "tool"]
        assert sum(r["tool_calls"] for r in tool_recs) == sum(len(t.plan) for t in get_tasks("tool"))
        # with thinking on, the first visible token comes after the first (reasoning) token
        assert all(a >= t for r in recs for t, a in zip(r["ttft_s"], r["ttfa_s"]))
        assert all(r["prompt_tokens"] > 0 and r["completion_tokens"] > 0 for r in recs)
    finally:
        srv.shutdown()


def test_score_is_inf_below_success_floor(capsys):
    import mock_vllm_server
    port = _free_port()
    srv = mock_vllm_server.serve(port)
    try:
        # an impossible floor (>1) must invalidate the run
        _run_bench(["--base-url", f"http://127.0.0.1:{port}/v1", "--model", "mock", "--repeats", "1",
                    "--warmup", "0", "--workload", "single", "--min-success", "1.01"])
        assert "score: inf" in capsys.readouterr().out
    finally:
        srv.shutdown()


def test_server_down_is_recorded_not_crashing(capsys):
    _run_bench(["--base-url", f"http://127.0.0.1:{_free_port()}/v1", "--model", "x", "--repeats", "1",
                "--warmup", "0", "--workload", "single"])
    out = capsys.readouterr().out
    assert "score: inf" in out and "success_rate: 0.0000" in out


def test_claim_checker_blocks_hand_typed_results(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "paper"))
    import check_claims
    f = tmp_path / "f.json"; f.write_text(json.dumps({"a.b": 1.234}))
    good = tmp_path / "good.md"
    good.write_text("## 5. Results\n### 5.1 x\nSped up {{a.b|.2f}}x on Qwen3.5-4B at 0.5 s.\n<!-- {{nope}} 9.9 -->\n")
    bad = tmp_path / "bad.md"; bad.write_text("## 5. Results\nIt was 3.14x faster, 69% LLM, {{missing.key}}.\n")
    assert check_claims.main(["--findings", str(f), "--paper", str(good), "--out", str(tmp_path / "g")]) == 0
    assert "1.23x" in (tmp_path / "g").read_text()
    assert check_claims.main(["--findings", str(f), "--paper", str(bad), "--out", str(tmp_path / "b")]) == 1


def test_server_metrics_scraped_from_mock(tmp_path, capsys):
    """vLLM /metrics counters (prefix-cache hits, spec-decode acceptance) land in the JSONL header."""
    import mock_vllm_server
    port = _free_port()
    srv = mock_vllm_server.serve(port)
    try:
        out = tmp_path / "m.jsonl"
        _run_bench(["--base-url", f"http://127.0.0.1:{port}/v1", "--model", "mock", "--repeats", "1",
                    "--warmup", "0", "--workload", "tool", "--tool-latency", "0", "--out", str(out)])
        head = json.loads(out.read_text().splitlines()[0])
        sm = head["server_metrics"]
        assert 0.45 < sm["prefix_hit_rate"] <= 0.5 and sm["spec_accept_rate"] == 0.75
        recs = [json.loads(l) for l in out.read_text().splitlines()[1:]]
        assert all(abs(r["prefill_s"] + r["decode_s"] - r["llm_s"]) < 1e-6 for r in recs)
    finally:
        srv.shutdown()


def test_prometheus_parser_handles_total_suffix_and_labels():
    txt = ('# HELP x\nvllm:prefix_cache_hits_total{model_name="a"} 3.0\n'
           'vllm:prefix_cache_hits_total{model_name="b"} 2.0\nvllm:prefix_cache_queries 10\nother_metric 99\n')
    m = bench.parse_prometheus(txt)
    assert m["prefix_cache_hits"] == 5.0 and m["prefix_cache_queries"] == 10.0
    assert bench.metrics_delta(None, m) == {}


def test_bootstrap_ci_is_paired_and_brackets_estimate():
    sys.path.insert(0, os.path.join(ROOT, "analysis"))
    import analyze
    def rec(t, e2e, kind="tool"):
        return {"task_id": t, "kind": kind, "error": "", "e2e_s": e2e, "completion_tokens": 10, "llm_calls": 1}
    base = [rec(f"t{i}", 2.0 + i) for i in range(8)]
    fast = [rec(f"t{i}", (2.0 + i) / 1.5) for i in range(8)]     # every task exactly 1.5x faster
    est, lo, hi, n = analyze.paired_ratio(base, fast)
    assert n == 8 and abs(est - 1.5) < 1e-9 and abs(lo - 1.5) < 1e-9 and abs(hi - 1.5) < 1e-9
    noisy = [rec(f"t{i}", (2.0 + i) * (0.6 if i % 2 else 1.2)) for i in range(8)]
    est, lo, hi, _ = analyze.paired_ratio(base, noisy)
    assert lo < est < hi and hi - lo > 0.1                         # heterogeneous effect -> wide CI


def test_sweep_falls_back_to_ngram_when_mtp_fails(tmp_path):
    """Real (non-SIM) path with a fake `vllm` that never becomes healthy for MTP configs."""
    import subprocess
    fake = tmp_path / "bin"; fake.mkdir()
    (fake / "vllm").write_text("#!/usr/bin/env bash\necho \"ERROR: fake vllm cannot start $*\"; exit 1\n")
    (fake / "vllm").chmod(0o755)
    (fake / "sleep").write_text("#!/usr/bin/env bash\nexit 0\n"); (fake / "sleep").chmod(0o755)
    env = dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}", OUTDIR=str(tmp_path / "r"),
               PORT=str(_free_port()), THINKS="off", TOOL_LATS="0", CONCS="1", LONG_CTX="")
    p = subprocess.run(["bash", "scripts/sweep_tpu.sh"], cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=600)
    log = (tmp_path / "r" / "logs" / "failed_configs.txt").read_text().splitlines()
    names = [l.split("\t")[0] for l in log]
    assert names == ["base", "noprefix", "mtp1", "mtp3", "ngram3"], (names, p.stdout[-500:])


FAKE_GCLOUD = r'''#!/usr/bin/env bash
echo "$*" >> "$GCLOUD_LOG"
case "$*" in
  "config get-value project"*) echo proj-x;;
  "projects describe"*) echo 123;;
  "billing projects describe"*) echo billingAccounts/AAA-BBB;;
  "billing accounts describe"*) echo AUD;;
  "services list"*) echo billingbudgets.googleapis.com;;
  "billing budgets list"*) cat "$BUDGETS_JSON";;
  *) echo "ok";;
esac
'''


def _tpu_create_with_fake_gcloud(tmp_path, budgets):
    import subprocess
    b = tmp_path / "bin"; b.mkdir(exist_ok=True)
    (b / "gcloud").write_text(FAKE_GCLOUD); (b / "gcloud").chmod(0o755)
    bj = tmp_path / "budgets.json"; bj.write_text(json.dumps(budgets))
    log = tmp_path / "gcloud.log"; log.write_text("")
    env = dict(os.environ, PATH=f"{b}:{os.environ['PATH']}", GCLOUD_LOG=str(log), BUDGETS_JSON=str(bj))
    p = subprocess.run(["bash", "scripts/tpu_create.sh"], cwd=ROOT, env=env, capture_output=True, text=True)
    return p, log.read_text()


def test_tpu_create_refuses_without_budget_guard(tmp_path):
    p, log = _tpu_create_with_fake_gcloud(tmp_path, [])
    assert p.returncode == 3 and "REFUSING" in p.stdout
    assert "queued-resources create" not in log and "tpu-vm create" not in log


def test_tpu_create_refuses_if_budget_counts_credits(tmp_path):
    """A budget that includes credits never fires while sprint credits last -> not a guard."""
    b = [{"displayName": "x", "budgetFilter": {"projects": ["projects/123"]},
          "amount": {"specifiedAmount": {"units": "38", "currencyCode": "AUD"}},
          "thresholdRules": [{"thresholdPercent": 0.5}]}]
    p, log = _tpu_create_with_fake_gcloud(tmp_path, b)
    assert p.returncode == 3 and "queued-resources create" not in log


def test_tpu_create_proceeds_with_budget_guard(tmp_path):
    b = [{"displayName": "TPU sprint guard",
          "budgetFilter": {"projects": ["projects/123"], "creditTypesTreatment": "EXCLUDE_ALL_CREDITS"},
          "amount": {"specifiedAmount": {"units": "38", "currencyCode": "AUD"}},
          "thresholdRules": [{"thresholdPercent": 0.5}, {"thresholdPercent": 1.0}]}]
    p, log = _tpu_create_with_fake_gcloud(tmp_path, b)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "BUDGET GUARD: OK" in p.stdout and "queued-resources create" in log
    assert "--provisioning-model=flex-start" in log


def test_lognormal_tool_latency_is_paired_and_heavy_tailed():
    xs = [bench.tool_latency_for(f"t{i}", r, n, 0.5, "lognormal") for i in range(60) for r in range(4) for n in range(5)]
    xs.sort()
    assert 0.4 < sum(xs) / len(xs) < 0.6                    # same mean as the fixed 0.5 s cells
    assert xs[int(0.95 * len(xs))] > 2.5 * 0.5               # long tail: p95 well above the mean
    # the same (task, repeat, call) gets the same latency under every server config -> paired
    assert bench.tool_latency_for("t2", 1, 0, 0.5, "lognormal") == bench.tool_latency_for("t2", 1, 0, 0.5, "lognormal")
    assert bench.tool_latency_for("t2", 1, 0, 0.5, "fixed") == 0.5


def test_staggered_start_keeps_measurements_valid(tmp_path, capsys):
    out = tmp_path / "st.jsonl"
    _run_bench(["--sim", "--repeats", "1", "--warmup", "0", "--concurrency", "4", "--stagger", "3.0",
                "--sim-ttft", "0.01", "--tool-latency", "0.01", "--tool-dist", "lognormal",
                "--sim-tps", "2000", "--out", str(out)])
    assert "success_rate: 1.0000" in capsys.readouterr().out
    lines = out.read_text().splitlines()
    recs = [json.loads(l) for l in lines[1:]]
    # start jitter (up to 3 s) happens before a task starts, so it is NOT counted in task time...
    assert all(r["e2e_s"] < 1.0 for r in recs)
    # ...but it did happen: the batch wall clock includes it
    assert json.loads(lines[0])["summary"]["all"]["wall_s"] > 1.0
    assert all(len(r["tool_latencies"]) == r["tool_calls"] for r in recs)


def test_grader_accepts_word_forms():
    """2026-11-14 TPU run: 'KV caching ...' and '... decoding ...' were graded wrong by exact stems."""
    from workloads import grade
    t = {x.id: x for x in get_tasks("all")}
    assert grade(t["s4"], "KV caching speeds up decoding by storing keys and values.")
    assert grade(t["s6"], "Prefill is compute-bound, whereas decoding is memory-bandwidth-bound.")
    assert not grade(t["s1"], "Sydney")
    assert not grade(t["m3"], "200")
