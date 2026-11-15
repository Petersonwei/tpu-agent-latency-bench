"""A tiny OpenAI/vLLM-compatible streaming server for testing the REAL client path offline.

It mimics what vLLM sends when serving with --enable-auto-tool-choice and --reasoning-parser:
  * `reasoning_content` deltas first (when thinking is enabled)
  * tool calls streamed in fragments: first chunk has id + name, later chunks carry argument pieces
  * a final usage-only chunk (choices == []) when stream_options.include_usage is set
  * GET /health
It answers by looking up the task (by its first user message) in harness/workloads.py and
following the task's scripted `plan`, so a correct client reaches 100% success.

    python tests/mock_vllm_server.py --port 8011
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
from workloads import get_tasks  # noqa: E402

TASKS = {t.turns[0]: t for t in get_tasks("all")}


def _chunk(delta: dict | None = None, finish: str | None = None, usage: dict | None = None) -> bytes:
    body = {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": int(time.time()),
            "model": "mock", "choices": [] if delta is None else
            [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        body["usage"] = usage
    return b"data: " + json.dumps(body).encode() + b"\n\n"


COUNTERS = {"prefix_cache_queries": 0.0, "prefix_cache_hits": 0.0,
            "spec_decode_num_accepted_tokens": 0.0, "spec_decode_num_draft_tokens": 0.0}
_LOCK = threading.Lock()


def _metrics_text() -> bytes:
    """Prometheus exposition like vLLM's /metrics (counters carry the _total suffix)."""
    lines = []
    for k, v in COUNTERS.items():
        lines += [f"# TYPE vllm:{k} counter", f'vllm:{k}_total{{model_name="mock"}} {v}']
    return ("\n".join(lines) + "\n").encode()


class Handler(BaseHTTPRequestHandler):
    token_delay = 0.002
    ttft_delay = 0.01

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path.rstrip("/").endswith("health"):
            self.send_response(200); self.end_headers()
        elif self.path.rstrip("/").endswith("metrics"):
            self.send_response(200); self.send_header("Content-Type", "text/plain"); self.end_headers()
            self.wfile.write(_metrics_text())
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if not self.path.endswith("/chat/completions"):
            self.send_response(404); self.end_headers(); return
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = req["messages"]
        first_user = next(m["content"] for m in msgs if m["role"] == "user")
        task = TASKS.get(first_user)
        thinking = (req.get("chat_template_kwargs") or {}).get("enable_thinking", True)
        done = 0
        for m in reversed(msgs):
            if m["role"] == "user":
                break
            done += m["role"] == "tool"
        n_user = sum(m["role"] == "user" for m in msgs)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        w = self.wfile.write
        time.sleep(self.ttft_delay)
        out = 0
        if thinking:
            for piece in ["Let me ", "think about ", "this step."]:
                w(_chunk({"role": "assistant", "reasoning_content": piece})); out += 3
                time.sleep(self.token_delay)
        if task and task.plan and done < len(task.plan) and req.get("tools"):
            name, args = task.plan[done]
            a = json.dumps(args)
            w(_chunk({"tool_calls": [{"index": 0, "id": f"call_{done}", "type": "function",
                                      "function": {"name": name, "arguments": ""}}]}))
            for i in range(0, len(a), 7):          # arguments arrive in fragments
                w(_chunk({"tool_calls": [{"index": 0, "function": {"arguments": a[i:i + 7]}}]}))
                time.sleep(self.token_delay); out += 2
            w(_chunk({}, finish="tool_calls"))
        else:
            last = task is None or n_user >= len(task.turns)
            text = f"The answer is {task.expect}." if (task and task.expect and last) else "OK, noted."
            for word in text.split(" "):
                w(_chunk({"content": word + " "})); out += 1
                time.sleep(self.token_delay)
            w(_chunk({}, finish="stop"))
        if (req.get("stream_options") or {}).get("include_usage"):
            pt = sum(len(json.dumps(m)) for m in msgs) // 4
            with _LOCK:  # pretend half the prompt hits the cache and 3 of 4 drafts are accepted
                COUNTERS["prefix_cache_queries"] += pt; COUNTERS["prefix_cache_hits"] += pt // 2
                COUNTERS["spec_decode_num_draft_tokens"] += 4 * out
                COUNTERS["spec_decode_num_accepted_tokens"] += 3 * out
            w(_chunk(None, usage={"prompt_tokens": pt, "completion_tokens": out, "total_tokens": pt + out}))
        w(b"data: [DONE]\n\n")


def serve(port: int) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8011)
    a = p.parse_args()
    serve(a.port)
    print(f"mock vLLM on http://127.0.0.1:{a.port}/v1  (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
