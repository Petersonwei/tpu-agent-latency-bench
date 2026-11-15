"""Fixed workloads: single-turn, multi-turn, tool-calling agent.

Each task has a checkable `expect` substring so task success is measured, not
assumed. `plan` is used ONLY by the offline simulator to script tool calls.

READ-ONLY for the optimisation loop (see program.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Task:
    id: str
    kind: str                      # "single" | "multi" | "tool"
    turns: list[str]               # user messages, in order
    expect: str | None = None      # case-insensitive substring in final answer
    plan: list[tuple[str, dict]] = field(default_factory=list)  # sim only
    max_tokens: int = 256


SYSTEM = ("You are a concise assistant. Use tools when they are needed. "
          "Give the final answer in one or two sentences and include any numbers explicitly.")

SINGLE = [
    Task("s1", "single", ["What is the capital of Australia? Answer in one word."], "canberra", max_tokens=32),
    Task("s2", "single", ["What is 17 * 23? Reply with just the number."], "391", max_tokens=32),
    Task("s3", "single", ["Name the chemical symbol for gold."], "au", max_tokens=32),
    Task("s4", "single", ["Explain in three sentences why KV caching speeds up LLM decoding."], "cach", max_tokens=200),  # stem: "cache"/"caching" (2026-11-14: "caching" was graded wrong)
    Task("s5", "single", ["Write a haiku about tensor processing units."], None, max_tokens=80),
    Task("s6", "single", ["Summarise in two sentences: Prefill processes the whole prompt in parallel and is "
                          "compute-bound; decode emits one token at a time and is memory-bandwidth-bound."],
         "decod", max_tokens=120),  # stem: "decode"/"decoding"
]

MULTI = [
    Task("m1", "multi", ["Remember the number 42. Just say OK.",
                         "Now remember the word 'trillium'. Just say OK.",
                         "What number and word did I ask you to remember?"], "trillium", max_tokens=64),
    Task("m2", "multi", ["I'm planning a 3-day trip to Tokyo. Suggest one activity per day, briefly.",
                         "Swap day 2 for something indoors.",
                         "How many days is my trip? Reply with a number."], "3", max_tokens=200),
    Task("m3", "multi", ["My budget is 500 dollars.",
                         "I spend 120 on a keyboard.",
                         "Then 80 on a dock. How much budget is left? Reply with the number."], "300", max_tokens=64),
]

TOOL = [
    Task("t1", "tool", ["What is the status of order A12?"], "shipped",
         plan=[("lookup_order", {"order_id": "A12"})]),
    Task("t2", "tool", ["How much did order C3 cost in AUD?"], "465",
         plan=[("lookup_order", {"order_id": "C3"}),
               ("convert_currency", {"amount": 310, "from_ccy": "USD", "to_ccy": "AUD"})]),
    Task("t3", "tool", ["What is the combined USD price of orders A12 and B7, converted to EUR?"], "180",
         plan=[("lookup_order", {"order_id": "A12"}), ("lookup_order", {"order_id": "B7"}),
               ("calculator", {"expression": "120+80"}),
               ("convert_currency", {"amount": 200, "from_ccy": "USD", "to_ccy": "EUR"})]),
    Task("t4", "tool", ["Is it warmer in Sydney or Melbourne right now? Give both temperatures."], "24",
         plan=[("get_weather", {"city": "Sydney"}), ("get_weather", {"city": "Melbourne"})]),
    Task("t5", "tool", ["Order B7 arrived broken. What does the refund policy say, and what was the item?"], "30 days",
         plan=[("lookup_order", {"order_id": "B7"}), ("search_docs", {"query": "refund policy"})]),
    Task("t6", "tool", ["What is (1234 * 5.5) - 789? Use the calculator."], "5998",
         plan=[("calculator", {"expression": "1234*5.5-789"})]),
]

WORKLOADS = {"single": SINGLE, "multi": MULTI, "tool": TOOL}


def get_tasks(kind: str) -> list[Task]:
    if kind == "all":
        return SINGLE + MULTI + TOOL
    return list(WORKLOADS[kind])


def grade(task: Task, answer: str, error: str = "") -> bool:
    """Substring check on the final answer. Single source of truth for bench.py AND analyze.py, so a
    grading fix can be applied to already-collected runs (final_answer is stored in every record)."""
    return (not error) and bool((answer or "").strip()) and (
        task.expect is None or task.expect.lower() in answer.lower().replace(",", ""))
