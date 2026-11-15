"""Deterministic mock tools with injectable latency.

Tools return fixed data so task success is checkable, and `latency_s` simulates
real-world API time (network, DB, sandbox). Sweeping latency_s is how we find
the point where faster inference stops mattering.

READ-ONLY for the optimisation loop (see program.md).
"""
from __future__ import annotations

import ast
import asyncio
import json
import operator

ORDERS = {
    "A12": {"item": "mechanical keyboard", "price_usd": 120, "status": "shipped"},
    "B7": {"item": "usb-c dock", "price_usd": 80, "status": "processing"},
    "C3": {"item": "27-inch monitor", "price_usd": 310, "status": "delivered"},
}
FX = {"USD:AUD": 1.5, "USD:EUR": 0.9, "USD:JPY": 150.0}
WEATHER = {
    "sydney": {"temp_c": 24, "condition": "sunny"},
    "melbourne": {"temp_c": 17, "condition": "rain"},
    "tokyo": {"temp_c": 12, "condition": "cloudy"},
}
DOCS = {
    "refund policy": "Refunds are accepted within 30 days of delivery.",
    "shipping time": "Standard shipping takes 5 business days.",
    "warranty": "All electronics carry a 2-year warranty.",
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "lookup_order", "description": "Get an order by id.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                       "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "convert_currency", "description": "Convert an amount between currencies.",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number"}, "from_ccy": {"type": "string"}, "to_ccy": {"type": "string"}},
            "required": ["amount", "from_ccy", "to_ccy"]}}},
    {"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "calculator", "description": "Evaluate an arithmetic expression.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "search_docs", "description": "Search the store policy docs.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
]

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg, ast.Pow: operator.pow}


def _safe_eval(expr: str) -> float:
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")
    return ev(ast.parse(expr, mode="eval"))


def _run(name: str, args: dict) -> dict:
    if name == "lookup_order":
        o = ORDERS.get(str(args.get("order_id", "")).upper())
        return o or {"error": "order not found"}
    if name == "convert_currency":
        key = f"{str(args.get('from_ccy', '')).upper()}:{str(args.get('to_ccy', '')).upper()}"
        if key not in FX:
            return {"error": f"no rate for {key}"}
        return {"result": round(float(args["amount"]) * FX[key], 2)}
    if name == "get_weather":
        return WEATHER.get(str(args.get("city", "")).lower(), {"error": "unknown city"})
    if name == "calculator":
        try:
            return {"result": round(_safe_eval(str(args.get("expression", ""))), 4)}
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}
    if name == "search_docs":
        q = str(args.get("query", "")).lower()
        for k, v in DOCS.items():
            if any(w in q for w in k.split()):
                return {"snippet": v}
        return {"snippet": "No matching document."}
    return {"error": f"unknown tool {name}"}


async def call_tool(name: str, args_json: str, latency_s: float) -> str:
    try:
        args = json.loads(args_json or "{}")
    except json.JSONDecodeError:
        args = {}
    if latency_s > 0:
        await asyncio.sleep(latency_s)
    return json.dumps(_run(name, args))
