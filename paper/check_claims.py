"""Render paper.md with numbers from findings.json and refuse unbacked numbers.

    python paper/check_claims.py --findings results/findings.json
Placeholders:  {{thinking.task_slowdown|.1f}}  {{breakdown.tl2.llm_share|%}}  {{amdahl.max_relative_error|.1%}}
Exit 1 on unknown keys, or on bare decimal numbers inside the Results/Discussion sections.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PH = re.compile(r"\{\{\s*([\w.\-]+)\s*(?:\|\s*([^}]+?))?\s*\}\}")
GUARDED = ("## 5", "## 6")           # Results, Discussion
# decimals (3.14, 3.14x, 3.14s) and integer percentages (69%) typed by hand
BARE_NUM = re.compile(r"(?<![\w{.])(?:\d+\.\d+(?!\d)|\d+(?=\s?%))")
COMMENT = re.compile(r"<!--.*?-->", re.S)
# experiment *settings* (not results) that may be typed: tool latencies etc.
DEFAULT_ALLOW = "0.5,2.0,1.0,95"  # settings + the CI level


def fmt(v, spec):
    if spec in (None, ""):
        return f"{v:.3g}" if isinstance(v, float) else str(v)
    if spec == "%":
        return f"{v:.0%}"
    return format(v, spec)


def render(text, findings):
    """Substitute placeholders outside HTML comments; return (rendered, missing_keys)."""
    missing = []
    parts, last = [], 0
    for m in COMMENT.finditer(text):
        parts.append((text[last:m.start()], True)); parts.append((m.group(0), False)); last = m.end()
    parts.append((text[last:], True))

    def sub(m):
        k, spec = m.group(1), m.group(2)
        if k not in findings:
            missing.append(k)
            return f"⟦MISSING {k}⟧"
        return fmt(findings[k], spec)
    return "".join(PH.sub(sub, t) if live else t for t, live in parts), missing


def bare_numbers(text, allow=()):
    """Decimal numbers typed by hand inside guarded sections (placeholders, headings, comments excluded)."""
    hits, guarded = [], False
    for line in COMMENT.sub("", text).splitlines():
        if line.startswith("## "):
            guarded = line.startswith(GUARDED)
        if guarded and not line.lstrip().startswith(("#", "|", "!")):
            stripped = PH.sub("", line)
            hits += [(line.strip()[:80], n) for n in BARE_NUM.findall(stripped) if n not in allow]
    return hits


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--findings", default="results/findings.json")
    ap.add_argument("--paper", default=os.path.join(HERE, "paper.md"))
    ap.add_argument("--out", default=os.path.join(HERE, "paper_rendered.md"))
    ap.add_argument("--allow", default=DEFAULT_ALLOW, help="comma list of setting values allowed as typed numbers")
    a = ap.parse_args(argv)
    findings = json.load(open(a.findings))
    src = open(a.paper).read()
    out, missing = render(src, findings)
    bare = bare_numbers(src, {x.strip() for x in a.allow.split(",") if x.strip()})
    open(a.out, "w").write(out)
    n = len(PH.findall(COMMENT.sub("", src)))
    print(f"placeholders: {n}, missing: {len(missing)}, bare numbers in results: {len(bare)}")
    for k in sorted(set(missing)):
        print("  MISSING", k)
    for line, num in bare:
        print(f"  BARE {num!r} in: {line}")
    ok = not missing and not bare
    print("CLAIMS OK" if ok else "CLAIMS FAILED", "->", a.out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
