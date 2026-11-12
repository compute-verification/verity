"""Whether a `verity-vllm serve-openai` run numbers its requests by first row in the order it admitted them.

    python tools/admit_rows_check.py SERVE_DIR [--json OUT]

The advice numbers requests in order of their first row (step, then place in the step's batch), so under ``--admit
release-order`` that order must be the admission order (``requests.jsonl``'s ``admitted``, the order the engine added them).
Two sources hold each request's first row, and both are checked:

- ``advice_trace.jsonl``, the scheduler step trace the advice reads (`engine.step_trace`): each step's rows in the model
  runner's order, each window's open and close;
- ``requests.jsonl``'s committed rows (window, step, lo, ...), read back from the Commit, as the served rows are.

For each window: the requests whose first row it holds, and every pair of them whose first rows are in the other order than
their admissions (an inversion), with the requests named (user, admitted, release, rows). Also each step's layout: whether
a decode row follows a prefill row. Exit 0 when neither source has an inversion, 1 otherwise, 2 when a file is missing.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def trace_first_rows(trace: Iterable[Mapping[str, Any]]) -> tuple[dict[str, tuple[int, int, int]], dict[str, Any]]:
    """Each request's first row in the step trace, (window, step, place in the step), and the steps' layout."""
    recs = list(trace)
    opens = sorted((int(r["before_step"]), int(r["window"])) for r in recs if r.get("kind") == "window" and r.get("event") == "open")
    prompt = {r["request"]: int(r["prompt_len"]) for r in recs if r.get("kind") == "arrive"}

    def window_of(step: int) -> int:
        w = -1
        for before, k in opens:
            if before <= step:
                w = k
        return w

    first: dict[str, tuple[int, int, int]] = {}
    layout = {"steps": 0, "orders": set(), "decode_after_prefill": [], "mixed": 0}
    for r in sorted((r for r in recs if r.get("kind") == "step"), key=lambda r: r["step"]):
        s, kinds = int(r["step"]), []
        layout["steps"] += 1
        layout["orders"].add(r.get("order"))
        for i, (rid, start, n) in enumerate(r["rows"]):
            first.setdefault(rid, (window_of(s), s, i))
            kinds.append("decode" if int(n) == 1 and start is not None and int(start) >= prompt.get(rid, 1 << 62) else "prefill")
        if "decode" in kinds and "prefill" in kinds:
            layout["mixed"] += 1
        if "prefill" in kinds and "decode" in kinds[kinds.index("prefill"):]:
            layout["decode_after_prefill"].append(s)
    layout["orders"] = sorted(map(str, layout["orders"]))
    return first, layout


def committed_first_rows(requests: Iterable[Mapping[str, Any]]) -> dict[str, tuple[int, int, int]]:
    """Each request's first committed row, (window, window step, first token row)."""
    return {q["engine_id"]: min((int(w), int(s), int(lo)) for w, s, lo, *_ in q["rows"]) for q in requests if q.get("rows")}


def inversions(first: Mapping[str, tuple[int, int, int]], requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The requests numbered by first row against their admission order, by window."""
    by_id = {q["engine_id"]: q for q in requests}
    have = sorted((e for e in first if e in by_id), key=lambda e: first[e])
    out: dict[str, Any] = {"requests": len(have), "unscheduled": sorted(q.get("user") or q["engine_id"] for q in requests
                                                                       if q["engine_id"] not in first),
                           "numbering_is_admission": [by_id[e]["admitted"] for e in have] == sorted(by_id[e]["admitted"] for e in have),
                           "windows": {}}
    total = 0
    for w in sorted({first[e][0] for e in have}):
        ids = [e for e in have if first[e][0] == w]
        bad = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:] if by_id[a]["admitted"] > by_id[b]["admitted"]]
        total += len(bad)

        def named(e: str) -> dict[str, Any]:
            q = by_id[e]
            return {"user": q.get("user"), "admitted": q["admitted"], "release": q.get("release"), "prompt": len(q.get("prompt_ids") or ()),
                    "first_row": list(first[e])}
        out["windows"][str(w)] = {"requests": len(ids), "inversions": len(bad),
                                  "same_step": sum(first[a][1] == first[b][1] for a, b in bad),
                                  "examples": [[named(a), named(b)] for a, b in bad[:3]]}
    out["inversions"] = total
    return out


def check(serve_dir: str | Path) -> dict[str, Any]:
    d = Path(serve_dir)
    reqs = _lines(d / "requests.jsonl")
    first, layout = trace_first_rows(_lines(d / "advice_trace.jsonl"))
    serve = json.loads((d / "serve.json").read_text()) if (d / "serve.json").exists() else {}
    trace, committed = inversions(first, reqs), inversions(committed_first_rows(reqs), reqs)
    runs: dict[str, list[int]] = {}
    for q in sorted((q for q in reqs if q.get("release")), key=lambda q: q["admitted"]):
        runs.setdefault(str(q["release"][0]), []).append(int(q["release"][1]))
    return {"serve_dir": str(d), "admit": serve.get("admit"), "step_rows": serve.get("step_rows"),
            "flash_attn_version": ((serve.get("engine") or {}).get("effective") or {}).get("flash_attn_version"), "requests": len(reqs),
            "released": sum(map(len, runs.values())), "admitted_in_release_order": all(ns == sorted(ns) for ns in runs.values()),
            "layout": layout, "trace": trace, "committed": committed,
            "passed": trace["inversions"] == 0 and committed["inversions"] == 0 and trace["numbering_is_admission"]
            and committed["numbering_is_admission"]}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("serve_dir")
    ap.add_argument("--json", help="also write the report here")
    a = ap.parse_args(argv)
    try:
        rep = check(a.serve_dir)
    except FileNotFoundError as e:
        print(f"admit-rows-check: {e}", file=sys.stderr)
        return 2
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1) + "\n")
    for src in ("trace", "committed"):
        r = rep[src]
        print(f"{src}: {r['requests']} requests in {len(r['windows'])} windows, {r['inversions']} inversions, numbering is admission "
              f"order: {r['numbering_is_admission']}")
        for w, x in r["windows"].items():
            if x["inversions"]:
                print(f"  w{w}: {x['inversions']} inversions ({x['same_step']} in one step), e.g. {json.dumps(x['examples'][0])}")
    print(f"layout: {json.dumps(rep['layout'])}")
    keys = ("admit", "step_rows", "flash_attn_version", "requests", "released", "admitted_in_release_order")
    print(f"ADMIT-ROWS {'PASS' if rep['passed'] else 'FAIL'} {json.dumps({k: rep[k] for k in keys})}")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
