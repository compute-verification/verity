"""Render run A11's per-kind table and its gap lines (Markdown) from `a11_kinds.py`'s table and `a11.py`'s summaries.

    python examples/one_stage/a11_table.py --kinds kinds.json [--summary RUN_ID=<run>/out ...]

A row per computing kind (Const32 families, one gate each, left out): its root Calls, gates and share of the step's
computing gates, the unit M0 was given, its ANDs and k_log, the lowering outcome, and from the audits whether its units were
staged from the step's values, proved and held by Lean, and in which run. A kind audited in several runs is shown from the
one that got furthest (Lean's verdict over a staging over nothing), the later on a tie. A kind the probe left undecided
that an audit staged (`a11.py --try`) lowers: the stager composed its circuit. A gap line names what keeps a kind from Lean
acceptance.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


STAGED = re.compile(r"^A(?:11|8) staged (\S+): (\d+) units of .+? \(cap \d+\), (\d+) ANDs, k_log (\d+), unit twin \w+, "
                    r"Call twin (\{.*\}), (\{'build'.*\})$")


def load_run(path: Path) -> dict:
    """An audit's summary.json (the file, or a run's ``out`` directory holding ``a11/summary.json``), else, for a run that
    stopped after its audit record, one rebuilt from ``a11/audit.json`` and ``a11.log``'s staging lines, its negatives not run.
    A run recorded while this run was named A8 holds ``a8/`` and ``a8.log`` instead."""
    if path.is_file():
        return json.loads(path.read_text())
    name = "a11" if (path / "a11").is_dir() else "a8"
    if (path / name / "summary.json").exists():
        return json.loads((path / name / "summary.json").read_text())
    a = json.loads((path / name / "audit.json").read_text())
    values, members = {}, []
    for line in (path / f"{name}.log").read_text().splitlines():
        m = STAGED.match(line)
        if m:
            f, n = m[1], int(m[2])
            values[f] = {"staged": True, "n": n, "twin": ast.literal_eval(m[5])}
            members.append({"name": f, "n": n, "ands": int(m[3]), "k_log": int(m[4])})
    v = a.get("verdicts") or {}
    return {"values": values, "members": members, "draw_counts": a.get("draw_counts"), "negatives": {}, "negatives_run": False,
            "audit": {"accepted": a.get("accepted"), "verdicts": {x: y.get("accepted") for x, y in v.items()},
                      "why": {x: y.get("why") for x, y in v.items() if y.get("accepted") is not True}}}


def short(uid: str | None, n: int = 60) -> str:
    if not uid:
        return "-"
    return uid if len(uid) <= n else uid[:n - 1] + "…"


def last(r: dict) -> dict:
    return (r.get("attempts") or [{}])[-1]


def verdicts(s: dict | None) -> dict:
    return ((s or {}).get("audit") or {}).get("verdicts") or {}


def member(s: dict | None, f: str) -> dict | None:
    return next((m for m in (s or {}).get("members", []) if m["name"] == f), None)


def pick(f: str, runs: list[tuple[str, dict]]) -> tuple[str | None, dict | None]:
    """The run whose audit got furthest with kind f, and its summary."""
    best = (0, None, None)
    for rid, s in runs:
        v = (s.get("values") or {}).get(f)
        if v is None:
            continue
        lv = verdicts(s).get(f"lean-verify/{f}")
        rank = 4 if lv is True else 3 if lv is False else 2 if member(s, f) else 1
        if rank >= best[0]:
            best = (rank, rid, s)
    return best[1], best[2]


def outcome(f: str, r: dict, s: dict | None) -> str:
    if r.get("outcome") != "lowers" and f in (s or {}).get("tried", []) and member(s, f):
        return "lowers"
    return r.get("outcome") or "not probed"


def size_of(r: dict) -> str:
    """The unit M0 was given (else the fold chain's smallest the circuit's row shape admits), its size and widest row."""
    chain = r.get("fold_chain") or []
    su = r.get("shape_unit") or {}
    if r.get("unit"):
        uid, g, n = r["unit"], r.get("unit_gates"), r.get("units_per_call")
    else:
        fits = [c for c in chain if c["id"] != su.get("unit")]
        if not fits:
            return "unit -"
        uid, g, n = fits[-1]["id"], fits[-1]["gates"], fits[-1]["per_call"]
    rows = next((c.get("rows") for c in chain if c["id"] == uid), None) or []
    widest = f", widest row {max(b for _r, b in rows)} bits" if rows else ""
    s = f"unit {short(uid, 80)} ({g} word gates, {n} per Call{widest})"
    if su and su.get("unit") != uid:
        bits = max(b for _r, b in su.get("row_bits") or [["", 0]])
        s += f"; the smaller {short(su['unit'], 60)} ({su.get('gates')} gates) has {bits}-bit rows, inside one SHA-512 block, which M0 refuses"
    return s


def world(r: dict) -> int:
    """The ranks whose values a kind's unit reads (its Definition's ``WORLD``), 1 for a rank-local kind."""
    ids = [r.get("unit") or ""] + [c["id"] for c in r.get("fold_chain") or []]
    m = next((m for m in (re.search(r"\bWORLD=(\d+)", i) for i in ids) if m), None)
    return int(m[1]) if m else 1


def gap(f: str, r: dict, s: dict | None) -> str | None:
    """Why kind f has no Lean-accepted unit, or None."""
    a = last(r)
    size = size_of(r)
    if world(r) > 1:
        size += f"; it reads {world(r)} ranks' values and A11's slice holds rank 0's only"
    share = f"{100 * r['share']:.3f}% of the step"
    o = outcome(f, r, s)
    if o == "no-piece":
        sub = r.get("sub_unit") or {}
        tail = f"; largest sub-unit with every piece: {short(sub.get('unit'), 60)} ({sub.get('outcome')})" if sub else ""
        return f"{f}: no C-Flock piece for {', '.join(r.get('missing') or [a.get('piece', '?')])}; {size}; {share}{tail}"
    if o == "not probed":
        return f"{f}: not probed; {size}; {share}"
    if o != "lowers":
        why = a.get("why") or r.get("why") or o
        return f"{f}: does not lower ({o}: {str(why)[:160]}); {size}; {share}"
    if s is None:
        return f"{f}: lowers, not staged in an audit; {size}; {share}"
    v = (s.get("values") or {}).get(f)
    if not v.get("staged"):
        return f"{f}: lowers, no values in the slice ({str(v.get('why'))[:200]}); {size}; {share}"
    lean = verdicts(s).get(f"lean-verify/{f}")
    if lean is None and (s.get("draw_counts") or {}).get(f, 0) == 0:
        return None
    if lean is not True:
        return f"{f}: staged but Lean did not accept ({(s.get('audit') or {}).get('why', {}).get(f'lean-verify/{f}')}); {size}; {share}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", required=True)
    ap.add_argument("--summary", action="append", default=[],
                    help="RUN_ID=PATH, an audit's summary.json or its run's out directory (repeatable, oldest first)")
    a = ap.parse_args()
    t = json.loads(Path(a.kinds).read_text())
    runs = []
    for x in a.summary:
        rid, _, path = x.rpartition("=")
        runs.append((rid or path, load_run(Path(path))))
    kinds = {f: r for f, r in t["kinds"].items() if r.get("gates", 0) > 0 and not f.startswith("Const")}
    chosen = {f: pick(f, runs) for f in kinds}
    print("| kind | calls | gates | share | unit | ANDs | k_log | outcome | staged (twin) | M0 | Lean | run |")
    print("|---|---:|---:|---:|---|---:|---:|---|---|---|---|---|")
    for f, r in sorted(kinds.items(), key=lambda kv: -kv[1]["gates"]):
        rid, s = chosen[f]
        a_, m = last(r), member(s, f)
        v = ((s or {}).get("values") or {}).get(f) or {}
        twin = "-"
        if v.get("staged"):
            tw = v.get("twin") or {}
            twin = f"{m['n'] if m else v.get('n')} ({'agree' if tw.get('agree') else tw.get('agree')}, {tw.get('compared')} words)"
        elif v:
            twin = "no values"
        mv, lv = verdicts(s).get(f"m0-verifier/{f}"), verdicts(s).get(f"lean-verify/{f}")
        ands = a_.get("ands") or (m or {}).get("ands", "-")
        k_log = a_.get("k_log") or (m or {}).get("k_log", "-")
        print(f"| {f} | {r['calls']} | {r['gates']:.3e} | {100 * r['share']:.3f}% | {short(r.get('unit'), 48)} | "
              f"{ands} | {k_log} | {outcome(f, r, s)} | {twin} | "
              f"{'-' if mv is None else ('accepted' if mv else 'refused')} | {'-' if lv is None else ('accepted' if lv else 'refused')} | "
              f"{rid if mv is not None or m else '-'} |")
    lowers = [f for f, r in kinds.items() if outcome(f, r, chosen[f][1]) == "lowers"]
    held = [f for f in kinds if verdicts(chosen[f][1]).get(f"lean-verify/{f}") is True]
    print(f"\n{len(lowers)} of {len(kinds)} kinds lower ({100 * sum(kinds[f]['share'] for f in lowers):.2f}% of the step's computing "
          f"gates); {len(held)} have a unit proved by M0 and held by Lean "
          f"({100 * sum(kinds[f]['share'] for f in held):.2f}%).\n")
    if runs:
        print("Audits:\n")
        for rid, s in runs:
            au = s.get("audit") or {}
            neg = s.get("negatives") or {}
            negs = ("negatives not run" if s.get("negatives_run") is False else f"negatives refused {sum(neg.values())} of {len(neg)}"
                    + ("" if all(neg.values()) else " (not refused: " + ", ".join(x for x, y in neg.items() if not y) + ")"))
            print(f"- {rid}: members {[m['name'] for m in s.get('members', [])]}, draw {s.get('draw_counts')}, "
                  f"audit accepted {au.get('accepted')}, {negs}")
    print("\nGaps:\n")
    for f, r in sorted(kinds.items(), key=lambda kv: -kv[1]["gates"]):
        g = gap(f, r, chosen[f][1])
        if g:
            print(f"- {g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
