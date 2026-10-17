"""Comparing results and rendering the throughput frontier, from results only (files or the evidence store), never a table
kept in Git.

``compare`` is the judge: two results are compared only when they conform to the contract (``contract.validate``), their
identity fields agree (``contract.incomparable``) apart from those ``allow`` names, and their instances don't conflict
(``contract.compare_instances``). ``frontier`` puts one identity's points side by side over B, the one field a sweep
varies, and refuses a set whose identities differ in anything else.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from verity_numerical.bench import contract

from .result import dominant_bucket, value

HEADLINE = ("vu.per_second", "overhead.vs_native_peak", "t.total", "mem.peak_host_bytes", "mem.peak_device_bytes", "proof_bytes",
            "transcript.bytes", "verifier.seconds")


class Refused(Exception):
    """Two results that must not be compared, and why."""


def compare(baseline: Mapping[str, Any], candidate: Mapping[str, Any], allow: Sequence[str] = ()) -> dict[str, Any]:
    """The candidate's headline numbers against the baseline's, or :class:`Refused` naming every reason they can't be."""
    reasons = []
    for who, r in (("baseline", baseline), ("candidate", candidate)):
        problems = contract.validate(r)
        if problems:
            reasons.append(f"{who} does not conform to the contract: " + "; ".join(problems[:5]))
        elif r["validation"]["status"] != "passed":
            reasons.append(f"{who} did not pass: {r['validation'].get('detail')}")
    fa, fb = baseline.get("workload_fingerprint") or {}, candidate.get("workload_fingerprint") or {}
    for field, a, b in contract.incomparable(fa, fb, allow):
        reasons.append(f"{field} differs: {_show(a)} vs {_show(b)}")
    conflict, warnings = contract.compare_instances(fa, fb)
    if conflict:
        reasons.append(f"instances conflict: {conflict[0]} vs {conflict[1]}")
    if fa.get("workload") != fb.get("workload") or fa.get("config") != fb.get("config"):
        reasons.append(f"workload/config differ: {fa.get('workload')}/{fa.get('config')} vs {fb.get('workload')}/{fb.get('config')}")
    if reasons:
        raise Refused("; ".join(reasons))
    rows = {}
    for name in HEADLINE:
        a, b = value(baseline, name), value(candidate, name)
        if a is not None and b is not None:
            rows[name] = {"baseline": a, "candidate": b, "ratio": (b / a) if a else None}
    return {"baseline": baseline.get("run_id"), "candidate": candidate.get("run_id"), "allow": list(allow),
            "warnings": warnings, "measurements": rows}


def _show(v: Any) -> str:
    return "absent" if v is contract.MISSING else json.dumps(v, sort_keys=True, default=str)[:200]


def frontier(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per B (the newest result of each), refusing results whose identities differ apart from B."""
    passed = [r for r in results if (r.get("validation") or {}).get("status") == "passed" and not contract.validate(r)]
    if not passed:
        return []
    base = passed[0]["workload_fingerprint"]
    for r in passed[1:]:
        fp = r["workload_fingerprint"]
        bad = contract.incomparable(base, fp, allow=("B",))
        if bad or fp.get("workload") != base.get("workload") or fp.get("config") != base.get("config"):
            raise Refused(f"{r.get('run_id')} is not the same sweep as {passed[0].get('run_id')}: "
                          + "; ".join(f"{f} {_show(a)} vs {_show(b)}" for f, a, b in bad))
    by_b: dict[int, Mapping[str, Any]] = {}
    for r in passed:
        by_b[r["workload_fingerprint"]["B"]] = r
    rows = []
    for b in sorted(by_b):
        r = by_b[b]
        dom = dominant_bucket(r)
        rows.append({"B": b, "run": r.get("run_id"), **{n: value(r, n) for n in HEADLINE},
                     "dominant": dom[0] if dom else None, "dominant_share": dom[1] if dom else None})
    return rows


BASELINE_SCHEMA = "proof-throughput/baseline/v1"


def baseline(results: Sequence[Mapping[str, Any]], commit: str) -> dict[str, Any]:
    """The pin of one sweep: its workload, configuration and commit, and each B's result by run id (``baseline.json``).
    The numbers stay in the results; ``table --baseline`` renders them."""
    rows = frontier(results)
    if not rows:
        raise Refused("no passing result to pin")
    fp = next(r for r in results if r.get("run_id") == rows[0]["run"])["workload_fingerprint"]
    return {"schema": BASELINE_SCHEMA, "workload": fp["workload"], "config": fp["config"], "vu": fp["vu"], "commit": commit,
            "instances": {k: fp["instances"][k] for k in ("dataset", "tier", "manifest_sha256")},
            "points": {str(r["B"]): r["run"] for r in rows}}


def best(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The row with the highest VU throughput."""
    rows = [r for r in rows if r.get("vu.per_second")]
    return max(rows, key=lambda r: r["vu.per_second"]) if rows else None


def _fmt(v: Any, kind: str) -> str:
    if v is None:
        return "-"
    if kind == "bytes":
        for unit, scale in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
            if v >= scale:
                return f"{v / scale:.2f} {unit}"
        return f"{v} B"
    if kind == "share":
        return f"{v:.0%}"
    if kind == "int":
        return f"{v:,.0f}"
    return f"{v:.3g}"


def markdown(rows: Sequence[Mapping[str, Any]], title: str = "") -> str:
    cols = [("B", "B", "int"), ("VU/s", "vu.per_second", "num"), ("overhead vs H200 peak", "overhead.vs_native_peak", "int"),
            ("t.total s", "t.total", "num"), ("peak host", "mem.peak_host_bytes", "bytes"),
            ("peak device", "mem.peak_device_bytes", "bytes"), ("proof", "proof_bytes", "bytes"),
            ("verify s (Lean)", "verifier.seconds", "num"), ("dominant bucket", "dominant", "str"),
            ("share", "dominant_share", "share"), ("run", "run", "str")]
    out = [f"### {title}", ""] if title else []
    out.append("| " + " | ".join(c[0] for c in cols) + " |")
    out.append("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(r.get(k) if r.get(k) is not None else "-") if kind == "str" else _fmt(r.get(k), kind)
                                     for _h, k, kind in cols) + " |")
    return "\n".join(out) + "\n"
