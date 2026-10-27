#!/usr/bin/env python3
"""Judge two recorded runs: refuse unless they are the same campaign number; otherwise ratios and fingerprint deltas.

    uv run python benchmarks/judge.py BASELINE CANDIDATE [--allow FIELD ...] [--runs-dir DIR] [--json]

``BASELINE`` and ``CANDIDATE`` are run ids (or unique prefixes) under ``~/.research/runs`` (``$RESEARCH_RUNS`` or
``--runs-dir``), or directories holding a ``result.json``.

``research compare`` shows every recorded difference between two runs and leaves to the workload whether a difference
invalidates the comparison.  This script is that judgment for the proof campaign.  Two results are comparable only
when they agree on the contract's identity fields -- ``proof_class``, ``authentication``, ``profile``, ``domain``,
``K`` (``verity_numerical.bench.contract.IDENTITY_FIELDS``; never compare across ``proof_class``).  A field either result lacks counts as a disagreement.  On disagreement the judge REFUSES (exit 2)
and prints nothing that looks like a ratio.  ``--allow K`` (repeatable) lets a named identity field differ when that
difference *is* the experiment (a scaling curve); the output then labels it.  Exit 0: judged; 1: a run could not be read.

The ``instances`` field (which ``bench-instances/v1`` ids were proved, ``contract.compare_instances``) is judged
alongside: two results that both record it and disagree (dataset, tier, manifest, or different id ranges for the same
batch size) are REFUSED; a result that lacks it or records ``null`` was not measured on the shared input set, and the
judge says so in a WARNING and goes on -- the comparison then rests on the identity fields alone.

Each result is also checked against the contract; problems are listed, not enforced (an early baseline has many).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
for candidate, module in ((REPO, "verity"), (REPO / "benchmarks" / "numerical" / "python", "verity_numerical")):
    try:
        __import__(module)
    except ImportError:
        sys.path.insert(0, str(candidate))

from verity_numerical.bench import contract  # noqa: E402

EX_REFUSED = 2


def runs_root(arg: str | None) -> Path:
    return Path(arg or os.environ.get("RESEARCH_RUNS") or Path.home() / ".research" / "runs").expanduser()


def resolve(ref: str, root: Path) -> Path:
    """A directory as given, else ``<root>/<ref>``, else the unique run under ``root`` whose id starts with ``ref``."""

    p = Path(ref).expanduser()
    if p.is_dir():
        return p
    if (root / ref).is_dir():
        return root / ref
    matches = sorted(d for d in root.glob(f"{ref}*") if d.is_dir()) if root.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"{ref!r}: not a directory, not a run under {root}" + (f", {len(matches)} runs match the prefix" if matches else ""))


def load(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "result.json"
    if not p.is_file():
        raise FileNotFoundError(f"{run_dir}: no result.json (not fetched, or the workload wrote none)")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{p}: top level must be an object")
    return doc


def flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(d, dict):
        return {prefix.rstrip("."): d}
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def judge(base_dir: Path, cand_dir: Path, allow: list[str]) -> dict[str, Any]:
    rb, rc = load(base_dir), load(cand_dir)
    fb, fc = dict(rb.get("workload_fingerprint") or {}), dict(rc.get("workload_fingerprint") or {})
    refused = contract.incomparable(fb, fc, allow)
    conflict, warnings = contract.compare_instances(fb, fc)
    ib, ic = contract.identity(fb), contract.identity(fc)
    identity = [{"field": f, "baseline": _show(ib[f]), "candidate": _show(ic[f]),
                 "same": ib[f] is not contract.MISSING and ib[f] == ic[f], "allowed": f in allow} for f in contract.IDENTITY_FIELDS]
    identity.append({"field": contract.INSTANCES_FIELD, "baseline": _show(fb.get(contract.INSTANCES_FIELD, contract.MISSING)),
                     "candidate": _show(fc.get(contract.INSTANCES_FIELD, contract.MISSING)),
                     "same": conflict is None and not warnings, "allowed": False})
    rep: dict[str, Any] = {
        "baseline": {"id": rb.get("run_id"), "dir": str(base_dir)},
        "candidate": {"id": rc.get("run_id"), "dir": str(cand_dir)},
        "identity": identity,
        "refused": [{"field": f, "baseline": _show(a), "candidate": _show(b)} for f, a, b in refused]
        + ([{"field": contract.INSTANCES_FIELD, "baseline": conflict[0], "candidate": conflict[1]}] if conflict else []),
        "warnings": warnings,
        "contract": {"baseline": contract.validate(rb), "candidate": contract.validate(rc)},
    }
    refused = rep["refused"]
    if refused:
        return rep
    flat_b, flat_c = flatten(fb), flatten(fc)
    rep["fingerprint_deltas"] = [{"field": k, "baseline": flat_b.get(k, "<absent>"), "candidate": flat_c.get(k, "<absent>")}
                                 for k in sorted(set(flat_b) | set(flat_c)) if flat_b.get(k, "<absent>") != flat_c.get(k, "<absent>")]
    mb = {m["name"]: m for m in rb.get("measurements") or [] if isinstance(m, dict) and "name" in m}
    mc = {m["name"]: m for m in rc.get("measurements") or [] if isinstance(m, dict) and "name" in m}
    rows = []
    for name in sorted(set(mb) | set(mc)):
        b, c = mb.get(name), mc.get(name)
        row: dict[str, Any] = {"name": name, "baseline": b and b.get("value"), "candidate": c and c.get("value"),
                               "unit": (b or c).get("unit"), "unit_mismatch": bool(b and c and b.get("unit") != c.get("unit")), "ratio": None}
        if b and c and not row["unit_mismatch"] and _num(b.get("value")) and _num(c.get("value")):
            row["ratio"] = (c["value"] / b["value"]) if b["value"] else None
        rows.append(row)
    rep["measurements"] = rows
    return rep


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _show(v: Any) -> Any:
    return "<missing>" if v is contract.MISSING else v


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.6g}"
    s = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
    return s if len(s) <= 48 else s[:45] + "..."


def render(rep: dict[str, Any]) -> str:
    lines = [f"baseline   {rep['baseline']['id']}  {rep['baseline']['dir']}", f"candidate  {rep['candidate']['id']}  {rep['candidate']['dir']}", ""]
    lines.append("[identity]  the fields two results must share before a number is compared")
    for row in rep["identity"]:
        mark = "=" if row["same"] else "->"
        tag = "  (allowed to differ)" if row["allowed"] and not row["same"] else ("" if row["same"] else "  DIFFERS")
        lines.append(f"  {row['field']:<16} {_fmt(row['baseline']):>36}  {mark:>2}  {_fmt(row['candidate']):<36}{tag}")
    for w in rep.get("warnings", ()):
        lines.append(f"  WARNING: {w}")
    lines.append("")
    lines.append("[contract]")
    for who in ("baseline", "candidate"):
        probs = rep["contract"][who]
        lines.append(f"  {who:<10} " + ("conforms" if not probs else f"{len(probs)} problem(s)"))
        lines += [f"             - {p}" for p in probs]
    if rep["refused"]:
        lines.append("")
        lines.append("REFUSED: not the same campaign number; these runs must not be compared:")
        lines += [f"  {r['field']}: {_fmt(r['baseline'])} vs {_fmt(r['candidate'])}" for r in rep["refused"]]
        lines.append(f"  (--allow FIELD lets a named identity field differ when that difference is the experiment; fields: {', '.join(contract.IDENTITY_FIELDS)})")
        return "\n".join(lines)
    lines.append("")
    lines.append("[fingerprint deltas]" + ("" if rep["fingerprint_deltas"] else "  none"))
    for d in rep["fingerprint_deltas"]:
        lines.append(f"  {d['field']:<40} {_fmt(d['baseline'])}  ->  {_fmt(d['candidate'])}")
    lines.append("")
    lines.append(f"[measurements]  {'name':<52} {'baseline':>14}  {'candidate':>14}  {'cand/base':>10}  unit")
    for m in rep["measurements"]:
        ratio = "-" if m["ratio"] is None else f"x{m['ratio']:.4f}"
        lines.append(f"  {m['name']:<66} {_fmt(m['baseline']):>14}  {_fmt(m['candidate']):>14}  {ratio:>10}  {m['unit']}" + ("  UNIT MISMATCH" if m["unit_mismatch"] else ""))
    lines.append("")
    lines.append("one pair of runs is one measured ratio, not a confidence interval; ratios are candidate / baseline")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--allow", action="append", default=[], metavar="FIELD", help=f"let this identity field differ (one of {', '.join(contract.IDENTITY_FIELDS)}); repeatable")
    ap.add_argument("--runs-dir", default=None, help="registry root (default $RESEARCH_RUNS or ~/.research/runs)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    allow = [f for item in a.allow for f in item.split(",") if f]
    bad = sorted(set(allow) - set(contract.IDENTITY_FIELDS))
    if bad:
        ap.error(f"--allow takes identity fields {contract.IDENTITY_FIELDS}, not {bad}")
    root = runs_root(a.runs_dir)
    try:
        base_dir, cand_dir = resolve(a.baseline, root), resolve(a.candidate, root)
        rep = judge(base_dir, cand_dir, allow)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"judge: {e}", file=sys.stderr)
        return 1
    print(json.dumps(rep, indent=1, sort_keys=True, default=str) if a.json else render(rep))
    return EX_REFUSED if rep["refused"] else 0


if __name__ == "__main__":
    sys.exit(main())
