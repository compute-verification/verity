"""Every labelled decision of fixtures.toml on one screen -- the review list for the morning.

    cd integrations/vllm && python -m tests.regression.decisions            # all
    cd integrations/vllm && python -m tests.regression.decisions --pending  # status = pending_review only
    cd integrations/vllm && python -m pytest tests/regression -k decisions -s

stdlib only (tomllib): no store, no `research`, no VERITY_REGRESSION.  A decision is `[[rows.<row>.decisions]]` (resolver.sanction:
`check`, `since`, `finding`, `reason`, `expected` = what the check's actual must state for the difference from v1 to be sanctioned;
optional `status` = accepted | pending_review, `since_commit`, `pins."<rel>" = { bytes, sha256 }` = an input the decision's outcome
replaces downstream, resolver.Row._checked).  `problem` names a malformed one (sanctions nothing; the test fails on it).
The retired checks follow (`[rows.<row>.retired.<check>]`: `reason`, `since`, `needs`), each skipped by its reason.
"""
from __future__ import annotations

import argparse
import textwrap
import sys
from typing import Any

from . import resolver


def listing() -> list[dict[str, Any]]:
    fx = resolver.load_fixtures()
    out: list[dict[str, Any]] = []
    for key, spec in sorted((fx.get("rows") or {}).items(), key=lambda kv: int(kv[1].get("row", 0))):
        for d in spec.get("decisions") or []:
            if not isinstance(d, dict):
                out.append({"row": spec.get("row"), "row_id": key, "check": "?", "finding": "?", "status": "?", "since": "?", "since_commit": "",
                            "reason": "", "expected": {}, "pins": {}, "problem": f"not a table: {d!r}"})
                continue
            status = resolver.decision_status(d)
            missing = [k for k in resolver.DECISION_FIELDS if not d.get(k)]
            problem = ""
            if missing:
                problem = f"missing {missing}"
            elif status not in resolver.DECISION_STATUSES:
                problem = f"unknown status {status!r} (one of {resolver.DECISION_STATUSES})"
            elif not isinstance(d.get("expected"), dict):
                problem = "expected is not a table"
            for rel, pin in (d.get("pins") or {}).items():
                if not isinstance(pin, dict) or not (pin.get("bytes") or pin.get("sha256")):
                    problem = problem or f"pins.{rel!r} names neither bytes nor sha256"
            out.append({"row": spec.get("row"), "row_id": key, "check": d.get("check") or "?", "finding": d.get("finding") or "?", "status": status,
                        "since": d.get("since") or "?", "since_commit": d.get("since_commit") or "", "reason": d.get("reason") or "",
                        "expected": d.get("expected") if isinstance(d.get("expected"), dict) else {}, "pins": d.get("pins") or {}, "problem": problem})
    return out


def retirements() -> list[dict[str, Any]]:
    """Every `[rows.<row>.retired.<check>]` (`resolver.Row.retired`), with `problem` naming a malformed one (a missing field of
    RETIREMENT_FIELDS, or a check no module under `checks/` defines)."""
    fx = resolver.load_fixtures()
    out: list[dict[str, Any]] = []
    for key, spec in sorted((fx.get("rows") or {}).items(), key=lambda kv: int(kv[1].get("row", 0))):
        for check, r in sorted((spec.get("retired") or {}).items()):
            r = r if isinstance(r, dict) else {}
            missing = [k for k in resolver.RETIREMENT_FIELDS if not r.get(k)]
            problem = (f"missing {missing}" if missing else "") or ("" if (resolver.HERE / "checks" / f"{check}.py").is_file() else f"no check named {check!r}")
            out.append({"row": spec.get("row"), "row_id": key, "check": check, "reason": r.get("reason") or "", "since": r.get("since") or "?",
                        "needs": r.get("needs") or "", "problem": problem})
    return out


def render_retired(rows: list[dict[str, Any]], *, width: int = 132) -> str:
    lines = [f"retired checks in {resolver.FIXTURES.relative_to(resolver.VLLM_ROOT)}: {len(rows)}", ""]
    for r in rows:
        lines.append(f"{'MALFORMED ' if r['problem'] else ''}#{r['row']} {r['check']}  since {r['since']}")
        if r["problem"]:
            lines.append(f"    PROBLEM  {r['problem']}")
        for label, text in (("reason", r["reason"]), ("needs", r["needs"])):
            for i, chunk in enumerate(textwrap.wrap(text, width=width - 13) or []):
                lines.append((f"    {label:<8} " if i == 0 else "             ") + chunk)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render(rows: list[dict[str, Any]], *, width: int = 132) -> str:
    findings = {(d["finding"], d["row"]) for d in rows}                      # one decision = one finding on one row; a block per check it touches
    n_pending = sum(1 for d in rows if d["status"] == "pending_review")
    lines = [f"labelled decisions in {resolver.FIXTURES.relative_to(resolver.VLLM_ROOT)}: {len(findings)} (by finding x row), "
             f"{len(rows)} check consequence(s), {n_pending} pending_review", ""]
    for d in rows:
        flag = "PENDING " if d["status"] == "pending_review" else ("MALFORMED " if d["problem"] else "")
        lines.append(f"{flag}#{d['row']} {d['check']}  {d['finding']}  [{d['status']}]  since {d['since']}"
                     + (f" @ {d['since_commit'][:12]}" if d["since_commit"] else ""))
        lines.append(f"    row      {d['row_id']}")
        if d["expected"]:
            lines.append(f"    expected {resolver.canonical(d['expected'])}")
        for rel, pin in sorted(d["pins"].items()):
            lines.append(f"    pins     {rel}: {pin.get('bytes')} B / {str(pin.get('sha256') or '')[:16]}…")
        if d["problem"]:
            lines.append(f"    PROBLEM  {d['problem']}")
        for i, chunk in enumerate(textwrap.wrap(d["reason"], width=width - 13) or []):
            lines.append(("    reason   " if i == 0 else "             ") + chunk)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pending", action="store_true", help="status = pending_review only")
    a = ap.parse_args(argv)
    rows = listing()
    if a.pending:
        rows = [d for d in rows if d["status"] == "pending_review"]
    sys.stdout.write(render(rows))
    retired = [] if a.pending else retirements()
    if retired:
        sys.stdout.write("\n" + render_retired(retired))
    return 1 if any(d["problem"] for d in rows + retired) else 0


if __name__ == "__main__":
    raise SystemExit(main())
