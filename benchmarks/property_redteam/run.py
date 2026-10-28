"""Grade a stacking-property check against the red team's cases (`cases.py`).

    uv run python benchmarks/property_redteam/run.py --check MODULE:FUNCTION [--case ID ...] [--out FILE.json]
    uv run python benchmarks/property_redteam/run.py --json > benchmarks/property_redteam/cases.json

A check is ``FUNCTION(program) -> {"refused": bool, "charged": {class: units}}`` over cases.py's `CLASSES` (``accept-all`` and
``refuse-all`` name the two trivial ones); a ``reasons`` entry, when it returns one, is copied into the report. The check's module is imported before the cases, so a Definition it registers is
the one the cases call. Each case's outcome:

  ok           the case's verdict, charged at least the case's charge (an attack) or within its allowance (an honest case)
  sound        an attack refused where its verdict budgets it, or accepted at its full charge where its verdict refuses it
  misclassed   an attack accepted at its charge in total, but not in the class the case names
  unsound      an attack accepted at less than its charge, or at all where no charge makes it sound
  over-strict  an honest case refused
  over-budget  an honest case charged past its allowance
  error        the check raised or returned no verdict, or an honest case did not build
  skipped      the case binds a module this environment lacks (the s8 family)

An attack the IR refuses to build counts as refused. The exit status is 0 when every graded case is ok or sound.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from collections.abc import Mapping
from typing import Any, Callable

PASSING = ("ok", "sound")
OUTCOMES = ("ok", "sound", "misclassed", "unsound", "over-strict", "over-budget", "error", "skipped")


def accept_all(program: Any) -> dict:
    return {"refused": False, "charged": {}}


def refuse_all(program: Any) -> dict:
    return {"refused": True, "charged": {}}


BUILTIN = {"accept-all": accept_all, "refuse-all": refuse_all}


def resolve(spec: str) -> Callable[[Any], Any]:
    if spec in BUILTIN:
        return BUILTIN[spec]
    module, sep, attr = spec.partition(":")
    if not sep or not module or not attr:
        raise SystemExit(f"--check is MODULE:FUNCTION or one of {sorted(BUILTIN)}, not {spec!r}")
    return getattr(importlib.import_module(module), attr)


def verdict(v: Any, classes: tuple[str, ...]) -> tuple[bool, dict[str, int]]:
    """(refused, charged) of a check's return, or ValueError naming what is wrong with it."""
    if not isinstance(v, Mapping) or not isinstance(v.get("refused"), bool):
        raise ValueError(f"a verdict is a mapping with a bool 'refused', not {v!r:.200}")
    charged = v.get("charged", {})
    if not isinstance(charged, Mapping):
        raise ValueError(f"'charged' is a mapping {{class: units}}, not {charged!r:.200}")
    bad = {k: n for k, n in charged.items() if k not in classes or isinstance(n, bool) or not isinstance(n, int) or n < 0}
    if bad:
        raise ValueError(f"'charged' maps the classes {classes} to ints >= 0; not {bad!r:.200}")
    return v["refused"], dict(charged)


def _have(charged: Mapping[str, int], cls: str) -> int:
    return sum(charged.values()) if cls == "any" else charged.get(cls, 0)


def grade(case: Any, refused: bool, charged: Mapping[str, int]) -> str:
    """The outcome of one verdict on one case (the module docstring)."""
    if case.kind == "honest":
        if refused:
            return "over-strict"
        allow = case.allow or {}
        within = (all(_have(charged, c) <= n for c, n in allow.items()) if "any" in allow
                  else all(n <= allow.get(c, 0) for c, n in charged.items()))
        return "ok" if within else "over-budget"
    if refused:
        return "ok" if case.verdict == "refuse" else "sound"
    if case.charge is None:
        return "unsound"
    if all(_have(charged, c) >= n for c, n in case.charge.items()):
        return "ok" if case.verdict == "budget" else "sound"
    return "misclassed" if sum(charged.values()) >= sum(case.charge.values()) else "unsound"


def run(check: Callable[[Any], Any], ids: list[str] | None = None) -> dict:
    """Every case (or those `ids`) graded under `check`: ``{"rows": [...], "outcomes": {outcome: n}, "passed": bool}``."""
    import cases as C

    chosen = [C.BY_ID[i] for i in ids] if ids else list(C.CASES)
    rows = []
    for case in chosen:
        row: dict[str, Any] = {"id": case.id, "kind": case.kind, "verdict": case.verdict, "charge": case.charge,
                               "allow": case.allow}
        if not case.available:
            rows.append(row | {"outcome": "skipped", "note": f"needs {', '.join(case.needs)}"})
            continue
        try:
            program = case.program()
        except (TypeError, ValueError) as e:
            outcome = "error" if case.kind == "honest" else grade(case, True, {})
            rows.append(row | {"outcome": outcome, "refused": True, "note": f"the IR refuses it at build: {e}"[:300]})
            continue
        try:
            v = check(program)
            refused, charged = verdict(v, C.CLASSES)
        except Exception as e:  # noqa: BLE001  (a check's failure is its outcome, recorded)
            rows.append(row | {"outcome": "error", "note": f"{type(e).__name__}: {e}"[:300]})
            continue
        row |= {"outcome": grade(case, refused, charged), "refused": refused, "charged": charged}
        rows.append(row | ({"reasons": v["reasons"]} if "reasons" in v else {}))
    outcomes = Counter(r["outcome"] for r in rows)
    return {"rows": rows, "outcomes": {o: outcomes[o] for o in OUTCOMES if outcomes[o]},
            "passed": all(r["outcome"] in PASSING or r["outcome"] == "skipped" for r in rows)}


def metadata() -> list[dict]:
    import cases as C

    return [c.to_json() for c in C.CASES]


def render(metadata_rows: list[dict]) -> str:
    return json.dumps(metadata_rows, indent=2) + "\n"


def _table(report: dict) -> str:
    lines = [f"{'case':20s} {'outcome':12s} {'refused':8s} charged"]
    for r in report["rows"]:
        lines.append(f"{r['id']:20s} {r['outcome']:12s} {str(r.get('refused', '')):8s} "
                     f"{json.dumps(r.get('charged', ''))} {r.get('note', '')} {json.dumps(r.get('reasons', ''))[:120]}".rstrip())
    lines.append("outcomes: " + ", ".join(f"{o} {n}" for o, n in report["outcomes"].items()))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", help="MODULE:FUNCTION, or accept-all / refuse-all")
    ap.add_argument("--case", action="append", default=None, help="grade only this case (repeatable)")
    ap.add_argument("--out", help="write the report as JSON here")
    ap.add_argument("--json", action="store_true", help="print the cases' metadata (cases.json) and exit")
    args = ap.parse_args(argv)
    if args.json:
        sys.stdout.write(render(metadata()))
        return 0
    if not args.check:
        ap.error("--check is required (or --json)")
    check = resolve(args.check)
    report = run(check, args.case)
    print(_table(report))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
