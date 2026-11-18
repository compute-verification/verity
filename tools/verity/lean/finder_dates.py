"""`finder_dates`: the due dates of the definitions the locks list under `choosing` (`check.py`'s `choosing_failures`: a
definition a finder handed to a collision premise picks by choice through, listed until it is restated as a search).

    python3 tools/verity/lean/finder_dates.py --now ISO [--rev REV] [--base REV] [--json]

It is the form for drive to run after each landing with its own clock (``--now``, an ISO 8601 time; one without a zone is
UTC), where an entry past its date rings its owner and lean and holds nothing: a candidate's red is a change's, and the
passage of time is none. It runs from a checkout, never as a landing gate's copy of one file, since it loads
`finder_gate.py` from its own directory. It reads every tracked `lean-audit.json` at ``REV`` (default HEAD) from git
alone, the auditor's controls aside, and fails on each `choosing` entry whose `due` is before ``--now``'s day in UTC. With
``--base``, it fails only on an entry overdue at ``REV`` that is not overdue at the base at the same time: the change's
own. It builds nothing and reads no clock. Each failure names the definition, its lock, owner, issue and date, and the two
ways out: restate the definition as a search of data the finder holds, or its lead moves the date in a pull request,
which `finder_gate.py` shows.

Exit 0 when no entry is overdue, 1 when one is (``--json``: `KIND`, `{"failures": [{"definition", "lock", "owner",
"issue", "due", "why", "fix"}]}`), 2 when it can't judge: a git error, or a lock that is not JSON or whose `choosing` is
not a map of entries.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finder_gate import CantJudge, commit_of, entries, locks_at

KIND = "lean-finder-dates/v1"


def utc_day(now: str) -> datetime.date:
    """The UTC day of the ISO 8601 time ``now``; a time without a zone is UTC."""
    t = datetime.datetime.fromisoformat(now)
    return (t.astimezone(datetime.timezone.utc) if t.tzinfo else t).date()


def overdue(listed: dict[tuple[str, str], dict], today: datetime.date) -> set[tuple[str, str]]:
    """The entries of ``listed`` whose `due` is a day before ``today``; a `due` that is no `YYYY-MM-DD` is the check's
    `choosing.form` to refuse."""
    return {k for k, e in listed.items()
            if isinstance(d := e.get("due"), str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d) and d < today.isoformat()}


def dates(rev: str, today: datetime.date, base: str | None = None) -> dict:
    """The verdict (the module's docstring) of ``rev``'s locks on ``today``, against ``base``'s when given."""
    top = commit_of(rev)
    listed = entries(locks_at(top))
    late = overdue(listed, today)
    below = commit_of(base) if base else None
    if below:
        was = {d for _, d in overdue(entries(locks_at(below)), today)}
        late = {k for k in late if k[1] not in was}
    failures = []
    for path, d in sorted(late):
        e = listed[(path, d)]
        failures.append({"definition": d, "lock": path, "owner": e.get("owner"), "issue": e.get("issue"),
                         "due": e.get("due"),
                         "why": f"{d} ({path}, {e.get('owner')}, {e.get('issue')}) was due {e.get('due')}, and a finder "
                                f"handed to a collision premise still picks by choice through it",
                         "fix": [f"restate {d} as a search of data the finder holds, so no finder reaches choice "
                                 f"through it and `check.py --update` drops its entry",
                                 "or its lead moves its `due` in a pull request, which `finder_gate.py` shows"]})
    return {"kind": KIND, "rev": top, "base": below, "today": today.isoformat(), "listed": len(listed),
            "failures": failures}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="finder_dates", description=__doc__.split("\n\n")[0])
    ap.add_argument("--now", required=True, type=utc_day, metavar="ISO",
                    help="the time an entry is judged at, drive's clock (ISO 8601; UTC without a zone); nothing here reads one")
    ap.add_argument("--rev", default="HEAD", help="the commit whose locks it reads (default HEAD)")
    ap.add_argument("--base", help="judge only the entries overdue at --rev that are not at this commit, at the same time")
    ap.add_argument("--json", action="store_true", help=f"print {KIND}")
    a = ap.parse_args(argv)
    try:
        doc = dates(a.rev, a.now, a.base)
    except CantJudge as e:
        print(f"finder dates: can't judge: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(doc, indent=1))
    else:
        print(f"finder dates {doc['rev'][:12]}" + (f" (base {doc['base'][:12]})" if doc["base"] else "")
              + f" on {doc['today']}: {doc['listed']} listed, {len(doc['failures'])} overdue")
        print("".join(f"  {f['why']}\n" + "".join(f"    fix: {x}\n" for x in f["fix"]) for f in doc["failures"]), end="")
    return 1 if doc["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
