"""`finder_dates`: the due dates of the definitions the locks list under `choosing` (`check.py`'s `choosing_failures`: a
definition a finder handed to a collision premise picks by choice through, listed until it is restated as a search).

    python3 tools/verity/lean/finder_dates.py --today YYYY-MM-DD [--rev REV] [--base REV] [--json]

It reads every tracked `lean-audit.json` at ``REV`` (default HEAD) from git alone, the auditor's controls aside
(`FIXTURES`), and fails on each `choosing` entry whose `due` is before ``--today``. With ``--base``, it fails only on an
entry overdue at ``REV`` that is not overdue at the base under the same day: the change's own. It builds nothing and reads
no clock: the day is given, so its verdict holds for that day alone and is reused within it, never across. Each failure
names the definition, its lock, owner, issue and date, and the two ways out: restate the definition as a search of data
the finder holds, or its lead moves the date in a pull request, which `finder_gate.py` shows.

Exit 0 when no entry is overdue, 1 when one is (``--json``: `KIND`, `{"failures": [{"definition", "lock", "owner",
"issue", "due", "why", "fix"}]}`), 2 when it can't judge: a git error, or a lock that is not JSON or whose `choosing` is
not a map of entries.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys

KIND = "lean-finder-dates/v1"
POLICY = "lean-audit.json"
#: the auditor's controls, whose locks are not the repository's
FIXTURES = "tools/verity/lean/tests/fixtures/"


class CantJudge(RuntimeError):
    pass


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], capture_output=True, encoding="utf-8", errors="replace")
    if r.returncode:
        raise CantJudge(f"git {' '.join(args[:3])}: {r.stderr.strip()[-300:]}")
    return r.stdout


def commit_of(rev: str) -> str:
    return git("rev-parse", "--verify", "--end-of-options", f"{rev}^{{commit}}").strip()


def locks_at(rev: str) -> dict[str, dict]:
    """{path: the lock} of every tracked lock at ``rev`` but the controls'."""
    paths = [p for p in git("ls-tree", "-r", "-z", "--full-tree", "--name-only", rev).split("\0")
             if p.rpartition("/")[2] == POLICY and not p.startswith(FIXTURES)]
    out = {}
    for p in paths:
        try:
            lock = json.loads(git("show", f"{rev}:{p}"))
        except ValueError as e:
            raise CantJudge(f"{p} at {rev}: not JSON ({e})") from None
        if not isinstance(lock, dict):
            raise CantJudge(f"{p} at {rev}: not a JSON object")
        out[p] = lock
    return out


def entries(locks: dict[str, dict]) -> dict[tuple[str, str], dict]:
    """{(lock, definition): its `choosing` entry} of ``locks``."""
    out = {}
    for path, lock in sorted(locks.items()):
        chosen = lock.get("choosing", {})
        if not isinstance(chosen, dict) or not all(isinstance(e, dict) for e in chosen.values()):
            raise CantJudge(f"{path}: `choosing` is not a map from a definition to its entry")
        out |= {(path, d): e for d, e in chosen.items()}
    return out


def overdue(listed: dict[tuple[str, str], dict], today: datetime.date) -> set[tuple[str, str]]:
    """The entries of ``listed`` whose `due` is a date before ``today``."""
    def day(e: dict) -> datetime.date | None:
        try:
            return datetime.date.fromisoformat(e.get("due")) if isinstance(e.get("due"), str) else None
        except ValueError:
            return None
    return {k for k, e in listed.items() if (d := day(e)) is not None and d < today}


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
    ap.add_argument("--today", required=True, type=datetime.date.fromisoformat, metavar="YYYY-MM-DD",
                    help="the day a due date is overdue after; nothing here reads a clock")
    ap.add_argument("--rev", default="HEAD", help="the commit whose locks it reads (default HEAD)")
    ap.add_argument("--base", help="judge only the entries overdue at --rev that are not at this commit, under one day")
    ap.add_argument("--json", action="store_true", help=f"print {KIND}")
    a = ap.parse_args(argv)
    try:
        doc = dates(a.rev, a.today, a.base)
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
