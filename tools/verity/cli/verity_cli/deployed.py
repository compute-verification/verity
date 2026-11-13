"""`verity lean deployed [--rev REV] [--today YYYY-MM-DD]`: the deployed guarantees' rows and the premises they rest on, read
from the Lean locks (`lean-audit.json`) at a commit, as one JSON object on standard output.

It reads git objects only (`git ls-tree`, `git show`): no checkout of REV, no build, no Lean. Every tracked lock at REV is
read, the auditor's fixtures aside (`check.FIXTURES`, which `check.packages` leaves out too). Each deployed guarantee's gap is
the Lean check's own (`trust_gaps` of `tools/verity/lean/check.py`, loaded by path from this checkout), beside the gap the
lock holds (`trust_gap`). It prints what the locks record and no verdict: whether a guarantee is proved is the Lean check's
to say, at the commit it passed on.

The object, its keys in this order and every list sorted by name:

- `format` (`FORMAT`), `commit` (the full sha REV resolves to), `today` (the day dates are compared to), `locks` (the path
  of every lock read);
- `guarantees`: for each guarantee a lock's `deployed` lists, its `name`, `lock` and `owner`, its record's `type_hash` and
  `unfolded_hash`, its `gap` (`[]` at zero), and `held_differs`: null, or `{"only_held", "only_computed"}` where the lock's
  `trust_gap` entry and the computed gap differ;
- `premises`: each open premise of any lock and each premise of any gap, once by name: its `kind` (`KINDS`, by the section
  that lists it; where locks list it in different sections, the first of `KINDS` among them), its `owner`, `issue` and `due`
  (the value every lock listing it gives, else null), `listed` (each lock that lists it, with that lock's `owner`, `issue`
  and `due`), `overdue` (its earliest `due` is before `today`) and `in_gap_of` (the deployed guarantees whose gap holds it);
- `totals`: `deployed`, `gap_zero`, `gap_premises` (distinct names over all gaps), `open` (premises of kind `open`),
  `overdue`, `owners_differ` (premises whose locks name different owners) and `by_owner` ({owner: its `deployed`,
  `gap_premises`, `open` and `overdue`}, a premise counted under each owner its locks name; what names no owner is under
  none).

A field a lock leaves out is null. A lock that is not a JSON object, whose `deployed` is not a list of names, or whose
`deployed` names a guarantee it has no record of, ends the command with exit 1 and one line naming the lock and what is
wrong.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import subprocess
import sys
import zoneinfo
from pathlib import Path

LEAN = Path(__file__).resolve().parents[2] / "lean"
FORMAT = 1
POLICY = "lean-audit.json"
#: the lock's sections that list premises, and each one's `kind`, in the order a premise listed in several takes its kind
KINDS = {"open": "open", "assumptions": "assumption", "domain": "domain"}
#: the zone `--today` defaults to today's date in
ZONE = "America/Los_Angeles"


class Refused(Exception):
    """A lock this command can't read, with the line it ends on."""


def lean_check():
    """`tools/verity/lean/check.py`, whose `trust_gaps` is the gap (the directory is copied, not installed)."""
    spec = importlib.util.spec_from_file_location("lean_check_for_verity_cli", LEAN / "check.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _str(v) -> str | None:
    return v if isinstance(v, str) else None


def _day(v) -> datetime.date | None:
    try:
        return datetime.date.fromisoformat(v) if isinstance(v, str) else None
    except ValueError:
        return None


def _parse(path: str, text: str, check) -> dict:
    try:
        lock = json.loads(text)
    except ValueError as e:
        raise Refused(f"verity lean deployed: {path}: not JSON ({e})") from None
    if not isinstance(lock, dict):
        raise Refused(f"verity lean deployed: {path}: not a JSON object")
    deployed = lock.get("deployed", [])
    if deployed != check.name_list(deployed):
        raise Refused(f"verity lean deployed: {path}: `deployed` is not a list of names")
    records = lock.get("guarantees") if isinstance(lock.get("guarantees"), dict) else {}
    if missing := sorted(g for g in deployed if not isinstance(records.get(g), dict)):
        raise Refused(f"verity lean deployed: {path}: `deployed` names {', '.join(missing)}, which `guarantees` has no "
                      "record of")
    return lock


def rows(texts: dict[str, str], today: datetime.date, commit: str, check=None) -> dict:
    """The object (the module's docstring) of the locks ``texts`` ({path: its text}) at ``commit``, dates compared to
    ``today``; `Refused` names the first lock, by path, that can't be read."""
    check = check or lean_check()
    locks = {p: _parse(p, texts[p], check) for p in sorted(texts)}
    guarantees, listings = [], {}
    for path, lock in locks.items():
        held = lock.get("trust_gap") if isinstance(lock.get("trust_gap"), dict) else {}
        for g, gap in check.trust_gaps(lock).items():
            rec, kept = lock["guarantees"][g], set(check.name_list(held.get(g)))
            differs = None if kept == set(gap) else {"only_held": sorted(kept - set(gap)),
                                                     "only_computed": sorted(set(gap) - kept)}
            guarantees.append({"name": g, "lock": path, "owner": _str(rec.get("owner")), "type_hash": _str(rec.get("type_hash")),
                               "unfolded_hash": _str(rec.get("unfolded_hash")), "gap": gap, "held_differs": differs})
        for section in KINDS:
            for name, entry in (lock.get(section) if isinstance(lock.get(section), dict) else {}).items():
                entry = entry if isinstance(entry, dict) else {}
                listings.setdefault(name, []).append((section, {"lock": path, **{f: _str(entry.get(f))
                                                                                  for f in ("owner", "issue", "due")}}))
    guarantees.sort(key=lambda r: (r["name"], r["lock"]))
    in_gap: dict[str, set[str]] = {}
    for r in guarantees:
        for p in r["gap"]:
            in_gap.setdefault(p, set()).add(r["name"])
    premises = []
    for name in sorted({n for n, ls in listings.items() if any(s == "open" for s, _ in ls)} | set(in_gap)):
        ls = listings.get(name, [])
        listed = sorted((e for _, e in ls), key=lambda e: e["lock"])
        agreed = {f: vs.pop() if len(vs := {e[f] for e in listed}) == 1 else None for f in ("owner", "issue", "due")}
        dues = [d for e in listed if (d := _day(e["due"]))]
        premises.append({"name": name, "kind": next((KINDS[s] for s in KINDS if any(s == t for t, _ in ls)), None), **agreed,
                         "listed": listed, "overdue": bool(dues) and min(dues) < today,
                         "in_gap_of": sorted(in_gap.get(name, ()))})
    return {"format": FORMAT, "commit": commit, "today": today.isoformat(), "locks": sorted(locks),
            "guarantees": guarantees, "premises": premises, "totals": totals(guarantees, premises)}


def totals(guarantees: list[dict], premises: list[dict]) -> dict:
    """`totals` of ``guarantees`` and ``premises`` (the module's docstring)."""
    def owners(p: dict) -> set[str]:
        return {e["owner"] for e in p["listed"] if e["owner"]}

    by_owner: dict[str, dict[str, int]] = {}

    def count(owner: str, key: str) -> None:
        by_owner.setdefault(owner, dict.fromkeys(("deployed", "gap_premises", "open", "overdue"), 0))[key] += 1

    for r in guarantees:
        if r["owner"]:
            count(r["owner"], "deployed")
    for p in premises:
        for o in owners(p):
            for key, holds in (("gap_premises", p["in_gap_of"]), ("open", p["kind"] == "open"), ("overdue", p["overdue"])):
                if holds:
                    count(o, key)
    return {"deployed": len(guarantees), "gap_zero": sum(not r["gap"] for r in guarantees),
            "gap_premises": sum(bool(p["in_gap_of"]) for p in premises), "open": sum(p["kind"] == "open" for p in premises),
            "overdue": sum(p["overdue"] for p in premises), "owners_differ": sum(len(owners(p)) > 1 for p in premises),
            "by_owner": {o: by_owner[o] for o in sorted(by_owner)}}


def _git(repo: Path, *argv: str) -> str:
    return subprocess.run(["git", *argv], cwd=repo, capture_output=True, encoding="utf-8", errors="replace",
                          check=True).stdout


def locks_at(repo: Path, rev: str, check=None) -> tuple[str, dict[str, str]]:
    """The full sha ``rev`` resolves to in ``repo``, and {path: text} of every tracked lock there but the auditor's
    fixtures."""
    check = check or lean_check()
    try:
        commit = _git(repo, "rev-parse", "--verify", "--end-of-options", f"{rev}^{{commit}}").strip()
    except subprocess.CalledProcessError as e:
        raise Refused(f"verity lean deployed: no commit {rev!r} here ({e.stderr.strip() or 'git rev-parse failed'})") from None
    fixtures = check.FIXTURES.relative_to(check.ROOT).as_posix() + "/"
    paths = [p for p in _git(repo, "ls-tree", "-r", "-z", "--full-tree", "--name-only", commit).split("\0")
             if p.rpartition("/")[2] == POLICY and not p.startswith(fixtures)]
    return commit, {p: _git(repo, "show", f"{commit}:{p}") for p in paths}


def today_here() -> datetime.date:
    return datetime.datetime.now(zoneinfo.ZoneInfo(ZONE)).date()


def main(args: list[str], repo: Path | None = None) -> int:
    ap = argparse.ArgumentParser(prog="verity lean deployed", description="Print the deployed guarantees' rows as one JSON "
                                 "object: each deployed guarantee's gap as the Lean check computes it (`trust_gaps`) beside "
                                 "the one its lock holds, and each open or gap premise once by name, with its owners, issue "
                                 "and due date in every lock that lists it, read from the locks (lean-audit.json) at REV by "
                                 "git alone. It reports what the locks record, no verdict.")
    ap.add_argument("--rev", default="HEAD", help="the commit whose locks it reads (default HEAD)")
    ap.add_argument("--today", type=datetime.date.fromisoformat, metavar="YYYY-MM-DD",
                    help=f"the day a due date is overdue after (default today's date in {ZONE})")
    a = ap.parse_args(args)
    check = lean_check()
    try:
        commit, texts = locks_at(repo or Path.cwd(), a.rev, check)
        out = rows(texts, a.today or today_here(), commit, check)
    except Refused as e:
        print(e, file=sys.stderr)
        return 1
    print(json.dumps(out, indent=2))
    return 0
