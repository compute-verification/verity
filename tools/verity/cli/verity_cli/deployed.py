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
  `unfolded_hash`, its `gap` (`[]` at zero), `lacks_witness` (null when its lock lists it under neither `witnesses` nor
  `unwitnessed`, else the halves of `WITNESS_HALVES` its `witnesses` entry names by no non-empty string or list; `[]` when
  it names both) and `held_differs`: null, or `{"only_held", "only_computed"}` where the lock's `trust_gap` entry and the
  computed gap differ;
- `premises`: each open premise of any lock and each premise of any gap, once by name: its `kind` (`KINDS`, by the section
  that lists it; where locks list it in different sections, the first of `KINDS` among them), its `owner`, `issue` and `due`
  (the value every lock listing it gives, else null), `listed` (each lock that lists it, with the `kind`, `owner`, `issue`
  and `due` of that lock's entry), `overdue` (its earliest `due` written YYYY-MM-DD, `check.OPEN_DUE`, is before `today`)
  and `in_gap_of` (the deployed guarantees whose gap holds it). A gap's `choice:<head>` item (`check.CHOICE`: a finder
  handed to that collision premise picks by choice) is of kind `choice`, listed once for each `choosing` entry of a
  definition the records of its guarantee's closure list under `premises.choosing` (with that `definition`), and its
  `due` is the earliest of theirs;
- `totals`: `deployed`, `gap_zero`, `gap_premises` (distinct names over all gaps), `open` (premises of kind `open`),
  `overdue`, `owners_differ` (premises whose locks name different owners), `no_honest_witness` (deployed guarantees whose
  `lacks_witness` holds `honest`) and `by_owner` ({owner: its `deployed`, `gap_premises`, `open`, `overdue` and
  `no_honest_witness`}, a premise counted under each owner its locks name; what names no owner is under none).

Every owner is printed bare, as the lock writes it less one leading `@`, and held to no registry. A field a lock leaves out
is null. A lock that is not a JSON object, whose `deployed` is not a list of names, whose `deployed` names a guarantee it
has no record of, or whose `witnesses` or `unwitnessed` is not an object, ends the command with exit 1 and one line naming
the lock and what is wrong.
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
KINDS = {"open": "open", "assumptions": "assumption", "domain": "domain", "choosing": "choice"}
#: the two witnesses a `witnesses` entry names for a guarantee of what an acceptance means
WITNESS_HALVES = ("honest", "nontrivial")
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


def _owner(v) -> str | None:
    """An owner as the metrics and the handle registry write it: the lock's, one leading `@` dropped."""
    return v.removeprefix("@") if isinstance(v, str) else None


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
    for section in ("witnesses", "unwitnessed"):
        if not isinstance(lock.get(section, {}), dict):
            raise Refused(f"verity lean deployed: {path}: `{section}` is not an object")
    return lock


def lacks_witness(lock: dict, name: str) -> list[str] | None:
    """The halves of `WITNESS_HALVES` that ``lock``'s `witnesses` entry for ``name`` names by no non-empty string or list,
    or None when the lock lists ``name`` under neither `witnesses` nor `unwitnessed`."""
    listed, exempt = lock.get("witnesses") or {}, lock.get("unwitnessed") or {}
    if name not in listed and name not in exempt:
        return None
    entry = listed.get(name) if isinstance(listed.get(name), dict) else {}
    return [h for h in WITNESS_HALVES if not (isinstance(v := entry.get(h), (str, list)) and v)]


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
            guarantees.append({"name": g, "lock": path, "owner": _owner(rec.get("owner")), "type_hash": _str(rec.get("type_hash")),
                               "unfolded_hash": _str(rec.get("unfolded_hash")), "gap": gap,
                               "lacks_witness": lacks_witness(lock, g), "held_differs": differs})
        chosen = lock.get("choosing") if isinstance(lock.get("choosing"), dict) else {}
        for g, gap in ((r["name"], r["gap"]) for r in guarantees if r["lock"] == path):
            for item in (p for p in gap if p.startswith(check.CHOICE)):
                for d in sorted(check.premise_closure(lock, g, ("choosing",)) & set(chosen)):
                    entry = chosen[d] if isinstance(chosen[d], dict) else {}
                    row = {"lock": path, "kind": "choice", "owner": _owner(entry.get("owner")), "issue": _str(entry.get("issue")),
                           "due": _str(entry.get("due")), "definition": d}
                    if ("choosing", row) not in listings.setdefault(item, []):
                        listings[item].append(("choosing", row))
        for section in (s for s in KINDS if s != "choosing"):
            for name, entry in (lock.get(section) if isinstance(lock.get(section), dict) else {}).items():
                entry = entry if isinstance(entry, dict) else {}
                listings.setdefault(name, []).append((section, {"lock": path, "kind": KINDS[section],
                                                                "owner": _owner(entry.get("owner")),
                                                                "issue": _str(entry.get("issue")), "due": _str(entry.get("due"))}))
    guarantees.sort(key=lambda r: (r["name"], r["lock"]))
    holders: dict[str, set[str]] = {}
    for r in guarantees:
        for p in r["gap"]:
            holders.setdefault(p, set()).add(r["name"])
    premises = []
    for name in sorted({n for n, ls in listings.items() if any(s == "open" for s, _ in ls)} | set(holders)):
        ls = listings.get(name, [])
        listed = sorted((e for _, e in ls), key=lambda e: (e["lock"], list(KINDS.values()).index(e["kind"]),
                                                           e.get("definition") or ""))
        agreed = {f: vs.pop() if len(vs := {e[f] for e in listed}) == 1 else None for f in ("owner", "issue", "due")}
        dues = [e["due"] for e in listed if e["due"] and check.OPEN_DUE.fullmatch(e["due"])]
        if name.startswith(check.CHOICE):
            agreed["due"] = min(dues, default=None)
        premises.append({"name": name, "kind": next((KINDS[s] for s in KINDS if any(s == t for t, _ in ls)), None), **agreed,
                         "listed": listed, "overdue": bool(dues) and min(dues) < today.isoformat(),
                         "in_gap_of": sorted(holders.get(name, ()))})
    return {"format": FORMAT, "commit": commit, "today": today.isoformat(), "locks": sorted(locks),
            "guarantees": guarantees, "premises": premises, "totals": totals(guarantees, premises)}


def totals(guarantees: list[dict], premises: list[dict]) -> dict:
    """`totals` of ``guarantees`` and ``premises`` (the module's docstring)."""
    def owners(p: dict) -> set[str]:
        return {e["owner"] for e in p["listed"] if e["owner"]}

    by_owner: dict[str, dict[str, int]] = {}

    def count(owner: str, key: str) -> None:
        keys = ("deployed", "gap_premises", "open", "overdue", "no_honest_witness")
        by_owner.setdefault(owner, dict.fromkeys(keys, 0))[key] += 1

    def unwitnessed(r: dict) -> bool:
        return "honest" in (r["lacks_witness"] or ())

    for r in guarantees:
        for key, holds in (("deployed", True), ("no_honest_witness", unwitnessed(r))):
            if r["owner"] and holds:
                count(r["owner"], key)
    for p in premises:
        for o in owners(p):
            for key, holds in (("gap_premises", p["in_gap_of"]), ("open", p["kind"] == "open"), ("overdue", p["overdue"])):
                if holds:
                    count(o, key)
    return {"deployed": len(guarantees), "gap_zero": sum(not r["gap"] for r in guarantees),
            "gap_premises": sum(bool(p["in_gap_of"]) for p in premises), "open": sum(p["kind"] == "open" for p in premises),
            "overdue": sum(p["overdue"] for p in premises), "owners_differ": sum(len(owners(p)) > 1 for p in premises),
            "no_honest_witness": sum(map(unwitnessed, guarantees)), "by_owner": {o: by_owner[o] for o in sorted(by_owner)}}


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
                                 "git alone. The gap is computed by the running checkout's Lean check "
                                 "(tools/verity/lean/check.py), on REV's locks. It reports what the locks record, no "
                                 "verdict.")
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
