"""`finder_gate`: the landing gate on the finder rule's two lists, the walk's table (`finder-walk.json`) and the locks'
`choosing` sections (`check.py`'s `choosing_failures`), each of which only refuses more or lists less.

    python3 tools/verity/lean/finder_gate.py MAIN HEAD [--json]

Drive runs it at the ready mark's preflight as it is at ``MAIN`` (`tools/check/drive.toml`'s `landing_gates`), a copy of
this file alone, so it loads nothing beside it. The change judged is ``HEAD``'s against its merge base with ``MAIN``,
whatever its merges' subjects say. From git alone, with no build and no clock, it fails, naming each, on a change that:

- adds a `pass` row (the walk stops at one without a finding);
- drops a `refuse` row, a head of `heads`, or a position of a head;
- adds a `fields` row, or drops a field of one (the walk reads only the listed fields of that constructor);
- adds a `types` row (a head's argument of that type counts as a finder's position);
- adds a `choosing` entry to a lock (a definition listed that the check would refuse), but for one re-keyed onto a
  dropped entry whose definition's `reads` hash it has;
- moves a `choosing` entry's `due` later, or drops it, or changes its `owner` or `issue`;
- has a guarantee's record newly reach a `choosing` entry whose `due` is before the day of ``HEAD``'s commit (newly
  against ``HEAD``'s first parent where ``HEAD`` is a merge).

A refusal the Lean check makes stands until the definition is cured, and a list the check reads only shrinks: what this
gate fails on is a person's decision, which the pull request shows to the lead who answers for it. Exit 0 when it passes,
1 when it fails (``--json``: `KIND`, `{"failures": [{"why", "fix"}]}`), 2 when it can't judge: a git error, or a table or
lock that is not well formed.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys

KIND = "lean-finder-gate/v1"
POLICY = "lean-audit.json"
#: the auditor's controls, whose locks are not the repository's
FIXTURES = "tests/tools/verity/lean/fixtures/"
TABLE = "tools/verity/lean/finder-walk.json"
MOVES = "tools/move/"
#: `check.py`'s `GENERATED`, which its test holds this to
GENERATED = re.compile(r"^(_.*|match_\d+|proof_\d+|eq_\d+|eq_def|mk|rec|recOn|casesOn|noConfusion|noConfusionType|"
                       r"below|brecOn|binductionOn|ibelow|ctorIdx|inj|injEq|sizeOf_spec)$")


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


def table_at(rev: str) -> dict:
    """``rev``'s `finder-walk.json`, its rows as sets ({} where it has none)."""
    if TABLE not in git("ls-tree", "-r", "--name-only", rev, "--", TABLE).split("\n"):
        return {"heads": {}, "fields": {}, "refuse": set(), "pass": set(), "types": set()}
    try:
        t = json.loads(git("show", f"{rev}:{TABLE}"))
        out = {"heads": {h: set(ps) for h, ps in t["heads"].items()}, "fields": {c: set(fs) for c, fs in t["fields"].items()},
               "refuse": set(t["refuse"]), "pass": set(t["pass"]), "types": set(t.get("types", []))}
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise CantJudge(f"{TABLE} at {rev}: not the walk's table ({type(e).__name__}: {e})") from None
    if not all(isinstance(p, int) for ps in out["heads"].values() for p in ps):
        raise CantJudge(f"{TABLE} at {rev}: a head's positions are not numbers")
    return out


def table_failures(lo: dict, hi: dict) -> list[dict]:
    fix = "keep the row, or have soundness rule on the change and say so in the pull request"
    out = [{"why": f"{TABLE} adds the `pass` row {c}: the walk would stop there without a finding", "fix": [fix]}
           for c in sorted(hi["pass"] - lo["pass"])]
    out += [{"why": f"{TABLE} drops the `refuse` row {c}: the walk would no longer refuse it", "fix": [fix]}
            for c in sorted(lo["refuse"] - hi["refuse"])]
    out += [{"why": f"{TABLE} adds the `types` row {c}: a head's argument of that type would count as a finder's",
             "fix": [fix]} for c in sorted(hi["types"] - lo["types"])]
    for h, ps in sorted(lo["heads"].items()):
        if h not in hi["heads"]:
            out.append({"why": f"{TABLE} drops the head {h}: no finder handed to it would be walked", "fix": [fix]})
        elif gone := sorted(ps - hi["heads"][h]):
            out.append({"why": f"{TABLE} drops the position{'s' * (len(gone) > 1)} {', '.join(map(str, gone))} of {h}: "
                               f"no finder there would be walked", "fix": [fix]})
    for c, fs in sorted(hi["fields"].items()):
        if c not in lo["fields"]:
            out.append({"why": f"{TABLE} adds the `fields` row {c}: the walk would read only {', '.join(sorted(fs))} "
                               f"of it", "fix": [fix]})
        elif gone := sorted(lo["fields"][c] - fs):
            out.append({"why": f"{TABLE} narrows the `fields` row {c}: the walk would no longer read "
                               f"{', '.join(gone)}", "fix": [fix]})
    return out


def parent(name: str) -> str:
    parts = name.split(".")
    while len(parts) > 1 and GENERATED.match(parts[-1]):
        parts.pop()
    return ".".join(parts)


def renamed(name: str, moved: dict[str, str]) -> str:
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        if (p := ".".join(parts[:i])) in moved:
            return ".".join([moved[p], *parts[i:]])
    return name


def moves_of(lo: str, top: str) -> dict[str, str]:
    """{old: new} of the moves files under `tools/move/` the change adds or changes."""
    out = {}
    for p in git("diff", "--name-only", "-z", "--diff-filter=AM", lo, top, "--", MOVES).split("\0"):
        if not (p.endswith(".json") and "moves" in p.rpartition("/")[2]):
            continue
        try:
            doc = json.loads(git("show", f"{top}:{p}"))
        except ValueError as e:
            raise CantJudge(f"{p} at {top}: not JSON ({e})") from None
        if not isinstance(doc, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in doc.items()):
            raise CantJudge(f"{p} at {top}: not a map from an old name to its new one")
        out |= doc
    return out


def hash_of(lock: dict, d: str) -> str | None:
    """The `reads` hash of ``d``'s definition, its declaration's for a generated constant."""
    for rec in (lock.get("reads") or {}).values():
        defs = rec.get("definitions") if isinstance(rec, dict) else None
        if isinstance(defs, dict) and (h := defs.get(d) or defs.get(parent(d))):
            return h
    return None


def moved_keys(lo: dict[str, dict], hi: dict[str, dict]) -> dict[tuple[str, str], str]:
    """{(lock, new key): old key} of the entries re-keyed onto a dropped key whose `reads` hash at ``lo`` is the new key's,
    as built or under the old names (the `moved` that `check.py --update --moved` writes); one new key to each old one."""
    elo, ehi = entries(lo), entries(hi)
    out, taken = {}, set()
    for path, d in sorted(set(ehi) - set(elo)):
        m = ehi[(path, d)].get("moved")
        m = m if isinstance(m, dict) else {}
        for k in [m.get("as")] + sorted(k for p, k in elo if p == path):
            if (path, k) not in elo or (path, k) in ehi or (path, k) in taken or not (h := hash_of(lo[path], k)):
                continue
            if h == hash_of(hi[path], d) or (k == m.get("as") and h == m.get("sha256")):
                out[(path, d)] = k
                taken.add((path, k))
                break
    return out


def lock_failures(lo: dict[str, dict], hi: dict[str, dict]) -> list[dict]:
    elo, ehi, moved = entries(lo), entries(hi), moved_keys(lo, hi)
    out = []
    for (path, d), e in sorted(ehi.items()):
        if (was := elo.get((path, moved.get((path, d), d)))) is None:
            out.append({"why": f"{path} lists {d} under `choosing` ({e.get('owner')}, {e.get('issue')}, due "
                               f"{e.get('due')}): a definition a finder picks by choice through, which the check would "
                               f"refuse", "fix": [f"restate {d} as a search of data the finder holds",
                                                  "or soundness lists it, which the pull request says",
                                                  "or, for a rename, write the lock with `check.py --update --moved`"]})
            continue
        if isinstance(was.get("due"), str) and (not isinstance(e.get("due"), str) or e["due"] > was["due"]):
            out.append({"why": f"{path} moves {d}'s `due` from {was['due']} to {e.get('due')}",
                        "fix": [f"keep {was['due']}", f"or {d}'s lead moves it, which the pull request says"]})
        for k in ("owner", "issue"):
            if e.get(k) != was.get(k):
                out.append({"why": f"{path} changes {d}'s `{k}` from {was.get(k)} to {e.get(k)}",
                            "fix": [f"keep {was.get(k)}", f"or {d}'s lead changes it, which the pull request says"]})
    return out


def reach(locks: dict[str, dict], moves: dict[str, str] | None = None) -> set[tuple[str, str, str]]:
    """{(lock, guarantee, definition)} of each `choosing` entry a guarantee's record reaches, under ``moves``."""
    return {(path, renamed(g, moves or {}), renamed(d, moves or {})) for path, lock in locks.items()
            for g, rec in (lock.get("guarantees") or {}).items() if isinstance(rec, dict)
            for d in ((rec.get("premises") or {}).get("choosing") or [])}


def gone(moves: dict[str, str], locks: dict[str, dict]) -> dict[str, str]:
    """The moves whose old name ``locks`` no longer hold, as a guarantee, an entry or a definition a record reaches, nor
    anything under it: a move whose old name is still there renames nothing."""
    names = {*(n for _, n in entries(locks)), *(x for _, g, d in reach(locks) for x in (g, d)),
             *(g for lock in locks.values() for g in (lock.get("guarantees") or {}))}
    held = {".".join(p[:i]) for n in names for p in [n.split(".")] for i in range(1, len(p) + 1)}
    return {old: new for old, new in moves.items() if old not in held}


def overdue_failures(lo: dict[str, dict], hi: dict[str, dict], moves: dict[str, str], today: str) -> list[dict]:
    moves = gone(moves, hi)
    back = {(path, old): d for (path, d), old in moved_keys(lo, hi).items()}
    was = reach(lo, moves) | {(p, renamed(g, moves), back.get((p, d), d)) for p, g, d in reach(lo)}
    ehi = entries(hi)
    out = []
    for path, g, d in sorted(reach(hi) - was):
        if isinstance(due := (ehi.get((path, d)) or {}).get("due"), str) and due < today:
            out.append({"why": f"{path}: {g} newly reaches {d}, whose `choosing` entry was due {due}",
                        "fix": [f"cure {d} first", f"or state {g} without it"]})
    return out


def gate(base: str, top: str) -> dict:
    base, top = commit_of(base), commit_of(top)
    lo = git("merge-base", base, top).strip()
    locks_lo, locks_hi = locks_at(lo), locks_at(top)
    # the members below a merge onto `next` were each judged at their own merge, on its day
    parents = git("rev-list", "--parents", "-n", "1", top).split()[1:]
    since = parents[0] if len(parents) > 1 else lo
    when = int(git("show", "-s", "--format=%ct", top))
    today = datetime.datetime.fromtimestamp(when, datetime.timezone.utc).date().isoformat()
    failures = (table_failures(table_at(lo), table_at(top)) + lock_failures(locks_lo, locks_hi)
                + overdue_failures(locks_at(since) if since != lo else locks_lo, locks_hi, moves_of(since, top), today))
    return {"kind": KIND, "base": base, "below": lo, "top": top, "failures": failures}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="finder_gate", description=__doc__.split("\n\n")[0])
    ap.add_argument("base")
    ap.add_argument("top")
    ap.add_argument("--json", action="store_true", help=f"print {KIND}")
    a = ap.parse_args(argv)
    try:
        doc = gate(a.base, a.top)
    except CantJudge as e:
        print(f"finder gate: can't judge: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(doc, indent=1))
    else:
        print(f"finder gate {doc['below'][:12]}..{doc['top'][:12]} (base {doc['base'][:12]}): "
              f"{len(doc['failures'])} failure(s)")
        print("".join(f"  {f['why']}\n" + "".join(f"    fix: {x}\n" for x in f["fix"]) for f in doc["failures"]), end="")
    return 1 if doc["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
