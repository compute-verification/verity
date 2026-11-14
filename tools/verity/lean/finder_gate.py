"""`finder_gate`: the landing gate on the finder rule's two lists, the walk's table (`finder-walk.json`) and the locks'
`choosing` sections (`check.py`'s `choosing_failures`), each of which only refuses more or lists less.

    python3 tools/verity/lean/finder_gate.py MAIN HEAD [--json]

The change judged is ``HEAD``'s against the commit below it (``LO``): the first parent of drive's merge of a PR (a
``Queue: #N`` merge), else its merge base with ``MAIN``, as `unread_gate.py` judges it. From git alone, with no build and
no clock, it fails, naming each, on a change that:

- adds a `pass` row (the walk stops at one without a finding);
- drops a `refuse` row, a head of `heads`, or a position of a head;
- adds a `fields` row, or drops a field of one (the walk reads only the listed fields of that constructor);
- adds a `choosing` entry to a lock (a definition listed that the check would refuse);
- moves a `choosing` entry's `due` later, or drops it.

A refusal the Lean check makes stands until the definition is cured, and a list the check reads only shrinks: what this
gate fails on is a person's decision, which the pull request shows to the lead who answers for it. Exit 0 when it passes,
1 when it fails (``--json``: `KIND`, `{"failures": [{"why", "fix"}]}`), 2 when it can't judge: a git error, or a table or
lock that is not well formed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finder_dates import CantJudge, commit_of, entries, git, locks_at

KIND = "lean-finder-gate/v1"
#: the subject of the queue's merge of one PR onto `next`
QUEUE_SUBJECT = "Queue: #"
TABLE = "tools/verity/lean/finder-walk.json"
ROWS = ("heads", "fields", "refuse", "pass")


def below(base: str, top: str) -> str:
    """The commit ``top``'s change is judged against: the tip it merged onto when it is the queue's merge of a PR, else its
    merge base with ``base``."""
    parents, _, subject = git("log", "-1", "--format=%P%x00%s", top).strip().partition("\0")
    if len(parents.split()) == 2 and subject.startswith(QUEUE_SUBJECT):
        return parents.split()[0]
    return git("merge-base", base, top).strip()


def table_at(rev: str) -> dict:
    """``rev``'s `finder-walk.json`, its rows as sets ({} where it has none)."""
    if TABLE not in git("ls-tree", "-r", "--name-only", rev, "--", TABLE).split("\n"):
        return {"heads": {}, "fields": {}, "refuse": set(), "pass": set()}
    try:
        t = json.loads(git("show", f"{rev}:{TABLE}"))
        out = {"heads": {h: set(ps) for h, ps in t["heads"].items()}, "fields": {c: set(fs) for c, fs in t["fields"].items()},
               "refuse": set(t["refuse"]), "pass": set(t["pass"])}
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


def lock_failures(lo: dict, hi: dict) -> list[dict]:
    out = []
    for (path, d), e in sorted(hi.items()):
        if (was := lo.get((path, d))) is None:
            out.append({"why": f"{path} lists {d} under `choosing` ({e.get('owner')}, {e.get('issue')}, due "
                               f"{e.get('due')}): a definition a finder picks by choice through, which the check would "
                               f"refuse", "fix": [f"restate {d} as a search of data the finder holds",
                                                  "or soundness lists it, which the pull request says"]})
        elif isinstance(was.get("due"), str) and (not isinstance(e.get("due"), str) or e["due"] > was["due"]):
            out.append({"why": f"{path} moves {d}'s `due` from {was['due']} to {e.get('due')}",
                        "fix": [f"keep {was['due']}", f"or {d}'s lead moves it, which the pull request says"]})
    return out


def gate(base: str, top: str) -> dict:
    base, top = commit_of(base), commit_of(top)
    lo = below(base, top)
    failures = table_failures(table_at(lo), table_at(top)) + lock_failures(entries(locks_at(lo)), entries(locks_at(top)))
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
