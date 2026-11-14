"""`python3 tools/verity/lean/posture.py`: each deployed guarantee's posture, one line each, read from the Lean locks.

For each guarantee a lock's `deployed` lists, the premises it rests on (`check.premise_closure`: its record's named
assumptions and open premises, through the guarantees it cites): its named assumptions, then its open premises with each
one's owner, issue and due, then what its witnesses take beyond those (the premises of its `witnesses` entry's theorems,
`RecDeploy.HonestFixture` for an honest fixture). Each premise
it rests on is marked against the trust list (`trust-list.json`):

- `on`: the list holds it for this guarantee;
- `gap`: off the list, and in the guarantee's `trust_gap` entry, the gap Daniel drives to zero;
- `pending`: off the list, and in `trust-list.json`'s `pending` for this guarantee, waiting on a ruling;
- `OFF`: off the list and in neither, which `tests/test_trust_list.py` fails on;
- `chooses`: for a guarantee resting on a collision premise (an entry of the list marked `collision`), a definition under
  it that still chooses its witness (`Classical.choose`, `Exists.choose`, `Classical.choice`, `Classical.epsilon`), as
  the lock's list of choosing definitions holds it (`CHOOSING`: {definition: {"owner", "issue", "due"}}; the ones a
  record reads under its collision premises are its `premises`' `CHOOSING`). Until the definition leaves that list, the
  guarantee doesn't rest on SHA-512's collision resistance alone: the collision is the chosen witness's, not one a
  finder computes, so the definition is shown among its open premises, with the entry's owner, issue and due.

The trust list is `trust-list.json`'s `list`: {premise: {"reason": one line, "only": [deployed guarantee, ...],
"collision": true}}, `only` present where the list holds a premise for those guarantees alone, `collision` on SHA-512's
collision resistance. Lean's kernel is on it too, and is no premise. Every tracked lock is read, the auditor's fixtures
aside (`check.FIXTURES`).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TRUST_LIST = HERE / "trust-list.json"
FIXTURES = "tools/verity/lean/tests/fixtures/"
# The lock section of definitions still by choice, and each record's `premises` key for those it reads under its
# collision premises: the Lean check writes both, and until it does no lock has them and no line shows one.
CHOOSING = "choosing"


def _check():
    spec = importlib.util.spec_from_file_location("lean_check_for_posture", HERE / "check.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def locks(root: Path = ROOT) -> dict[str, dict]:
    files = subprocess.run(["git", "-C", str(root), "ls-files", "*lean-audit.json"], check=True, capture_output=True,
                           text=True).stdout.split()
    return {f: json.loads((root / f).read_text()) for f in sorted(files)
            if f.endswith("lean-audit.json") and not f.startswith(FIXTURES)}


def on_list(trust: dict, premise: str, guarantee: str) -> bool:
    entry = (trust.get("list") or {}).get(premise)
    return isinstance(entry, dict) and ("only" not in entry or guarantee in entry["only"])


def mark(lock: dict, trust: dict, premise: str, guarantee: str) -> str:
    if on_list(trust, premise, guarantee):
        return "on"
    if premise in ((lock.get("trust_gap") or {}).get(guarantee) or []):
        return "gap"
    if premise in ((trust.get("pending") or {}).get(guarantee) or {}):
        return "pending"
    return "OFF"


def ticket(entry: dict | None) -> dict:
    return {k: (entry or {}).get(k) for k in ("owner", "issue", "due")}


def postures(locks: dict[str, dict], trust: dict, check) -> list[dict]:
    """One record per deployed guarantee: its `name`, `lock`, its `assumptions` and `open` premises ({premise: its mark},
    an open one with its `owner`, `issue` and `due`, a choosing definition under its collision premises among them) and
    what its `witnesses` take beyond them, each sorted by name."""
    collision = {p for p, e in (trust.get("list") or {}).items() if isinstance(e, dict) and e.get("collision")}
    out = []
    for path, lock in locks.items():
        named, opened, choosing = lock.get("assumptions") or {}, lock.get("open") or {}, lock.get(CHOOSING) or {}
        for g in check.name_list(lock.get("deployed")):
            rests = check.premise_closure(lock, g)
            reads = check.premise_closure(lock, g, (CHOOSING,)) & set(choosing) if rests & collision else set()
            w = (lock.get("witnesses") or {}).get(g) or {}
            theorems = [*check.name_list(w.get("honest")), *check.name_list([w.get("nontrivial")])]
            out.append({
                "name": g, "lock": path,
                "assumptions": {p: mark(lock, trust, p, g) for p in sorted(rests) if p in named},
                "open": dict(sorted({**{p: {"mark": mark(lock, trust, p, g), **ticket(opened.get(p))}
                                        for p in rests if p not in named},
                                     **{d: {"mark": "chooses", **ticket(choosing[d])} for d in reads}}.items())),
                "witnesses": sorted(set().union(*(check.premise_closure(lock, t) for t in theorems)) - rests),
            })
    return out


def line(p: dict) -> str:
    assumed = ", ".join(f"{n} [{m}]" for n, m in p["assumptions"].items()) or "none"
    opened = ", ".join(f"{n} [{e['mark']}] {e['owner'] or '-'} {e['issue'] or '-'} due {e['due'] or '-'}"
                       for n, e in p["open"].items()) or "none"
    return (f"{p['name']} ({p['lock']}): assumptions: {assumed}; open: {opened}; witnesses take: "
            f"{', '.join(p['witnesses']) or 'none'}")


def main(argv: list[str] | None = None) -> int:
    import argparse
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args(argv)
    for p in postures(locks(), json.loads(TRUST_LIST.read_text()), _check()):
        print(line(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
