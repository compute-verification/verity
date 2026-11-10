"""For each statement the export's Lean check fails as `shape.unstated`: the theorem that proves it, why the boundary
withholds that theorem's module, the withheld modules of its import closure, and what one boundary line would do.

    python unstated18.py EXPORT_BRANCH_CHECKOUT MAIN_LOCK.json [--json OUT.json]

Everything is read through the export's own code in the checkout given: `public_boundary.py` (the boundary's globs, the
Lean header reading through `tools/verity/lean/headers.py`, `judge`), `public_export.py` (`Held`, `rewrite_lock`,
`move_check.lean_declared`) and `public_lock.py`. Nothing is built and no file changes: a boundary line is tried on the
boundary's document in memory.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
MAIN_LOCK = json.loads(Path(sys.argv[2]).read_text())
OUT = Path(sys.argv[sys.argv.index("--json") + 1]) if "--json" in sys.argv else None

STATEMENTS = [
    *(f"Accounting.Guarantees.Compose.{n}" for n in
      ("CircuitCredit", "CircuitCreditSeeded", "CircuitWork", "CircuitWorkSeeded", "CreditOfWork")),
    *(f"Accounting.Guarantees.Sketch.{n}" for n in
      ("CommitThenProject", "Exact", "LittlewoodOfford", "NonCollision", "SmallCode", "TensorCoreNonCollision")),
    "Hash.Guarantees.EmbeddingReadBinding",
    *(f"Verity.Guarantees.{n}" for n in
      ("BlockDescription", "CKSize", "Counting", "CountingSum", "InferenceCap", "ServingBlockSum")),
]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


PE = load("unstated18_export", ROOT / "tools/verity/check/public_export.py")
PB, MOVE, PUBLIC_LOCK = PE.PB, PE.MOVE, PE.PUBLIC_LOCK
DOC = tomllib.loads((ROOT / "tools/verity/check/public_boundary.toml").read_text(encoding="utf-8"))
ALLOWED = set(PB.read_allowed(ROOT / "tools/verity/check/public_boundary_allowed.txt"))
tree = PB.Tree(ROOT)
B0 = PB.Boundary.load(DOC)
J0 = PB.judge(tree, B0)
MODULES, OWNER, _ = tree.lake
NAME = {f: m for m, f in MODULES.items()}
LEAN = sorted(f for f in tree.files if f.endswith(".lean"))
DECLARED = {f: {n.removeprefix("_root_.") for n in MOVE.lean_declared(tree.text(f))} for f in LEAN}


def proof_of(statement: str) -> str:
    """`X.Guarantees.G`'s guarantee, as the lock names it: `X.SecurityProofs.G` (check.py's `four_named`)."""
    parts = statement.split(".")
    at = parts.index("Guarantees")
    return ".".join([*parts[:at], "SecurityProofs", *parts[at + 1:]])


def covering(f: str) -> dict:
    """Every glob of the boundary that covers ``f``, by the table it is in."""
    one = lambda g: PB.globs([g]).fullmatch(f)  # noqa: E731
    return {
        "private": [f"private.{k}: {g}" for k, gs in DOC.get("private", {}).items() for g in gs if one(g)],
        "public": [f"public.{k}: {g}" for k, gs in DOC.get("public", {}).items() for g in gs if one(g)],
        "needed": [{"glob": g, "from": e.get("from", "*"), "via": e.get("via", "*"), "why": e.get("why", "")}
                   for e in DOC.get("needed", []) for g in e["paths"] if one(g)],
        "when_closed": [g for g in DOC.get("when_closed", []) if one(g)],
    }


def rule(f: str, judgment=J0) -> str:
    """Why the boundary leaves ``f`` out, in the order `judge` decides it: a `private` glob wins; else a `needed` glob
    covers it and no public file of the entry's groups needs it; else no glob covers it."""
    if f in judgment.public:
        return f"public ({judgment.public[f]})"
    c = covering(f)
    if c["private"]:
        return "a `private` glob: " + "; ".join(c["private"])
    if c["public"]:
        return "covered by " + "; ".join(c["public"]) + " yet not public (unexpected)"
    if c["needed"]:
        return "; ".join(f"`needed` glob `{n['glob']}` (from = {json.dumps(n['from'])}"
                         + (f", via = {json.dumps(n['via'])}" if n["via"] != "*" else "")
                         + f"; \"{n['why']}\"): exported only as far as a public file "
                         + ("" if n["from"] == "*" else f"of {json.dumps(n['from'])} ")
                         + "needs it, and none does" for n in c["needed"])
    if c["when_closed"]:
        return "`when_closed` and something it needs is not public"
    return "no glob covers it: every file no glob covers stays private"


def kind(f: str) -> str:
    c = covering(f)
    return "private" if c["private"] else "needed-unreached" if c["needed"] else "uncovered"


def imports(f: str) -> list[str]:
    return [e.target for e in tree.edges(f) if e.kind == "lean import"]


def closure(f: str) -> list[str]:
    """The Lean files ``f`` imports, directly or through one another, in the workspace's packages (itself left out)."""
    seen, todo = set(), [f]
    while todo:
        for t in imports(todo.pop()):
            if t not in seen and t != f:
                seen.add(t)
                todo.append(t)
    return sorted(seen)


def importers(f: str, among) -> list[str]:
    return sorted(g for g in among if g.endswith(".lean") and f in imports(g))


def try_line(files: list[str], group: str) -> dict:
    """The judgment with ``files`` added to the public group ``group``, against today's: what else goes public, the
    edges that open, and whether the export's lock filter keeps each lock."""
    doc = copy.deepcopy(DOC)
    doc["public"][group] = [*doc["public"][group], *files]
    b = PB.Boundary.load(doc)
    j = PB.judge(tree, b)
    old = {e.key for e in J0.edges}
    opened = [str(e) for e in j.edges if e.key not in old and e.key not in ALLOWED]
    return {"group": group, "boundary": b, "judgment": j, "added": sorted(set(j.public) - set(J0.public)),
            "lost": sorted(set(J0.public) - set(j.public)), "opened": opened}


def lock_after(judgment, lock_path: str = "Security/lean-audit.json", lock: dict | None = None):
    """(the lock as the export would write it, what it dropped) for the copied set of ``judgment``; a `ValueError`'s
    text when the filter refuses it."""
    copied = set(judgment.public)
    held = PE.Held(tree, copied)
    here = str(Path(lock_path).parent)
    lock = lock if lock is not None else json.loads(tree.text(lock_path))
    try:
        out, dropped = PUBLIC_LOCK.public(lock, lambda k, n: held.has(here, k, n))
    except ValueError as e:
        return None, str(e), held
    return out, dropped, held


# ---- today's export --------------------------------------------------------------------------------------------------
LOCK_HERE = json.loads(tree.text("Security/lean-audit.json"))
out0, dropped0, held0 = lock_after(J0)
assert out0 is not None, dropped0
kept0 = set(out0["guarantees"])

rows = []
for s in STATEMENTS:
    t = proof_of(s)
    entry = MAIN_LOCK["guarantees"].get(t)
    where = sorted(f for f, names in DECLARED.items() if t in names)
    s_where = sorted(f for f, names in DECLARED.items() if s in names)
    row = {"statement": s, "statement_file": s_where, "theorem": t, "in_main_lock": entry is not None,
           "in_branch_lock": t in LOCK_HERE["guarantees"],
           "signature": (entry or {}).get("signature"), "owner": (entry or {}).get("owner"),
           "premises": (entry or {}).get("premises"), "theorem_file": where}
    if len(where) != 1:
        row["verdict"] = "NOT ESTABLISHED"
        rows.append(row)
        continue
    f = where[0]
    cl = closure(f)
    withheld = [c for c in cl if c not in J0.public]
    row |= {"module": NAME[f], "rule": rule(f), "kind": kind(f), "closure": len(cl),
            "closure_withheld": [{"file": c, "module": NAME[c], "kind": kind(c), "rule": rule(c)} for c in withheld],
            "statement_public": all(x in J0.public for x in s_where),
            "statement_why": [J0.public.get(x) for x in s_where],
            "kept_today": t in kept0}
    rows.append(row)

# ---- one line per proving module, tried one at a time and then all together ----------------------------------------
by_file: dict[str, list[dict]] = {}
for r in rows:
    if r.get("module"):
        by_file.setdefault(r["theorem_file"][0], []).append(r)

trials = {}
for f, rs in by_file.items():
    per = {}
    for group in DOC["public"]:
        t = try_line([f], group)
        out, dropped, held = lock_after(t["judgment"])
        per[group] = {"added": t["added"], "lost": t["lost"], "opened": t["opened"],
                      "lock_refused": dropped if out is None else None,
                      "guarantees_kept": None if out is None else sorted(set(out["guarantees"]) - kept0),
                      "guarantees_lost": None if out is None else sorted(kept0 - set(out["guarantees"])),
                      "closure_still_withheld": [c for c in closure(f) if c not in t["judgment"].public]}
    trials[f] = per

result = {"root": str(ROOT), "public_files": len(J0.public), "open_edges_today": len(J0.edges),
          "guarantees_kept_today": len(kept0), "rows": rows, "trials": trials}
if OUT:
    OUT.write_text(json.dumps(result, indent=1, default=str) + "\n")

for r in rows:
    print(f"\n== {r['statement']}\n   theorem {r['theorem']} in {r['theorem_file']} (main lock: {r['in_main_lock']}, "
          f"branch lock: {r['in_branch_lock']}, kept by today's export: {r.get('kept_today')})")
    if not r.get("module"):
        continue
    print(f"   module {r['module']}: {r['rule']}")
    print(f"   statement in {r['statement_file']}: {r['statement_why']}")
    print(f"   closure {r['closure']} modules, {len(r['closure_withheld'])} withheld")
    for c in r["closure_withheld"]:
        print(f"     - {c['module']} [{c['kind']}]: {c['rule']}")
for f, per in trials.items():
    print(f"\n## line \"{f}\"")
    for group, t in per.items():
        ok = not t["opened"] and not t["lock_refused"] and not t["closure_still_withheld"] and not t["lost"]
        print(f"   in public.{group}: {'OK' if ok else 'no'}; +{len(t['added'])} files, {len(t['opened'])} edges open, "
              f"closure still withheld {len(t['closure_still_withheld'])}, lock "
              f"{'REFUSED' if t['lock_refused'] else '+' + str(len(t['guarantees_kept'])) + ' guarantees'}")
        if ok:
            print(f"      added: {t['added']}")
            print(f"      guarantees kept: {t['guarantees_kept']}")
        elif t["lock_refused"]:
            print(f"      {t['lock_refused'][:400]}")
