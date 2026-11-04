"""`verity graph --json` of the checkout ROOT with its Lean families at their size once its modules are `module` files: a
lean-module or lean-lock task reads, of each module it reads, the module system's public outputs (`PUBLIC`: `olean`, and `ir`
where the module compiles code, here every one), and its other reads (an env's `axioms`) as listed, and a lean-module task
outputs all that `lean` writes for a `module` file (`PORTED`).  The graph's bytes on stdout, its size on stderr; exit 3 when
no Lean family lists here.

    python ported_listing.py ROOT
"""
from __future__ import annotations

import sys
from pathlib import Path

from verity_graph import contract, families, graph

PUBLIC = ("ir", "olean")
PORTED = ("olean", "ilean", "olean.server", "olean.private", "ir.sig", "ir")
LEAN = ("lean-module", "lean-lock")


def ported(lister):
    def tasks(root: Path, files: list[str]) -> list[dict]:
        out = []
        for t in lister(root, files):
            reads = t.get("reads", [])
            modules = sorted({r["task"] for r in reads if r["task"].startswith("lean-module:")})
            t = {**t, "reads": [*({"task": m, "output": o} for m in modules for o in PUBLIC),
                                *(r for r in reads if not r["task"].startswith("lean-module:"))]}
            if t["id"].startswith("lean-module:"):
                have = {o["name"]: o["path"] for o in t.get("outputs", [])}
                if "olean" not in have:
                    raise SystemExit(f"{t['id']} has no output olean, which this script's PORTED puts the others beside")
                t["outputs"] = [*t["outputs"], *({"name": n, "path": have["olean"].removesuffix("olean") + n}
                                                 for n in PORTED if n not in have)]
            out.append(t)
        return out
    return tasks


real = families.listers()
if "lean-module" not in real:
    sys.exit(3)
doc = graph.listing(Path(sys.argv[1]).resolve(), listers={**real, **{n: ported(real[n]) for n in LEAN if n in real}})
sys.stdout.buffer.write(contract.canonical(doc) + b"\n")
print(f"{len(doc['tasks'])} tasks, {sum(len(t['reads']) for t in doc['tasks'])} reads", file=sys.stderr)
