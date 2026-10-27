"""The size of a package's trusted text, per area: what LEAN-67 tracks over time.

    python tools/verity/lean/trusted_size.py [PKG] [--facts FACTS.json | --run RUN] [--unread] [--json]

PKG (default `Security`) is a Lake package with a `lean-audit.json`; its trusted text is the modules under `--prefix`
(default `Definitions`), grouped by their next component, the area (`Definitions.Pouw.Basic` is in `Pouw`). For each area
it prints the files, their lines (and the lines that are neither blank nor comment nor docstring), the theorems written
there, the definitions, how many of them a listed statement reads, and how many guarantees read the area. The
definitions and what reads them come from a run of the Lean check on PKG: its `facts.json` (`--facts`), or that of `--run
RUN`, a records run (`lean_changed.py --records PKG`) the evidence store holds. A definition is one of `spec_definitions`
(projections and constructors go with their structure), read when a listed guarantee's statement reads it, followed through
`meaning`, so transitively: when the lock's `reads` lists it (`--lock`, default PKG's; a run's own with `--run`), or the
lock of a package requiring PKG by path does (`check.read_by_dependents`), as the audit's `unread_spec` counts them.
Without facts the definitions are counted from the sources and the read ones are the lock's (projections and constructors
included), both marked `~`. `--unread` lists the unread ones per module.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
_spec = importlib.util.spec_from_file_location("lean_check", HERE / "check.py")
C = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("lean_check", C)
_spec.loader.exec_module(C)

MODIFIERS = r"(?:@\[[^\]]*\]\s*)*(?:(?:private|protected|noncomputable|partial|unsafe|nonrec)\s+)*"
THEOREM = re.compile(rf"^\s*{MODIFIERS}(?:theorem|lemma)\s", re.MULTILINE)
DEFINITION = re.compile(rf"^\s*{MODIFIERS}(?:def|abbrev|structure|inductive|class|instance|opaque)\s", re.MULTILINE)


def code_lines(text: str) -> int:
    """The lines of Lean source ``text`` holding something besides blanks, `--` comments and `/- … -/` blocks (docstrings
    and nested comments included); a string literal holding a comment opener is rare enough to count wrong."""
    n, depth = 0, 0
    for line in text.splitlines():
        i, code = 0, False
        while i < len(line):
            if depth:
                if line.startswith("/-", i):
                    depth, i = depth + 1, i + 2
                elif line.startswith("-/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            elif line.startswith("/-", i):
                depth, i = 1, i + 2
            elif line.startswith("--", i):
                break
            else:
                code |= not line[i].isspace()
                i += 1
        n += code
    return n


def area_of(prefix: str, module: str) -> str | None:
    """The area of ``module`` under ``prefix`` (`Definitions.Pouw.Basic` → `Pouw`), `(root)` for ``prefix`` itself."""
    if module == prefix:
        return "(root)"
    return module[len(prefix) + 1:].split(".")[0] if module.startswith(prefix + ".") else None


def fetch_run(run: str, pkg: Path) -> tuple[Path, Path]:
    """``run``'s `facts.json` and lock for ``pkg`` from the evidence store (its `run_files` tree), in a temporary
    directory."""
    shown = json.loads(subprocess.run(["research", "data", "show", run, "--json"], capture_output=True, text=True,
                                      check=True, cwd=ROOT).stdout)
    tree = shown.get("outputs", {}).get("run_files")
    if not tree:
        raise SystemExit(f"trusted_size: {run} has no run_files in the store")
    want = [f"lean-audit/audit-{pkg.name}/{f}" for f in ("facts.json", C.POLICY)]
    out = Path(tempfile.mkdtemp(prefix="trusted-size-"))
    subprocess.run(["research", "data", "fetch", tree, "--to", str(out), *(x for w in want for x in ("--path", w))],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    if not all((out / w).is_file() for w in want):
        raise SystemExit(f"trusted_size: {run}'s outputs hold no {' or '.join(want)}: was it a records run of {pkg}?")
    return out / want[0], out / want[1]


def measure(pkg: Path, prefix: str, facts: dict | None = None, lock: dict | None = None) -> dict:
    """{area: its row} and {module: its unread definitions} for ``pkg``'s modules under ``prefix``: the definitions from
    ``facts`` (a Lean check's `facts.json` for ``pkg``), else from the sources; the read ones from ``lock`` (default
    ``pkg``'s `lean-audit.json`) and the locks of the packages requiring ``pkg`` by path."""
    modules = {m: rel for m, rel in C.lean_modules(pkg).items() if area_of(prefix, m)}
    read_by: dict[str, set[str]] = defaultdict(set)
    read: dict[str, set[str]] = defaultdict(set)
    for m, g in C.as_guarantees(lock if lock is not None else C.load(pkg)).get("reads", {}).items():
        read[m] |= {C.parent(n) for n in C.definitions(g)}
        read_by[m] |= set(g.get("guarantees", []))
    elsewhere = C.read_by_dependents(pkg, ROOT)
    defs: dict[str, set[str]] | None = None
    if facts is not None:
        defs = defaultdict(set)
        for d in facts.get("spec_definitions", []):
            if d["module"] in modules and C.parent(d["name"]) == d["name"]:
                defs[d["module"]].add(d["name"])
    rows: dict[str, dict] = {}
    unread: dict[str, list[str]] = {}
    for m, rel in sorted(modules.items()):
        text = (pkg / rel).read_text(errors="replace")
        r = rows.setdefault(area_of(prefix, m), {"files": 0, "lines": 0, "code": 0, "theorems": 0, "definitions": 0,
                                                 "read": 0, "unread": None, "guarantees": set()})
        r["files"] += 1
        r["lines"] += len(text.splitlines())
        r["code"] += code_lines(text)
        r["theorems"] += len(THEOREM.findall(text))
        r["guarantees"] |= read_by.get(m, set())
        if defs is None:
            r["definitions"] += len(DEFINITION.findall(text))
            r["read"] += len(read.get(m, ()))
            continue
        mine = defs.get(m, set())
        got = {d for d in mine if d in read.get(m, ()) or d in elsewhere}
        r["definitions"] += len(mine)
        r["read"] += len(got)
        if left := sorted(mine - got):
            unread[m] = left
    for r in rows.values():
        r["unread"] = r["definitions"] - r["read"] if defs is not None else None
        r["guarantees"] = len(r["guarantees"])
    return {"exact": defs is not None, "areas": rows, "unread": unread}


def table(m: dict) -> str:
    cols = ["files", "lines", "code", "theorems", "definitions", "read", "unread", "guarantees"]
    rows = sorted(m["areas"].items(), key=lambda kv: -kv[1]["lines"])
    total = {c: sum((r[c] or 0) for _, r in rows) for c in cols}
    mark = "" if m["exact"] else "~"
    out = ["| area | " + " | ".join(cols) + " | read % |", "|" + "---|" * (len(cols) + 2)]
    for a, r in [*rows, ("total", total)]:
        cells = [str(r[c]) if c not in ("definitions", "unread") else f"{mark}{r[c] if r[c] is not None else '-'}"
                 for c in cols]
        pct = f"{100 * r['read'] / r['definitions']:.0f}" if r["definitions"] and m["exact"] else "-"
        out.append(f"| {a} | " + " | ".join(cells) + f" | {pct} |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pkg", nargs="?", default="Security", help="the Lake package (default Security)")
    ap.add_argument("--prefix", default="Definitions", help="the trusted modules' prefix (default Definitions)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--facts", type=Path, help="a Lean check's facts.json for PKG")
    src.add_argument("--run", help="a records run of PKG whose facts.json and lock the evidence store holds")
    ap.add_argument("--lock", type=Path, help="the lock whose reads count (default PKG's lean-audit.json)")
    ap.add_argument("--unread", action="store_true", help="list the unread definitions per module")
    ap.add_argument("--json", action="store_true", help="print the measurement as JSON")
    a = ap.parse_args(argv)
    pkg = (ROOT / a.pkg) if not Path(a.pkg).is_absolute() else Path(a.pkg)
    facts, lock = fetch_run(a.run, pkg) if a.run else (a.facts, a.lock)
    m = measure(pkg, a.prefix, json.loads(facts.read_text()) if facts else None,
                json.loads(lock.read_text()) if lock else None)
    if a.json:
        print(json.dumps(m, indent=1, sort_keys=True))
        return 0
    print(table(m))
    if not m["exact"]:
        print("\n~ no facts given: definitions counted from the sources, read ones from the lock, unread not known")
    if a.unread:
        for mod, ds in m["unread"].items():
            print(f"\n{mod} ({len(ds)}): " + ", ".join(ds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
