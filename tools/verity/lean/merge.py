"""`merge.py`: git's merge driver for `lean-audit.json`, so that two branches that each add guarantees merge without a rebuild.

    python tools/verity/lean/merge.py BASE OURS THEIRS [PATH]   what git runs (`%O %A %B %P`): the merge is written to OURS
    python tools/verity/lean/merge.py --lean BASE OURS THEIRS [PATH]   the same for a `.lean` file (`lean-imports`)
    python tools/verity/lean/merge.py --install [REPO]         register both drivers in the repository's git config, once per clone
    python tools/verity/lean/merge.py --normalize FILE...      recompute each `reads` digest from its definitions

`.gitattributes` names this driver for every `lean-audit.json`, and git runs it when both sides changed the file. A record is
a map of entries by name: the guarantees, the modules under `reads`, the dependencies, and the policy's own sections. Each
side's change to an entry (added, changed or removed) applies, and an entry both sides changed differently refuses the merge.
A guarantee's record (its `owner` aside, which merges on its own) and a dependency's digest are one entry each. Under `reads`, which two branches adding guarantees that
read one module both change, the module's definitions merge the same way, its readers (`guarantees`) as sets, and its digest
is recomputed from the merged definitions (`audit.module_digest`). The merge also refuses when one side stops reading a
definition in a module the other side changed: the record does not say which guarantees read which definition. A side
written before guarantees were called that (`pins`) is read under today's names (`audit.as_guarantees`).

Exit 0: merged. Exit 1: refused, the entries named on stderr and OURS left as it was, which git reports as a conflict; then
regenerate the record on the merged tree (`check.py --update`), as without the driver. A merged record is a candidate, never
a verdict: `check`'s Lean audit rebuilds every record of the merged tree and fails on any difference, as for any record.

The layer rules (`layers`), which two branches that each widen a layer for a guarantee both change, merge field by field: `imports`
and `except` as lists, `why` as a list of its `; ` clauses (`layers`). For a `.lean` file (`--lean`, `lean-imports` in
`.gitattributes`), git's own merge runs and only a conflict all of whose sides are `import` lines is merged, as a list: two branches
that each add a guarantee's module to `Guarantees.lean` merge, and any other conflict stays git's.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("lean_audit_for_merge", HERE / "check.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

#: the driver's name in `.gitattributes` and git's config
NAME = "lean-audit"
#: the `.lean` files' driver (`--lean`)
LEAN = "lean-imports"
#: the sections `check.py --update` writes in sorted order; the others keep ours' order, with theirs' new entries after it
SORTED = {"assumptions", "guarantees", "reads", "dependencies", "compile_time"}
#: sections whose entries are maps merged entry by entry in turn (`upstream`'s `scan` and `watch`, `shape`'s `properties`,
#: `frozen` and `conjectures`)
NESTED = {"upstream", "shape"}
MISSING = object()


def one(path: str, o, a, b, why: list[str]):
    """The three-way merge of one value: a side's change applies, and a value both sides changed differently is refused."""
    if a == b:
        return a
    if a == o:
        return b
    if b == o:
        return a
    why.append(f"{path}: " + ("removed on one side and changed on the other" if a is MISSING or b is MISSING else
                              "added on both sides, differently" if o is MISSING else "changed on both sides, differently"))
    return a


def keys(section: str, a: dict, b: dict) -> list:
    ks = list(a) + [k for k in b if k not in a]
    return sorted(ks) if section in SORTED else ks


def entries(path: str, o: dict, a: dict, b: dict, why: list[str], depth: int = 1) -> dict:
    """Maps merged entry by entry; ``depth`` > 1 merges entries that are maps entry by entry too."""
    out = {}
    for k in keys(path, a, b):
        x, y, z = o.get(k, MISSING), a.get(k, MISSING), b.get(k, MISSING)
        if depth > 1 and isinstance(y, dict) and isinstance(z, dict) and (x is MISSING or isinstance(x, dict)):
            v = entries(f"{path}[{k}]", {} if x is MISSING else x, y, z, why, depth - 1)
        else:
            v = one(f"{path}[{k}]", x, y, z, why)
        if v is not MISSING:
            out[k] = v
    return out


def guarantees(o: dict, a: dict, b: dict, why: list[str]) -> dict:
    """The guarantees entry by entry, each one's `owner` apart from the rest of its record: a branch that restates a guarantee
    and one that gives it an owner merge."""
    out = {}
    for k in keys("guarantees", a, b):
        x, y, z = o.get(k, MISSING), a.get(k, MISSING), b.get(k, MISSING)
        if MISSING in (y, z) or not all(isinstance(r, dict) for r in (y, z, {} if x is MISSING else x)):
            v = one(f"guarantees[{k}]", x, y, z, why)
        else:
            v = one(f"guarantees[{k}]", x if x is MISSING else A.unowned(x), A.unowned(y), A.unowned(z), why)
            who = one(f"guarantees[{k}].owner", MISSING if x is MISSING else x.get("owner", MISSING), y.get("owner", MISSING),
                      z.get("owner", MISSING), why)
            v = v if who is MISSING else {"owner": who} | v
        if v is not MISSING:
            out[k] = v
    return out


def core(group):
    """What a `reads` entry says, its digest (which it determines) aside."""
    return None if group is None else (A.definitions(group), sorted(group.get("guarantees", [])))


def group(m: str, x: dict, y: dict, z: dict, why: list[str]) -> dict:
    """One module's `reads` entry, both sides having changed it: definitions entry by entry, readers as sets."""
    xd, yd, zd = A.definitions(x), A.definitions(y), A.definitions(z)
    defs = {}
    for d in sorted(set(xd) | set(yd) | set(zd)):
        p, q, r = xd.get(d, MISSING), yd.get(d, MISSING), zd.get(d, MISSING)
        v = one(f"reads[{m}].definitions[{d}]", p, q, r, why)
        if v is not MISSING:
            defs[d] = v
        elif p is not MISSING and (q is MISSING) != (r is MISSING):
            why.append(f"reads[{m}].definitions[{d}]: no longer read on one side, while the other side changed what the "
                       f"guarantees read in {m}; the record does not say which guarantees read {d}")
    if len({len(h) for h in defs.values() if isinstance(h, str)}) > 1:
        why.append(f"reads[{m}]: its definitions would mix 32-bit and SHA-256 hashes (one side's record was rehashed)")
    xp, yp, zp = set(x.get("guarantees", [])), set(y.get("guarantees", [])), set(z.get("guarantees", []))
    return {"definitions": defs, "guarantees": sorted((yp & zp) | (yp - xp) | (zp - xp))}


def reads(o: dict, a: dict, b: dict, why: list[str]) -> dict:
    out = {}
    for m in sorted(set(a) | set(b)):
        x, y, z = o.get(m), a.get(m), b.get(m)
        if core(y) == core(z):
            g = y
        elif core(y) == core(x):
            g = z
        elif core(z) == core(x):
            g = y
        elif y is None or z is None:
            why.append(f"reads[{m}]: no longer read on one side, while the other side changed what the guarantees read there")
            continue
        else:
            g = group(m, x or {}, y, z, why)
        if g is not None:
            defs = A.definitions(g)
            out[m] = {"digest": A.module_digest(defs), "definitions": dict(sorted(defs.items())),
                      "guarantees": sorted(g.get("guarantees", []))}
    return out


def items(o: list, a: list, b: list) -> list:
    """Two sides' changes to a list of distinct items: ours with what theirs added, each after the item theirs has before it,
    less what either side removed."""
    out = [i for i in a if i in b or i not in o]
    for n, i in enumerate(b):
        if i not in a and i not in o:
            before = next((p for p in reversed(b[:n]) if p in out), None)
            out.insert(0 if before is None else out.index(before) + 1, i)
    return out


def layers(o: dict, a: dict, b: dict, why: list[str]) -> dict:
    """The layer rules entry by entry; a rule both sides changed merges field by field: `imports` and `except` as lists of
    items, and `why` as a list of its `; ` clauses, so two branches that each widen a layer for a guarantee merge.  A merged
    rule widens a layer only by what one side's own audit passed with; the merged tree's audit judges the result."""
    out = {}
    for k in keys("layers", a, b):
        x, y, z = o.get(k, MISSING), a.get(k, MISSING), b.get(k, MISSING)
        if y == z or x == y or x == z or not all(isinstance(r, dict) for r in (x, y, z)):
            v = one(f"layers[{k}]", x, y, z, why)
        else:
            v = {}
            for f in keys("", y, z):
                p, q, r = x.get(f, MISSING), y.get(f, MISSING), z.get(f, MISSING)
                if f in ("imports", "except") and all(isinstance(s, list) for s in (p, q, r)):
                    v[f] = items(p, q, r)
                elif f == "why" and all(isinstance(s, str) for s in (p, q, r)):
                    v[f] = "; ".join(items(p.split("; "), q.split("; "), r.split("; ")))
                elif (w := one(f"layers[{k}].{f}", p, q, r, why)) is not MISSING:
                    v[f] = w
        if v is not MISSING:
            out[k] = v
    return out


def merge(base: dict, ours: dict, theirs: dict) -> tuple[dict, list[str]]:
    """(the merged record, why it is refused: empty when it merged).  A pure function of the three records."""
    why: list[str] = []
    out = {}
    for k in keys("", ours, theirs):
        o, a, b = base.get(k, MISSING), ours.get(k, MISSING), theirs.get(k, MISSING)
        if k == "reads":
            v = reads({} if o is MISSING else o, {} if a is MISSING else a, {} if b is MISSING else b, why)
        elif k == "guarantees" and isinstance(a, dict) and isinstance(b, dict) and (o is MISSING or isinstance(o, dict)):
            v = guarantees({} if o is MISSING else o, a, b, why)
        elif k == "layers" and isinstance(a, dict) and isinstance(b, dict) and (o is MISSING or isinstance(o, dict)):
            v = layers({} if o is MISSING else o, a, b, why)
        elif isinstance(a, dict) and isinstance(b, dict) and (o is MISSING or isinstance(o, dict)):
            v = entries(k, {} if o is MISSING else o, a, b, why, 2 if k in NESTED else 1)
        else:
            v = one(k, o, a, b, why)
        if v is not MISSING:
            out[k] = v
    kept = set(out.get("guarantees", {}))
    for m, g in out.get("reads", {}).items():
        if not g["definitions"] or not g["guarantees"]:
            why.append(f"reads[{m}]: the merge leaves it without {'definitions' if not g['definitions'] else 'readers'}")
        if extra := sorted(set(g["guarantees"]) - kept):
            why.append(f"reads[{m}]: read by {', '.join(extra)}, which the merged record does not list as guarantees")
    return out, why


def _imports(lines: list[str]) -> bool:
    return all(s.startswith("import ") or not s.strip() for s in lines)


def lean(base: Path, ours: Path, theirs: Path) -> tuple[str, int]:
    """(the merged text, the number of conflicts left) of a `.lean` file: git's own merge (`git merge-file`), then each conflict
    whose three sides are only `import` lines merged as lists of imports (`items`), so two branches that each add a module to
    an import list merge.  Any other conflict is left as git writes it."""
    p = subprocess.run(["git", "merge-file", "-p", "--diff3", "-L", "ours", "-L", "base", "-L", "theirs", str(ours), str(base),
                        str(theirs)], capture_output=True, text=True)
    if p.returncode < 0 or p.returncode > 127:
        raise ValueError(f"git merge-file failed: {p.stderr.strip()}")
    out, left, lines, n = [], 0, p.stdout.splitlines(keepends=True), 0
    while n < len(lines):
        if not lines[n].startswith("<<<<<<< ours"):
            out.append(lines[n])
            n += 1
            continue
        mid = next(i for i in range(n, len(lines)) if lines[i].startswith("||||||| base"))
        sep = next(i for i in range(mid, len(lines)) if lines[i].rstrip("\n") == "=======")
        end = next(i for i in range(sep, len(lines)) if lines[i].startswith(">>>>>>> theirs"))
        a, o, b = lines[n + 1:mid], lines[mid + 1:sep], lines[sep + 1:end]
        if _imports(a) and _imports(o) and _imports(b):
            out += [s for s in items([s for s in o if s.strip()], [s for s in a if s.strip()], [s for s in b if s.strip()])]
        else:
            out += lines[n:end + 1]
            left += 1
        n = end + 1
    return "".join(out), left


def load(path: Path) -> dict:
    text = path.read_text()
    return A.as_guarantees(json.loads(text)) if text.strip() else {}


def install(repo: Path) -> None:
    """Register the driver in ``repo``'s git config, as this file's path from the top of the worktree (git runs drivers
    there)."""
    top = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=repo, capture_output=True, text=True,
                              check=True).stdout.strip())
    driver = f"python3 {Path(os.path.relpath(HERE / 'merge.py', top)).as_posix()} %O %A %B %P"
    for name, what, argv in ((NAME, "lean-audit.json: each side's guarantees, reads and dependencies", driver),
                             (LEAN, ".lean: each side's imports", driver.replace(" %O", " --lean %O"))):
        for k, v in (("name", f"{what} (tools/verity/lean/merge.py)"), ("driver", argv)):
            subprocess.run(["git", "config", f"merge.{name}.{k}", v], cwd=repo, check=True)
        print(f"merge.py: merge.{name}.driver = {argv} (in {top})")


def normalize(path: Path) -> int:
    """Recompute each `reads` digest of the record at ``path`` from its definitions; the number that changed."""
    doc = load(path)
    n = 0
    for g in doc.get("reads", {}).values():
        if g.get("digest") != (d := A.module_digest(A.definitions(g))):
            g["digest"], n = d, n + 1
    path.write_text(A.text(doc))
    return n


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="merge.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*", type=Path, help="BASE OURS THEIRS [PATH], as git passes them; with --normalize, records")
    ap.add_argument("--install", nargs="?", const=Path.cwd(), type=Path, metavar="REPO", help="register the driver in REPO")
    ap.add_argument("--normalize", action="store_true", help="recompute the `reads` digests of each record named")
    ap.add_argument("--lean", action="store_true", help="BASE OURS THEIRS [PATH] are a `.lean` file's: merge its import lists")
    a = ap.parse_args(argv)
    if a.install is not None:
        install(a.install)
        return 0
    if a.normalize:
        for p in a.files:
            print(f"merge.py: {p}: {normalize(p)} digests recomputed")
        return 0
    if len(a.files) not in (3, 4):
        ap.error("name BASE OURS THEIRS [PATH], or --install, or --normalize FILE...")
    base, ours, theirs = a.files[:3]
    path = a.files[3] if len(a.files) == 4 else ours
    if a.lean:
        text, left = lean(base, ours, theirs)
        ours.write_text(text)
        if left:
            print(f"merge.py: {path}: {left} conflict(s) beyond import lists", file=sys.stderr)
        return 1 if left else 0
    try:
        merged, why = merge(load(base), load(ours), load(theirs))
    except ValueError as e:
        why = [f"a side is not JSON ({e})" if isinstance(e, json.JSONDecodeError) else f"a side: {e}"]
    if why:
        print(f"merge.py: {path}: refused; regenerate the record on the merged tree (`audit.py --update`):", file=sys.stderr)
        print("\n".join(f"  {w}" for w in why), file=sys.stderr)
        return 1
    ours.write_text(A.text(merged))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
