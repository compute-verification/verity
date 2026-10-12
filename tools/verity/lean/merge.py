"""`merge.py`: git's merge driver for `lean-audit.json`, so that two branches that each add guarantees merge without a rebuild.

    python tools/verity/lean/merge.py BASE OURS THEIRS [PATH]   what git runs (`%O %A %B %P`): the merge is written to OURS
    python tools/verity/lean/merge.py --install [REPO]         register the driver in the repository's git config, once per clone
    python tools/verity/lean/merge.py --normalize FILE...      recompute each `reads` digest from its definitions

`.gitattributes` names this driver for every `lean-audit.json`, and git runs it when both sides changed the file. A record is
a map of entries by name: the guarantees, the modules under `reads`, the dependencies, and the policy's own sections. Each
side's change to an entry (added, changed or removed) applies, and an entry both sides changed differently refuses the merge.
A guarantee's record (its `owner` aside, which merges on its own) and a dependency's digest are one entry each. Under `reads`, which two branches adding guarantees that
read one module both change, the module's definitions merge the same way, its readers (`guarantees`) as sets, and its digest
is recomputed from the merged definitions (`audit.module_digest`). The merge also refuses when one side stops reading a
definition in a module the other side changed: the record does not say which guarantees read which definition. A side
written before guarantees were called that (`pins`) is read under today's names (`audit.as_guarantees`). A `layers` rule both
sides changed, or both added, merges too (`layers`): its `imports` and `except` as sets, each side's additions and removals
applied, and both `why`s kept; the driver prints the merged rule, for the PR.

Exit 0: merged. Exit 1: refused, the entries named on stderr and OURS left as it was, which git reports as a conflict; then
regenerate the record on the merged tree (`check.py --update`), as without the driver. A merged record is a candidate, never
a verdict: `check`'s Lean audit rebuilds every record of the merged tree and fails on any difference, as for any record.
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


def members(o: list, a: list, b: list) -> list:
    """A list merged as a set, three ways: each side's additions and removals apply; ours' order, then theirs' new ones."""
    keep = (set(a) & set(b)) | (set(a) - set(o)) | (set(b) - set(o))
    return [x for x in a if x in keep] + [x for x in b if x in keep and x not in a]


def layers(o: dict, a: dict, b: dict, why: list[str], notes: list[str]) -> dict:
    """The `layers` rules entry by entry. A rule both sides changed (or both added) merges its `imports` and `except` as sets,
    each side's additions and removals applied, so it allows what each side allows, and keeps both `why`s; ``notes`` gets it
    as merged, for the PR. A rule removed on one side and changed on the other is refused."""
    out = {}
    for k in keys("layers", a, b):
        x, y, z = o.get(k, MISSING), a.get(k, MISSING), b.get(k, MISSING)
        if MISSING in (y, z) or y == z or x in (y, z) or not all(isinstance(r, dict) for r in (y, z, {} if x is MISSING else x)):
            v = one(f"layers[{k}]", x, y, z, why)
        else:
            base = {} if x is MISSING else x
            v = {}
            for f in keys("", y, z):
                p, q, r = base.get(f, MISSING), y.get(f, MISSING), z.get(f, MISSING)
                if f in ("imports", "except") and all(isinstance(s, list) for s in (q, r)):
                    v[f] = members([] if p is MISSING else p, q, r)
                elif f == "why" and all(isinstance(s, str) for s in (q, r)) and q != r and p not in (q, r):
                    v[f] = "; ".join(s for s in (q, r) if s)
                elif (w := one(f"layers[{k}].{f}", p, q, r, why)) is not MISSING:
                    v[f] = w
            notes.append(f"layers[{k}]: changed on both sides; merged as each side's imports and except together, with both "
                         f"whys: {json.dumps(v, ensure_ascii=False)}")
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


def merge(base: dict, ours: dict, theirs: dict, notes: list[str] | None = None) -> tuple[dict, list[str]]:
    """(the merged record, why it is refused: empty when it merged).  A pure function of the three records; ``notes`` gets
    what the PR should show of the merge (`layers`)."""
    why: list[str] = []
    notes = [] if notes is None else notes
    out = {}
    for k in keys("", ours, theirs):
        o, a, b = base.get(k, MISSING), ours.get(k, MISSING), theirs.get(k, MISSING)
        if k == "reads":
            v = reads({} if o is MISSING else o, {} if a is MISSING else a, {} if b is MISSING else b, why)
        elif k == "layers" and isinstance(a, dict) and isinstance(b, dict) and (o is MISSING or isinstance(o, dict)):
            v = layers({} if o is MISSING else o, a, b, why, notes)
        elif k == "guarantees" and isinstance(a, dict) and isinstance(b, dict) and (o is MISSING or isinstance(o, dict)):
            v = guarantees({} if o is MISSING else o, a, b, why)
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


def load(path: Path) -> dict:
    text = path.read_text()
    return A.as_guarantees(json.loads(text)) if text.strip() else {}


def install(repo: Path) -> None:
    """Register the driver in ``repo``'s git config, as this file's path from the top of the worktree (git runs drivers
    there)."""
    top = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=repo, capture_output=True, text=True,
                              check=True).stdout.strip())
    driver = f"python3 {Path(os.path.relpath(HERE / 'merge.py', top)).as_posix()} %O %A %B %P"
    for k, v in (("name", "lean-audit.json: each side's guarantees, reads and dependencies (tools/verity/lean/merge.py)"), ("driver", driver)):
        subprocess.run(["git", "config", f"merge.{NAME}.{k}", v], cwd=repo, check=True)
    print(f"merge.py: merge.{NAME}.driver = {driver} (in {top})")


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
    notes: list[str] = []
    try:
        merged, why = merge(load(base), load(ours), load(theirs), notes)
    except ValueError as e:
        why = [f"a side is not JSON ({e})" if isinstance(e, json.JSONDecodeError) else f"a side: {e}"]
    if why:
        print(f"merge.py: {path}: refused; regenerate the record on the merged tree (`audit.py --update`):", file=sys.stderr)
        print("\n".join(f"  {w}" for w in why), file=sys.stderr)
        return 1
    if notes:
        print(f"merge.py: {path}: merged; put these in the PR:", file=sys.stderr)
        print("\n".join(f"  {n}" for n in notes), file=sys.stderr)
    ours.write_text(A.text(merged))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
