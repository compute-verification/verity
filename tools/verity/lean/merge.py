"""`merge.py`: git's merge driver for `lean-audit.json`, so that two branches that each add guarantees merge without a rebuild.

    python tools/verity/lean/merge.py BASE OURS THEIRS [PATH]   what git runs (`%O %A %B %P`): the merge is written to OURS
    python tools/verity/lean/merge.py --lean BASE OURS THEIRS [PATH]   the same for a `.lean` file (`lean-imports`)
    python tools/verity/lean/merge.py --lakefile BASE OURS THEIRS [PATH]   the same for a `lakefile.toml` (`lakefile-lists`)
    python tools/verity/lean/merge.py --install [REPO]         register the drivers in the repository's git config, once per clone
    python tools/verity/lean/merge.py --normalize FILE...      recompute each `reads` digest from its definitions

`.gitattributes` names this driver for every `lean-audit.json`, and git runs it when both sides changed the file. A record is
a map of entries by name: the guarantees, the modules under `reads`, the assumptions, the dependencies, and the policy's own
sections. Each side's change to an entry (added, changed or removed) applies. Under `reads`, which two branches adding
guarantees that read one module both change, the module's definitions merge the same way, its readers (`guarantees`) as
sets, and its digest is recomputed from the merged definitions (`audit.module_digest`). A side written before guarantees
were called that (`pins`) is read under today's names (`audit.as_guarantees`).

What the build determines never refuses: the records `check.py --update` writes from it and a train's Lean check
(`--regenerate`) rewrites on the merged tree whenever they differ (`audit.RECORD_DIFFERENCE`, `audit.regenerated`). Those are
a guarantee's record (signature, named assumptions, type hash: restated on both sides, the merged tree's statement decides
it), each `reads` group (which guarantees read which definition, and each definition's hash) and an assumption's `digest`.
Where the sides disagree the merge takes the union (a definition or a module one side stopped reading, unless that side
moved the definition to another module, where alone it stays (`moved`); readers as sets less those the merged record no
longer lists, a group left without readers or definitions dropped) or ours' value (a
hash or a record both sides changed differently), and names each in one line on stderr. Whether a guarantee is listed is
not the build's: one side's removal applies over the other's change to its record. What only a person writes still
refuses when both sides changed it differently: an `owner`, a `shape` or `frozen` entry, a dependency's revision, a policy
entry (a layer's field, a `compile_time` digest).

Exit 0: merged. Exit 1: refused, the entries named on stderr and OURS left as it was, which git reports as a conflict; then
regenerate the record on the merged tree (`check.py --update`), as without the driver. A merged record is a candidate, never
a verdict: `check`'s Lean audit rebuilds every record of the merged tree and fails on any difference, as for any record; a
train's (`--regenerate`) commits the regenerated records on landing, and Daniel's landing DM shows each one that changed
against `main`.

The layer rules (`layers`), which two branches that each widen a layer for a guarantee both change, merge field by field: `imports`
and `except` as lists, `why` as a list of its `; ` clauses (`layers`). Every list of names (the roots, `meaning`, `proved_in`,
`claims.modules`, `shape.premises`...), which two branches that each add a module both change, merges as a list (`items`). For a `.lean` file (`--lean`, `lean-imports` in
`.gitattributes`), git's own merge runs and only a conflict all of whose sides are `import` lines is merged, as a list: two branches
that each add a guarantee's module to `Guarantees.lean` merge, and any other conflict stays git's. For a `lakefile.toml`
(`--lakefile`, `lakefile-lists`), a conflict all of whose sides are `[[require]]` blocks merges as a set of blocks, and one all
of whose sides are a string array's items (`roots`) merges as a list, wrapped as ours wraps it: two branches that each require a
package or add a root to `Security/Properties` merge. Any other conflict, and a result that is not TOML, stays git's. In
both, an import, a require or a root that one side removed stays removed, judged on the whole files (`kept`), so a line the
other side moved can't bring it back.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import tomllib
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
#: the lakefiles' driver (`--lakefile`)
LAKEFILE = "lakefile-lists"
#: a lakefile's `[[require]]` block line, and a line of a string array's items (`roots = ["A", "B",`, `"C"]`)
REQUIRE_LINE = re.compile(r'\s*(\[\[require\]\]|(name|path|git|rev|scope|version|subDir)\s*=\s*\S.*)\s*$')
ITEM_LINE = re.compile(r'\s*(\w+\s*=\s*\[)?\s*("[^"]*"\s*,?\s*)+\]?\s*$')
#: where a merged string array wraps
WIDTH = 120
#: the sections `check.py --update` writes in sorted order; the others keep ours' order, with theirs' new entries after it
SORTED = {"assumptions", "guarantees", "reads", "dependencies", "compile_time"}
#: sections whose entries are maps merged entry by entry in turn (`upstream`'s `scan` and `watch`, `shape`'s `properties`,
#: `frozen` and `conjectures`, `trees`' `except`)
NESTED = {"upstream", "shape", "claims", "trees"}
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


def names(o, a, b) -> bool:
    """Whether the three sides are lists of names (`roots`, `meaning`, `claims.modules`, `shape.premises`...), which merge as
    lists (`items`): two branches that each add a module to one merge."""
    def ok(x):
        return isinstance(x, list) and all(isinstance(i, str) for i in x)
    return ok(a) and ok(b) and (o is MISSING or ok(o))


def entries(path: str, o: dict, a: dict, b: dict, why: list[str], depth: int = 1) -> dict:
    """Maps merged entry by entry; ``depth`` > 1 merges entries that are maps entry by entry too, and lists of names merge
    as lists."""
    out = {}
    for k in keys(path, a, b):
        x, y, z = o.get(k, MISSING), a.get(k, MISSING), b.get(k, MISSING)
        if depth > 1 and isinstance(y, dict) and isinstance(z, dict) and (x is MISSING or isinstance(x, dict)):
            v = entries(f"{path}[{k}]", {} if x is MISSING else x, y, z, why, depth - 1)
        elif depth > 1 and names(x, y, z):
            v = items([] if x is MISSING else x, y, z)
        else:
            v = one(f"{path}[{k}]", x, y, z, why)
        if v is not MISSING:
            out[k] = v
    return out


def derived(path: str, o, a, b, candidate: list[str]):
    """The three-way merge of a value the build determines (a definition's hash, an assumption's digest, a guarantee's
    signature, named assumptions and type hash): as `one`, but no clash refuses.  A value both sides changed differently is
    ours' (named ``path (ours)`` in ``candidate``), and one removed on one side and changed on the other is kept (the union,
    named ``path``): the train's Lean check regenerates it from the merged tree."""
    if a == b or a == o:
        return b
    if b == o:
        return a
    if a is MISSING or b is MISSING:
        candidate.append(path)
        return b if a is MISSING else a
    candidate.append(f"{path} (ours)")
    return a


def owned(rec, k: str = "owner"):
    return MISSING if rec is MISSING else rec.get(k, MISSING)


def guarantees(o: dict, a: dict, b: dict, why: list[str], candidate: list[str]) -> dict:
    """The guarantees entry by entry, each one's `owner` apart from the rest of its record: a branch that restates a guarantee
    and one that gives it an owner merge.  The rest of a record (signature, named assumptions, type hash) is the build's, so
    two sides that restated it differently merge as ours' record (`derived`), which the train's Lean check rebuilds from the
    merged statement.  Whether a guarantee is listed is not the build's: one side's removal applies over the other's
    change to its record, and refuses only when that side changed its owner."""
    out = {}
    for k in keys("guarantees", a, b):
        x, y, z = o.get(k, MISSING), a.get(k, MISSING), b.get(k, MISSING)
        if not all(isinstance(r, dict) for r in (x, y, z) if r is not MISSING):
            v = one(f"guarantees[{k}]", x, y, z, why)
        elif MISSING in (y, z) and x is not MISSING and y != z:
            kept = y if z is MISSING else z
            if kept == x:
                v = MISSING
            elif owned(kept) == owned(x):
                v = MISSING
                candidate.append(f"guarantees[{k}] (removed)")
            else:
                v = one(f"guarantees[{k}]", x, y, z, why)
        elif MISSING in (y, z):
            v = one(f"guarantees[{k}]", x, y, z, why)
        else:
            v = derived(f"guarantees[{k}]", x if x is MISSING else A.unowned(x), A.unowned(y), A.unowned(z), candidate)
            who = one(f"guarantees[{k}].owner", owned(x), owned(y), owned(z), why)
            v = v if who is MISSING else {"owner": who} | v
        if v is not MISSING:
            out[k] = v
    return out


def assumptions(o: dict, a: dict, b: dict, why: list[str], candidate: list[str]) -> dict:
    """The named assumptions entry by entry, each one's `owner` (hand-written) apart from its `digest` (the build's: the hash
    its `reads` record holds), which merges as `derived`.  An entry one side removed and the other changed is kept: the
    train's Lean check drops it when no guarantee of the merged tree reads it (`assumptions.unread`)."""
    out = {}
    for k in keys("assumptions", a, b):
        x, y, z = o.get(k, MISSING), a.get(k, MISSING), b.get(k, MISSING)
        if not all(isinstance(r, dict) for r in (x, y, z) if r is not MISSING):
            v = one(f"assumptions[{k}]", x, y, z, why)
        elif MISSING in (y, z):
            v = derived(f"assumptions[{k}]", x, y, z, candidate)
        else:
            d = derived(f"assumptions[{k}].digest", owned(x, "digest"), owned(y, "digest"), owned(z, "digest"), candidate)
            who = one(f"assumptions[{k}].owner", owned(x), owned(y), owned(z), why)
            v = ({} if who is MISSING else {"owner": who}) | ({} if d is MISSING else {"digest": d})
        if v is not MISSING:
            out[k] = v
    return out


def core(group):
    """What a `reads` entry says, its digest (which it determines) aside."""
    return None if group is None else (A.definitions(group), sorted(group.get("guarantees", [])))


def group(m: str, x: dict, y: dict, z: dict, why: list[str], candidate: list[str]) -> dict:
    """One module's `reads` entry, both sides having changed it: definitions entry by entry, readers as sets.  A definition one
    side no longer reads while the other still reads it stays, with the other side's hash: the record does not say which
    guarantees read it, so the union is the candidate (``candidate``), and the train's Lean check regenerates the group.  A
    hash both sides changed differently is ours' (`derived`)."""
    xd, yd, zd = A.definitions(x), A.definitions(y), A.definitions(z)
    defs = {}
    for d in sorted(set(xd) | set(yd) | set(zd)):
        p, q, r = xd.get(d, MISSING), yd.get(d, MISSING), zd.get(d, MISSING)
        if p is not MISSING and (q is MISSING) != (r is MISSING):
            defs[d] = q if r is MISSING else r
            candidate.append(f"reads[{m}].definitions[{d}]")
        elif (v := derived(f"reads[{m}].definitions[{d}]", p, q, r, candidate)) is not MISSING:
            defs[d] = v
    if len({len(h) for h in defs.values() if isinstance(h, str)}) > 1:
        candidate.append(f"reads[{m}] (32-bit and SHA-256 hashes mixed)")
    xp, yp, zp = set(x.get("guarantees", [])), set(y.get("guarantees", [])), set(z.get("guarantees", []))
    return {"definitions": defs, "guarantees": sorted((yp & zp) | (yp - xp) | (zp - xp))}


def reads(o: dict, a: dict, b: dict, why: list[str], candidate: list[str]) -> dict:
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
            g = group(m, x or {}, y or {}, z or {}, why, [])
            candidate.append(f"reads[{m}]")
        else:
            g = group(m, x or {}, y, z, why, candidate)
        if g is not None:
            out[m] = {"definitions": dict(sorted(A.definitions(g).items())), "guarantees": sorted(g.get("guarantees", []))}
    moved(o, a, b, out, candidate)
    return {m: {"digest": A.module_digest(g["definitions"]), **g} for m, g in out.items()}


def moved(o: dict, a: dict, b: dict, out: dict, candidate: list[str]) -> None:
    """A Lean declaration lives in one module: a definition one side moved (it added it under another module of the merged
    table and no longer lists it under this one) leaves the module the other side's group kept it in, and is no longer the
    candidate `group` named.  Its hash is the mover's; when the other side changed it in the old module, or added
    guarantees there that may read it in the new one, the new module's group is the candidate."""
    def defs(s: dict, m: str) -> dict:
        return A.definitions(s.get(m) or {})
    where: dict[str, list[str]] = {}
    for m, g in out.items():
        for d in g["definitions"]:
            where.setdefault(d, []).append(m)
    for d, ms in where.items():
        for m in ms:
            to = next((n for n in ms if n != m and d in out[n]["definitions"] for s in (a, b)
                       if d in defs(s, n) and d not in defs(o, n) and d not in defs(s, m)), None)
            if to is None:
                continue
            del out[m]["definitions"][d]
            if f"reads[{m}].definitions[{d}]" in candidate:
                candidate.remove(f"reads[{m}].definitions[{d}]")
            kept = a if d in defs(a, m) else b
            new = set((kept.get(m) or {}).get("guarantees", [])) - set((o.get(m) or {}).get("guarantees", []))
            if defs(kept, m).get(d) != defs(o, m).get(d) or new & set(out[m]["guarantees"]):
                candidate.append(f"reads[{to}] (definitions moved from {m})")


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


def merge(base: dict, ours: dict, theirs: dict, candidate: list[str] | None = None) -> tuple[dict, list[str]]:
    """(the merged record, why it is refused: empty when it merged).  A pure function of the three records.  ``candidate``
    collects the records the build determines (`RECORD_DIFFERENCE`'s kinds: guarantees' records, `reads`, assumptions'
    digests) merged although the sides disagree on them: the union or ours' value, which the train's Lean check
    regenerates."""
    why: list[str] = []
    candidate = [] if candidate is None else candidate
    out = {}
    for k in keys("", ours, theirs):
        o, a, b = base.get(k, MISSING), ours.get(k, MISSING), theirs.get(k, MISSING)
        if k == "reads":
            v = reads({} if o is MISSING else o, {} if a is MISSING else a, {} if b is MISSING else b, why, candidate)
        elif k == "guarantees" and isinstance(a, dict) and isinstance(b, dict) and (o is MISSING or isinstance(o, dict)):
            v = guarantees({} if o is MISSING else o, a, b, why, candidate)
        elif k == "assumptions" and all(isinstance(s, dict) for s in (o, a, b) if s is not MISSING):
            v = assumptions(*({} if s is MISSING else s for s in (o, a, b)), why, candidate)
            v = MISSING if not v and MISSING in (a, b) else v
        elif k == "layers" and isinstance(a, dict) and isinstance(b, dict) and (o is MISSING or isinstance(o, dict)):
            v = layers({} if o is MISSING else o, a, b, why)
        elif names(o, a, b):
            v = items([] if o is MISSING else o, a, b)
        elif isinstance(a, dict) and isinstance(b, dict) and (o is MISSING or isinstance(o, dict)):
            v = entries(k, {} if o is MISSING else o, a, b, why, 2 if k in NESTED else 1)
        else:
            v = one(k, o, a, b, why)
        if v is not MISSING:
            out[k] = v
    kept = set(out.get("guarantees", {}))
    for m, g in list(out.get("reads", {}).items()):
        if extra := sorted(set(g["guarantees"]) - kept):
            g["guarantees"] = [r for r in g["guarantees"] if r in kept]
            candidate.append(f"reads[{m}].guarantees[{', '.join(extra)}]")
        if not g["guarantees"] or not g["definitions"]:
            del out["reads"][m]
            candidate.append(f"reads[{m}]")
    return out, why


def kept(o: set, a: set, b: set) -> set:
    """The names a three-way merge of whole files keeps: (A ∩ B) ∪ (A − O) ∪ (B − O), what both sides have and what either
    added.  A name one side removed while the other kept it stays removed, however git splits the file into hunks: a line
    one side moved shows in a hunk as an addition, and the hunk alone would bring back what the other side removed
    (#4690 onto `next`, 26 Oct: `Definitions.Accounting`'s retired import came back and Security stopped building)."""
    return (a & b) | (a - o) | (b - o)


def _imports(lines: list[str]) -> bool:
    return all(s.startswith("import ") or not s.strip() for s in lines)


def _import_names(text: str) -> set[str]:
    return {s.strip() for s in text.splitlines() if s.startswith("import ")}


def _blocks(lines: list[str]) -> list[str] | None:
    """A conflict side's `[[require]]` blocks (each its lines, blank ones dropped), or None when it holds anything else."""
    body = [s for s in lines if s.strip()]
    if not all(REQUIRE_LINE.match(s) for s in body) or (body and not body[0].strip().startswith("[[require]]")):
        return None
    out: list[str] = []
    for s in body:
        if s.strip().startswith("[[require]]"):
            out.append("")
        out[-1] += s if s.endswith("\n") else s + "\n"
    return out


def _require_key(require: dict) -> str:
    return json.dumps(require, sort_keys=True)


def _block_key(block: str) -> str | None:
    try:
        return _require_key(tomllib.loads(block)["require"][0])
    except (tomllib.TOMLDecodeError, KeyError, IndexError):
        return None


def _requires(o: list[str], a: list[str], b: list[str], keep: set | None = None) -> list[str] | None:
    """A conflict all of whose sides are `[[require]]` blocks: ours' blocks with theirs' new ones, less what either removed
    (in the hunk, and in the whole files: ``keep``, `kept` of each side's requires)."""
    x, y, z = _blocks(o), _blocks(a), _blocks(b)
    if x is None or y is None or z is None:
        return None
    filled = [i for i, s in enumerate(a) if s.strip()]
    lead, trail = (a[:filled[0]], a[filled[-1] + 1:]) if filled else (a, [])
    merged = [k for k in y if k in z or k not in x] + [k for k in z if k not in y and k not in x]
    merged = [k for k in merged if keep is None or (key := _block_key(k)) is None or key in keep]
    return lead + "\n".join(merged).splitlines(keepends=True) + trail


def _array(o: list[str], a: list[str], b: list[str], keep: set | None = None) -> list[str] | None:
    """A conflict all of whose sides are a string array's items (`roots`): ours' items with theirs' new ones, less what
    either removed in the hunk (and not in ``keep``), wrapped at `WIDTH` with ours' opening, indent and close.  `lakefile`
    then holds every array to the whole files (`_hold`)."""
    sides = [[s for s in side if s.strip()] for side in (o, a, b)]
    if not sides[1] or not all(ITEM_LINE.match(s) for side in sides for s in side):
        return None
    opening = lambda side: (m.group(1) if side and (m := ITEM_LINE.match(side[0])) and m.group(1) else None)
    closes = lambda side: bool(side) and side[-1].rstrip().endswith("]")
    if sides[2] and (opening(sides[2]) != opening(sides[1]) or closes(sides[2]) != closes(sides[1])):
        return None
    head = opening(sides[1])
    lead = re.match(r"\s*", sides[1][0]).group(0)
    indent = re.match(r"\s*", sides[1][1]).group(0) if len(sides[1]) > 1 else (lead + "  " if head else lead)
    names = [re.findall(r'"([^"]*)"', "".join(side)) for side in sides]
    merged = [n for n in items(*names) if keep is None or n in keep]
    if not merged:
        return None
    return _wrap(merged, head, lead, indent, closes(sides[1]))


def _wrap(merged: list[str], head: str | None, lead: str, indent: str, close: bool) -> list[str]:
    """A string array's items as lines: ``head`` (`roots = [`) and the first item after ``lead``, the rest after ``indent``,
    wrapped at `WIDTH`, closed with `]` when ``close``."""
    if not merged:
        return [lead + (head or "") + ("]" if close else "") + "\n"]
    pieces = [f'"{n}"' + ("]" if i == len(merged) - 1 and close else ",") for i, n in enumerate(merged)]
    out, line = [], lead + head if head else lead
    start = line
    for piece in pieces:
        if line == start:
            line += piece
        elif len(line) + 1 + len(piece) <= WIDTH:
            line += " " + piece
        else:
            out.append(line + "\n")
            line = indent + piece
    return out + [line + "\n"]


def _lakefile_names(text: str) -> tuple[set[str], dict[str, set[str]]] | None:
    """A lakefile's requires (`_require_key`) and its string arrays' items by the array's key (`roots`, every library's
    together), or None when it is not TOML."""
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return None
    strings: dict[str, set[str]] = {}

    def walk(v):
        if isinstance(v, list):
            for i in v:
                walk(i)
        elif isinstance(v, dict):
            for k, i in v.items():
                if isinstance(i, list) and all(isinstance(x, str) for x in i):
                    strings.setdefault(k, set()).update(i)
                walk(i)
    walk(doc)
    return {_require_key(r) for r in doc.get("require", []) if isinstance(r, dict)}, strings


def _drop(span: list[str], head: str, merged: list[str]) -> list[str]:
    """A string array's lines (``span``, opening with ``head``) holding only ``merged`` (its items, in order, less some): each
    item left on its own line, a line left without items dropped; rewrapped (`_wrap`) when the opening line keeps none."""
    left, rows = list(merged), []
    for s in span:
        mine = []
        for x in re.findall(r'"([^"]*)"', s):
            if left and left[0] == x:
                mine.append(left.pop(0))
        rows.append((s, mine))
    lead = re.match(r"\s*", span[0]).group(0)
    if not rows[0][1]:
        indent = re.match(r"\s*", span[1]).group(0) if len(span) > 1 else lead + "  "
        return _wrap(merged, head, lead, indent, True)
    out = []
    for i, (s, mine) in enumerate(rows):
        if mine:
            start = lead + head if i == 0 else re.match(r"\s*", s).group(0)
            out.append(start + " ".join(f'"{x}",' for x in mine))
    out[-1] = out[-1][:-1] + "]"
    return [s + "\n" for s in out]


def _hold(text: str, requires: set[str], strings: dict[str, set[str]]) -> str:
    """The merged lakefile ``text`` with every `[[require]]` block and every string array's item held to the whole files
    (``requires``, once each, and ``strings`` by the array's key: `kept` of each side's), outside any conflict left.  Git
    merges a removal on one side and a move on the other without a conflict, bringing back what was removed."""
    lines, out, seen, n = text.splitlines(keepends=True), [], set(), 0
    while n < len(lines):
        s = lines[n]
        if s.startswith("<<<<<<<"):
            end = next((i for i in range(n, len(lines)) if lines[i].startswith(">>>>>>>")), len(lines) - 1)
            out += lines[n:end + 1]
            n = end + 1
            continue
        m = ITEM_LINE.match(s)
        if m and m.group(1):
            end = n
            while not lines[end].rstrip().endswith("]") and end + 1 < len(lines) and ITEM_LINE.match(lines[end + 1]) \
                    and not ITEM_LINE.match(lines[end + 1]).group(1):
                end += 1
            span = lines[n:end + 1]
            key = re.match(r"\s*(\w+)", m.group(1)).group(1)
            names = re.findall(r'"([^"]*)"', "".join(span))
            if span[-1].rstrip().endswith("]") and key in strings:
                merged = [x for x in names if x in strings[key]]
                if merged != names:
                    span = _drop(span, m.group(1), merged)
            out += span
            n = end + 1
            continue
        if s.strip() == "[[require]]":
            end = n + 1
            while end < len(lines) and lines[end].strip() and not lines[end].lstrip().startswith("[") \
                    and not lines[end].startswith(("<<<<<<<", "|||||||", "=======", ">>>>>>>")):
                end += 1
            key = _block_key("".join(lines[n:end]))
            if key is not None and (key not in requires or key in seen):
                if out and not out[-1].strip():
                    out.pop()
                n = end
                continue
            seen.add(key)
        out.append(s)
        n += 1
    return "".join(out)


def lakefile(base: Path, ours: Path, theirs: Path) -> tuple[str, int]:
    """(the merged text, the number of conflicts left) of a `lakefile.toml`: git's own merge, then each conflict whose sides are
    only `[[require]]` blocks merged as a set of blocks, and each whose sides are only a string array's items (`roots`) merged
    as a list (`items`), so two branches that each require a package or add a root merge; a block or an item one side
    removed stays removed anywhere in the merged file, conflict or not (`kept`, `_hold`).  Any other conflict (a comment, a value) is left as git writes it, and so is a result
    that is not TOML."""
    sides = [_lakefile_names(p.read_text()) for p in (base, ours, theirs)]
    requires = None if None in sides else kept(*(s[0] for s in sides))
    text, left = lean(base, ours, theirs, (lambda o, a, b: _requires(o, a, b, requires), _array))
    if requires is not None:
        keys = set().union(*(s[1] for s in sides))
        text = _hold(text, requires, {k: kept(*(s[1].get(k, set()) for s in sides)) for k in keys})
    if not left:
        try:
            tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            return lean(base, ours, theirs, ())[0], 1
    return text, left


def lean(base: Path, ours: Path, theirs: Path, resolvers=None) -> tuple[str, int]:
    """(the merged text, the number of conflicts left) of a `.lean` file: git's own merge (`git merge-file`), then each conflict
    whose three sides are only `import` lines merged as lists of imports (`items`), so two branches that each add a module to
    an import list merge.  Every import outside a conflict left is then held to the whole files' imports (`kept`), once
    each: one side removed stays removed wherever git's hunks put it.  Any other conflict is left as git writes it.
    ``resolvers`` (each `(base, ours, theirs)` lines to the merged lines, or None) replace the imports rule, and with them
    no import is filtered."""
    keep = None
    if resolvers is None:
        keep = kept(*(_import_names(p.read_text()) for p in (base, ours, theirs)))
        resolvers = (lambda o, a, b: [s for s in items([s for s in o if s.strip()], [s for s in a if s.strip()],
                                                       [s for s in b if s.strip()])]
                     if _imports(a) and _imports(o) and _imports(b) else None,)
    p = subprocess.run(["git", "merge-file", "-p", "--diff3", "-L", "ours", "-L", "base", "-L", "theirs", str(ours), str(base),
                        str(theirs)], capture_output=True, text=True)
    if p.returncode < 0 or p.returncode > 127:
        raise ValueError(f"git merge-file failed: {p.stderr.strip()}")
    out, left, lines, n, seen = [], 0, p.stdout.splitlines(keepends=True), 0, set()

    def emit(s: str):
        if keep is not None and s.startswith("import "):
            if s.strip() not in keep or s.strip() in seen:
                return
            seen.add(s.strip())
        out.append(s)
    while n < len(lines):
        if not lines[n].startswith("<<<<<<< ours"):
            emit(lines[n])
            n += 1
            continue
        mid = next(i for i in range(n, len(lines)) if lines[i].startswith("||||||| base"))
        sep = next(i for i in range(mid, len(lines)) if lines[i].rstrip("\n") == "=======")
        end = next(i for i in range(sep, len(lines)) if lines[i].startswith(">>>>>>> theirs"))
        a, o, b = lines[n + 1:mid], lines[mid + 1:sep], lines[sep + 1:end]
        merged = next((m for r in resolvers if (m := r(o, a, b)) is not None), None)
        if merged is not None:
            for s in merged:
                emit(s)
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
                             (LEAN, ".lean: each side's imports", driver.replace(" %O", " --lean %O")),
                             (LAKEFILE, "lakefile.toml: each side's requires and roots", driver.replace(" %O", " --lakefile %O"))):
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
    ap.add_argument("--lakefile", action="store_true", help="BASE OURS THEIRS [PATH] are a `lakefile.toml`'s: merge its "
                    "`[[require]]` blocks and string arrays")
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
    if a.lean or a.lakefile:
        text, left = lakefile(base, ours, theirs) if a.lakefile else lean(base, ours, theirs)
        ours.write_text(text)
        if left:
            what = "requires and string arrays" if a.lakefile else "import lists"
            print(f"merge.py: {path}: {left} conflict(s) beyond {what}", file=sys.stderr)
        return 1 if left else 0
    candidate: list[str] = []
    try:
        merged, why = merge(load(base), load(ours), load(theirs), candidate)
    except ValueError as e:
        why = [f"a side is not JSON ({e})" if isinstance(e, json.JSONDecodeError) else f"a side: {e}"]
    if why:
        print(f"merge.py: {path}: refused; regenerate the record on the merged tree (`check.py --update`):", file=sys.stderr)
        print("\n".join(f"  {w}" for w in why), file=sys.stderr)
        return 1
    if candidate:
        print(f"merge.py: {path}: merged derived records as a candidate (the union, or ours' value where marked), which the "
              f"train's Lean check regenerates: {'; '.join(dict.fromkeys(candidate))}", file=sys.stderr)
    ours.write_text(A.text(merged))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
