"""`merge.py`: git's merge driver for `lean-audit.json`, so that two branches that each add guarantees merge without a rebuild.

    python tools/verity/lean/merge.py BASE OURS THEIRS [PATH]   what git runs (`%O %A %B %P`): the merge is written to OURS
    python tools/verity/lean/merge.py --lean BASE OURS THEIRS [PATH]   the same for a `.lean` file (`lean-imports`)
    python tools/verity/lean/merge.py --lakefile BASE OURS THEIRS [PATH]   the same for a `lakefile.toml` (`lakefile-lists`)
    python tools/verity/lean/merge.py --caps BASE OURS THEIRS [PATH]   the same for `tests/test_lean_packages.py` (`lean-caps`)
    python tools/verity/lean/merge.py --pyproject BASE OURS THEIRS [PATH]   the same for a `pyproject.toml` (`pyproject-test-lists`)
    python tools/verity/lean/merge.py --install [REPO]         register the drivers in the repository's git config, once per clone
    python tools/verity/lean/merge.py --normalize FILE...      recompute each `reads` digest from its definitions, and drop
                                                               the readers an older record lists

`.gitattributes` names this driver for every `lean-audit.json`, and git runs it when both sides changed the file. A record is
a map of entries by name: the guarantees, the modules under `reads`, the assumptions, the open premises, the domain
hypotheses, the dependencies, and the policy's own sections. Each side's change to an entry (added, changed or removed) applies.
An entry sits in one section of a group (`EXCLUSIVE`: a claim's `claims.witnesses`, `not`, `generic`; a premise's
`assumptions`, `domain`, `open`): one side moved between them from the base is where that side put it, and one both sides
placed differently is where theirs put it, which `check.py --update` classifies (`exclusive`). Under `reads`, which two
branches adding guarantees that read one module both change, the module's definitions merge the same way, and its digest
is recomputed from the merged definitions (`audit.module_digest`). A side written before guarantees were called that
(`pins`) is read under today's names (`audit.as_guarantees`), and a side whose `reads` groups still list their readers
(`audit.READER_KEYS`) is read without them, so a branch from before the lock dropped them merges with one from after.

The merge refuses only where it can't vouch for itself (the guards below): otherwise lock files never conflict on merge.
What the build determines is the records
`check.py --update` writes from it, which the lock run after landing rewrites on the merged tree whenever they differ
(`audit.RECORD_DIFFERENCE`, `audit.regenerated`): a guarantee's record (signature, named assumptions, type hash: restated
on both sides, the merged tree's statement decides it), each `reads` group (which definitions the guarantees read, and
each one's hash) and an assumption's or open premise's `digest`.
Where the sides disagree the merge takes the union (a definition the other side changed while one side stopped reading
it) or theirs' value (a hash or a record both sides changed differently). A definition one side reads nowhere while the
other kept it as the base had it is dropped (`gone`): none of that side's guarantees reads it, so in the merged tree only
the other side's new or restated guarantees can, and one moved to another module leaves the old one (`moved`); a group
left without definitions is dropped. Whether a guarantee is listed is not the build's: one side's removal applies over
the other's change to its record, and only a change to its owner keeps it. What only a person writes (an `owner`, an open
premise's `issue` or `due`, a domain hypothesis's `why` or `cites`, a `shape`, `frozen` or `except` entry, a dependency's
revision, a policy entry such as a layer's field or a `compile_time` digest) merges as the union of both sides' entries:
one both sides changed differently is theirs', and one a side removed while the other changed it is the changed one
(`one`). A `shape.inline` entry is one premise of a guarantee, so two sides that each add or drop premises of one
guarantee merge premise by premise (`DEEPER`). A `shape.inline` premise one side dropped while the other has it as the
base had it goes, as in any three-way merge, unless that other side restated its guarantee, otherwise than the dropping
side, and the merged record lists it (`held`): a side whose `--update` ran on an older tree drops a text the other
side's restated statement still prints, and the Lean check fails on a missing entry but not on one whose premise no
longer offends, which the next `--update` drops. An `unwitnessed` or `witnesses` entry merges the same way (`JUDGED`),
named as a candidate, which the Lean check of the merged commit judges, but for one both sides changed differently: it is
theirs only when theirs alone holds the hashes of its guarantee's merged record, and ours otherwise (a `witnesses` entry,
which no record keys, always), so an incoming side's stale value can't re-key the entry silently, and a kept one the
merged tree doesn't state fails that Lean check as a witness failure (`judged`).

Every entry taken from one side where the sides disagree is named on stderr, one line per key. A side that is not a
record (not JSON, not an object, or holding both `pins` and `guarantees`) is the other side's record, named; only when
neither side is one, or a guard below refuses, does the merge leave git's text merge of the file, as a conflict. Exit 0: merged, the record written
to OURS. Exit 1: neither side a record, or a guard refused. A merged record is a candidate, never a verdict: `check`'s Lean
audit rebuilds every record of the merged tree and fails on any difference, as for any record; the lock run after landing
regenerates the records, and the landing DM shows each one that changed against `main`.

Git runs a driver from the worktree of the side merged into, so a branch whose checkout took older merge rules would merge
`main` by those: an older copy that merged `main` into an old branch collapsed `Security`'s 1,316 `reads` groups to one. The
command `--install` registers (`driver`, the one `research.clone.RECORD_DRIVER` holds) runs `origin/main`'s copy of this
directory wherever the repository has that ref, and the worktree's otherwise; a path already absolute (`research queue`'s
`base_drivers`, the drivers of the tip it merges onto) runs as given. Three guards leave git's conflict, saying why on
stderr, where a merge could lose what this copy doesn't know, and they are the one case where a conflict beats the merge:
a conflict is seen and resolved, while a silent loss lands.

- A side's lock is in a `format` newer than this copy's `FORMAT` (`check.py` writes its own, held equal to this one): merge
  `main`'s `tools/verity/lean` first.
- Git runs the worktree's own copy while the side `git merge` merges in (`GITHEAD_<commit>`) has `merge.py` changes this
  checkout lacks (`stale`).
- The merged record drops more than half of a side's entries in a section (the `reads` groups, the guarantees, any
  section's keys or items) that the base doesn't account for, as the other side's removal of an entry the base had
  (`losses`).

The layer rules (`layers`), which two branches that each widen a layer for a guarantee both change, merge field by field: `imports`
and `except` as lists, `why` as a list of its `; ` clauses (`layers`). Every list of names (the roots, `meaning`, `proved_in`,
`claims.modules`, `shape.premises`...), which two branches that each add a module both change, merges as a list (`items`). For a `.lean` file (`--lean`, `lean-imports` in
`.gitattributes`), git's own merge runs and only a conflict all of whose sides are import lines (each one import in any form
`headers.py` reads: `public import`, `meta import`, `import all`) is merged, as a list: two branches
that each add a guarantee's module to `Guarantees.lean` merge, and any other conflict stays git's. When one side's file is on
the module system and the other's isn't (a branch made before the port), the one that isn't is ported first (`modules.py`), as
rerunning the codemod on that branch would. For a `lakefile.toml`
(`--lakefile`, `lakefile-lists`), a conflict all of whose sides are `[[require]]` blocks merges as a set of blocks, and one all
of whose sides are a string array's items (`roots`) merges as a list, wrapped as ours wraps it: two branches that each require a
package or add a root to `Security/Properties` merge. Any other conflict, and a result that is not TOML, stays git's. For a
`pyproject.toml` (`--pyproject`, `pyproject-test-lists`), the same merge of string arrays stands only when every value outside
`[tool.verity.tests]`'s lists (`inputs`, `untraced`, a `reads` entry's `paths` and `tests`) is the three-way merge of the
sides: two branches that each add a file a suite reads merge, and anything else stays git's. In
both, an import, a require or a root that one side removed stays removed, judged on the whole files (`kept`), so a line the
other side moved can't bring it back. For `tests/test_lean_packages.py` (`--caps`, `lean-caps`), each of its ratchet caps
(`CAP_NAMES`: `UNWITNESSED = {...}`, `INLINE = {...}`...) merges over the whole files key by key as the larger of the two
sides' caps (a key a side lacks counting as the base's), so two branches that each move a cap merge, a cap one side brings
is that side's, and a conflict whose sides differ only in comment lines takes theirs'; any other conflict stays git's.
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import os
import re
import subprocess
import tempfile
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
#: `tests/test_lean_packages.py`'s driver (`--caps`)
CAPS = "lean-caps"
#: the pyprojects' driver (`--pyproject`), and the string lists of `[tool.verity.tests]` (and its `reads` entries) it merges
#: as lists: what a suite reads, which two branches that each add a file a suite reads both change
PYPROJECT = "pyproject-test-lists"
TEST_LISTS = {"inputs", "untraced", "paths", "tests"}
#: a lakefile's `[[require]]` block line, and a line of a string array's items (`roots = ["A", "B",`, `"C"]`)
REQUIRE_LINE = re.compile(r'\s*(\[\[require\]\]|(name|path|git|rev|scope|version|subDir)\s*=\s*\S.*)\s*$')
ITEM_LINE = re.compile(r'\s*(\w+\s*=\s*\[)?\s*("[^"]*"\s*,?\s*)+\]?\s*$')
#: the ratchet caps of `tests/test_lean_packages.py`, each on one line there
CAP_NAMES = ("UNWITNESSED", "INLINE", "ROOM", "TREES_EXCEPT", "FOUR_NAMES_EXCEPT", "UNDOCUMENTED", "SPEC_THEOREMS", "OVERLONG",
             "RETIRED_NAMES", "IMPORT_ALL_USES")
#: a ratchet cap's line (`UNWITNESSED = {"Security": 355, "verity/protocols/flock/verifier": 7}`), and one of its entries
CAP_LINE = re.compile(rf'(\s*(?:{"|".join(CAP_NAMES)}) = )\{{([^{{}}]*)\}}\s*$')
CAP_ENTRY = re.compile(r'\s*"([^"\\]*)"\s*:\s*(\d+)\s*')
#: where a merged string array wraps
WIDTH = 120
#: the sections `check.py --update` writes in sorted order; the others keep ours' order, with theirs' new entries after it
SORTED = {"assumptions", "domain", "open", "guarantees", "reads", "dependencies", "compile_time", "trust_gap"}
#: sections whose entries are maps merged entry by entry in turn (`upstream`'s `scan` and `watch`, `shape`'s `properties`,
#: `frozen` and `conjectures`, `trees`' `except`, `trust`'s `base` and `extra`), or lists of names merged as lists
#: (`trust_gap`'s premises, so a side's discharge and the other's new premise both apply)
NESTED = {"upstream", "shape", "claims", "trees", "trust", "trust_gap"}
#: nested maps whose entries are maps merged one level deeper still: `shape.inline`'s guarantees, each {premise: why} merged
#: premise by premise, a guarantee left with none dropped (the check refuses an empty one, and `--update` drops it), and
#: `trust.extra`'s, each {premise: reason} merged the same way
DEEPER = {"shape[inline]", "trust[extra]"}
#: the witness rule's sections, merged entry by entry with theirs' value where the sides disagree (`judged`)
JUDGED = {"unwitnessed", "witnesses"}
MISSING = object()
#: the newest lock `format` this file merges: bump it with `check.py`'s `FORMAT`, which writes it, so a copy older than a
#: lock leaves it to git's conflict
FORMAT = 1
#: this directory, from the top of the worktree; the registered drivers run `origin/main`'s copy of it (`driver`)
DIR = "tools/verity/lean"
#: sections an entry sits in at most one of (`exclusive`), by the section holding them (None: the record): a claim is
#: witnessed, excused or generic, and a premise is a named assumption, a domain hypothesis or open
EXCLUSIVE = (("claims", ("witnesses", "not", "generic")), (None, ("assumptions", "domain", "open")))


def one(path: str, o, a, b, why: list[str]):
    """The three-way merge of one value: a side's change applies.  Where the sides disagree it still merges, naming ``path``
    in ``why``: a value one side removed and the other changed is the changed one, and one both sides added or changed
    differently is theirs'."""
    if a == b:
        return a
    if a == o:
        return b
    if b == o:
        return a
    if a is MISSING or b is MISSING:
        kept = "ours" if b is MISSING else "theirs"
        why.append(f"{path}: removed on one side and changed on the other: kept {kept}' change")
        return b if a is MISSING else a
    why.append(f"{path}: {'added' if o is MISSING else 'changed'} on both sides, differently: took theirs")
    return b


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
    """Maps merged entry by entry (`one`); ``depth`` > 1 merges entries that are maps entry by entry too (one level more
    under a `DEEPER` path), and lists of names merge as lists."""
    out = {}
    for k in keys(path, a, b):
        x, y, z = o.get(k, MISSING), a.get(k, MISSING), b.get(k, MISSING)
        if depth > 1 and isinstance(y, dict) and isinstance(z, dict) and (x is MISSING or isinstance(x, dict)):
            sub = f"{path}[{k}]"
            v = entries(sub, {} if x is MISSING else x, y, z, why, depth - 1 + (sub in DEEPER))
            v = MISSING if path in DEEPER and not v else v
        elif depth > 1 and names(x, y, z):
            v = items([] if x is MISSING else x, y, z)
        else:
            v = one(f"{path}[{k}]", x, y, z, why)
        if v is not MISSING:
            out[k] = v
    return out


def at_record(v, rec) -> bool:
    """Whether the `unwitnessed` value ``v`` holds the hashes of ``rec``, its guarantee's merged record: the type hash and
    the `unfolded_hash`, which neither a bare hash nor a record of a guarantee not stated as a definition has."""
    e = A.unwitnessed_entry(v)
    return (isinstance(rec, dict) and e is not None and e[0] == rec.get("type_hash")
            and A.unwitnessed_unfolded(v) == rec.get("unfolded_hash"))


def judged(section: str, o: dict, a: dict, b: dict, candidate: list[str], records: dict) -> dict:
    """A `JUDGED` section merged entry by entry as `one` merges it, the entries the sides disagree on named in ``candidate``
    with the side taken.  An entry both sides changed differently is theirs' only when theirs alone holds the hashes of
    its guarantee's merged record (``records``, `at_record`), and ours' otherwise, a `witnesses` entry, which no record
    keys, always: a stale value on the incoming side would re-key the entry silently, while one kept from ours that the
    merged tree doesn't state fails its Lean check as a witness failure.  An entry one side removed is the changed one."""
    out = {}
    for k in keys(section, a, b):
        x, y, z = o.get(k, MISSING), a.get(k, MISSING), b.get(k, MISSING)
        if MISSING not in (y, z) and y != z and x not in (y, z):
            rec = records.get(k)
            v = z if section == "unwitnessed" and at_record(z, rec) and not at_record(y, rec) else y
            if section != "unwitnessed":
                side = "ours"
            elif v is z:
                side = "theirs, at its merged record"
            else:
                side = "ours, at its merged record" if at_record(y, rec) else "ours, neither side at its merged record"
            candidate.append(f"{section}[{k}] ({side})")
        else:
            why: list[str] = []
            v = one(f"{section}[{k}]", x, y, z, why)
            if why:
                candidate.append(f"{section}[{k}] ({'ours' if z is MISSING else 'theirs'})")
        if v is not MISSING:
            out[k] = v
    return out


def derived(path: str, o, a, b, candidate: list[str]):
    """The three-way merge of a value the build determines (a definition's hash, an assumption's digest, a guarantee's
    signature, named assumptions and type hash): as `one`, named in ``candidate`` rather than ``why``.  A value both sides
    changed differently is theirs' (named ``path (theirs)``), and one removed on one side and changed on the other is kept
    (the union, named ``path``): the lock run after landing regenerates it from the merged tree."""
    if a == b or a == o:
        return b
    if b == o:
        return a
    if a is MISSING or b is MISSING:
        candidate.append(path)
        return b if a is MISSING else a
    candidate.append(f"{path} (theirs)")
    return b


def owned(rec, k: str = "owner"):
    return MISSING if rec is MISSING else rec.get(k, MISSING)


def guarantees(o: dict, a: dict, b: dict, why: list[str], candidate: list[str]) -> dict:
    """The guarantees entry by entry, each one's `owner` apart from the rest of its record: a branch that restates a guarantee
    and one that gives it an owner merge.  The rest of a record (signature, named assumptions, type hash) is the build's, so
    two sides that restated it differently merge as theirs' record (`derived`), which the lock run after landing rebuilds
    from the merged statement.  Whether a guarantee is listed is not the build's: one side's removal applies over the
    other's change to its record, and only a change to its owner, which a person writes, keeps it (`one`)."""
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


def assumptions(o: dict, a: dict, b: dict, why: list[str], candidate: list[str], section: str = "assumptions") -> dict:
    """The named assumptions (or, ``section`` `open` or `domain`, the open premises or the domain hypotheses) entry by
    entry, each one's hand-written fields (`owner`; an open premise's `issue` and `due`, a domain hypothesis's `why` and
    `cites`) apart from its `digest` (the build's: the hash its `reads` record holds), which merges as `derived`; a field
    both sides changed differently is theirs' (`one`).  An entry one side removed and the other changed is kept: the lock
    run drops it when no guarantee of the merged tree reads it (`assumptions.unread`, `open.unread`, `domain.unread`)."""
    out = {}
    for k in keys(section, a, b):
        x, y, z = o.get(k, MISSING), a.get(k, MISSING), b.get(k, MISSING)
        if not all(isinstance(r, dict) for r in (x, y, z) if r is not MISSING):
            v = one(f"{section}[{k}]", x, y, z, why)
        elif MISSING in (y, z):
            v = derived(f"{section}[{k}]", x, y, z, candidate)
        else:
            v = {}
            for f in ("owner", "digest", "issue", "due", "why", "cites"):
                w = (derived if f == "digest" else one)(f"{section}[{k}].{f}", owned(x, f), owned(y, f), owned(z, f),
                                                        candidate if f == "digest" else why)
                v |= {} if w is MISSING else {f: w}
        if v is not MISSING:
            out[k] = v
    return out


def core(group):
    """What a `reads` entry says, its digest (which it determines) aside."""
    return None if group is None else A.definitions(group)


def group(m: str, x: dict, y: dict, z: dict, why: list[str], candidate: list[str], dropped: dict | None = None) -> dict:
    """One module's `reads` entry, both sides having changed it: definitions entry by entry.  A definition one side no
    longer reads while the other still reads it stays, with the other side's hash, unless the dropping side's `gone`
    (``dropped``, by "ours" and "theirs") says the merged tree reads it nowhere: the union is the candidate
    (``candidate``), and the lock run after landing regenerates the group.  A hash both sides changed differently is
    theirs' (`derived`)."""
    xd, yd, zd = A.definitions(x), A.definitions(y), A.definitions(z)
    defs = {}
    for d in sorted(set(xd) | set(yd) | set(zd)):
        p, q, r = xd.get(d, MISSING), yd.get(d, MISSING), zd.get(d, MISSING)
        if p is not MISSING and (q is MISSING) != (r is MISSING):
            h = q if r is MISSING else r
            if dropped and dropped["ours" if q is MISSING else "theirs"](d, h != p):
                continue
            defs[d] = h
            candidate.append(f"reads[{m}].definitions[{d}]")
        elif (v := derived(f"reads[{m}].definitions[{d}]", p, q, r, candidate)) is not MISSING:
            defs[d] = v
    if len({len(h) for h in defs.values() if isinstance(h, str)}) > 1:
        candidate.append(f"reads[{m}] (32-bit and SHA-256 hashes mixed)")
    return {"definitions": defs}


def gone(m: str, side: dict, other_new: bool, candidate: list[str]):
    """Whether a definition of module ``m`` that ``side`` (its `reads`) lists nowhere, while the other side lists it, leaves
    the merged record, as `group` asks (with whether the other side changed its hash).  None of that side's guarantees
    reads it, and a guarantee the other side left as the base had it is in the merged tree as that side states it, or
    removed, so only the other side's new or restated guarantees (``other_new``) may read it there.  Without any, the
    removal applies.  With some, it applies while the other side kept the definition's hash, named in ``candidate``: the
    union would bring back a read of a definition the merged tree may no longer have (a `J10R.Layout` renamed
    `NoiseLayout` would stay beside the new name); where the other side changed the hash, the union stays."""
    listed = {d for g in side.values() for d in A.definitions(g)}

    def ask(d: str, rehashed: bool) -> bool:
        if d in listed or (other_new and rehashed):
            return False
        if other_new:
            candidate.append(f"reads[{m}].definitions[{d}] (removed)")
        return True
    return ask


def reads(o: dict, a: dict, b: dict, why: list[str], candidate: list[str],
          records: tuple[dict, dict, dict] = ({}, {}, {})) -> dict:
    """The `reads` sections merged module by module (`group`), ``records`` being the base's, ours' and theirs' guarantees,
    of which each side's new or restated ones (`audit.restated`) may read what the other side reads nowhere.  A module left
    without definitions is dropped."""
    base = {"guarantees": records[0], "reads": o}
    new = {"ours": bool(A.restated(base, {"guarantees": records[1], "reads": a})),
           "theirs": bool(A.restated(base, {"guarantees": records[2], "reads": b}))}
    out = {}
    for m in sorted(set(a) | set(b)):
        x, y, z = o.get(m), a.get(m), b.get(m)
        if core(y) == core(z):
            g = y
        elif core(y) == core(x):
            g = z
        elif core(z) == core(x):
            g = y
        else:
            drop = {"ours": gone(m, a, new["theirs"], candidate), "theirs": gone(m, b, new["ours"], candidate)}
            if y is None or z is None:
                g = group(m, x or {}, y or {}, z or {}, why, [], drop)
                candidate.append(f"reads[{m}]")
            else:
                g = group(m, x or {}, y, z, why, candidate, drop)
        if g is not None:
            out[m] = {"definitions": dict(sorted(A.definitions(g).items()))}
    moved(o, a, b, out, candidate, new)
    for m in [m for m, g in out.items() if not g["definitions"]]:
        del out[m]
        if f"reads[{m}]" not in candidate:
            candidate.append(f"reads[{m}]")
    return {m: {"digest": A.module_digest(g["definitions"]), **g} for m, g in out.items()}


def moved(o: dict, a: dict, b: dict, out: dict, candidate: list[str], new: dict[str, bool]) -> None:
    """A Lean declaration lives in one module: a definition one side moved (it added it under another module of the merged
    table and no longer lists it under this one) leaves the module the other side's group kept it in, and is no longer the
    candidate `group` named.  Its hash is the mover's; when the other side changed it in the old module, or added or
    restated guarantees (``new``, by side), which may read it in the new one, the new module's group is the candidate."""
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
            kept = "ours" if d in defs(a, m) else "theirs"
            if defs(a if kept == "ours" else b, m).get(d) != defs(o, m).get(d) or new[kept]:
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


def exclusive(base: dict, ours: dict, theirs: dict, why: list[str]) -> tuple[dict, dict]:
    """Ours and theirs with each entry of an `EXCLUSIVE` group where the side that moved it put it: an entry one side moved
    between the group's sections from the base leaves the sections the other side still has it in, whatever that side
    changed there (`Claims.ZKOutLeak`, which `main` moved from `claims.not` to `claims.generic`, kept under both beside a
    branch's older reason, and the Lean check refuses an entry in two sections).  An entry both sides placed differently,
    each away from the base, is where theirs put it, with theirs' value, named in ``why``: `check.py --update` on the merged
    tree classifies it.  A side that removed an entry places it nowhere, and the sections merge it as any removal."""
    sides = [base, ours, theirs]
    for parent, group in EXCLUSIVE:
        def held(rec: dict) -> dict:
            top = rec if parent is None else rec.get(parent)
            return {s: top[s] if isinstance(top, dict) and isinstance(top.get(s), dict) else {} for s in group}
        o, a, b = (held(s) for s in sides)
        for n in sorted({n for side in (o, a, b) for s in group for n in side[s]}):
            x, y, z = (tuple(s for s in group if n in side[s]) for side in (o, a, b))
            if y == z or () in (y, z):
                continue
            both = x not in (y, z)
            if both:
                stale, kept, left = 1, z, y
                where = " and ".join(f"{s}[{n}]" if parent is None else f"{parent}.{s}[{n}]" for s in (*y, *z))
                why.append(f"{n}: ours put it under {', '.join(y)} and theirs under {', '.join(z)}, from "
                           f"{', '.join(x) or 'neither'} in the base ({where}): took theirs, which `check.py --update` "
                           f"classifies")
            else:
                stale, kept, left = (1, z, x) if y == x else (2, y, x)
            sides[stale] = rec = dict(sides[stale])
            if parent is not None:
                rec[parent] = rec = dict(rec[parent])
            for s in set(left) - set(kept):
                rec[s] = {k: v for k, v in rec[s].items() if k != n}
            if both:
                for s in z:
                    rec[s] = (rec[s] if isinstance(rec.get(s), dict) else {}) | {n: b[s][n]}
    return sides[1], sides[2]


def merge(base: dict, ours: dict, theirs: dict, candidate: list[str] | None = None) -> tuple[dict, list[str]]:
    """(the merged record, the entries a person writes that the sides disagreed on, each with the side taken: empty when
    none).  A pure function of the three records, which never refuses.  ``candidate`` collects the records the build
    determines (`RECORD_DIFFERENCE`'s kinds: guarantees' records, `reads`, assumptions' digests, `JUDGED` entries) merged
    although the sides disagree on them: the union or theirs' value, which the lock run after landing regenerates, or the
    side `judged` keeps; and the `shape.inline` premises `held` keeps."""
    why: list[str] = []
    candidate = [] if candidate is None else candidate
    ours, theirs = exclusive(base, ours, theirs, why)
    o, a, b = (s.get("guarantees", MISSING) for s in (base, ours, theirs))
    records = guarantees({} if o is MISSING else o, a, b, [], []) if (
        isinstance(a, dict) and isinstance(b, dict) and (o is MISSING or isinstance(o, dict))) else {}
    out = {}
    for k in keys("", ours, theirs):
        o, a, b = base.get(k, MISSING), ours.get(k, MISSING), theirs.get(k, MISSING)
        if k == "reads":
            records = tuple(s.get("guarantees") if isinstance(s.get("guarantees"), dict) else {} for s in (base, ours, theirs))
            v = reads({} if o is MISSING else o, {} if a is MISSING else a, {} if b is MISSING else b, why, candidate, records)
        elif k == "guarantees" and isinstance(a, dict) and isinstance(b, dict) and (o is MISSING or isinstance(o, dict)):
            v = guarantees({} if o is MISSING else o, a, b, why, candidate)
        elif k in ("assumptions", "domain", "open") and all(isinstance(s, dict) for s in (o, a, b) if s is not MISSING):
            v = assumptions(*({} if s is MISSING else s for s in (o, a, b)), why, candidate, k)
            v = MISSING if not v and MISSING in (a, b) else v
        elif k in JUDGED and isinstance(a, dict) and isinstance(b, dict) and (o is MISSING or isinstance(o, dict)):
            v = judged(k, {} if o is MISSING else o, a, b, candidate, records)
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
    held(base, ours, theirs, out, candidate)
    return out, why


def held(base: dict, ours: dict, theirs: dict, out: dict, candidate: list[str]) -> None:
    """Put back into the merged record ``out`` each `shape.inline` premise of a guarantee ``out`` lists that one side dropped
    while the other has it with the base's reason and states the guarantee otherwise than both the base and the dropping
    side (`restated`), each named in ``candidate`` as `shape[inline][G][P] (kept)`; any other such premise goes, as a
    three-way merge drops it.  A side whose `--update` ran on an older tree drops a text the other side's restated statement
    still prints (a renamed definition's new text, which the base already listed beside the old one), and the Lean check
    fails on an offending premise of a listed guarantee that `shape.inline` doesn't list, while `--update` adds none
    (`check.py`'s `inline_failures`).  A side that states the guarantee as the base or the dropping side does prints it,
    in the merged tree, as the dropping side's tree does, where that side's `--update` found the premise no longer offends:
    kept, the entry would be residue (a renamed premise's old text beside its new one), which fails nothing but counts
    against `INLINE` until an `--update` drops it.  A guarantee ``out`` doesn't list gets nothing back: `--update` keeps an
    unlisted guarantee's entries until the lock lists it."""
    def inline(s: dict) -> dict:
        spec = s.get("shape")
        v = spec.get("inline") if isinstance(spec, dict) else None
        return v if isinstance(v, dict) else {}

    def premises(s: dict, g: str) -> dict | None:
        v = s.get(g, {})
        return v if isinstance(v, dict) else None

    sides = (base, ours, theirs)

    @functools.cache
    def hashes(i: int) -> dict:
        r = sides[i].get("reads")
        return {d: h for grp in (r.values() if isinstance(r, dict) else ()) if isinstance(grp, dict)
                for d, h in A.definitions(grp).items()}

    def record(i: int, g: str):
        r = sides[i].get("guarantees")
        r = r.get(g) if isinstance(r, dict) else None
        return {k: v for k, v in r.items() if k != "owner"} if isinstance(r, dict) else r

    def restated(i: int, j: int, g: str) -> bool:
        """Whether side ``j`` (`sides`' index) states ``g`` otherwise than side ``i``: its record, its owner aside, or the
        hash of a definition either's signature names where both read it (the queue's `dropped_entries` tells a restated
        guarantee so)."""
        recs = record(i, g), record(j, g)
        names = {n for r in recs if isinstance(r, dict) for s in r.get("signature") or [] if isinstance(s, str)
                 for n in A.NAMES.findall(s)}
        x, y = hashes(i), hashes(j)
        return recs[0] != recs[1] or any(x[n] != y[n] for n in names if n in x and n in y)
    spec, listed = out.get("shape"), out.get("guarantees")
    if not isinstance(spec, dict) or not isinstance(listed, dict) or not isinstance(spec.get("inline", {}), dict):
        return
    o, a, b, merged = inline(base), inline(ours), inline(theirs), dict(spec.get("inline", {}))
    for g in o:
        x, y, z, m = premises(o, g), premises(a, g), premises(b, g), premises(merged, g)
        if g not in listed or None in (x, y, z, m):
            continue
        keep = {}
        for p, w in x.items():
            if (p in y) == (p in z) or (y if p in y else z)[p] != w:
                continue
            k, d = (1, 2) if p in y else (2, 1)
            if restated(0, k, g) and restated(d, k, g):
                keep[p] = w
        if keep:
            merged[g] = m | keep
            candidate.extend(f"shape[inline][{g}][{p}] (kept)" for p in keep)
    if merged != spec.get("inline", {}):
        spec["inline"] = merged


def kept(o: set, a: set, b: set) -> set:
    """The names a three-way merge of whole files keeps: (A ∩ B) ∪ (A − O) ∪ (B − O), what both sides have and what either
    added.  A name one side removed while the other kept it stays removed, however git splits the file into hunks: a line
    one side moved shows in a hunk as an addition, and the hunk alone would bring back what the other side removed
    (a retired import, such as `Definitions.Accounting`'s, would come back and stop Security building)."""
    return (a & b) | (a - o) | (b - o)


@functools.cache
def _headers():
    """`headers.py`, loaded only to merge a `.lean` file: the record driver runs where `merge.py` and `check.py` are alone."""
    return A.sibling("headers")


def _import(line: str) -> bool:
    """Whether ``line`` is one import and nothing else, in any form a header takes (`headers.py`: `public import`...)."""
    H = _headers()
    h = H.header(line)
    return h is not None and len(h.imports) == 1 and not h.module and not h.prelude and H.skip(line, h.end) == len(line)


def _imports(lines: list[str]) -> bool:
    return all(_import(s) or not s.strip() for s in lines)


@functools.cache
def _modules():
    """`modules.py`, loaded only to merge a `.lean` file: the record driver runs where `merge.py` and `check.py` are alone."""
    return A.sibling("modules")


def _ported(sides: tuple[Path, Path, Path], tmp: Path) -> tuple[Path, Path, Path]:
    """A `.lean` file's three sides, each that isn't on the module system ported (`modules.py`, its copy under ``tmp``) when
    ours or theirs is: what a branch made before the port takes by rerunning the codemod.  Unported, the port's change to
    every header line conflicts with that branch's imports, and an import either side removed is still in the other's
    spelling.  A side the codemod refuses stays as it is."""
    M = _modules()
    texts = [p.read_text() for p in sides]
    if not any(M.is_module(t) for t in texts[1:]):
        return sides
    out = []
    for name, p, t in zip(("base", "ours", "theirs"), sides, texts):
        try:
            new = M.port(t)
        except ValueError as e:
            print(f"merge.py --lean: the {name} side stays off the module system: {e}", file=sys.stderr)
            new = t
        if new != t:
            (p := tmp / name).write_text(new)
        out.append(p)
    return out[0], out[1], out[2]


def _import_names(text: str) -> set[str]:
    return {s.strip() for s in text.splitlines() if _import(s)}


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
    def opening(side):
        return m.group(1) if side and (m := ITEM_LINE.match(side[0])) and m.group(1) else None

    def closes(side):
        return bool(side) and side[-1].rstrip().endswith("]")
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


def pyproject(base: Path, ours: Path, theirs: Path) -> tuple[str, int]:
    """(the merged text, the number of conflicts left) of a `pyproject.toml`: `lakefile`'s merge of string arrays, kept only
    when it changed nothing but `[tool.verity.tests]`'s lists (`TEST_LISTS`, each now `kept` of the sides' items) and every
    other value is the three-way merge of the sides; otherwise git's own merge, its conflicts left."""
    text, left = lakefile(base, ours, theirs)
    if left:
        return text, left
    docs = [tomllib.loads(p.read_text()) for p in (base, ours, theirs)] + [tomllib.loads(text)]
    flat = [_flat(d) for d in docs]
    for k in set().union(*flat):
        o, a, b, m = (f.get(k, MISSING) for f in flat)
        if k[0] == "lists":
            want = kept(*(set(x) if x is not MISSING else set() for x in (o, a, b)))
            if set(m if m is not MISSING else ()) != want:
                break
        elif m != (a if a == b or b == o else b if a == o else None):
            break
    else:
        return text, 0
    return lean(base, ours, theirs, ())


def _flat(doc, path: tuple = ()) -> dict:
    """``doc``'s values by path, each `[tool.verity.tests]` string list (`TEST_LISTS`) as `("lists", path)`."""
    out = {}
    if isinstance(doc, dict):
        for k, v in doc.items():
            out.update(_flat(v, (*path, k)))
    elif isinstance(doc, list) and doc and all(isinstance(x, dict) for x in doc):
        for i, v in enumerate(doc):
            out.update(_flat(v, (*path, i)))
    elif path[:3] == ("tool", "verity", "tests") and path[-1] in TEST_LISTS and isinstance(doc, list) \
            and all(isinstance(x, str) for x in doc):
        out[("lists", path)] = doc
    else:
        out[("value", path)] = doc
    return out


def lean(base: Path, ours: Path, theirs: Path, resolvers=None) -> tuple[str, int]:
    """(the merged text, the number of conflicts left) of a `.lean` file: git's own merge (`git merge-file`), then each conflict
    whose three sides are only `import` lines merged as lists of imports (`items`), so two branches that each add a module to
    an import list merge.  Every import outside a conflict left is then held to the whole files' imports (`kept`), once
    each: one side removed stays removed wherever git's hunks put it.  Any other conflict is left as git writes it.  When
    ours or theirs is on the module system, the sides that aren't are ported first (`_ported`), so the merge is of what
    each side changed beyond the port.  ``resolvers`` (each `(base, ours, theirs)` lines to the merged lines, or None)
    replace the imports rule, and with them no side is ported and no import is filtered."""
    keep = None
    with tempfile.TemporaryDirectory() as tmp:
        if resolvers is None:
            base, ours, theirs = _ported((base, ours, theirs), Path(tmp))
            keep = kept(*(_import_names(p.read_text()) for p in (base, ours, theirs)))
            resolvers = (lambda o, a, b: [s for s in items([s for s in o if s.strip()], [s for s in a if s.strip()],
                                                           [s for s in b if s.strip()])]
                         if _imports(a) and _imports(o) and _imports(b) else None,)
        p = subprocess.run(["git", "merge-file", "-p", "--diff3", "-L", "ours", "-L", "base", "-L", "theirs", str(ours),
                            str(base), str(theirs)], capture_output=True, text=True)
    if p.returncode < 0 or p.returncode > 127:
        raise ValueError(f"git merge-file failed: {p.stderr.strip()}")
    out, left, lines, n, seen = [], 0, p.stdout.splitlines(keepends=True), 0, set()

    def emit(s: str):
        if keep is not None and _import(s):
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


def _cap_lines(lines: list[str]) -> dict[str, dict[str, int]] | None:
    """A conflict side's ratchet caps ({its line up to `{`: its counts by key}), blank lines aside, or None when it holds
    anything else."""
    out: dict[str, dict[str, int]] = {}
    for s in lines:
        if not s.strip():
            continue
        m = CAP_LINE.match(s)
        if m is None or m.group(1) in out:
            return None
        found = [CAP_ENTRY.fullmatch(p) for p in m.group(2).split(",")] if m.group(2).strip() else []
        if None in found:
            return None
        out[m.group(1)] = {e.group(1): int(e.group(2)) for e in found}
    return out


def _maxed(base: dict[str, int], ours: dict[str, int], theirs: dict[str, int]) -> dict[str, int]:
    """One cap's counts merged as the larger of the two sides' (a key a side lacks counts as the base's there; one both sides
    lack is gone), in ours' order with theirs' new keys after the key theirs has before them.  Each side's check held its
    count under its cap, and a cap left above its count only warns (`_held`) until drive's lock run lowers it to the count
    and its room.  Counters (theirs plus ours' change) can take a cap below its count, where both sides lowered it for
    entries that overlap or from different bases, and then fail a PR that did nothing wrong."""
    return {k: max(n for n in (ours.get(k, base.get(k)), theirs.get(k, base.get(k))) if n is not None)
            for k in items([], list(ours), list(theirs))}


def _cap_line(head: str, counts: dict[str, int], end: str) -> str:
    return head + "{" + ", ".join(f'"{k}": {n}' for k, n in counts.items()) + "}" + end


def _caps(o: list[str], a: list[str], b: list[str]) -> list[str] | None:
    """A conflict all of whose sides are the same ratchet caps' lines, each merged as the larger (`_maxed`).  None when a
    side holds anything else or the sides hold different caps."""
    x, y, z = _cap_lines(o), _cap_lines(a), _cap_lines(b)
    if x is None or y is None or z is None or not y or not set(x) == set(y) == set(z):
        return None
    out = []
    for s in a:
        m = CAP_LINE.match(s)
        if m is None:
            out.append(s)
            continue
        out.append(_cap_line(m.group(1), _maxed(x[m.group(1)], y[m.group(1)], z[m.group(1)]), "\n" if s.endswith("\n") else ""))
    return out


def _comments(o: list[str], a: list[str], b: list[str]) -> list[str] | None:
    """A conflict whose sides differ only in comment lines (`#`), blank lines aside: theirs'.  None when their code differs."""
    def code(side):
        return [s for s in side if s.strip() and not s.lstrip().startswith("#")]
    return list(b) if code(a) == code(b) else None


def _file_caps(text: str) -> dict[str, tuple[str, dict[str, int]]] | None:
    """A whole file's ratchet caps ({its line up to `{`: (the line, its counts by key)}), or None when a cap's line appears
    twice (a side git left in conflict) or holds anything but counts."""
    out: dict[str, tuple[str, dict[str, int]]] = {}
    for s in text.splitlines(keepends=True):
        if CAP_LINE.match(s) is None:
            continue
        got = _cap_lines([s])
        if got is None:
            return None
        (head, counts), = got.items()
        if head in out:
            return None
        out[head] = (s, counts)
    return out


def caps(base: Path, ours: Path, theirs: Path) -> tuple[str, int]:
    """(the merged text, the number of conflicts left) of `tests/test_lean_packages.py`.  Each ratchet cap merges over the
    whole files: a line both sides changed alike as that line, else key by key as the larger (`_maxed`, the base's count
    standing in for a key a side lacks, so a cap both sides brought over a base without it is the larger of theirs too).
    While git's own merge runs, each cap's line stands as one placeholder (`HEAD{}`) on every side, so the count never
    conflicts with the comment above it, and a conflict whose sides differ only in comment lines takes theirs'
    (`_comments`); the merged counts then replace the placeholders.  So two branches that each move a cap and reword its
    comment merge, whatever the bases git's recursive merge combines (it runs this driver on them too).  A cap only a side
    holds (a new cap, which the base lacks too) is that side's line.  Where a side's caps can't be read whole (a cap's line
    twice, a side without a cap the base holds), the conflicts merge one by one instead: one whose sides are only cap lines
    as the larger (`_caps`), one that differs only in comments as theirs'.  Any other conflict is left as git writes it."""
    def one_by_one():
        return lean(base, ours, theirs, (_caps, _comments))

    texts = [p.read_text() for p in (base, ours, theirs)]
    found = [_file_caps(t) for t in texts]
    if None in found or not found[1] or not set(found[0]) <= set(found[1]) & set(found[2]):
        return one_by_one()
    merged: dict[str, str] = {}
    for head in [*found[1], *(h for h in found[2] if h not in found[1])]:
        if head not in found[1] or head not in found[2]:
            merged[head] = (found[1].get(head) or found[2][head])[0].rstrip("\n")
            continue
        line, counts = found[1][head]
        if found[2][head][0] == line:
            merged[head] = line.rstrip("\n")
        else:
            merged[head] = _cap_line(head, _maxed(found[0].get(head, ("", {}))[1], counts, found[2][head][1]), "")
    with tempfile.TemporaryDirectory() as tmp:
        sides = []
        for name, text in zip(("base", "ours", "theirs"), texts):
            (side := Path(tmp) / name).write_text("".join(
                m.group(1) + "{}" + ("\n" if s.endswith("\n") else "") if (m := CAP_LINE.match(s)) else s
                for s in text.splitlines(keepends=True)))
            sides.append(side)
        text, left = lean(*sides, (_comments,))
    lines = text.splitlines(keepends=True)
    for head, line in merged.items():
        at = [i for i, s in enumerate(lines) if s.rstrip("\n") == head + "{}"]
        if not at or (len(at) > 1 and not left):
            return one_by_one()
        for i in at:
            lines[i] = line + ("\n" if lines[i].endswith("\n") else "")
    return "".join(lines), left


def load(path: Path) -> dict:
    """The record at ``path`` under today's names, its `reads` groups without the readers an older record lists."""
    text = path.read_text()
    if not isinstance(doc := json.loads(text) if text.strip() else {}, dict):
        raise ValueError("not a JSON object")
    doc = A.as_guarantees(doc)
    for m in A.with_readers(doc):
        doc["reads"][m] = {k: v for k, v in doc["reads"][m].items() if k not in A.READER_KEYS}
    return doc


def driver(flag: str = "", where: str = DIR) -> str:
    """The command git runs for a driver whose mode ``flag`` names (`--caps `...): `origin/main`'s copy of ``where`` (this
    directory from the top of the worktree, where git runs drivers) when the repository has that ref, else the worktree's.
    `origin/main`'s copy is at least as new as any branch's, and the worktree's is the branch merged into's. A path made
    absolute (`research queue`'s `base_drivers`) runs as it is. The copy is kept per `origin/main` commit, at
    `<git common dir>/lean-merge/<full sha>/`: a call that finds none there extracts one into a temporary directory beside
    it (`git archive` to a file, then `tar`, each checked, and `merge.py` there) and renames it into place, so a killed
    extraction leaves nothing at the key, and of two calls racing for one key the rename of one wins and the other runs
    that copy; that call then keeps the two newest copies, each removed by renaming it away first. A call that can't keep
    one runs a temporary copy of its own; one that can extract neither, though `origin/main` resolves, leaves the file to
    git's conflict rather than merge it by the worktree's older copy."""
    m = f"{where}/merge.py"
    return (f"m={m}; f=$m; d=; x() {{ git archive -o \"$1.tar\" \"$s\" -- {where} ':(exclude){where}/tests' 2>/dev/null && "
            f"tar -x -C \"$1\" -f \"$1.tar\" 2>/dev/null && [ -f \"$1/$m\" ]; r=$?; rm -f \"$1.tar\"; return $r; }}; "
            f"case $m in /*) ;; *) if s=$(git rev-parse -q --verify 'origin/main^{{commit}}'); then f=; "
            f"if g=$(git rev-parse --git-common-dir) && b=$(cd \"$g\" && pwd)/lean-merge && mkdir -p \"$b\"; then c=$b/$s; "
            f"if [ ! -f \"$c/$m\" ] && t=$(mktemp -d \"$b/.tmp-XXXXXX\"); then x \"$t\" && python3 -c "
            f"'import os, sys; os.rename(*sys.argv[1:])' \"$t\" \"$c\" 2>/dev/null; rm -rf \"$t\"; "
            f"ls -t \"$b\" | grep -E '^[0-9a-f]{{40,64}}$' | tail -n +3 | while read -r o; do "
            f"mv \"$b/$o\" \"$b/.tmp-gone-$o\" 2>/dev/null && rm -rf \"$b/.tmp-gone-$o\"; done; "
            f"find \"$b\" -maxdepth 1 -name '.tmp-*' -mmin +60 -exec rm -rf {{}} + 2>/dev/null; fi; "
            f"[ -f \"$c/$m\" ] && f=$c/$m; fi; [ -n \"$f\" ] || {{ d=$(mktemp -d) && x \"$d\" && f=$d/$m; }}; "
            f"[ -n \"$f\" ] || {{ echo \"lean merge driver: origin/main ($s) resolves, but its {where} could not be extracted, "
            f"so this file is left to git's conflict, not merged by the worktree's older copy\" >&2; "
            f"[ -z \"$d\" ] || rm -rf \"$d\"; exit 1; }}; fi;; esac; "
            f"python3 \"$f\" {flag}%O %A %B %P; r=$?; [ -z \"$d\" ] || rm -rf \"$d\"; exit $r")


def stale() -> str | None:
    """Why this copy mustn't merge, or None: git runs the worktree's own copy (what `driver` falls back to without
    `origin/main`, or what an older registered command runs) while a side `git merge` merges in, which git names to its
    drivers as `GITHEAD_<commit>` (`settings.merged_in`), has `merge.py` changes this checkout lacks, so its records may
    hold what this copy can't merge. Git names that side only in `git merge` and `git pull`: `merge-tree`, a rebase and a
    cherry-pick merge as before, and a copy outside the worktree was chosen by whoever ran it, as was one of `merge.py` and
    `check.py` alone (a test's), which has no `settings.py`."""
    if not (HERE / "settings.py").is_file() or not (heads := A.sibling("settings").merged_in()):
        return None

    def git(*args: str) -> str:
        r = subprocess.run(["git", *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ""
    if not (top := git("rev-parse", "--show-toplevel")) or (rel := os.path.relpath(HERE, top)).startswith(".."):
        return None
    me = f"{Path(rel).as_posix()}/merge.py"
    mine = git("hash-object", "--", str(HERE / "merge.py"))
    for commit, name in sorted(heads.items()):
        theirs = git("rev-parse", "-q", "--verify", f"{commit}:{me}")
        if theirs and theirs != mine and git("rev-list", "-n1", commit, "^HEAD", "--", me):
            return (f"the side merged in, {name} ({commit[:10]}), has {me} {theirs[:10]}, with changes this checkout's "
                    f"{mine[:10]} lacks, and git ran this checkout's copy: `git merge --abort`, `git fetch origin main`, "
                    f"register the drivers that run origin/main's copy (`python3 {me} --install`), and merge again")
    return None


def losses(base: dict, ours: dict, theirs: dict, merged: dict) -> list[str]:
    """Each section of which ``merged`` drops more than half of a side's entries (a map's keys, a list's items) that the
    base doesn't account for. The base accounts for an entry it had that the other side removed; an entry of an `EXCLUSIVE`
    group of sections is kept wherever in the group the merge put it."""
    group = {s: g for parent, g in EXCLUSIVE if parent is None for s in g}

    def held(rec: dict, k: str) -> set:
        out = set()
        for s in group.get(k, (k,)):
            v = rec.get(s)
            out |= set(v) if isinstance(v, dict) else {json.dumps(i, sort_keys=True) for i in v} if isinstance(v, list) else set()
        return out
    out = []
    for side, rec, other in (("ours", ours, theirs), ("theirs", theirs, ours)):
        for k, v in rec.items():
            if not isinstance(v, (dict, list)) or not v:
                continue
            have = set(v) if isinstance(v, dict) else {json.dumps(i, sort_keys=True) for i in v}
            gone = have - held(merged, k) - (held(base, k) - held(other, k))
            if 2 * len(gone) > len(have):
                out.append(f"{k}: the merge keeps {len(have) - len(gone)} of {side}' {len(have)}, losing {len(gone)} the base "
                           f"doesn't account for ({', '.join(sorted(gone)[:3])}{', ...' if len(gone) > 3 else ''})")
    return out


def conflict(base: Path, ours: Path, theirs: Path, path: Path, why: str) -> int:
    """Leave ``path`` to git's text merge, written to OURS, saying ``why`` on stderr: exit 1, which git reports as a
    conflict."""
    ours.write_text(lean(base, ours, theirs, ())[0])
    print(f"merge.py: {path}: {why}: left to git's merge, as a conflict", file=sys.stderr)
    return 1


def install(repo: Path) -> None:
    """Register the drivers in ``repo``'s git config: `driver`'s command for this directory's path from the top of the
    worktree (git runs drivers there)."""
    top = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=repo, capture_output=True, text=True,
                              check=True).stdout.strip())
    where = Path(os.path.relpath(HERE, top)).as_posix()
    for name, what, argv in ((NAME, "lean-audit.json: each side's guarantees, reads and dependencies", driver("", where)),
                             (LEAN, ".lean: each side's imports", driver("--lean ", where)),
                             (LAKEFILE, "lakefile.toml: each side's requires and roots", driver("--lakefile ", where)),
                             (PYPROJECT, "pyproject.toml: each side's [tool.verity.tests] lists", driver("--pyproject ", where)),
                             (CAPS, "tests/test_lean_packages.py: each side's change to the ratchet caps",
                              driver("--caps ", where))):
        for k, v in (("name", f"{what} (tools/verity/lean/merge.py)"), ("driver", argv)):
            subprocess.run(["git", "config", f"merge.{name}.{k}", v], cwd=repo, check=True)
        print(f"merge.py: merge.{name}.driver = {argv} (in {top})")


def normalize(path: Path) -> int:
    """Recompute each `reads` digest of the record at ``path`` from its definitions, and drop the readers an older record
    lists (`load`); the number of groups that changed."""
    before = A.as_guarantees(json.loads(path.read_text())).get("reads", {})
    doc = load(path)
    for g in doc.get("reads", {}).values():
        g["digest"] = A.module_digest(A.definitions(g))
    path.write_text(A.text(doc))
    return sum(before.get(m) != g for m, g in doc.get("reads", {}).items())


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="merge.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*", type=Path, help="BASE OURS THEIRS [PATH], as git passes them; with --normalize, records")
    ap.add_argument("--install", nargs="?", const=Path.cwd(), type=Path, metavar="REPO", help="register the driver in REPO")
    ap.add_argument("--normalize", action="store_true", help="recompute the `reads` digests of each record named, and drop "
                    "the readers an older one lists")
    ap.add_argument("--lean", action="store_true", help="BASE OURS THEIRS [PATH] are a `.lean` file's: merge its import lists")
    ap.add_argument("--lakefile", action="store_true", help="BASE OURS THEIRS [PATH] are a `lakefile.toml`'s: merge its "
                    "`[[require]]` blocks and string arrays")
    ap.add_argument("--caps", action="store_true", help="BASE OURS THEIRS [PATH] are `tests/test_lean_packages.py`'s: merge "
                    "its ratchet caps (`UNWITNESSED`, `INLINE`) key by key as the larger of the two sides'")
    ap.add_argument("--pyproject", action="store_true", help="BASE OURS THEIRS [PATH] are a `pyproject.toml`'s: merge its "
                    "`[tool.verity.tests]` lists")
    a = ap.parse_args(argv)
    if a.install is not None:
        install(a.install)
        return 0
    if a.normalize:
        for p in a.files:
            print(f"merge.py: {p}: {normalize(p)} `reads` groups rewritten")
        return 0
    if len(a.files) not in (3, 4):
        ap.error("name BASE OURS THEIRS [PATH], or --install, or --normalize FILE...")
    base, ours, theirs = a.files[:3]
    path = a.files[3] if len(a.files) == 4 else ours
    if why := stale():
        return conflict(base, ours, theirs, path, why)
    if a.lean or a.lakefile or a.caps or a.pyproject:
        text, left = (lakefile if a.lakefile else caps if a.caps else pyproject if a.pyproject else lean)(base, ours, theirs)
        ours.write_text(text)
        if left:
            what = ("requires and string arrays" if a.lakefile else "the ratchet caps" if a.caps else
                    "[tool.verity.tests] lists" if a.pyproject else "import lists")
            print(f"merge.py: {path}: {left} conflict(s) beyond {what}", file=sys.stderr)
        return 1 if left else 0
    docs, bad = {}, {}
    for name, p in (("base", base), ("ours", ours), ("theirs", theirs)):
        try:
            docs[name] = load(p)
        except (ValueError, AttributeError, TypeError) as e:
            bad[name] = f"{name} is not a record ({e})"
    if "ours" in bad and "theirs" in bad:
        return conflict(base, ours, theirs, path, f"neither side is a record ({bad['ours']}; {bad['theirs']})")
    if newer := [f"{n} is lock format {f!r}" for n, d in docs.items()
                 if not (isinstance(f := d.get("format", 0), int) and not isinstance(f, bool) and f <= FORMAT)]:
        return conflict(base, ours, theirs, path, f"{'; '.join(newer)}, which this merge.py (format {FORMAT}) doesn't know: "
                        f"merge main's {DIR} first, or merge with that of the branch that made it, then merge again")
    why: list[str] = []
    candidate: list[str] = []
    if "ours" in bad or "theirs" in bad:
        took = "theirs" if "ours" in bad else "ours"
        merged = docs[took]
        why.append(f"{bad['ours' if took == 'theirs' else 'theirs']}: took {took}")
    else:
        if "base" in bad:
            why.append(f"{bad['base']}: merged as if the base were empty")
        try:
            merged, clashes = merge(docs.get("base", {}), docs["ours"], docs["theirs"], candidate)
            why += clashes
        except Exception as e:  # noqa: BLE001  (a merge that fails doesn't refuse: theirs stands as the candidate)
            merged, candidate = docs["theirs"], []
            why.append(f"the merge failed ({type(e).__name__}: {e}): took theirs")
        if lost := losses(docs.get("base", {}), docs["ours"], docs["theirs"], merged):
            return conflict(base, ours, theirs, path, "the merged record loses most of a section, which a conflict shows "
                            "where a merge would drop it silently: " + "; ".join(lost))
    for w in why:
        print(f"merge.py: {path}: {w}", file=sys.stderr)
    for c in dict.fromkeys(candidate):
        print(f"merge.py: {path}: {c}: " + (
            "a shape.inline premise one side dropped and the other, restating its guarantee, has as the base had it, kept; "
            "`check.py --update` drops it where the merged tree's premise no longer offends" if c.endswith(" (kept)") else
            "kept as named; the Lean check of the merged tree judges it, and a stale entry fails there as a witness failure"
            if c.split("[", 1)[0] in JUDGED else
            "taken without regenerating it; the lock run after landing regenerates it"), file=sys.stderr)
    ours.write_text(A.text(merged))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
