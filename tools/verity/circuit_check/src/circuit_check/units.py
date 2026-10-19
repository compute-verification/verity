"""The catalog's bindings, each reused while what its build read is unchanged: with ``--catalog-units``, a ``--all`` run whose
task list misses (`cli.run`) builds the bodies of only the bindings whose record misses, and takes what every other binding's
roots reach from its record.

Every binding still runs (`targets.catalog_units`, in catalog order, each under its units, `targets._produce`): its roots are
made now, unbuilt, so a registry a binding enumerates is read as it is now.  A binding's record (a `circuit_check.cache` entry of
the request ``catalog-unit``, keyed by the binding's units and which of that binding's occurrences it is) holds the ids of its
roots and what they reach (`_reach`): each primitive, specialization and reference by id, with its kind, family, gate count,
whether it is Boolean, the ids its body calls (in the order `targets.reachable` visits them) and, where `targets.select` could
keep it, its descriptor (`cache.descriptor`), name and descriptor id.  Its reads (`cache.unit_reads`) are what its units read
and ran while they made its roots, the build of every Definition they reach, the bindings that made those, those modules' import
closures, its units by digest and what the process read outside every recording; and the units of every binding that makes no
root (one run for what it sets, as `syn.knob`'s), whose reads every record holds.  The registered primitives and static-free
composites are one more record, ``<registered>``.

A record counts only when its every read is unchanged where it was (never through a move) and the roots its binding made now
are the ones it lists; else its binding is built as the catalog builds it and its record stored again.  The task list is then
what `cli._tasks` would build: `targets.select` over what the roots reach in `targets.reachable`'s order, PoUW's roots whole,
each target's units (`targets.BINDINGS`), whether it is a Call (`targets.call_families`: the word Call families, and the
Boolean version of each suite Definition of one, made from its binding where no built binding holds it, `_materialize`) and
its key.  A target that misses is made from a binding that reaches it, along the path its record lists, and its descriptor
computed again: one that differs from its record's, two records that differ on one Definition, or a registry that grew while
the bindings ran, is `Stale`, and the run builds the whole catalog.  A full build stores every binding's record only when
the task list derived from those records is the one it built (`recorded`).

Not covered, beyond what `circuit_check.cache` leaves out: a body whose structure depends on module state a binding that makes
roots sets (a binding's own state is in its reads; one that makes no root is held by every record), when no other record
reaches the Definitions that state shapes.  The nightly uncached check is what disagrees then.
"""
from __future__ import annotations

import json
from collections import Counter, deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import cache as CC
from . import targets as TG

REQUEST = "catalog-unit"
REGISTERED = ("<registered>",)


class Stale(Exception):
    """A record disagrees with what this run built: the run builds the whole catalog instead."""


@dataclass
class Entry:
    """One binding as this run has it: its units, kind and label (`targets.catalog_units`), its key, the roots it made now,
    what they reach (its record's, or `_reach` of the bodies built now) and whether its record counted."""
    units: tuple[str, ...]
    kind: str
    label: str
    key: dict
    roots: list[Any]
    nodes: dict[str, dict] = field(default_factory=dict)
    reused: bool = False

    @property
    def root_ids(self) -> list[str]:
        return [fn.id for fn in self.roots if _kind(fn) is not None]

    @property
    def name(self) -> str:
        return ",".join(self.units)


def _kind(fn: Any) -> str | None:
    from verity.ml.circuits.defs import PrimitiveDefinition, SpecializedDefinition
    from verity.ml.circuits.evaluation import ScalarReference

    return "P" if isinstance(fn, PrimitiveDefinition) else "S" if isinstance(fn, SpecializedDefinition) else \
        "R" if isinstance(fn, ScalarReference) else None


def key(units: tuple[str, ...], n: int) -> dict:
    return CC.key({"target": [REQUEST, list(units), n], "as_call": None, "options": {}})


def _calls(fn: Any) -> list[str]:
    """The ids ``fn``'s body calls, each at its last occurrence: `targets.reachable` pushes them in order and pops from the end,
    so an earlier repeat is never what it visits."""
    ids = [n.fn.id for n in fn.body.nodes if _kind(n.fn) is not None]
    seen: set[str] = set()
    out = []
    for i in reversed(ids):
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out[::-1]


def _reach(roots: list[Any]) -> tuple[dict[str, dict], dict[str, Any]]:
    """What ``roots`` reach (`targets.reachable`, building every body), each as a record holds it, and the objects."""
    from verity.ml.circuits import boolean as BL
    from verity.ml.circuits import codec as C

    objs = TG.reachable(roots)
    nodes: dict[str, dict] = {}
    for fid, fn in objs.items():
        k = _kind(fn)
        info = {"k": k, "f": TG.family(fn), "g": getattr(fn, "gates", 1), "b": bool(BL.is_boolean(fn))}
        if k == "S":
            info["c"] = _calls(fn)
        nodes[fid] = info
    least: dict[str, int] = {}
    bits: dict[str, int] = {}
    for info in nodes.values():
        least[info["f"]] = min(least.get(info["f"], info["g"]), info["g"])
        if info["b"]:
            bits[info["f"]] = min(bits.get(info["f"], info["g"]), info["g"])
    tops = {fn.id for fn in roots if _kind(fn) is not None}
    for fid, info in nodes.items():
        g, f = info["g"], info["f"]
        if fid in tops or g <= TG.MAX_GATES or g == least[f] or (info["b"] and g == bits.get(f)):
            fn = objs[fid]
            info["d"] = CC.descriptor(fn)
            if info["k"] != "R":
                info["n"] = getattr(fn, "name", None)
                if (di := C.descriptor_id(fn)) != fid:
                    info["di"] = di
    return nodes, objs


def _merged(entries: list[Entry]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for e in entries:
        for fid, info in e.nodes.items():
            have = out.get(fid)
            if have is None:
                out[fid] = dict(info)
                continue
            for k in have.keys() & info.keys():
                if have[k] != info[k]:
                    raise Stale(f"{fid}: two bindings' records differ on its {k} ({e.name}'s and an earlier one's)")
            for k in info.keys() - have.keys():
                have[k] = info[k]
    return out


def _select(reached: dict[str, dict]) -> set[str]:
    """`targets.select` over records."""
    by: dict[str, list[tuple[str, dict]]] = {}
    for fid, info in reached.items():
        by.setdefault(info["f"], []).append((fid, info))
    out: set[str] = set()
    for fns in by.values():
        keep = [f for f in fns if f[1]["g"] <= TG.MAX_GATES]
        if not keep:
            keep = [min(fns, key=lambda f: f[1]["g"])]
            if not keep[0][1]["b"] and (bits := [f for f in fns if f[1]["b"]]):
                keep.append(min(bits, key=lambda f: f[1]["g"]))
        out.update(fid for fid, _ in keep)
    return out


def _closure(ids: list[str], nodes: dict[str, dict]) -> set[str]:
    out: set[str] = set()
    stack = list(ids)
    while stack:
        fid = stack.pop()
        if fid in out:
            continue
        out.add(fid)
        stack.extend(nodes[fid].get("c", ()))
    return out


@dataclass
class Derived:
    """The suite as the records give it: its ids in order, each one's units and the merged records."""
    ids: list[str]
    units: dict[str, set[str]]
    nodes: dict[str, dict]


def derive(entries: list[Entry]) -> Derived:
    """`targets.suite` and its `targets.BINDINGS` over records: what the catalog's roots (each entry's, in catalog order)
    reach, in `targets.reachable`'s order, `select`'s, PoUW's roots whole."""
    nodes = _merged(entries)
    stack = [fid for e in entries for fid in e.root_ids]
    reached: dict[str, dict] = {}
    while stack:
        fid = stack.pop()
        if fid in reached:
            continue
        if fid not in nodes:
            raise Stale(f"{fid}: no record holds what it reaches")
        reached[fid] = nodes[fid]
        stack.extend(nodes[fid].get("c", ()))
    out = _select(reached)
    for e in entries:
        if e.kind == "_pouw_roots":
            out.update(e.root_ids)
    units: dict[str, set[str]] = {}
    for e in entries:
        if e.units == REGISTERED:
            continue
        for fid in _closure(e.root_ids, nodes) & out:
            units.setdefault(fid, set()).update(e.units)
    for fid in out:
        if "d" not in nodes[fid]:
            raise Stale(f"{fid}: its record has no descriptor")
    return Derived(sorted(out), units, nodes)


def _path(e: Entry, fid: str) -> list[str] | None:
    """Ids from one of ``e``'s roots to ``fid`` through the bodies its record lists."""
    parent: dict[str, str | None] = {r: None for r in e.root_ids}
    todo = deque(e.root_ids)
    while todo:
        cur = todo.popleft()
        if cur == fid:
            path = [cur]
            while (p := parent[path[-1]]) is not None:
                path.append(p)
            return path[::-1]
        for nxt in e.nodes.get(cur, {}).get("c", ()):
            if nxt not in parent:
                parent[nxt] = cur
                todo.append(nxt)
    return None


def _materialize(entries: list[Entry], objects: dict[str, Any], fid: str, in_unit: bool = True) -> Any:
    """The object of ``fid``: one built now, else made from the first binding whose record reaches it, building each body on
    the path its record lists (a model's under its units, as the catalog builds a model's body)."""
    if fid in objects:
        return objects[fid]
    for e in entries:
        if fid not in e.nodes or (path := _path(e, fid)) is None:
            continue
        fn = next(r for r in e.roots if r.id == path[0])
        with CC.producing(e.units) if in_unit and e.kind == "model" else nullcontext():
            for nxt in path[1:]:
                fn = next((n.fn for n in fn.body.nodes if n.fn.id == nxt), None)
                if fn is None:
                    raise Stale(f"{fid}: {e.name}'s record lists a call to {nxt} that its body no longer makes")
        objects[fid] = fn
        return fn
    raise Stale(f"{fid}: no binding's record reaches it")


def call_families(entries: list[Entry], d: Derived, objects: dict[str, Any], in_unit: bool = True) -> set[str]:
    """The families of `targets.call_families` the suite's Definitions are of: the word Call families (the replay's rows, the
    vocabulary kinds', the templates' Definitions) and of those Boolean versions of a suite Definition of one
    (`targets.boolean_family`) that are of a suite family: each from its object, made (`_materialize`) only where it could be
    of a suite family no other one gave (`targets.boolean_links`)."""
    vocab = next((e for e in entries if e.kind == "_vocabulary_roots"), None)
    words = TG.row_families() | {TG.family(fn) for fn in (vocab.roots if vocab else ())}
    words |= {TG.family(fn) for e in entries if e.kind == "template" for fn in e.roots[0::2]}
    wordy = [fid for fid in d.ids if (n := d.nodes[fid])["f"] in words and n["k"] != "R" and not n["b"]]
    found = {f for fid in wordy if fid in objects and (f := TG.boolean_family(objects[fid])) is not None}
    families = {d.nodes[fid]["f"] for fid in d.ids}
    open_ = (families - words - found) & TG.boolean_call_candidates(words)
    if open_:
        exact, by_name = TG.boolean_links()
        keys = {k for k, fams in exact.items() if fams & open_}
        names = {k for k, fams in by_name.items() if fams & open_}
        for fid in wordy:
            n = d.nodes[fid]
            if fid not in objects and (n.get("di", fid) in keys or n.get("n") in names):
                if (f := TG.boolean_family(_materialize(entries, objects, fid, in_unit))) is not None:
                    found.add(f)
    return words | found


def _sited(v: Any) -> Any:
    """A module's or a unit's digest with each import site by the name it resolves to: the content beside it is what follows a
    move (`cache._moved`), and a site's module is a read of its own where the binding's build ran or imported it."""
    return {**v, "sites": [s[0] for s in v["sites"]]} if isinstance(v, dict) and "sites" in v else v


def _unchanged(was: dict, now: dict) -> bool:
    """Every read as recorded, where it was: a module or unit by its content and where its import sites resolve (`_sited`), a
    package an enumerator listed by the modules it lists."""
    for kind, v in was.items():
        if kind == "enums":
            if any((a is None) != (b is None) or (a is not None and a.keys() != b.keys())
                   for a, b in ((v[p], now["enums"].get(p)) for p in v)):
                return False
        elif kind in ("modules", "units"):
            if v.keys() != now[kind].keys() or any(_sited(x) != _sited(now[kind][r]) for r, x in v.items()):
                return False
        elif now.get(kind) != v:
            return False
    return True


def _lookup(cache: Path, repo: Path, e: Entry) -> dict | None:
    """``e``'s record when every read it holds is unchanged (`_unchanged`) and it lists the roots made now; None otherwise, and
    for one that cannot be read."""
    with CC.internal():
        found = sorted(CC._dir(cache, e.key).glob("*.json"))
    for p in found:
        try:
            with CC.internal():
                rec = json.loads(p.read_text())
            if rec.get("key") != e.key:
                continue
            reads = rec["reads"]
            if not _unchanged(reads, CC._digests(repo, {k: list(v) for k, v in reads.items()})):
                continue
            got = rec["report"]
            if got.get("roots") != e.root_ids or not isinstance(got.get("nodes"), dict):
                continue
            for fid in e.root_ids:
                got["nodes"][fid]["k"]
            return got
        except (OSError, KeyError, TypeError, AttributeError, ValueError):
            continue
    return None


@dataclass
class Plan:
    """A run's catalog from records: each binding, the objects built now, the derived suite and whether each is a Call."""
    entries: list[Entry]
    objects: dict[str, Any]
    derived: Derived
    calls: set[str]

    def stats(self) -> dict:
        return {"reused": sum(e.reused for e in self.entries),
                "rebuilt": [e.name for e in self.entries if not e.reused]}


def plan(cache: Path, repo: Path, log: Callable[[str], None] | None = None) -> Plan | None:
    """The catalog from the bindings' records (the module's docstring); None when there are none of this checker's version (a
    full build stores them) or none counts.  `Stale` when a record disagrees with what was built now."""
    TG.load_registries()
    first = key(REGISTERED, 0)
    if not CC._dir(cache, first).is_dir():
        return None
    registered = TG._registered()
    ids0 = [fn.id for fn in registered]
    entries = [Entry(REGISTERED, "registered", "", first, registered)]
    TG.ROOT_ERRORS.clear()
    seen: Counter = Counter()
    for units, make, kind, label in TG.catalog_units():
        n, seen[units] = seen[units], seen[units] + 1
        try:
            roots = TG._produce(units, make)
        except Exception as e:  # noqa: BLE001  (a model's error is named, as the catalog names it)
            if kind != "model":
                raise
            TG.ROOT_ERRORS.append((label, f"{type(e).__name__}: {str(e)[:200]}"))
            continue
        entry = Entry(units, kind, label, key(units, n), roots)
        rec = _lookup(cache, repo, entry)
        if rec is not None:
            entry.nodes, entry.reused = rec["nodes"], True
        elif kind == "model":
            try:
                with CC.producing(units):
                    _ = roots[0].body
            except Exception as e:  # noqa: BLE001  (named, as the catalog names it)
                TG.ROOT_ERRORS.append((label, f"{type(e).__name__}: {str(e)[:200]}"))
                continue
        entries.append(entry)
    rec = _lookup(cache, repo, entries[0])
    if rec is not None:
        entries[0].nodes, entries[0].reused = rec["nodes"], True
    if not any(e.reused for e in entries):
        return None
    objects: dict[str, Any] = {}
    for e in entries:
        for fn in e.roots:
            if _kind(fn) is not None:
                objects.setdefault(fn.id, fn)
        if not e.reused:
            e.nodes, objs = _reach(e.roots)
            for fid, fn in objs.items():
                objects.setdefault(fid, fn)
    d = derive(entries)
    calls = call_families(entries, d, objects)
    if [fn.id for fn in TG._registered()] != ids0:
        raise Stale("the registry grew while the bindings ran")
    if log:
        log(f"circuit-check: catalog from records: {sum(e.reused for e in entries)} of {len(entries)} bindings reused, "
            f"{len(objects)} Definitions built")
    return Plan(entries, objects, d, calls)


def store(cache: Path, repo: Path, entries: list[Entry], only_rebuilt: bool) -> int:
    """Store each binding's record (each rebuilt one's, with ``only_rebuilt``), its reads digested now; the count stored."""
    side = [u for e in entries if e.units != REGISTERED and not e.root_ids for u in e.units]
    n = 0
    for e in entries:
        if only_rebuilt and e.reused:
            continue
        reads = CC.unit_reads((*e.units, *side) if e.units != REGISTERED else side, e.roots)
        rec = {"roots": e.root_ids, "nodes": e.nodes}
        if CC.store(cache, e.key, repo, reads, rec, CC.seen(repo, reads)) is not None:
            n += 1
    return n


def built_entries() -> list[Entry] | None:
    """The bindings of this process's first full catalog (`targets.CATALOGS`), with what they reach; None when the registry
    grew between its catalogs (a record would not say what a later run's catalog starts from)."""
    if not TG.CATALOGS:
        return None
    registered, made = TG.CATALOGS[0]
    if any([fn.id for fn in c[0]] != [fn.id for fn in registered] for c in TG.CATALOGS[1:]):
        return None
    entries = [Entry(REGISTERED, "registered", "", key(REGISTERED, 0), registered)]
    seen: Counter = Counter()
    for units, kind, label, roots in made:
        n, seen[units] = seen[units], seen[units] + 1
        entries.append(Entry(units, kind, label, key(units, n), roots))
    for e in entries:
        e.nodes, _ = _reach(e.roots)
    return entries


__all__ = ["REQUEST", "Derived", "Entry", "Plan", "Stale", "built_entries", "call_families", "derive", "key", "plan", "store"]
