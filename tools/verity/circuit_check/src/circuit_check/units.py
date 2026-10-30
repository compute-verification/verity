"""The catalog's bindings, each reused while what its build read is unchanged: with ``--catalog-units``, a ``--all`` run whose
task list misses (`cli.run`) builds the bodies of only the bindings whose record misses, and takes what every other binding's
roots reach from its record.

Every binding still runs (`targets.catalog_units`, in catalog order, each under its units, `targets._produce`): its roots are
made now, unbuilt, so a registry a binding enumerates is read as it is now.  A binding's record (a `circuit_check.cache` entry of
the request ``catalog-unit``, keyed by the binding's units and which of that binding's occurrences it is) holds the ids of its
roots and what they reach (`_reach`): each primitive, specialization and reference by id, with its kind, family, gate count,
whether it is Boolean, the ids its body calls (in the order `targets.reachable` visits them) and, where `targets.select` could
keep it, its descriptor (`cache.descriptor`), name and descriptor id; a by-calls root (`targets.is_by_calls`) as ``bc``, its
body unbuilt, so with no gates and no calls.  Its reads (`cache.unit_reads`) are what its units read
and ran while they made its roots, the build of every Definition they reach, the bindings that made those, those modules' import
closures, its units by digest and what the process read outside every recording; and the units of every binding that makes no
root (one run for what it sets, as `syn.knob`'s), whose reads every record holds.  The registered primitives and static-free
composites are `REGISTERED_RECORDS` more, ``<registered>:i``, each of a run of them in their order (`registered_entries`), so
that their builds run apart from each other.

A record counts only when its every read is unchanged where it was (never through a move) and the roots its binding made now
are the ones it lists; else its binding is built as the catalog builds it and its record stored again.  The task list is then
what `cli._tasks` would build: `targets.select` over what the roots reach in `targets.reachable`'s order, PoUW's roots whole,
each target's units (`targets.BINDINGS`), whether it is a Call (`targets.call_families`: the word Call families, and the
Boolean version of each suite Definition of one, made from its binding where no built binding holds it, `_materialize`) and
its key.  Two bindings can reach two Definitions of one id (`AttnDotStep_v1` with its DOT bound, from `boolean_version`, and
unbound, from its sidecar): `targets.reachable` keeps the one it pops first, and so each id is taken from the record of the
binding it is first reached through (`derive`).  A target that misses is made from that binding, along the path its record
lists, and its descriptor computed again: one that differs from its record's is `Stale`, and the run builds the whole
catalog (in a run that forks its checks, where it is checked: below).  What a body build registers (a constant's primitive, a
lifted Definition) is never a root: the catalog's registered roots are taken before any binding runs, here and in
`targets.catalog`, and `cli._tasks` reads the Calls off that one suite.  A full build stores every binding's record only when
the task list derived from those records is the one it built (`cli._record_units`).

Not covered, beyond what `circuit_check.cache` leaves out: a body whose structure depends on module state a binding that makes
roots sets (a binding's own state is in its reads; one that makes no root is held by every record), when no other record
reaches the Definitions that state shapes.  The nightly uncached check is what disagrees then.

The circuit_check suite builds the catalog from these same entries without a cache (`made_entries`, `built_apart`): every
binding's roots made in one process, then each binding's bodies built in a process forked for it alone, which returns what
they reach (`_reach`) with a mark on each Definition whose id makes the very object reached (`_by_id`).  Its xdist workers
share that as JSON (`manifest`, `from_manifest`), and each makes the suite from it (`planned`) and each target from its id
where it can, building no body (`make`).

A ``--catalog-units`` run that forks its targets' processes (`cli.run` with ``--jobs``) builds its bindings so too: each
binding whose record doesn't count, and every one when none does or there are none, is built in a process forked for it alone
(``apart`` in `plan`), which returns what its roots reach with each one's descriptor, what its build read (its record's reads,
`cache.unit_reads`) and its share of the task list's reads (`cache.task_list_reads` of its roots).  This process builds no
body.  A target that misses, through such a binding or one whose record counted, is made where it is checked (`cli._made`,
`make`): by the zygote that forks the checks of that binding's targets, each under a recording its check then holds
(`cli._prepare`), so the bodies they share are built once, as a full build builds them, and no target is built or described
here one after another; the target is held to the descriptor its binding's build gave or its record holds, and one that
differs is its check's error.  A primitive its id makes, which has no body, is made here (`built_here`), and so is each
Boolean Definition a binding's build registered (`made_words`, or its module imported): C-Flock lowers a word primitive by the
first one registered of that word, so a check forked from here finds the one a full build's would.  A binding whose process
dies fails the run, as it would kill a build in one process.  When every binding was built apart, the task list's entry is
stored as a full build stores it, its reads every binding's share and this process's own; it and each record are digested
where this process stores them, as a full build's are (a full build digests none of its reads where it builds them either).
``--records-only`` (`cli.records`) builds and stores the records so and checks no target, so that each run from a copy of
its cache (each shard of one run, on a machine of its own) builds no binding.

A run from the records, forked or not, has no process that imported every module a full build has imported before its
checks, and `sidecars.owner` reads those: a Definition a factory makes is owned by the module that binds it at its top level,
which only another binding's body may import, and the block check (`checks.UNIT_OWNERS`) and a by-calls root's callees go by
that owner.  So each Definition's owner is handed down as a full build has it (`owners`, `sidecars.HANDED`).
"""
from __future__ import annotations

import json
import os
import sys
import types
from collections import Counter, deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from . import cache as CC
from . import targets as TG

REQUEST = "catalog-unit"
REGISTERED = ("<registered>",)
#: how many records the registered roots take (`registered_entries`): the longest of their builds bounds a ``--records-only``
#: run, and one of them all took as long as the longest binding's
REGISTERED_RECORDS = 8


class Stale(Exception):
    """A record disagrees with what this run built: the run builds the whole catalog instead."""


@dataclass
class Entry:
    """One binding as this run has it: its units, kind and label (`targets.catalog_units`), its key, the roots it made now,
    what they reach (its record's, or `_reach` of the bodies built now) and whether its record counted, with that record's
    reads.  Built in a process of its own (``apart``, `built_apart`), with what that process read: its record's reads and its
    share of the task list's (``listed``), and the CPU seconds it took (``seconds``, which its record keeps).  ``words``: the
    Boolean Definitions its build registered (`_words`), where a process built it apart."""
    units: tuple[str, ...]
    kind: str
    label: str
    key: dict
    roots: list[Any]
    nodes: dict[str, dict] = field(default_factory=dict)
    reused: bool = False
    apart: bool = False
    reads: dict[str, list[str]] | None = None
    listed: dict[str, list[str]] | None = None
    seconds: float | None = None
    words: list[list[str | None]] | None = None

    @property
    def root_ids(self) -> list[str]:
        return [fn.id for fn in self.roots if _kind(fn) is not None]

    @property
    def name(self) -> str:
        return ",".join(self.units) + (f":{self.label}" if self.units == REGISTERED else "")


def registered_entries(roots: list[Any]) -> list[Entry]:
    """The registered roots (`targets._registered`) as at most `REGISTERED_RECORDS` entries, each of a run of them in their order
    (so `derive` meets them in the order `targets.reachable` does), entry i keyed and labelled by i; one entry when there are
    none."""
    k = max(1, -(-len(roots) // REGISTERED_RECORDS))
    return [Entry(REGISTERED, "registered", str(i), key(REGISTERED, i), roots[j:j + k])
            for i, j in enumerate(range(0, max(1, len(roots)), k))]


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


def _described(fn: Any, k: str) -> dict:
    """What a record holds of ``fn`` (of kind ``k``) beyond what it reaches: its descriptor and, unless it is a reference, its
    name and its descriptor id where that is not its id."""
    from verity.ml.circuits import codec as C

    out = {"d": CC.descriptor(fn)}
    if k != "R":
        out["n"] = getattr(fn, "name", None)
        if (di := C.descriptor_id(fn)) != fn.id:
            out["di"] = di
    return out


def _reach(roots: list[Any], described: bool = True, pinned: frozenset[str] | None = None) \
        -> tuple[dict[str, dict], dict[str, Any]]:
    """What ``roots`` reach (`targets.reachable`, building every body), each as a record holds it, and the objects; without
    ``described``, no descriptor, name or descriptor id (a sixth of the build, which only a record checks against).  Described
    (`_described`): each one `targets.select` could keep, the ``pinned`` ids (default `targets.pinned`) among them."""
    from verity.ml.circuits import boolean as BL

    objs = TG.reachable(roots)
    nodes: dict[str, dict] = {}
    for fid, fn in objs.items():
        k = _kind(fn)
        if TG.is_by_calls(fn):
            nodes[fid] = {"k": k, "f": TG.family(fn), "g": 0, "b": False, "bc": 1}
            continue
        info = {"k": k, "f": TG.family(fn), "g": getattr(fn, "gates", 1), "b": bool(BL.is_boolean(fn))}
        if k == "S":
            info["c"] = _calls(fn)
        nodes[fid] = info
    if not described:
        return nodes, objs
    pinned = TG.pinned() if pinned is None else pinned
    least: dict[str, int] = {}
    bits: dict[str, int] = {}
    for info in nodes.values():
        if info.get("bc"):
            continue
        least[info["f"]] = min(least.get(info["f"], info["g"]), info["g"])
        if info["b"]:
            bits[info["f"]] = min(bits.get(info["f"], info["g"]), info["g"])
    tops = {fn.id for fn in roots if _kind(fn) is not None}
    for fid, info in nodes.items():
        g, f = info["g"], info["f"]
        if fid in tops or fid in pinned or g <= TG.MAX_GATES or g == least[f] or (info["b"] and g == bits.get(f)):
            info.update(_described(objs[fid], info["k"]))
    return nodes, objs


def _select(reached: dict[str, dict], pinned: frozenset[str] = frozenset()) -> set[str]:
    """`targets.select` over records."""
    by: dict[str, list[tuple[str, dict]]] = {}
    out: set[str] = set()
    for fid, info in reached.items():
        if info.get("bc"):
            out.add(fid)
        else:
            by.setdefault(info["f"], []).append((fid, info))
    for fns in by.values():
        keep = [f for f in fns if f[1]["g"] <= TG.MAX_GATES]
        if not keep:
            keep = [min(fns, key=lambda f: f[1]["g"])]
            if not keep[0][1]["b"] and (bits := [f for f in fns if f[1]["b"]]):
                keep.append(min(bits, key=lambda f: f[1]["g"]))
        out.update(fid for fid, _ in keep)
        out.update(fid for fid, _ in fns if fid in pinned)
    return out


@dataclass
class Derived:
    """The suite as the records give it: its ids in order, each one's units, and of each id reached its record and the binding
    it was taken from (``via``)."""
    ids: list[str]
    units: dict[str, set[str]]
    nodes: dict[str, dict]
    via: dict[str, Entry] = field(default_factory=dict)


def derive(entries: list[Entry], described: bool = True, pinned: frozenset[str] | None = None) -> Derived:
    """`targets.suite` and its `targets.BINDINGS` over records: what the catalog's roots (each entry's, in catalog order)
    reach, in `targets.reachable`'s order, `select`'s (the ``pinned`` ids kept, default `targets.pinned`), PoUW's roots whole.
    Each id is the record's of the binding it is first reached through, as `reachable` keeps the object it pops first, and each
    binding's units go to what its own record reaches, as `targets.suite` gives each binding what its own roots reach.
    ``described``: each suite id's record holds its descriptor, as a stored one does, else `Stale`; but a pinned id's record
    stored before its pin holds none (`targets.select` dropped it then), and `plan` describes it."""
    pinned = TG.pinned() if pinned is None else pinned
    stack = [(fid, e) for e in entries for fid in e.root_ids]
    nodes: dict[str, dict] = {}
    via: dict[str, Entry] = {}
    while stack:
        fid, e = stack.pop()
        if fid in nodes:
            continue
        if fid not in e.nodes:
            raise Stale(f"{fid}: {e.name}'s record doesn't hold what its roots reach")
        nodes[fid], via[fid] = e.nodes[fid], e
        stack.extend((c, e) for c in e.nodes[fid].get("c", ()))
    out = _select(nodes, pinned)
    for e in entries:
        if e.kind == "_pouw_roots":
            out.update(e.root_ids)
    units: dict[str, set[str]] = {}
    for e in entries:
        if e.units == REGISTERED:
            continue
        for fid in e.nodes.keys() & out:
            units.setdefault(fid, set()).update(e.units)
    for fid in out if described else ():
        if "d" not in nodes[fid] and fid not in pinned:
            raise Stale(f"{fid}: its record has no descriptor")
    return Derived(sorted(out), units, nodes, via)


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


def _materialize(entries: list[Entry], objects: dict[str, Any], fid: str, in_unit: bool = True, via: Entry | None = None) -> Any:
    """The object of ``fid``: one built now, else made from ``via`` (the binding `derive` took it from) or else the first binding
    whose record reaches it, building each body on the path its record lists (a model's under its units, as the catalog builds
    a model's body)."""
    if fid in objects:
        return objects[fid]
    for e in ([via] if via is not None else []) + [e for e in entries if e is not via]:
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
    """The families of `targets.call_families` the suite's Definitions are of: the word Call families (`targets.
    word_call_families`, which refuses a vocabulary kind that doesn't bind), the Boolean version of each one made on demand
    (`targets.on_demand_boolean_families`) and of those Boolean versions of a suite Definition of one (`targets.boolean_family`)
    that are of a suite family: each from its object, made (`_materialize`) only where it could be of a suite family no other
    one gave (`targets.boolean_links`)."""
    words = TG.word_call_families()
    wordy = [fid for fid in d.ids if (n := d.nodes[fid])["f"] in words and n["k"] != "R" and not n["b"]]
    found = TG.on_demand_boolean_families(words)
    found |= {f for fid in wordy if fid in objects and (f := TG.boolean_family(objects[fid])) is not None}
    families = {d.nodes[fid]["f"] for fid in d.ids}
    open_ = (families - words - found) & TG.boolean_call_candidates(words)
    if open_:
        exact, by_name = TG.boolean_links()
        keys = {k for k, fams in exact.items() if fams & open_}
        names = {k for k, fams in by_name.items() if fams & open_}
        for fid in wordy:
            n = d.nodes[fid]
            if fid not in objects and (n.get("di", fid) in keys or n.get("n") in names):
                if (f := TG.boolean_family(_materialize(entries, objects, fid, in_unit, d.via.get(fid)))) is not None:
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


def _lookup(cache: Path, repo: Path, e: Entry) -> tuple[dict, dict[str, list[str]]] | None:
    """``e``'s record and its reads when every read it holds is unchanged (`_unchanged`) and it lists the roots made now; None
    otherwise, and for one that cannot be read."""
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
            return got, reads
        except (OSError, KeyError, TypeError, AttributeError, ValueError):
            continue
    return None


@dataclass
class Plan:
    """A run's catalog from records: each binding, the objects built now, the derived suite and whether each is a Call; when
    every binding was built apart, what the task list's entry holds beside this process's reads (``listed``); and the most
    bytes the results of those built apart held on disk at once (`built_apart`)."""
    entries: list[Entry]
    objects: dict[str, Any]
    derived: Derived
    calls: set[str]
    listed: dict[str, list[str]] | None = None
    scratch: int = 0

    def stats(self) -> dict:
        return {"reused": sum(e.reused for e in self.entries),
                "rebuilt": [e.name for e in self.entries if not e.reused]}


def merged(parts: Iterable[dict[str, list[str]]]) -> dict[str, list[str]]:
    """Read sets (`cache.attributed`'s) as one: each kind's reads, every part's."""
    out: dict[str, set[str]] = {}
    for p in parts:
        for k, v in p.items():
            out.setdefault(k, set()).update(v)
    return {k: sorted(v) for k, v in sorted(out.items())}


def plan(cache: Path | None, repo: Path, log: Callable[[str], None] | None = None,
         apart: tuple[int, float, dict[str, float]] | None = None) -> Plan | None:
    """The catalog from the bindings' records (the module's docstring); None when there are none of this checker's version (a
    full build stores them) or none counts.  With ``apart`` (`built_apart`'s ``jobs``, ``budget`` and ``weights``), each binding
    whose record doesn't count is built in a process of its own, and so is every binding when none counts, there are none or
    there is no ``cache``: this process builds no body.  `Stale` when a record disagrees with what was built now.  A pinned id
    a reused record holds no descriptor for (stored before its pin, `derive`) is made from the binding it is taken from
    (`_materialize`) and described in this run; that record is not stored again."""
    TG.load_registries()
    first = key(REGISTERED, 0)
    stored = cache is not None and CC._dir(cache, first).is_dir()
    if not stored and apart is None:
        return None
    registered = registered_entries(TG._registered())
    entries = list(registered)
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
        hit = _lookup(cache, repo, entry) if stored else None
        if hit is not None:
            rec, entry.reads = hit
            entry.nodes, entry.reused, entry.words, entry.seconds = rec["nodes"], True, rec.get("words"), rec.get("seconds")
        elif kind == "model" and apart is None:
            try:
                with CC.producing(units):
                    _ = roots[0].body
            except Exception as e:  # noqa: BLE001  (named, as the catalog names it)
                TG.ROOT_ERRORS.append((label, f"{type(e).__name__}: {str(e)[:200]}"))
                continue
        entries.append(entry)
    for e in registered:
        if (hit := _lookup(cache, repo, e) if stored else None) is not None:
            rec, e.reads = hit
            e.nodes, e.reused, e.words, e.seconds = rec["nodes"], True, rec.get("words"), rec.get("seconds")
    if not any(e.reused for e in entries) and apart is None:
        return None
    pinned = TG.pinned()
    listed, scratch = None, 0
    if apart is not None and (todo := [e for e in entries if not e.reused]):
        side = [u for e in entries if e.units != REGISTERED and not e.root_ids for u in e.units] if CC.tracing() else None
        if side is not None:
            CC.warm()
        kept, _, grew, scratch = built_apart(todo, *apart, described=True, side=side, strict=True, pinned=pinned)
        if log:
            over = sorted(((gb, n) for n, gb in grew.items() if gb > apart[2].get(n, APART_GB)), reverse=True)
            log(f"circuit-check: {len(kept)} of {len(todo)} bindings built apart, their results at most {scratch} bytes on "
                "disk at once" + (
                f"; {len(over)} grew past their heavy.json GiB: " + ", ".join(f"{n} {gb}" for gb, n in over) if over else ""))
        gone = {id(e) for e in todo} - {id(e) for e in kept}
        entries = [e for e in entries if id(e) not in gone]
        if side is not None and len(kept) == len(entries):
            listed = merged(e.listed for e in kept)
    made: list[dict[str, Any]] = []
    for e in entries:
        made.append({fn.id: fn for fn in e.roots if _kind(fn) is not None})
        if not e.reused and not e.apart:
            e.nodes, objs = _reach(e.roots, pinned=pinned)
            made[-1].update(objs)
    d = derive(entries, pinned=pinned)
    # an object built now stands for an id when its binding's record holds the descriptor of the one `derive` took
    objects: dict[str, Any] = {}
    for e, objs in zip(entries, made):
        for fid, fn in objs.items():
            if fid not in objects and (v := d.via.get(fid)) is not None and \
                    (v is e or e.nodes.get(fid, {}).get("d") == v.nodes[fid].get("d")):
                objects[fid] = fn
    for fid in d.ids:
        if "d" not in d.nodes[fid]:
            d.nodes[fid].update(_described(_materialize(entries, objects, fid, via=d.via.get(fid)), d.nodes[fid]["k"]))
    calls = call_families(entries, d, objects)
    out = Plan(entries, objects, d, calls, listed, scratch)
    words = made_words(out)
    if log:
        log(f"circuit-check: catalog from records: {sum(e.reused for e in entries)} of {len(entries)} bindings reused, "
            f"{sum(e.apart for e in entries)} built apart, {len(objects)} Definitions built here and {words} Boolean "
            "Definitions another process's build registered made or imported here")
    return out


def owners(plan: Plan) -> dict[str, str]:
    """`sidecars.owner` of each Definition a loaded module binds at its top level, by id, as a full build has it before its
    checks: in a process forked from this one that imports every module of `sidecars.PACKAGES` a binding's build imported (its
    reads' ``modules``, its record's or its process's), so this process imports none of them, as it builds no body.  Where two
    Definitions of one id are bound, the first owner by name.  `Stale` when one of them doesn't import there."""
    import importlib
    import warnings

    from . import sidecars as SC

    mods = sorted({m for e in plan.entries for m in (e.reads or {}).get("modules", ()) if m.partition(".")[0] in SC.PACKAGES})
    sys.stdout.flush()
    sys.stderr.flush()
    with CC.internal():
        out = _result()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)    # an xdist worker's idle threads
            pid = os.fork()
        if pid == 0:
            try:
                for m in mods:
                    importlib.import_module(m)
                SC._INDEX.clear()
                got: dict[str, str] = {}
                for v, mod in sorted(SC._index().values(), key=lambda e: e[1]):
                    got.setdefault(v.id, mod)
                _put(out, {"owners": got})
            except BaseException as ex:  # noqa: BLE001  (named where this process reads it: the run builds the whole catalog)
                _put(out, {"error": f"{type(ex).__name__}: {str(ex)[:200]}"})
            finally:
                os._exit(0)
        _, status = os.waitpid(pid, 0)
        if (got := _taken(out, status)) is None:
            got = {"error": f"its process died ({os.waitstatus_to_exitcode(status)})"}
    if "error" in got:
        raise Stale(f"the modules the bindings' builds imported, imported in one process for their Definitions' owners: {got['error']}")
    return got["owners"]


def made_words(plan: Plan) -> int:
    """Made here: each Boolean Definition (a ``word`` and a ``build``) that a binding's build registered where this process
    didn't build it (``words``: built apart, or its record's) and that this process doesn't hold, from the record that reaches
    it (`make`), else by importing the module that holds it (`_words`: one it registers as it is imported).  C-Flock lowers a
    word primitive by the build of the first registered Boolean Definition of that word (`verity_flock.ir_lower`'s
    ``_boolean_build``), so each check forked from here finds the build a full build's would.  `Stale` for one neither
    registers; the count made."""
    import importlib

    from verity.ml.circuits.defs import REGISTRY

    n = 0
    for e in plan.entries:
        for fid, _word, *where in e.words or ():
            if fid in REGISTRY:
                continue
            node = fid if fid in plan.derived.via else next((i for i, x in plan.derived.nodes.items() if x["f"] == fid), None)
            if node is not None:
                make(plan, node)
            elif where and where[0]:
                importlib.import_module(where[0])
            if node is None and fid not in REGISTRY:
                raise Stale(f"{fid}: {e.name}'s build registers it as a word's Boolean Definition, and neither a record nor the "
                            f"module that holds it ({where[0] if where else None}) does")
            n += 1
    return n


def store(cache: Path, repo: Path, entries: list[Entry], only_rebuilt: bool) -> int:
    """Store each binding's record (each rebuilt one's, with ``only_rebuilt``), its reads digested now, each read once for all
    of them (`cache.seen_all`); one built apart with the reads its process recorded; the count stored."""
    side = [u for e in entries if e.units != REGISTERED and not e.root_ids for u in e.units]
    todo = [e for e in entries if not (only_rebuilt and e.reused)]
    reads = [e.reads if e.reads is not None else CC.unit_reads((*e.units, *side) if e.units != REGISTERED else side, e.roots)
             for e in todo]
    n = 0
    for e, got, now in zip(todo, reads, CC.seen_all(repo, reads)):
        value = {"roots": e.root_ids, "nodes": e.nodes, **({"words": e.words} if e.words is not None else {}),
                 **({"seconds": e.seconds} if e.seconds is not None else {})}
        if CC.store(cache, e.key, repo, got, value, now) is not None:
            n += 1
    return n


def held(cache: Path, entries: list[Entry]) -> int:
    """The bytes ``entries``' records take in ``cache``: what a run from a copy of it copies."""
    with CC.internal():
        return sum(p.stat().st_size for e in entries for p in CC._dir(cache, e.key).glob("*.json"))


def made_entries() -> list[Entry]:
    """Every binding as the catalog makes it (`targets.catalog_units`, in catalog order), its roots made now and no body built,
    the registered roots first; a model that raises is named in `targets.ROOT_ERRORS`, as the catalog names it."""
    TG.load_registries()
    entries = registered_entries(TG._registered())
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
        entries.append(Entry(units, kind, label, key(units, n), roots))
    return entries


def _no_suite() -> dict:
    raise KeyError("no catalog here")


def _by_id(nodes: dict[str, dict], objs: dict[str, Any]) -> None:
    """Mark ``"i"`` on each node whose id `targets.definition` makes into the very object reached (`defs.bind` shares a
    specialization per binding record), so a process that builds no body makes it the same from its id (`make`)."""
    real = TG.suite
    TG.suite = _no_suite
    try:
        for fid, info in nodes.items():
            if info["k"] == "R" or TG._RECORD.search(fid):
                continue
            try:
                if TG.definition(fid) is objs[fid]:
                    info["i"] = 1
            except Exception:  # noqa: BLE001  (an id that doesn't make its object is made along its record's path)
                continue
    finally:
        TG.suite = real


def _from_id(fid: str) -> Any | None:
    """``fid``'s Definition as `targets.definition` makes it without the catalog, or None."""
    real = TG.suite
    TG.suite = _no_suite
    try:
        fn = TG.definition(fid)
    except Exception:  # noqa: BLE001  (a Definition only tracing registers: made along its record's path)
        return None
    finally:
        TG.suite = real
    return fn if fn.id == fid else None


def _reach_one(e: Entry, described: bool = False, pinned: frozenset[str] | None = None) -> dict[str, dict]:
    """``e``'s nodes (`_reach`), each body built here (a model's under its units, as the catalog builds it)."""
    if e.kind == "model":
        with CC.producing(e.units):
            _ = e.roots[0].body
    nodes, objs = _reach(e.roots, described=described, pinned=pinned)
    _by_id(nodes, objs)
    return nodes


def _words(before: set[str]) -> list[list[str | None]]:
    """``[id, word id, module]`` of each Boolean Definition (a ``word`` and a ``build``, as `verity.ml.circuits.boolean.trace`
    registers one) registered since the registry held ``before``, in the order it was registered; ``module`` the first imported
    that holds it by a name (one that registers it as it is imported), else None."""
    from verity.ml.circuits.defs import REGISTRY

    new = []
    for fid, d in REGISTRY.defs.items():
        w = getattr(getattr(d, "word", None), "id", None)
        if fid not in before and isinstance(w, str) and callable(getattr(d, "build", None)):
            new.append((fid, w, d))
    held: dict[int, str] = {}
    for name, m in list(sys.modules.items()) if new else ():
        if not isinstance(m, types.ModuleType):
            continue
        for v in list(object.__getattribute__(m, "__dict__").values()):    # not a lazy module's own lookup, which would load it
            held.setdefault(id(v), name)
    return [[fid, w, held.get(id(d))] for fid, w, d in new]


def _result() -> Any:
    """A file of no name for what a forked process returns (`_put`, `_taken`): the kernel frees it with the last process that
    holds it open, so none is left behind however the run ends."""
    import tempfile

    return tempfile.TemporaryFile()


def _put(f: Any, got: dict) -> None:
    f.seek(0)
    f.truncate()
    f.write(json.dumps(got).encode())
    f.flush()


def _taken(f: Any, status: int) -> dict | None:
    """What the process that ended with ``status`` (`os.waitpid`'s) put in ``f`` (`_put`), which is closed now; None when it
    didn't exit 0 or put nothing."""
    with f:
        if os.waitstatus_to_exitcode(status) != 0:
            return None
        f.seek(0)
        data = f.read()
    return json.loads(data) if data else None


def _child(e: Entry, out: Any, described: bool = False, side: list[str] | None = None, untraced: int = 0,
           pinned: frozenset[str] | None = None) -> None:
    """In a process forked for ``e``: its nodes or its error put in ``out`` (`_put`), and out through `os._exit`, so nothing of
    the parent's (its atexit, its buffers) runs twice.  With ``side`` (traced: the units of the bindings that make no root):
    what its build read (`cache.unit_reads`), its share of the task list's reads (`cache.task_list_reads` of its roots) and
    what it did that the tracer cannot see (`cache.untraced`, past the parent's ``untraced``); and the Boolean Definitions its
    build registered (`_words`).  ``pinned``: the parent's (`targets.pinned`), so the pins are no read of the binding's
    build."""
    from verity.ml.circuits.defs import REGISTRY

    from . import cli

    CC._INTERNAL.clear()              # forked inside `cache.internal`: what this process reads is its build's
    code = 0
    try:
        start = cli._rss_gb()
        before = set(REGISTRY.defs)
        try:
            got = {"nodes": _reach_one(e, described, pinned), "start": start, "words": _words(before)}
            if side is not None:
                got.update(reads=CC.unit_reads((*e.units, *side) if e.units != REGISTERED else side, e.roots),
                           listed=CC.task_list_reads(e.roots), untraced=CC.untraced()[untraced:])
        except BaseException as ex:  # noqa: BLE001  (a model's is named, as the catalog names it; any other fails the build)
            got = {"error": f"{type(ex).__name__}: {str(ex)[:200]}", "start": start}
        _put(out, got)
    except BaseException:  # noqa: BLE001
        code = 1
    finally:
        os._exit(code)


#: the GiB a binding's process is counted at in `built_apart` when `heavy.json` (`cache.HEAVY`) lists none for it: most bindings
#: grow less than 1.5 GiB past the process they were forked from, and it lists those that grow past 3 GiB
APART_GB = 2.0


def built_apart(entries: list[Entry], jobs: int, budget: float = float("inf"), weights: dict[str, float] | None = None, *,
                described: bool = False, side: list[str] | None = None, strict: bool = False,
                pinned: frozenset[str] | None = None) -> tuple[list[Entry], list[tuple[str, ...]], dict[str, float], int]:
    """``entries`` (`made_entries`) with each one's nodes (`_reach_one`, ``described`` or not, with this process's ``pinned``:
    `_child`), each binding built in a process forked from this one for it alone: this process builds no body, and each
    binding's bodies go with its process.  Up to
    ``jobs`` at once, the heaviest first, each started only while the ``weights`` (GiB by binding name, `cli.heavy`; `APART_GB`
    unlisted) of those running and its own fit ``budget``, as `cli.run` schedules its targets (`cli.admit`).  A model whose body
    raises, or whose process dies (but with ``strict``), is dropped (its units returned beside) and named in
    `targets.ROOT_ERRORS`, as the catalog names it; any other binding's error raises.  With ``side`` (`_child`), each with what
    its process read, and what that process did that the tracer cannot see is this one's too.  Returned with how far each
    binding's process grew, in GiB: its peak RSS past the RSS it started with, which it shares with this process.  Each process
    puts what it returns in a file of no name made for it (`_result`), which it alone of them holds open and this process reads
    once it ends: at most ``jobs`` such files at once, and none left behind however the run ends; returned last, the most bytes
    they held at once."""
    import gc
    import warnings

    from . import cli

    def weigh(i: int) -> float:
        return (weights or {}).get(entries[i].name, APART_GB)

    failed: dict[int, str] = {}
    died: set[int] = set()
    peaks: dict[str, float] = {}
    running: dict[int, int] = {}
    files: dict[int, Any] = {}
    scratch = 0
    queue = sorted(range(len(entries)), key=lambda i: -weigh(i))
    sys.stdout.flush()
    sys.stderr.flush()
    gc.collect()
    gc.freeze()                       # the children's collections then leave this process's pages shared
    try:
        with CC.internal():
            while queue or running:
                picked = cli.admit([weigh(i) for i in queue], [weigh(i) for i in running.values()], max(1, jobs), budget)
                for i in [queue[k] for k in picked]:
                    files[i] = _result()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)    # an xdist worker's idle threads
                        pid = os.fork()
                    if pid == 0:
                        for j, f in files.items():
                            if j != i:
                                f.close()     # else a sibling's file stays allocated until this process ends too
                        # what it inherits of the bindings that ended before it is this process's already: passed back
                        # again, the list would double with each binding that ends
                        _child(entries[i], files[i], described, side, len(CC._UNTRACED), pinned)
                    running[pid] = i
                queue = [i for i in queue if i not in running.values()]
                pid, status, usage = os.wait4(-1, 0)
                if (i := running.pop(pid, None)) is None:
                    continue
                scratch = max(scratch, sum(os.fstat(f.fileno()).st_size for f in files.values()))
                if (got := _taken(files.pop(i), status)) is None:
                    code = os.waitstatus_to_exitcode(status)
                    how = f"signal {-code}" if code < 0 else f"exit {code}"
                    got = {"error": f"its process died ({how}), its result unwritten"}
                    died.add(i)
                peaks[entries[i].name] = round(cli._maxrss_gb(usage.ru_maxrss) - (got.get("start") or 0.0), 2)
                if "error" in got:
                    failed[i] = got["error"]
                else:
                    e = entries[i]
                    e.nodes, e.apart = got["nodes"], True
                    e.reads, e.listed, e.words = got.get("reads"), got.get("listed"), got.get("words")
                    e.seconds = round(usage.ru_utime + usage.ru_stime, 2)
                    CC._UNTRACED.extend(got.get("untraced", ()))
    finally:
        for f in files.values():
            f.close()
        gc.unfreeze()
    for i, err in sorted(failed.items()):
        if entries[i].kind != "model" or (strict and i in died):
            raise RuntimeError(f"{entries[i].name}: {err} (its peak: {peaks[entries[i].name]} GiB past its start)")
        TG.ROOT_ERRORS.append((entries[i].label, err))
    return [e for i, e in enumerate(entries) if i not in failed], [entries[i].units for i in sorted(failed)], peaks, scratch


def planned(entries: list[Entry]) -> Plan:
    """The catalog of ``entries`` whose nodes `built_apart` set (or a manifest of them): no body built here, each root an object
    made now where its binding is the one `derive` took its id from.  No Call families: those nodes name no descriptor for
    `call_families` to match, and `targets.call_families` answers from the suite."""
    d = derive(entries, described=False)
    objects: dict[str, Any] = {}
    for e in entries:
        for fn in e.roots:
            fid = fn.id if _kind(fn) is not None else None
            if fid is not None and fid not in objects and d.via.get(fid) is e:
                objects[fid] = fn
    return Plan(entries, objects, d, set())


def built_here(plan: Plan, fid: str) -> Any | None:
    """``fid``'s object where this process makes it without building a body: a root made now, or from its id where its
    binding's build says that makes the very object it reached (`_by_id`); else None."""
    if fid in plan.objects:
        return plan.objects[fid]
    via = plan.derived.via.get(fid)
    if via is not None and via.nodes.get(fid, {}).get("i") and (fn := _from_id(fid)) is not None:
        plan.objects[fid] = fn
        return fn
    return None


def make(plan: Plan, fid: str) -> Any:
    """``fid``'s object: one `built_here` makes, else from the nearest Definition on its record's path to it whose id makes it
    (`_by_id`), building each body below it on the path; else from the binding's root (`_materialize`)."""
    if (fn := built_here(plan, fid)) is not None:
        return fn
    via = plan.derived.via.get(fid)
    path = _path(via, fid) if via is not None else None
    for j in range(len(path or ()) - 2, 0, -1):
        if via.nodes[path[j]].get("i") and (fn := _from_id(path[j])) is not None:
            for nxt in path[j + 1:]:
                fn = next((n.fn for n in fn.body.nodes if n.fn.id == nxt), None)
                if fn is None:
                    raise Stale(f"{fid}: {via.name}'s record lists a call to {nxt} that its body no longer makes")
            plan.objects[fid] = fn
            return fn
    return _materialize(plan.entries, plan.objects, fid, via=via)


def manifest(entries: list[Entry], dropped: list[tuple[str, ...]]) -> dict:
    """`built_apart`'s ``entries`` and ``dropped`` with `targets.ROOT_ERRORS`, as JSON: what `from_manifest` takes in another
    process."""
    return {"entries": [{"units": list(e.units), "kind": e.kind, "label": e.label, "roots": e.root_ids, "nodes": e.nodes}
                        for e in entries],
            "dropped": [list(u) for u in dropped], "root_errors": [list(x) for x in TG.ROOT_ERRORS]}


def from_manifest(entries: list[Entry], got: dict) -> list[Entry]:
    """``entries`` (`made_entries`, made in this process) with the nodes ``got`` (`manifest`) holds for them, in order, each the
    same binding with the same roots; a model whose body raised there is dropped and named, as the catalog does."""
    dropped = {tuple(u) for u in got["dropped"]}
    want = got["entries"]
    out = [e for e in entries if e.units not in dropped]
    if len(out) != len(want):
        raise Stale(f"the manifest holds {len(want)} bindings, this process made {len(out)}")
    for e, w in zip(out, want):
        if (list(e.units), e.kind, e.label, e.root_ids) != (w["units"], w["kind"], w["label"], w["roots"]):
            raise Stale(f"{e.name}: the manifest's binding differs")
        e.nodes = w["nodes"]
    TG.ROOT_ERRORS[:] = [tuple(x) for x in got["root_errors"]]
    return out


def built_entries() -> list[Entry] | None:
    """The bindings of this process's first full catalog (`targets.CATALOGS`), with what they reach; None when the registry
    grew between its catalogs (a record would not say what a later run's catalog starts from)."""
    if not TG.CATALOGS:
        return None
    registered, made = TG.CATALOGS[0]
    if any([fn.id for fn in c[0]] != [fn.id for fn in registered] for c in TG.CATALOGS[1:]):
        return None
    entries = registered_entries(registered)
    seen: Counter = Counter()
    for units, kind, label, roots in made:
        n, seen[units] = seen[units], seen[units] + 1
        entries.append(Entry(units, kind, label, key(units, n), roots))
    pinned = TG.pinned()
    for e in entries:
        e.nodes, _ = _reach(e.roots, pinned=pinned)
    return entries


__all__ = ["APART_GB", "REGISTERED_RECORDS", "REQUEST", "Derived", "Entry", "Plan", "Stale", "built_apart", "built_entries",
           "built_here", "call_families", "derive", "from_manifest", "key", "made_entries", "make", "manifest", "merged", "plan",
           "planned", "registered_entries", "store"]
