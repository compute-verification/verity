"""The catalog's bindings, each reused while what its build read is unchanged: with ``--catalog-units``, a ``--all`` run whose
task list misses (`cli.run`) builds the bodies of only the bindings whose record misses, and takes what every other binding's
roots reach from its record.

Every binding still runs (`targets.catalog_units`, in catalog order, each under its units, `targets._produce`): its roots are
made now, unbuilt, so a registry a binding enumerates is read as it is now (under ``--binding``, while the catalog's roots
record counts, only that binding's: below).  A binding's record (a `circuit_check.cache` entry of
the request ``catalog-unit``, keyed by the binding's units and which of that binding's occurrences it is) holds the ids of its
roots and what they reach (`_reach`): each primitive, specialization and reference by id, with its kind, family, gate count,
whether it is Boolean, the ids its body calls (in the order `targets.reachable` visits them) and, where `targets.select` could
keep it, its descriptor (`cache.descriptor`), name and descriptor id; a by-calls root (`targets.is_by_calls`) as ``bc``, its
body unbuilt, so with no gates and no calls.  Its reads (`cache.unit_reads`) are what its units read
and ran while they made its roots, the build of every Definition they reach, the bindings that made those, those modules' import
closures, its units by digest and what the process read outside every recording; and the units of every binding that makes no
root (one run for what it sets, as `syn.knob`'s), whose reads every record holds.  The registered primitives and static-free
composites are `REGISTERED_RECORDS` more, ``<registered>:i``, each of a run of them in their order (`registered_entries`), so
that their builds run apart from each other.  A record holds nothing of how its build went: the CPU seconds of a binding
built apart, which `cli._owners` weighs, are kept beside it (``<entry>.seconds``, `store`), so a record built again from an
unchanged tree is the record it replaces, byte for byte, and so is a tarball of the records' entries (the circuit-catalog
task's).

A record counts only when its every read is unchanged where it was (never through a move; a module whose source `inspect`
looked for only by whether it is there, `_unchanged`) and the roots its binding made now are the ones it lists; else its
binding is built as the catalog builds it and its record stored again.  The task list is then
what `cli._tasks` would build: `targets.select` over what the roots reach in `targets.reachable`'s order, PoUW's roots whole,
each target's units (`targets.BINDINGS`), whether it is a Call (`targets.call_families`: the word Call families,
`targets.CALLEE_CALLS`, and the Boolean version of each suite Definition of one, made from its binding where no built binding
holds it, `_materialize`) and its key.  Two bindings can reach two Definitions of one id (`AttnDotStep_v1` with its DOT bound, from `boolean_version`, and
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
`cache.unit_reads`) and the in-tree modules it imported (`cache.modules_since`), which with its record's reads are its share
of the task list's reads.  This process builds no body.  A target that misses, through such a binding or one whose record counted, is made where it is checked (`cli._made`,
`make`): by the zygote that forks the checks of that binding's targets, each under a recording its check then holds
(`cli._prepare`), so the bodies they share are built once, as a full build builds them, and no target is built or described
here one after another; the target is held to the descriptor its binding's build gave or its record holds, and one that
differs is its check's error.  A primitive its id makes, which has no body, is made here (`built_here`), and so is each
Boolean Definition a binding's build registered (`made_words`, or its module imported): Flock lowers a word primitive by the
first one registered of that word, so a check forked from here finds the one a full build's would.  A binding whose process
dies fails the run, as it would kill a build in one process.  When every binding was built apart, the task list's entry is
stored as a full build stores it, its reads every binding's share and this process's own; it and each record are digested
where this process stores them, as a full build's are (a full build digests none of its reads where it builds them either).
``--records-only`` (`cli.records`) builds and stores the records so and checks no target, so that each run from a copy of
its cache (each shard of one run, on a machine of its own) builds no binding.

It stores one more record, the catalog's roots (`store_roots`): every binding's root ids in catalog order and the suite's
Call families, held to what the task list's entry is held to (`cache.task_list_reads`: what every binding read while it made
its roots, every in-tree module imported and every binding file whole, so a new binding entry, a new Definition in a registry
module or a new module an enumerated package lists makes it miss) and to every binding's record's reads.  While it counts and
every binding's record counts, a ``--binding`` run makes only that binding's roots, each other binding's when something reads
them, held to the ids the record lists (`plan`'s ``binding``).  Beside each binding's root ids the record holds what making
them did beyond them, and so for working out the Call families (`_noting`: each Definition registered and each in-tree module
imported, in order); the run does that in each one's place, in catalog order (`_replay`), so its registry and its modules are
a full plan's: Flock lowers a word primitive by its registered Boolean Definition, which another binding's roots can be the
first to register (a ``MoeSelectBf16x{E}``'s, made per E).

Every binding's record is still read so, and the roots record holds every binding file's digest, so one binding's edit
changes what every ``--binding`` run reads.  ``--records-only --slices DIR`` cuts the full plan into what each of those runs
needs (`slices`): one part every run shares (what each binding's making its roots did beyond them, in catalog order, where
it did anything; the Call families and what working them out did; each Boolean Definition `made_words` made, with the path
its record reaches it by; each Definition's owner), and one part per binding (its records, and the targets `derive` took
from them with their units).  ``--binding NAME --slice SLICE SHARED`` plans from those alone (`sliced`), reading no other
binding's record: one binding's edit changes its own slice, and another's only where that binding's record or targets read
what changed; the shared part only where what a binding registers or imports, the families or the made words' paths do.

A run from the records, forked or not, has no process that imported every module a full build has imported before its
checks, and `sidecars.owner` reads those: a Definition a factory makes is owned by the module that binds it at its top level,
which only another binding's body may import, and a by-calls root's callees go by that owner.  So each Definition's owner is handed down as a full build has it (`owners`, `sidecars.HANDED`).
"""
from __future__ import annotations

import json
import os
import sys
import types
from collections import Counter, deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from . import cache as CC
from . import targets as TG

REQUEST = "catalog-unit"
REGISTERED = ("<registered>",)
#: the catalog's roots record (`store_roots`), beside the bindings'
ROOTS = ("<roots>",)
#: the version of what `write_slices` writes
SLICE = "circuit-check-slice/v1"
#: how many records the registered roots take (`registered_entries`): the longest of their builds bounds a ``--records-only``
#: run, and one of them all took as long as the longest binding's
REGISTERED_RECORDS = 8


class Stale(Exception):
    """A record disagrees with what this run built: the run builds the whole catalog instead."""


@dataclass
class Entry:
    """One binding as this run has it: its units, kind and label (`targets.catalog_units`), its key, the roots it made now,
    what they reach (its record's, or `_reach` of the bodies built now) and whether its record counted, with that record's
    reads.  Built in a process of its own (``apart``, `built_apart`), with what that process read: its record's reads and the
    in-tree modules it imported (``listed``, `cache.modules_since`), and the CPU seconds it took (``seconds``, kept beside its
    record).  ``words``: the
    Boolean Definitions its build registered (`_words`), where a process built it apart.  ``made`` None: its roots are the ids
    the catalog's roots record lists (``ids``, `plan`'s ``binding``), made by ``make`` when first read and held to those ids.
    ``effects``: what making its roots did beyond them (`_noting`), or doing that in their place (`_replay`).  ``n``: which of
    its units' occurrences in the catalog it is (its key's)."""
    units: tuple[str, ...]
    kind: str
    label: str
    key: dict
    made: list[Any] | None
    nodes: dict[str, dict] = field(default_factory=dict)
    reused: bool = False
    apart: bool = False
    reads: dict[str, list[str]] | None = None
    listed: dict[str, list[str]] | None = None
    seconds: float | None = None
    words: list[list[str | None]] | None = None
    ids: list[str] | None = None
    make: Callable[[], list[Any]] | None = None
    effects: list[list] | None = None
    n: int = 0

    @property
    def roots(self) -> list[Any]:
        if self.made is None:
            made = TG._produce(self.units, self.make)
            if [fn.id for fn in made if _kind(fn) is not None] != self.ids:
                raise Stale(f"{self.name}: its roots are not the ones the catalog's roots record lists")
            self.made = made
        return self.made

    @property
    def root_ids(self) -> list[str]:
        return list(self.ids) if self.made is None else [fn.id for fn in self.made if _kind(fn) is not None]

    @property
    def name(self) -> str:
        return ",".join(self.units) + (f":{self.label}" if self.units == REGISTERED else "")


def registered_entries(roots: list[Any]) -> list[Entry]:
    """The registered roots (`targets._registered`) as at most `REGISTERED_RECORDS` entries, each of a run of them in their order
    (so `derive` meets them in the order `targets.reachable` does), entry i keyed and labelled by i; one entry when there are
    none."""
    k = max(1, -(-len(roots) // REGISTERED_RECORDS))
    return [Entry(REGISTERED, "registered", str(i), key(REGISTERED, i), roots[j:j + k], n=i)
            for i, j in enumerate(range(0, max(1, len(roots)), k))]


def _kind(fn: Any) -> str | None:
    from verity.circuits.defs import PrimitiveDefinition, SpecializedDefinition
    from verity.circuits.evaluation import ScalarReference

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
    from verity.circuits import codec as C

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
    from verity.circuits import boolean as BL

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
    word_call_families`, which refuses a vocabulary kind that doesn't bind), `targets.CALLEE_CALLS`, the Boolean version of each
    one made on demand (`targets.on_demand_boolean_families`) and of those Boolean versions of a suite Definition of one
    (`targets.boolean_family`) that are of a suite family: each from its object, made (`_materialize`) only where it could be of
    a suite family no other one gave (`targets.boolean_links`)."""
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
    return words | found | set(TG.CALLEE_CALLS)


def _sited(v: Any) -> Any:
    """A module's or a unit's digest with each import site by the name it resolves to: the content beside it is what follows a
    move (`cache._moved`), and a site's module is a read of its own where the binding's build ran or imported it."""
    return {**v, "sites": [s[0] for s in v["sites"]]} if isinstance(v, dict) and "sites" in v else v


def _unchanged(was: dict, now: dict) -> bool:
    """Every read as recorded, where it was: a module or unit by its content and where its import sites resolve (`_sited`), a
    package an enumerator listed by the modules it lists, a module whose source `inspect` looked for (``present``) by whether
    it is there, as `cache.declared` declares it: torch's import asks for the source of every module the process loaded, and
    each record holds what the process read outside every recording."""
    for kind, v in was.items():
        if kind == "enums":
            if any((a is None) != (b is None) or (a is not None and a.keys() != b.keys())
                   for a, b in ((v[p], now["enums"].get(p)) for p in v)):
                return False
        elif kind in ("modules", "units"):
            if v.keys() != now[kind].keys() or any(_sited(x) != _sited(now[kind][r]) for r, x in v.items()):
                return False
        elif kind == "present":
            if v.keys() != now[kind].keys() or any((d is None) != (now[kind][n] is None) for n, d in v.items()):
                return False
        elif now.get(kind) != v:
            return False
    return True


def _held(reads: dict) -> dict:
    """What `_unchanged` holds a record's reads to, as JSON: a package an enumerator listed by the modules it lists, a module or
    unit by `_sited`, a module whose source `inspect` looked for by whether it is there, the rest as digested."""
    return json.loads(json.dumps({k: {p: None if a is None else sorted(a) for p, a in v.items()} if k == "enums" else
                                  {r: _sited(x) for r, x in v.items()} if k in ("modules", "units") else
                                  {n: d is not None for n, d in v.items()} if k == "present" else v
                                  for k, v in reads.items()}))


def _lookup(cache: Path, repo: Path, e: Entry) -> tuple[dict, dict[str, list[str]]] | None:
    """``e``'s record and its reads when every read it holds is unchanged (`_unchanged`) and it lists the roots made now; None
    otherwise, and for one that cannot be read."""
    with CC.internal():
        found = sorted(CC._dir(cache, e.key).glob("*.json"))
    for p in found:
        try:
            rec = CC.read(p)
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
            return {**got, "seconds": _took(p)}, reads
        except (OSError, KeyError, TypeError, AttributeError, ValueError) as x:
            CC._miss(p, x)
            continue
    return None


def _took(record: Path) -> float | None:
    """The CPU seconds of the build that last stored the record at ``record``, kept beside it (`store`); None when none was."""
    seconds = record.with_suffix(".seconds")
    with CC.internal():
        return float(seconds.read_text()) if seconds.is_file() else None


@dataclass
class Plan:
    """A run's catalog from records: each binding, the objects built now, the derived suite and whether each is a Call; when
    every binding was built apart, what the task list's entry holds beside this process's reads (``listed``); the most
    bytes the results of those built apart held on disk at once (`built_apart`); what working out the Call families did
    beyond them (``effects["families"]``) and making each model's roots that raised (``effects["raised"]``, by label), as
    `_noting` notes them, and every binding's place in the catalog, in order (``effects["at"]``, `_at`); and each Boolean
    Definition `made_words` made, in order (``words``)."""
    entries: list[Entry]
    objects: dict[str, Any]
    derived: Derived
    calls: set[str]
    listed: dict[str, list[str]] | None = None
    scratch: int = 0
    effects: dict[str, Any] = field(default_factory=dict)
    words: list[list[str | None]] = field(default_factory=list)

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


@contextmanager
def _noting(got: list) -> Iterator[None]:
    """``got`` set to what runs inside did beyond what it returns: ``[id, word id]`` of each Definition it registered, in the
    order it registered them (a Boolean Definition's ``word``'s id, else None), and each in-tree module it imported, in the
    order it imported them."""
    from verity.circuits.defs import REGISTRY

    defs, mods = set(dict.keys(REGISTRY.defs)), set(sys.modules)
    try:
        yield
    finally:
        got[:] = [[[fid, w if isinstance(w := getattr(getattr(dict.__getitem__(REGISTRY.defs, fid), "word", None), "id", None), str)
                    else None] for fid in dict.keys(REGISTRY.defs) if fid not in defs],
                  [m for m in list(sys.modules) if m not in mods and CC._rel(getattr(sys.modules[m], "__file__", None)) is not None]]


def _replay(units: tuple[str, ...], effects: list) -> bool:
    """Do what ``effects`` (`_noting`'s) notes, under ``units`` as making their roots ran (`cache.producing`, so what it makes
    is theirs): import each module, then make each Definition not registered yet, in order, by its id
    (`codec.lookup_primitive`: an input, a family made on demand) or as the Boolean version made on demand of its word
    (`verity.definitions.boolean.made_on_demand`).  False where a module doesn't import, the next Definition is made by
    neither or making it raises: the roots are to be made then, and register the rest as they did."""
    import importlib

    regs, mods = effects
    if not regs and not mods:
        return True
    from verity.circuits.codec import lookup_primitive
    from verity.circuits.defs import REGISTRY

    with CC.producing(units) if units else nullcontext():
        try:
            for m in mods:
                importlib.import_module(m)
            for fid, word in regs:
                if dict.__contains__(REGISTRY.defs, fid):
                    continue
                try:
                    lookup_primitive(REGISTRY, fid)
                except KeyError:
                    if word is not None:
                        from verity.definitions.boolean import made_on_demand

                        made_on_demand(word)
                if not dict.__contains__(REGISTRY.defs, fid):
                    return False
        except Exception:  # noqa: BLE001  (making the roots does it as a full plan does, or raises as one does)
            return False
    return True


def plan(cache: Path | None, repo: Path, log: Callable[[str], None] | None = None,
         apart: tuple[int, float, dict[str, float]] | None = None,
         binding: Callable[[tuple[str, ...]], bool] | None = None) -> Plan | None:
    """The catalog from the bindings' records (the module's docstring); None when there are none of this checker's version (a
    full build stores them) or none counts.  With ``apart`` (`built_apart`'s ``jobs``, ``budget`` and ``weights``), each binding
    whose record doesn't count is built in a process of its own, and so is every binding when none counts, there are none or
    there is no ``cache``: this process builds no body.  `Stale` when a record disagrees with what was built now.  A pinned id
    a reused record holds no descriptor for (stored before its pin, `derive`) is made from the binding it is taken from
    (`_materialize`) and described in this run; that record is not stored again.

    With ``binding`` (whether a binding's units are those of the one binding `cli.run`'s ``--binding`` checks), while the
    catalog's roots record counts (`store_roots`), only that binding's roots are made now: every other binding's are the ids
    that record lists, made where something reads them (`Entry.roots`), with what making them did beyond them done in their
    place (`_replay`; where it can't be, they are made), and the Call families are its, what working them out did done so
    too.  Where the catalog lists other bindings than the record, a binding makes other roots, making them or doing that in
    their place registers or imports otherwise than the record notes, or a binding's record misses, the plan is made as
    without ``binding``."""
    TG.load_registries()
    first = key(REGISTERED, 0)
    stored = cache is not None and CC._dir(cache, first).is_dir()
    if not stored and apart is None:
        return None
    ahead = CC.lookup(cache, key(ROOTS, 0), repo, moved=False) if stored and binding is not None else None
    want = deque(ahead[0]["entries"] if ahead is not None else ())
    raised = {label: (err, fx) for label, err, fx in ahead[0]["root_errors"]} if ahead is not None else {}
    registered = registered_entries(TG._registered())
    entries = list(registered)
    TG.ROOT_ERRORS.clear()
    effects: dict[str, Any] = {"raised": {}, "at": []}
    seen: Counter = Counter()
    for units, make, kind, label in TG.catalog_units():
        n, seen[units] = seen[units], seen[units] + 1
        effects["at"].append([list(units), kind, label, n])
        ids, fx = want.popleft()[3:] if want and want[0][:3] == [list(units), kind, label] else (None, None)
        noted: list = []
        if ahead is not None and ids is None:
            if kind == "model" and label in raised:
                err, fx = raised[label]
                with _noting(noted):
                    done = _replay(units, fx)
                if done and noted == fx:
                    TG.ROOT_ERRORS.append((label, err))
                    continue
            return plan(cache, repo, log, apart)
        try:
            with _noting(noted):
                roots = None if ids is not None and not binding(units) and _replay(units, fx) else TG._produce(units, make)
        except Exception as e:  # a model's error is named, as the catalog names it
            if kind != "model":
                raise
            if ahead is not None:
                return plan(cache, repo, log, apart)
            TG.ROOT_ERRORS.append((label, f"{type(e).__name__}: {str(e)[:200]}"))
            effects["raised"][label] = noted
            continue
        entry = Entry(units, kind, label, key(units, n), roots, effects=noted, n=n)
        if roots is None:
            entry.ids, entry.make = ids, make
        if ahead is not None and (entry.root_ids != ids or noted != fx):
            return plan(cache, repo, log, apart)
        hit = _lookup(cache, repo, entry) if stored else None
        if hit is not None:
            rec, entry.reads = hit
            entry.nodes, entry.reused, entry.words, entry.seconds = rec["nodes"], True, rec.get("words"), rec.get("seconds")
        elif ahead is not None:
            return plan(cache, repo, log, apart)
        elif kind == "model" and apart is None:
            try:
                with CC.producing(units):
                    _ = roots[0].body
            except Exception as e:  # noqa: BLE001  (named, as the catalog names it)
                TG.ROOT_ERRORS.append((label, f"{type(e).__name__}: {str(e)[:200]}"))
                effects["raised"][label] = noted
                continue
        entries.append(entry)
    hits = [_lookup(cache, repo, e) if stored else None for e in registered]
    if ahead is not None and (want or None in hits):
        return plan(cache, repo, log, apart)
    for e, hit in zip(registered, hits):
        if hit is not None:
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
        effects["raised"].update((e.label, e.effects) for e in todo if id(e) in gone)
        entries = [e for e in entries if id(e) not in gone]
        if side is not None and len(kept) == len(entries):
            listed = merged([*(e.reads for e in kept), *(e.listed for e in kept)])
    made: list[dict[str, Any]] = []
    for e in entries:
        made.append({fn.id: fn for fn in e.made or () if _kind(fn) is not None})
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
    fam: list = []
    with _noting(fam):
        if ahead is None or not _replay((), ahead[0]["families"]):
            calls = call_families(entries, d, objects)
        else:
            calls = set(ahead[0]["calls"])
    if ahead is not None and (fam != ahead[0]["families"] or calls != set(ahead[0]["calls"])):
        return plan(cache, repo, log, apart)
    effects["families"] = fam
    out = Plan(entries, objects, d, calls, listed, scratch, effects)
    words = made_words(out)
    if log:
        log(f"circuit-check: catalog from records: {sum(e.reused for e in entries)} of {len(entries)} bindings reused, "
            f"{sum(e.apart for e in entries)} built apart, {len(objects)} Definitions built here and {words} Boolean "
            "Definitions another process's build registered made or imported here" + (
            f"; {sum(e.made is not None for e in entries)} bindings' roots made, the others' listed by the catalog's roots "
            f"record, which {sum(e.made is None and any(e.effects) for e in entries)} of them registered or imported "
            "beyond, done here in their place" if ahead is not None else ""))
    return out


def store_roots(cache: Path, repo: Path, plan: Plan) -> str | None:
    """The catalog's roots record (`ROOTS`): each binding's units, kind, label, root ids and what making them did beyond them
    (`Entry.effects`) in catalog order, the models whose roots or body raised (`targets.ROOT_ERRORS`, each with what making
    its roots did), and the suite's Call families with what working them out did, held to what the task list's entry holds
    (`cache.task_list_reads` of every root and object, with what each binding built apart read of it, ``Plan.listed``) and
    to every binding's record's reads, digested now, unless one counts already; its entry id."""
    if (hit := CC.lookup(cache, key(ROOTS, 0), repo, moved=False)) is not None:
        return hit[1]
    reads = merged([CC.task_list_reads([*(fn for e in plan.entries for fn in e.roots), *plan.objects.values()]),
                    plan.listed or {}, *(e.reads or {} for e in plan.entries)])
    return CC.store(cache, key(ROOTS, 0), repo, reads, {
        "entries": [[list(e.units), e.kind, e.label, e.root_ids, e.effects] for e in plan.entries if e.units != REGISTERED],
        "root_errors": [[label, err, plan.effects["raised"].get(label, [[], []])] for label, err in TG.ROOT_ERRORS],
        "calls": sorted(plan.calls), "families": plan.effects["families"]})


def owners(plan: Plan) -> dict[str, str]:
    """`sidecars.owner` of each Definition a loaded module binds at its top level, by id, as a full build has it before its
    checks: in a process forked from this one that imports every module of `sidecars.packages` a binding's build imported (its
    reads' ``modules``, its record's or its process's), so this process imports none of them, as it builds no body.  Where two
    Definitions of one id are bound, the first owner by name.  `Stale` when one of them doesn't import there."""
    import importlib
    import warnings

    from . import sidecars as SC

    tops = SC.packages()
    mods = sorted({m for e in plan.entries for m in (e.reads or {}).get("modules", ()) if m.partition(".")[0] in tops})
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


def _word_steps(plan: Plan) -> Iterator[list]:
    """`made_words`' steps, each as it comes to it: ``[id, node, module, binding]`` of each Boolean Definition a binding's build
    registered (``words``) that this process doesn't hold yet, with the id its record reaches it by (``node``: its own, else
    the first of its family) and the module that holds it."""
    from verity.circuits.defs import REGISTRY

    for e in plan.entries:
        for fid, _word, *where in e.words or ():
            if fid in REGISTRY:
                continue
            node = fid if fid in plan.derived.via else next((i for i, x in plan.derived.nodes.items() if x["f"] == fid), None)
            yield [fid, node, where[0] if where else None, e.name]


def made_words(plan: Plan, steps: Iterable[list] | None = None) -> int:
    """Made here: each Boolean Definition (a ``word`` and a ``build``) that a binding's build registered where this process
    didn't build it (``words``: built apart, or its record's) and that this process doesn't hold, from the record that reaches
    it (`make`), else by importing the module that holds it (`_words`: one it registers as it is imported).  Flock lowers a
    word primitive by the build of the first registered Boolean Definition of that word (`verity_flock.ir_lower`'s
    ``_boolean_build``), so each check forked from here finds the build a full build's would.  ``steps``: those a full plan
    took (`slices`), else `_word_steps`; each one taken is in ``plan.words``, with whether an object built here already
    stood for its node (`make` then made nothing).  `Stale` for one neither registers; the count made."""
    import importlib

    from verity.circuits.defs import REGISTRY

    n = 0
    for fid, node, where, by in _word_steps(plan) if steps is None else steps:
        if fid in REGISTRY:
            continue
        held = node is not None and node in plan.objects
        if node is not None:
            make(plan, node)
        elif where:
            importlib.import_module(where)
        if node is None and fid not in REGISTRY:
            raise Stale(f"{fid}: {by}'s build registers it as a word's Boolean Definition, and neither a record nor the "
                        f"module that holds it ({where}) does")
        plan.words.append([fid, node, where, by, int(held)])
        n += 1
    return n


def _at(e: Entry) -> list:
    """``e``'s place in the catalog: its units, kind, label and which of its units' occurrences it is."""
    return [list(e.units), e.kind, e.label, e.n]


def _key(at: list) -> str:
    return json.dumps(at)


def _canonical(doc: dict) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()


def slices(plan: Plan, owners: dict[str, str], name: Callable[[tuple[str, ...]], str], repo: Path) -> tuple[dict, dict[str, dict]]:
    """What a ``--binding`` run needs of a full plan (`plan` with every root made, ``plan.words`` taken: ``--records-only``'s),
    cut so that a run from it reads no other binding's record (`sliced`): the part every binding's run shares, and each
    binding's (``name``, `shard.binding`) own.  Shared: what each binding's making its roots did beyond them, in catalog order,
    where it did anything (``replay``: `Entry.effects`; a model whose roots or body raised marked so), what working out the
    Call families did and the families (``families``, ``calls``), each Boolean Definition `made_words` made, with the path
    its record reaches it by (``words``; one an object built here already stood for made nothing, and is left out), and each
    Definition's owner (``owners``, `owners`).  A binding's: each of its entries' place, root ids, effects, key, reads
    (digested now, in the form `_unchanged` holds its record to: `_held`, so a slice changes only where its record would
    stop counting) and nodes, its record but its build's time; and each Definition `derive` took
    from one of them (the targets ``--binding`` checks, `cli._binding_of`), with its units and the root that stands for it
    as `plan` with ``binding`` has it (the registered roots' and the binding's own, held to the descriptor `derive` took).
    `Stale` where a binding has no reads to hold its record to, or a made word's record no longer reaches it."""
    d = plan.derived
    at = {id(e): _at(e) for e in plan.entries}
    by_at = {_key(at[id(e)]): e for e in plan.entries}
    replay: list[list] = []
    for a in plan.effects["at"]:
        if (e := by_at.get(_key(a))) is not None:
            if any(e.effects or ()):
                replay.append([a, e.effects])
        elif a[1] == "model" and any(fx := plan.effects["raised"].get(a[2]) or ()):
            replay.append([a, fx, 1])
    words = []
    for fid, node, where, by, held in plan.words:
        if held:
            continue
        w: dict[str, Any] = {"id": fid, "node": node, "module": where, "by": by}
        if node is not None:
            v = d.via[node]
            if (path := _path(v, node)) is None:
                raise Stale(f"{fid}: {v.name}'s record no longer reaches {node}, which made it")
            w["via"] = [at[id(v)], path, [1 if v.nodes[p].get("i") else 0 for p in path]]
        words.append(w)
    shared = {"schema": SLICE, "replay": replay, "families": plan.effects["families"], "calls": sorted(plan.calls),
              "words": words, "owners": dict(sorted(owners.items()))}
    groups: dict[str, list[Entry]] = {}
    for e in plan.entries:
        groups.setdefault(name(e.units), []).append(e)
    taken: dict[int, list[str]] = {}
    for fid in d.ids:
        taken.setdefault(id(d.via[fid]), []).append(fid)
    tops = {id(e): {fn.id: fn for fn in e.made or () if _kind(fn) is not None} for e in plan.entries}
    per: dict[str, dict] = {}
    for b, own in groups.items():
        mine = {id(e) for e in own}
        stand = [e for e in plan.entries if e.units == REGISTERED or id(e) in mine]
        targets = []
        for e in own:
            for fid in taken.get(id(e), ()):
                obj = next((at[id(s)] for s in stand if fid in tops[id(s)] and (
                    s is e or s.nodes.get(fid, {}).get("d") == e.nodes[fid].get("d"))), None)
                targets.append([fid, at[id(e)], sorted(d.units.get(fid, ())), obj])
        if any(e.reads is None for e in own):
            raise Stale(f"{b}: a record with no reads to hold it to")
        per[b] = {"schema": SLICE, "binding": b, "targets": sorted(targets, key=lambda t: t[0]), "entries": [
            {"at": at[id(e)], "ids": e.root_ids, "effects": e.effects, "key": e.key, "nodes": e.nodes,
             "reads": _held(CC._digests(repo, {"files": [], "dirs": [], **{k: list(v) for k, v in e.reads.items()}}))}
            for e in own]}
    return shared, per


def write_slices(to: Path, shared: dict, per: dict[str, dict]) -> dict:
    """`slices`' parts in ``to``, each canonical JSON: ``shared.json``, each binding's at the first 16 hex digits of its name's
    SHA-256 holding the SHA-512 of the shared part it was cut with, and ``index.json`` naming each binding's file.  Their
    counts and bytes."""
    import hashlib

    with CC.internal():
        to.mkdir(parents=True, exist_ok=True)
        body = _canonical(shared)
        (to / "shared.json").write_bytes(body)
        digest = "sha512:" + hashlib.sha512(body).hexdigest()
        files: dict[str, str] = {}
        sizes = []
        for b, part in sorted(per.items()):
            f = f"{hashlib.sha256(b.encode()).hexdigest()[:16]}.json"
            if f in files.values():
                raise Stale(f"{b}: its slice's file name is another binding's")
            data = _canonical({**part, "shared": digest})
            (to / f).write_bytes(data)
            files[b] = f
            sizes.append(len(data))
        (to / "index.json").write_bytes(_canonical({"schema": SLICE, "shared": "shared.json", "bindings": files}))
    return {"bindings": len(files), "shared_bytes": len(body), "bytes": sum(sizes), "largest": max(sizes, default=0)}


def read_slice(path: Path, shared: Path, binding: str) -> tuple[dict, dict]:
    """Binding ``binding``'s slice at ``path`` and the shared part at ``shared`` (`write_slices`); `Stale` unless they are of
    this version, the slice is that binding's and it was cut with that shared part."""
    import hashlib

    with CC.internal():
        mine, body = json.loads(path.read_bytes()), shared.read_bytes()
    got = json.loads(body)
    if mine.get("schema") != SLICE or got.get("schema") != SLICE:
        raise Stale(f"{path} and {shared}: not slices of {SLICE}")
    if mine.get("binding") != binding:
        raise Stale(f"{path}: binding {mine.get('binding')}'s slice, not {binding}'s")
    if mine.get("shared") != "sha512:" + hashlib.sha512(body).hexdigest():
        raise Stale(f"{path}: cut with another shared part than {shared}")
    return mine, got


def _first(roots: Iterable[Any], fid: str, where: str) -> Any:
    if (fn := next((r for r in roots if r.id == fid), None)) is None:
        raise Stale(f"{fid}: {where} makes no root of that id")
    return fn


def sliced(mine: dict, shared: dict, repo: Path, log: Callable[[str], None] | None = None) -> Plan:
    """The plan of a ``--binding`` run from its binding's slice and the shared part (`slices`, `read_slice`), as `plan` with
    ``binding`` makes it from every binding's record and the catalog's roots record, reading no other binding's record: every
    binding runs in catalog order, the binding's own making its roots, each other doing in its place what making them did
    beyond them (`_replay`; made where that can't be done) where the shared part notes something, the rest nothing; its own
    records held to its reads and the ids and effects of the roots it made now; the Call families' work done so too, the
    families the shared part's; and each Boolean Definition the full plan made, made here along the path its record
    reaches it by (`made_words`: another binding's roots made where that path starts at one).  The plan's entries are its
    own, its suite the Definitions they give, as `derive` took them.  `Stale` where the catalog, a record's reads, the roots
    or what making them did differ from the slice's: the slice is not of this tree."""
    TG.load_registries()
    regs = {_key(_at(e)): e for e in registered_entries(TG._registered())}
    TG.ROOT_ERRORS.clear()
    own = {_key(x["at"]): x for x in mine["entries"]}
    replay = {_key(r[0]): r for r in shared["replay"]}
    entries: dict[str, Entry] = {k: e for k, e in regs.items() if k in own}
    makes: dict[str, Callable[[], list[Any]]] = {}
    produced: dict[str, list[Any]] = {}
    met: list[str] = []
    seen: Counter = Counter()
    for units, make_, kind, label in TG.catalog_units():
        n, seen[units] = seen[units], seen[units] + 1
        k = _key([list(units), kind, label, n])
        makes[k] = make_
        x, r = own.get(k), replay.get(k)
        if x is None and r is None:
            continue
        met.append(k)
        noted: list = []
        try:
            with _noting(noted):
                if x is not None:
                    roots = TG._produce(units, make_)
                elif not _replay(units, r[1]):
                    if len(r) > 2:
                        raise Stale(f"{label}: what making its roots did before they raised can't be done in their place")
                    produced[k] = TG._produce(units, make_)
        except Stale:
            raise
        except Exception as e:  # a model's error is named, as the catalog names it
            if kind != "model":
                raise
            raise Stale(f"{label}: making its roots raised now ({type(e).__name__}: {str(e)[:200]})") from e
        if noted != (x["effects"] if x is not None else r[1]):
            raise Stale(f"{label}: making its roots, or doing in their place what that did, registered or imported otherwise "
                        "than the slice notes")
        if x is not None:
            entries[k] = Entry(units, kind, label, key(units, n), roots, effects=noted, n=n)
    if gone := (own.keys() - {*met, *regs}) | (replay.keys() - set(met)):
        raise Stale(f"the catalog no longer lists {len(gone)} of the slice's bindings ({sorted(gone)[0]})")
    if [k for k in met if k in replay] != [_key(r[0]) for r in shared["replay"]]:
        raise Stale("the catalog lists the bindings the shared part notes in another order")
    for k, x in own.items():
        e = entries[k]
        if e.root_ids != x["ids"]:
            raise Stale(f"{e.name}: its roots are not the ones its slice lists")
        now = CC._digests(repo, {r: list(v) for r, v in x["reads"].items()})
        if e.key != x["key"] or _held(now) != x["reads"]:
            raise Stale(f"{e.name}: its record in the slice doesn't count here (a read changed)")
        e.nodes, e.reads, e.reused = x["nodes"], now, True
    tops: dict[str, dict[str, Any]] = {}

    def top(k: str) -> dict[str, Any]:
        if k not in tops:
            if k not in regs and k not in entries:
                raise Stale(f"{k}: a root the slice names stands for a target, of no binding of its own")
            tops[k] = {fn.id: fn for fn in (regs[k] if k in regs else entries[k]).made if _kind(fn) is not None}
        return tops[k]

    ids: list[str] = []
    units_of: dict[str, set[str]] = {}
    nodes: dict[str, dict] = {}
    via: dict[str, Entry] = {}
    objects: dict[str, Any] = {}
    for fid, a, us, obj in mine["targets"]:
        e = entries.get(_key(a))
        if e is None or "d" not in e.nodes.get(fid, {}):
            raise Stale(f"{fid}: its slice's record holds no descriptor of it")
        ids.append(fid)
        units_of[fid], nodes[fid], via[fid] = set(us), e.nodes[fid], e
        if obj is not None:
            if (fn := top(_key(obj)).get(fid)) is None:
                raise Stale(f"{fid}: the root its slice names doesn't make it")
            objects[fid] = fn
    fam: list = []
    with _noting(fam):
        done = _replay((), shared["families"])
    if not done or fam != shared["families"]:
        raise Stale("working out the Call families can't be done in its place as the shared part notes")

    def produce(k: str) -> list[Any]:
        if k not in produced:
            if k not in makes:
                raise Stale(f"{k}: the catalog has no such binding")
            produced[k] = makes[k]()
        return produced[k]

    def stub(a: list, path: list[str], marks: list[int]) -> Entry:
        k = _key(a)
        if k in entries:
            return entries[k]
        on = {p: {**({"c": [path[i + 1]]} if i + 1 < len(path) else {}), **({"i": 1} if m else {})}
              for i, (p, m) in enumerate(zip(path, marks))}
        e = Entry(tuple(a[0]), a[1], a[2], {}, None, on, reused=True, ids=[path[0]], n=a[3])
        if k in regs:
            e.made = [_first(regs[k].made, path[0], "the registered roots")]
        else:
            e.make = lambda: [_first(produce(k), path[0], e.name)]
        return e

    plan = Plan(list(entries.values()), objects, Derived(ids, units_of, nodes, via), set(shared["calls"]),
                effects={"families": fam})
    steps = []
    for w in shared["words"]:
        if (node := w["node"]) is not None and node not in via:
            via[node] = stub(*w["via"])
        steps.append([w["id"], node, w["module"], w["by"]])
    words = made_words(plan, steps)
    if log:
        log(f"circuit-check: catalog from binding {mine['binding']}'s slice: {len(entries)} records of its own, {len(ids)} "
            f"targets, {sum(k not in own for k in met)} other bindings' effects done in their place ({len(produced)} bindings' "
            f"roots made) and {words} Boolean Definitions made or imported here")
    return plan


def store(cache: Path, repo: Path, entries: list[Entry], only_rebuilt: bool) -> int:
    """Store each binding's record (each rebuilt one's, with ``only_rebuilt``), its reads digested now, each read once for all
    of them (`cache.seen_all`); one built apart with the reads its process recorded, and the CPU seconds it took beside it
    (``<entry>.seconds``: `_took`), out of the record's bytes; the count stored."""
    side = [u for e in entries if e.units != REGISTERED and not e.root_ids for u in e.units]
    todo = [e for e in entries if not (only_rebuilt and e.reused)]
    reads = [e.reads if e.reads is not None else CC.unit_reads((*e.units, *side) if e.units != REGISTERED else side, e.roots)
             for e in todo]
    n = 0
    for e, got, now in zip(todo, reads, CC.seen_all(repo, reads)):
        value = {"roots": e.root_ids, "nodes": e.nodes, **({"words": e.words} if e.words is not None else {})}
        if (eid := CC.store(cache, e.key, repo, got, value, now)) is None:
            continue
        n += 1
        if e.seconds is not None:
            took = CC._dir(cache, e.key) / f"{eid}.seconds"
            # a temporary name of this writer's own, as `cache.store` writes an entry
            tmp = took.with_name(f".{took.name}.{os.getpid()}.{os.urandom(4).hex()}.tmp")
            try:
                tmp.write_text(f"{e.seconds}\n")
                os.replace(tmp, took)
            except FileNotFoundError as x:             # the version's directory went (`cache.expire`)
                CC.LOST.append(f"{took.parent.name}: {x!r}")
    return n


def held(cache: Path, entries: list[Entry]) -> int:
    """The bytes ``entries``' records and the catalog's roots record take in ``cache``, with the read records they share
    (`cache.held`): what a run from a copy of it copies."""
    with CC.internal():
        return CC.held(p for k in [*(e.key for e in entries), key(ROOTS, 0)] for p in CC._dir(cache, k).glob("*.json"))


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
        except Exception as e:  # a model's error is named, as the catalog names it
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
    """``[id, word id, module]`` of each Boolean Definition (a ``word`` and a ``build``, as `verity.circuits.boolean.trace`
    registers one) registered since the registry held ``before``, in the order it was registered; ``module`` the first imported
    that holds it by a name (one that registers it as it is imported), else None."""
    from verity.circuits.defs import REGISTRY

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
    what its build read (`cache.unit_reads`), the in-tree modules it imported (`cache.modules_since`), which with those are
    its share of the task list's reads, and what it did that the tracer cannot see (`cache.untraced`, past the parent's
    ``untraced``); and the Boolean Definitions its build registered (`_words`).  ``pinned``: the parent's (`targets.pinned`),
    so the pins are no read of the binding's build."""
    from verity.circuits.defs import REGISTRY

    from . import cli

    CC._INTERNAL.clear()              # forked inside `cache.internal`: what this process reads is its build's
    code = 0
    try:
        start = cli._rss_gb()
        before, loaded = set(REGISTRY.defs), set(sys.modules)
        try:
            got = {"nodes": _reach_one(e, described, pinned), "start": start, "words": _words(before)}
            if side is not None:
                got.update(reads=CC.unit_reads((*e.units, *side) if e.units != REGISTERED else side, e.roots),
                           listed=CC.modules_since(loaded), untraced=CC.untraced()[untraced:])
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


__all__ = ["APART_GB", "REGISTERED_RECORDS", "REQUEST", "ROOTS", "SLICE", "Derived", "Entry", "Plan", "Stale", "built_apart",
           "built_entries", "built_here", "call_families", "derive", "from_manifest", "key", "made_entries", "make", "manifest",
           "merged", "plan", "planned", "read_slice", "registered_entries", "sliced", "slices", "store", "store_roots",
           "write_slices"]
