"""The population programs a window registers under a template query: `population_program` (instances of one template,
`TemplateInstances_v1{template, N}`) and `mixed_population_program` (of several, `TemplatePopulations_v1{templates, counts}`),
whose binding records name the templates by descriptor id. A mixed population may also take values its window registers,
parameters no node reads (`TemplatePopulations_v1{templates, values, counts}`). v0 of each, which names only the counts, is
kept for the records made with it (`POPULATION_VERSIONS`).

`wired_population_program` builds a circuit record's templates as one Program, each argument's runs of rows one part.
A template query partitions each of these Programs into its templates' units, and a prover and its verifier build the
same Program from the same templates and counts.
"""
from __future__ import annotations

import bisect
import hashlib
import math
import re
from collections.abc import Collection, Iterator, Mapping
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from .codec import canonical_json, descriptor_id, register_lazy_family
from .defs import CompositeDefinition, PrimitiveDefinition, SpecializedDefinition, bind
from .partition_object import template_instances_query
from .ports import In, Leaves, Out, Ports, port_types
from .program import Program
from .refs import Affine, Coll, Concat, Grid, Refs, Strided, coll_concat, concat, recut, spell
from .types import Array, RecordType, Tuple, Value

#: The population programs' versions, the builders' default first. v1 names its templates (each one's descriptor id) in its
#: binding record beside the counts, so the Definition's identity determines its body. v0 names the counts alone, so two
#: populations of equal counts over different templates share its id; it stays buildable for the records made with it.
POPULATION_VERSIONS = (1, 0)


def _population_version(version: int) -> int:
    if version not in POPULATION_VERSIONS:
        raise ValueError(f"no population program v{version}: the versions are {POPULATION_VERSIONS}")
    return version


def leaf_type(width: int | Sequence[int]) -> Value | Array:
    """A value's leaf type (`mixed_population_program`'s ``values``): ``Value(width)`` for an integer width, or for a shape
    ``(d_1, ..., d_k, width)`` of two or more naturals ``Array(d_1, ... Array(d_k, Value(width)))``. A `Digest`, eight
    SHA-512 words of 64 one-bit leaves, is ``(8, 64, 1)``."""
    if _natural(width):
        return Value(width)
    *dims, w = width
    t: Value | Array = Value(w)
    for d in reversed(dims):
        t = Array(d, t)
    return t


def _leaf_ok(width: Any) -> bool:
    return _natural(width) or (isinstance(width, tuple) and len(width) >= 2 and all(_natural(x) for x in width))


def mixed_population_program(members: Sequence[tuple[SpecializedDefinition, int]], *, version: int = 1,
                             values: Sequence[tuple[str, int, int | Sequence[int]]] = (),
                             reads: Sequence[tuple[str, int, int, str, Sequence[int]]] = ()) -> Program:
    """`TemplatePopulations_v1{templates, counts}`: one batch per member `(template, n)`, in order; member j's parameters are
    `j.<name>`, arrays over its instances, and the result is the tuple of the batches' results. `templates` are the members'
    descriptor ids, in order. A member whose population declares ports (`_instance_ports`) keeps them, under `j.<name>` and at
    its batch's offset in the result. ``values``: ``(name, leaves, width)`` each, values a window registers that the root
    takes ahead of the members' parameters, each the parameter ``name`` of type ``Array(leaves, leaf_type(width))``
    (``width`` an integer, or a shape such as a `Digest`'s ``(8, 64, 1)``, which the binding record holds as a list), read
    by no node and declared by no port, named in order in the binding record (`TemplatePopulations_v1{templates, values,
    counts}`): one-stage's R7 registers only a program's parameters, reading each by its type, and the proof blocks read
    the values only through their registered reads. ``reads``: ``(name, rows, j, param, at)`` each, an earlier period's
    committed value of ``rows`` rows that member j's parameter ``param`` reads, its instance i the value's row ``at[i]``:
    the root takes it bound at registration (`Program`'s ``registered``) as ``name`` of type ``Array(rows, row)``, after
    ``values``, in place of ``j.<param>``, so a registered read of it opens each instance's row at the position the
    circuit gives. `version=0` builds `TemplatePopulations_v0{counts}` (`POPULATION_VERSIONS`), which takes no values."""
    if not members or any(n < 1 for _, n in members):
        raise ValueError("every member has at least one instance")
    _population_version(version)
    sigs = [t.signature() for t, _ in members]
    values = tuple((name, leaves, tuple(width) if isinstance(width, list) else width) for name, leaves, width in values)
    reads = tuple((name, rows, j, param, tuple(int(a) for a in at)) for name, rows, j, param, at in reads)
    read_at = {(j, param): (name, rows, at) for name, rows, j, param, at in reads}
    own = [f"{j}.{name}" for j, (ps, _) in enumerate(sigs) for name, _ in ps if (j, name) not in read_at]
    if (values or reads) and version == 0:
        raise ValueError("a v0 population takes no values: its id names its counts alone")
    if any(not (isinstance(v, str) and v and _natural(n) and _leaf_ok(w)) for v, n, w in values) \
            or len({v for v, _, _ in values} | set(own)) != len(values) + len(own):
        raise ValueError(f"values {[v for v, _, _ in values]}: each (name, leaves, width), a new name beside the members' "
                         f"parameters, at least one leaf of at least one bit, a shape two or more naturals")
    if len(read_at) != len(reads) or len({r[0] for r in reads} | {v for v, _, _ in values} | set(own)) \
            != len(reads) + len(values) + len(own) or any(
            not (0 <= j < len(sigs) and param in dict(sigs[j][0]) and _natural(rows) and len(at) == members[j][1]
                 and all(0 <= a < rows for a in at)) for _name, rows, j, param, at in reads):
        raise ValueError(f"reads {[r[:4] for r in reads]}: each (name, rows, j, param, at) a new name, a parameter of member "
                         "j read once, at one row of the value's rows an instance")

    def row_type(j: int, param: str) -> Any:
        return dict(sigs[j][0])[param]

    def signature(S):
        params = tuple((f"{j}.{name}", Array(n, t)) for j, ((ps, _), n) in enumerate(zip(sigs, S.counts)) for name, t in ps
                       if (j, name) not in read_at)
        held = tuple((v, Array(n, leaf_type(w))) for v, n, w in getattr(S, "values", ()))
        taken = tuple((name, Array(rows, row_type(j, param))) for name, rows, j, param, _at in getattr(S, "reads", ()))
        return held + taken + params, Tuple(*(Array(n, ret) for (_, ret), n in zip(sigs, S.counts)))

    def body(B, S, *args):
        taken = dict(zip((r[0] for r in reads), args[len(values):len(values) + len(reads)]))
        outs, at = [], len(values) + len(reads)
        for j, ((t, n), (ps, _)) in enumerate(zip(members, sigs)):
            argv = []
            for name, pt in ps:
                if (j, name) in read_at:
                    v, _rows, rows_at = read_at[j, name]
                    argv.append(Coll(Array(n, pt), _grid([taken[v][r].refs for r in rows_at], pt.leaves)))
                else:
                    argv.append(args[at])
                    at += 1
            outs.append(B.batch(t, *argv, axes=(0,) * len(ps)))
        return coll_concat(signature(S)[1], outs)

    each = [_instance_ports(t) for t, _ in members]

    def ports(S: Any) -> Ports:
        params, sinks, at = {}, [], 0
        for j, ((_ps, ret), f, n) in enumerate(zip(sigs, each, S.counts)):
            if f is not None:
                p = f(SimpleNamespace(N=n))
                params.update((f"{j}.{name}", leaves) for name, leaves in p.params.items())
                sinks += [(at + lo, at + hi, o) for lo, hi, o in (p.ret.runs if p.ret is not None else ())]
            at += n * len(ret.leaf_widths())
        return Ports(params=params, ret=Leaves(tuple(sinks)) if sinks else None)

    named = {} if version == 0 else {"templates": tuple(descriptor_id(t) for t, _ in members),
                                     **({"values": values} if values else {}), **({"reads": reads} if reads else {})}
    defn = CompositeDefinition("TemplatePopulations", version, (*named, "counts"), signature, body,
                               doc="independent instances of " + ", ".join(f"{n} x {t.id}" for t, n in members), register=False,
                               ports=ports if any(f is not None for f in each) else None)
    return Program(bind(defn, **named, counts=tuple(n for _, n in members)), registered=tuple(r[0] for r in reads))


#: the circuit record's fields `wired_population_program`'s ``record`` binding holds
WIRED_FIELDS = ("wires", "cuts", "registered", "order", "ingress", "boundary_reads", "free", "external")

BIT = Value(1)
BIT_OF_ID = re.compile(r"^BitOf([1-9][0-9]*)x(0|[1-9][0-9]*)_v1$")
WORD_OF_ID = re.compile(r"^WordOf([1-9][0-9]*)_v1$")
_WIRING: dict[tuple[int, ...], PrimitiveDefinition] = {}


def bit_of(w: int, i: int) -> PrimitiveDefinition:
    """`BitOf{W}x{I}_v1`: (word: Value<W>) -> bit I of the word, least significant first."""
    if not (_natural(w) and isinstance(i, int) and 0 <= i < w):
        raise ValueError(f"bit {i!r} of a {w!r}-bit word")
    p = _WIRING.get((w, i))
    if p is None:
        p = _WIRING[(w, i)] = PrimitiveDefinition(
            f"BitOf{w}x{i}", 1, (("word", Value(w)),), BIT, lambda x, _i=i: (x >> _i) & 1, doc=bit_of.__doc__ or "",
            conformance="protocol-defined")
    return p


def word_of(w: int) -> PrimitiveDefinition:
    """`WordOf{W}_v1`: (bits: Array<W, Value<1>>) -> the W-bit word whose bit i is bits[i]."""
    if not _natural(w):
        raise ValueError(f"a {w!r}-bit word")
    p = _WIRING.get((w,))
    if p is None:
        p = _WIRING[(w,)] = PrimitiveDefinition(
            f"WordOf{w}", 1, (("bits", Array(w, BIT)),), Value(w), lambda *b: sum((x & 1) << i for i, x in enumerate(b)),
            doc=word_of.__doc__ or "", conformance="protocol-defined")
    return p


register_lazy_family(BIT_OF_ID, lambda m: bit_of(int(m[1]), int(m[2])))
register_lazy_family(WORD_OF_ID, lambda m: word_of(int(m[1])))


def _word_bits(B: Any, S: Any, x: Coll) -> Coll:
    """A W-bit word as its W one-bit leaves, bit i = leaf i, least significant first."""
    return Coll(Array(S.W, BIT), concat([B.call(bit_of(S.W, i), x).refs for i in range(S.W)]))


def _bits_word(B: Any, S: Any, bits: Coll) -> Coll:
    """W one-bit leaves as the W-bit word whose bit i is leaf i."""
    return B.call(word_of(S.W), bits)


WordBits = CompositeDefinition("WordBits", 1, ("W",), lambda S: ((("x", Value(S.W)),), Array(S.W, BIT)), _word_bits,
                               doc=_word_bits.__doc__ or "", conformance="protocol-defined")
BitsWord = CompositeDefinition("BitsWord", 1, ("W",), lambda S: ((("bits", Array(S.W, BIT)),), Value(S.W)), _bits_word,
                               doc=_bits_word.__doc__ or "", conformance="protocol-defined")


def wired_population_program(templates: Sequence[tuple[SpecializedDefinition, int]], record: Mapping[str, Any],
                             registered: Collection[str] = ()) -> tuple[Program, dict[str, Any]]:
    """`WiredPopulations_v1{templates, record, counts}` and its template query, as `Circuit.program` gives the population's:
    circuit record ``record`` as one Program over ``templates``, each ``(template, units)`` as the record names them. Each
    range of the record's order (each template one range where it states none) is a batch; each argument's runs of equal
    rows at one stride are one strided part, and the result is `mixed_population_program`'s. An input row is its wire's writer's output
    row; a registered or boundary read's, the root parameter named for the value it reads, a supplied input (with its
    mark) where the record's ``registered["supplied"]`` names it; a table read's,
    ``<param>/<root or domain>``, a supplied input marked `registered.<root or domain>/v1` of the table's ``param``; an
    opening's port, ``<table>/<port>`` over the units of that table's openings in the
    record's order, each opening's after the one before; any other row, the
    population's own ``<j>.<port>``. The registered values are the first parameters, in order, the rest sorted by name; those
    ``registered`` names are bound at registration (`Program`'s ``registered``). A
    read into a row of narrower leaves reads its root parameter's leaves recut (`refs.recut`). The Program's
    ``annotations["free"]`` are the rows no wire, read, table read, opening or ingress fills, as the record's ``free``
    entries state rows. ``record`` is the
    binding's by the SHA-512 of the canonical JSON of its `WIRED_FIELDS`. Refused (ValueError), by name: a row two fill, a
    wire between leaves of two widths (wire-width), a read `refs.recut` refuses, a wire whose writer the order puts at or
    after its reader."""
    ids = [descriptor_id(t) for t, _ in templates]
    if [list(x) for x in record.get("templates", ())] != [[tid, n] for tid, (_, n) in zip(ids, templates)]:
        raise ValueError(f"record templates {record.get('templates')}: the templates' ids and units, {ids}")
    if any(_instance_ports(t) is not None for t, _ in templates):
        raise ValueError("a template that takes a coin or a position is not wired")
    sigs = [t.signature() for t, _ in templates]
    units = [n for _, n in templates]
    index = {tid: j for j, tid in enumerate(ids)}
    reg, cuts = record.get("registered") or {}, record.get("cuts", {})
    order = [(index[t], lo, n) for t, lo, n in record.get("order") or [(tid, 0, n) for tid, n in zip(ids, units)]]
    rank = {}
    for r, (j, lo, n) in enumerate(order):
        rank.update(((j, u), r) for u in range(lo, lo + n))
    if len(rank) != sum(units) or sum(n for _, _, n in order) != sum(units):
        raise ValueError(f"order {record.get('order')}: its ranges tile every template's units")
    roots: dict[str, Any] = {}
    fills: dict[tuple[int, str], dict[tuple[int, int], tuple]] = {}

    def cut(j: int, port: str) -> int:
        return int(cuts.get(ids[j], {}).get(port, 1))

    def row(j: int, port: str) -> Any:
        t, k = dict(sigs[j][0])[port], cut(j, port)
        return t if k == 1 else Array(t.n // k, t.elem)

    def root(name: str, t: Any) -> None:
        if roots.setdefault(name, t) != t:
            raise ValueError(f"{name}: read as {roots[name]!r} and as {t!r}")

    def fill(what: Any, j: int, port: str, u: int, c: int, src: tuple) -> None:
        if fills.setdefault((j, port), {}).setdefault((u, c), src) is not src:
            raise ValueError(f"{what}: {ids[j]} {port} unit {u} row {c} is filled twice")

    def reader(p: Mapping[str, Any]) -> tuple[int, bool]:
        j = index[p["reader"]]
        return j, p["port"] in dict(sigs[j][0])

    def read(p: Mapping[str, Any], name: str, width: int) -> None:
        j, inp = reader(p)
        if not inp:
            return
        rt = row(j, p["port"])
        n = rt.leaves
        if rt.width_set() != {width}:
            r = rt.homogeneous_width()
            if not width or not r:
                raise ValueError(f"{p}: rows of {rt!r} read {name!r} in leaves of {width} bits")
            n, q = divmod(rt.leaves, recut(width, r))
            if q:
                raise ValueError(f"recut-boundary: {p}: a row of {rt!r} is no whole number of {name!r}'s {width}-bit leaves")
        cover = [(u, c, s * 16 // width) for u, c, s in _cover(p)]
        end = max(s for _, _, s in cover) + n
        if name not in bits:
            root(name, Array(roots[name].n if name in roots else 1, Value(width)))
            roots[name] = Array(max(end, roots[name].n), Value(width))
        elif end > roots[name].n:
            raise ValueError(f"{p}: leaves to {end} of {name!r}'s {roots[name].n}")
        for u, c, s in cover:
            fill(p, j, p["port"], u, c, ("leaf", name, s))

    bits = {v: int(reg.get("formats", {}).get(v, {}).get("word_bits", 16)) for v in reg.get("params", {})}
    for v, n in reg.get("params", {}).items():
        root(v, Array(n * 16 // bits[v], Value(bits[v])))
    for w in record.get("wires", ()):
        j, k, f = index[w["reader"]], index[w["writer"]], int(w["out"][1:])
        ft, rt = _fields(sigs[k][1])[f], row(j, w["port"])
        if ft.leaf_widths() != rt.leaf_widths():
            raise ValueError(f"wire-width: wire {w}: {ids[k]} writes {ft!r} into a row of {rt!r}")
        for u, c, s in _cover(w):
            if rank[(k, s)] >= rank[(j, u)]:
                raise ValueError(f"wire {w}: its order puts writer unit {s} at or after reader unit {u}")
            fill(w, j, w["port"], u, c, ("wire", k, s, f))
    for r in reg.get("reads", ()):
        read(r, r["param"], bits[r["param"]])
    for r in record.get("boundary_reads", ()):
        j, inp = reader(r)
        if inp and r["param"] in bits:
            raise ValueError(f"boundary read {r}: its field is named like a registered parameter")
        if inp:
            read(r, r["param"], row(j, r["port"]).homogeneous_width() or 0)
    tables = {t["name"]: t for t in reg.get("tables", ())}
    anchors: dict[str, dict[str, Any]] = {}
    for r in reg.get("roots", ()):
        j, name = index[r["reader"]], f"{tables[r['table']]['param']}/{r['of']}"
        root(name, dict(sigs[j][0])[r["port"]])
        if r["of"] in ("root", "domain"):
            if roots[name].leaves != 512 or roots[name].width_set() != {1}:
                raise ValueError(f"table read {r}: a table's {r['of']} is a Digest, not {roots[name]!r}")
            anchors[name] = {"supplied": f"registered.{r['of']}/v1", "args": {"value": tables[r["table"]]["param"]},
                             "layout": "digest", "bits": 512}
        for u in range(r["at"], r["at"] + r["units"]):
            for c in range(cut(j, r["port"])):
                fill(r, j, r["port"], u, c, ("whole", name, None))
    opened: dict[str, int] = {}
    for o in reg.get("openings", ()):
        j = index[o["reader"]]
        for p in o["ports"]:
            name, t = f"{o['table']}/{p}", dict(sigs[j][0])[p]
            if name in roots and name not in opened:
                raise ValueError(f"opening {o}: {name} is named like a read's or a table read's root")
            off = opened.get(name, 0)
            if off and roots[name].elem != t:
                raise ValueError(f"opening {o}: {name} opened as {roots[name].elem!r} and as {t!r}")
            roots[name], opened[name] = Array(off + o["units"], t), off + o["units"]
            for u in range(o["at"], o["at"] + o["units"]):
                for c in range(cut(j, p)):
                    fill(o, j, p, u, c, ("whole", name, off + u - o["at"]))
    ingress = {(index[g["reader"]], g["port"], u) for g in record.get("ingress", ()) for u in range(g["at"], g["at"] + g["units"])}
    free = []
    for j, ((ps, _), n) in enumerate(zip(sigs, units)):
        for p, _t in ps:
            for u in range(n):
                k, rows = cut(j, p), []
                for c in range(k):
                    if (u, c) not in fills.get((j, p), {}):
                        fill("own", j, p, u, c, ("own", f"{j}.{p}", None))
                        if (j, p, u) not in ingress:
                            rows.append(c)
                for c0 in (c for c in rows if c - 1 not in rows):
                    c1 = next(c for c in range(c0, k + 1) if c not in rows)
                    last = free[-1] if free else None
                    if last and last[:2] == (ids[j], p) and last[2] + last[3] == u and last[4] == (c0, c1 - c0):
                        free[-1] = (*last[:3], last[3] + 1, last[4])
                    else:
                        free.append((ids[j], p, u, 1, (c0, c1 - c0)))
    values = list(reg.get("values", ()))
    mine = [(f"{j}.{p}", Array(n, t)) for j, ((ps, _), n) in enumerate(zip(sigs, units)) for p, t in ps
            if any(s[0] == "own" for s in fills[(j, p)].values())]
    if clash := {v for v, _ in mine} & set(roots):
        raise ValueError(f"{sorted(clash)}: a value named like a template's own parameter")
    used = {src[1] for fs in fills.values() for src in fs.values() if src[0] != "wire"}
    params = (tuple((v, roots[v]) for v in values) + tuple(sorted((v, t) for v, t in roots.items() if v in used - set(values)))
              + tuple(mine))
    rets = Tuple(*(Array(n, ret) for (_, ret), n in zip(sigs, units)))

    def body(B, S, *args):
        coll = dict(zip((v for v, _ in params), args))
        made: dict[int, list[tuple[int, Any]]] = {j: [] for j in range(len(sigs))}
        starts: dict[int, list[int]] = {j: [] for j in range(len(sigs))}

        def writer(a: int, b: int) -> int:
            return bisect.bisect_right(starts[a], b) - 1

        for j, lo, n in order:
            argv = []
            for p, t in sigs[j][0]:
                k, rt = cut(j, p), row(j, p)
                r = rt.homogeneous_width()
                parts = []
                for u in range(lo, lo + n):
                    for c in range(k):
                        kind, a, b, *f = fills[(j, p)][(u, c)]
                        if kind == "wire":
                            wlo, out = made[a][writer(a, b)]
                            e = out[b - wlo]
                            parts.append((e[f[0]] if isinstance(sigs[a][1], (Tuple, RecordType)) else e).refs)
                        elif kind == "leaf":
                            sp, ix, at = coll[a].refs[b]
                            q = roots[a].elem.w // r if r else 1
                            parts.append(Affine(sp, ix, at * q, rt.leaves) if q > 1 else coll[a].refs.slice(b, b + rt.leaves))
                        else:
                            v = coll[a] if kind == "whole" and b is None else coll[a][u if kind == "own" else b]
                            parts.append(v.refs.slice(c * rt.leaves, (c + 1) * rt.leaves))
                argv.append(Coll(Array(n, t), _grid(parts, t.leaves)))
            i = bisect.bisect_right(starts[j], lo)
            starts[j].insert(i, lo)
            made[j].insert(i, (lo, B.batch(templates[j][0], *argv, axes=(0,) * len(argv))))
        return coll_concat(rets, [coll_concat(Array(n, sigs[j][1]), [o for _, o in made[j]]) for j, n in enumerate(units)])

    named = {"templates": tuple(ids), "record": _record_digest(record), "counts": tuple(units)}
    defn = CompositeDefinition("WiredPopulations", 1, tuple(named), lambda S: (params, rets), body,
                               doc="wired units of " + ", ".join(f"{n} x {t.id}" for t, n in templates), register=False)
    free_rows = [{"reader": t, "port": p, "at": at, "units": n, **({} if rows == (0, cut(index[t], p)) else {"cut": list(rows)})}
                 for t, p, at, n, rows in free]
    given = reg.get("supplied") or {}
    if clash := sorted(v for v in anchors if v in given and given[v] != anchors[v]):
        raise ValueError(f"{clash}: a table's root or domain marked other than by its table read")
    supplied = {v: m for v, m in {**given, **anchors}.items() if v in dict(params)}
    return (Program(bind(defn, **named), annotations={"free": free_rows}, supplied=supplied,
                    registered=tuple(v for v, _ in params if v in registered)),
            template_instances_query([t for t, _ in templates]))


def _record_digest(record: Mapping[str, Any]) -> str:
    return hashlib.sha512(canonical_json({k: record[k] for k in WIRED_FIELDS if k in record})).hexdigest()


def _chunks(parts: Sequence[Refs]) -> Iterator[tuple[int, str, int, int, int]]:
    for p in parts:
        if isinstance(p, Concat):
            yield from _chunks(p.parts)
        elif isinstance(p, Grid):
            yield from _chunks(p.pieces())
        elif isinstance(p, Affine):
            if p.count:
                yield p.count, p.space, p.idx, p.base, 1
        elif isinstance(p, Strided):
            for b in range(0, p.count, p.inner):
                n = min(p.inner, p.count - b)
                yield n, p.space, p.idx, p.base + b // p.inner * p.outer_stride, p.inner_stride if n > 1 else 1
        else:
            yield from ((1, s, i, j, 1) for s, i, j in p)


def _rows(parts: Sequence[Refs], period: int) -> list[list[list]]:
    rows: list[list[list]] = []
    pos = 0
    for n, s, i, b, e in _chunks(parts):
        while n:
            t, o = divmod(pos, period)
            m = min(n, period - o)
            if t == len(rows):
                rows.append([])
            row = rows[t]
            q = row[-1] if row else None
            if q and (q[2], q[3]) == (s, i) and q[0] + q[1] == o and (q[1] == 1 or m == 1 or q[5] == e) \
                    and q[4] + q[1] * (q[5] if q[1] > 1 else e) == b:
                q[5] = q[5] if q[1] > 1 else e
                q[1] += m
            else:
                row.append([o, m, s, i, b, e])
            pos, n, b = pos + m, n - m, b + m * e
    return rows


def _levels(b: Sequence[int], whole: bool = True) -> list[tuple[int, int]] | None:
    n = len(b)
    if n == 1:
        return []
    s0, r = b[1] - b[0], 2
    while r < n and b[r] - b[r - 1] == s0:
        r += 1
    if r == n and not whole:
        return [(n, s0)]
    for r0 in (d for d in range(r, 1, -1) if not whole or n % d == 0):
        starts = b[::r0]
        if all(b[t] == starts[t // r0] + t % r0 * s0 for t in range(n)):
            rest = _levels(starts, whole)
            if rest is not None:
                return [(r0, s0), *rest]
    return None


def _fits(seqs: Sequence[Sequence[int]], t: int, T: int) -> int:
    if all(_levels(b[t:T]) is not None for b in seqs):
        return T - t
    lo, hi = 1, T - t
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if all(_levels(b[t:t + mid], whole=False) is not None for b in seqs):
            lo = mid
        else:
            hi = mid - 1
    return next((m for m in range(lo, max(1, lo - 64), -1) if all(_levels(b[t:t + m]) is not None for b in seqs)), 1)


def _grid(parts: Sequence[Refs], period: int) -> Refs:
    rows, out = _rows(parts, period), []
    t0 = 0
    while t0 < len(rows):
        sig = [(o, m, s, i, e) for o, m, s, i, _b, e in rows[t0]]
        t1 = t0 + 1
        while t1 < len(rows) and [(o, m, s, i, e) for o, m, s, i, _b, e in rows[t1]] == sig:
            t1 += 1
        L = math.gcd(*(m for _o, m, *_x in sig))
        slots = [(k, q) for k, (_o, m, *_x) in enumerate(sig) for q in range(m // L)]
        seqs = [[rows[t][k][4] + q * L * sig[k][4] for t in range(t0, t1)] for k, q in slots]
        t = 0
        while t < t1 - t0:
            n = _fits(seqs, t, t1 - t0)
            lanes = [(sig[k][2], sig[k][3], b[t], sig[k][4], tuple(_levels(b[t:t + n]) or ())) for (k, _q), b in zip(slots, seqs)]
            out.append(spell(L, lanes))
            t += n
        t0 = t1
    return concat(out)


def _cover(p: Mapping[str, Any]) -> Iterator[tuple[int, int, int]]:
    c0, c = p.get("cut", (0, 1))
    for i in range(math.prod(p["radix"])):
        q, s = i, p["base"]
        for r, st in zip(reversed(p["radix"]), reversed(p["step"])):
            s, q = s + q % r * st, q // r
        yield p["at"] + i // c, c0 + i % c, s


def _fields(ret: Any) -> tuple[Any, ...]:
    return tuple(f[1] if isinstance(f, tuple) else f for f in ret.fields) if isinstance(ret, (Tuple, RecordType)) else (ret,)


def _natural(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x > 0


def _instance_ports(template: SpecializedDefinition) -> Callable[[Any], Ports] | None:
    """The ports of `template`'s population, for a unit (a template that takes a coin or a position and returns a sink, a
    salted tile such as PoUW's `PouwH1Tile`) or a template whose declared inputs are positions alone (a key count, as in
    `ops.gemm.PvRowPad`): each instance's coin and position leaves keep their kind and each instance's sinks stay sinks, so
    the population is a unit too. None for any other template, whose population declares no ports (its id and digest
    unchanged); a seed's population (a coin and no sink) is so refused, since only PoUW's `seeds.py` mints seeds
    (`tests/test_seed_ports.py`)."""
    pt = port_types(template)
    if pt is None:
        return None
    params, ret = template.signature()
    kinds = [(name, len(t.leaf_widths()), [(lo, hi, p.kind) for lo, hi, p, _w in rs if p.kind in ("coin", "position")])
             for (name, t), rs in zip(params, pt.params)]
    kinds = [(name, w, rs) for name, w, rs in kinds if rs]
    sinks, w_ret = [(lo, hi) for lo, hi, o, _w in pt.ret if o.sink], len(ret.leaf_widths())
    if not kinds or not sinks and any(k == "coin" for _name, _w, rs in kinds for _lo, _hi, k in rs):
        return None

    def ports(S: Any) -> Ports:
        return Ports(params={name: Leaves(tuple((i * w + lo, i * w + hi, In(kind=k)) for i in range(S.N) for lo, hi, k in rs))
                             for name, w, rs in kinds},
                     ret=Leaves(tuple((i * w_ret + lo, i * w_ret + hi, Out(sink=True)) for i in range(S.N) for lo, hi in sinks))
                     if sinks else None)
    return ports


def population_program(template: SpecializedDefinition, n: int, *, version: int = 1) -> Program:
    """The Program of `population`: `TemplateInstances_v1{template, N=n}`."""
    return Program(population(template, n, version=version))


def population(template: SpecializedDefinition, n: int, *, version: int = 1) -> SpecializedDefinition:
    """`TemplateInstances_v1{template, N=n}`, unbuilt: `n` independent instances of `template` as one batch, every parameter an
    array over instances; `template` is its descriptor id. Instance i of the batch is instance i of the input set it is built
    for. A unit's population declares its coins, positions and sinks per instance (`_instance_ports`). `version=0` builds
    `TemplateInstances_v0{N=n}` (`POPULATION_VERSIONS`)."""
    if n < 1:
        raise ValueError("a population has at least one instance")
    _population_version(version)
    params, ret = template.signature()

    def signature(S):
        return tuple((name, Array(S.N, t)) for name, t in params), Array(S.N, ret)

    def body(B, S, *args):
        return B.batch(template, *args, axes=(0,) * len(args))

    named = {} if version == 0 else {"template": descriptor_id(template)}
    defn = CompositeDefinition("TemplateInstances", version, (*named, "N"), signature, body,
                               doc=f"{n} independent instances of {template.id}", register=False,
                               ports=_instance_ports(template))
    return bind(defn, **named, N=n)


__all__ = ["BitsWord", "POPULATION_VERSIONS", "WIRED_FIELDS", "WordBits", "bit_of", "leaf_type",
           "mixed_population_program", "population", "population_program", "wired_population_program", "word_of"]
