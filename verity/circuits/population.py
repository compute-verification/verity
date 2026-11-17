"""The population programs a window registers under a template query: `population_program` (instances of one template,
`TemplateInstances_v1{template, N}`) and `mixed_population_program` (of several, `TemplatePopulations_v1{templates, counts}`),
whose binding records name the templates by descriptor id. A mixed population may also take values its window registers,
parameters no node reads (`TemplatePopulations_v1{templates, values, counts}`). v0 of each, which names only the counts, is
kept for the records made with it (`POPULATION_VERSIONS`).

SPOT's one-stage protocol applies its template queries to them (`one_stage.partition`, which names them too), and an
integration builds its window's program with them: both sides build the same program from the same templates and counts.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterator, Mapping
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from .codec import canonical_json, descriptor_id, register_lazy_family
from .defs import CompositeDefinition, PrimitiveDefinition, SpecializedDefinition, bind
from .partition_object import template_instances_query
from .ports import In, Leaves, Out, Ports, port_types
from .program import Program
from .refs import Coll, coll_concat, concat
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
                             values: Sequence[tuple[str, int, int | Sequence[int]]] = ()) -> Program:
    """`TemplatePopulations_v1{templates, counts}`: one batch per member `(template, n)`, in order; member j's parameters are
    `j.<name>`, arrays over its instances, and the result is the tuple of the batches' results. `templates` are the members'
    descriptor ids, in order. A member whose population declares ports (`_instance_ports`) keeps them, under `j.<name>` and at
    its batch's offset in the result. ``values``: ``(name, leaves, width)`` each, values a window registers that the root
    takes ahead of the members' parameters, each the parameter ``name`` of type ``Array(leaves, leaf_type(width))``
    (``width`` an integer, or a shape such as a `Digest`'s ``(8, 64, 1)``, which the binding record holds as a list), read
    by no node and declared by no port, named in order in the binding record (`TemplatePopulations_v1{templates, values,
    counts}`): one-stage's R7 registers only a program's parameters, reading each by its type, and the proof blocks read
    the values only through their registered reads. `version=0` builds `TemplatePopulations_v0{counts}`
    (`POPULATION_VERSIONS`), which takes no values."""
    if not members or any(n < 1 for _, n in members):
        raise ValueError("every member has at least one instance")
    _population_version(version)
    sigs = [t.signature() for t, _ in members]
    values = tuple((name, leaves, tuple(width) if isinstance(width, list) else width) for name, leaves, width in values)
    own = [f"{j}.{name}" for j, (ps, _) in enumerate(sigs) for name, _ in ps]
    if values and version == 0:
        raise ValueError("a v0 population takes no values: its id names its counts alone")
    if any(not (isinstance(v, str) and v and _natural(n) and _leaf_ok(w)) for v, n, w in values) \
            or len({v for v, _, _ in values} | set(own)) != len(values) + len(own):
        raise ValueError(f"values {[v for v, _, _ in values]}: each (name, leaves, width), a new name beside the members' "
                         f"parameters, at least one leaf of at least one bit, a shape two or more naturals")

    def signature(S):
        params = tuple((f"{j}.{name}", Array(n, t)) for j, ((ps, _), n) in enumerate(zip(sigs, S.counts)) for name, t in ps)
        held = tuple((v, Array(n, leaf_type(w))) for v, n, w in getattr(S, "values", ()))
        return held + params, Tuple(*(Array(n, ret) for (_, ret), n in zip(sigs, S.counts)))

    def body(B, S, *args):
        outs, at = [], len(values)
        for (t, _), (ps, _) in zip(members, sigs):
            outs.append(B.batch(t, *args[at:at + len(ps)], axes=(0,) * len(ps)))
            at += len(ps)
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
                                     **({"values": values} if values else {})}
    defn = CompositeDefinition("TemplatePopulations", version, (*named, "counts"), signature, body,
                               doc="independent instances of " + ", ".join(f"{n} x {t.id}" for t, n in members), register=False,
                               ports=ports if any(f is not None for f in each) else None)
    return Program(bind(defn, **named, counts=tuple(n for _, n in members)))


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


def _converts(src: Any, dst: Any) -> int | None:
    if src.leaf_widths() == dst.leaf_widths():
        return None
    a, b = src.homogeneous_width(), dst.homogeneous_width()
    if not a or not b or 1 not in (a, b) or a * src.leaves != b * dst.leaves:
        raise ValueError(f"leaves of {src!r} into a row of {dst!r}: a wire or read changes width only between W-bit words and "
                         f"their one-bit leaves")
    return max(a, b)


def wired_population_program(templates: Sequence[tuple[SpecializedDefinition, int]], record: Mapping[str, Any]
                             ) -> tuple[Program, dict[str, Any]]:
    """`WiredPopulations_v1{templates, record, counts}` and its template query, as `Circuit.program` gives the population's:
    circuit record ``record`` as one Program over ``templates``, each ``(template, units)`` as the record names them. Each
    range of the record's order (each template one range where it states none) is a batch, and the result is
    `mixed_population_program`'s. An input row is its wire's writer's output row; a registered or boundary read's, the root
    parameter named for the value it reads; a table read's, ``<param>/<root or domain>``; an opening's port,
    ``<table>/<port>`` over the opening's units; any other row, the population's own ``<j>.<port>``. The registered values
    are the first parameters, in order, the rest sorted by name. A wire or read from W-bit words into a row of one-bit
    leaves passes through a batch of `WordBits{W}` just before its reader's, one from bits into words through
    `BitsWord{W}`, and the query lists each after the templates. The Program's ``annotations["free"]`` are the rows no
    wire, read, table read, opening or ingress fills, as the record's ``free`` entries state rows. ``record`` is the
    binding's by the SHA-512 of the canonical JSON of its `WIRED_FIELDS`. Refused (ValueError), by name: a row two fill, a
    wire or read between leaves of two widths neither of them one bit, a wire whose writer the order puts at or after its
    reader."""
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
    need: set[tuple[bool, int]] = set()

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
            if not width or not r or rt.leaves * r % width:
                raise ValueError(f"{p}: rows of {rt!r} read {name!r} in leaves of {width} bits")
            n = rt.leaves * r // width
            need.add((r == 1, _converts(Array(n, Value(width)), rt)))
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
        if ww := _converts(ft, rt):
            need.add((rt.homogeneous_width() == 1, ww))
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
    for r in reg.get("roots", ()):
        j, name = index[r["reader"]], f"{tables[r['table']]['param']}/{r['of']}"
        root(name, dict(sigs[j][0])[r["port"]])
        for u in range(r["at"], r["at"] + r["units"]):
            for c in range(cut(j, r["port"])):
                fill(r, j, r["port"], u, c, ("whole", name, None))
    for o in reg.get("openings", ()):
        j = index[o["reader"]]
        for p in o["ports"]:
            name = f"{o['table']}/{p}"
            if name in roots:
                raise ValueError(f"opening {o}: {name} is opened once")
            root(name, Array(o["units"], dict(sigs[j][0])[p]))
            for u in range(o["at"], o["at"] + o["units"]):
                for c in range(cut(j, p)):
                    fill(o, j, p, u, c, ("whole", name, u - o["at"]))
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
        for j, lo, n in order:
            argv = []
            for p, t in sigs[j][0]:
                k, rt = cut(j, p), row(j, p)
                r = rt.homogeneous_width()
                parts, widths = [], []
                for u in range(lo, lo + n):
                    for c in range(k):
                        kind, a, b, *f = fills[(j, p)][(u, c)]
                        w = r
                        if kind == "wire":
                            wlo, out = next(x for x in made[a] if x[0] <= b < x[0] + len(x[1]))
                            e = out[b - wlo]
                            parts.append((e[f[0]] if isinstance(sigs[a][1], (Tuple, RecordType)) else e).refs)
                            w = _fields(sigs[a][1])[f[0]].homogeneous_width()
                        elif kind == "leaf":
                            w = roots[a].elem.w
                            parts.append(coll[a].refs.slice(b, b + rt.leaves * (r or w) // w))
                        else:
                            v = coll[a] if kind == "whole" and b is None else coll[a][u if kind == "own" else b]
                            parts.append(v.refs.slice(c * rt.leaves, (c + 1) * rt.leaves))
                        widths.append(w)
                for w in sorted({w for w in widths if w != r}):
                    at = [i for i, x in enumerate(widths) if x == w]
                    bits = concat([parts[i] for i in at])
                    out = (B.batch(bind(WordBits, W=w), Coll(Array(len(at) * rt.leaves // w, Value(w)), bits), axes=(0,))
                           if r == 1 else
                           B.batch(bind(BitsWord, W=r), Coll(Array(len(at) * rt.leaves, Array(r, BIT)), bits), axes=(0,)))
                    for q, i in enumerate(at):
                        parts[i] = out.refs.slice(q * rt.leaves, (q + 1) * rt.leaves)
                argv.append(Coll(Array(n, t), concat(parts)))
            made[j].append((lo, B.batch(templates[j][0], *argv, axes=(0,) * len(argv))))
        return coll_concat(rets, [coll_concat(Array(n, sigs[j][1]), [o for _, o in sorted(made[j], key=lambda x: x[0])])
                                  for j, n in enumerate(units)])

    named = {"templates": tuple(ids), "record": hashlib.sha512(canonical_json(
        {k: record[k] for k in WIRED_FIELDS if k in record})).hexdigest(), "counts": tuple(units)}
    defn = CompositeDefinition("WiredPopulations", 1, tuple(named), lambda S: (params, rets), body,
                               doc="wired units of " + ", ".join(f"{n} x {t.id}" for t, n in templates), register=False)
    free_rows = [{"reader": t, "port": p, "at": at, "units": n, **({} if rows == (0, cut(index[t], p)) else {"cut": list(rows)})}
                 for t, p, at, n, rows in free]
    wiring = [bind(WordBits if to_bits else BitsWord, W=w) for to_bits, w in sorted(need)]
    return (Program(bind(defn, **named), annotations={"free": free_rows}),
            template_instances_query([t for t, _ in templates] + wiring))


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


__all__ = ["BitsWord", "POPULATION_VERSIONS", "WIRED_FIELDS", "WordBits", "bit_of", "leaf_type", "mixed_population_program",
           "population", "population_program", "wired_population_program", "word_of"]
