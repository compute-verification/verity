"""The partition as a named query over a program, P = Q(C): core's `verity/partition/v1` (`verity.ml.circuits.partition_object`),
applied here with the template queries `Q_template_instance` v0 and `Q_template_instances` v0, `Q_word` v1 at `X = 16`
and `Q_call` v1 at `X <= 16`, and the programs this protocol builds for the template queries.

    {"format": "verity/partition/v1", "program": "<SHA-512 of the program>",
     "query": {"name": "Q_template_instance", "version": 0, "params": {"template": "<Definition id>"}}}

* **Object, digest, program digest** are core's (`partition`, `digest`, `program_digest`, also named `program_sha512`). A
  record made before the canonical envelope names `{"program", "query"}`; `digest` gives it the envelope's rule,
  `SHA-512("verity/partition/v1\\0" || canon(object))`, which the roots registered with it bind.
* **Units** are core's evaluation (`partition_object.evaluate`) of a query of valid form (`partition_object.validate`):
  `QueryRefused` for a program that is not topological, or one the query does not apply to. The template queries,
  `Q_word` v1 at `X = 16` (the served query) and `Q_call` v1 at `X <= 16` are the ones applied here, so that every unit
  commits at most `UNIT_BITS` bits, the literal limit: `Q_word`'s width rule lets a unit of one value commit up to `W`
  bits, a template query has no width rule, and `Q_call`'s counts computing gates only, so a program with a wider unit
  (its Definition's activation's committed bits, constants and Input gates counted) is refused, as is one with a template
  of no gate. `ranges`, `units` and `strata` read their units in canonical order (node order,
  then member, for a template query; Call order, then the Call's own, for `Q_word` and `Q_call`), and `unit_gates` gives
  each unit's root gates.
* **Programs**: `population_program` (instances of one template) and `mixed_population_program` (of several).

The verifier holds its own copy of the program and applies the query itself (`units`); it never takes a population or a
unit index on the prover's word.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.cut import boundary_widths, call_scope
from verity.ml.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.ml.circuits.codec import descriptor_id, program_digest
from verity.ml.circuits.layout import resolve
from verity.ml.circuits.partition_object import (FORMAT, QueryRefused, program_sha512, template_id,
                                        template_instance_query, template_instances_query)
from verity.ml.circuits.program import Program, is_input
from verity.ml.circuits.refs import coll_concat
from verity.ml.circuits.types import Array, Tuple, Value

_PRE_CANONICAL = {"program", "query"}
Q_WORD = ("Q_word", 1)
#: The literal limit: the bits a unit commits.
UNIT_BITS = 16


def mixed_population_program(members: Sequence[tuple[SpecializedDefinition, int]]) -> Program:
    """`TemplatePopulations{counts}`: one batch per member `(template, n)`, in order; member j's parameters are `j.<name>`,
    arrays over its instances, and the result is the tuple of the batches' results."""
    if not members or any(n < 1 for _, n in members):
        raise ValueError("every member has at least one instance")
    sigs = [t.signature() for t, _ in members]

    def signature(S):
        params = tuple((f"{j}.{name}", Array(n, t)) for j, ((ps, _), n) in enumerate(zip(sigs, S.counts)) for name, t in ps)
        return params, Tuple(*(Array(n, ret) for (_, ret), n in zip(sigs, S.counts)))

    def body(B, S, *args):
        outs, at = [], 0
        for (t, _), (ps, _) in zip(members, sigs):
            outs.append(B.batch(t, *args[at:at + len(ps)], axes=(0,) * len(ps)))
            at += len(ps)
        return coll_concat(signature(S)[1], outs)

    defn = CompositeDefinition("TemplatePopulations", 0, ("counts",), signature, body,
                               doc="independent instances of " + ", ".join(f"{n} x {t.id}" for t, n in members), register=False)
    return Program(bind(defn, counts=tuple(n for _, n in members)))


def population_program(template: SpecializedDefinition, n: int) -> Program:
    """`TemplateInstances{N=n}`: `n` independent instances of `template` as one batch, every parameter an array over
    instances. Instance i of the batch is instance i of the input set it is built for."""
    if n < 1:
        raise ValueError("a population has at least one instance")
    params, ret = template.signature()

    def signature(S):
        return tuple((name, Array(S.N, t)) for name, t in params), Array(S.N, ret)

    def body(B, S, *args):
        return B.batch(template, *args, axes=(0,) * len(args))

    defn = CompositeDefinition("TemplateInstances", 0, ("N",), signature, body,
                               doc=f"{n} independent instances of {template.id}", register=False)
    return Program(bind(defn, N=n))


def partition(program: Program, query: Mapping[str, Any]) -> dict[str, Any]:
    return PO.build(program, query)


def digest(obj: Mapping[str, Any]) -> str:
    """Core's digest of a valid `verity/partition/v1` object (`ValueError` otherwise), or of a pre-canonical `{program, query}`."""
    if isinstance(obj, Mapping) and set(obj) == _PRE_CANONICAL:
        return hashlib.sha512(FORMAT.encode() + b"\0" + PO.canonical(dict(obj))).hexdigest()
    return PO.digest(obj)


def _check_word(part: PO.Partition) -> None:
    """`Q_word`'s cut commits at most `UNIT_BITS` bits a unit: each Definition's cut, once, on its Call's graph."""
    seen: set[str] = set()
    for p, (_k, c) in enumerate(part.calls):
        fid = descriptor_id(c.fn)
        if fid in seen:
            continue
        seen.add(fid)
        g, owner = part.graph(p), c.owner()
        if len(owner) != g.n:
            raise QueryRefused(f"{fid}'s cut leaves a computing gate out", {"definition": fid})
        ob, _ = boundary_widths(g, owner)
        wide = [u for u, b in enumerate(ob) if b > UNIT_BITS]
        if wide:
            raise QueryRefused(f"unit {wide[0]} of {fid} commits {ob[wide[0]]} bits, more than {UNIT_BITS}",
                               {"definition": fid, "unit": wide[0], "bits": ob[wide[0]]})


def _committed_bits(fn: Any, units: Sequence[tuple[int, int]]) -> list[int]:
    """Per unit `[lo, hi)` of one activation of `fn` (its local gates, as `Q_call` numbers them), the bits it commits: the
    summed declared widths of its gates that are Input gates, or that a later gate of the activation reads or the
    activation returns, constants included. Nothing outside an activation reads its gates but through its returns, so in
    any program this bounds the claims' `Prog.Boundary` count, as `Q_call`'s width rule bounds its computing gates'."""
    c, call = call_scope(fn)
    lo, n = call.offset, fn.gates
    width, inp, last = [0] * n, [False] * n, [-1] * n
    for j in range(n):
        g = c.gate(lo + j)
        width[j], inp[j] = (g.prim.ret.w if isinstance(g.prim.ret, Value) else 0), is_input(g.prim)
        for o in g.operands:
            if lo <= o < lo + j:
                last[o - lo] = j
    for r in ([0] if not hasattr(fn, "body") else (resolve(call, r) - lo for r in fn.body.ret)):
        if 0 <= r < n:
            last[r] = n
    return [sum(width[g] for g in range(a, b) if inp[g] or last[g] >= b) for a, b in units]


def _check_units(program: Program, part: PO.TemplatePartition | PO.CallPartition) -> None:
    """Refuses a template of no gate, and a template instance or `Q_call` unit that commits more than `UNIT_BITS` bits
    (`_committed_bits`, constants counted, which `Q_call`'s width rule leaves out), each Definition once, as activations of
    one Definition commit alike."""
    if isinstance(part, PO.TemplatePartition):
        templates = {t: program.root.body.nodes[k].fn for k, t, _first, n in part.ranges if n}
        for t, fn in templates.items():
            if fn.gates == 0:
                raise QueryRefused(f"template {t} has no gate", {"template": t})
            (bits,) = _committed_bits(fn, [(0, fn.gates)])
            if bits > UNIT_BITS:
                raise QueryRefused(f"an instance of {t} commits {bits} bits, more than {UNIT_BITS}",
                                   {"template": t, "bits": bits})
        return
    for cut in {cut.fn.id: cut for _k, cut in part.calls}.values():
        fn = cut.fn
        bits = _committed_bits(fn, [(lo, hi) for lo, hi, _path in cut.units])
        wide = next((u for u, b in enumerate(bits) if b > UNIT_BITS), None)
        if wide is not None:
            raise QueryRefused(f"unit {wide} of {fn.id} commits {bits[wide]} bits, more than {UNIT_BITS}",
                               {"definition": fn.id, "unit": wide, "bits": bits[wide]})


def evaluate(program: Program, query: Mapping[str, Any]) -> PO.TemplatePartition | PO.Partition | PO.CallPartition:
    """`query` applied to `program` by core, once core accepts the query's form; `QueryRefused` when the query is malformed,
    is not one applied here (a template query, `Q_word` v1 at `X = 16` or `Q_call` v1 at `X <= 16`), does not apply to the
    program (core's refusals, a program that is not topological among them), or has a unit of more than `UNIT_BITS` bits
    (`_check_word`, `_check_units`) or a template of no gate."""
    obj = PO.build(program, query)
    v, q = PO.validate(obj), obj["query"]
    if not v.ok:
        raise QueryRefused(f"not a {FORMAT} query ({', '.join(v.codes)}): {q!r}", {"query": q, "codes": list(v.codes)})
    kind, x = (q["name"], q["version"]), q["params"].get("X")
    if kind in (PO.Q_TEMPLATE_INSTANCE, PO.Q_TEMPLATE_INSTANCES):
        part = PO.evaluate(program, q)
        _check_units(program, part)
        return part
    if kind == Q_WORD and x == UNIT_BITS:
        part = PO.evaluate(program, q)
        _check_word(part)
        return part
    if kind == PO.Q_CALL and x <= UNIT_BITS:
        part = PO.evaluate(program, q)
        _check_units(program, part)
        return part
    raise QueryRefused(f"one-stage applies the template queries, Q_word v1 at X = {UNIT_BITS} and Q_call v1 at "
                       f"X <= {UNIT_BITS}, not {q['name']} v{q['version']} at X = {x}", {"query": q})


def ranges(program: Program, query: Mapping[str, Any]) -> list[tuple[str, int, int]]:
    """`query` applied to `program`: (template or Definition id, first unit index, unit count) per root node computing
    units (a template query's) or per Call with units (`Q_word`'s and `Q_call`'s), in canonical order."""
    part = evaluate(program, query)
    if isinstance(part, PO.TemplatePartition):
        return [(tid, lo, n) for _k, tid, lo, n in part.ranges]
    out, at = [], 0
    for _k, c in part.calls:
        n = c.units if isinstance(part, PO.Partition) else len(c.units)
        if n:
            out.append((descriptor_id(c.fn), at, n))
            at += n
    return out


def units(program: Program, query: Mapping[str, Any]) -> int:
    """The population of `query` applied to `program`: the number of units, indexed 0.. in canonical order."""
    return evaluate(program, query).population


def strata(program: Program, query: Mapping[str, Any]) -> list[tuple[str, list[list[int]]]]:
    """`query` applied to `program`, grouped by template or Definition: one stratum per descriptor id, its units as
    ascending, disjoint half-open ranges `[lo, hi)` with adjacent ones merged, the strata in the order of their first
    unit. These are the stratified law's strata, which each verifier derives from its own copies."""
    groups: dict[str, list[list[int]]] = {}
    for tid, lo, n in ranges(program, query):
        rs = groups.setdefault(tid, [])
        if rs and rs[-1][1] == lo:
            rs[-1][1] = lo + n
        else:
            rs.append([lo, lo + n])
    return list(groups.items())


def unit_gates(program: Program, query: Mapping[str, Any]) -> list[tuple[int, int]]:
    """Each unit's root gates `[lo, hi)`, by unit: an instance's (`TemplatePartition.gates`), a `Q_call` unit's interval
    (`CallPartition.gates`), or the span of the computing gates a `Q_word` unit owns, from the least to past the greatest,
    which another unit's gates may interleave."""
    part = evaluate(program, query)
    if not isinstance(part, PO.Partition):
        return [part.gates(u) for u in range(part.population)]
    out: list[tuple[int, int]] = []
    offsets = program.root.body.offsets
    for p, (k, c) in enumerate(part.calls):
        keep, span = part.graph(p).keep, [None] * c.units
        for i, u in enumerate(c.owner()):
            g = offsets[k] + keep[i]
            span[u] = (g, g + 1) if span[u] is None else (min(span[u][0], g), max(span[u][1], g + 1))
        if None in span:
            raise QueryRefused(f"a unit of {descriptor_id(c.fn)} owns no gate", {"definition": descriptor_id(c.fn)})
        out += span
    return out


__all__ = ["template_id", "FORMAT", "QueryRefused", "digest", "evaluate", "mixed_population_program", "partition",
           "population_program", "program_digest", "program_sha512", "ranges", "strata", "template_instance_query",
           "template_instances_query", "unit_gates", "units", "UNIT_BITS"]
