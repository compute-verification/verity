"""The partition as a named query over a program, P = Q(C): core's `verity/partition/v1` (`verity.ml.circuits.partition_object`),
applied here with the template queries `Q_template_instance` v0 and `Q_template_instances` v0, and the programs this
protocol builds for them.

    {"format": "verity/partition/v1", "program": "<SHA-512 of the program>",
     "query": {"name": "Q_template_instance", "version": 0, "params": {"template": "<Definition id>"}}}

* **Object, digest, program digest** are core's (`partition`, `digest`, `program_digest`, also named `program_sha512`). A
  record made before the canonical envelope names `{"program", "query"}`; `digest` gives it the envelope's rule,
  `SHA-512("verity/partition/v1\\0" || canon(object))`, which the roots registered with it bind.
* **Units** are core's evaluation (`partition_object.evaluate`) of a query of valid form (`partition_object.validate`):
  `QueryRefused` for a program that is not topological, or one the query does not apply to. The template queries are the
  only ones applied here; `ranges`, `units` and `strata` read their units in canonical order (node order, then member).
* **Programs**: `population_program` (instances of one template) and `mixed_population_program` (of several).

The verifier holds its own copy of the program and applies the query itself (`units`); it never takes a population or a
unit index on the prover's word.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.ml.circuits.codec import program_digest
from verity.ml.circuits.partition_object import (FORMAT, QueryRefused, program_sha512, template_id,
                                        template_instance_query, template_instances_query)
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import coll_concat
from verity.ml.circuits.types import Array, Tuple

_PRE_CANONICAL = {"program", "query"}


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


def evaluate(program: Program, query: Mapping[str, Any]) -> PO.TemplatePartition:
    """`query` applied to `program` by core, once core accepts the query's form; `QueryRefused` when the query is malformed,
    is not a template query, or does not apply to the program (core's refusals, a program that is not topological among them)."""
    obj = PO.build(program, query)
    v, q = PO.validate(obj), obj["query"]
    if not v.ok:
        raise QueryRefused(f"not a {FORMAT} query ({', '.join(v.codes)}): {q!r}", {"query": q, "codes": list(v.codes)})
    if (q["name"], q["version"]) not in (PO.Q_TEMPLATE_INSTANCE, PO.Q_TEMPLATE_INSTANCES):
        raise QueryRefused(f"one-stage applies the template queries, not {q['name']} v{q['version']}", {"query": q})
    return PO.evaluate(program, q)


def ranges(program: Program, query: Mapping[str, Any]) -> list[tuple[str, int, int]]:
    """`query` applied to `program`: per root node computing units, (template id, first unit index, unit count), in
    canonical order."""
    return [(tid, lo, n) for _k, tid, lo, n in evaluate(program, query).ranges]


def units(program: Program, query: Mapping[str, Any]) -> int:
    """The population of `query` applied to `program`: the number of units, indexed 0.. in canonical order."""
    return evaluate(program, query).population


def strata(program: Program, query: Mapping[str, Any]) -> list[tuple[str, list[list[int]]]]:
    """`query` applied to `program`, grouped by template: one stratum per template descriptor id, its units as ascending,
    disjoint half-open ranges `[lo, hi)` with adjacent ones merged, the strata in the order of their first unit. These are
    the stratified law's strata, which each verifier derives from its own copies."""
    groups: dict[str, list[list[int]]] = {}
    for tid, lo, n in ranges(program, query):
        rs = groups.setdefault(tid, [])
        if rs and rs[-1][1] == lo:
            rs[-1][1] = lo + n
        else:
            rs.append([lo, lo + n])
    return list(groups.items())


__all__ = ["template_id", "FORMAT", "QueryRefused", "digest", "evaluate", "mixed_population_program", "partition",
           "population_program", "program_digest", "program_sha512", "ranges", "strata", "template_instance_query",
           "template_instances_query", "units"]
