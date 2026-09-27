"""The partition as a named query over a program, P = Q(C), and the demo's query `Q_template_instance` v0.

    {"program": "<SHA-512 of the program>", "query": {"name": "Q_template_instance", "version": 0,
                                                      "template": "<Definition id>", "width_rule": null}}

* **Digest.** `SHA-512("verity/partition/v1\\0" || canon({program, query}))`, canon = `verity.ir.codec.canonical_json`.
* **Program.** SHA-512 over the canonical descriptor `codec.program_digest` hashes with SHA-256 (annotations excluded).
* **`Q_template_instance` v0.** Every instance of one template's Definition in the program's root body is one unit, in
  canonical order (node order, then batch member). The `Input` gates are the input unit: always checked, never drawn.
  No width rule is applied, so a unit is a whole instance however wide its outputs. The query refuses a program whose
  root body computes anything but instances of its template.
* **`Q_template_instances` v0.** The same over several templates (`templates`, a list of Definition ids): every instance
  of any of them is one unit, in canonical order, and every listed template must have an instance. `ranges` gives each
  root node's template and its units' first index and count.

The verifier holds its own copy of the program and applies the query itself (`units`); it never takes a population or a
unit index on the prover's word.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from verity.ir.codec import canonical_json, encode_program
from verity.ir.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.ir.program import Program
from verity.ir.refs import coll_concat
from verity.ir.types import Array, Tuple

FORMAT = "verity/partition/v1"
Q_TEMPLATE_INSTANCE = "Q_template_instance"
Q_TEMPLATE_INSTANCE_VERSION = 0
Q_TEMPLATE_INSTANCES = "Q_template_instances"
Q_TEMPLATE_INSTANCES_VERSION = 0


class QueryRefused(ValueError):
    """The query does not apply to the program."""


def template_instance_query(template_id: str) -> dict[str, Any]:
    return {"name": Q_TEMPLATE_INSTANCE, "version": Q_TEMPLATE_INSTANCE_VERSION, "template": template_id,
            "width_rule": None}


def template_instances_query(template_ids: Sequence[str]) -> dict[str, Any]:
    return {"name": Q_TEMPLATE_INSTANCES, "version": Q_TEMPLATE_INSTANCES_VERSION, "templates": list(template_ids),
            "width_rule": None}


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


def program_sha512(program: Program) -> str:
    desc = encode_program(program)
    return hashlib.sha512(canonical_json({k: v for k, v in desc.items() if k != "annotations"})).hexdigest()


def partition(program: Program, query: Mapping[str, Any]) -> dict[str, Any]:
    return {"program": program_sha512(program), "query": dict(query)}


def digest(obj: Mapping[str, Any]) -> str:
    return hashlib.sha512(FORMAT.encode() + b"\0" + canonical_json(dict(obj))).hexdigest()


def ranges(program: Program, query: Mapping[str, Any]) -> list[tuple[str, int, int]]:
    """`query` applied to `program`: per root node computing units, (template id, first unit index, unit count), in
    canonical order."""
    kind = (query.get("name"), query.get("version"))
    if kind == (Q_TEMPLATE_INSTANCE, Q_TEMPLATE_INSTANCE_VERSION):
        templates = [query.get("template")]
    elif kind == (Q_TEMPLATE_INSTANCES, Q_TEMPLATE_INSTANCES_VERSION):
        templates = list(query.get("templates") or [])
        if len(set(templates)) != len(templates):
            raise QueryRefused("Q_template_instances lists a template twice")
    else:
        raise QueryRefused(f"unknown query {query.get('name')!r} v{query.get('version')!r}")
    if query.get("width_rule") is not None:
        raise QueryRefused(f"{query['name']} v0 applies no width rule")
    out: list[tuple[str, int, int]] = []
    at = 0
    for k, node in enumerate(program.root.body.nodes):
        if getattr(node.fn, "family", None) == "Input":
            continue
        if node.fn.id not in templates or node.form not in ("call", "batch"):
            raise QueryRefused(f"root node {k} computes {node.fn.id!r} ({node.form}), not an instance of {templates!r}")
        n = node.n if node.form == "batch" else 1
        out.append((node.fn.id, at, n))
        at += n
    missing = set(templates) - {t for t, _, _ in out}
    if missing:
        raise QueryRefused(f"the program has no instance of {sorted(missing)!r}")
    return out


def units(program: Program, query: Mapping[str, Any]) -> int:
    """The population of `query` applied to `program`: the number of units, indexed 0.. in canonical order."""
    return sum(n for _, _, n in ranges(program, query))


__all__ = ["FORMAT", "Q_TEMPLATE_INSTANCE", "Q_TEMPLATE_INSTANCE_VERSION", "Q_TEMPLATE_INSTANCES", "Q_TEMPLATE_INSTANCES_VERSION",
           "QueryRefused", "digest", "mixed_population_program", "partition", "population_program", "program_sha512", "ranges",
           "template_instance_query", "template_instances_query", "units"]
