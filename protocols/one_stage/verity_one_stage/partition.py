"""The partition as a named query over a program, P = Q(C), in `verity/partition/v1`'s one canonical form (PR #111), and the
queries `Q_template_instance` v0 and `Q_template_instances` v0.

    {"format": "verity/partition/v1", "program": "<SHA-512 of the program>",
     "query": {"name": "Q_template_instance", "version": 0, "params": {"template": "<Definition id>"}}}

* **Digest.** `SHA-512("verity/partition/v1\\0" || canon(object))`, canon = `verity.ir.codec.canonical_json`.
* **Program.** SHA-512 over the canonical descriptor `codec.program_digest` hashes with SHA-256 (annotations excluded).
* **`Q_template_instance` v0**, `params = {"template": <Definition id>}`. The root body's nodes other than its `Input` nodes
  must each be a `call` or a `batch` of that Definition; each call is one unit and each batch member one unit, numbered in
  canonical order (node order, then member). The `Input` gates are the input unit: always checked, never drawn. No width
  rule is applied, so a unit is a whole instance however wide its outputs. The query refuses any other program.
* **`Q_template_instances` v0**, `params = {"templates": [<Definition id>, ...]}` (distinct). The same over several
  templates: every node is an instance of one of them, every listed template has an instance, and the unit order is
  still canonical node order. `ranges` gives each root node's template and its units' first index and count.

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
    return {"name": Q_TEMPLATE_INSTANCE, "version": Q_TEMPLATE_INSTANCE_VERSION, "params": {"template": template_id}}


def template_instances_query(template_ids: Sequence[str]) -> dict[str, Any]:
    return {"name": Q_TEMPLATE_INSTANCES, "version": Q_TEMPLATE_INSTANCES_VERSION, "params": {"templates": list(template_ids)}}


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
    return {"format": FORMAT, "program": program_sha512(program), "query": dict(query)}


def digest(obj: Mapping[str, Any]) -> str:
    return hashlib.sha512(FORMAT.encode() + b"\0" + canonical_json(dict(obj))).hexdigest()


def ranges(program: Program, query: Mapping[str, Any]) -> list[tuple[str, int, int]]:
    """`query` applied to `program`: per root node computing units, (template id, first unit index, unit count), in
    canonical order."""
    if set(query) != {"name", "version", "params"} or not isinstance(query.get("params"), Mapping):
        raise QueryRefused("a query is exactly {name, version, params}")
    kind, params = (query["name"], query["version"]), query["params"]
    if kind == (Q_TEMPLATE_INSTANCE, Q_TEMPLATE_INSTANCE_VERSION) and set(params) == {"template"} and isinstance(params["template"], str):
        templates = [params["template"]]
    elif (kind == (Q_TEMPLATE_INSTANCES, Q_TEMPLATE_INSTANCES_VERSION) and set(params) == {"templates"}
          and isinstance(params["templates"], list) and all(isinstance(t, str) for t in params["templates"])):
        templates = list(params["templates"])
        if len(set(templates)) != len(templates):
            raise QueryRefused("Q_template_instances lists a template twice")
    else:
        raise QueryRefused(f"unknown query {query['name']!r} v{query['version']!r}, or not its parameters")
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
