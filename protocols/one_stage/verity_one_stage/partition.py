"""The partition as a named query over a program, P = Q(C), and the demo's query `Q_template_instance` v0.

    {"program": "<SHA-512 of the program>", "query": {"name": "Q_template_instance", "version": 0,
                                                      "template": "<Definition id>", "width_rule": null}}

* **Digest.** `SHA-512("verity/partition/v1\\0" || canon({program, query}))`, canon = `verity.ir.codec.canonical_json`.
* **Program.** SHA-512 over the canonical descriptor `codec.program_digest` hashes with SHA-256 (annotations excluded).
* **`Q_template_instance` v0.** Every instance of one template's Definition in the program's root body is one unit, in
  canonical order (node order, then batch member). The `Input` gates are the input unit: always checked, never drawn.
  No width rule is applied, so a unit is a whole instance however wide its outputs. The query refuses a program whose
  root body computes anything but instances of its template.

The verifier holds its own copy of the program and applies the query itself (`units`); it never takes a population or a
unit index on the prover's word.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

from verity.ir.codec import canonical_json, encode_program
from verity.ir.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.ir.program import Program
from verity.ir.types import Array

FORMAT = "verity/partition/v1"
Q_TEMPLATE_INSTANCE = "Q_template_instance"
Q_TEMPLATE_INSTANCE_VERSION = 0


class QueryRefused(ValueError):
    """The query does not apply to the program."""


def template_instance_query(template_id: str) -> dict[str, Any]:
    return {"name": Q_TEMPLATE_INSTANCE, "version": Q_TEMPLATE_INSTANCE_VERSION, "template": template_id,
            "width_rule": None}


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


def units(program: Program, query: Mapping[str, Any]) -> int:
    """The population of `query` applied to `program`: the number of units, indexed 0.. in canonical order."""
    if (query.get("name"), query.get("version")) != (Q_TEMPLATE_INSTANCE, Q_TEMPLATE_INSTANCE_VERSION):
        raise QueryRefused(f"unknown query {query.get('name')!r} v{query.get('version')!r}")
    if query.get("width_rule") is not None:
        raise QueryRefused("Q_template_instance v0 applies no width rule")
    template = query.get("template")
    population = 0
    for k, node in enumerate(program.root.body.nodes):
        if getattr(node.fn, "family", None) == "Input":
            continue
        if node.fn.id != template or node.form not in ("call", "batch"):
            raise QueryRefused(f"root node {k} computes {node.fn.id!r} ({node.form}), not an instance of {template!r}")
        population += node.n if node.form == "batch" else 1
    if population == 0:
        raise QueryRefused(f"the program has no instance of {template!r}")
    return population


__all__ = ["FORMAT", "Q_TEMPLATE_INSTANCE", "Q_TEMPLATE_INSTANCE_VERSION", "QueryRefused", "digest", "partition",
           "population_program", "program_sha512", "template_instance_query", "units"]
