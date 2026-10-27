"""A specialization's Definitions in the codec's v0 form (`encode_specialization`), a Definition's digest (`definition_digest`), and the
values of a varying static a row's graph is partitioned at (`word_values`): what a row's Program graph
(`verity_vllm.pipeline.program_graph`) and C-Flock's statements (`verity_flock.boolean_export`, `class_statement`) share.
"""
from __future__ import annotations

import hashlib
from typing import Any


def encode_specialization(fn) -> tuple[str, dict[str, dict[str, Any]]]:
    """(the specialization's id, every definition it uses in the codec's v0 form): `verity.ml.circuits.codec.encode_program` of a one-node
    Program that calls it, keeping the definitions reachable from it (not the wrapper, its lowered root or their Input gates)."""
    from verity.ml.circuits.codec import SCHEMA_V0, encode_program
    from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition
    from verity.ml.circuits.program import Program
    if isinstance(fn, PrimitiveDefinition):
        return fn.id, {fn.id: {"form": "primitive", "id": fn.id}}
    params, ret = fn.signature()
    shell = CompositeDefinition(f"Show[{fn.id}]", 1, (), lambda S: (tuple(params), ret), lambda B, S, *a: B.call(fn, *a), register=False)
    desc = encode_program(Program(shell), schema=SCHEMA_V0)
    defs = desc["definitions"]
    top = next(n["fn"] for n in defs[desc["root"]]["body"]["nodes"] if n["form"] != "primitive" and n["fn"] in defs
               and defs[n["fn"]].get("form") == "composite")
    keep, todo = {}, [top]
    while todo:
        k = todo.pop()
        if k in keep or k not in defs:
            continue
        keep[k] = defs[k]
        todo += [n["fn"] for n in (defs[k].get("body") or {}).get("nodes", [])]
    return top, keep


def definition_digest(d: dict[str, Any]) -> str:
    """sha256 of a definition's canonical JSON (`verity.ml.circuits.codec.canonical_json`), fold hints excluded."""
    from verity.ml.circuits.codec import canonical_json
    return hashlib.sha256(canonical_json({k: v for k, v in d.items() if k != "fold"})).hexdigest()


#: a varying group with more values than this is partitioned at fewer: with a key-block size BN (attention's T), at its first values and
#: every multiple of BN with the value before it and the three after it -- a head's unit count is affine in T between those, its gates
#: and committed words nearly so (the k16 PV steps; within 0.1 % of the exact count) -- else at WORD_SAMPLES values; the rest
#: are interpolated
WORD_EXACT = 320
WORD_SAMPLES = 12


def word_values(vals: list[int], block: int | None = None) -> list[int]:
    if len(vals) <= WORD_EXACT:
        return vals
    if block:
        return sorted(set(vals[:4]) | {vals[-1]} | {v for v in vals if v % block in (block - 1, 0, 1, 2, 3)})
    return sorted({vals[round(i * (len(vals) - 1) / (WORD_SAMPLES - 1))] for i in range(WORD_SAMPLES)})


__all__ = ["WORD_EXACT", "WORD_SAMPLES", "definition_digest"]
