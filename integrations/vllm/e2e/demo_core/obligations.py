"""The thin adapter from ``DemoCore_v1`` VUs to the REAL approved statement builder.

``coverage_v1.ck_elem`` is the registered builder of the two kinds (``SiluMulBf16_v1|batch``, ``Bf16AddF2fp_v1|batch``, family
``veritor.elem-bf16@1``, guest ``9fd00e60``).  Its ``resolve()`` locates a VU's operand/output words through the Serve_v1 capture identity
(``ElemLocator``); that step — and only that step — is replaced here by this circuit's own gate -> (tree, leaf, word) map (``CoreRecord.position``).
Everything downstream is the shipped code unchanged: ``ck_elem.kind_subcircuit`` (the one-gate canonical body whose digest the verifier pins),
``ck_elem.build_many`` (typed object ``typed-obligation/v0``, ``validate``, format-3 statement bytes, packed witness with the authenticated
openings, the composed-record JSON ``rec_p1_prove.py`` consumes) and ``ck_elem.expectation``-shaped ``ExpectedObligation``s for
``coverage_v1.coverage.expected_statements`` / ``check_coverage``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from e2e.coverage_v1 import ck_elem
from e2e.coverage_v1.ck_elem import CommittedWord, Refusal, ResolvedUnit
from veritor.interface.expected import ExpectedObligation

from .circuit import VU, Partition
from .record import INTERIOR_TREE_PREFIX, CoreRecord, request_id

__all__ = ["resolve_vu", "expectation_of", "build_chunk", "chunk_units"]


def resolve_vu(vu: VU, rec: CoreRecord) -> ResolvedUnit:
    """The VU as one approved gate over committed words (the builder's ``ResolvedUnit``); refuses when a word is not committed."""
    spec = ck_elem.KINDS.get(vu.kind)
    if spec is None:
        raise Refusal(f"{vu.kind}: not a kind ck_elem knows")
    if len(vu.operands) != spec.arity:
        raise Refusal(f"{vu.kind}: gate {vu.gate} has {len(vu.operands)} operands, gate set arity {spec.arity}")
    words = []
    for og in vu.operands:
        p = rec.position(og)
        if p is None:
            raise Refusal(f"vu {vu.vu_index}: operand gate {og} is not committed in any tree (step-3 value of an RU not retained?)")
        tree, leaf, flat = p
        words.append(CommittedWord(tree, leaf, rec.leaves[leaf].shape, flat, spec.in_width))
    p = rec.position(vu.gate)
    if p is None:
        raise Refusal(f"vu {vu.vu_index}: output gate {vu.gate} is not committed in any tree")
    tree, leaf, flat = p
    out = CommittedWord(tree, leaf, rec.leaves[leaf].shape, flat, spec.out_width)
    # owner of the output's tree: ``ru/<req>/<n>`` -> n, ``interior/<req>/<step>`` -> step (E7's owner_of spelling), boundary -> 0 (its RU's step)
    ru = int(out.tree.split("/")[2]) if out.tree.startswith((INTERIOR_TREE_PREFIX + "/", "interior/")) else 0
    return ResolvedUnit(vu.vu_index, vu.kind, spec, vu.gate, f"stage{vu.ru_index}", request_id(vu.ru_index), tuple(words), out, ru)


def expectation_of(vu: VU, rec: CoreRecord) -> ExpectedObligation:
    """The verifier-shaped expected statement of a VU: operand positions, output position, the kind's canonical body digest (same construction
    as ``ck_elem.expectation``; positions from the circuit + committed id lists, never from a prover object)."""
    u = resolve_vu(vu, rec)
    sub = ck_elem.kind_subcircuit(u.spec)
    return ExpectedObligation(tuple(w.expected() for w in u.inputs), (u.output.expected(),), sub.digest, (None,), {}, f"vu={u.vu_index} {u.kind} {u.target}")


def chunk_units(vus: Sequence[VU], rec: CoreRecord, *, max_units: int = 64) -> list[list[ResolvedUnit]]:
    """Group resolved units by family into chunks of <= ``max_units`` (one proof object per chunk)."""
    by_family: dict[str, list[ResolvedUnit]] = {}
    for vu in vus:
        u = resolve_vu(vu, rec)
        by_family.setdefault(u.spec.family, []).append(u)
    chunks: list[list[ResolvedUnit]] = []
    for fam in sorted(by_family):
        units = sorted(by_family[fam], key=lambda u: u.vu_index)
        for k in range(0, len(units), max_units):
            chunks.append(units[k:k + max_units])
    return chunks


def build_chunk(units: Sequence[ResolvedUnit], rec: CoreRecord, *, name: str) -> ck_elem.Built:
    """One typed statement over ``units`` through the shipped builder (openings authenticated against ``commitments.json``): ``ck_elem.build_many``
    for the ``ru/`` spelling; for E2's ``interior/<req>/<step>`` trees E7's program-agnostic ``elem_builders.build_batch`` (owner = step) when it is
    on this branch, else the refusal by name that ``ck_elem`` gives (``no owner mapping``)."""
    if rec.interior == "e2":
        try:
            from e2e.coverage_v1.elem_builders import build_batch  # E7 (D3)
        except ImportError:
            build_batch = None
        if build_batch is not None:
            return build_batch(list(units), rec, name=name, authenticate=True)
    return ck_elem.build_many(list(units), rec, name=name, authenticate=True)


def built_summary(b: ck_elem.Built) -> dict[str, Any]:
    return {"family": b.family, "units": len(b.units), "vu_indices": [u.vu_index for u in b.units], "kinds": sorted({u.kind for u in b.units}),
            "object_digest": b.record["digest"], "statement_digest": b.record["statement_digest"], "statement_v3_bytes": len(b.statement_bytes),
            "witness_bytes": len(b.witness_bytes), "openings": len(b.openings), "trees": sorted({o.tree for o in b.openings.values()})}
