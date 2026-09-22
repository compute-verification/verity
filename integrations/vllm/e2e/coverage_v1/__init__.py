"""``coverage_v1`` -- proof coverage of every selected equation (PROTOCOL rows R9, R18).

Two things live here:

* :mod:`e2e.coverage_v1.coverage` -- ``expected_statements`` / ``check_coverage``: the verifier derives,
  from the fixed circuit, the partition and the actual sample ``Q``, the expected statement of every
  selected VU, and requires exactly one accepted proof covering each (R9).  The number of proofs proves
  nothing; a duplicate never hides an omission; an empty ``Q`` is legal and covered.
* :mod:`e2e.coverage_v1.kind_table` -- the SMALL kind table with, per kind, whether an obligation builder
  exists, which guest evaluates the specified relation and under which gate set, the leaf framings it can
  authenticate, and the status under the three approval scenarios; plus the unsupported-check exposure
  ``1 - (1 - s)^n`` (R18).  ``python -m e2e.coverage_v1.kind_table --expected-work <D4 expected_work.json>
  --census <D2 census_small.json> --out <dir>`` writes ``coverage.json`` + ``coverage.md``.
* :mod:`e2e.coverage_v1.elem_builder` -- fallback obligation builder for the elementwise kinds the approved guest
  already evaluates (``SiluMulBf16_v1``, ``Bf16AddF2fp_v1`` under ``veritor.elem-bf16@1``): raw format-3 statements
  over the record's packed rows, A2-shaped expectations, and the raw-statement -> ``ObligationView`` reader.
  (Transferred to CK-ELEM for dispatch/integration, integrator 20:40Z.)
* :mod:`e2e.coverage_v1.builders` -- the builder registry ``BUILDERS[kind].build(located_unit, record) -> Built`` A3's
  ``selective_prover`` dispatches through; :mod:`e2e.coverage_v1.interior_layout` -- the eight dominant per-primitive
  attention kinds, registered as refusers that name the uncommitted attention-interior leaf (PROPOSED ``attn-interior/v0``);
  :mod:`e2e.coverage_v1.attn_head_builder` -- the whole-head relation as a coarser VU (S55, PROPOSED partition change,
  flag ``COVERAGE_V1_HEAD_VU``): fa2-family statement over committed q/K/V/out rows, chained blocks, verifier-side
  head view recovery that refuses a broken chain.
"""

from .coverage import (  # noqa: F401
    CoverageReport,
    ExpectedStatement,
    ObligationView,
    check_coverage,
    expected_statements,
    statement_digest,
)

__all__ = [
    "CoverageReport",
    "ExpectedStatement",
    "ObligationView",
    "check_coverage",
    "expected_statements",
    "statement_digest",
]
