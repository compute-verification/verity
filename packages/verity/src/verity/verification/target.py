"""The first campaign target, frozen: exactly which function the proof backends are asked to prove.

Every proof-system experiment in the campaign proves *this* relation until the target is deliberately re-versioned.
Another engineer must be able to read this module and state the function without asking anyone; the test
``tests/verification/test_target.py`` pins every identity below against the recorded conformance vectors, so a
silent change to the semantics, the family manifest or the program shape fails the suite.

The function
============

``GemmCoordinate<K>`` in the gate family ``veritor.tc-ampere-bf16@1`` (:func:`verity.verification.programs.gemm_coordinate`):
one output coordinate ``y[j]`` of a BF16 GEMM, ``K`` a positive multiple of 16.

* **Sources.** ``2K`` incoming 16-bit BF16 words: ``x[0..K]`` (the activation row) then ``W[j, 0..K]`` (one weight
  row), each in ascending ``k`` order.  Weights are incoming values of the subcircuit; the plan decides which of
  them are sampled/authenticated.
* **Instruction semantics.** Each step is one accumulator coordinate of Ampere
  ``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`` as modelled by :data:`verity.ml.tc.silicon.AMPERE_BF16_M16N8K16`:
  the sixteen exact BF16 x BF16 products are summed in **two groups of eight** (``k = 0..7``, then ``8..15``); the
  incoming FP32 accumulator joins the first group and the first group's result joins the second; each grouped sum is
  a **25-bit truncating** fixed-point adder aligned at the group's maximum exponent, with the exponent **floor**
  ``-132`` (terms below it flush); the result is repacked as an FP32 word (round-toward-zero through the 25-bit
  window).  :func:`verity.ml.tc.silicon.tc_dot` is the executable definition; the Rust ``tc.rs`` and the Lean
  ``TCB.lean`` are its other columns.
* **Chaining and order.** Step ``0`` is ``tc_dot16_0(x[0:16], W[0:16])`` -- no incoming accumulator (``acc = +0``);
  step ``s`` for ``1 <= s < K/16`` is ``tc_dot16(acc_{s-1}, x[16s:16s+16], W[16s:16s+16])``.  The steps are chained
  **sequentially in ascending k**, one accumulator, no split-K, no re-association.
* **Epilogue.** The one outgoing value is ``f32_to_bf16(acc_{K/16-1})``: round to nearest, ties to even; finite
  words beyond BF16's range become ``+-inf``; NaN becomes the canonical ``0x7FC0`` (version ``@1``).
* **Supported domain -- finite only.** A NaN or infinite operand, or a NaN/infinite accumulator at any step, is
  *outside* the relation: the checker rejects the obligation (``E_...`` verdict ``0``), it does not compute a
  hardware answer.  This is the finite-domain relation of ``silicon``; the total semantics are a separate,
  unbound family (below).
* **Statement and binding.** An obligation is a ``typed-obligation/v0`` object (``statement_format = 2``) whose
  family digest is :data:`FAMILY_DIGEST`; boundary words are authenticated as packed words against the object's
  Merkle roots (``verity.verification.packed``, ``verity.commitments.leaves``).  The guest's public values are
  ``h_O || h_L || verdict`` and are accepted only under the approved verifying key (``verity_sp1.host.APPROVED``).

Reference instances
===================

Recorded under ``fixtures/typed-obligation-v0`` with the public values a correct prover produced:

* :data:`K16_VECTOR` -- ``K = 16``, one step, the smallest instance; the SP1 baseline proof.
* :data:`K1536_VECTOR` -- ``K = 1536``, 96 chained steps, one real coordinate of a captured vLLM ``qkv_proj`` decode
  step; the campaign's realistic size.

Explicitly unresolved (not part of the target)
==============================================

* **Hardware NaN word.** The production epilogue's ``cvt.rn.bf16.f32`` produces ``0x7FFF`` for NaN, not ``0x7FC0``;
  that is family ``veritor.tc-ampere-bf16@2`` (vector ``accept--tc-store-cast-nan-hw``).  The two agree on every
  non-NaN word, and NaN is outside the finite domain anyway, so the choice is immaterial for this target.
* **Total semantics.** :mod:`verity.ml.tc.total` (and the guest's ``tc_total.rs``) define the step on every
  encoding -- NaN, infinities, saturation -- as measured on A100/RTX 4090; no registered gate family binds them
  (``veritor.tensor-core.ampere_bf16_m16n8k16@3`` is the intended name).  Declaring the total relation is a
  protocol decision, not a backend optimisation.
* **Hopper.** ``wgmma`` on the H100 is one group of sixteen, a 26-bit adder, floor ``-133``
  (:data:`verity.ml.tc.silicon.HOPPER_BF16_M16N8K16`); a different pipeline, a different target.
* **Kernel-level accumulation order.** The target fixes the chain the *relation* checks (ascending ``k``, one
  accumulator).  Whether a given GEMM kernel launch (split-K, stream-K, different tile orders) realises exactly this
  chain is a frontend/integration binding question; the K=1536 vector is one captured case where it did.
* **Other dtypes.** FP8 (E4M3) pipelines exist in ``silicon`` but have no gate family.
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.ml.tc.silicon import AMPERE_BF16_M16N8K16, Pipeline

from .families import TC_AMPERE_BF16_NAME, TC_AMPERE_BF16_VERSION

__all__ = ["FAMILY", "FAMILY_DIGEST", "FIRST", "K16_VECTOR", "K1536_VECTOR", "Target"]

#: The gate family every campaign obligation is lowered in.
FAMILY = f"{TC_AMPERE_BF16_NAME}@{TC_AMPERE_BF16_VERSION}"
#: Its manifest digest (``GateSet.digest``); the guest refuses a statement whose family digest differs.
FAMILY_DIGEST = "cfe960f86e786bbbea42af6dc382342ad26ff280590a071db743e21bda01d307"
#: Reference vectors (names under ``fixtures/typed-obligation-v0``).
K16_VECTOR = "accept--packed-word-row"
K1536_VECTOR = "stage2--qkv-proj-decode-0-k1536"


@dataclass(frozen=True, slots=True)
class Target:
    """One frozen target: the family, the pipeline the steps are computed on, the program, and the reference sizes."""

    name: str
    family: str
    family_digest: str
    pipeline: Pipeline
    program: str
    statement_format: int
    domain: str
    epilogue: str
    reference_vectors: tuple[str, ...]


FIRST = Target(
    name="first-campaign-target/2026-09-21",
    family=FAMILY,
    family_digest=FAMILY_DIGEST,
    pipeline=AMPERE_BF16_M16N8K16,
    program="GemmCoordinate<K>: tc_dot16_0, then (K/16 - 1) x tc_dot16 chained in ascending k, then f32_to_bf16",
    statement_format=2,
    domain="finite BF16 operands and finite FP32 accumulators at every step; anything else is rejected, not computed",
    epilogue="f32_to_bf16: RN-even, overflow -> +-inf, NaN -> 0x7FC0 (software canonical word; @2 has the hardware 0x7FFF)",
    reference_vectors=(K16_VECTOR, K1536_VECTOR),
)
