"""The first campaign subcircuit, frozen: exactly which function the proof backends are asked to prove.  A :class:`Target` is
the code's name for a subcircuit (``gemm_coordinate(K, semantics)`` with every parameter bound; README Glossary).

Every proof-system experiment in the campaign proves *this* relation until the target is deliberately re-versioned.
Another engineer must be able to read this module and state the function without asking anyone; the test
``archive/sp1/tests/proofs/test_target.py`` pins every identity below against the recorded conformance vectors, so a
silent change to the semantics, the family manifest or the program shape fails the suite.

The function
============

``GemmCoordinate<K>`` in the gate family ``veritor.tc-ampere-bf16@1`` (:func:`verity_sp1.proofs.programs.gemm_coordinate`):
one output coordinate ``y[j]`` of a BF16 GEMM, ``K`` a positive multiple of 16.

* **Sources.** ``2K`` incoming 16-bit BF16 words: ``x[0..K]`` (the activation row) then ``W[j, 0..K]`` (one weight
  row), each in ascending ``k`` order.  Weights are incoming values of the subcircuit; the plan decides which of
  them are sampled/authenticated.
* **Instruction semantics.** Each step is one accumulator coordinate of Ampere
  ``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`` as modelled by :data:`verity.ml.circuits.silicon.silicon.AMPERE_BF16_M16N8K16`:
  the sixteen exact BF16 x BF16 products are summed in **two groups of eight** (``k = 0..7``, then ``8..15``); the
  incoming FP32 accumulator joins the first group and the first group's result joins the second; each grouped sum is
  a **25-bit truncating** fixed-point adder aligned at the group's maximum exponent, with the exponent **floor**
  ``-132`` (terms below it flush); the result is repacked as an FP32 word (round-toward-zero through the 25-bit
  window).  :func:`verity.ml.circuits.silicon.silicon.tc_dot` is the executable definition; the Rust ``tc.rs`` and the Lean
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
* **Saturation is inside the relation (log entry D8).** A finite input whose grouped sum leaves the FP32 range
  yields ``+-inf`` at once, sticky through the remaining group -- the hardware's word on every measured case.  That
  word is a valid output of a transition and, through the epilogue, of the VU; it is not a finite accumulator, so a
  chain whose intermediate transition saturates is rejected at the next step.  The pinned simulator masks the
  exponent instead and is not the reference there.
* **Statement and binding.** An obligation is a ``typed-obligation/v0`` object (``statement_format = 2``) whose
  family digest is :data:`FAMILY_DIGEST`; boundary words are authenticated as packed words against the object's
  Merkle roots (``verity_sp1.proofs.packed``, ``verity.ml.commitments.leaves``).  The guest's public values are
  ``h_O || h_L || verdict`` and are accepted only under the approved verifying key (``verity_sp1.host.APPROVED``).

Transition unit versus verification unit
========================================

Two objects, deliberately distinct (closed 2026-09-21):

* The **backend transition unit** is one 16-product update with an *arbitrary* finite incoming accumulator ``c``:
  ``tc_dot16(c, x[16s:16s+16], W[16s:16s+16])`` -- two grouped sums, ``c`` joining the first.  A numerical proof
  backend may take, expose and check these transitions and their intermediate accumulators internally in any
  layout it likes; those states are proof-internal auxiliary witness, never new runtime boundaries.
* The **Verity verification unit** (what the outer sampler selects and what a proof must establish in full) is
  ``GemmCoordinate<K>``: the chain of ``K/16`` transitions from ``c_0 = +0`` followed by the epilogue cast.

Backends must agree on the profile, the input/output bitstrings, the supported domain, the exact
function and the VU endpoints; they need not agree on auxiliary witness layouts.

References and the log of disagreements
=======================================

* :data:`HAWKEYE_PIN` pins the source-close reference (Hawkeye's simulator, commit ``30703fcb``); the
  parameters above are the ones it recovers on the A100, re-measured bit-exactly on both the RTX 4090 (sm_89) and
  the A100 (sm_80) over 25,698,304 finite elements each (12,065,535 with a nonzero accumulator; runs
  ``r20260921-231618-ea24`` and ``r20260921-232750-56c5``, log entries D2 and D9) and on the 7,713 all-finite
  structured A100 cases.
* :data:`DISCREPANCY_LOG` records every place where the paper, the simulator, Verity's reference and the hardware
  disagree or are silent, with an owner and what would resolve it.  Nothing in it is resolved by editing a
  reference implementation.

Reference inputs
================

Recorded under ``fixtures/typed-obligation-v0`` with the public values a correct prover produced:

* :data:`K16_VECTOR` -- ``K = 16``, one step, the smallest input; the SP1 baseline proof.
* :data:`K1536_VECTOR` -- ``K = 1536``, 96 chained steps, one real coordinate of a captured vLLM ``qkv_proj`` decode
  step; the campaign's realistic size.

Explicitly unresolved (not part of the target)
==============================================

* **Hardware NaN word.** The production epilogue's ``cvt.rn.bf16.f32`` produces ``0x7FFF`` for NaN, not ``0x7FC0``;
  that is family ``veritor.tc-ampere-bf16@2`` (vector ``accept--tc-store-cast-nan-hw``).  The two agree on every
  non-NaN word, and NaN is outside the finite domain anyway, so the choice is immaterial for this target.
* **Total semantics.** :mod:`verity.ml.circuits.silicon.total` (and the guest's ``tc_total.rs``) define the step on every
  encoding -- NaN, infinities, saturation -- as measured on A100/RTX 4090; no registered gate family binds them
  (``veritor.tensor-core.ampere_bf16_m16n8k16@3`` is the intended name).  Declaring the total relation is a
  protocol decision, not a backend optimisation.
* **Hopper.** ``wgmma`` on the H100 is one group of sixteen, a 26-bit adder, floor ``-133``
  (:data:`verity.ml.circuits.silicon.silicon.HOPPER_BF16_M16N8K16`); a different pipeline, a different target.
* **Kernel-level accumulation order.** The target fixes the chain the *relation* checks (ascending ``k``, one
  accumulator).  Whether a given GEMM kernel launch (split-K, stream-K, different tile orders) realises exactly this
  chain is a frontend/integration binding question; the K=1536 vector is one captured case where it did.
* **Other dtypes.** FP8 (E4M3) pipelines exist in ``silicon`` and now have *draft* targets below
  (:data:`FP8_HOPPER_WGMMA`, :data:`FP8_ADA_MMA`); nothing about them is frozen.

Draft targets (silicon Phase 0, 2026-09-22)
===========================================

A :class:`Target` carries a ``status``: ``"frozen"`` (:data:`FIRST`: every backend proves it, it changes only by
re-versioning) or ``"draft"`` (a relation stated precisely enough to generate vectors against, owned by the silicon
campaign, not yet bound to a guest or a registered family digest).  Each target names its ``instruction`` in
:data:`verity.ml.circuits.silicon.instructions.INSTRUCTIONS`; the instruction's status (pinned / hypothesis / unprobed) says how
much hardware stands behind the model the target computes with.

* :data:`FP8_HOPPER_WGMMA` -- ``sm90.wgmma.m64n8k32.e4m3`` (pinned: veritor H100, 144,543,744 words), model
  :data:`verity.ml.circuits.silicon.models.HOPPER_E4M3_K32` (one group of 32, 14-bit truncating adder, floor -139).  Program
  ``GemmAccumulator<K>`` (:func:`verity_sp1.proofs.programs.gemm_accumulator_fp8`), ``K`` a multiple of 32:
  ``tc_dot32_0`` then ``K/32 - 1`` chained ``tc_dot32`` in ascending ``k``, the FP32 accumulator word **public**
  (the bare dot).  Domain: finite E4M3 operands (every byte but ``0x7F``/``0xFF``; 448 is finite) and finite FP32
  accumulators at every step.  The scale-and-``cvt.rn.satfinite.e4m3x2`` epilogue of a production FP8 GEMM is a
  **separate, later relation** -- it composes with this one but is not part of it.  Note the lossy step: the
  incoming accumulator keeps only its top 14 significand bits, even through a group of zero products (D10).
* :data:`FP8_ADA_MMA` -- ``sm89.mma.m16n8k32.e4m3`` (pinned: veritor RTX 4090, 24,898,304 elements), model
  :data:`verity.ml.circuits.silicon.models.ADA_E4M3_M16N8K32` (two groups of 16, the same 14-bit adder and floor).  Same program,
  same domain, same absent epilogue; a different grouping and so a different function.
* :data:`FP8_SM120_MMA` (lane vllm-sm120-tc-gemm, 2026-09-30) -- ``sm120.mma.m16n8k32.e4m3`` (pinned: RTX PRO 6000, its
  conformance device, 25,041,024 elements / 0 mismatches; one QMMA.16832 per tile), model
  :data:`verity.ml.circuits.silicon.models.BLACKWELL_SM120_E4M3_M16N8K32`: one group of 32 in
  the 26-bit adder of the sm_120 BF16 pipeline, so the accumulator is not truncated.  Same program, domain and absent
  epilogue as the other two; the registry's ``sm120.mma.m16n8k32.e4m3`` is this pipeline, and trust anchors it here.
* The families are the cast-free ``veritor.tensor-core.<pipeline>@2`` sets of :func:`verity_sp1.proofs.families.make_tensor_core_gate_set`
  (``tc_dot32`` / ``tc_dot32_0`` with 8-bit operand arguments); their digests are recorded here but **no guest is
  bound to them** and no ``typed-obligation/v0`` lowering has been produced -- the reference vectors are model
  replays (``verity/ml/circuits/silicon/captures/vectors/``, schema ``verity/tc-reference-vector/v0``), not authenticated obligations.
* :data:`BF16_HOPPER` (lane b-hopper, 2026-09-22) -- ``sm90.mma.m16n8k16.bf16`` (pinned: Hawkeye H100, re-measured
  25.7M elements / 0 mismatches; the ``wgmma`` BF16 path runs the same pipeline), model
  :data:`verity.ml.circuits.silicon.models.HOPPER_BF16_M16N8K16` (one group of 16, the 26-bit adder, floor -133; exact BF16
  products).  FIRST's program and epilogue -- ``GemmCoordinate<K>`` with ``f32_to_bf16`` -- so the Hopper row of the
  table is the same function as the Ampere row up to the step; the family is
  ``veritor.tc-bf16.hopper_bf16_m16n8k16@1`` (:func:`verity_sp1.proofs.families.make_tc_bf16_gate_set`), FIRST's
  vocabulary with the steps on the Hopper pipeline.  Native peak: the H100 SXM5 dense BF16 rate (989.4 TFLOPS).
* :data:`NVFP4_SM120` -- ``sm120.mma.m16n8k64.e2m1.nvf4`` (pinned: RTX 5090, lanes fp4-re / fp4-proof), model
  :data:`verity.ml.circuits.silicon.models.BLACKWELL_SM120_NVF4` (``BlockScaledAlignAdd``: exact E2M1 x E2M1 group dots over four
  groups of 16, one UE4M3 scale byte per 16-element block on A and on B, the five-way align-add anchored on the leading
  participating group's scale exponent (27 fraction bits), the accumulator's 36-bit window, truncation toward zero,
  every zero result +0).  Program ``GemmAccumulator<K>`` (:func:`verity_sp1.proofs.programs.gemm_accumulator_nvfp4`),
  ``K`` a multiple of 64 (K = 1536 is 24 steps = 96 blocks): ``tc_dot64_0`` then chained ``tc_dot64`` in ascending
  ``k``, the FP32 accumulator word public.  Domain: every E2M1 nibble, UE4M3 scale bytes with the padding bit clear
  and not the NaN code ``0x7F``, finite FP32 accumulators at every step (D20: NaN / inf accumulators and sums leaving
  binary32 are outside; an NVFP4 chain from ``c_0 = 0`` cannot leave binary32).  Consumer die: the evidence is
  ``mma.sync kind::mxf4nvf4`` on the RTX 5090; B200 ``tcgen05.mma`` cannot be assumed.
* Not covered by any draft: ``sm90.mma.m16n8k32.e4m3`` (the Hopper legacy path -- unprobed; whether it is Hopper's
  one group of 32 or Ada's two of 16 is the probe lane's first question), E5M2 (hypotheses only), and the MXFP4
  instruction ``sm120.mma.m16n8k64.e2m1.mxf4`` (pinned; the same adder with UE8M0 scales over blocks of 32, no target).

Native peak: what a target's overhead is measured against
=========================================================

A target names its ``anchor_device``, the **conformance device of its instruction** -- the card the relation was pinned
on (A100 for ``sm80.mma.m16n8k16.bf16``, H100 for ``sm90.wgmma.m64n8k32.e4m3``, RTX 4090 for ``sm89.mma.m16n8k32.e4m3``)
-- not the machine a prover happens to run on (that is in the result's ``hardware`` fingerprint).  The number a
benchmark's ``overhead.vs_native_peak`` divides by is outside core: the census entry (``benchmarks/census/hardware.json``, data read
by the benchmarks) of that device and the target's operand dtype, the datasheet **dense** tensor-core peak at the
relation's FP32 accumulate width.  One VU of ``GemmCoordinate<K>`` is ``K`` multiply-accumulates = ``2K`` FLOP, so the native time of one
VU at a peak is ``2K / flop_per_second``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from verity.ml.circuits.silicon.models import (ADA_E4M3_M16N8K32, BLACKWELL_SM120_E4M3_M16N8K32, BLACKWELL_SM120_NVF4, HOPPER_BF16_M16N8K16,
                                  HOPPER_E4M3_K32, BlockScaledAlignAdd)
from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16, Pipeline

__all__ = [
    "BF16_HOPPER",
    "DISCREPANCY_LOG",
    "FAMILY",
    "FAMILY_DIGEST",
    "FIRST",
    "FP8_ADA_MMA",
    "FP8_SM120_FAMILY",
    "FP8_SM120_FAMILY_DIGEST",
    "FP8_SM120_MMA",
    "FP8_HOPPER_WGMMA",
    "HAWKEYE_PIN",
    "K16_VECTOR",
    "K1536_VECTOR",
    "NVFP4_SM120",
    "TARGETS",
    "TC_AMPERE_BF16_NAME",
    "TC_AMPERE_BF16_VERSION",
    "TC_BF16_NAME_PREFIX",
    "TC_BF16_VERSION",
    "TC_VECTORS",
    "Target",
]

#: The tensor-core gate families' names and versions (the frozen SP1 backend's ``verity_sp1.proofs.families`` builds them).
TC_AMPERE_BF16_NAME = "veritor.tc-ampere-bf16"
#: ``veritor.tc-bf16.<pipeline>@1``: the same vocabulary on another BF16 pipeline.
TC_BF16_NAME_PREFIX = "veritor.tc-bf16."
TC_BF16_VERSION = "1"
#: The version whose ``f32_to_bf16`` canonicalises NaN to ``0x7FC0``.
TC_AMPERE_BF16_VERSION = "1"

#: Repository-relative paths of the pinned source-close reference and of the discrepancy log.
HAWKEYE_PIN = "verity/ml/circuits/silicon/hawkeye/PIN.json"
DISCREPANCY_LOG = "verity/ml/circuits/silicon/discrepancy_log.json"

#: The gate family every campaign obligation is lowered in.
FAMILY = f"{TC_AMPERE_BF16_NAME}@{TC_AMPERE_BF16_VERSION}"
#: Its manifest digest (``GateSet.digest``); the guest refuses a statement whose family digest differs.
FAMILY_DIGEST = "cfe960f86e786bbbea42af6dc382342ad26ff280590a071db743e21bda01d307"
#: Reference vectors (names under ``fixtures/typed-obligation-v0``).
K16_VECTOR = "accept--packed-word-row"
K1536_VECTOR = "stage2--qkv-proj-decode-0-k1536"

#: Where the draft targets' model-replay vectors live (names below are files ``<name>.json`` in this directory).
TC_VECTORS = "verity/ml/circuits/silicon/captures/vectors"
#: ``veritor.tensor-core.<pipeline>@2`` manifest digests of the FP8 draft families (``make_tensor_core_gate_set``).
FP8_HOPPER_FAMILY = "veritor.tensor-core.hopper_e4m3_wgmma_k32@2"
FP8_HOPPER_FAMILY_DIGEST = "6d90478ba12f63f1171355f343f65bce6665541f1f7d4cca711ed9ea2ae9f777"
FP8_ADA_FAMILY = "veritor.tensor-core.ada_e4m3_m16n8k32@2"
FP8_ADA_FAMILY_DIGEST = "2bcdad67108dd15195372308784247d6792a3335f1dc19345d91d8ad68a17d16"
#: ``veritor.tc-bf16.hopper_bf16_m16n8k16@1`` (``make_tc_bf16_gate_set(HOPPER_BF16_M16N8K16)``): FIRST's vocabulary
#: with the steps on the Hopper BF16 pipeline; the family of the Hopper BF16 draft target.
BF16_HOPPER_FAMILY = f"{TC_BF16_NAME_PREFIX}{HOPPER_BF16_M16N8K16.name}@{TC_BF16_VERSION}"
BF16_HOPPER_FAMILY_DIGEST = "1dba640b0e83ba6dc0d8f692ac0089032cc98bfd39b1a210dbaf14b76a4d5a46"
#: The NVFP4 draft family: ``tc_dot64`` / ``tc_dot64_0`` over one operand = 64 E2M1 codes + 4 UE4M3 scale bytes.
NVFP4_FAMILY = "veritor.tensor-core.blackwell_sm120_nvf4_m16n8k64@2"
NVFP4_FAMILY_DIGEST = "3a928dc85d5c6515b49cf53472844e34412708ebc9404efe086de253e7cbc271"
#: ``veritor.tensor-core.blackwell_sm120_e4m3_m16n8k32@2``: the sm_120 FP8 draft family (``make_tensor_core_gate_set``).
FP8_SM120_FAMILY = "veritor.tensor-core.blackwell_sm120_e4m3_m16n8k32@2"
FP8_SM120_FAMILY_DIGEST = "1b1554efe28ae26cc1fd2177d4ab2f8aa8f0185afc31d43885e5caf23560a93c"

Status = Literal["frozen", "draft"]


@dataclass(frozen=True, slots=True)
class Target:
    """One target (a subcircuit): the gate family, the pipeline the steps are computed on, the program, the reference sizes, the
    conformance device its overhead is measured against -- and its ``status`` (``frozen``: every backend proves it; ``draft``: stated
    and vectorised, bound to nothing yet) with the id of the instruction class it is a relation over
    (``verity.ml.circuits.silicon.instructions.INSTRUCTIONS``)."""

    name: str
    family: str
    family_digest: str
    pipeline: Pipeline | BlockScaledAlignAdd
    program: str
    transition_unit: str
    statement_format: int
    domain: str
    epilogue: str
    reference_vectors: tuple[str, ...]
    #: The conformance device of the instruction: the card the relation was pinned on (module docstring, "Native peak").
    anchor_device: str
    status: Status = "frozen"
    instruction: str | None = None
    #: The ``trust-dossier/v1`` artifact id whose computed trust level this target cites (``benchmarks/tc_probe/trust.py``).
    dossier: str | None = None




FIRST = Target(
    name="first-campaign-target/2026-09-21",
    family=FAMILY,
    family_digest=FAMILY_DIGEST,
    pipeline=AMPERE_BF16_M16N8K16,
    program="GemmCoordinate<K>: tc_dot16_0, then (K/16 - 1) x tc_dot16 chained in ascending k, then f32_to_bf16",
    transition_unit="tc_dot16(c, a[0:16], b[0:16]) with any finite FP32 c: the backend's unit; its states are proof-internal",
    statement_format=2,
    domain="finite BF16 operands and finite FP32 accumulators at every step; anything else is rejected, not computed",
    epilogue="f32_to_bf16: RN-even, overflow -> +-inf, NaN -> 0x7FC0 (software canonical word; @2 has the hardware 0x7FFF)",
    reference_vectors=(K16_VECTOR, K1536_VECTOR),
    anchor_device="NVIDIA A100 SXM4 80GB",
    status="frozen",
    instruction="sm80.mma.m16n8k16.bf16",
    dossier="art:997eefe82d733c30d689801236cd682a47844b52dfcfe1d4ea61d44d99c837cc",
)

BF16_HOPPER = Target(
    name="bf16-hopper-mma-draft/2026-09-22",
    family=BF16_HOPPER_FAMILY,
    family_digest=BF16_HOPPER_FAMILY_DIGEST,
    pipeline=HOPPER_BF16_M16N8K16,
    program="GemmCoordinate<K>: tc_dot16_0, then (K/16 - 1) x tc_dot16 chained in ascending k, then f32_to_bf16",
    transition_unit=(
        "tc_dot16(c, a[0:16], b[0:16]) with any finite FP32 c: one group of 16 (c joins it), the 26-bit adder, "
        "floor -133; exact BF16 products"
    ),
    statement_format=2,
    domain="finite BF16 operands and finite FP32 accumulators at every step; anything else is rejected, not computed",
    epilogue=(
        "f32_to_bf16: RN-even, overflow -> +-inf, NaN -> 0x7FC0 -- FIRST's cast.  On the finite domain it equals the "
        "hardware cvt.rn.bf16.f32 word bit for bit (the versions differ only on NaN, which the domain excludes); the "
        "Hopper BF16 store is the same cvt as Ampere's, so nothing pipeline-specific is pinned for the epilogue"
    ),
    reference_vectors=("bf16-hopper-mma-k16", "bf16-hopper-mma-k1536"),
    anchor_device="NVIDIA H100 SXM5 80GB",
    status="draft",
    instruction="sm90.mma.m16n8k16.bf16",
)

_FP8_DOMAIN = (
    "finite E4M3 operands (every byte but 0x7F/0xFF; 448 = 0x7E is finite, there are no infinities) and finite FP32 "
    "accumulators at every step; anything else is rejected, not computed"
)
_FP8_EPILOGUE = (
    "none: the FP32 accumulator word is the public output (the bare dot).  The production FP8 store -- scale by a "
    "per-tensor factor, then cvt.rn.satfinite.e4m3x2 -- is a SEPARATE relation, to be stated and pinned later; it "
    "composes with this target, it is not part of it"
)

FP8_HOPPER_WGMMA = Target(
    name="fp8-hopper-wgmma-draft/2026-09-22",
    family=FP8_HOPPER_FAMILY,
    family_digest=FP8_HOPPER_FAMILY_DIGEST,
    pipeline=HOPPER_E4M3_K32,
    program="GemmAccumulator<K> (K a multiple of 32): tc_dot32_0, then (K/32 - 1) x tc_dot32 chained in ascending k; no cast",
    transition_unit=(
        "tc_dot32(c, a[0:32], b[0:32]) with any finite FP32 c: one group of 32, the 14-bit truncating adder (the "
        "incoming accumulator keeps its top 14 significand bits), floor -139"
    ),
    statement_format=2,
    domain=_FP8_DOMAIN,
    epilogue=_FP8_EPILOGUE,
    reference_vectors=("fp8-hopper-wgmma-k32", "fp8-hopper-wgmma-k1536"),
    anchor_device="NVIDIA H100 SXM5 80GB",
    status="draft",
    instruction="sm90.wgmma.m64n8k32.e4m3",
    dossier="art:f5e282ad4fe3a7d8ab2777cced6a1b0e8692375529fd031f99d24b3c98035755",
)

FP8_ADA_MMA = Target(
    name="fp8-ada-mma-draft/2026-09-22",
    family=FP8_ADA_FAMILY,
    family_digest=FP8_ADA_FAMILY_DIGEST,
    pipeline=ADA_E4M3_M16N8K32,
    program="GemmAccumulator<K> (K a multiple of 32): tc_dot32_0, then (K/32 - 1) x tc_dot32 chained in ascending k; no cast",
    transition_unit=(
        "tc_dot32(c, a[0:32], b[0:32]) with any finite FP32 c: two groups of 16 (c joins the first, its result the "
        "second), the 14-bit truncating adder, floor -139"
    ),
    statement_format=2,
    domain=_FP8_DOMAIN,
    epilogue=_FP8_EPILOGUE,
    reference_vectors=("fp8-ada-mma-k32", "fp8-ada-mma-k1536"),
    anchor_device="NVIDIA GeForce RTX 4090",
    status="draft",
    instruction="sm89.mma.m16n8k32.e4m3",
    dossier="art:2acd0d37dc729e36570bffdee35e6445d4e095d9be9f39bcd11222c439e8197c",
)


_NVFP4_DOMAIN = (
    "every E2M1 nibble (all 16 codes are finite), UE4M3 scale bytes with the padding bit clear and other than the NaN "
    "code 0x7F (PTX ISA: bit 7 set is invalid), and finite FP32 accumulators at every step; a chain from c_0 = 0 cannot "
    "leave binary32 (D20: NaN / inf accumulators and overflowing sums are outside the model, recorded, never computed)"
)
_NVFP4_EPILOGUE = (
    "none: the FP32 accumulator word is the public output (the bare block-scaled dot).  The production NVFP4 store -- "
    "a per-tensor scale, then quantisation to E2M1 with per-block UE4M3 scales -- is a SEPARATE relation; it composes "
    "with this target, it is not part of it"
)

NVFP4_SM120 = Target(
    name="nvfp4-sm120-mma-draft/2026-09-22",
    family=NVFP4_FAMILY,
    family_digest=NVFP4_FAMILY_DIGEST,
    pipeline=BLACKWELL_SM120_NVF4,
    program="GemmAccumulator<K> (K a multiple of 64): tc_dot64_0, then (K/64 - 1) x tc_dot64 chained in ascending k; no cast",
    transition_unit=(
        "tc_dot64(c, a codes[0:64] + a scales[0:4], b codes[0:64] + b scales[0:4]) with any finite FP32 c: four exact "
        "E2M1 x E2M1 group dots of 16, each scaled losslessly by its two UE4M3 block scales, then one five-way align-add "
        "with the accumulator on the grid lsb = max(max participating scale exponent - 27, E_acc - 35, -174), truncation "
        "toward zero, RZ normalisation to binary32, every zero result +0 (BlockScaledAlignAdd, D16-D20)"
    ),
    statement_format=2,
    domain=_NVFP4_DOMAIN,
    epilogue=_NVFP4_EPILOGUE,
    reference_vectors=("nvfp4-sm120-mma-k64", "nvfp4-sm120-mma-k1536"),
    anchor_device="NVIDIA GeForce RTX 5090",
    status="draft",
    instruction="sm120.mma.m16n8k64.e2m1.nvf4",
)

# The conformance device is the RTX PRO 6000, where the step was measured, not the arch's default anchor (the RTX 5090): the
# vLLM sm_120 target runs on the PRO 6000, and no FP8 capture exists on the 5090.
FP8_SM120_MMA = Target(
    name="fp8-sm120-mma-draft/2026-09-30",
    family=FP8_SM120_FAMILY,
    family_digest=FP8_SM120_FAMILY_DIGEST,
    pipeline=BLACKWELL_SM120_E4M3_M16N8K32,
    program="GemmAccumulator<K> (K a multiple of 32): tc_dot32_0, then (K/32 - 1) x tc_dot32 chained in ascending k; no cast",
    transition_unit=(
        "tc_dot32(c, a[0:32], b[0:32]) with any finite FP32 c: one group of 32 (c joins it), the 26-bit adder of the sm_120 BF16 "
        "pipeline (the accumulator is not truncated), floor -133 (unobservable with FP8 operands)"
    ),
    statement_format=2,
    domain=_FP8_DOMAIN,
    epilogue=_FP8_EPILOGUE,
    reference_vectors=("fp8-sm120-mma-k32", "fp8-sm120-mma-k1536"),
    anchor_device="NVIDIA RTX PRO 6000 Blackwell Server Edition",
    status="draft",
    instruction="sm120.mma.m16n8k32.e4m3",
    dossier="art:b3074aba4c9ddeef7f87454bfc23e73a5a3c5f73f1676260545115fe8589cc89",
)

#: Every target by name; ``verity/ml/circuits/silicon/discrepancy_log.json`` entries name one of these (or an instruction id).
TARGETS: dict[str, Target] = {t.name: t for t in (FIRST, BF16_HOPPER, FP8_HOPPER_WGMMA, FP8_ADA_MMA, NVFP4_SM120, FP8_SM120_MMA)}
