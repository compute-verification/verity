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
* **Saturation is inside the relation (log entry D8).** A finite input whose grouped sum leaves the FP32 range
  yields ``+-inf`` at once, sticky through the remaining group -- the hardware's word on every measured case.  That
  word is a valid output of a transition and, through the epilogue, of the VU; it is not a finite accumulator, so a
  chain whose intermediate transition saturates is rejected at the next step.  The pinned simulator masks the
  exponent instead and is not the reference there.
* **Statement and binding.** An obligation is a ``typed-obligation/v0`` object (``statement_format = 2``) whose
  family digest is :data:`FAMILY_DIGEST`; boundary words are authenticated as packed words against the object's
  Merkle roots (``verity.verification.packed``, ``verity.commitments.leaves``).  The guest's public values are
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

Candidate backends must agree on the profile, the input/output bitstrings, the supported domain, the exact
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
* **Other dtypes.** FP8 (E4M3) pipelines exist in ``silicon`` and now have *draft* targets below
  (:data:`FP8_HOPPER_WGMMA`, :data:`FP8_ADA_MMA`); nothing about them is frozen.

Draft targets (silicon Phase 0, 2026-09-22)
===========================================

A :class:`Target` carries a ``status``: ``"frozen"`` (:data:`FIRST`: every backend proves it, it changes only by
re-versioning) or ``"draft"`` (a relation stated precisely enough to generate vectors against, owned by the silicon
campaign, not yet bound to a guest or a registered family digest).  Each target names its ``instruction`` in
:data:`verity.ml.tc.instructions.INSTRUCTIONS`; the instruction's status (pinned / hypothesis / unprobed) says how
much hardware stands behind the model the target computes with.

* :data:`FP8_HOPPER_WGMMA` -- ``sm90.wgmma.m64n8k32.e4m3`` (pinned: veritor H100, 144,543,744 words), model
  :data:`verity.ml.tc.models.HOPPER_E4M3_K32` (one group of 32, 14-bit truncating adder, floor -139).  Program
  ``GemmAccumulator<K>`` (:func:`verity.verification.programs.gemm_accumulator_fp8`), ``K`` a multiple of 32:
  ``tc_dot32_0`` then ``K/32 - 1`` chained ``tc_dot32`` in ascending ``k``, the FP32 accumulator word **public**
  (the bare dot).  Domain: finite E4M3 operands (every byte but ``0x7F``/``0xFF``; 448 is finite) and finite FP32
  accumulators at every step.  The scale-and-``cvt.rn.satfinite.e4m3x2`` epilogue of a production FP8 GEMM is a
  **separate, later relation** -- it composes with this one but is not part of it.  Note the lossy step: the
  incoming accumulator keeps only its top 14 significand bits, even through a group of zero products (D10).
* :data:`FP8_ADA_MMA` -- ``sm89.mma.m16n8k32.e4m3`` (pinned: veritor RTX 4090, 24,898,304 elements), model
  :data:`verity.ml.tc.models.ADA_E4M3_M16N8K32` (two groups of 16, the same 14-bit adder and floor).  Same program,
  same domain, same absent epilogue; a different grouping and so a different function.
* The families are the cast-free ``veritor.tensor-core.<pipeline>@2`` sets of :func:`verity.verification.families.make_tensor_core_gate_set`
  (``tc_dot32`` / ``tc_dot32_0`` with 8-bit operand arguments); their digests are recorded here but **no guest is
  bound to them** and no ``typed-obligation/v0`` lowering has been produced -- the reference vectors are model
  replays (``fixtures/tc/vectors/``, schema ``verity/tc-reference-vector/v0``), not authenticated obligations.
* Not covered by any draft: ``sm90.mma.m16n8k32.e4m3`` (the Hopper legacy path -- unprobed; whether it is Hopper's
  one group of 32 or Ada's two of 16 is the probe lane's first question), E5M2 (hypotheses only), and the
  Blackwell block-scaled FP4 path (a model exists, :class:`verity.ml.tc.models.BlockScaledAlignAdd`, no target).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from verity.ml.tc.models import ADA_E4M3_M16N8K32, HOPPER_E4M3_K32
from verity.ml.tc.silicon import AMPERE_BF16_M16N8K16, Pipeline

from .families import TC_AMPERE_BF16_NAME, TC_AMPERE_BF16_VERSION

__all__ = [
    "DISCREPANCY_LOG",
    "FAMILY",
    "FAMILY_DIGEST",
    "FIRST",
    "FP8_ADA_MMA",
    "FP8_HOPPER_WGMMA",
    "HAWKEYE_PIN",
    "K16_VECTOR",
    "K1536_VECTOR",
    "TARGETS",
    "TC_VECTORS",
    "Target",
]

#: Repository-relative paths of the pinned source-close reference and of the discrepancy log.
HAWKEYE_PIN = "fixtures/hawkeye/PIN.json"
DISCREPANCY_LOG = "fixtures/discrepancy_log.json"

#: The gate family every campaign obligation is lowered in.
FAMILY = f"{TC_AMPERE_BF16_NAME}@{TC_AMPERE_BF16_VERSION}"
#: Its manifest digest (``GateSet.digest``); the guest refuses a statement whose family digest differs.
FAMILY_DIGEST = "cfe960f86e786bbbea42af6dc382342ad26ff280590a071db743e21bda01d307"
#: Reference vectors (names under ``fixtures/typed-obligation-v0``).
K16_VECTOR = "accept--packed-word-row"
K1536_VECTOR = "stage2--qkv-proj-decode-0-k1536"

#: Where the draft targets' model-replay vectors live (names below are files ``<name>.json`` in this directory).
TC_VECTORS = "fixtures/tc/vectors"
#: ``veritor.tensor-core.<pipeline>@2`` manifest digests of the FP8 draft families (``make_tensor_core_gate_set``).
FP8_HOPPER_FAMILY = "veritor.tensor-core.hopper_e4m3_wgmma_k32@2"
FP8_HOPPER_FAMILY_DIGEST = "6d90478ba12f63f1171355f343f65bce6665541f1f7d4cca711ed9ea2ae9f777"
FP8_ADA_FAMILY = "veritor.tensor-core.ada_e4m3_m16n8k32@2"
FP8_ADA_FAMILY_DIGEST = "2bcdad67108dd15195372308784247d6792a3335f1dc19345d91d8ad68a17d16"

Status = Literal["frozen", "draft"]


@dataclass(frozen=True, slots=True)
class Target:
    """One target: the family, the pipeline the steps are computed on, the program, the reference sizes -- and
    its ``status`` (``frozen``: every backend proves it; ``draft``: stated and vectorised, bound to nothing yet)
    with the id of the instruction class it is a relation over (``verity.ml.tc.instructions.INSTRUCTIONS``)."""

    name: str
    family: str
    family_digest: str
    pipeline: Pipeline
    program: str
    transition_unit: str
    statement_format: int
    domain: str
    epilogue: str
    reference_vectors: tuple[str, ...]
    status: Status = "frozen"
    instruction: str | None = None


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
    status="frozen",
    instruction="sm80.mma.m16n8k16.bf16",
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
    status="draft",
    instruction="sm90.wgmma.m64n8k32.e4m3",
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
    status="draft",
    instruction="sm89.mma.m16n8k32.e4m3",
)

#: Every target by name; ``fixtures/discrepancy_log.json`` entries name one of these (or an instruction id).
TARGETS: dict[str, Target] = {t.name: t for t in (FIRST, FP8_HOPPER_WGMMA, FP8_ADA_MMA)}
