"""The instruction registry: which tensor-core instruction is modelled by which :class:`~verity.ml.tc.models.Model`,
and how well the model is validated.

The ontology of the silicon campaign has three code objects and two store objects:

    Instruction   one PTX instruction class on one architecture (this module)        code
    Model         the executable accumulator-step semantics (``models.py``)          code
    Target        a frozen relation over one instruction (``verification.target``)   code
    Evidence      recorded hardware runs                                             store (``tc-evidence/v1``)
    Pin           the decision that evidence validates a model for an instruction    store

An :class:`Instruction` carries its ``status``:

    pinned       a model reproduces every recorded hardware word of this instruction (the evidence names the run)
    hypothesis   a model exists, chosen by analogy or by another instruction's evidence; nothing measured on this one
    unprobed     no model, or a model this repository will not vouch for; the probe lane goes first

``INSTRUCTIONS`` is keyed by stable ids other lanes use as strings (``sm90.wgmma.m64n8k32.e4m3``); the ids are
part of the interface and do not change when a status does.  ``ptx`` is recorded only where a source repository
recorded the mnemonic (``None`` otherwise -- never invented); ``shape`` is ``(m, n, k)`` of the *modelled*
instruction, which for the ``wgmma`` families is the N=8 member of the same instruction class as the measured
``n64`` one (an accumulator coordinate's semantics do not depend on N).  ``evidence`` entries are free-text
pointers for now (repository paths, veritor/autoproof records, run ids); they become store artifacts later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .models import (
    ADA_BF16_M16N8K16,
    ADA_E4M3_M16N8K32,
    ADA_E5M2_M16N8K32,
    AMPERE_BF16_M16N8K16,
    BLACKWELL_SM120_MXF4,
    BLACKWELL_SM120_NVF4,
    HOPPER_BF16_M16N8K16,
    HOPPER_E4M3_K32,
    HOPPER_E5M2_K32,
    Model,
)

__all__ = ["INSTRUCTIONS", "Instruction", "Status", "instructions_for"]

Status = Literal["pinned", "hypothesis", "unprobed"]


@dataclass(frozen=True, slots=True)
class Instruction:
    id: str
    sm: str
    ptx: str | None
    shape: tuple[int, int, int] | None
    a_dtype: str
    b_dtype: str
    acc_dtype: str
    scale: Literal["ue4m3", "ue8m0"] | None
    model: Model | None
    status: Status
    evidence: tuple[str, ...] = ()
    #: The ``trust-dossier/v1`` artifact id (``art:...``) whose computed level this pin cites (``verity.verification.trust``).
    dossier: str | None = None

    def __post_init__(self) -> None:
        if self.status == "pinned" and self.model is None:
            raise ValueError(f"{self.id}: a pinned instruction needs a model")
        if self.status == "hypothesis" and self.model is None:
            raise ValueError(f"{self.id}: a hypothesis needs the model it hypothesises")
        if self.model is not None and self.model.dtype != self.a_dtype:
            raise ValueError(f"{self.id}: model dtype {self.model.dtype} is not the operand dtype {self.a_dtype}")
        if self.shape is not None and self.model is not None and self.shape[2] != self.model.k:
            raise ValueError(f"{self.id}: shape k {self.shape[2]} is not the model's k {self.model.k}")


# --- evidence pointers (free text; store artifacts later) -------------------------------------------------------------

_ADA_4090 = (
    "veritor gpu/tensor-core-semantics: RTX 4090 (sm_89, driver 580.159.03), one mma.sync per record, "
    "24,898,304/24,898,304 elements bit-exact over 194,518 tiles (results/validate_{bf16,e4m3}_final.json)"
)
_A100_RUN = (
    "verity run r20260921-232750-56c5 (A100 80GB, sm_80): 25,698,304 finite elements, 12,065,535 with nonzero "
    "accumulators, 0 mismatches against silicon.tc_dot (fixtures/discrepancy_log.json D2, D9)"
)
_ADA_RUN = "verity run r20260921-231618-ea24 (RTX 4090, sm_89): the same 25,698,304 elements, 0 mismatches (D2, D9)"
_GOLDEN_BF16 = "packages/verity/tests/ml/fixtures/golden/ada_bf16_m16n8k16.json (360 records)"
_GOLDEN_E4M3 = "packages/verity/tests/ml/fixtures/golden/ada_e4m3_m16n8k32.json"
_H100_PROBE = "verity lane tc-probe-fp8 on vy-h100 (H100 80GB HBM3, cc 9.0, driver 570.211.01, nvcc 12.4.131), one PTX op per tile, seed 20260902"
_H100_WGMMA_E4M3 = (_H100_PROBE + ": run r20260922-183435-fd5d, wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 / 1x QGMMA.64x8x32.F32.E4M3.E4M3, "
                    "descriptor (LBO,SBO)=(128,256) K-major no-swizzle, 25,554,432 elements, 0 mismatches; adder-width residual survives at 2^-13 and "
                    "truncates at 2^-14 in all 30 slots k=2..31 (one group of 32, 14-bit adder); zero products: D == C & ~0x3ff for 262,144/262,144")
_H100_WGMMA_E5M2 = (_H100_PROBE + ": run r20260922-183837-940b, wgmma ... f32.e5m2.e5m2 / 1x QGMMA.64x8x32.F32.E5M2.E5M2, 25,554,432 elements, "
                    "0 mismatches against the e4m3 pipeline with exact e5m2 products")
_H100_MMA_BF16 = (_H100_PROBE + ": run r20260922-185339-e337, mma.sync.aligned.m16n8k16 bf16 / 1x HMMA.16816.F32.BF16, 25,698,304 elements, 0 mismatches "
                  "against HOPPER_BF16_M16N8K16; AMPERE_BF16_M16N8K16 fails 30,927 of a 192,000-element subset")
_H100_MMA_FP8 = (_H100_PROBE + ": runs r20260922-184111-671e (e4m3) and r20260922-184835-4b95 (e5m2), 25,040,128 elements each: ptxas 12.4 lowers "
                 "mma.sync.aligned.m16n8k32 FP8 on sm_90a to 12x F2FP unpack + 2x HMMA.16816.F32 (the BF16 K=16 pipeline over k=0,1 mod 4 and k=2,3 mod 4 "
                 "with zero accumulators) + IEEE FP32 FADD of the partials and of C; that lowering model gives 0 mismatches, HOPPER_E4M3_K32 fails 74%/62%, "
                 "ADA_E4M3_M16N8K32 fails 76%/63%; D == C for zero products (no 14-bit accumulator truncation)")
_ADA_E4M3_RUN = ("verity run r20260922-175246-0477 (lane tc-probe-fp8, RTX 4090): 25,040,128 elements + 141,824 FP8-family words, 0 mismatches against "
                 "ADA_E4M3_M16N8K32; adder-width probe: 14-bit / (16,16) signature measured directly; zero products: D == C & ~0x3ff (one signed-zero exception, "
                 "C=0x800001a9 -> +0)")
_ADA_E5M2_RUN = ("verity runs r20260922-175732-172a and r20260922-182641-e27d (lane tc-probe-fp8, RTX 4090): 25,040,128 + 141,824 elements, 0 mismatches "
                 "against the Ada e4m3 pipeline with exact e5m2 products; SASS 1x QMMA.16832.F32.E5M2.E5M2")
_RTX5090_BF16 = (
    "verity run r20260922-182911-1f10 (lane fp4-5090; RTX 5090 cc 12.0, driver 570.153.02, nvcc 12.8.93, -arch=sm_120a, "
    "SASS HMMA.16816.F32.BF16): 204,800/204,800 elements agree with HOPPER_BF16_M16N8K16, 48,569 disagree with the Ampere/Ada pipeline"
)
_RTX5090_NVF4 = (
    "verity run r20260922-182011-88f8 (lane fp4-5090; RTX 5090, our harness tools/tc_probe_fp4, OMMA.SF UE4M3.4X, one MMA per tile): "
    "autoproof probe3 vectors replayed 1601/1601; 1,736,704 swept elements with +0 accumulators (263,781 hot-clip firings, full UE4M3 "
    "code sweep), 0 mismatches; probe2 unscaled 1400/1400 (run r20260922-182132-d6db)"
)
_H100_FP8 = (
    "veritor docs/data/tc-fp8-2026-09-08 (lane QZ, H100 80GB HBM3, driver 580.126.09, triton 3.7.1): "
    "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 / SASS QGMMA.64x64x32.F32.E4M3.E4M3, 16 families, "
    "35,289 tiles x3 reps, 144,543,744 words, 0 mismatches against HOPPER_E4M3_K32; fit grid leaves groups=(32,) "
    "and width=14 unique, floor only bounded to [-152, -126] -- replica fixtures/tc/veritor-tc-fp8-2026-09-08.json"
)
_H100_BF16 = (
    "veritor docs/data/tc-hopper-2026-09-07 (H100, driver 580.178.04): wgmma.mma_async.sync.aligned.m64n128k16."
    "f32.bf16.bf16 / HGMMA.64x128x16.F32.BF16, 307,281,920 words, 0 mismatches against groups=(16,), width 26, "
    "floor -133 (verity.ml.tc.total.HOPPER_BF16_WGMMA_K16) -- the wgmma path, not mma.sync"
)
_AUTOPROOF = (
    "autoproof spec/SEMANTICS.lock.v1.json + probes/results/probe3_1783788526.json (RTX 5090 sm_120, driver "
    "570.195.03, SASS 166a24a0...): lockfile records 1,574,998/1,574,998 dot statements bit-exact under model5 "
    "(the JSON itself scores the clip-less model2 at 1,574,997/1,574,998, D12); the 1,601 recorded sample vectors "
    "replay 1,601/1,601 in verity (tests/ml/test_models.py) -- replicas fixtures/tc/autoproof-{SEMANTICS.lock.v1,"
    "probe-vectors}.json"
)
_AUTOPROOF_UNSCALED = (
    "autoproof probes/results/probe2_1783788240.json: mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1"
    ".f32 (unscaled FP4), 13,082 dots, 0 mismatches, RZ at the K=116,544 parity boundary"
)


def _i(**kw) -> Instruction:
    return Instruction(**kw)


INSTRUCTIONS: dict[str, Instruction] = {
    ins.id: ins
    for ins in (
        # --- BF16 mma.sync -------------------------------------------------------------------------------------
        _i(
            id="sm80.mma.m16n8k16.bf16",
            sm="sm_80",
            ptx="mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
            shape=(16, 8, 16),
            a_dtype="bf16", b_dtype="bf16", acc_dtype="f32", scale=None,
            model=AMPERE_BF16_M16N8K16,
            status="pinned",
            evidence=("Hawkeye (A100) recovered groups (8,8), width 25, floor -132", _A100_RUN, _GOLDEN_BF16),
        ),
        _i(
            id="sm89.mma.m16n8k16.bf16",
            sm="sm_89",
            ptx="mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
            shape=(16, 8, 16),
            a_dtype="bf16", b_dtype="bf16", acc_dtype="f32", scale=None,
            model=ADA_BF16_M16N8K16,
            status="pinned",
            evidence=(_ADA_4090, _ADA_RUN, _GOLDEN_BF16),
        ),
        _i(
            # veritor measured Hopper BF16 through wgmma (HOPPER_BF16_WGMMA_K16: one group of 16, 26 bits, floor
            # -133); Hawkeye's H100 model has the same parameters.  The legacy mma.sync path was measured on 2026-09-22
            # (lane tc-probe-fp8): same pipeline, 0/25.7M mismatches -- pinned.
            id="sm90.mma.m16n8k16.bf16",
            sm="sm_90",
            ptx="mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
            shape=(16, 8, 16),
            a_dtype="bf16", b_dtype="bf16", acc_dtype="f32", scale=None,
            model=HOPPER_BF16_M16N8K16,
            status="pinned",
            evidence=(_H100_MMA_BF16, "Hawkeye (H100) recovered groups (16,), width 26, floor -133 -- instruction path not stated", _H100_BF16),
        ),
        # --- FP8 Ada mma.sync ----------------------------------------------------------------------------------
        _i(
            id="sm89.mma.m16n8k32.e4m3",
            sm="sm_89",
            ptx="mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32",
            shape=(16, 8, 32),
            a_dtype="e4m3", b_dtype="e4m3", acc_dtype="f32", scale=None,
            model=ADA_E4M3_M16N8K32,
            status="pinned",
            evidence=(_ADA_4090, _ADA_E4M3_RUN, _GOLDEN_E4M3,
                      "Hawkeye's Hopper_fp8_simulator reproduces only 82.40% of the same tiles (D10)"),
        ),
        _i(
            id="sm89.mma.m16n8k32.e5m2",
            sm="sm_89",
            ptx="mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32",
            shape=(16, 8, 32),
            a_dtype="e5m2", b_dtype="e5m2", acc_dtype="f32", scale=None,
            model=ADA_E5M2_M16N8K32,
            status="pinned",
            evidence=(_ADA_E5M2_RUN, "model = sm89.mma.m16n8k32.e4m3's pipeline (groups (16,16), width 14, floor -139) with e5m2 products"),
        ),
        # --- FP8 Hopper ----------------------------------------------------------------------------------------
        _i(
            id="sm90.wgmma.m64n8k32.e4m3",
            sm="sm_90",
            ptx="wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3",
            shape=(64, 8, 32),
            a_dtype="e4m3", b_dtype="e4m3", acc_dtype="f32", scale=None,
            model=HOPPER_E4M3_K32,
            status="pinned",
            evidence=(_H100_WGMMA_E4M3, _H100_FP8, "Hawkeye (H100 FP8, zero accumulator only): groups (32,), width 14, floor -139"),
        ),
        _i(
            id="sm90.wgmma.m64n8k32.e5m2",
            sm="sm_90",
            ptx="wgmma.mma_async.sync.aligned.m64n8k32.f32.e5m2.e5m2",
            shape=(64, 8, 32),
            a_dtype="e5m2", b_dtype="e5m2", acc_dtype="f32", scale=None,
            model=HOPPER_E5M2_K32,
            status="pinned",
            evidence=(_H100_WGMMA_E5M2, "model = sm90.wgmma.m64n8k32.e4m3's pipeline (groups (32,), width 14, floor -139) with e5m2 products"),
        ),
        _i(
            # Measured 2026-09-22 (D14 resolved): on sm_90a this PTX op is NOT a tensor-core FP8 instruction.  ptxas lowers
            # it to two HMMA.16816.F32 (the BF16 K=16 pipeline, HOPPER_BF16_M16N8K16) over the two k-halves with zero
            # accumulators plus IEEE FP32 FADDs; that composite reproduces 25,040,128/25,040,128 words for e4m3 and e5m2.
            # `model` stays None because the composite is not a single GroupSum; expressing it as a Model is a follow-up,
            # and the status stays "unprobed" only in the registry's narrow sense (no single-pipeline model to pin).
            id="sm90.mma.m16n8k32.e4m3",
            sm="sm_90",
            ptx="mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32",
            shape=(16, 8, 32),
            a_dtype="e4m3", b_dtype="e4m3", acc_dtype="f32", scale=None,
            model=None,
            status="unprobed",
            evidence=(_H100_MMA_FP8, "not a pipeline: 2x HMMA.16816.F32 (HOPPER_BF16_M16N8K16) + FP32 FADD lowering (D14, resolved)"),
        ),
        # --- Blackwell consumer (sm_120) --------------------------------------------------------------------------
        _i(
            # lane fp4-5090, run r20260922-182911-1f10 (RTX 5090, cc 12.0, driver 570.153.02, -arch=sm_120a, SASS HMMA.16816):
            # 204,800/204,800 elements agree with the Hopper BF16 pipeline (one group of 16, 26 bits, floor -133); the Ampere/Ada
            # pipeline fails 48,569 of them.  The consumer Blackwell keeps Hopper's BF16 datapath for legacy mma.sync.
            id="sm120.mma.m16n8k16.bf16",
            sm="sm_120",
            ptx="mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
            shape=(16, 8, 16),
            a_dtype="bf16", b_dtype="bf16", acc_dtype="f32", scale=None,
            model=HOPPER_BF16_M16N8K16,
            status="pinned",
            evidence=(_RTX5090_BF16,),
        ),
        _i(
            id="sm120.mma.m16n8k64.e2m1.nvf4",
            sm="sm_120",
            ptx="mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3",
            shape=(16, 8, 64),
            a_dtype="e2m1", b_dtype="e2m1", acc_dtype="f32", scale="ue4m3",
            model=BLACKWELL_SM120_NVF4,
            status="pinned",
            evidence=(_AUTOPROOF, _AUTOPROOF_UNSCALED, _RTX5090_NVF4,
                      "pinned with a zero-initialised accumulator and chain-reachable accumulators only (D12); "
                      "-0 accumulators (D16) and opposite-sign accumulators at the 42-bit window edge (D17) disagree with the model"),
        ),
        _i(
            # autoproof did not probe UE8M0 scales; the PTX mnemonic of the mxf4 kind is not recorded in either
            # source repository (PTX ISA: kind::mxf4 with .scale_vec::2X and .ue8m0) so it is left None.
            id="sm120.mma.m16n8k64.e2m1.mxf4",
            sm="sm_120",
            ptx=None,
            shape=(16, 8, 64),
            a_dtype="e2m1", b_dtype="e2m1", acc_dtype="f32", scale="ue8m0",
            model=BLACKWELL_SM120_MXF4,
            status="hypothesis",
            evidence=("no measurement; hypothesis = the nvf4 five-way align-add with UE8M0 block-32 scales and no "
                      "hot-group clip (autoproof native.rs); alternative BLACKWELL_SM120_MXF4_UNDERFLOW_PLUS0 (D13)",),
        ),
        # --- Blackwell datacenter (sm_100, tcgen05) ---------------------------------------------------------------
        _i(
            id="sm100.tcgen05.mma.e2m1.nvf4",
            sm="sm_100",
            ptx=None, shape=None,
            a_dtype="e2m1", b_dtype="e2m1", acc_dtype="f32", scale="ue4m3",
            model=None, status="unprobed",
            evidence=("tcgen05.mma kind::mxf4nvf4 on B200 -- no measurement; sm_120's model is NOT assumed to carry over",),
        ),
        _i(
            id="sm100.tcgen05.mma.e2m1.mxf4",
            sm="sm_100",
            ptx=None, shape=None,
            a_dtype="e2m1", b_dtype="e2m1", acc_dtype="f32", scale="ue8m0",
            model=None, status="unprobed",
            evidence=("tcgen05.mma kind::mxf4 on B200 -- no measurement",),
        ),
        _i(
            id="sm100.tcgen05.mma.bf16",
            sm="sm_100",
            ptx=None, shape=None,
            a_dtype="bf16", b_dtype="bf16", acc_dtype="f32", scale=None,
            model=None, status="unprobed",
            evidence=("tcgen05.mma kind::f16 with bf16 operands on B200 -- no measurement",),
        ),
        _i(
            id="sm100.tcgen05.mma.e4m3",
            sm="sm_100",
            ptx=None, shape=None,
            a_dtype="e4m3", b_dtype="e4m3", acc_dtype="f32", scale=None,
            model=None, status="unprobed",
            evidence=("tcgen05.mma kind::f8f6f4 with e4m3 operands on B200 -- no measurement",),
        ),
    )
}


def instructions_for(model: Model) -> tuple[Instruction, ...]:
    """Every registered instruction ``model`` is the model of."""

    return tuple(ins for ins in INSTRUCTIONS.values() if ins.model is model)
