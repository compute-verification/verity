"""The instruction registry: which tensor-core instruction is modelled by which :class:`~verity.ml.circuits.silicon.models.Model`,
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

from .integer import BLACKWELL_SM120_S8_M16N8K32
from .models import (
    ADA_BF16_M16N8K16,
    ADA_E4M3_M16N8K32,
    ADA_E5M2_M16N8K32,
    AMPERE_BF16_M16N8K16,
    BLACKWELL_SM100_MXF4,
    BLACKWELL_SM100_NVF4,
    BLACKWELL_SM120_E4M3_M16N8K32,
    BLACKWELL_SM120_E5M2_M16N8K32,
    BLACKWELL_SM120_MXF4,
    BLACKWELL_SM120_NVF4,
    HOPPER_BF16_M16N8K16,
    HOPPER_BF16_WGMMA_K16,
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
    #: The ``trust-dossier/v1`` artifact id (``art:...``) whose computed level this pin cites (``benchmarks/tc_probe/trust.py``).
    dossier: str | None = None
    #: The device trust anchors this instruction on when no Target wraps it and it was measured on a device other than the
    #: arch's default anchor (``trust.ANCHOR_SKU``); a Target's own ``anchor_device`` takes precedence.
    anchor_device: str | None = None

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
    "accumulators, 0 mismatches against silicon.tc_dot (verity/ml/circuits/silicon/discrepancy_log.json D2, D9)"
)
_ADA_RUN = "verity run r20260921-231618-ea24 (RTX 4090, sm_89): the same 25,698,304 elements, 0 mismatches (D2, D9)"
_GOLDEN_BF16 = "verity/ml/circuits/ops/tests/fixtures/golden/ada_bf16_m16n8k16.json (360 records)"
_GOLDEN_E4M3 = "verity/ml/circuits/ops/tests/fixtures/golden/ada_e4m3_m16n8k32.json"
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
    "verity run r20260922-182011-88f8 (lane fp4-5090; RTX 5090, our harness benchmarks/tc_probe_fp4, OMMA.SF UE4M3.4X, one MMA per tile): "
    "autoproof probe3 vectors replayed 1601/1601; 1,736,704 swept elements with +0 accumulators (263,781 hot-clip firings, full UE4M3 "
    "code sweep), 0 mismatches; probe2 unscaled 1400/1400 (run r20260922-182132-d6db)"
)
_RTX5090_FP4_RE = (
    "verity runs r20260922-215616-3c10 and r20260922-221716-65b9 (nvf4) and r20260922-215831-c3d8, r20260922-221824-8d99, "
    "r20260922-222148-6342 (mxf4) -- lane fp4-re, RTX 5090 cc 12.0, driver 580.178.04, nvcc 12.8.93 -arch=sm_120a, one "
    "mma.sync per output element with arbitrary finite FP32 accumulators: BlockScaledAlignAdd (scale-exponent anchor "
    "frac_bits 27, accumulator window 36 on the exponent field, grid floor 2^-174, every zero +0) reproduces 8,339,456 "
    "nvf4 and 7,389,184 mxf4 words (lane fp4-5090's captures replayed, fit runs, fresh-seed confirmation runs) with 0 "
    "mismatches; autoproof model5 is the recorded control (D16 -0 accumulators, D17 opposite-sign accumulators 37..42 "
    "binades below the leading exponent, D18 ue8m0 scales, D13 underflow sign, D19 subnormal-regime anchor and floor). "
    "Record-only family `specials`: NaN accumulators -> 0x7FFFFFFF, +-inf -> itself, sums leaving binary32 -> +-inf "
    "(ue8m0 only). Snapshot dataset fp4-evidence-v2."
)
_H100_FP8 = (
    "veritor docs/data/tc-fp8-2026-09-08 (lane QZ, H100 80GB HBM3, driver 580.126.09, triton 3.7.1): "
    "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 / SASS QGMMA.64x64x32.F32.E4M3.E4M3, 16 families, "
    "35,289 tiles x3 reps, 144,543,744 words, 0 mismatches against HOPPER_E4M3_K32; fit grid leaves groups=(32,) "
    "and width=14 unique, floor only bounded to [-152, -126] -- replica verity/ml/circuits/silicon/captures/veritor-tc-fp8-2026-09-08.json"
)
_H100_BF16 = (
    "veritor docs/data/tc-hopper-2026-09-07 (H100, driver 580.178.04): wgmma.mma_async.sync.aligned.m64n128k16."
    "f32.bf16.bf16 / HGMMA.64x128x16.F32.BF16, 307,281,920 words, 0 mismatches against groups=(16,), width 26, "
    "floor -133 (verity.ml.circuits.silicon.total.HOPPER_BF16_WGMMA_K16) -- the wgmma path, not mma.sync"
)
_H100_WGMMA_BF16 = (
    "lane wgmma-bf16 (2026-09-26), benchmarks/tc_probe on vy-wgmma-bf16-h100 (H100 80GB HBM3, driver 580.126.09, nvcc 12.4.131, -arch=sm_90a, "
    "GPU-d5e245b4-5429-c607-9efb-a3578c7334e2): one wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 per tile, SASS 1x "
    "HGMMA.64x8x16.F32.BF16, against the total semantics (specials asserted).  A from shared memory (SS) r20260926-020157-242a "
    "art:8f903385776fdb04bd3008120f44422b76fa4e2cbb5ad6533275a67548d7a6a0 and from registers (RS) r20260926-020210-aee8 "
    "art:f9508ae817053e57d93332518608430d7988a7bf2fdf0fc0d58bcf6695ec74d5: each 26,361,344 swept elements in 14 families, 2,876,928 "
    "elements of four back-to-back wgmma on one accumulator (K = 64: the chain of single steps), veritor's 44,046 records replayed, "
    "0 mismatches; the Ampere pipeline misses 3.15M.  Words identical to the first pair (r20260926-015601-b530, -015605-bc89; same "
    "seeds).  Canary r20260926-021005-456c art:4ca1294d1cf34183e57d638873ea6cbee75ee42054e88c6b105ac61e0f49b513 (26,624, 0)"
)
_H100_MMA_EQ_WGMMA = (
    "lane wgmma-bf16 r20260926-023321-ecb5 (a second H100 80GB HBM3, GPU-7e095246-1454-01f4-2b76-197d7ab9d18b, driver 580.126.09): "
    "mma.sync.m16n8k16 bf16, 25,833,472 swept elements, specials included, 0 mismatches against HOPPER_BF16_WGMMA_K16's total step; the "
    "specials words equal the wgmma runs' word for word; veritor's 44,046 wgmma-recorded words re-executed through mma.sync, 0 mismatches"
)
_AUTOPROOF = (
    "autoproof spec/SEMANTICS.lock.v1.json + probes/results/probe3_1783788526.json (RTX 5090 sm_120, driver "
    "570.195.03, SASS 166a24a0...): lockfile records 1,574,998/1,574,998 dot statements bit-exact under model5 "
    "(the JSON itself scores the clip-less model2 at 1,574,997/1,574,998, D12); the 1,601 recorded sample vectors "
    "replay 1,601/1,601 in verity (verity/ml/circuits/ops/tests/test_models.py) -- replicas verity/ml/circuits/silicon/captures/autoproof-{SEMANTICS.lock.v1,"
    "probe-vectors}.json"
)
_AUTOPROOF_UNSCALED = (
    "autoproof probes/results/probe2_1783788240.json: mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1"
    ".f32 (unscaled FP4), 13,082 dots, 0 mismatches, RZ at the K=116,544 parity boundary"
)
_B200_TCGEN05 = (
    "lane tcgen05 2026-09-23, NVIDIA B200 cc 10.0, driver 580.126.09, nvcc/ptxas 12.9.86 (-gencode arch=compute_100a,code=sm_100a; "
    "nvcc 12.9's plain -arch=sm_100a also emits compute_100 PTX, which refuses every tcgen05 mnemonic): benchmarks/tc_probe_fp4/"
    "tcgen05_fp4.cu issues ONE tcgen05.mma.cta_group::1 (M128 N8 K64) per tile from one thread, accumulator preloaded into "
    "tensor memory (tcgen05.st, enable_input_d = 1; all-+0 accumulators also with enable_input_d = 0), scale factors in TMEM, "
    "completion via tcgen05.commit -> mbarrier, readback tcgen05.ld; the 16 harness rows replicated into the 8 lane groups "
    "agreed word for word. The sm_120 model (scale anchor -27, window 36, floor 2^-174, zero -> +0, truncation) reproduced "
    "every gated word of the same 20 families, the K=128 chain and autoproof's probe3 vectors; the refuted alternatives "
    "(model5, signed underflow) miss on the B200 by the same counts as on the RTX 5090"
)
_B200_STAGE0 = (
    "stage 0 r20260923-001336-5a50: ptxas 12.9 refuses mma.sync kind::mxf4nvf4 / .block_scale / kind::f8f6f4 for sm_100a "
    "('mma with block scale not supported on .target sm_100a'; the same PTX assembles for sm_120a), so the consumer "
    "instruction does not exist on the B200; CUTLASS 3.9.2 example 72a (nvfp4 GEMM, sm_100a) lowers to UTCOMMA.2CTA.4X "
    "(cta_group::2) with UTMALDG/UTCCP/UTCBAR -- the probe's cta_group::1 UTCOMMA is the single-CTA form of the same opcode; "
    "cta_group::2 itself is not probed"
)

_PIN_FP4 = (
    "lane fp4-proof (2026-09-22): fresh-seed tc_probe_fp4 sweeps on a second RTX 5090 die (GPU-1257f022-5d0f-1ea9-7922-bc7bc7959672, "
    "driver 570.195.03, -arch=sm_120a; label device=), all 20 probe families plus fp4-re's D16/D17 tiles and specials: nvf4 "
    "r20260922-234714-0fc0 10,657,792 elements, 0 mismatches for BlockScaledAlignAdd (autoproof-model5 misses 192,196), "
    "mxf4 r20260922-235348-a405 9,658,368 elements, 0 mismatches; with fp4-re's runs the anchor totals 17.29M (nvf4) and 14.42M "
    "(mxf4) zero-mismatch elements, so P2 (>= 1e7) holds and the trust assessment computed PINNED for both.  This is "
    "mma.sync kind::mxf4nvf4 on a consumer die; B200 tcgen05.mma is a separate instruction (sm100.*, lane tcgen05)"
)
_PIN_BF16 = (
    "lane pin-bf16 (2026-09-22): full canonical family set -- randn/uniform/subnormal/cancellation/mixed/near-overflow/signed-zero, "
    "tiny_acc_zero_products, zero_acc_tiny_cancellation, zero_products_nonzero_acc, acc_only_lsb_walk, specials, adder_width_cancellation -- "
    "25,833,472 asserted elements per anchor, 0 mismatches, SASS 1x HMMA.16816.F32.BF16, GPU UUID recorded (label device=): A100 r20260922-231411-1d5e "
    "(sm80), RTX 4090 r20260922-231751-38ca (sm89, + golden and mma_special replays), H100 r20260922-231751-2442 (sm90), RTX 5090 r20260922-231751-6617 "
    "(sm120); canaries r20260922-231832-9980 / -232813-0d92 / -232442-2eba / -232443-9ce8 (3-25 s).  Silicon facts: every BF16 pipeline passes all "
    "32 accumulator bits through unchanged with zero products (D == C for 65,536/65,536); the adder-width residual survives at 2^-24 and is dropped "
    "at 2^-25 inside the pair's k-group of 8 and is never dropped in the other group on sm_80/sm_89 ((8,8), 25 bits), and survives at 2^-25 / is "
    "dropped at 2^-26 in every slot k=2..15 on sm_90/sm_120 ((16,), 26 bits)"
)
_PRO6000_FP8 = (
    "lane vllm-sm120-tc-gemm (2026-09-30), benchmarks/tc_probe as Kueue port-capture jobs on vy-nebius-1 (RTX PRO 6000 Blackwell Server "
    "Edition, cc 12.0, driver 580.173.02, SM clock locked at 2,100 MHz, CUDA 13.0 V13.0.88, -arch=sm_120a; compile check "
    "r20260930-063200-8640: one QMMA.16832 per tile kernel), one mma.sync per tile, seed 20260931, 14 families"
)
_PRO6000_E4M3 = (_PRO6000_FP8 + ": run r20260930-063457-5410 (GPU-83455c5b-f2a1-c810-0a97-a01ddf05c4ba, SASS QMMA.16832.F32.E4M3.E4M3), "
                 "25,041,024 elements, 0 mismatches against BLACKWELL_SM120_E4M3_M16N8K32; Ada's pipeline misses 19,033,876 and the sm_90 "
                 "two-HMMA lowering 3,049,017")
_PRO6000_E5M2 = (_PRO6000_FP8 + ": run r20260930-063527-4d6b (GPU-6fbd6701-ccfb-e008-dc6e-986175046bc7, SASS QMMA.16832.F32.E5M2.E5M2), "
                 "25,041,024 elements, 0 mismatches against BLACKWELL_SM120_E5M2_M16N8K32; Ada's pipeline with e5m2 products misses 15,747,373")
_PRO6000_E5M2_SECOND = ("a second PRO 6000 die: run r20260930-073650-c9c5 (GPU-5f1149a4-e5f5-1007-ec78-7cd350a69a4f), 6,440,960 elements, "
                        "0 mismatches")
_PRO6000_MXF4 = ("lane proofs-tc-defs (2026-10-01), benchmarks/tc_probe_fp4 run r20261001-000520-2440, a Kueue prover-dev job on vy-nebius-1 "
                 "(RTX PRO 6000 Blackwell Server Edition, GPU-a98a36f6-1393-7cad-32ea-e9f7ec25376a, driver 580.173.02, CUDA 13.0 V13.0.88 "
                 "-arch=sm_120a, SASS one OMMA.SF.16864.F32.E2M1.E2M1.E8 per tile kernel): 1,089,536 gated elements, 0 mismatches against "
                 "BLACKWELL_SM120_MXF4; its 90,112 record-only elements (a 0xFF scale byte, sums leaving binary32, NaN and infinite "
                 "accumulators: fixtures tc-sm120-mxf4-2026-10-01) are outside the model's domain")


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
            dossier="art:997eefe82d733c30d689801236cd682a47844b52dfcfe1d4ea61d44d99c837cc",
            evidence=("Hawkeye (A100) recovered groups (8,8), width 25, floor -132", _A100_RUN, _GOLDEN_BF16, _PIN_BF16),
        ),
        _i(
            id="sm89.mma.m16n8k16.bf16",
            sm="sm_89",
            ptx="mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
            shape=(16, 8, 16),
            a_dtype="bf16", b_dtype="bf16", acc_dtype="f32", scale=None,
            model=ADA_BF16_M16N8K16,
            status="pinned",
            dossier="art:e37fcb69bb1d7020cb1a81b71b525b76da2389b5bf3240f9668ed053637d34fe",
            evidence=(_ADA_4090, _ADA_RUN, _GOLDEN_BF16, _PIN_BF16),
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
            dossier="art:0664d5c367791927273cf038f0f6caa326a00801a3d563210c1969f159d3afff",
            evidence=(_H100_MMA_BF16, "Hawkeye (H100) recovered groups (16,), width 26, floor -133 -- instruction path not stated", _H100_BF16, _PIN_BF16,
                      _H100_MMA_EQ_WGMMA),
        ),
        # --- BF16 Hopper wgmma -----------------------------------------------------------------------------------
        _i(
            # The instruction vLLM's H100 BF16 GEMMs (cuBLAS, CUTLASS, Triton) and FA3 issue.  Same parameters as the mma.sync
            # path's model, a separate entry so that a subcircuit names the instruction production runs.
            id="sm90.wgmma.m64n8k16.bf16",
            sm="sm_90",
            ptx="wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16",
            shape=(64, 8, 16),
            a_dtype="bf16", b_dtype="bf16", acc_dtype="f32", scale=None,
            model=HOPPER_BF16_WGMMA_K16,
            status="pinned",  # benchmarks/tc_probe/trust.py computed PINNED (P1-P5, Ta) on 2026-09-26 before this entry was added
            dossier="art:50508ab77719ba3d93bd2985416eba0029116f313304a4632e67ad0dd28579d9",
            evidence=(_H100_WGMMA_BF16, _H100_BF16),
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
            dossier="art:2acd0d37dc729e36570bffdee35e6445d4e095d9be9f39bcd11222c439e8197c",
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
            dossier="art:9a5f8e71b80b72a70e5a2536dff85c169e3a592404c092f00c1cd976dd479dc8",
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
            dossier="art:f5e282ad4fe3a7d8ab2777cced6a1b0e8692375529fd031f99d24b3c98035755",
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
            dossier="art:27fbc67eed97901502f646218cf25d8fe4db2fe2d65d30eefd9efb77923b02a3",
            evidence=(_H100_WGMMA_E5M2, "model = sm90.wgmma.m64n8k32.e4m3's pipeline (groups (32,), width 14, floor -139) with e5m2 products"),
        ),
        # --- TF32 Hopper -----------------------------------------------------------------------------------------
        _i(
            # DeepSeek-V4's mHC prenorm GEMM (DeepGEMM 8b1392b9, sm90_tf32_hc_prenorm_gemm.cuh) issues CUTLASS's
            # MMA_64x32x8_F32TF32TF32_RS_TN: A from registers, B the FP32 weight from shared memory, read as TF32.
            id="sm90.wgmma.m64n8k8.tf32",
            sm="sm_90",
            ptx="wgmma.mma_async.sync.aligned.m64n8k8.f32.tf32.tf32",
            shape=(64, 8, 8),
            a_dtype="tf32", b_dtype="tf32", acc_dtype="f32", scale=None,
            model=None,
            status="unprobed",
            evidence=("r20261013-115825-580e on an H200 fitted HopperTf32WgmmaDot8_v1 alone: one group of 8 exact TF32 products, "
                      "a 26-bit adder and floor -133; B is rounded RNE from FP32 to TF32 (A was widened BF16 and exact in TF32). "
                      "The end-to-end mHC capture was NO-GO because the old Definition truncated B and used the wrong NaN word in "
                      "the separate squared sum; integrations/vllm/tests/properties/dsv4_mhc_capture_gpu.py reruns the gate",),
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
            dossier="art:b87e5a5b4dfaa5e4cfc501fa834df2c968e5ce3d53839660e37a42a2525218d2",
            evidence=(_RTX5090_BF16, _PIN_BF16),
        ),
        _i(
            # Measured on the RTX PRO 6000, the conformance device of target FP8_SM120_MMA, whose anchor trust uses for this id;
            # trust.py computed PINNED (P1-P5) on 2026-09-30 before this entry was added.
            id="sm120.mma.m16n8k32.e4m3",
            sm="sm_120",
            ptx="mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32",
            shape=(16, 8, 32),
            a_dtype="e4m3", b_dtype="e4m3", acc_dtype="f32", scale=None,
            model=BLACKWELL_SM120_E4M3_M16N8K32,
            status="pinned",
            dossier="art:b3074aba4c9ddeef7f87454bfc23e73a5a3c5f73f1676260545115fe8589cc89",
            evidence=(_PRO6000_E4M3,),
        ),
        _i(
            # No target wraps this id; it was measured on the RTX PRO 6000 only, and anchors there like the e4m3 target does.
            id="sm120.mma.m16n8k32.e5m2",
            sm="sm_120",
            ptx="mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32",
            shape=(16, 8, 32),
            a_dtype="e5m2", b_dtype="e5m2", acc_dtype="f32", scale=None,
            model=BLACKWELL_SM120_E5M2_M16N8K32,
            status="pinned",
            dossier="art:fc4bb8915dd22f22b3d0260a00a6d694750322852c5163ddb80a05948370a94f",
            anchor_device="NVIDIA RTX PRO 6000 Blackwell Server Edition",
            evidence=(_PRO6000_E5M2, _PRO6000_E5M2_SECOND,
                      "model = sm120.mma.m16n8k32.e4m3's pipeline (groups (32,), width 26, floor -133) with e5m2 products"),
        ),
        _i(
            id="sm120.mma.m16n8k64.e2m1.nvf4",
            sm="sm_120",
            ptx="mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3",
            shape=(16, 8, 64),
            a_dtype="e2m1", b_dtype="e2m1", acc_dtype="f32", scale="ue4m3",
            model=BLACKWELL_SM120_NVF4,
            # Model BlockScaledAlignAdd (lane fp4-re) reproduces every word of 8.34M sweeps; lane fp4-proof's fresh-seed sweep
            # (_PIN_FP4) supplied P2's missing 3.4M elements and benchmarks/tc_probe/trust.py computed PINNED (P1-P5) on 2026-09-23
            # before this flip (registry <= computed level, lane pin-bf16).  Re-published once lane fp4-proof's B-Ligero
            # relation supplied Tc (2,048 honest VUs / 3,131 negatives, art:098c184a...): still PINNED, Ta and Tc now PASS,
            # Td (hardware-in-the-loop replay of the frozen input set) open.  Previous: art:de42d6cad7c63a387d64535efc...
            status="pinned",
            dossier="art:3e02931df28cccd398de4219f9d977371a58cdc6e8addfb7ff0a6269e5481b0e",
            evidence=(_RTX5090_FP4_RE, _AUTOPROOF, _AUTOPROOF_UNSCALED, _RTX5090_NVF4, _PIN_FP4,
                      "model agrees for arbitrary finite FP32 accumulators (D12's zero-init limitation lifted; D16 and D17 resolved by "
                      "lane fp4-re); NaN/inf accumulators and sums leaving binary32 are outside the model's domain (recorded)"),
        ),
        _i(
            # autoproof did not probe UE8M0 scales; the mnemonic below is the one ptxas 12.8 accepts on sm_120a (kind::mxf4nvf4
            # with .scale_vec::2X and .ue8m0; lane fp4-5090) and its SASS is OMMA.SF.16864.F32.E2M1.E2M1.E8.
            id="sm120.mma.m16n8k64.e2m1.mxf4",
            sm="sm_120",
            ptx="mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0",
            shape=(16, 8, 64),
            a_dtype="e2m1", b_dtype="e2m1", acc_dtype="f32", scale="ue8m0",
            model=BLACKWELL_SM120_MXF4,
            # As nvf4: 0 mismatches over 7.39M words (fp4-re) + 9.66M (fp4-proof, _PIN_FP4); trust computed PINNED on 2026-09-23.
            status="pinned",
            dossier="art:d8d4d1b602c9fde61df218c315430535fe5181882ab6b26c2b95dd81dc1b5d92",
            evidence=(_RTX5090_FP4_RE, _PIN_FP4, _PRO6000_MXF4,
                      "the same adder as nvf4 with the UE8M0 scale 2^(code-127) written as mantissa 8 at 2^(code-130) (D18); a "
                      "nonzero sum below the subnormal grid is +0 (D13: BLACKWELL_SM120_MXF4_UNDERFLOW_SIGNED refuted); the "
                      "accumulator window anchors on the exponent field and the grid never drops below 2^-174 (D19)"),
        ),
        _i(
            # The int8 grid's instruction (lane compute-accounting, 2026-10-15).  Exact integer arithmetic, so the model has
            # no adder parameters to fit.  benchmarks/tc_probe/trust.py computed PINNED (P1-P5) from the store on 2026-10-15
            # before this flip (registry <= computed level).  Measured on one RTX PRO 6000 only, and anchors there.
            id="sm120.mma.m16n8k32.s8",
            sm="sm_120",
            ptx="mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32",
            shape=(16, 8, 32),
            a_dtype="s8", b_dtype="s8", acc_dtype="s32", scale=None,
            model=BLACKWELL_SM120_S8_M16N8K32,
            status="pinned",
            dossier="art:86f39a17b5a0c8a8c92e5e25dcdfbee95d6f65408130bc1db6c54d440dfd8474",
            anchor_device="NVIDIA RTX PRO 6000 Blackwell Server Edition",
            evidence=(BLACKWELL_SM120_S8_M16N8K32.validated,
                      "SASS IMMA.16832.S8.S8.SAT, one per tile kernel; 0 mismatches over 10,243,968 elements: "
                      "r20261015-084000-06d8 (tc-evidence art:67b817fc8b2237694c271a97110f6a9881644c4d7701cdd4abcc090b3d220148)",
                      "census r20261015-062626-f65b: IMMA.16832 at 1,023.6 MACs/SM/clk (QMMA.16832: 1,023.8), 16 SMSP cycles "
                      "per warp MMA, dependent latency 27.2 cycles"),
        ),
        # --- Blackwell datacenter (sm_100, tcgen05) ---------------------------------------------------------------
        _i(
            id="sm100.tcgen05.mma.e2m1.nvf4",
            sm="sm_100",
            ptx="tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [d_tmem], a_desc, b_desc, idesc(e2m1 x e2m1 -> f32, "
                "ue4m3, M128 N8 K64, K-major), [sfa_tmem], [sfb_tmem], enable_input_d",
            shape=(128, 8, 64),
            a_dtype="e2m1", b_dtype="e2m1", acc_dtype="f32", scale="ue4m3",
            model=BLACKWELL_SM100_NVF4,
            status="pinned",  # benchmarks/tc_probe/trust.py computed PINNED (P1-P5) from the store on 2026-09-23 before this flip
            evidence=(_B200_TCGEN05, _B200_STAGE0,
                      "SASS UTCOMMA.4X, one per tile kernel; 0 mismatches over 16,461,824 elements: r20260923-001528-f5c5 (303,104), "
                      "r20260923-001759-d869 (6,463,488 + probe3 1,601/1,601), r20260923-001856-7d63 (1,134,592, K=128 chain), "
                      "r20260923-002315-b793 (8,560,640); the sm_120 model's parameters carry over unchanged"),
        ),
        _i(
            # CUTLASS 3.9.2's SM100_MMA_MXF4_SS emits kind::mxf4 with scale_vec::2X for UE8M0 (and kind::mxf4nvf4 4X for UE4M3);
            # kind::mxf4nvf4 with scale_vec::2X and UE8M0 is also legal PTX and was measured as well: the same words.
            id="sm100.tcgen05.mma.e2m1.mxf4",
            sm="sm_100",
            ptx="tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc(e2m1 x e2m1 -> f32, "
                "ue8m0, M128 N8 K64, K-major), [sfa_tmem], [sfb_tmem], enable_input_d",
            shape=(128, 8, 64),
            a_dtype="e2m1", b_dtype="e2m1", acc_dtype="f32", scale="ue8m0",
            model=BLACKWELL_SM100_MXF4,
            status="pinned",  # benchmarks/tc_probe/trust.py computed PINNED (P1-P5) from the store on 2026-09-23 before this flip
            evidence=(_B200_TCGEN05, _B200_STAGE0,
                      "SASS UTCOMMA, one per tile kernel; 0 mismatches over 12,898,304 elements: kind::mxf4 r20260923-001838-42a4 "
                      "(3,956,736) and r20260923-002320-6fbf (3,956,736), kind::mxf4nvf4 2X r20260923-001832-17dc (3,956,736) and "
                      "r20260923-001903-c420 (1,028,096, K=128 chain); the signed-underflow alternative misses 14,710 / 14,384 / "
                      "14,734 / 5,535 (D13 holds on sm_100)"),
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
