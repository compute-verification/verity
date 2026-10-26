"""cuBLAS's bf16 GEMMs on cc 9.0 with 132 SMs: MoE router gates at (K, N) = (6144, 256) and (7168, 384) past the cuteDSL tier
(`torch.mm(x, W.T, out_dtype=float32)`, `GateLinear` tier 4); bf16-output linears at K/N = 6144/{19360,160} and
7168/{16160,64}; K = 7168 compressor `torch.mm(..., out_dtype=float32)` calls at N = 2048, 1024 and 512; and MLA absorbs
(`torch.bmm`) at (K, N) = (192, 512) and (512, 256).

    GemmF32Out_v1{K, N, DOT}        (x: Array<K, BF16>, w: Array<N, Array<K, BF16>>) -> Array<N, F32>
                                    `DotBf16_v2{K, DOT}` of x against each W row: the f32 accumulator is the output
    BatchedGemm_v1{B, K, N, DOT}    (x: Array<B, Array<K, BF16>>, w: Array<B, Array<N, Array<K, BF16>>>) -> Array<B, Array<N, BF16>>
                                    `Gemm_v2{K, N, DOT}` of member b's x against member b's W, ``w[b][n][k]`` the right operand at
                                    ``[b, k, n]``
    CublasSplitKGemmF32Out_v1{K, N, SPLIT_K, DOT}
                                    the default cuBLAS split-K hypothesis for f32 output
    CublasSplitKGemm_v1{K, N, SPLIT_K, DOT}
                                    the same split-K hypothesis, rounded once to bf16 after reduction
    CublasSplitKGemmBf16Partials_v1{K, N, SPLIT_K, DOT}
                                    cc 12.0's split-K with a bf16 workspace: each chunk's partial rounded to bf16, widened and
                                    reduced as above, rounded once more to bf16
    CublasSerialSplitKGemm_v1{K, N, SPLIT_K, DOT}
                                    cc 12.0's serial split-K: the same chunks in order, each chain added to the previous
                                    chunk's bf16 output and rounded to bf16

on words, and each one's v2 the same function on bits (its ``word`` the v1, so `boolean.boolean_version` finds it): `GemmF32Out_v2`
takes the K/16 steps of every W row at once, each step one batch over the rows, and `BatchedGemm_v2` is `GemmF32Out_v2` of each
member, then `F2fpBf16_v2` of each word.  Their only `DOT` on bits is `HopperBF16WgmmaDot16_v2`, cut into `HOPPER_STEP`'s four
element Definitions (`chain_step`): the first step's from the +0 accumulator, its 32 constant bits folded into the gates, and a
later step's; and each of those in two, the gates that read only the x block, once per block, and the rest, once per row.  So no
row computes a value another computes, nor a step a value its accumulator's constant bits fix.  Each of the four is bound to the
k16 step's silicon model (`census.silicon_step`), so census/v1 counts their gates as the step's, `tc:bf16`.  `BatchedGemm_v2` is the function
of `mla.MlaBmm_v1` (vLLM's batch-invariant Triton `bmm_kernel` on sm_120), which names no word; `tests/program/test_bf16_gemm.py`
checks that they agree.  `bf16_gemm_check` holds the numpy models, `kernels.batched_gemm_rows` and `kernels.cublas_gemm_rows` the
sampled replay's row kernels.

The two split-K Definitions model cuBLAS 12.9.1's cc 9.0, 132-SM route: split K into equal 64-element-aligned chunks (the final chunk holds
the remainder); run each chunk's ascending `HopperBF16WgmmaDot16_v1` chain from +0; store f32 partials; and have the separate
reduction kernel add them serially in increasing split order in f32, rounding only the final result for a bf16 output.  The device
capture records cuBLAS's version, each selected kernel and launch grid, pins `SPLIT_K` by (M,K,N), and fits crafted cancellation
rows against partition, reduction-order and partial-type alternatives.  On cc 12.0 with 188 SMs the same cuBLAS picks, at some
(M, K, N), a `splitKreduce_kernel` whose workspace is bf16: `CublasSplitKGemmBf16Partials` is that route, the same chunks and chains
with each partial rounded to bf16 before the reduction widens it; at others it launches a cutlass kernel alone, whose grid.z
slices meet in its epilogue through the bf16 output: `CublasSerialSplitKGemm`.
"""
from __future__ import annotations

from typing import Any, Callable, NamedTuple, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import const_bits
from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.census import silicon_step
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops import fp32 as F
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops.boolean import cast as BCAST
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.refs import array_of
from verity.ml.circuits.types import Array, Type

__all__ = ["BatchedGemm", "BatchedGemmWord", "ChainStep", "CublasSerialSplitKGemm", "CublasSerialSplitKGemmWord",
           "CublasSplitKGemm", "CublasSplitKGemmBf16Partials",
           "CublasSplitKGemmBf16PartialsWord", "CublasSplitKGemmF32Out", "CublasSplitKGemmF32OutWord", "CublasSplitKGemmWord",
           "GemmF32Out", "GemmF32OutWord", "HOPPER_STEP",
           "HopperBf16Dot16Step", "HopperBf16Dot16StepFirst", "HopperBf16Dot16StepFirstX", "HopperBf16Dot16StepX", "chain_step",
           "conformance_cc12", "dot_step", "split_bounds"]

BF16, F32 = Array(16, BL.BIT), Array(32, BL.BIT)
CONFORMANCE = ("exact-model-tested: every output word at K/N statics 6144/{19360,160,256}, 7168/{16160,64,384,2048,1024,512}, "
               "and batched (B 8) 192/512 and 512/256, at the captured M (1, 4, 16, 128, 256; the gates' cuBLAS tier 17, 32, 128, "
               "256), on cc 9.0 with 132 SMs (torch 2.13.0+cu129 on its libcublas.so.12), under vLLM's served default (split counts "
               "2 to 5 each fitting only the stated hypothesis; the absorbs never split) and the split-K-off variant: property "
               "glm53_bf16_gemm_capture, run r20261016-234133-bbc8, record c0e09902b0a2057a, "
               "art:5fcc5a4937ac75ceec31bd22abc11194ec0c55d27e77ffd3690b1ad8976feeaf "
               "(tests/properties/glm53_bf16_gemm_capture_gpu.py)")
CC12_PROBES = "runs r20261018-193726-ac32, r20261018-195453-7c31 and r20261018-200645-a453"


def conformance_cc12(launch: str) -> str:
    """The cc 12.0 routes' conformance: the probes that read cuBLAS's words where it makes ``launch``."""
    return (f"exact-model-tested: every probed output word of torch.mm on cc 12.0 with 188 SMs (torch 2.13.0+cu129 on its "
            f"libcublas.so.12) at the served linears' (M, K, N) where cuBLAS launches {launch}, on spread, cancelling and class-edge "
            f"words (specials, any exponent, subnormal products), fitting only this hypothesis among its alternatives: {CC12_PROBES}")


CONFORMANCE_CC12 = conformance_cc12("`splitKreduce_kernel` with a bf16 workspace (M 1 to 303, split counts 2 to 16)")
CONFORMANCE_CC12_SERIAL = conformance_cc12("a cutlass kernel whose grid.z slices meet through the bf16 output, no reduction kernel "
                                           "(M 303, K 12288, 5 slices)")


class ChainStep(NamedTuple):
    """One step of an f32 accumulator chain on bits, ``(acc, x, w) -> acc'``, as four element Definitions: ``first_x`` and
    ``first`` the chain's first step, from the +0 accumulator, ``x`` and ``step`` a later one, each cut by `EW.shared_operand`
    at the gates that read only x (``first_x``, ``x``) and the rest."""

    first_x: CompositeDefinition
    first: CompositeDefinition
    x: CompositeDefinition
    step: CompositeDefinition

    def __call__(self, B, acc, x, w, rows: bool = False):
        """The step on ``acc`` (None: the chain's first), x's own gates computed here; with ``rows``, over W's rows: ``w`` and
        ``acc`` hold one member per row, and x and its gates are shared."""
        if acc is None:
            xd = B.call(self.first_x, x)
            return B.batch(self.first, x, w, xd, axes=(None, 0, None)) if rows else B.call(self.first, x, w, xd)
        xd = B.call(self.x, x)
        return B.batch(self.step, acc, x, w, xd, axes=(0, None, 0, None)) if rows else B.call(self.step, acc, x, w, xd)


def chain_step(name: str, build: Callable, params: Sequence[tuple[str, Type]], docs: tuple[str, str]) -> ChainStep:
    """The `ChainStep` of ``build(C, *elements)`` over ``params`` = (acc, x, w): ``<name>FirstX_v1`` and ``<name>First_v1``
    trace it with acc the 32 constant-0 bits, ``<name>X_v1`` and ``<name>_v1`` as it is.  ``docs``: what the step computes, and
    what x is."""
    what, x = docs
    zero = const_bits(0, 32)
    first = EW.shared_operand(
        (f"{name}FirstX", f"{name}First"), 1, lambda C, *els: build(C, zero, *els), params[1:], F32, 0,
        (f"the gates of {what} from acc = +0 that read only {x}, once per {x}", f"{what} from acc = +0, from {x}'s own gates"))
    later = EW.shared_operand(
        (f"{name}X", name), 1, build, params, F32, 1,
        (f"the gates of {what} that read only {x}, once per {x}", f"{what}, from {x}'s own gates"))
    return ChainStep(*first, *later)


_A16 = Array(16, BF16)
HOPPER_STEP = chain_step("HopperBf16Dot16Step", BG.HopperBF16WgmmaDot16.build, (("acc", F32), ("a", _A16), ("b", _A16)),
                         ("HopperBF16WgmmaDot16_v1(acc, a, b)", "a"))
HopperBf16Dot16StepFirstX, HopperBf16Dot16StepFirst, HopperBf16Dot16StepX, HopperBf16Dot16Step = HOPPER_STEP
for _part in HOPPER_STEP:
    silicon_step(_part, BG.HopperBF16WgmmaDot16.silicon)
_DOT_STEPS = {BG.HopperBF16WgmmaDot16.id: HOPPER_STEP}


def dot_step(name: str, DOT: Any) -> ChainStep:
    """The `ChainStep` of the k16 step ``DOT`` on bits; ``name`` refuses any other."""
    if getattr(DOT, "id", None) not in _DOT_STEPS:
        raise ValueError(f"{name}: unsupported statics (DOT {getattr(DOT, 'id', DOT)} on bits; only {', '.join(_DOT_STEPS)})")
    return _DOT_STEPS[DOT.id]


def _gemm_f32_type(X, T):
    return lambda S: ((("x", Array(S.K, X)), ("w", Array(S.N, Array(S.K, X)))), Array(S.N, T))


def _batched_type(X):
    return lambda S: ((("x", Array(S.B, Array(S.K, X))), ("w", Array(S.B, Array(S.N, Array(S.K, X))))), Array(S.B, Array(S.N, X)))


def _gemm_f32(dot):
    def body(B, S, x, w):
        return B.batch(bind(dot, K=S.K, DOT=S.DOT), x, w, axes=(None, 0))
    return body


def _batched(gemm):
    def body(B, S, x, w):
        return B.batch(bind(gemm, K=S.K, N=S.N, DOT=S.DOT), x, w, axes=(0, 0))
    return body


def _gemm_f32_bits(B, S, x, w):
    step = dot_step("GemmF32Out", S.DOT)
    if S.K <= 0 or S.K % 16:
        raise ValueError("GemmF32Out: unsupported statics (K_multiple_of_16)")
    blocks = w.reshape(Array(S.N, Array(S.K // 16, _A16)))
    acc = None
    for s in range(S.K // 16):
        acc = step(B, acc, x[16 * s:16 * s + 16], blocks.take(1, s), rows=True)
    return acc


def _batched_bits(B, S, x, w):
    acc = B.batch(bind(GemmF32Out, K=S.K, N=S.N, DOT=S.DOT), x, w, axes=(0, 0))
    out = B.batch(BG.F2fpBf16, acc.reshape(Array(S.B * S.N, F32)), axes=(0,))
    return out.reshape(Array(S.B, Array(S.N, BF16)))


def split_bounds(K: int, splits: int) -> list[tuple[int, int]]:
    """cuBLAS 12.9.1's contiguous split-K intervals: equal 64-element chunks, then the final remainder."""
    if K <= 0 or K % 16:
        raise ValueError("CublasSplitK: unsupported statics (K_positive_multiple_of_16)")
    blocks = K // 16
    if not 1 < splits <= blocks:
        raise ValueError("CublasSplitK: unsupported statics (SPLIT_K_in_2_to_K_over_16)")
    chunk = ((K + splits - 1) // splits + 63) // 64 * 64
    bounds = list(range(0, K, chunk)) + [K]
    if len(bounds) != splits + 1:
        # Small circuit-check/test statics cannot hold `splits` 64-element chunks; their exact k16 partition remains total.
        bounds = [16 * (z * blocks // splits) for z in range(splits)] + [K]
    return list(zip(bounds, bounds[1:]))


def _splitk(gemm, add, cast=None, partial=()):
    def body(B, S, x, w):
        parts = []
        for lo, hi in split_bounds(S.K, S.SPLIT_K):
            ws = array_of([w[n][lo:hi] for n in range(S.N)])
            p = B.call(bind(gemm, K=hi - lo, N=S.N, DOT=S.DOT), x[lo:hi], ws)
            for d in partial:
                p = B.batch(d, p, axes=(0,))
            parts.append(p)
        acc = parts[0]
        for p in parts[1:]:
            acc = B.batch(add, acc, p, axes=(0, 0))
        return B.batch(cast, acc, axes=(0,)) if cast is not None else acc
    return body


def _serial_splitk(gemm, add, cast, widen):
    def body(B, S, x, w):
        out = None
        for lo, hi in split_bounds(S.K, S.SPLIT_K):
            ws = array_of([w[n][lo:hi] for n in range(S.N)])
            p = B.call(bind(gemm, K=hi - lo, N=S.N, DOT=S.DOT), x[lo:hi], ws)
            if out is not None:
                p = B.batch(add, p, B.batch(widen, out, axes=(0,)), axes=(0, 0))
            out = B.batch(cast, p, axes=(0,))
        return out
    return body


GEMM_F32_DOC = "`x[K] @ W[N, K]^T` with the f32 accumulator as the output: `DotBf16{K, DOT}` on each W row (torch.mm out_dtype=float32)"
BATCHED_DOC = "torch.bmm on bf16: `Gemm{K, N, DOT}` of each member's x against its W (w[b][n][k] the operand at [b, k, n])"
SPLITK_DOC = ("cuBLAS 12.9.1's cc 9.0, 132-SM split-K: equal 64-element-aligned contiguous chunks and a final remainder, one ascending DOT "
              "chain from +0 per chunk, then a separate serial f32 reduction in increasing split order")
SPLITK_BF16_DOC = ("cuBLAS 12.9.1's cc 12.0, 188-SM split-K with a bf16 workspace: `split_bounds`' chunks, one ascending DOT chain from +0 "
                   "per chunk, each partial rounded to bf16 (F2fpBf16) into the workspace, then `splitKreduce_kernel`'s serial f32 "
                   "reduction of the widened partials in increasing split order and one final bf16 rounding")
SERIAL_SPLITK_DOC = ("cuBLAS 12.9.1's cc 12.0, 188-SM serial split-K: `split_bounds`' chunks in order, one ascending DOT chain from +0 "
                     "per chunk, added in f32 to the previous chunk's bf16 output (widened) and rounded to bf16, the first chunk's "
                     "rounded alone")

GemmF32OutWord = CompositeDefinition("GemmF32Out", 1, ("K", "N", "DOT"), _gemm_f32_type(P.BF16, P.F32), _gemm_f32(G.DotBf16),
                                     doc=GEMM_F32_DOC, conformance=CONFORMANCE)
BatchedGemmWord = CompositeDefinition("BatchedGemm", 1, ("B", "K", "N", "DOT"), _batched_type(P.BF16), _batched(G.Gemm),
                                      doc=BATCHED_DOC, conformance=CONFORMANCE)

GemmF32Out = CompositeDefinition("GemmF32Out", 2, ("K", "N", "DOT"), _gemm_f32_type(BF16, F32), _gemm_f32_bits,
                                 doc=f"{GEMM_F32_DOC} (on bits: each k16 step over every row at once, x's own gates once)",
                                 conformance=CONFORMANCE)
BatchedGemm = CompositeDefinition("BatchedGemm", 2, ("B", "K", "N", "DOT"), _batched_type(BF16), _batched_bits,
                                  doc=f"{BATCHED_DOC} (on bits: `GemmF32Out_v2` of each member, then `F2fpBf16_v2` of each word)",
                                  conformance=CONFORMANCE)
CublasSplitKGemmF32OutWord = CompositeDefinition(
    "CublasSplitKGemmF32Out", 1, ("K", "N", "SPLIT_K", "DOT"), _gemm_f32_type(P.BF16, P.F32),
    _splitk(GemmF32OutWord, F.F32Add), doc=f"{SPLITK_DOC}; f32 output", conformance=CONFORMANCE)
CublasSplitKGemmWord = CompositeDefinition(
    "CublasSplitKGemm", 1, ("K", "N", "SPLIT_K", "DOT"), _gemm_f32_type(P.BF16, P.BF16),
    _splitk(GemmF32OutWord, F.F32Add, P.F2fpBf16), doc=f"{SPLITK_DOC}; one final bf16 rounding", conformance=CONFORMANCE)
CublasSplitKGemmF32Out = CompositeDefinition(
    "CublasSplitKGemmF32Out", 2, ("K", "N", "SPLIT_K", "DOT"), _gemm_f32_type(BF16, F32),
    _splitk(GemmF32Out, EW.F32Add), doc=f"{SPLITK_DOC}; f32 output, on bits", conformance=CONFORMANCE)
CublasSplitKGemm = CompositeDefinition(
    "CublasSplitKGemm", 2, ("K", "N", "SPLIT_K", "DOT"), _gemm_f32_type(BF16, BF16),
    _splitk(GemmF32Out, EW.F32Add, BG.F2fpBf16), doc=f"{SPLITK_DOC}; one final bf16 rounding, on bits", conformance=CONFORMANCE)
CublasSplitKGemmBf16PartialsWord = CompositeDefinition(
    "CublasSplitKGemmBf16Partials", 1, ("K", "N", "SPLIT_K", "DOT"), _gemm_f32_type(P.BF16, P.BF16),
    _splitk(GemmF32OutWord, F.F32Add, P.F2fpBf16, (P.F2fpBf16, P.Bf16ToF32)), doc=SPLITK_BF16_DOC,
    conformance=CONFORMANCE_CC12)
CublasSplitKGemmBf16Partials = CompositeDefinition(
    "CublasSplitKGemmBf16Partials", 2, ("K", "N", "SPLIT_K", "DOT"), _gemm_f32_type(BF16, BF16),
    _splitk(GemmF32Out, EW.F32Add, BG.F2fpBf16, (BG.F2fpBf16, BCAST.Bf16ToF32)), doc=f"{SPLITK_BF16_DOC}, on bits",
    conformance=CONFORMANCE_CC12)
CublasSerialSplitKGemmWord = CompositeDefinition(
    "CublasSerialSplitKGemm", 1, ("K", "N", "SPLIT_K", "DOT"), _gemm_f32_type(P.BF16, P.BF16),
    _serial_splitk(GemmF32OutWord, F.F32Add, P.F2fpBf16, P.Bf16ToF32), doc=SERIAL_SPLITK_DOC, conformance=CONFORMANCE_CC12_SERIAL)
CublasSerialSplitKGemm = CompositeDefinition(
    "CublasSerialSplitKGemm", 2, ("K", "N", "SPLIT_K", "DOT"), _gemm_f32_type(BF16, BF16),
    _serial_splitk(GemmF32Out, EW.F32Add, BG.F2fpBf16, BCAST.Bf16ToF32), doc=f"{SERIAL_SPLITK_DOC}, on bits",
    conformance=CONFORMANCE_CC12_SERIAL)
GemmF32Out.word = lambda b: bind(GemmF32OutWord, K=b["K"], N=b["N"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
BatchedGemm.word = lambda b: bind(BatchedGemmWord, B=b["B"], K=b["K"], N=b["N"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
CublasSplitKGemmF32Out.word = lambda b: bind(CublasSplitKGemmF32OutWord, K=b["K"], N=b["N"], SPLIT_K=b["SPLIT_K"],  # type: ignore[attr-defined]
                                                      DOT=word_of(b["DOT"]))
CublasSplitKGemm.word = lambda b: bind(CublasSplitKGemmWord, K=b["K"], N=b["N"], SPLIT_K=b["SPLIT_K"],  # type: ignore[attr-defined]
                                                DOT=word_of(b["DOT"]))
CublasSplitKGemmBf16Partials.word = lambda b: bind(CublasSplitKGemmBf16PartialsWord, K=b["K"], N=b["N"],  # type: ignore[attr-defined]
                                                   SPLIT_K=b["SPLIT_K"], DOT=word_of(b["DOT"]))
CublasSerialSplitKGemm.word = lambda b: bind(CublasSerialSplitKGemmWord, K=b["K"], N=b["N"],  # type: ignore[attr-defined]
                                             SPLIT_K=b["SPLIT_K"], DOT=word_of(b["DOT"]))
