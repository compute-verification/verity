"""cuBLAS's bf16 GEMMs on sm_90: GLM-5.3's and DeepSeek-V4-Pro's MoE router gates past the cuteDSL tier (`torch.mm(x, W.T,
out_dtype=float32)`, `GateLinear` tier 4), their bf16-output linears, DeepSeek-V4-Pro's compressor `torch.mm(...,
out_dtype=float32)` calls, and GLM-5.3's MLA absorbs (`torch.bmm` of the queries against `W_UK_T`, of the attention output against
`W_UV`).

    GemmF32Out_v1{K, N, DOT}        (x: Array<K, BF16>, w: Array<N, Array<K, BF16>>) -> Array<N, F32>
                                    `DotBf16_v2{K, DOT}` of x against each W row: the f32 accumulator is the output
    BatchedGemm_v1{B, K, N, DOT}    (x: Array<B, Array<K, BF16>>, w: Array<B, Array<N, Array<K, BF16>>>) -> Array<B, Array<N, BF16>>
                                    `Gemm_v2{K, N, DOT}` of member b's x against member b's W, ``w[b][n][k]`` the right operand at
                                    ``[b, k, n]``
    CublasSplitKGemmF32Out_v1{K, N, SPLIT_K, DOT}
                                    the default cuBLAS split-K hypothesis for f32 output
    CublasSplitKGemm_v1{K, N, SPLIT_K, DOT}
                                    the same split-K hypothesis, rounded once to bf16 after reduction

on words, and each one's v2 the same function on bits (its ``word`` the v1, so `boolean.boolean_version` finds it): `GemmF32Out_v2`
takes the K/16 steps of every W row at once, each step one batch over the rows, and `BatchedGemm_v2` is `GemmF32Out_v2` of each
member, then `F2fpBf16_v2` of each word.  Their only `DOT` on bits is `HopperBF16WgmmaDot16_v2`, cut into `HOPPER_STEP`'s four
element Definitions (`chain_step`): the first step's from the +0 accumulator, its 32 constant bits folded into the gates, and a
later step's; and each of those in two, the gates that read only the x block, once per block, and the rest, once per row.  So no
row computes a value another computes, nor a step a value its accumulator's constant bits fix.  `BatchedGemm_v2` is the function
of `mla.MlaBmm_v1` (vLLM's batch-invariant Triton `bmm_kernel` on sm_120), which names no word; `tests/program/test_bf16_gemm.py`
checks that they agree.  `bf16_gemm_check` holds the numpy models.

The two split-K Definitions deliberately state a closed-source hypothesis, not an established cuBLAS contract: split the sequence
of k16 blocks into balanced contiguous intervals; run each interval's ascending `HopperBF16WgmmaDot16_v1` chain from +0; store f32
partials; and have the separate reduction kernel add them serially in increasing split order in f32, rounding only the final result
for a bf16 output.  The H200 capture records cuBLAS's version, each selected kernel and launch grid (whose z dimension pins
`SPLIT_K`), and fits crafted cancellation rows against this hypothesis and alternatives (strided partition, reverse/tree reduction,
and bf16 partials).  Until that capture passes, conformance remains not established.
"""
from __future__ import annotations

from typing import Any, Callable, NamedTuple, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import const_bits
from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.refs import array_of
from verity.ml.circuits.types import Array, Type

__all__ = ["BatchedGemm", "BatchedGemmWord", "ChainStep", "CublasSplitKGemm", "CublasSplitKGemmF32Out",
           "CublasSplitKGemmF32OutWord", "CublasSplitKGemmWord", "GemmF32Out", "GemmF32OutWord", "HOPPER_STEP",
           "HopperBf16Dot16Step", "HopperBf16Dot16StepFirst", "HopperBf16Dot16StepFirstX", "HopperBf16Dot16StepX", "chain_step",
           "dot_step", "split_bounds"]

BF16, F32 = Array(16, BL.BIT), Array(32, BL.BIT)
CONFORMANCE = ("not-yet-established: GLM-5.3's and DeepSeek-V4-Pro's shapes on the H200 "
               "(tests/properties/glm53_bf16_gemm_capture_gpu.py)")


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
    """Balanced contiguous intervals of k16 blocks, the default cuBLAS split-K partition hypothesis."""
    if K <= 0 or K % 16:
        raise ValueError("CublasSplitK: unsupported statics (K_positive_multiple_of_16)")
    blocks = K // 16
    if not 1 < splits <= blocks:
        raise ValueError("CublasSplitK: unsupported statics (SPLIT_K_in_2_to_K_over_16)")
    return [(16 * (z * blocks // splits), 16 * ((z + 1) * blocks // splits)) for z in range(splits)]


def _splitk(gemm, add, cast=None):
    def body(B, S, x, w):
        parts = []
        for lo, hi in split_bounds(S.K, S.SPLIT_K):
            ws = array_of([w[n][lo:hi] for n in range(S.N)])
            parts.append(B.call(bind(gemm, K=hi - lo, N=S.N, DOT=S.DOT), x[lo:hi], ws))
        acc = parts[0]
        for p in parts[1:]:
            acc = B.batch(add, acc, p, axes=(0, 0))
        return B.batch(cast, acc, axes=(0,)) if cast is not None else acc
    return body


GEMM_F32_DOC = "`x[K] @ W[N, K]^T` with the f32 accumulator as the output: `DotBf16{K, DOT}` on each W row (torch.mm out_dtype=float32)"
BATCHED_DOC = "torch.bmm on bf16: `Gemm{K, N, DOT}` of each member's x against its W (w[b][n][k] the operand at [b, k, n])"
SPLITK_DOC = ("default cuBLAS split-K hypothesis: balanced contiguous k16 intervals, one ascending DOT chain from +0 per interval, "
              "then a separate serial f32 reduction in increasing split order")

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
    _splitk(GemmF32OutWord, P.F32Add), doc=f"{SPLITK_DOC}; f32 output", conformance=CONFORMANCE)
CublasSplitKGemmWord = CompositeDefinition(
    "CublasSplitKGemm", 1, ("K", "N", "SPLIT_K", "DOT"), _gemm_f32_type(P.BF16, P.BF16),
    _splitk(GemmF32OutWord, P.F32Add, P.F2fpBf16), doc=f"{SPLITK_DOC}; one final bf16 rounding", conformance=CONFORMANCE)
CublasSplitKGemmF32Out = CompositeDefinition(
    "CublasSplitKGemmF32Out", 2, ("K", "N", "SPLIT_K", "DOT"), _gemm_f32_type(BF16, F32),
    _splitk(GemmF32Out, EW.F32Add), doc=f"{SPLITK_DOC}; f32 output, on bits", conformance=CONFORMANCE)
CublasSplitKGemm = CompositeDefinition(
    "CublasSplitKGemm", 2, ("K", "N", "SPLIT_K", "DOT"), _gemm_f32_type(BF16, BF16),
    _splitk(GemmF32Out, EW.F32Add, BG.F2fpBf16), doc=f"{SPLITK_DOC}; one final bf16 rounding, on bits", conformance=CONFORMANCE)
GemmF32Out.word = lambda b: bind(GemmF32OutWord, K=b["K"], N=b["N"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
BatchedGemm.word = lambda b: bind(BatchedGemmWord, B=b["B"], K=b["K"], N=b["N"], DOT=word_of(b["DOT"]))  # type: ignore[attr-defined]
CublasSplitKGemmF32Out.word = lambda b: bind(CublasSplitKGemmF32OutWord, K=b["K"], N=b["N"], SPLIT_K=b["SPLIT_K"],  # type: ignore[attr-defined]
                                                      DOT=word_of(b["DOT"]))
CublasSplitKGemm.word = lambda b: bind(CublasSplitKGemmWord, K=b["K"], N=b["N"], SPLIT_K=b["SPLIT_K"],  # type: ignore[attr-defined]
                                                DOT=word_of(b["DOT"]))
