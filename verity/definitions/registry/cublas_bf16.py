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
    CublasSplitKGemmBf16PartialsAligned_v1{K, N, SPLIT_K, ALIGN, TILE_K, DOT}
                                    cc 12.0's split-K over grid.z with a bf16 workspace: `aligned_bounds`' chunks, each
                                    chunk's chain over its k-tiles of TILE_K words residue first (`residue_first`), the
                                    partials rounded to bf16, widened and reduced in order, rounded once more to bf16
    CublasSplitKGemmAligned_v1{K, N, SPLIT_K, ALIGN, TILE_K, DOT}
                                    the same chunks and chains with an f32 workspace: reduced in order, rounded once to bf16
    CublasSerialSplitKGemmAligned_v1{K, N, SPLIT_K, ALIGN, TILE_K, DOT}
                                    the same chunks and chains with no reduction kernel: each chain added to the previous
                                    chunk's bf16 output and rounded to bf16
    GemmPad_v1{T, K, N, DOT}        (x: Array<T, BF16>, w: Array<N, Array<T, BF16>>) -> Array<N, BF16>
                                    `GemmCoordinatePad_v1{T, K, DOT}` of x against each W row: both padded with K - T +0
                                    words at the back

on words, and each one's v2 the same function on bits (its ``word`` the v1, so `boolean.boolean_version` finds it): `GemmF32Out_v2`
takes the K/16 steps of every W row at once, each step one batch over the rows, and `BatchedGemm_v2` is `GemmF32Out_v2` of each
member, then `F2fpBf16_v2` of each word.  Their `DOT` on bits is `HopperBF16WgmmaDot16_v2`, cut into `HOPPER_STEP`'s four
element Definitions (`chain_step`): the first step's from the +0 accumulator, its 32 constant bits folded into the gates, and a
later step's; and each of those in two, the gates that read only the x block, once per block, and the rest, once per row.  So no
row computes a value another computes, nor a step a value its accumulator's constant bits fix.  The training routes' `DOT`
(`TRAIN_DOT`) is `HopperBF16WgmmaDot16_v3`, the same function without the finite stand-ins, cut the same way into
`HOPPER_STEP_V3` (the four parts' `_v2`).  Each part is bound to the
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
slices meet in its epilogue through the bf16 output: `CublasSerialSplitKGemm`.  cuBLAS 13.1.1.3 there slices K over grid.z at
ceil(K / z) rounded up to ALIGN (`aligned_bounds`), its chunks not always whole k16 steps, with a bf16 workspace, an f32 one
or none: the three aligned Definitions.

smollm2-135m's training step (`TRAIN_CAPTURE`) reuses these Definitions on bits for every GEMM it runs, and the lf model's s step
(`LF_M`, `LF_LINEARS`) for its linears'.  F.linear's three GEMMs
at x (M, K), W (N, K) are each one row's `x @ w^T` (`TRAIN_LINEAR_ROLES`): forward x = X's row against W's rows, dX = dY's row
against W^T's, dW = dY^T's row against X^T's.  `train_route` is the route sm_120 takes for one at the captured (M, K, N), and
`route_gemm` binds a route, an explicit parameter, on `TRAIN_DOT`: unsplit `Gemm_v3` (at a K that is not a multiple of 16,
`GemmPad_v2`, x and each W row padded with +0 words at the back to the next multiple, which no capture at such a K has told
apart yet from the residue k-tile first that the grid.z kernels take, nor from the gemv kernel cuBLAS 12.9.1 launches at M 1,
`cublas_gemv`), `CublasSplitKGemmBf16Partials_v2`, `CublasSerialSplitKGemm_v2`, or, where cuBLAS slices K over grid.z
(a route with ``align`` and ``tile_k``: B 8 x S 512's dW), `CublasSplitKGemmBf16PartialsAligned_v2`,
`CublasSplitKGemmAligned_v2` or `CublasSerialSplitKGemmAligned_v2` by its workspace (`_GRID_Z_ROUTES`), each chunk's chain
taking its k-tiles of TILE_K words residue first, as the launched cutlass kernel does; `train_linear` is the two together.
Eager attention's six bmm's (`TRAIN_BMM`) are all unsplit, the batch on grid.z and no reduction kernel: `attention_bmm` binds
`BatchedGemm_v2` on `TRAIN_DOT` over B members (heads), one query or key row each.  Either refuses a shape or role the capture
does not cover.

Under cuda.matmul.allow_tf32 an f32 torch.mm on sm_120 at the rotation codecs' shapes (`TRAIN_TF32`) is an unsplit
`cutlass_80_tensorop_s1688gemm` kernel whose k8 TF32 steps are `mhc.HopperTf32WgmmaDot8_v1`'s (`CONFORMANCE_TF32`):

    DotTf32_v1{K}                   (x, wrow: Array<K, F32>) -> F32
                                    K/8 `HopperTf32WgmmaDot8_v1` steps in ascending k from +0, no epilogue
    GemmTf32_v1{K, N}               (x: Array<K, F32>, w: Array<N, Array<K, F32>>) -> Array<N, F32>
                                    `DotTf32_v1{K}` of x against each W row

`train_tf32` binds `GemmTf32_v1` at a captured (M, K, N) and refuses any other; `boolean_sigma` lifts both to bits (their v2).
"""
from __future__ import annotations

from typing import Any, Callable, NamedTuple, Sequence

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import const_bits
from verity.circuits.boolean.trace import word_of
from verity.definitions.circuits.census import silicon_step
from verity.circuits.defs import CompositeDefinition, bind
from verity.definitions.ops import fp32 as F
from verity.definitions.ops import gemm as G
from verity.definitions.ops import prims as P
from verity.definitions.ops.boolean import cast as BCAST
from verity.definitions.ops.boolean import elementwise as EW
from verity.definitions.ops.boolean import gemm as BG
from verity.circuits.refs import Coll, array_of, broadcast, concat
from verity.circuits.types import Array, Type
from verity.definitions.evidence import EVIDENCE
from verity.definitions.registry import mhc as MHC

__all__ = ["BatchedGemm", "BatchedGemmWord", "ChainStep", "CublasSerialSplitKGemm", "CublasSerialSplitKGemmAligned",
           "CublasSerialSplitKGemmAlignedWord", "CublasSerialSplitKGemmWord", "CublasSplitKGemm", "CublasSplitKGemmAligned",
           "CublasSplitKGemmAlignedWord", "CublasSplitKGemmBf16Partials", "CublasSplitKGemmBf16PartialsAligned",
           "CublasSplitKGemmBf16PartialsAlignedWord", "CublasSplitKGemmBf16PartialsWord", "CublasSplitKGemmF32Out",
           "CublasSplitKGemmF32OutWord", "CublasSplitKGemmWord", "DotTf32",
           "GemmF32Out", "GemmF32OutWord", "GemmPad", "GemmPadWord", "GemmTf32", "HOPPER_STEP", "HOPPER_STEP_V3",
           "HopperBf16Dot16Step", "HopperBf16Dot16StepFirst", "HopperBf16Dot16StepFirstV2", "HopperBf16Dot16StepFirstX",
           "HopperBf16Dot16StepFirstXV2", "HopperBf16Dot16StepV2", "HopperBf16Dot16StepX", "HopperBf16Dot16StepXV2",
           "LF_LINEARS", "LF_M", "TRAIN_BMM", "TRAIN_BMM_SHAPES", "TRAIN_CAPTURE", "TRAIN_CC", "TRAIN_DOT", "TRAIN_LINEARS",
           "TRAIN_LINEAR_ROLES", "TRAIN_M", "TRAIN_ROUTES", "TRAIN_TF32",
           "aligned_bounds", "attention_bmm", "chain_step", "conformance_cc12", "dot_step", "residue_first", "route_gemm",
           "split_bounds", "train_linear", "train_route", "train_tf32"]

BF16, F32 = Array(16, BL.BIT), Array(32, BL.BIT)
_CAPTURE = EVIDENCE["glm53_bf16_gemm_capture"]
CONFORMANCE = ("exact-model-tested: every output word at K/N statics 6144/{19360,160,256}, 7168/{16160,64,384,2048,1024,512}, "
               "and batched (B 8) 192/512 and 512/256, at the captured M (1, 4, 16, 128, 256; the gates' cuBLAS tier 17, 32, 128, "
               "256), on cc 9.0 with 132 SMs (torch 2.13.0+cu129 on its libcublas.so.12), under vLLM's served default (split counts "
               "2 to 5 each fitting only the stated hypothesis; the absorbs never split) and the split-K-off variant: property "
               f"glm53_bf16_gemm_capture, record {_CAPTURE['record'][:16]}, {_CAPTURE['art']['record']} "
               "(tests/properties/glm53_bf16_gemm_capture_gpu.py)")
_PROBES = EVIDENCE["cublas_cc12_probes"]["run_records"]
CC12_PROBES = f"the probes' run records {', '.join(_PROBES[:-1])} and {_PROBES[-1]}"


def conformance_cc12(launch: str) -> str:
    """The cc 12.0 routes' conformance: the probes that read cuBLAS's words where it makes ``launch``."""
    return (f"exact-model-tested: every probed output word of torch.mm on cc 12.0 with 188 SMs (torch 2.13.0+cu129 on its "
            f"libcublas.so.12) at the served linears' (M, K, N) where cuBLAS launches {launch}, on spread, cancelling and class-edge "
            f"words (specials, any exponent, subnormal products), fitting only this hypothesis among its alternatives: {CC12_PROBES}")


CONFORMANCE_CC12 = conformance_cc12("`splitKreduce_kernel` with a bf16 workspace (M 1 to 303, split counts 2 to 16)")
CONFORMANCE_CC12_SERIAL = conformance_cc12("a cutlass kernel whose grid.z slices meet through the bf16 output, no reduction kernel "
                                           "(M 303, K 12288, 5 slices)")
TRAIN_CAPTURE = ("captures of the training workloads' GEMMs on cc 12.0 with 188 SMs (torch 2.14.0+cu130 on "
                 "cuBLAS 13.1.1.3, deterministic, CUBLAS_WORKSPACE_CONFIG=:4096:8) at B 1 x S 256, B 8 x S 512 and B 2 x S "
                 "1024 (the lf model's s step's linears), every launch's kernel and grid recorded, each role's sampled words "
                 "fitting its route exactly on random and count-probe words")
GRID_Z_CAPTURE = ("captures of the lf model's m and xl steps and of the s step's variants (its linears at other widths, "
                  "Muon's and a rotation's GEMMs) on cc 12.0 with 188 SMs under `TRAIN_CAPTURE`'s torch and cuBLAS, every "
                  "launch's kernel and grid recorded.  Each grid.z launch's random and count-probe words fit, at its grid.z, "
                  "its route alone among the unsplit chain, `bf16_gemm_check.cublas_splitk_hypotheses`, serial split-K, and "
                  "`aligned_bounds`' chunks at ALIGN 8 to 64 with a bf16 workspace, an f32 one or none: at ALIGN 8 (16 or 32 "
                  "fit too only where they make the same chunks) and at every TILE_K from 16 to 128, so TILE_K is the "
                  "launched kernel's")
CONFORMANCE_ALIGNED = ("exact-model-tested: every output word of the B 8 x S 512 step's dW in a full-tensor dump of the step "
                       "(row GEMMs at K 4096: N 576 over 7 and 3 grid.z slices at TILE_K 32 and over 12 at TILE_K 64, N 1536 "
                       "over 3 at TILE_K 32), where cuBLAS launches a cutlass kernel over grid.z and `splitKreduce_kernel` with "
                       "a bf16 workspace, at ALIGN 8, and every sampled word of the same GEMMs in "
                       f"{TRAIN_CAPTURE}; and the words of each such launch (K 2048 over 5 to 26 slices, K 3456 over 10, K 4096 "
                       f"over 40, K 11264 over 51 and K 301056 over 80, at TILE_K 32 or 64) in {GRID_Z_CAPTURE}")
CONFORMANCE_ALIGNED_F32 = ("exact-model-tested: the words of the one launch over grid.z with `splitKreduce_kernel` on an f32 "
                           f"workspace (K 2048, N 160, over 12 slices at TILE_K 32) in {GRID_Z_CAPTURE}")
CONFORMANCE_SERIAL_ALIGNED = ("exact-model-tested: the words of each launch over grid.z with no reduction kernel (K 2048 over 3 "
                              f"slices at N 768, 1536 and 4224, TILE_K 32) in {GRID_Z_CAPTURE}")


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


def chain_step(name: str, build: Callable, params: Sequence[tuple[str, Type]], docs: tuple[str, str],
               version: int = 1) -> ChainStep:
    """The `ChainStep` of ``build(C, *elements)`` over ``params`` = (acc, x, w): ``<name>FirstX_v{version}`` and
    ``<name>First_v{version}`` trace it with acc the 32 constant-0 bits, ``<name>X_v{version}`` and ``<name>_v{version}`` as it
    is.  ``docs``: what the step computes, and what x is."""
    what, x = docs
    zero = const_bits(0, 32)
    first = EW.shared_operand(
        (f"{name}FirstX", f"{name}First"), version, lambda C, *els: build(C, zero, *els), params[1:], F32, 0,
        (f"the gates of {what} from acc = +0 that read only {x}, once per {x}", f"{what} from acc = +0, from {x}'s own gates"))
    later = EW.shared_operand(
        (f"{name}X", name), version, build, params, F32, 1,
        (f"the gates of {what} that read only {x}, once per {x}", f"{what}, from {x}'s own gates"))
    return ChainStep(*first, *later)


_A16 = Array(16, BF16)
_HOPPER_PARAMS = (("acc", F32), ("a", _A16), ("b", _A16))
HOPPER_STEP = chain_step("HopperBf16Dot16Step", BG.HopperBF16WgmmaDot16.build, _HOPPER_PARAMS,
                         ("HopperBF16WgmmaDot16_v1(acc, a, b)", "a"))
HopperBf16Dot16StepFirstX, HopperBf16Dot16StepFirst, HopperBf16Dot16StepX, HopperBf16Dot16Step = HOPPER_STEP
#: `HopperBF16WgmmaDot16_v3`'s four parts, the `_v2` of `HOPPER_STEP`'s
HOPPER_STEP_V3 = chain_step("HopperBf16Dot16Step", BG.HopperBF16WgmmaDot16V3.build, _HOPPER_PARAMS,
                            ("HopperBF16WgmmaDot16_v1(acc, a, b)", "a"), version=2)
HopperBf16Dot16StepFirstXV2, HopperBf16Dot16StepFirstV2, HopperBf16Dot16StepXV2, HopperBf16Dot16StepV2 = HOPPER_STEP_V3
for _part in (*HOPPER_STEP, *HOPPER_STEP_V3):
    silicon_step(_part, BG.HopperBF16WgmmaDot16.silicon)
_DOT_STEPS = {BG.HopperBF16WgmmaDot16.id: HOPPER_STEP, BG.HopperBF16WgmmaDot16V3.id: HOPPER_STEP_V3}
#: the k16 step the training routes bind (`route_gemm`, `attention_bmm`)
TRAIN_DOT = BG.HopperBF16WgmmaDot16V3


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


def _gemm_pad_type(X):
    return lambda S: ((("x", Array(S.T, X)), ("w", Array(S.N, Array(S.T, X)))), Array(S.N, X))


def _gemm_pad(coordinate):
    def body(B, S, x, w):
        return B.batch(bind(coordinate, T=S.T, K=S.K, DOT=S.DOT), x, w, axes=(None, 0))
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


def _split(B, S, x, w):
    """`split_bounds`' chunks of x and of each W row: (k, x's chunk, W's)."""
    for lo, hi in split_bounds(S.K, S.SPLIT_K):
        yield hi - lo, x[lo:hi], array_of([w[n][lo:hi] for n in range(S.N)])


def aligned_bounds(K: int, splits: int, align: int) -> list[tuple[int, int]]:
    """cuBLAS 13.1.1.3's cc 12.0 grid.z split-K intervals: chunks of ceil(K / splits) rounded up to ``align``, then the final
    remainder; refuses statics that do not make exactly ``splits`` chunks."""
    if K <= 0 or align <= 0:
        raise ValueError("CublasSplitKAligned: unsupported statics (K_and_ALIGN_positive)")
    if splits < 2:
        raise ValueError("CublasSplitKAligned: unsupported statics (SPLIT_K_at_least_2)")
    chunk = -(-(-(-K // splits)) // align) * align
    bounds = list(range(0, K, chunk)) + [K]
    if len(bounds) != splits + 1:
        raise ValueError("CublasSplitKAligned: unsupported statics (SPLIT_K_chunks_at_ALIGN)")
    return list(zip(bounds, bounds[1:]))


def residue_first(lo: int, hi: int, tile_k: int) -> tuple[int, int]:
    """A chunk [lo, hi) as the cutlass kernel's predicated k-tiles take it: (r, z), its first r = (hi - lo) mod ``tile_k``
    words, then z constant-0 words to a whole tile, then its remaining words in whole tiles; refuses a ``tile_k`` that is not
    whole k16 steps."""
    if tile_k <= 0 or tile_k % 16:
        raise ValueError("CublasSplitKAligned: unsupported statics (TILE_K_whole_k16_steps)")
    r = (hi - lo) % tile_k
    return r, (tile_k - r) % tile_k


def _aligned(X: Type, zero: Callable, width: int):
    """`aligned_bounds`' chunks of x and of each W row, each as `residue_first` lays it out: (k, x's chunk, W's), k its padded
    length.  ``X``: an element's type; ``zero(B)``: the constant-0 leaf, ``width`` of them an element."""
    def chunks(B, S, x, w):
        leaf = None
        for lo, hi in aligned_bounds(S.K, S.SPLIT_K, S.ALIGN):
            r, z = residue_first(lo, hi, S.TILE_K)
            if not z:
                yield hi - lo, x[lo:hi], array_of([w[n][lo:hi] for n in range(S.N)])
                continue
            leaf = leaf or zero(B)
            zeros, k = broadcast(*leaf, width * z), hi - lo + z
            yield (k, Coll(Array(k, X), concat([x[lo:lo + r].refs, zeros, x[lo + r:hi].refs])),
                   Coll(Array(S.N, Array(k, X)),
                        concat([p for n in range(S.N) for p in (w[n][lo:lo + r].refs, zeros, w[n][lo + r:hi].refs)])))
    return chunks


_ALIGNED_WORDS = _aligned(P.BF16, lambda B: B.call(P.const(16, 0)).refs[0], 1)
_ALIGNED_BITS = _aligned(BF16, lambda B: B.call(BL.constant(0)).refs[0], 16)


def _splitk(gemm, add, cast=None, partial=(), chunks=_split):
    def body(B, S, x, w):
        parts = []
        for k, xs, ws in chunks(B, S, x, w):
            p = B.call(bind(gemm, K=k, N=S.N, DOT=S.DOT), xs, ws)
            for d in partial:
                p = B.batch(d, p, axes=(0,))
            parts.append(p)
        acc = parts[0]
        for p in parts[1:]:
            acc = B.batch(add, acc, p, axes=(0, 0))
        return B.batch(cast, acc, axes=(0,)) if cast is not None else acc
    return body


def _serial_splitk(gemm, add, cast, widen, chunks=_split):
    def body(B, S, x, w):
        out = None
        for k, xs, ws in chunks(B, S, x, w):
            p = B.call(bind(gemm, K=k, N=S.N, DOT=S.DOT), xs, ws)
            if out is not None:
                p = B.batch(add, p, B.batch(widen, out, axes=(0,)), axes=(0, 0))
            out = B.batch(cast, p, axes=(0,))
        return out
    return body


GEMM_F32_DOC = "`x[K] @ W[N, K]^T` with the f32 accumulator as the output: `DotBf16{K, DOT}` on each W row (torch.mm out_dtype=float32)"
GEMM_PAD_DOC = ("`x[T] @ W[N, T]^T` with x and each W row padded with K - T +0 words at the back, what a k loop that masks its "
                "loads past T multiplies: `GemmCoordinatePad{T, K, DOT}` on each W row")
CONFORMANCE_PAD = ("not-yet-established: `GemmCoordinatePad`'s k16 chain on each W row, which no capture of a cuBLAS route at a K "
                   "that is not a multiple of 16 has fitted (cc 12.0's grid.z kernels take a residue k-tile first, "
                   "`residue_first`, and cuBLAS 12.9.1 launches a gemv kernel there at M 1, `cublas_gemv`)")
BATCHED_DOC = "torch.bmm on bf16: `Gemm{K, N, DOT}` of each member's x against its W (w[b][n][k] the operand at [b, k, n])"
SPLITK_DOC = ("cuBLAS 12.9.1's cc 9.0, 132-SM split-K: equal 64-element-aligned contiguous chunks and a final remainder, one ascending DOT "
              "chain from +0 per chunk, then a separate serial f32 reduction in increasing split order")
SPLITK_BF16_DOC = ("cuBLAS 12.9.1's cc 12.0, 188-SM split-K with a bf16 workspace: `split_bounds`' chunks, one ascending DOT chain from +0 "
                   "per chunk, each partial rounded to bf16 (F2fpBf16) into the workspace, then `splitKreduce_kernel`'s serial f32 "
                   "reduction of the widened partials in increasing split order and one final bf16 rounding")
SERIAL_SPLITK_DOC = ("cuBLAS 12.9.1's cc 12.0, 188-SM serial split-K: `split_bounds`' chunks in order, one ascending DOT chain from +0 "
                     "per chunk, added in f32 to the previous chunk's bf16 output (widened) and rounded to bf16, the first chunk's "
                     "rounded alone")
_GRID_Z = ("cuBLAS 13.1.1.3's cc 12.0, 188-SM split-K over grid.z: `aligned_bounds`' chunks (ceil(K / SPLIT_K) rounded up "
           "to ALIGN, then the remainder), each chunk's ascending DOT chain from +0 over its k-tiles of TILE_K words as the "
           "cutlass kernel's predicated mainloop takes them, the residue tile first (`residue_first`: the chunk's first "
           "(length mod TILE_K) words, then constant-0 words to a whole tile, then the rest)")
ALIGNED_DOC = (f"{_GRID_Z}; each partial rounded to bf16 (F2fpBf16) into a bf16 workspace, then `splitKreduce_kernel`'s serial "
               "f32 reduction of the widened partials in increasing split order and one final bf16 rounding")
ALIGNED_F32_DOC = (f"{_GRID_Z}; the f32 partials in an f32 workspace, then `splitKreduce_kernel`'s serial f32 reduction in "
                   "increasing split order and one bf16 rounding")
SERIAL_ALIGNED_DOC = (f"{_GRID_Z}; no reduction kernel: the chunks in order, each chain added in f32 to the previous chunk's bf16 "
                      "output (widened) and rounded to bf16, the first chunk's rounded alone")

GemmF32OutWord = CompositeDefinition("GemmF32Out", 1, ("K", "N", "DOT"), _gemm_f32_type(P.BF16, P.F32), _gemm_f32(G.DotBf16),
                                     doc=GEMM_F32_DOC, conformance=CONFORMANCE)
BatchedGemmWord = CompositeDefinition("BatchedGemm", 1, ("B", "K", "N", "DOT"), _batched_type(P.BF16), _batched(G.Gemm),
                                      doc=BATCHED_DOC, conformance=CONFORMANCE)

GemmF32Out = CompositeDefinition("GemmF32Out", 2, ("K", "N", "DOT"), _gemm_f32_type(BF16, F32), _gemm_f32_bits,
                                 doc=f"{GEMM_F32_DOC} (on bits: each k16 step over every row at once, x's own gates once)",
                                 conformance=CONFORMANCE)
GemmPadWord = CompositeDefinition("GemmPad", 1, ("T", "K", "N", "DOT"), _gemm_pad_type(P.BF16), _gemm_pad(G.GemmCoordinatePad),
                                  doc=GEMM_PAD_DOC, conformance=CONFORMANCE_PAD, refuse=G.GemmCoordinatePad.refuse)
GemmPad = CompositeDefinition("GemmPad", 2, ("T", "K", "N", "DOT"), _gemm_pad_type(BF16), _gemm_pad(BG.GemmCoordinatePad),
                              doc=f"{GEMM_PAD_DOC}, on bits", conformance=CONFORMANCE_PAD, refuse=G.GemmCoordinatePad.refuse)
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
_ALIGNED_STATICS = ("K", "N", "SPLIT_K", "ALIGN", "TILE_K", "DOT")
CublasSplitKGemmBf16PartialsAlignedWord = CompositeDefinition(
    "CublasSplitKGemmBf16PartialsAligned", 1, _ALIGNED_STATICS, _gemm_f32_type(P.BF16, P.BF16),
    _splitk(GemmF32OutWord, F.F32Add, P.F2fpBf16, (P.F2fpBf16, P.Bf16ToF32), _ALIGNED_WORDS), doc=ALIGNED_DOC,
    conformance=CONFORMANCE_ALIGNED)
CublasSplitKGemmBf16PartialsAligned = CompositeDefinition(
    "CublasSplitKGemmBf16PartialsAligned", 2, _ALIGNED_STATICS, _gemm_f32_type(BF16, BF16),
    _splitk(GemmF32Out, EW.F32Add, BG.F2fpBf16, (BG.F2fpBf16, BCAST.Bf16ToF32), _ALIGNED_BITS), doc=f"{ALIGNED_DOC}, on bits",
    conformance=CONFORMANCE_ALIGNED)
CublasSplitKGemmAlignedWord = CompositeDefinition(
    "CublasSplitKGemmAligned", 1, _ALIGNED_STATICS, _gemm_f32_type(P.BF16, P.BF16),
    _splitk(GemmF32OutWord, F.F32Add, P.F2fpBf16, chunks=_ALIGNED_WORDS), doc=ALIGNED_F32_DOC,
    conformance=CONFORMANCE_ALIGNED_F32)
CublasSplitKGemmAligned = CompositeDefinition(
    "CublasSplitKGemmAligned", 2, _ALIGNED_STATICS, _gemm_f32_type(BF16, BF16),
    _splitk(GemmF32Out, EW.F32Add, BG.F2fpBf16, chunks=_ALIGNED_BITS), doc=f"{ALIGNED_F32_DOC}, on bits",
    conformance=CONFORMANCE_ALIGNED_F32)
CublasSerialSplitKGemmAlignedWord = CompositeDefinition(
    "CublasSerialSplitKGemmAligned", 1, _ALIGNED_STATICS, _gemm_f32_type(P.BF16, P.BF16),
    _serial_splitk(GemmF32OutWord, F.F32Add, P.F2fpBf16, P.Bf16ToF32, _ALIGNED_WORDS), doc=SERIAL_ALIGNED_DOC,
    conformance=CONFORMANCE_SERIAL_ALIGNED)
CublasSerialSplitKGemmAligned = CompositeDefinition(
    "CublasSerialSplitKGemmAligned", 2, _ALIGNED_STATICS, _gemm_f32_type(BF16, BF16),
    _serial_splitk(GemmF32Out, EW.F32Add, BG.F2fpBf16, BCAST.Bf16ToF32, _ALIGNED_BITS), doc=f"{SERIAL_ALIGNED_DOC}, on bits",
    conformance=CONFORMANCE_SERIAL_ALIGNED)


def _word(word: CompositeDefinition) -> Callable[[dict], Any]:
    """A Definition on bits' ``word``: ``word`` at its bindings, DOT the word Definition its DOT computes."""
    return lambda b: bind(word, **{**b, "DOT": word_of(b["DOT"])})


for _bits, _words in ((GemmF32Out, GemmF32OutWord), (GemmPad, GemmPadWord), (BatchedGemm, BatchedGemmWord),
                      (CublasSplitKGemmF32Out, CublasSplitKGemmF32OutWord), (CublasSplitKGemm, CublasSplitKGemmWord),
                      (CublasSplitKGemmBf16Partials, CublasSplitKGemmBf16PartialsWord),
                      (CublasSerialSplitKGemm, CublasSerialSplitKGemmWord),
                      (CublasSplitKGemmBf16PartialsAligned, CublasSplitKGemmBf16PartialsAlignedWord),
                      (CublasSplitKGemmAligned, CublasSplitKGemmAlignedWord),
                      (CublasSerialSplitKGemmAligned, CublasSerialSplitKGemmAlignedWord)):
    _bits.word = _word(_words)  # type: ignore[attr-defined]

TRAIN_M = (256, 4096)
TRAIN_LINEARS = ((576, 576), (576, 192), (576, 1536), (1536, 576), (576, 49152))
#: the lf model's s step (d 512, SwiGLU 1408, vocabulary 32,000) at B 2 x S 1024
LF_M = (2048,)
LF_LINEARS = ((512, 512), (512, 1408), (1408, 512), (512, 32000))
#: F.linear's GEMMs at x (M, K), W (N, K): role -> (x's rows, w's rows, the row GEMM's (K, N) at (M, K, N))
TRAIN_LINEAR_ROLES: dict[str, tuple[str, str, Callable[[int, int, int], tuple[int, int]]]] = {
    "fwd": ("X", "W", lambda M, K, N: (K, N)),
    "dX": ("dY", "W^T", lambda M, K, N: (N, K)),
    "dW": ("dY^T", "X^T", lambda M, K, N: (M, K)),
}
#: eager attention's bmm's after repeat_kv, per head: op -> (x's rows, w's rows, the row GEMM's (K, N) at (S, D))
TRAIN_BMM: dict[str, tuple[str, str, Callable[[int, int], tuple[int, int]]]] = {
    "qk": ("q", "k", lambda S, D: (D, S)),
    "pv": ("p", "v^T", lambda S, D: (S, D)),
    "dp": ("do", "v", lambda S, D: (D, S)),
    "dv": ("p^T", "do^T", lambda S, D: (S, D)),
    "dq": ("ds", "k^T", lambda S, D: (S, D)),
    "dk": ("ds^T", "q^T", lambda S, D: (S, D)),
}
TRAIN_BMM_SHAPES = ((256, 64), (512, 64))
_NOSPLIT = {"kernel": "nosplit"}
#: the compute capability whose cuBLAS `TRAIN_ROUTES` are (`TRAIN_CAPTURE`)
TRAIN_CC = (12, 0)
#: (M, K, N, role) -> the route sm_120 takes, in `bf16_gemm_check.cublas_route_rows`' form
TRAIN_ROUTES: dict[tuple[int, int, int, str], dict[str, Any]] = {
    **{(M, K, N, role): _NOSPLIT for M in TRAIN_M for K, N in TRAIN_LINEARS for role in TRAIN_LINEAR_ROLES},
    (256, 576, 1536, "dX"): {"kernel": "splitk", "SPLIT_K": 8, "partial": "bf16"},
    (256, 576, 49152, "dX"): {"kernel": "splitk", "SPLIT_K": 17, "partial": "bf16"},
    (4096, 576, 49152, "dX"): {"kernel": "serialk", "SPLIT_K": 7},
    (4096, 576, 49152, "dW"): {"kernel": "serialk", "SPLIT_K": 2},
}
#: B 8 x S 512's grid.z dW: (K, N) -> (grid.z, the k-tile of the cutlass kernel launched, its name's ``_{TILE_K}x{stages}``)
_GRID_Z_DW = {(576, 576): (7, 32), (576, 192): (12, 64), (576, 1536): (3, 32), (1536, 576): (3, 32)}
for (_K, _N), (_z, _t) in _GRID_Z_DW.items():
    TRAIN_ROUTES[(4096, _K, _N, "dW")] = {"kernel": "splitk", "SPLIT_K": _z, "partial": "bf16", "align": 8, "tile_k": _t}
#: The s step's dW slices K 2048 over grid.z into whole k-tiles, where `aligned_bounds` is `split_bounds` and `residue_first`
#: is (0, 0): its route is the plain bf16-workspace split-K, which also has a Definition on words.
TRAIN_ROUTES.update({
    **{(M, K, N, role): _NOSPLIT for M in LF_M for K, N in LF_LINEARS for role in TRAIN_LINEAR_ROLES},
    (2048, 512, 512, "dW"): {"kernel": "splitk", "SPLIT_K": 8, "partial": "bf16"},
    (2048, 512, 1408, "dW"): {"kernel": "splitk", "SPLIT_K": 4, "partial": "bf16"},
    (2048, 1408, 512, "dW"): {"kernel": "splitk", "SPLIT_K": 4, "partial": "bf16"},
    (2048, 512, 32000, "dX"): {"kernel": "serialk", "SPLIT_K": 5},
})


def train_route(role: str, M: int, K: int, N: int) -> dict[str, Any]:
    """The cuBLAS route sm_120 takes for F.linear's ``role`` GEMM at x (M, K), W (N, K) (`TRAIN_CAPTURE`); refuses any the
    capture does not cover."""
    if role not in TRAIN_LINEAR_ROLES:
        raise ValueError(f"train_route: role {role!r} (only {', '.join(TRAIN_LINEAR_ROLES)})")
    if (M, K, N, role) not in TRAIN_ROUTES:
        raise ValueError(f"train_route: (M, K, N) = ({M}, {K}, {N}) is not captured (M in {TRAIN_M} at (K, N) in "
                         f"{TRAIN_LINEARS}, M in {LF_M} at (K, N) in {LF_LINEARS})")
    return dict(TRAIN_ROUTES[(M, K, N, role)])


#: a grid.z route's Definition on bits by its (kernel, partial): a bf16 workspace, an f32 one, or the slices meeting through the
#: bf16 output
_GRID_Z_ROUTES = {("splitk", "bf16"): CublasSplitKGemmBf16PartialsAligned, ("splitk", "f32"): CublasSplitKGemmAligned,
                  ("serialk", None): CublasSerialSplitKGemmAligned}


def route_gemm(route: dict[str, Any], K: int, N: int) -> Any:
    """``route``'s Definition on bits at the row GEMM (K, N), `HopperBF16WgmmaDot16_v3` (`TRAIN_DOT`) its k16 step:
    (x: Array<K, BF16>, w: Array<N, Array<K, BF16>>) -> Array<N, BF16>.  An unsplit route at a K that is not a multiple of 16
    is `GemmPad_v2`, its operands padded with +0 words to the next multiple."""
    kind, dot = route.get("kernel"), TRAIN_DOT
    if kind == "nosplit":
        return bind(BG.Gemm, K=K, N=N, DOT=dot) if K % 16 == 0 else bind(GemmPad, T=K, K=K + (-K) % 16, N=N, DOT=dot)
    if "align" in route:
        aligned = _GRID_Z_ROUTES.get((kind, route.get("partial")))
        if aligned is None:
            raise ValueError(f"route_gemm: route {route} has no Definition on bits")
        if "tile_k" not in route:
            raise ValueError(f"route_gemm: route {route} has no tile_k, the grid.z kernel's k-tile")
        z, align, tile = int(route["SPLIT_K"]), int(route["align"]), int(route["tile_k"])
        for lo, hi in aligned_bounds(K, z, align):
            residue_first(lo, hi, tile)
        return bind(aligned, K=K, N=N, SPLIT_K=z, ALIGN=align, TILE_K=tile, DOT=dot)
    partials = CublasSplitKGemmBf16Partials if route.get("partial") == "bf16" else None
    split = {"serialk": CublasSerialSplitKGemm, "splitk": partials}.get(kind)
    if split is None:
        raise ValueError(f"route_gemm: route {route} has no Definition on bits")
    split_bounds(K, int(route["SPLIT_K"]))
    return bind(split, K=K, N=N, SPLIT_K=int(route["SPLIT_K"]), DOT=dot)


def train_linear(role: str, M: int, K: int, N: int) -> Any:
    """`route_gemm` of `train_route`: one row of F.linear's ``role`` GEMM at x (M, K), W (N, K) as sm_120 computes it."""
    return route_gemm(train_route(role, M, K, N), *TRAIN_LINEAR_ROLES[role][2](M, K, N))


def attention_bmm(op: str, S: int, D: int, B: int = 1) -> Any:
    """`BatchedGemm_v2{B, K, N, HopperBF16WgmmaDot16_v3}` of eager attention's bmm ``op`` (`TRAIN_BMM`) at S tokens and head
    dimension D: B members (heads), each one x row against its head's w rows; refuses any (S, D) the capture does not cover."""
    if op not in TRAIN_BMM:
        raise ValueError(f"attention_bmm: op {op!r} (only {', '.join(TRAIN_BMM)})")
    if (S, D) not in TRAIN_BMM_SHAPES:
        raise ValueError(f"attention_bmm: (S, D) = ({S}, {D}) is not captured (only {TRAIN_BMM_SHAPES})")
    if B < 1:
        raise ValueError(f"attention_bmm: B = {B}")
    K, N = TRAIN_BMM[op][2](S, D)
    return bind(BatchedGemm, B=B, K=K, N=N, DOT=TRAIN_DOT)


#: torch.mm(a[M, K], b[K, N]) of f32 operands under cuda.matmul.allow_tf32 at these (M, K, N), each operand row- or
#: column-major (the rotation codecs': x.reshape(-1, 64) @ R and y @ R^T per tensor and over all of them, and y @ R^T, z @ R at
#: 2,048 rows of 512), where cuBLAS launches an unsplit `cutlass_80_tensorop_s1688gemm` kernel (`CONFORMANCE_TF32`)
TRAIN_TF32 = ((4096, 64, 64), (11264, 64, 64), (301056, 64, 64), (2048, 512, 512))
_TF32_CAPTURE = EVIDENCE["cc12_tf32_gemm_capture"]
CONFORMANCE_TF32 = ("exact-model-tested: every sampled output word (rows 0, 1, 2, M/2, M - 2 and M - 1, at N 512 the first 160 "
                    "and last 32 columns) of torch.mm of f32 operands under allow_tf32 on cc 12.0 with 188 SMs (torch "
                    "2.14.0+cu130 on cuBLAS 13.1.1.3, deterministic) at each (M, K, N) of `TRAIN_TF32` in all four operand "
                    "layouts, on normal words over 13 binades, where cuBLAS launches an unsplit "
                    "`cutlass_80_tensorop_s1688gemm` kernel, fitting it alone among the k8 step's alternatives (groups of 8, "
                    "4 + 4 or 2 + 2 + 2 + 2 products, adders 24 to 28 bits wide, each operand rounded to TF32 by RNE, RNA or "
                    f"truncation) and an f32 FFMA chain: capture {_TF32_CAPTURE['run_record']}, fit {_TF32_CAPTURE['fit']}")


def _refuse_k8(b: dict) -> str | None:
    K = b.get("K")
    if isinstance(K, int) and (K < 8 or K % 8):
        return f"K={K} is not a positive multiple of 8: the GEMM takes whole k8 TF32 steps"
    return None


def _dot_tf32(B, S, x, wrow):
    acc = B.call(P.ZERO32)
    for s in range(S.K // 8):
        acc = B.call(MHC.HopperTf32WgmmaDot8, acc, x[8 * s:8 * s + 8], wrow[8 * s:8 * s + 8])
    return acc


DotTf32 = CompositeDefinition(
    "DotTf32", 1, ("K",), lambda S: ((("x", Array(S.K, P.F32)), ("wrow", Array(S.K, P.F32))), P.F32), _dot_tf32,
    doc="one output word of an f32 `x @ W^T` under TF32: K/8 `HopperTf32WgmmaDot8_v1` steps in ascending k from +0 (each "
        "operand rounded RNE to TF32 inside the step), no epilogue",
    conformance=CONFORMANCE_TF32, refuse=_refuse_k8)
GemmTf32 = CompositeDefinition(
    "GemmTf32", 1, ("K", "N"), _gemm_f32_type(P.F32, P.F32),
    lambda B, S, x, w: B.batch(bind(DotTf32, K=S.K), x, w, axes=(None, 0)),
    doc="`x[K] @ W[N, K]^T -> [N]` of f32 words as cuBLAS's unsplit TF32 GEMM computes it: `DotTf32_v1{K}` of x against each "
        "W row",
    conformance=CONFORMANCE_TF32, refuse=_refuse_k8)


def train_tf32(M: int, K: int, N: int) -> Any:
    """`GemmTf32_v1{K, N}`: one row of torch.mm(a[M, K], b[K, N]) of f32 operands under allow_tf32 as sm_120's cuBLAS computes
    it, x a row of a and W b^T; refuses an (M, K, N) the capture does not cover (`TRAIN_TF32`)."""
    if (M, K, N) not in TRAIN_TF32:
        raise ValueError(f"train_tf32: (M, K, N) = ({M}, {K}, {N}) is not captured (only {TRAIN_TF32})")
    return bind(GemmTf32, K=K, N=N)
