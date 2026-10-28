"""The routed expert GEMMs of a padded static MoE layer as composites: one proof block per (token slot, top-k slot) pair, the expert
a value.

    MoeSelectBf16x{E}_v1              (e: i32, column: Array<E, bf16>) -> bf16: column[e] for 0 <= e < E (signed), else 0x7FFF
    MoePairCoordinate_v1{E, K, DOT}   (x: Array<K, bf16>, e: i32, slab: Array<E, Array<K, bf16>>) -> bf16:
                                      GemmCoordinate_v2{K, DOT}(x, slab[e])
    MoePairCoordinateW_v1{E, K, DOT}  (x, e, g: f32, slab) -> bf16: F2fpBf16(F32Mul_v2(DotBf16_v2{K, DOT}(x, slab[e]), g))
    MoePairGemm_v1{E, K, N, DOT}      (x, e, w: Array<E, Array<N, Array<K, bf16>>>) -> Array<N, bf16>: x @ w[e]^T
    MoePairGemmW_v1{E, K, N, DOT}     (x, e, g, w) -> Array<N, bf16>: g * (x @ w[e]^T), g multiplied into the f32 accumulator

The served Program of an MoE layer is shaped by public parameters alone (batch slots, top_k, E, H, I): pair (t, k) is one
`MoePairGemm_v1` over x[t], e = topk_ids[t, k] and the whole expert bank w13, one `MoePairGemmW_v1` over its activation,
e, g = topk_weights[t, k] and w2, and every one of them is wired the same whatever the router picks. Which expert's weights
a coordinate reads is decided inside its proof block, by a select over the E experts' words of its slab column (`MoeSelectBf16x{E}`
per k), so no proof block reads another's output through a value-dependent index.

This is Triton `fused_moe_kernel`'s row function, so a pair's words are the served kernel's whatever tile its row shared:
with SPLIT_K = 1 the kernel's accumulator for row (t, k) is one f32 chain of `tl.dot` k16 steps in ascending k from zero
over x[t] and W[e][n, :], and `acc.to(bf16)` (the w13 launch) or `acc * topk_weights[t, k]` then `.to(bf16)` (the w2
launch, MUL_ROUTED_WEIGHT). `DOT` is the k16 step (`gemm.GEMM_HOPPER`'s `HopperBF16WgmmaDot16_v1` for wgmma and for Triton's
mma.sync on sm_90 and sm_120, `gemm.GEMM_AMPERE`'s on sm_80/sm_89). These are the functions of the vLLM registry's
`MoeExpertCoordinate(W)_v2` and `MoeExpertGemm(W)_v2` (held equal in `verity_vllm`'s tests), defined here so the Boolean
forms (`ops.boolean.moe`, the next version of each) have a word view inside `verity.ml.circuits`.

The store ids the pair GEMMs' conformance cites are `EVIDENCE_IDS` (`moe_evidence.json`), keyed by Definition id and then
by role, which the conformance names in backticks.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from verity.ml.circuits.defs import PrimitiveDefinition, bind, composite
from verity.ml.circuits.refs import Coll, concat, strided_view
from verity.ml.circuits.types import Array, Type, Value

from . import fp32 as F
from . import gemm as G
from . import prims as P
from .prims import BF16, F32

__all__ = ["EVIDENCE_IDS", "FILL", "I32", "MoePairCoordinate", "MoePairCoordinateW", "MoePairGemm", "MoePairGemmW", "SELECT_ID",
           "columns", "coordinate_major", "select_bf16"]

#: Definition id -> role -> the run ids its conformance names by that role.
EVIDENCE_IDS: dict[str, dict] = json.loads(Path(__file__).with_name("moe_evidence.json").read_text())

I32 = Value(32)
#: the word an id outside [0, E) selects (the vocabulary gather's extension, `GatherBf16x{V}_v1`)
FILL = 0x7FFF
SELECT_ID = r"^MoeSelectBf16x([1-9][0-9]*)_v1$"

_SELECTS: dict[int, PrimitiveDefinition] = {}


def select_bf16(E: int) -> PrimitiveDefinition:
    """`MoeSelectBf16x{E}_v1`: ``column[e]`` for the signed i32 ``e`` in [0, E), else 0x7FFF."""
    if E < 1:
        raise ValueError(f"a select chooses among at least one expert, not {E}")
    p = _SELECTS.get(E)
    if p is None:
        def ev(e: int, *column: int, _E=E) -> int:
            return column[e] if e < _E else FILL

        def ev_lazy(get, _E=E) -> int:
            e = get(0)
            return get(1 + e) if e < _E else FILL

        p = PrimitiveDefinition(f"MoeSelectBf16x{E}", 1, (("e", I32), ("column", Array(E, BF16))), BF16, ev,
                                doc=f"column[e] for the signed i32 e in [0, {E}), else 0x7FFF: the bf16 word of expert e among a "
                                    f"weight element's {E} experts", conformance="exact-model-tested")
        p.evaluate_lazy = ev_lazy  # type: ignore[attr-defined]
        _SELECTS[E] = p
    return p


def columns(slab: Coll, E: int, K: int, word: Type = BF16) -> Coll:
    """``cols[k][e] = slab[e][k]`` of a row-major [E][K] slab of ``word``s: no gates."""
    L = word.leaves
    return Coll(Array(K, Array(E, word)), concat([strided_view(slab, L * k, L, L * K, 1, L * E) for k in range(K)]))


def coordinate_major(w: Coll, E: int, N: int, K: int, word: Type = BF16) -> Coll:
    """``wt[n][e][k] = w[e][n][k]``: the [E][K] slab of each output coordinate n, as N strided parts; no gates."""
    L = word.leaves
    return Coll(Array(N, Array(E, Array(K, word))),
                concat([strided_view(w, L * n * K, L * K, L * N * K, 1, L * E * K) for n in range(N)]))


def _coordinate_sig(routed: bool):
    g = (("g", F32),) if routed else ()
    return lambda S: ((("x", Array(S.K, BF16)), ("e", I32)) + g + (("slab", Array(S.E, Array(S.K, BF16))),), BF16)


def _gemm_sig(routed: bool):
    g = (("g", F32),) if routed else ()
    return lambda S: ((("x", Array(S.K, BF16)), ("e", I32)) + g + (("w", Array(S.E, Array(S.N, Array(S.K, BF16)))),),
                      Array(S.N, BF16))


def _row(B, S, e, slab):
    return B.batch(select_bf16(S.E), e, columns(slab, S.E, S.K), axes=(None, 0))


@composite("MoePairCoordinate", 1, ["E", "K", "DOT"], _coordinate_sig(False), conformance="exact-model-tested",
           refuse=G.GemmCoordinate.refuse)
def MoePairCoordinate(B, S, x, e, slab):
    """One output word of the w13 launch of a pair routed to expert e: `GemmCoordinate_v2{K, DOT}(x, slab[e])`, the row
    selected inside the proof block (`MoeSelectBf16x{E}_v1` per k over the column's E experts)."""
    return B.call(bind(G.GemmCoordinate, K=S.K, DOT=S.DOT), x, _row(B, S, e, slab))


@composite("MoePairCoordinateW", 1, ["E", "K", "DOT"], _coordinate_sig(True), conformance="exact-model-tested",
           refuse=G.GemmCoordinate.refuse)
def MoePairCoordinateW(B, S, x, e, g, slab):
    """One output word of the w2 launch of a pair routed to expert e with weight g: `F2fpBf16(F32Mul_v2(DotBf16_v2{K,
    DOT}(x, slab[e]), g))`, the routed weight multiplied into the f32 accumulator before the one bf16 conversion."""
    acc = B.call(bind(G.DotBf16, K=S.K, DOT=S.DOT), x, _row(B, S, e, slab))
    return B.call(P.F2fpBf16, B.call(F.F32MulV2, acc, g))


#: the served capture both pair GEMMs were replayed against (`MoePairGemm(W)_v1{E=128, ..., DOT=HopperBF16WgmmaDot16_v1}`)
_SERVED = ("vLLM's Triton fused_moe_kernel serving Qwen3-30B-A3B (bf16) on the RTX PRO 6000, layers 0, 24 and 47 at M = 1..256 "
          "(naive and sorted schedules, BLOCK_SIZE_M 16 and 64; capture `capture`, replays `replays`)")


@composite("MoePairGemm", 1, ["E", "K", "N", "DOT"], _gemm_sig(False),
           conformance=f"exact-model-tested; {_SERVED}: 11,612,160/11,612,160 w13 output words bit-exact",
           refuse=G.GemmCoordinate.refuse)
def MoePairGemm(B, S, x, e, w):
    """``y = x @ w[e]^T`` for one pair: `MoePairCoordinate_v1` over each output coordinate's [E][K] slab."""
    return B.batch(bind(MoePairCoordinate, E=S.E, K=S.K, DOT=S.DOT), x, e, coordinate_major(w, S.E, S.N, S.K),
                   axes=(None, None, 0))


@composite("MoePairGemmW", 1, ["E", "K", "N", "DOT"], _gemm_sig(True),
           conformance=f"exact-model-tested; {_SERVED}: 15,482,880/15,482,880 w2 output words bit-exact",
           refuse=G.GemmCoordinate.refuse)
def MoePairGemmW(B, S, x, e, g, w):
    """``y = g * (x @ w[e]^T)`` for one pair: `MoePairCoordinateW_v1` over each output coordinate's [E][K] slab."""
    return B.batch(bind(MoePairCoordinateW, E=S.E, K=S.K, DOT=S.DOT), x, e, g, coordinate_major(w, S.E, S.N, S.K),
                   axes=(None, None, None, 0))


# ---- the accelerator -----------------------------------------------------------------------------------------------------

def selected(e: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """``bank[i, e[i]]`` for each instance i (``bank`` (n, E, ...)), 0x7FFF for an e outside [0, E) (signed i32)."""
    E = bank.shape[1]
    ok = np.asarray(e, dtype=np.uint64) < E
    picked = bank[np.arange(bank.shape[0]), np.where(ok, e, 0).astype(np.int64)]
    return np.where(ok.reshape((-1,) + (1,) * (picked.ndim - 1)), picked, np.asarray(FILL, picked.dtype))


def _f32_mul_ptx(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`F32Mul_v2` on arrays: numpy's IEEE float32 product (RN-even, subnormals kept), every NaN `F32_NAN_PTX`."""
    with np.errstate(all="ignore"):
        p = (a.astype(np.uint32).view(np.float32) * b.astype(np.uint32).view(np.float32)).view(np.uint32)
    return np.where(np.isnan(p.view(np.float32)), np.uint32(F.F32_NAN_PTX), p)


def _sample(gemm: bool, routed: bool):
    """`self_check`'s instances: `sample_tc`'s operand words for x and the selected rows (an eighth exact cancellations),
    the other experts' rows drawn alike, e in [0, E) seven times in eight, else an edge id (E, -1, 2^31, a uniform word);
    g an accumulator word."""

    def sample(rng: np.random.Generator, n: int, params, *, E: int, K: int, N: int = 1, **_: Any) -> list[np.ndarray]:
        from verity.ml.kernels import tc as KT

        x, rows = KT.sample_tc(rng, n * N, [Array(K, BF16), Array(K, BF16)])
        x = x.reshape(n, N, K)[:, 0]
        bank = KT.sample_tc(rng, n * E * N, [Array(K, BF16)])[0].reshape(n, E, N, K)
        e = rng.integers(0, E, size=n).astype(np.uint32)
        edge = np.asarray([E, 0xFFFFFFFF, 0x80000000, 0], np.uint32)[rng.integers(0, 4, size=n)]
        edge[edge == 0] = rng.integers(E, 1 << 32, size=int((edge == 0).sum()), dtype=np.uint64).astype(np.uint32)
        e = np.where(rng.random(n) < 0.125, edge, e)
        ok = e < E
        bank[np.flatnonzero(ok), e[ok]] = rows.reshape(n, N, K)[ok]
        out = [x, e]
        if routed:
            out.append(KT.sample_tc(rng, n, [Value(32)])[0])
        out.append(bank if gemm else bank[:, :, 0])
        return out

    return sample


def _register() -> None:
    from verity.ml.circuits.codec import register_lazy_family
    from verity.ml.circuits.evaluation import register_kernel
    from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16
    from verity.ml.circuits.silicon.total import HOPPER_BF16_WGMMA_K16
    from verity.ml.kernels import tc as KT

    register_lazy_family(SELECT_ID, lambda m: select_bf16(int(m[1])))
    steps = {P.AmpereBF16TcDot16: AMPERE_BF16_M16N8K16, P.HopperBF16WgmmaDot16: HOPPER_BF16_WGMMA_K16}
    dot, coord = KT.dot_kernel(steps), KT.gemm_coordinate_kernel(steps)
    domain = "every encoding; DOT one of the core BF16 steps (other steps fall back to the reference)"

    def pair(x, e, slab, *, E, K, DOT):
        return coord(x, selected(e, slab), K=K, DOT=DOT)

    def pair_w(x, e, g, slab, *, E, K, DOT):
        res = dot(x, selected(e, slab), K=K, DOT=DOT)
        if res is NotImplemented:
            return res
        return KT.f2fp_bf16_batch(_f32_mul_ptx(res[0], g))[0], res[1]

    def gemm(x, e, w, *, E, K, N, DOT):
        n = x.shape[0]
        res = coord(np.repeat(x, N, axis=0), selected(e, w).reshape(n * N, K), K=K, DOT=DOT)
        if res is NotImplemented:
            return res
        return res[0].reshape(n, N), res[1].reshape(n, N).any(axis=1)

    def gemm_w(x, e, g, w, *, E, K, N, DOT):
        n = x.shape[0]
        res = dot(np.repeat(x, N, axis=0), selected(e, w).reshape(n * N, K), K=K, DOT=DOT)
        if res is NotImplemented:
            return res
        out = KT.f2fp_bf16_batch(_f32_mul_ptx(res[0], np.repeat(g, N)))[0]
        return out.reshape(n, N), res[1].reshape(n, N).any(axis=1)

    register_kernel(MoePairCoordinate, pair, domain=domain, sample=_sample(False, False))
    register_kernel(MoePairCoordinateW, pair_w, domain=domain, sample=_sample(False, True))
    register_kernel(MoePairGemm, gemm, domain=domain, sample=_sample(True, False))
    register_kernel(MoePairGemmW, gemm_w, domain=domain, sample=_sample(True, True))


_register()
