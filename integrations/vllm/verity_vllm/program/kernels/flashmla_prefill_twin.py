"""numpy twin of `registry.flashmla_prefill` (DeepSeek-V4-Pro's FlashMLA sparse prefill on the H200, 6bc49418, and vLLM
d9105ea80's CuTe K-cache gather).

`prefill` computes `FlashMlaPrefill_v1` over numpy words, its heads at once; `gather_rows` is `FlashMlaGatherRow_v1` over rows and
`head` the attention over staged rows.  The arithmetic is written here again from phase1.cuh and its SASS rather than read from the
registry, over `flashmla_twin`'s f32 steps (FADD / FMUL / FFMA round to nearest with subnormals, every NaN result 0x7FFFFFFF;
MUFU through core's measured tables; the wgmma k16 step through `verity.ml.kernels.tc`), so the self-check compares two writings.

`registry.flashmla_prefill` registers it (`register_twin`) with `kernel_registry` as kernel ``flashmla-twin`` of
`FlashMlaGatherRow_v1`, `FlashMlaPrefillHead_v1` and `FlashMlaPrefill_v1` (``DOT`` the Hopper wgmma k16 step).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from verity.ml.kernels import tc as CK
from verity_vllm.program.kernels import derived_rows as DR
from verity_vllm.program.kernels import flashmla_twin as FT
from verity_vllm.program.kernels.kernel_registry import register

U32 = np.uint32
KERNEL = FT.KERNEL
DOMAIN = FT.DOMAIN


def fdiv(a, b) -> np.ndarray:
    """`div.rn.f32`: RN(a / b) with subnormals, every NaN 0x7FFFFFFF."""
    a, b = np.broadcast_arrays(FT._w(a), FT._w(b))
    with np.errstate(all="ignore"):
        return FT._gpu(DR.div_rn(a, b))


# ---- the gather and the rows ------------------------------------------------------------------------------------------------

def e8m0_shl_bf16(x) -> np.ndarray:
    """The gather's scale word: the byte shifted into the bf16 exponent."""
    return ((np.asarray(x, dtype=np.uint16) & np.uint16(0xFF)) << np.uint16(7)).astype(np.uint16)


def gather_rows(nope, scale, rope, QT: int) -> np.ndarray:
    """`FlashMlaGatherRow_v1` over rows: (N, DN) e4m3 bytes, (N, DN / QT) e8m0 bytes, (N, DR) bf16 -> (N, DN + DR) bf16."""
    x = DR.f2fp_bf16(CK.e4m3_to_f32_batch(np.asarray(nope))[0])
    s = np.repeat(e8m0_shl_bf16(scale), QT, axis=-1)
    return np.concatenate([DR.f2fp_bf16(FT.fmul(FT._bf16(x), FT._bf16(s))), np.asarray(rope, dtype=np.uint16)], axis=-1)


def row_valid(idx, n, skv) -> np.ndarray:
    """`FlashMlaPrefillValid_v1` of every row in kernel order: its index in [0, skv) and its position below n, all signed."""
    idx = np.asarray(idx, dtype=U32).view(np.int32)
    n, skv = (int(np.asarray(v, dtype=U32).reshape(-1).view(np.int32)[0]) for v in (n, skv))
    return (idx >= 0) & (idx < skv) & (np.arange(idx.shape[-1]) < n)


# ---- one pair, one head -------------------------------------------------------------------------------------------------------

@dataclass
class State:
    m: np.ndarray        # (H,) the running max (the registry carries it negated)
    l: np.ndarray        # (H, 8) warpgroup 0's four lanes, then warpgroup 1's
    o: np.ndarray        # (H, D)


def _lanes(p) -> np.ndarray:
    """(H, BN) -> (H, 4): lane t sums columns 8j + 2t and 8j + 2t + 1, pair by pair, in ascending j."""
    pairs = [FT.fadd(p[:, 8 * j:8 * j + 8:2], p[:, 8 * j + 1:8 * j + 8:2]) for j in range(p.shape[1] // 8)]
    acc = pairs[0]
    for x in pairs[1:]:
        acc = FT.fadd(acc, x)
    return acc


def _pv(o, pb, v) -> np.ndarray:
    """o (H, C) after the k16 steps, rows ascending, of the bf16 P pb (H, BN) against V's columns v (BN, C)."""
    for ks in range(pb.shape[1] // 16):
        o = FT._dot_step(o, pb[:, None, 16 * ks:16 * ks + 16], np.asarray(v[16 * ks:16 * ks + 16], dtype=np.uint16).T[None, :, :])
    return o


def pair(q, k, valid, scale_bits: int, prev: State | None) -> State:
    """`FlashMlaPrefillPair_v1` for every head: q (H, D), k (2 BN, D) staged rows, valid (2 BN,); ``prev`` None on the first pair."""
    BN, H = k.shape[0] // 2, q.shape[1] // 2
    k0, k1 = k[:BN], k[BN:]
    v = np.asarray(valid, dtype=bool)
    s0 = np.where(v[None, :BN], FT.scores(q, k0), U32(FT.F_NINF))
    s1 = np.where(v[None, BN:], FT.scores(np.roll(q, -H, axis=1), np.roll(k1, -H, axis=1)), U32(FT.F_NINF))
    cur = []
    for s in (s0, s1):
        mx = np.full(s.shape[0], FT.F_NINF, dtype=U32)
        for c in range(BN):
            mx = FT.fmaxf(mx, s[:, c])
        cur.append(FT.fmul(mx, scale_bits))
    m = np.full(q.shape[0], FT.F_MAX_INIT, dtype=U32) if prev is None else prev.m
    m0 = FT.fmaxf(m, cur[0])
    m1 = FT.fmaxf(m0, cur[1])
    sf = FT.exp2f(FT.fadd(m0, FT.neg(m1)))
    p0 = FT.exp2f(FT.ffma(s0, scale_bits, FT.neg(m0)[:, None]))
    p1 = FT.exp2f(FT.ffma(s1, scale_bits, FT.neg(m1)[:, None]))
    u0, u1 = _lanes(p0), _lanes(p1)
    if prev is None:
        l0, l1 = FT.fmul(u0, sf[:, None]), u1
        lo = hi = np.zeros((q.shape[0], H), dtype=U32)
    else:
        so0, so1 = FT.exp2f(FT.fadd(m, FT.neg(m0))), FT.exp2f(FT.fadd(m, FT.neg(m1)))
        l0 = FT.fmul(FT.ffma(prev.l[:, :4], so0[:, None], u0), sf[:, None])
        l1 = FT.ffma(prev.l[:, 4:], so1[:, None], u1)
        lo, hi = FT.fmul(prev.o[:, :H], so0[:, None]), FT.fmul(prev.o[:, H:], so1[:, None])
    lo = _pv(FT.fmul(_pv(lo, DR.f2fp_bf16(p0), k0[:, :H]), sf[:, None]), DR.f2fp_bf16(p1), k1[:, :H])
    hi = _pv(_pv(hi, DR.f2fp_bf16(p1), k1[:, H:]), DR.f2fp_bf16(FT.fmul(p0, sf[:, None])), k0[:, H:])
    return State(m1, np.concatenate([l0, l1], axis=1), np.concatenate([lo, hi], axis=1))


def head(q, k, valid, scale_bits: int, BN: int, sink) -> np.ndarray:
    """`FlashMlaPrefillHead_v1` for every head: q (H, D), k (N, D) staged rows, valid (N,), sink (H,) -> bf16 (H, D)."""
    st = None
    for j in range(k.shape[0] // (2 * BN)):
        st = pair(q, k[2 * j * BN:2 * (j + 1) * BN], valid[2 * j * BN:2 * (j + 1) * BN], scale_bits, st)
    lw = [FT.fadd(FT.fadd(st.l[:, 4 * w], st.l[:, 4 * w + 1]), FT.fadd(st.l[:, 4 * w + 2], st.l[:, 4 * w + 3])) for w in (0, 1)]
    total = FT.fadd(lw[0], lw[1])
    den = FT.fadd(total, FT.exp2f(FT.fadd(FT.fmul(sink, FT.L2E), FT.neg(st.m))))
    return DR.f2fp_bf16(FT.fmul(st.o, np.where(FT.feq(total, 0), U32(0), fdiv(FT.F_ONE, den))[:, None]))


def prefill(q, kv, idx, n, skv, sink, *, BN: int, SCALE: float) -> np.ndarray:
    """`FlashMlaPrefill_v1`: q (NH, D) bf16, the N workspace rows (N, D) bf16 and their indices, len, skv, sink (NH,) -> (NH, D) bf16."""
    valid = row_valid(idx, n, skv)
    k = np.where(valid[:, None], np.asarray(kv, dtype=np.uint16), np.uint16(0)).astype(np.uint16)
    return head(np.asarray(q, dtype=np.uint16), k, valid, FT.scale_log2(SCALE), BN, FT._w(sink))


# ---- registration ----------------------------------------------------------------------------------------------------------

def _gather_fn(nope, scale, rope, *, DN, DR, QT):
    return gather_rows(np.asarray(nope).reshape(1, DN), np.asarray(scale).reshape(1, DN // QT), np.asarray(rope).reshape(1, DR),
                       QT).reshape(-1)


def _head_fn(q, k, sink, *, NP, D, BN, DOT, SCALE):
    k = np.asarray(k).reshape(2 * NP * BN, D + 1)
    return head(np.asarray(q).reshape(1, D), k[:, :D], k[:, D], FT.scale_log2(SCALE), BN, FT._w(sink).reshape(1)).reshape(-1)


def _prefill_fn(q, kv, idx, n, skv, sink, *, NH, NP, D, BN, DOT, SCALE):
    return prefill(np.asarray(q).reshape(NH, D), np.asarray(kv).reshape(-1, D), idx, n, skv, sink, BN=BN, SCALE=SCALE).reshape(-1)


def _sample_gather(rng, n, params, *, DN, DR, QT):
    return [FT._e4m3(rng, (n, DN)), FT._e8m0(rng, (n, DN // QT)), FT._bf16_words(rng, (n, DR))]


def _sample_head(rng, n, params, *, NP, D, BN, DOT, SCALE):
    return [FT._bf16_words(rng, (n, D), scale=2.0), FT._staged(rng, n, 2 * NP * BN, D), FT._f32(rng, (n,), scale=2.0, p=0.3)]


def _sample_prefill(rng, n, params, *, NH, NP, D, BN, DOT, SCALE):
    N = 2 * NP * BN
    skv = rng.integers(-1, 64, n)
    idx = np.where(rng.random((n, N)) < 0.9, rng.integers(-2, 66, (n, N)), rng.integers(-(1 << 31), 1 << 31, (n, N)))
    return [FT._bf16_words(rng, (n, NH, D), scale=2.0), FT._bf16_words(rng, (n, N, D), scale=2.0), idx.astype(np.int64).astype(U32),
            rng.integers(-1, N + 2, n).astype(np.int64).astype(U32), skv.astype(np.int64).astype(U32),
            FT._f32(rng, (n, NH), scale=2.0, p=0.3)]


def _targets(defs: dict, family: str) -> list:
    """The specialisations the self-check binds: small statics, one pair and two."""
    from verity.ml.circuits.defs import bind

    d, dot, s = defs, FT.HopperBF16WgmmaDot16, 32 ** -0.5
    blk = dict(D=32, BN=16, DOT=dot, SCALE=s)
    return {
        "FlashMlaGatherRow_v1": [bind(d["FlashMlaGatherRow_v1"], DN=32, DR=16, QT=16)],
        "FlashMlaPrefillHead_v1": [bind(d["FlashMlaPrefillHead_v1"], NP=1, **blk), bind(d["FlashMlaPrefillHead_v1"], NP=2, **blk)],
        "FlashMlaPrefill_v1": [bind(d["FlashMlaPrefill_v1"], NH=2, NP=1, **blk), bind(d["FlashMlaPrefill_v1"], NH=1, NP=2, **blk)],
    }[family]


KERNELS = {
    "FlashMlaGatherRow_v1": (_gather_fn, _sample_gather), "FlashMlaPrefillHead_v1": (_head_fn, _sample_head),
    "FlashMlaPrefill_v1": (_prefill_fn, _sample_prefill),
}


def register_twin(defs: dict) -> None:
    """Register the twin of each family in `KERNELS`; ``defs`` is `registry.flashmla_prefill`'s composites by id, which binds the
    self-check's specialisations (the registry calls this, so neither module imports the other both ways)."""
    for family, (fn, sample) in KERNELS.items():
        register(family, FT._instance(fn), kernel=KERNEL, domain=DOMAIN, sample=sample,
                 check_targets=lambda rng, f=family: _targets(defs, f), covers=FT._hopper)
