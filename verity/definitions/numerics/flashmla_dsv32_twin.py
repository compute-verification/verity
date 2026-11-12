"""numpy twin of FlashMLA's sparse FP8 decode at its V32 instance (6bc49418: `flash_fwd_splitkv_mla_fp8_sparse_kernel<
KernelTemplate<V32, 64>>` and its combine, on the H200), over `flashmla_twin`'s parts.

The instance differs from the one `flashmla_twin` was written for in what it reads and how it is called, not in the block's
arithmetic, the split's end or the combine: a cache row is 512 E4M3 bytes, four f32 scales (one per 128 elements, each rounded
to bf16 before it multiplies) and 64 bf16 words; a row is valid when its index is not -1 and nothing else; the key is 576 wide
and the value its first 512 words; there is no sink (the decode kernel runs its sink term on -inf, the combine skips its own)
and no second cache.  `kv_rows` is the staged row, `decode` a token's heads at its SPLITS, `step_splits` the SPLITS a step's
token count gives.

`registry.flashmla_dsv32` registers it (`register_twin`) as kernel ``flashmla-twin`` of `FlashMlaV32KvRow_v1`,
`FlashMlaV32Block_v1`, `FlashMlaV32SplitEnd_v1`, `FlashMlaV32CombineWeights_v1`, `FlashMlaV32CombineValues_v1`,
`FlashMlaV32DecodeHead_v1` and `FlashMlaV32Decode_v1` (``DOT`` the Hopper wgmma k16 step).
"""
from __future__ import annotations

import numpy as np

from verity.definitions.kernels import tc as CK
from verity.definitions.numerics import derived_rows as DR
from verity.definitions.numerics import flashmla_twin as T
from verity.definitions.numerics.kernel_registry import register
from verity.definitions.ops.prims import HopperBF16WgmmaDot16

U32 = np.uint32


def row_valid(idx) -> np.ndarray:
    return np.asarray(idx, dtype=np.int64).astype(U32) != U32(0xFFFFFFFF)


def kv_rows(nope, scale, rope, valid, QT: int) -> np.ndarray:
    """The staged rows: (N, DN) e4m3 bytes, (N, DN / QT) f32 scale words, (N, DR) bf16 words, (N,) valid -> (N, DN + DR) bf16.

    An invalid row's elements are +0 (its bytes and its scales are zeroed before the multiply).  Its last DR words are whatever
    row the masked index addresses; the score they enter is replaced by -inf and the value product never reads them, so no
    output word depends on them and they are 0 here.
    """
    rope = np.asarray(rope, dtype=np.uint16)
    x = DR.f2fp_bf16(CK.e4m3_to_f32_batch(np.asarray(nope))[0])
    s = np.repeat(DR.f2fp_bf16(np.asarray(scale, dtype=U32)), QT, axis=-1)
    el = DR.f2fp_bf16(T.fmul(T._bf16(x), T._bf16(s)))
    rows = np.concatenate([el, rope], axis=-1)
    return np.where(np.asarray(valid, dtype=bool)[..., None], rows, np.uint16(0)).astype(np.uint16)


def split_end(st: T.State, DV: int, partial: bool):
    """A split's end on no sink from its last block's state: bf16 (H, DV), or (``partial``) the partial (oacc (H, DV), lse (H,))."""
    total = T.fadd(T.fadd(st.l[:, 0], st.l[:, 1]), T.fadd(st.l[:, 2], st.l[:, 3]))
    empty = T.feq(total, 0)
    o = T._w(st.o)[:, :DV]
    if partial:
        oacc = T.fmul(o, np.where(empty, U32(0), T.rcp1(total))[:, None])
        return oacc, np.where(empty, U32(T.F_NINF), T.fadd(T.log2f(total), st.m)).astype(U32)
    den = T.fadd(total, T.exp2f(T.fadd(T.fmul(U32(T.F_NINF), T.L2E), T.neg(st.m))))
    return DR.f2fp_bf16(T.fmul(o, np.where(empty, U32(0), T.rcp1(den))[:, None]))


def combine_weights(lse) -> np.ndarray:
    """The weights of every head's partials on no sink: lse (H, NS) -> (H, NS) f32, split s weighing exp2f(lse[s] - g)."""
    lse = T._w(lse)
    return T.exp2f(T.fadd(lse, T.combine_scale(lse)[:, None]))


def head(q, k, valid, *, SPLITS, BN: int, DV: int, SCALE: float) -> np.ndarray:
    """q (NH, DK) bf16 over the token's staged rows k (N, DK) bf16 and their validity (N,) -> (NH, DV) bf16.  One split writes
    bf16 itself, its sink term on -inf; more leave partials, which the combine weighs with no sink term."""
    sp = tuple(SPLITS)
    q, sc = np.asarray(q, dtype=np.uint16), T.scale_log2(SCALE)
    if len(sp) == 2:
        return T.split(q, k, valid, sc, BN, np.full(q.shape[0], T.F_NINF, dtype=U32))[:, :DV]
    parts = [T.split(q, k[a * BN:b * BN], valid[a * BN:b * BN], sc, BN) for a, b in zip(sp, sp[1:])]
    return T.combine_values(combine_weights(np.stack([p[1] for p in parts], axis=1)), np.stack([p[0][:, :DV] for p in parts], axis=1))


def decode(q, nope, scale, rope, idx, *, SPLITS, QT: int, BN: int, DV: int, SCALE: float) -> np.ndarray:
    """q (NH, DK) bf16 and the token's N rows in index order (nope (N, DN), scale (N, DN / QT) f32 words, rope (N, DR), idx (N,))
    -> (NH, DV) bf16."""
    valid = row_valid(idx)
    return head(q, kv_rows(nope, scale, rope, valid, QT), valid, SPLITS=SPLITS, BN=BN, DV=DV, SCALE=SCALE)


def step_splits(tokens: int, *, topk: int, h_q: int = 64, BN: int = T.BLOCK) -> tuple[int, ...]:
    """SPLITS of every token of a step of ``tokens`` query rows: the call is one request of s_q = tokens rows over ``topk`` cache
    rows, which the planner cuts over num_sm_parts = max(132 / tokens / (h_q / 64), 1) partitions (`flashmla_twin.sched_meta`)."""
    metas, ns = T.sched_meta([topk], topk=topk, parts=T.num_sm_parts(h_q, s_q=tokens))
    return T.request_splits(metas, ns, 0, topk // BN)


# ---- registration ----------------------------------------------------------------------------------------------------------

def _rows(k, DK: int):
    k = np.asarray(k).reshape(-1, DK + 1)
    return k[:, :DK].astype(np.uint16), k[:, DK].astype(bool)


def _row_fn(nope, scale, rope, idx, *, DN, DR, QT):
    valid = row_valid(np.asarray(idx).reshape(1))
    row = kv_rows(np.asarray(nope).reshape(1, DN), np.asarray(scale).reshape(1, DN // QT), np.asarray(rope).reshape(1, DR), valid, QT)
    return np.concatenate([row.reshape(-1), valid.astype(np.uint16)])


def _block_fn(q, k, *state, DK, DV, BN, FIRST, DOT, SCALE):
    rows, valid = _rows(k, DK)
    prev = None
    if not FIRST:
        o = np.zeros((1, DK), dtype=U32)
        o[0, :DV] = T._w(state[2]).reshape(DV)
        prev = T.State(T.neg(T._w(state[0]).reshape(1)), T._w(state[1]).reshape(1, 4), o)
    st = T.block(np.asarray(q, dtype=np.uint16).reshape(1, DK), rows, valid, T.scale_log2(SCALE), prev)
    return np.concatenate([T.neg(st.m).reshape(-1), st.l.reshape(-1), st.o[0, :DV]])


def _end_fn(nm, l, o, *, DV, PARTIAL):
    out = split_end(T.State(T.neg(T._w(nm).reshape(1)), T._w(l).reshape(1, 4), T._w(o).reshape(1, DV)), DV, PARTIAL)
    return np.concatenate([out[0].reshape(-1), out[1].reshape(-1)]) if PARTIAL else out.reshape(-1)


def _weights_fn(lse, *, NS):
    return combine_weights(T._w(lse).reshape(1, NS)).reshape(-1)


def _values_fn(w, oacc, *, NS, G):
    return T.combine_values(T._w(w).reshape(1, NS), T._w(oacc).reshape(1, NS, G)).reshape(-1)


def _head_fn(q, k, *, SPLITS, DK, DV, BN, DOT, SCALE):
    rows, valid = _rows(k, DK)
    return head(np.asarray(q).reshape(1, DK), rows, valid, SPLITS=SPLITS, BN=BN, DV=DV, SCALE=SCALE).reshape(-1)


def _decode_fn(q, nope, scale, rope, idx, *, NH, SPLITS, DN, DR, QT, BN, DOT, SCALE):
    return decode(np.asarray(q).reshape(NH, DN + DR), np.asarray(nope).reshape(-1, DN), np.asarray(scale).reshape(-1, DN // QT),
                  np.asarray(rope).reshape(-1, DR), idx, SPLITS=SPLITS, QT=QT, BN=BN, DV=DN, SCALE=SCALE).reshape(-1)


#: f32 scales a row's sampler plants: zeros, infinities, NaNs, the smallest normal, a subnormal, and words whose low half is at,
#: just under and just over the tie of the rounding to bf16
_SCALE_EDGE = np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x00800000, 0x00000001, 0x3F808000, 0x3F818000,
                        0x3F807FFF, 0x3F808001, 0x7F7FFFFF, 0x7F7F8000], dtype=U32)
_INDEX_EDGE = np.array([0, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE], dtype=U32)


def _scales(rng, shape) -> np.ndarray:
    """f32 scale words: 2^e (1 + m) with e in [-20, 8), a quarter of them with the low 16 bits at the tie of the rounding to
    bf16 or one word to either side of it."""
    f = (2.0 ** rng.integers(-20, 8, shape) * (1.0 + rng.random(shape))).astype(np.float32).view(U32)
    f = np.where(rng.random(shape) < 0.25, (f & U32(0xFFFF0000)) | rng.choice(np.array([0x8000, 0x7FFF, 0x8001], dtype=U32), shape),
                 f).astype(U32)
    return T._edges(rng, f, _SCALE_EDGE)


def _indices(rng, shape) -> np.ndarray:
    """Indices: -1 in about one of seven, and the words next to it, 0 and the ends of the signed range among the others."""
    idx = np.where(rng.random(shape) < 0.85, rng.integers(0, 1 << 20, shape), -1).astype(np.int64).astype(U32)
    return np.where(rng.random(shape) < 0.05, rng.choice(_INDEX_EDGE, shape), idx).astype(U32)


def _sample_row(rng, n, params, *, DN, DR, QT):
    return [T._e4m3(rng, (n, DN)), _scales(rng, (n, DN // QT)), T._bf16_words(rng, (n, DR)), _indices(rng, (n,))]


def _state(rng, n, DV):
    """A carried state: -m for a running max m, the four threads' sums and O, edge words planted in a share of them and the
    sums all zero in a sixth (a split with no valid row so far)."""
    nm = T.neg(T._f32(rng, (n,), scale=8.0, p=0.1))
    l = np.abs(rng.standard_normal((n, 4)) * 4.0).astype(np.float32).view(U32)
    l = np.where(rng.random((n, 4)) < 0.1, rng.choice(T._F32_EDGE, (n, 4)), l).astype(U32)
    l[: max(1, n // 6)] = 0
    return [nm, l, T._edges(rng, T._f32(rng, (n, DV), scale=1.0, p=0.0), T._F32_EDGE, k=1)]


def _sample_block(rng, n, params, *, DK, DV, BN, FIRST, DOT, SCALE):
    return [T._bf16_words(rng, (n, DK), scale=2.0), T._staged(rng, n, BN, DK)] + ([] if FIRST else _state(rng, n, DV))


def _sample_end(rng, n, params, *, DV, PARTIAL):
    return _state(rng, n, DV)


def _sample_weights(rng, n, params, *, NS):
    return T._sample_combine(rng, n, params, NS=NS, D=1)[:1]


def _sample_values(rng, n, params, *, NS, G):
    """Weights as the combine gives them (exp2f of something at most 0, a tenth of them 0) with edge words planted, and
    partials."""
    w = np.exp2(-np.abs(rng.standard_normal((n, NS)) * 4.0)).astype(np.float32).view(U32)
    w = np.where(rng.random((n, NS)) < 0.1, U32(0), w).astype(U32)
    return [T._edges(rng, w, T._F32_EDGE, k=1), T._edges(rng, T._f32(rng, (n, NS, G), scale=1.0, p=0.0), T._F32_EDGE, k=1)]


def _sample_head(rng, n, params, *, SPLITS, DK, DV, BN, DOT, SCALE):
    return [T._bf16_words(rng, (n, DK), scale=2.0), T._staged(rng, n, tuple(SPLITS)[-1] * BN, DK)]


def _sample_decode(rng, n, params, *, NH, SPLITS, DN, DR, QT, BN, DOT, SCALE):
    N = tuple(SPLITS)[-1] * BN
    idx = _indices(rng, (n, N))
    idx[: max(1, n // 6)] = U32(0xFFFFFFFF)
    return [T._bf16_words(rng, (n, NH, DN + DR), scale=2.0), T._e4m3(rng, (n, N, DN)), _scales(rng, (n, N, DN // QT)),
            T._bf16_words(rng, (n, N, DR), scale=2.0), idx]


def _targets(defs: dict, family: str) -> list:
    """The specialisations the self-check binds: small statics, the value narrower than the key, every arm of FIRST, PARTIAL
    and SPLITS (one split, two, and three of unequal lengths), and the combine at 3 and 33 splits (one and two partials on a
    lane)."""
    from verity.circuits.defs import bind

    d, dot, s = defs, HopperBF16WgmmaDot16, 0.25
    blk = dict(DK=48, DV=32, BN=16, DOT=dot, SCALE=s)
    dec = dict(NH=2, DN=32, DR=16, QT=16, BN=16, DOT=dot, SCALE=s)
    return {
        "FlashMlaV32KvRow_v1": [bind(d["FlashMlaV32KvRow_v1"], DN=32, DR=16, QT=16)],
        "FlashMlaV32Block_v1": [bind(d["FlashMlaV32Block_v1"], FIRST=True, **blk), bind(d["FlashMlaV32Block_v1"], FIRST=False, **blk)],
        "FlashMlaV32SplitEnd_v1": [bind(d["FlashMlaV32SplitEnd_v1"], DV=32, PARTIAL=False),
                                   bind(d["FlashMlaV32SplitEnd_v1"], DV=32, PARTIAL=True)],
        "FlashMlaV32CombineWeights_v1": [bind(d["FlashMlaV32CombineWeights_v1"], NS=3), bind(d["FlashMlaV32CombineWeights_v1"], NS=33)],
        "FlashMlaV32CombineValues_v1": [bind(d["FlashMlaV32CombineValues_v1"], NS=3, G=16), bind(d["FlashMlaV32CombineValues_v1"], NS=33, G=2)],
        "FlashMlaV32DecodeHead_v1": [bind(d["FlashMlaV32DecodeHead_v1"], SPLITS=(0, 2), **blk),
                                     bind(d["FlashMlaV32DecodeHead_v1"], SPLITS=(0, 1, 2), **blk),
                                     bind(d["FlashMlaV32DecodeHead_v1"], SPLITS=(0, 2, 3, 4), **blk)],
        "FlashMlaV32Decode_v1": [bind(d["FlashMlaV32Decode_v1"], SPLITS=(0, 1, 2), **dec),
                                 bind(d["FlashMlaV32Decode_v1"], SPLITS=(0, 2), **dec)],
    }[family]


KERNELS = {"FlashMlaV32KvRow_v1": (_row_fn, _sample_row), "FlashMlaV32Block_v1": (_block_fn, _sample_block),
           "FlashMlaV32SplitEnd_v1": (_end_fn, _sample_end), "FlashMlaV32CombineWeights_v1": (_weights_fn, _sample_weights),
           "FlashMlaV32CombineValues_v1": (_values_fn, _sample_values), "FlashMlaV32DecodeHead_v1": (_head_fn, _sample_head),
           "FlashMlaV32Decode_v1": (_decode_fn, _sample_decode)}


def register_twin(defs: dict) -> None:
    """Register the twin of each family in `KERNELS`; ``defs`` is `registry.flashmla_dsv32`'s composites by id, which binds the
    self-check's specialisations (the registry calls this, so neither module imports the other both ways)."""
    for family, (fn, sample) in KERNELS.items():
        register(family, T._instance(fn), kernel=T.KERNEL, domain=T.DOMAIN, sample=sample,
                 check_targets=lambda rng, f=family: _targets(defs, f), covers=T._hopper)
