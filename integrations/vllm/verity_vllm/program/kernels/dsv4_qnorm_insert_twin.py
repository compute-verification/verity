"""numpy twin of `registry.dsv4_qnorm_insert` (DeepSeek-V4-Pro's query norm, RoPE and SWA cache insert, and the indexer's query
RoPE and FP8 quantisation, on the H200), and the fp8_ds_mla cache layout the insert writes.

`insert` computes `Dsv4QNormRopeKvInsert_v1` over numpy words for many tokens at once, `indexer_q` `Dsv4IndexerQ_v1`; `q_heads` and
`kv_rows` are the insert's parts.  The arithmetic is written here again from the kernels' SASS rather than read from the registry
(FADD / FMUL / FFMA / `div.rn` round to nearest with subnormals, every NaN result 0x7FFFFFFF; MUFU.RSQ and the libm pieces through
`derived_rows` and `flashmla_twin`), so the self-check compares two writings.  `cache_rows` and `scatter` are the bytes the insert
stores: per cache block, token p's 448 E4M3 bytes and 64 bf16 at p * 576, its 8 scale bytes at block_size * 576 + p * 8.

`registry.dsv4_qnorm_insert` registers it (`register_twin`) with `kernel_registry` as kernel ``dsv4-qnorm-insert-twin`` of
`Dsv4QNormRope_v1`, `Dsv4KvRopeQuant_v1`, `Dsv4QNormRopeKvInsert_v1`, `Dsv4IndexerQHead_v1` and `Dsv4IndexerQ_v1`.
"""
from __future__ import annotations

import numpy as np

from verity_vllm.program.kernels import derived_rows as DW
from verity_vllm.program.kernels import flashmla_twin as FT
from verity_vllm.program.kernels.fp8_twins import np_f32_to_e4m3_sat
from verity_vllm.program.kernels.kernel_registry import register

U32 = np.uint32
KERNEL = "dsv4-qnorm-insert-twin"
DOMAIN = "every word"
LANES = 32
GPU_NAN = 0x7FFFFFFF
F_1EM4, F_448, F_NEG448, F_127, F_255, F_INV448 = 0x38D1B717, 0x43E00000, 0xC3E00000, 0x42FE0000, 0x437F0000, 0x3B124925
#: the fp8_ds_mla row: 448 E4M3 + 64 bf16 data bytes, then (after the block's data) 7 scale bytes and a pad byte per token
DATA_BYTES, SCALE_BYTES = 448 + 2 * 64, 8

_w, fadd, fmul, ffma, neg, fmaxf, is_nan = FT._w, FT.fadd, FT.fmul, FT.ffma, FT.neg, FT.fmaxf, FT.is_nan


def f32(v: float) -> int:
    return int(np.float32(v).view(U32))


def fminf(a, b) -> np.ndarray:
    """`min.f32`: the smaller, a NaN yields the other operand (two: 0x7FFFFFFF), min(+0, -0) = -0."""
    a, b = np.broadcast_arrays(_w(a), _w(b))
    na, nb = is_nan(a), is_nan(b)
    zeros = ((a & U32(0x7FFFFFFF)) == 0) & ((b & U32(0x7FFFFFFF)) == 0)
    r = np.where(FT._f64(b) < FT._f64(a), b, a)
    r = np.where(zeros, a | b, r)
    r = np.where(na, b, np.where(nb, a, r))
    return np.where(na & nb, U32(GPU_NAN), r).astype(U32)


def fdiv(a, b) -> np.ndarray:
    a, b = np.broadcast_arrays(_w(a), _w(b))
    with np.errstate(all="ignore"):
        return FT._gpu(DW.div_rn(a, b))


def ceil_ftz(x) -> np.ndarray:
    """`cvt.rpi.ftz.f32.f32`: a subnormal flushed to its signed zero, then rounded up; a NaN 0x7FFFFFFF."""
    x = _w(x)
    sub = ((x & U32(0x7F800000)) == 0)
    xf = np.where(sub, x & U32(0x80000000), x).astype(U32).view(np.float32)
    with np.errstate(all="ignore"):
        r = np.ceil(xf).astype(np.float32).view(U32)
    r = np.where((xf > -1) & (xf < 0), U32(0x80000000), r)
    return np.where(is_nan(x), U32(GPU_NAN), r).astype(U32)


def u32_rz_byte(x) -> np.ndarray:
    """The low byte of `cvt.rzi.u32.f32`: a NaN and every negative 0, from 2^32 (and +inf) 0xFF, else the truncation's low byte."""
    x = _w(x)
    e = ((x >> U32(23)) & U32(0xFF)).astype(np.int64)
    m = ((x & U32(0x7FFFFF)) | U32(0x800000)).astype(np.int64)
    k = e - 127
    v = np.where(k >= 23, m << np.clip(k - 23, 0, 40), m >> np.clip(23 - k, 0, 63)) & 0xFF
    v = np.where(e >= 127 + 32, 0xFF, np.where(e < 127, 0, v))
    return np.where(is_nan(x) | (x >> U32(31) == 1), 0, v).astype(np.uint8)


def e4m3(x) -> np.ndarray:
    return np_f32_to_e4m3_sat(_w(x).view(np.float32))


def _rope(x: np.ndarray, cs: np.ndarray, DR: int) -> np.ndarray:
    """The last DR elements of x (..., D) rotated by cs (..., DR), broadcast over x's leading axes."""
    x = x.copy()
    half, base = DR // 2, x.shape[-1] - DR
    xe, xo = x[..., base::2].copy(), x[..., base + 1::2].copy()
    c, s = cs[..., :half], cs[..., half:]
    x[..., base::2] = ffma(xe, c, neg(fmul(xo, s)))
    x[..., base + 1::2] = ffma(xe, s, fmul(xo, c))
    return x


def q_heads(q, cs, *, DR: int, EPS: float) -> np.ndarray:
    """`Dsv4QNormRope_v1` of q (..., NH, D) bf16 with cs (..., DR) f32: (..., NH, D) bf16."""
    x = FT._bf16(q)
    D_ = x.shape[-1]
    L = D_ // LANES
    lanes = x.reshape(x.shape[:-1] + (LANES, L))
    acc = fmul(lanes[..., 0], lanes[..., 0])
    for i in range(1, L):
        acc = ffma(lanes[..., i], lanes[..., i], acc)
    for h in (16, 8, 4, 2, 1):
        acc = fadd(acc[..., :h], acc[..., h:2 * h])
    r = DW.rsqrt_approx(ffma(acc[..., 0], f32(1.0 / D_), f32(EPS)))
    n = fmul(x, r[..., None])
    return DW.f2fp_bf16(_rope(n, _w(cs)[..., None, :], DR))


def kv_rows(kv, cs, *, DR: int, QB: int):
    """`Dsv4KvRopeQuant_v1` of kv (..., DN + DR) bf16 with cs (..., DR) f32: nope (..., DN) e4m3, scale (..., DN/QB + 1) bytes,
    rope (..., DR) bf16."""
    x = _rope(FT._bf16(kv), _w(cs), DR)
    DN = x.shape[-1] - DR
    blk = x[..., :DN].reshape(x.shape[:-1] + (DN // QB, QB))
    amax = np.zeros(blk.shape[:-1], dtype=U32)
    for i in range(QB):
        amax = fmaxf(amax, blk[..., i] & U32(0x7FFFFFFF))
    e = ceil_ftz(FT.log2f(fdiv(fmaxf(amax, F_1EM4), F_448)))
    inv = FT.exp2f(neg(e))
    nope = e4m3(fminf(fmaxf(fmul(blk, inv[..., None]), F_NEG448), F_448)).reshape(x.shape[:-1] + (DN,))
    byte = u32_rz_byte(fmaxf(fminf(fadd(e, F_127), F_255), 0))
    scale = np.concatenate([byte, np.zeros(byte.shape[:-1] + (1,), dtype=np.uint8)], axis=-1)
    return nope, scale, DW.f2fp_bf16(x[..., DN:])


def insert(q, kv, cs, *, NHP: int, DR: int, QB: int, EPS: float):
    """`Dsv4QNormRopeKvInsert_v1` of q (..., NH, D), kv (..., D) bf16 and cs (..., DR) f32: q_out (..., NHP, D) bf16 with heads NH ..
    NHP - 1 zero, and `kv_rows`."""
    q = np.asarray(q, dtype=np.uint16)
    out = np.zeros(q.shape[:-2] + (NHP, q.shape[-1]), dtype=np.uint16)
    out[..., :q.shape[-2], :] = q_heads(q, cs, DR=DR, EPS=EPS)
    return (out, *kv_rows(kv, cs, DR=DR, QB=QB))


def indexer_q(q, cs, w, *, DR: int, SCALE: float):
    """`Dsv4IndexerQ_v1` of q (..., H, D) bf16, cs (..., DR) f32 and w (..., H) bf16: q (..., H, D) e4m3 and w (..., H) f32."""
    x = _rope(FT._bf16(q), _w(cs)[..., None, :], DR)
    D_ = x.shape[-1]
    x[..., D_ - DR:] = FT._bf16(DW.f2fp_bf16(x[..., D_ - DR:]))
    amax = np.zeros(x.shape[:-1], dtype=U32)
    for i in range(D_):
        amax = fmaxf(amax, x[..., i] & U32(0x7FFFFFFF))
    sb = ((fmul(fmaxf(amax, F_1EM4), F_INV448) + U32(0x7FFFFF)) & U32(0x7F800000)).astype(U32)
    inv = (U32(0x7F000000) - sb).astype(U32)
    return e4m3(fmul(x, inv[..., None])), fmul(fmul(FT._bf16(w), f32(SCALE)), sb)


# ---- the cache layout -------------------------------------------------------------------------------------------------------

def cache_rows(nope, scale, rope) -> tuple[np.ndarray, np.ndarray]:
    """Each token's stored bytes: its data bytes (DN e4m3, then DR bf16 little-endian) and its scale bytes."""
    nope = np.asarray(nope, dtype=np.uint8)
    rope = np.ascontiguousarray(np.asarray(rope, dtype=np.uint16)).view(np.uint8)
    return np.concatenate([nope, rope], axis=-1), np.asarray(scale, dtype=np.uint8)


def scatter(cache: np.ndarray, slots, data, scale, block_size: int) -> np.ndarray:
    """A copy of cache (num_blocks, block_stride bytes) with token t's `cache_rows` stored at slots[t] (a slot < 0 stores
    nothing): its data at (slot % block_size) * D bytes of block slot / block_size, its scale bytes after the block's data."""
    out = np.array(cache, dtype=np.uint8)
    D_, S_ = data.shape[-1], scale.shape[-1]
    for t, slot in enumerate(np.asarray(slots, dtype=np.int64).tolist()):
        if slot < 0:
            continue
        b, p = divmod(slot, block_size)
        out[b, p * D_:(p + 1) * D_] = data[t]
        out[b, block_size * D_ + p * S_:block_size * D_ + (p + 1) * S_] = scale[t]
    return out


# ---- registration -----------------------------------------------------------------------------------------------------------

def _q_fn(q, cs, *, D, DR, EPS):
    return q_heads(np.asarray(q).reshape(1, D), np.asarray(cs).reshape(DR), DR=DR, EPS=EPS).reshape(-1)


def _kv_fn(kv, cs, *, DN, DR, QB):
    return np.concatenate([np.asarray(a, dtype=np.uint64).reshape(-1)
                           for a in kv_rows(np.asarray(kv).reshape(DN + DR), np.asarray(cs).reshape(DR), DR=DR, QB=QB)])


def _insert_fn(q, kv, cs, *, NH, NHP, DN, DR, QB, EPS):
    out = insert(np.asarray(q).reshape(NH, DN + DR), np.asarray(kv).reshape(DN + DR), np.asarray(cs).reshape(DR), NHP=NHP,
                 DR=DR, QB=QB, EPS=EPS)
    return np.concatenate([np.asarray(a, dtype=np.uint64).reshape(-1) for a in out])


def _head_fn(q, cs, w, *, D, DR, SCALE):
    qq, ww = indexer_q(np.asarray(q).reshape(1, D), np.asarray(cs).reshape(DR), np.asarray(w).reshape(1), DR=DR, SCALE=SCALE)
    return np.concatenate([qq.reshape(-1).astype(np.uint64), ww.reshape(-1).astype(np.uint64)])


def _token_fn(q, cs, w, *, H, D, DR, SCALE):
    qq, ww = indexer_q(np.asarray(q).reshape(H, D), np.asarray(cs).reshape(DR), np.asarray(w).reshape(H), DR=DR, SCALE=SCALE)
    return np.concatenate([qq.reshape(-1).astype(np.uint64), ww.reshape(-1).astype(np.uint64)])


_BF16_EDGE = np.array([0, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x0001, 0x8001, 0x7F7F, 0xFF7F, 0x0080], dtype=np.uint16)
_F32_EDGE = np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 1, 0x3F800000, 0xBF800000, 0x7F7FFFFF], dtype=U32)


def _bf16(rng, shape, p: float = 0.3) -> np.ndarray:
    """bf16 words over many binades (each row scaled by one power of two, so a block's amax walks the scale bytes), about p of
    the rows with three edge words (NaN, infinities, subnormals, the largest finite)."""
    rows = rng.standard_normal(shape) * 2.0 ** rng.integers(-80, 72, shape[:-1] + (1,)) * 2.0 ** rng.integers(-3, 4, shape)
    a = ((rows.astype(np.float32).view(U32) >> U32(16)).astype(np.uint16))
    flat = a.reshape(-1, shape[-1])
    for i in np.flatnonzero(rng.random(flat.shape[0]) < p):
        flat[i, rng.integers(0, shape[-1], 3)] = rng.choice(_BF16_EDGE, 3)
    return a


def _cs(rng, shape, p: float = 0.05) -> np.ndarray:
    """A cos_sin row, (cos(pos f_p), sin(pos f_p)), its entries an edge word with probability p."""
    DR = shape[-1]
    ang = rng.uniform(0, 1 << 16, shape[:-1] + (1,)) * 10000.0 ** (-np.arange(DR // 2) / (DR // 2))
    v = np.concatenate([np.cos(ang), np.sin(ang)], axis=-1).astype(np.float32).view(U32)
    return np.where(rng.random(shape) < p, rng.choice(_F32_EDGE, shape), v).astype(U32)


def _sample_q(rng, n, params, *, D, DR, EPS):
    return [_bf16(rng, (n, D)), _cs(rng, (n, DR))]


def _sample_kv(rng, n, params, *, DN, DR, QB):
    return [_bf16(rng, (n, DN + DR)), _cs(rng, (n, DR))]


def _sample_insert(rng, n, params, *, NH, NHP, DN, DR, QB, EPS):
    return [_bf16(rng, (n, NH, DN + DR)), _bf16(rng, (n, DN + DR)), _cs(rng, (n, DR))]


def _sample_head(rng, n, params, *, D, DR, SCALE):
    return [_bf16(rng, (n, D)), _cs(rng, (n, DR)), _bf16(rng, (n, 1)).reshape(n)]


def _sample_token(rng, n, params, *, H, D, DR, SCALE):
    return [_bf16(rng, (n, H, D)), _cs(rng, (n, DR)), _bf16(rng, (n, H))]


def _targets(defs: dict, family: str) -> list:
    """The specialisations the self-check binds: DeepSeek-V4-Pro's head and row, a small token of each kernel, and an eps that
    dominates the sum of squares."""
    from verity.ml.circuits.defs import bind

    d, s = defs, 128 ** -0.5 * 64 ** -0.5
    return {
        "Dsv4QNormRope_v1": [bind(d["Dsv4QNormRope_v1"], D=512, DR=64, EPS=1e-6),
                             bind(d["Dsv4QNormRope_v1"], D=64, DR=16, EPS=0.5)],
        "Dsv4KvRopeQuant_v1": [bind(d["Dsv4KvRopeQuant_v1"], DN=448, DR=64, QB=64),
                               bind(d["Dsv4KvRopeQuant_v1"], DN=48, DR=16, QB=16)],
        "Dsv4QNormRopeKvInsert_v1": [bind(d["Dsv4QNormRopeKvInsert_v1"], NH=2, NHP=4, DN=48, DR=16, QB=16, EPS=1e-6)],
        "Dsv4IndexerQHead_v1": [bind(d["Dsv4IndexerQHead_v1"], D=128, DR=64, SCALE=s),
                                bind(d["Dsv4IndexerQHead_v1"], D=32, DR=16, SCALE=0.75)],
        "Dsv4IndexerQ_v1": [bind(d["Dsv4IndexerQ_v1"], H=4, D=32, DR=16, SCALE=s)],
    }[family]


KERNELS = {
    "Dsv4QNormRope_v1": (_q_fn, _sample_q), "Dsv4KvRopeQuant_v1": (_kv_fn, _sample_kv),
    "Dsv4QNormRopeKvInsert_v1": (_insert_fn, _sample_insert), "Dsv4IndexerQHead_v1": (_head_fn, _sample_head),
    "Dsv4IndexerQ_v1": (_token_fn, _sample_token),
}


def register_twin(defs: dict) -> None:
    """Register the twin of each family in `KERNELS`; ``defs`` is `registry.dsv4_qnorm_insert`'s composites by id, which binds the
    self-check's specialisations (the registry calls this, so neither module imports the other both ways)."""
    for family, (fn, sample) in KERNELS.items():
        register(family, fn, kernel=KERNEL, domain=DOMAIN, sample=sample, check_targets=lambda rng, f=family: _targets(defs, f))
