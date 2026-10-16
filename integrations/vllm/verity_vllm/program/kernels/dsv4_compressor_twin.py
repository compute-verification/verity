"""numpy twin of `registry.dsv4_compressor` (the head-512 compressor's CuTe DSL kernels, vLLM d9105ea80): the ratio-4 and
ratio-128 columns, the RMSNorm, the UE8M0 quantisation, the RoPE group and the `fp8_ds_mla` row, over numpy words with any
leading batch shape.

The arithmetic is written here again from the kernels' source and SASS rather than read from the registry (FADD / FMUL / FFMA /
div.rn round to nearest with subnormals, every NaN result 0x7FFFFFFF; MUFU.EX2 and MUFU.RSQ through core's measured tables;
F2FP's bf16), and each amax in its kernel's own lane order (ratio 4: per thread of 8 columns from +0, then the group's lanes;
ratio 128: per thread of 2, then its warp), where the registry writes one halving tree, so the self-check compares two writings.
`registry.dsv4_compressor` registers it (`register_twin`) with `kernel_registry` as kernel ``dsv4-compressor-twin`` of each of
its composites.
"""
from __future__ import annotations

import numpy as np

from verity.ml.circuits.ops import mufu as MU
from verity_vllm.program.kernels import derived_rows as DR
from verity_vllm.program.kernels import flashmla_twin as FT
from verity_vllm.program.kernels.fp8_twins import np_f32_to_e4m3_sat
from verity_vllm.program.kernels.kernel_registry import register

U32 = np.uint32
KERNEL = "dsv4-compressor-twin"
DOMAIN = "every word"

L2E, F_NINF, F_PINF = 0x3FB8AA3B, 0xFF800000, 0x7F800000
F_448, F_NEG448, MIN_SCALE = 0x43E00000, 0xC3E00000, 0x38D1B717


def _bits(v: float) -> int:
    return int(np.float32(v).view(U32))


def fsub(a, b) -> np.ndarray:
    a, b = np.broadcast_arrays(FT._w(a), FT._w(b))
    with np.errstate(all="ignore"):
        return FT._gpu(DR.as_u32(DR.as_f32(a) - DR.as_f32(b)))


def fdiv(a, b) -> np.ndarray:
    a, b = np.broadcast_arrays(FT._w(a), FT._w(b))
    with np.errstate(all="ignore"):
        return FT._gpu(DR.div_rn(a, b))


def fminf(a, b) -> np.ndarray:
    """`min.f32`: the smaller, a NaN yields the other operand (two: 0x7FFFFFFF), min(+0, -0) = -0."""
    a, b = np.broadcast_arrays(FT._w(a), FT._w(b))
    na, nb = FT.is_nan(a), FT.is_nan(b)
    zeros = ((a & U32(0x7FFFFFFF)) == 0) & ((b & U32(0x7FFFFFFF)) == 0)
    r = np.where(DR.as_f32(b) < DR.as_f32(a), b, a)
    r = np.where(zeros, a | b, r)
    r = np.where(na, b, np.where(nb, a, r))
    return np.where(na & nb, U32(0x7FFFFFFF), r).astype(U32)


def rsq(x) -> np.ndarray:
    """`rsqrt.approx.ftz.f32`: bare MUFU.RSQ."""
    return FT._w(MU.mufu_rsq_words(FT._w(x), MU.table("rsq")))


def bf16(x) -> np.ndarray:
    return DR.f2fp_bf16(FT._w(x))


def widen(h) -> np.ndarray:
    return np.asarray(h, dtype=U32) << U32(16)


def tree(v: np.ndarray, op) -> np.ndarray:
    """``op`` over the last axis (a power of two) as an xor butterfly leaves it in lane 0."""
    while v.shape[-1] > 1:
        h = v.shape[-1] // 2
        v = op(v[..., :h], v[..., h:])
    return v[..., 0]


# ---- the columns -------------------------------------------------------------------------------------------------------------

def c4_columns(kv, score) -> np.ndarray:
    """(..., ROWS) state words -> (...): the ratio-4 kernel's online softmax and division (cutedsl:259-280)."""
    kv, score = FT._w(kv), FT._w(score)
    m = np.full(kv.shape[:-1], F_NINF, U32)
    l = np.zeros_like(m)
    o = np.zeros_like(m)
    for r in range(kv.shape[-1]):
        s = score[..., r]
        m2 = FT.fmaxf(m, s)
        a = FT.ex2(FT.fmul(fsub(m, m2), L2E))
        b = FT.ex2(FT.fmul(fsub(s, m2), L2E))
        l = FT.ffma(l, a, b)
        o = FT.ffma(o, a, FT.fmul(kv[..., r], b))
        m = m2
    return fdiv(o, l)


def c128_columns(kv, score, W: int, R: int) -> np.ndarray:
    """(..., W*R) state words -> (...): each warp's max, sum and product, then the butterflies across warps
    (cutedsl:1119-1184)."""
    kv = FT._w(kv).reshape(*np.shape(kv)[:-1], W, R)
    score = FT._w(score).reshape(kv.shape)
    m = np.full(kv.shape[:-1], F_NINF, U32)
    for r in range(R):
        m = FT.fmaxf(m, score[..., r])
    sm, pr = np.zeros_like(m), np.zeros_like(m)
    for r in range(R):
        e = FT.ex2(FT.fmul(fsub(score[..., r], m), L2E))
        sm = FT.fadd(e, sm)
        pr = FT.ffma(kv[..., r], e, pr)
    c = FT.ex2(FT.fmul(fsub(m, tree(m, FT.fmaxf)[..., None]), L2E))
    return fdiv(tree(FT.fmul(pr, c), FT.fadd), tree(FT.fmul(sm, c), FT.fadd))


# ---- the row ---------------------------------------------------------------------------------------------------------------

def rms_norm(x, w, ratio: int, QB: int, eps: float) -> np.ndarray:
    """(..., D) words and (D,) or (..., D) bf16 weights -> (..., D): the kernel's sum of squares, rrms and the scaled words."""
    x = FT._w(x)
    D = x.shape[-1]
    if ratio == 4:
        lanes = x.reshape(*x.shape[:-1], D // 8, 8)
        acc = np.zeros(lanes.shape[:-1], U32)
        for e in range(8):
            acc = FT.ffma(lanes[..., e], lanes[..., e], acc)
        width = min(32, D // 8)
        part = tree(acc.reshape(*acc.shape[:-1], -1, width), FT.fadd)
        total = np.zeros(part.shape[:-1], U32)
        for k in range(part.shape[-1]):
            total = FT.fadd(total, part[..., k])
    else:
        p = x.reshape(*x.shape[:-1], D // 2, 2)
        lanes = FT.ffma(p[..., 0], p[..., 0], FT.fmul(p[..., 1], p[..., 1]))
        total = tree(tree(lanes.reshape(*lanes.shape[:-1], D // QB, QB // 2), FT.fadd), FT.fadd)
    rrms = rsq(FT.ffma(total, _bits(1.0 / D), _bits(eps)))
    return FT.fmul(FT.fmul(x, rrms[..., None]), widen(w))


def amax(q, ratio: int) -> np.ndarray:
    """max |q| over the last axis (a group) in kernel ``ratio``'s lane order: ratio 4 per thread of 8 from +0 over its 4 pairs,
    then the group's threads; ratio 128 per thread of 2, then the group's 32 (cutedsl:340-362, 1399-1409)."""
    a = FT._w(q) & U32(0x7FFFFFFF)
    if ratio == 4:
        lanes = a.reshape(*a.shape[:-1], -1, 4, 2)
        acc = np.zeros(lanes.shape[:-2], U32)
        for k in range(4):
            acc = FT.fmaxf(acc, FT.fmaxf(lanes[..., k, 0], lanes[..., k, 1]))
        return tree(acc, FT.fmaxf)
    pairs = a.reshape(*a.shape[:-1], -1, 2)
    return tree(FT.fmaxf(pairs[..., 0], pairs[..., 1]), FT.fmaxf)


def quant_groups(n, ratio: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """(..., QB) normed words -> (e4m3 bytes (..., QB), UE8M0 bytes (...)) (cutedsl:334-389, 1398-1435)."""
    q = widen(bf16(n))
    s = FT.fmaxf(fdiv(amax(q, ratio), F_448), MIN_SCALE)
    e = ((s + U32(0x7FFFFF)) >> U32(23)) & U32(0xFF)
    inv = (U32(254) - e) << U32(23)
    y = fminf(FT.fmaxf(FT.fmul(q, inv[..., None]), F_NEG448), F_448)
    return np_f32_to_e4m3_sat(DR.as_f32(y)), e.astype(np.uint8)


def rope(x, cs) -> np.ndarray:
    """(..., ROPE) normed words and the f32 cache row [cos | sin] -> (..., ROPE) bf16 (cutedsl:316-333, 1387-1397)."""
    x, cs = FT._w(x), FT._w(cs)
    h = x.shape[-1] // 2
    x0, x1, cos, sin = x[..., 0::2], x[..., 1::2], cs[..., :h], cs[..., h:]
    re = FT.ffma(x0, cos, FT.neg(FT.fmul(x1, sin)))
    im = FT.ffma(x1, cos, FT.fmul(x0, sin))
    return bf16(np.stack([re, im], axis=-1).reshape(x.shape))


def norm_rope_quant(x, w, cs, ratio: int, QB: int, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(..., D) words -> the `fp8_ds_mla` row: NoPE bytes (..., D - QB), RoPE bf16 (..., QB), scale bytes (..., D/QB), the
    last the pad 0."""
    n = rms_norm(x, w, ratio, QB, eps)
    D = n.shape[-1]
    groups = n[..., :D - QB].reshape(*n.shape[:-1], -1, QB)
    q, e = quant_groups(groups, ratio)
    pad = np.zeros((*e.shape[:-1], 1), np.uint8)
    return q.reshape(*n.shape[:-1], D - QB), rope(n[..., D - QB:], cs), np.concatenate([e, pad], axis=-1)


def compress_c4(kv, score, w, cs, QB: int, eps: float):
    """(..., ROWS, D) state words -> the ratio-4 row."""
    return norm_rope_quant(c4_columns(np.swapaxes(FT._w(kv), -1, -2), np.swapaxes(FT._w(score), -1, -2)), w, cs, 4, QB, eps)


def compress_c128(kv, score, w, cs, W: int, R: int, QB: int, eps: float):
    """(..., W*R, D) state words -> the ratio-128 row."""
    c = c128_columns(np.swapaxes(FT._w(kv), -1, -2), np.swapaxes(FT._w(score), -1, -2), W, R)
    return norm_rope_quant(c, w, cs, 128, QB, eps)


# ---- registration ----------------------------------------------------------------------------------------------------------

def _flat(*parts) -> np.ndarray:
    return np.concatenate([np.asarray(p).astype(np.uint64).reshape(-1) for p in parts])


def _row_fn(compress):
    def run(kv, score, w, cs, **S):
        D = S["D"]
        return _flat(*compress(np.asarray(kv).reshape(-1, D), np.asarray(score).reshape(-1, D), w, cs, S))
    return run


KERNELS_FN = {
    "Dsv4MlaC4Column_v1": lambda kv, score, *, ROWS: c4_columns(kv, score).reshape(-1),
    "Dsv4MlaC128Column_v1": lambda kv, score, *, W, R: c128_columns(kv, score, W, R).reshape(-1),
    "Dsv4MlaRmsNorm_v1": lambda x, w, *, RATIO, D, QB, EPS: rms_norm(x, w, RATIO, QB, EPS),
    "Dsv4MlaQuantGroup_v1": lambda x, *, QB: _flat(*quant_groups(x)),
    "Dsv4MlaRope_v1": lambda x, cs, *, ROPE: rope(x, cs),
    "Dsv4MlaNormRopeQuant_v1": lambda x, w, cs, *, RATIO, D, QB, EPS: _flat(*norm_rope_quant(x, w, cs, RATIO, QB, EPS)),
    "Dsv4MlaCompressC4_v1": _row_fn(lambda kv, sc, w, cs, S: compress_c4(kv, sc, w, cs, S["QB"], S["EPS"])),
    "Dsv4MlaCompressC128_v1": _row_fn(lambda kv, sc, w, cs, S: compress_c128(kv, sc, w, cs, S["W"], S["R"], S["QB"], S["EPS"])),
}

_SCORE_EDGE = np.array([0xFF800000, 0x7F800000, 0x7FC00000, 0, 0x80000000, 0x7F7FFFFF, 0xFF7FFFFF, 1], dtype=U32)
_X_EDGE = np.array([0x7F800000, 0xFF800000, 0x7FC00000, 0xFFFFFFFF, 0, 0x80000000, 1, 0x80000001, 0x00800000, 0x7F7FFFFF,
                    0x5F800000, 0x1F800000], dtype=U32)


def _state(rng, n, rows, D):
    kv = FT._edges(rng, FT._f32(rng, (n, rows, D), scale=2.0, p=0.0), _X_EDGE, k=1)
    score = FT._edges(rng, FT._f32(rng, (n, rows, D), scale=4.0, p=0.0), _SCORE_EDGE, k=1)
    return kv, score


def _cs(rng, n, k):
    a = rng.uniform(-np.pi, np.pi, (n, k // 2))
    return np.concatenate([np.cos(a), np.sin(a)], axis=1).astype(np.float32).view(U32)


def _row_x(rng, n, D):
    """Rows whose groups span the scale range: each 8-wide stretch at its own magnitude, some groups all zero, a few edges."""
    mag = np.repeat(rng.integers(-40, 40, (n, -(-D // 8))), 8, axis=1)[:, :D]
    x = (rng.standard_normal((n, D)) * 2.0 ** mag).astype(np.float32)
    x[rng.random((n, D)) < 0.05] = 0.0
    return FT._edges(rng, x.view(U32), _X_EDGE, frac=0.3, k=1)


def _sample_c4(rng, n, params, *, ROWS):
    kv, score = _state(rng, n, ROWS, 1)
    return [kv.reshape(n, ROWS), score.reshape(n, ROWS)]


def _sample_c128(rng, n, params, *, W, R):
    kv, score = _state(rng, n, W * R, 1)
    return [kv.reshape(n, W * R), score.reshape(n, W * R)]


def _sample_norm(rng, n, params, *, RATIO, D, QB, EPS):
    return [_row_x(rng, n, D), FT._bf16_words(rng, (n, D))]


def _sample_quant(rng, n, params, *, QB):
    x = _row_x(rng, n, QB)
    x[0] = 0
    if n > 1:
        x[1, :] = 0x7FC00000
    return [x]


def _sample_rope(rng, n, params, *, ROPE):
    return [_row_x(rng, n, ROPE), _cs(rng, n, ROPE)]


def _sample_nrq(rng, n, params, *, RATIO, D, QB, EPS):
    return [_row_x(rng, n, D), FT._bf16_words(rng, (n, D)), _cs(rng, n, QB)]


def _sample_compress(rows):
    def sample(rng, n, params, **S):
        kv, score = _state(rng, n, rows(S), S["D"])
        return [kv, score, FT._bf16_words(rng, (n, S["D"])), _cs(rng, n, S["QB"])]
    return sample


SAMPLES = {
    "Dsv4MlaC4Column_v1": _sample_c4, "Dsv4MlaC128Column_v1": _sample_c128, "Dsv4MlaRmsNorm_v1": _sample_norm,
    "Dsv4MlaQuantGroup_v1": _sample_quant, "Dsv4MlaRope_v1": _sample_rope, "Dsv4MlaNormRopeQuant_v1": _sample_nrq,
    "Dsv4MlaCompressC4_v1": _sample_compress(lambda S: S["ROWS"]),
    "Dsv4MlaCompressC128_v1": _sample_compress(lambda S: S["W"] * S["R"]),
}


def _targets(defs: dict, family: str) -> list:
    """The specialisations the self-check binds: the kernels' columns, norms and groups at the served widths, the composites
    at D = 32."""
    from verity.ml.circuits.defs import bind

    eps = 1e-6
    statics = {
        "Dsv4MlaC4Column_v1": [dict(ROWS=8), dict(ROWS=4)],
        "Dsv4MlaC128Column_v1": [dict(W=8, R=16), dict(W=2, R=3)],
        "Dsv4MlaRmsNorm_v1": [dict(RATIO=r, D=D, QB=QB, EPS=eps) for r in (4, 128) for D, QB in ((512, 64), (32, 8))],
        "Dsv4MlaQuantGroup_v1": [dict(QB=64), dict(QB=8)],
        "Dsv4MlaRope_v1": [dict(ROPE=64)],
        "Dsv4MlaNormRopeQuant_v1": [dict(RATIO=r, D=512, QB=64, EPS=eps) for r in (4, 128)],
        "Dsv4MlaCompressC4_v1": [dict(ROWS=rows, D=32, QB=16, EPS=eps) for rows in (8, 4)],
        "Dsv4MlaCompressC128_v1": [dict(W=2, R=4, D=32, QB=16, EPS=eps)],
    }[family]
    return [bind(defs[family], **st) for st in statics]


def register_twin(defs: dict) -> None:
    """Register the twin of each family in `KERNELS_FN`; ``defs`` is `registry.dsv4_compressor`'s composites by id, which binds
    the self-check's specialisations (the registry calls this, so neither module imports the other both ways)."""
    for family, fn in KERNELS_FN.items():
        register(family, fn, kernel=KERNEL, domain=DOMAIN, sample=SAMPLES[family],
                 check_targets=lambda rng, f=family: _targets(defs, f))
