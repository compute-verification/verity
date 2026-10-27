"""The replay row kernels of the exact-int8 profile's Definitions (`registry.int8_exact`, `ml.circuits.ops.gemm`), each the Calls
the int8 ops are folded into (`observe.fold.patterns.int8`):

    Int8RowQuant_v1{K}                (x: bf16[K]) -> (codes s8[K], s f32)                    verity_int8::row_quant
    ScaledMmS8_v1{K,N,DOT}            (x: s8[K], sx: f32, w: s8[N][K], sw: f32[N]) -> bf16[N]   verity_int8::scaled_mm
    Int8KvQuant_v1{S,D}               (k, v: bf16[S][D], sk: f32) -> (K', V' s8[S][D], s_v, f32(V'_0) f32[D])   kv_quant(_window)_batch
    Int8LutAttentionRow_v1{D,S,SCALE} (q: bf16[D], K': s8[S][D], sk: f32, V': s8[S][D], s_v, v0: f32[D]) -> bf16[D]   window_rows_batch

The quantizer: s = RN32(fmaxf(max_k |f32(x_k)|, RN32(1e-30)) / 127) (`Int8RowScale`, an exact max) and each code
s8(fminf(fmaxf(rint(RN32(x_k / s)), -127), 127) (`Int8QuantElement`: ties to even, the clamp, `cvt.rzi`; fmaxf takes the operand
that is not NaN, so the clamp is total).  Every s8 product is a chain of `BlackwellS8ImmaDot32_v1` steps from 0, 32 codes a step
summed exactly and clamped to s32 (`s8_chain`); then the epilogues as `registry.int8_exact` states them, one f32 or f64 rounding per
operation, with the attention's fixed-point A from the Definition's own primitives.  A non-finite bf16 word is declined before the
kernel (`rows._instance`).  Registered from `rows` (kernel "rows", `register_rows`).
"""
from __future__ import annotations

import json
from typing import Any, Callable, Sequence

import numpy as np

from verity.ml.definitions.numerics.kernel_registry import Declined, prim_static

FAMILY = "Int8RowQuant_v1"
SCALED_MM, KV_QUANT, ATTENTION_ROW = "ScaledMmS8_v1", "Int8KvQuant_v1", "Int8LutAttentionRow_v1"
DOT = "BlackwellS8ImmaDot32_v1"
EVALUATORS = {
    FAMILY: "verity.ml.definitions.numerics.int8_rows.int8_row_quant_rows (Int8RowQuant_v1{K}: the row's f32 |x| max from +0 floored at "
            "RN32(1e-30), one div.rn by 127, then rint(x / s) clamped to [-127, 127] per code; registry.int8_exact)",
    SCALED_MM: "verity.ml.definitions.numerics.int8_rows.scaled_mm (ScaledMmS8_v1{K,N,DOT=BlackwellS8ImmaDot32_v1}: K/32 satfinite s8 "
               "steps from 0 per coordinate, cvt.rn.f32.s32, RN32 by the token's scale, then by the channel's, cvt.rn.bf16.f32; "
               "ml.circuits.ops.gemm.ScaledMmS8Coordinate_v1)",
    KV_QUANT: "verity.ml.definitions.numerics.int8_rows.kv_quant (Int8KvQuant_v1{S,D}: K' each key under the KV head's K scale, V' "
              "Int8RowQuant over each channel's S values, f32(V'_0); registry.int8_exact)",
    ATTENTION_ROW: "verity.ml.definitions.numerics.int8_rows.lut_attention_row (Int8LutAttentionRow_v1{D,S,SCALE}: Int8RowQuant of q, "
                   "the s8 scores, Int8AttnScale's A by the Definition's primitives, the LUT softmax, P V' over the keys zero-padded "
                   "to a multiple of 32, then Int8AttnOut per channel; registry.int8_exact)",
}
TOP = np.float32(127.0)
FLOOR = np.float32(1e-30)
S32_MIN, S32_MAX = -(1 << 31), (1 << 31) - 1
CHUNK = 1024                            # weight rows per s8 product (bounds the int64 temporaries to CHUNK x K words)


def _s8(b: Any) -> np.ndarray:
    return np.asarray(b, dtype=np.uint8).view(np.int8).astype(np.int64)


def _u32(w: Any, n: int, what: str) -> np.ndarray:
    a = np.asarray(w, dtype=np.uint32).reshape(-1)
    if a.size != n:
        raise ValueError(f"{what} has {a.size} words, the signature has {n}")
    return a


def _codes(q: np.ndarray) -> np.ndarray:
    """s8 codes of RN32 quotients: fminf(fmaxf(rint(q), -127), 127), then `cvt.rzi` of an integer in [-127, 127]."""
    return np.fmin(np.fmax(np.rint(q), -TOP), TOP).astype(np.int8).view(np.uint8)


def int8_row_quant_rows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x (R, K) bf16 words -> (codes (R, K) s8 bytes, s (R,) f32 words)."""
    from verity.ml.definitions.numerics import derived_rows as D
    xf = D.as_f32(D.bf16_to_f32_words(np.ascontiguousarray(x, dtype=np.uint16)))
    m = np.maximum(np.max(np.abs(xf), axis=-1, initial=np.float32(0.0)), FLOOR).astype(np.float32)
    s = (m / TOP).astype(np.float32)
    with np.errstate(all="ignore"):
        return _codes((xf / s[..., None]).astype(np.float32)), D.as_u32(s)


def s8_chain(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The s32 accumulators of s8 values a (K,) against each row of b (N, K), K a multiple of 32: K/32 `BlackwellS8ImmaDot32_v1`
    steps from 0 in ascending k, each 32 exact products added and clamped to s32 once (int64 (N,)).  While 2^14 K fits s32 no step
    can clamp, and the sum is one exact product."""
    N, K = b.shape
    if K % 32 or a.shape != (K,):
        raise ValueError(f"an s8 chain steps 32 codes at a time over equal rows: a {a.shape}, b {b.shape}")
    out = np.empty(N, dtype=np.int64)
    for n0 in range(0, N, CHUNK):
        bc = b[n0:n0 + CHUNK]
        if K << 14 <= S32_MAX:
            out[n0:n0 + CHUNK] = bc @ a
            continue
        steps = (bc * a).reshape(bc.shape[0], K // 32, 32).sum(-1)
        acc = np.zeros(bc.shape[0], dtype=np.int64)
        for s in range(K // 32):
            acc = np.clip(acc + steps[:, s], S32_MIN, S32_MAX)
        out[n0:n0 + CHUNK] = acc
    return out


def _s32_to_f32(acc: np.ndarray) -> np.ndarray:
    """`cvt.rn.f32.s32` / `I32ToF32`: s32 values as their nearest f32 words."""
    from verity.ml.definitions.numerics import derived_rows as D
    return D.as_u32(np.asarray(acc, dtype=np.int64).astype(np.float32))


def _to_bf16_rn(u: np.ndarray) -> np.ndarray:
    """`F32ToBf16Rn_v1`: RNE, every NaN 0x7FC0 (`F2fpBf16_v1`'s word elsewhere)."""
    from verity.ml.definitions.numerics import derived_rows as D
    u = np.asarray(u, dtype=np.uint32)
    r = D.f2fp_bf16(u)
    r[((u & 0x7F800000) == 0x7F800000) & ((u & 0x007FFFFF) != 0)] = 0x7FC0
    return r


def _float(v: Any) -> float:
    """A float static as the spec text renders it (`{"f64":"<hex>"}`, a hex or decimal string) or as a binding holds it."""
    if isinstance(v, str) and v.lstrip().startswith("{"):
        v = json.loads(v)
    if isinstance(v, dict) and set(v) == {"f64"}:
        return float.fromhex(str(v["f64"]))
    if isinstance(v, str):
        return float.fromhex(v) if v.lstrip("-").startswith("0x") else float(v)
    return float(v)


def int8_row_quant(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    K = int(statics["K"])
    codes, s = int8_row_quant_rows(np.asarray(ops[0], dtype=np.uint16).reshape(1, K))
    return {"0": codes.reshape(K), "1": s.reshape(1)}


def scaled_mm(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """One token row of the int8 linear: y_n = bf16(RN32(RN32(RN32(Y_n) sx) sw_n)), Y_n the s8 chain of x against W's row n."""
    from verity.ml.definitions.numerics import derived_rows as D
    if (dot := prim_static(statics, "DOT", SCALED_MM)) != DOT:
        raise Declined(f"no registered evaluator for {SCALED_MM} with DOT={dot} (only {DOT})")
    if len(ops) != 4:
        raise Declined(f"{SCALED_MM}: the row wires {len(ops)} args, the Definition takes 4 (x, sx, w, sw)")
    K, N = int(statics["K"]), int(statics["N"])
    x, w = _s8(ops[0]).reshape(K), _s8(ops[2]).reshape(N, K)
    sx, sw = _u32(ops[1], 1, "sx"), _u32(ops[3], N, "sw")
    with np.errstate(all="ignore"):
        y = D.mul_rn(D.mul_rn(_s32_to_f32(s8_chain(x, w)), np.broadcast_to(sx, (N,))), sw)
    out = D.f2fp_bf16(y)
    return {"0": out, "out": out}


def kv_quant(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """One KV head's window: K' (each key under `sk`), V' by key (`Int8RowQuant{S}` over each channel), s_v and f32(V'_0)."""
    from verity.ml.definitions.numerics import derived_rows as D
    if len(ops) != 3:
        raise Declined(f"{KV_QUANT}: the row wires {len(ops)} args, the Definition takes 3 (k, v, sk)")
    S, Dh = int(statics["S"]), int(statics["D"])
    k = D.as_f32(D.bf16_to_f32_words(np.asarray(ops[0], dtype=np.uint16).reshape(S, Dh)))
    v = np.asarray(ops[1], dtype=np.uint16).reshape(S, Dh)
    sk = D.as_f32(_u32(ops[2], 1, "sk"))
    with np.errstate(all="ignore"):
        kq = _codes((k / sk).astype(np.float32))
    cq, sv = int8_row_quant_rows(v.T)
    vq = np.ascontiguousarray(cq.T)
    return {"0": kq.reshape(-1), "1": vq.reshape(-1), "2": sv, "3": D.as_u32(vq[0].view(np.int8).astype(np.float32))}


def attn_scale(sq: int, sk: int, scale: float) -> int:
    """`Int8AttnScale_v1{SCALE}`'s A by the Definition's primitives: F32MulWide, four F64Mul, F64ToU32RniSat."""
    from verity.ml.definitions.registry import int8_exact as X
    x = X.F32MulWide.evaluate(sq, sk)
    for c in (scale, X.LOG2E, 16.0, 4294967296.0):
        x = X.F64Mul.evaluate(x, X._f64_bits(c))
    return X.F64ToU32RniSat.evaluate(x)


def lut_attention_row(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """One query row over its S keys: Q' and s_q, the scores S_j, A, the LUT softmax (P, r, z), Y_c = P . V'[c] over the keys padded
    to a multiple of 32, and out_c = bf16(RN32(RN32(RN32(RN32(Y_c) r) + RN32(z v0_c)) sv_c))."""
    from verity.ml.definitions.numerics import derived_rows as D
    from verity.ml.definitions.registry import int8_exact as X
    if len(ops) != 6:
        raise Declined(f"{ATTENTION_ROW}: the row wires {len(ops)} args, the Definition takes 6 (q, kq, sk, vq, sv, v0)")
    Dh, S = int(statics["D"]), int(statics["S"])
    if Dh % 32:
        raise ValueError(f"{ATTENTION_ROW}: D = {Dh} is not a multiple of 32")
    qq, sq = int8_row_quant_rows(np.asarray(ops[0], dtype=np.uint16).reshape(1, Dh))
    kq, vq = _s8(ops[1]).reshape(S, Dh), _s8(ops[3]).reshape(S, Dh)
    sk, sv, v0 = _u32(ops[2], 1, "sk"), _u32(ops[4], Dh, "sv"), _u32(ops[5], Dh, "v0")
    sc = s8_chain(_s8(qq[0]), kq)
    a = attn_scale(int(sq[0]), int(sk[0]), _float(statics["SCALE"]))
    d = (int(sc.max()) - sc) & 0xFFFFFFFF
    d = np.maximum(np.where(d >= 1 << 31, d - (1 << 32), d), 0).astype(np.uint64)
    e = np.asarray(X.LUT, dtype=np.int64)[np.minimum((d * np.uint64(a)) >> np.uint64(32), X.LUT_SIZE - 1).astype(np.int64)]
    total = int(e.sum()) & 0xFFFFFFFF
    lf = D.as_f32(_s32_to_f32(np.asarray([total - (1 << 32) if total >> 31 else total])))
    with np.errstate(all="ignore"):
        r = D.as_u32((np.float32(1.0) / lf).astype(np.float32))
        z = D.as_u32((D.as_f32(_s32_to_f32(e[:1])) / lf).astype(np.float32))
        p = np.zeros(X.padded(S), dtype=np.int64)
        p[1:S] = ((e[1:] & 0xFF) ^ 0x80) - 0x80
        vt = np.zeros((Dh, X.padded(S)), dtype=np.int64)
        vt[:, :S] = vq.T
        yr = D.mul_rn(_s32_to_f32(s8_chain(p, vt)), np.broadcast_to(r, (Dh,)))
        f = D.mul_rn(D.add_rn(yr, D.mul_rn(np.broadcast_to(z, (Dh,)), v0)), sv)
    out = _to_bf16_rn(f)
    return {"0": out, "out": out}


ROWS: dict[str, tuple[Callable, tuple[str, ...]]] = {
    FAMILY: (int8_row_quant, ("0", "1")),
    SCALED_MM: (scaled_mm, ("0",)),
    KV_QUANT: (kv_quant, ("0", "1", "2", "3")),
    ATTENTION_ROW: (lut_attention_row, ("0",)),
}


def check_targets(family: str = FAMILY) -> list:
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops import gemm
    from verity.ml.definitions.registry import int8_exact as X
    return {
        FAMILY: lambda: [bind(X.Int8RowQuant, K=k) for k in (1, 8, 33)],
        SCALED_MM: lambda: [bind(gemm.ScaledMmS8, K=k, N=n, **gemm.GEMM_SM120_S8) for k, n in ((32, 1), (64, 3))],
        KV_QUANT: lambda: [bind(X.Int8KvQuant, S=s, D=32) for s in (1, 3)],
        ATTENTION_ROW: lambda: [bind(X.Int8LutAttentionRow, D=32, S=s, SCALE=32 ** -0.5) for s in (1, 5, 33)],
    }[family]()


def register_rows(evaluators: dict, row: Callable) -> None:
    """Register this module's row evaluators without growing the shared rows catalog."""
    evaluators |= EVALUATORS
    for family, (fn, returns) in ROWS.items():
        row(family, fn, returns, check_targets=lambda rng, f=family: check_targets(f))


__all__ = ["ATTENTION_ROW", "EVALUATORS", "FAMILY", "KV_QUANT", "ROWS", "SCALED_MM", "attn_scale", "check_targets", "int8_row_quant",
           "int8_row_quant_rows", "kv_quant", "lut_attention_row", "register_rows", "s8_chain", "scaled_mm"]
