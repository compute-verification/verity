"""The GEMM coordinate as a composite, parameterised by the k16 tensor-core step.

    DotBf16_v2{K, DOT}          f32 accumulator after K/16 `DOT` steps in ascending k from a zero accumulator
    GemmCoordinate_v2{K, DOT}   bf16 word: cvt.rn.bf16.f32(DotBf16)   (Triton `acc.to(bf16)`; the same on sm_80/sm_90)
    GemmCoordinatePad_v1{T, K, DOT}  (x, wrow: bf16[T]) -> bf16: `GemmCoordinate_v2{K, DOT}` on x and wrow each padded with
                                K - T bf16 +0 words (a masked k loop's loads), for T <= K: attention's PV at T keys
    PvRow_v1{T, K, D, DOT}      (p: bf16[T], v0..v{T-1}: bf16[D]) -> bf16[D]: one head's attention output row, dim d
                                `GemmCoordinatePad_v1{T, K, DOT}` of p and (v0[d], ..., v{T-1}[d]); each key's V row is its own
                                D-word row (128 bytes at D = 64), as a proof block reads it whole from the proof block that wrote it
    PvRowPad_v1{T, K, D, DOT}   (p: bf16[T], v0..v{T-1}: bf16[D], n: u16) -> bf16[D]: `PvRow_v1{T, K, D, DOT}` with key t's
                                p word and V row +0 words unless t < n, so for 1 <= n <= T it is `PvRow_v1{T=n}` on the first
                                n keys whatever the rest hold; n, the key count, is a position port (`KEY_COUNT`), the one
                                unsigned word `QkSoftmaxRowPad_v1` reads
    Gemm_v2{K, N, DOT}          x[K] @ W[N,K]^T -> [N], no epilogue

`DOT` is a static naming the step: `GEMM_AMPERE` binds `AmpereBF16TcDot16_v2`, `GEMM_HOPPER` binds
`HopperBF16WgmmaDot16_v1`.  A proof block for the FLOP proof is one `GemmCoordinate_v2{K, DOT}` call: K/16
transitions of `DOT` followed by one `F2fpBf16`.

The same coordinate over the sm_120 low-precision steps, each K/k `DOT` steps in ascending k from a zero accumulator and
one `F2fpBf16`:

    GemmCoordinateE4m3_v1{K, DOT}   (x, wrow: e4m3[K]) -> bf16; k = 32   GEMM_SM120_E4M3: BlackwellE4m3QmmaDot32_v1
    GemmCoordinateNvf4_v1{K, DOT}   (x, wrow: e2m1[K], sx, sw: ue4m3[K/16]) -> bf16; k = 64   GEMM_SM120_NVF4: BlackwellNvf4OmmaDot64_v1
    GemmCoordinateMxf4_v1{K, DOT}   (x, wrow: e2m1[K], sx, sw: ue8m0[K/32]) -> bf16; k = 64   GEMM_SM120_MXF4: BlackwellMxf4OmmaDot64_v1

and over the sm_120 s8 step (`GEMM_SM120_S8`: `BlackwellS8ImmaDot32_v1`), whose sum is exact while it stays in s32, so the
word does not depend on the order the k steps run in. `s8_gemm` gives another card's binding from its s8 step: the
registered step when the card's step is its function (`GEMM_SM90_S8`, the H200's `integer.HOPPER_SM90_S8_K32`), so a circuit
over it names no architecture (`prims`), and a refusal for any other step:

    GemmAccumulatorS8_v1{K, DOT}      (x, wrow: s8[K]) -> s32: K/32 `DOT` steps from a zero accumulator
    ScaledMmS8Coordinate_v1{K, DOT}   (x: s8[K], sx: f32, wrow: s8[K], sw: f32) -> bf16: the accumulator scaled as
                                      `benchmarks/ppl/steps.py`'s `int8_linear` (sx per token, sw per channel):
                                      cvt.rn.bf16.f32((cvt.rn.f32.s32(acc) * sx) * sw)
    ScaledMmS8CoordinateRows_v1{K, R, DOT}  (x0..x{R-1}: s8[K/R], sx: f32, wrow0..wrow{R-1}: s8[K/R], sw: f32) -> bf16:
                                      `ScaledMmS8Coordinate_v1{K, DOT}` on the concatenations of x's and wrow's R parts, so
                                      that each operand is R rows of at most 262,144 bits (`UNIT_CLASS_MAX`) when K is
                                      past 32,768
    ScaledMmS8ResidualCoordinateRows_v1{K, R, DOT}  (ScaledMmS8CoordinateRows_v1's ports, r: bf16) -> bf16: the residual
                                      add in the coordinate's epilogue, `Bf16AddF2fp_v1(r, ScaledMmS8CoordinateRows_v1{K, R,
                                      DOT})`: the coordinate rounded to bf16, then added to the residual stream's word r
                                      as `Bf16AddRow_v1` adds it, so the projection's own word is never committed
    ScaledMmS8_v1{K, N, DOT}          (x: s8[K], sx: f32, w: s8[N][K], sw: f32[N]) -> bf16[N]: the linear on one token row,
                                      `ScaledMmS8Coordinate_v1` over W's rows (each coordinate a `Q_word` proof block)

census/v1 counts `GemmAccumulatorS8_v1`'s products as served by none (`census.serves_none`): it has no committed words of its
own, so Lean's `Accounting.Gates.roleOf` puts its steps at `Site.other`.

`bind` refuses a K that isn't a multiple of the Definition's step width k (16, 32 or 64): its K/k steps would drop the last
K mod k products. The s8 Definitions also refuse K >= `S8_K_LIMIT` = 2^17 (`integer.exact_k_limit`), where an s8 x s8 sum can
leave s32 and the step clamps: below it the accumulator word is the exact sum, in any k order.

There are no `_v1` forms: they named the Ampere step in their bodies, which would now name a different primitive id (see
`prims.AmpereBF16TcDot16`), and the static form is the definition the proofs target.
"""
from __future__ import annotations

from collections.abc import Callable

from verity.ml.circuits.census import FAMILY_COUNTS, serves_none
from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.ports import In, Ports
from verity.ml.circuits.refs import Coll, broadcast, concat
from verity.ml.circuits.silicon.integer import BLACKWELL_SM120_S8_M16N8K32, HOPPER_SM90_S8_K32, IntegerMma, exact_k_limit
from verity.ml.circuits.types import Array, Value

from . import fp32 as F
from . import prims as P
from . import scalar as SC
from .prims import BF16, E2M1, E4M3, F32, S8, S32, ZERO32

__all__ = ["DotBf16", "GEMM_AMPERE", "GEMM_HOPPER", "GEMM_SM120_E4M3", "GEMM_SM120_MXF4", "GEMM_SM120_NVF4", "GEMM_SM120_S8",
           "GEMM_SM90_S8", "Gemm", "GemmAccumulatorS8", "GemmCoordinate", "GemmCoordinateE4m3", "GemmCoordinateMxf4",
           "GemmCoordinateNvf4", "GemmCoordinatePad", "KEY_COUNT", "PvRow", "PvRowPad", "S8_K_LIMIT", "ScaledMmS8",
           "ScaledMmS8Coordinate", "ScaledMmS8CoordinateRows", "ScaledMmS8ResidualCoordinateRows", "s8_gemm"]

_SCALE = Value(8)

#: the least K whose s8 x s8 sum can leave s32: 2^17
S8_K_LIMIT = exact_k_limit("s8", "s8")


def _refuse_k(k: int, limit: int | None = None) -> Callable[[dict], str | None]:
    """`bind`'s refusal for a body of K/k `k`-wide steps: a K that isn't a multiple of k, and a K at or past `limit`."""

    def refuse(b: dict) -> str | None:
        K = b.get("K")
        if not isinstance(K, int):
            return None
        if K % k:
            return (f"K={K} is not a multiple of the {k}-wide step, so K // {k} steps would drop the last {K % k} of its {K} "
                    f"products: pad every operand with zero codes to K={K - K % k + k} (what a masked k loop loads) and bind that")
        if limit is not None and K >= limit:
            return (f"K={K} is at or past {limit}, the least K whose sum can leave the s32 accumulator: a step can clamp, so the "
                    f"word is not always the exact sum (and past {limit} it can depend on the kernel's k order). Split the k "
                    f"range into accumulators of at most K={limit - 1 - (limit - 1) % k} each")
        return None

    return refuse


_K16, _K32, _K64, _S8 = _refuse_k(16), _refuse_k(32), _refuse_k(64), _refuse_k(32, S8_K_LIMIT)


def _refuse_t(base: Callable[[dict], str | None]) -> Callable[[dict], str | None]:
    """`base`'s refusal for K, and a T outside 1..K (a coordinate on T of its K words)."""

    def refuse(b: dict) -> str | None:
        T, K = b.get("T"), b.get("K")
        if isinstance(T, int) and isinstance(K, int) and not 1 <= T <= K:
            return f"T={T} is not in 1..K={K}: the coordinate reads T words of each operand and pads them to K with +0"
        return base(b)

    return refuse


def _refuse_r(base: Callable[[dict], str | None]) -> Callable[[dict], str | None]:
    """`base`'s refusal for K, and an R that is not a positive divisor of K (K/R codes a part)."""

    def refuse(b: dict) -> str | None:
        R, K = b.get("R"), b.get("K")
        if isinstance(R, int) and isinstance(K, int) and (R < 1 or K % R):
            return f"R={R} does not divide K={K} into R parts of K/R codes each"
        return base(b)

    return refuse


def _refuse_d(base: Callable[[dict], str | None]) -> Callable[[dict], str | None]:
    """`base`'s refusal, and a D under 1 (the width of V's key rows)."""

    def refuse(b: dict) -> str | None:
        D = b.get("D")
        if isinstance(D, int) and D < 1:
            return f"D={D} is not a head width: each of V's key rows is D >= 1 words"
        return base(b)

    return refuse


@composite("DotBf16", 2, ["K", "DOT"], lambda S: ((("x", Array(S.K, BF16)), ("w", Array(S.K, BF16))), F32),
           conformance="exact-model-tested", refuse=_K16)
def DotBf16(B, S, x, w):
    """f32 accumulator after K/16 `DOT` steps in ascending k from a zero accumulator (the Triton
    `matmul_kernel_persistent` k-loop on either architecture)."""
    acc = B.call(ZERO32)
    for s in range(S.K // 16):
        acc = B.call(S.DOT, acc, x[16 * s:16 * s + 16], w[16 * s:16 * s + 16])
    return acc


@composite("GemmCoordinate", 2, ["K", "DOT"], lambda S: ((("x", Array(S.K, BF16)), ("wrow", Array(S.K, BF16))), BF16),
           conformance="exact-model-tested", refuse=_K16)
def GemmCoordinate(B, S, x, wrow):
    """One output word of `x @ W^T`: bf16_rne(DotBf16{K, DOT}(x, W[j, :])) (`cvt.rn.bf16.f32`)."""
    return B.call(P.F2fpBf16, B.call(bind(DotBf16, K=S.K, DOT=S.DOT), x, wrow))


@composite("GemmCoordinatePad", 1, ["T", "K", "DOT"],
           lambda S: ((("x", Array(S.T, BF16)), ("wrow", Array(S.T, BF16))), BF16),
           conformance="exact-model-tested", refuse=_refuse_t(_K16))
def GemmCoordinatePad(B, S, x, wrow):
    """`GemmCoordinate_v2{K, DOT}` on T words of x and of W's row, each padded with K - T bf16 +0 words: what a k loop that
    masks its loads past T to 0.0 multiplies (attention's PV at T <= K keys)."""
    if S.T == S.K:
        return B.call(bind(GemmCoordinate, K=S.K, DOT=S.DOT), x, wrow)
    space, idx, leaf = B.call(P.const(16, 0)).refs[0]
    zeros = broadcast(space, idx, leaf, S.K - S.T)

    def pad(a):
        return Coll(Array(S.K, BF16), concat([a.refs, zeros]))

    return B.call(bind(GemmCoordinate, K=S.K, DOT=S.DOT), pad(x), pad(wrow))


@composite("PvRow", 1, ["T", "K", "D", "DOT"],
           lambda S: ((("p", Array(S.T, BF16)), *((f"v{t}", Array(S.D, BF16)) for t in range(S.T))), Array(S.D, BF16)),
           conformance="exact-model-tested", refuse=_refuse_d(_refuse_t(_K16)))
def PvRow(B, S, p, *vs):
    """One head's attention output row at T <= K keys: for each of V's D dims, `GemmCoordinatePad_v1{T, K, DOT}` of the
    probability row p and that dim's column of the T key rows v0..v{T-1}. One proof block a head: each key row is D words (128 bytes
    at D = 64), over the 111 bytes C-Flock needs of some row of a proof block, where a coordinate's own two rows are 2T bytes."""
    v = Coll(Array(S.T, Array(S.D, BF16)), concat([r.refs for r in vs]))
    return B.batch(bind(GemmCoordinatePad, T=S.T, K=S.K, DOT=S.DOT), p, v, axes=(None, 1))


#: a padded proof block's key count ``n``: a run input of kind position, public as an unpadded proof block's T is. No coin port is
#: beside it, so the Definition it types is never a seed (`tests/test_seed_ports.py`)
KEY_COUNT = Ports(params={"n": In(kind="position")})
#: the key count's type: one unsigned word
U16 = Value(16)
_SIGN = 1 << 31


def _refuse_u16_t(base: Callable[[dict], str | None]) -> Callable[[dict], str | None]:
    """`base`'s refusal, and a T past the largest key count a u16 n holds."""

    def refuse(b: dict) -> str | None:
        T = b.get("T")
        if isinstance(T, int) and T > 0xFFFF:
            return f"T={T} is past 65535, the largest key count n a u16 word holds"
        return base(b)

    return refuse


@composite("PvRowPad", 1, ["T", "K", "D", "DOT"],
           lambda S: ((("p", Array(S.T, BF16)), *((f"v{t}", Array(S.D, BF16)) for t in range(S.T)), ("n", U16)),
                      Array(S.D, BF16)),
           conformance="exact-model-tested", refuse=_refuse_u16_t(_refuse_d(_refuse_t(_K16))), ports=KEY_COUNT)
def PvRowPad(B, S, p, *rest):
    """`PvRow_v1{T, K, D, DOT}` on the first n of T keys: key t's p word and V row are kept when t < n, n unsigned, and are
    +0 words otherwise (`SelectBf16_v1`). t < n is (t + 1)·2^16 <= n·2^16 with each side's sign bit flipped, as signed
    i32 words (`I32Le_v1`; `Bf16ToF32_v1` is the shift by 16, and `I32Add_v1` of 2^31 flips the sign bit, which turns
    the unsigned order into the signed one). For
    1 <= n <= T, dim d is then `GemmCoordinate_v2{K, DOT}` on p[:n] and v0[d]..v{n-1}[d] padded with +0 words to K, which is
    `PvRow_v1{T=n}`'s dim d: the same operands, whatever the absent keys' words are (a +0 p word against an inf V word would
    be a NaN product). n = 0 keeps no key, and n > T every key."""
    *vs, n = rest
    zero = B.call(P.const(16, 0))
    count = B.call(SC.I32Add, B.call(P.Bf16ToF32, n), B.call(P.const(32, _SIGN)))
    keep = [B.call(SC.I32Le, B.call(P.const(32, ((t + 1) << 16) ^ _SIGN)), count) for t in range(S.T)]
    ps = Coll(Array(S.T, BF16), concat([B.call(SC.SelectBf16, keep[t], p[t], zero).refs for t in range(S.T)]))
    rows = [B.batch(SC.SelectBf16, keep[t], v, zero, axes=(None, 0, None)) for t, v in enumerate(vs)]
    return B.call(bind(PvRow, T=S.T, K=S.K, D=S.D, DOT=S.DOT), ps, *rows)


@composite("Gemm", 2, ["K", "N", "DOT"], lambda S: ((("x", Array(S.K, BF16)), ("w", Array(S.N, Array(S.K, BF16)))), Array(S.N, BF16)),
           conformance="exact-model-tested", refuse=_K16)
def Gemm(B, S, x, w):
    """`x[K] @ W[N,K]^T -> [N]`, no epilogue (a bias is a separate node)."""
    return B.batch(bind(GemmCoordinate, K=S.K, DOT=S.DOT), x, w, axes=(None, 0))


@composite("GemmCoordinateE4m3", 1, ["K", "DOT"], lambda S: ((("x", Array(S.K, E4M3)), ("wrow", Array(S.K, E4M3))), BF16),
           conformance="exact-model-tested", refuse=_K32)
def GemmCoordinateE4m3(B, S, x, wrow):
    """One output word of an E4M3 `x @ W^T`: K/32 `DOT` steps in ascending k from a zero accumulator, then `cvt.rn.bf16.f32`."""
    acc = B.call(ZERO32)
    for s in range(S.K // 32):
        acc = B.call(S.DOT, acc, x[32 * s:32 * s + 32], wrow[32 * s:32 * s + 32])
    return B.call(P.F2fpBf16, acc)


def _block_scaled_coordinate(B, S, x, wrow, sx, sw, ns: int):
    acc = B.call(ZERO32)
    for s in range(S.K // 64):
        acc = B.call(S.DOT, acc, x[64 * s:64 * s + 64], wrow[64 * s:64 * s + 64], sx[ns * s:ns * s + ns], sw[ns * s:ns * s + ns])
    return B.call(P.F2fpBf16, acc)


@composite("GemmCoordinateNvf4", 1, ["K", "DOT"],
           lambda S: ((("x", Array(S.K, E2M1)), ("wrow", Array(S.K, E2M1)), ("sx", Array(S.K // 16, _SCALE)),
                       ("sw", Array(S.K // 16, _SCALE))), BF16),
           conformance="exact-model-tested", refuse=_K64)
def GemmCoordinateNvf4(B, S, x, wrow, sx, sw):
    """One output word of an NVFP4 `x @ W^T` (one UE4M3 scale byte per 16 codes): K/64 `DOT` steps in ascending k from a zero
    accumulator, then `cvt.rn.bf16.f32`."""
    return _block_scaled_coordinate(B, S, x, wrow, sx, sw, 4)


@composite("GemmCoordinateMxf4", 1, ["K", "DOT"],
           lambda S: ((("x", Array(S.K, E2M1)), ("wrow", Array(S.K, E2M1)), ("sx", Array(S.K // 32, _SCALE)),
                       ("sw", Array(S.K // 32, _SCALE))), BF16),
           conformance="exact-model-tested", refuse=_K64)
def GemmCoordinateMxf4(B, S, x, wrow, sx, sw):
    """One output word of an MXFP4 `x @ W^T` (one UE8M0 scale byte per 32 codes): K/64 `DOT` steps in ascending k from a zero
    accumulator, then `cvt.rn.bf16.f32`."""
    return _block_scaled_coordinate(B, S, x, wrow, sx, sw, 2)


@composite("GemmAccumulatorS8", 1, ["K", "DOT"], lambda S: ((("x", Array(S.K, S8)), ("wrow", Array(S.K, S8))), S32),
           conformance="exact-model-tested", refuse=_S8)
def GemmAccumulatorS8(B, S, x, wrow):
    """The s32 accumulator of an s8 `x @ W^T` coordinate: K/32 `DOT` steps in ascending k from a zero accumulator."""
    acc = B.call(ZERO32)
    for s in range(S.K // 32):
        acc = B.call(S.DOT, acc, x[32 * s:32 * s + 32], wrow[32 * s:32 * s + 32])
    return acc


GemmAccumulatorS8.counts = FAMILY_COUNTS["GemmAccumulatorS8_v1"] = serves_none  # type: ignore[attr-defined]


@composite("ScaledMmS8Coordinate", 1, ["K", "DOT"],
           lambda S: ((("x", Array(S.K, S8)), ("sx", F32), ("wrow", Array(S.K, S8)), ("sw", F32)), BF16),
           conformance="model-untested: the epilogue is benchmarks/ppl/steps.py's int8_linear, the int8 step vLLM's "
                       "int8 profile serves through torch._int_mm: Y.float() * s_a[:, None] * s_w[None, :] (two f32 "
                       "multiplies, the token's scale first), then the bf16 cast; the sum is exact for K < 2^17 whatever "
                       "the kernel's k order; the k step pinned (BlackwellS8ImmaDot32_v1)", refuse=_S8)
def ScaledMmS8Coordinate(B, S, x, sx, wrow, sw):
    """One output word of a W8A8 linear: `cvt.rn.bf16.f32((cvt.rn.f32.s32(acc) * sx) * sw)`, `acc` the
    `GemmAccumulatorS8_v1{K, DOT}` sum, `sx` the token's scale and `sw` the output channel's. Multiplying the two scales
    first, or `sw` first (CUTLASS c2x's `ScaledEpilogue`), rounds differently."""
    acc = B.call(P.S32ToF32Rn, B.call(bind(GemmAccumulatorS8, K=S.K, DOT=S.DOT), x, wrow))
    return B.call(P.F2fpBf16, B.call(F.F32MulV2, B.call(F.F32MulV2, acc, sx), sw))


@composite("ScaledMmS8CoordinateRows", 1, ["K", "R", "DOT"],
           lambda S: ((*((f"x{r}", Array(S.K // S.R, S8)) for r in range(S.R)), ("sx", F32),
                       *((f"wrow{r}", Array(S.K // S.R, S8)) for r in range(S.R)), ("sw", F32)), BF16),
           conformance=ScaledMmS8Coordinate.conformance, refuse=_refuse_r(_S8))
def ScaledMmS8CoordinateRows(B, S, *ports):
    """`ScaledMmS8Coordinate_v1{K, DOT}` on x and W's row each given as R consecutive parts of K/R codes, called on their
    concatenations: the same word for every input. Past K = 32,768 an s8 operand is wider than `partition_v2.UNIT_CLASS_MAX`
    (262,144 bits), so a proof block reads it as R rows."""
    R = S.R

    def cat(parts):
        return Coll(Array(S.K, S8), concat([p.refs for p in parts]))

    x, sx, wrow, sw = cat(ports[:R]), ports[R], cat(ports[R + 1:2 * R + 1]), ports[2 * R + 1]
    return B.call(bind(ScaledMmS8Coordinate, K=S.K, DOT=S.DOT), x, sx, wrow, sw)


@composite("ScaledMmS8ResidualCoordinateRows", 1, ["K", "R", "DOT"],
           lambda S: ((*ScaledMmS8CoordinateRows.signature(S)[0], ("r", BF16)), BF16),
           conformance=ScaledMmS8Coordinate.conformance, refuse=_refuse_r(_S8))
def ScaledMmS8ResidualCoordinateRows(B, S, *ports):
    """The residual add in a W8A8 coordinate's epilogue: `Bf16AddF2fp_v1(r, y)`, y `ScaledMmS8CoordinateRows_v1{K, R, DOT}`
    of the other ports (its bf16 word, `cvt.rn.bf16.f32` of the scaled sum) and r the residual stream's word, in
    `Bf16AddRow_v1`'s order (a the stream, b the projection). The same word as that add of the coordinate's word for every
    input, with y never a committed value of its own."""
    y = B.call(bind(ScaledMmS8CoordinateRows, K=S.K, R=S.R, DOT=S.DOT), *ports[:-1])
    return B.call(F.Bf16AddF2fp, ports[-1], y)


@composite("ScaledMmS8", 1, ["K", "N", "DOT"],
           lambda S: ((("x", Array(S.K, S8)), ("sx", F32), ("w", Array(S.N, Array(S.K, S8))), ("sw", Array(S.N, F32))),
                      Array(S.N, BF16)),
           conformance=ScaledMmS8Coordinate.conformance, refuse=_S8)
def ScaledMmS8(B, S, x, sx, w, sw):
    """A W8A8 linear on one token row: the N output coordinates (`ScaledMmS8Coordinate_v1`) over the s8 row `x` with its scale
    `sx`, the s8 weight rows and their per-channel scales."""
    return B.batch(bind(ScaledMmS8Coordinate, K=S.K, DOT=S.DOT), x, sx, w, sw, axes=(None, None, 0, 0))


#: Binding records for the measured architectures.
GEMM_AMPERE: dict = {"DOT": P.AmpereBF16TcDot16}
GEMM_HOPPER: dict = {"DOT": P.HopperBF16WgmmaDot16}
GEMM_SM120_E4M3: dict = {"DOT": P.BlackwellE4m3QmmaDot32}
GEMM_SM120_NVF4: dict = {"DOT": P.BlackwellNvf4OmmaDot64}
GEMM_SM120_MXF4: dict = {"DOT": P.BlackwellMxf4OmmaDot64}
GEMM_SM120_S8: dict = {"DOT": P.BlackwellS8ImmaDot32}


def s8_gemm(model: IntegerMma) -> dict:
    """The s8 binding of a card whose s8 tensor-core step is ``model``: `GEMM_SM120_S8`'s step when ``model`` computes its
    function, s8 x s8 at k = 32 clamped once at D (the fields `IntegerMma.step` reads), whatever the card. Refused
    (ValueError) for any other step: a wrapping or u8 step is another function, with no registered Definition of its own here."""
    ref = BLACKWELL_SM120_S8_M16N8K32
    got, want = ((m.dtype, m.b_dtype, m.k, m.satfinite) for m in (model, ref))
    if got != want:
        raise ValueError(f"{model.name} ({model.arch}) computes {got} (dtype, b_dtype, k, satfinite), not {ref.name}'s {want}: "
                         f"BlackwellS8ImmaDot32_v1 is not its step, and no s8 binding here is")
    return dict(GEMM_SM120_S8)


#: sm_90's (the H200's `wgmma` m64nNk32 and `mma.sync` m16n8k32, PINNED as `silicon.instructions` sm90.*.s8)
GEMM_SM90_S8: dict = s8_gemm(HOPPER_SM90_S8_K32)


def _register_kernels() -> None:
    from verity.ml.circuits.evaluation import register_kernel

    from verity.ml.kernels import tc as K
    from verity.ml.circuits.silicon.models import BLACKWELL_SM120_E4M3_M16N8K32, BLACKWELL_SM120_MXF4, BLACKWELL_SM120_NVF4
    from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16
    from verity.ml.circuits.silicon.total import HOPPER_BF16_WGMMA_K16
    from verity.ml.circuits.silicon.total_fp8 import HOPPER_E4M3_WGMMA_K32

    steps = {P.AmpereBF16TcDot16: AMPERE_BF16_M16N8K16, P.HopperBF16WgmmaDot16: HOPPER_BF16_WGMMA_K16}
    domain = "every encoding; DOT one of the core BF16 steps (other steps fall back to the reference)"
    register_kernel(DotBf16, K.dot_kernel(steps), domain=domain, sample=K.sample_tc)
    register_kernel(GemmCoordinate, K.gemm_coordinate_kernel(steps), domain=domain, sample=K.sample_tc)
    register_kernel(GemmCoordinatePad, K.gemm_coordinate_pad_kernel(steps), domain=domain, sample=K.sample_tc)
    register_kernel(PvRow, K.pv_row_kernel(steps), domain=domain, sample=K.sample_tc)
    register_kernel(PvRowPad, K.pv_row_pad_kernel(steps), domain=domain, sample=K.sample_pv_row_pad)
    register_kernel(Gemm, K.gemm_kernel(steps), domain=domain)
    e4m3 = {P.BlackwellE4m3QmmaDot32: BLACKWELL_SM120_E4M3_M16N8K32, P.HopperE4m3QgmmaDot32: HOPPER_E4M3_WGMMA_K32}
    register_kernel(GemmCoordinateE4m3, K.gemm_coordinate_kernel(e4m3), sample=K.sample_tc,
                    domain="every encoding; DOT one of the core E4M3 steps (other steps fall back to the reference)")
    for coord, dot, model, fire in ((GemmCoordinateNvf4, P.BlackwellNvf4OmmaDot64, BLACKWELL_SM120_NVF4, K.nvf4_step_total_batch),
                                    (GemmCoordinateMxf4, P.BlackwellMxf4OmmaDot64, BLACKWELL_SM120_MXF4, K.mxf4_step_total_batch)):
        register_kernel(coord, K.block_scaled_coordinate_kernel({dot: (model, fire)}), sample=K._sample_block_scaled(model),
                        domain=f"every encoding; DOT {dot.id} (other steps fall back to the reference)")


_register_kernels()
