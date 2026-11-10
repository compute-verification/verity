"""`verity_int8::scaled_mm_zy`'s y pinned bit for bit to compute-accounting's dequant Definition, `h1zy.PouwZyDequant_v2`:
F2fpBf16_v2(F32Mul_v4(F32Mul_v4(S32ToF32Rn_v2(z), s_a), s_w)), the served order with no bias, every NaN 0x7FFF.

* **On CUDA** (skipped without it) every cell of every record's y, and of the served y, is `h1zy.dequant` (the Definition's
  word view) of the cell's exact z at its slot's s_a (0 on an empty slot) and its column's s_w; every record parses as
  `h1zy.RECORD` (its leaves the bytes' bits, little-endian) into that z and that y. The edge call (`edge_call`, K = 14,336)
  also meets `PouwZyDequant_v2` evaluated on bits and `h1zy.check`, each unit's ok bit on words, through the served op and
  every GEMM config; random calls at Llama-3.1-8B's linears go through the served op.
* **On the CPU** the op's torch body meets the same pin on the edge call but where y is NaN (torch's cast writes 0x7FC0 or
  0xFFFF where the GPU writes 0x7FFF), so the harness runs in `check` too; and `h1zy.dequant` is int8_exact's stated oracle
  RN_bf16(RN32(RN32(RN32(z) s_a) s_w)), computed step by step in numpy float32 with an explicit round-to-nearest-even cast
  to bf16, over random and edge encodings.
"""
from __future__ import annotations

import functools
import struct

import numpy as np
import pytest
import torch
from verity.ml.protocols.pouw.circuit import h1zy as Z
from verity.ml.circuits.evaluation.bits import evaluate_bits

from tests.engine.test_int8_zy import LLAMA, _configs, _cuda_call, _rows
from verity_vllm.engine import int8_exact as E
from verity_vllm.engine import int8_ops as OPS

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
UR, UC = E.UNIT
M32 = (1 << 32) - 1
#: down_proj's K: ±127 codes over it reach 127² K ≈ 2.3e8, past 2^24, where S32ToF32 rounds
K_EDGE = 14336


def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


NEG = 0x80000000
ONE, TWO, HALF, THREE, P2M10 = f32(1.0), f32(2.0), f32(0.5), f32(3.0), f32(2.0 ** -10)
INF, NINF, QNAN, SNAN, NQNAN, NAN1S = 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 0xFFC00000, 0x7FFFFFFF
MAX, HUGE, TINY = 0x7F7FFFFF, f32(2.0 ** 100), f32(2.0 ** -100)
SUB_MIN, SUB_MID, SUB_MAX, MIN_NORMAL = 0x00000001, 0x00400000, 0x007FFFFF, 0x00800000

#: the edge call's row scales s_a: signed zeros and ones, powers of two (which keep a tie a tie), served-like, subnormal,
#: huge and tiny (products overflow and underflow), infinities and NaNs (quiet, signalling, negative)
SA = [ONE, NEG | ONE, 0, NEG, P2M10, HALF, f32(0.0123 / 127), NEG | f32(0.0123 / 127), SUB_MIN, SUB_MID, SUB_MAX,
      NEG | SUB_MIN, MIN_NORMAL, MAX, NEG | MAX, HUGE, TINY, NEG | TINY, INF, NINF, QNAN, SNAN, NQNAN, NAN1S]
#: its column scales s_w, one column block's 32 columns
SW = [ONE, NEG | ONE, 0, NEG, P2M10, HALF, TWO, THREE, f32(1 / 3), f32(1 / 127), f32(0.0042 / 127),
      NEG | f32(0.0042 / 127), SUB_MIN, SUB_MID, SUB_MAX, NEG | SUB_MID, MIN_NORMAL, MAX, NEG | MAX, HUGE, NEG | HUGE, TINY,
      INF, NINF, QNAN, SNAN, NQNAN, NEG | NAN1S, f32(0.0003 / 127), f32(0.07 / 127), f32(1e-20), f32(1e20)]
#: z at its "fullc" columns (K - 1 codes 127, then 1) of a design row (`row_for`), the largest the last two
ZFULLC = 127 * 127 * (K_EDGE - 1) + 63
TARGETS = [0, 1, -1, 3, -7, 127, -16129,
           257, 259, -257, -259,                                               # bf16 ties at unit scales: to 256, to 260
           (1 << 23) + 0x8000, (1 << 23) + 0x18000, -((1 << 23) + 0x18000),    # ties at the top of f32's exact integers
           (1 << 24) + 1, (1 << 24) + 3, -((1 << 24) + 1), -((1 << 24) + 3),   # S32ToF32 ties: to 2^24, to 2^24 + 4
           (1 << 24) + (1 << 16) + 1,          # S32ToF32 ties to 2^24 + 2^16, a bf16 tie, so 2^24: rounded twice, 0x4B80
           (1 << 24) + (1 << 17) + (1 << 16),  # exact in f32, a bf16 tie rounded up
           (1 << 25) + 2, (1 << 25) + 6, (1 << 27) + 8, (1 << 27) + 24, (1 << 27) + 5, -((1 << 27) + 13),
           200_000_001, -199_999_999, ZFULLC, -ZFULLC]


# -- the oracle ----------------------------------------------------------------------------------------------------------

def rn_bf16(f: np.ndarray) -> np.ndarray:
    """float32 -> bf16 encodings, round to nearest even on the encoding (overflow to inf included), every NaN 0x7FFF."""
    b = f.view(np.uint32).astype(np.uint64)
    r = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)
    return np.where(np.isnan(f), np.uint16(0x7FFF), r)


def oracle(z: np.ndarray, sa: np.ndarray, sw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """int8_exact's RN_bf16(RN32(RN32(RN32(z) s_a) s_w)) in numpy, step by step, on encodings (z u32 as s32, s_a and s_w
    f32): (the f32 before the cast, y bf16)."""
    with np.errstate(all="ignore"):
        zf = np.asarray(z, np.uint32).view(np.int32).astype(np.float32)
        u = (zf * np.asarray(sa, np.uint32).view(np.float32)) * np.asarray(sw, np.uint32).view(np.float32)
    return u, rn_bf16(u)


def rn24(z: np.ndarray) -> np.ndarray:
    """s32 z rounded to 24 significant bits, ties to even, in integers: RN32(z)'s value."""
    z = np.asarray(z, np.int64)
    a = np.abs(z)
    s = np.maximum(np.frexp(a.astype(np.float64))[1].astype(np.int64) - 24, 0)
    q, r = a >> s, a & ((np.int64(1) << s) - 1)
    half = np.where(s > 0, np.int64(1) << np.maximum(s - 1, 0), 0)
    up = (s > 0) & ((r > half) | ((r == half) & ((q & 1) == 1)))
    return np.sign(z) * ((q + up) << s)


# -- the edge call -------------------------------------------------------------------------------------------------------

def row_for(t: int, k: int) -> np.ndarray:
    """K int8 codes whose dot with a "fullc" column (k - 1 codes 127, then 1) is t."""
    r = (t + 63) % 127 - 63
    s = (t - r) // 127
    n, rem = divmod(abs(s), 127)
    assert n + (rem > 0) <= k - 1, t
    q = np.zeros(k, np.int8)
    q[:n] = 127 if s >= 0 else -127
    if rem:
        q[n] = rem if s >= 0 else -rem
    q[k - 1] = r
    return q


def _flipped(k: int, d: int) -> np.ndarray:
    q = np.full(k, 127, np.int8)
    q[[5 + 1000 * i for i in range(d)]] = -127
    return q


def _designs(k: int) -> list[np.ndarray]:
    """The edge call's A rows before their scales: one a target, then ±127 rows (all +127, all -127, one and three codes
    flipped: z = 16129 (K - 2d) against the all-127 columns, past 2^24 and rounded), random signs and random codes."""
    rng = np.random.default_rng(3018)
    return ([row_for(t, k) for t in TARGETS] + [_flipped(k, 0), -_flipped(k, 0), _flipped(k, 1), _flipped(k, 3),
            rng.choice(np.array([-127, 127], np.int8), k), rng.integers(-127, 128, k).astype(np.int8)])


def _columns(k: int) -> list[np.ndarray]:
    """The edge call's B rows before their scales, a block of 32 columns each: "fullc" (z the design's target), all +127,
    all -127, all 0 (z = 0 at every scale), e0 (z = the row's first code) and random codes."""
    rng = np.random.default_rng(3053)
    fullc = np.full(k, 127, np.int8)
    fullc[-1] = 1
    e0 = np.zeros(k, np.int8)
    e0[0] = 1
    return [fullc, np.full(k, 127, np.int8), np.full(k, -127, np.int8), np.zeros(k, np.int8), e0,
            rng.integers(-127, 128, k).astype(np.int8)]


@functools.cache
def edge_call() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """(A [M, K] int8, s_a [M], B [N, K] int8, s_w [N], rows): each design row under each of `SA`, each column kind under
    each of `SW`, over its M rows' slots and a block of empty ones. Scales are f32 encodings (uint32)."""
    designs, cols = _designs(K_EDGE), _columns(K_EDGE)
    a = np.stack([d for _s in SA for d in designs])
    sa = np.array([s for s in SA for _d in designs], np.uint32)
    b = np.stack([c for c in cols for _s in SW])
    sw = np.array([s for _c in cols for s in SW], np.uint32)
    return a, sa, b, sw, -(-a.shape[0] // UR) * UR + UR


def exact_z(a: np.ndarray, b: np.ndarray, rows: int) -> np.ndarray:
    """z [rows, N] as u32 encodings: the slots' A Bᵀ in float64, exact (integer terms and sums below 2^53), empty slots 0."""
    z = np.zeros((rows, b.shape[0]), np.int64)
    z[:a.shape[0]] = (a.astype(np.float64) @ b.astype(np.float64).T).astype(np.int64)
    assert np.abs(z).max() < 1 << 31
    return (z & M32).astype(np.uint32)


def slot_scales(sa: np.ndarray, rows: int) -> np.ndarray:
    """The slots' s_a: row i's under slot i < M, +0 under an empty one."""
    out = np.zeros(rows, np.uint32)
    out[:sa.size] = sa
    return out


# -- the Definition ------------------------------------------------------------------------------------------------------

def _words_to_bits(w: np.ndarray, n: int) -> np.ndarray:
    return ((w.astype(np.uint64)[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(np.uint8)


def dequant_bits(tri: np.ndarray) -> np.ndarray:
    """`PouwZyDequant_v2` evaluated on bits, one instance a row of ``tri`` [n, 3] (z, s_a, s_w encodings)."""
    out = evaluate_bits(Z.PouwZyDequant.bind(), *(_words_to_bits(tri[:, i], 32) for i in range(3)))
    return (out.reshape(len(tri), 16).astype(np.uint64) << np.arange(16, dtype=np.uint64)).sum(axis=1).astype(np.uint16)


def expected(z: np.ndarray, sa: np.ndarray, sw: np.ndarray, bits: bool = False) -> np.ndarray:
    """y [rows, N] bf16 encodings: `h1zy.dequant` of each cell's (z, s_a of its slot, s_w of its column), once a distinct
    triple; with ``bits``, `PouwZyDequant_v2` on bits too, which must agree."""
    tri = np.stack([z, np.broadcast_to(sa[:, None], z.shape), np.broadcast_to(sw[None, :], z.shape)], axis=-1)
    uniq, inv = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    y = np.array([Z.dequant(int(p), int(q), int(r)) for p, q, r in uniq], np.uint16)
    if bits:
        yb = dequant_bits(uniq)
        bad = np.flatnonzero(yb != y)
        assert not bad.size, [(f"0x{int(uniq[i, 0]):08x}", f"0x{int(uniq[i, 1]):08x}", f"0x{int(uniq[i, 2]):08x}",
                               f"words 0x{int(y[i]):04x}", f"bits 0x{int(yb[i]):04x}") for i in bad[:8]]
    return y[inv.reshape(-1)].reshape(z.shape)


# -- the records ---------------------------------------------------------------------------------------------------------

def record_fields(zy: np.ndarray) -> dict[str, np.ndarray]:
    """Records [units, ZY_BYTES] read as `h1zy.RECORD(UR, UC)`: the bytes' bits, little-endian and LSB first, are its leaves,
    field after field, each an array of words whose leaves are their bits, LSB first. {field: [units, words]}."""
    t = Z.RECORD(UR, UC)
    assert zy.shape[1] * 8 == t.leaves == 48 * UR * UC
    out: dict[str, list[np.ndarray]] = {name: [] for name, _ft in t.fields}
    for lo in range(0, zy.shape[0], 64):
        leaves, at = np.unpackbits(zy[lo:lo + 64], axis=1, bitorder="little"), 0
        for name, ft in t.fields:
            n, w = ft.n, ft.elem.n
            assert ft.elem.elem.leaves == 1
            words = leaves[:, at:at + n * w].reshape(-1, n, w).astype(np.uint64)
            out[name].append((words << np.arange(w, dtype=np.uint64)).sum(axis=2))
            at += n * w
        assert at == t.leaves
    return {k: np.concatenate(v) for k, v in out.items()}


def grid(cells: np.ndarray, rows: int, n: int) -> np.ndarray:
    """Units' cells [units, UR UC] as [rows, n]: unit u is row block u // (n / UC) and column block u % (n / UC), cells
    row-major."""
    return cells.reshape(rows // UR, n // UC, UR, UC).transpose(0, 2, 1, 3).reshape(rows, n)


def _diff(what: str, got: np.ndarray, want: np.ndarray, z: np.ndarray, sa: np.ndarray, sw: np.ndarray) -> str:
    """The cells where ``got`` isn't ``want``, each with its (z, s_a, s_w), both words and the numpy oracle's, which says
    which side is wrong."""
    bad = np.argwhere(got != want)
    lines = [f"{what}: {len(bad)} of {got.size} cells differ from h1zy.dequant"]
    for i, j in bad[:12]:
        zz, a, b = z[i, j], sa[i], sw[j]
        o = int(oracle(np.array([zz]), np.array([a]), np.array([b]))[1][0])
        lines.append(f"  cell ({i}, {j}): z {int(np.uint32(zz).view(np.int32))} (0x{int(zz):08x}), s_a 0x{int(a):08x}, "
                     f"s_w 0x{int(b):08x}: kernel y 0x{int(got[i, j]):04x}, h1zy.dequant 0x{int(want[i, j]):04x}, "
                     f"numpy oracle 0x{o:04x}")
    return "\n".join(lines)


def pin(y: np.ndarray, zy: np.ndarray, z: np.ndarray, sa: np.ndarray, sw: np.ndarray, want: np.ndarray,
        nan_any: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Hold a call's served y [M, N] and records zy [units, ZY_BYTES] (uint16 and uint8 arrays) to its exact z, its slots'
    s_a, its s_w and ``want``, the Definition's y: every record parses as `h1zy.RECORD` into z and ``want``, and the served
    y is ``want``'s live rows. ``nan_any``: where ``want`` is 0x7FFF, any NaN encoding passes (torch's CPU cast). Returns
    the records' (z, y) grids."""
    rows, n = z.shape
    assert zy.shape == (E.zy_units(y.shape[0], n, rows), E.ZY_BYTES), zy.shape
    f = record_fields(zy)
    zr, yr = grid(f["z"].astype(np.uint32), rows, n), grid(f["y"].astype(np.uint16), rows, n)
    assert np.array_equal(zr, z), f"{int((zr != z).sum())} record z words differ from the exact product"
    for what, got, ref in (("records' y", yr, want), ("served y", y, want[:y.shape[0]])):
        if nan_any:
            nan = ((got & 0x7F80) == 0x7F80) & ((got & 0x7F) != 0)
            assert np.array_equal(nan, ref == 0x7FFF), f"{what}: NaN where the Definition's isn't"
            got = np.where(nan, np.uint16(0x7FFF), got)
        assert np.array_equal(got, ref), _diff(what, got, ref, z, sa, sw)
    return zr, yr


@functools.cache
def edge_expected() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The edge call's exact z, slots' s_a and the Definition's y (words and bits agreeing)."""
    a, sa, b, sw, rows = edge_call()
    z, sas = exact_z(a, b, rows), slot_scales(sa, rows)
    return z, sas, expected(z, sas, sw, bits=True)


def f32_tensor(bits: np.ndarray) -> torch.Tensor:
    """float32 with the given encodings, NaN payloads and signs kept."""
    return torch.from_numpy(bits.astype(np.uint32).view(np.int32).copy()).view(torch.float32)


def f32_bits(t: torch.Tensor) -> np.ndarray:
    return t.reshape(-1).cpu().numpy().view(np.uint32)


def _tensors(device: str):
    a, sa, b, sw, rows = edge_call()
    return (torch.from_numpy(a).to(device), f32_tensor(sa).to(device), torch.from_numpy(b).to(device),
            f32_tensor(sw).to(device), rows)


def _host(y: torch.Tensor, zy: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    return y.view(torch.int16).cpu().numpy().view(np.uint16), zy.cpu().numpy()


def _f32_class(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(finite, subnormal, NaN) of f32 encodings."""
    e, f = v & 0x7F800000, v & 0x7FFFFF
    return e != 0x7F800000, (e == 0) & (f != 0), (e == 0x7F800000) & (f != 0)


def coverage(z: np.ndarray, sa: np.ndarray, sw: np.ndarray) -> dict[str, int]:
    """How many of a grid's cells hit each edge, by the numpy oracle's f32 before the cast (u) and its y."""
    s = np.broadcast_to(sa[:, None], z.shape)
    w = np.broadcast_to(sw[None, :], z.shape)
    u, y = oracle(z, s, w)
    ub, zs = u.view(np.uint32), z.view(np.int32).astype(np.int64)
    (s_fin, s_sub, s_nan), (w_fin, w_sub, w_nan) = _f32_class(s), _f32_class(w)
    y_fin, y_nonzero = (y & 0x7F80) != 0x7F80, (y & 0x7FFF) != 0
    tie = np.isfinite(u) & ((ub & 0xFFFF) == 0x8000)
    return {"cells": int(z.size),
            "z = 0, y -0": int(((zs == 0) & (y == 0x8000)).sum()),
            "z < 0, y finite and nonzero": int(((zs < 0) & y_fin & y_nonzero).sum()),
            "negative scale, y finite and nonzero": int(((((s | w) & NEG) != 0) & ~s_nan & ~w_nan & y_fin & y_nonzero).sum()),
            "subnormal scale, y nonzero": int(((s_sub | w_sub) & y_nonzero).sum()),
            "f32 product subnormal": int(_f32_class(ub)[1].sum()),
            "finite scales, y ±inf": int((s_fin & w_fin & ((y == 0x7F80) | (y == 0xFF80))).sum()),
            "NaN scale, y 0x7FFF": int(((s_nan | w_nan) & (y == 0x7FFF)).sum()),
            "no NaN scale (0 x inf), y 0x7FFF": int((~s_nan & ~w_nan & (y == 0x7FFF)).sum()),
            "|z| > 2^24, S32ToF32 rounds": int(((np.abs(zs) > 1 << 24) & (rn24(zs) != zs)).sum()),
            "bf16 tie, kept even (down)": int((tie & ((ub & 0x10000) == 0)).sum()),
            "bf16 tie, to even (up)": int((tie & ((ub & 0x10000) != 0)).sum()),
            "S32ToF32 tie then bf16 tie, 0x4B80": int(((zs == (1 << 24) + (1 << 16) + 1) & (y == 0x4B80)).sum())}


# -- CPU -----------------------------------------------------------------------------------------------------------------

def test_the_edge_call_reaches_every_edge(record_property):
    """Every edge CA named is among the edge call's cells: -0 and negative scales, subnormal scales and a subnormal
    product, finite scales whose product overflows to ±inf, NaN scales and 0 x inf (0x7FFF), |z| past 2^24 where
    S32ToF32 rounds, z = 0 and negative z, and bf16 ties both ways (one of them after S32ToF32's own tie)."""
    a, sa, b, sw, rows = edge_call()
    z = exact_z(a, b, rows)
    assert a.shape == ((len(TARGETS) + 6) * len(SA), K_EDGE) and b.shape == (6 * len(SW), K_EDGE) and rows > a.shape[0] + UR - 1
    assert z[:len(TARGETS), 0].view(np.int32).tolist() == TARGETS, "the fullc column holds the targets"
    assert int(z[len(TARGETS), len(SW)]) == 127 * 127 * K_EDGE, "all +127 against all +127"
    cov = coverage(z, slot_scales(sa, rows), sw)
    record_property("coverage", cov)
    assert all(cov.values()), cov


def test_the_cpu_body_is_the_definition_on_the_edge_call_but_for_nans_encoding(record_property):
    """The op's torch body (`int8_exact.scaled_mm_zy`, the reference the CUDA body equals) on the edge call: records and
    served y are `h1zy.dequant`'s and `PouwZyDequant_v2`'s, except that a NaN y is torch's CPU encoding, not 0x7FFF."""
    xq, sa, w, sw, rows = _tensors("cpu")
    y, zy = OPS.scaled_mm_zy(xq, sa, w, sw, rows)
    z, sas, want = edge_expected()
    pin(*_host(y, zy), z, sas, edge_call()[3], want, nan_any=True)
    record_property("cells", int(want.size + y.numel()))


def _encodings(seed: int, n: int) -> np.ndarray:
    """[n, 3] (z, s_a, s_w) encodings: z over all of s32 and over served dots' range; scales uniform over every f32
    encoding (NaNs, infinities, subnormals), served-like, and near the exponent range's ends (products under- and
    overflow); and bf16 ties and their neighbours, both as an exact integer z at power-of-two scales and as a scale."""
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 1 << 32, n, dtype=np.uint64)
    near = rng.random(n) < 0.6
    z[near] = rng.integers(-(1 << 28), 1 << 28, int(near.sum())).astype(np.int64) & M32

    def scale(m: int) -> np.ndarray:
        s = rng.integers(0, 1 << 32, m, dtype=np.uint64)
        kind = rng.integers(0, 3, m)
        served = np.abs(rng.normal(0, 0.03 / 127, m)).astype(np.float32).view(np.uint32)
        exp = rng.choice(np.array([0, 1, 2, 3, 0x7C, 0x7D, 0xFC, 0xFD, 0xFE], np.uint64), m)
        made = (rng.integers(0, 2, m, dtype=np.uint64) << 31) | (exp << 23) | rng.integers(0, 1 << 23, m, dtype=np.uint64)
        s[kind == 1] = served[kind == 1]
        s[kind == 2] = made[kind == 2]
        return s

    sa, sw = scale(n), scale(n)
    t = n // 8
    width = rng.integers(9, 32, t)
    top = rng.integers(128, 256, t).astype(np.int64)
    tz = (top << (width - 8)) | (np.int64(1) << (width - 9))
    tz = tz + rng.choice(np.array([-1, 0, 0, 1]), t) * (width > 9)
    tz = np.where(rng.random(t) < 0.5, -tz, tz)
    z[:t] = tz.astype(np.int64) & M32
    sa[:t] = rng.choice(np.array([ONE, NEG | ONE, HALF, P2M10, TWO], np.uint64), t)
    sw[:t] = rng.choice(np.array([ONE, NEG | ONE, HALF, P2M10, TWO], np.uint64), t)
    v = (rng.integers(0, 1 << 16, t, dtype=np.uint64) << 16) | rng.choice(np.array([0x7FFF, 0x8000, 0x8000, 0x8001],
                                                                                       np.uint64), t)
    z[t:2 * t] = rng.choice(np.array([1, M32, 2, 1 << 20], np.uint64), t)
    sa[t:2 * t], sw[t:2 * t] = v, rng.choice(np.array([ONE, HALF, NEG | ONE], np.uint64), t)
    return np.stack([z, sa, sw], axis=1).astype(np.uint32)


def _edge_encodings() -> np.ndarray:
    zs = sorted({t & M32 for t in TARGETS} | {(1 << 31) - 1, 1 << 31, (1 << 31) - 64, (1 << 31) - 65, M32 - 127,
                                              (1 << 31) + 64, 1 << 30, (1 << 30) + 65})
    return np.array([(p, q, r) for p in zs for q in SA for r in SW], np.uint32)


@pytest.mark.parametrize("source", ["edges", "random"])
def test_h1zy_dequant_is_int8_exacts_stated_oracle(source, record_property):
    """`h1zy.dequant` against RN_bf16(RN32(RN32(RN32(z) s_a) s_w)) computed step by step in numpy float32, the cast an
    explicit round to nearest even on the encoding, every NaN 0x7FFF: on every (z, s_a, s_w) of the edge targets, the s32
    extremes and the edge call's scales, and on 300,000 random encodings with 75,000 bf16 ties and neighbours; and
    numpy's int32 -> float32 is RN32 (ties to even on 24 significant bits, in integers)."""
    tri = _edge_encodings() if source == "edges" else _encodings(33, 300_000)
    zs = tri[:, 0].view(np.int32).astype(np.int64)
    assert np.array_equal(zs.astype(np.float32).astype(np.int64), rn24(zs))
    want = oracle(tri[:, 0], tri[:, 1], tri[:, 2])[1]
    got = np.array([Z.dequant(int(p), int(q), int(r)) for p, q, r in tri], np.uint16)
    bad = np.flatnonzero(got != want)
    assert not bad.size, [(f"0x{int(tri[i, 0]):08x}", f"0x{int(tri[i, 1]):08x}", f"0x{int(tri[i, 2]):08x}",
                           f"h1zy 0x{int(got[i]):04x}", f"oracle 0x{int(want[i]):04x}") for i in bad[:8]]
    u = oracle(tri[:, 0], tri[:, 1], tri[:, 2])[0]
    tie = np.isfinite(u) & ((u.view(np.uint32) & 0xFFFF) == 0x8000)
    assert tie.any() and (want == 0x7FFF).any() and ((want & 0x7FFF) == 0x7F80).any()
    record_property("encodings", len(tri))
    record_property("bf16 ties", int(tie.sum()))


def test_the_definition_on_bits_is_the_oracle_on_the_same_encodings(record_property):
    """`PouwZyDequant_v2` itself, evaluated on bits, against the numpy oracle on the same edge and random encodings: the
    word view the GPU tests compare with is the Definition's on these words too."""
    tri = np.concatenate([_edge_encodings(), _encodings(33, 300_000)])
    want = oracle(tri[:, 0], tri[:, 1], tri[:, 2])[1]
    got = dequant_bits(tri)
    bad = np.flatnonzero(got != want)
    assert not bad.size, [(f"0x{int(tri[i, 0]):08x}", f"0x{int(tri[i, 1]):08x}", f"0x{int(tri[i, 2]):08x}",
                           f"bits 0x{int(got[i]):04x}", f"oracle 0x{int(want[i]):04x}") for i in bad[:8]]
    record_property("encodings", len(tri))


# -- CUDA ----------------------------------------------------------------------------------------------------------------

@cuda
def test_cuda_served_y_and_records_are_the_definition_on_the_edge_call(record_property):
    """The served op on CUDA (`int8_fused.product_zy` at `_plan`'s config) on the edge call: every record parses as
    `h1zy.RECORD` into the exact z and the Definition's y (words and bits), the served y is its live rows, and every unit
    passes `h1zy.check` (the unit's ok bit on words) against the exact z and its rows' and columns' scales."""
    from verity_vllm.engine import int8_fused as F
    xq, sa, w, sw, rows = _tensors("cuda")
    y, zy = OPS.scaled_mm_zy(xq, sa, w, sw, rows)
    torch.cuda.synchronize()
    record_property("config", str(F.pick(xq.shape[0], w.shape[0], xq.shape[1])))
    z, sas, want = edge_expected()
    sw_bits = edge_call()[3]
    zr, yr = pin(*_host(y, zy), z, sas, sw_bits, want)
    n = z.shape[1]
    for u in range(zy.shape[0]):
        bi, ci = divmod(u, n // UC)
        r, c = slice(bi * UR, bi * UR + UR), slice(ci * UC, ci * UC + UC)
        assert Z.check(zr[r, c].ravel().tolist(), yr[r, c].ravel().tolist(), z[r, c].ravel().tolist(),
                       sas[r].tolist(), sw_bits[c].tolist()) == 1, u
    record_property("cells", int(want.size + y.numel()))
    record_property("units", int(zy.shape[0]))


@cuda
@pytest.mark.slow
def test_cuda_every_config_writes_the_definitions_y_on_the_edge_call(record_property):
    """Every GEMM config (pointer tiles narrower and wider than a unit, split-K 2, 3 and 4, a row tile past T, TMA) on the
    edge call, into 0xA5-poisoned y and records, twice through one workspace: records and served y the Definition's."""
    from verity_vllm.engine import int8_fused as F
    xq, sa, w, sw, rows = _tensors("cuda")
    m, n = xq.shape[0], w.shape[0]
    z, sas, want = edge_expected()
    work = F.Workspace("cuda", words=m * n)
    done = []
    for text in _configs():
        cfg = F.Cfg.parse(text)
        y = torch.full((m, 2 * n), 0xA5, dtype=torch.uint8, device="cuda").view(torch.bfloat16)
        zy = torch.full((E.zy_units(m, n, rows), E.ZY_BYTES), 0xA5, dtype=torch.uint8, device="cuda")
        for _ in range(2):
            F.scaled_mm_zy(xq, sa, w, sw, y, zy, rows, cfg, work)
            torch.cuda.synchronize()
            pin(*_host(y, zy), z, sas, edge_call()[3], want)
        done.append(text)
    record_property("configs", done)


#: random calls at Llama-3.1-8B's linears through the served op: served-like scales (row_quant's s_a), and one call whose
#: scales are random encodings over most of f32's exponents and both signs
RANDOM = [("qkv", 200, "served"), ("o", 37, "served"), ("gate_up", 1, "served"), ("gate_up", 17, "served"),
          ("down", 64, "served"), ("down", 5, "served"), ("down", 33, "wide"), ("qkv", 32, "wide")]


@cuda
@pytest.mark.slow
@pytest.mark.parametrize("linear,m,scales", RANDOM)
def test_cuda_random_calls_at_llamas_linears_are_the_definition(linear, m, scales, record_property):
    """Every cell of every record's y and of the served y is `h1zy.dequant` of the cell's exact z (float64 on the device)
    at its slot's s_a and its column's s_w, and every record parses as `h1zy.RECORD`; T = max(64, M rounded up to 32)."""
    k, n = LLAMA[linear]
    rows = _rows(m)
    xq, sa, w, sw = _cuda_call(m, k, n, seed=m + k)
    if scales == "wide":
        g = np.random.default_rng(m * 7 + n)
        sa, sw = (f32_tensor((g.integers(0, 2, c) << 31) | (g.integers(0x30, 0xD0, c) << 23) | g.integers(0, 1 << 23, c)).cuda()
                  for c in (m, n))
    from verity_vllm.engine import int8_fused as F
    y, zy = OPS.scaled_mm_zy(xq, sa, w, sw, rows)
    record_property("config", str(F.pick(m, n, k)))
    slots = torch.zeros((rows, k), dtype=torch.float64, device="cuda")
    slots[:m] = xq.double()
    z = (slots @ w.double().t()).to(torch.int64).cpu().numpy()
    assert np.abs(z).max() < 1 << 31
    z = (z & M32).astype(np.uint32)
    sas, sw_bits = slot_scales(f32_bits(sa), rows), f32_bits(sw)
    want = expected(z, sas, sw_bits)
    pin(*_host(y, zy), z, sas, sw_bits, want)
    record_property("cells", int(want.size + y.numel()))
