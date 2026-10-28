"""``pearl-fp8-v4``: Pearl's FP8 proof of useful work, certificate v4, as published.

Sources: the whitepaper "Pearl Floating Point Scheme Specification" (Pearl Research, Sep 2026) and
pearl-research-labs/pearl's certificate-v4 code ("feat(fp8): certificate-v4 + mining") at ``284b147b``, whose verifier code
(``zk-pow/src/api/fp8``, ``zk-pow/src/circuit/fp8/unpredictability.rs``) rules where the two differ.  Conformance vectors
come from that code (``tests/vectors/pearl_kw.json``).  Pearl extends Komargodski–Weinstein's low-rank noise with
post-noise quantization, so the ticket hashes the output tile itself.

**The scheme, per tile** (rows I of A, rows J of B, full k; both operands stored as rows along k):

~~~text
commit     FP10 rows: k int8 codes, then k/8 BF16 scales (little-endian), one scale per 8 codes;  X = RNE_bf16(code·scale)
norms      l2 = grid4(RNE_bf16(sqrt(mean of squares))) exactly from the planes, linf = RNE_bf16(max |code|·|scale|)
seeds      key_A = H_"key-A"(σ̂), key_B = H_"key-B"(σ_d);  seed_B = H_"seed-B"(H_B ‖ key_B ‖ p_B);
           seed_A = H_"seed-A"(H_A ‖ seed_B ‖ key_A ‖ p_A);  H_ℓ(x; q) = BLAKE3(x; key = BLAKE3(ℓ; key = q))
noise      rank r = 32; line(seed, side, factor, ℓ) = 32 keyed-BLAKE3 bytes of (side ‖ factor ‖ ℓ u32 LE, zero-padded to 64),
           byte b -> ±(1 + b & 127), scaled to L2 norm 256 in BF16 and rounded to FP8;  E_A rows by seed_A, E_B rows,
           F_A and F_B (lines 0..k) by seed_B;  N = E·Fᵀ on the device's rank-32 FP8 atom, rounded to BF16
quantize   α = 448/(δ√r·l2 + linf), β = α·l2·δ√r/256² (BF16 ops);  X̃ = fp8(clamp(fma(α, X, β·N), ±448))
checked    the tile's FP32 accumulator of X̃_A·X̃_Bᵀ, exact for the device
digest     16 lanes, lane <- rotl32(lane·0x9E3779B1 + bits(x), 13) over the committed lane layout; J = H_"jackpot"(z; seed_A)
policy     liveness: dead entries (|X| >= 8δ·l2) <= 1/64 of each side; noise floor: δ·α·l2 >= 1 on every row;
           unpredictable summands: at most ⌊k·h·w/20⌋ skippable (λ_A + λ_B below the device grid of the cell's magnitude)
lottery    uint256LE(J) <= τ·h·w·k   (`Lottery`, ``benchmarks/pouw/clear/clear_audit.py``)
~~~

**Devices.**  ``b200`` (Pearl's: accumulation groups of 32 exact products plus the FP32 carry, 25 fractional bits below
the group's largest exponent, truncated, δ = 1/2, W = 25) is ported exactly.  ``ada`` is our extension for the RTX 4090,
not in Pearl's whitelist: its matmul is Verity's pinned ``sm89.mma.m16n8k32.e4m3`` model (`verity.ml.circuits.silicon`, groups (16, 16),
a 14-bit adder), δ = 1 and W = 13 as for Pearl's H100, and its census reads one magnitude per cell, as for B200.  Pearl's
``h100`` (a reset-and-promote replay over 128-product windows) is not ported.

**What the protocol replaces.**  The chain headers σ̂ and σ_d are the epoch's salt; H_A and H_B are the protocol's
activation and weight roots (bound to the unit's index and the weight id), not Pearl's keyed-BLAKE3 Merkle openings;
the tiles are the protocol's 16 x 16 grid with Pearl's layout ``[(4, Fold), (4, Blake)]`` on both axes.  The seed order is
kept: B is fixed before seed_B, and A before seed_A.

**Where Pearl differs from our milestone 1 (D1, the hardened KW transcript).**
- Datatype: FP8 E4M3 with FP32 accumulation, approximate and device-specific, against int7 on the int8 datapath, exact.
- Noise: rank 32 lines, ±{1..128} normalized to norm 256 then FP8, the same F basis for every row of a job; D1 has rank
  16 with inner ±1 factors, E_L and F_R in {−3..3}, and E_R shared only per group of 16 units (wider sharing admits a
  table attack).
- Checked values: the output tile only.  D1 checks the running sums after every 16-deep step, which Lemma 1 needs.
- Useful output: Pearl's peel and unscale (`useful`) lie outside the checked values (barrier class (a)).
- Inputs: Pearl restricts them by its policy; D1 takes worst-case inputs.
- Conjecture: Pearl's Assumption 1 (informal quantized-subspace hardness) with the policy; it names no γ_0, no concrete
  constant and no joint form.  The policy admits 5% skippable summands by design.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache

from verity.ml.crypto.blake3 import blake3, keyed
from verity.ml.circuits.silicon import models as tc_models

from ..protocol import GuaranteeBasis, Context, DomainError, Matmul, Shape, Tile, grid_tiles

NAME = "pearl-fp8-v4"
R = 32
BLOCK = 8
Q_MAX = 448.0
NOISE_NORM = 256.0
NORM_FLOOR_BF16 = 0x2F80                      # 2^-32
TAU_IDLE, EPS_IDLE, SIGMA_MIN = 8.0, 1 / 64, 1.0
TILE = 16
PATTERN = bytes([13, 14, 3, 3, 3, 3])         # AxisPattern [(4, Fold), (4, Blake)]: (len - 1) << 2 | type, padded with None
HASH_ID = 3                                   # Blake3Chunk1024; nominal, since the protocol's trees replace Pearl's
LABELS = {r: f"pearl/v4/FP8/{r}".encode() for r in ("key-A", "key-B", "seed-A", "seed-B", "noise-line", "jackpot")}


class Reject(ValueError):
    """Pearl's verifier rejects: a non-finite value, a malformed row."""


@dataclass(frozen=True, slots=True)
class CodeParams:
    name: str
    code: int            # the p_B device byte (H100 = 0, B200 = 1 in Pearl; 2 is ours)
    lg2_delta: int
    window_bits: int     # W, the census grid's bits below the window anchor
    width: int           # significand width of one rank-32 atom's grouped sum
    windowed: bool       # a fresh accumulator every 128 products, promoted to the total by one FP32 add (H100)

    @property
    def delta(self) -> float:
        return 2.0 ** self.lg2_delta

    @property
    def sigma_offset(self) -> int:
        return 14 + self.lg2_delta


H100 = CodeParams("h100", 0, 0, 13, 14, True)
B200 = CodeParams("b200", 1, -1, 25, 26, False)
ADA = CodeParams("ada", 2, 0, 13, 14, True)
DEVICES = {"h100": H100, "b200": B200, "ada": ADA}
WINDOW = 128


# ---- exact formats -------------------------------------------------------------------------------------------------

def f32_bits(x: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        raise Reject(f"{x} overflows FP32") from None


def f32_of_bits(w: int) -> float:
    return struct.unpack("<f", struct.pack("<I", w & 0xFFFFFFFF))[0]


def f32(x: float) -> float:
    """Round an f64 to FP32, nearest-even (exact for a single f32 op computed in f64: 53 >= 2·24 + 2)."""
    return f32_of_bits(f32_bits(x))


def bf16_to_f32(code: int) -> float:
    if (code >> 7) & 0xFF == 0xFF:
        raise Reject(f"non-finite bf16 {code:#06x}")
    return f32_of_bits(code << 16)


def f32_to_bf16(x: float) -> int:
    if not math.isfinite(x):
        raise Reject(f"non-finite {x}")
    bits = f32_bits(x)
    out = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
    if out & 0x7FFF >= 0x7F80:
        raise Reject(f"{x} overflows bf16")
    return out


def bf16_of_fraction(x: Fraction) -> int:
    """RNE of an exact rational to BF16 (subnormals included); overflow rejects."""
    if x == 0:
        return 0
    sign, a = (0x8000 if x < 0 else 0), abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    e = max(e, -126)
    ulp = Fraction(2) ** (e - 7)
    q, r = divmod(a, ulp)
    n = int(q)
    if r > ulp / 2 or (r == ulp / 2 and n & 1):
        n += 1
    value = n * ulp
    code = f32_to_bf16(float(value)) if value < Fraction(2) ** 128 else 0x7F80
    if code & 0x7FFF >= 0x7F80:
        raise Reject("bf16 overflow")
    return sign | code


def fp8_to_f32(code: int) -> float:
    if code & 0x7F == 0x7F:
        raise Reject("the E4M3 NaN code")
    s = -1.0 if code & 0x80 else 1.0
    e, m = (code >> 3) & 0xF, code & 7
    return s * (m * 2.0 ** -9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))


def f32_to_fp8(x: float) -> int:
    """Pearl's `f32_to_fp8_e4m3`: nearest-even, saturating to 448 (0x7E)."""
    if not math.isfinite(x):
        raise Reject(f"non-finite {x}")
    bits = f32_bits(x)
    sign = 0x80 if bits >> 31 else 0
    m = abs(x)
    e = ((bits >> 23) & 0xFF) - 127
    if e < -6:
        v = m * 512.0
        fl = math.floor(v)
        mag = fl + 1 if v - fl > 0.5 or (v - fl == 0.5 and fl & 1) else fl
    else:
        exp, fm = e + 7, bits & 0x7FFFFF
        mant, rem = fm >> 20, fm & 0xFFFFF
        if rem > 0x80000 or (rem == 0x80000 and mant & 1):
            mant += 1
            if mant == 8:
                mant, exp = 0, exp + 1
        mag = 0x7E if exp > 15 or (exp == 15 and mant >= 7) else (exp << 3) | mant
    return sign | mag


def bf16_mul(a: int, b: int) -> int:
    return f32_to_bf16(f32(bf16_to_f32(a) * bf16_to_f32(b)))


def bf16_div(a: int, b: int) -> int:
    x, y = bf16_to_f32(a), bf16_to_f32(b)
    if y == 0:
        raise Reject("division by zero")
    return f32_to_bf16(f32(x / y))


def bf16_fma(a: int, b: int, c: int) -> int:
    """One rounding of the exact ``a·b + c`` to BF16 (Pearl rounds to odd in FP32, then to nearest-even in BF16); an exact
    zero takes IEEE's sign (−0 only when both the product and c are −0)."""
    x, y, z = bf16_to_f32(a), bf16_to_f32(b), bf16_to_f32(c)
    exact = Fraction(x) * Fraction(y) + Fraction(z)
    return f32_to_bf16(x * y + z) if exact == 0 else bf16_of_fraction(exact)


def bf16_max(a: int, b: int) -> int:
    return a if bf16_to_f32(a) >= bf16_to_f32(b) else b


def bf16_min(a: int, b: int) -> int:
    return a if bf16_to_f32(a) <= bf16_to_f32(b) else b


def bf16_clamp_sym(x: int, hi: int) -> int:
    return bf16_max(bf16_min(x, hi), hi ^ 0x8000)


# ---- hashing and noise -------------------------------------------------------------------------------------------

def subkey(role: str, key: bytes | None = None) -> bytes:
    return keyed(key, LABELS[role]) if key is not None else blake3(LABELS[role])


def hash_labelled(msg: bytes, role: str, key: bytes | None = None) -> bytes:
    return keyed(subkey(role, key), msg)


@lru_cache(maxsize=64)
def _line_key(seed: bytes) -> bytes:
    return subkey("noise-line", seed)


def sample_line(seed: bytes, side: int, factor: int, line: int, norm: float = NOISE_NORM) -> list[int]:
    """One rank-32 noise line: FP8 codes.  r = 32 bytes is exactly BLAKE3's digest, the XOF's first block.  `norm` is the
    line's L2 norm, Pearl's 256 unless a scheme pins another (Pearl-C v1's F lines: 16)."""
    material = (bytes([side, factor]) + line.to_bytes(4, "little")).ljust(64, b"\0")
    raw = keyed(_line_key(seed), material)[:R]
    xs = [(-1 if b >> 7 else 1) * ((b & 0x7F) + 1) for b in raw]
    norm_scaled = math.isqrt(sum(x * x for x in xs) * 32 * 32)
    scale = bf16_div(f32_to_bf16(norm * 32), f32_to_bf16(float(norm_scaled)))
    return [f32_to_fp8(bf16_to_f32(bf16_mul(f32_to_bf16(float(x)), scale))) for x in xs]


# ---- the device matmul ---------------------------------------------------------------------------------------------

_ZERO_EXP = -139
_Term = tuple[bool, int, int]          # (negative, exponent, significand): Pearl's GFloat


def _product(x: int, y: int) -> _Term:
    ex, ey = (x >> 3) & 0xF, (y >> 3) & 0xF
    sx, sy = (x & 7) | (8 if ex else 0), (y & 7) | (8 if ey else 0)
    sig = (sx * sy) << 17
    neg = bool((x ^ y) & 0x80)
    return (neg, _ZERO_EXP, 0) if sig == 0 else (neg, max(ex, 1) + max(ey, 1) - 14, sig)


def _from_f32(v: float) -> _Term:
    w = f32_bits(v)
    s, e, m = bool(w >> 31), (w >> 23) & 0xFF, w & 0x7FFFFF
    if e == 0 and m == 0:
        return (s, -133, 0)
    return (s, (e if e else 1) - 127, (0x800000 | m) if e else m)


def _to_f32(g: _Term) -> float:
    s, e, m = g
    if m == 0:
        return -0.0 if s else 0.0
    eb = e + 127 - (0 if m & 0x800000 else 1)
    if eb < 0:
        raise Reject("exponent below FP32")
    out = f32_of_bits((int(s) << 31) | ((eb & 0xFF) << 23) | (m & 0x7FFFFF))
    if not math.isfinite(out):
        raise Reject("non-finite accumulator")
    return out


def _group(values: list[_Term], width: int) -> _Term:
    """Pearl's `windowed_group_sum`: align to the largest exponent, truncate each term to the window, sum exactly,
    renormalize to `width` bits and truncate to FP32."""
    live = [v for v in values if v[2]]
    if not live:
        return (False, _ZERO_EXP, 0)
    mx = max(v[1] for v in live)
    neg_shift, diff = width < 24, abs(width - 24)
    acc = 0
    for s, e, m in live:
        sh = mx - e
        if sh >= 32:
            continue
        aligned = ((m >> diff) if neg_shift else (m << diff)) >> sh
        acc += -aligned if s else aligned
    sign, sig = acc < 0, abs(acc) & 0xFFFFFFFF
    if sig == 0:
        return (sign, _ZERO_EXP, 0)
    w = sig.bit_length()
    exp = mx + w - width
    sig = sig >> (w - width) if w > width else sig << (width - w)
    if exp < -126:
        sh = -126 - exp
        sig = 0 if sh >= 32 else sig >> sh
        exp = -126
    sig = (sig << diff) if neg_shift else (sig >> diff)
    return (sign, _ZERO_EXP, 0) if sig == 0 else (sign, exp, sig)


def _atom(dev: CodeParams, local: float, a: Sequence[int], b: Sequence[int]) -> float:
    """One rank-32 atom into the local accumulator."""
    if dev is ADA:
        word = tc_models.ADA_E4M3_M16N8K32.step(f32_bits(local), list(a), list(b))
        if (word >> 23) & 0xFF == 0xFF:
            raise Reject("non-finite accumulator")
        return f32_of_bits(word)
    return _to_f32(_group([_from_f32(local), *(_product(x, y) for x, y in zip(a, b))], dev.width))


def _binade(x: float | Fraction) -> int | None:
    """floor(log2 |x|), exactly, or None for zero."""
    x = abs(Fraction(x))
    if x == 0:
        return None
    e = x.numerator.bit_length() - x.denominator.bit_length()
    return e if Fraction(2) ** e <= x else e - 1


def _canon(x: float) -> float:
    if not math.isfinite(x):
        raise Reject("non-finite accumulator")
    return 0.0 if x == 0 else x


def replay(dev: CodeParams, a: Sequence[int], b: Sequence[int]) -> tuple[float, list[float], list[tuple[int, int, int]]]:
    """One cell ``a·b`` on the device: (value, the partial after each atom, the census windows ``(start, end, grid)``).

    B200 chains every 32-product group into the carry (Pearl's `B200::matmul_fp8`); its census reads one grid from the
    largest product or partial.  H100 and ada start a fresh accumulator at +0 every 128 products, chain up to four atoms,
    and promote with one FP32 nearest-even add (Pearl's `H100::matmul_fp8_replay`); their census reads one grid per
    window, from the window's largest product or local value and the cell's largest exact ``total + local``."""
    k = len(a)
    if not dev.windowed:
        partials: list[float] = []
        acc = [_from_f32(0.0)]
        for x, y in zip(a, b):
            acc.append(_product(x, y))
            if len(acc) == 33:
                v = _to_f32(_group(acc, dev.width))
                partials.append(v)
                acc = [_from_f32(v)]
        value = _to_f32(_group(acc, dev.width))
        if len(acc) > 1:
            partials.append(value)
        mag = max([abs(f32(fp8_to_f32(x) * fp8_to_f32(y))) for x, y in zip(a, b)] + [abs(p) for p in partials] + [0.0])
        return value, partials, [(0, k, grid_from_binades(dev, _binade(mag), _binade(mag)))]
    if k % 32:
        raise Reject("the device multiplies in atoms of k = 32")
    total, m_cell, zs, partials = 0.0, None, [], []
    for ws in range(0, k, WINDOW):
        we, local, z = min(ws + WINDOW, k), 0.0, None
        for s in range(ws, we, 32):
            for x, y in zip(a[s:s + 32], b[s:s + 32]):
                pb = _binade(fp8_to_f32(x) * fp8_to_f32(y))
                if pb is not None:
                    z, m_cell = max(z, pb) if z is not None else pb, max(m_cell, pb) if m_cell is not None else pb
            local = _atom(dev, local, a[s:s + 32], b[s:s + 32])
            lb, sb = _binade(local), _binade(Fraction(total) + Fraction(local))
            if lb is not None:
                z = lb if z is None else max(z, lb)
            if sb is not None:
                m_cell = sb if m_cell is None else max(m_cell, sb)
            partials.append(_canon(f32(total + local)))
        zs.append((ws, we, z))
        total = _canon(f32(total + local))
    return total, partials, [(s, e, grid_from_binades(dev, z, m_cell)) for s, e, z in zs]


def device_dot(dev: CodeParams, a: Sequence[int], b: Sequence[int], partials: list | None = None) -> float:
    value, parts, _ = replay(dev, a, b)
    if partials is not None:
        partials.extend(parts)
    return value


# ---- FP10 planes, norms, noisy quantization ---------------------------------------------------------------------

def parse_row(row: bytes, k: int) -> tuple[list[int], list[int]]:
    if len(row) != k + 2 * (k // BLOCK):
        raise Reject("an FP10 row has k codes and k/8 BF16 scales")
    codes = [x - 256 if x >= 128 else x for x in row[:k]]
    scales = list(struct.unpack(f"<{k // BLOCK}H", row[k:]))
    for s in scales:
        if (s >> 7) & 0xFF == 0xFF:
            raise Reject("a non-finite scale")
    return codes, scales


def fp10_row(codes: Sequence[int], scales: Sequence[int]) -> bytes:
    """The row encoding: k two's-complement codes, then k/8 little-endian BF16 scales."""
    return bytes(c & 0xFF for c in codes) + struct.pack(f"<{len(scales)}H", *scales)


def open_row(codes: Sequence[int], scales: Sequence[int]) -> list[int]:
    return [f32_to_bf16(f32(c * bf16_to_f32(scales[j // BLOCK]))) for j, c in enumerate(codes)]


def exact_norms(codes: Sequence[int], scales: Sequence[int]) -> tuple[int, int]:
    """Pearl's `exact_norms` for one row: l2 = grid4(RNE_bf16(sqrt(v))) with v the exact mean of squares in Pearl's frame,
    linf = RNE_bf16(max |code|·|scale|)."""
    k = len(codes)
    wl2 = 27 - (k - 1).bit_length() if k > 1 else 27
    blocks = [(codes[i:i + BLOCK], scales[i // BLOCK]) for i in range(0, k, BLOCK)]

    def decode(s: int) -> tuple[int, int]:
        e = (s & 0x7FFF) >> 7
        return (s & 0x7F) + (128 if e else 0), 2 * max(e, 1)

    e_max, linf = 0, 0.0
    for blk, s in blocks:
        amax = max(abs(v) for v in blk)
        m, de = decode(s)
        if amax and m:
            e_max = max(e_max, de)
        linf = max(linf, f32(amax * bf16_to_f32(s & 0x7FFF)))
    total = 0
    for blk, s in blocks:
        m, de = decode(s)
        p = m * m * sum(v * v for v in blk)
        sh = max(e_max - de, 0)
        total += (p << wl2) >> sh if sh < 64 else 0
    if total == 0:
        l2 = 0
    else:
        v = Fraction(total, k) * Fraction(2) ** (e_max - 268 - wl2)
        l2 = bf16_sqrt(v)
        l2 = (l2 + 2) & ~3 & 0xFFFF
    if l2 >= 0x7F80:
        raise Reject("l2 overflows bf16")
    return l2, f32_to_bf16(linf)


def bf16_sqrt(v: Fraction) -> int:
    """RNE_bf16(sqrt(v)) exactly, for rational v > 0 (subnormal results included)."""
    y = math.sqrt(float(v)) if v < Fraction(2) ** 1000 else float("inf")
    code = f32_to_bf16(max(y, 2.0 ** -140))

    def val(c: int) -> Fraction:
        return Fraction(bf16_to_f32(c))

    def mid_sq(c1: int, c2: int) -> Fraction:
        return ((val(c1) + val(c2)) / 2) ** 2

    while True:
        lo_ok = code == 0 or mid_sq(code - 1, code) < v or (mid_sq(code - 1, code) == v and code % 2 == 0)
        hi_ok = mid_sq(code, code + 1) > v or (mid_sq(code, code + 1) == v and code % 2 == 0)
        if lo_ok and hi_ok:
            return code
        code += -1 if not lo_ok else 1


def delta_consts(dev: CodeParams) -> tuple[int, int]:
    """(δ√r, δ√r/256²) as BF16, each one rounding of the f64 value through FP32 (Pearl's double rounding, proved equal)."""
    d = dev.delta * math.sqrt(R)
    return f32_to_bf16(f32(d)), f32_to_bf16(f32(d / (NOISE_NORM * NOISE_NORM)))


@dataclass(slots=True)
class Built:
    """One side of a tile: clean BF16 rows, and what noisy quantization recorded."""

    clean: list[list[int]]
    codes: list[list[int]]
    alpha: list[int]
    beta: list[int]
    l2: list[int]
    noise: list[list[int]]      # N = E·Fᵀ rows, BF16


def noisy_quantize(dev: CodeParams, clean: list[list[int]], norms: list[tuple[int, int]], e: list[list[int]], f: list[list[int]]) -> Built:
    delta_r, delta_over_std = delta_consts(dev)
    q_max, floor = f32_to_bf16(Q_MAX), NORM_FLOOR_BF16
    codes, alphas, betas, l2s, noises = [], [], [], [], []
    for row, (l2, linf), e_row in zip(clean, norms, e):
        l2, linf = bf16_max(l2, floor), bf16_max(linf, floor)
        alpha = bf16_div(q_max, bf16_fma(delta_r, l2, linf))
        beta = bf16_mul(bf16_mul(alpha, l2), delta_over_std)
        noise = [f32_to_bf16(device_dot(dev, e_row, f_line)) for f_line in f]
        codes.append([f32_to_fp8(bf16_to_f32(bf16_clamp_sym(bf16_fma(alpha, x, bf16_mul(beta, nv)), q_max)))
                      for x, nv in zip(row, noise)])
        alphas.append(alpha)
        betas.append(beta)
        l2s.append(l2)
        noises.append(noise)
    return Built(clean, codes, alphas, betas, l2s, noises)


# ---- the policy's census -------------------------------------------------------------------------------------------

LOG_BOUNDS = (
    32768, 33125, 33486, 33851, 34219, 34592, 34969, 35349, 35734, 36123, 36517, 36914, 37316, 37723, 38133, 38549, 38968, 39393,
    39822, 40255, 40694, 41137, 41585, 42038, 42495, 42958, 43426, 43899, 44377, 44860, 45348, 45842, 46341, 46846, 47356, 47872,
    48393, 48920, 49453, 49991, 50536, 51086, 51642, 52205, 52773, 53348, 53929, 54516, 55109, 55710, 56316, 56929, 57549, 58176,
    58810, 59450, 60097, 60752, 61413, 62082, 62758, 63441, 64132, 64831,
)
GRID_BIAS, M_WINDOW_BITS = 157, 18
SKIP_OFFSET = 64 * (16 + 2 * 508) - 128 * GRID_BIAS


def _log2_fixed(n: int) -> int:
    if n == 0:
        return 0
    w = n.bit_length() - 1
    norm = n << (15 - w)
    frac = sum(1 for b in LOG_BOUNDS if b <= norm) - 1
    return 64 * w + frac


def _e_star_m(code: int) -> tuple[int, int]:
    exp, man = (code >> 7) & 0xFF, code & 0x7F
    if exp == 255:
        raise Reject("non-finite bf16")
    return (1, man) if exp == 0 else (exp, 128 + man)


def lam(dev: CodeParams, alpha: int, x: int, l2: int) -> int:
    """Pearl's λ: a fixed-point lower bound on 64·(log2(X̄² + σ²) + 508) for a clean entry x of a row with (α, l2)."""
    ae, am = _e_star_m(alpha)
    xe, xm = _e_star_m(x)
    prod = am * xm
    width = prod.bit_length()
    x_norm, x_enc = (0, 0) if xm == 0 else (prod << (16 - width), ae + xe - 1 + width)
    le, lm = _e_star_m(l2)
    sp = am * lm
    wide = int(sp >= 1 << 15)
    s_norm, s_enc = sp << (1 - wide), ae + le + wide + dev.sigma_offset
    x_dom = x_enc >= s_enc and x_norm != 0
    gap = abs(x_enc - s_enc)
    near = 0 if gap >= 16 else gap
    big, small = (x_norm, s_norm) if x_dom else (s_norm, x_norm)
    quotient = ((small * small) >> near) >> near
    sos = big * big + (0 if gap >= 16 else quotient)
    return 128 * max(x_enc, s_enc) - 2624 + _log2_fixed(sos >> 17)


def grid_from_binades(dev: CodeParams, z: int | None, m: int | None) -> int:
    """Pearl's `grid_exponent_from_binades`: the larger of e(Z) − W and e(M) − 18, biased by 157; 0 when both are zero."""
    qs = [q for q in ((None if z is None else z - dev.window_bits), (None if m is None else m - M_WINDOW_BITS)) if q is not None]
    return 0 if not qs else max(qs) + GRID_BIAS


def grid_exponent(dev: CodeParams, z: float, m: float) -> int:
    return grid_from_binades(dev, _binade(z), _binade(m))


def skip(la: int, lb: int, grid: int) -> bool:
    return grid != 0 and la + lb < 128 * grid + SKIP_OFFSET


def budget(k: int, h: int, w: int) -> int:
    return k * h * w // 20


# ---- the scheme ---------------------------------------------------------------------------------------------------

def lanes(h: int = TILE, w: int = TILE) -> list[list[int]]:
    """Pearl's `lane_assignment` for [(4, Fold), (4, Blake)] on both axes: lane (rb, cb) folds its 4 x 4 subtile row-major."""
    return [[(4 * rb + fr) * w + 4 * cb + fc for fr in range(4) for fc in range(4)] for rb in range(4) for cb in range(4)]


LANES = lanes()


def extract(tile: list[list[float]]) -> bytes:
    flat = [x for row in tile for x in row]
    out = b""
    for idx in LANES:
        lane = 0
        for i in idx:
            lane = ((lane * 0x9E3779B1 + f32_bits(flat[i])) & 0xFFFFFFFF)
            lane = ((lane << 13) | (lane >> 19)) & 0xFFFFFFFF
        out += lane.to_bytes(4, "little")
    return out


@dataclass(frozen=True, slots=True)
class Noise:
    seed_a: bytes
    seed_b: bytes


@dataclass(slots=True)
class Ops:
    a: Built
    b: Built
    tile: list[list[float]] | None = None
    windows: list[list[list[tuple[int, int, int]]]] = field(default_factory=list)


@lru_cache(maxsize=8)
def _f_basis(seed_b: bytes, side: int, k: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(sample_line(seed_b, side, 1, u)) for u in range(k))


def p_b(n: int, k: int, dev: CodeParams) -> bytes:
    return n.to_bytes(4, "little") + k.to_bytes(4, "little") + R.to_bytes(2, "little") + bytes([0, dev.code, HASH_ID]) + PATTERN + (0).to_bytes(2, "little")


def p_a(m: int) -> bytes:
    return m.to_bytes(4, "little") + bytes([HASH_ID]) + PATTERN


class PearlKW:
    """Pearl's certificate-v4 FP8 scheme on one device: ``ada`` for the RTX 4090 (the default), or Pearl's ``h100``, ``b200``."""

    def __init__(self, device: str = "ada"):
        self.device = DEVICES[device]
        self.name = NAME
        self.certificate = GuaranteeBasis(
            conjecture="pearl-quantized-subspace-hardness",
            gamma0=None,
            omega=Fraction(1),
            claims=("pearl-quantized-subspace-hardness", "random-oracle", "cr/blake3", "cr/sha-256", "device-reproducibility",
                    "live-verifier"),
            theorems=("Pouw.SecurityProofs.WorkWeightedSampling",),
            hypotheses=("Pearl's Assumption 1 with its jackpot policy (informal: no γ_0, no constant, no joint form)",
                        f"the {self.device.name} device's FP8 accumulation is reproduced exactly"),
            allowances=("up to ⌊k·h·w/20⌋ skippable summands per tile (5%)", "up to 1/64 dead entries per side",
                        "the useful output's peel and unscale are unchecked (barrier class (a))"),
        )

    def check(self, matmul: Matmul) -> None:
        s = matmul.shape
        if not (1024 <= s.k <= 1 << 16 and s.k % 32 == 0 and s.m % TILE == 0 and s.n % TILE == 0 and s.m <= 1 << 24 and s.n <= 1 << 24):
            raise DomainError(f"{NAME}: {s} is outside 1024 <= k <= 2^16 with 32 | k, and 16 | m, n <= 2^24")

    def tiles(self, shape: Shape) -> list[Tile]:
        return grid_tiles(shape, TILE, TILE)

    def wref(self, shape: Shape) -> int:
        """The FP8 GEMM with FP32 accumulation at 2 W1 units per MAC (512 per SM per clock on the 4090), and the peel in
        BF16 at 4 units per MAC: A·F_B (m·k·r) and the 2r extra columns (m·n·2r).  Noise, quantization, hashing and memory
        are Pearl's exclusions; the harness's γ accounting prices them."""
        m, k, n = shape.m, shape.k, shape.n
        return 2 * m * k * n + 4 * (m * k * R + 2 * R * m * n)

    def noise(self, ctx: Context) -> Noise:
        u, header = ctx.matmul, ctx.salt
        key_a, key_b = hash_labelled(header, "key-A"), hash_labelled(header, "key-B")
        seed_b = hash_labelled(ctx.root_b + key_b + p_b(u.shape.n, u.shape.k, self.device), "seed-B")
        seed_a = hash_labelled(ctx.root_a + seed_b + key_a + p_a(u.shape.m), "seed-A")
        return Noise(seed_a, seed_b)

    def _side(self, rows: Sequence[bytes], idx: Sequence[int], side: int, seed_e: bytes, seed_b: bytes, k: int) -> Built:
        planes = [parse_row(r, k) for r in rows]
        clean = [open_row(c, s) for c, s in planes]
        norms = [exact_norms(c, s) for c, s in planes]
        e = [sample_line(seed_e, side, 0, i) for i in idx]
        return noisy_quantize(self.device, clean, norms, e, [list(x) for x in _f_basis(seed_b, side, k)])

    def encode(self, ctx: Context, noise: Noise, a_rows: Sequence[bytes], b_rows: Sequence[bytes], tile: Tile) -> Ops | None:
        k = ctx.matmul.shape.k
        try:
            return Ops(self._side(a_rows, tile.rows, 0, noise.seed_a, noise.seed_b, k),
                       self._side(b_rows, tile.cols, 1, noise.seed_b, noise.seed_b, k))
        except Reject:
            return None

    def _replay(self, ops: Ops) -> list[list[float]]:
        if ops.tile is None:
            cells = [[replay(self.device, ra, rb) for rb in ops.b.codes] for ra in ops.a.codes]
            ops.tile = [[c[0] for c in row] for row in cells]
            ops.windows = [[c[2] for c in row] for row in cells]
        return ops.tile

    def admissible(self, ctx: Context, ops: Ops) -> bool:
        """Pearl's jackpot policy (certificate-v4): entry liveness, the noise floor, and the unpredictable-summands census."""
        dev, k = self.device, ctx.matmul.shape.k
        for side in (ops.a, ops.b):
            dead = sum(1 for row, l2 in zip(side.clean, side.l2) for x in row
                       if abs(bf16_to_f32(x)) >= TAU_IDLE * dev.delta * bf16_to_f32(l2))
            if dead > EPS_IDLE * len(side.clean) * k:
                return False
        for side in (ops.a, ops.b):
            if any(dev.delta * bf16_to_f32(a) * bf16_to_f32(l) < SIGMA_MIN for a, l in zip(side.alpha, side.l2)):
                return False
        try:
            self._replay(ops)
        except Reject:
            return False
        lam_a = [[lam(dev, a, x, l) for x in row] for row, a, l in zip(ops.a.clean, ops.a.alpha, ops.a.l2)]
        lam_b = [[lam(dev, a, x, l) for x in row] for row, a, l in zip(ops.b.clean, ops.b.alpha, ops.b.l2)]
        skippable = 0
        for i, la in enumerate(lam_a):
            for j, lb in enumerate(lam_b):
                for start, end, g in ops.windows[i][j]:
                    skippable += sum(1 for x, y in zip(la[start:end], lb[start:end]) if skip(x, y, g))
        return skippable <= budget(k, len(lam_a), len(lam_b))

    def checked(self, ctx: Context, ops: Ops) -> list[list[float]]:
        return self._replay(ops)

    def digest(self, ctx: Context, noise: Noise, checked: list[list[float]]) -> bytes:
        return hash_labelled(extract(checked), "jackpot", noise.seed_a)

    def useful(self, ctx: Context, noise: Noise, ops: Ops, checked: list[list[float]]) -> list[list[float]]:
        """Peel and unscale in f64: C ≈ (C̃ − αβ′·A·N_Bᵀ − βα′·N_A·Bᵀ − ββ′·N_A·N_Bᵀ)/(αα′).  Outside Pearl's protocol."""
        na = [[bf16_to_f32(x) for x in row] for row in ops.a.noise]
        nb = [[bf16_to_f32(x) for x in row] for row in ops.b.noise]
        a = [[bf16_to_f32(x) for x in row] for row in ops.a.clean]
        b = [[bf16_to_f32(x) for x in row] for row in ops.b.clean]
        out = []
        for i in range(len(a)):
            al, be = bf16_to_f32(ops.a.alpha[i]), bf16_to_f32(ops.a.beta[i])
            row = []
            for j in range(len(b)):
                al2, be2 = bf16_to_f32(ops.b.alpha[j]), bf16_to_f32(ops.b.beta[j])
                an = sum(x * y for x, y in zip(a[i], nb[j]))
                nbv = sum(x * y for x, y in zip(na[i], b[j]))
                nn = sum(x * y for x, y in zip(na[i], nb[j]))
                row.append((checked[i][j] - al * be2 * an - be * al2 * nbv - be * be2 * nn) / (al * al2))
            out.append(row)
        return out
