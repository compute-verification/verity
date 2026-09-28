"""``fp8-is-h100-*``: genuine-FP8 proof of useful work on the H100: fresh k32 E4M3 tensor-core atoms and their checked
FP32 running sums, over operands a swappable forming layer produces (``PROTOCOL.md`` §fp8-is-h100; the FP8 lane's C-2,
turned into a spec by track A).

**The core** (fixed, this module).  A forming layer turns a tile's committed E4M3 rows into stacked operands: ``x[i]``,
the tile's row i of X̃, and ``y[j]``, its column j of Ỹ, each K = blocks·k E4M3 codes, block-major.  Cut K into 32-wide
slices S_1 … S_T (T = K/32) in ascending order.  For every output cell (i, j):

    I_τ = HOPPER_E4M3_WGMMA_K32.step(+0, x[i][S_τ], y[j][S_τ])   one accumulator word of wgmma m64nNk32 f32.e4m3.e4m3
    R_1 = I_1,  R_τ = add.rn.f32(R_(τ−1), I_τ) for τ >= 2          one FADD (`verity.ml.tc.fp32.add`)
    checked = I_1 … I_T and R_2 … R_T;  useful = R_T

R_1 is not a checked word: from R_0 = +0 the first add returns I_1 exactly, a duplicate in every output
(`c2FirstRunCopy`), so the honest schedule copies it for free.  A tile's digest is SHAKE256 over, in order, the forming
layer's checked words for the tile's rows and then its columns, and the core's words, 4 bytes little-endian each,
step-major (τ ascending), row-major over the cells within a step, I_τ before R_τ.  The honest H100 schedule prices an
atom word at 32 and an FADD at 32 W1 units (MH100 nominal), so the core credits 2·blocks·m·k·n − 32·m·n (6mkn − 32mn
for three blocks).

**The forming layer** (`Forming`) owns the input domain, the noise, the operands, any checked words of its own and its
W1 price per element.  Two instances:

- ``fpncp-v0`` (`FpNcpV0`): C-2's FP-NCP forming as posted, every FP32 operation fixed.  Refuted (X-D1-1, X-C2-B1):
  kept as the negative control that reproduces both attacks.
- ``d2-draft`` (`D2Draft`): track D's D-2 structure, operands formed by checked FP8 Hadamard-transform atoms over E4M3
  noise codes, with a placeholder noise-code distribution (`MaxRelativeCodes`).  D-2 fell in Phase 17b (X-D2-1 … 8);
  it stays as the worked example of a layer with checked forming words.

The main line is repaired C-2 (track D's D-3): a new `Forming` that drops in beside these, the core unchanged.  No
certificate names a γ_0: the core's premise (`DistinctLive` of the forming layer) is refuted for both layers here.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

from verity.ml.tc import fp16, fp32
from verity.ml.tc.cast import f32_to_e4m3_sat_word
from verity.ml.tc.total_fp8 import HOPPER_E4M3_WGMMA_K32
from verity.randomness import Part, frame

from ..protocol import Certificate, Context, DomainError, Matmul, Shape, Tile, grid_tiles

PREFIX = "fp8-is-h100"
ATOM = HOPPER_E4M3_WGMMA_K32
SLICE = 32
#: MH100 nominal W1 prices (``Pouw.Dimension.H100Assumptions.h100Nominal``): an E4M3 k32 accumulator word, an FADD
ATOM_UNITS, FADD_UNITS = 32, 32
#: MH100 nominal W1 price of one E4M3 code written by ``cvt.rn.satfinite.e4m3x2.f32`` (32 per x2 instruction; unmeasured)
CVT_UNITS = 16
#: H100 W1 prices by instruction class (units per instruction; `hadd2_f32` per FP32 result register): Round 7 as measured
#: (``gpu-constants.md`` §12) and Round 8 loop-free, the §0 rule's two sets.  ``unpack`` (cvt e4m3x2 -> f16x2) and
#: ``hadd2_f32`` and ``f16pack`` (F2FP f16x2 from f32) are not yet measured and carry the red team's assumptions.  Round 9
#: adds a row when it lands.
PRICES: dict[str, dict[str, Fraction]] = {
    "round7": {"fadd": Fraction(67, 2), "half2": Fraction(401, 10), "int": Fraction(64), "f2fp": Fraction(667, 10),
               "unpack": Fraction(667, 10), "hadd2_f32": Fraction(401, 10), "f16pack": Fraction(667, 10)},
    "round8": {"fadd": Fraction(32), "half2": Fraction(36), "int": Fraction(64), "f2fp": Fraction(64),
               "unpack": Fraction(64), "hadd2_f32": Fraction(36), "f16pack": Fraction(64)},
}


# ---- the core -------------------------------------------------------------------------------------------------------

def e4m3_row(row: bytes, k: int) -> list[int] | None:
    """A committed row: k E4M3 codes, one byte each; NaN codes (0x7F, 0xFF) and a wrong length are malformed."""
    if len(row) != k or any(c & 0x7F == 0x7F for c in row):
        return None
    return list(row)


def cell_words(x: Sequence[int], y: Sequence[int], seed: int | None = None) -> tuple[list[int], list[int | None]]:
    """One output cell's words over the stacked depth: ``([I_1 … I_T], [R_1 … R_T])``.  Unseeded, R_0 = +0 and R_1 = I_1
    is not a checked word (``None``); seeded, R_0 is the salt word ``seed`` and R_1 = add(R_0, I_1) is checked."""
    if not x or len(x) != len(y) or len(x) % SLICE:
        raise DomainError(f"the stacked depth is a positive multiple of {SLICE}")
    atoms = [ATOM.step(0, x[s:s + SLICE], y[s:s + SLICE]) for s in range(0, len(x), SLICE)]
    r = atoms[0] if seed is None else fp32.add(seed, atoms[0])
    running: list[int | None] = [None if seed is None else r]
    for i in atoms[1:]:
        r = fp32.add(r, i)
        running.append(r)
    return atoms, running


def live_products(x: Sequence[int], y: Sequence[int]) -> int:
    """How many of one slice's 32 products survive the atom's 14-bit alignment (contribute a nonzero aligned term)."""
    terms = [ATOM.product(a, b) for a, b in zip(x, y, strict=True)]
    top = max((t.exponent for t in terms if t.magnitude), default=None)
    if top is None:
        return 0
    shift = 24 - ATOM.width
    return sum(1 for t in terms if t.magnitude and (t.magnitude >> shift) >> (top - t.exponent))


def f32_float(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word))[0]


@dataclass(frozen=True, slots=True)
class Stack:
    """A tile's stacked operands (``x[i]`` row i of X̃, ``y[j]`` column j of Ỹ, ``blocks * k`` E4M3 codes each), the
    forming layer's checked words for each of those rows and columns, serialized, and, for a seeded layer, each cell's
    R_0 salt word."""

    x: list[list[int]]
    y: list[list[int]]
    trace_x: list[bytes]
    trace_y: list[bytes]
    seeds: list[list[int]] | None = None


@dataclass(frozen=True, slots=True)
class Checked:
    """A tile's checked values: the forming layer's words (serialized) and the core's ``cells[i][j] = ([I_τ], [R_τ])``
    (R_1 is ``None`` when unseeded), with the seeds they started from."""

    trace: bytes
    cells: list[list[tuple[list[int], list[int | None]]]]
    seeds: list[list[int]] | None = None


# ---- the forming layer ----------------------------------------------------------------------------------------------

@runtime_checkable
class Forming(Protocol):
    """What the core needs from a noise-and-forming layer."""

    #: The layer's name; the scheme is registered as ``fp8-is-h100-<name>``.
    name: str
    #: The checked function's id, which the digest tag carries; variants that differ only in price share it.
    tag: str
    #: The number of stacked blocks (K = blocks·k).
    blocks: int
    #: Whether `form` gives every cell a salt word R_0 for the running accumulator (then R_1 = add(R_0, I_1) is checked);
    #: an unseeded layer starts from +0, and R_1 = I_1 is dropped.
    seeded: bool
    #: W1 units per operand element of the honest forming trace, for an A row and for a B row.
    units: tuple[int | Fraction, int | Fraction]
    #: The part of ``units`` spent writing the layer's own checked words, which a lower bound may credit.
    credited: tuple[int | Fraction, int | Fraction]
    #: The values the layer is instantiated at (``Certificate.parameters``).
    parameters: tuple[tuple[str, int], ...]

    def check(self, shape: Shape) -> None:
        """Raise `DomainError` when the layer cannot form this shape."""
        ...

    def parse(self, row: bytes, k: int, left: bool) -> list[int] | None:
        """One committed row (``left``: of A) in the layer's own format, or None when it is malformed."""
        ...

    def admits(self, codes: Sequence[int], left: bool) -> bool:
        """The input domain: whether one parsed row (``left``: of A) is formable."""
        ...

    def noise(self, ctx: Context) -> Any:
        """The matmul's shared hash-derived values (per-row noise is derived in `form`)."""
        ...

    def form(self, ctx: Context, noise: Any, a: list[list[int]], b: list[list[int]], tile: Tile) -> Stack:
        """The tile's stacked operands and forming words from its rows of A (``a[i]`` is row ``tile.rows[i]``) and of B."""
        ...


def xof(tag: str, n: int, *parts: Part) -> bytes:
    return hashlib.shake_256(tag.encode() + b"\0" + b"".join(frame(p) for p in parts)).digest(n)


def f32_word(x: Fraction | int | float) -> int:
    """The FP32 word of a value that is exactly representable (a constant of the spec)."""
    word = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    if Fraction(f32_float(word)) != Fraction(x):
        raise ValueError(f"{x} is not an FP32 value")
    return word


def e4m3_to_f32(code: int) -> int:
    """The FP32 word of an E4M3 code (exact: E4M3 is a subset of binary32)."""
    sign, e, m = code >> 7, (code >> 3) & 0xF, code & 7
    return fp32.round_to_f32(bool(sign), m, -9) if e == 0 else fp32.round_to_f32(bool(sign), 8 | m, e - 10)


def word_perm32(k: int, seed: bytes, g: int = 4) -> tuple[int, ...]:
    """A word-granular permutation of [0, k): a SHAKE256 Fisher–Yates over the k/g aligned g-entry words, the order inside a
    word kept, repaired so that no word lands in its own 32-slice.  Block 2 reads entry l's noise at π(l)."""
    w, wps = k // g, SLICE // g

    def stream(tag: bytes, count: int):
        buf = hashlib.shake_256(seed + tag).digest(4 * count)
        return iter(int.from_bytes(buf[4 * i:4 * i + 4], "little") for i in range(count))

    def uniform(ws, bound: int) -> int:
        limit = (1 << 32) - ((1 << 32) % bound)
        while True:
            r = next(ws)
            if r < limit:
                return r % bound

    ws, p = stream(b"", 16 * w), list(range(w))
    for i in range(w - 1, 0, -1):
        j = uniform(ws, i + 1)
        p[i], p[j] = p[j], p[i]
    ws = stream(b"/repair", 64 * w)
    try:
        for j in range(w):
            while p[j] // wps == j // wps:
                r = uniform(ws, w)
                if p[r] // wps != j // wps and p[j] // wps != r // wps:
                    p[j], p[r] = p[r], p[j]
    except StopIteration:
        raise DomainError(f"the permutation's repair exhausts its stream at k = {k}") from None
    return tuple(g * p[a] + b for a in range(w) for b in range(g))


def admissible32(pi: Sequence[int]) -> bool:
    """No slice meets its image, no slice's image is one slice, and π² ≠ I (the NCP rejection rule at d = 32)."""
    for t in range(len(pi) // SLICE):
        img = {pi[j] // SLICE for j in range(t * SLICE, (t + 1) * SLICE)}
        if t in img or len(img) == 1:
            return False
    return any(pi[pi[j]] != j for j in range(len(pi)))


@lru_cache(maxsize=16)
def ncp_perm(k: int, seed: bytes) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """π and π⁻¹ at k, refused unless admissible at d = 32."""
    pi = word_perm32(k, seed)
    if not admissible32(pi):
        raise DomainError(f"the permutation fails the d = 32 rejection rule at k = {k}")
    inv = [0] * k
    for l, p in enumerate(pi):
        inv[p] = l
    return pi, tuple(inv)


class CodeRows:
    """Row parsing for the layers whose rows are all E4M3 codes (the earlier layers)."""

    def parse(self, row: bytes, k: int, left: bool) -> list[int] | None:
        return e4m3_row(row, k)


# ---- fpncp-v0: C-2's forming as posted (refuted; the negative control) -----------------------------------------------

#: Per-element W1 trace of `FpNcpV0` on MH100 at nominal prices, one operand row, conversions at H_32's floor.
FPNCP_V0_TRACE = (
    ("unpack e4m3x2 -> f16x2 (1 per 2) and f16 -> f32 (HADD2.F32)", 16 + 32),
    ("row max |x| (FMNMX)", 32),
    ("row sum of squares (FFMA)", 32),
    ("noise word -> [1, 2) (LOP3)", 64),
    ("e1 = 2σ·t + c (FFMA)", 32),
    ("e3 = e1 + e2 (FADD; e2 is a free word move)", 32),
    ("three FFMA α·x ± e_b", 96),
    ("three cvt.rn.satfinite.e4m3x2.f32 with merge (1 per 2 each)", 3 * CVT_UNITS),
)


class FpNcpV0(CodeRows):
    """C-2's FP-NCP forming with its FP32 operations fixed.  Per operand row of k E4M3 codes:

        x_l   = the code's FP32 value;  inf = max |x_l|
        ss    = 32 lane sums p_t = fma(x_l, x_l, p_t) over l ≡ t (mod 32) ascending, then p_t = add(p_t, p_(t+h)) for
                h = 16, 8, 4, 2, 1 (a warp's shuffle-down reduction); ss = p_0
        rms   = sqrt(div(ss, k));  den = add(inf, mul(rms, 2δ))
        α     = div(448, den),  σ = mul(mul(α, δ), rms);  an all-zero row has α = 1, σ = 224
        t_l   = the FP32 word 0x3F800000 | (w_l & 0x7FFFFF), w_l the row's l-th 32-bit XOF word (t_l in [1, 2))
        e1_l  = fma(2σ, t_l, c) with c = mul(σ, −3)   (so e1_l ≈ σ·u_l, u_l uniform on [−1, 1))
        A row:  e2_l = e1_(π(l)),        e3_l = add(e1_l, e2_l)
        B row:  f2_(π(l)) = −f1_l,       f3_l = add(f1_l, f2_l)
        X̃ = [Q(fma(α, x, e1)) | Q(fma(α, x, e2)) | Q(fma(−α, x, −e3))],  Ỹ = [Q(fma(β, w, f1)); Q(fma(β, w, f2)); Q(fma(β, w, f3))]

    with Q = cvt.rn.satfinite.e4m3.f32 and δ = 2^−delta_log2.  E₁'s row i is SHAKE256("verity/pouw/fp8-is/E1/v0", salt,
    index, root_A, weight, i), F₁'s row j SHAKE256("verity/pouw/fp8-is/F1/v0", salt, weight, j), 4k bytes each.  In exact
    arithmetic X̃·Ỹ = α_i β_j·A·Bᵀ (NCP's cancellation holds for any E₁, F₁); the roundings make it approximate.
    """

    name = tag = "fpncp-v0"
    blocks = 3
    seeded = False
    units = (sum(u for _, u in FPNCP_V0_TRACE),) * 2
    credited = (0, 0)
    P_SEED = b"verity/pouw/fp8-is/P/v0"

    def __init__(self, delta_log2: int = 3):
        self.delta_log2 = delta_log2
        self.delta = f32_word(Fraction(1, 1 << delta_log2))
        self.two_delta = f32_word(Fraction(2, 1 << delta_log2))
        self.parameters = (("delta_log2", delta_log2), ("form_units", self.units[0]))

    def check(self, shape: Shape) -> None:
        if shape.k % SLICE or shape.k > 1 << 16:
            raise DomainError(f"{self.name}: k is a multiple of {SLICE} and at most 2^16")
        ncp_perm(shape.k, self.P_SEED)

    def admits(self, codes: Sequence[int], left: bool) -> bool:
        return True

    def noise(self, ctx: Context) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return ncp_perm(ctx.matmul.shape.k, self.P_SEED)

    def scale(self, x: Sequence[int]) -> tuple[int, int]:
        """The row's (α, σ) words."""
        inf = max(w & 0x7FFF_FFFF for w in x)          # non-negative finite words order as their values
        p = [0] * SLICE
        for l, w in enumerate(x):
            p[l % SLICE] = fp32.fma(w, w, p[l % SLICE])
        h = SLICE // 2
        while h:
            for t in range(h):
                p[t] = fp32.add(p[t], p[t + h])
            h //= 2
        rms = fp32.sqrt(fp32.div(p[0], f32_word(len(x))))
        den = fp32.add(inf, fp32.mul(rms, self.two_delta))
        if den == 0:
            return f32_word(1), f32_word(224)
        alpha = fp32.div(f32_word(448), den)
        return alpha, fp32.mul(fp32.mul(alpha, self.delta), rms)

    @staticmethod
    def raw_noise(sigma: int, stream: bytes) -> list[int]:
        """e1 (or f1) for one row from its XOF bytes."""
        two_sigma, c = fp32.mul(sigma, f32_word(2)), fp32.mul(sigma, f32_word(-3))
        words = [int.from_bytes(stream[4 * l:4 * l + 4], "little") for l in range(len(stream) // 4)]
        return [fp32.fma(two_sigma, 0x3F80_0000 | (w & 0x7F_FFFF), c) for w in words]

    def side(self, codes: Sequence[int], stream: bytes, pi: Sequence[int], inv: Sequence[int], left: bool) -> list[int]:
        """One stacked row of X̃ (``left``) or column of Ỹ: 3k codes."""
        k = len(codes)
        x = [e4m3_to_f32(c) for c in codes]
        alpha, sigma = self.scale(x)
        e1 = self.raw_noise(sigma, stream)
        e2 = [e1[pi[l]] for l in range(k)] if left else [fp32.neg(e1[inv[l]]) for l in range(k)]
        e3 = [fp32.add(p, q) for p, q in zip(e1, e2)]
        q = f32_to_e4m3_sat_word
        out = [q(fp32.fma(alpha, v, e)) for v, e in zip(x, e1)] + [q(fp32.fma(alpha, v, e)) for v, e in zip(x, e2)]
        if left:
            return out + [q(fp32.fma(fp32.neg(alpha), v, fp32.neg(e))) for v, e in zip(x, e3)]
        return out + [q(fp32.fma(alpha, v, e)) for v, e in zip(x, e3)]

    def form(self, ctx: Context, noise, a: list[list[int]], b: list[list[int]], tile: Tile) -> Stack:
        pi, inv = noise
        u, k = ctx.matmul, ctx.matmul.shape.k
        x = [self.side(r, xof("verity/pouw/fp8-is/E1/v0", 4 * k, ctx.salt, u.index, ctx.root_a, u.weight, i), pi, inv, True)
             for r, i in zip(a, tile.rows)]
        y = [self.side(r, xof("verity/pouw/fp8-is/F1/v0", 4 * k, ctx.salt, u.weight, j), pi, inv, False)
             for r, j in zip(b, tile.cols)]
        return Stack(x, y, [b""] * len(x), [b""] * len(y))


# ---- d2-draft: D-2's checked transform forming ----------------------------------------------------------------------

def hadamard_negative(l: int, c: int) -> bool:
    """Whether the Sylvester Hadamard H_32 has −1 at (l, c); H is symmetric and H·Hᵀ = 32·I."""
    return (l & c).bit_count() & 1 == 1


E4M3_EIGHTH, E4M3_QUARTER = 0x20, 0x28      # 1/8 and 1/4 as E4M3 codes (exact)


def transform_group(parts: Sequence[Sequence[int]], negated: Sequence[bool], scale_code: int) -> tuple[list[int], list[int]]:
    """One 32-entry group's transform: for each output lane c, T_c = Σ_p Σ_l ±scale·H[l][c]·parts[p][l] as a chain of
    len(parts) k32 atoms from +0, each atom mixing every part in aligned 4-code words (round-robin over the parts), and the
    lane's code Q(T_c).  Returns (the atom words, lane-major then atom order; the 32 codes)."""
    lanes = [(p, 4 * w + q) for w in range(SLICE // 4) for p in range(len(parts)) for q in range(4)]
    words, codes = [], []
    for c in range(SLICE):
        acc = 0
        for a in range(len(parts)):
            chunk = lanes[SLICE * a:SLICE * (a + 1)]
            xs = [parts[p][l] for p, l in chunk]
            ws = [scale_code | (0x80 if hadamard_negative(l, c) != negated[p] else 0) for p, l in chunk]
            acc = ATOM.step(acc, xs, ws)
            words.append(acc)
        codes.append(f32_to_e4m3_sat_word(acc))
    return words, codes


class MaxRelativeCodes:
    """A placeholder noise-code distribution for D-2 (track D's repair replaces it): one XOF byte b per entry gives the
    E4M3 code ``(b & 0x8F) + (e0 << 3)``, that is sign b₇, exponent field e0 + b₃, mantissa b₂..b₀, where
    ``e0 = clamp(top − d − 1, 1, 13)`` and ``top`` is the exponent field of the row's largest code.  So every noise
    magnitude lies in [2^−(d+1), 2^−(d−1)) of the row's binade, never NaN, and one LOP3 plus one IADD form four codes."""

    name = "maxrel"
    units = 64                  # row max (SWAR, 32) + LOP3 (16) + IADD (16) per code

    def __init__(self, d: int = 3):
        self.d = d

    def codes(self, row: Sequence[int], stream: bytes) -> list[int]:
        top = max(c & 0x7F for c in row) >> 3
        e0 = min(max(top - self.d - 1, 1), 13)
        return [((b & 0x8F) + (e0 << 3)) & 0xFF for b in stream]


class D2Draft(CodeRows):
    """D-2 (track D; fell in Phase 17b, kept as the worked example of checked forming words): every NCP block's operand is formed by checked FP8 Hadamard transforms, with the noise entering the
    atoms as E4M3 codes, so no elementwise A + E is ever computed.  Per operand row and 32-entry group g:

        left  (A row, scale α = 1/8):   U_1 = Q(α[A|E1]H),  U_2 = Q(α[A|E2]H),  U_3 = Q(−α[A|E1|E2]H)      E2_l = E1_(π(l))
        right (B row, scale β = 1/4):   V_1 = Q(β[B|F1]H),  V_2 = Q(β[B|−G]H),  V_3 = Q(β[B|F1|−G]H)       G_l = F1_(π⁻¹(l))

    each by `transform_group` (2, 2 and 3 atoms per lane; a negated part negates its transform weights, so no code is
    negated).  X̃ = [U_1 | U_2 | U_3] and Ỹ = [V_1; V_2; V_3], group-major inside each block, so each main slice is one
    transformed group.  In exact arithmetic αβ·H·Hᵀ = I and NCP's cancellation give X̃·Ỹ = A·Bᵀ.

    Checked forming words, per row: every transform atom word and every converted code (7 words and 3 codes per element),
    credited at 7·32 + 3·16 = 272 units per element under H_32.  The noise codes (E1 per unit row, F1 per epoch weight row)
    come from `MaxRelativeCodes` over SHAKE256("verity/pouw/fp8-is/d2/E1", salt, index, root_A, weight, i) and
    ("…/F1", salt, weight, j), k bytes each.  The input domain keeps every transform inside E4M3's range: |A| <= a_max,
    |B| <= b_max (64 and 32 by default: with this sampler's noise below a quarter of the row's binade, block 3's two noise
    parts give |U| <= 4·1.5·64 = 384 and |V| <= 8·1.5·32 = 384, below 448; a new sampler re-derives the range).
    """

    name = tag = "d2-draft"
    blocks = 3
    seeded = False
    P_SEED = b"verity/pouw/fp8-is/d2/P"

    def __init__(self, sampler: MaxRelativeCodes | None = None, a_max: int = 0x68, b_max: int = 0x60):
        self.sampler = sampler if sampler is not None else MaxRelativeCodes()
        self.a_max, self.b_max = a_max, b_max
        credit = 7 * ATOM_UNITS + 3 * CVT_UNITS
        self.credited = (credit, credit)
        self.units = (credit + self.sampler.units,) * 2
        self.parameters = (("a_max_code", a_max), ("b_max_code", b_max), ("noise_d", self.sampler.d),
                           ("form_units", self.units[0]), ("form_credited", credit))

    def check(self, shape: Shape) -> None:
        if shape.k % SLICE or shape.k > 1 << 16:
            raise DomainError(f"{self.name}: k is a multiple of {SLICE} and at most 2^16")
        ncp_perm(shape.k, self.P_SEED)

    def admits(self, codes: Sequence[int], left: bool) -> bool:
        cap = self.a_max if left else self.b_max
        return all(c & 0x7F <= cap for c in codes)

    def noise(self, ctx: Context) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return ncp_perm(ctx.matmul.shape.k, self.P_SEED)

    def parts(self, codes: Sequence[int], stream: bytes, pi: Sequence[int], inv: Sequence[int], left: bool):
        """Each block's transform inputs for one row: ``[(parts, negated)]``, parts being full-length code rows."""
        e1 = self.sampler.codes(codes, stream)
        e2 = [e1[pi[l]] if left else e1[inv[l]] for l in range(len(codes))]
        if left:
            return [((codes, e1), (False, False)), ((codes, e2), (False, False)), ((codes, e1, e2), (True, True, True))]
        return [((codes, e1), (False, False)), ((codes, e2), (False, True)), ((codes, e1, e2), (False, False, True))]

    def side(self, codes: Sequence[int], stream: bytes, pi: Sequence[int], inv: Sequence[int], left: bool) -> tuple[list[int], bytes]:
        """One stacked operand row (3k codes) and its serialized forming words."""
        k = len(codes)
        scale = E4M3_EIGHTH if left else E4M3_QUARTER
        out, trace = [], []
        for parts, negated in self.parts(codes, stream, pi, inv, left):
            for g in range(0, k, SLICE):
                words, lane_codes = transform_group([p[g:g + SLICE] for p in parts], negated, scale)
                out += lane_codes
                trace.append(struct.pack(f"<{len(words)}I", *words) + bytes(lane_codes))
        return out, b"".join(trace)

    def form(self, ctx: Context, noise, a: list[list[int]], b: list[list[int]], tile: Tile) -> Stack:
        pi, inv = noise
        u, k = ctx.matmul, ctx.matmul.shape.k
        xs = [self.side(r, xof("verity/pouw/fp8-is/d2/E1", k, ctx.salt, u.index, ctx.root_a, u.weight, i), pi, inv, True)
              for r, i in zip(a, tile.rows)]
        ys = [self.side(r, xof("verity/pouw/fp8-is/d2/F1", k, ctx.salt, u.weight, j), pi, inv, False)
              for r, j in zip(b, tile.cols)]
        return Stack([o for o, _ in xs], [o for o, _ in ys], [t for _, t in xs], [t for _, t in ys])


# ---- d3-v0: repaired C-2 (track D's D-3), the main line ----------------------------------------------------------------

F16_ONE, F16_TWO_INV4, F16_MINUS_3_8 = 0x3C00, 0x3400, 0xB600      # 1, 1/4, −3/8
F16_2_POW_M6 = 0x2400                                               # 2^−6: eight times E4M3's subnormal step

#: Per-element W1 trace of `D3` on MH100 (one operand row), nominal and at Round 7's measured prices
#: (half2 40.1 and INT 64 per instruction, F2FP 66.7 per x2 instruction, FADD 33.5).
D3_TRACE = (
    # (operation, nominal units, measured units)
    ("unpack e4m3x2 -> f16x2 (1 per 2 codes)", 16, Fraction(667, 20)),
    ("row max |h| (HMNMX2)", 16, Fraction(401, 20)),
    ("z = h·2^-p (HMUL2)", 16, Fraction(401, 20)),
    ("sum of z² (HFMA2; a widen and an FADD per 16 per lane)", 20, Fraction(401, 20) + Fraction(401 + 335, 160)),
    ("y = α·h (HMUL2, α a power of two)", 16, Fraction(401, 20)),
    ("exponent mask of y (LOP3 on half2)", 32, 32),
    ("clamp at the subnormal step (HMNMX2)", 16, Fraction(401, 20)),
    ("partner max (HMNMX2)", 16, Fraction(401, 20)),
    ("two 10-bit noise fields (one LOP3)", 64, 64),
    ("σ·u, v/8 and s·v + σ·u (three HFMA2)", 48, Fraction(1203, 20)),
    ("e3 = e1 + e2 (HADD2; e2 a free word move)", 16, Fraction(401, 20)),
    ("three y ± e_b (HADD2)", 48, Fraction(1203, 20)),
    ("three cvt.rn.satfinite.e4m3x2.f16x2 (1 per 2 codes each)", 48, Fraction(2001, 20)),
)


class D3(CodeRows):
    """Repaired C-2 (track D's D-3): C-2's three NCP blocks with **partner-max full-cell noise** and **column-keyed weight
    streams**, formed in binary16 (`verity.ml.tc.fp16`, the half2 pipe).  Per operand row of k E4M3 codes:

        h_l   = the code's f16 value (exact);  inf = max |h_l|;  p = ⌊log₂ inf⌋
        ss    = Σ (h_l·2^−p)² in 64 lanes (l mod 64): each lane chains fma.f16 over 16 of its entries, widens the chain to
                f32 and adds it; then an f32 butterfly over the lanes (h = 32 … 1)
        rms   = sqrt(ss / k)·2^p (f32);  den = (9/8)·inf + 2δ·rms (f32)
        α     = the largest power of two with α·den <= 448, at most 2^15;  σ = α·δ·rms, rounded to f16
        y_l   = α·h_l (f16, exact)
        st_l  = max(2^⌊log₂|y_l|⌋, 2^−6), 0 ↦ 2^−6: eight E4M3 cells at |y_l| (one subnormal step below the normal range)
        s_l   = max(st_l, st_(partner l)): A rows partner l with π⁻¹(l), where block 2 reads e1_l; B rows with π(l)
        t, t' = f16 in [1, 2) from the row's XOF: 0x3C00 | (w & 0x3FF), for the low and the high 16 bits of word l
        e1_l  = fma(s_l, (2t'_l − 3)/8, σ·u_l),  σ·u_l = fma(2σ, t_l, RN(−3σ))       (u, v uniform on [−1, 1))
        A:  e2_l = e1_(π(l)),  e3 = e1 + e2;   X̃ = [Q(y + e1) | Q(y + e2) | Q(−y − e3)]
        B:  f2_l = −f1_(π⁻¹(l)), f3 = f1 + f2;  Ỹ = [Q(y + f1); Q(y + f2); Q(y + f3)]

    with Q = cvt.rn.satfinite.e4m3x2.f16x2 and δ = 2^−delta_log2.  So every entry carries at least one full local E4M3
    cell of noise amplitude in both blocks that read it (the X-C2-B1 repair), and NCP's cancellation is unchanged (it
    holds for any E1, F1).  E₁'s row i is SHAKE256("verity/pouw/fp8-is/d3/E1", salt, index, root_A, weight, i); F₁'s
    row j is SHAKE256("verity/pouw/fp8-is/d3/F1", salt, weight, j), one stream per weight column (X-D2-5), 4k bytes.
    The core runs unseeded: R_1 = I_1 is dropped.  An all-zero row has α = 1 and σ = 224.
    """

    name = tag = "d3-v0"
    blocks = 3
    seeded = False
    credited = (0, 0)
    P_SEED = b"verity/pouw/fp8-is/d3/P"
    LANES, CHAIN = 64, 16

    def __init__(self, delta_log2: int = 3, form_units: int | None = None):
        self.delta_log2 = delta_log2
        self.delta = f32_word(Fraction(1, 1 << delta_log2))
        self.two_delta = f32_word(Fraction(2, 1 << delta_log2))
        self.trace_units = sum(u for _, u, _ in D3_TRACE)
        self.measured_units = sum(m for _, _, m in D3_TRACE)
        f = self.trace_units if form_units is None else form_units
        self.units = (f, f)
        self.parameters = (("delta_log2", delta_log2), ("form_units", f), ("trace_units", self.trace_units))

    def check(self, shape: Shape) -> None:
        if shape.k % SLICE or shape.k > 1 << 16:
            raise DomainError(f"{self.name}: k is a multiple of {SLICE} and at most 2^16")
        ncp_perm(shape.k, self.P_SEED)

    def admits(self, codes: Sequence[int], left: bool) -> bool:
        return True

    def noise(self, ctx: Context) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return ncp_perm(ctx.matmul.shape.k, self.P_SEED)

    def scale(self, h: Sequence[int]) -> tuple[int, int]:
        """The row's (α, σ) as f16 words."""
        inf = max(w & 0x7FFF for w in h)                 # non-negative finite f16 words order as their values
        if inf == 0:
            return F16_ONE, fp16.round_to_f16(False, 224, 0)
        p = ((inf >> 10) & 0x1F) - 15
        down = fp16.round_to_f16(False, 1, -p)
        z = [fp16.mul(w, down) for w in h]
        lanes = [0] * self.LANES
        for t in range(self.LANES):
            entries = z[t::self.LANES]
            for g in range(0, len(entries), self.CHAIN):
                q = 0
                for w in entries[g:g + self.CHAIN]:
                    q = fp16.fma(w, w, q)
                lanes[t] = fp32.add(lanes[t], fp16.to_f32(q))
        half = self.LANES // 2
        while half:
            for t in range(half):
                lanes[t] = fp32.add(lanes[t], lanes[t + half])
            half //= 2
        rms = fp32.mul(fp32.sqrt(fp32.div(lanes[0], f32_word(len(h)))), fp32.round_to_f32(False, 1, p))
        den = fp32.add(fp32.mul(fp16.to_f32(inf), f32_word(Fraction(9, 8))), fp32.mul(rms, self.two_delta))
        _, dm, de = fp32.decode(den)
        top = dm.bit_length() - 1 + de                   # den = d·2^top, 1 <= d < 2
        j = min((8 if Fraction(dm, 1 << (dm.bit_length() - 1)) <= Fraction(7, 4) else 7) - top, 15)
        alpha32 = fp32.round_to_f32(False, 1, j)
        sigma = fp32.mul(fp32.mul(alpha32, self.delta), rms)
        neg_s, sm, se = fp32.decode(sigma)
        return fp16.round_to_f16(False, 1, j), fp16.round_to_f16(neg_s, sm, se)

    def side(self, codes: Sequence[int], stream: bytes, pi: Sequence[int], inv: Sequence[int], left: bool) -> list[int]:
        """One stacked row of X̃ (``left``) or column of Ỹ: 3k codes."""
        k = len(codes)
        h = [fp16.from_e4m3(c) for c in codes]
        alpha, sigma = self.scale(h)
        y = [fp16.mul(alpha, w) for w in h]
        st = [fp16.max_(w & 0x7C00, F16_2_POW_M6) for w in y]
        partner = inv if left else pi
        s = [fp16.max_(st[l], st[partner[l]]) for l in range(k)]
        two_sigma = fp16.mul(sigma, 0x4000)
        c = fp16.mul(sigma, 0xC200)                      # RN(−3σ)
        e1 = []
        for l in range(k):
            w = int.from_bytes(stream[4 * l:4 * l + 4], "little")
            t, tp = F16_ONE | (w & 0x3FF), F16_ONE | ((w >> 16) & 0x3FF)
            su = fp16.fma(two_sigma, t, c)
            v8 = fp16.fma(F16_TWO_INV4, tp, F16_MINUS_3_8)
            e1.append(fp16.fma(s[l], v8, su))
        q = fp16.to_e4m3
        if left:
            e2 = [e1[pi[l]] for l in range(k)]
            e3 = [fp16.add(a, b) for a, b in zip(e1, e2)]
            return ([q(fp16.add(v, e)) for v, e in zip(y, e1)] + [q(fp16.add(v, e)) for v, e in zip(y, e2)]
                    + [q(fp16.add(fp16.neg(v), fp16.neg(e))) for v, e in zip(y, e3)])
        f2 = [fp16.neg(e1[inv[l]]) for l in range(k)]
        f3 = [fp16.add(a, b) for a, b in zip(e1, f2)]
        return [q(fp16.add(v, f)) for f_blk in (e1, f2, f3) for v, f in zip(y, f_blk)]

    def form(self, ctx: Context, noise, a: list[list[int]], b: list[list[int]], tile: Tile) -> Stack:
        pi, inv = noise
        u, k = ctx.matmul, ctx.matmul.shape.k
        x = [self.side(r, xof("verity/pouw/fp8-is/d3/E1", 4 * k, ctx.salt, u.index, ctx.root_a, u.weight, i), pi, inv, True)
             for r, i in zip(a, tile.rows)]
        y = [self.side(r, xof("verity/pouw/fp8-is/d3/F1", 4 * k, ctx.salt, u.weight, j), pi, inv, False)
             for r, j in zip(b, tile.cols)]
        return Stack(x, y, [b""] * len(x), [b""] * len(y))


# ---- the D-3 family on the committed activation input: d3s-v0 (the lead law) and d3m-v0 ------------------------------

F16_2_POW_M8 = 0x1C00                                               # 2^−8
F16_448 = 0x5F00                                                    # 448, the largest Y16 magnitude admitted
ROUTES = ("xq", "y16")
LAWS = ("d3s", "d3m")


def d3_trace(law: str, route: str, weight_copy: bool, prices: dict[str, Fraction],
             chain: int) -> tuple[tuple[tuple[str, Fraction], ...], tuple[tuple[str, Fraction], ...]]:
    """The honest per-element trace of each side, schedule M4 (the D-3m strike; Phase 17f for D-3s): (activation
    operations, weight operations per unit).  The weight side's statistics, cell widths, partner and floor maxima are
    functions of the static codes and the fixed P, registered offline."""
    h2, i, blk = prices["half2"] / 2, prices["int"] / 2, prices["hadd2_f32"]
    source = (("unpack X_q: cvt e4m3x2 -> f16x2", prices["unpack"] / 2) if route == "xq"
              else ("pack Y16 = f16(A/s): F2FP f16x2 from the model's f32 A/s", prices["f16pack"] / 2))
    noise = (("noise field (LOP3; two f16 lanes per XOF word)", i), ("centre (2t − 3)/8 (HFMA2)", h2),
             ("scale A·u (HMUL2)", h2), ("s = e + e∘P (HADD2)", h2), ("three block values (HADD2.F32)", 3 * blk),
             ("three casts f32 -> e4m3 (F2FP x2)", 3 * prices["f2fp"] / 2))
    act = (source,
           ("prescale h·2^-8 (HMUL2; 448² overflows f16)", h2),
           ("sum of squares (HFMA2; a widen and an FADD per chain)", h2 + (blk + prices["fadd"]) / chain),
           *((("cell width: exponent mask (LOP3 on half2)", i),) if law == "d3m" else ()),
           ("partner max (HMNMX2)", h2),
           ("floor max (HMNMX2)", h2),
           *noise)
    weight = ((() if weight_copy else (("decode the weight codes: cvt e4m3x2 -> f16x2", prices["unpack"] / 2),)) + noise)
    return act, weight


class D3Family:
    """C-2's three NCP blocks formed on the unit's committed activation input, with one f16 noise field per side and a
    per-entry amplitude that covers at least one E4M3 cell of the entry and of its NCP partner.  Parameters:

    - ``law``: **"d3s"** (Phase 17f, the lead) or "d3m" (the D-3m strike);
    - ``route``: **"xq"**, the model's FP8 codes X_q (the §0 input), or "y16", Y16 = f16(A/s) with the model's own row
      scale s (quality-safer; Daniel's ruling (i));
    - ``weight_copy``: Daniel's ruling (ii), whether a registered FP16 weight copy is free offline data (True) or the
      codes are decoded every unit (False).  Only the price depends on it;
    - ``prices``: a `PRICES` row.

    Per operand row of k values (an activation row, or a weight column's E4M3 codes):

        h_l    = the row's f16 values: X_q's codes decoded exactly, or Y16's words
        ss     = Σ (h_l·2^−8)² in 64 lanes (l mod 64): each lane chains fma.f16 over 256 entries, widens the chain to f32
                 and adds it; then an f32 butterfly (32 … 1);   rms = sqrt(ss/k)·2^8 (f32)
        d3s:   A_l = max(|h_l|, |h_partner(l)|, rms′),  rms′ = max(f16(rms), 2^−6)            (δ = 1/8)
        d3m:   A_l = max(c8_l, c8_partner(l), σ8),  c8 = 2^⌊log₂|h|⌋ (0 at 0),
               σ8 = max(f16(max(rms/2, 224/√k)), 2^−6)                                      (δ = 1/16)
        e_l    = A_l·(2t_l − 3)/8,  t_l = 0x3C00 | (w_l & 0x3FF), w_l the l-th 16-bit half of the row's XOF stream
        A:  e2_l = e_π(l),   s = e + e2;      X̃ = [Q(h + e) | Q(h + e2) | Q(−h − s)]      partner(l) = π⁻¹(l)
        B:  f2_l = −e_π⁻¹(l), s = e + f2;     Ỹ = [Q(h + e) ; Q(h + f2) ; Q(h + s)]      partner(l) = π(l)

    The block values h + e_b are HADD2.F32 (f16 operands, one f32 rounding) and Q is cvt.rn.satfinite.e4m3x2.f32, so a
    block value at or beyond ±464 saturates to ±448.  For normal h, |h|/8 lies in [cell(h), 2·cell(h)), and rms′ ≥ 2^−6
    gives every entry (the zeros of an all-zero row, and E4M3 subnormals) at least one subnormal cell 2^−9: D-3s's
    amplitude is never below D-3m's.  E₁'s row i is SHAKE256("verity/pouw/fp8-is/<tag>/E1", salt, index, root_A,
    weight, i); each weight column j has its own stream per unit, SHAKE256("verity/pouw/fp8-is/<tag>/F1", salt, index,
    weight, j), 2k bytes each: B̃ is re-formed for every unit (r = 1, X-D3-3).  Weights are E4M3 codes quantized
    offline per column.  A Y16 row is k little-endian f16 words, finite and at most 448 in magnitude.
    """

    blocks = 3
    seeded = False
    credited = (0, 0)
    LANES, CHAIN = 64, 256

    def __init__(self, law: str = "d3s", route: str = "xq", weight_copy: bool = True, prices: str = "round8"):
        if law not in LAWS or route not in ROUTES:
            raise ValueError(f"law is one of {LAWS} and route one of {ROUTES}")
        self.law, self.route, self.weight_copy, self.prices = law, route, weight_copy, prices
        self.tag = f"{law}-v0" + ("-y16" if route == "y16" else "")
        self.name = self.tag + ("" if weight_copy else "-decode")
        self.P_SEED = f"verity/pouw/fp8-is/{law}/P".encode()
        self.trace = d3_trace(law, route, weight_copy, PRICES[prices], self.CHAIN)
        self.units = tuple(sum(u for _, u in side) for side in self.trace)
        self.parameters = (("law_d3s", int(law == "d3s")), ("route_y16", int(route == "y16")),
                           ("weight_copy", int(weight_copy)), ("form_units_a_ceil", -(-self.units[0] // 1)),
                           ("form_units_b_ceil", -(-self.units[1] // 1)))

    def check(self, shape: Shape) -> None:
        if shape.k % SLICE or shape.k > 1 << 16:
            raise DomainError(f"{self.name}: k is a multiple of {SLICE} and at most 2^16")
        ncp_perm(shape.k, self.P_SEED)

    def parse(self, row: bytes, k: int, left: bool) -> list[int] | None:
        """A committed row as f16 words: X_q's (or the weights') E4M3 codes decoded, or Y16's words (finite, |y| <= 448)."""
        if left and self.route == "y16":
            if len(row) != 2 * k:
                return None
            words = [int.from_bytes(row[2 * l:2 * l + 2], "little") for l in range(k)]
            return words if all(w & 0x7FFF <= F16_448 for w in words) else None
        codes = e4m3_row(row, k)
        return None if codes is None else [fp16.from_e4m3(c) for c in codes]

    def admits(self, values: Sequence[int], left: bool) -> bool:
        return True

    def noise(self, ctx: Context) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return ncp_perm(ctx.matmul.shape.k, self.P_SEED)

    def rms(self, h: Sequence[int]) -> int:
        """The row's rms as an f32 word (the lane procedure of the class docstring)."""
        z = [fp16.mul(w, F16_2_POW_M8) for w in h]
        lanes = [0] * self.LANES
        for t in range(self.LANES):
            entries = z[t::self.LANES]
            for g in range(0, len(entries), self.CHAIN):
                q = 0
                for w in entries[g:g + self.CHAIN]:
                    q = fp16.fma(w, w, q)
                lanes[t] = fp32.add(lanes[t], fp16.to_f32(q))
        half = self.LANES // 2
        while half:
            for t in range(half):
                lanes[t] = fp32.add(lanes[t], lanes[t + half])
            half //= 2
        return fp32.mul(fp32.sqrt(fp32.div(lanes[0], f32_word(len(h)))), f32_word(256))

    def floor(self, h: Sequence[int]) -> int:
        """The row's floor term as an f16 word: rms′ (d3s) or σ8 (d3m)."""
        rms = self.rms(h)
        if self.law == "d3m":
            rms = max(fp32.mul(rms, f32_word(Fraction(1, 2))), fp32.div(f32_word(224), fp32.sqrt(f32_word(len(h)))))
        neg, m, e = fp32.decode(rms)                         # non-negative words order as their values
        return fp16.max_(fp16.round_to_f16(neg, m, e), F16_2_POW_M6)

    def amplitudes(self, h: Sequence[int], partner: Sequence[int]) -> list[int]:
        floor = self.floor(h)
        if self.law == "d3s":
            mag = [w & 0x7FFF for w in h]
        else:
            mag = [w & 0x7C00 for w in h]
        return [fp16.max_(fp16.max_(mag[l], mag[partner[l]]), floor) for l in range(len(h))]

    def side(self, h: Sequence[int], stream: bytes, pi: Sequence[int], inv: Sequence[int], left: bool) -> list[int]:
        """One stacked row of X̃ (``left``) or column of Ỹ from its f16 values: 3k codes."""
        k = len(h)
        amp = self.amplitudes(h, inv if left else pi)
        e = []
        for l in range(k):
            t = F16_ONE | (int.from_bytes(stream[2 * l:2 * l + 2], "little") & 0x3FF)
            e.append(fp16.mul(amp[l], fp16.fma(F16_TWO_INV4, t, F16_MINUS_3_8)))
        if left:
            e2 = [e[pi[l]] for l in range(k)]
            s = [fp16.add(a, b) for a, b in zip(e, e2)]
            blocks = ((h, e), (h, e2), ([fp16.neg(w) for w in h], [fp16.neg(w) for w in s]))
        else:
            f2 = [fp16.neg(e[inv[l]]) for l in range(k)]
            s = [fp16.add(a, b) for a, b in zip(e, f2)]
            blocks = ((h, e), (h, f2), (h, s))
        return [f32_to_e4m3_sat_word(fp32.add(fp16.to_f32(x), fp16.to_f32(n)))
                for base, nz in blocks for x, n in zip(base, nz)]

    def form(self, ctx: Context, noise, a: list[list[int]], b: list[list[int]], tile: Tile) -> Stack:
        pi, inv = noise
        u, k = ctx.matmul, ctx.matmul.shape.k
        x = [self.side(r, xof(f"verity/pouw/fp8-is/{self.tag}/E1", 2 * k, ctx.salt, u.index, ctx.root_a, u.weight, i),
                       pi, inv, True) for r, i in zip(a, tile.rows)]
        y = [self.side(r, xof(f"verity/pouw/fp8-is/{self.tag}/F1", 2 * k, ctx.salt, u.index, u.weight, j), pi, inv, False)
             for r, j in zip(b, tile.cols)]
        return Stack(x, y, [b""] * len(x), [b""] * len(y))


def D3s(weight_copy: bool = True, prices: str = "round8", route: str = "xq") -> D3Family:
    return D3Family("d3s", route, weight_copy, prices)


def D3m(weight_copy: bool = True, prices: str = "round8", route: str = "xq") -> D3Family:
    return D3Family("d3m", route, weight_copy, prices)


FORMINGS = {f"{law}-v0{'-y16' if route == 'y16' else ''}{'' if copy else '-decode'}":
            (lambda law=law, route=route, copy=copy: D3Family(law, route, copy))
            for law in LAWS for route in ROUTES for copy in (True, False)}
FORMINGS.update({D3.name: D3, FpNcpV0.name: FpNcpV0, D2Draft.name: D2Draft})


# ---- the scheme -----------------------------------------------------------------------------------------------------

class FP8IS:
    """Fresh-step split-K FP8 on the H100 over a forming layer.  The lead domain is 64 | m, 64 | n, 32 | k and
    8,192 <= m, k, n <= 2^16, on 64 x 64 tiles: one m64n64k32 atom per slice, since Round 7 measured the atom at 32 per
    word only at N >= 64 (m64n8 runs at 22% of peak).  Smaller bounds or tiles make a test instance.  No certificate names
    a γ_0 yet (``PROTOCOL.md``)."""

    def __init__(self, forming: Forming, mnk_min: int = 8192, tile: tuple[int, int] = (64, 64), prices: str = "round8"):
        self.forming, self.prices = forming, prices
        self.mnk_min, self.tile = mnk_min, tile
        self.name = f"{PREFIX}-{forming.name}"
        self.fadd = PRICES[prices]["fadd"]                                          # the honest running add, per word
        (f_a, f_b), (c_a, c_b) = forming.units, forming.credited
        slices = forming.blocks * mnk_min // SLICE                                   # T at k = N
        adds = slices if forming.seeded else slices - 1
        honest = ATOM_UNITS * slices + self.fadd * adds                              # per output cell
        credited = ATOM_UNITS * slices + FADD_UNITS * adds                           # 32 per checked word
        self.certificate = Certificate(
            conjecture="tt-fp8-h100-core",
            gamma0=None,
            omega=Fraction(honest + f_a + f_b) / (credited + c_a + c_b),
            claims=("tt-fp8-h100-core", "h32-mh100", "w1-cost-model", "random-oracle", "xof/shake-256", "beacon-unpredictability"),
            theorems=("Pouw.Proofs.theorem1", "Pouw.Proofs.workWeightedSampling", "Pouw.Proofs.endToEnd"),
            hypotheses=(
                "H_32 per class under additive W1: every MH100 writer class (mma.sync small-K, sparse wgmma, b1, "
                "conversions, memory-side reductions included) costs at least 32 per 32-bit word it writes "
                "(Round 8 loop-free: h32LoopFree)",
                "DistinctLive(1/400): for every legal input, all but 1/400 of each output's checked words are pairwise "
                "distinct on the salt set and none is free (a property of the forming layer; lifted with the hash as a "
                "random oracle)",
                "the named branching gap: data-dependent control flow is outside the proved bound (X-D3m-3: 13-17% of "
                "the credit sits in low-entropy spike slices when P is fixed)",
                "AdmitsRef: H_ref runs within wref on an H100",
            ),
            allowances=(
                f"forming layer {forming.name}: DistinctLive is refuted for fpncp-v0 (X-D1-1, X-C2-B1) and d2-draft "
                "(X-D2-4, X-D2-5), and claimed for d3m-v0 at r = 1 (the D-3m strike); γ_0 is withheld until it is "
                "certified and Round 9's prices land",
                f"prices: {prices} (PRICES); the X_q unpack and HADD2.F32 are unmeasured",
            ),
            parameters=(("mnk_min", mnk_min), ("tile_m", tile[0]), ("tile_n", tile[1]), ("blocks", forming.blocks),
                        ("seeded", int(forming.seeded)), *forming.parameters),
        )

    def check(self, matmul: Matmul) -> None:
        s, lo = matmul.shape, self.mnk_min
        if s.m % 64 or s.n % 64 or s.k % SLICE or not all(lo <= v <= 1 << 16 for v in (s.m, s.k, s.n)):
            raise DomainError(f"{self.name}: {s} is outside 64 | m, 64 | n, 32 | k, {lo} <= m, k, n <= 2^16")
        self.forming.check(s)

    def tiles(self, shape: Shape) -> list[Tile]:
        return grid_tiles(shape, *self.tile)

    def wref(self, shape: Shape) -> int:
        """The core (an atom word per cell per slice, and an FADD at ``prices``' rate per checked running word) plus the
        forming trace on both sides, rounded up.  The weight side is charged once per matmul (r = 1); reuse across the
        epoch's units is the spec's reuse condition, not this per-matmul figure."""
        f_a, f_b = self.forming.units
        slices = self.forming.blocks * shape.k // SLICE
        adds = slices if self.forming.seeded else slices - 1
        total = shape.m * shape.n * (ATOM_UNITS * slices + self.fadd * adds) + f_a * shape.m * shape.k + f_b * shape.n * shape.k
        return -(-Fraction(total) // 1)

    def noise(self, ctx: Context):
        return self.forming.noise(ctx)

    def encode(self, ctx: Context, noise, a_rows: Sequence[bytes], b_rows: Sequence[bytes], tile: Tile) -> Stack | None:
        k = ctx.matmul.shape.k
        a = [self.forming.parse(r, k, True) for r in a_rows]
        b = [self.forming.parse(r, k, False) for r in b_rows]
        if any(r is None or not self.forming.admits(r, True) for r in a):
            return None
        if any(r is None or not self.forming.admits(r, False) for r in b):
            return None
        ops = self.forming.form(ctx, noise, a, b, tile)
        if (ops.seeds is not None) != self.forming.seeded:
            raise DomainError(f"{self.forming.name}: a seeded layer gives every cell an R_0, an unseeded one none")
        return ops

    def admissible(self, ctx: Context, ops: Stack) -> bool:
        return True

    def checked(self, ctx: Context, ops: Stack) -> Checked:
        """The forming words of the tile's rows then columns, and each cell's core words (`cell_words`)."""
        seed = (lambda i, j: ops.seeds[i][j]) if ops.seeds is not None else (lambda i, j: None)
        cells = [[cell_words(x, y, seed(i, j)) for j, y in enumerate(ops.y)] for i, x in enumerate(ops.x)]
        return Checked(b"".join(ops.trace_x + ops.trace_y), cells, ops.seeds)

    def digest(self, ctx: Context, noise, checked: Checked) -> bytes:
        cells = [cell for row in checked.cells for cell in row]
        steps = len(cells[0][0]) if cells else 0
        words = b"".join(struct.pack("<I", atoms[t]) + (b"" if running[t] is None else struct.pack("<I", running[t]))
                         for t in range(steps) for atoms, running in cells)
        return hashlib.shake_256(f"verity/pouw/{PREFIX}-{self.forming.tag}/tile".encode() + b"\0" + checked.trace
                                 + words).digest(32)

    def useful(self, ctx: Context, noise, ops: Stack, checked: Checked) -> list[list[float]]:
        """R_T per cell as its FP32 value, less the seed (one add.rn.f32, glue) for a seeded layer."""
        def out(i: int, j: int, atoms: list[int], running: list[int | None]) -> float:
            last = running[-1] if running[-1] is not None else atoms[0]
            return f32_float(last if checked.seeds is None else fp32.add(last, fp32.neg(checked.seeds[i][j])))
        return [[out(i, j, *cell) for j, cell in enumerate(row)] for i, row in enumerate(checked.cells)]
