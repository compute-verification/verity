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
  noise codes, with the noise-code distribution a placeholder (`MaxRelativeCodes`) that track D's repair replaces.

Neither certificate names a γ_0: the core's premise (`DistinctLive` of the forming layer) is refuted for the first and
not established for the second.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

from verity.ml.tc import fp32
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


# ---- the core -------------------------------------------------------------------------------------------------------

def e4m3_row(row: bytes, k: int) -> list[int] | None:
    """A committed row: k E4M3 codes, one byte each; NaN codes (0x7F, 0xFF) and a wrong length are malformed."""
    if len(row) != k or any(c & 0x7F == 0x7F for c in row):
        return None
    return list(row)


def cell_words(x: Sequence[int], y: Sequence[int]) -> tuple[list[int], list[int]]:
    """One output cell's checked words over the stacked depth: ``([I_1 … I_T], [R_2 … R_T])``."""
    if not x or len(x) != len(y) or len(x) % SLICE:
        raise DomainError(f"the stacked depth is a positive multiple of {SLICE}")
    atoms = [ATOM.step(0, x[s:s + SLICE], y[s:s + SLICE]) for s in range(0, len(x), SLICE)]
    r, running = atoms[0], []
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
    """A tile's stacked operands (``x[i]`` row i of X̃, ``y[j]`` column j of Ỹ, ``blocks * k`` E4M3 codes each) and the
    forming layer's checked words for each of those rows and columns, serialized."""

    x: list[list[int]]
    y: list[list[int]]
    trace_x: list[bytes]
    trace_y: list[bytes]


@dataclass(frozen=True, slots=True)
class Checked:
    """A tile's checked values: the forming layer's words (serialized) and the core's ``cells[i][j] = ([I_τ], [R_τ, τ >= 2])``."""

    trace: bytes
    cells: list[list[tuple[list[int], list[int]]]]


# ---- the forming layer ----------------------------------------------------------------------------------------------

@runtime_checkable
class Forming(Protocol):
    """What the core needs from a noise-and-forming layer."""

    #: The layer's name; the scheme is registered as ``fp8-is-h100-<name>``, and the digest tag carries it.
    name: str
    #: The number of stacked blocks (K = blocks·k).
    blocks: int
    #: W1 units (MH100 nominal) per operand element of the honest forming trace, for an A row and for a B row.
    units: tuple[int, int]
    #: The part of ``units`` spent writing the layer's own checked words, which a lower bound may credit.
    credited: tuple[int, int]
    #: The values the layer is instantiated at (``Certificate.parameters``).
    parameters: tuple[tuple[str, int], ...]

    def check(self, shape: Shape) -> None:
        """Raise `DomainError` when the layer cannot form this shape."""
        ...

    def admits(self, codes: Sequence[int], left: bool) -> bool:
        """The input domain: whether one committed row (``left``: of A) is formable."""
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


class FpNcpV0:
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

    name = "fpncp-v0"
    blocks = 3
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


class D2Draft:
    """D-2 (track D): every NCP block's operand is formed by checked FP8 Hadamard transforms, with the noise entering the
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
    |B| <= b_max (64 and 32 by default: |U| <= 4·1.25·64 and |V| <= 8·1.25·32 stay below 448).
    """

    name = "d2-draft"
    blocks = 3
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


FORMINGS = {FpNcpV0.name: FpNcpV0, D2Draft.name: D2Draft}


# ---- the scheme -----------------------------------------------------------------------------------------------------

class FP8IS:
    """Fresh-step split-K FP8 on the H100 over a forming layer.  The lead domain is 64 | m, 8 | n, 32 | k and
    8,192 <= m, k, n <= 2^16, on 64 x 8 tiles (one physical m64n8 atom per slice); smaller bounds or tiles make a test
    instance.  No certificate names a γ_0 yet (``PROTOCOL.md``)."""

    def __init__(self, forming: Forming | None = None, mnk_min: int = 8192, tile: tuple[int, int] = (64, 8)):
        self.forming = forming if forming is not None else D2Draft()
        self.mnk_min, self.tile = mnk_min, tile
        self.name = f"{PREFIX}-{self.forming.name}"
        self.credit = self.forming.blocks * (ATOM_UNITS + FADD_UNITS) // SLICE     # core W1 units per useful MAC
        (f_a, f_b), (c_a, c_b) = self.forming.units, self.forming.credited
        core = self.credit * mnk_min - FADD_UNITS                                  # per N² at m = n = k = N
        self.certificate = Certificate(
            conjecture="tt-fp8-h100-core",
            gamma0=None,
            omega=Fraction(core + f_a + f_b, core + c_a + c_b),
            claims=("tt-fp8-h100-core", "h32-mh100", "w1-cost-model", "random-oracle", "xof/shake-256", "beacon-unpredictability"),
            theorems=("Pouw.Proofs.theorem1", "Pouw.Proofs.workWeightedSampling", "Pouw.Proofs.endToEnd"),
            hypotheses=(
                "H_32: every MH100 writer, mma.sync small-K shapes, sparse wgmma, b1 paths and memory-side reductions "
                "included, costs at least the honest atom per 32-bit word it writes (PROTOCOL.md)",
                "DistinctLive(γ_0): for every admitted input, all but a γ_0 fraction of the credited checked words are "
                "salt-dependent and pairwise distinct as functions of the salt (a property of the forming layer)",
                "A2_fp: data-dependent control flow, routing and grinding do not beat the per-word count",
                "AdmitsRef: H_ref runs within wref on an H100",
            ),
            allowances=(
                f"forming layer {self.forming.name}: DistinctLive is refuted (fpncp-v0: X-D1-1, X-C2-B1) or not established "
                "(d2-draft: placeholder noise codes); γ_0 is withheld until a repaired layer lands",
                "forming prices are MH100 nominal; the E4M3 conversion and SWAR rates are unmeasured",
            ),
            parameters=(("mnk_min", mnk_min), ("tile_m", tile[0]), ("tile_n", tile[1]), ("blocks", self.forming.blocks),
                        *self.forming.parameters),
        )

    def check(self, matmul: Matmul) -> None:
        s, lo = matmul.shape, self.mnk_min
        if s.m % 64 or s.n % 8 or s.k % SLICE or not all(lo <= v <= 1 << 16 for v in (s.m, s.k, s.n)):
            raise DomainError(f"{self.name}: {s} is outside 64 | m, 8 | n, 32 | k, {lo} <= m, k, n <= 2^16")
        self.forming.check(s)

    def tiles(self, shape: Shape) -> list[Tile]:
        return grid_tiles(shape, *self.tile)

    def wref(self, shape: Shape) -> int:
        """The core (an atom word per cell per slice, and an FADD for every slice after the first) plus the forming trace
        on both sides.  The weight side is charged once per matmul; sharing it across matmuls is not credited."""
        f_a, f_b = self.forming.units
        core = self.credit * shape.wmm - FADD_UNITS * shape.m * shape.n
        return core + f_a * shape.m * shape.k + f_b * shape.n * shape.k

    def noise(self, ctx: Context):
        return self.forming.noise(ctx)

    def encode(self, ctx: Context, noise, a_rows: Sequence[bytes], b_rows: Sequence[bytes], tile: Tile) -> Stack | None:
        k = ctx.matmul.shape.k
        a, b = [e4m3_row(r, k) for r in a_rows], [e4m3_row(r, k) for r in b_rows]
        if any(r is None or not self.forming.admits(r, True) for r in a):
            return None
        if any(r is None or not self.forming.admits(r, False) for r in b):
            return None
        return self.forming.form(ctx, noise, a, b, tile)

    def admissible(self, ctx: Context, ops: Stack) -> bool:
        return True

    def checked(self, ctx: Context, ops: Stack) -> Checked:
        """The forming words of the tile's rows then columns, and each cell's core words (`cell_words`)."""
        return Checked(b"".join(ops.trace_x + ops.trace_y), [[cell_words(x, y) for y in ops.y] for x in ops.x])

    def digest(self, ctx: Context, noise, checked: Checked) -> bytes:
        cells = [cell for row in checked.cells for cell in row]
        steps = len(cells[0][0]) if cells else 0
        words = b"".join(struct.pack("<I", atoms[t]) + (struct.pack("<I", running[t - 1]) if t else b"")
                         for t in range(steps) for atoms, running in cells)
        return hashlib.shake_256(f"verity/pouw/{self.name}/tile".encode() + b"\0" + checked.trace + words).digest(32)

    def useful(self, ctx: Context, noise, ops: Stack, checked: Checked) -> list[list[float]]:
        """R_T per cell as its FP32 value (≈ α_i·β_j·(A·Bᵀ)_ij for fpncp-v0, ≈ (A·Bᵀ)_ij for d2-draft)."""
        return [[f32_float(running[-1] if running else atoms[0]) for atoms, running in row] for row in checked.cells]
