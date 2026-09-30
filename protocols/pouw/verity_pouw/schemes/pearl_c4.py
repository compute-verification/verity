"""``pearl-c-nvfp4-v0`` and ``pearl-c-mxfp4-v0``: Pearl-C on sm_120's block-scaled FP4 (``mma.sync kind::mxf4nvf4``).

The design is research store ``internal/pouw/rtx-pro/fp4-design.md`` (GPU 5's lane).  It is Pearl-C (`pearl_c`) with the
FP8 product replaced by the E2M1 one: the noise enters inside one atom per element, one noisy cast forms the operand,
Pearl's tickets are drawn on C̃, the BF16 clean-up U is checked, and every C̃ and U word is hashed into the tile's leaf
(`pearl_c.leaf`).  γ rests on a conjecture the theory lane has not stated yet (TT_OUT for this Π); nothing here is proved.

**Per row x** of either operand (k BF16 words, the model's activation and weight dtype):

~~~text
stats     s = max|x|, ρ = pearl_c's every-8th RMS in its pinned order;  σ = ρ·δ (δ = 1/4);  t = s + 4σ
          nvfp4: α = 2688 / max(t, 2^-32)       mxfp4: α = that, floored to a power of two (its mantissa bits cleared)
          β_row = (σ·α)·f32(16/√2346.125);  a_E = ue4m3(β_row)                     (FP32 nearest even, in that order)
lines     32 uniform E2M1 codes per line: TurboSHAKE128(seed ‖ role ‖ index u32 LE, D = 0x24), 16 bytes, low nibble first.
          E_A (row i, seed_A), F_A and F_B (position t), E_B (row j) from seed_B
atom      D_t = NVF4.step(C = α·x_t, E ‖ 0^32 at scale a_E, F_t ‖ 0^32 at scale 2^-4)    (one k64 atom per element)
block     nvfp4, 16 along k: s_b = max(ue4m3(amax_b|D| · f32(1/6)), 2^-9), r_b = 1/s_b (FP32 nearest even)
          mxfp4, 32 along k: s_b = 2^⌈log2(amax_b|D| · f32(1/6))⌉ as UE8M0, r_b = 1/s_b (exact)
cast      code_t = e2m1(D_t · r_b)                                        (nearest even, saturating to ±6, sign kept)
~~~

2346.125 = 32·8.5625² is a line product's variance (a uniform E2M1 code has E[v²] = 8.5625), so the noise's standard
deviation is σ·α up to a_E's rounding; F's scale 2^-4 in the atom keeps a_E normal down to s/ρ ≈ 14,000.

**Per 64 x 64 tile** (rows I of A, rows J of B, full k):

~~~text
chain     C̃_ij = the atom (NVF4 or MXF4, the scheme's kind) chained from +0 over A′_i · B̃_j, k/64 steps, no promotion
fused     AF_ip = chain(A′_i, F_B[:, p]) at scale 1.0 (32 extra columns);  BF_jq = chain(B̃_j, F_A[:, q]) and
          G_pq = chain(F_B[:, p], F_A[:, q]) once per job
peel      Ẽ = bf16(E · a_E · 2^-4) (exact);  P_A,i = [Ẽ_A,i ‖ RNE_bf16(AF_i)],
          P_B,j = [RNE_bf16(FP32 add(Σ_p fma(Ẽ_B,jp, G_pq), −BF_jq)) ‖ −Ẽ_B,j];
          U_ij = C̃_ij, then four HOPPER_BF16_M16N8K16 steps (sm_120 mma.sync m16n8k16) over P_A,i · P_B,j
leaf      pearl_c.leaf over C̃ ‖ U;  tickets: Pearl's lottery over C̃ (pearl_c.PearlC.tickets)
useful    U_ij / (α_A,i · α_B,j), the model's own unscale (glue)
~~~

The clean-up's algebra is `pearl_c`'s with N_A = Ẽ_A·F_Aᵀ: (A′ − N_A)(B̃ − N_B)ᵀ = C̃ + P_A·P_Bᵀ.  The clean path (PoUW
off) is `clean`: the same α and block rule on α·x, no noise.

**Why the chain's exactness matters.**  On the domain the chain is exact, so rounding does not tie a word to its order,
as Pearl-C's H100 argument had it.  The design rests instead on every code and every block scale depending on the salt.
A block whose noise is below τ = 1/8 of its code step is *dead*: its codes barely depend on the salt.  `tile_cap` debits
dead blocks, and the per-row rules bound them (``fp4-design.md`` §6, §7; δ, τ and the bounds are placeholders for the
theory lane).

**Stand-ins.**  The atoms are the RTX 5090 pins (`BLACKWELL_SM120_NVF4`, `_MXF4`, `HOPPER_BF16_M16N8K16`), pending GPU 4's
RTX PRO recheck.  The E2M1, UE4M3 and UE8M0 casts are modelled from the PTX ISA, not captured.  A negative value that
rounds to zero keeps its sign (code 0x8); no checked word depends on it.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from verity.commitments.blake3 import blake3, keyed
from verity.commitments.turboshake import turboshake128
from verity.ml.tc import fp32
from verity.ml.tc.models import BLACKWELL_SM120_MXF4, BLACKWELL_SM120_NVF4, HOPPER_BF16_M16N8K16, BlockScaledAlignAdd

from ..protocol import Certificate, Context, DomainError, Matmul, Shape, Tile, grid_tiles
from . import pearl_c as C
from . import pearl_kw as P

R = 32
ATOM = 64
ALIGN_K = 128
PEEL_K = 2 * R
TILE = C.TILE
D_LINE = 0x24                      # TurboSHAKE128 domain byte of the noise lines
ROLE_EA, ROLE_FA, ROLE_EB, ROLE_FB = 0, 1, 2, 3
LABELS = {r: f"pearl-c4/v0/{r}".encode() for r in ("seed-A", "seed-B")}

DELTA = 0x3E800000                 # δ = 1/4
FOUR = 0x40800000
TOP = 0x45280000                   # 2688 = 6 · 448, NVFP4's largest value at unit FP32 scale
FLOOR = 0x2F800000                 # 2^-32
K_BETA = 0x3EA920A7                # f32(16 / √2346.125)
SIXTH = 0x3E2AAAAB                 # f32(1/6)
ONE = 0x3F800000
NOISE_F_SCALE = 0x18               # UE4M3 2^-4: F's scale inside the noise atom
UE4M3_ONE, UE8M0_ONE = 0x38, 127
UE4M3_MIN = 0x01                   # 2^-9
UE4M3_NORMAL = 0x08                # 2^-6
TAU_SQ_INV = 144                   # dead block: amax_b > 12ρ, i.e. δ·6/τ = 12 with δ = 1/4, τ = 1/8
DEAD_SHARE = 64                    # a row passes with at most one dead block in 64

#: W1 in FP4 units (1 = one dense FP4 MAC on sm_120).  Placeholder prices from public peak rates until GPU 4's
#: microbenchmarks: a BF16 MAC 4, an FP32 instruction 16.
BF16_MAC_UNITS, FP32_UNITS = 4, 16
#: Forming per operand element: the noise atom 64, the noisy block amax and scale about 19, the cast 32 (credited, they
#: read the salt); ρ, one FP32 FMA every 8th element, 2 (A alone, not credited).
FORMING_CREDITED, FORMING_A_ONLY = 115, 2
CAP = Fraction(1, 400)

_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_E2M1_HALVES = (0, 1, 2, 3, 4, 6, 8, 12)


@dataclass(frozen=True)
class Fmt:
    """One FP4 operand format: the chain's atom, the scale block along k, and the scale byte of 1.0."""

    name: str
    atom: BlockScaledAlignAdd
    block: int
    one: int


NVFP4 = Fmt("nvfp4", BLACKWELL_SM120_NVF4, 16, UE4M3_ONE)
MXFP4 = Fmt("mxfp4", BLACKWELL_SM120_MXF4, 32, UE8M0_ONE)
FORMATS = {f.name: f for f in (NVFP4, MXFP4)}
NAMES = {"nvfp4": "pearl-c-nvfp4-v0", "mxfp4": "pearl-c-mxfp4-v0"}


def _finite(word: int) -> int:
    if (word >> 23) & 0xFF == 0xFF:
        raise P.Reject("non-finite FP32 word")
    return word


# ---- casts --------------------------------------------------------------------------------------------------------

def e2m1(word: int) -> int:
    """cvt.rn.satfinite.e2m1 of one finite FP32 word: nearest even, ±6 past the top, the sign kept (−0 is 0x8)."""
    v = abs(P.f32_of_bits(_finite(word)))
    code = 0
    for i, mid in enumerate(_E2M1_MIDPOINTS):
        if v > mid or (v == mid and (i + 1) % 2 == 0):
            code = i + 1
        else:
            break
    return code | (8 if word >> 31 else 0)


def e2m1_value(code: int) -> Fraction:
    n = _E2M1_HALVES[code & 7]
    return Fraction(-n if code & 8 else n, 2)


def ue4m3(word: int) -> int:
    """The UE4M3 byte of a non-negative FP32 word: nearest even, saturating to 448 (Pearl's E4M3 cast on its sign-free half)."""
    return P.f32_to_fp8(P.f32_of_bits(word)) & 0x7F


def ue4m3_word(code: int) -> int:
    return P.f32_bits(P.fp8_to_f32(code))


def ue8m0_up(word: int) -> int:
    """The UE8M0 byte of 2^⌈log2 v⌉ for a non-negative FP32 word v (cvt.rp.satfinite.ue8m0); 0 for v = 0."""
    field, frac = (word >> 23) & 0xFF, word & 0x7FFFFF
    if field == 0 and frac == 0:
        return 0
    if field:
        e = field - 127 + (frac != 0)
    else:
        e = frac.bit_length() - 1 - 149 + (frac & (frac - 1) != 0)
    return min(max(e + 127, 0), 254)


def pow2_word(e: int) -> int:
    """2^e as an FP32 word, e in [−149, 127]."""
    return (e + 127) << 23 if e >= -126 else 1 << (e + 149)


def block_scale(fmt: Fmt, amax: int) -> tuple[int, int]:
    """(scale byte, FP32 reciprocal) of a block with |max| amax, by `fmt`'s rule (module docstring)."""
    v = fp32.mul(amax, SIXTH)
    if fmt is NVFP4:
        s = max(ue4m3(v), UE4M3_MIN)
        return s, fp32.div(ONE, ue4m3_word(s))
    s = ue8m0_up(v)
    return s, pow2_word(127 - s)


# ---- rows, statistics, lines ---------------------------------------------------------------------------------------

def parse_row(row: bytes, k: int) -> list[int]:
    """A row of k BF16 words, as FP32 words (each BF16 word shifted up 16 bits)."""
    if len(row) != 2 * k:
        raise P.Reject("a row has k BF16 words")
    words = [w << 16 for w in struct.unpack(f"<{k}H", row)]
    for w in words:
        _finite(w)
    return words


def bf16_row(words: Sequence[int]) -> bytes:
    """k BF16 words (given as BF16 codes) as a row."""
    return struct.pack(f"<{len(words)}H", *words)


def stats(fmt: Fmt, x: Sequence[int]) -> tuple[int, int, int, int, int]:
    """(s, ρ, α, β_row, a_E) of one row of FP32 words, as FP32 words and a_E's UE4M3 byte, in the pinned order."""
    s, rho = C.s5_stats(x)[:2]
    sigma = fp32.mul(rho, DELTA)
    t = _finite(fp32.add(s, fp32.mul(FOUR, sigma)))
    alpha = fp32.div(TOP, max(t, FLOOR))
    if fmt is MXFP4:
        alpha &= 0xFF800000
    beta = _finite(fp32.mul(fp32.mul(sigma, alpha), K_BETA))
    return s, rho, alpha, beta, ue4m3(beta)


def line(seed: bytes, role: int, index: int) -> list[int]:
    """One noise line: 32 E2M1 codes, uniform over the 16 nibbles."""
    raw = turboshake128(seed + bytes([role]) + index.to_bytes(4, "little"), D_LINE, R // 2)
    return [b >> s & 0xF for b in raw for s in (0, 4)]


@lru_cache(maxsize=16)
def basis(seed_b: bytes, role: int, k: int) -> tuple[tuple[int, ...], ...]:
    """F_A (ROLE_FA) or F_B (ROLE_FB): k lines of 32 codes, line t for position t."""
    return tuple(tuple(line(seed_b, role, t)) for t in range(k))


def column(f: Sequence[Sequence[int]], p: int) -> list[int]:
    return [ln[p] for ln in f]


# ---- forming ------------------------------------------------------------------------------------------------------

@dataclass(slots=True)
class Formed:
    """One side's formed rows: codes and scale bytes (A′ or B̃), α, β_row, a_E, and the input words."""

    codes: list[list[int]]
    scales: list[list[int]]
    alpha: list[int]
    beta: list[int]
    ae: list[int]
    x: list[list[int]]


def noise_atom(c_word: int, e_row: Sequence[int], ae: int, f_line: Sequence[int]) -> int:
    """D = NVF4.step(C, E ‖ 0^32 at a_E, F_t ‖ 0^32 at 2^-4)."""
    pad = [0] * (ATOM - R)
    return _finite(BLACKWELL_SM120_NVF4.step_scaled(c_word, list(e_row) + pad, list(f_line) + pad, [ae] * 4,
                                                    [NOISE_F_SCALE] * 4))


def quantize(fmt: Fmt, d: Sequence[int]) -> tuple[list[int], list[int]]:
    """Block scales and E2M1 codes of one row of FP32 words by `fmt`'s rule."""
    codes, scales = [], []
    for b in range(0, len(d), fmt.block):
        s, r = block_scale(fmt, max(w & 0x7FFFFFFF for w in d[b:b + fmt.block]))
        scales.append(s)
        codes.extend(e2m1(fp32.mul(w, r)) for w in d[b:b + fmt.block])
    return codes, scales


def form(fmt: Fmt, x_rows: Sequence[Sequence[int]], e_rows: Sequence[Sequence[int]], f: Sequence[Sequence[int]]) -> Formed:
    """The noisy forming: per element one noise atom with α·x as its accumulator, then the noisy block scale and cast."""
    out = Formed([], [], [], [], [], [])
    for x, e_row in zip(x_rows, e_rows, strict=True):
        _, _, alpha, beta, ae = stats(fmt, x)
        codes, scales = quantize(fmt, [noise_atom(fp32.mul(alpha, xt), e_row, ae, f[t]) for t, xt in enumerate(x)])
        for field, v in (("codes", codes), ("scales", scales), ("alpha", alpha), ("beta", beta), ("ae", ae),
                         ("x", list(x))):
            getattr(out, field).append(v)
    return out


def clean(fmt: Fmt, x: Sequence[int]) -> tuple[list[int], list[int], int]:
    """The adopted quantizer with PoUW off: (codes, scales, α) of α·x under the same block rule."""
    alpha = stats(fmt, x)[2]
    codes, scales = quantize(fmt, [fp32.mul(alpha, xt) for xt in x])
    return codes, scales, alpha


# ---- the chain and the clean-up --------------------------------------------------------------------------------

def chain(fmt: Fmt, a_codes: Sequence[int], b_codes: Sequence[int], a_scales: Sequence[int], b_scales: Sequence[int]) -> int:
    """The atom chained from +0 over the full k, no promotion."""
    if len(a_codes) % ATOM:
        raise P.Reject("the chain multiplies in atoms of k = 64")
    return _finite(fmt.atom.chain(0, a_codes, b_codes, a_scales, b_scales))


@lru_cache(maxsize=16)
def gram(fmt: Fmt, seed_b: bytes, k: int) -> tuple[tuple[int, ...], ...]:
    """G = F_Bᵀ·F_A (r x r) as FP32 words, G_pq = chain(F_B[:, p], F_A[:, q]) at scale 1.0; once per job."""
    fa, fb = basis(seed_b, ROLE_FA, k), basis(seed_b, ROLE_FB, k)
    ones = [fmt.one] * (k // fmt.block)
    fa_cols = [column(fa, q) for q in range(R)]
    return tuple(tuple(chain(fmt, column(fb, p), fa_cols[q], ones, ones) for q in range(R)) for p in range(R))


def e_tilde(e_row: Sequence[int], ae: int) -> list[int]:
    """Ẽ = bf16(E · a_E · 2^-4), exact: E2M1 times UE4M3 has at most 6 significant bits."""
    scale = Fraction(P.fp8_to_f32(ae)) / 16
    return [P.bf16_of_fraction(e2m1_value(e) * scale) for e in e_row]


def peel_factors_a(fmt: Fmt, side: Formed, e_rows: Sequence[Sequence[int]], seed_b: bytes, k: int) -> list[list[int]]:
    """P_A rows: [Ẽ_A ‖ RNE_bf16(A′·F_B)]."""
    fb = basis(seed_b, ROLE_FB, k)
    ones = [fmt.one] * (k // fmt.block)
    cols = [column(fb, p) for p in range(R)]
    return [e_tilde(e, ae) + [C._bf16_of_word(chain(fmt, codes, cols[p], scales, ones)) for p in range(R)]
            for codes, scales, e, ae in zip(side.codes, side.scales, e_rows, side.ae, strict=True)]


def peel_factors_b(fmt: Fmt, side: Formed, e_rows: Sequence[Sequence[int]], seed_b: bytes, k: int) -> list[list[int]]:
    """P_B rows: [RNE_bf16(Ẽ_B·G − B̃·F_A) ‖ −Ẽ_B], once per job.  Ẽ_B·G is the FP32 FMA chain from +0 in p order, and the
    difference one FP32 nearest-even add of the negated B̃·F_A."""
    fa = basis(seed_b, ROLE_FA, k)
    ones = [fmt.one] * (k // fmt.block)
    cols = [column(fa, q) for q in range(R)]
    g = gram(fmt, seed_b, k)
    rows = []
    for codes, scales, e, ae in zip(side.codes, side.scales, e_rows, side.ae, strict=True):
        et = e_tilde(e, ae)
        mid = []
        for q in range(R):
            acc = 0
            for p in range(R):
                acc = _finite(fp32.fma(et[p] << 16, g[p][q], acc))
            mid.append(C._bf16_of_word(_finite(fp32.add(acc, fp32.neg(chain(fmt, codes, cols[q], scales, ones))))))
        rows.append(mid + [w ^ 0x8000 for w in et])
    return rows


def peel(c_word: int, pa: Sequence[int], pb: Sequence[int]) -> int:
    """U from C̃: four HOPPER_BF16_M16N8K16 steps over the 64 BF16 peel operands, k16 slices in order."""
    acc = c_word
    for s in range(0, PEEL_K, 16):
        acc = _finite(HOPPER_BF16_M16N8K16.step(acc, pa[s:s + 16], pb[s:s + 16]))
    return acc


# ---- the per-row rules and the debit (placeholders for the theory lane) -------------------------------------------

def _exact_rho_sq(x: Sequence[int]) -> Fraction:
    positions = range(0, len(x), C.S5_STRIDE)
    return sum((Fraction(P.f32_of_bits(x[p])) ** 2 for p in positions), Fraction(0)) / len(positions)


def dead_blocks(fmt: Fmt, x: Sequence[int]) -> list[bool]:
    """Per scale block: dead when its clean |max| exceeds 12ρ (exact ρ² over every 8th position), i.e. the noise is
    below τ = 1/8 of the block's code step."""
    rho_sq = _exact_rho_sq(x)
    return [Fraction(P.f32_of_bits(max(w & 0x7FFFFFFF for w in x[b:b + fmt.block]))) ** 2 > TAU_SQ_INV * rho_sq
            for b in range(0, len(x), fmt.block)]


def row_passes(fmt: Fmt, x: Sequence[int]) -> bool:
    """The per-row rules: a_E is a normal UE4M3 (at least 2^-6), and at most one scale block in 64 is dead."""
    try:
        ae = stats(fmt, x)[4]
    except P.Reject:
        return False
    return ae >= UE4M3_NORMAL and DEAD_SHARE * sum(dead_blocks(fmt, x)) <= len(x) // fmt.block


def tile_debit(fmt: Fmt, a_x: Sequence[Sequence[int]], b_x: Sequence[Sequence[int]], n: int) -> Fraction:
    """The debit over a tile's credited words (rows a_x of A, b_x of B, as FP32 words), in FP4 units: `fmt.block` MACs
    per word and dead x dead block pair, and each dead A element's credited forming by the tile's share of its n columns."""
    da = [dead_blocks(fmt, x) for x in a_x]
    db = [dead_blocks(fmt, x) for x in b_x]
    pairs = sum(sum(p and q for p, q in zip(ra, rb)) for ra in da for rb in db)
    elements = fmt.block * sum(sum(r) for r in da)
    return fmt.block * pairs + Fraction(FORMING_CREDITED * elements * len(b_x), n)


# ---- the scheme ---------------------------------------------------------------------------------------------------

class PearlC4:
    """Pearl-C on sm_120's FP4: ``k_min`` = 1024 and 64 x 64 tiles; smaller bounds make a test instance."""

    tickets = C.PearlC.tickets

    def __init__(self, fmt: str = "nvfp4", k_min: int = 1024, tile: tuple[int, int] = (TILE, TILE)):
        if fmt not in FORMATS:
            raise ValueError(f"format {fmt!r}: one of {', '.join(FORMATS)}")
        self.fmt = FORMATS[fmt]
        self.name = NAMES[fmt]
        self.k_min, self.tile = k_min, tile
        self.certificate = Certificate(
            conjecture="tt-out-fp4",
            gamma0=None,
            omega=Fraction(1),
            claims=("tt-out-fp4", "random-oracle", "cr/blake3", "device-reproducibility", "beacon-unpredictability"),
            hypotheses=(
                "TT_OUT(1/400) for this Π and domain, not yet stated by the theory lane: no sm_120 W1 program outputs "
                "correct C̃ and U words for less than 99.75% of their honest cost; with exact chains it rests on the "
                "salt-dependence of every code and block scale, not on rounding (research store "
                "internal/pouw/rtx-pro/fp4-design.md §6)",
                f"sm_120's mma.sync kind::mxf4nvf4 and BF16 m16n8k16 are reproduced exactly ({self.fmt.atom.name}, "
                "blackwell_sm120_nvf4_m16n8k64 for the noise atom, hopper_bf16_m16n8k16), pinned on the RTX 5090",
                "the E2M1, UE4M3 and UE8M0 casts are as modelled (from the PTX ISA, not captured)",
            ),
            allowances=(
                "ρ over every 8th position depends on A alone (2 units per element) and is not credited",
                "dead blocks are debited by a threshold rule (τ = 1/8); zero-sum atoms are not debited",
                "the unscale is glue outside the checked values",
            ),
            parameters=(("k_min", k_min), ("tile_m", tile[0]), ("tile_n", tile[1]), ("rank", R), ("block", self.fmt.block),
                        ("delta_inv", 4), ("tau_inv", 8), ("dead_share", DEAD_SHARE)),
        )

    def check(self, matmul: Matmul) -> None:
        s = matmul.shape
        if not (self.k_min <= s.k <= 1 << 16 and s.k % ALIGN_K == 0 and s.m % self.tile[0] == 0
                and s.n % self.tile[1] == 0 and s.m <= 1 << 24 and s.n <= 1 << 24):
            raise DomainError(f"{self.name}: {s} is outside {self.k_min} <= k <= 2^16 with 128 | k, tiles of "
                              f"{self.tile}, m, n <= 2^24")

    def tiles(self, shape: Shape) -> list[Tile]:
        return grid_tiles(shape, *self.tile)

    def wref(self, shape: Shape) -> int:
        """The honest W1 cost of one unit in FP4 units: the chain (1 per MAC, no promotion), the fused A′·F_B (r per element
        of A), the BF16 peel (4 per MAC over depth 2r) and A's forming (117 per element).  The weight side is `job_cost`."""
        m, k, n = shape.m, shape.k, shape.n
        return m * n * k + m * k * R + BF16_MAC_UNITS * PEEL_K * m * n + (FORMING_CREDITED + FORMING_A_ONLY) * m * k

    def credit_of(self, shape: Shape) -> int:
        """`wref` less ρ's 2 units per element of A (A alone)."""
        return self.wref(shape) - FORMING_A_ONLY * shape.m * shape.k

    def job_cost(self, n: int, k: int) -> int:
        """The weight side, once per weight and epoch: B's forming, B̃·F_A and G as chains, and P_B's FP32 FMA chain."""
        return (FORMING_CREDITED + FORMING_A_ONLY) * n * k + n * R * k + R * R * k + FP32_UNITS * n * R * R

    def noise(self, ctx: Context) -> C.Noise:
        lab = lambda m, r: keyed(blake3(LABELS[r]), m)  # noqa: E731
        return C.Noise(lab(ctx.salt + ctx.root_a + ctx.root_b + ctx.matmul.index.to_bytes(8, "little"), "seed-A"),
                       lab(ctx.salt + ctx.root_b, "seed-B"))

    def parse_row(self, row: bytes, k: int) -> list[int]:
        return parse_row(row, k)

    def row_passes(self, words: Sequence[int]) -> bool:
        return row_passes(self.fmt, words)

    def tile_cap(self, record, ops: C.Ops, rows: Sequence[int], cols: Sequence[int]) -> tuple[Fraction, Fraction]:
        """(debit, cap) over the credited words at tile positions rows x cols: `tile_debit`, and ρ times the credited
        cells' credit (each row's `credit_of` share at |K|)."""
        s = record.matmul.shape
        debit = tile_debit(self.fmt, [ops.a.x[i] for i in rows], [ops.b.x[j] for j in cols], s.n)
        cell = Fraction(self.credit_of(Shape(1, record.k_kept, s.n)), s.n)
        return debit, CAP * cell * len(rows) * len(cols)

    def _side(self, rows: Sequence[bytes], idx: Sequence[int], role_e: int, role_f: int, seed_e: bytes, seed_b: bytes,
              k: int) -> tuple[Formed, list[list[int]]]:
        e = [line(seed_e, role_e, i) for i in idx]
        return form(self.fmt, [parse_row(r, k) for r in rows], e, basis(seed_b, role_f, k)), e

    def encode(self, ctx: Context, noise: C.Noise, a_rows: Sequence[bytes], b_rows: Sequence[bytes], tile: Tile) -> C.Ops | None:
        k = ctx.matmul.shape.k
        try:
            a, e_a = self._side(a_rows, tile.rows, ROLE_EA, ROLE_FA, noise.seed_a, noise.seed_b, k)
            b, e_b = self._side(b_rows, tile.cols, ROLE_EB, ROLE_FB, noise.seed_b, noise.seed_b, k)
            return C.Ops(a, b, e_a, e_b, peel_factors_a(self.fmt, a, e_a, noise.seed_b, k),
                         peel_factors_b(self.fmt, b, e_b, noise.seed_b, k))
        except P.Reject:
            return None

    def admissible(self, ctx: Context, ops: C.Ops) -> bool:
        """Every row of both operands passes `row_passes`."""
        return all(row_passes(self.fmt, x) for side in (ops.a, ops.b) for x in side.x)

    def checked(self, ctx: Context, ops: C.Ops) -> C.Checked:
        f = self.fmt
        c = [[chain(f, ca, cb, sa, sb) for cb, sb in zip(ops.b.codes, ops.b.scales)]
             for ca, sa in zip(ops.a.codes, ops.a.scales)]
        u = [[peel(c[i][j], ops.pa[i], ops.pb[j]) for j in range(len(ops.pb))] for i in range(len(ops.pa))]
        return C.Checked(c, u)

    def digest(self, ctx: Context, noise: C.Noise, checked: C.Checked) -> bytes:
        return C.leaf(checked)

    def useful(self, ctx: Context, noise: C.Noise, ops: C.Ops, checked: C.Checked) -> list[list[float]]:
        """U/(α_A,i · α_B,j) in double, for measuring quality; the served y is the integration's pinned unscale."""
        return [[P.f32_of_bits(w) / (P.f32_of_bits(ops.a.alpha[i]) * P.f32_of_bits(ops.b.alpha[j]))
                 for j, w in enumerate(row)] for i, row in enumerate(checked.u)]


__all__ = ["FORMATS", "MXFP4", "NAMES", "NVFP4", "PearlC4", "clean", "e2m1", "form", "line", "row_passes", "tile_debit"]
