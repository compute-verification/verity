"""``pearl-c-nvfp4-v0`` and ``pearl-c-mxfp4-v0``: Pearl-C on sm_120's block-scaled FP4 (``mma.sync kind::mxf4nvf4``).

The design is research store ``internal/pouw/rtx-pro/fp4-design.md`` (GPU 5's lane).  It is Pearl-C (`pearl_c`) with the
FP8 product replaced by the E2M1 one: the noise enters inside one atom per element, one noisy cast forms the operand,
Pearl's tickets are drawn on C̃, the BF16 clean-up U is checked, and every C̃ and U word is hashed into the tile's leaf
(`pearl_c.leaf`).  γ rests on TT_OUT-FP4, which the theory lane has stated and nobody has proved.

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
peel      Ẽ = bf16(E · a_E · 2^-4) (exact, E's sign kept);  P_A,i = [Ẽ_A,i ‖ RNE_bf16(AF_i)],
          P_B,j = [RNE_bf16(FP32 add(Σ_p fma(Ẽ_B,jp, G_pq), −BF_jq)) ‖ −Ẽ_B,j];
          U_ij = C̃_ij, then four HOPPER_BF16_M16N8K16 steps (sm_120 mma.sync m16n8k16) over P_A,i · P_B,j
leaf      pearl_c.leaf over C̃ ‖ U;  tickets: Pearl's lottery over C̃ (pearl_c.PearlC.tickets)
useful    U_ij / (α_A,i · α_B,j), the model's own unscale (glue)
~~~

The clean-up's algebra is `pearl_c`'s with N_A = Ẽ_A·F_Aᵀ: (A′ − N_A)(B̃ − N_B)ᵀ = C̃ + P_A·P_Bᵀ.  The clean path (PoUW
off) is `clean`: the same α and block rule on α·x, no noise.

**Why the chain's exactness matters.**  On the domain the chain is exact, so rounding does not tie a word to its order,
as Pearl-C's H100 argument had it.  The design rests instead on every code and every block scale depending on the salt.

**The domain and the debit** are the theory lane's (research store ``internal/pouw/rtx-pro/theory-pearl-c4-domain.md``
§1, §2; TT_OUT-FP4, ``tt-out/fp4-sm120``, proposed).  Admission is per row, salt-free (`row_passes`): D-NF (a_E a normal
UE4M3, one a_E per row) and D-SS's dead share (a block is dead when its clean |max| exceeds 10ρ; at most one in 64).  A
drawn tile's debit is replayed over its credited words (`tile_debit`):

~~~text
1  2:4 groups    per row and 4-group with ≥ 2 salt-dead codes: 0.5 per MAC (1 when all 4 are), on every word the row feeds
2  identity      per atom whose lone skip leaves the word unchanged: 64 per atom-word
3  dead pairs    per block dead on both the A row and the B row: `block` per pair-word
4  forming       per salt-dead or unneeded A element: f_s · |cols|/n
5  sub-grid      per (word, block) with 2304·m_a·m_b·2^(e_a+e_b−2) below its step's grid (it adds exactly 0): `block`
D-24 windows     per row and aligned 128 of k whose every 4-group has ≤ 2 nonzero codes: 0.5 per MAC, on every word it feeds
cap              debit ≤ ρ · the credited cells' share of creditOf, ρ = 1/400
~~~

A code is *salt-dead* unless two draws in the noise support give it two values (`salt_dead`), so the forming debit is on
codes, not on the atom's word.  D-24 is charged per row window rather than per 16-row span (a cheaper computation may
gather any 16 such rows), and the literal spans are counted.  D-SS's spread bound (2^15.8, implied by D-NF) is counted, not
enforced: items 2 and 5 charge what it would exclude.  D-SS's scale-byte range is enforced (`scale_bytes_ok`, server.md
09:38Z): no UE4M3 byte with bit 7 set, which sm_120 reads from its low 7 bits (a second encoding of the same C̃), and no
zero scale, which would zero a noise block's products.  The verifier forms every scale from the committed rows, so no
prover's byte reaches the checked values; the rule guards that forming, and every replayed step reads bytes as the
hardware does (`hw_scales`).

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
from verity.errors import InvalidArtifact
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
DEAD_RHO = 10                      # dead block: amax_b > 10ρ (theory-pearl-c4-domain.md §5, the code-level salt-dead fix)
DEAD_SHARE = 64                    # a row passes with at most one dead block in 64

#: W1 in FP4 units (1 = one dense FP4 MAC on sm_120): a BF16 MAC 4, an FP32 instruction 16 (placeholders from public peak
#: rates; the measured FADD is 16.9).
BF16_MAC_UNITS, FP32_UNITS = 4, 16
#: Forming per operand element (theory-pearl-c4-domain.md §2).  Credited, each step that reads the salt at its cheapest
#: bit-exact price: f_s = 98 (the noise atom 64, the dense k64's full shape; the block-scale operations about 1; the cast
#: 32.8, FMUL and F2FP).  Uncredited, 19: the noisy amax (one FMNMX, 17) and ρ's every-8th accumulate (2), in W_ref alone.
FORMING_CREDITED, FORMING_UNCREDITED = 98, 19
CAP = Fraction(1, 400)
SPAN_K = 128                       # D-24: one mma.sp m16n8k128 takes 16 rows x 128 of k
SPREAD_X5 = 79                     # D-SS: spread ≤ 2^15.8, as spread^5 ≤ 2^79
GROUP_MAX = 2304                   # the largest |group dot| in quarters: 16 products of 6 x 6

_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_E2M1_HALVES = (0, 1, 2, 3, 4, 6, 8, 12)


@dataclass(frozen=True)
class Fmt:
    """One FP4 operand format: the chain's atom, the scale block along k, and the scale byte of 1.0.  The module's rules
    tell the formats apart by identity (``fmt is NVFP4``), so a format unpickles as the module's own."""

    name: str
    atom: BlockScaledAlignAdd
    block: int
    one: int

    def __reduce__(self):
        return _format, (self.name,)


def _format(name: str) -> Fmt:
    return FORMATS[name]


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
def seed_b(salt: bytes, root_b: bytes) -> bytes:
    """B̃'s noise seed, keyed by this job's salt and B's root.  So B̃ exists only once the salt does, and no program
    can precompute with it: the salt-free base split is two-sided (A′·B̃ᵀ = A0·B0ᵀ + ΔA·B̃ᵀ + A0·ΔBᵀ, the cheater
    paying both corrections), which is what ties it on the stride rows (TT_OUT-FP4's rule D-SK,
    theory-pearl-c4-domain.md §2)."""
    return keyed(blake3(LABELS["seed-B"]), salt + root_b)


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

def hw_scales(fmt: Fmt, scales: Sequence[int]) -> list[int]:
    """Scale bytes as sm_120's block-scaled step reads them: a UE4M3 byte from its low 7 bits, bit 7 ignored (the
    assessor's run of all 256 bytes, ``r20260930-093120-b2f2``); UE8M0 bytes as written.  Every replayed step reads
    scales through it, though the domain (`scale_bytes_ok`) never lets bit 7 reach one."""
    return [s & 0x7F for s in scales] if fmt is NVFP4 else list(scales)


def scale_bytes_ok(fmt: Fmt, scales: Sequence[int]) -> bool:
    """D-SS's range, enforced (`check_scale_bytes`): one encoding per scale, no zero scale on a block that carries noise
    (every block does), and no NaN.  UE4M3 in [0x01, 0x7E] (bit 7 clear, not 0x00, not 0x7F); UE8M0, whose 8 bits are all
    its exponent and which has no zero, short of its NaN 0xFF.  Honest forming meets it by construction (`block_scale`)."""
    return all(UE4M3_MIN <= s <= 0x7E for s in scales) if fmt is NVFP4 else all(0 <= s < 0xFF for s in scales)


def check_scale_bytes(fmt: Fmt, side: Formed) -> None:
    for row in side.scales:
        if not scale_bytes_ok(fmt, row):
            raise P.Reject("a formed scale byte is outside D-SS's range (bit 7 set, a zero scale or NaN)")


def chain(fmt: Fmt, a_codes: Sequence[int], b_codes: Sequence[int], a_scales: Sequence[int], b_scales: Sequence[int]) -> int:
    """The atom chained from +0 over the full k, no promotion."""
    if len(a_codes) % ATOM:
        raise P.Reject("the chain multiplies in atoms of k = 64")
    return _finite(fmt.atom.chain(0, a_codes, b_codes, hw_scales(fmt, a_scales), hw_scales(fmt, b_scales)))


@lru_cache(maxsize=16)
def gram(fmt: Fmt, seed_b: bytes, k: int) -> tuple[tuple[int, ...], ...]:
    """G = F_Bᵀ·F_A (r x r) as FP32 words, G_pq = chain(F_B[:, p], F_A[:, q]) at scale 1.0; once per job."""
    fa, fb = basis(seed_b, ROLE_FA, k), basis(seed_b, ROLE_FB, k)
    ones = [fmt.one] * (k // fmt.block)
    fa_cols = [column(fa, q) for q in range(R)]
    return tuple(tuple(chain(fmt, column(fb, p), fa_cols[q], ones, ones) for q in range(R)) for p in range(R))


def e_tilde(e_row: Sequence[int], ae: int) -> list[int]:
    """Ẽ = bf16(E · a_E · 2^-4), exact (E2M1 times UE4M3 has at most 6 significant bits), with E's sign: code 0x8 is −0."""
    scale = Fraction(P.fp8_to_f32(ae)) / 16
    return [P.bf16_of_fraction(e2m1_value(e & 7) * scale) | (0x8000 if e & 8 else 0) for e in e_row]


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


# ---- the domain rules and the debit (theory-pearl-c4-domain.md §1, §2) --------------------------------------------

def _exact_rho_sq(x: Sequence[int]) -> Fraction:
    positions = range(0, len(x), C.S5_STRIDE)
    return sum((Fraction(P.f32_of_bits(x[p])) ** 2 for p in positions), Fraction(0)) / len(positions)


def dead_blocks(fmt: Fmt, x: Sequence[int]) -> list[bool]:
    """Per scale block: dead when its clean |max| exceeds 10ρ (exact ρ² over every 8th position): a salt-free admission
    screen, below which a dominant element's code still moves on some draw of the noise."""
    rho_sq = _exact_rho_sq(x)
    return [Fraction(P.f32_of_bits(max(w & 0x7FFFFFFF for w in x[b:b + fmt.block]))) ** 2 > DEAD_RHO ** 2 * rho_sq
            for b in range(0, len(x), fmt.block)]


def row_passes(fmt: Fmt, x: Sequence[int]) -> bool:
    """Admission, per row and salt-free: D-NF (a_E is a normal UE4M3, at least 2^-6) and D-SS's dead share (at most one
    scale block in 64 is dead)."""
    try:
        ae = stats(fmt, x)[4]
    except P.Reject:
        return False
    return ae >= UE4M3_NORMAL and DEAD_SHARE * sum(dead_blocks(fmt, x)) <= len(x) // fmt.block


def scale_dyadic(fmt: Fmt, byte: int) -> tuple[int, int]:
    """A scale byte's value as m·2^e (the atom's scaleDyadic): UE4M3 (8 + m, e − 10), or (m, −9) at exponent field 0,
    from the low 7 bits as the hardware reads them (`hw_scales`); UE8M0 (8, byte − 130)."""
    if fmt is MXFP4:
        return 8, byte - 130
    e, m = (byte & 0x7F) >> 3, byte & 7
    return (8 + m, e - 10) if e else (m, -9)


def _witness(ae: int, f: Sequence[int]) -> list[tuple[int, int, bool]]:
    """The noise atom's group terms under E ≡ +6 and F_t = f."""
    pad = [0] * (ATOM - R)
    return BLACKWELL_SM120_NVF4.group_terms([0x7] * R + pad, list(f) + pad, [ae] * 4, [NOISE_F_SCALE] * 4)


def salt_dead(fmt: Fmt, x: Sequence[int], alpha: int, ae: int) -> list[bool]:
    """Per element of one row: whether its E2M1 code may be the same at every noise draw (P3's saltDead, on codes).

    An element is salt-live when two draws in the support (every pair of 32-code E2M1 lines) give its code two values;
    every other element counts as salt-dead, the sound side.  The draws: E ≡ +6 on the row, the element's F_t ≡ +6, 0 or
    −6, and every other F_u of its block ≡ 0, so the block's other elements are the atom on α·x_u alone and its amax is
    theirs or the element's."""
    atom = BLACKWELL_SM120_NVF4
    up, zero, down = (_witness(ae, [f] * R) for f in (0x7, 0x0, 0xF))
    recip: dict[int, int] = {}

    def code(v: int, amax: int) -> int:
        """The element's code at one draw, with −0 (0x8) as 0: salt-deadness compares values."""
        if amax not in recip:
            recip[amax] = block_scale(fmt, amax)[1]
        c = e2m1(fp32.mul(v, recip[amax]))
        return c if c & 7 else 0

    dead = []
    for b in range(0, len(x), fmt.block):
        c = [fp32.mul(alpha, w) for w in x[b:b + fmt.block]]
        clean = [_finite(atom.align_add(w, zero)) for w in c]
        mags = [w & 0x7FFFFFFF for w in clean]
        top = max(mags)
        at = mags.index(top)
        second = max(mags[:at] + mags[at + 1:], default=0)
        for t, (ct, dt) in enumerate(zip(c, clean)):
            rest = second if t == at else top
            draws = (_finite(atom.align_add(ct, up)), dt, _finite(atom.align_add(ct, down)))
            dead.append(len({code(v, max(v & 0x7FFFFFFF, rest)) for v in draws}) == 1)
    return dead


def two_four_units(dead: Sequence[bool]) -> int:
    """Item 1's charge per word the row feeds: 2 (0.5 per MAC) per 4-group with ≥ 2 salt-dead codes, 4 when all 4 are."""
    units = 0
    for q in range(0, len(dead), 4):
        d = sum(dead[q:q + 4])
        units += 4 if d == 4 else 2 if d >= 2 else 0
    return units


def two_four_windows(codes: Sequence[int]) -> list[bool]:
    """D-24, per aligned 128 of k: every 4-group has at most 2 nonzero codes ((code & 7) ≠ 0)."""
    return [all(sum(1 for c in codes[q:q + 4] if c & 7) <= 2 for q in range(w, w + SPAN_K, 4))
            for w in range(0, len(codes), SPAN_K)]


def spread_ok(fmt: Fmt, scales: Sequence[int]) -> bool:
    """D-SS's spread: max_b val(s_b) / min_b val(s_b) ≤ 2^15.8."""
    vals = [Fraction(m) * Fraction(2) ** e for m, e in (scale_dyadic(fmt, s) for s in scales)]
    return min(vals) > 0 and (max(vals) / min(vals)) ** 5 <= 2 ** SPREAD_X5


def word_replay(fmt: Fmt, a_codes: Sequence[int], b_codes: Sequence[int], a_scales: Sequence[int],
                b_scales: Sequence[int]) -> tuple[int, int, int]:
    """One word's chain replayed from +0: (C̃, identity atoms, sub-grid blocks).

    An atom is an identity atom when its lone skip (its incoming word passed through) leaves the final word unchanged
    (item 2; a skipped chain that leaves the model's range counts, the sound side).  A block is sub-grid when its largest
    possible scaled dot, 2304·m_a·m_b·2^(e_a+e_b−2), lies below its step's grid `atom.grid`, so it adds exactly 0
    (item 5)."""
    atom, ns = fmt.atom, ATOM // fmt.block
    steps = len(a_codes) // ATOM
    a_scales, b_scales = hw_scales(fmt, a_scales), hw_scales(fmt, b_scales)
    da, db = [scale_dyadic(fmt, s) for s in a_scales], [scale_dyadic(fmt, s) for s in b_scales]
    terms, w, subgrid = [], [0], 0
    for s in range(steps):
        tm = atom.group_terms(a_codes[s * ATOM:(s + 1) * ATOM], b_codes[s * ATOM:(s + 1) * ATOM],
                              a_scales[s * ns:(s + 1) * ns], b_scales[s * ns:(s + 1) * ns])
        lsb = atom.grid(w[-1], tm)
        if lsb is not None:
            for blk in range(s * ns, (s + 1) * ns):
                (ma, ea), (mb, eb) = da[blk], db[blk]
                top, e = GROUP_MAX * ma * mb, lsb - ea - eb + 2
                subgrid += top == 0 or (e > 0 and top < 1 << e)
        terms.append(tm)
        w.append(_finite(atom.align_add(w[-1], tm)))
    identity = 0
    for t in range(steps):
        c, s = w[t], t + 1
        try:
            while s < steps and c != w[s]:
                c = atom.align_add(c, terms[s])
                s += 1
        except InvalidArtifact:
            identity += 1
            continue
        identity += c == w[s]
    return w[-1], identity, subgrid


@dataclass(frozen=True)
class Debit:
    """A drawn tile's replayed debit over its credited words in FP4 units, by item (module docstring), and the counts
    behind it and behind the domain checks."""

    two_four: int
    identity: int
    dead_pairs: int
    forming: Fraction
    subgrid: int
    windows: int
    counts: dict

    @property
    def total(self) -> Fraction:
        return self.two_four + self.identity + self.dead_pairs + self.forming + self.subgrid + self.windows

    def to_json(self) -> dict:
        return {"total": str(self.total), "two_four": self.two_four, "identity": self.identity,
                "dead_pairs": self.dead_pairs, "forming": str(self.forming), "subgrid": self.subgrid,
                "windows": self.windows, "counts": dict(self.counts)}


def tile_debit(fmt: Fmt, a: Formed, b: Formed, rows: Sequence[int], cols: Sequence[int], n: int) -> Debit:
    """The debit over the credited words at tile positions rows x cols of a tile's formed operands (A′ = a, B̃ = b)."""
    dead_a = {i: salt_dead(fmt, a.x[i], a.alpha[i], a.ae[i]) for i in rows}
    dead_b = {j: salt_dead(fmt, b.x[j], b.alpha[j], b.ae[j]) for j in cols}
    two_four = (len(cols) * sum(two_four_units(dead_a[i]) for i in rows)
                + len(rows) * sum(two_four_units(dead_b[j]) for j in cols))
    win_a = {i: two_four_windows(a.codes[i]) for i in range(len(a.codes))}
    win_b = {j: two_four_windows(b.codes[j]) for j in cols}
    windows = SPAN_K // 2 * (len(cols) * sum(sum(win_a[i]) for i in rows) + len(rows) * sum(sum(win_b[j]) for j in cols))
    identity = subgrid = 0
    for i in rows:
        for j in cols:
            _, idn, sg = word_replay(fmt, a.codes[i], b.codes[j], a.scales[i], b.scales[j])
            identity, subgrid = identity + idn, subgrid + sg
    blocks_a, blocks_b = {i: dead_blocks(fmt, a.x[i]) for i in rows}, {j: dead_blocks(fmt, b.x[j]) for j in cols}
    pairs = sum(p and q for i in rows for j in cols for p, q in zip(blocks_a[i], blocks_b[j]))
    unneeded = [all(b.codes[j][t] & 7 == 0 for j in cols) for t in range(len(a.codes[0]))]
    elements = sum(d or u for i in rows for d, u in zip(dead_a[i], unneeded))
    spans = sum(all(win_a[i][w] for i in range(r, r + 16)) for r in range(0, len(a.codes) - 15, 16)
                for w in range(len(win_a[0])))
    counts = {"salt_dead_a": sum(sum(dead_a[i]) for i in rows), "salt_dead_b": sum(sum(dead_b[j]) for j in cols),
              "two_four_windows_a": sum(sum(win_a[i]) for i in rows), "two_four_windows_b": sum(sum(win_b[j]) for j in cols),
              "d24_spans_a": spans, "identity_atom_words": identity, "subgrid_block_words": subgrid, "dead_pairs": pairs,
              "forming_elements": elements, "unneeded_positions": sum(unneeded),
              "dss_spread_rows": sum(not spread_ok(fmt, a.scales[i]) for i in rows)
              + sum(not spread_ok(fmt, b.scales[j]) for j in cols),
              "words": len(rows) * len(cols)}
    return Debit(two_four, ATOM * identity, fmt.block * pairs, Fraction(FORMING_CREDITED * elements * len(cols), n),
                 fmt.block * subgrid, windows, counts)


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
        self.last_debit: Debit | None = None
        self._encoded: tuple | None = None
        self.certificate = Certificate(
            conjecture="tt-out/fp4-sm120",
            gamma0=None,
            omega=Fraction(1),
            claims=("tt-out/fp4-sm120", "random-oracle", "cr/blake3", "cr/sha-256", "device-reproducibility",
                    "beacon-unpredictability"),
            hypotheses=(
                "TT_OUT-FP4(1/400) for this Π and domain (D-NF, D-24, D-SS, D-SB, and D-SK: B̃'s noise keyed by "
                "this job's salt, seed-B = keyed(salt ‖ root_b), recomputed by the verifier), as the theory lane "
                "states it, proposed and "
                "not proved: no sm_120 W1 program outputs correct C̃ and U words of the units whose replayed debit is under "
                "the cap for less than their credit less the debit (research store "
                "internal/pouw/rtx-pro/theory-pearl-c4-domain.md §2)",
                f"sm_120's mma.sync kind::mxf4nvf4 and BF16 m16n8k16 are reproduced exactly ({self.fmt.atom.name}, "
                "blackwell_sm120_nvf4_m16n8k64 for the noise atom, hopper_bf16_m16n8k16), pinned on the RTX 5090",
                "the E2M1, UE4M3 and UE8M0 casts are as modelled (from the PTX ISA, not captured)",
            ),
            allowances=(
                "the noisy block amax (17 units per element) and ρ over every 8th position (2) are in W_ref, not credited",
                "savings no replay reaches (salt-live codes that happen not to change, cross-atom cancellation) are "
                "TT_OUT-FP4's; the replayed debit charges the rest (tile_debit)",
                "the unscale is glue outside the checked values",
            ),
            parameters=(("k_min", k_min), ("tile_m", tile[0]), ("tile_n", tile[1]), ("rank", R), ("block", self.fmt.block),
                        ("delta_inv", 4), ("dead_rho", DEAD_RHO), ("dead_share", DEAD_SHARE)),
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
        return m * n * k + m * k * R + BF16_MAC_UNITS * PEEL_K * m * n + (FORMING_CREDITED + FORMING_UNCREDITED) * m * k

    def credit_of(self, shape: Shape) -> int:
        """`wref` less A's uncredited forming, 19 units per element: creditOf = mnk + 32·mk + 256·mn + f_s·mk."""
        return self.wref(shape) - FORMING_UNCREDITED * shape.m * shape.k

    def job_cost(self, n: int, k: int) -> int:
        """The weight side, once per weight and epoch: B's forming, B̃·F_A and G as chains, and P_B's FP32 FMA chain."""
        return (FORMING_CREDITED + FORMING_UNCREDITED) * n * k + n * R * k + R * R * k + FP32_UNITS * n * R * R

    def noise(self, ctx: Context) -> C.Noise:
        """The unit's seeds: A's keyed by the job's salt, both roots and the unit, B̃'s by `seed_b`.  The verifier forms
        both sides itself from these seeds on every drawn tile, so a word derived from any other B̃ fails its checked
        values."""
        unit = ctx.salt + ctx.root_a + ctx.root_b + ctx.matmul.index.to_bytes(8, "little")
        seed_a = keyed(blake3(LABELS["seed-A"]), unit)
        return C.Noise(seed_a, seed_b(ctx.salt, ctx.root_b))

    def parse_row(self, row: bytes, k: int) -> list[int]:
        return parse_row(row, k)

    def row_passes(self, words: Sequence[int]) -> bool:
        return row_passes(self.fmt, words)

    def tile_cap(self, record, ops: C.Ops, rows: Sequence[int], cols: Sequence[int]) -> tuple[Fraction, Fraction]:
        """(debit, cap) over the credited words at tile positions rows x cols: `tile_debit`'s total, and ρ times the
        credited cells' share of creditOf (each row's `credit_of` at |K|, over n).  The breakdown stays in `last_debit`."""
        s = record.matmul.shape
        self.last_debit = tile_debit(self.fmt, ops.a, ops.b, rows, cols, s.n)
        cell = Fraction(self.credit_of(Shape(1, record.k_kept, s.n)), s.n)
        return self.last_debit.total, CAP * cell * len(rows) * len(cols)

    def _side(self, rows: Sequence[bytes], idx: Sequence[int], role_e: int, role_f: int, seed_e: bytes, seed_b: bytes,
              k: int) -> tuple[Formed, list[list[int]]]:
        e = [line(seed_e, role_e, i) for i in idx]
        side = form(self.fmt, [parse_row(r, k) for r in rows], e, basis(seed_b, role_f, k))
        check_scale_bytes(self.fmt, side)
        return side, e

    def encode(self, ctx: Context, noise: C.Noise, a_rows: Sequence[bytes], b_rows: Sequence[bytes], tile: Tile) -> C.Ops | None:
        """The tile's formed operands and peel factors; the last tile's are kept, since the work law encodes a drawn tile
        twice (its digest, then its tile rule)."""
        key = (ctx, noise, tuple(a_rows), tuple(b_rows), tile)
        if self._encoded is not None and self._encoded[0] == key:
            return self._encoded[1]
        k = ctx.matmul.shape.k
        try:
            a, e_a = self._side(a_rows, tile.rows, ROLE_EA, ROLE_FA, noise.seed_a, noise.seed_b, k)
            b, e_b = self._side(b_rows, tile.cols, ROLE_EB, ROLE_FB, noise.seed_b, noise.seed_b, k)
            ops = C.Ops(a, b, e_a, e_b, peel_factors_a(self.fmt, a, e_a, noise.seed_b, k),
                        peel_factors_b(self.fmt, b, e_b, noise.seed_b, k))
        except P.Reject:
            ops = None
        self._encoded = (key, ops)
        return ops

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


__all__ = ["FORMATS", "MXFP4", "NAMES", "NVFP4", "Debit", "PearlC4", "clean", "e2m1", "form", "hw_scales", "line",
           "row_passes", "salt_dead", "scale_bytes_ok", "seed_b", "tile_debit", "word_replay"]
