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
UE4M3, one a_E per row).  A block is screened when (clean amax_b)² > 90·ρ² (9.5ρ, `dead_blocks`); its row is admitted
and its block is charged per word where both operands screen it (item 3), so a row with many screened blocks is rejected
only when its tile's debit passes the cap (server.md 10:35Z; the theory lane's 10:50Z ruling).  Item 4 charges the
reference's witness alone, not every element of a screened block (server.md 11:32Z).  A drawn tile's debit is replayed
over its credited words (`tile_debit`):

~~~text
F1′ base split   per row of A′ and of B̃ and aligned 128 of k: 128·max(0, S_w − min(c·M_w, 1/8)) on every word the row
                 feeds, c = 16.92/128 (theory §6.2; it replaces item 1, 2:4 groups of salt-dead codes)
2  identity      per atom whose lone skip leaves the word unchanged: 64 per atom-word
3  dead pairs    per block screened on both the A row and the B row: `block` per pair-word
4  forming       per salt-dead (the reference's witness, `salt_dead`) or unneeded A element: f_s · |cols|/n
5  sub-grid      per (word, block) with 2304·m_a·m_b·2^(e_a+e_b−2) below its step's grid (it adds exactly 0): `block`
D-24 windows     per row and aligned 128 of k whose every 4-group has ≤ 2 nonzero codes: 0.5 per MAC, on every word it feeds
F2 int8 route    per tile: max(0, 1 − c_L − 4·(2 − f_A − f_B)) per chain MAC of its credited words (theory §6.3)
cap              debit ≤ ρ · the credited cells' share of creditOf, ρ = 1/400
~~~

**F1′** prices the base split A′ = A0 + ΔA (A0 each code's modal value, ΔA 2:4 plus a CUDA-core patch, or a skipped or
patched k64 atom) at its expected saving under the noise law given the row's E line, less A0's scale corrections
(`f1_row`).  The law is exact: S = Σ_r h(e_r)·h(f_r) over an element's 16^32 F lines (`_law`, integer generating
functions), and the atom is D = RZ32(C truncated onto its grid + S·a_E/64) (`_atom_sum`; D = C on a line with no nonzero
product), so each P(code = c) at a byte and each P(|D| below a byte's ceiling) is a count ratio.  A block's modal byte is
the product formula's argmax over its elements (the smaller on a tie), p_t = 1 − max_c P(code_t = c) at it, and p_t = 0
on witness-dead codes and screened blocks (`row_law`).  S_w = min(1, E[max(0, ½ − c·X_w)] + Σ_h E[max(0, ½ − c·N_h)]):
X_w the excess beyond 2:4 at the cheater's layout inside each block (sorted p, neighbours paired, chunks of alternate
pairs), N_h the changed codes in each aligned k64 half (`window_saving`).  M_w counts the window's blocks whose realised
byte is not their modal one.  The DPs run in IEEE doubles from the correctly rounded p_t, and S_w enters the debit as its
exact double, the one departure from exact arithmetic (deterministic, and pinned by the vectors).  MXFP4 sorts each
32-block and chunks every 16 sorted codes as NVFP4 does a block (GPU 5's extension; §6.2 states the 16-block).

**F2** prices the masked int8 route on flat scales (`f2_save`): f_A and f_B are the credited rows' shares of blocks at
their most common realised byte, and c_L = min over 1 ≤ L ≤ log2(k/B) of 2·(7/8)^L + 4.25·(1/m + 1/n)·((7/4)^L − 1)/0.75
at the unit's (m, k, n) (`c_l`).

A code is *salt-dead* unless two draws in the noise support give it two values (`salt_dead`), so the forming debit is on
codes, not on the atom's word.  D-24 is charged per row window rather than per 16-row span (a cheaper computation may
gather any 16 such rows), and the literal spans are counted.  D-SS's spread bound (2^15.8, implied by D-NF) is counted, not
enforced: items 2 and 5 charge what it would exclude.  D-SB is enforced on every formed side before hashing
(`check_scale_bytes`, server.md 10:32Z): every block scale in 0x01–0x7E (UE8M0: short of 0xFF) and every a_E at most 0x7E,
so no UE4M3 byte has bit 7 set, which sm_120 reads from its low 7 bits (a second encoding of the same C̃), none is 0x7F
(NaN), and no block scale is zero, which would zero a noise block's products; every committed word is finite
(`parse_row`).  An a_E of 0x00 is D-NF's exclusion, not a rejection: a zero row forms it.  The verifier forms every
scale from the committed rows, so no prover's byte reaches the checked values; the rule guards that forming, and every
replayed step reads bytes as the hardware does (`hw_scales`).

**Stand-ins.**  The atoms are the RTX 5090 pins (`BLACKWELL_SM120_NVF4`, `_MXF4`, `HOPPER_BF16_M16N8K16`), pending GPU 4's
RTX PRO recheck.  The E2M1, UE4M3 and UE8M0 casts are modelled from the PTX ISA, not captured.  A negative value that
rounds to zero keeps its sign (code 0x8); no checked word depends on it.
"""

from __future__ import annotations

import math
import struct
from bisect import bisect_left
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import accumulate

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
UE4M3_TOP = 0x7E                   # 448; 0x7F is NaN
DEAD_RHO_SQ = 90                   # screened block: (clean amax_b)² > 90·ρ², 9.5ρ (theory-pearl-c4-domain.md D-SS, §5)

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


def recip(fmt: Fmt, byte: int) -> int:
    """The FP32 reciprocal the cast multiplies by at a block scale byte: nearest even for UE4M3, exact for UE8M0."""
    return fp32.div(ONE, ue4m3_word(byte)) if fmt is NVFP4 else pow2_word(127 - byte)


def block_scale(fmt: Fmt, amax: int) -> tuple[int, int]:
    """(scale byte, FP32 reciprocal) of a block with |max| amax, by `fmt`'s rule (module docstring)."""
    v = fp32.mul(amax, SIXTH)
    s = max(ue4m3(v), UE4M3_MIN) if fmt is NVFP4 else ue8m0_up(v)
    return s, recip(fmt, s)


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

def hw_scales(fmt: Fmt, scales: Sequence[int]) -> list[int]:
    """Scale bytes as sm_120's block-scaled step reads them: a UE4M3 byte from its low 7 bits, bit 7 ignored (the
    assessor's run of all 256 bytes, ``r20260930-093120-b2f2``); UE8M0 bytes as written.  Every replayed step reads
    scales through it, though the domain (`scale_bytes_ok`) never lets bit 7 reach one."""
    return [s & 0x7F for s in scales] if fmt is NVFP4 else list(scales)


def scale_bytes_ok(fmt: Fmt, scales: Sequence[int]) -> bool:
    """D-SB's range for block scales: one encoding per scale, no zero scale on a block that carries noise (every block
    does), and no NaN.  UE4M3 in [0x01, 0x7E] (bit 7 clear, not 0x00, not 0x7F); UE8M0, whose 8 bits are all its exponent
    and which has no zero, short of its NaN 0xFF.  Honest forming meets it by construction (`block_scale`)."""
    return all(UE4M3_MIN <= s <= UE4M3_TOP for s in scales) if fmt is NVFP4 else all(0 <= s < 0xFF for s in scales)


def check_scale_bytes(fmt: Fmt, side: Formed) -> None:
    """D-SB on one formed side, before hashing: every block scale in `scale_bytes_ok`'s range and every a_E at most 0x7E
    (bit 7 clear, not NaN).  An a_E below 0x01 is left to D-NF, which excludes its row: a zero row (decode's filler, which
    the work law requires to be zero) forms a_E = 0x00.  The committed words' finiteness is `parse_row`'s."""
    for row in side.scales:
        if not scale_bytes_ok(fmt, row):
            raise P.Reject("a formed scale byte is outside D-SB's range (bit 7 set, a zero scale or NaN)")
    if not all(0 <= ae <= UE4M3_TOP for ae in side.ae):
        raise P.Reject("a row's a_E is outside D-SB's range (bit 7 set or NaN)")


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
    """Per scale block: screened when (clean amax_b)² > 90·ρ² (9.5ρ; exact, ρ² over every 8th position).  Salt-free, and
    it covers every salt-dead element (Lean `saltDead4_screened`).  The debit charges a block screened on both an A row and
    a B row per pair-word (item 3) instead of rejecting its rows; item 4 stays on the witness (`salt_dead`)."""
    rho_sq = _exact_rho_sq(x)
    return [Fraction(P.f32_of_bits(max(w & 0x7FFFFFFF for w in x[b:b + fmt.block]))) ** 2 > DEAD_RHO_SQ * rho_sq
            for b in range(0, len(x), fmt.block)]


def row_passes(fmt: Fmt, x: Sequence[int]) -> bool:
    """Admission, per row and salt-free: D-NF (a_E is a normal UE4M3, at least 2^-6).  A row's screened blocks are
    charged in its tiles' debit (`tile_debit`), and the tile's cap rejects them."""
    try:
        ae = stats(fmt, x)[4]
    except P.Reject:
        return False
    return ae >= UE4M3_NORMAL


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


# ---- F1′ and F2: the base split and the masked int8 route (theory-pearl-c4-domain.md §6.2–6.5) --------------------

PATCH = Fraction(1692, 100)        # the CUDA-core patch in FP4 units: one FFMA per word (§6.1)
C_PATCH = PATCH / SPAN_K           # c, the patch per MAC of a row window
A0_DENSE = Fraction(1, 8)          # a dense FP8 correction GEMM over k/16, per MAC: the most A0's scale corrections cost
F2_SLOTS = 4                       # F2's patch: FP4 slots per useful MAC per side
XM = 4                             # ½ − 4c < 0: past 3 excess or changed codes nothing is saved, so the DPs stop at 4
LINES = 16 ** R                    # one element's F lines, 2^128
_H = tuple(-_E2M1_HALVES[c & 7] if c & 8 else _E2M1_HALVES[c & 7] for c in range(16))
_SLOT = 17                         # bytes per coefficient in `_law`'s packing: every count is at most 2^128
_VX = tuple(float(max(Fraction(0), Fraction(1, 2) - C_PATCH * x)) for x in range(XM + 1))


@lru_cache(maxsize=4096)
def _law(mags: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    """The noise atom's line sum S = Σ_r h(e_r)·h(f_r) (quarters, h the E2M1 numerator) over one element's 16^32 F lines,
    for an E line with mags[c] codes of |numerator| `_E2M1_HALVES[c]`: (lo, cdf), cdf[j] the lines with S ≤ lo + j.

    Every f_r is uniform over the 16 nibbles, so S's generating function is Π_a (Σ_f z^(a·h(f)))^(n_a); it is multiplied
    exactly as integers, `_SLOT` bytes per coefficient (Kronecker substitution)."""
    poly, offset = 1, 0
    for a, n in zip(_E2M1_HALVES, mags, strict=True):
        if n and a:
            poly *= sum(1 << (8 * _SLOT * a * (h + 12)) for h in _H) ** n
            offset += 12 * a * n
        elif n:
            poly *= 16 ** n
    raw = poly.to_bytes((2 * offset + 1) * _SLOT, "little")
    return -offset, tuple(accumulate(int.from_bytes(raw[j:j + _SLOT], "little") for j in range(0, len(raw), _SLOT)))


def _atom_sum(c_word: int, ae: int) -> tuple[int, int, int] | None:
    """(A, Q, g): on a line of sum S with a nonzero product, the noise atom's exact sum is (A + S·Q)·2^g (C truncated toward
    zero onto the step's grid 2^g, and both noise groups' scaled dots, exact on it), so D = RZ32((A + S·Q)·2^g).  A line
    whose every product is zero anchors no grid, and its D is C.  None when a_E's mantissa is zero: every line is so."""
    (ma, ea), (mf, ef) = scale_dyadic(NVFP4, ae), scale_dyadic(NVFP4, NOISE_F_SCALE)
    if not ma:
        return None
    unit = ea + ef - 2
    g = BLACKWELL_SM120_NVF4.grid(c_word, [(1, unit, True)])
    if g > unit:
        raise P.Reject("the noise atom's grid lies above its group dots' unit")
    return int(math.ldexp(P.f32_of_bits(c_word), -g)), ma * mf << (unit - g), g


def _dyadic(word: int) -> tuple[int, int]:
    """A positive finite FP32 word as m·2^e."""
    field, frac = word >> 23 & 0xFF, word & 0x7FFFFF
    return (frac | 1 << 23, field - 150) if field else (frac, -149)


def _s_from(y: tuple[int, int, int], word: int) -> int:
    """The least S with (A + S·Q)·2^g at least the positive FP32 value `word`."""
    a, q, g = y
    m, e = _dyadic(word)
    if e >= g:
        return -((a - (m << (e - g))) // q)
    return -(((a << (g - e)) - m) // (q << (g - e)))


def _s_to(y: tuple[int, int, int], word: int) -> int:
    """The greatest S with (A + S·Q)·2^g at most minus the positive FP32 value `word`."""
    a, q, g = y
    m, e = _dyadic(word)
    if e >= g:
        return (-(m << (e - g)) - a) // q
    return (-m - (a << (g - e))) // (q << (g - e))


def _least(pred) -> int | None:
    """The least finite non-negative FP32 word with `pred`, monotone in the word; None when no finite word has it."""
    lo, hi = 0, 0x7F800000
    while lo < hi:
        mid = (lo + hi) // 2
        if pred(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo if lo < 0x7F800000 else None


@lru_cache(maxsize=None)
def _cells(fmt: Fmt, byte: int) -> tuple[int, ...]:
    """τ_1 … τ_7 at a block scale byte: τ_j is the least FP32 magnitude whose cast has |code| ≥ j.  Since D = RZ32(y), the
    code's value is sign(y)·j exactly when τ_j ≤ |y| < τ_(j+1)."""
    r = recip(fmt, byte)

    def at_least(j: int):
        def pred(w: int) -> bool:
            v = fp32.mul(w, r)
            return (v >> 23) & 0xFF == 0xFF or e2m1(v) & 7 >= j
        return pred
    return tuple(_least(at_least(j)) for j in range(1, 8))


@lru_cache(maxsize=None)
def _above(fmt: Fmt, byte: int) -> int | None:
    """The least FP32 magnitude whose block scale byte exceeds `byte`: a block's scale is at most `byte` exactly when its
    amax lies below it (None: at every finite amax)."""
    return _least(lambda w: block_scale(fmt, w)[0] > byte)


def _upto(cdf: Sequence[int], lo: int, s: int) -> int:
    j = s - lo
    return 0 if j < 0 else cdf[min(j, len(cdf) - 1)]


def _below(law: tuple[int, tuple[int, ...]], idle: int, y, c_word: int, u: int | None) -> int:
    """The F lines on which one element's |D| lies below the FP32 magnitude u (all of them when u is None)."""
    if u is None:
        return LINES
    stay = idle if c_word & 0x7FFFFFFF < u else 0
    if y is None:
        return stay
    lo, cdf = law
    s_lo, s_hi = _s_to(y, u) + 1, _s_from(y, u) - 1
    n = _upto(cdf, lo, s_hi) - _upto(cdf, lo, s_lo - 1)
    return n - (idle if s_lo <= 0 <= s_hi else 0) + stay


def _modal_byte(fmt: Fmt, law, idle: int, ys: Sequence, cs: Sequence[int]) -> int:
    """One block's most likely scale byte given E (the smaller on a tie), by the product formula: P(byte ≤ b) is
    Π_t P(|D_t| below `_above`(b)), since the block's atoms are independent given E (their F lines are).  The search starts
    at the median byte and widens while a byte's mass could still reach the best: P(byte = b) ≤ P(byte ≤ b) below it and
    ≤ P(byte ≥ b) above it."""
    bottom, top, whole, memo = block_scale(fmt, 0)[0], block_scale(fmt, 0x7F7FFFFF)[0], LINES ** len(cs), {}

    def cum(b: int) -> int:
        if b < bottom:
            return 0
        if b not in memo:
            u, v = _above(fmt, b), 1
            for y, c in zip(ys, cs):
                v *= _below(law, idle, y, c, u)
                if not v:
                    break
            memo[b] = v
        return memo[b]
    lo, hi = bottom, top
    while lo < hi:
        mid = (lo + hi) // 2
        if 2 * cum(mid) >= whole:
            hi = mid
        else:
            lo = mid + 1
    mode, best = lo, cum(lo) - cum(lo - 1)
    b = lo - 1
    while b >= bottom and cum(b) >= best:
        if cum(b) - cum(b - 1) >= best:
            mode, best = b, cum(b) - cum(b - 1)
        b -= 1
    b = lo + 1
    while b <= top and whole - cum(b - 1) > best:
        if cum(b) - cum(b - 1) > best:
            mode, best = b, cum(b) - cum(b - 1)
        b += 1
    return mode


def _p_change(fmt: Fmt, law, idle: int, y, c_word: int, byte: int) -> float:
    """p_t = 1 − max_c P(code_t = c) for one element at the block scale byte, over its F lines (values: −0 is 0)."""
    if y is None:
        return 0.0
    lo, cdf = law
    taus = _cells(fmt, byte)
    edges = [_s_to(y, t) for t in reversed(taus)] + [_s_from(y, t) - 1 for t in taus]
    cum = [_upto(cdf, lo, s) for s in edges] + [LINES]
    counts = [cum[0]] + [cum[j] - cum[j - 1] for j in range(1, len(cum))]
    if idle:
        code = e2m1(fp32.mul(c_word, recip(fmt, byte)))
        counts[bisect_left(edges, 0)] -= idle
        counts[7 - (code & 7) if code & 8 else 7 + (code & 7)] += idle
    return (LINES - max(counts)) / LINES


def row_law(fmt: Fmt, x: Sequence[int], alpha: int, ae: int, e_line: Sequence[int]) -> tuple[list[int], list[float]]:
    """F1′'s law of one row given its E line (§6.2, exact): each block's modal scale byte, and each element's p_t at it
    (before `f1_row` zeroes the witness-dead and screened ones).  The p_t are IEEE doubles, each the exact count ratio
    correctly rounded."""
    mags = [0] * 8
    for e in e_line:
        mags[e & 7] += 1
    law = _law(tuple(mags))
    idle = 16 ** mags[0] * 2 ** (R - mags[0])
    cs = [fp32.mul(alpha, w) for w in x]
    ys = [_atom_sum(c, ae) for c in cs]
    modes, p = [], []
    for b in range(0, len(x), fmt.block):
        mode = _modal_byte(fmt, law, idle, ys[b:b + fmt.block], cs[b:b + fmt.block])
        modes.append(mode)
        p.extend(_p_change(fmt, law, idle, y, c, mode) for y, c in zip(ys[b:b + fmt.block], cs[b:b + fmt.block]))
    return modes, p


def _conv(a: Sequence[float], b: Sequence[float]) -> list[float]:
    out = [0.0] * (XM + 1)
    for i, u in enumerate(a):
        for j, v in enumerate(b):
            out[min(i + j, XM)] += u * v
    return out


def _chunk(pairs: Sequence[tuple[float, float]]) -> list[float]:
    """One chunk's excess beyond 2:4: two of its 4 pairs change for free; past that, the cheapest changed pairs' changed
    elements (a pair changes 0, 1 or 2 of its codes)."""
    d = {(0, 0): 1.0}
    for a, b in pairs:
        steps = (((0, 0), (1 - a) * (1 - b)), ((1, 0), a * (1 - b) + (1 - a) * b), ((0, 1), a * b))
        nd: dict[tuple[int, int], float] = {}
        for (n1, n2), v in d.items():
            for (d1, d2), w in steps:
                nd[n1 + d1, n2 + d2] = nd.get((n1 + d1, n2 + d2), 0.0) + v * w
        d = nd
    out = [0.0] * (XM + 1)
    for (n1, n2), v in d.items():
        extra = max(0, n1 + n2 - 2)
        out[min(min(extra, n1) + 2 * max(0, extra - n1), XM)] += v
    return out


def _changed(ps: Sequence[float]) -> list[float]:
    """The number of changed codes among independent ones, truncated at XM (Poisson-binomial)."""
    d = [1.0] + [0.0] * XM
    for q in ps:
        nd = [v * (1 - q) for v in d]
        for n in range(XM):
            nd[n + 1] += d[n] * q
        nd[XM] += d[XM] * q
        d = nd
    return d


def window_saving(fmt: Fmt, p: Sequence[float]) -> float:
    """S_w over one aligned 128 of k: min(1, E[max(0, ½ − c·X_w)] + Σ_h E[max(0, ½ − c·N_h)]), X_w the excess beyond 2:4
    at the cheater's best layout inside each block (sorted p, neighbours paired, chunks of alternate pairs, per 16 sorted
    codes), N_h the changed codes in each aligned k64 half."""
    x = None
    for b in range(0, SPAN_K, fmt.block):
        s = sorted(p[b:b + fmt.block])
        pairs = list(zip(s[0::2], s[1::2]))
        block = None
        for g in range(0, len(pairs), 8):
            for chunk in (pairs[g:g + 8:2], pairs[g + 1:g + 8:2]):
                block = _chunk(chunk) if block is None else _conv(block, _chunk(chunk))
        x = block if x is None else _conv(x, block)
    total = sum(v * w for v, w in zip(x, _VX))
    for h in (p[:ATOM], p[ATOM:SPAN_K]):
        total += sum(v * w for v, w in zip(_changed(h), _VX))
    return min(1.0, total)


def f1_row(fmt: Fmt, side: Formed, i: int, e_line: Sequence[int], dead: Sequence[bool],
           screened: Sequence[bool]) -> tuple[Fraction, int, int]:
    """F1′ for row i of a formed side (§6.2): its charge per word it feeds, Σ_w 128·max(0, S_w − min(c·M_w, 1/8)) with
    p_t = 0 on witness-dead codes and screened blocks and M_w the window's blocks whose realised scale byte is not
    their modal byte; and the windows charged and the blocks off their modal byte."""
    modes, p = row_law(fmt, side.x[i], side.alpha[i], side.ae[i], e_line)
    p = [0.0 if d or screened[t // fmt.block] else q for t, (q, d) in enumerate(zip(p, dead, strict=True))]
    per, units, charged = SPAN_K // fmt.block, Fraction(0), 0
    off = [m != r for m, r in zip(modes, side.scales[i], strict=True)]
    for w in range(len(p) // SPAN_K):
        d = Fraction(window_saving(fmt, p[w * SPAN_K:(w + 1) * SPAN_K])) - min(C_PATCH * sum(off[w * per:(w + 1) * per]),
                                                                              A0_DENSE)
        if d > 0:
            units, charged = units + SPAN_K * d, charged + 1
    return units, charged, sum(off)


def c_l(m: int, k: int, n: int, block: int) -> Fraction | None:
    """F2's cost per useful MAC of the int8 route at its best depth 1 ≤ L ≤ log2(k/B) (§6.3): int8 leaves at 2 slots,
    (7/8)^L of them under Strassen–Winograd, and its pre-adds on both sides, every post-add a free merge.  None when
    k < 2B."""
    return min((2 * Fraction(7, 8) ** L + Fraction(17, 4) * (Fraction(1, m) + Fraction(1, n)) * (Fraction(7, 4) ** L - 1)
                / Fraction(3, 4) for L in range(1, (k // block).bit_length())), default=None)


def modal_share(scales: Sequence[Sequence[int]]) -> Fraction:
    """The share of blocks at the most common realised scale byte."""
    counts = Counter(s for row in scales for s in row)
    return Fraction(max(counts.values()), sum(counts.values()))


def f2_save(fmt: Fmt, shape: Shape, f_a: Fraction, f_b: Fraction) -> Fraction:
    """F2's saving per chain MAC: max(0, 1 − c_L − 4·(2 − f_A − f_B)), c_L at the tile's GEMM."""
    cl = c_l(shape.m, shape.k, shape.n, fmt.block)
    return Fraction(0) if cl is None else max(Fraction(0), 1 - cl - F2_SLOTS * (2 - f_a - f_b))


@dataclass(frozen=True)
class Debit:
    """A drawn tile's replayed debit over its credited words in FP4 units, by item (module docstring), and the counts
    behind it and behind the domain checks."""

    f1: Fraction
    f2: Fraction
    identity: int
    dead_pairs: int
    forming: Fraction
    subgrid: int
    windows: int
    counts: dict

    @property
    def total(self) -> Fraction:
        return self.f1 + self.f2 + self.identity + self.dead_pairs + self.forming + self.subgrid + self.windows

    def to_json(self) -> dict:
        return {"total": str(self.total), "f1": str(self.f1), "f2": str(self.f2), "identity": self.identity,
                "dead_pairs": self.dead_pairs, "forming": str(self.forming), "subgrid": self.subgrid,
                "windows": self.windows, "counts": dict(self.counts)}


def tile_debit(fmt: Fmt, a: Formed, b: Formed, e_a: Sequence[Sequence[int]], e_b: Sequence[Sequence[int]],
               rows: Sequence[int], cols: Sequence[int], shape: Shape) -> Debit:
    """The debit over the credited words at tile positions rows x cols of a tile's formed operands (A′ = a, B̃ = b, with
    their E lines e_a and e_b) in a unit of `shape`."""
    n = shape.n
    dead_a = {i: salt_dead(fmt, a.x[i], a.alpha[i], a.ae[i]) for i in rows}
    dead_b = {j: salt_dead(fmt, b.x[j], b.alpha[j], b.ae[j]) for j in cols}
    blocks_a, blocks_b = {i: dead_blocks(fmt, a.x[i]) for i in rows}, {j: dead_blocks(fmt, b.x[j]) for j in cols}
    f1_a = {i: f1_row(fmt, a, i, e_a[i], dead_a[i], blocks_a[i]) for i in rows}
    f1_b = {j: f1_row(fmt, b, j, e_b[j], dead_b[j], blocks_b[j]) for j in cols}
    f1 = len(cols) * sum(u for u, _, _ in f1_a.values()) + len(rows) * sum(u for u, _, _ in f1_b.values())
    f_a, f_b = modal_share([a.scales[i] for i in rows]), modal_share([b.scales[j] for j in cols])
    save = f2_save(fmt, shape, f_a, f_b)
    f2 = save * len(rows) * len(cols) * len(a.codes[0])
    win_a = {i: two_four_windows(a.codes[i]) for i in range(len(a.codes))}
    win_b = {j: two_four_windows(b.codes[j]) for j in cols}
    windows = SPAN_K // 2 * (len(cols) * sum(sum(win_a[i]) for i in rows) + len(rows) * sum(sum(win_b[j]) for j in cols))
    identity = subgrid = 0
    for i in rows:
        for j in cols:
            _, idn, sg = word_replay(fmt, a.codes[i], b.codes[j], a.scales[i], b.scales[j])
            identity, subgrid = identity + idn, subgrid + sg
    pairs = sum(p and q for i in rows for j in cols for p, q in zip(blocks_a[i], blocks_b[j]))
    unneeded = [all(b.codes[j][t] & 7 == 0 for j in cols) for t in range(len(a.codes[0]))]
    elements = sum(d or u for i in rows for d, u in zip(dead_a[i], unneeded))
    screened_only = sum(not (d or u) and blocks_a[i][t // fmt.block]
                        for i in rows for t, (d, u) in enumerate(zip(dead_a[i], unneeded)))
    spans = sum(all(win_a[i][w] for i in range(r, r + 16)) for r in range(0, len(a.codes) - 15, 16)
                for w in range(len(win_a[0])))
    counts = {"salt_dead_a": sum(sum(dead_a[i]) for i in rows), "salt_dead_b": sum(sum(dead_b[j]) for j in cols),
              "two_four_windows_a": sum(sum(win_a[i]) for i in rows), "two_four_windows_b": sum(sum(win_b[j]) for j in cols),
              "d24_spans_a": spans, "identity_atom_words": identity, "subgrid_block_words": subgrid, "dead_pairs": pairs,
              "screened_blocks_a": sum(sum(blocks_a[i]) for i in rows), "screened_blocks_b": sum(sum(blocks_b[j]) for j in cols),
              "forming_elements": elements, "screened_only_elements": screened_only, "unneeded_positions": sum(unneeded),
              "dss_spread_rows": sum(not spread_ok(fmt, a.scales[i]) for i in rows)
              + sum(not spread_ok(fmt, b.scales[j]) for j in cols),
              "f1_windows_a": sum(w for _, w, _ in f1_a.values()), "f1_windows_b": sum(w for _, w, _ in f1_b.values()),
              "off_modal_blocks_a": sum(o for _, _, o in f1_a.values()),
              "off_modal_blocks_b": sum(o for _, _, o in f1_b.values()),
              "f_a": str(f_a), "f_b": str(f_b), "f2_save": str(save), "words": len(rows) * len(cols)}
    return Debit(f1, f2, ATOM * identity, fmt.block * pairs, Fraction(FORMING_CREDITED * elements * len(cols), n),
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
                "TT_OUT-FP4(1/400) for this Π and domain (D-NF, D-24, D-SS, D-SB), as the theory lane states it, proposed and "
                "not proved: no sm_120 W1 program outputs correct C̃ and U words of the units whose replayed debit is under "
                "the cap for less than their credit less the debit (research store "
                "internal/pouw/rtx-pro/theory-pearl-c4-domain.md §2)",
                f"sm_120's mma.sync kind::mxf4nvf4 and BF16 m16n8k16 are reproduced exactly ({self.fmt.atom.name}, "
                "blackwell_sm120_nvf4_m16n8k64 for the noise atom, hopper_bf16_m16n8k16), pinned on the RTX 5090",
                "the E2M1, UE4M3 and UE8M0 casts are as modelled (from the PTX ISA, not captured)",
            ),
            allowances=(
                "the noisy block amax (17 units per element) and ρ over every 8th position (2) are in W_ref, not credited",
                "savings no replay reaches (cross-atom cancellation, and theory §6.5's open routes: gathering across "
                "blocks on operands flat on both sides, layouts inside a block other than sorted pairing, regrouping "
                "after the salt) are TT_OUT-FP4's; the replayed debit charges the rest (tile_debit, F1′ and F2 included)",
                "the unscale is glue outside the checked values",
            ),
            parameters=(("k_min", k_min), ("tile_m", tile[0]), ("tile_n", tile[1]), ("rank", R), ("block", self.fmt.block),
                        ("delta_inv", 4), ("dead_rho_sq", DEAD_RHO_SQ)),
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
        lab = lambda m, r: keyed(blake3(LABELS[r]), m)  # noqa: E731
        return C.Noise(lab(ctx.salt + ctx.root_a + ctx.root_b + ctx.matmul.index.to_bytes(8, "little"), "seed-A"),
                       lab(ctx.salt + ctx.root_b, "seed-B"))

    def parse_row(self, row: bytes, k: int) -> list[int]:
        return parse_row(row, k)

    def row_passes(self, words: Sequence[int]) -> bool:
        return row_passes(self.fmt, words)

    def tile_cap(self, record, ops: C.Ops, rows: Sequence[int], cols: Sequence[int]) -> tuple[Fraction, Fraction]:
        """(debit, cap) over the credited words at tile positions rows x cols: `tile_debit`'s total, and ρ times the
        credited cells' share of creditOf (each row's `credit_of` at |K|, over n).  The breakdown stays in `last_debit`."""
        s = record.matmul.shape
        self.last_debit = tile_debit(self.fmt, ops.a, ops.b, ops.e_a, ops.e_b, rows, cols, s)
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


__all__ = ["FORMATS", "MXFP4", "NAMES", "NVFP4", "Debit", "PearlC4", "c_l", "clean", "e2m1", "f1_row", "f2_save", "form",
           "hw_scales", "line", "modal_share", "recip", "row_law", "row_passes", "salt_dead", "scale_bytes_ok", "tile_debit",
           "window_saving", "word_replay"]
