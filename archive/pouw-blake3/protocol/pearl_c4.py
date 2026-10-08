"""``pearl-c-nvfp4-v0``: Pearl-C on sm_120's block-scaled FP4 (``mma.sync kind::mxf4nvf4``), NVFP4 only.  MXFP4
(``pearl-c-mxfp4-v0``) stays here as a contrast line, outside the claimed domain (Daniel, 15:34Z): same forming and
chain, UE8M0 scales per 32, no certificate claims and not registered in `schemes.get`.

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
          (-h3: row i's E_A is line 0 of its own seed, `row_seed` over the row's leaf in A's tree; the tickets keep seed_A)
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
§1, §2; TT_OUT-FP4, ``tt-out/fp4-sm120``, proposed).  Admission is per row, salt-free (`row_passes`): D-NF (`dnf`: 0 ≤ β
and a_E's E4M3 byte v in 0x08–0x7E, a normal UE4M3 with bit 7 clear and not NaN; one a_E per row).  A block is
screened when (clean amax_b)² > 90·ρ² (9.5ρ, `dead_blocks`); its row is admitted
and its salt-dead codes are charged per element (items 3 and 4), so a row with many screened blocks is rejected only
when its tile's debit passes the cap (server.md 10:35Z; the theory lane's 10:50Z ruling).  Items 3 and 4 charge the
reference's witness alone, not every element of a screened block (server.md 11:32Z; the assessor's 17:25Z B for item 3
per element): F1′ charges a screened block's near-certain codes at their own p_t.  A drawn tile's debit is replayed
over its credited words (`tile_debit`):

~~~text
1  F1′ split     per row and aligned 128 of k: 128·S_w, κ = min(c·M_w, 0.125) inside S_w's expectation, on every word
                the row feeds (below)
1b F2 int8      per tile: max(0, 1 − c_L − 4·(2 − f_A − f_B)) per chain MAC, f the realised scale bytes' flatness (below)
2  identity      per atom whose lone skip leaves the word unchanged: 64 per atom-word
3  dead pairs    per word and position t whose A code and B code are both salt-dead (`salt_dead`, each side): 1
4  forming       per salt-dead (the reference's witness, `salt_dead`) or unneeded A element: f_s · |cols|/n
5  sub-grid      per (word, block) with 2304·m_a·m_b·2^(e_a+e_b−2) below its step's grid (it adds exactly 0): `block`
D-24 windows     per row and aligned 128 of k whose every 8-chunk has ≤ 2 of 4 pairs nonzero: 0.5 per MAC, on every word it feeds
cap              debit ≤ ρ · the credited cells' share of creditOf, ρ = 1/400
~~~

**F1′ and F2** (theory §6.2–6.3, the fix for the assessor's D on ``tt-out/fp4-sm120``, fp4-base-split-break.md; the
assessor's FP4 conditions, ratings.md 15:18Z).  F1′ charges the salt-free base split A′ = A0 + ΔA its expected saving
under the exact noise law given this salt's E lines (`noise_law`): p_t = P(code ≠ modal code) at the block's modal scale
byte (`change_probabilities`); per window, min(1, E[max(0, R_24 − κ)] + E[max(0, R_skip − κ)]) with R_24 one 2:4 pass
plus a CUDA-core patch of the excess, R_skip a skip or patch per k64 atom, at the cheater's best layout inside each block,
and κ inside the expectation, A0's scale corrections: one FFMA per word for each of the M_w blocks whose realised byte
is not their modal byte (`split_saving`, `split_units`).  F2 charges the masked int8-Strassen route on the tile's realised scale bytes, read at
one tile byte, per row, and per block-column class (`int8_saving`), with c_L the fast-matmul catalogue's minimum at its
lower-bound add count, pinned per shape (`int8_depth_cost`, `pearl_c4_c_L`; the assessor's 20:00Z ruling).
They are the rows `base-split-debit/fp4-sm120` (F1′ charges the base split's saving) and `int8-route-debit/nvfp4` (F2
charges the masked int8 route's saving per drawn tile), which replace `no-base-split/fp4-sm120` and the retired domain
claim `scale-flatness/nvfp4`.  A tile whose debit passes the cap fails the tile rule outright (`pearl_c_work.check_opened`),
and D-SK stays.  **R1** (theory §6.8): a row whose own F1′ per word passes ρ·(its cell credit) is uncredited, declared by
the prover (`r1_rows`, through `pearl_c_work.declare`) and checked on drawn tiles (`row_over_debit`, through
`pearl_c_work.check_opened`); it is still formed, checked and hashed.

A code is *salt-dead* unless two draws in the noise support give it two values (`salt_dead`), so the forming debit is on
codes, not on the atom's word.  F1′ takes p_t = 0 on salt-dead codes.  D-24 is charged per row window rather than per 16-row span (a cheaper computation may
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

import itertools
import math
import struct
from bisect import bisect_left
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from verity.ml.crypto.blake3 import blake3, keyed
from verity.ml.crypto.turboshake import turboshake128
from verity.errors import InvalidArtifact
from verity.ml.circuits.silicon import fp32
from verity.ml.circuits.silicon.models import BLACKWELL_SM120_MXF4, BLACKWELL_SM120_NVF4, HOPPER_BF16_M16N8K16, BlockScaledAlignAdd

from ..audit import ROW_SCHEMA, activation_binding, rows_domain
from ..protocol import Certificate, Context, DomainError, Matmul, Shape, Tile, grid_tiles
from . import pearl_c as C
from . import pearl_c4_c_L as CL
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
#: F1′ and F2's prices in FP4 units: a CUDA-core FFMA 16.92 (the measured 8.46 FP8 units x 2), the dense correction of
#: A0's block scales at least 0.125 per MAC (an FP8 GEMM over k/16), the int8 route's patch 4 slots per useful MAC per side,
#: and its int8 leaf at W1's issue ratio, 2 slots per MAC (the assessor's FP4 conditions, ratings.md 15:18Z; the
#: wall-time window below it, 1.81, is `concurrent-budgets/sm120`'s, not the rule's).
FFMA_UNITS, DENSE_CORRECTION, INT8_PATCH_SLOTS, INT8_LEAF_SLOTS = 16.92, 0.125, 4, 2

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
#: the hashing formats Pearl-C4 implements (`serving.HASH_FORMATS`): h0, and h3 (h2's trees and digests, E_A per row)
HASHINGS = ("h0", "h3")


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


def unit_seed_a(salt: bytes, root_a: bytes, root_b: bytes, index: int) -> bytes:
    """A unit's seed_A: keyed BLAKE3 under blake3("pearl-c4/v0/seed-A") of salt ‖ root_A ‖ root_B ‖ index u64 LE."""
    return keyed(blake3(LABELS["seed-A"]), salt + root_a + root_b + index.to_bytes(8, "little"))


def row_seed(salt: bytes, leaf: bytes, root_b: bytes, unit: int, row: int) -> bytes:
    """-h3's seed of row `row` of unit `unit`, `pearl_c.row_seed`'s layout under Pearl-C4's seed-A label: the one-row
    unit's seed_A over the row's own leaf in A's tree, at index unit · 2^32 + row, the unit and the row each below 2^32."""
    if not (0 <= unit < 1 << 32 and 0 <= row < 1 << 32):
        raise ValueError(f"row_seed: unit {unit} and row {row} must each be below 2^32")
    return unit_seed_a(salt, leaf, root_b, unit << 32 | row)


@lru_cache(maxsize=16)
def seed_b(salt: bytes, root_b: bytes) -> bytes:
    """B̃'s noise seed, keyed by this job's salt and B's root.  So B̃ exists only once the salt does, and no program
    can precompute with it: the salt-free base split is two-sided (A′·B̃ᵀ = A0·B0ᵀ + ΔA·B̃ᵀ + A0·ΔBᵀ, the cheater
    paying both corrections), which is what ties it on the stride rows (TT_OUT-FP4's rule D-SK,
    theory-pearl-c4-domain.md §2)."""
    return keyed(blake3(LABELS["seed-B"]), salt + root_b)


@lru_cache(maxsize=4)
def basis(seed_b: bytes, role: int, k: int) -> tuple[tuple[int, ...], ...]:
    """F_A (ROLE_FA) or F_B (ROLE_FB): k lines of 32 codes, line t for position t.  Cached: one costs about three rows'
    F1′ to derive at k = 4,096, and `r1_row` forms one row at a time."""
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
    scales through it, though the domain (`scale_bytes_ok`) never lets bit 7 reach one.  It is the decode the Lean
    `scaleDyadic` proposal pins, whose `nvf4Specials` table has 28 rows (research store
    ``internal/pouw-fp8/pearl-c4-scale-decode/``, bc-5a715b19's review).  Four of them are NaN accumulators whose scale
    bytes set bit 7: valid scales under this decode, which the pinned model (`verity.ml.circuits.silicon.models._scale_dyadic`) now
    shares after GPU 4 replayed all 17 bit-7 vectors on the card (r20260930-161935-83a8), and outside the model through
    the accumulator (the card writes 0x7FFFFFFF)."""
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
    it covers every salt-dead element (Lean `saltDead4_screened`), so items 3 and 4, on the witness (`salt_dead`), never
    charge more than their block forms; the debit counts co-screened block pairs (`screened_block_pairs`) but charges
    none, instead of rejecting the rows."""
    rho_sq = _exact_rho_sq(x)
    return [Fraction(P.f32_of_bits(max(w & 0x7FFFFFFF for w in x[b:b + fmt.block]))) ** 2 > DEAD_RHO_SQ * rho_sq
            for b in range(0, len(x), fmt.block)]


def dnf(beta: int) -> bool:
    """D-NF on a row's β_row word: 0 ≤ β, and a_E's E4M3 byte v, the cast with its sign, has 8 ≤ v < 0x7F (a normal
    UE4M3, at least 2^-6, and not NaN): the bytes 0x08–0x7E, Lean's.  The pinned `stats` never makes β negative or
    non-finite, but the rule states it: a negative β casts to a byte with bit 7 set, which a check of 8 ≤ v alone admits
    and sm_120 reads from its low 7 bits, so at β = −1/4096 the byte 0x80 would be a zero scale and the row's noise zero
    (bc-824e54a2's post-M4 rebuild); a NaN β with its sign clear casts to 0x7F (bc-22298e90's review)."""
    try:
        v = P.f32_to_fp8(P.f32_of_bits(beta))
    except P.Reject:
        return False
    return not beta >> 31 and UE4M3_NORMAL <= v < 0x7F


def row_passes(fmt: Fmt, x: Sequence[int]) -> bool:
    """Admission, per row and salt-free: every word a finite widened BF16 word (its low 16 bits zero, as `parse_row`
    writes them; Lean's `Fp4Sem.RowOK`), and D-NF (`dnf`).  A row's screened blocks are charged in its tiles' debit
    (`tile_debit`), and the tile's cap rejects them."""
    if any(w & 0xFFFF or (w >> 23) & 0xFF == 0xFF for w in x):
        return False
    try:
        beta = stats(fmt, x)[3]
    except P.Reject:
        return False
    return dnf(beta)


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


def _scale_value(fmt: Fmt, byte: int) -> float:
    m, e = scale_dyadic(fmt, byte)
    return m * 2.0 ** e


def noise_sigma(e_row: Sequence[int], ae: int) -> float:
    """The noise's standard deviation in the atom's C units given this salt's E line: a_E·2^-4·‖E‖·√8.5625 (F uniform)."""
    return _scale_value(NVFP4, ae) / 16 * math.sqrt(8.5625 * sum(float(e2m1_value(c)) ** 2 for c in e_row))


_F2 = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)    # 2·value of each E2M1 code, in half units


def _pmf_conv(a: tuple[int, tuple[float, ...]], b: tuple[int, tuple[float, ...]]) -> tuple[int, tuple[float, ...]]:
    (la, pa), (lb, pb) = a, b
    nz = [(j, y) for j, y in enumerate(pb) if y]
    out = [0.0] * (len(pa) + len(pb) - 1)
    for i, x in enumerate(pa):
        if x:
            for j, y in nz:
                out[i + j] += x * y
    return la + lb, tuple(out)


@lru_cache(maxsize=64)
def _f_power(m: int) -> tuple[int, tuple[float, ...]]:
    """The law of the sum of m independent uniform draws of 2·F (F one E2M1 code), on the integers from its lowest value."""
    if m == 0:
        return 0, (1.0,)
    if m == 1:
        pmf = [0.0] * 25
        for v in _F2:
            pmf[v + 12] += 1 / 16
        return -12, tuple(pmf)
    return _pmf_conv(_f_power(m // 2), _f_power(m - m // 2))


@lru_cache(maxsize=512)
def noise_law(e_row: tuple[int, ...]) -> tuple[int, tuple[float, ...]]:
    """The exact law of n = Σ_r (2E_r)·(2F_r) given this salt's E line, F_r uniform over the 16 E2M1 codes and independent
    (one element's noise is a_E·2^-6·n): (lowest value, pmf over the consecutive integers from it).  F is symmetric, so
    only |E_r| matters, and each magnitude's m-fold sum is a stretched power of one draw's law."""
    law: tuple[int, tuple[float, ...]] = (0, (1.0,))
    for v in sorted({abs(_F2[c]) for c in e_row} - {0}):
        lo, pmf = _f_power(sum(1 for c in e_row if abs(_F2[c]) == v))
        stretched = [0.0] * ((len(pmf) - 1) * v + 1)
        stretched[::v] = pmf
        law = _pmf_conv(law, (lo * v, tuple(stretched)))
    return law


@lru_cache(maxsize=512)
def _noise_cdf(e_row: tuple[int, ...]) -> tuple[int, tuple[float, ...]]:
    lo, pmf = noise_law(e_row)
    return lo, (0.0, *itertools.accumulate(pmf))


def _below_law(e_row: tuple[int, ...], scale: float) -> Callable[[float], float]:
    """y ↦ P(noise < y) for noise = scale·n, n ~ `noise_law(e_row)`."""
    lo, cum = _noise_cdf(e_row)

    def f(y: float) -> float:
        i = math.ceil(y / scale) - lo if math.isfinite(y) else (len(cum) if y > 0 else 0)
        return cum[min(max(i, 0), len(cum) - 1)]
    return f


def _below(v: Sequence[float], cdf: Callable[[float], float], u: float) -> float:
    """P(every |v_t + noise| < u), the noise independent per element given E."""
    p = 1.0
    for x in v:
        p *= max(0.0, cdf(u - x) - cdf(-u - x))
    return p


def modal_byte(fmt: Fmt, v: Sequence[float], cdf: Callable[[float], float], sigma: float) -> int:
    """A block's most likely noisy scale byte under the law given E (salt-free given E): the byte whose amax interval
    [6·lower midpoint, 6·upper midpoint) holds the noisy amax most often.  Among the bytes the clean amax ± 6σ reaches."""
    top = max(abs(x) for x in v)
    lo_b = block_scale(fmt, P.f32_bits(max(top - 6 * sigma, 0.0)))[0]
    hi_b = block_scale(fmt, P.f32_bits(top + 6 * sigma))[0]
    if fmt is MXFP4:
        edges = {s: (3 * _scale_value(fmt, s), 6 * _scale_value(fmt, s)) for s in range(lo_b, hi_b + 1)}
    else:
        def mid(s, t):
            return 3 * (_scale_value(fmt, s) + _scale_value(fmt, t))
        edges = {s: (mid(s - 1, s) if s > UE4M3_MIN else 0.0, mid(s, s + 1) if s < UE4M3_TOP else math.inf)
                 for s in range(lo_b, hi_b + 1)}
    return max(edges, key=lambda s: (_below(v, cdf, edges[s][1]) - _below(v, cdf, edges[s][0]), -s))


def change_probabilities_exact(fmt: Fmt, x: Sequence[int], alpha: int, e_row: Sequence[int], ae: int,
                               dead: Sequence[bool]) -> tuple[list[float], list[int]]:
    """F1′'s p_t as first written (the reading C bounds): P(code ≠ modal code) over all 15 cells, at the modal byte by the
    product formula over the bytes the clean amax ± 6σ reaches.  Kept to check that `change_probabilities` charges at
    least it; the rule is `change_probabilities`."""
    e = tuple(e_row)
    cdf = _below_law(e, _scale_value(NVFP4, ae) / 64)
    sigma = noise_sigma(e_row, ae)
    p, modes = [], []
    for b in range(0, len(x), fmt.block):
        v = [P.f32_of_bits(fp32.mul(alpha, w)) for w in x[b:b + fmt.block]]
        mode = modal_byte(fmt, v, cdf, sigma)
        modes.append(mode)
        vs = _scale_value(fmt, mode)
        edges = [-m * vs for m in reversed(_E2M1_MIDPOINTS)] + [m * vs for m in _E2M1_MIDPOINTS]
        for t, vt in enumerate(v):
            if dead[b + t]:
                p.append(0.0)
                continue
            c = [0.0] + [cdf(e - vt) for e in edges] + [1.0]
            p.append(1.0 - max(c[i + 1] - c[i] for i in range(len(c) - 1)))
    return p, modes


#: The float margin on every comparison C's evaluation decides (the pmf's error is about 10^-16), and the slack its
#: lower bound on p_t gives up; both on the charging side.
F1_EPS = 1e-12
#: R1's tolerance band (the assessor's condition 3, 16:10Z): a row within this relative distance of its bound may be
#: declared either way, and a tile's debit is compared with its cap at the same margin.  Soundness cost ≤ 10^-9·ρ.
R1_TOLERANCE = 1e-9
#: B̃'s overfit factor, measured, not derived (theory-pearl-c4-domain.md §6.12; research store
#: cheap-binding/pearlc4-fix/btilde_mirror.py, btilde_honest.py): a layout chosen after the salt for B̃'s residual is
#: shared by all n weight rows and re-lays no A′; its hill-climb needs 6.8× F1′'s expectation on the worst family.  B̃'s
#: F1′ is charged at 10× that expectation, per window, capped at the window's MACs; A′'s at 1× (its layout must re-gather
#: B̃, which doesn't pay at n ≤ 2^18).  A stronger search pays past 10× at n ≤ 1,024, which `B_OVF` covers.
F1_B_OVERFIT = 10
#: B-OVF, measured, not derived (theory-pearl-c4-domain.md §6.12b; the assessor's B, ratings.md 18:45Z): a unit's
#: creditOf is discounted by β(n_b)·mnk, n_b the largest tabled n ≤ its n.  β is the worst net saving of a B̃ layout
#: annealed and block-optimised after the salt, over the families R1 admits at `F1_B_OVERFIT`, after A0's gather and ρ,
#: with a 1.5× margin on the measured gain and made non-increasing in n.  It is taken on each bucket's own R1 admission
#: boundary at k = 1,024 (``k_min``), where the cap is largest, and just inside it (the worst family can sit there): the larger of
#: that need, the assessor's boundary check and the earlier grid's cells under the same cap, never below the grid's
#: table (condition 7; research store art:cb8395f9d67e84c354611d00bd2b6bacd490db25b535cbfb1f12490d3f6653ca,
#: beta_final.py; the grid's beta_table.py on strong.jsonl, strong_ext.jsonl).  The fork (tt-out-aw/pearl-c-nvfp4-sm120) takes the same discount.
B_OVF = ((128, Fraction(414, 10_000)), (256, Fraction(223, 10_000)), (512, Fraction(119, 10_000)),
         (1024, Fraction(50, 10_000)), (2048, Fraction(18, 10_000)), (4096, Fraction(0)))
#: The claimed domain's least n: n = 64 (an MQA unit of one 64-wide KV head) would need 5.7% and is not claimed.
B_OVF_MIN_N = B_OVF[0][0]


def b_ovf(n: int) -> Fraction:
    """B-OVF's β for a unit of n weight rows: the tabled value at the largest tabled n ≤ n.  Below the table (only a test
    instance's n: the claimed domain has n ≥ `B_OVF_MIN_N`, `PearlC4.check`) the first bucket's."""
    return next((b for nb, b in reversed(B_OVF) if nb <= n), B_OVF[0][1])


#: Registered weights' keyed transforms (`check_registration`; Daniel, 1:36 PM and 5:52 PM PDT on 30 Sep, research store
#: internal/pouw/rtx-pro/server.md and internal/pouw/keyed-transforms/): the keyed 8-block rotation (rung 3) is the rule,
#: and the keyed V/O rotation (each KV head's v_proj rows and o_proj columns) plus the keyed head interleave (o_proj's
#: heads) are adopted inside it, after it.
ROTATION_8 = "keyed-8-block-rotation"
VO_ROTATION = "keyed-vo-rotation"
HEAD_INTERLEAVE = "keyed-head-interleave"
#: Every RMSNorm scale γ folded into the linears it feeds, with no rotation (the `fold` rung, research store
#: docs/pouw/keyed-transforms.md §14).  The rotation folds γ itself (rung 3), so a rotated stack never declares it.
GAMMA_FOLD = "gamma-fold"
#: The transform stacks a registration may apply, in their order
REGISTRATION_STACKS = ((ROTATION_8,), (ROTATION_8, VO_ROTATION, HEAD_INTERLEAVE))
#: The stacks of an unrotated checkpoint, which must be curated
UNROTATED_STACKS = ((), (GAMMA_FOLD,))


def check_registration(transforms: Sequence[str], curated: bool = False, needs_gamma_fold: bool = False) -> None:
    """Refuse a registration of a checkpoint's weights outside Pearl-C4's rules.  `transforms` are the transforms it
    applies, in order; `curated` says the checkpoint is on the curated list, the fallback for an unrotated one;
    `needs_gamma_fold` says it registers only with γ folded, as Llama-3.1-70B does: the cast moves its layer 0's update
    by 158% unfolded, and by under 4% with γ folded (keyed-transforms.md §14).

    The keyed 8-block rotation is the registered-weights rule, alone or followed by the V/O rotation and the head
    interleave together (the stack measured and adopted).  V/O or the interleave on an unrotated checkpoint is refused,
    curated or not.  An unrotated checkpoint that needs γ folded declares `GAMMA_FOLD`.  This reads what the
    registration declares; applying the transforms and keying them are the registration's own."""
    t = tuple(transforms)
    if t in REGISTRATION_STACKS:
        return
    if ROTATION_8 not in t and {VO_ROTATION, HEAD_INTERLEAVE} & set(t):
        raise DomainError(f"registration {t}: the V/O rotation and the head interleave apply inside the keyed 8-block "
                          "rotation only, never on an unrotated checkpoint")
    if t not in UNROTATED_STACKS:
        raise DomainError(f"registration {t}: not an adopted stack, one of {REGISTRATION_STACKS + UNROTATED_STACKS}")
    if not curated:
        raise DomainError("registration: an unrotated checkpoint must be on the curated list")
    if needs_gamma_fold and GAMMA_FOLD not in t:
        raise DomainError(f"registration {t}: this checkpoint registers only with γ folded, by the keyed 8-block "
                          f"rotation or by {GAMMA_FOLD!r}")


def _byte_edges(fmt: Fmt, s: int) -> tuple[float, float]:
    """The noisy amax interval [lo, hi) whose block takes scale byte s."""
    if fmt is MXFP4:
        return 3 * _scale_value(fmt, s), 6 * _scale_value(fmt, s)
    lo = 3 * (_scale_value(fmt, s - 1) + _scale_value(fmt, s)) if s > UE4M3_MIN else 0.0
    hi = 3 * (_scale_value(fmt, s) + _scale_value(fmt, s + 1)) if s < UE4M3_TOP else math.inf
    return lo, hi


def certified_modal_byte(fmt: Fmt, v: Sequence[float], cdf: Callable[[float], float], realised: int) -> int | None:
    """A block's modal scale byte by a certified local search around its realised byte r (bc-f5bf55c8's C,
    pearl-c4-fix.md §7.2; the assessor's B, ratings.md 16:10Z): from the product formula at the edges of bytes
    r − w … r + w, the byte probabilities there and the mass outside them.  The largest certifies when it beats the
    outside mass and the runner-up by more than `F1_EPS` (no other byte can then win, and no float near-tie decides).
    w = 1, then 4, then 16; None if still uncertified (the caller charges the block on the safe side)."""
    top = UE4M3_TOP if fmt is NVFP4 else 254
    for w in (1, 4, 16):
        lo_b, hi_b = max(realised - w, UE4M3_MIN if fmt is NVFP4 else 0), min(realised + w, top)
        edges = [_byte_edges(fmt, s)[0] for s in range(lo_b, hi_b + 1)] + [_byte_edges(fmt, hi_b)[1]]
        below = [1.0 if math.isinf(u) else _below(v, cdf, u) for u in edges]
        probs = [below[j + 1] - below[j] for j in range(len(edges) - 1)]
        outside = below[0] + 1.0 - below[-1]
        order = sorted(range(len(probs)), key=lambda j: (-probs[j], j))
        best = probs[order[0]]
        runner = max(outside, probs[order[1]] if len(order) > 1 else 0.0)
        if best > runner + F1_EPS:
            return lo_b + order[0]
    return None


def change_probabilities(fmt: Fmt, x: Sequence[int], alpha: int, e_row: Sequence[int], ae: int,
                         dead: Sequence[bool], scales: Sequence[int]) -> tuple[list[float], list[int | None]]:
    """F1′'s p_t, C's evaluation, normative for prover and verifier alike (the assessor's condition 1, 16:10Z), under
    the exact noise law given this salt's E line (`noise_law`):
    - each block's modal byte certified around its realised byte (`certified_modal_byte`); a block left uncertified is
      charged on the safe side, p = 0 on its codes and no κ relief (its mode is None);
    - p_t from 4 lookups, a lower bound: the element's own cell and its two neighbours exactly, each tail beyond them
      as one cell, so 1 − max(three cells, two tails) − `F1_EPS` ≤ p_t; S_w only grows as any p_t falls;
    - 0 on salt-dead codes.
    It charges at least `change_probabilities_exact` in every window.  The atom's one RZ rounding of C + noise, 2^-23
    relative, is not modelled."""
    cdf = _below_law(tuple(e_row), _scale_value(NVFP4, ae) / 64)
    hw = hw_scales(fmt, scales)
    p: list[float] = []
    modes: list[int | None] = []
    for bi, b in enumerate(range(0, len(x), fmt.block)):
        v = [P.f32_of_bits(fp32.mul(alpha, w)) for w in x[b:b + fmt.block]]
        mode = certified_modal_byte(fmt, v, cdf, hw[bi])
        modes.append(mode)
        if mode is None:
            p.extend([0.0] * len(v))
            continue
        vs = _scale_value(fmt, mode)
        edges = [-math.inf] + [-m * vs for m in reversed(_E2M1_MIDPOINTS)] + [m * vs for m in _E2M1_MIDPOINTS] + [math.inf]
        for t, vt in enumerate(v):
            if dead[b + t]:
                p.append(0.0)
                continue
            c = sum(1 for e in edges[1:-1] if e <= vt)              # own cell, 0 … 14: [edges[c], edges[c + 1])
            f = [_edge_cdf(cdf, edges[j], vt) for j in (max(c - 1, 0), c, c + 1, min(c + 2, 15))]
            cells = (f[1] - f[0], f[2] - f[1], f[3] - f[2], f[0], 1.0 - f[3])
            p.append(max(0.0, 1.0 - max(cells) - F1_EPS))
    return p, modes


def _edge_cdf(cdf: Callable[[float], float], edge: float, v: float) -> float:
    return 0.0 if edge == -math.inf else 1.0 if edge == math.inf else cdf(edge - v)


_EXCESS_MAX = 4                    # 0.5 − c·x < 0 from x = 4 on


def _pair_excess(js: tuple[int, ...]) -> int:
    dev = sorted(j for j in js if j)
    return min(sum(dev[:len(dev) - 2]) if len(dev) > 2 else 0, _EXCESS_MAX)


_COMBOS = [(js, _pair_excess(js)) for js in itertools.product(range(3), repeat=4)]


def _conv(a: Sequence[float], b: Sequence[float]) -> list[float]:
    out = [0.0] * (_EXCESS_MAX + 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[min(i + j, _EXCESS_MAX)] += x * y
    return out


def _chunk(pairs: Sequence[tuple[float, float]]) -> list[float]:
    """One 8-chunk's excess (changed elements beyond 2:4, patched): if m > 2 of its 4 pairs change, the m − 2 pairs with
    the fewest changed elements."""
    pj = [((1 - a) * (1 - b), a * (1 - b) + (1 - a) * b, a * b) for a, b in pairs]
    out = [0.0] * (_EXCESS_MAX + 1)
    for js, x in _COMBOS:
        out[x] += pj[0][js[0]] * pj[1][js[1]] * pj[2][js[2]] * pj[3][js[3]]
    return out


def _block_excess(p: Sequence[float]) -> list[float]:
    """A block's excess at the cheater's layout: the block keeps its scale under any reordering, so its codes are sorted
    by change probability, neighbours paired (reliable with reliable), and the pairs dealt round-robin to its chunks."""
    s = sorted(p)
    pairs = [(s[i], s[i + 1]) for i in range(0, len(s), 2)]
    n = len(s) // 8
    d = [1.0] + [0.0] * _EXCESS_MAX
    for c in range(n):
        d = _conv(d, _chunk(pairs[c::n]))
    return d


def _changed(p: Sequence[float]) -> list[float]:
    """Changed codes among p, truncated at _EXCESS_MAX (Poisson-binomial)."""
    d = [1.0] + [0.0] * _EXCESS_MAX
    for q in p:
        nd = [x * (1 - q) for x in d]
        for i in range(_EXCESS_MAX):
            nd[i + 1] += d[i] * q
        nd[_EXCESS_MAX] += d[_EXCESS_MAX] * q
        d = nd
    return d


def split_saving(fmt: Fmt, p: Sequence[float], kappa: float = 0.0) -> float:
    """F1′'s per-MAC charge on one aligned 128-window, with A0's scale correction κ inside the expectation (a cheater who
    sees the salt routes each row after it):  min(1, E[max(0, R_24 − κ)] + E[max(0, R_skip − κ)]), where
    R_24 = max(0, ½ − c·X) is one 2:4 pass plus a CUDA-core patch of the excess X, and R_skip = Σ_h max(0, ½ − c·N_h)
    skips or patches each aligned k64 atom.  The sum bounds E[max(0, max(R_24, R_skip) − κ)] on any law."""
    c = FFMA_UNITS / SPAN_K
    gain = [max(0.0, 0.5 - c * x) for x in range(_EXCESS_MAX + 1)]
    d = [1.0] + [0.0] * _EXCESS_MAX
    for b in range(0, SPAN_K, fmt.block):
        d = _conv(d, _block_excess(p[b:b + fmt.block]))
    s = sum(x * max(0.0, g - kappa) for x, g in zip(d, gain))
    h0, h1 = _changed(p[:ATOM]), _changed(p[ATOM:SPAN_K])
    s += sum(x * y * max(0.0, gain[i] + gain[j] - kappa) for i, x in enumerate(h0) for j, y in enumerate(h1))
    return min(1.0, s)


def split_units(fmt: Fmt, p: Sequence[float], modes: Sequence[int | None], scales: Sequence[int],
                overfit: float = 1) -> Fraction:
    """F1′ per word the row feeds, in FP4 units: 128·`split_saving` per window at κ = min(c·M_w, 0.125), M_w the window's
    blocks whose realised scale byte is not their modal byte (each costs A0's part one correction FFMA per word; a dense
    correction GEMM over k/16 in FP8 costs at least 0.125 per MAC)."""
    c, per = FFMA_UNITS / SPAN_K, SPAN_K // fmt.block
    hw = hw_scales(fmt, scales)
    total = 0.0
    for w in range(0, len(p) - len(p) % SPAN_K, SPAN_K):
        b0 = w // fmt.block
        moved = sum(modes[b] is not None and hw[b] != modes[b] for b in range(b0, b0 + per))
        total += SPAN_K * min(1.0, overfit * split_saving(fmt, p[w:w + SPAN_K], min(c * moved, DENSE_CORRECTION)))
    return Fraction(total)


def int8_depth_cost(fmt: Fmt, m: int, k: int, n: int) -> float:
    """F2's c_L: the masked int8 route's least cost per honest MAC before its patch over the fast-matmul catalogue, at
    each scheme's lower-bound add count (`pearl_c4_c_L`; the assessor's 20:00Z ruling), read at the smallest tabled m, k
    and n at least the shape's.  c_L falls as any of them grows, so a shape between table points takes the lower c_L.
    A route's depth is capped by k/block, so a 32-block format reads the table at half its k.  The claimed domain is
    inside the table (`PearlC4.check`); a shape beyond it raises."""
    kk = k * NVFP4.block // fmt.block

    def at(grid: tuple[int, ...], v: int) -> int:
        i = bisect_left(grid, v)
        if i == len(grid):
            raise ValueError(f"c_L: {v} is past the pinned table's {grid[-1]}")
        return i

    return CL.C_L[at(CL.MS, m)][at(CL.KS, max(kk, 1))][at(CL.NS, n)] / 10_000


def _share(rows: Sequence[Sequence[int]]) -> float:
    cells = [s for r in rows for s in r]
    return max(cells.count(s) for s in set(cells)) / len(cells) if cells else 0.0


def _row_share(rows: Sequence[Sequence[int]]) -> float:
    """The mean over rows of each row's own modal share: flatness up to per-row scale factors, which the route absorbs."""
    return sum(_share([r]) for r in rows) / len(rows) if rows else 0.0


def _labels(rows: Sequence[Sequence[int]]) -> list[int]:
    """Per block column, its modal byte offset from each row's modal byte."""
    modes = [max(set(r), key=lambda s: (r.count(s), -s)) for r in rows]
    out = []
    for b in range(len(rows[0])):
        offs = [r[b] - m for r, m in zip(rows, modes)]
        out.append(max(set(offs), key=lambda o: (offs.count(o), -o)))
    return out


def int8_saving(fmt: Fmt, a_scales: Sequence[Sequence[int]], b_scales: Sequence[Sequence[int]], m: int, n: int) -> float:
    """F2 per chain MAC (theory-pearl-c4-domain.md §6.3): the masked int8 route's saving max(0, 1 − c_L − 4·(2 − f_A −
    f_B)) on the tile's realised scale bytes, at the largest of three readings of flatness: the tile's modal byte; each
    row's own modal byte (row factors); and per block-column class (the class of a column is its modal byte offset from
    each row's modal byte, on A′ and B̃ together), each class a sub-GEMM over its own k with row factors free."""
    a = [hw_scales(fmt, s) for s in a_scales]
    b = [hw_scales(fmt, s) for s in b_scales]
    nb = len(a[0])
    k = nb * fmt.block

    def save(fa: float, fb: float, kk: int) -> float:
        return max(0.0, 1 - int8_depth_cost(fmt, m, kk, n) - INT8_PATCH_SLOTS * (2 - fa - fb))

    best = max(save(_share(a), _share(b), k), save(_row_share(a), _row_share(b), k))
    classes: dict[tuple[int, int], list[int]] = {}
    for col, key in enumerate(zip(_labels(a), _labels(b))):
        classes.setdefault(key, []).append(col)
    split = sum(len(cols) / nb * save(_row_share([[r[c] for c in cols] for r in a]),
                                      _row_share([[r[c] for c in cols] for r in b]), len(cols) * fmt.block)
                for cols in classes.values())
    return max(best, split)


def two_four_windows(codes: Sequence[int]) -> list[bool]:
    """D-24, per aligned 128 of k: every aligned 8-chunk has at most 2 of its 4 pairs (2j, 2j + 1) holding a nonzero code
    ((code & 7) ≠ 0).  That is the pair rule of sm_120's NVFP4 sparse MMA (`mma.sp…kind::mxf4nvf4…m16n8k128`), whose A
    keeps 2 of each 8-chunk's 4 pairs (4 metadata bits a chunk); the element-wise 2:4 of `kind::f8f6f4` carries no UE4M3
    16-block scales and runs at the FP8 rate, so it saves nothing over dense NVFP4."""
    return [all(sum(1 for p in range(q, q + 8, 2) if any(c & 7 for c in codes[p:p + 2])) <= 2
                for q in range(w, w + SPAN_K, 8))
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

    split: Fraction
    int8: Fraction
    identity: int
    dead_pairs: int
    forming: Fraction
    subgrid: int
    windows: int
    counts: dict

    @property
    def total(self) -> Fraction:
        return self.split + self.int8 + self.identity + self.dead_pairs + self.forming + self.subgrid + self.windows

    def to_json(self) -> dict:
        return {"total": str(self.total), "split": str(self.split), "int8": str(self.int8), "identity": self.identity,
                "dead_pairs": self.dead_pairs, "forming": str(self.forming), "subgrid": self.subgrid,
                "windows": self.windows, "counts": dict(self.counts)}


#: A row's F1′ per word: (format, formed side, its E lines, the row's position, its salt-dead codes, the overfit factor)
#: -> (FP4 units, uncertified blocks).  An implementation other than `row_f1` must compute the same C to float rounding.
RowF1 = Callable[[Fmt, Formed, Sequence[Sequence[int]], int, Sequence[bool], float], tuple[Fraction, int]]


def row_f1(fmt: Fmt, side: Formed, e_rows: Sequence[Sequence[int]], pos: int, dead: Sequence[bool],
           overfit: float = 1) -> tuple[Fraction, int]:
    """The rule's evaluation of a row's F1′ (`change_probabilities`, `split_units`), with its uncertified blocks."""
    p, modes = change_probabilities(fmt, side.x[pos], side.alpha[pos], e_rows[pos], side.ae[pos], dead, side.scales[pos])
    return split_units(fmt, p, modes, side.scales[pos], overfit), sum(m is None for m in modes)


#: Per row of one tile's formed operands, by ("A" | "B", tile position): its salt-dead codes, its F1′ per word and its
#: uncertified blocks.  Valid for one (operands, E lines, F1′ evaluation) only.
RowTerms = dict[tuple[str, int], tuple[list[bool], Fraction, int]]


def row_terms(fmt: Fmt, memo: RowTerms, label: str, side: Formed, e_rows: Sequence[Sequence[int]], pos: int,
              f1: RowF1 = row_f1) -> tuple[list[bool], Fraction, int]:
    """A row of A′ ("A") or B̃ ("B"): its salt-dead codes and its F1′ (B̃'s at `F1_B_OVERFIT` times its expectation),
    evaluated once per `memo`, so R1 and the debit read the same evaluation."""
    if (label, pos) not in memo:
        dead = salt_dead(fmt, side.x[pos], side.alpha[pos], side.ae[pos])
        memo[label, pos] = (dead, *f1(fmt, side, e_rows, pos, dead, 1 if label == "A" else F1_B_OVERFIT))
    return memo[label, pos]


def tile_debit(fmt: Fmt, a: Formed, b: Formed, rows: Sequence[int], cols: Sequence[int], n: int,
               e_a: Sequence[Sequence[int]], e_b: Sequence[Sequence[int]], m: int | None = None,
               f1: RowF1 | None = None, memo: RowTerms | None = None) -> Debit:
    """The debit over the credited words at tile positions rows x cols of a tile's formed operands (A′ = a, B̃ = b, with
    this salt's E lines e_a and e_b), for a unit of m x k x n (m defaults to n).  `f1` evaluates a row's F1′; a verifier
    may pass a faster evaluation that charges at least the rule's (`benchmarks/pouw/pearl_c4/f1_fast.py`).  `memo` holds
    rows already evaluated on these operands with this `f1` (`row_terms`)."""
    m = n if m is None else m
    f1 = row_f1 if f1 is None else f1
    memo = {} if memo is None else memo
    t_a = {i: row_terms(fmt, memo, "A", a, e_a, i, f1) for i in rows}
    t_b = {j: row_terms(fmt, memo, "B", b, e_b, j, f1) for j in cols}
    dead_a, dead_b = {i: t[0] for i, t in t_a.items()}, {j: t[0] for j, t in t_b.items()}
    split = len(cols) * sum(t[1] for t in t_a.values()) + len(rows) * sum(t[1] for t in t_b.values())
    uncertified = sum(t[2] for t in t_a.values()) + sum(t[2] for t in t_b.values())
    k = len(a.codes[0])
    int8 = Fraction(int8_saving(fmt, [a.scales[i] for i in rows], [b.scales[j] for j in cols], m, n)) * len(rows) * len(cols) * k
    win_a = {i: two_four_windows(a.codes[i]) for i in range(len(a.codes))}
    win_b = {j: two_four_windows(b.codes[j]) for j in cols}
    windows = SPAN_K // 2 * (len(cols) * sum(sum(win_a[i]) for i in rows) + len(rows) * sum(sum(win_b[j]) for j in cols))
    identity = subgrid = 0
    for i in rows:
        for j in cols:
            _, idn, sg = word_replay(fmt, a.codes[i], b.codes[j], a.scales[i], b.scales[j])
            identity, subgrid = identity + idn, subgrid + sg
    blocks_a, blocks_b = {i: dead_blocks(fmt, a.x[i]) for i in rows}, {j: dead_blocks(fmt, b.x[j]) for j in cols}
    block_pairs = sum(p and q for i in rows for j in cols for p, q in zip(blocks_a[i], blocks_b[j]))
    mask_a = {i: sum(1 << t for t, d in enumerate(dead_a[i]) if d) for i in rows}
    mask_b = {j: sum(1 << t for t, d in enumerate(dead_b[j]) if d) for j in cols}
    pairs = sum((mask_a[i] & mask_b[j]).bit_count() for i in rows for j in cols)
    unneeded = [all(b.codes[j][t] & 7 == 0 for j in cols) for t in range(len(a.codes[0]))]
    elements = sum(d or u for i in rows for d, u in zip(dead_a[i], unneeded))
    screened_only = sum(not (d or u) and blocks_a[i][t // fmt.block]
                        for i in rows for t, (d, u) in enumerate(zip(dead_a[i], unneeded)))
    spans = sum(all(win_a[i][w] for i in range(r, r + 16)) for r in range(0, len(a.codes) - 15, 16)
                for w in range(len(win_a[0])))
    counts = {"salt_dead_a": sum(sum(dead_a[i]) for i in rows), "salt_dead_b": sum(sum(dead_b[j]) for j in cols),
              "uncertified_blocks": uncertified,
              "two_four_windows_a": sum(sum(win_a[i]) for i in rows), "two_four_windows_b": sum(sum(win_b[j]) for j in cols),
              "d24_spans_a": spans, "identity_atom_words": identity, "subgrid_block_words": subgrid, "dead_pairs": pairs,
              "screened_block_pairs": block_pairs,
              "screened_blocks_a": sum(sum(blocks_a[i]) for i in rows), "screened_blocks_b": sum(sum(blocks_b[j]) for j in cols),
              "forming_elements": elements, "screened_only_elements": screened_only, "unneeded_positions": sum(unneeded),
              "dss_spread_rows": sum(not spread_ok(fmt, a.scales[i]) for i in rows)
              + sum(not spread_ok(fmt, b.scales[j]) for j in cols),
              "words": len(rows) * len(cols)}
    return Debit(split, int8, ATOM * identity, pairs, Fraction(FORMING_CREDITED * elements * len(cols), n),
                 fmt.block * subgrid, windows, counts)


# ---- the scheme ---------------------------------------------------------------------------------------------------

class PearlC4:
    """Pearl-C on sm_120's FP4: ``k_min`` = 1024, 64 x 64 tiles and ``n_min`` = 128 (`B_OVF_MIN_N`); smaller bounds make a
    test instance.  ``hashing`` h3 names the scheme ``…-h3``: `pearl_c`'s -h3 on this scheme, frame-b3s trees, BLAKE3
    digests and each row of A drawing its E_A line from its own seed (`row_seeds`); h0 is the default."""

    tickets = C.PearlC.tickets
    cap = CAP

    def __init__(self, fmt: str = "nvfp4", k_min: int = 1024, tile: tuple[int, int] = (TILE, TILE),
                 f1: RowF1 | None = None, n_min: int = B_OVF_MIN_N, hashing: str = "h0"):
        if fmt not in FORMATS:
            raise ValueError(f"format {fmt!r}: one of {', '.join(FORMATS)}")
        if hashing not in HASHINGS:
            raise ValueError(f"hashing {hashing!r}: one of {', '.join(HASHINGS)}")
        self.fmt, self.hashing = FORMATS[fmt], hashing
        self.commitment_hash, self.digest_keys = C.TREE_HASH[hashing], C.digest_keys(hashing)
        #: F1′'s evaluation: the rule's own, or a verifier's that charges at least it (staged: `f1_fast`, pending the
        #: assessor's rating of its bounds)
        self.f1 = row_f1 if f1 is None else f1
        self.name = NAMES[fmt] + ("" if hashing == "h0" else f"-{hashing}")
        #: MXFP4 is contrast only: its int8 closure holds under F2, but F1′ puts 9.5–57% of honest MXFP4 credit over the
        #: cap (UE8M0 scales don't move with the salt, and its round-up scales make honest zeros reliable), so its line is
        #: D on honest coverage and Pearl-C4 claims NVFP4 alone (ratings.md 15:12Z; Daniel, 15:34Z).
        self.contrast = self.fmt is MXFP4
        self.r1_tolerance = R1_TOLERANCE
        self.k_min, self.tile, self.n_min = k_min, tile, n_min
        self.last_debit: Debit | None = None
        self._encoded: tuple | None = None
        self._rows: tuple[C.Ops | None, RowF1 | None, RowTerms] = (None, None, {})
        self.certificate = Certificate(
            conjecture="tt-out/fp4-sm120",
            gamma0=None,
            omega=Fraction(1),
            claims=("tt-out/fp4-sm120", "random-oracle", *(("xof/blake3",) if hashing != "h0" else ()), "cr/blake3",
                    "cr/sha-256", "device-reproducibility", "beacon-unpredictability"),
            hypotheses=(
                "TT_OUT-FP4(1/400) for this Π and domain (D-NF, D-24, D-SS, D-SB, and D-SK: B̃'s noise keyed by "
                "this job's salt, seed-B = keyed(salt ‖ root_b), recomputed by the verifier), with the debit's F1′ and "
                "F2 (theory §6.2–6.3; the assessor's FP4 conditions, ratings.md 15:18Z), as the theory lane states it, proposed and "
                "not proved: no sm_120 W1 program outputs correct C̃ and U words of the units whose replayed debit is under "
                "the cap for less than their credit less the debit (research store "
                "internal/pouw/rtx-pro/theory-pearl-c4-domain.md §2)",
                f"sm_120's mma.sync kind::mxf4nvf4 and BF16 m16n8k16 are reproduced exactly ({self.fmt.atom.name}, "
                "blackwell_sm120_nvf4_m16n8k64 for the noise atom, hopper_bf16_m16n8k16), pinned on the RTX 5090",
                "the E2M1, UE4M3 and UE8M0 casts are as modelled (from the PTX ISA, not captured)",
            ),
            allowances=(
                "the noisy block amax (17 units per element) and ρ over every 8th position (2) are in W_ref, not credited",
                "savings no replay reaches (cross-atom cancellation, gathering across blocks, adaptive regrouping after "
                "the salt beyond F1′'s κ-inside route choice) are TT_OUT-FP4's; the replayed debit charges the rest "
                "(tile_debit), and B̃'s layout chosen after the salt is covered by B-OVF's measured credit discount "
                "(B_OVF, credit_of)",
                "the unscale is glue outside the checked values",
            ),
            parameters=(("k_min", k_min), ("tile_m", tile[0]), ("tile_n", tile[1]), ("rank", R), ("block", self.fmt.block),
                        ("delta_inv", 4), ("dead_rho_sq", DEAD_RHO_SQ), ("n_min", n_min),
                        ("b_ovf", tuple((nb, str(b)) for nb, b in B_OVF)), *((("hashing", hashing),) if hashing != "h0" else ())),
        )
        if self.contrast:
            self.certificate = Certificate(
                conjecture="none: a contrast line outside Pearl-C4's claimed domain (NVFP4 only)",
                gamma0=None,
                omega=Fraction(1),
                claims=(),
                hypotheses=(),
                allowances=("MXFP4 is contrast only: its line is D on honest coverage under F1′ (ratings.md 15:12Z), and "
                            "Pearl-C4 claims NVFP4 alone (Daniel, 15:34Z)",),
                parameters=self.certificate.parameters,
            )

    def check(self, matmul: Matmul) -> None:
        s = matmul.shape
        if not (self.k_min <= s.k <= 1 << 16 and s.k % ALIGN_K == 0 and s.m % self.tile[0] == 0
                and s.n % self.tile[1] == 0 and s.m <= 1 << 24 and self.n_min <= s.n <= 1 << 18):
            raise DomainError(f"{self.name}: {s} is outside {self.k_min} <= k <= 2^16 with 128 | k, tiles of "
                              f"{self.tile}, m <= 2^24, {self.n_min} <= n <= 2^18")

    def check_registration(self, transforms: Sequence[str], curated: bool = False, needs_gamma_fold: bool = False) -> None:
        """The registered-weights rule (`check_registration`)."""
        check_registration(transforms, curated, needs_gamma_fold)

    def tiles(self, shape: Shape) -> list[Tile]:
        return grid_tiles(shape, *self.tile)

    def wref(self, shape: Shape) -> int:
        """The honest W1 cost of one unit in FP4 units: the chain (1 per MAC, no promotion), the fused A′·F_B (r per element
        of A), the BF16 peel (4 per MAC over depth 2r) and A's forming (117 per element).  The weight side is `job_cost`."""
        m, k, n = shape.m, shape.k, shape.n
        return m * n * k + m * k * R + BF16_MAC_UNITS * PEEL_K * m * n + (FORMING_CREDITED + FORMING_UNCREDITED) * m * k

    def credit_of(self, shape: Shape) -> int:
        """`wref` less A's uncredited forming, 19 units per element, and B-OVF's discount rounded up:
        creditOf = mnk + 32·mk + 256·mn + f_s·mk − ⌈β(n)·mnk⌉ (`b_ovf`).  R1's bound and the tile cap read it."""
        m, k, n = shape.m, shape.k, shape.n
        return self.wref(shape) - FORMING_UNCREDITED * m * k - math.ceil(b_ovf(n) * m * n * k)

    def job_cost(self, n: int, k: int) -> int:
        """The weight side, once per weight and epoch: B's forming, B̃·F_A and G as chains, and P_B's FP32 FMA chain."""
        return (FORMING_CREDITED + FORMING_UNCREDITED) * n * k + n * R * k + R * R * k + FP32_UNITS * n * R * R

    def noise(self, ctx: Context) -> C.Noise:
        """The unit's seeds: A's keyed by the job's salt, both roots and the unit, B̃'s by `seed_b`.  The verifier forms
        both sides itself from these seeds on every drawn tile, so a word derived from any other B̃ fails its checked
        values."""
        return C.Noise(unit_seed_a(ctx.salt, ctx.root_a, ctx.root_b, ctx.matmul.index), seed_b(ctx.salt, ctx.root_b))

    def row_seeds(self, ctx: Context, a_rows: Sequence[bytes], idx: Sequence[int]) -> list[bytes]:
        """-h3's seeds of A's rows `idx` (values `a_rows`): `row_seed` over each row's leaf in A's tree, as the prover
        commits it (`audit.commit_rows` at the unit's activation binding)."""
        u = ctx.matmul
        dom = rows_domain(activation_binding(ctx.salt, u.index), u.index, u.shape.m, self.commitment_hash)
        return [row_seed(ctx.salt, dom.leaf(dom.positions.rank(i), i, ROW_SCHEMA, r), ctx.root_b, u.index, i)
                for i, r in zip(idx, a_rows, strict=True)]

    def _lines_a(self, ctx: Context, noise: C.Noise, a_rows: Sequence[bytes], idx: Sequence[int]) -> list[list[int]] | None:
        """E_A's lines of A's rows `idx` under -h3 (line 0 of each row's seed); None under h0, which draws line i of seed_A."""
        return [line(x, ROLE_EA, 0) for x in self.row_seeds(ctx, a_rows, idx)] if self.hashing == "h3" else None

    def parse_row(self, row: bytes, k: int) -> list[int]:
        return parse_row(row, k)

    def row_passes(self, words: Sequence[int]) -> bool:
        return row_passes(self.fmt, words)

    def tile_cap(self, record, ops: C.Ops, rows: Sequence[int], cols: Sequence[int]) -> tuple[Fraction, Fraction]:
        """(debit, cap) over the credited words at tile positions rows x cols: `tile_debit`'s total, and ρ times the
        credited cells' share of creditOf (each row's `credit_of` at |K|, over n).  The breakdown stays in `last_debit`."""
        s = record.matmul.shape
        self.last_debit = tile_debit(self.fmt, ops.a, ops.b, rows, cols, s.n, ops.e_a, ops.e_b, s.m, self.f1,
                                     self._memo(ops))
        cell = Fraction(self.credit_of(Shape(1, record.k_kept, s.n)), s.n)
        return self.last_debit.total, CAP * cell * len(rows) * len(cols)

    def _memo(self, ops: C.Ops) -> RowTerms:
        """The encoded tile's `row_terms`, shared by R1 (`row_debit_ratio`) and the debit (`tile_cap`), so a drawn tile's
        rows are evaluated once; a new tile or a new `f1` starts a new one."""
        if self._rows[0] is not ops or self._rows[1] is not self.f1:
            self._rows = (ops, self.f1, {})
        return self._rows[2]

    def _r1_cap(self, k_kept: int, n: int) -> Fraction:
        return CAP * Fraction(self.credit_of(Shape(1, k_kept, n)), n)

    def row_debit_ratio(self, record, ops: C.Ops, side: str, pos: int) -> Fraction:
        """R1's reading (theory-pearl-c4-domain.md §6.8; ratings.md 15:18Z condition 4): the F1′ per word of the row at
        tile position `pos` of A′ ("A") or B̃ ("B"), over ρ·(its cell credit).  Over 1 the row must be declared
        excluded; `pearl_c_work.check_opened` allows either declaration within `R1_TOLERANCE` of 1.  An excluded row is
        still formed, checked and hashed; it earns nothing."""
        formed, e_rows = (ops.a, ops.e_a) if side == "A" else (ops.b, ops.e_b)
        units = row_terms(self.fmt, self._memo(ops), side, formed, e_rows, pos, self.f1)[1]
        return units / self._r1_cap(record.k_kept, record.matmul.shape.n)

    def row_over_debit(self, record, ops: C.Ops, side: str, pos: int) -> bool:
        """Whether R1 requires the row excluded (its ratio over 1)."""
        return self.row_debit_ratio(record, ops, side, pos) > 1

    def r1_rows(self, ctx: Context, a_rows: Sequence[bytes], b_rows: Sequence[bytes], k_kept: int) -> tuple[set[int], set[int]]:
        """R1 over a whole unit, for an honest prover's declaration (`pearl_c_work.declare`): the A rows and the weight's
        B rows that pass the per-row rules but whose F1′ per word passes ρ·(their cell credit) (`r1_row`)."""
        k = ctx.matmul.shape.k
        a, b = ({i for i, row in enumerate(rows)
                 if row_passes(self.fmt, parse_row(row, k)) and self.r1_row(ctx, side, i, row, k_kept)}
                for side, rows in (("A", a_rows), ("B", b_rows)))
        return a, b

    def r1_row(self, ctx: Context, side: str, index: int, row: bytes, k_kept: int) -> bool:
        """R1 on one row that passes the per-row rules, row `index` of the unit's A ("A") or of its weight's B ("B"):
        whether its F1′ per word, formed under the unit's noise, passes ρ·(its cell credit).  A row's term reads only that
        row, its index and the side's basis, so a prover may map it over the rows in any order and in any process."""
        noise, k = self.noise(ctx), ctx.matmul.shape.k
        role_e, role_f, seed_e = (ROLE_EA, ROLE_FA, noise.seed_a) if side == "A" else (ROLE_EB, ROLE_FB, noise.seed_b)
        lines = self._lines_a(ctx, noise, [row], [index]) if side == "A" else None
        formed, e = self._side([row], [index], role_e, role_f, seed_e, noise.seed_b, k, lines)
        return row_terms(self.fmt, {}, side, formed, e, 0, self.f1)[1] > self._r1_cap(k_kept, ctx.matmul.shape.n)

    def _side(self, rows: Sequence[bytes], idx: Sequence[int], role_e: int, role_f: int, seed_e: bytes, seed_b: bytes,
              k: int, e: list[list[int]] | None = None) -> tuple[Formed, list[list[int]]]:
        e = [line(seed_e, role_e, i) for i in idx] if e is None else e
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
            a, e_a = self._side(a_rows, tile.rows, ROLE_EA, ROLE_FA, noise.seed_a, noise.seed_b, k,
                                self._lines_a(ctx, noise, a_rows, tile.rows))
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
        return C.leaf(checked, self.hashing)

    def useful(self, ctx: Context, noise: C.Noise, ops: C.Ops, checked: C.Checked) -> list[list[float]]:
        """U/(α_A,i · α_B,j) in double, for measuring quality; the served y is the integration's pinned unscale."""
        return [[P.f32_of_bits(w) / (P.f32_of_bits(ops.a.alpha[i]) * P.f32_of_bits(ops.b.alpha[j]))
                 for j, w in enumerate(row)] for i, row in enumerate(checked.u)]


__all__ = ["FORMATS", "HASHINGS", "MXFP4", "NAMES", "NVFP4", "Debit", "PearlC4", "check_registration", "clean", "dnf", "e2m1",
           "form", "hw_scales", "line", "row_passes", "row_seed", "salt_dead", "scale_bytes_ok", "seed_b", "tile_debit",
           "unit_seed_a", "word_replay"]
