"""``pearl-c-h100-v0``, ``pearl-c-h100-v1``, ``pearl-c-h100-v1-h3``, ``pearl-c-sm120-v1`` and ``pearl-c-sm120-v1-h3``:
Pearl-C, Pearl's FP8 product with its noisy clean-up checked, on the H100 and on the RTX PRO 6000 Blackwell.

The design is Pearl's H100 product and its output tickets, plus Pearl v4's noisy clean-up
("peel") pinned bit for bit, so that the useful output U is a checked value
and no part of the clean-up can be skipped at a degenerate input.  γ rests on the named conjecture TT_OUT(ρ) at the device
record's cap ρ (`Device.cap`: 1/400 on the H100, 1/1,000 on sm_120); nothing here is proved by counting.  Pearl's own pieces (noise lines, BF16 compute ops, norms, the noisy quantization, the H100
reset-and-promote replay) are `pearl_kw`'s ports of pearl-research-labs/pearl's certificate-v4 code and its ``zk-pow``
verifier.

**The scheme, per 64 x 64 tile** (rows I of A, rows J of B, full k; both operands stored as rows along k):

~~~text
seeds     seed_B = H_"pearl-c/v0/seed-B"(salt ‖ root_B);  seed_A = H_"pearl-c/v0/seed-A"(salt ‖ root_A ‖ root_B ‖ index u64 LE)
          H_ℓ(x) = BLAKE3(x; key = BLAKE3(ℓ)).  E_A lines by seed_A (row i); E_B lines and the bases F_A, F_B (lines
          0 … k−1) by seed_B: Pearl's line rule (`pearl_kw.sample_line`), so F_A, E_B, F_B are per job (weight)
          (-h3, SHA-512 throughout: every H_ℓ `s512` (SHA-512 after a 128-byte label block, 32 bytes kept), the line rule
          `sample_line`'s 32 bytes of H_"line" for BLAKE3's, and row i's E_A line 0 of its own seed, `row_seed` keyed on
          the row's `sha512/row-seg/v1` digest; the tickets keep seed_A)
forming  v0, Pearl v4's as ported (`pearl_kw`): FP10 rows -> BF16 X, exact norms, α, β, N = E·Fᵀ on the H100 atom,
          X̃ = fp8(clamp(fma(α, X, β·N), ±448));  A′ = X̃_A, B̃ = X̃_B
chain     C̃_ij = the H100 replay of A′_i · B̃_j: every 128 products a fresh accumulator at +0 chains four
          HOPPER_E4M3_K32 atoms, and one FP32 nearest-even add promotes it into the total (−0 becomes +0)
fused     AF_ip = chain(A′_i, F_B[:, p]) (32 extra columns);  BF_jq = chain(B̃_j, F_A[:, q]) and
          G_pq = chain(F_B[:, p], F_A[:, q]) once per job
peel      P_A,i = [bf16(β_A,i · E_A,ip) ‖ RNE_bf16(AF_ip)],   P_B,j = [RNE_bf16(FP32 add(Σ_p fma, −BF_jq)) ‖ −bf16(β_B,j · E_B,jp)]
          where Σ_p fma = the FP32 FMA chain from +0 of bf16(β_B,j · E_B,jp) · G_pq in p order;
          U_ij = C̃_ij, then four HOPPER_BF16_WGMMA_K16 steps over P_A,i · P_B,j (k16 slices in order)
leaf      for each tile row r and each 32-column chunk h: d_rh = TurboSHAKE128(le32 C̃ of the chunk ‖ le32 U of the chunk,
          D = 0x21); leaf = TurboSHAKE128(d_00 ‖ d_01 ‖ d_10 ‖ … , D = 0x22), 32 bytes each
          (-h3: the leaf is `sha512/row-seg/v1` over the messages themselves, 64 bytes)
trees     A's rows, B's rows and the tile leaves: `verity.core.protocols.pouw.audit`'s framed SHA-256 Merkle trees
          (-h3: `audit.HM96`, frame-v3-sha512 with hm96-sha512 leaves over each row's `sha512/row-seg/v1` digest and each
          tile's leaf, salted by the prover)
tickets   Pearl's lottery over C̃: per 16 x 16 subtile, the 16-lane fold (`pearl_kw.extract`) hashed as H_"jackpot"(z; seed_A)
          (-h3: `s512`'s H_"jackpot"(seed_A ‖ z))
useful    U_ij / (α_A,i · α_B,j), the model's own unscale (glue)
~~~

**The clean-up's algebra.**  With N_A = E_A·F_Aᵀ and N_B = E_B·F_Bᵀ, A′ ≈ α_A·A + β_A⊙N_A and B̃ ≈ α_B·B + β_B⊙N_B, and
(A′ − β_A⊙N_A)(B̃ − β_B⊙N_B)ᵀ = C̃ + (β_A⊙E_A)·[(β_B⊙E_B)·G − B̃·F_A]ᵀ + (A′·F_B)·(−β_B⊙E_B)ᵀ with G = F_Bᵀ·F_A, which is
the rank-64 BF16 product P_A·P_Bᵀ.  Pearl v4 runs G and the B-side factor on a tolerance path; here both are pinned.

**Two formings.**  ``pearl-c-h100-v0`` keeps Pearl v4's FP10 decode and exact norms (the port), whose 176–256 A-only units
are skippable.  ``pearl-c-h100-v1`` is the rework, on rows of FP32
words x, with every rounding pinned here:

~~~text
stats     s = max|x|;  S = Σ x_p² over p = 0, 8, 16, …: 32 partial FMA chains from +0 (partial j takes the positions 8(j + 32i),
          i ascending), then the butterfly v_j <- v_j + v_(j xor o) for o = 16, 8, 4, 2, 1;  ρ = sqrt(S / (k/8));  σ = ρ (δ = 1)
          α = 448 / max(s + 4ρ, 2^−32);  β_row = (σ·α)·f32(√r/(256·16))                           (FP32, nearest even, in that order)
lines     E at Pearl's norm 256, F_A and F_B at norm 16;  E′ = e4m3(β_row · E)        (one FP32 product, one nearest-even cast)
atom      D = HOPPER_E4M3_K32.step(C = α·x, E′_row, F line), then X̃ = e4m3(D)          (saturating nearest-even cast, no BF16)
peel      P_A = [bf16(E′_A) ‖ RNE_bf16(A′·F_B)],  P_B = [RNE_bf16(Σ_p fma(E′_B, G) − B̃·F_A) ‖ −bf16(E′_B)]
~~~

Only ρ depends on A alone (about 4 units per element).  The model adopts s, ρ and α as its own FP8 quantizer (e4m3(α·x)
with PoUW off), and the accounting is the cheap-binding lane's ``Costs.adopted``: 68 units per element, 64 of them credited
(the noise atom and the cast), at the H100's prices.  Every price is a device's (`Prices`): `H100_PRICES` (sm_90) for
the registered schemes, and `SM120_PRICES` (the device statement's `Prices.sm120`, measured) for sm_120's atom.  v1 shares v0's seed labels (``pearl-c/v0/seed-*``): the two formings draw the same seeds.

**The domain** (TT_OUT's `pearlCDomain` and `OperandOK`): 128 ∣ k (whole promotion groups), and on every row of both
operands the noise floor ρ·α ≥ 1 (the exact product of the FP32 words) and liveness `LiveRow 1 x Stat.every8`: with the
exact ρ² = Σ_{p ≡ 0 (8)} x_p² / ⌈k/8⌉ over the words' values, at most k/64 entries have x_l² ≥ 64 ρ².

**The device** (`pearl_c_device.Device`) is a scheme parameter: the atom, the peel's step, G and the W1 prices above are
the record's, and the H100 names in this docstring are `pearl_c_device.H100`'s.  ``pearl-c-sm120-v1`` is v1 on
`pearl_c_device.SM120`: the RTX PRO 6000's captured atom at `SM120_PRICES`, with the peel, G and the k bound as stand-ins
(its `standins`); the promotion window is G atoms and the domain's whole groups are 32·G ∣ k.  An unpromoted record (G None,
`pearl_c_device.SM120_UNPROMOTED`) chains one accumulator over all of k, and its domain is 32 ∣ k.  v0 is the H100's only.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from verity.ml.commitments.rowleaf import sha512_row_seg_digest
from verity.ml.crypto.blake3 import blake3, keyed
from verity.ml.crypto.turboshake import turboshake128
from verity.ml.circuits.silicon import fp32

from verity.core.protocols.pouw.protocol import CALL_RANGE, Certificate, Context, DomainError, Matmul, Shape, Tile, grid_tiles
from verity.core.protocols.pouw.registration import check_registration
from verity.core.protocols.pouw.serving import HASH_FORMATS
from . import pearl_kw as P
from .pearl_c_device import H100, H100_PRICES, PRICES, SM120_PRICES, Device, Prices  # noqa: F401 - re-exported

NAME = "pearl-c-h100-v0"
NAME_V1 = "pearl-c-h100-v1"
R = P.R
WINDOW = P.WINDOW                  # 128 products: four k32 atoms per promotion
SLICE = 32
PEEL_K = 2 * R                     # the peel's BF16 depth, four k16 slices
TILE = 64
CHUNK = 32                         # columns per ticket message
D_MESSAGE, D_LEAF = 0x21, 0x22     # TurboSHAKE128 domain bytes
LABELS = {r: f"pearl-c/v0/{r}".encode() for r in ("seed-A", "seed-B")}
#: The hashing formats this scheme implements, of the one table (`verity.core.protocols.pouw.serving.HASH_FORMATS`): h0 (TurboSHAKE128
#: messages and leaf, SHA-256 trees) and h3 (SHA-512 throughout: `s512` seeds, lines and tickets, a `sha512/row-seg/v1` tile
#: leaf and `audit.HM96` trees, each row of A seeding its own E_A line from its row digest).  The BLAKE3 formats h1, h2 and
#: BLAKE3 -h3 are in ``archive/pouw-blake3/`` (one hash, SHA-512).
HASHINGS = ("h0", "h3")
#: the formats whose rows seed their own E_A lines (`HashFormat.seed_a` ``row``)
ROW_SEEDED = tuple(h for h in HASHINGS if HASH_FORMATS[h].seed_a == "row")
#: each hashing format's frame hash for the row and transcript trees (`verity.ml.commitments.merkle`), from the one table
TREE_HASH = {h: HASH_FORMATS[h].tree_hash for h in HASHINGS}
#: the per-row cap at a census k, as a multiple of ρ: there each credited row's debit
#: is at most 4ρ times its cells' credit, the tile's still at most ρ (`pearlCTilesDevRev1RowCapAt` at ρr = 4ρ; γ rests on
#: TT_OUT per tile at ρr, `pearlCSampledH100v1RowSeedCodeRowCap100_6144x256`); at k >= k_min a row's cap is ρ, (C2)'s
CENSUS_ROW_CAP = 4


#: Forming v0 (Pearl v4's, as ported) per operand element: 240–320 for the port, at the upper end
#: of the H100's prices; v0 is priced on the H100 only
FORMING_V0_UNITS = 320
#: F's line norm under v1 (E keeps Pearl's 256)
LINE_NORM_V1 = 16.0
S5_STRIDE = 8
S5_FLOOR = 0x2F800000              # 2^-32
S5_FOUR, S5_QMAX = 0x40800000, 0x43E00000
S5_LINE = P.f32_bits(P.f32((R ** 0.5) / (P.NOISE_NORM * LINE_NORM_V1)))


def _labelled(message: bytes, role: str) -> bytes:
    return keyed(blake3(LABELS[role]), message)


#: -h3's labels: H_ℓ(x) = SHA-512(``pearl-c/h3s/<ℓ>\0`` zero-padded to one 128-byte block ‖ x), its first 32 bytes.  The
#: ``h3s`` in the label bytes is a domain tag, kept from the format's first name, so the pinned digests are unchanged
S512_ROLES = ("seed-A", "seed-B", "noise-line", "line", "jackpot")
S512_LABELS = {r: f"pearl-c/h3s/{r}\0".encode().ljust(128, b"\0") for r in S512_ROLES}


def s512(role: str, message: bytes) -> bytes:
    """-h3's H_ℓ (`S512_LABELS`): the first 32 bytes of SHA-512 over the label's block and the message."""
    return hashlib.sha512(S512_LABELS[role] + message).digest()[:32]


class S512Seed(bytes):
    """A seed of -h3: its noise lines are drawn by `sample_line`'s SHA-512 rule, not Pearl's BLAKE3 one."""


def unit_seed_a(salt: bytes, root_a: bytes, root_b: bytes, index: int, hashing: str = "h0") -> bytes:
    """A unit's seed_A: H_"pearl-c/v0/seed-A"(salt ‖ root_A ‖ root_B ‖ index u64 LE); under -h3 H_"seed-A" (`s512`)."""
    msg = salt + root_a + root_b + index.to_bytes(8, "little")
    return S512Seed(s512("seed-A", msg)) if hashing == "h3" else _labelled(msg, "seed-A")


def seed_b(salt: bytes, root_b: bytes, hashing: str = "h0") -> bytes:
    """An epoch weight's seed_B: H_"pearl-c/v0/seed-B"(salt ‖ root_B); under -h3 H_"seed-B" (`s512`)."""
    return S512Seed(s512("seed-B", salt + root_b)) if hashing == "h3" else _labelled(salt + root_b, "seed-B")


def line_key(seed: bytes) -> bytes:
    """The key a seed's noise lines are drawn under: Pearl's (`pearl_kw._line_key`), or an `S512Seed`'s H_"noise-line"(seed)."""
    return _s512_line_key(seed) if isinstance(seed, S512Seed) else P._line_key(seed)


def row_seed(salt: bytes, leaf: bytes, root_b: bytes, unit: int, row: int, hashing: str = "h3") -> bytes:
    """-h3's seed of row `row` of unit `unit`, whose E_A line is drawn from it alone: the one-row unit's seed_A over the
    row's 64-byte SHA-512 digest `leaf` (`row_digest`), H_"seed-A"(salt ‖ leaf ‖ root_B ‖ (unit · 2^32 + row) u64 LE)
    (`s512`, 168 bytes after the label's block with root_B's 64), with the unit index and the row each below 2^32
    (`CALL_RANGE`). `hash_h3.cuh`'s `s512_seeds` computes it."""
    if not (0 <= unit < CALL_RANGE and 0 <= row < 1 << 32):
        raise ValueError(f"row_seed: unit {unit} and row {row} must each be below 2^32")
    return unit_seed_a(salt, leaf, root_b, unit << 32 | row, hashing)


def row_digest(row: bytes) -> bytes:
    """-h3's key of a row's seed: ``sha512/row-seg/v1`` over the row's bytes (its FP32 words little-endian, the row A's
    tree commits), `rowleaf.sha512_row_seg_digest`: 1 KiB segments hashed on their own, then their digests.  It is also
    the inner digest x of the row's hm96 leaf in A's tree (`audit.Hm96Domain`)."""
    return sha512_row_seg_digest(row, 8 * len(row))


def sample_line(seed: bytes, side: int, factor: int, line: int, norm: float = P.NOISE_NORM) -> list[int]:
    """One noise line: Pearl's (`pearl_kw.sample_line`), or for an `S512Seed` the same rule over 32 bytes of SHA-512,
    H_"line"(H_"noise-line"(seed) ‖ side ‖ factor ‖ line u32 LE), in place of BLAKE3's keyed by Pearl's line key."""
    if not isinstance(seed, S512Seed):
        return P.sample_line(seed, side, factor, line, norm)
    raw = s512("line", _s512_line_key(seed) + bytes([side, factor]) + line.to_bytes(4, "little"))
    xs = [(-1 if b >> 7 else 1) * ((b & 0x7F) + 1) for b in raw]
    scale = P.bf16_div(P.f32_to_bf16(norm * 32), P.f32_to_bf16(float(math.isqrt(sum(x * x for x in xs) * 32 * 32))))
    return [P.f32_to_fp8(P.bf16_to_f32(P.bf16_mul(P.f32_to_bf16(float(x)), scale))) for x in xs]


@lru_cache(maxsize=64)
def _s512_line_key(seed: bytes) -> bytes:
    return s512("noise-line", seed)


def _word(x: float) -> int:
    return P.f32_bits(x)


def _check_finite(word: int) -> int:
    if (word >> 23) & 0xFF == 0xFF:
        raise P.Reject("non-finite FP32 accumulator")
    return word


def chain(a: Sequence[int], b: Sequence[int], device: Device = H100) -> int:
    """The device's replay of ``a · b`` (E4M3 codes) as an FP32 word: per window of G atoms (128 products on the H100) a
    fresh +0 accumulator chains up to G of the device's atoms, then one FP32 nearest-even add promotes it into the total;
    a zero total is +0.  Unpromoted (G None), one accumulator from +0 chains every atom and is the word."""
    if len(a) != len(b) or len(a) % SLICE:
        raise P.Reject("the chain multiplies in atoms of k = 32")
    total, atom, window = 0, device.atom, device.window
    if device.g is None:
        for s in range(0, len(a), SLICE):
            total = _check_finite(atom.step(total, a[s:s + SLICE], b[s:s + SLICE]))
        return total
    for ws in range(0, len(a), window):
        local = 0
        for s in range(ws, min(ws + window, len(a)), SLICE):
            local = _check_finite(atom.step(local, a[s:s + SLICE], b[s:s + SLICE]))
        total = _check_finite(fp32.add(total, local))
        if total == 0x80000000:
            total = 0
    return total


def peel(c_word: int, pa: Sequence[int], pb: Sequence[int], device: Device = H100) -> int:
    """U from C̃: four of the device's BF16 k16 steps over the 64 BF16 peel operands, k16 slices in order, into C̃'s word."""
    acc = c_word
    for s in range(0, PEEL_K, 16):
        acc = _check_finite(device.peel.step(acc, pa[s:s + 16], pb[s:s + 16]))
    return acc


def _bf16_of_word(word: int) -> int:
    return P.f32_to_bf16(P.f32_of_bits(word))


def _bf16_scaled(beta: int, code: int) -> int:
    """bf16(β · e) for an E4M3 noise code e (Pearl's BF16 compute semantics: exact decode, one f32 product, RNE)."""
    return P.bf16_mul(beta, P.f32_to_bf16(P.fp8_to_f32(code)))


def _bf16_of_code(code: int) -> int:
    return P.f32_to_bf16(P.fp8_to_f32(code))


@dataclass(frozen=True, slots=True)
class Noise:
    seed_a: bytes
    seed_b: bytes


@dataclass(slots=True)
class Formed:
    """One side formed by v1 (§5): FP32 words for s, ρ, α and β_row, the E′ codes, the operand's codes and its input."""

    codes: list[list[int]]
    alpha: list[int]
    beta: list[int]
    s: list[int]
    rho: list[int]
    ep: list[list[int]]
    x: list[list[int]]


@dataclass(slots=True)
class Ops:
    """One tile's formed operands and peel factors."""

    a: P.Built | Formed             # A′ = a.codes, with α_A, β_A
    b: P.Built | Formed             # B̃ = b.codes, with α_B, β_B
    e_a: list[list[int]]            # E_A lines of the tile's rows
    e_b: list[list[int]]            # E_B lines of the tile's columns
    pa: list[list[int]]             # P_A rows, 64 BF16 words
    pb: list[list[int]]             # P_B rows, 64 BF16 words


@dataclass(slots=True)
class Checked:
    """The tile's checked words: C̃ (the tickets' accumulator) and U (the useful output before its unscale)."""

    c: list[list[int]]
    u: list[list[int]]


@lru_cache(maxsize=16, typed=True)
def _basis(seed_b: bytes, side: int, k: int, norm: float = P.NOISE_NORM) -> tuple[tuple[int, ...], ...]:
    """F_A (side 0) or F_B (side 1) as k lines of 32 codes (line t is row t of the k x r basis), each at L2 norm `norm`."""
    return tuple(tuple(sample_line(seed_b, side, 1, t, norm)) for t in range(k))


def _column(basis: tuple[tuple[int, ...], ...], p: int) -> list[int]:
    return [line[p] for line in basis]


@lru_cache(maxsize=16, typed=True)
def gram(seed_b: bytes, k: int, norm: float = P.NOISE_NORM, device: Device = H100) -> tuple[tuple[int, ...], ...]:
    """G = F_Bᵀ·F_A (r x r) as FP32 words, G_pq = chain(F_B[:, p], F_A[:, q]); once per job."""
    fa, fb = _basis(seed_b, 0, k, norm), _basis(seed_b, 1, k, norm)
    fa_cols = [_column(fa, q) for q in range(R)]
    return tuple(tuple(chain(_column(fb, p), fa_cols[q], device) for q in range(R)) for p in range(R))


def scaled_noise(side: P.Built | Formed, e_rows: Sequence[Sequence[int]]) -> list[list[int]]:
    """The peel's noise rows as BF16 words: bf16(β·E) for v0, bf16(E′) (exact) for v1."""
    if isinstance(side, Formed):
        return [[_bf16_of_code(c) for c in row] for row in side.ep]
    return [[_bf16_scaled(beta, e) for e in e_row] for beta, e_row in zip(side.beta, e_rows)]


def peel_factors_a(codes: Sequence[Sequence[int]], be_rows: Sequence[Sequence[int]], seed_b: bytes, k: int,
                   norm: float = P.NOISE_NORM, device: Device = H100) -> list[list[int]]:
    """P_A rows: [be ‖ RNE_bf16(A′·F_B)], the second half the 32 fused columns of the product."""
    fb = _basis(seed_b, 1, k, norm)
    fb_cols = [_column(fb, p) for p in range(R)]
    return [list(be) + [_bf16_of_word(chain(row, fb_cols[p], device)) for p in range(R)] for row, be in zip(codes, be_rows)]


def peel_factors_b(codes: Sequence[Sequence[int]], be_rows: Sequence[Sequence[int]], seed_b: bytes, k: int,
                   norm: float = P.NOISE_NORM, device: Device = H100) -> list[list[int]]:
    """P_B rows: [RNE_bf16(be·G − B̃·F_A) ‖ −be], once per job.  be·G is the FP32 FMA chain from +0 in p order, and the
    difference one FP32 nearest-even add of the negated B̃·F_A."""
    fa = _basis(seed_b, 0, k, norm)
    fa_cols = [_column(fa, q) for q in range(R)]
    g = gram(seed_b, k, norm, device)
    rows = []
    for row, be in zip(codes, be_rows):
        mid = []
        for q in range(R):
            acc = 0
            for p in range(R):
                acc = _check_finite(fp32.fma(be[p] << 16, g[p][q], acc))
            mid.append(_bf16_of_word(_check_finite(fp32.add(acc, fp32.neg(chain(row, fa_cols[q], device))))))
        rows.append(mid + [x ^ 0x8000 for x in be])
    return rows


def peel_a(built: P.Built, e_rows: Sequence[Sequence[int]], seed_b: bytes, k: int) -> list[list[int]]:
    """v0's P_A rows: [bf16(β_A·E_A) ‖ RNE_bf16(A′·F_B)]."""
    return peel_factors_a(built.codes, scaled_noise(built, e_rows), seed_b, k)


def peel_b(built: P.Built, e_rows: Sequence[Sequence[int]], seed_b: bytes, k: int) -> list[list[int]]:
    """v0's P_B rows: [RNE_bf16((β_B·E_B)·G − B̃·F_A) ‖ −bf16(β_B·E_B)]."""
    return peel_factors_b(built.codes, scaled_noise(built, e_rows), seed_b, k)


# ---- forming v1 (§5) ------------------------------------------------------------------------------------------------

def parse_f32_row(row: bytes, k: int) -> list[int]:
    if len(row) != 4 * k:
        raise P.Reject("a v1 row has k FP32 words")
    words = list(struct.unpack(f"<{k}I", row))
    for w in words:
        _check_finite(w)
    return words


def f32_row(words: Sequence[int]) -> bytes:
    return struct.pack(f"<{len(words)}I", *words)


def s5_stats(x: Sequence[int]) -> tuple[int, int, int, int]:
    """(s, ρ, α, β_row) of one row of FP32 words, as FP32 words, in the pinned order (module docstring)."""
    s = max((w & 0x7FFFFFFF for w in x), default=0)
    part = [0] * 32
    positions = range(0, len(x), S5_STRIDE)
    for i, p in enumerate(positions):
        part[i % 32] = _check_finite(fp32.fma(x[p], x[p], part[i % 32]))
    for o in (16, 8, 4, 2, 1):
        part = [_check_finite(fp32.add(part[j], part[j ^ o])) for j in range(32)]
    rho = fp32.sqrt(fp32.div(part[0], P.f32_bits(float(len(positions)))))
    t = _check_finite(fp32.add(s, fp32.mul(S5_FOUR, rho)))
    alpha = fp32.div(S5_QMAX, max(t, S5_FLOOR))
    return s, rho, alpha, _check_finite(fp32.mul(fp32.mul(rho, alpha), S5_LINE))


def live_row(words: Sequence[int]) -> bool:
    """TT_OUT's liveness on one row of FP32 words, exactly: reject when 64 · #{l : x_l² · ⌈k/8⌉ ≥ 64 · Σ_{p ≡ 0 (8)} x_p²} > k.
    Each x² is exact as a float (48 significant bits), so the squares are compared as integers at the scale 2^300; a float
    prefilter leaves only entries within 2^-30 of the threshold to the exact comparison."""
    k = len(words)
    vals = [P.f32_of_bits(w) for w in words]
    positions = range(0, k, S5_STRIDE)
    total = sum(int(vals[p] * vals[p] * 2.0 ** 300) for p in positions)
    k8, bound = len(positions), 64 * total
    approx = bound / k8 / 2.0 ** 300 if total else 0.0
    dead = 0
    for x in vals:
        sq = x * x
        if sq > approx * (1 + 2.0 ** -30):
            dead += 1
        elif sq >= approx * (1 - 2.0 ** -30) and int(sq * 2.0 ** 300) * k8 >= bound:
            dead += 1
    return 64 * dead <= k


def eprime(beta: int, e_row: Sequence[int]) -> list[int]:
    """E′ = e4m3(β_row · E): one FP32 product per code, then Pearl's nearest-even saturating cast."""
    return [P.f32_to_fp8(P.f32_of_bits(fp32.mul(beta, P.f32_bits(P.fp8_to_f32(e))))) for e in e_row]


def form_v1(x_rows: Sequence[Sequence[int]], e_rows: Sequence[Sequence[int]], f: Sequence[Sequence[int]],
            device: Device = H100) -> Formed:
    """§5's forming: per element one of the device's atoms with α·x as its accumulator, then the cast."""
    out = Formed([], [], [], [], [], [], [])
    for x, e_row in zip(x_rows, e_rows):
        s, rho, alpha, beta = s5_stats(x)
        ep = eprime(beta, e_row)
        out.codes.append([P.f32_to_fp8(P.f32_of_bits(_check_finite(device.atom.step(fp32.mul(alpha, xt), ep, f[t]))))
                          for t, xt in enumerate(x)])
        for field, v in (("alpha", alpha), ("beta", beta), ("s", s), ("rho", rho), ("ep", ep), ("x", list(x))):
            getattr(out, field).append(v)
    return out


def message_digest(msg: bytes) -> bytes:
    """One (row, 32-column chunk) message's digest under h0: TurboSHAKE128 (D = 0x21)."""
    return turboshake128(msg, D_MESSAGE, 32)


def messages(checked: Checked) -> list[bytes]:
    """The tile's (row, 32-column chunk) messages, row-major, chunks in order: le32 C̃ of the chunk ‖ le32 U of it."""
    out = []
    for crow, urow in zip(checked.c, checked.u):
        for h in range(0, len(crow), CHUNK):
            out.append(struct.pack(f"<{len(crow[h:h + CHUNK])}I", *crow[h:h + CHUNK])
                       + struct.pack(f"<{len(urow[h:h + CHUNK])}I", *urow[h:h + CHUNK]))
    return out


def leaf(checked: Checked, hashing: str = "h0") -> bytes:
    """The tile leaf.  Under h0, TurboSHAKE128 (D = 0x22) over the per-(row, 32-column chunk) message digests, row-major,
    chunks in order.  Under -h3, the 64-byte ``sha512/row-seg/v1`` digest of the messages themselves, so each 1 KiB
    segment's digest (four 256-byte messages of a 64-column tile) takes the message digests' place."""
    if hashing == "h3":
        joined = b"".join(messages(checked))
        return sha512_row_seg_digest(joined, 8 * len(joined))
    return turboshake128(b"".join(message_digest(msg) for msg in messages(checked)), D_LEAF, 32)


def tickets(seed_a: bytes, checked: Checked, hashing: str = "h0") -> list[bytes]:
    """Pearl's lottery tickets over C̃: one per 16 x 16 subtile, row-major, H_"jackpot"(fold; seed_A); under -h3 `s512`'s
    H_"jackpot"(seed_A ‖ fold)."""
    out = []
    for i in range(0, len(checked.c), P.TILE):
        for j in range(0, len(checked.c[0]), P.TILE):
            sub = [[P.f32_of_bits(w) for w in row[j:j + P.TILE]] for row in checked.c[i:i + P.TILE]]
            fold = P.extract(sub)
            out.append(s512("jackpot", seed_a + fold) if hashing == "h3" else P.hash_labelled(fold, "jackpot", seed_a))
    return out


class PearlC:
    """Pearl-C on a device (`pearl_c_device`, the H100 by default): ``k_min`` = 1024 (Pearl's domain), and under forming v1
    and -h3 the device record's `census_ks` below it (each row capped at 4ρ there, `row_cap_at`), and 64 x 64 tiles; smaller bounds make a test instance.  ``hashing`` h3
    names the scheme ``…-h3``: SHA-512 throughout (the module docstring), each row of A drawing its E_A line from its own seed keyed on its SHA-512 row digest (`row_seeds`,
    `row_digest`), the work law checking the cap per row too.  The tree hash is `TREE_HASH[hashing]`, on the instance.  ``prices`` is the W1 price record that `wref`, `credit_of`, `job_cost` and the debit read, the
    device's own by default; another record marks the name ``…@<device>`` (the H100's atom at sm_120's prices is
    ``pearl-c-h100-v1@sm_120``)."""

    def __init__(self, k_min: int = 1024, tile: tuple[int, int] = (TILE, TILE), forming: str = "v0", device: Device = H100,
                 hashing: str = "h0", prices: Prices | None = None):
        if forming not in ("v0", "v1"):
            raise ValueError(f"forming {forming!r}: v0 (#311's, as ported) or v1 (§5's)")
        if forming == "v0" and device != H100:
            raise ValueError(f"forming v0 is #311's H100 port; {device.name} takes v1")
        if hashing not in HASHINGS:
            raise ValueError(f"hashing {hashing!r}: h0 (TurboSHAKE128 and SHA-256) or h3 (SHA-512, per-row seeds)")
        prices = prices or device.prices
        if forming == "v0" and prices != H100_PRICES:
            raise ValueError(f"forming v0 is #311's H100 port, priced on the H100 only, not {prices.device}")
        self.name = ((NAME if forming == "v0" else f"pearl-c-{device.name}-v1") + ("" if hashing == "h0" else f"-{hashing}")
                     + ("" if prices == device.prices else f"@{prices.device}"))
        self.k_min, self.tile, self.forming, self.device, self.hashing, self.prices = k_min, tile, forming, device, hashing, prices
        #: the ks below k_min the domain takes: the device record's census-passed ks, whose census ran §5's forming (v1)
        #: under a row-capped format (-h3), where a row's cap there is `CENSUS_ROW_CAP`·ρ (`row_cap_at`)
        self.census_ks = (tuple(k for k in device.census_ks if k < k_min)
                          if forming == "v1" and HASH_FORMATS[hashing].row_cap else ())
        #: ρ, the device record's: the tile rule's debit cap and the credit's (1 − ρ) (`clear_work`)
        self.cap = device.cap
        self.commitment_hash = TREE_HASH[hashing]
        self.line_norm = P.NOISE_NORM if forming == "v0" else LINE_NORM_V1
        allowance = ("forming v0 is #311's as ported (FP10 decode and exact norms depend on A alone and are skippable, "
                     "X-PC-2); pearl-c-h100-v1 has the reworked forming") if forming == "v0" else (
                     "forming v1 is §5's: ρ over every 8th position depends on A alone (about 4 units per element) and is "
                     "not credited; the model adopts s, ρ and α as its own FP8 quantizer (Costs.adopted)")
        rho, honest = device.cap, f"{float(100 * (1 - device.cap)):g}%"
        if device == H100:
            hardware = (f"TT_OUT({rho}) for Pearl-C's Π and domain (δ = 1, promotion every 4 atoms, k <= 2^16): no H100 W1 "
                        f"program outputs correct C̃ and U words for less than {honest} of their honest cost (research store "
                        "docs/pouw/cheap-binding.md §1); a conjecture, not proved by counting",
                        "the H100's FP8 wgmma and BF16 wgmma are reproduced exactly (HOPPER_E4M3_K32, HOPPER_BF16_WGMMA_K16)")
        else:
            promotion = f"promotion every {device.g} atoms" if device.g else "no promotion"
            hardware = (f"TT_OUT({rho}) for Pearl-C's Π and domain on {device.name} (δ = 1, {promotion}, "
                        f"k <= {device.k_max}): no {device.name} W1 program outputs correct C̃ and U words for less than "
                        f"{honest} of their honest cost; a conjecture stated for the H100 (research store "
                        "docs/pouw/cheap-binding.md §1), not yet re-derived on this device's atom",
                        f"the {device.name} FP8 and BF16 mma steps are reproduced exactly ({device.atom.name}, "
                        f"{device.peel.name})",
                        *(f"stand-in: {s}" for s in device.standins))
        if self.census_ks:
            hardware += (f"at k in {', '.join(map(str, self.census_ks))} (the census's): TT_OUT per tile at "
                         f"ρr = {CENSUS_ROW_CAP * rho}, each credited row's debit within ρr of its cells' credit and the "
                         f"tile's within ρ (Lean `Pouw.PearlC.pearlCSampledH100v1RowSeedCodeRowCap100_6144x256`)",)
        self.certificate = Certificate(
            conjecture="tt-out",
            gamma0=None,
            omega=Fraction(1),
            claims=("tt-out", "random-oracle", "cr/blake3", "device-reproducibility", "live-verifier") if hashing == "h0"
            else ("tt-out", "random-oracle", "cr/sha-512", "cr/sha-256", "device-reproducibility", "live-verifier"),
            theorems=("Pouw.SecurityProofs.WorkWeightedSampling",),
            hypotheses=hardware,
            allowances=(allowance, "the unscale is glue outside the checked values"),
            parameters=(("k_min", k_min), ("tile_m", tile[0]), ("tile_n", tile[1]), ("rank", R), ("window", device.window),
                        ("forming", forming), ("line_norm_f", self.line_norm), ("device", device.name),
                        ("atom", device.atom.name), ("peel", device.peel.name), ("g", device.g), ("hashing", hashing),
                        ("prices", prices.device), ("cap", str(device.cap)),
                        ("census_ks", self.census_ks),
                        *((("census_row_cap", str(CENSUS_ROW_CAP * device.cap)),) if self.census_ks else ())),
        )

    def row_cap_at(self, k: int) -> Fraction:
        """A credited row's debit cap at depth ``k``, as a share of its cells' credit: `CENSUS_ROW_CAP`·ρ at a census k,
        else ρ.  The tile's cap is ρ at every k.  Only a row-capped format (`HashFormat.row_cap`) applies it."""
        return CENSUS_ROW_CAP * self.cap if k in self.census_ks else self.cap

    def check(self, matmul: Matmul) -> None:
        """The domain: whole promotion groups, whole column tiles, and down the rows whole ticket subtiles (whole tiles, for
        a test instance's tile under 16 rows), so the last row tile may be partial (decode's 32 x 64 tile)."""
        s, d, rows = matmul.shape, self.device, min(self.tile[0], P.TILE)
        if not ((self.k_min <= s.k or s.k in self.census_ks) and s.k <= d.k_max and s.k % d.window == 0 and s.m % rows == 0
                and s.n % self.tile[1] == 0 and s.m <= 1 << 24 and s.n <= 1 << 24):
            also = f" or k in {', '.join(map(str, self.census_ks))} (the census's)" if self.census_ks else ""
            raise DomainError(f"{self.name}: {s} is outside {self.k_min} <= k <= {d.k_max}{also} with {d.window} | k (whole "
                              f"promotion groups), {rows} | m, {self.tile[1]} | n, m, n <= 2^24")

    def check_registration(self, transforms: Sequence[str], curated: bool = False, needs_gamma_fold: bool = False) -> None:
        """The registered-weights rule, Pearl-C4's too (`registration.check_registration`)."""
        check_registration(transforms, curated, needs_gamma_fold)

    def tiles(self, shape: Shape) -> list[Tile]:
        return grid_tiles(shape, *self.tile)

    def _forming(self) -> int:
        return FORMING_V0_UNITS if self.forming == "v0" else self.prices.forming

    def _chain_units(self, rows: int, cols: int, k: int) -> int:
        return rows * cols * (k + self.prices.fadd * self.device.promotions(k))

    def first_adds(self, shape: Shape) -> int:
        """Lean's `firstAddDev` (`DeviceRev1.lean`; `firstAddOf` at the H100's prices): the first promotion add of every
        word, m·n adds at the add price, none without promotion.  It adds into T_0 = +0 and so returns its group's word:
        no program needs to execute it, the honest kernel included."""
        return self.prices.fadd * shape.m * shape.n if self.device.g else 0

    def wref(self, shape: Shape) -> int:
        """The honest W1 cost of one unit at the scheme's prices under rev1, Lean's `wrefDevRev1` (`DeviceRev1.lean`;
        `wrefOfRev1` at the H100's prices, and `wrefDevRev1K` at the cast offset 0): the chain (1 per MAC) with its
        promotion adds (one FADD per G atoms) less the first of every word (`first_adds`), the fused A′·F_B (r per element
        of A), the peel in BF16 (a BF16 MAC per product over depth 2r) and A's forming (fs + qa per element under v1, 320
        under v0).  The weight side (B's forming, B̃·F_A, G, P_B) is done once per weight and epoch, not per unit: it is
        `job_cost`, outside W_ref and the credit.  Hashing is the random oracle's."""
        m, k, n = shape.m, shape.k, shape.n
        return (self._chain_units(m, n, k) - self.first_adds(shape) + m * k * R + self.prices.bf16_mac * PEEL_K * m * n
                + self._forming() * m * k)

    def credit_of(self, shape: Shape) -> int:
        """Lean's `creditDevRev1` (`creditOfRev1` at the H100's prices): `wref` less ρ's qa per element of A, which
        depends on A alone, so `creditDev` less `first_adds`.  An accepted unit is credited (1 − ρ)·`credit_of`
        (`pearlCProtocolDevRev1`).  v0 has no TT_OUT accounting."""
        if self.forming != "v1":
            raise ValueError(f"{self.name}: TT_OUT's accounting is stated for forming v1 only")
        return self.wref(shape) - self.prices.forming_a_only * shape.m * shape.k

    def job_cost(self, n: int, k: int) -> int:
        """The weight side of an (n, k) weight, once per weight and epoch: B's forming, B̃·F_A and G as chains, and P_B's
        FP32 FMA chain (r² per row at an add's price).  Neither W_ref nor the credit counts it (`creditDevRev1` leaves it
        out, which only raises γ)."""
        return (self._forming() * n * k + self._chain_units(n, R, k) + self._chain_units(R, R, k)
                + self.prices.fadd * n * R * R)

    def noise(self, ctx: Context) -> Noise:
        return Noise(unit_seed_a(ctx.salt, ctx.root_a, ctx.root_b, ctx.matmul.index, self.hashing),
                     seed_b(ctx.salt, ctx.root_b, self.hashing))

    def row_seeds(self, ctx: Context, a_rows: Sequence[bytes], idx: Sequence[int]) -> list[bytes]:
        """-h3's seeds of A's rows `idx` (values `a_rows`): `row_seed` over each row's `row_digest`, the inner digest of its
        hm96 leaf in A's tree, which needs no salt."""
        u = ctx.matmul
        return [row_seed(ctx.salt, row_digest(r), ctx.root_b, u.index, i, self.hashing) for i, r in zip(idx, a_rows, strict=True)]

    def _side(self, rows: Sequence[bytes], idx: Sequence[int], side: int, seed_e: bytes, seed_b: bytes, k: int,
              e: list[list[int]] | None = None) -> tuple[P.Built | Formed, list[list[int]]]:
        e = [sample_line(seed_e, side, 0, i) for i in idx] if e is None else e
        f = _basis(seed_b, side, k, self.line_norm)
        if self.forming == "v1":
            return form_v1([parse_f32_row(r, k) for r in rows], e, f, self.device), e
        planes = [P.parse_row(r, k) for r in rows]
        clean = [P.open_row(c, s) for c, s in planes]
        norms = [P.exact_norms(c, s) for c, s in planes]
        return P.noisy_quantize(P.H100, clean, norms, e, [list(x) for x in f]), e

    def encode(self, ctx: Context, noise: Noise, a_rows: Sequence[bytes], b_rows: Sequence[bytes], tile: Tile) -> Ops | None:
        k = ctx.matmul.shape.k
        try:
            rows_e = ([sample_line(s, 0, 0, 0) for s in self.row_seeds(ctx, a_rows, tile.rows)] if self.hashing in ROW_SEEDED
                      else None)
            a, e_a = self._side(a_rows, tile.rows, 0, noise.seed_a, noise.seed_b, k, rows_e)
            b, e_b = self._side(b_rows, tile.cols, 1, noise.seed_b, noise.seed_b, k)
            d = self.device
            return Ops(a, b, e_a, e_b, peel_factors_a(a.codes, scaled_noise(a, e_a), noise.seed_b, k, self.line_norm, d),
                       peel_factors_b(b.codes, scaled_noise(b, e_b), noise.seed_b, k, self.line_norm, d))
        except P.Reject:
            return None

    def admissible(self, ctx: Context, ops: Ops) -> bool:
        """v0: TT_OUT's domain is every input under δ = 1 with promotion every 4 atoms.  v1: every row of both operands passes
        TT_OUT's `OperandOK`, the noise floor ρ·α ≥ 1 (exact product) and `live_row` (per row, on the exact statistic).
        Pearl's census policy is not used."""
        if self.forming == "v0":
            return True
        return all(P.f32_of_bits(r) * P.f32_of_bits(a) >= P.SIGMA_MIN and live_row(row)
                   for side in (ops.a, ops.b) for r, a, row in zip(side.rho, side.alpha, side.x))

    def checked(self, ctx: Context, ops: Ops) -> Checked:
        d = self.device
        c = [[chain(ra, rb, d) for rb in ops.b.codes] for ra in ops.a.codes]
        u = [[peel(c[i][j], ops.pa[i], ops.pb[j], d) for j in range(len(ops.pb))] for i in range(len(ops.pa))]
        return Checked(c, u)

    def digest(self, ctx: Context, noise: Noise, checked: Checked) -> bytes:
        return leaf(checked, self.hashing)

    def tickets(self, noise: Noise, checked: Checked) -> list[bytes]:
        """Pearl's lottery tickets over C̃ (`tickets`)."""
        return tickets(noise.seed_a, checked, self.hashing)

    def useful(self, ctx: Context, noise: Noise, ops: Ops, checked: Checked) -> list[list[float]]:
        """U/(α_A,i · α_B,j) in double, for measuring quality.  It is not the served y: the served y is the integration's
        pinned y_unit = RN_f32(RN_f32(U · RN_f32(1/α_A)) · RN_f32(1/α_B)), two FP32 products in that order, then its glue."""
        value = P.bf16_to_f32 if self.forming == "v0" else P.f32_of_bits
        return [[P.f32_of_bits(w) / (value(ops.a.alpha[i]) * value(ops.b.alpha[j]))
                 for j, w in enumerate(row)] for i, row in enumerate(checked.u)]
