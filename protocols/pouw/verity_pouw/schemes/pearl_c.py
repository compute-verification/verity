"""``pearl-c-h100-v0``: Pearl-C, Pearl's FP8 product with its noisy clean-up checked, on the H100.

The design is the cheap-binding lane's (research store ``docs/pouw/cheap-binding.md`` §1): Pearl's H100 product and its
output tickets, plus #311's noisy clean-up ("peel") pinned bit for bit, so that the useful output U is a checked value
and no part of the clean-up can be skipped at a degenerate input.  γ rests on the named conjecture TT_OUT(1/400); nothing
here is proved by counting.  Pearl's own pieces (noise lines, BF16 compute ops, norms, the noisy quantization, the H100
reset-and-promote replay) are `pearl_kw`'s ports of pearl-research-labs/pearl PR #311 and its ``zk-pow`` verifier.

**The scheme, per 64 x 64 tile** (rows I of A, rows J of B, full k; both operands stored as rows along k):

~~~text
seeds     seed_B = H_"pearl-c/v0/seed-B"(salt ‖ root_B);  seed_A = H_"pearl-c/v0/seed-A"(salt ‖ root_A ‖ root_B ‖ index u64 LE)
          H_ℓ(x) = BLAKE3(x; key = BLAKE3(ℓ)).  E_A lines by seed_A (row i); E_B lines and the bases F_A, F_B (lines
          0 … k−1) by seed_B: Pearl's line rule (`pearl_kw.sample_line`), so F_A, E_B, F_B are per job (weight)
forming  v0, #311's as ported (`pearl_kw`): FP10 rows -> BF16 X, exact norms, α, β, N = E·Fᵀ on the H100 atom,
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
tickets   Pearl's lottery over C̃: per 16 x 16 subtile, the 16-lane fold (`pearl_kw.extract`) hashed as H_"jackpot"(z; seed_A)
useful    U_ij / (α_A,i · α_B,j), the model's own unscale (glue)
~~~

**The clean-up's algebra.**  With N_A = E_A·F_Aᵀ and N_B = E_B·F_Bᵀ, A′ ≈ α_A·A + β_A⊙N_A and B̃ ≈ α_B·B + β_B⊙N_B, and
(A′ − β_A⊙N_A)(B̃ − β_B⊙N_B)ᵀ = C̃ + (β_A⊙E_A)·[(β_B⊙E_B)·G − B̃·F_A]ᵀ + (A′·F_B)·(−β_B⊙E_B)ᵀ with G = F_Bᵀ·F_A, which is
the rank-64 BF16 product P_A·P_Bᵀ.  #311 runs G and the B-side factor on a tolerance path; here both are pinned.

**Forming v0 is the port, not the design's.**  The design (§1) replaces the model's FP8 quantize with the noisy one on the
previous layer's BF16 output, leaving 2–32 units per element that depend on A alone.  v0 keeps #311's FP10 decode and
exact norms, whose 176–256 A-only units the red team counts as skippable (X-PC-2).  The rework swaps in when it is specced.
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
from verity.ml.tc.models import HOPPER_BF16_WGMMA_K16, HOPPER_E4M3_K32

from ..protocol import Certificate, Context, DomainError, Matmul, Shape, Tile, grid_tiles
from . import pearl_kw as P

NAME = "pearl-c-h100-v0"
R = P.R
WINDOW = P.WINDOW                  # 128 products: four k32 atoms per promotion
SLICE = 32
PEEL_K = 2 * R                     # the peel's BF16 depth, four k16 slices
TILE = 64
CHUNK = 32                         # columns per ticket message
D_MESSAGE, D_LEAF = 0x21, 0x22     # TurboSHAKE128 domain bytes
LABELS = {r: f"pearl-c/v0/{r}".encode() for r in ("seed-A", "seed-B")}
#: W1 units (1 = one dense FP8 MAC on the H100): FADD 32 per word, a BF16 MAC 2
FADD_UNITS, BF16_MAC_UNITS = 32, 2
#: Forming v0 (#311 as ported) per operand element: the red team's 240–320 for #311's port (X-PC-2), at the upper end
FORMING_V0_UNITS = 320


def _labelled(message: bytes, role: str) -> bytes:
    return keyed(blake3(LABELS[role]), message)


def _word(x: float) -> int:
    return P.f32_bits(x)


def _check_finite(word: int) -> int:
    if (word >> 23) & 0xFF == 0xFF:
        raise P.Reject("non-finite FP32 accumulator")
    return word


def chain(a: Sequence[int], b: Sequence[int]) -> int:
    """The H100 replay of ``a · b`` (E4M3 codes) as an FP32 word: per 128-product window a fresh +0 accumulator chains up
    to four HOPPER_E4M3_K32 atoms, then one FP32 nearest-even add promotes it into the total; a zero total is +0."""
    if len(a) != len(b) or len(a) % SLICE:
        raise P.Reject("the chain multiplies in atoms of k = 32")
    total = 0
    for ws in range(0, len(a), WINDOW):
        local = 0
        for s in range(ws, min(ws + WINDOW, len(a)), SLICE):
            local = _check_finite(HOPPER_E4M3_K32.step(local, a[s:s + SLICE], b[s:s + SLICE]))
        total = _check_finite(fp32.add(total, local))
        if total == 0x80000000:
            total = 0
    return total


def peel(c_word: int, pa: Sequence[int], pb: Sequence[int]) -> int:
    """U from C̃: four HOPPER_BF16_WGMMA_K16 steps over the 64 BF16 peel operands, k16 slices in order, into C̃'s word."""
    acc = c_word
    for s in range(0, PEEL_K, 16):
        acc = _check_finite(HOPPER_BF16_WGMMA_K16.step(acc, pa[s:s + 16], pb[s:s + 16]))
    return acc


def _bf16_of_word(word: int) -> int:
    return P.f32_to_bf16(P.f32_of_bits(word))


def _bf16_scaled(beta: int, code: int) -> int:
    """bf16(β · e) for an E4M3 noise code e (Pearl's BF16 compute semantics: exact decode, one f32 product, RNE)."""
    return P.bf16_mul(beta, P.f32_to_bf16(P.fp8_to_f32(code)))


@dataclass(frozen=True, slots=True)
class Noise:
    seed_a: bytes
    seed_b: bytes


@dataclass(slots=True)
class Ops:
    """One tile's formed operands and peel factors."""

    a: P.Built                      # A′ = a.codes, with α_A, β_A
    b: P.Built                      # B̃ = b.codes, with α_B, β_B
    e_a: list[list[int]]            # E_A lines of the tile's rows
    e_b: list[list[int]]            # E_B lines of the tile's columns
    pa: list[list[int]]             # P_A rows, 64 BF16 words
    pb: list[list[int]]             # P_B rows, 64 BF16 words


@dataclass(slots=True)
class Checked:
    """The tile's checked words: C̃ (the tickets' accumulator) and U (the useful output before its unscale)."""

    c: list[list[int]]
    u: list[list[int]]


@lru_cache(maxsize=16)
def _basis(seed_b: bytes, side: int, k: int) -> tuple[tuple[int, ...], ...]:
    """F_A (side 0) or F_B (side 1) as k lines of 32 codes (line t is row t of the k x r basis)."""
    return tuple(tuple(P.sample_line(seed_b, side, 1, t)) for t in range(k))


def _column(basis: tuple[tuple[int, ...], ...], p: int) -> list[int]:
    return [line[p] for line in basis]


@lru_cache(maxsize=16)
def gram(seed_b: bytes, k: int) -> tuple[tuple[int, ...], ...]:
    """G = F_Bᵀ·F_A (r x r) as FP32 words, G_pq = chain(F_B[:, p], F_A[:, q]); once per job."""
    fa, fb = _basis(seed_b, 0, k), _basis(seed_b, 1, k)
    fa_cols = [_column(fa, q) for q in range(R)]
    return tuple(tuple(chain(_column(fb, p), fa_cols[q]) for q in range(R)) for p in range(R))


def peel_a(built: P.Built, e_rows: Sequence[Sequence[int]], seed_b: bytes, k: int) -> list[list[int]]:
    """P_A rows: [bf16(β_A·E_A) ‖ RNE_bf16(A′·F_B)], the second half the 32 fused columns of the product."""
    fb = _basis(seed_b, 1, k)
    fb_cols = [_column(fb, p) for p in range(R)]
    return [[_bf16_scaled(beta, e) for e in e_row] + [_bf16_of_word(chain(codes, fb_cols[p])) for p in range(R)]
            for codes, beta, e_row in zip(built.codes, built.beta, e_rows)]


def peel_b(built: P.Built, e_rows: Sequence[Sequence[int]], seed_b: bytes, k: int) -> list[list[int]]:
    """P_B rows: [RNE_bf16((β_B·E_B)·G − B̃·F_A) ‖ −bf16(β_B·E_B)], once per job.  (β_B·E_B)·G is the FP32 FMA chain from
    +0 in p order, and the difference one FP32 nearest-even add of the negated B̃·F_A."""
    fa = _basis(seed_b, 0, k)
    fa_cols = [_column(fa, q) for q in range(R)]
    g = gram(seed_b, k)
    rows = []
    for codes, beta, e_row in zip(built.codes, built.beta, e_rows):
        be = [_bf16_scaled(beta, e) for e in e_row]
        mid = []
        for q in range(R):
            acc = 0
            for p in range(R):
                acc = _check_finite(fp32.fma(be[p] << 16, g[p][q], acc))
            mid.append(_bf16_of_word(_check_finite(fp32.add(acc, fp32.neg(chain(codes, fa_cols[q]))))))
        rows.append(mid + [x ^ 0x8000 for x in be])
    return rows


def leaf(checked: Checked) -> bytes:
    """TurboSHAKE128 over the per-(row, 32-column chunk) message digests, row-major, chunks in order."""
    digests = []
    for crow, urow in zip(checked.c, checked.u):
        for h in range(0, len(crow), CHUNK):
            msg = struct.pack(f"<{len(crow[h:h + CHUNK])}I", *crow[h:h + CHUNK]) + \
                struct.pack(f"<{len(urow[h:h + CHUNK])}I", *urow[h:h + CHUNK])
            digests.append(turboshake128(msg, D_MESSAGE, 32))
    return turboshake128(b"".join(digests), D_LEAF, 32)


class PearlC:
    """Pearl-C on the H100: ``k_min`` = 1024 (Pearl's domain) and 64 x 64 tiles; smaller bounds make a test instance."""

    def __init__(self, k_min: int = 1024, tile: tuple[int, int] = (TILE, TILE)):
        self.name = NAME
        self.k_min, self.tile = k_min, tile
        self.certificate = Certificate(
            conjecture="tt-out",
            gamma0=None,
            omega=Fraction(1),
            claims=("tt-out", "random-oracle", "cr/blake3", "device-reproducibility", "beacon-unpredictability"),
            theorems=("Pouw.Proofs.workWeightedSampling",),
            hypotheses=(
                "TT_OUT(1/400) for Pearl-C's Π and domain (δ = 1, promotion every 4 atoms, k <= 2^16): no H100 W1 program "
                "outputs correct C̃ and U words for less than 99.75% of their honest cost (research store "
                "docs/pouw/cheap-binding.md §1); a conjecture, not proved by counting",
                "the H100's FP8 wgmma and BF16 wgmma are reproduced exactly (HOPPER_E4M3_K32, HOPPER_BF16_WGMMA_K16)",
            ),
            allowances=(
                "forming v0 is #311's as ported (FP10 decode and exact norms depend on A alone and are skippable, X-PC-2); "
                "the reworked forming replaces it",
                "the unscale is glue outside the checked values",
            ),
            parameters=(("k_min", k_min), ("tile_m", tile[0]), ("tile_n", tile[1]), ("rank", R), ("window", WINDOW)),
        )

    def check(self, matmul: Matmul) -> None:
        s = matmul.shape
        if not (self.k_min <= s.k <= 1 << 16 and s.k % SLICE == 0 and s.m % self.tile[0] == 0 and s.n % self.tile[1] == 0
                and s.m <= 1 << 24 and s.n <= 1 << 24):
            raise DomainError(f"{NAME}: {s} is outside {self.k_min} <= k <= 2^16 with 32 | k, tiles of {self.tile}, "
                              f"m, n <= 2^24")

    def tiles(self, shape: Shape) -> list[Tile]:
        return grid_tiles(shape, *self.tile)

    def wref(self, shape: Shape) -> int:
        """The honest W1 cost of one unit: the chain (1 per MAC) with its promotion FADDs, the fused A′·F_B, the peel in
        BF16 (2 per MAC), B̃·F_A and B's forming (per job, charged here in full) and forming v0 on both sides.  Hashing is
        the random oracle's."""
        m, k, n = shape.m, shape.k, shape.n
        windows = -(-k // WINDOW)
        chain_units = lambda rows, cols: rows * cols * (k + FADD_UNITS * windows)
        return (chain_units(m, n) + chain_units(m, R) + chain_units(n, R) + BF16_MAC_UNITS * PEEL_K * m * n
                + FORMING_V0_UNITS * (m + n) * k)

    def noise(self, ctx: Context) -> Noise:
        seed_b = _labelled(ctx.salt + ctx.root_b, "seed-B")
        seed_a = _labelled(ctx.salt + ctx.root_a + ctx.root_b + ctx.matmul.index.to_bytes(8, "little"), "seed-A")
        return Noise(seed_a, seed_b)

    def _side(self, rows: Sequence[bytes], idx: Sequence[int], side: int, seed_e: bytes, seed_b: bytes, k: int
              ) -> tuple[P.Built, list[list[int]]]:
        planes = [P.parse_row(r, k) for r in rows]
        clean = [P.open_row(c, s) for c, s in planes]
        norms = [P.exact_norms(c, s) for c, s in planes]
        e = [P.sample_line(seed_e, side, 0, i) for i in idx]
        return P.noisy_quantize(P.H100, clean, norms, e, [list(x) for x in _basis(seed_b, side, k)]), e

    def encode(self, ctx: Context, noise: Noise, a_rows: Sequence[bytes], b_rows: Sequence[bytes], tile: Tile) -> Ops | None:
        k = ctx.matmul.shape.k
        try:
            a, e_a = self._side(a_rows, tile.rows, 0, noise.seed_a, noise.seed_b, k)
            b, e_b = self._side(b_rows, tile.cols, 1, noise.seed_b, noise.seed_b, k)
            return Ops(a, b, e_a, e_b, peel_a(a, e_a, noise.seed_b, k), peel_b(b, e_b, noise.seed_b, k))
        except P.Reject:
            return None

    def admissible(self, ctx: Context, ops: Ops) -> bool:
        """TT_OUT's domain is every input under δ = 1 with promotion every 4 atoms; Pearl's jackpot policy is not used."""
        return True

    def checked(self, ctx: Context, ops: Ops) -> Checked:
        c = [[chain(ra, rb) for rb in ops.b.codes] for ra in ops.a.codes]
        u = [[peel(c[i][j], ops.pa[i], ops.pb[j]) for j in range(len(ops.pb))] for i in range(len(ops.pa))]
        return Checked(c, u)

    def digest(self, ctx: Context, noise: Noise, checked: Checked) -> bytes:
        return leaf(checked)

    def tickets(self, noise: Noise, checked: Checked) -> list[bytes]:
        """Pearl's lottery tickets over C̃: one per 16 x 16 subtile, row-major, H_"jackpot"(fold; seed_A)."""
        out = []
        for i in range(0, len(checked.c), P.TILE):
            for j in range(0, len(checked.c[0]), P.TILE):
                sub = [[P.f32_of_bits(w) for w in row[j:j + P.TILE]] for row in checked.c[i:i + P.TILE]]
                out.append(P.hash_labelled(P.extract(sub), "jackpot", noise.seed_a))
        return out

    def useful(self, ctx: Context, noise: Noise, ops: Ops, checked: Checked) -> list[list[float]]:
        """U unscaled by the model's own α_A,i · α_B,j (glue)."""
        return [[P.f32_of_bits(w) / (P.bf16_to_f32(ops.a.alpha[i]) * P.bf16_to_f32(ops.b.alpha[j]))
                 for j, w in enumerate(row)] for i, row in enumerate(checked.u)]
