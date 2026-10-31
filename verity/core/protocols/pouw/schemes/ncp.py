"""``ncp-v1``: noise-cancelling polarization under TT_NCP(0.5%) (``PROTOCOL.md`` §NCP; Lean `Pouw.NCP`).

For int7 activations ``A`` (m x k) and weights ``B`` (committed as its n rows along k, so the product is ``A·Bᵀ``):

    E₁ ∈ [−31, 32]^(m x k)   per unit: row i = SHAKE256("verity/pouw/ncp/E1/v1", salt, index, root_A, weight, i)
    F₁ ∈ [−31, 32]^(k x n)   per (epoch, weight): column j = SHAKE256("verity/pouw/ncp/F1/v1", salt, weight, j)
    R                        a public signed permutation, R[π(j)][j] = σ_j (Lean `SignedPerm`)
    E₂ = E₁·R,  F₂ = −R·F₁,  E₃ = E₁ + E₂,  F₃ = F₁ + F₂
    X = [A + E₁ | A + E₂ | −(A + E₃)]   (m x 3k),   Y = [B + F₁ ; B + F₂ ; B + F₃]   (3k x n)

The checked values are route U's (Lean `Pouw.NCP.mmWU β` at β = 128): an accumulator started at c₀ = −β·Σ_l Y[l][j] that
adds (X + β)·Y, read after every 16-deep step, mod 2^32 (3k/16 per output entry).  X + β is u8 and Y is s8 (`rangeU`), so
this is the u8 x s8 tensor-core schedule the W1 accounting prices; the last value is A·B (`checkedU_final`, `wrapU`).  The
unbiased running sums (`checked_unbiased`, Lean `mmW`) are retired as checked values.  Each noise entry is one XOF byte's
low six bits minus 31, exactly uniform on [−31, 32].  A tile's digest is SHAKE256 over its checked words, little-endian, step by step and
row-major within a step (the order an accumulator produces them).

``ncp-v1`` uses the word-granular production P v1.1 (`word_perm`: it moves aligned 4-entry words and keeps the order
inside each, so forming E₂ is a free word move under W1 with byte moves priced).  It is certified
(independence rank T − 1, λ₃ = 32, MinRank 16 at k = 1,088, 2,048, 4,096 and 11,008); ``tests/test_pouw_word_p.py``
re-checks the rejection rule, the MinRank argument's hypotheses and λ₃ ≥ 16 at the k the MVP uses.  ``ncp-v1-shift24``
uses the shift by 24, admissible with independent checkpoints for every k >= 1024 with 16 | k (Lean
`Pouw.SecurityProofs.NCP.ShiftBy24Admissible`, `Pouw.SecurityProofs.NCP.IndependentWitness`), which the Lean calls a
witness, not a proposal.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from verity.ml.commitments import domains
from verity.ml.randomness import Part, frame

from ..protocol import Certificate, Context, DomainError, Matmul, Shape, Tile, grid_tiles

NAME = "ncp-v1"
NAME_SHIFT24 = "ncp-v1-shift24"
D = 16
M32 = (1 << 32) - 1
WORD_P_SEED = b"pous-pouw-ncp-Pword-v1"
#: Route U's bias: X + β is u8 on the int7 domain for 127 <= β <= 129 with an unsigned P (Lean `biasU8Iff`)
BETA = 128
#: f: W1 units per activation position for forming the operands on route U
FORM_UNITS = 24


def xof(tag: str, n: int, *parts: Part) -> bytes:
    return hashlib.shake_256(tag.encode() + b"\0" + b"".join(frame(p) for p in parts)).digest(n)


def noise_row(tag: str, k: int, *parts: Part) -> list[int]:
    return [(b & 63) - 31 for b in xof(tag, k, *parts)]


def shift_perm(k: int, c: int = 24) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The cyclic shift by c: π(j) = j + c mod k, every sign +1 (Lean `shiftBy`)."""
    return tuple((j + c) % k for j in range(k)), (1,) * k


def inside_cycles(p: Sequence[int], wps: int = 4) -> set[int]:
    """The words on a cycle of the arcs w -> p[p[w]] that stay inside w's step (a 2-cycle of p is such a cycle)."""
    nxt = {w: p[p[w]] for w in range(len(p)) if p[p[w]] // wps == w // wps}
    bad: set[int] = set()
    done: set[int] = set()
    for s in nxt:
        path, x = [], s
        while x in nxt and x not in done and x not in path:
            path.append(x)
            x = nxt[x]
        if x in path:
            bad.update(path[path.index(x):])
        done.update(path)
    return bad


def word_perm(k: int, seed: bytes = WORD_P_SEED, g: int = 4) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The word-granular production P, v1.1 (``pword-repair2.py``): a SHAKE256 Fisher-Yates over the k/g aligned words;
    repair 1 so no word lands in its own 16-step; repair 2 so no cycle of π² stays inside one step (hence no 2-cycle).
    The order inside each word is kept and every sign is +1.  Neither repair guards the rejection rule's other clauses or
    λ₃ ≥ 16, which are certified per k in ``tests/test_pouw_word_p.py``."""
    w, wps = k // g, D // g

    def words(s: bytes, n: int):
        buf = hashlib.shake_256(s).digest(4 * n)
        return iter(int.from_bytes(buf[4 * i:4 * i + 4], "little") for i in range(n))

    def uniform(ws, bound: int) -> int:
        limit = (1 << 32) - ((1 << 32) % bound)
        while True:
            r = next(ws)
            if r < limit:
                return r % bound

    ws, p = words(seed, 16 * w), list(range(w))
    for i in range(w - 1, 0, -1):
        j = uniform(ws, i + 1)
        p[i], p[j] = p[j], p[i]
    ws = words(seed + b"/repair", 64 * w)
    for j in range(w):
        while p[j] // wps == j // wps:
            r = uniform(ws, w)
            if p[r] // wps != j // wps and p[j] // wps != r // wps:
                p[j], p[r] = p[r], p[j]
    ws = words(seed + b"/repair2", 64 * w)
    try:
        while bad := inside_cycles(p, wps):
            j, r = min(bad), uniform(ws, w)
            if p[r] // wps != j // wps and p[j] // wps != r // wps:
                p[j], p[r] = p[r], p[j]
    except StopIteration:
        raise DomainError(f"the word P's second repair exhausts its stream at k = {k}") from None
    return tuple(g * p[a] + b for a in range(w) for b in range(g)), (1,) * k


def perm_digest(pi: Sequence[int]) -> str:
    """SHA-256 of the permutation's entries as 2-byte little-endian integers (the research store's convention)."""
    return hashlib.sha256(b"".join(x.to_bytes(2, "little") for x in pi)).hexdigest()


def admissible_perm(k: int, pi: Sequence[int], sigma: Sequence[int]) -> bool:
    """The rejection rule at d = 16 (Lean `Admissible`): no step meets its image, no step's image is a step, R² ≠ ±I."""
    for t in range(k // D):
        img = {pi[j] // D for j in range(t * D, (t + 1) * D)}
        if t in img or len(img) == 1:
            return False
    square_perm = all(pi[pi[j]] == j for j in range(k))
    return not (square_perm and len({sigma[j] * sigma[pi[j]] for j in range(k)}) == 1)


def int7_rows(rows: Sequence[bytes], k: int) -> list[list[int]] | None:
    out = []
    for r in rows:
        if len(r) != k:
            return None
        v = [x - 256 if x >= 128 else x for x in r]
        if any(not -64 <= x <= 63 for x in v):
            return None
        out.append(v)
    return out


def int7_bytes(values: Sequence[int]) -> bytes:
    """The row encoding: one two's-complement byte per int7 entry."""
    if any(not -64 <= x <= 63 for x in values):
        raise ValueError("int7 entries lie in [-64, 63]")
    return bytes(x & 0xFF for x in values)


@dataclass(frozen=True, slots=True)
class Ops:
    x: list[list[int]]      # the tile's rows of X (length 3k)
    y: list[list[int]]      # the tile's columns of Y (length 3k)


def assemble(a: list[list[int]], b: list[list[int]], e1: list[list[int]], f1: list[list[int]],
             pi: Sequence[int], sigma: Sequence[int]) -> Ops:
    """Lean `stackX` / `stackY` from explicit noise: `a`, `e1` are rows of A and E₁; `b`, `f1` are columns of B and F₁
    (B's committed rows)."""
    k = len(pi)
    x = []
    for ai, e in zip(a, e1):
        e2 = [sigma[j] * e[pi[j]] for j in range(k)]
        x.append([p + q for p, q in zip(ai, e)] + [p + q for p, q in zip(ai, e2)] + [-(p + q + r) for p, q, r in zip(ai, e, e2)])
    y = []
    for bj, f in zip(b, f1):
        f2 = [0] * k
        for l in range(k):
            f2[pi[l]] = -sigma[l] * f[l]
        y.append([p + q for p, q in zip(bj, f)] + [p + q for p, q in zip(bj, f2)] + [p + q + r for p, q, r in zip(bj, f, f2)])
    return Ops(x, y)


def checked_route_u(ops: Ops, beta: int = BETA) -> list[list[list[int]]]:
    """Route U's checked values (Lean `mmWU β`): the accumulator starts at c₀ = −β·Σ_l Y[l][j] and adds (X + β)·Y; entry
    ``[i][j][t]`` is its value after step t, mod 2^32.  The last is A·B (`checkedU_final`, `wrapU`)."""
    out = []
    for xi in ops.x:
        xb = [v + beta for v in xi]
        row = []
        for yj in ops.y:
            acc, sums = -beta * sum(yj), []
            for t in range(0, len(xb), D):
                acc += sum(p * q for p, q in zip(xb[t:t + D], yj[t:t + D]))
                sums.append(acc & M32)
            row.append(sums)
        out.append(row)
    return out


def checked_unbiased(ops: Ops) -> list[list[list[int]]]:
    """The unbiased running sums of X·Y (Lean `mmW`), mod 2^32: retired as NCP's checked values, kept for comparison."""
    return checked_route_u(ops, 0)


def route_u_shift(unbiased: list[list[list[int]]], y: Sequence[Sequence[int]], beta: int = BETA) -> list[list[list[int]]]:
    """Route U's values from the unbiased ones (`checkedU_eq`, checkpoints t = 1 … 3k/16): C^U_t = C_t − β·Σ_{l >= 16t} Y[l][j],
    mod 2^32."""
    out = []
    for row in unbiased:
        new = []
        for j, sums in enumerate(row):
            suffix, tail = [], 0
            for t in range(len(sums) - 1, -1, -1):
                suffix.append(tail)
                tail += sum(y[j][D * t:D * (t + 1)])
            new.append([(c - beta * s) & M32 for c, s in zip(sums, reversed(suffix))])
        out.append(new)
    return out


def stack(ctx: Context, pi: Sequence[int], sigma: Sequence[int], a: list[list[int]], b: list[list[int]], tile: Tile) -> Ops:
    u, k = ctx.matmul, ctx.matmul.shape.k
    e1 = [noise_row(domains.POUW_NCP_E1, k, ctx.salt, u.index, ctx.root_a, u.weight, i) for i in tile.rows]
    f1 = [noise_row(domains.POUW_NCP_F1, k, ctx.salt, u.weight, j) for j in tile.cols]
    return assemble(a, b, e1, f1, pi, sigma)


class NCP:
    """NCP at checked depth 16.  The lead domain is 16 | m, 64 | k, n, k in [1088, 2^16] (so γ_0 = 1/200 >= 16/(3k)) and
    n in [1600, 2^16] (Ω* = 1 + 8/1600 over the credited 3·m·k·n, so γ <= 1%).  Smaller bounds make a test instance,
    whose certificate has no γ."""

    def __init__(self, perm: str = "word", k_min: int = 1088, n_min: int = 1600, tile: tuple[int, int] = (16, 16)):
        if perm not in ("word", "shift24"):
            raise ValueError("perm is 'word' or 'shift24'")
        self.perm, self.k_min, self.n_min, self.tile = perm, k_min, n_min, tile
        self.name = NAME if perm == "word" else NAME_SHIFT24
        self.beta = BETA
        lead = k_min >= 1088 and n_min >= 1600
        self.certificate = Certificate(
            conjecture="tt-ncp-u",
            gamma0=Fraction(1, 200) if lead else None,
            omega=Fraction(3 * n_min + FORM_UNITS, 3 * n_min),
            claims=("tt-ncp-u", "random-oracle", "xof/shake-256", "cr/sha-256", "w1-cost-model", "live-verifier"),
            theorems=("Pouw.SecurityProofs.NCP.GammaFromTTNCP_U_v1", "Pouw.SecurityProofs.NCP.TTNCPUIff",
                      "Pouw.SecurityProofs.NCP.TTNCPUWitness", "Pouw.SecurityProofs.Theorem1",
                      "Pouw.SecurityProofs.WorkWeightedSampling", "Pouw.SecurityProofs.EndToEnd",
                      "Pouw.SecurityProofs.NCP.Cancel", "Pouw.SecurityProofs.NCP.Stacked",
                      "Pouw.SecurityProofs.NCP.CheckedUEq", "Pouw.SecurityProofs.NCP.CheckedUFinal",
                      "Pouw.SecurityProofs.NCP.WrapU", "Pouw.SecurityProofs.NCP.RangeU", "Pouw.SecurityProofs.NCP.MmWUExact",
                      "Pouw.SecurityProofs.NCP.YUChecked", "Pouw.SecurityProofs.NCP.Int7Int32UPos",
                      "Pouw.SecurityProofs.NCP.NCPWordFloorIndU", "Pouw.SecurityProofs.NCP.WrapFinalOnlyIndU",
                      "Pouw.SecurityProofs.NCP.WrapWordFloorIndU", "Pouw.SecurityProofs.Barrier.Generic",
                      *(("Pouw.SecurityProofs.NCP.IndependentWitness", "Pouw.SecurityProofs.NCP.ShiftBy24Admissible")
                        if perm == "shift24" else ())),
            hypotheses=("TTNCP_U(β, γ_0) in M_4090 under W1 with sub-word moves priced (Pouw.NCP.Assumptions.TTNCP_U)",
                        "AdmitsRef: H_ref runs within wref",
                        "Word.NzNCP of the wrapped route-U values (the word floors' hypothesis)",
                        *(("P v1.1's checkpoint independence and MinRank 16 at each k (script-certified; not in Lean)",)
                          if perm == "word" else ())),
            allowances=("the free final value, d/(3k) per output entry, inside γ_0",),
            parameters=(("beta", BETA), ("f", FORM_UNITS), ("n0", n_min), ("k_max", 1 << 16)),
        )

    def check(self, matmul: Matmul) -> None:
        s = matmul.shape
        if s.m % 16 or s.k % 64 or s.n % 64 or not self.k_min <= s.k <= 1 << 16 or not self.n_min <= s.n <= 1 << 16:
            raise DomainError(f"{self.name}: {s} is outside 16 | m, 64 | k, n, {self.k_min} <= k <= 2^16, {self.n_min} <= n <= 2^16")
        self._perm(self.perm, s.k)

    def tiles(self, shape: Shape) -> list[Tile]:
        return grid_tiles(shape, *self.tile)

    def wref(self, shape: Shape) -> int:
        """Route U: the 3k-deep int8 GEMM and 24 units per activation position for forming the operands.  The per-(epoch,
        weight) forming of B + F_i is apportioned over the weight's rows and left out."""
        return 3 * shape.m * shape.k * shape.n + FORM_UNITS * shape.m * shape.k

    @staticmethod
    @lru_cache(maxsize=16)
    def _perm(kind: str, k: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        pi, sigma = word_perm(k) if kind == "word" else shift_perm(k)
        if not admissible_perm(k, pi, sigma):
            raise DomainError(f"the {kind} permutation fails the rejection rule at k = {k}")
        return pi, sigma

    def noise(self, ctx: Context) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self._perm(self.perm, ctx.matmul.shape.k)

    def encode(self, ctx: Context, noise, a_rows: Sequence[bytes], b_rows: Sequence[bytes], tile: Tile) -> Ops | None:
        k = ctx.matmul.shape.k
        a, b = int7_rows(a_rows, k), int7_rows(b_rows, k)
        return None if a is None or b is None else stack(ctx, *noise, a, b, tile)

    def admissible(self, ctx: Context, ops: Ops) -> bool:
        return True

    def checked(self, ctx: Context, ops: Ops) -> list[list[list[int]]]:
        """``checked[i][j][t]``: entry (i, j)'s route-U accumulator after step t, mod 2^32 (`checked_route_u`)."""
        return checked_route_u(ops, self.beta)

    def digest(self, ctx: Context, noise, checked: list[list[list[int]]]) -> bytes:
        steps = len(checked[0][0]) if checked and checked[0] else 0
        words = b"".join(checked[i][j][t].to_bytes(4, "little")
                         for t in range(steps) for i in range(len(checked)) for j in range(len(checked[i])))
        return hashlib.shake_256(domains.POUW_NCP_TILE.encode() + b"\0" + words).digest(32)

    def useful(self, ctx: Context, noise, ops: Ops, checked) -> list[list[int]]:
        return [[s[-1] - (1 << 32) if s[-1] >> 31 else s[-1] for s in row] for row in checked]
