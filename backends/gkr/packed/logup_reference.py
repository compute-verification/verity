"""logUp fractional-sum GKR (PROTOCOL.md 4.2; ``backends/gkr/src/logup.rs`` transliterated), pure Python on
:mod:`reference`'s BabyBear^6.  The oracle for the GPU LogUp kernels (:mod:`logup_gpu`).

For table ``T`` with queries ``w_j`` and multiplicities ``m_v``: ``sum_j 1/(z - w_j) = sum_v m_v/(z - t_v)``.  Leaves
``(p, q)``: witness ``(1, z - w_j)``, table ``(-m_v, z - t_v)``, padding ``(0, 1)``; layer ``k`` combines pairs by
``(p0 q1 + p1 q0, q0 q1)``; the root numerator must be 0.  No division anywhere.  Vector-valued rows are compressed with
powers of ``beta``.

**Index convention.**  ``logup.rs`` binds the *top* bit of the level index first (``v[x], v[x + half]``;
``field::eq_table`` puts ``point[0]`` on the top bit).  Everything here binds the *low* bit first (pairs ``(2y, 2y+1)``,
``reference.fold``; ``reference.eq_table`` puts ``rho[0]`` on bit 0) because that is the layout the packed kernels
stream.  The two are the same protocol under bit-reversal of the level index: the message sequence produced here on a
level table equals ``logup.rs`` on the bit-reversed table, with the same coins; the claim point is ``rho = [mu] + x*``
here (``mu`` binds bit 0, the children pairing) where ``logup.rs`` writes ``x* ++ [mu]`` (its ``mu`` binds the top
bit) -- the same vector read in the respective bit convention.  The children pairing ``p0 = p[2x], p1 = p[2x + 1]`` is
the same in both.

Coins come from a :class:`Coins` object (a deterministic stand-in for the verifier; not Fiat-Shamir) so the prover and
the verifier draw the same sequence: ``lambda_0``, then per level ``k``: ``r_1..r_k``, ``mu``, ``lambda``.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from . import reference as ref

Ext = ref.Ext


class Coins:
    """Deterministic challenge sequence (the verifier's coins), drawn in protocol order."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def next(self) -> Ext:
        return ref.random_ext(self.rng)


def neg(a: Ext) -> Ext:
    return tuple((-x) % ref.P for x in a)


# ---------------------------------------------------------------------------------------------------------------------
# leaves and tree
# ---------------------------------------------------------------------------------------------------------------------


def log2ceil(n: int) -> int:
    s = 0
    while (1 << s) < n:
        s += 1
    return s


def multiplicities(queries: Sequence[int], table: Sequence[int]) -> list[int]:
    """``m_v`` = how many queries equal row ``v`` (single-column table; queries not in the table count nowhere)."""
    idx = {t: i for i, t in enumerate(table)}
    m = [0] * len(table)
    for w in queries:
        i = idx.get(w)
        if i is not None:
            m[i] += 1
    return m


def leaves(queries: Sequence[int], table: Sequence[int], mult: Sequence[int], z: Ext) -> tuple[list[Ext], list[Ext], int]:
    """Single-column table.  ``(p, q, n)`` with ``2^n`` leaves: witness ``(1, z - w)``, table ``(-m, z - t)``, pad ``(0, 1)``."""
    n = log2ceil(len(queries) + len(table))
    size = 1 << n
    p = [ref.ZERO] * size
    q = [ref.ONE] * size
    for j, w in enumerate(queries):
        p[j] = ref.ONE
        q[j] = ref.sub(z, ref.base(w))
    base = len(queries)
    for i, (t, m) in enumerate(zip(table, mult)):
        p[base + i] = neg(ref.base(m))
        q[base + i] = ref.sub(z, ref.base(t))
    return p, q, n


def compress(row: Sequence[int], beta: Ext) -> Ext:
    """``sum_c beta^c row[c]`` (Hab22 3.4)."""
    acc, b = ref.ZERO, ref.ONE
    for v in row:
        acc = ref.add(acc, ref.scale(b, v % ref.P))
        b = ref.mul(b, beta)
    return acc


def leaves_multi(queries: Sequence[Sequence[int]], table: Sequence[Sequence[int]], mult: Sequence[int], z: Ext, beta: Ext
                 ) -> tuple[list[Ext], list[Ext], int]:
    """Vector-valued rows: the compressed value replaces ``w`` / ``t``."""
    n = log2ceil(len(queries) + len(table))
    size = 1 << n
    p = [ref.ZERO] * size
    q = [ref.ONE] * size
    for j, row in enumerate(queries):
        p[j] = ref.ONE
        q[j] = ref.sub(z, compress(row, beta))
    base = len(queries)
    for i, (row, m) in enumerate(zip(table, mult)):
        p[base + i] = neg(ref.base(m))
        q[base + i] = ref.sub(z, compress(row, beta))
    return p, q, n


def tree(p_leaf: Sequence[Ext], q_leaf: Sequence[Ext], n: int) -> list[tuple[list[Ext], list[Ext]]]:
    """``levels[k]`` has ``2^k`` fractions; ``levels[n]`` = the leaves, ``levels[0]`` = the root."""
    levels = [(list(p_leaf), list(q_leaf))]
    for _ in range(n):
        p, q = levels[-1]
        half = len(p) // 2
        np_ = [ref.add(ref.mul(p[2 * x], q[2 * x + 1]), ref.mul(p[2 * x + 1], q[2 * x])) for x in range(half)]
        nq = [ref.mul(q[2 * x], q[2 * x + 1]) for x in range(half)]
        levels.append((np_, nq))
    levels.reverse()
    return levels


# ---------------------------------------------------------------------------------------------------------------------
# prover
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class LeafClaim:
    rho: list[Ext]
    lam: Ext
    value: Ext


@dataclass
class Transcript:
    """Prover messages in order: ``q_root``, then per level ``k``: ``k`` round polynomials (4 values each) and the 4 final values."""

    q_root: Ext
    levels: list[dict] = field(default_factory=list)   # {"rounds": [[S0..S3], ...], "vals": [p0, p1, q0, q1]}


def round_poly(p0: Sequence[Ext], p1: Sequence[Ext], q0: Sequence[Ext], q1: Sequence[Ext], e: Sequence[Ext], lam: Ext) -> list[Ext]:
    """``sum_y e_y(t) [p0_y(t) q1_y(t) + p1_y(t) q0_y(t) + lambda q0_y(t) q1_y(t)]`` at ``t = 0..3`` over pairs ``(2y, 2y+1)``."""
    half = len(e) // 2
    out = []
    for t in range(4):
        s = ref.ZERO
        for y in range(half):
            a, b, c, d, ee = (ref.eval_at(v[2 * y], v[2 * y + 1], t) for v in (p0, p1, q0, q1, e))
            u = ref.add(ref.add(ref.mul(a, d), ref.mul(b, c)), ref.mul(lam, ref.mul(c, d)))
            s = ref.add(s, ref.mul(ee, u))
        out.append(s)
    return out


def prove_table(p_leaf: Sequence[Ext], q_leaf: Sequence[Ext], n: int, coins: Coins) -> tuple[Transcript, LeafClaim]:
    """``logup.rs::prove_table`` with low-bit-first binding.  Returns the messages and the residual leaf claim."""
    levels = tree(p_leaf, q_leaf, n)
    p_root, q_root = levels[0][0][0], levels[0][1][0]
    if p_root != ref.ZERO:
        raise ValueError("honest root numerator is not zero: the multisets differ")
    tr = Transcript(q_root)
    lam = coins.next()
    rho: list[Ext] = []
    for k in range(n):
        pc, qc = levels[k + 1]
        p0, p1 = pc[0::2], pc[1::2]
        q0, q1 = qc[0::2], qc[1::2]
        e = ref.eq_table(rho)
        rounds, x_star = [], []
        for _ in range(k):
            h = round_poly(p0, p1, q0, q1, e, lam)
            rounds.append(h)
            r = coins.next()
            x_star.append(r)
            p0, p1, q0, q1, e = (ref.fold(v, r) for v in (p0, p1, q0, q1, e))
        vals = [p0[0], p1[0], q0[0], q1[0]]
        tr.levels.append({"rounds": rounds, "vals": vals})
        mu = coins.next()
        lam = coins.next()
        rho = [mu] + x_star          # mu binds bit 0 (the children pairing); x_star the bits above it
    p_next = ref.add(vals[0], ref.mul(mu, ref.sub(vals[1], vals[0]))) if n else ref.ZERO
    q_next = ref.add(vals[2], ref.mul(mu, ref.sub(vals[3], vals[2]))) if n else q_root
    return tr, LeafClaim(rho, lam, ref.add(p_next, ref.mul(lam, q_next)))


# ---------------------------------------------------------------------------------------------------------------------
# verifier
# ---------------------------------------------------------------------------------------------------------------------


def verify_table(n: int, tr: Transcript, coins: Coins) -> LeafClaim:
    """``logup.rs::verify_table``; raises ``ValueError`` on a failed check."""
    if tr.q_root == ref.ZERO:
        raise ValueError("zero root denominator")
    lam = coins.next()
    rho: list[Ext] = []
    claim = ref.mul(lam, tr.q_root)          # p_root = 0 asserted
    if len(tr.levels) != n:
        raise ValueError("level count")
    for k in range(n):
        lv = tr.levels[k]
        running = claim
        x_star = []
        if len(lv["rounds"]) != k:
            raise ValueError(f"level {k}: round count")
        for rd, h in enumerate(lv["rounds"]):
            if ref.add(h[0], h[1]) != running:
                raise ValueError(f"level {k} round {rd}: sum mismatch")
            r = coins.next()
            running = ref.lagrange_0123(h, r)
            x_star.append(r)
        p0, p1, q0, q1 = lv["vals"]
        e_star = ref.eq_points(rho, x_star)
        if ref.mul(e_star, ref.add(ref.add(ref.mul(p0, q1), ref.mul(p1, q0)), ref.mul(lam, ref.mul(q0, q1)))) != running:
            raise ValueError(f"level {k}: final identity")
        mu = coins.next()
        lam = coins.next()
        rho = [mu] + x_star          # mu binds bit 0 (the children pairing); x_star the bits above it
        claim = ref.add(ref.add(p0, ref.mul(mu, ref.sub(p1, p0))), ref.mul(lam, ref.add(q0, ref.mul(mu, ref.sub(q1, q0)))))
    return LeafClaim(rho, lam, claim)


def check_leaves(p_leaf: Sequence[Ext], q_leaf: Sequence[Ext], c: LeafClaim) -> bool:
    pv = ref.multilinear_eval(p_leaf, c.rho)
    qv = ref.multilinear_eval(q_leaf, c.rho)
    return ref.add(pv, ref.mul(c.lam, qv)) == c.value
