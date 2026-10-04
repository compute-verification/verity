"""V*'s algebra: the inner verifier's algebraic checks on one rep of a C-Flock session (`verifier/PROTOCOL.md` §9.2, §10, §12,
§13), at the session's coins, as residuals over F that are all zero iff the checks hold.

With the coins values, every check is linear in the rep's messages with coefficients that are functions of the coins and the
public statement, except two products of messages (zerocheck's ``v_A · v_B``, lincheck's ``comb · z_partial``). So each check is

    residual = Σ_k coef_k · w_k  +  const  +  Σ (message · message)

and V* computes it with one GF(2^128) product per term (`verity.ml.boolean.gf2k.GfResiduals`), the coefficients and constants
being verifier rows. `run` is the inner verifier's algebra written once over values: on ints it is the reference (`residuals`);
on symbolic messages (`Lin`) it yields the coefficients, from which `structure` reads the circuit and `verifier_rows` its
coefficient inputs. Merkle openings, caps and the transcript's framing are rec-v0's; the comb's hidden part is rec-amat's.

**Ports of `InnerRepCheck` = ``GfResiduals_v1{S = structure(shape)}``** (`structure`'s `Layout` names every slot):

    w   the rep's free message elements in protocol order (`free_names`: zerocheck's round1_ab, round1_c, rounds, v_A, v_B;
        lincheck's rounds, z_partial; each claim's ring switch s; Ligerito's OOD values, messages (u0, u2) and yr as they are
        absorbed), then the consistency sums (per Ligerito level, per lane l, Σ_j eq(α)[j] · row_j[l] over the level's
        opened rows: `GfScale`'s outputs summed), then the comb's hidden part (64)
    v   per residual, its coefficients in its terms' order and its constant, then the comb's public part (64)
    out zerocheck, lincheck, one per claim (its ring switch), Ligerito's final check (two: K's c0, c1)

``z_partial``'s comb is ``comb_pub[l] + comb_hid[l]``: `run` takes it whole (the circuit fold of §10 at the rep's coins), V* as
the sum of its two inputs. The zerocheck's ``v_C`` is never absorbed: V* takes it as ``Σ_s L_s(z) round1_c[s]``, the value the
inner verifier requires it to equal (`CEvalMismatch`). Ligerito's ``T`` is the verifier's own value, not a message. Ligerito's
last message is absorbed but enters no check (no claim evaluates it): no residual reads its four slots, and the Lean verifier
accepts a session whose last message alone was changed.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from verity.ml.boolean import gf2k as G
from verity.ml.boolean.gf2k import XINV, inv, kmul, mul, phi8

__all__ = ["Claim", "Coins", "Layout", "Lin", "Proof", "Schedule", "Shape", "decode", "fast100", "free_names", "messages",
           "query_rows", "rep_rounds", "residuals", "run", "structure", "symbolic", "verifier_rows", "witness"]


# --- §15: the fast100 schedule ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Level:
    rate: int
    q: int
    cols: int
    d: int
    strata: tuple[int, ...]
    cap: int
    lanes: int


@dataclass(frozen=True)
class Schedule:
    m: int
    n: int
    k0: int
    levels: tuple[Level, ...]
    ext: int


QUERIES = (218, 106, 71, 53, 43, 36, 32)


def _decompose(q: int, d: int) -> tuple[int, ...]:
    out, rem = [], q
    while rem:
        c = min(rem.bit_length() - 1, d)
        out.append(c)
        rem -= 1 << c
    return tuple(out)


def fast100(m: int) -> Schedule:
    if not 22 <= m <= 35:
        raise ValueError(f"m = {m}: no fast100 schedule")
    nl = 3 if m <= 24 else 4 if m <= 27 else 5 if m <= 30 else 6 if m <= 33 else 7
    k0 = 4 if m == 28 else 5 if m == 29 else 6
    n = m - 7
    ls, cols = [], n - k0
    for l in range(nl):
        rate, q = (4, 53) if m == 22 and l == 2 else (l + 1, QUERIES[l] if l < len(QUERIES) else 32)
        d = cols + rate
        st = _decompose(q, d)
        ls.append(Level(rate, q, cols, d, st, st[0], k0 if l == 0 else 4))
        cols -= 3
    return Schedule(m, n, k0, tuple(ls), ls[-1].cols)


@dataclass(frozen=True)
class Shape:
    """What a rep's algebra depends on: the table's m and k_log, the number of claims (2 + the statement's extra claims), and
    the Merkle scheme's digest and salt widths (hm96-sha512: 64 and 192 bytes)."""

    m: int
    k_log: int
    claims: int
    hash_len: int = 64
    salt_len: int = 192

    @property
    def sched(self) -> Schedule:
        return fast100(self.m)


# --- §8: the proof ------------------------------------------------------------------------------------------------------------

@dataclass
class Proof:
    ab: list[int]
    c: list[int]
    zc: list[tuple[int, int]]
    fa: int
    fb: int
    fc: int
    lc: list[tuple[int, int]]
    zp: list[int]
    rs: list[list[int]]
    #: each Ligerito level's opened rows (level 0 first, the final level last)
    rows: list[list[list[int]]]
    yr: list[int]
    msgs: list[tuple[tuple[int, int], tuple[int, int]]]
    ood: list[int]


class _Reader:
    def __init__(self, data: bytes):
        self.b, self.o = data, 0

    def take(self, n: int) -> bytes:
        if self.o + n > len(self.b):
            raise ValueError("proof: truncated")
        out = self.b[self.o:self.o + n]
        self.o += n
        return out

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def f(self) -> int:
        return G.from_bytes(self.take(16))

    def vec(self, want: int, elem: Callable[[], Any], what: str) -> list:
        n = self.u64()
        if n != want:
            raise ValueError(f"{what}: length {n}, want {want}")
        return [elem() for _ in range(n)]


def decode(sh: Shape, data: bytes) -> Proof:
    """The proof container (`decodeProof`); bytes after it are ignored (§8.1)."""
    s = sh.sched
    R = len(s.levels) - 1
    r = _Reader(data)
    h = lambda: r.take(sh.hash_len)
    salts = lambda q: r.vec(q, lambda: r.take(sh.salt_len), "opened_salts") if sh.salt_len else None
    pair = lambda: (r.f(), r.f())
    r.vec(1 << s.levels[0].cap, h, "cap")
    r.take(8 * 3 + 4)
    if r.take(1)[0]:
        r.u64()
    r.take(4)
    ab = r.vec(64, r.f, "round1_ab")
    c = r.vec(64, r.f, "round1_c")
    zc = r.vec(sh.m - 6, pair, "multilinear_rounds")
    fa, fb, fc = r.f(), r.f(), r.f()
    r.vec(0, r.u64, "zerocheck grinding_nonces")
    lc = r.vec(sh.k_log - 6, pair, "lincheck rounds")
    zp = r.vec(64, r.f, "z_partial")
    for _ in range(r.u64()):
        pair()
    r.vec(0, r.u64, "lincheck grinding_nonces")

    def ring_switch():
        v = r.vec(128, r.f, "s_hat_v")
        if r.u64():
            raise ValueError("ring switch grinding_nonce != 0")
        return v

    rs = r.vec(sh.claims, ring_switch, "ring_switches")
    r.vec(0, r.u64, "batching_nonces")
    r.vec(1 << s.levels[0].cap, h, "initial_cap")
    rows = []

    def level(lv: Level, lanes: int) -> None:
        rows.append(r.vec(lv.q, lambda: r.vec(lanes, r.f, "row"), "opened_rows"))
        r.vec(lv.q * (lv.d - lv.cap), h, "merkle_proof")
        salts(lv.q)

    level(s.levels[0], 1 << s.levels[0].lanes)
    if r.u64() != R:
        raise ValueError("recursive_caps: wrong length")
    for i in range(R):
        r.vec(1 << s.levels[i + 1].cap, h, "recursive cap")
    if r.u64() + 1 != R:
        raise ValueError("recursive_proofs: wrong length")
    for i in range(R - 1):
        level(s.levels[i + 1], 16)
    yr = r.vec(1 << (s.ext + 1), r.f, "yr")
    level(s.levels[R], 16)
    r.vec(0, pair, "sumcheck_transcript")
    msgs = r.vec(s.k0 + 8 + 7 * (R - 1), lambda: ((r.f(), r.f()), (r.f(), r.f())), "sumcheck_transcript_f256")
    r.vec(R + 1, r.u64, "grinding_nonces")
    ood = r.vec(1 + 2 * R, r.f, "ood_values")
    r.vec(0, r.u64, "fold_grinding_nonces")
    r.vec(3 * R + 2, r.u64, "claim_batch_grinding_nonces")
    r.vec(R + 1, r.u64, "consistency_batch_grinding_nonces")
    return Proof(ab, c, zc, fa, fb, fc, lc, zp, rs, rows, yr, msgs, ood)


# --- coins --------------------------------------------------------------------------------------------------------------------

def rep_rounds(record: dict, stream: str) -> list[list[int]]:
    """A stream's coins by round, from a session record (`flock-live-session/v1`)."""
    [st] = [s for s in record["streams"] if s["stream"] == stream]
    return [[G.of_hex(c) for c in rd["coins"]] for rd in st["rounds"]]


class Coins:
    """The rep's coins in the order the verifier draws them (§6.2: a draw of n coins is one round of n; none is no round)."""

    def __init__(self, rounds: Sequence[Sequence[int]]):
        self.rounds, self.k = list(rounds), 0

    def vec(self, n: int) -> list[int]:
        if n == 0:
            return []
        if self.k >= len(self.rounds) or len(self.rounds[self.k]) != n:
            raise ValueError(f"round {self.k}: {n} coins drawn, the record has "
                             f"{len(self.rounds[self.k]) if self.k < len(self.rounds) else 'no round'}")
        self.k += 1
        return list(self.rounds[self.k - 1])

    def one(self) -> int:
        return self.vec(1)[0]

    def k2(self) -> tuple[int, int]:
        a, b = self.vec(2)
        return (a, b)


# --- the verifier's public functions --------------------------------------------------------------------------------------------

def _x(v: int) -> int:
    return G.reduce(v << 1)


def _den(d: int) -> int:
    acc = 1
    for i in range(1, 1 << d):
        acc = mul(acc, phi8(i))
    return acc


NODES_S = [phi8(i) for i in range(64)]
NODES_L = [phi8(64 + i) for i in range(64)]
DEN6_INV = inv(_den(6))
DEN7_INV = inv(_den(7))


def _frac(y: int) -> int:
    return mul(y, inv(y ^ 1))


PINNED = [phi8(0xF7), phi8(0x53), phi8(0xB5), _frac(2), _frac(4), _frac(16), _frac(256)]


def lagrange(nodes: list[int], den_inv: int, z: int) -> list[int]:
    if z in nodes:
        return [int(n == z) for n in nodes]
    zn = 1
    for n in nodes:
        zn = mul(zn, z ^ n)
    c = mul(zn, den_inv)
    return [mul(c, inv(z ^ n)) for n in nodes]


def skip_weights(z: int) -> list[int]:
    return lagrange(NODES_S, DEN6_INV, z)


def comb_weights(z: int) -> list[int]:
    if z in NODES_S:
        return [0] * 64
    if z in NODES_L:
        return [int(n == z) for n in NODES_L]
    zall = 1
    for n in NODES_S + NODES_L:
        zall = mul(zall, z ^ n)
    c = mul(zall, DEN7_INV)
    return [mul(c, inv(z ^ n)) for n in NODES_L]


def eq_table(r: Sequence[int]) -> list[int]:
    t = [1]
    for rj in r:
        t = [mul(x, rj ^ 1) for x in t] + [mul(x, rj) for x in t]
    return t


def eq_table_k(r: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    t = [(1, 0)]
    for rj in r:
        a = (rj[0] ^ 1, rj[1])
        t = [kmul(x, a) for x in t] + [kmul(x, rj) for x in t]
    return t


def _bits(c: int) -> list[int]:
    out = []
    while c:
        low = c & -c
        out.append(low.bit_length() - 1)
        c ^= low
    return out


def _map_bits(g: int) -> list[list[int]]:
    """`mulColumns g`, each column as the bits set in it."""
    out, v = [], g
    for _ in range(128):
        out.append(_bits(v))
        v = _x(v)
    return out


def _apply(cols: list[list[int]], s: list[int]) -> list[int]:
    """`applyMap` on K elements packed as ``c0 | c1 << 128``."""
    out = [0] * 128
    for u, su in enumerate(s):
        if su:
            for l in cols[u]:
                out[l] ^= su
    return out


def _pk(a: tuple[int, int]) -> int:
    return a[0] | a[1] << 128


def _upk(a: int) -> tuple[int, int]:
    return (a & G.MASK, a >> 128)


def basis_for_claim(gamma_rpp_z: tuple, q: list[tuple[int, int]], e: int) -> list[tuple[int, int]]:
    """§12.4's branching program for one switched claim (`basisForClaim`): ``b̂_k(q ‖ bits(y))`` for ``y < 2^e``."""
    _, rpp, z = gamma_rpp_z
    eq_r = eq_table(rpp)
    s = [1] + [0] * 127
    for j, qj in enumerate(q):
        m = _apply(_map_bits(1 ^ z[j]), s)
        s = [m[l] ^ _pk(kmul(qj, _upk(s[l]))) for l in range(128)]
    ones = [_map_bits(z[len(q) + i]) for i in range(e)]
    zeros = [_map_bits(1 ^ z[len(q) + i]) for i in range(e)]
    level = [s]
    for i in range(e):
        level = [_apply(zeros[i], t) for t in level] + [_apply(ones[i], t) for t in level]
    out = []
    for t in level:
        v0 = v1 = 0
        for u, tu in enumerate(t):
            if tu:
                v0 ^= mul(tu & G.MASK, eq_r[u])
                v1 ^= mul(tu >> 128, eq_r[u])
        out.append((v0, v1))
    return out


def _sv_table(n: int) -> list[int]:
    sv: list[int] = []
    for k in range(n):
        y = 1 << k
        for j in range(k):
            y = mul(y, y) ^ mul(sv[j], y)
        sv.append(y)
    return sv


def _cff(c: tuple[int, int]) -> tuple[int, int]:
    """``1 + c·(1 + u)``."""
    p = kmul(c, (1, 1))
    return (p[0] ^ 1, p[1])


def final_weights(sch: Schedule, init_ch: list, levels: list[list], oods: list, conss: list, sws: list) -> list[tuple[int, int]]:
    """§13.6's ``res[y]`` for ``y < 2^ext``: the batched basis at the folded point, the OOD terms and the consistency terms,
    all public (the coins and the claims' points)."""
    n, ext = sch.n, sch.ext
    orig = list(init_ch) + [c for l in levels for c in l[1:]]
    if len(orig) + ext != n:
        raise ValueError("ligerito: |orig| + ext != n")
    heads = [l[0] for l in levels]

    def prod_from(a: int) -> tuple[int, int]:
        acc = (1, 0)
        for c in heads[a:]:
            acc = kmul(acc, _cff(c))
        return acc

    pi0 = prod_from(0)
    res = [(0, 0)] * (1 << ext)
    for gamma, rpp, z in sws:
        b = basis_for_claim((gamma, rpp, z), orig, ext)
        res = [(r[0] ^ mul(x[0], gamma), r[1] ^ mul(x[1], gamma)) for r, x in zip(res, b)]
    res = [kmul(r, pi0) for r in res]
    for z, beta, split in oods:
        if split is None:
            fixed, scale = orig, pi0
        else:
            fixed = list(levels[split]) + [c for l in levels[split + 1:] for c in l[1:]]
            scale = prod_from(split + 1)
        if len(z) != len(fixed) + ext:
            raise ValueError("ligerito: ood dimension")
        head = (1, 0)
        for t, ft in enumerate(fixed):
            head = kmul(head, (1 ^ z[t] ^ ft[0], ft[1]))
        c = G.smul(kmul(head, scale), beta)
        tail = eq_table(z[len(fixed):])
        res = [(r[0] ^ mul(c[0], t), r[1] ^ mul(c[1], t)) for r, t in zip(res, tail)]
    sv = _sv_table(n + 1)
    sv_inv = [inv(x) for x in sv]
    for cols, pos, alpha, beta, start in conss:
        fixed = [c for l in levels[start:] for c in l[1:]]
        if len(fixed) + ext != cols:
            raise ValueError("ligerito: consistency dimension")
        ea = eq_table(alpha)
        scale = prod_from(start)
        add = [(0, 0)] * (1 << ext)
        for j, p in enumerate(pos):
            wh, s = [], p
            for i in range(cols):
                wh.append(mul(s, sv_inv[i]))
                s = mul(s, s) ^ mul(sv[i], s)
            pj = (1, 0)
            for i, fi in enumerate(fixed):
                w1 = 1 ^ wh[i]
                pj = kmul(pj, (1 ^ mul(fi[0], w1), mul(fi[1], w1)))
            coef = G.smul(pj, ea[j])
            tj = [1]
            for i in range(ext):
                w = wh[len(fixed) + i]
                tj = tj + [mul(t, w) for t in tj]
            add = [(a[0] ^ mul(coef[0], t), a[1] ^ mul(coef[1], t)) for a, t in zip(add, tj)]
        res = [(r[0] ^ x[0], r[1] ^ x[1]) for r, x in zip(res, (G.smul(kmul(a, scale), beta) for a in add))]
    return res


def positions(lv: Level, coins: Sequence[int]) -> list[int]:
    """§13.5: stratified positions; reads the low ``d - c`` bits of each coin's low word."""
    out, j = [], 0
    for c in lv.strata:
        for s in range(1 << c):
            out.append((s << (lv.d - c)) | ((coins[j] & ((1 << 64) - 1)) % (1 << (lv.d - c))))
            j += 1
    return out


# --- values: ints, or symbolic -------------------------------------------------------------------------------------------------

class Lin:
    """A symbolic F value: ``Σ t[k]·k + c + Σ q[(a, b)]·a·b`` over message elements ``k``. ``c`` is None when the value has no
    constant term; a term or product stays (as a structural slot) when its coefficient happens to be zero."""

    __slots__ = ("t", "c", "q")

    def __init__(self, t: dict | None = None, c: int | None = None, q: dict | None = None):
        self.t = t if t is not None else {}
        self.c = c
        self.q = q if q is not None else {}

    def __xor__(self, o):
        if isinstance(o, int):
            if o:
                raise TypeError("a constant enters a symbolic value through `cst`")
            return self
        t = dict(self.t)
        for k, v in o.t.items():
            t[k] = t.get(k, 0) ^ v
        q = dict(self.q)
        for k, v in o.q.items():
            q[k] = q.get(k, 0) ^ v
        c = self.c if o.c is None else o.c if self.c is None else self.c ^ o.c
        return Lin(t, c, q)

    __rxor__ = __xor__

    def scale(self, a: int) -> "Lin":
        return Lin({k: mul(a, v) for k, v in self.t.items()}, None if self.c is None else mul(a, self.c),
                   {k: mul(a, v) for k, v in self.q.items()})


def sm(c: int, x):
    """A coin-derived constant times a value."""
    return mul(c, x) if isinstance(x, int) else x.scale(c)


def mv(x, y):
    """A message times a message (``v_A · v_B``, ``comb[l] · z_partial[l]``)."""
    if isinstance(x, int) and isinstance(y, int):
        return mul(x, y)
    (kx,), (ky,) = x.t, y.t
    if x.t[kx] != 1 or y.t[ky] != 1 or x.c is not None or y.c is not None or x.q or y.q:
        raise TypeError("a product of two messages takes two messages")
    return Lin(q={(kx, ky): 1})


def dot(cs: Sequence[int], xs: Sequence) -> Any:
    acc = 0
    for c, x in zip(cs, xs, strict=True):
        acc = acc ^ sm(c, x)
    return acc


def kc(c: tuple[int, int], x: tuple) -> tuple:
    """A constant of K times a K value: ``(x0 c0 + x^-1 x1 c1, x0 c1 + x1 (c0 + c1))``."""
    return (sm(c[0], x[0]) ^ sm(mul(XINV, c[1]), x[1]), sm(c[1], x[0]) ^ sm(c[0] ^ c[1], x[1]))


def kf(c: tuple[int, int], x) -> tuple:
    """A constant of K times an F value."""
    return (sm(c[0], x), sm(c[1], x))


def fk(b: int, x: tuple) -> tuple:
    """A constant of F times a K value (`F256.smul`)."""
    return (sm(b, x[0]), sm(b, x[1]))


def kadd(*xs: tuple) -> tuple:
    a, b = 0, 0
    for x in xs:
        a, b = a ^ x[0], b ^ x[1]
    return (a, b)


@dataclass(frozen=True)
class Claim:
    """A claim handed to the opening (§11): value, 64 skip weights, m - 6 coordinates. An extra claim's are public."""

    value: Any
    skip: tuple[int, ...]
    x: tuple[int, ...]


@dataclass
class Messages:
    """One rep's message elements as values (ints, or `Lin` for `symbolic`). ``rs_t[k][u]`` is the ring switch's transpose
    ``t[u]`` of claim k; ``cons(level, ea)`` the level's consistency sums ``Σ_j ea[j] · row_j[l]``; ``comb`` the circuit fold."""

    ab: list
    c: list
    zc: list
    fa: Any
    fb: Any
    lc: list
    zp: list
    rs: list
    rs_t: list
    ood: list
    msgs: list
    yr: list
    comb: list
    cons: Callable[[int, list[int]], list]
    symbolic: bool = False
    #: Ligerito's T as an atom (symbolic only), the batched sum it stands for substituted once at the end
    t_atom: Any = None


def run(sh: Shape, coins: Coins, M: Messages, extras: Sequence[Claim]) -> tuple[list, dict]:
    """The rep's residuals (zerocheck, lincheck, one per claim, Ligerito's c0 and c1) and the coin-derived points beside them."""
    cst = (lambda v: Lin(c=v)) if M.symbolic else (lambda v: v)
    m, k_log, sch = sh.m, sh.k_log, sh.sched
    if sh.claims != 2 + len(extras):
        raise ValueError(f"{sh.claims} claims, {2 + len(extras)} given")
    info: dict[str, Any] = {}
    # §9.2 zerocheck
    coins.vec(6)
    r = PINNED + coins.vec(m - 13)
    z = coins.one()
    pc = dot(lagrange(NODES_L, DEN6_INV, z), M.c)
    claim = dot(comb_weights(z), [a ^ c for a, c in zip(M.ab, M.c)]) ^ pc
    rho = []
    for i, (g1, ginf) in enumerate(M.zc):
        g0 = sm(inv(1 ^ r[i]), claim ^ sm(r[i], g1))
        rr = coins.one()
        rho.append(rr)
        claim = sm(1 ^ rr, g0) ^ sm(rr, g1) ^ sm(mul(rr, 1 ^ rr), ginf)
    out = [claim ^ mv(M.fa, M.fb)]
    info.update(z=z, rho=rho, r=r, v_c=pc)
    # §10 lincheck
    alpha, beta = coins.one(), coins.one()
    claim = sm(alpha, M.fa) ^ M.fb ^ cst(beta)
    ts = []
    for q1, qinf in M.lc:
        t = coins.one()
        ts.append(t)
        e0 = claim ^ q1
        claim = sm(mul(t, t), qinf) ^ sm(t, e0 ^ q1 ^ qinf) ^ e0
    acc = 0
    for cl, zl in zip(M.comb, M.zp, strict=True):
        acc = acc ^ mv(cl, zl)
    out.append(acc ^ claim)
    r_prime = coins.one()
    w = dot(skip_weights(r_prime), M.zp)
    info.update(fold=dict(alpha=alpha, beta=beta, lw=skip_weights(z), rho_in=rho[:k_log - 6], ts=ts), w=w, r_prime=r_prime,
                r_in=ts[::-1])
    # §11, §12.2-§12.3 ring switching
    claims = [Claim(w, tuple(skip_weights(r_prime)), tuple(ts[::-1] + rho[k_log - 6:m - 6])),
              Claim(pc, tuple(skip_weights(z)), tuple(r))] + [Claim(cst(e.value), e.skip, e.x) for e in extras]
    betas, sws = [], []
    for k, cl in enumerate(claims):
        x0 = cl.x[0]
        wv = [mul(cl.skip[v % 64], x0 if v >= 64 else 1 ^ x0) for v in range(128)]
        out.append(dot(wv, M.rs[k]) ^ cl.value)
        rpp = coins.vec(7)
        betas.append(dot(eq_table(rpp), M.rs_t[k]))
        sws.append([None, rpp, list(cl.x[1:])])
    gamma = coins.vec(len(claims))
    for k, g in enumerate(gamma):
        sws[k][0] = g
    T = dot(gamma, betas)
    info.update(T=T, gamma=gamma)
    # §13 Ligerito
    R = len(sch.levels) - 1
    msgs, oods = iter(M.msgs), iter(M.ood)
    st: dict[str, Any] = {"s": (M.t_atom if M.symbolic else T, 0), "q": None}

    def of_msg(mm, s):
        return (mm[0], kadd(s, mm[1]), mm[1])

    def challenge():
        rk = coins.k2()
        a0, a1, a2 = st["q"]
        st["s"] = kadd(a0, kc(rk, a1), kc(kmul(rk, rk), a2))
        st["q"] = of_msg(next(msgs), st["s"])
        return rk

    def add_scaled(p, b):
        st["q"] = tuple(kadd(x, fk(b, y)) for x, y in zip(st["q"], p))

    ood_ctx, cons_ctx = [], []

    def ood_intro(dim, split):
        zz = coins.vec(dim)
        y = next(oods)
        mm = next(msgs)
        b = coins.one()
        add_scaled(of_msg(mm, (y, 0)), b)
        st["s"] = kadd(st["s"], (sm(b, y), 0))
        ood_ctx.append((zz, b, split))

    def query(level_i, fold):
        lv = sch.levels[level_i]
        pos = positions(lv, coins.vec(lv.q))
        al = coins.vec((lv.q - 1).bit_length())
        ea = eq_table(al)[:lv.q]
        ef = eq_table_k(fold)
        e = kadd(*(kf(f, a) for f, a in zip(ef, M.cons(level_i, ea), strict=True)))
        return pos, al, e

    zz = coins.vec(sch.n)
    y = next(oods)
    b = coins.one()
    st["s"] = kadd(st["s"], (sm(b, y), 0))
    ood_ctx.append((zz, b, None))
    st["q"] = of_msg(next(msgs), st["s"])
    init_ch = [challenge() for _ in range(sch.k0)]
    split = sch.n - sch.k0 + 1
    ood_intro(split, 0)
    ood_intro(split, 0)
    pos, al, e = query(0, init_ch)
    mm = next(msgs)
    b0 = coins.one()
    add_scaled(of_msg(mm, e), b0)
    st["s"] = kadd(st["s"], fk(b0, e))
    cons_ctx.append((sch.n - sch.k0, pos, al, b0, 0))
    levels: list[list] = []
    ext = 0
    for i in range(R):
        if split < 4:
            raise ValueError("ligerito: split_dim < k")
        levels.append([challenge() for _ in range(4)])
        ext = split - 4
        if i + 1 < R:
            split = ext + 1
            ood_intro(split, i + 1)
            ood_intro(split, i + 1)
            pos, al, e = query(i + 1, levels[-1])
            mm = next(msgs)
            bb = coins.one()
            add_scaled(of_msg(mm, e), bb)
            st["s"] = kadd(st["s"], fk(bb, e))
            cons_ctx.append((ext, pos, al, bb, i + 1))
        else:
            pos, al, e = query(R, levels[-1])
            bb = coins.one()
            st["s"] = kadd(st["s"], fk(bb, e))
            cons_ctx.append((ext, pos, al, bb, R))
    if coins.k != len(coins.rounds):
        raise ValueError(f"{len(coins.rounds) - coins.k} recorded rounds were never drawn")
    if next(msgs, None) is not None or next(oods, None) is not None:
        raise ValueError("ligerito: leftover messages")
    res = final_weights(sch, init_ch, levels, ood_ctx, cons_ctx, sws)
    if len(M.yr) != 2 << ext:
        raise ValueError("ligerito: |yr|")
    lhs = kadd(*(kc(rk, (M.yr[2 * j], M.yr[2 * j + 1])) for j, rk in enumerate(res)))
    fin = kadd(lhs, st["s"])
    if M.symbolic:
        fin = tuple(_subst(f, M.t_atom, T) for f in fin)
    out += list(fin)
    info.update(levels=levels, init_ch=init_ch, res=res, queries=[(c[1], eq_table(c[2])[:len(c[1])]) for c in cons_ctx])
    return out, info


def _subst(f, atom: Lin, value: Lin):
    (k,) = atom.t
    if not isinstance(f, Lin) or k not in f.t:
        return f
    t = dict(f.t)
    coef = t.pop(k)
    return Lin(t, f.c, dict(f.q)) ^ value.scale(coef)


# --- the rep's messages ----------------------------------------------------------------------------------------------------

def _transpose(s: Sequence[int]) -> list[int]:
    return [sum(((s[v] >> u) & 1) << v for v in range(128)) for u in range(128)]


def messages(sh: Shape, p: Proof, comb: Sequence[int]) -> Messages:
    """The rep's messages as ints, the consistency sums from its opened rows."""
    def cons(level: int, ea: list[int]) -> list[int]:
        rows = p.rows[level]
        return [dot(ea, [row[l] for row in rows]) for l in range(len(rows[0]))]

    return Messages(p.ab, p.c, p.zc, p.fa, p.fb, p.lc, p.zp, p.rs, [_transpose(s) for s in p.rs], p.ood, p.msgs, p.yr,
                    list(comb), cons)


def free_names(sh: Shape) -> list[tuple]:
    """The rep's free message elements, in protocol order: the names of `InnerRepCheck`'s first w slots."""
    s = sh.sched
    R = len(s.levels) - 1
    out = [("ab", i) for i in range(64)] + [("c", i) for i in range(64)]
    out += [(nm, i) for i in range(sh.m - 6) for nm in ("g1", "ginf")] + [("fa",), ("fb",)]
    out += [(nm, j) for j in range(sh.k_log - 6) for nm in ("q1", "qinf")] + [("zp", l) for l in range(64)]
    out += [("rs", k, v) for k in range(sh.claims) for v in range(128)]
    mi, oi = iter(range(10 ** 6)), iter(range(10 ** 6))

    def msg():
        i = next(mi)
        return [("msg", i, h, c) for h in range(2) for c in range(2)]

    def ood():
        return [("ood", next(oi))]

    lg = ood() + msg() + [x for _ in range(s.k0) for x in msg()] + ood() + msg() + ood() + msg() + msg()
    for i in range(R):
        lg += [x for _ in range(4) for x in msg()]
        if i + 1 < R:
            lg += ood() + msg() + ood() + msg() + msg()
        else:
            lg += [("yr", j) for j in range(2 << s.ext)]
    return out + lg


@dataclass
class Layout:
    """`InnerRepCheck`'s ports: the name of every w and v slot."""

    w: list[tuple]
    v: list[tuple]
    index: dict = field(repr=False, default_factory=dict)


def _cons_names(sh: Shape) -> list[tuple]:
    return [("cons", i, l) for i, lv in enumerate(sh.sched.levels) for l in range(1 << lv.lanes)]


def symbolic(sh: Shape) -> Messages:
    """The rep's messages as symbolic values, named as `free_names` (and the consistency sums and the comb)."""
    var = lambda k: Lin({k: 1})
    s = sh.sched
    ab = [var(("ab", i)) for i in range(64)]
    c = [var(("c", i)) for i in range(64)]
    zc = [(var(("g1", i)), var(("ginf", i))) for i in range(sh.m - 6)]
    lc = [(var(("q1", j)), var(("qinf", j))) for j in range(sh.k_log - 6)]
    zp = [var(("zp", l)) for l in range(64)]
    rs = [[var(("rs", k, v)) for v in range(128)] for k in range(sh.claims)]
    rs_t = [[var(("rs.t", k, u)) for u in range(128)] for k in range(sh.claims)]
    R = len(s.levels) - 1
    n_msgs = s.k0 + 8 + 7 * (R - 1)
    msgs = [tuple((var(("msg", i, h, 0)), var(("msg", i, h, 1))) for h in range(2)) for i in range(n_msgs)]
    ood = [var(("ood", i)) for i in range(1 + 2 * R)]
    yr = [var(("yr", j)) for j in range(2 << s.ext)]
    comb = [var(("comb", l)) for l in range(64)]
    cons = lambda level, ea: [var(("cons", level, l)) for l in range(1 << s.levels[level].lanes)]
    return Messages(ab, c, zc, var(("fa",)), var(("fb",)), lc, zp, rs, rs_t, ood, msgs, yr, comb, cons, symbolic=True,
                    t_atom=var(("T",)))


def _w_names(sh: Shape) -> list[tuple]:
    return free_names(sh) + _cons_names(sh) + [("comb_hid", l) for l in range(64)]


def _operand(k: tuple, wi: dict) -> list:
    """The operand a message variable stands for, as GfResiduals terms."""
    if k[0] == "rs.t":
        return [["wt", wi[("rs", k[1], 0)], k[2]]]
    if k[0] == "comb":
        return [["w", wi[("comb_hid", k[1])]], ["v", None]]
    return [["w", wi[k]]]


def _order(sh: Shape) -> dict:
    names = _w_names(sh)
    out = {k: i for i, k in enumerate(names)}
    base = len(names)
    out.update({("rs.t", k, u): base + 128 * k + u for k in range(sh.claims) for u in range(128)})
    out.update({("comb", l): out[("comb_hid", l)] for l in range(64)})
    return out


def _residual_lins(sh: Shape, coins: Coins, extras: Sequence[Claim]) -> tuple[list[Lin], dict]:
    out, info = run(sh, coins, symbolic(sh), extras)
    return [x if isinstance(x, Lin) else Lin() for x in out], info


def _build(sh: Shape, lins: list[Lin]) -> tuple[dict, Layout, list[int]]:
    """`GfResiduals`'s structure, the ports' layout, and the coefficient slots' values, from the symbolic residuals."""
    names = _w_names(sh)
    wi = {k: i for i, k in enumerate(names)}
    order = _order(sh)
    ops: list[list] = []
    op_of: dict = {}
    v_names: list[tuple] = []
    v_vals: list[int] = []

    def var_op(k):
        if k not in op_of:
            terms = _operand(k, wi)
            if k[0] == "comb":
                terms[1][1] = 0  # set below, once the comb's public slots are placed
            op_of[k] = len(ops)
            ops.append(terms)
        return op_of[k]

    res = []
    for ri, f in enumerate(lins):
        mulp, add = [], []
        for k in sorted(f.t, key=order.__getitem__):
            slot = len(v_names)
            v_names.append(("coef", ri, k))
            v_vals.append(f.t[k])
            ops.append([["v", slot]])
            mulp.append([len(ops) - 1, var_op(k)])
        for (a, b), coef in sorted(f.q.items(), key=lambda kv: (order[kv[0][0]], order[kv[0][1]])):
            if coef != 1:
                raise AssertionError(f"residual {ri}: a product of messages with coefficient {coef:#x}")
            mulp.append([var_op(a), var_op(b)])
        if f.c is not None:
            v_names.append(("const", ri))
            v_vals.append(f.c)
            add.append(["v", len(v_names) - 1])
        res.append({"mul": mulp, "add": add})
    pub0 = len(v_names)
    v_names += [("comb_pub", l) for l in range(64)]
    for k, i in op_of.items():
        if k[0] == "comb":
            ops[i][1][1] = pub0 + k[1]
    S = {"nw": len(names), "nv": len(v_names), "ops": ops, "res": res}
    return S, Layout(names, v_names, wi), v_vals


def _dummy_coins(sh: Shape) -> Coins:
    """Coins in the rounds a rep draws, all equal to one value: the structure depends on the rounds, not on the coins."""
    import random

    g = random.Random(0)
    rounds = []

    class _Rec(Coins):
        def vec(self, n):
            if n == 0:
                return []
            v = [g.getrandbits(128) for _ in range(n)]
            rounds.append(v)
            self.k += 1
            self.rounds.append(v)
            return v

    return _Rec([])


def _extras_like(sh: Shape) -> list[Claim]:
    import random

    g = random.Random(1)
    return [Claim(g.getrandbits(128), tuple(g.getrandbits(128) for _ in range(64)), tuple(g.getrandbits(128) for _ in range(sh.m - 6)))
            for _ in range(sh.claims - 2)]


_STRUCTURES: dict[Shape, tuple[dict, Layout]] = {}


def structure(sh: Shape) -> tuple[dict, Layout]:
    """`InnerRepCheck` = ``GfResiduals_v1{S}``'s S, and its ports' layout: a function of the shape alone."""
    if sh not in _STRUCTURES:
        lins, _ = _residual_lins(sh, _dummy_coins(sh), _extras_like(sh))
        S, lay, _ = _build(sh, lins)
        _STRUCTURES[sh] = (S, lay)
    return _STRUCTURES[sh]


def verifier_rows(sh: Shape, rounds: Sequence[Sequence[int]], extras: Sequence[Claim], comb_pub: Sequence[int]) -> tuple[list[int], dict]:
    """`InnerRepCheck`'s v port at the rep's coins (the public part of the comb last), and the coin-derived points (`run`'s
    info: the fold's points for `HiddenComb`, each Ligerito level's query positions and eq(α) for `GfScale`)."""
    lins, info = _residual_lins(sh, Coins(rounds), extras)
    S, lay, vals = _build(sh, lins)
    if S != structure(sh)[0]:
        raise AssertionError("the structure moved with the coins")
    return vals + list(comb_pub), info


def query_rows(info: dict) -> list[tuple[list[int], list[int]]]:
    """Per Ligerito level, its query positions and the eq(α)[j] that `GfScale` multiplies opened row j by."""
    return info["queries"]


def witness(sh: Shape, p: Proof, rounds: Sequence[Sequence[int]], comb_hid: Sequence[int], extras: Sequence[Claim]) -> list[int]:
    """`InnerRepCheck`'s w port: the free message elements, the consistency sums (from the opened rows at the coins' eq(α)),
    the comb's hidden part."""
    names = free_names(sh)
    val: dict[tuple, int] = {}
    val.update({("ab", i): x for i, x in enumerate(p.ab)})
    val.update({("c", i): x for i, x in enumerate(p.c)})
    for i, (g1, ginf) in enumerate(p.zc):
        val[("g1", i)], val[("ginf", i)] = g1, ginf
    val[("fa",)], val[("fb",)] = p.fa, p.fb
    for j, (q1, qinf) in enumerate(p.lc):
        val[("q1", j)], val[("qinf", j)] = q1, qinf
    val.update({("zp", l): x for l, x in enumerate(p.zp)})
    val.update({("rs", k, v): x for k, s in enumerate(p.rs) for v, x in enumerate(s)})
    val.update({("msg", i, h, c): mm[h][c] for i, mm in enumerate(p.msgs) for h in range(2) for c in range(2)})
    val.update({("ood", i): x for i, x in enumerate(p.ood)})
    val.update({("yr", j): x for j, x in enumerate(p.yr)})
    _, info = run(sh, Coins(rounds), messages(sh, p, [0] * 64), extras)
    cons = []
    for level, (_, ea) in enumerate(info["queries"]):
        rows = p.rows[level]
        cons += [dot(ea, [row[l] for row in rows]) for l in range(len(rows[0]))]
    return [val[k] for k in names] + cons + list(comb_hid)


def residuals(sh: Shape, p: Proof, rounds: Sequence[Sequence[int]], comb: Sequence[int], extras: Sequence[Claim]) -> tuple[list[int], dict]:
    """The reference: the rep's residuals on its proof at its coins (all zero iff the inner verifier's algebra accepts)."""
    return run(sh, Coins(rounds), messages(sh, p, comb), extras)
