"""Falsification test of `Pous.P2.Assumptions.P2RingAuditSecure` (`Security/Definitions/Pous/Assumptions/P2.lean`), the
test that `shape.conjectures` in `Security/lean-audit.json` and `Security/Properties/lean-audit.json` names for it and
for `P2RingReducesToM1p`, which implies it: P2 in the ring model, with sequential reveal, accepts every `⌊ρ·|C|⌋`-bit, `(D, Q, L)`-bounded responder with probability at
most `δ + ε`. The same draws test the per-round form `P2RingRoundSecure`
(`test_p2_ring_round_secure_at_a_scaled_point`).

The audit is `p2ring` as `Security/Definitions/Pous/P2/Cost.lean` defines it: a uniform odd labeling of `Z/p`
(sampled lazily, which is the same distribution), per-block keys `(K_j, τ_j)`, a responder that sees elements only
as labels through metered `add`, `mul`, `one` and `key` queries, and the trusted encoder `ρ(σ_K(ρ(m + t)))`, which at
the identity labeling is `schemes/p2.py`'s field (`test_the_model_is_the_reference_codec`). It is played at a scaled
point with the deployed structure (`ρ = 18/19`, `δ = 1%`, `p = 2^w − c0` prime and `≡ 3 mod 4`, 16 payload bits less
than a label), at `w = 184`, `p = 2^184 − 33` and 16 labels:

* the conjecture's bound is the ideal row's with `λ` bits of slack per label: a responder holding `⌊ρ·|C|⌋ + λ·B`
  bits keeps at most `KEPT = ⌊(⌊ρ·|C|⌋ + λ·B)/w⌋` labels, and `K` is the least `k` with `(KEPT/B)^k ≤ δ` (the
  deployed `k = 105` is the `λ = 128` row's), so the bound is `δ` and the crypto terms are zero;
* the region is the `(D, Q, L)` a deadline gives: `D_CAP` one multiplication round short of a recomputed label
  (`PousP2Ring.tight`'s `2(n + 1)`), two key queries, and `L_CAP` the ring work of searching `λ` dropped bits.
  As at the deployed point (a root in `65,789` ring queries, `λ = 128`, `w/19 = 865`), a recomputed label's ring work
  is below `2^λ` and `λ < w/19`, so the depth-limited recompute is inside the region and the slack is not vacuous.

A responder's answer to label `i` depends only on `i`, its state and the oracle (`Responder`), so given the sample
`ω` it is accepted with probability exactly `(s_ω/B)^K`, `s_ω` the labels it answers right; the test averages that
over seeded draws, and a responder inside the region whose average is above `δ + margin(TRIALS)` (Hoeffding, at
`1 − 10⁻⁹`) is a counterexample, which fails the test with the responder and the samples. Outside the region the test
shows the known-false side as expected failures: the honest responder (it holds all of `C`), `PousP2Ring.tight`'s
recomputation (`D = 2(n + 1)`) and `PousP2Ring.not_meetsOn_b19_k88`'s truncation one bit past the slack."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from fractions import Fraction

import pytest

from verity.core.protocols.pous.schemes import p2 as P2

W, C0, B = 184, 33, 16
P = (1 << W) - C0
PAYLOAD = W - 16
RHO, DELTA = Fraction(18, 19), Fraction(1, 100)
S = math.floor(RHO * B * W)
LAM = 9
N = ((P + 1) // 4).bit_length()
D_CAP, Q_CAP, L_CAP = 2 * N + 1, 2, (1 << (LAM + 1)) + 1
KEPT = min(B, (S + LAM * B) // W)
K = next(k for k in range(1, 10_000) if Fraction(KEPT, B) ** k <= DELTA)
SEED = 2026101201
TRIALS = 32
RANDOM = 12


def margin(n: int) -> float:
    """How far above the bound an average of `n` draws' acceptance probabilities (each in `[0, 1]`) must be to be a
    counterexample with probability at least `1 − 10⁻⁹` (Hoeffding)."""
    return math.sqrt(math.log(1e9) / (2 * n))


class Counterexample(AssertionError):
    """A responder accepted more often than the bound allows."""


# ---------------------------------------------------------------------------------------------------------------
# the ring model
# ---------------------------------------------------------------------------------------------------------------

class Ring:
    """One sample `ω` of `ringRO B w p`: an odd labeling `φ` (`φ(−x) = p − φ(x)`), sampled lazily and uniformly, and
    label `j`'s keys `(K_j, τ_j)`. `identity` fixes `φ = id`, the reference codec's."""

    def __init__(self, rng: random.Random, identity: bool = False) -> None:
        self.rng, self.identity = rng, identity
        self.lab, self.elt = {0: 0}, {0: 0}
        self.keys = [(rng.getrandbits(W), rng.randrange(P)) for _ in range(B)]

    def _pair(self, x: int, a: int) -> None:
        self.lab[x], self.elt[a] = a, x
        self.lab[P - x], self.elt[P - a] = P - a, P - x

    def label(self, x: int) -> int:
        x %= P
        if self.identity:
            return x
        if (a := self.lab.get(x)) is None:
            while (a := self.rng.randrange(1, P)) in self.elt:
                pass
            self._pair(x, a)
        return a

    def element(self, a: int) -> int:
        """`eltD`: the element behind a label below `p`, `0` for an invalid label."""
        if a >= P:
            return 0
        if self.identity:
            return a
        if (x := self.elt.get(a)) is None:
            while (x := self.rng.randrange(1, P)) in self.lab:
                pass
            self._pair(x, a)
        return x


class Meter:
    """The oracle for one answer, counting `RingBounded`'s three costs: rounds with a `mul`, key queries, ring queries."""

    def __init__(self, ring: Ring) -> None:
        self.ring, self.D, self.Q, self.L = ring, 0, 0, 0

    def round(self, qs: list[tuple]) -> list[int]:
        self.D += any(q[0] == "mul" for q in qs)
        return [self._answer(q) for q in qs]

    def _answer(self, q: tuple) -> int:
        r = self.ring
        if q[0] == "key":
            self.Q += 1
            k, tau = r.keys[q[1]]
            return k if q[2] == "mask" else tau
        self.L += 1
        if q[0] == "one":
            return r.label(1)
        a, b = q[1], q[2]
        if a >= P or b >= P:
            return a
        x, y = r.element(a), r.element(b)
        return r.label(x * y if q[0] == "mul" else x + y)


def sigma_k(k: int, r: int) -> int:
    return r ^ k if r ^ k < P else r


def neg_l(a: int) -> int:
    return a if a == 0 else P - a


def sq_fix(a: int, s: int) -> int:
    """`sq` on labels once a `mul` returned `s`, the label of `a²`."""
    return neg_l(s) if a % 2 else s


def rho_e(ring: Ring, y: int) -> int:
    """`rhoE`: the element `a` with `sqE(a) = y`, its label even iff `y` is a square."""
    y %= P
    if y == 0:
        return 0
    r = pow(y, (P + 1) // 4, P)
    odd = r * r % P != y
    return r if (ring.label(r) % 2 == 1) == odd else P - r


def enc_block(ring: Ring, j: int, v: int) -> int:
    """`encBlock`: the trusted encoder sees `φ` whole and takes the two roots."""
    k, tau = ring.keys[j]
    u = sigma_k(k, ring.label(rho_e(ring, ring.element(v) + ring.element(tau))))
    return ring.label(rho_e(ring, ring.element(u)))


def dec_block(m: Meter, j: int, c: int) -> int:
    """`decBlock`: `sq(σ_K(sq(c))) − t`, three rounds, two with a `mul`; the label read mod `2^b`."""
    k, tau, s = m.round([("key", j, "mask"), ("key", j, "offset"), ("mul", c, c)])
    u = sigma_k(k, sq_fix(c, s))
    (s2,) = m.round([("mul", u, u)])
    (out,) = m.round([("add", sq_fix(u, s2), neg_l(tau))])
    return out % (1 << PAYLOAD)


@dataclass
class Draw:
    ring: Ring
    w: list[int]
    code: list[int]


def draw(trial: int, salt: str) -> Draw:
    ring = Ring(random.Random(f"{SEED}/{salt}/{trial}"))
    w = [ring.rng.getrandbits(PAYLOAD) for _ in range(B)]
    return Draw(ring, w, [enc_block(ring, j, w[j]) for j in range(B)])


# ---------------------------------------------------------------------------------------------------------------
# the responders
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Responder:
    """A preprocessor and a sequential responder whose answer to label `i` reads only `i`, its state and the oracle.
    `plan[i]` is what it does for label `i`: `("keep", t)` stores the label without its low `t` bits and searches the
    `2^t` completions with the decoder (`truncProg`; `t = 0` answers the stored label); `("recompute", n)` takes both
    roots from `W` with `n` bits of the exponent (`recBlock`, right only at `n ≥ N`); `("guess",)` answers `0`."""
    name: str
    plan: tuple[tuple, ...]

    def state_bits(self) -> int:
        return sum(W - a[1] for a in self.plan if a[0] == "keep")

    def preprocess(self, d: Draw) -> dict[int, int]:
        return {i: d.code[i] >> a[1] for i, a in enumerate(self.plan) if a[0] == "keep"}

    def answer(self, m: Meter, wi: int, state: dict[int, int], i: int) -> int:
        a = self.plan[i]
        if a[0] == "keep":
            return state[i] if a[1] == 0 else _search(m, i, wi, state[i], a[1])
        if a[0] == "recompute":
            return _recompute(m, i, wi, a[1])
        return 0


def _search(m: Meter, i: int, v: int, h: int, t: int) -> int:
    cands = [c for e in range(1 << t) if (c := (h << t) | e) < P]
    k, tau, *s = m.round([("key", i, "mask"), ("key", i, "offset")] + [("mul", c, c) for c in cands])
    us = [sigma_k(k, sq_fix(c, sc)) for c, sc in zip(cands, s)]
    *s2, target = m.round([("mul", u, u) for u in us] + [("add", v, tau)])
    hits = [c for c, u, su in zip(cands, us, s2) if sq_fix(u, su) == target]
    return hits[0] if hits else 0


def _root(m: Meter, one: int, y: int, n: int) -> int:
    """`y^((p+1)/4)` over the low `n` bits of the exponent, right to left (`powAux`), then the parity fix (`rootSel`)."""
    e, acc, base = (P + 1) // 4, one, y
    for bit in range(n):
        prod, base = m.round([("mul", acc, base), ("mul", base, base)])
        acc = prod if e >> bit & 1 else acc
    (r2,) = m.round([("mul", acc, acc)])
    odd = r2 != y
    return acc if (acc % 2 == 1) == odd else neg_l(acc)


def _recompute(m: Meter, i: int, v: int, n: int) -> int:
    k, tau, one = m.round([("key", i, "mask"), ("key", i, "offset"), ("one",)])
    (y,) = m.round([("add", v, tau)])
    return _root(m, one, sigma_k(k, _root(m, one, y, n)), n)


def keep_first(name: str, count: int, t: int, rest: tuple = ("guess",)) -> Responder:
    return Responder(name, tuple(("keep", t) if i < count else rest for i in range(B)))


def random_responder(index: int) -> Responder:
    """Within the region: labels in a random order, each kept with a random truncation `t ≤ λ` while the state
    allows, recomputed with fewer than `N` exponent bits, or guessed."""
    rng, plan, budget = random.Random(f"{SEED}/responder/{index}"), [None] * B, S
    for i in rng.sample(range(B), B):
        t = rng.randint(0, LAM)
        if rng.random() < 0.95 and W - t <= budget:
            plan[i], budget = ("keep", t), budget - (W - t)
        else:
            plan[i] = ("recompute", rng.randrange(N)) if rng.random() < 0.5 else ("guess",)
    return Responder(f"random-{index}", tuple(plan))


@dataclass(frozen=True)
class Case:
    responder: Responder
    inside: bool


def cases() -> list:
    inside = [keep_first("store-blocks", S // W, 0),
              keep_first(f"truncation-t{LAM}", min(B, S // (W - LAM)), LAM),
              keep_first("depth-limited-recompute", S // W, 0, ("recompute", N - 1))]
    inside += [random_responder(i) for i in range(RANDOM)]
    outside = [(keep_first("honest", B, 0), "holds all of C, past the state bound"),
               (Responder("recompute", (("recompute", N),) * B), "PousP2Ring.tight: D = 2(n + 1), past D_CAP"),
               (keep_first(f"truncation-t{LAM + 1}", min(B, S // (W - LAM - 1)), LAM + 1),
                "PousP2Ring.not_meetsOn_b19_k88's truncation one bit past the slack: L past L_CAP")]
    return ([pytest.param(Case(r, True), id=r.name) for r in inside]
            + [pytest.param(Case(r, False), id=r.name,
                            marks=pytest.mark.xfail(strict=True, raises=Counterexample, reason=why))
               for r, why in outside])


@dataclass
class Play:
    acceptance: list[Fraction]
    right: list[int] = field(default_factory=list)
    D: int = 0
    Q: int = 0
    L: int = 0


def play(r: Responder) -> Play:
    out = Play([])
    for trial in range(TRIALS):
        d = draw(trial, r.name)
        state, right = r.preprocess(d), 0
        for i in range(B):
            m = Meter(d.ring)
            right += r.answer(m, d.w[i], state, i) == d.code[i]
            out.D, out.Q, out.L = max(out.D, m.D), max(out.Q, m.Q), max(out.L, m.L)
        out.acceptance.append(Fraction(right, B) ** K)
        out.right.append(right)
    return out


def inside_region(r: Responder, p: Play) -> bool:
    return r.state_bits() <= S and p.D <= D_CAP and p.Q <= Q_CAP and p.L <= L_CAP


# ---------------------------------------------------------------------------------------------------------------
# the tests
# ---------------------------------------------------------------------------------------------------------------

def test_the_scaled_point():
    """The constants are the scaled point the module docstring describes, with a recompute's work inside the region
    and the slack short of a whole block."""
    assert P2.P2Params(W, C0, B).p == P and P % 4 == 3
    assert all(pow(a, P - 1, P) == 1 for a in (2, 3, 5, 7, 11, 13))
    assert (S, N, KEPT, K) == (2789, 182, 15, 72)
    assert 4 * N + 4 <= L_CAP and LAM < W / 19 and B * (W - LAM) > S >= B * (W - LAM - 1)


def test_the_model_is_the_reference_codec():
    """At `φ = id` the ring model's encoder and decoder are `schemes/p2.py`'s field: `c = ρ(σ_K(ρ(m + t)))`, and under a
    random labeling the decoder program inverts the encoder (`Correct`) in two multiplication rounds."""
    f = P2._field(P2.P2Params(W, C0, B))
    for identity in (True, False):
        ring = Ring(random.Random(f"{SEED}/model/{identity}"), identity)
        for j in range(B):
            v = ring.rng.getrandbits(PAYLOAD)
            c = enc_block(ring, j, v)
            if identity:
                k, t = ring.keys[j]
                assert c == f.rho(f.sigma(f.rho((v + t) % P), k))
            m = Meter(ring)
            assert dec_block(m, j, c) == v and (m.D, m.Q, m.L) == (2, 2, 3)


@pytest.mark.parametrize("case", cases())
def test_p2_ring_audit_secure_at_a_scaled_point(case: Case):
    """Every responder inside the region is accepted at most `δ` (up to the sampling margin); the ones outside it,
    the conjecture's known-false side, beat it."""
    r = case.responder
    p = play(r)
    assert inside_region(r, p) == case.inside, (r.name, r.state_bits(), p.D, p.Q, p.L)
    estimate = sum(p.acceptance) / len(p.acceptance)
    if estimate > DELTA + Fraction(margin(TRIALS)):
        raise Counterexample(f"{r.name} (state {r.state_bits()} of {S} bits, D={p.D}, Q={p.Q}, L={p.L}) is accepted "
                             f"with probability {float(estimate):.4f} over {TRIALS} draws (seed {SEED}), above "
                             f"δ = {float(DELTA)} + {margin(TRIALS):.3f}; plan {r.plan}")


@pytest.mark.parametrize("case", cases())
def test_p2_ring_round_secure_at_a_scaled_point(case: Case):
    """`Pous.P2.Assumptions.P2RingRoundSecure` at the same point, and so `P2RingRoundSecureWide`, the same statement to
    `2^27` labels, which the scaled point doesn't tell apart, with `1 − p = KEPT/B`: given the first `j` answers
    right, the next uniform challenge is answered right with probability at most `KEPT/B`, at every `j < K`. Given the
    sample `ω` a responder answers each challenge right with probability exactly `s_ω/B`, so over the samples the rate at
    round `j` is `Σ (s_ω/B)^(j+1) / Σ (s_ω/B)^j`, and it is above `KEPT/B` only if some draw has `s_ω > KEPT`. That
    sample would have to lie in the conjecture's `bad`, which at `ε₀ = 2^-137` shows up among `TRIALS` seeded draws with
    probability at most `TRIALS·K·ε₀`, so the rate is compared with `KEPT/B` with no margin. The responders outside
    the region beat it, the side `RoundTests.lean` proves false (`round_tight`, `round_trunc`)."""
    r = case.responder
    p = play(r)
    assert inside_region(r, p) == case.inside, (r.name, r.state_bits(), p.D, p.Q, p.L)
    xs = [Fraction(s, B) for s in p.right]
    for j in range(K):
        if (seen := sum(x ** j for x in xs)) and (rate := sum(x ** (j + 1) for x in xs) / seen) > Fraction(KEPT, B):
            raise Counterexample(f"{r.name} (state {r.state_bits()} of {S} bits, D={p.D}, Q={p.Q}, L={p.L}) answers "
                                 f"round {j + 1} right with probability {float(rate):.4f} given the first {j} over "
                                 f"{TRIALS} draws (seed {SEED}), above 1 − p = {KEPT}/{B}; blocks right {p.right}")
