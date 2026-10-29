"""SI-THEORY: an executable reference of the closed in-unit fold (closure), with the provers that attack it.

The unit is the published function M1-M5 of note compute-research/20261029T2203Z-report-silicon-theory:
- `Unit.mma(shape, A, B, C)`: s8 x s8 products summed into wrapping s32 accumulators, acc = A.B + C, exact.
- The fold: a 512-bit state, SHA-512's eight 64-bit working variables. Only an issue of a shape in `advance` moves it
  (closure; the default is the one credited shape, m16n8k32). Each such issue absorbs its shape tag (two words) and then
  every accumulator word, two 32-bit words a SHA-512 round, W raw with K_t added inside the round: half a round a word,
  R80X's rate. Any other shape computes acc and leaves the fold alone.
- `Unit.fold_init(coins)` is the only other writer of the state, and the verifier recomputes its value from the coins.- The seal: SHA-512(salt || domain || tag || issues || state) under a hiding salt.

`F(state, word, coins)` is the per-step form SI-RED's harness takes, state_{t+1} = F(state_t, word_t, coins), one 32-bit
word a step; state None starts from the coins. A credited issue is F over [tag_hi, tag_lo, w_0, ..., w_127] (checked).

Costs are counted in three currencies, kept apart on purpose:
- issues and cycles of the unit (`ISSUE_CYCLES`, operand-independent: the IssueCycles premise, a measured device fact);
- W1 register writes for software (a SHA-512 round 28, R80X's count; forming a zero-data word 0.109, COUNTER's `decomp`;
  a SHA-512 compression 3,532);
- energy and power-capped seconds are not modelled: they depend on the operands (see the note).

    python3 closure_ref.py --json OUT.json [--steps T]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass, replace

import numpy as np

M64 = (1 << 64) - 1
M32 = (1 << 32) - 1

SHAPE_CREDITED = (16, 8, 32)
SHAPES = {SHAPE_CREDITED: 1.0, (16, 8, 16): 0.5, (16, 8, 8): 0.25}  # issue cycles relative to k32 (dense, int8)
ISSUE_CYCLES = SHAPES
WORDS_PER_ISSUE = SHAPE_CREDITED[0] * SHAPE_CREDITED[1]
MACS_PER_ISSUE = WORDS_PER_ISSUE * SHAPE_CREDITED[2]
W1_PER_ISSUE = float(WORDS_PER_ISSUE)  # a k32 step writes one accumulator: 1 W1 a word
W1_PER_ROUND = 28.0  # R80X's software count for one SHA-512 round (note 20261029T2120Z-report-hash-fold-rounds)
W1_DECOMP_FORM = 0.109  # COUNTER's decomp: forming a zero-data word without writing it
W1_PER_COMPRESSION = 3532.0  # a chained SHA-512 compression in W1 (pouw_space price.sha-3532, as SI-FOLD uses)

DOMAIN_INIT = b"verity/si-theory/fold-init/v0"
DOMAIN_SEAL = b"verity/si-theory/fold-seal/v0"


def _iroot(n: int, r: int) -> int:
    lo, hi = 0, 1 << (n.bit_length() // r + 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid**r <= n:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _primes(n: int) -> list[int]:
    out, k = [], 2
    while len(out) < n:
        if all(k % p for p in out if p * p <= k):
            out.append(k)
        k += 1
    return out


K = tuple(_iroot(p << 192, 3) & M64 for p in _primes(80))  # frac(cbrt(p)) * 2^64, FIPS 180-4
IV = tuple(_iroot(p << 128, 2) & M64 for p in _primes(8))  # frac(sqrt(p)) * 2^64


def _rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (64 - n))) & M64


def sha512_round(v: tuple[int, ...], w: int, k: int) -> tuple[int, ...]:
    a, b, c, d, e, f, g, h = v
    s1 = _rotr(e, 14) ^ _rotr(e, 18) ^ _rotr(e, 41)
    ch = (e & f) ^ (~e & g & M64)
    t1 = (h + s1 + ch + k + w) & M64
    s0 = _rotr(a, 28) ^ _rotr(a, 34) ^ _rotr(a, 39)
    maj = (a & b) ^ (a & c) ^ (b & c)
    return ((t1 + s0 + maj) & M64, a, b, c, (d + t1) & M64, e, f, g)


def sha512_digest(msg: bytes) -> bytes:
    """SHA-512 from `sha512_round`, only to check the round against hashlib."""
    ml = len(msg)
    msg = msg + b"\x80" + b"\x00" * ((111 - ml) % 128) + (8 * ml).to_bytes(16, "big")
    hs = IV
    for off in range(0, len(msg), 128):
        w = list(struct.unpack(">16Q", msg[off:off + 128]))
        for t in range(16, 80):
            s0 = _rotr(w[t - 15], 1) ^ _rotr(w[t - 15], 8) ^ (w[t - 15] >> 7)
            s1 = _rotr(w[t - 2], 19) ^ _rotr(w[t - 2], 61) ^ (w[t - 2] >> 6)
            w.append((w[t - 16] + s0 + w[t - 7] + s1) & M64)
        v = hs
        for t in range(80):
            v = sha512_round(v, w[t], K[t])
        hs = tuple((x + y) & M64 for x, y in zip(hs, v))
    return struct.pack(">8Q", *hs)


@dataclass(frozen=True)
class FoldState:
    v: tuple[int, ...]
    t: int  # rounds done; the round constant is K[t mod 80]
    pending: int | None  # the even word of a pair, waiting for its partner


def fold_init(coins: bytes) -> FoldState:
    return FoldState(struct.unpack(">8Q", hashlib.sha512(DOMAIN_INIT + coins).digest()), 0, None)


def F(state: FoldState | None, word: int, coins: bytes) -> FoldState:
    """The per-step update: absorb one 32-bit word. Two words make one 64-bit W, and W enters raw (K_t is added inside)."""
    s = fold_init(coins) if state is None else state
    if s.pending is None:
        return replace(s, pending=word & M32)
    w = (s.pending << 32) | (word & M32)
    return FoldState(sha512_round(s.v, w, K[s.t % 80]), s.t + 1, None)


def shape_tag(shape: tuple[int, int, int], fmt: str = "s8s8s32") -> tuple[int, int]:
    d = hashlib.sha512(f"{fmt}/m{shape[0]}n{shape[1]}k{shape[2]}".encode()).digest()
    return struct.unpack(">II", d[:8])


def issue_words(shape: tuple[int, int, int], acc: np.ndarray) -> list[int]:
    return [*shape_tag(shape), *(int(x) for x in acc.astype(np.uint32).ravel())]


class Unit:
    """The published function. `advance` is the set of shapes that move the fold: closure is advance = {k32}."""

    def __init__(self, advance: frozenset = frozenset({SHAPE_CREDITED}), same_tag: bool = False):
        self.advance = advance
        self.same_tag = same_tag  # the closure-off variant: every advancing shape absorbs the credited tag
        self.state: FoldState | None = None
        self.coins = b""
        self.issues = 0
        self.cycles = 0.0

    def fold_init(self, coins: bytes) -> None:
        self.coins, self.state = coins, fold_init(coins)

    def mma(self, shape: tuple[int, int, int], A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
        m, n, k = shape
        assert A.shape == (m, k) and B.shape == (k, n) and C.shape == (m, n), "shape mismatch"
        acc = ((A.astype(np.int64) @ B.astype(np.int64) + C.astype(np.int64)) & M32).astype(np.uint32).view(np.int32)
        self.cycles += ISSUE_CYCLES[shape]
        if shape in self.advance:
            assert self.state is not None, "fold not initialised"
            for wd in issue_words(SHAPE_CREDITED if self.same_tag else shape, acc):
                self.state = F(self.state, wd, self.coins)
            self.issues += 1
        return acc


def seal(state: FoldState, issues: int, salt: bytes) -> bytes:
    assert state.pending is None
    tag = struct.pack(">II", *shape_tag(SHAPE_CREDITED))
    return hashlib.sha512(salt + DOMAIN_SEAL + tag + issues.to_bytes(8, "big") + struct.pack(">8Q", *state.v)).digest()


@dataclass
class Statement:
    """What is committed before the coins: one 16 x 8 output tile over T k32 steps, C_0 = 0. In the protocol A and B
    are committed under hiding leaves and the relation below is proved in zero knowledge; here they are in the clear."""
    A: list[np.ndarray]
    B: list[np.ndarray]
    coins: bytes
    salt: bytes


def honest_words(st: Statement) -> list[np.ndarray]:
    m, n, _ = SHAPE_CREDITED
    acc, out = np.zeros((m, n), np.int32), []
    for a, b in zip(st.A, st.B):
        acc = ((a.astype(np.int64) @ b.astype(np.int64) + acc.astype(np.int64)) & M32).astype(np.uint32).view(np.int32)
        out.append(acc)
    return out


def verify(st: Statement, sl: bytes, claimed_issues: int) -> tuple[bool, int]:
    """The relation: the seal is the closed unit's on the committed operands from coins-derived state. Credit: issues."""
    s = fold_init(st.coins)
    for acc in honest_words(st):
        for wd in issue_words(SHAPE_CREDITED, acc):
            s = F(s, wd, st.coins)
    ok = claimed_issues == len(st.A) and sl == seal(s, claimed_issues, st.salt)
    return ok, (claimed_issues * MACS_PER_ISSUE if ok else 0)


@dataclass
class Outcome:
    name: str
    seal: bytes
    claimed: int
    issues: int  # credited issues on the unit
    cycles: float  # unit cycles, relative to one k32 issue
    sw_w1: float  # software W1 writes beside the unit (the seal excluded: every prover pays it)
    note: str


def prove_honest(st: Statement, unit: Unit | None = None) -> Outcome:
    u = unit or Unit()
    u.fold_init(st.coins)
    m, n, _ = SHAPE_CREDITED
    acc = np.zeros((m, n), np.int32)
    for a, b in zip(st.A, st.B):
        acc = u.mma(SHAPE_CREDITED, a, b, acc)
    return Outcome("honest", seal(u.state, u.issues, st.salt), u.issues, u.issues, u.cycles, 0.0, "served operands")


def prove_c_inject(st: Statement, shape: tuple[int, int, int] = SHAPE_CREDITED, unit: Unit | None = None,
                   name: str = "c-inject") -> Outcome:
    """A = B = 0 and C = the honest word tile each step. The words are formed elsewhere; priced here at decomp's
    zero-data rate, a lower bound (on served data forming them is the useful MMA itself)."""
    u = unit or Unit()
    u.fold_init(st.coins)
    m, n, k = shape
    za, zb = np.zeros((m, k), np.int8), np.zeros((k, n), np.int8)
    for acc in honest_words(st):
        u.mma(shape, za, zb, acc)
    form = W1_DECOMP_FORM * WORDS_PER_ISSUE * len(st.A)
    return Outcome(name, seal(u.state, len(st.A), st.salt), len(st.A), u.issues, u.cycles, form,
                   f"zero operands, words injected through C on m{m}n{n}k{k}")


def prove_split_k16(st: Statement) -> Outcome:
    """Each k32 step as two k16 issues on the served operands: the same words, under closure the fold doesn't move."""
    u = Unit()
    u.fold_init(st.coins)
    m, n, _ = SHAPE_CREDITED
    acc = np.zeros((m, n), np.int32)
    for a, b in zip(st.A, st.B):
        acc = u.mma((16, 8, 16), a[:, :16], b[:16, :], acc)
        acc = u.mma((16, 8, 16), a[:, 16:], b[16:, :], acc)
    return Outcome("two k16 issues a step", seal(u.state, len(st.A), st.salt), len(st.A), u.issues, u.cycles, 0.0,
                   "the served words from a non-credited shape")


def prove_skip(st: Statement) -> Outcome:
    """Issue every other step only, injecting the step's final word: half the credited issues."""
    u = Unit()
    u.fold_init(st.coins)
    m, n, k = SHAPE_CREDITED
    za, zb = np.zeros((m, k), np.int8), np.zeros((k, n), np.int8)
    for acc in honest_words(st)[1::2]:
        u.mma(SHAPE_CREDITED, za, zb, acc)
    return Outcome("skip half", seal(u.state, len(st.A), st.salt), len(st.A), u.issues, u.cycles, 0.0,
                   "claims T issues, issues T/2")


def prove_software(st: Statement, from_step: int = 0, form_w1: float = W1_DECOMP_FORM) -> Outcome:
    """The bypass: the unit for steps < from_step, then the fold in software. from_step 0 is the pure bypass. Forming a
    word costs `form_w1` (decomp's 0.109 at zero data; 1, the MMA's own write, on served data)."""
    u = Unit()
    u.fold_init(st.coins)
    words = honest_words(st)
    m, n, _ = SHAPE_CREDITED
    acc = np.zeros((m, n), np.int32)
    for a, b in zip(st.A[:from_step], st.B[:from_step]):
        acc = u.mma(SHAPE_CREDITED, a, b, acc)
    s = u.state
    rounds = 0
    for acc in words[from_step:]:
        for wd in issue_words(SHAPE_CREDITED, acc):
            s = F(s, wd, st.coins)
        rounds += (WORDS_PER_ISSUE + 2) // 2
    sw = rounds * W1_PER_ROUND + form_w1 * WORDS_PER_ISSUE * (len(st.A) - from_step)
    name = "software bypass" if from_step == 0 else f"hybrid (unit to step {from_step})"
    return Outcome(name, seal(s, len(st.A), st.salt), len(st.A), u.issues, u.cycles, sw, "no unit after the hand-off")


def prove_wrong_init(st: Statement) -> Outcome:
    o = prove_honest(replace(st, coins=st.coins + b"!"))
    return replace(o, name="stale or chosen fold state", note="sigma_0 not from this call's coins")


def statement(rng: np.random.Generator, T: int, zero: bool = False) -> Statement:
    m, n, k = SHAPE_CREDITED
    if zero:
        A = [np.zeros((m, k), np.int8) for _ in range(T)]
        B = [np.zeros((k, n), np.int8) for _ in range(T)]
    else:
        A = [rng.integers(-128, 128, (m, k), dtype=np.int64).astype(np.int8) for _ in range(T)]
        B = [rng.integers(-128, 128, (k, n), dtype=np.int64).astype(np.int8) for _ in range(T)]
    return Statement(A, B, rng.bytes(32), rng.bytes(64))


def run(T: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    st = statement(rng, T)
    zst = statement(rng, T, zero=True)
    checks: dict[str, bool] = {}

    msgs = [rng.bytes(int(x)) for x in (0, 111, 112, 127, 128, 300)]
    checks["SHA-512 from sha512_round equals hashlib on 6 messages (round, K and IV checked)"] = all(
        sha512_digest(x) == hashlib.sha512(x).digest() for x in msgs)

    u = Unit()
    u.fold_init(st.coins)
    m, n, k = SHAPE_CREDITED
    acc = u.mma(SHAPE_CREDITED, st.A[0], st.B[0], np.zeros((m, n), np.int32))
    s = None
    for wd in issue_words(SHAPE_CREDITED, acc):
        s = F(s, wd, st.coins)
    checks["the unit's issue equals F over [tag, words] (per-step form)"] = s == u.state

    full = sum(a.astype(np.int64) @ b.astype(np.int64) for a, b in zip(st.A, st.B))
    checks["the served output is exact: acc_T = sum A_t.B_t mod 2^32"] = bool(
        np.array_equal(honest_words(st)[-1].astype(np.uint32), (full & M32).astype(np.uint32)))

    outcomes = []
    h = prove_honest(st)
    outcomes.append((h, st))
    ok, credit = verify(st, h.seal, h.claimed)
    checks[f"honest verifies; credit = T x {MACS_PER_ISSUE} MACs"] = ok and credit == T * MACS_PER_ISSUE

    z = prove_honest(zst)
    z = replace(z, name="zero operands (zero-data statement)", note="A = B = 0 committed")
    outcomes.append((z, zst))
    checks["zero operands verify against a zero-data commitment, at the honest issue count and cycles"] = (
        verify(zst, z.seal, z.claimed)[0] and z.issues == h.issues and z.cycles == h.cycles)
    z2 = prove_honest(replace(st, A=zst.A, B=zst.B))
    checks["zero operands fail against the served commitment"] = not verify(st, z2.seal, T)[0]

    ci = prove_c_inject(st)
    outcomes.append((ci, st))
    checks["C-injection verifies only with every credited issue at full cycles (SI-FOLD's E4)"] = (
        verify(st, ci.seal, ci.claimed)[0] and ci.issues == T and ci.cycles == h.cycles)

    ss = prove_split_k16(st)
    outcomes.append((ss, st))
    checks["two k16 issues a step fail under closure (the fold doesn't move)"] = (
        not verify(st, ss.seal, ss.claimed)[0] and ss.issues == 0)

    k16 = (16, 8, 16)
    ck = prove_c_inject(st, k16, name="c-inject on k16")
    outcomes.append((ck, st))
    checks["C-injection on k16 fails under closure"] = not verify(st, ck.seal, ck.claimed)[0]

    open_tagged = prove_c_inject(st, k16, Unit(advance=frozenset({SHAPE_CREDITED, k16})),
                                 name="c-inject on k16, k16 advances under its own tag")
    outcomes.append((open_tagged, st))
    checks["k16 advancing under its own shape tag still fails (absorbing the tag is closure)"] = (
        not verify(st, open_tagged.seal, open_tagged.claimed)[0])

    open_ = prove_c_inject(st, k16, Unit(advance=frozenset({SHAPE_CREDITED, k16}), same_tag=True),
                           name="c-inject on k16, closure off")
    outcomes.append((open_, st))
    checks["closure off (k16 advances under the k32 tag): served words pass at half the cycles"] = (
        verify(st, open_.seal, open_.claimed)[0] and open_.cycles == 0.5 * h.cycles)

    sk = prove_skip(st)
    outcomes.append((sk, st))
    checks["skipping half the issues fails"] = not verify(st, sk.seal, sk.claimed)[0]

    wi = prove_wrong_init(st)
    outcomes.append((wi, st))
    checks["a fold state not derived from this call's coins fails"] = not verify(st, wi.seal, wi.claimed)[0]

    sw = prove_software(zst)
    outcomes.append((sw, zst))
    hz = h.issues * W1_PER_ISSUE
    margin = sw.sw_w1 / hz
    checks["the software bypass verifies (zero data, no unit)"] = verify(zst, sw.seal, sw.claimed)[0]
    checks["the bypass costs at least 14x the unit's W1 (margin M)"] = margin >= 14.0

    hy = prove_software(st, from_step=T // 2, form_w1=1.0)
    outcomes.append((hy, st))
    checks["the hybrid verifies, and its software half costs more than the unit's half"] = (
        verify(st, hy.seal, hy.claimed)[0] and hy.issues == T // 2 and hy.sw_w1 > (T - T // 2) * W1_PER_ISSUE)

    st2 = replace(st, A=[st.A[1], st.A[0], *st.A[2:]], B=[st.B[1], st.B[0], *st.B[2:]])
    checks["reordering two steps changes the seal"] = prove_honest(st2).seal != h.seal
    flipA = [a.copy() for a in st.A]
    flipA[T // 2][3, 5] ^= 1
    checks["one operand bit changes the seal"] = prove_honest(replace(st, A=flipA)).seal != h.seal
    checks["fresh coins on zero data give a fresh seal (no precomputation across calls)"] = (
        prove_honest(replace(zst, coins=zst.coins[::-1])).seal != z.seal)

    words = T * WORDS_PER_ISSUE
    seal_w1 = 2 * W1_PER_COMPRESSION  # salt 64 + domain + tag + issues + state 64 > 111 bytes: two compressions
    rows = []
    for o, s_ in outcomes:
        ok_, credit_ = verify(s_, o.seal, o.claimed)
        rows.append({
            "prover": o.name, "verifies": ok_, "credit_macs": credit_, "credited_issues": o.issues,
            "unit_cycles_vs_honest": round(o.cycles / h.cycles, 4),
            "w1_unit": o.issues * W1_PER_ISSUE, "w1_software": round(o.sw_w1, 1),
            "w1_total_vs_honest": round((o.issues * W1_PER_ISSUE + o.sw_w1) / hz, 3), "note": o.note,
        })
    return {
        "params": {"T": T, "seed": seed, "shape": SHAPE_CREDITED, "words": words, "macs": T * MACS_PER_ISSUE},
        "margin_M_w1": round(margin, 3),
        "h_seal": round(seal_w1 / words, 4),
        "checks": checks,
        "provers": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True)
    ap.add_argument("--steps", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    res = {f"T{T}": run(T, args.seed + T) for T in args.steps}
    with open(args.json, "w") as f:
        json.dump(res, f, indent=1, default=str)
    bad = 0
    for key, r in res.items():
        print(f"== {key}: {r['params']['words']} words, margin M = {r['margin_M_w1']}, h_seal = {r['h_seal']}")
        for c, v in r["checks"].items():
            bad += not v
            print(f"  [{'ok' if v else 'FAIL'}] {c}")
        print("  prover | verifies | credited issues | unit cycles / honest | W1 / honest")
        for p in r["provers"]:
            print(f"  {p['prover']} | {p['verifies']} | {p['credited_issues']} | {p['unit_cycles_vs_honest']} | "
                  f"{p['w1_total_vs_honest']}")
    print(f"{sum(len(r['checks']) for r in res.values()) - bad} / {sum(len(r['checks']) for r in res.values())} checks")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
