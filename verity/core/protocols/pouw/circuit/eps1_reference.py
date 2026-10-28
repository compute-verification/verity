"""`eps1-sm120-h1fx1` on words: a non-registered reference for the first EPS1 design, sm_120's H1 + fx1 (decision 102).

The one registered `eps1-sm120-h1fx1` is `h1tile`'s compile-pass scheme (`compile.SCHEMES`); decision 102 is one design across
schemes, so there is one registration.  This module is kept only as that design on words and on sampled proofs' primitives
(`circuit.segment`): a second, independent implementation that `h1tile`'s gates are tested against (`test_h1tile_matches_eps1`),
and the home of the scheme's attack catalog (`test_eps1_attacks`).  Nothing registers it, and `circuit.traced_as` does not
reach it.

Under decision 101 PoUW is a compile pass over the served Program, not an executor: a matmul `A·Bᵀ` keeps serving its exact
s8 product, and PoUW adds, around it, a per-call seed, the noise the seed derives, a committed receipt and one spot check.
Everything else is sampled proofs': hm96 leaves, the live sampling coin, one audit, one ε.

**The seed** (decision 101, with §5 item 10's call-index binding):

    seed_j = SHA-512(TAG ‖ salt ‖ forward ‖ position ‖ root(A_j) ‖ root(B_j))

for matmul j, with `salt` the epoch coin, `(forward, position)` the call's place in the serve (so two calls on the same
operand roots take different noise, `test_eps1_attacks`), and `root(A_j)`, `root(B_j)` the operands' hiding commitments.
The verifier computes the seed and, from it, the noise (`noise`), and passes the noise as a public input: there is no hash
in gates, and the noise is a function of a coin and two commitments, each of which may be public (the everything-ZK rule's
public kinds).  ncp-v2 instead derives its noise in gates (`ncp2.NcpFormX` from a two-level key).

The receipt folds take their keys, and their chains their starts, from a second seed of the same inputs under its own tag
(decision 121; main's `seeds.PouwUnitSeeds` returns the pair):

    σ_R = SHA-512(RECEIPT_TAG ‖ salt ‖ forward ‖ position ‖ root(A_j) ‖ root(B_j))

so no fold key is a function of the noise seed seed_j.  `noise` derives the dither, coins and keyed word from seed_j and the
fx1 fold keys `st` from σ_R (`receipt_seed`).

**The served arm and the receipt.** sm_120's H1 (co32c-u16, decision 82; `benchmarks/pouw/edited_gemm/cgemm_ref.py`) serves
`O = (W + c) Xᵀ` with the dithered `X = V + b` and `W' = W + c + U`, `U` a keyed per-row word with period 32 along k
(fx1, decision 83: one keyed IMAD a word before duo's fold).  The arm is the s8 family at its noised layout, `S8Dot_v1`
with `NOISE "weight"`, `COMMIT "prefix"` and the byteclamp peel of `U` as its tail (`ops.s8`, #2397); its committed prefix
words `T_0 .. T_{KS-1}` are the receipt the fold reads.  The family is a pure exact matmul: the coin, the noise and the
limbs are its input values, so the arm's own soundness is the s8 census's exact product.

**The spot check** is PIN's `PouwCallCheck_v1{KS, FOLD "fx1", OUT "chains"}` (`circuit.segment`, #2417): a thread's whole
call against its receipt leaf, every absorbed word recomputed from the committed operand leaves and the public noise, from
`T_{-1} = 0`.  It takes no prefix word as witness, so a thread's whole call is one unit and the committed receipt is
binding by SHA-512 alone (a meet-in-the-middle over free witness words, which v2 admits, cannot forge it —
`test_eps1_attacks`, `test_circuit_segment`).

**The unit, the work table and the closure.** A unit is a whole dot product (decision 101): here a thread's call, drawn
under the work law in proportion to `W_ref`, the served MACs it covers (`work_table`; `ops.s8.counts` is the per-coordinate
census, `K` served MACs a dot).  The closure is empty: no unit's soundness rests on another's, because the noise is the
verifier's and every absorbed word is recomputed in the unit's own proof (ncp-v2's digest, node and key units, and their
closures, do not arise).

**The conjecture.** Correct checked values become work through the TS model (`Accounting.Guarantees.TS.*`, the EPS1
designs' conjecture; `CompilerC` for H1's receipt).  Its link to `Pouw.Guarantees.EndToEnd` is open and owned by proofs
with compute accounting writing the statement (POUW-SPEC §5 item 3; `LINK_OPEN`).  This module states the scheme's
circuit; it asserts no guarantee.

**Scope.** This states the design on words (its arm, fold and spot check as Definitions, the seed and noise, the work
table, the closure and vectors) and an `Eps1Traced`, so a test can build the same design two ways and compare.  It is a
reference, not a registration: the served scheme is `h1tile`'s, and the compile pass itself (`pouw.compile`, finding every
matmul Call and replacing it) and the vLLM adapter that serves the compiled Program are POUW-SPEC §4 step-4 work, owned
jointly with circuits and vllm.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops import s8 as OS8
from verity.ml.circuits.ops import prims as P

from ..protocol import DomainError
from . import segment as SG

#: the scheme's id and the kind `circuit.SCHEMES` records for it (its silicon target, decision 102's table)
SCHEME = "eps1-sm120-h1fx1"
KIND = "h1fx1"
#: the seed's domain separator (`seed`); a byte string, as every SHA-512 tag here is
TAG = b"eps1-sm120-h1fx1/seed/v1"
#: σ_R's domain separator (`receipt_seed`, decision 121): the receipt folds' keys and their chains' starts, under a tag of
#: their own (main's `seeds.PouwUnitSeeds` RECEIPT_TAG), never the noise seed's, so no fold key is a function of seed_j
RECEIPT_TAG = b"verity/pouw/receipt/v1"
#: the fold H1's receipt uses and its receipt layout (`segment.PouwCallCheck`): fx1, the eight chain words a thread
FOLD, OUT = "fx1", "chains"
#: the per-dot work the TS conjecture credits is open until proofs link TS to EndToEnd (§5 item 3)
LINK_OPEN = "Accounting.Guarantees.TS.* -> Pouw.Guarantees.EndToEnd (proofs; compute accounting writes the statement)"

M32 = SG.M32


@dataclass(frozen=True)
class Shape:
    """A served matmul `A·Bᵀ`: `m` rows of A, `n` rows of B, contraction `k`.  H1 tiles it 128 × 128 over 512 threads;
    `KS = k / 32` is a thread's segment count."""

    m: int
    k: int
    n: int

    def check(self) -> None:
        if self.k % 32 or self.m % 16 or self.n % 16 or min(self.m, self.k, self.n) <= 0:
            raise DomainError(f"{SCHEME}'s circuit needs 16 | m, 16 | n, 32 | k: {self}")
        if self.ks < 2:
            raise DomainError(f"{SCHEME}: a call checks KS >= 2 (one digit word); k = {self.k}")

    @property
    def ks(self) -> int:
        return self.k // 32


def _u64(x: int) -> bytes:
    return int(x).to_bytes(8, "big")


def seed(salt: bytes, forward: int, position: int, root_a: bytes, root_b: bytes) -> bytes:
    """`seed_j = SHA-512(TAG ‖ salt ‖ forward ‖ position ‖ root(A_j) ‖ root(B_j))` (decision 101, §5 item 10).  `salt` is the
    epoch coin, `root_a`/`root_b` the operands' hiding roots, `(forward, position)` the call's place; all public."""
    return hashlib.sha512(TAG + bytes(salt) + _u64(forward) + _u64(position) + bytes(root_a) + bytes(root_b)).digest()


def receipt_seed(salt: bytes, forward: int, position: int, root_a: bytes, root_b: bytes) -> bytes:
    """`σ_R = SHA-512(RECEIPT_TAG ‖ salt ‖ forward ‖ position ‖ root(A_j) ‖ root(B_j))` (decision 121): seed_j's message under
    the receipt tag.  It keys the receipt folds and starts their chains, so no fold key is a function of the noise seed
    seed_j (main's `seeds.PouwUnitSeeds` returns the pair [seed_j, σ_R], one seed role)."""
    return hashlib.sha512(RECEIPT_TAG + bytes(salt) + _u64(forward) + _u64(position) + bytes(root_a) + bytes(root_b)).digest()


class _Stream:
    """A public pseudorandom byte stream from a seed, SHA-512 only (the one-hash rule): block i is SHA-512(seed ‖ label ‖ i).
    The noise is derived outside gates, so this costs nothing on the soundness path."""

    def __init__(self, seed: bytes, label: bytes) -> None:
        self._seed, self._label, self._i, self._buf = bytes(seed), bytes(label), 0, b""

    def _fill(self) -> None:
        self._buf += hashlib.sha512(self._seed + self._label + _u64(self._i)).digest()
        self._i += 1

    def bytes(self, n: int) -> bytes:
        while len(self._buf) < n:
            self._fill()
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def bits(self, n: int) -> list[int]:
        return [b & 1 for b in self.bytes(n)]

    def u32(self) -> int:
        return int.from_bytes(self.bytes(4), "big") & M32

    def i8word(self, lo: int = -64, hi: int = 64) -> int:
        """A signed byte in [lo, hi] as its u8 code (the keyed word U reaches ±64)."""
        return (lo + self.bytes(1)[0] % (hi - lo + 1)) & 0xFF


def noise(the_seed: bytes, receipt: bytes, shape: Shape, tile: int, ctid: int) -> dict[str, Any]:
    """The public noise of one thread's H1 call: the dither bits `b`, the served coin bits `c`, the keyed word `U` and the
    digit coins `z` from `seed_j` (`the_seed`, decision 101), and the fx1 fold keys `st` (the chains' starts) from `σ_R`
    (`receipt`, decision 121), never from seed_j.  The verifier computes these and passes them as public inputs to
    `PouwCallCheck_v1`; nothing here is a witness.  `tile`, `ctid` place the thread (the draw's, from the same seed)."""
    ks, m = shape.ks, SG.n_digits(shape.ks)
    sb = _Stream(the_seed, b"b")
    sc = _Stream(the_seed, b"c")
    su = _Stream(the_seed, b"u")
    sz = _Stream(the_seed, b"z")
    sf = _Stream(receipt, b"fx1")  # the fold keys from σ_R, never from seed_j (decision 121)
    b = [[sb.bits(32) for _ in range(4)] for _ in range(ks)]
    c = [[sc.bits(32) for _ in range(8)] for _ in range(ks)]
    u = [[su.i8word() for _ in range(32)] for _ in range(8)]
    z = [[sz.bits(32) for _ in range(4)] for _ in range(m - 1)]
    fold_seed = sf.u32()
    st = SG.fold_keys("duo", fold_seed, tile, ctid, 0, 1)  # fx1 folds over duo's y, z keys
    return {"b": b, "c": c, "u": u, "z": z, "st": st, "fold_seed": fold_seed, "tile": tile, "ctid": ctid, "m": m}


# ---- the Definitions ----------------------------------------------------------------------------------------------------

#: H1's byteclamp peel of the keyed word `U` (the s8 family's `LIMBS "byteclamp"` tail, two fresh groups), at K = 32·KS: the
#: dithered X against W' = W + c + U, then S_0 / (S_1, S_2) of U's limbs on fresh accumulators (`ops.s8`, SMOKE_EDITED)
_TAIL = ((2, "fresh", (-127, 127), (-64, 64)), (1, "fresh", (-128, 127), (-64, 64)))


def arm(k: int):
    """H1's served noised arm as the s8 family: `S8Dot_v1{K, L=32, NOISE "weight", LIMBS "byteclamp", COMMIT "prefix"}`, whose
    committed prefix words `T_0 .. T_{KS-1}` are the receipt the fx1 fold reads (`ops.s8`, #2397).  A pure exact s8 matmul:
    its noise, coin and limbs are input values."""
    if k % 32:
        raise DomainError(f"{SCHEME}: 32 | k, not {k}")
    return bind(OS8.S8Dot, K=k, L=32, T0="zero", C=None, DOT=P.BlackwellS8ImmaDot32, A=(-127, 127), B=(-127, 127),
                TAIL=_TAIL, TDOT=P.BlackwellS8ImmaDot32, NOISE="weight", LIMBS="byteclamp", COMMIT="prefix")


def check(ks: int):
    """H1 + fx1's spot check, PIN's `PouwCallCheck_v1{KS, FOLD "fx1", OUT "chains"}` (`circuit.segment`, #2417): a thread's
    whole call against its receipt, every absorbed word recomputed, no prefix word a witness."""
    if ks < 2:
        raise DomainError(f"{SCHEME}: a call checks KS >= 2 (one digit word), not {ks}")
    return bind(SG.PouwCallCheck, KS=ks, FOLD=FOLD, OUT=OUT)


def templates(shape: Shape) -> dict[str, Any]:
    """The scheme's Definitions by role: the served arm, the fold and the spot check (the sampled proof block)."""
    shape.check()
    return {"arm": arm(shape.k), "fold": SG.PouwFx1Absorb, "check": check(shape.ks)}


def unit(shape: Shape):
    """The sampled proof block: a thread's whole call under `PouwCallCheck_v1` (decision 101: a proof block is a whole dot product)."""
    shape.check()
    return check(shape.ks)


# ---- the work table and the closure -------------------------------------------------------------------------------------

def work_ref(shape: Shape) -> dict[str, int]:
    """`W_ref` per dot product: `K` served MACs a output coordinate (`ops.s8.counts`), and the call's and the window's totals.
    The sample law weights a unit by the MACs its thread's call covers; the closure is empty (`closure`)."""
    shape.check()
    per_dot = OS8.counts({"N": 1, "K": shape.k, "TAIL": _TAIL})["served_macs"]  # = K
    entries = 4 * 8  # 32 accumulator words a thread (4 row slots × 8 column slots)
    return {"per_dot": per_dot, "per_call": per_dot * entries, "per_window": per_dot * shape.m * shape.n}


def closure(shape: Shape) -> tuple:
    """Empty under decision 101: the noise is the verifier's and every absorbed word is recomputed in the unit's own proof,
    so no unit's soundness rests on another's (ncp-v2's digest/node/key closures do not arise)."""
    shape.check()
    return ()


# ---- vectors ------------------------------------------------------------------------------------------------------------

def vector(the_seed: bytes, receipt: bytes, shape: Shape, tile: int, ctid: int) -> dict[str, Any]:
    """An honest thread's call and its receipt, the noise from `seed_j` and the fold keys from `σ_R` (`noise`): the committed
    operand leaves (`x`, `w`), the public noise and draw (`xsel`, `wsel`, `b`, `c`, `u`, `z`, `st`), and the receipt leaf
    (`rc`) and the thread's one-hot position (`rsel`).  `check`'s word view returns 1 on it (`segment.call_check`); the attack
    library and the one-stage harness build from this."""
    shape.check()
    ks = shape.ks
    nz = noise(the_seed, receipt, shape, tile, ctid)
    sx = _Stream(the_seed, b"x-rows")
    sw = _Stream(the_seed, b"w-rows")
    x = [[[sx.i8word(-127, 126) for _ in range(32)] for _ in range(SG.RL)] for _ in range(ks)]
    w = [[[sw.i8word(-60, 60) for _ in range(32)] for _ in range(SG.RL)] for _ in range(ks)]
    warp, ln = divmod(ctid, 32)
    wm, wn, g, q = warp // 4, warp % 4, ln >> 2, ln & 3
    xsel = _onehot(16, 8 * (wm % 2) + g)
    wsel = _onehot(8, 4 * (wn % 2) + q) + [0] * 8
    state = list(nz["st"])
    for p in SG.lane_words(x, w, xsel, nz["b"], wsel, nz["c"], nz["u"], nz["z"]):
        state = SG.absorb(FOLD, state, p)
    words = SG.chain_words(state)
    nwords = len(words)
    slot = ctid % (SG.RW // nwords)
    rc = [sx.u32() for _ in range(SG.RW)]
    rc[nwords * slot:nwords * slot + nwords] = words
    rsel = _onehot(SG.RW // nwords, slot)
    return {"x": x, "xsel": xsel, "b": nz["b"], "w": w, "wsel": wsel, "c": nz["c"], "u": nz["u"], "z": nz["z"],
            "st": nz["st"], "rc": rc, "rsel": rsel, "slot": slot, "noise": nz, "shape": shape}


def check_args(v: dict[str, Any]) -> list:
    """`segment.call_check`'s / `PouwCallCheck_v1`'s flat arguments from a vector."""
    return [v["x"], v["xsel"], v["b"], v["w"], v["wsel"], v["c"], v["u"], v["z"], v["st"], v["rc"], v["rsel"]]


def _onehot(n: int, i: int) -> list[int]:
    return [int(j == i) for j in range(n)]


# ---- traced_as ----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Eps1Traced:
    """What sampled proofs traces for one H1 + fx1 call of a shape (the EPS1 analogue of `circuit.Traced`): the served arm,
    the spot check that is the sampled proof block, the work table and the empty closure, each from the shape alone.  The verifier
    builds these itself and never takes the prover's."""

    scheme: str
    shape: Shape
    arm: Any
    check: Any
    work: dict[str, int]
    closure: tuple = field(default_factory=tuple)

    @property
    def kind(self) -> str:
        return KIND

    def seed(self, salt: bytes, forward: int, position: int, root_a: bytes, root_b: bytes) -> bytes:
        return seed(salt, forward, position, root_a, root_b)

    def receipt_seed(self, salt: bytes, forward: int, position: int, root_a: bytes, root_b: bytes) -> bytes:
        return receipt_seed(salt, forward, position, root_a, root_b)

    def noise(self, the_seed: bytes, receipt: bytes, tile: int, ctid: int) -> dict[str, Any]:
        return noise(the_seed, receipt, self.shape, tile, ctid)

    def digest(self) -> str:
        from verity.ml.circuits.codec import program_digest
        from verity.ml.circuits.program import Program

        return program_digest(Program(self.check))


def traced_as(m: int, k: int, n: int) -> Eps1Traced:
    """The circuit sampled proofs traces for one H1 + fx1 call of shape m × k → n, the verifier's own (never a prover's)."""
    shape = Shape(m, k, n)
    shape.check()
    return Eps1Traced(SCHEME, shape, arm(k), check(shape.ks), work_ref(shape), closure(shape))


__all__ = ["FOLD", "KIND", "LINK_OPEN", "OUT", "RECEIPT_TAG", "SCHEME", "TAG", "Eps1Traced", "Shape", "arm", "check",
           "check_args", "closure", "noise", "receipt_seed", "seed", "templates", "traced_as", "unit", "vector", "work_ref"]
