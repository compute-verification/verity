"""`circuit.h1tile`'s gates implement the same design as the `circuit.eps1_reference` reference (decision 102, one design
across schemes).

The two schemes differ only where decision 102 lets them: the noise map (each derives its own public noise from `seed_j`)
and the receipt view (h1tile's CP-W warp checkpoint, two words a set; the reference's chains, eight a thread).  On the
substance they agree, and that is what these pin: both serve the matmul's exact s8 product unchanged (decision 101), and
both fold the identical H1 sets (`segment.call_words`) with fx1.  `test_circuit_h1tile` shows h1tile's gates equal its word
reference (`h1tile.reference`); these show that word reference is the eps1 reference's design, so the gates are.  Pure
Python, fast (no Boolean lowering)."""
from __future__ import annotations

import hashlib
import random

import pytest

from verity.ml.protocols.pouw.circuit import eps1_reference as EPS
from verity.ml.protocols.pouw.circuit import h1tile as H
from verity.ml.protocols.pouw.circuit import segment as SG

pytestmark = pytest.mark.skipif(H.SG is None, reason="H1's receipts (circuit.segment, E2E-120) are not on this tree")

SALT = hashlib.sha512(b"salt").digest()
ROOTS = (hashlib.sha512(b"A0").digest(), hashlib.sha512(b"B0").digest())


def _sx8(v: int) -> int:
    v &= 0xFF
    return v - 256 if v & 0x80 else v


def _dot(a, b) -> int:
    """The exact int8 dot of two rows, mod 2^32, computed independently of `segment`."""
    return sum(_sx8(x) * _sx8(w) for x, w in zip(a, b, strict=True)) & SG.M32


def _rows(rng, n, k, lo, hi):
    return [[rng.randrange(lo, hi + 1) & 0xFF for _ in range(k)] for _ in range(n)]


def test_one_design_across_schemes():
    """Both name the one scheme and the one fold, h1tile takes the s8 Boolean kinds the reference's arm serves, and the only
    difference is the receipt view (CP-W, two words a set, vs chains, eight a thread)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops import s8 as OS8

    assert H.SCHEME == EPS.SCHEME == "eps1-sm120-h1fx1"
    assert H.FOLD == EPS.FOLD == "fx1"
    assert EPS.arm(64) == bind(OS8.S8Dot, **OS8.SMOKE_EDITED)          # the reference's arm is the s8 family
    assert H.KINDS == ("GemmAccumulatorS8_v2", "S8Dot_v2")            # h1tile takes the s8 kinds on bits
    assert EPS.OUT == "chains"                                        # the reference's receipt view
    _z, rc = H.reference(_rows(random.Random(0), 4, 64, -1, 1), _rows(random.Random(1), 8, 64, -1, 1), SALT, 0, ROOTS)
    assert len(rc) == 2 * H.Layout(4, 8, 64).sets                    # h1tile's CP-W: two words a set, not eight a thread


def test_both_serve_the_exact_s8_product():
    """Decision 101: the matmul keeps serving its exact product.  h1tile's served z is each cell's exact int8 dot, the
    product the reference's arm (`S8Dot{COMMIT "prefix"}`) serves; no noise reaches it."""
    rng = random.Random(102)
    a, b = _rows(rng, 4, 128, -127, 127), _rows(rng, 16, 128, -127, 127)
    z, _rc = H.reference(a, b, SALT, 3, ROOTS)
    want = [_dot(a[i], b[j]) for i in range(4) for j in range(16)]
    assert z == want                                                 # h1tile serves the exact product
    assert want == [SG.seg_dot(0, a[i], b[j]) for i in range(4) for j in range(16)]   # the same product the arm serves


def test_both_fold_the_same_H1_sets_with_fx1():
    """The substance of the design.  For one thread's lane, fed the same operands and the same noise, the reference's fold
    path (`eps1_reference.vector`'s `segment.lane_words`) and h1tile's (`h1tile.reference`'s `segment.call_words`) build the
    identical H1 sets (T_0 .. T_{KS-1}, O', D_1 .. D_{m-1}) and fold them to the identical fx1 state.  h1tile reads that one
    state as CP-W and the reference as `chain_words`; both are views of the same fold."""
    rng = random.Random(83)
    ks = 4
    m = SG.n_digits(ks)
    # one thread's committed leaves and its one-hot row and column selects (the reference's vector path)
    x = [_rows(rng, SG.RL, 32, -127, 126) for _ in range(ks)]
    w = [_rows(rng, SG.RL, 32, -60, 60) for _ in range(ks)]
    xsel = EPS._onehot(16, rng.randrange(16))
    wsel = EPS._onehot(8, rng.randrange(8)) + [0] * 8
    # one explicit noise, shared by both paths: dither b, coin c, keyed word u, digit coins z, fx1 keys
    b = [[[rng.getrandbits(1) for _ in range(32)] for _ in range(4)] for _ in range(ks)]
    c = [[[rng.getrandbits(1) for _ in range(32)] for _ in range(8)] for _ in range(ks)]
    u = [[rng.randrange(-64, 65) & 0xFF for _ in range(32)] for _ in range(8)]
    z = [[[rng.getrandbits(1) for _ in range(32)] for _ in range(4)] for _ in range(m - 1)]
    keys = [rng.getrandbits(32) for _ in range(16)]

    # the reference folds lane_words of the leaves and selects (eps1_reference.vector)
    sets_ref = SG.lane_words(x, w, xsel, b, wsel, c, u, z)
    # h1tile folds call_words of the thread's rows directly (h1tile.reference, per lane)
    xr, wr = zip(*[SG.thread_rows(x[j], xsel, w[j], wsel) for j in range(ks)], strict=True)
    sets_h1 = SG.call_words(list(xr), b, list(wr), c, u, z)

    assert sets_ref == sets_h1
    assert len(sets_ref) == ks + m                                   # KS prefixes, O', D_1 .. D_{m-1}

    st_ref = list(keys)
    for p in sets_ref:
        st_ref = SG.absorb(EPS.FOLD, st_ref, p)
    st_h1 = list(keys)
    for p in sets_h1:
        st_h1 = SG.absorb(H.FOLD, st_h1, p)
    assert st_ref == st_h1                                           # one fx1 fold of the one set of sets

    # the two receipt views are each a deterministic read of that one state, and they differ (the allowed difference)
    chains = SG.chain_words(st_ref)
    assert len(chains) == SG.NCH == 8                                # the reference's view: eight chain words a thread
    assert chains != st_ref[:SG.NCH]                                 # a read of y and z, not the raw state
