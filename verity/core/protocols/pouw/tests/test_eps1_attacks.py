"""The attack library for `eps1-sm120-h1fx1` (the first EPS1 scheme, sm_120's H1 + fx1): each way a prover might claim a
window's credit without doing its work, as a test on the scheme's spot check and seed.

The scheme serves `O = (W + c) Xᵀ` exactly on sm_120's H1 (co32c-u16, decision 82) with the fx1 fold (decision 83), commits
a thread's receipt (its eight chain words), and draws the thread's whole call as a unit.  The drawn unit's proof is PIN's
`PouwCallCheck_v1{KS, FOLD "fx1", OUT "chains"}` (`circuit.segment`, #2417), which recomputes every absorbed word from the
committed operand leaves and the public noise, from `T_{-1} = 0`; the noise is `eps1_reference.noise` of the per-call seed
`seed_j = SHA-512(TAG ‖ salt ‖ forward ‖ position ‖ root(A_j) ‖ root(B_j))` (decision 101).  Each attack below leaves one
honest call and alters one thing; the scheme's check refuses it.

The windowed check `PouwSegCheck_v3` (another of the scheme's checks) appears in the meet-in-the-middle case: v2's prefix
sets are free witness, which a meet-in-the-middle opens (`test_circuit_segment`); v3 recomputes every word, so there is no
witness to meet, and the forgery fails.

`ATTACKS` is the machine-readable table (the PR cites it): each row is (name, what the cheater alters, why it is refused).
"""
from __future__ import annotations

import hashlib
import random

import numpy as np
import pytest

from verity.core.protocols.pouw.circuit import eps1_reference as EPS
from verity.core.protocols.pouw.circuit import segment as SG

M32 = SG.M32
SALT = bytes(range(32))
ROOT_A, ROOT_B = hashlib.sha512(b"A").digest(), hashlib.sha512(b"B").digest()
SHAPE = EPS.Shape(16, 128, 16)  # KS = 4, two digit words
TILE, CTID = 3, 5


def _honest(shape=SHAPE, forward=0, position=0, tile=TILE, ctid=CTID):
    """An honest thread's call of the scheme: its noise from seed_j and its fold keys from σ_R (decision 121), the vector and
    its `call_check` arguments."""
    sd = EPS.seed(SALT, forward, position, ROOT_A, ROOT_B)
    sr = EPS.receipt_seed(SALT, forward, position, ROOT_A, ROOT_B)
    v = EPS.vector(sd, sr, shape, tile, ctid)
    return v, EPS.check_args(v)


def _ok(args) -> int:
    return SG.call_check(*args, fold=EPS.FOLD, out=EPS.OUT)


def _with(args, **changes) -> list:
    """`call_check` arguments with named positions replaced (deep-copying the one replaced)."""
    names = ["x", "xsel", "b", "w", "wsel", "c", "u", "z", "st", "rc", "rsel"]
    out = list(args)
    for k, val in changes.items():
        out[names.index(k)] = val
    return out


def _forge_receipt(args, words) -> list:
    """The call's receipt leaf with the thread's eight chain words replaced by ``words`` at its committed slot."""
    rc, rsel, n = list(args[9]), args[10], SG.NCH
    slot = next(s for s, bit in enumerate(rsel) if bit & 1)
    rc[n * slot:n * slot + n] = [w & M32 for w in words]
    return _with(args, rc=rc)


def _fold_words(args, sets) -> list[int]:
    """The eight chain words the honest fold would commit over ``sets`` (fx1 from the thread's keys)."""
    state = list(args[8])
    for p in sets:
        state = SG.absorb(EPS.FOLD, state, p)
    return SG.chain_words(state)


# --- the honest baseline ------------------------------------------------------------------------------------------------------

def test_honest_call_is_accepted():
    """The scheme accepts an honest thread's call: `call_check` is 1 on `eps1_reference.vector`'s receipt, and the Definition agrees."""
    _v, args = _honest()
    assert _ok(args) == 1


# --- the attack catalog -------------------------------------------------------------------------------------------------------

def _skipped_words(args):
    """The prover folds one fewer served segment (it skips a segment's MACs) but claims the full receipt."""
    sets = SG.lane_words(args[0], args[3], args[1], args[2], args[4], args[5], args[6], args[7])
    return _forge_receipt(args, _fold_words(args, sets[1:]))  # T_0 dropped


def _forged_receipt(args):
    """One committed receipt word has a bit flipped."""
    rc = list(args[9])
    rsel = args[10]
    slot = next(s for s, bit in enumerate(rsel) if bit & 1)
    rc[SG.NCH * slot] ^= 1 << 17
    return _with(args, rc=rc)


def _swapped_weights(args):
    """The receipt is folded over a different weight than the committed `w` leaf (two of its segment leaves swapped)."""
    w = [list(leaf) for leaf in args[3]]
    w[0], w[1] = w[1], w[0]
    swapped = _with(args, w=w)
    sets = SG.lane_words(swapped[0], swapped[3], swapped[1], swapped[2], swapped[4], swapped[5], swapped[6], swapped[7])
    return _forge_receipt(args, _fold_words(args, sets))  # committed w unchanged; receipt from the swap


def _skipped_filler(args):
    """The prover omits one of H1's residue-digit filler words (a `D_d`) from the fold, doing less of the checked work."""
    sets = SG.lane_words(args[0], args[3], args[1], args[2], args[4], args[5], args[6], args[7])
    return _forge_receipt(args, _fold_words(args, sets[:-1]))  # the last digit word D_{m-1} dropped


def _replay(args):
    """A receipt honest for the call at one position is reused at another position in the same leaf."""
    rsel = args[10]
    slot = next(s for s, bit in enumerate(rsel) if bit & 1)
    other = (slot + 1) % len(rsel)
    return _with(args, rsel=EPS._onehot(len(rsel), other))


def _commit_after_salt(args):
    """The prover commits and folds under a salt different from the epoch's: the seed, and so the noise, change, and the
    receipt no longer matches the verifier's recomputation under the real salt."""
    other_salt = hashlib.sha512(b"other-salt").digest()[:32]
    other = EPS.seed(other_salt, 0, 0, ROOT_A, ROOT_B)
    other_r = EPS.receipt_seed(other_salt, 0, 0, ROOT_A, ROOT_B)
    v2 = EPS.vector(other, other_r, SHAPE, TILE, CTID)
    return _forge_receipt(args, _fold_words(args, SG.lane_words(
        args[0], args[3], args[1], v2["b"], args[4], v2["c"], v2["u"], v2["z"])))


def _reused_operand_roots(args):
    """A second call on the same operand roots but at a different (forward, position) is served with the first call's
    receipt.  The seed binds (forward, position), so the noise differs and the receipt does not match (POUW-SPEC §5 item
    10: two calls must not share noise)."""
    _v2, args2 = _honest(forward=1, position=1)
    return _with(args2, rc=args[9])  # call 2's operands and noise, but call 1's committed receipt


#: name -> (build the attacked call_check arguments, one-line reason it is refused)
ATTACKS = {
    "skipped-words": (_skipped_words, "the fold recomputes every segment's MACs from the committed operands; a dropped "
                      "segment changes the receipt"),
    "forged-receipt": (_forged_receipt, "the committed receipt word is bound by SHA-512; a flipped bit is not the fold"),
    "swapped-weights": (_swapped_weights, "the fold reads the committed w leaf; a receipt from a different weight does not "
                        "match"),
    "skipped-filler": (_skipped_filler, "H1's residue-digit words are folded too; a dropped D_d changes the receipt"),
    "replay": (_replay, "the receipt is the thread's at its committed slot; reading another slot reads another thread's "
               "word"),
    "commit-after-salt": (_commit_after_salt, "the seed binds the salt; noise under another salt does not fold to the "
                          "committed receipt"),
    "reused-operand-roots": (_reused_operand_roots, "the seed binds (forward, position); a second call's noise differs, so "
                             "the first call's receipt fails"),
}


@pytest.mark.parametrize("name", sorted(ATTACKS))
def test_attack_is_refused(name):
    """Each attack leaves one honest call and alters one thing; the scheme's spot check returns 0."""
    build, _why = ATTACKS[name]
    _v, args = _honest()
    assert _ok(args) == 1, "the baseline is honest"
    assert _ok(build(args)) == 0, name


# --- the seed binds the salt and the call index -------------------------------------------------------------------------------

def test_seed_binds_the_call_index_and_the_salt():
    """`seed_j` separates two calls on the same operand roots (different forward/position) and two epochs (different salt),
    which closes the reuse §5 item 10 names; dropping the index from the hash would collide them."""
    base = EPS.seed(SALT, 0, 0, ROOT_A, ROOT_B)
    assert EPS.seed(SALT, 1, 0, ROOT_A, ROOT_B) != base  # forward binds
    assert EPS.seed(SALT, 0, 1, ROOT_A, ROOT_B) != base  # position binds
    assert EPS.seed(hashlib.sha512(b"s2").digest()[:32], 0, 0, ROOT_A, ROOT_B) != base  # salt binds
    assert EPS.seed(SALT, 0, 0, ROOT_B, ROOT_A) != base  # the operand roots bind, in order

    # the vulnerability the binding closes: an unbound seed (roots only) collides two calls, so one receipt serves both
    def unbound(ra, rb):
        return hashlib.sha512(EPS.TAG + SALT + ra + rb).digest()
    assert unbound(ROOT_A, ROOT_B) == unbound(ROOT_A, ROOT_B)
    _v1, a1 = _honest(forward=0, position=0)
    _v2, a2 = _honest(forward=1, position=1)
    assert a1[8] != a2[8] or a1[5] != a2[5]  # the noise (fold keys / coin bits) differs between the two calls


# --- meet-in-the-middle: v2's free witness vs. the scheme's recomputed check --------------------------------------------------

KS, S = 4, 2


def _onehot(n, i):
    return [int(j == i) for j in range(n)]


def _leaf(rnd, lo=-128, hi=127):
    return [[rnd.randint(lo, hi) & 0xFF for _ in range(32)] for _ in range(SG.RL)]


def _rsel(pos):
    return [(1 << pos % 32) if pos // 32 == k else 0 for k in range(SG.RW // 32)]


def _honest_v3(rnd, ks, s, fold):
    a, b = [_leaf(rnd) for _ in range(ks)], [_leaf(rnd) for _ in range(ks)]
    asel, bsel = _onehot(16, rnd.randrange(16)), _onehot(8, rnd.randrange(8)) + [0] * 8
    u = [[rnd.getrandbits(8) for _ in range(32)] for _ in range(4)]
    f = [[[rnd.getrandbits(1) for _ in range(32)] for _ in range(8)] for _ in range(ks)]
    nw, seed, tile, ctid = ks // s, rnd.getrandbits(32), rnd.randrange(64), rnd.randrange(SG.NCT)
    st = [SG.fold_keys(fold, seed, tile, ctid, w, nw) for w in range(nw)]
    ap, bp = [], []
    for j in range(ks):
        ar, br = SG.thread_rows(a[j], asel, b[j], bsel)
        ap.append([[(x + v) & 0xFF for x, v in zip(ar[r], u[r])] for r in range(4)])
        bp.append([[(x + (1 - 2 * (q & 1))) & 0xFF for x, q in zip(br[c], f[j][c])] for c in range(8)])
    T = SG.prefixes(ap, bp)
    pos, rc = rnd.randrange(SG.RW), []
    for w in range(nw):
        state = SG.absorb(fold, list(st[w]), T[w * s - 1] if w else [0] * 32)
        for j in range(w * s, w * s + s):
            state = SG.absorb(fold, state, T[j])
        leaf = [rnd.getrandbits(32) for _ in range(SG.RW)]
        leaf[pos] = SG.fold_final(fold, state)
        rc.append(leaf)
    return dict(a=a, asel=asel, u=u, b=b, bsel=bsel, f=f, st=st, rc=rc, rsel=_rsel(pos), pos=pos, T=T, ap=ap, bp=bp)


def _rotl_np(x, r):
    return ((x << np.uint64(r)) | (x >> np.uint64(32 - r))) & np.uint64(M32)


def _contrib(y, z, t, w):
    u = (np.uint64(y) ^ w ^ np.uint64(SG.KK[t])) & np.uint64(M32)
    y2 = (u * np.uint64(SG.KA[t]) + np.uint64(z)) & np.uint64(M32)
    return y2 ^ _rotl_np((_rotl_np(u, SG.ROT[t]) + y2) & np.uint64(M32), 16)


def _mitm(st, p, target, rng):
    """LEDGER's opening: solve p's last set's entries 24 and 28 over 2^18 candidates each so duo's fold of p from st ends at
    ``target`` (`test_circuit_segment._mitm`)."""
    state = list(st)
    for ps in p[:-1]:
        state = SG.fold_absorb("duo", state, ps)
    y, z = state[:8], state[8:]
    pre = {}
    for t in range(8):
        h = 4 * (t % 2)
        for e in range(4):
            if (t, e) in ((6, 0), (7, 0)):
                pre[h + e] = (y[h + e], z[h + e])
            d = 0 if 4 * t + e in (24, 28) else p[-1][4 * t + e]
            u = y[h + e] ^ d ^ SG.KK[t]
            y[h + e] = (u * SG.KA[t] + z[h + e]) & M32
            z[h + e] = (SG.rotl(u, SG.ROT[t]) + y[h + e]) & M32
    rest = 0
    for c in range(8):
        if c not in (0, 4):
            rest ^= y[c] ^ SG.rotl(z[c], 16)
    w0 = rng.integers(0, 1 << 32, 1 << 18, dtype=np.uint64)
    w4 = rng.integers(0, 1 << 32, 1 << 18, dtype=np.uint64)
    _, i0, i4 = np.intersect1d(_contrib(*pre[0], 6, w0), _contrib(*pre[4], 7, w4) ^ np.uint64(rest ^ target),
                               return_indices=True)
    assert len(i0), "no meeting point at 2^18 a side"
    last = list(p[-1])
    last[24], last[28] = int(w0[i0[0]]), int(w4[i4[0]])
    return [*p[:-1], last]


def test_mitm_opens_v2_but_the_scheme_refuses_it():
    """A meet-in-the-middle forges a committed window word through `PouwSegCheck_v2`'s free prefix sets; `PouwSegCheck_v3`,
    on the same committed leaves, coin and receipts, recomputes every word and refuses it (and accepts the honest ones)."""
    rnd, rng = random.Random(20261018), np.random.default_rng(309)
    h = _honest_v3(rnd, KS, S, "duo")
    target = rnd.getrandbits(32)
    forged = [list(h["rc"][0]), list(h["rc"][1])]
    forged[1][h["pos"]] = target
    p0 = [rnd.getrandbits(32) for _ in range(32)]
    p1 = [SG.seg_dot(p0[q], h["ap"][2][SG.row_slot(q)], h["bp"][2][SG.col_slot(q)]) for q in range(32)]
    p = _mitm(h["st"][1], [p0, p1, [rnd.getrandbits(32) for _ in range(32)]], target, rng)
    jsel = _onehot(SG.jsel_bits(S), 0)
    assert target != h["rc"][1][h["pos"]]
    assert SG.win_check(h["a"][2], h["asel"], h["u"], h["b"][2], h["bsel"][:8], h["f"][2], p, h["st"][1],
                        forged[1], _onehot(SG.RW, h["pos"]), jsel, 0) == 1   # v2's free witness opens the word
    v3 = [h["a"], h["asel"], h["u"], h["b"], h["bsel"], h["f"], h["st"], forged, h["rsel"]]
    assert SG.seg_check3(*v3) == 0                                          # the scheme's check refuses it
    assert SG.seg_check3(*v3[:7], h["rc"], h["rsel"]) == 1                  # and accepts the honest receipts


# --- the catalog is complete --------------------------------------------------------------------------------------------------

def test_catalog_lists_every_attack_from_the_brief():
    """Every attack the brief names has a row (the `call_check` ones in `ATTACKS`; the meet-in-the-middle has its own test)."""
    assert set(ATTACKS) == {"skipped-words", "forged-receipt", "swapped-weights", "skipped-filler", "replay",
                            "commit-after-salt", "reused-operand-roots"}
