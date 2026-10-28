"""The receipt spot checks (`verity.core.protocols.pouw.circuit.segment`): the meet-in-the-middle check
opens any committed window word through `PouwSegCheck_v2`, whose prefix sets are
witness, and `PouwSegCheck_v3`, which recomputes every absorbed word, refuses the same committed data; each new check
on bits is its word view, 1 on an honest instance and 0 once one receipt bit changes; and census/v1 counts the segment dots
on bits as unserved `tc:int8` steps."""
from __future__ import annotations

import random

import numpy as np
import pytest

from verity.core.protocols.pouw.circuit import segment as SG
from verity.ml.circuits.census import COUNTERS, PRODUCTS, census
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array, Value

M32 = SG.M32
KS, S = 4, 2


def _bits(t, v):
    if isinstance(t, Value):
        return int(v) & 1
    if isinstance(v, int):
        return [(v >> i) & 1 for i in range(t.n)]
    return [_bits(t.elem, x) for x in v]


def _on_bits(d, args) -> int:
    params, _ = d.signature()
    ins = [np.array([_bits(t, a)], dtype=np.uint8) for (_, t), a in zip(params, args, strict=True)]
    return int(np.asarray(evaluate_bits(d, *ins)).reshape(-1)[0])


def _onehot(n: int, i: int) -> list[int]:
    return [int(j == i) for j in range(n)]


def _leaf(rnd, lo=-128, hi=127) -> list[list[int]]:
    return [[rnd.randint(lo, hi) & 0xFF for _ in range(32)] for _ in range(SG.RL)]


def _rsel(pos: int) -> list[int]:
    return [(1 << pos % 32) if pos // 32 == k else 0 for k in range(SG.RW // 32)]


def _honest_v3(rnd, ks, s, fold, dither):
    """A thread's whole call under `PouwSegCheck_v3`: its committed leaves, coin and draw, and receipt leaves holding its
    honest window words at one position."""
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
        bp.append([[(x + ((q & 1) if dither else 1 - 2 * (q & 1))) & 0xFF for x, q in zip(br[c], f[j][c])] for c in range(8)])
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


def _rotl(x, r: int):
    return ((x << np.uint64(r)) | (x >> np.uint64(32 - r))) & np.uint64(M32)


def _contrib(y: int, z: int, t: int, w):
    """A chain's share y' ^ rotl(z', 16) of the window word after absorbing the candidate words ``w`` at position t."""
    u = (np.uint64(y) ^ w ^ np.uint64(SG.KK[t])) & np.uint64(M32)
    y2 = (u * np.uint64(SG.KA[t]) + np.uint64(z)) & np.uint64(M32)
    return y2 ^ _rotl((_rotl(u, SG.ROT[t]) + y2) & np.uint64(M32), 16)


def _mitm(st: list[int], p: list[list[int]], target: int, rng) -> list[list[int]]:
    """LEDGER's opening: p's last set with chain 0's and chain 4's last words (entries 24 and 28) solved, over 2^18
    candidates each, so that duo's fold of p from st ends at ``target``."""
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
    assert len(i0), "no meeting point at 2^18 a side (about e^-16)"
    last = list(p[-1])
    last[24], last[28] = int(w0[i0[0]]), int(w4[i4[0]])
    return [*p[:-1], last]


def _attack():
    """A thread's call at KS = 4, S = 2 whose window-1 receipt word the cheater replaces by a random one, and v2's
    opening of it at segment 2 (window 1, position 0): p_0 free, p_1 its honest step from p_0, p_2 met in the middle."""
    rnd, rng = random.Random(20261017), np.random.default_rng(1723)
    h = _honest_v3(rnd, KS, S, "duo", False)
    target = rnd.getrandbits(32)
    forged = [list(h["rc"][0]), list(h["rc"][1])]
    forged[1][h["pos"]] = target
    p0 = [rnd.getrandbits(32) for _ in range(32)]
    p1 = [SG.seg_dot(p0[q], h["ap"][2][SG.row_slot(q)], h["bp"][2][SG.col_slot(q)]) for q in range(32)]
    p = _mitm(h["st"][1], [p0, p1, [rnd.getrandbits(32) for _ in range(32)]], target, rng)
    jsel = _onehot(SG.jsel_bits(S), 0)
    v2 = [h["a"][2], h["asel"], h["u"], h["b"][2], h["bsel"][:8], h["f"][2], p, h["st"][1], forged[1], h["rsel"], jsel, 0]
    h["rsel_bits"] = _onehot(SG.RW, h["pos"])
    v3 = [h["a"], h["asel"], h["u"], h["b"], h["bsel"], h["f"], h["st"], forged, h["rsel"]]
    return h, target, v2, v3


def test_mitm_opens_v2_and_v3_refuses_it():
    """v2's word view accepts the cheater's random window word with the forged prefix sets; v3, on the same committed
    leaves, coin and receipts, refuses it, and accepts the honest receipts."""
    h, target, v2, v3 = _attack()
    assert target != h["rc"][1][h["pos"]]
    assert SG.win_check(*v2[:9], h["rsel_bits"], *v2[10:]) == 1
    assert SG.seg_check3(*v3) == 0
    assert SG.seg_check3(*v3[:7], h["rc"], h["rsel"]) == 1


@pytest.mark.slow
def test_mitm_on_bits():
    """The same on the Definitions themselves: `PouwSegCheck_v2{2, duo}` evaluates the opening to 1, and
    `PouwSegCheck_v3{4, 2, duo}` evaluates the forged receipts to 0 and the honest ones to 1."""
    h, _, v2, v3 = _attack()
    assert _on_bits(bind(SG.PouwSegCheck, S=S, FOLD="duo"), v2) == 1
    d3 = bind(SG.PouwSegCheckV3, KS=KS, S=S, FOLD="duo")
    assert _on_bits(d3, v3) == 0
    assert _on_bits(d3, [*v3[:7], h["rc"], h["rsel"]]) == 1


def _call(rnd, ks, fold, out, nl=None):
    """A thread's (``nl`` None) or ``nl`` lanes' whole H1 call at the served ranges (V in [-127, 126], W' int8) and its honest
    receipt leaf; the index of one receipt word in the leaf."""
    m = SG.n_digits(ks)
    x, w = [_leaf(rnd, -127, 126) for _ in range(ks)], [_leaf(rnd, -60, 60) for _ in range(ks)]
    seed, tile = rnd.getrandbits(32), rnd.randrange(64)

    def lane(ctid):
        warp, ln = divmod(ctid, 32)
        wm, wn, g, q = warp // 4, warp % 4, ln >> 2, ln & 3
        return (_onehot(16, 8 * (wm % 2) + g),
                [[[rnd.getrandbits(1) for _ in range(32)] for _ in range(4)] for _ in range(ks)],
                _onehot(8, 4 * (wn % 2) + q) + [0] * 8,
                [[[rnd.getrandbits(1) for _ in range(32)] for _ in range(8)] for _ in range(ks)],
                [[rnd.randint(-64, 64) & 0xFF for _ in range(32)] for _ in range(8)],
                [[[rnd.getrandbits(1) for _ in range(32)] for _ in range(4)] for _ in range(m - 1)])

    if nl is None:
        ctid = rnd.randrange(SG.NCT)
        xsel, b, wsel, c, u, z = lane(ctid)
        y, zz = SG.init(seed, tile, ctid, 0, 1)
        state = y + zz
        for p in SG.lane_words(x, w, xsel, b, wsel, c, u, z):
            state = SG.absorb(fold, state, p)
        words = SG.chain_words(state) if out == "chains" else state
        n = len(words)
        slot = ctid % (SG.RW // n)
        rc = [rnd.getrandbits(32) for _ in range(SG.RW)]
        rc[n * slot:n * slot + n] = words
        return [x, xsel, b, w, wsel, c, u, z, y + zz, rc, _onehot(SG.RW // n, slot)], n * slot + n - 1
    warp, parts = rnd.randrange(16), 1 if fold == "li" else 2
    lanes = [lane(32 * warp + ln) for ln in range(nl)]
    st = [[v for half in SG.init(seed, tile, 32 * warp + ln, 0, 1)[:parts] for v in half] for ln in range(nl)]
    sets = [SG.lane_words(x, w, *L) for L in lanes]
    states, rc = [list(v) for v in st], [0] * (SG.RW * -(-parts * (ks + m) // SG.RW))
    for s in range(ks + m):
        for ln in range(nl):
            states[ln] = SG.absorb(fold, states[ln], sets[ln][s])
            for i, v in enumerate(states[ln]):
                rc[parts * s + i // SG.NCH] ^= v
    cols = [list(col) for col in zip(*lanes)]
    return [x, cols[0], cols[1], w, cols[2], cols[3], cols[4], cols[5], st, rc], parts * (ks + m) - 1


#: the sidecar's shapes: (Definition, word view, instance)
CASES = {
    "v3-duo": (lambda: bind(SG.PouwSegCheckV3, KS=2, S=1, FOLD="duo"),
               lambda r: _v3_case(r, 2, 1, "duo", False), lambda a: SG.seg_check3(*a)),
    "v3-dither-xin": (lambda: bind(SG.PouwSegCheckDitherV3, KS=2, S=2, FOLD="xin"),
                      lambda r: _v3_case(r, 2, 2, "xin", True), lambda a: SG.seg_check3(*a, fold="xin", dither=True)),
    "call-duo-chains": (lambda: bind(SG.PouwCallCheck, KS=2, FOLD="duo", OUT="chains"),
                        lambda r: _call(r, 2, "duo", "chains"), lambda a: SG.call_check(*a)),
    "call-fx1-state": (lambda: bind(SG.PouwCallCheck, KS=2, FOLD="fx1", OUT="state"),
                       lambda r: _call(r, 2, "fx1", "state"), lambda a: SG.call_check(*a, fold="fx1", out="state")),
    "warp-li": (lambda: bind(SG.PouwWarpCheck, KS=2, NL=2), lambda r: _call(r, 2, "li", "cp", nl=2),
                lambda a: SG.warp_check(*a)),
    "warp2-duo": (lambda: bind(SG.PouwWarpCheckV2, KS=2, NL=2, FOLD="duo"), lambda r: _call(r, 2, "duo", "cp", nl=2),
                  lambda a: SG.warp_check(*a, fold="duo")),
    "warp2-fx1": (lambda: bind(SG.PouwWarpCheckV2, KS=2, NL=2, FOLD="fx1"), lambda r: _call(r, 2, "fx1", "cp", nl=2),
                  lambda a: SG.warp_check(*a, fold="fx1")),
}


def _v3_case(rnd, ks, s, fold, dither):
    h = _honest_v3(rnd, ks, s, fold, dither)
    return [h["a"], h["asel"], h["u"], h["b"], h["bsel"], h["f"], h["st"], h["rc"], h["rsel"]], ("rc", -1, h["pos"])


def _tamper(args, at):
    """One bit of one committed receipt word changed: ``at`` an index into ``rc`` (the call checks) or (rc, window, word)."""
    out = [a for a in args]
    if isinstance(at, tuple):
        rc = [list(leaf) for leaf in out[7]]
        rc[at[1]][at[2]] ^= 1 << 17
        out[7] = rc
    else:
        rc = list(out[9])
        rc[at] ^= 1 << 17
        out[9] = rc
    return out


@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(CASES))
def test_checks_on_bits_are_their_word_views(case):
    """Each check at the sidecar's shape, on bits and on words: 1 on an honest instance, 0 with one receipt bit changed."""
    make, inst, word = CASES[case]
    d = make()
    args, at = inst(random.Random(case))
    bad = _tamper(args, at)
    assert (word(args), _on_bits(d, args)) == (1, 1)
    assert (word(bad), _on_bits(d, bad)) == (0, 0)


@pytest.mark.parametrize("fold", ["duo", "fx1"])
def test_warp_check_v2_reads_both_words_of_every_set(fold):
    """`PouwWarpCheck_v2`'s word view over 4 lanes at KS = 4: 1 on the honest checkpoints, 0 with one bit of a set's y word
    or of its z word changed."""
    args, last = _call(random.Random(fold), 4, fold, "cp", nl=4)
    assert SG.warp_check(*args, fold=fold) == 1
    for at in (2, 3, last):
        assert SG.warp_check(*_tamper(args, at), fold=fold) == 0, at


def test_a_warp_receipt_past_a_leaf_takes_the_next():
    """At K = 8192 (KS = 256, m = 3) li's 259 checkpoint words fit one receipt leaf and duo's and fx1's 518 take two: the
    port is 1024 words, and the word view reads the second leaf's checkpoints and nothing after them."""
    assert [dict(bind(SG.PouwWarpCheckV2, KS=256, NL=1, FOLD=f).signature()[0])["rc"].n for f in ("li", "duo", "fx1")] \
        == [SG.RW, 2 * SG.RW, 2 * SG.RW]
    args, last = _call(random.Random("k8192"), 256, "duo", "cp", nl=1)
    assert (last, len(args[9]), SG.warp_check(*args, fold="duo")) == (517, 2 * SG.RW, 1)
    assert [SG.warp_check(*_tamper(args, at), fold="duo") for at in (SG.RW, last, last + 1)] == [0, 0, 1]


def test_the_served_receipt_is_not_injective_at_a_state():
    """#2438's `DuoReceiptNotInjective` on this module's duo keys (word 0's): from one state two words give one served
    receipt y ^ rotl(z, 16) and two full states, which only a checkpoint of the whole state (`PouwWarpCheck_v2`) tells
    apart."""
    y, z = 0x247A23F6, 0xC48C9971

    def step(d):
        u = y ^ d ^ SG.KK[0]
        y1 = (u * SG.KA[0] + z) & M32
        return y1, (SG.rotl(u, SG.ROT[0]) + y1) & M32

    (y1, z1), (y2, z2) = step(0x1632C), step(0x2237A)
    assert y1 ^ SG.rotl(z1, 16) == y2 ^ SG.rotl(z2, 16) == 0x26D90889
    assert (y1, z1) != (y2, z2)


def test_segment_dots_are_unserved_int8_tensor_core_steps():
    """census/v1 reads the segment dots on bits as the wrapping s8 IMMA coordinate's k32: their gates are `tc:int8`, and they
    perform their 32 products each and serve none (Lean's `Accounting.Gates.roleOf` puts an H1 tile's lanes at `Site.other`)."""
    A = Array(32, SG.I8)
    lanes = CompositeDefinition("SegTestLanes", 1, (), lambda S: ((("a", A), ("b", A)), SG.U32),
                                lambda B, S, a, b: B.call(SG.PouwSegDot, B.call(SG.PouwSegDot0, a, b), a, b),
                                conformance="protocol-defined", register=False)
    c = census(Program(bind(lanes)), boolean=True)
    dots = {e["family"]: e for e in c["classes"].values() if e["family"] in PRODUCTS}
    assert {f: (e["instances"], PRODUCTS[f], e["served_macs"]) for f, e in dots.items()} == {
        "PouwSegDot_v2": (1, 32, 0), "PouwSegDot0_v2": (1, 32, 0)}
    assert {k: c["totals"][k] for k in COUNTERS} == dict.fromkeys(COUNTERS, 0) and c["totals"]["served_macs_by_precision"] == {}
    want = [BS.compile(bind(d)).counts for d in (SG.PouwSegDot, SG.PouwSegDot0)]
    assert c["totals"]["work"] == {"tc:int8": {"gates": sum(sum(w.values()) for w in want), "and": sum(w["and"] for w in want)}}


def test_call_words_take_no_witness():
    """Every port of the new checks is committed, the coin's or the draw's: none carries a prefix word."""
    committed, coin, draw = {"a", "b", "x", "w", "rc"}, {"u", "f", "c", "z", "st"}, {"asel", "bsel", "xsel", "wsel", "rsel"}
    for d in (bind(SG.PouwSegCheckV3, KS=2, S=1, FOLD="duo"), bind(SG.PouwCallCheck, KS=2, FOLD="fx1", OUT="state"),
              bind(SG.PouwWarpCheck, KS=2, NL=1), bind(SG.PouwWarpCheckV2, KS=2, NL=1, FOLD="duo")):
        names = {n for n, _ in d.signature()[0]}
        assert names <= committed | coin | draw, (d.id, names - committed - coin - draw)
