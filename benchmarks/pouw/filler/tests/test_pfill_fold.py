"""pfill's receipt folds against the cheats on them, on the twin
(pfill_ref.py) and on pfill.cuh's host `dfill_ref` built with g++, the reference dgate holds the GPU to:
ram (NC 4), duo (NC 2), and the r9 folds LOP3+IMAD (LI, -DPFILL_LI) and XOR-in (13, 19) (XIN13, -DPFILL_XIN13).

Ram's step (rotl(y, 7) + d0) a0 + d1 = rotl(y, 7) a0 + Q reads a pair only through Q = a0 d0 + d1, so a receipt from Q
alone (one product a pair) is the tile's ram receipt. A fold binds both words of a pair when no change of them maps every
state alike: exhaustively on small analogs of each fold (n-bit words, the rotations mod n), ram's such changes are the
2^n - 1 of the line (d0 + δ, d1 - a0 δ) at a pair; XIN13's and LI's the one joint flip of both words' top bit
(2^(n-1) (a0 + 1) = 0 mod 2^n for odd a0) at a pair, and over S pairs a group of 2^(3S - 2) that holds at every width
(`_xin_group`, checked at S = 2 and 3, LI's with rotation 0), each element a fixed bijection of every word, whose
complement element flips three words' low bits; at 32 bits the same flips leave the tile's receipt unchanged
(`test_complement_flips_at_32_bits`); duo's none, at a pair and at an MMA."""

from __future__ import annotations

import ctypes
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pfill_ref", HERE / "pfill_ref.py")
R = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = R
_spec.loader.exec_module(R)

C0, C1 = 0x0123456789ABCDEF, 0xFEDCBA9876543210
INT_KINDS = (R.S8, R.BITS, R.BU)


FOLDS = (R.RAM, R.DUO, R.LI, R.XIN13)
FLAG = {R.LI: "-DPFILL_LI", R.XIN13: "-DPFILL_XIN13"}


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    """pfill_host's reference: R.LI and R.XIN13 from its -DPFILL_LI and -DPFILL_XIN13 builds (NC 4 there)."""
    gxx = shutil.which("g++")
    if not gxx:
        pytest.skip("no g++ for the host build of pfill.cuh")
    d, libs = tmp_path_factory.mktemp("pfill-host"), {}
    for fold in (None, R.LI, R.XIN13):
        so = d / f"pfill_host{len(libs)}.so"
        subprocess.run([gxx, "-O2", "-std=c++17", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror", "-Wno-unknown-pragmas",
                        *([FLAG[fold]] if fold else []), "-o", str(so), str(HERE / "pfill_host.cpp")], check=True)
        cdll = ctypes.CDLL(str(so))
        cdll.pfill_ref.restype = ctypes.c_uint64
        cdll.pfill_ref.argtypes = [ctypes.c_uint64] * 3 + [ctypes.c_int] * 4
        libs[fold] = cdll

    def ref(c0, c1, tile, nc, steps, kind=R.S8, w=R.W):
        lib, nc = (libs[nc], R.RAM) if nc in FLAG else (libs[None], nc)
        return lib.pfill_ref(c0, c1, tile, nc, steps, int(kind), w)

    return ref


@pytest.mark.parametrize("kind", INT_KINDS)
@pytest.mark.parametrize(("tile", "steps"), [(0, 4), (1000003, 13), ((1 << 62) | 99, 8)])
def test_q_only_cheat_passes_ram_and_fails_the_others(host, kind, tile, steps):
    """The break: Q per pair from one product a pair gives ram's receipt bit for bit; under duo, LI and XIN13 it
    doesn't."""
    assert R.q_receipt(C0, C1, tile, R.RAM, steps, kind) == host(C0, C1, tile, R.RAM, steps, kind)
    for nc in (R.DUO, R.LI, R.XIN13):
        assert R.q_receipt(C0, C1, tile, nc, steps, kind) != host(C0, C1, tile, nc, steps, kind)


@pytest.mark.parametrize("kind", INT_KINDS)
def test_q_words_are_half_the_information(kind):
    """The cheat's words agree with the tile's in Q = a0 d0 + d1 (a1 on rows 8..15) and nowhere else in general."""
    cw, qw = R.tile_words(C0, C1, 5, 4, kind), R.q_words(C0, C1, 5, 4, kind)
    a0, a1 = R.keys(R.block(C0, C1, 5, R.CTR_KEY, R.dom_of(kind)))[:2]
    a = np.where(np.arange(cw.shape[1]) % 16 < 8, a0, a1).astype(np.uint64)[None, :, None]
    q = (a * cw[..., 0::2] + cw[..., 1::2]) & R.M32
    assert (q == qw[..., 1::2]).all() and not qw[..., 0::2].any()
    assert (cw[..., 0::2] != 0).any()


@pytest.mark.parametrize("kind", [R.S8, R.BITS])
@pytest.mark.parametrize("nc", FOLDS)
def test_packed_cheat_recovers_every_word_on_pm1_and_bits(host, kind, nc):
    """The packed route, (4 A)(33 B0 + B1) and a table, gets both words of a pair from one product on the ±1 and {0,1}
    tiles, under every fold: no fold closes it (the fold analysis prices it by COINC)."""
    assert R.packed_receipt(C0, C1, 3, nc, 8, kind) == host(C0, C1, 3, nc, 8, kind)


def flipped(cw: np.ndarray, which: str, nc: int, keys: tuple, v: int = 0, g: int = 0, t: int = 0) -> np.ndarray:
    """The tile's words with one MMA's four (warp v, lane (g, t), fragment 0, B fragment 0) changed by a kernel generator
    at 32 bits: P = (d0 ^ 2^31, d1 + 2^31); X = (d0 ^ 2^31, d2 ^ rotl(2^31, r)), r the rotation before d2's XOR (19,
    LI's 0); N = (~d0, a0 - 1 - d1, ~d2); and LI's carry element E = (d1 + 2^30, d3 - s a1 2^30), s = 1 - 2 bit 30 of
    d2 ^ k1 (`_li_group`). Each is a fixed bijection of words the cheat has already computed."""
    cw, top = cw.copy(), np.uint32(1 << 31)
    a0, a1, _, k1 = keys
    i = (v, [g, g, g + 8, g + 8], [2 * t, 2 * t + 1, 2 * t, 2 * t + 1])
    d0, d1, d2, d3 = (int(x) for x in cw[i])
    r, s = 0 if nc == R.LI else 19, 1 - 2 * (((d2 ^ k1) >> 30) & 1)
    new = {"P": (d0 ^ (1 << 31), (d1 + (1 << 31)) & R.M32, d2, d3),
           "X": (d0 ^ (1 << 31), d1, d2 ^ int(R.rotl(top, r)), d3),
           "N": (d0 ^ R.M32, (a0 - 1 - d1) & R.M32, d2 ^ R.M32, d3),
           "E": (d0, (d1 + (1 << 30)) & R.M32, d2, (d3 - s * a1 * (1 << 30)) & R.M32)}[which]
    cw[i] = np.array(new, np.uint32)
    return cw


LANES = [(0, 0, 0), (1, 3, 2), (2, 7, 1), (3, 5, 3), (4, 1, 1), (5, 6, 0), (6, 2, 3), (7, 4, 2)]  # (warp, g, t)
# of the 8 chains, how many receipts each change leaves unchanged: all for an all-state kernel element, none for duo;
# ram's X holds where bit 18 of rotl(y, 19) and of d2 differ, a state-dependent share
FLIPS = {R.RAM: {"P": 8, "N": 0}, R.DUO: {"P": 0, "X": 0, "N": 0, "E": 0}, R.LI: {"P": 8, "X": 8, "N": 8, "E": 8},
         R.XIN13: {"P": 8, "X": 8, "N": 8, "E": 0}}


@pytest.mark.parametrize("kind", INT_KINDS)
@pytest.mark.parametrize("nc", FOLDS)
def test_complement_flips_at_32_bits(host, kind, nc):
    """The kernel generators applied to a tile's true words at one MMA of one chain, for 8 chains: the LI and XIN13
    receipts are unchanged under P, X and N (each in their all-state kernel), LI's under its carry element E too, ram's
    under P (on its Q line) and on some states under X, duo's under none. A cheat gains nothing: each is a fixed
    bijection of words it must already hold."""
    tile, steps = 11, 4
    cw = R.tile_words(C0, C1, tile, steps, kind)
    keys = R.keys(R.block(C0, C1, tile, R.CTR_KEY, R.dom_of(kind)), nc in R.R9)
    want = host(C0, C1, tile, nc, steps, kind)
    assert R.fold(C0, C1, tile, nc, steps, cw, kind) == want
    same = {w: sum(R.fold(C0, C1, tile, nc, steps, flipped(cw, w, nc, keys, *ln), kind) == want for ln in LANES)
            for w in "PXNE"}
    assert {w: n for w, n in same.items() if w in FLIPS[nc]} == FLIPS[nc]
    assert nc != R.RAM or 0 < same["X"] < 8


def test_bu_does_not_pack():
    """No c >= 1 keeps c U0 + U1 in s8 over bU's alphabet {±1, ..., ±64} (33 is 2176), so bU's B can't pack."""
    u = np.concatenate([np.arange(-64, 0), np.arange(1, 65)])
    for c in range(1, 128):
        assert abs(c * u[:, None] + u[None, :]).max() > 127
    assert abs(33 * u[:, None] + u[None, :]).max() == 2176
    with pytest.raises(ValueError):
        R.packed_words(C0, C1, 0, 4, R.BU)


# ---- small analogs: n-bit words, every state ----------------------------------------------------------------------------


def _rotl(x, r, n):
    r %= n
    m = (1 << n) - 1
    return ((x << r) | (x >> (n - r))) & m if r else x


def _ram(y, d0, d1, a, r, n):
    return ((_rotl(y, r, n) + d0) * a + d1) & ((1 << n) - 1)


def _xin(y, d0, d1, a, r, n):
    return ((_rotl(y, r, n) ^ d0) * a + d1) & ((1 << n) - 1)


def _li(y, d0, d1, a, r, n, k=22):
    """LOP3+IMAD's pair step with a free key k (the rotation r unused: LI has none)."""
    return ((y ^ d0 ^ (k % (1 << n))) * a + d1) & ((1 << n) - 1)


def _duo(y, z, ps, ks, n):
    m = (1 << n) - 1
    for p, (k, a, r) in zip(ps, ks):
        u = y ^ p ^ k
        y = (u * a + z) & m
        z = (_rotl(u, r, n) + y) & m
    return y, z


def _pair_kernel(step, n, a, d0, d1, r=7):
    """The changes (e0, e1) != (d0, d1) with step(y, e0, e1) = step(y, d0, d1) for every n-bit y."""
    y = np.arange(1 << n, dtype=np.int64)
    want = step(y, d0, d1, a, r, n)
    e1 = np.arange(1 << n, dtype=np.int64)[:, None]
    out = []
    for e0 in range(1 << n):
        same = (step(y[None, :], e0, e1, a, r, n) == want[None, :]).all(axis=1)
        out += [(e0, int(v)) for v in np.flatnonzero(same) if (e0, int(v)) != (d0, d1)]
    return out


@pytest.mark.parametrize(("a", "d0", "d1"), [(5, 17, 200), (0x8B, 0, 0), (0xF7, 255, 1)])
def test_ram_pair_kernel_is_the_q_line(a, d0, d1):
    n = 8
    ker = _pair_kernel(_ram, n, a, d0, d1)
    assert len(ker) == (1 << n) - 1
    assert all((a * e0 + e1 - a * d0 - d1) % (1 << n) == 0 for e0, e1 in ker)


@pytest.mark.parametrize(("step", "r"), [(_xin, 13), (_li, 0)])
@pytest.mark.parametrize(("a", "d0", "d1"), [(5, 17, 200), (0x8B, 0, 0), (0xF7, 255, 1)])
def test_xin13_and_li_pair_kernels_are_the_joint_top_bit_flip(step, r, a, d0, d1):
    n = 8
    top = 1 << (n - 1)
    assert _pair_kernel(step, n, a, d0, d1, r) == [(d0 ^ top, (d1 + top) % (1 << n))]


def _mma_kernel(step, n, keys, ds, rots=(7, 19)):
    """The changes of one MMA's four words, two steps (rotations rots mod n), that map every n-bit y alike."""
    (a0, a1), m = keys, 1 << n
    y = np.arange(m, dtype=np.int64)

    def two(y, e):
        return step(step(y, e[0], e[1], a0, rots[0], n), e[2], e[3], a1, rots[1], n)

    want = two(y, ds)
    e2, e3 = (v.reshape(-1, 1) for v in np.meshgrid(np.arange(m), np.arange(m), indexing="ij"))
    out = []
    for e0 in range(m):
        for e1 in range(m):
            same = (two(y[None, :], (e0, e1, e2, e3)) == want[None, :]).all(axis=1)
            out += [(e0, e1, int(e2[i, 0]), int(e3[i, 0])) for i in np.flatnonzero(same)]
    return [e for e in out if e != tuple(ds)]


def _xin_group(n, keys, rots, ds):
    """The changes of S pairs' words that XOR-in maps alike at every width n, closed under composition: each pair's
    joint top-bit flip P_p = (d_p0 ^ t, d_p1 + t); the cross flip X_p = (d_p0 ^ t, d_(p+1)0 ^ rotl(t, r_(p+1))), since
    x ^ t = x + t passes the odd multiply and the add and the rotation moves it to the next XOR-in word; and
    N_p = (~d_p0, a_p - 1 - d_p1, ~d_(p+1)0), since (~w) a = -w a - a and ~ commutes with the rotation. Each maps every
    word by a fixed bijection; the 3S - 2 commuting involutions make a group of 2^(3S - 2)."""
    m, top, S = (1 << n) - 1, 1 << (n - 1), len(keys)

    def edit(**at):
        return lambda d: tuple(at[f"w{i}"](v) if f"w{i}" in at else v for i, v in enumerate(d))

    gens = [edit(**{f"w{2 * p}": lambda v: v ^ top, f"w{2 * p + 1}": lambda v: (v + top) & m}) for p in range(S)]
    for p in range(S - 1):
        r = _rotl(top, rots[p + 1], n)
        gens.append(edit(**{f"w{2 * p}": lambda v: v ^ top, f"w{2 * p + 2}": lambda v, r=r: v ^ r}))
        gens.append(edit(**{f"w{2 * p}": lambda v: v ^ m, f"w{2 * p + 1}": lambda v, a=keys[p]: (a - 1 - v) & m,
                            f"w{2 * p + 2}": lambda v: v ^ m}))
    seen, todo = {tuple(ds)}, [tuple(ds)]
    while todo:
        d = todo.pop()
        for g in gens:
            if (e := g(d)) not in seen:
                seen.add(e)
                todo.append(e)
    return seen - {tuple(ds)}


def _li_group(n, keys, ds, k=22):
    """LI's all-state kernel over S pairs as far as it is generated by `_xin_group`'s moves at rotation 0 and LI's carry
    elements: with no rotation between a pair's add and the next XOR, adding 2^j to the add word d_p1 moves the next
    XOR's output by exactly s 2^j on every state when bits j..n-2 of c = d_(p+1)0 ^ k are all equal (s = 1 - 2 bit j of
    c: always at j = n - 2, half the time at n - 3, ...), so E_pj = (d_p1 + 2^j, d_(p+1)1 - s a_(p+1) 2^j)."""
    m, S = (1 << n) - 1, len(keys)

    def carries(d):
        for p in range(S - 1):
            c = d[2 * p + 2] ^ (k % (1 << n))
            for j in range(n - 1):
                if len({(c >> b) & 1 for b in range(j, n - 1)}) == 1:
                    e = list(d)
                    e[2 * p + 1] = (e[2 * p + 1] + (1 << j)) & m
                    e[2 * p + 3] = (e[2 * p + 3] - (1 - 2 * ((c >> j) & 1)) * keys[p + 1] * (1 << j)) & m
                    yield tuple(e)

    seen, todo = {tuple(ds)}, [tuple(ds)]
    while todo:
        d = todo.pop()
        for e in (*carries(d), *_xin_group(n, keys, (0,) * S, d)):
            if e not in seen:
                seen.add(e)
                todo.append(e)
    return seen - {tuple(ds)}


def _three_pair_kernel(step, n, keys, rots, ds):
    """The changes of three pairs' words (three steps) that map every n-bit y alike, met in the middle: the first two
    steps' changes as permutations of the state after the first true step, against the third's inverses."""
    m = 1 << n
    y = np.arange(m, dtype=np.int64)
    es = np.array([(e0, e1) for e0 in range(m) for e1 in range(m)], dtype=np.int64)

    def run(p, yy, e0, e1):
        return step(yy, e0, e1, keys[p], rots[p], n)

    first = run(0, np.argsort(run(0, y, ds[0], ds[1]))[None, :], es[:, :1], es[:, 1:])
    want = run(2, run(1, y, ds[2], ds[3]), ds[4], ds[5])
    undo = np.take_along_axis(np.argsort(run(2, y[None, :], es[:, :1], es[:, 1:]), axis=1),
                              np.broadcast_to(want, (len(es), m)), axis=1)
    last = {}
    for i, row in enumerate(undo.astype(np.uint8)):
        last.setdefault(row.tobytes(), []).append(i)
    out = set()
    for i0 in range(len(es)):
        for i1, row in enumerate(run(1, first[i0][None, :], es[:, :1], es[:, 1:]).astype(np.uint8)):
            for i2 in last.get(row.tobytes(), ()):
                out.add(tuple(int(v) for v in (*es[i0], *es[i1], *es[i2])))
    return out - {tuple(ds)}


def _f2_rank(rows):
    rank, rows = 0, [int("".join(map(str, r)), 2) for r in rows]
    while rows:
        if (top := max(rows)) == 0:
            break
        rows = [r ^ top if r.bit_length() == top.bit_length() else r for r in rows if r != top]
        rank += 1
    return rank


@pytest.mark.parametrize(("keys", "ds"), [((11, 21), (3, 30, 17, 6)), ((7, 25), (0, 0, 31, 9))])
def test_ram_xin13_and_li_mma_kernels(keys, ds):
    """At an MMA's four words (n = 5), every state: ram's are the product of the two Q lines, (2^n)^2 - 1, data-free
    representatives such as (0, Q) among them; XIN13's are exactly `_xin_group`'s 15, so a pair of word changes can
    cancel across its two pairs (X), but only as fixed bijections of words the cheat must already have; LI's are
    `_li_group`'s 31, the same moves and the carry element at its hand-off (n = 6: 63 or 31, as c's bits allow)."""
    n = 5
    assert len(_mma_kernel(_ram, n, keys, ds)) == (1 << n) ** 2 - 1
    ker = set(_mma_kernel(_xin, n, keys, ds, (13, 19)))
    assert ker == _xin_group(n, keys, (13, 19), ds) and len(ker) == 15
    for n, size in ((5, 31), (6, 63 if keys == (11, 21) else 31)):
        ker = set(_mma_kernel(_li, n, keys, ds, (0, 0)))
        assert ker == _li_group(n, keys, ds) and len(ker) == size


@pytest.mark.parametrize(("keys", "ds"), [((11, 21, 13), (3, 30, 17, 6, 9, 20)), ((7, 25, 19), (0, 0, 31, 9, 4, 1))])
def test_xin13_kernel_grows_three_generators_a_pair(keys, ds):
    """Over three pairs (n = 5), every state, XIN13's kernel is `_xin_group`'s 2^7 - 1: N carries the complement across
    pairs. Only N touches low bits, three at a time, so mod 2 the receipt fixes S + 1 of the 2S words' shadows (ram S,
    duo 2S): the question for Theorem K, which reads V mod 2, is whether its count survives N's data-chosen flips. LI
    has no such small analog past an MMA: with no rotation its steps are T-functions (bit i of the output reads bits <= i
    alone), which on 5 bits form a group of 2^31 permutations, so three pairs' 2^30 word tuples meet by counting
    (5631 and 2047 here, against `_li_group`'s 4095 and 1023)."""
    n, rots = 5, (13, 19, 13)
    ker = _three_pair_kernel(_xin, n, keys, rots, ds)
    assert ker == _xin_group(n, keys, rots, ds) and len(ker) == 2 ** 7 - 1
    assert _f2_rank({tuple((e ^ d) & 1 for e, d in zip(k, ds)) for k in ker}) == len(keys) - 1


@pytest.mark.parametrize(("n", "words"), [(6, 2), (4, 4)])
def test_duo_has_no_kernel(n, words):
    """Duo, every (y, z): no change of a pair's words (n = 6) or of an MMA's four (n = 4) maps every state alike."""
    m = 1 << n
    ks = [(0x9 % m, 5, 5), (0x6 % m, 11, 11), (0x9 % m, 5, 17), (0x6 % m, 11, 29)][:words]
    ds = [3, m - 1, 7, 0][:words]
    y, z = (v.ravel() for v in np.meshgrid(np.arange(m), np.arange(m), indexing="ij"))
    want = _duo(y, z, ds, ks, n)
    grid = np.stack([v.ravel() for v in np.meshgrid(*[np.arange(m)] * (words - 1), indexing="ij")])
    hits = 0
    for p0 in range(m):
        ps = [p0] + [g[:, None] for g in grid]
        gy, gz = _duo(y[None, :], z[None, :], ps, ks, n)
        same = ((gy == want[0][None, :]) & (gz == want[1][None, :])).all(axis=1)
        hits += int(same.sum()) - int(p0 == ds[0] and same[np.ravel_multi_index(ds[1:], [m] * (words - 1))])
    assert hits == 0


@pytest.mark.parametrize(("n", "rots"), [pytest.param(4, (5, 11, 17, 29, 7, 19, 13, 9), marks=pytest.mark.slow),
                                         (5, (11, 17, 29, 13, 7, 19))])
def test_duo_has_no_kernel_over_six_and_eight_words(n, rots):
    """Duo over 8 words (n = 4) and 6 (n = 5), every (y, z), met in the middle: each step is a bijection of the state,
    so the first half's changes are permutations of the true middle state, matched against the second half's inverses.
    The rotations are nonzero mod n, as pfill's 5, 11, 17, 29 are mod 32 (at r = 0 the top flips of two words cancel)."""
    m, words = 1 << n, len(rots)
    ks = [(k % m, a, r) for (k, a), r in zip([(9, 5), (6, 11), (9, 5), (6, 11), (3, 13), (10, 7), (5, 3), (12, 9)], rots)]
    ds, h = [3, m - 1, 7, 0, 5, 9, 1, 12][:words], words // 2
    y, z = (v.ravel() for v in np.meshgrid(np.arange(m), np.arange(m), indexing="ij"))
    states = y * m + z
    mid = _duo(y, z, ds[:h], ks[:h], n)
    mid = mid[0] * m + mid[1]
    end = np.empty(m * m, dtype=np.int64)
    end[mid] = np.ravel_multi_index(_duo(*_duo(y, z, ds[:h], ks[:h], n), ds[h:], ks[h:], n), (m, m))
    back = np.empty(m * m, dtype=np.int64)
    back[mid] = states
    first = {}
    for ps in np.ndindex(*[m] * h):
        gy, gz = _duo(y, z, list(ps), ks[:h], n)
        first.setdefault((gy * m + gz)[back].astype(np.uint16).tobytes(), []).append(ps)
    hits = []
    for ps in np.ndindex(*[m] * (words - h)):
        gy, gz = _duo(y, z, list(ps), ks[h:], n)
        inv = np.full(m * m, -1, dtype=np.int64)
        inv[gy * m + gz] = states
        need = inv[end]
        if (need >= 0).all():
            hits += [(*f, *ps) for f in first.get(need.astype(np.uint16).tobytes(), ()) if (*f, *ps) != tuple(ds)]
    assert hits == []
