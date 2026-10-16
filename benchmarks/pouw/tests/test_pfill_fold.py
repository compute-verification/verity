"""pfill's receipt folds against the cheats of note:20261016T0917Z-draft-fullfloor-s-r8 §1.2, on the twin
(decode_capacity/pfill_ref.py) and on pfill.cuh's host `dfill_ref` built with g++, the reference dgate holds the GPU to.

Ram's step (rotl(y, 7) + d0) a0 + d1 = rotl(y, 7) a0 + Q reads a pair only through Q = a0 d0 + d1, so a receipt from Q
alone (one product a pair) is the tile's ram receipt. A fold binds both words of a pair when no change of them maps every
state alike: exhaustively on small analogs of each fold (n-bit words, the rotations mod n), ram's such changes are the
2^n - 1 of the line (d0 + δ, d1 - a0 δ) at a pair; XOR-in's (-DPFILL_XIN) the one joint flip of both words' top bit
(2^(n-1) (a0 + 1) = 0 mod 2^n for odd a0) at a pair, and at an MMA's four words a group of 16 that holds at every width,
32 included (`_xin_group`), each element a fixed bijection of every word; duo's none, at a pair and at an MMA."""

from __future__ import annotations

import ctypes
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "decode_capacity"
_spec = importlib.util.spec_from_file_location("pfill_ref", HERE / "pfill_ref.py")
R = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = R
_spec.loader.exec_module(R)

C0, C1 = 0x0123456789ABCDEF, 0xFEDCBA9876543210
INT_KINDS = (R.S8, R.BITS, R.BU)


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    """pfill_host's reference, R.XIN from its -DPFILL_XIN build (NC 4 there)."""
    gxx = shutil.which("g++")
    if not gxx:
        pytest.skip("no g++ for the host build of pfill.cuh")
    d, libs = tmp_path_factory.mktemp("pfill-host"), []
    for flags in ([], ["-DPFILL_XIN"]):
        so = d / f"pfill_host{len(libs)}.so"
        subprocess.run([gxx, "-O2", "-std=c++17", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror", "-Wno-unknown-pragmas",
                        *flags, "-o", str(so), str(HERE / "pfill_host.cpp")], check=True)
        cdll = ctypes.CDLL(str(so))
        cdll.pfill_ref.restype = ctypes.c_uint64
        cdll.pfill_ref.argtypes = [ctypes.c_uint64] * 3 + [ctypes.c_int] * 4
        libs.append(cdll)

    def ref(c0, c1, tile, nc, steps, kind=R.S8, w=R.W):
        lib, nc = (libs[1], R.RAM) if nc == R.XIN else (libs[0], nc)
        return lib.pfill_ref(c0, c1, tile, nc, steps, int(kind), w)

    return ref


@pytest.mark.parametrize("kind", INT_KINDS)
@pytest.mark.parametrize(("tile", "steps"), [(0, 4), (1000003, 13), ((1 << 62) | 99, 8)])
def test_q_only_cheat_passes_ram_and_fails_duo_and_xin(host, kind, tile, steps):
    """The break: Q per pair from one product a pair gives ram's receipt bit for bit; under duo and XOR-in it doesn't."""
    assert R.q_receipt(C0, C1, tile, R.RAM, steps, kind) == host(C0, C1, tile, R.RAM, steps, kind)
    for nc in (R.DUO, R.XIN):
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
@pytest.mark.parametrize("nc", [R.RAM, R.DUO, R.XIN])
def test_packed_cheat_recovers_every_word_on_pm1_and_bits(host, kind, nc):
    """The packed route, (4 A)(33 B0 + B1) and a table, gets both words of a pair from one product on the ±1 and {0,1}
    tiles, under every fold: no fold closes it (P2-FF prices it by COINC)."""
    assert R.packed_receipt(C0, C1, 3, nc, 8, kind) == host(C0, C1, 3, nc, 8, kind)


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


def _duo(y, z, ps, ks, n):
    m = (1 << n) - 1
    for p, (k, a, r) in zip(ps, ks):
        u = y ^ p ^ k
        y = (u * a + z) & m
        z = (_rotl(u, r, n) + y) & m
    return y, z


def _pair_kernel(step, n, a, d0, d1):
    """The changes (e0, e1) != (d0, d1) with step(y, e0, e1) = step(y, d0, d1) for every n-bit y."""
    y = np.arange(1 << n, dtype=np.int64)
    want = step(y, d0, d1, a, 7, n)
    e1 = np.arange(1 << n, dtype=np.int64)[:, None]
    out = []
    for e0 in range(1 << n):
        same = (step(y[None, :], e0, e1, a, 7, n) == want[None, :]).all(axis=1)
        out += [(e0, int(v)) for v in np.flatnonzero(same) if (e0, int(v)) != (d0, d1)]
    return out


@pytest.mark.parametrize(("a", "d0", "d1"), [(5, 17, 200), (0x8B, 0, 0), (0xF7, 255, 1)])
def test_ram_pair_kernel_is_the_q_line(a, d0, d1):
    n = 8
    ker = _pair_kernel(_ram, n, a, d0, d1)
    assert len(ker) == (1 << n) - 1
    assert all((a * e0 + e1 - a * d0 - d1) % (1 << n) == 0 for e0, e1 in ker)


@pytest.mark.parametrize(("a", "d0", "d1"), [(5, 17, 200), (0x8B, 0, 0), (0xF7, 255, 1)])
def test_xin_pair_kernel_is_the_joint_top_bit_flip(a, d0, d1):
    n = 8
    top = 1 << (n - 1)
    assert _pair_kernel(_xin, n, a, d0, d1) == [(d0 ^ top, (d1 + top) % (1 << n))]


def _mma_kernel(step, n, keys, ds):
    """The changes of one MMA's four words, two steps (rotations 7 and 19 mod n), that map every n-bit y alike."""
    (a0, a1), m = keys, 1 << n
    y = np.arange(m, dtype=np.int64)

    def two(y, e):
        return step(step(y, e[0], e[1], a0, 7, n), e[2], e[3], a1, 19, n)

    want = two(y, ds)
    e2, e3 = (v.reshape(-1, 1) for v in np.meshgrid(np.arange(m), np.arange(m), indexing="ij"))
    out = []
    for e0 in range(m):
        for e1 in range(m):
            same = (two(y[None, :], (e0, e1, e2, e3)) == want[None, :]).all(axis=1)
            out += [(e0, e1, int(e2[i, 0]), int(e3[i, 0])) for i in np.flatnonzero(same)]
    return [e for e in out if e != tuple(ds)]


def _xin_group(n, a0, ds):
    """The changes of an MMA's words that XOR-in maps alike at every width n, closed under composition: each pair's
    joint top-bit flip P1, P2; the cross flip X = (d0 ^ m, d2 ^ rotl(m, 19)), since x ^ m = x + m passes the odd
    multiply and the add and the rotation moves it to the next XOR-in word; and N = (~d0, a0 - 1 - d1, ~d2), since
    (~w) a0 = -w a0 - a0 and ~ commutes with the rotation. Each maps every word by a fixed bijection."""
    m, top = (1 << n) - 1, 1 << (n - 1)
    gens = (lambda d: (d[0] ^ top, (d[1] + top) & m, d[2], d[3]),
            lambda d: (d[0], d[1], d[2] ^ top, (d[3] + top) & m),
            lambda d: (d[0] ^ top, d[1], d[2] ^ _rotl(top, 19, n), d[3]),
            lambda d: (d[0] ^ m, (a0 - 1 - d[1]) & m, d[2] ^ m, d[3]))
    seen, todo = {tuple(ds)}, [tuple(ds)]
    while todo:
        d = todo.pop()
        for g in gens:
            if (e := g(d)) not in seen:
                seen.add(e)
                todo.append(e)
    return seen - {tuple(ds)}


@pytest.mark.parametrize(("keys", "ds"), [((11, 21), (3, 30, 17, 6)), ((7, 25), (0, 0, 31, 9))])
def test_ram_and_xin_mma_kernels(keys, ds):
    """At an MMA's four words (n = 5), every state: ram's are the product of the two Q lines, (2^n)^2 - 1, data-free
    representatives such as (0, Q) among them; XOR-in's are exactly `_xin_group`'s 15, so a pair of word changes can
    cancel across XOR-in's two pairs (X), but only as fixed bijections of words the cheat must already have."""
    n = 5
    assert len(_mma_kernel(_ram, n, keys, ds)) == (1 << n) ** 2 - 1
    ker = set(_mma_kernel(_xin, n, keys, ds))
    assert ker == _xin_group(n, keys[0], ds) and len(ker) == 15


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
