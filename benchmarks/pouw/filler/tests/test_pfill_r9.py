"""The two r9 receipt folds in pfill: NC 4 built with -DPFILL_LI (LOP3+IMAD) or -DPFILL_XIN13 (XOR-in with rotations 13,
19). pfill_ref.py's receipts are held to pfill.cuh's host `dfill_ref` built with g++ and each flag, the reference dgate
holds the GPU to, in each integer tile kind; pfill_ref's step to the reference twins; and the key path to the
reference's `key_ok_r9`, the reference functions below copied verbatim from its receipt_folds_r9.py."""

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
KINDS = {R.S8: [], R.BITS: ["-DPFILL_BITS"], **({R.BU: ["-DPFILL_BU", "-mpopcnt"]} if hasattr(R, "BU") else {})}
FOLDS = {R.LI: "-DPFILL_LI", R.XIN13: "-DPFILL_XIN13"}
# a tile of each kind whose key block's first two key_ok rows hold one that key_ok_r9 rejects
R9_TILE = {R.S8: 86424, R.BITS: 16165, **({R.BU: 40682} if hasattr(R, "BU") else {})}
SHIM = """#include <cstdio>
#include "pfill_host.cpp"
extern "C" uint64_t r9_dfill_ref(uint64_t c0, uint64_t c1, uint64_t tile, int nc, int steps) {
  return dfill_ref(c0, c1, tile, nc, steps);
}
extern "C" int r9_key_ok(uint32_t a) { return pf_key_ok_r9(a); }
extern "C" void r9_keys(const uint64_t* st, uint32_t* k) {
  const PfKeys p = pf_keys(st);
  k[0] = p.a0; k[1] = p.a1; k[2] = p.k0; k[3] = p.k1;
}
"""

# ---------------------------------------------------------------- receipt_folds_r9.py, verbatim
M32 = 0xFFFFFFFF


def key_ok(a: int) -> bool:
    """pfill's key condition (pfill_ref_bu.key_ok)."""
    m, r = a & 0xFFFFFF, a & 31
    return bool(a & 1) and 256 < m < (1 << 24) - 256 and r not in (1, 31)


def int8_ratio(lam: int, bits: int = 30) -> bool:
    """lam b = b' mod 2^bits for some int8 b != 0 and b': a merge one combined int8 row or column computes."""
    return any(((lam * b + 128) % (1 << bits)) < 256 for b in range(-128, 128) if b)


def key_ok_r9(a: int) -> bool:
    """key_ok, and neither a nor -a is an int8 ratio mod 2^30, so no int8 merge sits at a bad lam of either fold.
    (XOR-in's other key-derived bad lam, ±a1 2^19, is never an int8 ratio: it needs b = 0 mod 2^11.)"""
    return key_ok(a) and not int8_ratio(a) and not int8_ratio((-a) & M32)


def rotl(x: np.ndarray, r: int) -> np.ndarray:
    x = x.astype(np.uint32)
    return (x << np.uint32(r)) | (x >> np.uint32(32 - r))


def xorin_step(y, d0, d1, d2, d3, a0, a1):
    y = (rotl(y, 13) ^ d0) * a0 + d1
    return (rotl(y, 19) ^ d2) * a1 + d3


def lop3imad_step(y, d0, d1, d2, d3, a0, a1, k0, k1):
    y = (y ^ d0 ^ k0) * a0 + d1
    return (y ^ d2 ^ k1) * a1 + d3
# ----------------------------------------------------------------


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    gxx = shutil.which("g++")
    if not gxx:
        pytest.skip("no g++ for the host build of pfill.cuh")
    d = tmp_path_factory.mktemp("pfill-r9")
    (d / "shim.cpp").write_text(SHIM)
    libs = {}

    def get(kind, flag=None):
        if (kind, flag) not in libs:
            so = d / f"r9-{kind}-{(flag or 'ram').lstrip('-')}.so"
            subprocess.run([gxx, "-O2", "-std=c++17", "-shared", "-fPIC", "-Wno-unknown-pragmas", f"-I{HERE}",
                            *KINDS[kind], *([flag] if flag else []), "-o", str(so), str(d / "shim.cpp")], check=True)
            lib = ctypes.CDLL(str(so))
            lib.r9_dfill_ref.restype = ctypes.c_uint64
            lib.r9_dfill_ref.argtypes = [ctypes.c_uint64] * 3 + [ctypes.c_int] * 2
            lib.r9_key_ok.restype = ctypes.c_int
            lib.r9_key_ok.argtypes = [ctypes.c_uint32]
            lib.r9_keys.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32)]
            libs[kind, flag] = lib
        return libs[kind, flag]

    return get


@pytest.mark.parametrize("fold", list(FOLDS))
@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize(("tile", "steps"), [(0, 1), (1000003, 13), (77, 32), ("r9", 4)])
def test_r9_twin_equals_host_reference(build, kind, fold, tile, steps):
    """NC 4 of a -DPFILL_LI or -DPFILL_XIN13 build is the twin's fold; NC 2 (duo) there takes the build's r9 keys; the
    flagless build's NC 4 stays ram. Tile "r9" is one whose keys key_ok_r9 changes."""
    tile = R9_TILE[kind] if tile == "r9" else tile
    lib, ram = build(kind, FOLDS[fold]), build(kind)
    got = lib.r9_dfill_ref(C0, C1, tile, 4, steps)
    assert R.receipt(C0, C1, tile, fold, steps, kind) == got
    assert got != ram.r9_dfill_ref(C0, C1, tile, 4, steps)
    assert ram.r9_dfill_ref(C0, C1, tile, 4, steps) == R.receipt(C0, C1, tile, R.RAM, steps, kind)
    assert lib.r9_dfill_ref(C0, C1, tile, 2, steps) == R.receipt(C0, C1, tile, R.DUO, steps, kind, r9=True)


@pytest.mark.parametrize("kind", list(R9_TILE))
def test_r9_tile_draws_past_a_rejected_key(kind):
    kb = R.block(C0, C1, R9_TILE[kind], R.CTR_KEY, R.DOMS[kind])
    assert R.keys(kb)[:2] != R.keys(kb, r9=True)[:2]
    assert not all(key_ok_r9(a) for a in R.keys(kb)[:2]) and all(key_ok_r9(a) for a in R.keys(kb, r9=True)[:2])


def test_r9_step_is_p2ffs():
    """pfill_ref.absorb's LI and XIN13 steps are the r9 twins, over a chain of 8 steps from y = pos."""
    rng = np.random.default_rng(9)
    while True:
        a0, a1, k0, k1 = (int(x) for x in rng.integers(0, 1 << 32, 4, dtype=np.uint64))
        if key_ok_r9(a0) and key_ok_r9(a1):
            break
    a0, a1, k0, k1 = (np.uint32(v) for v in (a0, a1, k0, k1))
    d = (rng.integers(-(1 << 12), 1 << 12, (8, 4, 4096)) & M32).astype(np.uint32)
    y0 = np.arange(4096, dtype=np.uint32)
    z0 = (y0.astype(np.uint64) * 0x9E3779B9 & M32).astype(np.uint32)
    with np.errstate(over="ignore"):
        for fold, ref in ((R.LI, lambda y, w: lop3imad_step(y, *w, a0, a1, k0, k1)),
                          (R.XIN13, lambda y, w: xorin_step(y, *w, a0, a1))):
            y, z, yr = y0, z0, y0
            for s in range(8):
                y, z = R.absorb(fold, y, z, *d[s], a0, a1, k0, k1)
                yr = ref(yr, d[s])
                assert np.array_equal(y, yr), (fold, s)
            assert np.array_equal(z, z0)


def _adversarial_keys(rng: np.random.Generator, n: int) -> list[int]:
    """Keys with a b = e mod 2^30 for b in 1..128 and e near the edges of [-128, 128]: a = (e / 2^s) b'^-1 mod 2^(30 - s),
    b = 2^s b' with b' odd, under random high bits."""
    out = []
    while len(out) < n:
        b = int(rng.integers(1, 129))
        s = (b & -b).bit_length() - 1
        e = int(rng.choice([-131, -130, -129, -128, -127, -126, 126, 127, 128, 129, 130, 131]))
        if e % (1 << s):
            continue
        lo = ((e >> s) * pow(b >> s, -1, 1 << (30 - s))) % (1 << (30 - s))
        out.append((lo + (int(rng.integers(0, 1 << (2 + s))) << (30 - s))) & M32)
    return out


def test_r9_key_check_is_p2ffs(build):
    """pf_key_ok_r9 (host build) and pfill_ref.key_ok_r9 agree with the r9 twin's key_ok_r9 on random keys and on keys
    built at the edges of its int8 window, which it both accepts and rejects; the default keys pass."""
    rng = np.random.default_rng(1117)
    lib = build(R.S8, "-DPFILL_LI")
    rand = [int(x) for x in rng.integers(0, 1 << 32, 1500, dtype=np.uint64)] + [0x9E3779B9, 0x7F4A7C15]
    edge = [a for a in _adversarial_keys(rng, 3000) if key_ok(a)]
    for keys in (rand, edge):
        want = [key_ok_r9(a) for a in keys]
        assert [bool(lib.r9_key_ok(a)) for a in keys] == want
        assert [R.key_ok_r9(a) for a in keys] == want
    assert 0 < sum(key_ok_r9(a) for a in edge) < len(edge)
    assert key_ok_r9(0x9E3779B9) and key_ok_r9(0x7F4A7C15)


def test_r9_keys_skip_a_rejected_row(build):
    """pf_keys in an r9 build takes the first two rows passing pf_key_ok_r9, past a key_ok row it rejects."""
    bad = next(a for a in _adversarial_keys(np.random.default_rng(3), 4000) if key_ok(a) and not key_ok_r9(a))
    good = [a for a in (0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D) if key_ok_r9(a)]
    assert len(good) >= 2
    rows = np.array([bad] + good[:2] + [0] * 11 + [0x11111111, 0x22222222], np.uint32)
    bs = [int.from_bytes(int(r).to_bytes(4, "little"), "big") for r in rows]
    st = (ctypes.c_uint64 * 8)(*[bs[2 * i] << 32 | bs[2 * i + 1] for i in range(8)])
    for flag, want in ((None, (bad, good[0])), ("-DPFILL_LI", tuple(good[:2])), ("-DPFILL_XIN13", tuple(good[:2]))):
        k = (ctypes.c_uint32 * 4)()
        build(R.S8, flag).r9_keys(st, k)
        assert tuple(k) == want + (0x11111111, 0x22222222) == R.keys(rows, r9=flag is not None)


def test_r9_flags_exclude_xin7_and_each_other(build, tmp_path):
    gxx = shutil.which("g++")
    for flags, msg in ((["-DPFILL_XIN"], "retired"), (["-DPFILL_LI", "-DPFILL_XIN13"], "one receipt fold")):
        r = subprocess.run([gxx, "-std=c++17", "-fsyntax-only", "-Wno-unknown-pragmas", f"-I{HERE}", *flags,
                            str(HERE / "pfill_host.cpp")], capture_output=True, text=True)
        assert r.returncode != 0 and msg in r.stderr


@pytest.mark.parametrize("kind", list(KINDS))
def test_r9_ho_holds_on_hashed_rows(kind, monkeypatch):
    """(HO), LI's layout premise: no hand-off of a chain, (d1, d2) or (d3, the next d0), joins two words with an equal or
    complementary row or column. The filler's rows and columns are hashed, so a tile of S = 512 fails it with
    probability about 2^-15 (2^17 row pairs, each 2^-31). A planted duplicate row fails it."""
    assert all(R.ho_ok(C0, C1, tile, steps, kind) for tile, steps in ((0, 8), (77, 32)))
    rows = R.rows

    def planted(c0, c1, tile, base, i, kd):
        r = rows(c0, c1, tile, base, i, kd)
        return np.concatenate([r[:8], r[:8]]) if base == 0 and i == 3 else r

    monkeypatch.setattr(R, "rows", planted)
    assert not R.ho_ok(C0, C1, 0, 8, kind)
