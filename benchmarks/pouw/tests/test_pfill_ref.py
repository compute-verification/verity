"""pfill's Python twin (decode_capacity/pfill_ref.py, hashlib's SHA-512) against pfill.cuh's host `dfill_ref` built with g++
(decode_capacity/pfill_host.cpp), the reference `dcap dgate` holds the GPU kernel to bit for bit, in both tiles (s8, and
e2m1 under -DPFILL_E2M1); and the properties the filler's proved floor reads: the operands are uniform {±1} noise on
both sides with nothing else in them, every k32 (e2m1: k64) product is exactly 32 - 2 popcount(a ^ b) (64 - ...), and the
receipt depends on every fragment, on the tile id, on the coins and on the tile's kind."""

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


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    gxx = shutil.which("g++")
    if not gxx:
        pytest.skip("no g++ for the host build of pfill.cuh")
    so = tmp_path_factory.mktemp("pfill-host") / "pfill_host.so"
    subprocess.run([gxx, "-O2", "-std=c++17", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror", "-Wno-unknown-pragmas",
                    "-o", str(so), str(HERE / "pfill_host.cpp")], check=True)
    cdll = ctypes.CDLL(str(so))
    cdll.pfill_ref.restype = ctypes.c_uint64
    cdll.pfill_ref.argtypes = [ctypes.c_uint64] * 3 + [ctypes.c_int] * 3
    cdll.pfill_e2m1x8.restype = ctypes.c_uint32
    cdll.pfill_e2m1x8.argtypes = [ctypes.c_uint32]
    return cdll


@pytest.fixture(scope="module")
def host(lib):
    return lambda c0, c1, tile, nc, steps, e2m1=False: lib.pfill_ref(c0, c1, tile, nc, steps, int(e2m1))


@pytest.mark.parametrize("e2m1", [False, True])
@pytest.mark.parametrize("nc", [R.XOR, R.DUO, R.RAM])
@pytest.mark.parametrize("steps", [1, 4, 8, 13])
@pytest.mark.parametrize("tile", [0, 1, 1000003])
def test_twin_equals_host_reference(host, nc, steps, tile, e2m1):
    assert R.receipt(C0, C1, tile, nc, steps, e2m1) == host(C0, C1, tile, nc, steps, e2m1)


@pytest.mark.parametrize("e2m1", [False, True])
def test_twin_equals_host_reference_at_measured_depth(host, e2m1):
    """S = 32, dgate's short depth, at a second pair of coins."""
    c0, c1 = 0x7F88ED5DFF52F729, 0xE9FECEAE8FAABB27
    assert R.receipt(c0, c1, 77, R.RAM, 32, e2m1) == host(c0, c1, 77, R.RAM, 32, e2m1)


def test_operands_are_balanced_pm1_noise():
    """Each entry is 1 - 2 bit of a SHA-512 output bit: over a tile's B and a warp's first fragments the two values are
    equally likely (to 5 sigma), and the s8 bytes pf_pm1 writes are exactly 1 and -1."""
    for e2m1 in (False, True):
        bb = np.concatenate([R.rows(C0, C1, 9, R.CTR_B, b, e2m1) for b in range(R.TN // 16)])
        bits = np.unpackbits((bb if e2m1 else bb.astype(np.uint32)).view(np.uint8))
        n = bits.size
        assert n == R.TN * (64 if e2m1 else 32)
        assert abs(int(bits.sum()) - n / 2) < 5 * np.sqrt(n) / 2
    for nib in range(16):
        word = 0x01010101 | (((nib * 0x00204081) & 0x01010101) * 0xFE) & 0xFFFFFFFF
        got = np.frombuffer(np.uint32(word).tobytes(), np.int8)
        assert got.tolist() == [1 - 2 * ((nib >> q) & 1) for q in range(4)]


def _e2m1(code: int) -> float:
    s, e, m = code >> 3, (code >> 1) & 3, code & 1
    return (-1.0) ** s * (m * 0.5 if e == 0 else 2.0 ** (e - 1) * (1 + m / 2))


def test_e2m1_codes_are_pm1(lib):
    """pf_e2m1x8 puts entry q of a byte in nibble q as the E2M1 code of 1 - 2 bit: +1.0 (0x2) or -1.0 (0xA)."""
    for byte in range(256):
        word = lib.pfill_e2m1x8(byte)
        assert [_e2m1((word >> (4 * q)) & 15) for q in range(8)] == [1.0 - 2 * ((byte >> q) & 1) for q in range(8)]


@pytest.mark.parametrize("e2m1", [False, True])
def test_product_is_the_popcount_identity(e2m1):
    """C[r][n] = sum_k A[r][k] B[k][n] = K - 2 popcount(a_r ^ b_n) for the bit rows the twin and dfill_ref use, and the
    e2m1 tile's word is that integer's f32 bits (exact: |C| <= 64)."""
    k = 64 if e2m1 else 32
    a_rows, b_cols = R.rows(C0, C1, 3, 0, 0, e2m1), R.rows(C0, C1, 3, R.CTR_B, 0, e2m1)
    val = lambda w: np.array([1 - 2 * ((int(w) >> i) & 1) for i in range(k)])
    a = np.stack([val(w) for w in a_rows])
    b = np.stack([val(w) for w in b_cols]).T
    want = k - 2 * R.popcount(a_rows[:, None] ^ b_cols[None, :])
    assert (a @ b == want).all()
    got = R.words(a_rows[:, None] ^ b_cols[None, :], e2m1)
    exp = want.astype(np.float32).view(np.uint32) if e2m1 else (want & R.M32).astype(np.uint32)
    assert (got == exp).all()


def test_keys_pass_the_screen():
    for tile in range(64):
        for dom in (R.DOM, R.DOM4):
            a0, a1, _, _ = R.keys(R.block(C0, C1, tile, R.CTR_KEY, dom))
            assert R.key_ok(a0) and R.key_ok(a1) and a0 != a1


def test_receipt_depends_on_fragments_tile_coins_and_kind(host):
    """dgate's control (a tile cut short) differs, as do the next tile, the next coins and the other kind: a receipt
    replays nowhere."""
    for e2m1 in (False, True):
        for nc in (R.XOR, R.DUO, R.RAM):
            full = host(C0, C1, 5, nc, 32, e2m1)
            assert full != host(C0, C1, 5, nc, 24, e2m1)
            assert full != host(C0, C1, 5, nc, 31, e2m1)
            assert full != host(C0, C1, 6, nc, 32, e2m1)
            assert full != host(C0 ^ 1, C1, 5, nc, 32, e2m1)
            assert full != host(C0, C1, 5, nc, 32, not e2m1)
