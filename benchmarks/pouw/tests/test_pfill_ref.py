"""pfill's Python twin (decode_capacity/pfill_ref.py, hashlib's SHA-512) against pfill.cuh's host `dfill_ref` built with g++
(decode_capacity/pfill_host.cpp), the reference `dcap dgate` holds the GPU kernel to bit for bit; and the properties the
filler's proved floor reads: the operands are uniform {±1} noise on both sides with nothing else in them, every k32
product is exactly C = 32 - 2 popcount(a ^ b), and the receipt depends on every fragment, on the tile id and on the
coins."""

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
def host(tmp_path_factory):
    gxx = shutil.which("g++")
    if not gxx:
        pytest.skip("no g++ for the host build of pfill.cuh")
    lib = tmp_path_factory.mktemp("pfill-host") / "pfill_host.so"
    subprocess.run([gxx, "-O2", "-std=c++17", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror", "-Wno-unknown-pragmas",
                    "-o", str(lib), str(HERE / "pfill_host.cpp")], check=True)
    cdll = ctypes.CDLL(str(lib))
    cdll.pfill_ref.restype = ctypes.c_uint64
    cdll.pfill_ref.argtypes = [ctypes.c_uint64] * 3 + [ctypes.c_int] * 2
    return cdll.pfill_ref


@pytest.mark.parametrize("nc", [R.XOR, R.DUO, R.RAM])
@pytest.mark.parametrize("steps", [1, 4, 8, 13])
@pytest.mark.parametrize("tile", [0, 1, 1000003])
def test_twin_equals_host_reference(host, nc, steps, tile):
    assert R.receipt(C0, C1, tile, nc, steps) == host(C0, C1, tile, nc, steps)


def test_twin_equals_host_reference_at_measured_depth(host):
    """S = 32, dgate's short depth, at a second pair of coins."""
    c0, c1 = 0x7F88ED5DFF52F729, 0xE9FECEAE8FAABB27
    assert R.receipt(c0, c1, 77, R.RAM, 32) == host(c0, c1, 77, R.RAM, 32)


def test_operands_are_balanced_pm1_noise():
    """Each entry is 1 - 2 bit of a SHA-512 output bit: over a tile's B and a warp's first fragments the two values are
    equally likely (to 5 sigma), and the s8 bytes pf_pm1 writes are exactly 1 and -1."""
    bits = np.unpackbits(np.concatenate([R.block(C0, C1, 9, R.CTR_B + b) for b in range(R.TN // 16)]).view(np.uint8))
    n = bits.size
    assert abs(int(bits.sum()) - n / 2) < 5 * np.sqrt(n) / 2
    for nib in range(16):
        word = 0x01010101 | (((nib * 0x00204081) & 0x01010101) * 0xFE) & 0xFFFFFFFF
        got = np.frombuffer(np.uint32(word).tobytes(), np.int8)
        assert got.tolist() == [1 - 2 * ((nib >> q) & 1) for q in range(4)]


def test_product_is_the_popcount_identity():
    """C[r][n] = sum_k A[r][k] B[k][n] = 32 - 2 popcount(a_r ^ b_n) for the bit rows the twin and dfill_ref use."""
    a_rows, b_cols = R.block(C0, C1, 3, 0), R.block(C0, C1, 3, R.CTR_B)
    val = lambda w: 1 - 2 * ((int(w) >> np.arange(32)) & 1)
    a = np.stack([val(w) for w in a_rows])
    b = np.stack([val(w) for w in b_cols]).T
    want = 32 - 2 * R.popcount(a_rows[:, None] ^ b_cols[None, :])
    assert (a @ b == want).all()


def test_keys_pass_the_screen():
    for tile in range(64):
        a0, a1, _, _ = R.keys(R.block(C0, C1, tile, R.CTR_KEY))
        assert R.key_ok(a0) and R.key_ok(a1) and a0 != a1


def test_receipt_depends_on_fragments_tile_and_coins(host):
    """dgate's control (a tile cut short) differs, as do the next tile and the next coins: a receipt replays nowhere."""
    for nc in (R.XOR, R.DUO, R.RAM):
        full = host(C0, C1, 5, nc, 32)
        assert full != host(C0, C1, 5, nc, 24)
        assert full != host(C0, C1, 5, nc, 31)
        assert full != host(C0, C1, 6, nc, 32)
        assert full != host(C0 ^ 1, C1, 5, nc, 32)
