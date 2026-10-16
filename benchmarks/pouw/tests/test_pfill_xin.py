"""pfill's XOR-in fold (-DPFILL_XIN, NC 4's fold in that build): pfill_ref.py's XIN receipt against pfill.cuh's host
`dfill_ref` built with g++ and the flag, the reference dgate holds the GPU to, in each integer tile kind; and the fold's
point, that a receipt from Q = a0 d0 + d1 alone (ram's only read of a pair) no longer matches
(note:20261016T0917Z-draft-fullfloor-s-r8 §1.2)."""

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
FLAGS = {R.S8: [], R.BITS: ["-DPFILL_BITS"], **({R.BU: ["-DPFILL_BU", "-mpopcnt"]} if hasattr(R, "BU") else {})}
SHIM = ('#include "pfill_host.cpp"\nextern "C" uint64_t xin_dfill_ref(uint64_t c0, uint64_t c1, uint64_t tile, '
        "int nc, int steps) { return dfill_ref(c0, c1, tile, nc, steps); }\n")


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    gxx = shutil.which("g++")
    if not gxx:
        pytest.skip("no g++ for the host build of pfill.cuh")
    d = tmp_path_factory.mktemp("pfill-xin")
    (d / "shim.cpp").write_text(SHIM)
    libs = {}

    def get(kind, xin):
        if (kind, xin) not in libs:
            so = d / f"xin-{kind}-{int(xin)}.so"
            subprocess.run([gxx, "-O2", "-std=c++17", "-shared", "-fPIC", "-Wno-unknown-pragmas", f"-I{HERE}",
                            *FLAGS[kind], *(["-DPFILL_XIN"] if xin else []), "-o", str(so), str(d / "shim.cpp")],
                           check=True)
            lib = ctypes.CDLL(str(so))
            lib.xin_dfill_ref.restype = ctypes.c_uint64
            lib.xin_dfill_ref.argtypes = [ctypes.c_uint64] * 3 + [ctypes.c_int] * 2
            libs[kind, xin] = lib
        return libs[kind, xin]

    return get


@pytest.mark.parametrize("kind", list(FLAGS))
@pytest.mark.parametrize(("tile", "steps"), [(0, 1), (1000003, 13), (77, 32)])
def test_xin_twin_equals_host_reference(build, kind, tile, steps):
    got = build(kind, True).xin_dfill_ref(C0, C1, tile, 4, steps)
    assert R.receipt(C0, C1, tile, R.XIN, steps, kind) == got
    assert got != build(kind, False).xin_dfill_ref(C0, C1, tile, 4, steps)
    assert build(kind, False).xin_dfill_ref(C0, C1, tile, 4, steps) == R.receipt(C0, C1, tile, R.RAM, steps, kind)
    assert build(kind, True).xin_dfill_ref(C0, C1, tile, 2, steps) == R.receipt(C0, C1, tile, R.DUO, steps, kind)


def test_xin_reads_both_words_of_a_pair():
    """Ram's step is the same map of y for every pair with one Q; XOR-in's isn't (2^8 states, every d0 at one Q)."""
    y = np.arange(1 << 8, dtype=np.uint32) * np.uint32(0x01010101)
    a0, q = 0x9E3779B1, 12345
    ram, xin = set(), set()
    for d in range(-32, 33):
        d0, d1 = np.uint32(d & R.M32), np.uint32((q - a0 * d) & R.M32)
        ram.add(((R.rotl(y, 7) + d0) * np.uint32(a0) + d1).tobytes())
        xin.add(((R.rotl(y, 7) ^ d0) * np.uint32(a0) + d1).tobytes())
    assert len(ram) == 1 and len(xin) == 65
