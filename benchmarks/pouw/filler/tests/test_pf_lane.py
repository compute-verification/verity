"""pfill's lane chains (-DPF_LANE=k, note:20261017T0652Z-draft-filler §4): k verified lane steps a B fragment on chains
independent of the MMAs, folded into the tile's receipt. Here, on pfill.cuh's host tile built with g++ (the reference
`dcap dgate` holds the GPU to): -DPF_LANE=0 is byte-identical to the default build; at k > 0 the host receipt equals
pfill_ref.py's, which reads each lane's words from B's matrix as the MMA's B operand rather than from the shared-memory
image the device and the host build, in the s8, bits and bU tiles and in a phase tile's width; and lanes move the
receipt."""
from __future__ import annotations

import ctypes
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("pfill_ref", HERE / "pfill_ref.py")
R = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = R
_spec.loader.exec_module(R)

C0, C1 = 0x0123456789ABCDEF, 0xFEDCBA9876543210
CASES = [(0, 4), (1000003, 8), ((1 << 62) | 99, 4)]  # (tile, steps); the last a phase id (PF_PW warps)


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    gxx = shutil.which("g++")
    if not gxx:
        pytest.skip("no g++ for the host build of pfill.cuh")
    d = tmp_path_factory.mktemp("pf-lane")
    cache: dict[tuple[str, ...], ctypes.CDLL] = {}

    def _build(flags: list[str]) -> ctypes.CDLL:
        key = tuple(flags)
        if key not in cache:
            so = d / ("pfill_host_" + ("_".join(f.lstrip("-D") for f in flags) or "default") + ".so")
            subprocess.run([gxx, "-O2", "-std=c++17", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror",
                            "-Wno-unknown-pragmas", *flags, "-o", str(so), str(HERE / "pfill_host.cpp")], check=True)
            lib = ctypes.CDLL(str(so))
            lib.pfill_ref.restype = ctypes.c_uint64
            lib.pfill_ref.argtypes = [ctypes.c_uint64] * 3 + [ctypes.c_int] * 4
            lib.pfill_width.restype = ctypes.c_int
            lib.pfill_width.argtypes = [ctypes.c_uint64]
            cache[key] = lib
        return cache[key]

    return _build


def _host(lib, tile, steps, kind, nc=R.DUO):
    return lib.pfill_ref(C0, C1, tile, nc, steps, int(kind), lib.pfill_width(tile))


def test_no_lanes_is_the_default_build(build):
    base, l0 = build([]), build(["-DPF_LANE=0"])
    for tile, steps in CASES:
        for kind in (R.S8, R.BU):
            assert _host(base, tile, steps, kind) == _host(l0, tile, steps, kind)


@pytest.mark.parametrize("lanes", [1, 4, 6])
@pytest.mark.parametrize("kind", [R.BU, R.S8, R.BITS])
def test_host_lane_chains_equal_the_matrix_reference(build, lanes, kind):
    lib = build([f"-DPF_LANE={lanes}"])
    for tile, steps in CASES:
        w = lib.pfill_width(tile)
        assert _host(lib, tile, steps, kind) == R.receipt(C0, C1, tile, R.DUO, steps, kind, w, lanes=lanes)


def test_lanes_move_the_receipt_and_the_fold_is_untouched(build):
    base, l4 = build([]), build(["-DPF_LANE=4"])
    for tile, steps in CASES:
        w = base.pfill_width(tile)
        h0, h4 = _host(base, tile, steps, R.BU), _host(l4, tile, steps, R.BU)
        assert h0 != h4
        assert h4 ^ h0 == R.lane_receipt(C0, C1, tile, steps, 4, R.BU, w)


def test_lane_count_is_bounded(tmp_path):
    gxx = shutil.which("g++")
    if not gxx:
        pytest.skip("no g++")
    p = subprocess.run([gxx, "-O2", "-std=c++17", "-shared", "-fPIC", "-Wno-unknown-pragmas", "-DPF_LANE=65",
                        "-o", str(tmp_path / "x.so"), str(HERE / "pfill_host.cpp")], capture_output=True, text=True)
    assert p.returncode != 0 and "PF_LANE is 0..64" in p.stderr
