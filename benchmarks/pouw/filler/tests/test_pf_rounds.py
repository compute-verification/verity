"""pfill's PF_ROUNDS cost dial: the NC 4 absorb folds each committed word
PF_ROUNDS times. The default (and -DPF_ROUNDS=1) take the single-absorb path, byte-identical to the fold of record;
-DPF_ROUNDS=R>1 re-folds the same word R times (a serial u32 chain the GPU measures the cost of a round), which moves the
receipt but leaves the GPU and the host dfill_ref bit-exact. Here, on pfill.cuh's host tile built with g++: the default
build equals an explicit -DPF_ROUNDS=1, three rounds move the receipt for XIN13, LI and ram, and -DPF_ROUNDS=0 fails the
static_assert."""
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
#: the NC 4 folds the dial wraps: ram (no flag), and the r9 LI and XIN13 (each its build flag)
FOLD_FLAG = {"ram": [], "li": ["-DPFILL_LI"], "xin13": ["-DPFILL_XIN13"]}
CASES = [(0, 4), (1000003, 13), ((1 << 62) | 99, 8)]  # (tile, steps)


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    """Build pfill.cuh's host tile once per flag set and reuse it, so --quick builds only the libs its tests ask for."""
    gxx = shutil.which("g++")
    if not gxx:
        pytest.skip("no g++ for the host build of pfill.cuh")
    d = tmp_path_factory.mktemp("pf-rounds")
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
            cache[key] = lib
        return cache[key]

    return _build


def _ref(lib, tile, steps):
    return lib.pfill_ref(C0, C1, tile, R.RAM, steps, int(R.S8), R.W)


def test_round_1_equals_the_default_build(build):
    """The macro: -DPF_ROUNDS=1 takes the single-absorb path, byte-identical to the default build (the fold of record)."""
    base = build(FOLD_FLAG["xin13"])
    r1 = build([*FOLD_FLAG["xin13"], "-DPF_ROUNDS=1"])
    for tile, steps in CASES:
        assert _ref(base, tile, steps) == _ref(r1, tile, steps)


@pytest.mark.parametrize("fold", ["xin13", "li", pytest.param("ram", marks=pytest.mark.slow)])
def test_rounds_change_the_receipt(build, fold):
    """-DPF_ROUNDS=3 re-folds each committed word three times: the receipt moves, so the dial's loop is live."""
    base = build(FOLD_FLAG[fold])
    r3 = build([*FOLD_FLAG[fold], "-DPF_ROUNDS=3"])
    for tile, steps in CASES:
        assert _ref(base, tile, steps) != _ref(r3, tile, steps)


def test_pf_rounds_zero_is_rejected(tmp_path):
    """PF_ROUNDS is bounded to 1..64 by a static_assert, so -DPF_ROUNDS=0 fails to compile, with the fix in the message."""
    gxx = shutil.which("g++")
    if not gxx:
        pytest.skip("no g++")
    p = subprocess.run([gxx, "-O2", "-std=c++17", "-shared", "-fPIC", "-Wno-unknown-pragmas", "-DPF_ROUNDS=0",
                        "-o", str(tmp_path / "bad.so"), str(HERE / "pfill_host.cpp")], capture_output=True, text=True)
    assert p.returncode != 0 and "PF_ROUNDS is 1..64" in p.stderr + p.stdout
