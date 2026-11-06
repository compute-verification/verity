"""`cuda_lib.load`, how the benchmarks open a built CUDA library, against the C library this process already maps."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import cuda_lib  # noqa: E402

STRLEN = ([ctypes.c_char_p], ctypes.c_size_t)


@pytest.fixture(scope="module")
def libc() -> str:
    maps = Path("/proc/self/maps")
    found = sorted({line.split()[-1] for line in maps.read_text().splitlines()
                    if line.split()[-1].rsplit("/", 1)[-1].startswith("libc.so")}) if maps.exists() else []
    if not found:
        pytest.skip("no mapped C library to load")
    return found[0]


def test_functions_get_their_argument_and_result_types(libc):
    lib = cuda_lib.load(libc, {"abs": [ctypes.c_int], "strlen": STRLEN, "free": ([ctypes.c_void_p], None)})
    assert (lib.abs.argtypes, lib.abs.restype) == ([ctypes.c_int], ctypes.c_int)
    assert (lib.strlen.argtypes, lib.strlen.restype) == ([ctypes.c_char_p], ctypes.c_size_t)
    assert lib.free.restype is None
    assert lib.abs(-3) == 3 and lib.strlen(b"verity") == 6
    with pytest.raises(ctypes.ArgumentError):
        lib.abs("three")


def test_an_optional_function_is_bound_where_the_library_exports_it(libc):
    lib = cuda_lib.load(libc, {}, {"strlen": STRLEN, "verity_not_exported": [ctypes.c_int]})
    assert lib.strlen.restype is ctypes.c_size_t
    assert not hasattr(lib, "verity_not_exported")


def test_a_missing_function_names_itself(libc):
    with pytest.raises(AttributeError, match="verity_not_exported"):
        cuda_lib.load(libc, {"strlen": STRLEN, "verity_not_exported": [ctypes.c_int]})
