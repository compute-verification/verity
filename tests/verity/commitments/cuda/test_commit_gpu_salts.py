"""commit_gpu's refusal of a salt buffer (``commit_gpu/salts_check.h``), on the CPU: ``Committer::salts`` hands a salt
kernel its buffer's address only after the buffer's stated length is exactly the launch's leaves' 192 bytes each, its
address is a multiple of 16 bytes and the slots are the committer's."""
from __future__ import annotations

import ctypes
from pathlib import Path

import cuda_build
import pytest

DIR = Path(__file__).resolve().parents[4] / "verity" / "commitments" / "cuda" / "commit_gpu"
SHIM = """#include "{header}"
extern "C" int refused(uint64_t addr, uint64_t bytes, uint64_t n, int small, int slots, int s0, int ns, char* why) {{
  return commit_gpu::salts_refused(addr, bytes, n, small != 0, slots, s0, ns, why, 160);
}}
"""


@pytest.fixture(scope="module")
def refused(tmp_path_factory: pytest.TempPathFactory):
    tmp = tmp_path_factory.mktemp("salts")
    src = tmp / "shim.cpp"
    src.write_text(SHIM.format(header=DIR / "salts_check.h"))
    lib = ctypes.CDLL(str(cuda_build.host_lib(src, (), tmp / "shim.so")))
    u64, i = ctypes.c_uint64, ctypes.c_int
    lib.refused.argtypes, lib.refused.restype = [u64, u64, u64, i, i, i, i, ctypes.c_char_p], i

    def call(addr: int, nbytes: int, n: int, small: bool = False, slots: int = 1, s0: int = 0, ns: int = 1) -> str | None:
        why = ctypes.create_string_buffer(160)
        return why.value.decode() if lib.refused(addr, nbytes, n, small, slots, s0, ns, why) else None
    return call


def test_a_salt_buffer_is_its_leaves_192_bytes_each_at_a_16_byte_address(refused) -> None:
    n, at = 5, 0x7000_0000
    assert refused(at, 192 * n, n) is None
    for nbytes in (192 * (n - 1), 192 * (n + 1), 192 * n - 1):
        assert refused(at, nbytes, n) == f"salts: {nbytes} bytes for {n} leaves in 1 slot, which take {192 * n} (192 a leaf)"
    for addr in (0, at + 1, at + 8):
        assert "multiple of 16 bytes" in refused(addr, 192 * n, n)
    assert refused(at, 192 * n, n, s0=1) == "salt slots 1 .. 1 of 1"


def test_the_small_paths_slots_each_take_their_own_salts_after_the_last_slots(refused) -> None:
    n, at, slots = 5, 0x7000_0000, 3
    assert refused(at, 192 * n * slots, n, True, slots, 0, slots) is None
    assert refused(at, 192 * n, n, True, slots, 2, 1) is None
    assert refused(at, 192 * n, n, True, slots, 0, 2) == f"salts: {192 * n} bytes for {n} leaves in 2 slots, which take {384 * n} (192 a leaf)"
    for s0, ns in ((-1, 1), (0, 0), (2, 2), (3, 1), (0, 4)):
        assert refused(at, 192 * n * max(ns, 1), n, True, slots, s0, ns) == f"salt slots {s0} .. {s0 + ns - 1} of {slots}"


def test_the_committer_asks_before_either_salt_kernel_and_its_callers_state_the_length() -> None:
    cuh = (DIR / "commit_gpu.cuh").read_text()
    body = cuh[cuh.index("  void salts(const uint8_t* salts, u64 bytes,"):]
    assert body.index("commit_gpu::salts_refused(") < body.index("salt_small<<<") < body.index("salt_kernel<false><<<")
    calls = [line.strip() for line in (DIR / "commit_gpu.cu").read_text().splitlines() if "cm.salts(" in line]
    assert len(calls) == 3 and all("cm.salts(dsalts, s.size() * salt_slots," in c or "cm.salts(salts, 192 * n," in c for c in calls)
