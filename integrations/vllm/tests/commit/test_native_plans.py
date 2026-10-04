"""The native collector's plans across commitment runs (`native_plans.h`): two runs' worth, not the process's.

The header is the C++ collector's own plan book, compiled here on its own with a C shim (no torch, no CUDA). A step drives its
plan as `native_collect.cpp` does: `begin_step` takes it with `Plans::use`, a step whose tensors the plan does not hold learns it
(entries at 4-byte offsets) and `end_step` bumps its version, and the committer's `reset` calls `end_run` between runs. The
committer over these plans, its exports and its roots, is tested on a restatement of the collector in
`test_native_collect_plan_cache.py`; the collector itself, with its copies, in `test_native_collect.py` on a GPU.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import pytest

from verity_vllm.commit.committer import native_jit

CXX = shutil.which(os.environ.get("CXX", "g++"))
IN_CHECK = bool(os.environ.get("VERITY_SUITE_ROOT"))
TENSORS, DIMS = 9, 2

SHIM = r"""
#include "native_plans.h"

using native_plans::Entry;
using native_plans::Plan;
using native_plans::Plans;

extern "C" {
void* plans_new() { return new Plans(); }
void plans_delete(void* h) { delete static_cast<Plans*>(h); }
void plans_end_run(void* h) { static_cast<Plans*>(h)->end_run(); }
void plans_clear(void* h) { static_cast<Plans*>(h)->clear(); }
int64_t plans_size(void* h) { return static_cast<Plans*>(h)->size(); }
int64_t plans_entries(void* h) { return static_cast<Plans*>(h)->entries(); }
int64_t plans_version(void* h, int64_t key) {
  const Plan* p = static_cast<Plans*>(h)->find(key);
  return p ? p->version : -1;
}
// one step of `key` with `n` tensors of `dims` dimensions: followed when its plan holds them, else learned; returns the version
int64_t plans_step(void* h, int64_t key, int64_t n, int64_t dims, int* learned) {
  Plan& p = static_cast<Plans*>(h)->use(key);
  *learned = static_cast<int64_t>(p.entries.size()) != n;
  if (*learned) {
    p.entries.clear();
    int64_t cursor = 0;
    for (int64_t i = 0; i < n; ++i) {
      Entry e;
      e.name_id = i; e.nbytes = 4 * i + 2; e.off = (cursor + 3) & ~static_cast<int64_t>(3); e.gap = e.off - cursor; e.dtype = 15;
      e.sizes.assign(static_cast<size_t>(dims), i + 1);
      cursor = e.off + e.nbytes;
      p.entries.push_back(std::move(e));
    }
    p.version += 1;
    p.used = cursor;
  }
  return p.version;
}
}
"""


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    if CXX is None and not IN_CHECK:
        pytest.skip("needs a C++ compiler")
    d = tmp_path_factory.mktemp("native_plans")
    (d / "shim.cpp").write_text(SHIM)
    so = d / "libplans.so"
    r = subprocess.run([CXX or "g++", "-std=c++17", "-O1", "-shared", "-fPIC", f"-I{native_jit.HERE}", str(d / "shim.cpp"), "-o", str(so)],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"native_plans.h does not build:\n{r.stderr}"
    lib = ctypes.CDLL(str(so))
    lib.plans_new.restype = ctypes.c_void_p
    for fn in ("plans_delete", "plans_end_run", "plans_clear"):
        getattr(lib, fn).argtypes = [ctypes.c_void_p]
    for fn in ("plans_size", "plans_entries"):
        getattr(lib, fn).argtypes, getattr(lib, fn).restype = [ctypes.c_void_p], ctypes.c_int64
    lib.plans_version.argtypes, lib.plans_version.restype = [ctypes.c_void_p, ctypes.c_int64], ctypes.c_int64
    lib.plans_step.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_int)]
    lib.plans_step.restype = ctypes.c_int64
    return lib


class Book:
    """One collector's plans."""

    def __init__(self, lib) -> None:
        self.lib, self.h = lib, ctypes.c_void_p(lib.plans_new())

    def __del__(self) -> None:
        self.lib.plans_delete(self.h)

    def step(self, key: int, n: int = TENSORS) -> tuple[int, bool]:
        learned = ctypes.c_int(0)
        version = self.lib.plans_step(self.h, key, n, DIMS, ctypes.byref(learned))
        return version, bool(learned.value)

    def end_run(self) -> None:
        self.lib.plans_end_run(self.h)

    def clear(self) -> None:
        self.lib.plans_clear(self.h)

    def size(self) -> int:
        return self.lib.plans_size(self.h)

    def entries(self) -> int:
        return self.lib.plans_entries(self.h)

    def version(self, key: int) -> int:
        return self.lib.plans_version(self.h, key)


RUNS, STEPS = 200, 50


def test_the_plans_stay_bounded_when_no_plan_recurs(lib):
    """200 runs of 50 steps, every step a plan of its own (network traces' decode windows): after each run the collector holds that
    run's plans, during one at most two runs' worth. Without `end_run` (main's collector) it holds every plan it ever learned."""
    book, kept = Book(lib), Book(lib)
    peak, after = 0, set()
    for r in range(RUNS):
        for s in range(STEPS):
            assert book.step(r * STEPS + s)[1]
            kept.step(r * STEPS + s)
        peak = max(peak, book.size())
        book.end_run()
        after.add((book.size(), book.entries()))
    assert after == {(STEPS, STEPS * TENSORS)} and peak == 2 * STEPS
    assert (kept.size(), kept.entries()) == (RUNS * STEPS, RUNS * STEPS * TENSORS)


def test_a_record_run_follows_the_plans_of_the_warm_up_before_it(lib):
    """commit.py: the warm-up learns every plan, the committer is reset twice (after the warm-up, and before the instrumented arm),
    and every record run after it follows the same plans at their versions, learning none."""
    book = Book(lib)
    assert all(book.step(k) == (1, True) for k in range(STEPS))
    book.end_run()
    book.end_run()
    for _pair in range(3):
        assert [book.step(k) for k in range(STEPS)] == [(1, False)] * STEPS
        book.end_run()
    assert book.size() == STEPS


def test_a_plan_every_run_uses_is_never_dropped(lib):
    """A prefill plan every window repeats stays at its version across runs whose other plans never recur."""
    book = Book(lib)
    for r in range(20):
        assert book.step(0) == (1, r == 0)
        for s in range(1, STEPS):
            book.step(r * STEPS + s)
        book.end_run()
        assert book.version(0) == 1
    assert book.size() == STEPS


def test_a_plan_learned_again_after_a_drop_takes_a_version_above_every_dropped_one(lib):
    """A plan relearned once in its run (version 2), dropped after a run that did not use it, is learned again when its key
    recurs, at version 4: it starts above the dropped version 2, so a (key, version) the worker may still hold never names it. A
    new key starts there too. `clear` drops every plan the same way: above the highest version it held (4)."""
    book = Book(lib)
    assert book.step(7) == (1, True)
    assert book.step(7, TENSORS + 1) == (2, True)
    book.end_run()
    book.step(8)
    book.end_run()
    assert (book.version(7), book.version(8)) == (-1, 1)
    assert book.step(7) == (4, True)
    assert book.step(9) == (4, True)
    book.clear()
    assert book.size() == 0 and book.step(8) == (6, True)
