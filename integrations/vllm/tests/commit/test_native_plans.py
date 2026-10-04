"""The native collector's plans across commitment runs (`native_plans.h`): two runs' worth, not the process's.

The header is the C++ collector's own plan book, compiled here on its own with a C shim (no torch, no CUDA). A step drives its
plan as `native_collect.cpp` does: `begin_step` takes it with `Plans::use`, a step whose tensors the plan does not hold learns it
(entries at 4-byte offsets) and `end_step` bumps its version, and the committer's `reset` calls `end_run` between runs. The shim
has its own `operator new` and `delete` (bound inside it, `-Bsymbolic`), which count the blocks the header frees on the calling
thread, so a test can tell what a reset frees from what a step frees. The committer over these plans, its exports and its roots,
is tested on a restatement of the collector in `test_native_collect_plan_cache.py`; the collector itself, with its copies, in
`test_native_collect.py` on a GPU.
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
#include <cstdlib>
#include <new>

#include "native_plans.h"

using native_plans::Entry;
using native_plans::Plan;
using native_plans::Plans;

static thread_local int64_t frees_here = 0;
void* operator new(std::size_t n) {
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { if (p) { ++frees_here; std::free(p); } }
void operator delete(void* p, std::size_t) noexcept { if (p) { ++frees_here; std::free(p); } }

extern "C" {
int64_t plans_frees() { return frees_here; }
void* plans_new() { return new Plans(); }
void plans_delete(void* h) { delete static_cast<Plans*>(h); }
void plans_end_run(void* h) { static_cast<Plans*>(h)->end_run(); }
void plans_clear(void* h) { static_cast<Plans*>(h)->clear(); }
int64_t plans_size(void* h) { return static_cast<Plans*>(h)->size(); }
int64_t plans_entries(void* h) { return static_cast<Plans*>(h)->entries(); }
int64_t plans_pending(void* h) { return static_cast<Plans*>(h)->pending(); }
int64_t plans_pending_entries(void* h) { return static_cast<Plans*>(h)->pending_entries(); }
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
    r = subprocess.run([CXX or "g++", "-std=c++17", "-O1", "-shared", "-fPIC", "-Wl,-Bsymbolic", f"-I{native_jit.HERE}", str(d / "shim.cpp"),
                        "-o", str(so)], capture_output=True, text=True)
    assert r.returncode == 0, f"native_plans.h does not build:\n{r.stderr}"
    lib = ctypes.CDLL(str(so))
    lib.plans_new.restype = ctypes.c_void_p
    lib.plans_frees.restype = ctypes.c_int64
    for fn in ("plans_delete", "plans_end_run", "plans_clear"):
        getattr(lib, fn).argtypes = [ctypes.c_void_p]
    for fn in ("plans_size", "plans_entries", "plans_pending", "plans_pending_entries"):
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

    def pending(self) -> tuple[int, int]:
        """Dropped plans not freed yet, and their entries."""
        return self.lib.plans_pending(self.h), self.lib.plans_pending_entries(self.h)

    def version(self, key: int) -> int:
        return self.lib.plans_version(self.h, key)

    def frees(self) -> int:
        """Blocks the shim's code has freed on this thread so far."""
        return self.lib.plans_frees()


RUNS, STEPS = 200, 50


def test_the_plans_stay_bounded_when_no_plan_recurs(lib):
    """200 runs of 50 steps, every step a plan of its own (network traces' decode windows): after each run the collector holds that
    run's plans, during one at most two runs' worth, counting the plans a reset dropped that the steps have not freed yet. Without
    `end_run` (main's collector) it holds every plan it ever learned."""
    book, kept = Book(lib), Book(lib)
    peak, after = 0, set()
    for r in range(RUNS):
        for s in range(STEPS):
            assert book.step(r * STEPS + s)[1]
            kept.step(r * STEPS + s)
            peak = max(peak, book.size() + book.pending()[0])
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


DROPPED, KEPT = 50, 3
RELEASE_PER_STEP = 2            # dropped plans a step frees at most: twice the rate a run of one new plan per step drops them
PLAN_BLOCKS = TENSORS + 2       # a plan's blocks: its node, its entries and each entry's sizes


def drop(book: Book) -> int:
    """Run A learns DROPPED plans, run B learns KEPT others, and B's reset drops A's: the blocks that reset freed."""
    for k in range(DROPPED):
        book.step(k)
    book.end_run()
    for k in range(KEPT):
        book.step(1000 + k)
    f = book.frees()
    book.end_run()
    return book.frees() - f


def test_a_reset_leaves_the_plans_it_drops_to_the_steps_of_the_next_run(lib):
    """The committer's reset runs between windows, where the served engine waits for it, and on a served chunk every plan of a
    window is dropped two resets later: a reset that frees them pays one block per committed tensor of a window. `end_run` takes
    the dropped plans out of the book at once and frees fewer blocks than it drops plans; the next run's steps free them, at most
    RELEASE_PER_STEP plans' worth per step, and all of them before that run ends."""
    probe = Book(lib)
    for k in range(DROPPED):
        probe.step(k)
    f = probe.frees()
    probe.clear()
    assert probe.frees() - f >= DROPPED * TENSORS, "the shim does not count the header's frees"

    book = Book(lib)
    at_reset = drop(book)
    assert book.size() == KEPT and book.version(0) == -1
    assert at_reset < DROPPED, f"end_run freed {at_reset} blocks on the resetting thread, dropping {DROPPED} plans"
    per_step = []
    for s in range(STEPS):
        f = book.frees()
        assert book.step(1000 + s % KEPT) == (1, False)
        per_step.append(book.frees() - f)
    assert max(per_step) <= RELEASE_PER_STEP * PLAN_BLOCKS, per_step
    assert sum(per_step) >= DROPPED * TENSORS, f"the next run's steps freed {sum(per_step)} blocks of {DROPPED * PLAN_BLOCKS} dropped"
    f = book.frees()
    book.end_run()
    assert book.frees() - f < DROPPED


def test_a_run_too_short_to_free_the_plans_dropped_before_it_frees_the_rest_at_its_end(lib):
    """A window with fewer steps than it takes to free the drop before it (an idle stretch after a busy one): its reset frees what
    is left, so every dropped plan is freed by the end of the run after the drop."""
    book = Book(lib)
    drop(book)
    f = book.frees()
    for s in range(3):
        book.step(1000 + s % KEPT)
    book.end_run()
    assert book.frees() - f >= DROPPED * TENSORS, f"{book.frees() - f} blocks freed by the end of the run after the drop"
    assert book.pending() == (0, 0)


def test_the_pending_plans_are_counted_apart_from_the_book_and_never_found(lib):
    """`stats()` reports the dropped plans the steps have not freed yet (`plans_pending`, `plan_entries_pending`) apart from the
    book's (`plans`, `plan_entries`). A pending plan is not found: a step of its key learns a plan of its own, at a version above
    the dropped one. Each step frees RELEASE_PER_STEP of them, and `clear` frees the rest."""
    book = Book(lib)
    drop(book)
    assert (book.size(), book.entries()) == (KEPT, KEPT * TENSORS)
    assert book.pending() == (DROPPED, DROPPED * TENSORS)
    assert all(book.version(k) == -1 for k in range(DROPPED))
    assert book.step(0) == (3, True) and book.version(0) == 3
    left = DROPPED - RELEASE_PER_STEP
    assert book.pending() == (left, left * TENSORS)
    for s in range(10):
        book.step(1000 + s % KEPT)
    left -= 10 * RELEASE_PER_STEP
    assert book.pending() == (left, left * TENSORS)
    book.clear()
    assert (book.size(), *book.pending()) == (0, 0, 0)
