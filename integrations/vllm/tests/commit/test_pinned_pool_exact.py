"""[R17 fp8, F-fp8-POW2] The collector's pinned pool holds EXACTLY what it accounts for.

torch's CachingHostAllocator rounds every pinned request up to the next power of two and never returns freed blocks to the OS (torch
2.13.0+cu129 on cp-h100: 3 GiB -> RSS +4 GiB; 5 GiB + 7 B -> +8 GiB); pinned pages are committed at allocation, so the process RSS and the
cgroup carried up to 2x the pool's `bytes_total` -- 21,152 MiB of the B8 FP8 row's run-phase RSS outside every term of the host bound
(FP8-STAGING §6).  Exact mode (default) registers page-aligned anonymous memory instead: RSS == bytes_total, released on `release()`.
"""
from __future__ import annotations

import ctypes
import errno
import threading

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.commit.committer import native_host as nh  # noqa: E402
from verity_vllm.commit.committer.native_host import _PinnedPool  # noqa: E402

GPU = torch.cuda.is_available()


def _rss_bytes() -> int:
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("no VmRSS")


@pytest.mark.skipif(not GPU, reason="needs CUDA (cudaHostRegister)")
def test_exact_pool_rss_equals_its_accounting_and_is_released():
    torch.zeros(1, device="cuda")
    n = (1536 << 20) + 4097                       # 1.5 GiB + a page + 1 B: torch's allocator would take 2 GiB for this
    pool = _PinnedPool(exact=True)
    r0 = _rss_bytes()
    v = pool.take(n)
    assert pool.mode == "exact-registered-shared"
    assert v.numel() == n and pool.bytes_total == (n + 4095) // 4096 * 4096
    grew = _rss_bytes() - r0
    # RSS grows by the page-rounded request (touched by the registration), not by the power of two above it
    assert abs(grew - pool.bytes_total) < (64 << 20), (grew, pool.bytes_total)
    assert grew < (2048 << 20) - (256 << 20)
    # torch's `is_pinned()` only knows ITS allocator's blocks (False here) -- page-locking is the driver's: the exact mode exists only
    # after `cudaHostRegister` succeeded (`_pinned_block` raises otherwise); an async D2H copy into the block lands the device's bytes
    src = torch.arange(0, 4096, dtype=torch.uint8, device="cuda").repeat(n // 4096 + 1)[:n]
    torch.cuda.synchronize()
    v.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    assert int(v[4095]) == 255 and int(v[4096]) == 0 and torch.equal(v[:8192].cuda(), src[:8192])
    freed = pool.release()
    assert freed == (n + 4095) // 4096 * 4096 and pool.bytes_total == 0 and pool.mode is None
    assert pool.deferred_unmaps == 1, "the view `v` is alive: the mmap unmaps when it dies, and release() says so"
    del v, src
    import gc

    gc.collect()
    assert _rss_bytes() - r0 < (256 << 20), "the exact pool's memory goes back to the OS on release()"


def _cgroup_anon_shmem() -> tuple[int, int] | None:
    try:
        st = {l.split()[0]: int(l.split()[1]) for l in open("/sys/fs/cgroup/memory.stat")}
        return st["anon"], st["shmem"]
    except Exception:  # noqa: BLE001
        return None


@pytest.mark.skipif(not GPU, reason="needs CUDA (cudaHostRegister)")
def test_exact_pool_blocks_are_shared_across_fork_not_copied():
    """[F-fp8-FORKPIN] the C2 sampled replay forks evaluator children while the frozen pool is live.  Linux >= 5.9 copies DMA-pinned
    PRIVATE pages eagerly at fork (B8 attempt 2: 59,350 MiB per child, cgroup at the cap, 20 OOM-killed children); the pool's blocks are
    MAP_SHARED mappings, which fork shares -- measured on cp-h100 with a 2 GiB block: private-registered fork -> cgroup anon +2,653 MiB,
    shared-registered -> +336 (the child's RssShmem 2,048 = the shared pages).  Here: a child that reads a 1 GiB block adds far less
    than the block to the cgroup's anon, and sees the block as RssShmem, not RssAnon."""
    import os

    torch.zeros(1, device="cuda")
    n = 1 << 30
    pool = _PinnedPool(exact=True)
    v = pool.take(n)
    v.fill_(5)
    before = _cgroup_anon_shmem()
    r, w = os.pipe()
    sampled_r, sampled_w = os.pipe()
    pid = os.fork()
    if pid == 0:                                   # the child: read the block in pure Python (a torch op after fork deadlocks on OMP)
        try:
            os.close(sampled_w)
            mv = memoryview(v.numpy())
            s = sum(mv[::4096])
            st = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/self/status") if l.startswith(("RssAnon", "RssShmem"))}
            os.write(w, f"{s} {st['RssAnon']} {st['RssShmem']}".encode())
            os.read(sampled_r, 1)                 # stay alive until the parent has sampled the cgroup and closed its end
        finally:
            os._exit(0)
    os.close(w)
    os.close(sampled_r)
    msg = os.read(r, 200).decode().split()
    after = _cgroup_anon_shmem()
    os.close(sampled_w)
    os.waitpid(pid, 0)
    s, child_anon_kb, child_shmem_kb = int(msg[0]), int(msg[1]), int(msg[2])
    assert s == 5 * (n // 4096), "the child read the parent's block in place"
    assert child_shmem_kb >= (n >> 10) - 4096, f"the block is not a shared mapping in the child (RssShmem {child_shmem_kb} kB)"
    assert child_anon_kb < (n >> 10) // 2, f"the child holds a private copy (RssAnon {child_anon_kb} kB)"
    if before and after:
        assert after[0] - before[0] < n // 2, f"fork copied the block: cgroup anon +{(after[0] - before[0]) >> 20} MiB"
    del v
    pool.release()


@pytest.mark.skipif(not GPU, reason="needs CUDA")
def test_torch_mode_is_the_named_fallback():
    pool = _PinnedPool(exact=False)
    v = pool.take(1 << 20)
    assert pool.mode == "torch-pinned-pow2" and v.is_pinned()      # torch's own blocks are the ones its is_pinned() knows
    pool.release()


def test_pool_accounting_without_cuda_uses_torch_pinned_or_skips():
    if GPU:
        pytest.skip("covered above")
    pool = _PinnedPool(exact=True)
    with pytest.raises(Exception):
        pool.take(1 << 16)                        # pin_memory without CUDA raises in torch; the mode is never claimed
    assert pool.mode is None


# Growing the pool on a CPU build: `cudaHostRegister` is a stand-in that records, per call, its thread and how many pages of the range
# were already resident (mincore) -- the real call pins whatever is there and faults the rest in itself, one page at a time.
_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
_LIBC.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
MiB = 1 << 20


def _resident_pages(addr: int, n: int) -> int:
    vec = (ctypes.c_ubyte * (n // 4096))()
    assert _LIBC.mincore(addr, n, vec) == 0, ctypes.get_errno()
    return sum(v & 1 for v in vec)


class _FakeCudart:
    def __init__(self, events: list) -> None:
        self.events = events

    def cudaHostRegister(self, ptr: int, n: int, flags: int) -> int:
        self.events.append(("register", ptr, n, _resident_pages(ptr, n), threading.current_thread()))
        return 0

    def cudaHostUnregister(self, ptr: int) -> int:
        return 0


@pytest.fixture
def fake_cuda(monkeypatch):
    """(events, madvise calls): every registration and every populate piece, in order, with its thread."""
    events: list = []

    def madvise(addr: int, n: int, advice: int) -> int:
        events.append(("populate", addr, n, advice, threading.current_thread()))
        return _LIBC.madvise(addr, n, advice)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: _FakeCudart(events))
    monkeypatch.setattr(nh, "_POPULATE", {"ok": True, "madvise": madvise})
    monkeypatch.setattr(nh, "POPULATE_THREADS", 4)
    monkeypatch.setattr(nh, "_POPULATE_PIECE", MiB)
    pools: list = []
    yield events, pools
    for p in pools:
        p.release()
    if "pool" in nh._POPULATE:
        nh._POPULATE["pool"].shutdown()


def _of(events: list, kind: str) -> list:
    return [e for e in events if e[0] == kind]


def test_a_fresh_block_is_faulted_in_by_helper_threads_before_the_caller_registers_it(fake_cuda):
    events, pools = fake_cuda
    pools.append(pool := _PinnedPool(block_bytes=MiB, exact=True))
    v = pool.take(6 * MiB + 5)
    n = 6 * MiB + 4096
    (reg,) = _of(events, "register")
    assert reg[1:4] == (v.data_ptr(), n, n // 4096), "every page of the block is present when it is registered"
    assert reg[4] is threading.current_thread(), "only the caller makes the CUDA call"
    pieces = _of(events, "populate")
    assert all(e[3] == nh._MADV_POPULATE_WRITE and e[2] <= MiB for e in pieces)
    assert sorted((e[1], e[2]) for e in pieces) == [(v.data_ptr() + o, min(MiB, n - o)) for o in range(0, n, MiB)]
    assert all(e[4] is not threading.current_thread() and e[4].name.startswith("pinned-populate") for e in pieces), \
        "the helpers make no CUDA call and the caller no page fault"
    assert pool.mode == "exact-registered-shared" and set(pool.times) == {"populate_s", "register_s"}


def test_a_kernel_without_populate_write_leaves_the_faults_to_the_registration(fake_cuda, monkeypatch):
    events, pools = fake_cuda
    tried: list[int] = []

    def no_populate(addr: int, n: int, advice: int) -> int:
        tried.append(n)
        ctypes.set_errno(errno.EINVAL)
        return -1

    monkeypatch.setitem(nh._POPULATE, "madvise", no_populate)
    pools.append(pool := _PinnedPool(block_bytes=MiB, exact=True))
    pool.take(3 * MiB)
    pool.take(2 * MiB)
    assert nh._POPULATE["ok"] is False and len(tried) == 3, "one refused block, then no more attempts"
    assert [e[2:4] for e in _of(events, "register")] == [(3 * MiB, 0), (2 * MiB, 0)], "both blocks still registered"


def _warm(pool: _PinnedPool) -> None:
    """The state a priming finds: free blocks of mixed sizes and a partly used current block."""
    for n in (3 * MiB, MiB, 5 * MiB):
        pool.take(n)
    pool.reset()
    pool.take(512 << 10)


# fits the current block; first fit from the free list; new; new; a small one into the 1 MiB free block; new, page-rounded; new at block_bytes
SIZES = [2 * MiB, 2 * MiB + 100, 4 * MiB, 7 * MiB, 300, 6 * MiB + 1, MiB]


def test_reserve_makes_exactly_the_blocks_the_takes_after_it_allocate(fake_cuda, monkeypatch):
    events, pools = fake_cuda
    pools.extend([plain := _PinnedPool(block_bytes=MiB, exact=True), reserved := _PinnedPool(block_bytes=MiB, exact=True)])
    _warm(plain)
    _warm(reserved)
    before = plain.bytes_total
    plain_views = [plain.take(n) for n in SIZES]

    del events[:]
    made = reserved.reserve(SIZES)
    new = [4 * MiB, 7 * MiB, 6 * MiB + 4096, MiB]
    assert made == sum(new) == plain.bytes_total - before == reserved.reserved_bytes
    regs = _of(events, "register")
    assert [e[2] for e in regs] == new and all(e[3] == e[2] // 4096 for e in regs)
    first_reg = events.index(regs[0])
    assert all(e[0] == "populate" for e in events[:first_reg]) and not _of(events[first_reg:], "populate"), \
        "the burst's pages are faulted in together, across its blocks, before any registration"

    calls: list[int] = []
    real = nh._pinned_block
    monkeypatch.setattr(nh, "_pinned_block", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
    views = [reserved.take(n) for n in SIZES]
    assert calls == [], "the takes find every block ready"
    assert [int(b.numel()) for b in reserved.in_use] == [int(b.numel()) for b in plain.in_use]
    assert [v.numel() for v in views] == [v.numel() for v in plain_views] and reserved.off == plain.off
    assert (reserved.allocs, reserved.bytes_total, len(reserved.all)) == (plain.allocs, plain.bytes_total, len(plain.all))
    reserved.reset()
    assert reserved.reserve_unused == 0 and not reserved._ready


def test_reserved_blocks_no_take_asked_for_join_the_free_list_and_are_counted(fake_cuda):
    _events, pools = fake_cuda
    pools.append(pool := _PinnedPool(block_bytes=MiB, exact=True))
    assert pool.reserve([2 * MiB, 3 * MiB]) == 5 * MiB
    pool.take(2 * MiB)
    pool.reset()
    assert pool.reserve_unused == 1 and sorted(int(b.numel()) for b in pool.free) == [2 * MiB, 3 * MiB]
    assert pool.reserve([MiB]) == 0, "the free list covers it: nothing new"


def test_reserve_is_a_no_op_off_the_exact_path(fake_cuda, monkeypatch):
    _events, pools = fake_cuda
    assert _PinnedPool(exact=False).reserve([MiB]) == 0
    monkeypatch.setattr(nh, "POPULATE_THREADS", 0)
    pools.append(pool := _PinnedPool(block_bytes=MiB, exact=True))
    assert pool.reserve([MiB]) == 0 and pool.all == []


@pytest.mark.parametrize("j", [0, 1, 3, 7, 12])
def test_the_primed_level_sizes_are_begin_trees(j):
    from verity_vllm.commit.committer.native_collect import _window_level_sizes

    for n_total in [1, 2, 3, 5, 8, 127, 128, 129, 4097, 1 << 15]:
        height, m = 0, n_total
        while m > 1:
            m = (m + 1) // 2
            height += 1
        assert _window_level_sizes(n_total, j) == [-(-n_total // (1 << L)) for L in range(min(j, height) + 1)]
