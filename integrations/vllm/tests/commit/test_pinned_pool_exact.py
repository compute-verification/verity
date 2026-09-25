"""[R17 fp8, F-fp8-POW2] The collector's pinned pool holds EXACTLY what it accounts for.

torch's CachingHostAllocator rounds every pinned request up to the next power of two and never returns freed blocks to the OS (torch
2.13.0+cu129 on cp-h100: 3 GiB -> RSS +4 GiB; 5 GiB + 7 B -> +8 GiB); pinned pages are committed at allocation, so the process RSS and the
cgroup carried up to 2x the pool's `bytes_total` -- 21,152 MiB of the B8 FP8 row's run-phase RSS outside every term of the host bound
(FP8-STAGING §6).  Exact mode (default) registers page-aligned anonymous memory instead: RSS == bytes_total, released on `release()`.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

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
    # torch's `is_pinned()` only knows ITS allocator's blocks (False here) -- page-locking is the driver's: an async D2H copy into the
    # block issues without blocking (a pageable destination makes copy_ synchronous: 1.5 GiB at ~1.6 GB/s ~ 1 s) and lands at PCIe speed
    src = torch.arange(0, 4096, dtype=torch.uint8, device="cuda").repeat(n // 4096 + 1)[:n]
    torch.cuda.synchronize()
    import time

    t0 = time.perf_counter()
    v.copy_(src, non_blocking=True)
    issue_s = time.perf_counter() - t0
    torch.cuda.synchronize()
    total_s = time.perf_counter() - t0
    assert issue_s < 0.05, f"copy_ blocked for {issue_s:.3f}s: the destination is not page-locked"
    assert total_s < 0.5, f"{n / total_s / 1e9:.1f} GB/s: pageable-speed D2H"
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
    import time

    torch.zeros(1, device="cuda")
    n = 1 << 30
    pool = _PinnedPool(exact=True)
    v = pool.take(n)
    v.fill_(5)
    before = _cgroup_anon_shmem()
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:                                   # the child: read the block in pure Python (a torch op after fork deadlocks on OMP)
        try:
            mv = memoryview(v.numpy())
            s = sum(mv[::4096])
            st = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/self/status") if l.startswith(("RssAnon", "RssShmem"))}
            os.write(w, f"{s} {st['RssAnon']} {st['RssShmem']}".encode())
            time.sleep(1.0)                       # stay alive while the parent samples the cgroup
        finally:
            os._exit(0)
    os.close(w)
    msg = os.read(r, 200).decode().split()
    after = _cgroup_anon_shmem()
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
