"""R10 acq-native regression tests for the native collector (arena reuse, plan hit/miss, gap zeroing, byte layout, gather path).

Needs a CUDA device and nvcc (JIT build via torch.utils.cpp_extension); skipped otherwise.  The byte layout under test is CMT-3's
(`NativeHostCommitter`, `ALIGN = 4`): 4-B aligned offsets, gaps zero-filled, tensors laid out in acquisition order -- the candidate must reproduce it
bit-for-bit (same-layout root equality is the correctness gate of the lane).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA device required", allow_module_level=True)

from verity_vllm.acquire.native_collect import ext  # noqa: E402


@pytest.fixture(scope="module")
def mod():
    return ext(verbose=False)


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().reshape(-1).view(torch.uint8)


def _reference_layout(tensors):
    """CMT-3's rule (`NativeHostCommitter.ALIGN == 4`): off = align4(cursor); gap zero-filled; then the tensor's bytes."""
    off, out = 0, []
    for t in tensors:
        nb = t.numel() * t.element_size()
        aligned = (off + 3) & ~3
        out.append((aligned, aligned - off, nb))
        off = aligned + nb
    return out, off


def _run_step(c, arena, tensors, key, *, gather=False):
    c.begin_step(1, arena.data_ptr(), arena.numel(), key)
    fn = c.stage if gather else c.collect
    for i, t in enumerate(tensors):
        fn(i, t)
    if gather:
        c.flush()
    return c.end_step()


@pytest.mark.parametrize("gather", [False, True])
def test_layout_matches_reference_and_gaps_are_zero(mod, gather):
    c = mod.Collector()
    if gather:
        c.enable_gather(256)
    arena = torch.full((1 << 16,), 0xAB, dtype=torch.uint8, device="cuda")  # dirty arena: gaps must be re-zeroed
    ts = [torch.arange(5, device="cuda", dtype=torch.int8), torch.randn(3, 7, device="cuda", dtype=torch.bfloat16),
          torch.randn(2, 8, 3, device="cuda", dtype=torch.float32)[:, :, 0], torch.arange(9, device="cuda", dtype=torch.int64)]
    used, n, changed, version = _run_step(c, arena, ts, key=1, gather=gather)
    torch.cuda.synchronize()
    ref, total = _reference_layout(ts)
    assert (used, n, changed, version) == (total, len(ts), True, 1)
    for t, (off, gap, nb) in zip(ts, ref):
        if gap:
            assert int(arena[off - gap:off].max()) == 0, "alignment gap not zero-filled"
        assert torch.equal(arena[off:off + nb], _as_bytes(t))
    plan = c.export_plan(1)
    assert [(e[4], e[3]) for e in plan] == [(o, nb) for o, _, nb in ref]      # (name_id, dtype, sizes, nbytes, off, gap)


def test_plan_hit_reuses_offsets_and_detects_shape_change(mod):
    c = mod.Collector()
    arena = torch.zeros(1 << 16, dtype=torch.uint8, device="cuda")
    ts = [torch.randn(4, 4, device="cuda"), torch.randn(6, device="cuda", dtype=torch.float16)]
    _run_step(c, arena, ts, key=3)
    used, n, changed, version = _run_step(c, arena, ts, key=3)      # hit: same shapes -> no plan change
    assert (changed, version, n) == (False, 1, 2)
    st = c.stats()
    assert st["misses"] == 0 and st["copies"] == 4
    ts2 = [torch.randn(4, 4, device="cuda"), torch.randn(7, device="cuda", dtype=torch.float16)]   # shape drift under the same key
    used2, n2, changed2, version2 = _run_step(c, arena, ts2, key=3)
    assert changed2 and version2 == 2 and c.stats()["misses"] >= 1
    torch.cuda.synchronize()
    ref, total = _reference_layout(ts2)
    assert used2 == total
    for t, (off, _, nb) in zip(ts2, ref):
        assert torch.equal(arena[off:off + nb], _as_bytes(t))


def test_arena_reuse_regression(mod):
    """Arena-reuse safety (kept from CMT-3): a step's bytes must be complete in the arena at the step's end event and must not be
    disturbed by the NEXT step writing into a different (or the same, after release) arena block."""
    c = mod.Collector()
    arena = torch.zeros(1 << 16, dtype=torch.uint8, device="cuda")
    ts = [torch.randn(64, device="cuda"), torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)]
    _run_step(c, arena, ts, key=5)
    ev = torch.cuda.Event()
    ev.record()
    snapshot = arena.clone()                                          # what the hasher would see after the step event
    ts_next = [torch.randn(64, device="cuda"), torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)]
    _run_step(c, arena, ts_next, key=5)                               # reuse the SAME block (worker released it)
    torch.cuda.synchronize()
    ref, _ = _reference_layout(ts)
    for t, (off, _, nb) in zip(ts, ref):
        assert torch.equal(snapshot[off:off + nb], _as_bytes(t))     # first step's bytes were intact at its event
    for t, (off, _, nb) in zip(ts_next, ref):
        assert torch.equal(arena[off:off + nb], _as_bytes(t))        # second step overwrote in order


def test_overflow_is_reported_not_silent(mod):
    c = mod.Collector()
    arena = torch.zeros(64, dtype=torch.uint8, device="cuda")
    c.begin_step(1, arena.data_ptr(), arena.numel(), 9)
    with pytest.raises(RuntimeError):
        c.collect(0, torch.randn(1024, device="cuda"))


def test_unflushed_stage_is_an_error(mod):
    c = mod.Collector()
    c.enable_gather(16)
    arena = torch.zeros(1 << 12, dtype=torch.uint8, device="cuda")
    c.begin_step(1, arena.data_ptr(), arena.numel(), 11)
    c.stage(0, torch.randn(8, device="cuda"))
    with pytest.raises(RuntimeError):
        c.end_step()


def test_late_read_after_inplace_overwrite_binds_different_bytes(mod):
    """Late-read regression (R10 acq-life `lifetimes.md`: qkv_proj out is rotated IN PLACE by RoPE at hook+1; o_proj / down_proj outputs
    are overwritten by the fused norm at hook+1 -- zero-hook windows).  Staging the value and flushing only AFTER the in-place consumer
    binds the WRONG bytes; flushing before it (the P0/P1/P2 placement) binds the produced value.  Engine-level twin: `VERITY_FAULT=late_flush`
    (qkv_proj flush dropped -> read at o_proj, after RoPE) must yield a run root different from arm 2."""
    c = mod.Collector()
    c.enable_gather(64)
    arena = torch.zeros(1 << 12, dtype=torch.uint8, device="cuda")
    t = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    produced = t.clone()
    c.begin_step(plan_key=1, arena_ptr=arena.data_ptr(), arena_bytes=arena.numel(), use_plan=True)
    c.stage(0, t)
    c.flush()                                   # correct placement: enqueued BEFORE the destructive consumer on the same stream
    t.mul_(-1.0)                                # the destructive consumer (RoPE / fused norm stand-in)
    c.end_step()
    torch.cuda.synchronize()
    assert torch.equal(arena[:512], _as_bytes(produced))
    t2 = produced.clone()
    c.begin_step(plan_key=2, arena_ptr=arena.data_ptr(), arena_bytes=arena.numel(), use_plan=True)
    c.stage(0, t2)
    t2.mul_(-1.0)
    c.flush()                                   # LATE read: after the consumer
    c.end_step()
    torch.cuda.synchronize()
    assert not torch.equal(arena[:512], _as_bytes(produced)), "a late read must NOT reproduce the produced value"
    assert torch.equal(arena[:512], _as_bytes(t2))


def test_native_hook_body_collects_like_the_python_wrapper(mod):
    """R11 v2: `HookFwd` (C++ forward wrapper) -- inactive collector: pass-through; active: every tensor of the output (tuple slots by
    position), the first positional input when `nid_in >= 0`, then a flush when the module is a collection point."""
    c = mod.Collector()
    c.enable_gather(64)
    c.use_gather = True
    arena = torch.zeros(1 << 14, dtype=torch.uint8, device="cuda")

    class Pair(torch.nn.Module):
        def forward(self, x):
            return x * 2, x + 1

    class Single(torch.nn.Module):
        def forward(self, x, scale=1.0):
            return x * scale

    p, s = Pair().cuda(), Single().cuda()
    x = torch.randn(8, device="cuda")
    p.forward = mod.HookFwd(c, p.forward, 0, tuple(range(1, 9)), -1, 9, True)      # tuple output, flush point
    s.forward = mod.HookFwd(c, s.forward, 10, tuple(range(11, 19)), 20, 21, False)  # input acquired first (embedding-style pre hook)
    out = p(x)
    assert isinstance(out, tuple) and c.stats()["staged"] == 0                        # inactive: nothing collected
    c.begin_step(plan_key=1, arena_ptr=arena.data_ptr(), arena_bytes=arena.numel(), use_plan=True)
    y = s(x, scale=3.0)
    out = p(y)
    c.flush()
    c.end_step()
    torch.cuda.synchronize()
    st = c.stats()
    assert st["staged"] == 4 and st["flushes"] == 1                                 # Pair's flush point took everything; the step-end flush was empty
    plan = c.export_plan(1)
    assert [e[0] for e in plan] == [20, 10, 1, 2]                                    # in, out(Single), slot 0, slot 1 (Pair)
    ref, _ = _reference_layout([x, y, out[0], out[1]])
    for t, (off, _, nb) in zip([x, y, out[0], out[1]], ref):
        assert torch.equal(arena[off:off + nb], _as_bytes(t))
    assert c.drain_errors() == []
    del p.forward, s.forward
    assert isinstance(p(x), tuple)
