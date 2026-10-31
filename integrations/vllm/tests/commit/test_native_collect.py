"""Regression tests for the native collector (arena reuse, plan hit/miss, gap zeroing, byte layout, gather path).

Needs a CUDA device and nvcc (JIT build via torch.utils.cpp_extension); skipped otherwise.  The byte layout under test is
`NativeHostCommitter`'s (`ALIGN = 4`): 4-B aligned offsets, gaps zero-filled, tensors laid out in acquisition order -- the candidate must reproduce it
bit-for-bit (same-layout root equality is the correctness gate).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA device required", allow_module_level=True)

from verity_vllm.commit.committer.native_collect import ext  # noqa: E402


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


@pytest.mark.parametrize("bpd,auto", [(1, False), (4, False), (32, False), (32, True), (1024, True)])
def test_gather_writes_the_reference_layout_at_every_alignment_and_grid(mod, bpd, auto):
    """The gather kernel's paths (16 B words between byte heads and tails when source and destination share their offset mod 16, 16 B
    words joined from two source loads when both are 4 B aligned at other offsets, pitched rows by 16 B or 4 B words, bytes otherwise)
    write exactly the reference layout's bytes: tensors of every length and source offset (slices of one buffer), three dtypes, pitched
    views, small ones and ones that take many grid-stride rounds, every (source, destination) offset pair mod 16 of the 4 B aligned
    ones, under any grid."""
    c = mod.Collector()
    c.enable_gather(256)
    c.blocks_per_desc, c.auto_blocks = bpd, auto
    g = torch.Generator(device="cuda").manual_seed(bpd * 2 + auto)
    base = torch.randint(-128, 128, (1 << 21,), dtype=torch.int8, device="cuda", generator=g)
    f32, bf16 = base.view(torch.float32), base.view(torch.bfloat16)
    ts = [base[off:off + nb] for off, nb in [(0, 1), (1, 7), (0, 16), (4, 36), (6, 70), (3, 255), (8, 4096), (12, 4100), (2, 4098),
                                             (0, 300_001), (5, 262_147), (4, 524_292), (16, 65_536)]]
    ts += [f32[1:1001], f32[3:70_003], bf16[3:2000], bf16[1:40_001], f32[:16_384].view(128, 128)[:, 3:67],
           f32[:16_384].view(128, 128)[:, 4:68], bf16[:16_384].view(128, 128)[:, 1:4], bf16[:65_536].view(256, 256)[:, 2:200]]
    ts += [f32[k // 4:k // 4 + 4097] for k in range(16)]           # each moves the next destination by 4 B: 4 sources x 4 destinations
    arena = torch.full((4 << 20,), 0xAB, dtype=torch.uint8, device="cuda")
    used, n, _changed, _version = _run_step(c, arena, ts, key=1, gather=True)
    torch.cuda.synchronize()
    ref, total = _reference_layout(ts)
    assert (used, n) == (total, len(ts))
    pairs = {(t.data_ptr() % 16, (arena.data_ptr() + off) % 16) for t, (off, _g, nb) in zip(ts, ref)
             if t.is_contiguous() and t.data_ptr() % 4 == 0 and off % 4 == 0 and nb >= 64}
    assert pairs == {(s, d) for s in range(0, 16, 4) for d in range(0, 16, 4)}
    for k, (t, (off, gap, nb)) in enumerate(zip(ts, ref)):
        if gap:
            assert int(arena[off - gap:off].max()) == 0, f"tensor {k}: alignment gap not zero-filled"
        assert torch.equal(arena[off:off + nb], _as_bytes(t)), f"tensor {k} (src offset {t.data_ptr() % 16}, dst offset {off % 16}, {nb} B)"


@pytest.mark.parametrize(("gather", "fast"), [(False, True), (True, False), (True, True)], ids=["memcpy", "gather", "gather-fast"])
def test_slot_plan_stages_what_one_acquisition_per_view_stages(mod, gather, fast):
    """`stage_slot_plan` (a replayed graph's slots in one call: whole, first n rows, first pad rows, a row count past the slot clamped
    as `slot[:n]` is, no rows at all) writes the bytes and plan entries, and counts the expected tensors, of one acquisition per view in
    the same order, on plan misses and hits alike, with the slots staged from their pointers (`slot_fast`) or as views; it refuses an
    unknown handle and an unknown kind."""
    g = torch.Generator(device="cuda").manual_seed(11)
    rows_max = 64

    def rnd(*shape, dtype):
        return torch.randint(-100, 100, shape, generator=g, device="cuda").to(dtype)

    slots = [rnd(rows_max, 96, dtype=torch.bfloat16), rnd(rows_max, 33, dtype=torch.int8), rnd(rows_max, dtype=torch.int64),
             rnd(5, 7, dtype=torch.float32), rnd(rows_max, 3, 5, dtype=torch.float16), rnd(1, dtype=torch.int32), rnd(16, 8, dtype=torch.float32)]
    kinds = [1, 1, 2, 0, 1, 0, 1]
    one, batch = mod.Collector(), mod.Collector()
    for c in (one, batch):
        if gather:
            c.enable_gather(256)
        c.use_gather = gather
    batch.slot_fast = fast
    h = batch.add_slot_plan(list(range(len(slots))), slots, kinds)
    a_one = torch.full((1 << 20,), 0xAB, dtype=torch.uint8, device="cuda")
    a_batch = a_one.clone()
    steps = ((37, 40), (37, 40), (12, 16), (0, 8))
    for n, pad in steps:
        views = [s if k == 0 else s[:n] if k == 1 else s[:pad] for s, k in zip(slots, kinds)]
        one.begin_step(1, a_one.data_ptr(), a_one.numel(), True)
        for i, v in enumerate(views):
            one.note_expected(1)
            (one.stage if gather else one.collect)(i, v)
        if gather:
            one.flush()
        r_one = one.end_step()
        batch.begin_step(1, a_batch.data_ptr(), a_batch.numel(), True)
        assert batch.stage_slot_plan(h, n, pad) == len(slots)
        if gather:
            batch.flush()
        r_batch = batch.end_step()
        torch.cuda.synchronize()
        views = [v for v in views if v.numel()]
        ref, total = _reference_layout(views)
        assert r_batch == r_one and r_one[:2] == (total, len(views))
        assert torch.equal(a_batch, a_one)
        for v, (off, _gap, nb) in zip(views, ref):
            assert torch.equal(a_batch[off:off + nb], _as_bytes(v))
        assert batch.export_plan(1) == one.export_plan(1)
    st_one, st_batch = one.stats(), batch.stats()
    assert st_batch["expected"] == st_one["expected"] == len(steps) * len(slots) and st_batch["committed"] == len(steps) * len(slots)
    with pytest.raises(RuntimeError, match="unknown handle"):
        batch.stage_slot_plan(h + 1, 1, 1)
    with pytest.raises(RuntimeError, match="kind must be"):
        batch.add_slot_plan([0], [slots[0]], [3])


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
    """Late-read regression (qkv_proj out is rotated IN PLACE by RoPE at hook+1; o_proj / down_proj outputs
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
    """`HookFwd` (C++ forward wrapper) -- inactive collector: pass-through; active: every tensor of the output (tuple slots by
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


def test_end_run_keeps_the_plans_the_ended_run_used_and_a_dropped_plan_is_learned_again_at_the_same_offsets(mod):
    """`Collector.end_run` (native_plans.h through the committer's reset): the plan of the run that ended is followed by the next
    one across commit.py's second reset; a plan a whole run did not use is dropped, pending until the next step frees it, and when
    its key recurs it is learned again at the same offsets, under a version above the dropped one."""
    c = mod.Collector()
    arena = torch.zeros(1 << 16, dtype=torch.uint8, device="cuda")
    ts = [torch.randn(4, 4, device="cuda"), torch.randn(7, device="cuda", dtype=torch.float16), torch.arange(5, device="cuda", dtype=torch.int8)]

    def step(key):
        c.begin_step(plan_key=key, arena_ptr=arena.data_ptr(), arena_bytes=arena.numel(), use_plan=True)
        for i, t in enumerate(ts):
            c.collect(i, t)
        return c.end_step()

    assert step(3)[2:] == (True, 1)
    learned = c.export_plan(3)
    c.end_run()
    c.end_run()
    assert step(3)[2:] == (False, 1) and c.stats()["plans"] == 1
    c.end_run()
    step(4)
    c.end_run()
    assert c.plan_version(3) == -1 and c.export_plan(3) == [] and (c.stats()["plans"], c.stats()["plan_entries"]) == (1, len(ts))
    st = c.stats()
    assert (st["plans_pending"], st["plan_entries_pending"], st["plans_dropped"], st["plans_pending_emptied_step"]) == (1, len(ts), 1, -1)
    used, n, changed, version = step(3)
    torch.cuda.synchronize()
    assert changed and version > 1 and c.export_plan(3) == learned
    st = c.stats()
    assert (st["plans_pending"], st["plan_entries_pending"], st["plans_dropped"], st["plans_pending_emptied_step"]) == (0, 0, 1, 1)
    for t, (off, _, nb) in zip(ts, _reference_layout(ts)[0]):
        assert torch.equal(arena[off:off + nb], _as_bytes(t))
