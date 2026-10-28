"""The binder's pieces on the CPU, against fake vLLM objects: the KV page layout read off a cache tensor's strides,
weights and KV registration from a runner, the scheduler's appends, FIFO reuse of freed pages, and a step's pass."""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.protocols.poub.binder import (  # noqa: E402
    Binder, Every, Inbox, PagedBacking, TorchBacking, page_layout, storage_bytes,
)
from verity_vllm.protocols.poub.interface import Append  # noqa: E402
from verity_vllm.protocols.poub.registry import Store  # noqa: E402

PAGES, SLOTS, HEADS, DIM = 6, 16, 2, 32          # a slot is 2 x 32 bf16 = 128 bytes; a plane-page 2 KiB


def test_page_layout_of_flash_attention_and_flashinfer_caches():
    fa = torch.zeros(2, PAGES, SLOTS, HEADS, DIM, dtype=torch.bfloat16)
    lay = page_layout(fa, PAGES, SLOTS)
    assert (lay.slot_bytes, lay.strides(), lay.slot_major) == (128, (2048, PAGES * 2048), True)
    fi = torch.zeros(PAGES, 2, SLOTS, HEADS, DIM, dtype=torch.bfloat16)
    assert page_layout(fi, PAGES, SLOTS).strides() == (4096, 2048)
    hnd = torch.zeros(PAGES, 2, HEADS, SLOTS, DIM, dtype=torch.bfloat16).permute(0, 1, 3, 2, 4)
    assert not page_layout(hnd, PAGES, SLOTS).slot_major
    kv = torch.zeros(PAGES, SLOTS, HEADS, 2 * DIM, dtype=torch.bfloat16).permute(0, 2, 1, 3)
    lay = page_layout(kv, PAGES, SLOTS)
    assert (lay.planes, lay.slot_bytes, lay.strides(), lay.slot_major) == (1, 256, (4096, 4096), True)
    with pytest.raises(ValueError):
        page_layout(torch.zeros(PAGES, SLOTS, HEADS, 2 * DIM)[..., :DIM], PAGES, SLOTS)
    with pytest.raises(ValueError):
        page_layout(torch.zeros(PAGES, SLOTS, SLOTS, DIM), PAGES, SLOTS)


def test_torch_backing_reads_and_writes_bytes():
    t = torch.arange(1024, dtype=torch.int32)
    b = TorchBacking.of_tensor(t, 1024)
    assert b.nbytes == 4096 and b.read(4, 4) == (1).to_bytes(4, "little")
    b.write(0, b"\xff" * 4)
    assert int(t[0]) == -1
    assert TorchBacking.of_tensor(torch.zeros(10), 4096) is None


def test_a_transposed_weight_is_backed_in_storage_order_and_a_strided_one_is_not():
    base = torch.arange(64 * 32, dtype=torch.int16).reshape(64, 32)
    b = TorchBacking.of_tensor(base[16:].t(), 1024)
    assert b.nbytes == 3 * 1024 and b.read(0, b.nbytes) == base[16:].numpy().tobytes()
    assert TorchBacking.of_tensor(base[:, :16], 1024) is None


class Block:
    def __init__(self, i: int) -> None:
        self.block_id, self.ref_cnt, self.is_null = i, 0, i == 0


class Queue:
    def __init__(self, blocks) -> None:
        self.q = deque(blocks)

    def remove(self, b) -> None:
        self.q.remove(b)

    def append_n(self, bs) -> None:
        self.q.extend(bs)


def runner(shared: bool = False, interleaved: bool = False):
    """Three layers' caches: each its own tensor, or views of one storage; ``interleaved``: vLLM 0.28's FlashAttention
    layers, contiguous in one storage, each (pages, heads, slots, 2 * DIM) over a slot-major page with K and V side by
    side per head."""
    model = torch.nn.Sequential(torch.nn.Linear(64, 32, bias=False), torch.nn.LayerNorm(32))
    if interleaved:
        whole = torch.zeros(3, PAGES, SLOTS, HEADS, 2 * DIM, dtype=torch.bfloat16)
        caches = [whole[i].permute(0, 2, 1, 3) for i in range(3)]
    elif shared:
        whole = torch.zeros(PAGES, 3, 2, SLOTS, HEADS, DIM, dtype=torch.bfloat16)
        caches = [whole[:, i] for i in range(3)]
    else:
        caches = [torch.zeros(2, PAGES, SLOTS, HEADS, DIM, dtype=torch.bfloat16) for _ in range(3)]
    return SimpleNamespace(model=model, kv_caches=caches, cache_config=SimpleNamespace(block_size=SLOTS),
                           _kernel_block_sizes=[SLOTS])


def test_a_layer_of_a_shared_storage_reads_as_its_own_pages():
    r = runner(shared=True)
    t = r.kv_caches[1]
    t[2, 1, 0, 0, 0] = 7.0
    b = PagedBacking(storage_bytes(t), t.storage_offset() * 2, page_layout(t, PAGES, SLOTS))
    assert b.nbytes == PAGES * 2 * 2048
    k = (2 * 2 + 1) * 2048
    assert b.read(k, 2) == torch.tensor([7.0], dtype=torch.bfloat16).view(torch.uint8).numpy().tobytes()
    b.write(k + 2046, b"\x01\x02\x03\x04")
    assert t[2, 1, -1, -1, -1:].view(torch.uint8).tolist() == [1, 2] and t[3, 0, 0, 0, :1].view(torch.uint8).tolist() == [3, 4]


@pytest.mark.parametrize("shared,interleaved", [(False, False), (True, False), (False, True)])
def test_registration_from_a_runner(shared, interleaved):
    s = Store()
    b = Binder(s, block_bytes=1024)
    r = runner(shared, interleaved)
    b.on_load(r)
    b.on_kv(r, SimpleNamespace(num_blocks=PAGES))
    st = b.stats()
    assert st["bytes_by_kind"] == {"weights": 64 * 32 * 4, "kv": 3 * 2 * PAGES * 2048}
    assert st["unregistered"] == 2 and st["unregistered_bytes"] == 2 * 32 * 4
    assert [x.name for x in s.layout().regions][:1] == ["0.weight"]
    assert s.pool.ids.shape == (PAGES, 4) and s.pool.ids[1].tolist() == [4, 5, 6, 7]


def test_a_registered_tensor_moved_or_reshaped_since_registration_is_named():
    """`Binder.moved`: an in-place reader holds each registered tensor's device address, so a parameter whose data
    is reallocated, or a KV cache the runner rebinds, is named; one left where it was is not."""
    s, r = Store(), runner()
    b = Binder(s, block_bytes=1024)
    b.on_load(r)
    b.on_kv(r, SimpleNamespace(num_blocks=PAGES))
    assert b.moved() == []
    w = r.model[0].weight
    w.data = w.data.clone()
    r.kv_caches[2] = r.kv_caches[2].clone()
    assert b.moved() == ["0.weight", "kv[2]"]


@pytest.mark.parametrize("budget, registered", [(8, 64 * 32 * 4), (7, 0)])
def test_a_weight_block_budget_registers_the_tensors_that_fit_and_counts_the_rest(budget, registered):
    s = Store()
    b = Binder(s, block_bytes=1024, weight_blocks=budget)
    r = runner()
    b.on_load(r)
    b.on_kv(r, SimpleNamespace(num_blocks=PAGES))
    st = b.stats()
    assert st["bytes_by_kind"].get("weights", 0) == registered and st["bytes_by_kind"]["kv"] == 3 * 2 * PAGES * 2048
    assert st["unregistered_bytes"] == 2 * 32 * 4 + 64 * 32 * 4 - registered


def test_with_kv_off_the_registered_store_is_the_weights_and_filler_and_no_step_reaches_a_kv_page():
    """The Commit's session (`Binder(kv=False)`): the KV pool is counted unregistered, the filler is its own region,
    every region is under the driver's sketch-sha512 receipt, no registered byte is a KV cache's, and a step that
    writes KV (its appends, its frees) writes no registered block: the memory map numbers the step and records no
    line."""
    s = Store()
    b = Binder(s, passes=False, kv=False)
    r = runner()
    b.on_load(r)
    b.on_kv(r, SimpleNamespace(num_blocks=PAGES))
    b.register_filler(3 * 4096 + 5, "cpu")
    st = b.stats()
    assert [x.kind for x in s.layout().regions] == ["weights", "filler"] and s._pool is None
    assert [x.receipt for x in s.layout().regions] == ["sketch-sha512"] * 2
    assert st["bytes_by_kind"] == {"weights": 64 * 32 * 4, "filler": 3 * 4096}
    assert st["unregistered_bytes"] == 2 * 32 * 4 + 3 * 2 * PAGES * 2048 == 2 * 32 * 4 + b.unregistered["kv"]
    kv = [(c.untyped_storage().data_ptr(), c.untyped_storage().data_ptr() + c.untyped_storage().nbytes())
          for c in r.kv_caches]
    for x in s.layout().regions:
        flat = s.backing(x.name).flat
        lo, hi = flat.data_ptr(), flat.data_ptr() + flat.numel()
        assert all(hi <= a or z <= lo for a, z in kv), f"{x.name} holds KV cache bytes"
    with pytest.raises(RuntimeError, match="filler is registered"):
        b.register_filler(4096, "cpu")
    m = s.record_memory_map(registration=bytes(range(64)))
    page = Block(1)
    page.ref_cnt = 1
    stm = SimpleNamespace(req_to_blocks={"a": [page]})
    sched = SimpleNamespace(kv_cache_manager=SimpleNamespace(coordinator=SimpleNamespace(single_type_managers=[stm])),
                            requests={"a": SimpleNamespace(num_computed_tokens=SLOTS)})
    b.on_schedule(sched, SimpleNamespace(num_scheduled_tokens={"a": SLOTS}))
    r.kv_caches[0][:, 1] = 1.5
    b.end()
    page.ref_cnt = 0
    b.on_free(SimpleNamespace(free_block_queue=Queue([Block(2), page])), [page])
    b.on_schedule(sched, SimpleNamespace(num_scheduled_tokens={}))
    b.end()
    assert m.last == 1 and memory_map_lines(m) == [] and b.step == 2
    with pytest.raises(RuntimeError, match="no KV pool"):
        s.kv_append([Append(1, 0, 1)], 2)


def memory_map_lines(m) -> list:
    from verity.core.protocols.poub import memory_map
    return list(memory_map.parse(m.text().splitlines(keepends=True))[1])


def test_steps_append_the_scheduled_slots_and_reuse_freed_pages_fifo():
    s = Store()
    b = Binder(s, block_bytes=1024)
    r = runner()
    b.on_kv(r, SimpleNamespace(num_blocks=PAGES))
    blocks = [Block(i) for i in range(PAGES)]
    pool = SimpleNamespace(free_block_queue=Queue(blocks[1:]))
    a, c = blocks[1:3], blocks[3:4]
    for x in (*a, *c):
        x.ref_cnt = 1
    s.kv_alloc([1, 2, 3])
    stm = SimpleNamespace(req_to_blocks={"a": a, "c": c})
    reqs = {"a": SimpleNamespace(num_computed_tokens=20), "c": SimpleNamespace(num_computed_tokens=3)}
    sched = SimpleNamespace(kv_cache_manager=SimpleNamespace(coordinator=SimpleNamespace(single_type_managers=[stm])),
                            requests=reqs)
    so = SimpleNamespace(num_scheduled_tokens={"a": 20, "c": 3})
    assert b.appends(sched, so) == [Append(1, 0, 16), Append(2, 0, 4), Append(3, 0, 3)]
    b.on_schedule(sched, so)
    assert s.in_pass and s._open_pass.blocks[0].tolist() == [4, 5, 6, 7]
    b.end()
    assert not s.in_pass and b.step == 1
    reqs["a"].num_computed_tokens = 21
    b.on_schedule(sched, SimpleNamespace(num_scheduled_tokens={"a": 1}))
    b.on_schedule(sched, SimpleNamespace(num_scheduled_tokens={}))
    assert s.counters.calls["end_pass"] == 2
    for x in a:
        x.ref_cnt = 0
    pool.free_block_queue.q = deque([blocks[4], blocks[5], *a])
    b.on_free(pool, a)
    assert [x.block_id for x in pool.free_block_queue.q] == [4, 5, 1, 2]
    assert not s.pool.allocated[[1, 2]].any() and s.pool.allocated[3]


EMPTY = SimpleNamespace(kv_cache_manager=SimpleNamespace(coordinator=SimpleNamespace(
    single_type_managers=[SimpleNamespace(req_to_blocks={})])), requests={})


def test_steps_run_in_chained_windows_with_the_readers_roots():
    from verity_vllm.protocols.poub.registry import chain
    s, recs = Store(), []
    sigma = [bytes([t]) * 32 for t in range(3)]
    b = Binder(s, block_bytes=1024, messages=Every(2, sigma.__getitem__, windows=3), on_window=recs.append,
               reader=lambda segments: b"\x01" * 64)
    r = runner()
    b.on_load(r)
    b.on_kv(r, SimpleNamespace(num_blocks=PAGES))
    for _ in range(8):
        b.on_schedule(EMPTY, SimpleNamespace(num_scheduled_tokens={}))
    assert [len(w.passes) for w in recs] == [2, 2, 2] and [w.challenge for w in recs] == sigma
    assert recs[1].passes[0].seed == chain(sigma[1], recs[0].passes[-1].root, 0)
    assert (recs[0].passes[0].root, recs[0].passes[0].segment_roots) == (b"\x01" * 64, ()) and b.stats()["windows"] == 3
    assert s._open_pass.seed is None
    b.end_window()
    assert len(recs) == 3 and not s.in_pass


def test_windows_end_on_the_verifiers_messages_and_the_first_chains_from_the_registration():
    """The verifier's message ends the window at the next pass's start, however many passes it held; σ_0's window
    chains from the registration digest the binder was given."""
    from verity.core.protocols.poub import receipts as rb
    s, recs, inbox = Store(), [], Inbox()
    digest = bytes(range(64))
    b = Binder(s, block_bytes=1024, messages=inbox, registration=digest, on_window=recs.append,
               reader=lambda segments: b"\x02" * 64)
    r = runner()
    b.on_load(r)
    b.on_kv(r, SimpleNamespace(num_blocks=PAGES))
    step = lambda: b.on_schedule(EMPTY, SimpleNamespace(num_scheduled_tokens={}))
    step()
    assert s._open_pass.seed is None
    sigma = [bytes([t + 7]) * 32 for t in range(2)]
    inbox.put(sigma[0])
    for _ in range(3):
        step()
    inbox.put(sigma[1])
    step()
    inbox.put(None)
    step()
    b.end()
    assert [len(w.passes) for w in recs] == [3, 1] and [w.challenge for w in recs] == sigma
    assert recs[0].passes[0].seed == rb.seed(sigma[0], 0, digest)
    assert recs[1].passes[0].seed == rb.seed(sigma[1], 0, b"\x02" * 64) and b.windows == 2


@pytest.mark.parametrize("passes", [True, False])
def test_a_steps_memory_map_digests_are_of_the_kv_its_forward_wrote(passes):
    """The binder appends at `Scheduler.schedule`, before the step's forward writes its KV; the memory map takes the
    step's digests at `end` (`Scheduler.update_from_output`), from the bytes the forward wrote.  A page vLLM frees after
    that (a finished request's, in `update_from_output`) frees its appended blocks at the same step, whose line closes
    when the next step opens, and leaves its untouched filler resident; each pass names the step it covers.  Without passes (a remote session runs them), the map is the
    same."""
    from verity.core.protocols.poub import codec, memory_map

    s = Store()
    b = Binder(s, passes=passes)
    r = runner()
    r.kv_caches = [torch.zeros(2, PAGES, SLOTS, HEADS, 2 * DIM, dtype=torch.bfloat16)]
    b.on_load(r)
    b.on_kv(r, SimpleNamespace(num_blocks=PAGES))
    m = s.record_memory_map(registration=bytes(range(64)))
    open_step = lambda: s._open_pass and s._open_pass.step
    page = Block(1)
    page.ref_cnt = 1
    s.kv_alloc([1, 2])
    stm = SimpleNamespace(req_to_blocks={"a": [page]})
    sched = SimpleNamespace(kv_cache_manager=SimpleNamespace(coordinator=SimpleNamespace(single_type_managers=[stm])),
                            requests={"a": SimpleNamespace(num_computed_tokens=SLOTS)})
    b.on_schedule(sched, SimpleNamespace(num_scheduled_tokens={"a": SLOTS}))
    assert open_step() == (0 if passes else None)
    r.kv_caches[0][:, 1] = 1.5
    b.end()
    s.kv_free([2])
    assert m.text().count("\n") == 1
    b.on_schedule(sched, SimpleNamespace(num_scheduled_tokens={}))
    assert open_step() == (1 if passes else None)
    (step,) = memory_map.parse(m.text().splitlines())[1]
    kv = s.region("kv/0")
    want = r.kv_caches[0][0, 1].contiguous().view(torch.uint8).numpy().tobytes()
    assert step.step == 0 and [w.slot for w in step.written] == [kv.first + 2, kv.first + 3]
    assert step.freed == () and {4, 5} <= set(s.idle_blocks("kv/0").tolist())
    assert [w.digest for w in step.written] == [codec.digest(want)] * 2 and want != bytes(4096)


def test_with_the_drivers_tiles_a_steps_pass_is_the_drivers_and_opens_against_its_z():
    """Each receipt bound to the tile output before its block (`NoiseTiles`, PoUW's stub), under the driver's own
    receipt: the pass's root is the driver's `Prover`'s over the same blocks, and an opening, made after the next pass,
    verifies with the tile's z."""
    from verity.core.protocols.poub import audit
    from verity.core.protocols.poub import receipts as rb
    from verity_vllm.protocols.poub.registry import PoubReceipt
    from verity_vllm.protocols.poub.verify import verify_opening

    s, recs, tiles = Store(salt_key=bytes(32)), [], audit.NoiseTiles()
    b = Binder(s, messages=Every(2, lambda t: bytes([t]) * 32, windows=1), on_window=recs.append, tiles=tiles)
    assert isinstance(b.receipt, PoubReceipt)
    r = runner()
    r.kv_caches = [torch.zeros(2, PAGES, SLOTS, HEADS, 2 * DIM, dtype=torch.bfloat16)]
    b.on_load(r)
    b.on_kv(r, SimpleNamespace(num_blocks=PAGES))
    for _ in range(3):
        b.on_schedule(EMPTY, SimpleNamespace(num_scheduled_tokens={}))
    pc, info = recs[0].passes[1], s.region("0.weight")
    data = [s._regions[0].backing.read(i * 4096, 4096) for i in range(info.blocks)]
    plan = rb.Plan.whole(len(data))
    ys = [rb.Receipter(pc.seed)(j, data[j], tiles.output(pc.seed, j)) for j in plan]
    assert pc.root == rb.tree(pc.seed, plan, ys, [s.salt(pc.seed, 0)]).commitment.root
    o = s.open([(1, 1)], window=0)[0]
    assert verify_opening(info, pc, s.plan(1, window=0), o, b.receipt, lambda w, i: w, None, tiles.output(pc.seed, 1))
    assert not verify_opening(info, pc, s.plan(1, window=0), o, b.receipt, lambda w, i: w, None, bytes(64))
