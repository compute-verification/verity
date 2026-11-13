"""The binder's pieces on the CPU, against fake vLLM objects: the KV page layout read off a cache tensor's strides,
weights and KV registration from a runner, the scheduler's appends, FIFO reuse of freed pages, a step's pass, and
the binder installed on vLLM's classes under the stand-in for vLLM."""
from __future__ import annotations

import importlib.util
from collections import deque
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")

from verity.protocols.accounting.poub.verifier import audit, codec, memory_map
from verity.protocols.accounting.poub.verifier import receipts as rb

from tests import fake_vllm
from verity_vllm.engine.hooks import Hooks, Patch
from verity_vllm.protocols.remote_bandwidth.binder import (
    KERNEL_BLOCK_SIZES,
    Binder,
    Every,
    Inbox,
    PagedBacking,
    TorchBacking,
    model_weights,
    page_layout,
    storage_bytes,
)
from verity_vllm.protocols.remote_bandwidth.interface import Append, WindowRecord
from verity_vllm.protocols.remote_bandwidth.registry import (
    NodeMemory,
    RemoteBandwidthReceipt,
    chain,
)
from verity_vllm.protocols.remote_bandwidth.verify import verify_opening

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
                           kernel_block_sizes=[SLOTS])


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
    s = NodeMemory()
    b = Binder(s, block_bytes=1024)
    r = runner(shared, interleaved)
    b.on_load(r)
    b.on_kv(r, SimpleNamespace(num_blocks=PAGES))
    st = b.stats()
    assert st["bytes_by_kind"] == {"weights": 64 * 32 * 4, "kv": 3 * 2 * PAGES * 2048}
    assert st["unregistered"] == 2 and st["unregistered_bytes"] == 2 * 32 * 4
    assert [x.name for x in s.layout().regions][:1] == ["0.weight"]
    assert s.pool.ids.shape == (PAGES, 4) and s.pool.ids[1].tolist() == [4, 5, 6, 7]


def test_the_models_weights_are_each_storage_once_none_empty_and_found_through_its_wrappers():
    """`model_weights`: a parameter that shares its storage with an earlier one is listed once, under the first name;
    one of no elements is not listed; and a model held inside wrappers that have no parameters of their own (each with
    a ``runnable``) is the one listed, which a ``runnable`` of its own does not change."""

    class Model(torch.nn.Module):
        runnable = None

    w = torch.zeros(4, 8)
    model = Model()
    model.first, model.tied = torch.nn.Parameter(w), torch.nn.Parameter(w)
    model.empty, model.last = torch.nn.Parameter(torch.zeros(0)), torch.nn.Parameter(torch.ones(3))
    assert [n for n, _ in model.named_parameters()] == ["first", "tied", "empty", "last"]
    for held in (model, SimpleNamespace(runnable=SimpleNamespace(runnable=model))):
        got = model_weights(SimpleNamespace(model=held))
        assert [(x.name, x.source, x.encode) for x in got] == [("first", "param:first", False), ("last", "param:last", False)]
        assert got[0].tensor.data_ptr() == w.data_ptr() and got[1].tensor.tolist() == [1.0, 1.0, 1.0]


def test_a_weight_that_ends_inside_a_block_is_registered_to_its_last_whole_block_and_its_tail_is_counted():
    """A tensor of 8 blocks and 256 bytes: the region is the 8 blocks, the 256 bytes are counted unregistered under the
    tensor's name, and under a budget of blocks the tensor takes 8 of it."""
    for budget, registered in ((None, 8 * 1024), (8, 8 * 1024), (7, 0)):
        s = NodeMemory()
        b = Binder(s, block_bytes=1024, weight_blocks=budget)
        b.on_load(SimpleNamespace(model=torch.nn.Sequential(torch.nn.Linear(64, 33, bias=False))))
        assert b.unregistered == ({"0.weight[tail]": 256} if registered else {"0.weight": 8 * 1024 + 256}), budget
        assert [(x.name, x.blocks) for x in s.layout().regions] == ([("0.weight", 8)] if registered else [])


@pytest.mark.parametrize("name", KERNEL_BLOCK_SIZES)
def test_a_kernel_block_size_other_than_the_managers_is_refused(name):
    """Under either runner's name for them.  vLLM gives a cache group whose backend doesn't take the manager's block size
    a kernel block size that divides it, and the pages registered here are the manager's; the same runner with the
    manager's size under that name registers."""
    b, r = Binder(NodeMemory(), block_bytes=1024), runner()
    del r.kernel_block_sizes
    setattr(r, name, [SLOTS, SLOTS // 2])
    with pytest.raises(NotImplementedError, match=rf"kernel block sizes \[{SLOTS}, {SLOTS // 2}\] other than the manager's {SLOTS}$"):
        b.on_kv(r, SimpleNamespace(num_blocks=PAGES))
    assert "kv" not in b.stats()["bytes_by_kind"]
    setattr(r, name, [SLOTS])
    b.on_kv(r, SimpleNamespace(num_blocks=PAGES))
    assert b.stats()["bytes_by_kind"]["kv"] == 3 * 2 * PAGES * 2048


@pytest.mark.parametrize("named", [{}, {"kernel_block_sizes": None}, {"_kernel_block_sizes": []}])
def test_a_runner_that_names_no_kernel_block_sizes_is_refused(named):
    """A runner with neither name, or with one that holds no size, is refused, by a message that says both names: it is
    never taken to use the manager's block size."""
    b, r = Binder(NodeMemory(), block_bytes=1024), runner()
    del r.kernel_block_sizes
    vars(r).update(named)
    with pytest.raises(RuntimeError, match=r"names no kernel block sizes \(kernel_block_sizes or _kernel_block_sizes\)"):
        b.on_kv(r, SimpleNamespace(num_blocks=PAGES))
    assert "kv" not in b.stats()["bytes_by_kind"]


@pytest.mark.parametrize("bound", [[], None, "absent"])
def test_a_runner_that_bound_no_kv_caches_is_refused(bound):
    b, r = Binder(NodeMemory(), block_bytes=1024), runner()
    if bound == "absent":
        del r.kv_caches
    else:
        r.kv_caches = bound
    with pytest.raises(RuntimeError, match="the model runner bound no KV caches"):
        b.on_kv(r, SimpleNamespace(num_blocks=PAGES))
    assert "kv" not in b.stats()["bytes_by_kind"]


def test_kv_layers_of_different_page_shapes_are_refused():
    """One layer's slot is 128 bytes and the next one's 256: the pool is one layout, so the second layer is refused and
    no layer is registered."""
    b, r = Binder(NodeMemory(), block_bytes=1024), runner()
    r.kv_caches = [torch.zeros(2, PAGES, SLOTS, HEADS, d, dtype=torch.bfloat16) for d in (DIM, 2 * DIM)]
    with pytest.raises(NotImplementedError, match="KV layers of different page shapes"):
        b.on_kv(r, SimpleNamespace(num_blocks=PAGES))
    assert "kv" not in b.stats()["bytes_by_kind"]


@pytest.mark.parametrize("planes_apart, overlap", [(0, True), (0.5, True), (1, True), (2, False)])
def test_two_kv_layers_whose_planes_overlap_in_one_storage_are_refused(planes_apart, overlap):
    """Two layers in one storage, the second's pages starting ``planes_apart`` planes after the first's. On the same
    planes, half a plane on and one plane on (its K plane the first's V plane) they overlap and are refused, with
    nothing registered; two planes on, the next layer of the storage, they register."""
    b, r = Binder(NodeMemory(), block_bytes=1024), runner()
    whole = torch.zeros(PAGES, 3, 2, SLOTS, HEADS, DIM, dtype=torch.bfloat16)
    first, plane = whole[:, 0], SLOTS * HEADS * DIM
    second = torch.as_strided(whole, first.shape, first.stride(), int(planes_apart * plane))
    r.kv_caches = [first, second]
    if overlap:
        with pytest.raises(ValueError, match="two KV layers' pages overlap in one storage"):
            b.on_kv(r, SimpleNamespace(num_blocks=PAGES))
        assert "kv" not in b.stats()["bytes_by_kind"]
    else:
        b.on_kv(r, SimpleNamespace(num_blocks=PAGES))
        assert b.stats()["bytes_by_kind"]["kv"] == 2 * 2 * PAGES * 2048 and b.unregistered == {"kv[unpaged]": 2 * PAGES * 2048}


def test_the_bytes_of_a_kv_storage_that_no_layers_page_holds_are_counted_unregistered():
    """A cache that is the first ``PAGES`` pages of a storage of one page more: the pool is the ``PAGES`` pages, and
    the last page's two planes are counted, under ``kv[unpaged]``."""
    b, r = Binder(NodeMemory(), block_bytes=1024), runner()
    r.kv_caches = [torch.zeros(PAGES + 1, 2, SLOTS, HEADS, DIM, dtype=torch.bfloat16)[:PAGES]]
    b.on_kv(r, SimpleNamespace(num_blocks=PAGES))
    assert b.stats()["bytes_by_kind"] == {"kv": 2 * PAGES * 2048} and b.unregistered == {"kv[unpaged]": 2 * 2048}


@pytest.mark.parametrize("budget, registered", [(8, 64 * 32 * 4), (7, 0)])
def test_a_weight_block_budget_registers_the_tensors_that_fit_and_counts_the_rest(budget, registered):
    s = NodeMemory()
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
    s = NodeMemory()
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
    return list(memory_map.parse(m.text().splitlines(keepends=True))[1])


def test_a_filler_of_less_than_one_block_is_refused_and_a_longer_one_is_its_whole_blocks():
    s = NodeMemory()
    b = Binder(s, passes=False, kv=False)
    with pytest.raises(ValueError, match="filler of 4095 bytes holds no 4096-byte block"):
        b.register_filler(4095, "cpu")
    assert s.layout().regions == ()
    b.register_filler(4096 + 4095, "cpu")
    assert b.filler.numel() == 4096 and [(x.name, x.kind, x.blocks) for x in s.layout().regions] == [("filler", "filler", 1)]


def test_a_fixed_schedule_of_no_passes_a_window_is_refused_and_one_with_no_last_window_never_closes():
    with pytest.raises(ValueError, match="a window is at least one pass"):
        Every(0, lambda t: bytes([t]) * 32)
    every = Every(1, lambda t: bytes([t]) * 32)
    assert [every() for _ in range(3)] == [[bytes([t]) * 32] for t in range(3)]
    two = Every(2, lambda t: bytes([t]) * 32)
    assert [two() for _ in range(5)] == [[bytes(32)], [], [b"\x01" * 32], [], [b"\x02" * 32]]


def test_steps_append_the_scheduled_slots_and_reuse_freed_pages_fifo():
    s = NodeMemory()
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
    s = NodeMemory()
    recs: list[WindowRecord] = []
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
    """The verifier's message ends the window at the next pass's start, however many passes it held; chal_0's window
    chains from the registration digest the binder was given."""
    s, inbox = NodeMemory(), Inbox()
    recs: list[WindowRecord] = []
    digest = bytes(range(64))
    b = Binder(s, block_bytes=1024, messages=inbox, registration=digest, on_window=recs.append,
               reader=lambda segments: b"\x02" * 64)
    r = runner()
    b.on_load(r)
    b.on_kv(r, SimpleNamespace(num_blocks=PAGES))

    def step() -> None:
        b.on_schedule(EMPTY, SimpleNamespace(num_scheduled_tokens={}))

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
    s = NodeMemory()
    b = Binder(s, passes=passes)
    r = runner()
    r.kv_caches = [torch.zeros(2, PAGES, SLOTS, HEADS, 2 * DIM, dtype=torch.bfloat16)]
    b.on_load(r)
    b.on_kv(r, SimpleNamespace(num_blocks=PAGES))
    m = s.record_memory_map(registration=bytes(range(64)))

    def open_step():
        return s._open_pass and s._open_pass.step

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
    s, tiles = NodeMemory(salt_key=bytes(32)), audit.NoiseTiles()
    recs: list[WindowRecord] = []
    b = Binder(s, messages=Every(2, lambda t: bytes([t]) * 32, windows=1), on_window=recs.append, tiles=tiles)
    assert isinstance(b.receipt, RemoteBandwidthReceipt)
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


FIRST_RUNNER, RUNNER_V2 = "vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu.model_runner"
SITES = {"runner": ("load_model", "initialize_kv_cache"), "pool": ("get_new_blocks", "free_blocks", "touch"),
         "scheduler": ("schedule", "update_from_output")}


def vllms_classes(monkeypatch, store: NodeMemory, first_runner: bool = False) -> SimpleNamespace:
    """vLLM's model runner, block pool and scheduler as `Binder.install` finds them, under the stand-in for vLLM
    (`tests.fake_vllm`): each method records its call in ``calls`` (the scheduler's, with whether the store had a pass
    open) and answers with a value of its own.  As vLLM's, a runner has its model (``loads``) once its own `load_model`
    ran and its KV caches (``binds``) once its own `initialize_kv_cache` did, so a hook that reads either does so after
    vLLM's own.  The surface holds Model Runner V2 alone; with ``first_runner`` the first runner's module is there
    too, with a class of its own."""
    calls: list[tuple] = []

    def runner_class():
        class GPUModelRunner:
            def __init__(self) -> None:
                held = vars(runner())
                self.loads, self.binds = held.pop("model"), held.pop("kv_caches")
                vars(self).update(held)

            def load_model(self, *a, **kw):
                calls.append(("load_model", a, kw))
                self.model = self.loads
                return "loaded"

            def initialize_kv_cache(self, kv_cache_config, is_profiling=False):
                calls.append(("initialize_kv_cache", is_profiling))
                self.kv_caches = self.binds
                return "kv"

        return GPUModelRunner

    class BlockPool:
        """As vLLM's: a freed page goes back on the free queue, here at its front."""

        def __init__(self) -> None:
            self.blocks = [Block(i) for i in range(PAGES)]
            self.free_block_queue = Queue(self.blocks[1:])

        def get_new_blocks(self, n):
            got = [self.free_block_queue.q.popleft() for _ in range(n)]
            for b in got:
                b.ref_cnt = 1
            calls.append(("get_new_blocks", n))
            return got

        def free_blocks(self, ordered):
            calls.append(("free_blocks", type(ordered).__name__))
            for b in ordered:
                b.ref_cnt -= 1
                self.free_block_queue.q.appendleft(b)

        def touch(self, blocks):
            calls.append(("touch", [b.block_id for b in blocks]))
            return "touched"

    class Scheduler:
        def __init__(self) -> None:
            vars(self).update(vars(EMPTY))
            self.output = SimpleNamespace(num_scheduled_tokens={})

        def schedule(self, *a, **kw):
            calls.append(("schedule", a, kw, store.in_pass))
            return self.output

        def update_from_output(self, so, out):
            calls.append(("update_from_output", so is self.output, out, store.in_pass))
            return "outputs"

    fake = fake_vllm.install(monkeypatch, extra=[f"{FIRST_RUNNER}:GPUModelRunner"] if first_runner else ())
    runners = [runner_class()]
    fake.set(f"{RUNNER_V2}:GPUModelRunner", runners[0])
    if first_runner:
        runners.append(runner_class())
        fake.set(f"{FIRST_RUNNER}:GPUModelRunner", runners[1])
    fake.set("vllm.v1.core.block_pool:BlockPool", BlockPool)
    fake.set("vllm.v1.core.sched.scheduler:Scheduler", Scheduler)
    owners = [*((c, SITES["runner"]) for c in runners), (BlockPool, SITES["pool"]), (Scheduler, SITES["scheduler"])]
    return SimpleNamespace(calls=calls, runners=runners, pool=BlockPool, scheduler=Scheduler,
                           sites={(cls, name): vars(cls)[name] for cls, names in owners for name in names})


@pytest.fixture
def binder():
    """Makes a test's binders (`Binder`'s arguments) and uninstalls each when the test ends, pass or fail, so a test
    that fails with its binder installed leaves no live hook behind it."""
    made: list[Binder] = []

    def make(store: NodeMemory, **kw) -> Binder:
        made.append(Binder(store, **kw))
        return made[-1]

    yield make
    for b in made:
        b.uninstall()


@pytest.mark.parametrize("first_runner", [False, True])
def test_install_wraps_vllms_classes_and_uninstall_puts_them_back(monkeypatch, binder, first_runner):
    """`Binder.install` wraps the two methods of each runner module that is there (the first runner's is skipped where
    it is absent, as under the surface), the pool's three and the scheduler's two, each under the hooks' tag; a
    runner's wrappers hand back what vLLM's own methods returned, and its loaded model's weights are registered;
    `uninstall` leaves each class holding what it held, and does nothing on a binder never installed."""
    s = NodeMemory()
    v = vllms_classes(monkeypatch, s, first_runner)
    assert (importlib.util.find_spec(FIRST_RUNNER) is not None) == first_runner
    b = binder(s, block_bytes=1024)
    assert b.install() is b
    hooks = b.hooks
    assert hooks is not None and len(v.sites) == (9 if first_runner else 7) == len(hooks.entries)
    assert all(vars(cls)[name] is not own for (cls, name), own in v.sites.items())
    patches = [e for e in hooks.entries if isinstance(e, Patch)]
    assert {(e.owner, e.name, e.tag) for e in patches} == {(cls, name, "verity-store") for cls, name in v.sites}
    for i, cls in enumerate(v.runners):
        r = cls()
        r.loads = torch.nn.Sequential(*[torch.nn.Identity()] * i, torch.nn.Linear(64, 32, bias=False))
        assert r.load_model("x", y=1) == "loaded" and v.calls[-1] == ("load_model", ("x",), {"y": 1})
        assert r.initialize_kv_cache(SimpleNamespace(num_blocks=PAGES), True) == "kv"
        assert v.calls[-1] == ("initialize_kv_cache", True)
    assert [x.name for x in s.layout().regions] == [f"{i}.weight" for i in range(len(v.runners))]
    b.uninstall()
    assert all(vars(cls)[name] is own for (cls, name), own in v.sites.items()) and hooks.entries == ()
    Binder(NodeMemory()).uninstall()


def test_installed_a_profiling_kv_cache_registers_nothing_and_the_real_one_registers_the_pool(monkeypatch, binder):
    """vLLM initializes a KV cache once to profile memory and once for real. The profiling call, its flag by position
    or by name, registers no pool; the next call, its flag false by position, registers it; the hooks may be the
    caller's own."""
    s = NodeMemory()
    v = vllms_classes(monkeypatch, s)
    mine = Hooks(tag="mine")
    b = binder(s, block_bytes=1024).install(mine)
    assert b.hooks is mine and {e.tag for e in mine.entries} == {"mine"}
    r, config = v.runners[0](), SimpleNamespace(num_blocks=PAGES)
    assert r.initialize_kv_cache(config, True) == "kv" and r.initialize_kv_cache(config, is_profiling=True) == "kv"
    assert v.calls == [("initialize_kv_cache", True)] * 2 and b.stats()["regions"] == 0
    assert r.initialize_kv_cache(config, False) == "kv" and b.stats()["bytes_by_kind"] == {"kv": 3 * 2 * PAGES * 2048}


@pytest.mark.parametrize("kv", [True, False])
def test_installed_the_pools_pages_are_the_stores_and_a_freed_page_is_not_touched_back(monkeypatch, binder, kv):
    """`BlockPool.get_new_blocks` allocates the store's pages and `free_blocks` frees them, moved to the back of
    vLLM's free queue, when it is handed an iterator too; `touch` of a freed page (a prefix-cache hit) is refused, and
    of a page in use or the null block is vLLM's own. With ``kv=False`` the same calls reach no store."""
    s = NodeMemory()
    v = vllms_classes(monkeypatch, s)
    binder(s, block_bytes=1024, kv=kv).install()
    v.runners[0]().initialize_kv_cache(SimpleNamespace(num_blocks=PAGES))
    pool = v.pool()
    got = pool.get_new_blocks(3)
    assert [x.block_id for x in got] == [1, 2, 3] and v.calls[-1] == ("get_new_blocks", 3)
    assert s.pool.allocated[[1, 2, 3]].all() and not s.pool.allocated[4] if kv else s._pool is None
    assert pool.touch([got[0], pool.blocks[0]]) == "touched" and v.calls[-1] == ("touch", [1, 0])
    assert pool.free_blocks(iter(got[:2])) is None and v.calls[-1] == ("free_blocks", "list")
    assert [x.block_id for x in pool.free_block_queue.q] == [4, 5, 1, 2]
    assert not s.pool.allocated[[1, 2]].any() and s.pool.allocated[3] if kv else s._pool is None
    with pytest.raises(NotImplementedError, match="the store's binder runs without prefix caching"):
        pool.touch([got[2], got[0]])
    assert v.calls[-1] == ("free_blocks", "list")


def test_installed_a_steps_pass_opens_after_vllm_schedules_and_ends_before_vllm_reads_the_output(monkeypatch, binder):
    """`Scheduler.schedule` runs vLLM's own first, with its arguments, and then opens the step's pass;
    `Scheduler.update_from_output` ends the pass before vLLM's own runs, so the pages vLLM frees there are freed after
    the step's digests are taken."""
    s = NodeMemory()
    v = vllms_classes(monkeypatch, s)
    b = binder(s, block_bytes=1024).install()
    v.runners[0]().initialize_kv_cache(SimpleNamespace(num_blocks=PAGES))
    sched = v.scheduler()
    assert sched.schedule("a", k=1) is sched.output and v.calls[-1] == ("schedule", ("a",), {"k": 1}, False)
    assert s.in_pass and b.step == 1
    assert sched.update_from_output(sched.output, "out") == "outputs"
    assert v.calls[-1] == ("update_from_output", True, "out", False) and not s.in_pass
    assert s.counters.calls["end_pass"] == 1
