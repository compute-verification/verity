"""The binder's pieces on the CPU, against fake vLLM objects: the KV page layout read off a cache tensor's strides,
weights and KV registration from a runner, the scheduler's appends, FIFO reuse of freed pages, and a step's pass."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from collections import deque
from collections.abc import Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")

from verity.protocols.poub.verifier import audit, codec, memory_map
from verity.protocols.poub.verifier import receipts as rb

from verity_vllm.engine import hooks
from verity_vllm.protocols.remote_bandwidth.binder import (
    KERNEL_BLOCK_SIZES,
    RUNNERS,
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


def test_the_models_weights_are_each_storage_once_with_no_empty_one_read_through_what_wraps_the_model():
    """`model_weights`: a parameter over a storage already listed (a tied weight held as a second parameter) and one
    of no elements are left out, and a model without ``named_parameters`` is read through its ``runnable``, however
    many times it is wrapped."""
    model = torch.nn.Sequential(torch.nn.Linear(4, 2, bias=False), torch.nn.Linear(4, 2, bias=False))
    model[1].weight = torch.nn.Parameter(model[0].weight.data)
    model.register_parameter("none", torch.nn.Parameter(torch.zeros(0)))
    assert [name for name, _ in model.named_parameters()] == ["none", "0.weight", "1.weight"]
    for held in (model, SimpleNamespace(runnable=model), SimpleNamespace(runnable=SimpleNamespace(runnable=model))):
        (w,) = model_weights(SimpleNamespace(model=held))
        assert (w.name, w.source, w.encode, w.receipt) == ("0.weight", "param:0.weight", False, None)
        assert w.tensor.data_ptr() == model[0].weight.data_ptr() and w.tensor.shape == (2, 4)


def test_a_weights_bytes_past_its_last_whole_block_are_counted_unregistered():
    """`Binder.on_load`: a weight of 8 blocks and 128 bytes registers the 8 blocks, and the 128 bytes are counted
    under the weight's name with ``[tail]``."""
    b, r = Binder(NodeMemory(), block_bytes=1024), runner()
    r.model = torch.nn.Sequential(torch.nn.Linear(65, 32, bias=False))
    b.on_load(r)
    st = b.stats()
    assert b.unregistered == {"0.weight[tail]": 128} and st["bytes_by_kind"] == {"weights": 8 * 1024}
    assert (st["unregistered"], st["unregistered_bytes"]) == (1, 128)


@pytest.mark.parametrize("named", [{}, {"kv_caches": None}, {"kv_caches": []}])
def test_a_runner_that_bound_no_kv_caches_is_refused(named):
    b, r = Binder(NodeMemory(), block_bytes=1024), runner()
    del r.kv_caches
    vars(r).update(named)
    with pytest.raises(RuntimeError, match="^the model runner bound no KV caches$"):
        b.on_kv(r, SimpleNamespace(num_blocks=PAGES))
    assert "kv" not in b.stats()["bytes_by_kind"] and b.slots == 0


def test_kv_layers_of_different_page_shapes_are_refused():
    """A layer whose slot holds twice the bytes of the others': the pool has one page layout."""
    b, r = Binder(NodeMemory(), block_bytes=1024), runner()
    r.kv_caches[2] = torch.zeros(2, PAGES, SLOTS, HEADS, 2 * DIM, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="^KV layers of different page shapes$"):
        b.on_kv(r, SimpleNamespace(num_blocks=PAGES))
    assert "kv" not in b.stats()["bytes_by_kind"] and b.slots == 0


def two_layers(shift: int):
    """A runner of two layers in one storage, each (pages, 2, slots, heads, dim) and contiguous, the second ``shift``
    elements after the first."""
    n = PAGES * 2 * SLOTS * HEADS * DIM
    whole = torch.zeros(2 * n, dtype=torch.bfloat16)
    r = runner()
    r.kv_caches = [whole[at:at + n].view(PAGES, 2, SLOTS, HEADS, DIM) for at in (0, shift)]
    return r


@pytest.mark.parametrize("shift", [0, 32, 1024])
def test_kv_layers_whose_pages_overlap_in_one_storage_are_refused(shift):
    """The second layer over the first's pages, 64 bytes into them, or one plane on: a plane of one starts inside a
    plane of the other.  With the second right after the first, the two register."""
    b = Binder(NodeMemory(), block_bytes=1024)
    with pytest.raises(ValueError, match="^two KV layers' pages overlap in one storage$"):
        b.on_kv(two_layers(shift), SimpleNamespace(num_blocks=PAGES))
    assert "kv" not in b.stats()["bytes_by_kind"] and b.slots == 0
    b.on_kv(two_layers(PAGES * 2 * SLOTS * HEADS * DIM), SimpleNamespace(num_blocks=PAGES))
    assert b.stats()["bytes_by_kind"] == {"kv": 2 * 2 * PAGES * 2048} and b.unregistered == {} and b.slots == SLOTS


def test_a_kv_storages_bytes_that_no_layers_page_holds_are_counted_unregistered():
    """Two of the three layers of one storage bound: the third's bytes are in the storage and in no page."""
    b, r = Binder(NodeMemory(), block_bytes=1024), runner(shared=True)
    del r.kv_caches[1]
    b.on_kv(r, SimpleNamespace(num_blocks=PAGES))
    assert b.unregistered == {"kv[unpaged]": 2 * PAGES * 2048}
    assert b.stats()["bytes_by_kind"] == {"kv": 2 * 2 * PAGES * 2048}


def test_a_filler_of_less_than_one_block_is_refused():
    b = Binder(NodeMemory(), block_bytes=1024)
    with pytest.raises(ValueError, match="^filler of 1023 bytes holds no 1024-byte block$"):
        b.register_filler(1023, "cpu")
    assert b.filler is None and b.stats()["bytes_by_kind"] == {}
    b.register_filler(1024, "cpu")
    assert b.stats()["bytes_by_kind"] == {"filler": 1024}


def test_a_fixed_schedule_takes_windows_of_one_pass_and_refuses_none():
    """`Every`: with one pass a window, each call is a window's first, the call after the last window's is the close,
    and the calls after that have no message."""
    with pytest.raises(ValueError, match="^a window is at least one pass$"):
        Every(0, lambda t: bytes(32))
    every = Every(1, lambda t: bytes([t]) * 32, windows=2)
    assert [every() for _ in range(5)] == [[bytes(32)], [b"\x01" * 32], [None], [], []]


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


def module(monkeypatch, name: str, package: bool = False, **names: Any) -> Any:
    """A stand-in module in ``sys.modules`` for the test, with a spec as an imported one has."""
    m = importlib.util.module_from_spec(importlib.machinery.ModuleSpec(name, None, is_package=package))
    vars(m).update(names)
    monkeypatch.setitem(sys.modules, name, m)
    return m


def vllm_standin(monkeypatch, said: list, runners: Sequence[str] = RUNNERS) -> SimpleNamespace:
    """vLLM's modules the binder patches, as stand-ins: the packages over them, a ``GPUModelRunner`` in each of
    ``runners`` (a `RUNNERS` module left out is None in ``sys.modules``: one that cannot be imported, for which
    `importlib.util.find_spec` answers None), ``BlockPool`` and ``Scheduler``.  Each method tells ``said`` its name,
    its object and what it was handed, and returns a value of its own."""

    def runner_class() -> type:
        class GPUModelRunner:
            def load_model(self, *a: Any, **kw: Any) -> str:
                said.append(("load_model", self, a, kw))
                return "loaded"

            def initialize_kv_cache(self, kv_cache_config: Any, *a: Any, **kw: Any) -> str:
                said.append(("initialize_kv_cache", self, kv_cache_config, a, kw))
                return "bound"

        return GPUModelRunner

    class BlockPool:
        def get_new_blocks(self, n: int) -> list[Block]:
            said.append(("get_new_blocks", self, n))
            return [Block(i) for i in range(1, n + 1)]

        def free_blocks(self, ordered: Any) -> None:
            said.append(("free_blocks", self, ordered))

        def touch(self, blocks: Any) -> str:
            said.append(("touch", self, blocks))
            return "touched"

    class Scheduler:
        def schedule(self, *a: Any, **kw: Any) -> str:
            said.append(("schedule", self, a, kw))
            return "scheduled"

        def update_from_output(self, so: Any, out: Any) -> str:
            said.append(("update_from_output", self, so, out))
            return "updated"

    for name in ("vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu", "vllm.v1.core", "vllm.v1.core.sched"):
        module(monkeypatch, name, package=True)
    classes = []
    for name in RUNNERS:
        if name in runners:
            classes.append(module(monkeypatch, name, GPUModelRunner=runner_class()).GPUModelRunner)
        else:
            monkeypatch.setitem(sys.modules, name, None)
    module(monkeypatch, "vllm.v1.core.block_pool", BlockPool=BlockPool)
    module(monkeypatch, "vllm.v1.core.sched.scheduler", Scheduler=Scheduler)
    return SimpleNamespace(runners=classes, BlockPool=BlockPool, Scheduler=Scheduler)


def wrapped(v: SimpleNamespace) -> list[tuple[type, str]]:
    """The methods `Binder.install` wraps, in its order."""
    return [*((cls, name) for cls in v.runners for name in ("load_model", "initialize_kv_cache")),
            (v.BlockPool, "get_new_blocks"), (v.BlockPool, "free_blocks"), (v.BlockPool, "touch"),
            (v.Scheduler, "schedule"), (v.Scheduler, "update_from_output")]


class ToldMemory(NodeMemory):
    """A store that only tells ``said`` the pages it is asked to allocate."""

    def __init__(self, said: list) -> None:
        super().__init__()
        self.said = said

    def kv_alloc(self, pages: Sequence[int]) -> None:
        self.said.append(("kv_alloc", pages))


class Told(Binder):
    """A `Binder` whose own work at each hook is only told to ``said``, so a test reads what a wrapper calls, with
    what, and on which side of vLLM's own method."""

    def __init__(self, said: list, **kw: Any) -> None:
        super().__init__(ToldMemory(said), **kw)
        self.said = said

    def on_load(self, runner: Any) -> None:
        self.said.append(("on_load", runner))

    def on_kv(self, runner: Any, kv_cache_config: Any) -> None:
        self.said.append(("on_kv", runner, kv_cache_config))

    def on_free(self, pool: Any, blocks: Sequence[Any]) -> None:
        self.said.append(("on_free", pool, blocks))

    def on_schedule(self, sched: Any, so: Any) -> None:
        self.said.append(("on_schedule", sched, so))

    def end(self) -> None:
        self.said.append(("end",))


@pytest.fixture
def installed(monkeypatch):
    """``(v, b, said)``: the stand-in vLLM, a `Told` binder installed on it and what both tell; uninstalled after."""
    said: list = []
    v = vllm_standin(monkeypatch, said)
    b = Told(said).install()
    try:
        yield v, b, said
    finally:
        b.uninstall()


def test_install_wraps_both_runners_the_pool_and_the_scheduler_under_its_tag_and_uninstall_puts_them_back(monkeypatch):
    said: list = []
    v = vllm_standin(monkeypatch, said)
    own = {at: vars(at[0])[at[1]] for at in wrapped(v)}
    assert len(own) == 9
    b = Told(said)
    b.uninstall()
    assert b.hooks is None and all(vars(cls)[name] is f for (cls, name), f in own.items())
    try:
        assert b.install() is b and b.hooks is not None and b.hooks.tag == "verity-store"
        assert [(e.owner, e.name) for e in b.hooks.entries] == wrapped(v)
        assert all(e.tag == "verity-store" and e in hooks.live("verity-store") for e in b.hooks.entries)
        assert all(vars(cls)[name] is not f for (cls, name), f in own.items())
    finally:
        b.uninstall()
    assert b.hooks.entries == () and all(vars(cls)[name] is f for (cls, name), f in own.items())
    b.uninstall()
    mine = hooks.Hooks(tag="mine")
    try:
        assert Told(said).install(mine).hooks is mine
        assert [(e.owner, e.name, e.tag) for e in mine.entries] == [(*at, "mine") for at in wrapped(v)]
    finally:
        mine.uninstall()
    assert all(vars(cls)[name] is f for (cls, name), f in own.items()) and said == []


@pytest.mark.parametrize("has", RUNNERS)
def test_install_leaves_alone_a_runner_module_vllm_does_not_have(monkeypatch, has):
    said: list = []
    v = vllm_standin(monkeypatch, said, runners=[has])
    assert len(v.runners) == 1
    b = Told(said).install()
    try:
        assert [(e.owner, e.name) for e in b.hooks.entries] == wrapped(v) and len(b.hooks.entries) == 7
        r = v.runners[0]()
        assert r.load_model() == "loaded" and said == [("load_model", r, (), {}), ("on_load", r)]
    finally:
        b.uninstall()


def test_a_runners_hooks_run_after_vllms_own_and_a_profiling_kv_cache_is_not_registered(installed):
    """`GPUModelRunner.load_model` and `initialize_kv_cache` of both runners: vLLM's own runs first with what it was
    handed and its value is returned; then the weights, or the KV pool unless the call is a profiling one, by keyword or
    by position."""
    v, _, said = installed
    for cls in v.runners:
        r = cls()
        assert r.load_model(1, x=2) == "loaded"
        assert said == [("load_model", r, (1,), {"x": 2}), ("on_load", r)]
        for a, kw, registers in [((), {}, True), ((False,), {}, True), ((), {"is_profiling": False}, True),
                                 ((True,), {}, False), ((), {"is_profiling": True}, False)]:
            said.clear()
            assert r.initialize_kv_cache("config", *a, **kw) == "bound"
            assert said == [("initialize_kv_cache", r, "config", a, kw)] + [("on_kv", r, "config")] * registers
        said.clear()


def test_the_pools_hooks_allocate_and_free_after_vllms_own_and_refuse_a_freed_pages_return(installed):
    """`BlockPool.get_new_blocks`: the pages vLLM handed out are allocated in the store, and returned.
    `free_blocks`: vLLM's own gets every block of what it was handed, an iterator too, and then the binder the same
    blocks.  `touch`: a page with no reference that is not the null page is refused before vLLM's own runs (a prefix
    cache hit would bring it back with its KV); the null page and a referenced one pass."""
    v, _, said = installed
    pool = v.BlockPool()
    blocks = pool.get_new_blocks(2)
    assert [x.block_id for x in blocks] == [1, 2]
    assert said == [("get_new_blocks", pool, 2), ("kv_alloc", [1, 2])]
    said.clear()
    freed = [Block(3), Block(4)]
    assert pool.free_blocks(iter(freed)) is None
    assert said == [("free_blocks", pool, freed), ("on_free", pool, freed)] and said[0][2] is said[1][2]
    said.clear()
    held, null, free = Block(5), Block(0), Block(6)
    held.ref_cnt = 1
    assert pool.touch([held, null]) == "touched" and said == [("touch", pool, [held, null])]
    said.clear()
    with pytest.raises(NotImplementedError, match="^the store's binder runs without prefix caching$"):
        pool.touch([held, free])
    assert said == []


def test_with_kv_off_the_pools_hook_allocates_no_page_in_the_store(monkeypatch):
    said: list = []
    v = vllm_standin(monkeypatch, said)
    b = Told(said, kv=False).install()
    try:
        pool = v.BlockPool()
        assert [x.block_id for x in pool.get_new_blocks(3)] == [1, 2, 3] and said == [("get_new_blocks", pool, 3)]
    finally:
        b.uninstall()


def test_the_schedulers_hooks_open_the_step_after_schedule_and_end_it_before_update_from_output(installed):
    """`Scheduler.schedule`: vLLM's own runs first, and the binder is handed the scheduler and what it returned, which
    is returned.  `update_from_output`: the binder ends the step first, then vLLM's own runs and its value is returned."""
    v, _, said = installed
    sched = v.Scheduler()
    assert sched.schedule(1, x=2) == "scheduled"
    assert said == [("schedule", sched, (1,), {"x": 2}), ("on_schedule", sched, "scheduled")]
    said.clear()
    assert sched.update_from_output("so", "out") == "updated"
    assert said == [("end",), ("update_from_output", sched, "so", "out")]
