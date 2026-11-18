"""The binder's pieces on the CPU, against fake vLLM objects: the KV page layout read off a cache tensor's strides,
weights and KV registration from the memory hook's events of a runner (`engine.kv_blocks`), its step events as the
store's steps, allocations and frees, a step's pass, and the binder subscribed on the exchange with the hook on vLLM's
classes under the stand-in for vLLM."""
from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")

from verity.protocols.resource_accounting.memory.poub.prover.node_store.interface import (
    Append,
    WindowRecord,
)
from verity.protocols.resource_accounting.memory.poub.prover.node_store.registry import (
    NodeMemory,
    PoubReceipt,
    chain,
)
from verity.protocols.resource_accounting.memory.poub.prover.node_store.verify import verify_opening
from verity.protocols.resource_accounting.memory.poub.verifier import audit, codec, memory_map
from verity.protocols.resource_accounting.memory.poub.verifier import receipts as rb
from verity.hooks import RangeAllocated, RangeFreed, RangeWritten, StepBegan, StepEnded

from tests import fake_vllm
from tests.engine.test_kv_blocks import kv_classes
from verity_vllm import messages
from verity_vllm.engine import kv_blocks
from verity_vllm.engine.hooks import live
from verity_vllm.engine.kv_blocks import KERNEL_BLOCK_SIZES
from verity_vllm.protocols.poub.binder import (
    Binder,
    Every,
    Inbox,
    PagedBacking,
    TorchBacking,
    model_weights,
    page_layout,
    storage_bytes,
)

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


def load(b: Binder, r) -> None:
    b.on_event(kv_blocks.registered_values(r))


def declare(b: Binder, r) -> None:
    b.on_event(kv_blocks.ranges_declared(r, SimpleNamespace(num_blocks=PAGES)))


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
    load(b, r)
    declare(b, r)
    st = b.stats()
    assert st["bytes_by_kind"] == {"weights": 64 * 32 * 4, "kv": 3 * 2 * PAGES * 2048}
    assert st["unregistered"] == 2 and st["unregistered_bytes"] == 2 * 32 * 4
    assert [x.name for x in s.layout().regions][:1] == ["0.weight"]
    assert s.pool.ids.shape == (PAGES, 4) and s.pool.ids[1].tolist() == [4, 5, 6, 7]


def test_the_models_weights_are_each_storage_once_none_empty_and_found_through_its_wrappers():
    """`registered_values` and `model_weights`: a parameter that shares its storage with an earlier one is listed once, under the first name;
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
        got = model_weights(kv_blocks.registered_values(SimpleNamespace(model=held)).values)
        assert [(x.name, x.source, x.encode) for x in got] == [("first", "param:first", False), ("last", "param:last", False)]
        assert got[0].tensor.data_ptr() == w.data_ptr() and got[1].tensor.tolist() == [1.0, 1.0, 1.0]


def test_a_weight_that_ends_inside_a_block_is_registered_to_its_last_whole_block_and_its_tail_is_counted():
    """A tensor of 8 blocks and 256 bytes: the region is the 8 blocks, the 256 bytes are counted unregistered under the
    tensor's name, and under a budget of blocks the tensor takes 8 of it."""
    for budget, registered in ((None, 8 * 1024), (8, 8 * 1024), (7, 0)):
        s = NodeMemory()
        b = Binder(s, block_bytes=1024, weight_blocks=budget)
        load(b, SimpleNamespace(model=torch.nn.Sequential(torch.nn.Linear(64, 33, bias=False))))
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
        declare(b, r)
    assert "kv" not in b.stats()["bytes_by_kind"]
    setattr(r, name, [SLOTS])
    declare(b, r)
    assert b.stats()["bytes_by_kind"]["kv"] == 3 * 2 * PAGES * 2048


@pytest.mark.parametrize("named", [{}, {"kernel_block_sizes": None}, {"_kernel_block_sizes": []}])
def test_a_runner_that_names_no_kernel_block_sizes_is_refused(named):
    """A runner with neither name, or with one that holds no size, is refused, by a message that says both names: it is
    never taken to use the manager's block size."""
    b, r = Binder(NodeMemory(), block_bytes=1024), runner()
    del r.kernel_block_sizes
    vars(r).update(named)
    with pytest.raises(RuntimeError, match=r"names no kernel block sizes \(kernel_block_sizes or _kernel_block_sizes\)"):
        declare(b, r)
    assert "kv" not in b.stats()["bytes_by_kind"]


@pytest.mark.parametrize("bound", [[], None, "absent"])
def test_a_runner_that_bound_no_kv_caches_is_refused(bound):
    b, r = Binder(NodeMemory(), block_bytes=1024), runner()
    if bound == "absent":
        del r.kv_caches
    else:
        r.kv_caches = bound
    with pytest.raises(RuntimeError, match="the model runner bound no KV caches"):
        declare(b, r)
    assert "kv" not in b.stats()["bytes_by_kind"]


def test_kv_layers_of_different_page_shapes_are_refused():
    """One layer's slot is 128 bytes and the next one's 256: the pool is one layout, so the second layer is refused and
    no layer is registered."""
    b, r = Binder(NodeMemory(), block_bytes=1024), runner()
    r.kv_caches = [torch.zeros(2, PAGES, SLOTS, HEADS, d, dtype=torch.bfloat16) for d in (DIM, 2 * DIM)]
    with pytest.raises(NotImplementedError, match="KV layers of different page shapes"):
        declare(b, r)
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
            declare(b, r)
        assert "kv" not in b.stats()["bytes_by_kind"]
    else:
        declare(b, r)
        assert b.stats()["bytes_by_kind"]["kv"] == 2 * 2 * PAGES * 2048 and b.unregistered == {"kv[unpaged]": 2 * PAGES * 2048}


def test_the_bytes_of_a_kv_storage_that_no_layers_page_holds_are_counted_unregistered():
    """A cache that is the first ``PAGES`` pages of a storage of one page more: the pool is the ``PAGES`` pages, and
    the last page's two planes are counted, under ``kv[unpaged]``."""
    b, r = Binder(NodeMemory(), block_bytes=1024), runner()
    r.kv_caches = [torch.zeros(PAGES + 1, 2, SLOTS, HEADS, DIM, dtype=torch.bfloat16)[:PAGES]]
    declare(b, r)
    assert b.stats()["bytes_by_kind"] == {"kv": 2 * PAGES * 2048} and b.unregistered == {"kv[unpaged]": 2 * 2048}


@pytest.mark.parametrize("budget, registered", [(8, 64 * 32 * 4), (7, 0)])
def test_a_weight_block_budget_registers_the_tensors_that_fit_and_counts_the_rest(budget, registered):
    s = NodeMemory()
    b = Binder(s, block_bytes=1024, weight_blocks=budget)
    r = runner()
    load(b, r)
    declare(b, r)
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
    load(b, r)
    declare(b, r)
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
    b.on_event(RangeAllocated((1,), "a", step=0))
    b.on_event(RangeWritten(1, 0, SLOTS, 0))
    b.on_event(StepBegan(0, ("a",), SLOTS))
    r.kv_caches[0][:, 1] = 1.5
    b.on_event(StepEnded(0))
    b.on_event(RangeFreed((1,), step=0))
    b.on_event(StepBegan(1))
    b.on_event(StepEnded(1))
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


def test_steps_append_their_writes_and_free_the_freed_pages():
    s = NodeMemory()
    b = Binder(s, block_bytes=1024)
    r = runner()
    declare(b, r)
    b.on_event(RangeAllocated((1, 2), "a", step=0))
    b.on_event(RangeAllocated((3,), "c", step=0))
    for w in (RangeWritten(1, 0, 16, 0), RangeWritten(2, 0, 4, 0), RangeWritten(3, 0, 3, 0)):
        b.on_event(w)
    b.on_event(StepBegan(0, ("a", "c"), 23))
    assert s.in_pass and s._open_pass.blocks[0].tolist() == [4, 5, 6, 7]
    b.on_event(StepEnded(0))
    assert not s.in_pass and b.step == 1
    b.on_event(RangeWritten(2, 4, 1, 1))
    b.on_event(StepBegan(1, ("a",), 1))
    b.on_event(StepBegan(2))
    assert s.counters.calls["end_pass"] == 2 and b.step == 3
    b.on_event(RangeFreed((1, 2), step=2))
    assert not s.pool.allocated[[1, 2]].any() and s.pool.allocated[3]
    with pytest.raises(NotImplementedError, match="the store's binder runs without prefix caching"):
        b.on_event(RangeAllocated((1,), "c", reused=True, step=2))


def steps(b: Binder, n: int) -> None:
    """``n`` empty steps after the binder's last."""
    for _ in range(n):
        b.on_event(StepBegan(b.step))


def test_steps_run_in_chained_windows_with_the_readers_roots():
    s = NodeMemory()
    recs: list[WindowRecord] = []
    sigma = [bytes([t]) * 32 for t in range(3)]
    b = Binder(s, block_bytes=1024, messages=Every(2, sigma.__getitem__, windows=3), on_window=recs.append,
               reader=lambda segments: b"\x01" * 64)
    r = runner()
    load(b, r)
    declare(b, r)
    steps(b, 8)
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
    load(b, r)
    declare(b, r)

    def step() -> None:
        steps(b, 1)

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
    """The binder appends at `StepBegan`, before the step's forward writes its KV; the memory map takes the step's
    digests at `StepEnded`, from the bytes the forward wrote.  A page vLLM frees after
    that (a finished request's, in `update_from_output`) frees its appended blocks at the same step, whose line closes
    when the next step opens, and leaves its untouched filler resident; each pass names the step it covers.  Without passes (a remote session runs them), the map is the
    same."""
    s = NodeMemory()
    b = Binder(s, passes=passes)
    r = runner()
    r.kv_caches = [torch.zeros(2, PAGES, SLOTS, HEADS, 2 * DIM, dtype=torch.bfloat16)]
    load(b, r)
    declare(b, r)
    m = s.record_memory_map(registration=bytes(range(64)))

    def open_step():
        return s._open_pass and s._open_pass.step

    b.on_event(RangeAllocated((1, 2), "a", step=0))
    b.on_event(RangeWritten(1, 0, SLOTS, 0))
    b.on_event(StepBegan(0, ("a",), SLOTS))
    assert open_step() == (0 if passes else None)
    r.kv_caches[0][:, 1] = 1.5
    b.on_event(StepEnded(0))
    b.on_event(RangeFreed((2,), step=0))
    assert m.text().count("\n") == 1
    b.on_event(StepBegan(1))
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
    assert isinstance(b.receipt, PoubReceipt)
    r = runner()
    r.kv_caches = [torch.zeros(2, PAGES, SLOTS, HEADS, 2 * DIM, dtype=torch.bfloat16)]
    load(b, r)
    declare(b, r)
    for _ in range(3):
        steps(b, 1)
    pc, info = recs[0].passes[1], s.region("0.weight")
    data = [s._regions[0].backing.read(i * 4096, 4096) for i in range(info.blocks)]
    plan = rb.Plan.whole(len(data))
    ys = [rb.Receipter(pc.seed)(j, data[j], tiles.output(pc.seed, j)) for j in plan]
    assert pc.root == rb.tree(pc.seed, plan, ys, [s.salt(pc.seed, 0)]).commitment.root
    o = s.open([(1, 1)], window=0)[0]
    assert verify_opening(info, pc, s.plan(1, window=0), o, b.receipt, lambda w, i: w, None, tiles.output(pc.seed, 1))
    assert not verify_opening(info, pc, s.plan(1, window=0), o, b.receipt, lambda w, i: w, None, bytes(64))


FIRST_RUNNER = kv_blocks.RUNNERS[0]


def vllms_classes(monkeypatch, store: NodeMemory, first_runner: bool = False) -> SimpleNamespace:
    """vLLM's classes as the memory hook finds them under the stand-in for vLLM (`kv_classes`), the scheduler's calls
    recording whether the store had a pass open, a runner holding `runner`'s model and caches."""
    fake = fake_vllm.install(monkeypatch, extra=[f"{FIRST_RUNNER}:GPUModelRunner"] if first_runner else ())
    return kv_classes(fake, lambda: store.in_pass, first_runner, lambda: dict(vars(runner())))


@pytest.fixture
def binder():
    """Makes a test's binders (`Binder`'s arguments), with ``install`` on a fresh exchange with the memory hook attached,
    and uninstalls each when the test ends, pass or fail, so a test that fails with its binder installed leaves no live
    hook behind it."""
    messages.close()
    made: list[Binder] = []
    offs: list = []

    def make(store: NodeMemory, install: bool = False, **kw) -> Binder:
        made.append(Binder(store, **kw))
        if install:
            offs.append(kv_blocks.attach())
            made[-1].install(messages.current())
        return made[-1]

    yield make
    for b in made:
        b.uninstall()
    for off in offs:
        off()
    messages.close()


@pytest.mark.parametrize("first_runner", [False, True])
def test_installed_the_runners_load_registers_the_weights_and_uninstall_leaves_vllm_unpatched(monkeypatch, binder,
                                                                                              first_runner):
    """`Binder.install` subscribes to the exchange's memory and steps hooks and files ReuseOldestFreed, patching
    nothing: every patch is the memory hook's (each runner module that is there, the first runner's skipped where it is
    absent, as under the surface); a runner's loaded model's weights reach the store as registered values; `uninstall`
    removes the subscriptions, and does nothing on a binder never installed."""
    s = NodeMemory()
    v = vllms_classes(monkeypatch, s, first_runner)
    assert (importlib.util.find_spec(FIRST_RUNNER) is not None) == first_runner
    before = live()
    b = binder(s, block_bytes=1024)
    ex = messages.current()
    assert b.install(ex) is b and live() == before and ex.oldest_freed_first()
    assert ex.wants("memory") and ex.wants("steps")
    with pytest.raises(RuntimeError, match="installed"):
        b.install(ex)
    off = kv_blocks.attach()
    sites = v.sites
    assert len(sites) == (10 if first_runner else 8) and {(e.owner, e.name) for e in live(kv_blocks.TAG)} == set(sites)
    for i, cls in enumerate(v.runners):
        r = cls()
        r.loads = torch.nn.Sequential(*[torch.nn.Identity()] * i, torch.nn.Linear(64, 32, bias=False))
        assert r.load_model("x", y=1) == "loaded" and v.calls[-1] == ("load_model", ("x",), {"y": 1})
        assert r.initialize_kv_cache(SimpleNamespace(num_blocks=PAGES), True) == "kv"
        assert v.calls[-1] == ("initialize_kv_cache", True)
    assert [x.name for x in s.layout().regions] == [f"{i}.weight" for i in range(len(v.runners))]
    b.uninstall()
    assert not ex.wants("memory") and not ex.wants("steps")
    off()
    assert live(kv_blocks.TAG) == () and all(vars(cls)[name] is own for (cls, name), own in sites.items())
    Binder(NodeMemory()).uninstall()


def test_installed_a_profiling_kv_cache_registers_nothing_and_the_real_one_registers_the_pool(monkeypatch, binder):
    """vLLM initializes a KV cache once to profile memory and once for real. The profiling call, its flag by position
    or by name, declares no ranges; the next call, its flag false by position, declares them and the pool is
    registered."""
    s = NodeMemory()
    v = vllms_classes(monkeypatch, s)
    b = binder(s, install=True, block_bytes=1024)
    r, config = v.runners[0](), SimpleNamespace(num_blocks=PAGES)
    assert r.initialize_kv_cache(config, True) == "kv" and r.initialize_kv_cache(config, is_profiling=True) == "kv"
    assert v.calls == [("initialize_kv_cache", True)] * 2 and b.stats()["regions"] == 0
    assert r.initialize_kv_cache(config, False) == "kv" and b.stats()["bytes_by_kind"] == {"kv": 3 * 2 * PAGES * 2048}


@pytest.mark.parametrize("kv", [True, False])
def test_installed_the_pools_pages_are_the_stores_and_a_freed_page_is_not_touched_back(monkeypatch, binder, kv):
    """Through the memory hook, `BlockPool.get_new_blocks` allocates the store's pages and `free_blocks` frees them,
    moved to the back of vLLM's free queue (the binder asks ReuseOldestFreed); `touch` of a freed page (a prefix-cache hit)
    is refused, and of a page in use or the null block is vLLM's own. With ``kv=False`` the same calls reach no
    store."""
    s = NodeMemory()
    v = vllms_classes(monkeypatch, s)
    binder(s, install=True, block_bytes=1024, kv=kv)
    v.runners[0]().initialize_kv_cache(SimpleNamespace(num_blocks=PAGES))
    pool = v.pool()
    got = pool.get_new_blocks(3)
    assert [x.block_id for x in got] == [1, 2, 3] and v.calls[-1] == ("get_new_blocks", 3)
    assert s.pool.allocated[[1, 2, 3]].all() and not s.pool.allocated[4] if kv else s._pool is None
    assert pool.touch([got[0], pool.blocks[0]]) == "touched" and v.calls[-1] == ("touch", [1, 0])
    assert pool.free_blocks(got[:2]) is None and v.calls[-1] == ("free_blocks", "list")
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
    b = binder(s, install=True, block_bytes=1024)
    v.runners[0]().initialize_kv_cache(SimpleNamespace(num_blocks=PAGES))
    sched = v.scheduler()
    assert sched.schedule("a", k=1) is sched.output and v.calls[-1] == ("schedule", ("a",), {"k": 1}, False)
    assert s.in_pass and b.step == 1
    assert sched.update_from_output(sched.output, "out") == "outputs"
    assert v.calls[-1] == ("update_from_output", True, "out", False) and not s.in_pass
    assert s.counters.calls["end_pass"] == 1
