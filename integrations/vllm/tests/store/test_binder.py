"""The binder's pieces on the CPU, against fake vLLM objects: the KV page layout read off a cache tensor's strides,
weights and KV registration from a runner, the scheduler's appends, FIFO reuse of freed pages, and a step's pass."""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.store.binder import Binder, PagedBacking, TorchBacking, page_layout, storage_bytes  # noqa: E402
from verity_vllm.store.interface import Append  # noqa: E402
from verity_vllm.store.registry import Store  # noqa: E402

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


def test_steps_run_in_chained_windows_with_the_readers_roots():
    from verity_vllm.store.registry import chain
    s, recs = Store(), []
    sigma = [bytes([t]) * 32 for t in range(3)]
    b = Binder(s, block_bytes=1024, challenge=sigma.__getitem__, window_passes=2, on_window=recs.append,
               reader=lambda segments: b"\x01" * 64)
    r = runner()
    b.on_load(r)
    b.on_kv(r, SimpleNamespace(num_blocks=PAGES))
    sched = SimpleNamespace(kv_cache_manager=SimpleNamespace(coordinator=SimpleNamespace(
        single_type_managers=[SimpleNamespace(req_to_blocks={})])), requests={})
    for _ in range(5):
        b.on_schedule(sched, SimpleNamespace(num_scheduled_tokens={}))
    b.end_window()
    assert [len(w.passes) for w in recs] == [2, 2, 1] and [w.challenge for w in recs] == sigma
    assert recs[1].passes[0].seed == chain(sigma[1], recs[0].passes[-1].root, 0)
    assert (recs[0].passes[0].root, recs[0].passes[0].segment_roots) == (b"\x01" * 64, ()) and b.stats()["windows"] == 3
