"""The KV block hook (`engine.kv_blocks`) on vLLM's KV cache manager, block pool and scheduler under the stand-in for
vLLM: the events on the exchange in the order vLLM acts, each at its step and an allocation for its request,
ReuseOldestFreed's reuse order, HoldResident's held blocks, a prefix-cache hit as a reused allocation, a step's writes,
and one set of patches for every attachment."""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from verity.hooks import (HoldResident, RangeAllocated, RangeFreed, RangeWritten, ReuseOldestFreed, StepBegan,
                          StepEnded)

from tests import fake_vllm
from verity_vllm import messages
from verity_vllm.engine import hooks, kv_blocks

PAGES, SLOTS = 6, 16
SITES = {"manager": ("allocate_slots",), "pool": ("get_new_blocks", "free_blocks", "touch"),
         "scheduler": ("schedule", "update_from_output")}


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


def kv_classes(fake: fake_vllm.FakeVllm, state: Callable[[], Any] = lambda: None) -> SimpleNamespace:
    """vLLM's KV cache manager, block pool and scheduler, set on the installed stand-in ``fake``: each method records its
    call in ``calls`` (the scheduler's with ``state()`` at the call) and answers with a value of its own; a freed block
    goes to the free queue's front, as vLLM's does, and ``touch`` only records."""
    calls: list[tuple] = []

    class BlockPool:
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

    class KVCacheManager:
        def __init__(self, pool) -> None:
            self.block_pool = pool

        def allocate_slots(self, request, n):
            calls.append(("allocate_slots", request.request_id, n))
            return self.block_pool.get_new_blocks(n)

    class Scheduler:
        def __init__(self) -> None:
            self.stm = SimpleNamespace(block_size=SLOTS, req_to_blocks={})
            self.kv_cache_manager = SimpleNamespace(coordinator=SimpleNamespace(single_type_managers=[self.stm]))
            self.requests: dict[str, Any] = {}
            self.output = SimpleNamespace(num_scheduled_tokens={})
            self.during: Callable[[], None] = lambda: None

        def schedule(self, *a, **kw):
            calls.append(("schedule", a, kw, state()))
            self.during()
            return self.output

        def update_from_output(self, so, out):
            calls.append(("update_from_output", so is self.output, out, state()))
            return "outputs"

    fake.set("vllm.v1.core.block_pool:BlockPool", BlockPool)
    fake.set("vllm.v1.core.kv_cache_manager:KVCacheManager", KVCacheManager)
    fake.set("vllm.v1.core.sched.scheduler:Scheduler", Scheduler)
    owners = [(KVCacheManager, SITES["manager"]), (BlockPool, SITES["pool"]), (Scheduler, SITES["scheduler"])]
    return SimpleNamespace(calls=calls, pool=BlockPool, manager=KVCacheManager, scheduler=Scheduler,
                           sites={(cls, name): vars(cls)[name] for cls, names in owners for name in names})


@pytest.fixture
def subscribe():
    """A fresh exchange; each call attaches the hook (after the test installed the stand-in) and subscribes a listener
    to both hooks, filing ``requests`` first.  The hook comes off and the exchange closes when the test ends, so no
    patch or request outlives its test."""
    messages.close()
    ex, offs = messages.current(), []

    def make(*requests) -> list:
        heard: list = []
        offs.append(kv_blocks.attach())
        for r in requests:
            ex.request("test", r)
        offs.extend(ex.subscribe("test", hook, heard.append) for hook in ("memory", "steps"))
        return heard

    yield make
    for off in offs:
        off()
    messages.close()


def test_the_events_come_in_vllms_order_each_at_its_step_and_for_its_request(monkeypatch, subscribe):
    """Blocks allocated inside ``allocate_slots`` are the request's, at the step being scheduled; after vLLM's
    ``schedule`` come the slots the step writes and then its beginning, and its end before vLLM's
    ``update_from_output``; blocks freed there are freed at that step; vLLM's own methods run with their arguments and
    answer."""
    v = kv_classes(fake_vllm.install(monkeypatch))
    heard = subscribe()
    pool, sched = v.pool(), v.scheduler()
    manager = v.manager(pool)
    a, c = SimpleNamespace(request_id="a"), SimpleNamespace(request_id="c")
    sched.requests = {"a": SimpleNamespace(num_computed_tokens=20), "c": SimpleNamespace(num_computed_tokens=3)}
    sched.output = SimpleNamespace(num_scheduled_tokens={"a": 20, "c": 3})
    sched.during = lambda: sched.stm.req_to_blocks.update(a=manager.allocate_slots(a, 2), c=manager.allocate_slots(c, 1))
    so = sched.schedule("x", k=1)
    assert so is sched.output and v.calls[0] == ("schedule", ("x",), {"k": 1}, None)
    assert sched.update_from_output(so, "out") == "outputs" and v.calls[-1] == ("update_from_output", True, "out", None)
    pool.free_blocks(iter(sched.stm.req_to_blocks["a"]))
    assert heard == [RangeAllocated((1, 2), "a", step=0), RangeAllocated((3,), "c", step=0),
                     RangeWritten(1, 0, 16, 0), RangeWritten(2, 0, 4, 0), RangeWritten(3, 0, 3, 0),
                     StepBegan(0, ("a", "c"), 23), StepEnded(0), RangeFreed((1, 2), step=0)]
    assert [x.block_id for x in pool.free_block_queue.q] == [2, 1, 4, 5]


def test_the_patches_are_one_set_for_every_attachment_and_the_last_detach_takes_them_off(monkeypatch):
    v = kv_classes(fake_vllm.install(monkeypatch))
    messages.close()
    heard: list = []
    off = messages.current().subscribe("test", "memory", heard.append)
    d1 = kv_blocks.attach()
    entries = [e for e in hooks.live(kv_blocks.TAG)]
    assert {(e.owner, e.name) for e in entries} == set(v.sites)
    d2 = kv_blocks.attach()
    assert len(hooks.live(kv_blocks.TAG)) == len(entries)
    pool = v.pool()
    pool.get_new_blocks(1)
    d1()
    d1()
    pool.get_new_blocks(1)
    assert heard == [RangeAllocated((1,), step=0), RangeAllocated((2,), step=0)] and hooks.live(kv_blocks.TAG)
    d2()
    assert hooks.live(kv_blocks.TAG) == () and all(vars(cls)[n] is own for (cls, n), own in v.sites.items())
    off()
    messages.close()


def test_with_no_subscriber_the_hook_still_keeps_the_step(monkeypatch):
    v = kv_classes(fake_vllm.install(monkeypatch))
    messages.close()
    off, sched = kv_blocks.attach(), v.scheduler()
    try:
        sched.schedule()
        sched.schedule()
        assert messages.current().step == 1
    finally:
        off()
        messages.close()


@pytest.mark.parametrize("hold", [False, True])
def test_reuse_oldest_freed_puts_a_freed_block_at_the_free_queues_back(monkeypatch, subscribe, hold):
    """vLLM puts a freed block with no hash at the free queue's front; while a subscriber asks ReuseOldestFreed it goes
    to the back, also when vLLM was handed an iterator."""
    v = kv_classes(fake_vllm.install(monkeypatch))
    subscribe()
    subscribe(*([ReuseOldestFreed()] if hold else []))
    pool = v.pool()
    got = pool.get_new_blocks(3)
    assert pool.free_blocks(iter(got[:2])) is None and v.calls[-1] == ("free_blocks", "list")
    assert [x.block_id for x in pool.free_block_queue.q] == ([4, 5, 1, 2] if hold else [2, 1, 4, 5])


def test_a_prefix_cache_hit_is_a_reused_allocation_and_a_block_in_use_is_none(monkeypatch, subscribe):
    v = kv_classes(fake_vllm.install(monkeypatch))
    heard = subscribe()
    pool = v.pool()
    got = pool.get_new_blocks(3)
    assert pool.touch([got[0], pool.blocks[0]]) == "touched" and heard[1:] == []
    pool.free_blocks(got[2:])
    assert pool.touch([got[2], got[0]]) == "touched"
    assert heard[1:] == [RangeFreed((3,), step=0), RangeAllocated((3,), reused=True, step=0)]


def test_a_block_held_resident_is_freed_at_the_first_step_it_no_longer_is(monkeypatch, subscribe):
    """vLLM's free of a block a scheme holds resident (`HoldResident`) is withheld, and the block freed, with its
    RangeFreed, when the scheduler begins the step the hold ends at."""
    v = kv_classes(fake_vllm.install(monkeypatch))
    heard = subscribe(HoldResident(2, 2))
    pool, sched = v.pool(), v.scheduler()
    got = pool.get_new_blocks(2)
    sched.schedule()
    pool.free_blocks(got)
    assert heard[-1] == RangeFreed((1,), step=0) and [x.block_id for x in pool.free_block_queue.q] == [1, 3, 4, 5]
    sched.schedule()
    assert heard[-1] == StepBegan(1, (), 0) and got[1].ref_cnt == 1
    sched.schedule()
    assert heard[-2:] == [RangeFreed((2,), step=2), StepBegan(2, (), 0)] and got[1].ref_cnt == 0
    assert [x.block_id for x in pool.free_block_queue.q] == [2, 1, 3, 4, 5]
