"""The KV block hook: vLLM's KV cache manager, block pool and scheduler patched once, through `hooks`, for every
subscriber, each of which hears them as `verity.hooks`' ``memory`` and ``steps`` events in the order vLLM acts: a KV
block is a range, a request an input, a scheduler step a step.  It names no scheme.

~~~text
KVCacheManager.allocate_slots   around: the request the pool's allocations inside it are for
BlockPool.get_new_blocks        after: RangeAllocated, at the step being scheduled, for that request
BlockPool.touch                 before: RangeAllocated reused, the blocks it takes back off the free queue (a
                                prefix-cache hit)
BlockPool.free_blocks           after: RangeFreed, the blocks left with no reader; while a subscriber asks
                                ReuseOldestFreed, first moved to the back of the free queue (vLLM puts a freed block with
                                no hash at its front)
Scheduler.schedule              around: the step's index taken before vLLM's own (its allocations are at that step),
                                then a RangeWritten for each run of slots a scheduled request writes in the step, and
                                StepBegan
Scheduler.update_from_output    before: StepEnded, the step's KV written by now; the blocks vLLM frees inside it are
                                freed at this step
~~~

A step's RangeWritten come before its StepBegan, so a subscriber has the step's writes when it begins.  The patches are
on the classes, so a subscriber attaches before the engine is built; they go on with the first subscriber and come off
with the last, and the step index starts at 0 when they go on.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from verity.hooks import RangeAllocated, RangeFreed, RangeWritten, ReuseOldestFreed, StepBegan, StepEnded

from verity_vllm.engine.hooks import Hooks

__all__ = ["Event", "REQUESTS", "Subscription", "subscribe", "writes"]

TAG = "kv-blocks"

type Event = RangeAllocated | RangeFreed | RangeWritten | StepBegan | StepEnded
#: the requests a subscriber may file with the hook
REQUESTS = (ReuseOldestFreed,)


@dataclass(eq=False)
class Subscription:
    """One subscriber: ``on`` hears every event, and ``requests`` are what it filed with the hook (`REQUESTS`)."""

    on: Callable[[Event], None]
    requests: tuple[ReuseOldestFreed, ...] = ()
    live: bool = True

    def cancel(self) -> None:
        """Stop hearing events; the last subscription's cancel takes the patches off.  Idempotent."""
        if self.live:
            self.live = False
            _HOOK.leave(self)


def subscribe(on: Callable[[Event], None], *requests: ReuseOldestFreed) -> Subscription:
    """Hear vLLM's KV block events from now on; the first subscription puts the patches on.  Refuses a request the hook
    doesn't take."""
    if bad := [r for r in requests if not isinstance(r, REQUESTS)]:
        raise TypeError(f"the KV block hook takes {[r.__name__ for r in REQUESTS]}, not {bad}")
    return _HOOK.join(Subscription(on, tuple(requests)))


def writes(sched: Any, so: Any, step: int) -> tuple[RangeWritten, ...]:
    """The slots each request in ``so`` (vLLM's scheduler output) writes in step ``step``: ``[computed - n, computed)``
    of its blocks, the scheduler having counted the step's tokens as computed already."""
    stm = sched.kv_cache_manager.coordinator.single_type_managers[0]
    out, slots = [], int(stm.block_size)
    for rid, n in so.num_scheduled_tokens.items():
        end = int(sched.requests[rid].num_computed_tokens)
        blocks, s = stm.req_to_blocks[rid], end - int(n)
        while s < end:
            pi, off = divmod(s, slots)
            c = min(slots - off, end - s)
            out.append(RangeWritten(blocks[pi].block_id, off, c, step))
            s += c
    return tuple(out)


class _Hook:
    def __init__(self) -> None:
        self.subs: list[Subscription] = []
        self.hooks: Hooks | None = None
        self.step = self.next = 0
        self.request: str | None = None

    def join(self, sub: Subscription) -> Subscription:
        if self.hooks is None:
            self.install()
        self.subs.append(sub)
        return sub

    def leave(self, sub: Subscription) -> None:
        self.subs = [s for s in self.subs if s is not sub]
        if not self.subs and self.hooks is not None:
            hooks, self.hooks = self.hooks, None
            hooks.uninstall()

    def emit(self, e: Event) -> None:
        for s in tuple(self.subs):
            s.on(e)

    def freed(self, pool: Any, blocks: Sequence[Any]) -> None:
        freed = [b for b in blocks if b.ref_cnt == 0 and not b.is_null]
        if not freed:
            return
        if any(isinstance(r, ReuseOldestFreed) for s in self.subs for r in s.requests):
            for b in freed:
                pool.free_block_queue.remove(b)
            pool.free_block_queue.append_n(freed)
        self.emit(RangeFreed(tuple(b.block_id for b in freed), step=self.step))

    def install(self) -> None:
        from vllm.v1.core.block_pool import BlockPool
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.core.sched.scheduler import Scheduler

        h, hook = Hooks(tag=TAG), self
        self.hooks, self.step, self.next, self.request = h, 0, 0, None

        def allocate(orig):
            def allocate_slots(manager, request, *a, **kw):
                hook.request = request.request_id
                try:
                    return orig(manager, request, *a, **kw)
                finally:
                    hook.request = None
            return allocate_slots

        def get_new(orig):
            def get_new_blocks(pool, n):
                blocks = orig(pool, n)
                if blocks:
                    hook.emit(RangeAllocated(tuple(b.block_id for b in blocks), hook.request, step=hook.step))
                return blocks
            return get_new_blocks

        def touch(orig):
            def touched(pool, blocks):
                if back := tuple(b.block_id for b in blocks if b.ref_cnt == 0 and not b.is_null):
                    hook.emit(RangeAllocated(back, hook.request, reused=True, step=hook.step))
                return orig(pool, blocks)
            return touched

        def free(orig):
            def free_blocks(pool, ordered):
                blocks = list(ordered)
                orig(pool, blocks)
                hook.freed(pool, blocks)
            return free_blocks

        def schedule(orig):
            def scheduled(sched, *a, **kw):
                hook.step, hook.next = hook.next, hook.next + 1
                so = orig(sched, *a, **kw)
                for w in writes(sched, so, hook.step):
                    hook.emit(w)
                n = so.num_scheduled_tokens
                hook.emit(StepBegan(hook.step, tuple(n), sum(int(x) for x in n.values())))
                return so
            return scheduled

        def update(orig):
            def update_from_output(sched, so, out):
                hook.emit(StepEnded(hook.step))
                return orig(sched, so, out)
            return update_from_output

        h.wrap(KVCacheManager, "allocate_slots", allocate)
        h.wrap(BlockPool, "get_new_blocks", get_new)
        h.wrap(BlockPool, "touch", touch)
        h.wrap(BlockPool, "free_blocks", free)
        h.wrap(Scheduler, "schedule", schedule)
        h.wrap(Scheduler, "update_from_output", update)


_HOOK = _Hook()
