"""The memory hook: vLLM's model runner, KV cache manager, block pool and scheduler patched once, through `hooks`,
emitting to the process's exchange (`messages.current`) its ``memory`` and ``steps`` events (`verity.hooks`) in the
order vLLM acts, and taking its schemes' `HoldResident` and `ReuseOldestFreed`: a model parameter is a registered
value, a KV block a range, a request an input, a scheduler step a step.  It names no scheme, and builds no event no
scheme reads.

~~~text
GPUModelRunner.load_model       after: RegisteredValues, each parameter of the loaded model once (tied ones once)
GPUModelRunner.initialize_kv_cache
                                after, unless profiling: RangesDeclared, the pool's blocks as ranges of the manager's
                                block size in slots, each layer's KV cache a store (refused where a kernel block size
                                is another)
KVCacheManager.allocate_slots   around: the request the pool's allocations inside it are for
BlockPool.get_new_blocks        after: RangeAllocated, at the step being scheduled, for that request
BlockPool.touch                 before: RangeAllocated reused, the blocks it takes back off the free queue (a
                                prefix-cache hit)
BlockPool.free_blocks           around: a block a scheme holds resident at this step is kept from vLLM's own, which
                                frees it at the first step it no longer is; then RangeFreed, the blocks left with no
                                reader, under ReuseOldestFreed first moved to the back of the free queue (vLLM puts a
                                freed block with no hash at its front)
Scheduler.schedule              around: the step's index taken, and the held blocks no longer resident freed, before
                                vLLM's own (its allocations are at that step); then a RangeWritten for each run of slots
                                a scheduled request writes in the step, and StepBegan
Scheduler.update_from_output    before: StepEnded, the step's KV written by now; the blocks vLLM frees inside it are
                                freed at this step
~~~

A step's RangeWritten come before its StepBegan, so a subscriber has the step's writes when it begins.  The patches are
on the classes: they go on with the first `attach`, before the engine is built, and come off with the last detach; the
step index starts at 0 when they go on.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from verity import hooks as H

from verity_vllm import messages
from verity_vllm.engine.hooks import Hooks

__all__ = ["attach", "ranges_declared", "registered_values", "writes"]

TAG = "kv-blocks"
RUNNERS = ("vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu.model_runner")
#: What a `RUNNERS` runner calls its kernel block sizes, one per KV cache group, set in `initialize_kv_cache`: Model Runner
#: V2's name (`vllm.v1.worker.gpu.model_runner`), then the first runner's (`vllm.v1.worker.gpu_model_runner`).
KERNEL_BLOCK_SIZES = ("kernel_block_sizes", "_kernel_block_sizes")


def attach() -> Callable[[], None]:
    """Put the patches on (once for every attachment); returns the detach, the last of which takes them off."""
    return _HOOK.attach()


def registered_values(runner: Any, step: int = 0) -> H.RegisteredValues:
    """Every parameter of the runner's model once (tied ones once), by its name, as the model holds it."""
    model, seen, out = runner.model, set(), []
    while not hasattr(model, "named_parameters") and hasattr(model, "runnable"):
        model = model.runnable
    for name, p in model.named_parameters():
        if p.numel() == 0 or p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())
        out.append((name, p.data))
    return H.RegisteredValues(tuple(out), step)


def ranges_declared(runner: Any, kv_cache_config: Any, step: int = 0) -> H.RangesDeclared:
    """The KV pool as the runner bound it: its blocks, each the manager's block size in slots, a slot all of a layer's
    KV for one token in one plane (where K and V sit is the store's to read), each layer's cache a store."""
    caches = tuple(getattr(runner, "kv_caches", None) or ())
    if not caches:
        raise RuntimeError("the model runner bound no KV caches")
    slots = int(runner.cache_config.block_size)
    kbs = next((v for v in (getattr(runner, name, None) for name in KERNEL_BLOCK_SIZES) if v), None)
    if kbs is None:
        raise RuntimeError(f"the model runner names no kernel block sizes ({' or '.join(KERNEL_BLOCK_SIZES)}) to hold to "
                           f"the manager's {slots}")
    if any(int(k) != slots for k in kbs):
        raise NotImplementedError(f"kernel block sizes {kbs} other than the manager's {slots}")
    pages, t = int(kv_cache_config.num_blocks), caches[0]
    return H.RangesDeclared(pages, slots, t.numel() * t.element_size() // (pages * slots), 1, caches, step)


def writes(sched: Any, so: Any, step: int) -> tuple[H.RangeWritten, ...]:
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
            out.append(H.RangeWritten(blocks[pi].block_id, off, c, step))
            s += c
    return tuple(out)


class _Hook:
    def __init__(self) -> None:
        self.attached = 0
        self.hooks: Hooks | None = None
        self.step = self.next = 0
        self.request: str | None = None
        self.held: list[tuple[Any, Any, Callable[..., Any]]] = []

    def attach(self) -> Callable[[], None]:
        if self.hooks is None:
            self.install()
        self.attached += 1
        done = False

        def detach() -> None:
            nonlocal done
            if done:
                return
            done = True
            self.attached -= 1
            if not self.attached and self.hooks is not None:
                hooks, self.hooks = self.hooks, None
                hooks.uninstall()
        return detach

    def freed(self, pool: Any, blocks: Sequence[Any], ex: messages.Exchange) -> None:
        freed = [b for b in blocks if b.ref_cnt == 0 and not b.is_null]
        if not freed:
            return
        if ex.oldest_freed_first():
            for b in freed:
                pool.free_block_queue.remove(b)
            pool.free_block_queue.append_n(freed)
        if ex.wants("memory"):
            ex.emit(H.RangeFreed(tuple(b.block_id for b in freed), step=self.step))

    def release(self, ex: messages.Exchange) -> None:
        """Free the held blocks no scheme holds resident at this step any more."""
        due = [x for x in self.held if not ex.resident(x[1].block_id, self.step)]
        self.held = [x for x in self.held if not any(x is d for d in due)]
        for pool, b, free in due:
            free(pool, [b])
            self.freed(pool, [b], ex)

    def install(self) -> None:
        from vllm.v1.core.block_pool import BlockPool
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.core.sched.scheduler import Scheduler

        import importlib
        import importlib.util

        h, hook = Hooks(tag=TAG), self
        self.hooks, self.step, self.next, self.request, self.held = h, 0, 0, None, []

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
                if blocks and (ex := messages.current()).wants("memory"):
                    ex.emit(H.RangeAllocated(tuple(b.block_id for b in blocks), hook.request, step=hook.step))
                return blocks
            return get_new_blocks

        def touch(orig):
            def touched(pool, blocks):
                back = tuple(b.block_id for b in blocks if b.ref_cnt == 0 and not b.is_null)
                if back and (ex := messages.current()).wants("memory"):
                    ex.emit(H.RangeAllocated(back, hook.request, reused=True, step=hook.step))
                return orig(pool, blocks)
            return touched

        def free(orig):
            def free_blocks(pool, ordered):
                blocks, ex = list(ordered), messages.current()
                keep = [b for b in blocks if not b.is_null and ex.resident(b.block_id, hook.step)]
                if keep:
                    hook.held += [(pool, b, orig) for b in keep]
                    blocks = [b for b in blocks if not any(b is k for k in keep)]
                orig(pool, blocks)
                hook.freed(pool, blocks, ex)
            return free_blocks

        def schedule(orig):
            def scheduled(sched, *a, **kw):
                hook.step, hook.next = hook.next, hook.next + 1
                ex = messages.current()
                hook.release(ex)
                so = orig(sched, *a, **kw)
                if ex.wants("memory"):
                    for w in writes(sched, so, hook.step):
                        ex.emit(w)
                if ex.wants("steps"):
                    ex.emit(H.StepBegan(hook.step, tuple(so.num_scheduled_tokens),
                                        sum(int(x) for x in so.num_scheduled_tokens.values())))
                else:
                    ex.step = hook.step
                return so
            return scheduled

        def update(orig):
            def update_from_output(sched, so, out):
                if (ex := messages.current()).wants("steps"):
                    ex.emit(H.StepEnded(hook.step))
                return orig(sched, so, out)
            return update_from_output

        def load(orig):
            def load_model(runner, *a, **kw):
                r = orig(runner, *a, **kw)
                if (ex := messages.current()).wants("memory"):
                    ex.emit(registered_values(runner, hook.step))
                return r
            return load_model

        def init_kv(orig):
            def initialize_kv_cache(runner, kv_cache_config, *a, **kw):
                r = orig(runner, kv_cache_config, *a, **kw)
                if not kw.get("is_profiling", a[0] if a else False) and (ex := messages.current()).wants("memory"):
                    ex.emit(ranges_declared(runner, kv_cache_config, hook.step))
                return r
            return initialize_kv_cache

        for mod in RUNNERS:
            if importlib.util.find_spec(mod) is not None:
                cls = importlib.import_module(mod).GPUModelRunner
                h.wrap(cls, "load_model", load)
                h.wrap(cls, "initialize_kv_cache", init_kv)
        h.wrap(KVCacheManager, "allocate_slots", allocate)
        h.wrap(BlockPool, "get_new_blocks", get_new)
        h.wrap(BlockPool, "touch", touch)
        h.wrap(BlockPool, "free_blocks", free)
        h.wrap(Scheduler, "schedule", schedule)
        h.wrap(Scheduler, "update_from_output", update)


_HOOK = _Hook()
