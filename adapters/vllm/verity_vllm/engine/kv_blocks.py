"""The memory hook: vLLM's model runner, KV cache manager, block pool and scheduler patched once, through `hooks`,
emitting to the process's exchange (`messages.current`) its ``memory`` events and `StepComputed` (`verity.hooks`) in the order vLLM acts,
and taking its schemes' `ReuseOldestFreed` and `ReleaseValue` (after vLLM's ``load_model``, restored at
the last detach): a model parameter is a registered value, a KV block a
range, a request an input.  Each event is at the exchange's step, whose `StepBegan` and `StepEnded` are the admission
hook's (`engine.admission`): inside ``schedule``, the step being scheduled (`Exchange.next_step`), else the last one
begun.  It names no scheme, and builds no event no scheme reads.

~~~text
GPUModelRunner.load_model       after: the linear site attached (`vllm_site`) where a scheme computes blocks; then
                                RegisteredValues, each parameter of the loaded model once (tied ones once), each
                                linear layer's weight an operand with its block key
GPUModelRunner.execute_model    after: StepComputed, for a step that computes; the forward has run, and its outputs
                                are not yet sampled or read
GPUModelRunner.initialize_kv_cache
                                after, unless profiling: RangesDeclared, the pool's blocks as ranges of the manager's
                                block size in slots, each layer's KV cache a store (refused where a kernel block size
                                is another)
KVCacheManager.allocate_slots   around: the request the pool's allocations inside it are for
BlockPool.get_new_blocks        after: RangeAllocated, at the step being scheduled, for that request
BlockPool.touch                 before: RangeAllocated reused, the blocks it takes back off the free queue (a
                                prefix-cache hit)
BlockPool.free_blocks           after: RangeFreed, the blocks left with no reader, under ReuseOldestFreed first moved
                                to the back of the free queue (vLLM puts a freed block with no hash at its front)
Scheduler.schedule              around: the step being scheduled is the one its allocations are at; then a RangeWritten
                                for each run of slots a scheduled request writes in the step
~~~

The admission hook is on the scheduler instance, around these class patches, so a step's RangeWritten come before its
StepBegan and a subscriber has the step's writes when it begins; the blocks vLLM frees in ``update_from_output`` are
freed at the step begun last.  The patches are on the classes: they go on with the first `attach`, before the engine is
built, and come off with the last detach.
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
#: the scheduler outputs whose step `StepComputed` names, the latest kept: more than the steps vLLM has in flight
SCHEDULED = 4


def attach() -> Callable[[], None]:
    """Put the patches on (once for every attachment); returns the detach, the last of which takes them off."""
    return _HOOK.attach()


def model_of(runner: Any) -> Any:
    model = runner.model
    while not hasattr(model, "named_parameters") and hasattr(model, "runnable"):
        model = model.runnable
    return model


def registered_values(runner: Any, step: int = 0, sites: Sequence[tuple[Any, Any]] = ()) -> H.RegisteredValues:
    """Every parameter of the runner's model once (tied ones once), by its name, as the model holds it; the weight of
    each of ``sites`` (`vllm_site.discover`'s) an operand, under the name its storage is registered by, with its site's
    block key."""
    model, kept, out = model_of(runner), {}, []
    for name, p in model.named_parameters():
        if p.numel() == 0 or p.data_ptr() in kept:
            continue
        kept[p.data_ptr()] = name
        out.append((name, p.data))
    operands = {kept[layer.weight.data_ptr()]: site.weight_id for site, layer in sites
                if layer.weight.data_ptr() in kept}
    return H.RegisteredValues(tuple(out), step, operands)


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
        self.scheduling = False
        self.request: str | None = None
        self.released: list[tuple[list[Any], H.ReleaseValue]] = []
        self.scheduled: list[tuple[Any, int]] = []

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
                self.restore_values()
        return detach

    def release_values(self, model: Any, ex: messages.Exchange) -> None:
        """Release each `ReleaseValue`'s parameter: every parameter of ``model`` on its storage holds an empty tensor."""
        asks = [r for _, r in ex.requests if isinstance(r, H.ReleaseValue)]
        params = dict(model.named_parameters(remove_duplicate=False)) if asks else {}
        for r in asks:
            if r.value not in params:
                raise RuntimeError(f"a scheme asks registered value {r.value} released, and the model has no such parameter")
            ptr = params[r.value].data_ptr()
            held = [p for p in {id(p): p for p in params.values()}.values() if p.data_ptr() == ptr]
            for p in held:
                p.data = p.data.new_empty(0)
            self.released.append((held, r))

    def restore_values(self) -> None:
        released, self.released = self.released, []
        for held, r in released:
            t = r.restore()
            for p in held:
                p.data = t

    def at(self, ex: messages.Exchange) -> int:
        return ex.next_step if self.scheduling else ex.step

    def freed(self, pool: Any, blocks: Sequence[Any], ex: messages.Exchange) -> None:
        freed = [b for b in blocks if b.ref_cnt == 0 and not b.is_null]
        if not freed:
            return
        if ex.oldest_freed_first():
            for b in freed:
                pool.free_block_queue.remove(b)
            pool.free_block_queue.append_n(freed)
        if ex.wants("memory"):
            ex.emit(H.RangeFreed(tuple(b.block_id for b in freed), step=self.at(ex)))

    def install(self) -> None:
        from vllm.v1.core.block_pool import BlockPool
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.core.sched.scheduler import Scheduler

        import importlib
        import importlib.util

        h, hook = Hooks(tag=TAG), self
        self.hooks, self.scheduling, self.request, self.released, self.scheduled = h, False, None, [], []

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
                    ex.emit(H.RangeAllocated(tuple(b.block_id for b in blocks), hook.request, step=hook.at(ex)))
                return blocks
            return get_new_blocks

        def touch(orig):
            def touched(pool, blocks):
                back = tuple(b.block_id for b in blocks if b.ref_cnt == 0 and not b.is_null)
                if back and (ex := messages.current()).wants("memory"):
                    ex.emit(H.RangeAllocated(back, hook.request, reused=True, step=hook.at(ex)))
                return orig(pool, blocks)
            return touched

        def free(orig):
            def free_blocks(pool, ordered):
                blocks = list(ordered)
                orig(pool, blocks)
                hook.freed(pool, blocks, messages.current())
            return free_blocks

        def schedule(orig):
            def scheduled(sched, *a, **kw):
                ex = messages.current()
                hook.scheduling = True
                try:
                    so = orig(sched, *a, **kw)
                    hook.scheduled = [*hook.scheduled[-(SCHEDULED - 1):], (so, ex.next_step)]
                    if ex.wants("memory"):
                        for w in writes(sched, so, ex.next_step):
                            ex.emit(w)
                finally:
                    hook.scheduling = False
                return so
            return scheduled

        def load(orig):
            def load_model(runner, *a, **kw):
                from verity_vllm.protocols.linear import vllm_site

                r = orig(runner, *a, **kw)
                ex = messages.current()
                computes = ex.computes() or ex.wants("blocks")
                sites = vllm_site.discover(model_of(runner)) if computes or ex.wants("memory") else []
                if computes:
                    vllm_site.attach(sites, h)
                if ex.wants("memory"):
                    ex.emit(registered_values(runner, ex.step, sites))
                hook.release_values(model_of(runner), ex)
                return r
            return load_model

        def execute(orig):
            def execute_model(runner, so, *a, **kw):
                r = orig(runner, so, *a, **kw)
                step = next((s for o, s in reversed(hook.scheduled) if o is so), None)
                if step is not None and so.total_num_scheduled_tokens and (ex := messages.current()).wants("steps"):
                    ex.emit(H.StepComputed(step))
                return r
            return execute_model

        def init_kv(orig):
            def initialize_kv_cache(runner, kv_cache_config, *a, **kw):
                r = orig(runner, kv_cache_config, *a, **kw)
                if not kw.get("is_profiling", a[0] if a else False) and (ex := messages.current()).wants("memory"):
                    ex.emit(ranges_declared(runner, kv_cache_config, ex.step))
                return r
            return initialize_kv_cache

        for mod in RUNNERS:
            if importlib.util.find_spec(mod) is not None:
                cls = importlib.import_module(mod).GPUModelRunner
                h.wrap(cls, "load_model", load)
                h.wrap(cls, "initialize_kv_cache", init_kv)
                h.wrap(cls, "execute_model", execute)
        h.wrap(KVCacheManager, "allocate_slots", allocate)
        h.wrap(BlockPool, "get_new_blocks", get_new)
        h.wrap(BlockPool, "touch", touch)
        h.wrap(BlockPool, "free_blocks", free)
        h.wrap(Scheduler, "schedule", schedule)


_HOOK = _Hook()
