"""The served run's ``kv`` hook on one in-process vLLM engine (``VLLM_ENABLE_V1_MULTIPROCESSING=0``): what vLLM's model
runner, block pool and scheduler do, as the events of `verity.protocols.resource_accounting.scheme`, handed to one
subscriber. Every patch goes through `verity_vllm.engine.hooks` and is undone by its `Hooks`.

~~~text
GPUModelRunner.load_model           after: `Loaded`, each of the model's weight tensors once
GPUModelRunner.initialize_kv_cache  after, unless profiling: `KvPool`, a storage handle an attention layer
BlockPool.get_new_blocks            after: `KvBlocks` allocated, the pages it handed out
BlockPool.free_blocks               after: `KvBlocks` freed, the pages left with no reader; with ``fifo`` they go to the
                                    back of the free queue, so a freed page is reused as late as the pool allows
BlockPool.touch                     before: `KvBlocks` touched, the freed pages a prefix-cache hit would bring back
                                    with their KV, when there are any (a subscriber that can't take one raises)
Scheduler.schedule                  after: `Step` begins, with the slots the scheduled tokens write (none without
                                    ``writes``)
Scheduler.update_from_output        before: `Step` ends (its KV is written by now; the pages vLLM frees after this are
                                    freed at this step)
~~~
"""
from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable, Sequence
from typing import Any

from verity.protocols.resource_accounting.scheme import KvBlocks, KvPool, KvWrite, Loaded, Step

from verity_vllm.engine.hooks import Hooks

__all__ = ["KERNEL_BLOCK_SIZES", "RUNNERS", "KvEvents", "loaded"]

RUNNERS = ("vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu.model_runner")
#: What a `RUNNERS` runner calls its kernel block sizes, one per KV cache group, set in `initialize_kv_cache`: Model Runner
#: V2's name (`vllm.v1.worker.gpu.model_runner`), then the first runner's (`vllm.v1.worker.gpu_model_runner`).
KERNEL_BLOCK_SIZES = ("kernel_block_sizes", "_kernel_block_sizes")

type Event = Loaded | KvPool | KvBlocks | Step


def loaded(runner: Any) -> Loaded:
    """Every parameter of the runner's model once (tied ones once, under the first name), none empty."""
    model, seen, out = _unwrap(runner.model), set(), []
    for name, p in model.named_parameters():
        if p.numel() == 0 or p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())
        out.append((name, p.data))
    return Loaded(tuple(out))


def _unwrap(m: Any) -> Any:
    while not hasattr(m, "named_parameters") and hasattr(m, "runnable"):
        m = m.runnable
    return m


class KvEvents:
    """The hook: ``sink`` takes each event as it happens, and what it raises is raised in vLLM's call. ``fifo`` reuses
    freed pages in the order they were freed; without ``writes`` a step's slots are not read off the scheduler."""

    def __init__(self, sink: Callable[[Event], None], *, fifo: bool = False, writes: bool = True) -> None:
        self.sink, self.fifo, self.writes = sink, fifo, writes
        self.hooks: Hooks | None = None
        self.step = 0
        self.slots = 0

    def on_load(self, runner: Any) -> None:
        self.sink(loaded(runner))

    def on_kv(self, runner: Any, kv_cache_config: Any) -> None:
        caches = list(getattr(runner, "kv_caches", None) or ())
        if not caches:
            raise RuntimeError("the model runner bound no KV caches")
        slots = int(runner.cache_config.block_size)
        kbs = next((v for v in (getattr(runner, name, None) for name in KERNEL_BLOCK_SIZES) if v), None)
        if kbs is None:
            raise RuntimeError(f"the model runner names no kernel block sizes ({' or '.join(KERNEL_BLOCK_SIZES)}) to hold to "
                               f"the manager's {slots}")
        if any(int(k) != slots for k in kbs):
            raise NotImplementedError(f"kernel block sizes {kbs} other than the manager's {slots}")
        self.slots = slots
        self.sink(KvPool(int(kv_cache_config.num_blocks), slots, None, None, tuple(caches)))

    def writes_of(self, sched: Any, so: Any) -> tuple[KvWrite, ...]:
        """The slots each scheduled request writes this step: ``[computed - n, computed)`` of its pages (the
        scheduler has already counted the step's tokens as computed)."""
        stm = sched.kv_cache_manager.coordinator.single_type_managers[0]
        out, slots = [], self.slots
        for rid, n in so.num_scheduled_tokens.items():
            end = int(sched.requests[rid].num_computed_tokens)
            blocks, s = stm.req_to_blocks[rid], end - int(n)
            while s < end:
                pi, off = divmod(s, slots)
                c = min(slots - off, end - s)
                out.append(KvWrite(blocks[pi].block_id, off, c))
                s += c
        return tuple(out)

    def on_free(self, pool: Any, blocks: Sequence[Any]) -> None:
        freed = [b for b in blocks if b.ref_cnt == 0 and not b.is_null]
        if self.fifo and freed:
            for b in freed:
                pool.free_block_queue.remove(b)
            pool.free_block_queue.append_n(freed)
        self.sink(KvBlocks("freed", tuple(b.block_id for b in freed)))

    def on_schedule(self, sched: Any, so: Any) -> None:
        s, self.step = self.step, self.step + 1
        self.sink(Step(s, self.writes_of(sched, so) if self.writes else ()))

    def end(self) -> None:
        self.sink(Step(self.step - 1, ended=True))

    def install(self, hooks: Hooks | None = None) -> KvEvents:
        h = self.hooks = hooks or Hooks(tag="verity-kv")
        events = self
        for mod in RUNNERS:
            if importlib.util.find_spec(mod) is None:
                continue
            cls = importlib.import_module(mod).GPUModelRunner

            def load(orig):
                def load_model(runner, *a, **kw):
                    r = orig(runner, *a, **kw)
                    events.on_load(runner)
                    return r
                return load_model

            def init_kv(orig):
                def initialize_kv_cache(runner, kv_cache_config, *a, **kw):
                    r = orig(runner, kv_cache_config, *a, **kw)
                    if not kw.get("is_profiling", a[0] if a else False):
                        events.on_kv(runner, kv_cache_config)
                    return r
                return initialize_kv_cache

            h.wrap(cls, "load_model", load)
            h.wrap(cls, "initialize_kv_cache", init_kv)
        self._install_pool(h)
        self._install_scheduler(h)
        return self

    def _install_pool(self, h: Hooks) -> None:
        from vllm.v1.core.block_pool import BlockPool
        events = self

        def get_new(orig):
            def get_new_blocks(pool, n):
                blocks = orig(pool, n)
                events.sink(KvBlocks("allocated", tuple(b.block_id for b in blocks)))
                return blocks
            return get_new_blocks

        def free(orig):
            def free_blocks(pool, ordered):
                blocks = list(ordered)
                orig(pool, blocks)
                events.on_free(pool, blocks)
            return free_blocks

        def touch(orig):
            def touched(pool, blocks):
                if back := tuple(b.block_id for b in blocks if b.ref_cnt == 0 and not b.is_null):
                    events.sink(KvBlocks("touched", back))
                return orig(pool, blocks)
            return touched

        h.wrap(BlockPool, "get_new_blocks", get_new)
        h.wrap(BlockPool, "free_blocks", free)
        h.wrap(BlockPool, "touch", touch)

    def _install_scheduler(self, h: Hooks) -> None:
        from vllm.v1.core.sched.scheduler import Scheduler
        events = self

        def schedule(orig):
            def wrapped(sched, *a, **kw):
                so = orig(sched, *a, **kw)
                events.on_schedule(sched, so)
                return so
            return wrapped

        def update(orig):
            def update_from_output(sched, so, out):
                events.end()
                return orig(sched, so, out)
            return update_from_output

        h.wrap(Scheduler, "schedule", schedule)
        h.wrap(Scheduler, "update_from_output", update)

    def uninstall(self) -> None:
        if self.hooks is not None:
            self.hooks.uninstall()
