"""`Binder`: the store's hooks into one in-process vLLM engine (``VLLM_ENABLE_V1_MULTIPROCESSING=0``, so the scheduler and
the model runner share one `NodeMemory`), every patch through `verity_vllm.engine.hooks` and undone by its `Hooks`.

~~~text
GPUModelRunner.load_model           after: each weight tensor a region (kind weights, tier hbm), registered once
GPUModelRunner.initialize_kv_cache  after, unless profiling: the KV pool, a region per attention layer, every page filler
                                    (with kv=False, unregistered)
BlockPool.get_new_blocks            after: kv_alloc of the pages it handed out (with kv=False, none)
BlockPool.free_blocks               after: kv_free of the pages it freed (with kv=False, none), moved to the back of the free queue (FIFO
                                    reuse, so a page freed inside a window is reused as late as the pool allows)
BlockPool.touch                     refused: a prefix-cache hit would bring a page back with its KV (run without it)
Scheduler.schedule                  after: a verifier's message since the last pass ends the window, the next
                                    challenge begins one; then kv_append of the scheduled tokens' slots, which closes
                                    the last step's memory-map line (with its re-keys for the window just begun), and
                                    begin_pass over the allocated pages, tagged with the step (a pass still open from
                                    an earlier schedule is ended first)
Scheduler.update_from_output        before: end_pass, with the device reader's roots, and end_step (the step's KV is
                                    written by now, so a memory map takes its digests; the pages vLLM frees after
                                    this, in update_from_output, are freed at this step)
~~~

The step's own device work (a codec's ``encode_slots_`` in attention's KV write, a fused GEMM's receipts) is the
codec's and the reader's: the identity codec has none, so the binder's cost under it is the hooks' host time, which
`stats` reports per hook.
"""
from __future__ import annotations

import importlib
import importlib.util
import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from verity_vllm.engine.hooks import Hooks
from verity_vllm.protocols.remote_bandwidth.interface import IDENTITY, Append, Codec, PageLayout, Receipt, Segment, WindowRecord
from verity_vllm.protocols.remote_bandwidth.registry import KindReceipt, NodeMemory, RemoteBandwidthReceipt

if TYPE_CHECKING:
    from verity.core.protocols.remote_bandwidth.audit import Tiles

RUNNERS = ("vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu.model_runner")


class TorchBacking:
    """A region's bytes in a torch tensor on any device: ``flat``, a uint8 view, its first ``nbytes`` bytes."""

    def __init__(self, flat: Any, nbytes: int) -> None:
        self.flat, self.nbytes = flat, nbytes

    @classmethod
    def of_tensor(cls, t: Any, block_bytes: int) -> TorchBacking | None:
        """The whole blocks of a tensor whose elements fill one range of its storage, in storage order: a contiguous
        tensor, or a permuted one such as stock vLLM's FP8 weights (held transposed).  None if it holds less than one
        block or has gaps."""
        dims, step = sorted((s, n) for s, n in zip(t.stride(), t.shape) if n != 1), 1
        for s, n in dims:
            if s != step:
                return None
            step *= n
        es = t.element_size()
        if t.numel() * es < block_bytes:
            return None
        flat = storage_bytes(t.detach())[t.storage_offset() * es:(t.storage_offset() + t.numel()) * es]
        return cls(flat, flat.numel() // block_bytes * block_bytes)

    def read(self, offset: int, n: int) -> bytes:
        return self.flat[offset:offset + n].cpu().numpy().tobytes()

    def write(self, offset: int, data: bytes) -> None:
        torch = _torch()
        self.flat[offset:offset + len(data)].copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))


def storage_bytes(t: Any) -> Any:
    """A tensor's whole storage as a flat uint8 tensor."""
    torch = _torch()
    return torch.empty(0, dtype=torch.uint8, device=t.device).set_(t.untyped_storage())


class PagedBacking:
    """A KV layer's pages as one region, wherever its planes sit in a storage it may share with other layers: the
    region's bytes are its pages in order, a page's planes in order (`PageLayout`'s default strides), and plane ``k``
    of page ``p`` is ``plane_bytes`` at byte ``base + p * page_stride + k * plane_stride`` of ``flat``."""

    def __init__(self, flat: Any, base: int, physical: PageLayout) -> None:
        self.flat, self.base, self.physical = flat, base, physical
        self.plane = physical.plane_bytes
        self.nbytes = physical.pages * physical.planes * self.plane

    def where(self, offset: int) -> int:
        """The storage byte of region byte ``offset``."""
        q, x = divmod(offset, self.plane)
        p, k = divmod(q, self.physical.planes)
        ps, ks = self.physical.strides()
        return self.base + p * ps + k * ks + x

    def _runs(self, offset: int, n: int):
        while n > 0:
            c = min(n, self.plane - offset % self.plane)
            yield offset, self.where(offset), c
            offset, n = offset + c, n - c

    def read(self, offset: int, n: int) -> bytes:
        return b"".join(self.flat[w:w + c].cpu().numpy().tobytes() for _, w, c in self._runs(offset, n))

    def write(self, offset: int, data: bytes) -> None:
        torch = _torch()
        for o, w, c in self._runs(offset, len(data)):
            self.flat[w:w + c].copy_(torch.frombuffer(bytearray(data[o - offset:o - offset + c]), dtype=torch.uint8))


def _torch():
    import torch
    return torch


def page_layout(t: Any, pages: int, slots: int) -> PageLayout:
    """A KV layer's `PageLayout` from its tensor: one dimension of ``pages`` (the page stride), one of ``slots``, and
    K and V either two planes (a dimension of 2 outside the slot: of those, the one with the largest stride) or
    within the slot (vLLM's FlashAttention cache, a slot's K and V interleaved by head); the rest are a slot's bytes,
    contiguous within a page's plane."""
    elt, shape, dims = t.element_size(), tuple(t.shape), range(t.dim())
    dp = [d for d in dims if shape[d] == pages]
    ds = [d for d in dims if shape[d] == slots and d not in dp]
    if len(dp) != 1 or len(ds) != 1:
        raise ValueError(f"a KV cache of shape {shape}, strides {t.stride()} is not (pages, slots, ...) in an order "
                         f"the store can read ({pages} pages of {slots} slots)")
    twos = sorted((d for d in dims if shape[d] == 2 and d not in dp + ds and t.stride(d) > t.stride(ds[0])),
                  key=t.stride)
    d2 = twos[-1:]
    inner = [d for d in dims if d not in dp + d2]
    slot_bytes = elt * math.prod(shape[d] for d in inner if d != ds[0])
    span = elt * (1 + sum((shape[d] - 1) * t.stride(d) for d in inner))
    if span != slots * slot_bytes:
        raise ValueError(f"a KV page's plane is not contiguous: shape {shape}, strides {t.stride()}")
    planes = len(d2) + 1
    return PageLayout(pages, slots, slot_bytes, planes, t.stride(dp[0]) * elt,
                      t.stride(d2[0]) * elt if d2 else 0, t.stride(ds[0]) * elt == slot_bytes)


@dataclass
class WeightRegion:
    """A weight tensor to register: with ``encode``, the codec encodes it in place at registration (the reference
    path); without, it already holds the codec's encoding (identity, or a device codec encoded it at load)."""

    name: str
    tensor: Any
    source: str
    codec: Codec = IDENTITY
    receipt: Receipt | None = None
    encode: bool = False
    live: Any = None
    """The object the model holds (its parameter), whose storage the binder checks stays put (`Binder.moved`)."""


def model_weights(runner: Any) -> list[WeightRegion]:
    """Every parameter of the runner's model once (tied ones once), as the identity codec holds them."""
    model, seen, out = _unwrap(runner.model), set(), []
    for name, p in model.named_parameters():
        if p.numel() == 0 or p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())
        out.append(WeightRegion(name, p.data, f"param:{name}", live=p))
    return out


def _where(t: Any) -> tuple:
    return t.data_ptr(), tuple(t.shape), tuple(t.stride())


def _check_disjoint(starts: list[tuple[int, np.ndarray]], plane_bytes: int) -> None:
    """No two layers' planes overlap in a storage."""
    by: dict[int, list[np.ndarray]] = {}
    for key, xs in starts:
        by.setdefault(key, []).append(xs)
    for xs in by.values():
        a = np.sort(np.concatenate(xs))
        if (np.diff(a) < plane_bytes).any():
            raise ValueError("two KV layers' pages overlap in one storage")


def _unwrap(m: Any) -> Any:
    while not hasattr(m, "named_parameters") and hasattr(m, "runnable"):
        m = m.runnable
    return m


class Inbox:
    """The verifier's messages as they arrive: `put` (from any thread) each window's 32-byte challenge, `σ_0` first,
    then at each window's end the next one's or None, the close after the last (the driver's `AuditRun.start` and
    `AuditRun.boundary`).  Called, it hands over those not yet taken, in order."""

    def __init__(self) -> None:
        self._q: deque[bytes | None] = deque()

    def put(self, message: bytes | None) -> None:
        self._q.append(message)

    def __call__(self) -> list[bytes | None]:
        out = []
        while self._q:
            out.append(self._q.popleft())
        return out


class Every:
    """Messages on a fixed schedule, for a run with no verifier: ``challenge(t)`` at pass ``t · passes``, and with
    ``windows`` the close after the last window's passes."""

    def __init__(self, passes: int, challenge: Callable[[int], bytes], windows: int | None = None) -> None:
        if passes < 1:
            raise ValueError("a window is at least one pass")
        self.passes, self.challenge, self.windows, self._calls = passes, challenge, windows, 0

    def __call__(self) -> list[bytes | None]:
        t, i = divmod(self._calls, self.passes)
        self._calls += 1
        if i or (self.windows is not None and t > self.windows):
            return []
        return [None] if t == self.windows else [self.challenge(t)]


class Binder:
    """The store's hooks.  ``weights(runner)`` lists the weight regions after load (default `model_weights`);
    ``reader(segments)``, if given, returns the step's aligned segments' roots (by segment number), or the pass's root
    when it built the whole tree, for `end_pass`; ``fifo`` reuses freed pages in the order they were freed.
    ``messages()``, polled at each pass's start, returns the verifier's messages since the last poll (`Inbox`, or
    `Every` for a fixed schedule): each ends the open window, whose last pass is the one that just ended, and a
    challenge begins the next, the session's first chained from ``registration`` (the driver's
    `Registration.digest`).  Each window's record goes to ``on_window``; without a reader, a window's receipts are the
    reference's, every block read on the CPU.  With ``tiles`` (the driver's `audit.Tiles`), each receipt is bound to
    ``tiles.output(seed, j)``, PoUW's tile output before block ``j`` is read, and the default receipt is the driver's
    (`RemoteBandwidthReceipt`); without them, its default kind, sketch-sha512, of the block alone (`KindReceipt`).
    With ``passes=False`` the binder runs no pass: a remote session's backend runs them, and its windows (`verity_vllm.protocols.remote_bandwidth.remote.CommitSession`).
    With ``weight_blocks`` it registers weight tensors in order while their blocks fit that many, and counts the rest
    unregistered: a store small enough for a remote session's CPU passes (`remote.StoreBackend`).  With ``kv=False``
    the KV pool stays outside the registration (counted unregistered): its pages' hooks reach no store, and a step
    writes no registered block, so its `kv_append` only numbers the step; `register_filler` adds the store's filler."""

    def __init__(self, store: NodeMemory, *, block_bytes: int = 4096, kv_codec: Codec = IDENTITY,
                 receipt: Receipt | None = None, weights: Callable[[Any], Sequence[WeightRegion]] = model_weights,
                 reader: Callable[[Sequence[Segment]], dict[int, bytes] | bytes | None] | None = None, fifo: bool = True,
                 messages: Callable[[], Sequence[bytes | None]] | None = None, registration: bytes | None = None,
                 on_window: Callable[[WindowRecord], None] | None = None, tiles: Tiles | None = None,
                 passes: bool = True, weight_blocks: int | None = None, kv: bool = True) -> None:
        self.store, self.block_bytes, self.kv_codec, self.kv = store, block_bytes, kv_codec, kv
        self.receipt = receipt or (RemoteBandwidthReceipt() if tiles is not None else KindReceipt())
        self.weights, self.reader, self.fifo, self.tiles = weights, reader, fifo, tiles
        self.messages, self.registration, self.on_window, self.passes = messages, registration, on_window, passes
        self.weight_blocks = weight_blocks
        self.hooks: Hooks | None = None
        self.step = 0
        self.windows = 0
        self._in_window = False
        self._seed: bytes | None = None
        self._segments: Sequence[Segment] = ()
        self.unregistered: dict[str, int] = {}
        self.slots = 0
        self.tensors: list[tuple[str, Callable[[], Any], tuple]] = []
        self.model: Any = None
        self.filler: Any = None

    # --- registration -----------------------------------------------------------------------------------------------

    def on_load(self, runner: Any) -> None:
        room, self.model = self.weight_blocks, _unwrap(runner.model)
        for w in self.weights(runner):
            b = TorchBacking.of_tensor(w.tensor, self.block_bytes)
            nbytes = w.tensor.numel() * w.tensor.element_size()
            if b is None or (room is not None and b.nbytes // self.block_bytes > room):
                self.unregistered[w.name] = nbytes
                continue
            if room is not None:
                room -= b.nbytes // self.block_bytes
            if b.nbytes < nbytes:
                self.unregistered[f"{w.name}[tail]"] = nbytes - b.nbytes
            self.store.register(w.name, "hbm", "weights", b, block_bytes=self.block_bytes, codec=w.codec,
                                receipt=w.receipt or self.receipt, source=w.source, encode=w.encode)
            self._held(w.name, lambda t=w.tensor if w.live is None else w.live: t)

    def on_kv(self, runner: Any, kv_cache_config: Any) -> None:
        """Register the KV pool: each attention layer's pages a region (`PagedBacking`, so layers that share one
        storage are each their own pages); the storages' bytes no layer's page holds are counted unregistered."""
        caches = list(getattr(runner, "kv_caches", None) or ())
        if not caches:
            raise RuntimeError("the model runner bound no KV caches")
        slots = int(runner.cache_config.block_size)
        kbs = getattr(runner, "_kernel_block_sizes", None) or [slots]
        if any(int(k) != slots for k in kbs):
            raise NotImplementedError(f"kernel block sizes {kbs} other than the manager's {slots}")
        pages = int(kv_cache_config.num_blocks)
        if not self.kv:
            self.slots = slots
            storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in caches}
            self.unregistered["kv"] = sum(storages.values())
            return
        flats: dict[int, Any] = {}
        backings, starts, layout = [], [], None
        for i, t in enumerate(caches):
            self._held(f"kv[{i}]", lambda i=i: runner.kv_caches[i])
            phys = page_layout(t, pages, slots)
            key = t.untyped_storage().data_ptr()
            flat = flats.setdefault(key, storage_bytes(t))
            b = PagedBacking(flat, t.storage_offset() * t.element_size(), phys)
            ps, ks = phys.strides()
            starts.append((key, (b.base + np.arange(pages)[:, None] * ps + np.arange(phys.planes)[None, :] * ks).ravel()))
            backings.append(b)
            lay = PageLayout(pages, slots, phys.slot_bytes, phys.planes, slot_major=phys.slot_major)
            if layout not in (None, lay):
                raise NotImplementedError("KV layers of different page shapes")
            layout = lay
        _check_disjoint(starts, layout.plane_bytes)
        held = sum(f.numel() for f in flats.values()) - sum(b.nbytes for b in backings)
        if held:
            self.unregistered["kv[unpaged]"] = held
        self.slots = slots
        self.store.register_kv_pool("kv", backings, layout, block_bytes=self.block_bytes, codec=self.kv_codec,
                                    receipt=self.receipt, encode=False)

    def register_filler(self, nbytes: int, device: Any) -> None:
        """Register ``nbytes`` of zeros on ``device`` (whole blocks) as the store's filler, one region (kind filler,
        tier hbm) after the weights: memory the store holds that no step reads or writes."""
        torch, n = _torch(), nbytes // self.block_bytes
        if n < 1:
            raise ValueError(f"filler of {nbytes} bytes holds no {self.block_bytes}-byte block")
        if self.filler is not None:
            raise RuntimeError("the store's filler is registered")
        self.filler = torch.zeros(n * self.block_bytes, dtype=torch.uint8, device=device)
        self.store.register("filler", "hbm", "filler", TorchBacking(self.filler, self.filler.numel()),
                            block_bytes=self.block_bytes, codec=IDENTITY, receipt=self.receipt, source="filler:zeros",
                            encode=False)
        self._held("filler", lambda: self.filler)

    def _held(self, name: str, live: Callable[[], Any]) -> None:
        self.tensors.append((name, live, _where(live())))

    def moved(self) -> list[str]:
        """The registered tensors whose storage, size or strides are no longer those at registration: a reader that
        holds their device addresses (`benchmarks/remote_session/inplace.py`) would read memory the store left."""
        return [name for name, live, was in self.tensors if _where(live()) != was]

    # --- the scheduler's steps --------------------------------------------------------------------------------------

    def appends(self, sched: Any, so: Any) -> list[Append]:
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
                out.append(Append(blocks[pi].block_id, off, c))
                s += c
        return out

    def on_free(self, pool: Any, blocks: Sequence[Any]) -> None:
        """After `BlockPool.free_blocks`: the pages it freed go to filler, and (``fifo``) to the free queue's back."""
        freed = [b for b in blocks if b.ref_cnt == 0 and not b.is_null]
        if self.fifo and freed:
            for b in freed:
                pool.free_block_queue.remove(b)
            pool.free_block_queue.append_n(freed)
        if self.kv:
            self.store.kv_free([b.block_id for b in freed])

    def on_schedule(self, sched: Any, so: Any) -> None:
        if self.store.in_pass:
            self.end()
        for m in self.messages() if self.messages is not None else ():
            self.end_window()
            if m is not None:
                self.store.begin_window(m, None if self.windows else self.registration)
                self._in_window = True
        t = time.perf_counter()
        apps = self.appends(sched, so) if self.kv else []
        self.store.counters.add("appends", time.perf_counter() - t)
        s, self.step = self.step, self.step + 1
        self.store.kv_append(apps, s)
        if self.passes:
            ps = self.store.begin_pass(None, step=s)
            self._segments, self._seed = ps.segments, ps.seed

    def z(self, seed: bytes) -> Callable[[str, int], bytes]:
        """The tile outputs a pass under ``seed`` binds its receipts to, by region and region block."""
        tiles, region = self.tiles, self.store.region
        return lambda name, b: tiles.output(seed, region(name).first + b)

    def end(self) -> None:
        self.store.end_step()
        if self.store.in_pass:
            read = self.reader(self._segments) if self.reader else None
            z = self.z(self._seed) if self.tiles is not None and self._seed is not None else None
            self.store.end_pass(root=read) if isinstance(read, bytes) else self.store.end_pass(read, z)

    def end_window(self) -> None:
        """Close the open window (after its last pass), if one is."""
        self.end()
        if self._in_window:
            rec = self.store.end_window()
            self.windows += 1
            self._in_window = False
            if self.on_window is not None:
                self.on_window(rec)

    # --- install ----------------------------------------------------------------------------------------------------

    def install(self, hooks: Hooks | None = None) -> Binder:
        h = self.hooks = hooks or Hooks(tag="verity-store")
        binder = self
        for mod in RUNNERS:
            if importlib.util.find_spec(mod) is None:
                continue
            cls = importlib.import_module(mod).GPUModelRunner

            def load(orig):
                def load_model(runner, *a, **kw):
                    r = orig(runner, *a, **kw)
                    binder.on_load(runner)
                    return r
                return load_model

            def init_kv(orig):
                def initialize_kv_cache(runner, kv_cache_config, *a, **kw):
                    r = orig(runner, kv_cache_config, *a, **kw)
                    if not kw.get("is_profiling", a[0] if a else False):
                        binder.on_kv(runner, kv_cache_config)
                    return r
                return initialize_kv_cache

            h.wrap(cls, "load_model", load)
            h.wrap(cls, "initialize_kv_cache", init_kv)
        self._install_pool(h)
        self._install_scheduler(h)
        return self

    def _install_pool(self, h: Hooks) -> None:
        from vllm.v1.core.block_pool import BlockPool
        binder = self

        def get_new(orig):
            def get_new_blocks(pool, n):
                blocks = orig(pool, n)
                if binder.kv:
                    binder.store.kv_alloc([b.block_id for b in blocks])
                return blocks
            return get_new_blocks

        def free(orig):
            def free_blocks(pool, ordered):
                blocks = list(ordered)
                orig(pool, blocks)
                binder.on_free(pool, blocks)
            return free_blocks

        def touch(orig):
            def refuse(pool, blocks):
                if any(b.ref_cnt == 0 and not b.is_null for b in blocks):
                    raise NotImplementedError("the store's binder runs without prefix caching")
                return orig(pool, blocks)
            return refuse

        h.wrap(BlockPool, "get_new_blocks", get_new)
        h.wrap(BlockPool, "free_blocks", free)
        h.wrap(BlockPool, "touch", touch)

    def _install_scheduler(self, h: Hooks) -> None:
        from vllm.v1.core.sched.scheduler import Scheduler
        binder = self

        def schedule(orig):
            def wrapped(sched, *a, **kw):
                so = orig(sched, *a, **kw)
                binder.on_schedule(sched, so)
                return so
            return wrapped

        def update(orig):
            def update_from_output(sched, so, out):
                binder.end()
                return orig(sched, so, out)
            return update_from_output

        h.wrap(Scheduler, "schedule", schedule)
        h.wrap(Scheduler, "update_from_output", update)

    def uninstall(self) -> None:
        if self.hooks is not None:
            self.hooks.uninstall()

    def stats(self) -> dict[str, Any]:
        lay, c = self.store.layout(), self.store.counters
        kinds: dict[str, int] = {}
        for r in lay.regions:
            kinds[r.kind] = kinds.get(r.kind, 0) + r.blocks * r.block_bytes
        return {"regions": len(lay.regions), "blocks": lay.blocks, "block_bytes": self.block_bytes,
                "bytes_by_tier": lay.bytes_by_tier, "bytes_by_kind": kinds,
                "unregistered_bytes": sum(self.unregistered.values()), "unregistered": len(self.unregistered),
                "registration": self.store.registration().count, "steps": self.step, "windows": self.windows,
                "hooks": {k: {"calls": c.calls[k], "seconds": c.seconds[k], "us_per_call": 1e6 * c.seconds[k] / c.calls[k]}
                          for k in sorted(c.calls)}}
