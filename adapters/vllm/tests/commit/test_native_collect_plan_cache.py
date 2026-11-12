"""The collector's plans and the worker's plan metas in `native_collect` across commitment runs: bounded, and the roots unchanged.

The collector keys each step's plan by its geometry (`plan_key_of`: every request's KV length and new tokens), so each decode
step of a growing KV length learns a plan of its own, and the worker builds one `TensorMeta` per staged tensor for it. The
served committer of the network traces commits one run per window and resets between them; a cache that kept every plan
grows with every committed step for the life of the process, and every full collection of the serving engine's heap walks
it. Here the windowed worker runs on the CPU: the C++ collector is restated in Python from `native_collect.cpp` (a plan per key,
4-byte offsets with zeroed gaps, a hit followed entry by entry, a miss truncated and relearned, the version bumped on every
change, the step's packed stream handed to the worker a window at a time through the ring's slots) and `native_plans.h` (the
plans the run that ended did not use dropped at `end_run`, a plan created after a drop starting above every dropped version;
the header itself is tested in `test_native_plans.py`), the rows-sha512 kernels (`native_rows.cu`, which `test_rows_cuda.py`
holds to `step_rows`) are computed with `step_rows`' own hiding leaf, frame leaf, pad and node, and CUDA streams and events
are inert. The collector keeps its plans as bytes, out of the cyclic collector's sight, as the C++ keeps them out of Python's
heap. Every step's root is checked against `step_rows`' tree over the same tensors packed into one stream, under the salt key
the step recorded, so the bytes, layout and digests are those of a committer with no cache at all.
"""
from __future__ import annotations

import functools
import gc
import hashlib
import pickle
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.commit import step_rows as R  # noqa: E402
from verity_vllm.commit.committer import native_collect as NC  # noqa: E402
from verity_vllm.commit.committer import native_host as nh  # noqa: E402
from verity_vllm.commit.committer.pinned_pool import PinnedPool  # noqa: E402

RUN_ID = "plan-cache"
PD, GEO = hashlib.sha256(b"program").digest(), hashlib.sha256(b"geo").digest()
DTYPES = (torch.int32, torch.bfloat16, torch.float32)
LAYERS, H, BN = 2, 512, 16                                     # a step is about 17 KiB: five rows, three windows and a pad window
WINDOW_MB, SLOTS = str(2 * R.ROW_BYTES / (1 << 20)), 4         # two rows a window, four slots
CACHES = ("_metas_cache", "_ld_cache", "_nbytes_cache")
ENV = ("VERITY_RETAIN", "VERITY_WINDOW_MB", "VERITY_WINDOW_SLOTS", "VERITY_STAGING_BOUNDED", "VERITY_LEARN_HOST_BUDGET_MB",
       "VERITY_RETAIN_EXCLUDE", "VERITY_FAULT", "VERITY_FOOTPRINT", "VERITY_HI_PRIO", "VERITY_CONSUMER_DELAY_MS", "VERITY_SKIP",
       "VERITY_INLINE_D2H", "VERITY_ABL")


class _Event:
    cuda_event = 0

    def __init__(self, enable_timing: bool = False) -> None:
        pass

    def record(self, stream=None) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def elapsed_time(self, other) -> float:
        return 0.0


class _Stream:
    cuda_stream = 0

    def __init__(self, priority: int = 0) -> None:
        pass

    def wait_event(self, event) -> None:
        pass

    def wait_stream(self, stream) -> None:
        pass

    def synchronize(self) -> None:
        pass


class _Cuda:
    Event, Stream = _Event, _Stream

    @staticmethod
    def current_stream() -> _Stream:
        return _Stream()

    @staticmethod
    def stream(s):
        import contextlib

        return contextlib.nullcontext()

    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def mem_get_info() -> tuple[int, int]:
        return 1 << 40, 1 << 40


class _Torch:
    """torch with inert CUDA: a device tensor is a host one."""

    cuda = _Cuda

    @staticmethod
    def empty(*a, device=None, **k):
        return torch.empty(*a, **k)

    def __getattr__(self, name):
        return getattr(torch, name)


def _nodes(digests: list[bytes]) -> torch.Tensor:
    if not digests:
        return torch.empty(0, 64, dtype=torch.uint8)
    return torch.frombuffer(bytearray(b"".join(digests)), dtype=torch.uint8).view(len(digests), 64)


def _digests(t: torch.Tensor) -> list[bytes]:
    b = t.contiguous().numpy().tobytes()
    return [b[i:i + 64] for i in range(0, len(b), 64)]


class _RowStep:
    """`RowStep`: the step's frame domain and salt key (`max_blocks`, the launch cap, changes no byte)."""

    def __init__(self, dom, ks: bytes) -> None:
        self.dom, self.ks, self.max_blocks = dom, ks, 0


class _Hasher:
    """`StepHasher`'s rows-sha512 kernels over `step_rows`: each row's h from the step's salt key, a position's frame leaf when it
    has a row and is below the step's row count and its pad otherwise, a window's levels and the upper tree's."""

    BITS = 8 * R.ROW_BYTES

    def rows_h(self, rs: _RowStep, src, nbytes: int, first: int, tn, stream):
        assert nbytes % R.ROW_BYTES == 0, "the worker hashes whole rows"
        raw = src[:nbytes].numpy().tobytes()
        return _nodes([R.hiding_leaf(raw[o:o + R.ROW_BYTES], R.salt(rs.ks, first + o // R.ROW_BYTES), self.BITS)[1]
                       for o in range(0, nbytes, R.ROW_BYTES)]), None

    def rows_frame(self, rs: _RowStep, h, first: int, m: int, n_total: int, stream):
        hs = _digests(h)
        assert rs.dom.count == n_total
        return _nodes([R.leaf(rs.dom, first + i, hs[i]) if i < len(hs) and first + i < n_total else rs.dom.pad(first + i)
                       for i in range(m)])

    def rows_leaves(self, rs: _RowStep, src, nbytes: int, first: int, m: int, n_total: int, tn, stream):
        h, _bc = self.rows_h(rs, src, nbytes, first, tn, stream)
        return self.rows_frame(rs, h, first, m, n_total, stream), None, h

    def rows_tree_window(self, rs: _RowStep, leaves, n_levels: int, first: int, stream, dsts):
        level = _digests(leaves)
        levels = [level]
        for L in range(n_levels - 1):
            level = [rs.dom.node(L, (first >> (L + 1)) + i, level[2 * i], level[2 * i + 1]) for i in range(len(level) // 2)]
            levels.append(level)
        assert len(dsts) in (0, n_levels)
        for dst, lv in zip(dsts, levels):
            dst.view(-1).copy_(_nodes(lv).view(-1))
        return _nodes(level)

    def rows_tree(self, rs: _RowStep, nodes, level0: int, stream, levels_host: bool = False, dst=None):
        level = _digests(nodes)
        assert level and len(level) & (len(level) - 1) == 0, "a power of two of nodes"
        levels = [level]
        while len(level) > 1:
            level = [rs.dom.node(level0 + len(levels) - 1, i, level[2 * i], level[2 * i + 1]) for i in range(len(level) // 2)]
            levels.append(level)
        return ([_nodes(lv) for lv in levels] if levels_host else []), level[0], 0.0


class _Collector:
    """`native_collect.cpp`'s `Collector` on the windowed path (gather, the window ring), with `native_plans.h`'s plans. A step's
    tensors are packed into its stream as the gather lays them out; the worker takes window k into slot k mod the ring's size,
    whose previous bytes are overwritten with 0xA5 first, so a window shorter than its slot shows whether the worker zero-pads."""

    RELEASE_PER_STEP = 2

    def __init__(self) -> None:
        self.plans: dict[int, tuple[int, int, bytes, int]] = {}  # key -> (version, used, pickled entries, the last run that used it)
        self.pending: list[bytes] = []                           # dropped plans' entries, out of the book, freed by the next run's steps
        self.run, self.run_used, self.floor = 0, False, 0
        self.active = False
        self.check_sizes = True
        self.use_gather = False
        self.win_wait_timeout_s = 600.0
        self.failure = ""
        self.slots: list = []                                    # the committer's window tensors (`committer` sets them)
        self.win_bytes = 0
        self.streams: dict[int, bytes] = {}                      # step seq -> its packed stream, until the worker finishes it
        self.exports = 0
        self.learned = 0
        self.expected = 0

    def enable_gather(self, max_descs: int) -> None:
        pass

    def enable_windows(self, ptrs: list[int], nbytes: int) -> None:
        self.win_bytes = int(nbytes)

    def _entries(self, key: int) -> tuple:
        return pickle.loads(self.plans[key][2]) if key in self.plans else ()

    def begin_step(self, key, arena_ptr, nbytes, use_plan, step_id=-1):
        assert not self.active
        del self.pending[-self.RELEASE_PER_STEP:]
        version, used, blob, _run = self.plans.get(key, (self.floor, 0, pickle.dumps(()), 0))
        self.plans[key] = (version, used, blob, self.run)
        self.run_used = True
        self.key, self.seq = key, step_id
        self.planned = self._entries(key)
        self.learning = not use_plan or not self.planned
        self.entries: list[tuple] = []
        self.stream = bytearray()
        self.step_bytes = 0
        self.changed = False
        self.active = True

    def _put(self, t, off: int, nb: int) -> None:
        self.stream.extend(bytes(off - len(self.stream)))
        self.stream.extend(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        self.step_bytes += nb

    def stage(self, nid, t):
        if not self.active:
            return -1
        nb = t.numel() * t.element_size()
        if nb == 0:
            return -1
        e = (int(nid), DTYPES.index(t.dtype), tuple(int(x) for x in t.shape), int(nb))
        if not self.learning:
            i = len(self.entries)
            if i < len(self.planned) and self.planned[i][:4] == e:
                self._put(t, self.planned[i][4], nb)
                self.entries.append(self.planned[i])
                return self.planned[i][4]
            self.learning = self.changed = True                  # a miss: the plan is relearned from here on
        off = (len(self.stream) + 3) & ~3
        gap = off - len(self.stream)
        self._put(t, off, nb)
        self.entries.append((*e, off, gap))
        self.changed = True
        return off

    def flush(self) -> None:
        pass

    def note_expected(self, n):
        self.expected += n

    def end_step(self):
        assert self.active
        self.active = False
        if not self.learning and len(self.entries) < len(self.planned):
            self.changed = True
        version, used, blob, run = self.plans[self.key]
        if self.changed:
            version += 1
            used = self.entries[-1][4] + self.entries[-1][3] if self.entries else 0
            self.plans[self.key] = (version, used, pickle.dumps(tuple(self.entries)), run)
            self.learned += 1
        self.streams[self.seq] = bytes(self.stream)
        return len(self.stream), len(self.entries), self.changed, version

    def end_step_full(self, flush, drain):
        return (*self.end_step(), self.step_bytes, 0.0, 0, 0.0, [])

    def wait_window(self, seq, k, stream):
        data = self.streams[seq]
        lo = k * self.win_bytes
        if lo >= len(data):
            return False, 0, -1
        win = data[lo:lo + self.win_bytes]
        slot = k % len(self.slots)
        self.slots[slot].fill_(0xA5)
        self.slots[slot][:len(win)].copy_(torch.frombuffer(bytearray(win), dtype=torch.uint8))
        return True, len(win), slot

    def release_slot(self, slot, stream) -> None:
        pass

    def free_slots(self) -> int:
        return len(self.slots)

    def finish_step(self, seq) -> None:
        del self.streams[seq]

    def abandon_step(self, seq, why, stream) -> None:
        self.failure = self.failure or why
        self.streams.pop(seq, None)

    def progress(self) -> None:
        pass

    def drain_errors(self) -> list:
        return []

    def export_plan(self, key):
        self.exports += 1
        return [(nid, DTYPES[dt], sizes, nb, off, gap) for nid, dt, sizes, nb, off, gap in self._entries(key)]

    def plan_used(self, key):
        return self.plans[key][1] if self._entries(key) else -1

    def plan_version(self, key):
        return self.plans[key][0] if key in self.plans else -1

    def step_stats(self):
        return {"bytes": self.step_bytes, "issue_s": 0.0}

    def stats(self):
        return {"plans": len(self.plans), "plan_entries": sum(len(self._entries(k)) for k in self.plans), "expected": self.expected,
                "committed": 0, "plans_pending": len(self.pending), "plan_entries_pending": sum(len(pickle.loads(b)) for b in self.pending)}

    def window_stats(self):
        return {}

    def reset_stats(self):
        self.expected = 0

    def end_run(self):
        self.reset_stats()
        if not self.run_used:
            return
        self.pending.clear()
        for key, (version, _used, blob, run) in list(self.plans.items()):
            if run != self.run:
                self.floor = max(self.floor, version + 1)
                self.pending.append(blob)
                del self.plans[key]
        self.run, self.run_used = self.run + 1, False


def fake_ext() -> SimpleNamespace:
    """The collector's extension (`native_ext.ext()`) over the fakes."""
    return SimpleNamespace(Collector=_Collector, StepHasher=_Hasher, cuda_runtime_version=lambda: 0)


class _HostPool(PinnedPool):
    def _alloc(self, nbytes: int):
        blk = torch.empty(max(int(nbytes), self.block_bytes), dtype=torch.uint8)
        self.all.append(blk)
        self.allocs += 1
        return blk


class _Runner:
    """The model runner: `execute_model` runs the step's acquisitions (the wrapped forwards' bodies) for the queued tensors."""

    def __init__(self) -> None:
        self.committer: NC.NativeCollectCommitter | None = None
        self.tensors: list[tuple[str, object]] = []
        self.input_batch = None

    def execute_model(self, scheduler_output, *a, **kw):
        com = self.committer
        for name, t in self.tensors:
            com._acquire(name, t, com._cur)


class _Sched:
    """A decode step's scheduler output: one new token for each request at its KV length."""

    def __init__(self, kv: tuple[int, ...]) -> None:
        self.req_ids = [f"r{i}" for i in range(len(kv))]
        self.num_scheduled_tokens = dict.fromkeys(self.req_ids, 1)
        self.total_num_scheduled_tokens = len(kv)
        self.scheduled_new_reqs = []
        self.scheduled_cached_reqs = SimpleNamespace(req_ids=self.req_ids, num_computed_tokens=list(kv))


def committer(mp, *, retain: str = "none", bounded: bool | None = None) -> NC.NativeCollectCommitter:
    """The served committer of the network traces (the windowed worker: wrap, gather, the native worker and the window ring, under
    rows-sha512) over the fakes; `bounded` None leaves the staging's default (bounded, since it is windowed)."""
    for k in ENV:
        mp.delenv(k, raising=False)
    mp.setenv("VERITY_RETAIN", retain)
    mp.setenv("VERITY_WINDOW_MB", WINDOW_MB)
    mp.setenv("VERITY_WINDOW_SLOTS", str(SLOTS))
    if bounded is not None:
        mp.setenv("VERITY_STAGING_BOUNDED", "1" if bounded else "0")
    fake = _Torch()
    mp.setattr(nh, "torch", fake)
    mp.setattr(NC, "torch", fake)
    ext = fake_ext()
    mp.setattr(NC, "ext", lambda *a, **k: ext)
    eng = SimpleNamespace(model=torch.nn.Module(), runner=_Runner())
    com = NC.NativeCollectCommitter(eng, run_id=RUN_ID, program_digest=PD, geo_digest=GEO, retain_values=False, arena_bytes=None,
                                    fast_copy=False, gather=True, native_worker=True)
    com.enable_windows(None)
    com._nc.slots = com.win_tensors
    com.pool = _HostPool(block_bytes=4096, exact=False, populate_threads=0)
    com._row_table = lambda: None
    com._row_step = lambda step, n, ks: _RowStep(com._row_domain(step, n), ks)
    com._start_worker = lambda: None                    # the test runs the worker's steps on its own thread, in order
    eng.runner.committer = com
    return com


def step_tensors(kv: tuple[int, ...]) -> list[tuple[str, torch.Tensor]]:
    """One decode step's acquired tensors: token ids, per layer the qkv output, a hidden plane sized by every request's KV blocks
    (as the attention tap's) and the MLP output, then logits and sampled ids. The values are a function of the geometry."""
    ntok, seed = len(kv), sum(kv)
    shapes = [("model.embed_tokens/in", torch.int32, (ntok,))]
    for layer in range(LAYERS):
        shapes += [(f"model.layers.{layer}.self_attn.qkv_proj", torch.bfloat16, (ntok, 3 * H)),
                   (f"fa2.m1/L{layer}/acc", torch.float32, (sum(-(-k // BN) for k in kv), 4)),
                   (f"model.layers.{layer}.mlp.down_proj", torch.bfloat16, (ntok, H))]
    shapes += [("logits", torch.float32, (ntok, 40)), ("runner.sampler/sampled_token_ids", torch.int32, (ntok,))]
    out = []
    for i, (name, dtype, shape) in enumerate(shapes):
        n = 1
        for d in shape:
            n *= d
        v = (torch.arange(n, dtype=torch.int64) * 7 + seed * 13 + i * 31) % 251 - 125
        out.append((name, v.to(dtype).reshape(shape)))
    return out


def schedule(kv0: int, steps: int, batch: int = 2) -> list[tuple[int, ...]]:
    """`steps` decode steps of `batch` requests whose KV lengths start at kv0, kv0 + 5, ... and grow by one per step."""
    return [tuple(kv0 + 5 * r + s for r in range(batch)) for s in range(steps)]


def serve(com: NC.NativeCollectCommitter, kv: tuple[int, ...]):
    """The serving thread's side of one decode step; returns the job it hands the worker."""
    so = _Sched(kv)
    com.runner.tensors = step_tensors(kv)
    com.runner.input_batch = SimpleNamespace(req_ids=so.req_ids, num_reqs=len(kv), num_scheduled_tokens=None)
    com.runner.execute_model(so)
    com._end_step()
    return com._q.get_nowait()


def run(com: NC.NativeCollectCommitter, sched: list[tuple[int, ...]]):
    """One commitment run, as the network traces' window loop runs it: install, the steps, finalize, uninstall, reset. Returns the
    run's commitment and the salt keys its steps took."""
    com.install()
    for kv in sched:
        com._commit_step(serve(com, kv))
    rc = com.finalize()
    com.uninstall()
    assert com.errors == [], com.errors[:3]
    keys = dict(com.row_keys)
    com.reset()
    return rc, keys


@functools.cache
def _offline() -> nh.NativeHostCommitter:
    return nh.NativeHostCommitter(None, run_id=RUN_ID, program_digest=PD, geo_digest=GEO, retain_values=False, gpu_tree=True)


def one_block_root(step: int, tensors: list[tuple[str, torch.Tensor]], ks: bytes) -> bytes:
    """The step committed with no cache: the tensors packed at 4-byte alignment into one stream, zero-padded to a row, its rows in
    `step_rows`' tree under the step's salt key, the layout counted in rows."""
    ref = _offline()
    stream, metas, first_leaf = bytearray(), [], 0
    for i, (name, t) in enumerate(tensors):
        raw = t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        stream.extend(b"\0" * (-len(stream) % nh.NativeHostCommitter.ALIGN))
        n_leaves = -(-len(raw) // R.ROW_BYTES)
        m = nh.TensorMeta(i, name, nh._dtype_name(t), tuple(t.shape), len(raw), R.ROW_BYTES, first_leaf, n_leaves)
        m.dev_off = m.stream_off = len(stream)
        stream.extend(raw)
        metas.append(m)
        first_leaf += n_leaves
    stream.extend(b"\0" * (-len(stream) % R.ROW_BYTES))
    rows = [bytes(stream[o:o + R.ROW_BYTES]) for o in range(0, len(stream), R.ROW_BYTES)]
    ctx = ref._ctx_digest(step)
    tree_root = R.levels(R.domain(PD, ctx, GEO, len(rows)), rows, ks, 8 * R.ROW_BYTES)[-1][0]
    return R.step_root(PD, ctx, GEO, ref.layout_digest(metas), len(rows), tree_root)


def reference(sched: list[tuple[int, ...]], keys: dict[int, bytes]) -> list[bytes]:
    return [one_block_root(i, step_tensors(kv), keys[i]) for i, kv in enumerate(sched)]


def tracked(root) -> int:
    """The objects the cyclic collector tracks that `root` keeps alive through containers and metas (not types, modules or functions)."""
    seen: set[int] = set()
    todo = [root]
    while todo:
        o = todo.pop()
        if id(o) in seen or not gc.is_tracked(o) or isinstance(o, (type, types.ModuleType, types.FunctionType)):
            continue
        seen.add(id(o))
        if isinstance(o, (dict, list, tuple, set, nh.TensorMeta)) or type(o).__module__.startswith("verity_vllm"):
            todo.extend(gc.get_referents(o))
    return len(seen)


def retained(com: NC.NativeCollectCommitter) -> int:
    """What the worker's plan caches keep alive between runs."""
    return tracked([getattr(com, nm) for nm in CACHES if hasattr(com, nm)])


def heap() -> int:
    """The objects a full collection walks: every tracked object outside the permanent generation."""
    gc.collect()
    return len(gc.get_objects())


RUNS, STEPS = 8, 12
TENSORS = 3 + 3 * LAYERS                                       # per step: ids, logits, sampled ids, three per layer


def rising() -> list[list[tuple[int, ...]]]:
    """RUNS runs of STEPS decode steps over the same two requests, the KV lengths rising run after run: no plan recurs."""
    return [schedule(40 + r * STEPS, STEPS) for r in range(RUNS)]


def test_the_plan_caches_stay_bounded_while_every_step_learns_a_new_plan(monkeypatch):
    """Over runs whose every decode step learns a new plan, the collector holds the plans of the run that just ended (and those of
    the run before it, dropped and freed by the next run's steps) and the worker the metas of the two latest runs' plans, not every
    step's since the process started, and the heap a full collection walks stops growing after the second run."""
    com = committer(monkeypatch)
    sizes, kept, heaps, plans = [], [], [], []
    for sched in rising():
        run(com, sched)
        sizes.append(len(com._metas_cache))
        kept.append(retained(com))
        heaps.append(heap())
        st = com._nc.stats()
        plans.append((st["plans"], st["plan_entries"], st["plans_pending"], st["plan_entries_pending"]))
    assert com._nc.learned == RUNS * STEPS
    book = (STEPS, STEPS * TENSORS)
    assert plans == [(*book, 0, 0)] + [(*book, *book)] * (RUNS - 1), f"the collector's plans, entries and pending after each run: {plans}"
    assert max(sizes) <= 2 * STEPS, f"plans cached after each run: {sizes}"
    per_plan = TENSORS + 1                                     # the metas list and one TensorMeta per tensor
    assert max(kept) <= 2 * STEPS * per_plan + 8, f"objects the caches keep alive after each run: {kept}"
    assert max(heaps[2:]) - heaps[1] <= STEPS * per_plan // 2, f"heap after each run: {heaps}"


@pytest.mark.parametrize("retain", ["none", "host"])
@pytest.mark.parametrize("native_end", [True, False], ids=["native-end", "python-end"])
def test_every_root_is_the_root_of_the_same_step_committed_from_one_block(monkeypatch, retain, native_end):
    """Every step's root equals the one-block reference over the same tensors under the step's salt key, on runs that learn every
    plan (keeping no bytes, each window hashed to its rows' h in place; keeping them, re-hashed from host once the step's length is
    known) and on a run that follows the plans of the run before it (each window hashed in place), with the serving thread's step
    end in one native call or in several."""
    com = committer(monkeypatch, retain=retain)
    com.native_end = native_end
    first, second = rising()[:2]
    for sched, learns in ((first, True), (second, True), (second, False)):
        rc, keys = run(com, sched)
        roots = [s.root for s in rc.steps]
        assert sorted(keys) == list(range(len(sched))) and len(set(keys.values())) == len(sched), "a fresh salt key per step"
        assert roots == reference(sched, keys)
        assert rc.run_root == R.run_root(PD, GEO, roots) and len(rc.run_root) == R.ROOT_BYTES
        w = rc.extra["windows"]
        want = ((len(sched), False) if retain == "none" else (0, True)) if learns else (0, False)
        assert (w["hfirst_steps"], w["reupload_bytes"] > 0) == want, w


def test_a_record_run_follows_the_warm_up_plans_without_learning_or_exporting_them_again(monkeypatch):
    """commit.py: the warm-up learns every plan, the committer is reset again before the instrumented arm, and the record run of
    the same schedule hits each plan: the collector learns none again, the serving thread exports nothing and the worker commits
    from the warm-up's metas; a replay of the warm-up's salt keys commits its roots."""
    com = committer(monkeypatch)
    sched = schedule(100, STEPS)
    warm, warm_keys = run(com, sched)
    com.reset()
    exports, learned = com._nc.exports, com._nc.learned
    com.replay_row_keys = warm_keys
    rec, rec_keys = run(com, sched)
    assert com._nc.learned == learned == STEPS and com._nc.exports == exports == STEPS
    assert rec_keys == warm_keys and [s.root for s in rec.steps] == [s.root for s in warm.steps] == reference(sched, warm_keys)


def test_a_plan_dropped_two_runs_back_is_learned_again_and_commits_the_same_roots(monkeypatch):
    """A schedule, two runs of others, then the first schedule again: the collector and the worker have both dropped its plans, so
    the collector learns each one again, at a version above the dropped one, the serving thread exports it, and the steps commit
    the bytes of the first run, and its roots under its salt keys."""
    com = committer(monkeypatch)
    first, other = schedule(200, STEPS), [schedule(300 + 20 * r, STEPS) for r in range(2)]
    a, a_keys = run(com, first)
    for sched in other:
        run(com, sched)
    learned, exports = com._nc.learned, com._nc.exports
    com.replay_row_keys = a_keys
    again, _ = run(com, first)
    assert com._nc.learned == learned + STEPS and com._nc.exports == exports + STEPS
    assert com._nc.plan_version(NC.plan_key_of(_Sched(first[0]), 2, 2)[0]) > 1
    assert [(s.n_bytes, s.root) for s in again.steps] == [(s.n_bytes, s.root) for s in a.steps] and again.run_root == a.run_root


def test_a_plan_learned_after_a_drop_never_takes_a_version_the_worker_still_holds(monkeypatch):
    """The collector can drop a plan the worker still holds: a run whose steps the collector began and the worker never absorbed
    (they failed) retires the collector's generation and not the worker's. A later step of the same key with other tensors learns
    a plan above the dropped version, so the worker builds its metas from the export, and the step commits its own tensors."""
    monkeypatch.setattr(NC, "plan_key_of", lambda *a: (1, "geometry"))
    com = committer(monkeypatch)
    run(com, schedule(700, 1))
    assert (1, 1) in com._metas_cache
    com._nc.begin_step(2, 0, 0, True)
    com._nc.end_step()
    com._nc.end_run()
    kv = schedule(900, 1)[0]
    rc, keys = run(com, [kv])
    assert rc.steps[0].root == one_block_root(0, step_tensors(kv), keys[0])
    assert com._nc.plan_version(1) > 1


def test_a_windowed_hit_step_exports_the_hinted_plan_the_worker_has_not_absorbed(monkeypatch):
    """Bounded windowed staging with excluded families places its retained ranges from the metas of the plan a hit step is
    predicted to follow, before the step ends. When the serving thread runs ahead of the worker, the step that learned that plan
    may not be absorbed yet: the serving thread exports the plan at step begin, and the worker builds the same metas from it."""
    com = committer(monkeypatch)
    com.install()
    learn = serve(com, schedule(400, 1)[0])
    key, _version, entries = learn.plan
    com.retain_exclude, com._plan_key = {"fa2"}, key
    com._begin_step()
    job = com._q.get_nowait()
    com._nc.active = False
    assert (key, 1) not in com._metas_cache and job.plan_hint == (key, 1) and job.hint_entries is not None
    got = com._metas_for(job, (*job.plan_hint, job.hint_entries))
    assert [(m.name, m.dev_off, m.nbytes) for m in got] == [(com._names[nid], off, nb) for nid, _dt, _sz, nb, off, _gap in entries]
