"""The collector's plans and the worker's plan metas in `native_collect` across commitment runs: bounded, and the roots unchanged.

The collector keys each step's plan by its geometry (`plan_key_of`: every request's KV length and new tokens), so each decode
step of a growing KV length learns a plan of its own, and the worker builds one `TensorMeta` per staged tensor for it. The
served committer of the network traces commits one run per window and resets between them; a cache that kept every plan
grows with every committed step for the life of the process, and every full collection of the serving engine's heap walks
it. Here the committer runs on the CPU: the C++ collector is restated in Python from `native_collect.cpp` (a plan per key,
4-byte offsets with zeroed gaps, a hit followed entry by entry, a miss truncated and relearned, the version bumped on every
change) and `native_plans.h` (the plans the run that ended did not use dropped at `end_run`, a plan created after a drop
starting above every dropped version; the header itself is tested in `test_native_plans.py`), the GPU chunk tree is computed
with the scheme's own leaf and fold, and CUDA streams and events are inert. The
collector keeps its plans as bytes, out of the cyclic collector's sight, as the C++ keeps them out of Python's heap. Every
step's root is checked against `commit_block_offline`, the CPU reference of the GPU tree, over the same tensors packed into
one block, so the bytes, layout and digests are those of a committer with no cache at all.
"""
from __future__ import annotations

import gc
import hashlib
import pickle
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.commit import scheme  # noqa: E402
from verity_vllm.commit.committer import native_collect as NC  # noqa: E402
from verity_vllm.commit.committer import native_host as nh  # noqa: E402
from verity_vllm.commit.committer.pinned_pool import PinnedPool  # noqa: E402

PD, GEO = hashlib.sha256(b"program").digest(), hashlib.sha256(b"geo").digest()
DTYPES = (torch.int32, torch.bfloat16, torch.float32)
BLOCK = 1 << 16
LAYERS, H, BN = 2, 24, 16
CACHES = ("_metas_cache", "_ld_cache", "_nbytes_cache")
#: sha256 over every run root of `_rising` (8 runs of 12 steps, B=2), committed by main's `native_collect.py` at 0d4e61d5d
MAIN_RUN_ROOTS = "e481d224a5a6c19eff85d18d163ecf4fdcd7c3f2dfcc0c44706a1579d1329dc6"


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


class _Torch:
    cuda = _Cuda

    def __getattr__(self, name):
        return getattr(torch, name)


class _TreeExt:
    """The GPU chunk tree (`hidden_gpu.ext()`): 256-byte chunk leaves under the CMT-2 header, folded by the production rule."""

    def chunk_leaves(self, stream_i32, chunk_words, launch_tag, src_mask, HB, M, NB):
        layout = type("_L", (), {"HB": HB, "M": M, "NB": NB})
        raw, cb = stream_i32.numpy().tobytes(), chunk_words * 4
        leaves = [scheme.chunk_leaf(scheme.chunk_header(layout, launch_tag, i, chunk_words, src_mask), raw[i * cb:(i + 1) * cb])
                  for i in range(len(raw) // cb)]
        return torch.frombuffer(bytearray(b"".join(leaves)), dtype=torch.int32).reshape(len(leaves), 8)

    def tree_levels(self, leaves):
        level = [leaves[i].numpy().tobytes() for i in range(int(leaves.shape[0]))]
        return [torch.frombuffer(bytearray(b"".join(lv)), dtype=torch.int32).reshape(len(lv), 8) for lv in scheme.fold_levels(level)]


class _StepHasher:
    """`StepHasher.commit` (the native worker's step): pad, chunk leaves and levels over the block, no retained copy."""

    def __init__(self, tree: _TreeExt) -> None:
        self.tree = tree

    def commit(self, arena, cursor, step, src, chunk, stream, event, keep_mode, keep_dst, to_host, levels_dst):
        padded = -(-int(cursor) // chunk) * chunk
        arena[cursor:padded].zero_()
        levels = self.tree.tree_levels(self.tree.chunk_leaves(arena[:padded].view(torch.int32), chunk // 4, step, src, step, padded // 4, 0))
        return levels, None, levels[-1].numpy().astype("uint32").tobytes(), padded, 0.0, 0.0


class _Collector:
    """`native_collect.cpp`'s `Collector` on the arena path (no gather, no windows), with `native_plans.h`'s plans."""

    RELEASE_PER_STEP = 2

    def __init__(self) -> None:
        self.plans: dict[int, tuple[int, int, bytes, int]] = {}  # key -> (version, used, pickled entries, the last run that used it)
        self.pending: list[bytes] = []                           # dropped plans' entries, out of the book, freed by the next run's steps
        self.run, self.run_used, self.floor = 0, False, 0
        self.arena_of = None
        self.active = False
        self.check_sizes = True
        self.use_gather = False
        self.exports = 0
        self.learned = 0
        self.expected = 0

    def _entries(self, key: int) -> tuple:
        return pickle.loads(self.plans[key][2]) if key in self.plans else ()

    def begin_step(self, key, arena_ptr, nbytes, use_plan, step_id=-1):
        assert not self.active
        del self.pending[-self.RELEASE_PER_STEP:]
        version, used, blob, _run = self.plans.get(key, (self.floor, 0, pickle.dumps(()), 0))
        self.plans[key] = (version, used, blob, self.run)
        self.run_used = True
        self.key, self.cap = key, nbytes
        self.block = self.arena_of(arena_ptr) if arena_ptr else None
        self.planned = self._entries(key)
        self.learning = not use_plan or not self.planned
        self.entries: list[tuple] = []
        self.byte_cursor = self.step_bytes = 0
        self.changed = False
        self.active = True

    def _put(self, t, off: int, gap: int, nb: int) -> None:
        self.block[off - gap:off].zero_()
        self.block[off:off + nb].copy_(t.contiguous().reshape(-1).view(torch.uint8))
        self.byte_cursor = off + nb
        self.step_bytes += nb

    def collect(self, nid, t):
        if not self.active:
            return -1
        nb = t.numel() * t.element_size()
        if nb == 0:
            return -1
        e = (int(nid), DTYPES.index(t.dtype), tuple(int(x) for x in t.shape), int(nb))
        if not self.learning:
            i = len(self.entries)
            if i < len(self.planned) and self.planned[i][:4] == e:
                self._put(t, self.planned[i][4], self.planned[i][5], nb)
                self.entries.append(self.planned[i])
                return self.planned[i][4]
            self.learning = self.changed = True                  # a miss: the plan is relearned from here on
        off = (self.byte_cursor + 3) & ~3
        if off + nb > self.cap:
            raise RuntimeError("native_collect: arena block too small for this step")
        gap = off - self.byte_cursor
        self._put(t, off, gap, nb)
        self.entries.append((*e, off, gap))
        self.changed = True
        return off

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
        return self.byte_cursor, len(self.entries), self.changed, version

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


class _HostPool(PinnedPool):
    def _alloc(self, nbytes: int):
        blk = torch.empty(max(int(nbytes), self.block_bytes), dtype=torch.uint8)
        self.all.append(blk)
        self.allocs += 1
        return blk


class _CpuArena(nh._DeviceArena):
    def _alloc(self, nbytes: int):
        blk = torch.zeros(nbytes, dtype=torch.uint8)
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


def committer(mp, *, native_worker: bool = False) -> NC.NativeCollectCommitter:
    """The served committer of the network traces (wrap, no gather, the arena and the GPU tree, retain none) over the fakes."""
    for k in ("VERITY_RETAIN", "VERITY_WINDOW_MB", "VERITY_STAGING_BOUNDED", "VERITY_RETAIN_EXCLUDE", "VERITY_FAULT", "VERITY_FOOTPRINT",
              "VERITY_HI_PRIO", "VERITY_CONSUMER_DELAY_MS", "VERITY_SKIP", "VERITY_INLINE_D2H", "VERITY_ABL"):
        mp.delenv(k, raising=False)
    fake = _Torch()
    mp.setattr(nh, "torch", fake)
    mp.setattr(NC, "torch", fake)
    tree = _TreeExt()
    mp.setattr(NC, "ext", lambda *a, **k: SimpleNamespace(Collector=_Collector, StepHasher=lambda: _StepHasher(tree), cuda_runtime_version=lambda: 0))
    eng = SimpleNamespace(model=torch.nn.Module(), runner=_Runner())
    com = NC.NativeCollectCommitter(eng, run_id="plan-cache", program_digest=PD, geo_digest=GEO, retain_values=False, arena_bytes=None,
                                    fast_copy=False, native_worker=native_worker)
    com.arena = _CpuArena(BLOCK, n_blocks=2)
    com.arena.prealloc()
    com._nc.arena_of = lambda ptr: next(b for b in com.arena.all if b.data_ptr() == ptr)
    com.pool = _HostPool(block_bytes=4096, exact=False, populate_threads=0)
    com._gpu_ext = tree
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
    """One commitment run, as the network traces' window loop runs it: install, the steps, finalize, uninstall, reset."""
    com.install()
    for kv in sched:
        com._commit_step(serve(com, kv))
    rc = com.finalize()
    com.uninstall()
    com.reset()
    return rc


def one_block_root(step: int, tensors: list[tuple[str, torch.Tensor]]) -> bytes:
    """The step committed with no cache: the tensors packed at 4-byte alignment, through the CPU reference of the GPU tree."""
    ref = nh.NativeHostCommitter(None, run_id="plan-cache", program_digest=PD, geo_digest=GEO, retain_values=False, gpu_tree=True)
    stream, metas, first_leaf = bytearray(), [], 0
    for i, (name, t) in enumerate(tensors):
        raw = t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        stream.extend(b"\0" * (-len(stream) % nh.NativeHostCommitter.ALIGN))
        n_leaves = -(-len(raw) // nh.GPU_CHUNK_BYTES)
        m = nh.TensorMeta(i, name, nh._dtype_name(t), tuple(t.shape), len(raw), nh.GPU_CHUNK_BYTES, first_leaf, n_leaves)
        m.dev_off = len(stream)
        stream.extend(raw)
        metas.append(m)
        first_leaf += n_leaves
    return ref.commit_block_offline(step, stream, metas).root


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


WORKERS = pytest.mark.parametrize("native_worker", [False, True], ids=["python-worker", "native-worker"])


@WORKERS
def test_the_plan_caches_stay_bounded_while_every_step_learns_a_new_plan(monkeypatch, native_worker):
    """Over runs whose every decode step learns a new plan, the collector holds the plans of the run that just ended (and those of
    the run before it, dropped and freed by the next run's steps) and the worker the metas of the two latest runs' plans, not every
    step's since the process started, and the heap a full collection walks stops growing after the second run."""
    com = committer(monkeypatch, native_worker=native_worker)
    sizes, kept, heaps, plans = [], [], [], []
    for sched in rising():
        run(com, sched)
        sizes.append(len(com._metas_cache))
        kept.append(retained(com))
        heaps.append(heap())
        st = com._nc.stats()
        plans.append((st["plans"], st["plan_entries"], st["plans_pending"], st["plan_entries_pending"]))
    assert com._nc.learned == RUNS * STEPS and com.errors == []
    book = (STEPS, STEPS * TENSORS)
    assert plans == [(*book, 0, 0)] + [(*book, *book)] * (RUNS - 1), f"the collector's plans, entries and pending after each run: {plans}"
    assert max(sizes) <= 2 * STEPS, f"plans cached after each run: {sizes}"
    per_plan = TENSORS + 1                                     # the metas list and one TensorMeta per tensor
    assert max(kept) <= 2 * STEPS * per_plan + 8, f"objects the caches keep alive after each run: {kept}"
    assert max(heaps[2:]) - heaps[1] <= STEPS * per_plan // 2, f"heap after each run: {heaps}"


@WORKERS
def test_every_root_is_the_root_of_the_same_step_committed_from_one_block(monkeypatch, native_worker):
    """Every step's root equals the one-block CPU reference over the same tensors, and the run roots are those main's committer
    produced on the same schedule."""
    com = committer(monkeypatch, native_worker=native_worker)
    roots = []
    for sched in rising():
        rc = run(com, sched)
        assert [s.root for s in rc.steps] == [one_block_root(i, step_tensors(kv)) for i, kv in enumerate(sched)]
        roots.append(rc.run_root)
    assert hashlib.sha256(b"".join(roots)).hexdigest() == MAIN_RUN_ROOTS


def test_a_record_run_follows_the_warm_up_plans_without_learning_or_exporting_them_again(monkeypatch):
    """commit.py: the warm-up learns every plan, the committer is reset again before the instrumented arm, and the record run of
    the same schedule hits each plan: the collector learns none again, the serving thread exports nothing and the worker commits
    from the warm-up's metas."""
    com = committer(monkeypatch)
    sched = schedule(100, STEPS)
    warm = run(com, sched)
    com.reset()
    exports, learned = com._nc.exports, com._nc.learned
    rec = run(com, sched)
    assert com._nc.learned == learned == STEPS and com._nc.exports == exports == STEPS
    assert [s.root for s in rec.steps] == [s.root for s in warm.steps]


def test_a_plan_dropped_two_runs_back_is_learned_again_and_commits_the_same_roots(monkeypatch):
    """A schedule, two runs of others, then the first schedule again: the collector and the worker have both dropped its plans, so
    the collector learns each one again, at a version above the dropped one, the serving thread exports it, and the steps commit
    the bytes and roots of the first run."""
    com = committer(monkeypatch)
    first, other = schedule(200, STEPS), [schedule(300 + 20 * r, STEPS) for r in range(2)]
    a = run(com, first)
    for sched in other:
        run(com, sched)
    learned, exports = com._nc.learned, com._nc.exports
    again = run(com, first)
    assert com._nc.learned == learned + STEPS and com._nc.exports == exports + STEPS and com.errors == []
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
    rc = run(com, [kv])
    assert rc.steps[0].root == one_block_root(0, step_tensors(kv)) and com.errors == []
    assert com._nc.plan_version(1) > 1


def test_a_windowed_hit_step_exports_the_hinted_plan_the_worker_has_not_absorbed(monkeypatch):
    """Bounded windowed staging with excluded families places its retained ranges from the metas of the plan a hit step is
    predicted to follow, before the step ends. When the serving thread runs ahead of the worker, the step that learned that plan
    may not be absorbed yet: the serving thread exports the plan at step begin, and the worker builds the same metas from it."""
    com = committer(monkeypatch)
    com.install()
    learn = serve(com, schedule(400, 1)[0])
    key, _version, entries = learn.plan
    com.windowed, com.retain_exclude, com._plan_key = True, {"fa2"}, key
    com._begin_step()
    job = com._q.get_nowait()
    com._nc.active = False
    assert (key, 1) not in com._metas_cache and job.plan_hint == (key, 1) and job.hint_entries is not None
    got = com._metas_for(job, (*job.plan_hint, job.hint_entries))
    assert [(m.name, m.dev_off, m.nbytes) for m in got] == [(com._names[nid], off, nb) for nid, _dt, _sz, nb, off, _gap in entries]
