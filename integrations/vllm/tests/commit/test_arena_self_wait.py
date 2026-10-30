"""The device staging arena of `native_host` with the GPU tree (`--gpu-tree`), on the CPU: no take of a block waits for a block
that only the waiting thread can give back.

The worker is the one thread that calls `_DeviceArena.release`. A step whose bytes overflow the block it began in is chained into
further blocks by the hooks, and the worker concatenates them into one block before hashing (`_commit_step_gpu`). When the only
block that can hold the concatenation is one the step itself holds, a waiting `take` waits on itself: the deadlock the network
traces met on the first prefill of a Commit with `native_host --gpu-tree` (a prefill begun in an unsized, decode-sized block).

The device here is the CPU: arena blocks are CPU tensors, streams and events are inert, the GPU chunk tree is computed with the
scheme's own leaf and fold, and the pinned pool hands out ordinary tensors. The worker's step runs on the test's thread, so a
`take` that waits could only be woken by the thread that waits: the arena's condition raises `SelfWait` on any wait instead of
blocking, which detects the deadlock without a clock. Each committed step's root is checked against `commit_block_offline` (the
CPU reference of the GPU tree) over the same tensors packed into one block: the bytes and digests are those of a step that
never overflowed.
"""
from __future__ import annotations

import contextlib
import hashlib
import threading

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.commit import scheme  # noqa: E402
from verity_vllm.commit.committer import native_host as nh  # noqa: E402
from verity_vllm.commit.committer.pinned_pool import PinnedPool  # noqa: E402

DECODE, PREFILL = 1024, 8192            # the auto-sized ring of `size_arena`: two decode-sized blocks and one prefill-sized one
PD, GEO = hashlib.sha256(b"program").digest(), hashlib.sha256(b"geo").digest()


class SelfWait(AssertionError):
    pass


class _Event:
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

    def wait_event(self, event) -> None:
        pass

    def synchronize(self) -> None:
        pass


class _Cuda:
    """`torch.cuda` as the committer sees it: a copy has landed when it returns."""
    Event, Stream = _Event, _Stream

    @staticmethod
    def current_stream() -> _Stream:
        return _Stream()

    @staticmethod
    def stream(s):
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


class _OnlyTheWaiterReleases(threading.Condition):
    """The arena's condition while one thread plays both the serving thread and the worker: whatever a `take` would wait for, only
    the waiting thread could release it, so a wait is the self-wait, raised by name instead of blocking."""

    def __init__(self, arena: nh._DeviceArena) -> None:
        super().__init__(arena._lock)
        self.arena = arena

    def wait(self, timeout=None):
        busy = [int(b.numel()) for b in self.arena.all if not any(b is f for f in self.arena.free)]
        raise SelfWait(f"take() waits for an arena block, but the blocks it could get ({busy} B busy) are held by steps only the "
                       f"waiting thread releases")


class _Runner:
    """The model runner: `execute_model` runs the step's acquisitions (the hooks' bodies) for the tensors the test queued."""

    def __init__(self) -> None:
        self.committer: nh.NativeHostCommitter | None = None
        self.tensors: list[tuple[str, object]] = []

    def execute_model(self, scheduler_output, *a, **kw):
        com = self.committer
        for name, t in self.tensors:
            com._acquire_impl(name, t, com._cur)

    def sample(self, *a, **kw):
        return None


class _Engine:
    def __init__(self) -> None:
        self.model = torch.nn.Module()
        self.runner = _Runner()


class _Sched:
    """The scheduler's output for one step: `ntok` tokens over `nreq` requests."""

    def __init__(self, ntok: int, nreq: int) -> None:
        self.total_num_scheduled_tokens = ntok
        self.num_scheduled_tokens = {f"r{i}": 1 for i in range(nreq)}


@pytest.fixture
def committer(monkeypatch):
    monkeypatch.delenv("VERITY_RETAIN", raising=False)
    monkeypatch.setattr(nh, "torch", _Torch())
    eng = _Engine()
    com = nh.NativeHostCommitter(eng, run_id="arena-self-wait", program_digest=PD, geo_digest=GEO, retain_values=False,
                                 arena_bytes=None, gpu_tree=True)
    com.arena = _CpuArena(DECODE, n_blocks=2, extra_blocks=[PREFILL])
    com.arena.prealloc()
    com.arena._cv = _OnlyTheWaiterReleases(com.arena)
    com.arena_sizing = {"decode_block_bytes": DECODE, "prefill_block_bytes": PREFILL}
    com.pool = _HostPool(block_bytes=4096, exact=False, populate_threads=0)
    com._gpu_ext = _TreeExt()
    com._start_worker = lambda: None                    # the test runs the worker's steps on its own thread, in order
    eng.runner.committer = com
    com.install()
    yield com
    com.uninstall()


def _tensors(seed: int, shapes: list[tuple[str, torch.dtype, tuple[int, ...]]]) -> list[tuple[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    return [(name, torch.randint(-1000, 1000, shape, generator=g, dtype=torch.int32).to(dtype)) for name, dtype, shape in shapes]


def _step(com: nh.NativeHostCommitter, sched: _Sched, tensors: list[tuple[str, torch.Tensor]]) -> nh._StepJob:
    """One engine step through the runner the committer wrapped; returns the job it queued for the worker."""
    com.runner.tensors = tensors
    com.runner.execute_model(sched)
    com.runner.sample()
    return com._q.get_nowait()


def _one_block_root(step: int, tensors: list[tuple[str, torch.Tensor]]) -> bytes:
    """The step committed from ONE block: the tensors packed at 4-byte alignment, through the CPU reference of the GPU tree."""
    ref = nh.NativeHostCommitter(None, run_id="arena-self-wait", program_digest=PD, geo_digest=GEO, retain_values=False, gpu_tree=True)
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


PREFILL_STEP = [("model.embed_tokens/in", torch.int32, (8,)), ("layers.0.qkv_proj", torch.bfloat16, (8, 40)),
                ("layers.0.mlp", torch.bfloat16, (8, 100)), ("logits", torch.float32, (1, 50))]


def test_a_prefill_takes_the_prefill_block_and_the_worker_never_waits_on_its_own_step(committer):
    """The reported deadlock: a prefill (more tokens than requests) whose bytes exceed a decode block. Begun unsized, it took the
    smallest block, overflowed into the prefill block, and the worker's concatenation waited for the prefill block the step held.
    Sized from the scheduler, as `native_collect` sizes it, the step stages in the prefill block alone."""
    tensors = _tensors(0, PREFILL_STEP)
    job = _step(committer, _Sched(ntok=8, nreq=1), tensors)
    staged = [int(b.numel()) for b, _u in job.overflow] + [int(job.arena.numel())]
    committer._commit_step(job)
    assert committer.errors == []
    assert committer._done[0].root == _one_block_root(0, tensors)
    assert staged == [PREFILL]
    assert committer.arena.allocs == 3 and sorted(int(b.numel()) for b in committer.arena.free) == [DECODE, DECODE, PREFILL]


def test_an_overflowed_step_is_concatenated_without_waiting_and_commits_the_one_block_root(committer):
    """A step larger than the block its shape was sized for (a decode step past the decode bound) still overflows: begun in the
    second decode block while the first is busy with the step before it, it chains into the prefill block. The worker's
    concatenation never waits, since nothing but the worker releases a block, and the step's root is the root of the same tensors
    committed from one block."""
    small = _tensors(1, [("model.embed_tokens/in", torch.int32, (2,)), ("layers.0.mlp", torch.bfloat16, (2, 100))])
    big = _tensors(2, [("model.embed_tokens/in", torch.int32, (2,)), ("layers.0.qkv_proj", torch.bfloat16, (2, 200)),
                       ("layers.0.mlp", torch.bfloat16, (2, 200)), ("logits", torch.float32, (2, 30))])
    first = _step(committer, _Sched(ntok=2, nreq=2), small)
    second = _step(committer, _Sched(ntok=2, nreq=2), big)
    assert [int(b.numel()) for b, _u in second.overflow] == [DECODE] and second.arena.numel() == PREFILL
    committer._commit_step(first)
    committer._commit_step(second)
    assert committer.errors == []
    assert committer._done[0].root == _one_block_root(0, small)
    assert committer._done[1].root == _one_block_root(1, big)
    assert len(committer.arena.free) == len(committer.arena.all)        # every block back in the ring, the grown one included


def test_a_step_the_worker_fails_gives_its_blocks_back_and_the_next_step_takes_one_without_waiting(committer, monkeypatch):
    """A step whose commit raises on the worker (here the GPU tree's extension failing to load) is recorded as an error with an
    empty commitment and gives back every block it held, its overflow included. Kept, two such steps held the whole ring and
    serve-openai's engine waited the arena's full timeout on the next step, with the worker idle."""
    def no_extension():
        raise RuntimeError("hidden_gpu: build failed")

    monkeypatch.setattr(committer, "_gpu_ext_load", no_extension)
    big = _tensors(3, [("model.embed_tokens/in", torch.int32, (2,)), ("layers.0.qkv_proj", torch.bfloat16, (2, 200)),
                       ("layers.0.mlp", torch.bfloat16, (2, 200)), ("logits", torch.float32, (2, 30))])
    job = _step(committer, _Sched(ntok=2, nreq=2), big)
    assert job.overflow
    committer._q.put(job)
    committer._q.put(committer._stop)
    committer._run_worker()
    assert committer.errors == ["worker step 0: RuntimeError('hidden_gpu: build failed')"]
    assert committer._done[0].root == b""
    assert len(committer.arena.free) == len(committer.arena.all)
    monkeypatch.setattr(committer, "_gpu_ext_load", lambda: _TreeExt())
    nxt = _step(committer, _Sched(ntok=8, nreq=1), _tensors(4, PREFILL_STEP))
    committer._commit_step(nxt)
    assert committer.errors[1:] == [] and committer._done[1].root == _one_block_root(1, _tensors(4, PREFILL_STEP))
