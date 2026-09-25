"""Real committers for value-check tests.

The value checks read committed bytes through `oracle_compare.OpenedReader`, which opens them from a committer and verifies every read
against the run commitment.  A test world's tensors are therefore committed by the production CPU core and read back through that reader:

  * `commit_steps` packs each step's tensors into one step block and commits it with the GPU-tree committer's CPU reference
    (`NativeHostCommitter.commit_block_offline`: 256 B chunk leaves spanning tensor boundaries, the production fold, the `retain` policy) --
    the layout the pod's collector produces.  Needs torch.
  * `commit_host_steps` commits each tensor as its own leaves (`NativeHostCommitter.commit_offline`, the per-tensor host path; torch-free);
    the retained copy of a tensor is the buffer passed in, so a test that passes a bytearray can change it after the commit.

A tensor is `(name, bytes[, ordinal[, dtype, shape]])`.  Ordinals are kept, so a shared module's `<name>#ord<k>` reads select the same meta
they select on the pod (the per-tensor path numbers tensors by position, so an ordinal there must equal it).
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

from verity_vllm.commit.committer.native_host import GPU_CHUNK_BYTES, NativeHostCommitter, TensorMeta
from verity_vllm.check import oracle_compare as OC

PLACEHOLDER = ("placeholder", b"\0" * 4)


def raw_bytes(host: Any) -> bytes:
    """A test double's retained copy (numpy array, an object with `.numpy()`, memoryview or bytes) as its bytes in C order."""
    if hasattr(host, "numpy"):
        host = host.numpy()
    return host.tobytes() if hasattr(host, "tobytes") else bytes(host)


def _spec(t: tuple, i: int) -> tuple[str, Any, int, str, tuple | None]:
    ordinal = int(t[2]) if len(t) > 2 and t[2] is not None else i
    return str(t[0]), t[1], ordinal, (str(t[3]) if len(t) > 3 else "uint8"), (tuple(t[4]) if len(t) > 4 else None)


def committer(retain: str = "host", run_id: str = "opened-test") -> NativeHostCommitter:
    saved = os.environ.get("VERITY_RETAIN")
    os.environ["VERITY_RETAIN"] = retain
    try:
        return NativeHostCommitter(None, run_id=run_id, gpu_tree=True)
    finally:
        if saved is None:
            del os.environ["VERITY_RETAIN"]
        else:
            os.environ["VERITY_RETAIN"] = saved


def commit_steps(steps: dict[int, list[tuple]], *, retain: str = "host", run_id: str = "opened-test"):
    """Commit `steps[s]` (tensors in acquisition order) as packed step blocks -> (committer, run commitment).  Tensors sit at 4-byte-aligned
    offsets of the step block; a step missing below the last one is a one-tensor placeholder (steps are numbered densely)."""
    com = committer(retain, run_id)
    C = GPU_CHUNK_BYTES
    for s in range(max(steps) + 1 if steps else 0):
        metas, buf = [], bytearray()
        for i, t in enumerate(steps.get(s) or [PLACEHOLDER]):
            name, raw, ordinal, dtype, shape = _spec(t, i)
            raw = bytes(raw)
            off, n = len(buf), len(raw)
            m = TensorMeta(ordinal, name, dtype, shape or (n,), n, C, off // C, -(-(off + n) // C) - off // C)
            m.dev_off = off
            metas.append(m)
            buf += raw + b"\0" * ((-n) % 4)
        com.commit_block_offline(s, bytes(buf), metas)
    return com, com.finalize()


def commit_host_steps(steps: dict[int, list[tuple]], *, run_id: str = "opened-test", chunk: int = 64):
    """Commit `steps[s]` on the per-tensor host path -> (committer, run commitment); missing steps below the last are placeholders."""
    com = NativeHostCommitter(None, run_id=run_id, chunk=chunk, hash_threads=1, program_digest=hashlib.sha256(b"program").digest())
    stream = []
    for s in range(max(steps) + 1 if steps else 0):
        tensors = []
        for i, t in enumerate(steps.get(s) or [PLACEHOLDER]):
            name, raw, ordinal, dtype, shape = _spec(t, i)
            if ordinal != i:
                raise ValueError(f"step {s} {name}: ordinal {ordinal} != position {i} (the per-tensor path numbers tensors by position)")
            tensors.append((name, raw, dtype, shape or (len(raw),)))
        stream.append(tensors)
    return com, com.commit_offline(stream)


def commit_layouts(layouts: dict[int, list], *, host: bool = False, **kw):
    """A committer double's layouts (metas with `.name`, optional `.ordinal`, retained `.host`) committed -> (committer, run commitment)."""
    steps = {int(s): [(m.name, raw_bytes(m.host), getattr(m, "ordinal", i)) for i, m in enumerate(ms)] for s, ms in layouts.items()}
    return commit_host_steps(steps, **kw) if host else commit_steps(steps, **kw)


def commit_members(members: dict[tuple[int, str], bytes], **kw):
    """{(step, member name): bytes} committed in insertion order per step -> (committer, run commitment)."""
    steps: dict[int, list] = {}
    for (s, name), raw in members.items():
        steps.setdefault(int(s), []).append((name, raw))
    return commit_steps(steps, **kw)


def opened_reader(layouts: dict[int, list], **kw) -> OC.OpenedReader:
    return OC.OpenedReader(*commit_layouts(layouts, **kw))


def retained_block(com: NativeHostCommitter, step: int) -> Any:
    """The committer's retained copy of packed `step` (a CPU uint8 tensor under retain=host): what a mutate-after-commit test writes to."""
    return com._gpu_blocks[step][0]


def meta(com: NativeHostCommitter, step: int, name: str) -> TensorMeta:
    return next(m for m in com._layouts[step] if m.name == name)
