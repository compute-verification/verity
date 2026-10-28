"""Real committers for value-check tests.

The value checks read committed bytes through `oracle_compare.OpenedReader`, which opens them from a committer and verifies every read
against the run commitment.  A test world's tensors are therefore committed by the production CPU core and read back through that reader:

  * `commit_steps` packs each step's tensors into one step block and commits it with the committer's CPU reference
    (`NativeHostCommitter.commit_block_offline`: `step_rows`'s rows spanning tensor boundaries, salted from a fresh step key in
    `row_keys`, the `retain` policy) -- the layout the pod's collector produces.
  * `commit_host_steps` commits each step through the direct staging path (`NativeHostCommitter.commit_offline`: the step's tensors packed
    at 4-byte-aligned offsets, then the same rows).  Both need torch, and both retain the committer's own copy of each step
    (`retained_block`), so a mutate-after-commit test writes there.

A tensor is `(name, bytes[, ordinal[, dtype, shape]])`.  Ordinals are kept, so a shared module's `<name>#ord<k>` reads select the same meta
they select on the pod (the direct path numbers tensors by position, so an ordinal there must equal it).
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

from verity_vllm.commit.committer.native_host import NativeHostCommitter, TensorMeta

PLACEHOLDER = ("placeholder", b"\0" * 4)


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


def commit_steps(steps: dict[int, list[tuple]], *, retain: str = "host", run_id: str = "opened-test",
                 row_keys: dict[int, bytes] | None = None):
    """Commit `steps[s]` (tensors in acquisition order) as packed step blocks -> (committer, run commitment).  Tensors sit at 4-byte-aligned
    offsets of the step block; a step missing below the last one is a one-tensor placeholder (steps are numbered densely).  `row_keys`,
    another commit's step keys (`replay_row_keys`), commit its trees again; unset, every step draws a fresh key."""
    com = committer(retain, run_id)
    com.replay_row_keys = None if row_keys is None else dict(row_keys)
    C = com.chunk
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


def commit_host_steps(steps: dict[int, list[tuple]], *, run_id: str = "opened-test"):
    """Commit `steps[s]` on the direct staging path -> (committer, run commitment); missing steps below the last are placeholders."""
    com = NativeHostCommitter(None, run_id=run_id, hash_threads=1, program_digest=hashlib.sha256(b"program").digest())
    stream = []
    for s in range(max(steps) + 1 if steps else 0):
        tensors = []
        for i, t in enumerate(steps.get(s) or [PLACEHOLDER]):
            name, raw, ordinal, dtype, shape = _spec(t, i)
            if ordinal != i:
                raise ValueError(f"step {s} {name}: ordinal {ordinal} != position {i} (the direct path numbers tensors by position)")
            tensors.append((name, raw, dtype, shape or (len(raw),)))
        stream.append(tensors)
    return com, com.commit_offline(stream)


def commit_members(members: dict[tuple[int, str], bytes], **kw):
    """{(step, member name): bytes} committed in insertion order per step -> (committer, run commitment)."""
    steps: dict[int, list] = {}
    for (s, name), raw in members.items():
        steps.setdefault(int(s), []).append((name, raw))
    return commit_steps(steps, **kw)


def retained_block(com: NativeHostCommitter, step: int) -> Any:
    """The committer's retained copy of packed `step` (a CPU uint8 tensor under retain=host, tensors at their `stream_off`): what a
    mutate-after-commit test writes to."""
    return com._gpu_blocks[step][0]


def meta(com: NativeHostCommitter, step: int, name: str) -> TensorMeta:
    return next(m for m in com._layouts[step] if m.name == name)
