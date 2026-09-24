"""Real committers for value-check tests.

The value checks read committed bytes through `oracle_compare.OpenedReader`, which opens them from a committer and verifies every read
against the run commitment.  A test world's tensors are therefore committed by the production CPU core and read back through that reader:
`commit_steps` packs each step's tensors into one step block and commits it with the GPU-tree committer's CPU reference
(`NativeHostCommitter.commit_block_offline`: 256 B chunk leaves, the production fold, the `retain` policy) -- the layout the pod's
collector produces.  Tensor ordinals are kept, so a shared module's `<name>#ord<k>` reads select the same meta they select on the pod.
"""
from __future__ import annotations

import os
from typing import Any

from verity_vllm.acquire.native_host import GPU_CHUNK_BYTES, NativeHostCommitter, TensorMeta
from verity_vllm.check import oracle_compare as OC


def raw_bytes(host: Any) -> bytes:
    """A test double's retained copy (numpy array, an object with `.numpy()`, memoryview or bytes) as its bytes in C order."""
    if hasattr(host, "numpy"):
        host = host.numpy()
    return host.tobytes() if hasattr(host, "tobytes") else bytes(host)


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
    """Commit `steps[s]` = [(name, bytes) | (name, bytes, ordinal)] in acquisition order -> (committer, run commitment).  Tensors sit at
    4-byte-aligned offsets of the step block; a step missing below the last one is a one-tensor placeholder (steps are numbered densely)."""
    com = committer(retain, run_id)
    C = GPU_CHUNK_BYTES
    for s in range(max(steps) + 1 if steps else 0):
        metas, buf = [], bytearray()
        for i, t in enumerate(steps.get(s) or [("placeholder", b"\0" * 4)]):
            name, raw = str(t[0]), bytes(t[1])
            off, n = len(buf), len(raw)
            m = TensorMeta(int(t[2]) if len(t) > 2 else i, name, "uint8", (n,), n, C, off // C, -(-(off + n) // C) - off // C)
            m.dev_off = off
            metas.append(m)
            buf += raw + b"\0" * ((-n) % 4)
        com.commit_block_offline(s, bytes(buf), metas)
    return com, com.finalize()


def commit_layouts(layouts: dict[int, list], **kw):
    """A committer double's layouts (metas with `.name`, optional `.ordinal`, retained `.host`) committed -> (committer, run commitment)."""
    return commit_steps({int(s): [(m.name, raw_bytes(m.host), getattr(m, "ordinal", i)) for i, m in enumerate(ms)] for s, ms in layouts.items()}, **kw)


def commit_members(members: dict[tuple[int, str], bytes], **kw):
    """{(step, member name): bytes} committed in insertion order per step -> (committer, run commitment)."""
    steps: dict[int, list] = {}
    for (s, name), raw in members.items():
        steps.setdefault(int(s), []).append((name, raw))
    return commit_steps(steps, **kw)


def opened_reader(layouts: dict[int, list], **kw) -> OC.OpenedReader:
    return OC.OpenedReader(*commit_layouts(layouts, **kw))


def retained_block(com: NativeHostCommitter, step: int) -> Any:
    """The committer's retained copy of `step` (a CPU uint8 tensor under retain=host): what a mutate-after-commit test writes to."""
    return com._gpu_blocks[step][0]


def meta(com: NativeHostCommitter, step: int, name: str) -> TensorMeta:
    return next(m for m in com._layouts[step] if m.name == name)
