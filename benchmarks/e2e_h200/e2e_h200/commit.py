"""The per-step commit call: every unit's outputs at one step, hashed as `hm96-sha512` row tables in a `frame-v3-sha512` step
tree by the serving commit path (`verity_vllm.commit.row_tables`), the stand-in for kernels' CMP-95 item 3 (the tree with one
subtree per sequence slot) until it is posted.

A table is one op's output at one step, one row a sequence slot, keyed ``(slot, position)``; a row is the op's output bytes
and a 4-byte word (the quantizer's f32 scale, else zero), as kernels' `commit.py` lays out a bf16 row. Backends, all giving
the same `StepRoots` (`row_tables.step_roots` is the reference the other two are held to):

    reference   `row_tables.step_roots` (Python)
    host        `row_tables_device.host_step` (the C++ twin)
    device      `row_tables_device.step_roots` (`row_table.cu` on a side stream), CUDA tensors in
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from verity_vllm.commit import row_tables as RTB

BACKENDS = ("reference", "host", "device")


def row_bytes(t: Any) -> Any:
    """A (B, ...) tensor or array as (B, n) s8 bytes, n even (one zero byte appended where odd)."""
    if isinstance(t, np.ndarray):
        a = np.ascontiguousarray(t).reshape(t.shape[0], -1).view(np.int8)
        return a if a.shape[1] % 2 == 0 else np.concatenate([a, np.zeros((a.shape[0], 1), np.int8)], 1)
    import torch
    a = t.contiguous().reshape(t.shape[0], -1).view(torch.int8)
    return a if a.shape[1] % 2 == 0 else torch.cat([a, torch.zeros(a.shape[0], 1, dtype=torch.int8, device=a.device)], 1)


@dataclass
class StepTables:
    """One step's tables: ``tables[op_path] = (codes (B, n) s8, word (B,) f32)``, in the order the step wrote them. Row b is
    sequence slot ``slots[b]`` (ascending) at its position ``positions[b]``."""
    step: int
    slots: np.ndarray
    positions: np.ndarray
    tables: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    @property
    def batch(self) -> int:
        return len(self.slots)

    def row_of(self, slot: int) -> int:
        """The row of sequence slot ``slot`` in this step's tables."""
        b = int(np.searchsorted(self.slots, slot))
        if b == len(self.slots) or self.slots[b] != slot:
            raise KeyError(f"slot {slot} has no row at step {self.step}")
        return b

    def put(self, op: str, value: Any, word: Any = None) -> None:
        if op in self.tables:
            raise ValueError(f"{op} written twice at step {self.step}")
        codes = row_bytes(value)
        if word is None and isinstance(codes, np.ndarray):
            word = np.zeros(self.batch, np.float32)
        elif word is None:
            import torch
            word = torch.zeros(self.batch, dtype=torch.float32, device=codes.device)
        self.tables[op] = (codes, word)

    def ids(self) -> list[tuple[str, int]]:
        return [(f"s{int(j):06d}", int(p)) for j, p in zip(self.slots, self.positions)]

    def host(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        out = {}
        for op, (c, w) in self.tables.items():
            out[op] = (c if isinstance(c, np.ndarray) else c.cpu().numpy(), w if isinstance(w, np.ndarray) else w.cpu().numpy())
        return out

    def value_bytes(self) -> int:
        return sum(int(c.shape[0]) * (int(c.shape[1]) + 4) for c, _ in self.tables.values())


class Committer:
    def __init__(self, backend: str, context: bytes, key: bytes) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"the commit backend is one of {BACKENDS}, not {backend!r}")
        if len(context) != 64 or len(key) != 32:
            raise ValueError("the context is 64 bytes (the live coin commitment's root) and the row key 32")
        self.backend, self.context, self.key = backend, context, key

    def commit(self, st: StepTables) -> RTB.StepRoots:
        ids = st.ids()
        if self.backend == "device":
            import torch
            from verity_vllm.commit import row_tables_device as D

            def cuda(a: Any) -> Any:
                return torch.from_numpy(np.ascontiguousarray(a)).cuda() if isinstance(a, np.ndarray) else a
            outs = {op: (ids, cuda(c), cuda(w)) for op, (c, w) in st.tables.items()}
            return D.step_roots(self.context, st.step, outs, self.key).result()
        outs = {op: (ids, c, w) for op, (c, w) in st.host().items()}
        if self.backend == "host":
            from verity_vllm.commit import row_tables_device as D
            return D.host_step(self.context, st.step, outs, self.key)
        return RTB.step_roots(self.context, st.step, outs, self.key)


def fixture_secrets(run_id: str) -> tuple[bytes, bytes]:
    """A FIXTURE context and row key from the run id: the live coin commitment's root and the run's row key replace them."""
    ctx = hashlib.sha512(b"e2e-h200/context-fixture\0" + run_id.encode()).digest()
    key = hashlib.sha512(b"e2e-h200/row-key-fixture\0" + run_id.encode()).digest()[:32]
    return ctx, key
