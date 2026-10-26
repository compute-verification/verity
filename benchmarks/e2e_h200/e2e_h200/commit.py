"""The per-step commit call: every unit's outputs at one step, committed as the step's groups (the embedding, each layer,
the head) come out of the forward pass.

A table is one op's output at one step, one row a sequence slot (`TABLES` names them, in the order the step writes them); a
quantizer's table also has the f32 scale as each row's word. Backends:

    slots       kernels' commit call (CMP-95 item 3, #4469, ``benchmarks/torch_gpu/commit_slots``): ``hm96-sha512/row/v2``
                rows under hiding leaves in ``frame-v3-sha512`` trees, step -> slot -> unit -> rows, so a drawn unit opens
                by replaying its own request. `layout` lays a slot's tables out as its units: one group a step group, each
                table cut into one-row units of at most 32 KiB, a quantizer's codes and scale one two-part row when they
                fit (else the codes cut and the scale a unit of its own). Streamed: each group goes to
                `slots_gpu.StepCommitter.add` (a side stream on CUDA, the engine's host twin on the CPU) as it is written,
                so a group's tables can be dropped once added. The salt key is fresh each step (`slots.fresh_salt_key`)
                unless the committer is given a ``salt`` function (tests and fixture runs, which say so).
    reference   the by-table step tree (`verity_vllm.commit.row_tables.step_roots`, Python), the one the small tests and
    host        the replay of a whole step's tables read: ``host`` is its C++ twin (`row_tables_device.host_step`) and
    device      ``device`` its CUDA kernel (`row_table.cu`); each commits the step's tables at its end.
"""
from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .shape import Shape

COMMIT_SLOTS = Path(__file__).resolve().parents[2] / "torch_gpu" / "commit_slots"
BACKENDS = ("slots", "reference", "host", "device")
LAYER_OPS = ("attn_norm", "attn_quant", "wq", "wk", "wv", "scores", "softmax", "pv", "o_quant", "wo", "attn_residual",
             "mlp_norm", "up_quant", "wup", "down_quant", "wdown", "mlp_residual")


@dataclass(frozen=True)
class Table:
    op: str
    nbytes: int      # a slot's bytes of the op's output
    quant: bool      # the row's word (the f32 scale) is committed after the codes


def tables(shape: Shape) -> list[list[Table]]:
    """The step's groups of tables in the order the step writes them: the embedding, each layer's ops, the head's."""
    w, hd, c, f, v = shape.width, shape.heads * shape.head_dim, shape.context, shape.mlp, shape.vocab
    kv = shape.kv_heads * shape.head_dim
    sizes = {"attn_norm": 2 * w, "attn_quant": w, "wq": 2 * hd, "wk": 2 * kv, "wv": 2 * kv, "scores": 2 * shape.heads * c,
             "softmax": 2 * shape.heads * c, "pv": 2 * hd, "o_quant": hd, "wo": 2 * w, "attn_residual": 2 * w,
             "mlp_norm": 2 * w, "up_quant": w, "wup": 2 * f, "down_quant": f, "wdown": 2 * w, "mlp_residual": 2 * w}
    groups = [[Table("embed", 2 * w, False)]]
    groups += [[Table(f"l{i}.{op}", sizes[op], op.endswith("_quant")) for op in LAYER_OPS] for i in range(shape.layers)]
    groups.append([Table("final_norm", 2 * w, False), Table("head_quant", w, True), Table("head", 2 * v, False),
                   Table("argmax", 2, False)])
    return groups


def _slots() -> tuple[Any, Any]:
    if str(COMMIT_SLOTS) not in sys.path:
        sys.path.insert(0, str(COMMIT_SLOTS))
    import slots as S
    import slots_gpu as G
    return S, G


def layout(shape: Shape) -> Any:
    """The step's `slots.Layout`: each group of `tables`, its tensors the tables' codes (and a quantizer's scale after its
    codes), each cut into units of at most ``slots.UNIT_MAX_BITS``."""
    return _layout(shape)[0]


def _layout(shape: Shape) -> tuple[Any, dict[str, tuple[int, int, bool]]]:
    """The layout, and each table's (first unit, code units, whether its scale is a unit of its own after them)."""
    S, _ = _slots()
    cap = S.UNIT_MAX_BITS // 8
    groups, spans, u0 = [], {}, 0
    for group in tables(shape):
        sources: list[int] = []
        units: list[tuple[Any, ...]] = []
        for t in group:
            src, first = len(sources), u0 + len(units)
            sources.append(t.nbytes)
            if t.quant:
                sources.append(4)
                if t.nbytes + 4 <= cap:
                    units.append((S.Row((S.Part(src, 0, t.nbytes), S.Part(src + 1, 0, 4))),))
                    spans[t.op] = (first, 1, False)
                    continue
                cut = S._cut(src, 0, t.nbytes, cap)
                units += cut + [(S.Row((S.Part(src + 1, 0, 4),)),)]
                spans[t.op] = (first, len(cut), True)
            else:
                cut = S._cut(src, 0, t.nbytes, cap)
                units += cut
                spans[t.op] = (first, len(cut), False)
        groups.append(S.Group(tuple(sources), tuple(units)))
        u0 += len(units)
    return S.Layout(f"e2e-h200/{shape.name}/c{shape.context}/every-op/v1", tuple(groups)), spans


def row_bytes(t: Any) -> Any:
    """A (B, ...) tensor or array as (B, n) s8 bytes, n even (one zero byte appended where odd)."""
    if isinstance(t, np.ndarray):
        a = np.ascontiguousarray(t).reshape(t.shape[0], -1).view(np.int8)
        return a if a.shape[1] % 2 == 0 else np.concatenate([a, np.zeros((a.shape[0], 1), np.int8)], 1)
    a = t.contiguous().reshape(t.shape[0], -1).view(torch.int8)
    return a if a.shape[1] % 2 == 0 else torch.cat([a, torch.zeros(a.shape[0], 1, dtype=torch.int8, device=a.device)], 1)


@dataclass
class StepTables:
    """One step's tables: ``tables[op_path] = (codes (B, n) s8, word (B,) f32)``, in the order the step wrote them; slot b
    is the KV pool's slot ``slots[b]`` at position ``positions[b]``."""
    step: int
    positions: np.ndarray
    batch: int
    tables: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    slots: np.ndarray | None = None
    salt_key: bytes | None = None
    pending: Any = None

    def __post_init__(self) -> None:
        self.positions = np.broadcast_to(np.asarray(self.positions, np.int64), (self.batch,)).copy()
        if self.slots is None:
            self.slots = np.arange(self.batch, dtype=np.int64)

    @property
    def position(self) -> int:
        """The step's one position (a lockstep step's)."""
        if self.batch and (self.positions != self.positions[0]).any():
            raise ValueError(f"step {self.step} runs its slots at several positions")
        return int(self.positions[0]) if self.batch else 0

    def row_of(self, slot: int) -> int:
        """The row of sequence slot ``slot`` in this step's tables."""
        rows = np.flatnonzero(self.slots == slot)
        if not len(rows):
            raise KeyError(f"slot {slot} has no row at step {self.step}")
        return int(rows[0])

    def put(self, op: str, value: Any, word: Any = None) -> None:
        if op in self.tables:
            raise ValueError(f"{op} written twice at step {self.step}")
        codes = row_bytes(value)
        if word is None and isinstance(codes, np.ndarray):
            word = np.zeros(self.batch, np.float32)
        elif word is None:
            word = torch.zeros(self.batch, dtype=torch.float32, device=codes.device)
        self.tables[op] = (codes, word)

    def alloc(self, t: Table, device: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Table ``t``'s codes (B, nbytes) s8 and word (B,) f32, to be written in place (the word stays +0 unless ``t.quant``)."""
        if t.op in self.tables:
            raise ValueError(f"{t.op} written twice at step {self.step}")
        codes = torch.empty((self.batch, t.nbytes), dtype=torch.int8, device=device)
        word = torch.zeros(self.batch, dtype=torch.float32, device=device)
        self.tables[t.op] = (codes, word)
        return codes, word

    def drop(self, ops: list[str]) -> None:
        for op in ops:
            del self.tables[op]

    def ids(self) -> list[tuple[str, int]]:
        return [(f"s{int(s):06d}", int(p)) for s, p in zip(self.slots, self.positions, strict=True)]

    def host(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        out = {}
        for op, (c, w) in self.tables.items():
            out[op] = (c if isinstance(c, np.ndarray) else c.cpu().numpy(), w if isinstance(w, np.ndarray) else w.cpu().numpy())
        return out


@dataclass(frozen=True)
class Roots:
    root: bytes
    tables: tuple[tuple[str, bytes], ...] = ()
    slot_roots: tuple[bytes, ...] = ()


class Committer:
    """``begin`` a step, ``group`` each of its groups as its tables are written (in `tables` order), ``finish`` it."""

    def __init__(self, backend: str, context: bytes, key: bytes, shape: Shape | None = None, device: Any = "cpu",
                 salt: Callable[[int], bytes] | None = None) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"the commit backend is one of {BACKENDS}, not {backend!r}")
        if len(context) != 64 or len(key) != 32:
            raise ValueError("the context is 64 bytes (the live coin commitment's root) and the row key 32")
        self.backend, self.context, self.key, self.salt = backend, context, key, salt
        self.streams = backend == "slots"
        if self.streams:
            if shape is None:
                raise ValueError("the slots backend lays out a shape's tables")
            S, G = _slots()
            self.S, self.G, self.E = S, G, G.E
            self.groups = tables(shape)
            self.layout, self.spans = _layout(shape)
            self.cap = S.UNIT_MAX_BITS // 8
            dev = torch.device(device)
            self.plan = G.Plan(context, self.layout, device=dev if dev.type == "cuda" else "host")

    def describe(self) -> dict:
        if not self.streams:
            return {"backend": self.backend, "layout": "table-major: a table an op, a row a slot"}
        S = self.S
        return {"backend": "slots", "layout": self.layout.name, "layout_digest": S.layout_digest(self.layout).hex(),
                "units_a_slot": self.layout.unit_count, "rows_a_slot": self.layout.row_count,
                "bytes_a_slot": self.layout.slot_bytes, "row_schema": S.ROW_SCHEMA,
                "salt_key": "fixture (the run id's)" if self.salt else "fresh a step (os.urandom)",
                "engine": "host twin" if self.plan.host else "CUDA side stream"}

    def begin(self, st: StepTables, salt_key: bytes | None = None, nonce0: int = 0) -> None:
        """``nonce0``: slot b's salts use nonce ``nonce0 + b`` (a replay of slot s alone passes s)."""
        if not self.streams:
            return
        st.salt_key = salt_key or (self.salt(st.step) if self.salt else self.S.fresh_salt_key())
        st.pending = self.G.StepCommitter(self.plan, st.step, st.batch, st.salt_key)
        st.pending.salts = self.E.SaltKey(st.salt_key, nonce0)

    def group(self, st: StepTables, i: int) -> None:
        if not self.streams:
            return
        tensors = []
        for t in self.groups[i]:
            codes, word = st.tables[t.op]
            tensors += [codes, word] if t.quant else [codes]
        if self.plan.host:
            tensors = [x if isinstance(x, np.ndarray) else x.contiguous().numpy() for x in tensors]
        st.pending.add(tensors)

    def finish(self, st: StepTables) -> Roots:
        if self.streams:
            res = st.pending.finish().result()
            st.pending = None
            return Roots(res.root, (), res.slot_roots)
        return self.commit(st)

    def commit(self, st: StepTables) -> Roots:
        """A whole step's tables at once (the by-table backends)."""
        from verity_vllm.commit import row_tables as RTB
        ids = st.ids()
        if self.backend == "device":
            from verity_vllm.commit import row_tables_device as D

            def cuda(a: Any) -> Any:
                return torch.from_numpy(np.ascontiguousarray(a)).cuda() if isinstance(a, np.ndarray) else a.cuda()
            outs = {op: (ids, cuda(c), cuda(w)) for op, (c, w) in st.tables.items()}
            r = D.step_roots(self.context, st.step, outs, self.key).result()
        else:
            outs = {op: (ids, c, w) for op, (c, w) in st.host().items()}
            if self.backend == "host":
                from verity_vllm.commit import row_tables_device as D
                r = D.host_step(self.context, st.step, outs, self.key)
            else:
                r = RTB.step_roots(self.context, st.step, outs, self.key)
        return Roots(r.root, tuple(r.tables))

    def units_of(self, op: str, lo: int, hi: int, scale: bool = False) -> list[int]:
        """The commit units holding bytes ``lo .. hi - 1`` of table ``op``'s row (and its scale, with ``scale``)."""
        first, n, own = self.spans[op]
        return list(range(first + lo // self.cap, first + (hi - 1) // self.cap + 1)) + ([first + n] if scale and own else [])

    def slot_rows(self, st: StepTables, b: int) -> list[bytes]:
        """Slot b's committed rows from a kept step's tables (`slots.slot_rows`), for the reference's `commit_slot`."""
        outs = []
        for group in self.groups:
            ts = []
            for t in group:
                codes, word = st.tables[t.op]
                ts += [np.asarray(codes), np.asarray(word)] if t.quant else [np.asarray(codes)]
            outs.append(ts)
        return self.S.slot_rows(self.layout, outs, b)


def fixture_secrets(run_id: str) -> tuple[bytes, bytes]:
    """A FIXTURE context and row key from the run id: the live coin commitment's root and the run's row key replace them."""
    ctx = hashlib.sha512(b"e2e-h200/context-fixture\0" + run_id.encode()).digest()
    key = hashlib.sha512(b"e2e-h200/row-key-fixture\0" + run_id.encode()).digest()[:32]
    return ctx, key


def fixture_salt(run_id: str) -> Callable[[int], bytes]:
    """A FIXTURE salt key a step, from the run id (a test's roots then repeat); a run of record draws them fresh."""
    return lambda step: hashlib.sha512(b"e2e-h200/salt-key-fixture\0" + run_id.encode() + step.to_bytes(4, "big")).digest()[:32]
