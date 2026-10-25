"""The fallback serve-time commit call: a step's unit outputs committed by vLLM's existing row-table committer, as it is.

``commit_step(unit_outputs, salt_source) -> StepCommit`` (``.root``: 64 bytes) commits one row-table segment with
`verity_vllm.commit.row_tables_device.step_roots` (``row_table.cu`` on a side stream) for CUDA tensors, or its CPU reference
`verity_vllm.commit.row_tables.step_roots` for host arrays; ``commit_step_async`` returns the device path's `Pending`.
Nothing of the committer is copied here: this module only lays unit outputs out as the committer takes them.

**Layout, by table.** One table per unit output (``op_path``), its rows the batch's sequences, one ``hm96-sha512/row-seg/v1``
hiding leaf each, in a ``frame-v3-sha512`` tree under ``row_table.table_domain(context, segment, index, B)``; the segment's
root is the tree over its tables' roots (``row_table.step_tree``).  Tables go in ``row_tables.table_key`` order (layer, then
op_path), each table's rows in ``(request, token row)`` order (``rows``; by default sequence slot ``b`` is ``("%08d" % b,
0)``, so rank = slot).  A row is ``row_table.s8_row(codes, scale)``: a quantizer output's s8 codes then its f32 scale, given
as ``(codes, scales)``; any other unit output (bf16, i32, ...) is given as one tensor whose row bytes are the codes, with
scale +0.0f.  Salts are ``ml.windows.rows.segment_salts(key, segment, n)``: ChaCha20 under the run's row key with the
segment as nonce, row ``g`` of the segment (tables in order, rows in rank order) taking blocks ``3g .. 3g + 2``.

**A step that doesn't fit.** The committer hashes the values it is handed, in device memory.  At shape H's planned batch a
step's unit outputs are ~717 GB (`shape_h`), so the loop commits each layer as it finishes, one segment a layer
(`segment_of`), and records each segment's root with the challenger; a window's ``row-table`` port is the tree over its
segments' roots (``ml.windows.rows.Window``), as for vLLM's engine steps.

**Opening.** ``StepCommit`` keeps each table's root; `open_rows` rebuilds one table on the CPU from that table's rows (every
sequence's, which the by-table layout needs) and opens chosen ranks against the segment's root (``row_table.StepOpening``,
checked by ``row_table.verify_step``).  The per-sequence-slot layout (KCOMMIT, HASH-13) replaces this: there a drawn unit
opens from its own request's replay.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from verity.ml.commitments import row_table as RT
from verity.ml.windows import rows as CS
from verity_vllm.commit import row_tables as RTB
from verity_vllm.commit import row_tables_device as RTD

__all__ = ["SaltSource", "StepCommit", "commit_step", "commit_step_async", "open_rows", "reference", "segment_of", "shape_h",
           "slot_rows"]

#: a sequence slot's row id: ``(request, token row)`` sorting in slot order
SLOT_FORMAT = "%08d"


@dataclass(frozen=True)
class SaltSource:
    """Where a segment's salts and domains come from: ``key``, the run's 32-byte row key (ChaCha20's key; the prover's own
    randomness, never shown), ``segment``, the salt stream's nonce and the domains' step (`segment_of`), and ``context``, 64
    bytes the loop knows before its first step (the live coin commitment's root)."""
    key: bytes
    segment: int
    context: bytes

    def __post_init__(self) -> None:
        if len(self.key) != 32 or len(self.context) != 64 or not 0 <= int(self.segment) < 1 << 64:
            raise ValueError("a salt source is a 32-byte row key, a segment in [0, 2^64) and a 64-byte context")


@dataclass(frozen=True)
class StepCommit:
    """A committed segment: ``root`` (64 bytes, what the loop records before the coin), ``tables`` each table's ``(op_path,
    root)`` in the segment's order, ``rows`` the batch's row ids and ``ranks[b]`` batch row ``b``'s leaf index in every table."""
    root: bytes
    tables: tuple[tuple[str, bytes], ...]
    rows: tuple[tuple[str, int], ...]
    ranks: tuple[int, ...]

    def index(self, op_path: str) -> int:
        return [op for op, _r in self.tables].index(op_path)


def slot_rows(batch: int) -> list[tuple[str, int]]:
    """The default row ids: slot ``b`` is ``(SLOT_FORMAT % b, 0)``, so a table's rank order is slot order."""
    return [(SLOT_FORMAT % b, 0) for b in range(batch)]


def segment_of(engine_step: int, part: int, parts: int) -> int:
    """The segment of ``part`` (a layer, or the head after the layers) of engine step ``engine_step`` when a step is committed
    in ``parts`` segments."""
    if not 0 <= part < parts:
        raise ValueError(f"part {part} of {parts}")
    return engine_step * parts + part


def _ranks(rows: Sequence[tuple[str, int]]) -> tuple[int, ...]:
    keys = [(str(r), int(t)) for r, t in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("a (request, token row) twice in one batch")
    perm = sorted(range(len(keys)), key=keys.__getitem__)
    ranks = [0] * len(keys)
    for r, b in enumerate(perm):
        ranks[b] = r
    return tuple(ranks)


def _host_table(op: str, v: Any, batch: int) -> tuple[np.ndarray, np.ndarray]:
    """A unit output as host ``(codes (B, K) int8, scales (B,) f32)``."""
    if isinstance(v, tuple):
        codes, scales = (np.ascontiguousarray(np.asarray(_host(x))) for x in v)
        if codes.dtype != np.int8 or codes.ndim != 2 or scales.dtype != np.float32 or scales.shape != (codes.shape[0],):
            raise ValueError(f"{op}: a quantizer output is (codes (B, K) int8, scales (B,) float32)")
    else:
        a = np.ascontiguousarray(np.asarray(_host(v)))
        if a.ndim < 1:
            raise ValueError(f"{op}: a unit output has one row per sequence")
        codes = a.reshape(a.shape[0], -1).view(np.int8)
        scales = np.zeros(codes.shape[0], dtype=np.float32)
    _checked(op, codes.shape, batch)
    return codes, scales


def _checked(op: str, shape: tuple[int, ...], batch: int) -> None:
    if len(shape) != 2 or shape[0] != batch or shape[1] < 2 or shape[1] % 2:
        raise ValueError(f"{op}: rows of {tuple(shape)} bytes; every table has the batch's {batch} rows of an even number "
                         f"(>= 2) of bytes before the scale (the committer's K)")


def _host(x: Any) -> Any:
    """A torch tensor on the host as numpy (bf16 by its bytes); anything else as is."""
    if type(x).__module__.startswith("torch"):
        import torch

        t = x.detach().cpu().contiguous()
        return t.view(torch.uint8).numpy() if t.dtype == torch.bfloat16 else t.numpy()
    return x


def _host_tables(unit_outputs: Mapping[str, Any], batch: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {op: _host_table(op, v, batch) for op, v in unit_outputs.items()}


def _batch(unit_outputs: Mapping[str, Any]) -> int:
    if not unit_outputs:
        raise ValueError("no unit outputs")
    v = next(iter(unit_outputs.values()))
    return int((v[0] if isinstance(v, tuple) else v).shape[0])


def _on_device(unit_outputs: Mapping[str, Any]) -> bool:
    v = next(iter(unit_outputs.values()))
    x = v[0] if isinstance(v, tuple) else v
    return getattr(getattr(x, "device", None), "type", None) == "cuda"


def _device_tables(unit_outputs: Mapping[str, Any], batch: int) -> dict[str, tuple[Any, Any]]:
    """Unit outputs as the device committer takes them, without copying a value: a tensor's rows viewed as int8 bytes, one
    zero scale vector shared by the tables that have none."""
    import torch

    first = next(iter(unit_outputs.values()))
    dev = (first[0] if isinstance(first, tuple) else first).device
    zero = torch.zeros(batch, dtype=torch.float32, device=dev)
    out = {}
    for op, v in unit_outputs.items():
        if isinstance(v, tuple):
            codes, scales = v
        else:
            if not v.is_contiguous():
                raise ValueError(f"{op}: a unit output is a contiguous tensor (the committer reads it in place)")
            codes, scales = v.reshape(v.shape[0], -1).view(torch.int8), zero
        _checked(op, tuple(codes.shape), batch)
        out[op] = (codes, scales)
    return out


def commit_step_async(unit_outputs: Mapping[str, Any], salt_source: SaltSource, *, rows: Sequence[tuple[str, int]] | None = None,
                      stream: Any = None, timing: bool = False) -> tuple[RTD.Pending, tuple[tuple[str, int], ...], tuple[int, ...]]:
    """The device path of `commit_step`, not waiting: `row_tables_device.step_roots`'s `Pending` (``ready()``, ``result()``,
    ``kernel_ms()`` with ``timing``), the row ids and each batch row's rank.  The tensors must hold until it is ready."""
    batch = _batch(unit_outputs)
    ids = list(rows) if rows is not None else slot_rows(batch)
    if len(ids) != batch:
        raise ValueError(f"{len(ids)} row ids for a batch of {batch}")
    tables = _device_tables(unit_outputs, batch)
    p = RTD.step_roots(salt_source.context, int(salt_source.segment), {op: (ids, c, s) for op, (c, s) in tables.items()},
                       salt_source.key, stream=stream, timing=timing)
    return p, tuple((str(r), int(t)) for r, t in ids), _ranks(ids)


def commit_step(unit_outputs: Mapping[str, Any], salt_source: SaltSource, *, rows: Sequence[tuple[str, int]] | None = None,
                stream: Any = None) -> StepCommit:
    """Commit one segment of unit outputs: ``unit_outputs[op_path]`` is a tensor of B rows (any dtype, its row bytes the
    committed codes, scale +0.0f) or a quantizer output ``(codes (B, K) int8, scales (B,) float32)``; every table has the
    batch's B rows, ``rows`` their ids (default `slot_rows`).  CUDA tensors go to the device committer and the call waits for
    the root; host arrays to its CPU reference.  Same values, same root."""
    batch = _batch(unit_outputs)
    ids = list(rows) if rows is not None else slot_rows(batch)
    if len(ids) != batch:
        raise ValueError(f"{len(ids)} row ids for a batch of {batch}")
    if _on_device(unit_outputs):
        p, keys, ranks = commit_step_async(unit_outputs, salt_source, rows=ids, stream=stream)
        r = p.result()
    else:
        tables = _host_tables(unit_outputs, batch)
        r = RTB.step_roots(salt_source.context, int(salt_source.segment), {op: (ids, c, s) for op, (c, s) in tables.items()},
                           salt_source.key)
        keys, ranks = tuple((str(a), int(t)) for a, t in ids), _ranks(ids)
    return StepCommit(r.root, r.tables, keys, ranks)


def _row(codes: np.ndarray, scale: np.float32) -> tuple[bytes, int]:
    return RT.s8_row(codes.tobytes(), int(np.asarray(scale, dtype="<f4").view("<u4")))


def _salts(salt_source: SaltSource, first: int, n: int) -> list[bytes]:
    raw = CS.chacha20(salt_source.key, int(salt_source.segment).to_bytes(12, "little"), n * CS.SALT_BYTES,
                      counter=first * CS.SALT_BYTES // 64)
    return [raw[i * CS.SALT_BYTES:(i + 1) * CS.SALT_BYTES] for i in range(n)]


def reference(unit_outputs: Mapping[str, Any], salt_source: SaltSource, *,
              rows: Sequence[tuple[str, int]] | None = None) -> RT.Step:
    """The segment as `verity.ml.commitments.row_table.commit_step` commits it from the rows, salts and domains above: the
    reference `commit_step`'s root is held to, independent of vLLM's code, with its openings (``Step.open``)."""
    batch = _batch(unit_outputs)
    ids = list(rows) if rows is not None else slot_rows(batch)
    ranks = _ranks(ids)
    perm = sorted(range(batch), key=ranks.__getitem__)
    tables = _host_tables(unit_outputs, batch)
    order = sorted(tables, key=RTB.table_key)
    salts = _salts(salt_source, 0, batch * len(order))
    out = []
    for i, op in enumerate(order):
        codes, scales = tables[op]
        out.append(([_row(codes[b], scales[b]) for b in perm], salts[i * batch:(i + 1) * batch]))
    return RT.commit_step(salt_source.context, int(salt_source.segment), out)


def open_rows(commit: StepCommit, op_path: str, table: Any, salt_source: SaltSource,
              batch_rows: Sequence[int]) -> tuple[RT.StepOpening, dict[int, tuple[bytes, int, bytes]]]:
    """Open ``op_path``'s rows of batch rows ``batch_rows`` against the committed segment's root: the table rebuilt on the CPU
    from ``table`` (that unit output's every row, as `commit_step` took it) and the recorded table roots.  Returns the
    opening and the opened rows' ``{rank: (row, n_bits, salt)}``, which ``row_table.verify_step`` checks in the clear;
    refused when ``table`` is not the committed table."""
    batch = len(commit.rows)
    i = commit.index(op_path)
    codes, scales = _host_tables({op_path: table}, batch)[op_path]
    perm = sorted(range(batch), key=commit.ranks.__getitem__)
    salts = _salts(salt_source, i * batch, batch)
    tab = RT.commit(RT.table_domain(salt_source.context, int(salt_source.segment), i, batch),
                    [_row(codes[b], scales[b]) for b in perm], salts)
    if tab.root != commit.tables[i][1]:
        raise ValueError(f"{op_path}: these rows are not the committed table (its root differs)")
    ranks = sorted({commit.ranks[b] for b in batch_rows})
    tree = RT.step_tree(salt_source.context, int(salt_source.segment), [r for _op, r in commit.tables])
    if tree.root != commit.root:
        raise ValueError("the recorded table roots are not the segment's root")
    witness = {r: (*_row(codes[perm[r]], scales[perm[r]]), salts[r]) for r in ranks}
    return RT.StepOpening(tree.open([i]), tab.open(ranks)), witness


# --- the frozen shape -----------------------------------------------------------------------------------------------------

def shape_h(d: int = 16384, heads: int = 256, head_dim: int = 64, kv_heads: int = 1, mlp: int = 65536, ctx: int = 32,
            vocab: int = 256) -> dict[str, dict[str, tuple[str, int]]]:
    """Shape H's unit outputs per token (compute-accounting's frozen MQA shape, CMP-92's ops at d 16,384): ``{"layer": {name:
    (kind, n)}, "head": {...}}``, ``kind`` ``"s8"`` (a quantizer output: n codes and an f32 scale), ``"bf16"`` or ``"i32"`` (n
    elements).  A layer's table is ``model.layers.<l>.<name>``, the head's ``model.head.<name>``.  The attention scores and
    softmax rows are the ``ctx`` positions with the masked ones in, so a table's rows have one width."""
    qkv = heads * head_dim + 2 * kv_heads * head_dim
    layer = {
        "input_layernorm": ("bf16", d),
        "self_attn.qkv_proj.quant": ("s8", d),
        "self_attn.qkv_proj": ("bf16", qkv),
        "self_attn.qk": ("bf16", heads * ctx),
        "self_attn.softmax": ("bf16", heads * ctx),
        "self_attn.pv": ("bf16", heads * head_dim),
        "self_attn.o_proj.quant": ("s8", heads * head_dim),
        "self_attn.o_proj": ("bf16", d),
        "residual_attn": ("bf16", d),
        "post_attention_layernorm": ("bf16", d),
        "mlp.up_proj.quant": ("s8", d),
        "mlp.up_proj": ("bf16", mlp),
        "mlp.relu": ("bf16", mlp),
        "mlp.down_proj.quant": ("s8", mlp),
        "mlp.down_proj": ("bf16", d),
        "residual_mlp": ("bf16", d),
    }
    head = {"norm": ("bf16", d), "lm_head.quant": ("s8", d), "lm_head": ("bf16", vocab), "argmax": ("i32", 1)}
    return {"layer": layer, "head": head}


#: the units compute-accounting's "fused cuts" leave uncommitted (their outputs feed the next unit inside one proof unit)
FUSED_OUT = ("input_layernorm", "post_attention_layernorm", "mlp.relu", "self_attn.qk")
ELEMENT_BYTES = {"s8": 1, "bf16": 2, "i32": 4}


def row_bytes(kind: str, n: int) -> int:
    """A unit output's committed row in bytes: the elements, plus the f32 scale (a quantizer's, or +0.0f)."""
    return ELEMENT_BYTES[kind] * n + 4
