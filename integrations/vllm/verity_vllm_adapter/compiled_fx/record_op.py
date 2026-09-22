"""``verity::record(Tensor x, int id) -> ()``: the observation op inserted into the compiled graph.

* ``mutates_args=()``: it mutates nothing, so Inductor gives it only a *read* dependency on ``x``'s
  buffer (a ``StarDep``): consumers of ``x`` and the fusion decisions around ``x``'s producer are
  untouched. (A mutating schema would version the buffer and make every later consumer depend on the
  record call, splitting fusion groups.)
* ``torch.fx.node.has_side_effect`` marks the overload side-effectful so FX ``eliminate_dead_code``
  and Inductor's scheduler (``FallbackKernel.has_side_effects`` -> ``torch._library.utils.is_impure``)
  keep an op that has no outputs.
* Inductor lowers it as a fallback kernel: the generated wrapper contains a literal
  ``torch.ops.verity.record.default(bufN, <id>)`` line after the kernel that wrote ``bufN``, which is
  also how :mod:`.wrapper_parse` recovers ``id -> buffer -> producing kernel``.
* The implementation is the Python function below. With CUDA graphs off in replay it runs on every
  step and enqueues an async D2H copy of the buffer into a pinned host arena on the current stream
  (the same stream the producing kernel ran on), so later kernels / allocator reuse are ordered behind
  the copy (§15.4). A device-side ring-buffer implementation for CUDA-graph replay (rung (c)) would
  replace only this function.

The sink is process-global: the compiled wrapper calls the op with no way to pass a Python object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch.fx.node import has_side_effect

OP_NAME = "verity::record"
_INSTALLED: dict[str, Any] = {}


@dataclass
class Recorded:
    id: int
    seq: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    data_ptr: int
    host: torch.Tensor | None  # pinned snapshot (contiguous) or None when the sink is disabled
    runtime_piece: int = -1  # piecewise_compile_index of the PiecewiseBackend executing when the op ran


# Set by the replay hook around ``PiecewiseBackend.__call__``. Inductor's FX-graph cache serves identical
# middle-layer pieces from one compiled artifact, so the id prefix identifies the *compiled template*
# while this tells which piece (layer) is actually running.
RUNTIME: dict[str, int] = {"piece": -1}


@dataclass
class RecordSink:
    """Where ``verity::record`` puts what it sees. ``snapshot`` is ``PinnedArena.snapshot``-like:
    tensor -> pinned host tensor (async copy enqueued on the current stream)."""

    snapshot: Callable[[torch.Tensor], torch.Tensor] | None = None
    enabled: bool = False
    records: list[Recorded] = field(default_factory=list)
    calls_total: int = 0
    calls_while_disabled: int = 0
    seen_ids: dict[int, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.records.clear()
        self.seen_ids.clear()

    def on_record(self, x: torch.Tensor, id: int) -> None:
        self.calls_total += 1
        if not self.enabled:
            self.calls_while_disabled += 1
            return
        host = self.snapshot(x) if self.snapshot is not None else None
        self.seen_ids[id] = self.seen_ids.get(id, 0) + 1
        self.records.append(
            Recorded(id=int(id), seq=len(self.records), shape=tuple(x.shape), stride=tuple(x.stride()),
                     dtype=x.dtype, data_ptr=x.data_ptr(), host=host, runtime_piece=RUNTIME["piece"])
        )


SINK = RecordSink()


class DeviceRing:
    """Rung (c): the device-side implementation of ``verity::record`` for CUDA-graph replay.

    One preallocated device slot per ``(runtime piece, id)`` of ``rows`` rows. The op body does
    ``slot[:M].copy_(x)``; during CUDA-graph *capture* that copy is recorded into the graph and replays
    with it (the Python body does not run on replay), so the slot address is fixed at capture time and
    chosen from the piece the ``PiecewiseBackend`` wrapper published while capturing. Slots are
    allocated on the first non-capturing call for a key (vLLM's profile run and warm-up precede
    capture); allocating during capture is refused. After each engine step :meth:`drain` snapshots
    ``slot[:num_tokens]`` of every key into the pinned arena and returns ``Recorded`` rows in
    (piece, id) order, which is graph order. Batches larger than ``rows`` cannot be captured by vLLM
    either (``rows`` >= the largest CUDA-graph capture size); they run with the Python body and fall
    back to the host path (``stored`` False).
    """

    def __init__(self, rows: int) -> None:
        self.rows = int(rows)
        self.slots: dict[tuple[int, int], torch.Tensor] = {}
        self.written: set[tuple[int, int]] = set()  # keys written since the last drain by a non-captured call
        self.captured_keys: set[tuple[int, int]] = set()
        self.capture_calls = 0
        self.eager_calls = 0
        self.refused_capture_alloc: list[tuple[int, int]] = []
        self.oversize_calls = 0
        self.bytes_allocated = 0

    def write(self, piece: int, id: int, x: torch.Tensor) -> bool:
        key = (int(piece), int(id))
        capturing = bool(x.is_cuda and torch.cuda.is_current_stream_capturing())
        slot = self.slots.get(key)
        if slot is None:
            if capturing:
                self.refused_capture_alloc.append(key)
                return False
            slot = torch.empty((self.rows, *x.shape[1:]), dtype=x.dtype, device=x.device)
            self.slots[key] = slot
            self.bytes_allocated += slot.numel() * slot.element_size()
        m = int(x.shape[0])
        if m > self.rows:
            self.oversize_calls += 1
            return False
        slot[:m].copy_(x)
        if capturing:
            self.capture_calls += 1
            self.captured_keys.add(key)
        else:
            self.eager_calls += 1
            self.written.add(key)
        return True

    def drain(self, m: int, snapshot: Callable[[torch.Tensor], torch.Tensor], skip: set[tuple[int, int]] | None = None) -> list[Recorded]:
        out: list[Recorded] = []
        for key in sorted(self.slots):
            if skip and key in skip:
                continue
            slot = self.slots[key]
            view = slot[: int(m)]
            out.append(Recorded(id=key[1], seq=len(out), shape=tuple(view.shape), stride=tuple(view.stride()), dtype=view.dtype,
                                data_ptr=view.data_ptr(), host=snapshot(view), runtime_piece=key[0]))
        self.written.clear()
        return out

    def stats(self) -> dict[str, Any]:
        return {"rows": self.rows, "slots": len(self.slots), "bytes_allocated": self.bytes_allocated, "capture_calls": self.capture_calls,
                "captured_keys": len(self.captured_keys), "eager_calls": self.eager_calls, "oversize_calls": self.oversize_calls,
                "refused_capture_alloc": len(self.refused_capture_alloc)}


RING: DeviceRing | None = None  # set by the replay driver for rung (c); None -> host path only


def install() -> Any:
    """Define the op once per process and return the overload (``torch.ops.verity.record.default``)."""
    if "op" in _INSTALLED:
        return _INSTALLED["op"]

    @torch.library.custom_op(OP_NAME, mutates_args=())
    def record(x: torch.Tensor, id: int) -> None:
        ring = RING
        if ring is not None:
            # device path: every call (enabled or not, capturing or not) goes through the ring so slots exist
            # before capture and captured graphs copy into them on replay; the host path is used only when
            # the batch is larger than the ring (never inside a graph).
            if ring.write(RUNTIME["piece"], id, x):
                SINK.calls_total += 1
                if not SINK.enabled:
                    SINK.calls_while_disabled += 1
                return
        SINK.on_record(x, id)

    @record.register_fake
    def _(x: torch.Tensor, id: int) -> None:
        return None

    overload = torch.ops.verity.record.default
    has_side_effect(overload)
    _INSTALLED["op"] = overload
    _INSTALLED["custom_op"] = record
    return overload


def install_inductor_lowering() -> str:
    """Register ``verity::record``'s Inductor lowering explicitly (compile process only).

    Without this Inductor creates an *implicit* fallback for the op at first sight (``make_fallback``),
    which (1) adds it to ``needs_realized_inputs`` and (2) attaches the custom-op default layout
    constraint. (1) changes codegen of *other* kernels: ``GraphLowering.run_node`` calls
    ``realize_hint()`` on every pointwise node that has >1 users when one of them needs realised
    inputs, so a not-yet-materialised pointwise value we record (e.g. Qwen2's qkv bias add, recorded
    because a view of it is a graph output) gets materialised *before* its consumers are lowered, and a
    consumer that production inlined (k-rope reading the fp32 sum) now reads the bf16 buffer instead:
    different kernel source, different low bits (B1, §3 of compiled-fx-capture.md). (1) also feeds
    ``fx_utils.is_node_realized`` used by the reinplace pass. The explicit lowering is the same
    ``FallbackKernel`` (an extern call in the wrapper that realises its input at *its own* position,
    which with ``placement='late'`` is after the last reader), but the op is a member of neither
    ``needs_realized_inputs`` nor ``fallbacks``, so no other node's lowering can observe it.

    Returns ``"explicit"`` (registered here), ``"preexisting"`` (a lowering was already registered,
    e.g. an implicit fallback from an earlier compile in this process: codegen of pointwise producers
    may then differ from production) or the cached value.
    """
    if "lowering" in _INSTALLED:
        return str(_INSTALLED["lowering"])
    op = install()
    from torch._inductor import lowering as L  # heavy import; the compile process has it anyway

    if op in L.lowerings:
        _INSTALLED["lowering"] = "preexisting"
        return "preexisting"

    from torch._inductor import ir
    from torch._inductor.virtualized import V

    fallback = L.fallback_handler(op, add_to_fallback_set=False)

    def existing_materialisation(x: Any) -> Any:
        """If the recorded FX value is still an unrealised Pointwise/Reduction at the record's lowering
        point but Inductor already materialised a *clone* of it for an extern consumer, record that clone.

        Inductor's ``constrain_to_fx_strides`` (applied to every ``triton_kernel_wrapper_mutation``
        input and to custom ops with the default ``needs_fixed_stride_order`` constraint) turns an
        unrealised producer into ``copy_input``: a clone Pointwise that inlines the producer and is
        fused into the producer's kernel. The FX value itself never becomes a buffer (Qwen2: the
        rmsnorm output feeding vLLM's Triton GEMM, the silu_mul output). Realising it for the record
        adds a second store of the same registers to that kernel -> different source. The clone's loop
        body *is* the producer's (``copy_input`` uses ``x.make_loader()``, which for an unrealised
        ``Loops`` node is its ``inner_fn`` itself), so it is identified by ``inner_fn`` identity plus
        size/dtype and recorded instead; the value is the same (same fused computation, same registers)."""
        LOWERING_STATS["calls"] += 1
        if not isinstance(x, ir.TensorBox) or not isinstance(x.data, ir.StorageBox):
            LOWERING_STATS["view_or_other"] += 1
            return x
        loops = x.data.data
        if not isinstance(loops, (ir.Pointwise, ir.Reduction)):
            LOWERING_STATS["already_realised"] += 1
            return x
        n = getattr(V.graph, "current_node", None)
        src = n.args[0] if n is not None and n.args else None
        want_size = [str(s) for s in x.get_size()]
        for b in V.graph.buffers:
            if not (isinstance(b, ir.ComputedBuffer) and isinstance(b.data, ir.Loops)):
                continue
            if b.get_dtype() != x.get_dtype() or [str(s) for s in b.get_size()] != want_size:
                continue
            if b.data.inner_fn is loops.inner_fn or (src is not None and b.get_origin_node() is src):
                LOWERING_STATS["reused_clone"] += 1
                return ir.TensorBox(ir.StorageBox(b))
        LOWERING_STATS["realised_here"] += 1
        return x

    def handler(x: Any, id: Any) -> Any:
        return fallback(existing_materialisation(x), id)

    handler._is_fallback_handler = True  # type: ignore[attr-defined]  (compile_fx_ext classifies lowerings by this)
    L.register_lowering(op, type_promotion_kind=None)(handler)
    _INSTALLED["lowering"] = "explicit"
    return "explicit"


LOWERING_STATS: dict[str, int] = {"calls": 0, "already_realised": 0, "reused_clone": 0, "realised_here": 0, "view_or_other": 0}


def lowering_kind() -> str | None:
    v = _INSTALLED.get("lowering")
    return None if v is None else str(v)


def overload() -> Any:
    return install()
