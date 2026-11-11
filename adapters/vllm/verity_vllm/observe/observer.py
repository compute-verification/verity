"""The observer (runtime side): a `TorchDispatchMode` that records every dispatcher call as facts.

For each call the mode writes `DispatchEnter(id, parent, step, phase, op, schema, args, kwargs, stream, module)`,
runs the op, and writes `DispatchExit(id, outs)`.  `parent` is the innermost open recorded call.  Custom ops in
`REENTER_OPS` (vLLM's opaque attention / KV-update / MoE wrappers) are *re-entered*: the mode is popped while its
handler runs, so the op's Python body would otherwise be invisible; instead the body (the Python function vLLM
registered with `direct_register_custom_op`) is executed under a **fresh nested mode instance** sharing the same
recorder, so the body's dispatcher calls (`_C.reshape_and_cache_flash`, the FA2 op, ...) appear as children of the
outer call.  Calling the `OpOverload` again under the nested mode would re-enter this handler forever, hence the
direct call of the registered implementation (the trick of `verity_vllm/observe/capture.py:493-511`).

Non-interference: the observer never mutates tensors, never synchronises, keeps no tensor references beyond the
call (only `ViewDesc`s, which are plain ints/strings), and records a `Note` before re-raising an exception.

`encode_arg` is the one encoder for arguments / outputs: tensors -> `ViewDesc` via `StorageAdapter.view_of`;
None / bool / int / float / str as they are; lists and tuples -> lists; dtype / device / memory_format / layout ->
their `str`; `SymInt`/`SymFloat`/`SymBool` -> plain numbers; everything else -> `{"$repr": repr(x)[:200]}`.
"""

from __future__ import annotations

import importlib
import re
import time
from functools import partial
from typing import Any, Callable

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .events import DispatchEnter, DispatchExit, Event, Note
from .storage import StorageAdapter

# custom ops whose registered Python body is re-entered under a nested mode (children recorded)
REENTER_OPS = {"vllm.unified_attention_with_output", "vllm.unified_kv_cache_update"}
REENTER_PREFIXES = ("vllm.moe_forward",)

# where vLLM defines the functions it registered as the CUDA implementation of those ops
# (`direct_register_custom_op(op_name=..., op_func=<this function>)`); first hit wins, verified on the pinned wheel
REENTER_IMPL_PATHS: dict[str, list[tuple[str, str]]] = {
    "vllm.unified_attention_with_output": [
        ("vllm.model_executor.layers.attention.attention", "unified_attention_with_output"),  # d9105ea80
        ("vllm.attention.layer", "unified_attention_with_output"),                              # older layouts
    ],
    "vllm.unified_kv_cache_update": [
        ("vllm.model_executor.layers.attention.attention", "unified_kv_cache_update"),
        ("vllm.attention.layer", "unified_kv_cache_update"),
    ],
    "vllm.moe_forward": [
        ("vllm.model_executor.layers.fused_moe.runner.moe_runner", "_moe_forward"),
        ("vllm.model_executor.layers.fused_moe.layer", "moe_forward"),
    ],
    "vllm.moe_forward_shared": [
        ("vllm.model_executor.layers.fused_moe.runner.moe_runner", "_moe_forward_shared"),
        ("vllm.model_executor.layers.fused_moe.layer", "moe_forward_shared"),
    ],
}
_REENTER_IMPL: dict[str, Any] = {}
#: why each path of an op whose implementation was not found failed (the re-entry note carries it)
_REENTER_MISSING: dict[str, list[str]] = {}
_MISSING_IMPL_NOTED: set[str] = set()


def reenters(op_base: str) -> bool:
    return op_base in REENTER_OPS or op_base.startswith(REENTER_PREFIXES)


def reenter_impl(op_base: str) -> Any | None:
    """The Python function registered as the op's real implementation, or None when not found."""
    if op_base in _REENTER_IMPL:
        return _REENTER_IMPL[op_base]
    fn, why = None, []
    for mod_name, attr in REENTER_IMPL_PATHS.get(op_base, []):
        try:
            fn = getattr(importlib.import_module(mod_name), attr)
            break
        except Exception as e:  # noqa: BLE001
            why.append(f"{mod_name}.{attr}: {e!r}"[:200])
    _REENTER_IMPL[op_base] = fn
    if fn is None:
        _REENTER_MISSING[op_base] = why
    return fn


def reenter_impl_name(op_base: str) -> str | None:
    fn = reenter_impl(op_base)
    if fn is None:
        return None
    inner = getattr(fn, "__wrapped__", None)
    target = inner if inner is not None else fn
    return f"{target.__module__}.{target.__qualname__}"


def op_name(func: Any) -> str:
    """'aten.mm.default', '_C.fused_add_rms_norm.default', 'vllm.unified_attention_with_output.default'."""
    ns = func.namespace
    name = func._schema.name.split("::")[-1]
    ov = getattr(func, "_overloadname", "") or "default"
    return f"{ns}.{name}.{ov}"


def op_base_of(name: str) -> str:
    return name.rsplit(".", 1)[0]


_SYM_TYPES = tuple(t for t in (getattr(torch, "SymInt", None), getattr(torch, "SymFloat", None),
                               getattr(torch, "SymBool", None)) if t is not None)
_STR_TYPES = tuple(t for t in (torch.dtype, torch.device, torch.memory_format, torch.layout) if t is not None)


def encode_arg(x: Any, storage: StorageAdapter) -> Any:
    if isinstance(x, torch.Tensor):
        return storage.view_of(x)
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (list, tuple)):
        return [encode_arg(v, storage) for v in x]
    if isinstance(x, _STR_TYPES):
        return str(x)
    if _SYM_TYPES and isinstance(x, _SYM_TYPES):
        try:
            v = x.node.maybe_as_int() if hasattr(x, "node") else None
            return int(v) if v is not None else int(x)
        except Exception:  # noqa: BLE001
            try:
                return float(x)
            except Exception:  # noqa: BLE001
                return {"$repr": repr(x)[:200]}
    if isinstance(x, dict):
        return {str(k): encode_arg(v, storage) for k, v in x.items()}
    tn = type(x).__name__
    if tn == "constexpr" and hasattr(x, "value"):          # triton.language.constexpr
        return encode_arg(x.value, storage)
    if tn == "dtype" and type(x).__module__.startswith("triton"):
        return str(x)
    return {"$repr": repr(x)[:200]}


class Recorder:
    """State shared by every mode instance, the Triton adapter and the step wrappers: event ids, the stack of open
    recorded calls, the current step / phase, the module-name stack, the sink, optional snapshot hooks."""

    def __init__(self, storage: StorageAdapter, sink: Callable[[Event], None]) -> None:
        self.storage = storage
        self.sink = sink
        self.next_id = 1
        self.open: list[int] = []
        self.step = -1
        self.phase = "other"
        self.active = False
        self.module_stack: list[str] = []
        self.after_call: Callable[[int, str, tuple, dict, Any], None] | None = None
        self.after_launch: Callable[[int, str, dict], None] | None = None
        self.counts = {"dispatch": 0, "triton": 0, "notes": 0, "reentered": 0}
        self.encode_s = 0.0
        self._cuda = torch.cuda.is_available()

    def new_id(self) -> int:
        i = self.next_id
        self.next_id += 1
        return i

    def parent(self) -> int | None:
        return self.open[-1] if self.open else None

    def stream(self) -> int:
        if not self._cuda:
            return 0
        try:
            return int(torch.cuda.current_stream().cuda_stream)
        except RuntimeError:
            return -1

    def module_hint(self) -> str | None:
        return self.module_stack[-1] if self.module_stack else None

    def emit(self, ev: Event) -> None:
        st = self.storage
        if st.has_pending():
            for e in st.events_pending():
                self.sink(e)
        self.sink(ev)

    def note(self, text: str, **data: Any) -> None:
        self.counts["notes"] += 1
        self.emit(Note(text, data))


class Observer(TorchDispatchMode):
    """Records every dispatcher call while entered (`Recorder.active` must be True; otherwise pass-through).
    `reenter=False` disables the nested re-entry of custom ops (M0 rung c)."""

    def __init__(self, rec: Recorder, *, reenter: bool = True) -> None:
        super().__init__()
        self.rec = rec
        self.reenter = reenter

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        rec = self.rec
        if not rec.active:
            return func(*args, **kwargs)
        eid = rec.new_id()
        name = op_name(func)
        t0 = time.perf_counter()
        try:
            enc_args = [encode_arg(a, rec.storage) for a in args]
            enc_kwargs = {k: encode_arg(v, rec.storage) for k, v in kwargs.items()}
        except Exception as e:  # noqa: BLE001
            enc_args, enc_kwargs = [], {}
            rec.note("encode error (args)", id=eid, op=name, error=repr(e)[:300])
        try:
            schema = str(func._schema)
        except AttributeError:
            schema = ""
        rec.counts["dispatch"] += 1
        rec.emit(DispatchEnter(eid, rec.parent(), rec.step, rec.phase, name, schema, enc_args, enc_kwargs,
                               rec.stream(), rec.module_hint()))
        rec.encode_s += time.perf_counter() - t0
        rec.open.append(eid)
        try:
            impl = None
            if self.reenter and reenters(op_base_of(name)):
                impl = reenter_impl(op_base_of(name))
                if impl is None and op_base_of(name) not in _MISSING_IMPL_NOTED:
                    _MISSING_IMPL_NOTED.add(op_base_of(name))
                    rec.note("re-entry impl not found; op recorded without children", op=name,
                             tried=_REENTER_MISSING.get(op_base_of(name), []))
            if impl is not None:
                rec.counts["reentered"] += 1
                with Observer(rec, reenter=True):
                    out = impl(*args, **kwargs)
            else:
                out = func(*args, **kwargs)
        except BaseException as e:
            _pop(rec, eid)
            rec.note("exception in dispatcher call", id=eid, op=name, error=repr(e)[:300])
            rec.emit(DispatchExit(eid, None))
            raise
        _pop(rec, eid)
        t1 = time.perf_counter()
        try:
            enc_out = encode_arg(out, rec.storage)
        except Exception as e:  # noqa: BLE001
            enc_out = None
            rec.note("encode error (outs)", id=eid, op=name, error=repr(e)[:300])
        out = _defined_tensor_list(rec, name, out)
        rec.emit(DispatchExit(eid, enc_out))
        rec.encode_s += time.perf_counter() - t1
        if rec.after_call is not None:
            rec.after_call(eid, name, args, kwargs, out)
        return out


_UNDEFINED_RETURN_NOTED: set[str] = set()


def _defined_tensor_list(rec: Recorder, name: str, out: Any) -> Any:
    """An op whose `Tensor[]` return carries UNDEFINED tensors (vLLM's FA3 `_vllm_fa3_C.fwd` returns `[out, softmax_lse,
    out_accum, lse_accum]` with the last two undefined when `num_splits == 1`) surfaces them here as `None`; handing that list back to
    the dispatcher fails its `Tensor[]` cast (`pushPyOutToStack`: "Unable to cast Python instance of type list"), so the op cannot run
    under any TorchDispatchMode.  The recorded outputs (`enc_out`, encoded before this) keep the `None`s; only what is RETURNED to the
    caller gets an empty placeholder per undefined slot (vLLM discards those slots)."""
    if not (isinstance(out, list) and out and any(x is None for x in out) and any(isinstance(x, torch.Tensor) for x in out)):
        return out
    dev = next(x.device for x in out if isinstance(x, torch.Tensor))
    if name not in _UNDEFINED_RETURN_NOTED:
        _UNDEFINED_RETURN_NOTED.add(name)
        rec.note("undefined Tensor[] return elements replaced by empty placeholders for the dispatcher (recorded as null)", op=name,
                 undefined=[i for i, x in enumerate(out) if x is None])
    return [x if x is not None else torch.empty(0, device=dev) for x in out]


def _pop(rec: Recorder, eid: int) -> None:
    if rec.open and rec.open[-1] == eid:
        rec.open.pop()
    elif eid in rec.open:
        rec.open.remove(eid)


class EmptyMode(TorchDispatchMode):
    """M0 rung (b): a mode that only forwards the call."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


_LAYER_PREFIX = re.compile(r"^(.*?layers\.\d+\.)")


class ModuleNamer:
    """Forward pre/post hooks on every named submodule maintain `Recorder.module_stack` (a diagnostic hint on every
    DispatchEnter).  A module instance shared across layers (vLLM's cached rotary embedding) is registered under
    the first layer's name; it is reported under the layer that is calling it.  `remove()` uninstalls."""

    def __init__(self, rec: Recorder, model: torch.nn.Module) -> None:
        self.rec = rec
        self.handles = []
        for name, mod in model.named_modules():
            if name == "":
                continue
            self.handles.append(mod.register_forward_pre_hook(partial(self._pre, name)))
            self.handles.append(mod.register_forward_hook(partial(self._post, name)))

    def _pre(self, name: str, mod: Any, args: Any) -> None:
        stack = self.rec.module_stack
        if stack:
            m_par = _LAYER_PREFIX.match(stack[-1])
            m_own = _LAYER_PREFIX.match(name)
            if m_par and m_own and m_par.group(1) != m_own.group(1):
                name = m_par.group(1) + name[len(m_own.group(1)):]
        stack.append(name)

    def _post(self, name: str, mod: Any, args: Any, output: Any) -> None:
        stack = self.rec.module_stack
        if stack:
            stack.pop()

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
