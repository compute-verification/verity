"""Structural execution of the real vLLM model on the `meta` device.

Mechanism (Bet B):
  1. `build_vllm_config(model_path)` — vLLM's own `EngineArgs.create_engine_config()` for the eager bf16 TP1 profile
     (no engine, no scheduler, no weights).
  2. `instantiate_model(cfg)` — vLLM's own `initialize_model` under `torch.device("meta")` inside
     `set_current_vllm_config` (so `Attention` layers register in `static_forward_context`).  Parameters are meta
     tensors: NO values exist anywhere.  Weights are never loaded.
  3. `StructuralMode` — an outermost `TorchDispatchMode`.  Every dispatcher call is recorded as `OpCall(op, args,
     kwargs, outs)` with tensors encoded as `TView(storage id, offset, shape, strides, dtype)`.  Calls in `OPAQUE`
     (vLLM custom ops / C++ kernels with no meta kernel) are NOT executed: their outputs are synthesised by a
     per-op *meta rule* (`META_RULES`, shape-only, counted).  vLLM's re-entrant wrappers (`vllm.unified_attention_with_output`,
     `vllm.unified_kv_cache_update`) are re-entered exactly like `verity_capture.observer` so their children
     (`_C.reshape_and_cache_flash`, `_vllm_fa2_C.varlen_fwd`) are visible.  Everything else runs the meta kernel.
  4. Triton launches (`JITFunction.run`) are intercepted the way `verity_capture.triton_adapter` does, but NOT run:
     the bound arguments (incl. constexprs) are recorded as a `TritonCall`.
  5. `run_step(...)` builds the fake `ForwardContext` / `FlashAttentionMetadata` / KV caches for one engine step of a
     single request (scaffolding, counted) and runs `model(...)` + `compute_logits(...)`.

Integer index tensors that carry structural meaning (input_ids, positions, slot_mapping, seq_lens, block_table,
query_start_loc) are meta tensors whose *known values* are kept in `Known` (side table keyed by storage id):
exactly the fold's StepStart facts, here computed from (prompt length, step) instead of read from a log.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

import torch
from torch.utils._python_dispatch import TorchDispatchMode

# --------------------------------------------------------------------------------------------------------------------
# tensor views
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TView:
    sid: int
    offset: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    def to_json(self) -> dict:
        return {"sid": self.sid, "offset": self.offset, "shape": list(self.shape), "strides": list(self.strides), "dtype": self.dtype}


def dtype_name(dt: torch.dtype) -> str:
    return str(dt).replace("torch.", "")


class Storages:
    """Storage identity for meta tensors.  `untyped_storage()._cdata` is the identity; we keep a reference to one
    tensor per storage so the storage is never freed during the step (ids cannot be reused)."""

    def __init__(self) -> None:
        self.keep: dict[int, torch.Tensor] = {}
        self.order: list[int] = []
        self.nbytes: dict[int, int] = {}

    def sid(self, t: torch.Tensor) -> int:
        st = t.untyped_storage()
        k = int(st._cdata)
        if k not in self.keep:
            self.keep[k] = t
            self.order.append(k)
            self.nbytes[k] = int(st.nbytes())
        return k

    def view(self, t: torch.Tensor) -> TView:
        return TView(self.sid(t), int(t.storage_offset()), tuple(int(s) for s in t.shape), tuple(int(s) for s in t.stride()),
                     dtype_name(t.dtype))


def encode(x: Any, S: Storages) -> Any:
    if isinstance(x, torch.Tensor):
        return S.view(x)
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (list, tuple)):
        return [encode(v, S) for v in x]
    if isinstance(x, (torch.dtype, torch.device, torch.memory_format, torch.layout)):
        return str(x)
    if isinstance(x, dict):
        return {str(k): encode(v, S) for k, v in x.items()}
    tn = type(x).__name__
    if tn == "constexpr" and hasattr(x, "value"):
        return encode(x.value, S)
    if tn == "dtype" and type(x).__module__.startswith("triton"):
        return str(x)
    return {"$repr": repr(x)[:200]}


def to_json(x: Any) -> Any:
    if isinstance(x, TView):
        return x.to_json()
    if isinstance(x, list):
        return [to_json(v) for v in x]
    if isinstance(x, dict):
        return {k: to_json(v) for k, v in x.items()}
    return x


# --------------------------------------------------------------------------------------------------------------------
# trace
# --------------------------------------------------------------------------------------------------------------------


@dataclass
class OpCall:
    id: int
    parent: int | None
    op: str
    args: list
    kwargs: dict
    outs: Any = None
    module: str | None = None
    executed: str = "meta"          # meta | rule | reentered | error
    error: str | None = None
    schema: str = ""
    step: int = -1
    phase: str = "forward"

    def to_json(self) -> dict:
        return {"kind": "op", "id": self.id, "parent": self.parent, "op": self.op, "args": to_json(self.args),
                "kwargs": to_json(self.kwargs), "outs": to_json(self.outs), "module": self.module, "executed": self.executed,
                "error": self.error, "schema": self.schema, "step": self.step, "phase": self.phase}


@dataclass
class TritonCall:
    id: int
    parent: int | None
    fn: str
    args: dict
    constexprs: dict
    grid: Any
    num_warps: int | None
    num_stages: int | None
    module: str | None = None
    step: int = -1
    phase: str = "forward"
    src_sha: str = ""

    def to_json(self) -> dict:
        return {"kind": "triton", "id": self.id, "parent": self.parent, "fn": self.fn, "args": to_json(self.args),
                "constexprs": to_json(self.constexprs), "grid": to_json(self.grid), "num_warps": self.num_warps,
                "num_stages": self.num_stages, "module": self.module, "step": self.step, "phase": self.phase}


@dataclass
class Trace:
    events: list = field(default_factory=list)
    storages: Storages = field(default_factory=Storages)
    known: dict[int, list[int]] = field(default_factory=dict)   # sid -> known integer contents (structural inputs)
    known_names: dict[int, str] = field(default_factory=dict)
    params: dict[int, str] = field(default_factory=dict)        # sid -> parameter / buffer name
    open: list[int] = field(default_factory=list)
    module_stack: list[str] = field(default_factory=list)
    next_id: int = 1
    rules_used: dict[str, int] = field(default_factory=dict)
    real_kernels: int = 0
    step: int = -1
    phase: str = "forward"
    steps: list = field(default_factory=list)                   # per-step facts (StepSpec + views), filled by run_step
    notes: list = field(default_factory=list)                   # scaffolding facts worth reporting (KV layout, ...)
    kv_groups: list[list[str]] = field(default_factory=list)    # KV-cache groups (layer names) from vLLM's kv_cache_config
    request_blocks: int = 0                                     # blocks one request can occupy (block-id stride between groups)
    kv_cache_config: Any = None                                 # vLLM's KVCacheConfig (engine core), set by bind_kv_caches
    runner_prep: Any = None                                     # RunnerPrep (vLLM's own per-step attention-input code)
    prep_cuda_kernels: int = 0                                  # real index kernels run by the runner's block-table code (integer data)

    def new_id(self) -> int:
        i = self.next_id
        self.next_id += 1
        return i

    def parent(self) -> int | None:
        return self.open[-1] if self.open else None

    def module(self) -> str | None:
        return self.module_stack[-1] if self.module_stack else None

    def mark_known(self, t: torch.Tensor, values: list[int], name: str) -> None:
        sid = self.storages.sid(t)
        self.known[sid] = [int(v) for v in values]
        self.known_names[sid] = name

    def mark_param(self, t: torch.Tensor, name: str) -> None:
        self.params[self.storages.sid(t)] = name

    def op_hist(self) -> dict[str, int]:
        h: dict[str, int] = {}
        for e in self.events:
            k = e.op if isinstance(e, OpCall) else f"triton:{e.fn}"
            h[k] = h.get(k, 0) + 1
        return dict(sorted(h.items(), key=lambda kv: -kv[1]))

    def dump(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(json.dumps({"events": [e.to_json() for e in self.events],
                                "known": {str(k): {"name": self.known_names[k], "values": v} for k, v in self.known.items()},
                                "params": {str(k): v for k, v in self.params.items()},
                                "storage_nbytes": {str(k): v for k, v in self.storages.nbytes.items()},
                                "rules_used": self.rules_used, "real_kernels": self.real_kernels}))


# --------------------------------------------------------------------------------------------------------------------
# per-op meta rules (shape-only output synthesis for ops without a meta kernel).  Each entry is one hand-written rule;
# none knows anything about a model architecture — they restate the op's schema (which arguments are written).
# --------------------------------------------------------------------------------------------------------------------

MetaRule = Callable[[tuple, dict], Any]


def _rule_mutating_none(args: tuple, kwargs: dict) -> Any:
    return None


def _rule_fa2_varlen_fwd(args: tuple, kwargs: dict) -> Any:
    # (q, k, v, out, cu_seqlens_q, cu_seqlens_k, seqused_k, leftpad_k, block_table, alibi_slopes, max_seqlen_q,
    #  max_seqlen_k, p_dropout, softmax_scale, zero_tensors, is_causal, window_left, window_right, softcap,
    #  return_softmax, num_splits, gen) -> (out, softmax_lse)
    q, out = args[0], args[3]
    if out is None:
        out = torch.empty_like(q)
    total_q, nheads = q.shape[0], q.shape[1]
    lse = torch.empty((nheads, total_q), dtype=torch.float32, device=q.device)
    return [out, lse]


#: ops executed by a meta rule instead of a kernel (name -> rule); the schema's mutated arguments are the outputs
META_RULES: dict[str, MetaRule] = {
    "_C.rms_norm.default": _rule_mutating_none,                 # rms_norm(out!, input, weight, eps)
    "_C.fused_add_rms_norm.default": _rule_mutating_none,       # fused_add_rms_norm(input!, residual!, weight, eps)
    "_C.rotary_embedding.default": _rule_mutating_none,         # rotary_embedding(positions, query!, key!, head_size, cos_sin, is_neox)
    "_C.silu_and_mul.default": _rule_mutating_none,             # silu_and_mul(out!, input)
    "_C.gelu_tanh_and_mul.default": _rule_mutating_none,        # gelu_tanh_and_mul(out!, input)   [rule 10, added for family 2 (GeGLU)]
    "_C_cache_ops.reshape_and_cache_flash.default": _rule_mutating_none,  # reshape_and_cache_flash(k, v, kc!, vc!, slot_mapping, dtype, k_scale, v_scale)
    "_vllm_fa2_C.varlen_fwd.default": _rule_fa2_varlen_fwd,
}

#: vLLM wrappers whose registered Python body is re-entered (children recorded) — same set as the eager observer
#: aten ops vLLM's batch-invariant mode overrides on the CUDA dispatch key (`_batch_invariant_LIB.impl(...)` in
#: `vllm/model_executor/determinism/batch_invariant.py`: mm/addmm/matmul/linear/mean.dim/softmax/_softmax/_log_softmax/bmm).
#: On meta the dispatcher never selects a CUDA-key kernel, so the structural mode re-enters the recorded override itself
#: (its Triton launches are then observed exactly as on the GPU).  Filled by `record_bi_overrides` — nothing is named here.
#: [rule 11, generic dispatcher-parity mechanism; added for family 2 where GemmaRMSNorm's `x.pow(2).mean(-1)` needs it]
BI_OVERRIDES: dict[str, Any] = {}


def record_bi_overrides() -> None:
    """Wrap `torch.library.Library.impl` so every `aten` override vLLM registers is remembered by op name."""
    import torch.library as TL

    if getattr(TL.Library.impl, "_cbb_wrapped", False):
        return
    orig = TL.Library.impl

    def impl(self, op_name, fn, dispatch_key="", **kw):
        if getattr(self, "ns", None) == "aten" and isinstance(op_name, str) and callable(fn):
            n = op_name.replace("::", ".")
            if n.count(".") == 1:
                n += ".default"
            BI_OVERRIDES[n] = fn
        return orig(self, op_name, fn, dispatch_key, **kw)

    impl._cbb_wrapped = True  # type: ignore[attr-defined]
    TL.Library.impl = impl


REENTER = {
    "vllm.unified_attention_with_output.default": ("vllm.model_executor.layers.attention.attention", "unified_attention_with_output"),
    "vllm.unified_kv_cache_update.default": ("vllm.model_executor.layers.attention.attention", "unified_kv_cache_update"),
}


def op_name(func: Any) -> str:
    ns = func.namespace
    name = func._schema.name.split("::")[-1]
    ov = getattr(func, "_overloadname", "") or "default"
    return f"{ns}.{name}.{ov}"


class StructuralMode(TorchDispatchMode):
    def __init__(self, tr: Trace) -> None:
        super().__init__()
        self.tr = tr

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        tr = self.tr
        name = op_name(func)
        eid = tr.new_id()
        S = tr.storages
        try:
            schema = str(func._schema)
        except Exception:  # noqa: BLE001
            schema = ""
        call = OpCall(eid, tr.parent(), name, encode(list(args), S), encode(kwargs, S), module=tr.module(), schema=schema,
                      step=tr.step, phase=tr.phase)
        tr.events.append(call)
        tr.open.append(eid)
        try:
            if name in REENTER:
                import importlib
                mod, attr = REENTER[name]
                impl = getattr(importlib.import_module(mod), attr)
                call.executed = "reentered"
                with StructuralMode(tr):
                    out = impl(*args, **kwargs)
            elif name in META_RULES:
                call.executed = "rule"
                tr.rules_used[name] = tr.rules_used.get(name, 0) + 1
                out = META_RULES[name](args, kwargs)
            elif name in BI_OVERRIDES:
                # a CUDA-key kernel: the eager observer sees the op and its Triton launches only (the mode is not active
                # inside a dispatcher kernel), so the override runs here *without* the mode; the Triton hook still records
                # the launch as this op's child (parity with the record: `aten.mean.dim` → one `mean_kernel` launch).
                call.executed = "reentered-bi"
                tr.rules_used["bi:" + name] = tr.rules_used.get("bi:" + name, 0) + 1
                out = BI_OVERRIDES[name](*args, **kwargs)
            else:
                out = func(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001
            call.executed = "error"
            call.error = f"{type(e).__name__}: {str(e)[:400]}"
            _pop(tr, eid)
            raise
        _pop(tr, eid)
        call.outs = encode(out, S)
        return out


def _pop(tr: Trace, eid: int) -> None:
    if tr.open and tr.open[-1] == eid:
        tr.open.pop()
    elif eid in tr.open:
        tr.open.remove(eid)


# --------------------------------------------------------------------------------------------------------------------
# Triton launch interception (JITFunction.run): record the bound arguments, never launch
# --------------------------------------------------------------------------------------------------------------------

_TRITON: dict[str, Any] = {"orig": None, "tr": None}


def _triton_run(self, *args, grid, warmup=False, **kwargs):
    tr: Trace | None = _TRITON["tr"]
    if tr is None:
        return _TRITON["orig"](self, *args, grid=grid, warmup=warmup, **kwargs)
    if warmup:
        return None
    # bind exactly as Triton 3.7.1 does (verity_capture.triton_adapter, D63); the binder is pure
    options: dict = {}
    try:
        from triton.runtime.driver import driver

        device = driver.active.get_current_device()
        *_, backend, binder = self.device_caches[device]
        bound_args, _specialization, options = binder(*args, **kwargs)
        bound_args = dict(bound_args)
        try:                                     # the compiled kernel's options (defaults filled), as `_pack_args` does
            parsed = backend.parse_options(dict(options) if isinstance(options, dict) else kwargs)
            options = {"num_warps": parsed.num_warps, "num_stages": parsed.num_stages}
        except Exception:  # noqa: BLE001
            options = dict(options) if isinstance(options, dict) else {}
    except Exception as e:  # noqa: BLE001
        names = [p.name for p in self.params]
        bound_args = dict(zip(names, args))
        bound_args.update({k: v for k, v in kwargs.items() if k not in ("num_warps", "num_stages", "num_ctas")})
        options = {"binder_error": repr(e)[:200]}
    constexpr_names = {p.name for p in self.params if getattr(p, "is_constexpr", False)}
    S = tr.storages
    consts = {k: encode(v, S) for k, v in bound_args.items() if k in constexpr_names}
    targs = {k: encode(v, S) for k, v in bound_args.items()}          # eager TritonLaunch.args carries constexprs too
    g = grid(bound_args) if callable(grid) else grid
    nw = options.get("num_warps", kwargs.get("num_warps"))
    ns = options.get("num_stages", kwargs.get("num_stages"))
    ev = TritonCall(tr.new_id(), tr.parent(), f"{self.fn.__module__}.{self.fn.__qualname__}", targs, consts, encode(g, S),
                    None if nw is None else int(nw), None if ns is None else int(ns), tr.module(),
                    step=tr.step, phase=tr.phase, src_sha=_src_sha(self))
    tr.events.append(ev)
    tr.rules_used[f"triton:{ev.fn}"] = tr.rules_used.get(f"triton:{ev.fn}", 0) + 1
    return None


_SRC_SHA: dict[int, str] = {}


def _src_sha(jit_fn: Any) -> str:
    k = id(jit_fn)
    if k not in _SRC_SHA:
        import hashlib
        import inspect

        try:
            src = inspect.getsource(jit_fn.fn)
        except Exception:  # noqa: BLE001
            src = repr(jit_fn)
        _SRC_SHA[k] = hashlib.sha256(src.encode()).hexdigest()
    return _SRC_SHA[k]


@contextlib.contextmanager
def triton_hook(tr: Trace):
    from triton.runtime.jit import JITFunction

    if _TRITON["orig"] is None:
        _TRITON["orig"] = JITFunction.run
        JITFunction.run = _triton_run
    _TRITON["tr"] = tr
    try:
        yield
    finally:
        _TRITON["tr"] = None


# --------------------------------------------------------------------------------------------------------------------
# module names (diagnostic hints only; never enter the Program)
# --------------------------------------------------------------------------------------------------------------------


class ModuleNamer:
    def __init__(self, tr: Trace, model: torch.nn.Module) -> None:
        self.tr = tr
        self.handles = []
        for name, mod in model.named_modules():
            if name == "":
                continue
            self.handles.append(mod.register_forward_pre_hook(partial(self._pre, name)))
            self.handles.append(mod.register_forward_hook(self._post))

    def _pre(self, name: str, mod: Any, args: Any) -> None:
        self.tr.module_stack.append(name)

    def _post(self, mod: Any, args: Any, output: Any) -> None:
        if self.tr.module_stack:
            self.tr.module_stack.pop()

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


# --------------------------------------------------------------------------------------------------------------------
# no-real-kernel guard: any CUDA kernel launch during construction is a violation
# --------------------------------------------------------------------------------------------------------------------


class CudaGuard:
    """Asserts that no CUDA work is issued while active (profiler-based: counts kernel events)."""

    def __enter__(self):
        self.prof = None
        self.by_name, self.prep_kernels, self.model_kernels = {}, 0, -1
        try:
            from torch.profiler import ProfilerActivity, profile

            self.prof = profile(activities=[ProfilerActivity.CUDA])
            self.prof.__enter__()
        except Exception:  # noqa: BLE001
            self.prof = None
        return self

    def __exit__(self, *exc):
        self.kernels = -1
        if self.prof is not None:
            self.prof.__exit__(*exc)
            try:
                evs = [e for e in self.prof.events() if str(getattr(e, "device_type", "")).endswith("CUDA")]
                self.kernels = len(evs)
                h: dict[str, int] = {}
                for e in evs:
                    n = str(getattr(e, "name", "?"))[:80]
                    h[n] = h.get(n, 0) + 1
                self.by_name = dict(sorted(h.items(), key=lambda kv: -kv[1]))
                # the runner's block-table index kernels (integer data) and their copies vs anything else (= model work)
                prep = ("_gather_block_tables_kernel", "_compute_slot_mappings_kernel", "_apply_write_kernel", "_prepare_pos_seq_lens_kernel",
                        "_combine_sampled_and_draft_tokens_kernel", "_prepare_prefill_inputs_kernel", "_post_update_num_computed_tokens_kernel",
                        "FillFunctor<int", "FillFunctor<long", "FillFunctor<bool", "elementwise_kernel_with_index<int",
                        "Memcpy", "Memset", "memcpy", "memset")
                # = the V2 runner's index kernels (v1/worker/gpu/{block_table,input_batch,states}.py) and int/bool buffer fills/copies
                # over request-state integers; anything else (a bf16/fp32 kernel) would be model work
                self.prep_kernels = sum(v for k, v in h.items() if any(p in k for p in prep))
                self.model_kernels = self.kernels - self.prep_kernels
            except Exception:  # noqa: BLE001
                self.kernels = -1
        return False


# --------------------------------------------------------------------------------------------------------------------
# vLLM scaffolding (generic: engine config, distributed init, model instantiation, forward context)
# --------------------------------------------------------------------------------------------------------------------


def apply_env() -> None:
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    os.environ.setdefault("HF_HOME", "/workspace/hf")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")


def build_vllm_config(model_path: str, *, max_model_len: int = 8192, max_num_seqs: int = 8):
    from vllm.engine.arg_utils import EngineArgs

    args = EngineArgs(model=model_path, tokenizer=model_path, dtype="bfloat16", seed=0, enforce_eager=True,
                      max_model_len=max_model_len, gpu_memory_utilization=0.5, trust_remote_code=False,
                      enable_prefix_caching=False, max_num_seqs=max_num_seqs)
    return args.create_engine_config()


def init_distributed(vllm_config) -> None:
    from vllm.config import set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
    from vllm.distributed.parallel_state import get_world_group

    try:
        get_world_group()
        return
    except Exception:  # noqa: BLE001
        pass
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(world_size=1, rank=0, distributed_init_method=f"tcp://127.0.0.1:{port}", local_rank=0,
                                     backend="gloo")
        ensure_model_parallel_initialized(1, 1)


def instantiate_model(vllm_config):
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader.utils import initialize_model

    from vllm.utils.torch_utils import set_default_torch_dtype

    # exactly what the worker does before the loader (`gpu_worker.init_worker_distributed_environment` →
    # `init_batch_invariance`; `DefaultModelLoader.load_model` → `set_default_torch_dtype(model_config.dtype)`)
    from vllm.model_executor.determinism.batch_invariant import init_batch_invariance

    record_bi_overrides()
    init_batch_invariance()
    # `WorkerBase.__init__` (v1/worker/worker_base.py): IR-op provider priority + torch-wrap state for the worker's lifetime.
    # In eager mode `ir_enable_torch_wrap` is False, so `vllm.ir` ops (e.g. GemmaRMSNorm's `ir.ops.rms_norm`) run their
    # Python implementation directly and the observer sees the decomposed aten / Triton ops, exactly as in the eager record.
    # (Found by family 2: without it the mode sees an opaque `vllm_ir.rms_norm` and the fold has no pattern for it.)
    import vllm.ir

    vllm_config.kernel_config.ir_op_priority.set_default()
    vllm.ir.set_default_torch_wrap(vllm_config.compilation_config.ir_enable_torch_wrap)
    with set_current_vllm_config(vllm_config), set_default_torch_dtype(vllm_config.model_config.dtype), torch.device("meta"):
        model = initialize_model(vllm_config=vllm_config, model_config=vllm_config.model_config)
    return model


@dataclass
class StepSpec:
    """Structural inputs of one engine step of ONE request: query rows `[pos0, pos0+M)`, KV length after the step
    `pos0 + M`.  Decode: M=1.  Prefill: pos0=0, M=LP."""
    pos0: int
    M: int
    block_size: int = 16

    @property
    def seq_len(self) -> int:
        return self.pos0 + self.M


def make_forward_inputs(tr: Trace, spec: StepSpec, vllm_config, model):
    """Fake ForwardContext ingredients for one step (scaffolding).  Integer tensors are meta with known values.

    KV-cache groups: vLLM's hybrid KV-cache manager gives layers of different attention kinds (e.g. Gemma-2 sliding-window
    vs global layers) separate groups, each with its OWN block table and slot mapping into the SAME physical tensors (the
    runner's `attn_groups`; the block pool hands every group disjoint block ids).  The scaffold reproduces exactly that:
    group g of `tr.kv_groups` (read from vLLM's `kv_cache_config`, `bind_kv_caches`) gets block ids offset by
    `g * tr.request_blocks`.  One group (Llama): offset 0, identical to the single-table case."""
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    dev = torch.device("meta")
    M, pos0, bs = spec.M, spec.pos0, spec.block_size
    seq_len = spec.seq_len
    nblocks = (seq_len + bs - 1) // bs
    positions_v = list(range(pos0, pos0 + M))
    input_ids = torch.empty((M,), dtype=torch.int64, device=dev)
    positions = torch.empty((M,), dtype=torch.int64, device=dev)
    query_start_loc = torch.empty((2,), dtype=torch.int32, device=dev)
    seq_lens = torch.empty((1,), dtype=torch.int32, device=dev)
    logits_indices = torch.empty((1,), dtype=torch.int64, device=dev)
    tr.mark_known(input_ids, [-1] * M, "input_ids")        # token VALUES are circuit values; the builder knows only positions
    tr.mark_known(positions, positions_v, "positions")
    tr.mark_known(query_start_loc, [0, M], "query_start_loc")
    tr.mark_known(seq_lens, [seq_len], "seq_lens")
    tr.mark_known(logits_indices, [M - 1], "logits_indices")

    layers = vllm_config.compilation_config.static_forward_context
    attn_layers = {n: l for n, l in layers.items() if hasattr(l, "kv_cache")}
    groups = tr.kv_groups or [list(attn_layers)]
    stride = tr.request_blocks or nblocks
    attn_metadata, slot_by_layer, groups_facts = {}, {}, []
    for g, names in enumerate(groups):
        block_v = [g * stride + i for i in range(nblocks)]
        slot_v = [g * stride * bs + p for p in positions_v]          # block * bs + offset, one request, blocks in order
        slot_mapping = torch.empty((M,), dtype=torch.int64, device=dev)
        block_table = torch.empty((1, nblocks), dtype=torch.int32, device=dev)
        tr.mark_known(slot_mapping, slot_v, f"slot_mapping[g{g}]")
        tr.mark_known(block_table, block_v, f"block_table[g{g}]")
        md = FlashAttentionMetadata(num_actual_tokens=M, max_query_len=M, query_start_loc=query_start_loc, max_seq_len=seq_len,
                                    seq_lens=seq_lens, block_table=block_table, slot_mapping=slot_mapping, use_cascade=False,
                                    common_prefix_len=0, cu_prefix_query_lens=None, prefix_kv_lens=None, suffix_kv_lens=None,
                                    max_num_splits=1, causal=True)
        for n in names:
            attn_metadata[n] = md
            slot_by_layer[n] = slot_mapping
        groups_facts.append({"layers": list(names), "block_size": bs, "block_tables": {"r0": block_v}, "slot_mapping": slot_v})
    missing = [n for n in attn_layers if n not in attn_metadata]
    if missing:
        raise RuntimeError(f"attention layers without a KV-cache group: {missing[:4]}")
    facts = {"spec": spec, "input_ids": tr.storages.view(input_ids), "positions": tr.storages.view(positions),
             "positions_v": positions_v, "slot_v": groups_facts[0]["slot_mapping"], "block_table_v": groups_facts[0]["block_tables"]["r0"],
             "seq_len": seq_len, "logits_indices_v": [M - 1], "kv_groups": groups_facts if len(groups) > 1 else None}
    return input_ids, positions, logits_indices, attn_metadata, slot_by_layer, attn_layers, facts


def bind_kv_caches(tr: Trace, attn_layers: dict, vllm_config, num_blocks: int, *, step_inputs: str = "runner") -> dict[str, torch.Tensor]:
    """The runner's own KV-cache allocation, on `meta`: `get_kv_cache_spec` (layers → specs) → `get_kv_cache_configs`
    (engine core; `available_memory` sized for `num_blocks`) → `resolve_kv_cache_layout` (the backends' preferred
    layout, e.g. FA2 → LBNHC) → `allocate_kv_cache` (one int8 backing buffer, per-layer `[B, H, N, C]` views).  No
    layout fact is hand-authored here.  Idempotent: the buffers persist across the steps of one request (KV state)."""
    from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts, resolve_kv_cache_layout
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
    from vllm.v1.worker.gpu.attn_utils import get_kv_cache_spec
    from vllm.v1.worker.utils import allocate_kv_cache

    first = next(iter(attn_layers.values()))
    kc0 = getattr(first, "kv_cache", None)
    if isinstance(kc0, torch.Tensor) and kc0.device.type == "meta" and kc0.dim() == 4 and tr.kv_groups:
        return {n: l.kv_cache for n, l in attn_layers.items()}
    specs = get_kv_cache_spec(vllm_config)
    page = sum(sp.page_size_bytes for sp in specs.values())
    # the engine core insists the cache holds one max_model_len request; on meta the size is free.  Several KV-cache
    # groups each take a disjoint block range of one request (`make_forward_inputs`), so allow that many.
    bs = vllm_config.cache_config.block_size
    num_blocks = max(num_blocks, -(-vllm_config.model_config.max_model_len // bs) + 1, (len(specs) + 1) * max(num_blocks, 1))
    with torch.device("meta"):
        layouts = get_supported_kv_cache_layouts({l.get_attn_backend() for l in attn_layers.values()})
        layout = resolve_kv_cache_layout(vllm_config, [[l.name for l in layouts]], specs.values())   # engine core order
        cfg = get_kv_cache_configs(vllm_config, [specs], [num_blocks * page])[0]
    tr.kv_cache_config = cfg
    if step_inputs in ("runner", "runner-attn"):
        # the runner's order (`initialize_kv_cache`): attention groups + kernel block sizes, then `init_kv_cache`
        # (= allocate_kv_cache + bind_kv_cache into the forward context) — nothing bound by hand
        from vllm.v1.worker.gpu.attn_utils import init_kv_cache

        tr.runner_prep = (RunnerFull if step_inputs == "runner" else RunnerPrep)(tr, vllm_config, cfg)
        runner_kv_caches: list = []
        with torch.device("meta"):
            kv_caches = init_kv_cache(runner_kv_caches, vllm_config.compilation_config.static_forward_context, cfg, torch.device("meta"),
                                      tr.runner_prep.kernel_block_sizes, vllm_config)
        tr.kv_groups = [[str(n) for n in g.layer_names] for g in cfg.kv_cache_groups]
        tr.notes.append({"kv_cache": {"num_blocks": cfg.num_blocks, "layout": layout.name, "page_bytes": page, "bound_by": "runner init_kv_cache",
                                      "groups": [{"layers": len(g), "first": g[0] if g else None} for g in tr.kv_groups]}})
        for name in kv_caches:
            kc = attn_layers[name].kv_cache
            kc = kc[0] if isinstance(kc, list) else kc
            tr.mark_param(kc, f"kv_cache:{name}")
        return kv_caches
    with torch.device("meta"):
        kv_caches = allocate_kv_cache(cfg, torch.device("meta"), layout)
    tr.kv_groups = [[str(n) for n in g.layer_names] for g in cfg.kv_cache_groups]
    tr.notes.append({"kv_cache": {"num_blocks": cfg.num_blocks, "layout": layout.name, "page_bytes": page,
                                  "backing_bytes": cfg.kv_cache_tensors[0].size if cfg.kv_cache_tensors else 0,
                                  "groups": [{"layers": len(g), "first": g[0] if g else None} for g in tr.kv_groups],
                                  "tensors_shared_by": [len(getattr(t, "shared_by", []) or []) for t in cfg.kv_cache_tensors[:3]]}})
    for name, kc in kv_caches.items():
        attn_layers[name].kv_cache = kc                  # d9105ea80: `Attention.kv_cache` is one tensor (bound by the runner)
        tr.mark_param(kc, f"kv_cache:{name}")
    return kv_caches


class RunnerPrep:
    """The runner's OWN per-step attention-input construction, replacing the hand-built `FlashAttentionMetadata` /
    block table / slot mapping of `make_forward_inputs` (the scaffold surface that produced a wrong Gemma-2 Program).

    vLLM code executed (v1/worker/gpu): `attn_utils.init_attn_backend` (attention groups per KV-cache group, each
    backend's metadata builder, kernel block sizes), `block_table.BlockTables` (`append_block_ids` → staged writes →
    `gather_block_tables` / `compute_slot_mappings`: the runner's Triton index kernels), `attn_utils.build_attn_metadata`
    (`CommonAttentionMetadata` → `FlashAttentionMetadataBuilder.build`), `attn_utils.build_slot_mappings_by_layer`.

    Devices: `BlockTables` needs a real CUDA device (UVA pinned buffers + Triton kernels over INTEGER index data — the
    scheduler's block ids and the request's positions; no model value exists anywhere).  Its two kernel launches per step
    run for real and are counted separately (`tr.prep_cuda_kernels`); the model forward stays on meta with 0 kernels.
    Their results are copied to meta tensors with the known values (`mark_known`) that the fold's StepStart needs.

    Still hand-supplied (scheduler / request-state facts, see `step_inputs`): the request's block ids (the KV manager's
    allocation: `g * request_blocks + i` per group), `idx_mapping=[0]`, `query_start_loc=[0, M]`, `seq_lens=[pos0+M]`,
    `positions=arange(pos0, pos0+M)`, `logits_indices=[M-1]`, `input_ids` (unknown values).  The runner computes these
    from `RequestState`/`InputBuffers` in `prepare_inputs` (Triton kernels over its persistent request table driven by
    a `SchedulerOutput`); reproducing that needs a runner object (`GPUModelRunner.__init__` + `load_model` state)."""

    def __init__(self, tr: Trace, vllm_config, kv_cache_config, *, prep_device: str = "cuda") -> None:
        from vllm.v1.worker.gpu.attn_utils import init_attn_backend
        from vllm.v1.worker.gpu.block_table import BlockTables

        self.tr, self.cfg, self.kvc = tr, vllm_config, kv_cache_config
        self.dev = torch.device(prep_device)
        # builders on meta (their metadata tensors are what the meta forward consumes); fall back to the prep device
        try:
            with torch.device("meta"):
                self.attn_groups, cg, self.kernel_block_sizes = init_attn_backend(kv_cache_config, vllm_config, torch.device("meta"))
            self.builder_device = "meta"
        except Exception as e:  # noqa: BLE001
            tr.notes.append({"runner_prep": {"init_attn_backend_on_meta_refused": f"{type(e).__name__}: {str(e)[:300]}"}})
            self.attn_groups, cg, self.kernel_block_sizes = init_attn_backend(kv_cache_config, vllm_config, self.dev)
            self.builder_device = prep_device
        sc = vllm_config.scheduler_config
        block_sizes = [g.kv_cache_spec.block_size for g in kv_cache_config.kv_cache_groups]
        max_blocks = [g.kv_cache_spec.max_num_blocks_per_req(vllm_config, vllm_config.model_config.max_model_len)
                      for g in kv_cache_config.kv_cache_groups]
        self.block_tables = BlockTables(block_sizes=block_sizes, max_num_reqs=sc.max_num_seqs, max_num_batched_tokens=sc.max_num_batched_tokens,
                                        max_num_blocks_per_group=max_blocks, device=self.dev, kernel_block_sizes=self.kernel_block_sizes)
        tr.notes.append({"runner_prep": {"attn_groups": [[{"backend": g.backend.__name__ if hasattr(g, "backend") else str(type(g)),
                                                            "layers": len(g.layer_names)} for g in grp] for grp in self.attn_groups],
                                         "kernel_block_sizes": list(self.kernel_block_sizes), "block_sizes": block_sizes,
                                         "block_table_width": max_blocks, "builders_on": self.builder_device, "block_tables_on": prep_device}})

    def step_inputs(self, spec: StepSpec, block_ids: list[list[int]]):
        """Returns (attn_metadata by layer, slot_mapping by layer, known-value facts) for one step of one request."""
        from vllm.v1.worker.gpu.attn_utils import build_attn_metadata, build_slot_mappings_by_layer

        tr, M, pos0 = self.tr, spec.M, spec.pos0
        seq_len = spec.seq_len
        positions_v = list(range(pos0, pos0 + M))
        dev, meta = self.dev, torch.device("meta")
        # --- scheduler / request-state facts (hand-supplied, see class doc) ---
        idx_mapping = torch.tensor([0], dtype=torch.int32, device=dev)
        qsl_d = torch.tensor([0, M], dtype=torch.int32, device=dev)
        pos_d = torch.tensor(positions_v, dtype=torch.int64, device=dev)
        # --- the runner's block-table / slot-mapping path (real index kernels, integer data only) ---
        _TRITON["tr"], saved = None, _TRITON["tr"]          # the runner's index kernels must really run (not be recorded)
        try:
            self.block_tables.append_block_ids(0, tuple(block_ids), overwrite=True)
            self.block_tables.apply_staged_writes()
            bts = self.block_tables.gather_block_tables(idx_mapping, num_reqs_padded=1)
            sms = self.block_tables.compute_slot_mappings(idx_mapping, qsl_d, pos_d, num_tokens_padded=M)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            bts_v = [bt[0].tolist() for bt in bts]
            sms_v = [sm.tolist() for sm in sms]
        finally:
            _TRITON["tr"] = saved
        tr.prep_cuda_kernels += 2                            # gather_block_tables + compute_slot_mappings (see CudaGuard.by_name)
        # --- meta twins with known values (what the meta forward and the fold consume) ---
        query_start_loc = torch.empty((2,), dtype=torch.int32, device=meta)
        seq_lens = torch.empty((1,), dtype=torch.int32, device=meta)
        positions = torch.empty((M,), dtype=torch.int64, device=meta)
        tr.mark_known(query_start_loc, [0, M], "query_start_loc")
        tr.mark_known(seq_lens, [seq_len], "seq_lens")
        tr.mark_known(positions, positions_v, "positions")
        block_tables_m, slot_rows = [], []
        for gi, (bt_v, sm_v) in enumerate(zip(bts_v, sms_v)):
            bt = torch.empty((1, len(bt_v)), dtype=torch.int32, device=meta)
            tr.mark_known(bt, bt_v, f"block_table[g{gi}] (runner gather_block_tables)")
            block_tables_m.append(bt)
            slot_rows.append(sm_v)
        slot_mappings_m = torch.empty((len(sms_v), M), dtype=torch.int64, device=meta)
        tr.mark_known(slot_mappings_m, [v for row in slot_rows for v in row], "slot_mappings (runner compute_slot_mappings)")
        # --- the runner's attention-metadata construction (each backend's own builder) ---
        attn_metadata = build_attn_metadata(self.attn_groups, num_reqs=1, num_tokens=M, query_start_loc_gpu=query_start_loc,
                                            query_start_loc_cpu=torch.tensor([0, M], dtype=torch.int32), max_query_len=M,
                                            seq_lens=seq_lens, max_seq_len=seq_len, block_tables=block_tables_m,
                                            slot_mappings=slot_mappings_m, kv_cache_config=self.kvc,
                                            seq_lens_cpu_upper_bound=torch.tensor([seq_len], dtype=torch.int32), positions=positions)
        slot_by_layer = build_slot_mappings_by_layer(slot_mappings_m, self.kvc)
        nb = (seq_len + spec.block_size - 1) // spec.block_size
        groups_facts = [{"layers": [str(n) for n in grp.layer_names], "block_size": spec.block_size,
                         "block_tables": {"r0": bts_v[gi][:nb]}, "slot_mapping": sms_v[gi]} for gi, grp in enumerate(self.kvc.kv_cache_groups)]
        facts = {"positions": positions, "positions_v": positions_v, "seq_len": seq_len, "groups": groups_facts,
                 "attn_metadata_types": sorted({type(m).__name__ for m in attn_metadata.values()})}
        return attn_metadata, slot_by_layer, facts


class RunnerFull(RunnerPrep):
    """`RunnerPrep` + the runner's OWN request-state path for the remaining per-step inputs: a real `GPUModelRunner`
    (constructed on the CUDA device WITHOUT `load_model`: no weights, no KV allocation on the device) driven by a
    `SchedulerOutput` per engine step exactly as the engine core drives it — `add_requests` / `update_requests` →
    `gather_batch_req_state` → `prepare_inputs` (positions, query_start_loc, seq_lens, logits_indices, input_ids slots via
    its Triton index kernels) → `prepare_attn` (block tables, slot mappings) → after the step `postprocess_num_computed_tokens`.
    Attention metadata is then built by `build_attn_metadata` as in `RunnerPrep` (the runner's `execute_model` does the same).

    What `load_model` would have set is stubbed with bookkeeping no-ops (`model_state`, `sampler`, `prompt_logprobs_worker`
    — token-value bookkeeping; `decode_query_len = 1`, no speculative decoding).  Hand-supplied now: only the scheduler's
    decisions (block ids per KV-cache group, `num_scheduled_tokens`, `num_computed_tokens`) and placeholder token values."""

    class _Stub:
        num_new_sampled_tokens_per_step = 1

        def __getattr__(self, k):
            return lambda *x, **kw: None

    def __init__(self, tr: Trace, vllm_config, kv_cache_config, *, prep_device: str = "cuda") -> None:
        super().__init__(tr, vllm_config, kv_cache_config, prep_device=prep_device)
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner

        self.runner = GPUModelRunner(vllm_config, self.dev)
        self.stubbed = []
        for k, v in {"model_state": self._Stub(), "decode_query_len": 1, "sampler": self._Stub(), "prompt_logprobs_worker": self._Stub()}.items():
            if getattr(self.runner, k, None) is None:
                setattr(self.runner, k, v)
                self.stubbed.append(k)
        self.runner.block_tables, self.runner.kv_cache_config, self.runner.attn_groups = self.block_tables, kv_cache_config, self.attn_groups
        self.last_batch = None
        tr.notes.append({"runner_full": {"GPUModelRunner": "constructed on cuda without load_model", "stubbed": list(self.stubbed)}})

    def step_inputs(self, spec: StepSpec, block_ids: list[list[int]]):
        from vllm import SamplingParams
        from vllm.config import CUDAGraphMode
        from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
        from vllm.v1.worker.gpu.attn_utils import build_attn_metadata, build_slot_mappings_by_layer
        from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

        tr, r, M, pos0 = self.tr, self.runner, spec.M, spec.pos0
        G = len(block_ids)
        _TRITON["tr"], saved = None, _TRITON["tr"]
        try:
            if self.last_batch is not None:
                r.postprocess_num_computed_tokens(self.last_batch)          # the runner's post-step bookkeeping
            if pos0 == 0:
                toks = [0] * M                                                # placeholder token VALUES (circuit values)
                so = SchedulerOutput(scheduled_new_reqs=[NewRequestData(req_id="r0", prompt_token_ids=toks, mm_features=[],
                                                                        sampling_params=SamplingParams(max_tokens=1 << 20, temperature=0.0),
                                                                        pooling_params=None, block_ids=tuple(block_ids), num_computed_tokens=0,
                                                                        lora_request=None, prefill_token_ids=toks)],
                                     scheduled_cached_reqs=CachedRequestData.make_empty(), num_scheduled_tokens={"r0": M},
                                     total_num_scheduled_tokens=M, scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={},
                                     num_common_prefix_blocks=[0] * G, finished_req_ids=set(), free_encoder_mm_hashes=[])
                r.add_requests(so)
            else:
                prev = self.prev_block_ids
                new_ids = [[b for b in block_ids[g] if b not in prev[g]] for g in range(G)]
                kw = dict(req_ids=["r0"], resumed_req_ids=set(), new_token_ids=[[]], all_token_ids={},
                          new_block_ids=[tuple(new_ids) if any(new_ids) else None], num_computed_tokens=[pos0])
                if "num_output_tokens" in CachedRequestData.__dataclass_fields__:
                    kw["num_output_tokens"] = [pos0 - self.prompt_len + 1]
                so = SchedulerOutput(scheduled_new_reqs=[], scheduled_cached_reqs=CachedRequestData(**kw), num_scheduled_tokens={"r0": M},
                                     total_num_scheduled_tokens=M, scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={},
                                     num_common_prefix_blocks=[0] * G, finished_req_ids=set(), free_encoder_mm_hashes=[])
                r.update_requests(so)
            if pos0 == 0:
                self.prompt_len = M
            self.prev_block_ids = block_ids
            r.block_tables.apply_staged_writes()
            brs, _ = r.gather_batch_req_state(so, False)
            ib = r.prepare_inputs(so, brs, BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, num_tokens=M, num_reqs=None))
            bts, sms = r.prepare_attn(ib)
            if self.dev.type == "cuda":
                torch.cuda.synchronize(self.dev)
            positions_v = ib.positions[:M].tolist()
            qsl_v = ib.query_start_loc.tolist()
            seq_lens_v = ib.seq_lens.tolist()
            logits_v = ib.logits_indices.tolist()
            bts_v = [bt[0].tolist() for bt in bts]
            sms_v = [sm[:M].tolist() for sm in sms]
            self.last_batch = ib
        finally:
            _TRITON["tr"] = saved
        tr.prep_cuda_kernels += 1
        seq_len = seq_lens_v[0]
        if positions_v != list(range(pos0, pos0 + M)) or seq_len != spec.seq_len or qsl_v[:2] != [0, M]:
            raise RuntimeError(f"runner prepare_inputs disagrees with the step spec: positions {positions_v[:3]}.. seq_lens {seq_lens_v} qsl {qsl_v}")
        meta = torch.device("meta")
        query_start_loc = torch.empty((len(qsl_v),), dtype=torch.int32, device=meta)
        seq_lens = torch.empty((1,), dtype=torch.int32, device=meta)
        positions = torch.empty((M,), dtype=torch.int64, device=meta)
        logits_indices = torch.empty((len(logits_v),), dtype=torch.int64, device=meta)
        tr.mark_known(query_start_loc, qsl_v, "query_start_loc (runner prepare_inputs)")
        tr.mark_known(seq_lens, seq_lens_v, "seq_lens (runner prepare_inputs)")
        tr.mark_known(positions, positions_v, "positions (runner prepare_inputs)")
        tr.mark_known(logits_indices, logits_v, "logits_indices (runner prepare_inputs)")
        block_tables_m = []
        for gi, bt_v in enumerate(bts_v):
            bt = torch.empty((1, len(bt_v)), dtype=torch.int32, device=meta)
            tr.mark_known(bt, bt_v, f"block_table[g{gi}] (runner prepare_attn)")
            block_tables_m.append(bt)
        slot_mappings_m = torch.empty((len(sms_v), M), dtype=torch.int64, device=meta)
        tr.mark_known(slot_mappings_m, [v for row in sms_v for v in row], "slot_mappings (runner prepare_attn)")
        attn_metadata = build_attn_metadata(self.attn_groups, num_reqs=1, num_tokens=M, query_start_loc_gpu=query_start_loc,
                                            query_start_loc_cpu=torch.tensor(qsl_v, dtype=torch.int32), max_query_len=M,
                                            seq_lens=seq_lens, max_seq_len=seq_len, block_tables=block_tables_m,
                                            slot_mappings=slot_mappings_m, kv_cache_config=self.kvc,
                                            seq_lens_cpu_upper_bound=torch.tensor(seq_lens_v, dtype=torch.int32), positions=positions)
        slot_by_layer = build_slot_mappings_by_layer(slot_mappings_m, self.kvc)
        nb = (seq_len + spec.block_size - 1) // spec.block_size
        groups_facts = [{"layers": [str(n) for n in grp.layer_names], "block_size": spec.block_size,
                         "block_tables": {"r0": bts_v[gi][:nb]}, "slot_mapping": sms_v[gi]} for gi, grp in enumerate(self.kvc.kv_cache_groups)]
        facts = {"positions": positions, "positions_v": positions_v, "seq_len": seq_len, "groups": groups_facts, "logits_indices": logits_indices,
                 "logits_indices_v": logits_v, "attn_metadata_types": sorted({type(m).__name__ for m in attn_metadata.values()})}
        return attn_metadata, slot_by_layer, facts


def make_forward_inputs_runner(tr: Trace, spec: StepSpec, vllm_config, model, prep: "RunnerPrep"):
    """`make_forward_inputs` with the attention side produced by vLLM's runner code (`RunnerPrep.step_inputs`)."""
    dev = torch.device("meta")
    M, bs = spec.M, spec.block_size
    nblocks = (spec.seq_len + bs - 1) // bs
    stride = tr.request_blocks or nblocks
    block_ids = [[g * stride + i for i in range(nblocks)] for g in range(len(prep.kvc.kv_cache_groups))]   # KV-manager stand-in
    attn_metadata, slot_by_layer, f = prep.step_inputs(spec, block_ids)
    if "logits_indices" in f:
        input_ids = torch.empty((M,), dtype=torch.int64, device=dev)
        tr.mark_known(input_ids, [-1] * M, "input_ids")
        layers = vllm_config.compilation_config.static_forward_context
        attn_layers = {n: l for n, l in layers.items() if hasattr(l, "kv_cache")}
        gf = f["groups"]
        facts = {"spec": spec, "input_ids": tr.storages.view(input_ids), "positions": tr.storages.view(f["positions"]),
                 "positions_v": f["positions_v"], "slot_v": gf[0]["slot_mapping"], "block_table_v": gf[0]["block_tables"]["r0"],
                 "seq_len": f["seq_len"], "logits_indices_v": f["logits_indices_v"], "kv_groups": gf if len(gf) > 1 else None,
                 "attn_metadata_types": f["attn_metadata_types"], "step_inputs": "runner"}
        return input_ids, f["positions"], f["logits_indices"], attn_metadata, slot_by_layer, attn_layers, facts
    input_ids = torch.empty((M,), dtype=torch.int64, device=dev)
    logits_indices = torch.empty((1,), dtype=torch.int64, device=dev)
    tr.mark_known(input_ids, [-1] * M, "input_ids")
    tr.mark_known(logits_indices, [M - 1], "logits_indices")
    layers = vllm_config.compilation_config.static_forward_context
    attn_layers = {n: l for n, l in layers.items() if hasattr(l, "kv_cache")}
    missing = [n for n in attn_layers if n not in attn_metadata]
    if missing:
        raise RuntimeError(f"attention layers without runner attention metadata: {missing[:4]}")
    gf = f["groups"]
    facts = {"spec": spec, "input_ids": tr.storages.view(input_ids), "positions": tr.storages.view(f["positions"]),
             "positions_v": f["positions_v"], "slot_v": gf[0]["slot_mapping"], "block_table_v": gf[0]["block_tables"]["r0"],
             "seq_len": f["seq_len"], "logits_indices_v": [M - 1], "kv_groups": gf if len(gf) > 1 else None,
             "attn_metadata_types": f["attn_metadata_types"], "step_inputs": "runner"}
    return input_ids, f["positions"], logits_indices, attn_metadata, slot_by_layer, attn_layers, facts


def run_step(tr: Trace, model, vllm_config, spec: StepSpec, *, kv_num_blocks: int = 64, sampler: str = "vllm", step_inputs: str = "runner"):
    """One engine step of one request under the structural mode: `model.forward` → `hidden[logits_indices]` →
    `compute_logits` → sampler (the V2 runner's sequence, `gpu/model_runner.py:861-867`).  Appends the step's facts to
    `tr.steps`; returns (hidden, logits, sampled) meta tensors."""
    from vllm.forward_context import set_forward_context

    tr.step += 1
    if tr.step == 0:
        for name, p in model.named_parameters():
            tr.mark_param(p, name)
        for name, b in model.named_buffers():
            tr.mark_param(b, name)
    layers0 = vllm_config.compilation_config.static_forward_context
    bind_kv_caches(tr, {n: l for n, l in layers0.items() if hasattr(l, "kv_cache")}, vllm_config, kv_num_blocks, step_inputs=step_inputs)
    if step_inputs in ("runner", "runner-attn"):
        input_ids, positions, logits_indices, attn_md, slot_by_layer, attn_layers, facts = make_forward_inputs_runner(tr, spec, vllm_config, model,
                                                                                                                      tr.runner_prep)
    else:
        input_ids, positions, logits_indices, attn_md, slot_by_layer, attn_layers, facts = make_forward_inputs(tr, spec, vllm_config, model)
    namer = ModuleNamer(tr, model)
    try:
        with triton_hook(tr), StructuralMode(tr), set_forward_context(attn_md, vllm_config, num_tokens=spec.M,
                                                                       slot_mapping=slot_by_layer):
            tr.phase = "forward"
            hidden = model(input_ids, positions, None, None)
            sample_hidden = hidden[logits_indices]
            logits = model.compute_logits(sample_hidden)
            tr.phase = "sample"
            sampled = greedy_sample(tr, logits, positions_last=spec.pos0 + spec.M - 1, kind=sampler)
    finally:
        namer.remove()
        tr.phase = "forward"
    facts["logits"] = tr.storages.view(logits)
    facts["sampled"] = tr.storages.view(sampled) if sampled is not None else None
    tr.steps.append(facts)
    return hidden, logits, sampled


def greedy_sample(tr: Trace, logits: torch.Tensor, *, positions_last: int, kind: str = "vllm") -> torch.Tensor:
    """Sampling kind = greedy (structural input).  `kind="vllm"`: vLLM's own V2 sampler `gumbel_sample`
    (`gpu/sample/gumbel.py`) with `apply_temperature=False`, `is_drafting=False` — its Triton `_gumbel_sample_kernel`
    is intercepted, then `aten.argmax` + `aten.gather` run on meta.  The temperature/seed buffers are the public
    sampling parameters (greedy: temperature 0), never read.  `kind="argmax"`: plain `aten.argmax`."""
    if kind == "vllm":
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

        R = logits.shape[0]
        dev = logits.device
        expanded_idx_mapping = torch.empty((R,), dtype=torch.int64, device=dev)
        temperature = torch.empty((R,), dtype=torch.float32, device=dev)
        seed = torch.empty((R,), dtype=torch.int64, device=dev)
        pos = torch.empty((R,), dtype=torch.int64, device=dev)
        tr.mark_known(expanded_idx_mapping, list(range(R)), "expanded_idx_mapping")
        tr.mark_known(temperature, [0] * R, "temperature (greedy)")
        tr.mark_known(seed, [0] * R, "seed (unused under greedy)")
        tr.mark_known(pos, [positions_last] * R, "sample_pos")
        return gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=False, is_drafting=False)
    return torch.argmax(logits, dim=-1)


def run_request(tr: Trace, model, vllm_config, *, prompt_len: int, max_tokens: int, block_size: int = 16, sampler: str = "vllm",
                step_inputs: str = "runner"):
    """One request's engine steps: prefill of `prompt_len` rows, then `max_tokens - 1` decode steps (vLLM serves
    `max_tokens` samples: the prefill's plus one per decode step).  Structural inputs only."""
    total = prompt_len + max_tokens
    nblocks = (total + block_size - 1) // block_size + 1
    tr.request_blocks = nblocks
    specs = [StepSpec(pos0=0, M=prompt_len, block_size=block_size)]
    specs += [StepSpec(pos0=prompt_len + t, M=1, block_size=block_size) for t in range(max_tokens - 1)]
    outs = []
    for sp in specs:
        outs.append(run_step(tr, model, vllm_config, sp, kv_num_blocks=nblocks, sampler=sampler, step_inputs=step_inputs))
    return specs, outs
