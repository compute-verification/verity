"""A vLLM general plugin that records one replayed step's interior, per TP rank, in a compiled (``-O2``, CUDA graph) engine.

    VERITY_SPOT_CAPTURE=DIR VERITY_SPOT_STEPS=s[,s2] [VERITY_SPOT_MLA_TOKENS=4] <a vLLM engine with this plugin registered>

Registered as a ``vllm.general_plugins`` entry point (`site` writes the dist-info), so it loads in every worker. It runs on
the V2 model runner, vLLM's default at the integration's pin. A step is counted when a worker's ``execute_model`` has
scheduled tokens and is not a dummy run. At step s (0-based), when the step dispatches to a CUDA graph (FULL or PIECEWISE),
the runner's graph call is preceded by one more forward of the same model on the same input buffers, attention metadata,
slot mappings and padded size, with the forward context's graph mode set to NONE. That forward runs the same compiled
Inductor code with its Python wrapper live, and is the one recorded:

* every op the compiled pieces dispatch (`Recorder`, a TorchDispatchMode entered inside `PiecewiseBackend.__call__`): the
  custom ops (group quant, FP8 GEMMs, allreduce, the fused MoE op, the DSA indexer) and aten's extern ops, with each tensor
  argument before the call and each return and written argument after it. A parameter is named, not copied. A tensor above
  ``BIG`` bytes (a KV cache) is recorded by shape only. Values that pass only between two Inductor Triton kernels are not
  recorded;
* inside the opaque ops: each Triton fused-MoE launch (``invoke_fused_moe_triton_kernel``: A, its scales, the expert ids,
  the sorted token ids, the routed weights, the config, and C after the launch), and each sparse MLA kernel call
  (``_bf16_flash_mla_kernel``: q, the global top-k rows, out and lse, and the KV rows those indices read for the first
  ``VERITY_SPOT_MLA_TOKENS`` tokens).

Then the graph call runs as served, and its hidden states must equal the recorded forward's bit for bit (``check``). The
extra forward writes the same KV values to the same slots, so the step and every later one are unchanged.

Each recorded tensor's bytes are written once under ``DIR/rank<r>/blobs/<sha512>``. ``DIR/rank<r>/step<s>/ops.jsonl`` lists
the events in order. ``step<s>/meta.json`` holds the step (tokens, padding, graph mode, the scheduled requests and their
token counts), the check, and the SHA-512 of ``ops.jsonl``: the rank's interior root for the step. Every recorded event
names its blobs by SHA-512, so the root binds every recorded byte. ``params.json`` gives each parameter that a recorded op read:
its shape, dtype and SHA-512 over its bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

BIG = 1 << 30
#: aten ops that only allocate, view or restride: their outputs alias or are uninitialized, never a value of the step
SKIP = {"empty", "empty_strided", "empty_like", "as_strided", "view", "_unsafe_view", "reshape", "alias", "detach", "t",
        "permute", "transpose", "expand", "unsqueeze", "squeeze", "select", "slice", "_reinterpret_tensor", "lift_fresh",
        "split", "split_with_sizes", "unbind", "chunk", "narrow", "view_as", "resize_", "set_", "record_stream"}


class _State:
    def __init__(self) -> None:
        self.dir = Path(os.environ["VERITY_SPOT_CAPTURE"])
        self.targets = {int(s) for s in os.environ.get("VERITY_SPOT_STEPS", "").split(",") if s.strip()}
        self.mla_tokens = int(os.environ.get("VERITY_SPOT_MLA_TOKENS", "4"))
        self.step = -1
        self.active = False
        self.recording = False
        self.runner: Any = None
        self.sched: Any = None
        self.input_batch: Any = None
        self.attn: Any = None
        self.slot_maps: Any = None
        self.params: dict[tuple, str] = {}
        self.param_of: dict[str, Any] = {}
        self.used_params: set[str] = set()
        self.seen: set[str] = set()
        self.events: list[dict] = []
        self.piece = -1
        self.rank = 0
        self.errors: list[str] = []
        self.bytes = 0

    def root(self) -> Path:
        return self.dir / f"rank{self.rank}"


S: _State | None = None


def register() -> None:
    global S
    if not os.environ.get("VERITY_SPOT_CAPTURE") or S is not None:
        return
    S = _State()
    try:
        _install()
    except Exception:  # noqa: BLE001 -- a plugin that fails to install must say so and leave the engine as served
        print("[spot-capture] install failed:\n" + traceback.format_exc(), file=sys.stderr, flush=True)


def _wrap(owner: Any, name: str, make) -> None:
    orig = getattr(owner, name)
    setattr(owner, name, make(orig))


def _install() -> None:
    from vllm.compilation import piecewise_backend as pb
    from vllm.v1.worker.gpu import cudagraph_utils as cgu
    from vllm.v1.worker.gpu import model_runner as mr

    def execute_model(orig):
        def run(self, scheduler_output, *a, **kw):
            dummy = kw.get("dummy_run", a[1] if len(a) > 1 else False)
            real = (not dummy) and getattr(scheduler_output, "total_num_scheduled_tokens", 0) > 0
            if real:
                S.step += 1
                S.active = S.step in S.targets
                S.runner, S.sched = self, scheduler_output
                if S.active:
                    _bind_model_state(self)
            try:
                return orig(self, scheduler_output, *a, **kw)
            finally:
                if S.active and S.events:
                    _finish(self)
                S.active = False
        return run

    def prepare_inputs(orig):
        def run(self, *a, **kw):
            out = orig(self, *a, **kw)
            if S.active:
                S.input_batch = out
            return out
        return run

    def slot_maps(orig):
        def run(*a, **kw):
            out = orig(*a, **kw)
            if S.active:
                S.slot_maps = out
            return out
        return run

    def run_fullgraph(orig):
        def run(self, batch_desc, *a, **kw):
            if not S.active:
                return orig(self, batch_desc, *a, **kw)
            mine = _eager(S.runner, None, "FULL", batch_desc)
            out = orig(self, batch_desc, *a, **kw)
            _check(mine, out, "FULL", batch_desc)
            return out
        return run

    def run_pw_graph(orig):
        def run(self, model, model_inputs, *a, **kw):
            if not S.active:
                return orig(self, model, model_inputs, *a, **kw)
            mine = _eager(S.runner, model_inputs, "PIECEWISE", None)
            out = orig(self, model, model_inputs, *a, **kw)
            _check(mine, out, "PIECEWISE", None)
            return out
        return run

    def pb_call(orig):
        def call(self, *args):
            if not S.recording:
                return orig(self, *args)
            S.piece = int(getattr(self, "piecewise_compile_index", -1))
            with Recorder():
                return orig(self, *args)
        return call

    _wrap(mr.GPUModelRunner, "execute_model", execute_model)
    _wrap(mr.GPUModelRunner, "prepare_inputs", prepare_inputs)
    _wrap(mr, "build_slot_mappings_by_layer", slot_maps)
    manager = getattr(cgu, "ModelCudaGraphManager", cgu.CudaGraphManager)
    _wrap(manager, "run_fullgraph", run_fullgraph)
    _wrap(manager, "run_pw_graph", run_pw_graph)
    _wrap(pb.PiecewiseBackend, "__call__", pb_call)
    _install_moe()
    _install_mla()


def _bind_model_state(runner: Any) -> None:
    ms = runner.model_state
    if getattr(ms, "_spot_bound", False):
        return
    orig = ms.prepare_attn

    def prepare_attn(*a, **kw):
        out = orig(*a, **kw)
        if S.active:
            S.attn = out
        return out
    ms.prepare_attn = prepare_attn
    ms._spot_bound = True


def _install_moe() -> None:
    from vllm.model_executor.layers.fused_moe import fused_moe as fm
    try:
        from vllm.model_executor.layers.fused_moe.experts import triton_moe
    except ImportError:
        triton_moe = None
    names = ("A", "B", "C", "A_scale", "B_scale", "topk_weights", "sorted_token_ids", "expert_ids", "num_tokens_post_padded",
             "mul_routed_weight", "top_k", "config")

    def expert(orig):
        def launch(*a, **kw):
            if not S.recording:
                return orig(*a, **kw)
            args = dict(zip(names, a))
            args.update({k: v for k, v in kw.items() if k in names or k == "block_shape"})
            ins = {k: _ref(v) for k, v in args.items() if k != "C"}
            out = orig(*a, **kw)
            _event("moe_launch", ins, {"C": _ref(args["C"])})
            return out
        return launch

    for mod in (fm, triton_moe):
        if mod is not None and hasattr(mod, "invoke_fused_moe_triton_kernel"):
            _wrap(mod, "invoke_fused_moe_triton_kernel", expert)


def _install_mla() -> None:
    try:
        from vllm.v1.attention.backends.mla import flashmla_sparse as fs
    except ImportError:
        return
    import torch

    def kernel(orig):
        def run(self, q, kv_rows, topk_indices, *a, **kw):
            if not S.recording:
                return orig(self, q, kv_rows, topk_indices, *a, **kw)
            rest = dict(zip(("topk_length", "actual_num_heads"), a))
            rest.update(kw)
            ins = {"q": _ref(q), "kv_rows": _ref(kv_rows), "topk_indices": _ref(topk_indices), **{k: _ref(v) for k, v in rest.items()}}
            n = min(S.mla_tokens, int(topk_indices.shape[0]))
            if n:
                idx = topk_indices[:n].long().clamp(min=0)
                ins["kv_gathered"] = _ref(kv_rows.reshape(-1, kv_rows.shape[-1])[idx.reshape(-1)].reshape(n, idx.shape[1], -1))
                ins["kv_gathered_tokens"] = n
            out = orig(self, q, kv_rows, topk_indices, *a, **kw)
            outs = out if isinstance(out, tuple) else (out,)
            _event("sparse_mla_kernel", ins, {f"r{i}": _ref(t) for i, t in enumerate(outs) if isinstance(t, torch.Tensor)})
            return out
        return run

    if hasattr(fs.FlashMLASparseImpl, "_bf16_flash_mla_kernel"):
        _wrap(fs.FlashMLASparseImpl, "_bf16_flash_mla_kernel", kernel)


def _recorder_base():
    from torch.utils._python_dispatch import TorchDispatchMode
    return TorchDispatchMode


class Recorder(_recorder_base()):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        packet = getattr(func, "_overloadpacket", None)
        base = getattr(packet, "__name__", str(func)).split(".")[-1]
        ns = getattr(func, "namespace", "aten")
        if ns == "aten" and base in SKIP:
            return func(*args, **kwargs)
        schema = func._schema
        flat = list(args) + [kwargs.get(a.name) for a in schema.arguments[len(args):]]
        ins = {a.name: _ref(v) for a, v in zip(schema.arguments, flat)}
        out = func(*args, **kwargs)
        outs = {f"ret{i}": _ref(t) for i, t in enumerate(out if isinstance(out, (tuple, list)) else (out,))}
        for a, v in zip(schema.arguments, flat):
            if a.alias_info is not None and a.alias_info.is_write:
                outs[a.name] = _ref(v)
        _event(f"{ns}::{base}", ins, outs)
        return out


def _params(runner: Any) -> None:
    if S.params:
        return
    for name, t in list(runner.model.named_parameters()) + list(runner.model.named_buffers()):
        S.params[(t.data_ptr(), tuple(t.shape), tuple(t.stride()), str(t.dtype))] = name
        S.param_of[name] = t


def _ref(v: Any) -> Any:
    import torch
    if isinstance(v, torch.Tensor):
        key = (v.data_ptr(), tuple(v.shape), tuple(v.stride()), str(v.dtype))
        if key in S.params:
            S.used_params.add(S.params[key])
            return {"param": S.params[key], "shape": list(v.shape), "dtype": str(v.dtype)}
        nb = v.numel() * v.element_size()
        d = {"shape": list(v.shape), "dtype": str(v.dtype), "stride": list(v.stride())}
        if v.device.type != "cuda" and v.device.type != "cpu":
            return {**d, "device": v.device.type}
        if nb > BIG:
            return {**d, "big": nb}
        raw = v.detach().contiguous().view(torch.uint8).reshape(-1).cpu().numpy().tobytes() if nb else b""
        h = hashlib.sha512(raw).hexdigest()
        if h not in S.seen:
            S.seen.add(h)
            p = S.root() / "blobs" / h[:2] / h
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(raw)
            S.bytes += len(raw)
        return {**d, "sha512": h}
    if isinstance(v, (list, tuple)):
        return [_ref(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _ref(x) for k, x in v.items()}
    if isinstance(v, (int, float, bool, str)) or v is None:
        return v
    return {"repr": repr(v)[:200]}


def _event(kind: str, ins: dict, outs: dict) -> None:
    S.events.append({"seq": len(S.events), "piece": S.piece, "op": kind, "in": ins, "out": outs})


def _eager(runner: Any, model_inputs: dict | None, mode: str, batch_desc: Any) -> Any:
    import torch
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor, get_forward_context, set_forward_context

    S.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    _params(runner)
    S.t0 = time.time()
    S.mode = mode
    ib = S.input_batch
    S.recording = True
    try:
        if model_inputs is None:
            inputs = {"input_ids": ib.input_ids, "positions": ib.positions, "inputs_embeds": None, "intermediate_tensors": None,
                      **runner.model_state.prepare_inputs(ib, runner.req_states)}
            with set_forward_context(S.attn, runner.vllm_config, num_tokens=ib.num_tokens_after_padding,
                                     cudagraph_runtime_mode=CUDAGraphMode.NONE,
                                     batch_descriptor=BatchDescriptor(num_tokens=ib.num_tokens_after_padding,
                                                                      has_lora=runner.lora_config is not None,
                                                                      num_active_loras=getattr(batch_desc, "num_active_loras", 0)),
                                     slot_mapping=S.slot_maps, is_padding=getattr(ib, "is_padding", None)):
                out = runner.model(**inputs)
        else:
            fc = get_forward_context()
            was = fc.cudagraph_runtime_mode
            fc.cudagraph_runtime_mode = CUDAGraphMode.NONE
            try:
                out = runner.model(**model_inputs)
            finally:
                fc.cudagraph_runtime_mode = was
        out = out[0] if isinstance(out, tuple) else out
        S.piece = -1
        _event("model_output", {}, {"hidden": _ref(out)})
        return out.clone()
    except Exception:  # noqa: BLE001
        S.errors.append(traceback.format_exc()[-4000:])
        return None
    finally:
        S.recording = False


def _check(mine: Any, out: Any, mode: str, batch_desc: Any) -> None:
    import torch
    got = out[0] if isinstance(out, tuple) else out
    ib = S.input_batch
    n = int(getattr(ib, "num_tokens", got.shape[0] if got is not None else 0))
    S.check = {"mode": mode, "tokens": n, "padded": int(getattr(ib, "num_tokens_after_padding", 0)),
               "recorded_forward": mine is not None}
    if mine is not None and isinstance(got, torch.Tensor):
        S.check.update(equal_tokens=bool(torch.equal(mine[:n], got[:n])), equal_padded=bool(torch.equal(mine, got)),
                       differ_words=int((mine[:n].view(torch.int16) != got[:n].view(torch.int16)).sum().item())
                       if mine.dtype == torch.bfloat16 else None)


def _finish(runner: Any) -> None:
    import torch
    d = S.root() / f"step{S.step}"
    d.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n" for e in S.events).encode()
    (d / "ops.jsonl").write_bytes(lines)
    sched = S.sched
    reqs = {str(k): int(v) for k, v in getattr(sched, "num_scheduled_tokens", {}).items()}
    from concurrent.futures import ThreadPoolExecutor

    def digest(name: str) -> tuple[str, dict]:
        t = S.param_of[name]
        raw = t.detach().contiguous().view(torch.uint8).reshape(-1).cpu().numpy()
        return name, {"shape": list(t.shape), "dtype": str(t.dtype), "sha512": hashlib.sha512(memoryview(raw)).hexdigest()}
    with ThreadPoolExecutor(8) as pool:
        params = dict(pool.map(digest, sorted(S.used_params)))
    (S.root() / "params.json").write_text(json.dumps(params, indent=0, sort_keys=True))
    meta = {"rank": S.rank, "step": S.step, "check": getattr(S, "check", None), "scheduled": reqs, "events": len(S.events),
            "blob_bytes": S.bytes, "errors": S.errors, "seconds": round(time.time() - S.t0, 1),
            "ops_sha512": hashlib.sha512(lines).hexdigest(),
            "params_sha512": hashlib.sha512((S.root() / "params.json").read_bytes()).hexdigest()}
    (d / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    print(f"[spot-capture] rank {S.rank} step {S.step}: {len(S.events)} events, {S.bytes >> 20} MiB, check {meta['check']}, "
          f"errors {len(S.errors)}", flush=True)
    S.events, S.errors = [], []


def site(d: Path) -> Path:
    """A directory that, on PYTHONPATH, registers this file as the ``vllm.general_plugins`` entry ``verity_spot_capture``."""
    import shutil
    info = d / "verity_spot_capture-0.0.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: verity-spot-capture\nVersion: 0.0\n")
    (info / "entry_points.txt").write_text("[vllm.general_plugins]\nverity_spot_capture = spot_capture:register\n")
    shutil.copyfile(__file__, d / "spot_capture.py")
    return d


if __name__ == "__main__":
    if sys.argv[1:2] != ["site"] or len(sys.argv) != 3:
        raise SystemExit("capture.py site DIR")
    print(site(Path(sys.argv[2])))
