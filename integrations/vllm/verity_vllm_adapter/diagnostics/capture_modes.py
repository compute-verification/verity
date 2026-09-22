"""Capture op-boundary values of ONE request under one execution mode (CONTRACT §16.6).

Run ON THE POD, one process per mode (the batch-invariant flag and the compilation mode are
process-lifetime state)::

    PYTHONPATH=. python -m verity_vllm_adapter.diagnostics.capture_modes --mode eager \
        --checkpoints manifests/checkpoints.json --workload results/B0/workload.json \
        --request 784c37e1 --out /workspace/diag/eager
    PYTHONPATH=. python -m verity_vllm_adapter.diagnostics.capture_modes --mode compiled \
        ... --out /workspace/diag/compiled --cache-root /workspace/vllm_cache/diag_compiled

Eager (``enforce_eager=True``, the adapter's replay/eager-production profile): forward hooks on
every op-level ``nn.Module`` (embedding, RMSNorm, the three GEMMs, rotary, attention, SiluAndMul,
logits processor) plus the decoder layers; the pre-hook also stores the *inputs* of RMSNorm,
SiluAndMul and rotary modules so the reproducer has authenticated operands.

Compiled (stock ``mode 3`` / Inductor, ``custom_ops ['none']``, IR-op priority ``native``):
``torch._inductor.config.generate_intermediate_hooks = True`` makes the generated wrapper call
``run_intermediate_hooks(<fx node name>, <buffer>)`` after every kernel whose output buffer
corresponds to a post-grad FX node. Nothing about the kernels changes (the hook is a Python call
in the wrapper after the launch). CUDA graphs are disabled in this diagnostic run
(``cudagraph_mode NONE``; a graph replay would skip the wrapper). The effective compilation
config is recorded. Each compiled piece's post-grad FX graph (after vLLM's passes, i.e. exactly
the graph Inductor lowered) is dumped with node name, target, shapes, ``nn_module_stack`` and
the last frame of ``stack_trace`` so buffers can be mapped to canonical ops.

Every captured tensor is copied to host at hook time (synchronising copy) and saved as raw words
(``.npy`` of uint16/uint32/int32/int64 views): BF16 words are compared bit for bit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["eager", "compiled"], required=True)
ap.add_argument("--case", default="B0")
ap.add_argument("--checkpoints", required=True)
ap.add_argument("--workload", required=True)
ap.add_argument("--request", default="784c37e1", help="request id prefix")
ap.add_argument("--max-tokens", type=int, default=1)
ap.add_argument("--out", required=True)
ap.add_argument("--cache-root", default=None, help="VLLM_CACHE_ROOT (fresh dir for the compiled run)")
ap.add_argument("--max-rows", type=int, default=None, help="keep only the first N rows of per-token tensors")
ap.add_argument("--custom-ops", default=None, choices=["all", "none"])
args = ap.parse_args()

if args.cache_root:
    os.environ["VLLM_CACHE_ROOT"] = args.cache_root

from verity_vllm_adapter import profile as prof  # noqa: E402

ENV_FLAGS = prof.apply_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

if args.mode == "compiled":
    import torch._inductor.config as inductor_config

    inductor_config.generate_intermediate_hooks = True

from torch._inductor.hooks import intermediate_hook  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from verity_vllm_adapter.driver import run_requests  # noqa: E402
from verity_vllm_sampler.workload import load_workload  # noqa: E402

OUT = Path(args.out)
(OUT / "values").mkdir(parents=True, exist_ok=True)
(OUT / "graphs").mkdir(exist_ok=True)
(OUT / "kernels").mkdir(exist_ok=True)

# ------------------------------------------------------------------------------------------
# raw-word serialisation
# ------------------------------------------------------------------------------------------
WORD_VIEW = {
    torch.bfloat16: (torch.int16, np.uint16),
    torch.float16: (torch.int16, np.uint16),
    torch.float32: (torch.int32, np.uint32),
    torch.int32: (torch.int32, np.int32),
    torch.int64: (torch.int64, np.int64),
    torch.bool: (torch.uint8, np.uint8),
    torch.uint8: (torch.uint8, np.uint8),
    torch.float8_e4m3fn: (torch.uint8, np.uint8),
}


def to_words(t: torch.Tensor) -> np.ndarray:
    tv, nv = WORD_VIEW[t.dtype]
    return t.detach().contiguous().view(tv).cpu().numpy().view(nv)


SEQ = 0
EVENTS: list[dict] = []


def record(kind: str, node: str, port: int, t: torch.Tensor, extra: dict | None = None) -> None:
    """Snapshot ``t`` now (synchronising host copy) and append an event."""
    global SEQ
    if not isinstance(t, torch.Tensor):
        return
    torch.cuda.synchronize()
    t2 = t
    if args.max_rows is not None and t.dim() >= 2 and t.shape[0] > args.max_rows:
        t2 = t[: args.max_rows]
    if t.dtype not in WORD_VIEW:
        return
    w = to_words(t2)
    fname = f"{SEQ:05d}.npy"
    np.save(OUT / "values" / fname, w)
    ev = {
        "seq": SEQ,
        "kind": kind,
        "node": node,
        "port": port,
        "shape": list(t.shape),
        "kept_shape": list(t2.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "stride": list(t.stride()),
        "data_ptr": t.data_ptr(),
        "file": fname,
        "sha256": hashlib.sha256(w.tobytes()).hexdigest(),
    }
    if extra:
        ev.update(extra)
    EVENTS.append(ev)
    SEQ += 1


# ------------------------------------------------------------------------------------------
# compiled: post-grad graph dumps (one per compiled piece) + intermediate hooks
# ------------------------------------------------------------------------------------------
GRAPH_DUMPS: list[dict] = []


def _val_desc(v):
    if isinstance(v, torch.Tensor):
        return {"shape": [str(s) for s in v.shape], "dtype": str(v.dtype).replace("torch.", "")}
    if isinstance(v, (list, tuple)):
        return [_val_desc(x) for x in v]
    return None


def dump_graph(graph) -> dict:
    nodes = []
    for n in graph.nodes:
        meta = n.meta or {}
        mstack = meta.get("nn_module_stack") or {}
        mods = [v[0] if isinstance(v, (tuple, list)) else str(v) for v in mstack.values()]
        st = meta.get("stack_trace")
        st_last = None
        if st:
            lines = [ln.strip() for ln in st.strip().splitlines() if ln.strip()]
            st_last = lines[-2:] if len(lines) >= 2 else lines
        nodes.append(
            {
                "name": n.name,
                "op": n.op,
                "target": str(n.target) if n.op != "placeholder" else None,
                "args": [a.name if hasattr(a, "name") and hasattr(a, "op") else repr(a) for a in n.args],
                "val": _val_desc(meta.get("val")),
                "module_stack": mods,
                "stack_trace_tail": st_last,
                "original_aten": str(meta.get("original_aten")) if meta.get("original_aten") is not None else None,
                "users": [u.name for u in n.users],
            }
        )
    return {"index": len(GRAPH_DUMPS), "nodes": nodes, "utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}


if args.mode == "compiled":
    from vllm.compilation.passes import pass_manager as _pm

    _orig_pm_call = _pm.PostGradPassManager.__call__

    def _pm_call(self, graph):
        _orig_pm_call(self, graph)
        d = dump_graph(graph)
        GRAPH_DUMPS.append(d)
        (OUT / "graphs" / f"piece_{d['index']:03d}.json").write_text(json.dumps(d))
        (OUT / "graphs" / f"piece_{d['index']:03d}.txt").write_text(str(graph))

    _pm.PostGradPassManager.__call__ = _pm_call

HOOK_FILES: dict[str, int] = {}
KERNEL_CALLS: list[dict] = []
PARAM_PTRS: dict[int, str] = {}
MAX_SAVE_NUMEL = 8_000_000

if args.mode == "compiled":
    # The pinned nightly's own ``generate_intermediate_hooks`` emission is dead code for Triton
    # kernels: ``codegen_node_schedule`` compares scheduler *op* names (``op1``) against
    # ``live_output_buffers()`` (``buf1``) so no ``run_intermediate_hooks`` line is ever written
    # (verified with a 10-line torch.compile micro-test: counter ``intermediate_hooks`` stays 0).
    # We therefore emit our own line after every kernel launch in the generated Python wrapper:
    # ``run_intermediate_hooks('K<i>|<kernel>', {'buf3': locals().get('buf3'), ...})`` for every
    # plain buffer/arg name passed to the kernel. The kernels themselves are untouched; the line
    # is a host-side Python call between launches (which is why CUDA graphs must be off).
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen
    from torch._inductor.virtualized import V

    _orig_gkc = PythonWrapperCodegen.generate_kernel_call
    _ARG = re.compile(r"^(buf\d+|arg\d+_\d+)$")

    def _gkc(self, kernel_name, call_args, **kw):
        _orig_gkc(self, kernel_name, call_args, **kw)
        names = [a for a in call_args if isinstance(a, str) and _ARG.match(a)]
        meta = {}
        for a in names:
            info: dict = {}
            try:
                b = V.graph.try_get_buffer(a)
            except Exception:
                b = None
            if b is not None:
                for key, fn in (("dtype", lambda: str(b.get_dtype()).replace("torch.", "")), ("shape", lambda: [str(s) for s in b.get_size()])):
                    try:
                        info[key] = fn()
                    except Exception:
                        pass
                try:
                    on = b.get_origin_node()
                    info["origin_node"] = on.name if on is not None else None
                except Exception:
                    info["origin_node"] = None
                try:
                    info["origins"] = sorted(n.name for n in b.origins)
                except Exception:
                    pass
                try:
                    info["kind"] = type(b).__name__
                except Exception:
                    pass
            meta[a] = info
        idx = len(KERNEL_CALLS)
        KERNEL_CALLS.append(
            {
                "index": idx,
                "graph": str(getattr(V.graph, "name", "")),
                "kernel": kernel_name,
                "call_args": [str(a) for a in call_args],
                "tensor_args": meta,
                "graph_dump_index": len(GRAPH_DUMPS) - 1,
            }
        )
        dict_src = ", ".join(f"{a!r}: locals().get({a!r})" for a in names)
        self.writeline(f"run_intermediate_hooks({f'K{idx}|{kernel_name}'!r}, {{{dict_src}}})")

    PythonWrapperCodegen.generate_kernel_call = _gkc


def _wrapper_frame():
    fr = sys._getframe(1)
    while fr is not None and fr.f_code.co_name != "call":
        fr = fr.f_back
    return fr


def inductor_hook(name: str, val) -> None:
    fr = _wrapper_frame()
    fn = fr.f_code.co_filename if fr is not None else "?"
    HOOK_FILES[fn] = HOOK_FILES.get(fn, 0) + 1
    extra = {"wrapper_file": fn, "wrapper_line": fr.f_lineno if fr is not None else None}
    if isinstance(val, dict):
        kidx = int(name.split("|", 1)[0][1:]) if name.startswith("K") else None
        for arg, t in val.items():
            if not isinstance(t, torch.Tensor):
                continue
            ex = dict(extra, kernel_call_index=kidx, arg=arg)
            pname = PARAM_PTRS.get(t.data_ptr())
            if pname is not None:
                ex["param"] = pname
                EVENTS.append({"seq": None, "kind": "inductor_param", "node": name, "port": 0, "arg": arg, "param": pname, "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""), "kernel_call_index": kidx})
                continue
            if t.numel() > MAX_SAVE_NUMEL:
                EVENTS.append({"seq": None, "kind": "inductor_large", "node": name, "port": 0, "arg": arg, "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""), "kernel_call_index": kidx})
                continue
            record("inductor", name, 0, t, ex)
    else:
        record("inductor", name, 0, val, extra)


# ------------------------------------------------------------------------------------------
# eager: module hooks
# ------------------------------------------------------------------------------------------
RECORD_CLASSES = re.compile(
    r"(VocabParallelEmbedding|RMSNorm|QKVParallelLinear|RowParallelLinear|MergedColumnParallelLinear|"
    r"ColumnParallelLinear|ReplicatedLinear|SiluAndMul|^Attention$|Rotary|LogitsProcessor|DecoderLayer|"
    r"LlamaModel|Qwen2Model)"
)
INPUT_CLASSES = re.compile(r"(RMSNorm|SiluAndMul|Rotary|^Attention$|LogitsProcessor)")


def install_module_hooks(model: torch.nn.Module) -> list:
    handles = []
    for name, mod in model.named_modules():
        cls = mod.__class__.__name__
        if not RECORD_CLASSES.search(cls):
            continue
        node = name or "model"

        def post(m, inputs, output, node=node, cls=cls):
            outs = output if isinstance(output, (tuple, list)) else (output,)
            for i, o in enumerate(outs):
                record("module_out", node, i, o, {"cls": cls})

        handles.append(mod.register_forward_hook(post))
        if INPUT_CLASSES.search(cls):

            def pre(m, inputs, node=node, cls=cls):
                for i, o in enumerate(inputs):
                    record("module_in", node, i, o, {"cls": cls})

            handles.append(mod.register_forward_pre_hook(pre))
    return handles


# ------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------
def main() -> None:
    cp = prof.checkpoint_for(args.case, args.checkpoints)
    wl = load_workload(args.workload)
    req = next(r for r in wl.requests if r.request_id.startswith(args.request))

    kw = prof.engine_kwargs(cp, replay=False, enforce_eager=(args.mode == "eager"), custom_ops=args.custom_ops)
    if args.mode == "compiled":
        cc = dict(kw.get("compilation_config") or {})
        cc["cudagraph_mode"] = "NONE"
        cc["inductor_compile_config"] = {"generate_intermediate_hooks": True}
        kw["compilation_config"] = cc
    t0 = time.perf_counter()
    llm = LLM(**kw)
    build_s = time.perf_counter() - t0
    runner = prof.get_model_runner(llm)
    model = runner.get_model()
    vc = llm.llm_engine.vllm_config
    from vllm.ir.op import IrOp

    effective = {
        "compilation": prof.effective_compilation(llm),
        "cache": prof.cache_info(llm),
        "backends": prof.backends_info(llm),
        "compilation_config": {
            "splitting_ops": list(getattr(vc.compilation_config, "splitting_ops", []) or []),
            "use_inductor_graph_partition": bool(getattr(vc.compilation_config, "use_inductor_graph_partition", False)),
            "inductor_compile_config": {k: str(v) for k, v in dict(getattr(vc.compilation_config, "inductor_compile_config", {}) or {}).items()},
            "backend": str(getattr(vc.compilation_config, "backend", None)),
            "compile_sizes": [str(s) for s in (getattr(vc.compilation_config, "compile_sizes", None) or [])],
            "pass_config": {k: str(v) for k, v in vars(vc.compilation_config.pass_config).items()} if hasattr(vc.compilation_config, "pass_config") else None,
        },
        "ir_op_priority": {name: op.get_priority() for name, op in IrOp.registry.items()},
        "ir_enable_torch_wrap": bool(getattr(vc.compilation_config, "ir_enable_torch_wrap", True)),
        "inductor_generate_intermediate_hooks": bool(torch._inductor.config.generate_intermediate_hooks),
        "env_flags": ENV_FLAGS,
        "VLLM_CACHE_ROOT": os.environ.get("VLLM_CACHE_ROOT"),
        "TORCHINDUCTOR_CACHE_DIR": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "engine_kwargs": {k: v for k, v in kw.items() if k not in ("model", "tokenizer")},
        "model_class": model.__class__.__name__,
        "engine_build_s": build_s,
    }
    (OUT / "effective.json").write_text(json.dumps(effective, indent=2, default=str) + "\n")
    print("EFFECTIVE", json.dumps(effective["compilation"]), effective["ir_op_priority"], flush=True)

    # small parameters (norm weights) for the reproducer; GEMM weights are referenced by checkpoint sha
    params = {}
    for pname, p in model.named_parameters():
        PARAM_PTRS[p.data_ptr()] = pname
        if p.numel() <= 200_000:
            params[pname] = to_words(p.data)
    for bname, b in model.named_buffers():
        PARAM_PTRS.setdefault(b.data_ptr(), "buffer:" + bname)
        if b.numel() <= 2_000_000 and b.dtype in WORD_VIEW:
            params["buffer:" + bname] = to_words(b.data)
    np.savez(OUT / "small_params.npz", **params)
    (OUT / "small_params.json").write_text(json.dumps({k: {"shape": list(v.shape), "sha256": hashlib.sha256(v.tobytes()).hexdigest()} for k, v in params.items()}, indent=1) + "\n")

    # hooks only around OUR request (never during warm-up / compile / profile runs)
    handles = install_module_hooks(model)

    class _Req:
        request_id = req.request_id
        prompt_token_ids = list(req.prompt_token_ids)
        max_tokens = args.max_tokens

    sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=0, max_tokens=args.max_tokens, seed=None, logprobs=1)
    t1 = time.perf_counter()
    if args.mode == "compiled":
        with intermediate_hook(inductor_hook):
            outs, timing = run_requests(llm, [_Req()], lambda r: sp)
    else:
        outs, timing = run_requests(llm, [_Req()], lambda r: sp)
    run_s = time.perf_counter() - t1
    for h in handles:
        h.remove()
    o = outs[req.request_id]
    c = o.outputs[0]
    lps = []
    if c.logprobs:
        for tok, lp in zip(c.token_ids, c.logprobs, strict=False):
            e = lp.get(tok)
            lps.append(e.logprob if e is not None else None)
    result = {
        "mode": args.mode,
        "request_id": req.request_id,
        "prompt_len": len(req.prompt_token_ids),
        "prompt_token_ids_sha256": hashlib.sha256(np.asarray(req.prompt_token_ids, dtype=np.int64).tobytes()).hexdigest(),
        "output_token_ids": list(c.token_ids),
        "chosen_logprobs": lps,
        "num_cached_tokens": int(getattr(o, "num_cached_tokens", 0) or 0),
        "events": len(EVENTS),
        "graph_dumps": len(GRAPH_DUMPS),
        "hook_wrapper_files": HOOK_FILES,
        "timing": timing,
        "run_s": run_s,
        "utc": prof.utc_stamp(),
    }
    # copy the generated wrapper files (they embed the Triton kernel sources) out of the cache
    kernels = []
    for fn in HOOK_FILES:
        p = Path(fn)
        if p.exists():
            dst = OUT / "kernels" / p.name
            shutil.copy2(p, dst)
            kernels.append({"src": fn, "file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size})
    result["wrapper_files"] = kernels
    result["kernel_calls"] = len(KERNEL_CALLS)
    (OUT / "kernel_calls.json").write_text(json.dumps(KERNEL_CALLS, indent=1) + "\n")
    (OUT / "index.json").write_text(json.dumps({"result": result, "events": EVENTS}, indent=1) + "\n")
    (OUT / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT", json.dumps({k: result[k] for k in ("mode", "output_token_ids", "chosen_logprobs", "events", "graph_dumps", "num_cached_tokens")}), flush=True)


if __name__ == "__main__":
    main()
