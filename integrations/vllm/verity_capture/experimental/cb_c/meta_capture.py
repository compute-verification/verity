"""[R9 CB-C] The no-GPU route: instantiate the real pinned vLLM model class (``LlamaForCausalLM``) through vLLM's own
config + model construction on the **meta device** (no weights, no CUDA context) and Dynamo-trace ``LlamaModel.forward``
(the ``@support_torch_compile`` graph) with the same recording backend as ``dynamo_capture``, under a
``set_forward_context`` with no attention metadata (the attention custom op is opaque; its fake impl needs none).

    PYTHONPATH=. /workspace/venv-cu129/bin/python -m verity_capture.experimental.cb_c.meta_capture --out DIR [--num-tokens 8]

Answers the selection question "is a GPU needed merely to construct?": either the same Dynamo graph as the GPU pipeline
hands to ``VllmBackend`` (compare node-for-node with ``dump_fx``'s ``dynamo_0.py``), or the exact first refusal.  Every step's
failure is recorded in ``meta_run.json`` (``steps``), the process continues with the next independent step where possible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

_OPENS: list[dict[str, Any]] = []
_SEEN: set[str] = set()


def _audit(event: str, args: tuple) -> None:
    if event != "open":
        return
    try:
        path = args[0]
        path = path if isinstance(path, str) else (os.fsdecode(path) if isinstance(path, (bytes, os.PathLike)) else str(path))
        key = f"{path}|{args[1] if len(args) > 1 else None}"
        if key not in _SEEN:
            _SEEN.add(key)
            _OPENS.append({"path": path, "mode": args[1] if len(args) > 1 else None})
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    sys.addaudithook(_audit)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--case", default="B0")
    ap.add_argument("--num-tokens", type=int, default=8)
    ap.add_argument("--custom-ops", default="all")
    ap.add_argument("--attention-backend", default=None, help="VLLM_ATTENTION_BACKEND override (structure-only; recorded)")
    ap.add_argument("--device", default="meta", choices=["meta", "cpu"])
    ap.add_argument("--num-sms", type=int, default=142, help="injected SM count for matmul_persistent's grid (L40S=142, RTX4090=128)")
    ap.add_argument("--arch-family", default="ada", help="injected tuned-matmul arch family (hopper|ada|none); L40S/RTX4090 = ada (sm_89)")
    ap.add_argument("--dynamic", action="store_true", help="mark the token dim dynamic (Dynamo cannot build Triton HOP grids from SymInts -> fails; default static)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["VERITY_CBC_OUT"] = str(out)
    if args.attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = args.attention_backend
    from verity_vllm_adapter import profile as prof

    env_flags = prof.apply_env()
    import torch

    doc: dict[str, Any] = {"schema": "verity-cbc/meta-capture/v1", "argv": sys.argv, "env_flags": env_flags, "steps": [], "device": args.device,
                           "t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "torch": torch.__version__,
                           "cuda_available": torch.cuda.is_available()}

    def step(name: str, fn):
        t0 = time.perf_counter()
        rec: dict[str, Any] = {"step": name}
        try:
            r = fn()
            rec["ok"] = True
            rec["info"] = r if isinstance(r, (dict, list, str, int, float, type(None))) else str(r)[:500]
        except BaseException as e:  # noqa: BLE001
            rec["ok"] = False
            rec["error"] = {"type": type(e).__name__, "msg": str(e)[:4000], "tb": traceback.format_exc()[-8000:]}
            r = None
        rec["s"] = round(time.perf_counter() - t0, 3)
        doc["steps"].append(rec)
        (out / "meta_run.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k != "info"}, default=str)[:600], flush=True)
        return r

    cp = prof.checkpoint_for(args.case, args.checkpoints)
    doc["checkpoint"] = {k: cp.get(k) for k in ("repo", "revision", "local_path", "config_sha256")}

    # 1. VllmConfig through vLLM's own EngineArgs (no engine, no worker)
    def mk_config():
        import vllm
        from vllm.config import CompilationConfig
        from vllm.engine.arg_utils import EngineArgs

        doc["vllm_version"] = getattr(vllm, "__version__", None)
        ea = EngineArgs(model=cp["local_path"], tokenizer=cp["local_path"], dtype="bfloat16", seed=0, max_model_len=prof.max_model_len_for(cp),
                        trust_remote_code=False, enforce_eager=False, enable_prefix_caching=False, max_num_seqs=1,
                        compilation_config=CompilationConfig(mode=3, backend="eager", custom_ops=[s for s in args.custom_ops.split(",") if s], cudagraph_mode="NONE"))
        vc = ea.create_engine_config()
        cc = vc.compilation_config
        return vc, {"mode": str(cc.mode), "backend": cc.backend, "custom_ops": list(cc.custom_ops), "splitting_ops": list(cc.splitting_ops or []),
                    "attention_backend": str(getattr(getattr(vc, "attention_config", None), "backend", None)), "architectures": list(vc.model_config.architectures or [])}

    r = step("vllm_config", mk_config)
    if r is None:
        _finish(doc, out)
        return
    vc, cfg_info = r
    doc["config_info"] = cfg_info

    # 2. distributed (TP=1, gloo, CPU) -- required by the parallel layers' rank queries
    def mk_dist():
        from vllm.config import set_current_vllm_config
        from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment

        with set_current_vllm_config(vc, check_compile=False):
            init_distributed_environment(world_size=1, rank=0, distributed_init_method="tcp://127.0.0.1:29517", local_rank=0, backend="gloo")
            ensure_model_parallel_initialized(1, 1)
        return "ok"

    step("distributed_init", mk_dist)

    # 3. the real model class on the meta device via vLLM's own construction path
    def mk_model():
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.utils import initialize_model
        from vllm.utils.torch_utils import set_default_torch_dtype

        # same three contexts vLLM's DefaultModelLoader.load_model uses (dtype, config, device); device=meta here
        with set_default_torch_dtype(vc.model_config.dtype), set_current_vllm_config(vc, check_compile=False), torch.device(args.device):
            model = initialize_model(vllm_config=vc, model_config=vc.model_config)
        n_params = sum(1 for _ in model.named_parameters())
        return model, {"class": type(model).__module__ + "." + type(model).__qualname__, "n_params": n_params,
                       "param_names": [n for n, _ in model.named_parameters()][:12], "buffer_names": [n for n, _ in model.named_buffers()][:12]}

    r = step("model_meta", mk_model)
    if r is None:
        _finish(doc, out)
        return
    model, model_info = r
    doc["model_info"] = model_info

    # 3b. Dynamo only recognises triton JITFunction launches (-> triton_kernel_wrapper_mutation HOP) when
    # torch.utils._triton.has_triton() is True, which requires a visible CUDA device. Without a GPU it
    # would inline JITFunction.run and hit the triton driver lookup. Force the predicate (structure only:
    # the HOP node records the kernel object, grid and args; nothing is launched under meta tensors).
    def force_triton():
        import torch.utils._triton as tu

        before = bool(tu.has_triton())
        tu.has_triton = lambda: True  # type: ignore[assignment]
        # matmul_persistent's launch grid is min(#SMs, #tiles): a device constant queried through
        # torch.cuda.get_device_properties (needs a CUDA context). Inject it (recorded; grid-only, does
        # not change which kernel is launched nor its per-tile K-reduction).
        from vllm.model_executor.determinism import batch_invariant as bi

        bi.num_compute_units = lambda idx=None: args.num_sms  # type: ignore[assignment]
        # matmul_persistent's tile configs (BLOCK_M/N/K, stages, warps) are chosen per device arch family
        # ("hopper"/"ada"/None) from current_platform.get_device_capability() -- vLLM normally resolves
        # this before torch.compile. SEMANTICS-RELEVANT (BLOCK_K fixes the K-reduction blocking): inject
        # the target family and record it.
        from vllm.model_executor.determinism import batch_invariant_configs as bic

        fam = args.arch_family if args.arch_family != "none" else None
        bic._TUNED_MATMUL_CONFIGS_FOR_DEVICE = bic._BATCH_INVARIANT_MATMUL_TUNED_CONFIGS.get(fam) if fam else None
        bic._TUNED_MATMUL_CONFIGS_RESOLVED = True
        return {"has_triton_before": before, "forced": True, "num_sms_injected": args.num_sms, "arch_family_injected": fam,
                "tuned_matmul_configs_available": sorted(bic._BATCH_INVARIANT_MATMUL_TUNED_CONFIGS.keys())}

    step("force_has_triton", force_triton)

    # 4. Dynamo trace of LlamaModel.forward under a forward context without attention metadata
    def trace_model():
        from vllm.forward_context import set_forward_context

        from verity_vllm_adapter.compiled_fx import dynamo_capture

        M = args.num_tokens
        dev = torch.device(args.device)
        input_ids = torch.zeros(M, dtype=torch.int64, device=dev)
        positions = torch.arange(M, dtype=torch.int64, device=dev)
        if args.dynamic:
            torch._dynamo.mark_dynamic(input_ids, 0)
            torch._dynamo.mark_dynamic(positions, 0)
        inner = model.model  # the @support_torch_compile class (LlamaModel)
        with set_forward_context(None, vc, num_tokens=M):
            rec = dynamo_capture.capture_callable(f"meta_model_forward_M{M}", inner.forward, input_ids, positions, None, None, dynamic=args.dynamic)
        return {k: v for k, v in rec.items() if k != "graphs"} | {"node_counts": [g["graph"]["node_count"] for g in rec["graphs"]]}

    step("dynamo_trace_model", trace_model)

    def trace_logits():
        from verity_vllm_adapter.compiled_fx import dynamo_capture

        H = int(vc.model_config.hf_config.hidden_size)
        hs = torch.zeros(1, H, dtype=torch.bfloat16, device=torch.device(args.device))
        rec = dynamo_capture.capture_callable("meta_compute_logits", model.compute_logits, hs)
        return {k: v for k, v in rec.items() if k != "graphs"}

    step("dynamo_trace_compute_logits", trace_logits)
    _finish(doc, out)


def _finish(doc: dict[str, Any], out: Path) -> None:
    from verity_capture.experimental.cb_c.dump_fx import FORBIDDEN_SUBSTRINGS, classify

    classes: dict[str, int] = {}
    for o in _OPENS:
        o["class"] = classify(o["path"])
        classes[o["class"]] = classes.get(o["class"], 0) + 1
    doc["inputs_trace"] = {"opens": len(_OPENS), "by_class": classes,
                           "forbidden_hits": [o for o in _OPENS if any(s in o["path"] for s in FORBIDDEN_SUBSTRINGS)],
                           "model_dir_files": sorted({o["path"].rsplit("/", 1)[-1] for o in _OPENS if o["class"] == "model-dir"}),
                           "vllm_source_files": sorted({o["path"].split("site-packages/")[-1] for o in _OPENS if o["class"] == "vllm-source" and o["path"].endswith(".py")})}
    with (out / "opens.jsonl").open("w") as f:
        for o in _OPENS:
            f.write(json.dumps(o) + "\n")
    (out / "meta_run.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(json.dumps({"steps": [(s["step"], s["ok"]) for s in doc["steps"]], "forbidden_hits": len(doc["inputs_trace"]["forbidden_hits"])}))


if __name__ == "__main__":
    main()
