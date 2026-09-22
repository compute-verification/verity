"""DERIVE-VLLM-1 pre-check (i): does `torch.export(strict=False)` trace the actual pinned vLLM model class on meta?
Records each stage's outcome and the EXACT first refusal (exception type, message, traceback tail) to a JSON file.

    PYTHONPATH=. python -m verity_capture.experimental.cb_a.probe_export --out /workspace/cp/cba/probe.json [--M 1 --pos0 8]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

from .inputs_trace import InputsTrace


def _git_rev() -> str:
    try:
        return json.load(open("EXPORT.json"))["commit"]
    except Exception:  # noqa: BLE001
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:  # noqa: BLE001
            return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--pos0", type=int, default=8)
    ap.add_argument("--backend", default="gloo")
    ap.add_argument("--no-intercept-triton", action="store_true", help="plain export (records the raw first refusal)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    res: dict = {"schema": "cb-a/probe_export/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "code_rev": _git_rev(),
                 "M": a.M, "pos0": a.pos0, "stages": []}

    def stage(name, fn):
        t0 = time.time()
        try:
            out = fn()
            res["stages"].append({"stage": name, "ok": True, "secs": round(time.time() - t0, 2)})
            return out
        except BaseException as e:  # noqa: BLE001
            tb = traceback.format_exc()
            res["stages"].append({"stage": name, "ok": False, "secs": round(time.time() - t0, 2), "exception": type(e).__name__,
                                  "message": str(e)[:4000], "traceback_tail": tb[-6000:]})
            res["first_refusal"] = {"stage": name, "exception": type(e).__name__, "message": str(e)[:4000]}
            json.dump(res, open(a.out, "w"), indent=1)
            print(f"[probe] FAILED at {name}: {type(e).__name__}: {str(e)[:500]}", file=sys.stderr)
            raise SystemExit(2)

    with InputsTrace(a.out.replace(".json", ".inputs.json")):
        from . import vllm_meta as vm
        import torch
        res["torch"] = torch.__version__
        import vllm
        res["vllm"] = vllm.__version__
        cfg = stage("vllm_config", lambda: vm.make_vllm_config())
        res["model_config"] = {"architectures": list(cfg.model_config.architectures), "dtype": str(cfg.model_config.dtype),
                               "hf_config": {k: v for k, v in cfg.model_config.hf_config.to_dict().items()
                                             if k in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "intermediate_size",
                                                      "vocab_size", "rms_norm_eps", "rope_theta", "tie_word_embeddings", "max_position_embeddings", "head_dim")},
                               "enforce_eager": cfg.model_config.enforce_eager, "compilation_mode": str(cfg.compilation_config.mode),
                               "custom_ops": list(cfg.compilation_config.custom_ops), "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT")}
        res["dist_backend"] = stage("distributed_init", lambda: vm.ensure_distributed(cfg, a.backend))
        model = stage("instantiate_meta", lambda: vm.instantiate_meta(cfg))
        res["model_class"] = f"{type(model).__module__}.{type(model).__qualname__}"
        res["n_params_tensors"] = sum(1 for _ in model.parameters())
        res["n_buffers"] = sum(1 for _ in model.buffers())
        res["attn_layers"] = sorted(cfg.compilation_config.static_forward_context)[:4] + ["..."]
        res["attn_backend"] = None
        try:
            l0 = next(iter(cfg.compilation_config.static_forward_context.values()))
            res["attn_backend"] = f"{type(l0.impl).__module__}.{type(l0.impl).__qualname__}"
            res["attn_use_direct_call"] = bool(l0.use_direct_call)
        except Exception as e:  # noqa: BLE001
            res["attn_backend_err"] = str(e)
        from .triton_capture import TritonLaunchLog
        log = TritonLaunchLog()
        res["intercept_triton"] = not a.no_intercept_triton
        ep = stage("torch_export", lambda: vm.export_step(model, cfg, a.pos0, a.M, intercept_triton=not a.no_intercept_triton, launch_log=log))
        res["ok"] = True
        res["triton_launches"] = {"n": len(log.launches), "kernels": sorted({l["kernel"] for l in log.launches}), "first": log.launches[:2]}
        res["op_histogram"] = vm.op_histogram(ep)
        res["n_nodes"] = len(list(ep.graph.nodes))
        sig = ep.graph_signature
        res["signature"] = {"user_inputs": list(sig.user_inputs), "n_params": len(sig.inputs_to_parameters), "n_buffers": len(sig.inputs_to_buffers),
                            "lifted_constants": list(sig.inputs_to_lifted_tensor_constants.values())[:20],
                            "custom_objs": list(getattr(sig, "inputs_to_lifted_custom_objs", {}) or {}),
                            "output_kinds": [s.kind.name for s in sig.output_specs]}
        gtxt = a.out.replace(".json", ".graph.txt")
        with open(gtxt, "w") as f:
            f.write(str(ep.graph))
        res["graph_txt"] = gtxt
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("ok", "n_nodes", "op_histogram") if k in res}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
