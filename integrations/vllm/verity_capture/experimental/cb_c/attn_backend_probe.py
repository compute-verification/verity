"""Observability probe (cb-c validation, derive-vs-declare spec): the attention backend / FlashAttention version that the pinned vLLM
selects for the B0 model is an attribute of the instantiated module tree (meta device, no kernel, no CUDA context) — provided
`current_platform.get_device_capability()` answers (pynvml on this host; a declared capability elsewhere). Prints JSON.

Usage: CUDA_VISIBLE_DEVICES=<as desired> PYTHONPATH=. python -m verity_capture.experimental.cb_c.attn_backend_probe [--case B0]"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="B0")
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--declare-capability", default=None, help="major.minor to inject when pynvml cannot answer (e.g. 8.9)")
    args = ap.parse_args()
    from verity_vllm_adapter import profile as prof

    env = prof.apply_env()
    import torch
    from vllm.config import CompilationConfig, set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform

    out: dict = {"env": env, "cuda_available": torch.cuda.is_available(), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    cap = None
    try:
        cap = current_platform.get_device_capability()
    except Exception as e:  # noqa: BLE001
        out["get_device_capability_error"] = f"{type(e).__name__}: {e}"
    out["platform"] = type(current_platform).__name__
    out["device_capability_observed"] = None if cap is None else (cap.major, cap.minor)
    if cap is None and args.declare_capability:
        from vllm.platforms.interface import DeviceCapability

        mj, mn = (int(x) for x in args.declare_capability.split("."))
        type(current_platform).get_device_capability = classmethod(lambda cls, device_id=0: DeviceCapability(major=mj, minor=mn))
        out["device_capability_declared"] = (mj, mn)
    cp = prof.checkpoint_for(args.case, args.checkpoints)
    ea = EngineArgs(model=cp["local_path"], tokenizer=cp["local_path"], dtype="bfloat16", seed=0, max_model_len=prof.max_model_len_for(cp),
                    trust_remote_code=False, enforce_eager=True, enable_prefix_caching=False, max_num_seqs=1,
                    compilation_config=CompilationConfig(mode=0, backend="eager", custom_ops=["all"], cudagraph_mode="NONE"))
    vc = ea.create_engine_config()
    out["attention_config.backend"] = str(getattr(getattr(vc, "attention_config", None), "backend", None))
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.utils.torch_utils import set_default_torch_dtype

    with set_current_vllm_config(vc, check_compile=False):
        init_distributed_environment(world_size=1, rank=0, distributed_init_method="tcp://127.0.0.1:29519", local_rank=0, backend="gloo")
        ensure_model_parallel_initialized(1, 1)
    with set_default_torch_dtype(vc.model_config.dtype), set_current_vllm_config(vc, check_compile=False), torch.device("meta"):
        model = initialize_model(vllm_config=vc, model_config=vc.model_config)
    attn = None
    for name, mod in model.named_modules():
        if hasattr(mod, "attn_backend") and hasattr(mod, "impl"):
            attn = (name, mod)
            break
    if attn is None:
        out["attention_layer"] = None
    else:
        name, mod = attn
        impl = mod.impl
        be = mod.attn_backend
        rec = {"layer": name, "attn_backend": type(be).__module__ + "." + type(be).__qualname__,
               "get_name": be.get_name() if hasattr(be, "get_name") else None,
               "impl": type(impl).__module__ + "." + type(impl).__qualname__,
               "vllm_flash_attn_version": getattr(impl, "vllm_flash_attn_version", None),
               "kv_cache_dtype": getattr(impl, "kv_cache_dtype", None), "head_size": getattr(impl, "head_size", None),
               "num_heads": getattr(impl, "num_heads", None), "num_kv_heads": getattr(impl, "num_kv_heads", None)}
        try:
            import vllm.vllm_flash_attn as vfa

            fa_dir = Path(vfa.__file__).parent
            mods = sorted(p.name for p in fa_dir.glob("_vllm_fa*_C*.so"))
            rec["fa_modules_in_wheel"] = mods
            v = rec["vllm_flash_attn_version"]
            so = [p for p in fa_dir.glob(f"_vllm_fa{v}_C*.so")]
            if so:
                rec["fa_op_module_sha256"] = hashlib.sha256(so[0].read_bytes()).hexdigest()
                rec["fa_op"] = f"_vllm_fa{v}_C." + ("varlen_fwd" if v == 2 else "fwd")
        except Exception as e:  # noqa: BLE001
            rec["fa_modules_error"] = f"{type(e).__name__}: {e}"
        out["attention_layer"] = rec
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
