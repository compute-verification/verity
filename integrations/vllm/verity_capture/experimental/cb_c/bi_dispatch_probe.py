"""Observability probe (cb-c validation, derive-vs-declare spec): which Python callable chain does `RMSNorm(x, residual)` take in the
pinned vLLM under `VLLM_BATCH_INVARIANT=1` vs `0`, on meta tensors, without a GPU? Prints JSON. Run twice with the env var set.

The C++ op `torch.ops._C.fused_add_rms_norm` is the same in both modes; the kernel variant (`fused_add_rms_norm_kernel<T, 8, ·>` width-8
vs `<T, 0, ·>` generic) is chosen inside the .so by `vllm::vllm_is_batch_invariant()` (a cached env read). What *is* observable from
Python: the env value, `vllm_is_batch_invariant()`, and the branch `RMSNorm.forward_cuda` takes (`rms_norm_batch_invariant` -> `_custom_ops`
vs `forward_native` -> `ir.ops.fused_add_rms_norm.maybe_inplace` -> impl selection)."""

from __future__ import annotations

import json
import os
import sys

import torch  # noqa: E402

out: dict = {"env_VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT")}
import vllm.envs as envs  # noqa: E402
from vllm.model_executor.determinism import batch_invariant as bi_mod  # noqa: E402
from vllm.model_executor.layers.layernorm import RMSNorm  # noqa: E402

out["envs.VLLM_BATCH_INVARIANT"] = envs.VLLM_BATCH_INVARIANT
out["batch_invariant._batch_invariant_MODE (before init)"] = bi_mod._batch_invariant_MODE
out["python_getter_exists"] = {n: hasattr(bi_mod, n) for n in ("vllm_is_batch_invariant", "is_batch_invariant_mode_enabled", "init_batch_invariance", "enable_batch_invariant_mode")}
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402

_ctx = set_current_vllm_config(VllmConfig(), check_compile=False)
_ctx.__enter__()
norm = RMSNorm(576, eps=1e-5).to(torch.device("meta"))
x = torch.empty(8, 576, dtype=torch.bfloat16, device="meta")
r = torch.empty_like(x)
chain: list[str] = []


def prof(frame, event, arg):
    if event == "call":
        co = frame.f_code
        chain.append(co.co_filename.split("site-packages/")[-1] + ":" + co.co_name)
    elif event == "c_call":
        chain.append("<c> " + str(getattr(arg, "__qualname__", getattr(arg, "__name__", arg))))


sys.setprofile(prof)
try:
    norm.forward_cuda(x, r)
    res = "ok"
except Exception as e:  # noqa: BLE001
    res = f"{type(e).__name__}: {str(e)[:300]}"
finally:
    sys.setprofile(None)
keep = [c for c in chain if ("vllm/" in c and any(k in c for k in ("layernorm", "batch_invariant", "custom_ops", "/ir/", "kernels/")))
        or "_C" in c or "rms_norm" in c]
out["result"] = res
out["python_call_chain"] = keep[:60]
try:
    from vllm import ir

    op = ir.ops.fused_add_rms_norm
    impls = getattr(op, "impls", {})
    out["ir_impls_registered"] = {k: type(v).__name__ for k, v in impls.items()} if isinstance(impls, dict) else str(impls)
except Exception as e:  # noqa: BLE001
    out["ir_impls_registered"] = f"{type(e).__name__}: {e}"
print(json.dumps(out, indent=1))
