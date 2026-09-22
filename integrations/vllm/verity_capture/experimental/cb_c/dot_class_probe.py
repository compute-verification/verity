"""Observability probe (cb-c validation, derive-vs-declare spec): the tensor-core instruction class of vLLM's batch-invariant GEMM
kernel is a *compile-time* fact of (kernel source, constexprs, target arch). Triton can compile the pinned `matmul_kernel_persistent`
ahead of time for a declared target without any GPU (`triton.compile(ASTSource, target=GPUTarget("cuda", arch, 32))`); the emitted PTX
names the instruction (`mma.sync.aligned.m16n8k16` on sm8x, `wgmma.mma_async` on sm90). Prints JSON per arch.

Usage: python dot_class_probe.py 89 90 [80]   (arches to compile for; no CUDA device needed)"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import sys

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from vllm.model_executor.determinism.batch_invariant import matmul_kernel_persistent

# The B0 profile pins (verity_capture/profiles/vllm_d9105ea80_sm89_eager.py): BLOCK 128/128/64, GROUP 8, NUM_SMS 128, *_LARGE False,
# HAS_BIAS False, num_warps 8, num_stages 3.
CONSTEXPRS = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "NUM_SMS": 128,
              "A_LARGE": False, "B_LARGE": False, "C_LARGE": False, "HAS_BIAS": False}
SIG = {"a_ptr": "*bf16", "b_ptr": "*bf16", "c_ptr": "*bf16", "bias_ptr": "*bf16", "M": "i32", "N": "i32", "K": "i32",
       "stride_am": "i32", "stride_ak": "i32", "stride_bk": "i32", "stride_bn": "i32", "stride_cm": "i32", "stride_cn": "i32"}
SIG.update({k: "constexpr" for k in CONSTEXPRS})

src_sha = hashlib.sha256(inspect.getsource(matmul_kernel_persistent.fn).encode()).hexdigest()
res = {"kernel": f"{matmul_kernel_persistent.fn.__module__}.{matmul_kernel_persistent.fn.__name__}", "src_sha256": src_sha,
       "constexprs": CONSTEXPRS, "num_warps": 8, "num_stages": 3, "triton": triton.__version__, "per_arch": {}}
for arch in [int(a) for a in sys.argv[1:]] or [89, 90]:
    try:
        target = GPUTarget("cuda", arch, 32)
        src = ASTSource(fn=matmul_kernel_persistent, signature=SIG, constexprs=CONSTEXPRS)
        comp = triton.compile(src, target=target, options={"num_warps": 8, "num_stages": 3})
        ptx = comp.asm["ptx"]
        insts = sorted({m for m in re.findall(r"\b(mma\.sync\.aligned\.[a-z0-9.]+|wgmma\.mma_async\.[a-z0-9.]+|"
                                             r"wgmma\.[a-z_]+|mma\.[a-z0-9.]+|cp\.async\.bulk[a-z0-9.]*|ldmatrix[a-z0-9.]*)", ptx)})
        res["per_arch"][arch] = {
            "target_ptx": re.search(r"\.target\s+(\S+)", ptx).group(1),
            "tensor_core_instructions": [i for i in insts if i.startswith(("mma", "wgmma"))],
            "other_hw_specific": [i for i in insts if not i.startswith(("mma", "wgmma"))][:6],
            "metadata": {k: comp.metadata._asdict().get(k) if hasattr(comp.metadata, "_asdict") else getattr(comp.metadata, k, None)
                         for k in ("target", "num_warps", "num_stages", "shared", "hash", "name")},
            "cubin_sha256": hashlib.sha256(comp.asm.get("cubin", b"")).hexdigest()[:16],
            "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest()[:16],
        }
    except Exception as e:  # noqa: BLE001
        res["per_arch"][arch] = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
print(json.dumps(res, indent=1, default=str))
