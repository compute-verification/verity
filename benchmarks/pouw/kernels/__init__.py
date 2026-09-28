"""The PoUW honest GEMMs for the RTX 4090 (``pouw_gemm.cu``), JIT-built as a torch extension on first use."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load():
    from torch.utils.cpp_extension import load as _load

    return _load(name="pouw_gemm", sources=[str(Path(__file__).with_name("pouw_gemm.cu"))],
                 extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_89,code=sm_89", "-lineinfo"], verbose=False)
