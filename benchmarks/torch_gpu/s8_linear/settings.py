"""s8_linear's settings: the variables wgmma.py reads, declared once (`verity_config.settings`).

It owns ``S8_WGMMA_``: a variable with it that nothing here declares is refused when the registry loads.  It doesn't own
``S8_``, which names constants across the tree (S8_K_LIMIT).
run.sh's own variables (S8_MAX_MIN, S8_KEEP_GOING, VERITY_BENCH_VENV, CUDA) are the shell's.
"""
from __future__ import annotations

from verity_config.settings import VALUE, WHERE, Registry, Setting as S

REGISTRY = Registry(
    __file__, ("S8_WGMMA_",),
    S("S8_WGMMA_CUBIN", str, None, VALUE, "the cubin wgmma.lib() loads (default build/s8_wgmma.cubin; run.sh sets the ship's): "
                                          "its sha256 is recorded"),
    S("S8_LIBCUDA", str, "libcuda.so.1", WHERE, "the CUDA driver library the cubin loads through"))
