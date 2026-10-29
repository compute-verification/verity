"""libvc's settings: every variable vclib.py and build.sh read, declared once (`verity_config.settings`).

It owns ``VC_``: a variable with it that nothing here declares is refused when the registry loads.  build.sh reads
VC_NVCC_FLAGS itself (it is shell); it is declared here so that a process holding it isn't refused and its record names it.
gpu.sh's own variables (QUICK, BENCH, LEASE_ARGS) are the shell's.
"""
from __future__ import annotations

from verity_config.settings import VALUE, WHERE, Registry, Setting as S

REGISTRY = Registry(
    __file__, ("VC_",),
    S("VC_LIB", str, None, VALUE, "the libvc.so vclib.Lib loads when it is given no path (default build/libvc.so beside "
                                  "vclib.py; gpu.sh sets its fresh build): the library whose launches are checked"),
    S("VC_CUDART", str, None, WHERE, "the libcudart.so vclib.Cuda tries first (then CUDA's usual names): the runtime the "
                                     "tests' device memory and streams come from, not what libvc computes"),
    S("VC_NVCC_FLAGS", str, None, VALUE, "build.sh's extra nvcc flags (-DVC_LEAF_MIN_BLOCKS=1), which change the library "
                                         "it builds"))
