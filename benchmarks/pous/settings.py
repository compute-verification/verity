"""The environment variables the POUS-only harness reads, each declared once (`verity_config.settings`).

`pod.sh` runs the harness in a venv of its own (numpy, ninja and torch) with the tree, research and difftest on
``PYTHONPATH``, so its files (`STANDALONE`) can't import the registry and read the environment themselves: the run and
the lease that research's runner and gpu-lease set, which `SHARED` declares, and the one variable declared here.
``tests/benchmarks/pous/test_settings.py`` holds each of their reads to its declaration.
"""
from __future__ import annotations

from verity_config.settings import VALUE, Registry
from verity_config.settings import Setting as S

__all__ = ["REGISTRY", "STANDALONE"]

#: the harness's files that read the environment themselves, relative to this one
STANDALONE = ("bench.py", "side_by_side.py", "band_gpu/harness.py", "band_gpu/native.py", "p2_v1/decode_bench.py",
              "p2_v1/harness.py", "p2_v1/scale.py")

REGISTRY = Registry(
    __file__, (),
    S("TORCH_CUDA_ARCH_LIST", str, None, VALUE,
      "the architectures torch's extension build compiles the band kernels for; `band_gpu/native.py` holds it to the "
      "card's own and sets it for the build"),
    standalone=STANDALONE)
