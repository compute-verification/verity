"""Independent numerical models for the GEMMs production vLLM runs.

Layers, each a separate Table 2 row class (contract §7):

* :mod:`.mma` -- one ``mma.sync.m16n8k16`` on Ampere (BF16 = ``veritor.core.silicon``, FP16 added);
* :mod:`.cpu_model` -- the C++ twin and the parametrised full-GEMM model (K-tile chain, split-K
  reduction, bias epilogue, BF16 output rounding);
* :mod:`.relations` -- ``independent_gemm(node, opened_inputs) -> bytes`` for the replay checker.

GPU-side scripts (probe kernels, Hawkeye's test sequence, random-tile validation, kernel
resolution, model search on production shapes) live alongside and run on the pod only.
"""

__all__ = ["mma", "cpu_model", "relations"]
