"""CPU models of the kernels production vLLM runs, which tests and the sampled replay hold Definitions and committed words to.

* :mod:`.mma` -- one ``mma.sync.m16n8k16`` on Ampere (BF16 = ``verity.ml.circuits.silicon.models.AMPERE_BF16_M16N8K16``,
  FP16 added);
* :mod:`.cpu_model` -- the C++ twin and the parametrised full-GEMM model (K-tile chain, split-K reduction, bias epilogue,
  BF16 output rounding);
* :mod:`.gemm_relation` -- ``independent_gemm(node, opened_inputs) -> bytes`` for the replay checker, and
  :mod:`.fa2_relation` and :mod:`.rms_relation` the same for attention and RMSNorm.

The C++ models are in ``cpp/``; the GPU probe kernels in ``cuda/`` run on a GPU host only.
"""

__all__ = ["mma", "cpu_model", "gemm_relation"]
