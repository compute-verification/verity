"""The C++ models the MUFU primitives of `verity_vllm.program.registry.prims` were transliterated from, with their tables:
FA2's (`fa2_model`, `fa2_relation`) and the Triton RMSNorm's (`rms_triton_model`, `rms_relation`), loaded once for the
twins and the replay driver."""
from __future__ import annotations

from functools import lru_cache


# ---------------------------------------------------------------------------------------------------------
# the C++ models the MUFU primitives were transliterated from, with their tables (twins, the replay driver)
# ---------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _fa2():
    from verity_vllm.program.kernels import fa2_model, fa2_relation
    return fa2_model, fa2_relation.tables()


@lru_cache(maxsize=None)
def _rms():
    from verity_vllm.program.kernels import rms_relation, rms_triton_model
    return rms_triton_model, rms_relation.tables("cuda")
