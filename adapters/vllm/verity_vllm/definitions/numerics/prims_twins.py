"""The C++ models `registry.prims`' MUFU and norm primitives were transliterated from, with their tables, each loaded once:
the twins (`kernels.twins`) and the replay driver read them.  The primitives are the definition; these are not."""

from __future__ import annotations

from functools import cache


@cache
def fa2_twin():
    from verity_vllm.definitions.numerics import fa2_model, fa2_relation
    return fa2_model, fa2_relation.tables()


@cache
def rms_twin():
    from verity_vllm.definitions.numerics import rms_relation, rms_triton_model
    return rms_triton_model, rms_relation.tables("cuda")
