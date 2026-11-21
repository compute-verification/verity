"""circuit-check's bindings of `verity_vllm.definitions.registry.boolean_topp_mask_word`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any

#: the keep word on bits at the vocabularies a Program's sampler reaches, made on demand as its word primitive is
BODY_FAMILIES = (r"TopPMaskWordx\d+_v2",)


def _boolean_roots() -> list[Any]:
    """The keep word on bits at V = 2, and its keep words at the smallest and the registered split count. Each is over the
    suite's gate limit (`targets.MAX_GATES`) and bound at no served V, so none holds a pin; the tests hold V = 2, 16 and 300 to
    the word (`tests/adapters/vllm/tests/program/test_boolean_topp_mask_word.py`)."""
    from verity.circuits.defs import bind

    from verity_vllm.definitions.registry import boolean_topp_mask_word as BTM

    return [bind(BTM.topp_mask_word(2)), bind(BTM.TopPKeepWord, V=2, S=1), bind(BTM.TopPKeepWord, V=2, S=32)]
