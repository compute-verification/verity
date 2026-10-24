"""circuit-check's bindings of `verity.ml.definitions.registry.sampling`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import sampling

    return [
        ("GumbelTokenSelect_v1", lambda: bind(sampling.GumbelTokenSelect, V=16)),
        ("GumbelTokenSelectSharedGreedy_v1", lambda: bind(sampling.GumbelTokenSelectSharedGreedy, V=16)),
        ("GumbelTopPTokenSelectSharedGreedy_v1", lambda: bind(sampling.GumbelTopPTokenSelectSharedGreedy, V=16)),
        # the static-seed Gumbel argmax (T = 0.7f) and the erf GELU; LayerNormAten is the b1-eager-v3 vocabulary's
        ("GumbelArgmaxBf16_v1", lambda: bind(sampling.GumbelArgmaxBf16, V=16, SEED=12345, POS=3)),
        ("GumbelArgmaxTempBf16_v1", lambda: bind(sampling.GumbelArgmaxTempBf16, V=16, SEED=12345, POS=3, T=0x3F333333)),
    ]
