"""circuit-check's bindings of `verity.ml.definitions.registry.pearl_c_noise`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pearl_c_noise_roots() -> list[Any]:
    """A noise line's integer, norm root, scale, code and codes, on words and on bits: static-free, so they are catalog roots
    anyway, and here so each is checked whatever its size. The scale and the code reach their word primitives."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import pearl_c_noise as N

    return [bind(N.PearlCLineScale), bind(N.PearlCLineCode), bind(N.PearlCLineCodes), bind(N.PearlCLineIntV2),
            bind(N.PearlCLineNormRootV2), bind(N.PearlCLineScaleV2), bind(N.PearlCLineCodeV2), bind(N.PearlCLineCodesV2)]
