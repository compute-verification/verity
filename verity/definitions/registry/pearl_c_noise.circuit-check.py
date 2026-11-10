"""circuit-check's bindings of `verity.definitions.registry.pearl_c_noise`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pearl_c_noise_roots() -> list[Any]:
    """-h3's four hashes (one SHA-512 compression each, the row seed two), a line's integer, norm root, scale, code and codes,
    and a noise line, on words and on bits: static-free, so they are catalog roots anyway, and here so each is checked
    whatever its size. A hash's word form reaches its byte primitive, and the scale and the code theirs."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import pearl_c_noise as N

    return [bind(N.PearlCSeedB), bind(N.PearlCRowSeed), bind(N.PearlCLineKey), bind(N.PearlCLineRaw), bind(N.PearlCLineScale),
            bind(N.PearlCLineCode), bind(N.PearlCLineCodes), bind(N.PearlCNoiseLine),
            bind(N.PearlCSeedBV2), bind(N.PearlCRowSeedV2), bind(N.PearlCLineKeyV2), bind(N.PearlCLineRawV2),
            bind(N.PearlCLineIntV2), bind(N.PearlCLineNormRootV2), bind(N.PearlCLineScaleV2), bind(N.PearlCLineCodeV2),
            bind(N.PearlCLineCodesV2), bind(N.PearlCNoiseLineV2)]
