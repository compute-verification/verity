"""circuit-check's bindings of `verity.definitions.registry.pearl_c_noise`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`).

Two Boolean Definitions have no pin, since circuit-check lowers a Definition whole only up to 65,536 ANDs:
`PearlCRowSeed_v2` is 112,691 (two SHA-512 compressions) and `PearlCNoiseLine_v2` 73,459 (a compression and a line's codes).
Their other checks run: the word view, the partition, the proof block and the gates. The noise line's two halves,
`PearlCLineRaw_v2` and `PearlCLineCodes_v2`, are pinned on their own; the row seed's two compressions are not Definitions
of their own."""
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
