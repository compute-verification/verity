"""circuit-check's bindings of `verity.ml.definitions.registry.pearl_c_noise`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`).

Two Boolean Definitions are over the 65,536 ANDs circuit-check lowers a Definition whole at, so neither is lowered whole
and neither has a pin of its own: `PearlCRowSeed_v2` is 112,691 and `PearlCNoiseLine_v2` 73,459. Each is the σ of a
composition of units under that size, bound and pinned here (`pearl_c_noise.UNITS`): the row seed's two compressions,
`PearlCRowSeedHead_v2` and `PearlCRowSeedTail_v2`, and the noise line's hash and codes, `PearlCLineRaw_v2` and
`PearlCLineCodes_v2`. Their other checks run on the whole: the word view, the partition, the proof block and the gates."""
from __future__ import annotations

from typing import Any


def _pearl_c_noise_roots() -> list[Any]:
    """-h3's four hashes (one SHA-512 compression each; the row seed two, each a unit of its own), a line's integer, norm
    root, scale, code and codes, and a noise line, on words and on bits: static-free, so they are catalog roots anyway, and
    here so each is checked whatever its size. A compression's word form reaches its byte primitive, and the scale and the
    code theirs."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import pearl_c_noise as N

    return [bind(N.PearlCSeedB), bind(N.PearlCRowSeedHead), bind(N.PearlCRowSeedTail), bind(N.PearlCRowSeed),
            bind(N.PearlCLineKey), bind(N.PearlCLineRaw), bind(N.PearlCLineScale),
            bind(N.PearlCLineCode), bind(N.PearlCLineCodes), bind(N.PearlCNoiseLine),
            bind(N.PearlCSeedBV2), bind(N.PearlCRowSeedHeadV2), bind(N.PearlCRowSeedTailV2), bind(N.PearlCRowSeedV2),
            bind(N.PearlCLineKeyV2), bind(N.PearlCLineRawV2),
            bind(N.PearlCLineIntV2), bind(N.PearlCLineNormRootV2), bind(N.PearlCLineScaleV2), bind(N.PearlCLineCodeV2),
            bind(N.PearlCLineCodesV2), bind(N.PearlCNoiseLineV2)]
