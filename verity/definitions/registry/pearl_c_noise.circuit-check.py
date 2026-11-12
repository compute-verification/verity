"""circuit-check's bindings of `verity.definitions.registry.pearl_c_noise`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`).

Two Boolean Definitions are over the 65,536 ANDs circuit-check lowers a Definition whole at, so neither is lowered whole
and neither has a pin of its own: `PearlCRowSeed_v2` is 112,691 and `PearlCNoiseLine_v2` 73,459. Each is the σ of a
composition of units under that size, bound and pinned here (`pearl_c_noise.UNITS`): the row seed's two compressions,
`PearlCRowSeedHead_v2` and `PearlCRowSeedTail_v2`, and the noise line's hash and codes, `PearlCLineRaw_v2` and
`PearlCLineCodes_v2`. Their other checks run on the whole: the word view, the partition, the proof block and the gates.

A row's digest and a row's line are over that size at every K, and are bound once each, small: `PearlCRowDigest_v2{K=40}` is
three compressions, each a Call of `PearlCRowBlock_v2` or `PearlCShaBlock_v2`, bound and pinned here, and
`PearlCRowLine_v2{K=40}` that digest, the row seed, the line key and the noise line."""
from __future__ import annotations

from typing import Any


def _pearl_c_noise_roots() -> list[Any]:
    """-h3's four hashes (one SHA-512 compression each; the row seed two, each a unit of its own), a line's integer, norm
    root, scale, code and codes, and a noise line, on words and on bits: static-free, so they are catalog roots anyway, and
    here so each is checked whatever its size. A compression's word form reaches its byte primitive, and the scale and the
    code theirs. Then a row's digest, its two block Definitions, and a row's line, the two with K small."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import pearl_c_noise as N

    return [bind(N.PearlCSeedB), bind(N.PearlCRowSeedHead), bind(N.PearlCRowSeedTail), bind(N.PearlCRowSeed),
            bind(N.PearlCLineKey), bind(N.PearlCLineRaw), bind(N.PearlCLineScale),
            bind(N.PearlCLineCode), bind(N.PearlCLineCodes), bind(N.PearlCNoiseLine),
            bind(N.PearlCSeedBV2), bind(N.PearlCRowSeedHeadV2), bind(N.PearlCRowSeedTailV2), bind(N.PearlCRowSeedV2),
            bind(N.PearlCLineKeyV2), bind(N.PearlCLineRawV2),
            bind(N.PearlCLineIntV2), bind(N.PearlCLineNormRootV2), bind(N.PearlCLineScaleV2), bind(N.PearlCLineCodeV2),
            bind(N.PearlCLineCodesV2), bind(N.PearlCNoiseLineV2),
            bind(N.PearlCRowBlock), bind(N.PearlCShaBlock), bind(N.PearlCRowDigest, K=40), bind(N.PearlCRowLine, K=40),
            bind(N.PearlCRowBlockV2), bind(N.PearlCShaBlockV2), bind(N.PearlCRowDigestV2, K=40),
            bind(N.PearlCRowLineV2, K=40)]
