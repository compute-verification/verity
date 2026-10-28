"""The Definition the differential test of `GumbelTopPTokenSelect_v1{V,S}(logits, top_p, seed, pos, temp)` checks
production against (`spec`: the registered composite's body that returns ALSO its intermediate masked fp32 row, so the
`TopPMask_v1{V,S}` value is checked lane by lane, not only the sampled token), and the registered family (`spec_registered`).
The adapter (gate G4, `verity_vllm.properties.admission`: cases and the pinned production chain of vLLM d9105ea80's
`Sampler.sample`) is `verity_vllm.program.sampling_topp_adapter`."""

from __future__ import annotations

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import tuple_of
from verity.ml.definitions.registry.prims import BF16, F32, I32
from verity.ml.definitions.registry.b1 import const
from verity.ml.definitions.registry.sampling import I64, GumbelSelectF32, GumbelTopPTokenSelect, TemperatureScale, TopPMask
from verity.ml.circuits.types import Array, Tuple


def _difftest_body(B, S, logits, top_p, seed, pos, temp):
    """`GumbelTopPTokenSelect_v1{V}` returning ALSO its intermediate masked fp32 row (production output 1), so the difftest checks
    `TopPMask_v1{V}` lane by lane and not only the sampled token.  Same body as the registered composite; S is an OPERAND
    of the registered forms (`splits`, i32) -- here the launch geometry `production` recorded (`splits_for(1, num_SMs)` of the device)
    enters as a `Const32[S]`, the same way the fold carries the split kernels' constexpr."""
    scaled = B.call(bind(TemperatureScale, V=S.V), logits, temp)
    masked = B.call(bind(TopPMask, V=S.V), scaled, top_p, B.call(const(32, int(S.S))))
    tok = B.call(bind(GumbelSelectF32, V=S.V), masked, seed, pos, temp)
    return tuple_of(tok, masked)


GumbelTopPTokenSelectDifftest = CompositeDefinition(
    "GumbelTopPTokenSelectDifftest", 1, ("V", "S"),
    lambda S: ((("logits", Array(S.V, BF16)), ("top_p", F32), ("seed", I64), ("pos", I64), ("temp", F32)), Tuple(I32, Array(S.V, F32))),
    _difftest_body, doc=_difftest_body.__doc__ or "", conformance="derived: difftest", register=False)


def spec(statics: dict):
    return bind(GumbelTopPTokenSelectDifftest, V=int(statics["V"]), S=int(statics["S"]))


def spec_registered(statics: dict):
    """The registered family (token only) — what the Program uses; S is its 6th operand (`splits`, i32), not a static."""
    return bind(GumbelTopPTokenSelect, V=int(statics["V"]))
