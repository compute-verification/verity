"""The FP32/BF16 arithmetic primitives of `verity.ml.fp32`: ids, descriptor digests and outputs.

These primitives were registered by the vLLM integration (`verity_vllm.program.registry.prims`) under the same ids, and its
stored Programs cite them. `PINNED` holds, per primitive, the digests recorded from that registration: the Definition's
descriptor digest, the digest of a program making one call to it, and a fingerprint of its outputs over edge and seeded random
encodings. A primitive whose digest or outputs change is a new version of it, not an edit.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random

import pytest

from verity.ir.annotations import definition_digest
from verity.ir.codec import encode_program, program_digest
from verity.ir.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition, bind
from verity.ir.program import Program
from verity.ml import fp32 as F

PINNED = {  # id: (definition digest, one-call program digest, outputs fingerprint)
    "F32Add_v1": ("b011ec610ad80fdb7115c987245a3a629043d1409aa0b6a82bbd01f99d41b4cb",
                  "168c31f970cb0f992bbe104d57a88e22673b2b36741f2628cdda3ea4addb118f",
                  "4cf9c7aa0f064e619dff90fb84bda778d4d90d207873843af416d6c6cf58c006"),
    "F32Mul_v1": ("2390267b8f55f4e59e648205873977ee8e5dd13efa4d453a6842d9bf5bbd6862",
                  "2508f09a9cbe0f52442abd34dc1d9066570d0d969a5904bc634883bf099c923e",
                  "ddc6bf5cb03d3cfaa68ce4da6aa91b74ab3c1de81a47dd627eee72173dcbb357"),
    "F32Fma_v1": ("8f64321ddf02d59b831aec87edf7d1336c3d2e4e2a27468b5819b44f49678af6",
                  "70b251fe165707d71c0b7b6062002ae6fe0d79cbe2c6a72a8d940458db505de9",
                  "54f324eaf0a562b52c664be37233cf69463d1a4b0f3ea6b2c6eab08a5e784771"),
    "F32Div_v1": ("05331ebc8acb50858f42bfc59937e288a0a2c67033f6debe2ccdd76f3d69e4c4",
                  "4b4728341fd3e014bbc884245002823dff47a64cfb4bdde095fbebbdd855b61c",
                  "e3ec34582deabc4404bab9a936871c0e202e0eafcf1ecb000eb9160b90ece538"),
    "F32AddFtz_v1": ("720dc90457ef3543ff220319c28f0960b23671fc23aade004532a732fce8206e",
                     "3ebbc4609194128fb7cde267a860b514314aed972ce4a54442b554a3206d0398",
                     "adb8f54f23993f1e4b4ea900e9e05ebd6b77212dfeb7c7bc9eaa43d6939da2e4"),
    "F32SubFtz_v1": ("f63ce33ab3d07799a7269b294ae92b3ee1535c2b289bbabc3c444285a30dfd1c",
                     "692462fc79f9cd14e71d236426dd511ed4d64ec6f8b2e28d11c1c6f4877ef09e",
                     "f7c526ed180632e27d0fcdbaaa1ee7da315684bb728a4149a20a417b0931efb0"),
    "F32MulFtz_v1": ("79665387c6095b7ddd58b6a752e1b486d4e442b53b34ced6e1dd5f030aad1362",
                     "8f64299806e31df6a52b0284d0db994c85c742eac3ebdcdf88ce5b6915f1a626",
                     "6c4b0edbfec033d7f3e84597877971f4c7ac32bf4b263699511d207d5437ec61"),
    "F32FmaFtz_v1": ("7bcb27f5f2877b385829a771e103ebe7db75ca78b41a64cf306f0c2eeb211dae",
                     "292b8fdedf60fd30b0a8a8c1b6ca583034c5304a1a555e59957cf08a6aece8d1",
                     "692d413409c71658f3f536fc722379ed0c80e3a8f0967b201f0dde7d2c6ef4e0"),
    "F32FmaSubFtz_v1": ("adc7226403e77dc97c327dcdfb468a0eca3c9d3a00da9c7732b36ffbe80b7dbe",
                        "a4e302fc37af23d4f15f63035a4c758efee7b531d3f3d460d27d3b013938bcb7",
                        "bef0361b1ff5c3894d2b94d83acf2836cf6972c0d82764bf56c56df3551984dc"),
    "F32Max_v1": ("97ed961c7f37e1b45e378018a96de03f3b9e8e86454122c6d92db4ac537130c8",
                  "53d1bf3ed680c698249de133207d93ae26e9567d2dba2df4d8b2d7369e148957",
                  "f9da795c8467141e30c3ba99b1ad8a3ed3533c4a6074a59309cc8870a7659145"),
    "GuardNegInfZero_v1": ("787c9c60b7f839e790e01717c481dd629f4d25b1f92496f5fe463fa6dbbe3a84",
                           "8d21eb0c2b000cdb2e17caa837c5c1bea9d308b6ee0739b0c1f9e611f6cc9d2f",
                           "439221d8fad50ae1518aae9cf0fe4e2365b25e5d47f75819b1f92d3506908e2b"),
    "Bf16Add_v1": ("95e76485479e7c81d65c47ae63f39f711d94f043f389b3ebd46c979d62f415d0",
                   "e9e9465d0b5b5f0bbf06209bba6ac85b77cf06b1cfd745610b817c77daf5270b",
                   "10c5eadf59f5e6463fc2615a478186ffb81e0310d98de4c9cf5a060f9fae65a2"),
    "Bf16AddF2fp_v1": ("c76f691158993bbeb5d896c7f0fb5accb27aadd9051813fc23cb54ea65d619e9",
                       "1521b2a20b99673b2793d1b4835a68b060ebacc45698bc379632236e1c244cb0",
                       "d58b6a186a6107deef6ab3d04f3fa87550f343301bd21852d3a78a3cf5edb168"),
    "Bf16MulF32_v1": ("544d579049f4e05ad14cc7c9f996413912cbb8469e763e95ec3c12463c7b0cef",
                      "93b7349ca8ee619f9d622d21f0f8a18b6aee5889e53342f47aa54ebcec2ad3c5",
                      "e15c5e495b1fa7a1a1d6705bd508f92d4140de9e7581b29fe4d62f48078e625e"),
    "Bf16MulBf16_v1": ("1f941d28d9a31be1062a79a57835af7e1efbc9c9770003bbd96f546be0cc4f9a",
                       "54608dde522b61cbead534ac067255b02b17c465df06901f02b8b8de2c548031",
                       "684a6118a19bdd56bb71ca893f121a40443b17526116899b2c0214f36986900b"),
}

F32_EDGES = [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000, 0x3F800000,
             0xBF800000, 0x3F800001, 0x3EFFFFFF, 0x4B000000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000,
             0xFFC00000, 0x7F800001, 0x7FFFFFFF, 0x33800000, 0x0C000000]
BF16_EDGES = [0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x3F80, 0xBF80, 0x3F81, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0,
              0xFFC0, 0x7F81, 0x7FFF, 0x4000]


def _prim(fid: str) -> PrimitiveDefinition:
    return getattr(F, fid.rsplit("_v", 1)[0])


def _inputs(p: PrimitiveDefinition, n_random: int = 4000) -> list[tuple[int, ...]]:
    """The edge grid (every other edge per operand for three operands) then `n_random` seeded random encodings."""
    widths = [t.w for _, t in p.params]
    edges = [F32_EDGES if w == 32 else BF16_EDGES for w in widths]
    if len(widths) > 2:
        edges = [e[::2] for e in edges]
    rng = random.Random(f"fp32-move/{p.id}")
    return list(itertools.product(*edges)) + [tuple(rng.getrandbits(w) for w in widths) for _ in range(n_random)]


def _one_call_program(p: PrimitiveDefinition) -> Program:
    sig = p.signature()
    probe = CompositeDefinition(f"Probe{p.name}", 1, (), lambda S: sig, lambda B, S, *args: B.call(p, *args), register=False)
    return Program(bind(probe))


def test_registered_once_under_the_pinned_ids() -> None:
    public = {n for n in F.__all__ if isinstance(getattr(F, n), PrimitiveDefinition)}
    assert {f"{n}_v1" for n in public} == set(PINNED)
    for fid in PINNED:
        assert _prim(fid).id == fid and REGISTRY[fid] is _prim(fid)


@pytest.mark.parametrize("fid", sorted(PINNED))
def test_descriptor_and_program_digests(fid: str) -> None:
    desc = encode_program(_one_call_program(_prim(fid)))
    assert (definition_digest(desc, fid), program_digest(desc)) == PINNED[fid][:2]


@pytest.mark.parametrize("fid", sorted(PINNED))
def test_outputs(fid: str) -> None:
    p = _prim(fid)
    h = hashlib.sha256()
    for x in _inputs(p):
        h.update(json.dumps([list(x), p.evaluate(*x)]).encode())
    assert h.hexdigest() == PINNED[fid][2]


def test_stated_semantics() -> None:
    assert F.F32Fma.evaluate(0x7F800000, 0, 0x3F800000) == F.F32_NAN  # inf * 0
    assert F.F32Fma.evaluate(0x3F800001, 0x3F800001, 0xBF800002) == 0x28800000  # one rounding: (1+u)^2 - (1+2u) = u^2
    assert (F.F32Add.evaluate(1, 0), F.F32AddFtz.evaluate(1, 0)) == (1, 0)
    assert (F.F32Mul.evaluate(0x00800000, 0x3F000000), F.F32MulFtz.evaluate(0x00800000, 0x3F000000)) == (0x00400000, 0)
    assert (F.Bf16Add.evaluate(0x7F81, 0x3F80), F.Bf16AddF2fp.evaluate(0x7F81, 0x3F80)) == (F.BF16_NAN_TORCH, F.BF16_NAN_F2FP)
    assert (F.GuardNegInfZero.evaluate(0xFF800000), F.GuardNegInfZero.evaluate(0x7F800000)) == (0, 0x7F800000)
    assert (F.F32Max.evaluate(0x7FC00000, 0x3F800000), F.F32Max.evaluate(0x3F800000, 0x7FC00000)) == (0x3F800000, 0x7FC00000)
