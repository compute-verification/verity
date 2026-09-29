"""The FP32/BF16 arithmetic primitives of `verity.ml.fp32`: ids, descriptor digests and outputs.

The `_v1` primitives were registered by the vLLM integration (`verity_vllm.program.registry.prims`) under the same ids, and its
stored Programs cite them. `PINNED` holds, per primitive, the digests recorded from that registration (for `_v2`, from this
module's): the Definition's descriptor digest, the digest of a program making one call to it, and a fingerprint of its outputs
over edge and seeded random encodings. A primitive whose digest or outputs change is a new version of it, not an edit.
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
    "F32Add_v2": ("12f180926e83c75a1f0759c7b3780386f16acdcd1a5f985c3c93c189c69e1e41",
                  "cf4ed39d72af7fd37ac579a12d3cb7d2f0a235ffe2c5eb3de00e6ee798d197ad",
                  "3089c1922cd621a2beec4a0bd8e0bb0998a326879d379e23d86b9006fa44faf9"),
    "F32Mul_v2": ("7825875ba870b09be23a8abd4441d452d31a17076005b2508b739bb6649b5177",
                  "b63986b87e38a5bc52150a8e4592fe04edae432ed91f271bc887d6d0b99fd458",
                  "db263544bdac0e6b654f543bdf8ae44a11aa896b3aa18a4712d55231246db97f"),
    "F32Fma_v2": ("fc3dcb5dae699ce495fbf36e06864cdad57cf4cce1763acd40259f8ee41e6906",
                  "cfdef9a122bb17c8bd7404b3f69c5a9f7839a750329e9575a0891e1cdc146612",
                  "3b0be66ac9a8e8840cdd9b1b6930f245acd355fcf1c14d9b77c30e96b5279c52"),
    "F32Div_v2": ("0fb99aab89af013c258442a26f0a4fa8e544f8bd529455e1c326a1c0194af2f7",
                  "74a8839475d52428d05c8f4fd35b4641988f95d3ff2e25ce109d60d44b05d99d",
                  "f168e1bb8f339894aa3d50191b2cb7d91a799ba802c6e139649d69c1e427f0f7"),
    "F32AddFtz_v2": ("2c50545bc04fe67cfb2786e6cb42a1b656183015e703d2b32dc222aada0e7792",
                     "6ba7279d0a7943df465af4fb4e9c64e76aa68f983a99da2934599bff92fe79e0",
                     "9f46cfa8a21694f1c514aa768f497ad60c0b3a4b1ac85fc8b6a24e52670423c7"),
    "F32SubFtz_v2": ("13b228d534fa1374860a606b75dbde77f974483226d604796fc37b6984927731",
                     "53d5b3b054d24095a029b579ebe623874b3b22531a6526e1808a52ca5948fe79",
                     "9389a10e652cc3f52b124c1a22fe57a5ea36a27532540961cfabbdae6e05a3de"),
    "F32MulFtz_v2": ("4abff5f191ef31148be33ca8133757f61d51ea294e15380abba63c8cc55d8952",
                     "d5395a57b3b9787cda3cff6af323c19021ec49ccbacc3d19f968a73c590afcb6",
                     "e6e8e1f05b16054b70c55e9fc929f1f6dcffded8a312afa35172561e8fa9994a"),
    "F32FmaFtz_v2": ("55d8a86a7e02dbbcd67992fc0bea017bcb1e2767b46baeecb2cdf100e092e406",
                     "51453e957266337f269e04d3c43e365c5df62080c8abd4d39ec7fbf95082fc4f",
                     "f8b4ebfd65a8a625421c34e9461400604c7489e0a8a69b46a2d593d58fe8e1c5"),
    "F32FmaSubFtz_v2": ("b0b9ab8025aff90508c3a1522c2ee9125c06c63c7988067e00c7bb497b50fcec",
                        "130f2bf26bdc5094d03a3d00f89df9958d42f2399ac71cda8b66e6315c815b97",
                        "b44a13653c99dd7514ac53267fb1cacfb9bb7e55f361ba9a2f20891a2c710c3c"),
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


PUBLIC = {p.id: p for p in (getattr(F, n) for n in F.__all__) if isinstance(p, PrimitiveDefinition)}


def _prim(fid: str) -> PrimitiveDefinition:
    return PUBLIC[fid]


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
    assert set(PUBLIC) == set(PINNED)
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
    assert (F.F32Add.evaluate(0x7F800000, 0xFF800000), F.F32AddV2.evaluate(0x7F800000, 0xFF800000)) == (0xFFC00000, F.F32_NAN_PTX)
    assert (F.F32Mul.evaluate(0xFFC00123, 1), F.F32MulV2.evaluate(0xFFC00123, 1)) == (0xFFC00123, F.F32_NAN_PTX)


V2 = sorted(fid for fid in PUBLIC if fid.endswith("_v2"))


@pytest.mark.parametrize("fid", V2)
def test_v2_is_v1_but_for_nan_results(fid: str) -> None:
    v1, v2 = _prim(fid.replace("_v2", "_v1")), _prim(fid)
    assert v1.signature() == v2.signature()
    bad = [x for x in _inputs(v1) if v2.evaluate(*x) != F.f32_gpu_nan(v1.evaluate(*x))]
    assert not bad, (fid, len(bad), [tuple(map(hex, x)) for x in bad[:4]])


#: the GPU's words (fixtures/gpu-nan-2026-09-28: the L40S and the H100, every build alike) for each primitive's PTX instruction
GPU_OPS = {"F32Add_v2": "add.rn.f32", "F32AddFtz_v2": "add.rn.ftz.f32", "F32SubFtz_v2": "sub.rn.ftz.f32", "F32Mul_v2": "mul.rn.f32",
           "F32MulFtz_v2": "mul.rn.ftz.f32", "F32Div_v2": "div.rn.f32", "F32Fma_v2": "fma.rn.f32", "F32FmaFtz_v2": "fma.rn.ftz.f32"}


def _gpu_cases(op: str) -> list[tuple[tuple[int, ...], int]]:
    from pathlib import Path
    d = json.loads((Path(__file__).parent / "fixtures" / "gpu-nan-2026-09-28" / "fp32_ops.json").read_text())
    specials = [int(w, 16) for w in d["specials"]]
    words = d["ops"][op]
    outs = [int(words[i:i + 8], 16) for i in range(0, len(words), 8)]
    cases = list(itertools.product(specials, repeat=3 if op.startswith("fma") else 2))
    assert len(cases) == len(outs)
    return list(zip(cases, outs))


@pytest.mark.parametrize("fid", sorted(GPU_OPS))
def test_every_special_case_is_the_gpus_word(fid: str) -> None:
    """Every pair (or triple) of the probe's 15 specials: ±0, ±1, ±inf, quiet and signalling NaNs of both signs with payloads.  The
    GPU's word, finite results included, and every NaN result `F32_NAN_PTX`: no operand's payload or sign survives."""
    p = _prim(fid)
    bad = [(tuple(hex(x) for x in c), hex(p.evaluate(*c)), hex(w)) for c, w in _gpu_cases(GPU_OPS[fid]) if p.evaluate(*c) != w]
    assert not bad, (fid, len(bad), bad[:4])


def test_sub_is_the_gpus_sub() -> None:
    """`f32_gpu_nan(f32_sub_bits)` (vLLM's `F32Sub_v2`, FSUB without ftz) on the probe's pairs."""
    got = [(c, F.f32_gpu_nan(F.f32_sub_bits(*c)), w) for c, w in _gpu_cases("sub.rn.f32")]
    assert all(g == w for _c, g, w in got), [(tuple(map(hex, c)), hex(g), hex(w)) for c, g, w in got if g != w][:4]


def test_fma_sub_is_the_gpus_fma_of_the_negated_addend() -> None:
    cases = _gpu_cases("fma.rn.ftz.f32")
    specials = {c[0] for c, _ in cases}
    got = [(c, F.F32FmaSubFtzV2.evaluate(c[0], c[1], c[2] ^ 0x80000000), w) for c, w in cases if c[2] ^ 0x80000000 in specials]
    assert got and all(g == w for _c, g, w in got), [(tuple(map(hex, c)), hex(g), hex(w)) for c, g, w in got if g != w][:4]


def test_v1_is_the_hosts_nan_on_the_probes_words() -> None:
    """`_v1` keeps the host's NaN words, which the GPU doesn't give: its ids are recorded, so it stays as it was."""
    assert any(F.F32Add.evaluate(*c) != w for c, w in _gpu_cases("add.rn.f32"))
    assert any(F.F32Fma.evaluate(*c) != w for c, w in _gpu_cases("fma.rn.f32"))
