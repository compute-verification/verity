"""The superseded Definition versions that stay registered so old records replay (`KEPT`): each at a small binding keeps
its id, its definition entry's digest and its Program's digest.
A changed digest means the version now computes or encodes something else, and the records that name it no longer replay."""
from __future__ import annotations

import hashlib

import pytest
from verity.circuits.codec import canonical_json, definition_entry, program_digest
from verity.circuits.defs import bind
from verity.circuits.program import Program

from verity_vllm.definitions.registry import b1, serve3
from verity_vllm.definitions.registry import boolean_silu as BS
from verity_vllm.definitions.registry import prims as P

TINY: dict[str, object] = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64,
        "QKV": 64, "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 8, "BIAS": True, "tie_word_embeddings": True, "profile": {}}

#: version: (binding, its id, sha512 of its definition entry, its Program's digest or None for a primitive)
KEPT = {
    "RMSNormFusedCuda_v1": (lambda: bind(b1.RMSNormFusedCudaV1, N=64, EPS=1e-5), "RMSNormFusedCuda_v1{N=64,EPS=1e-05}",
                            "723ee5100d3ad9643667c10ed75701cfc74c6a4209465505713003ed9d52190feb5b5b522005aebca48f91a6699f720ed8acd9805b839a8763f4e236084c2dfb",
                            "f023b5e7964a5780962a1014bd122e6881998262547c0b04f8396503441cd3d3de66685d1d5ac09942b85332488825de2ec41d8ad33c19e47b67629047a1c1e1"),
    "SiluMul_v1": (lambda: bind(b1.SiluMul, I=8), "SiluMul_v1{I=8}",
                   "2b6a631db3bcf8fdb50f25f700e7616e45878d18193a02956ed7d1ccb72b955c77b2c80a1efe6b392975a57e4890b1bbfb91a17174681bf3752ae89e0d9a0629",
                   "2ec2ccbd949ab2fa8a8ae2ede5c694292a41297af9addcc9f09f99c36a81ac8a307f47b1983d390ca474bd8d2e1ad04b38cd73183e72d8f698e9c06b8f9bdcaf"),
    "SiluMulBf16_v1": (lambda: P.SiluMulBf16, "SiluMulBf16_v1",
                       "42c178e6c08a3af7f19df304e7508c39f54b3d7664e3535a105fae5f9734272aff252abc27c6fb17f1b77286fac2c553c5cfce65f898969bad198680d3c1f0ef", None),
    "SiluMul_v3": (lambda: bind(BS.SiluMul, I=8), "SiluMul_v3{I=8}",
                   "d2bc7966067f52e23c2ef22b0a84dbb489b0815a6203c1dd91742a83c6b7837f959ff6d9e6708379fa77d39b600c4e8a17a098d79224d96e4f6a385a7a41b8b8",
                   "af4361135ee12be4e44738df8472c3baf10f398a2c109bfae84c531c7f7876aadb39da74c99b11ece573b0b8cccd3032bf167ebd91f23bb0a735cf1788f57801"),
    "SiluMulBf16_v3": (lambda: bind(BS.SiluMulBf16), "SiluMulBf16_v3",
                       "618c3ebb4c96158c3c08b8355c5c3c1e84699720d770cd0b7a67de1c02e73046d36fa0f91e42f6c630f8ed79a35eea0a796b4c630a47c37f9c613b7fd6cc703e",
                       "3d487d3863c89a3f0a095a34be46464bc9a875a9e659c85a87e19931c9193d1689a12ec36a56721c3276f4425c64a2a8b20c005ca3272329a6f23d8e06b13929"),
    "Attention_v1": (lambda: bind(b1.Attention, T=3, NH=2, KVH=1, D=16), "Attention_v1{T=3,NH=2,KVH=1,D=16}",
                     "281d195af5bb7756a896b56971cdc9daf08d5dae91becd00cbe2c2d5359cd9fed7361920021cefc3a21fe894c41e0546b6dec5959e5d8f54cf159edb05e048b2",
                     "1016b20b2b3a1792e4e8542cf2a54e25e778ae1771d8f724d3d34217fe4fef00161720ffda3b80f1038c5b39ebb8e3fc0acfdafbd6583316168f85fc021ea6a4"),
    "AttentionHead_v1": (lambda: bind(b1.AttentionHead, T=3, D=16), "AttentionHead_v1{T=3,D=16}",
                         "cbcdc5435770266bf77eff172177589bb389e7c4b18a25dfcfca4f92a070055f92b237551c29476542bc709b9652f7821011d35e5cab788a8fe1028917dd5abf",
                         "cfcb9afae05d6929d33cf27fce62eb72413c32dcf26311ba600d417dc93dc767b72ea0d2b91364bc829bf71104f4b30494dc11b25431d94290b066e3236b2dcb"),
    "AttnBlock_v1": (lambda: bind(b1.AttnBlock, D=16, NVIS=3, FIRST=True), "AttnBlock_v1{D=16,NVIS=3,FIRST=True}",
                     "af3d05441b14d4ad3f1b3ced84b229d99305606425306e3bc15e92b436095c48e6a93d728ef8de1d523f1fc5f485720aae47168ff91621f8fa3a4185b9fc1a10",
                     "3920e030c59aad593f63a046a8c448b1a0efb97a5731c2195866d91ad3fb0eee9ff29bb5295d4e5f3862df6526fc6cbe3ea18874280ceb3154a758696971e2ff"),
    "Serve_v1": (lambda: bind(b1.Serve, C=TINY, LP=3, STEPS=2), "Serve_v1{C={e2f2f43eb046},LP=3,STEPS=2}",
                 "1adfd4b4963f56ad20f62090cafb82f43f05ee8204e467a35e79e0866d599552f0e742b6e7591d9af3fe50ab443b2f58d4e6dd078f2a5b64467772b92c3d91b5",
                 "f38ca17d9a754aa699c93f4411af7fac17c3d0e6163b172b055acbfbb7158bf2f27ff2eef44acc92bc03965e9030b3e28d87101ec49262217920107f733466fd"),
    "Serve_v2": (lambda: b1.serve_v2(TINY, 3, 2),
                 "Serve_v2{C={e2f2f43eb046},LP=3,STEPS=2,GEMM={4130e3ca726d},ATTENTION={35db6738c571},NORM={4d524c30089f}}",
                 "1a12b2c3d9943948127220e1f2a0cd86d69acb327b58837a61a166acbe717802c3cf6950ce629c882fd0be255e7096c4e1ce06f11afe5823e2cd06c78f57bdad",
                 "a8e562d73ecee1b476e292fd1198cd010ce322d1c2978561323de832c7238452cefc8e123ce77b303092aa87e6cab62bdbd57fe91f9c3b9cfbab34daf422239b"),
    "Serve_v3": (lambda: serve3.serve_v3({**TINY, "EOS_ID": 7}, 3, 2),
                 "Serve_v3{C={fa22b0376a3a},LP=3,D_MAX=2,GEMM={4130e3ca726d},ATTENTION={35db6738c571},NORM={4d524c30089f}}",
                 "4bd455bd1ece645d5201fea695025ec50001a8c115df1834754a95019bb368b397643355e0435c68bda690d5fca4ee9051fbbbf50ebfd9cf03d5d44c366dec6c",
                 "9d9b4b8285a9f1442c396e058305aa842527b306bbc8b7b2ab206255b900348049124e8059c8825f4d5d7dbf74bd13bef43551779d9b3dbfbcde566ae78b4ad3"),
    "LayerPre_v1": (lambda: bind(b1.LayerPre, C=TINY, L0=True), "LayerPre_v1{C={e2f2f43eb046},L0=True}",
                    "04fe569c68f7f7c50f3747f933716df6db693fa485f3be3fdd7b3be039c53bd8909dbaca70f4e36e513d752dfc3d0b7defeb352a58f5d379b14fa1c337f6a418",
                    "71d92c42ca08fc199c184a4e9ef3bed2a3ba80a70f56fd12a829e256a03429af92c0427c16891bd162eb4830550f4dd3e0778956f55ebd997340cb96c9693826"),
    "LayerPost_v1": (lambda: bind(b1.LayerPost, C=TINY), "LayerPost_v1{C={e2f2f43eb046}}",
                     "8dd07e8362083203e50af31d4c6b0cc17ca4417df6b4713fd47b0fa7e307eaab5261f628d1bd0ee6ae9cc71f53fd6fd4192874c82b8cb9fcbb1cad18674f0545",
                     "32a9498fb64ef315619827ef0e4ec34fdf8bbb7b9adb1694ea89072d07c2ffa10a6b1b687df039ab2f3f6e860d9e01c5e1cb7ae1f75b400cff2ad026be7e4375"),
    "Final_v1": (lambda: bind(b1.Final, C=TINY), "Final_v1{C={e2f2f43eb046}}",
                 "f32e9bd0842c07d8a17561ee22f6a1fda708a79ff490a4792d7eb6d12c1ad4ff920d41b4c5c9613ec146f31d0023c40fa05a0add3d5fdfc0370783f8251a82c5",
                 "81a7f92007026048f0b4494eac017be66f1e56e4589e980e1de0cb7f0081d60d9a85e9403f5c0b2aa746d64c20e8b20d2c62474415de9d00fc146780244386bc"),
}


@pytest.mark.parametrize("version", sorted(KEPT))
def test_a_kept_superseded_version_keeps_its_identity(version):
    make, fid, entry, prog = KEPT[version]
    fn = make()
    assert fn.id == fid
    assert hashlib.sha512(canonical_json(definition_entry(fn))).hexdigest() == entry
    assert (program_digest(Program(fn)) if prog is not None else None) == prog

