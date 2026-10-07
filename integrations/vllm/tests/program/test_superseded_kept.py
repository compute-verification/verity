"""The superseded Definition versions that stay registered so old records replay (`tools/migrate/superseded_quarantine.py`'s
`KEPT`, each with what keeps it): each at a small binding keeps its id, its definition entry's digest and its Program's digest.
A changed digest means the version now computes or encodes something else, and the records that name it no longer replay."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

from verity.primitives.circuits.codec import canonical_json, definition_entry, program_digest
from verity.primitives.circuits.defs import bind
from verity.primitives.circuits.program import Program
from verity_vllm.program.registry import b1, prims as P, serve3
from verity_vllm.program.registry import boolean_silu as BS

TINY = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64,
        "QKV": 64, "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 8, "BIAS": True, "tie_word_embeddings": True, "profile": {}}

#: version: (binding, its id, sha512 of its definition entry, its Program's digest or None for a primitive)
KEPT = {
    "RMSNormFusedCuda_v1": (lambda: bind(b1.RMSNormFusedCudaV1, N=64, EPS=1e-5), "RMSNormFusedCuda_v1{N=64,EPS=1e-05}",
                            "723ee5100d3ad9643667c10ed75701cfc74c6a4209465505713003ed9d52190feb5b5b522005aebca48f91a6699f720ed8acd9805b839a8763f4e236084c2dfb",
                            "814972001da1a880887c7693093a958057cc6d068b35830ed7c9624d9669320a"),
    "SiluMul_v1": (lambda: bind(b1.SiluMul, I=8), "SiluMul_v1{I=8}",
                   "2b6a631db3bcf8fdb50f25f700e7616e45878d18193a02956ed7d1ccb72b955c77b2c80a1efe6b392975a57e4890b1bbfb91a17174681bf3752ae89e0d9a0629",
                   "c7ffc775940aa9bd5d316b05e5dc86552d9ff7f9803a980c73787e41e150c67b"),
    "SiluMulBf16_v1": (lambda: P.SiluMulBf16, "SiluMulBf16_v1",
                       "42c178e6c08a3af7f19df304e7508c39f54b3d7664e3535a105fae5f9734272aff252abc27c6fb17f1b77286fac2c553c5cfce65f898969bad198680d3c1f0ef", None),
    "SiluMul_v3": (lambda: bind(BS.SiluMul, I=8), "SiluMul_v3{I=8}",
                   "d2bc7966067f52e23c2ef22b0a84dbb489b0815a6203c1dd91742a83c6b7837f959ff6d9e6708379fa77d39b600c4e8a17a098d79224d96e4f6a385a7a41b8b8",
                   "d8cadfc24d8827baaa050e6c3644c344a8702bf34e4ce354b2abf277d389e130"),
    "SiluMulBf16_v3": (lambda: bind(BS.SiluMulBf16), "SiluMulBf16_v3",
                       "618c3ebb4c96158c3c08b8355c5c3c1e84699720d770cd0b7a67de1c02e73046d36fa0f91e42f6c630f8ed79a35eea0a796b4c630a47c37f9c613b7fd6cc703e",
                       "67dcc25469da1ae006bbeaec5c962536ac252162ddeaef70ad1bc0adc8efdcad"),
    "Attention_v1": (lambda: bind(b1.Attention, T=3, NH=2, KVH=1, D=16), "Attention_v1{T=3,NH=2,KVH=1,D=16}",
                     "281d195af5bb7756a896b56971cdc9daf08d5dae91becd00cbe2c2d5359cd9fed7361920021cefc3a21fe894c41e0546b6dec5959e5d8f54cf159edb05e048b2",
                     "b6a6af59ee12949310a86acee4bbb36add02164e34f33c91b91a65ee3c73927e"),
    "AttentionHead_v1": (lambda: bind(b1.AttentionHead, T=3, D=16), "AttentionHead_v1{T=3,D=16}",
                         "cbcdc5435770266bf77eff172177589bb389e7c4b18a25dfcfca4f92a070055f92b237551c29476542bc709b9652f7821011d35e5cab788a8fe1028917dd5abf",
                         "a80cdc28b8ea6fcd3d02ba98511fedb7b80e4c459441bc6a89865b3eb67f9ee4"),
    "AttnBlock_v1": (lambda: bind(b1.AttnBlock, D=16, NVIS=3, FIRST=True), "AttnBlock_v1{D=16,NVIS=3,FIRST=True}",
                     "40776b615d1c1cb15980075e3994ccf561383b12e62cbe65ee21b6a0cd3c2e0a",
                     "5cb33c281d41f1928eb5b781433b32959a8367fe112c3317cb9b0354c23a97b8"),
    "Serve_v1": (lambda: bind(b1.Serve, C=TINY, LP=3, STEPS=2), "Serve_v1{C={e2f2f43eb046},LP=3,STEPS=2}",
                 "7dfcbe536c47809f3eec22b0469bfa7abfbe0f409be6fb43f2f25d1ebec10018",
                 "be1b90b83b5c47eb5962ae709bdacd22cf5f36fada54000022dc1b52a431852c"),
    "Serve_v2": (lambda: b1.serve_v2(TINY, 3, 2),
                 "Serve_v2{C={e2f2f43eb046},LP=3,STEPS=2,GEMM={4130e3ca726d},ATTENTION={35db6738c571},NORM={4d524c30089f}}",
                 "1b35a5a3c11b0e667429a340e51a0aea0234901eeb225d6f975e13723e2f7c3f",
                 "361800f6f8004d9397a04b3c6996352a1240fe38c9383ca1b7b6c37f72b6079c"),
    "Serve_v3": (lambda: serve3.serve_v3({**TINY, "EOS_ID": 7}, 3, 2),
                 "Serve_v3{C={fa22b0376a3a},LP=3,D_MAX=2,GEMM={4130e3ca726d},ATTENTION={35db6738c571},NORM={4d524c30089f}}",
                 "51de5c8127fca745dfedcca8eba7dee8e1d649ad1925aee48e49e81a887cc130",
                 "5b76ff21ed540885f73c49e7f5efc1fe4f306414b1a8857f93306e67d5e79847"),
    "LayerPre_v1": (lambda: bind(b1.LayerPre, C=TINY, L0=True), "LayerPre_v1{C={e2f2f43eb046},L0=True}",
                    "04fe569c68f7f7c50f3747f933716df6db693fa485f3be3fdd7b3be039c53bd8909dbaca70f4e36e513d752dfc3d0b7defeb352a58f5d379b14fa1c337f6a418",
                    "5d7b0a740fcc8443933949ca2e3f67d5d0a7ec845e18d3ee733a618bfb84f2df"),
    "LayerPost_v1": (lambda: bind(b1.LayerPost, C=TINY), "LayerPost_v1{C={e2f2f43eb046}}",
                     "8dd07e8362083203e50af31d4c6b0cc17ca4417df6b4713fd47b0fa7e307eaab5261f628d1bd0ee6ae9cc71f53fd6fd4192874c82b8cb9fcbb1cad18674f0545",
                     "2581028e1247e29c9c4a9402d06784655727ee75e91954acc7d77d6e55a4c959"),
    "Final_v1": (lambda: bind(b1.Final, C=TINY), "Final_v1{C={e2f2f43eb046}}",
                 "f32e9bd0842c07d8a17561ee22f6a1fda708a79ff490a4792d7eb6d12c1ad4ff920d41b4c5c9613ec146f31d0023c40fa05a0add3d5fdfc0370783f8251a82c5",
                 "7bd0fc25bd4caf590c56939b59e195f4ceda277f448ba536ebe2540c0101ae0e"),
}


@pytest.mark.parametrize("version", sorted(KEPT))
def test_a_kept_superseded_version_keeps_its_identity(version):
    make, fid, entry, prog = KEPT[version]
    fn = make()
    assert fn.id == fid
    assert hashlib.sha512(canonical_json(definition_entry(fn))).hexdigest() == entry
    assert (program_digest(Program(fn)) if prog is not None else None) == prog


def test_the_pinned_versions_are_the_migration_script_s_kept_list():
    script = Path(__file__).resolve().parents[2] / "tools" / "migrate" / "superseded_quarantine.py"
    spec = importlib.util.spec_from_file_location("superseded_quarantine_kept", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert set(mod.KEPT) == set(KEPT) and not set(mod.DELETE) & set(KEPT)
