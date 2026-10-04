"""The superseded Definition versions that stay registered so old records replay (`tools/migrate/superseded_quarantine.py`'s
`KEPT`, each with what keeps it): each at a small binding keeps its id, its definition entry's digest and its Program's digest.
A changed digest means the version now computes or encodes something else, and the records that name it no longer replay."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

from verity.ir.codec import canonical_json, definition_entry, program_digest
from verity.ir.defs import bind
from verity.ir.program import Program
from verity_vllm.program.registry import b1, prims as P, serve3
from verity_vllm.program.registry import boolean_silu as BS

TINY = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64,
        "QKV": 64, "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 8, "BIAS": True, "tie_word_embeddings": True, "profile": {}}

#: version: (binding, its id, sha256 of its definition entry, its Program's digest or None for a primitive)
KEPT = {
    "RMSNormFusedCuda_v1": (lambda: bind(b1.RMSNormFusedCudaV1, N=64, EPS=1e-5), "RMSNormFusedCuda_v1{N=64,EPS=1e-05}",
                            "67ac99f54ae3df23cc1103eb752e5b1a63effadb990d134b2d7e521e939399b8",
                            "814972001da1a880887c7693093a958057cc6d068b35830ed7c9624d9669320a"),
    "SiluMul_v1": (lambda: bind(b1.SiluMul, I=8), "SiluMul_v1{I=8}",
                   "c169850ec136ae85289bcb8cf374e1b29650faa6ef75eb5b1f06003ee9893c32",
                   "c7ffc775940aa9bd5d316b05e5dc86552d9ff7f9803a980c73787e41e150c67b"),
    "SiluMulBf16_v1": (lambda: P.SiluMulBf16, "SiluMulBf16_v1",
                       "dc36f7e20ee20a889b7cfeaa7b9f3a04128c2eafbe6abec245476606083d2f3c", None),
    "SiluMul_v3": (lambda: bind(BS.SiluMul, I=8), "SiluMul_v3{I=8}",
                   "b0a848b9543a6cabe91c635edda0b75f90ae45f15126a5af66bb55a143e7f878",
                   "d8cadfc24d8827baaa050e6c3644c344a8702bf34e4ce354b2abf277d389e130"),
    "SiluMulBf16_v3": (lambda: bind(BS.SiluMulBf16), "SiluMulBf16_v3",
                       "509a6221ffab1e5971e7f9c34d2b56b727085d9216a5ef969f4b041d83088ce9",
                       "67dcc25469da1ae006bbeaec5c962536ac252162ddeaef70ad1bc0adc8efdcad"),
    "Attention_v1": (lambda: bind(b1.Attention, T=3, NH=2, KVH=1, D=16), "Attention_v1{T=3,NH=2,KVH=1,D=16}",
                     "91c9a57f411eca3e8e3b3715418cd75a3eb47549e1fd142a1c925d4cbdbddac5",
                     "b6a6af59ee12949310a86acee4bbb36add02164e34f33c91b91a65ee3c73927e"),
    "AttentionHead_v1": (lambda: bind(b1.AttentionHead, T=3, D=16), "AttentionHead_v1{T=3,D=16}",
                         "a1d27a7952a13cffa60e875324669376ce617054a762f5c0c22d50bef9793303",
                         "a80cdc28b8ea6fcd3d02ba98511fedb7b80e4c459441bc6a89865b3eb67f9ee4"),
    "AttnBlock_v1": (lambda: bind(b1.AttnBlock, D=16, NVIS=3, FIRST=True), "AttnBlock_v1{D=16,NVIS=3,FIRST=True}",
                     "40776b615d1c1cb15980075e3994ccf561383b12e62cbe65ee21b6a0cd3c2e0a",
                     "5cb33c281d41f1928eb5b781433b32959a8367fe112c3317cb9b0354c23a97b8"),
    "Serve_v1": (lambda: bind(b1.Serve, C=TINY, LP=3, STEPS=2), "Serve_v1{C={d0dc81b7fa41},LP=3,STEPS=2}",
                 "7dfcbe536c47809f3eec22b0469bfa7abfbe0f409be6fb43f2f25d1ebec10018",
                 "be1b90b83b5c47eb5962ae709bdacd22cf5f36fada54000022dc1b52a431852c"),
    "Serve_v2": (lambda: b1.serve_v2(TINY, 3, 2),
                 "Serve_v2{C={d0dc81b7fa41},LP=3,STEPS=2,GEMM={1af1018d695c},ATTENTION={f63c865fa1fc},NORM={c0107547e710}}",
                 "1b35a5a3c11b0e667429a340e51a0aea0234901eeb225d6f975e13723e2f7c3f",
                 "361800f6f8004d9397a04b3c6996352a1240fe38c9383ca1b7b6c37f72b6079c"),
    "Serve_v3": (lambda: serve3.serve_v3({**TINY, "EOS_ID": 7}, 3, 2),
                 "Serve_v3{C={25679f5c0a60},LP=3,D_MAX=2,GEMM={1af1018d695c},ATTENTION={f63c865fa1fc},NORM={c0107547e710}}",
                 "51de5c8127fca745dfedcca8eba7dee8e1d649ad1925aee48e49e81a887cc130",
                 "5b76ff21ed540885f73c49e7f5efc1fe4f306414b1a8857f93306e67d5e79847"),
    "LayerPre_v1": (lambda: bind(b1.LayerPre, C=TINY, L0=True), "LayerPre_v1{C={d0dc81b7fa41},L0=True}",
                    "7e003f5aee7f844b5187f34c3f60d6e4a11acb858f05d5c66c8e5428ad32bb11",
                    "5d7b0a740fcc8443933949ca2e3f67d5d0a7ec845e18d3ee733a618bfb84f2df"),
    "LayerPost_v1": (lambda: bind(b1.LayerPost, C=TINY), "LayerPost_v1{C={d0dc81b7fa41}}",
                     "9cb638dfd7d544e4385a5dabc57391c85aaa4d98793a4271e78b35af9c84bb4b",
                     "2581028e1247e29c9c4a9402d06784655727ee75e91954acc7d77d6e55a4c959"),
    "Final_v1": (lambda: bind(b1.Final, C=TINY), "Final_v1{C={d0dc81b7fa41}}",
                 "966b992b9c0bf2ad687edbd52d1dc9932e065995820f48d1931925c831104ee5",
                 "7bd0fc25bd4caf590c56939b59e195f4ceda277f448ba536ebe2540c0101ae0e"),
}


@pytest.mark.parametrize("version", sorted(KEPT))
def test_a_kept_superseded_version_keeps_its_identity(version):
    make, fid, entry, prog = KEPT[version]
    fn = make()
    assert fn.id == fid
    assert hashlib.sha256(canonical_json(definition_entry(fn))).hexdigest() == entry
    assert (program_digest(Program(fn)) if prog is not None else None) == prog


def test_the_pinned_versions_are_the_migration_script_s_kept_list():
    script = Path(__file__).resolve().parents[2] / "tools" / "migrate" / "superseded_quarantine.py"
    spec = importlib.util.spec_from_file_location("superseded_quarantine_kept", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert set(mod.KEPT) == set(KEPT) and not set(mod.DELETE) & set(KEPT)
