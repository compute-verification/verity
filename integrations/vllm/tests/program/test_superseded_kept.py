"""The superseded Definition versions that stay registered so old records replay (`tools/migrate/superseded_quarantine.py`'s
`KEPT`, each with what keeps it): each at a small binding keeps its id, its definition entry's digest and its Program's digest.
A changed digest means the version now computes or encodes something else, and the records that name it no longer replay."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

from verity.ml.circuits.codec import canonical_json, definition_entry, program_digest
from verity.ml.circuits.defs import bind
from verity.ml.circuits.program import Program
from verity_vllm.program.registry import b1, prims as P, serve3
from verity_vllm.program.registry import boolean_silu as BS

TINY = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64,
        "QKV": 64, "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 8, "BIAS": True, "tie_word_embeddings": True, "profile": {}}

#: version: (binding, its id, sha512 of its definition entry, its Program's digest or None for a primitive)
KEPT = {
    "RMSNormFusedCuda_v1": (lambda: bind(b1.RMSNormFusedCudaV1, N=64, EPS=1e-5), "RMSNormFusedCuda_v1{N=64,EPS=1e-05}",
                            "723ee5100d3ad9643667c10ed75701cfc74c6a4209465505713003ed9d52190feb5b5b522005aebca48f91a6699f720ed8acd9805b839a8763f4e236084c2dfb",
                            "c89869546e9bf48d3a944556eabaf19a9fb6c35f2c1767fccbd4e51fc41f671548b8a0268f581c77cbf171c918d654ca456da338dee576eb70919de386fb3e38"),
    "SiluMul_v1": (lambda: bind(b1.SiluMul, I=8), "SiluMul_v1{I=8}",
                   "2b6a631db3bcf8fdb50f25f700e7616e45878d18193a02956ed7d1ccb72b955c77b2c80a1efe6b392975a57e4890b1bbfb91a17174681bf3752ae89e0d9a0629",
                   "6a53a54e6ebf11ddb540487fff7627ffa29185781dc2409728dd17404fa0714a62f8cc66ad379afebcfff686a926e83ef83d857141a33c672c38d055e5dbbff8"),
    "SiluMulBf16_v1": (lambda: P.SiluMulBf16, "SiluMulBf16_v1",
                       "42c178e6c08a3af7f19df304e7508c39f54b3d7664e3535a105fae5f9734272aff252abc27c6fb17f1b77286fac2c553c5cfce65f898969bad198680d3c1f0ef", None),
    "SiluMul_v3": (lambda: bind(BS.SiluMul, I=8), "SiluMul_v3{I=8}",
                   "d2bc7966067f52e23c2ef22b0a84dbb489b0815a6203c1dd91742a83c6b7837f959ff6d9e6708379fa77d39b600c4e8a17a098d79224d96e4f6a385a7a41b8b8",
                   "d53cfe4f5ee4151a777b3a7a9ae26dcd1873f3d4ed344f9ffaff7e13cf29194bc1b7dbbfd2bddf9d5221d05f6d064ab244bc918a9c2858da1fbd199abee87536"),
    "SiluMulBf16_v3": (lambda: bind(BS.SiluMulBf16), "SiluMulBf16_v3",
                       "618c3ebb4c96158c3c08b8355c5c3c1e84699720d770cd0b7a67de1c02e73046d36fa0f91e42f6c630f8ed79a35eea0a796b4c630a47c37f9c613b7fd6cc703e",
                       "3bb873ca3d8e742ca7a40364a5ce05e1ec11dd94c9059d3c995c97f662f790294c04edbe385b5de5486d68a35b7136a7f3f889e7974c178cc411f267a6ba8255"),
    "Attention_v1": (lambda: bind(b1.Attention, T=3, NH=2, KVH=1, D=16), "Attention_v1{T=3,NH=2,KVH=1,D=16}",
                     "281d195af5bb7756a896b56971cdc9daf08d5dae91becd00cbe2c2d5359cd9fed7361920021cefc3a21fe894c41e0546b6dec5959e5d8f54cf159edb05e048b2",
                     "859cc88da075d4ead8dd0737b82a5ff771aa218583f38209821f1834a773be2cf475b8d25b0e669c91384d1e6cb258234a9669d9ade0fda284aa2773aafdef2a"),
    "AttentionHead_v1": (lambda: bind(b1.AttentionHead, T=3, D=16), "AttentionHead_v1{T=3,D=16}",
                         "cbcdc5435770266bf77eff172177589bb389e7c4b18a25dfcfca4f92a070055f92b237551c29476542bc709b9652f7821011d35e5cab788a8fe1028917dd5abf",
                         "a2a5674b612b6b3d2a0e572f899f0df586239af92df752aca74f01c4bd9da4a0d6b36774578e2805548cb4c22a743f7e0cb092380e399a31fc2bda02751e5ba4"),
    "AttnBlock_v1": (lambda: bind(b1.AttnBlock, D=16, NVIS=3, FIRST=True), "AttnBlock_v1{D=16,NVIS=3,FIRST=True}",
                     "af3d05441b14d4ad3f1b3ced84b229d99305606425306e3bc15e92b436095c48e6a93d728ef8de1d523f1fc5f485720aae47168ff91621f8fa3a4185b9fc1a10",
                     "afbf1956bc4c8f1d16838fe4ecebdd56d9bbbf0be75eebb60ef29a66b81c59d456507cca2ecd88fbc844d84e188e7360e6cf66b013dfff9d1ac2453b9272b781"),
    "Serve_v1": (lambda: bind(b1.Serve, C=TINY, LP=3, STEPS=2), "Serve_v1{C={e2f2f43eb046},LP=3,STEPS=2}",
                 "1adfd4b4963f56ad20f62090cafb82f43f05ee8204e467a35e79e0866d599552f0e742b6e7591d9af3fe50ab443b2f58d4e6dd078f2a5b64467772b92c3d91b5",
                 "228daae01285d688d8fd8e0c8df72b24efaffc55bd9c5df48d71120d7e059d506720058dd64b2d881f4763d670c23a18f6a3201221ba9f45662b3c9a6bec4af3"),
    "Serve_v2": (lambda: b1.serve_v2(TINY, 3, 2),
                 "Serve_v2{C={e2f2f43eb046},LP=3,STEPS=2,GEMM={4130e3ca726d},ATTENTION={35db6738c571},NORM={4d524c30089f}}",
                 "1a12b2c3d9943948127220e1f2a0cd86d69acb327b58837a61a166acbe717802c3cf6950ce629c882fd0be255e7096c4e1ce06f11afe5823e2cd06c78f57bdad",
                 "354b54347ccfdc2149683b0582a33e510ba3418a6b00c38790fc18ee86292cd76de064097ecbc5c40136dea4903683df6d703b3658465d4646803a52ff797cbb"),
    "Serve_v3": (lambda: serve3.serve_v3({**TINY, "EOS_ID": 7}, 3, 2),
                 "Serve_v3{C={fa22b0376a3a},LP=3,D_MAX=2,GEMM={4130e3ca726d},ATTENTION={35db6738c571},NORM={4d524c30089f}}",
                 "4bd455bd1ece645d5201fea695025ec50001a8c115df1834754a95019bb368b397643355e0435c68bda690d5fca4ee9051fbbbf50ebfd9cf03d5d44c366dec6c",
                 "84da93254a6c7af097af4be6ff379bd135aeb4cdfb646f0dc393cd7d0f03cb7a074dfa5e42bcd176268bd459ea869e168f32ceddcf9507c29e04a2cf24dcd416"),
    "LayerPre_v1": (lambda: bind(b1.LayerPre, C=TINY, L0=True), "LayerPre_v1{C={e2f2f43eb046},L0=True}",
                    "04fe569c68f7f7c50f3747f933716df6db693fa485f3be3fdd7b3be039c53bd8909dbaca70f4e36e513d752dfc3d0b7defeb352a58f5d379b14fa1c337f6a418",
                    "354c79c8a05916314a61bc07c7c5f22ca2e9c9ba4981d65b283a5fe00f814e766481a5fbbdf87b783d1a70c441a664de1fffe771474677e69c4a44db26e4f782"),
    "LayerPost_v1": (lambda: bind(b1.LayerPost, C=TINY), "LayerPost_v1{C={e2f2f43eb046}}",
                     "8dd07e8362083203e50af31d4c6b0cc17ca4417df6b4713fd47b0fa7e307eaab5261f628d1bd0ee6ae9cc71f53fd6fd4192874c82b8cb9fcbb1cad18674f0545",
                     "2a1b4767164f742ae0b23b1c9592d332bc4d937ed0d5acbe52b512eb95a984c886b7122a190296c1e4eeab12f0f8fe35824db69756d848144164cf6012a77138"),
    "Final_v1": (lambda: bind(b1.Final, C=TINY), "Final_v1{C={e2f2f43eb046}}",
                 "f32e9bd0842c07d8a17561ee22f6a1fda708a79ff490a4792d7eb6d12c1ad4ff920d41b4c5c9613ec146f31d0023c40fa05a0add3d5fdfc0370783f8251a82c5",
                 "95d0d960aaba214fb80b654aeb56c0302e99028a9bc5a9f93db0cbb03b7d211991659459d6ebf3d2ae5fc80d22603a76a0506efb20bc6a648711d5bb2f5ec57e"),
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
