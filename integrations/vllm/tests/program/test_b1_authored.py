"""The torch-authored B1 composites (`tests/program/b1_authored.py`) against TA1's hand-built definitions
at test scale (a tiny configuration and the B0 head dimension). Digest equality where the encodings are the
same; graph/reference correspondence where the broadcast encoding legitimately differs (§8.5). The full B0/B1
`Serve` rows are `verity_vllm.program.frontend.b1_milestone` (pod). Needs torch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verity.ml.circuits import Program, bind, encode_program, program_digest  # noqa: E402
from verity.ml.circuits.codec import descriptor_size  # noqa: E402
from tests.program import b1_authored as A  # noqa: E402
from verity_vllm.program.frontend import export_report  # noqa: E402
from verity_vllm.program.frontend.correspond import correspond  # noqa: E402
from verity_vllm.program.registry import b1  # noqa: E402

TINY = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64,
        "QKV": 64, "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 8, "BIAS": True, "tie_word_embeddings": True, "profile": {}}
TINY_NOBIAS = {**TINY, "BIAS": False, "KVH": 2, "QKV": 96}

EQUAL = [  # identical encodings expected: digest equality
    ("RMSNormTriton{N=64}", A.RMSNormTriton, b1.RMSNormTriton, {"N": 64, "EPS": 1e-5}),
    ("RMSNormTriton{N=1088}", A.RMSNormTriton, b1.RMSNormTriton, {"N": 1088, "EPS": 1e-6}),  # two chunks, a partial one
    ("RMSNormFusedCuda{N=64}", A.RMSNormFusedCuda, b1.RMSNormFusedCuda, {"N": 64, "EPS": 1e-5}),
    ("RMSNormFusedCuda{N=1536}", A.RMSNormFusedCuda, b1.RMSNormFusedCuda, {"N": 1536, "EPS": 1e-6}),  # threads own 2 elements
    ("AttnBlock{D=16,NVIS=64,FIRST}", A.AttnBlock, b1.AttnBlock, {"D": 16, "NVIS": 64, "FIRST": True}),
    ("AttnBlock{D=64,NVIS=37,not FIRST}", A.AttnBlock, b1.AttnBlock, {"D": 64, "NVIS": 37, "FIRST": False}),
    ("AttnBlock{D=16,NVIS=3,FIRST}", A.AttnBlock, b1.AttnBlock, {"D": 16, "NVIS": 3, "FIRST": True}),
    ("LayerPre{TINY,L0}", A.LayerPre, b1.LayerPre, {"C": TINY, "L0": True}),
    ("LayerPre{TINY,not L0}", A.LayerPre, b1.LayerPre, {"C": TINY, "L0": False}),
    ("LayerPre{TINY no bias}", A.LayerPre, b1.LayerPre, {"C": TINY_NOBIAS, "L0": False}),
    ("LayerPost{TINY}", A.LayerPost, b1.LayerPost, {"C": TINY}),
    ("Final{TINY}", A.Final, b1.Final, {"C": TINY}),
]

CORRESPOND = [  # the zero-row / initial-state broadcasts: TA1 writes `Concat`/`Explicit` runs, the frontend zero-stride `Strided`
    ("AttentionHead{T=70,D=16}", A.AttentionHead, b1.AttentionHead, {"T": 70, "D": 16}),
    ("AttentionHead{T=64,D=16}", A.AttentionHead, b1.AttentionHead, {"T": 64, "D": 16}),  # one full block, no padding but l0/o0
    ("Attention{T=5,KVH=1}", A.Attention, b1.Attention, {"T": 5, "NH": 2, "KVH": 1, "D": 16}),
    ("Attention{T=5,KVH=2}", A.Attention, b1.Attention, {"T": 5, "NH": 4, "KVH": 2, "D": 16}),
    ("Serve{TINY,LP=3,STEPS=2}", A.Serve, b1.Serve, {"C": TINY, "LP": 3, "STEPS": 2}),
]


@pytest.mark.parametrize("label,front,ta1,statics", EQUAL, ids=[e[0] for e in EQUAL])
@pytest.mark.slow
def test_authored_digest_equals_ta1(label, front, ta1, statics):
    rep = export_report(front, static=statics, dead_gates="ignore")
    P_t = Program(bind(ta1, **statics))
    assert rep.gates == P_t.gates
    assert rep.digest == program_digest(P_t), label
    assert encode_program(rep.program)["definitions"] == encode_program(P_t)["definitions"]


@pytest.mark.parametrize("label,front,ta1,statics", CORRESPOND, ids=[e[0] for e in CORRESPOND])
def test_authored_corresponds_to_ta1_across_broadcast_encodings(label, front, ta1, statics):
    rep = export_report(front, static=statics, dead_gates="ignore")
    P_t = Program(bind(ta1, **statics))
    r = correspond(rep.program, P_t)
    assert r.corresponds, (label, [m.to_json() for m in r.mismatches][:3], [u.to_json() for u in r.unverified][:3])
    # DC B.1 made TA1's builder emit the frontend's zero-stride Strided for broadcasts (descriptor-codec.md §6): the two
    # AttentionHead cases and Attention{KVH=2} are now byte-identical; Attention{KVH=1} (the GQA head-sharing view, one
    # kv head read by every q head) and the TINY Serve still correspond through same-references runs (lead, 2026-09-08,
    # measured on the RC pod). Either way the correspondence, gate count and definition set are the claim.
    assert r.root_out == "identical"
    if label in ("Attention{T=5,KVH=1}", "Serve{TINY,LP=3,STEPS=2}"):
        assert not r.digest_equal and r.same_references >= 1
    else:
        assert r.digest_equal
    assert rep.gates == P_t.gates
    assert set(encode_program(rep.program)["definitions"]) == set(encode_program(P_t)["definitions"])
    assert descriptor_size(rep.program) <= descriptor_size(P_t)


def test_serve_tiny_dead_gates_are_reported_and_not_on_s_out():
    rep = export_report(A.Serve, static={"C": TINY, "LP": 3, "STEPS": 2}, dead_gates="ignore", id_limit=10 ** 6)
    assert rep.dead.total == 758 and rep.dead.input == 48 and rep.dead.on_s_out == 0 and rep.dead.ids_complete
    assert rep.unreachable.total == 4987
    assert rep.provenance["mode"] == "authoring" and rep.provenance["registry"]["digest"]
