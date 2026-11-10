"""`call_scope.specialization`: a spec id's family bound to its own record, a function-valued static resolved to the Definition it
names, a specialized one (`boolean_fp8_moe.MoeExpertFp8ReadGemm(W)_v1`'s ``READ``) included."""
from __future__ import annotations

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import CompositeDefinition, bind
from verity.definitions.ops.boolean import gemm as BG
from verity.commitments.gates import embedding as EMB
from verity.definitions.call_scope import spec_base, spec_statics, specialization
from verity.definitions.registry import boolean_fp8_moe as BM


def test_a_specialized_definition_valued_static_binds_to_its_family_at_its_own_record():
    want = bind(BM.MoeExpertFp8ReadGemmWV1, READ=EMB.read(8, 64), K=64, N=2, G=32, KB=32, SWAP=True, DOT=BG.HopperE4m3QgmmaDot32)
    t = descriptor_id(want)
    assert spec_statics(t)["READ"] == "EmbeddingRead_v1{V=8,H=64}"
    got = specialization(spec_base(t), spec_statics(t))
    assert descriptor_id(bind(got) if isinstance(got, CompositeDefinition) else got) == t
