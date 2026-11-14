"""`call_scope.specialization`: a spec id's family bound to its own record, a function-valued static resolved to the Definition it
names, a specialized one (`boolean_fp8_moe.MoeExpertFp8ReadGemm(W)_v1`'s ``READ``) included; the family is the one the registry
it is handed holds, and it loads no catalog."""
from __future__ import annotations

import pytest

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.ops.boolean import gemm as BG
from verity.commitments.gates import embedding as EMB
from verity.circuits.call_scope import spec_base, spec_statics, specialization
from verity.definitions.registry.catalog import load
from verity.definitions.registry import boolean_fp8_moe as BM


def test_a_specialized_definition_valued_static_binds_to_its_family_at_its_own_record():
    want = bind(BM.MoeExpertFp8ReadGemmWV1, READ=EMB.read(8, 64), K=64, N=2, G=32, KB=32, SWAP=True, DOT=BG.HopperE4m3QgmmaDot32)
    t = descriptor_id(want)
    assert spec_statics(t)["READ"] == "EmbeddingRead_v1{V=8,H=64}"
    got = specialization(spec_base(t), spec_statics(t), load()[0])
    assert descriptor_id(bind(got) if isinstance(got, CompositeDefinition) else got) == t


def test_the_family_is_the_one_the_handed_registry_holds():
    from verity.circuits.defs import REGISTRY, DefinitionTable
    from verity.circuits import call_scope as CS
    from verity.definitions.registry import b1  # noqa: F401  (registers Gemm_v1)

    reg = DefinitionTable()
    reg.add(REGISTRY.defs["Gemm_v1"])
    assert CS.specialization("Gemm_v1", {"K": 16, "N": 1}, reg).id == "Gemm_v1{K=16,N=1}"
    with pytest.raises(KeyError, match="Gemm_v1"):
        CS.specialization("Gemm_v1", {"K": 16, "N": 1}, DefinitionTable())
