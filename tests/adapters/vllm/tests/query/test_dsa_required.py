"""The request manifest of GLM-5.3's attention (MLA + the DSA indexer) on the Program the Build derives (`tests/program/test_dsa_derive.py`'s
indexer and IndexShare layers over a prefill of two and a decode), its modules relabelled to GLM's `model.layers.<i>.self_attn` and
each step closed by an `o_proj` reading the W_UV product and a sampling event:

* UNRETURNED: the indexer's MQA logits and top-k, which the correspondence lowers to no runtime tensor and no Call reads, are not
  required, so the prefill's logits rows of N = 1 and N = 2 keys never meet in one manifest group and the manifest is complete;
  q_c, which no Call of this Program reads either, keeps its rows (its fx node returns it);
* every required attention Value binds from what the capture map's attention binder commits (`verity_vllm.query.manifest.bind`): the
  module's own outputs by their label, the indexer's ports and the two call boundaries (the W_UK_T absorb and the sparse MLA) by the
  ports the source declares."""
from __future__ import annotations

import copy

import pytest
from verity_torch.frontend.emit import NAMESPACE

from tests.acquire.glm53_model import CMAP, ENV
from tests.program import test_dsa_derive as T
from verity_vllm.commit import capture_map as CM
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.definitions.registry import hopper
from verity_vllm.program.frontend.rules import vllm_bindings as VB
from verity_vllm.program.frontend.rules import vllm_dsa
from verity_vllm.program.frontend.rules.vocab import PROFILES
from verity_vllm.query.manifest import format as MF
from verity_vllm.query.program_view import CallView, Port, ProgramView, from_instances
from verity_vllm.query.required import Population, request_manifest
from verity_vllm.query.units import ValueRef

INDEXER_LAYER = min(CM.layer_set(CMAP, "indexer", ENV))
SHARED_LAYER = min(i for i in CM.layer_set(CMAP, "shared_topk", ENV) if i > INDEXER_LAYER)
MODULE = {"layers.0": f"model.layers.{INDEXER_LAYER}.self_attn", "layers.1": f"model.layers.{SHARED_LAYER}.self_attn"}
ATTENTION = {"q_c", "kv_c", "k_pe", "q_pe", "out"}
#: the indexer's DsaIndexerKRow_v1 / DsaIndexerQ_v1, both two ports, under the Build's hints
INDEX = {"index_k.0", "index_k.1", "index_q.0", "index_q.1"}
#: the Calls of the attention body whose outputs only another Call of it reads: the W_UK_T absorb and the sparse MLA
CALL_BOUNDARIES = {"BatchedGemm_v1", "SparseMla_v1"}
RES = {"program": {"digest": "d0", "function": "f"}, "target": {"declared": {"compute_capability": [9, 0]}},
       "attention_impls_observed": {f"{m}.attn": {"kind": "mla", "impl": T.IMPL} for m in MODULE.values()}}


def _derived():
    from verity_vllm.pipeline.build_instances import _instance_rows

    with pytest.MonkeyPatch.context() as mp:
        saved = dict(VB._OBSERVED)
        mp.setitem(PROFILES, T.STANDIN.name, T.STANDIN)
        mp.setitem(vllm_dsa.FA3_SPARSE_GEOMETRY, (T.R, T.L), {"BN": T.BN, "DOT": T.DOT, "INV": hopper.Fa3InvSum})
        VB.observe_target_profile(T.H200)
        VB.observe_attention_impls({n: T._observation() for n in T.NAMES})
        try:
            model = T._model()
            rep = T._derive(model, T._inputs(model))
        finally:
            VB._OBSERVED.clear()
            VB._OBSERVED.update(saved)
    assert rep.program is not None, rep.unsupported
    prog = rep.program
    doc = {"schema": "cb-a/instance-sequence/v2", "digest": "d", "params": [[n, str(t)] for n, t in prog.fn.body.params],
           "rows": list(_instance_rows(prog))}
    return from_instances(doc), copy.deepcopy(prog.annotations[NAMESPACE])


def _served(P0: ProgramView, doc: dict) -> tuple[ProgramView, dict]:
    """`P0` with GLM's module paths, an `o_proj` Call (module `<attention>.o_proj`) reading each W_UV product and, after each step, an
    `ArgmaxTokenSelect_v1` reading the step's last `o_proj` row; the correspondence document gains a record per added Call."""
    for occ in doc["occurrences"]:
        occ["module"] = MODULE.get(occ.get("module"), occ.get("module"))
    corr0 = Correspondence(P0, document=doc)
    sparse = {c.id for c in P0.calls if c.family == "SparseMla_v1"}
    w_uv = [c.id for c in P0.calls if c.family == "BatchedGemm_v1" and any(r.call in sparse for r in c.reads)]
    step_nodes = len(MODULE)
    second = next(c.id for c in P0.calls if corr0.name(c.id).node == f"dsa_norm_rope_{step_nodes}")
    calls: list[CallView] = []
    records, new_id, last = [], {}, None

    def add(definition: str, name: str, module: str, reads: tuple, elems: int, width: int) -> int:
        cid = len(calls)
        calls.append(CallView(cid, definition, definition.split("{")[0], "call", name, (Port("out", 0, elems, elems, width),), reads))
        doc["occurrences"].append({"module": module, "node": name.split("/")[-1].split("[")[0], "outputs": [{"shape": [elems], "dtype": "bfloat16"}], "inputs": []})
        records.append({"occ": len(doc["occurrences"]) - 1, "member": None, "row": None, "hint": None,
                        "returns": [{"port": 0, "output": 0, "leaves": [0, elems]}], "args": []})
        return cid

    def close_step() -> None:
        add("ArgmaxTokenSelect_v1{N=1}", "argmax", "", (ValueRef(last, "out"),), 1, 32)

    for c in P0.calls:
        if c.id == second:
            close_step()
        new_id[c.id] = len(calls)
        calls.append(CallView(len(calls), c.definition, c.family, c.form, c.name, c.port_table,
                              tuple(ValueRef(new_id[r.call], r.port) if r.call >= 0 else r for r in c.reads)))
        records.append(doc["calls"][c.id])
        if c.id in w_uv:
            m = corr0.module_of(c.id)
            last = add("Gemm_v1{N=4}", f"{m}.o_proj/linear[0]", f"{m}.o_proj", (ValueRef(new_id[c.id], "out"),), 4, 16)
    close_step()
    doc["calls"] = records
    return ProgramView(tuple(calls), P0.params, (ValueRef(len(calls) - 1, "out"),), "d", "test"), doc


@pytest.fixture(scope="module")
def glm():
    P, doc = _served(*_derived())
    corr = Correspondence(P, document=doc)
    return P, corr, request_manifest(P, corr, RES, request_id="r0").manifest


def _unread_stand_ins(P: ProgramView) -> set[int]:
    """The first layer's stand-in o_proj rows: each step's argmax reads the last layer's (TENSOR-RETURN keeps every row of that node),
    so no Call reads these and no policy requires them, and they are unreturned too."""
    return {c.id for c in P.calls if c.name.startswith(f"{MODULE['layers.0']}.o_proj/")}








def test_without_the_runtime_record_the_logits_stay_required():
    """The fallback correspondence records no return slots: only the unread stand-ins are unreturned, and the hinted logits stay
    return slots."""
    P, _doc = _served(*_derived())
    pop = Population(P, Correspondence(P))
    assert pop.unreturned == _unread_stand_ins(P)
    assert {c.id for c in P.calls if c.family == "DsaMqaLogits_v3"} <= {v.call for v in pop.required.values}






def test_the_wrapper_prefix_of_glm53_rows_is_determined_from_the_attention_paths_parent():
    """GLM-5.3's MLA rows sit under `inner.model.model.layers.N.self_attn` while the implementation is `model.layers.N.self_attn.attn`
    (result.json): the prefix is `inner.model.`, not the `model.` default; a Llama-style Attention row keeps its answer."""
    glm = [{"spec": "EmbeddingShard_v1{V=8,H=4}", "name": "inner.model.model.embed_tokens/embedding[0]"},
           {"spec": "RoPEHeadGptJ_v2{D=64}", "name": "inner.model.model.layers.0.self_attn/dsa_norm_rope/k_pe[0]"}]
    assert MF.wrapper_prefix_of(glm, ["model.layers.0.self_attn.attn"]) == "inner.model."
    llama = [{"spec": "Attention_v3{T=1,NH=2,KVH=1,D=4,BN=16}", "name": "model.model.layers.0.self_attn.attn/attn[t=0]"}]
    assert MF.wrapper_prefix_of(llama, ["model.layers.0.self_attn.attn"]) == "model."
    assert MF.wrapper_prefix_of(llama, []) == "model." and MF.wrapper_prefix_of([], ["model.layers.0.self_attn.attn"]) == "model."
