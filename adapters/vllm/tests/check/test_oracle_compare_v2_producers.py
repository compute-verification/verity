"""Form (B) over a `v2-query` manifest: the oracle's producer facts come from the Programs of record by dataflow.

The gemma2-2b L40S b8 Commit left 45,360 `instance_outputs` not compared under a v2 manifest, whose rows
carry none of the v1 producer annotations: 44,928 fused-norm narrowings ("ambiguous producer: 2 instances of NarrowF32ToBf16_v1 per
row") and 432 `model/out` ("no fold instance of this op at this step": the embed scale is hinted `model.normalizer` by the fold).
`producers_of_programs` reads, per required Value, the producing family, the family its first operand reads and the modules reading it
outside its own module, under the population the manifest was built with; `merge_producers` adds them where the manifest row declares
no selector; `member_bytes` resolves a member through the Value when the fold has no instance of its family under the identity's path.
"""
from __future__ import annotations

import json
import types

import pytest

from verity_vllm.query.units import ValueRef as R
from verity_vllm.check import oracle_compare as OC
from verity_vllm.check.replay.coverage import form_b_families
from verity_vllm.query.manifest import format as MF
from tests.check.test_oracle_compare import ROWS, U0, U1, H, _exploded_norm_match, make_match, reader
from tests.check.test_oracle_compare_promoted_key import CONSUMER_OP, PRODUCER_OP, _add_promoted, _promoted_bmap
from verity_vllm.query import program_view as pv
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.query.program_view import CallView, InputView, Port, ProgramView
from verity_vllm.query.required import request_manifest
from verity_vllm.records.fields import MissingField

NORM, MLP, LAYER = "model.layers.0.input_layernorm", "model.layers.0.mlp", "model.layers.0"
EMBED_W, NORM_W, MLP_W = "model.embed_tokens.weight", f"{NORM}.weight", f"{MLP}.weight"
RES = {"program": {"digest": "d0", "function": "f"}, "target": {"declared": {"compute_capability": [8, 9]}}}


def _call(cid, spec, name, reads, *, n=4, width=16):
    return CallView(cid, spec, pv.spec_base(spec), "call", name, (Port("out", 0, n, n, width),), tuple(reads))


def gemma_like_program(*, residual_operand: int = 2) -> ProgramView:
    """One step of Gemma2's shape under the v1-annotation fallback: the embed scale is the model body's own Call (read by layer 0's
    fused norm), the fused norm narrows the residual (member 1, from AddWidenedBf16) and the normed row (member 0, from MulVecF32),
    the MLP reads member 0, the decoder layer's body reads member 1.  `residual_operand`: the Call member 1's narrowing reads."""
    params = (InputView(-1, "weights", "{...}", (Port(EMBED_W, 0, 8, 8, 16), Port(NORM_W, 8, 12, 4, 16), Port(MLP_W, 12, 28, 16, 16))),
              InputView(-2, "ids", "Value<32>", (Port("out", 0, 1, 1, 32),)))
    calls = (
        _call(0, "Embedding_v1{V=2,H=4}", "model.embed_tokens/embedding[0]", [R(-2, "out"), R(-1, EMBED_W)]),
        _call(1, "Bf16MulScalarTensor_v1{N=4}", "model/mul[0]", [R(0, "out")]),
        _call(2, "AddWidenedBf16_v1{N=4}", f"{NORM}/add[0]", [R(1, "out")], width=32),
        _call(3, "NarrowF32ToBf16_v1{N=4}", f"{NORM}/narrow/1[0]", [R(residual_operand, "out")]),
        _call(4, "MulVecF32_v1{N=4}", f"{NORM}/mul[0]", [R(2, "out"), R(-1, NORM_W)], width=32),
        _call(5, "NarrowF32ToBf16_v1{N=4}", f"{NORM}/narrow_1/0[0]", [R(4, "out")]),
        _call(6, "Gemm_v1{K=4,N=4}", f"{MLP}/linear[0]", [R(5, "out"), R(-1, MLP_W)]),
        _call(7, "AddBf16_v1{N=4}", f"{LAYER}/add[0]", [R(6, "out"), R(3, "out")]),
        _call(8, "TokenSelect_v1{V=4}", "sampler[0]", [R(7, "out")], n=1, width=32),
    )
    return ProgramView(calls, params, (R(8, "out"),), "synthetic", "synthetic")


def test_program_facts_name_the_selector_and_the_consumers_by_dataflow():
    P = gemma_like_program()
    corr = Correspondence(P)
    facts = OC.program_producer_facts(P, corr)
    assert facts[("model", "out")] == {"spec": "Bf16MulScalarTensor_v1", "operand": "Embedding_v1", "consumers": [NORM]}
    assert facts[(NORM, "1")] == {"spec": "NarrowF32ToBf16_v1", "operand": "AddWidenedBf16_v1", "consumers": [LAYER]}
    assert facts[(NORM, "0")] == {"spec": "NarrowF32ToBf16_v1", "operand": "MulVecF32_v1", "consumers": [MLP]}
    # the facts are keyed as the manifest built from the same Program names its instance outputs (one population, one spelling)
    man = request_manifest(P, corr, RES, request_id="r0").manifest
    named = {(r["op_path"], r["output_member"]) for r in man["identities"] if r["family"] == "instance_outputs"}
    assert set(facts) == named, (sorted(facts), sorted(named))
    assert all(OC.producers_of(man)[k]["spec"] == facts[k]["spec"] for k in named)


def test_the_commit_derives_the_facts_of_a_manifest_of_either_query_under_that_query():
    """A Q_word manifest (the record since S1) carries no producer annotations either: the Commit derives its facts from the Programs,
    keyed as that manifest names its instance outputs, and the handed-down `model/out` is promoted; a manifest the query path did not
    build keeps its own entries."""
    from verity_vllm.pipeline import commit_plan as C
    from verity_vllm.query.required import QUERIES
    P = gemma_like_program()
    corr = Correspondence(P)
    for query in QUERIES:
        man = request_manifest(P, corr, RES, request_id="r0", query=query).manifest
        facts = OC.program_producer_facts(P, corr, query=query)
        assert set(facts) == {(r["op_path"], r["output_member"]) for r in man["identities"] if r["family"] == "instance_outputs"}, query
        assert OC.promote_handed_down(man, facts)["members"] == {"model/out": [NORM]}, query
        why = C.oracle_producers_of_record(types.SimpleNamespace(), man, [], {})[1]["why"]
        assert why.startswith("no --programs-root / --program-dir"), (query, why)                  # past the query gate
    why = C.oracle_producers_of_record(types.SimpleNamespace(), {"query": {"engine": "traversal"}, "identities": []}, [], {})[1]["why"]
    assert why.startswith("manifest is not the query path's"), why


def test_program_facts_that_disagree_across_programs_carry_no_selector():
    """Member 1 read AddWidenedBf16 in one Program and MulVecF32 in another: no operand (the oracle names the ambiguity), never one of them."""
    out: dict = {}
    for P in (gemma_like_program(), gemma_like_program(residual_operand=4)):
        OC.program_producer_facts(P, Correspondence(P), into=out)
    assert out[(NORM, "1")]["operand"] is None and out[(NORM, "0")]["operand"] == "MulVecF32_v1"
    assert out[("model", "out")]["consumers"] == [NORM]


def test_merge_adds_facts_only_where_the_manifest_declares_no_selector():
    declared = {("a", "0"): {"spec": "N_v1", "ordinal": None, "operand": None},
                ("a", "1"): {"spec": "N_v1", "ordinal": 0, "operand": None},
                ("b", "0"): {"spec": "N_v1", "ordinal": None, "operand": "X_v1"},
                ("c", "out"): {"spec": "S_v1", "ordinal": None, "operand": None, "promoted": True, "consumers": ["d"]},
                ("e", "0"): {"spec": "G_v1", "ordinal": None, "operand": None},
                ("f", "0"): {"spec": None, "ordinal": None, "operand": None, "conflict": True}}
    derived = {("a", "0"): {"spec": "N_v1", "operand": "Y_v1", "consumers": ["m"]},
               ("a", "1"): {"spec": "N_v1", "operand": "Z_v1", "consumers": ["m"]},
               ("b", "0"): {"spec": "N_v1", "operand": "Y_v1", "consumers": []},
               ("c", "out"): {"spec": "S_v1", "operand": "W_v1", "consumers": ["x"]},
               ("e", "0"): {"spec": "Other_v1", "operand": "Y_v1", "consumers": ["m"]},
               ("f", "0"): {"spec": "N_v1", "operand": "Y_v1", "consumers": ["m"]},
               ("new", "0"): {"spec": "N_v1", "operand": "Y_v1", "consumers": ["m"]}}
    m = OC.merge_producers(declared, derived)
    assert m[("a", "0")] == {"spec": "N_v1", "ordinal": None, "operand": "Y_v1", "consumers": ["m"]}
    assert m[("a", "1")] == {"spec": "N_v1", "ordinal": 0, "operand": None, "consumers": ["m"]}        # the manifest's ordinal stands
    assert m[("b", "0")] == declared[("b", "0")]                                                        # the manifest's operand stands
    assert m[("c", "out")]["consumers"] == ["d"] and m[("c", "out")]["operand"] == "W_v1"               # a promoted row keeps its consumers
    assert m[("e", "0")] == declared[("e", "0")] and m[("f", "0")] == declared[("f", "0")]              # another family / a conflict: untouched
    assert ("new", "0") not in m and declared[("a", "0")]["operand"] is None                            # never adds, never mutates


def test_fused_norm_members_are_read_at_the_program_selected_narrowing(tmp_path):
    """#57's 44,928: a v2 row names the family only -> 'ambiguous producer'; with the Program's operand each member is read at the
    narrowing it feeds from, in either fold emission order, and swapped committed bytes are still named."""
    for permute in (False, True):
        d, bmap, committed, op = _exploded_norm_match(str(tmp_path / f"p{int(permute)}"), permute=permute)
        man = {"query": {"engine": "v2-query"}, "identities": [
            {"family": "instance_outputs", "op_path": op, "output_member": m, "spec": "NarrowF32ToBf16_v1"} for m in ("0", "1")]}
        base = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), producers=OC.producers_of(man))
        assert base["compared"] == 0 and base["result"] is None and all(k.startswith("ambiguous producer") for k in base["not_compared"]), base["not_compared"]
        derived = {(op, "0"): {"spec": "NarrowF32ToBf16_v1", "operand": "MulVecF32_v1", "consumers": []},
                   (op, "1"): {"spec": "NarrowF32ToBf16_v1", "operand": "AddWidenedBf16_v1", "consumers": []}}
        prod = OC.merge_producers(OC.producers_of(man), derived)
        res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), producers=prod)
        assert res["compared"] == 4 and res["equal"] == 4 and res["mismatch_n"] == 0 and not res["not_compared"], (permute, res)
        committed[(0, f"{op}/0")], committed[(0, f"{op}/1")] = committed[(0, f"{op}/1")], committed[(0, f"{op}/0")]
        res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), producers=prod)
        assert res["result"] is False and res["mismatch_n"] == 4


def test_member_the_fold_hints_elsewhere_is_resolved_through_its_consumers(tmp_path):
    """#57's 432 `model/out`: the v2 row has no `promoted` flag; the Program's consumers resolve it through the Value (the fold instance
    of the producer family whose output a consumer reads), compared, complete -- and a wrong committed byte is a mismatch."""
    d, vals = make_match(str(tmp_path))
    data = _add_promoted(d)
    bmap, committed = _promoted_bmap(vals, data)
    man = {"query": {"engine": "v2-query"}, "identities": [{"family": "instance_outputs", "op_path": PRODUCER_OP, "output_member": "out", "spec": "ScaleRow_v1"}]}
    base = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), producers=OC.producers_of(man))
    assert base["result"] is None and base["not_compared"].get("no fold instance of this op at this step") == 2, base["not_compared"]
    prod = OC.merge_producers(OC.producers_of(man), {(PRODUCER_OP, "out"): {"spec": "ScaleRow_v1", "operand": None, "consumers": [CONSUMER_OP]}})
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), producers=prod)
    assert res["result"] is True and res["complete"] is True and res["compared"] == 2 * len(vals) + 2 + 2, res
    o = OC.MatchOracle(d)
    b, why, n = o.member_bytes(0, PRODUCER_OP, "out", "r0", producer=prod[(PRODUCER_OP, "out")])
    assert why is None and n == ROWS[U0] and b == data[ROWS[U1] * H * 2:]
    bad = bytearray(data)
    bad[ROWS[U1] * H * 2] ^= 1
    bmap, committed = _promoted_bmap(vals, bytes(bad))
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), producers=prod)
    assert res["result"] is False and res["mismatch_n"] == 1 and res["mismatch"][0]["identity"]["op_path"] == PRODUCER_OP


def test_consumers_are_not_consulted_when_the_fold_names_the_producer_under_the_path(tmp_path):
    """A member whose producer family the fold DOES hold under the identity's path is read there exactly as before: the same compare
    with and without Program consumers (which here name a module with no instance at all)."""
    d, bmap, committed, op = _exploded_norm_match(str(tmp_path))
    prod = {(op, "0"): {"spec": "NarrowF32ToBf16_v1", "ordinal": None, "operand": "MulVecF32_v1"},
            (op, "1"): {"spec": "NarrowF32ToBf16_v1", "ordinal": None, "operand": "AddWidenedBf16_v1"}}
    with_cons = {k: dict(v, consumers=["model.layers.9.mlp"]) for k, v in prod.items()}
    a = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), producers=prod)
    b = OC.oracle_compare(OC.MatchOracle(d), bmap, reader(committed), producers=with_cons)
    for k in ("result", "complete", "compared", "equal", "mismatch_n", "not_compared"):
        assert a[k] == b[k], k
    assert a["result"] is True


def test_handed_down_member_is_observed_at_its_first_consumer_input():
    """#57's rerun: with the v2 row unannotated the acquisition hooked Gemma2Model's forward output for
    `model/out` (the final norm's rows) while the Value is the embed scale model's body hands to layer 0 -- 432 form (B) mismatches.
    Marked promoted from the Program's consumers, the manifest's own observation rule arms `<first consumer>/in` for it and nothing else."""
    P = gemma_like_program()
    corr = Correspondence(P)
    man = request_manifest(P, corr, RES, request_id="r0").manifest
    facts = OC.program_producer_facts(P, corr)
    assert MF.promoted_observation(man)["names"] == {}
    n_model = sum(1 for r in man["identities"] if (r["op_path"], r["output_member"]) == ("model", "out"))
    hd = OC.promote_handed_down(man, facts)
    assert hd == {"members": {"model/out": [NORM]}, "rows": n_model} and n_model >= 1
    rows = [r for r in man["identities"] if r["family"] == "instance_outputs"]
    assert all(r.get("promoted") is True and r["consumers"] == [NORM] for r in rows if r["op_path"] == "model")
    assert not any(r.get("promoted") for r in rows if r["op_path"] != "model")          # the norm's members and the MLP output are returns
    obs = MF.promoted_observation(man)
    assert obs["problems"] == [] and obs["names"] == {f"{NORM}/in": ["model", "out"]}
    assert OC.promote_handed_down(man, facts)["rows"] == 0                              # already promoted: untouched


def test_only_a_value_every_reader_of_which_is_inside_the_producer_subtree_is_promoted():
    derived = {("m", "out"): {"spec": "S_v1", "consumers": ["m.a", "m.b.c"]},              # handed down to two descendants
               ("m.a", "0"): {"spec": "G_v1", "consumers": ["m.b"]},                       # a sibling reads it: a return
               ("m.b", "out"): {"spec": "G_v1", "consumers": ["m.b.c", "m"]},              # a child AND the parent read it: a return
               ("m.c", "0"): {"spec": "G_v1", "consumers": []},                            # no reader named
               ("m.d", "0"): {"spec": None, "consumers": ["m.d.x"], "conflict": True},     # the Programs disagree on the family
               ("m.e", "0"): {"spec": "Other_v1", "consumers": ["m.e.x"]},                 # not the row's family
               ("mm", "0"): {"spec": "G_v1", "consumers": ["mmx.y"]},                      # a string prefix, not a module ancestor
               ("m.f", "out"): {"spec": "S_v1", "consumers": ["m.f.x"]}}                   # already promoted by the manifest
    ids = [{"family": "instance_outputs", "op_path": op, "output_member": mem, "spec": "S_v1" if op in ("m", "m.f") else "G_v1"}
           for op, mem in derived]
    ids[-1].update(promoted=True, consumers=["m.f.y"])
    ids.append({"family": "input_token_ids", "op_path": "m", "output_member": "out", "spec": "S_v1"})
    hd = OC.promote_handed_down({"identities": ids}, derived)
    assert hd == {"members": {"m/out": ["m.a", "m.b.c"]}, "rows": 1}
    assert [(r["op_path"], r.get("promoted"), r.get("consumers")) for r in ids if r.get("promoted")] == [
        ("m", True, ["m.a", "m.b.c"]), ("m.f", True, ["m.f.y"])]
    assert OC.promote_handed_down(None, derived) == {"members": {}, "rows": 0}


def test_form_b_families_are_the_required_outputs_the_replay_cannot_evaluate():
    header = {"schema": MF.SCHEMA, "program_digest": "ab" * 64, "manifest_digest": "cd" * 64, "requests": ["r0"]}
    man = dict(header, identities=[
        {"family": "instance_outputs", "spec": "LayerNormAten_v1"}, {"family": "instance_outputs", "spec": "LayerNormAten_v1{N=4}"},
        {"family": "instance_outputs", "spec": "Gemm_v1"}, {"family": "instance_outputs", "spec": "AttentionHead_v1"},
        {"family": "instance_outputs", "spec": "TokenSelect_v1"}, {"family": "sampled_token_ids", "spec": "Bf16MulScalar_v1"},
        {"family": "instance_outputs"}, {"family": "moe_block_stream", "spec": "MoeExpertGemm_v1"}])
    assert form_b_families(man) == {"LayerNormAten_v1": 2}
    assert form_b_families(None) == {} and form_b_families(dict(header, identities=[{"family": "instance_outputs", "spec": "Gemm_v1"}])) == {}
    assert json.loads(json.dumps(form_b_families(man))) == {"LayerNormAten_v1": 2}
    with pytest.raises(MissingField, match="field 'schema' is missing"):          # identities without the header its writers write
        form_b_families({"identities": man["identities"]})
