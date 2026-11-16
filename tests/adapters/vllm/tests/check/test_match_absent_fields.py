"""What the Match stage's readers make of a field their document leaves out, one case per reader: the Build's artifact and the
declaration (`verity_vllm.check.match.declaration`), a fold's supplied parameters (`verity_vllm.check.match.supplied_inputs`) and
its instance rows (`verity_vllm.check.match.program_loaders`), a derived directory's shape
(`verity_vllm.check.match.derived_index`), the capture log (`verity_vllm.check.match.record`), the declared prefix-cache hit
(`verity_vllm.check.match.prefix_reads`), the sampler's geometry (`verity_vllm.check.match.sampler_geometry`) and the cross-rank
pass (`verity_vllm.check.match.xrank_collectives`).  An absent field reads as nothing stated, and where the check needs it the
failure names it."""
from __future__ import annotations

import json
import types

import pytest

from verity_vllm.check.match import declaration as GD
from verity_vllm.check.match import derived_index as DI
from verity_vllm.check.match import prefix_reads as PX
from verity_vllm.check.match import program_loaders as PL
from verity_vllm.check.match import record as REC
from verity_vllm.check.match import sampler_geometry as SG
from verity_vllm.check.match import supplied_inputs as SI
from verity_vllm.check.match import xrank_collectives as XR

# --- the Build's artifact and the declaration ------------------------------------------------------------------------------------


def test_an_artifact_that_states_no_inputs_address_map_or_chronology_gives_a_declaration_that_names_none(tmp_path):
    """No `inputs`, `address_map`, `chronology` or `descriptor` block and no descriptor file beside it: no manifest is read and no
    component is resolved, and the digest is recorded as not verified."""
    art = {"schema": "verity/workload-program/v1", "ok": True, "workload_digest": "00" * 32, "requests": []}
    (tmp_path / "workload_program.json").write_text(json.dumps(art))
    gp = GD.from_gprog(str(tmp_path / "workload_program.json"))
    assert gp["schema"] == GD.GP_SCHEMA == "cb-a/global-program/v1"
    assert gp["workload"] == {"path": None, "sha256": None, "n_requests": 0} and gp["requests"] == []
    assert (gp["chronology"], gp["address_space"], gp["applicability"]) == (None, None, None)
    assert gp["gprog_artifact"]["digest_verification"] == {"verified": False, "why": "descriptor.json.gz not beside the artifact"}
    assert gp["gprog_artifact"]["relocated_components"] == {}


def test_the_relocated_components_of_a_declaration_not_read_from_an_artifact_are_none():
    moved = {"3,3": {"from": "/pod/build_request_LP3_T3", "to": "/here/build_request_LP3_T3"}}
    assert GD.relocated_components({"gprog_artifact": {"relocated_components": moved}}) == moved
    assert GD.relocated_components({"gprog_artifact": {"path": "p"}}) == {} and GD.relocated_components({"schema": GD.GP_SCHEMA}) == {}
    assert GD.relocated_components(None) == {}


def test_a_declaration_without_applicability_fails_by_name_and_reports_nothing_declared():
    problems: list[str] = []
    res: dict = {}
    header = {"engine_kwargs": {"enforce_eager": True, "enable_prefix_caching": False}}
    facts = GD._execution_conditions(types.SimpleNamespace(fail=problems.append, info={}), res, {"schema": GD.GP_SCHEMA}, header, "")
    assert facts["engine"] == {"enforce_eager": True, "enable_prefix_caching": False}
    assert [p[:63] for p in problems] == ["the workload Program declares no applicability.engine (an older"]
    assert (res["execution"]["declared_execution"], res["execution"]["row_execution"], res["execution"]["declared_engine"]) == (None, None, None)
    assert "declared_compilation" not in res["execution"]


# --- a fold's supplied parameters ----------------------------------------------------------------------------------------------------


def _view(leaves: dict[int, int]) -> types.SimpleNamespace:
    """A view as `set_supplied` reads it: its Input nodes (node -> (width, leaves)) and no supplied parameter yet."""
    return types.SimpleNamespace(inputs={k: (16, n) for k, n in leaves.items()}, supplied_fields={}, supplied_owner={}, seed_nodes=set(), seed_index={})


def _entry(name: str, n: int) -> dict:
    return {"name": name, "fields": [{"name": f"f{i}", "width": 16, "offset": i} for i in range(n)]}


def test_a_fold_that_states_no_supplied_parameters_has_none():
    P = _view({0: 4})
    assert SI.load_oracle_supplied(P, {"n_input_nodes": 2}, "fold") is None and (P.supplied_fields, P.supplied_owner) == ({}, {})


def test_supplied_parameters_without_their_input_nodes_are_refused():
    with pytest.raises(ValueError, match=r"fold: program.json states supplied parameters but not the parameters' Input nodes \(`input_nodes`\)"):
        SI.load_oracle_supplied(_view({0: 4}), {"supplied": {"registration": _entry("values", 2)}}, "fold")


def test_a_registration_and_one_run_are_bound_at_their_nodes_and_an_empty_run_at_none():
    P = _view({1: 2, 2: 1, 5: 9})
    sup = {"registration": _entry("values", 2), "run": [{"name": "run_0"}, _entry("run_1", 1)],
           "input_nodes": {"values": [1], "run_1": [2], "weights": [5]}}
    assert SI.load_oracle_supplied(P, {"supplied": sup}, "fold") == {5}
    assert P.supplied_fields == {1: [(0, "f0", 16), (1, "f1", 16)], 2: [(0, "f0", 16)]} and P.supplied_owner == {1: None, 2: 1}
    assert SI.load_oracle_supplied(_view({1: 2}), {"supplied": {"registration": _entry("values", 2), "input_nodes": {"values": [1]}}}, "fold") is None


@pytest.mark.parametrize("nodes, n", [(None, 0), ([], 0), ([1, 2], 2)])
def test_a_supplied_parameter_with_fields_is_one_input_node(nodes, n):
    in_nodes = {"other": [7]} if nodes is None else {"values": nodes}
    with pytest.raises(ValueError, match=f"fold: supplied parameter 'values' lowers to {n} Input nodes, one homogeneous Record is one"):
        SI.load_oracle_supplied(_view({1: 2, 2: 2}), {"supplied": {"registration": _entry("values", 2), "input_nodes": in_nodes}}, "fold")


def test_an_empty_supplied_parameter_has_no_input_node():
    sup = {"run": [{"name": "run_0", "fields": []}], "input_nodes": {"run_0": [3]}}
    with pytest.raises(ValueError, match=r"fold: supplied parameter 'run_0' has no fields but Input nodes \[3\]"):
        SI.load_oracle_supplied(_view({3: 1}), {"supplied": sup}, "fold")


def test_a_supplied_entry_without_fields_tiles_no_leaf_of_its_node():
    with pytest.raises(ValueError, match=r"derived: supplied parameter 'values': fields at offsets \[\] do not tile its Input node of 2 leaves"):
        SI.set_supplied(_view({1: 2}), [(1, None, {"name": "values"})], "derived")


def test_the_registration_of_a_derived_program_is_read_from_its_report():
    reg = {"param": 1, "name": "values"}
    assert SI.derived_registration({"weights": []}, 1, "derived") is None
    assert SI.derived_registration({"supplied": {}}, 1, "derived") is None
    assert SI.derived_registration({"supplied": {"registration": reg}}, 1, "derived") == reg
    with pytest.raises(ValueError, match="derived: derive-report puts `values` at parameter 1, after 2 user inputs it is 2"):
        SI.derived_registration({"supplied": {"registration": reg}}, 2, "derived")


# --- a fold's instance rows ----------------------------------------------------------------------------------------------------------


def _kept_rows(tmp_path, monkeypatch, line: str) -> list[dict]:
    """The rows `load_oracle` keeps of a fold whose `instances.jsonl` is the one line, as it hands them on."""
    (tmp_path / "program.json").write_text(json.dumps({"program": {}}))
    (tmp_path / "instances.jsonl").write_text(line + "\n")
    kept: list[dict] = []
    monkeypatch.setattr(PL, "_finish_oracle", lambda d, label, pj, desc, rows, cls, compact: kept.extend(rows))
    PL.load_oracle(str(tmp_path), "fold")
    return kept


@pytest.mark.parametrize("refs", [None, {}, [], 0, "", False])
def test_an_instance_row_whose_operand_references_are_left_out_null_or_empty_retains_none(tmp_path, monkeypatch, refs):
    assert _kept_rows(tmp_path, monkeypatch, json.dumps({"node": 4}))[0]["operand_refs"] == []
    assert _kept_rows(tmp_path, monkeypatch, json.dumps({"node": 4, "operand_refs": refs}))[0]["operand_refs"] == []


def test_an_instance_row_retains_the_parameter_space_runs_of_each_operand_and_its_arity(tmp_path, monkeypatch):
    row = {"node": 4, "operand_refs": [[["p", 0, 8], ["n", 2, 3]], [["n", 5, 1]], [["P", 1, 2, 3]]]}
    assert _kept_rows(tmp_path, monkeypatch, json.dumps(row))[0]["operand_refs"] == [[["p", 0, 8]], [], [["P", 1, 2, 3]]]


@pytest.mark.parametrize("line, refusal", [('"node"', "string indices must be integers"), ('["node"]', "list indices must be integers")])
def test_a_line_of_the_instance_rows_that_is_not_an_object_is_refused_at_its_node(tmp_path, monkeypatch, line: str, refusal: str):
    """A JSON string or array that holds the word is taken for a row; the read of its node refuses it, before any field a row may
    leave out is looked for."""
    with pytest.raises(TypeError, match=refusal):
        _kept_rows(tmp_path, monkeypatch, line)


# --- a derived directory's shape -----------------------------------------------------------------------------------------------------


def test_the_shape_is_read_from_the_manifest_then_from_the_report_at_its_top_or_under_its_provenance(tmp_path):
    def shape(manifest: dict | None, report: dict | None):
        d = tmp_path / f"d{len(list(tmp_path.iterdir()))}"
        d.mkdir()
        if manifest is not None:
            (d / "construction_manifest.json").write_text(json.dumps(manifest))
        if report is not None:
            (d / "derive-report.json").write_text(json.dumps(report))
        return DI.derived_shape(str(d))

    assert shape(None, None) is None and shape({"schema": "s"}, None) is None and shape({"structural_inputs": {"LP": 5}}, None) is None
    assert shape({"structural_inputs": {"LP": 5, "T": 2}}, None) == (5, 2)
    assert shape({"schema": "s"}, {"structural_inputs": {"LP": 6, "T": 3}}) == (6, 3)
    assert shape(None, {"provenance": {"structural_inputs": {"LP": 7, "T": 1}}}) == (7, 1)
    assert shape(None, {"provenance": {"shape": {"LP": 8, "T": 4}}}) == (8, 4) and shape(None, {"shape": {"LP": 9, "T": 0}}) == (9, 0)
    assert shape(None, {"provenance": {"model": "m"}}) is None and shape(None, {"structural_inputs": {"T": 1}}) is None


# --- the capture log -----------------------------------------------------------------------------------------------------------------


def _log(tmp_path, *records: dict) -> str:
    p = tmp_path / "capture.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in records))
    return str(p)


def test_a_step_without_input_tokens_and_a_sample_without_its_logits_view_read_as_none(tmp_path):
    rec = REC.read_record(_log(
        tmp_path, {"$e": "RunHeader"},
        {"$e": "StepStart", "step": 0, "rows": [["u0", 0]], "logits_rows": [["u0", 0]]},
        {"$e": "SampleEnd", "step": 0, "sampled_token_ids": [7]},
        {"$e": "StepStart", "step": 1, "rows": [["u0", 1]], "logits_rows": [["u0", 1]], "input_token_ids": [7]},
        {"$e": "SampleEnd", "step": 1, "sampled_token_ids": [8], "logits_view": {"$v": ["t", "f32", [3, 64]]}},
        {"$e": "SampleEnd", "step": 2, "sampled_token_ids": [9], "logits_view": "elided"}))
    assert (rec["steps"][0]["input_token_ids"], rec["steps"][1]["input_token_ids"]) == ([], [7])
    assert rec["samples"] == {0: [7], 1: [8], 2: [9]} and rec["sampler_batch"] == {1: 3}


@pytest.mark.parametrize("workload", [None, {}, {"sha256": "ab"}, {"requests": [{"request_id": "u0"}]},
                                      {"requests": [{"request_id": "u0"}], "children": [{"request_id": "u1", "parent": "u0"}]}])
def test_a_record_is_renamed_whatever_its_header_states_of_the_workload(workload: dict | None):
    rec: dict = {"header": {} if workload is None else {"workload": workload}, "links": [{"request_id": "u0"}],
                 "steps": {0: {"rows": [("u0", 0), ("zz", 1)], "logits_rows": [("u0", 0)]}}}
    out = REC.canonicalize_record(rec, None, None, {"u0": "r0", "u1": "r1"})
    assert out == {"renamed": 2, "unknown_ids": {"zz": 1}} and rec["steps"][0]["rows"] == [("r0", 0), ("zz", 1)] and rec["links"] == [{"request_id": "r0"}]
    if workload and "requests" in workload:
        assert workload["requests"] == [{"request_id": "r0", "workload_request_id": "u0"}]
    if workload and "children" in workload:
        assert workload["children"] == [{"request_id": "r1", "parent": "r0"}]


# --- the declared prefix-cache hit ---------------------------------------------------------------------------------------------------


def test_a_request_declared_cold_and_served_cold_has_no_prefix_entry_and_one_served_warm_is_named():
    cold = types.SimpleNamespace(cached_rows=0)
    assert PX.check_hits({"request_id": "r1"}, cold) == ([], None)
    warm = types.SimpleNamespace(cached_rows=2, prefix_producers=lambda: {0: {"r0"}, 1: {"r0"}})
    why, entry = PX.check_hits({"request_id": "r1"}, warm)
    assert why == ["r1: the engine served 2 prompt rows from the prefix cache (its first scheduled position), the Program declares 0"]
    assert entry == {"rows": 2, "declared_rows": 0, "block_size": None, "declared_producers": [], "read_from": [], "linked_rows": 2}
    why, entry = PX.check_hits({"request_id": "r1", "prefix": {"rows": 2}}, warm)
    assert why == [] and entry["declared_rows"] == 2 and entry["declared_producers"] == []
    why, entry = PX.check_hits({"request_id": "r1", "prefix": {"rows": 2, "producers": ["r0"]}}, warm)
    assert why == [] and (entry["block_size"], entry["declared_producers"], entry["read_from"]) == (2, ["r0"], [["r0"]])


# --- the sampler's geometry ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("gp", [{}, {"applicability": None}, {"applicability": {"execution": "bi-eager"}}, {"applicability": {"components": {}}}])
def test_a_declaration_that_states_no_num_sms_cannot_have_its_split_count_checked(gp):
    fold = types.SimpleNamespace(inst=[{"spec": "GumbelTopPTokenSelect_v1{V=64,S=32}"}])
    res = SG.sampler_geometry_check(gp, fold, {0: 0}, {0: {"logits_rows": [("r0", 9)]}})
    assert res["info"]["num_sms"] is None and res["info"]["declared_by"] is None and len(res["problems"]) == 1


def test_num_sms_is_read_from_the_declarations_components_where_its_geometry_block_states_none():
    fold = types.SimpleNamespace(inst=[{"spec": "GumbelTopPTokenSelect_v1{V=64,S=32}"}])
    res = SG.sampler_geometry_check({"applicability": {"components": {"num_sms": 142}}}, fold, {0: 0}, {0: {"logits_rows": [("r0", 9)]}})
    assert (res["info"]["num_sms"], res["info"]["declared_by"]) == (142, "applicability.components.num_sms")


# --- the cross-rank pass -------------------------------------------------------------------------------------------------------------


def test_pairs_that_state_no_families_and_no_coverage_fold_to_none_and_to_not_complete():
    assert XR.fold_pairs([])["coverage_of_unreplayable"] == {"complete": None, "per_pair": []}
    pairs = [{"result": True, "picks": 2, "equal": 2, "by_family": {"all_reduce": {"picks": 2, "equal": 2}},
              "coverage_of_unreplayable": {"complete": True}},
             {"result": True, "picks": 1, "equal": 1}]
    out = XR.fold_pairs(pairs)
    assert out["by_family"] == {"all_reduce": {"picks": 2, "equal": 2, "mismatch": 0, "missing": 0}} and out["picks"] == 3
    assert out["coverage_of_unreplayable"] == {"complete": False, "per_pair": [{"complete": True}, {}]}
    assert XR.fold_pairs(pairs[:1])["coverage_of_unreplayable"] == {"complete": True, "per_pair": [{"complete": True}]}


def test_a_pass_that_drew_no_stratum_covers_no_site_of_a_family_its_programs_name():
    dumps = [{"rank": 0, "unreplayable_sites": {"all_reduce": ["model.layers.0.mlp"]}}]
    out = XR.coverage_of_unreplayable({}, dumps)
    assert out["complete"] is False and out["strata_sites_n"] == 0
    assert out["families"] == {"all_reduce": {"sites": 1, "covered": 0, "missing": ["model.layers.0.mlp"], "picks": None, "equal": None}}
    sampled = {"by_stratum": {"model.layers.0.mlp|0": {"families": ["all_reduce"]}}, "by_family": {"all_reduce": {"picks": 4, "equal": 4}}}
    out = XR.coverage_of_unreplayable(sampled, dumps)
    assert out["complete"] is True
    assert out["families"] == {"all_reduce": {"sites": 1, "covered": 1, "missing": [], "picks": 4, "equal": 4}}
    out = XR.coverage_of_unreplayable({"by_stratum": sampled["by_stratum"], "by_family": {"all_gather": {"picks": 1}}}, dumps)
    assert (out["families"]["all_reduce"]["picks"], out["families"]["all_reduce"]["equal"]) == (None, None)
