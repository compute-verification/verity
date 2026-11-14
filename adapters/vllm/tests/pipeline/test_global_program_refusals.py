"""What `verity_vllm.pipeline.global_program` refuses to compose, by name and before a Program is written: a directory that
is no derive's, two directories of one shape, a shape no directory holds, components of two executions, a component derived
for another scheduler or another MoE construction, a descriptor that is not the one its derive recorded, components of two
registrations, a weights record it cannot read or that has no root, a stop set that is not the checkpoint's, and a stochastic
sampler whose split count is not the registered one.  A refusal is the build's whole result, and `workload_program.json`
holds it alone.  A component's launch context is recorded as its own, a configuration that states no stop token is passed
over, and a compose started beside its derives hands on the components it loaded, or ends 5 when the Build failed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from verity.definitions.registry.topp_geometry import registered_splits

from tests.pipeline.test_global_program_regress import REQUESTS, _derived_dir, _manifest
from verity_vllm.config import VLLM
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import global_program as GP

RESULT = "verity/workload-program/v1/result"
SMS = 142


def _row(tmp_path, **sampling):
    """A workload of two requests of two shapes and its two components; the first's directory is the one a test changes."""
    return _manifest(tmp_path, REQUESTS[:2], **sampling), [_derived_dir(tmp_path, 3, 2), _derived_dir(tmp_path, 5, 0)]


def _stated(d, change):
    p = Path(d) / "result.json"
    doc = json.loads(p.read_text())
    change(doc)
    p.write_text(json.dumps(doc))


def _refused(tmp_path, m, derived, reason, message, block=None, **given):
    """The build's result, held to a refusal's form: its keys in order, and the one file that holds it."""
    res = GP.build(m, derived, str(tmp_path / "out"), **given)
    head = {"schema": RESULT, "ok": False, "reason": reason, "message": message}
    assert list(res.items()) == [*head.items(), *([(block, res[block])] if block else []),
                                 ("manifest_sha256", hashlib.sha256(Path(m).read_bytes()).hexdigest())]
    assert block is None or res[block]["refusal"] == message
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["workload_program.json"]
    assert (tmp_path / "out" / "workload_program.json").read_text() == json.dumps(res, indent=1)
    return res


def test_a_directory_that_is_no_derive_s_and_two_directories_of_one_shape_are_refused_before_anything_is_written(tmp_path):
    m, derived = _row(tmp_path)
    (tmp_path / "bare").mkdir()
    with pytest.raises(ValueError) as refused:
        GP.build(m, [derived[0], str(tmp_path / "bare")], str(tmp_path / "out"))
    assert refused.value.args == (f"{tmp_path / 'bare'}: not a derive_step request artifact (no structural_inputs LP/T)",)
    (tmp_path / "again").mkdir()
    twin = _derived_dir(tmp_path / "again", 3, 2)
    with pytest.raises(ValueError) as refused:
        GP.build(m, [*derived, twin], str(tmp_path / "out"))
    assert refused.value.args == (f"two derived dirs for shape (3, 2): {derived[0]} and {twin}",)
    assert not (tmp_path / "out").exists()
    assert GP.build(m, [*derived, derived[0]], str(tmp_path / "out"))["ok"]                # one directory named twice is one directory


def test_a_shape_no_directory_holds_is_listed_with_the_derive_that_makes_it(tmp_path):
    m, derived = _row(tmp_path)
    res = GP.build(m, derived[:1], str(tmp_path / "out"))
    assert res == {"schema": RESULT, "ok": False, "reason": "missing_derived", "missing_derived": [(5, 0)],
                   "hint": "derive_step --wrapper request --LP <lp> --T <t> --workload <manifest> for each"}
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["workload_program.json"]
    assert (tmp_path / "out" / "workload_program.json").read_text() == json.dumps(res, indent=1)


def test_components_of_two_executions_are_refused_with_the_applicability_they_declare(tmp_path):
    m, derived = _row(tmp_path)
    _stated(derived[0], lambda r: r["applicability"].update(execution="bi-compiled"))
    app = _refused(tmp_path, m, derived, "applicability", "components declare different execution profiles ['bi-compiled', 'bi-eager']",
                   "applicability")["applicability"]
    assert (app["execution"], app["row_execution"]) == (["bi-compiled", "bi-eager"], "bi-eager")
    assert app["components_disagree"] == {"execution": ['"bi-compiled"', '"bi-eager"']} and app["components"]["numerics_profile"] == "toy"


def test_a_component_derived_for_another_scheduler_is_refused_with_the_lag_the_workload_declares(tmp_path):
    m, derived = _row(tmp_path)
    _stated(derived[0], lambda r: r.update(target={"declared": {"async_scheduling": False}}))
    message = (f"component (LP=3, T=2) {derived[0]} was derived under a target resolving async_scheduling=False (LAG 0) but the workload's "
               f"declared target resolves async_scheduling=True (LAG 1): one workload Program cannot be specialized for two schedulers")
    lag = _refused(tmp_path, m, derived, "lag", message, "lag")["lag"]
    assert (lag["value"], lag["async_scheduling"], lag["declared_explicitly"]) == (1, True, False)
    assert lag["components"] == {"3,2": {"dir": derived[0], "async_scheduling": False, "lag": 0, "declared_explicitly": True, "vllm_revision": VLLM}}


def test_a_component_that_binds_another_moe_construction_is_refused_with_the_one_the_workload_declares(tmp_path):
    m, derived = _row(tmp_path)
    _stated(derived[0], lambda r: r.update(spec_histogram={"MoeFfnFixedL_v1{L=4}": 2}))
    message = (f"component (LP=3, T=2) {derived[0]} binds the fused-MoE block with ['MoeFfnFixedL_v1'] -- the padded construction -- while the "
               f"workload declares 'indexed-read-ordered': refused by name (a Program is of one construction; the digest would be another row's)")
    moe = _refused(tmp_path, m, derived, "moe_construction", message, "moe_construction")["moe_construction"]
    assert (moe["value"], moe["declared_explicitly"], moe["moe_block_present"]) == ("indexed-read-ordered", False, False)
    assert moe["components"] == {"3,2": {
        "dir": derived[0], "moe_construction": None, "binds_declared_construction": False, "binds_other_construction": ["MoeFfnFixedL_v1"],
        "note": "component carries no target.declared (pre-v3 derive): the construction is read from its Definitions only"}}


def test_a_descriptor_that_is_not_the_one_its_derive_recorded_is_refused_by_both_digests_and_its_directory(tmp_path):
    m, derived = _row(tmp_path)
    recorded = json.loads((Path(derived[0]) / "result.json").read_text())["program"]["digest"]
    _stated(derived[0], lambda r: r["program"].update(digest="0" * 128))
    res = GP.build(m, derived, str(tmp_path / "out"))
    assert res == {
        "schema": RESULT, "ok": False, "reason": "component_provenance",
        "message": f"{derived[0]}: descriptor digest {recorded[:16]} != the digest the derive recorded {'0' * 16}: "
                   f"the component descriptor is not the Program derive_step built",
        "component": derived[0], "manifest_sha256": hashlib.sha256(Path(m).read_bytes()).hexdigest()}
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["workload_program.json"]
    assert (tmp_path / "out" / "workload_program.json").read_text() == json.dumps(res, indent=1)


def test_components_that_register_different_values_are_not_one_program_s_registration(tmp_path):
    m, derived = _row(tmp_path)
    _stated(derived[0], lambda r: r["program"].update(supplied={"registration": {"fields": [["w", 16]]}}))
    _refused(tmp_path, m, derived, "registration",
             "the components register different values (result.json program.supplied, 2 distinct): not one program's registration")


def _weights_record(d, **doc):
    digest = json.loads((Path(d) / "result.json").read_text())["program"]["digest"]
    (Path(d) / "weights_of_record.json").write_text(json.dumps({"program_digest": digest, **doc}))
    return digest


def test_a_weights_record_that_cannot_be_read_or_has_no_root_is_refused_with_the_components_read_before_it(tmp_path):
    m, derived = _row(tmp_path)
    (Path(derived[0]) / "weights_of_record.json").write_text("{not JSON")
    message = "component 3,2: weights_of_record.json unreadable (Expecting property name enclosed in double quotes: line 1 column 2 (char 1))"
    res = _refused(tmp_path, m, derived, "weights_of_record", message, "weights_of_record")
    assert res["weights_of_record"] == {"refusal": message, "components": {}, "per_rank": None}
    (tmp_path / "out" / "workload_program.json").unlink()
    digest = _weights_record(derived[0], complete=True)
    message = "component 3,2: weights_of_record.json carries no root_of_record"
    res = _refused(tmp_path, m, derived, "weights_of_record", message, "weights_of_record")
    assert res["weights_of_record"] == {"refusal": message, "per_rank": None, "components": {"3,2": {
        "root_of_record": None, "program_digest": digest, "tp": {"world": 1, "rank": 0}, "complete": True, "n_pinned": None, "n_fields": None,
        "unresolved": [], "schema": None, "checkpoint_dir": None}}}


def test_a_weights_record_of_the_pinned_shards_is_carried_and_a_component_without_one_is_carried_as_null(tmp_path):
    m, derived = _row(tmp_path)
    digest = _weights_record(derived[0], complete=True, root_of_record="ab" * 64, checkpoint={"all_shards_pinned": True, "dir": "/ck"})
    art = GP.build(m, derived, str(tmp_path / "out"))
    assert art["ok"] and art["weights_of_record"] == {
        "refusal": None, "per_rank": {"0": "ab" * 64}, "complete": False,
        "note": "1 of 2 components carry no weights_of_record.json: the row's Commit fails 'weights unpinned' (MAN-09)",
        "components": {"5,0": None, "3,2": {
            "root_of_record": "ab" * 64, "program_digest": digest, "tp": {"world": 1, "rank": 0}, "complete": True, "n_pinned": None,
            "n_fields": None, "unresolved": [], "schema": None, "checkpoint_dir": "/ck", "all_shards_pinned": True, "source": None}}}


def _checkpoint(tmp_path, **files):
    (tmp_path / "ck").mkdir()
    for name, doc in files.items():
        (tmp_path / "ck" / f"{name}.json").write_text(json.dumps(doc))
    return str(tmp_path / "ck")


def test_a_stop_set_that_is_not_the_checkpoint_s_is_refused_with_both_sets(tmp_path):
    m, derived = _row(tmp_path, eos_token_ids=[7])
    ck = _checkpoint(tmp_path, generation_config={"eos_token_id": 2}, config={})
    message = "manifest sampling.eos_token_ids [7] != the pinned checkpoint's eos_token_id set [2]: the manifest was pinned from another tokenizer/generation config"
    res = _refused(tmp_path, m, derived, "stop_rule", message, "stop_rule", checkpoint_dir=ck)
    assert res["stop_rule"] == {"schema": "verity/workload-program/v1/stop-rule", "eos_token_ids": None, "checkpoint_dir": ck, "refusal": message,
                                "source": {"generation_config.json": [2], "manifest.sampling.eos_token_ids": [7]}}


@pytest.mark.parametrize("sampling, files, ids, source", [
    ({"eos_token_ids": [7]}, {"generation_config": {"eos_token_id": [7]}, "config": {}},
     [7], {"generation_config.json": [7], "manifest.sampling.eos_token_ids": [7]}),
    ({}, {"generation_config": {"eos_token_id": None}, "config": {"eos_token_id": 9}}, [9], {"config.json": [9]}),
    ({}, {"generation_config": {"eos_token_id": [4, 2]}, "config": {"eos_token_id": 9}}, [2, 4, 9],
     {"generation_config.json": [2, 4], "config.json": [9]}),
], ids=["the-manifest-s-set-agreed", "one-file-states-none", "both-files-state-one"])
def test_a_configuration_that_states_no_stop_token_is_passed_over_and_the_rule_carries_the_program_s_lag(tmp_path, sampling, files, ids, source):
    m, derived = _row(tmp_path, **sampling)
    ck = _checkpoint(tmp_path, **files)
    art = GP.build(m, derived, str(tmp_path / "out"), checkpoint_dir=ck)
    rule = art["stop_rule"]
    assert art["ok"] and (rule["eos_token_ids"], rule["source"], rule["checkpoint_dir"]) == (ids, source, ck)
    assert rule["eos_lag_rows"] == rule["lag"] == art["lag"]["value"] == 1 and "refusal" not in rule
    assert list(rule) == ["schema", "eos_token_ids", "source", "checkpoint_dir", "rule", "eos_lag_rows", "lag", "eos_lag_source"]


GEOMETRY = {"schema": "verity/workload-program/v1/sampler-geometry", "relation": "registered"}
NO_SMS = ("stochastic sampler but the components declare no num_SMs (applicability.num_sms): the registered split count cannot be evaluated "
          "-- the Program does not model this row's sampler geometry")
OTHER_SPLITS = f"the components declare splits 3, not the registered split count {registered_splits(SMS)} of num_SMs={SMS}: rebuild them (`pipeline.build.sampler_geometry`)"


@pytest.mark.parametrize("declared, message, stated", [
    ({"sampler": {"kind": "stochastic", "splits": 3}}, NO_SMS, {"num_sms": None, "component_splits": 3}),
    ({"sampler": {"kind": "stochastic", "splits": 3}, "num_sms": SMS}, OTHER_SPLITS,
     {"num_sms": SMS, "component_splits": 3, "splits": registered_splits(SMS)}),
], ids=["no-sm-count", "another-split-count"])
def test_a_stochastic_sampler_whose_split_count_is_not_the_registered_one_is_refused_with_its_geometry(tmp_path, declared, message, stated):
    m, derived = _row(tmp_path)
    for d in derived:
        _stated(d, lambda r: r["applicability"].update(declared))
    geometry = _refused(tmp_path, m, derived, "sampler_geometry", message, "sampler_geometry")["sampler_geometry"]
    assert {k: v for k, v in geometry.items() if k not in ("rule", "source")} == {**GEOMETRY, **stated, "refusal": message}


def test_a_stochastic_sampler_at_the_registered_split_count_is_declared_and_a_greedy_one_declares_none(tmp_path):
    m, derived = _row(tmp_path)
    assert "sampler_geometry" not in GP.build(m, derived, str(tmp_path / "greedy"))
    splits = registered_splits(SMS)
    for d in derived:                               # the count read from the sampler's own geometry where the components state none
        _stated(d, lambda r: r["applicability"].update(sampler={"kind": "stochastic", "splits": splits, "geometry": {"num_sms": SMS}}))
    art = GP.build(m, derived, str(tmp_path / "out"))
    assert art["ok"] and {k: v for k, v in art["sampler_geometry"].items() if k not in ("rule", "source")} == {
        **GEOMETRY, "num_sms": SMS, "component_splits": splits, "splits": splits}
    assert list(art).index("sampler_geometry") == list(art).index("stop_rule") + 1


def test_a_component_s_launch_context_is_recorded_as_its_own(tmp_path):
    m, derived = _row(tmp_path)
    _stated(derived[0], lambda r: r["applicability"].update(launch_context={"tile": 4}))
    art = GP.build(m, derived, str(tmp_path / "out"))
    provenance = art["inputs"]["component_provenance"]
    assert art["ok"] and provenance["3,2"]["launch_context"] == {"tile": 4} and "launch_context" not in provenance["5,0"]
    assert "launch_context" not in art["applicability"]["components"] and art["applicability"]["refusal"] is None


def test_a_compose_started_beside_its_derives_hands_on_what_it_loaded_and_ends_5_when_the_build_failed(tmp_path, monkeypatch, capsys):
    m, derived = _row(tmp_path)
    asked, built = [], []

    def follow(dirs, need, canon, **how):
        asked.append((dirs, need, canon, how))
        return loaded[0]

    def build(*given, **more):
        built.append((given, more))
        return {"ok": False, "reason": "refused", "message": "why"}

    monkeypatch.setattr(GP, "follow_components", follow)
    monkeypatch.setattr(GP, "build", build)
    line = ["--workload", m, "--derived", derived[0], "--derived", derived[1], "--out", str(tmp_path / "out"), "--follow", "2.5",
            "--follow-done", "done", "--follow-stop", "stop", "--eos-token-ids", "7, 2,", "--theta-registered"]
    loaded: list[dict | None] = [None]
    assert GP.main(cli.parse("global-program", line, {})) == 5 and not built and capsys.readouterr().out == ""
    assert asked == [(derived, {(3, 2), (5, 0)}, {}, {"timeout": 2.5, "done": "done", "stop": "stop"})]
    loaded[0] = {derived[0]: {"prov": {}}}
    assert GP.main(cli.parse("global-program", line, {})) == 4
    ((given, more),) = built
    assert given == (m, derived, str(tmp_path / "out")) and more == {
        "checkpoint_dir": None, "eos_token_ids": [7, 2], "artifact_root": None, "early": loaded[0], "canon": {}, "theta_registered": True}
    assert more["canon"] is asked[1][2]                                                    # the definitions the components were loaded into
    del asked[:], built[:]
    assert GP.main(cli.parse("global-program", line[:8], {})) == 4 and not asked           # without --follow no derive is waited for
    assert built[0][1] == {"checkpoint_dir": None, "eos_token_ids": None, "artifact_root": None, "early": None, "canon": {}, "theta_registered": False}
