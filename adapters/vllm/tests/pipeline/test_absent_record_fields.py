"""What four readers make of a record that leaves an optional field out: the field absent, null and empty read alike.

* `verity_vllm.pipeline.program_cache`: a workload without `sweep` or `requests` keys a derive by nulls and no requests.
* `verity_vllm.pipeline.vu_store`: a unit that states no evaluations names no value; a replay record that states no evaluator kept
  no block on the device, and one that states no weights has no root for the export's provenance.
* `verity_vllm.pipeline.manifest`: a Build's record without `program` has no recorded digest; an incomplete manifest that states
  neither its unmodelled keys nor its peer binding is still refused, and says so.
* `verity_vllm.pipeline.program_graph`: a group without `varying` is one specialization and one without `definition_ids` indexes
  none; a replay record states its strata under `sampled_replay`, else under `replay_partition`, else the run has no units; a stored
  Program without `build_dirs` places no request.
* `verity_vllm.pipeline.match`: the realized shape of a run counts a request's tokens only where its record states no count, reads
  the engine's steps at the record's top, else under its `timing`, and the steps kept under either of two names; the Program
  writer's request table and instance rows list nothing for a request with no prompt ids or samples and an instance with no views.
* `verity_vllm.pipeline.global_program`: a component states its applicability at its record's top, else under `artifact`; a
  declaration without components, or without a sampler, is a greedy one; a weights record without `unresolved` or `checkpoint`
  lists none and names no directory; the command's summary states null for a block the result does not have.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from types import SimpleNamespace

import pytest

from tests.check.test_sampled_replay import _program
from tests.pipeline.test_vu_export import _world
from verity_vllm.check.replay import driver as SR
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import global_program as GP
from verity_vllm.pipeline import manifest as M
from verity_vllm.pipeline import match as RC
from verity_vllm.pipeline import program_cache as PC
from verity_vllm.pipeline import program_graph as PG
from verity_vllm.pipeline import vu_store as VS

NO_SWEEP = dict.fromkeys(PC.DERIVE_INPUTS)
SWEEP = {"sampling": "greedy", "sampling_params": {"top_k": 4}, "batch": 2, "sampling_seeds": [7, 8]}


@pytest.mark.parametrize("doc, want", [
    ({}, {**NO_SWEEP, "requests": 0}),
    ({"sweep": None, "requests": None}, {**NO_SWEEP, "requests": 0}),
    ({"sweep": {}, "requests": []}, {**NO_SWEEP, "requests": 0}),
    ({"sweep": {"batch": 2}, "requests": [{}, {}, {}]}, {**NO_SWEEP, "batch": 2, "requests": 3}),
    ({"sweep": {**SWEEP, "row_id": "not a derive's input"}, "requests": [{}]}, {**SWEEP, "requests": 1}),
])
def test_a_derive_is_keyed_by_the_sweep_fields_it_reads_and_the_number_of_requests(tmp_path, doc, want):
    p = tmp_path / "workload.json"
    p.write_text(json.dumps(doc))
    assert PC.derive_inputs(p) == want


def test_a_unit_that_states_no_evaluations_names_no_value_of_the_store(tmp_path):
    values, program = tmp_path / "values", tmp_path / "programs" / "p0"
    values.mkdir()
    program.mkdir(parents=True)
    blob = b"\x00\x01"
    held = {"value": hashlib.sha256(blob).hexdigest(), "dtype": "u8"}
    (values / f"{held['value']}.u8").write_bytes(blob)
    units = [{}, {"evaluations": None}, {"evaluations": []}, {"evaluations": [{"args": [held], "outs": {"y": {"sample": held}}}]}]
    (program / "units.jsonl").write_text("".join(json.dumps(u) + "\n" for u in units))
    assert VS.verify_store(tmp_path) == {"ok": True, "bad_values": [], "missing_values": 0, "values": 1}
    units.append({"evaluations": [{"args": [{"value": "0" * 64, "dtype": "u8"}], "outs": {}}]})
    (program / "units.jsonl").write_text("".join(json.dumps(u) + "\n" for u in units))
    assert VS.verify_store(tmp_path) == {"ok": False, "bad_values": [], "missing_values": 1, "values": 1}


class _Engine:
    cp = {"repo": "tiny/model", "revision": "abc"}


def _exported(tmp_path, monkeypatch, sr: dict) -> dict:
    """What `after_replay` hands the export for the replay record `sr`: its keyword arguments."""
    seen: list[dict] = []

    def export(*_a, **kw):
        seen.append(kw)
        return {"sets": [], "wall_s": 0}

    monkeypatch.setattr(VS, "export_vus", export)
    monkeypatch.setattr(VS, "memory_headroom", lambda: 1.0)
    (tmp_path / "limits.json").write_text(json.dumps({"workers": 4}))
    assert VS.after_replay(str(tmp_path), {"result": True, **sr}, 0, b"\x00" * 32, {}, {}, None, None, {}, None, {}, {}, {}, {}, _Engine()) == {
        "sets": [], "wall_s": 0}
    (kw,) = seen
    return kw


@pytest.mark.parametrize("sr", [{}, {"evaluator": None, "weights": None}, {"evaluator": {}, "weights": {}},
                                {"evaluator": {"retention": "host"}, "weights": {"root": "another field"}}])
def test_a_replay_that_states_no_device_retention_and_no_weights_root_exports_forked_with_no_root(tmp_path, monkeypatch, sr):
    kw = _exported(tmp_path, monkeypatch, sr)
    assert kw["limits"].workers == 4
    assert kw["provenance"]["weights"] == {"checkpoint": {"repo": "tiny/model", "revision": "abc", "weights_of_record_root": None}}


def test_a_replay_that_kept_its_blocks_on_the_device_exports_in_one_process_and_its_weights_root_is_recorded(tmp_path, monkeypatch):
    kw = _exported(tmp_path, monkeypatch, {"evaluator": {"retention": "device"}, "weights": {"committed_root": "ab" * 64}})
    assert kw["limits"].workers == 1
    assert kw["provenance"]["weights"]["checkpoint"]["weights_of_record_root"] == "ab" * 64


@pytest.mark.parametrize("res", [{}, {"program": None}, {"program": {}}])
def test_a_build_record_that_states_no_program_has_no_recorded_digest(tmp_path, res):
    assert M.recorded_program_digest(str(tmp_path), res) is None


@pytest.mark.parametrize("man, said", [
    ({"complete": False}, "manifest INCOMPLETE: 0 unmodelled key(s), first []; tp_peer_binding n_unbound None"),
    ({"complete": False, "unmodelled": None, "tp_peer_binding": None},
     "manifest INCOMPLETE: 0 unmodelled key(s), first []; tp_peer_binding n_unbound None"),
    ({"complete": False, "unmodelled": {}, "tp_peer_binding": {}},
     "manifest INCOMPLETE: 0 unmodelled key(s), first []; tp_peer_binding n_unbound None"),
    ({"complete": False, "unmodelled": {"d": 1, "b": 2, "a": 1, "c": 1}, "tp_peer_binding": {"n_unbound": 2}},
     "manifest INCOMPLETE: 4 unmodelled key(s), first ['a', 'b', 'c']; tp_peer_binding n_unbound 2"),
])
def test_an_incomplete_manifest_is_refused_whatever_it_states_of_its_unmodelled_keys_and_its_peer_binding(capsys, man, said):
    assert M.incomplete_rc(man) == 4
    assert capsys.readouterr().err.splitlines() == [said]


@pytest.mark.parametrize("binding, unbound", [(None, None), ({}, None), ({"n_unbound": 0}, 0), ({"n_unbound": 3, "bound": 1}, 3)])
def test_the_global_manifest_s_summary_states_the_unbound_peers_only_where_the_manifest_does(tmp_path, monkeypatch, capsys, binding, unbound):
    man = {"populations": {"identities": 5, "by_family": {"call_boundaries": 5}}, "manifest_digest": "m" * 64, "program_digest": "p" * 128,
           "complete": True, **({} if binding is None else {"tp_peer_binding": binding})}
    monkeypatch.setattr(M, "build_global_ranks", lambda *_a, **_k: man)
    fixture, out = tmp_path / "workload.json", tmp_path / "manifest.json"
    fixture.write_text("{}")
    a = cli.parse("manifest", ["build-global", "--fixture", str(fixture), "--programs-root", str(tmp_path), "--out", str(out)], {})
    assert M.main(a) == 0 and json.loads(out.read_text()) == man
    assert json.loads(capsys.readouterr().out) == {
        "identities": 5, "manifest_digest": "m" * 64, "program_digest": "p" * 128, "complete": True, "by_family": {"call_boundaries": 5},
        "tp_peer_binding_n_unbound": unbound, "out": str(out)}


def _graph(tmp_path) -> tuple[str, dict]:
    inst, _weights = _program(attn=True)
    path = tmp_path / "instances.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(inst, f)
    return str(path), PG.program_graph([path])


def test_a_group_without_varying_is_one_specialization_and_one_without_definition_ids_indexes_none(tmp_path):
    _path, g = _graph(tmp_path)
    fixed = [x["id"] for x in g["groups"] if not x.get("varying")]
    assert fixed and len(fixed) >= 3, "the test Program has groups of one specialization"
    for name, form in (("stated", lambda x: x), ("absent", lambda x: {k: v for k, v in x.items() if k not in ("varying", "definition_ids")}),
                       ("null", lambda x: dict(x, varying=None, definition_ids=None))):
        d = tmp_path / name
        d.mkdir()
        doc = dict(g, groups=[form(x) if x["id"] in fixed else x for x in g["groups"]])
        (d / "row.program.json").write_text(json.dumps(doc))
        PG.catalog(d)
    ops = {name: json.loads((tmp_path / name / "row.ops.json").read_text())["ops"] for name in ("stated", "absent", "null")}
    assert ops["absent"] == ops["null"] == ops["stated"]
    assert all(o["varying"] == {} and o["specs_enumerated"] == 1 for o in ops["absent"] if o["group"] in fixed)
    index = {name: json.loads((tmp_path / name / "row.program.json").read_text())["definitions_index"] for name in ops}
    kept = {k for x in g["groups"] if x["id"] not in fixed for k in x.get("definition_ids") or [] if k}
    assert index["absent"] == index["null"] == {k: v for k, v in index["stated"].items() if k in kept}
    assert set(index["stated"]) > kept, "the groups of one specialization index Definitions where they state their ids"


def test_a_replay_record_states_its_strata_under_sampled_replay_else_under_replay_partition_else_none(tmp_path):
    world = _world(attn=True)
    bmap, committed = world.bmap_and_reader()
    rep = SR.sampled_replay({"r0": world.inst}, bmap, committed,
                            lambda f, lo, hi: world.weights[world.woff[f][0] + lo: world.woff[f][0] + hi], per_stratum=1, seed=1, workers=0)
    path = tmp_path / "instances.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(world.inst, f)
    with_units, without = PG.program_graph([path], strata=rep["strata"]), PG.program_graph([path])
    assert with_units != without and sum(x["units_in_run"] or 0 for x in with_units["groups"]) == rep["population"]["vus"]

    def written(record: dict) -> dict:
        (tmp_path / "record.json").write_text(json.dumps(record))
        out = tmp_path / "program.json"
        assert PG.main(cli.parse("program-graph", [str(out), str(path), "--strata", str(tmp_path / "record.json")], {})) == 0
        return json.loads(out.read_text())

    stated = json.loads(json.dumps(with_units, default=str))
    assert written({"sampled_replay": {"strata": rep["strata"]}}) == stated
    assert written({"replay_partition": {"strata": rep["strata"]}}) == stated
    assert written({"sampled_replay": None, "replay_partition": {"strata": rep["strata"]}}) == stated
    assert written({"sampled_replay": {}, "replay_partition": {"strata": rep["strata"]}}) == stated
    assert written({"sampled_replay": {"strata": rep["strata"]}, "replay_partition": {"strata": {}}}) == stated
    none = json.loads(json.dumps(without, default=str))
    unstated: list[dict] = [{}, {"sampled_replay": None, "replay_partition": None}, {"sampled_replay": {}, "replay_partition": {}},
                            {"replay_partition": {"other": 1}}, {"sampled_replay": {"strata": None}}]
    for record in unstated:
        assert written(record) == none, record


def test_a_stored_program_without_build_dirs_places_no_request(tmp_path):
    for name, doc in (("a", {}), ("b", {"build_dirs": None}), ("c", {"build_dirs": {}}), ("d", {"build_dirs": {"r0": "/builds/r0", "r1": ""}}),
                      ("e", {"build_dirs": {"r2": "/builds/r2", "r3": None}})):
        (tmp_path / "programs" / name).mkdir(parents=True)
        (tmp_path / "programs" / name / "program.json").write_text(json.dumps(doc))
    assert PG.request_programs_of(tmp_path) == {"r0": "/builds/r0", "r2": "/builds/r2"}
    assert PG.request_programs_of(tmp_path / "no such store") == {}


def _plan(tmp_path, monkeypatch) -> RC.Plan:
    """A run directory's plan, with nothing captured yet."""
    monkeypatch.setattr(RC, "profile_config", lambda name: None)
    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "wl.json").write_text(json.dumps({"requests": []}))
    (run / RC.G.ARTIFACTS.get("run", "run.json")).write_text(json.dumps(
        {"case": "B0", "workload_path": str(tmp_path / "wl.json"), "profile": "p", "config_id": "x", "snapshot_steps": [0, 1],
         "snapshot_max_bytes": 777}))
    a = cli.parse("match", ["--run-dir", str(run), "--phase", "cpu"], {})
    RC._defaults_from_run_dir(a)
    plan = RC.Plan(a)
    plan.capture_dir.mkdir()
    return plan


@pytest.mark.parametrize("stated, lens", [
    ({}, (0, 0)),
    ({"prompt_token_ids": None, "generated_token_ids": []}, (0, 0)),
    ({"prompt_token_ids": [1, 2, 3], "generated_token_ids": [7, 8]}, (3, 2)),
    ({"prompt_len": 5, "n_generated": 4, "prompt_token_ids": [1], "generated_token_ids": []}, (5, 4)),
    ({"prompt_len": None, "n_generated": 0, "prompt_token_ids": [1, 2], "generated_token_ids": [7]}, (None, 0)),
])
def test_the_realized_shape_counts_a_request_s_tokens_only_where_its_record_states_no_count(tmp_path, monkeypatch, stated, lens):
    plan = _plan(tmp_path, monkeypatch)
    (plan.capture_dir / "tokens.json").write_text(json.dumps({"requests": [{"request_id": "r0", "first_step": 0, "finish_step": 1, **stated}]}))
    real = RC.write_realized(plan)
    assert real is not None and (real["prompt_lens"], real["n_generated"]) == ([lens[0]], [lens[1]])
    assert real["max_concurrent_requests"] == 1 and json.loads(plan.p("run").read_text())["realized"] == real


@pytest.mark.parametrize("tokens, steps", [
    ({}, None), ({"timing": None}, None), ({"timing": {}}, None), ({"engine_steps": None, "timing": {"other": 1}}, None),
    ({"timing": {"engine_steps": 9}}, 9), ({"engine_steps": 0, "timing": {"engine_steps": 9}}, 9),
    ({"engine_steps": 4}, 4), ({"engine_steps": 4, "timing": {"engine_steps": 9}}, 4), ({"engine_steps": 4, "timing": None}, 4),
])
def test_the_realized_engine_steps_are_the_record_s_own_else_its_timing_s_else_none(tmp_path, monkeypatch, tokens, steps):
    plan = _plan(tmp_path, monkeypatch)
    (plan.capture_dir / "tokens.json").write_text(json.dumps({"requests": [], **tokens}))
    real = RC.write_realized(plan)
    assert real is not None and (real["engine_steps_capture"], real["engine_steps_control"]) == (steps, None)
    plan.control_tokens_path().parent.mkdir()
    plan.control_tokens_path().write_text(json.dumps({"requests": [], **tokens}))
    real = RC.write_realized(plan)
    assert real is not None and (real["engine_steps_capture"], real["engine_steps_control"]) == (steps, steps)
    (plan.capture_dir / "tokens.json").unlink()
    real = RC.write_realized(plan)
    assert real is not None and (real["engine_steps_capture"], real["engine_steps_control"]) == (None, steps)


def test_a_run_with_no_tokens_record_has_no_realized_shape(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    assert RC.write_realized(plan) is None and "realized" not in json.loads(plan.p("run").read_text())


@pytest.mark.parametrize("kept, steps", [
    ('{"steps": [3, 1, 1]}', [1, 3]), ('{"snapshot_steps": ["2", 0]}', [0, 2]), ('{"steps": [4], "snapshot_steps": [5]}', [4]),
    ('{"steps": [], "snapshot_steps": [5]}', [5]), ('{"steps": null, "snapshot_steps": [5, 6]}', [5, 6]),
    ("{}", None), ('{"steps": null, "snapshot_steps": null}', None), ('{"steps": [], "snapshot_steps": []}', None),
    ("[1, 2]", None), ('{"steps": ["x"]}', None), ("not json", None), (None, None),
])
def test_the_realized_snapshot_steps_are_the_ones_the_capture_kept_under_either_name(tmp_path, monkeypatch, kept, steps):
    plan = _plan(tmp_path, monkeypatch)
    (plan.capture_dir / "tokens.json").write_text(json.dumps({"requests": []}))
    if kept is not None:
        (plan.snapshots / "snapshots.json").write_text(kept)
    real = RC.write_realized(plan)
    assert real is not None and real["snapshot_steps_realized"] == steps


def _fold(**stated) -> SimpleNamespace:
    """A fold result that holds no Program: what the Program writer reads of it for `instances.jsonl`."""
    return SimpleNamespace(program=None, body=None, digest="d" * 64, accesses=None, **stated)


def test_the_program_writer_lists_nothing_for_a_request_without_ids_or_samples_and_an_instance_without_views(tmp_path, monkeypatch):
    monkeypatch.setenv(RC.INSTANCES_FORM_ENV, "runs")
    requests = [{"request_id": "r0", "prompt_token_ids": [1, 2, 3], "max_tokens": 4}, {"request_id": "r1", "prompt_token_ids": None},
                {"request_id": "r2", "prompt_token_ids": []}, {"request_id": "r3"}, {"prompt_token_ids": [9]}]
    instances = [{"step": 0, "name": "a"}, {"step": 1, "name": "b", "operand_views": None, "output_views": []},
                 {"step": 2, "name": "c", "operand_views": [{"alloc": 1}], "output_views": [{"alloc": 2}, {"alloc": 3}]}]
    table = {}
    for name, sampled in (("none", {}), ("null", {"sampled": None}), ("empty", {"sampled": {}}),
                          ("some", {"sampled": {"r0": [5, 6], "r1": [], "r2": None, None: [1]}})):
        d = tmp_path / name
        d.mkdir()
        written = RC.persist_program(_fold(requests=requests, instances=instances, **sampled), d, profile_name="toy", log_path="log")
        assert written == {"instances": {"path": str(d / "instances.jsonl"), "bytes": (d / "instances.jsonl").stat().st_size, "n": 3,
                                         "operand_refs_form": "runs"}}
        head, *rows = [json.loads(line) for line in (d / "instances.jsonl").read_text().splitlines()]
        assert [(r["operand_views"], r["output_views"]) for r in rows] == [([], []), ([], []), ([{"alloc": 1}], [{"alloc": 2}, {"alloc": 3}])]
        assert [(r["index"], r["request_id"], r["prompt_len"], r["max_tokens"]) for r in head["requests"]] == [
            (0, "r0", 3, 4), (1, "r1", 0, None), (2, "r2", 0, None), (3, "r3", 0, None), (4, None, 1, None)]
        table[name] = [r["served_samples"] for r in head["requests"]]
    assert table == {"none": [0] * 5, "null": [0] * 5, "empty": [0] * 5, "some": [2, 0, 0, 0, 1]}
    for name, absent in (("no requests", {}), ("null requests", {"requests": None}), ("empty requests", {"requests": []})):
        d = tmp_path / name
        d.mkdir()
        RC.persist_program(_fold(instances=[], **absent), d, profile_name="toy", log_path="log")
        assert json.loads((d / "instances.jsonl").read_text())["requests"] == []


APPLICABILITY = {"components": {"num_sms": 132}}


@pytest.mark.parametrize("result, artifact, want", [
    ({"applicability": APPLICABILITY}, None, APPLICABILITY),
    ({"artifact": {"applicability": APPLICABILITY}}, None, APPLICABILITY),
    ({"applicability": {}, "artifact": {"applicability": APPLICABILITY}}, None, APPLICABILITY),
    ({"applicability": APPLICABILITY, "artifact": {"applicability": {"other": 1}}}, None, APPLICABILITY),
    ({}, {"applicability": APPLICABILITY}, APPLICABILITY),
    ({"artifact": None}, {"artifact": {"applicability": APPLICABILITY}}, APPLICABILITY),
    ({"artifact": {}}, None, {}), ({"artifact": {"applicability": None}}, {"applicability": None, "artifact": {}}, {}),
    (None, None, {}),
])
def test_a_component_states_its_applicability_at_its_record_s_top_else_under_artifact(tmp_path, result, artifact, want):
    for name, doc in (("result.json", result), ("artifact.json", artifact)):
        if doc is not None:
            (tmp_path / name).write_text(json.dumps(doc))
    assert GP.component_applicability(str(tmp_path)) == want


@pytest.mark.parametrize("app", [{}, {"components": None}, {"components": {}}, {"components": {"sampler": None}},
                                 {"components": {"sampler": "top-p"}}, {"components": {"sampler": {}}},
                                 {"components": {"sampler": {"kind": None}}}, {"components": {"sampler": {"kind": "greedy"}}}])
def test_a_declaration_without_components_or_without_a_stochastic_sampler_declares_no_split_geometry(app):
    assert GP.sampler_geometry(app, []) is None


@pytest.mark.parametrize("components, num_sms", [
    ({"num_sms": 132, "sampler": {"kind": "stochastic", "geometry": {"num_sms": 7}}}, 132),
    ({"sampler": {"kind": "stochastic", "geometry": {"num_sms": 132}}}, 132),
    ({"num_sms": None, "sampler": {"kind": "stochastic", "geometry": {"num_sms": 132, "other": 1}}}, 132),
    ({"sampler": {"kind": "stochastic"}}, None), ({"sampler": {"kind": "stochastic", "geometry": None}}, None),
    ({"sampler": {"kind": "stochastic", "geometry": {}}}, None), ({"sampler": {"kind": "stochastic", "geometry": [132]}}, None),
])
def test_a_stochastic_sampler_s_sm_count_is_the_components_else_its_own_geometry_s_else_a_refusal(components, num_sms):
    out = GP.sampler_geometry({"components": components}, [])
    assert out is not None and out["num_sms"] == num_sms
    assert ("refusal" in out and "splits" not in out) if num_sms is None else out["splits"] > 0


def _weights_record(tmp_path, name: str, **stated) -> dict:
    d = tmp_path / name
    d.mkdir()
    (d / "weights_of_record.json").write_text(json.dumps({"root_of_record": "A" * 64, "program_digest": "d" * 64, "complete": True, **stated}))
    return {"dir": str(d), "recorded_digest": "d" * 64}


@pytest.mark.parametrize("stated, unresolved, directory", [
    ({}, [], None), ({"unresolved": None, "checkpoint": None}, [], None), ({"unresolved": [], "checkpoint": {}}, [], None),
    ({"checkpoint": {"all_shards_pinned": True}}, [], None),
    ({"checkpoint": {"all_shards_pinned": True, "dir": "/ck"}}, [], "/ck"),
    ({"unresolved": ["model.w"], "checkpoint": {"dir": "/ck"}}, ["model.w"], "/ck"),
])
def test_a_weights_record_without_unresolved_or_checkpoint_lists_none_and_names_no_directory(tmp_path, stated, unresolved, directory):
    out = GP.weights_of_record_binding({"8,3": _weights_record(tmp_path, "c", **stated)})
    entry = out["components"]["8,3"]
    assert (entry["unresolved"], entry["checkpoint_dir"]) == (unresolved, directory)
    pinned = (stated.get("checkpoint") or {}).get("all_shards_pinned") is True
    assert (out["refusal"] is None) == (pinned and not unresolved)
    assert ("self-contradicting" in out["refusal"]) if unresolved else (pinned or "pinned shards" in out["refusal"])


RESULT = {"ok": True, "workload_digest": "w" * 128, "gates": 10, "input_gates": 4, "requests": [{}, {}], "descriptor": {"bytes": 99},
          "cost": {"units": 3}}
SUMMARY = {"workload_digest": "w" * 128, "gates": 10, "input_gates": 4, "requests": 2, "descriptor_bytes": 99, "cost": {"units": 3}}
NO_BLOCK = {"eos_token_ids": None, "lag": None, "async_scheduling": None, "moe_construction": None, "moe_block_present": None}


@pytest.mark.parametrize("blocks, stated", [
    ({}, NO_BLOCK), ({"stop_rule": None, "lag": None, "moe_construction": None}, NO_BLOCK),
    ({"stop_rule": {}, "lag": {}, "moe_construction": {}}, NO_BLOCK),
    ({"stop_rule": {"eos_token_ids": [2]}, "lag": {"value": 1}, "moe_construction": {"moe_block_present": False}},
     {**NO_BLOCK, "eos_token_ids": [2], "lag": 1, "moe_block_present": False}),
    ({"stop_rule": {"eos_token_ids": [2, 7], "rule": "r"}, "lag": {"value": 0, "async_scheduling": True},
      "moe_construction": {"value": "block", "moe_block_present": True}},
     {"eos_token_ids": [2, 7], "lag": 0, "async_scheduling": True, "moe_construction": "block", "moe_block_present": True}),
])
def test_the_workload_program_s_summary_states_null_for_a_block_its_result_does_not_have(tmp_path, monkeypatch, capsys, blocks, stated):
    monkeypatch.setattr(GP, "build", lambda *_a, **_k: {**RESULT, **blocks})
    (tmp_path / "w.json").write_text("{}")
    a = cli.parse("global-program", ["--workload", str(tmp_path / "w.json"), "--derived", str(tmp_path), "--out", str(tmp_path)], {})
    assert GP.main(a) == 0
    assert json.loads(capsys.readouterr().out) == {**SUMMARY, **stated}
    monkeypatch.setattr(GP, "build", lambda *_a, **_k: {"ok": False, "reason": "refused", "message": "why"})
    assert GP.main(a) == 4 and json.loads(capsys.readouterr().out) == {"reason": "refused", "message": "why"}
