"""What four readers make of a record that leaves an optional field out: the field absent, null and empty read alike.

* `verity_vllm.pipeline.program_cache`: a workload without `sweep` or `requests` keys a derive by nulls and no requests.
* `verity_vllm.pipeline.vu_store`: a unit that states no evaluations names no value; a replay record that states no evaluator kept
  no block on the device, and one that states no weights has no root for the export's provenance.
* `verity_vllm.pipeline.manifest`: a Build's record without `program` has no recorded digest; an incomplete manifest that states
  neither its unmodelled keys nor its peer binding is still refused, and says so.
* `verity_vllm.pipeline.program_graph`: a group without `varying` is one specialization and one without `definition_ids` indexes
  none; a replay record states its strata under `sampled_replay`, else under `replay_partition`, else the run has no units; a stored
  Program without `build_dirs` places no request.
"""
from __future__ import annotations

import gzip
import hashlib
import json

import pytest

from tests.check.test_sampled_replay import _program
from tests.pipeline.test_vu_export import _world
from verity_vllm.check.replay import driver as SR
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import manifest as M
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
