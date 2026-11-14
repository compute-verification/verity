"""The weights of record a workload Program carries (`verity_vllm.pipeline.global_program.weights_of_record_binding`): each
component's `weights_of_record.json` as it is carried, the roots by rank, and each refusal whole, with the components read
before it.  Refused by name: a record tied to no Program, a record of another Program, a walk that is not complete, a record
that says complete and lists unresolved fields, a record of live parameters, a root over shards that are not pinned, and two
roots for one rank.  A component with no record is carried as null and refuses nothing."""
from __future__ import annotations

import json

import pytest

from verity_vllm.pipeline import global_program as GP

ROOT, OTHER = "ab" * 64, "cd" * 64
DIGEST, STALE = "12" * 64, "34" * 64
SCHEMA = "verity-vllm/weights-of-record/v1"
#: what `_record()` is carried as, before the two fields the binding reads last
READ = {"root_of_record": ROOT, "program_digest": DIGEST, "tp": {"world": 1, "rank": 0}, "complete": True, "n_pinned": 7,
        "n_fields": 8, "unresolved": [], "schema": SCHEMA, "checkpoint_dir": "/ck"}
CARRIED = {**READ, "all_shards_pinned": True, "source": "pinned safetensors"}
LONG = ["a" * 81, "b", "c", "d", "e", "f"]
CUT = ["a" * 80, "b", "c", "d", "e"]


def _record(**change):
    return {"schema": SCHEMA, "root_of_record": ROOT, "program_digest": DIGEST, "complete": True, "n_pinned": 7, "n_fields": 8,
            "unresolved": [], "source": "pinned safetensors", "checkpoint": {"dir": "/ck", "all_shards_pinned": True}, **change}


def _components(tmp_path, records, recorded=DIGEST):
    """The binding's argument for ``records`` (a component's key to its record, None for a component that has none), each
    component in a directory of its own and derived under the digest ``recorded``."""
    out = {}
    for i, (key, record) in enumerate(records.items()):
        d = tmp_path / f"component{i}"
        d.mkdir()
        if record is not None:
            (d / "weights_of_record.json").write_text(json.dumps(record))
        out[key] = {"dir": str(d), "recorded_digest": recorded}
    return out


def _refusal(message, **components):
    return {"refusal": message, "components": components, "per_rank": None}


def test_every_component_s_record_is_carried_and_the_roots_are_given_by_rank(tmp_path):
    second = _record(root_of_record=OTHER, tp={"world": 2, "rank": 1}, source=None, n_pinned=None)
    got = GP.weights_of_record_binding(_components(tmp_path, {"5,0": second, "3,2": _record(), "3,0": _record(tp="none")}))
    assert got == {
        "refusal": None, "per_rank": {"0": ROOT, "1": OTHER}, "complete": True,
        "note": "every component carries its weights of record; Commit compares each served parameter with it by name (MAN-09)",
        "components": {"3,0": CARRIED, "3,2": CARRIED,
                       "5,0": {**CARRIED, "root_of_record": OTHER, "tp": {"world": 2, "rank": 1}, "source": None, "n_pinned": None}}}
    assert list(got["components"]) == ["3,0", "3,2", "5,0"] and list(got["per_rank"]) == ["0", "1"]


def test_a_component_derived_under_no_recorded_digest_carries_its_record_and_one_with_no_record_is_null(tmp_path):
    comps = _components(tmp_path, {"3,2": _record(program_digest=STALE), "5,0": None}, recorded=None)
    assert GP.weights_of_record_binding(comps) == {
        "refusal": None, "per_rank": {"0": ROOT}, "complete": False, "components": {"3,2": {**CARRIED, "program_digest": STALE}, "5,0": None},
        "note": "1 of 2 components carry no weights_of_record.json: the row's Commit fails 'weights unpinned' (MAN-09)"}
    (tmp_path / "none").mkdir()
    assert GP.weights_of_record_binding(_components(tmp_path / "none", {"3,2": None, "5,0": None})) == {
        "refusal": None, "per_rank": None, "complete": False, "components": {"3,2": None, "5,0": None},
        "note": "2 of 2 components carry no weights_of_record.json: the row's Commit fails 'weights unpinned' (MAN-09)"}


@pytest.mark.parametrize("recorded, named", [(DIGEST, "1212121212121212"), (None, "None")])
def test_a_record_tied_to_no_program_is_refused_with_the_component_s_recorded_digest(tmp_path, recorded, named):
    comps = _components(tmp_path, {"3,2": _record(program_digest=None)}, recorded=recorded)
    assert GP.weights_of_record_binding(comps) == _refusal(
        "component 3,2: weights_of_record.json carries no program_digest -- it cannot be tied to the component "
        f"(recorded digest {named})", **{"3,2": {**READ, "program_digest": None}})


def test_a_record_of_another_program_is_refused_as_stale_with_both_digests(tmp_path):
    comps = _components(tmp_path, {"3,2": _record(), "5,0": _record(program_digest=STALE)})
    assert GP.weights_of_record_binding(comps) == _refusal(
        "component 5,0: weights_of_record.json is for program_digest 3434343434343434, the component's recorded digest is "
        "1212121212121212 (stale record beside a re-derived component)", **{"3,2": CARRIED, "5,0": {**READ, "program_digest": STALE}})


def test_a_walk_that_is_not_complete_and_a_complete_one_that_lists_fields_are_refused_with_five_fields_of_eighty_characters(tmp_path):
    comps = _components(tmp_path, {"3,2": _record(complete=False, unresolved=LONG)})
    assert GP.weights_of_record_binding(comps) == _refusal(
        "component 3,2: weights of record incomplete -- 6 Program weight field(s) without a composition rule: "
        f"{CUT} (MAN-09: fail closed)", **{"3,2": {**READ, "complete": False, "unresolved": LONG}})
    (tmp_path / "says").mkdir()
    comps = _components(tmp_path / "says", {"3,2": _record(unresolved=LONG)})
    assert GP.weights_of_record_binding(comps) == _refusal(
        f"component 3,2: weights_of_record.json says complete but lists 6 unresolved field(s) {CUT}: self-contradicting "
        "record, refused", **{"3,2": {**READ, "unresolved": LONG}})


def test_a_record_of_live_parameters_is_refused_with_forty_characters_of_its_source(tmp_path):
    source = "LIVE parameters of the engine, re-registered at Commit"
    comps = _components(tmp_path, {"3,2": _record(source=source)})
    assert GP.weights_of_record_binding(comps) == _refusal(
        "component 3,2: weights_of_record.json describes live parameters (source 'LIVE parameters of the engine, re-regist'), "
        "not the pinned checkpoint -- not a weights of record (W1)", **{"3,2": {**CARRIED, "source": source}})


@pytest.mark.parametrize("checkpoint, pinned, directory", [({"dir": "/ck", "all_shards_pinned": False}, False, "/ck"),
                                                           ({"dir": "/ck"}, None, "/ck"), (None, None, None)])
def test_a_root_over_shards_that_are_not_pinned_is_refused_with_what_the_record_says_of_them(tmp_path, checkpoint, pinned, directory):
    comps = _components(tmp_path, {"3,2": _record(checkpoint=checkpoint)})
    assert GP.weights_of_record_binding(comps) == _refusal(
        "component 3,2: weights_of_record.json root was not composed from pinned shards "
        f"(checkpoint.all_shards_pinned={pinned}; MAN-09 W1: only a root over the pinned checkpoint is of record)",
        **{"3,2": {**CARRIED, "checkpoint_dir": directory, "all_shards_pinned": pinned}})


def test_two_roots_for_one_rank_are_refused_with_both_components_and_sixteen_characters_of_each_root(tmp_path):
    rank1 = {"world": 2, "rank": 1}
    records = {"3,2": _record(tp=rank1), "5,0": _record(root_of_record=OTHER), "7,1": _record(tp=rank1, root_of_record=OTHER)}
    assert GP.weights_of_record_binding(_components(tmp_path, records)) == _refusal(
        "components 3,2 and 7,1 (rank 1) carry different weights roots abababababababab != cdcdcdcdcdcdcdcd: one workload "
        "Program cannot span two checkpoints",
        **{"3,2": {**CARRIED, "tp": rank1}, "5,0": {**CARRIED, "root_of_record": OTHER},
           "7,1": {**CARRIED, "tp": rank1, "root_of_record": OTHER}})
