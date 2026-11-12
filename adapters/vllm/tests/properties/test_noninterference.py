"""The world-parametric non-interference record: a run's observed Match arm against its bare arm, at world 1 and 2, and refused above."""
from __future__ import annotations

import json

import pytest

from verity_vllm.properties import noninterference as NI
from verity_vllm.properties.record import REGISTRY, PropertyRecord


def _arm(d, sub, gens, world=None):
    doc = {"requests": [{"request_id": f"r{i}", "prompt_token_ids": [1, 2, i], "generated_token_ids": g} for i, g in enumerate(gens)]}
    if world is not None:
        doc["tp"] = world
    (d / sub).mkdir(parents=True, exist_ok=True)
    (d / sub / "tokens.json").write_text(json.dumps(doc))


@pytest.mark.parametrize("world", [1, 2])
def test_equal_arms_hold_at_world_one_and_two(tmp_path, world):
    for sub in ("capture", "control"):
        _arm(tmp_path, sub, [[5, 6], [7]], None if world == 1 else world)
    doc = NI.record(tmp_path, world, run="row")
    rec = PropertyRecord(doc)
    assert rec.name == "noninterference" and rec.ok and rec.problems() == [] and doc["world"] == world
    assert doc["tokens"] == {"equal": True, "requests": 2, "requests_bare": 2, "differing": []} and set(doc["arms"]) == {"observed", "bare"}


def test_a_differing_request_is_named_and_does_not_hold(tmp_path):
    _arm(tmp_path, "capture", [[5, 6], [7]], 2)
    _arm(tmp_path, "control", [[5, 6], [8]], 2)
    doc = NI.record(tmp_path, 2)
    assert doc["ok"] is False and doc["tokens"]["differing"] == ["r1"] and PropertyRecord(doc).problems() == []


def test_missing_arm_wrong_world_and_boundary_hashes(tmp_path):
    _arm(tmp_path, "capture", [[5]], 2)
    assert NI.record(tmp_path, 2)["problems"] == ["missing control/tokens.json"]
    _arm(tmp_path, "control", [[5]], 2)
    doc = NI.record(tmp_path, 1)
    assert doc["ok"] is False and doc["problems"] == ["capture/tokens.json was taken at world 2, not 1", "control/tokens.json was taken at world 2, not 1"]
    rep = tmp_path / "noninterference.json"
    rep.write_text(json.dumps({"PASS": False, "hash_comparisons": 4, "hash_equal": 3, "problems": ["x"]}))
    with pytest.raises(ValueError):
        NI.record(tmp_path, 2, hashes=rep)
    _arm(tmp_path, "capture", [[5]])
    _arm(tmp_path, "control", [[5]])
    doc = NI.record(tmp_path, 1, hashes=rep)
    assert doc["ok"] is False and doc["boundary_hashes"]["hash_equal"] == 3 and doc["tokens"]["equal"] is True


def test_the_record_is_the_registered_property():
    assert REGISTRY["noninterference"].schema == NI.SCHEMA and NI.WORLDS == (1, 2)


@pytest.mark.parametrize("world", [0, 3, 4])
def test_other_worlds_are_refused(tmp_path, world):
    with pytest.raises(ValueError, match="refused"):
        NI.record(tmp_path, world)
