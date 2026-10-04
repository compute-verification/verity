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


def test_shared_expert_layer_output_is_the_add_after_moe_forward_shared(tmp_path):
    from verity_vllm.observe.events import DispatchEnter, DispatchExit, StepEnd, StepStart, ViewDesc
    from verity_vllm.observe.log import LogWriter

    v = ViewDesc(1, 0, (1, 8), (8, 1), "bfloat16", "cuda:0")
    exp = "model.layers.{}.mlp.experts".format

    def op(i, parent, name, mod):
        return [DispatchEnter(i, parent, 0, "forward", name, "sig", [v], {}, 7, mod), DispatchExit(i, v)]

    evs = [StepStart(0, [("r0", 3)], [42], [3], {"r0": 4}, {"r0": 3}, {"r0": [0]}, 16, [3], [("r0", 3)], [0], 1, 1, 1),
           DispatchEnter(10, None, 0, "forward", "vllm.moe_forward_shared.default", "sig", [v], {}, 7, exp(1)),
           *op(11, 10, "aten.add.Tensor", exp(1)), DispatchExit(10, [v, v]),
           *op(12, None, "aten.mul_.Tensor", exp(1)), *op(13, None, "aten.add.Tensor", exp(1)),
           *op(20, None, "vllm.moe_forward_shared.default", exp(2)),
           *op(30, None, "_C.fused_add_rms_norm.default", "model.norm"), StepEnd(0)]
    with LogWriter(tmp_path / "log.jsonl.gz") as w:
        for ev in evs:
            w.write(ev)
    snaps = {(13, "outs"): "routed+shared", (11, "outs"): "nested", (10, "outs[0]"): "shared", (30, "args[0]"): "final"}
    (tmp_path / "snapshots.json").write_text(json.dumps({"entries": [{"step": 0, "event": e, "path": p, "sha256": h}
                                                                     for (e, p), h in snaps.items()]}))
    got, problems = NI.observer_boundary_hashes(tmp_path)
    assert got["steps"][0]["hidden"] == {1: "routed+shared"} and got["steps"][0]["final"] == "final"
    assert problems == ["step 0: moe_forward_shared id 20 (layer 2) has no shared + routed add after it"]


def test_the_record_is_the_registered_property():
    assert REGISTRY["noninterference"].schema == NI.SCHEMA and NI.WORLDS == (1, 2)


@pytest.mark.parametrize("world", [0, 3, 4])
def test_other_worlds_are_refused(tmp_path, world):
    with pytest.raises(ValueError, match="refused"):
        NI.record(tmp_path, world)
