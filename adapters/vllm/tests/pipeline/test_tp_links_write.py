"""`verity_vllm.pipeline.tp.links.write_tp_links`: a rank's `tp_links.json` at any world.  Each probed collective is bound to one
application of a collective rule, in call order; a Build whose rules applied to another number of collectives than the probe
recorded writes no file and says both counts.  Past two ranks a collective has one peer input per other rank, in rank order, and
no two-rank field.  The Program, the probe's record and the rules' applications are stand-ins."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from verity_vllm.pipeline.tp import links as LINKS

PROBED = [{"kind": "all_reduce", "shape": [4, 8], "dtype": "bf16"}, {"kind": "all_gather", "shape": [4, 2], "dtype": "bf16"}]
REDUCE = {"collective": "all_reduce", "node": "n0", "module_path": "inner.model.layers.0.mlp (MLP)", "emitted": 3}
GATHER = {"collective": "all_gather", "node": "n1", "module_path": "inner.model.lm_head (Linear)", "emitted": 1}
OTHER = {"node": "a rule that is no collective's", "module_path": "inner.model.norm (RMSNorm)"}


def _write(tmp_path: Path, *, world: int, rank: int, user: list[str], applications: list[dict], probed: list[dict] = PROBED,
           model_root: str | None = "inner.model") -> tuple[dict, dict]:
    """`write_tp_links` for a Program whose parameters are the weights and then ``user``: (what it returned, the file it wrote)."""
    params = [("weights", "W"), *((n, "T") for n in user)]
    program = SimpleNamespace(fn=SimpleNamespace(body=SimpleNamespace(params=params)))
    res = {"tp": {"collectives": {"record": probed}}, "program": {"digest": "d" * 64}}
    tmp_path.mkdir(exist_ok=True)
    wrote = LINKS.write_tp_links(str(tmp_path), res, program, SimpleNamespace(provenance={"applications": applications}), world, rank,
                                 model_root=model_root)
    return wrote, json.loads((tmp_path / "tp_links.json").read_text())


# ------------------------------------------------------------------------------------------------ a count that does not match

@pytest.mark.parametrize("applications, n", [([REDUCE], 1), ([OTHER, REDUCE, OTHER], 1), ([], 0), ([REDUCE, GATHER, REDUCE], 3)],
                         ids=["one of two", "one of two among other rules", "none", "one more"])
def test_a_build_whose_collective_rules_applied_to_another_number_than_the_probe_recorded_writes_no_file(tmp_path, applications, n):
    with pytest.raises(RuntimeError) as e:
        _write(tmp_path, world=2, rank=1, user=["prompt", "peer0", "peer1"], applications=applications)
    assert str(e.value) == f"tp_links: {n} collective rule applications vs 2 probed collectives"
    assert list(tmp_path.iterdir()) == []


def test_a_rule_that_is_no_collectives_is_not_counted(tmp_path):
    wrote, doc = _write(tmp_path, world=2, rank=1, user=["prompt", "peer0", "peer1"], applications=[OTHER, REDUCE, OTHER, GATHER, OTHER])
    assert wrote == {"n_peer_inputs": 2, "n_prompt_inputs": 1, "file": "tp_links.json"}
    assert [(ln["k"], ln["kind"], ln["node"]) for ln in doc["links"]] == [(0, "all_reduce", "n0"), (1, "all_gather", "n1")]


# ------------------------------------------------------------------------------------------------ past two ranks

USER4 = ["prompt", "positions", "p0", "p1", "p2", "p3", "p4", "p5"]


def test_past_two_ranks_a_collective_has_one_peer_input_for_each_other_rank_in_rank_order(tmp_path):
    wrote, doc = _write(tmp_path, world=4, rank=2, user=USER4, applications=[REDUCE, GATHER])
    assert wrote == {"n_peer_inputs": 6, "n_prompt_inputs": 2, "file": "tp_links.json"}
    assert (doc["world"], doc["rank"], doc["peer_ranks"]) == (4, 2, [0, 1, 3])
    assert (doc["user_inputs"], doc["prompt_inputs"], doc["peer_inputs"]) == (USER4, ["prompt", "positions"], ["p0", "p1", "p2", "p3", "p4", "p5"])
    assert [ln["peers"] for ln in doc["links"]] == [
        [{"rank": 0, "input": "p0", "operand": "part_rank0", "collective_k": 0}, {"rank": 1, "input": "p1", "operand": "part_rank1", "collective_k": 0},
         {"rank": 3, "input": "p2", "operand": "part_rank3", "collective_k": 0}],
        [{"rank": 0, "input": "p3", "operand": "part_rank0", "collective_k": 1}, {"rank": 1, "input": "p4", "operand": "part_rank1", "collective_k": 1},
         {"rank": 3, "input": "p5", "operand": "part_rank3", "collective_k": 1}]]
    assert [(ln["own_operand"], ln["peer_collective_k"], ln["shape"], ln["dtype"], ln["emitted"]) for ln in doc["links"]] == [
        ("part_rank2", 0, [4, 8], "bf16", 3), ("part_rank2", 1, [4, 2], "bf16", 1)]


def test_only_a_two_rank_link_carries_the_two_rank_fields(tmp_path):
    two = _write(tmp_path / "two", world=2, rank=0, user=["prompt", "peer0", "peer1"], applications=[REDUCE, GATHER])[1]
    four = _write(tmp_path / "four", world=4, rank=2, user=USER4, applications=[REDUCE, GATHER])[1]
    fields = {"peer_input", "peer_rank", "peer_operand"}
    assert [{k: ln[k] for k in sorted(fields)} for ln in two["links"]] == [
        {"peer_input": "peer0", "peer_operand": "part_rank1", "peer_rank": 1}, {"peer_input": "peer1", "peer_operand": "part_rank1", "peer_rank": 1}]
    assert all(not fields & set(ln) for ln in four["links"])
    assert two["binding"].endswith(" (world 2: AllReduce2/AllGather2, the recorded identities)") and "world 2:" not in four["binding"]


def test_a_rank_with_no_collective_writes_an_empty_list_and_every_input_is_the_prompts(tmp_path):
    wrote, doc = _write(tmp_path, world=4, rank=0, user=["prompt", "positions"], applications=[OTHER], probed=[])
    assert wrote == {"n_peer_inputs": 0, "n_prompt_inputs": 2, "file": "tp_links.json"}
    assert (doc["links"], doc["peer_inputs"], doc["prompt_inputs"]) == ([], [], ["prompt", "positions"])


def test_the_weights_are_no_user_input(tmp_path):
    doc = _write(tmp_path, world=2, rank=1, user=["prompt", "peer0", "peer1"], applications=[REDUCE, GATHER])[1]
    assert "weights" not in doc["user_inputs"] and doc["program_digest"] == "d" * 64 and doc["schema"] == LINKS.TP_LINKS_SCHEMA


# ------------------------------------------------------------------------------------------------ a link's module

@pytest.mark.parametrize("root, modules", [
    ("inner.model", ["layers.0.mlp", "lm_head"]),
    (None, ["inner.model.layers.0.mlp", "inner.model.lm_head"]),
    ("", ["inner.model.layers.0.mlp", "inner.model.lm_head"]),
    ("inner", ["model.layers.0.mlp", "model.lm_head"]),
    ("inner.mod", ["inner.model.layers.0.mlp", "inner.model.lm_head"]),
    ("inner.model.lm_head", ["inner.model.layers.0.mlp", ""]),
], ids=["the recorded root", "no root", "an empty root", "a shorter root", "a root that is no dotted head", "a root that is the module"])
def test_a_links_module_is_its_path_less_the_recorded_model_root_and_the_class(tmp_path, root, modules):
    doc = _write(tmp_path, world=2, rank=1, user=["prompt", "peer0", "peer1"], applications=[REDUCE, GATHER], model_root=root)[1]
    assert [ln["module"] for ln in doc["links"]] == modules and doc["model_root"] == root
    assert [ln["module_path"] for ln in doc["links"]] == [REDUCE["module_path"], GATHER["module_path"]], "the path as the rule wrote it"


@pytest.mark.parametrize("path, want", [(None, None), (7, None), ("model.norm", "norm"), (" model.norm  (RMSNorm)", "norm"), ("model (Root)", ""),
                                        ("model.a(b).c (Cls)", "a(b).c"), ("outer.model.norm (RMSNorm)", "outer.model.norm")],
                         ids=["no path", "a number", "no class", "spaces around the path", "the root itself", "a parenthesis inside the path",
                              "the root's name further in"])
def test_a_module_path_in_the_models_vocabulary(path: Any, want):
    assert LINKS.module_in_model_vocabulary(path, "model") == want


def test_an_application_with_no_module_path_has_no_module(tmp_path):
    bare = {"collective": "all_reduce", "node": "n0"}
    doc = _write(tmp_path, world=2, rank=1, user=["prompt", "peer0"], applications=[bare], probed=PROBED[:1])[1]
    assert [(ln["module_path"], ln["module"], ln["emitted"]) for ln in doc["links"]] == [(None, None, None)]
