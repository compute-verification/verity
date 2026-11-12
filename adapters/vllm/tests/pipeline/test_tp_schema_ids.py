"""The ids the TP row's records carry.  Each of `verity_vllm.pipeline.tp.c2`, `verity_vllm.pipeline.tp.commit_replay`,
`verity_vllm.pipeline.tp.emit_ranks`, `verity_vllm.pipeline.tp.fold_match`, `verity_vllm.pipeline.tp.rank_replay` and
`verity_vllm.pipeline.tp.relabel_build` reads its record's id from `verity_vllm.records.schema_ids` under the name it
always had, and `verity_vllm.pipeline.tp.rank_replay`, `verity_vllm.pipeline.tp.links` and
`verity_vllm.pipeline.tp.xrank_dump` write the id their records on disk have."""
from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from verity_vllm.pipeline.tp import (
    c2,
    commit_replay,
    emit_ranks,
    fold_match,
    links,
    rank_replay,
    relabel_build,
    xrank_dump,
)
from verity_vllm.records import schema_ids

#: a module, its own name for its record's id, the constant it reads, and the id as the records on disk have it
NAMED: list[tuple[ModuleType, str, str, str]] = [
    (c2, "SCHEMA", "TP_REPLAY_DEFERRED_SCHEMA", "verity-vllm/tp-replay-deferred/v1"),
    (commit_replay, "RESULT_SCHEMA", "TP_RANK_REPLAY_SCHEMA", "verity-vllm/tp-rank-replay/v1"),
    (emit_ranks, "SCHEMA", "BUILD_RELABEL_SCHEMA", "verity-vllm/build-relabel/v1"),
    (fold_match, "SCHEMA", "TP_FOLD_MATCH_SCHEMA", "verity/tp2/fold-match/v1"),
    (rank_replay, "SCHEMA", "TP_SAMPLED_REPLAY_SCHEMA", "verity-sweep/tp-sampled-replay/v1"),
    (relabel_build, "SCHEMA", "BUILD_RELABEL_GUARD_SCHEMA", "verity-vllm/build-relabel-guard/v1"),
]


@pytest.mark.parametrize("module, name, constant, on_disk", NAMED, ids=[m.__name__.rsplit(".", 1)[-1] for m, *_ in NAMED])
def test_a_modules_name_for_its_records_id_is_the_constant_and_the_id_on_disk(module, name, constant, on_disk):
    assert getattr(module, name) is getattr(schema_ids, constant)
    assert getattr(module, name) == on_disk


def test_a_replay_that_did_not_run_carries_the_sampled_replays_id_and_the_linkages():
    replay, linkage = rank_replay.not_run(1, ValueError("no bundle"), "the trace")
    assert replay["schema"] == "verity-sweep/sampled-replay/v1" and linkage["schema"] == "verity-sweep/boundary-linkage/v1"
    assert (replay["result"], replay["grade"], replay["rank"]) == (None, "not-evaluated", 1)
    assert (linkage["result"], linkage["rank"], linkage["pairs"], linkage["linked"]) == (None, 1, 0, 0)


def test_a_ranks_links_are_written_under_their_id_and_read_back_as_written(tmp_path):
    params = [("weights", "W"), ("prompt", "T"), ("peer0", "T"), ("peer1", "T")]
    program = SimpleNamespace(fn=SimpleNamespace(body=SimpleNamespace(params=params)))
    probed = [{"kind": "all_reduce", "shape": [4, 8], "dtype": "bf16"}, {"kind": "all_gather", "shape": [4, 2], "dtype": "bf16"}]
    applications = [{"collective": "all_reduce", "node": "n0", "module_path": "model.layers.0.mlp (MLP)", "emitted": 3},
                    {"node": "a rule that is no collective's"},
                    {"collective": "all_gather", "node": "n1", "module_path": "model.lm_head (Linear)", "emitted": 1}]
    result = {"tp": {"collectives": {"record": probed}}, "program": {"digest": "d" * 64}}
    wrote = links.write_tp_links(str(tmp_path), result, program, SimpleNamespace(provenance={"applications": applications}), 2, 1,
                                 model_root="model")
    assert wrote == {"n_peer_inputs": 2, "n_prompt_inputs": 1, "file": "tp_links.json"}
    doc = json.loads((tmp_path / "tp_links.json").read_text())
    assert doc["schema"] == "cb-a/tp_links/v2"
    assert (doc["world"], doc["rank"], doc["peer_ranks"], doc["program_digest"]) == (2, 1, [0], "d" * 64)
    assert (doc["prompt_inputs"], doc["peer_inputs"]) == (["prompt"], ["peer0", "peer1"])
    assert [(ln["k"], ln["kind"], ln["node"], ln["module"], ln["peer_input"], ln["peers"]) for ln in doc["links"]] == [
        (0, "all_reduce", "n0", "layers.0.mlp", "peer0", [{"rank": 0, "input": "peer0", "operand": "part_rank0", "collective_k": 0}]),
        (1, "all_gather", "n1", "lm_head", "peer1", [{"rank": 0, "input": "peer1", "operand": "part_rank0", "collective_k": 1}])]
    assert links.read_tp_links(str(tmp_path / "tp_links.json")) == doc


def test_a_ranks_picks_are_saved_under_their_id(tmp_path):
    """A rank that committed no step picks nothing, and its file still says what it is."""
    state = {"com": SimpleNamespace(_layouts={}, step_rows={}), "rank": 1, "world": 2, "rc": None,
             "eng": SimpleNamespace(model=SimpleNamespace(named_modules=lambda: []))}
    wrote = xrank_dump.dump(state, settings={}, out_dir=str(tmp_path), seed_hex="00" * 8, pair=3)
    assert (wrote["rank"], wrote["picks"], Path(wrote["file"]).name) == (1, 0, "xrank_pair3_rank1.pt")
    saved = torch.load(wrote["file"], weights_only=False)
    assert saved == {"schema": "verity/tp2/xrank-picks/v1", "rank": 1, "pair": 3, "seed": "00" * 8, "strata": {}, "picks": [],
                     "per_stratum": 2, "max_rows": 16, "steps_in_layout": 0}
