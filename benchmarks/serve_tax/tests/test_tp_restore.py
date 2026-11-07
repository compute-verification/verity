"""`tp_restore`'s reading of a served run and its comparison with a replay, on synthetic records (stdlib only)."""
from __future__ import annotations

import json

import pytest

import tp_restore as T


def arrive(n: int, before: int, **sampling) -> dict:
    return {"kind": "arrive", "request": f"cmpl-tpr-{n:05d}-0", "before_step": before, "sampling": {"max_tokens": 4, **sampling}}


def step(k: int, emitted: dict[int, int], rows=None) -> dict:
    return {"kind": "step", "step": k, "rows": rows if rows is not None else [[f"cmpl-tpr-{n:05d}-0", 0, 1] for n in emitted],
            "emitted": {f"cmpl-tpr-{n:05d}-0": c for n, c in emitted.items()}}


def test_client_ids_come_out_of_engine_ids_with_or_without_a_random_suffix():
    assert T.client_id("cmpl-tpr-00012-0") == "tpr-00012"
    assert T.client_id("cmpl-tpr-00012-0-1a2b3c4d") == "tpr-00012"
    with pytest.raises(ValueError, match="no client id"):
        T.client_id("cmpl-other-3-0")


def test_the_arrivals_between_two_edges_carry_their_prompts_and_sampling():
    recs = [arrive(1, 0), arrive(2, 3, temperature=1.0, top_p=0.95), arrive(3, 5), arrive(4, 7)]
    plan = {i: {"i": i, "prompt_ids": [i] * 3} for i in range(1, 5)}
    client = {f"tpr-{i:05d}": {"id": f"tpr-{i:05d}", "i": i, "output_ids": []} for i in range(1, 5)}
    got = T.arrivals(recs, plan, client, 3, 7)
    assert [g["engine_id"] for g in got] == ["cmpl-tpr-00002-0", "cmpl-tpr-00003-0"]
    assert got[0] == {"engine_id": "cmpl-tpr-00002-0", "before_step": 3, "prompt_ids": [2, 2, 2], "max_tokens": 4,
                      "sampling": {"max_tokens": 4, "temperature": 1.0, "top_p": 0.95}}


def test_a_requests_tokens_between_two_edges_start_where_its_earlier_steps_left_off():
    recs = [step(k, {1: 1, **({2: 2} if k >= 2 else {})}) for k in range(6)]
    client = {"tpr-00001": {"output_ids": list(range(10, 16))}, "tpr-00002": {"output_ids": list(range(20, 28))}}
    assert T.interval_tokens(recs, client, 2, 4) == {"cmpl-tpr-00001-0": [12, 13], "cmpl-tpr-00002-0": [20, 21, 22, 23]}
    assert T.interval_tokens(recs, client, 4, 6) == {"cmpl-tpr-00001-0": [14, 15], "cmpl-tpr-00002-0": [24, 25, 26, 27]}


def test_the_first_differing_step_is_by_index_and_a_missing_one_counts():
    a = [step(k, {1: 1}) for k in range(3, 6)]
    assert T.first_step_difference(a, [dict(s) for s in a]) is None
    b = [dict(s) for s in a]
    b[1] = step(4, {1: 1, 2: 1})
    assert T.first_step_difference(a, b)["step"] == 4
    assert T.first_step_difference(a, a[:2]) == {"step": 5, "served": a[2]["rows"], "replayed": None}


def test_the_first_differing_token_is_the_earliest_position_over_requests():
    want = {"a": [1, 2, 3, 4], "b": [5, 6, 7], "c": [8]}
    assert T.first_token_difference(want, {k: list(v) for k, v in want.items()}) is None
    got = {"a": [1, 2, 3, 9], "b": [5, 0, 7], "c": [8]}
    assert T.first_token_difference(want, got) == {"request": "b", "position": 1, "tokens_served": 3, "tokens_replayed": 3,
                                                  "requests_differing": 2}
    assert T.first_token_difference({"a": [1, 2]}, {"a": [1]})["position"] == 1
    assert T.first_token_difference({"a": [1]}, {})["position"] == 0


def test_an_edges_first_differing_block_reads_the_two_digest_files(tmp_path):
    layers = [{"name": "l0"}, {"name": "l1"}]
    ids = [0, 3, 5]
    digests = bytes(range(16)) * 6
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "digests.rank0.bin").write_bytes(digests)
    other = bytearray(digests)
    other[16 * 4] ^= 1                                       # layer 1, its second block (id 3)
    (tmp_path / "b" / "digests.rank0.bin").write_bytes(bytes(other))
    served = {"ids": ids, "layers": layers, "ranks": [{"layer_digests": ["x", "y"]}]}
    assert T.edge_difference(served, {"ids": ids, "ranks": [{"layer_digests": ["x", "y"]}]}, tmp_path / "a", tmp_path / "b") is None
    got = T.edge_difference(served, {"ids": ids, "ranks": [{"layer_digests": ["x", "z"]}]}, tmp_path / "a", tmp_path / "b")
    assert got == {"layer": 1, "name": "l1", "block": 3, "index": 1, "blocks_differing": 1}
    assert T.edge_difference(served, {"ids": [0, 3, 6], "ranks": [{}]}, tmp_path / "a", tmp_path / "b") == {
        "blocks": "differ", "served_only": [5], "replayed_only": [6]}


def test_the_served_argv_is_the_record_arms_with_the_context_and_tp_given_last():
    from argparse import Namespace
    from pathlib import Path

    argv = T.served_argv(Namespace(model_id="glm53", max_model_len=32768, tp=None), Path("/ck"), 0)
    assert argv[1:3] == ["-m", "vllm.entrypoints.openai.api_server"] and argv[argv.index("--model") + 1] == "/ck"
    assert argv[argv.index("--tensor-parallel-size") + 1] == "8" and argv[-2:] == ["--max-model-len", "32768"]
    argv = T.served_argv(Namespace(model_id="moonlight", max_model_len=None, tp=2), Path("/ck"), 0)
    assert argv[-2:] == ["--tensor-parallel-size", "2"] and argv.count("--max-model-len") == 1


def test_a_restored_engines_settings_that_differ_are_named_with_both_values():
    served = {"max_num_batched_tokens": 8192, "max_num_seqs": 256, "num_gpu_blocks": 900, "async_scheduling": True}
    assert T.config_difference(served, dict(served)) == {}
    assert T.config_difference(served, served | {"max_num_batched_tokens": 16384, "policy": "fcfs"}) == {
        "max_num_batched_tokens": [8192, 16384], "policy": [None, "fcfs"]}


def test_the_plugin_site_declares_the_edges_beside_the_step_trace(tmp_path):
    T.edge_site(tmp_path)
    eps = sorted(p.read_text() for p in tmp_path.glob("*.dist-info/entry_points.txt"))
    assert eps == ["[vllm.general_plugins]\nverity_serve_tax = servetax_plugin:register\n",
                   "[vllm.general_plugins]\nverity_tp_edge = tp_edge:register\n"]


def test_each_edges_record_is_copied_out_of_the_scratch_directory(tmp_path):
    for e in (4, 9):
        d = tmp_path / "scratch" / f"edge-{e:06d}"
        d.mkdir(parents=True)
        (d / "edge.json").write_text(json.dumps({"edge": e, "current_step": e, "trace_step": e - 1, "running": 2, "waiting": 1,
                                                 "pool": {}, "check": {}, "seconds": {}, "ids": [0, 1, 2]}))
        (d / "digests.rank0.bin").write_bytes(b"x")
    got = T.copy_edges(tmp_path / "scratch", tmp_path / "out")
    assert [g["edge"] for g in got] == [4, 9] and got[0]["trace_step"] == 3 and got[0]["blocks"] == 3
    assert got[1]["dir"] == str(tmp_path / "scratch" / "edge-000009")
    assert sorted(p.name for p in (tmp_path / "out").rglob("*")) == ["edge-000004", "edge-000009", "edge.json", "edge.json"]
