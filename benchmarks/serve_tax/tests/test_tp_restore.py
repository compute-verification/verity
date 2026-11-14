"""`tp_restore`'s reading of a served run and its comparison with a replay, on synthetic records (stdlib only)."""
from __future__ import annotations

import gzip
import json
from types import SimpleNamespace

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


def test_the_served_trace_is_the_one_under_serves_server_directory(tmp_path):
    (tmp_path / "server").mkdir()
    (tmp_path / "server" / "step_trace.7.jsonl").write_text('{"kind": "header"}\n{"kind": "step", "i": 0}\n')
    assert T.served_trace(tmp_path) == [{"kind": "header"}, {"kind": "step", "i": 0}]


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


def test_the_served_argv_is_the_record_arms_with_the_context_tp_and_pool_given():
    from argparse import Namespace
    from pathlib import Path

    argv = T.served_argv(Namespace(model_id="glm53", max_model_len=32768, tp=None), Path("/ck"), 0)
    assert argv[1:3] == ["-m", "vllm.entrypoints.openai.api_server"] and argv[argv.index("--model") + 1] == "/ck"
    assert argv[argv.index("--tensor-parallel-size") + 1] == "8" and argv[-2:] == ["--max-model-len", "32768"]
    argv = T.served_argv(Namespace(model_id="moonlight", max_model_len=None, tp=2), Path("/ck"), 0)
    assert argv[-2:] == ["--tensor-parallel-size", "2"] and argv.count("--max-model-len") == 1
    assert "--num-gpu-blocks-override" not in argv
    argv = T.served_argv(Namespace(model_id="moonlight", max_model_len=None, tp=None, num_gpu_blocks=4096), Path("/ck"), 0)
    assert argv[argv.index("--num-gpu-blocks-override") + 1] == "4096" and "--kv-cache-dtype" not in argv
    # a cache format goes on the served argv, which is what the restore's engine is built from (`engine_args`)
    argv = T.served_argv(Namespace(model_id="glm53", max_model_len=None, tp=None, kv_cache_dtype="fp8"), Path("/ck"), 0)
    assert argv[-2:] == ["--kv-cache-dtype", "fp8"]
    assert T.served_argv(Namespace(model_id="glm53", max_model_len=None, tp=None, kv_cache_dtype=None), Path("/ck"), 0) == argv[:-2]


def test_a_restored_engines_settings_that_differ_are_named_with_both_values():
    served = {"max_num_batched_tokens": 8192, "max_num_seqs": 256, "num_gpu_blocks": 900, "async_scheduling": True}
    assert T.config_difference(served, dict(served)) == {}
    assert T.config_difference(served, served | {"max_num_batched_tokens": 16384, "policy": "fcfs"}) == {
        "max_num_batched_tokens": [8192, 16384], "policy": [None, "fcfs"]}


def test_a_restore_is_exact_on_blocks_and_generators_and_on_seeds_only_when_rows_were_restored():
    rng = {"runner": "v2", "numpy": [1], "cuda": "ab", "seeds": {"r1": [7, False]}}
    served = [{"layer_digests": ["a", "b"], "rng": rng}, {"layer_digests": ["a", "b"], "rng": rng}]
    no_rows = [{"layer_digests": ["a", "b"], "rng": rng | {"seeds": {}}}] * 2
    assert T.restored_exact(served, no_rows, seeds=False) == {"kv": True, "generators": True}
    assert T.restored_exact(served, no_rows, seeds=True) == {"kv": True, "generators": True, "seeds": False}
    moved = [served[0], {"layer_digests": ["a", "c"], "rng": rng | {"cuda": "ac"}}]
    assert T.restored_exact(served, moved, seeds=True) == {"kv": False, "generators": False, "seeds": True}


def test_the_served_autotune_results_go_where_the_restored_engines_config_hash_reads_them(tmp_path):
    served = tmp_path / "served"
    assert T.carry_autotune(served, tmp_path / "restored", "cfg-b")["served"] is None
    src = T.autotune_file(served, "cfg-a")
    src.parent.mkdir(parents=True)
    src.write_text('{"op": "tactic 3"}')
    got = T.carry_autotune(served, tmp_path / "restored", "cfg-b")
    assert got["same_config"] is False and T.autotune_file(tmp_path / "restored", "cfg-b").read_text() == '{"op": "tactic 3"}'
    T.autotune_file(served, "cfg-c").parent.mkdir()
    T.autotune_file(served, "cfg-c").write_text("{}")
    with pytest.raises(SystemExit, match="2 autotune files"):
        T.carry_autotune(served, tmp_path / "restored", "cfg-b")


def test_a_share_of_the_plan_gets_long_prompts_joined_from_the_plans_own():
    plan = [{"i": i, "prompt_ids": [i] * 5, "prompt_len": 5, "max_tokens": 4} for i in range(40)]
    got = T.lengthen(plan, 0.25, 12, 20, seed=3)
    long = [r for r in got if r.get("kind") == "long"]
    assert 0 < len(long) < 40 and [r["i"] for r in got] == list(range(40))
    for r in long:
        n = r["i"]
        assert 12 <= r["prompt_len"] == len(r["prompt_ids"]) <= 20 and r["max_tokens"] == 4
        assert r["prompt_ids"][:5] == [(n + 1) % 40] * 5
    assert got == T.lengthen(plan, 0.25, 12, 20, seed=3) and T.lengthen(plan, 0.0, 12, 20, seed=3) == plan


def test_the_plugin_site_declares_the_edges_beside_the_step_trace(tmp_path):
    T.edge_site(tmp_path, T.S.ARMS["record"])
    eps = sorted(p.read_text() for p in tmp_path.glob("*.dist-info/entry_points.txt"))
    assert eps == ["[vllm.general_plugins]\nverity_serve_tax = servetax_plugin:register\n",
                   "[vllm.general_plugins]\nverity_tp_edge = tp_edge:register\n"]


def test_each_edges_record_is_copied_out_of_the_scratch_directory(tmp_path):
    for e in (4, 9):
        d = tmp_path / "scratch" / f"edge-{e:06d}"
        d.mkdir(parents=True)
        (d / "edge.json").write_text(json.dumps({"edge": e, "current_step": e, "trace_step": e - 1, "running": 2, "waiting": 1,
                                                 "pool": {}, "check": {}, "seconds": {}, "ids": [0, 1, 2],
                                                 "ranks": [{"layer_digests": ["a", f"b{e}"]}]}))
        (d / "digests.rank0.bin").write_bytes(b"x")
    (tmp_path / "scratch" / "outputs.jsonl").write_text('{"step":3,"out":{}}\n')
    got = T.copy_edges(tmp_path / "scratch", tmp_path / "out")
    assert [g["edge"] for g in got] == [4, 9] and got[0]["trace_step"] == 3 and got[0]["blocks"] == 3
    assert got[1]["dir"] == str(tmp_path / "scratch" / "edge-000009")
    assert got[0]["files"]["digests.rank0.bin"] == 1 and len(got[0]["kv_root"]) == 128
    assert got[0]["kv_root"] != got[1]["kv_root"]
    assert sorted(p.name for p in (tmp_path / "out").rglob("*")) == ["edge-000004", "edge-000009", "edge.json", "edge.json",
                                                                    "outputs.jsonl"]


def test_a_production_plans_requests_send_their_own_sampling_and_a_share_of_the_sampled_go_unseeded(monkeypatch):
    corpus = SimpleNamespace(request_seed=lambda rid: int(rid[4:]) + 1000, SAMPLING={"production": {}})
    monkeypatch.setattr(T.S, "corpus", lambda: corpus)
    greedy = {"i": 0, "prompt_ids": [1, 2], "max_tokens": 8, "seeded": False, "sampling": {"temperature": 0.0}}
    sampled = {"i": 1, "prompt_ids": [3], "max_tokens": 4, "seeded": True, "sampling": {"temperature": 0.7}}
    ids = [f"tpr-{i:05d}" for i in range(400)]
    got = [T.request_class(sampled, rid, 0.5) for rid in ids]
    assert set(got) == {"seeded", "unseeded"} and 140 < got.count("unseeded") < 260
    assert {T.request_class(sampled, rid, 0.0) for rid in ids} == {"seeded"}
    assert {T.request_class(sampled, rid, 1.0) for rid in ids} == {"unseeded"}
    assert T.request_class(greedy, ids[0], 0.5) == "greedy"
    assert T.request_class(sampled, ids[0], 0.5, "production") == "production"

    body = T.own_body(lambda served, r, rid, sampling: {"model": served, "real": sampling}, 0.5)
    seeded = next(rid for rid in ids if T.request_class(sampled, rid, 0.5) == "seeded")
    unseeded = next(rid for rid in ids if T.request_class(sampled, rid, 0.5) == "unseeded")
    b = body("glm", sampled, seeded, "production")
    assert b["temperature"] == 0.7 and b["logprobs"] == 0 and b["seed"] == int(seeded[4:]) + 1000
    assert b["prompt"] == [3] and b["max_tokens"] == 4 and b["user"] == seeded and "ignore_eos" not in b
    assert "seed" not in body("glm", sampled, unseeded, "production")
    assert "seed" not in body("glm", greedy, seeded, "production")
    assert body("glm", {"prompt_ids": []}, seeded, "production") == {"model": "glm", "real": "production", "logprobs": 0}
    named = T.own_body(lambda *a: {}, 0.5, "production")("glm", sampled, seeded, "production")
    assert "temperature" not in named and "seed" not in named and "ignore_eos" not in named and named["logprobs"] == 0


def test_step_outputs_are_read_between_two_edges_and_compared_per_sampling_class(tmp_path):
    a, b = "cmpl-tpr-00001-0", "cmpl-tpr-00002-0"
    served = {3: {a: [[5], [1]], b: [[6], [2]]}, 4: {a: [[7], [3]]}}
    replayed = {3: {a: [[5], [1]], b: [[6], [2]]}, 4: {a: [[7], [4]]}}
    assert T.by_request(served) == {a: ([5, 7], [1, 3]), b: ([6], [2])}
    assert T.class_outputs(served, replayed, {"tpr-00001": "seeded", "tpr-00002": "greedy"}) == {
        "greedy": {"requests": 1, "tokens": 1, "differing": 0}, "seeded": {"requests": 1, "tokens": 2, "differing": 1}}
    plan = {1: {"kind": "long", "prompt_ids": [0] * 9}, 2: {"kind": "sharegpt", "prompt_ids": [0]}}
    first = {a: {"prompt": 9, "output": 3, "computed": 11, "state": "decoding", "blocks": [1, 2]}}
    assert T.differing_requests(served, replayed, plan, {"tpr-00001": 13}, first) == [
        {"id": "tpr-00001", "kind": "long", "prompt_len": 9, "keys": 13, "held": "decoding"}]
    assert T.differing_requests(served, served, plan, {}, first) == []
    p = tmp_path / "outputs.jsonl"
    p.write_text("".join(json.dumps({"step": s, "out": {a: [[s], None]}}) + "\n" for s in range(2, 7)))
    assert T.read_outputs(p, 3, 5) == {3: {a: [[3], None]}, 4: {a: [[4], None]}}
    assert T.read_outputs(tmp_path / "none.jsonl", 0, 9) == {}
    (tmp_path / "outputs.jsonl.gz").write_bytes(gzip.compress(p.read_bytes()))
    p.unlink()
    assert T.read_outputs(p, 3, 5) == {3: {a: [[3], None]}, 4: {a: [[4], None]}}


def test_requests_fall_at_most_or_past_the_indexers_top_k_by_the_most_keys_they_attend_over():
    held, done, new, waiting = "cmpl-tpr-00001-0", "cmpl-tpr-00002-0", "cmpl-tpr-00003-0", "cmpl-tpr-00004-0"
    k = T.TOPK_KEYS
    served = {3: {held: [[5], None], done: [[6], None], new: [[7], None]}, 4: {done: [[8], None], new: [[9], None]},
              5: {done: [[10], None]}}
    first = {held: {"prompt": k - 3, "output": 2, "computed": k - 2, "state": "decoding", "blocks": [1]},
             done: {"prompt": k - 3, "output": 2, "computed": k - 2, "state": "decoding", "blocks": [2]},
             waiting: {"prompt": 9 * k, "output": 0, "computed": 0, "state": "waiting", "blocks": []}}
    far = {held: {"prompt": k - 3, "output": 3, "computed": k - 1, "state": "decoding", "blocks": [1, 3]},
           new: {"prompt": 4 * k, "output": 2, "computed": 4 * k + 1, "state": "decoding", "blocks": [3, 4]},
           waiting: first[waiting]}
    plan = {3: {"prompt_ids": [0] * (4 * k)}}
    keys = T.max_keys(served, first, far, plan)
    assert keys == {"tpr-00001": k - 1, "tpr-00002": k + 1, "tpr-00003": 4 * k + 1, "tpr-00004": 0}
    by = {c: T.keys_class(n) for c, n in keys.items()}
    assert by == {"tpr-00001": "keys<=2048", "tpr-00002": "keys>2048", "tpr-00003": "keys>2048", "tpr-00004": "keys<=2048"}
    assert T.held_classes(first, by) == {"keys<=2048": {"prefilling": 0, "decoding": 1, "waiting": 1},
                                         "keys>2048": {"prefilling": 0, "decoding": 1, "waiting": 0}}
    assert T.block_classes([1, 3, 4, 9], far, by) == {"free": 1, "keys<=2048": 2, "keys>2048": 2}


def test_a_snapshots_bytes_are_its_kv_dump_its_state_and_each_ranks_rows():
    files = {"kv.rank0/000.bin": 100, "kv.rank0/layers.json": 10, "rank-0.pkl": 5, "rank-1.pkl": 7, "state.pkl": 3,
             "digests.rank0.bin": 9, "edge.json": 4}
    assert T.snapshot_bytes(files) == {"kv": 110, "state": 3, "ranks": {0: 5, 1: 7}, "per_rank": 117}
