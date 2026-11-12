"""The replay's served window, cut from a synthetic arm's step trace, and its comparison with a replay."""
from __future__ import annotations

import gzip
import json
import sys
import types
from pathlib import Path

import pytest
import replay as R


def rid(tag: str, i: int) -> str:
    return f"cmpl-{tag}-{i:05d}-0-0123abcd"


def arm(tmp_path: Path, foreign: bool = False) -> Path:
    """Load ``warm`` (one request, step 0), then ``sat1`` (two requests, steps 1 and 2) in one engine's trace."""
    d = tmp_path / "record"
    (d / "server").mkdir(parents=True)
    (d / "sat" / "rep-1").mkdir(parents=True)
    plan = tmp_path / "plan.jsonl"
    plan.write_text("".join(json.dumps({"i": i, "prompt_ids": [i, i + 1]}) + "\n" for i in range(3)))
    (d / "meta.json").write_text(json.dumps({"plan": str(plan), "checkpoint": "ck"}))
    (d / "sat" / "rep-1" / "meta.json").write_text(json.dumps({"tag": "sat1"}))
    (d / "sat" / "rep-1" / "client.jsonl").write_text("".join(
        json.dumps({"id": f"sat1-{i:05d}", "i": i, "output_ids": [7, i]}) + "\n" for i in range(2)))
    w, a, b = rid("warm", 2), rid("sat1", 0), rid("sat1", 1)
    recs = [{"kind": "header", "config": {}},
            {"kind": "arrive", "request": w, "before_step": 0, "sampling": {"max_tokens": 2}, "t": 0.0},
            {"kind": "step", "step": 0, "rows": [[w, 0, 2]], "t": 0.1},
            {"kind": "arrive", "request": a, "before_step": 1, "sampling": {"max_tokens": 2}, "t": 1.0},
            {"kind": "arrive", "request": b, "before_step": 1, "sampling": {"max_tokens": 2}, "t": 1.0},
            {"kind": "step", "step": 1, "rows": [[a, 0, 2], [b, 0, 2]] + ([[w, 2, 1]] if foreign else []), "t": 1.1},
            {"kind": "step", "step": 2, "rows": [[a, 2, 1], [b, 2, 1]], "t": 1.2}]
    with gzip.open(d / "server" / "step_trace.9.jsonl.gz", "wt") as f:
        f.write("".join(json.dumps(r) + "\n" for r in recs))
    return d


def test_window_is_the_loads_steps_from_its_first_arrival(tmp_path: Path):
    w = R.window(arm(tmp_path), "sat", 1)
    assert [s["step"] for s in w["steps"]] == [0, 1]
    assert [(r["engine_id"], r["before_step"], r["prompt_ids"], r["served"]) for r in w["requests"]] == [
        (rid("sat1", 0), 0, [0, 1], [7, 0]), (rid("sat1", 1), 0, [1, 2], [7, 1])]
    assert w["served_s"] == pytest.approx(0.2)


def test_window_reads_the_plan_beside_the_arm_when_the_named_one_isnt_on_this_node(tmp_path: Path):
    d = arm(tmp_path)
    (d / "meta.json").write_text(json.dumps({"plan": "/elsewhere/out/plan.jsonl", "checkpoint": "ck"}))
    assert [r["prompt_ids"] for r in R.window(d, "sat", 1)["requests"]] == [[0, 1], [1, 2]]


def test_window_reads_the_plan_in_the_arm_when_meta_names_none(tmp_path: Path):
    d = arm(tmp_path)
    (d / "meta.json").write_text(json.dumps({"plan": None, "checkpoint": "ck"}))   # serve_tax.py run without --plan
    (tmp_path / "plan.jsonl").rename(d / "plan.jsonl")
    assert [r["prompt_ids"] for r in R.window(d, "sat", 1)["requests"]] == [[0, 1], [1, 2]]
    (d / "plan.jsonl").unlink()
    with pytest.raises(SystemExit, match="no plan at"):
        R.window(d, "sat", 1)


def test_window_refuses_a_step_shared_with_another_load(tmp_path: Path):
    with pytest.raises(SystemExit, match="another load"):
        R.window(arm(tmp_path, foreign=True), "sat", 1)


def test_a_capture_is_of_the_check_the_serve_committed_under_unless_told():
    """The capture's rows are units of the check the served arm's boundary commitment names (its meta.json's, the
    launch line's ``check`` in place of the model's), else of the model's check; one given wins, and an arm with neither
    is refused by name before anything is replayed."""
    served = {"model_id": "glm53", "boundary_rows": {"check": "GreedySelect_v1{V=154880,N=7744}", "inline": True}}
    assert R.capture_template(served) == "GreedySelect_v1{V=154880,N=7744}"
    assert R.capture_template({"model_id": "glm53", "boundary_rows": None}) == R.S.MODELS["glm53"].check
    assert R.capture_template(served, "TokenSelect_v4{V=154880}") == "TokenSelect_v4{V=154880}"
    with pytest.raises(SystemExit, match="--capture-template T"):
        R.capture_template({"model_id": "qwen3-30b"})


def test_the_replays_engine_takes_the_headers_cache_format_and_draft(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A serve with an FP8 cache or speculative decoding is replayed on an engine with them; a header without them builds
    the engine it did."""
    stub_vllm(monkeypatch)
    cfg = {"max_model_len": 4096, "max_num_seqs": 8, "max_num_batched_tokens": 512, "enable_chunked_prefill": True,
           "enable_prefix_caching": True, "async_scheduling": True, "block_size": 16, "num_gpu_blocks": 100}
    plain = R.engine(tmp_path, {"config": cfg}, eager=True)
    assert not {"long_prefill_token_threshold", "kv_cache_dtype", "speculative_config"} & set(plain)
    draft = {"method": "mtp", "num_speculative_tokens": 5}
    got = R.engine(tmp_path, {"config": {**cfg, "long_prefill_token_threshold": 256, "kv_cache_dtype": "fp8_ds_mla",
                                         "speculative": draft}}, eager=True)
    assert {k: v for k, v in got.items() if k not in plain} == {
        "long_prefill_token_threshold": 256, "kv_cache_dtype": "fp8_ds_mla", "speculative_config": draft}
    assert {k: got[k] for k in plain} == plain


def stub_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """vLLM's ``EngineArgs`` as its keyword arguments and ``LLMEngine.from_engine_args`` as the identity."""
    vllm, llm_engine = types.ModuleType("vllm"), types.ModuleType("vllm.v1.engine.llm_engine")
    vllm.EngineArgs = lambda **kw: kw
    llm_engine.LLMEngine = types.SimpleNamespace(from_engine_args=lambda args: args)
    for name, mod in (("vllm", vllm), ("vllm.v1", types.ModuleType("vllm.v1")),
                      ("vllm.v1.engine", types.ModuleType("vllm.v1.engine")), ("vllm.v1.engine.llm_engine", llm_engine)):
        monkeypatch.setitem(sys.modules, name, mod)


#: a header's config of a served engine whose launch left the context limit to vLLM, which fitted 1,008 to 64 blocks of 16
FITTED = {"max_model_len": 1008, "max_model_len_auto": True, "max_num_seqs": 8, "max_num_batched_tokens": 512,
          "enable_chunked_prefill": True, "enable_prefix_caching": True, "async_scheduling": True, "block_size": 16,
          "num_gpu_blocks": 64}


def test_the_replays_engine_is_launched_as_the_served_one_where_vllm_fitted_the_context_limit(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """vLLM builds the model before it fits the limit to the KV pool, so an engine launched at the fitted number is built at
    another length than the served one: the replay's is launched at -1 on the header's block count. A header of a launch that
    stated its length builds the engine it did."""
    stub_vllm(monkeypatch)
    stated = {k: v for k, v in FITTED.items() if k != "max_model_len_auto"}
    got, plain = R.engine(tmp_path, {"config": FITTED}, eager=True), R.engine(tmp_path, {"config": stated}, eager=True)
    assert (got["max_model_len"], got["num_gpu_blocks_override"]) == (-1, 64)
    assert (plain["max_model_len"], plain["num_gpu_blocks_override"]) == (1008, 64)
    assert {k: v for k, v in got.items() if k != "max_model_len"} == {k: v for k, v in plain.items() if k != "max_model_len"}


class Scheduler:
    """A scheduler that holds a context limit and no request, in an engine the step trace finds it in."""

    def __init__(self, max_model_len: int):
        self.max_model_len, self.requests = max_model_len, {}
        self.cache_config = types.SimpleNamespace(block_size=16, num_gpu_blocks=64)
        for name in ("add_request", "schedule", "update_from_output", "finish_requests"):
            setattr(self, name, lambda *a, **kw: None)

    def engine(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(engine_core=types.SimpleNamespace(scheduler=self), has_unfinished_requests=lambda: False)


def test_the_replay_refuses_an_engine_whose_context_limit_is_not_the_served_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Before it adds a request: an engine on the header's block count that vLLM fitted another limit to stops a request
    at another length than the served engine did."""
    monkeypatch.setitem(sys.modules, "verity_vllm.engine.sampler_capture", types.ModuleType("verity_vllm.engine.sampler_capture"))
    w = {"header": {"config": FITTED}, "requests": []}
    monkeypatch.setattr(R, "engine", lambda *a, **kw: Scheduler(1008).engine())
    assert R.replay(w, tmp_path, eager=True)["steps"] == []
    monkeypatch.setattr(R, "engine", lambda *a, **kw: Scheduler(992).engine())
    with pytest.raises(SystemExit, match="context limit of 992, the served engine's was 1008"):
        R.replay(w, tmp_path, eager=True)


def test_compare_counts_steps_rows_and_tokens_that_differ(tmp_path: Path):
    w = R.window(arm(tmp_path), "sat", 1)
    same = {"steps": [dict(s) for s in w["steps"]], "tokens": {r["engine_id"]: list(r["served"]) for r in w["requests"]}}
    assert R.compare(w, same)["bit_exact"]
    other = {"steps": [dict(w["steps"][0]), {"step": 1, "rows": [[rid("sat1", 0), 2, 1]]}],
             "tokens": {**same["tokens"], rid("sat1", 1): [7, 9]}}
    c = R.compare(w, other)
    assert (c["steps_differing"], c["rows_differing"], c["first_differing_step"]) == (1, 1, 1)
    assert (c["requests_tokens_differing"], c["token_positions_differing"], c["bit_exact"]) == (1, 1, False)
    assert c["requests_differing"] == [{"request": rid("sat1", 1), "tokens": 2, "tokens_replayed": 2, "first_position": 1,
                                        "positions": 1}]


def test_token_diff_counts_a_shorter_replay_from_where_it_stops():
    assert R.token_diff({"a": [1, 2, 3], "b": [4]}, {"a": [1, 2], "b": [4]}) == [
        {"request": "a", "tokens": 3, "tokens_replayed": 2, "first_position": 2, "positions": 1}]


def test_logprob_diff_finds_each_requests_first_differing_step():
    steps = [{"step": 0, "emitted": {"a": 1}}, {"step": 1, "emitted": {"a": 1, "b": 1}}, {"step": 2, "emitted": {"a": 1, "b": 1}}]
    d = R.logprob_diff({"a": [-0.5, -1.0, -2.0], "b": [-0.1, -0.2]}, {"a": [-0.5, -1.0, -2.5], "b": [-0.1, -0.3]}, steps)
    assert (d["requests_logprobs_differing"], d["first_step"]) == (2, 2)
    assert [(r["request"], r["first_position"], r["step"], r["want"], r["got"]) for r in d["requests_differing"]] == [
        ("a", 2, 2, -2.0, -2.5), ("b", 1, 2, -0.2, -0.3)]
    assert R.logprob_diff({"a": [-0.5]}, {"a": [-0.5]}, steps)["requests_logprobs_differing"] == 0


def test_replay_tokens_round_trip(tmp_path: Path):
    tokens = {rid("sat1", 1): [7, 9], rid("sat1", 0): [7, 0]}
    R.write_tokens(tmp_path, tokens)
    assert R.read_tokens(tmp_path) == tokens
