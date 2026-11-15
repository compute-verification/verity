"""The restore's reading of a served restore arm (edges, states, the arrivals it replays, the served outputs) and its
comparisons, on a synthetic arm."""
from __future__ import annotations

import gzip
import json
import pickle
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import boundary as B
import pytest
import restore as R


class Request:
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id


def rid(tag: str, i: int) -> str:
    return f"cmpl-{tag}-{i:05d}-0-0123abcd"


def arm(tmp_path: Path, gz: bool = False) -> tuple[Path, Path]:
    """A restore arm whose edge 0 holds ``a`` and ``b`` at step 2 and whose edge 1 holds ``c`` and ``d`` at step 4; ``b``
    arrived before step 1, ``c`` before step 3, and ``d`` and ``e`` before step 4, ``d`` while edge 1 drained."""
    d, kv = tmp_path / "restore", tmp_path / "kv"
    (d / "server" / "edges").mkdir(parents=True)
    (d / "medium" / "rep-1").mkdir(parents=True)
    kv.mkdir()
    plan = tmp_path / "plan.jsonl"
    plan.write_text("".join(json.dumps({"i": i, "prompt_ids": [i, i + 1]}) + "\n" for i in range(5)))
    (d / "meta.json").write_text(json.dumps({"plan": str(plan)}))
    (d / "medium" / "rep-1" / "client.jsonl").write_text("".join(
        json.dumps({"id": f"medium1-{i:05d}", "i": i}) + "\n" for i in range(5)))
    a, b, c, e4, e5 = (rid("medium1", i) for i in range(5))
    s = {"max_tokens": 4, "seed": 7}
    recs = [{"kind": "header", "config": {"block_size": 16}},
            {"kind": "arrive", "request": a, "before_step": 0, "sampling": s},
            {"kind": "step", "step": 0, "rows": [[a, 0, 2]]},
            {"kind": "arrive", "request": b, "before_step": 1, "sampling": s},
            {"kind": "step", "step": 1, "rows": [[a, 2, 1], [b, 0, 2]]},
            {"kind": "step", "step": 2, "rows": [[a, 3, 1], [b, 2, 1]]},
            {"kind": "arrive", "request": c, "before_step": 3, "sampling": s},
            {"kind": "step", "step": 3, "rows": [[c, 0, 2]]},
            {"kind": "arrive", "request": e4, "before_step": 4, "sampling": s},
            {"kind": "arrive", "request": e5, "before_step": 4, "sampling": s},
            {"kind": "step", "step": 4, "rows": [[c, 2, 1]]}]
    trace = d / "server" / "step_trace.9.jsonl"
    trace.write_text("".join(json.dumps(r) + "\n" for r in recs))
    edges = [{"edge": 0, "next_step": 2}, {"edge": 0, "state_write_s": 0.1}, {"edge": 1, "next_step": 4}]
    (d / "server" / "edges.9.jsonl").write_text("".join(json.dumps(r) + "\n" for r in edges))
    for k, held in ((0, [a, b]), (1, [c, e4])):
        st = {"requests": B.pickle_requests([Request(x) for x in held], {"block_hasher": None}), "next_step": 2 + 2 * k}
        (kv / f"edge-{k}.state.pkl").write_bytes(pickle.dumps(st))
        (d / "server" / "edges" / f"edge-{k}.dump.json").write_text(json.dumps({"sched.x": k}))
    outs = d / "server" / "edges" / "outputs.9.jsonl"
    outs.write_text("".join(json.dumps({"step": t, "out": {a: [[t], None]}}) + "\n" for t in range(1, 5)))
    if gz:
        for f in (trace, outs):
            with gzip.open(f.with_suffix(".jsonl.gz"), "wt") as z:
                z.write(f.read_text())
            f.unlink()
    return d, kv


@pytest.mark.parametrize("gz", [False, True])
def test_served_cuts_the_replay_block_and_its_arrivals(tmp_path: Path, gz: bool):
    d, kv = arm(tmp_path, gz)
    w = R.served(d, kv, 0, 1)
    assert (w["lo"], w["hi"], sorted(w["steps"]), sorted(w["outputs"])) == (2, 4, [2, 3], [2, 3])
    assert [(r["engine_id"], r["before_step"], r["prompt_ids"], r["max_tokens"]) for r in w["arrive"]] == [
        (rid("medium1", 2), 3, [2, 3], 4), (rid("medium1", 3), 4, [3, 4], 4)]
    assert w["dumps"][1] == {"sched.x": 1} and w["header"]["kind"] == "header"


def test_served_refuses_an_edge_not_taken_and_a_missing_plan(tmp_path: Path):
    d, kv = arm(tmp_path)
    with pytest.raises(SystemExit, match="not 0 and 2"):
        R.served(d, kv, 0, 2)
    (d / "meta.json").write_text(json.dumps({"plan": str(tmp_path / "elsewhere.jsonl")}))
    with pytest.raises(SystemExit, match="no plan at"):
        R.served(d, kv, 0, 1)


def test_step_diff_names_the_first_differing_step_and_fields():
    want = {2: {"rows": [["a", 3, 1]], "new": [], "t": 1.0}, 3: {"rows": [["c", 0, 2]], "new": ["c"], "t": 2.0}}
    got = {2: {"rows": [["a", 3, 1]], "new": [], "t": 9.0}, 3: {"rows": [["c", 0, 1]], "new": ["c"], "t": 9.5}}
    assert R.step_diff(want, {k: dict(v) for k, v in want.items()})["steps_differing"] == 0
    d = R.step_diff(want, got)
    assert (d["steps_differing"], d["rows_served"], d["first"]["step"], d["first"]["fields"]) == (1, 2, 3, ["rows"])
    assert R.step_diff(want, {2: got[2]})["first"]["step"] == 3


def test_fingerprint_diff_counts_blocks_and_layers():
    np = pytest.importorskip("numpy")
    a = np.arange(12, dtype=np.int64).reshape(4, 3)
    b = a.copy()
    assert R.fingerprint_diff(a, b)["blocks_differing"] == 0
    b[2, 1] += 1
    d = R.fingerprint_diff(a, b)
    assert (d["blocks_differing"], d["first"], d["layers_differing_first_block"]) == (1, [2], [1])
    assert R.fingerprint_diff(a, b[:3]) == {"compared": False} and R.fingerprint_diff(None, b) == {"compared": False}


def test_dump_diff_sets_the_differences_by_construction_apart():
    served = {"core.engine_index": 0, "core.step": 3, "sched.requests['a'].sampling_params.detokenize": True,
              "sched.requests['a'].num_computed_tokens": 5, "sched.requests['a'].arrival_time": "1.0"}
    replayed = {"core.step": 3, "sched.requests['a'].sampling_params.detokenize": False,
                "sched.requests['a'].num_computed_tokens": 5, "sched.requests['a'].arrival_time": "2.0"}
    d = R.dump_diff(served, replayed)
    assert d["clean"] and d["clock_differing"] == 1
    assert d["by_construction"] == {"core.engine_index": 1, "sched.requests[*].sampling_params.detokenize": 1}
    assert not R.dump_diff(served, {**replayed, "core.step": 4})["clean"]
    assert not R.dump_diff({**served, "sched.extra": 1}, replayed)["clean"]
    assert not R.dump_diff(served, {**replayed, "core.extra": 1})["clean"]


def test_warm_serves_its_requests_to_the_end_and_empties_the_batch_queue(monkeypatch: pytest.MonkeyPatch):
    class Eng:
        """Each request takes two steps; under async scheduling (a core with a batch queue) one batch is left in the
        queue past the last finish."""

        def __init__(self, core: NS) -> None:
            self.added: list[str] = []
            self.left, self.core = 0, core

        def add_request(self, rid: str, prompt: dict, params: object) -> None:
            self.added.append(rid)
            self.left += 2

        def has_unfinished_requests(self) -> bool:
            return self.left > 0

        def step(self) -> None:
            if self.left:
                self.left -= 1
                if self.left == 0 and hasattr(self.core, "batch_queue"):
                    self.core.batch_queue = [1]
            else:
                self.core.batch_queue = []

    monkeypatch.setattr(R, "params", lambda r: r["sampling"])
    core = NS(batch_queue=[])
    eng = Eng(core)
    assert R.warm(eng, core, [{"prompt_ids": [1], "sampling": {}}] * 2) == 5
    assert eng.added == ["restore-warm-0", "restore-warm-1"] and core.batch_queue == []
    sync = NS()
    assert R.warm(Eng(sync), sync, [{"prompt_ids": [1], "sampling": {}}]) == 2


def test_the_restores_engine_takes_the_headers_cache_format_and_draft(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A serve with an FP8 cache or speculative decoding is restored into an engine with them, as it is replayed
    (`replay.HEADER_ARGS`); a header without them builds the engine it did."""
    vllm, llm_engine = types.ModuleType("vllm"), types.ModuleType("vllm.v1.engine.llm_engine")
    vllm.EngineArgs = lambda **kw: kw
    llm_engine.LLMEngine = NS(from_engine_args=lambda args: args)
    for name, mod in (("vllm", vllm), ("vllm.v1", types.ModuleType("vllm.v1")),
                      ("vllm.v1.engine", types.ModuleType("vllm.v1.engine")), ("vllm.v1.engine.llm_engine", llm_engine)):
        monkeypatch.setitem(sys.modules, name, mod)
    cfg = {"max_model_len": 4096, "max_num_seqs": 8, "max_num_batched_tokens": 512, "enable_chunked_prefill": True,
           "enable_prefix_caching": True, "async_scheduling": True, "block_size": 16, "num_gpu_blocks": 100}
    plain = R.engine(tmp_path, {"config": cfg}, eager=True, log_stats=False)
    assert plain["disable_log_stats"] and not {"long_prefill_token_threshold", "kv_cache_dtype", "speculative_config"} & set(plain)
    draft = {"method": "mtp", "num_speculative_tokens": 5}
    got = R.engine(tmp_path, {"config": {**cfg, "long_prefill_token_threshold": 256, "kv_cache_dtype": "fp8_ds_mla",
                                         "speculative": draft}}, eager=True, log_stats=False)
    assert {k: v for k, v in got.items() if k not in plain} == {
        "long_prefill_token_threshold": 256, "kv_cache_dtype": "fp8_ds_mla", "speculative_config": draft}
    assert {k: got[k] for k in plain} == plain
    replayed = R.replay.engine(tmp_path, {"config": {**cfg, "kv_cache_dtype": "fp8_ds_mla"}}, eager=True)
    assert replayed["kv_cache_dtype"] == R.engine(tmp_path, {"config": {**cfg, "kv_cache_dtype": "fp8_ds_mla"}}, True, True)[
        "kv_cache_dtype"]
