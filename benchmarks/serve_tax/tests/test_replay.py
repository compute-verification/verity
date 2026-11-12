"""The replay's served window, cut from a synthetic arm's step trace, and its comparison with a replay."""
from __future__ import annotations

import gzip
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import replay as R
import servetax_plugin as P


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


def bench_arm(tmp_path: Path, finished: tuple[int, ...] = (0, 1)) -> Path:
    """A pass sent by vLLM's client, two users a request each, after a warm-up: no plan and no client.jsonl; the ids
    are the ones the arm's server kept (`servetax_plugin.ServedIds`), a line a request of ``finished``."""
    d = tmp_path / "prod-record"
    (d / "agentic@2" / "rep-1").mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"plan": None, "checkpoint": "ck"}))
    (d / "agentic@2" / "rep-1" / "meta.json").write_text(json.dumps({"tag": "agentic@2-r1"}))
    w, a, b = "cmpl-warm-u0-0-0-0123abcd", "cmpl-agentic@2-r1-u0-0-0-0123abcd", "cmpl-agentic@2-r1-u1-0-0-0123abcd"
    ids = P.ServedIds(P._lines(d / "server" / "ids.9.jsonl"))
    for k, rid in enumerate((w, a, b)):
        ids.prompt(NS(request_id=rid, prompt_token_ids=[k, k + 1]))
    ids.sampled({0: NS(outputs=[NS(request_id=w, new_token_ids=[5, 5], finish_reason=1)])})
    for u, rid in enumerate((a, b)):
        ids.sampled({0: NS(outputs=[NS(request_id=rid, new_token_ids=[7], finish_reason=None)])})
        if u in finished:
            ids.sampled({0: NS(outputs=[NS(request_id=rid, new_token_ids=[u], finish_reason=1)])})
    recs = [{"kind": "header", "config": {}},
            {"kind": "arrive", "request": w, "before_step": 0, "sampling": {"max_tokens": 2}, "t": 0.0},
            {"kind": "step", "step": 0, "rows": [[w, 0, 2]], "t": 0.1},
            {"kind": "arrive", "request": a, "before_step": 1, "sampling": {"max_tokens": 2}, "t": 1.0},
            {"kind": "step", "step": 1, "rows": [[a, 0, 2]], "t": 1.1},
            {"kind": "arrive", "request": b, "before_step": 2, "after_empty": 1, "sampling": {"max_tokens": 2}, "t": 1.2},
            {"kind": "step", "step": 2, "rows": [[a, 2, 1], [b, 0, 2]], "t": 1.3}]
    (d / "server" / "step_trace.9.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    return d


def test_window_of_a_pass_sent_by_vllms_client_reads_the_ids_the_arms_server_kept(tmp_path: Path):
    d = bench_arm(tmp_path)
    w = R.window(d, "agentic@2", 1)
    a, b = "cmpl-agentic@2-r1-u0-0-0-0123abcd", "cmpl-agentic@2-r1-u1-0-0-0123abcd"
    assert [(r["engine_id"], r["before_step"], r["after_empty"], r["prompt_ids"], r["served"]) for r in w["requests"]] == [
        (a, 0, 0, [1, 2], [7, 0]), (b, 1, 1, [2, 3], [7, 1])]
    assert [s["step"] for s in w["steps"]] == [0, 1]
    # the lines are read gzipped too, as run.sh leaves them, and only the pass's own are parsed
    kept = d / "server" / "ids.9.jsonl"
    with gzip.open(kept.with_name("ids.9.jsonl.gz"), "wt") as f:
        f.write(kept.read_text() + "not a line of this pass\n")
    kept.unlink()
    assert R.window(d, "agentic@2", 1)["requests"] == w["requests"]
    assert sorted(R.served_ids(d, "warm")) == ["cmpl-warm-u0-0-0-0123abcd"]


def test_window_refuses_a_pass_whose_ids_nothing_kept(tmp_path: Path):
    d = bench_arm(tmp_path / "unfinished", finished=(0,))
    with pytest.raises(SystemExit, match="cmpl-agentic@2-r1-u1-0-0-0123abcd arrived in the window and the server kept no ids"):
        R.window(d, "agentic@2", 1)
    d = bench_arm(tmp_path / "stock")
    (d / "server" / "ids.9.jsonl").unlink()
    with pytest.raises(SystemExit, match=r"no client.jsonl and .*server kept no ids of load agentic@2 rep 1 \(agentic@2-r1\)"):
        R.window(d, "agentic@2", 1)


class Core:
    """A stand-in for an engine core's scheduler under a draft: a request runs a row a step until it has run its
    ``max_tokens``, and a request of ``held`` is held back from one schedule after its first row, as a request at its
    token limit is until its step's output is back: that schedule runs the others, or nothing."""

    def __init__(self, held: tuple[str, ...] = ()):
        self.requests: dict[str, int] = {}
        self.held, self.waiting = set(held), set()

    def add_request(self, request):
        self.requests[request.request_id] = request.sampling_params.max_tokens

    def schedule(self):
        rows = [r for r in self.requests if r not in self.waiting]
        self.waiting.clear()
        return NS(total_num_scheduled_tokens=len(rows), num_scheduled_tokens={r: 1 for r in rows})

    def update_from_output(self, so, mro):
        outs = []
        for r in so.num_scheduled_tokens:
            self.requests[r] -= 1
            if r in self.held:
                self.held.discard(r)
                self.waiting.add(r)
            outs.append(NS(request_id=r, new_token_ids=[len(r)], finish_reason=None if self.requests[r] else 1, stop_reason=None))
            if not self.requests[r]:
                del self.requests[r]
        return {0: NS(outputs=outs)}

    def finish_requests(self, request_ids, finished_status):
        return []


class Engine:
    """The engine `replay.replay` drives, over `Core`: its requests added and stepped as vLLM's ``LLMEngine`` names them."""

    def __init__(self, held: tuple[str, ...] = ()):
        self.engine_core = NS(scheduler=Core(held))
        self.tokens: dict[str, list[int]] = {}

    def add_request(self, rid, prompt, params):
        self.engine_core.scheduler.add_request(NS(request_id=rid, prompt_token_ids=prompt["prompt_token_ids"], sampling_params=params))

    def has_unfinished_requests(self):
        return bool(self.engine_core.scheduler.requests)

    def step(self):
        sched = self.engine_core.scheduler
        so = sched.schedule()
        if not so.total_num_scheduled_tokens:
            return []
        done = []
        for o in sched.update_from_output(so, NS(req_ids=list(so.num_scheduled_tokens)))[0].outputs:
            self.tokens.setdefault(o.request_id, []).extend(o.new_token_ids)
            if o.finish_reason is not None:
                done.append(NS(finished=True, request_id=o.request_id, outputs=[NS(token_ids=self.tokens[o.request_id])]))
        return done


def test_a_replay_runs_under_the_served_arms_other_plugins_and_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The canonical top-k decides a long row's keys, so a replay without it is not a replay of an arm that served on it."""
    loaded = []
    monkeypatch.setattr(R.S, "check_plugins", lambda env, arm: loaded.append((env, arm)))
    monkeypatch.setenv("VLLM_PLUGINS", "none")
    site, private = tmp_path / "site", tmp_path / "private"
    for arm in ("prod-record", "two-stage", "canonical-topk"):
        under = R.served_under(arm, site, None)
        assert under["VERITY_DSA_TOPK"] == "canonical" and "VERITY_DSA_TOPK_LOG" not in under        # nothing kept without a place
        assert under["PYTHONPATH"].split(R.os.pathsep)[:2] == [str(site), str(R.ROOT / "benchmarks/dsa_topk")]
        # the top-k alone: none of the arm's recording (the replay keeps its own trace), and nothing of the job's that would drop it
        assert R.S.PLUGIN_ENV not in under and "VLLM_PLUGINS" not in under
        assert ((site / "verity_serve_tax-0.0.dist-info" / "entry_points.txt").read_text()
                == "[vllm.general_plugins]\nverity_dsa_topk = dsa_topk:register\n")
        env, ours = loaded[-1]
        assert env is under and R.S.entry_points(ours) == ["verity_dsa_topk = dsa_topk:register"]   # loaded once before the engine
    assert R.served_under("prod-record", site, private)["VERITY_DSA_TOPK_LOG"] == f"{private}/topk-log"
    # a job that already runs the replay under the arm's environment is left as it is, its plugins seen to load there
    monkeypatch.setenv("VERITY_DSA_TOPK", "canonical")
    monkeypatch.setenv("VERITY_DSA_TOPK_LOG", "the job's own")
    n = len(loaded)
    assert R.served_under("prod-record", site, private) == {} and len(loaded) == n + 1
    assert loaded[-1][0]["VERITY_DSA_TOPK_LOG"] == "the job's own" and R.S.entry_points(loaded[-1][1]) == ["verity_dsa_topk = dsa_topk:register"]
    monkeypatch.delenv("VERITY_DSA_TOPK")
    n = len(loaded)
    for arm in ("prod", "prod-again", "record", "compiled", "no such arm", None):                   # stock, or the record alone
        assert R.served_under(arm, site, None) == {}
    assert len(loaded) == n
    # the replay's own files leave the node, so what the plugin keeps is not put among them
    with pytest.raises(SystemExit, match="is inside .*which leaves the node"):
        R.main(["--arm-dir", str(tmp_path / "prod-record"), "--load", "agentic@2", "--out", str(tmp_path / "out"),
                "--private", str(tmp_path / "out" / "private")])


def test_replay_json_holds_the_value_settings_in_force_in_the_environment_the_replays_engine_is_given(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The served arm's meta.json has its server's (`serve_tax.value_settings`); with the replay's, a reader sees in which
    variables the two engines' environments differ. The replay's is taken once its own two variables are set."""
    d = arm(tmp_path)
    (d / "meta.json").write_text(json.dumps({"plan": str(tmp_path / "plan.jsonl"), "checkpoint": "ck", "server_argv": []}))
    given: dict[str, str] = {}

    def replayed(w: dict, *args: object) -> dict:
        given.update(R.os.environ)                                 # what the engine and its processes inherit
        return {"wall_s": 1.0, "ready_s": 0.5, "steps": [dict(s) for s in w["steps"]],
                "tokens": {r["engine_id"]: list(r["served"]) for r in w["requests"]}}

    monkeypatch.setattr(R, "replay", replayed)
    # `main` refuses a process with hash randomization on: its own view of `sys`, since the import system reads the real flags
    monkeypatch.setattr(R, "sys", NS(flags=NS(hash_randomization=0), path=sys.path))
    monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
    monkeypatch.setenv("VLLM_API_KEY", "a credential's value")
    for name in ("VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"):
        monkeypatch.setenv(name, "the job's")                      # `main` sets both in this process: put back after the test
    assert R.main(["--arm-dir", str(d), "--load", "sat", "--out", str(tmp_path / "out")]) == 0
    out = json.loads((tmp_path / "out" / "replay.json").read_text())
    assert out["bit_exact"] and out["settings"] == R.S.value_settings(given)
    assert {k: out["settings"]["values"][k] for k in ("VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION",
                                                      "NCCL_P2P_DISABLE", "VLLM_API_KEY")} == {
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1", "NCCL_P2P_DISABLE": "1",
        "VLLM_API_KEY": "<redacted>"}
    assert "a credential's value" not in json.dumps(out)


def test_the_replay_puts_a_request_back_after_the_empty_schedules_its_arrival_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Serving held ``a`` back from one schedule, which ran nothing, and ``b`` arrived after it: the next step ran both.
    The replay adds ``b`` after that empty schedule; added before it, as an arrival that names none is, ``b`` runs alone
    a schedule early and every later step differs."""
    from verity_vllm.engine.step_trace import StepTrace

    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = lambda **kw: NS(**kw)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    # the replay's capture of logits rows is not asked for here, so its module (and the records it reads) is not loaded
    monkeypatch.setitem(sys.modules, "verity_vllm.engine.sampler_capture", types.ModuleType("sampler_capture"))
    serving = Engine(held=("a",))
    trace = StepTrace().install(serving)
    serving.add_request("a", {"prompt_token_ids": [1]}, NS(max_tokens=2))
    serving.step()
    assert serving.step() == [] and trace.empty == 1
    serving.add_request("b", {"prompt_token_ids": [2]}, NS(max_tokens=1))
    assert sorted(o.request_id for o in serving.step()) == ["a", "b"]
    trace.uninstall()
    arrive = [r for r in trace.records if r["kind"] == "arrive"]
    assert [(r["request"], r["before_step"], r.get("after_empty")) for r in arrive] == [("a", 0, None), ("b", 1, 1)]
    served = [[row[0] for row in r["rows"]] for r in trace.records if r["kind"] == "step"]
    assert served == [["a"], ["a", "b"]]

    def replayed(after_empty: bool) -> list[list[str]]:
        w = {"header": trace.records[0],
             "requests": [{"engine_id": r["request"], "before_step": r["before_step"],
                           "after_empty": r.get("after_empty", 0) if after_empty else 0, "prompt_ids": [1],
                           "max_tokens": {"a": 2, "b": 1}[r["request"]], "sampling": {}} for r in arrive]}
        monkeypatch.setattr(R, "engine", lambda *a, **kw: Engine(held=("a",)))
        got = R.replay(w, tmp_path, eager=True)
        assert got["tokens"] == serving.tokens
        return [[row[0] for row in s["rows"]] for s in got["steps"]]

    assert replayed(True) == served
    assert replayed(False) == [["a"], ["b"], ["a"]]


class AheadCore(Core):
    """`Core` whose update drops a row of a request that finished while the row's step was in flight, as vLLM's does."""

    def update_from_output(self, so, mro):
        live = {r: n for r, n in so.num_scheduled_tokens.items() if r in self.requests}
        return super().update_from_output(NS(num_scheduled_tokens=live), mro)


class Ahead(Engine):
    """`Engine` scheduling a step while the one before it runs, as vLLM's engine core does under async scheduling
    (``step_with_batch_queue``, two steps in flight at most): a step is scheduled while a request is unfinished and
    its output taken a call later, so a request's last output comes back with one more row of it in flight."""

    def __init__(self):
        self.engine_core = NS(scheduler=AheadCore())
        self.tokens: dict[str, list[int]] = {}
        self.flying: list = []

    def step(self):
        sched = self.engine_core.scheduler
        if sched.requests:
            self.flying.insert(0, sched.schedule())
            if len(self.flying) < 2:
                return []
        elif not self.flying:
            return []
        so = self.flying.pop()
        done = []
        for o in sched.update_from_output(so, NS(req_ids=list(so.num_scheduled_tokens)))[0].outputs:
            self.tokens.setdefault(o.request_id, []).extend(o.new_token_ids)
            if o.finish_reason is not None:
                done.append(NS(finished=True, request_id=o.request_id, outputs=[NS(token_ids=self.tokens[o.request_id])]))
        return done


def test_the_replay_takes_the_step_in_flight_when_the_last_request_finishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An engine that schedules ahead has a step in flight when the last request's finish comes back. Serving took its
    output, so the served trace ends on a step that emitted nothing; a replay that stopped at the finish would have one
    step fewer and differ from the served schedule in its last step, with every token equal."""
    from verity_vllm.engine.step_trace import StepTrace

    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = lambda **kw: NS(**kw)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "verity_vllm.engine.sampler_capture", types.ModuleType("sampler_capture"))
    serving = Ahead()
    trace = StepTrace().install(serving)
    serving.add_request("a", {"prompt_token_ids": [1]}, NS(max_tokens=2))
    while serving.has_unfinished_requests() or serving.flying:
        serving.step()
    trace.uninstall()
    steps = [r for r in trace.records if r["kind"] == "step"]
    # two rows ran the request's two tokens; the third was scheduled before the second's finish came back
    assert [(s["step"], [row[0] for row in s["rows"]], s["emitted"], sorted(s["finished"])) for s in steps] == [
        (0, ["a"], {"a": 1}, []), (1, ["a"], {"a": 1}, ["a"]), (2, ["a"], {}, [])]
    w = {"header": trace.records[0], "steps": steps,
         "requests": [{"engine_id": "a", "before_step": 0, "after_empty": 0, "prompt_ids": [1], "max_tokens": 2,
                       "sampling": {}, "served": serving.tokens["a"]}]}
    monkeypatch.setattr(R, "engine", lambda *a, **kw: Ahead())
    got = R.replay(w, tmp_path, eager=True)
    assert [s["rows"] for s in got["steps"]] == [s["rows"] for s in steps] and got["tokens"] == serving.tokens
    out = R.compare(w, got)
    assert (out["steps_served"], out["steps_replayed"], out["steps_differing"], out["bit_exact"]) == (3, 3, 0, True)
    # an engine with nothing in flight at the end is stepped no further
    monkeypatch.setattr(R, "engine", lambda *a, **kw: Engine())
    assert len(R.replay(w, tmp_path, eager=True)["steps"]) == 2


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


def test_glm53s_two_stage_launch_settings_say_what_the_arm_and_the_model_say(tmp_path):
    """The two-stage serve's launch settings for GLM-5.3, the boundary object as ``BOUNDARY_ROWS`` (the greedy select in
    units as its check, ``ids``, the arm's tables) and the capture's template as ``--capture-template``, say what the
    ``two-stage`` arm and the model's check say by default: the plugin is handed the same boundary object either way,
    and a capture is labelled with the same check whether given it, reading the serve's meta, or the model's."""
    S, glm, two = R.S, R.S.MODELS["glm53"], R.S.ARMS["two-stage"]
    check = "GreedySelect_v1{V=154880,N=7744}"
    setting = {"check": check, "inline": True, "per_prompt": 0, "per_output": 1, "drained": True, "ids": True,
               "tables": {"layers": 1, "private": S.PRIVATE}}

    def handed(rows: dict | None) -> dict:
        env = S.server_env(two, tmp_path, tmp_path / "site", None, 3600.0, 600.0, boundary_rows=rows, check=glm.check,
                           private=tmp_path / "private")
        return json.loads(env[S.PLUGIN_ENV])["boundary_rows"]

    assert handed(json.loads(json.dumps(setting))) == handed(None)
    assert {R.capture_template({"model_id": "glm53", "boundary_rows": handed(None)}),
            R.capture_template({"model_id": "glm53"}), R.capture_template({}, check)} == {check}
