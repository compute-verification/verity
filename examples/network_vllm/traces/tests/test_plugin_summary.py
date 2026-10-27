"""The scheduler hook on a stand-in for vLLM's Scheduler (the attributes the hook reads, as vLLM d9105ea80 has them), and the
summary over what it wrote."""
from __future__ import annotations

import enum
import hashlib
import json
from types import SimpleNamespace as NS

import pytest


class Finish(enum.IntEnum):
    STOP = 0
    LENGTH = 1


class Request:
    def __init__(self, rid: str, prompt: list[int], max_tokens: int):
        self.request_id, self.prompt_token_ids, self.max_tokens = rid, prompt, max_tokens
        self.num_prompt_tokens, self.output_token_ids, self.num_preemptions = len(prompt), [], 0
        self.arrival_time, self.stop_reason, self.status, self.finish = 1000.0, None, NS(name="RUNNING"), None
        self.prefill_stats = None

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    def get_finished_reason(self):
        return self.finish


class Scheduler:
    """Hands out the scripted SchedulerOutputs and frees requests as told."""

    def __init__(self, script):
        self.script, self.requests, self.waiting, self.running = list(script), {}, [], []
        self.current_step = 0

    def add_request(self, request):
        self.requests[request.request_id] = request
        self.waiting.append(request)

    def schedule(self, throttle_prefills: bool = False):
        return self.script.pop(0)

    def update_from_output(self, scheduler_output, model_runner_output):
        return {}

    def _free_request(self, request, delay_free_blocks: bool = False):
        return None, None


def so(new=(), cached=(), preempted=(), finished=()):
    """new: (rid, computed, n, prompt_len); cached: (rid, computed, n, outputs, resumed)."""
    nst = {r[0]: r[2] for r in new} | {r[0]: r[2] for r in cached}
    return NS(scheduled_new_reqs=[NS(req_id=r, num_computed_tokens=c, prompt_len=L) for r, c, _, L in new],
              scheduled_cached_reqs=NS(req_ids=[r[0] for r in cached], num_computed_tokens=[r[1] for r in cached],
                                       num_output_tokens=[r[3] for r in cached],
                                       resumed_req_ids={r[0] for r in cached if r[4]}),
              num_scheduled_tokens=nst, total_num_scheduled_tokens=sum(nst.values()), preempted_req_ids=set(preempted),
              finished_req_ids=set(finished))


A, B = "cmpl-t-00000-0-0123abcd", "cmpl-t-00001-0-89abcdef"


def serve(tmp_path, monkeypatch, plugin) -> Scheduler:
    """A, 40 prompt tokens with a 16-token prefix-cache hit, prefilled in two chunks then decoded to EOS; B, 20 tokens,
    prefilled whole, preempted after two outputs, recomputed (22 tokens) and cut at its cap."""
    monkeypatch.setenv(plugin.TRACE_DIR, str(tmp_path / "t" / "raw"))
    a, b = Request(A, [1] * 40, 256), Request(B, [2] * 20, 4)
    # (scheduler output, model runner output, tokens appended after it); so() schedules nothing, and the malformed one
    # makes the hook raise
    script = [(so(new=[(A, 16, 16, 40), (B, 0, 20, 20)]), NS(req_ids=[B, A], sampled_token_ids=[[5], []]), {B: 5}),
              (so(cached=[(A, 32, 8, 0, False), (B, 20, 1, 1, False)]), NS(req_ids=[B, A], sampled_token_ids=[[6], [7]]),
               {B: 6, A: 7}),
              (so(cached=[(A, 40, 1, 1, False)], preempted=[B]), NS(req_ids=[A], sampled_token_ids=[[8]]), {A: 8}),
              (so(), None, {}),
              (so(new=[(B, 0, 22, 20)], cached=[(A, 41, 1, 2, False)]),
               NS(req_ids=[A, B], sampled_token_ids=[[128001], [6]]), {A: 128001, B: 6}),
              (NS(scheduled_new_reqs=None), None, {}),
              (so(cached=[(B, 22, 1, 3, False)], finished=[A]), NS(req_ids=[B], sampled_token_ids=[[3]]), {B: 3})]

    class S(Scheduler):
        pass

    plugin.install(S)
    s = S([out for out, _, _ in script])
    s.add_request(a)
    s.add_request(b)
    for k, (_, mro, toks) in enumerate(script):
        if k == 4:
            b.num_preemptions = 1
        out = s.schedule()
        if mro is not None:
            s.update_from_output(out, mro)
        for rid, t in toks.items():
            s.requests[rid].output_token_ids.append(t)
        if k == 4:
            a.finish, a.status = Finish.STOP, NS(name="FINISHED_STOPPED")
            s._free_request(a)
    b.finish, b.status = Finish.LENGTH, NS(name="FINISHED_LENGTH_CAPPED")
    s._free_request(b)
    return s


def records(tmp_path, kind):
    (p,) = (tmp_path / "t" / "raw").glob(f"{kind}.*.jsonl")
    return [json.loads(x) for x in p.read_text().splitlines()]


def test_hook_records_steps_outputs_and_requests(tmp_path, monkeypatch, mod):
    plugin = mod("vllm_trace_plugin")
    serve(tmp_path, monkeypatch, plugin)
    arrivals = records(tmp_path, "arrivals")
    assert [(r["id"], r["arrival"], r["sched_step"], r["waiting"]) for r in arrivals] == [("t-00000", 1000.0, 0, 0),
                                                                                         ("t-00001", 1000.0, 0, 1)]
    steps = records(tmp_path, "steps")
    assert [s["step"] for s in steps] == [0, 1, 2, 3, 4]
    s0 = steps[0]
    assert [r["id"] for r in s0["new"]] == ["t-00000", "t-00001"]
    assert s0["new"][0] == {"id": "t-00000", "num_computed_tokens": 16, "num_scheduled_tokens": 16, "prompt_len": 40,
                                 "resumed": False, "num_output_tokens": 0, "cache_hit": 16}
    assert steps[2]["preempted"] == ["t-00001"] and steps[3]["empty_before"] == 1
    assert steps[3]["new"][0]["resumed"] is True and "cache_hit" not in steps[3]["new"][0]
    assert steps[4]["finished"] == ["t-00000"]
    outputs = records(tmp_path, "outputs")
    assert [o["step"] for o in outputs] == [0, 1, 2, 3, 4]
    assert outputs[0] == {"step": 0, "t": outputs[0]["t"], "batch": ["t-00001", "t-00000"], "sampled": [[5], []]}
    reqs = {r["id"]: r for r in records(tmp_path, "requests")}
    assert reqs["t-00000"]["output_ids"] == [7, 8, 128001] and reqs["t-00000"]["finish_reason"] == "stop"
    assert reqs["t-00001"]["finish_reason"] == "length" and reqs["t-00001"]["num_preemptions"] == 1
    assert reqs["t-00000"]["cache_hit"] == 16 and reqs["t-00000"]["engine_id"] == A and reqs["t-00001"]["cache_hit"] == 0
    (err,) = (tmp_path / "t" / "raw").glob("hook_errors.*.txt")
    assert err.read_text().count("--- schedule") == 1


def test_client_id(mod):
    plugin = mod("vllm_trace_plugin")
    assert plugin.client_id("cmpl-heavy-00012-0-0a1b2c3d") == "heavy-00012"
    assert plugin.client_id("chatcmpl-x-1-ffffffff") == "x"
    assert plugin.client_id("something-else") == "something-else"


def test_summary_joins_and_counts(tmp_path, monkeypatch, mod):
    plugin, summary = mod("vllm_trace_plugin"), mod("summary")
    serve(tmp_path, monkeypatch, plugin)
    (tmp_path / "plan.jsonl").write_text("".join(json.dumps({"i": i, "prefix": p, "prefix_len": q}) + "\n"
                                                 for i, p, q in ((0, 3, 16), (1, None, 0))))
    (tmp_path / "meta.json").write_text(json.dumps({"model": "m", "requests": 2, "seed": 0, "traces": {"t": {}}}))
    (tmp_path / "t" / "meta.json").write_text(json.dumps({"rate": 1.0, "why": "test", "unfinished": 0}))
    client = [{"id": "t-00000", "i": 0, "status": 200, "output_ids": [7, 8, 128001], "send_unix": 999.5},
              {"id": "t-00001", "i": 1, "status": 200, "output_ids": [5, 6, 6, 3]}]
    (tmp_path / "t" / "client.jsonl").write_text("".join(json.dumps(c) + "\n" for c in client))
    (tmp_path / "t" / "raw" / "accept.2.jsonl").write_text(json.dumps(
        {"id": "t-00000", "path": "/v1/completions", "accept": 999.6, "conn_request": 1, "headers": 999.7, "body": 999.8}) + "\n")
    out = summary.summarize(tmp_path)
    t = out["traces"]["t"]
    assert t["warden"] is None and "views" not in t
    assert t["arrival_order"]["engine add"] == {"requests": 2, "inversions_vs_engine": 0, "inversions_vs_previous": 0,
                                                "lag_s": t["arrival_order"]["engine add"]["lag_s"]}
    assert t["requests"]["finish"] == {"eos": 1, "length": 1} and t["requests"]["client_ids_match"] == 2
    assert t["rows"] == {"total": 8, "prefill": 4, "decode": 4, "partial_prefill_chunks": 1,
                         "requests_prefilled_in_chunks": 1, "prefill_tokens_per_row": t["rows"]["prefill_tokens_per_row"]}
    assert t["preemption"] == {"events": 1, "requests_preempted": 1, "max_per_request": 1, "resumed_rows": 1}
    assert t["prefix_cache"]["requests_with_hit"] == 1 and t["prefix_cache"]["hit_tokens"] == 16
    assert t["steps"]["mixed_prefill_decode"] == 2 and t["hook_errors"] == 1
    assert t["overrun"] == {"rows": 0, "by_finish": {}, "requests": 0, "max_per_request": 0}
    assert out["outputs"]["verdict"] == "identical: all 2 requests have the same output ids in all 1 traces"
    steps = [json.loads(x) for x in (tmp_path / "t" / "steps.jsonl").read_text().splitlines()]
    assert steps[0]["batch"] == ["t-00001", "t-00000"] and steps[0]["new"][0]["kind"] == "prefill"
    assert steps[0]["new"][0]["final"] is False and steps[1]["cached"][1]["kind"] == "decode"
    reqs = [json.loads(x) for x in (tmp_path / "t" / "requests.jsonl").read_text().splitlines()]
    assert [r["first_step"] for r in reqs] == [0, 0] and reqs[0]["prefix"] == 3 and reqs[0]["eos"] is True
    assert (reqs[0]["accept_unix"], reqs[0]["body_unix"], reqs[1]["accept_unix"]) == (999.6, 999.8, None)
    assert [r["engine_add_index"] for r in reqs] == [0, 1] and "warden" not in reqs[0]
    assert "| t | 1.0 | test | 2 of 2 |" in (tmp_path / "README.md").read_text()


def test_summary_keeps_a_failed_trace_and_drops_a_cut_line(tmp_path, mod):
    summary = mod("summary")
    (tmp_path / "plan.jsonl").write_text('{"i": 0}\n{"i": 1')
    assert summary.read_jsonl(tmp_path / "plan.jsonl") == [{"i": 0}]
    (tmp_path / "meta.json").write_text(json.dumps({"traces": {"u": {"error": "RuntimeError: not ready"}}}))
    out = summary.summarize(tmp_path)
    assert out["traces"]["u"] == {"trace": "u", "error": "RuntimeError: not ready"}
    assert "| u | None | None | failed: RuntimeError: not ready |" in (tmp_path / "README.md").read_text()


def row(rid, s, n, L, o, resumed=False):
    return {"id": rid, "num_computed_tokens": s, "num_scheduled_tokens": n, "prompt_len": L, "num_output_tokens": o,
            "resumed": resumed}


def trace(d, steps, reqs):
    """A summarized trace: steps as (rows, batch, sampled, preempted), reqs as (i, id, output_ids, finish, preemptions)."""
    d.mkdir(parents=True)
    lines = []
    for k, (rows, batch, sampled, preempted) in enumerate(steps):
        lines.append(json.dumps({"step": k, "new": [], "cached": rows, "batch": batch, "sampled": sampled,
                                 "preempted": preempted, "total_num_scheduled_tokens": sum(r["num_scheduled_tokens"]
                                                                                            for r in rows)}))
    (d / "steps.jsonl").write_text("".join(x + "\n" for x in lines))
    (d / "requests.jsonl").write_text("".join(json.dumps(
        {"i": i, "id": rid, "output_ids": out, "output_len": len(out), "finish_reason": fin, "eos": fin == "stop",
         "num_preemptions": p, "cache_hit": 0}) + "\n" for i, rid, out, fin, p in reqs))


def test_overrun_counts_rows_sampling_past_the_last_output(mod):
    summary = mod("summary")
    # async: A (prompt 4) stops on EOS at output 1 while the step after is already scheduled; its cached rows count the
    # in-flight output as an output (num_output_tokens includes placeholders). B is capped at 2 outputs and is not
    # scheduled past them.
    steps = [{"new": [row("A", 0, 4, 4, 0), row("B", 0, 3, 3, 0)], "cached": []},
             {"new": [], "cached": [row("A", 4, 1, 4, 1), row("B", 3, 1, 3, 1)]},
             {"new": [], "cached": [row("A", 5, 1, 4, 2)]}]
    for s in steps:
        for r in s["new"] + s["cached"]:
            summary.classify(r)
    assert [r["kind"] for r in steps[1]["cached"]] == ["decode", "decode"]
    assert [summary.sampled_index(r) for s in steps for r in s["new"] + s["cached"]] == [0, 0, 1, 1, 2]
    reqs = [{"id": "A", "output_len": 2, "eos": True, "finish_reason": "stop"},
            {"id": "B", "output_len": 2, "eos": False, "finish_reason": "length"}]
    assert summary.overrun(steps, reqs) == {"rows": 1, "by_finish": {"eos": 1}, "requests": 1, "max_per_request": 1}


def test_compare_places_the_first_difference_in_every_trace(tmp_path, mod):
    summary = mod("summary")
    # request 0 is the same in both; request 1 (prompt 3) differs at output index 1, which x samples in a decode step of
    # two rows and y in its step 3, after a preemption and a resumed recompute of prompt and first output
    trace(tmp_path / "x", [([row("a", 0, 2, 2, 0), row("b", 0, 3, 3, 0)], ["a", "b"], [[7], [5]], []),
                           ([row("a", 2, 1, 2, 1), row("b", 3, 1, 3, 1)], ["a", "b"], [[8], [6]], [])],
          [(0, "a", [7, 8], "length", 0), (1, "b", [5, 6], "length", 0)])
    trace(tmp_path / "y", [([row("a", 0, 2, 2, 0), row("b", 0, 3, 3, 0)], ["a", "b"], [[7], [5]], []),
                           ([row("a", 2, 1, 2, 1)], ["a"], [[8]], ["b"]),
                           ([], [], [], []),
                           ([row("b", 0, 4, 3, 1, resumed=True)], ["b"], [[9]], [])],
          [(0, "a", [7, 8], "length", 0), (1, "b", [5, 9], "length", 1)])
    for d in ("x", "y"):
        steps = summary.read_jsonl(tmp_path / d / "steps.jsonl")
        for s in steps:
            for r in s["cached"]:
                summary.classify(r)
        (tmp_path / d / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    c = summary.compare({"x": tmp_path / "x", "y": tmp_path / "y"})
    assert c["verdict"] == "differ: 1 of 2 requests have different output ids across 2 traces"
    assert c["identical"] == 1 and c["pairs_identical"] == {"x = y": 1}
    (d,) = c["differ"]
    assert d["i"] == 1 and d["token"] == 1 and d["differs_from_first"] == ["y"]
    assert d["at"]["x"]["step"] == 1 and d["at"]["x"]["rows"] == 2
    assert d["at"]["x"]["row"] == "decode" and d["at"]["x"]["token"] == 6 and d["at"]["x"]["prefill_chunks"] == 1
    y = d["at"]["y"]
    assert (y["step"], y["rows"], y["row"], y["resumed"], y["preempted_before"], y["token"]) == (3, 1, "prefill", True, 1, 9)
    assert (d["at"]["x"]["path"], y["path"], y["start"], y["n"], y["forced_before"]) == \
        ("decode, decode-only step", "prefill from 0", 0, 4, 0)
    assert c["same_as_first"] == {"x": 2, "y": 1} and c["bits_per_request"] == {"mean": 0.5, "max": 1.0}
    # a's second token is a decode in a step of one row in y, not two; b's is y's resumed prefill from 0 at batch row 0
    assert c["contexts"] == {"y": {"tokens": 4, "rows_differ": 2, "batch_row_differs": 1, "path_differs": 1,
                                   "start_differs": 1, "any_differs": 2}}
    assert c["first"] == {"y": {**y, "i": 1, "index": 1}}
    text = "\n".join(summary.outputs_section(c))
    assert "| 1 | 1 | y (differs) | 9 | 3 | 1 | 4 | prefill (resumed) | prefill from 0 | 0 / 4 | 1 | 1 [0] | 2 (length) |" in text
    assert ("First divergence in `y`: request 1, output index 1 (token 9), step 3 (1 rows, 4 tokens), prefill from 0 from "
            "position 0 (4 tokens), after 1 prefill chunks and 1 preemptions (0 forced).") in text
    assert "Requests with the output ids of `x`, per trace: y: 1 of 2." in text and "mean 0.5, max 1.0" in text


def test_summary_joins_the_adversarys_choices_to_its_steps(tmp_path, mod):
    summary = mod("summary")
    d = tmp_path / "refuse-1"
    (d / "raw").mkdir(parents=True)
    # call 3 cuts a's 40-token prefill to 16 and refuses its 32-token hit; call 4 schedules nothing (no tracer line) and
    # holds b back; call 5 preempts b and finishes a's prefill over the KV of call 3's chunk
    steps = [{"step": 0, "sched_step": 3, "new": [row("a", 0, 16, 40, 0)], "cached": [], "preempted": [], "finished": [],
              "total_num_scheduled_tokens": 16, "waiting": 1, "running": 1, "empty_before": 0},
             {"step": 1, "sched_step": 5, "new": [], "cached": [row("a", 16, 24, 40, 0)], "preempted": ["b"],
              "finished": [], "total_num_scheduled_tokens": 24, "waiting": 1, "running": 1, "empty_before": 1}]
    calls = [{"step": 3, "preempted": [], "held": 1, "admissible": 1, "splits": [["a", 40, 16]], "hits": [["a", 32, 0]]},
             {"step": 4, "preempted": [], "held": 1, "admissible": 0, "splits": [], "hits": []},
             {"step": 5, "preempted": ["b"], "held": 1, "admissible": 0, "splits": [], "hits": []}]
    for name, recs in (("steps", steps), ("adversary", calls), ("outputs", [{"step": 1, "t": 0, "batch": ["a"],
                                                                             "sampled": [[9]]}]), ("requests", [])):
        (d / "raw" / f"{name}.1.jsonl").write_text("".join(json.dumps(x) + "\n" for x in recs))
    (d / "meta.json").write_text(json.dumps({"adversary": {"cache": "refuse", "seed": 1}}))
    t = summary.join_trace(d, {})
    a = t["adversary"]
    assert (a["seed"], a["cache"], a["calls"], a["forced_preemptions"], a["requests_force_preempted"]) == (1, "refuse", 3, 1, 1)
    assert (a["calls_holding_back"], a["held_share"]["max"], a["splits"], a["hits_seen"], a["hits_refused"]) == (3, 1.0, 1, 1, 1)
    joined = summary.read_jsonl(d / "steps.jsonl")
    assert joined[0]["adversary"]["splits"] == [["a", 40, 16]] and joined[1]["adversary"]["preempted"] == ["b"]
    (where,) = summary.located(d, {"a": {0}, "b": {0}}).values()
    assert (where["step"], where["path"], where["start"], where["n"], where["cut"], where["prefill_chunks"]) == \
        (1, "prefill over KV", 16, 24, False, 2)
    honest = tmp_path / "honest"
    (honest / "raw").mkdir(parents=True)
    assert summary.join_trace(honest, {})["adversary"] is None


def test_compare_corpora_refuses_different_plans_and_labels_by_variant(tmp_path, mod):
    summary = mod("summary")
    for name, variant, plan in (("c1", "seeded", "p"), ("c2", "async-seeded", "p"), ("c3", "greedy", "q")):
        root = tmp_path / name
        trace(root / "heavy", [([row("a", 0, 2, 2, 0)], ["a"], [[7]], [])], [(0, "a", [7], "stop", 0)])
        (root / "plan.jsonl").write_text(plan)
        (root / "meta.json").write_text(json.dumps({"variant": variant, "traces": {"heavy": {}}}))
    c = summary.compare_corpora([summary.corpus_spec(str(tmp_path / "c1")), summary.corpus_spec(f"{tmp_path / 'c2'}:heavy")],
                                tmp_path / "out")
    assert c["traces"] == ["seeded/heavy", "async-seeded/heavy"] and c["identical"] == 1
    assert (tmp_path / "out" / "compare.json").exists() and "identical: all 1" in (tmp_path / "out" / "COMPARE.md").read_text()
    with pytest.raises(SystemExit, match="2 different plans"):
        summary.compare_corpora([(tmp_path / "c1", None), (tmp_path / "c3", None)], tmp_path / "out2")


def test_seeded_requests_carry_the_public_seed_of_their_trace_independent_id(mod):
    corpus = mod("corpus")
    r = {"i": 12, "prompt_ids": [128000, 5], "max_tokens": 64}
    rid = corpus.request_id(12)
    assert rid == "req-00012"
    want = int.from_bytes(hashlib.sha512(b"verity-vllm/served-default-seed/v1/req-00012").digest()[:8], "big") >> 1
    assert corpus.request_seed(rid) == want and 0 <= want < 1 << 63
    seeded = corpus.request_body("m", r, rid, "seeded")
    assert (seeded["temperature"], seeded["top_p"], seeded["top_k"], seeded["seed"]) == (0.8, 0.95, 50, want)
    greedy = corpus.request_body("m", r, rid, "greedy")
    assert greedy["temperature"] == 0.0 and "seed" not in greedy and "top_k" not in greedy
    assert corpus.VARIANTS["async-seeded"] == ("seeded", True)
