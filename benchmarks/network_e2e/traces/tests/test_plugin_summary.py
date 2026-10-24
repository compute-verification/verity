"""The scheduler hook on a stand-in for vLLM's Scheduler (the attributes the hook reads, as vLLM d9105ea80 has them), and the
summary over what it wrote."""
from __future__ import annotations

import enum
import json
from types import SimpleNamespace as NS


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
    s.requests = {A: a, B: b}
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
    out = summary.summarize(tmp_path)
    t = out["traces"]["t"]
    assert t["requests"]["finish"] == {"eos": 1, "length": 1} and t["requests"]["client_ids_match"] == 2
    assert t["rows"] == {"total": 8, "prefill": 4, "decode": 4, "partial_prefill_chunks": 1,
                         "requests_prefilled_in_chunks": 1, "prefill_tokens_per_row": t["rows"]["prefill_tokens_per_row"]}
    assert t["preemption"] == {"events": 1, "requests_preempted": 1, "max_per_request": 1, "resumed_rows": 1}
    assert t["prefix_cache"]["requests_with_hit"] == 1 and t["prefix_cache"]["hit_tokens"] == 16
    assert t["steps"]["mixed_prefill_decode"] == 2 and t["hook_errors"] == 1
    steps = [json.loads(x) for x in (tmp_path / "t" / "steps.jsonl").read_text().splitlines()]
    assert steps[0]["batch"] == ["t-00001", "t-00000"] and steps[0]["new"][0]["kind"] == "prefill"
    assert steps[0]["new"][0]["final"] is False and steps[1]["cached"][1]["kind"] == "decode"
    reqs = [json.loads(x) for x in (tmp_path / "t" / "requests.jsonl").read_text().splitlines()]
    assert [r["first_step"] for r in reqs] == [0, 0] and reqs[0]["prefix"] == 3 and reqs[0]["eos"] is True
    assert "| t | 1.0 | test | 2 of 2 |" in (tmp_path / "README.md").read_text()


def test_summary_keeps_a_failed_trace_and_drops_a_cut_line(tmp_path, mod):
    summary = mod("summary")
    (tmp_path / "plan.jsonl").write_text('{"i": 0}\n{"i": 1')
    assert summary.read_jsonl(tmp_path / "plan.jsonl") == [{"i": 0}]
    (tmp_path / "meta.json").write_text(json.dumps({"traces": {"u": {"error": "RuntimeError: not ready"}}}))
    out = summary.summarize(tmp_path)
    assert out["traces"]["u"] == {"trace": "u", "error": "RuntimeError: not ready"}
    assert "| u | None | None | failed: RuntimeError: not ready |" in (tmp_path / "README.md").read_text()
