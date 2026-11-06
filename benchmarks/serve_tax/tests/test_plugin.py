"""The plugin's changes on a stand-in for vLLM's V1 scheduler (the methods and attributes they wrap and read, as vLLM
d9105ea80 has them)."""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import servetax_plugin as P


class Request:
    def __init__(self, rid: str, prompt: list[int]):
        self.request_id, self.prompt_token_ids, self.num_prompt_tokens = rid, prompt, len(prompt)
        self.sampling_params = NS(temperature=0.0, max_tokens=4, ignore_eos=True)
        self.eos_token_id = None


class Scheduler:
    def __init__(self):
        self.requests: dict[str, Request] = {}
        self.order: list[str] = []
        self.finished: list[str] = []
        self.scheduler_config = NS(max_num_seqs=8, max_num_batched_tokens=64, policy="fcfs", async_scheduling=False)
        self.cache_config = NS(block_size=16, enable_prefix_caching=True, num_gpu_blocks=100)

    def add_request(self, request):
        self.requests[request.request_id] = request
        self.order.append(request.request_id)

    def schedule(self):
        return NS(total_num_scheduled_tokens=0)

    def update_from_output(self, so, mro):
        return {}

    def finish_requests(self, request_ids, finished_status):
        ids = [request_ids] if isinstance(request_ids, str) else list(request_ids or ())
        self.finished += ids
        return [self.requests.pop(r) for r in ids if r in self.requests]


def eid(run: str, n: int) -> str:
    return f"cmpl-{run}-{n:05d}-0-0123abcd"


def test_release_key_reads_the_client_number_inside_the_engine_id():
    assert P.release_key(eid("saturated2", 17)) == ("saturated2", 17)
    assert P.release_key("warm-00003") == ("warm", 3)
    assert P.release_key("cmpl-noid-0-0123abcd") is None


def test_the_gate_adds_in_release_order_and_the_trace_sees_that_order(tmp_path):
    sched = Scheduler()
    done = P.install(sched, {"out": str(tmp_path), "record": True, "release_order": True})
    for n in (2, 0, 3, 1):
        sched.add_request(Request(eid("light1", n), [n]))
    assert sched.order == [eid("light1", n) for n in range(4)]
    done[0].drain()
    (trace,) = tmp_path.glob("step_trace.*.jsonl")
    recs = [json.loads(x) for x in trace.read_text().splitlines()]
    assert [r["request"] for r in recs if r["kind"] == "arrive"] == sched.order
    done[-1].uninstall()
    done[0].uninstall()


def test_a_missing_number_is_skipped_after_its_hold_and_a_held_abort_never_reaches_the_scheduler():
    sched, now = Scheduler(), [0.0]
    gate = P.ReleaseGate(sched.add_request, lambda ids: sched.finish_requests(ids, "aborted"), key=P.release_key,
                         hold_s=5.0, clock=lambda: now[0])
    gate.add(Request(eid("r", 1), [1]))
    gate.add(Request(eid("r", 2), [2]))
    assert sched.order == [] and gate.held()
    assert gate.finished([eid("r", 2), "other"]) == ["other"]
    now[0] = 4.9
    gate.poll()
    assert sched.order == []
    now[0] = 5.0
    gate.poll()
    assert sched.order == [eid("r", 1)] and not gate.held()


def test_a_new_run_starts_from_zero_and_unnumbered_requests_pass():
    sched = Scheduler()
    gate = P.ReleaseGate(sched.add_request, lambda ids: None, key=P.release_key)
    gate.add(Request(eid("a", 0), [0]))
    gate.add(Request(eid("b", 0), [0]))
    gate.add(Request("plain-request", [0]))
    assert sched.order == [eid("a", 0), eid("b", 0), "plain-request"]


def test_token_commit_commits_each_window_prompt_and_sampled_ids_in_order():
    now, got, seen = [0.0], [], []
    tc = P.TokenCommit(10.0, got.append, clock=lambda: now[0], commit=lambda w, data: (seen.append(data), b"\x01" * 64)[1])
    tc.prompt(Request("a", [5, 6, 7]))
    tc.sampled({0: NS(outputs=[NS(new_token_ids=[8]), NS(new_token_ids=[])])})
    tc.tick()
    now[0] = 10.0
    tc.tick()
    tc.prompt(Request("b", [9]))
    tc.stop()
    assert [g["tokens"] for g in got] == [4, 1] and [g["window"] for g in got] == [0, 1]
    assert seen[0] == bytes([5, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0, 8, 0, 0, 0])
