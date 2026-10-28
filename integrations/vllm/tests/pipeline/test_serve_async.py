"""serve-openai's loop under async scheduling on CPU: for a fixed load its Commit windows see the same steps, rows and values,
and H1 gets the same rounds and admissions, as synchronously. The committer is a stand-in (`Committer`), not the native
collector; ``serve_async_commit_gpu.py`` holds the native collector's rows-sha512 roots under the two schedulers equal on an
H200, and `test_the_gpu_comparison_holds_every_async_window_committed_with_syncs_roots` its comparison here.

The engine stands in for vLLM's in-process engine core at d9105ea80: a schedule runs on the runner at once (its
``execute_model``, an empty schedule's too, then ``sample``), and under async scheduling ``step`` is
``EngineCore.step_with_batch_queue`` with a queue of two, so a call runs the next schedule and returns the previous one's
outputs, and a request whose last token is scheduled keeps its seat but isn't scheduled again (`AsyncScheduler`'s
placeholders). The rest is the server's own: `serve_openai.EngineLoop`, `serve_commit.CommitWindows` and its record's
checks, and H1's `hooks.admitting` on top of the committer's patches. A load with no early stop schedules the same batches
either way while the scheduler has seats for a step's requests and the next's; with fewer, async scheduling runs an empty
schedule between forwards, which commits nothing and admits nothing."""
from __future__ import annotations

import collections
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from tests.pipeline import test_serve_openai as TSO
from verity_vllm.pipeline import serve_commit as sc
from verity_vllm.pipeline import serve_openai as so
from verity_vllm.pipeline.serve_admit import Admission
from verity_vllm.protocols.pouw.h1 import hooks as H1H
from verity_vllm.protocols.pouw.h1 import requests as RQ


class Engine:
    """vLLM's ``LLMEngine`` over its engine core, synchronous or with the batch queue (module docstring). FCFS, at most
    ``seqs`` requests running and ``budget`` tokens a schedule, a new request's whole prompt in its first; its rows in Model
    Runner V2's batch order (`requests.sort_batch_req_ids`); each sample a function of the request's ids. ``calls`` holds each
    ``step``'s (the forward it ran, the forward whose outputs it returned), None for none."""

    def __init__(self, runner: Any, *, async_scheduling: bool, seqs: int, budget: int) -> None:
        self.runner, self.async_scheduling, self.seqs, self.budget = runner, async_scheduling, seqs, budget
        self.reqs: dict[str, dict[str, Any]] = {}
        self.waiting: collections.deque[str] = collections.deque()
        self.running: list[str] = []
        self.queue: collections.deque = collections.deque()
        self.n = self.forwards = self.empty = 0
        self.calls: list[tuple[int | None, int | None]] = []

    def add_request(self, request_id: str, prompt: dict, params: Any) -> None:
        self.n += 1
        self.runner.sampler.sampling_states.seeds.gpu[slot := self.n % 64] = params["seed"]
        self.reqs[request_id] = {"internal": f"{request_id}-{self.n:08x}", "prompt": list(prompt["prompt_token_ids"]),
                                 "max": params["max_tokens"], "slot": slot, "sampled": [], "placed": 0, "out": 0}
        self.waiting.append(request_id)

    def abort_request(self, ids: list[str]) -> None:
        raise AssertionError(f"no request is aborted here: {ids}")

    def has_unfinished_requests(self) -> bool:
        return bool(self.reqs)

    def _schedule(self) -> tuple[dict[str, tuple[list[int], int]], list[str]]:
        rows, new, left = {}, [], self.budget
        for e in self.running:
            q = self.reqs[e]
            if q["placed"] < q["max"] and left:
                rows[e], left = ([q["sampled"][-1]], len(q["prompt"]) + q["placed"] - 1), left - 1
                q["placed"] += 1
        while self.waiting and len(self.running) < self.seqs and len(self.reqs[self.waiting[0]]["prompt"]) <= left:
            q = self.reqs[e := self.waiting.popleft()]
            rows[e], left, q["placed"] = (q["prompt"], 0), left - len(q["prompt"]), 1
            self.running.append(e)
            new.append(e)
        return rows, new

    def _launch(self) -> tuple[int | None, list[tuple[str, int]]]:
        rows, new = self._schedule()
        name = {self.reqs[e]["internal"]: e for e in rows}
        live = [name[i] for i in RQ.sort_batch_req_ids({i: len(rows[e][0]) for i, e in name.items()}, {}, 1)]
        ids, qsl, pos = [], [0], []
        for e in live:
            ids += rows[e][0]
            qsl.append(len(ids))
            pos += range(rows[e][1], rows[e][1] + len(rows[e][0]))
        ib = SimpleNamespace(num_reqs=len(live), req_ids=[self.reqs[e]["internal"] for e in live], query_start_loc_np=np.array(qsl),
                             idx_mapping=torch.tensor([self.reqs[e]["slot"] for e in live], dtype=torch.int32), positions=torch.tensor(pos),
                             logits_indices=torch.tensor(qsl[1:]) - 1, num_computed_tokens_np=np.array([rows[e][1] for e in live]),
                             prefill_len_np=np.array([len(self.reqs[e]["prompt"]) for e in live]))
        self.runner.execute_model(SimpleNamespace(
            ib=ib, ids=ids, total_num_scheduled_tokens=len(ids), scheduled_spec_decode_tokens={},
            num_scheduled_tokens={self.reqs[e]["internal"]: len(rows[e][0]) for e in rows},
            scheduled_new_reqs=[SimpleNamespace(req_id=self.reqs[e]["internal"], prompt_token_ids=self.reqs[e]["prompt"], prompt_embeds=None,
                                                sampling_params=SimpleNamespace(max_tokens=self.reqs[e]["max"])) for e in new]))
        if not ids:
            self.empty += 1
            return None, []
        sampled = [(sum(ids[qsl[i]:qsl[i + 1]]) + 31 * len(self.reqs[e]["sampled"])) % 32000 for i, e in enumerate(live)]
        for e, t in zip(live, sampled):
            self.reqs[e]["sampled"].append(t)
        self.runner.sample(ib, sampled)
        self.forwards += 1
        return self.forwards - 1, list(zip(live, sampled))

    def _update(self, launched: tuple[int | None, list[tuple[str, int]]]) -> tuple[int | None, list[Any]]:
        outs = []
        for e, t in launched[1]:
            (q := self.reqs[e])["out"] += 1
            done = q["out"] >= q["max"]
            outs.append(SimpleNamespace(request_id=e, finished=done, prompt_token_ids=q["prompt"],
                                        outputs=[SimpleNamespace(token_ids=[t], text=f" w{t}", finish_reason="length" if done else None,
                                                                 stop_reason=None)]))
            if done:
                self.running.remove(e)
                del self.reqs[e]
        return launched[0], outs

    def step(self) -> list[Any]:
        if not self.async_scheduling:
            ran, outs = self._update(self._launch())
            self.calls.append((ran, ran))
            return outs
        ran = None
        if self.waiting or self.running:
            self.queue.appendleft(launched := self._launch())
            ran = launched[0]
            if len(self.queue) < 2:
                self.calls.append((ran, None))
                return []
        elif not self.queue:
            return []
        got, outs = self._update(self.queue.pop())
        self.calls.append((ran, got))
        return outs


class Committer(TSO.Committer):
    """`TSO.Committer` with `NativeHostCommitter`'s rule for a step's start: only an ``execute_model`` that schedules tokens
    begins one, so an empty schedule commits nothing. ``committed`` keeps each finalized window's bytes, step by step."""

    def __init__(self, runner: Any) -> None:
        super().__init__(runner)
        self.committed: list[list[list[tuple[str, bytes]]]] = []

    def install(self) -> None:
        bare = self.runner.execute_model
        super().install()
        hooked = self.runner.execute_model
        self.runner.execute_model = lambda s: hooked(s) if s.total_num_scheduled_tokens else bare(s)

    def finalize(self) -> Any:
        self.committed.append([[(m.name, bytes(m.host)) for m in self._layouts[s]] for s in range(self._step_index)])
        return super().finalize()


class Arm:
    """`serve_h1.H1Windows` as `CommitWindows` drives it: at each window's open, H1's `hooks.admitting` on the runner, on top of
    the committer's patches; ``rounds`` gets each admission's (window, the committed step it precedes, new requests, batch
    order)."""

    def __init__(self, runner: Any, com: Committer) -> None:
        self.runner, self.com, self.rounds, self.bare = runner, com, [], None

    def open(self, window: int) -> None:
        run = SimpleNamespace(admit=lambda new, order: self.rounds.append((window, self.com._step_index, new, order)))
        self.bare = self.runner.execute_model
        self.runner.execute_model = H1H.admitting(run, self.bare, self.runner)

    def close(self, window: int) -> dict[str, int]:
        self.runner.execute_model = self.bare
        return {"rounds": sum(r[0] == window for r in self.rounds)}


def serve(out: Path, load: list[tuple[list[int], int]], *, async_scheduling: bool, seqs: int, budget: int,
          window_steps: int) -> SimpleNamespace:
    """``load`` (each request's prompt ids and max_tokens, all admitted before the first step) served to the end."""
    out.mkdir()
    runner = TSO.Runner()
    runner.decode_query_len = 1
    com = Committer(runner)
    arm = Arm(runner, com)
    windows = sc.CommitWindows(com, out, run_id="t", window_steps=window_steps, eos_ids=[TSO.Tokenizer.eos_token_id],
                               clock=lambda: 0.0, wall=lambda: 0.0, arm=arm)
    engine = Engine(runner, async_scheduling=async_scheduling, seqs=seqs, budget=budget)
    loop = so.EngineLoop(engine, windows, clock=lambda: 0.0)
    streamed: dict[str, list[int]] = collections.defaultdict(list)
    for i, (prompt, max_tokens) in enumerate(load):
        eid = f"cmpl-{i:04x}-0"
        loop.submit(Admission(f"cmpl-{i:04x}", eid, prompt, {"seed": 1000 + i, "max_tokens": max_tokens},
                              lambda d, eid=eid: streamed[eid].extend(d.token_ids),
                              {"seed": 1000 + i, "params": {"max_tokens": max_tokens}}))
    loop.drain("the load is served")
    loop.run()
    lines = {f: [json.loads(x) for x in (out / f).read_text().splitlines()] for f in ("windows.jsonl", "requests.jsonl", "step_trace.jsonl")}
    held = [w for w in lines["windows.jsonl"] if w["steps"]]
    return SimpleNamespace(
        engine=engine, error=loop.error, empty_windows=[w for w in lines["windows.jsonl"] if not w["steps"]], streamed=dict(streamed),
        windows=[{k: w.get(k) for k in ("window", "steps", "step_roots", "run_root", "passed", "pouw_h1")} for w in held],
        committed=com.committed, rounds=arm.rounds, step_trace=lines["step_trace.jsonl"],
        requests=sorted(({k: v for k, v in r.items() if k != "times"} for r in lines["requests.jsonl"]), key=lambda r: r["engine_id"]))


def prompts(n: int, lengths: tuple[int, ...], max_tokens: int) -> list[tuple[list[int], int]]:
    return [([1] + [(7 * i + 3 * j) % 997 + 3 for j in range(lengths[i % len(lengths)] - 1)], max_tokens) for i in range(n)]


@pytest.mark.parametrize("load, seqs, budget, window_steps, overlap", [
    (prompts(24, (8,), 1), 8, 32, 2, True),          # the served prefill mix: a step's 4 prompts, seats for 8
    (prompts(24, (8,), 1), 4, 32, 2, False),         # seats for one step's prompts only: async runs an empty schedule between
    (prompts(12, (5, 6, 7, 8), 3), 16, 40, 3, True),  # prompts and decodes, three tokens each
], ids=["prefill", "prefill-one-step-of-seats", "decode"])
def test_async_scheduling_commits_and_admits_what_sync_scheduling_does(tmp_path, load, seqs, budget, window_steps, overlap):
    """Under async scheduling each ``engine.step`` runs one schedule and returns the one before's outputs (``overlap``: a
    forward runs while the outputs of the one before are delivered, checked and windowed), yet the Commit's windows hold the
    same steps, each step commits the same bytes and rows, H1's round r is the window's committed step r with the same
    admissions and batch order, and every request's record checks against what it was streamed. A window whose bound falls on
    the last forward closes before the call that only delivers its outputs, so async scheduling may close one more window,
    empty, at the stop; it holds no step."""
    kw = {"seqs": seqs, "budget": budget, "window_steps": window_steps}
    sync, asyn = (serve(tmp_path / name, load, async_scheduling=a, **kw) for name, a in (("sync", False), ("async", True)))
    assert sync.error is None and asyn.error is None
    assert all(ran == got is not None for ran, got in sync.engine.calls) and sync.engine.empty == 0
    assert asyn.engine.calls[0] == (0, None) and sync.engine.forwards == asyn.engine.forwards
    assert any(ran is not None and got is not None and ran == got + 1 for ran, got in asyn.engine.calls) is overlap
    assert asyn.engine.empty == (1 if overlap else asyn.engine.forwards)     # the last call only returns the last outputs
    assert asyn.windows == sync.windows and all(w["passed"] for w in sync.windows)
    assert asyn.committed == sync.committed and len(sync.committed) == len(sync.windows)
    assert [w["pouw_h1"] for w in asyn.empty_windows] == [{"rounds": 0}] * len(asyn.empty_windows) and not sync.empty_windows
    assert asyn.step_trace == sync.step_trace and asyn.requests == sync.requests and asyn.streamed == sync.streamed
    assert all(r["check"]["passed"] for r in sync.requests) and len(sync.requests) == len(load)
    assert asyn.rounds == sync.rounds and len(sync.rounds) == sync.engine.forwards
    for w in sync.windows:
        assert [s for v, s, _, _ in sync.rounds if v == w["window"]] == list(range(w["steps"])) == list(range(w["pouw_h1"]["rounds"]))


def test_the_gpu_comparison_holds_every_async_window_committed_with_syncs_roots():
    """`serve_async_commit_gpu.compare`, the H200 A/B's verdict over the two arms' windows: PASS only when every window of
    each arm passed and committed bytes and the two committed the same steps under equal roots, in order; a window that
    held no step (async scheduling's empty one at the stop) is left out."""
    from tests.pipeline import serve_async_commit_gpu as AB

    def win(i: int, roots: list[str], passed: bool = True) -> dict:
        return {"window": i, "steps": len(roots), "step_roots": roots, "passed": passed, "committed_bytes": 9 if passed else 0}

    sync = [win(0, ["a"]), win(1, ["b", "c"])]
    got = AB.compare(sync, [*sync, {"window": 2, "steps": 0}])
    assert got["passed"] and got["roots_equal"] == 3 and got["windows"] == [2, 2]
    assert AB.compare(sync, [win(0, ["a"]), win(1, ["b", "x"])])["differ"] == [2]
    assert AB.compare(sync, [win(0, ["a"]), win(1, ["", ""], passed=False)])["async_failed"] == [1]
    assert AB.compare([win(0, ["a"], passed=False), sync[1]], sync)["sync_failed"] == [0]
    for asy in ([win(0, ["a"])], [win(0, ["a"]), win(1, ["", ""], passed=False)], []):
        assert not AB.compare(sync, asy)["passed"]
    assert not AB.compare([], [])["passed"]
