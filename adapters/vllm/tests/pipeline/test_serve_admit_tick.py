"""``--admit tick`` (`serve_admit.TickGate`): the certified tick in front of a vLLM-like FCFS scheduler, run in simulated
physical time, so that the batches can be compared across wall-clock jitter and against the rule in logical time alone;
what isn't certified (unnumbered, late, lost, aborted, a new run, a stalled or broken watermark); the watermark's lines;
`tick_plugin` under vLLM's own server (`TickFeed`, `FeedReader` and the plugin's engine-core loop); and serve-openai's
``--admit tick`` over HTTP."""
from __future__ import annotations

import asyncio
import heapq
import json
import queue
import random
import socket
import sys
import threading
import types
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.pipeline.test_serve_openai import CERTIFIER_BODY, Tokenizer
from verity_vllm.engine import hooks
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import serve_admit as sa
from verity_vllm.pipeline import serve_openai as so
from verity_vllm.pipeline import tick_plugin as tp

TICK = sa.Tick(bucket_us=100_000, sat_us=30_000, free_us=40_000)
RUN = "ab"


class Req:
    def __init__(self, rid: str, out: int = 1) -> None:
        self.request_id, self.left = rid, out


class Sched:
    """vLLM's V1 scheduler as the gate reads it (FCFS ``waiting``, ``skipped_waiting``, ``running``, ``finished_req_ids``):
    each step schedules every running request one token, after moving waiting ones in while it holds under ``seqs``;
    ``steps`` lists each scheduled step's request ids."""

    def __init__(self, seqs: int = 3, policy: str = "fcfs", dp: int = 1) -> None:
        self.waiting: deque[Req] = deque()
        self.skipped_waiting: deque[Req] = deque()
        self.running: list[Req] = []
        self.finished_req_ids: set[str] = set()
        self.seqs, self.steps, self.added = seqs, [], []
        self.scheduler_config = SimpleNamespace(policy=policy)
        self.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=dp))

    def add_request(self, req: Req) -> None:
        self.added.append(req.request_id)
        self.waiting.append(req)

    def has_requests(self) -> bool:
        return bool(self.waiting or self.running or self.finished_req_ids)

    def schedule(self) -> SimpleNamespace:
        while self.waiting and len(self.running) < self.seqs:
            self.running.append(self.waiting.popleft())
        rows = [r.request_id for r in self.running]
        if rows:
            self.steps.append(rows)
        self.finished_req_ids = set()
        return SimpleNamespace(total_num_scheduled_tokens=len(rows))

    def update(self) -> None:
        for r in list(self.running):
            r.left -= 1
            if not r.left:
                self.running.remove(r)
                self.finished_req_ids.add(r.request_id)

    def finish_requests(self, ids: Any, status: Any) -> list[str]:
        live = {r.request_id for r in (*self.waiting, *self.running)}
        ids = live if ids is None else live & set(ids)
        self.waiting = deque(r for r in self.waiting if r.request_id not in ids)
        self.running = [r for r in self.running if r.request_id not in ids]
        self.finished_req_ids |= ids
        return sorted(ids)


def plan(n: int = 40, seed: int = 7) -> tuple[list[float], list[int]]:
    """A fixed release sequence (the certified data): each request's release time and its output length."""
    rng = random.Random(seed)
    rel = sorted(rng.uniform(0, 3.0) for _ in range(n))
    return rel, [rng.randint(1, 8) for _ in range(n)]


def simulate(seed: int, *, gated: bool = True, n: int = 40) -> list[list[str]]:
    """Serve `plan` in simulated physical time: request i is released at rel[i] (its bucket ⌊rel/τ_b⌋), the release record
    reaches the gate up to 20 ms later and the request the engine up to 250 ms later, watermark line t up to 60 ms after
    its bucket closes, and each step takes 20 to 50 ms, all sampled from ``seed``. Without the gate each request is added
    when it reaches the engine. Returns the scheduled steps."""
    rel, out = plan(n)
    rng = random.Random(seed)
    now = 0.0
    sched = Sched()
    gate = sa.TickGate(TICK, hold_s=600.0, clock=lambda: now)
    if gated:
        sa.install(sched, gate, key=lambda rid: rid)
    bucket = [int(r * 1e6) // TICK.bucket_us for r in rel]
    events: list[tuple[float, int, str, Any]] = []
    for i in range(n):
        events.append((rel[i] + rng.uniform(0, 0.02), len(events), "release", i))
        events.append((rel[i] + rng.uniform(0.02, 0.25), len(events), "request", i))
    last = bucket[-1] + 2
    for t in range(last + 1):
        events.append(((t + 1) * TICK.bucket_us / 1e6 + rng.uniform(0, 0.06), len(events), "line", t))
    events.append(((last + 2) * TICK.bucket_us / 1e6, len(events), "end", None))
    heapq.heapify(events)
    while events or sched.has_requests() or gate.pending():
        while events and events[0][0] <= now:
            _, _, kind, x = heapq.heappop(events)
            if kind == "release":
                gate.take({"key": f"r{x}", "run": RUN, "n": x, "bucket": bucket[x]})
            elif kind == "request":
                sched.add_request(Req(f"r{x}", out[x]))
            elif kind == "line":
                gate.take({"run": RUN, "t": x, "n": sum(b <= x for b in bucket)})
            else:
                gate.take({"run": RUN, "end": True})
        if gated and not gate.ready(sched):
            now = events[0][0] if events else now + gate.poll_s
            continue
        if sched.has_requests():
            sched.schedule()
            now += rng.uniform(0.02, 0.05)
            sched.update()
        elif events:
            now = events[0][0]
    assert sorted(sched.added, key=lambda r: int(r[1:])) == [f"r{i}" for i in range(n)]
    if gated:
        s = gate.summary()
        assert (s["admitted"], s["unordered"], s["skipped"], s["late"], s["opened"]) == (n, 0, 0, 0, None)
        assert s["waits"]["n"] > 0
    return sched.steps


def logical(n: int = 40) -> list[list[str]]:
    """The certified tick in logical time alone, over the same scheduler: no clock, no records, no waits."""
    rel, out = plan(n)
    bucket = [int(r * 1e6) // TICK.bucket_us for r in rel]
    sched, T, nxt = Sched(), 0, 0
    while nxt < n or sched.has_requests():
        quiet = not (sched.waiting or sched.running)
        if not (quiet and sched.finished_req_ids):
            if quiet and nxt < n:
                T = max(T, TICK.close(bucket[nxt]))
            while nxt < n and TICK.close(bucket[nxt]) <= T:
                sched.add_request(Req(f"r{nxt}", out[nxt]))
                nxt += 1
        if sched.has_requests():
            if sched.schedule().total_num_scheduled_tokens:
                T += TICK.step(bool(sched.waiting))
            sched.update()
    return sched.steps


def test_the_batches_are_a_function_of_the_certified_releases_whatever_the_jitter():
    want = logical()
    assert len(want) > 40 and max(map(len, want)) == 3
    assert simulate(1) == simulate(2) == want
    plain = [simulate(s, gated=False) for s in (1, 2)]
    assert plain[0] != plain[1]                     # what the gate takes away: the steps followed the clock


def test_the_tick_rule_reads_whole_buckets_in_microseconds():
    assert (TICK.close(0), TICK.closed(99_999), TICK.closed(100_000), TICK.closed(250_000)) == (100_000, -1, 0, 1)
    assert sa.Tick.of_ms(100, 27.5, 41) == sa.Tick(100_000, 27_500, 41_000)
    assert TICK.step(True) == 30_000 and TICK.step(False) == 40_000
    for bad in ((0, 1, 1), (1, -1, 1), (1.5, 1, 1)):
        with pytest.raises(ValueError, match="positive integer"):
            sa.Tick(*bad)


class Gated:
    """A gate over `Sched` with a hand-turned clock."""

    def __init__(self, hold_s: float = 5.0) -> None:
        self.now = 0.0
        self.sched = Sched()
        self.log: list[dict] = []
        self.gate = sa.TickGate(TICK, hold_s=hold_s, clock=lambda: self.now, log=self.log.append)
        self.hooks = sa.install(self.sched, self.gate, key=lambda rid: rid)

    def release(self, n: int, bucket: int | None = None, run: str = RUN, key: str | None = None) -> None:
        self.gate.take({"key": key or f"r{n}", "run": run, "n": n, "bucket": bucket})

    def lines(self, *ns: int, run: str = RUN) -> None:
        for n in ns:
            self.gate.take({"run": run, "t": len(self.gate.lines) if run == self.gate.run else 0, "n": n})

    def arrive(self, rid: str, out: int = 1) -> None:
        self.sched.add_request(Req(rid, out))

    def step(self) -> bool:
        if not self.gate.ready(self.sched):
            return False
        if self.sched.has_requests():
            self.sched.schedule()
            self.sched.update()
        return True


def test_the_engine_waits_for_the_watermark_and_admits_at_the_bucket_close():
    g = Gated()
    g.release(0, bucket=0)
    g.arrive("r0")
    assert not g.step() and g.sched.added == [] and g.gate.pending()
    g.lines(1)
    assert g.step() and g.sched.added == ["r0"] and g.gate.T == TICK.close(0) + TICK.free_us
    assert g.step()                                  # the step that frees r0
    g.release(1, bucket=1)
    g.arrive("r1", out=5)
    assert not g.step()                              # quiet: the next release's bucket is the watermark's to say
    g.lines(2)
    assert g.step() and g.sched.added == ["r0", "r1"] and g.sched.steps[-1] == ["r1"]
    assert g.gate.T == TICK.close(1) + TICK.free_us
    g.release(2, bucket=2)
    g.arrive("r2")
    assert g.step() and g.step() and g.sched.added == ["r0", "r1"]   # bucket 2 hasn't closed by T: r1 steps on alone
    assert not g.step() and g.gate.T == 320_000                      # T is past bucket 2's close: its line is wanted
    g.lines(3)
    assert g.step() and g.sched.added == ["r0", "r1", "r2"] and g.sched.steps[-1] == ["r1", "r2"]
    assert g.log[0] == {"step": 0, "T": TICK.close(0), "bucket": 0, "admitted": [0, 1], "t": 0.0}


def test_a_lost_release_holds_the_later_ones_until_its_skip_and_comes_late():
    g = Gated(hold_s=5.0)
    g.release(1, bucket=0)
    g.arrive("r1")
    g.lines(2)
    assert not g.step() and g.sched.added == []      # release 0 is certified but hasn't reached the engine
    g.now = 4.9
    assert not g.step()
    g.now = 5.0
    assert g.step() and g.sched.added == ["r1"]
    g.release(0, bucket=0)
    g.arrive("r0")
    assert g.sched.added == ["r1", "r0"]
    s = g.gate.summary()
    assert (s["skipped"], s["late"], s["skipped_unseen"], s["admitted"]) == (1, 1, [], 1)
    assert [e["event"] for e in g.log if "event" in e] == ["skipped", "late"]


def test_unnumbered_aborted_passed_and_new_runs():
    g = Gated()
    g.arrive("probe", out=5)
    assert g.sched.added == ["probe"] and g.gate.counts["unordered"] == 1
    assert g.step()
    for n in range(3):
        g.release(n, bucket=0)
    g.arrive("r0")
    g.arrive("r2")
    assert g.sched.finish_requests(["r0", "probe"], "aborted") == ["probe"]   # a held request's abort drops it
    g.gate.take({"run": RUN, "n": 1, "passed": True})                         # refused before it reached the engine
    g.lines(3)
    assert g.step() and g.sched.added == ["probe"]                            # the step that frees the probe comes first
    assert g.step() and g.sched.added == ["probe", "r2"] and g.gate.counts["aborted"] == 1
    g.release(1, bucket=4, run="cd")
    g.arrive("r1")
    assert g.sched.added[-1] == "r2"                                          # held: cd's release 0 hasn't come
    g.gate.take({"key": "old", "run": RUN, "n": 9, "bucket": 0})
    g.arrive("old")
    assert g.sched.added[-1] == "old" and g.gate.counts["stale"] == 1
    g.release(0, bucket=0, run="ef")                                          # a new run: what cd held goes in at once
    assert g.sched.added[-1] == "r1" and g.gate.summary()["runs"] == 3


def test_a_stalled_or_broken_watermark_opens_the_gate():
    g = Gated(hold_s=2.0)
    g.release(0, bucket=0)
    g.arrive("r0", out=50)
    g.lines(1)
    assert g.step()
    while g.step():
        pass                                         # T runs ahead of the watermark
    g.now = 1.0
    g.lines(1)                                       # a line is progress: the stall restarts
    while g.step():
        pass
    g.now = 2.9
    assert not g.step()
    g.now = 3.0
    assert g.step() and g.gate.opened.startswith("no watermark line for 2 s")
    g.arrive("late-comer")
    assert g.sched.added[-1] == "late-comer"
    h = Gated()
    h.release(0, bucket=0)
    h.arrive("r0")
    h.gate.take({"run": RUN, "error": "reset by peer"})
    assert h.sched.added == ["r0"] and h.gate.opened == "the watermark broke: reset by peer"


def test_finishing_every_request_adds_the_held_ones_first_and_the_end_admits_what_it_never_numbered():
    g = Gated()
    g.release(0, bucket=0)
    g.arrive("r0")
    g.sched.finish_requests(None, "aborted")
    assert g.sched.added == ["r0"] and g.gate.opened == "every request finished"
    h = Gated()
    h.release(0, bucket=0)
    h.release(5, bucket=0)
    h.arrive("r0")
    h.arrive("r5")
    h.lines(1)
    h.gate.take({"run": RUN, "end": True})
    assert h.step() and h.sched.added == ["r0", "r5"] and h.gate.counts["past_end"] == 1


def test_the_header_bucket_is_checked_against_the_watermark():
    g = Gated()
    g.release(0, bucket=3)
    g.arrive("r0")
    g.lines(1)
    assert g.step() and g.gate.counts["bucket_mismatch"] == 1


def test_install_refuses_what_the_tick_does_not_order_and_comes_off():
    with pytest.raises(RuntimeError, match="FCFS"):
        sa.install(Sched(policy="priority"), sa.TickGate(TICK, hold_s=1))
    with pytest.raises(RuntimeError, match="data-parallel"):
        sa.install(Sched(dp=2), sa.TickGate(TICK, hold_s=1))
    g = Gated()
    assert {e.name for e in hooks.live("admit-tick")} >= {"add_request", "schedule", "finish_requests", "_verity_tick"}
    g.hooks.uninstall()
    assert not hasattr(g.sched, "_verity_tick") and "add_request" not in vars(g.sched)


def test_watermark_lines_as_the_certifier_sends_them():
    assert sa.watermark_line(b"0 0\n") == (0, 0) and sa.watermark_line(b"3 7\n", (2, 5)) == (3, 7)
    for line, last in ((b"1 0\n", None), (b"3 4\n", (2, 5)), (b"0 1", None), (b"01 1\n", None), (b"0  1\n", None)):
        with pytest.raises(ValueError):
            sa.watermark_line(line, last)
    got: list[dict] = []
    w = sa.Watermark("ab", got.append)
    assert w.feed(b"0 0\n1 ") and w.feed(b"2\n2 2\n") and w.close()
    assert got == [{"run": "ab", "t": 0, "n": 0}, {"run": "ab", "t": 1, "n": 2}, {"run": "ab", "t": 2, "n": 2},
                   {"run": "ab", "end": True}]
    bad: list[dict] = []
    w = sa.Watermark("ab", bad.append)
    assert not w.feed(b"0 0\n2 0\n") and not w.feed(b"3 0\n") and bad[-1]["error"].startswith("watermark (2, 0) after (0, 0)")
    cut: list[dict] = []
    w = sa.Watermark("ab", cut.append)
    w.feed(b"0 0\n0")
    assert not w.close() and "inside a line" in cut[-1]["error"]


# ---------------------------------------------------------------------------------------------------------------- vLLM's server


def config(tmp_path: Path) -> dict[str, str]:
    return {tp.TICK_ENV: json.dumps({"bucket_us": 100_000, "sat_us": 30_000, "free_us": 40_000, "hold_ms": 2000,
                                     "feed": str(tmp_path / "feed.jsonl"), "log": str(tmp_path)})}


def asgi(app: Any, scope: dict, body: list[bytes]) -> list[dict]:
    """Run one ASGI request through ``app``: ``body`` its parts, then a disconnect; returns what it sent."""
    sent: list[dict] = []
    parts = [{"type": "http.request", "body": b, "more_body": i < len(body) - 1} for i, b in enumerate(body)]

    async def receive() -> dict:
        return parts.pop(0) if parts else {"type": "http.disconnect"}

    async def send(m: dict) -> None:
        sent.append(m)

    asyncio.run(app(scope, receive, send))
    return sent


def scope_of(path: str, headers: dict[str, str]) -> dict:
    return {"type": "http", "method": "POST", "path": path, "headers": [(k.encode(), v.encode()) for k, v in headers.items()]}


GREEDY = b'{"prompt": [1], "temperature": 0}'


class App:
    def __init__(self, status: int = 200) -> None:
        self.status, self.seen = status, []

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        self.seen.append(dict(scope["headers"]))
        await send({"type": "http.response.start", "status": self.status, "headers": []})
        await send({"type": "http.response.body", "body": b""})


def test_tick_feed_writes_the_releases_and_the_watermark_the_gate_reads(tmp_path, monkeypatch):
    for k, v in config(tmp_path).items():
        monkeypatch.setenv(k, v)
    ok, refused = App(), App(400)
    feed, read = tp.TickFeed(ok), tp.FeedReader(tmp_path / "feed.jsonl")
    assert read() == []
    asgi(feed, scope_of("/v1/completions", {"x-verity-release": "0", "x-verity-release-run": "ab",
                                            "x-verity-release-bucket": "0", "x-request-id": "c0"}), [GREEDY])
    asgi(feed, scope_of("/v1/completions", {"x-verity-release": "1", "x-verity-release-run": "ab"}), [GREEDY])
    assert ok.seen[1][b"x-request-id"] == b"verity-ab-1"
    dup = asgi(feed, scope_of("/v1/completions", {"x-verity-release": "1", "x-verity-release-run": "ab"}), [GREEDY])
    assert dup[0]["status"] == 400 and len(ok.seen) == 2
    asgi(tp.TickFeed(refused), scope_of("/v1/completions", {"x-verity-release": "2", "x-verity-release-run": "ab"}), [GREEDY])
    plain = asgi(feed, scope_of("/v1/completions", {}), [GREEDY])
    assert plain[0]["status"] == 200
    wm = asgi(feed, scope_of(sa.WATERMARK_PATH, {"x-verity-release-run": "ab"}), [b"0 1", b"\n1 3\n", b""])
    assert wm[0]["status"] == 200
    bad = asgi(feed, scope_of(sa.WATERMARK_PATH, {"x-verity-release-run": "cd"}), [b"0 1\n5 1\n"])
    assert bad[0]["status"] == 400
    cut = asgi(feed, scope_of(sa.WATERMARK_PATH, {"x-verity-release-run": "ef"}), [])
    assert cut == []
    assert read() == [{"key": "c0", "run": "ab", "n": 0, "bucket": 0}, {"key": "verity-ab-1", "run": "ab", "n": 1, "bucket": None},
                      {"key": "verity-ab-2", "run": "ab", "n": 2, "bucket": None}, {"run": "ab", "n": 2, "passed": True},
                      {"run": "ab", "t": 0, "n": 1}, {"run": "ab", "t": 1, "n": 3}, {"run": "ab", "end": True},
                      {"run": "cd", "t": 0, "n": 1}, {"run": "cd", "error": "watermark (5, 1) after (0, 1): t steps by 1 from 0 "
                                                                            "and n never decreases"},
                      {"run": "ef", "error": "the connection closed before the watermark's end"}]
    with open(tmp_path / "feed.jsonl", "ab") as f:
        f.write(b'{"run": "ab", "t": 2, ')
    assert read() == []
    with open(tmp_path / "feed.jsonl", "ab") as f:
        f.write(b'"n": 3}\n')
    assert read() == [{"run": "ab", "t": 2, "n": 3}]


class BodyApp(App):
    """`App` that reads the whole request body first, as vLLM's server does, keeping each request's."""

    def __init__(self) -> None:
        super().__init__()
        self.bodies: list[bytes] = []

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        body, more = b"", True
        while more:
            m = await receive()
            body, more = body + m.get("body", b""), m.get("more_body", False)
        self.bodies.append(body)
        await super().__call__(scope, receive, send)


def test_tick_feed_refuses_a_stochastic_request_without_a_seed(tmp_path, monkeypatch):
    for k, v in config(tmp_path).items():
        monkeypatch.setenv(k, v)
    app = BodyApp()
    feed, read = tp.TickFeed(app), tp.FeedReader(tmp_path / "feed.jsonl")
    numbered = {"x-verity-release": "0", "x-verity-release-run": "ab"}
    for path in sorted(tp.SAMPLED_PATHS):
        for body in ({"prompt": [1], "temperature": 0.8}, {"prompt": [1]}, {"temperature": 0.8, "seed": None},
                     {"temperature": 0.8, "seed": "7"}, {"temperature": "0", "seed": 1.0}):
            sent = asgi(feed, scope_of(path, {}), [json.dumps(body).encode()])
            assert sent[0]["status"] == 400 and b"needs an integer seed" in sent[1]["body"], (path, body)
    refused = asgi(feed, scope_of("/v1/completions", numbered), [b'{"temperature": 1', b".0}"])
    assert b"temperature 1.0 needs an integer seed, not null" in refused[1]["body"]
    assert app.bodies == [] and read() == [{"run": "ab", "n": 0, "passed": True}]
    assert asgi(feed, scope_of("/v1/completions", numbered), [GREEDY])[0]["status"] == 400

    passing = [b'{"prompt": [1], "temperature": 0.8, "seed": 7}', b'{"prompt": [1], "temperature": 0}',
               b'{"temperature": 0.0, "seed": null}', b"not json", b"[1, 2]", b""]
    for body in passing:
        assert asgi(feed, scope_of("/v1/chat/completions", {}), [body[:5], body[5:]])[0]["status"] == 200
    assert asgi(feed, scope_of("/v1/models", {}), [b"{}"])[0]["status"] == 200
    assert app.bodies == [*passing, b"{}"]


def test_tick_feed_serves_only_its_allow_list_under_any_root_path(tmp_path, monkeypatch):
    for k, v in config(tmp_path).items():
        monkeypatch.setenv(k, v)
    app = BodyApp()
    feed, read = tp.TickFeed(app), tp.FeedReader(tmp_path / "feed.jsonl")
    numbered = {"x-verity-release": "0", "x-verity-release-run": "ab"}
    responses = asgi(feed, scope_of("/v1/responses", numbered), [b'{"input": "hi", "temperature": 0}'])
    assert responses[0]["status"] == 400 and b"not '/v1/responses'" in responses[1]["body"]
    assert read() == [{"run": "ab", "n": 0, "passed": True}]
    for path in ("/v1/completions/render", "/v1/audio/transcriptions", "/pooling", "/invocations", "/v1/completions/",
                 "//v1/completions", "/"):
        assert asgi(feed, scope_of(path, {}), [GREEDY])[0]["status"] == 400, path
    assert app.bodies == []

    health = asgi(feed, {**scope_of("/health", {}), "method": "GET"}, [b""])
    assert health[0]["status"] == 200 and app.bodies == [b""]
    mounted = {"root_path": "/api"}
    assert tp.route_of({"path": "/api/v1/completions", **mounted}) == "/v1/completions"
    assert tp.route_of({"path": "/apiv1/completions", **mounted}) == "/apiv1/completions"
    assert tp.route_of({"path": "/v1/completions", "root_path": "/api/"}) == "/v1/completions"
    unseeded = asgi(feed, {**scope_of("/api/v1/completions", {}), **mounted}, [b'{"temperature": 0.8}'])
    assert unseeded[0]["status"] == 400 and b"needs an integer seed" in unseeded[1]["body"]
    assert asgi(feed, {**scope_of("/api/v1/responses", {}), **mounted}, [GREEDY])[0]["status"] == 400
    assert asgi(feed, {**scope_of("/api/v1/chat/completions", {}), **mounted}, [GREEDY])[0]["status"] == 200
    assert app.bodies == [b"", GREEDY]

    sent: list[dict] = []

    async def connect() -> dict:
        return {"type": "websocket.connect"}

    async def send(m: dict) -> None:
        sent.append(m)

    asyncio.run(feed({"type": "websocket", "path": "/v1/realtime", "headers": []}, connect, send))
    assert sent == [{"type": "websocket.close", "code": 1008}] and len(app.seen) == 2


def test_without_its_variable_the_feed_passes_everything(monkeypatch):
    monkeypatch.delenv(tp.TICK_ENV, raising=False)
    app = App()
    sent = asgi(tp.TickFeed(app), scope_of("/v1/completions", {"x-verity-release": "x"}), [b"{}"])
    assert sent[0]["status"] == 200 and len(app.seen) == 1 and tp.tick_config({}) is None
    with pytest.raises(ValueError, match="feed"):
        tp.tick_config({tp.TICK_ENV: json.dumps({"bucket_us": 1, "sat_us": 1, "free_us": 1})})
    with pytest.raises(ValueError, match="VERITY_TICK="):
        tp.tick_config({tp.TICK_ENV: "{not json"})


class EngineCoreProc:
    """vLLM's ``EngineCoreProc`` as `register`'s hooks drive it."""

    def __init__(self, sched: Any) -> None:
        self.scheduler, self.input_queue, self.aborts_queue = sched, queue.Queue(), queue.Queue()
        self.running, self.idle_calls = True, 0

    def _process_input_queue(self) -> None:
        raise AssertionError("the gated loop replaces it")

    def has_work(self) -> bool:
        return self.scheduler.has_requests()

    def is_running(self) -> bool:
        return self.running

    def _notify_idle_state_callbacks(self) -> None:
        self.idle_calls += 1

    def _handle_client_request(self, kind: str, req: Any) -> None:
        if kind == "add":
            self.scheduler.add_request(req)


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    """`register` over stand-ins of vLLM's scheduler and engine core modules; its hooks come off afterwards."""
    for k, v in config(tmp_path).items():
        monkeypatch.setenv(k, v)
    sched_mod, core_mod = types.ModuleType("vllm.v1.core.sched.scheduler"), types.ModuleType("vllm.v1.engine.core")

    class Scheduler(Sched):
        pass

    sched_mod.Scheduler, core_mod.EngineCoreProc = Scheduler, EngineCoreProc
    for name, mod in (("vllm", types.ModuleType("vllm")), ("vllm.v1", types.ModuleType("vllm.v1")),
                      ("vllm.v1.core", types.ModuleType("vllm.v1.core")), ("vllm.v1.core.sched", types.ModuleType("vllm.v1.core.sched")),
                      ("vllm.v1.engine", types.ModuleType("vllm.v1.engine")), (sched_mod.__name__, sched_mod), (core_mod.__name__, core_mod)):
        monkeypatch.setitem(sys.modules, name, mod)
    tp.register()
    tp.register()
    try:
        yield Scheduler
    finally:
        tp._PLUGIN.pop().uninstall()


def test_the_plugin_gates_the_engine_cores_input_loop(plugin, tmp_path):
    assert len(tp._PLUGIN) == 1
    sched = plugin()
    core = EngineCoreProc(sched)
    write = tp.TickFeed(App()).write
    write({"key": "c0", "run": "ab", "n": 0, "bucket": 0})
    core.input_queue.put(("add", Req("cmpl-c0-0-0123abcd")))
    core.input_queue.put(("add", Req("cmpl-probe-0-0123abcd")))
    released = threading.Event()

    def watermark() -> None:
        released.wait()
        write({"run": "ab", "t": 0, "n": 1})

    feeder = threading.Thread(target=watermark)
    feeder.start()
    poll = sched._verity_tick.ready

    def ready(s: Any) -> bool:
        ok = poll(s)
        if not ok:
            released.set()
        return ok

    sched._verity_tick.ready = ready
    core._process_input_queue()
    assert sched.added == ["cmpl-probe-0-0123abcd"]      # the probe at once; c0 waits for bucket 0's line
    sched.schedule()
    sched.update()
    core._process_input_queue()                          # at once: the step that frees the probe comes first
    assert not released.is_set()
    sched.schedule()
    sched.update()
    core._process_input_queue()                          # waits until the feeder writes bucket 0's line
    feeder.join()
    assert sched.added == ["cmpl-probe-0-0123abcd", "cmpl-c0-0-0123abcd"]
    assert sched._verity_tick.summary()["unordered"] == 1 and sched._verity_tick.T == TICK.close(0)
    logged = [json.loads(x) for x in next(tmp_path.glob("tick.*.jsonl")).read_text().splitlines()]
    assert logged[0]["event"] == "unordered" and logged[-1]["admitted"] == [0, 1]
    write({"key": "c9", "run": "ab", "n": 1, "bucket": 5})
    sched.add_request(Req("cmpl-c9-0-0123abcd"))
    assert sched.added[-1] == "cmpl-c0-0-0123abcd"
    core.running = False
    core._process_input_queue()
    assert sched._verity_tick.opened == "the engine core is shutting down" and sched.added[-1] == "cmpl-c9-0-0123abcd"


# ---------------------------------------------------------------------------------------------------------------- serve-openai


class SchedEngine:
    """vLLM's ``LLMEngine`` over `Sched`, as `so.EngineLoop` drives it: a request is unfinished from its ``add_request``
    (as the output processor holds it) whether or not the gate has let the scheduler see it; each step emits one token a
    scheduled request."""

    def __init__(self) -> None:
        self.sched, self.live, self.n = Sched(), {}, 0

    def add_request(self, request_id: str, prompt: dict, params: Any) -> None:
        self.n += 1
        req = Req(f"{request_id}-{self.n:08x}", params["max_tokens"])
        self.live[req.request_id] = request_id
        self.sched.add_request(req)

    def abort_request(self, ids: list[str]) -> None:
        internal = [k for k, v in self.live.items() if v in ids]
        self.sched.finish_requests(internal, "aborted")
        for k in internal:
            self.live.pop(k)

    def has_unfinished_requests(self) -> bool:
        return bool(self.live)

    def step(self) -> list[Any]:
        if not self.sched.has_requests():
            return []
        self.sched.schedule()
        rows = list(self.sched.running)
        self.sched.update()
        outs = []
        for r in rows:
            done = r.left == 0
            outs.append(SimpleNamespace(request_id=self.live[r.request_id], prompt_token_ids=None, finished=done,
                                        outputs=[SimpleNamespace(token_ids=[7], text="x", finish_reason="length" if done else None,
                                                                 stop_reason=None)]))
            if done:
                self.live.pop(r.request_id)
        return outs


def request(body: dict, release: int, bucket: int) -> bytes:
    data = json.dumps(body).encode()
    return (b"POST /v1/completions HTTP/1.1\r\nhost: s\r\ncontent-type: application/json\r\nx-verity-release: %d\r\n"
            b"x-verity-release-run: ab\r\nx-verity-release-bucket: %d\r\ncontent-length: %d\r\n\r\n" % (release, bucket, len(data))) + data


def read_all(c: socket.socket) -> bytes:
    out = b""
    while chunk := c.recv(1 << 16):
        out += chunk
    c.close()
    return out


def test_serve_openai_admits_at_the_tick_over_http():
    engine = SchedEngine()
    clock = [0.0]
    gate = sa.TickGate(TICK, hold_s=60.0, clock=lambda: clock[0])
    sa.install(engine.sched, gate, key=lambda rid: rid.rsplit("-", 1)[0])
    loop = so.EngineLoop(engine, None, gate=gate, sched=engine.sched)
    admit = sa.TickAdmit(loop.submit, loop.abort, loop.tick)
    held = threading.Semaphore(0)
    add = gate.add

    def counted(*a: Any) -> Any:
        got = add(*a)
        held.release()
        return got

    gate.add = counted
    server = so.Server(proto=so.PlainProtocol(True), served_name="m", max_model_len=512, tokenizer=Tokenizer(), defaults={},
                       submit=loop.submit, abort=loop.abort, clock=lambda: 7, order=admit)
    http = so.HttpThread(server, "127.0.0.1", 0)
    runner = threading.Thread(target=loop.run, daemon=True)
    runner.start()
    http.start()
    http.ready.wait()
    try:
        wm = socket.create_connection(http.address)
        wm.sendall(b"POST /verity/watermark HTTP/1.1\r\nhost: network-certifier\r\nx-verity-release-run: ab\r\n"
                   b"content-type: text/plain\r\ntransfer-encoding: chunked\r\n\r\n4\r\n0 0\n\r\n")
        c = socket.create_connection(http.address)
        c.sendall(request(dict(CERTIFIER_BODY, max_tokens=2, min_tokens=2), 0, 1))
        held.acquire()
        assert engine.sched.added == [] and engine.sched.steps == []
        wm.sendall(b"4\r\n1 1\n\r\n")
        raw = read_all(c)
        wm.sendall(b"0\r\n\r\n")
        end = read_all(wm)
    finally:
        loop.stop()
        runner.join()
        http.shutdown()
    assert raw.startswith(b"HTTP/1.1 200") and end.startswith(b"HTTP/1.1 200")
    assert len(engine.sched.added) == 1 and engine.sched.steps == [engine.sched.added] * 2
    assert gate.summary()["admitted"] == 1 and gate.T == TICK.close(1) + 2 * TICK.free_us and gate.ended
    assert admit.summary() == {"numbered": 1, "unordered": 0, "passed": 0, "watermarks": 1, "runs": 1}


def test_the_flag_takes_tick_with_the_engine_class_step_times():
    base = ["--model", "m", "--revision", "r", "--no-commit"]
    opts = cli.parse("serve-openai", [*base, "--admit", "tick", "--tick-sat-ms", "27.5", "--tick-free-ms", "41"], environ={})
    assert (opts.admit, opts.tick_bucket_ms, opts.tick_sat_ms, opts.tick_free_ms) == ("tick", 100.0, 27.5, 41.0)
    so.preflight(opts, which=lambda _: None)
    with pytest.raises(cli.UsageError, match="registered step times"):
        so.preflight(cli.parse("serve-openai", [*base, "--admit", "tick"], environ={}), which=lambda _: None)
    with pytest.raises(cli.UsageError, match="window-requests"):
        so.preflight(cli.parse("serve-openai", [*base, "--admit", "tick", "--tick-sat-ms", "1", "--tick-free-ms", "1",
                                                "--window-requests", "4"], environ={}), which=lambda _: None)
