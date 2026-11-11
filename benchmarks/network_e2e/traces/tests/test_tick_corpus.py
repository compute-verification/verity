"""``--admit tick`` in the corpus: the client's stand-in network certifier, and the network certifier's release headers
and watermark as `verity.protocols.certifier.verifier.release` makes them, against the server's real `tick_plugin.TickFeed`;
which traces take the tick, τ's calibration on a trace's busy cycle, the checkpoint's stop ids, the hook's record of the
KV pool, and what the summary keeps of a tick trace, a warden trace's releases taken from the server's feed and checked
against the certifier's log."""
from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace as NS

import pytest


async def app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b""})


async def nothing():
    return {"type": "http.request", "body": b"", "more_body": False}


async def ignore(message):
    pass


async def watermark(feed_mw, head: dict, lines: list[bytes]) -> list[dict]:
    """POST the watermark's `lines`, one body message each, to `feed_mw` under the request `head`; its answer."""
    body = [{"type": "http.request", "body": line, "more_body": True} for line in lines]
    body.append({"type": "http.request", "body": b"", "more_body": False})
    answer: list[dict] = []

    async def receive():
        return body.pop(0)

    async def send(message):
        answer.append(message)

    await feed_mw(head, receive, send)
    return answer


def test_stand_in_numbers_each_release_and_counts_it_in_its_buckets_line(mod):
    corpus = mod("corpus")
    c = corpus.StandIn(0.1, 100.0, run="r")
    assert [c.release(t) for t in (100.01, 100.05, 100.27)] == [(0, 0), (1, 0), (2, 2)]
    assert c.due() == pytest.approx(100.1) and c.close() == b"0 2\n"
    assert c.due() == pytest.approx(100.2) and c.close() == b"1 2\n"
    # a timer run a little early: the release after bucket 1's line goes to bucket 2, whatever its time says
    assert c.release(100.19) == (3, 2)
    assert not c.done(4)
    assert c.close() == b"2 4\n" and c.done(4) and not c.done(5)
    assert len(corpus.StandIn(0.1, 0.0).run) == 16


def test_stand_in_speaks_the_servers_tick_feed(tmp_path, monkeypatch, mod):
    corpus = mod("corpus")
    sa, tp = corpus.tick_modules()
    feed = tmp_path / "feed.jsonl"
    monkeypatch.setenv(tp.TICK_ENV, json.dumps({"bucket_us": 100000, "sat_us": 1000, "free_us": 2000, "feed": str(feed)}))
    feed_mw = tp.TickFeed(app)
    c = corpus.StandIn(0.1, 0.0, run="ab12")

    async def serve() -> list[dict]:
        for k, t in enumerate((0.01, 0.05, 0.27)):
            n, a = c.release(t)
            headers = {"X-Request-Id": corpus.request_id(k), sa.RELEASE_HEADER: str(n), sa.RUN_HEADER: c.run,
                       sa.BUCKET_HEADER: str(a)}
            await feed_mw({"type": "http", "path": "/v1/completions",
                           "headers": [(h.lower().encode(), v.encode()) for h, v in headers.items()]}, nothing, ignore)
        return await watermark(feed_mw, {"type": "http", "path": sa.WATERMARK_PATH,
                                         "headers": [(sa.RUN_HEADER.encode(), c.run.encode())]}, [c.close() for _ in range(3)])

    answer = asyncio.run(serve())
    assert answer[0]["status"] == 200 and c.done(3)
    assert tp.FeedReader(feed)() == [{"key": "req-00000", "run": "ab12", "n": 0, "bucket": 0},
                                     {"key": "req-00001", "run": "ab12", "n": 1, "bucket": 0},
                                     {"key": "req-00002", "run": "ab12", "n": 2, "bucket": 2},
                                     {"run": "ab12", "t": 0, "n": 2}, {"run": "ab12", "t": 1, "n": 2},
                                     {"run": "ab12", "t": 2, "n": 3}, {"run": "ab12", "end": True}]
    module, name = corpus.TICK_MIDDLEWARE.rsplit(".", 1)
    assert module == tp.__name__ and getattr(tp, name) is tp.TickFeed


def test_every_sampling_passes_the_tick_feeds_seed_rule(mod):
    corpus = mod("corpus")
    _, tp = corpus.tick_modules()
    r = {"prompt_ids": [1, 2], "max_tokens": 4}
    for sampling in corpus.SAMPLING:
        body = corpus.request_body("m", r, corpus.request_id(3), sampling)
        assert tp.unseeded(json.dumps(body).encode()) is None, sampling


# A runs from 0.0 and is freed inside its last step's update (0.2999), before that step's output stamp (0.30); the engine
# idles until B arrives at 1.0, whose outputs are back at 1.10 and 1.25
STEPS = [{"total_num_scheduled_tokens": n, "t_output": t} for n, t in ((8, 0.10), (1, 0.22), (1, 0.30), (0, None),
                                                                        (8, 1.10), (1, 1.25))]
REQS = [{"engine_add_unix": 0.0, "t_free": 0.2999}, {"engine_add_unix": 1.0, "t_free": 1.2499},
        {"engine_add_unix": None, "t_free": None}]


def test_busy_cycle_leaves_out_the_idle_wait(mod):
    summary = mod("summary")
    assert summary.busy_cycle(STEPS, REQS) == {"cycles": 3, "idle": 1, "mean_us": 116666.7}
    assert summary.busy_cycle(STEPS[:1], REQS) == {"cycles": 0, "idle": 0, "mean_us": None}


def test_tau_is_calibrated_on_a_trace_or_given(tmp_path, mod):
    corpus = mod("corpus")
    d = tmp_path / "heavy"
    d.mkdir()
    (d / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in STEPS))
    (d / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in REQS))
    t = corpus.tick_from(d)
    assert t == {"bucket_us": 100000, "sat_us": 105000, "free_us": 128333,
                 "calibrated_from": {"trace": str(d), "cycles": 3, "idle": 1, "mean_us": 116666.7}}
    assert corpus.tick_args(str(d), None) == t
    assert corpus.tick_args(None, "2.5,3") == {"bucket_us": 100000, "sat_us": 2500, "free_us": 3000,
                                               "calibrated_from": None}
    for given in ((None, None), (str(d), "1,2")):
        with pytest.raises(SystemExit, match="one of --tick-from"):
            corpus.tick_args(*given)
    with pytest.raises(SystemExit, match="positive"):
        corpus.tick_args(None, "0,1")
    (d / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in STEPS[:1]))
    with pytest.raises(SystemExit, match="no busy cycle"):
        corpus.tick_from(d)


def test_stop_ids_are_the_checkpoints_end_of_text_ids(tmp_path, mod):
    corpus = mod("corpus")

    def checkpoint(name, files):
        d = tmp_path / name
        d.mkdir()
        for f, doc in files.items():
            (d / f).write_text(json.dumps(doc))
        return d

    added = {"added_tokens": [{"id": 3, "content": "<s>"}, {"id": 154820, "content": "<|endoftext|>"}]}
    glm = checkpoint("glm", {"tokenizer_config.json": {"eos_token": {"content": "<|endoftext|>"}}, "tokenizer.json": added,
                             "config.json": {"eos_token_id": [154820, 154827, 154829]}})
    assert corpus.stop_ids(glm) == [154820, 154827, 154829]
    llama = checkpoint("llama", {"tokenizer_config.json": {"eos_token": "<|end_of_text|>"},
                                 "tokenizer.json": {"added_tokens": [{"id": 128001, "content": "<|end_of_text|>"}]},
                                 "generation_config.json": {"eos_token_id": 128009},
                                 "config.json": {"eos_token_id": 128001}})
    assert corpus.stop_ids(llama) == [128001, 128009]
    assert corpus.stop_ids(checkpoint("bare", {})) == [128001]


def test_the_tick_loads_the_gate_after_the_tracer(tmp_path, mod):
    corpus = mod("corpus")

    def entries(site):
        return (site / "verity_trace_corpus-0.0.dist-info" / "entry_points.txt").read_text().splitlines()[1:]

    assert entries(corpus.plugin_site(tmp_path / "a")) == ["verity_adversary = vllm_adversarial_plugin:register",
                                                           "verity_trace = vllm_trace_plugin:register"]
    assert entries(corpus.plugin_site(tmp_path / "b", tick=True))[2:] == [
        "verity_tick = verity_vllm.pipeline.tick_plugin:register"]


def test_hook_records_the_schedulers_kv_pool(tmp_path, monkeypatch, mod):
    plugin = mod("vllm_trace_plugin")
    monkeypatch.setenv(plugin.TRACE_DIR, str(tmp_path / "raw"))
    sched = NS(block_size=64, cache_config=NS(num_gpu_blocks=1234), kv_cache_config=NS(kv_cache_groups=[object()]))
    plugin._tracer(sched)
    (p,) = (tmp_path / "raw").glob("engine.*.json")
    assert json.loads(p.read_text()) == {"block_size": 64, "num_gpu_blocks": 1234, "kv_cache_groups": 1,
                                         "scheduler": "SimpleNamespace"}

    class Broken:
        @property
        def kv_cache_config(self):
            raise RuntimeError("no pool yet")

    monkeypatch.setenv(plugin.TRACE_DIR, str(tmp_path / "broken"))
    assert plugin._tracer(Broken()) is not None
    (err,) = (tmp_path / "broken").glob("hook_errors.*.txt")
    assert err.read_text().startswith("--- engine")


#: what the client recorded of the two requests `tick_corpus` serves, without the stand-in's releases
CLIENT = [{"id": "a", "i": 0, "status": 200, "output_ids": [5, 7], "send_unix": 9.95, "first_token_s": 0.2, "done_s": 0.3},
          {"id": "b", "i": 1, "status": 200, "output_ids": [6, 8], "send_unix": 10.1, "first_token_s": 0.15, "done_s": 0.25}]
TICK = {"bucket_us": 100000, "sat_us": 9000, "free_us": 11000, "calibrated_from": None, "origin_unix": 9.9}


def tick_corpus(root, name: str, client: list[dict]):
    """A corpus at `root` of one trace `name` served under ``--admit tick``: the hook's records of requests a and b, their
    gate's log and KV pool, and the client's `client`. Returns the trace's directory."""
    d = root / name
    (d / "raw").mkdir(parents=True)

    def row(rid, start, n, prompt, outputs):
        return {"id": rid, "num_computed_tokens": start, "num_scheduled_tokens": n, "prompt_len": prompt,
                "num_output_tokens": outputs, "resumed": False}

    steps = [{"step": 0, "sched_step": 0, "new": [row("a", 0, 4, 4, 0)], "cached": [], "total_num_scheduled_tokens": 4},
             {"step": 1, "sched_step": 1, "new": [row("b", 0, 3, 3, 0)], "cached": [row("a", 4, 1, 4, 1)],
              "total_num_scheduled_tokens": 4},
             {"step": 2, "sched_step": 2, "new": [], "cached": [row("b", 3, 1, 3, 1)], "total_num_scheduled_tokens": 1}]
    for s in steps:
        s |= {"t": 10.0 + s["step"] / 10, "preempted": [], "finished": [], "waiting": 0, "running": 1, "empty_before": 0}
    raw = {"steps": steps,
           "outputs": [{"step": 0, "t": 10.1, "batch": ["a"], "sampled": [[5]]},
                       {"step": 1, "t": 10.2, "batch": ["a", "b"], "sampled": [[7], [6]]},
                       {"step": 2, "t": 10.3, "batch": ["b"], "sampled": [[8]]}],
           "requests": [{"id": "a", "arrival": 9.96, "t_free": 10.19, "prompt_ids": [1, 2, 3, 4], "output_ids": [5, 7],
                         "max_tokens": 64, "finish_reason": "stop", "stop_reason": None, "cache_hit": 0, "num_preemptions": 0},
                        {"id": "b", "arrival": 10.11, "t_free": 10.29, "prompt_ids": [1, 2, 3], "output_ids": [6, 8],
                         "max_tokens": 2, "finish_reason": "length", "stop_reason": None, "cache_hit": 0,
                         "num_preemptions": 0}],
           "arrivals": [{"id": "a", "t": 10.0}, {"id": "b", "t": 10.15}],
           "tick": [{"step": 0, "T": 100000, "bucket": 0, "admitted": [0, 1]},
                    {"step": 1, "T": 200000, "bucket": 1, "admitted": [1, 2]}]}
    for kind, recs in raw.items():
        (d / "raw" / f"{kind}.7.jsonl").write_text("".join(json.dumps(x) + "\n" for x in recs))
    (d / "raw" / "engine.7.json").write_text(json.dumps({"block_size": 64, "num_gpu_blocks": 900, "kv_cache_groups": 1}))
    (d / "client.jsonl").write_text("".join(json.dumps(c) + "\n" for c in client))
    (d / "meta.json").write_text(json.dumps({"rate": 4.0, "why": "test", "unfinished": 0, "server_argv": []}))
    (d / "server.log").write_text("")
    (root / "plan.jsonl").write_text('{"i": 0}\n{"i": 1}\n')
    (root / "meta.json").write_text(json.dumps({"model": "m", "requests": 2, "eos_token_ids": [7], "admit": "tick",
                                                "tick": TICK, "traces": {name: {}}}))
    return d


def test_summary_keeps_a_tick_traces_releases_pool_and_gate(tmp_path, mod):
    summary = mod("summary")
    d = tick_corpus(tmp_path, "heavy", [CLIENT[0] | {"release": 0, "bucket": 0}, CLIENT[1] | {"release": 1, "bucket": 1}])
    out = summary.summarize(tmp_path)
    t = out["traces"]["heavy"]
    assert (out["admit"], out["tick"], out["eos_token_ids"]) == ("tick", TICK, [7])
    assert t["requests"]["finish"] == {"eos": 1, "length": 1}
    assert t["engine"] == {"block_size": 64, "num_gpu_blocks": 900, "kv_cache_groups": 1}
    assert json.loads((d / "engine.json").read_text()) == t["engine"]
    assert t["tick"] == {"steps": 2, "requests": 2, "events": {}, "opened": None}
    v = t["serving"]
    assert (v["requests"], v["output_tokens"], v["span_s"], v["req_per_s"]) == (2, 4, 0.4, 5.0)
    assert (v["ttft_s"]["max"], v["e2e_s"]["min"]) == (0.2, 0.25)
    reqs = summary.read_jsonl(d / "requests.jsonl")
    assert [(r["release"], r["bucket"], r["eos"]) for r in reqs] == [(0, 0, True), (1, 1, False)]
    readme = (tmp_path / "README.md").read_text()
    assert "Admission: the certified tick" in readme and "| heavy | 2 | 2 | none | - |" in readme
    assert "| heavy | 2 | 0.4 | 5.0 | 10.0 |" in readme


def test_the_tick_takes_the_warden_traces_but_not_the_hostile_ones_nor_a_warden_bucket_off_tau(monkeypatch, mod):
    corpus = mod("corpus")
    tick = corpus.tick_for(["heavy", "warden-heavy", "warden-smallkv"], None, "2,3")
    assert tick == corpus.tick_args(None, "2,3") and tick["bucket_us"] == corpus.WARDEN_SCHEDULE["bucket_ms"] * 1000
    for hostile in ("honest", "refuse-1"):
        with pytest.raises(SystemExit, match="hostile"):
            corpus.tick_for(["warden-heavy", hostile], None, "2,3")
    monkeypatch.setitem(corpus.WARDEN_SCHEDULE, "bucket_ms", 50)
    with pytest.raises(SystemExit, match=r"50 ms .* isn't τ_b, 100000 µs, so \['warden-smallkv'\]"):
        corpus.tick_for(["heavy", "warden-smallkv"], None, "2,3")
    assert corpus.tick_for(["heavy"], None, "2,3") == tick


def test_a_warden_under_the_tick_streams_the_watermark_to_the_server(tmp_path, monkeypatch, mod):
    corpus = mod("corpus")
    started = []

    class Warden:
        def __init__(self, argv, **kwargs):
            started.append(argv)
            self.stdout = io.StringIO(json.dumps({"listening": ["127.0.0.1", 1]}) + "\n")

    monkeypatch.setattr(corpus.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(corpus.subprocess, "Popen", Warden)
    (tmp_path / "warden").mkdir()
    (tmp_path / "warden" / "deployment.json").write_text('{"run_id": "ab"}')
    for on in (False, True):
        assert corpus.start_warden(tmp_path, 1, 300, 10.0, tmp_path / "key", {}, watermark=on)[1] == {"run_id": "ab"}
    assert [argv.count("--watermark") for argv in started] == [0, 1]


RUN = "ab" * 32


def asgi(head: bytes) -> dict:
    """The ASGI scope a server makes of an HTTP/1 request's head (through its blank line)."""
    line, *fields = head.partition(b"\r\n\r\n")[0].split(b"\r\n")
    method, target, _ = line.split(b" ")
    return {"type": "http", "method": method.decode(), "path": target.decode(),
            "headers": [(k.strip().lower(), v.strip()) for k, _, v in (f.partition(b":") for f in fields)]}


def test_the_servers_feed_reads_the_wardens_releases_and_watermark(tmp_path, monkeypatch, mod):
    corpus = mod("corpus")
    _, tp = corpus.tick_modules()
    monkeypatch.syspath_prepend(str(corpus.ROOT))
    from verity.protocols.certifier.verifier import release
    feed = tmp_path / "feed.jsonl"
    monkeypatch.setenv(tp.TICK_ENV, json.dumps({"bucket_us": 100000, "sat_us": 1000, "free_us": 2000, "feed": str(feed)}))
    feed_mw, run = tp.TickFeed(app), bytes.fromhex(RUN)

    def line(t: int, n: int) -> bytes:
        size, _, rest = release.watermark_chunk(t, n).partition(b"\r\n")
        return rest[:int(size, 16)]

    async def serve() -> list[dict]:
        for k, (n, a) in enumerate(((0, 0), (1, 0), (2, 2))):
            request = corpus.http_request({"prompt": "x"}, corpus.request_id(k), "127.0.0.1:1")
            await feed_mw(asgi(release.with_release(request, n, run, a)), nothing, ignore)
        return await watermark(feed_mw, asgi(release.watermark_head(run)), [line(0, 2), line(1, 2), line(2, 3)])

    assert asyncio.run(serve())[0]["status"] == 200
    assert tp.FeedReader(feed)() == [{"key": "req-00000", "run": RUN, "n": 0, "bucket": 0},
                                     {"key": "req-00001", "run": RUN, "n": 1, "bucket": 0},
                                     {"key": "req-00002", "run": RUN, "n": 2, "bucket": 2},
                                     {"run": RUN, "t": 0, "n": 2}, {"run": RUN, "t": 1, "n": 2},
                                     {"run": RUN, "t": 2, "n": 3}, {"run": RUN, "end": True}]


def warden_trace(root, released: dict[int, tuple[int, int]], schedule: dict):
    """`tick_corpus`'s requests a and b as a warden trace's, its network certifier on `schedule`: through its connections
    1 and 2, which its log releases as `released` (ordinal -> (N, a)) in run `RUN`, while the server's feed says the
    release headers gave a N 0 in bucket 0 and b N 1 in bucket 1."""
    d = tick_corpus(root, "warden-heavy", [c | {"warden": {"ordinal": k + 1, "outcome": "end"}} for k, c in enumerate(CLIENT)])
    (d / "warden").mkdir()
    (d / "warden" / "deployment.json").write_text(json.dumps({"run_id": RUN, "params": schedule | {"delay": 300},
                                                              "origin_unix": 9.9, "windows": 1}))
    conns = [{"ordinal": o, "arrived": a, "admitted": a, "ingress_end": a, "release": n, "ended": a + 3, "ingress_cells": 1,
              "egress_cells": 4, "failed": None, "cut": False} for o, (n, a) in released.items()]
    (d / "warden" / "warden.json").write_text(json.dumps({"connections": conns,
                                                          "watermark": {"last": [3, 2], "lost": [], "error": None}}))
    feed = [{"key": "a", "run": RUN, "n": 0, "bucket": 0}, {"key": "b", "run": RUN, "n": 1, "bucket": 1},
            {"run": RUN, "t": 0, "n": 1}, {"run": RUN, "t": 1, "n": 2}, {"run": RUN, "end": True}]
    (d / "raw" / "tick-feed.jsonl").write_text("".join(json.dumps(x) + "\n" for x in feed))
    return d


def test_summary_takes_a_warden_traces_releases_from_the_server_where_the_warden_agrees(tmp_path, mod):
    summary = mod("summary")
    warden_trace(tmp_path, {1: (0, 0), 2: (1, 1)}, mod("corpus").WARDEN_SCHEDULE)
    t = summary.summarize(tmp_path)["traces"]["warden-heavy"]
    assert t["tick"]["releases"] == {"from": "the server's feed (raw/tick-feed.jsonl)", "run": RUN, "released": 2,
                                     "agree": 2, "unlogged": 0, "disagree": 0, "disagreeing": []}
    assert t["tick"]["steps"] == 2 and t["warden"]["connections"] == 2
    for trace in ("warden-heavy", "warden-heavy.admitted", "warden-heavy.released"):
        reqs = summary.read_jsonl(tmp_path / trace / "requests.jsonl")
        assert [(r["release"], r["bucket"]) for r in reqs] == [(0, 0), (1, 1)]
    assert ("- warden-heavy: 2 released, 2 agree, 0 the certifier logged no release for, 0 disagree"
            in (tmp_path / "README.md").read_text())


def test_summary_leaves_out_a_release_the_warden_didnt_make(tmp_path, mod):
    summary, schedule = mod("summary"), mod("corpus").WARDEN_SCHEDULE
    d = warden_trace(tmp_path, {1: (0, 0), 2: (1, 2)}, schedule)
    r = summary.summarize(tmp_path)["traces"]["warden-heavy"]["tick"]["releases"]
    assert (r["agree"], r["disagree"], r["disagreeing"]) == (1, 1, [{"id": "b", "server": [RUN, 1, 1], "warden": [RUN, 1, 2]}])
    assert [q.get("release") for q in summary.read_jsonl(d / "requests.jsonl")] == [0, None]
    # one the warden has no record of keeps the server's; another run's releases are none of this warden's
    warden_trace(tmp_path / "unlogged", {1: (0, 0)}, schedule)
    r = summary.summarize(tmp_path / "unlogged")["traces"]["warden-heavy"]["tick"]["releases"]
    assert (r["agree"], r["unlogged"], r["disagree"]) == (1, 1, 0)
    dep = d / "warden" / "deployment.json"
    dep.write_text(json.dumps(json.loads(dep.read_text()) | {"run_id": "cd" * 32}))
    r = summary.summarize(tmp_path)["traces"]["warden-heavy"]["tick"]["releases"]
    assert (r["agree"], r["disagree"]) == (0, 2)
    assert [q.get("release") for q in summary.read_jsonl(d / "requests.jsonl")] == [None, None]
