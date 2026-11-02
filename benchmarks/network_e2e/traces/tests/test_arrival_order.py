"""A request's arrival on each side of the wire: the client's read of a response the warden's cells carried, the API
server's accept stamp, and the summary's arrival orders and its views of a warden trace for the replay."""
from __future__ import annotations

import asyncio
import itertools
import json
import math
import random
import re
from types import SimpleNamespace as NS


def sse(doc: dict) -> bytes:
    data = b"data: " + json.dumps(doc).encode() + b"\n\n"
    return b"%x\r\n%s\r\n" % (len(data), data)


def tokens(ids: list[int], finish: str | None = None) -> dict:
    return {"choices": [{"index": 0, "text": "x", "token_ids": ids, "finish_reason": finish, "stop_reason": None}]}


def test_read_response_times_each_event_by_the_cell_that_completed_it(mod):
    corpus = mod("corpus")
    head = b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream; charset=utf-8\r\ntransfer-encoding: chunked\r\n\r\n"
    usage = {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    data = head + b"".join([sse(tokens([11])), sse(tokens([12, 13])), sse(tokens([14], "length")),
                            sse({"choices": [], "usage": usage}), b"e\r\ndata: [DONE]\n\n\r\n0\r\n\r\n"])
    P = 50
    times = [100.0 + k for k in range(math.ceil(len(data) / P))]
    rec = corpus.read_response(data, times, P, 99.0)
    ends = [m.start() + 1 for m in re.finditer(rb"\n\n", data)]
    assert rec["status"] == 200 and rec["output_ids"] == [11, 12, 13, 14] and rec["usage"] == usage
    assert rec["chunks"] == [[100.0 + ends[k] // P - 99.0, n] for k, n in ((0, 1), (1, 2), (2, 1))]
    assert rec["chunks"][0][0] < rec["chunks"][2][0] and rec["first_token_s"] == rec["chunks"][0][0]
    assert (rec["finish_reason"], rec["stop_reason"], rec["done_s"]) == ("length", None, times[-1] - 99.0)
    bad = b'HTTP/1.1 400 Bad Request\r\ncontent-length: 16\r\n\r\n{"error": "bad"}'
    rec = corpus.read_response(bad, [5.0], 1015, 4.0)
    assert rec["status"] == 400 and "bad" in rec["error"] and rec["output_ids"] == []


def test_http_request_carries_the_id_and_closes(mod):
    corpus = mod("corpus")
    req = corpus.http_request({"prompt": [1, 2]}, "req-00007", "127.0.0.1:8000")
    head, _, body = req.partition(b"\r\n\r\n")
    assert head.startswith(b"POST /v1/completions HTTP/1.1\r\n") and b"\r\nX-Request-Id: req-00007" in head
    assert b"\r\nConnection: close" in head and f"Content-Length: {len(body)}".encode() in head
    assert json.loads(body) == {"prompt": [1, 2]}


def test_accept_stamp_records_accept_headers_and_body(tmp_path, monkeypatch, mod):
    monkeypatch.setenv("VERITY_TRACE_DIR", str(tmp_path))
    stamp = mod("vllm_accept_stamp")

    class Protocol:
        def connection_made(self, transport):
            self.made = True

    stamp.stamp_accepts(Protocol)
    stamp.stamp_accepts(Protocol)
    peer = ("127.0.0.1", 40123)
    p = Protocol()
    p.connection_made(NS(get_extra_info=lambda key: peer if key == "peername" else None))
    assert p.made and stamp.ACCEPTS[peer][1] == 0
    sent = []

    async def app(scope, receive, send):
        if scope["type"] == "http":
            assert (await receive())["body"] == b"{}"
            await send({"type": "http.response.start", "status": 200})

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(msg):
        sent.append(msg)

    mw = stamp.AcceptStamp(app)
    scope = {"type": "http", "path": "/v1/completions", "client": peer, "headers": [(b"x-request-id", b"req-00003")]}
    for _ in range(2):
        asyncio.run(mw(scope, receive, send))
    asyncio.run(mw({"type": "lifespan"}, receive, send))
    (f,) = tmp_path.glob("accept.*.jsonl")
    a, b = (json.loads(x) for x in f.read_text().splitlines())
    assert (a["id"], a["path"], a["peer"], a["conn_request"], b["conn_request"]) == ("req-00003", "/v1/completions",
                                                                                      list(peer), 1, 2)
    assert a["accept"] == b["accept"] <= a["headers"] <= a["body"] and len(sent) == 2
    assert not list(tmp_path.glob("accept_errors.*"))


def test_inversions_count_the_pairs_put_the_other_way_round(mod):
    summary = mod("summary")
    rng = random.Random(7)
    for n in (0, 1, 2, 5, 40):
        ref = [f"r{k}" for k in range(n)]
        order = ref[:]
        rng.shuffle(order)
        pos = {x: k for k, x in enumerate(ref)}
        want = sum(1 for a, b in itertools.combinations(order, 2) if pos[a] > pos[b])
        assert summary.inversions(order, ref) == want
    assert summary.inversions(["b", "x", "a"], ["a", "b"]) == 1


def warden_trace(root):
    """A corpus of one trace through the warden: four requests sent in order r0..r3 and admitted together at bucket 1;
    r0's request takes four ingress buckets, so the warden releases r1, r2 (bucket 2) and r3 (3) before it (4), and the
    engine schedules them in that order."""
    d = root / "w"
    (d / "raw").mkdir(parents=True)
    (d / "warden").mkdir()
    released = {"r0": 4, "r1": 2, "r2": 2, "r3": 3}
    engine = ["r1", "r2", "r3", "r0"]
    arrival = {"r1": 1000.30, "r2": 1000.31, "r3": 1000.40, "r0": 1000.50}

    def row(rid):
        return {"id": rid, "num_computed_tokens": 0, "num_scheduled_tokens": 4, "prompt_len": 4, "resumed": False,
                "num_output_tokens": 0, "cache_hit": 0}

    steps = [{"step": k, "sched_step": k, "t": 1000.6 + k / 10, "new": [row(x) for x in new], "cached": [],
              "total_num_scheduled_tokens": 4 * len(new), "preempted": [], "finished": [], "waiting": 0, "running": 1,
              "empty_before": 0} for k, new in enumerate([["r1", "r2"], ["r3"], ["r0"]])]
    files = {
        "steps": steps,
        "outputs": [],
        "requests": [{"id": x, "engine_id": f"cmpl-{x}-0-0123abcd", "arrival": arrival[x], "t_free": 1002.0,
                      "prompt_ids": [1, 2, 3, 4], "output_ids": [9], "finish_reason": "length", "stop_reason": None,
                      "status": "FINISHED_LENGTH_CAPPED", "num_preemptions": 0, "max_tokens": 1, "cache_hit": 0}
                     for x in engine],
        "arrivals": [{"id": x, "t": arrival[x] + 0.01, "arrival": arrival[x], "sched_step": 0, "waiting": k, "running": 0}
                     for k, x in enumerate(engine)],
        "accept": [{"id": f"r{k}", "path": "/v1/completions", "peer": ["127.0.0.1", 50000 + k], "accept": 1000.105 + k / 1000,
                    "conn_request": 1, "headers": 1000.205 + k / 1000, "body": arrival[f"r{k}"] - 0.01} for k in range(4)]
                  + [{"id": None, "path": "/health", "accept": 990.0, "conn_request": 1, "headers": 990.0, "body": 990.0}],
    }
    for name, recs in files.items():
        (d / "raw" / f"{name}.1.jsonl").write_text("".join(json.dumps(x) + "\n" for x in recs))
    (d / "client.jsonl").write_text("".join(json.dumps(
        {"id": f"r{k}", "i": k, "send_unix": 1000.0 + k / 100, "status": 200, "output_ids": [9], "chunks": [[1.5, 1]],
         "first_token_s": 1.5, "done_s": 1.6, "warden": {"ordinal": k, "outcome": "end"}}) + "\n" for k in range(4)))
    (d / "meta.json").write_text(json.dumps({"rate": 300, "why": "w", "server_argv": ["--max-num-seqs", "256"],
                                             "warden": {"where": "beside", "egress_base": "release a (the ingress end)"},
                                             "cpus": {"server": [0, 1], "client": [2], "warden": [3]}}))
    (d / "warden" / "ticks.json").write_text(json.dumps({"bucket_ms": 100, "window": 600, "slack_ms": 25.0,
                                                         "lateness_ms": [0.2, 30.0], "tick_ms": [1.0, 2.0]}))
    (d / "warden" / "client_loop.json").write_text(json.dumps({"cpus": [2], "lags": [], "max_lag_ms": None}))
    (d / "server.log").write_text("GPU KV cache size: 1,024 tokens\n")
    params = {"cell_bytes": 1024, "slots": 1, "bucket_ms": 100, "delay": 3, "window": 600}
    (d / "warden" / "deployment.json").write_text(json.dumps({"params": params, "origin_unix": 999.95, "windows": 1}))
    (d / "warden" / "warden.json").write_text(json.dumps({"connections": [
        {"ordinal": k, "arrived": 1, "admitted": 1, "ingress_end": released[f"r{k}"], "ended": 9, "ingress_cells": 2,
         "egress_cells": 1, "bytes_in": 9, "bytes_out": 9, "failed": None, "failed_at": None, "upstream": None,
         "cut": False} for k in range(4)], "refused": 0, "lateness_ms": {"n": 10, "p50": 0.1, "p99": 0.5, "max": 0.6,
                                                                         "over_slack": 0}}))
    (root / "plan.jsonl").write_text("".join(json.dumps({"i": k}) + "\n" for k in range(4)))
    (root / "meta.json").write_text(json.dumps({"requests": 4, "traces": {"w": {}}}))
    return d


def test_summary_orders_arrivals_along_the_wire_and_writes_the_wardens_views(tmp_path, mod):
    summary = mod("summary")
    d = warden_trace(tmp_path)
    t = summary.summarize(tmp_path)["traces"]["w"]
    reqs = {r["id"]: r for r in summary.read_jsonl(d / "requests.jsonl")}
    r0 = reqs["r0"]
    assert (r0["accept_unix"], r0["conn_request"], r0["engine_add_index"], reqs["r1"]["engine_add_index"]) == \
        (1000.105, 1, 3, 0)
    assert r0["warden"] == {"ordinal": 0, "outcome": "end", "arrived": 1, "admitted": 1, "released": 4, "ended": 9,
                            "ingress_cells": 2, "egress_cells": 1, "failed": None}
    o = t["arrival_order"]
    got = {k: (v["requests"], v["inversions_vs_engine"], v["inversions_vs_previous"]) for k, v in o.items() if k != "pairs"}
    assert got == {"client send": (4, 3, None), "warden arrival": (4, 3, 0), "warden admission": (4, 3, 0),
                   "warden release": (4, 0, 3), "server accept": (4, 3, 3), "server headers": (4, 3, 0),
                   "server body": (4, 0, 3), "vLLM arrival": (4, 0, 0), "engine add": (4, 0, 0),
                   "engine first scheduled": (4, 0, 0)}
    assert o["pairs"] == 6 and math.isclose(o["warden admission"]["lag_s"]["max"], 0.05)
    w = t["warden"]
    assert (w["connections"], w["failed"], w["ingress_buckets"]["max"], w["check"]) == (4, {}, 4, None)
    assert t["views"] == ["w.admitted", "w.released"]
    for view, want in (("admitted", ["r0", "r1", "r2", "r3"]), ("released", ["r1", "r2", "r3", "r0"])):
        v = {r["id"]: r for r in summary.read_jsonl(tmp_path / f"w.{view}" / "requests.jsonl")}
        assert sorted(v, key=lambda x: v[x]["send_unix"]) == want
        for x, r in v.items():
            b = r0["warden"]["admitted"] if view == "admitted" else reqs[x]["warden"]["released"]
            assert math.floor(r["send_unix"] / 0.1 + 1e-9) == math.floor((999.95 + b * 0.1) / 0.1)
            assert r["client_send_unix"] == reqs[x]["send_unix"]
            assert math.isclose(r["send_unix"] + r["chunks"][0][0], reqs[x]["send_unix"] + 1.5, abs_tol=1e-6)
        assert (tmp_path / f"w.{view}" / "steps.jsonl").read_text() == (d / "steps.jsonl").read_text()
    text = (tmp_path / "README.md").read_text()
    assert "## The warden" in text and "## Arrival order" in text and "| warden release | 4 | 0 | 3 |" in text
    assert "D counted from the warden's release a (the ingress end)." in text
    assert "| w | 0-1 / 2 / 3 | 2 | 1 / 0 / 0 / 0 / 1 / 0 / 0 / 0 | 1 | 2.0 / 2.0 | None | None |" in text
    assert (w["ticks"]["over_slack"], w["client_loop"]["stalls"], w["egress_base"]) == (1, 0, "release a (the ingress end)")
