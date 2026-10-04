"""The active replay on a tiny synthetic trace calibrated by the library, on an injected clock: the proxy, the server and
the peer over localhost sockets, then the audit of the published records. An honest run is accepted on every
link-window, raw and committed, and is `run_shaper`'s, rule for rule; servers that signal by their write times give
its wire byte for byte, and one that holds a session past every sync is rejected."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from verity_network_warden import calibration
from verity_network_warden.active import DATA, LAST, encode_frame
from verity_network_warden.grid import COMPLETE, GridParams, Reason, Record, Status, Warden

HERE = Path(__file__).resolve().parents[1]
T = 60


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_active_replay", HERE / "active_replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ar = _load()


def trace() -> dict:
    """Three windows of T buckets, each with three sessions of four frames, one frame per bucket, the first ready 1 to 3
    buckets after the anchor; each request arrives 30 ms into its anchor's bucket, the second one in two frames."""
    frames, arrival_ms = [], []
    for w in range(3):
        for j, (offset, latency) in enumerate(((0, 1), (10, 3), (20, 2))):
            anchor = w * T + offset
            frames += [{"session": f"s{w}.{j}", "index": i, "anchor": anchor, "ready": anchor + latency + i}
                       for i in range(4)]
            arrival_ms += [anchor * 100 + 30.0] * (2 if j == 1 else 1)
    frames.sort(key=lambda f: (f["ready"], f["anchor"], f["session"], f["index"]))
    return {"bucket_ms": 100, "window_buckets": T, "frame_bytes": 4096, "queue_capacity": 1 << 16,
            "configurations": [{"name": "synthetic/w0-3", "labels": {"run": "r0", "windows": [0, 3]},
                                "egress": [{"link": "responses", "frames": frames}],
                                "ingress": [{"link": "requests", "arrival_ms": arrival_ms}]}]}


@pytest.fixture(scope="module")
def setup():
    doc = trace()
    report = calibration.calibrate(doc)
    assert report["clock_sync"]["honest_miss_rate"] == 0
    return doc, report, ar.Setup(report)


@pytest.fixture(scope="module")
def runs(setup, tmp_path_factory):
    doc, report, s = setup
    plan = ar.plan_configuration(doc["configurations"][0], s)
    out = tmp_path_factory.mktemp("runs")
    results = {}
    for strategy in ar.STRATEGIES:
        run = out / strategy
        mine = plan if strategy == "honest" else plan | {
            "writes": ar.adversary_writes(plan, s.sync_set, s.egress_params.buckets, strategy, seed=1)}
        assert ar.run_virtual(mine, report, run) == {"server": 0, "proxy": 0, "peer": 0}
        results[strategy] = (run, ar.audit_configuration(mine, report, run))
    return plan, results


def test_requests_take_the_frames_the_trace_gives_them():
    for n in (1, 2, 3):
        body = ar.request_bytes("s", n, 4096)
        frames = ar.request_frames(7, body, 4096)
        assert len(frames) == n * 4096
        kinds = [frames[k * 4096] for k in range(n)]
        assert kinds == [DATA] * (n - 1) + [LAST]


def test_records_round_trip_through_their_packing():
    p = GridParams(3, 2, 64)
    w = Warden("responses", p)
    w.enqueue(encode_frame(DATA, 9, b"id: 0\n\n", 64))
    w.enqueue(bytes(range(64)))
    w.tick()
    w.tick()
    w.fault(Reason.MONITOR_UNAVAILABLE)
    w.tick()
    record = w.close_window()
    assert record.status == Status(Reason.MONITOR_UNAVAILABLE)
    clean = Record("requests", 4, ((None, None),) * 3, COMPLETE)
    for r in (record, clean):
        assert ar.unpack_record(ar.pack_record(r), 64) == r


def test_the_plan_matches_each_session_to_its_request(setup):
    doc, _, s = setup
    plan = ar.plan_configuration(doc["configurations"][0], s)
    assert plan["unmatched_sessions"] == plan["ingress_only_requests"] == plan["model_misses"] == 0
    frames = {q["session"]: q["frames"] for q in plan["requests"]}
    assert frames == {f"s{w}.{j}": 2 if j == 1 else 1 for w in range(3) for j in range(3)}
    assert all(q["send_ms"] % 100 == 50 for q in plan["requests"])
    # enqueued at the first ingress boundary after the arrival, the bucket after the anchor, and released r_in a bucket
    r_in = s.ingress_params.allocation
    assert {int(k): v for k, v in plan["model_release"].items()} == \
        {q["stream"]: int(q["trace_ms"] // 100) + 1 + (q["frames"] - 1) // r_in for q in plan["requests"]}


def test_an_honest_run_is_accepted_and_is_run_shapers(runs):
    _, results = runs
    _, audit = results["honest"]
    assert [(r["link"], r["window"], r["raw"], r["committed"]) for r in audit["rows"]] == \
        [("responses", w, None, None) for w in range(3)] + [("requests", w, None, None) for w in range(3)]
    c = audit["comparisons"]
    assert c["records_are_run_shapers"] and c["hands_are_the_rules"] and c["frames_log_is_the_programs"]
    assert c["wire_is_the_records"]
    assert c["frames_read"] == c["frames_meant"] == 36
    assert c["ready_off_meant"] == c["releases_off_model"] == c["hands_max_diff_from_model"] == 0
    assert audit["timing"]["tick_late_ms"]["egress"]["max"] < 0.001


@pytest.mark.parametrize("strategy", ["burst", "spread", "bits"])
def test_a_server_that_signals_by_its_write_times_leaves_the_wire_identical(runs, strategy):
    _, results = runs
    run, audit = results[strategy]
    assert ar.against_honest(run, results["honest"][0]) == {"wire_is_honests": True, "committed_is_honests": True,
                                                            "windows_whose_wire_differs": []}
    assert all(r["raw"] is None and r["committed"] is None for r in audit["rows"])
    assert audit["comparisons"]["records_are_run_shapers"]


def test_the_write_times_do_differ(runs):
    plan, _ = runs
    s = ar.Setup(calibration.calibrate(trace()))
    burst, spread = (ar.adversary_writes(plan, s.sync_set, T, k) for k in ("burst", "spread"))
    assert burst.keys() == spread.keys()
    assert all([p for _, p in burst[n]] == [p for _, p in spread[n]] for n in burst)
    assert any([b for b, _ in burst[n]] != [b for b, _ in spread[n]] for n in burst)


def test_a_session_held_past_every_sync_is_rejected(runs):
    _, results = runs
    run, audit = results["late"]
    egress = [(r["window"], r["raw"], r["committed"]) for r in audit["rows"] if r["link"] == "responses"]
    assert egress[0] == (0, "mismatch", "mismatch")
    assert all(raw == committed for _, raw, committed in egress)
    assert all(r["raw"] is None for r in audit["rows"] if r["link"] == "requests")
    assert not ar.against_honest(run, results["honest"][0])["wire_is_honests"]


TOKENS = 4


async def _fake_vllm(reader, writer):
    """A streaming completions endpoint that answers at once: `max_tokens` chunk events and `[DONE]`, chunked, one event
    split across two chunks."""
    head = await reader.readuntil(b"\r\n\r\n")
    body = __import__("json").loads(await reader.readexactly(int(head.lower().split(b"content-length: ")[1].split(b"\r\n")[0])))
    assert head.startswith(b"POST /v1/completions ") and "user" not in body
    events = [b'data: {"choices":[{"text":" t%d","token_ids":[%d]}]}\n\n' % (k, k) for k in range(body["max_tokens"])]
    pieces = [events[0][:7], events[0][7:], *events[1:], b"data: [DONE]\n\n"]
    writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"
                 + b"".join(b"%x\r\n%s\r\n" % (len(p), p) for p in pieces) + b"0\r\n\r\n")
    await writer.drain()
    writer.close()


@pytest.fixture(scope="module")
def live(setup, tmp_path_factory):
    import asyncio

    doc, report, s = setup
    mp = pytest.MonkeyPatch()
    mp.setattr(ar, "live_body", lambda session, model: {"model": model, "prompt": [1, 2, 3], "max_tokens": TOKENS,
                                                        "stream": True})
    plan = ar.plan_configuration(doc["configurations"][0], s, "served", "envelope")
    run = tmp_path_factory.mktemp("live")

    async def go():
        server = await asyncio.start_server(_fake_vllm, "127.0.0.1", 0)
        host, port = server.sockets[0].getsockname()[:2]
        await asyncio.to_thread(ar.run_virtual, plan, report, run, (host, port))
        server.close()

    asyncio.run(go())
    yield plan, run, ar.audit_configuration(plan, report, run)
    mp.undo()


def test_a_live_server_behind_the_shim_is_accepted(live, setup):
    plan, run, audit = live
    _, _, s = setup
    assert plan["syncs"] == [[s.envelope.delta, str(s.envelope.rho)]] * 3
    assert all(r["raw"] is None and r["committed"] is None for r in audit["rows"]) and len(audit["rows"]) == 6
    c = audit["comparisons"]
    assert c["records_are_run_shapers"] and c["hands_are_the_rules"] and c["frames_log_is_the_programs"]
    assert c["wire_is_the_records"]
    # every request's TOKENS events and its [DONE], numbered in write order
    assert c["frames_read"] == c["frames_meant"] == len(plan["requests"]) * (TOKENS + 1)
    events = ar.read_events(run / "server.events.gz")
    assert [p for p, *_ in events] == list(range(len(events)))
    assert [e for p, _, _, _, e in events if e.endswith(b"data: [DONE]\n\n")] == \
        [b"id: %d\ndata: [DONE]\n\n" % p for p, _, k, _, _ in events if k == TOKENS]
    assert audit["timing"]["server_failed"] == 0 and audit["timing"]["server_cut_at_horizon"] == 0
    assert c["hindsight"]["misses"] == 0


def test_a_live_request_takes_the_frames_asked_for():
    body = {"model": "served", "prompt": list(range(1000, 3000)), "max_tokens": 128}
    fewest = ar.live_request_bytes("b64.3.7", body, None, 4096)
    n = -(-len(fewest) // (4096 - 9))
    for frames in (n, n + 1):
        out = ar.live_request_bytes("b64.3.7", body, frames, 4096)
        assert -(-len(out) // (4096 - 9)) == frames
        head, _, raw = out.partition(b"\r\n\r\n")
        assert int(head.split(b"content-length: ")[1]) == len(raw)
        assert __import__("json").loads(raw) == body | {"user": "b64.3.7"}
