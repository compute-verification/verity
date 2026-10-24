"""The active replay on a tiny synthetic trace calibrated by the library, on an injected clock: the proxy, the server and
the peer over localhost sockets, then the audit of the published records. An honest run is accepted on every
link-window, raw and committed, and is `run_shaper`'s, rule for rule; servers that signal by their write times give
its wire byte for byte, and one that holds a session past every sync is rejected."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
from verity.core.protocols.certifier import calibration
from verity.core.protocols.certifier import active
from verity.core.protocols.certifier.active import DATA, encode_frame
from verity.core.protocols.certifier.grid import COMPLETE, GridParams, Reason, Record, Status, Certifier

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
        mine = plan if strategy == "honest" else ar.adversary_plan(plan, s, strategy, seed=1)
        assert ar.run_virtual(mine, report, run) == {"server": 0, "proxy": 0, "peer": 0}
        results[strategy] = (run, ar.audit_configuration(mine, report, run))
    return plan, results


def test_requests_take_the_frames_the_trace_gives_them():
    header = active.request_placement(4096)[2]
    for n in (1, 2, 3):
        body = ar.request_bytes("s", n, 4096)
        frames = ar.request_frames(body, 4096)
        assert len(frames) == n * 4096
        assert {frames[k * 4096 : k * 4096 + active.HEADER.size] for k in range(n)} == {header}
        split = active.Requests()
        cut = [split.feed(frames[k * 4096 + active.HEADER.size : (k + 1) * 4096]) for k in range(n)]
        assert b"".join(d for _, d, _ in cut) == body and [e for *_, e in cut] == [False] * (n - 1) + [True]


def test_records_round_trip_through_their_packing():
    p = GridParams(3, 2, 64)
    w = Certifier("responses", p)
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
    assert c["released_not_the_programs"] == 0
    assert audit["timing"]["tick_late_ms"]["egress"]["max"] < 0.001


def test_a_salted_run_puts_the_salt_frame_on_ingress_only(setup, runs, tmp_path):
    """With the certifier's salt on, each window's ingress record holds one salt frame, the placeholder's, and no
    egress record holds any, nor any salt; the server receives every window's salt, the audit reads each salt frame
    and the width of the salt `reveal` gave, and every verdict and comparison is the unsalted run's. No record binds the
    salt, so another salt of the width passes the salt check, and only the server's log tells it apart."""
    _, report, _ = setup
    plan, results = runs
    assert ar.run_virtual(plan, report, tmp_path, salt=True) == {"server": 0, "proxy": 0, "peer": 0}
    salts = json.loads((tmp_path / "salts.json").read_text())
    assert sorted(salts) == ["0", "1", "2"] and all(len(bytes.fromhex(s)) == 64 for s in salts.values())
    audit = ar.audit_configuration(plan, report, tmp_path)
    honest = results["honest"][1]
    assert audit["rows"] == honest["rows"]
    assert audit["comparisons"] == honest["comparisons"]
    assert audit["salt"] == {"accepted": True, "violations": [], "ingress_salt_frames": [1, 1, 1],
                             "egress_salt_frames": 0, "egress_holds_salt": False, "block_received": True}
    assert ar.against_honest(tmp_path, results["honest"][0])["wire_is_honests"]
    assert "salt" not in honest

    for other, accepted in ((bytes(64), True), (bytes(32), False)):
        (tmp_path / "salts.json").write_text(json.dumps(dict(salts, **{"1": other.hex()})))
        audit = ar.audit_configuration(plan, report, tmp_path)
        assert audit["salt"]["accepted"] is accepted and audit["salt"]["violations"] == ([] if accepted else [[1, "salt"]])
        assert audit["salt"]["block_received"] is False and audit["rows"] == honest["rows"]


def test_a_certifier_down_for_a_window_publishes_nothing_for_it_and_the_rest_as_if_never_down(setup, tmp_path):
    """`outage_plan`: the block is drained for window 1, its proxy serves window 0 and closes the link, nothing
    listens for window 1, and a proxy back from maintenance serves window 2 on the peer's second connection. The
    audit has no verdict for window 1 on either link and accepts the others; the records of windows 0 and 2, raw and
    committed, and the wire are those of a certifier that never went down, on the same plan."""
    doc, report, s = setup
    plan = ar.plan_configuration(doc["configurations"][0], s)
    outage = ar.outage_plan(plan, s)
    assert outage["outage"] == {"window": 1, "position": 12, "bucket": 2 * T}
    assert len(outage["requests"]) == 6 and all(q["enqueue"] // T != 1 for q in outage["requests"])
    down, never = tmp_path / "down", tmp_path / "never"
    assert ar.run_virtual(outage, report, down) == {"server": 0, "proxy": 0, "peer": 0}
    always = {k: v for k, v in outage.items() if k != "outage"}
    assert ar.run_virtual(always, report, never) == {"server": 0, "proxy": 0, "peer": 0}
    audit = ar.audit_configuration(outage, report, down)
    assert [(r["link"], r["window"], r["raw"], r["committed"]) for r in audit["rows"]] == \
        [(link, w, *(("no-verdict",) * 2 if w == 1 else (None, None))) for link in ("responses", "requests")
         for w in range(3)]
    c = audit["comparisons"]
    assert c["records_are_run_shapers"] and c["hands_are_the_rules"] and c["frames_log_is_the_programs"]
    assert c["wire_is_the_records"] and c["frames_read"] == c["frames_meant"] == 24
    for name, params in (("responses.records", s.egress_params), ("requests.records", s.ingress_params)):
        assert ar.read_blobs(down / name) == [b for b in ar.read_blobs(never / name)
                                              if ar.unpack_record(b, params.frame_bytes).window != 1]
    assert sorted((down / "committed.jsonl").read_text().splitlines()) == \
        sorted(line for line in (never / "committed.jsonl").read_text().splitlines()
               if json.loads(line)["window"] != 1)
    wire = json.loads((never / "peer.json").read_text())["window_sha256"]
    assert json.loads((down / "peer.json").read_text())["window_sha256"] == [wire[0], wire[2]]
    import gzip
    assert json.load(gzip.open(down / "proxy.json.gz", "rt"))["served"] == [[0, 1], [2, 3]]


def test_an_outage_is_refused_where_the_windows_before_are_still_in_flight(setup):
    """An outage in the first or last window has no window before or after it, an online prover declares from window
    0 on, and a session of window 0 whose frames are ready only in window 1 would be cut by the outage."""
    doc, _, s = setup
    plan = ar.plan_configuration(doc["configurations"][0], s)
    for window in (0, 2):
        with pytest.raises(ValueError, match="one of windows 1"):
            ar.outage_plan(plan, s, window)
    with pytest.raises(ValueError, match="online prover"):
        ar.outage_plan(ar.plan_configuration(doc["configurations"][0], s, syncs="cover-2-1"), s)
    name, stream, ds = plan["sessions"][0]
    slow = plan | {"sessions": [[name, stream, [d + T for d in ds]], *plan["sessions"][1:]]}
    with pytest.raises(ValueError, match="ready or handed"):
        ar.outage_plan(slow, s)


def test_a_released_frame_that_isnt_the_programs_is_counted_not_a_crash(setup, runs, tmp_path):
    """The records of an honest run with one released frame replaced, as a server that numbers an event other than
    the Program does: the audit counts it, and its window is a mismatch raw (the committed records are the run's)."""
    _, report, s = setup
    plan, results = runs
    run = tmp_path / "honest"
    shutil.copytree(results["honest"][0], run)
    e = s.egress_params
    records = [ar.unpack_record(b, e.frame_bytes) for b in ar.read_blobs(run / "responses.records")]
    (run / "responses.records").unlink()
    for r in records:
        if r.window == 1:
            t = next(t for t, row in enumerate(r.grid) if row[0] is not None)
            row = (encode_frame(DATA, 99, b"id: 999\n\n", e.frame_bytes), *r.grid[t][1:])
            r = Record(r.link, r.window, r.grid[:t] + (row,) + r.grid[t + 1 :], r.status)
        ar.write_blobs(run / "responses.records", ar.pack_record(r))
    audit = ar.audit_configuration(plan, report, run)
    assert audit["comparisons"]["released_not_the_programs"] == 1
    assert [(r["link"], r["window"], r["raw"], r["committed"]) for r in audit["rows"]] == \
        [("responses", w, "mismatch" if w == 1 else None, None) for w in range(3)] + \
        [("requests", w, None, None) for w in range(3)]


@pytest.mark.parametrize("strategy", ["burst", "spread", "bits", "covert"])
def test_a_server_that_signals_by_its_write_times_leaves_the_wire_identical(runs, strategy):
    _, results = runs
    run, audit = results[strategy]
    assert ar.against_honest(run, results["honest"][0]) == {"wire_is_honests": True, "committed_is_honests": True,
                                                            "windows_whose_wire_differs": [],
                                                            "windows_whose_declared_differs": []}
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
    assert egress[0] == (0, "status:coverage-never-established", "status:coverage-never-established")
    assert all(raw == committed for _, raw, committed in egress)
    assert all(r["raw"] is None for r in audit["rows"] if r["link"] == "requests")
    assert not ar.against_honest(run, results["honest"][0])["wire_is_honests"]


def test_a_proxy_that_declares_its_own_syncs_is_accepted_and_is_run_shapers(setup, tmp_path):
    """Under an online rule the proxy's prover declares each window's sync from the frames it has read: the envelope
    (δ = 3) with no history, then a bucket past what the earlier windows needed. They are what the plan declares on the
    model's run, the audit accepts every link-window, and the records are `run_shaper`'s under them."""
    doc, report, s = setup
    plan = ar.plan_configuration(doc["configurations"][0], s, syncs="cover-2-1")
    assert plan["sync_rule"] == "cover-2-1" and [d for d, _ in plan["syncs"]] == [3, 4, 4]
    assert ar.run_virtual(plan, report, tmp_path) == {"server": 0, "proxy": 0, "peer": 0}
    audit = ar.audit_configuration(plan, report, tmp_path)
    assert all(r["raw"] is None and r["committed"] is None for r in audit["rows"])
    c = audit["comparisons"]
    assert c["declared_are_planned"] and c["records_are_run_shapers"] and c["hands_are_the_rules"]
    with pytest.raises(ValueError):
        ar.plan_configuration(doc["configurations"][0], s, syncs="cover")


def test_a_lagged_prover_declares_off_the_tick_and_is_accepted(setup, tmp_path):
    """`--prover-lag`: window w's sync is the rule's on the frames ready by w·T − lag, chosen right after that
    bucket's tick, so the tick at w·T only takes it. Every window takes the rule's choice, none the envelope for want
    of it; they are what the plan declares on the model's run, and the run is accepted and is `run_shaper`'s under
    what was declared."""
    from verity.core.protocols.certifier import online

    doc, report, s = setup
    plan = ar.plan_configuration(doc["configurations"][0], s, syncs="cover-2-1", lag=5)
    assert plan["prover_lag"] == 5
    assert ar.run_virtual(plan, report, tmp_path) == {"server": 0, "proxy": 0, "peer": 0}
    import gzip
    log = json.load(gzip.open(tmp_path / "proxy.json.gz", "rt"))
    assert log["prover"]["lag"] == 5 and log["prover"]["unready"] == [] and len(log["prover"]["choose_ms"]) == 3
    audit = ar.audit_configuration(plan, report, tmp_path)
    assert all(r["raw"] is None and r["committed"] is None for r in audit["rows"])
    assert audit["comparisons"]["records_are_run_shapers"] and audit["comparisons"]["hands_are_the_rules"]
    assert audit["comparisons"]["declared_are_planned"]
    # the rule's choice on the model's frames ready by w·T − 5, window by window
    release = {int(k): v for k, v in plan["model_release"].items()}
    frames, ready = ar.model_frames(plan["sessions"], release, plan["windows"] * T)
    want = [s.envelope]
    for w in range(1, plan["windows"]):
        seen = [(f, r) for f, r in zip(frames, ready) if r <= w * T - 5]
        p = online.Prover(s.sync_set, T, online.rule("cover-2-1"), s.envelope)
        for k, (f, r) in enumerate(seen):
            p.arrive(0, k, f, r)
        p.syncs = list(want)
        want.append(p.declare())
    assert log["syncs"] == [[x.delta, str(x.rho)] for x in want]


def test_an_online_rule_chooses_off_the_tick(setup, tmp_path):
    """The proxy never chooses at its tick: an online rule's lag is 1 … T − 1 buckets, in the plan and on the command
    line."""
    doc, _, s = setup
    for lag in (0, T):
        with pytest.raises(ValueError, match="off its tick"):
            ar.plan_configuration(doc["configurations"][0], s, syncs="last", lag=lag)
    assert "prover_lag" not in ar.plan_configuration(doc["configurations"][0], s, syncs="envelope", lag=0)
    (tmp_path / "c.json").write_text(json.dumps(setup[1]))
    with pytest.raises(SystemExit, match="--prover-lag"):
        ar.main(["t.json", "--calibration", str(tmp_path / "c.json"), "--out", str(tmp_path), "--syncs", "last",
                 "--prover-lag", "0"])


def test_the_command_line_parses_a_sync_rule(tmp_path, capsys):
    """`--syncs` takes an online rule's name and refuses anything else, as an argument error."""
    argv = ["t.json", "--calibration", "c.json", "--out", str(tmp_path)]
    with pytest.raises(SystemExit) as bad:
        ar.main([*argv, "--syncs", "cover"])
    assert bad.value.code == 2 and "--syncs" in capsys.readouterr().err
    assert ar.parse_sync_policy("cover-3-2") == "cover-3-2"


@pytest.mark.parametrize("ms, responses, requests", [
    (15, None, None),
    (110, "status:delivery-refused", "status:delivery-refused"),
    (300, "status:monitor-unavailable", "status:monitor-unavailable"),
])
def test_a_stalled_proxy_fails_its_window_closed(setup, runs, tmp_path, ms, responses, requests):
    """The proxy stopped right after window 1's first release (`--stall`): under the slack nothing faults; 110 ms,
    between two ticks, forwards the request late (`delivery-refused`), though no tick is late enough to fault; 300 ms
    also makes the next boundary's ticks late, and its ingress tick, first (`active.TICK_ORDER`), names its window's
    reason before the egress tick finds the late forward. Only window 1 fails, both links, and nowhere is there a
    mismatch."""
    _, report, _ = setup
    plan, _ = runs
    run = tmp_path / "stalled"
    ar.run_virtual(plan, report, run, stalls=[(T + 1, ms / 1000)])
    assert [(s["tick"], s["paused_ms"]) for s in json.loads((run / "stalls.json").read_text())] == \
        [(T + 1, ms)]
    audit = ar.audit_configuration(plan, report, run)
    assert [(r["link"], r["window"], r["raw"], r["committed"]) for r in audit["rows"]] == \
        [("responses", w, responses if w == 1 else None, responses if w == 1 else None) for w in range(3)] + \
        [("requests", w, requests if w == 1 else None, requests if w == 1 else None) for w in range(3)]
    assert audit["comparisons"]["released_not_the_programs"] == 0
    assert audit["comparisons"]["frames_log_is_the_programs"]


def test_a_window_its_online_declaration_misses_fails_closed(setup, tmp_path):
    """Under `last`, a 110 ms stall after window 0's first release perturbs what the prover read of window 0, and it
    declares window 1 a δ the Program's frames don't meet. Frames read after their hand times are handed late, which
    faults window 1 `coverage-never-established`: an honest run's miss fails closed instead of a `mismatch`."""
    doc, report, s = setup
    plan = ar.plan_configuration(doc["configurations"][0], s, syncs="last")
    ar.run_virtual(plan, report, tmp_path, stalls=[(1, 0.11)])
    audit = ar.audit_configuration(plan, report, tmp_path)
    assert [(r["link"], r["window"], r["raw"]) for r in audit["rows"] if r["raw"]] == \
        [("responses", 0, "status:delivery-refused"), ("responses", 1, "status:coverage-never-established"),
         ("requests", 0, "status:delivery-refused")]
    assert audit["comparisons"]["mismatches"] == 0


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

    try:
        asyncio.run(go())
        return plan, run, ar.audit_configuration(plan, report, run)
    finally:
        mp.undo()


def test_a_live_server_behind_the_shim_is_accepted(live, setup):
    plan, run, audit = live
    _, _, s = setup
    assert plan["syncs"] == [[s.envelope.delta, str(s.envelope.rho)]] * 3
    assert all(r["raw"] is None and r["committed"] is None for r in audit["rows"]) and len(audit["rows"]) == 6
    c = audit["comparisons"]
    assert c["records_are_run_shapers"] and c["hands_are_the_rules"] and c["frames_log_is_the_programs"]
    assert c["wire_is_the_records"]
    # every request's TOKENS chunks, one canonical event each and none for [DONE], numbered in write order
    assert c["frames_read"] == c["frames_meant"] == len(plan["requests"]) * TOKENS
    events = ar.read_events(run / "server.events.gz")
    assert [p for p, *_ in events] == list(range(len(events)))
    assert [active.event_data(active.event_body(e)) for *_, e in events] == \
        [b" t%d" % k for *_, k, _, _ in events]
    assert audit["timing"]["server_failed"] == 0 and audit["timing"]["server_cut_at_horizon"] == 0
    assert c["hindsight"]["misses"] == 0


def _live_run(setup, run, syncs: str, strategy: str = "honest", lag: int = 1) -> tuple[dict, dict]:
    import asyncio

    doc, report, s = setup
    mp = pytest.MonkeyPatch()
    mp.setattr(ar, "live_body", lambda session, model: {"model": model, "prompt": [1, 2, 3], "max_tokens": TOKENS,
                                                        "stream": True})
    plan = ar.plan_configuration(doc["configurations"][0], s, "served", syncs, lag)
    if strategy != "honest":
        plan |= {"strategy": strategy,
                 "writes": ar.adversary_writes(plan, s.sync_set, s.egress_params.buckets, strategy, seed=1)}

    async def go():
        server = await asyncio.start_server(_fake_vllm, "127.0.0.1", 0)
        await asyncio.to_thread(ar.run_virtual, plan, report, run, server.sockets[0].getsockname()[:2])
        server.close()

    try:
        asyncio.run(go())
        return plan, ar.audit_configuration(plan, report, run)
    finally:
        mp.undo()


@pytest.mark.parametrize("syncs", ["envelope", "cover-2-1"])
def test_a_live_shim_that_holds_its_writes_is_accepted(setup, tmp_path, syncs):
    """A live endpoint that answers at once, behind a shim that holds each event to the bucket before its hand time
    (`spread`): accepted on every link-window. Under the envelope its wire is honest's; under an online rule the proxy
    declares from what it read, so the later windows' syncs, and with them the wire, may move, never window 0's."""
    _, honest = _live_run(setup, tmp_path / "honest", syncs)
    plan, audit = _live_run(setup, tmp_path / "spread", syncs, "spread")
    assert all(r["raw"] is None and r["committed"] is None for r in audit["rows"]) and len(audit["rows"]) == 6
    c = audit["comparisons"]
    assert c["records_are_run_shapers"] and c["hands_are_the_rules"] and c["frames_log_is_the_programs"]
    assert c["wire_is_the_records"]
    events = ar.read_events(tmp_path / "spread" / "server.events.gz")
    honest_events = ar.read_events(tmp_path / "honest" / "server.events.gz")
    assert len(events) == len(honest_events) == c["frames_read"]
    assert sorted(b for *_, b, _ in events) != sorted(b for *_, b, _ in honest_events)
    vs = ar.against_honest(tmp_path / "spread", tmp_path / "honest")
    if syncs == "envelope":
        assert vs["wire_is_honests"] and vs["windows_whose_declared_differs"] == []
    else:
        assert 0 not in vs["windows_whose_wire_differs"] and 0 not in vs["windows_whose_declared_differs"]
        assert vs["windows_whose_declared_differs"]


def test_a_lagged_prover_in_front_of_a_live_shim_is_accepted(setup, tmp_path):
    """The lagged prover's choices wake the driver on the injected clock, so a run behind the live shim, where nothing
    else may move the proxy, still ends: every link-window accepted, no window declared the envelope for want of a choice."""
    _, audit = _live_run(setup, tmp_path, "cover-2-1", lag=5)
    assert all(r["raw"] is None and r["committed"] is None for r in audit["rows"]) and len(audit["rows"]) == 6
    import gzip
    assert json.load(gzip.open(tmp_path / "proxy.json.gz", "rt"))["prover"]["unready"] == []


class _Stop(Exception):
    pass


class _Parent:
    """A role's parent that runs `check` on the one value the role sends once it listens or is connected, then stops
    the role at its first `recv`, before any origin."""

    def __init__(self, check) -> None:
        self.check = check

    def send(self, value) -> None:
        self.check(value)

    def recv(self):
        raise _Stop


def test_the_server_proxy_and_peer_listen_on_or_connect_to_the_host_they_are_given(setup, tmp_path):
    """`mockdc.py` runs them on its namespaces' addresses; here, on a loopback address that isn't the default."""
    import asyncio
    import socket

    doc, report, s = setup
    plan, params, host = ar.plan_configuration(doc["configurations"][0], s), ar.wire_params(s), "127.0.0.2"
    listening = []

    def reached(port: int) -> None:
        socket.create_connection((host, port)).close()
        listening.append(port)

    with pytest.raises(_Stop):
        asyncio.run(ar._server(plan, params, _Parent(reached), tmp_path, host=host))
    with pytest.raises(_Stop):
        asyncio.run(ar._proxy(plan, s, (host, 1), _Parent(reached), tmp_path, host=host))
    assert len(listening) == 2

    link = socket.create_server((host, 0))

    def connected(said: str) -> None:
        accepted, _ = link.accept()
        assert said == "connected" and accepted.getsockname() == link.getsockname()
        accepted.close()
        listening.append(said)

    with pytest.raises(_Stop):
        asyncio.run(ar._peer(plan, params, link.getsockname()[1], _Parent(connected), tmp_path, host=host))
    link.close()
    assert len(listening) == 3


def test_a_live_request_asks_for_no_token_ids():
    # vLLM puts the prompt's token ids in the first event with `return_token_ids`, longer than a frame
    body = ar.live_body(None, "served")
    assert body["stream"] and body["ignore_eos"] and "return_token_ids" not in body


def test_a_live_request_takes_the_frames_asked_for():
    body = {"model": "served", "prompt": list(range(1000, 3000)), "max_tokens": 128}
    fewest = ar.live_request_bytes("b64.3.7", body, None, 4096)
    n = active.request_frame_count(len(fewest), 4096)
    for frames in (n, n + 1):
        out = ar.live_request_bytes("b64.3.7", body, frames, 4096)
        assert len(ar.request_frames(out, 4096)) == frames * 4096
        head, _, raw = out.partition(b"\r\n\r\n")
        assert int(head.split(b"content-length: ")[1]) == len(raw)
        assert __import__("json").loads(raw) == body | {"user": "b64.3.7"}
