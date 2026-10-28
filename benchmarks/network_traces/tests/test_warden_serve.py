"""warden-serve without the wall clock: its varied traffic, its HTTP bytes against the fake endpoint over loopback, and
the offline check of a run whose warden is driven tick by tick on a `VirtualClock`, its records certified by an LMS
key: the check accepts the run, whose server got each request with its release order under the run id `deployment.json`
names, and refuses a record that isn't what was certified, a cell that reached its client in
another bucket, and a response that isn't the record's; a window in which the warden refused a connection is certified
and signed nowhere; and a window's slow commit, which faults the next window when it runs in the warden's process,
moves no tick in `serve`'s publisher process."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import pytest

from verity.core.protocols.certifier import warden_rows as WR
from verity.core.protocols.certifier.active import DATA, VirtualClock
from verity.core.protocols.certifier.grid import Reason
from verity.core.protocols.certifier.release import with_release
from verity.core.protocols.certifier.warden import (END, REFUSED, Shim, Warden, WardenParams, cells, read_cell,
                                                    reason_frame)

HERE = Path(__file__).resolve().parents[1]
PARAMS = WardenParams(cell_bytes=64, slots=2, bucket_ms=100, delay=3, lanes=2, queue=4, window=60, ingress_cells=40,
                      ingress_buckets=30, egress_cells=128)
SESSION = bytes(range(32))
#: the clocks `slow_commit` moves: a run's while it ticks in this process; a spawned process's list stays empty
PROCESS_CLOCK: list[VirtualClock] = []
#: what a window's commit takes of its process's time: the trace corpus's small-KV run measured 378-392 ms
SLOW_COMMIT_S = 0.4


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_warden_serve", HERE / "warden_serve.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ws = _load()


async def _no_sleep(_: float) -> None:
    pass


def test_varied_traffic_is_fixed_by_its_seed_and_varies():
    a = ws.varied_traffic(200, model="m", rate=0.5, seed=3)
    assert a == ws.varied_traffic(200, model="m", rate=0.5, seed=3) != ws.varied_traffic(200, model="m", seed=4)
    assert all(x["at"] <= y["at"] for x, y in zip(a, a[1:]))
    bodies = [x["body"] for x in a]
    assert all(16 <= b["max_tokens"] <= 256 and b["stream"] and b["model"] == "m" for b in bodies)
    assert {b["temperature"] for b in bodies} == set(ws.TEMPERATURES)
    assert 0 < sum("seed" in b for b in bodies) < len(bodies)
    words = [len(b["prompt"].split()) for b in bodies]
    assert min(words) < 30 and max(words) > 500


def test_the_gates_warden_rehearsal_is_a_rehearse_and_its_least_minutes_cover_every_request_counts_traffic(tmp_path):
    """The auditor host's `warden-rehearse` (`benchmarks/live_challenger/auditor.py`) runs `rehearse` on traffic the
    gate fixes, its key at the request's height (the deployed key's, 20, by default), and refuses a request whose
    minutes are under `warden_minutes` of its requests and height: for every request count the gate takes, at its
    lowest height, those minutes cover the traffic's last arrival, its tail, the rest of the window it ends in, the
    origin's delay and 30 s for the key and the check."""
    sys.path.insert(0, str(HERE.parent / "live_challenger"))
    import auditor as A
    argv = A.warden_command({"device": "warden", "requests": 7, "tokens": 64}, tmp_path, "http://127.0.0.1:40001")
    args = ws.parser().parse_args(argv[2:])
    assert (args.command, args.upstream, args.fake, args.out, args.requests, args.max_tokens, args.height) == \
        ("rehearse", ("127.0.0.1", 40001), False, tmp_path, 7, 64, 20)
    assert ws.parser().parse_args(A.warden_command({"requests": 7, "height": 10}, tmp_path, "http://h:1")[2:]).height \
        == 10
    assert (args.rate, args.seed, args.tail) == (A.WARDEN_RATE, A.WARDEN_SEED, A.WARDEN_TAIL_S)
    assert ws.parser().parse_args(A.warden_command({"requests": 7}, tmp_path, "http://h:1")[2:]).max_tokens == 256
    window_s = args.window * args.bucket_ms / 1000
    most = ws.varied_traffic(A.WARDEN_REQUESTS[1], model="m", rate=A.WARDEN_RATE, seed=A.WARDEN_SEED)
    assert ws.varied_traffic(400, model="m", rate=A.WARDEN_RATE, seed=A.WARDEN_SEED) == most[:400]
    for n in range(A.WARDEN_REQUESTS[0], A.WARDEN_REQUESTS[1] + 1):
        took = math.ceil((most[n - 1]["at"] + A.WARDEN_TAIL_S) / window_s) * window_s + args.start_in + 30
        assert took <= A.warden_minutes(n, A.REHEARSAL_HEIGHT[0]) * 60, n


def test_the_gates_load_relay_is_a_rehearse_with_its_load_first_and_its_minutes_cover_the_link(tmp_path):
    """The auditor host's `load-relay` runs `rehearse` as `warden-rehearse` does, with the load first: the store's
    stream of the request's shape relayed to the sink tunnels' local ends for its program, the endpoint given the
    gate's `LOAD_READY_S`; a request without a load passes none. The gate's shapes are `load_relay`'s, and its
    `load_minutes` cover each shape's weights stream and witness at `LOAD_RATE` and the endpoint's wait: 46 at H200."""
    sys.path.insert(0, str(HERE.parent / "live_challenger"))
    import auditor as A
    program, sinks = "ab" * 64, [("127.0.0.1", 50001 + i) for i in range(4)]
    req = {"device": "warden", "requests": 7, "scan": True,
           "load": {"shape": "tiny", "program": program, "sink_port": 9001, "streams": 4, "minutes": 16}}
    argv = A.warden_command(req, tmp_path, "http://127.0.0.1:40001", sinks=sinks, store=tmp_path / "store")
    args = ws.parser().parse_args(argv[2:])
    assert (args.load_store, args.load_shape, args.load_sinks, args.load_program, args.load_ready_s) == \
        (tmp_path / "store", "tiny", sinks, program, A.LOAD_READY_S)
    without = A.warden_command({k: v for k, v in req.items() if k != "load"}, tmp_path, "http://127.0.0.1:40001")
    assert argv[:len(without)] == without and ws.parser().parse_args(without[2:]).load_store is None
    assert args.ids and ws._params(args) == ws._params(ws.parser().parse_args(without[2:]))
    assert set(A.LOAD_SHAPES) == set(ws.lr.SHAPES)
    for shape in A.LOAD_SHAPES:
        values = ws.lr.manifest(ws.lr.SHAPES[shape])
        link = values[-1].offset + values[-1].nbytes + ws.lr.SALT_BYTES * sum(v.leaves for v in values)
        assert A.load_minutes(shape) * 60 >= link / A.LOAD_RATE + A.LOAD_READY_S > (A.load_minutes(shape) - 1) * 60
    assert (A.load_minutes("h200"), A.load_minutes("tiny")) == (46, 16)


def test_the_gates_sampling_is_on_every_request_and_moves_nothing_else_of_the_traffic(tmp_path):
    """`warden-rehearse` always passes its temperature and top_p (0 and 1 by default), and `rehearse` puts both on
    every request, so an endpoint's own generation config never applies; the arrivals, prompts, lengths and seeds stay
    the seed's, so `warden_minutes` still bounds the run."""
    sys.path.insert(0, str(HERE.parent / "live_challenger"))
    import auditor as A
    given = ws.parser().parse_args(
        A.warden_command({"requests": 7, "temperature": 0.6, "top_p": 0.9}, tmp_path, "http://h:1")[2:])
    plain = ws.parser().parse_args(A.warden_command({"requests": 7}, tmp_path, "http://h:1")[2:])
    assert (plain.temperature, plain.top_p, given.temperature, given.top_p) == (0.0, 1.0, 0.6, 0.9)
    varied = ws.varied_traffic(7, model="m", rate=A.WARDEN_RATE, seed=A.WARDEN_SEED)

    def rest(traffic):
        return [{**x, "body": {k: v for k, v in x["body"].items() if k not in ("temperature", "top_p")}}
                for x in traffic]

    for args, want in ((plain, (0.0, 1.0)), (given, (0.6, 0.9))):
        traffic = ws.traffic_of(args, "m")
        assert {(x["body"]["temperature"], x["body"]["top_p"]) for x in traffic} == {want}
        assert rest(traffic) == rest(varied)
    assert all("top_p" not in x["body"] for x in varied)


def test_the_gates_scan_rehearsal_is_rehearse_ids_at_scan_zks_schedule_and_its_least_minutes_cover_its_traffic(
        tmp_path, capsys):
    """`warden-rehearse --scan` runs `rehearse --ids` at `scan_zk.PARAMS` (the schedule `scan_zk.py prove
    --warden-run` refuses any other of), which the scan's class serves: EVERY 16 buckets, 4 Program slots a
    boundary, 8 a link-window and D 2.7 s, the gate sessions' schedule. Its id traffic's prompts and outputs are
    within the scan's KI and KO, on the gate's arrivals at the gate's sampling; for every request count the gate takes,
    at its lowest height, `warden_minutes` covers the id traffic's last arrival, its tail, the rest of the window it
    ends in, the origin's delay and 30 s for the key and the check."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent / "live_challenger"))
    import auditor as A
    import scan_zk as sz
    req = {"device": "warden", "requests": 7, "scan": True, "temperature": 0.6, "top_p": 0.9}
    args = ws.parser().parse_args(A.warden_command(req, tmp_path, "http://127.0.0.1:40001")[2:])
    p = ws._params(args)
    assert args.ids and not args.fake and p == sz.PARAMS
    sz.CLASS.serves(p)
    assert (p.every, p.program_slots, p.window_programs, p.delay) == (16, 4, 8, 27)
    assert args.prompt_max <= sz.CLASS.KI and min(args.max_tokens, p.egress_cells - 1) == sz.CLASS.KO
    assert (args.rate, args.seed, args.tail) == (A.WARDEN_RATE, A.WARDEN_SEED, A.WARDEN_TAIL_S)
    assert sz.main(["schedule"]) == 0 and capsys.readouterr().out.split() == list(A.SCAN_SCHEDULE)
    traffic = ws.traffic_of(args, "m")
    assert {(x["ids"]["temperature"], x["ids"]["top_p"]) for x in traffic} == {(0.6, 0.9)}
    drawn = ws.id_traffic(7, model="m", rate=A.WARDEN_RATE, seed=A.WARDEN_SEED, max_tokens=(1, 64))

    def rest(t):
        return [{**x, "ids": {k: v for k, v in x["ids"].items() if k not in ("temperature", "top_p")}} for x in t]
    assert rest(traffic) == rest(drawn) and {x["ids"]["top_p"] for x in drawn} == {1.0}
    plain = ws.parser().parse_args(A.warden_command({"requests": 7, "scan": True}, tmp_path, "http://h:1")[2:])
    assert {(x["ids"]["temperature"], x["ids"]["top_p"]) for x in ws.traffic_of(plain, "m")} == {(0.0, 1.0)}

    window_s = sz.PARAMS.window * sz.PARAMS.bucket_ms / 1000
    most = ws.id_traffic(A.WARDEN_REQUESTS[1], model="m", rate=A.WARDEN_RATE, seed=A.WARDEN_SEED,
                         max_tokens=(1, sz.PARAMS.egress_cells - 1))
    for n in range(A.WARDEN_REQUESTS[0], A.WARDEN_REQUESTS[1] + 1):
        took = math.ceil((most[n - 1]["at"] + A.WARDEN_TAIL_S) / window_s) * window_s + args.start_in + 30
        assert took <= A.warden_minutes(n, A.REHEARSAL_HEIGHT[0]) * 60, n


def test_the_gates_warden_windows_are_at_least_every_request_counts_link_windows_which_its_key_signs(tmp_path):
    """The gate sizes a warden rehearsal's key on `warden_windows` of its requests (`warden_height`), so for every
    request count it takes, those are at least the link-windows `rehearse` runs (its traffic's last arrival and tail):
    at the scan's schedule on the gate's id traffic and on `--prompts`' arrivals, and at `warden_serve.py`'s default
    (`WARDEN_WINDOW_MS`) on its varied traffic; a request count's traffic is the first of the most's. The gate's most
    requests are 1064 link-windows of `--prompts`, about 57 minutes, which height 13 signs and 12 doesn't."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent / "live_challenger"))
    import auditor as A
    import scan_zk as sz
    plain = ws.parser().parse_args(A.warden_command({"requests": 7}, tmp_path, "http://h:1")[2:])
    assert A.WARDEN_WINDOW_MS == plain.window * plain.bucket_ms
    assert A.SCAN["--window"] * A.SCAN["--bucket-ms"] == sz.PARAMS.window * sz.PARAMS.bucket_ms
    most, at = A.WARDEN_REQUESTS[1], dict(rate=A.WARDEN_RATE, seed=A.WARDEN_SEED)
    scan = {"ids": ws.id_traffic(most, model="m", max_tokens=(1, sz.PARAMS.egress_cells - 1), **at),
            "prompts": ws.prompt_traffic([[1]], most, model="m", params=sz.PARAMS, max_tokens=24, **at)}
    varied = ws.varied_traffic(most, model="m", **at)
    assert ws.prompt_traffic([[1]], 400, model="m", params=sz.PARAMS, max_tokens=24, **at) == scan["prompts"][:400]

    def windows(traffic, n, window_ms):
        return math.ceil((traffic[n - 1]["at"] + A.WARDEN_TAIL_S) * 1000 / window_ms)
    scan_ms = sz.PARAMS.window * sz.PARAMS.bucket_ms
    for n in range(A.WARDEN_REQUESTS[0], most + 1):
        assert all(windows(t, n, scan_ms) <= A.warden_windows(n) for t in scan.values()), n
        assert windows(varied, n, A.WARDEN_WINDOW_MS) <= A.warden_windows(n, scan=False), n
    assert (windows(scan["prompts"], 400, scan_ms), windows(scan["prompts"], most, scan_ms)) == (259, 1064)
    assert A.warden_height(windows(scan["prompts"], most, scan_ms)) == A.warden_height(A.warden_windows(most)) == 13


def test_the_gates_prompts_rehearsal_sends_its_prompts_a_group_a_span_of_windows_and_its_minutes_cover_them(tmp_path):
    """`warden-rehearse --scan --prompts --window-per-group` runs `rehearse --ids --prompts FILE --window-per-group` at
    the scan's schedule, on the gate's witness file: request j is the (j mod n)-th prompt with exactly the request's
    tokens, group g's n requests together half a bucket into link-window g·G, G the gate's `scan_group_windows`
    (`group_windows` at the scan's schedule, for every token count the gate takes), and the run exactly its groups'
    windows (`groups`), which `warden_group_minutes` covers with the origin's delay and 30 s for the key and the
    check. Without `--window-per-group` the prompts go at the gate's arrivals."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent / "live_challenger"))
    import auditor as A
    import scan_zk as sz
    prompts = [[70001, 3], [5], [7]]
    (tmp_path / A.PROMPTS).write_text(json.dumps(prompts))
    req = {"device": "warden", "requests": 7, "scan": True, "prompts": {"count": 3, "ids": 4}, "window_per_group": True,
           "tokens": 2, "height": 10, "temperature": 0.0, "top_p": 1.0}
    args = ws.parser().parse_args(A.warden_command(req, tmp_path, "http://127.0.0.1:40001")[2:])
    assert (args.ids, args.window_per_group, args.prompts, args.max_tokens, args.requests) == \
        (True, True, tmp_path / A.PROMPTS, 2, 7)
    params = ws._params(args)
    assert params == sz.PARAMS
    assert [ws.group_windows(params, t) for t in range(1, A.WARDEN_PROMPT_TOKENS[1] + 1)] == \
        [A.scan_group_windows(t) for t in range(1, A.WARDEN_PROMPT_TOKENS[1] + 1)]
    G, window_s = A.scan_group_windows(2), params.window * params.bucket_ms / 1000
    traffic = ws.traffic_of(args, "torch-loop/mlp-w8a8-h128")
    assert [x["ids"]["prompt"] for x in traffic] == [prompts[j % 3] for j in range(7)]
    assert {(x["ids"]["max_tokens"], x["ids"]["temperature"], x["ids"]["top_p"]) for x in traffic} == {(2, 0.0, 1.0)}
    assert [x["at"] for x in traffic] == [round((j // 3) * G * window_s + params.bucket_ms / 2000, 3) for j in range(7)]
    assert [x["ids"]["user"] for x in traffic] == [f"user-{j}" for j in range(7)]
    assert ws.groups(7, 3, params, 2) == [{"group": g, "requests": list(range(3 * g, min(7, 3 * g + 3))),
                                           "windows": [g * G, (g + 1) * G]} for g in range(3)]
    for groups in range(1, A.WARDEN_PROMPT_REQUESTS[1] + 1):
        for tokens in (1, 2, A.WARDEN_PROMPT_TOKENS[1]):
            took = groups * ws.group_windows(params, tokens) * window_s + args.start_in + 30
            assert took <= A.warden_group_minutes(groups, tokens, A.REHEARSAL_HEIGHT[0]) * 60, (groups, tokens)

    loose = ws.parser().parse_args(A.warden_command(req | {"window_per_group": False}, tmp_path, "u:1")[2:])
    assert not loose.window_per_group and loose.prompts == tmp_path / A.PROMPTS
    spread = ws.traffic_of(loose, "m")
    assert [x["ids"]["prompt"] for x in spread] == [x["ids"]["prompt"] for x in traffic]
    assert spread[0]["at"] == 0 and all(x["at"] < y["at"] for x, y in zip(spread, spread[1:]))
    assert "--prompts" not in A.warden_command({"requests": 7, "scan": True}, tmp_path, "u")


def test_the_gates_scan_rehearsal_at_each_tick_it_takes_runs_the_scans_schedule_there_and_its_minutes_cover_it(tmp_path):
    """`warden-rehearse --scan --bucket-ms MS` runs `rehearse` at `scan_zk.PARAMS` with the tick MS, which the scan's
    class serves, and the gate counts its minutes and its key's sampling blocks at MS: for each tick it takes, in
    100 ms steps up to its longest (a block of 128 s, past the minute `warden_minutes` counts for a shorter one), at
    its lowest height, they cover every request count's id traffic (its last arrival, its tail, the rest of the block
    it ends in, the origin's delay and 30 s for the key and the check) and every group count's `--window-per-group`
    run."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent / "live_challenger"))
    import auditor as A
    import scan_zk as sz
    lo, hi = A.SCAN_BUCKET_MS
    assert lo == sz.PARAMS.bucket_ms and hi * sz.PARAMS.window > 60_000
    most = ws.id_traffic(A.WARDEN_REQUESTS[1], model="m", rate=A.WARDEN_RATE, seed=A.WARDEN_SEED,
                         max_tokens=(1, sz.PARAMS.egress_cells - 1))
    height = A.REHEARSAL_HEIGHT[0]
    for tick in range(lo, hi + 1, 100):
        req = {"device": "warden", "requests": 7, "scan": True, "bucket_ms": tick}
        args = ws.parser().parse_args(A.warden_command(req, tmp_path, "http://127.0.0.1:40001")[2:])
        params = ws._params(args)
        assert params == dataclasses.replace(sz.PARAMS, bucket_ms=tick), tick
        sz.CLASS.serves(params)
        window_s = params.window * tick / 1000
        assert A.warden_window_ms(req) == params.window * tick
        for n in range(A.WARDEN_REQUESTS[0], A.WARDEN_REQUESTS[1] + 1):
            blocks = math.ceil((most[n - 1]["at"] + A.WARDEN_TAIL_S) / window_s)
            assert blocks * window_s + args.start_in + 30 <= A.warden_minutes(n, height, params.window * tick) * 60, \
                (tick, n)
            assert blocks <= A.warden_windows(n, bucket_ms=tick), (tick, n)
        for groups in range(1, A.WARDEN_PROMPT_REQUESTS[1] + 1):
            for tokens in (1, 4, A.WARDEN_PROMPT_TOKENS[1]):
                took = groups * ws.group_windows(params, tokens) * window_s + args.start_in + 30
                assert took <= A.warden_group_minutes(groups, tokens, height, tick) * 60, (tick, groups, tokens)


def test_rehearse_reads_prompts_the_ingress_bound_holds_and_takes_them_and_groups_only_with_what_they_need(tmp_path):
    """`read_prompts` takes a JSON array of non-empty arrays of ids below 2^17 whose id requests the schedule's ingress
    bound holds, the gate's longest (`WARDEN_PROMPT_IDS` ids of 17 bits, 62-byte names) among them at the scan's, and
    the gate's most prompt ids and tokens the scan's KI and KO;
    `rehearse` takes `--prompts` only with `--ids` and `--window-per-group` only with `--prompts`, and a group is at most
    `lanes` prompts."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent / "live_challenger"))
    import auditor as A
    import scan_zk as sz
    longest = [[(1 << A.WARDEN_PROMPT_ID_BITS) - 1] * A.WARDEN_PROMPT_IDS[1]]
    assert A.WARDEN_PROMPT_ID_BITS == ws.PROMPT_ID_BITS
    assert (A.WARDEN_PROMPT_IDS[1], A.WARDEN_PROMPT_TOKENS[1]) == (sz.CLASS.KI, sz.CLASS.KO)
    assert ws.read_prompts(json.dumps(longest).encode(), sz.PARAMS) == longest
    with pytest.raises(SystemExit, match=f"takes 48 ingress cells, past the schedule's {sz.PARAMS.ingress_cells}"):
        ws.read_prompts(json.dumps([[0] * 2098]).encode(), sz.PARAMS)
    for bad in (b"x", b"[]", b"[[]]", b"[1]", b"[[1.0]]", b"[[131072]]", b"[[-1]]", b"[[true]]", b'[["1"]]', b"{}"):
        with pytest.raises(SystemExit, match="--prompts: "):
            ws.read_prompts(bad, sz.PARAMS)
    with pytest.raises(SystemExit, match="a group of 5 prompts, past the schedule's 4 lanes"):
        ws.prompt_traffic([[1]] * 5, 5, model="m", params=sz.PARAMS, max_tokens=2, window_per_group=True)
    (tmp_path / "p.json").write_text("[[1]]")
    base = ["rehearse", "--fake", "--out", str(tmp_path / "run")]
    for argv, why in ((["--prompts", str(tmp_path / "p.json")], "--prompts: id requests' prompts, with --ids"),
                      (["--ids", "--window-per-group"], "--window-per-group: groups of the --prompts")):
        with pytest.raises(SystemExit, match=why):
            ws.rehearse(ws.parser().parse_args(base + argv))
    assert not (tmp_path / "run").exists()


def test_a_group_of_id_requests_admitted_at_a_windows_start_ends_within_its_group_windows(tmp_path):
    """The bound `group_windows` rests on, on the warden itself at the scan's schedule: a group of `lanes` id requests
    admitted together at a window's first bucket, each answered with `tokens` id events and the trailer (the loop's
    two, and the gate's most, 64), ends every connection with its last cell D + ⌊tokens/r⌋ buckets after its
    admission, so that one admitted two buckets late (`prompt_traffic`'s half a bucket, and the client's lag) still
    ends within the group's G windows; the check accepts the run."""
    sys.path.insert(0, str(HERE))
    import scan_zk as sz
    p = sz.PARAMS
    for tokens in (2, 64):
        G = ws.group_windows(p, tokens)
        requests = [ws.id_request(model="m", user=f"user-{j}", prompt=[j + 1], max_tokens=tokens,
                                  frame_bytes=p.cell_bytes, offset=0) for j in range(p.lanes)]
        out = tmp_path / f"t{tokens}"
        out.mkdir()
        responses = [ws.id_response(list(range(tokens)), p.payload)] * p.lanes
        warden = _run(out, tmp_path / f"key{tokens}", requests, responses, windows=G, params=p, id_payload=p.payload)
        assert len(warden.log) == p.lanes and all(c["failed"] is None and not c["cut"] for c in warden.log)
        assert {c["ended"] - c["admitted"] for c in warden.log} == {p.delay + tokens // p.slots}
        assert all(c["admitted"] + 2 + p.delay + tokens // p.slots < G * p.window for c in warden.log)
        result = ws.check(out)
        assert result["accepted"], result["problems"]
        assert (result["summary"]["ended"], result["summary"]["windows"]) == (p.lanes, G)
    assert (ws.group_windows(p, 2), ws.group_windows(p, 64)) == (1, 2)


def test_the_gates_demo_traffic_through_the_warden_is_the_demo_stand_ins_class_and_the_check_accepts_its_answers(
        tmp_path):
    """`warden-rehearse --scan --prompts --tokens 24` of the demo stand-in's prompts (`demo_stub.py prompts`) sends each
    prompt as it is, with exactly 24 tokens and greedy; on the warden at the scan's schedule, the stand-in, given exactly
    what the warden sent the server (its release headers on), serves every request and answers the bytes the warden
    certified, every connection ends within the two link-windows 24 tokens take at D = 27, and the check accepts the
    run."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent / "live_challenger"))
    import auditor as A
    import demo_stub as D
    import scan_zk as sz
    from verity.core.protocols.certifier.release import release_of
    prompts = D.prompts(4)
    req = {"device": "warden", "requests": None, "tokens": 24, "height": 10, "minutes": 4, "scan": True,
           "prompts": True, "temperature": 0.0, "top_p": 1.0}
    (tmp_path / A.PROMPTS).write_bytes(A.warden_prompts(req, json.dumps(prompts).encode()))
    args = ws.parser().parse_args(A.warden_command(req, tmp_path, "http://127.0.0.1:40001")[2:])
    params = ws._params(args)
    assert params == sz.PARAMS and (args.requests, args.max_tokens) == (4, 24)
    traffic = ws.traffic_of(args, D.Shape().name)
    assert [x["ids"]["prompt"] for x in traffic] == prompts
    assert {(x["ids"]["max_tokens"], x["ids"]["temperature"], x["ids"]["top_p"]) for x in traffic} == {(24, 0.0, 1.0)}
    requests = [ws.id_request(**x["ids"], frame_bytes=params.cell_bytes, offset=0) for x in traffic]
    responses, servers, out = [D.answer(p) for p in prompts], [], tmp_path / "run"
    out.mkdir()
    G = ws.group_windows(params, 24)
    assert G == 2
    warden = _run(out, tmp_path / "key", requests, responses, windows=G, params=params, id_payload=params.payload,
                  servers=servers)
    assert len(warden.log) == 4 and all(c["failed"] is None and not c["cut"] for c in warden.log)
    for j, (server, response) in enumerate(zip(servers, responses)):
        sent = b"".join(data for _, data in server.writes)
        fields = D.warden_request(sent)
        assert release_of(sent)[0] == j and fields["user"] == f"user-{j}" and D.refusal(fields) is None
        assert D.answer(fields["prompt"]) == response
    result = ws.check(out)
    assert result["accepted"], result["problems"]
    assert (result["summary"]["ended"], result["summary"]["windows"]) == (4, G)


def test_the_fake_endpoint_answers_a_request_as_vllm_streams_it():
    """Over loopback: the request's bytes reach the endpoint, which answers a chunked event stream and closes; the
    response parses to the endpoint's tokens, its finish reason and usage, and a cut response isn't complete."""
    body = ws.varied_traffic(1, model="m", seed=1)[0]["body"]

    async def main() -> bytes:
        server = await asyncio.start_server(lambda r, w: ws.fake_endpoint(r, w, sleep=_no_sleep), "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(ws.http_request(body, f"127.0.0.1:{port}"))
        data = await reader.read()
        writer.close()
        server.close()
        return data

    data = asyncio.run(main())
    got = ws.parse_response(data)
    n = ws.fake_tokens(body)
    assert got | {"prompt_tokens": None} == {"status": 200, "complete": True, "events": n + 1, "text_bytes":
                                             len("".join(f" w{k}" for k in range(n))),
                                             "finish_reason": "length" if n == body["max_tokens"] else "stop",
                                             "prompt_tokens": None, "completion_tokens": n}
    assert not ws.parse_response(data[:-5])["complete"]

    async def models() -> str:
        server = await asyncio.start_server(lambda r, w: ws.fake_endpoint(r, w, sleep=_no_sleep), "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            return await asyncio.to_thread(ws.cs.served_model, f"http://127.0.0.1:{port}")
        finally:
            server.close()

    assert asyncio.run(models()) == "fake"
    assert ws.dechunk(b"3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n") == (b"abcde", True)
    assert ws.dechunk(b"3\r\nabc\r\n2\r\nd") == (b"abcd", False)


def test_the_fake_endpoint_answers_an_id_request_id_form_as_the_scans_egress_reads_it():
    """`rehearse --ids`: an `id_traffic` request, sent warden-form (`active.id_request`), is answered with its
    `fake_ids` as id events of P bytes and the trailer, every cell an id-form `DATA` cell, then `END`'s trailer;
    `parse_id_response` reads its count, and a cut or altered response isn't complete."""
    sys.path.insert(0, str(HERE))
    import scan_zk as sz
    traffic = ws.id_traffic(20, model="m", seed=2, max_tokens=(1, sz.PARAMS.egress_cells - 1))
    assert traffic == ws.id_traffic(20, model="m", seed=2, max_tokens=(1, sz.PARAMS.egress_cells - 1))
    assert all(1 <= x["ids"]["max_tokens"] <= 64 and all(i < 1 << 17 for i in x["ids"]["prompt"]) for x in traffic)
    fields, P, B = traffic[0]["ids"], sz.PARAMS.payload, sz.PARAMS.cell_bytes
    request = ws.id_request(**fields, frame_bytes=B, offset=0)

    async def main() -> bytes:
        server = await asyncio.start_server(lambda r, w: ws.fake_endpoint(r, w, sleep=_no_sleep, id_payload=P),
                                            "127.0.0.1", 0)
        reader, writer = await asyncio.open_connection("127.0.0.1", server.sockets[0].getsockname()[1])
        writer.write(request)
        data = await reader.read()
        writer.close()
        server.close()
        return data

    data = asyncio.run(main())
    ids = ws.fake_ids(json.loads(request.partition(b"\r\n\r\n")[2]))
    assert data == ws.id_response(ids, P) and 1 <= len(ids) <= fields["max_tokens"]
    assert data[len(ids) * P:] == sz.IT.TRAILER
    assert [sz.IT.data_cell(x, 1, B)[sz.IT.HEADER.size:sz.IT.HEADER.size + P] for x in ids] == [
        data[i:i + P] for i in range(0, len(ids) * P, P)]
    got = ws.parse_id_response(data, P)
    assert (got["ids"], got["complete"], got["completion_tokens"]) == (len(ids), True, len(ids))
    assert not ws.parse_id_response(data[:-1], P)["complete"] and not ws.parse_id_response(data[:-len(sz.IT.TRAILER)],
                                                                                            P)["complete"]
    assert not ws.parse_id_response(data.replace(b"data:", b"datum", 1), P)["complete"]


class _Writer:
    def __init__(self, warden: Warden | None = None) -> None:
        self.warden, self.writes, self.closed, self.transport = warden, [], False, self

    def write(self, data: bytes) -> None:
        self.writes.append((self.warden.ticks if self.warden else None, bytes(data)))

    def is_closing(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True

    def get_write_buffer_size(self) -> int:
        return 0


def _response(texts: list[str]) -> bytes:
    events = [{"choices": [{"index": 0, "text": t, "finish_reason": "stop" if i == len(texts) - 1 else None}]}
              for i, t in enumerate(texts)]
    events.append({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": len(texts)}})
    return (b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"
            + b"".join(ws._sse(e) for e in events) + b"e\r\ndata: [DONE]\n\n\r\n0\r\n\r\n")


def slow_commit(*args, **kwargs):
    """`Cells.of`, taking `SLOW_COMMIT_S` of the clock of the process it runs in (`PROCESS_CLOCK`): pure Python
    holds its process's GIL, so a ticker in that process, on any thread, waits as long. A spawned process has none."""
    for clock in PROCESS_CLOCK:
        clock.advance(clock() + SLOW_COMMIT_S)
    return WR.Cells.of(*args, **kwargs)


def refusing_commit(*args, **kwargs):
    raise ValueError("a refusing commit")


def _run(out: Path, key: Path, requests: list[bytes], responses: list[bytes], *, windows: int = 1, commit=None,
         process: bool = False, servers: list | None = None, params: WardenParams = PARAMS,
         id_payload: int | None = None, load_heads: Path | None = None) -> Warden:
    """A run of the warden at `params` for `windows` windows on a `VirtualClock`, published as `serve` and `client` publish one:
    its records and certificates through `Publisher` inline (`serve`'s `PublisherProcess` when `process`, which first
    signs the load's heads `load_heads` when given), each window committed by `commit`, `deployment.json`, and
    `client.json` with every cell's arrival 1 ms into the bucket it was sent in, each response read id-form at P =
    `id_payload` when given; `servers` gets the server's side of each connection, in connect order. Returns the
    warden."""
    tau = params.bucket_ms / 1000
    if process:
        import warden_serve  # by name, so that the spawned process finds its functions

        ws.cs.load_key(key, 5 if load_heads else 4)
        publisher = warden_serve.PublisherProcess(out, key, SESSION, params, commit, load_heads=load_heads)
        hooks = {"publish": publisher, "withhold": publisher.withhold, "release": publisher.release}
        public = publisher.key
    else:
        signer = ws.cs.signer(key, 4)
        publisher = ws.Publisher(out, ws.Issuer(signer, SESSION), params, commit or WR.Cells.of)
        hooks, public = {"publish": publisher, "withhold": publisher.withhold}, signer.public_key()
    answers = iter(responses)

    async def connect():
        reader = asyncio.StreamReader()
        reader.feed_data(next(answers))
        reader.feed_eof()
        writer = _Writer()
        if servers is not None:
            servers.append(writer)
        return reader, writer

    def settled(w: Warden) -> bool:
        return all(c.server is not None and c.ingress_ended and c.egress_ended for c in w.active)

    async def main() -> tuple[list[_Writer], Warden]:
        clock = VirtualClock(-tau / 2)
        warden = Warden(params, connect, origin=0.0, windows=windows, clock=clock, sleep=clock.sleep, **hooks)
        running = asyncio.create_task(warden.run())
        clients, tasks = [], []
        for request in requests:
            reader = asyncio.StreamReader()
            reader.feed_data(cells(request, params.cell_bytes))
            reader.feed_eof()
            clients.append(_Writer(warden))
            tasks.append(asyncio.create_task(warden.handle(reader, clients[-1])))
        await warden.until(lambda w: w.ordinals + w.refused == len(requests))
        PROCESS_CLOCK.append(clock)
        try:
            for t in range(windows * params.window):
                await warden.until(settled)
                clock.advance(t * tau)
                await warden.until(lambda w, t=t: w.ticks > t)
            await running
        finally:
            PROCESS_CLOCK.remove(clock)
        await asyncio.gather(*tasks)
        return clients, warden

    clients, warden = asyncio.run(main())
    publisher.close()
    deployment = ws.deployment_record(params, public, SESSION, origin_unix=0.0, windows=windows,
                                      listen=("127.0.0.1", 1), upstream=("127.0.0.1", 2), slack=warden.slack,
                                      run_id=warden.run_id)
    (out / "deployment.json").write_text(json.dumps(deployment))
    log = []
    for index, (request, client) in enumerate(zip(requests, clients)):
        shim = Shim(params.cell_bytes)
        B = params.cell_bytes
        for t, data in client.writes:
            for i in range(0, len(data), B):
                kind, shim.ordinal, payload = read_cell(data[i:i + B], B)
                shim.arrivals.append((t * tau + 0.001, kind))
                shim.data += payload if kind in (DATA, END) else b""
        kinds = [k for _, k in shim.arrivals]
        shim.outcome = "refused" if REFUSED in kinds else "end" if END in kinds else "cut"
        log.append(ws.connection_log(index, 0.0, 0.0, request, shim, 0.0, id_payload))
    (out / "client.json").write_text(json.dumps({"connections": log}))
    return warden


def test_the_check_accepts_a_run_and_refuses_what_wasnt_certified_or_scheduled(tmp_path):
    out, key = tmp_path / "run", tmp_path / "key"
    out.mkdir()
    requests = [ws.http_request({"prompt": "x" * n}, "h:1") for n in (10, 200, 30)]
    responses = [_response([" a", " b"]), _response([" word"] * 40), _response([])]
    servers: list[_Writer] = []
    warden = _run(out, key, requests, responses, servers=servers)
    release = {e["ordinal"]: e["release"] for e in warden.log}
    assert sorted(release.values()) == [0, 1, 2]
    assert [b"".join(d for _, d in s.writes) for s in servers] == \
        [with_release(r, release[k + 1], warden.run_id) for k, r in enumerate(requests)]
    assert json.loads((out / "deployment.json").read_text())["run_id"] == warden.run_id.hex()
    result = ws.check(out)
    assert result["accepted"], result["problems"]
    s = result["summary"]
    assert (s["requests"], s["ended"], s["certified_windows"], s["completion_tokens"]) == (3, 3, 3, 42)
    assert s["cells_in_their_bucket"][0] == s["cells_in_their_bucket"][1] > 0
    assert s["wire_bits"]["egress"] == s["cells"]["egress"] * PARAMS.cell_bytes * 8
    assert s["data_bytes"] == {"ingress": sum(map(len, requests)), "egress": sum(map(len, responses))}
    assert "ACCEPTED: 3/3 ended" in ws.table(result)
    assert {k: s["cell_rows"][k] for k in ("registered", "of", "rows", "row_bytes")} == \
        {"registered": 2, "of": 2, "rows": [PARAMS.window], "row_bytes": [PARAMS.lanes * PARAMS.slots * PARAMS.cell_bytes]}
    assert s["cell_rows"]["commit_ms"]["n"] == s["cell_rows"]["sign_ms"]["n"] == 2

    def refused(edit) -> list[str]:
        saved = {p: p.read_bytes() for p in out.iterdir()}
        edit()
        try:
            return ws.check(out)["problems"]
        finally:
            for p, data in saved.items():
                p.write_bytes(data)

    def flip_record() -> None:
        records = ws.read_records(out, PARAMS)
        record = records["responses"][0]
        row = next(b for b, r in enumerate(record.grid) if any(f is not None and f[0] == DATA for f in r))
        frames = list(record.grid[row])
        k = next(i for i, f in enumerate(frames) if f is not None and f[0] == DATA)
        frames[k] = frames[k][:-1] + bytes([frames[k][-1] ^ 1])
        grid = record.grid[:row] + (tuple(frames),) + record.grid[row + 1:]
        (out / "responses.records").unlink()
        ws.ar.write_blobs(out / "responses.records", ws.ar.pack_record(ws.replace(record, grid=grid)))

    problems = refused(flip_record)
    assert "responses window 0 isn't certified" in problems and "responses window 0's cells aren't registered" in problems

    def edit_cells(change) -> None:
        lines = [json.loads(x) for x in (out / "cells.jsonl").read_text().splitlines()]
        salts = ws.ar.read_blobs(out / "cells.salts")
        change(lines, salts)
        (out / "cells.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        (out / "cells.salts").unlink()
        for blob in salts:
            ws.ar.write_blobs(out / "cells.salts", blob)

    def other_root(lines, salts) -> None:
        lines[1]["registration"]["root"] = "00" * 64

    def other_salt(lines, salts) -> None:
        salts[1] = bytes([salts[1][0] ^ 1]) + salts[1][1:]

    def other_wiring_salt(lines, salts) -> None:
        salts[1] = salts[1][:-1] + bytes([salts[1][-1] ^ 1])

    for change in (other_root, other_salt, other_wiring_salt):
        assert "responses window 0's cells aren't registered" in refused(lambda: edit_cells(change))

    def forge_reason() -> None:
        (record,) = ws.read_records(out, PARAMS)["fails"]
        grid = ((reason_frame(0, "late", PARAMS.cell_bytes),) + record.grid[0][1:],) + record.grid[1:]
        (out / "fails.records").unlink()
        ws.ar.write_blobs(out / "fails.records", ws.ar.pack_record(ws.replace(record, grid=grid)))

    assert "fails window 0 isn't certified" in refused(forge_reason)

    def edit_client(change) -> None:
        doc = json.loads((out / "client.json").read_text())
        change(doc["connections"][1])
        (out / "client.json").write_text(json.dumps(doc))

    def late_cell(conn: dict) -> None:
        conn["arrivals_ms"][-1] += PARAMS.bucket_ms

    problems = refused(lambda: edit_client(late_cell))
    assert any(p.startswith("connection 2:") and "'in_bucket'" in p for p in problems)
    problems = refused(lambda: edit_client(lambda c: c.update(data_sha256="0" * 64)))
    assert any(p.startswith("connection 2:") and "'response_certified': False" in p for p in problems)
    assert ws.check(out)["accepted"]


@pytest.mark.parametrize("process", [False, True])
def test_a_window_with_a_refused_connection_is_certified_nowhere(tmp_path, process):
    """Five connections at once on two lanes and a queue of four: the fifth is refused, traffic past the window's
    Program slots, so window 0's three records are kept and its two rows committed but none is certified or signed,
    each line naming the link-window under `refused`, withheld, inline and in `serve`'s publisher process; window 1
    is certified whole, and the check refuses the run on window 0 alone."""
    out = tmp_path / "run"
    out.mkdir()
    requests = [ws.http_request({"prompt": "x" * n}, "h:1") for n in (10, 20, 30, 40, 50)]
    responses = [_response([" a"]) for _ in range(4)]
    warden = _run(out, tmp_path / "key", requests, responses, windows=2, process=process)
    assert (warden.refused, warden.refused_windows, len(warden.log)) == (1, {0}, 4)
    certified = [json.loads(x) for x in (out / "certified.jsonl").read_text().splitlines()]
    cells_ = [json.loads(x) for x in (out / "cells.jsonl").read_text().splitlines()]
    assert [(x["refused"][2], x["withheld"]) for x in certified if "refused" in x] == [(0, True)] * 3
    assert [ws.cs.certified_from_json(x).window for x in certified if "refused" not in x] == [1] * 3
    assert [("refused" in x, x.get("withheld")) for x in cells_] == [(True, True)] * 2 + [(False, None)] * 2
    result = ws.check(out)
    s = result["summary"]
    assert (s["refused"], s["ended"], s["certified_windows"], s["cell_rows"]["registered"]) == (1, 4, 3, 2)
    assert {p for p in result["problems"] if "window" in p} == \
        {f"{name} window 0 isn't certified" for name in ws.CERTIFIED} | {
            f"{name} window 0's cells aren't registered" for name in ("requests", "responses")}


def test_a_windows_commit_and_signatures_move_no_tick_past_its_slack_in_the_publisher_process(tmp_path):
    """A window's commit that takes `SLOW_COMMIT_S` of its process's time (`slow_commit`), run in the warden's process
    as the inline `Publisher` runs it, puts the ticks after the window's end past their slack: the next window faults
    (`monitor-unavailable`) and the check refuses it. In `serve`'s `PublisherProcess` the same commit runs in the
    spawned process, which signs and commits from the rows released to it: every tick is on time, and the check
    accepts every link-window, certified and registered. A commit that fails there fails `close`, and the warden's
    process doesn't wait on it."""
    requests = [ws.http_request({"prompt": "x" * n}, "h:1") for n in (10, 200, 30)]
    responses = [_response([" a", " b"]), _response([" word"] * 40), _response([])]
    runs = {name: tmp_path / name for name in ("inline", "process", "refusing")}
    for out in runs.values():
        out.mkdir()

    inline = _run(runs["inline"], tmp_path / "inline-key", requests, responses, windows=2, commit=slow_commit)
    late = [t for t, x in enumerate(inline.lateness) if x > inline.slack]
    assert late[0] == PARAMS.window and max(inline.lateness) > SLOW_COMMIT_S
    problems = ws.check(runs["inline"])["problems"]
    assert {f"{name} window 1: {Reason.MONITOR_UNAVAILABLE.value}" for name in ws.CERTIFIED} <= set(problems)

    process = _run(runs["process"], tmp_path / "process-key", requests, responses, windows=2, commit=slow_commit,
                   process=True)
    assert len(process.lateness) == 2 * PARAMS.window and max(process.lateness) == 0
    result = ws.check(runs["process"])
    assert result["accepted"], result["problems"]
    s = result["summary"]
    assert (s["certified_windows"], s["cell_rows"]["registered"], s["ended"], s["signer_refused"]) == (6, 4, 3, [])

    with pytest.raises(RuntimeError, match="a refusing commit"):
        _run(runs["refusing"], tmp_path / "refusing-key", requests, responses, commit=refusing_commit, process=True)


def test_the_load_is_signed_by_the_serving_publisher_in_the_runs_session_before_its_windows(tmp_path):
    """The gate's load end to end at the tiny shape: the store relayed over four streams to the prover's sink on
    loopback (`load_relay.py sink`, a process), then a run whose `PublisherProcess`, the warden's one signer, signs
    the relay's heads first: `load.json` is under the deployment's key and session, on the key's first leaves, and
    `check` reads every value's registration at the relay's roots; another session's, values swapped, heads that
    aren't the relay's and an unsigned load are refused."""
    import hashlib
    import subprocess

    store, out = tmp_path / "store", tmp_path / "run"
    store.mkdir()
    out.mkdir()
    lr, program = ws.lr, hashlib.sha512(b"the reading program").hexdigest()
    lr.write_store(store, lr.TINY)
    sink = subprocess.Popen([sys.executable, str(HERE / "load_relay.py"), "sink", "--listen", "127.0.0.1:0",
                             "--weights", str(tmp_path / "weights.bin"), "--witness", str(tmp_path / "witness.bin")],
                            stdout=subprocess.PIPE, text=True)
    port = json.loads(sink.stdout.readline())["listening"][1]
    lr.relay_load(store, lr.TINY, [("127.0.0.1", port)] * 4, program, out)
    assert json.loads(sink.communicate()[0].splitlines()[-1])["accepted"] and sink.returncode == 0
    requests = [ws.http_request({"prompt": "x" * n}, "h:1") for n in (10, 30)]
    _run(out, tmp_path / "key", requests, [_response([" a", " b"]), _response([])], process=True,
         load_heads=out / "load-heads.json")
    result = ws.check(out)
    assert result["accepted"], result["problems"]
    n = len(lr.manifest(lr.TINY))
    assert result["summary"]["load"] == {"values": n, "registered": n, "relayed": True}
    assert ws.table(result).startswith(f"ACCEPTED: load {n}/{n} values registered; 2/2 ended")
    doc, dep = (json.loads((out / f).read_text()) for f in ("load.json", "deployment.json"))
    assert (doc["key"], doc["session"]) == (dep["key"], dep["session"]) and doc["session"] == SESSION.hex()
    leaf = lambda c: int(c["signature"][:8], 16)  # noqa: E731
    windows = [leaf(json.loads(x)["certificate"]) for x in (out / "certified.jsonl").read_text().splitlines()]
    assert sorted(leaf(r["certificate"]) for r in doc["values"]) == list(range(n)) and min(windows) >= n

    def problems(name: str, edit) -> list[str]:
        saved = (out / name).read_text()
        edit(out / name)
        try:
            return ws.check(out)["problems"]
        finally:
            (out / name).write_text(saved)

    def rewrite(change):
        return lambda p: p.write_text(json.dumps(change(json.loads(p.read_text()))))

    assert "load: not signed under the deployment's key and session" in \
        problems("load.json", rewrite(lambda d: d | {"session": "00" * 32}))
    swapped = problems("load.json", rewrite(lambda d: d | {"values": [d["values"][1], d["values"][0], *d["values"][2:]]}))
    assert sum(p.startswith("load tok: ") or p.startswith("load pos: ") for p in swapped) == 2
    assert "load: its registrations aren't the relay's heads (load-heads.json)" in \
        problems("load-heads.json", rewrite(lambda d: d | {"roots": {**d["roots"], "tok": "00" * 64}}))
    assert problems("load.json", lambda p: p.unlink()) == ["load: the relay's heads aren't signed (no load.json)"]


def test_reserve_fds_grows_the_descriptor_table_and_leaves_every_descriptor_as_it_was():
    """`serve`'s process holds a table for every connection before its origin (`prepare_process`), so that no tick
    waits on the kernel growing it (`reserve_fds`); the table's size is `FDSize` in /proc/self/status."""
    def table() -> int:
        status = Path("/proc/self/status").read_text().splitlines()
        return int(next(x.split()[1] for x in status if x.startswith("FDSize:")))

    held = sorted(os.listdir("/proc/self/fd"))
    n = ws.reserve_fds(table() + 1)
    assert table() >= n and sorted(os.listdir("/proc/self/fd")) == held
