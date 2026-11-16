"""The gate's network attester commands (`network_attester_command`, `auditor.py`) against the runner they start, the device's
`network_attester_serve.py`: each request parses to the run whose minutes, execution phases and key height the gate counts, at
`network_attester_serve.py`'s default schedule and at the scan's; the forms the gate publishes are the verifier's constants and
hold a run's own lines, which the released keys' lists declare; and its relay session is the `session` the device's own tests run."""

from __future__ import annotations

import dataclasses
import json
import math
import re
import socket
import sys
from pathlib import Path

import pytest

from verity.protocols.resource_accounting.network.prover import released_keys as RK
from verity.protocols.resource_accounting.network.attester import network_attester_serve as ws
from verity.protocols.resource_accounting.network.attester.tests.relay_runs import _response, _run, _session_argv
from verity.protocols.resource_accounting.network.prover import scan_zk as sz
from verity.protocols.resource_accounting.network.verifier import load_check as LK, relay_check as RC, network_attester_rows as WR

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import auditor as A  # noqa: E402



def test_the_gates_network_attester_rehearsal_is_a_rehearse_and_its_least_minutes_cover_every_request_counts_traffic(tmp_path):
    """The auditor device's `warden-rehearse` (`verity/protocols/compose/verifier/live_challenger/auditor.py`) runs `rehearse` on traffic the
    gate fixes, its key at the request's height (the deployed key's, 20, by default), and refuses a request whose
    minutes are under `network_attester_minutes` of its requests and height: for every request count the gate takes, at its
    lowest height, those minutes cover the traffic's last arrival, its tail, the rest of the window it ends in, the
    origin's delay and 30 s for the key and the check."""
    argv = A.network_attester_command({"device": "warden", "requests": 7, "tokens": 64}, tmp_path, "http://127.0.0.1:40001")
    args = ws.parser().parse_args(argv[2:])
    assert (args.command, args.upstream, args.fake, args.out, args.requests, args.max_tokens, args.height) == \
        ("rehearse", ("127.0.0.1", 40001), False, tmp_path, 7, 64, 20)
    assert ws.parser().parse_args(A.network_attester_command({"requests": 7, "height": 10}, tmp_path, "http://h:1")[2:]).height \
        == 10
    assert (args.rate, args.seed, args.tail) == (A.NETWORK_CERTIFIER_RATE, A.NETWORK_CERTIFIER_SEED, A.NETWORK_CERTIFIER_TAIL_S)
    assert ws.parser().parse_args(A.network_attester_command({"requests": 7}, tmp_path, "http://h:1")[2:]).max_tokens == 256
    window_s = args.window * args.bucket_ms / 1000
    most = ws.varied_traffic(A.NETWORK_CERTIFIER_REQUESTS[1], model="m", rate=A.NETWORK_CERTIFIER_RATE, seed=A.NETWORK_CERTIFIER_SEED)
    assert ws.varied_traffic(400, model="m", rate=A.NETWORK_CERTIFIER_RATE, seed=A.NETWORK_CERTIFIER_SEED) == most[:400]
    for n in range(A.NETWORK_CERTIFIER_REQUESTS[0], A.NETWORK_CERTIFIER_REQUESTS[1] + 1):
        took = math.ceil((most[n - 1]["at"] + A.NETWORK_CERTIFIER_TAIL_S) / window_s) * window_s + args.start_in + 30
        assert took <= A.network_attester_minutes(n, A.REHEARSAL_HEIGHT[0]) * 60, n


def test_the_gates_load_relay_is_a_rehearse_with_its_load_first_and_its_minutes_cover_the_link(tmp_path):
    """The auditor device's `load-relay` runs `rehearse` as `warden-rehearse` does, with the load first: the store's
    stream of the request's shape relayed to the sink tunnels' local ends for its program, the endpoint given the
    gate's `LOAD_READY_S`; a request without a load passes none. The gate's shapes are `load_relay`'s, and its
    `load_minutes` cover each shape's weights stream and witness at `LOAD_RATE` and the endpoint's wait: 46 at H200."""
    program, sinks = "ab" * 64, [("127.0.0.1", 50001 + i) for i in range(4)]
    req = {"device": "warden", "requests": 7, "scan": True,
           "load": {"shape": "tiny", "program": program, "sink_port": 9001, "streams": 4, "minutes": 16}}
    argv = A.network_attester_command(req, tmp_path, "http://127.0.0.1:40001", sinks=sinks, store=tmp_path / "store")
    args = ws.parser().parse_args(argv[2:])
    assert (args.load_store, args.load_shape, args.load_sinks, args.load_program, args.load_ready_s) == \
        (tmp_path / "store", "tiny", sinks, program, A.LOAD_READY_S)
    without = A.network_attester_command({k: v for k, v in req.items() if k != "load"}, tmp_path, "http://127.0.0.1:40001")
    assert argv[:len(without)] == without and ws.parser().parse_args(without[2:]).load_store is None
    assert args.ids and ws._params(args) == ws._params(ws.parser().parse_args(without[2:]))
    assert set(A.LOAD_SHAPES) == set(LK.SHAPES)
    for shape in A.LOAD_SHAPES:
        values = LK.manifest(LK.SHAPES[shape])
        link = values[-1].offset + values[-1].nbytes + ws.lr.SALT_BYTES * sum(v.leaves for v in values)
        assert A.load_minutes(shape) * 60 >= link / A.LOAD_RATE + A.LOAD_READY_S > (A.load_minutes(shape) - 1) * 60
    assert (A.load_minutes("h200"), A.load_minutes("tiny")) == (46, 16)


def test_the_gates_sampling_is_on_every_request_and_moves_nothing_else_of_the_traffic(tmp_path):
    """`warden-rehearse` always passes its temperature and top_p (0 and 1 by default), and `rehearse` puts both on
    every request, so an endpoint's own generation config never applies; the arrivals, prompts, lengths and seeds stay
    the seed's, so `network_attester_minutes` still bounds the run."""
    given = ws.parser().parse_args(
        A.network_attester_command({"requests": 7, "temperature": 0.6, "top_p": 0.9}, tmp_path, "http://h:1")[2:])
    plain = ws.parser().parse_args(A.network_attester_command({"requests": 7}, tmp_path, "http://h:1")[2:])
    assert (plain.temperature, plain.top_p, given.temperature, given.top_p) == (0.0, 1.0, 0.6, 0.9)
    varied = ws.varied_traffic(7, model="m", rate=A.NETWORK_CERTIFIER_RATE, seed=A.NETWORK_CERTIFIER_SEED)

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
    boundary, 8 an execution phase and D 2.7 s, the gate sessions' schedule. Its id traffic's prompts and outputs are
    within the scan's KI and KO, on the gate's arrivals at the gate's sampling; for every request count the gate takes,
    at its lowest height, `network_attester_minutes` covers the id traffic's last arrival, its tail, the rest of the window it
    ends in, the origin's delay and 30 s for the key and the check."""
    req = {"device": "warden", "requests": 7, "scan": True, "temperature": 0.6, "top_p": 0.9}
    args = ws.parser().parse_args(A.network_attester_command(req, tmp_path, "http://127.0.0.1:40001")[2:])
    p = ws._params(args)
    assert args.ids and not args.fake and p == sz.PARAMS
    sz.CLASS.serves(p)
    assert (p.every, p.program_slots, p.window_programs, p.delay) == (16, 4, 8, 27)
    assert args.prompt_max <= sz.CLASS.KI and min(args.max_tokens, p.egress_cells - 1) == sz.CLASS.KO
    assert (args.rate, args.seed, args.tail) == (A.NETWORK_CERTIFIER_RATE, A.NETWORK_CERTIFIER_SEED, A.NETWORK_CERTIFIER_TAIL_S)
    assert sz.main(["schedule"]) == 0 and capsys.readouterr().out.split() == list(A.SCAN_SCHEDULE)
    traffic = ws.traffic_of(args, "m")
    assert {(x["ids"]["temperature"], x["ids"]["top_p"]) for x in traffic} == {(0.6, 0.9)}
    sampled = ws.id_traffic(7, model="m", rate=A.NETWORK_CERTIFIER_RATE, seed=A.NETWORK_CERTIFIER_SEED, max_tokens=(1, 64))

    def rest(t):
        return [{**x, "ids": {k: v for k, v in x["ids"].items() if k not in ("temperature", "top_p")}} for x in t]
    assert rest(traffic) == rest(sampled) and {x["ids"]["top_p"] for x in sampled} == {1.0}
    plain = ws.parser().parse_args(A.network_attester_command({"requests": 7, "scan": True}, tmp_path, "http://h:1")[2:])
    assert {(x["ids"]["temperature"], x["ids"]["top_p"]) for x in ws.traffic_of(plain, "m")} == {(0.0, 1.0)}

    window_s = sz.PARAMS.window * sz.PARAMS.bucket_ms / 1000
    most = ws.id_traffic(A.NETWORK_CERTIFIER_REQUESTS[1], model="m", rate=A.NETWORK_CERTIFIER_RATE, seed=A.NETWORK_CERTIFIER_SEED,
                         max_tokens=(1, sz.PARAMS.egress_cells - 1))
    for n in range(A.NETWORK_CERTIFIER_REQUESTS[0], A.NETWORK_CERTIFIER_REQUESTS[1] + 1):
        took = math.ceil((most[n - 1]["at"] + A.NETWORK_CERTIFIER_TAIL_S) / window_s) * window_s + args.start_in + 30
        assert took <= A.network_attester_minutes(n, A.REHEARSAL_HEIGHT[0]) * 60, n


def test_each_scan_class_the_gate_takes_is_scan_zks_setup_of_its_name_at_its_schedule_ki_and_ko(tmp_path, capsys):
    """Each class `warden-rehearse --scan --prompts --scan-class` takes (`auditor.SCAN_CLASSES`, the class of record
    `scan_zk.W63`'s by default) is `scan_zk.SETUPS`' of its name: the gate's network attester runs at its schedule (`scan_zk.py
    schedule --class`, the one `scan_zk.py prove --class --warden-run` refuses any other of), which its class serves
    and which is the class of record's but for its egress cells, so the gate's group windows hold at it; and the gate
    bounds a prompt by its KI and the tokens by its KO. w63k4's class refuses the class of record's schedule (65
    egress cells past its KO + 1 = 5), so a w63k4 run needs its own."""
    assert A.SCAN_CLASS == sz.W63.name and set(A.SCAN_CLASSES) == {"w63", "w63k4"} <= set(sz.SETUPS)
    (tmp_path / A.PROMPTS).write_text("[[1]]")
    for name in A.SCAN_CLASSES:
        s = sz.SETUPS[name]
        assert sz.main(["schedule", "--class", name]) == 0
        assert capsys.readouterr().out.split() == list(A.scan_schedule(scan_class=name)), name
        req = {"device": "warden", "requests": 4, "scan": True, "prompts": {"count": 1, "ids": 1}, "scan_class": name}
        args = ws.parser().parse_args(A.network_attester_command(req, tmp_path, "http://127.0.0.1:40001")[2:])
        params = ws._params(args)
        assert params == s.params == dataclasses.replace(sz.PARAMS, egress_cells=s.params.egress_cells), name
        s.cls.serves(params)
        assert A.scan_bounds(name) == ((1, s.cls.KI), (1, s.cls.KO)), name
        assert [ws.group_windows(params, t) for t in range(1, s.cls.KO + 1)] == \
            [A.scan_group_windows(t) for t in range(1, s.cls.KO + 1)], name
    with pytest.raises(ValueError, match="egress_cells"):
        sz.W63K4.cls.serves(sz.PARAMS)


def test_the_gates_network_attester_windows_are_at_least_every_request_counts_phases_which_its_key_signs(tmp_path):
    """The gate sizes a network attester rehearsal's key on `network_attester_windows` of its requests (`network_attester_height`), so for
    every request count it takes, those are at least the execution phases `rehearse` runs (its traffic's last arrival
    and tail): at the scan's schedule on the gate's id traffic and on `--prompts`' arrivals, and at `network_attester_serve.py`'s
    default (`NETWORK_CERTIFIER_WINDOW_MS`) on its varied traffic; a request count's traffic is the first of the most's. The gate's
    most requests are 1064 execution phases of `--prompts`, about 57 minutes, which height 13 signs and 12 doesn't."""
    plain = ws.parser().parse_args(A.network_attester_command({"requests": 7}, tmp_path, "http://h:1")[2:])
    assert A.NETWORK_CERTIFIER_WINDOW_MS == plain.window * plain.bucket_ms
    assert A.SCAN["--window"] * A.SCAN["--bucket-ms"] == sz.PARAMS.window * sz.PARAMS.bucket_ms
    most, at = A.NETWORK_CERTIFIER_REQUESTS[1], dict(rate=A.NETWORK_CERTIFIER_RATE, seed=A.NETWORK_CERTIFIER_SEED)
    scan = {"ids": ws.id_traffic(most, model="m", max_tokens=(1, sz.PARAMS.egress_cells - 1), **at),
            "prompts": ws.prompt_traffic([[1]], most, model="m", params=sz.PARAMS, max_tokens=24, **at)}
    varied = ws.varied_traffic(most, model="m", **at)
    assert ws.prompt_traffic([[1]], 400, model="m", params=sz.PARAMS, max_tokens=24, **at) == scan["prompts"][:400]

    def windows(traffic, n, window_ms):
        return math.ceil((traffic[n - 1]["at"] + A.NETWORK_CERTIFIER_TAIL_S) * 1000 / window_ms)
    scan_ms = sz.PARAMS.window * sz.PARAMS.bucket_ms
    for n in range(A.NETWORK_CERTIFIER_REQUESTS[0], most + 1):
        assert all(windows(t, n, scan_ms) <= A.network_attester_windows(n) for t in scan.values()), n
        assert windows(varied, n, A.NETWORK_CERTIFIER_WINDOW_MS) <= A.network_attester_windows(n, scan=False), n
    assert (windows(scan["prompts"], 400, scan_ms), windows(scan["prompts"], most, scan_ms)) == (259, 1064)
    assert A.network_attester_height(windows(scan["prompts"], most, scan_ms)) == A.network_attester_height(A.network_attester_windows(most)) == 13


def test_the_gates_prompts_rehearsal_sends_its_prompts_a_group_a_span_of_windows_and_its_minutes_cover_them(tmp_path):
    """`warden-rehearse --scan --prompts --window-per-group` runs `rehearse --ids --prompts FILE --window-per-group` at
    the scan's schedule, on the gate's witness file: request j is the (j mod n)-th prompt with exactly the request's
    tokens, group g's n requests together half a bucket into execution phase g·G, G the gate's `scan_group_windows`
    (`group_windows` at the scan's schedule, for every token count the gate takes), and the run exactly its groups'
    windows (`groups`), which `network_attester_group_minutes` covers with the origin's delay and 30 s for the key and the
    check. Without `--window-per-group` the prompts go at the gate's arrivals."""
    prompts = [[70001, 3], [5], [7]]
    (tmp_path / A.PROMPTS).write_text(json.dumps(prompts))
    req = {"device": "warden", "requests": 7, "scan": True, "prompts": {"count": 3, "ids": 4}, "window_per_group": True,
           "tokens": 2, "height": 10, "temperature": 0.0, "top_p": 1.0}
    args = ws.parser().parse_args(A.network_attester_command(req, tmp_path, "http://127.0.0.1:40001")[2:])
    assert (args.ids, args.window_per_group, args.prompts, args.max_tokens, args.requests) == \
        (True, True, tmp_path / A.PROMPTS, 2, 7)
    params = ws._params(args)
    assert params == sz.PARAMS
    assert [ws.group_windows(params, t) for t in range(1, A.NETWORK_CERTIFIER_PROMPT_TOKENS[1] + 1)] == \
        [A.scan_group_windows(t) for t in range(1, A.NETWORK_CERTIFIER_PROMPT_TOKENS[1] + 1)]
    G, window_s = A.scan_group_windows(2), params.window * params.bucket_ms / 1000
    traffic = ws.traffic_of(args, "torch-loop/mlp-w8a8-h128")
    assert [x["ids"]["prompt"] for x in traffic] == [prompts[j % 3] for j in range(7)]
    assert {(x["ids"]["max_tokens"], x["ids"]["temperature"], x["ids"]["top_p"]) for x in traffic} == {(2, 0.0, 1.0)}
    assert [x["at"] for x in traffic] == [round((j // 3) * G * window_s + params.bucket_ms / 2000, 3) for j in range(7)]
    assert [x["ids"]["user"] for x in traffic] == [f"user-{j}" for j in range(7)]
    assert ws.groups(7, 3, params, 2) == [{"group": g, "requests": list(range(3 * g, min(7, 3 * g + 3))),
                                           "windows": [g * G, (g + 1) * G]} for g in range(3)]
    for groups in range(1, A.NETWORK_CERTIFIER_PROMPT_REQUESTS[1] + 1):
        for tokens in (1, 2, A.NETWORK_CERTIFIER_PROMPT_TOKENS[1]):
            took = groups * ws.group_windows(params, tokens) * window_s + args.start_in + 30
            assert took <= A.network_attester_group_minutes(groups, tokens, A.REHEARSAL_HEIGHT[0]) * 60, (groups, tokens)

    loose = ws.parser().parse_args(A.network_attester_command(req | {"window_per_group": False}, tmp_path, "u:1")[2:])
    assert not loose.window_per_group and loose.prompts == tmp_path / A.PROMPTS
    spread = ws.traffic_of(loose, "m")
    assert [x["ids"]["prompt"] for x in spread] == [x["ids"]["prompt"] for x in traffic]
    assert spread[0]["at"] == 0 and all(x["at"] < y["at"] for x, y in zip(spread, spread[1:]))
    assert "--prompts" not in A.network_attester_command({"requests": 7, "scan": True}, tmp_path, "u")


def test_the_gates_scan_rehearsal_at_each_tick_it_takes_runs_the_scans_schedule_there_and_its_minutes_cover_it(tmp_path):
    """`warden-rehearse --scan --bucket-ms MS` runs `rehearse` at `scan_zk.PARAMS` with the tick MS, which the scan's
    class serves, and the gate counts its minutes and its key's execution phases at MS: for each tick it takes, in
    100 ms steps up to its longest (a block of 128 s, past the minute `network_attester_minutes` counts for a shorter one), at
    its lowest height, they cover every request count's id traffic (its last arrival, its tail, the rest of the block
    it ends in, the origin's delay and 30 s for the key and the check) and every group count's `--window-per-group`
    run."""
    lo, hi = A.SCAN_BUCKET_MS
    assert lo == sz.PARAMS.bucket_ms and hi * sz.PARAMS.window > 60_000
    most = ws.id_traffic(A.NETWORK_CERTIFIER_REQUESTS[1], model="m", rate=A.NETWORK_CERTIFIER_RATE, seed=A.NETWORK_CERTIFIER_SEED,
                         max_tokens=(1, sz.PARAMS.egress_cells - 1))
    height = A.REHEARSAL_HEIGHT[0]
    for tick in range(lo, hi + 1, 100):
        req = {"device": "warden", "requests": 7, "scan": True, "bucket_ms": tick}
        args = ws.parser().parse_args(A.network_attester_command(req, tmp_path, "http://127.0.0.1:40001")[2:])
        params = ws._params(args)
        assert params == dataclasses.replace(sz.PARAMS, bucket_ms=tick), tick
        sz.CLASS.serves(params)
        window_s = params.window * tick / 1000
        assert A.network_attester_window_ms(req) == params.window * tick
        for n in range(A.NETWORK_CERTIFIER_REQUESTS[0], A.NETWORK_CERTIFIER_REQUESTS[1] + 1):
            blocks = math.ceil((most[n - 1]["at"] + A.NETWORK_CERTIFIER_TAIL_S) / window_s)
            assert blocks * window_s + args.start_in + 30 <= A.network_attester_minutes(n, height, params.window * tick) * 60, \
                (tick, n)
            assert blocks <= A.network_attester_windows(n, bucket_ms=tick), (tick, n)
        for groups in range(1, A.NETWORK_CERTIFIER_PROMPT_REQUESTS[1] + 1):
            for tokens in (1, 4, A.NETWORK_CERTIFIER_PROMPT_TOKENS[1]):
                took = groups * ws.group_windows(params, tokens) * window_s + args.start_in + 30
                assert took <= A.network_attester_group_minutes(groups, tokens, height, tick) * 60, (tick, groups, tokens)


def test_the_gates_relay_session_is_session_at_the_scans_schedule_and_its_minutes_cover_its_length(tmp_path):
    """`relay-session` runs `session` at the listener the gate gives it, for the request's execution phases, at the
    schedule a scan rehearsal of its tick and class runs at, in the id egress form `serve` releases, and takes none
    of `rehearse`'s traffic. For every tick the gate takes, in 100 ms steps, and every count of execution phases
    that fits its minutes there, at the least height that signs them, `session_minutes` covers their span, the
    origin's delay and 30 s for the key and the check."""
    req = {"device": "warden", "client_port": 8012, "windows": 20, "height": 10, "scan": True}
    argv = A.network_attester_command(req, tmp_path, "http://127.0.0.1:40001", listen=40002)
    args = ws.parser().parse_args(argv[2:])
    assert (args.command, args.upstream, args.listen, args.windows, args.height, args.out) == \
        ("session", ("127.0.0.1", 40001), ("127.0.0.1", 40002), 20, 10, tmp_path)
    assert args.ids and not args.fake and not args.watermark and args.egress == "id"
    assert ws._params(args) == sz.PARAMS and sz.PARAMS.egress == "id"
    for name in A.SCAN_CLASSES:
        given = ws.parser().parse_args(A.network_attester_command(req | {"scan_class": name, "watermark": True}, tmp_path,
                                                        "http://127.0.0.1:40001", listen=40002)[2:])
        assert ws._params(given) == sz.SETUPS[name].params and given.watermark, name
        rehearsal = {"device": "warden", "requests": 4, "scan": True, "scan_class": name, "prompts": {"count": 1}}
        (tmp_path / A.PROMPTS).write_text("[[1]]")
        assert ws._params(ws.parser().parse_args(A.network_attester_command(rehearsal, tmp_path, "http://h:1")[2:])) == \
            ws._params(given), name
    for flag in (["--requests", "4"], ["--rate", "1"], ["--seed", "1"], ["--prompts", "p.json"],
                 ["--window-per-group"], ["--tail", "1"], ["--ids"], ["--model", "m"], ["--lengths", "1"]):
        with pytest.raises(SystemExit):
            ws.parser().parse_args(argv[2:] + flag)

    lo, hi = A.SCAN_BUCKET_MS
    for tick in range(lo, hi + 1, 100):
        at = ws.parser().parse_args(A.network_attester_command(req | {"bucket_ms": tick}, tmp_path, "http://h:1", listen=1)[2:])
        params = ws._params(at)
        assert params == dataclasses.replace(sz.PARAMS, bucket_ms=tick), tick
        window_ms = params.window * tick
        for windows in range(A.SESSION_WINDOWS[0], A.SESSION_WINDOWS[1] + 1):
            minutes = A.session_minutes(windows, A.network_attester_height(windows), window_ms)
            if minutes > A.CERTIFIER_MINUTES[1]:
                break
            assert windows * window_ms / 1000 + at.start_in + 30 <= minutes * 60, (tick, windows)


def test_rehearse_reads_prompts_the_ingress_bound_holds_and_takes_them_and_groups_only_with_what_they_need(tmp_path):
    """`read_prompts` takes a JSON array of non-empty arrays of ids below 2^17 whose id requests the schedule's ingress
    bound holds, the gate's longest (`NETWORK_CERTIFIER_PROMPT_IDS` ids of 17 bits, 62-byte names) among them at the scan's, and
    the gate's most prompt ids and tokens the scan's KI and KO;
    `rehearse` takes `--prompts` only with `--ids` and `--window-per-group` only with `--prompts`, and a group is at most
    `lanes` prompts."""
    longest = [[(1 << A.NETWORK_CERTIFIER_PROMPT_ID_BITS) - 1] * A.NETWORK_CERTIFIER_PROMPT_IDS[1]]
    assert A.NETWORK_CERTIFIER_PROMPT_ID_BITS == ws.PROMPT_ID_BITS
    assert (A.NETWORK_CERTIFIER_PROMPT_IDS[1], A.NETWORK_CERTIFIER_PROMPT_TOKENS[1]) == (sz.CLASS.KI, sz.CLASS.KO)
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


def test_of_a_deployment_the_gate_gives_out_every_key_but_the_auditors_devices(tmp_path):
    """`auditor.DEPLOYMENT_PUBLIC` is the deployment's record less the origin on the device's raw clock
    (`relay_check.DEVICE_KEYS`); where the relay listened and where it reached the endpoint are `endpoints.json`'s,
    which the gate never publishes: a key the record gains is given out only once the gate names it."""
    out = tmp_path / "run"
    out.mkdir()
    _run(out, tmp_path / "key", [ws.http_request({"prompt": "x"}, "h:1")], [_response([" a"])])
    record = set(json.loads((out / "deployment.json").read_text()))
    assert record - set(A.DEPLOYMENT_PUBLIC) == set(RC.DEVICE_KEYS) == {"origin_raw"}
    assert set(A.DEPLOYMENT_PUBLIC) <= record and len(set(A.DEPLOYMENT_PUBLIC)) == len(A.DEPLOYMENT_PUBLIC)
    assert set(A.public_deployment(out / "deployment.json")) == set(A.DEPLOYMENT_PUBLIC)
    assert A.public_deployment(out / "none.json") is None


def test_the_gates_forms_are_the_verifiers_constants_and_hold_a_runs_own_lines(tmp_path):
    """The gate publishes a certificate and a cells' registration only in forms it states itself, since it loads
    nothing of the certifier's verifier: its constants are `network_attester_rows`' and the runner's word for word, and
    every line a run writes passes them under the run's own deployment, so an honest run publishes all its
    certificates and registrations. A value changed in a line is out of form, and nothing of its file is
    published."""
    assert (A.ROW_FORMAT, A.ROW_SCHEMA, A.ROW_WORD_BITS) == (WR.REGISTRATION, WR.SCHEMA, WR.WORD_BITS)
    assert {ws.LINK + follows for follows in A.CERTIFIED_LINKS} == {link for link, _ in ws.CERTIFIED.values()}
    assert set(A.DIRECTIONS) == {d.value for _, d in ws.CERTIFIED.values()}
    run, out = tmp_path / "run", tmp_path / "out"
    run.mkdir()
    out.mkdir()
    _run(run, tmp_path / "key", [ws.http_request({"prompt": "x" * n}, "h:1") for n in (10, 200)],
         [_response([" a", " b"]), _response([" word"] * 40)], windows=2)
    given = A.deployment_forms(run / "deployment.json")
    dep = json.loads((run / "deployment.json").read_text())
    assert given["link"] == (ws.LINK,) and given["parameters"] == (dep["digest"],)
    assert not list(out.iterdir()) and A.network_attester_outcome(out, run)["verdict"] is None
    assert {p.name for p in out.iterdir()} == {"deployment.json"}, "a run with no accepted check publishes no line"
    check = ws.check(run)
    (run / "check.json").write_text(json.dumps(check))
    outcome = A.network_attester_outcome(out, run)
    assert check["accepted"] is True and outcome["verdict"]["accepted"] is True and "unruled" not in outcome
    for name, public in (("certified.jsonl", "certificates.jsonl"), ("cells.jsonl", "cells-certificates.jsonl"),
                         ("cells.jsonl", "cells-registrations.jsonl")):
        lines = (run / name).read_text().splitlines()
        assert lines and len((out / public).read_text().splitlines()) == len(lines), public
    line = json.loads((run / "cells.jsonl").read_text().splitlines()[0])
    assert A.row_registration(json.dumps(line), given)
    for key, value in (("value", "another/egress/0"), ("parameters", "00" * 64), ("row_schema", "cells/v1"),
                       ("root", line["registration"]["root"][:-2]), ("rows", str(line["registration"]["rows"]))):
        with pytest.raises(A.Unruled, match=f"registration's {key} isn't in the form ROW_LINE"):
            A.row_registration(json.dumps(line | {"registration": line["registration"] | {key: value}}), given)


def test_the_gates_relay_session_is_the_session_the_devices_tests_run(tmp_path):
    """`relay-session`'s command, with its listener's descriptor and without, is the device's tests' `session`
    (`_session_argv`), and `held_listener` a socket listening on this host's loopback."""
    req = {"device": "warden", "client_port": 8012, "windows": 3, "height": 5, "scan": True, "scan_class": "w63k4"}
    for fd in (None, 9):
        gate = A.network_attester_command(req, tmp_path / "run", "http://127.0.0.1:40001", listen=40002, listen_fd=fd)
        assert ws.parser().parse_args(gate[2:]) == \
            ws.parser().parse_args(_session_argv(tmp_path / "run", 3, listen_fd=fd)), fd
    held = A.held_listener()
    assert held.getsockname()[0] == "127.0.0.1"
    with socket.create_connection(held.getsockname()):
        conn, _ = held.accept()
        conn.close()
    held.close()


def test_the_gates_lists_are_what_the_gate_publishes_of_a_line():
    """The list of a gate run's cells' registrations names exactly the keys the gate publishes of a line (those
    `ROW_LINE` gives a form), so a key the gate starts publishing fails a record until the list gives it a
    kind, and the list holds none the gate leaves out. The certificates' list names exactly a certificate's fields
    as the gate writes them (`CERTIFICATE_LINE`, `row_certificate`). Neither names the execution phase a line was
    refused for: which ones the signer refused is the traffic's, so the gate writes no line that says it, and a
    line that does is not declared."""

    def public(rule, at=()):
        if isinstance(rule, dict):
            return {p for k, v in rule.items() for p in public(v, (*at, k))}
        return set() if rule == A.WITHHELD else {"/".join(at)}

    def declared(listing):
        doc = next(d for d in json.loads(RK.DECLARED.read_text())["lists"] if d["name"] == listing.name)
        return {k.removesuffix("/*") for k in doc["keys"]}

    lists = RK.load()
    listing = lists.listing(f"gates/c0/r0/out/{A.ROW_REGISTRATIONS['cells.jsonl']}")
    assert declared(listing) == public(A.ROW_LINE)
    signed = re.compile("[0-9a-f]{4}")
    given = {"link": ("l",), "links": ("l",), "parameters": ("31" * 64,), "cell_bytes": (64,), "buckets": (1,),
             "signature": signed, "refused": [("l",), A.DIRECTIONS, int]}
    line = {"registration": {"format": A.ROW_FORMAT, "wiring": "w", "link": "l", "direction": "ingress", "window": 2,
                             "layout": {"buckets": 1},
                             "certificate": {"key": "0c" * 16, "digest": "9d" * 64, "signature": "e5e5"}},
            "cost": {"s": 1}}
    refused = {"refused": ["l", "ingress", 3], "withheld": True}
    assert listing.undeclared(A.row_registration(json.dumps(line), given).encode()) == {}
    assert set(listing.undeclared(json.dumps(line).encode())) == {"registration/wiring", "cost"}
    assert listing.undeclared(json.dumps({"refused": refused["refused"]}).encode()) == {"refused": 1}
    for write in (lambda: A.row_registration(json.dumps(refused), given), lambda: A.row_certificate(refused, given),
                  lambda: A.certified_certificate(refused, given)):
        with pytest.raises(A.Unruled, match="the line says the signer refused its execution phase"):
            write()
    certificates = {lists.listing(f"gates/c0/r0/out/{f}").name
                    for f in ("certificates.jsonl", *A.ROW_CERTIFICATES.values())}
    assert len(certificates) == 1
    listing = lists.listing("gates/c0/r0/out/certificates.jsonl")
    written = A.row_certificate(line, given)
    assert declared(listing) == {f"certificate/{k}" for k in written["certificate"]} == public(A.CERTIFICATE_LINE)
    for doc in (written, A.row_certificate({A.ROWS_NO_HEAD: line["registration"]}, given)):
        assert listing.undeclared(json.dumps(doc).encode()) == {}
    assert listing.undeclared(json.dumps({"refused": refused["refused"]}).encode()) == {"refused": 1}
    assert set(listing.undeclared(json.dumps(line).encode())) == {"registration", "cost"}
