"""warden-serve without the wall clock: its varied traffic, its HTTP bytes against the fake endpoint over loopback, and
the offline check of a run whose network certifier is driven tick by tick on a `VirtualClock`, its records certified by
an LMS key: the check accepts the run, whose server got each request with its release order and bucket under the run
id `deployment.json` names, and refuses a record that isn't what was certified, a cell that reached its client in
another bucket, and a response that isn't the record's; a window in which the network certifier refused a connection is
certified and signed nowhere; and a window's slow commit, which faults the next window when it runs in the network
certifier's process, moves no tick in `serve`'s publisher process."""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import importlib.util
import io
import json
import math
import os
import socket
import sys
import types
from pathlib import Path

import pytest

from verity.protocols.accounting.certifier.verifier import warden_rows as WR
from verity.protocols.accounting.certifier.verifier.active import DATA, VirtualClock
from verity.protocols.accounting.certifier.verifier.grid import Reason
from verity.protocols.accounting.certifier.verifier.release import WATERMARK_END, watermark_chunk, watermark_head, with_release
from verity.protocols.accounting.certifier.verifier.warden import (END, REFUSED, Shim, Warden, WardenParams, cells, read_cell,
                                                    reason_frame)
from verity.protocols.accounting.certifier.verifier import load_check as LK, relay_check as RC, record_files as RF

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
#: a text rehearsal's schedule: its responses are vLLM's stock framing, so its egress form is bytes
PARAMS = WardenParams(cell_bytes=64, slots=2, bucket_ms=100, delay=3, lanes=2, queue=4, window=60, ingress_cells=40,
                      ingress_buckets=30, egress_cells=128, egress="bytes")
SESSION = bytes(range(32))
#: the clocks `slow_commit` moves: a run's while it ticks in this process; a spawned process's list stays empty
PROCESS_CLOCK: list[VirtualClock] = []
#: what a window's commit takes of its process's time: the trace corpus's small-KV run measured 378-392 ms
SLOW_COMMIT_S = 0.4


def _load():
    sys.path.insert(0, str(HERE))
    sys.path.append(str(ROOT))  # for benchmarks.network_traces
    spec = importlib.util.spec_from_file_location("network_traces_warden_serve",
                                                  ROOT / "benchmarks/network_traces/warden_serve.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ws = _load()
import runner_parts as rp  # noqa: E402
from verity.protocols.accounting.certifier.verifier import warden as warden_, link_frame  # noqa: E402


async def _no_sleep(_: float) -> None:
    pass


#: what the relay's runner still imports from the older form's files, by the file: nothing. The keys stay so that a
#: read of one comes back only by a line here, and the list only shrinks.
OLDER_FORM_READS = {"certifier_serve": set(), "active_replay": set(), "active": set()}


def _imports(path: Path) -> dict[Path, set[str]]:
    """{a file of the tree: the names `path` imports from it, "*" for the module itself}, the module named bare beside
    `path` or dotted from the root, the imports inside `path`'s functions included."""
    def file(module: str) -> Path | None:
        beside, rooted = path.with_name(f"{module}.py"), ROOT.joinpath(*module.split(".")).with_suffix(".py")
        return beside if beside.is_file() else rooted if rooted.is_file() else None

    out: dict[Path, set[str]] = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            if f := file(node.module):
                out.setdefault(f, set()).update(a.name for a in node.names)
            for a in node.names:
                if f := file(f"{node.module}.{a.name}"):
                    out.setdefault(f, set()).add("*")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if f := file(a.name):
                    out.setdefault(f, set()).add("*")
    return out


def test_the_relays_runner_imports_from_the_older_forms_files_only_what_is_listed():
    """A file of the relay's runner is the runner's own or a file beside it that it reaches by its imports, the two
    older runners left out. Of the older runners and the older form's protocol file, only the runner's own imports, and
    `OLDER_FORM_READS` and no more, so once that is empty the older form can go without the relay; and the list names
    nothing the runner no longer imports."""
    runner = Path(ws.__file__)
    older = {runner.with_name("certifier_serve.py"), runner.with_name("active_replay.py"),
             ROOT.joinpath(*WR.__package__.split("."), "active.py")}
    assert all(f.is_file() for f in older) and {f.stem for f in older} == set(OLDER_FORM_READS), older
    files: dict[Path, dict[Path, set[str]]] = {}
    todo = [runner]
    while todo:
        if (path := todo.pop()) not in files:
            files[path] = _imports(path)
            todo += [f for f in files[path] if f.parent == runner.parent and f not in older]
    assert len(files) > 4, sorted(p.name for p in files)
    found = {(p.name, f.stem): names for p, imports in files.items() for f, names in imports.items() if f in older}
    more = {k: sorted(names - (OLDER_FORM_READS[k[1]] if k[0] == runner.name else set())) for k, names in found.items()}
    assert not any(more.values()), (
        "\n".join(f"{f} imports {', '.join(names)} from {m}.py" for (f, m), names in sorted(more.items()) if names)
        + f"\nA file of the relay reads nothing of the older form's files. Move the definition to a module of the relay's "
          f"own, or to one named for what it holds, beside {runner.name} (a definition of the protocol's: beside the "
          f"relay's protocol file), and have the older form's file import it from there, so that it keeps one definition.")
    stale = {m: sorted(gone) for m, names in OLDER_FORM_READS.items() if (gone := names - found.get((runner.name, m), set()))}
    assert not stale, (f"OLDER_FORM_READS lists what {runner.name} no longer imports from the older form's files: "
                       f"{stale}. Delete it from the list, which only shrinks.")


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
    """The auditor host's `warden-rehearse` (`verity/protocols/compose/device/live_challenger/auditor.py`) runs `rehearse` on traffic the
    gate fixes, its key at the request's height (the deployed key's, 20, by default), and refuses a request whose
    minutes are under `warden_minutes` of its requests and height: for every request count the gate takes, at its
    lowest height, those minutes cover the traffic's last arrival, its tail, the rest of the window it ends in, the
    origin's delay and 30 s for the key and the check."""
    sys.path.insert(0, str(ROOT / "verity/protocols/compose/device/live_challenger"))
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
    sys.path.insert(0, str(ROOT / "verity/protocols/compose/device/live_challenger"))
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
    assert set(A.LOAD_SHAPES) == set(LK.SHAPES)
    for shape in A.LOAD_SHAPES:
        values = LK.manifest(LK.SHAPES[shape])
        link = values[-1].offset + values[-1].nbytes + ws.lr.SALT_BYTES * sum(v.leaves for v in values)
        assert A.load_minutes(shape) * 60 >= link / A.LOAD_RATE + A.LOAD_READY_S > (A.load_minutes(shape) - 1) * 60
    assert (A.load_minutes("h200"), A.load_minutes("tiny")) == (46, 16)


def test_the_gates_sampling_is_on_every_request_and_moves_nothing_else_of_the_traffic(tmp_path):
    """`warden-rehearse` always passes its temperature and top_p (0 and 1 by default), and `rehearse` puts both on
    every request, so an endpoint's own generation config never applies; the arrivals, prompts, lengths and seeds stay
    the seed's, so `warden_minutes` still bounds the run."""
    sys.path.insert(0, str(ROOT / "verity/protocols/compose/device/live_challenger"))
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
    boundary, 8 an execution phase and D 2.7 s, the gate sessions' schedule. Its id traffic's prompts and outputs are
    within the scan's KI and KO, on the gate's arrivals at the gate's sampling; for every request count the gate takes,
    at its lowest height, `warden_minutes` covers the id traffic's last arrival, its tail, the rest of the window it
    ends in, the origin's delay and 30 s for the key and the check."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(ROOT / "verity/protocols/compose/device/live_challenger"))
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


def test_each_scan_class_the_gate_takes_is_scan_zks_setup_of_its_name_at_its_schedule_ki_and_ko(tmp_path, capsys):
    """Each class `warden-rehearse --scan --prompts --scan-class` takes (`auditor.SCAN_CLASSES`, the class of record
    `scan_zk.W63`'s by default) is `scan_zk.SETUPS`' of its name: the gate's network certifier runs at its schedule (`scan_zk.py
    schedule --class`, the one `scan_zk.py prove --class --warden-run` refuses any other of), which its class serves
    and which is the class of record's but for its egress cells, so the gate's group windows hold at it; and the gate
    bounds a prompt by its KI and the tokens by its KO. w63k4's class refuses the class of record's schedule (65
    egress cells past its KO + 1 = 5), so a w63k4 run needs its own."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(ROOT / "verity/protocols/compose/device/live_challenger"))
    import auditor as A
    import scan_zk as sz
    assert A.SCAN_CLASS == sz.W63.name and set(A.SCAN_CLASSES) == {"w63", "w63k4"} <= set(sz.SETUPS)
    (tmp_path / A.PROMPTS).write_text("[[1]]")
    for name in A.SCAN_CLASSES:
        s = sz.SETUPS[name]
        assert sz.main(["schedule", "--class", name]) == 0
        assert capsys.readouterr().out.split() == list(A.scan_schedule(scan_class=name)), name
        req = {"device": "warden", "requests": 4, "scan": True, "prompts": {"count": 1, "ids": 1}, "scan_class": name}
        args = ws.parser().parse_args(A.warden_command(req, tmp_path, "http://127.0.0.1:40001")[2:])
        params = ws._params(args)
        assert params == s.params == dataclasses.replace(sz.PARAMS, egress_cells=s.params.egress_cells), name
        s.cls.serves(params)
        assert A.scan_bounds(name) == ((1, s.cls.KI), (1, s.cls.KO)), name
        assert [ws.group_windows(params, t) for t in range(1, s.cls.KO + 1)] == \
            [A.scan_group_windows(t) for t in range(1, s.cls.KO + 1)], name
    with pytest.raises(ValueError, match="egress_cells"):
        sz.W63K4.cls.serves(sz.PARAMS)


def test_the_gates_warden_windows_are_at_least_every_request_counts_link_windows_which_its_key_signs(tmp_path):
    """The gate sizes a network certifier rehearsal's key on `warden_windows` of its requests (`warden_height`), so for
    every request count it takes, those are at least the execution phases `rehearse` runs (its traffic's last arrival
    and tail): at the scan's schedule on the gate's id traffic and on `--prompts`' arrivals, and at `warden_serve.py`'s
    default (`WARDEN_WINDOW_MS`) on its varied traffic; a request count's traffic is the first of the most's. The gate's
    most requests are 1064 execution phases of `--prompts`, about 57 minutes, which height 13 signs and 12 doesn't."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(ROOT / "verity/protocols/compose/device/live_challenger"))
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
    tokens, group g's n requests together half a bucket into execution phase g·G, G the gate's `scan_group_windows`
    (`group_windows` at the scan's schedule, for every token count the gate takes), and the run exactly its groups'
    windows (`groups`), which `warden_group_minutes` covers with the origin's delay and 30 s for the key and the
    check. Without `--window-per-group` the prompts go at the gate's arrivals."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(ROOT / "verity/protocols/compose/device/live_challenger"))
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
    class serves, and the gate counts its minutes and its key's execution phases at MS: for each tick it takes, in
    100 ms steps up to its longest (a block of 128 s, past the minute `warden_minutes` counts for a shorter one), at
    its lowest height, they cover every request count's id traffic (its last arrival, its tail, the rest of the block
    it ends in, the origin's delay and 30 s for the key and the check) and every group count's `--window-per-group`
    run."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(ROOT / "verity/protocols/compose/device/live_challenger"))
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


def test_the_gates_relay_session_is_session_at_the_scans_schedule_and_its_minutes_cover_its_length(tmp_path):
    """`relay-session` runs `session` at the listener the gate gives it, for the request's execution phases, at the
    schedule a scan rehearsal of its tick and class runs at, in the id egress form `serve` releases, and takes none
    of `rehearse`'s traffic. For every tick the gate takes, in 100 ms steps, and every count of execution phases
    that fits its minutes there, at the least height that signs them, `session_minutes` covers their span, the
    origin's delay and 30 s for the key and the check."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(ROOT / "verity/protocols/compose/device/live_challenger"))
    import auditor as A
    import scan_zk as sz
    req = {"device": "warden", "client_port": 8012, "windows": 20, "height": 10, "scan": True}
    argv = A.warden_command(req, tmp_path, "http://127.0.0.1:40001", listen=40002)
    args = ws.parser().parse_args(argv[2:])
    assert (args.command, args.upstream, args.listen, args.windows, args.height, args.out) == \
        ("session", ("127.0.0.1", 40001), ("127.0.0.1", 40002), 20, 10, tmp_path)
    assert args.ids and not args.fake and not args.watermark and args.egress == "id"
    assert ws._params(args) == sz.PARAMS and sz.PARAMS.egress == "id"
    for name in A.SCAN_CLASSES:
        given = ws.parser().parse_args(A.warden_command(req | {"scan_class": name, "watermark": True}, tmp_path,
                                                        "http://127.0.0.1:40001", listen=40002)[2:])
        assert ws._params(given) == sz.SETUPS[name].params and given.watermark, name
        rehearsal = {"device": "warden", "requests": 4, "scan": True, "scan_class": name, "prompts": {"count": 1}}
        (tmp_path / A.PROMPTS).write_text("[[1]]")
        assert ws._params(ws.parser().parse_args(A.warden_command(rehearsal, tmp_path, "http://h:1")[2:])) == \
            ws._params(given), name
    for flag in (["--requests", "4"], ["--rate", "1"], ["--seed", "1"], ["--prompts", "p.json"],
                 ["--window-per-group"], ["--tail", "1"], ["--ids"], ["--model", "m"], ["--lengths", "1"]):
        with pytest.raises(SystemExit):
            ws.parser().parse_args(argv[2:] + flag)

    lo, hi = A.SCAN_BUCKET_MS
    for tick in range(lo, hi + 1, 100):
        at = ws.parser().parse_args(A.warden_command(req | {"bucket_ms": tick}, tmp_path, "http://h:1", listen=1)[2:])
        params = ws._params(at)
        assert params == dataclasses.replace(sz.PARAMS, bucket_ms=tick), tick
        window_ms = params.window * tick
        for windows in range(A.SESSION_WINDOWS[0], A.SESSION_WINDOWS[1] + 1):
            minutes = A.session_minutes(windows, A.warden_height(windows), window_ms)
            if minutes > A.CERTIFIER_MINUTES[1]:
                break
            assert windows * window_ms / 1000 + at.start_in + 30 <= minutes * 60, (tick, windows)


def test_rehearse_reads_prompts_the_ingress_bound_holds_and_takes_them_and_groups_only_with_what_they_need(tmp_path):
    """`read_prompts` takes a JSON array of non-empty arrays of ids below 2^17 whose id requests the schedule's ingress
    bound holds, the gate's longest (`WARDEN_PROMPT_IDS` ids of 17 bits, 62-byte names) among them at the scan's, and
    the gate's most prompt ids and tokens the scan's KI and KO;
    `rehearse` takes `--prompts` only with `--ids` and `--window-per-group` only with `--prompts`, and a group is at most
    `lanes` prompts."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(ROOT / "verity/protocols/compose/device/live_challenger"))
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


def test_the_warden_always_paces_and_serve_and_rehearse_refuse_a_release_within_the_slack(tmp_path):
    """`serve` and `rehearse`, the deployed warden and the gate's rehearsal of it, always pace: `--paced-ms 0` (once the
    loop's write) and a release within the slack or past the next boundary are refused before anything runs."""
    serve = ["serve", "--upstream", "127.0.0.1:9", "--listen", "127.0.0.1:0", "--key", str(tmp_path / "key"),
             "--session", "00" * 32, "--windows", "1", "--out", str(tmp_path / "run")]
    for paced in ("0", "25", "100"):
        for argv in (serve, ["rehearse", "--fake", "--out", str(tmp_path / "run")]):
            with pytest.raises(SystemExit, match="the warden always paces"):
                ws.main([*argv, "--paced-ms", paced])
    assert not (tmp_path / "run").exists()


def test_a_group_of_id_requests_admitted_at_a_windows_start_ends_within_its_group_windows(tmp_path):
    """The bound `group_windows` rests on, on the network certifier itself at the scan's schedule: a group of `lanes` id
    requests admitted together at a window's first bucket, each answered with `tokens` id events and the trailer (the
    loop's two, and the gate's most, 64), ends every connection with its last cell D + ⌊tokens/r⌋ buckets after its
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
        result = RC.check(out)
        assert result["accepted"], result["problems"]
        assert (result["summary"]["ended"], result["summary"]["windows"]) == (p.lanes, G)
    assert (ws.group_windows(p, 2), ws.group_windows(p, 64)) == (1, 2)


def test_the_gates_demo_traffic_through_the_warden_is_the_demo_stand_ins_class_and_the_check_accepts_its_answers(
        tmp_path):
    """`warden-rehearse --scan --prompts --tokens 24` of the demo stand-in's prompts (`demo_stub.py prompts`) sends each
    prompt as it is, with exactly 24 tokens and greedy; on the network certifier at the scan's schedule, the stand-in,
    given exactly what the network certifier sent the server (its release headers on), serves every request and answers
    the bytes the network certifier certified, every connection ends within the two execution phases 24 tokens take at D
    = 27, and the check accepts the run."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(ROOT / "verity/protocols/compose/device/live_challenger"))
    import auditor as A
    import demo_stub as D
    import scan_zk as sz
    from verity.protocols.accounting.certifier.verifier.release import release_of
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
    result = RC.check(out)
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
            return await asyncio.to_thread(rp.served_model, f"http://127.0.0.1:{port}")
        finally:
            server.close()

    assert asyncio.run(models()) == "fake"
    assert ws.dechunk(b"3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n") == (b"abcde", True)
    assert ws.dechunk(b"3\r\nabc\r\n2\r\nd") == (b"abcd", False)


def test_the_fake_endpoint_reads_the_watermark_stream_to_its_end_and_answers_nothing():
    """`rehearse --fake --watermark`: the network certifier's watermark stream (`release`'s `POST /verity/watermark`)
    isn't a completion request, so the fake endpoint takes all of it, answers nothing and closes; `serve --watermark`
    and `rehearse --watermark` both take the flag."""
    stream = watermark_head(bytes(32)) + b"".join(watermark_chunk(t, n) for t, n in ((0, 0), (1, 2), (2, 2))) + \
        watermark_chunk(3, 4, [2]) + WATERMARK_END

    async def main() -> bytes:
        server = await asyncio.start_server(lambda r, w: ws.fake_endpoint(r, w, sleep=_no_sleep), "127.0.0.1", 0)
        reader, writer = await asyncio.open_connection("127.0.0.1", server.sockets[0].getsockname()[1])
        writer.write(stream)
        writer.write_eof()
        data = await reader.read()
        writer.close()
        server.close()
        return data

    assert asyncio.run(main()) == b""
    serve = ["serve", "--upstream", "h:1", "--listen", "h:0", "--out", "o", "--key", "k", "--session", "00",
             "--windows", "1"]
    assert [ws.parser().parse_args(serve + extra).watermark for extra in ([], ["--watermark"])] == [False, True]
    assert ws.parser().parse_args(["rehearse", "--fake", "--out", "o", "--watermark"]).watermark


def test_serve_releases_the_id_form_unless_told_and_rehearse_names_its_traffics_form(tmp_path):
    """`serve` is the deployed network certifier, so its egress form is the egress record's id form unless `--egress`
    names another; `rehearse` names its traffic's, id with `--ids` and bytes for its text, and passes it to its
    `serve`, whose deployment's digest binds it. A text response through the id form fails its connection at its first
    cell (format), and the check names it."""
    serve = ["serve", "--upstream", "h:1", "--listen", "h:0", "--out", "o", "--key", "k", "--session", "00",
             "--windows", "1"]
    rehearse = ["rehearse", "--fake", "--out", "o"]
    forms = [ws._params(ws.parser().parse_args(argv)).egress for argv in
             (serve, serve + ["--egress", "bytes"], rehearse, rehearse + ["--ids"], rehearse + ["--egress", "id"])]
    assert forms == ["id", "bytes", "bytes", "id", "id"]
    for argv, form in ((rehearse, "bytes"), (rehearse + ["--ids"], "id")):
        flags = ws._flags(ws.parser().parse_args(argv))
        assert flags[flags.index("--egress") + 1] == form
        assert ws._params(ws.parser().parse_args(serve + flags)) == ws._params(ws.parser().parse_args(argv))
    with pytest.raises(SystemExit):
        ws.parser().parse_args(serve + ["--egress", "text"])

    out = tmp_path / "run"
    out.mkdir()
    gated = dataclasses.replace(PARAMS, egress="id")
    requests = [ws.http_request({"prompt": "x"}, "h:1")]
    warden = _run(out, tmp_path / "key", requests, [_response([" a"])], params=gated)
    assert [(c["failed"], c["failed_at"]) for c in warden.log] == [("format", 0)]
    deployment = json.loads((out / "deployment.json").read_text())
    assert deployment["params"]["egress"] == "id" and deployment["digest"] == gated.digest.hex() != PARAMS.digest.hex()
    problems = RC.check(out)["problems"]
    assert any("connection 1 failed closed (format) at egress cell 0" in p for p in problems), problems


def test_a_response_cap_is_a_schedule_flag_rehearse_passes_to_serve():
    """`--max-tokens-cap` sets the schedule's response cap, which `rehearse` passes to its `serve` and the digest
    binds; without it the schedule has none and no flag carries it."""
    serve = ["serve", "--upstream", "h:1", "--listen", "h:0", "--out", "o", "--key", "k", "--session", "00",
             "--windows", "1"]
    rehearse = ["rehearse", "--fake", "--out", "o", "--ids"]
    capped = ws._params(ws.parser().parse_args(rehearse + ["--max-tokens-cap", "7"]))
    assert capped.max_tokens_cap == 7 and capped.digest != ws._params(ws.parser().parse_args(rehearse)).digest
    flags = ws._flags(ws.parser().parse_args(rehearse + ["--max-tokens-cap", "7"]))
    assert flags[flags.index("--max-tokens-cap") + 1] == "7"
    assert ws._params(ws.parser().parse_args(serve + flags)) == capped
    assert "--max-tokens-cap" not in ws._flags(ws.parser().parse_args(rehearse))
    assert ws._params(ws.parser().parse_args(serve)).max_tokens_cap is None


def test_the_fake_endpoint_answers_an_id_request_id_form_as_the_scans_egress_reads_it():
    """`rehearse --ids`: an `id_traffic` request, sent network-certifier-form (`id_form.id_request`), is answered with
    its `fake_ids` as id events of P bytes and the trailer, every cell an id-form `DATA` cell, then `END`'s trailer;
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
         id_payload: int | None = None, load_heads: Path | None = None, arrive: dict[int, int] | None = None,
         late: dict[int, float] | None = None) -> Warden:
    """A run of the network certifier at `params` for `windows` windows on a `VirtualClock`, published as `serve` and
    `client` publish one: its records and certificates through `Publisher` inline (`serve`'s `PublisherProcess` when
    `process`, which first signs the load's heads `load_heads` when given), each window committed by `commit`, its
    connection rows kept as the publisher process keeps them when the schedule has lengths rows (its key then a
    height taller, for their heads), `deployment.json`, and `client.json` with every cell's arrival 1 ms into the
    bucket it was sent in, each response read id-form at P = `id_payload` when given; `servers` gets the server's side
    of each connection, in connect order. Returns the network certifier."""
    tau = params.bucket_ms / 1000
    height = 5 if load_heads or params.lengths else 4
    if process:
        from benchmarks.network_traces import warden_serve  # by name, so that the spawned process finds its functions

        ws.ck.load_key(key, height)
        publisher = warden_serve.PublisherProcess(out, key, SESSION, params, commit, load_heads=load_heads)
        hooks = {"publish": publisher, "withhold": publisher.withhold, "release": publisher.release,
                 "lengths": publisher.lengths}
        public = publisher.key
    else:
        signer = ws.ck.signer(key, height)
        issuer = ws.Issuer(signer, SESSION)
        publisher = ws.Publisher(out, issuer, params, commit or WR.Cells.of)
        hooks, public = {"publish": publisher, "withhold": publisher.withhold,
                         "lengths": rp.Keeper(out, issuer, params, RC.LINK).keep}, signer.public_key()
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

        async def connect_due(t: int) -> None:
            # request j connects before the run's first tick, or just before tick `arrive[j]`: a later request no
            # earlier, so that the clients are in the requests' order
            for j, request in enumerate(requests):
                if (arrive or {}).get(j, 0) == t:
                    reader = asyncio.StreamReader()
                    reader.feed_data(cells(request, params.cell_bytes))
                    reader.feed_eof()
                    clients.append(_Writer(warden))
                    tasks.append(asyncio.create_task(warden.handle(reader, clients[-1])))
            await warden.until(lambda w: w.ordinals + w.refused == len(clients))

        await connect_due(0)
        PROCESS_CLOCK.append(clock)
        try:
            for t in range(windows * params.window):
                if t:
                    await connect_due(t)
                await warden.until(settled)
                # tick t runs `late[t]` seconds past its time
                clock.advance(t * tau + (late or {}).get(t, 0))
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
    release = {e["ordinal"]: (e["release"], e["ingress_end"]) for e in warden.log}
    assert sorted(n for n, _ in release.values()) == [0, 1, 2]
    assert [b"".join(d for _, d in s.writes) for s in servers] == \
        [with_release(r, release[k + 1][0], warden.run_id, release[k + 1][1]) for k, r in enumerate(requests)]
    assert json.loads((out / "deployment.json").read_text())["run_id"] == warden.run_id.hex()
    result = RC.check(out)
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
            return RC.check(out)["problems"]
        finally:
            for p, data in saved.items():
                p.write_bytes(data)

    def flip_record() -> None:
        records = RC.read_records(out, PARAMS)
        record = records["responses"][0]
        row = next(b for b, r in enumerate(record.grid) if any(f is not None and f[0] == DATA for f in r))
        frames = list(record.grid[row])
        k = next(i for i, f in enumerate(frames) if f is not None and f[0] == DATA)
        frames[k] = frames[k][:-1] + bytes([frames[k][-1] ^ 1])
        grid = record.grid[:row] + (tuple(frames),) + record.grid[row + 1:]
        (out / "responses.records").unlink()
        ws.bf.write_blobs(out / "responses.records", RF.pack_record(ws.replace(record, grid=grid)))

    problems = refused(flip_record)
    assert "responses window 0 isn't certified" in problems and "responses window 0's cells aren't registered" in problems

    def edit_cells(change) -> None:
        lines = [json.loads(x) for x in (out / "cells.jsonl").read_text().splitlines()]
        salts = ws.bf.read_blobs(out / "cells.salts")
        change(lines, salts)
        (out / "cells.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        (out / "cells.salts").unlink()
        for blob in salts:
            ws.bf.write_blobs(out / "cells.salts", blob)

    def other_root(lines, salts) -> None:
        lines[1]["registration"]["root"] = "00" * 64

    def other_salt(lines, salts) -> None:
        salts[1] = bytes([salts[1][0] ^ 1]) + salts[1][1:]

    def other_wiring_salt(lines, salts) -> None:
        salts[1] = salts[1][:-1] + bytes([salts[1][-1] ^ 1])

    for change in (other_root, other_salt, other_wiring_salt):
        assert "responses window 0's cells aren't registered" in refused(lambda: edit_cells(change))

    def forge_reason() -> None:
        (record,) = RC.read_records(out, PARAMS)["fails"]
        grid = ((reason_frame(0, "late", PARAMS.cell_bytes),) + record.grid[0][1:],) + record.grid[1:]
        (out / "fails.records").unlink()
        ws.bf.write_blobs(out / "fails.records", RF.pack_record(ws.replace(record, grid=grid)))

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
    assert RC.check(out)["accepted"]


@pytest.mark.parametrize("process", [False, True])
def test_a_window_with_a_refused_connection_is_certified_nowhere(tmp_path, process):
    """Five connections at once on two lanes and a queue of four: the fifth is refused, traffic past the window's
    Program slots, so window 0's three records are kept and its two rows committed but none is certified or signed,
    each line naming the execution phase under `refused`, withheld, inline and in `serve`'s publisher process; window 1
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
    assert [RF.certified_from_json(x).window for x in certified if "refused" not in x] == [1] * 3
    assert [("refused" in x, x.get("withheld")) for x in cells_] == [(True, True)] * 2 + [(False, None)] * 2
    result = RC.check(out)
    s = result["summary"]
    assert (s["refused"], s["ended"], s["certified_windows"], s["cell_rows"]["registered"]) == (1, 4, 3, 2)
    assert {p for p in result["problems"] if "window" in p} == \
        {f"{name} window 0 isn't certified" for name in RC.CERTIFIED} | {
            f"{name} window 0's cells aren't registered" for name in ("requests", "responses")}


def _session(tmp_path: Path, windows: int):
    """The gate's `relay-session` command for `windows` execution phases at the scan's w63k4 schedule, parsed, with its
    schedule, and a maker of `n` id requests of the class's most tokens with their id-form answers."""
    sys.path.insert(0, str(ROOT / "verity/protocols/compose/device/live_challenger"))
    import auditor as A
    req = {"device": "warden", "client_port": 8012, "windows": windows, "height": 5, "scan": True,
           "scan_class": "w63k4"}
    args = ws.parser().parse_args(A.warden_command(req, tmp_path / "run", "http://127.0.0.1:40001", listen=40002)[2:])
    p = ws._params(args)
    tokens = p.egress_cells - 1

    def load(n: int) -> tuple[list[bytes], list[bytes]]:
        return ([ws.id_request(model="m", user=f"user-{j}", prompt=[j + 1], max_tokens=tokens,
                               frame_bytes=p.cell_bytes, offset=0) for j in range(n)],
                [ws.id_response(list(range(tokens)), p.payload)] * n)
    return args, p, load


def test_a_session_serves_its_length_to_clients_not_its_own_and_its_check_accepts_the_records_alone(
        tmp_path, monkeypatch):
    """`session` as the gate's `relay-session` runs it: one `serve` at the gate's listener for the request's
    execution phases and no other process, so no traffic and no client of its own. In `serve`'s place here, the
    relay on a virtual clock takes a load from another party's clients, eight connections arriving across the
    session's three execution phases, the last at the last bucket that still ends inside the run, and their log
    stays with them. The run's own check, on the records alone, accepts: every execution phase certified and
    registered and the audit, counting no connection. The clients' check is theirs, from their own log against the
    run's records: every connection ended, its request and its answer the audited connection's, each cell of the
    egress schedule."""
    args, p, load = _session(tmp_path, 3)
    last = args.windows * p.window - p.span - 1
    arrive = [0, 0, 7, p.window - 1, p.window, p.window + 8, 2 * p.window, last]
    requests, responses = load(len(arrive))
    served, own = [], {}

    class Serve:
        """`serve` as a process: the relay for its `--windows` on a virtual clock, its clients' log kept apart."""

        def __init__(self, argv, **kw):
            a = ws.parser().parse_args(argv[2:])
            assert a.command == "serve" and ws._params(a) == p
            served.append((a, _run(a.out, a.key, requests, responses, windows=a.windows, params=p,
                                   id_payload=p.payload, arrive=dict(enumerate(arrive)))))
            own.update(json.loads((a.out / "client.json").read_text()))
            (a.out / "client.json").unlink()
            self.stdout = io.StringIO(json.dumps({"listening": list(a.listen)}) + "\n")

        def wait(self) -> int:
            return 0

        poll = wait

    monkeypatch.setattr(ws, "subprocess", types.SimpleNamespace(Popen=Serve, PIPE=None))
    assert ws.rehearse(args) == 0
    ((a, relay),) = served
    out = args.out
    assert (a.listen, a.upstream, a.windows) == (("127.0.0.1", 40002), ("127.0.0.1", 40001), 3)
    assert {e["admitted"] // p.window for e in relay.log} == {0, 1, 2} and relay.log[-1]["admitted"] == last
    assert not {"traffic.json", "client.json", "client.log", "groups.json"} & {f.name for f in out.iterdir()}
    result = json.loads((out / "check.json").read_text())
    assert result["accepted"] and result["problems"] == [] and result["connections"] == []
    assert (result["model"], result["exit"]) == (None, {"serve": 0, "client": None})
    s = result["summary"]
    assert (s["requests"], s["ended"], s["cells_in_their_bucket"], s["windows"], s["certified_windows"]) == \
        (0, 0, [0, 0], 3, 9)
    assert (s["cell_rows"]["registered"], s["cell_rows"]["of"]) == (6, 6)
    assert s["data_bytes"] == {"ingress": sum(map(len, requests)), "egress": sum(map(len, responses))}

    records = RC.read_records(out, p)
    audited = {c["ordinal"]: c for c in
               warden_.audit(p, records[RC.INGRESS], records[RC.EGRESS], records[RC.FAILS])["connections"]}
    schedule = RC.egress_cells(records[RC.EGRESS], p)
    assert len(own["connections"]) == len(audited) == len(arrive)
    for conn in own["connections"]:
        o = conn["ordinal"]
        assert conn["outcome"] == "end" and audited[o]["ended"] is not None, o
        assert (audited[o]["ingress_sha256"], audited[o]["egress_sha256"]) == \
            (conn["request_sha256"], conn["data_sha256"]), o
        assert "".join(RC.KIND_CHARS[k] for _, k in schedule[o]) == conn["kinds"], o
    (out / "client.json").write_text(json.dumps(own))
    theirs = RC.check(out)
    assert theirs["accepted"] and (theirs["summary"]["requests"], theirs["summary"]["ended"]) == (8, 8)
    assert theirs["summary"]["cells_in_their_bucket"][0] == theirs["summary"]["cells_in_their_bucket"][1] > 0


def test_a_session_given_its_callers_listener_hands_it_to_serve_which_serves_on_it_and_leaves_it_listening(
        tmp_path, monkeypatch):
    """The gate's unit holds a session's listener and gives the runner its descriptor (`--listen-fd`). `session`
    passes that descriptor on to `serve`'s process and names it there, and without one passes none. The relay's
    server on a caller's socket binds nothing: a connection made to the caller's listener before the server starts
    is the one its handler gets, and once the server has closed the caller's socket still listens, so a later
    connection arrives there with nothing accepting it."""
    sys.path.insert(0, str(ROOT / "verity/protocols/compose/device/live_challenger"))
    import auditor as A
    req = {"device": "warden", "client_port": 8012, "windows": 1, "height": 5, "scan": True, "scan_class": "w63k4"}

    class Started(Exception):
        pass

    def popen(argv, **kw):
        raise Started(argv, kw)

    monkeypatch.setattr(ws, "subprocess", types.SimpleNamespace(Popen=popen, PIPE=None))
    for fd in (None, 9):
        argv = A.warden_command(req, tmp_path / f"run-{fd}", "http://127.0.0.1:40001", listen=40002, listen_fd=fd)
        with pytest.raises(Started) as started:
            ws.rehearse(ws.parser().parse_args(argv[2:]))
        serve, kw = started.value.args
        a = ws.parser().parse_args(serve[2:])
        assert (a.command, a.listen, a.listen_fd) == ("serve", ("127.0.0.1", 40002), fd)
        assert kw["pass_fds"] == (() if fd is None else (fd,))

    held = A.held_listener()
    early = socket.create_connection(held.getsockname())
    early.sendall(b"early")

    async def relay() -> list[bytes]:
        got, done = [], asyncio.Event()

        async def handle(reader, writer):
            got.append(await reader.readexactly(5))
            writer.write(b"served")
            await writer.drain()
            writer.close()
            done.set()

        server = await ws.listening(handle, held.getsockname(), socket.socket(fileno=os.dup(held.fileno())))
        assert [s.getsockname() for s in server.sockets] == [held.getsockname()]
        await done.wait()
        server.close()
        await server.wait_closed()
        return got

    assert asyncio.run(relay()) == [b"early"] and early.recv(6) == b"served"
    with socket.create_connection(held.getsockname()) as late:
        late.sendall(b"late")
        conn, _ = held.accept()
        assert conn.recv(4) == b"late"
        conn.close()
    early.close()
    held.close()


def test_serve_through_its_command_line_serves_on_the_socket_its_descriptor_names_and_binds_none(
        tmp_path, monkeypatch, capsys):
    """`serve --listen-fd FD` through `main` and `serve` themselves, on the command `session` issues for a listener
    its caller holds: `main` makes the socket FD names, and `serve` gives it to the relay's server. The relay, its
    publisher, the stall probe and the process's preparation are stood in for, so nothing ticks: the relay here ends
    once its handler has answered one connection. That connection is one made to the caller's listener before
    `serve` started, the deployment and the line `serve` prints name the caller's port, and the caller's socket
    still listens once `serve` has returned. With either link cut, the flag to the socket or the socket to the
    server, `serve` binds `--listen`, which is the caller's own address and is taken."""
    sys.path.insert(0, str(ROOT / "verity/protocols/compose/device/live_challenger"))
    import auditor as A
    args, _, _ = _session(tmp_path, 1)
    held = A.held_listener()
    args.listen, args.listen_fd = held.getsockname(), os.dup(held.fileno())

    class Started(Exception):
        pass

    def popen(argv, **kw):
        raise Started(argv, kw)

    monkeypatch.setattr(ws, "subprocess", types.SimpleNamespace(Popen=popen, PIPE=None))
    with pytest.raises(Started) as started:
        ws.rehearse(args)
    serve, kw = started.value.args
    assert kw["pass_fds"] == (args.listen_fd,)

    public, got = ws.ck.signer(tmp_path / "key", 4).public_key(), []

    class Publisher:
        """`PublisherProcess` with no process: the key's public half, and nothing published."""

        usage = feeder = None
        process = types.SimpleNamespace(pid=None)

        def __init__(self, *a, **kw) -> None:
            self.key = public

        def __call__(self, *a) -> None:
            pass

        withhold = release = lengths = __call__

        def close(self) -> list:
            return []

    class Relay:
        """The relay's stand-in: its run ends once its handler has answered one connection."""

        run_id, lateness, log, ordinals, refused, refused_windows, ticks, releases = bytes(16), [], [], 0, 0, set(), 0, []

        def __init__(self, *a, **kw) -> None:
            self.answered = asyncio.Event()

        async def handle(self, reader, writer) -> None:
            got.append(await reader.readexactly(5))
            writer.write(b"served")
            await writer.drain()
            writer.close()
            self.answered.set()

        async def run(self) -> None:
            await self.answered.wait()

    monkeypatch.setattr(ws, "PublisherProcess", Publisher)
    monkeypatch.setattr(ws, "Warden", Relay)
    monkeypatch.setattr(ws, "prepare_process", lambda params: {})
    monkeypatch.setattr(ws, "stall_probe", types.SimpleNamespace(start=lambda *a: None, stop=lambda *a: None))
    monkeypatch.setattr(ws, "_quiet_runtime", lambda: None)
    for name in ("PYTHONMALLOC", ws.C_ALLOCATOR):
        monkeypatch.delenv(name, raising=False)
    early = socket.create_connection(held.getsockname())
    early.sendall(b"early")
    capsys.readouterr()
    assert ws.main(serve[2:]) == 0
    assert got == [b"early"] and early.recv(6) == b"served"
    a = ws.parser().parse_args(serve[2:])
    said = json.loads(capsys.readouterr().out.splitlines()[0])
    assert said["listening"] == list(held.getsockname())
    assert json.loads((a.out / "deployment.json").read_text())["listen"] == list(held.getsockname())
    with socket.create_connection(held.getsockname()) as late:
        late.sendall(b"late")
        conn, _ = held.accept()
        assert conn.recv(4) == b"late"
        conn.close()
    early.close()
    held.close()


def test_a_connection_still_open_when_a_session_ends_is_cut_and_the_records_say_which(tmp_path):
    """A session stops at its last execution phase's end and cuts what is open, and the records say what each was. One
    admitted at bucket s with s + D inside the run and no room to end (the first bucket past windows · T − span − 1)
    has data and no `END` in the record: the audit refuses it in its admission's execution phase, and the check the
    run, every execution phase still certified. Ones admitted later are cut before s + D: their requests are whole in
    the ingress record, their egress is padding, the audit takes them and the check accepts, counting them as not
    ended. One still in the queue then (it arrived behind four that hold the lanes) got no cell and is in no
    record. The clients' own logs say `cut` for each."""
    args, p, load = _session(tmp_path, 2)
    end = args.windows * p.window
    assert (p.lanes, p.queue, p.delay, p.span, end) == (4, 4, 27, 28, 64)

    def run(name: str, arrive: list[int]):
        out = tmp_path / name
        out.mkdir()
        relay = _run(out, tmp_path / f"key-{name}", *load(len(arrive)), windows=args.windows, params=p,
                     id_payload=p.payload, arrive=dict(enumerate(arrive)))
        own = json.loads((out / "client.json").read_text())["connections"]
        (out / "client.json").unlink()
        records = RC.read_records(out, p)
        return relay, own, RC.check(out), warden_.audit(p, records[RC.INGRESS], records[RC.EGRESS], records[RC.FAILS])

    relay, own, result, audit_ = run("ends", [end - p.span - 1])
    assert result["accepted"] and relay.log[0]["ended"] == end - 1 and not relay.log[0]["cut"]
    assert [c["outcome"] for c in own] == ["end"] and audit_["connections"][0]["ended"] == end - 1

    relay, own, result, audit_ = run("late", [end - p.span])
    assert relay.log[0]["cut"] and relay.log[0]["ended"] is None and [c["outcome"] for c in own] == ["cut"]
    assert own[0]["kinds"].count(RC.KIND_CHARS[DATA]) == p.slots and RC.KIND_CHARS[END] not in own[0]["kinds"]
    assert not result["accepted"] and result["problems"] == [
        "audit: connection 1: its egress isn't padding to s + D, data in every slot, one END or FAIL, padding"]
    assert [(w["window"], w["accepted"]) for w in audit_["windows"]] == [(0, True), (1, False)]
    assert result["summary"]["certified_windows"] == 6 and all(w["certified"] for w in result["windows"])

    early = end - p.delay + 3
    relay, own, result, audit_ = run("early", [early] * p.lanes + [early + 1])
    assert result["accepted"] and result["problems"] == [] and audit_["accepted"]
    assert [(e["admitted"], e["ended"], e["cut"], e["ingress_cells"], e["egress_cells"]) for e in relay.log] == \
        [(early, None, True, 1, 0)] * p.lanes + [(None, None, True, 0, 0)]
    assert [(c["ordinal"], c["admitted"], c["ended"], c["egress_bytes"]) for c in audit_["connections"]] == \
        [(o, early, None, 0) for o in range(1, p.lanes + 1)]
    assert all(c["ingress_bytes"] > 0 for c in audit_["connections"])
    assert [c["outcome"] for c in own] == ["cut"] * (p.lanes + 1)
    assert [set(c["kinds"]) for c in own] == [{RC.KIND_CHARS[link_frame.PAD]}] * p.lanes + [set()]
    assert result["summary"]["data_bytes"] == {"ingress": sum(map(len, load(p.lanes)[0])), "egress": 0}
    assert result["summary"]["certified_windows"] == 6


def test_one_connection_failed_closed_refuses_a_sessions_whole_run_whichever_client_it_was(tmp_path):
    """The check's verdict is the whole session's. Of four connections, three end and one's answer is out of the id
    form, which the relay fails closed: every execution phase is still certified, the audit refuses the one the
    failure is in and accepts the other, and the check refuses the run, whoever's the three that ended were."""
    args, p, load = _session(tmp_path, 2)
    requests, responses = load(4)
    out = tmp_path / "run"
    out.mkdir()
    relay = _run(out, tmp_path / "key", requests, responses[:3] + [_response([" a"])], windows=args.windows,
                 params=p, id_payload=p.payload, arrive=dict(enumerate([0, 0, 1, 2])))
    (out / "client.json").unlink()
    records = RC.read_records(out, p)
    audit_ = warden_.audit(p, records[RC.INGRESS], records[RC.EGRESS], records[RC.FAILS])
    result = RC.check(out)
    assert sorted(e["failed"] or "" for e in relay.log) == ["", "", "", "format"]
    assert [(c["failed"], c["ended"] is not None) for c in audit_["connections"]] == \
        [(None, True)] * 3 + [("format", True)]
    assert sorted(w["accepted"] for w in audit_["windows"]) == [False, True]
    assert not result["accepted"] and all(w["certified"] for w in result["windows"])
    assert result["problems"] == ["audit: connection 4 failed closed (format) at egress cell 0"]


def test_the_check_names_each_execution_phase_the_audit_refused_and_why_uncut(tmp_path, monkeypatch):
    """`check.json`'s `audit_windows` is the audit's verdict of each execution phase, whole. Of a session's two, the
    second, where a connection fails closed, is refused with that problem and the first accepted, while `problems`
    gives the problem's text without its execution phase; a run the audit accepts has each one accepted, with no
    problem. And where the audit has more problems than the 50 `problems` keeps, `audit_windows` has them all."""
    args, p, load = _session(tmp_path, 2)
    requests, responses = load(4)

    def run(name: str, answers: list[bytes]) -> dict:
        out = tmp_path / name
        out.mkdir()
        _run(out, tmp_path / f"key-{name}", requests, answers, windows=args.windows, params=p, id_payload=p.payload,
             arrive=dict(enumerate([0, 0, 1, p.window])))
        (out / "client.json").unlink()
        return RC.check(out)

    result = run("ends", responses)
    assert result["accepted"] and result["audit_windows"] == [
        {"window": 0, "accepted": True, "problems": []}, {"window": 1, "accepted": True, "problems": []}]

    result = run("fails", responses[:3] + [_response([" a"])])
    why = "connection 4 failed closed (format) at egress cell 0"
    assert not result["accepted"] and result["problems"] == [f"audit: {why}"]
    assert result["audit_windows"] == [
        {"window": 0, "accepted": True, "problems": []}, {"window": 1, "accepted": False, "problems": [why]}]

    audit = warden_.audit

    def many(*records) -> dict:
        verdict = audit(*records)
        verdict["windows"][1]["problems"] += [f"problem {i}" for i in range(60)]
        return verdict

    monkeypatch.setattr(RC, "audit", many)
    result = RC.check(tmp_path / "fails")
    assert len(result["problems"]) == 50 and result["problems"][:2] == [f"audit: {why}", "audit: problem 0"]
    assert result["audit_windows"][1]["problems"] == [why] + [f"problem {i}" for i in range(60)]


def test_a_windows_commit_and_signatures_move_no_tick_past_its_slack_in_the_publisher_process(tmp_path):
    """A window's commit that takes `SLOW_COMMIT_S` of its process's time (`slow_commit`), run in the network certifier's process
    as the inline `Publisher` runs it, puts the ticks after the window's end past their slack: the next window faults
    (`monitor-unavailable`), the issuer certifies none of its records, and the check refuses it. In `serve`'s
    `PublisherProcess` the same commit runs in the spawned process, which signs and commits from the rows released to
    it: every tick is on time, and the check accepts every link-window, certified and registered. A commit that fails
    there fails `close`, and the warden's process doesn't wait on it."""
    requests = [ws.http_request({"prompt": "x" * n}, "h:1") for n in (10, 200, 30)]
    responses = [_response([" a", " b"]), _response([" word"] * 40), _response([])]
    runs = {name: tmp_path / name for name in ("inline", "process", "refusing")}
    for out in runs.values():
        out.mkdir()

    inline = _run(runs["inline"], tmp_path / "inline-key", requests, responses, windows=2, commit=slow_commit)
    late = [t for t, x in enumerate(inline.lateness) if x > inline.slack]
    assert late[0] == PARAMS.window and max(inline.lateness) > SLOW_COMMIT_S
    result = RC.check(runs["inline"])
    assert {f"{name} window 1 isn't certified" for name in RC.CERTIFIED} <= set(result["problems"])
    assert {(x["record"], x["status"]) for x in result["windows"] if x["window"] == 1} == \
        {(name, Reason.MONITOR_UNAVAILABLE.value) for name in RC.CERTIFIED}
    assert sorted(w for *_, w in result["summary"]["signer_refused"]) == [1] * len(RC.CERTIFIED)

    process = _run(runs["process"], tmp_path / "process-key", requests, responses, windows=2, commit=slow_commit,
                   process=True)
    assert len(process.lateness) == 2 * PARAMS.window and max(process.lateness) == 0
    result = RC.check(runs["process"])
    assert result["accepted"], result["problems"]
    s = result["summary"]
    assert (s["certified_windows"], s["cell_rows"]["registered"], s["ended"], s["signer_refused"]) == (6, 4, 3, [])

    with pytest.raises(RuntimeError, match="a refusing commit"):
        _run(runs["refusing"], tmp_path / "refusing-key", requests, responses, commit=refusing_commit, process=True)


def test_the_load_is_signed_by_the_serving_publisher_in_the_runs_session_before_its_windows(tmp_path):
    """The gate's load end to end at the tiny shape: the store relayed over four streams to the prover's sink on
    loopback (`load_relay.py sink`, a process), then a run whose `PublisherProcess`, the network certifier's one signer,
    signs the relay's heads first: `load.json` is under the deployment's key and session, on the key's first leaves, and
    `check` reads every value's registration at the relay's roots; another session's, values swapped, heads that
    aren't the relay's and an unsigned load are refused."""
    import hashlib
    import subprocess

    store, out = tmp_path / "store", tmp_path / "run"
    store.mkdir()
    out.mkdir()
    lr, program = ws.lr, hashlib.sha512(b"the reading program").hexdigest()
    lr.write_store(store, LK.TINY)
    sink = subprocess.Popen([sys.executable, str(ROOT / "benchmarks/network_traces/load_relay.py"), "sink",
                             "--listen", "127.0.0.1:0",
                             "--weights", str(tmp_path / "weights.bin"), "--witness", str(tmp_path / "witness.bin")],
                            stdout=subprocess.PIPE, text=True)
    port = json.loads(sink.stdout.readline())["listening"][1]
    lr.relay_load(store, LK.TINY, [("127.0.0.1", port)] * 4, program, out)
    assert json.loads(sink.communicate()[0].splitlines()[-1])["accepted"] and sink.returncode == 0
    requests = [ws.http_request({"prompt": "x" * n}, "h:1") for n in (10, 30)]
    _run(out, tmp_path / "key", requests, [_response([" a", " b"]), _response([])], process=True,
         load_heads=out / "load-heads.json")
    result = RC.check(out)
    assert result["accepted"], result["problems"]
    n = len(LK.manifest(LK.TINY))
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
            return RC.check(out)["problems"]
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
