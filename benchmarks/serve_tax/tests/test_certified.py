"""The arm behind the network certifier (`serve_tax.certified`): its data, its schedule, each side's commands, what
ends the server's side, and the client's side from the certifier's start to its record, with stand-ins for the
processes it starts."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import serve_tax as S
import servetax_plugin as P

ARM = "tick-certifier"


def test_the_arm_is_the_tick_arm_with_the_certifier_in_front(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The arm's engine is the tick arm's, with the two changes of the plugin that an id request and the tick's
    server make necessary: each request held to its length, and the prefix cache emptied by the engine. Its tick's
    bucket is the certifier's. Its cost holds a change no part of the target counts, so it has no verdict against
    it."""
    glm, tick, arm = S.MODELS["glm53"], S.ARMS["tick"], S.ARMS[ARM]
    assert replace(arm, parent=tick.parent, what=tick.what, plugin=tick.plugin, certifier=False) == tick
    assert (arm.parent, arm.plugin, arm.certifier) == ("tick", ("record", "to_length", "idle_reset"), True)
    assert [n for n, x in S.ARMS.items() if x.certifier] == [ARM]
    assert S.server_argv(arm, glm, Path("/ck"), 8000) == S.server_argv(tick, glm, Path("/ck"), 8000)
    assert S.entry_points(arm) == S.entry_points(tick)
    (tmp_path / "tick_plugin.py").touch()
    monkeypatch.setattr(S, "TICK_PLUGIN", tmp_path / "tick_plugin.py")
    monkeypatch.setattr(S, "corpus", lambda: NS(TICK_BUCKET_MS=100))
    cells = S.certifier_cells(None, {"chat": (96,)})
    tau = S.tick_args(None, "220,230", bucket_ms=cells["bucket_ms"])
    assert tau == {"bucket_us": 250000, "sat_us": 220000, "free_us": 230000, "calibrated_from": None}
    assert S.tick_args(None, "220,230")["bucket_us"] == 100000
    env = S.server_env(arm, tmp_path, tmp_path / "site", tau, 600.0, 600.0)
    plugin = json.loads(env[S.PLUGIN_ENV])
    assert {k: plugin.get(k) for k in ("record", "to_length", "idle_reset", "release_order")} == {
        "record": True, "to_length": True, "idle_reset": True, "release_order": None}
    assert {k: json.loads(env["VERITY_TICK"])[k] for k in ("bucket_us", "sat_us", "free_us")} == {
        "bucket_us": 250000, "sat_us": 220000, "free_us": 230000}
    assert S.changes_of(ARM) == [*S.changes_of("tick"), "the network certifier in front"]
    assert S.not_in_target(glm, ARM) == ["the step record", "the network certifier in front"]
    assert [x["off"] for x in S.target_parts(glm, ARM)][0] == []
    # a pass waits out the engine's idle time, so that the engine empties its prefix cache for it
    assert S.IDLE_S > P.IDLE_RESET_S


def test_the_schedule_has_a_lane_and_a_queue_place_for_each_request_in_flight():
    cells = S.certifier_cells(None, {"chat": (48, 96)})
    assert cells == {**S.CELLS, "lanes": 96, "queue": 96}
    assert S.ceiling(cells) == 384.0 and S.cell_payload(cells) == 4096 - 9
    assert S.certifier_cells("delay=20, lanes=128,max_tokens_cap=256", {"chat": (96,)}) == {
        **S.CELLS, "delay": 20, "lanes": 128, "queue": 96, "max_tokens_cap": 256}
    for text, why in (("queue=64", "a queue of 64 under the rung 96"), ("cap=256", "NAME=N, NAME one of cell_bytes"),
                      ("slots=two", "NAME=N"), ("max_tokens_cap=4096", "max_tokens_cap is a u32 from 1 to"),
                      ("cell_bytes=8", "a cell holds its")):
        with pytest.raises(SystemExit, match=why):
            S.certifier_cells(text, {"chat": (96,)})
    assert S.certified_loads(ARM, "chat=96") == {"chat": (96,)}
    for text, why in (("long", "not long"), ("chat=96,long", "not long"), ("prefix", "not prefix"), (None, "--loads")):
        with pytest.raises(SystemExit, match=why):
            S.certified_loads(ARM, text)


def test_each_sides_commands_are_the_fronts_the_certifiers_and_the_shims(tmp_path: Path):
    cells = S.certifier_cells(None, {"chat": (96,)})
    assert S.front_argv(8000, 18746, S.cell_payload(cells), tmp_path / "front.jsonl")[1:] == [
        "-m", "verity_vllm.pipeline.cli", "ids-front", "--upstream", "127.0.0.1:8000", "--listen", "127.0.0.1:18746",
        "--id-payload", "4087", "--log", str(tmp_path / "front.jsonl")]
    argv = S.certifier_argv(cells, tmp_path / "certifier", 18746, tmp_path / "key", "ab" * 32, 16)
    assert argv[1:3] == [str(S.CERTIFIER), "serve"] and S.CERTIFIER.is_file() and "--watermark" in argv
    named = {argv[i]: argv[i + 1] for i in range(3, len(argv) - 1) if argv[i].startswith("--") and argv[i] != "--watermark"}
    assert named == {"--out": str(tmp_path / "certifier"), "--listen": "127.0.0.1:0", "--upstream": "127.0.0.1:18746",
                     "--key": str(tmp_path / "key"), "--session": "ab" * 32, "--windows": "16", "--egress": "id",
                     "--cell-bytes": "4096", "--slots": "1", "--bucket-ms": "250", "--delay": "60", "--window": "600",
                     "--ingress-cells": "4", "--ingress-buckets": "16", "--egress-cells": "4096",
                     "--max-tokens-cap": "2048", "--lanes": "96", "--queue": "96"}
    assert S.shim_argv(tmp_path / "deployment.json", Path("/ck"), tmp_path / "shim.jsonl")[1:] == [
        str(S.SHIM), "--deployment", str(tmp_path / "deployment.json"), "--tokenizer", "/ck", "--log",
        str(tmp_path / "shim.jsonl"), "--listen", "127.0.0.1:0"]
    assert S.SHIM.is_file()


def test_the_servers_side_holds_its_engine_until_the_certifiers_run_ends_or_its_stop_time(tmp_path: Path):
    feed = tmp_path / "tick-feed.jsonl"
    assert S.runs_ended(feed) == 0
    lines = [{"key": "chat@96-r1-0", "run": "aa", "n": 0, "bucket": 21}, {"run": "aa", "t": 21, "n": 1},
             {"run": "aa", "n": 0, "passed": True}, {"run": "aa", "error": "the connection closed before the watermark's end"}]
    feed.write_text("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in lines))
    assert S.runs_ended(feed) == 0                      # a watermark that broke is no run's end: the side holds on
    with feed.open("a") as f:
        f.write('{"run":"bb","t":0,"n":0}\n{"run":"bb","end":true}\n')
    assert S.runs_ended(feed) == 1
    assert S.hold_ends(None, None, 0, None) is None and S.hold_ends(None, None, 0, S.STOP_S + 1) is None
    assert S.hold_ends(None, None, 1, 3600.0) == ("the network certifier's run ended", True)
    assert S.hold_ends(None, None, 0, S.STOP_S) == ("--stop-at", True)
    assert S.hold_ends(1, None, 1, None) == ("the server exited with 1 (server/server.log)", False)
    assert S.hold_ends(None, 0, 1, None) == ("the id front exited with 0 (server/front.log)", False)


def line(index: int, outcome: str, ids: int | None, max_tokens: int = 8) -> dict:
    """A connection's record as the shim writes it (`cell_shim.HttpShim`), less what nothing here reads."""
    return {"index": index, "id": f"chat@2-r1-{index}", "max_tokens": max_tokens, "ordinal": index, "outcome": outcome,
            "kinds": "", "arrivals_ms": [], "response": {"completion_tokens": ids}}


def test_the_certifiers_record_says_whether_its_run_is_accepted_and_every_answer_whole():
    lines = [line(0, "end", 8), line(1, "end", 5), line(2, "refused", None), line(3, "fail", 2), line(4, "cut", 8),
             {"index": 5, "id": "x", "status": 400, "unsent": "an id request has no stop"}]
    check = {"accepted": False, "problems": [f"connection {k}" for k in range(9)],
             "summary": {"certified_windows": 6, "withheld_windows": [1], "cells_in_their_bucket": [40, 41],
                         "tick_lateness_ms": {"max": 3.5}}}
    assert S.certifier_record(check, lines) == {
        "accepted": False, "problems": [f"connection {k}" for k in range(5)], "connections": 5, "unsent": 1, "ended": 2,
        "refused": 1, "failed": 2, "short": 1, "certified_windows": 6, "withheld_windows": [1],
        "cells_in_their_bucket": [40, 41], "tick_lateness_ms": {"max": 3.5}}
    none = S.certifier_record(None, lines[:1])
    assert (none["accepted"], none["problems"], none["ended"], none["short"]) == (False, [], 1, 0)


def test_an_arm_behind_the_certifier_is_served_by_its_two_sides_and_by_vllms_client(tmp_path: Path,
                                                                                    monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    run = ["run", "--arm", ARM, "--model-id", "glm53", "--out", str(tmp_path / "out")]
    bench = [*run, "--client", "vllm-bench", "--loads", "chat=96"]
    for argv, why in (
            ([*run, "--c4", "c4", "--rates", "light=1"], "sent by vLLM's client, through the network certifier"),
            (bench, "served by two commands: --side server on the node"),
            ([*bench, "--side", "client"], "--certifier-windows N"),
            ([*bench, "--side", "client", "--sampled"], "is served greedy"),
            ([*run, "--client", "vllm-bench", "--loads", "long", "--side", "client"], "not long"),
            ([*run, "--client", "vllm-bench", "--loads", "agentic", "--side", "client"], "not agentic"),
            ([*bench, "--side", "server", "--cells", "queue=8", "--tick-ms", "220,220"], "a queue of 8 under the rung 96")):
        with pytest.raises(SystemExit, match=why):
            S.main(argv)
    assert not (tmp_path / "out").exists()
    # before a GPU is held the arm's passes are checked as another's, where a tick arm of the corpus's client is refused
    with pytest.raises(SystemExit, match="not long"):
        S.main(["passes", "--loads", "chat,long", "--arms", ARM, "--model-id", "glm53"])
    with pytest.raises(SystemExit, match="load chat: no dataset file"):
        S.main(["passes", "--loads", "chat=96", "--arms", ARM, "--model-id", "glm53"])


def test_a_job_of_the_nodes_side_serves_stock_whole_first_and_has_no_cost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """One job on the node serves stock and then the arm's side under one lease: the side is passed to every arm, an
    arm one command serves takes no notice of it, and the job computes no cost, the arm's passes being in the other
    side's directory."""
    sh = (S.HERE / "run.sh").read_text()
    assert '${SIDE:+--side "$SIDE"} ${FRONT:+--front "$FRONT"} ${CELLS:+--cells "$CELLS"} \\\n' in sh
    assert '&& [ "${ARMS##* }" != prod ] && [ "${SIDE:-}" != server ]; then' in sh
    served = []
    monkeypatch.setattr(S, "serve", lambda a, private: served.append((a.arm, a.side)) or 0)
    S.main(["run", "--arm", "compiled", "--model-id", "glm53", "--client", "vllm-bench", "--loads", "chat=96", "--side",
            "server", "--out", str(tmp_path / "out")])
    assert served == [("compiled", "server")]


class Proc:
    """A process a side starts, standing in: it has said where it listens, and ends when it is waited for."""

    def __init__(self, listening: dict, events: list, name: str) -> None:
        self.stdout = NS(readline=lambda: json.dumps(listening) + "\n")
        self.events, self.name, self.returncode = events, name, None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.events.append(("ended", self.name, timeout))
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.events.append(("terminated", self.name))
        self.returncode = -15


COUNTERS = dict.fromkeys(S.SCHEDULE, 0.0)


@pytest.mark.parametrize(("ends", "exits"), [("certifier", 0), ("stop-at", 0), ("front", 1)])
def test_the_servers_side_starts_the_engine_then_the_id_front_and_holds_them(
        ends: str, exits: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """From the engine's start to the side's record: the engine under the arm's plugins and the certifier's bucket,
    the id front before it once it answers, both held while the certifier's run goes on, and stopped at its end or at
    --stop-at, the engine's counters read first. An id front that ends first fails the side."""
    (tmp_path / "tick_plugin.py").touch()
    monkeypatch.setattr(S, "TICK_PLUGIN", tmp_path / "tick_plugin.py")
    monkeypatch.setattr(S, "corpus", lambda: NS(TICK_BUCKET_MS=100, free_port=lambda: 8100))
    monkeypatch.setattr(S, "signal", NS(signal=lambda *a: None, SIGTERM=15))
    monkeypatch.setattr(S, "open_files", lambda: 4096)
    monkeypatch.setattr(S, "source_commit", lambda: "c0ffee")
    monkeypatch.setattr(S, "check_plugins", lambda env, arm: events.append(("plugins", arm.plugin)))
    monkeypatch.setattr(S, "wait_ready", lambda server, port, startup_s: events.append(("ready", port)))
    monkeypatch.setattr(S, "scrape", lambda port: events.append(("scrape", port)) or COUNTERS)
    monkeypatch.setattr(S, "stop", lambda server, grace_s: events.append(("stop", server.name, grace_s)) or "exited")
    out, t, events, procs = tmp_path / "out", [5000.0], [], {}
    feed = out / "server" / "tick-feed.jsonl"

    def popen(argv, stdout, stderr, env, **how):
        name = "front" if "ids-front" in argv else "server"
        if name == "server":
            assert how == {"start_new_session": True} and json.loads(env[S.PLUGIN_ENV])["idle_reset"] is True
            assert json.loads(env["VERITY_TICK"]) | {"bucket_us": 250000, "sat_us": 220000, "feed": str(feed)} == \
                json.loads(env["VERITY_TICK"])
            assert argv[argv.index("--port") + 1] == "8100"
        else:
            assert how == {"text": True} and str(S.ROOT / "adapters/vllm") in env["PYTHONPATH"].split(":")
            assert argv == S.front_argv(8100, 18750, 4087, out / "server" / "front.jsonl")
        events.append(("started", name))
        procs[name] = Proc({"listening": ["127.0.0.1", 18750]}, events, name)
        return procs[name]

    def sleep(s: float) -> None:
        # the engine empties its prefix cache, the certifier's watermark goes on, and at the third look the hold ends
        t[0] += s
        events.append(("waited", s))
        if len([e for e in events if e[0] == "waited"]) == 1:
            (out / "server" / "prefix_resets.7.jsonl").write_text('{"t":5001.0,"idle_s":null,"reset":true}\n')
            feed.write_text('{"run":"aa","t":0,"n":0}\n')
        elif ends == "certifier":
            with feed.open("a") as f:
                f.write('{"run":"aa","end":true}\n')
        elif ends == "front":
            procs["front"].returncode = 1

    a = argparse.Namespace(
        arm=ARM, model_id="glm53", checkpoint="/ck", client="vllm-bench", side="server", loads="chat=96", cells=None,
        front=18750, tick_from=None, tick_ms="220,220", stop_at="1970-01-01T01:25:00Z" if ends == "stop-at" else None,
        num_gpu_blocks=None, max_model_len=None, window_s=30.0, kv_window_s=30.0, startup_s=1800.0, lifetime=None,
        out=str(out))
    rc = S.certified_server(a, popen=popen, sleep=sleep, now=lambda: t[0], clock=lambda: t[0])
    said = capsys.readouterr().out
    # --stop-at is 100 s off, and the side stops `STOP_S` before it
    held = [("waited", 2.0)] * (round((100 - S.STOP_S) / 2) if ends == "stop-at" else 2)
    assert rc == exits
    assert events == [("plugins", S.ARMS[ARM].plugin), ("started", "server"), ("ready", 8100), ("started", "front"),
                      *held, *([] if ends == "front" else [("scrape", 8100)]),
                      *([] if ends == "front" else [("terminated", "front"), ("ended", "front", None)]),
                      ("stop", "server", 120.0)]
    rec = json.loads((out / S.SERVER_RECORD).read_text())
    assert (rec["arm"], rec["side"], rec["cells"]["lanes"], rec["tick"]["bucket_us"]) == (ARM, "server", 96, 250000)
    assert rec["stopped"] == {"certifier": "the network certifier's run ended", "stop-at": "--stop-at",
                              "front": "the id front exited with 1 (server/front.log)"}[ends]
    assert rec["prefix_resets"] == [{"t": 5001.0, "idle_s": None, "reset": True}] and rec["server_stop"] == "exited"
    assert ("error" in rec, "schedule" in rec) == (ends == "front", ends != "front")
    assert "the id front listens on loopback port 18750" in said and f"stopping: {rec['stopped']}" in said
    # the side's directory is no arm's: no table and no cost reads it
    assert not (out / "meta.json").exists() and not (out / ".site").exists()


def result(sends: int, tok: float, duration: float = 300.0) -> dict:
    """A pass's result as vLLM's client saves it, less what nothing here reads."""
    return {"completed": sends, "failed": 0, "duration": duration, "total_input_tokens": 30 * sends,
            "total_output_tokens": int(tok * duration), "request_throughput": sends / duration, "request_goodput": 0.0,
            "output_throughput": tok, "total_token_throughput": tok + 30 * sends / duration, "median_ttft_ms": 15250.0,
            "p99_ttft_ms": 15500.0, "median_tpot_ms": 250.0, "p99_tpot_ms": 250.0, "median_itl_ms": 250.0,
            "p99_itl_ms": 250.0}


@pytest.mark.parametrize(("windows", "short", "exits"), [(200, 0, 0), (200, 1, 1), (130, 0, 1)])
def test_the_clients_side_sends_each_pass_through_the_shim_and_records_what_the_certifier_found(
        windows: int, short: int, exits: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture):
    """From the certifier's start to its record: the certifier before the id front and the shim to it are started in
    that order, each pass is sent at the shim after the engine's idle time, with the engine's counters around it, a
    pass that would outlast the certifier's windows is not started, and once the certifier has ended its check reads
    the shim's record of the connections. The side exits 0 only where every pass was sent and no answer is short of
    its length."""
    data = b'[{"id": "a", "conversations": [{"value": "hello"}, {"value": "hi there"}]}]'
    ds = S.Dataset("org/set", "f" * 40, "set.json", len(data), hashlib.sha512(data).hexdigest())
    monkeypatch.setitem(S.LOADS, "chat", S.replace(S.LOADS["chat"], file=ds))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    ds.path().parent.mkdir(parents=True)
    ds.path().write_bytes(data)
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(S.tempfile, "tempdir", str(tmp_path / "tmp"))
    monkeypatch.setattr(S, "source_commit", lambda: "c0ffee")
    monkeypatch.setattr(S, "served_models", lambda front: ["glm-5.3"])
    out, t, events, scraped = tmp_path / "out", [1000.0], [], [0]

    def scrape(port: int) -> dict:
        scraped[0] += 1
        events.append(("scrape", port))
        return COUNTERS | {"prompt_tokens": 100.0 * scraped[0], "prompt_tokens_cached": 1.0 * scraped[0]}

    monkeypatch.setattr(S, "scrape", scrape)

    def popen(argv, stdout, stderr, env, text):
        assert str(S.ROOT) in env["PYTHONPATH"].split(":") and text
        if argv[1] == str(S.CERTIFIER):
            (out / "certifier" / "deployment.json").write_text(json.dumps({"origin_unix": 1005.0}))
            events.append(("certifier", int(argv[argv.index("--windows") + 1]), Path(argv[argv.index("--key") + 1])))
            return Proc({"listening": ["127.0.0.1", 4000]}, events, "certifier")
        assert argv[1] == str(S.SHIM) and argv[argv.index("--deployment") + 1] == str(out / "certifier" / "deployment.json")
        events.append(("shim", argv[argv.index("--log") + 1]))
        return Proc({"listening": ["127.0.0.1", 5000]}, events, "shim")

    def run(argv, stdout, stderr=None, env=None):
        if argv[1] == str(S.CERTIFIER):
            events.append((argv[2],))
            if argv[2] == "check":
                assert argv[3:] == ["--run", str(out / "certifier")]
                sent = json.loads((out / "certifier" / "client.json").read_text())["connections"]
                (out / "certifier" / "check.json").write_text(json.dumps(
                    {"accepted": True, "problems": [], "summary": {"certified_windows": 9, "requests": len(sent)}}))
            return NS(returncode=0)
        sends = int(argv[argv.index("--num-prompts") + 1])
        events.append(("pass", argv[argv.index("--base-url") + 1], argv[argv.index("--max-concurrency") + 1], sends, t[0]))
        d = Path(argv[argv.index("--result-dir") + 1])
        (d / argv[argv.index("--result-filename") + 1]).write_text(json.dumps(result(sends, 6.0)))
        with (out / "shim.jsonl").open("a") as f:
            for k in range(sends):
                f.write(json.dumps(line(k, "end", 8 - (short and k == 1))) + "\n")
        t[0] += 300.0
        return NS(returncode=0)

    def sleep(s: float) -> None:
        events.append(("waited", s))
        t[0] += s

    a = argparse.Namespace(
        arm=ARM, model_id="glm53", checkpoint=str(tmp_path / "ck"), client=S.BENCH, loads="chat=2", dataset=None,
        pass_prompts="chat=3", reps=3, seed=0, warm_s=20.0, front=18746, cells="window=40", certifier_windows=windows,
        lifetime=None, leave_out=0, out=str(out), side=S.CLIENT)
    rc = S.certified_client(a, run=run, popen=popen, sleep=sleep, now=lambda: t[0], clock=lambda: t[0])
    said = capsys.readouterr().out
    # the certifier runs windows of 10 s from its origin, and a pass takes 300 s: of 130 windows the warm-up and two
    # reps fit and the third is not sent, of 200 all three are
    reps, end = (2, 2305.0) if windows == 130 else (3, 3005.0)
    key = events[1][2]
    assert events[:4] == [("keys",), ("certifier", windows, key), ("shim", str(out / "shim.jsonl")), ("waited", 5.0)]
    assert not key.exists() and not key.is_relative_to(out)
    sent = [x for k in range(reps + 1) for x in (
        ("waited", S.IDLE_S), ("scrape", 18746), ("pass", "http://127.0.0.1:5000", "2", 3, 1011.0 + 306.0 * k),
        ("scrape", 18746))]
    last = 1011.0 + 306.0 * reps + 300.0
    assert events[4:] == [*sent, ("ended", "certifier", end - last + S.CERTIFIER_END_S), ("check",),
                          ("terminated", "shim"), ("ended", "shim", None)]
    meta = json.loads((out / "meta.json").read_text())
    assert (meta["arm"], meta["parent"], meta["side"], meta["client"]) == (ARM, "tick", "client", "vllm-bench")
    assert meta["passes"] == {"chat@2": {"load": "chat", "concurrency": 2}} and meta["pass_prompts"] == {"chat": 3}
    assert [(x["load"], x["rep"], x["requests"], x["wall_s"]) for x in meta["loads"]] == [
        ("chat@2", rep, 3, 300.0) for rep in range(1, reps + 1)]
    if reps == 2:
        assert meta["stopped"] == {"before": "chat@2", "rep": 3, "need_s": (300.0 + S.LOAD_OVERHEAD_S) * 1.1 + S.STOP_S}
        assert "stopped before chat@2 rep 3" in said
    else:
        assert "stopped" not in meta and "stopped before" not in said
    k = meta["certifier"]
    assert k["cells"] == {**S.CELLS, "window": 40, "lanes": 2, "queue": 2} and k["ceiling_tok_per_s"] == 8.0
    n = 3 * (reps + 1)
    assert (k["windows"], k["origin_unix"], k["end_unix"], k["exit"], k["check_exit"]) == (windows, 1005.0, end, 0, 0)
    assert (k["accepted"], k["connections"], k["ended"], k["refused"], k["failed"]) == (True, n, n, 0, 0)
    assert k["short"] == (reps + 1) * short
    assert f"{n} of {n} connections ended, 0 refused, 0 failed, {k['short']} answers short of their length" in said
    assert len(json.loads((out / "certifier" / "client.json").read_text())["connections"]) == n
    rep = json.loads((out / "chat@2" / "rep-2" / "meta.json").read_text())
    assert (rep["load"], rep["rep"], rep["concurrency"], rep["requests"], rep["unfinished"]) == ("chat@2", 2, 2, 3, 0)
    assert rep["schedule"]["prompt_tokens"] == 100.0 and rep["schedule"]["prompt_tokens_cached"] == 1.0
    assert json.loads((out / "warm" / "meta.json").read_text())["schedule"]["prompt_tokens"] == 100.0
    assert rc == exits
    # the benchmark's command reads the directory as an arm's: its cost over stock at the same rung and prompts has
    # no verdict against the target, and is said to be the schedule's and the engine's together
    stock = tmp_path / "compiled"
    (stock / "chat@2" / "rep-1").mkdir(parents=True)
    (stock / "meta.json").write_text(json.dumps({"arm": "compiled", "model_id": "glm53", "passes": meta["passes"]}))
    (stock / "chat@2" / "rep-1" / "result.json").write_text(json.dumps(result(3, 9.0, 200.0)))
    (stock / "chat@2" / "rep-1" / "meta.json").write_text(json.dumps({"rate": None, "concurrency": 2, "unfinished": 0}))
    r = S.cost_report([S.arm_metrics(stock), S.arm_metrics(out)], ARM)
    (chat,) = r["loads"]
    assert (chat["load"], chat["rung"], chat["reps"], chat["value"], chat["verdict"]) == ("chat", 2, 1, 50.0, S.NOT_COMPARABLE)
    assert (r["value"], r["verdict"]) == (None, S.NOT_COMPARABLE)
    assert r["not_in_target"] == ["the step record", "the network certifier in front"] and r["certifier"] == k
    text = S.cost_text(r)
    assert f"{ARM} is served through the network certifier" in text and "is not GPU time a token" in text
    assert "its 2 lanes carry at most 8 output tokens a second whatever its engine does" in text
    assert "turns on the step record and the network certifier in front, which the target has no part for" in text


def test_the_cost_of_an_arm_behind_the_certifier_says_what_it_is_of():
    """The report's text never calls such an arm's cost GPU time a token: it names the schedule's ceiling on its
    tokens a second, and what the certifier's check found of the run."""
    cells = S.certifier_cells(None, {"chat": (96,)})
    k = {"cells": cells, "windows": 16, "ceiling_tok_per_s": S.ceiling(cells)}
    assert S.certifier_text(ARM, None) == []
    first, second = S.certifier_text(ARM, k)
    assert ("lets one id leave a connection each 250 ms, the first 15 s after its admission: its 96 lanes carry at "
            "most 384 output tokens a second whatever its engine does") in first
    assert "is not GPU time a token" in first and "has no check: the arm stopped before it" in second
    found = S.certifier_record({"accepted": True, "problems": [], "summary": {}}, [line(k, "end", 8) for k in range(1152)])
    assert S.certifier_text(ARM, k | found)[1] == (
        "    the network certifier's run is accepted by its offline check: 1,152 of 1,152 connections ended, 0 "
        "refused, 0 failed, 0 answers short of their length")
    assert "NOT accepted" in S.certifier_text(ARM, k | found | {"accepted": False})[1]
