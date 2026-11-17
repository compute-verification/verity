"""The release front of ``prod-all-on`` (`release_front`) before a stand-in server on loopback, on a hand-turned clock."""
from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import release_front as RF
import serve_tax as S
from verity_vllm.pipeline.serve_admit import BUCKET_HEADER, RELEASE_HEADER, RUN_HEADER, WATERMARK_PATH


class Upstream:
    """A server that keeps each request's head and the watermark's lines, and answers ``ok``."""

    def __init__(self) -> None:
        self.heads: list[dict] = []
        self.lines: list[bytes] = []
        self.line, self.ended = asyncio.Event(), asyncio.Event()

    async def connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        first, *fields = (await reader.readuntil(b"\r\n\r\n")).decode().rstrip("\r\n").split("\r\n")
        headers = {k.strip().lower(): v.strip() for k, v in (f.split(":", 1) for f in fields)}
        self.heads.append({"line": first, **headers})
        if first.split()[1] == WATERMARK_PATH:
            while (size := int(await reader.readline(), 16)) > 0:
                self.lines.append((await reader.readexactly(size + 2))[:-2])
                self.line.set()
            await reader.readline()
            self.ended.set()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
        else:
            await reader.readexactly(int(headers.get("content-length", 0)))
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\nKeep-Alive: timeout=5\r\n\r\nok")
        await writer.drain()
        writer.close()


class Time:
    """The front's clock, which moves a bucket only as the test lets its watermark's sleep return."""

    def __init__(self) -> None:
        self.now = 0.0
        self.turns = asyncio.Semaphore(0)

    async def sleep(self, dt: float) -> None:
        await self.turns.acquire()
        self.now += dt

    def turn(self, k: int = 1) -> None:
        for _ in range(k):
            self.turns.release()


async def request(port: int, line: str, body: bytes = b"", extra: str = "") -> tuple[str, dict, bytes]:
    r, w = await asyncio.open_connection("127.0.0.1", port)
    w.write(f"{line} HTTP/1.1\r\nHost: x\r\nConnection: keep-alive\r\n{extra}Content-Length: {len(body)}\r\n\r\n".encode()
            + body)
    first, *fields = (await r.readuntil(b"\r\n\r\n")).decode().rstrip("\r\n").split("\r\n")
    rest = await r.read()
    w.close()
    return first, {k.strip().lower(): v.strip() for k, v in (f.split(":", 1) for f in fields)}, rest


def test_the_front_numbers_completions_in_arrival_order_and_streams_their_buckets_lines():
    async def main() -> tuple[Upstream, dict, list]:
        up, t = Upstream(), Time()
        server = await asyncio.start_server(up.connection, "127.0.0.1", 0)
        front = RF.ReleaseFront(server.sockets[0].getsockname()[1], 0.1, clock=lambda: t.now, sleep=t.sleep)
        stop, port = asyncio.Event(), asyncio.get_running_loop().create_future()
        serving = asyncio.create_task(front.serve(port.set_result, stop))
        port = await port
        answers = [await request(port, "POST /v1/completions", b"{}", f"{RELEASE_HEADER}: 99\r\n"),
                   await request(port, "GET /metrics")]
        t.turn()
        await up.line.wait()
        answers.append(await request(port, "POST /v1/chat/completions?x=1", b"{}"))
        stop.set()
        t.turn(2)
        summary = await serving
        server.close()
        return up, summary, answers

    up, summary, answers = asyncio.run(main())
    mark, *heads = up.heads
    run = summary["run"]
    assert mark["line"] == f"POST {WATERMARK_PATH} HTTP/1.1" and mark[RUN_HEADER] == run
    assert [(h["line"], h.get(RELEASE_HEADER), h.get(RUN_HEADER), h.get(BUCKET_HEADER), h["connection"]) for h in heads] == [
        ("POST /v1/completions HTTP/1.1", "0", run, "0", "close"),
        ("GET /metrics HTTP/1.1", None, None, None, "close"),
        ("POST /v1/chat/completions?x=1 HTTP/1.1", "1", run, "1", "close")]
    assert up.lines == [b"0 1\n", b"1 2\n"]                  # each line counts the releases of its buckets
    assert all(first == "HTTP/1.1 200 OK" and h["connection"] == "close" and "keep-alive" not in h and rest == b"ok"
               for first, h, rest in answers)
    assert summary == {"run": run, "released": 2, "lines": 2, "watermark": 200}


@contextlib.contextmanager
def upstream() -> Iterator[tuple[int, Upstream]]:
    """An `Upstream` on a loop of its own thread: its port, and it."""
    async def start() -> tuple[asyncio.AbstractServer, Upstream]:
        up = Upstream()
        return await asyncio.start_server(up.connection, "127.0.0.1", 0), up

    loop = asyncio.new_event_loop()
    server, up = loop.run_until_complete(start())
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        yield server.sockets[0].getsockname()[1], up
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()


def test_a_front_on_its_thread_stops_at_once_when_nothing_was_released():
    with upstream() as (port, up):
        front = RF.Running(port, 0.1)
        assert front.port > 0
        summary = front.stop()
        assert (summary["run"], summary["released"], summary["watermark"]) == (front.front.stand.run, 0, 200)
        assert up.ended.is_set()


def test_prod_all_on_is_sent_through_its_release_front_and_records_what_a_price_counts(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """vLLM's client sends prod-all-on's passes to its release front, which numbers them before the server; each pass
    waits for the engine to empty its prefix cache and never asks the server to, and records what the tick's gate and
    each canonical top-k process counted over it; the arm's record says what it has on and what it leaves off."""
    (tmp_path / "tick_plugin.py").touch()
    monkeypatch.setattr(S, "TICK_PLUGIN", tmp_path / "tick_plugin.py")
    out, envs, k = tmp_path / "out", [], [0]
    srv = out / "server"
    for name, stub in {"signal": NS(signal=lambda *a: None, SIGTERM=15), "open_files": lambda: 4096,
                       "source_commit": lambda: "c0ffee", "check_plugins": lambda env, arm: None,
                       "wait_ready": lambda server, port, startup_s: None, "nvml": lambda: (None, "no NVML here"),
                       "scrape": lambda port: dict.fromkeys(S.SCHEDULE, 0.0), "versions": lambda: {},
                       "stop": lambda server, grace_s: "exited", "end_of_text_ids": lambda ck: [1], "IDLE_S": 0.0}.items():
        monkeypatch.setattr(S, name, stub)

    def popen(argv, stdout, stderr, env, **how):
        envs.append(env)
        topk = Path(env["VERITY_DSA_TOPK_LOG"])
        topk.mkdir(parents=True)
        (topk / "dsa_topk.11.json").write_text(json.dumps({"calls": {"a": 5, "b": 0}, "rows": {}}))
        return NS(returncode=None, poll=lambda: None)

    def drive(sends_to, model, checkpoint, load, c, tag, d, *_args):
        k[0] += 1
        d.mkdir(parents=True)
        topk = Path(envs[0]["VERITY_DSA_TOPK_LOG"]) / "dsa_topk.11.json"
        topk.write_text(json.dumps({"calls": {"a": 5 + 30 * k[0], "b": 10 * k[0]}, "rows": {}}))
        with (srv / "tick.12.jsonl").open("a") as f:
            f.write(json.dumps({"step": 3 * k[0], "T": 1, "bucket": 0, "admitted": [k[0] - 1, k[0]],
                                "held_steps": [k[0] - 1], "t": 1.0}) + "\n")
            f.write(json.dumps({"event": "late", "run": "r", "key": "x", "n": 0, "t": 1.0}) + "\n" if k[0] == 2 else "")
        with (srv / "prefix_resets.12.jsonl").open("a") as f:
            f.write(json.dumps({"t": 1.0, "idle_s": None, "reset": True}) + "\n")
        req = urllib.request.Request(f"http://127.0.0.1:{sends_to}/v1/completions", data=b"{}", method="POST")
        with urllib.request.urlopen(req) as r:  # noqa: S310
            assert r.read() == b"ok"
        return {"requests": 1, "unfinished": 0, "unserved": 0, "concurrency": 1, "rate": None, "wall_s": 1.0,
                "t0_unix": 0.0}

    monkeypatch.setattr(S.subprocess, "Popen", popen)
    monkeypatch.setattr(S, "drive_bench", drive)
    with upstream() as (port, up):
        monkeypatch.setattr(S, "corpus", lambda: NS(TICK_BUCKET_MS=100, free_port=lambda: port))
        assert S.main(["run", "--arm", "prod-all-on", "--model-id", "glm53", "--checkpoint", str(tmp_path / "ck"),
                       "--client", "vllm-bench", "--loads", "long=1", "--reps", "2", "--warm-s", "0", "--tick-ms",
                       "34.7,42.4", "--edges-every-s", "60", "--out", str(out)]) == 0
    tick, cfg = json.loads(envs[0]["VERITY_TICK"]), json.loads(envs[0][S.PLUGIN_ENV])
    assert (tick["bucket_us"], tick["sat_us"], tick["free_us"]) == (100000, 34700, 42400)
    assert (cfg["record"], cfg["ids"], cfg["idle_reset"], cfg["every_s"], cfg["kv_copy"]) == (True, True, True, 60.0, "all")
    mark, *heads = up.heads
    run = mark[RUN_HEADER]
    assert [(h["line"], h[RELEASE_HEADER], h[RUN_HEADER]) for h in heads] == [
        ("POST /v1/completions HTTP/1.1", "0", run), ("POST /v1/completions HTTP/1.1", "1", run)]
    meta = json.loads((out / "meta.json").read_text())
    assert meta["parts"] == [
        "the step record: on", "the served ids: on", "the canonical top-k: on", "admission on the certified tick: on",
        "the edge snapshot: on, every 60 s against the cost model's 600 s, so its cost errs high by 10×",
        "the network attester's egress pacing: off", "useful-work kernels: off", "commitment of values: off"]
    assert meta["release_front"] == {"run": run, "released": 2, "lines": meta["release_front"]["lines"],
                                     "watermark": 200}
    reps = [json.loads((out / "long@1" / f"rep-{r}" / "meta.json").read_text()) for r in (1, 2)]
    assert [m["prefix_cache_reset"] for m in reps] == [True, True]
    assert [m["counts"] for m in reps] == [
        {"tick": {"admitted": 1, "deferred": 0, "held_steps": {"0": 1}, "admitting_steps": 1, "events": {}},
         "topk_calls": {"11": 40}},
        {"tick": {"admitted": 1, "deferred": 1, "held_steps": {"1": 1}, "admitting_steps": 1, "events": {"late": 1}},
         "topk_calls": {"11": 40}}]
