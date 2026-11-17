"""The release front of ``prod-all-on`` (`release_front`) before a stand-in server on loopback, on a hand-turned clock."""
from __future__ import annotations

import asyncio
import threading

import release_front as RF
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


def test_a_front_on_its_thread_stops_at_once_when_nothing_was_released():
    async def upstream() -> tuple[asyncio.AbstractServer, Upstream]:
        up = Upstream()
        return await asyncio.start_server(up.connection, "127.0.0.1", 0), up

    loop = asyncio.new_event_loop()
    server, up = loop.run_until_complete(upstream())
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        front = RF.Running(server.sockets[0].getsockname()[1], 0.1)
        assert front.port > 0
        summary = front.stop()
        assert (summary["run"], summary["released"], summary["watermark"]) == (front.front.stand.run, 0, 200)
        assert up.ended.is_set()
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()
