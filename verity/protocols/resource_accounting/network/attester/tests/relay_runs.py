"""A network attester run driven tick by tick on a `VirtualClock` and published as `serve` and `client` publish one
(`_run`), with the text schedule it runs at by default and its stand-ins for a connection's writer and an endpoint's
response; and the gate's relay session at the scan's w63k4 schedule (`_session_argv`), which the challenger's tests
hold the gate's command to. The device's tests and those of the programs that read a run import it by its package
path."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from verity.protocols.resource_accounting.network.attester import runner_parts as rp, network_certifier_serve as ws
from verity.protocols.resource_accounting.network.verifier import relay_check as RC, network_certifier_rows as WR
from verity.protocols.resource_accounting.network.verifier.link_frame import DATA
from verity.protocols.resource_accounting.network.verifier.run_clock import VirtualClock
from verity.protocols.resource_accounting.network.verifier.network_certifier import (END, INTEGER_FIELDS, REFUSED, W63K4, Shim, NetworkCertifier,
                                                    NetworkCertifierParams, cells, read_cell)

#: a text rehearsal's schedule: its responses are vLLM's stock framing, so its egress form is bytes
PARAMS = NetworkCertifierParams(cell_bytes=64, slots=2, bucket_ms=100, delay=3, lanes=2, queue=4, window=60, ingress_cells=40,
                      ingress_buckets=30, egress_cells=128, egress="bytes")
SESSION = bytes(range(32))
#: the clocks `slow_commit` moves: a run's while it ticks in this process; a spawned process's list stays empty
PROCESS_CLOCK: list[VirtualClock] = []
#: what a window's commit takes of its process's time: the trace corpus's small-KV run measured 378-392 ms
SLOW_COMMIT_S = 0.4


class _Writer:
    def __init__(self, network_certifier: NetworkCertifier | None = None) -> None:
        self.network_certifier, self.writes, self.closed, self.transport = network_certifier, [], False, self

    def write(self, data: bytes) -> None:
        self.writes.append((self.network_certifier.ticks if self.network_certifier else None, bytes(data)))

    def can_write_eof(self) -> bool:
        return False

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
         process: bool = False, servers: list | None = None, params: NetworkCertifierParams = PARAMS,
         id_payload: int | None = None, load_heads: Path | None = None, arrive: dict[int, int] | None = None,
         late: dict[int, float] | None = None, back: dict[int, int | None] | None = None) -> NetworkCertifier:
    """A run of the network attester at `params` for `windows` windows on a `VirtualClock`, published as `serve` and
    `client` publish one: its records and certificates through `Publisher` inline (`serve`'s `PublisherProcess` when
    `process`, which first signs the load's heads `load_heads` when given), the session's start certified at base 0
    before any record, each window committed by `commit`, its connection rows kept as the publisher process keeps
    them when the schedule has lengths rows (its key then a height taller, for their heads), `deployment.json`, and
    `client.json` with every cell's arrival 1 ms into the bucket it was sent in, each response read id-form at P =
    `id_payload` when given; `servers` gets the server's side of each connection, in connect order. In a schedule
    with a head offset the publisher hands each execution phase's line of connection rows back and the network
    attester's tick writes it, as in `serve`: the line is back at the hand itself when published inline and before
    the next tick when by the process, or, inline, just before the tick `back` names for its execution phase, never
    where that is None. The run then ticks on past its last window while a line may still leave. Returns the
    network attester."""
    tau = params.bucket_ms / 1000
    height = 5 if load_heads or params.lengths else 4
    timed = params.head_offset is not None
    assert not back or (timed and not process)
    lines = rp.Lines(out, ws.LINK) if timed else None
    held: dict[int, dict] = {}
    device: list[NetworkCertifier] = []

    def taken(window: int, line: dict) -> None:
        if window in (back or {}):
            held[window] = line
        else:
            device[0].head_back(window, line)

    if process:
        from verity.protocols.resource_accounting.network.attester import network_certifier_serve  # by name, so that the spawned process finds its functions

        ws.ck.load_key(key, height)
        publisher = network_certifier_serve.PublisherProcess(out, key, SESSION, params, commit, load_heads=load_heads, base_ms=0,
                                                  ordinal=0)
        hooks = {"publish": publisher, "withhold": publisher.withhold, "release": publisher.release,
                 "lengths": publisher.lengths}
        public, start = publisher.key, publisher.start
    else:
        signer = ws.ck.signer(key, height)
        issuer = ws.Issuer(signer, SESSION)
        publisher = ws.Publisher(out, issuer, params, commit or WR.Cells.of)
        keeper = rp.Keeper(out, issuer, params, RC.LINK, back=taken if timed else None)
        hooks, public = {"publish": publisher, "withhold": publisher.withhold, "lengths": keeper.keep}, \
            signer.public_key()
        start = ws.start_json(ws.certify_start(issuer, 0, 0, params))
    answers = iter(responses)

    async def connect():
        reader = asyncio.StreamReader()
        reader.feed_data(next(answers))
        reader.feed_eof()
        writer = _Writer()
        if servers is not None:
            servers.append(writer)
        return reader, writer

    def settled(w: NetworkCertifier) -> bool:
        return all(c.server is not None and c.ingress_ended and c.egress_ended for c in w.active)

    async def main() -> tuple[list[_Writer], NetworkCertifier]:
        clock = VirtualClock(-tau / 2)
        network_certifier = NetworkCertifier(params, connect, origin=0.0, windows=windows, clock=clock, sleep=clock.sleep, leave=lines,
                        **hooks)
        device.append(network_certifier)
        over: list[bool] = []
        if process and timed:
            loop = asyncio.get_running_loop()

            def on_loop(call, *args) -> None:
                # the process's last word comes at `close`, when the run is over and its loop closed
                if not loop.is_closed():
                    loop.call_soon_threadsafe(call, *args)

            # the process's end wakes a wait for a line that then never comes, and `close` raises its failure
            publisher.heads(lambda window, line: on_loop(taken, window, line),
                            ended=lambda: on_loop(lambda: (over.append(True), network_certifier._changed.set())))
        running = asyncio.create_task(network_certifier.run())
        running.add_done_callback(lambda _: network_certifier._changed.set())
        clients, tasks = [], []

        async def tick(t: int) -> None:
            # the lines the process was handed are back before tick t runs, and `back`'s are handed back
            if process and timed:
                await network_certifier.until(lambda w: over or not w.awaited)
            for window, at in (back or {}).items():
                if at == t:
                    network_certifier.head_back(window, held.pop(window))
            # tick t runs `late[t]` seconds past its time
            clock.advance(t * tau + (late or {}).get(t, 0))
            await network_certifier.until(lambda w: w.ticks > t or running.done())

        async def connect_due(t: int) -> None:
            # request j connects before the run's first tick, or just before tick `arrive[j]`: a later request no
            # earlier, so that the clients are in the requests' order
            for j, request in enumerate(requests):
                if (arrive or {}).get(j, 0) == t:
                    reader = asyncio.StreamReader()
                    reader.feed_data(cells(request, params.cell_bytes))
                    reader.feed_eof()
                    clients.append(_Writer(network_certifier))
                    tasks.append(asyncio.create_task(network_certifier.handle(reader, clients[-1])))
            await network_certifier.until(lambda w: w.ordinals + w.refused + w.turned_away == len(clients))

        await connect_due(0)
        PROCESS_CLOCK.append(clock)
        try:
            for t in range(windows * params.window):
                if t:
                    await connect_due(t)
                await network_certifier.until(settled)
                await tick(t)
            t = windows * params.window
            while timed and not running.done():
                await tick(t)
                t += 1
            await running
        finally:
            PROCESS_CLOCK.remove(clock)
        await asyncio.gather(*tasks)
        return clients, network_certifier

    clients, network_certifier = asyncio.run(main())
    publisher.close()
    if lines is not None:
        lines.close()
    deployment = ws.deployment_record(params, public, SESSION, origin_unix=0.0, windows=windows, slack=network_certifier.slack,
                                      run_id=network_certifier.run_id, start=start, origin_raw=0.0)
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
    return network_certifier


def _session_argv(out: Path, windows: int, listen_fd: int | None = None) -> list[str]:
    """`session` for `windows` execution phases at the scan's w63k4 schedule, at height 5, as the gate's `relay-session`
    command runs it (the challenger's tests hold the gate's command to this one)."""
    return ["session", "--upstream", "127.0.0.1:40001", "--listen", "127.0.0.1:40002", "--out", str(out),
            "--height", "5", "--windows", str(windows),
            *(x for k in INTEGER_FIELDS for x in (f"--{k.replace('_', '-')}", str(getattr(W63K4, k)))),
            *(() if listen_fd is None else ("--listen-fd", str(listen_fd)))]
