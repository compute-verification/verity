"""served-cheats: a cheating server against the network certifier's record, four switches, each refused or charged.

One command runs each of four cheats and prints, per switch, what the audit was expected to do and what it did. The
cheats, each a switch:

- ``tokens``: the server puts ids of its own choosing in a response (the shim's ``token`` tamper, `active_replay`).
  The frame counts stay honest's, so only a check of the frames' tokens against the committed sampled ids (the I/O
  tie) refuses it. This harness runs N2's counts, which don't, and the switch says so.
- ``length``: the server ends a response late or early (the ``extra`` and ``drop`` tampers), which moves its event
  count. The switch reports what N2's counts say of each under ``--profile``. A check of where a response stops
  against the served computation's stop is not part of this harness.
- ``timing``: the server writes an interval's bytes in a burst around the network certifier's release time, to move
  when they leave (the jitter channel). The switch runs ``benchmarks/certifier_timing``'s burst attack against this
  checkout's certifier for ``--timing-intervals`` intervals and reports its figure, the bits an interval a decoder
  shows, and its ceiling against the budget (`channel.target`). It is a measurement of the host it runs on, on that
  host's clock, so it is run on a team machine; a run short of `channel.samples_to_show`'s intervals can't show a
  ceiling under the budget, and the row says so.
- ``failures``: the server fails connections on purpose. Only the network certifier as a relay (`warden`) has a
  failed connection on its record: a server that sends some of a response's cells and withholds the rest is failed
  late at the next egress cell, a `FAIL` cell in the egress record and its reason in the ``fails`` record. The switch
  runs the relay over the same responses with some connections failed (`FAILED`) and with none, its control, and
  reports the record, the relay's offline check (`warden_serve.check`) and what the leak bound's table charges the
  fail cells (`leak_bounds.terms`). The relay runs in its bytes egress form, a rehearsal's.

``--stand-in`` (the dry run) builds every record in process on an injected clock from responses of GLM-5.3's tokens,
so it needs no GPU and no served model; only the timing switch reads a wall clock. It is the only mode wired: no
switch reads a kept served record.

    python served_cheats.py --stand-in [--profile planned|served] [--switch all|tokens|length|timing|failures]
        [--timing-intervals N] [--out DIR]

It exits 0 when every control ran as an honest run does (the proxy's honest run accepted, the relay's run with no
failure accepted with none failed) and no switch errored. What a cheat's row says is a finding, not an exit code.
"""
from __future__ import annotations

import argparse
import asyncio
import codecs
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import active_replay as ar  # noqa: E402
import certifier_serve as cs  # noqa: E402
import leak_bounds as LB  # noqa: E402
import warden_serve as ws  # noqa: E402
from verity.protocols.certifier.verifier import active  # noqa: E402
from verity.protocols.certifier.verifier import warden_rows as WR  # noqa: E402
from verity.protocols.certifier.verifier.active import DATA, VirtualClock  # noqa: E402
from verity.protocols.certifier.verifier.certificate import Issuer  # noqa: E402
from verity.protocols.certifier.verifier.warden import (END, FAIL, REFUSED, Shim, Warden, WardenParams,  # noqa: E402
                                                    cells, read_cell)


def _channel():
    """`benchmarks/certifier_timing/channel.py`, the one statement of the timing budget (standard library only)."""
    spec = importlib.util.spec_from_file_location("certifier_timing_channel",
                                                  HERE.parent / "certifier_timing" / "channel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


channel = _channel()

SESSION = bytes(range(32))
REQUESTS, MERGED_TOKENS, PROMPT_TOKENS = 8, 64, 128
SWITCHES = ("tokens", "length", "timing", "failures")
#: what a run that can't complete raises (a verifier that can't be built or run, a refusal, a connection): its switch
#: reports it as its row's error and the others run on; anything else is this harness's bug and is raised
RUN_ERRORS = (RuntimeError, ValueError, OSError)
#: the relay's schedule for the failures switch: `test_warden_serve`'s text rehearsal schedule (the bytes egress form,
#: which relays any server's bytes in cells) with a lane and a place in the queue a connection: they arrive together,
#: before the first tick admits any, and a connection past the queue is refused, which leaves its window uncertified
RELAY = WardenParams(cell_bytes=64, slots=2, bucket_ms=100, delay=3, lanes=REQUESTS, queue=REQUESTS, window=60,
                     ingress_cells=40, ingress_buckets=30, egress_cells=128, egress="bytes")
#: the failures switch's cheat: the connections whose server withholds, by connect order, and how many whole cells of
#: its response each sends first, so the egress cell the relay fails it at
FAILED = {1: 0, 4: 5}

#: Tokens of GLM-5.3's tokenizer (zai-org/GLM-5.3 at aca966e4): words and punctuation of two and three bytes a
#: character, and tokens that split a character's UTF-8, as the stand-in streams them (`test_certifier_serve.py`)
GLM_TOKENS = ("数据".encode(), " 计算".encode(), "，".encode(),
              b"\xf0\x9f", b"\xa6", b"\x99",  # 🦙, split across three tokens
              "模型".encode(),
              b"\xe9\xbe", b"\x98",  # 龘, split across two
              "。".encode(),
              b" GL", b"M")


def glm_token(k: int) -> bytes:
    return GLM_TOKENS[k % len(GLM_TOKENS)]


# The proxy: the tokens and length switches, and the timing switch's plan.


class StandIn:
    """An OpenAI streaming completions endpoint that serves `group` requests side by side, as vLLM serves requests in
    flight, streaming GLM-5.3's tokens. ``/v1/models`` lists one model and ``/tokenize`` makes 200 tokens of any
    prompt. For the requests `merges` picks, the last two tokens share one chunk, as vLLM v1's collector merges deltas
    the API server hasn't read yet."""

    def __init__(self, group: int, *, merges=lambda r: False) -> None:
        self.group, self.arrived, self.merges = group, 0, merges
        self.turns: dict = defaultdict(asyncio.Event)

    async def __call__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        n = re.search(rb"content-length: (\d+)", head, re.I)
        body = json.loads(await reader.readexactly(int(n.group(1)))) if n else None
        if not head.startswith(b"POST /v1/completions "):
            doc = ({"data": [{"id": "stub-model"}]} if head.startswith(b"GET /v1/models ")
                   else {"tokens": list(range(200)), "count": 200, "prompt": body["prompt"], "model": body["model"]})
            data = json.dumps(doc).encode()
            writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: %d\r\n\r\n%s"
                         % (len(data), data))
            await writer.drain()
            writer.close()
            return
        me, self.arrived = self.arrived, self.arrived + 1
        g, i = divmod(me, self.group)
        if i == self.group - 1:
            self.turns[(g, 0, 0)].set()
        writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n")
        text, n_tokens, detokenize = "", body["max_tokens"], codecs.getincrementaldecoder("utf-8")()
        for k in range(n_tokens):
            await self.turns[(g, k, i)].wait()
            text += detokenize.decode(glm_token(k))
            if not (k == n_tokens - 2 and self.merges(me)):
                chunk = json.dumps({"choices": [{"text": text}]}, ensure_ascii=False, separators=(",", ":"))
                p, text = b"data: %s\n\n" % chunk.encode(), ""
                writer.write(b"%x\r\n%s\r\n" % (len(p), p))
                await writer.drain()
            self.turns[(g, k, i + 1) if i + 1 < self.group else (g, k + 1, 0)].set()
        writer.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(b"data: [DONE]\n\n"), b"data: [DONE]\n\n"))
        await writer.drain()
        writer.close()


def plan_args(profile: str) -> argparse.Namespace:
    """`certifier_serve.plan_of`'s arguments for the stand-in: four in flight, 64 tokens, D = 30 buckets."""
    return argparse.Namespace(model=None, api_key_env=None, prompt_tokens=PROMPT_TOKENS, tokens=MERGED_TOKENS,
                              response_frames=None, probes=4, in_flight=4, requests=REQUESTS, delta_buckets=30,
                              profile=profile, temperature=0.0, top_p=1.0)


def run_one(run: Path, plan: dict, report: dict, url: str, upstream: tuple, certifier, challenger,
            strategy: str) -> dict:
    """One served run on the injected clock and its offline check, the shim tampered for `strategy` (honest runs the
    plan as is): the certifier (`active.Proxy`, salt on) in front of the endpoint at `upstream`, then `check` while a
    reveal endpoint serves on the run's loop. Lifts `test_certifier_serve._run`. Returns the check and, read from the
    run's own events, how many events each connection carried."""
    setup = ar.Setup(report)
    run.mkdir(parents=True, exist_ok=True)
    (run / "plan.json").write_text(json.dumps(plan))
    (run / "calibration.json").write_text(json.dumps(report))
    shim = plan if strategy == "honest" else ar.adversary_plan(plan, setup, strategy)
    out: dict = {}

    async def after(proxies, clock):
        reveals = await active.serve_reveals(proxies[0], "127.0.0.1", 0)
        record = cs.deployment(proxies[0], setup, plan["windows"], session=SESSION, n2_link=cs.N2_LINK,
                               link=("127.0.0.1", 0), reveal=reveals.sockets[0].getsockname()[:2], origin_unix=0.0,
                               url=url)
        (run / "deployment.json").write_text(json.dumps(record))
        out["check"] = await asyncio.to_thread(cs.check, run, challenger, clock=lambda: 0.0, timeout=None)
        reveals.close()

    publish = cs.CertifyingPublisher(run, setup, Issuer(certifier, SESSION), cs.N2_LINK, threaded=False)
    ar.run_virtual(shim, report, run, upstream, salt=True, publish=publish, after=after,
                   proxy_args={"challenger_key": challenger.public_key(), "signer": certifier,
                               "served": cs.served_profile(plan)})
    carried: dict = defaultdict(int)
    for _, name, *_ in ar.read_events(run / "server.events.gz"):
        carried[name] += 1
    out["events"] = sorted(carried.values())
    return out


def _codes(check: dict) -> list[str]:
    """The N2 refusal codes the check names, over its windows."""
    out: list[str] = []
    for row in check.get("rows", ()):
        n2 = row.get("n2")
        if isinstance(n2, dict):
            out += n2.get("codes", [])
    return sorted(set(out))


# The relay: the failures switch.


def relay_response(tokens: int) -> bytes:
    """A streamed completion of `tokens` GLM-5.3 tokens as vLLM frames it (an event a token, its usage, `[DONE]`); a
    token that ends inside a character gives an event of no text."""
    detokenize = codecs.getincrementaldecoder("utf-8")()
    events = [{"choices": [{"index": 0, "text": detokenize.decode(glm_token(k)),
                            "finish_reason": "length" if k == tokens - 1 else None}]} for k in range(tokens)]
    events.append({"choices": [], "usage": {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": tokens}})
    return (b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"
            + b"".join(ws._sse(e) for e in events) + b"e\r\ndata: [DONE]\n\n\r\n0\r\n\r\n")


class _Side:
    """One end of an in-memory connection as the relay holds it: a reader fed what the peer sends, and a transport
    that keeps each write with the tick it was made in."""

    def __init__(self, warden: Warden | None = None) -> None:
        self.warden, self.writes, self.closed, self.transport = warden, [], False, self
        self.reader, self.fed, self.eof = asyncio.StreamReader(), 0, False

    def feed(self, data: bytes, end: bool) -> None:
        if data:
            self.reader.feed_data(data)
        self.fed += len(data)
        if end:
            self.reader.feed_eof()
            self.eof = True

    def write(self, data: bytes) -> None:
        self.writes.append((self.warden.ticks if self.warden else None, bytes(data)))

    def is_closing(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True

    def get_write_buffer_size(self) -> int:
        return 0


def relay_run(out: Path, key: Path, failed: Mapping[int, int], params: WardenParams = RELAY, *,
              requests: Sequence[bytes] | None = None,
              served: Sequence[tuple[bytes, bool]] | None = None) -> list[dict]:
    """`REQUESTS` connections through the relay for one window on a `VirtualClock`, every server answering
    `relay_response` whole but those of `failed` (by connect order), which send that many whole cells' bytes of it and
    withhold the rest. With `requests` and `served` the connections are those requests instead, the k-th server to be
    connected sending `served[k]`'s bytes and, when its flag says so, ending; a request that has the id form's head is
    logged as an id request. Published in `out` as `warden_serve`'s `serve` and `client` publish a run (lifts
    `test_warden_serve._run`), with the relay's log (`warden.json`), each bucket ticked once the relay has read what
    its servers sent (as the relay's own tests settle a bucket). Returns the relay's log, an entry a connection."""
    out.mkdir(parents=True, exist_ok=True)
    tau, B, P = params.bucket_ms / 1000, params.cell_bytes, params.payload
    if served is None:
        response = relay_response(MERGED_TOKENS)
        whole = len(response) // P
        if whole + 1 > params.egress_cells or any(not 0 <= at <= whole for at in failed.values()):
            raise ValueError(f"a response of {whole + 1} cells doesn't fit the relay's {params.egress_cells} egress "
                             f"cells, or a connection of {dict(failed)} fails past its last whole cell")
        served = [(response[:failed[k] * P], False) if k in failed else (response, True) for k in range(REQUESTS)]
    if requests is None:
        requests = [ws.http_request({"model": "glm-5.3", "prompt": f"request {k}", "max_tokens": MERGED_TOKENS,
                                     "stream": True, "stream_options": {"include_usage": True}}, "stand-in:1")
                    for k in range(REQUESTS)]
    signer = cs.signer(key, 4)
    public = signer.public_key()
    publisher = ws.Publisher(out, Issuer(signer, SESSION), params, WR.Cells.of)
    servers: list[_Side] = []

    async def connect():
        side, k = _Side(), len(servers)
        servers.append(side)
        side.feed(*served[k])
        return side.reader, side

    def settled(w: Warden) -> bool:
        for c in w.active:
            if c.server is None or not c.ingress_ended:
                return False
            if c.bytes_out != c.server[1].fed or c.egress_ended != c.server[1].eof:
                return False
        return True

    async def main() -> tuple[list[_Side], Warden]:
        clock = VirtualClock(-tau / 2)
        warden = Warden(params, connect, origin=0.0, windows=1, clock=clock, sleep=clock.sleep, publish=publisher,
                        withhold=publisher.withhold)
        running = asyncio.create_task(warden.run())
        clients, tasks = [], []
        for request in requests:
            side = _Side(warden)
            side.feed(cells(request, B), True)
            clients.append(side)
            tasks.append(asyncio.create_task(warden.handle(side.reader, side)))
        await warden.until(lambda w: w.ordinals + w.refused == len(requests))
        for t in range(params.window):
            await warden.until(settled)
            clock.advance(t * tau)
            await warden.until(lambda w, t=t: w.ticks > t)
        await running
        await asyncio.gather(*tasks)
        return clients, warden

    try:
        clients, warden = asyncio.run(main())
    finally:
        publisher.close()
        signer.close()
    deployment = ws.deployment_record(params, public, SESSION, origin_unix=0.0, windows=1, listen=("127.0.0.1", 1),
                                      upstream=("127.0.0.1", 2), slack=warden.slack, run_id=warden.run_id)
    (out / "deployment.json").write_text(json.dumps(deployment))
    log = []
    for index, (request, client) in enumerate(zip(requests, clients)):
        shim = Shim(B)
        for t, data in client.writes:
            for i in range(0, len(data), B):
                kind, shim.ordinal, payload = read_cell(data[i:i + B], B)
                shim.arrivals.append((t * tau + 0.001, kind))
                shim.data += payload if kind in (DATA, END) else b""
        kinds = [k for _, k in shim.arrivals]
        shim.outcome = ("refused" if REFUSED in kinds else "end" if END in kinds else "fail" if FAIL in kinds
                        else "cut")
        log.append(ws.connection_log(index, 0.0, 0.0, request, shim, 0.0,
                                     P if request.startswith(active.ID_REQUEST_HEAD) else None))
    (out / "client.json").write_text(json.dumps({"connections": log}))
    (out / "warden.json").write_text(json.dumps({"connections": warden.log}))
    return warden.log


def relay(tmp: Path) -> dict:
    """The failures switch's two relay runs, each with the relay's log and its offline check: `FAILED`'s connections
    failed, and none (the control)."""
    out = {}
    for name, failed in (("control", {}), ("failed", FAILED)):
        log = relay_run(tmp / name, tmp / "keys" / name, failed)
        out[name] = {"log": log, "check": ws.check(tmp / name)}
    return out


# The timing test: the timing switch.


def timing(out: Path, intervals: int) -> dict:
    """`benchmarks/certifier_timing`'s run of its burst attack with one observer against this checkout's certifier,
    `intervals` intervals a case: its result and its exit code, or why it wrote none. It measures the host it runs on."""
    root = str(HERE.parents[1])
    env = os.environ | {"PYTHONPATH": root}
    done = subprocess.run([sys.executable, str(HERE.parent / "certifier_timing" / "run.py"), "run", "--attacks",
                           "burst", "--observers", "spin", "--intervals", str(intervals), "--out", str(out)],
                          env=env, capture_output=True, text=True)
    if not (out / "result.json").exists():
        return {"error": f"the timing test wrote no result (exit {done.returncode}): {done.stderr[-300:].strip()}"}
    return json.loads((out / "result.json").read_text()) | {"exit": done.returncode}


# The rows.


async def _serve(tmp: Path, switches: tuple[str, ...], profile: str, timing_intervals: int = 1200) -> dict:
    """Run each switch's runs; returns a row a switch, and before them the proxy's honest run (`baseline`) when a
    switch needs the proxy's runs. A switch that errors reports its error and leaves the others untouched."""
    rows: dict = {}
    proxied = [s for s in ("tokens", "length") if s in switches]
    if proxied:
        challenger = cs.signer(tmp / "keys" / "challenger", 6)

        async def one(where: Path, strategy: str, up: tuple, u: str, plan: dict, report: dict) -> dict:
            certifier = cs.signer(tmp / "keys" / where.name, 6)
            try:
                return await asyncio.to_thread(run_one, where, plan, report, u, up, certifier, challenger, strategy)
            except RUN_ERRORS as err:
                return {"check": {"error": repr(err)}, "events": None}
            finally:
                certifier.close()

        server = await asyncio.start_server(StandIn(1, merges=lambda r: r % 2 == 1), "127.0.0.1", 0)
        upstream = server.sockets[0].getsockname()[:2]
        url = "http://%s:%d" % upstream
        try:
            _, report, plan = await asyncio.to_thread(cs.plan_of, plan_args(profile), url, lambda: 0.0)
            need = ["honest"] + (["token"] if "tokens" in switches else []) + \
                   (["extra", "drop"] if "length" in switches else [])
            runs = {s: await one(tmp / s, s, upstream, url, plan, report) for s in need}
            rows["baseline"] = _baseline_row(runs["honest"])
            if "tokens" in switches:
                rows["tokens"] = _tokens_row(runs["token"], runs["honest"])
            if "length" in switches:
                rows["length"] = _length_row(runs["extra"], runs["drop"], runs["honest"], profile)
        finally:
            server.close()
            challenger.close()
    if "timing" in switches:
        rows["timing"] = _timing_row(await asyncio.to_thread(timing, tmp / "timing", timing_intervals))
    if "failures" in switches:
        try:
            rows["failures"] = _failures_row(await asyncio.to_thread(relay, tmp / "relay"))
        except RUN_ERRORS as err:
            rows["failures"] = {"switch": "failures", "cheat": "the server fails connections on purpose",
                                "expected": "accepted, and charged in the table", "actual": f"error: {err!r}",
                                "error": repr(err)}
    return {s: rows[s] for s in ("baseline", *switches) if s in rows}


def _err(run: dict) -> str | None:
    return run["check"].get("error")


def _verdict(run: dict) -> str:
    check = run["check"]
    return f"{check['verdict']} {_codes(check)}" if _codes(check) else check["verdict"]


def _baseline_row(honest: dict) -> dict:
    err, check = _err(honest), honest["check"]
    return {"switch": "baseline", "cheat": "honest serving through the proxy (no cheat)",
            "expected": "accepted on every link-window",
            "actual": err or (f"{check['verdict']} (network_verdict {check['network_verdict'].get('accepted')}, "
                              f"audit {check['audit_accepted']}, salt {check['salt_accepted']}); events a connection "
                              f"{honest['events']}"),
            "ok": err is None and check.get("verdict") == "accepted", **({"error": err} if err else {})}


def _tokens_row(token: dict, honest: dict) -> dict:
    err = _err(token)
    caught = (not err) and token["check"]["verdict"] == "refused"
    events = "the honest run's" if (not err) and token["events"] == honest["events"] else token["events"]
    return {"switch": "tokens", "cheat": "ids of the server's own choosing in a response, its frame counts honest's",
            "expected": "refused by the audit at the sample's stated probability (the I/O tie: the frames' tokens "
                        "against the committed sampled ids)",
            "actual": err or (f"{_verdict(token)} by N2's counts, the only check of it this harness runs; events a "
                              f"connection {events}"
                              + ("" if caught else ": not caught, the I/O tie is not run here")),
            "caught": caught, **({"error": err} if err else {})}


def _length_row(extra: dict, drop: dict, honest: dict, profile: str) -> dict:
    ee, de = _err(extra), _err(drop)
    both = (not ee) and (not de) and extra["check"]["verdict"] == "refused" and drop["check"]["verdict"] == "refused"
    late = ee or f"{_verdict(extra)}, events a connection {extra['events']}"
    early = de or f"{_verdict(drop)}, events a connection {drop['events']}"
    return {"switch": "length", "cheat": "a response ended late (one event more) or early (one event fewer)",
            "expected": "refused by the stop check",
            "actual": f"by N2's counts on its {profile} profile, late: {late}; early: {early}; the honest run's "
                      f"{honest['events']}",
            "caught": both, **({"error": ee or de} if ee or de else {})}


def _timing_row(result: dict) -> dict:
    row = {"switch": "timing", "caught": None,
           "cheat": "the server writes an interval's bytes in a burst around the certifier's release time",
           "expected": "the timing test's bits an interval, under its budget"}
    if result.get("error"):
        return row | {"actual": result["error"], "error": result["error"]}
    goal, guard = result["target"], result.get("guard") or {}
    lo, hi = sorted(goal["bits"])
    row["expected"] += f" of {lo:.3g} to {hi:.3g} at a release interval of {goal['interval_s']:.3g} s"
    if result["exit"] or result.get("problems") or result.get("value") is None:
        why = (f"no figure (the timing test exited {result['exit']}; its guard "
               f"{'passed' if guard.get('ok') else 'failed'}): {'; '.join(map(str, result.get('problems') or []))[:300]}")
        return row | {"actual": why, "error": why}
    met, best = goal["met"], result["best"]
    under = "under both ends of" if all(met) else "under the looser end only of" if any(met) else "not under"
    short = "" if all(met) else (f", which a decoder at chance needs at least {channel.samples_to_show(lo)['least']} "
                                 f"scored intervals to show ({channel.samples_to_show(hi)['least']} for its looser end)")
    return row | {"actual": (f"{result['value']:.3g} bits an interval (95% {result['lower']:.3g} to "
                             f"{result['upper']:.3g}, {result['n']} scored intervals, the {best['stamp']} stamp) "
                             f"with its guard {'passed' if guard.get('ok') else 'failed'}; its ceiling "
                             f"{result['ceiling']:.3g} is {under} the budget{short}"),
                  "under_budget": met}


def _failures_row(runs: dict, params: WardenParams = RELAY) -> dict:
    control, cheat = runs["control"], runs["failed"]
    n = len(cheat["log"])
    at = {c["ordinal"]: (c["failed"], c["failed_at"]) for c in cheat["log"] if c["failed"] is not None}
    clean = control["check"]["accepted"] and not any(c["failed"] for c in control["log"])
    s = cheat["check"]["summary"]
    named = [p for p in cheat["check"]["problems"] if "failed closed" in p]
    today = LB.terms(n, 1, cells=params.egress_cells, steps=params.window)
    held = LB.terms(n, 1, cells=params.egress_cells, steps=params.window, changes=("lengths",), failures=len(at))
    where = ", ".join(f"connection {o} {why} at egress cell {cell}" for o, (why, cell) in sorted(at.items()))
    verdict = "ACCEPTED" if cheat["check"]["accepted"] else "REFUSED"
    return {"switch": "failures",
            "cheat": "the server sends part of a response and withholds the rest, so the relay fails the connection",
            "expected": "accepted, and charged in the table (leak_bounds' fail cells)",
            "actual": (f"{len(at)} of {n} connections failed on the certifier's record ({where or 'none'}), "
                       f"{s['certified_windows']}/{len(ws.CERTIFIED) * s['windows']} link-windows certified; the "
                       f"relay's check: {verdict}" + (f" ({'; '.join(named[:4])})" if named else "")
                       + f", and {'ACCEPTED' if clean else 'NOT as an honest run'} with none failed; the table "
                       f"charges the fail cells {today.cells} bits over the {n} connections as the scan counts them, "
                       f"failed or not, and {held.cells:.1f} bits pooled where the lengths are held"),
            "caught": not cheat["check"]["accepted"], "accepted": cheat["check"]["accepted"], "failed": len(at),
            "charged_bits": {"scan": today.cells, "lengths_held": held.cells}, "ok": clean}


def table(rows: dict) -> str:
    lines = []
    for r in rows.values():
        lines.append(f"[{r['switch']}] {r['cheat']}")
        lines.append(f"    expected: {r['expected']}")
        lines.append(f"    actual:   {r['actual']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="served_cheats.py", description=__doc__.splitlines()[0])
    ap.add_argument("--stand-in", action="store_true",
                    help="the dry run: every record built in process on an injected clock from GLM-5.3's tokens")
    ap.add_argument("--switch", choices=("all", *SWITCHES), default="all")
    ap.add_argument("--profile", choices=cs.PROFILES, default="served",
                    help="N2's profile for the tokens and length switches")
    ap.add_argument("--timing-intervals", type=int, default=1200,
                    help="the intervals of the timing switch's case (100 ms each, on the wall clock)")
    ap.add_argument("--out", type=Path, help="keep each switch's run directory and the rows here")
    a = ap.parse_args(argv)
    if not a.stand_in:
        raise SystemExit("only --stand-in (the dry run) is wired: no switch reads a kept served record")
    switches = SWITCHES if a.switch == "all" else (a.switch,)
    tmp = (a.out if a.out is not None else Path(tempfile.mkdtemp(prefix="served-cheats-"))).resolve()
    tmp.mkdir(parents=True, exist_ok=True)
    rows = asyncio.run(_serve(tmp, switches, a.profile, a.timing_intervals))
    (tmp / "rows.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(table(rows), flush=True)
    broken = [s for s, r in rows.items() if r.get("error") or r.get("ok") is False]
    if broken:
        print(f"\nnot a result: {', '.join(broken)} errored or its control did not run as an honest run does",
              flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
