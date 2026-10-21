"""N3Window_v5 with the canonical framing's templates (`workload.n3_canonical`) on the active replay's raw records, on an
injected clock: an honest run is accepted on every window, egress and ingress, and so is one whose tokens are longer
than an event (cut over events of `TOKEN_BYTES`), empty or of every byte value, and whose prompt takes five frames, and
so is a live endpoint's behind the shim, with a token of 200 bytes in every response. A byte flipped in a frame's text,
header, suffix or fill, a request byte that isn't the peer's, or one more event than the Program has, is refused in its
own window and nowhere else."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from verity.core.protocols.certifier import active, calibration
from verity.core.protocols.certifier.grid import Record

HERE = Path(__file__).resolve().parents[1]
B = 4096


@pytest.fixture(scope="module")
def m():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "path", [str(HERE), *sys.path])
        spec = importlib.util.spec_from_file_location("network_sim_mockdc", HERE / "mockdc.py")
        mod = importlib.util.module_from_spec(spec)
        mp.setitem(sys.modules, spec.name, mod)
        spec.loader.exec_module(mod)
        mod.C = importlib.import_module("workload.n3_canonical")
        yield mod


def replay(m, doc: dict, out: Path, edit=None, tokens: dict | None = None) -> tuple:
    report = calibration.calibrate(doc)
    s = m.ar.Setup(report)
    plan = m.ar.plan_configuration(doc["configurations"][0], s)
    if edit is not None:
        plan = edit(plan, s)
    if tokens is not None:
        plan = plan | {"tokens": {k: [t.hex() for t in v] for k, v in tokens.items()}}
    assert m.ar.run_virtual(plan, report, out) == {"server": 0, "proxy": 0, "peer": 0}
    return plan, s


def records(m, run: Path) -> dict:
    return {link: [m.ar.unpack_record(b, B) for b in m.ar.read_blobs(run / f"{link}.records")]
            for link in ("responses", "requests")}


def verdicts(m, plan, s, run, recs=None) -> dict[tuple[str, int], bool]:
    n3, windows = m.C.replay_windows(plan, s, run, recs)
    return {(r["port"], r["window"]): r["ok"] for r in m.C.verdicts(n3, windows)}


EVERY = {(p, w): True for p in ("responses", "requests") for w in range(3)}


@pytest.fixture(scope="module")
def honest(m, tmp_path_factory):
    run = tmp_path_factory.mktemp("honest")
    plan, s = replay(m, m.trace(), run)
    return plan, s, run


def test_an_honest_run_is_accepted_on_every_window(m, honest):
    assert verdicts(m, *honest) == EVERY


def long_trace(m) -> dict:
    """The synthetic trace with window 0's first request five frames long."""
    doc = m.trace()
    ingress = doc["configurations"][0]["ingress"][0]
    first = ingress["arrival_ms"][0]
    ingress["arrival_ms"] = [first] * 4 + ingress["arrival_ms"]
    return doc


def test_long_tokens_and_a_long_prompt_are_accepted_on_every_window(m, tmp_path):
    """A token of 200 bytes takes four events (64, 64, 64 and 8 bytes); one of 65 takes two; an empty token is an event
    of no bytes; the tokens hold NUL, `=` and 0xff. The first request is five frames, its length in the first."""
    long = bytes(range(200))
    tokens = {"s0.0": [long[i:i + 64] for i in range(0, 200, 64)],
              "s0.1": [b"\xff" * 64, b"\xff", b"", b"="], "s1.2": [b"\0" * 63, b" t1", b"=" * 64, b"\0"]}
    assert tokens["s0.0"] == [bytes.fromhex(h) for h in m.ar.token_frames(long)]
    plan, s = replay(m, long_trace(m), tmp_path, tokens=tokens)
    assert max(q["frames"] for q in plan["requests"]) == 5
    assert verdicts(m, plan, s, tmp_path) == EVERY


LIVE_TOKENS = [" t0", "é" * 100, "", " t3"]


async def _fake_vllm(reader, writer):
    """A streaming completions endpoint that answers at once with `LIVE_TOKENS`' chunks and `[DONE]`: a token of 200
    bytes, which `active_replay.canonical_events` cuts over four events, and an empty one, an event of no bytes."""
    head = await reader.readuntil(b"\r\n\r\n")
    await reader.readexactly(int(head.lower().split(b"content-length: ")[1].split(b"\r\n")[0]))
    chunks = [b"data: " + json.dumps({"choices": [{"text": t}]}).encode() + b"\n\n" for t in LIVE_TOKENS]
    writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"
                 + b"".join(b"%x\r\n%s\r\n" % (len(p), p) for p in [*chunks, b"data: [DONE]\n\n"]) + b"0\r\n\r\n")
    await writer.drain()
    writer.close()


def test_a_live_endpoint_behind_the_shim_is_accepted_on_every_window(m, tmp_path):
    """The live shim (`active_replay.LiveServer`), the path a served vLLM endpoint takes, writes each completion chunk's
    text as canonical events: with a token of 200 bytes and an empty one in every response, v5 accepts every window.
    The server's outbox is the shim's own log (`n3_canonical.replay_units`), so this checks that the shim's frames fit
    the template, not what they carry."""
    doc = m.trace()
    report = calibration.calibrate(doc)
    s = m.ar.Setup(report)

    async def go(plan):
        server = await asyncio.start_server(_fake_vllm, "127.0.0.1", 0)
        await asyncio.to_thread(m.ar.run_virtual, plan, report, tmp_path, server.sockets[0].getsockname()[:2])
        server.close()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m.ar, "live_body", lambda session, model: {"model": model, "prompt": [1, 2, 3],
                                                              "max_tokens": len(LIVE_TOKENS), "stream": True})
        plan = m.ar.plan_configuration(doc["configurations"][0], s, "served", "envelope")
        asyncio.run(go(plan))
        got = verdicts(m, plan, s, tmp_path)
    responses: dict[str, list] = {}
    for _, session, k, _, e in sorted(m.ar.read_events(tmp_path / "server.events.gz")):
        responses.setdefault(session, []).append((k, active.event_data(active.event_body(e))))
    long = ("é" * 100).encode()
    each = [b" t0", long[:64], long[64:128], long[128:192], long[192:], b"", b" t3"]
    assert len(responses) == len(plan["requests"])
    assert all([d for _, d in sorted(r)] == each for r in responses.values())
    assert got == EVERY


def flip(recs: dict, link: str, window: int, k: int, at: int) -> dict:
    """The records with byte `at` of the k-th frame link's window `window` released flipped in its low bit."""
    out = dict(recs)
    edited = []
    for r in recs[link]:
        if r.window == window:
            seen, grid = 0, []
            for row in r.grid:
                new = []
                for f in row:
                    real = f is not None and any(f)
                    if real and seen == k:
                        f = f[:at] + bytes([f[at] ^ 1]) + f[at + 1:]
                    seen += real
                    new.append(f)
                grid.append(tuple(new))
            r = Record(r.link, r.window, tuple(grid), r.status)
        edited.append(r)
    out[link] = edited
    return out


P, S, _, _ = active.event_placement(B)
TAMPERS = {"event-text": ("responses", 0, 0, P), "event-header": ("responses", 1, 2, 3),
           "event-suffix": ("responses", 0, 1, S + 2), "event-fill": ("responses", 2, 0, B - 1),
           "request-byte": ("requests", 0, 0, 20), "request-header": ("requests", 1, 0, 8),
           "request-fill": ("requests", 2, 0, B - 1)}


@pytest.mark.parametrize("name", TAMPERS)
def test_a_tampered_frame_is_refused_in_its_window(m, honest, name):
    link, window, *_ = TAMPERS[name]
    plan, s, run = honest
    got = verdicts(m, plan, s, run, flip(records(m, run), *TAMPERS[name]))
    assert got == EVERY | {(link, window): False}


def test_an_extra_event_is_refused_in_its_window(m, tmp_path):
    """mockdc's `extra` server: honest, but for one more event on the first session at the position after the
    Program's last, so the last window holds it; declared against the honest Program."""
    plan, s = replay(m, m.trace(), tmp_path, edit=lambda p, s: m.extra_plan(p))
    honest_plan = {k: v for k, v in plan.items() if k not in ("strategy", "writes")}
    assert verdicts(m, honest_plan, s, tmp_path) == EVERY | {("responses", 2): False}
