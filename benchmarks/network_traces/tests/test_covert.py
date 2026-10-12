"""The covert sender through the online prover's declared sync (`covert.py`), on an injected clock: the proxy, the server
and the peer over localhost sockets, then the audit. The sender stays accepted in every window, the declarations it
predicts are the proxy's, under `last` the receiver reads every symbol it sent, and under a sync fixed before the run
(the control) it conveys nothing. Every measured rate is at most log2 #Σ_sync."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from verity.core.protocols.certifier import calibration

HERE = Path(__file__).resolve().parents[1]
T = 60
W = 8


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_active_replay", HERE / "active_replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ar = _load()
import covert  # noqa: E402  (on the path `_load` set)


def trace() -> dict:
    """W windows of T buckets, each with four sessions of four frames, the first ready 1 to 4 buckets after its anchor."""
    frames, arrival_ms = [], []
    for w in range(W):
        for j, (offset, latency) in enumerate(((0, 1), (10, 3), (20, 2), (35, 4))):
            anchor = w * T + offset
            frames += [{"session": f"s{w}.{j}", "index": i, "anchor": anchor, "ready": anchor + latency + i}
                       for i in range(4)]
            arrival_ms.append(anchor * 100 + 30.0)
    frames.sort(key=lambda f: (f["ready"], f["anchor"], f["session"], f["index"]))
    return {"bucket_ms": 100, "window_buckets": T, "frame_bytes": 4096, "queue_capacity": 1 << 16,
            "configurations": [{"name": f"synthetic/w0-{W}", "labels": {"run": "r0", "windows": [0, W]},
                                "egress": [{"link": "responses", "frames": frames}],
                                "ingress": [{"link": "requests", "arrival_ms": arrival_ms}]}]}


@pytest.fixture(scope="module")
def setup():
    doc = trace()
    report = calibration.calibrate(doc)
    return doc, report, ar.Setup(report)


def run(setup, tmp_path, syncs: str, strategy: str, seed: int = 3) -> dict:
    doc, report, s = setup
    plan = ar.plan_configuration(doc["configurations"][0], s, syncs=syncs)
    mine = plan if strategy == "honest" else ar.adversary_plan(plan, s, strategy, seed)
    out = tmp_path / f"{syncs}-{strategy}"
    assert ar.run_virtual(mine, report, out) == {"server": 0, "proxy": 0, "peer": 0}
    return ar.audit_configuration(mine, report, out)


@pytest.mark.parametrize("syncs", ["last", "cover-2-1"])
def test_the_covert_sender_stays_accepted_and_predicts_the_proxys_declarations(setup, tmp_path, syncs):
    audit = run(setup, tmp_path, syncs, "covert")
    assert all(r["raw"] is None and r["committed"] is None for r in audit["rows"])
    c = audit["comparisons"]
    assert c["covert_declared_are_predicted"] and c["records_are_run_shapers"] and c["hands_are_the_rules"]
    assert c["covert"]["mutual_information_full_bits"] <= setup[2].sync_set.bits


def test_under_last_the_receiver_reads_every_symbol_sent(setup, tmp_path):
    """`last` declares exactly the frontier the sender set in the window before: δ_declared(w + 1) = g_w, so the
    decoder makes no error and the information it recovers is the symbols' entropy."""
    c = run(setup, tmp_path, "last", "covert")["comparisons"]["covert"]
    assert [int(d) for d, _ in c["declared"][1:]] == c["symbols"][:-1]
    assert c["symbol_error_rate"] == 0 and c["decoded_bits_per_window"] > 0
    assert c["mutual_information_full_bits"] > 0


def test_under_a_fixed_sync_the_covert_sender_conveys_nothing(setup, tmp_path):
    """The control: syncs chosen before the run don't read the sender, so every window declares the envelope and the
    mutual information is exactly 0, though the sender wrote the same way."""
    audit = run(setup, tmp_path, "envelope", "covert")
    assert all(r["raw"] is None and r["committed"] is None for r in audit["rows"])
    c = audit["comparisons"]["covert"]
    assert len({tuple(d) for d in c["declared"]}) == 1
    assert c["mutual_information_full_bits"] == c["mutual_information_corrected_bits"] == 0
    assert c["decoded_bits_per_window"] == 0


async def _fake_vllm(reader, writer):
    """A streaming completions endpoint that answers at once: `max_tokens` events and `[DONE]`."""
    head = await reader.readuntil(b"\r\n\r\n")
    body = json.loads(await reader.readexactly(int(head.lower().split(b"content-length: ")[1].split(b"\r\n")[0])))
    events = [b'data: {"choices":[{"text":" t%d"}]}\n\n' % k for k in range(body["max_tokens"])] + [b"data: [DONE]\n\n"]
    writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n" + b"".join(events))
    await writer.drain()
    writer.close()


def live_run(setup, tmp_path, syncs: str) -> tuple[dict, dict]:
    doc, report, s = setup
    out = tmp_path / f"live-{syncs}"

    async def go(plan):
        server = await asyncio.start_server(_fake_vllm, "127.0.0.1", 0)
        await asyncio.to_thread(ar.run_virtual, plan, report, out, server.sockets[0].getsockname()[:2])
        server.close()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ar, "live_body", lambda session, model: {"model": model, "prompt": [1], "max_tokens": 3,
                                                            "stream": True})
        plan = ar.adversary_plan(ar.plan_configuration(doc["configurations"][0], s, "served", syncs), s, "covert", 3)
        asyncio.run(go(plan))
    return plan, ar.audit_configuration(plan, report, out)


def test_a_live_shim_that_only_delays_steers_last_and_stays_accepted(setup, tmp_path):
    """`covert.LiveSender` in the shim holds each stream's first event until its window's symbol: on an endpoint that
    answers at once the frontier is the symbol, so under `last` the next window declares it, and its replica of the
    proxy's prover predicted every declaration."""
    _, audit = live_run(setup, tmp_path, "last")
    assert all(r["raw"] is None and r["committed"] is None for r in audit["rows"])
    c = audit["comparisons"]
    assert c["covert_declared_are_predicted"] and c["frames_log_is_the_programs"] and c["wire_is_the_records"]
    assert [int(d) for d, _ in c["covert"]["declared"][1:]] == c["covert"]["symbols"][:-1]
    assert c["covert"]["symbol_error_rate"] == 0


def test_a_live_shim_under_the_envelope_conveys_nothing(setup, tmp_path):
    _, audit = live_run(setup, tmp_path, "envelope")
    assert all(r["raw"] is None and r["committed"] is None for r in audit["rows"])
    c = audit["comparisons"]["covert"]
    assert len({tuple(d) for d in c["declared"]}) == 1
    assert c["mutual_information_full_bits"] == c["decoded_bits_per_window"] == 0


def test_in_steady_state_last_carries_nothing_and_a_margin_keeps_the_channel_open(setup):
    """Accepted under `last`, a window's frontier is at most the δ it declared, and the next declares that frontier: δ
    never rises, so past the start a sender's rate goes to 0. A margin lets δ rise again."""
    s = setup[2]
    rate = {r: covert.steady_state(s.sync_set, T, r, s.envelope, 300, 1)["mutual_information_corrected_bits"]
            for r in ("last", "cover-1-1", "envelope")}
    assert rate["last"] < 0.1 and rate["envelope"] == 0
    assert 1 < rate["cover-1-1"] <= s.sync_set.bits


def test_the_measure_pools_runs_and_is_exact_on_a_noiseless_channel():
    s = ar.Setup(calibration.calibrate(trace())).sync_set
    symbols = [3, 1, 2, 1, 3, 2]
    m = covert.measure_runs([(symbols, [[0, "1"]] + [[g, "1"] for g in symbols[:-1]])] * 2, s)
    assert m["window_decisions"] == 10 and m["symbol_error_rate"] == 0
    assert m["mutual_information_delta_bits"] == pytest.approx(m["declared_delta_entropy_bits"])
    assert covert.margin_of("last") == 0 and covert.margin_of("cover-3-2") == 2
    assert covert.stream_seed(0, "a/w0-4") != covert.stream_seed(0, "a/w5-9") != covert.stream_seed(1, "a/w5-9")
