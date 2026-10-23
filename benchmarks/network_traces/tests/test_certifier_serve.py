"""certifier-serve on an injected clock, end to end: the served plan probed from a stub OpenAI streaming endpoint, the
certifier (`active.Proxy` with its salt on, its records certified and its reveals signed) in front of it behind the
shim, and the offline check of what it published, which asks each window's salt through a `RemoteCertifier` signed
with the challenger's key. Honest traffic is accepted on every link-window (N2 on the planned profile, the audit and
the salt), a shim that holds one session past every sync or writes one event more than the endpoint sent is refused,
and each window's salt is revealed once. The same holds four in flight against an endpoint that sends some responses'
last two tokens in one event, as vLLM does: the plan's F is the body's tokens, and the shim pads a short response."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest
from verity.core.protocols.certifier import active
from verity.core.protocols.certifier.certificate import Issuer

HERE = Path(__file__).resolve().parents[1]
TOKENS, REQUESTS = 4, 8
SESSION = bytes(range(32))


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_certifier_serve", HERE / "certifier_serve.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cs = _load()
ar = cs.ar


async def _endpoint(reader, writer):
    """A streaming completions endpoint that answers at once: `max_tokens` chunk events and `[DONE]`, chunked."""
    head = await reader.readuntil(b"\r\n\r\n")
    body = json.loads(await reader.readexactly(int(re.search(rb"content-length: (\d+)", head, re.I).group(1))))
    assert head.startswith(b"POST /v1/completions ") and "user" not in body and body["ignore_eos"]
    events = [b'data: {"choices":[{"text":" t%d"}]}\n\n' % k for k in range(body["max_tokens"])] + [b"data: [DONE]\n\n"]
    writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"
                 + b"".join(b"%x\r\n%s\r\n" % (len(p), p) for p in events) + b"0\r\n\r\n")
    await writer.drain()
    writer.close()


class _SideBySide:
    """An OpenAI endpoint that serves `group` streaming completions side by side, as vLLM serves requests in flight:
    each group's responses wait for the whole group, then write their chunks in turn, one each, so the shim reads them
    interleaved (`writes`: (request, chunk) in write order). `/v1/models` lists one model and `/tokenize` makes 200
    tokens of any prompt. For the requests `merges` picks (by arrival), the last two tokens share one chunk, as vLLM
    v1's output collector merges a request's deltas the API server hasn't read yet."""

    def __init__(self, group: int, merges=lambda request: False) -> None:
        self.group, self.arrived, self.writes, self.bodies, self.merges = group, 0, [], [], merges
        self.turns: dict = defaultdict(asyncio.Event)

    async def __call__(self, reader, writer):
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
        self.bodies.append(body)
        me, self.arrived = self.arrived, self.arrived + 1
        g, i = divmod(me, self.group)
        if i == self.group - 1:
            self.turns[(g, 0, 0)].set()
        writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n")
        text, n = b"", body["max_tokens"]
        for k in range(n):
            await self.turns[(g, k, i)].wait()
            text += b" t%d" % k
            if not (k == n - 2 and self.merges(me)):
                p, text = b'data: {"choices":[{"text":"%s"}]}\n\n' % text, b""
                writer.write(b"%x\r\n%s\r\n" % (len(p), p))
                await writer.drain()
                self.writes.append((me, k))
            self.turns[(g, k, i + 1) if i + 1 < self.group else (g, k + 1, 0)].set()
        writer.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(b"data: [DONE]\n\n"), b"data: [DONE]\n\n"))
        await writer.drain()
        writer.close()


def _run(run: Path, plan: dict, report: dict, url: str, upstream: tuple, certifier, challenger, strategy: str) -> dict:
    """One served run on the injected clock and its check, which runs once the run is over, while the proxy's reveal
    endpoint serves on the run's loop."""
    setup = ar.Setup(report)
    run.mkdir(parents=True)
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
        out["revealed"] = sorted(proxies[0]._revealed)
        reveals.close()

    publish = cs.CertifyingPublisher(run, setup, Issuer(certifier, SESSION), cs.N2_LINK, threaded=False)
    ar.run_virtual(shim, report, run, upstream, salt=True, publish=publish, after=after,
                   proxy_args={"challenger_key": challenger.public_key(), "signer": certifier})
    out["events"] = json.loads((run / "server.json").read_text())["events"]
    return out


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("served")
    challenger = cs.signer(tmp / "keys" / "challenger", 6)

    async def go():
        endpoint = await asyncio.start_server(_endpoint, "127.0.0.1", 0)
        upstream = endpoint.sockets[0].getsockname()[:2]
        url = "http://%s:%d" % upstream
        probed = await asyncio.to_thread(cs.probe, url, cs.request_body("stub", TOKENS), 2, clock=lambda: 0.0)
        report, plan = cs.served_plan(probed, REQUESTS)
        runs = {}
        for strategy in cs.STRATEGIES:
            certifier = cs.signer(tmp / "keys" / strategy, 6)
            try:
                runs[strategy] = await asyncio.to_thread(_run, tmp / strategy, plan, report, url, upstream, certifier,
                                                         challenger, strategy)
            finally:
                certifier.close()
        endpoint.close()
        return probed, report, plan, runs

    try:
        return asyncio.run(go())
    finally:
        challenger.close()


def test_the_served_plan_is_request_major_with_a_planned_egress_profile(served):
    probed, report, plan, _ = served
    assert [r["events"] for r in probed["responses"]] == [TOKENS, TOKENS]
    s = plan["served"]
    assert s["tokens"] == plan["pad_to"] == TOKENS and s["delta_buckets"] == 5 and plan["windows"] == 2
    sends = [q["send_ms"] for q in plan["requests"]]
    assert all(b - a == (s["delta_buckets"] + 1) * 100 for a, b in zip(sends, sends[1:]))
    assert len({q["frames"] for q in plan["requests"]}) == 1
    profile = plan["occupancy"]
    assert profile["kind"] == "planned" and profile["ingress"] is None
    assert len(profile["egress"]) == plan["windows"] and sum(map(sum, profile["egress"])) == REQUESTS * TOKENS
    setup = ar.Setup(report)
    assert plan["syncs"] == [[setup.envelope.delta, str(setup.envelope.rho)]] * plan["windows"]


def test_every_window_of_the_served_plan_ends_idle():
    """A latency budget of 20 buckets fits two requests a window: the third goes to the next window's second bucket,
    never to a first (the salt frame's), and each window's planned egress counts are its own requests' responses."""
    probed = {"url": "stub", "body": cs.request_body("stub", TOKENS),
              "responses": [{"events": TOKENS, "first_s": 0.0, "last_s": 0.0}]}
    report, plan = cs.served_plan(probed, 7, delta_buckets=20)
    T = cs.WINDOW_BUCKETS
    anchors = [q["enqueue"] for q in plan["requests"]]
    assert [a // T for a in anchors] == [0, 0, 1, 1, 2, 2, 3] and all(a % T for a in anchors)
    assert plan["windows"] == 5
    assert [sum(w) for w in plan["occupancy"]["egress"]] == [2 * TOKENS] * 3 + [TOKENS, 0]


def test_a_window_that_faults_drops_its_requests_and_the_audit_follows_the_released_streams(tmp_path):
    """The proxy stopped for 110 ms as it releases window 0's first request forwards it late, which faults window 0
    (`delivery-refused`) and drops its second request, so the proxy releases window 1's two as streams 2 and 3, not the
    plan's 3 and 4: the audit maps each served session to the stream the ingress records released it as, and refuses
    window 0 alone."""
    async def go():
        endpoint = await asyncio.start_server(_endpoint, "127.0.0.1", 0)
        upstream = endpoint.sockets[0].getsockname()[:2]
        probed = await asyncio.to_thread(cs.probe, "http://%s:%d" % upstream, cs.request_body("stub", TOKENS), 1,
                                         clock=lambda: 0.0)
        report, plan = cs.served_plan(probed, 4, delta_buckets=20)
        await asyncio.to_thread(ar.run_virtual, plan, report, tmp_path, upstream, stalls=[(1, 0.11)])
        endpoint.close()
        return report, plan

    report, plan = asyncio.run(go())
    records = [ar.unpack_record(b, cs.FRAME_BYTES) for b in ar.read_blobs(tmp_path / "requests.records")]
    assert ar.released_sessions(records) == {"served.0": 1, "served.2": 2, "served.3": 3}
    audit = ar.audit_configuration(plan, report, tmp_path)
    faulted = [r for r in audit["rows"] if r["raw"]]
    assert {(r["direction"], r["window"]) for r in faulted} == {("egress", 0), ("ingress", 0)}
    assert all(r["raw"] == r["committed"] == "status:delivery-refused" for r in faulted)


def test_four_in_flight_the_endpoints_shape_is_sent_and_the_shim_writes_request_major(tmp_path):
    """The plan sends the endpoint's shape (its one model, a prompt of exactly 128 of its tokens, `min_tokens` =
    `max_tokens`, `ignore_eos`, greedy, streamed) four at a time, the ingress releasing them two a bucket, and the
    endpoint interleaves the responses of each two released together chunk by chunk; the shim still writes each
    response whole, in the order their requests reached it, so the egress counts are the planned profile's and honest
    traffic is accepted on every link-window."""
    endpoint = _SideBySide(2)
    args = argparse.Namespace(model=None, api_key_env=None, prompt_tokens=128, tokens=TOKENS, probes=4, in_flight=4,
                              requests=REQUESTS, delta_buckets=None)

    async def go():
        server = await asyncio.start_server(endpoint, "127.0.0.1", 0)
        upstream = server.sockets[0].getsockname()[:2]
        url = "http://%s:%d" % upstream
        probed, report, plan = await asyncio.to_thread(cs.plan_of, args, url, lambda: 0.0)
        release = [plan["model_release"][str(q["stream"])] for q in plan["requests"]]
        assert release == [2, 2, 3, 3, 8, 8, 9, 9]
        endpoint.writes.clear()
        challenger, certifier = cs.signer(tmp_path / "keys" / "challenger", 6), cs.signer(tmp_path / "keys" / "c", 6)
        try:
            out = await asyncio.to_thread(_run, tmp_path / "run", plan, report, url, upstream, certifier, challenger,
                                          "honest")
        finally:
            certifier.close()
            challenger.close()
        server.close()
        return probed, plan, out

    probed, plan, out = asyncio.run(go())
    assert probed["body"] == {"model": "stub-model", "prompt": list(range(128)), "max_tokens": TOKENS,
                              "min_tokens": TOKENS, "stream": True, "ignore_eos": True, "temperature": 0}
    assert all(b == probed["body"] for b in endpoint.bodies) and len(endpoint.bodies) == 4 + 4 + REQUESTS
    assert plan["served"]["in_flight"] == 4 and plan["request_major"]
    sends = [q["send_ms"] for q in plan["requests"]]
    assert sends[:4] == [sends[0]] * 4 and sends[4:] == [sends[4]] * 4 and sends[4] > sends[0]
    assert endpoint.writes[:4] == [(8, 0), (9, 0), (8, 1), (9, 1)]
    written = [name for _, name, *_ in sorted(ar.read_events(tmp_path / "run" / "server.events.gz"))]
    whole = [name for name, _ in itertools.groupby(written)]
    names = [s for s, *_ in plan["sessions"]]
    assert len(written) == REQUESTS * TOKENS and len(whole) == REQUESTS
    assert all(set(whole[j:j + 2]) == set(names[j:j + 2]) for j in range(0, REQUESTS, 2))
    check = out["check"]
    assert check["verdict"] == "accepted", cs.table(check)
    assert all(r["n2"]["accepted"] for r in check["rows"]) and check["profile"] == "planned"


def test_honest_served_traffic_is_accepted_on_every_link_window(served):
    """N2's verdict of record accepts every link-window on the planned profile, `network_verdict` accepts the link,
    the certifier's audit accepts every record raw and committed, and the shim received every salt the certifier
    revealed to the challenger, which no egress frame holds."""
    *_, runs = served
    check = runs["honest"]["check"]
    assert check["verdict"] == "accepted", cs.table(check)
    assert check["network_verdict"]["accepted"] and check["network_verdict"]["refused"] == []
    assert all(r["n2"]["accepted"] and r["n2"]["public"] == {"verdict": 1, "deadline": 1} for r in check["rows"])
    assert all(r["audit"] == {d: {"raw": None, "committed": None} for d in ("egress", "ingress")} for r in check["rows"])
    assert check["salt"] == {"accepted": True, "violations": [], "ingress_salt_frames": [1, 1],
                             "egress_salt_frames": 0, "egress_holds_salt": False, "block_received": True}
    assert check["issuer_refused"] == [] and check["comparisons"]["wire_is_the_records"]


def test_each_salt_is_revealed_once_to_the_challengers_signed_request(served):
    *_, runs = served
    for strategy in cs.STRATEGIES:
        check = runs[strategy]["check"]
        assert [r["salt"] for r in check["rows"]] == ["revealed", "revealed"]
        assert re.fullmatch(r"refused: the certifier refused window 0's salt: window 0's salt is revealed already",
                            check["salt_again"])
        assert runs[strategy]["revealed"] == [0, 1]
        assert len({r["salt_fingerprint"] for r in check["rows"]}) == 2


def test_a_shim_that_holds_a_session_past_every_sync_is_refused(served):
    """The late shim's window-0 egress record is not established (`coverage-never-established`): the certifier's
    audit refuses it, N2's head check rejects the link-window, and `network_verdict` refuses the link."""
    *_, runs = served
    check = runs["late"]["check"]
    assert check["verdict"] == "refused" and not check["network_verdict"]["accepted"]
    assert 0 in check["network_verdict"]["refused"]
    first = check["rows"][0]
    assert first["n2"] == {"accepted": False, "codes": ["reject"], "public": {}}
    assert first["audit"]["egress"] == {"raw": "status:coverage-never-established",
                                        "committed": "status:coverage-never-established"}
    assert check["salt_accepted"]


def test_a_shim_that_writes_one_event_more_is_refused_on_n2s_counts_alone(served):
    """The extra event is on time and logged, so every record is complete and the certifier's audit accepts it: only
    N2's counts against the planned profile catch it, and `network_verdict` refuses the link at window 0."""
    *_, runs = served
    assert runs["honest"]["events"] == REQUESTS * TOKENS and runs["extra"]["events"] == REQUESTS * TOKENS + 1
    check = runs["extra"]["check"]
    assert check["comparisons"]["wire_is_the_records"]
    assert check["verdict"] == "refused" and check["network_verdict"]["refused"] == [0]
    assert check["audit_accepted"] and check["salt_accepted"] and check["issuer_refused"] == []
    assert all(r["audit"] == {d: {"raw": None, "committed": None} for d in ("egress", "ingress")} for r in check["rows"])
    assert check["rows"][0]["n2"] == {"accepted": False, "codes": ["reject"], "public": {}}
    assert check["rows"][1]["n2"]["accepted"]


def test_the_served_plan_takes_f_from_the_bodys_tokens_and_refuses_a_response_longer():
    """F is the body's token count, so probed responses of F − 1 and F events plan F frames each; a response of more
    than F events, or of none, or a body that doesn't ask for exactly its tokens, has no plan."""
    body = cs.request_body("stub", TOKENS)

    def probed(*lengths):
        return {"url": "stub", "body": body,
                "responses": [{"events": n, "first_s": 0.0, "last_s": 0.0} for n in lengths]}

    _, plan = cs.served_plan(probed(TOKENS - 1, TOKENS), 4, delta_buckets=20)
    assert plan["pad_to"] == plan["served"]["tokens"] == TOKENS
    assert plan["served"]["probe_events"] == [TOKENS - 1, TOKENS]
    assert sum(map(sum, plan["occupancy"]["egress"])) == 4 * TOKENS
    for lengths in ((TOKENS + 1,), (0, TOKENS)):
        with pytest.raises(ValueError, match="events long"):
            cs.served_plan(probed(*lengths), 4, delta_buckets=20)
    with pytest.raises(ValueError, match="exactly its tokens"):
        cs.served_plan(probed(TOKENS) | {"body": body | {"min_tokens": 1}}, 4, delta_buckets=20)


MERGED_TOKENS = 64


@pytest.fixture(scope="module")
def merged(tmp_path_factory):
    """The served plan probed and run four in flight, as the gate's rehearsal runs, against an endpoint that sends
    every other response's last two tokens in one chunk: honest, late and extra, each checked."""
    tmp = tmp_path_factory.mktemp("merged")
    endpoint = _SideBySide(2, merges=lambda request: request % 2 == 1)
    args = argparse.Namespace(model=None, api_key_env=None, prompt_tokens=128, tokens=MERGED_TOKENS, probes=4,
                              in_flight=4, requests=REQUESTS, delta_buckets=None)
    challenger = cs.signer(tmp / "keys" / "challenger", 6)

    async def go():
        server = await asyncio.start_server(endpoint, "127.0.0.1", 0)
        upstream = server.sockets[0].getsockname()[:2]
        url = "http://%s:%d" % upstream
        probed, report, plan = await asyncio.to_thread(cs.plan_of, args, url, lambda: 0.0)
        runs = {}
        for strategy in cs.STRATEGIES:
            certifier = cs.signer(tmp / "keys" / strategy, 6)
            try:
                runs[strategy] = await asyncio.to_thread(_run, tmp / strategy, plan, report, url, upstream, certifier,
                                                         challenger, strategy)
            finally:
                certifier.close()
        server.close()
        return probed, plan, runs

    try:
        return (tmp, *asyncio.run(go()))
    finally:
        challenger.close()


def test_a_response_whose_last_two_tokens_share_an_event_is_padded_to_the_bodys_tokens(merged):
    """The endpoint's responses to one body of 64 tokens are 63 or 64 events long, as vLLM's are four in flight: the
    plan takes F = 64 from the body, and the shim pads each short response with one empty canonical event after its
    last, so every response it writes is 64 events, the planned profile's."""
    tmp, probed, plan, _ = merged
    assert sorted({r["events"] for r in probed["responses"]}) == [MERGED_TOKENS - 1, MERGED_TOKENS]
    assert plan["pad_to"] == plan["served"]["tokens"] == MERGED_TOKENS
    assert plan["served"]["probe_events"] == [MERGED_TOKENS - 1, MERGED_TOKENS]
    assert sum(map(sum, plan["occupancy"]["egress"])) == REQUESTS * MERGED_TOKENS
    server = json.loads((tmp / "honest" / "server.json").read_text())
    assert server["events"] == REQUESTS * MERGED_TOKENS
    padded = dict(server["padded"])
    assert sorted(padded.values()) == [1] * (REQUESTS // 2)
    carried: dict = defaultdict(list)
    for _, name, _, _, e in sorted(ar.read_events(tmp / "honest" / "server.events.gz")):
        carried[name].append(active.event_data(active.event_body(e)))
    assert len(carried) == REQUESTS and all(len(c) == MERGED_TOKENS for c in carried.values())
    for name, c in carried.items():
        assert c[-2:] == ([b" t62 t63", b""] if name in padded else [b" t62", b" t63"])


def test_honest_traffic_with_merged_events_is_accepted_on_every_window_and_late_and_extra_are_refused(merged):
    """Padded to F, honest traffic is accepted on every link-window, N2 on the planned profile among it; the late shim
    is refused on its window-0 egress record, the extra one (F + 1 events in its first response) on N2's counts alone,
    and each window's salt is revealed once in every run."""
    *_, runs = merged
    honest = runs["honest"]["check"]
    assert honest["verdict"] == "accepted", cs.table(honest)
    assert all(r["n2"]["accepted"] for r in honest["rows"]) and honest["network_verdict"]["refused"] == []
    assert honest["timing"]["server_padded"] == {"responses": REQUESTS // 2, "events": REQUESTS // 2}
    late = runs["late"]["check"]
    assert late["verdict"] == "refused" and 0 in late["network_verdict"]["refused"]
    assert late["rows"][0]["audit"]["egress"]["raw"] == "status:coverage-never-established"
    extra = runs["extra"]["check"]
    assert runs["extra"]["events"] == REQUESTS * MERGED_TOKENS + 1
    assert extra["verdict"] == "refused" and extra["network_verdict"]["refused"] == [0] and extra["audit_accepted"]
    for strategy in cs.STRATEGIES:
        check = runs[strategy]["check"]
        assert all(r["salt"] == "revealed" for r in check["rows"]) and check["salt_again"].startswith("refused")
        assert check["salt_accepted"]
