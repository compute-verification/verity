"""certifier-serve on an injected clock, end to end: the served plan probed from a stub OpenAI streaming endpoint, the
certifier (`active.Proxy` with its salt on, its records certified and its reveals signed) in front of it behind the
shim, and the offline check of what it published, which asks each window's salt through a `RemoteCertifier` signed
with the challenger's key. Honest traffic is accepted on every link-window (N2 on the planned profile, the audit and
the salt), a shim that holds one session past every sync or writes one event more than the endpoint sent is refused,
and each window's salt is revealed once. The same holds four in flight against an endpoint that sends some responses'
last two tokens in one event, as vLLM does: the plan's F is the body's tokens, and the shim pads a short response. On
N2's served profile (`--profile served`) the plan fixes no egress counts: the proxy admits requests by NET-54's rule,
a group sent in a window's first bucket goes behind the salt frame, and N2's served counts accept honest traffic on
every window and refuse the extra event. Against an endpoint that streams GLM-5.3's tokens (multi-byte, and tokens
that split a character's UTF-8, which vLLM sends as chunks of no text) both profiles accept honest traffic with F the
body's tokens, and a token of more than 64 bytes needs F declared (`--response-frames`)."""

from __future__ import annotations

import argparse
import asyncio
import codecs
import importlib.util
import itertools
import json
import re
import shutil
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
    v1's output collector merges a request's deltas the API server hasn't read yet. Token k's bytes are `token(k)`,
    detokenized as vLLM's detokenizer does: bytes that end inside a UTF-8 character are held back, so their chunk has
    no text and the next carries them, and an incomplete character at the end is never sent; the text goes in the
    chunk as UTF-8, not escaped, as vLLM's JSON writes it."""

    def __init__(self, group: int, merges=lambda request: False, token=lambda k: b" t%d" % k) -> None:
        self.group, self.arrived, self.writes, self.bodies, self.merges = group, 0, [], [], merges
        self.token = token
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
        text, n, detokenize = "", body["max_tokens"], codecs.getincrementaldecoder("utf-8")()
        for k in range(n):
            await self.turns[(g, k, i)].wait()
            text += detokenize.decode(self.token(k))
            if not (k == n - 2 and self.merges(me)):
                chunk = json.dumps({"choices": [{"text": text}]}, ensure_ascii=False, separators=(",", ":"))
                p, text = b"data: %s\n\n" % chunk.encode(), ""
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

    out["published"] = []

    class Recording(cs.CertifyingPublisher):
        def keep(self, record, params) -> None:
            out["published"].append((record, params))
            super().keep(record, params)

    publish = Recording(run, setup, Issuer(certifier, SESSION), cs.N2_LINK, threaded=False)
    ar.run_virtual(shim, report, run, upstream, salt=True, publish=publish, after=after,
                   proxy_args={"challenger_key": challenger.public_key(), "signer": certifier,
                               "served": cs.served_profile(plan)})
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


def test_the_gates_certifier_rehearsal_parses_here_with_its_sampling(tmp_path):
    """`certifier-rehearse`'s command (`benchmarks/live_challenger/auditor.py`) is a `rehearse` this parser takes, its
    temperature and top_p (0 and 1 by default) always among it, and they are the body every request sends."""
    sys.path.insert(0, str(HERE.parent / "live_challenger"))
    import auditor as A
    req = {"requests": 8, "tamper": None, "profile": "served", "in_flight": 2, "tokens": 32, "temperature": 0.6,
           "top_p": 0.9}
    args = cs.parser().parse_args(A.certifier_command(req, tmp_path, "http://127.0.0.1:40001")[2:])
    assert (args.command, args.upstream, args.requests, args.in_flight, args.tokens, args.temperature, args.top_p,
            args.profile, args.strategies) == ("rehearse", "http://127.0.0.1:40001", 8, 2, 32, 0.6, 0.9, "served",
                                               "honest")
    body = cs.request_body("m", args.tokens, cs.PROMPT, args.temperature, args.top_p)
    assert (body["temperature"], body["top_p"]) == (0.6, 0.9)
    plain = cs.parser().parse_args(A.certifier_command({"requests": 8, "tamper": None}, tmp_path, "u")[2:])
    assert (plain.temperature, plain.top_p) == (0.0, 1.0)
    assert {k: cs.request_body("m", 16)[k] for k in ("temperature", "top_p")} == {"temperature": 0.0, "top_p": 1.0}


def test_four_in_flight_the_endpoints_shape_is_sent_and_the_shim_writes_request_major(tmp_path):
    """The plan sends the endpoint's shape (its one model, a prompt of exactly 128 of its tokens, `min_tokens` =
    `max_tokens`, `ignore_eos`, streamed, the given temperature and top_p on every request, the probes' and the run's
    alike, so that an endpoint's own generation config never applies) four at a time, the ingress releasing them two a
    bucket, and the
    endpoint interleaves the responses of each two released together chunk by chunk; the shim still writes each
    response whole, in the order their requests reached it, so the egress counts are the planned profile's and honest
    traffic is accepted on every link-window."""
    endpoint = _SideBySide(2)
    args = argparse.Namespace(model=None, api_key_env=None, prompt_tokens=128, tokens=TOKENS, response_frames=None,
                              probes=4, in_flight=4, requests=REQUESTS, delta_buckets=None, profile="planned",
                              temperature=0.6, top_p=0.9)

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
                              "min_tokens": TOKENS, "stream": True, "ignore_eos": True, "temperature": 0.6,
                              "top_p": 0.9}
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


def test_the_certifier_process_keeps_and_signs_as_the_inline_publisher_does(served, tmp_path):
    """`serve`'s `CertifierProcess`, on a copy of one key, writes the honest run's records and commitments byte for
    byte as `CertifyingPublisher` inline does, and the same heads, each certificate at the same leaf and valid under its
    own fresh salt; a reveal signed between two records takes the same leaf: its calls run in the order they are
    made."""
    import certifier_serve  # by name, so that the spawned process finds its functions

    _, report, _, runs = served
    published = runs["honest"]["published"]
    cs.load_key(tmp_path / "inline", 6)
    shutil.copytree(tmp_path / "inline", tmp_path / "process")
    (tmp_path / "inline-run").mkdir()
    (tmp_path / "process-run").mkdir()
    own = cs.signer(tmp_path / "inline")
    inline = cs.CertifyingPublisher(tmp_path / "inline-run", ar.Setup(report), Issuer(own, SESSION), cs.N2_LINK,
                                    threaded=False)
    process = certifier_serve.CertifierProcess(tmp_path / "process", tmp_path / "process-run", report, SESSION,
                                               cs.N2_LINK)
    try:
        signed = []
        for k, (record, params) in enumerate(published):
            if k == len(published) // 2:
                signed.append((own.sign(b"reveal"), process.sign(b"reveal")))
            inline(record, params)
            process(record, params)
        process.close()
        assert process.refused()[0] == inline.issuer.refused == []
        assert process.public_key() == own.public_key()
    finally:
        process.shutdown()
        own.close()
    assert len(published) == 4 and signed[0][0] == signed[0][1]
    for name in ("requests.records", "responses.records", "committed.jsonl"):
        assert (tmp_path / "process-run" / name).read_bytes() == (tmp_path / "inline-run" / name).read_bytes(), name
    inline_docs, process_docs = ([json.loads(line) for line in
                                  (tmp_path / run / "certified.jsonl").read_text().splitlines()]
                                 for run in ("inline-run", "process-run"))
    assert [d["head"] for d in process_docs] == [d["head"] for d in inline_docs]
    leaves = [[bytes.fromhex(d["certificate"]["signature"])[:4] for d in docs] for docs in (inline_docs, process_docs)]
    assert leaves[0] == leaves[1] and len(set(leaves[0])) == len(published)
    assert all(cs.certified_from_json(d).verify(process.public_key(), SESSION) for d in process_docs)


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
    _, plan = cs.served_plan(probed(TOKENS, TOKENS + 1), 4, delta_buckets=20, frames=TOKENS + 1)
    assert plan["pad_to"] == plan["served"]["response_frames"] == TOKENS + 1 and plan["served"]["tokens"] == TOKENS
    assert sum(map(sum, plan["occupancy"]["egress"])) == 4 * (TOKENS + 1)
    with pytest.raises(ValueError, match="at least the body's"):
        cs.served_plan(probed(TOKENS), 4, delta_buckets=20, frames=TOKENS - 1)


MERGED_TOKENS = 64


def test_a_latency_budget_that_doesnt_fit_a_window_is_cut_to_the_most_that_does():
    """vLLM's endpoint took up to 1.7 s for 64 tokens four in flight: four times that and half a second is 73 buckets,
    and four such responses don't fit in a window of 60, so the plan takes the largest D that does, 52. At 56 they fit
    from a later window's bucket 1 but not from window 0's bucket 2, which would leave window 0 without a session for
    `late` to hold, so 56 doesn't fit. A plan is refused only when twice the slowest and half a second doesn't fit
    either."""
    body = cs.request_body("stub", MERGED_TOKENS)

    def probed(slowest):
        return {"url": "stub", "body": body,
                "responses": [{"events": MERGED_TOKENS, "first_s": 0.0, "last_s": slowest}]}

    report, plan = cs.served_plan(probed(1.7), 40, in_flight=4)
    assert plan["served"]["delta_buckets"] == 52 and plan["windows"] == 11
    assert [q["enqueue"] for q in plan["requests"]][::4] == [2] + [60 * w + 1 for w in range(1, 10)]
    assert ar.adversary_plan(plan, ar.Setup(report), "late")["writes"]
    with pytest.raises(cs.DoesntFit, match="window 0 from its bucket 2"):
        cs.served_plan(probed(1.7), 40, delta_buckets=56, in_flight=4)
    assert cs.served_plan(probed(0.6), 40, in_flight=4)[1]["served"]["delta_buckets"] == 29
    with pytest.raises(cs.DoesntFit, match=r"D of 65 … 125 buckets"):
        cs.served_plan(probed(3.0), 40, in_flight=4)


@pytest.fixture(scope="module")
def merged(tmp_path_factory):
    """The served plan probed and run four in flight, as the gate's rehearsal runs, against an endpoint that sends
    every other response's last two tokens in one chunk: honest, late and extra, each checked."""
    tmp = tmp_path_factory.mktemp("merged")
    endpoint = _SideBySide(2, merges=lambda request: request % 2 == 1)
    args = argparse.Namespace(model=None, api_key_env=None, prompt_tokens=128, tokens=MERGED_TOKENS,
                              response_frames=None, probes=4, in_flight=4, requests=REQUESTS, delta_buckets=None,
                              profile="planned", temperature=0.0, top_p=1.0)
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


@pytest.fixture(scope="module")
def served_profile(tmp_path_factory):
    """The gate's shape on N2's served profile (`plan --profile served`): four in flight, 64 tokens, every other
    response's last two tokens in one chunk, at D = 30, so the second group doesn't fit in window 0 and is sent in
    window 1's first bucket; honest, late and extra, each checked. The endpoint answers each request as it arrives:
    the salt frame puts window 1's requests a bucket apart, and the injected clock moves only once every request the
    server holds is answered, so an endpoint that held one for its partner would never answer it."""
    tmp = tmp_path_factory.mktemp("served-profile")
    endpoint = _SideBySide(1, merges=lambda request: request % 2 == 1)
    args = argparse.Namespace(model=None, api_key_env=None, prompt_tokens=128, tokens=MERGED_TOKENS,
                              response_frames=None, probes=4, in_flight=4, requests=REQUESTS, delta_buckets=30,
                              profile="served", temperature=0.0, top_p=1.0)
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
        return report, plan, runs

    try:
        return (tmp, *asyncio.run(go()))
    finally:
        challenger.close()


def test_the_served_profile_plans_no_egress_occupancy_and_the_proxy_admits_by_the_served_rule(served_profile):
    """On the served profile the plan fixes no egress counts: its occupancy is F, F_in and the salt frame, and its
    model is the proxy's admission rule, under which a group sent in a window's first bucket is released behind the
    salt frame and a group that wouldn't end its window idle is held for the next window's first bucket."""
    _, report, plan, runs = served_profile
    T = cs.WINDOW_BUCKETS
    assert plan["occupancy"] == {"kind": "served", "egress": None, "ingress": None, "response_frames": MERGED_TOKENS,
                                 "request_frames": 1, "own": 1}
    assert cs.served_profile(plan) == cs.ServedProfile(MERGED_TOKENS, 1, plan["windows"], 1)
    assert [q["enqueue"] for q in plan["requests"]] == [2] * 4 + [T] * 4 and plan["windows"] == 3
    assert [plan["model_release"][str(q["stream"])] for q in plan["requests"]] == [2, 2, 3, 3, T, T + 1, T + 1, T + 2]
    assert runs["honest"]["check"]["comparisons"]["releases_off_model"] == 0

    setup = ar.Setup(report)
    e, i = setup.egress_params, setup.ingress_params
    admission = cs.Admission(cs.served_profile(plan), e.allocation, lambda w: setup.envelope)
    late = [q | {"enqueue": T - 10} if q["enqueue"] == T else q for q in plan["requests"]]
    assert ar.model_releases(late, i, plan["windows"], e.bucket_ms) == {1: 2, 2: 2, 3: 3, 4: 3, 5: 50, 6: 50, 7: 51,
                                                                        8: 51}
    assert ar.model_releases(late, i, plan["windows"], e.bucket_ms, admission) == \
        {int(s): r for s, r in plan["model_release"].items()}


def test_honest_is_accepted_on_every_link_window_under_the_served_profile(served_profile):
    """N2's verdict of record accepts every link-window on the served profile, each window's egress counts the ones its
    ingress counts derive (`served_counts`), and `network_verdict` accepts the link; the audit and the salt accept."""
    *_, runs = served_profile
    check = runs["honest"]["check"]
    assert check["verdict"] == "accepted", cs.table(check)
    assert check["profile"] == "served" and check["network_verdict"]["refused"] == []
    assert all(r["n2"]["accepted"] and r["n2"]["public"] == {"verdict": 1, "deadline": 1} and r["served_counts"]
               for r in check["rows"])
    assert all(r["audit"] == {d: {"raw": None, "committed": None} for d in ("egress", "ingress")} for r in check["rows"])
    assert check["salt"]["accepted"] and check["salt"]["ingress_salt_frames"] == [1, 1, 1]
    assert check["comparisons"]["wire_is_the_records"] and check["timing"]["server_padded"]["responses"] == 4


def test_a_request_in_a_windows_first_bucket_is_released_behind_the_salt_frame_and_its_window_derives(served_profile):
    """Window 1's group reaches the proxy in that window's first bucket, the salt frame's: the bucketer enqueues the
    salt frame first and the group behind it, two a bucket, and the served counts take the salt frame off bucket 0, so
    the window's egress counts derive from its ingress counts and N2 accepts it."""
    tmp, report, plan, runs = served_profile
    setup = ar.Setup(report)
    records = sorted((ar.unpack_record(b, cs.FRAME_BYTES) for b in ar.read_blobs(tmp / "honest" / "requests.records")),
                     key=lambda r: r.window)
    first = records[1]
    kinds = [[active.decode_frame(f)[0] for f in row[:active.row_count(row)]] for row in first.grid[:4]]
    assert kinds == [[active.SALT, active.DATA], [active.DATA, active.DATA], [active.DATA], []]
    assert setup.ingress_params.allocation == 2
    assert runs["honest"]["check"]["rows"][1]["served_counts"] and runs["honest"]["check"]["rows"][1]["n2"]["accepted"]


def test_late_is_refused_and_extra_is_refused_by_the_served_counts(served_profile):
    """The late shim's window-0 egress record is not established: the audit refuses it and N2 rejects the window. The
    extra shim's first response is F + 1 events, on time and logged, so every record is complete and the audit accepts
    it: only the served counts refuse window 0, whose egress counts are one more than its ingress counts derive."""
    *_, runs = served_profile
    late = runs["late"]["check"]
    assert late["verdict"] == "refused" and 0 in late["network_verdict"]["refused"]
    assert late["rows"][0]["n2"] == {"accepted": False, "codes": ["reject"], "public": {}}
    assert late["rows"][0]["audit"]["egress"]["raw"] == "status:coverage-never-established"
    assert runs["extra"]["events"] == REQUESTS * MERGED_TOKENS + 1
    extra = runs["extra"]["check"]
    assert extra["verdict"] == "refused" and extra["network_verdict"]["refused"] == [0]
    assert extra["audit_accepted"] and extra["salt_accepted"] and extra["comparisons"]["wire_is_the_records"]
    assert [r["served_counts"] for r in extra["rows"]] == [False, True, True]
    assert extra["rows"][0]["n2"] == {"accepted": False, "codes": ["reject"], "public": {}}
    assert all(r["n2"]["accepted"] for r in extra["rows"][1:])


def test_each_salt_is_revealed_once_under_the_served_profile(served_profile):
    *_, runs = served_profile
    for strategy in cs.STRATEGIES:
        check = runs[strategy]["check"]
        assert [r["salt"] for r in check["rows"]] == ["revealed"] * 3
        assert check["salt_again"].startswith("refused: the certifier refused window 0's salt")
        assert runs[strategy]["revealed"] == [0, 1, 2]
        assert len({r["salt_fingerprint"] for r in check["rows"]}) == 3


#: Tokens of GLM-5.3's tokenizer (zai-org/GLM-5.3 at aca966e4, their ids beside them): words and punctuation of two
#: and three bytes a character, and tokens that split a character's UTF-8, as 🦙 (three tokens) and 龘 (two) are split;
#: the tokenizer has 1095 tokens that aren't whole UTF-8.
GLM_TOKENS = ("数据".encode(),  # 99080
              " 计算".encode(),  # 118648
              "，".encode(),  # 3837
              b"\xf0\x9f", b"\xa6", b"\x99",  # 9281, 99, 247: 🦙
              "模型".encode(),  # 100484
              b"\xe9\xbe", b"\x98",  # 82225, 246: 龘
              "。".encode(),  # 1773
              b" GL", b"M")  # 5588, 44
#: GLM-5.3's token 96598, 80 spaces: one of its 124 tokens of more than `active.TOKEN_BYTES` (64) bytes
LONG_TOKEN = b" " * 80


def _glm_token(k: int) -> bytes:
    return GLM_TOKENS[k % len(GLM_TOKENS)]


def _rehearse(tmp: Path, endpoint, args, strategies=cs.STRATEGIES) -> tuple[dict, dict, dict, dict]:
    """(probe, calibration, plan, runs): `endpoint` on a port, planned by `plan_of` with `args`, then each of
    `strategies` run on the injected clock and checked."""
    challenger = cs.signer(tmp / "keys" / "challenger", 6)

    async def go():
        server = await asyncio.start_server(endpoint, "127.0.0.1", 0)
        upstream = server.sockets[0].getsockname()[:2]
        url = "http://%s:%d" % upstream
        try:
            probed, report, plan = await asyncio.to_thread(cs.plan_of, args, url, lambda: 0.0)
            runs = {}
            for strategy in strategies:
                certifier = cs.signer(tmp / "keys" / strategy, 6)
                try:
                    runs[strategy] = await asyncio.to_thread(_run, tmp / strategy, plan, report, url, upstream,
                                                             certifier, challenger, strategy)
                finally:
                    certifier.close()
        finally:
            server.close()
        return probed, report, plan, runs

    try:
        return asyncio.run(go())
    finally:
        challenger.close()


def _glm_args(profile: str, response_frames: int | None = None) -> argparse.Namespace:
    return argparse.Namespace(model=None, api_key_env=None, prompt_tokens=128, tokens=MERGED_TOKENS,
                              response_frames=response_frames, probes=4, in_flight=4, requests=REQUESTS,
                              delta_buckets=30, profile=profile, temperature=0.0, top_p=1.0)


@pytest.fixture(scope="module", params=cs.PROFILES)
def glm(request, tmp_path_factory):
    """The gate's shape (four in flight, 64 tokens, every other response's last two tokens in one chunk, D = 30) on
    each of N2's profiles, against an endpoint that streams GLM-5.3's tokens: honest, late and extra, each checked."""
    tmp = tmp_path_factory.mktemp(f"glm-{request.param}")
    endpoint = _SideBySide(1, merges=lambda r: r % 2 == 1, token=_glm_token)
    return (tmp, *_rehearse(tmp, endpoint, _glm_args(request.param)))


def test_a_glm_token_is_one_event_and_the_shim_carries_the_text_byte_for_byte(glm):
    """F is still the body's tokens: a token that ends inside a character is a chunk of no text, so an empty event,
    and the next event carries the character whole; each response's events, joined, are the UTF-8 text the endpoint
    sent, byte for byte."""
    tmp, probed, _, plan, _ = glm
    assert sorted({r["events"] for r in probed["responses"]}) == [MERGED_TOKENS - 1, MERGED_TOKENS]
    assert plan["pad_to"] == plan["served"]["response_frames"] == plan["served"]["tokens"] == MERGED_TOKENS
    detokenize = codecs.getincrementaldecoder("utf-8")()
    text = "".join(detokenize.decode(_glm_token(k)) for k in range(MERGED_TOKENS)).encode()
    carried: dict = defaultdict(list)
    for _, name, _, _, e in sorted(ar.read_events(tmp / "honest" / "server.events.gz")):
        carried[name].append(active.event_data(active.event_body(e)))
    assert len(carried) == REQUESTS
    for c in carried.values():
        assert len(c) == MERGED_TOKENS and b"".join(c) == text
        assert c[:10] == [w.encode() for w in ("数据", " 计算", "，", "", "", "🦙", "模型", "", "龘", "。")]


def test_honest_glm_traffic_is_accepted_and_late_and_extra_are_refused(glm):
    """On each of N2's profiles, honest GLM traffic is accepted on every link-window, the late shim is refused on its
    window-0 egress record, the extra one on N2's counts alone, and each window's salt is revealed once."""
    *_, plan, runs = glm
    honest = runs["honest"]["check"]
    assert honest["verdict"] == "accepted", cs.table(honest)
    assert honest["profile"] == plan["occupancy"]["kind"] and honest["network_verdict"]["refused"] == []
    assert all(r["n2"]["accepted"] for r in honest["rows"])
    late = runs["late"]["check"]
    assert late["verdict"] == "refused" and 0 in late["network_verdict"]["refused"]
    assert late["rows"][0]["audit"]["egress"]["raw"] == "status:coverage-never-established"
    extra = runs["extra"]["check"]
    assert extra["verdict"] == "refused" and extra["network_verdict"]["refused"] == [0] and extra["audit_accepted"]
    for strategy in cs.STRATEGIES:
        check = runs[strategy]["check"]
        assert all(r["salt"] == "revealed" for r in check["rows"]) and check["salt_again"].startswith("refused")
        assert runs[strategy]["revealed"] == list(range(plan["windows"]))


def test_a_glm_token_longer_than_an_event_needs_a_declared_f(tmp_path):
    """GLM-5.3's token of 80 spaces is two events, so a response that holds one is one event longer than its tokens:
    the default plan refuses it, naming `--response-frames`; with F declared one more, every response is padded to it,
    and on the served profile honest traffic is accepted on every window and the extra event refused."""
    endpoint = _SideBySide(1, merges=lambda r: r % 2 == 1, token=lambda k: LONG_TOKEN if k == 10 else _glm_token(k))
    with pytest.raises(ValueError, match=r"\[64, 65\] events long.*--response-frames"):
        _rehearse(tmp_path / "default", endpoint, _glm_args("served"), strategies=())
    tmp = tmp_path / "declared"
    probed, _, plan, runs = _rehearse(tmp, endpoint, _glm_args("served", MERGED_TOKENS + 1), ("honest", "extra"))
    assert sorted({r["events"] for r in probed["responses"]}) == [MERGED_TOKENS, MERGED_TOKENS + 1]
    assert plan["pad_to"] == plan["occupancy"]["response_frames"] == MERGED_TOKENS + 1
    carried: dict = defaultdict(list)
    for _, name, _, _, e in sorted(ar.read_events(tmp / "honest" / "server.events.gz")):
        carried[name].append(active.event_data(active.event_body(e)))
    assert all(len(c) == MERGED_TOKENS + 1 and c[10:12] == [b" " * 64, b" " * 16] for c in carried.values())
    honest, extra = runs["honest"]["check"], runs["extra"]["check"]
    assert honest["verdict"] == "accepted", cs.table(honest)
    assert all(r["n2"]["accepted"] and r["served_counts"] for r in honest["rows"])
    assert extra["verdict"] == "refused" and extra["network_verdict"]["refused"] == [0] and extra["audit_accepted"]
