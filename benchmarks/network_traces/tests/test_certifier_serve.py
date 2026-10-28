"""certifier-serve on an injected clock, end to end: the served plan probed from a stub OpenAI streaming endpoint, the
certifier (`active.Proxy` with its salt on, its records certified and its reveals signed) in front of it behind the
shim, and the offline check of what it published, which asks each window's salt through a `RemoteCertifier` signed
with the challenger's key. Honest traffic is accepted on every link-window (N2 on the planned profile, the audit and
the salt), a shim that holds one session past every sync or writes one event more than the endpoint sent is refused,
and each window's salt is revealed once. The same holds four in flight against an endpoint that sends some responses'
last two tokens in one event, as vLLM does: the plan's F is the body's tokens, and the shim pads a short response. On
N2's served profile (`--profile served`) the plan fixes no egress counts: the proxy admits requests by the served profile's rule,
a group sent in a window's first bucket goes behind the salt frame, and N2's served counts accept honest traffic on
every window and refuse the extra event. Against an endpoint that streams GLM-5.3's tokens (multi-byte, and tokens
that split a character's UTF-8, which vLLM sends as chunks of no text) both profiles accept honest traffic with F the
body's tokens, and a token of more than 64 bytes needs F declared (`--response-frames`). On varied traffic
(`--traffic varied`: Poisson arrivals, prompts of varied lengths as token ids, varied `max_tokens`, the model's own
EOS) the shim writes one id event a sampled token, tagged with its request's ingress ordinal and interleaved as the
endpoint streams them, and the certified I/O read from the records alone gives back every request's fields and every
response's token ids: honest traffic is certified on every window and its I/O accepted, the late shim is refused by
the audit and the extra token by the I/O's finish rule. The I/O tie on those runs (`negatives.py tie-run`), given the
endpoint's sampled ids, accepts honest traffic and refuses the token shim by its content."""

from __future__ import annotations

import argparse
import asyncio
import codecs
import importlib.util
import itertools
import json
import math
import re
import shutil
import sys
import tarfile
from collections import defaultdict
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
from verity.core.protocols.certifier import active
from verity.core.protocols.certifier.certificate import Issuer
from verity.core.protocols.certifier.release import release_of

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
#: the release order (`release.release_of`) of every request `_endpoint` received, in arrival order
RELEASES: list = []


async def _endpoint(reader, writer):
    """A streaming completions endpoint that answers at once: `max_tokens` chunk events and `[DONE]`, chunked."""
    head = await reader.readuntil(b"\r\n\r\n")
    body = json.loads(await reader.readexactly(int(re.search(rb"content-length: (\d+)", head, re.I).group(1))))
    assert head.startswith(b"POST /v1/completions ") and "user" not in body and body["ignore_eos"]
    RELEASES.append(release_of(head))
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
    chunk as UTF-8, not escaped, as vLLM's JSON writes it. `releases` is each completion's release order
    (`release.release_of`) as it arrived."""

    def __init__(self, group: int, merges=lambda request: False, token=lambda k: b" t%d" % k) -> None:
        self.group, self.arrived, self.writes, self.bodies, self.merges = group, 0, [], [], merges
        self.token, self.releases = token, []
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
        self.releases.append(release_of(head))
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
        def keep(self, record, params, sync=None) -> None:
            out["published"].append((record, params))
            super().keep(record, params, sync)

    publish = Recording(run, setup, Issuer(certifier, SESSION), cs.N2_LINK, threaded=False)
    ar.run_virtual(shim, report, run, upstream, salt=True, publish=publish, after=after,
                   proxy_args={"challenger_key": challenger.public_key(), "signer": certifier,
                               "served": cs.served_profile(plan)})
    server = json.loads((run / "server.json").read_text())
    out["events"], out["released"], out["run"] = server["events"], dict(server["released"]), run
    return out


def _released_as_signed(out: dict) -> list[int]:
    """The release order of a run of `_run`'s, as the endpoint got it: the shim gave each session N, the stream the
    signed ingress records released it as less one (`ar.released_sessions`), and the endpoint got each N once,
    0 … n − 1, under the run id `deployment.json` names; no ingress frame holds a release header. Each N in arrival
    order."""
    run = out["run"]
    records = [ar.unpack_record(b, cs.FRAME_BYTES) for b in ar.read_blobs(run / "requests.records")]
    signed = {name: stream - 1 for name, stream in ar.released_sessions(records).items()}
    run_id = bytes.fromhex(json.loads((run / "deployment.json").read_text())["run_id"])
    assert out["released"] == signed and sorted(signed.values()) == list(range(len(signed)))
    assert sorted(out["releases"]) == [(n, run_id) for n in range(len(signed))]
    frames = b"".join(f for r in records for row in r.grid for f in row[:active.row_count(row)])
    assert b"x-verity-release" not in frames.lower()
    return [n for n, _ in out["releases"]]


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
            RELEASES.clear()
            try:
                runs[strategy] = await asyncio.to_thread(_run, tmp / strategy, plan, report, url, upstream, certifier,
                                                         challenger, strategy)
            finally:
                certifier.close()
            runs[strategy]["releases"] = list(RELEASES)
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
    window 0 alone. The release order skips the dropped request: the endpoint gets N = 0, 1, 2."""
    async def go():
        endpoint = await asyncio.start_server(_endpoint, "127.0.0.1", 0)
        upstream = endpoint.sockets[0].getsockname()[:2]
        probed = await asyncio.to_thread(cs.probe, "http://%s:%d" % upstream, cs.request_body("stub", TOKENS), 1,
                                         clock=lambda: 0.0)
        report, plan = cs.served_plan(probed, 4, delta_buckets=20)
        RELEASES.clear()
        await asyncio.to_thread(ar.run_virtual, plan, report, tmp_path, upstream, stalls=[(1, 0.11)])
        endpoint.close()
        return report, plan, list(RELEASES)

    report, plan, releases = asyncio.run(go())
    records = [ar.unpack_record(b, cs.FRAME_BYTES) for b in ar.read_blobs(tmp_path / "requests.records")]
    assert ar.released_sessions(records) == {"served.0": 1, "served.2": 2, "served.3": 3}
    released = dict(json.loads((tmp_path / "server.json").read_text())["released"])
    assert released == {"served.0": 0, "served.2": 1, "served.3": 2}
    assert [n for n, _ in releases] == [0, 1, 2] and len({run for _, run in releases}) == 1
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


def test_the_gates_keys_height_reaches_rehearse_and_a_key_above_this_files_is_made_in_its_workers(tmp_path,
                                                                                                    monkeypatch):
    """`certifier-rehearse`'s command passes its keys' height, the deployed key's (20) unless the request gives one,
    as `rehearse`'s `--height`, which makes both keys at it; `load_key` makes a key above `HEIGHT` in
    `KEYGEN_WORKERS` processes (at 20, minutes, not the quarter hour one process takes) and one at or below it in one,
    and either loads back from its files."""
    sys.path.insert(0, str(HERE.parent / "live_challenger"))
    import auditor as A
    plain = cs.parser().parse_args(A.certifier_command({"requests": 8, "tamper": None}, tmp_path, "u")[2:])
    given = cs.parser().parse_args(A.certifier_command({"requests": 8, "tamper": None, "height": 10}, tmp_path,
                                                       "u")[2:])
    assert (plain.height, given.height) == (cs.lms.DEPLOYED.H, 10) and cs.KEYGEN_WORKERS == 8
    keygen, workers = cs.lms.keygen, []
    monkeypatch.setattr(cs.lms, "keygen", lambda params, seed, workers=1, seen=workers: (
        seen.append((params.H, workers)), keygen(params, seed, workers))[1])
    for h in (cs.HEIGHT, cs.HEIGHT + 1):
        made = cs.load_key(tmp_path / f"h{h}", h)
        assert cs.load_key(tmp_path / f"h{h}").tree == made.tree and len(made.tree) == 2 ** (h + 1)
    assert workers == [(cs.HEIGHT, 1), (cs.HEIGHT + 1, cs.KEYGEN_WORKERS)]


def test_the_gate_takes_every_tamper_the_rehearsal_has_and_each_is_a_serve_this_parser_takes(tmp_path):
    """The gate's tampers are `serve`'s `--tamper` choices, so the auditor host runs every shim this file has, and
    each reaches `serve` (which alone checks it, after the tunnel is up) as one strategy of `rehearse`."""
    sys.path.insert(0, str(HERE.parent / "live_challenger"))
    import auditor as A
    assert A.TAMPERS == cs.TAMPERS == ("late", "extra", "drop", "token", "byte")
    for t in A.TAMPERS:
        args = cs.parser().parse_args(A.certifier_command({"requests": 8, "tamper": t}, tmp_path, "u")[2:])
        assert args.strategies.split(",") == [t]
        served = cs.parser().parse_args(["serve", "--plan", "p", "--calibration", "c", "--upstream", "u", "--out", "o",
                                         "--link", "127.0.0.1:1", "--reveal", "127.0.0.1:2", "--key", "k",
                                         "--challenger-key", "k.json", "--session", "00", "--tamper", t])
        assert served.tamper == t


def test_the_byte_shim_changes_one_output_byte_of_a_hex_response_and_keeps_it_hex():
    data = bytes([0xa5, 0x00, 0xff]) * 8
    got = cs.ar.other_hex(data.hex().encode())
    changed = bytes.fromhex(got.decode())
    assert changed[0] == 0xb5 and changed[1:] == data[1:]
    assert cs.ar.other_hex(b"\0\0f0") == b"\0\0e0"
    with pytest.raises(ValueError, match="no output byte"):
        cs.ar.other_hex(b"\0\0")
    assert cs.ar.adversary_plan({"sessions": [["s.0", "s.1"]]}, None, "byte")["byte"] == "s.0"


def test_the_gates_varied_rehearsal_is_a_varied_rehearse_named_for_its_run_with_every_flag_it_was_given(tmp_path):
    """The gate's traffic is `rehearse`'s `--traffic`, and a varied request's command is a varied `rehearse` this
    parser takes, named for its run, with each varied flag the request has and no fixed one: without them, the plan's
    own 64, 2 a second, 16:1024, seed 0 and the probe's EOS."""
    sys.path.insert(0, str(HERE.parent / "live_challenger"))
    import auditor as A
    assert A.CERTIFIER_TRAFFIC == cs.TRAFFIC
    req = {"run": "c20261024T120000Z-ab12", "requests": 40, "tamper": "token", "traffic": "varied", "temperature": 0.0,
           "top_p": 1.0, "tokens": 32, "rate": 1.5, "prompt_range": "16:512", "seed": 7, "eos_ids": [128001, 128009]}
    args = cs.parser().parse_args(A.certifier_command(req, tmp_path, "http://127.0.0.1:40001")[2:])
    assert (args.command, args.upstream, args.traffic, args.name, args.requests, args.tokens, args.rate,
            args.prompt_range, args.seed, args.eos_ids, args.strategies) == \
        ("rehearse", "http://127.0.0.1:40001", "varied", req["run"], 40, 32, 1.5, "16:512", 7, "128001,128009",
         "token")
    plain = cs.parser().parse_args(A.certifier_command(
        {k: req[k] for k in ("run", "requests", "traffic", "temperature", "top_p")} | {"tamper": None}, tmp_path,
        "u")[2:])
    assert (plain.tokens, plain.rate, plain.prompt_range, plain.seed, plain.eos_ids, plain.in_flight, plain.profile,
            plain.strategies) == (64, A.CERTIFIER_VARIED_RATE, "16:1024", 0, None, 4, cs.PROFILES[0], "honest")
    assert (plain.margin, plain.prompts_file) == (2.0, None)
    given = cs.parser().parse_args(A.certifier_command(
        {k: req[k] for k in ("run", "requests", "traffic", "temperature", "top_p", "tamper")}
        | {"margin": 4.0, "prompts": {"count": 3, "ids": 4}}, tmp_path / "out", "u")[2:])
    assert (given.margin, given.prompts_file, given.out) == (4.0, tmp_path / A.WITNESS / A.PROMPTS, tmp_path / "out")


def test_four_in_flight_the_endpoints_shape_is_sent_and_the_shim_writes_request_major(tmp_path):
    """The plan sends the endpoint's shape (its one model, a prompt of exactly 128 of its tokens, `max_tokens` and
    `ignore_eos` with no `min_tokens`, streamed, the given temperature and top_p on every request, the probes' and the
    run's alike, so that an endpoint's own generation config never applies) four at a time, the ingress releasing them
    two a bucket, and the
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
                              "stream": True, "ignore_eos": True, "temperature": 0.6, "top_p": 0.9}
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


def test_the_endpoint_gets_each_request_with_the_release_order_the_certifier_signed(served):
    """Every request reaches the endpoint with `x-verity-release: N` and the proxy's run id, N its place in
    the release order the signed ingress records hold, while the records hold the client's request without them, and
    honest traffic is accepted with them present (`test_honest_served_traffic_is_accepted_on_every_link_window`). One
    request in flight, the endpoint gets them in that order."""
    *_, runs = served
    assert runs["honest"]["check"]["verdict"] == "accepted"
    assert _released_as_signed(runs["honest"]) == list(range(REQUESTS))


def test_the_certifier_process_keeps_and_signs_as_the_inline_publisher_does(served, tmp_path):
    """`serve`'s `CertifierProcess`, on a copy of one key, writes the honest run's records and commitments byte for
    byte as `CertifyingPublisher` inline does, and the same heads, each certificate at the same leaf and valid under its
    own fresh salt, and salted clock certificates of the syncs handed with the egress records, the same leaves too,
    each opening under its own fresh salt; a reveal signed between two records takes the same leaf: its calls run in
    the order they are made."""
    import certifier_serve  # by name, so that the spawned process finds its functions

    _, report, _, runs = served
    published = runs["honest"]["published"]
    setup = ar.Setup(report)
    ran = list(setup.sync_set)

    def sync(record):
        return ran[record.window % len(ran)] if record.link == setup.egress_link else None

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
            inline(record, params, sync(record))
            process(record, params, sync(record))
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
    clocks = [cs.read_clocks(tmp_path / run / "clocks.jsonl") for run in ("inline-run", "process-run")]
    egress = [r for r, _ in published if r.link == setup.egress_link]
    assert [[(c.window, c.sync) for c in k] for k in clocks] == [[(r.window, sync(r)) for r in egress]] * 2
    assert [c.certificate.signature[:4] for c in clocks[0]] == [c.certificate.signature[:4] for c in clocks[1]]
    assert all(c.verify(process.public_key(), SESSION) for c in clocks[1])
    assert len({c.salt for k in clocks for c in k}) == 2 * len(egress)


def test_the_served_proxy_hands_its_certifier_each_egress_windows_sync_which_it_signs(tmp_path):
    """`ServedProxy` hands its `publish` the sync its own shaper ran each egress window under, and none with an ingress
    record or past the shaper's syncs; `CertifyingPublisher` signs it salted beside the window's record (`clocks.jsonl`,
    with its opening), and the certifier's key opens exactly those syncs back (`certificate.SignedClocks`)."""
    from verity.core.protocols.certifier.certificate import SignedClocks
    from verity.core.protocols.certifier.grid import Certifier

    probed = {"url": "stub", "body": cs.request_body("stub", TOKENS),
              "responses": [{"events": TOKENS, "first_s": 0.0, "last_s": 0.0}]}
    report, _ = cs.served_plan(probed, 2)
    setup = ar.Setup(report)
    e, i, syncs = setup.egress_params, setup.ingress_params, list(setup.sync_set)[:2]

    def record(link, params, window):
        cert = Certifier(link, params)
        cert.window = window
        for _ in range(params.buckets):
            cert.tick()
        return cert.close_window()

    def proxy(publish):
        return cs.ServedProxy(("127.0.0.1", 9), e, i, setup.sync_set, syncs, origin=0.0, publish=publish,
                              egress_link=setup.egress_link, ingress_link=setup.ingress_link)

    handed = []
    p = proxy(lambda rec, params, sync: handed.append((rec.link, rec.window, sync)))
    for w in (0, 1, 2):
        p.publish(record(setup.egress_link, e, w), e)
    p.publish(record(setup.ingress_link, i, 0), i)
    assert handed == [(setup.egress_link, 0, syncs[0]), (setup.egress_link, 1, syncs[1]), (setup.egress_link, 2, None),
                      (setup.ingress_link, 0, None)]

    signer = cs.signer(tmp_path / "key", 6)
    try:
        publish = cs.CertifyingPublisher(tmp_path, setup, Issuer(signer, SESSION), cs.N2_LINK, threaded=False)
        p = proxy(publish)
        for w in (0, 1):
            p.publish(record(setup.egress_link, e, w), e)
            p.publish(record(setup.ingress_link, i, w), i)
        key = signer.public_key()
    finally:
        signer.close()
    clocks = cs.read_clocks(tmp_path / "clocks.jsonl")
    assert [(c.link, c.window, c.sync) for c in clocks] == [(cs.N2_LINK, w, syncs[w]) for w in (0, 1)]
    signed = SignedClocks(key, SESSION, tuple(clocks))
    assert signed.syncs(cs.N2_LINK) == tuple(syncs) and signed.refused == ()
    assert cs.read_clocks(tmp_path / "absent.jsonl") == []


def _egress_record(window: int, text: bytes, frame_bytes: int = 4096):
    """An egress record whose canonical events carry `text`, 64 bytes an event, then a padding slot and an id event."""
    def frame(kind, body):
        return (active.HEADER.pack(kind, active.CANONICAL_STREAM, len(body)) + body).ljust(frame_bytes, b"\0")

    slots = [frame(active.DATA, active.event(text[i:i + active.TOKEN_BYTES]))
             for i in range(0, len(text), active.TOKEN_BYTES)]
    return SimpleNamespace(window=window, grid=[slots, [None, frame(active.PAD, b""),
                                                        frame(active.DATA, active.id_event(0, 7))]])


def _out_shape(words: int, rows: int, program: str = "cd" * 64):
    from verity.core.protocols.certifier import out_commit as OC

    return OC.OutShape(program, "out", words, rows)


def test_the_network_certifier_commits_and_signs_each_egress_windows_output_in_its_own_session(tmp_path):
    """`--out-shape`'s commitment (`certify_out`): the output an egress record's canonical events carry, as the hex
    of u16 little-endian words, cut into the shape's rows and committed under the running issuer, which signs its head
    in the certifier's session for the link-window: the public record `check_out` pins under the certifier's key, and
    the witness (each row and its salt) apart, for the prover. A record whose text isn't that commits nothing, one
    that isn't the shape's rows is an ``out-shape`` line, and a refused signature is the link-window under
    ``refused``, with no witness."""
    from verity.core.protocols.certifier import out_commit as OC

    words = [0x0102, 0xfffe, 7, 0]
    text = b"".join(w.to_bytes(2, "little") for w in words).hex().encode() * 40
    shape = _out_shape(1, 160)
    signer = cs.signer(tmp_path / "key", 4)
    try:
        issuer, key = Issuer(signer, SESSION), signer.public_key()
        assert cs.out_words(_egress_record(2, text)) == words * 40
        assert cs.out_rows(words * 40, shape) == [[w] for w in words] * 40
        got = cs.certify_out(tmp_path, _egress_record(2, text), issuer, shape)
        assert cs.certify_out(tmp_path, _egress_record(3, b"not hex " * 8), issuer, shape) is None
        assert cs.certify_out(tmp_path, _egress_record(3, b"abc"), issuer, shape) is None
        assert cs.certify_out(tmp_path, _egress_record(4, text), Issuer(signer, bytes(32)), shape) is None
        assert cs.certify_out(tmp_path, _egress_record(5, text[:-4]), issuer, shape) is None
    finally:
        signer.close()
    records = [json.loads(x) for x in (tmp_path / cs.OUT_RECORDS).read_text().splitlines()]
    assert records[0] == got.public and records[1] == {"refused": [OC.LINK, "egress", 4]}
    assert records[2:] == [{"out-shape": [OC.LINK, "egress", 5], "detail": "159 words, not 160 rows of 1"}]
    assert OC.check_out(records[0], key, SESSION, shape, window=2) == got.root
    witness = [json.loads(x) for x in (tmp_path / cs.OUT_WITNESS).read_text().splitlines()]
    assert witness == [{"window": 2, "program": shape.program, "rows": [[w] for w in words] * 40,
                        "salts": [s.hex() for s in got.salts]}]
    assert not any(s.hex() in json.dumps(records) for s in got.salts)


def test_the_out_value_is_committed_at_the_registered_width_in_the_records_order(tmp_path):
    """At a wide shape (the wired toy's: one row of h words a response step) the words go into rows in the record's
    order, ``shape.words`` a row, so response j's step s is row ``j·S + s`` (`out_rows`); the record pins at that
    shape and is ``out-width`` at the per-coordinate one, and words that aren't whole rows of it, or more rows than
    it holds, are ``out-shape``, never padded."""
    from verity.core.protocols.certifier import out_commit as OC

    words = list(range(1, 4 * 8 + 1))
    text = b"".join(w.to_bytes(2, "little") for w in words).hex().encode()
    wide, narrow = _out_shape(8, 4), _out_shape(1, 32)
    assert cs.out_rows(words, wide) == [words[i:i + 8] for i in range(0, 32, 8)]
    assert cs.out_rows(words[:-1], wide) is None and cs.out_rows(words + words[:8], wide) is None
    signer = cs.signer(tmp_path / "key", 4)
    try:
        issuer, key = Issuer(signer, SESSION), signer.public_key()
        got = cs.certify_out(tmp_path, _egress_record(0, text), issuer, wide)
        assert cs.certify_out(tmp_path, _egress_record(1, text + text[:32]), issuer, wide) is None
    finally:
        signer.close()
    records = [json.loads(x) for x in (tmp_path / cs.OUT_RECORDS).read_text().splitlines()]
    assert (got.head.words, got.head.rows) == (8, 4) and OC.check_out(records[0], key, SESSION, wide, window=0) == got.root
    with pytest.raises(OC.OutRefused, match="out-width"):
        OC.check_out(records[0], key, SESSION, narrow, window=0)
    assert records[1] == {"out-shape": [OC.LINK, "egress", 1], "detail": "40 words, not 4 rows of 8"}
    witness = [json.loads(x) for x in (tmp_path / cs.OUT_WITNESS).read_text().splitlines()]
    assert [w["rows"] for w in witness] == [cs.out_rows(words, wide)]


def test_the_out_shape_reaches_each_rehearsed_serve_and_is_an_out_shape(tmp_path, monkeypatch):
    shape = _out_shape(128, 4).doc()
    path = tmp_path / "shape.json"
    path.write_text(json.dumps(shape))
    serve = ["serve", "--plan", "p", "--calibration", "c", "--upstream", "u", "--out", "o", "--link", "h:1",
             "--reveal", "h:2", "--key", "k", "--challenger-key", "c", "--session", "00"]
    assert cs.parser().parse_args(serve).out_shape is None
    assert cs.parser().parse_args(["rehearse", "--out", "o", "--out-shape", str(path)]).out_shape == shape
    for bad in ({**shape, "program": shape["program"].upper()}, {**shape, "words": 0}, {"program": shape["program"]}):
        path.write_text(json.dumps(bad))
        with pytest.raises(SystemExit):
            cs.parser().parse_args([*serve, "--out-shape", str(path)])
    with pytest.raises(SystemExit):
        cs.parser().parse_args([*serve, "--out-shape", str(tmp_path / "absent.json")])
    seen = []

    def popen(argv, **_):
        seen.append(argv)
        raise KeyboardInterrupt

    monkeypatch.setattr(cs.subprocess, "Popen", popen)
    with pytest.raises(KeyboardInterrupt):
        cs.rehearse_one(tmp_path, "honest", url="u", certifier_dir=tmp_path / "k", challenger_dir=tmp_path / "c",
                        api_key_env=None, start_in=1.0, out_shape=shape)
    assert cs.parser().parse_args(seen[0][2:]).out_shape == shape


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
    than F events, or of none, or a body that doesn't ask for exactly its tokens (`ignore_eos`, a `max_tokens` of at
    least 1) or asks for `min_tokens`, which the vLLM integration's advice refuses, has no plan."""
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
    assert "min_tokens" not in body and body["ignore_eos"] is True
    for bad in ({"min_tokens": TOKENS}, {"min_tokens": 1}, {"ignore_eos": False}, {"max_tokens": 0}):
        with pytest.raises(ValueError, match="exactly its tokens"):
            cs.served_plan(probed(TOKENS) | {"body": body | bad}, 4, delta_buckets=20)
    no_eos = {k: v for k, v in body.items() if k != "ignore_eos"}
    with pytest.raises(ValueError, match="exactly its tokens"):
        cs.served_plan(probed(TOKENS) | {"body": no_eos}, 4, delta_buckets=20)
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
            endpoint.releases.clear()
            try:
                runs[strategy] = await asyncio.to_thread(_run, tmp / strategy, plan, report, url, upstream, certifier,
                                                         challenger, strategy)
            finally:
                certifier.close()
            runs[strategy]["releases"] = list(endpoint.releases)
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


def test_four_in_flight_each_request_still_carries_the_release_order_the_certifier_signed(served_profile):
    """Four in flight under the served profile, the endpoint gets every request with N its place in the signed release
    order, in whatever order the shim's connections reach it, and honest traffic is accepted with the headers."""
    *_, runs = served_profile
    assert runs["honest"]["check"]["verdict"] == "accepted"
    assert sorted(_released_as_signed(runs["honest"])) == list(range(REQUESTS))


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


# Varied traffic (`plan --traffic varied`).

EOS, VARIED_TOKENS = 2, 8


def _answer(prompt: list[int], max_tokens: int) -> tuple[list[int], list[str], str]:
    """The stub's response to a prompt, greedy: its token ids, each token's text and its finish reason. It stops at its
    EOS (`EOS`, no text) after 0 … 8 tokens, or at `max_tokens`; unless its prompt's ids sum to 1 mod 3, the text of its
    last token before EOS is 150 bytes, more than an event of text holds, which no id event carries."""
    s = sum(prompt)
    ids = ([100 + (s + k) % 50 for k in range(s % 9)] + [EOS])[:max_tokens]
    texts = ["" if t == EOS else f" w{t}" for t in ids]
    if s % 3 != 1 and ids[0] != EOS:
        texts[max(k for k, t in enumerate(ids) if t != EOS)] = "x" * 150
    return ids, texts, "stop" if ids[-1] == EOS else "length"


class _Varied:
    """An OpenAI endpoint of varied responses (`_answer`), each chunk carrying its token ids (`return_token_ids`, which
    every request must ask for) and the first its prompt's (`prompt_token_ids`), as vLLM's do. `/v1/models` lists one
    model and `/tokenize` makes one id a character, with no BOS. It serves `group` requests (by arrival) side by side:
    each waits for its whole group, then they write one chunk each in turn, a finished response leaving the turn, so
    the shim reads them interleaved (`writes`: (request, chunk) in write order). For the requests `merges` picks, the
    last two tokens share one chunk and the finish reason comes in a chunk of no tokens, as vLLM v1 merges a
    request's deltas the API server hasn't read yet. Each completion it serves is a row of `served` as serve-openai's
    `requests.jsonl` keeps it: its HTTP id, the body's `user` (None without one), its prompt and its streamed ids; and
    of `releases`, its `user` and release order (`release.release_of`)."""

    def __init__(self, group: int = 1, merges=lambda request: False) -> None:
        self.group, self.merges, self.arrived, self.bodies, self.writes = group, merges, 0, [], []
        self.served: list[dict] = []
        self.releases: list[tuple] = []
        self.turns: dict = defaultdict(asyncio.Event)
        self.done: dict = defaultdict(set)

    def _pass(self, g: int, i: int) -> None:
        for j in [*range(i + 1, self.group), *range(i + 1)]:
            if j not in self.done[g]:
                self.turns[(g, j)].set()
                return

    async def __call__(self, reader, writer):
        head = await reader.readuntil(b"\r\n\r\n")
        n = re.search(rb"content-length: (\d+)", head, re.IGNORECASE)
        body = json.loads(await reader.readexactly(int(n.group(1)))) if n else None
        if not head.startswith(b"POST /v1/completions "):
            doc = ({"data": [{"id": "stub-model"}]} if head.startswith(b"GET /v1/models ")
                   else {"tokens": [ord(c) for c in body["prompt"]], "model": body["model"]})
            data = json.dumps(doc).encode()
            writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: %d\r\n\r\n%s"
                         % (len(data), data))
            await writer.drain()
            writer.close()
            return
        assert body["return_token_ids"] is True and not {"ignore_eos", "min_tokens"} & set(body)
        assert isinstance(body.get("user", ""), str)
        assert body["temperature"] == 0 and body["top_p"] == 1 and body["stop"] == [] and type(body["seed"]) is int
        self.bodies.append(body)
        self.releases.append((body.get("user"), release_of(head)))
        me, self.arrived = self.arrived, self.arrived + 1
        g, i = divmod(me, self.group)
        if i == self.group - 1:
            self.turns[(g, 0)].set()
        ids, texts, finish = _answer(body["prompt"], body["max_tokens"])
        self.served.append({"http_id": f"cmpl-{me}", "user": body.get("user"), "prompt_ids": body["prompt"],
                            "streamed_ids": ids})
        chunks = [[k] for k in range(len(ids))]
        merged = self.merges(me) and len(ids) > 1
        if merged:
            chunks[-2:] = [chunks[-2] + chunks[-1]]
        choices = [{"index": 0, "text": "".join(texts[k] for k in ks), "token_ids": [ids[k] for k in ks],
                    "finish_reason": None} for ks in chunks] + ([{"index": 0, "text": "", "token_ids": []}] * merged)
        choices[0]["prompt_token_ids"] = body["prompt"]
        choices[-1]["finish_reason"] = finish
        writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n")
        for c, choice in enumerate(choices):
            await self.turns[(g, i)].wait()
            self.turns[(g, i)].clear()
            p = b"data: %s\n\n" % json.dumps({"object": "text_completion", "choices": [choice]}).encode()
            writer.write(b"%x\r\n%s\r\n" % (len(p), p))
            await writer.drain()
            self.writes.append((me, c))
            if c == len(choices) - 1:
                self.done[g].add(i)
            self._pass(g, i)
        writer.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(b"data: [DONE]\n\n"), b"data: [DONE]\n\n"))
        await writer.drain()
        writer.close()


def _sse(choice: dict) -> bytes:
    return b"data: %s\n\n" % json.dumps({"choices": [choice]}).encode()


def test_a_chunk_is_one_id_event_a_sampled_token_and_none_of_its_text():
    """One id event a sampled token, merged chunks included, none for a chunk of no tokens, and nothing of a chunk's
    text: the ids come from `token_ids`, or from logprobs' `token_id:N` tokens."""
    got = ar.id_events(_sse({"text": "a" * 70, "token_ids": [5, 6], "prompt_token_ids": [1, 2, 3]}), 9)
    got += ar.id_events(_sse({"text": "", "token_ids": []}), 9)
    got += ar.id_events(_sse({"text": "b" * 100, "logprobs": {"tokens": ["token_id:7"]}}), 9)
    assert got == [active.id_event(9, t) for t in (5, 6, 7)]
    assert ar.id_events(b"data: [DONE]\n\n", 9) == [] and ar.chunk_tokens(b": keep-alive\n\n") is None


@pytest.mark.parametrize("length", [1, 16, 500, 513, 1024, 2048])
def test_a_varied_request_is_an_id_request_in_frames_its_prompts_length_fixes(length):
    """A varied request is the id request of its fields and its session, in `id_request_frames` of its prompt's length
    whatever the fields, and its body is what the shim forwards the endpoint, with its `user`."""
    q = {"prompt": [128255] * length, "max_tokens": 64, "seed": 2**32 - 1}
    fields = cs.varied_fields("meta-llama/Llama-3.1-8B-Instruct", q)
    raw = active.id_request(**fields, user="varied.99999", frame_bytes=cs.FRAME_BYTES)
    assert active.request_frame_count(len(raw), cs.FRAME_BYTES) == active.id_request_frames(length, cs.FRAME_BYTES)
    assert active.id_request_fields(raw, cs.FRAME_BYTES) == fields | {"user": "varied.99999"}
    assert json.loads(raw.partition(b"\r\n\r\n")[2]) == cs.varied_body(fields["model"], q) | {"user": "varied.99999"}


def test_given_prompts_are_sent_in_turn_at_the_same_seeds_arrivals_max_tokens_and_seeds(monkeypatch):
    """With `prompts` (`--prompts-file`, the gate's `--prompts`) request j's prompt is the (j mod n)-th of the n and
    nothing is tokenized; its arrival, `max_tokens`, seed and session are what the same seed gives the chat turns."""
    monkeypatch.setattr(cs, "tokenize", lambda url, model, text, key=None: list(text.encode()))
    prompts = [[70001, 3], [5], [7]]
    turns = cs.varied_requests("u", "m", 7, rate=0.2, prompt_tokens=(8, 8), max_tokens=24, seed=5, name="r")
    monkeypatch.setattr(cs, "tokenize", lambda *a, **k: pytest.fail("tokenized given prompts"))
    given = cs.varied_requests("u", "m", 7, rate=0.2, prompt_tokens=(8, 8), max_tokens=24, seed=5, name="r",
                               prompts=prompts)
    assert [q["prompt"] for q in given] == [prompts[j % 3] for j in range(7)]
    assert [{k: v for k, v in q.items() if k != "prompt"} for q in given] == \
        [{k: v for k, v in q.items() if k != "prompt"} for q in turns]
    assert len({tuple(q["prompt"]) for q in turns}) == 1
    for bad in ([], [[1], []]):
        with pytest.raises(ValueError, match="prompts: at least one"):
            cs.varied_requests("u", "m", 7, prompts=bad)


def test_the_varied_plan_takes_the_prompts_file_and_its_timing_widens_with_the_margin(tmp_path, monkeypatch):
    """`plan_of` on varied traffic reads `--prompts-file` for the requests' prompts and plans at `--margin`: a larger
    margin waits longer for each request's first frame (Δ0) and plans a slower rate (ρ0), the plan recording both;
    fixed traffic refuses the file. The probes are the toy engine's: a token every 0.33 s after 0.4 s."""
    (tmp_path / "prompts.json").write_text(json.dumps([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]]))

    def probe(url, model, requests, key, clock, sleep):
        return {"url": url, "model": model, "upstream_fields": cs.UPSTREAM_FIELDS, "responses": [
            {"session": q["session"], "max_tokens": q["max_tokens"], "prompt_tokens": len(q["prompt"]),
             "tokens": q["max_tokens"], "token_s": [0.4 + 0.33 * k for k in range(q["max_tokens"])],
             "finish": "length", "last": 1} for q in requests]}
    monkeypatch.setattr(cs, "probe_varied", probe)
    monkeypatch.setattr(cs, "tokenize", lambda *a, **k: pytest.fail("tokenized given prompts"))
    plans = {}
    for margin in (2.0, 4.0):
        args = argparse.Namespace(model="toy", api_key_env=None, traffic="varied", requests=6, rate=0.2,
                                  prompt_range="16:1024", tokens=24, seed=0, probes=4, eos_ids="0", margin=margin,
                                  prompts_file=tmp_path / "prompts.json")
        _, _, plans[margin] = cs.plan_of(args, "u")
    for margin, plan in plans.items():
        assert [q["id_request"]["prompt"][0] for q in plan["requests"]] == [1, 9, 1, 9, 1, 9]
        assert plan["varied"]["margin"] == margin and plan["varied"]["prompt_tokens"] == [8, 8]
    two, four = plans[2.0]["varied"], plans[4.0]["varied"]
    assert four["delta0_buckets"] > two["delta0_buckets"] and Fraction(four["rho0"]) < Fraction(two["rho0"])
    fixed = argparse.Namespace(model="toy", api_key_env=None, traffic="fixed", prompts_file=tmp_path / "prompts.json")
    with pytest.raises(SystemExit, match="--prompts-file: varied traffic's"):
        cs.plan_of(fixed, "u")


def _no_sleep(seconds: float) -> None:
    pass


def _varied_runs(tmp: Path, endpoint: _Varied, plan_of, strategies) -> tuple:
    """The plan `plan_of(url)` makes against `endpoint`, then each of `strategies` run on it and checked."""
    challenger = cs.signer(tmp / "keys" / "challenger", 6)

    async def go():
        server = await asyncio.start_server(endpoint, "127.0.0.1", 0)
        upstream = server.sockets[0].getsockname()[:2]
        url = "http://{}:{}".format(*upstream)
        probed, report, plan = await asyncio.to_thread(plan_of, url)
        runs = {}
        for strategy in strategies:
            certifier = cs.signer(tmp / "keys" / strategy, 6)
            try:
                runs[strategy] = await asyncio.to_thread(_run, tmp / strategy, plan, report, url, upstream, certifier,
                                                         challenger, strategy)
            finally:
                certifier.close()
        server.close()
        return probed, report, plan, runs

    try:
        return (tmp, *asyncio.run(go()))
    finally:
        challenger.close()


@pytest.fixture(scope="module")
def varied(tmp_path_factory):
    """`plan --traffic varied` against `_Varied`: 16 requests, Poisson at 2 a second, prompts of 16 … 1024 ids, at most
    8 tokens out, 4 probed, every other response's last two tokens in one chunk; honest, late, extra, drop and token,
    each checked."""
    endpoint = _Varied(merges=lambda request: request % 2 == 1)
    args = argparse.Namespace(model=None, api_key_env=None, traffic="varied", requests=16, rate=2.0,
                              prompt_range="16:1024", tokens=VARIED_TOKENS, seed=3, probes=4, eos_ids=None)
    out = _varied_runs(tmp_path_factory.mktemp("varied"), endpoint,
                       lambda url: cs.plan_of(args, url, lambda: 0.0, _no_sleep), (*cs.STRATEGIES, "drop", "token"))
    return (*out, endpoint)


def _io(run: Path) -> dict:
    return {r["user"]: r for r in json.loads((run / "io.json").read_text())["requests"]}


def test_varied_traffic_reaches_the_endpoint_with_n_its_ingress_ordinal_less_one(varied):
    """On varied traffic, whose proxy opens each server connection with the request's ingress ordinal (`ordinals`),
    each request reaches the endpoint with N its ordinal in the certified I/O (`io.json`) less one, under the honest
    run's run id, and the I/O tie accepts with the headers present: they never reach a request frame."""
    tmp, *_, runs, endpoint = varied
    run_id = bytes.fromhex(json.loads((tmp / "honest" / "deployment.json").read_text())["run_id"])
    got = [(user, r[0]) for user, r in endpoint.releases if r is not None and r[1] == run_id]
    io = _io(tmp / "honest")
    assert dict(got) == {name: r["stream"] - 1 for name, r in io.items()} and len(got) == len(io) == 16
    check = runs["honest"]["check"]
    assert check["verdict"] == "accepted" and check["io"]["accepted"] and check["io"]["requests_are_the_plans"]


@pytest.fixture(scope="module", params=[8, 16])
def compact(request, tmp_path_factory):
    """`plan --traffic varied --compact B` against `_Varied`, B = 8 and 16, as `varied` but for the egress framing:
    16 requests, every other response's last two tokens in one chunk and its finish in a chunk of no tokens; honest,
    late and extra, each checked."""
    endpoint = _Varied(merges=lambda r: r % 2 == 1)
    args = argparse.Namespace(model=None, api_key_env=None, traffic="varied", requests=16, rate=2.0,
                              prompt_range="16:1024", tokens=VARIED_TOKENS, seed=3, probes=4, eos_ids=None,
                              compact=request.param)
    out = _varied_runs(tmp_path_factory.mktemp(f"compact{request.param}"), endpoint,
                       lambda url: cs.plan_of(args, url, lambda: 0.0, _no_sleep), cs.STRATEGIES)
    return (request.param, *out, endpoint)


def test_compact_varied_traffic_is_certified_its_io_the_endpoints_and_its_egress_b_bytes_a_frame(compact, varied):
    """On compact records in B-byte cells the honest run is accepted as on id events: every link-window certified, its
    salt revealed, the audit accepting raw and committed. The certified I/O read from the cells is the id-event run's,
    request for request (the same fields, ordinals, releases and token ids, each stop's EOS id too, and the same
    finishes, the merged responses' `length` by their count), and its ingress records are the id-event run's. Its
    egress carries the same frames a window, each B bytes: the link's bytes a window are B/4096 of the id-event
    run's."""
    B, tmp, _, report, plan, runs, _ = compact
    vtmp, _, vreport, vplan, vruns, _ = varied
    assert plan["compact"] == B and ar.Setup(report).egress_params.frame_bytes == B
    assert ar.Setup(report).ingress_params == ar.Setup(vreport).ingress_params
    check = runs["honest"]["check"]
    assert check["verdict"] == "accepted", cs.table(check)
    assert check["certified_accepted"] and check["audit_accepted"] and check["salt_accepted"]
    io = check["io"]
    assert io["accepted"] and io["violations"] == [] and io["ids_are_the_shims"] and io["requests_are_the_plans"]
    assert set(io["finish"]) == {"eos", "length"} and io["finish"] == vruns["honest"]["check"]["io"]["finish"]
    ours, theirs = _io(tmp / "honest"), _io(vtmp / "honest")
    assert ours.keys() == theirs.keys()
    for name, r in ours.items():
        v = theirs[name]
        assert {k: r[k] for k in r if k != "frames"} == {k: v[k] for k in v if k != "frames"}
    egress = check["egress_bytes"]
    assert egress["frame_bytes"] == B and egress["frames_per_window"] == \
        vruns["honest"]["check"]["egress_bytes"]["frames_per_window"]
    assert egress["wire_per_window"]["run"] * cs.FRAME_BYTES == egress["wire_per_window"]["id_event_frame"] * B
    assert egress["wire_per_window"]["id_event_cell"] == egress["wire_per_window"]["run"] * 256 // B


def test_a_compact_late_shim_is_refused_by_the_audit_and_an_extra_record_by_the_finish_rule(compact):
    """As on id events: the late shim's window-0 egress record isn't established, which the audit refuses; the extra
    shim's record after its first response's finish is on time, so the records are complete, certified and the audit
    accepts them, and only the certified I/O's finish rule refuses it: a record after the response's marked last."""
    _, tmp, _, _, plan, runs, _ = compact
    late = runs["late"]["check"]
    assert late["verdict"] == "refused" and not late["audit_accepted"]
    assert late["rows"][0]["audit"]["egress"]["raw"] == "status:coverage-never-established"
    extra = runs["extra"]["check"]
    assert runs["extra"]["events"] == runs["honest"]["events"] + 1
    assert extra["verdict"] == "refused" and extra["certified_accepted"] and extra["audit_accepted"]
    assert extra["salt_accepted"] and extra["comparisons"]["wire_is_the_records"]
    r = _io(tmp / "extra")[plan["sessions"][0][0]]
    assert r["output"][-1] == 0 and r["finish"] in ("a token after its finish", "longer than max_tokens")
    assert extra["io"]["violations"] == [[r["stream"], r["finish"]]]


def test_the_varied_plan_sends_poisson_arrivals_of_varied_prompts_and_plans_no_egress_profile(varied):
    """No served-profile premise holds of the plan: its requests' prompts (ids), lengths in frames, `max_tokens` and
    seeds vary, nothing asks for exactly its tokens, the canonical order isn't request-major and no response is padded;
    each request is sent mid-bucket in the bucket before its arrival's, as the id request of its fields, in
    `id_request_frames` of its prompt's length, and its stream is its place in the plan, its ingress ordinal; the shim
    forwards it with its `user`, its session, for an endpoint that records it. N2's
    profile is the advice: no egress counts are planned. The EOS is the one the probed responses stopped at."""
    _, probed, report, plan, runs, endpoint = varied
    T, tau = cs.WINDOW_BUCKETS, cs.BUCKET_MS
    assert plan["occupancy"] == {"kind": cs.ADVICE, "egress": None, "ingress": None}
    assert not plan["request_major"] and plan["token_ids"] and "pad_to" not in plan and "body" not in plan
    assert plan["varied"]["eos_token_ids"] == [EOS] and plan["varied"]["max_tokens"] == VARIED_TOKENS
    fields = [q["id_request"] for q in plan["requests"]]
    assert all(set(f) == {"model", "prompt", "max_tokens", "temperature", "top_p", "seed"} and f["temperature"] == 0
               and f["top_p"] == 1 and 1 <= f["max_tokens"] <= VARIED_TOKENS for f in fields)
    assert len({len(f["prompt"]) for f in fields}) > 8 and len({f["max_tokens"] for f in fields}) > 3
    assert len({f["seed"] for f in fields}) == len(fields)
    assert len({q["frames"] for q in plan["requests"]}) > 1
    assert all(q["frames"] == active.id_request_frames(len(f["prompt"]), cs.FRAME_BYTES)
               for q, f in zip(plan["requests"], fields))
    assert [q["stream"] for q in plan["requests"]] == list(range(1, len(fields) + 1))
    assert all(q["send_ms"] == (q["enqueue"] - 0.5) * tau and q["enqueue"] == max(2, math.ceil(q["trace_ms"] / tau))
               for q in plan["requests"])
    assert plan["windows"] >= 3 and len({q["enqueue"] // T for q in plan["requests"]}) >= 2
    assert {r["finish"] for r in probed["responses"]} == {"stop", "length"}

    def key(body):
        return json.dumps({k: v for k, v in body.items() if k != "return_token_ids"}, sort_keys=True)

    bodies = [key(f | {"stop": [], "stream": True}) for f in fields]
    live = [key(f | {"stop": [], "stream": True, "user": q["session"]}) for f, q in zip(fields, plan["requests"])]
    sent = [key(b) for b in endpoint.bodies]
    assert sorted(sent[:4]) == sorted(bodies[:4]) and all(b in sent[4:] for b in live)
    setup = ar.Setup(report)
    assert plan["syncs"] == [[setup.envelope.delta, str(setup.envelope.rho)]] * plan["windows"]
    assert runs["honest"]["check"]["comparisons"]["releases_off_model"] == 0


def test_honest_varied_traffic_is_certified_on_every_window_and_its_io_read_from_the_frames(varied):
    """Every link-window's records are certified under the certifier's key and session and its salt revealed; N2 gives
    no verdict on the advice profile. The certified I/O, read from the records' frames alone, is each request's fields
    as the plan sent them, under its ingress ordinal, and each response's token ids as the endpoint streamed them, EOS
    included, one id event a token and no other egress frame."""
    tmp, _, _, plan, runs, _ = varied
    check = runs["honest"]["check"]
    assert check["verdict"] == "accepted", cs.table(check)
    assert check["certified_accepted"] and check["audit_accepted"] and check["salt_accepted"]
    assert check["network_verdict"] == {"accepted": None, "windows": plan["windows"], "error": cs.NO_N2}
    assert all(r["certified"] and r["n2"] == cs.NO_N2 and r["salt"] == "revealed" for r in check["rows"])
    assert all(r["audit"] == {d: {"raw": None, "committed": None} for d in ("egress", "ingress")} for r in check["rows"])
    io = check["io"]
    assert io["accepted"] and io["violations"] == [] and io["ids_are_the_shims"] and io["requests_are_the_plans"]
    assert io["requests"] == len(plan["requests"]) and set(io["finish"]) == {"eos", "length"}
    records = _io(tmp / "honest")
    for q in plan["requests"]:
        f = q["id_request"]
        r, (ids, _, finish) = records[q["session"]], _answer(f["prompt"], f["max_tokens"])
        assert {k: r[k] for k in f} == f and r["stream"] == q["stream"]
        assert r["output"] == ids and len(r["frames"]) == len(ids)
        assert r["finish"] == {"stop": "eos", "length": "length"}[finish]
        assert r["release"] == int(plan["model_release"][str(r["stream"])]) < min(b for _, b in r["frames"])
    assert sum(len(r["output"]) for r in records.values()) == io["tokens"] == runs["honest"]["events"]


def test_a_late_shim_is_refused_by_the_audit_and_an_extra_token_by_the_finish_rule(varied):
    """The late shim's window-0 egress record is not established, which the certifier's audit refuses. The extra
    shim's token after its first response's last is on time and logged, so the records are complete, certified, and
    the audit accepts them: only the certified I/O's finish rule refuses it, the response going past its EOS or its
    `max_tokens`."""
    tmp, _, _, plan, runs, _ = varied
    late = runs["late"]["check"]
    assert late["verdict"] == "refused" and not late["audit_accepted"]
    assert late["rows"][0]["audit"]["egress"]["raw"] == "status:coverage-never-established"
    extra = runs["extra"]["check"]
    assert runs["extra"]["events"] == runs["honest"]["events"] + 1
    assert extra["verdict"] == "refused" and extra["certified_accepted"] and extra["audit_accepted"]
    assert extra["salt_accepted"] and extra["comparisons"]["wire_is_the_records"]
    first = plan["sessions"][0][0]
    r = _io(tmp / "extra")[first]
    assert r["output"][-1] == 0 and r["finish"] in ("a token after EOS", "longer than max_tokens")
    assert extra["io"]["violations"] == [[r["stream"], r["finish"]]]
    for strategy in cs.STRATEGIES:
        check = runs[strategy]["check"]
        assert all(row["salt"] == "revealed" for row in check["rows"]) and check["salt_again"].startswith("refused")


def test_a_token_shim_on_varied_traffic_changes_window_0s_ids_and_nothing_the_check_reads(varied):
    """The token shim's id events of the requests enqueued in window 0 carry other ids (`active_replay.other_id`),
    their EOS ids left as they are, so every response keeps its length and finish, and `check` accepts it as it does
    on fixed traffic: only a tie of the frames' ids to the committed sampled ones refuses it."""
    tmp, _, _, plan, runs, _ = varied
    assert runs["token"]["check"]["verdict"] == "accepted", cs.table(runs["token"]["check"])
    assert runs["token"]["events"] == runs["honest"]["events"]
    window0 = {q["session"] for q in plan["requests"] if q["enqueue"] // cs.WINDOW_BUCKETS == 0}
    honest, token = _io(tmp / "honest"), _io(tmp / "token")
    assert window0 and any(t != EOS for name in window0 for t in honest[name]["output"])
    for name, r in honest.items():
        t = token[name]
        assert len(t["output"]) == len(r["output"]) and t["finish"] == r["finish"]
        assert [a != b for a, b in zip(r["output"], t["output"])] == [name in window0 and a != EOS for a in r["output"]]
        assert EOS not in t["output"][:-1]


@pytest.fixture(scope="module")
def negatives():
    spec = importlib.util.spec_from_file_location("network_traces_negatives", HERE / "negatives.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # its dataclasses look their module up
    spec.loader.exec_module(mod)
    return mod


def _ids(out: Path, plan: dict) -> Path:
    """The endpoint's sampled ids of the plan's requests (`_answer`'s), by user, as `negatives.read_ids` reads them."""
    path = out / "ids.jsonl"
    path.write_text("".join(json.dumps({"user": q["session"], "output_ids": _answer(
        q["id_request"]["prompt"], q["id_request"]["max_tokens"])[0]}) + "\n" for q in plan["requests"]))
    return path


def test_the_io_tie_accepts_honest_varied_traffic_and_refuses_the_token_shims_ids_by_its_content(varied, negatives):
    """`negatives.tie_run` on each varied run, given the endpoint's sampled ids: every window's frames hold to their
    certified heads, both ends, and every id event is a data cell placed on a committed output. Honest's ids are their
    port entries' (accepted); the token shim's are not in exactly the windows its changed ids fall in, each refused by
    IoTieWindow.content alone; so is the extra shim's token past a response's end, which no committed output is, and
    the drop shim's response without its first token, whose later ids each sit one output early and whose last
    committed output no cell carries. The late shim's held frames never reach a certified record. A committed output no
    cell carries is refused by content at window -1, since no advice places it in a window."""
    pytest.importorskip(negatives.IO_TIE_MODULE)
    tmp, _, _, plan, runs, _ = varied
    ids = negatives.read_ids(_ids(tmp, plan))
    got = {s: negatives.tie_run(tmp / s, ids) for s in runs}
    tokens, W = sum(map(len, ids.values())), plan["windows"]
    for row in got.values():
        assert row["held"] == {"egress": W, "ingress": W} and row["missing"] == {} and row["outputs"] == tokens
    honest, token, extra = got["honest"], got["token"], got["extra"]
    assert honest["verdict"] == "accepted" and honest["failing"] == {} and honest["codes"] == []
    assert honest["id_events"] == honest["placed"] == tokens and honest["unplaced"] == 0
    h, t = _io(tmp / "honest"), _io(tmp / "token")
    changed = {b // cs.WINDOW_BUCKETS for name, r in t.items()
               for (_, b), x, y in zip(r["frames"], r["output"], h[name]["output"]) if x != y}
    assert changed and token["verdict"] == "refused" and token["codes"] == ["IoTieWindow.content"]
    assert token["failing"] == {w: ["content"] for w in changed} and token["placed"] == tokens
    assert extra["verdict"] == "refused" and extra["codes"] == ["IoTieWindow.content"]
    assert extra["id_events"] == tokens + 1 and extra["placed"] == tokens and len(extra["failing"]) == 1
    drop = got["drop"]
    assert drop["verdict"] == "refused" and drop["codes"] == ["IoTieWindow.content"] and -1 in drop["failing"]
    assert drop["id_events"] == drop["placed"] == tokens - 1 and drop["unplaced"] == 1
    late = got["late"]
    assert late["verdict"] == "refused" and late["failing"] == {-1: ["content"]}
    assert 0 < late["unplaced"] == tokens - late["id_events"] and runs["late"]["events"] == tokens
    assert all(negatives.tie_ok(s, got[s]) for s in ("honest", "token", "extra", "drop"))


def test_a_frame_other_than_the_certified_one_is_refused_by_the_tie_on_the_cells_root(varied, negatives, tmp_path):
    """The honest run with one id event in the prover's copy of the frames carrying another id, the certified heads
    kept: that window's frames no longer hold to its certified head (IoTieWindow.cells_root), and the cell no longer
    carries its port entry's id (content); every other window is accepted."""
    pytest.importorskip(negatives.IO_TIE_MODULE)
    tmp, _, report, plan, _, _ = varied
    run, e = shutil.copytree(tmp / "honest", tmp_path / "frame"), ar.Setup(report).egress_params
    records = [ar.unpack_record(b, e.frame_bytes) for b in ar.read_blobs(run / "responses.records")]
    k = next(k for k, r in enumerate(records) if any(row[0] is not None for row in r.grid))
    t = next(t for t, row in enumerate(records[k].grid) if row[0] is not None)
    q, x = active.id_event_data(active.decode_frame(records[k].grid[t][0])[2])
    row = (active.encode_frame(active.DATA, active.CANONICAL_STREAM, active.id_event(q, x ^ 1), e.frame_bytes),
           *records[k].grid[t][1:])
    records[k] = replace(records[k], grid=records[k].grid[:t] + (row,) + records[k].grid[t + 1:])
    (run / "responses.records").unlink()
    for r in records:
        ar.write_blobs(run / "responses.records", ar.pack_record(r))
    got = negatives.tie_run(run, negatives.read_ids(_ids(tmp_path, plan)))
    w = records[k].window
    assert got["held"] == {"egress": plan["windows"] - 1, "ingress": plan["windows"]}
    assert got["verdict"] == "refused" and got["failing"] == {w: ["cells_root", "content"]}
    assert negatives.tie_ok("frame", got)


def test_tie_run_prints_each_runs_verdict_from_a_gate_runs_tarball(varied, negatives, tmp_path, capsys):
    """`negatives.py tie-run --run TARBALL --ids FILE` on a tarball laid out as the gate preserves a run
    (`c…/out/STRATEGY/`) prints its document: each strategy's verdict, `ok` and codes, exit 0. Without the ids each run
    has no verdict, the Commit's rows missing, exit 1."""
    pytest.importorskip(negatives.IO_TIE_MODULE)
    tmp, _, _, plan, _, _ = varied
    gate = tmp_path / "c20261024T000000Z-tie"
    for s in ("honest", "token"):
        shutil.copytree(tmp / s, gate / "out" / s)
    tgz = tmp_path / f"{gate.name}.tgz"
    with tarfile.open(tgz, "w:gz") as tf:
        tf.add(gate, arcname=gate.name)
    assert negatives.main(["tie-run", "--run", str(tgz), "--ids", str(_ids(tmp_path, plan))]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["format"] == negatives.TIE_RUN and doc["check"] == "IoTieWindow" and doc["source"] == str(tgz)
    assert set(doc["stand_ins"]) == {"cells_root", "cells", "port", "map"}
    assert {s: (r["verdict"], r["ok"], r["codes"]) for s, r in doc["runs"].items()} == {
        "honest": ("accepted", True, []), "token": ("refused", True, ["IoTieWindow.content"])}
    assert negatives.main(["tie-run", "--run", str(tgz)]) == 1
    doc = json.loads(capsys.readouterr().out)
    assert {s: (r["verdict"], r["ok"], list(r["missing"])) for s, r in doc["runs"].items()} == {
        s: ("not-applicable", None, ["commit"]) for s in ("honest", "token")}


def test_the_certified_io_reads_each_response_by_its_ordinal_wherever_its_frames_fall(varied):
    """`certified_io` reads a response from the id events that name its request's ingress ordinal, in any
    interleaving: the honest run's egress frames dealt round-robin over their requests, after every release, give back
    the same responses; a frame that names no request is a violation, and so is one before its request's release."""
    tmp, _, report, plan, _, _ = varied
    run, setup = tmp / "honest", ar.Setup(report)
    e, i = setup.egress_params, setup.ingress_params
    ins = [ar.unpack_record(b, i.frame_bytes) for b in ar.read_blobs(run / "requests.records")]
    outs = [ar.unpack_record(b, e.frame_bytes) for b in ar.read_blobs(run / "responses.records")]
    frames = [s for rec in sorted(outs, key=lambda r: r.window) for row in rec.grid for s in row if s is not None]
    by_ordinal: dict = defaultdict(list)
    for f in frames:
        by_ordinal[active.id_event_data(active.decode_frame(f)[2])[0]].append(f)
    dealt = [f for group in itertools.zip_longest(*by_ordinal.values()) for f in group if f is not None]
    assert dealt != frames and sorted(by_ordinal) == [q["stream"] for q in plan["requests"]]
    start = 1 + max(plan["model_release"].values())

    def one_record(fs, early=()):
        return [SimpleNamespace(window=0, grid=[[f] for f in early] + [[]] * (start - len(early)) + [[f] for f in fs])]

    def responses(io):
        return {r["user"]: (r["stream"], r["output"], r["finish"]) for r in io["records"]}

    honest, mixed = cs.certified_io(plan, setup, ins, outs), cs.certified_io(plan, setup, ins, one_record(dealt))
    assert honest["accepted"] and mixed["accepted"] and responses(mixed) == responses(honest)
    stray = active.encode_frame(active.DATA, active.CANONICAL_STREAM, active.id_event(len(plan["requests"]) + 1, 7),
                                e.frame_bytes)
    refused = cs.certified_io(plan, setup, ins, one_record(dealt + [stray]))
    assert not refused["accepted"] and refused["violations"] == [
        [len(dealt), "an egress frame that isn't an id event of a released request"]]
    early = cs.certified_io(plan, setup, ins, one_record(dealt, early=[by_ordinal[len(plan["requests"])][0]]))
    assert not early["accepted"] and early["violations"] == [[0, "an id event before its request's release"]]


def test_the_certified_io_refuses_two_requests_with_one_user(varied):
    """A request's `user` is its own id, never its session's: with the second request's `user` rewritten in its
    certified ingress frame to the first's (the field's place and every frame's template kept), the certified I/O
    refuses that request, and still reads each response by its ordinal."""
    tmp, _, report, plan, _, _ = varied
    run, setup = tmp / "honest", ar.Setup(report)
    e, i = setup.egress_params, setup.ingress_params
    ins = [ar.unpack_record(b, i.frame_bytes) for b in ar.read_blobs(run / "requests.records")]
    outs = [ar.unpack_record(b, e.frame_bytes) for b in ar.read_blobs(run / "responses.records")]
    first, second = (json.dumps(q["session"]).encode() for q in plan["requests"][:2])
    shared = [SimpleNamespace(window=r.window, grid=[[f if f is None else f.replace(second, first) for f in row]
                                                     for row in r.grid]) for r in ins]
    assert sum(f != g for r, s in zip(ins, shared) for row, other in zip(r.grid, s.grid)
               for f, g in zip(row, other)) == 1
    honest, refused = cs.certified_io(plan, setup, ins, outs), cs.certified_io(plan, setup, shared, outs)
    assert honest["accepted"] and len({r["user"] for r in honest["records"]}) == len(plan["requests"])
    assert not refused["accepted"] and refused["violations"] == [[2, "a request whose user names request 1 too"]]
    assert [(r["stream"], r["output"]) for r in refused["records"]] == \
        [(r["stream"], r["output"]) for r in honest["records"]]


#: two gate runs of `certifier-rehearse --traffic varied` against one endpoint, each strategy's (run, --tamper)
GATE_RUNS = {"honest": ("c20261024T120000Z-ab12", None), "token": ("c20261024T121500Z-cd34", "token")}


@pytest.fixture(scope="module")
def gate_varied(tmp_path_factory):
    """`GATE_RUNS` against one `_Varied` endpoint: each the rehearsal the gate's command asks for
    (`auditor.certifier_command` of a varied request, as `certifier_serve.py rehearse` parses it), its plan made, run
    and checked, laid out as the gate keeps it (`<run>/out/`: the plan, its calibration and probe, and the strategy's
    run) and tarred as `fetch` gives it; and the endpoint's `requests.jsonl`, every request it served, probes included,
    as serve-openai records them."""
    import tarfile

    sys.path.insert(0, str(HERE.parent / "live_challenger"))
    import auditor as A
    tmp, endpoint, got = tmp_path_factory.mktemp("gate-varied"), _Varied(), {}
    for strategy, (run, tamper) in GATE_RUNS.items():
        req = {"run": run, "requests": 16, "tamper": tamper, "traffic": "varied", "temperature": 0.0, "top_p": 1.0,
               "tokens": 16, "seed": 3}
        out = tmp / run / "out"
        args = cs.parser().parse_args(A.certifier_command(req, out, "http://127.0.0.1:40001")[2:])
        runs_dir, probed, report, plan, runs = _varied_runs(
            tmp / f"{run}.runs", endpoint, lambda url, a=args: cs.plan_of(a, url, lambda: 0.0, _no_sleep),
            args.strategies.split(","))
        out.mkdir(parents=True)
        for name, doc in (("probe.json", probed), ("plan.json", plan), ("calibration.json", report)):
            (out / name).write_text(json.dumps(doc))
        shutil.copytree(runs_dir / strategy, out / strategy)
        with tarfile.open(tmp / f"{run}.tgz", "w:gz") as tf:
            tf.add(tmp / run, arcname=run)
        got[strategy] = {"run": run, "args": args, "plan": plan, "check": runs[strategy]["check"],
                         "out": out, "tgz": tmp / f"{run}.tgz"}
    ids = tmp / "requests.jsonl"
    ids.write_text("".join(json.dumps(r) + "\n" for r in endpoint.served))
    return got, endpoint, ids


def test_the_gates_varied_rehearsal_names_its_requests_for_its_run_and_the_endpoint_records_each_under_that_name(
        gate_varied):
    """The gate's varied request runs `rehearse --traffic varied --name RUN` with its flags: each request's `user`,
    its session, is `RUN.j`, which the shim forwards, so the endpoint records every request of both runs under its
    own name, once, beside the probes it was sent without one. The honest run's certified I/O is what the endpoint
    recorded, request by request; the token run's, which `check` accepts, differs from it in window 0's ids alone.
    The run the gate keeps holds the plan, the records the I/O is read from (`requests.records`,
    `responses.records`, `certified.jsonl`) and `io.json`."""
    got, endpoint, _ = gate_varied
    rows = [r for r in endpoint.served if r["user"] is not None]
    assert sum(r["user"] is None for r in endpoint.served) == 2 * 8
    assert len(rows) == len({r["user"] for r in rows}) == 2 * 16
    recorded = {r["user"]: r["streamed_ids"] for r in rows}
    for strategy, (run, tamper) in GATE_RUNS.items():
        g = got[strategy]
        a, plan = g["args"], g["plan"]
        assert (a.traffic, a.name, a.tokens, a.seed, a.requests, a.strategies) == \
            ("varied", run, 16, 3, 16, tamper or "honest")
        assert [q["session"] for q in plan["requests"]] == [f"{run}.{j:05d}" for j in range(16)]
        assert g["check"]["verdict"] == "accepted", cs.table(g["check"])
        io = {r["user"]: r for r in json.loads((g["out"] / strategy / "io.json").read_text())["requests"]}
        assert set(io) == {q["session"] for q in plan["requests"]}
        window0 = {q["session"] for q in plan["requests"] if q["enqueue"] // cs.WINDOW_BUCKETS == 0}
        differs = {u for u, r in io.items() if r["output"] != recorded[u]}
        assert differs == (set() if tamper is None else window0) and (tamper is None or differs)
        assert {"requests.records", "responses.records", "certified.jsonl", "io.json"} <= \
            {p.name for p in (g["out"] / strategy).iterdir()}


@pytest.fixture(scope="module")
def gate_compact(tmp_path_factory):
    """`GATE_RUNS` as `gate_varied`, each request with `compact` 8: the rehearsal the gate's command asks for, its plan
    made, run and checked, against one `_Varied` endpoint whose every other response ends in a merged chunk."""
    sys.path.insert(0, str(HERE.parent / "live_challenger"))
    import auditor as A
    tmp, endpoint, got = tmp_path_factory.mktemp("gate-compact"), _Varied(merges=lambda r: r % 2 == 1), {}
    for strategy, (run, tamper) in GATE_RUNS.items():
        req = {"run": run, "requests": 16, "tamper": tamper, "traffic": "varied", "temperature": 0.0, "top_p": 1.0,
               "tokens": 16, "seed": 3, "compact": 8}
        args = cs.parser().parse_args(A.certifier_command(req, tmp / run / "out", "http://127.0.0.1:40001")[2:])
        runs_dir, _, _, plan, runs = _varied_runs(
            tmp / f"{run}.runs", endpoint, lambda url, a=args: cs.plan_of(a, url, lambda: 0.0, _no_sleep),
            args.strategies.split(","))
        got[strategy] = {"args": args, "plan": plan, "check": runs[strategy]["check"], "io": _io(runs_dir / strategy),
                         "dir": runs_dir / strategy}
    return got, endpoint


def test_the_gates_compact_rehearsal_runs_compact_cells_and_its_check_reads_them_as_on_id_events(gate_compact):
    """`certifier-rehearse --traffic varied --compact 8`'s command is a compact varied `rehearse` this parser takes,
    named for its run, and the gate's bounds on B are a compact record's bytes and an id event's frame. Its runs are
    on 8-byte cells: the honest run accepted, its certified I/O the endpoint's ids (each stop's EOS id too), and the
    token run accepted by `check` too, its ids differing in window 0's alone, which only the I/O tie refuses. The
    tie's compact rule (`active.compact_finish_violations`), with the endpoint's ids as the Commit's, finds nothing
    in the honest run's cells and refuses exactly window 0's responses in the token run's."""
    sys.path.insert(0, str(HERE.parent / "live_challenger"))
    import auditor as A
    assert A.CERTIFIER_COMPACT == (active.COMPACT_RECORD.size, cs.FRAME_BYTES)
    got, endpoint = gate_compact
    recorded = {r["user"]: r["streamed_ids"] for r in endpoint.served if r["user"] is not None}
    for strategy, (run, tamper) in GATE_RUNS.items():
        g = got[strategy]
        assert (g["args"].compact, g["args"].name, g["args"].strategies) == (8, run, tamper or "honest")
        check = g["check"]
        assert g["plan"]["compact"] == 8 and check["egress_bytes"]["frame_bytes"] == 8
        assert check["verdict"] == "accepted" and check["io"]["violations"] == [], cs.table(check)
        window0 = {q["session"] for q in g["plan"]["requests"] if q["enqueue"] // cs.WINDOW_BUCKETS == 0}
        differs = {u for u, r in g["io"].items() if r["output"] != recorded[u]}
        assert differs == (set() if tamper is None else window0) and (tamper is None or differs)
        e = ar.Setup(json.loads((g["dir"] / "calibration.json").read_text())).egress_params
        outs = sorted((ar.unpack_record(b, e.frame_bytes) for b in ar.read_blobs(g["dir"] / "responses.records")),
                      key=lambda r: r.window)
        cells = [s for r in outs for row in r.grid for s in row[:active.row_count(row)]]
        sampled = {r["stream"]: recorded[u] for u, r in g["io"].items()}
        bad = active.compact_finish_violations(sampled, g["plan"]["varied"]["eos_token_ids"], cells,
                                               max_tokens={r["stream"]: r["max_tokens"] for r in g["io"].values()})
        assert {o for o, _ in bad} == {g["io"][u]["stream"] for u in differs}, bad
    plain = cs.parser().parse_args(A.certifier_command(
        {"run": "c20261024T120000Z-ab12", "requests": 16, "tamper": None, "traffic": "varied"}, Path("o"), "u")[2:])
    assert plain.compact is None


def test_tie_run_on_the_gates_compact_runs_reads_their_records_as_data_cells_and_holds_their_finish(
        gate_compact, negatives, tmp_path):
    """`negatives.tie_run` on the gate's compact runs, the endpoint's `requests.jsonl` the Commit's ids: each compact
    record is the data cell of its id, so neither run lacks id events, and the tie holds the link's records to the
    ids (`finish`, `active.compact_finish_violations`). Honest's finish holds; token's fails on exactly window 0's
    responses, which refuses it. Honest's fails too, at that request's ordinal alone, where the Commit says a user's
    client aborted it (`aborted`) and its records end on `COMPACT_LAST`. With `io_tie` in the tree honest is accepted
    and token refused by content too, at window 0; without it neither has a verdict, only `io_tie` missing."""
    got, endpoint = gate_compact
    path = tmp_path / "requests.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in endpoint.served))
    ids, IT = negatives.read_ids(path), negatives.io_tie_module()
    rows = {}
    for strategy, g in got.items():
        row = rows[strategy] = negatives.tie_run(g["dir"], ids)
        window0 = {g["io"][q["session"]]["stream"] for q in g["plan"]["requests"]
                   if q["enqueue"] // cs.WINDOW_BUCKETS == 0}
        assert row["id_events"] == sum(len(r["output"]) for r in g["io"].values()) > 0
        assert list(row["missing"]) == ([] if IT else ["io-tie"])
        assert {o for o, _ in row["finish_violations"]} == (window0 if strategy == "token" else set()) and window0
        assert row["finish"] is (strategy != "token")
    gone = {u: r for u, r in got["honest"]["io"].items() if r["finish"] == "length"}
    u = min(gone)
    marked = negatives.tie_run(got["honest"]["dir"], ids, aborted={u})
    assert marked["finish"] is False and marked["verdict"] == ("refused" if IT else negatives.NOT_APPLICABLE)
    assert marked["finish_violations"] == [
        [gone[u]["stream"], f"finish length on output {len(gone[u]['output']) - 1} where the Commit's is none"]]
    if IT is None:
        assert {s: r["verdict"] for s, r in rows.items()} == {s: negatives.NOT_APPLICABLE for s in got}
        return
    honest, token = rows["honest"], rows["token"]
    assert (honest["verdict"], honest["codes"], honest["unplaced"]) == ("accepted", [], 0)
    assert (token["verdict"], token["codes"]) == ("refused", ["IoTieWindow.content"])
    assert token["failing"] == {0: ["content"]}
    assert negatives.tie_ok("honest", honest) and negatives.tie_ok("token", token)


def test_tie_run_on_the_gates_varied_runs_with_the_endpoints_ids_accepts_honest_and_refuses_token(gate_varied,
                                                                                                    capsys):
    """`negatives.py tie-run --run TARBALL --ids REQUESTS.jsonl` on each gate run's tarball, the ids the endpoint's
    `requests.jsonl` (both runs' and the probes'), exits 0: the honest run is accepted and the token run refused by
    IoTieWindow.content, each verdict the one it owes (`ok`)."""
    spec = importlib.util.spec_from_file_location("network_traces_negatives_gate", HERE / "negatives.py")
    negatives = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = negatives  # its dataclasses look their module up
    spec.loader.exec_module(negatives)
    if not hasattr(negatives, "tie_runs"):
        pytest.skip("this tree's negatives.py has no tie-run")
    pytest.importorskip(negatives.IO_TIE_MODULE)
    got, _, ids = gate_varied
    verdicts = {}
    for strategy, g in got.items():
        assert negatives.main(["tie-run", "--run", str(g["tgz"]), "--ids", str(ids)]) == 0
        doc = json.loads(capsys.readouterr().out)
        assert set(doc["runs"]) == {strategy} and doc["runs"][strategy]["missing"] == {}
        r = doc["runs"][strategy]
        verdicts[strategy] = (r["verdict"], r["ok"], r["codes"])
    assert verdicts == {"honest": ("accepted", True, []), "token": ("refused", True, ["IoTieWindow.content"])}


def test_interleaved_responses_are_recovered_by_their_requests_ordinals(tmp_path):
    """Requests sent four at a time are released two a bucket, and the endpoint streams each two released together
    chunk by chunk: in whatever order the shim reads them, each id event names its request's ingress ordinal, and
    honest traffic's certified I/O gives back every response."""
    endpoint = _Varied(group=2)

    def plan_of(url):
        reqs = cs.varied_requests(url, "stub-model", 8, prompt_tokens=(16, 300), max_tokens=VARIED_TOKENS, seed=5)
        reqs = [q | {"at_ms": reqs[j - j % 4]["at_ms"]} for j, q in enumerate(reqs)]
        probed = cs.probe_varied(url, "stub-model", reqs[:4], clock=lambda: 0.0, sleep=_no_sleep)
        report, plan = cs.varied_plan(probed, reqs, eos=[EOS])
        release = [plan["model_release"][str(q["stream"])] for q in plan["requests"]]
        assert ar.Setup(report).ingress_params.allocation == 2 and release[::2] == release[1::2]
        assert len(set(release)) == 4 and {q["frames"] for q in plan["requests"]} == {1}
        return probed, report, plan

    _, _, _, plan, runs = _varied_runs(tmp_path, endpoint, plan_of, ["honest"])
    check = runs["honest"]["check"]
    assert check["verdict"] == "accepted", cs.table(check)
    records = _io(tmp_path / "honest")
    for q in plan["requests"]:
        ids, _, _ = _answer(q["id_request"]["prompt"], q["id_request"]["max_tokens"])
        assert records[q["session"]]["output"] == ids and records[q["session"]]["stream"] == q["stream"]
    run = [w for w in endpoint.writes if w[0] >= 4]
    assert any(a[0] != b[0] for a, b in itertools.pairwise(run) if a[0] // 2 == b[0] // 2)
