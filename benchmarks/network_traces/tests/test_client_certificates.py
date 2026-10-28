"""The client keeps the certificates: once an honest served run's wire has ended (on an injected clock, behind a stub
endpoint), the reveal endpoint (`active.serve_reveals(certificates=…)`) serves each window's certificates' public part,
unsigned, and the client (`certifier_serve.keep_certificates`) keeps every one, each verified under the deployment's
key and session, in `certificates.jsonl` and `clocks.jsonl`: the certifier's own certificates and salted clock
certificates, and nothing of a record's head, the sync a clock certificate signs, or either's salt. A forged signature,
an altered digest, a wrong window, a clock certificate under a reused salt, one of another key and an answer of another
session are each refused by name, and the client keeps none of them."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from verity.core.protocols.certifier import active
from verity.core.protocols.certifier.certificate import Certificate, Direction, Issuer, clock_commitment
from verity.core.protocols.certifier.schedule import ClockSync

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
    events = [b'data: {"choices":[{"text":" t%d"}]}\n\n' % k for k in range(body["max_tokens"])] + [b"data: [DONE]\n\n"]
    writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"
                 + b"".join(b"%x\r\n%s\r\n" % (len(p), p) for p in events) + b"0\r\n\r\n")
    await writer.drain()
    writer.close()


def _ask(address, window):
    return cs.ask_certificates(address, window, timeout=None)


@pytest.fixture(scope="module")
def honest(tmp_path_factory):
    """One honest served run: its directory, its deployment record, the endpoint's answer for each window, its answers
    past the run and on an endpoint without certificates, and the client's refusals."""
    tmp = tmp_path_factory.mktemp("honest")
    run, out = tmp / "run", {}
    challenger, certifier = cs.signer(tmp / "challenger", 6), cs.signer(tmp / "certifier", 6)

    async def go():
        endpoint = await asyncio.start_server(_endpoint, "127.0.0.1", 0)
        upstream = endpoint.sockets[0].getsockname()[:2]
        url = "http://%s:%d" % upstream
        probed = await asyncio.to_thread(cs.probe, url, cs.request_body("stub", TOKENS), 2, clock=lambda: 0.0)
        report, plan = cs.served_plan(probed, REQUESTS)
        setup = ar.Setup(report)
        syncs = [ClockSync(d, Fraction(r)) for d, r in plan["syncs"]]
        run.mkdir()

        class Clocked(cs.CertifyingPublisher):
            """Each egress record handed with its window's sync, as `ServedProxy` hands it."""

            def __call__(self, record, params, sync=None) -> None:
                egress = record.link == setup.egress_link and record.window < len(syncs)
                super().__call__(record, params, syncs[record.window] if egress else None)

        publish = Clocked(run, setup, Issuer(certifier, SESSION), cs.N2_LINK, threaded=False)

        async def after(proxies, clock):
            reveals = await active.serve_reveals(proxies[0], "127.0.0.1", 0, certificates=publish.public)
            bare = await active.serve_reveals(proxies[0], "127.0.0.1", 0)
            record = cs.deployment(proxies[0], setup, plan["windows"], session=SESSION, n2_link=cs.N2_LINK,
                                   link=("127.0.0.1", 0), reveal=reveals.sockets[0].getsockname()[:2],
                                   origin_unix=0.0, url=url)
            out["deployment"] = record
            out["answers"] = [await asyncio.to_thread(_ask, record["reveal"], w) for w in range(plan["windows"])]
            out["past"] = await asyncio.to_thread(_ask, record["reveal"], plan["windows"])
            out["bare"] = await asyncio.to_thread(_ask, bare.sockets[0].getsockname()[:2], 0)
            out["refused"] = await asyncio.to_thread(cs.keep_certificates, record, tmp / "client", _ask)
            reveals.close()
            bare.close()

        (tmp / "client").mkdir()
        await asyncio.to_thread(
            ar.run_virtual, plan, report, run, upstream, salt=True, publish=publish, after=after,
            proxy_args={"challenger_key": challenger.public_key(), "signer": certifier,
                        "served": cs.served_profile(plan)})
        endpoint.close()

    try:
        asyncio.run(go())
    finally:
        challenger.close()
        certifier.close()
    return tmp, out


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_the_client_keeps_every_windows_certificates_verified_and_nothing_of_a_head(honest):
    tmp, out = honest
    deployment, windows = out["deployment"], out["deployment"]["windows"]
    key, session = cs.read_public(deployment["certifier_key"]), bytes.fromhex(deployment["session"])
    assert out["refused"] == [] and windows >= 2
    kept = _lines(tmp / "client" / "certificates.jsonl")
    certified = _lines(tmp / "run" / "certified.jsonl")
    assert len(kept) == len(certified) == 2 * windows
    assert all(set(d) == {"session", *Certificate.__slots__} and d["session"] == SESSION.hex() for d in kept)
    assert all(cs.certificate_from_json(d).verify(key, session) for d in kept)
    assert sorted(map(json.dumps, (d["certificate"] for d in certified))) == \
        sorted(json.dumps({f: d[f] for f in Certificate.__slots__}) for d in kept)
    text = (tmp / "client" / "certificates.jsonl").read_text()
    assert not any(d["salt_c"] in text or d["head"]["root"] in text for d in certified)
    signed, clocks = cs.read_clocks(tmp / "run" / "clocks.jsonl"), _lines(tmp / "client" / "clocks.jsonl")
    assert [c.window for c in signed] == list(range(windows)) and all(c.verify(key, session) for c in signed)
    assert len({c.salt for c in signed}) == windows
    assert all(set(d) == {"session", *Certificate.__slots__} and d["session"] == SESSION.hex() for d in clocks)
    assert [{f: d[f] for f in Certificate.__slots__} for d in clocks] == \
        [cs.certificate_json(c.certificate) for c in signed]
    assert all(cs.certificate_from_json(d).verify(key, session) for d in clocks)
    assert all(set(d) == set(Certificate.__slots__) for a in out["answers"] for d in a["clocks"])
    text = (tmp / "client" / "clocks.jsonl").read_text()
    assert not any(c.salt.hex() in text for c in signed)
    with pytest.raises(ValueError, match="without its opening's salt"):
        cs.read_clocks(tmp / "client" / "clocks.jsonl")


def test_the_endpoint_serves_no_window_it_hasnt_certified_and_nothing_without_certificates(honest):
    _, out = honest
    windows = out["deployment"]["windows"]
    assert out["past"] == {"refused": f"window {windows}'s records aren't all certified, so its certificates aren't "
                                      "served"}
    assert out["bare"] == {"refused": "this certifier serves no certificates"}
    assert [a["window"] for a in out["answers"]] == list(range(windows))


def _altered(answer: dict, part: str, field: str, change) -> dict:
    doc = json.loads(json.dumps(answer))
    doc[part][0][field] = change(doc[part][0][field])
    return doc


def _flip(text: str) -> str:
    """The hex `text` with its last digit changed."""
    return text[:-1] + ("1" if text[-1] == "0" else "0")


def _refusals(tmp_path: Path, deployment: dict, answers: list[dict]) -> tuple[list[str], list[dict], list[dict]]:
    refused = cs.keep_certificates(deployment, tmp_path, lambda address, w: answers[w])
    return refused, _lines(tmp_path / "certificates.jsonl"), _lines(tmp_path / "clocks.jsonl")


@pytest.mark.parametrize("part, field", [
    ("certificates", "signature"),
    ("certificates", "digest"),
    ("clocks", "signature"),
    ("clocks", "digest"),
])
def test_a_forged_or_altered_certificate_is_refused_by_name_and_not_kept(honest, tmp_path, part, field):
    """A forged signature or an altered digest, of a record's certificate or a clock certificate: the client holds no
    head, sync or salt, so either is the signature failing over the fields it was served with."""
    _, out = honest
    answers = list(out["answers"])
    answers[0] = _altered(answers[0], part, field, _flip)
    c = answers[0][part][0]
    what = "certificate" if part == "certificates" else "clock certificate"
    refused, kept, clocks = _refusals(tmp_path, out["deployment"], answers)
    assert refused == [f"window 0's {c['direction']} {what} of {cs.N2_LINK} (key {c['key_id'][:16]}…): its signature "
                       "doesn't verify under the deployment's key in its session"]
    windows = out["deployment"]["windows"]
    assert len(kept) + len(clocks) == 3 * windows - 1 and c[field] not in json.dumps([kept, clocks])


@pytest.mark.parametrize("part", ["certificates", "clocks"])
def test_a_certificate_of_another_link_window_is_refused_by_name(honest, tmp_path, part):
    _, out = honest
    answers = list(out["answers"])
    answers[0] = _altered(answers[0], part, "window", lambda w: w + 1)
    c = answers[0][part][0]
    what = "certificate" if part == "certificates" else "clock certificate"
    refused, kept, clocks = _refusals(tmp_path, out["deployment"], answers)
    assert refused == [f"window 0's {c['direction']} {what} of {cs.N2_LINK} (key {c['key_id'][:16]}…): it names "
                       f"{cs.N2_LINK} window 1, not {cs.N2_LINK} window 0"]
    assert len(kept) + len(clocks) == 3 * out["deployment"]["windows"] - 1


def test_a_clock_certificate_under_a_reused_salt_is_refused_by_name(tmp_path):
    """A certifier whose clock salt repeats across two windows of one sync signs one digest for both, which shows that
    their syncs are equal: the client, holding no sync or salt, sees window 1's clock certificate sign window 0's
    digest and refuses it by name. The same syncs under the issuer's fresh salts sign two digests, both kept."""
    own = cs.signer(tmp_path / "key", 6)
    try:
        issuer, sync, salt = Issuer(own, SESSION), ClockSync(1, Fraction(1)), bytes(range(64))
        reused = [issuer.sign(cs.N2_LINK, Direction.EGRESS, w, clock_commitment(salt, sync)) for w in (0, 1)]
        fresh = [issuer.salted_clock(cs.N2_LINK, w, sync).certificate for w in (0, 1)]
        public = cs.public_json(own.public_key())
    finally:
        own.close()
    deployment = {"certifier_key": public, "session": SESSION.hex(), "links": {"n2": cs.N2_LINK}, "windows": 2,
                  "reveal": None}

    def answers(certificates):
        return [{"window": w, "session": SESSION.hex(), "certificates": [], "clocks": [cs.certificate_json(c)],
                 "unsigned": []} for w, c in enumerate(certificates)]

    (tmp_path / "reused").mkdir()
    (tmp_path / "fresh").mkdir()
    refused, _, clocks = _refusals(tmp_path / "reused", deployment, answers(reused))
    assert refused == [f"window 1's egress clock certificate of {cs.N2_LINK} (key {public['key_id'][:16]}…): its "
                       "signed digest is also window 0's clock certificate's, so its salt was reused, under which two "
                       "windows of one sync sign one digest"]
    assert [d["window"] for d in clocks] == [0]
    refused, _, clocks = _refusals(tmp_path / "fresh", deployment, answers(fresh))
    assert refused == [] and [d["window"] for d in clocks] == [0, 1]


def test_every_certificate_is_refused_by_name_under_another_key(honest, tmp_path):
    _, out = honest
    cs.load_key(tmp_path / "other", 6)
    other = json.loads((tmp_path / "other" / "public.json").read_text())
    deployment = out["deployment"] | {"certifier_key": other}
    refused, kept, clocks = _refusals(tmp_path, deployment, out["answers"])
    assert kept == clocks == [] and len(refused) == 3 * deployment["windows"]
    assert all(r.endswith(f"another key's, not the deployment's {other['key_id'][:16]}…") for r in refused)
    own = out["deployment"]["certifier_key"]["key_id"][:16]
    assert refused[0].startswith(f"window 0's ") and f"(key {own}…)" in refused[0]


def test_an_answer_of_another_session_is_refused_whole(honest, tmp_path):
    _, out = honest
    deployment = out["deployment"] | {"session": bytes(32).hex()}
    refused, kept, clocks = _refusals(tmp_path, deployment, out["answers"])
    assert kept == clocks == []
    assert refused == [f"window {w}'s certificates: the answer is window {w}'s in session {SESSION.hex()[:16]}…, not "
                       f"the deployment's session {bytes(32).hex()[:16]}…" for w in range(deployment["windows"])]
