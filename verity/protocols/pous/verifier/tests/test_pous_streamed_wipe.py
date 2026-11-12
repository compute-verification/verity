"""The streamed-label wipe (`streamed_wipe`): the label format, an honest session accepted at batch size 1 and refused
above it, which the guarantee doesn't cover, a wrong or late batch that ends the session with no retry, the fill before
any challenge, the start mark the challenger holds before the fill and signs in its header, the start and end messages,
a session of another format or with a mark that is not one refused by the reader, the tiers the fill covers, a verdict
that fails closed, and the guarantee's bound at the deployed instance and against an exhaustive count on a tiny one.
Every clock is injected."""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math

import pytest

import pous_test_signer as ts
from pous_fakes import Clock
from verity.protocols.pous.device import memory_challenger as mc
from verity.protocols.pous.verifier import streamed_wipe as sw
from verity.protocols.pous.verifier import wipe_party as WP
from verity.protocols.pous.verifier import wipe_period
from verity.commitments.randomness import derive

SECRET, SERVER = b"auditor secret", b"server-7"
SESSION = hashlib.sha512(b"wipe session").digest()[:32]
N, K, LIMIT = 256, 40, 1_000_000


def params(batch: int = 1, limit: int = LIMIT) -> sw.WipeParams:
    return sw.WipeParams(N, K, batch, limit, (("gpu", 128 * sw.LABEL_BYTES), ("host-dram", 128 * sw.LABEL_BYTES)))


def challenger(p: sw.WipeParams | None = None, *, clock=None, signer=None, log=None, mark_source=None,
               server: bytes = SERVER) -> sw.Challenger:
    return sw.Challenger(SECRET, server, SESSION, p or params(), signer or ts.TestSigner(b"wipe challenger", height=2),
                         log=log or ts.session_log(), clock=clock or Clock(1_000_000), coin_secret=b"coins",
                         mark_source=mark_source)


def run(c: sw.Challenger, respond=None, *, clock: Clock | None = None, spend=lambda b: 20_000) -> sw.SignedWipe:
    prover = sw.Prover(c.params)
    for start, data in c.fill(100):
        prover.receive(start, data)
    b = 0

    def answer(idx):
        nonlocal b
        if clock is not None:
            clock.now += spend(b)
        b += 1
        return (respond or prover.answer)(idx)

    c.run(answer)
    return c.close()


def test_label_format():
    key = sw.wipe_key(SECRET, SERVER, SESSION)
    assert key == derive(SECRET, sw.KEY_DOMAIN, {"server": SERVER, "session": SESSION}).value
    assert len(sw.LABEL_TAG) + len(key) + 8 == 75 < 112  # one SHA-512 block a label
    assert sw.label(key, 5) == hashlib.sha512(b"verity-pous/streamed-wipe/label/v1\0" + key + (5).to_bytes(8, "big")).digest()
    assert sw.labels(key, 3, 4) == b"".join(sw.label(key, i) for i in range(3, 7))
    assert sw.wipe_key(SECRET, b"server-8", SESSION) != key
    assert sw.wipe_key(SECRET, SERVER, bytes(32)) != key
    with pytest.raises(ValueError):
        sw.label(key, 1 << 64)


def test_draws_are_uniform_indices_from_the_coin_secret():
    idx = sw.draw_indices(b"coins", SERVER, SESSION, N, K)
    key = derive(b"coins", sw.COIN_DOMAIN, {"server": SERVER, "session": SESSION, "n": N})
    assert idx == tuple(key.uniform(N, j) for j in range(K))
    assert all(0 <= i < N for i in idx) and idx != sw.draw_indices(b"other", SERVER, SESSION, N, K)


@pytest.mark.parametrize("batch", [1, 2, 8, K])
def test_honest_session_is_accepted_at_batch_1_and_refused_above(batch):
    """An honest session runs to completion at any batch size, but only the sequential reveal is accepted: the guarantee
    doesn't cover a larger batch, so the verdict refuses it by name."""
    clock = Clock(1_000_000)
    signer = ts.TestSigner(b"wipe challenger", height=2)
    c = challenger(params(batch), clock=clock, signer=signer)
    signed = run(c, clock=clock)
    assert signed.ended == sw.COMPLETE and len(signed.batches) == -(-K // batch)
    assert sum(r.count for r in signed.batches) == K and all(r.wrong == 0 for r in signed.batches)
    v = sw.verdict(signed, c.commitment, trusted={signer.public_key}, schemes=ts.SCHEMES)
    uncovered = f"batch size {batch} is uncovered: the guarantee covers the sequential reveal, batch size 1"
    assert v.reasons == (() if batch == 1 else (uncovered,))
    assert v.public() == {"accepted": batch == 1}
    assert c.coins() == sw.draw_indices(b"coins", SERVER, SESSION, N, K)
    assert c.params.covered == (batch == 1)


def test_wrong_answer_ends_the_session():
    clock = Clock(1_000_000)
    signer = ts.TestSigner(b"wipe challenger", height=2)
    c = challenger(clock=clock, signer=signer)
    key = sw.wipe_key(SECRET, SERVER, SESSION)
    seen = []

    def respond(idx):
        seen.append(idx)
        return bytes(64) if len(seen) == 5 else b"".join(sw.label(key, i) for i in idx)

    signed = run(c, respond, clock=clock)
    assert signed.ended == sw.WRONG and len(signed.batches) == 5 and signed.batches[-1].wrong == 1
    with pytest.raises(RuntimeError, match="no batch is retried"):
        c.challenge()
    v = sw.verdict(signed, c.commitment, trusted={signer.public_key}, schemes=ts.SCHEMES)
    assert not v.accepted and v.public() == {"accepted": False}


def test_late_batch_ends_the_session():
    clock = Clock(1_000_000)
    signer = ts.TestSigner(b"wipe challenger", height=2)
    c = challenger(clock=clock, signer=signer)
    signed = run(c, clock=clock, spend=lambda b: LIMIT + 1 if b == 3 else 1000)
    assert signed.ended == sw.LATE and len(signed.batches) == 4
    assert not sw.verdict(signed, c.commitment, trusted={signer.public_key}, schemes=ts.SCHEMES).accepted


def test_challenges_follow_the_whole_fill_once():
    c = challenger()
    with pytest.raises(RuntimeError, match="once the fill is complete"):
        c.challenge()
    stream = c.fill(100)
    next(stream)
    with pytest.raises(RuntimeError, match="once the fill is complete"):
        c.challenge()
    list(stream)
    assert c.filled == N
    with pytest.raises(RuntimeError, match="streams once"):
        next(c.fill(100))
    c.challenge()
    with pytest.raises(RuntimeError, match="unanswered"):
        c.challenge()
    with pytest.raises(ValueError, match="answered with"):
        c.receive(bytes(63))


def test_one_session_an_id():
    log = ts.session_log()
    challenger(log=log)
    with pytest.raises(mc.SessionLogError):
        challenger(log=log)


DEFAULT_CLOCK = sw.Challenger.__init__.__kwdefaults__["clock"]
"""`Challenger`'s default clock as the module has it: read at import, before `conftest.py` replaces it in each test."""


def test_the_default_clock_is_the_raw_one_and_its_resolution_is_checked(monkeypatch):
    """`raw_clock_ns` reads `CLOCK_MONOTONIC_RAW`, the clock the header names. It is the default of `Challenger`, of the
    wipe's party and of an in-process wipe, and a challenger on it is refused where that clock ticks too coarsely."""
    asked = []
    monkeypatch.setattr(sw.time, "clock_gettime_ns", lambda clk: asked.append(("ns", clk)) or 7)
    assert sw.raw_clock_ns() == 7 and asked == [("ns", sw.time.CLOCK_MONOTONIC_RAW)]
    assert DEFAULT_CLOCK is sw.raw_clock_ns
    assert WP.serve.__kwdefaults__["clock"] is sw.raw_clock_ns and wipe_period.run.__kwdefaults__["clock"] is sw.raw_clock_ns
    monkeypatch.setattr(sw.time, "clock_getres", lambda clk: asked.append(("res", clk)) or 1e-9)
    assert challenger(clock=sw.raw_clock_ns).clock is sw.raw_clock_ns
    monkeypatch.setattr(sw.time, "clock_getres", lambda clk: asked.append(("res", clk)) or 4e-3)
    with pytest.raises(ValueError, match=f"the clock ticks every .* ns, coarser than {mc.CLOCK_RESOLUTION_NS} ns"):
        challenger(clock=sw.raw_clock_ns)
    assert asked[1:] == [("res", sw.time.CLOCK_MONOTONIC_RAW)] * 2
    assert challenger(clock=Clock(1_000_000)).commitment.header()["clock"] == sw.CLOCK == "CLOCK_MONOTONIC_RAW ns"


CERTIFICATE = b"a start certificate, as the network certifier returned it"


class Reads:
    """A clock that counts its reads."""

    def __init__(self, now: int = 1_000_000) -> None:
        self.now, self.reads = now, 0

    def __call__(self) -> int:
        self.reads += 1
        return self.now


def stream(c: sw.Challenger, how: str, sent: list) -> None:
    """The whole fill, by `fill` or by `fill_by`, each thing streamed noted in `sent`."""
    if how == "fill":
        for first, _ in c.fill(100):
            sent.append(first)
    else:
        c.fill_by(lambda key, n: sent.append(n))


def session(certificate: bytes | None = CERTIFICATE, server: bytes = SERVER):
    """An honest session and its signer: with `certificate`, the challenger's mark source returns it."""
    clock = Clock(1_000_000)
    signer = ts.TestSigner(b"wipe challenger", height=2)
    c = challenger(clock=clock, signer=signer, server=server,
                   mark_source=None if certificate is None else lambda m: certificate)
    return c, run(c, clock=clock), signer


def signed_bytes(s: sw.SignedWipe) -> bytes:
    """What a session signs, rebuilt here from the format: the header's 16 keys in sorted order, with no spaces, and no
    egress count among them."""
    p = s.commitment.params
    header = {"batch": p.batch, "batches": len(s.batches), "clock": "CLOCK_MONOTONIC_RAW ns",
              "ended": s.ended, "fill_end_ns": s.fill_end_ns, "fill_start_ns": s.fill_start_ns, "k": p.k,
              "label_bytes": 64, "labels": p.labels, "limit_ns": p.limit_ns, "protocol": "verity-pous/streamed-wipe/v2",
              "records_tree": "verity-pous/streamed-wipe/batch-records/frame-v3-sha512/v1",
              "server": s.commitment.server.hex(), "session": s.commitment.session.hex(),
              "start_mark": s.start_mark.hex(), "tiers": [[t, b] for t, b in p.tiers]}
    assert list(header) == sorted(header) and len(header) == 16 and "egress_bits" not in header
    head = json.dumps(header, separators=(",", ":")).encode()
    return hashlib.sha512(b"verity-pous/streamed-wipe/session/v2\0" + len(head).to_bytes(8, "big") + head
                          + sw.records_root(s.commitment.session, s.batches)).digest()


@pytest.mark.parametrize("how", ["fill", "fill_by"])
def test_the_start_mark_is_held_before_the_fill(how):
    """The mark source is called once, with the start message, before the fill's first clock read and before anything
    is streamed."""
    clock, sent, calls = Reads(), [], []

    def source(m: bytes) -> bytes:
        calls.append((m, clock.reads, list(sent)))
        return CERTIFICATE

    c = challenger(clock=clock, mark_source=source)
    assert calls == [] and c.start_mark == b"" and clock.reads == 0
    stream(c, how, sent)
    assert calls == [(sw.start_message(c.commitment), 0, [])] and sent and clock.reads == 2
    assert c.start_certificate == CERTIFICATE and c.start_mark == hashlib.sha512(CERTIFICATE).digest()


@pytest.mark.parametrize("how", ["fill", "fill_by"])
def test_a_mark_source_that_raises_streams_no_label(how):
    clock, sent = Reads(), []

    def source(m: bytes) -> bytes:
        raise ConnectionError("the network certifier is away")

    c = challenger(clock=clock, mark_source=source)
    with pytest.raises(ConnectionError, match="is away"):
        stream(c, how, sent)
    assert sent == [] and clock.reads == 0 and c.filled == 0 and c.fill_start_ns is None and c.start_mark == b""
    with pytest.raises(RuntimeError, match="once the fill is complete"):
        c.challenge()


@pytest.mark.parametrize("how", ["fill", "fill_by"])
@pytest.mark.parametrize("nothing", [b"", None])
def test_a_mark_source_that_returns_no_bytes_raises(how, nothing):
    clock, sent = Reads(), []
    c = challenger(clock=clock, mark_source=lambda m: nothing)
    with pytest.raises(ValueError, match="returned no start certificate"):
        stream(c, how, sent)
    assert sent == [] and clock.reads == 0 and c.filled == 0 and c.fill_start_ns is None and c.start_mark == b""


def test_the_signed_header_carries_the_certificates_digest():
    """The header's `start_mark` is the SHA-512 of exactly the bytes the source returned, under the session's one
    signature. One byte of the certificate changed is another mark and another message, and the first session's
    signature doesn't verify over it."""
    c, signed, signer = session()
    trusted = {signer.public_key}
    assert signed.start_mark == hashlib.sha512(CERTIFICATE).digest()
    assert signed.message() == signed_bytes(signed)
    assert sw.verdict(signed, c.commitment, trusted=trusted, schemes=ts.SCHEMES).accepted
    for i in (0, len(CERTIFICATE) // 2, len(CERTIFICATE) - 1):
        changed = bytes(b ^ (j == i) for j, b in enumerate(CERTIFICATE))
        other = session(changed)[1]
        assert other.start_mark == hashlib.sha512(changed).digest() != signed.start_mark
        assert other.message() == signed_bytes(other) != signed.message()
        assert dataclasses.replace(other, start_mark=signed.start_mark, signature=signed.signature) == signed
        forged = dataclasses.replace(signed, start_mark=other.start_mark)
        assert sw.verdict(forged, c.commitment, trusted=trusted, schemes=ts.SCHEMES).reasons == \
            ("the signature does not verify",)
    for longer in (CERTIFICATE + b"\0", CERTIFICATE[:-1]):
        assert session(longer)[1].start_mark == hashlib.sha512(longer).digest() != signed.start_mark


def test_a_session_with_no_mark_source_signs_an_empty_mark():
    """With no mark source the header's `start_mark` is the empty string, and the session is judged as any other."""
    c, signed, signer = session(None)
    assert c.start_certificate == b"" and signed.start_mark == b"" and signed.to_json()["start_mark"] == ""
    assert signed.message() == signed_bytes(signed)
    assert sw.verdict(signed, c.commitment, trusted={signer.public_key}, schemes=ts.SCHEMES).accepted
    marked = dataclasses.replace(signed, start_mark=hashlib.sha512(CERTIFICATE).digest())
    assert sw.verdict(marked, c.commitment, trusted={signer.public_key}, schemes=ts.SCHEMES).reasons == \
        ("the signature does not verify",)


def test_the_start_message_is_the_commitment():
    """The start message is the tagged SHA-512 of the commitment's header alone, and one byte of the commitment
    changed is another start message, which the mark source is handed."""
    c = challenger()
    head = ('{"batch":1,"clock":"CLOCK_MONOTONIC_RAW ns","k":40,"label_bytes":64,"labels":256,"limit_ns":1000000,'
            '"protocol":"verity-pous/streamed-wipe/v2",'
            '"records_tree":"verity-pous/streamed-wipe/batch-records/frame-v3-sha512/v1",'
            f'"server":"{SERVER.hex()}","session":"{SESSION.hex()}",'
            '"tiers":[["gpu",8192],["host-dram",8192]]}').encode()
    start = sw.start_message(c.commitment)
    assert start == hashlib.sha512(b"verity-pous/streamed-wipe/start/v1\0" + len(head).to_bytes(8, "big")
                                   + head).digest()
    flip = lambda b, i: bytes(x ^ (j == i) for j, x in enumerate(b))  # noqa: E731
    others = [dataclasses.replace(c.commitment, server=flip(SERVER, 0)),
              dataclasses.replace(c.commitment, server=flip(SERVER, len(SERVER) - 1)),
              dataclasses.replace(c.commitment, session=flip(SESSION, 0)),
              dataclasses.replace(c.commitment, session=flip(SESSION, 31)),
              dataclasses.replace(c.commitment, params=params(limit=LIMIT + 1)),
              dataclasses.replace(c.commitment, params=params(batch=2)),
              dataclasses.replace(c.commitment, params=sw.WipeParams(N, K + 1, 1, LIMIT, params().tiers)),
              dataclasses.replace(c.commitment, params=sw.WipeParams(N - 1, K, 1, LIMIT, params().tiers)),
              dataclasses.replace(c.commitment, params=sw.WipeParams(N, K, 1, LIMIT, (("gpu", N * sw.LABEL_BYTES),)))]
    assert len({start, *(sw.start_message(x) for x in others)}) == 1 + len(others)
    seen = []
    elsewhere = challenger(server=flip(SERVER, 0), mark_source=lambda m: seen.append(m) or CERTIFICATE)
    next(elsewhere.fill(100))
    assert seen == [sw.start_message(others[0])] and seen != [start]


def test_the_end_message_is_the_signed_message_and_its_signature():
    """The end message is the tagged SHA-512 of the session's signed message and its signature, each behind its length:
    it changes with either."""
    c, signed, _ = session()
    m, sig = signed.message(), signed.signature
    end = sw.end_message(signed)
    assert end == hashlib.sha512(b"verity-pous/streamed-wipe/end/v1\0" + len(m).to_bytes(8, "big") + m
                                 + len(sig).to_bytes(8, "big") + sig).digest()
    assert len(end) == len(sw.start_message(c.commitment)) == 64 and end != sw.start_message(c.commitment)
    r = signed.batches[0]
    others = [dataclasses.replace(signed, signature=bytes([sig[0] ^ 1]) + sig[1:]),
              dataclasses.replace(signed, signature=sig[:-1] + bytes([sig[-1] ^ 0x80])),
              dataclasses.replace(signed, signature=sig[:-1]), dataclasses.replace(signed, signature=b""),
              dataclasses.replace(signed, fill_end_ns=signed.fill_end_ns + 1),
              dataclasses.replace(signed, start_mark=b""),
              dataclasses.replace(signed, start_mark=bytes([signed.start_mark[0] ^ 1]) + signed.start_mark[1:]),
              dataclasses.replace(signed, batches=(dataclasses.replace(r, receipt_ns=r.receipt_ns + 1),
                                                   *signed.batches[1:]))]
    assert len({end, *(sw.end_message(s) for s in others)}) == 1 + len(others)


def test_the_reader_refuses_another_format_or_mark():
    """`wipe_party.signed_of` reads a session back with its mark, empty or not; it refuses a v1 session by name, a v2
    session whose header names another clock than the raw one, and a `start_mark` of another length, in upper case, or
    that is no hex."""
    for certificate in (CERTIFICATE, None):
        signed = session(certificate)[1]
        j = json.loads(json.dumps(signed.to_json()))
        assert set(j) == {"commitment", "ended", "start_mark", "fill_start_ns", "fill_end_ns", "batches",
                          "scheme", "public_key", "signature"}
        assert WP.signed_of(j) == signed
        assert j["commitment"]["clock"] == sw.CLOCK == "CLOCK_MONOTONIC_RAW ns"
        for other in ("CLOCK_MONOTONIC ns", "CLOCK_MONOTONIC_RAW", "CLOCK_REALTIME ns", None):
            with pytest.raises(ValueError, match=f"a signed wipe's times are on CLOCK_MONOTONIC_RAW ns, not {other!r}"):
                WP.signed_of(j | {"commitment": j["commitment"] | {"clock": other}})
        with pytest.raises(ValueError, match="a signed wipe's times are on CLOCK_MONOTONIC_RAW ns, not None"):
            WP.signed_of(j | {"commitment": {k: v for k, v in j["commitment"].items() if k != "clock"}})
        v1 = {k: v for k, v in j.items() if k != "start_mark"} | {"egress_bits": 1234}
        v1["commitment"] = j["commitment"] | {"protocol": "verity-pous/streamed-wipe/v1"}
        for old in (v1, v1 | {"start_mark": j["start_mark"]}):
            with pytest.raises(ValueError, match="a signed wipe is verity-pous/streamed-wipe/v2, not "
                                                 "'verity-pous/streamed-wipe/v1'"):
                WP.signed_of(old)
        with pytest.raises(KeyError, match="start_mark"):
            WP.signed_of({k: v for k, v in j.items() if k != "start_mark"})
    mark = j["start_mark"] = hashlib.sha512(CERTIFICATE).hexdigest()
    assert len(mark) == 128 and WP.signed_of(j).start_mark == bytes.fromhex(mark)
    letter = next(i for i, x in enumerate(mark) if x in "abcdef")
    for bad in (mark.upper(), mark[:letter] + mark[letter].upper() + mark[letter + 1:], mark[:-2], mark + "00",
                mark[:64], "00", mark[:-1], mark[:-1] + "g", " " + mark[1:], mark[:-1] + " ", "0x" + mark[2:], None, 0,
                bytes.fromhex(mark)):
        with pytest.raises(ValueError, match="a start mark is 128 lowercase hex digits, or empty"):
            WP.signed_of(j | {"start_mark": bad})
    assert sw.mark_of("") == b"" and sw.mark_of(mark) == bytes.fromhex(mark)


def test_the_declared_tiers_hold_the_fill():
    with pytest.raises(ValueError, match="the declared tiers hold the fill"):
        sw.WipeParams(N - 1, K, 1, LIMIT, (("gpu", (N - 2) * sw.LABEL_BYTES),))
    p = sw.WipeParams(N - 1, K, 1, LIMIT, (("gpu", N * sw.LABEL_BYTES), ("host-dram", 3)))
    assert p.outside_bits == sw.LABEL_BITS + 24
    with pytest.raises(ValueError, match="each named once"):
        sw.WipeParams(N, K, 1, LIMIT, (("gpu", 128 * sw.LABEL_BYTES), ("gpu", 128 * sw.LABEL_BYTES)))
    with pytest.raises(ValueError):
        sw.WipeParams(N, K, K + 1, LIMIT, (("gpu", N * sw.LABEL_BYTES),))


def test_verdict_fails_closed():
    clock = Clock(1_000_000)
    signer = ts.TestSigner(b"wipe challenger", height=2)
    c = challenger(clock=clock, signer=signer)
    signed = run(c, clock=clock)
    trusted = {signer.public_key}
    assert not sw.verdict(signed, c.commitment, trusted=set(), schemes=ts.SCHEMES).accepted
    assert not sw.verdict(signed, c.commitment, trusted=trusted).accepted  # a test scheme outside a test's mapping
    other = dataclasses.replace(c.commitment, params=params(limit=LIMIT + 1))
    assert not sw.verdict(signed, other, trusted=trusted, schemes=ts.SCHEMES).accepted
    r = signed.batches[0]
    forged = dataclasses.replace(signed, batches=(dataclasses.replace(r, receipt_ns=r.receipt_ns + 1), *signed.batches[1:]))
    assert "the signature does not verify" in sw.verdict(forged, c.commitment, trusted=trusted, schemes=ts.SCHEMES).reasons
    more = dataclasses.replace(signed, fill_start_ns=signed.fill_start_ns - 1)
    assert not sw.verdict(more, c.commitment, trusted=trusted, schemes=ts.SCHEMES).accepted


def test_the_session_signs_no_egress():
    """`close` takes no egress count, the signed session holds none and its verdict returns no amount."""
    clock = Clock(1_000_000)
    signer = ts.TestSigner(b"wipe challenger", height=2)
    c = challenger(clock=clock, signer=signer)
    signed = run(c, clock=clock)
    with pytest.raises(TypeError):
        c.close(0)
    with pytest.raises(TypeError):
        c.close(egress_bits=0)
    assert c.close() is signed
    assert not hasattr(signed, "egress_bits") and "egress_bits" not in signed.to_json()
    assert signed.message() == signed_bytes(signed)
    v = sw.verdict(signed, c.commitment, trusted={signer.public_key}, schemes=ts.SCHEMES)
    assert v == sw.Verdict(True, ()) and v.public() == {"accepted": True} and not hasattr(v, "egress_mib")


def test_public_egress_is_15_bits_of_mib_rounded_up():
    """The bound reads the egress `E`, so the wipe's check publishes it, rounded up to whole MiB and saturating at 15
    bits: with the verdict, 16 bits, the least public size class."""
    mib = sw.EGRESS_UNIT_BITS
    assert [sw.egress_units(e) for e in (0, 1, mib, mib + 1)] == [0, 1, 1, 2]
    assert sw.egress_units(sw.EGRESS_MAX_UNITS * mib) == sw.egress_units(10**15) == sw.EGRESS_MAX_UNITS == 2**15 - 1


def test_deployed_bound():
    n, k = sw.DEPLOYED_LABELS, sw.DEPLOYED_K
    assert n == 40_625_000_000
    q, g = sw.retained_bits(n, k, n * sw.LABEL_BITS, 0, -40)
    assert n - g == 15_598_671 and q == 8_186_012_770  # 0.998 GB of labels missing; 1.0233 GB of prior content kept
    assert sw.log2_error(n, k, g, n * sw.LABEL_BITS, 0, q) <= -40
    assert sw.log2_error(n, k, g, n * sw.LABEL_BITS, 0, q - 64) > -40
    q_e, _ = sw.retained_bits(n, k, n * sw.LABEL_BITS, 10**9, -40)
    assert q_e == q + 10**9  # egress adds to what can be kept, bit for bit
    assert sw.challenges_for(n, 10**9 // 64, -40) == 72_074
    # The Lean corollary's explicit point (`Pous.Guarantees.StreamedWipeMeets2600GB`): D = 15,610,000, its crude count.
    d = 15_610_000
    assert k * math.log2(1 - d / n) < -41
    assert math.log2(k) + 36 * (d - 1) + (d - 1) * sw.LABEL_BITS + 41 <= 8_600_000_000


@pytest.mark.parametrize("n", [1 << 19, (1 << 19) + 7, 1_610_612_736, 2_203_125_000, 1 << 50])
def test_bound_at_2152_reads(n: int) -> None:
    """The Lean family's point (`Pous.Guarantees.StreamedWipeMeetsFamily`, `⌈M/19⌉ + E` at `2^-128` with 2152 reads):
    `D = ⌈n/24⌉` labels missed, its crude count `C(n, r) ≤ 2^n`, each term `2^-129`; the exact count needs less."""
    k, m = 2152, n * sw.LABEL_BITS
    d = -(-n // 24)
    assert k * d / n >= 129 * 0.6931471808 and k * math.log2(1 - d / n) < -129
    assert 12 + n + (d - 1) * sw.LABEL_BITS + 129 <= -(-m // 19)
    q, _ = sw.retained_bits(n, k, m, 0, -128)
    assert q <= -(-m // 19)
    assert 18**k * 2**128 <= 19**k  # (18/19)^2152 ≤ 2^-128, though the proof's sampling is at 23/24


def reads_threshold(n: int, r: int, k: int, reserve_bits: int = 0) -> int:
    """`Pous.Guarantees.StreamedWipeMeetsReads`'s threshold before the egress: `512·(⌈n/r⌉ − 1) + n + ⌊log₂ k⌋ + 130 + R`."""
    return sw.LABEL_BITS * (-(-n // r) - 1) + n + (k.bit_length() - 1) + 130 + reserve_bits


def least_reads(r: int) -> int:
    """The least `k` with `(r − 1)^k · 2^129 ≤ r^k`, the statement's sampling condition."""
    k = math.floor(129 / -math.log2(1 - 1 / r))
    while (r - 1) ** k * 2**129 > r**k:
        k += 1
    while (r - 1) ** (k - 1) * 2**129 <= r ** (k - 1):
        k -= 1
    return k


def test_h200_at_5GB() -> None:
    """`StreamedWipeMeets141GB5`: at 141·10^9 bytes with 2.3 GB of driver reserve, `r = 59` is the least ratio whose
    threshold is at most 5 GB, `k = 5231` the least read count for it, and the threshold `39,721,768,854` bits; the
    family's own point, `r = 24` and `k = 2152`, is `512·(⌈n/24⌉ − 1) + n + 141`."""
    n, reserve, cap = 2_203_125_000, 23 * 10**8 * 8, 5 * 10**9 * 8
    assert least_reads(59) == 5231 and least_reads(24) == 2101
    assert reads_threshold(n, 59, 5231, reserve) == 39_721_768_854 <= cap
    assert reads_threshold(n, 59, 5231) == 21_321_768_854 <= cap - reserve
    assert reads_threshold(n, 58, least_reads(58), reserve) > cap
    assert reads_threshold(n, 24, 2152) == sw.LABEL_BITS * (-(-n // 24) - 1) + n + 141
    q, _ = sw.retained_bits(n, 5231, n * sw.LABEL_BITS + reserve, 0, -128)
    assert q <= reads_threshold(n, 59, 5231, reserve)  # the exact count needs less


def log2_floor(x: int) -> int:
    """`Nat.log2`: `⌊log₂ x⌋`, and `0` at `x = 0`."""
    return max(x.bit_length() - 1, 0)


def count_bits(n: int, m: int) -> int:
    """`Pous.StreamedWipe.countBits`: `m·(⌊log₂⌊3n/m⌋⌋ + 1)`, `0` at `m = 0`."""
    return m * (log2_floor(3 * n // m) + 1) if m else 0


def server_bound(n: int, r: int, k: int, reserve_bits: int = 0) -> int:
    """`Pous.StreamedWipe.serverBound`, `B(n, r, k, R)`: the fill's share `512·m + countBits n m + ⌊log₂ k⌋ + 130`,
    `m = ⌈n/r⌉ − 1`, plus `R`."""
    m = -(-n // r) - 1
    return sw.LABEL_BITS * m + count_bits(n, m) + log2_floor(k) + 130 + reserve_bits


def reads_exponent(r: int, k: int) -> float:
    """`−k·log₂(1 − 1/r)`: `(r − 1)^k · 2^129 ≤ r^k` is this at least 129."""
    return -k * math.log1p(-1 / r) / math.log(2)


FIVE_GB = 5 * 10**9 * 8
H200, RAM, DISK = 141 * 10**9, 2 * 10**12, 12 * 10**12
MINIMAL = (8 * 3 * 10**8 + 15 * 10**8) * 8  # 8 × 0.3 GB of minimal contexts and a 1.5 GB host OS
# vy-dc-h200-2's card as measured (`Pous.StreamedWipe.h200`): 143,771 MiB, a fill of 4,425
# segments of 32 MiB.
H200_MS, H200_FILL = 143_771 * 2**20 * 8, 4425 * 2**25
H200_R = H200_MS - sw.LABEL_BITS * (H200_FILL // sw.LABEL_BYTES)


def test_count_bits_bounds_the_count() -> None:
    """`PousStreamedWipe.choose_le_two_pow_countBits`, `C(n, m) ≤ 2^countBits n m`, at every `m ≤ n ≤ 80`; it needs
    the `+ 1`, which `m·(⌊log₂⌊n/m⌋⌋ + 2)` lacks at large `m`."""
    for n in range(81):
        for m in range(n + 1):
            assert math.comb(n, m) <= 2 ** count_bits(n, m)
    n, m = 3150, 50  # n/m = 63, just below 2^6
    assert 2 ** (m * (log2_floor(n // m) + 2)) < math.comb(n, m) <= 2 ** count_bits(n, m)


def test_tight_threshold_below_reads_from_r_4() -> None:
    """`StreamedWipeMeetsReadsTight`'s threshold is at most `StreamedWipeMeetsReads`'s once `r ≥ 4`, and above it at
    `r = 3`, where the count `2^n` is the smaller."""
    for n in (1, 2, 7, 100, 4096, 10**6 + 3, 2_203_125_000):
        for r in (4, 5, 7, 24, 59, 1000, 15_583):
            assert server_bound(n, r, 2152) <= reads_threshold(n, r, 2152)
    assert server_bound(10**6, 3, 2152) > reads_threshold(10**6, 3, 2152)


@pytest.mark.parametrize("fill, reserve, r, k, bound", [
    (H200_FILL, H200_R, 56, 4963, 39_754_287_390),
    (8 * H200, MINIMAL, 1050, 93_843, 39_995_714_282),
    (8 * H200 + RAM, MINIMAL, 2922, 261_229, 39_998_169_129),
    (8 * H200 + RAM + DISK, MINIMAL, 14_183, 1_268_143, 39_999_689_862),
])
def test_servers_at_5GB(fill: int, reserve: int, r: int, k: int, bound: int) -> None:
    """The measured H200's least point, `…8H200HBM`, `…8H200HBMRAM` and `…8H200`: `B(n, r, k, R)` at the row's `r`
    and `k` is at most 5 GB, `r` is the least ratio that is, and `k` the least read count for `r` (the kernel decides
    `(r − 1)^k · 2^129 ≤ r^k`; here the exponent, with a margin far above the float's error, and exactly where cheap).
    The exact count needs less than `countBits`."""
    n = fill // sw.LABEL_BYTES
    assert n * sw.LABEL_BYTES == fill
    assert server_bound(n, r, k, reserve) == bound <= FIVE_GB
    assert reads_exponent(r, k) > 129 + 1e-6 > 129 - 1e-6 > reads_exponent(r, k - 1)
    if k < 10**5:
        assert least_reads(r) == k
    k_less = math.ceil(129 / reads_exponent(r - 1, 1))
    assert reads_exponent(r - 1, k_less) > 129 + 1e-9 and server_bound(n, r - 1, k_less, reserve) > FIVE_GB
    m = -(-n // r) - 1
    assert sw._log2_choose(n, m) <= count_bits(n, m)
    assert server_bound(n, r, k, 9_600_000_000 * 8) > FIVE_GB  # R = 8 × 1.2 GB, today's contexts: 9.6 GB alone


def test_h200_as_deployed() -> None:
    """`StreamedWipeMeetsServerH200` and the honest lemmas at `Pous.StreamedWipe.h200`: the measured card is
    `512·n + R` bits, the deployed `k = 5500` reads at `r = 62`, the largest `r` with `(r − 1)^k · 2^129 ≤ r^k`, and
    `B = 37,669,517,430` bits (4.709 GB) within the 5 GB point, the fill's share `19,457,849,462`."""
    n = H200_FILL // sw.LABEL_BYTES
    assert (n, H200_MS, H200_R) == (2_319_974_400, 1_206_038_560_768, 18_211_667_968)
    assert 61**5500 * 2**129 <= 62**5500 and 62**5500 * 2**129 > 63**5500
    assert server_bound(n, 62, 5500, H200_R) == 37_669_517_430 <= FIVE_GB
    assert server_bound(n, 62, 5500) == 19_457_849_462


def test_tight_count_at_the_whole_server() -> None:
    """The coordinator's estimate: the fill's share of a whole 8×H200 server (HBM, 2 TB of RAM, 12 TB of disk) is
    1.0 GB at `r = 15,583` and `k = 1,393,325`, where `StreamedWipeMeetsReads`'s is 30.5 GB, and of its HBM alone
    1.0 GB at `k = 103,142`."""
    n = (8 * H200 + RAM + DISK) // sw.LABEL_BYTES
    assert reads_exponent(15_583, 1_393_325) > 129 > reads_exponent(15_583, 1_393_324)
    assert server_bound(n, 15_583, 1_393_325) == 8_009_112_294
    assert reads_threshold(n, 15_583, 1_393_325) == 244_141_411_926
    assert reads_exponent(1154, 103_142) > 129 > reads_exponent(1154, 103_141)
    assert server_bound(8 * H200 // sw.LABEL_BYTES, 1154, 103_142) == 8_003_032_758


def test_bound_against_an_exhaustive_count():
    """N = 3 labels of 2 bits, k = 2, q = 2 bits of prior content, M = 4 bits inside, no egress: a prover that keeps the
    prior content and label 0, and answers 0 for every other label, wins exactly as often as counted, and the bound
    (minimized over g, in exact arithmetic) is above it."""
    n, b, k, q, m = 3, 2, 2, 2, 4
    wins = total = 0
    for z, labels, c in itertools.product(range(4), itertools.product(range(4), repeat=n),
                                          itertools.product(range(n), repeat=k)):
        total += 1
        sigma = (z, labels[0])
        answers = [sigma[1] if i == 0 else 0 for i in c]
        wins += all(a == labels[i] for a, i in zip(answers, c)) and sigma[0] == z
    pr = wins / total
    bound = min((g / n) ** k + k * math.comb(n, g + 1) * 2 ** (m - q - (g + 1) * b) for g in range(n + 1))
    assert pr <= bound
    assert pr == pytest.approx((1 + 4 / 4 + 2 / 4 + 2 / 16) / 9)  # c by the distinct labels it asks beyond label 0
