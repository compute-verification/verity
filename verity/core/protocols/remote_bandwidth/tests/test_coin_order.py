"""The coin commitment reaches the prover only at the session's end: soundness takes no hiding premise on a challenger's
root, so the prover is sent each coin as its value and the root it was opened after (`coins.Released`, no salt or
path), and once the session has its outcome the verifier discloses the commitment and every released coin's opening
(`audit.Session.disclose`, refused before), against which the prover checks each coin it was sent (`audit.Prover.end`).
In one process (`period.session`) and across a connection (`verifier_party`) no message before the end carries the
commitment, and at the end a coin the commitment doesn't open as it was sent is refused."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import socket
import threading

import pytest

from verity.core.protocols.remote_bandwidth import audit, verifier_party
from verity.core.protocols.remote_bandwidth import coins as C
from verity.core.protocols.remote_bandwidth import period as P

STORE = hashlib.shake_256(b"a tiny MLP's weights").digest(3 * 4032 - 17)
COINS = hashlib.sha512(b"the period's live coins").digest()


class Clock:
    """The verifier's clock: each reading a microsecond after the last, and waiting moves it on."""

    def __init__(self) -> None:
        self.now = 10 ** 9

    def __call__(self) -> int:
        self.now += 1_000
        return self.now

    def sleep_until(self, clock, t: int) -> None:
        self.now = max(self.now, t)


class Entropy:
    """A deterministic stand-in for the operating system's generator: SHA-512 in counter mode under `label`."""

    def __init__(self, label: bytes) -> None:
        self.label, self.count = label, 0

    def __call__(self, n: int) -> bytes:
        out = b""
        while len(out) < n:
            out += hashlib.sha512(self.label + self.count.to_bytes(8, "big")).digest()
            self.count += 1
        return out[:n]


class Recorded(P.Local):
    """The in-process prover, each call it takes kept in order (`calls`)."""

    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)
        self.calls: list[tuple] = []

    def hold(self, registration) -> None:
        self.calls.append(("hold", registration))
        super().hold(registration)

    def run(self, challenge, t, i, previous) -> bytes:
        self.calls.append(("run", challenge))
        return super().run(challenge, t, i, previous)

    def open(self, t, draws):
        self.calls.append(("open", t))
        return super().open(t, draws)

    def end(self, commitment, openings) -> None:
        self.calls.append(("end", commitment, tuple(openings)))
        super().end(commitment, openings)


def honest(verify: int | None = None) -> tuple[P.Period, Recorded]:
    c = Clock()
    prover = Recorded("n0", STORE, entropy=Entropy(b"salts"))
    return P.session("n0", prover, COINS, verify=verify, clock=c, sleep_until=c.sleep_until), prover


@pytest.fixture(scope="module")
def ended() -> tuple[P.Period, Recorded]:
    return honest()


def test_each_coin_goes_bare_and_the_commitment_only_with_the_end(ended) -> None:
    period, prover = ended
    assert period.verdict == "accept"
    kinds = [c[0] for c in prover.calls]
    assert kinds[0] == "hold" and kinds[-1] == "end" and kinds.count("end") == 1, kinds
    assert all(type(c[1]) is C.Released for c in prover.calls if c[0] == "run")
    _, commitment, openings = prover.calls[-1]
    assert C.layout_reason(commitment, 0, 2) is None
    sent = prover.prover.received
    assert sorted(sent) == [0, 1] and C.received_reason(commitment, [sent[0], sent[1]], openings) is None
    labels = [(o.label, o.index) for o in openings]
    assert labels == [(C.CHALLENGE, 0), (C.CHALLENGE, 1), (C.OPENINGS, 0), (C.OPENINGS, 1)]
    case = json.loads(bytes.fromhex(period.case["evidence"]))
    assert [C.coin(o).hex() for o in openings] == case["coins"]


def test_a_session_left_undecided_discloses_nothing() -> None:
    period, prover = honest(verify=1)
    assert period.verdict == "undecided" and "end" not in [c[0] for c in prover.calls]


def test_the_disclosure_is_refused_until_the_session_has_its_outcome() -> None:
    n, root = P.Local("n0", STORE, entropy=Entropy(b"salts")).register()
    reg = audit.register(COINS, b"n0", n, root)
    params = audit.Params(blocks=reg.blocks, deadline_ns=500_000_000, windows=2, window_ns=100_000_000, codec=reg.codec,
                          receipt=reg.receipt, rows=reg.rows)
    clock = Clock()
    s = audit.Session(params, reg, None, C.local(reg.digest, 0, 2, Entropy(b"coins")), not_before_ns=clock.now,
                      clock=clock)
    with pytest.raises(audit.SessionError, match="disclosed at its end"):
        s.disclose()
    sigma0 = s.start()
    with pytest.raises(audit.SessionError, match="disclosed at its end"):
        s.disclose()
    v = s.receive(0, 0, bytes(64))
    assert v is not None and not v.accepted and s.outcome is not None
    assert s.disclose() == (s.commitment, (sigma0,))


def test_the_end_refuses_a_coin_the_commitment_doesnt_open_as_it_was_sent(ended) -> None:
    _, prover = ended
    _, commitment, openings = prover.calls[-1]
    sent = [prover.prover.received[t] for t in (0, 1)]
    flipped = dataclasses.replace(sent[1], coin=bytes([sent[1].coin[0] ^ 1]) + sent[1].coin[1:])
    assert "is not the one opened" in C.received_reason(commitment, [sent[0], flipped], openings)
    moved = dataclasses.replace(sent[1], after=bytes(64))
    assert "is not the one opened" in C.received_reason(commitment, [sent[0], moved], openings)
    assert "has no opening" in C.received_reason(commitment, sent, openings[1:])
    assert "opened twice" in C.received_reason(commitment, sent, openings + openings[:1])
    forged = dataclasses.replace(openings[1], coin=flipped.coin)
    why = C.received_reason(commitment, [sent[0], flipped], (openings[0], forged) + openings[2:])
    assert why is not None and why.startswith("the challenge coin 1 the prover was sent: ")
    other = C.local(b"\x01" * 64, 0, 2, Entropy(b"other")).commitment
    assert C.received_reason(other, sent, openings) is not None
    with pytest.raises(ValueError, match="has no opening"):
        prover.prover.end(commitment, openings[1:])


def test_a_prover_is_refused_a_second_coin_for_a_window(ended) -> None:
    _, prover = ended
    sent = prover.prover.received[0]
    again = audit.Prover(prover.prover.store, None, entropy=Entropy(b"again"))
    again.run(sent, 0, prover.prover.store.registration.digest)
    with pytest.raises(ValueError, match="sent again, another coin"):
        again.run(dataclasses.replace(sent, coin=bytes(64)), 1, prover.prover.store.registration.digest)
    with pytest.raises(ValueError, match="not a window's challenge"):
        again.run(dataclasses.replace(sent, label=C.OPENINGS), 0, prover.prover.store.registration.digest)


def test_a_released_coin_round_trips_its_json_and_nothing_else_reads_as_one(ended) -> None:
    _, prover = ended
    sent = prover.prover.received[0]
    doc = json.loads(json.dumps(sent.to_json()))
    assert sorted(doc) == ["after", "coin", "epoch", "index", "label"] and C.Released.from_json(doc) == sent
    opening = prover.calls[-1][2][0]
    for bad in (opening.to_json(), doc | {"salt": "00"}, doc | {"coin": doc["coin"].upper()}, doc | {"coin": "0"},
                doc | {"index": True}, doc | {"epoch": "0"}, [doc]):
        with pytest.raises(C.CoinError, match="a released coin is"):
            C.Released.from_json(bad)
    assert C.released(opening) == C.Released.of(opening) and C.released(sent) is sent
    with pytest.raises(C.CoinError):
        C.released(doc)


class Tee:
    """The prover's end of the connection, every line it reads kept (`lines`)."""

    def __init__(self, f) -> None:
        self.f, self.lines = f, []

    def readline(self) -> bytes:
        line = self.f.readline()
        self.lines.append(json.loads(line))
        return line

    def write(self, b: bytes) -> int:
        return self.f.write(b)

    def flush(self) -> None:
        self.f.flush()


def test_across_the_connection_no_message_before_the_end_carries_the_commitment() -> None:
    a, b = socket.socketpair()
    got: dict = {}

    def verifier() -> None:
        c = Clock()
        with a, a.makefile("rwb") as f:
            got["period"] = verifier_party.serve(f, verifier_party.request_of("n0"), coins=COINS, clock=c,
                                                 sleep_until=c.sleep_until)

    th = threading.Thread(target=verifier)
    th.start()
    prover = Recorded("n0", STORE, entropy=Entropy(b"salts"))
    with b, b.makefile("rwb") as f:
        tee = Tee(f)
        verifier_party.prove(tee, prover)
    th.join()
    assert got["period"].verdict == "accept"
    *during, last = tee.lines
    assert [next(iter(m)) for m in during][0] == "hold" and next(iter(last)) == "end"
    for m in during:
        text = json.dumps(m)
        assert all(word not in text for word in ('"commitment"', '"salt"', '"path"')), m
    assert sorted(last["end"]) == ["commitment", "openings"]
    assert prover.calls[-1][1] == C.LiveCommitment.from_json(last["end"]["commitment"])
