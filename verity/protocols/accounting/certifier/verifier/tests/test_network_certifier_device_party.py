"""The network certifier's device as a party (`device_party`): its declared sends are exactly the manifest's, each part of
its life not yet written raises `NotImplementedError` naming its goal, what it hands out encodes on the manifest's
channels, and its turns run a session to its end once those parts answer."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from verity.channel import Message
from verity.party import Finish, Identity, Received, Send, Timer
from verity.protocols.accounting.certifier.device import device_party as D
from verity.protocols.accounting.certifier.verifier import out_commit as O
from verity.protocols.accounting.certifier.verifier import parties as NC
from verity.protocols.accounting.certifier.verifier import warden_rows as WR
from verity.protocols.accounting.certifier.verifier.warden import WardenParams

LINK = "node0/uplink0"
SCHEDULE = WardenParams(window=4, bucket_ms=10)
PARAMS = {"params": asdict(SCHEDULE), "windows": 2, "origin": 1_000}
IDENTITY = Identity(NC.NETWORK_CERTIFIER, capabilities={"signing-key": object()})


def _kept(window: int) -> tuple[D.Kept, ...]:
    return tuple(D.Kept(name, LINK, bytes([window, i]) * 8) for i, name in enumerate(NC.RECORD_NAMES))


def _phase(kept) -> D.Phase:
    window = kept[0].packed[0]
    records = tuple(D.SignedRecord(k, b'{"certified":1}', *((b'{"registration":1}', b"\x07" * WR.WIRING_SALT_BYTES)
                                                             if k.name != "fails" else ()))
                    for k in kept)
    return D.Phase(window, LINK, records, D.Out(b"\x05" * O.SALT_BYTES * 2, b'{"root":"00"}'))


class Answering(D.NetworkCertifier):
    """The party with its unwritten parts answering from the fixtures above."""

    def open_session(self, identity: Identity) -> D.Opened:
        return D.Opened(b'{"deployment":1}', b'{"load":1}')

    def certify(self, window: int) -> tuple[D.Kept, ...]:
        return _kept(window)

    def sign(self, kept) -> D.Phase:
        return _phase(kept)

    def collect(self) -> bytes | None:
        return b'{"groups":[]}'


def test_the_partys_sends_are_exactly_the_manifests():
    assert set(D.SENDS) == set(NC.MANIFEST.peers(NC.NETWORK_CERTIFIER))
    for peer, types in D.SENDS.items():
        assert len(set(types)) == len(types)
        assert set(types) == NC.MANIFEST.sends(NC.NETWORK_CERTIFIER, peer).types
    assert D.PROTOCOL.manifest is NC.MANIFEST and set(D.PROTOCOL.parties) == {NC.NETWORK_CERTIFIER}
    assert isinstance(D.PROTOCOL.parties[NC.NETWORK_CERTIFIER](), D.NetworkCertifier)


@pytest.mark.parametrize("call", [
    lambda p: p.open_session(IDENTITY),
    lambda p: p.certify(0),
    lambda p: p.sign(_kept(0)),
    lambda p: p.collect(),
    lambda p: p.start(PARAMS, IDENTITY),
], ids=["open_session", "certify", "sign", "collect", "start"])
def test_each_unwritten_part_raises_naming_its_goal(call):
    with pytest.raises(NotImplementedError, match=f"^{D.GOAL}: "):
        call(D.NetworkCertifier())


def _encoded(actions) -> list[tuple[str, str]]:
    sent = []
    for a in actions:
        if isinstance(a, Send):
            types = NC.MANIFEST.sends(NC.NETWORK_CERTIFIER, a.channel)
            assert types.decode(types.encode(a.message)) == a.message
            sent.append((a.channel, a.message.type))
    return sent


def test_what_it_hands_out_encodes_on_the_manifests_channels():
    p = D.NetworkCertifier()
    sent = _encoded(p.hand_out(_phase(_kept(1)))) + _encoded(p.hand_out_run(D.Opened(b"{}", b"{}")))
    sent += _encoded(p.close(b"{}"))
    assert {t for _, t in sent} == {t for types in D.SENDS.values() for t in types}
    assert all(t in D.SENDS[ch] for ch, t in sent)
    assert sent.count((NC.WORKERS, NC.RECORDS)) == sent.count((NC.VERIFIER, NC.CERTIFICATE)) == len(NC.RECORD_NAMES)


def test_its_turns_run_a_session_to_its_end():
    p = Answering()
    first = p.start(PARAMS, IDENTITY)
    span = SCHEDULE.window * SCHEDULE.bucket_ms * 1_000_000
    assert [a.at for a in first if isinstance(a, Timer)] == [1_000 + span, 1_000 + 2 * span]
    assert _encoded(first) == [(NC.VERIFIER, NC.DEPLOYMENT), (NC.VERIFIER, NC.LOAD)]
    timer = Received("clock", Message("ClockTimer", {"at": 0, "ns": 0}))
    one, two = p.on(timer), p.on(timer)
    assert not any(isinstance(a, Finish) for a in one)
    assert {m.message.fields.get("window") for m in one if isinstance(m, Send)} == {0}
    assert (NC.VERIFIER, NC.GROUPS) in _encoded(two) and two[-1] == Finish({"windows": 2})
    assert p.on(timer) == [Finish(refused="out-of-phase")]
