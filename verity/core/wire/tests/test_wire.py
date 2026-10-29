"""The wire: frames round-trip, heads are the commitments' canonical JSON, every rule broken is refused by its name, and the
handshake binds a channel to its parties' keys, its protocol and its configuration."""

from __future__ import annotations

import hashlib

import pytest

from verity.core.wire import (HEADER, REFUSALS, Body, FrameReader, Hex, Int, List, Message, Phase, Refused, Registry, Schema,
                              Str, canonical, frame, frames, parse, split)
from verity.core.wire import handshake as H
from verity.ml.commitments.identity import canonical_json_bytes
from verity.ml.crypto import lms

PING = Schema("Ping", 1, "structure", {"n": Int(0, 9), "who": Str(8), "ids": List(Hex(4), 3)}, Body(max=16, unit=2))
PING2 = Schema("Ping", 2, "output", {"n": Int()})
R = Registry([PING, PING2])
OK = Message("Ping", {"n": 3, "who": "é", "ids": ["00ff00ff"]}, b"\x00\x01")


def refused(f, *a) -> str:
    with pytest.raises(Refused) as e:
        f(*a)
    assert e.value.name in REFUSALS
    return e.value.name


def test_a_message_round_trips_and_its_head_is_the_commitments_canonical_json():
    data = R.encode(OK)
    head, body = split(data)
    assert head == canonical_json_bytes(OK.head()) == canonical(OK.head())
    assert body == b"\x00\x01" and data == frame(head, body)
    assert R.decode(data) == OK
    assert R.kind("Ping", 2) == "output"


def test_each_rule_broken_is_refused_by_its_name():
    good = R.encode(OK)
    head, body = split(good)
    assert refused(R.decode, good[:5]) == "malformed"
    assert refused(R.decode, good + b"x") == "malformed"
    assert refused(R.decode, frame(b'{"type": "Ping","v":1}')) == "malformed"
    assert refused(R.decode, frame(b'{"n":1,"n":1,"type":"Ping","v":1}')) == "malformed"
    assert refused(R.decode, frame(b'{"n":1.0,"type":"Ping","v":2}')) == "malformed"
    assert refused(R.decode, frame(b'[1]')) == "malformed"
    assert refused(R.decode, HEADER.pack(1 << 20, 0) + b"{}") == "oversized"
    assert refused(R.decode, frame(head, bytes(18))) == "oversized"
    assert refused(R.decode, frame(canonical({"type": "Pong", "v": 1}))) == "unknown-type"
    assert refused(R.decode, frame(canonical({"type": "Ping", "v": 3}))) == "unknown-version"
    assert refused(R.decode, frame(canonical({"type": "Ping", "v": "1"}))) == "unknown-version"
    assert refused(R.decode, frame(canonical({"type": "Ping", "v": 2}))) == "schema"
    assert refused(R.decode, frame(canonical({"n": 10, "who": "", "ids": [], "type": "Ping", "v": 1}))) == "schema"
    assert refused(R.decode, frame(canonical({"n": 1, "who": "", "ids": ["0F0F0F0F"], "type": "Ping", "v": 1}))) == "schema"
    assert refused(R.decode, frame(canonical({"n": 1, "who": "", "ids": [], "x": 0, "type": "Ping", "v": 1}))) == "schema"
    assert refused(R.decode, frame(head, b"\x00")) == "schema"
    assert refused(R.encode, Message("Ping", {"n": True, "who": "", "ids": []})) == "schema"
    assert refused(R.encode, Message("Ping", {"n": 1}, version=7)) == "unknown-version"


def test_canonical_json_holds_no_float():
    with pytest.raises(TypeError):
        canonical({"x": 0.5})
    assert parse(b'{"a":[1,"\xc3\xa9"]}') == {"a": [1, "é"]}


def test_a_stream_is_cut_into_frames_and_an_oversized_header_refused_before_its_body():
    a, b = R.encode(OK), R.encode(Message("Ping", {"n": 1}, version=2))
    reader = FrameReader()
    for i in range(len(a + b)):
        reader.feed((a + b)[i:i + 1])
    assert frames(a + b) == [a, b]
    r = FrameReader(max_head=64, max_body=8)
    r.feed(HEADER.pack(65, 0))
    assert refused(r.next) == "oversized"
    assert refused(frames, a + b[:-1]) == "malformed"


def test_a_phase_refuses_what_it_doesnt_expect():
    p = Phase("Hello")
    assert p.check(Message("Hello")).type == "Hello"
    p.expect("Signature")
    assert refused(p.check, Message("Hello")) == "out-of-phase"


def test_a_registry_restricted_to_some_types():
    only = (R | Registry([Schema("Pong", 1, "private")])).only(["Pong"])
    assert only.types == {"Pong"}
    assert refused(only.decode, R.encode(Message("Ping", {"n": 1}, version=2))) == "unknown-type"
    # The header's bounds come first: a body where the registry allows none is refused before its type is read.
    assert refused(only.decode, R.encode(OK)) == "oversized"


# --- the handshake ------------------------------------------------------------------------------------------------------

PARAMS = lms.Params(64, 1, 3)


class Key:
    def __init__(self, name: str) -> None:
        self.private = lms.keygen(PARAMS, hashlib.sha512(name.encode()).digest())
        self.public_key, self.q = lms.public_key(self.private), 0

    def sign(self, message: bytes) -> bytes:
        self.q += 1
        return lms.sign_at(self.private, self.q - 1, message).encode()


def verify(public: bytes, message: bytes, signature: bytes) -> bool:
    return lms.verify(PARAMS, public, message, lms.Signature.decode(PARAMS, signature))


CONF = hashlib.sha512(b"configuration").digest()


def side(party, peer, nonce, *, key=None, protocol="p", version=1, configuration=CONF, peer_key=None):
    return H.Handshake(party=party, peer=peer, protocol=protocol, version=version, configuration=configuration,
                       nonce=bytes([nonce]) * 32, key=key or Key(party), verify=verify, peer_key=peer_key)


def exchange(a: H.Handshake, b: H.Handshake) -> None:
    to_b, to_a = a.start(), b.start()
    while to_a or to_b:
        out_a = [f for x in to_a for f in a.on(x)]
        out_b = [f for x in to_b for f in b.on(x)]
        to_a, to_b = out_b, out_a


def test_the_handshake_completes_and_both_sides_hold_one_exchange():
    a, b = side("alice", "bob", 1), side("bob", "alice", 2, peer_key=Key("alice").public_key)
    exchange(a, b)
    assert a.done and b.done
    assert a.frames == [b.frames[1], b.frames[0], b.frames[3], b.frames[2]]
    assert a.peer_hello["nonce"] == "02" * 32 and b.peer_hello["party"] == "alice"


@pytest.mark.parametrize(("change", "name"), [
    ({"protocol": "q"}, "handshake-protocol"),
    ({"version": 2}, "handshake-protocol"),
    ({"configuration": hashlib.sha512(b"other").digest()}, "handshake-configuration"),
    ({"key": Key("mallory")}, "handshake-key"),
])
def test_a_hello_that_doesnt_match_is_refused_by_name(change, name):
    a = side("alice", "bob", 1, peer_key=Key("bob").public_key)
    b = side("bob", "alice", 2, **change)
    with pytest.raises(Refused) as e:
        exchange(a, b)
    assert e.value.name == name


def test_a_hello_from_another_party_or_with_the_same_nonce_is_refused():
    with pytest.raises(Refused) as e:
        exchange(side("alice", "bob", 1), side("carol", "alice", 2))
    assert e.value.name == "handshake-party"
    with pytest.raises(Refused) as e:
        exchange(side("alice", "bob", 1), side("bob", "alice", 1))
    assert e.value.name == "handshake-nonce"


def test_a_signature_that_doesnt_verify_is_refused():
    a, b = side("alice", "bob", 1), side("bob", "alice", 2)
    hb = b.start()
    a.start()
    sig = a.on(hb[0])[0]
    b.on(a.frames[0])
    forged = H.REGISTRY.encode(Message("Signature", body=bytes(len(split(sig)[1]))))
    with pytest.raises(Refused) as e:
        b.on(forged)
    assert e.value.name == "handshake-signature"
    with pytest.raises(Refused) as e:
        side("alice", "bob", 1).on(hb[0])
    assert e.value.name == "out-of-phase"


def test_a_signature_that_doesnt_parse_is_refused_with_what_was_caught():
    a, b = side("alice", "bob", 1), side("bob", "alice", 2)
    hb = b.start()
    a.start()
    a.on(hb[0])
    b.on(a.frames[0])
    with pytest.raises(Refused) as e:
        b.on(H.REGISTRY.encode(Message("Signature", body=b"\x00")))
    assert e.value.name == "handshake-signature"
    assert "doesn't parse" in e.value.detail and type(e.value.__cause__).__name__ in e.value.detail


def test_a_side_signs_its_own_name_so_the_other_sides_signature_doesnt_pass_for_it():
    a, b = side("alice", "bob", 1), side("bob", "alice", 2)
    ha, hb = a.start()[0], b.start()[0]
    sig_a = a.on(hb)[0]
    b.on(ha)
    with pytest.raises(Refused) as e:
        a.on(sig_a)
    assert e.value.name == "handshake-signature"
