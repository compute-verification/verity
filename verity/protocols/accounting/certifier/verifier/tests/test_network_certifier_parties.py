"""The network certifier's manifest (`parties`): its three parties and two channels as its docstring states them, the
workers and the verifier as SPOT declares them, a placement that holds the certifier to the auditor's machine
and takes one a link, and a committed output value's two parts as the two messages: the witness to the workers alone,
the record to the verifier alone, each refused by name past its schema; and its id and its check are the Lean's."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from verity.commitments import domains
from verity.commitments.crypto import lms
from verity import channel as wire
from verity.protocol import Role, place, site_of
from verity.channel import Message, Refused
from verity.protocols.accounting.certifier.verifier import out_commit as O
from verity.protocols.accounting.certifier.verifier import parties as NC
from verity.protocols.accounting.certifier.verifier import warden_rows as WR
from verity.protocols.accounting.certifier.verifier.tests import signers as S
from verity.protocols.proofs.spot.verifier import parties as SP

PROGRAM = "5a" * 64
ROWS, WORDS, WINDOW, LINK = 4, 8, 3, "node0/uplink0"
SHAPE = O.OutShape(PROGRAM, O.VALUE, WORDS, ROWS)
VALUES = [[(i * 31 + j * 7) % 65536 for j in range(WORDS)] for i in range(ROWS)]
HEAD = {"link": LINK, "window": WINDOW}
MACHINES = {"gate": ("auditor", ["clock", "coins", "signing-key"]),
            "auditor": ("auditor", ["audit", "component", "product"]), "node": ("developer", ["serve"])}
LINKS = (("gate", "node"), ("gate", "auditor"), ("auditor", "node"))
SITE = site_of([{"name": n, "owner": o, "provides": p} for n, (o, p) in MACHINES.items()],
               [{"ends": [a, b], "latency_ms": 1} for a, b in LINKS])
PLACED = {NC.NETWORK_CERTIFIER: "gate", NC.WORKERS: "node", NC.VERIFIER: "auditor"}
LEAN = Path(__file__).resolve().parents[1] / "lean"


@pytest.fixture(scope="module")
def out(tmp_path_factory) -> tuple[O.CertifiedOut, O.PublicKey]:
    s = S.signer(tmp_path_factory.mktemp("parties-signer"))
    return O.commit_out(VALUES, shape=SHAPE, key=s, session=S.SESSION, window=WINDOW, link=LINK), s.public_key()


def test_the_manifest_states_the_certifier_and_spots_two() -> None:
    m = NC.MANIFEST
    assert (m.protocol, m.version) == (domains.CERTIFIER_PROTOCOL, 1) and m.protocol == SP.IDS["certifier"]
    assert [r.name for r in m.parties] == ["network-certifier", "workers", "verifier"]
    assert m.role(NC.NETWORK_CERTIFIER) == Role("network-certifier", "auditor", per="link",
                                                needs=("clock", "coins", "signing-key"))
    assert [m.role(p) for p in (NC.WORKERS, NC.VERIFIER)] == [SP.MANIFEST.role(p) for p in (SP.WORKERS, SP.VERIFIER)]
    assert [(m.role(p).principal, m.role(p).per) for p in (NC.WORKERS, NC.VERIFIER)] == [("developer", "one"),
                                                                                        ("auditor", "one")]


def test_the_certifier_sends_the_opening_to_the_workers_and_the_record_to_the_verifier() -> None:
    m = NC.MANIFEST
    assert [(c.sender, c.receiver, c.messages, c.after, c.carries_coins, c.latency_ms) for c in m.channels] == [
        ("network-certifier", "workers", ("out-opening", "records", "cells-salts"), (), False, None),
        ("network-certifier", "verifier", ("out-record", "certificate", "registration", "load", "deployment"), (), False,
         None)]
    assert m.peers(NC.NETWORK_CERTIFIER) == ("verifier", "workers")
    assert {t: m.registry.kind(t) for t in m.registry.types} == {
        "out-opening": "private", "records": "private", "cells-salts": "private", "out-record": "commitment",
        "certificate": "commitment", "registration": "commitment", "load": "commitment", "deployment": "structure"}
    assert not m.sends(NC.WORKERS, NC.VERIFIER).types and not m.sends(NC.VERIFIER, NC.NETWORK_CERTIFIER).types


def test_it_places_the_certifier_on_the_auditors_machine_one_a_link() -> None:
    assert place(NC.MANIFEST, SITE, PLACED).mode == "placed"
    two = {k: v for k, v in PLACED.items() if k != NC.NETWORK_CERTIFIER}
    two.update({f"{NC.NETWORK_CERTIFIER}@uplink0": "gate", f"{NC.NETWORK_CERTIFIER}@uplink1": "gate"})
    assert place(NC.MANIFEST, SITE, two).mode == "placed"
    with pytest.raises(Refused) as e:
        place(NC.MANIFEST, SITE, {**PLACED, NC.NETWORK_CERTIFIER: "node"})
    assert e.value.name == "principal"
    with pytest.raises(Refused) as e:
        place(NC.MANIFEST, SITE, {**PLACED, NC.NETWORK_CERTIFIER: "auditor"})
    assert e.value.name == "needs" and "['clock', 'coins', 'signing-key']" in e.value.detail
    with pytest.raises(Refused) as e:
        place(NC.MANIFEST, SITE, {k: v for k, v in PLACED.items() if k != NC.WORKERS})
    assert e.value.name == "unbound"


def test_an_output_value_goes_as_its_two_messages(out) -> None:
    value, key = out
    to_workers = NC.MANIFEST.sends(NC.NETWORK_CERTIFIER, NC.WORKERS)
    to_verifier = NC.MANIFEST.sends(NC.NETWORK_CERTIFIER, NC.VERIFIER)
    opening = to_workers.decode(to_workers.encode(Message(NC.OUT_OPENING, HEAD, value.witness)))
    assert opening.fields == HEAD and O.read_witness(opening.body) == list(value.salts)
    record = to_verifier.decode(to_verifier.encode(Message(NC.OUT_RECORD, HEAD, wire.canonical(value.public))))
    assert record.fields == HEAD
    assert O.check_out(wire.parse(record.body), key, S.SESSION, SHAPE, window=record.fields["window"],
                       link=record.fields["link"]) == value.root
    for channel, message in ((to_verifier, Message(NC.OUT_OPENING, HEAD, value.witness)),
                             (to_workers, Message(NC.OUT_RECORD, HEAD, wire.canonical(value.public)))):
        with pytest.raises(Refused) as e:
            channel.encode(message)
        assert e.value.name == "unknown-type"


def _sent(channel, message: Message) -> Message:
    return channel.decode(channel.encode(message))


#: what the device's run hands out beside the output value, each with a stand-in body, by the channel that carries it
RUN = {NC.WORKERS: [Message(NC.RECORDS, {**HEAD, "record": r}, bytes(range(256)) * 3) for r in NC.RECORD_NAMES]
       + [Message(NC.CELLS_SALTS, HEAD, bytes(3 * WR.SALT_BYTES + WR.WIRING_SALT_BYTES))],
       NC.VERIFIER: [Message(NC.CERTIFICATE, HEAD, b'{"refused": ["node0/uplink0", "egress", 3]}'),
                     Message(NC.REGISTRATION, HEAD, b'{"cost": {"row_bytes": 512, "rows": 4}, "registration": {}}'),
                     Message(NC.LOAD, {}, b'{"values": []}'),
                     Message(NC.DEPLOYMENT, {}, b'{"digest": "00", "params": {}, "windows": 2}')]}


def test_the_runs_witness_goes_to_the_workers_and_its_certificates_and_registrations_to_the_verifier() -> None:
    """The records and the rows' salts reach the workers alone, the certificates, the rows' registrations, the
    load and the deployment the verifier alone, each as it was sent; the salts read back as the rows' and the wiring's."""
    for receiver, other in ((NC.WORKERS, NC.VERIFIER), (NC.VERIFIER, NC.WORKERS)):
        to, wrong = (NC.MANIFEST.sends(NC.NETWORK_CERTIFIER, p) for p in (receiver, other))
        for message in RUN[receiver]:
            got = _sent(to, message)
            assert (got.type, got.fields, got.body) == (message.type, message.fields, message.body)
            with pytest.raises(Refused) as e:
                wrong.encode(message)
            assert e.value.name == "unknown-type", message.type
    salts, wiring = WR.read_witness(RUN[NC.WORKERS][-1].body)
    assert (len(salts), len(wiring)) == (3, WR.WIRING_SALT_BYTES)


def test_a_message_past_its_schema_is_refused_by_name(out) -> None:
    value, _ = out
    types = NC.MANIFEST.registry
    for message, name in ((Message(NC.OUT_OPENING, HEAD, value.witness[:-1]), "schema"),
                          (Message(NC.OUT_OPENING, {"link": LINK}, value.witness), "schema"),
                          (Message(NC.OUT_OPENING, {**HEAD, "window": 1 << 64}, value.witness), "schema"),
                          (Message(NC.OUT_RECORD, {**HEAD, "window": -1}, b"{}"), "schema"),
                          (Message(NC.OUT_RECORD, HEAD, bytes(NC.MAX_RECORD + 1)), "oversized"),
                          (Message(NC.RECORDS, {**HEAD, "record": "cells"}, b""), "schema"),
                          (Message(NC.RECORDS, HEAD, b""), "schema"),
                          (Message(NC.CELLS_SALTS, HEAD, bytes(WR.SALT_BYTES + 1)), "schema"),
                          (Message(NC.CERTIFICATE, {"link": LINK}, b"{}"), "schema"),
                          (Message(NC.REGISTRATION, HEAD, bytes(NC.MAX_RECORD + 1)), "oversized"),
                          (Message(NC.LOAD, HEAD, b"{}"), "schema"),
                          (Message(NC.DEPLOYMENT, HEAD, b"{}"), "schema"),
                          (Message(NC.DEPLOYMENT, {}, bytes(NC.MAX_RECORD + 1)), "oversized")):
        with pytest.raises(Refused) as e:
            types.encode(message)
        assert e.value.name == name, message.type
    types.encode(Message(NC.OUT_OPENING, {**HEAD, "window": (1 << 64) - 1}))


def test_the_bounds_are_the_frames_and_the_deployed_signatures() -> None:
    frame = wire.HEADER.unpack(b"\xff" * wire.HEADER.size)[1]
    assert NC.MAX_OPENING % O.SALT_BYTES == 0 and NC.MAX_OPENING <= frame < NC.MAX_OPENING + O.SALT_BYTES
    assert wire.MAX_HEAD < 2 * lms.DEPLOYED.signature_bytes() < NC.MAX_RECORD
    assert NC.MAX_PACKED == frame and 1024 * NC.MAX_RECORD <= NC.MAX_LOAD <= frame
    assert WR.SALT_BYTES % WR.WIRING_SALT_BYTES == 0 and NC.MAX_SALTS % WR.SALT_BYTES == 0
    assert NC.MAX_SALTS <= frame < NC.MAX_SALTS + WR.WIRING_SALT_BYTES


def test_its_id_and_its_check_are_the_leans() -> None:
    exe, command, argument = NC.MANIFEST.params.split()
    assert (command, argument) == ("check", "PARAMS")
    lake = (LEAN / "lakefile.toml").read_text(encoding="utf-8")
    root = re.search(rf'^name = "{re.escape(exe)}"\nroot = "([\w.]+)Main"$', lake, re.MULTILINE)
    assert root, f"{exe} is no executable of {LEAN.name}/lakefile.toml"
    text = (LEAN / (root[1].replace(".", "/") + ".lean")).read_text(encoding="utf-8")
    assert f'def protocolId : String := "{NC.ID}"' in text
