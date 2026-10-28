"""The network certifier's transfer rows (`transfer_rows`): a value committed as a registered value under the certifier's
salts, its head signed for one link-window, and the verifier's read of its registration and of a unit's rows."""

from __future__ import annotations

import hashlib
import random
from dataclasses import replace

import pytest

from verity.ml.crypto import lms
from verity.core.protocols.certifier import transfer_rows as TR
from verity.core.protocols.certifier import warden_rows as WR
from verity.core.protocols.certifier.certificate import Direction, Issuer, LmsKey
from verity.core.protocols.sampled_proofs.one_stage import registered as G

import signers

PROGRAM = hashlib.sha512(b"tiny class program").hexdigest()
LINK = "s0/uplink"


def rows(n: int, words: int, seed: int = 1) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(1 << 16) for _ in range(words)] for _ in range(n)]


def attested(tmp_path, data=None):
    """A response value of 4 one-word rows in window 3 on `LINK`, its certificate and the issuer that signed it."""
    issuer = Issuer(signers.signer(tmp_path), signers.SESSION)
    t, cert = TR.attest(issuer, LINK, Direction.EGRESS, 3, "response", PROGRAM, "response", data or rows(4, 1))
    assert cert is not None
    return t, cert, issuer


def public(seed: bytes = b"certifier") -> LmsKey:
    k = signers.key(seed)
    return LmsKey(k.params, lms.public_key(k))


def expect(t: TR.Transfer, **over):
    h = t.head
    return {"link": h.link, "direction": h.direction, "window": h.window, "kind": h.kind, "value": h.value,
            "program": PROGRAM, "words": h.words, "rows": h.rows, **over}


def test_commitment_is_a_registered_values_and_the_head_its_root(tmp_path):
    data = rows(4, 1)
    t, cert, _issuer = attested(tmp_path, data=data)
    same = G.Registered.commit(PROGRAM, "response", data, t.salts, 16, G.SCHEMA_V2)
    assert t.root == same.root and t.head.domain == G.domain(PROGRAM, "response", 1, 4).domain_id
    assert (t.head.words, t.head.rows) == (1, 4) and cert.digest == t.head.digest()
    assert TR.read_witness(t.witness) == list(t.salts) and len(set(t.salts)) == 4
    doc = TR.registration(t.head, cert)
    head, dom = TR.check_registration(doc, public(), signers.SESSION, **expect(t))
    assert head == t.head and dom.domain_id == t.head.domain


def test_a_units_row_opens_the_signed_root_and_a_forged_byte_does_not(tmp_path):
    data = rows(4, 1)
    t, _cert, _issuer = attested(tmp_path, data=data)
    reads = [2, 0, 3, 1]
    port = TR.port(t.head, reads, reads_from=5)
    assert port["value"] == "response" and port["reads"] == reads and port[G.READS_FROM] == 5
    assert port["root"] == t.root.hex() and port["row_schema"] == G.SCHEMA_V2 and port["words"] == 1
    got, salts, entry = t.statement(reads)
    dom = TR.domain(PROGRAM, "response", 1, 4)
    bcs = [TR.bc(r, s) for r, s in zip(got, salts)]
    assert WR.refusal(port, dom, entry, bcs) is None
    forged = [list(r) for r in got]
    forged[0][0] ^= 1
    why = WR.refusal(port, dom, entry, [TR.bc(r, s) for r, s in zip(forged, salts)])
    assert why is not None and "does not open" in why


@pytest.mark.parametrize("word_bits,schema,words", [(32, G.SCHEMA_V2, 1), (16, G.SCHEMA_SEG, 48)])
def test_a_value_at_another_width_or_row_format_is_registered_so_and_its_head_binds_it(tmp_path, word_bits, schema, words):
    """A value of one 32-bit leaf a row and a token's embedding (a row-seg row longer than one SHA-512 block):
    `Registered.commit`'s commitment at that width and format, its head binding both, read by the verifier at its own
    width and format alone, and each row opening the signed root under that format."""
    rng = random.Random(3)
    data = [[rng.randrange(1 << word_bits) for _ in range(words)] for _ in range(4)]
    issuer = Issuer(signers.signer(tmp_path), signers.SESSION)
    t, cert = TR.attest(issuer, LINK, Direction.INGRESS, 0, "load", PROGRAM, "w", data, word_bits=word_bits, schema=schema)
    assert t.root == G.Registered.commit(PROGRAM, "w", data, t.salts, word_bits, schema).root
    assert t.head.domain == TR.domain(PROGRAM, "w", words, 4, word_bits).domain_id == \
        G.domain(PROGRAM, "w", words, 4, word_bits).domain_id
    flip = {"word_bits": {32: 16, 16: 32}[word_bits], "schema": G.SCHEMA_SEG if schema == G.SCHEMA_V2 else G.SCHEMA_V2}
    assert all(replace(t.head, **{k: v}).digest() != t.head.digest() for k, v in flip.items())
    doc = TR.registration(t.head, cert)
    assert (doc["words"], doc["word_bits"], doc["row_schema"]) == (words, word_bits, schema)
    head, dom = TR.check_registration(doc, public(), signers.SESSION, **expect(t, word_bits=word_bits, schema=schema))
    assert head == t.head
    with pytest.raises(ValueError, match="registration's word_bits is 32" if word_bits == 32 else "registration's row_schema"):
        TR.check_registration(doc, public(), signers.SESSION, **expect(t))
    port = TR.port(t.head, [3, 0])
    assert (port["word_bits"], port.get("row_schema", G.SCHEMA)) == (word_bits, schema)
    got, salts, entry = t.statement([3, 0])
    bcs = [TR.bc(r, s, word_bits, schema) for r, s in zip(got, salts)]
    assert all(G.verify_read(dom, t.root, p, b, bytes.fromhex(path), schema)
               for p, b, path in zip(entry["positions"], bcs, entry["paths"]))
    with pytest.raises(ValueError, match="are 16 or 32 bits, not 8"):
        TR.Transfer.commit(LINK, Direction.INGRESS, 0, "load", PROGRAM, "w", data, word_bits=8)
    with pytest.raises(ValueError, match="rows, not 'hm96-sha512/row/v1'"):
        TR.Transfer.commit(LINK, Direction.INGRESS, 0, "load", PROGRAM, "w", data, schema=G.SCHEMA)


@pytest.mark.parametrize("field,value", [("link", "s1/uplink"), ("direction", "ingress"), ("window", 4), ("kind", "wire"),
                                         ("value", "request"), ("program", "00" * 64), ("words", 2), ("rows", 2),
                                         ("word_bits", 32), ("row_schema", G.SCHEMA), ("domain", "00" * 64)])
def test_check_registration_refuses_a_field_not_the_verifiers(tmp_path, field, value):
    t, cert, _issuer = attested(tmp_path)
    doc = TR.registration(t.head, cert) | {field: value}
    with pytest.raises(ValueError, match=f"registration's {field} is"):
        TR.check_registration(doc, public(), signers.SESSION, **expect(t))


@pytest.mark.parametrize("other", ["link", "window", "direction"])
def test_check_registration_refuses_a_certificate_for_another_link_window_or_direction(tmp_path, other):
    t, _cert, issuer = attested(tmp_path)
    at = {"link": LINK, "direction": Direction.EGRESS, "window": 3}
    at[other] = {"link": "s1/uplink", "window": 2, "direction": Direction.INGRESS}[other]
    cert = issuer.sign(at["link"], at["direction"], at["window"], t.head.digest())
    with pytest.raises(ValueError, match="doesn't verify under the network certifier's key"):
        TR.check_registration(TR.registration(t.head, cert), public(), signers.SESSION, **expect(t))


def test_check_registration_refuses_another_key_session_root_or_signed_digest(tmp_path):
    t, cert, _issuer = attested(tmp_path)
    doc = TR.registration(t.head, cert)
    with pytest.raises(ValueError, match="doesn't verify"):
        TR.check_registration(doc, public(b"another certifier"), signers.SESSION, **expect(t))
    with pytest.raises(ValueError, match="doesn't verify"):
        TR.check_registration(doc, public(), signers.OTHER_SESSION, **expect(t))
    own = TR.Transfer.commit(LINK, Direction.EGRESS, 3, "response", PROGRAM, "response", rows(4, 1, seed=2))
    with pytest.raises(ValueError, match="doesn't sign its head"):
        TR.check_registration(doc | {"root": own.root.hex()}, public(), signers.SESSION, **expect(t))
    with pytest.raises(ValueError, match="holds the keys"):
        TR.check_registration(doc | {"salts": []}, public(), signers.SESSION, **expect(t))
    with pytest.raises(ValueError, match="doesn't read"):
        TR.check_registration(doc | {"root": "zz"}, public(), signers.SESSION, **expect(t))


def test_a_kind_is_in_its_direction():
    with pytest.raises(ValueError, match="a transfer is a"):
        TR.Transfer.commit(LINK, Direction.EGRESS, 0, "load", PROGRAM, "w", rows(1, 2))
    with pytest.raises(ValueError, match="a transfer is a"):
        TR.Transfer.commit(LINK, Direction.INGRESS, 0, "response", PROGRAM, "w", rows(1, 2))
    for d in (Direction.EGRESS, Direction.INGRESS):
        TR.Transfer.commit(LINK, d, 0, "wire", PROGRAM, "h", rows(1, 2))


def test_request_and_response_rows():
    assert TR.request_rows([b"hi", b"ab"], 2, 2) == [[104, 0], [105, 0], [97, 0], [98, 0]]
    assert TR.response_rows([b"\xff"], 1, 1) == [[255, 0]]
    req, resp = [bytes(range(8)), b"\x80" * 8], [bytes(range(24)), bytes(range(255, 231, -1))]
    rows = TR.request_rows(req, 2)
    assert len(rows) == 2 * TR.PROMPT and rows[TR.PROMPT + 3] == [0x80, 0] and rows[5] == [5, 0]
    rows = TR.response_rows(resp, 2)
    assert len(rows) == 2 * TR.GENERATED and rows[TR.GENERATED] == [255, 0] and rows[23] == [23, 0]
    with pytest.raises(ValueError, match="is 5 bytes, not the class's 4: its rows carry no length"):
        TR.request_rows([b"hello"], 1, 4)
    with pytest.raises(ValueError, match="is 3 bytes, not the class's 4"):
        TR.request_rows([b"abc"], 1, 4)
    with pytest.raises(ValueError, match="2 requests in a window of 1 slots"):
        TR.request_rows([b"a", b"b"], 1, 1)
    with pytest.raises(ValueError, match="1 responses in a window of 2 slots: .* no absent marker"):
        TR.response_rows([b"a"], 2, 1)
    with pytest.raises(ValueError, match="is 23 bytes, not the class's 24"):
        TR.response_rows([bytes(23)], 1)
