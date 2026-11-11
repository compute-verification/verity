"""The network certifier's connection rows (`connection_rows`): a link-window's lengths and fails tables are the
schedule's size whatever the load and read back as the releases the certifier handed, each row opens its table's root
at its position and nowhere else, one head lists both tables and is signed once, the registration reads back under the
network certifier's key and session and the schedule and refuses any field, table, key, session or certificate but its
own, and a schedule without lengths rows has none of it."""

from __future__ import annotations

import json
import struct

import pytest

import signers
from verity.commitments import hm96
from verity.commitments.crypto import lms
from verity.protocols.certifier.verifier import connection_rows as C
from verity.protocols.certifier.verifier import warden_rows as W
from verity.protocols.certifier.verifier.certificate import Direction, Issuer, LmsKey
from verity.protocols.certifier.verifier.tests.test_network_certifier_lengths import LEN, PLAIN, traffic
from verity.protocols.certifier.verifier.tests.test_network_certifier_warden import records, run
from verity.protocols.certifier.verifier.warden import Released
from verity.protocols.spot.verifier.one_stage import registered as G

T, K = LEN.window // LEN.every, LEN.stride_releases
LINK = "responses"


@pytest.fixture(scope="module")
def handed():
    """The releases of the lengths tests' traffic, by window: two requests in one slot stride and one later, then two
    the server failed."""
    reqs, arrivals, answers = traffic()
    warden, _, _ = run(answers, arrivals, params=LEN, windows=3)
    return reqs, warden, dict(warden.lengths)


def _attest(tmp_path, strides, window=0, session=signers.SESSION, seed=b"certifier"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    issuer = Issuer(signers.signer(tmp_path, seed), session)
    rows, cert = C.attest(issuer, LINK, window, strides, LEN, None)
    return issuer, rows, cert


def test_the_two_tables_are_the_schedules_size_whatever_the_load_and_read_back(handed):
    """T / every rows of K slots in every window, loaded or idle: a lengths slot is N + 1, the request's bytes and
    its outputs, a fails slot its class and where it failed, both zero where no request is; `releases` reads them back
    as the releases handed."""
    reqs, _, by_window = handed
    assert C.tables_of(LEN) == (("lengths", T, K * 12), ("fails", T, K * 4)) and C.strides_of(LEN) == T
    for strides in (*by_window.values(), ((),) * T):
        made = C.tables(LEN, strides)
        assert list(made) == ["lengths", "fails"]
        for name, _, size in C.tables_of(LEN):
            assert len(made[name]) == T and {len(r) for r in made[name]} == {size}
        assert C.releases(LEN, made["lengths"], made["fails"]) == [list(s) for s in strides]
    idle = C.tables(LEN, ((),) * T)
    assert idle == {"lengths": [bytes(K * 12)] * T, "fails": [bytes(K * 4)] * T}
    first, second = C.tables(LEN, by_window[0]), C.tables(LEN, by_window[1])
    assert struct.unpack("<6I", first["lengths"][0]) == (1, len(reqs[0]), 3, 2, len(reqs[1]), 8)
    assert struct.unpack("<6I", first["lengths"][5]) == (3, len(reqs[2]), 2, 0, 0, 0)
    assert first["fails"] == [bytes(K * 4)] * T
    assert struct.unpack("<6I", second["lengths"][1]) == (4, len(reqs[3]), 0, 5, len(reqs[4]), 2)
    assert struct.unpack("<4H", second["fails"][1]) == (1, 0, 1, 2)
    assert [r for j, r in enumerate(second["fails"]) if j != 1] == [bytes(K * 4)] * (T - 1)


def test_releases_that_are_not_a_window_of_the_schedule_make_no_tables(handed):
    _, _, by_window = handed
    strides = by_window[0]
    one = Released(0, 400, 3, False)
    with pytest.raises(ValueError, match="no connection rows"):
        C.tables(PLAIN, strides)
    with pytest.raises(ValueError, match="no connection rows"):
        C.tables_of(PLAIN)
    for bad in (strides[:-1], ((one,) * (K + 1),) + strides[1:]):
        with pytest.raises(ValueError, match="slot strides of at most"):
            C.tables(LEN, bad)
    for change in ({"order": -1}, {"order": (1 << 32) - 1}, {"request_bytes": 0}, {"request_bytes": 1 << 32},
                   {"outputs": LEN.egress_cells}, {"outputs": -1}, {"failed": 1}):
        with pytest.raises(ValueError, match="no release of the schedule"):
            C.tables(LEN, ((one._replace(**change),),) + ((),) * (T - 1))


def test_tables_that_are_not_laid_out_as_releases_are_refused(handed):
    """A slot after an empty one, anything in an empty slot, a fails slot that isn't its lengths slot's, and a table
    of another size."""
    _, _, by_window = handed
    made = C.tables(LEN, by_window[1])
    lengths, fails = made["lengths"], made["fails"]

    def with_row(rows, j, row):
        return [*rows[:j], row, *rows[j + 1:]]

    def refused(why, lengths_=lengths, fails_=fails):
        with pytest.raises(ValueError, match=why):
            C.releases(LEN, lengths_, fails_)

    refused("follows an empty slot", with_row(lengths, 1, bytes(12) + lengths[1][12:]),
            with_row(fails, 1, bytes(4) + fails[1][4:]))
    refused("empty and not zero", with_row(lengths, 0, struct.pack("<6I", 0, 7, 0, 0, 0, 0)))
    refused("empty and not zero", fails_=with_row(fails, 0, struct.pack("<4H", 1, 0, 0, 0)))
    for cls, at in ((1, 1), (2, 2), (0, 2)):
        refused("fails slot", fails_=with_row(fails, 1, struct.pack("<4H", 1, 0, cls, at)))
    refused("rows of", lengths[:-1])
    refused("rows of", fails_=with_row(fails, 0, bytes(K * 4 + 2)))
    refused("rows of", fails, lengths)


def test_every_row_opens_its_tables_signed_root_at_its_position_and_nowhere_else(handed, tmp_path):
    """A row's b‖c is the statement's (the `hm96-sha512/row/v2` x of its u16 words under its salt), climbs its path
    to its table's root under the verifier's own domain, and fails at another position, with a changed bit, under
    another salt, under the other table's root or domain, and under another commitment's root."""
    _, _, by_window = handed
    _, rows, cert = _attest(tmp_path, by_window[1], window=1)
    head = rows.head
    assert cert.digest == head.digest() and [e.name for e in head.tables] == ["lengths", "fails"]
    doms = {name: C.domain(LEN, signers.SESSION, LINK, 1, name, count, size) for name, count, size in C.tables_of(LEN)}
    for table in rows.tables:
        entry, dom = head.table(table.name), doms[table.name]
        assert dom.domain_id == entry.domain == table.domain.domain_id and table.value == f"{LINK}/{table.name}/1"
        for k, row in enumerate(table.rows):
            bc = W.bc(row, table.salts[k])
            assert bc == hm96.SHA512.commit_string(
                hm96.SHA512.default_key, G.row_digest(W.words(row), 16, G.SCHEMA_V2), table.salts[k])
            assert W.opens(dom, entry.root, k, bc, table.path(k))
        row, salt, path = table.rows[1], table.salts[1], table.path(1)
        flipped = bytes([row[0] ^ 1]) + row[1:]
        assert not W.opens(dom, entry.root, 1, W.bc(flipped, salt), path)
        assert not W.opens(dom, entry.root, 1, W.bc(row, hm96.SHA512.fresh_salt()), path)
        assert not W.opens(dom, entry.root, 0, W.bc(row, salt), path)
        other = next(e for e in head.tables if e.name != table.name)
        assert not W.opens(dom, other.root, 1, W.bc(row, salt), path)
        assert not W.opens(doms[other.name], entry.root, 1, W.bc(row, salt), path)
    again = C.Rows.commit(LINK, 1, C.tables(LEN, by_window[1]), LEN, signers.SESSION)
    assert all(a.root != b.root for a, b in zip(again.head.tables, head.tables))
    for window, session in ((0, signers.SESSION), (1, signers.OTHER_SESSION)):
        assert C.domain(LEN, session, LINK, window, "lengths", T, K * 12).domain_id != doms["lengths"].domain_id


def test_one_head_lists_the_tables_and_is_signed_once(handed, tmp_path):
    """The head is the link-window, the schedule's digest and each table's name, rows, row bytes, domain and root, in
    the schedule's order, and its one certificate takes one leaf of the signer. A head of other tables, another order,
    another root or another link-window has another digest, and a head lists no table twice."""
    _, _, by_window = handed
    issuer, rows, cert = _attest(tmp_path, by_window[0])
    head = rows.head
    lengths, fails = head.tables
    assert head.encode() == b"".join([
        b"\0\x09responses", (0).to_bytes(8, "big"), LEN.digest, b"\0\2",
        b"\0\7lengths", T.to_bytes(4, "big"), (K * 12).to_bytes(4, "big"), lengths.domain, lengths.root,
        b"\0\5fails", T.to_bytes(4, "big"), (K * 4).to_bytes(4, "big"), fails.domain, fails.root])
    assert signers.used(issuer.signer) == [0] and issuer.refused == []
    assert (cert.link, cert.direction, cert.window, cert.digest) == (LINK, Direction.EGRESS, 0, head.digest())
    assert cert.verify(issuer.signer.public_key(), signers.SESSION)
    others = [C.Head(LINK, 0, LEN.digest, (lengths,)), C.Head(LINK, 0, LEN.digest, (fails, lengths)),
              C.Head(LINK, 1, LEN.digest, head.tables), C.Head("requests", 0, LEN.digest, head.tables),
              C.Head(LINK, 0, PLAIN.digest, head.tables),
              C.Head(LINK, 0, LEN.digest, (lengths, C.Entry("fails", T, K * 4, fails.domain, lengths.root))),
              C.Head(LINK, 0, LEN.digest, (*head.tables, C.Entry("ends", T, K * 4, fails.domain, fails.root)))]
    assert len({head.digest(), *(h.digest() for h in others)}) == len(others) + 1
    for bad in ((), (lengths, lengths)):
        with pytest.raises(ValueError, match="each of its tables once"):
            C.Head(LINK, 0, LEN.digest, bad)
    # no other certificate of the link-window signs it: a wire record's and a cells' head's sign other digests
    plain, _, _ = run([], {}, params=LEN, windows=1)
    record = records(plain)[1][0]
    wire = issuer.certify(record, Direction.EGRESS, LEN.grid, LEN.digest)
    _, cells_cert = W.attest(issuer, record, LEN, Direction.EGRESS)
    assert len({cert.digest, wire.certificate.digest, cells_cert.digest}) == 3


def test_salts_are_fresh_never_shared_and_the_witness_is_theirs(handed):
    _, _, by_window = handed
    made = C.tables(LEN, by_window[0])
    a, b = (C.Rows.commit(LINK, 0, made, LEN, signers.SESSION) for _ in range(2))
    salts = [s for rows in (a, b) for t in rows.tables for s in t.salts]
    assert len(set(salts)) == 4 * T and {len(s) for s in salts} == {W.SALT_BYTES}
    assert a.witness == b"".join(a.table("lengths").salts) + b"".join(a.table("fails").salts)
    mine = {t.name: t.salts for t in a.tables}
    assert C.Rows.commit(LINK, 0, made, LEN, signers.SESSION, mine).head == a.head
    for shared in ({**mine, "fails": mine["lengths"]}, {**mine, "lengths": (mine["lengths"][0],) * T}):
        with pytest.raises(ValueError, match="never reused"):
            C.Rows.commit(LINK, 0, made, LEN, signers.SESSION, shared)
    with pytest.raises(ValueError, match="salt a row"):
        C.Rows.commit(LINK, 0, made, LEN, signers.SESSION, {**mine, "fails": mine["fails"][:-1]})
    for bad, why in (({"fails": made["fails"], "lengths": made["lengths"]}, "in that order"),
                     ({"lengths": made["lengths"]}, "in that order"),
                     ({**made, "fails": made["fails"][:-1]}, "the fails table is"),
                     ({**made, "lengths": made["fails"]}, "the lengths table is")):
        with pytest.raises(ValueError, match=why):
            C.Rows.commit(LINK, 0, bad, LEN, signers.SESSION)
    with pytest.raises(ValueError, match="no connection rows"):
        C.Rows.commit(LINK, 0, made, PLAIN, signers.SESSION)


def test_the_registration_reads_back_and_refuses_anything_but_its_own(handed, tmp_path):
    """The verifier's read takes the link-window it expects and the schedule's parameters; any other field, an extra
    key, a table more, fewer or out of order, a root the certificate doesn't sign, another key, session or link-window,
    a schedule without lengths rows, or the wire record's or the cells' certificate of the same link-window, is
    refused."""
    _, warden, by_window = handed
    issuer, rows, cert = _attest(tmp_path / "a", by_window[1], window=1)
    doc = json.loads(json.dumps(C.registration(rows.head, cert, LEN)))
    key = issuer.signer.public_key()
    args = (LEN, LINK, 1)
    head, doms = C.check_registration(doc, key, signers.SESSION, *args)
    assert head == rows.head
    assert {n: d.domain_id for n, d in doms.items()} == {t.name: t.domain.domain_id for t in rows.tables}
    assert (doc["word_bits"], doc["row_schema"]) == (16, G.SCHEMA_V2)
    lengths, fails = doc["tables"]
    assert (lengths["value"], lengths["rows"], lengths["words"]) == ("responses/lengths/1", T, K * 6)
    assert (fails["value"], fails["rows"], fails["words"]) == ("responses/fails/1", T, K * 2)
    assert lengths["layout"] == {"strides": T, "every": LEN.every, "first_bucket": LEN.window, "slots": K,
                                 "slot_bytes": 12}
    assert fails["layout"] == {**lengths["layout"], "slot_bytes": 4}

    def refused(changed, why=None):
        with pytest.raises(ValueError, match=why):
            C.check_registration(changed, key, signers.SESSION, *args)

    for k, v in [("format", "x"), ("link", "requests"), ("window", 0), ("parameters", PLAIN.digest.hex()),
                 ("word_bits", 32), ("row_schema", G.SCHEMA), ("tables", [lengths]), ("tables", [fails, lengths]),
                 ("tables", [lengths, fails, fails]), ("tables", {"lengths": lengths, "fails": fails}),
                 ("tables", [lengths, "fails"])]:
        refused({**doc, k: v})
    for i, table in enumerate(doc["tables"]):
        for k, v in [("name", "ends"), ("value", f"responses/{table['name']}/0"), ("rows", T - 1), ("words", K),
                     ("domain", "00" * 64), ("root", "00" * 64), ("root", "00"), ("root", 7),
                     ("layout", {**table["layout"], "first_bucket": 0}),
                     ("layout", {**table["layout"], "slots": K + 1})]:
            refused({**doc, "tables": [{**t, k: v} if j == i else t for j, t in enumerate(doc["tables"])]})
        refused({**doc, "tables": [{**t, "extra": 1} if j == i else t for j, t in enumerate(doc["tables"])]},
                "no other")
    refused({**doc, "extra": 1}, "no other")
    refused({k: v for k, v in doc.items() if k != "certificate"}, "no other")
    refused({**doc, "certificate": {"key": "00"}}, "doesn't read")
    refused({**doc, "tables": [lengths, {**fails, "root": lengths["root"]}]}, "doesn't sign its head")
    with pytest.raises(ValueError, match="domain"):
        C.check_registration(doc, key, signers.OTHER_SESSION, *args)
    other = LmsKey(signers.SMALL, lms.public_key(signers.key(b"another")))
    with pytest.raises(ValueError, match="doesn't verify"):
        C.check_registration(doc, other, signers.SESSION, *args)
    with pytest.raises(ValueError):
        C.check_registration(doc, key, signers.SESSION, LEN, LINK, 2)
    with pytest.raises(ValueError, match="no connection rows"):
        C.check_registration(doc, key, signers.SESSION, PLAIN, LINK, 1)
    record = records(warden)[1][1]
    wire = issuer.certify(record, Direction.EGRESS, LEN.grid, LEN.digest).certificate
    _, cells_cert = W.attest(issuer, record, LEN, Direction.EGRESS)
    for foreign in (wire, cells_cert):
        assert foreign.verify(key, signers.SESSION) and (foreign.link, foreign.window) == (LINK, 1)
        swapped = {"key": foreign.key_id.hex(), "digest": foreign.digest.hex(), "signature": foreign.signature.hex()}
        refused({**doc, "certificate": swapped}, "doesn't sign its head")
        refused({**doc, "certificate": {**doc["certificate"], "digest": foreign.digest.hex()}}, "doesn't sign its head")
    refused({**doc, "certificate": {**doc["certificate"], "signature": cells_cert.signature.hex()}}, "doesn't verify")


def test_a_refused_signer_gives_no_certificate(handed, tmp_path):
    _, _, by_window = handed
    issuer = Issuer(signers.signer(tmp_path), signers.SESSION)
    issuer.signer.start(signers.OTHER_SESSION)
    rows, cert = C.attest(issuer, LINK, 0, by_window[0], LEN)
    assert cert is None and issuer.refused == [(LINK, Direction.EGRESS, 0)]
    assert [len(t.rows) for t in rows.tables] == [T, T]
