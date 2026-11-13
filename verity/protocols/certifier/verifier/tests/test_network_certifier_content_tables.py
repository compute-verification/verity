"""The network certifier's content tables (`content_rows`' "The tables", `connection_rows`, `warden`'s "The content
rows"), driven tick by tick on `test_network_certifier_warden`'s harness. A schedule without them has the digest,
records, heads and certificates it had. With them the device hands each release its request's bytes and its ids as
its own records hold them, the tables are the schedule's size whatever the load, a release is filed in the first
class that holds it, each row opens the one head's root at its place, the registration reads back, and the response
tie run on a response row and an end row the network certifier wrote returns the signed response row: for a
connection that ended, one that failed and one the client aborted. Under the lengths rows' form 2 a class is (P, O)
and its tables hold its held rows and no other: the head counts them, a class's three alike and at most
`WardenParams.phase_releases`, the verifier's read holds a registration to that, and a class's port is its held
rows."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json

import pytest

import signers
from verity.commitments.identity import canonical_json_bytes, tagged_sha512
from verity.protocols.certifier.prover import lengths_row as LR
from verity.protocols.certifier.prover import response_tie as RT
from verity.protocols.certifier.verifier import connection_rows as CR
from verity.protocols.certifier.verifier import content_rows as CT
from verity.protocols.certifier.verifier import warden_rows as WR
from verity.protocols.certifier.verifier.certificate import Direction, Issuer
from verity.protocols.certifier.verifier.tests.test_network_certifier_lengths import (B, LEN, PLAIN, PLAIN_DIGEST, P,
                                                                                      T, answer, id_req, response,
                                                                                      traffic)
from verity.protocols.certifier.verifier.tests.test_network_certifier_warden import Client, reasons, records, run
from verity.protocols.certifier.verifier.warden import KIND_NAMES, PARAMETERS_TAG, Released, WardenParams, cells

LINK = "responses"
#: the lengths tests' schedule with four classes: one of short requests, one of short responses, and two of one size,
#: as a stop class and a no-stop class of the same sizes are
CLASSES = ((4, 4, 2), (16, 4, 3), (128, 8, 2), (128, 8, 2))
CON = dataclasses.replace(LEN, content=CLASSES)
#: the same classes at the lengths rows' form 2, where a class is its two sizes and its tables hold its held rows
HELD = dataclasses.replace(LEN, lengths=2, content=tuple(c[:2] for c in CLASSES))
STRIDES, K = CR.strides_of(LEN), LEN.stride_releases
#: the lengths tests' schedule with lengths rows, as its digest was before the content rows' classes were a parameter
LEN_DIGEST = ("98280d93904a794e8b443e0c2bb917e0ce0fbb20462a8e99f42fdbe06b0b8510"
              "bc65791cc41703428a43976695a2fb75330780df7434190990751efd1202bd13")
#: `fingerprint` of the lengths tests' traffic without lengths rows and with them, as each was before the content
#: rows' classes were a parameter
PLAIN_RUN = ("8eed1068b18a7003962f93e66c503bbbda24485f4867e77084f20a7197d6bb28"
             "15c46c90d5f69c8058d22622cde13b00cf0db1c3437c64476acbdbc05b9c4edb")
LEN_RUN = ("efef231a72824c26fb520744475a0877de02f3dcb512ca382026808bf19ad84e"
           "20130b82af5fb5a0c42aead372a2dcc5c9da2b597c58636463ff8028121def54")


# --- a schedule without content rows -------------------------------------------------------------------------------


def salt(*name) -> bytes:
    """A row's salt, fixed by its name."""
    return b"".join(hashlib.sha512(repr((*name, i)).encode()).digest() for i in range(WR.SALT_BYTES // 64))


def fingerprint(params: WardenParams, directory) -> str:
    """`leaves` of the lengths tests' traffic under `params`."""
    _, arrivals, answers = traffic()
    device, _, _ = run(answers, arrivals, params=params, windows=3)
    return leaves(device, params, directory)


def leaves(device, params: WardenParams, directory) -> str:
    """A hash of everything a run of `device` leaves under `params`, with the signer's key, its coins and every salt
    fixed: each wire record's certificate (or that it has none, not being complete), each cells' head and its
    signature, the `fails` records, and each handed execution phase's lengths slots, connection rows' head and
    certificate."""
    counter = itertools.count()
    issuer = Issuer(signers.signer(directory), signers.SESSION,
                    coins=lambda n: hashlib.sha512(b"coins%d" % next(counter)).digest()[:n])
    h = hashlib.sha512()
    for record in device.records:
        if record.link == "fails":
            h.update(repr((record.link, record.window, record.status.reason, record.grid)).encode())
            continue
        direction = Direction.INGRESS if record.link == "requests" else Direction.EGRESS
        wire = issuer.certify(record, direction, params.grid, params.digest)
        if wire is None:
            h.update(repr(("refused", record.link, record.window, record.status.reason)).encode())
            continue
        h.update(wire.certificate.digest + wire.certificate.signature)
        held, cert = WR.attest(issuer, record, params, direction,
                               [salt(record.link, record.window, k) for k in range(params.window)],
                               hashlib.sha512(b"wiring").digest())
        h.update(held.head.digest() + cert.signature)
    for phase, strides in device.lengths:
        salts = {name: [salt(name, phase, k) for k in range(count)] for name, count, _ in CR.tables_of(params)}
        rows, cert = CR.attest(issuer, LINK, phase, strides, params, salts)
        h.update(rows.head.encode() + cert.digest + cert.signature)
        h.update(repr([[tuple(r)[:4] for r in stride] for stride in strides]).encode())
    return h.hexdigest()


def registered(device, params: WardenParams, directory) -> str:
    """A hash of every handed execution phase's registration, the public record the verifier reads, with the
    signer's key, its coins and every salt fixed as `leaves` fixes them."""
    counter = itertools.count()
    issuer = Issuer(signers.signer(directory), signers.SESSION,
                    coins=lambda n: hashlib.sha512(b"coins%d" % next(counter)).digest()[:n])
    h = hashlib.sha512()
    for phase, strides in device.lengths:
        salts = {name: [salt(name, phase, k) for k in range(count)] for name, count, _ in CR.tables_of(params)}
        rows, cert = CR.attest(issuer, LINK, phase, strides, params, salts)
        h.update(canonical_json_bytes(CR.registration(rows.head, cert, params)))
    return h.hexdigest()


def test_a_schedule_without_content_rows_has_the_digest_records_heads_and_certificates_it_had(tmp_path):
    """The classes are hashed only by a schedule that lists them, so a schedule without them, with lengths rows or
    without, has the parameters digest it had, and its run leaves the records, the heads and the certificates it left,
    byte for byte. Its device keeps and hands nothing more: a release has no request and no ids."""
    assert PLAIN.content is None and LEN.content is None and WardenParams().content is None
    assert PLAIN.digest.hex() == PLAIN_DIGEST and LEN.digest.hex() == LEN_DIGEST
    for name, params, was in (("plain", PLAIN, PLAIN_RUN), ("lengths", LEN, LEN_RUN)):
        (tmp_path / name).mkdir()
        assert fingerprint(params, tmp_path / name) == was
    _, arrivals, answers = traffic()
    device, _, _ = run(answers, arrivals, params=LEN, windows=3)
    assert all((r.request, r.ids) == (b"", b"") for _, strides in device.lengths for stride in strides for r in stride)
    assert [name for name, _, _ in CR.tables_of(LEN)] == ["lengths", "fails"] and CT.tables_of(LEN) == ()
    with pytest.raises(ValueError, match="no content rows"):
        CT.tables(LEN, [], [])


def test_the_classes_are_a_parameter_held_to_their_form():
    """A schedule's classes are a list of (P, O, rows) in a schedule with lengths rows: P and O powers of two of at
    least 2, P ids no longer than a request of the schedule, O at most a response row's ids, and rows from 1 to the
    lengths table's slots. Two classes may have one size. The list enters the digest in its order, and reads back
    from the parameters as a deployment writes them."""
    assert CON.content == CLASSES and CON.digest not in (LEN.digest, PLAIN.digest)
    assert dataclasses.replace(CON, content=None) == LEN
    assert dataclasses.replace(LEN, content=[list(c) for c in CLASSES]) == CON
    assert WardenParams(**json.loads(json.dumps(dataclasses.asdict(CON)))) == CON
    others = [CLASSES[::-1], CLASSES[:-1], ((4, 4, 3), *CLASSES[1:]), ((8, 4, 2), *CLASSES[1:]),
              ((4, 8, 2), *CLASSES[1:])]
    assert len({CON.digest, *(dataclasses.replace(LEN, content=c).digest for c in others)}) == len(others) + 1
    slots = STRIDES * K
    assert dataclasses.replace(LEN, content=((2, 2, 1), (128, 8, slots))).content == ((2, 2, 1), (128, 8, slots))
    with pytest.raises(ValueError, match="has lengths rows"):
        dataclasses.replace(PLAIN, content=CLASSES)
    for bad, why in (((), "1 to"), ([], "1 to"), ("4,4,2", "1 to"), (7, "1 to"), (((4, 4),), r"is \(P, O, rows\)"),
                     ((4, 4, 2), r"is \(P, O, rows\)"), (((3, 4, 1),), "power of two"), (((1, 4, 1),), "power of two"),
                     (((4, 3, 1),), "power of two"), (((4, 8192, 1),), "at most 4096 ids"),
                     (((4, 4, 0),), f"1 to {slots} rows"), (((4, 4, slots + 1),), f"1 to {slots} rows"),
                     (((4, 4, True),), f"1 to {slots} rows"), (((4096, 4, 1),), "is at most 2012 bytes")):
        with pytest.raises(ValueError, match=why):
            dataclasses.replace(LEN, content=bad)


def test_under_form_2_a_class_is_its_two_sizes_and_each_form_refuses_the_others_class():
    """Under the lengths rows' form 2 a class is (P, O): its tables hold a row a release filed in it and no other,
    so it has no rows of its own, and the most they hold is R, the bound on an execution phase's releases. A third
    number is refused under form 2 and a missing one under form 1, each naming what to change. The two sizes are
    held as form 1's are, and the two forms' schedules of the same sizes have two digests."""
    pairs = tuple(c[:2] for c in CLASSES)
    assert HELD.content == pairs and CT.counted(HELD) and not CT.counted(CON)
    assert len({HELD.digest, CON.digest, LEN.digest, dataclasses.replace(LEN, lengths=2).digest}) == 4
    assert WardenParams(**json.loads(json.dumps(dataclasses.asdict(HELD)))) == HELD
    R = HELD.phase_releases
    assert R == 6 and CT.sized(HELD) == tuple((p, o, R) for p, o in pairs) and CT.sized(CON) == CLASSES
    assert CT.sized(LEN) == CT.sized(dataclasses.replace(LEN, lengths=2)) == ()
    assert [(name, rows) for name, rows, _ in CR.tables_of(HELD)] == [
        ("lengths", STRIDES), ("fails", STRIDES), *((name, R) for i in range(len(pairs)) for name in CT.names(i))]
    assert [(name, size) for name, _, size in CR.tables_of(HELD)] == \
        [(name, size) for name, _, size in CR.tables_of(CON)]
    for built in (lambda: dataclasses.replace(HELD, content=CLASSES), lambda: dataclasses.replace(CON, lengths=2)):
        with pytest.raises(ValueError, match=r"class 0 is \(P, O\) under the lengths rows' form 2, not \(4, 4, 2\): "
                                             r"drop its third number"):
            built()
    for built in (lambda: dataclasses.replace(CON, content=pairs), lambda: dataclasses.replace(HELD, lengths=1)):
        with pytest.raises(ValueError, match=r"class 0 is \(P, O, rows\) under the lengths rows' form 1, not \(4, 4\): "
                                             r"give the rows of each of its tables, 1 to 16"):
            built()
    for bad, why in (((), r"each \(P, O\), or unset"), (((4,),), r"is \(P, O\), not"),
                     (((4, 4, 2, 1),), r"is \(P, O\), not"), ((4, 4), r"is \(P, O\), not 4"),
                     (((3, 4),), "power of two"), (((4, 3),), "power of two"),
                     (((4, 8192),), "at most 4096 ids"), (((4096, 4),), "is at most 2012 bytes")):
        with pytest.raises(ValueError, match=why):
            dataclasses.replace(HELD, content=bad)


# --- the traffic ---------------------------------------------------------------------------------------------------


#: what the server sent on each released connection, by its place in the release order, and how each ended
SENT = ((11, 22, 33), tuple(range(8)), (), (7, 8, 9, 10), (), (1, 2), (99,))
FAILED = (False, False, False, False, True, True, True)
#: the class the network certifier's rule files each in, and its row there
FILED = ((0, 0), (2, 0), (2, 1), (0, 1), (1, 0), (1, 1), (1, 2))
#: the clients that go away after their release, by connect order from 1, and the bucket each is gone from: one whose
#: server ends the response, and one whose server stops answering
GONE = {4: 16, 8: 27}


def mixed():
    """Seven released requests over two execution phases, and one that never is. In the first: two released in one
    slot stride, one whose response has no id, and one whose client goes away after its first two ids and whose
    response ends in the next execution phase. In the second: one whose server never answers, one whose server stops
    after two ids, a client cut short in its request (never released), and one whose client goes away before any id
    and whose server stops after one."""
    reqs = [id_req(1, 3), id_req(2, 40), id_req(3, 100), id_req(4, 3), id_req(5, 5), id_req(6, 7), id_req(7, 6)]
    short = cells(id_req(8, 40), B)[:-10]
    arrivals = {0: [cells(reqs[0], B), cells(reqs[1], B)], 8: [cells(reqs[2], B)], 10: [cells(reqs[3], B)],
                17: [cells(reqs[4], B)], 18: [cells(reqs[5], B)], 19: [short], 24: [cells(reqs[6], B)]}
    answers = [answer(SENT[0]), answer(SENT[1]), answer(SENT[2]), answer(SENT[3]), [],
               [(1, response(SENT[5])[:2 * P])], [], [(1, response(SENT[6])[:P])]]
    return reqs, arrivals, answers


def leaving(gone: dict[int, int]) -> type[Client]:
    """Clients of which the k-th to connect, k from 1, goes away at bucket `gone[k]`: from then its connection is
    closing, as a transport's is whose peer is gone, and the network certifier writes it nothing."""
    count = itertools.count(1)

    class Leaving(Client):
        def __init__(self, device, data) -> None:
            self.gone = gone.get(next(count))
            super().__init__(device, data)

        def is_closing(self) -> bool:
            return self.closed or (self.gone is not None and self.warden.ticks >= self.gone)

    return Leaving


@pytest.fixture(scope="module")
def ran():
    """`mixed` through the network certifier under the schedule with content rows: its requests, the device, its
    clients and its handed releases by execution phase."""
    reqs, arrivals, answers = mixed()
    device, clients, _ = run(answers, arrivals, params=CON, windows=3, client=leaving(GONE))
    return reqs, device, clients, dict(device.lengths)


@pytest.fixture(scope="module")
def signed(ran, tmp_path_factory):
    """Each handed execution phase's tables committed and their one head signed, and the registration as the
    verifier reads it: by execution phase, the opening, the registration, and the head and domains the verifier
    derived."""
    _, _, _, handed = ran
    issuer = Issuer(signers.signer(tmp_path_factory.mktemp("signer")), signers.SESSION)
    out = {phase: read(issuer, phase, strides, CON) for phase, strides in handed.items()}
    assert signers.used(issuer.signer) == [0, 1]
    return issuer, out


def read(issuer: Issuer, phase: int, strides, params: WardenParams) -> tuple:
    """An execution phase's tables committed and their head signed, and the registration read as the verifier reads
    it: the opening, the registration, and the head and domains the verifier derived."""
    opened, cert = CR.attest(issuer, LINK, phase, strides, params)
    doc = json.loads(json.dumps(CR.registration(opened.head, cert, params)))
    head, domains = CR.check_registration(doc, issuer.signer.public_key(), signers.SESSION, params, LINK, phase)
    assert head == opened.head and cert.digest == head.digest()
    return opened, doc, head, domains


#: the releases three loads file in each class: the traffic's first execution phase, its second, and one of none
COUNTS = {"first": (2, 0, 2, 0), "second": (0, 3, 0, 0), "none": (0, 0, 0, 0)}


@pytest.fixture(scope="module")
def loads(ran):
    """The three loads of `COUNTS`, each an execution phase's releases by slot stride."""
    _, _, _, handed = ran
    return {"first": handed[0], "second": handed[1], "none": ((),) * STRIDES}


@pytest.fixture(scope="module")
def counted(loads, tmp_path_factory):
    """Each load as execution phase 0 of the form-2 schedule, signed under one key: the key, and by load what
    `read` gives."""
    out = {}
    for load, strides in loads.items():
        issuer = Issuer(signers.signer(tmp_path_factory.mktemp(load)), signers.SESSION)
        out[load] = read(issuer, 0, strides, HELD)
    return issuer.signer.public_key(), out


@pytest.fixture(scope="module", params=(1, 2), ids=("form1", "form2"))
def tied(request, ran, signed, tmp_path_factory):
    """The traffic's handed execution phases that released a request, and what `read` gives of each, under each of
    the lengths rows' forms."""
    if request.param == 1:
        return ran[3], signed[1]
    _, arrivals, answers = mixed()
    device, _, _ = run(answers, arrivals, params=HELD, windows=3, client=leaving(GONE))
    handed = {phase: strides for phase, strides in device.lengths if any(strides)}
    issuer = Issuer(signers.signer(tmp_path_factory.mktemp("held")), signers.SESSION)
    return handed, {phase: read(issuer, phase, strides, HELD) for phase, strides in handed.items()}


def released(handed) -> list[Released]:
    """Every handed release, in release order."""
    return [r for _, strides in sorted(handed.items()) for stride in strides for r in stride]


def rows_of(tables, i: int, r: int) -> tuple[bytes, bytes, bytes]:
    """Row r of class i's request, response and end tables."""
    return tuple(tables[name][r] for name in CT.names(i))


# --- the device ----------------------------------------------------------------------------------------------------


def test_the_device_hands_each_release_its_request_and_its_ids_as_its_own_records_hold_them(ran):
    """With content rows a release carries the request's bytes as the network certifier cut them and the id of each
    `DATA` cell it released: the bytes the client sent and the ids the server sent before its `END` or the `FAIL`.
    They are what the device's own signed records of the same cells hold (`connection_rows.of_records`), release for
    release, and keeping them changes no cell: the records are the ones the schedule without content rows leaves."""
    reqs, device, _, handed = ran
    assert sorted(handed) == [0, 1]
    got = released(handed)
    assert [r.order for r in got] == list(range(7))
    assert [(r.request, CT.ids_of(r.ids), r.failed) for r in got] == list(zip(reqs, SENT, FAILED))
    assert all((r.request_bytes, 4 * r.outputs) == (len(r.request), len(r.ids)) for r in got)
    assert got[0].ids == b"\x0b\0\0\0\x16\0\0\0\x21\0\0\0"
    ingress, egress, _ = records(device)
    read = CR.of_records(CON, ingress, egress)
    assert sorted(read) == [0, 1, 2] and {w: read[w] for w in handed} == handed
    assert read[2] == ((),) * STRIDES
    _, arrivals, answers = mixed()
    plain, _, _ = run(answers, arrivals, params=LEN, windows=3, client=leaving(GONE))
    assert [[r.grid for r in link] for link in records(plain)] == [[r.grid for r in link] for link in records(device)]
    bare = CR.of_records(LEN, *records(plain)[:2])
    assert {w: bare[w] for w in handed} == dict(plain.lengths)
    assert [tuple(r)[:4] for r in released(dict(plain.lengths))] == [tuple(r)[:4] for r in got]
    assert all((r.request, r.ids) == (b"", b"") for r in released(dict(plain.lengths)))


def test_the_records_reading_takes_a_runs_records_from_its_first_and_leaves_out_a_phase_still_live(ran):
    """N counts from the run's first release, so the reading takes each direction's records from the first execution
    phase on, in order. An execution phase with a release whose connection hasn't ended in the records given has no
    entry: the first's, here, whose last response ends in the second."""
    _, device, _, handed = ran
    ingress, egress, _ = records(device)
    assert sorted(CR.of_records(CON, ingress[:1], egress[:1])) == []
    assert CR.of_records(CON, ingress[:2], egress[:2]) == handed
    for bad in ((ingress[1:], egress[1:]), (ingress, egress[:2]), (ingress[::-1], egress[::-1])):
        with pytest.raises(ValueError, match="from its first execution phase"):
            CR.of_records(CON, *bad)
    with pytest.raises(ValueError, match="no connection rows"):
        CR.of_records(PLAIN, ingress, egress)


# --- the tables ----------------------------------------------------------------------------------------------------


def test_the_tables_are_the_schedules_size_whatever_the_load_and_hold_each_release_in_its_class(ran):
    """Three tables a class after the lengths and fails tables, in the schedule's order, each of the class's rows
    and of rows of the class's size, in an execution phase with releases and in one with none. Row r of class i is
    the r-th release of the execution phase the rule files there: its request, its ids and how it ended, as the
    decoders read them back. Every row past the held ones is zero bytes."""
    _, _, _, handed = ran
    want = [("lengths", STRIDES, K * 12), ("fails", STRIDES, K * 4)]
    for i, (p, o, rows) in enumerate(CLASSES):
        want += [(f"request.{i}", rows, CT.request_bytes(B, p)), (f"response.{i}", rows, 4 * (o + 1)),
                 (f"end.{i}", rows, 4)]
    assert list(CR.tables_of(CON)) == want and CT.tables_of(CON) == tuple(want[2:])
    assert [CT.request_bytes(B, p) for p, _, _ in CLASSES] == [418, 554, 1802, 1802]
    idle = CR.tables(CON, ((),) * STRIDES)
    assert idle == {name: [bytes(size)] * rows for name, rows, size in want}
    made = {phase: CR.tables(CON, strides) for phase, strides in handed.items()}
    for tables in (*made.values(), idle):
        assert list(tables) == [name for name, _, _ in want]
        assert [(len(tables[name]), {len(row) for row in tables[name]}) for name, _, _ in want] == \
            [(rows, {size}) for _, rows, size in want]
    by_phase = {phase: [r for stride in strides for r in stride] for phase, strides in handed.items()}
    assert [len(by_phase[0]), len(by_phase[1])] == [4, 3]
    held = {phase: {FILED[r.order]: r for r in by_phase[phase]} for phase in handed}
    for phase, tables in made.items():
        for i, (p, o, rows) in enumerate(CLASSES):
            for r in range(rows):
                request, ids, end = rows_of(tables, i, r)
                release = held[phase].get((i, r))
                if release is None:
                    assert (request, ids, end) == CT.empty_rows(B, p, o)
                    continue
                assert CT.request_of(request, B, p) == release.request and CT.held(request)
                assert CT.response_of(ids, o) == CT.ids_of(release.ids)
                assert CT.end_of(end) == (release.failed, release.outputs)
        assert CR.releases(CON, tables["lengths"], tables["fails"]) == \
            [[Released(*tuple(r)[:4]) for r in stride] for stride in handed[phase]]
    # the lengths and fails tables are the ones the schedule without content rows lays from the same releases
    for phase, strides in handed.items():
        bare = CR.tables(LEN, strides)
        assert bare == {name: made[phase][name] for name in ("lengths", "fails")}


def test_a_release_is_filed_in_the_first_class_that_holds_it_and_the_tables_lay_the_class_they_are_given(
        ran, monkeypatch):
    """The network certifier's rule reads a release's own sizes and nothing else: the first class the schedule lists
    whose request row holds its request and whose response row holds its ids. So of two classes of one size the rule
    files in the first only. The class is an argument of `content_rows.tables`, which lays a release in whichever
    class it is told that holds it: with another rule in the one function, the tables are that rule's. A release
    that no class holds, or whose class's rows are all held, has no content row, and its lengths and fails slots are
    the ones it had."""
    _, _, _, handed = ran
    got = released(handed)
    assert [CT.file(CON, r) for r in got] == [i for i, _ in FILED]
    first = [r for stride in handed[0] for r in stride]
    ruled = {name: rows for name, rows in CR.tables(CON, handed[0]).items() if name not in ("lengths", "fails")}
    assert ruled == CT.tables(CON, first, [0, 2, 2, 0])
    assert not any(any(row) for name in CT.names(3) for row in ruled[name])
    told = CT.tables(CON, first, [0, 3, 3, 0])
    assert [told[name] for name in CT.names(3)] == [ruled[name] for name in CT.names(2)]
    assert [told[name] for name in CT.names(0)] == [ruled[name] for name in CT.names(0)]
    assert not any(any(row) for name in CT.names(2) for row in told[name])
    assert [CT.held(row) for row in told["request.3"]] == [True, True]
    monkeypatch.setattr(CT, "file", lambda params, r: max(i for i, (p, o, _) in enumerate(params.content)
                                                          if len(r.request) <= CT.request_room(B, p)
                                                          and r.outputs <= o))
    swapped = CR.tables(CON, handed[0])
    assert {name: swapped[name] for name in ruled} == CT.tables(CON, first, [3, 3, 3, 3]) != ruled
    monkeypatch.undo()
    # a class with a row for one release holds the execution phase's first that fits; a class too small holds none
    one, small = (dataclasses.replace(LEN, content=c) for c in (((128, 8, 1),), ((2, 2, 1), (4, 2, 1))))
    assert [CT.file(one, r) for r in first] == [0, 0, 0, 0]
    made = CR.tables(one, handed[0])
    assert CT.request_of(made["request.0"][0], B, 128) == first[0].request and len(made["request.0"]) == 1
    assert [CT.file(small, r) for r in first] == [None] * 4
    none = CR.tables(small, handed[0])
    assert not any(any(row) for name in (*CT.names(0), *CT.names(1)) for row in none[name])
    for params, tables in ((one, made), (small, none)):
        assert {name: tables[name] for name in ("lengths", "fails")} == CR.tables(LEN, handed[0])
    # what `content_rows.tables` refuses
    a, b, c, d = first
    for releases, filing, why in (
            (first, [0, 2, 2], "a class, or None, a release"), (first, [0, 4, 2, 0], "the schedule lists 4"),
            (first, [0, -1, 2, 0], "the schedule lists 4"), (first, [0, True, 2, 0], "the schedule lists 4"),
            (first, [0, 0, 2, 0], "which holds 413 bytes and 4 ids"), (first, [1, 2, 2, 0], None),
            ([b, a], [2, 0], "in release order"), ([a, a], [0, 0], "in release order"),
            ([Released(*tuple(a)[:4])], [0], "not the content of its lengths slot"),
            ([a._replace(ids=a.ids[:-4])], [0], "not the content of its lengths slot"),
            ([a._replace(ids=a.ids[:-1])], [0], "not the content of its lengths slot"),
            ([a._replace(request=a.request + b"x")], [0], "not the content of its lengths slot")):
        if why is None:  # a release whose ids fit the class it is told: laid, never refused
            assert CT.held(CT.tables(CON, releases, filing)["request.1"][0])
            continue
        with pytest.raises(ValueError, match=why):
            CT.tables(CON, releases, filing)
    with pytest.raises(ValueError, match="not the content of its lengths slot"):
        CR.tables(CON, tuple(tuple(Released(*tuple(r)[:4]) for r in stride) for stride in handed[0]))


def test_a_row_of_zero_bytes_reads_as_each_tables_decoder_says(ran):
    """Where no release is filed every table holds zero bytes. The request table's decoder refuses them, since a
    released request has a byte; the response table's reads a response of no id, and the end table's an `END` after
    no `DATA` cell. A connection that ended before any id therefore has the response row and the end row of a row
    that holds none, and its request row alone says it is held."""
    _, _, _, handed = ran
    tables = CR.tables(CON, handed[0])
    for i, (p, o, rows) in enumerate(CLASSES):
        zero = CT.empty_rows(B, p, o)
        assert zero == (bytes(CT.request_bytes(B, p)), bytes(CT.response_bytes(o)), bytes(CT.END_BYTES))
        assert not CT.held(zero[0])
        with pytest.raises(ValueError, match="the request row's bytes is 0"):
            CT.request_of(zero[0], B, p)
        assert CT.response_of(zero[1], o) == () and CT.end_of(zero[2]) == (False, 0)
        assert zero[1] == CT.response_row((), o) and zero[2] == CT.end_row(False, 0)
    no_id = released(handed)[2]
    assert (no_id.ids, no_id.failed) == (b"", False) and FILED[2] == (2, 1)
    request, ids, end = rows_of(tables, 2, 1)
    assert (ids, end) == (bytes(36), bytes(4)) and CT.held(request)
    assert CT.request_of(request, B, 128) == no_id.request
    # a connection that failed before any id is told apart by its end row too
    late = released(handed)[4]
    assert (late.ids, late.failed) == (b"", True)
    assert rows_of(CR.tables(CON, handed[1]), 1, 0)[1:] == (bytes(20), CT.end_row(True, 0))


# --- the head and the registration ---------------------------------------------------------------------------------


def test_one_head_lists_every_table_and_says_nothing_of_the_load(ran, signed, tmp_path):
    """The head lists the lengths table, the fails table and each class's three, in the schedule's order, and is
    signed once. Its fields and sizes, and the registration's, are the schedule's: an execution phase with four
    releases, one with three and one with none differ in their roots and the certificate alone, so nothing public
    counts a class's held rows."""
    _, _, _, handed = ran
    issuer, by_phase = signed
    names = [name for name, _, _ in CR.tables_of(CON)]
    assert len(names) == 2 + 3 * len(CLASSES)
    for phase, (opened, doc, head, _) in by_phase.items():
        assert [e.name for e in head.tables] == names == [t["name"] for t in doc["tables"]]
        assert [(e.rows, e.row_bytes) for e in head.tables] == [(rows, size) for _, rows, size in CR.tables_of(CON)]
        assert head.encode()[:len(LINK) + 2 + 8 + 64 + 2] == b"".join([
            b"\0\x09responses", phase.to_bytes(8, "big"), CON.digest, len(names).to_bytes(2, "big")])
        for i, (p, o, rows) in enumerate(CLASSES):
            for name, words in zip(CT.names(i), (CT.request_bytes(B, p) // 2, 2 * (o + 1), 2)):
                t = doc["tables"][names.index(name)]
                assert (t["value"], t["rows"], t["words"]) == (f"{LINK}/{name}/{phase}", rows, words)
                assert t["layout"] == {"class": i, "prompt_ids": p, "ids": o, "first_bucket": phase * T,
                                       "buckets": T}
    loads = {"first": handed[0], "second": handed[1], "none": ((),) * STRIDES}
    docs = {}
    for name, strides in loads.items():
        (tmp_path / name).mkdir()
        other = Issuer(signers.signer(tmp_path / name), signers.SESSION)
        opened, cert = CR.attest(other, LINK, 0, strides, CON)
        assert signers.used(other.signer) == [0]
        docs[name] = (CR.registration(opened.head, cert, CON), len(opened.head.encode()))

    def public(doc):
        return {**doc, "certificate": None, "tables": [{**t, "root": None} for t in doc["tables"]]}

    assert len({size for _, size in docs.values()}) == 1
    assert public(docs["first"][0]) == public(docs["second"][0]) == public(docs["none"][0])
    assert len({t["root"] for doc, _ in docs.values() for t in doc["tables"]}) == 3 * len(names)


def test_the_registration_of_a_schedule_with_content_rows_refuses_anything_but_its_own(signed):
    """The verifier's read lists the schedule's tables and no other: a registration without the content tables, with
    two of them in another order, with a class's layout or size changed, or read under the schedule without content
    rows or with other classes, is refused."""
    issuer, by_phase = signed
    _, doc, _, _ = by_phase[1]
    key = issuer.signer.public_key()
    tables = doc["tables"]

    def refused(changed, why=None, params=CON):
        with pytest.raises(ValueError, match=why):
            CR.check_registration(changed, key, signers.SESSION, params, LINK, 1)

    refused({**doc, "tables": tables[:2]}, "lists the schedule's 14 tables")
    refused({**doc, "tables": [*tables[:2], tables[3], tables[2], *tables[4:]]}, "request.0 table has name")
    refused({**doc, "tables": [*tables[:8], *tables[11:], *tables[8:11]]}, "request.2 table has name")
    for i, (k, v) in ((2, ("layout", {**tables[2]["layout"], "class": 1})), (2, ("rows", 3)), (3, ("words", 12)),
                      (4, ("layout", {**tables[4]["layout"], "ids": 8})), (4, ("root", tables[7]["root"])),
                      (8, ("root", tables[11]["root"])), (8, ("domain", tables[11]["domain"]))):
        refused({**doc, "tables": [{**t, k: v} if j == i else t for j, t in enumerate(tables)]})
    refused(doc, "parameters", LEN)
    refused(doc, "parameters", dataclasses.replace(LEN, content=CLASSES[::-1]))
    refused({**doc, "parameters": LEN.digest.hex()}, "parameters")


# --- form 2: a class's tables are its held rows --------------------------------------------------------------------


def test_under_form_2_a_classes_tables_are_its_held_rows_and_no_other(ran, loads):
    """Under form 2 the same hand lays, for each class, the rows form 1 lays first and nothing after them: a row a
    release filed in the class, in release order, and three tables of no row where none is filed. The lengths and
    fails tables are form 1's. A class's rows are all held at R: R releases filed in it are laid, and with one more
    the hand is refused, naming the release, the class and R, where form 1 leaves a release past a class's rows
    unfiled. The commitment takes a class's three tables only at one count of at most R. The device hands under
    form 2 what it hands under form 1, and its last execution phase, of no release, too; the records' reading
    agrees."""
    for load, strides in loads.items():
        one, two = CR.tables(CON, strides), CR.tables(HELD, strides)
        assert list(two) == list(one) and all(two[name] == one[name] for name in ("lengths", "fails"))
        for i, count in enumerate(COUNTS[load]):
            assert [two[name] for name in CT.names(i)] == [one[name][:count] for name in CT.names(i)]
            assert all(CT.held(row) for row in two[f"request.{i}"])
            assert [CT.held(row) for row in one[f"request.{i}"]] == [True] * count + [False] * (CLASSES[i][2] - count)
    R = HELD.phase_releases
    a = loads["first"][0][0]
    assert CT.file(HELD, a) == CT.file(CON, a) == 0
    many = [a._replace(order=n) for n in range(R + 1)]

    def by_stride(releases: list) -> tuple:
        return tuple(tuple(releases[j * K:(j + 1) * K]) for j in range(STRIDES))

    full = CR.tables(HELD, by_stride(many[:R]))
    assert {name: full[name] for name in CT.names(0)} == \
        {name: [row] * R for name, row in zip(CT.names(0), rows_of(CR.tables(HELD, loads["first"]), 0, 0))}
    assert sum(map(len, CR.releases(HELD, full["lengths"], full["fails"]))) == R
    assert [e.rows for e in CR.Rows.commit(LINK, 0, full, HELD, signers.SESSION).head.tables[2:5]] == [R] * 3
    past = f"release {R} is filed in class 0 past the class's {R} rows: R = {R} \\(WardenParams.phase_releases\\)"
    with pytest.raises(ValueError, match=past):
        CR.tables(HELD, by_stride(many))
    with pytest.raises(ValueError, match=past):
        CT.tables(HELD, many, [0] * (R + 1))
    form_1 = CR.tables(CON, by_stride(many))
    assert [CT.held(row) for row in form_1["request.0"]] == [True] * CLASSES[0][2] and CLASSES[0][2] < R + 1
    assert sum(map(len, CR.releases(CON, form_1["lengths"], form_1["fails"]))) == R + 1 <= STRIDES * K
    row = full["request.0"][0]
    for changed, why in (({"request.0": [row] * (R + 1)}, f"request.0 table is at most {R} rows of 418 bytes"),
                         ({"response.0": full["response.0"][:-1]}, "response.0 table has 5 rows, and the request.0 "
                                                                   "table 6"),
                         ({"end.0": []}, "end.0 table has 0 rows, and the response.0 table 6"),
                         ({"end.1": [bytes(4)]}, "end.1 table has 1 rows, and the response.1 table 0"),
                         ({"lengths": full["lengths"][:-1]}, "lengths table is 8 rows of 24 bytes"),
                         ({"request.0": [row[:-1]] * R}, f"request.0 table is at most {R} rows of 418 bytes")):
        with pytest.raises(ValueError, match=why):
            CR.Rows.commit(LINK, 0, full | changed, HELD, signers.SESSION)
    _, arrivals, answers = mixed()
    device, _, _ = run(answers, arrivals, params=HELD, windows=3, client=leaving(GONE))
    assert device.lengths == [(0, loads["first"]), (1, loads["second"]), (2, loads["none"])]
    assert CR.of_records(HELD, *records(device)[:2]) == dict(device.lengths)


def test_under_form_2_one_head_counts_each_classes_held_rows_and_its_other_fields_are_the_schedules(loads, counted):
    """The opposite of form 1's head: under form 2 the same three loads give heads whose content entries are the
    load's. A class's three entries hold the count of the releases filed in it, each with the domain of that count;
    two loads that file different counts in a class differ there in rows, domain and root; a class in which none is
    filed has 0 rows and the zero root under every load; and the same load committed again has the rows and domains
    it had and other roots where it holds a row. The lengths and fails entries are the schedule's whatever the load
    but for their roots, and so is the rest of the head and of the registration: the tables' names and order, a
    row's size, the layouts and the head's bytes."""
    _, by_load = counted
    names = [name for name, _, _ in CR.tables_of(HELD)]
    heads = {load: head for load, (_, _, head, _) in by_load.items()}
    for load, head in heads.items():
        assert [e.name for e in head.tables] == names
        assert [(e.rows, e.row_bytes) for e in head.tables[:2]] == [(STRIDES, K * 12), (STRIDES, K * 4)]
        assert [e.rows for e in head.tables[2:]] == [count for count in COUNTS[load] for _ in range(3)]
        assert [e.row_bytes for e in head.tables] == [size for _, _, size in CR.tables_of(CON)]
        for e in head.tables:
            assert (e.root == CR.NO_ROWS_ROOT) == (e.rows == 0) and CR.NO_ROWS_ROOT == bytes(64)
            assert e.domain == CR.domain(HELD, signers.SESSION, LINK, 0, e.name, e.rows, e.row_bytes).domain_id
    assert len({len(head.encode()) for head in heads.values()}) == 1
    for a, b in itertools.combinations(heads, 2):
        for name in names[:2]:
            x, y = heads[a].table(name), heads[b].table(name)
            assert (x.rows, x.row_bytes, x.domain) == (y.rows, y.row_bytes, y.domain) and x.root != y.root
        for i in range(len(CLASSES)):
            for x, y in ((heads[a].table(name), heads[b].table(name)) for name in CT.names(i)):
                if COUNTS[a][i] == COUNTS[b][i]:
                    assert x == y and x.rows == 0
                else:
                    assert x.rows != y.rows and x.domain != y.domain and x.root != y.root
    again = CR.Rows.commit(LINK, 0, CR.tables(HELD, loads["first"]), HELD, signers.SESSION).head
    for x, y in zip(again.tables, heads["first"].tables):
        assert (x.name, x.rows, x.row_bytes, x.domain) == (y.name, y.rows, y.row_bytes, y.domain)
        assert (x.root == y.root) == (x.rows == 0)

    def public(doc):
        """A registration less what a load may move: its certificate, each root, and a class's rows and domains."""
        counts = {"rows": None, "domain": None}
        return {**doc, "certificate": None, "tables": [{**t, "root": None} | (counts if k >= 2 else {})
                                                       for k, t in enumerate(doc["tables"])]}

    docs = {load: doc for load, (_, doc, _, _) in by_load.items()}
    assert public(docs["first"]) == public(docs["second"]) == public(docs["none"])
    assert [[t["rows"] for t in doc["tables"][2::3]] for doc in docs.values()] == [list(c) for c in COUNTS.values()]


def recounted(doc: dict, rows: dict, derive: bool = True) -> dict:
    """A form-2 registration of execution phase 0 with the rows of each table `rows` names changed and, with
    `derive`, each one's domain the one the schedule gives at those rows."""
    sizes = {name: size for name, _, size in CR.tables_of(HELD)}
    return {**doc, "tables": [
        t if t["name"] not in rows else {**t, "rows": rows[t["name"]]} | ({"domain": CR.domain(
            HELD, signers.SESSION, LINK, 0, t["name"], rows[t["name"]], sizes[t["name"]]).domain_id.hex()}
            if derive else {}) for t in doc["tables"]]}


def test_a_form_2_registration_that_counts_more_rows_than_an_execution_phase_releases_is_refused(counted):
    """The verifier reads a class's count from the registration and takes one from 0 to R = 6 and no other: three
    tables of a class at R + 1 rows, each with the domain of that count, are refused naming the bound, and so is a
    count that is no such integer. A count inside the bound that isn't the signed one is refused too: by its domain
    where that is the signed count's, and by the certificate where it is its own."""
    key, by_load = counted
    _, doc, _, _ = by_load["first"]
    R = HELD.phase_releases

    def check(changed) -> None:
        CR.check_registration(changed, key, signers.SESSION, HELD, LINK, 0)

    check(doc)
    bound = f"request.0 table has rows .*, at most the {R} an execution phase of the schedule releases"
    with pytest.raises(ValueError, match=bound):
        check(recounted(doc, dict.fromkeys(CT.names(0), R + 1)))
    for rows in (R + 1, 1 << 32, -1, True, "2", 2.0, None):
        with pytest.raises(ValueError, match=bound):
            check(recounted(doc, dict.fromkeys(CT.names(0), rows), derive=False))
    for rows in (R, 1, 0):
        with pytest.raises(ValueError, match="request.0 table has domain"):
            check(recounted(doc, dict.fromkeys(CT.names(0), rows), derive=False))
    for rows in (R, 1):
        with pytest.raises(ValueError, match="doesn't sign its head"):
            check(recounted(doc, dict.fromkeys(CT.names(0), rows)))


def test_a_form_2_registration_whose_classes_three_tables_differ_in_rows_is_refused(counted):
    """A class's three tables hold a row a release filed in it, so they have one count. A registration that gives a
    class's response table or its end table another count than the table before it, each inside the bound and with
    the domain of its own count, is refused naming the two tables."""
    key, by_load = counted
    for load, name, rows, like in (
            ("first", "response.0", 1, "request.0 table 2"), ("first", "end.0", 3, "response.0 table 2"),
            ("first", "response.1", 1, "request.1 table 0"), ("first", "end.2", 0, "response.2 table 2"),
            ("second", "end.1", 2, "response.1 table 3"), ("none", "response.3", 6, "request.3 table 0")):
        _, doc, _, _ = by_load[load]
        with pytest.raises(ValueError, match=f"{name} table has rows {rows}, and its class's {like}: a class's three"):
            CR.check_registration(recounted(doc, {name: rows}), key, signers.SESSION, HELD, LINK, 0)
    # a class's request table alone at another count: the response table is then the one that differs
    _, doc, _, _ = by_load["first"]
    with pytest.raises(ValueError, match="response.0 table has rows 2, and its class's request.0 table 3"):
        CR.check_registration(recounted(doc, {"request.0": 3}), key, signers.SESSION, HELD, LINK, 0)


def test_a_form_2_registration_takes_a_table_of_no_row_only_with_the_zero_root(counted):
    """A class in which no release is filed has three entries of 0 rows and the zero root, and the verifier's read
    takes them: the head it returns holds them, under the domains of 0 rows. With any other root, a held table's or
    the zero root with a bit set, a table of no row is refused naming it, before the certificate is read. A table
    that holds rows is not taken at no row either: at 0 rows and the zero root its class's other tables differ, and
    all three at 0 rows are not what the certificate signs."""
    key, by_load = counted
    _, doc, head, domains = by_load["none"]
    assert all((e.rows, e.root) == (0, bytes(64)) for e in head.tables[2:])
    assert all(domains[e.name].domain_id == e.domain for e in head.tables)
    held_root = by_load["first"][1]["tables"][2]["root"]
    assert held_root != "00" * 64
    for k in range(2, len(doc["tables"])):
        for root in (held_root, "00" * 63 + "01"):
            changed = {**doc, "tables": [{**t, "root": root} if j == k else t for j, t in enumerate(doc["tables"])]}
            with pytest.raises(ValueError, match=f"{doc['tables'][k]['name']} table has no row and the root"):
                CR.check_registration(changed, key, signers.SESSION, HELD, LINK, 0)
    _, doc, _, _ = by_load["first"]

    def emptied(names) -> dict:
        changed = recounted(doc, dict.fromkeys(names, 0))
        return {**changed, "tables": [{**t, "root": "00" * 64} if t["name"] in names else t for t in changed["tables"]]}

    with pytest.raises(ValueError, match="response.0 table has rows 2, and its class's request.0 table 0"):
        CR.check_registration(emptied(("request.0",)), key, signers.SESSION, HELD, LINK, 0)
    with pytest.raises(ValueError, match="doesn't sign its head"):
        CR.check_registration(emptied(CT.names(0)), key, signers.SESSION, HELD, LINK, 0)
    with pytest.raises(ValueError, match="request.0 table has no row and the root"):
        CR.check_registration(recounted(doc, dict.fromkeys(CT.names(0), 0)), key, signers.SESSION, HELD, LINK, 0)


#: the digest of a form-2 schedule whose classes are `CLASSES`, rows and all, and whose tables are those rows
#: whatever the load, and the digest of its head of `one_release`, as the code that read form 2 so made them
EARLIER_DIGEST = ("217da74969e7a0375e5aea5c9f9e79cf61518e5ca907aed43bc11c4cf67dc485"
                  "90cc819a86a67768f9e9423d65d585b021b94395dcaa2a35a024720f9943ac1a")
EARLIER_HEAD = ("c8b080ed53b9b68fa9ce267dca0ace2c170e7350c6183f6bc5ea1a37b878886f"
                "45eac7bedc10511c2cca76ea6559196740277537283d7ffce34474adfdea5e2d")


def one_release() -> tuple:
    """An execution phase's hand of one release, of five bytes and two ids, in its first slot stride."""
    ids = (7).to_bytes(4, "little") + (9).to_bytes(4, "little")
    return ((Released(0, 5, 2, False, b"hello", ids),), *((),) * (STRIDES - 1))


def test_a_registration_of_a_form_2_schedule_whose_classes_have_rows_is_another_schedules(tmp_path):
    """A form-2 schedule whose classes are (P, O, rows), its tables of those rows whatever the load, is no schedule
    now: `WardenParams` refuses it. Its digest hashed the classes with their rows and is not the digest of the
    schedule of the same sizes. A registration under it is rebuilt here, from form 1's tables of the same classes
    committed under that digest, and is that schedule's own: its head has the pinned digest. The verifier's read
    under the form-2 schedule refuses it as another schedule's, by its parameters, and with this schedule's digest
    written in their place by the first domain it derives."""
    values = dataclasses.asdict(CON) | {"lengths": 2}
    earlier = tagged_sha512(PARAMETERS_TAG, canonical_json_bytes(
        {k: v for k, v in values.items() if v is not None}
        | {"kinds": sorted(KIND_NAMES), "reasons": list(CON.reasons)}))
    assert earlier.hex() == EARLIER_DIGEST and earlier not in (HELD.digest, CON.digest)
    with pytest.raises(ValueError, match="drop its third number"):
        WardenParams(**values)
    stand_in = dataclasses.replace(CON)
    stand_in.__dict__["digest"] = earlier  # form 1's tables and layouts, under the other schedule's digest
    salts = {name: [salt(name, 0, k) for k in range(count)] for name, count, _ in CR.tables_of(CON)}
    rows = CR.Rows.commit(LINK, 0, CR.tables(CON, one_release()), stand_in, signers.SESSION, salts)
    assert rows.head.digest().hex() == EARLIER_HEAD and rows.head.parameters == earlier
    issuer = Issuer(signers.signer(tmp_path), signers.SESSION)
    cert = issuer.sign(LINK, Direction.EGRESS, 0, rows.head.digest())
    doc = json.loads(json.dumps(CR.registration(rows.head, cert, stand_in)))
    key = issuer.signer.public_key()
    assert CR.check_registration(doc, key, signers.SESSION, stand_in, LINK, 0)[0] == rows.head
    with pytest.raises(ValueError, match=f"the registration's parameters is '{EARLIER_DIGEST}', and the schedule's"):
        CR.check_registration(doc, key, signers.SESSION, HELD, LINK, 0)
    with pytest.raises(ValueError, match="lengths table has domain"):
        CR.check_registration(doc | {"parameters": HELD.digest.hex()}, key, signers.SESSION, HELD, LINK, 0)
    # the same hand under the form-2 schedule: one row in the first class, and another head
    held = CR.Rows.commit(LINK, 0, CR.tables(HELD, one_release()), HELD, signers.SESSION).head
    assert [e.rows for e in held.tables[2:]] == [1, 1, 1] + [0] * 9 and held.digest().hex() != EARLIER_HEAD
    assert [e.rows for e in rows.head.tables[2:]] == [rows for _, _, rows in CLASSES for _ in range(3)]


def _refusal(opened, head, domains, table: str, rows) -> str | None:
    """The verifier's registered read of `rows` as every row of `table`, each under the network certifier's salt and
    path for its position (`warden_rows.refusal` on `connection_rows.port`): None, or why not."""
    reads = [LR.reads(opened.table(table), p) for p in range(len(rows))]
    entry = {"value": reads[0].entry["value"], "positions": list(range(len(rows))),
             "paths": [r.entry["paths"][0] for r in reads]}
    return WR.refusal(CR.port(head, table), domains[table], entry, [WR.bc(row, r.salt) for row, r in zip(rows, reads)])


def test_every_content_row_opens_its_tables_signed_root_at_its_place_and_nowhere_else(ran, signed):
    """A response row and an end row of each release, and its request row, climb their paths to their tables' roots
    in the head the verifier checked, under the verifier's own domains: at the release's row of its class, and not at
    another row, with another bit, under another table's root or domain, or under the other execution phase's head.
    The port of a content table reads its every row, held or not, as the head alone fixes."""
    _, _, _, handed = ran
    _, by_phase = signed
    for phase, (opened, _, head, domains) in by_phase.items():
        other_head = by_phase[1 - phase][2]
        for release in (r for stride in handed[phase] for r in stride):
            i, r = FILED[release.order]
            p, o, rows = CLASSES[i]
            want = (CT.request_row(release.request, B, p), CT.response_row(CT.ids_of(release.ids), o),
                    CT.end_row(release.failed, release.outputs))
            for name, row in zip(CT.names(i), want):
                table, entry, dom = opened.table(name), head.table(name), domains[name]
                assert table.rows[r] == row and dom.domain_id == entry.domain
                bc, path = WR.bc(row, table.salts[r]), table.path(r)
                assert WR.opens(dom, entry.root, r, bc, path)
                assert not WR.opens(dom, entry.root, (r + 1) % rows, bc, path)
                assert not WR.opens(dom, entry.root, r, WR.bc(bytes([row[0] ^ 1]) + row[1:], table.salts[r]), path)
                assert not WR.opens(dom, other_head.table(name).root, r, bc, path)
                twin = CT.names({2: 3, 3: 2}.get(i, (i + 1) % 2))[CT.names(i).index(name)]
                assert not WR.opens(dom, head.table(twin).root, r, bc, path)
                assert not WR.opens(domains[twin], entry.root, r, bc, path)
        for name, rows, size in CT.tables_of(CON):
            port = CR.port(head, name)
            assert (port["value"], port["reads"]) == (f"{LINK}/{name}/{phase}", list(range(rows)))
            assert _refusal(opened, head, domains, name, opened.table(name).rows) is None
        with pytest.raises(ValueError, match="not a table"):
            CR.port(head, "response.4")


def test_under_form_2_a_classes_port_is_its_held_rows_and_a_table_of_no_row_has_none(counted):
    """Under form 2 the port of a class's table is a population of the head's count: proof block j reads held row
    j, every held row opens the signed root through it at its place, a row changed in a bit doesn't, and the
    opening has no path for a row past the count. A table of no row has no port: the verifier refuses to make one,
    naming the table, and its opening has no path. The lengths and fails tables' ports are a slot stride each, as
    under form 1."""
    _, by_load = counted
    for load, (opened, _, head, domains) in by_load.items():
        for name in ("lengths", "fails"):
            assert CR.port(head, name)["reads"] == list(range(STRIDES))
        for i, count in enumerate(COUNTS[load]):
            for name in CT.names(i):
                table, entry = opened.table(name), head.table(name)
                assert len(table.rows) == entry.rows == count
                with pytest.raises(ValueError, match=f"row {count} is outside the {name} table's {count} rows"):
                    table.path(count)
                if not count:
                    with pytest.raises(ValueError, match=f"table '{name}' has no row, so no proof block reads it"):
                        CR.port(head, name)
                    continue
                port = CR.port(head, name)
                assert (port["value"], port["leaves"], port["reads"]) == (f"{LINK}/{name}/0", count, list(range(count)))
                assert (port["root"], port["domain"]) == (entry.root.hex(), domains[name].domain_id.hex())
                assert _refusal(opened, head, domains, name, table.rows) is None
                flipped = bytes([table.rows[0][0] ^ 1]) + table.rows[0][1:]
                assert "does not open the registered root" in _refusal(
                    opened, head, domains, name, [flipped, *table.rows[1:]])


# --- the response tie on the network certifier's rows ---------------------------------------------------------------


#: the forms the tie is run in at each class's size: a no-stop class's and a stop class's
FORMS = pytest.mark.parametrize("stop", (False, True), ids=("nostop", "stop"))


def committed(order: int, o: int) -> tuple[int, ...]:
    """The compute side's committed row of a release's ids: the ids the server sent, then ids the network certifier
    never released, to the class's size."""
    return (*SENT[order], *range(900, 900 + o - len(SENT[order])))


def word(order: int, o: int) -> int:
    """A stop class's stop word of the release: the stop rule ends the response after the ids the server sent where
    the connection ended, and after the row's last id where the network certifier failed it first."""
    return RT.stop_word(o if FAILED[order] else len(SENT[order]), 1)


@FORMS
def test_the_response_tie_on_the_certifiers_end_row_returns_the_response_row_it_signed(tied, stop):
    """Each release's proof block, in a stop class's form and a no-stop class's, on the compute side's committed row
    of the ids the server sent and the end row the network certifier wrote: it accepts, and the row its gates return
    is the response row the network certifier wrote, so the verifier's registered read of the response table accepts
    the blocks' rows. That holds for a connection that ended, one that failed after two ids and before any, and the
    two the client aborted. With an id the server never sent below the count the block's row is another, and the
    read refuses it. The same blocks read the tables of either of the lengths rows' forms: under form 2 a class's
    table is the releases' rows alone, a block each, and a class without one has no row to read."""
    handed, by_phase = tied
    for phase, (opened, _, head, domains) in by_phase.items():
        cases: dict[int, list] = {i: [] for i in range(len(CLASSES))}
        for release in (r for stride in handed[phase] for r in stride):
            i, r = FILED[release.order]
            o = CLASSES[i][1]
            end = opened.table(f"end.{i}").rows[r]
            assert CT.end_of(end) == (FAILED[release.order], len(SENT[release.order]))
            args = (committed(release.order, o), end, *((word(release.order, o),) if stop else ()))
            assert RT.refusals(RT.Shape(o, stop), *args) == []
            cases[i].append((r, args))
        for i, (_, o, rows) in enumerate(CLASSES):
            sh, name = RT.Shape(o, stop), f"response.{i}"
            signed_rows = list(opened.table(name).rows)
            assert len(signed_rows) == (len(cases[i]) if CT.counted(opened.params) else rows)
            if not signed_rows:
                continue
            got = RT.evaluate(sh, [args for _, args in cases[i]]) if cases[i] else []
            assert got == [RT.outputs(sh, *args) for _, args in cases[i]]
            assert got == [(RT.ACCEPT, signed_rows[r]) for r, _ in cases[i]]
            computed = list(signed_rows)
            for (r, _), (_, row) in zip(cases[i], got):
                computed[r] = row
            assert _refusal(opened, head, domains, name, computed) is None
            for r, (ids, end, *rest) in cases[i]:
                if not CT.end_of(end)[1]:
                    continue
                verdict, row = RT.outputs(sh, (ids[0] ^ 1, *ids[1:]), end, *rest)
                assert verdict == RT.ACCEPT and row != signed_rows[r]
                assert "does not open the registered root" in _refusal(
                    opened, head, domains, name, [*computed[:r], row, *computed[r + 1:]])


def test_a_connection_the_client_aborted_ends_as_its_server_does(ran):
    """A client that goes away after its request's release changes nothing the network certifier cuts, records or
    hands: it has read the client's `END` and reads the client no more, so the abort has no reason, and the cells it
    no longer writes are still cut and counted on schedule. The connection ends as its server does: in its `END`,
    an end row (0, n), or in a server's `FAIL`, (1, n). A client cut short before its release is its own failure:
    never released, so it has no lengths slot and no content row."""
    reqs, device, clients, handed = ran
    _, arrivals, answers = mixed()
    stayed, stayers, _ = run(answers, arrivals, params=CON, windows=3)
    assert dict(stayed.lengths) == handed
    assert [[r.grid for r in link] for link in records(stayed)] == [[r.grid for r in link] for link in records(device)]
    assert sorted(stayed.log, key=lambda e: e["ordinal"]) == sorted(device.log, key=lambda e: e["ordinal"])
    log = {e["ordinal"]: e for e in device.log}
    for k, at in GONE.items():
        gone, here = clients[k - 1], stayers[k - 1]
        assert log[k]["release"] is not None and log[k]["ingress_end"] < at <= log[k]["ended"]
        assert [t for t, _ in gone.writes] == [t for t, _ in here.writes if t < at] != [t for t, _ in here.writes]
    assert [(o, why) for _, o, why in reasons(device)] == [(5, "late"), (6, "late"), (7, "malformed"), (8, "late")]
    tables = {phase: CR.tables(CON, strides) for phase, strides in handed.items()}
    ended, failed = released(handed)[3], released(handed)[6]
    assert (log[4]["release"], log[8]["release"]) == (ended.order, failed.order) == (3, 6)
    assert rows_of(tables[0], *FILED[3]) == (CT.request_row(reqs[3], B, 4), CT.response_row(SENT[3], 4),
                                             CT.end_row(False, 4))
    assert rows_of(tables[1], *FILED[6]) == (CT.request_row(reqs[6], B, 16), CT.response_row(SENT[6], 4),
                                             CT.end_row(True, 1))
    # the client cut short: no release, no slot, no row
    assert (log[7]["failed"], log[7]["release"], log[7]["ingress_end"]) == ("malformed", None, None)
    assert len(released(handed)) == 7 == sum(
        CT.held(row) for made in tables.values() for i in range(len(CLASSES)) for row in made[f"request.{i}"])
    assert sum(order != 0 for made in tables.values() for row in made["lengths"]
               for order, _, _ in CR.SLOT[CR.LENGTHS].iter_unpack(row)) == 7


def test_an_execution_phase_that_is_not_handed_gets_no_content_rows_and_no_head():
    """A tick past its slack faults the third execution phase's records, so neither its releases nor the second's,
    whose responses may reach into it, are handed: nothing lays their tables or signs a head of them, and the device
    keeps their requests and ids no longer than it would have. The first's are handed as they were, and so are the
    fourth's."""
    _, arrivals, answers = mixed()
    late, _, _ = run(answers, arrivals, params=CON, windows=5, late={2 * T + 3: 0.05}, client=leaving(GONE))
    whole, _, _ = run(answers, arrivals, params=CON, windows=5, client=leaving(GONE))
    assert [r.status.complete for r in records(late)[0]] == [True, True, False, True, True]
    assert [w for w, _ in whole.lengths] == [0, 1, 2, 3] and [w for w, _ in late.lengths] == [0, 3]
    assert dict(late.lengths) == {w: strides for w, strides in whole.lengths if w in (0, 3)}
    assert len(released(dict(whole.lengths))) == 7 and len(released(dict(late.lengths))) == 4
    # what the device still holds of releases is the run's last execution phase's, which is never handed
    assert sorted(late._released) == sorted(whole._released) == [4]
