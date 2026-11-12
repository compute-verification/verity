"""The network certifier's request words (`WardenParams.request_words`, `content_rows`' "The words row" and "A
request's words", `connection_rows`), driven tick by tick on `test_network_certifier_warden`'s harness. A schedule
with them has a fourth table a class, `words.i`: row r is the `max_tokens`, the `ignore_eos` and the count of prompt
ids of the r-th release filed in the class, as the network certifier reads them from the request's bytes, whatever
its server answers. A request without `max_tokens` at its place, or with an `ignore_eos` key, is never released, so
it has no row. The head lists the table after the class's other three at the class's count, with the zero root
where the class has no release, the verifier's read holds a registration to that, and the table's port is by its
name. A schedule without them has the digest, the tables and the heads it had."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json

import pytest

import signers
from verity.commitments.identity import canonical_json_bytes
from verity.protocols.certifier.verifier import connection_rows as CR
from verity.protocols.certifier.verifier import content_rows as CT
from verity.protocols.certifier.verifier.certificate import Direction, Issuer
from verity.protocols.certifier.verifier.tests.test_network_certifier_content_tables import (
    CON, GONE, HELD, LINK, _refusal, leaving, mixed, one_release, read, released, salt)
from verity.protocols.certifier.verifier.tests.test_network_certifier_lengths import (B, LEN, answer, id_req,
                                                                                      no_id_requests)
from verity.protocols.certifier.verifier.tests.test_network_certifier_warden import reasons, records, run
from verity.protocols.certifier.verifier.warden import Released, WardenParams, cells

#: the content tables tests' form-2 schedule under a response cap, and the same with request words
CAPPED = dataclasses.replace(HELD, max_tokens_cap=8)
WORDED = dataclasses.replace(CAPPED, request_words=True)
R = WORDED.phase_releases
#: each class's four tables, in the head's order
NAMES = tuple(CT.names_of(WORDED, i) for i in range(len(WORDED.content)))
#: the form-2 schedule's digest, and `registrations` of the content tables tests' traffic under it, as each was
#: before the request words were a parameter
HELD_DIGEST = ("fe09d8179188559f4744e18efbcc9371ecdf7bbaae4e2cc1fbe26e4df92299a4"
               "a378a5a47d2b1de609f79f8c81b54362d5dbb2e19b488c143839714b5f972069")
HELD_REGISTERED = ("57251fa45c593f1c7bfdd05bee62a08a9caef4d78dc969c29609baf7e7ae7549"
                   "c09710f4ccfc6b5ba00bfd0caef9f801ffb6f4208058e31ed472ae945ca50b10")

#: each released request's `max_tokens` and its prompt's count of ids, by its place in the release order: a prompt
#: of one cell, of two and of three, a `max_tokens` of 0 and one at the cap
ASKED = ((8, 3), (1, 40), (8, 100), (0, 3), (5, 16), (7, 2))
#: what the server sends on each, and on each in a second run of the same requests
SENT = ((11, 22, 33), (9,), tuple(range(8)), (), (5,), (1, 2))
OTHER = ((1,), (), (3, 4), (), (), (6, 7, 8))
#: the class the network certifier's rule files each in, and its row there, under either run's answers
FILED = ((0, 0), (2, 0), (2, 1), (0, 1), (1, 0), (0, 0))
#: the releases each execution phase of the run files in each class
COUNTS = {0: (2, 0, 2, 0), 1: (1, 1, 0, 0), 2: (0, 0, 0, 0)}


def traffic(sent=SENT):
    """Six released requests over two execution phases, each with its own `max_tokens` and prompt, and between them
    two clients that are never released: one whose request has an `ignore_eos` key, and one whose request has no
    `max_tokens`."""
    reqs = [id_req(i + 1, n, max_tokens=m) for i, (m, n) in enumerate(ASKED)]
    never = no_id_requests()
    arrivals = {0: [cells(reqs[0], B), cells(reqs[1], B)], 8: [cells(reqs[2], B)], 10: [cells(reqs[3], B)],
                17: [cells(reqs[4], B)], 18: [cells(never["an ignore_eos key"][0], B)], 24: [cells(reqs[5], B)],
                25: [cells(never["no max_tokens"][0], B)]}
    answers = [answer(sent[0]), answer(sent[1]), answer(sent[2]), answer(sent[3]), answer(sent[4]), [],
               answer(sent[5]), []]
    return reqs, arrivals, answers


@pytest.fixture(scope="module")
def ran():
    """`traffic` through the network certifier under the schedule with request words: its requests, the device and
    its handed releases by execution phase."""
    reqs, arrivals, answers = traffic()
    device, _, _ = run(answers, arrivals, params=WORDED, windows=3)
    return reqs, device, dict(device.lengths)


@pytest.fixture(scope="module")
def signed(ran, tmp_path_factory):
    """Each handed execution phase's tables committed and their one head signed under the schedule with request
    words, and the registration as the verifier reads it: the issuer, and by execution phase what `read` gives."""
    _, _, handed = ran
    issuer = Issuer(signers.signer(tmp_path_factory.mktemp("signer")), signers.SESSION)
    return issuer, {phase: read(issuer, phase, strides, WORDED) for phase, strides in handed.items()}


def registrations(params: WardenParams, handed, directory) -> str:
    """A hash of every handed execution phase's registration under `params`, the public record the verifier reads,
    with the signer's key, its coins and every salt fixed."""
    counter = itertools.count()
    issuer = Issuer(signers.signer(directory), signers.SESSION,
                    coins=lambda n: hashlib.sha512(b"coins%d" % next(counter)).digest()[:n])
    h = hashlib.sha512()
    for phase, strides in handed:
        tables = CR.tables(params, strides)
        salts = {name: [salt(name, phase, k) for k in range(len(rows))] for name, rows in tables.items()}
        rows = CR.Rows.commit(LINK, phase, tables, params, signers.SESSION, salts)
        cert = issuer.sign(LINK, Direction.EGRESS, phase, rows.head.digest())
        h.update(canonical_json_bytes(CR.registration(rows.head, cert, params)))
    return h.hexdigest()


# --- the parameter -------------------------------------------------------------------------------------------------


def test_request_words_are_a_parameter_of_a_capped_form_2_schedule_with_content_rows():
    """A schedule has request words only where its parameters say so: true, or unset, in a schedule with content
    rows under the lengths rows' form 2 and a response cap, and any other is refused where it is built, naming what
    the schedule needs. They enter the digest of a schedule that has them, and read back from the parameters as a
    deployment writes them. Such a schedule lists a fourth table a class, `words.i`, after the class's other three:
    at most R rows of 12 bytes, with its class's layout."""
    assert WardenParams().request_words is None and HELD.request_words is None and WORDED.request_words is True
    assert len({HELD.digest, CAPPED.digest, WORDED.digest}) == 3
    assert dataclasses.replace(WORDED, request_words=None) == CAPPED
    assert WardenParams(**json.loads(json.dumps(dataclasses.asdict(WORDED)))) == WORDED
    for bad in (False, 1, 0, "true", (True,)):
        with pytest.raises(ValueError, match="request_words is true, for a schedule with request words, or unset"):
            dataclasses.replace(WORDED, request_words=bad)
    needs = "has content rows under the lengths rows' form 2 and a response cap"
    for other in (HELD, dataclasses.replace(CON, max_tokens_cap=8), dataclasses.replace(LEN, max_tokens_cap=8),
                  dataclasses.replace(LEN, lengths=2, max_tokens_cap=8), WardenParams()):
        with pytest.raises(ValueError, match=needs):
            dataclasses.replace(other, request_words=True)
    assert [t for t in CR.tables_of(WORDED) if not t[0].startswith("words.")] == list(CR.tables_of(CAPPED))
    assert [name for name, _, _ in CR.tables_of(WORDED)] == ["lengths", "fails", *(n for four in NAMES for n in four)]
    assert NAMES[0] == ("request.0", "response.0", "end.0", "words.0") and CT.names_of(HELD, 0) == CT.names(0)
    for i in range(len(WORDED.content)):
        assert CT.words_name(i) == f"words.{i}" == NAMES[i][3]
        assert [t for t in CR.tables_of(WORDED) if t[0] == f"words.{i}"] == [(f"words.{i}", R, 12)]
        assert CT.layout(WORDED, f"words.{i}") == CT.layout(WORDED, f"end.{i}") == CT.layout(CAPPED, f"end.{i}")
    with pytest.raises(ValueError, match="no content table 'words.0'"):
        CT.layout(CAPPED, "words.0")
    # the head counts its tables in a u16: four a class after the lengths and fails tables
    most = CT.most_classes(WORDED)
    assert 2 + 4 * most <= 0xFFFF < 2 + 4 * (most + 1) and CT.most_classes(CAPPED) == CT.MOST_CLASSES > most
    with pytest.raises(ValueError, match=f"content is 1 to {most} classes"):
        dataclasses.replace(WORDED, content=((4, 4),) * (most + 1))


def test_a_schedule_without_request_words_has_the_digest_the_tables_and_the_heads_it_had(tmp_path):
    """The request words are hashed only by a schedule that has them, so a form-2 schedule without them has the
    parameters digest it had, lists three tables a class, and its run's registrations are the ones it left, byte
    for byte. Its tables take any released request, as they did: nothing reads its words."""
    assert HELD.digest.hex() == HELD_DIGEST and all(len(CT.names_of(p, 0)) == 3 for p in (HELD, CAPPED, CON))
    _, arrivals, answers = mixed()
    device, _, _ = run(answers, arrivals, params=HELD, windows=3, client=leaving(GONE))
    assert registrations(HELD, device.lengths, tmp_path) == HELD_REGISTERED
    for params in (HELD, CAPPED):
        laid = CR.tables(params, one_release())
        assert list(laid) == [name for name, _, _ in CR.tables_of(params)] and not any("words" in n for n in laid)
        assert CT.request_of(laid["request.0"][0], B, 4) == b"hello"


# --- the device ----------------------------------------------------------------------------------------------------


def test_a_releases_words_are_its_requests_whatever_its_server_answers(ran):
    """Each release's row of its class's words table is its request's `max_tokens`, 0 for `ignore_eos` and its
    prompt's count of ids, as the test built the request: at the release's row of its class, the row of its request
    in the request table. The same requests answered otherwise, with other ids, fewer or none, file as they did and
    have the same words tables, byte for byte, where their response tables differ. The schedule's other tables are
    the ones the same hand lays without request words, and the device cuts, records and hands what it does without
    them."""
    reqs, device, handed = ran
    assert sorted(handed) == [0, 1, 2]
    got = released(handed)
    assert [r.request for r in got] == reqs and len(got) == len(ASKED) == device.requests_released
    tables = {phase: CR.tables(WORDED, strides) for phase, strides in handed.items()}
    for phase, strides in handed.items():
        assert [len(tables[phase][f"words.{i}"]) for i in range(len(NAMES))] == list(COUNTS[phase])
        base = CR.tables(CAPPED, strides)
        assert {name: rows for name, rows in tables[phase].items() if name in base} == base
        for release in (r for stride in strides for r in stride):
            i, r = FILED[release.order]
            max_tokens, prompt = ASKED[release.order]
            row = tables[phase][f"words.{i}"][r]
            assert row == max_tokens.to_bytes(4, "little") + bytes(4) + prompt.to_bytes(4, "little")
            assert CT.words_of(row) == (max_tokens, 0, prompt) == CT.request_words(release.request, B)
            assert CT.request_of(tables[phase][f"request.{i}"][r], B, WORDED.content[i][0]) == release.request
            assert prompt <= WORDED.content[i][0] and max_tokens <= WORDED.max_tokens_cap
    _, arrivals, answers = traffic(OTHER)
    other, _, _ = run(answers, arrivals, params=WORDED, windows=3)
    again = {phase: CR.tables(WORDED, strides) for phase, strides in other.lengths}
    for phase in handed:
        for i, four in enumerate(NAMES):
            assert again[phase][four[3]] == tables[phase][four[3]] and again[phase][four[0]] == tables[phase][four[0]]
            assert (again[phase][four[1]] == tables[phase][four[1]]) == (not COUNTS[phase][i])
    _, arrivals, answers = traffic()
    plain, _, _ = run(answers, arrivals, params=CAPPED, windows=3)
    assert plain.lengths == device.lengths and plain.log == device.log
    assert [[r.grid for r in link] for link in records(plain)] == [[r.grid for r in link] for link in records(device)]


def test_a_request_without_max_tokens_at_its_place_or_with_an_ignore_eos_key_is_never_released(ran):
    """The two clients whose requests the rule doesn't read: the network certifier cuts no cell of either, releases
    neither and fails each as its client's, so neither has a place in the release order, a lengths slot or a row of
    any table. Under the response cap the request without `max_tokens` fails at its first cell's read of it (cap),
    and the one with an `ignore_eos` key as no canonical id request (malformed). Every request of the lengths
    tests' that is no id request is refused by the rule by name, as the device never releases it."""
    _, device, handed = ran
    log = {e["ordinal"]: e for e in device.log}
    assert [(o, why) for _, o, why in reasons(device)] == [(6, "malformed"), (8, "cap")]
    for ordinal in (6, 8):
        assert (log[ordinal]["release"], log[ordinal]["ingress_end"], log[ordinal]["ingress_cells"]) == (None, None, 0)
        assert log[ordinal]["failed_at"] == 0
    assert sorted(e["release"] for e in device.log if e["release"] is not None) == list(range(len(ASKED)))
    rows = [row for strides in handed.values() for name, table in CR.tables(WORDED, strides).items()
            if name.startswith("words.") for row in table]
    assert len(rows) == len(ASKED) and all(CT.words_of(row)[1] == 0 for row in rows)
    for name, (bad, _) in no_id_requests().items():
        with pytest.raises(ValueError, match="is not that cell of a canonical id request"):
            CT.request_words(bad, B)
        with pytest.raises(ValueError, match="release 0, filed in class 2, has a request without words: cell"):
            CT.tables(WORDED, [Released(0, len(bad), 0, False, bad, b"")], [2])
        assert len(CT.tables(CAPPED, [Released(0, len(bad), 0, False, bad, b"")], [2])["request.2"]) == 1, name


# --- the head ------------------------------------------------------------------------------------------------------


def test_the_head_lists_each_classes_words_table_at_the_classes_count(signed):
    """The head of a schedule with request words lists, after each class's request, response and end tables, its
    words table: at the class's count of releases, one count for the four, in rows of 12 bytes under the domain the
    schedule gives the table at that count. A class of no release has the table of no row, with the zero root. The
    registration says the same of it, in six 16-bit words a row and its class's layout, and reads back."""
    _, by_phase = signed
    assert sorted(by_phase) == [0, 1, 2]
    for phase, (opened, doc, head, domains) in by_phase.items():
        assert [e.name for e in head.tables] == ["lengths", "fails", *(n for four in NAMES for n in four)]
        said = {t["name"]: t for t in doc["tables"]}
        for i, count in enumerate(COUNTS[phase]):
            assert [head.table(name).rows for name in NAMES[i]] == [count] * 4
            e, t = head.table(f"words.{i}"), said[f"words.{i}"]
            assert e.row_bytes == CT.WORDS_BYTES == 12 and len(opened.table(e.name).rows) == count
            assert e.domain == CR.domain(WORDED, signers.SESSION, LINK, phase, e.name, count, 12).domain_id
            assert e.domain == domains[e.name].domain_id
            assert (e.root == CR.NO_ROWS_ROOT) == (count == 0)
            assert (t["value"], t["rows"], t["words"]) == (f"{LINK}/words.{i}/{phase}", count, 6)
            assert t["layout"] == said[f"end.{i}"]["layout"] and t["layout"]["class"] == i
            assert (t["domain"], t["root"]) == (e.domain.hex(), e.root.hex())
    # two execution phases that file the same count in a class have other roots there: each row has a salt of its own
    assert by_phase[0][2].table("words.1").root == by_phase[2][2].table("words.1").root == CR.NO_ROWS_ROOT
    assert by_phase[0][2].table("words.0").rows == 2 and by_phase[1][2].table("words.0").rows == 1


def test_a_words_tables_port_is_its_held_rows_by_its_name_and_a_table_of_no_row_has_none(signed):
    """The port of a class's words table is `connection_rows.port` by the table's name: a population of the head's
    count, proof block j reading row j under the table's root and domain. Every held row opens the signed root
    through it at its place; a row with another `max_tokens`, another count of prompt ids or a set `ignore_eos`
    doesn't, nor do two rows in each other's places. A table of no row has no port, and its opening no path."""
    _, by_phase = signed
    for phase, (opened, _, head, domains) in by_phase.items():
        for i, count in enumerate(COUNTS[phase]):
            name = f"words.{i}"
            table, entry = opened.table(name), head.table(name)
            with pytest.raises(ValueError, match=f"row {count} is outside the {name} table's {count} rows"):
                table.path(count)
            if not count:
                with pytest.raises(ValueError, match=f"table '{name}' has no row, so no proof block reads it"):
                    CR.port(head, name)
                continue
            port = CR.port(head, name)
            assert (port["value"], port["leaves"], port["reads"]) == (f"{LINK}/{name}/{phase}", count, list(range(count)))
            assert (port["root"], port["domain"]) == (entry.root.hex(), domains[name].domain_id.hex())
            assert _refusal(opened, head, domains, name, table.rows) is None
            max_tokens, eos, prompt = CT.words_of(table.rows[0])
            for other in ((max_tokens ^ 1, eos, prompt), (max_tokens, eos, prompt + 1), (max_tokens, 1, prompt)):
                assert "does not open the registered root" in _refusal(
                    opened, head, domains, name, [CT.words_row(*other), *table.rows[1:]])
            if count > 1:
                assert "does not open the registered root" in _refusal(opened, head, domains, name, table.rows[::-1])
    with pytest.raises(ValueError, match="not a table 'words.4'"):
        CR.port(by_phase[0][2], "words.4")


def recounted(doc: dict, phase: int, rows: dict) -> dict:
    """A registration of execution phase `phase` under the schedule with request words, with the rows of each table
    `rows` names changed and each one's domain the one the schedule gives at those rows."""
    sizes = {name: size for name, _, size in CR.tables_of(WORDED)}
    return {**doc, "tables": [
        t if t["name"] not in rows else {**t, "rows": rows[t["name"]], "domain": CR.domain(
            WORDED, signers.SESSION, LINK, phase, t["name"], rows[t["name"]], sizes[t["name"]]).domain_id.hex()}
        for t in doc["tables"]]}


def test_the_verifiers_read_holds_a_words_table_to_its_classes_count_and_its_schedule(ran, signed):
    """The verifier's read takes a words table at its class's count and no other: a registration that gives it
    another count than the class's end table, inside the bound and under the domain of its own count, is refused
    naming the two tables, one past R naming the bound, and one of no row only with the zero root. A registration
    without the words tables is a schedule without request words': the read under the schedule with them refuses
    it as another schedule's, by its parameters and, with this schedule's digest written in their place, by the
    tables it lists; and that schedule's read refuses this one's."""
    issuer, by_phase = signed
    _, doc, _, _ = by_phase[0]

    def check(changed, params=WORDED) -> None:
        CR.check_registration(changed, issuer.signer.public_key(), signers.SESSION, params, LINK, 0)

    check(doc)
    for name, rows, like in (("words.0", 1, "end.0 table 2"), ("words.0", 0, "end.0 table 2"),
                             ("words.1", 1, "end.1 table 0"), ("words.2", R, "end.2 table 2")):
        with pytest.raises(ValueError, match=f"{name} table has rows {rows}, and its class's {like}: a class's "
                                             f"tables hold a row a release filed in it"):
            check(recounted(doc, 0, {name: rows}))
    with pytest.raises(ValueError, match=f"words.0 table has rows {R + 1}, and a class's tables hold a row a release "
                                         f"filed in it, at most the {R}"):
        check(recounted(doc, 0, {"words.0": R + 1}))
    with pytest.raises(ValueError, match="response.0 table has rows 2, and its class's request.0 table 3"):
        check(recounted(doc, 0, {"request.0": 3}))
    with pytest.raises(ValueError, match="doesn't sign its head"):
        check(recounted(doc, 0, dict.fromkeys(NAMES[0], 1)))
    held_root = next(t["root"] for t in doc["tables"] if t["name"] == "words.0")
    for root in (held_root, "00" * 63 + "01"):
        changed = {**doc, "tables": [{**t, "root": root} if t["name"] == "words.1" else t for t in doc["tables"]]}
        with pytest.raises(ValueError, match="words.1 table has no row and the root"):
            check(changed)
    # the same hand under the schedule without request words, signed by the same key: each read refuses the other's
    _, bare, _, _ = read(issuer, 0, ran[2][0], CAPPED)
    assert [t["name"] for t in bare["tables"]] == [t["name"] for t in doc["tables"] if "words" not in t["name"]]
    for other, params in ((bare, WORDED), (doc, CAPPED)):
        with pytest.raises(ValueError, match="the registration's parameters is '[0-9a-f]{128}', and the schedule's"):
            check(other, params)
    with pytest.raises(ValueError, match="the registration lists the schedule's 18 tables"):
        check(bare | {"parameters": WORDED.digest.hex()})
    with pytest.raises(ValueError, match="the registration lists the schedule's 14 tables"):
        check(doc | {"parameters": CAPPED.digest.hex()}, CAPPED)


def test_a_hand_whose_request_has_no_words_is_refused_and_a_words_table_is_committed_at_its_classes_count(ran):
    """The tables of a schedule with request words are laid only from releases whose requests the rule reads: a
    hand with a request that is no canonical id request is refused, naming the release and its class, and none of
    it is laid, where the same hand under the schedule without request words is laid as it was. So is a hand with
    a request whose `max_tokens` is past the response cap, or with more ids than its request's `max_tokens`, which
    the device never releases; at the cap and at its `max_tokens` it is laid. The commitment takes a class's words
    table only at the class's count, in rows of 12 bytes."""
    _, _, handed = ran
    with pytest.raises(ValueError, match="release 0, filed in class 0, has a request without words: cell 0 of the "
                                         "request's 1 is not that cell of a canonical id request"):
        CR.tables(WORDED, one_release())
    assert [len(rows) for rows in CR.tables(CAPPED, one_release()).values()][2:5] == [1, 1, 1]

    def hand(max_tokens: int, ids: int) -> tuple:
        request = id_req(0, 3, max_tokens=max_tokens)
        return ((Released(0, len(request), ids, False, request, bytes(4 * ids)),), *((),) * (len(handed[0]) - 1))

    cap = WORDED.max_tokens_cap
    for max_tokens, ids in ((cap + 1, 2), (2, 3), (0, 1)):
        with pytest.raises(ValueError, match=f"release 0, filed in class 0, has a max_tokens of {max_tokens} and {ids} "
                                             f"ids under a response cap of {cap}: the network certifier releases no"):
            CR.tables(WORDED, hand(max_tokens, ids))
        assert len(CR.tables(CAPPED, hand(max_tokens, ids))["request.0"]) == 1
    for max_tokens, ids in ((cap, 4), (2, 2), (0, 0)):
        assert [CT.words_of(row) for row in CR.tables(WORDED, hand(max_tokens, ids))["words.0"]] == [(max_tokens, 0, 3)]
    full = CR.tables(WORDED, handed[0])
    CR.Rows.commit(LINK, 0, full, WORDED, signers.SESSION)
    for changed, why in (({"words.0": full["words.0"][:-1]}, "words.0 table has 1 rows, and the end.0 table 2"),
                         ({"words.1": [bytes(12)]}, "words.1 table has 1 rows, and the end.1 table 0"),
                         ({"words.0": [bytes(11)] * 2}, f"words.0 table is at most {R} rows of 12 bytes"),
                         ({"words.0": [full["words.0"][0]] * (R + 1)}, f"words.0 table is at most {R} rows of 12")):
        with pytest.raises(ValueError, match=why):
            CR.Rows.commit(LINK, 0, full | changed, WORDED, signers.SESSION)
    without = {name: rows for name, rows in full.items() if not name.startswith("words.")}
    with pytest.raises(ValueError, match="the schedule's tables are .*'end.0', 'words.0', 'request.1'"):
        CR.Rows.commit(LINK, 0, without, WORDED, signers.SESSION)
    with pytest.raises(ValueError, match="the schedule's tables are .*'end.0', 'request.1'"):
        CR.Rows.commit(LINK, 0, full, CAPPED, signers.SESSION)
