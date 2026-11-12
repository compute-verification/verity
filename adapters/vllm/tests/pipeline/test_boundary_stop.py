"""A stop class's certified words (`pipeline.boundary_stop`): the stop block reads its request's ``max_tokens`` and count
of prompt ids from the network certifier's table and no word the serve wrote; a stop record registers the table as a
certified value and is accepted with the verifier's own entry of it; and each way the read fails is refused under its own
key, with the fix in its message: a block that reads another release's row, a statement that reads no row of the table,
a table whose ``max_tokens`` is not the one the block was proved at, a table of another count of rows than the class's
stop blocks, and a class with no certified words, which registers no stop record.

The table is a fixture: rows committed as the network certifier commits a table of an execution phase
(`connection_rows.Table`), listed in a head that a test-size key of the certifier's scheme signs, and read back under
that key."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Sequence

import pytest

from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.evaluation import evaluate
from verity.circuits.population import population
from verity.circuits.types import Array, Value as Bits
from verity.commitments.crypto import lms
from verity.commitments.merkle import CommitmentDomain
from verity.commitments.windows import rows as CS
from verity.definitions.registry import stop_check, stop_rule
from verity.protocols.certifier.verifier import connection_rows as CR
from verity.protocols.certifier.verifier.certificate import Certificate, Direction, Issuer, LmsKey, LmsSigner
from verity.protocols.certifier.verifier.warden import WardenParams
from verity.protocols.spot.verifier.one_stage import registration as R, verifier as V
from verity_vllm.pipeline import boundary_rows as BR, boundary_stop as BS

XOR8 = PrimitiveDefinition("BoundaryStopTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)
#: the check template: a proof block of one gate's worth
PAIR = bind(CompositeDefinition("BoundaryStopTestPair", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                                lambda B, S_, x: B.call(XOR8, x[0], x[1]), register=False))
SAMPLING = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "max_tokens": 4, "min_tokens": 0,
            "ignore_eos": False, "seed": None, "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}
DESIGN = {"law": "subset", "k": 2}
#: a serve's stop rule: two stop ids, a model of 64 ids, responses of at most 4
RULE = BS.Stop((3, 7), 64, 4)
#: the fixture's link, execution phase, class's words table, certifier session and schedule
LINK, WINDOW, TABLE, SESSION, PARAMS = "link-0", 3, "words.0", bytes(range(32)), WardenParams()
ROW = struct.Struct("<II")


def _ended(rid: str, served: tuple[int, ...], finish: str, max_tokens: int = 4, release: int | None = None) -> BR.Span:
    return BR.Span(rid, (1, 2, 5), {**SAMPLING, "max_tokens": max_tokens}, served, finish,
                   None if release is None else ("run", release))


#: a stop class's spans in release order: ended at a stop id, at ``max_tokens``, and at a ``max_tokens`` of 2
SPANS = [_ended("stop", (9, 9, 7), "stop", release=0), _ended("length", (9, 9, 9, 9), "length", release=1),
         _ended("short", (9, 9), "length", 2, release=2)]


def _word(n: int, finish: int) -> int:
    return n + (finish << stop_rule.N_BITS)


def _parsed(spans: Sequence[BR.Span]) -> list[tuple[int, int]]:
    """Each request's two words as the certifier parses them from the request it admitted: here the engine's own."""
    return [(s.sampling["max_tokens"], len(s.prompt)) for s in spans]


def _table(rows: Sequence[tuple[int, int]], name: str = TABLE) -> CR.Table:
    """The certifier's opening of a words table of ``rows``, one (``max_tokens``, count of prompt ids) a release."""
    return CR.Table(LINK, WINDOW, name, PARAMS, SESSION, tuple(ROW.pack(*r) for r in rows),
                    tuple(bytes([k + 1]) * CR.SALT_BYTES for k in range(len(rows))))


@pytest.fixture(scope="module")
def certifier(tmp_path_factory) -> Issuer:
    """The network certifier's signing step under a test-size key of its scheme: N = 64, W = 1, a tree of 2^6 leaves."""
    key = lms.keygen(lms.Params(64, 1, 6), hashlib.sha512(b"boundary-stop").digest())
    return Issuer(LmsSigner.create(key, tmp_path_factory.mktemp("certifier") / "signer-state.json"), SESSION)


def _signed(certifier: Issuer, *tables: CR.Table) -> tuple[CR.Head, Certificate]:
    """An execution phase's head listing ``tables``, and the certifier's one certificate of it."""
    head = CR.Head(LINK, WINDOW, PARAMS.digest, tuple(t.entry for t in tables))
    certificate = certifier.sign(LINK, Direction.EGRESS, WINDOW, head.digest())
    assert certificate is not None
    return head, certificate


def _read(head: CR.Head, certificate: Certificate, key: LmsKey,
          table: str = TABLE) -> tuple[dict | None, CommitmentDomain | None]:
    """The verifier's read of a signed head under the certifier's key: its own ``--registered`` port of the head's
    table ``table`` and that table's domain as it derives it from the schedule, or neither where the head lists no such
    table. A head the key didn't sign, or a table under another domain than the schedule's, is no head of the
    certifier's."""
    if certificate.digest != head.digest() or not certificate.verify(key, SESSION):
        raise ValueError("the head isn't one the network certifier signed in the session")
    if table not in [e.name for e in head.tables]:
        return None, None
    e = head.table(table)
    dom = CR.domain(PARAMS, SESSION, LINK, WINDOW, table, e.rows, e.row_bytes)
    if dom.domain_id != e.domain:
        raise ValueError("the head's table is under another domain than the schedule's")
    return CR.port(head, table), dom


def _class(spans: Sequence[BR.Span] = SPANS):
    """A stop class of ``spans`` as the serve commits it: its ``commit.json``, the verifier's layout and its boundary
    record."""
    c = BR.commit(list(spans), BR.Layout(PAIR, 0, 1, 4, RULE.outputs, len(spans), "a" * 128, stop=RULE), 0)
    pub = json.loads(json.dumps(c.public()))
    return pub, BR.layout_of(pub, PAIR, RULE), BR.record(pub, PAIR, DESIGN, stop=RULE)


def _statement(table: CR.Table, positions: Sequence[int] | None = None) -> tuple[dict, list[bytes]]:
    """A statement's registered entry of its ``words`` port and its table rows' ``b || c``: the rows of ``table`` at
    ``positions`` (default: every row in order), each with its path."""
    at = list(range(len(table.rows)) if positions is None else positions)
    return ({"value": table.value, "positions": at, "paths": [table.path(p).hex() for p in at]},
            [table.bcs[p] for p in at])


def _codes(record: dict, layout: BR.Layout, log: list[dict], reg: BS.StopRegistration, verifier=R) -> tuple[str, ...]:
    """R1-R8's codes on a stop record against the verifier's copies ``reg``, by one-stage's Python or its Lean port."""
    return verifier.check(record, layout.stop_program, layout.stop_query, DESIGN, CS.SCHEME, layout.stop_ports(), log=log,
                          **reg.copies()).codes


def _refused(call, key: str, fix: str) -> BS.StopRefused:
    with pytest.raises(BS.StopRefused, match=fix) as e:
        call()
    assert e.value.key == key and str(e.value).startswith(f"{key}: ")
    return e.value


def test_the_stop_block_reads_its_requests_two_certified_words_and_the_served_ids():
    """`egress_stop` is `StopRule16_v1` at the registered stop ids and model length over a row of two u32,
    ``max_tokens`` then the count of prompt ids, and the served ids: a stop id ends a response at it and counts, else
    ``max_tokens``, else the model's length; with no registered stop id, no id stops it. It reads no ``ignore_eos``: a
    span that sets it is in no stop class. On a class's certified rows and its `b.egress` rows the population's outputs
    are the words the serve commits as `b.stop`."""
    params, ret = RULE.template.signature()
    assert [(k, t.leaves) for k, t in params] == [(BS.WORDS, len(BS.COLUMNS)), (BS.IDS, 4)] and ret == Bits(16)
    assert BS.COLUMNS == ("max_tokens", "prompt_ids") and RULE.template.id.startswith("EgressStop_v2{")
    for ids, mt, lp in [((9, 9, 7, 0), 4, 3), ((9, 3, 7, 9), 4, 3), ((9, 9, 9, 9), 3, 1), ((7, 9, 9, 9), 4, 62),
                        ((9, 9, 9, 9), 4, 61), ((0, 0, 0, 0), 9, 0), ((9, 9, 9, 3), 0, 70)]:
        assert evaluate(RULE.template, [mt, lp], list(ids)) == (
            stop_rule.reference16(list(ids), mt, RULE.ids, lp, RULE.max_model_len),)
    none = BS.Stop((), 64, 4)
    assert evaluate(none.template, [4, 3], [9, 9, 7, 9]) == (_word(4, 2),) == (
        stop_rule.reference16([9, 9, 7, 9], 4, (stop_check.NOT_AN_ID,), 3, 64),)
    assert [RULE.word(s) for s in SPANS] == [_word(3, 1), _word(4, 2), _word(2, 2)]
    ignores = BR.Span("i", (1, 2, 5), {**SAMPLING, "ignore_eos": True}, (9, 9, 9, 9), "length")
    assert RULE.word(ignores) is None and list(BS.filed([ignores], RULE)) == [(4, 4, False)]
    _pub, layout, _rec = _class()
    rows = BR.port_rows(SPANS, layout)
    ids = [int(lo) | int(hi) << 16 for r in rows[BR.EGRESS] for lo, hi in r.reshape(-1, 2)]
    words = [x for r in _parsed(SPANS) for x in r]
    assert list(evaluate(population(RULE.template, 3), words, ids)) == rows[BR.STOP].ravel().tolist()


def test_an_honest_stop_record_registers_the_certified_table_and_is_accepted(certifier):
    """A stop class's record registers two values: ``ids``, its boundary record's `b.egress` port as an output value,
    and the class's words table under the root and domain the certifier signed, one row a release. With the verifier's
    own entry of the table, read from the signed head under the certifier's key, R1-R8 accept it, in Python and in the
    Lean port; without the entry the record registers a value the verifier doesn't expect. The verifier's ``--registered`` file has both
    parameters' ports, block r at row r of each, and the statement's read of the table's own rows is accepted. The
    record names neither `b.request` nor a word the serve wrote of the request."""
    pub, layout, rec = _class()
    table = _table(_parsed(SPANS))
    head, certificate = _signed(certifier, table)
    port, dom = _read(head, certificate, certifier.signer.public_key())
    assert (port["value"], port["leaves"], port["words"], port["word_bits"]) == (f"{LINK}/{TABLE}/{WINDOW}", 3, 4, 16)
    reg = BS.stop_registration(pub, PAIR, RULE, rec, port)
    srec = BS.stop_record(pub, PAIR, RULE, DESIGN, rec, port)
    entry = {"root": table.entry.root.hex(), "leaves": 3, "words": 4, "domain": table.entry.domain.hex()}
    assert reg.certified == {table.value: entry} == BS.certified(port, 3) and reg.table == table.value
    by_port = {r["port"]: r for r in rec["roots"]}
    assert srec["registered"] == reg.registered == [
        {"value": BS.IDS, "root": by_port[BR.EGRESS]["root"], "leaves": 3, "words": 4, "word_bits": 32,
         "domain": by_port[BR.EGRESS]["domain"]}, {"value": table.value, **entry}]
    assert reg.copies()["registered"] == {BS.IDS: 4} and reg.copies()["outputs"] == {
        BS.IDS: {"record": R.digest(rec), "port": BR.EGRESS}}
    assert BR.REQUEST not in json.dumps(srec) and srec["previous"] == R.digest(rec)
    assert _codes(srec, layout, [rec], reg) == () == _codes(srec, layout, [rec], reg, V)
    assert R.check(srec, layout.stop_program, layout.stop_query, DESIGN, CS.SCHEME, layout.stop_ports(), log=[rec],
                   **{**reg.copies(), "certified": None}).codes == ("R7-values",)
    assert sorted(reg.file["ports"]) == [BS.IDS, BS.WORDS]
    assert reg.file["ports"][BS.WORDS] == port and reg.file["ports"][BS.IDS]["reads"] == [0, 1, 2]
    assert BS.read_refusal(port, dom, *_statement(table)) is None
    other = LmsSigner.create(lms.keygen(lms.Params(64, 1, 6), hashlib.sha512(b"another").digest()),
                             certifier.signer.state.with_name("other-state.json"))
    with pytest.raises(ValueError, match="isn't one the network certifier signed"):
        _read(head, certificate, other.public_key())


def test_a_read_naming_another_row_is_refused(certifier):
    """A statement whose stop block reads another release's row than its own is refused under `READ_ROW`, though every
    row it holds opens the certified root at the position it names: two blocks' rows swapped, and one block on a row
    twice."""
    table = _table(_parsed(SPANS))
    port, dom = _read(*_signed(certifier, table), certifier.signer.public_key())
    for positions, block, got in (((1, 0, 2), 0, 1), ((0, 1, 1), 2, 1)):
        e = BS.read_refusal(port, dom, *_statement(table, positions))
        assert e.key == BS.READ_ROW == "stop-read-row"
        assert f"stop block {block} reads row {got} of {table.value}, and its own is row {block}" in str(e)
        assert "Stage each block on the row of its own release" in str(e)
    entry, bcs = _statement(table)
    e = BS.read_refusal(port, dom, entry, bcs, refs=[0, 1, 2], units=[0, 2, 1])
    assert e.key == BS.READ_ROW and "stop block 2 reads row 1" in str(e)


def test_a_read_naming_no_row_is_refused(certifier):
    """A statement that ties its ``words`` rows to no row of the certified table is refused under `READ_NONE`: one
    that commits them afresh with no registered entry, one whose entry names another value, one with a row that has no
    path, and one with an instance that reads no row of its table."""
    table = _table(_parsed(SPANS))
    port, dom = _read(*_signed(certifier, table), certifier.signer.public_key())
    entry, bcs = _statement(table)
    for got, why in ((BS.read_refusal(port, dom, None, bcs), "rows of no registered value"),
                     (BS.read_refusal(port, dom, {**entry, "value": "b.request"}, bcs), "rows of another value, 'b.request'"),
                     (BS.read_refusal(port, dom, {**entry, "paths": entry["paths"][:2]}, bcs), "3 positions and 2 paths"),
                     (BS.read_refusal(port, dom, entry, bcs, refs=[0, 1, 3]), "instance 2 reads no row")):
        assert got.key == BS.READ_NONE == "stop-read-none" and why in str(got)
        assert f"as a registered read of {table.value}" in str(got)


def test_a_table_whose_max_tokens_differs_from_the_one_the_block_was_proved_at_is_refused(certifier):
    """The engine served the second request to 4 ids, and the request the certifier admitted says ``max_tokens`` 3.
    The stop block at the certified row ends the response at 3, which is not the word the serve committed, so the
    serve's block is one proved at another row than the certifier's. A stop record that registers that row's table is
    refused by R7 (``R7-certified``): its root is not the one the verifier read under the certifier's key. A record
    that registers the certifier's root, with a statement that holds the engine's row in the place of the certifier's
    and the certifier's path beside it, is refused under `READ_ROOT`: the row does not open the certified root."""
    pub, layout, rec = _class()
    engine = _parsed(SPANS)
    admitted = [engine[0], (3, engine[1][1]), engine[2]]
    assert evaluate(RULE.template, list(admitted[1]), [9, 9, 9, 9]) == (_word(3, 2),) != (RULE.word(SPANS[1]),)
    proved, signed = _table(engine), _table(admitted)
    port, dom = _read(*_signed(certifier, signed), certifier.signer.public_key())
    reg = BS.stop_registration(pub, PAIR, RULE, rec, port)
    theirs = BS.stop_record(pub, PAIR, RULE, DESIGN, rec, CR.port(CR.Head(LINK, WINDOW, PARAMS.digest, (proved.entry,)),
                                                                   TABLE))
    assert proved.entry.domain == signed.entry.domain and proved.entry.root != signed.entry.root
    assert _codes(theirs, layout, [rec], reg) == ("R7-certified",) == _codes(theirs, layout, [rec], reg, V)
    assert _codes(BS.stop_record(pub, PAIR, RULE, DESIGN, rec, port), layout, [rec], reg) == ()
    entry, bcs = _statement(signed)
    e = BS.read_refusal(port, dom, entry, [bcs[0], proved.bcs[1], bcs[2]])
    assert e.key == BS.READ_ROOT == "stop-read-root"
    assert f"table row 1 does not open {signed.value}'s certified root at position 1" in str(e)
    assert "Prove the block on the certifier's own row" in str(e)
    assert BS.read_refusal(port, dom, *_statement(signed)) is None


def test_a_class_with_no_certified_length_gets_no_stop_class(certifier):
    """A class whose requests have no certified words registers no stop record: where the certifier's head lists no
    words table for it, as a schedule in the text form's, `stop_registration` and `stop_record` refuse under
    `UNCERTIFIED` and say to file its spans without a stop block. A table of another count of rows than the class's
    stop blocks is refused under `COUNT`, and a port that is no words table's under `PORT`: rows of another width,
    another row format, or reads that start at another row."""
    pub, _layout, rec = _class()
    other = _table([(4, 3)] * 3, "ids.0")
    key = certifier.signer.public_key()
    port, dom = _read(*_signed(certifier, other), key)
    assert port is None and dom is None
    for call in (lambda: BS.stop_registration(pub, PAIR, RULE, rec, port),
                 lambda: BS.stop_record(pub, PAIR, RULE, DESIGN, rec, port), lambda: BS.certified(None, 3)):
        _refused(call, "stop-uncertified", "File the class's spans by size class, without a stop block")
    assert BS.UNCERTIFIED == "stop-uncertified"
    short, _ = _read(*_signed(certifier, _table(_parsed(SPANS[:2]))), key)
    e = _refused(lambda: BS.stop_registration(pub, PAIR, RULE, rec, short), "stop-certified-count",
                 "File in the stop class exactly the requests the network certifier released in the class")
    assert BS.COUNT == "stop-certified-count" and "3 stop blocks and its certified table 2 rows" in str(e)
    good, _ = _read(*_signed(certifier, _table(_parsed(SPANS))), key)
    wide = CR.Table(LINK, WINDOW, TABLE, PARAMS, SESSION, tuple(struct.pack("<III", 4, 0, 3) for _ in SPANS),
                    tuple(bytes([k + 9]) * CR.SALT_BYTES for k in range(3)))
    three, _ = _read(*_signed(certifier, wide), key)
    for bad in (three, {**good, "row_schema": "hm96-sha512/row/v1"}, {**good, "reads": [1, 2, 0]},
                {**good, "reads_from": 1}, {k: v for k, v in good.items() if k != "root"}):
        _refused(lambda bad=bad: BS.stop_registration(pub, PAIR, RULE, rec, bad), "stop-certified-port",
                 "Hand the verifier's own port of the class's words table")
    assert BS.PORT == "stop-certified-port"
    as_u32 = BS.certified({**good, "words": 2, "word_bits": 32}, 3)[good["value"]]
    assert as_u32 == {**BS.certified(good, 3)[good["value"]], "words": 2, "word_bits": 32}
