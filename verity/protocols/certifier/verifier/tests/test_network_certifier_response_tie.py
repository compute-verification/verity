"""The response tie's proof block (`response_tie`): the Boolean Definition computes its reference's outputs bit for
bit, in both forms, on rows whose checks hold and on rows that fail each one. Where the checks hold its second
output is the content rows' response row of the egress row's first ids (`content_rows.response_row`), whatever the
egress row holds past the count; where one fails the verdict refuses and the row's count word is all ones, which no
response row has."""

from __future__ import annotations

import random
import struct

import numpy as np
import pytest

from verity.circuits.defs import bind
from verity.circuits.evaluation.bits import evaluate_bits
from verity.circuits.partition_v2 import UNIT_CLASSES
from verity.protocols.certifier.prover import response_tie as RT
from verity.protocols.certifier.prover.response_tie import ACCEPT
from verity.protocols.certifier.verifier import content_rows as CT
from verity.protocols.certifier.verifier import head_check as HC

#: every size of row the widest public size class holds
SIZES = tuple(1 << k for k in range(1, 13))
#: the sizes the circuit is held to its reference at on random rows: the smallest, each parity of log2 O, the sizes
#: either side of the split's halves, and the largest
HELD = (2, 4, 8, 16, 32, 128, 1024, 4096)
ENDED, NO_COUNT = CT.end_row(False, 0), struct.pack("<I", CT.NO_COUNT)


def end(cls: int, at: int) -> bytes:
    return struct.pack("<HH", cls, at)


def count_of(row: bytes) -> int:
    return int.from_bytes(row[:4], "little")


# --- the shape -------------------------------------------------------------------------------------------------------


def test_the_shape_and_the_statics_are_checked():
    """A row of O ids is 32·(O + 1) bits behind the u16, within the widest size class, so O is a power of two from 2
    to 4,096 (`content_rows.check_ids`, the one rule); STOP is the form, 1 or 0. The Definition refuses the same
    statics when its signature is read."""
    assert SIZES[-1] == CT.most_ids() == 4096
    sh = RT.Shape(4096, 1)
    assert (sh.k, sh.egress_bits, sh.row_bits, sh.input_bits, sh.output_bits) == (12, 131_072, 131_104, 131_120, 131_120)
    assert sh.output_bits <= UNIT_CLASSES[-1] < 16 + 32 * (2 * sh.O + 1)
    assert (RT.Shape(16, 0).input_bits, RT.Shape(16, 0).output_bits) == (544, 560)
    refused = ((dict(O=0), "power of two of at least 2"), (dict(O=1), "power of two of at least 2"),
               (dict(O=24), "power of two of at least 2"), (dict(O=True), "power of two of at least 2"),
               (dict(O=8192), "at most 4096 ids, not O=8192"), (dict(STOP=2), "STOP is 1"),
               (dict(STOP=True), "STOP is 1"), (dict(STOP=-1), "STOP is 1"))
    for kw, match in refused:
        statics = RT.Shape(16, 1).statics() | kw
        with pytest.raises(ValueError, match=match):
            RT.Shape(**statics)
        with pytest.raises(ValueError, match=match):
            bind(RT.ResponseTie, **statics).signature()


def test_a_row_whose_count_its_field_doesnt_hold_is_refused(monkeypatch):
    """Were the widest size class wider, a stop class's row would still be at most 4,096 ids, the most its stop
    word's 13 bits count, and a failed class's at most 32,768, the end row's 16."""
    monkeypatch.setattr(CT, "UNIT_CLASS_MAX", 1 << 22)
    assert CT.most_ids() == 65_536
    assert RT.Shape(4096, 1).k == 12 and RT.Shape(32_768, 0).k == 15
    with pytest.raises(ValueError, match="count field, of 13 bits, doesn't hold: at most 4096 ids"):
        RT.Shape(8192, 1)
    with pytest.raises(ValueError, match="count field, of 16 bits, doesn't hold: at most 32768 ids"):
        RT.Shape(65_536, 0)


def test_the_block_reads_the_egress_row_the_end_row_and_in_a_stop_class_the_stop_word():
    """The proof block's inputs are the egress row, the stop word where its class has one, and the end row, and
    nothing else (every switch computed, none witness); its outputs are the u16 and the response row, gates of the
    circuit."""
    for O in (2, 16, 1024):
        for stop in (1, 0):
            sh = RT.Shape(O, stop)
            params, out = RT.block(sh).signature()
            assert [(name, t.leaves) for name, t in params] == \
                [("egress", 32 * O)] + [("stop", 16)] * stop + [("end", 32)]
            assert sum(t.leaves for _, t in params) == sh.input_bits
            assert [t.leaves for t in out.fields] == [16, 32 * (O + 1)] and out.width_set() == frozenset({1})
            assert sum(t.leaves for t in out.fields) == sh.output_bits


def test_the_reference_refuses_rows_of_no_form():
    """The reference reads an egress row of at most O u32 ids, an end row of four bytes, and a stop word exactly
    where its class has one."""
    stop, failed = RT.Shape(4, 1), RT.Shape(4, 0)
    assert RT.stop_word(5, 2) == 5 + 2 * 8192 and RT.stop_word(8191, 7) == 0xFFFF
    for n, finish in ((8192, 1), (-1, 1), (3, 8), (3, -1), (True, 1)):
        with pytest.raises(ValueError, match="a count of 13 bits and a finish of 3"):
            RT.stop_word(n, finish)
    for ids in ([1] * 5, [1 << 32], [-1], [True]):
        with pytest.raises(ValueError, match="at most O = 4 ids, each a u32"):
            RT.outputs(stop, ids, ENDED, 1)
    with pytest.raises(ValueError, match="an end row is 4 bytes"):
        RT.outputs(failed, [1], bytes(8))
    with pytest.raises(ValueError, match="has a stop word, a u16"):
        RT.outputs(stop, [1], ENDED)
    with pytest.raises(ValueError, match="has a stop word, a u16"):
        RT.arguments(stop, [1], ENDED, 1 << 16)
    with pytest.raises(ValueError, match="a failed class's row has no stop word"):
        RT.refusals(failed, [1], end(1, 1), 1)
    assert RT.outputs(stop, [7], ENDED, RT.stop_word(1, 1)) == RT.outputs(stop, [7, 0, 0, 0], ENDED, RT.stop_word(1, 1))


# --- the circuit -----------------------------------------------------------------------------------------------------


def _rows(sh: RT.Shape, rng: random.Random, n: int) -> list[tuple]:
    """Egress rows with end rows and stop words, of their form and not: counts at 0, 1, O and either side of it and
    anywhere the count's field reaches, ids of 0 and of every bit among them, every finish, and end rows of every
    class with `at` at, below and above O."""
    top = (1 << sh.count_bits) - 1
    out = []
    for _ in range(n):
        ids = [rng.choice((0, 1, 0xFFFF_FFFF, rng.randrange(1 << 32))) for _ in range(sh.O)]
        count = rng.choice((0, 1, sh.O - 1, sh.O, sh.O + 1, rng.randrange(sh.O + 1), rng.randrange(top + 1), top))
        if sh.STOP:
            row = ENDED if rng.random() < 0.7 else \
                end(rng.choice((0, 1, 2, 0x100, 0x8000)), rng.choice((0, 1, count, 0xFFFF)))
            finish = rng.choice((1, 2, 3)) if rng.random() < 0.7 else rng.choice((0, 4, 7))
            out.append((ids, row, RT.stop_word(count, finish)))
        else:
            cls = 1 if rng.random() < 0.7 else rng.choice((0, 2, 3, 0x101, 0x8001, 0x100, 0xFFFF))
            out.append((ids, end(cls, count)))
    return out


@pytest.mark.parametrize("O", HELD)
@pytest.mark.parametrize("stop", (1, 0), ids=("stop", "failed"))
def test_the_circuit_is_its_reference(O, stop):
    """The Boolean Definition's two outputs are the reference's on every row: the verdict, and the row it still
    computes where the verdict refuses, with rows it accepts and rows it refuses both among them. An accepted row
    is a response row of the form; a refused one is not."""
    sh = RT.Shape(O, stop)
    rows = _rows(sh, random.Random(O * 2 + stop), 400 if O <= 128 else 60)
    want = [RT.outputs(sh, *row) for row in rows]
    assert RT.evaluate(sh, rows) == want
    assert {verdict for verdict, _ in want} == {0, ACCEPT}
    for (ids, *_), (verdict, row) in zip(rows, want):
        if verdict == ACCEPT:
            assert CT.response_of(row, O) == tuple(ids[:count_of(row)])
        else:
            assert row[:4] == NO_COUNT
            with pytest.raises(ValueError, match="past its"):
                CT.response_of(row, O)
    leaves = evaluate_bits(RT.block(sh), *[np.asarray([a], dtype=np.uint8) for a in RT.arguments(sh, *rows[0])])
    assert np.asarray(leaves).reshape(-1).tolist() == RT.output_bits(sh, *rows[0])


@pytest.mark.parametrize("stop", (1, 0), ids=("stop", "failed"))
def test_every_count_of_a_small_row(stop):
    """At O = 4, every word the count is read from: all 2^16 stop words under an end row of (0, 0), and all 2^16
    values of `at` under a class of 1. The circuit is the reference on each, accepts exactly the counts 0 to 4
    (with, in a stop class, a finish that is not 0), and returns those counts."""
    sh = RT.Shape(4, stop)
    ids = [0xDEAD_BEEF, 0, 0xFFFF_FFFF, 7]
    rows = [(ids, ENDED, word) if stop else (ids, end(1, word)) for word in range(1 << 16)]
    got = RT.evaluate(sh, rows)
    assert got == [RT.outputs(sh, *row) for row in rows]
    accepted = {word: count_of(row) for word, (verdict, row) in enumerate(got) if verdict == ACCEPT}
    assert accepted == ({n + (finish << 13): n for n in range(5) for finish in range(1, 8)} if stop else
                        {n: n for n in range(5)})
    assert all(row[:4] == NO_COUNT for word, (_, row) in enumerate(got) if word not in accepted)


@pytest.mark.parametrize("O", (2, 16, 64, 4096))
@pytest.mark.parametrize("stop", (1, 0), ids=("stop", "failed"))
def test_an_honest_row_is_the_response_row(O, stop):
    """Rows whose checks hold, in either form: a response of no id, of one, of O, one with an id of 0 inside its
    count, and an egress row that holds ids past the count, which the output doesn't carry. The circuit accepts and
    its row is the content rows' response row of the first m ids."""
    sh = RT.Shape(O, stop)
    rng = random.Random(O)
    full = [rng.randrange(1, 1 << 32) for _ in range(O)]
    zero = [full[0], 0, *full[2:]]
    cases = [(full, 0), (full, 1), (full[:1], 1), ([0] * O, 1), ([0] * O, 2), (full, O), (full, O - 1), (zero, 2),
             (zero, O), ([0] * O, O), ([], 0)]
    rows = [(ids, ENDED, RT.stop_word(m, 1 + m % 3)) if stop else (ids, CT.end_row(True, m)) for ids, m in cases]
    got = RT.evaluate(sh, rows)
    for (ids, m), row, (verdict, out) in zip(cases, rows, got):
        assert RT.refusals(sh, *row) == []
        assert verdict == ACCEPT and out == CT.response_row(ids[:m], O)
        assert CT.response_of(out, O) == tuple(ids[:m])
        assert (verdict, out) == RT.outputs(sh, *row)
    # the ids past the count are in the egress row and not in the output
    assert any(full[1:]) and got[1][1] == got[2][1] == CT.response_row(full[:1], O)
    # an id of 0 inside the count is an id: the rows of one and of two zero ids differ in the count alone
    assert got[3][1][4:] == got[4][1][4:] == bytes(4 * O) and (count_of(got[3][1]), count_of(got[4][1])) == (1, 2)


#: each check of a form, failed alone on a row of 16 ids that is honest but for it: the form, the end row, the stop
#: word's (n, finish) in a stop class, and the line of `refusals` that names it
REFUSALS = [
    ("the finish is 0", 1, (0, 0), (5, 0), "the stop word's finish is 0"),
    ("the finish is 0 and no id", 1, (0, 0), (0, 0), "the stop word's finish is 0"),
    ("n above O", 1, (0, 0), (17, 1), "the stop word counts 17 ids and the row holds 16"),
    ("n far above O", 1, (0, 0), (8191, 3), "the stop word counts 8191 ids and the row holds 16"),
    ("n above O by a high bit", 1, (0, 0), (16 + 4096, 2), "the stop word counts 4112 ids"),
    ("an end row of (0, at)", 1, (0, 5), (5, 1), "the end row has no failure and at = 5, not 0"),
    ("an end row of (0, at) at a high bit", 1, (0, 0x8000), (5, 1), "no failure and at = 32768"),
    ("class 1 in a stop class", 1, (1, 5), (5, 1), "the end row has a failure, and a stop class's connection"),
    ("class 1 and no cell in a stop class", 1, (1, 0), (5, 2), "the end row has a failure"),
    ("a class of 2 in a stop class", 1, (2, 0), (5, 1), "the end row's class is 2, not 0 or 1"),
    ("a class of 256 in a stop class", 1, (0x100, 0), (5, 1), "the end row's class is 256"),
    ("class 0 in a failed class", 0, (0, 0), None, "the end row has no failure, and a failed class's connection"),
    ("class 0 with cells in a failed class", 0, (0, 5), None, "the end row has no failure"),
    ("at above O", 0, (1, 17), None, "the connection failed after 17 cells and the row holds 16 ids"),
    ("at far above O", 0, (1, 0xFFFF), None, "failed after 65535 cells"),
    ("at above O by a high bit", 0, (1, 16 + 0x2000), None, "failed after 8208 cells"),
    ("a class of 2 in a failed class", 0, (2, 5), None, "the end row's class is 2, not 0 or 1"),
    ("a class of 3 in a failed class", 0, (3, 5), None, "the end row's class is 3"),
    ("a class of 257 in a failed class", 0, (0x101, 5), None, "the end row's class is 257"),
]


@pytest.mark.parametrize("stop,row,word,why", [r[1:] for r in REFUSALS], ids=[r[0] for r in REFUSALS])
def test_each_failed_check_refuses(stop, row, word, why):
    """One check failed and every other holding: the circuit's verdict is 0 and its row's count word is all ones,
    so the row is no response row (`content_rows.response_of` refuses it) whatever its ids; the reference names the
    check; and the same rows with the check put right are accepted."""
    sh = RT.Shape(16, stop)
    ids = [3, 0, 9, 0xFFFF_FFFF, *range(100, 112)]
    case = (ids, end(*row), RT.stop_word(*word)) if stop else (ids, end(*row))
    honest = (ids, ENDED, RT.stop_word(5, 1)) if stop else (ids, end(1, 5))
    (verdict, out), (was, before) = RT.evaluate(sh, [case, honest])
    assert RT.refusals(sh, *honest) == [] and (was, before) == (ACCEPT, CT.response_row(ids[:5], 16))
    assert why in RT.refusals(sh, *case)[0] and len(RT.refusals(sh, *case)) == 1
    assert verdict == 0 and out[:4] == NO_COUNT == b"\xff" * 4
    assert (verdict, out) == RT.outputs(sh, *case)
    with pytest.raises(ValueError, match="count is 4294967295, past its 16 ids"):
        CT.response_of(out, 16)


def test_a_refused_rows_ids_are_still_the_gates():
    """Where a check fails the ids of the row are what the gates compute, the egress row's below the count the form
    reads and 0 from there: all of them for a count past O, the first n where the finish is 0. The count word alone
    says the row is refused."""
    sh, ids = RT.Shape(4, 1), [11, 12, 13, 14]
    got = RT.evaluate(sh, [(ids, ENDED, RT.stop_word(9, 1)), (ids, ENDED, RT.stop_word(2, 0)),
                           (ids, end(1, 0), RT.stop_word(4, 2)), (ids, end(2, 7), RT.stop_word(0, 0))])
    assert got == [(0, struct.pack("<5I", CT.NO_COUNT, 11, 12, 13, 14)), (0, struct.pack("<5I", CT.NO_COUNT, 11, 12, 0, 0)),
                   (0, struct.pack("<5I", CT.NO_COUNT, 11, 12, 13, 14)), (0, struct.pack("<5I", CT.NO_COUNT, 0, 0, 0, 0))]
    assert RT.refusals(sh, ids, end(2, 7), RT.stop_word(9, 0)) == [
        "the end row's class is 2, not 0 or 1", "the stop word's finish is 0: the stop rule doesn't end the response "
        "within the row's ids", "the stop word counts 9 ids and the row holds 4"]
    failed = RT.Shape(4, 0)
    assert RT.evaluate(failed, [(ids, end(1, 6)), (ids, end(0, 3)), (ids, end(4, 0))]) == \
        [(0, struct.pack("<5I", CT.NO_COUNT, 11, 12, 13, 14)), (0, struct.pack("<5I", CT.NO_COUNT, 11, 12, 13, 0)),
         (0, struct.pack("<5I", CT.NO_COUNT, 0, 0, 0, 0))]


def test_the_blocks_ands():
    """The proof block's ANDs are its shape's formula at every size of row and in both forms, and at the bindings
    circuit-check pins they are these."""
    memo: dict = {}
    for O in SIZES:
        for stop in (1, 0):
            sh = RT.Shape(O, stop)
            assert HC.ands(RT.block(sh), memo) == sh.ands, (O, stop)
    assert {shape: RT.Shape(*shape).ands for shape in ANDS} == ANDS
    assert RT.Shape(1024, 1).ands - 33 * 1024 == 2 * 32 - 10 + 48


#: the ANDs of `ResponseTie_v1{O, STOP}` at each binding circuit-check pins, by (O, STOP)
ANDS = {(16, 1): 580, (16, 0): 564, (1024, 1): 33_894, (2, 0): 99, (4096, 1): 135_331}
