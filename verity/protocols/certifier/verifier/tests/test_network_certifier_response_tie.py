"""The response tie's proof block (`response_tie`): each of its two Boolean Definitions, a stop class's and a no-stop
class's, computes its reference's outputs bit for bit in every branch: a connection that ended in its `END` and one
that ended in a `FAIL`, on rows whose checks hold and on rows that fail each one. Where the checks hold its second
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
from verity.commitments.windows import interface as IF
from verity.definitions.registry import stop_rule
from verity.protocols.certifier.prover import response_tie as RT
from verity.protocols.certifier.prover.response_tie import ACCEPT
from verity.protocols.certifier.verifier import content_rows as CT
from verity.protocols.certifier.verifier import head_check as HC

#: every size of row the widest public size class holds
SIZES = tuple(1 << k for k in range(1, 13))
#: the sizes the circuit is held to its reference at on random rows: the smallest, each parity of log2 O, the sizes
#: either side of the split's halves, and the largest
HELD = (2, 4, 8, 16, 32, 128, 1024, 4096)
NO_COUNT = struct.pack("<I", CT.NO_COUNT)
#: the two forms: whether the class is a stop class, and its Definition
FORMS = ((True, RT.ResponseTieStop), (False, RT.ResponseTieNoStop))
#: a test's two kinds of class
CLASSES = pytest.mark.parametrize("stop", (True, False), ids=("stop", "nostop"))


def end(cls: int, at: int) -> bytes:
    return struct.pack("<HH", cls, at)


def ended(cells: int) -> bytes:
    """The end row of a connection that ended in its `END` after `cells` cells."""
    return CT.end_row(False, cells)


def failed(cells: int) -> bytes:
    """The end row of a connection that ended in a `FAIL` after `cells` cells."""
    return CT.end_row(True, cells)


def count_of(row: bytes) -> int:
    return int.from_bytes(row[:4], "little")


def decided(sh: RT.Shape, cls: int, at: int, word: int | None = None) -> int | None:
    """The two forms' rule stated apart from `refusals`: the count of the row a form accepts, or None for one it
    refuses. A stop class: the finish is not 0, n is at most O and the class is 0 or 1; then n for an `END`, and
    `at` for a `FAIL` whose `at` is at most n. A no-stop class: the class is 0 or 1 and `at` is at most O; then
    `at`."""
    if cls not in (0, 1):
        return None
    if not sh.stop:
        return at if at <= sh.O else None
    n, finish = word & (1 << RT.COUNT_BITS) - 1, word >> RT.COUNT_BITS
    if not finish or n > sh.O:
        return None
    if cls == 0:
        return n
    return at if at <= n else None


def case(sh: RT.Shape, ids, cls: int, at: int, n: int = 0, finish: int = 0) -> tuple:
    """A form's case: the egress row, the end row and, in a stop class, the stop word."""
    return (ids, end(cls, at), RT.stop_word(n, finish)) if sh.stop else (ids, end(cls, at))


def held(sh: RT.Shape, ids, fields, got) -> None:
    """Each output is the rule's (`decided`): the response row of the first ids to its count, or a refusal."""
    for (cls, at, *word), (verdict, row) in zip(fields, got, strict=True):
        count = decided(sh, cls, at, RT.stop_word(*word) if sh.stop else None)
        if count is None:
            assert verdict == 0 and row[:4] == NO_COUNT, (cls, at, word)
        else:
            assert verdict == ACCEPT and row == CT.response_row(ids[:count], sh.O), (cls, at, word)


# --- the shape -------------------------------------------------------------------------------------------------------


def test_the_shape_and_the_static_are_checked():
    """A row of O ids is 32·(O + 1) bits behind the u16, within the widest size class, so O is a power of two from 2
    to 4,096 (`content_rows.check_ids`, the one rule), and a shape's class is a stop class or a no-stop class. Each
    Definition has the one static, O, and refuses the same sizes when its signature is read."""
    assert SIZES[-1] == CT.most_ids() == 4096
    sh = RT.Shape(4096, True)
    assert (sh.k, sh.egress_bits, sh.row_bits, sh.input_bits, sh.output_bits) == (12, 131_072, 131_104, 131_120, 131_120)
    assert sh.output_bits <= UNIT_CLASSES[-1] < 16 + 32 * (2 * sh.O + 1)
    assert (RT.Shape(16, False).input_bits, RT.Shape(16, False).output_bits) == (544, 560)
    assert sh.statics() == RT.Shape(4096, False).statics() == {"O": 4096}
    refused = ((0, "power of two of at least 2"), (1, "power of two of at least 2"), (24, "power of two of at least 2"),
               (True, "power of two of at least 2"), (8192, "at most 4096 ids, not O=8192"))
    for O, match in refused:
        for stop, definition in FORMS:
            with pytest.raises(ValueError, match=match):
                RT.Shape(O, stop)
            with pytest.raises(ValueError, match=match):
                bind(definition, O=O).signature()
    for stop in (1, 0, 2, None, "stop"):
        with pytest.raises(ValueError, match=r"a stop class \(True\) or a no-stop class \(False\)"):
            RT.Shape(16, stop)


def test_a_row_whose_count_its_field_doesnt_hold_is_refused(monkeypatch):
    """Were the widest size class wider, a stop class's row would still be at most 4,096 ids, the most its stop
    word's 13 bits count, and a no-stop class's at most 32,768, the end row's 16."""
    monkeypatch.setattr(CT, "UNIT_CLASS_MAX", 1 << 22)
    assert CT.most_ids() == 65_536
    assert RT.Shape(4096, True).k == 12 and RT.Shape(32_768, False).k == 15
    with pytest.raises(ValueError, match="count field, of 13 bits, doesn't hold: at most 4096 ids"):
        RT.Shape(8192, True)
    with pytest.raises(ValueError, match="count field, of 16 bits, doesn't hold: at most 32768 ids"):
        RT.Shape(65_536, False)


def test_each_form_is_a_definition_of_its_own_with_no_optional_port():
    """A stop class's proof block and a no-stop class's are two Definitions, each with the one static O, so neither
    has a port by a static's leave: a stop class's reads the egress row, the stop word and the end row, a no-stop
    class's the egress row and the end row, and nothing else (every switch computed, none witness). The outputs of
    both are the u16 and the response row, gates of the circuit."""
    assert [(d.name, d.version, tuple(d.statics)) for _, d in FORMS] == \
        [("ResponseTieStop", 1, ("O",)), ("ResponseTieNoStop", 1, ("O",))]
    for O in (2, 16, 1024):
        for stop, definition in FORMS:
            sh = RT.Shape(O, stop)
            assert RT.block(sh) is bind(definition, O=O) and RT.block(sh).id == f"{definition.name}_v1{{O={O}}}"
            params, out = RT.block(sh).signature()
            assert [(name, t.leaves) for name, t in params] == \
                [("egress", 32 * O)] + [("stop", 16)] * stop + [("end", 32)]
            assert sum(t.leaves for _, t in params) == sh.input_bits
            assert [t.leaves for t in out.fields] == [16, 32 * (O + 1)] and out.width_set() == frozenset({1})
            assert sum(t.leaves for t in out.fields) == sh.output_bits


def test_a_statement_has_a_row_past_one_block_only_from_32_ids():
    """Flock's composer takes a statement only where one of its rows is past the bytes a last SHA-512 block holds
    (`windows.interface.LAST_BLOCK_BYTES`). The block's longest row is o1, 4·(O + 1) bytes, so the least class
    with such a row is O = 32, in either form: the module docstring's floor, which the gates' own floor of 2 is
    below."""
    for stop, _ in FORMS:
        longest = {O: max(RT.Shape(O, stop).egress_bits, RT.Shape(O, stop).row_bits, RT.STOP_BITS, RT.END_BITS) // 8
                   for O in SIZES}
        assert longest[16] == CT.response_bytes(16) == 68 and longest[32] == 132
        assert min(O for O in SIZES if longest[O] > IF.LAST_BLOCK_BYTES) == 32 > CT.MIN_SIZE


def test_the_reference_refuses_rows_of_no_form():
    """The reference reads an egress row of at most O u32 ids, an end row of four bytes, and a stop word exactly
    where its class has one."""
    stop, nostop = RT.Shape(4, True), RT.Shape(4, False)
    assert RT.stop_word(5, 2) == 5 + 2 * 8192 and RT.stop_word(8191, 7) == 0xFFFF
    for n, finish in ((8192, 1), (-1, 1), (3, 8), (3, -1), (True, 1)):
        with pytest.raises(ValueError, match="a count of 13 bits and a finish of 3"):
            RT.stop_word(n, finish)
    for ids in ([1] * 5, [1 << 32], [-1], [True]):
        with pytest.raises(ValueError, match="at most O = 4 ids, each a u32"):
            RT.outputs(stop, ids, ended(1), 1)
    with pytest.raises(ValueError, match="an end row is 4 bytes"):
        RT.outputs(nostop, [1], bytes(8))
    with pytest.raises(ValueError, match="has a stop word, a u16"):
        RT.outputs(stop, [1], ended(1))
    with pytest.raises(ValueError, match="has a stop word, a u16"):
        RT.arguments(stop, [1], ended(1), 1 << 16)
    with pytest.raises(ValueError, match="a no-stop class's row has no stop word"):
        RT.refusals(nostop, [1], failed(1), 1)
    assert RT.outputs(stop, [7], ended(1), RT.stop_word(1, 1)) == RT.outputs(stop, [7, 0, 0, 0], ended(1), RT.stop_word(1, 1))


def test_the_stop_word_is_the_stop_rules():
    """The stop word the block reads is `StopRule16_v1`'s: its count bits are the rule's, its packing is the rule's,
    and its finish is 0 exactly where the rule doesn't end the response within the row's ids. So the block accepts a
    response the rule ends, with the rule's n as its count where the connection ended in its `END` and the cells
    released where it failed before the n-th, and refuses one the rule doesn't end."""
    assert (RT.COUNT_BITS, RT.STOP_BITS) == (stop_rule.N_BITS, 16)
    assert stop_rule.FINISH[0] is None and max(stop_rule.FINISH) < 1 << RT.FINISH_BITS
    sh = RT.Shape(4, True)
    for ids, max_tokens, prompt_len, want in (([5, 7, 6, 8], 9, 10, (2, 1)), ([5, 6, 8, 9], 3, 10, (3, 2)),
                                              ([5, 6, 8, 9], 9, 98, (2, 3)), ([5, 6, 8, 7], 9, 10, (4, 1)),
                                              ([5, 6, 8, 9], 9, 10, (4, 0))):
        rule = (ids, max_tokens, [7], prompt_len, 100)
        (n, finish), word = stop_rule.reference(*rule), stop_rule.reference16(*rule)
        assert (n, finish) == want and word == RT.stop_word(n, finish)
        rows = [(ids, ended(n), word), (ids, failed(n - 1), word), (ids, failed(n + 1), word)]
        got = RT.evaluate(sh, rows)
        assert got == [RT.outputs(sh, *row) for row in rows]
        if finish:
            assert got[:2] == [(ACCEPT, CT.response_row(ids[:n], sh.O)), (ACCEPT, CT.response_row(ids[:n - 1], sh.O))]
        else:
            assert [verdict for verdict, _ in got[:2]] == [0, 0]
        assert got[2][0] == 0 and all(row[:4] == NO_COUNT for verdict, row in got if not verdict)


# --- the circuit -----------------------------------------------------------------------------------------------------


def _branches(sh: RT.Shape) -> list[tuple]:
    """A form's branches at its O, each as (class, at) and in a stop class (n, finish) after them, h half of O. A
    stop class: an `END` after some ids, O and none, and one whose `at` is not n; a `FAIL` with `at` below n, at 0
    and at n; then the refusals: `at` above n, above n = O and by a bit n has none of, a class of 2 and of 3, a
    finish of 0 and an n above O under an `END` and under a `FAIL`. A no-stop class: an `END` and a `FAIL` at 0,
    at h and at O; then `at` above O under both, by one and by a high bit, and a class of 2 and of 3."""
    O, h = sh.O, sh.O // 2
    if not sh.stop:
        return [(0, 0), (1, 0), (0, h), (1, h), (0, O), (1, O), (0, O + 1), (1, O + 1), (0, O | 0x8000),
                (1, 0x2000), (2, h), (3, h)]
    return [(0, h, h, 1), (0, O, O, 2), (0, 0, 0, 3), (0, O + 5, h, 1), (0, 0, h, 7),
            (1, h - 1, h, 1), (1, 0, O, 2), (1, h, h, 3), (1, O, O, 1), (1, 0, 0, 4),
            (1, h + 1, h, 1), (1, O + 1, O, 1), (1, h | 0x8000, O, 1), (1, 0x2000, O, 2),
            (2, h, h, 1), (3, h, h, 1), (0x100, h, h, 1), (0, h, h, 0), (1, h, h, 0),
            (0, O + 1, O + 1, 1), (1, h, O + 1, 1), (0, 0, 8191, 3), (1, 0, 8191, 3)]


def _rows(sh: RT.Shape, rng: random.Random, n: int) -> list[tuple]:
    """Egress rows with end rows and stop words, of their form and not: counts at 0, 1, O and either side of it and
    anywhere a count's field reaches, ids of 0 and of every bit among them, every finish, end rows of every class,
    and in a stop class `at` at n, either side of it and anywhere."""
    top = (1 << sh.count_bits) - 1
    out = []
    for _ in range(n):
        ids = [rng.choice((0, 1, 0xFFFF_FFFF, rng.randrange(1 << 32))) for _ in range(sh.O)]
        count = rng.choice((0, 1, sh.O - 1, sh.O, sh.O + 1, rng.randrange(sh.O + 1), rng.randrange(top + 1), top))
        cls = rng.choice((0, 1)) if rng.random() < 0.8 else rng.choice((2, 3, 0x100, 0x101, 0x8000, 0x8001, 0xFFFF))
        if sh.stop:
            at = rng.choice((0, count, max(count - 1, 0), count + 1, rng.randrange(count + 1), rng.randrange(1 << 16),
                             count | 1 << rng.randrange(16)))
            finish = rng.choice((1, 2, 3)) if rng.random() < 0.8 else rng.choice((0, 4, 7))
            out.append((ids, end(cls, at), RT.stop_word(count, finish)))
        else:
            out.append((ids, end(cls, count)))
    return out


@pytest.mark.parametrize("O", HELD)
@CLASSES
def test_the_circuit_is_its_reference(O, stop):
    """Each Boolean Definition's two outputs are the reference's on every row: the verdict, and the row it still
    computes where the verdict refuses. Every branch of the form is among the rows at each size (`_branches`), and
    each is decided as the rule states it (`decided`); then random rows. An accepted row is a response row of the
    form; a refused one is not."""
    sh = RT.Shape(O, stop)
    rng = random.Random(O * 2 + stop)
    ids = [rng.randrange(1, 1 << 32) for _ in range(O)]
    fields = _branches(sh)
    rows = [case(sh, ids, *f) for f in fields] + _rows(sh, rng, 400 if O <= 128 else 60)
    want = [RT.outputs(sh, *row) for row in rows]
    got = RT.evaluate(sh, rows)
    assert got == want
    held(sh, ids, fields, got[:len(fields)])
    took = [decided(sh, *f[:2], RT.stop_word(*f[2:]) if stop else None) is not None for f in fields]
    assert took == ([True] * 10 + [False] * 13 if stop else [True] * 6 + [False] * 6)
    for (row_ids, *_), (verdict, row) in zip(rows, want):
        if verdict == ACCEPT:
            assert CT.response_of(row, O) == tuple(row_ids[:count_of(row)])
        else:
            assert row[:4] == NO_COUNT
            with pytest.raises(ValueError, match="past its"):
                CT.response_of(row, O)
    leaves = evaluate_bits(RT.block(sh), *[np.asarray([a], dtype=np.uint8) for a in RT.arguments(sh, *rows[0])])
    assert np.asarray(leaves).reshape(-1).tolist() == RT.output_bits(sh, *rows[0])


@pytest.mark.parametrize("O", (2, 4))
@CLASSES
def test_every_branch_of_a_small_row(O, stop):
    """At O = 2 and O = 4, every end row and stop word whose fields are below 4·O: each class from 0 to 3, each
    `at` and each n on either side of O and of each other, and each finish. The circuit is the reference on each,
    and accepts exactly the rows the rule does (`decided`), each with the rule's count."""
    sh = RT.Shape(O, stop)
    ids = [0xDEAD_BEEF, 0, 0xFFFF_FFFF, 7][:O]
    span = range(4 * O)
    fields = [(cls, at, n, finish) for cls in range(4) for at in span for n in span for finish in range(8)] if stop \
        else [(cls, at) for cls in range(4) for at in span]
    rows = [case(sh, ids, *f) for f in fields]
    got = RT.evaluate(sh, rows)
    assert got == [RT.outputs(sh, *row) for row in rows]
    held(sh, ids, fields, got)
    accepted = sum(verdict == ACCEPT for verdict, _ in got)
    # a stop class: 7 finishes, and for each n to O the ENDs of every `at` and the FAILs at 0 to n; a no-stop class:
    # `at` from 0 to O under each of the two classes
    assert accepted == (7 * sum(len(span) + n + 1 for n in range(O + 1)) if stop else 2 * (O + 1))


#: each word a form reads, swept over its 2^16 values with the others at a row the form accepts: whether the class
#: is a stop class, the fields at the word's value w in a class of O ids, and the rows of the sweep the form accepts
WORDS = {
    "the stop word under an END": (True, lambda O, w: (0, 1, w & 0x1FFF, w >> 13), lambda O: 7 * (O + 1)),
    "the stop word under a FAIL after one cell": (True, lambda O, w: (1, 1, w & 0x1FFF, w >> 13), lambda O: 7 * O),
    "at under a FAIL in a stop class": (True, lambda O, w: (1, w, O, 1), lambda O: O + 1),
    "the class in a stop class": (True, lambda O, w: (w, 1, O, 1), lambda O: 2),
    "at in a no-stop class": (False, lambda O, w: (1, w), lambda O: O + 1),
    "the class in a no-stop class": (False, lambda O, w: (w, 1), lambda O: 2),
}


@pytest.mark.parametrize("O", (2, 4))
@pytest.mark.parametrize("word", WORDS)
def test_every_value_of_a_word_of_a_small_row(O, word):
    """At O = 2 and O = 4, all 2^16 values of each word a form reads (`WORDS`), so no bit of a field is outside its
    check: a finish of every value, an n and an `at` with each bit above O's, a class with each bit above its low
    one. The circuit is the reference on each, and accepts exactly the rows the rule does (`decided`)."""
    stop, fields_at, accepts = WORDS[word]
    sh = RT.Shape(O, stop)
    ids = [0xDEAD_BEEF, 0, 0xFFFF_FFFF, 7][:O]
    fields = [fields_at(O, w) for w in range(1 << 16)]
    rows = [case(sh, ids, *f) for f in fields]
    got = RT.evaluate(sh, rows)
    assert got == [RT.outputs(sh, *row) for row in rows]
    held(sh, ids, fields, got)
    assert sum(verdict == ACCEPT for verdict, _ in got) == accepts(O)


@pytest.mark.parametrize("O", (2, 16, 64, 4096))
@CLASSES
def test_an_honest_row_is_the_response_row(O, stop):
    """Rows whose checks hold, in either form and however the connection ended: a response of no id, of one, of O,
    one with an id of 0 inside its count, and an egress row that holds ids past the count, which the output
    doesn't carry. A stop class's connection ended in its `END` after the stop rule's n ids, or failed at n or
    before it; a no-stop class's ended in its `END` or failed. The circuit accepts and its row is the content
    rows' response row of the first m ids."""
    sh = RT.Shape(O, stop)
    rng = random.Random(O)
    full = [rng.randrange(1, 1 << 32) for _ in range(O)]
    zero = [full[0], 0, *full[2:]]
    cases = [(full, 0), (full, 1), (full[:1], 1), ([0] * O, 1), ([0] * O, 2), (full, O), (full, O - 1), (zero, 2),
             (zero, O), ([0] * O, O), ([], 0)]
    # how the connection ended: (class, at, n) of a response of m ids
    ways = (lambda m: (0, m, m), lambda m: (1, m, m), lambda m: (1, m, O), lambda m: (0, 0, m)) if stop else \
        (lambda m: (0, m, 0), lambda m: (1, m, 0))
    for way in ways:
        rows = [case(sh, ids, *way(m), 1 + m % 3) for ids, m in cases]
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


def test_a_stop_classs_failed_connection_carries_no_id_past_the_stop_rules():
    """A check soundness needs. The stop word is the same whatever a stop class's egress row holds past its n-th
    id, so the ids past n are held by nothing. A connection the network certifier failed after more cells than n is
    refused, and one it failed after at most n carries the first `at` ids, none of them past n. Under an `END` the
    row carries the first n whatever `at` is: there the signed row holds the count."""
    sh = RT.Shape(16, True)
    own = [0xBAD0_0000 + p for p in range(11)]
    ids, word = [1, 2, 0, 4, 5, *own], RT.stop_word(5, 1)
    inside, past = range(6), (6, 7, 15, 16, 17, 0x25, 0x1005, 0x8000, 0xFFFF)
    got = RT.evaluate(sh, [(ids, failed(at), word) for at in (*inside, *past)])
    assert got[:6] == [(ACCEPT, CT.response_row(ids[:at], 16)) for at in inside]
    assert all(verdict == 0 and row[:4] == NO_COUNT for verdict, row in got[6:])
    for at in past:
        assert RT.refusals(sh, ids, failed(at), word) == [
            f"the connection failed after {at} cells and the stop rule ends the response after 5 ids"]
    under_end = RT.evaluate(sh, [(ids, ended(at), word) for at in (0, 5, 6, 16, 0xFFFF)])
    assert under_end == [(ACCEPT, CT.response_row(ids[:5], 16))] * 5
    for verdict, row in (*got, *under_end):
        if verdict == ACCEPT:
            assert not set(own) & set(CT.response_of(row, 16))


def test_a_no_stop_class_takes_an_end_and_a_fail_alike():
    """A no-stop class's form reads its count from `at` however the connection ended, so its outputs on an end row
    of class 0 and of class 1 are the same, bit for bit: no cell, O cells, and one cell more, which both refuse."""
    for O in (2, 16, 1024):
        sh = RT.Shape(O, False)
        rng = random.Random(O)
        ids = [rng.randrange(1, 1 << 32) for _ in range(O)]
        ats = (0, 1, O // 2, O, O + 1, 2 * O, 0xFFFF)
        under_end, under_fail = (RT.evaluate(sh, [(ids, end(cls, at)) for at in ats]) for cls in (0, 1))
        assert under_end == under_fail
        assert under_end[:4] == [(ACCEPT, CT.response_row(ids[:at], O)) for at in ats[:4]]
        assert all(verdict == 0 and row[:4] == NO_COUNT for verdict, row in under_end[4:])
        assert RT.refusals(sh, ids, ended(O + 1)) == RT.refusals(sh, ids, failed(O + 1)) == [
            f"the connection ended after {O + 1} cells and the row holds {O} ids"]


#: each check of a form, failed alone on a row of 16 ids that is honest but for it: whether the class is a stop
#: class, the end row, the stop word's (n, finish) in a stop class, and the line of `refusals` that names it
REFUSALS = [
    ("the finish is 0 under an END", True, (0, 5), (5, 0), "the stop word's finish is 0"),
    ("the finish is 0 and no id", True, (0, 0), (0, 0), "the stop word's finish is 0"),
    ("the finish is 0 under a FAIL", True, (1, 3), (5, 0), "the stop word's finish is 0"),
    ("n above O under an END", True, (0, 17), (17, 1), "the stop word counts 17 ids and the row holds 16"),
    ("n far above O", True, (0, 0), (8191, 3), "the stop word counts 8191 ids and the row holds 16"),
    ("n above O by a high bit", True, (0, 16), (16 + 4096, 2), "the stop word counts 4112 ids"),
    ("n above O under a FAIL", True, (1, 3), (17, 1), "the stop word counts 17 ids and the row holds 16"),
    ("at above n", True, (1, 6), (5, 1), "failed after 6 cells and the stop rule ends the response after 5 ids"),
    ("at above n and no id", True, (1, 1), (0, 2), "failed after 1 cells and the stop rule ends the response after 0"),
    ("at above n = O", True, (1, 17), (16, 1), "failed after 17 cells and the stop rule ends the response after 16"),
    ("at far above n", True, (1, 0xFFFF), (5, 1), "failed after 65535 cells"),
    ("at above n by the bit past O's", True, (1, 3 + 32), (5, 1), "failed after 35 cells"),
    ("at above n by a bit n has none of", True, (1, 0x2000), (16, 3), "failed after 8192 cells"),
    ("a class of 2 in a stop class", True, (2, 5), (5, 1), "the end row's class is 2, not 0 or 1"),
    ("a class of 3 in a stop class", True, (3, 5), (5, 1), "the end row's class is 3"),
    ("a class of 256 in a stop class", True, (0x100, 5), (5, 1), "the end row's class is 256"),
    ("at above O under an END", False, (0, 17), None, "the connection ended after 17 cells and the row holds 16 ids"),
    ("at above O under a FAIL", False, (1, 17), None, "the connection ended after 17 cells and the row holds 16 ids"),
    ("at far above O", False, (1, 0xFFFF), None, "ended after 65535 cells"),
    ("at above O by a high bit", False, (0, 16 + 0x2000), None, "ended after 8208 cells"),
    ("a class of 2 in a no-stop class", False, (2, 5), None, "the end row's class is 2, not 0 or 1"),
    ("a class of 3 in a no-stop class", False, (3, 5), None, "the end row's class is 3"),
    ("a class of 257 in a no-stop class", False, (0x101, 5), None, "the end row's class is 257"),
]


@pytest.mark.parametrize("stop,row,word,why", [r[1:] for r in REFUSALS], ids=[r[0] for r in REFUSALS])
def test_each_failed_check_refuses(stop, row, word, why):
    """One check failed and every other holding: the circuit's verdict is 0 and its row's count word is all ones,
    so the row is no response row (`content_rows.response_of` refuses it) whatever its ids; the reference names the
    check; and the same rows with the check put right, under an `END` and under a `FAIL`, are accepted."""
    sh = RT.Shape(16, stop)
    ids = [3, 0, 9, 0xFFFF_FFFF, *range(100, 112)]
    refused = (ids, end(*row), RT.stop_word(*word)) if stop else (ids, end(*row))
    honest = [(ids, e, RT.stop_word(5, 1)) if stop else (ids, e) for e in (ended(5), failed(5))]
    (verdict, out), *right = RT.evaluate(sh, [refused, *honest])
    assert [RT.refusals(sh, *h) for h in honest] == [[], []] and right == [(ACCEPT, CT.response_row(ids[:5], 16))] * 2
    assert why in RT.refusals(sh, *refused)[0] and len(RT.refusals(sh, *refused)) == 1
    assert verdict == 0 and out[:4] == NO_COUNT == b"\xff" * 4
    assert (verdict, out) == RT.outputs(sh, *refused)
    with pytest.raises(ValueError, match="count is 4294967295, past its 16 ids"):
        CT.response_of(out, 16)


def test_a_refused_rows_ids_are_still_the_gates():
    """Where a check fails the ids of the row are what the gates compute, the egress row's below the count the form
    reads and 0 from there: `at` in a no-stop class and where a stop class's end row's class has its low bit, the
    stop word's n otherwise; all of them for a count past O. The count word alone says the row is refused."""
    sh, ids = RT.Shape(4, True), [11, 12, 13, 14]

    def refused(*kept: int) -> tuple[int, bytes]:
        return 0, struct.pack("<5I", CT.NO_COUNT, *kept, *[0] * (4 - len(kept)))

    got = RT.evaluate(sh, [(ids, ended(9), RT.stop_word(9, 1)), (ids, ended(0), RT.stop_word(2, 0)),
                           (ids, failed(3), RT.stop_word(2, 1)), (ids, failed(9), RT.stop_word(4, 2)),
                           (ids, end(2, 3), RT.stop_word(1, 1)), (ids, end(3, 3), RT.stop_word(1, 1)),
                           (ids, failed(1), RT.stop_word(3, 0)), (ids, end(2, 7), RT.stop_word(0, 0))])
    assert got == [refused(11, 12, 13, 14), refused(11, 12), refused(11, 12, 13), refused(11, 12, 13, 14),
                   refused(11), refused(11, 12, 13), refused(11), refused()]
    assert RT.refusals(sh, ids, end(2, 7), RT.stop_word(9, 0)) == [
        "the end row's class is 2, not 0 or 1", "the stop word's finish is 0: the stop rule doesn't end the response "
        "within the row's ids", "the stop word counts 9 ids and the row holds 4"]
    assert RT.refusals(sh, ids, failed(7), RT.stop_word(6, 0)) == [
        "the stop word's finish is 0: the stop rule doesn't end the response within the row's ids",
        "the stop word counts 6 ids and the row holds 4",
        "the connection failed after 7 cells and the stop rule ends the response after 6 ids"]
    nostop = RT.Shape(4, False)
    assert RT.evaluate(nostop, [(ids, failed(6)), (ids, ended(5)), (ids, end(2, 3)), (ids, end(4, 0))]) == \
        [refused(11, 12, 13, 14), refused(11, 12, 13, 14), refused(11, 12, 13), refused()]


def test_the_blocks_ands():
    """Each Definition's ANDs are its shape's formula at every size of row, and at the bindings circuit-check pins
    they are these."""
    for O in SIZES:
        for stop, _ in FORMS:
            sh = RT.Shape(O, stop)
            assert HC.ands(RT.block(sh)) == sh.ands, (O, stop)
    pinned = [RT.Shape(O, stop) for O, stop in ((16, True), (16, False), (1024, True), (2, False), (4096, True))]
    assert {RT.block(sh).id: sh.ands for sh in pinned} == ANDS
    assert RT.Shape(1024, True).ands - 33 * 1024 == 2 * 32 + 10 + 49
    assert RT.Shape(1024, False).ands - 33 * 1024 == 2 * 32 - 10 + 31


#: the ANDs at each binding circuit-check pins, by the binding's id
ANDS = {"ResponseTieStop_v1{O=16}": 589, "ResponseTieNoStop_v1{O=16}": 563, "ResponseTieStop_v1{O=1024}": 33_915,
        "ResponseTieNoStop_v1{O=2}": 98, "ResponseTieStop_v1{O=4096}": 135_357}
