"""A Boolean Program's instance sequence through its word link, and where a Program's checks are computed
(`verity_vllm.pipeline.build_instances`): row k is the word Program's node k with ``bits``, the Boolean Definition Boolean node k
Calls; a Boolean Definition whose word is another Definition than the word node's is refused by name; the file's header says
``ir: "boolean"`` and keeps the word Program's parameters; and `check_rows` gives the row and leaf of each of the last returned
leaves, in the order they are returned."""
from __future__ import annotations

import gzip
import json

import pytest
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition
from verity.circuits.program import Program
from verity.circuits.refs import Coll, concat
from verity.circuits.types import Array, Value

from verity_vllm.pipeline import build_instances as BI

V16 = Value(16)


def _prim(name: str, word=None) -> PrimitiveDefinition:
    d = PrimitiveDefinition(name, 1, (("a", V16),), V16, lambda a: -a & 0xFFFF, register=False)
    if word is not None:
        d.word = word
    return d


def _comp(name: str, params, ret, body) -> CompositeDefinition:
    return CompositeDefinition(name, 1, (), lambda S: (tuple(params), ret), body, register=False)


def _pair(name: str, neg, word=None) -> CompositeDefinition:
    d = _comp(name, [("a", V16)], Array(2, V16), lambda B, S, a: Coll(Array(2, V16), concat([a.refs, B.call(neg, a).refs])))
    if word is not None:
        d.word = word
    return d


NEG, INC = _prim("TWNeg16"), _prim("TWInc16")
PAIR = _pair("TWPair", NEG)
NEG_BITS, INC_BITS = _prim("TWNeg16Bits", NEG), _prim("TWInc16Bits", INC)
PAIR_BITS = _pair("TWPairBits", NEG_BITS, PAIR)


def _program(name: str, param: str, calls, returned=lambda x, out: [out[3], out[2]], leaves: int = 3) -> Program:
    """Four rows: three Calls in a chain from the parameter's first leaf, then a Call on the third row's second leaf.  It returns
    the fourth row's leaf and then the third row's two, or what `returned` lists."""
    def body(B, S, x):
        out = [B.call(calls[0], x[0])]
        out.append(B.call(calls[1], out[0]))
        out.append(B.call(calls[2], out[1]))
        out.append(B.call(calls[3], out[2][1]))
        return Coll(Array(leaves, V16), concat([c.refs for c in returned(x, out)]))
    return Program(_comp(name, [(param, Array(4, V16))], Array(leaves, V16), body))


def _linked(calls=(NEG_BITS, INC_BITS, PAIR_BITS, NEG_BITS), words=(NEG, INC, PAIR, NEG)) -> tuple[Program, Program]:
    word, boolean = _program("TWRoot", "x", words), _program("TWRootBits", "x_bits", calls)
    boolean.word = word
    return boolean, word


WORD_ROWS = [{"i": 0, "spec": "TWNeg16_v1", "form": "primitive", "name": None, "args": [[["p", 0, 0, 1]]]},
             {"i": 1, "spec": "TWInc16_v1", "form": "primitive", "name": None, "args": [[["n", 0, 0, 1]]]},
             {"i": 2, "spec": "TWPair_v1", "form": "call", "name": None, "args": [[["n", 1, 0, 1]]]},
             {"i": 3, "spec": "TWNeg16_v1", "form": "primitive", "name": None, "args": [[["n", 2, 1, 2]]]}]
BITS = ["TWNeg16Bits_v1", "TWInc16Bits_v1", "TWPairBits_v1", "TWNeg16Bits_v1"]
ROWS = [{"i": r["i"], "spec": r["spec"], "bits": b, "form": r["form"], "name": r["name"], "args": r["args"]} for r, b in zip(WORD_ROWS, BITS)]


def _file(tmp_path) -> str:
    return gzip.decompress((tmp_path / "instances.json.gz").read_bytes()).decode()


def test_a_boolean_programs_row_is_its_word_programs_with_the_boolean_definition_as_bits():
    boolean, word = _linked()
    assert list(BI.instance_rows(word)) == WORD_ROWS
    for rows in (list(BI._instance_rows(boolean)), BI._instance_sequence(boolean), list(BI._word_rows(boolean, word))):
        assert rows == ROWS and [list(r) for r in rows] == [["i", "spec", "bits", "form", "name", "args"]] * 4
    assert list(BI._instance_rows(word)) == BI._instance_sequence(word) == WORD_ROWS


def test_a_definition_that_is_its_own_word_is_linked_without_naming_one():
    """The word node's own Definition, or another object of its name, on the Boolean side: it names no word (`word_of` raises
    for it) and its row's ``bits`` is its ``spec``."""
    for same in (INC, _prim("TWInc16")):
        boolean, word = _linked(calls=(NEG_BITS, same, PAIR_BITS, NEG_BITS))
        assert list(BI._word_rows(boolean, word)) == [ROWS[0], {**ROWS[1], "bits": "TWInc16_v1"}, *ROWS[2:]]


def test_a_boolean_definitions_word_is_looked_up_once_however_many_rows_call_it():
    asked: list[dict] = []

    def word(bindings: dict) -> PrimitiveDefinition:
        asked.append(bindings)
        return NEG

    boolean, linked = _linked(calls=(_prim("TWNeg16Bits", word), INC_BITS, PAIR_BITS, _prim("TWNeg16Bits", word)))
    assert list(BI._word_rows(boolean, linked)) == ROWS and asked == [{}]


@pytest.mark.parametrize("calls,words,refusal", [
    ((NEG_BITS, NEG_BITS, PAIR_BITS, NEG_BITS), (NEG, INC, PAIR, NEG),
     "row 1: TWNeg16Bits_v1 names the word Definition TWNeg16_v1, and the word Program's node 1 Calls TWInc16_v1"),
    ((NEG_BITS, INC_BITS, PAIR_BITS, NEG_BITS), (NEG, INC, _pair("TWPairs", NEG), NEG),
     "row 2: TWPairBits_v1 names the word Definition TWPair_v1, and the word Program's node 2 Calls TWPairs_v1"),
    ((INC_BITS, INC_BITS, PAIR_BITS, NEG_BITS), (NEG, INC, PAIR, NEG),
     "row 0: TWInc16Bits_v1 names the word Definition TWInc16_v1, and the word Program's node 0 Calls TWNeg16_v1"),
])
def test_a_boolean_definition_whose_word_is_not_the_word_nodes_is_refused_by_row_and_name(calls, words, refusal):
    boolean, word = _linked(calls=calls, words=words)
    rows = BI._word_rows(boolean, word)
    k = int(refusal.split(":")[0].removeprefix("row "))
    assert [next(rows) for _ in range(k)] == ROWS[:k]
    with pytest.raises(ValueError) as e:
        next(rows)
    assert str(e.value) == refusal


def test_a_boolean_program_of_another_length_than_its_word_program_is_refused():
    boolean, word = _linked()
    short = Program(_comp("TWShort", [("x", Array(4, V16))], V16, lambda B, S, x: B.call(INC, B.call(NEG, x[0]))))
    for b, w in ((boolean, short), (short, word)):
        rows = BI._word_rows(b, w)
        with pytest.raises(ValueError, match=r"^zip\(\) argument 2 is (shorter|longer) than argument 1$"):
            list(rows)


def test_the_builds_writer_writes_a_boolean_program_by_its_word_programs_rows_and_parameters(tmp_path):
    boolean, word = _linked()
    BI.write_instances(str(tmp_path), boolean, "d" * 128, None, {"supplied": {"x": "registered"}})
    head = {"schema": "cb-a/instance-sequence/v2", "digest": "d" * 128, "correspondence_digest": None,
            "params": [["x", "Array<4,Value<16>>"]], "ir": "boolean", "supplied": {"x": "registered"}}
    assert _file(tmp_path) == json.dumps({**head, "rows": ROWS})
    BI.write_instances(str(tmp_path), boolean, "d" * 128, "c" * 64, {}, names={0: "a", 2: "c\u00e9", 3: None}, params=[["y", "Array<2,Value<16>>"]])
    named = [{**r, "name": n} for r, n in zip(ROWS, ["a", None, "c\u00e9", None])]
    assert _file(tmp_path) == json.dumps({"schema": "cb-a/instance-sequence/v2", "digest": "d" * 128, "correspondence_digest": "c" * 64,
                                          "params": [["y", "Array<2,Value<16>>"]], "ir": "boolean", "rows": named})
    BI.write_instances(str(tmp_path), word, "d" * 128, None, {})
    assert json.loads(_file(tmp_path)) == {"schema": "cb-a/instance-sequence/v2", "digest": "d" * 128, "correspondence_digest": None,
                                           "params": [["x", "Array<4,Value<16>>"]], "rows": WORD_ROWS}


@pytest.mark.parametrize("ir", ["word", "boolean"])
def test_the_checks_are_written_last_in_the_header_with_the_row_and_leaf_of_each(tmp_path, ir):
    boolean, word = _linked()
    program, rows, said = (word, WORD_ROWS, {}) if ir == "word" else (boolean, ROWS, {"ir": "boolean"})
    BI.write_instances(str(tmp_path), program, "d" * 128, None, {"supplied": {}}, checks={"of": ["finite", "in_range"], "rule": "r1"})
    head = {"schema": "cb-a/instance-sequence/v2", "digest": "d" * 128, "correspondence_digest": None, "params": [["x", "Array<4,Value<16>>"]],
            **said, "supplied": {}, "checks": {"of": ["finite", "in_range"], "rule": "r1", "at": [[2, 0, 1], [2, 1, 2]]}}
    assert _file(tmp_path) == json.dumps({**head, "rows": rows})
    none: tuple[dict | None, ...] = (None, {})
    for checks in none:
        BI.write_instances(str(tmp_path), program, "d" * 128, None, {"supplied": {}}, checks=checks)
        assert _file(tmp_path) == json.dumps({**{k: v for k, v in head.items() if k != "checks"}, "rows": rows})


def test_check_rows_are_the_last_returned_leaves_in_the_order_returned():
    body = _linked()[1].fn.body                    # returns row 3's leaf, then row 2's two
    assert [BI.check_rows(body, n) for n in range(4)] == [[], [[2, 1, 2]], [[2, 0, 1], [2, 1, 2]], [[3, 0, 1], [2, 0, 1], [2, 1, 2]]]
    whole = _program("TWWhole", "x", (NEG, INC, PAIR, NEG), lambda x, out: [out[0], out[2], out[2]], leaves=5).fn.body
    assert BI.check_rows(whole, 4) == [[2, 0, 1], [2, 1, 2], [2, 0, 1], [2, 1, 2]]
    assert BI.check_rows(whole, 5) == [[0, 0, 1], [2, 0, 1], [2, 1, 2], [2, 0, 1], [2, 1, 2]]


def test_check_rows_refuse_more_checks_than_leaves_and_a_leaf_no_row_computes():
    body = _linked()[1].fn.body
    with pytest.raises(ValueError) as e:
        BI.check_rows(body, 4)
    assert str(e.value) == "the Program's last 4 returned leaves are not 4 checks computed by its rows: [('n', 3, 0), ('n', 2, 0), ('n', 2, 1)]"
    read = _program("TWRead", "x", (NEG, INC, PAIR, NEG), lambda x, out: [out[3], x[3], out[2]], leaves=4).fn.body
    assert BI.check_rows(read, 2) == [[2, 0, 1], [2, 1, 2]]
    with pytest.raises(ValueError) as e:
        BI.check_rows(read, 4)
    assert str(e.value) == "the Program's last 4 returned leaves are not 4 checks computed by its rows: [('n', 3, 0), ('p', 0, 3), ('n', 2, 0), ('n', 2, 1)]"
    five = _program("TWFive", "x", (NEG, INC, PAIR, NEG), lambda x, out: [out[0], out[2], out[2]], leaves=5).fn.body
    with pytest.raises(ValueError) as e:
        BI.check_rows(five, 6)                     # the refusal names the first four of the leaves it found
    assert str(e.value) == "the Program's last 6 returned leaves are not 6 checks computed by its rows: [('n', 0, 0), ('n', 2, 0), ('n', 2, 1), ('n', 2, 0)]"
