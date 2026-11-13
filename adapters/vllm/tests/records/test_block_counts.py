"""`records.block_counts`: a day's inference-block judgment reads back to what `dump` wrote, its table's and its classes' counts
in their primitives' order, and is refused by field when it is no JSON object or another format, holds a day, a digest, a
count, a class order, an outcome or a reason no writer writes, or says a gates or fresh outcome that isn't its classes' or a
verdict that isn't its checks'."""
from __future__ import annotations

import pytest

from verity_vllm.records.block_counts import RULES, SCHEMA, BlockClass, BlockJudgment
from verity_vllm.records.fields import RecordError

A, B = "0a" * 64, "0b" * 64
J = BlockJudgment(day="2026-01-02", declaration="0d" * 64, c=100, k=8, n=3, table={"and": 1, "xor": 1},
                  classes=(BlockClass(definition=A, slots=4, steps=2, gates=100, unpriced={}, fresh=8),
                           BlockClass(definition=B, slots=8, steps=2, gates=90, unpriced={}, fresh=6)),
                  checks=dict.fromkeys(RULES, "accept"), verdict="accept", reason=None)
REFUSED = {"checks": {**J.checks, "count": "refuse"}, "verdict": "refuse"}


def _class(i: int, **change) -> dict:
    return {"classes": [{**c, **(change if j == i else {})} for j, c in enumerate(J.dump()["classes"])]}


def test_a_judgment_reads_back_to_what_dump_wrote() -> None:
    doc = J.dump()
    assert doc["format"] == SCHEMA
    assert BlockJudgment.of(doc, "j.json") == J
    refused = BlockJudgment.of({**doc, **REFUSED, "reason": "more than n blocks"}, "j.json")
    assert (refused.verdict, refused.reason, refused.checks["count"]) == ("refuse", "more than n blocks", "refuse")


def test_a_judgments_counts_are_held_in_their_primitives_order_however_they_were_written() -> None:
    doc = {**J.dump(), "table": {"xor": 1, "mux": 4, "and": 1}, **_class(1, unpriced={"sel": 2, "cmp": 1}),
           "checks": {**J.checks, "gates": "undecided"}, "verdict": "refuse", "reason": "a block's gates are a lower bound"}
    got = BlockJudgment.of(doc, "j.json")
    assert list(got.table.items()) == [("and", 1), ("mux", 4), ("xor", 1)]
    assert list(got.classes[1].unpriced.items()) == [("cmp", 1), ("sel", 2)]
    assert [list(got.dump()["table"]), list(got.dump()["classes"][1]["unpriced"])] == [["and", "mux", "xor"], ["cmp", "sel"]]


def test_a_judgment_that_is_no_json_object_is_refused() -> None:
    with pytest.raises(RecordError, match="'format' cannot be read: the judgment is a list, not a JSON object"):
        BlockJudgment.of([J.dump()], "j.json")


@pytest.mark.parametrize(("change", "field"), [
    ({"format": "verity-vllm/inference-blocks/v1"}, "format"),
    ({"day": "2 Jan 2026"}, "day"),
    ({"declaration": "0d" * 32}, "declaration"),
    (_class(0, definition="0a" * 32), "classes[0].definition"),
    ({"classes": J.dump()["classes"][::-1]}, "classes"),                                 # not in their definitions' order
    ({"classes": J.dump()["classes"][:1] * 2}, "classes"),                               # two of one definition
    ({"checks": {"gates": "accept", "fresh": "accept"}}, "checks"),                      # no outcome for a rule
    ({"checks": {**J.checks, "count": "maybe"}}, "checks"),                              # an outcome that is none of the three
    ({"c": -1}, "c"),
    ({"n": True}, "n"),
    (_class(1, slots=-1), "classes[1].slots"),
    ({"table": {"and": -1}}, "table"),
    ({"table": {"and": True}}, "table"),
    ({"table": {"and": 1.5}}, "table"),
    (_class(0, unpriced={"mux": -1}), "classes[0].unpriced"),
    ({"reason": "a sentence on an accept"}, "reason"),
    ({**REFUSED, "reason": None}, "reason"),                                             # a refusal says why
    ({**REFUSED, "reason": 7}, "reason"),
    ({"c": 99}, "checks.gates"),                                                         # a class of 100 gates, over c
    ({"k": 7}, "checks.fresh"),                                                          # a class of 8 fresh bits, over k
    (_class(1, unpriced={"mux": 1}), "checks.gates"),                                    # gates that are a lower bound: undecided
    ({"verdict": "refuse", "reason": "a refusal no check gives"}, "verdict"),
    ({"checks": {**J.checks, "count": "undecided"}}, "verdict"),                         # an accept over a check that isn't one
])
def test_a_judgment_that_isnt_one_is_refused_by_field(change: dict, field: str) -> None:
    with pytest.raises(RecordError) as e:
        BlockJudgment.of({**J.dump(), **change}, "j.json")
    assert (e.value.where, e.value.field) == ("j.json", field)
