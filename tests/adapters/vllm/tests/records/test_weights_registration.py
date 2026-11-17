"""`records.weights_registration`: the committer's weights registration reads back to what `dump` wrote, and is refused by field
when it is no JSON object or another schema, holds a field no writer writes or no weight, or an entry that doesn't name its
parameter under a root of 64 bytes."""
from __future__ import annotations

import pytest

from verity_vllm.records.fields import MissingField, RecordError, UnknownField
from verity_vllm.records.weights_registration import SCHEMA, WeightsRegistration

P0, P1 = "weights/model.embed_tokens.weight@4096", "weights/model.norm.weight@64"


def _entry(name: str, **change) -> dict:
    return {"value": name, "root": "ab" * 64, "leaves": 4, "words": 4096, "domain": "cd" * 64, **change}


REG = WeightsRegistration(program="ef" * 64, repo="org/model", revision="0" * 40, grain={"hidden": 64},
                          values={P0: _entry(P0), P1: _entry(P1, root="ba" * 64)},
                          unrooted={"weights/model.bias@1": "fewer words than a leaf"})


def test_a_registration_reads_back_to_what_dump_wrote() -> None:
    doc = REG.dump()
    assert doc["schema"] == SCHEMA
    assert WeightsRegistration.of(doc, "w.json") == REG
    assert REG.roots == {P0: "ab" * 64, P1: "ba" * 64}


def test_a_registration_that_is_no_json_object_is_refused() -> None:
    with pytest.raises(RecordError, match="'schema' cannot be read: the registration is a list, not a JSON object"):
        WeightsRegistration.of([REG.dump()], "w.json")


@pytest.mark.parametrize(("change", "error", "field"), [
    ({"schema": "verity-vllm/expert-tables/v1"}, RecordError, "schema"),
    ({"key": "k"}, UnknownField, "key"),
    ({"values": {}}, RecordError, "values"),
    ({"values": {P0: _entry(P1)}}, RecordError, f"values.{P0}"),                         # an entry that names another parameter
    ({"values": {P0: _entry(P0, root="ab" * 32)}}, RecordError, f"values.{P0}"),
    ({"values": {P0: _entry(P0, root=None)}}, RecordError, f"values.{P0}"),
    ({"values": {P0: _entry(P0, salt="s")}}, UnknownField, f"values.{P0}.salt"),
    ({"model": {"repo": "org/model"}}, MissingField, "model.revision"),
])
def test_a_registration_that_isnt_one_is_refused_by_field(change: dict, error: type, field: str) -> None:
    with pytest.raises(error) as e:
        WeightsRegistration.of({**REG.dump(), **change}, "w.json")
    assert (e.value.where, e.value.field) == ("w.json", field)
