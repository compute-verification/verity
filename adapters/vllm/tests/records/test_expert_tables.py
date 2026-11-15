"""`records.expert_tables`: a rank's expert tables registration reads back to what `dump` wrote, and is refused by field when it
is no JSON object or another schema, holds a field no writer writes or no table, an entry that doesn't name its table under a
root of 64 bytes, or nonces that aren't one for each table, each its own."""
from __future__ import annotations

import pytest

from verity_vllm.records.expert_tables import SCHEMA, ExpertTables
from verity_vllm.records.fields import MissingField, RecordError, UnknownField

T0, T1 = "weights/model.layers.0.mlp.experts.gate_up@8/b0/table", "weights/model.layers.0.mlp.experts.gate_up@8/b1/table"


def _entry(name: str, **change) -> dict:
    return {"value": name, "root": "ab" * 64, "leaves": 4, "words": 32, "word_bits": 8, "domain": "cd" * 64, **change}


REG = ExpertTables(program="ef" * 64, repo="org/model", revision="0" * 40, world=2, rank=1, prefix="model", blocks={"gate_up": 2},
                   values={T0: _entry(T0), T1: _entry(T1)}, nonces={T0: 0, T1: 1})


def test_a_registration_reads_back_to_what_dump_wrote() -> None:
    doc = REG.dump()
    assert doc["schema"] == SCHEMA
    assert ExpertTables.of(doc, "t.json") == REG


def test_a_registration_that_is_no_json_object_is_refused() -> None:
    with pytest.raises(RecordError, match="'schema' cannot be read: the registration is a list, not a JSON object"):
        ExpertTables.of([REG.dump()], "t.json")


@pytest.mark.parametrize(("change", "error", "field"), [
    ({"schema": "verity-vllm/weights-registration/v1"}, RecordError, "schema"),
    ({"key": "k"}, UnknownField, "key"),
    ({"values": {}}, RecordError, "values"),
    ({"values": {T0: _entry(T1), T1: _entry(T1)}}, RecordError, f"values.{T0}"),         # an entry that names another table
    ({"values": {T0: _entry(T0, root="ab" * 32), T1: _entry(T1)}}, RecordError, f"values.{T0}"),
    ({"values": {T0: _entry(T0, root=None), T1: _entry(T1)}}, RecordError, f"values.{T0}"),
    ({"values": {T0: _entry(T0, salt="s"), T1: _entry(T1)}}, UnknownField, f"values.{T0}.salt"),
    ({"nonces": {T0: 0}}, RecordError, "nonces"),                                        # a table with no nonce
    ({"nonces": {T0: 0, T1: 0}}, RecordError, "nonces"),                                 # two tables under one
    ({"nonces": {T0: 0, T1: True}}, RecordError, "nonces"),
    ({"rank": None}, MissingField, "rank"),
])
def test_a_registration_that_isnt_one_is_refused_by_field(change: dict, error: type, field: str) -> None:
    with pytest.raises(error) as e:
        ExpertTables.of({**REG.dump(), **change}, "t.json")
    assert (e.value.where, e.value.field) == ("t.json", field)
