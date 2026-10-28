"""`records.inference_blocks`: the inference-block table's public statement reads back to what `dump` wrote, and is refused
by field when it is another format, holds a field no writer writes, misses one, or names a root that isn't 64 bytes in hex."""
from __future__ import annotations

import pytest

from verity_vllm.records.fields import MissingField, RecordError, UnknownField
from verity_vllm.records.inference_blocks import SCHEMA, Statement

ST = Statement(query={"name": "label", "version": 0, "params": {"family": "instances(root)", "by": "slot"}}, c=7, k=64,
               blocks=8, rows=32, root="ab" * 64)


def test_a_statement_reads_back_to_what_dump_wrote() -> None:
    doc = ST.dump()
    assert doc["format"] == SCHEMA
    assert Statement.of(doc, "s.json") == ST


@pytest.mark.parametrize(("change", "error", "field"), [
    ({"format": "verity-vllm/weights-registration/v1"}, RecordError, "format"),
    ({"slots": 8}, UnknownField, "slots"),
    ({"root": "ab" * 32}, RecordError, "root"),
    ({"c": None}, MissingField, "c"),
])
def test_a_statement_that_isnt_one_is_refused_by_field(change: dict, error: type, field: str) -> None:
    with pytest.raises(error, match=field):
        Statement.of({**ST.dump(), **change}, "s.json")
