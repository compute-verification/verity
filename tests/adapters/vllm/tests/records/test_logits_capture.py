"""`records.logits_capture`: a sampler capture reads back to what `dump` wrote, and is refused by field when it is no JSON object
or another schema, holds a field no writer writes, or a span whose rows, tokens and steps don't count one row a token."""
from __future__ import annotations

import pytest

from verity_vllm.records.fields import MissingField, RecordError, UnknownField
from verity_vllm.records.logits_capture import SCHEMA, LogitsCapture

SPAN = {"file": "r0.bf16", "sha256": "ab" * 32, "rows": [2, 1000], "tokens": [5, 6], "steps": [0, 1], "program": "ef" * 64}
CAPTURE = LogitsCapture(template="token-select", spans={"r0": SPAN, "r1": {**SPAN, "file": "r1.bf16"}})


def _span(**change) -> dict:
    return {"spans": {"r0": {k: v for k, v in {**SPAN, **change}.items() if v is not ...}}}


def test_a_capture_reads_back_to_what_dump_wrote() -> None:
    doc = CAPTURE.dump()
    assert doc["schema"] == SCHEMA and list(doc["spans"]) == ["r0", "r1"]
    assert LogitsCapture.of(doc, "capture.json") == CAPTURE


@pytest.mark.parametrize("doc", [[CAPTURE.dump()], {**CAPTURE.dump(), "schema": "verity-vllm/expert-tables/v1"}], ids=["a list", "another schema"])
def test_what_is_no_capture_is_refused(doc) -> None:
    with pytest.raises(RecordError, match="'schema' is not verity-vllm/logits-capture/v1: a sampler capture"):
        LogitsCapture.of(doc, "capture.json")


@pytest.mark.parametrize(("change", "error", "field"), [
    ({"note": "n"}, UnknownField, "note"),
    ({"template": None}, MissingField, "template"),
    ({"spans": None}, MissingField, "spans"),
    ({"spans": {"r0": ["r0.bf16"]}}, RecordError, "spans.r0"),
    (_span(logits=[[0.0]]), UnknownField, "spans.r0.logits"),
    (_span(rows=[3, 1000]), RecordError, "spans.r0"),                                    # three rows for two tokens
    (_span(rows=[2]), RecordError, "spans.r0"),                                          # rows are a count and a width
    (_span(steps=[0]), RecordError, "spans.r0"),                                         # a token with no step
    (_span(tokens=...), MissingField, "spans.r0.tokens"),
    (_span(file=...), MissingField, "spans.r0.file"),
    (_span(sha256=...), MissingField, "spans.r0.sha256"),
    (_span(program=7), RecordError, "spans.r0.program"),
])
def test_a_capture_that_isnt_one_is_refused_by_field(change: dict, error: type, field: str) -> None:
    with pytest.raises(error) as e:
        LogitsCapture.of({**CAPTURE.dump(), **change}, "capture.json")
    assert (e.value.where, e.value.field) == ("capture.json", field)
