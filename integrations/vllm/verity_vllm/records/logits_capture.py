"""What a replay at the served step shapes took at the sampler (`verity-vllm/logits-capture/v1`, ``capture.json``): for each
sampled request, the file of the logits rows its tokens were selected from (bf16 bits, row j its j-th token's), their
sha256, the ids selected, the engine steps, and what fixes its Program. `pipeline.interior_rows.write_capture` writes it and
`pipeline.interior_rows.captured` reads a span of it as a token select's units."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from verity_vllm.records.fields import RecordError, only, required

SCHEMA = "verity-vllm/logits-capture/v1"
SPAN = ("file", "sha256", "rows", "tokens", "steps", "program")


@dataclass(frozen=True)
class LogitsCapture:
    template: str                                     # the token-select row template the rows are units of
    spans: Mapping[str, Mapping[str, Any]]            # each request's `SPAN` fields

    @classmethod
    def of(cls, doc: Any, where: str) -> LogitsCapture:
        """The capture as its writer wrote it, refused by field (`fields.RecordError`) where it isn't one: another schema, a
        field no writer writes, or a span whose rows, tokens and steps don't count one row a token."""
        if not isinstance(doc, Mapping) or doc.get("schema") != SCHEMA:
            raise RecordError(where, "schema", f"is not {SCHEMA}: a sampler capture")
        only(doc, ("schema", "template", "spans"), where)
        spans = required(doc, "spans", where, Mapping)
        for rid in spans:
            s = required(spans, rid, where, Mapping, f"spans.{rid}")
            only(s, SPAN, where, f"spans.{rid}.")
            rows, tokens, steps = (required(s, k, where, list, f"spans.{rid}.{k}") for k in ("rows", "tokens", "steps"))
            if len(rows) != 2 or rows[0] != len(tokens) or len(steps) != len(tokens):
                raise RecordError(where, f"spans.{rid}", f"holds rows {rows} for {len(tokens)} tokens at {len(steps)} steps")
            for k in ("file", "sha256", "program"):
                required(s, k, where, str, f"spans.{rid}.{k}")
        return cls(template=required(doc, "template", where, str), spans=spans)

    def dump(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "template": self.template, "spans": {r: dict(s) for r, s in self.spans.items()}}
