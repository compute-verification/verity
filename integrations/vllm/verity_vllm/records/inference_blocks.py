"""The inference-block table's public statement (`verity-vllm/inference-blocks/v1`), as B emits it beside its window circuit
(`program.predict.inference_blocks.commit`): the label query, every block's gates ``c`` and fresh-input bits ``k``, the grid
levels of the block count and of the table's rows, and the table's hiding root. The table itself is private; what this states
of it is its budgets, its padded counts and its root."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from verity_vllm.records.fields import RecordError, only, required

SCHEMA = "verity-vllm/inference-blocks/v1"
ROOT = re.compile(r"[0-9a-f]{128}")


@dataclass(frozen=True)
class Statement:
    query: Mapping[str, Any]                          # the label rule over the root's template instances
    c: int                                            # every block's gates
    k: int                                            # every block's fresh input, in bits
    blocks: int                                       # the block count's grid level
    rows: int                                         # the table's rows' grid level
    root: str                                         # the table's hiding root, 64 bytes in hex

    @classmethod
    def of(cls, doc: Any, where: str) -> Statement:
        """The statement as its writer wrote it, refused by field (`fields.RecordError`) where it isn't one: another
        format, a field no writer writes, or a root that isn't 64 bytes in hex."""
        if not isinstance(doc, Mapping):
            raise RecordError(where, "format", f"cannot be read: the statement is a {type(doc).__name__}, not a JSON object")
        if doc.get("format") != SCHEMA:
            raise RecordError(where, "format", f"is {doc.get('format')!r}, not {SCHEMA}")
        only(doc, ("format", "query", "c", "k", "blocks", "rows", "root"), where)
        root = required(doc, "root", where, str)
        if not ROOT.fullmatch(root):
            raise RecordError(where, "root", "isn't 64 bytes in hex")
        return cls(query=required(doc, "query", where, Mapping), c=required(doc, "c", where, int),
                   k=required(doc, "k", where, int), blocks=required(doc, "blocks", where, int),
                   rows=required(doc, "rows", where, int), root=root)

    def dump(self) -> dict[str, Any]:
        return {"format": SCHEMA, "query": dict(self.query), "c": self.c, "k": self.k, "blocks": self.blocks,
                "rows": self.rows, "root": self.root}
