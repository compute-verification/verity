"""A served window's boundary commitment of one size class (`verity-vllm/boundary-rows/v1`, `FILE`): the public part
`pipeline.boundary_rows` writes under ``w<window>/p<P>-o<O>/`` as the window closes, which the challenger's copy of the
class's layout (`boundary_rows.layout_of`), its registration (`boundary_rows.record`) and a recommit from the prover's files
(`boundary_rows.reopen`) read. Its id is under ``format``, since ``schema`` is each value's row schema."""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from verity_vllm.records.fields import RecordError, only, optional, required

SCHEMA = "verity-vllm/boundary-rows/v1"
#: its name in the class's directory, beside the prover's ``index.json``, ``index-salt.bin`` and ``salts.bin``
FILE = "commit.json"


@dataclass(frozen=True)
class BoundaryCommit:
    window: int                                       # the window's index in its serve
    run: str                                          # the window's run, which every value's domain binds
    prompt: int                                       # the class's prompt cap P
    outputs: int                                      # its output cap O
    check: str                                        # the check template's id
    per_prompt: int                                   # its instances a prompt row
    per_output: int                                   # its instances an output
    inline: bool                                      # whether the replay block holds them inlined (`Layout.inline`)
    spans: int                                        # the class's spans N, one replay block each
    replay: str                                       # the replay block's id
    program: str                                      # the registration's program digest
    partition: str                                    # its partition's digest
    schema: str                                       # each value's row schema
    values: Mapping[str, Mapping[str, Any]]           # each value's root, words a row, leaves and domain id, by port
    index: str                                        # the prover's index's hiding leaf
    index_rule: str                                   # how that leaf hashes the index
    seconds: float                                    # how long the commit took
    tables: Mapping[str, Any] | None = None           # its history tables' layers and capacity, when it has any

    @classmethod
    def of(cls, doc: Any, where: str) -> BoundaryCommit:
        """The record as `boundary_rows.write` wrote it, refused by field (`fields.RecordError`) where it isn't one: another
        format, a field no writer writes, or a field missing or of another kind."""
        if not isinstance(doc, Mapping):
            raise RecordError(where, "format", f"cannot be read: the record is a {type(doc).__name__}, not a JSON object")
        if doc.get("format") != SCHEMA:
            raise RecordError(where, "format", f"is {doc.get('format')!r}, not {SCHEMA}: not a boundary commitment")
        only(doc, ("format", *KINDS, *OPTIONAL), where)
        return cls(**{k: required(doc, k, where, kind) for k, kind in KINDS.items()},
                   **{k: optional(doc, k, where, kind) for k, kind in OPTIONAL.items()})

    def dump(self) -> dict[str, Any]:
        return {"format": SCHEMA, **{k: v for k, v in dataclasses.asdict(self).items() if k in KINDS or v is not None}}


KINDS: dict[str, type | tuple[type, ...]] = {"window": int, "run": str, "prompt": int, "outputs": int, "check": str,
                                             "per_prompt": int, "per_output": int, "inline": bool, "spans": int,
                                             "replay": str, "program": str, "partition": str, "schema": str,
                                             "values": Mapping, "index": str, "index_rule": str, "seconds": (int, float)}
#: what a writer writes only where it holds one
OPTIONAL: dict[str, type | tuple[type, ...]] = {"tables": Mapping}
