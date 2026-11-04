"""A predicted row's record (`verity-vllm/predicted-row/v1`, `FILE`): what the ahead-of-time compiler wrote as a single-GPU
row's Build of record (`program.predict.aot.write_row`, ``verity-vllm predict row``), beside its Build dirs and its manifest,
whose bytes it names by their sha256. ``predict compare --predicted`` holds it to a traced Build of the same row."""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from verity_vllm.records.fields import RecordError, only, required

SCHEMA = "verity-vllm/predicted-row/v1"
#: its name in the Build's dir, beside ``manifest.json``
FILE = "predicted.json"


@dataclass(frozen=True)
class PredictedRow:
    row: str                                          # the registered row
    config: str                                       # the checkpoint's config.json the Programs were predicted from
    code_version: str                                 # the predictor's code (`program.predict.score.code_version`)
    requests: Mapping[str, str]                       # each request Build dir's Program digest, by the dir's name
    workload_digest: str                              # GP-01's workload Program over them
    lag: int                                          # its LAG static
    manifest_digest: str                              # the manifest's digest over its identities
    manifest_sha256: str                              # the sha256 of manifest.json's bytes as written
    program_digest: str                               # the Program digest the manifest names
    identities: int                                   # its identities
    secs: float                                       # how long the Build took

    @classmethod
    def of(cls, doc: Any, where: str) -> PredictedRow:
        """The record as `write_row` wrote it, refused by field (`fields.RecordError`) where it isn't one: another schema, a
        field no writer writes, or a field missing or of another kind."""
        if not isinstance(doc, Mapping):
            raise RecordError(where, "schema", f"cannot be read: the record is a {type(doc).__name__}, not a JSON object")
        if doc.get("schema") != SCHEMA:
            raise RecordError(where, "schema", f"is {doc.get('schema')!r}, not {SCHEMA}: not a predicted row's record")
        only(doc, ("schema", *KINDS), where)
        return cls(**{k: required(doc, k, where, kind) for k, kind in KINDS.items()})

    def dump(self) -> dict[str, Any]:
        return {"schema": SCHEMA, **dataclasses.asdict(self)}


KINDS: dict[str, type | tuple[type, ...]] = {"row": str, "config": str, "code_version": str, "requests": Mapping,
                                             "workload_digest": str, "lag": int, "manifest_digest": str, "manifest_sha256": str,
                                             "program_digest": str, "identities": int, "secs": (int, float)}
