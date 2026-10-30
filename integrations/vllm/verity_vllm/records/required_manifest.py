"""The required-value manifest (`verity-sweep/required-manifest/v1`): the header fields the replay's coverage reads
(`check.replay.coverage`) and the identity rows as written, one per required Value.

Its writers are `query.required.request_manifest` (a request Program), `query.compose` (the global composition and a TP workload's
rank merge), `query.guarded_max.with_ms`, `acquire.compiled` and `acquire.served`; each writes every field typed here as
required.  `commit.binding.load_required_manifest` checks the digests against the Program, and `query.manifest.format` holds the
identity rows' encoding (`IDENTITY_KEY`, `manifest_digest`).

    m = RequiredManifest.load(row_dir / "manifest.json")      # MissingField names the file and the field
    m = RequiredManifest.of(doc, "the manifest the Commit carries")
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.records.fields import RecordError, optional, required

#: the writers stamp `query.manifest.format.SCHEMA`, which tests/records holds equal to this
SCHEMA = "verity-sweep/required-manifest/v1"


@dataclass(frozen=True)
class RequiredManifest:
    where: str                                        # the file it was read from, or what handed it over: a refusal names it
    program_digest: str
    manifest_digest: str
    requests: Sequence[Any]                           # the request ids the identities cover
    identities: Sequence[Mapping[str, Any]]
    # the query header: the served-compiled manifest (`acquire.served`) writes none
    query: Mapping[str, Any] | None
    # Program op_path -> [canonical op_path, invocation] of a shared module's calls: the compiled and served manifests
    # (`acquire.compiled`, `acquire.served`) address no shared module and write none
    op_path_aliases: Mapping[str, Sequence[Any]] | None
    # the engine that built it, its dataflow guard's state and its one-release cross-check: an older header's, which a
    # manifest of record may still carry; no writer in this tree writes them
    engine: str | Mapping[str, Any] | None
    guard: Mapping[str, Any] | None
    cross_check: Mapping[str, Any] | None

    @classmethod
    def of(cls, doc: RequiredManifest | Mapping[str, Any], where: str) -> RequiredManifest:
        """The record of a manifest as its writer wrote it (`doc`, the parsed JSON); a record passes through."""
        if isinstance(doc, cls):
            return doc
        if not isinstance(doc, Mapping):
            raise RecordError(where, "schema", f"cannot be read: the manifest is a {type(doc).__name__}, not a JSON object")
        schema = required(doc, "schema", where, str)
        if schema != SCHEMA:
            raise RecordError(where, "schema", f"is {schema!r}, not {SCHEMA!r}")
        return cls(where=where,
                   program_digest=required(doc, "program_digest", where, str),
                   manifest_digest=required(doc, "manifest_digest", where, str),
                   requests=required(doc, "requests", where, (list, tuple)),
                   identities=required(doc, "identities", where, (list, tuple)),
                   query=optional(doc, "query", where, Mapping),
                   op_path_aliases=optional(doc, "op_path_aliases", where, Mapping),
                   engine=optional(doc, "engine", where, (str, Mapping)),
                   guard=optional(doc, "guard", where, Mapping),
                   cross_check=optional(doc, "cross_check", where, Mapping))

    @classmethod
    def load(cls, path: str | Path) -> RequiredManifest:
        with open(path) as f:
            return cls.of(json.load(f), str(path))
