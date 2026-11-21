"""The committer's weights registration (`verity-vllm/weights-registration/v1`): the hiding root of each weight parameter the
window circuit registers (``weights/<label>@<words>``), its rows committed under the committer's secret salts once per
deployment, before its first epoch's coins (`program.predict.window_weights.weights_registration` writes it), as the verifier
pins it (`window_weights.registration_pins`, `records_audit.py count --pins`) and a window's registration records it
(`pipeline.advice.registered_weights`). It names no key and no salt: what it states of the weights is each one's size and
its hiding root."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.circuits.codec import canonical_json

from verity_vllm.records.fields import RecordError, only, required

__all__ = ["OF_RECORD", "WeightsRegistration", "write_of_record"]

SCHEMA = "verity-vllm/weights-registration/v1"
#: the registration of record's file, beside the registration, which the audit hands the verifier's check
OF_RECORD = "weights_of_record.json"
#: an entry's fields: core's `registered_value.Registered.entry()` of a value of 16-bit words
ENTRY = ("value", "root", "leaves", "words", "domain")
ROOT = re.compile(r"[0-9a-f]{128}")


@dataclass(frozen=True)
class WeightsRegistration:
    program: str                                      # the weights record's program digest, which each entry's domain binds
    repo: str                                         # the model registered, at its revision
    revision: str
    grain: Mapping[str, int]                          # the window grain its weight parameters are cut at
    values: Mapping[str, Mapping[str, Any]]           # each weight parameter's entry, `ENTRY`'s fields
    unrooted: Mapping[str, str]                       # each weight parameter with no root, and why

    @classmethod
    def of(cls, doc: Any, where: str) -> WeightsRegistration:
        """The registration as its writer wrote it, refused by field (`fields.RecordError`) where it isn't one: another
        schema (a weights record's among them), a field no writer writes, no value, or an entry that doesn't name its
        parameter or whose root isn't 64 bytes in hex."""
        if not isinstance(doc, Mapping):
            raise RecordError(where, "schema", f"cannot be read: the registration is a {type(doc).__name__}, not a JSON object")
        if doc.get("schema") != SCHEMA:
            raise RecordError(where, "schema", f"is {doc.get('schema')!r}, not {SCHEMA}: a committer's weights registration, "
                                               "never a weights record")
        only(doc, ("schema", "program", "model", "grain", "values", "unrooted"), where)
        model, values = required(doc, "model", where, Mapping), required(doc, "values", where, Mapping)
        if not values:
            raise RecordError(where, "values", "is empty: a registration registers at least one weight")
        for p in values:
            e = required(values, p, where, Mapping, f"values.{p}")
            only(e, ENTRY, where, f"values.{p}.")
            if e.get("value") != p or not (isinstance(e.get("root"), str) and ROOT.fullmatch(e["root"])):
                raise RecordError(where, f"values.{p}", "doesn't name its parameter with a root of 64 bytes in hex")
        return cls(program=required(doc, "program", where, str), repo=required(model, "repo", where, str, "model.repo"),
                   revision=required(model, "revision", where, str, "model.revision"),
                   grain=required(doc, "grain", where, Mapping), values=values, unrooted=required(doc, "unrooted", where, Mapping))

    def dump(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "program": self.program, "model": {"repo": self.repo, "revision": self.revision},
                "grain": dict(self.grain), "values": {p: dict(e) for p, e in sorted(self.values.items())},
                "unrooted": dict(sorted(self.unrooted.items()))}

    def of_record(self) -> bytes:
        """The registration of record as the verifier's check reads it (``records_audit.py --registered``): exactly
        ``{program, values}`` in canonical JSON, whose SHA-512 is the digest the auditor's side holds."""
        return canonical_json({"program": self.program, "values": {p: dict(e) for p, e in sorted(self.values.items())}})

    @property
    def roots(self) -> dict[str, str]:
        """Each weight parameter's root, ``{param: root}``."""
        return {p: e["root"] for p, e in sorted(self.values.items())}


def write_of_record(registration: str | Path) -> str:
    """The registration at ``registration``'s document of record (`WeightsRegistration.of_record`), written beside it as
    `OF_RECORD`; -> its SHA-512. Refused (`fields.RecordError`) where the file is no registration."""
    path = Path(registration)
    body = WeightsRegistration.of(json.loads(path.read_text()), str(path)).of_record()
    path.with_name(OF_RECORD).write_bytes(body)
    return hashlib.sha512(body).hexdigest()
