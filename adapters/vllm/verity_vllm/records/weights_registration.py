"""The committer's weights registration (`verity-vllm/weights-registration/v1`): the hiding root of each weight parameter the
window circuit registers (``weights/<label>@<words>``), its rows committed under the committer's secret salts once per model
and key, before the first window's coins (`program.predict.window_weights.weights_registration` writes it), as the verifier
pins it (`window_weights.registration_pins`, `records_audit.py count --pins`) and a window's registration records it
(`pipeline.advice.registered_weights`). It names no key and no salt: what it states of the weights is each one's size and
its hiding root."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from verity_vllm.records.fields import RecordError, only, required

SCHEMA = "verity-vllm/weights-registration/v1"
#: an entry's fields: core's `registered.Registered.entry()` of a value of 16-bit words
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

    @property
    def roots(self) -> dict[str, str]:
        """Each weight parameter's root, ``{param: root}``."""
        return {p: e["root"] for p, e in sorted(self.values.items())}
