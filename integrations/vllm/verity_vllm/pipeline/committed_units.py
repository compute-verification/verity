"""Policy and codecs for explicitly listed, committed-but-unverified Program inputs.

These units are test-only until the claims ruling named below changes.  A non-empty
list never silently becomes a Build or Commit of record.
"""
from __future__ import annotations

import fnmatch
import json
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA = "verity-vllm/committed-units/v1"

# Fail closed.  A later claims PR may flip this only while citing Daniel's ruling.
OF_RECORD_APPROVED = False
OF_RECORD_RULING = None
FAMILY = "unverified_committed_values"


def parse(entries: Sequence[str | Mapping[str, Any]] | None) -> tuple[dict[str, Any], ...]:
    """Validate repeatable CLI JSON objects (or already-decoded workload entries)."""
    out: list[dict[str, Any]] = []
    for k, raw in enumerate(entries or ()):
        try:
            value = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (TypeError, ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"committed-unit entry {k} is not a JSON object: {e}") from None
        if not isinstance(value, dict):
            raise ValueError(f"committed-unit entry {k} must be a JSON object")
        extra = sorted(set(value) - {"op", "module", "reason"})
        if extra:
            raise ValueError(f"committed-unit entry {k} has unknown fields {extra}")
        for field in ("op", "module", "reason"):
            if not isinstance(value.get(field), str) or not value[field].strip():
                raise ValueError(f"committed-unit entry {k} needs a non-empty string {field!r}")
        out.append({"op": value["op"].strip(), "module": value["module"].strip(), "reason": value["reason"].strip()})
    return tuple(out)


def matches(entry: Mapping[str, Any], op: str, module: str) -> bool:
    return str(entry["op"]) == op and fnmatch.fnmatchcase(module, str(entry["module"]))


def record(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The record carried beside a Build/profile when listed units exist."""
    names = [str(u["input"]) for u in units]
    return {"schema": SCHEMA, "inputs": names, "units": [dict(u) for u in units],
            "integrity": {"linked": False, "unverified": names},
            "of_record": bool(OF_RECORD_APPROVED),
            "of_record_policy": {"approved": bool(OF_RECORD_APPROVED), "ruling": OF_RECORD_RULING}}


def from_record(doc: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(doc, Mapping):
        return None
    value = doc.get("committed_units")
    return dict(value) if isinstance(value, Mapping) and value.get("units") else None


def require_of_record(doc: Mapping[str, Any] | None, *, where: str) -> None:
    """Refuse a listed Build/manifest on every of-record path until the ruling flips."""
    value = from_record(doc)
    if value and not OF_RECORD_APPROVED:
        names = ", ".join(str(x) for x in value.get("inputs") or ())
        raise RuntimeError(f"{where}: committed-only inputs [{names}] are unverified and not approved for a Build or Commit of record")


__all__ = ["FAMILY", "OF_RECORD_APPROVED", "OF_RECORD_RULING", "SCHEMA", "from_record", "matches", "parse", "record",
           "require_of_record"]
