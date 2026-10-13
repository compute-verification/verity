"""The benchmark result schema and identity-strict comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from verity_vllm.config import option

SCHEMA = "verity-vllm/overhead-benchmark/v1"


class IdentityMismatch(ValueError):
    """Two results do not describe the same benchmark identity."""


def validate(doc: Mapping[str, Any]) -> None:
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"benchmark result schema must be {SCHEMA!r}")
    for key in ("identity", "baseline", "sweep", "gates", "execution", "points"):
        if key not in doc:
            raise ValueError(f"benchmark result has no {key!r}")
    arm_ids = [a.get("id") for a in doc["identity"].get("arms", [])]
    if len(set(arm_ids)) != len(arm_ids) or doc["baseline"] not in arm_ids:
        raise ValueError("benchmark result arm ids must be unique and include its baseline")
    if any(not g.get("passed", False) for g in doc["gates"]) and doc["points"]:
        raise ValueError("a failed benchmark gate must have no timed points")


def write(doc: Mapping[str, Any], path: str | Path) -> None:
    validate(doc)
    Path(path).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def _leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(_leaves(child, path))
        return out
    if isinstance(value, list):
        out = {}
        for i, child in enumerate(value):
            out.update(_leaves(child, f"{prefix}[{i}]"))
        return out
    return {prefix: value}


def identity_mismatches(a: Mapping[str, Any], b: Mapping[str, Any], allow: tuple[str, ...] = ()) -> dict[str, tuple[Any, Any]]:
    left, right = _leaves(a.get("identity", {})), _leaves(b.get("identity", {}))
    allowed = set(allow)

    def permitted(path: str) -> bool:
        leaf = path.rsplit(".", 1)[-1]
        return path in allowed or leaf in allowed or any(path.startswith(x + ".") or path.startswith(x + "[") for x in allowed)

    return {
        path: (left.get(path), right.get(path))
        for path in sorted(set(left) | set(right))
        if left.get(path) != right.get(path) and not permitted(path)
    }


def judge(a: Mapping[str, Any], b: Mapping[str, Any], allow: tuple[str, ...] = ()) -> dict[str, Any]:
    """Accept a comparison only when every unallowed identity field agrees."""

    validate(a)
    validate(b)
    bad = identity_mismatches(a, b, allow)
    if bad:
        fields = ", ".join(bad)
        raise IdentityMismatch(f"benchmark identities differ at {fields}; name intentional fields with --allow")
    return {"accepted": True, "allowed": list(allow)}


@dataclass(frozen=True, kw_only=True)
class Options:
    """Compare two benchmark results only when their identities agree."""

    left: Path = option("left", type=Path)
    right: Path = option("right", type=Path)
    allow: list[str] = option("--allow", action="append", default=[],
                              help="identity field or path allowed to differ; repeatable")


def main(a: Options) -> int:
    left = json.loads(a.left.read_text())
    right = json.loads(a.right.read_text())
    print(json.dumps(judge(left, right, tuple(a.allow))))
    return 0
