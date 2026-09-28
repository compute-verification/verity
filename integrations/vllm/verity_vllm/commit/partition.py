"""The Commit binds the manifest's partition of record: under Q_word (`query.required.QUERY_OF_RECORD`) the manifest names each Program's
`verity/partition/v1` object and digest (`query.partition`), the Commit refuses a partition that is malformed, carries another digest or
names a Program that is not of record, and the binding map records it (`partition_entry`)."""

from __future__ import annotations

import contextlib
import glob
import json
import os
from typing import Any

__all__ = ["program_sha512s", "partition_of_record", "require_partition", "partition_entry"]


def program_sha512s(*roots: str | None) -> set[str]:
    """The SHA-512s the Builds under `roots` recorded (`result.json` program.sha512, every request / rank Program dir)."""
    out: set[str] = set()
    for root in roots:
        if not root:
            continue
        for p in glob.glob(os.path.join(root, "**", "result.json"), recursive=True):
            with contextlib.suppress(Exception):
                s = (json.load(open(p)).get("program") or {}).get("sha512")
                if isinstance(s, str) and s:
                    out.add(s)
    return out


def _partitions(part: Any, at: str = "") -> dict[str, dict[str, Any]]:
    """`query.partition`'s objects by where they sit (`by_component/r0`, `by_rank/1/by_component/r3`, ...)."""
    if isinstance(part, dict) and "object" in part:
        return {at or "partition": part}
    if isinstance(part, dict) and any(k in part for k in ("by_component", "by_rank")):
        out: dict[str, dict[str, Any]] = {}
        for k in ("by_component", "by_rank"):
            for i, v in sorted((part.get(k) or {}).items()):
                out.update(_partitions(v, f"{at}/{k}/{i}".lstrip("/")))
        return out
    raise SystemExit(f"[delta] REFUSED: query.partition {at or '(top)'} is neither a partition nor a composition of them")


def partition_of_record(man: dict[str, Any], programs: set[str] | None = None) -> dict[str, Any]:
    """Under the query of record (Q_word) the manifest names its Program's partition (`query.partition`: one `{object, digest}`, or
    `by_component` / `by_rank` of them).  Each object must be a valid `verity/partition/v1` object of Q_word v1 {X: 16, W: 32} with that
    digest and, when the Commit holds the Builds' records (`programs`: their SHA-512s), name one of those Programs.  Refused by name
    (SystemExit) before any engine; returns what it checked, which the binding map records."""
    from verity.ir import partition_object as PO
    from verity_vllm.query.required import Q_WORD_ID
    q = man.get("query") or {}
    if q.get("query_id") != Q_WORD_ID:
        return {"query": q.get("query_id"), "checked": False, "why": "not the query of record: no partition to bind"}
    part = q.get("partition")
    if not isinstance(part, dict):
        raise SystemExit(f"[delta] REFUSED: the manifest is built under {Q_WORD_ID} but names no partition (query.partition)")
    flat = _partitions(part)
    bad = []
    for k, v in sorted(flat.items()):
        obj = v["object"]
        if not PO.validate(obj).ok or obj.get("query") != PO.query(16, 32):
            bad.append(f"{k}: not a verity/partition/v1 object of Q_word v1 {{16, 32}}")
        elif PO.digest(obj) != v.get("digest"):
            bad.append(f"{k}: digest {str(v.get('digest'))[:16]} != the object's {PO.digest(obj)[:16]}")
        elif programs and obj["program"] not in programs:
            bad.append(f"{k}: names Program {obj['program'][:16]}, not a Program of record")
    if bad:
        raise SystemExit("[delta] REFUSED: the manifest's partition is not the Program's: " + "; ".join(bad[:4]))
    return {"query": Q_WORD_ID, "checked": True, "partitions": len(flat), "programs_checked": bool(programs),
            "digests": sorted(v["digest"] for v in flat.values())}


def require_partition(man: dict[str, Any], programs_root: str | None, program_dir: str | None) -> dict[str, Any]:
    """`partition_of_record` over the Programs the Builds under `programs_root` / `program_dir` recorded; prints what it checked and
    returns `man`."""
    print(f"[delta] partition of record: {partition_of_record(man, program_sha512s(programs_root, program_dir))}", flush=True)
    return man


def partition_entry(man: dict[str, Any] | None) -> dict[str, Any]:
    """The binding map's `partition` entry: the manifest's partition of record, or nothing (no manifest, or not the query of record)."""
    part = ((man or {}).get("query") or {}).get("partition")
    return {"partition": part} if part else {}
