"""The Commit binds the manifest's partition of record: under Q_word (`query.required.QUERY_OF_RECORD`) the manifest names each Program's
`verity/partition/v1` object and digest (`query.partition`), the Commit refuses a partition that is malformed, carries another digest or
names a Program that is not of record, and the binding map records it (`partition_entry`).  The manifest also records, per part, the strict
word check its Build passed (`query.word_check`, `pipeline.manifest.word_record`), and the Commit refuses one without it
(`word_check_refusal`, which `pipeline.row_records` reads too before it reuses a Build's manifest)."""

from __future__ import annotations

import contextlib
import glob
import json
import os
from typing import Any

__all__ = ["program_digests", "partition_of_record", "word_check_refusal", "require_partition", "partition_entry"]


def program_digests(*roots: str | None) -> set[str]:
    """The program digests the Builds under `roots` recorded (`result.json` program.digest, every request / rank Program dir)."""
    out: set[str] = set()
    for root in roots:
        if not root:
            continue
        for p in glob.glob(os.path.join(root, "**", "result.json"), recursive=True):
            with contextlib.suppress(Exception):
                s = (json.load(open(p)).get("program") or {}).get("digest")
                if isinstance(s, str) and s:
                    out.add(s)
    return out


def _by_part(x: Any, leaf: str, field: str, noun: str, at: str = "") -> dict[str, dict[str, Any]]:
    """`query.<field>`'s entries (a dict holding `leaf`) by where they sit ("" at the top, `by_component/r0`, `by_rank/1/by_component/r3`)."""
    if isinstance(x, dict) and leaf in x:
        return {at: x}
    if isinstance(x, dict) and any(k in x for k in ("by_component", "by_rank")):
        out: dict[str, dict[str, Any]] = {}
        for k in ("by_component", "by_rank"):
            for i, v in sorted((x.get(k) or {}).items()):
                out.update(_by_part(v, leaf, field, noun, f"{at}/{k}/{i}".lstrip("/")))
        return out
    raise SystemExit(f"[delta] REFUSED: query.{field} {at or '(top)'} is neither {noun} nor a composition of them")


def _partitions(part: Any) -> dict[str, dict[str, Any]]:
    return {k or "partition": v for k, v in _by_part(part, "object", "partition", "a partition").items()}


def partition_of_record(man: dict[str, Any], programs: set[str] | None = None) -> dict[str, Any]:
    """Under the query of record (Q_word) the manifest names its Program's partition (`query.partition`: one `{object, digest}`, or
    `by_component` / `by_rank` of them).  Each object must be a valid `verity/partition/v1` object of Q_word v1 {X: 16, W: 32} with that
    digest and, when the Commit holds the Builds' records (`programs`: their SHA-512s), name one of those Programs.  Refused by name
    (SystemExit) before any engine; returns what it checked, which the binding map records."""
    from verity.ml.circuits import partition_object as PO
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


def word_check_refusal(q: dict[str, Any]) -> str | None:
    """Why the query header `q` of a manifest under Q_word does not record, for every part it names a partition for, a strict word check of
    its own query that passed (`query.word_check`); None when it does."""
    if not isinstance(q.get("word_check"), dict):
        return "no word check recorded (query.word_check): built with --word-check off, or before the header recorded it"
    try:
        checks = _by_part(q["word_check"], "query", "word_check", "a word check")
        parts = _by_part(q["partition"], "object", "partition", "a partition") if isinstance(q.get("partition"), dict) else {"": {}}
    except SystemExit as e:
        return str(e.code).removeprefix("[delta] REFUSED: ")
    bad = [f"{k or '(top)'}: {v.get('query')} strict={v.get('strict')} ok={v.get('ok')} violations={v.get('violations')}"
           for k, v in sorted(checks.items())
           if not (v.get("query") == q.get("query_id") and v.get("strict") is True and v.get("ok") is True and v.get("violations") == 0)]
    bad += [f"{k or '(top)'}: no word check" for k in sorted(set(parts) - set(checks))]
    return f"the word check is not a passing strict {q.get('query_id')} check: " + "; ".join(bad[:4]) if bad else None


def require_partition(man: dict[str, Any], programs_root: str | None, program_dir: str | None) -> dict[str, Any]:
    """`partition_of_record` over the Programs the Builds under `programs_root` / `program_dir` recorded and, under the query of record,
    a passing word check for every part (`word_check_refusal`; SystemExit naming what is missing); prints what it checked and returns `man`."""
    rec = partition_of_record(man, program_digests(programs_root, program_dir))
    why = word_check_refusal(man.get("query") or {}) if rec["checked"] else None
    if why:
        raise SystemExit(f"[delta] REFUSED: {why}; rebuild the manifest (verity-vllm manifest build / build-global, --word-check 16/32)")
    print(f"[delta] partition of record: {rec}" + (" (word check recorded and passed)" if rec["checked"] else ""), flush=True)
    return man


def partition_entry(man: dict[str, Any] | None) -> dict[str, Any]:
    """The binding map's `partition` entry: the manifest's partition of record, or nothing (no manifest, or not the query of record)."""
    part = ((man or {}).get("query") or {}).get("partition")
    return {"partition": part} if part else {}
