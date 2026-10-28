"""Query artifact identities and the use document.

Four identities, each with one meaning:

    query_id        sha512 of the canonical JSON of the query AST -- the program-independent identity of the query
                    text (`query_codec.query_id` / `gateset.query_id` when present; the same construction here).
    program_digest  `verity.ml.circuits.codec.program_digest` -- re-exported, untouched, never recomputed differently.
    binding_id      sha512(schema || query_id || program_digest) over the query document's schema string: the
                    identity of "this query over this Program".  UTF-8, plain concatenation (the two digests are
                    fixed-length hex, so the concatenation is unambiguous).
    use document    `verity-ir/query-use/v1`: what the binding was used FOR (a VU partition at `w_max`) and the
                    verdict.  Partitionhood and `w_max` are properties of a use, not of a query: the query document
                    (`verity-ir/query/v1`) carries NO `kind` / `w_max`.

Documents:

    query document   {"schema": "verity-ir/query/v1", "query_id", "program_digest", "binding_id", "query": <AST JSON>,
                      "provenance": {...}?}
    use document     {"schema": "verity-ir/query-use/v1", "binding_id", "use": {"kind": "vu-partition", "w_max": 32},
                      "verdict": VuVerdict.to_json(), "evaluator": {"version": EVALUATOR_VERSION, "utc": "...Z"}}

Both live beside `program.json` in a run directory and are indexed by `experiment.json`; nothing here changes
`program_digest` or the Program descriptor.  G8 (`verity_vllm/check/gates.py`, integration owner) reads the use document:
`skipped` when absent, else pass iff `verdict.status == "ok"` and the binding matches the run's `program_digest`.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from verity.ml.circuits.codec import canonical_json, program_digest  # re-exported identity; never wrapped or altered
from verity.ml.circuits.query_codec import SCHEMA as QUERY_SCHEMA

USE_SCHEMA = "verity-ir/query-use/v1"
USE_KIND_VU_PARTITION = "vu-partition"
EVALUATOR_VERSION = "verity-ir/partition/0.1"


# ---------------------------------------------------------------------------------------------------------
# identities
# ---------------------------------------------------------------------------------------------------------


def _as_query(query: Any) -> Any:
    """A `gateset.Query` when `query` is one (or its string form and `gateset` is importable), else None."""
    try:
        from verity.ml.circuits import query_ast as gateset
    except ImportError:
        return None
    if isinstance(query, gateset.Query):
        return query
    if isinstance(query, str):
        return gateset.parse(query)
    return None


def query_json(query: Any) -> Any:
    """The JSON AST of a query: a dict/list is taken as already-encoded JSON; a `gateset.Query` (or its string
    form) goes through `query_codec.to_json`; otherwise the object's own `to_json()`."""
    if isinstance(query, (dict, list)):
        return query
    q = _as_query(query)
    if q is not None:
        from verity.ml.circuits.query_codec import to_json
        return to_json(q)
    fn = getattr(query, "to_json", None)
    if fn is not None:
        return fn()
    raise TypeError(f"cannot encode query {type(query).__name__} as JSON (not a gateset.Query, no to_json)")


def query_id(query: Any) -> str:
    """sha512 of the canonical JSON (sorted keys, compact separators, ASCII) of the query AST.  A `gateset.Query`
    is identified by `query_codec.query_id` (one owner of the identity; paths re-rendered canonically); anything
    else by the same construction over `query_json(query)`."""
    q = _as_query(query)
    if q is not None:
        from verity.ml.circuits.query_codec import query_id as _qid
        return _qid(q)
    return hashlib.sha512(canonical_json(query_json(query))).hexdigest()


def binding_id(qid: str, pdigest: str, schema: str = QUERY_SCHEMA) -> str:
    """sha512(schema || query_id || program_digest), UTF-8, plain concatenation."""
    _check_hex(qid, "query_id")
    _check_hex(pdigest, "program_digest")
    return hashlib.sha512((schema + qid + pdigest).encode("utf-8")).hexdigest()


def _check_hex(s: str, what: str) -> None:
    if not (isinstance(s, str) and len(s) == 128 and all(c in "0123456789abcdef" for c in s)):
        raise ValueError(f"{what} must be a 128-character lowercase hex sha512, got {s!r}")


# ---------------------------------------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------------------------------------


def query_document(query: Any, program_or_digest: Any, provenance: dict | None = None) -> dict:
    """The `verity-ir/query/v1` document binding `query` to a Program (or its digest).  No `kind`, no `w_max`."""
    pdigest = program_or_digest if isinstance(program_or_digest, str) else program_digest(program_or_digest)
    qid = query_id(query)
    doc = {"schema": QUERY_SCHEMA, "query_id": qid, "program_digest": pdigest,
           "binding_id": binding_id(qid, pdigest), "query": query_json(query)}
    if provenance:
        doc["provenance"] = dict(provenance)
    return doc


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def use_document(bid: str, verdict: Any, w_max: int = 32, kind: str = USE_KIND_VU_PARTITION,
                 evaluator_version: str = EVALUATOR_VERSION, utc: str | None = None) -> dict:
    """The `verity-ir/query-use/v1` document: binding, use configuration, verdict, evaluator."""
    _check_hex(bid, "binding_id")
    v = verdict.to_json() if hasattr(verdict, "to_json") else dict(verdict)
    return {"schema": USE_SCHEMA, "binding_id": bid, "use": {"kind": kind, "w_max": int(w_max)}, "verdict": v,
            "evaluator": {"version": evaluator_version, "utc": utc or utc_now()}}


def vu_use_document(query: Any, program: Any, verdict: Any, w_max: int = 32, utc: str | None = None) -> tuple[dict, dict]:
    """(query document, use document) for a VU-partition use of `query` over `program` with `verdict`."""
    qdoc = query_document(query, program)
    return qdoc, use_document(qdoc["binding_id"], verdict, w_max=w_max, utc=utc)


def check_query_document(doc: dict) -> list[str]:
    """Structural problems of a query document (empty list = well-formed and internally consistent)."""
    problems: list[str] = []
    if doc.get("schema") != QUERY_SCHEMA:
        problems.append(f"schema is {doc.get('schema')!r}, expected {QUERY_SCHEMA!r}")
    for k in ("query_id", "program_digest", "binding_id", "query"):
        if k not in doc:
            problems.append(f"missing {k!r}")
    for k in ("kind", "w_max", "purpose", "use"):
        if k in doc:
            problems.append(f"query document carries {k!r}; use configuration belongs to the use document")
    if problems:
        return problems
    try:
        if query_id(doc["query"]) != doc["query_id"]:
            problems.append("query_id does not match the query AST")
        if binding_id(doc["query_id"], doc["program_digest"]) != doc["binding_id"]:
            problems.append("binding_id does not match sha512(schema || query_id || program_digest)")
    except ValueError as e:
        problems.append(str(e))
    return problems


def check_use_document(doc: dict, program_digest_expected: str | None = None, query_doc: dict | None = None) -> list[str]:
    """Structural problems of a use document; with `program_digest_expected` / `query_doc`, also the binding."""
    problems: list[str] = []
    if doc.get("schema") != USE_SCHEMA:
        problems.append(f"schema is {doc.get('schema')!r}, expected {USE_SCHEMA!r}")
    for k in ("binding_id", "use", "verdict", "evaluator"):
        if k not in doc:
            problems.append(f"missing {k!r}")
    if problems:
        return problems
    use = doc["use"]
    if not isinstance(use, dict) or "kind" not in use:
        problems.append("use must be a dict with a kind")
    elif use["kind"] == USE_KIND_VU_PARTITION and not isinstance(use.get("w_max"), int):
        problems.append("a vu-partition use needs an integer w_max")
    verdict = doc["verdict"]
    if not isinstance(verdict, dict) or "status" not in verdict:
        problems.append("verdict must be a dict with a status")
    ev = doc["evaluator"]
    if not isinstance(ev, dict) or "version" not in ev or "utc" not in ev:
        problems.append("evaluator must carry version and utc")
    if query_doc is not None:
        if query_doc.get("binding_id") != doc["binding_id"]:
            problems.append("binding_id differs from the query document's")
        if program_digest_expected is not None and query_doc.get("program_digest") != program_digest_expected:
            problems.append("the query document is bound to a different program_digest")
    return problems


__all__ = ["QUERY_SCHEMA", "USE_SCHEMA", "USE_KIND_VU_PARTITION", "EVALUATOR_VERSION", "program_digest", "query_json",
           "query_id", "binding_id", "query_document", "use_document", "vu_use_document", "check_query_document",
           "check_use_document", "utc_now"]
