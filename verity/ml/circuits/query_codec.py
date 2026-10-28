"""JSON codec for the structural query AST (`verity.ml.circuits.query_ast`) and `query_id`.

    to_json(query)   -> {"op": ..., ...}        one object per constructor; paths / selectors as their canonical strings
    from_json(doc)   -> Query                   inverse
    query_id(query)  -> sha512 hex of canonical_json(to_json(query)); program-independent identity of the query text

`SCHEMA` (`verity-ir/query/v1`) names the wrapping artifact, which `verity_vllm.query.query_artifact` writes.  The query
document carries no purpose (`kind` / `w_max`) -- that is the use configuration, not the query.
"""
from __future__ import annotations

import hashlib

from .codec import canonical_json
from .query_ast import (Concat, Default, Each, Instances, Inter, Members, Minus, NameSel, Nodes, NodeSel, Override, Path,
                      Pred, Query, QueryError, Rest, Root, Union, Where, Zip)

SCHEMA = "verity-ir/query/v1"


def to_json(q) -> dict:
    """Query AST -> plain JSON object.  Also accepts an already-encoded document (returned as is) and any object with a
    `to_json()` method (duck-typed, for artifact writers that wrap a query)."""
    if isinstance(q, dict):
        return q
    if not isinstance(q, Query) and hasattr(q, "to_json"):
        return to_json(q.to_json())
    if isinstance(q, Instances):
        return {"op": "instances", "path": str(q.path)}
    if isinstance(q, Members):
        return {"op": "members", "path": str(q.path), "sel": str(q.sel), "k": q.k}
    if isinstance(q, Nodes):
        return {"op": "nodes", "path": str(q.path), "sel": str(q.sel)}
    if isinstance(q, Zip):
        return {"op": "zip", "path": str(q.path), "sels": [str(s) for s in q.sels]}
    if isinstance(q, Each):
        return {"op": "each", "base": to_json(q.base), "inner": to_json(q.inner)}
    if isinstance(q, Concat):
        return {"op": "concat", "parts": [to_json(p) for p in q.parts]}
    if isinstance(q, Rest):
        return {"op": "rest", "base": None if q.base is None else to_json(q.base)}
    if isinstance(q, Default):
        return {"op": "default", "w": q.w}
    if isinstance(q, Override):
        return {"op": "override", "base": to_json(q.base), "cases": [{"path": str(p), "query": to_json(g)} for p, g in q.cases]}
    if isinstance(q, Where):
        pred: dict = {}
        if q.pred.spec is not None:
            pred["spec"] = str(q.pred.spec)
        if q.pred.form is not None:
            pred["form"] = q.pred.form
        if q.pred.w_max is not None:
            pred["w_max"] = q.pred.w_max
        if q.pred.w_min is not None:
            pred["w_min"] = q.pred.w_min
        return {"op": "where", "base": to_json(q.base), "pred": pred}
    if isinstance(q, Union):
        return {"op": "union", "a": to_json(q.a), "b": to_json(q.b)}
    if isinstance(q, Minus):
        return {"op": "minus", "a": to_json(q.a), "b": to_json(q.b)}
    if isinstance(q, Inter):
        return {"op": "inter", "a": to_json(q.a), "b": to_json(q.b)}
    if isinstance(q, Root):
        return {"op": "root"}
    raise TypeError(f"cannot encode {type(q).__name__} as a query document (not a Query, dict or to_json() object)")


def from_json(doc: dict) -> Query:
    """Decode a query document (the `to_json` shape).  Malformed documents raise `QueryError` (missing / mistyped
    fields included), never KeyError / TypeError."""
    if not isinstance(doc, dict) or "op" not in doc:
        raise QueryError(f"not a query document: {doc!r}")
    try:
        return _decode(doc)
    except QueryError:
        raise
    except KeyError as e:
        raise QueryError(f"query document {doc.get('op')!r} is missing field {e.args[0]!r}") from None
    except (TypeError, ValueError, AttributeError) as e:
        raise QueryError(f"malformed query document {doc.get('op')!r}: {e}") from None


def _decode(doc: dict) -> Query:
    op = doc["op"]
    if op == "instances":
        return Instances(Path.parse(doc["path"]))
    if op == "members":
        return Members(Path.parse(doc["path"]), NodeSel.parse(doc["sel"]), int(doc.get("k", 1)))
    if op == "nodes":
        return Nodes(Path.parse(doc["path"]), NodeSel.parse(doc["sel"]))
    if op == "zip":
        return Zip(Path.parse(doc["path"]), tuple(NodeSel.parse(s) for s in doc["sels"]))
    if op == "each":
        return Each(from_json(doc["base"]), from_json(doc["inner"]))
    if op == "concat":
        return Concat(tuple(from_json(p) for p in doc["parts"]))
    if op == "rest":
        b = doc.get("base")
        return Rest(None if b is None else from_json(b))
    if op == "default":
        return Default(int(doc["w"]))
    if op == "override":
        return Override(from_json(doc["base"]), tuple((Path.parse(c["path"]), from_json(c["query"])) for c in doc["cases"]))
    if op == "where":
        p = doc.get("pred", {})
        pred = Pred(None if "spec" not in p else NameSel.parse(p["spec"]), p.get("form"),
                    None if p.get("w_max") is None else int(p["w_max"]), None if p.get("w_min") is None else int(p["w_min"]))
        return Where(from_json(doc["base"]), pred)
    if op == "union":
        return Union(from_json(doc["a"]), from_json(doc["b"]))
    if op == "minus":
        return Minus(from_json(doc["a"]), from_json(doc["b"]))
    if op == "inter":
        return Inter(from_json(doc["a"]), from_json(doc["b"]))
    if op == "root":
        return Root()
    raise QueryError(f"unknown query op {op!r}")


def query_id(q) -> str:
    """SHA-512 of the canonical JSON of the AST: independent of the Program, stable across spellings that parse to the
    same AST (paths and selectors are re-rendered canonically).  Accepts a Query, an encoded document, or an object with
    `to_json()`; a document is hashed as given (callers who want normalisation pass `from_json(doc)`)."""
    return hashlib.sha512(canonical_json(to_json(q))).hexdigest()


def query_document(q: Query) -> dict:
    """The bare query artifact (no binding, no purpose): `{"schema", "query_id", "query"}`."""
    return {"schema": SCHEMA, "query_id": query_id(q), "query": to_json(q)}


__all__ = ["SCHEMA", "to_json", "from_json", "query_id", "query_document"]
