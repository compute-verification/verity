"""`verity/annotations/v0`: names, descriptions, formats and groupings of a program's parts, keyed by ids the program already
has, outside every digest (prototype; see the Project doc ``docs/workstreams.md``, "Annotations").

    {"format": "verity/annotations/v0",
     "entries": [
       {"key": "def:RoPE_v1", "name": "RoPE", "description": "rotary position embedding of one token's heads"},
       {"key": "port:RoPE_v1/in/x", "name": "heads", "format": "BF16"},
       {"key": "node:AttnBlock_v1/4", "callee": "F32Max_v1", "name": "running max"},
       {"key": "wire:AttnBlock_v1/7/1", "name": "scaled scores"},
       {"key": "group:softmax", "members": ["node:AttnBlock_v1/4", "node:AttnBlock_v1/5"], "name": "softmax"},
       {"key": "call:12", "program": "<SHA-512 of the program>", "name": "layer 0, q/k/v projection"}]}

**Keys.**  ``D`` is a Definition as the descriptor names it: a full descriptor id (``RoPE_v1{NHEADS=2,D=16}``, one
specialization) or its stem ``Name_vN`` (every specialization in the program).

* ``def:D``; ``node:D/k`` (body node k); ``wire:D/k/a`` (argument a of node k: the values it reads) or ``wire:D/ret``;
  ``port:D/in/p`` (parameter p, by index or name) or ``port:D/out`` / ``port:D/out/a:b`` (result leaves [a, b));
* ``call:k``, a Call of one program (root body node k, not an Input node), with the entry's ``program`` digest;
* ``group:<slug>``, a named set: ``members`` lists keys (groups included), each of which must resolve.

A stem key must resolve in every specialization the program has.  ``callee`` (on a node or wire key: the stem of node k's callee)
and ``digest`` (on a ``def`` key with a full id: `definition_digest`) are staleness guards: the key still resolves, but the
entry was written against another body.

**Digests.**  Annotations never enter a program digest: `attach` puts them in the descriptor's ``annotations`` field, which
`codec.program_digest` and `partition_object.program_sha512` exclude (``PROTOCOL.md`` §1).  Validation reads only the descriptor's
structure; it evaluates nothing.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .codec import canonical_json
from .program import INPUT_ID

FORMAT = "verity/annotations/v0"
TEXT_FIELDS = ("name", "description", "format")
FIELDS = {"key", "members", "callee", "digest", "program", "aliases", *TEXT_FIELDS}
_KIND = re.compile(r"^(def|node|wire|port|call|group):(.+)$")
_STEM = re.compile(r"^[A-Za-z0-9_\[\]]+_v[1-9][0-9]*$")
_SHA512 = re.compile(r"^[0-9a-f]{128}$")


def _resolved(desc: Mapping[str, Any], entry: Mapping[str, Any]) -> dict:
    """A definition's descriptor entry with the v1 interning tables resolved (callee ids and types inline)."""
    if "callees" not in desc or entry.get("form") != "composite":
        return dict(entry)
    callees, types = desc["callees"], desc["types"]
    nodes = [dict(n, fn=callees[n["fn"]], out=types[n["out"]], args=[dict(a, type=types[a["type"]]) for a in n["args"]])
             for n in entry["body"]["nodes"]]
    return dict(entry, body=dict(entry["body"], nodes=nodes))


def definition_digest(desc: Mapping[str, Any], fid: str) -> str:
    """SHA-256 of the definition's descriptor entry, interning resolved: the same for one definition in any program."""
    return hashlib.sha256(canonical_json(_resolved(desc, desc["definitions"][fid]))).hexdigest()


def _stem(fid: str) -> str:
    return fid.split("{", 1)[0]


def _specs(desc: Mapping[str, Any], d: str) -> list[str]:
    defs = desc["definitions"]
    if d in defs:
        return [d]
    return [k for k, e in defs.items() if _stem(k) == d] if _STEM.match(d) else []


def _callee(desc: Mapping[str, Any], node: Mapping[str, Any]) -> str:
    fn = node["fn"]
    return desc["callees"][fn] if "callees" in desc else fn


@dataclass
class Verdict:
    ok: bool
    errors: list[dict] = field(default_factory=list)     # malformed or unresolvable entries
    stale: list[dict] = field(default_factory=list)      # a guard (callee, digest) no longer matches
    absent: list[str] = field(default_factory=list)      # keys of Definitions this program does not have


def _part(desc: Mapping[str, Any], fid: str, kind: str, rest: str) -> str | None:
    """Why ``kind:fid/rest`` does not resolve in the definition ``fid``, or None."""
    e = desc["definitions"][fid]
    if kind == "def":
        return None
    primitive = e.get("form") != "composite"
    if primitive and kind in ("node", "wire"):
        return f"{fid} is a primitive: it has no body"
    body = {"params": e["params"], "ret_type": e["ret"], "nodes": []} if primitive else e["body"]
    nodes = body["nodes"]
    if kind == "node":
        return None if rest.isdigit() and int(rest) < len(nodes) else f"{fid} has no node {rest}"
    if kind == "wire":
        if rest == "ret":
            return None
        m = re.fullmatch(r"(\d+)/(\d+)", rest)
        if not m or int(m[1]) >= len(nodes) or int(m[2]) >= len(nodes[int(m[1])]["args"]):
            return f"{fid} has no argument {rest}"
        return None
    m = re.fullmatch(r"in/(.+)", rest)
    if m:
        names = [p[0] for p in body["params"]]
        ok = (m[1].isdigit() and int(m[1]) < len(names)) or m[1] in names
        return None if ok else f"{fid} has no parameter {m[1]}"
    m = re.fullmatch(r"out(?:/(\d+):(\d+))?", rest)
    if m:
        if m[1] is None:
            return None
        from .types import type_from_json
        n = type_from_json(body["ret_type"]).leaves
        return None if int(m[1]) < int(m[2]) <= n else f"{fid} returns {n} leaves, not [{m[1]}, {m[2]})"
    return f"port {rest!r}: in/<parameter> or out[/a:b]"


def _split(kind: str, target: str) -> tuple[str, str]:
    """(D, rest) of a node/wire/port target; D may hold '/' only inside its binding record."""
    pat = {"node": r"(.+)/(\d+)", "wire": r"(.+)/(\d+/\d+|ret)", "port": r"(.+)/(in/[^/]+|out(?:/\d+:\d+)?)"}[kind]
    m = re.fullmatch(pat, target)
    return (m[1], m[2]) if m else (target, "")


def validate(annotations: Mapping[str, Any], desc: Mapping[str, Any], program_sha512: str | None = None) -> Verdict:
    """The annotations against one program's descriptor: every entry well formed, every key resolved (or absent: a Definition the
    program does not have), groups closed, each (key, field) set once, the guards current."""
    v = Verdict(True)

    def err(code: str, **what: Any) -> None:
        v.errors.append({"code": code, **what})

    if annotations.get("format") != FORMAT or set(annotations) != {"format", "entries"} or not isinstance(annotations["entries"], list):
        err("format", got=sorted(annotations) if isinstance(annotations, Mapping) else type(annotations).__name__)
        v.ok = False
        return v
    entries = annotations["entries"]
    keys = {e.get("key") for e in entries if isinstance(e, Mapping)}
    seen: dict[tuple[str, str], int] = {}
    calls = {k for k, n in enumerate(desc["definitions"][desc["root"]]["body"]["nodes"]) if not INPUT_ID.match(_callee(desc, n))}
    for i, e in enumerate(entries):
        if not isinstance(e, Mapping) or not isinstance(e.get("key"), str) or set(e) - FIELDS:
            err("entry", entry=i, extra=sorted(set(e) - FIELDS) if isinstance(e, Mapping) else None)
            continue
        key = e["key"]
        if any(not isinstance(e.get(f, ""), str) for f in TEXT_FIELDS):
            err("text", key=key)
        for f in (*TEXT_FIELDS, "members"):
            if f in e:
                if (key, f) in seen:
                    err("duplicate", key=key, field=f, entries=[seen[(key, f)], i])
                seen[(key, f)] = i
        m = _KIND.match(key)
        if not m:
            err("key", key=key)
            continue
        kind, target = m[1], m[2]
        if kind == "group":
            members = e.get("members")
            if not isinstance(members, list) or not members or not all(isinstance(x, str) for x in members):
                err("members", key=key)
            else:
                missing = [x for x in members if x not in keys and not _KIND.match(x)]
                if missing:
                    err("members", key=key, unknown=missing)
            continue
        if kind == "call":
            if not isinstance(e.get("program"), str) or not _SHA512.match(e["program"]):
                err("program", key=key)
            elif program_sha512 is not None and e["program"] == program_sha512 and not (target.isdigit() and int(target) in calls):
                err("unresolved", key=key, why=f"root node {target} is not a Call")
            continue
        d, rest = (target, "") if kind == "def" else _split(kind, target)
        specs = _specs(desc, d)
        if not specs:
            if _STEM.match(d) or "{" in d:
                v.absent.append(key)
            else:
                err("key", key=key)
            continue
        for fid in specs:
            why = _part(desc, fid, kind, rest)
            if why:
                err("unresolved", key=key, definition=fid, why=why)
                break
            if "callee" in e and kind in ("node", "wire") and rest != "ret":
                got = _stem(_callee(desc, desc["definitions"][fid]["body"]["nodes"][int(rest.split("/")[0])]))
                if got != e["callee"]:
                    v.stale.append({"key": key, "definition": fid, "callee": got, "written_for": e["callee"]})
        if "digest" in e and kind == "def" and len(specs) == 1 and definition_digest(desc, specs[0]) != e["digest"]:
            v.stale.append({"key": key, "definition": specs[0], "digest": definition_digest(desc, specs[0])})
    for e in entries:
        if isinstance(e, Mapping) and str(e.get("key", "")).startswith("group:"):
            for x in e.get("members") or ():
                if isinstance(x, str) and x.startswith("group:") and x not in keys:
                    err("members", key=e["key"], unknown=[x])
    v.ok = not v.errors
    return v


def attach(desc: Mapping[str, Any], annotations: Mapping[str, Any]) -> dict:
    """A copy of the descriptor carrying the annotations in its ``annotations`` field (no digest moves)."""
    out = dict(desc)
    out["annotations"] = dict(desc.get("annotations", {}), **{FORMAT: list(annotations["entries"])})
    return out


def merge(*layers: Mapping[str, Any]) -> dict:
    """Annotation files in order, a later (key, field) replacing an earlier one: the repo's files, then a site's pending overlay."""
    by_key: dict[str, dict] = {}
    for layer in layers:
        for e in layer["entries"]:
            by_key.setdefault(e["key"], {"key": e["key"]}).update({k: x for k, x in e.items() if k != "key"})
    return {"format": FORMAT, "entries": list(by_key.values())}


def keys_in(annotations: Mapping[str, Any]) -> Iterable[str]:
    return (e["key"] for e in annotations["entries"])


__all__ = ["FORMAT", "Verdict", "definition_digest", "validate", "attach", "merge", "keys_in"]
