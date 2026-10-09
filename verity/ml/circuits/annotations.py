"""`verity/annotations/v0`: names, descriptions, formats and groupings of a program's parts, keyed by ids the program already
has, outside every digest.

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
* ``group:<slug>``, a named set: ``members`` lists keys, each of which must resolve (a group member must be an entry).

A stem key must resolve in every specialization the program has.  ``callee`` (on a node or wire key: the stem of node k's callee)
and ``digest`` (on a ``def`` key with a full id: `definition_digest`) are staleness guards: the key still resolves, but the entry
was written against another body.  Fields: ``name``, ``description``, ``format`` (strings), ``aliases`` (strings), each set at most
once per key in one file; `merge` layers files, a later file's field replacing an earlier one's.

**Digests.**  Annotations never enter a program digest: `attach` puts them in the descriptor's ``annotations`` field, which
`codec.program_digest` excludes (``PROTOCOL.md`` §1).  Validation reads only the descriptor's
structure; it evaluates nothing.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from .codec import canonical_json
from .program import input_id
from .types import type_from_json

FORMAT = "verity/annotations/v0"
TEXT_FIELDS = ("name", "description", "format")
FIELDS = {"key", "members", "callee", "digest", "program", "aliases", *TEXT_FIELDS}
_KIND = re.compile(r"^(def|node|wire|port|call|group):(.+)$")
_STEM = re.compile(r"^[A-Za-z0-9_\[\]]+_v[1-9][0-9]*$")
_SHA512 = re.compile(r"^[0-9a-f]{128}$")
_TARGET = {"node": r"(.+)/(\d+)", "wire": r"(.+)/(\d+/\d+|ret)", "port": r"(.+)/(in/[^/]+|out(?:/\d+:\d+)?)"}


def _resolved(desc: Mapping[str, Any], entry: Mapping[str, Any]) -> dict:
    """A definition's descriptor entry with the v1 interning tables resolved (callee ids and types inline)."""
    if "callees" not in desc or entry.get("form") != "composite":
        return dict(entry)
    callees, types = desc["callees"], desc["types"]
    nodes = [dict(n, fn=callees[n["fn"]], out=types[n["out"]], args=[dict(a, type=types[a["type"]]) for a in n["args"]])
             for n in entry["body"]["nodes"]]
    return dict(entry, body=dict(entry["body"], nodes=nodes))


def definition_digest(desc: Mapping[str, Any], fid: str) -> str:
    """SHA-512 of the definition's descriptor entry, interning resolved: the same for one definition in any program."""
    return hashlib.sha512(canonical_json(_resolved(desc, desc["definitions"][fid]))).hexdigest()


def _stem(fid: str) -> str:
    return fid.split("{", 1)[0]


def _callee(desc: Mapping[str, Any], node: Mapping[str, Any]) -> str:
    return desc["callees"][node["fn"]] if "callees" in desc else node["fn"]


def _in_definition(desc: Mapping[str, Any], fid: str, kind: str, rest: str) -> str | None:
    """Why ``kind:fid/rest`` names nothing in the definition ``fid``, or None."""
    e = desc["definitions"][fid]
    if kind == "def":
        return None
    primitive = e.get("form") != "composite"
    if primitive and kind in ("node", "wire"):
        return f"{fid} is a primitive: it has no body"
    body = {"params": e["params"], "ret_type": e["ret"], "nodes": []} if primitive else e["body"]
    nodes = body["nodes"]
    if kind == "node":
        return None if int(rest) < len(nodes) else f"{fid} has no node {rest}"
    if kind == "wire":
        if rest == "ret":
            return None
        k, a = map(int, rest.split("/"))
        return None if k < len(nodes) and a < len(nodes[k]["args"]) else f"{fid} has no argument {rest}"
    if rest.startswith("in/"):
        p, names = rest[3:], [n for n, _ in body["params"]]
        return None if (p.isdigit() and int(p) < len(names)) or p in names else f"{fid} has no parameter {p}"
    m = re.fullmatch(r"out(?:/(\d+):(\d+))?", rest)
    n = type_from_json(body["ret_type"]).leaves
    return None if m[1] is None or int(m[1]) < int(m[2]) <= n else f"{fid} returns {n} leaves, not [{m[1]}, {m[2]})"


@dataclass
class Verdict:
    ok: bool
    errors: list[dict] = field(default_factory=list)     # malformed or unresolvable entries
    stale: list[dict] = field(default_factory=list)      # a guard (callee, digest) no longer matches
    absent: list[str] = field(default_factory=list)      # keys of Definitions this program does not have


class _Program:
    """What keys resolve against: one descriptor (and, for ``call`` keys, its program digest)."""

    def __init__(self, desc: Mapping[str, Any], digest: str | None) -> None:
        self.desc, self.digest = desc, digest
        root = desc["definitions"][desc["root"]]["body"]["nodes"]
        self.calls = {k for k, n in enumerate(root) if input_id(_callee(desc, n)) is None}

    def specs(self, d: str) -> list[str]:
        defs = self.desc["definitions"]
        if d in defs:
            return [d]
        return [k for k in defs if _stem(k) == d] if _STEM.fullmatch(d) else []

    def resolve(self, key: str, entry: Mapping[str, Any] | None = None) -> tuple[str, Any]:
        """("ok" | "absent" | "error", detail) for a non-group key; ``entry`` supplies a ``call`` key's program."""
        m = _KIND.fullmatch(key)
        if not m or m[1] == "group":
            return "error", {"code": "key", "key": key}
        kind, target = m[1], m[2]
        if kind == "call":
            program = (entry or {}).get("program")
            if not isinstance(program, str) or not _SHA512.fullmatch(program):
                return "error", {"code": "program", "key": key}
            if self.digest is None or program != self.digest:
                return "ok", []                                          # another program's Call: checked against that program
            ok = target.isdigit() and int(target) in self.calls
            return ("ok", []) if ok else ("error", {"code": "unresolved", "key": key, "why": f"root node {target} is not a Call"})
        if kind == "def":
            d, rest = target, ""
        else:
            t = re.fullmatch(_TARGET[kind], target)
            if not t:
                return "error", {"code": "key", "key": key}
            d, rest = t[1], t[2]
        specs = self.specs(d)
        if not specs:
            return ("absent", key) if _STEM.fullmatch(d) or "{" in d else ("error", {"code": "key", "key": key})
        for fid in specs:
            why = _in_definition(self.desc, fid, kind, rest)
            if why:
                return "error", {"code": "unresolved", "key": key, "definition": fid, "why": why}
        return "ok", specs


def _entry_errors(e: Mapping[str, Any], i: int) -> list[dict]:
    out = []
    if set(e) - FIELDS:
        out.append({"code": "entry", "entry": i, "extra": sorted(set(e) - FIELDS)})
    if any(not isinstance(e[f], str) for f in TEXT_FIELDS if f in e) or \
            ("aliases" in e and not (isinstance(e["aliases"], list) and all(isinstance(x, str) for x in e["aliases"]))):
        out.append({"code": "text", "key": e["key"]})
    return out


def validate(annotations: Mapping[str, Any], desc: Mapping[str, Any], program_digest: str | None = None) -> Verdict:
    """The annotations against one program's descriptor: every entry well formed, every key resolved (or absent: a Definition the
    program does not have), groups closed, each (key, field) set once, the guards current.  ``program_digest`` enables the checks of
    ``call`` keys naming this program."""
    v = Verdict(True)
    if not isinstance(annotations, Mapping) or set(annotations) != {"format", "entries"} or annotations.get("format") != FORMAT \
            or not isinstance(annotations.get("entries"), list):
        v.errors.append({"code": "format"})
        v.ok = False
        return v
    prog = _Program(desc, program_digest)
    entries = annotations["entries"]
    keys = {e["key"] for e in entries if isinstance(e, Mapping) and isinstance(e.get("key"), str)}
    seen: dict[tuple[str, str], int] = {}
    for i, e in enumerate(entries):
        if not isinstance(e, Mapping) or not isinstance(e.get("key"), str):
            v.errors.append({"code": "entry", "entry": i})
            continue
        v.errors += _entry_errors(e, i)
        key = e["key"]
        for f in (*TEXT_FIELDS, "aliases", "members"):
            if f in e:
                if (key, f) in seen:
                    v.errors.append({"code": "duplicate", "key": key, "field": f, "entries": [seen[(key, f)], i]})
                seen[(key, f)] = i
        if key.startswith("group:"):
            members = e.get("members")
            if not isinstance(members, list) or not members or not all(isinstance(x, str) for x in members):
                v.errors.append({"code": "members", "key": key})
                continue
            for x in members:
                if x.startswith("group:"):
                    if x not in keys:
                        v.errors.append({"code": "members", "key": key, "unknown": x})
                    continue
                status, detail = prog.resolve(x, next((y for y in entries if isinstance(y, Mapping) and y.get("key") == x), None))
                if status == "error":
                    v.errors.append({"code": "members", "key": key, "member": detail})
            continue
        status, detail = prog.resolve(key, e)
        if status == "error":
            v.errors.append(detail)
            continue
        if status == "absent":
            v.absent.append(detail)
            continue
        specs = detail
        if "callee" in e and key.startswith(("node:", "wire:")) and not key.endswith("/ret"):
            k = int(re.fullmatch(_TARGET[key[:4]], key[5:])[2].split("/")[0])
            for fid in specs:
                got = _stem(_callee(desc, desc["definitions"][fid]["body"]["nodes"][k]))
                if got != e["callee"]:
                    v.stale.append({"key": key, "definition": fid, "callee": got, "written_for": e["callee"]})
        if "digest" in e and key.startswith("def:") and len(specs) == 1:
            got = definition_digest(desc, specs[0])
            if got != e["digest"]:
                v.stale.append({"key": key, "definition": specs[0], "digest": got})
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


__all__ = ["FORMAT", "Verdict", "definition_digest", "validate", "attach", "merge"]
