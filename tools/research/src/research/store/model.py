"""Envelopes (README §2, §4, §8): the artifact manifest, its payload, labels, and the top-level shape of an attempt.

The store validates envelopes only.  `ArtifactManifest.validate()` enforces §2: schema, kind syntax, `meta` canonical size
<= 256 KiB, `refs` values are `art:` ids or lists of them, payload file/tree rules (relative sorted unique paths, no `..`,
no absolute, no path that is both a file and a directory, tree sha256 = sha256(canon(files)), bytes = sum).  Attempts are plain
dicts built by `research run` (the other half of this package); `validate_attempt_doc` checks schema, id and the §4 keys.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from . import canon, ids, kinds

ARTIFACT_SCHEMA = "research/artifact/v0.1"
ATTEMPT_SCHEMA = "research/attempt/v0.1"
LABEL_SCHEMA = "research/label/v0.1"
META_MAX_BYTES = 256 * 1024
ATTEMPT_KEYS = ("tool", "derivation", "reuse_key", "argv", "params", "key_params", "inputs", "conditions", "key_conditions", "source",
                "closure", "execution", "cost", "status", "outputs", "telemetry")
ATTEMPT_NULLABLE = ("reuse_key", "closure", "cost", "telemetry")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _path_errors(path: Any, where: str) -> list[str]:
    if not isinstance(path, str) or not path:
        return [f"{where}: path must be a non-empty string, got {path!r}"]
    if path.startswith("/") or "\\" in path:
        return [f"{where}: path must be relative with '/' separators: {path!r}"]
    parts = path.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return [f"{where}: path has an empty, '.' or '..' component: {path!r}"]
    return []


@dataclass
class TreeFile:
    path: str
    sha256: str
    bytes: int

    def to_doc(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


@dataclass
class Payload:
    """`{"type": "file", sha256, bytes, name}` or `{"type": "tree", sha256, bytes, files}`."""

    type: str
    sha256: str
    bytes: int
    name: str | None = None
    files: list[TreeFile] | None = None

    @staticmethod
    def tree_digest(files: list[TreeFile]) -> str:
        return canon.sha256([f.to_doc() for f in files])

    @classmethod
    def file(cls, sha256: str, size: int, name: str) -> Payload:
        return cls(type="file", sha256=sha256, bytes=size, name=name)

    @classmethod
    def tree(cls, files: list[TreeFile]) -> Payload:
        files = sorted(files, key=lambda f: f.path)
        return cls(type="tree", sha256=cls.tree_digest(files), bytes=sum(f.bytes for f in files), files=files)

    def blobs(self) -> list[tuple[str, int]]:
        """(sha256, bytes) of every blob this payload needs."""
        if self.type == "file":
            return [(self.sha256, self.bytes)]
        return [(f.sha256, f.bytes) for f in self.files or []]

    def to_doc(self) -> dict:
        if self.type == "file":
            return {"type": "file", "sha256": self.sha256, "bytes": self.bytes, "name": self.name}
        return {"type": "tree", "sha256": self.sha256, "bytes": self.bytes, "files": [f.to_doc() for f in self.files or []]}

    @classmethod
    def from_doc(cls, doc: dict) -> Payload:
        if not isinstance(doc, dict) or doc.get("type") not in ("file", "tree"):
            raise ValueError(f"payload must be a file or tree object, got {doc!r}")
        if doc["type"] == "file":
            return cls(type="file", sha256=doc.get("sha256"), bytes=doc.get("bytes"), name=doc.get("name"))
        files = doc.get("files")
        if not isinstance(files, list) or not all(isinstance(f, dict) for f in files):
            raise ValueError("payload.files must be a list of objects")
        return cls(type="tree", sha256=doc.get("sha256"), bytes=doc.get("bytes"),
                   files=[TreeFile(path=f.get("path"), sha256=f.get("sha256"), bytes=f.get("bytes")) for f in files])

    def errors(self) -> list[str]:
        errs: list[str] = []
        if not isinstance(self.bytes, int) or isinstance(self.bytes, bool) or self.bytes < 0:
            errs.append("payload.bytes must be a non-negative int")
        if not ids.is_sha256(self.sha256):
            errs.append("payload.sha256 must be 64 lowercase hex digits")
        if self.type == "file":
            if not isinstance(self.name, str) or not self.name or "/" in self.name or self.name in (".", ".."):
                errs.append(f"payload.name must be a bare file name, got {self.name!r}")
            return errs
        if self.type != "tree":
            return [f"payload.type must be 'file' or 'tree', got {self.type!r}"]
        files = self.files or []
        paths = [f.path for f in files]
        for i, f in enumerate(files):
            errs += _path_errors(f.path, f"payload.files[{i}]")
            if not ids.is_sha256(f.sha256):
                errs.append(f"payload.files[{i}].sha256 must be 64 lowercase hex digits")
            if not isinstance(f.bytes, int) or isinstance(f.bytes, bool) or f.bytes < 0:
                errs.append(f"payload.files[{i}].bytes must be a non-negative int")
        if errs:
            return errs
        if paths != sorted(paths):
            errs.append("payload.files must be sorted by path")
        if len(set(paths)) != len(paths):
            errs.append("payload.files has duplicate paths")
        dirs = {p.rsplit("/", 1)[0] for p in paths if "/" in p}
        prefixes = set()
        for d in dirs:
            parts = d.split("/")
            prefixes.update("/".join(parts[: i + 1]) for i in range(len(parts)))
        clash = sorted(prefixes & set(paths))
        if clash:
            errs.append(f"payload.files: {clash[0]!r} is both a file and a directory")
        if self.bytes != sum(f.bytes for f in files):
            errs.append("payload.bytes must equal the sum of file sizes")
        if self.sha256 != self.tree_digest(files):
            errs.append("payload.sha256 must equal sha256(canon(files))")
        return errs


def _ref_errors(refs: Any) -> list[str]:
    if not isinstance(refs, dict):
        return ["refs must be an object"]
    errs = []
    for name, v in refs.items():
        if not isinstance(name, str) or not name:
            errs.append(f"refs: name {name!r} must be a non-empty string")
        elif ids.is_art(v):
            continue
        elif isinstance(v, list) and all(ids.is_art(x) for x in v):
            continue
        else:
            errs.append(f"refs.{name}: must be an art: id or a list of them, got {v!r}")
    return errs


@dataclass
class ArtifactManifest:
    """`research/artifact/v0.1`.  Identity (`id`) is the sha256 of `to_doc()`; producer, time and labels are not inside."""

    kind: str
    meta: dict = field(default_factory=dict)
    refs: dict = field(default_factory=dict)
    payload: Payload | None = None
    schema: str = ARTIFACT_SCHEMA

    def to_doc(self) -> dict:
        return {"schema": self.schema, "kind": self.kind, "meta": self.meta, "refs": self.refs,
                "payload": self.payload.to_doc() if self.payload else None}

    @classmethod
    def from_doc(cls, doc: dict) -> ArtifactManifest:
        if not isinstance(doc, dict):
            raise ValueError("manifest must be an object")
        payload = doc.get("payload")
        return cls(schema=doc.get("schema"), kind=doc.get("kind"), meta=doc.get("meta"), refs=doc.get("refs"),
                   payload=Payload.from_doc(payload) if payload is not None else None)

    @property
    def id(self) -> str:
        return ids.art_id(self.to_doc())

    def canonical(self) -> bytes:
        return canon.dumps(self.to_doc())

    def blobs(self) -> list[tuple[str, int]]:
        return self.payload.blobs() if self.payload else []

    def errors(self) -> list[str]:
        errs: list[str] = []
        if self.schema != ARTIFACT_SCHEMA:
            errs.append(f"schema must be {ARTIFACT_SCHEMA!r}, got {self.schema!r}")
        if not kinds.is_kind(self.kind):
            errs.append(f"kind must look like <name>/v<int>, got {self.kind!r}")
        if not isinstance(self.meta, dict):
            errs.append("meta must be an object")
        else:
            try:
                n = len(canon.dumps(self.meta))
            except ValueError as e:
                errs.append(f"meta is not canonicalisable: {e}")
            else:
                if n > META_MAX_BYTES:
                    errs.append(f"meta is {n} canonical bytes (> {META_MAX_BYTES}); put large data in the payload")
        errs += _ref_errors(self.refs)
        if self.payload is not None:
            errs += self.payload.errors()
        return errs

    def validate(self) -> ArtifactManifest:
        errs = self.errors()
        if errs:
            raise ValueError("invalid artifact manifest: " + "; ".join(errs))
        return self


@dataclass
class Label:
    """One immutable assertion about a target (README §8).  `ts` is `YYYY-MM-DDTHH:MM:SSZ`; `ref` names the evidence."""

    target: str
    key: str
    value: Any
    by: str
    ts: str
    ref: str | None = None
    schema: str = LABEL_SCHEMA

    def to_doc(self) -> dict:
        return {"schema": self.schema, "target": self.target, "key": self.key, "value": self.value, "by": self.by, "ts": self.ts, "ref": self.ref}

    @classmethod
    def from_doc(cls, doc: dict) -> Label:
        return cls(schema=doc.get("schema", LABEL_SCHEMA), target=doc["target"], key=doc["key"], value=doc.get("value"), by=doc["by"],
                   ts=doc["ts"], ref=doc.get("ref"))

    def errors(self) -> list[str]:
        errs = []
        if self.schema != LABEL_SCHEMA:
            errs.append(f"schema must be {LABEL_SCHEMA!r}")
        if ids.kind_of_id(self.target) == "unknown":
            errs.append(f"target must be an art:, drv: or run id, got {self.target!r}")
        for name in ("key", "by", "ts"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                errs.append(f"{name} must be a non-empty string")
        if self.ref is not None and not isinstance(self.ref, str):
            errs.append("ref must be a string or null")
        try:
            canon.check(self.value, "value")
        except ValueError as e:
            errs.append(str(e))
        return errs

    def validate(self) -> Label:
        errs = self.errors()
        if errs:
            raise ValueError("invalid label: " + "; ".join(errs))
        return self


def attempt_errors(doc: Any) -> list[str]:
    """Top-level shape of `research/attempt/v0.1` (README §4).  Contents of the sections are the runner's business."""
    if not isinstance(doc, dict):
        return ["attempt must be an object"]
    errs = []
    if doc.get("schema") != ATTEMPT_SCHEMA:
        errs.append(f"schema must be {ATTEMPT_SCHEMA!r}, got {doc.get('schema')!r}")
    if not ids.is_run_id(doc.get("id")):
        errs.append(f"id must be a run id (rYYYYMMDD-HHMMSS-hhhh), got {doc.get('id')!r}")
    missing = [k for k in ATTEMPT_KEYS if k not in doc]
    if missing:
        errs.append("missing keys: " + ", ".join(missing))
    tool = doc.get("tool")
    if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
        errs.append("tool must be {name: str, version: str|null}")
    if not ids.is_drv(doc.get("derivation")):
        errs.append(f"derivation must be a drv: id, got {doc.get('derivation')!r}")
    rk = doc.get("reuse_key")
    if rk is not None and not ids.is_sha256(rk):
        errs.append("reuse_key must be 64 hex digits or null")
    for k in ("params", "inputs", "conditions", "source", "execution", "status", "outputs"):
        if k in doc and not isinstance(doc[k], dict):
            errs.append(f"{k} must be an object")
    for k in ATTEMPT_NULLABLE:
        if k in doc and doc[k] is not None and k != "reuse_key" and not isinstance(doc[k], dict):
            errs.append(f"{k} must be an object or null")
    for k in ("inputs", "outputs"):
        for name, v in (doc.get(k) or {}).items() if isinstance(doc.get(k), dict) else []:
            if v is not None and not ids.is_art(v):
                errs.append(f"{k}.{name} must be an art: id, got {v!r}")
    try:
        canon.check(doc)
    except ValueError as e:
        errs.append(f"not canonicalisable: {e}")
    return errs


def validate_attempt_doc(doc: Any) -> dict:
    errs = attempt_errors(doc)
    if errs:
        raise ValueError("invalid attempt manifest: " + "; ".join(errs))
    return doc
