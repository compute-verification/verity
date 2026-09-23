"""Repo replicas (README §7) and `research data refresh-fixtures`: keeping `fixtures/artifacts.json` honest when a fixture's bytes change.

A repo-registered fixture has *content-only identity*: kind `fixture/v1`, `meta {}`, `refs {}`, payload = the file (bare name) or
the directory as a tree, so its id is a function of the bytes alone and `tools/research/tests/test_repo_replicas.py` recomputes it
with no store state.  Edit the bytes and the id changes; until today that meant a `research data put --kind fixture/v1 --file …` by
hand and an edit of the map.  `refresh` does exactly that for every entry whose path no longer hashes to its id: `put` under the new
id (`push --preserve` on request), rewrite the entry, and record the superseded id under the map's `previous` block:

    {"art:<new>": "fixtures/x.json", ..., "previous": {"art:<new>": ["art:<oldest>", ..., "art:<old>"]}}

`LocalStore.replica_map()` reads `art:` keys only, so the block is invisible to `fetch` in every worktree.  Idempotent: nothing is
put or written when every entry already hashes to its id.  `dry_run` computes and reports, writes nothing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import ids
from .local import REPLICA_MAP, LocalStore, _rel_files, sha256_file
from .model import ArtifactManifest, Payload, TreeFile

FIXTURE_KIND = "fixture/v1"
PREVIOUS = "previous"


def fixture_manifest(path: Path) -> ArtifactManifest:
    """The manifest a repo replica is registered under (content-only identity); FileNotFoundError when `path` is neither file nor dir."""
    path = Path(path)
    if path.is_file():
        h, n = sha256_file(path)
        payload = Payload.file(h, n, path.name)
    elif path.is_dir():
        files = []
        for p in _rel_files(path):
            h, n = sha256_file(p)
            files.append(TreeFile(p.relative_to(path).as_posix(), h, n))
        payload = Payload.tree(files)
    else:
        raise FileNotFoundError(str(path))
    return ArtifactManifest(kind=FIXTURE_KIND, meta={}, refs={}, payload=payload).validate()


def fixture_id(path: Path) -> str:
    return fixture_manifest(path).id


def load_map(repo: Path) -> dict[str, Any]:
    p = Path(repo) / REPLICA_MAP
    doc = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{p}: the replica map must be a JSON object")
    return doc


def entries(doc: dict[str, Any]) -> dict[str, str]:
    """The `art id -> repo-relative path` entries of a map document; the `previous` block (or anything else) is not an entry."""
    return {k: v for k, v in doc.items() if ids.is_art(k) and isinstance(v, str)}


def previous(doc: dict[str, Any]) -> dict[str, list[str]]:
    blk = doc.get(PREVIOUS)
    return {k: list(v) for k, v in blk.items() if isinstance(v, list)} if isinstance(blk, dict) else {}


def _history(prev: list[str], old: str, new: str) -> list[str]:
    """The `previous` list for `new`: every id the path has hashed to before, oldest first, each once, never `new` itself (bytes can revert)."""
    out: list[str] = []
    for x in [*prev, old]:
        if x != new and x not in out:
            out.append(x)
    return out


def dump_map(doc: dict[str, Any]) -> str:
    """The map as the repo writes it: two-space indent, one entry per line, trailing newline; `previous` last."""
    body = {k: v for k, v in doc.items() if k != PREVIOUS}
    if doc.get(PREVIOUS):
        body[PREVIOUS] = doc[PREVIOUS]
    return json.dumps(body, indent=2, ensure_ascii=False) + "\n"


def refresh(store: LocalStore, repo: Path, *, preserve: bool = False, dry_run: bool = False) -> dict[str, Any]:
    """Recompute every entry's id from its bytes; put + rewrite the changed ones.  Returns the report (`entries`: one per map entry with
    `status` unchanged | changed | missing | conflict, `new_id`, `previous`, `preserved`, `errors`; `changed`, `missing`, `written`)."""
    repo = Path(repo)
    doc = load_map(repo)
    prev = previous(doc)
    rep: dict[str, Any] = {"repo": str(repo), "map": str(repo / REPLICA_MAP), "dry_run": dry_run, "preserve": preserve, "entries": [],
                           "changed": 0, "missing": 0, "unchanged": 0, "written": False}
    new_doc: dict[str, Any] = {}
    new_prev: dict[str, list[str]] = dict(prev)
    for k, v in doc.items():
        if k == PREVIOUS:
            continue
        if not (ids.is_art(k) and isinstance(v, str)):
            new_doc[k] = v                                          # not an entry: carried over untouched
            continue
        path = repo / v
        e: dict[str, Any] = {"id": k, "path": v, "status": None}
        try:
            m = fixture_manifest(path)
        except FileNotFoundError:
            e["status"] = "missing"
            rep["missing"] += 1
            rep["entries"].append(e)
            new_doc[k] = v
            continue
        if m.id == k:
            e["status"] = "unchanged"
            rep["unchanged"] += 1
            rep["entries"].append(e)
            new_doc[k] = v
            continue
        if m.id in new_doc or m.id in entries(doc):
            e.update(status="conflict", new_id=m.id, errors=[f"{v} now hashes to {m.id}, which is already registered as {doc.get(m.id, new_doc.get(m.id))}"])
            rep["entries"].append(e)
            new_doc[k] = v
            continue
        e.update(status="changed", new_id=m.id, previous=_history(prev.get(k, []), k, m.id), preserved=None, errors=[])
        if not dry_run:
            got = store.put_artifact(FIXTURE_KIND, {}, None, file=path if path.is_file() else None, tree=path if path.is_dir() else None)
            if got != m.id:                                         # the same computation twice; anything else is a bug worth stopping on
                raise RuntimeError(f"{v}: put_artifact returned {got}, expected {m.id}")
            if preserve:
                try:
                    push = store.push(m.id, preserve=True)
                    e["preserved"], e["errors"] = bool(push["preserved"]), list(push["errors"])
                except Exception as ex:  # noqa: BLE001  one entry's vault trouble is reported; the map is still made honest
                    e["preserved"], e["errors"] = False, [f"{type(ex).__name__}: {ex}"]
        new_doc[m.id] = v
        new_prev.pop(k, None)
        new_prev[m.id] = e["previous"]
        rep["changed"] += 1
        rep["entries"].append(e)
    if new_prev:
        new_doc[PREVIOUS] = new_prev
    if rep["changed"] and not dry_run:
        (repo / REPLICA_MAP).write_text(dump_map(new_doc), encoding="utf-8")
        rep["written"] = True
    return rep
