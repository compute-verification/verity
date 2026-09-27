"""`fixtures/artifacts.json` (store README §7): every repo replica this repository advertises hashes to the art id it is registered
under.  A repo-registered `fixture/v1` carries no meta and no refs (identity is the bytes and the bare file name -- rule 6: content,
not pathname), so the manifest, and with it the id, is rebuilt here from the file alone; no store state is needed.  When a fixture's
bytes change on purpose, `research data refresh-fixtures` re-registers it and keeps the superseded id under the map's `previous`
block, which is not an entry."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MAP = REPO / "fixtures" / "artifacts.json"
MAX_BYTES = 20 << 20
PREVIOUS = "previous"


def _doc() -> dict:
    if not MAP.is_file():
        pytest.skip(f"{MAP.relative_to(REPO)} absent")
    doc = json.loads(MAP.read_text(encoding="utf-8"))
    assert isinstance(doc, dict) and doc, "the replica map is a non-empty {art id: repo-relative path} object"
    return doc


def _entries() -> dict[str, str]:
    entries = {k: v for k, v in _doc().items() if k != PREVIOUS}
    assert entries, "the replica map has no entries"
    return entries


def test_replica_map_entries_hash_to_their_ids() -> None:
    try:
        from research.store import ids
        from research.store.model import ArtifactManifest, Payload, TreeFile
    except ImportError as e:  # pragma: no cover
        pytest.skip(f"research.store not importable: {e}")
    problems = []
    for art, rel in _entries().items():
        assert ids.is_art(art), f"{art!r} is not an art: id"
        assert isinstance(rel, str) and rel and not rel.startswith("/") and ".." not in rel.split("/"), f"{art}: bad path {rel!r}"
        p = REPO / rel
        if p.is_file():
            data = p.read_bytes()
            payload = Payload.file(hashlib.sha256(data).hexdigest(), len(data), p.name)
            if len(data) > MAX_BYTES:
                problems.append(f"{rel}: {len(data)} bytes > 20 MB; large arrays stay out of the repo replica map")
        elif p.is_dir():
            files = []
            for f in sorted(x for x in p.rglob("*") if x.is_file()):
                b = f.read_bytes()
                files.append(TreeFile(f.relative_to(p).as_posix(), hashlib.sha256(b).hexdigest(), len(b)))
            payload = Payload.tree(files)
        else:
            problems.append(f"{art}: {rel} is not in the working tree")
            continue
        got = ArtifactManifest(kind="fixture/v1", meta={}, refs={}, payload=payload).validate().id
        if got != art:
            problems.append(f"{rel}: hashes to {got}, registered as {art} (`research data refresh-fixtures --preserve` re-registers it)")
    assert not problems, "\n".join(problems)


def test_replica_map_previous_block_is_history_of_current_entries() -> None:
    """`previous` (written by `refresh-fixtures`) maps a current entry's id to the ids its path hashed to before: art ids, each once,
    never the current id itself; nothing else lives outside the entries."""
    try:
        from research.store import ids
    except ImportError as e:  # pragma: no cover
        pytest.skip(f"research.store not importable: {e}")
    doc = _doc()
    prev = doc.get(PREVIOUS, {})
    assert isinstance(prev, dict), "`previous` is an object {current art id: [superseded art ids]}"
    for art, olds in prev.items():
        assert art in doc and ids.is_art(art), f"previous[{art!r}] is not a current entry"
        assert isinstance(olds, list) and olds and all(ids.is_art(o) for o in olds), f"previous[{art}] must be a non-empty list of art ids"
        assert len(set(olds)) == len(olds) and art not in olds, f"previous[{art}] lists an id twice or lists itself"


def test_replica_map_paths_are_tracked() -> None:
    import subprocess

    tracked = set(subprocess.run(["git", "ls-files", "-z"], cwd=REPO, check=False, capture_output=True).stdout.decode().split("\0"))
    if not tracked - {""}:
        pytest.skip("not a git checkout")
    untracked = [rel for rel in _entries().values() if rel not in tracked and not any(t.startswith(rel.rstrip("/") + "/") for t in tracked)]
    assert not untracked, "replica map names paths git does not track (a fresh clone would not hold the bytes):\n" + "\n".join(untracked)
