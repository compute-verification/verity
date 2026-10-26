"""The store's GC (`research.store.gc`, README §7.4): a blob goes only when uncited artifacts past their kind's TTL alone hold it,
it has been a candidate for a grace, and no manifest listed since holds it; a citation by a label, a citing tree or a cited run's
attempt keeps it; anything it can't read deletes nothing."""
from __future__ import annotations

import hashlib
import os

import pytest

from research.store import canon, gc, ids
from research.store.model import Label
from research.store.remote import FsRemote

NOW = 1_793_000_000.0
DAY = 86400.0


def _put(root, key: str, data: bytes, age_s: float = 0.0) -> None:
    p = root / key
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    os.utime(p, (NOW - age_s, NOW - age_s))


def blob(root, data: bytes) -> tuple[str, int]:
    h = hashlib.sha256(data).hexdigest()
    _put(root, f"objects/sha256/{h}", data)
    return h, len(data)


def artifact(root, kind: str, *blobs: tuple[str, int], age_days: float = 0.0, meta=None, refs=None) -> str:
    files = [{"path": f"f{i}", "sha256": h, "bytes": n} for i, (h, n) in enumerate(blobs)]
    payload = {"type": "tree", "sha256": canon.sha256(files), "bytes": sum(n for _, n in blobs), "files": files} if blobs else None
    doc = {"schema": "research/artifact/v0.1", "kind": kind, "meta": meta or {}, "refs": refs or {}, "payload": payload}
    aid = ids.art_id(doc)
    _put(root, f"manifests/{ids.hex_of(aid)}.json", canon.dumps(doc), age_days * DAY)
    return aid


def attempt(root, rid: str, outputs: dict) -> None:
    _put(root, f"attempts/{rid}.json", canon.dumps({"schema": "research/attempt/v0.1", "id": rid, "outputs": outputs}))


def label(root, target: str, ref=None) -> None:
    lab = Label(target=target, key="note", value="x", by="lane", ts="2026-10-26T00:00:00Z", ref=ref)
    _put(root, f"labels/{ids.target_dirname(target)}/{canon.sha256(lab.to_doc())}.json", canon.dumps(lab.to_doc()))


@pytest.fixture
def env(tmp_path):
    b = tmp_path / "bucket"
    notes = tmp_path / "notes"
    notes.mkdir()
    (b / "labels").mkdir(parents=True)
    return b, notes, tmp_path / "cache", tmp_path / "marks.json"


def present(b, h: str) -> bool:
    return (b / "objects" / "sha256" / h).exists()


def test_an_uncited_old_record_goes_after_a_grace_and_a_young_one_stays(env):
    b, notes, cache, marks = env
    old, young = blob(b, b"old record"), blob(b, b"young record")
    artifact(b, "run-record/v1", old, age_days=8)
    artifact(b, "run-record/v1", young, age_days=2)
    r = FsRemote(b)
    p = gc.plan(r, cache, [notes], now=NOW)
    assert set(p.candidates) == {old[0]} and p.candidates[old[0]] == old[1]
    first = gc.apply(r, p, cache, marks, now=NOW)
    assert first["deleted"] == 0 and first["marked"] == 1 and present(b, old[0])
    early = gc.apply(r, gc.plan(r, cache, [notes], now=NOW + 3600), cache, marks, now=NOW + 3600)
    assert early["deleted"] == 0 and present(b, old[0])
    later = NOW + gc.GRACE_S
    done = gc.apply(r, gc.plan(r, cache, [notes], now=later), cache, marks, now=later)
    assert done["deleted"] == 1 and done["deleted_bytes"] == old[1]
    assert not present(b, old[0]) and present(b, young[0])
    assert (b / "manifests").is_dir() and len(list((b / "manifests").iterdir())) == 2


def test_a_multipart_sidecar_beside_a_record_is_no_record(env):
    """`manifests/<hex>.json.parts.json` (a large manifest's multipart sidecar, 26 Oct on the bucket) stopped the first pass."""
    b, notes, cache, _ = env
    a = artifact(b, "run-record/v1", blob(b, b"x"), age_days=40)
    _put(b, f"manifests/{ids.hex_of(a)}.json.parts.json", b'{"parts": []}')
    _put(b, "attempts/r20261001-000000-0001.json.parts.json", b'{"parts": []}')
    assert set(gc.plan(FsRemote(b), cache, [notes], now=NOW).expired) == {a}


def test_a_blob_a_kept_artifact_also_holds_stays(env):
    b, notes, cache, _ = env
    shared = blob(b, b"circuit.txt")
    artifact(b, "run-record/v1", shared, age_days=30)
    artifact(b, "run-record/v1", shared, blob(b, b"other log"), age_days=1)
    p = gc.plan(FsRemote(b), cache, [notes], now=NOW)
    assert p.candidates == {} and len(p.expired) == 1


def test_citations_by_label_note_and_cited_run_keep_what_they_name(env):
    b, notes, cache, _ = env
    by_label = artifact(b, "run-record/v1", blob(b, b"a"), age_days=40)
    by_note = artifact(b, "run-record/v1", blob(b, b"b"), age_days=40)
    by_run = artifact(b, "run-record/v1", blob(b, b"c"), age_days=40)
    by_ref = artifact(b, "telemetry/v1", blob(b, b"d"), age_days=40)
    gone = artifact(b, "run-record/v1", blob(b, b"e"), age_days=40)
    label(b, by_label)
    label(b, "r20261001-000000-0002", ref=by_ref)
    (notes / "n.md").write_text(f"see {by_note[:16]} and run r20261001-000000-0001\n")
    attempt(b, "r20261001-000000-0001", {"run_record": by_run})
    p = gc.plan(FsRemote(b), cache, [notes], now=NOW)
    assert set(p.expired) == {gone}
    assert p.kept_by["cited"] == 3 and p.kept_by["cited run"] == 1


def test_untimed_kinds_stay_and_only_a_cited_artifact_meta_keeps_a_run(env):
    b, notes, cache, _ = env
    rec_a = artifact(b, "run-record/v1", blob(b, b"ra"), age_days=40)
    rec_b = artifact(b, "run-record/v1", blob(b, b"rb"), age_days=40)
    attempt(b, "r20261001-000000-000a", {"run_record": rec_a})
    attempt(b, "r20261001-000000-000b", {"run_record": rec_b})
    artifact(b, "bench-result/v1", age_days=400, meta={"run_id": "r20261001-000000-000a"})
    cited = artifact(b, "bench-result/v1", age_days=400, meta={"run_id": "r20261001-000000-000b"})
    label(b, cited)
    p = gc.plan(FsRemote(b), cache, [notes], now=NOW)
    assert set(p.expired) == {rec_a}


def test_refs_of_any_kept_artifact_are_kept(env):
    b, notes, cache, _ = env
    bundle = artifact(b, "vllm-replay-bundle/v1", blob(b, b"bundle"), age_days=40)
    artifact(b, "vllm-verdict/v1", age_days=40, refs={"bundle": bundle})
    assert gc.plan(FsRemote(b), cache, [notes], now=NOW).expired == {}


def test_a_manifest_listed_after_the_plan_rescues_its_blobs(env):
    b, notes, cache, marks = env
    old = blob(b, b"reused")
    artifact(b, "run-record/v1", old, age_days=40)
    r = FsRemote(b)
    gc.apply(r, gc.plan(r, cache, [notes], now=NOW), cache, marks, now=NOW)
    later = NOW + gc.GRACE_S
    p = gc.plan(r, cache, [notes], now=later)
    artifact(b, "run-record/v1", old, blob(b, b"new log"))
    out = gc.apply(r, p, cache, marks, now=later)
    assert out["deleted"] == 0 and out["rescued"] == 1 and present(b, old[0])
    assert gc.load_marks(marks) == {}


def test_an_unreadable_manifest_or_citing_tree_deletes_nothing(env, tmp_path):
    b, notes, cache, _ = env
    artifact(b, "run-record/v1", blob(b, b"x"), age_days=40)
    (b / "manifests" / ("0" * 64 + ".json")).write_bytes(b"{not json")
    with pytest.raises(gc.GcError, match="unparseable"):
        gc.plan(FsRemote(b), cache, [notes], now=NOW)
    (b / "manifests" / ("0" * 64 + ".json")).unlink()
    with pytest.raises(gc.GcError, match="not a directory"):
        gc.plan(FsRemote(b), cache, [tmp_path / "missing"], now=NOW)
    with pytest.raises(gc.GcError, match="no citing tree"):
        gc.plan(FsRemote(b), cache, [], now=NOW)
