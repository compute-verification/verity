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


class PutDuringDeletes(FsRemote):
    """Runs `hook(key)` after each delete: a put that lands while the deletes run."""

    def __init__(self, root, hook):
        super().__init__(root)
        self.hook = hook

    def delete(self, key: str) -> bool:
        out = super().delete(key)
        self.hook(key)
        return out


def _due(b, cache, marks, notes, *datas: bytes):
    blobs = [blob(b, d) for d in datas]
    for x in blobs:
        artifact(b, "run-record/v1", x, age_days=40)
    gc.apply(FsRemote(b), gc.plan(FsRemote(b), cache, [notes], now=NOW), cache, marks, now=NOW)
    return blobs


def test_each_batch_of_deletes_follows_a_fresh_listing(env):
    """ci, 26 Oct: one listing before 395,277 deletes left hours in which a put reusing a candidate blob went unseen."""
    b, notes, cache, marks = env
    blobs = sorted(_due(b, cache, marks, notes, b"first", b"second"))
    later = NOW + gc.GRACE_S
    p = gc.plan(FsRemote(b), cache, [notes], now=later)
    put = []
    r = PutDuringDeletes(b, lambda k: put or put.append(artifact(b, "run-record/v1", blobs[1], meta={"put": "between"})))
    out = gc.apply(r, p, cache, marks, now=later, batch=1)
    assert out["deleted"] == 1 and out["rescued"] == 1 and out["listings"] == 3 and out["lost_count"] == 0
    assert not present(b, blobs[0][0]) and present(b, blobs[1][0])


def test_a_blob_deleted_under_a_put_in_its_own_batch_is_reported_lost(env):
    b, notes, cache, marks = env
    gone, = _due(b, cache, marks, notes, b"gone")
    later = NOW + gc.GRACE_S
    put = []
    r = PutDuringDeletes(b, lambda k: put or put.append(artifact(b, "run-record/v1", gone, meta={"put": "during"})))
    out = gc.apply(r, gc.plan(FsRemote(b), cache, [notes], now=later), cache, marks, now=later)
    assert out["deleted"] == 1 and out["lost"] == [gone[0]] and out["lost_count"] == 1


def test_a_put_that_uploads_the_blob_again_loses_nothing(env):
    b, notes, cache, marks = env
    again, = _due(b, cache, marks, notes, b"again")
    later = NOW + gc.GRACE_S
    put = []
    r = PutDuringDeletes(b, lambda k: k.endswith(".parts.json") and not put and put.append(
        (blob(b, b"again"), artifact(b, "run-record/v1", again, meta={"put": "during"}))))
    out = gc.apply(r, gc.plan(FsRemote(b), cache, [notes], now=later), cache, marks, now=later)
    assert out["deleted"] == 1 and out["lost_count"] == 0 and present(b, again[0])


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


def fake_linear(pages: dict):
    """`gql` over canned pages: (field, after) -> connection."""
    asked = []

    def gql(query, variables):
        asked.append(variables)
        if "issue(id:" in query:
            return {"issue": {"comments": pages[("comments", variables["after"])]}}
        field = next(f for f in ("issues", "projects", "projectUpdates", "documents") if f"{f}(first:" in query)
        return {field: pages[(field, variables.get("after"))]}
    return gql, asked


def conn(nodes, after=None):
    return {"nodes": nodes, "pageInfo": {"hasNextPage": after is not None, "endCursor": after}}


def test_linear_cites_issues_their_comments_projects_updates_and_documents(env):
    """top, 26 Oct: lanes cite run ids and art ids mostly in Linear (r20261026-171233-4e86 on INF-263 and VLLM-33)."""
    b, notes, cache, _ = env
    by_issue = artifact(b, "run-record/v1", blob(b, b"1"), age_days=40)
    by_comment = artifact(b, "run-record/v1", blob(b, b"2"), age_days=40)
    by_update = artifact(b, "run-record/v1", blob(b, b"3"), age_days=40)
    by_run = artifact(b, "run-record/v1", blob(b, b"4"), age_days=40)
    gone = artifact(b, "run-record/v1", blob(b, b"5"), age_days=40)
    attempt(b, "r20261026-171233-4e86", {"run_record": by_run})
    gql, asked = fake_linear({
        ("issues", None): conn([{"id": "i1", "identifier": "INF-263", "description": f"see {by_issue[:20]}",
                                 "comments": conn([{"body": "first"}], "c1")}], "p2"),
        ("comments", "c1"): conn([{"body": f"and {by_comment}"}]),
        ("issues", "p2"): conn([{"id": "i2", "identifier": "VLLM-33", "description": "run r20261026-171233-4e86",
                                 "comments": conn([])}]),
        ("projects", None): conn([{"description": "x", "content": None}]),
        ("projectUpdates", None): conn([{"body": f"K from {by_update[:16]}"}]),
        ("documents", None): conn([]),
    })
    p = gc.plan(FsRemote(b), cache, [notes], now=NOW, linear=gql)
    assert set(p.expired) == {gone}
    with pytest.raises(gc.GcError, match="Linear"):
        gc.plan(FsRemote(b), cache, [notes], now=NOW, linear=lambda q, v: (_ for _ in ()).throw(RuntimeError("rate limited")))


def test_apply_refuses_without_linear(tmp_path, capsys):
    from research.store import cli
    with pytest.raises(SystemExit):
        cli.main(["gc", "--cite", str(tmp_path), "--cache", str(tmp_path), "--apply", "--state", str(tmp_path / "m.json")])
    assert "--cite-linear" in capsys.readouterr().err
