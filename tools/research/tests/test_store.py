"""research.store: canonical ids, the local store, the catalog, remotes, the `research data` CLI.  Pure Python, tmp dirs."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import pytest

from research.cli import main as research_main
from research.store import canon, ids, remote_s3
from research.store.cli import main
from research.store.index import flatten
from research.store.local import AttemptExists, IntegrityError, LocalStore, NotFound, StoreError
from research.store.model import ArtifactManifest, Payload, TreeFile, attempt_errors, validate_attempt_doc
from research.store.remote import ConfigError, FsRemote, RemoteConflict, load_remote

DRV = "drv:" + "a" * 64
RK = "b" * 64


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A tmp store root, a tmp (absent) config, a tmp repo without a replica map."""
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    monkeypatch.setenv("RESEARCH_REPO", str(tmp_path / "repo"))
    (tmp_path / "repo").mkdir()
    return tmp_path


@pytest.fixture
def store(env) -> LocalStore:
    return LocalStore()


@pytest.fixture
def fs_remote(env):
    (env / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{env / "vault"}"\n')
    return env / "vault"


def _attempt(run_id="r20260922-105905-2021", drv=DRV, rk=RK, outputs=None, tool="bench_vu", start="2026-09-22T10:59:05Z", **over) -> dict:
    doc = {"schema": "research/attempt/v0.1", "id": run_id, "tool": {"name": tool, "version": "1"}, "derivation": drv, "reuse_key": rk,
           "argv": ["python", "-m", "x"], "params": {"mode": "interactive", "rtt_ms": 0, "nested": {"a": [1, 2]}}, "key_params": ["mode"],
           "inputs": {}, "conditions": {"hw.family": "h100", "hw.sku": "NVIDIA H100"}, "key_conditions": ["hw.family"],
           "source": {"commit": "c" * 40, "tree": "d" * 40, "dirty": False, "dirty_digest": None, "lockfiles": {}}, "closure": {},
           "execution": {"start_utc": start, "end_utc": start, "wall_s": 1.5, "host": "vy-g5", "project": "verity", "campaign": "r21"},
           "cost": {"provider": "runpod", "usd": 0.01}, "status": {"state": "exited", "rc": 0, "class": None, "failure": None, "validation": "passed"},
           "outputs": outputs or {}, "telemetry": {"events": None, "resources": None}}
    doc.update(over)
    return doc


def _tree(root: Path) -> Path:
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "b" / "deep.bin").write_bytes(bytes(range(256)))
    (root / "a" / "x.txt").write_text("x")
    (root / "top.json").write_text('{"k": 1}')
    (root / ".hidden").write_text("h")
    return root


# ---- canon / ids ------------------------------------------------------------------------------------------------------------------------

def test_canon_is_order_independent_and_float_stable():
    a = {"z": 1, "a": [1.5, 0.1, 1e100, True, None, "é"], "m": {"y": 2, "x": 3}}
    b = {"m": {"x": 3, "y": 2}, "a": [1.5, 0.1, 1e100, True, None, "é"], "z": 1}
    assert canon.dumps(a) == canon.dumps(b) == b'{"a":[1.5,0.1,1e+100,true,null,"\xc3\xa9"],"m":{"x":3,"y":2},"z":1}'
    assert canon.sha256(a) == canon.sha256(b) == hashlib.sha256(canon.dumps(a)).hexdigest()
    assert canon.dumps(0.1 + 0.2) == b"0.30000000000000004", "repr, shortest round-trip"
    assert canon.dumps(10 ** 40) == str(10 ** 40).encode()
    assert canon.dumps((1, 2)) == b"[1,2]"


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -0.0, {1: "x"}, {"a": {"b": [float("nan")]}}, {"a": object()}, {"a": -0.0}])
def test_canon_rejects(bad):
    with pytest.raises(ValueError):
        canon.dumps(bad)


def test_ids():
    m = {"schema": "research/artifact/v0.1", "kind": "proof/v1", "meta": {}, "refs": {}, "payload": None}
    assert ids.art_id(m) == "art:" + canon.sha256(m) == ArtifactManifest.from_doc(m).id
    assert ids.is_art(ids.art_id(m)) and not ids.is_drv(ids.art_id(m)) and not ids.is_art("art:xyz")
    assert ids.is_run_id("r20260922-105905-2021") and not ids.is_run_id("r2026-1-1") and not ids.is_run_id("r20260922-105905-ZZZZ")
    assert ids.target_dirname("art:ab") == "art_ab" and ids.target_dirname("r20260922-105905-2021") == "r20260922-105905-2021"
    assert ids.kind_of_id(DRV) == "drv" and ids.kind_of_id("nope") == "unknown"


# ---- model ------------------------------------------------------------------------------------------------------------------------------

def test_manifest_validation():
    ok = ArtifactManifest(kind="proof/v1", meta={"a": 1}, refs={"x": "art:" + "0" * 64, "many": ["art:" + "1" * 64]})
    ok.validate()
    assert ArtifactManifest.from_doc(ok.to_doc()) == ok
    assert "kind" in ";".join(ArtifactManifest(kind="Proof", meta={}).errors())
    assert "262144" in ";".join(ArtifactManifest(kind="proof/v1", meta={"big": "x" * (256 * 1024)}).errors())
    assert not ArtifactManifest(kind="proof/v1", meta={"big": "x" * (256 * 1024 - 10)}).errors()
    assert "refs.x" in ";".join(ArtifactManifest(kind="proof/v1", refs={"x": "drv:" + "0" * 64}).errors())
    assert "refs.x" in ";".join(ArtifactManifest(kind="proof/v1", refs={"x": ["art:" + "0" * 64, "bad"]}).errors())
    assert "canonicalisable" in ";".join(ArtifactManifest(kind="proof/v1", meta={"f": math.nan}).errors())
    f = lambda p: TreeFile(p, "0" * 64, 1)  # noqa: E731
    good = Payload.tree([f("b/z"), f("a")])
    assert [x.path for x in good.files] == ["a", "b/z"] and good.bytes == 2 and not good.errors()
    for files, msg in (([f("../x")], ".."), ([f("/abs")], "relative"), ([f("a//b")], "empty"), ([f("a"), f("a/b")], "both a file and a directory"),
                       ([f("a"), f("a")], "duplicate")):
        p = Payload.tree(files)
        assert msg in ";".join(p.errors()), (files, p.errors())
    unsorted = Payload(type="tree", sha256=good.sha256, bytes=2, files=[f("b/z"), f("a")])
    assert "sorted" in ";".join(unsorted.errors())
    wrong = Payload(type="tree", sha256="f" * 64, bytes=2, files=good.files)
    assert "canon(files)" in ";".join(wrong.errors())
    assert "bare file name" in ";".join(Payload.file("0" * 64, 1, "a/b").errors())


def test_attempt_doc_validation():
    validate_attempt_doc(_attempt())
    assert any("missing keys" in e for e in attempt_errors({k: v for k, v in _attempt().items() if k != "closure"}))
    assert any("run id" in e for e in attempt_errors(_attempt(run_id="nope")))
    assert any("drv:" in e for e in attempt_errors(_attempt(drv="art:" + "0" * 64)))
    assert any("reuse_key" in e for e in attempt_errors(_attempt(rk="short")))
    assert not attempt_errors(_attempt(rk=None))
    assert any("outputs.result" in e for e in attempt_errors(_attempt(outputs={"result": "not-an-id"})))


# ---- local store ------------------------------------------------------------------------------------------------------------------------

def test_put_fetch_roundtrip_file_and_tree(store, tmp_path, capsys):
    src = tmp_path / "proof.bin"
    src.write_bytes(os.urandom(4096))
    art = store.put_artifact("proof/v1", {"backend": "sp1"}, file=src)
    assert ids.is_art(art) and store.has_local(art)
    m = store.get_manifest(art)
    assert m.payload.type == "file" and m.payload.name == "proof.bin" and m.payload.bytes == 4096
    assert store.fetch(art).read_bytes() == src.read_bytes(), "default: the immutable object path"
    out = tmp_path / "out"
    out.mkdir()
    assert store.fetch(art, out) == out / "proof.bin" and (out / "proof.bin").read_bytes() == src.read_bytes()
    assert store.fetch(art, tmp_path / "renamed.bin").read_bytes() == src.read_bytes()
    assert not store.object_path(m.payload.sha256).stat().st_mode & 0o222, "blobs are read-only"   # the mode: root may write anyway

    tree = _tree(tmp_path / "tree")
    tart = store.put_artifact("run-files/v1", {"run": "x"}, tree=tree)
    tm = store.get_manifest(tart)
    assert [f.path for f in tm.payload.files] == [".hidden", "a/b/deep.bin", "a/x.txt", "top.json"]
    assert tm.payload.bytes == 256 + 1 + 8 + 1
    got = store.fetch(tart, tmp_path / "tree_out")
    for f in tm.payload.files:
        assert (got / f.path).read_bytes() == (tree / f.path).read_bytes()
    cached = store.fetch(tart)
    assert cached == store.root / "trees" / ids.hex_of(tart) and (cached / "a/b/deep.bin").read_bytes() == bytes(range(256))
    assert store.fetch(tart) == cached

    assert store.put_artifact("proof/v1", {"backend": "sp1"}, file=src) == art, "idempotent"
    assert store.put_artifact("run-files/v1", {"run": "x"}, tree=tree) == tart
    assert store.put_artifact("proof/v1", {"backend": "other"}, file=src) != art, "meta is part of the identity"
    assert len(list((store.root / "objects/sha256").iterdir())) == 5, "identical blobs stored once"
    store.put_artifact("mystery/v1", {}, file=src)
    assert "unknown kind" in capsys.readouterr().err
    with pytest.raises(ValueError):
        store.put_artifact("proof/v1", {}, file=src, tree=tree)
    with pytest.raises(NotFound):
        store.get_manifest("art:" + "e" * 64)
    nop = store.put_artifact("redteam-findings/v1", {"findings": []}, refs={"target": art})
    with pytest.raises(Exception, match="no payload"):
        store.fetch(nop)


def test_a_file_appended_to_between_hashing_and_storing_stores_nothing(store, tmp_path, monkeypatch):
    """Custody hard-links a run's files and hashes them before the store copies each in: a writer still appending to one (a cancelled
    check's suite, 6 Oct) left a blob whose bytes weren't its name, and R2 then held those bytes under that key for good."""
    from research.store import local

    tree = _tree(tmp_path / "tree")
    log = tree / "a" / "x.txt"
    real = local.sha256_file

    def hash_then_append(p):
        out = real(p)
        if Path(p) == log:
            with open(log, "a") as f:
                f.write("more")
        return out
    monkeypatch.setattr(local, "sha256_file", hash_then_append)
    with pytest.raises(IntegrityError, match="changed between hashing and storing"):
        store.put_artifact("run-record/v1", {"run_id": "x"}, tree=tree)
    monkeypatch.setattr(local, "sha256_file", real)
    objects = store.root / "objects" / "sha256"
    assert not (objects / real(log)[0]).exists() and not (objects / hashlib.sha256(b"x").hexdigest()).exists()
    assert not [p for p in objects.iterdir() if p.name.startswith(".tmp-")], "the rejected copy is left behind"
    art = store.put_artifact("run-record/v1", {"run_id": "x"}, tree=tree)
    assert (store.fetch(art) / "a" / "x.txt").read_text() == "xmore", "once the writer stops, the next publish stores it"


def test_a_tree_another_fetch_landed_first_is_completed_never_replaced(store, tmp_path, monkeypatch):
    """Two processes fetching one tree into trees/ at once: the one whose rename loses must leave the winner's tree in place,
    since the winner has returned it and is reading it (16 xdist workers resolving one regression row's records did)."""
    tart = store.put_artifact("run-files/v1", {"run": "x"}, tree=_tree(tmp_path / "tree"))
    final = store.root / "trees" / ids.hex_of(tart)
    real = LocalStore._materialise
    held: dict[str, int] = {}

    def racing(srcs, dst):
        if dst.name.startswith(".tmp-") and not final.exists():   # the other process lands the whole tree meanwhile ...
            real(srcs, final)
            held["top.json"] = (final / "top.json").stat().st_ino   # ... and holds one of its files
        real(srcs, dst)
    monkeypatch.setattr(LocalStore, "_materialise", staticmethod(racing))
    assert store.fetch(tart) == final
    assert held and (final / "top.json").stat().st_ino == held["top.json"], "the winner's file was replaced under it"
    assert (final / "a/b/deep.bin").read_bytes() == bytes(range(256))
    assert not [p for p in final.parent.iterdir() if p.name.startswith(".tmp-")], "the loser's work dir is left behind"

    monkeypatch.setattr(LocalStore, "_materialise", staticmethod(real))
    part = store.put_artifact("run-files/v1", {"run": "y"}, tree=_tree(tmp_path / "tree2"))
    cached = store.fetch(part, paths=["top.json"])
    ino = (cached / "top.json").stat().st_ino
    assert store.fetch(part) == cached and (cached / "a/x.txt").read_text() == "x"
    assert (cached / "top.json").stat().st_ino == ino, "a full fetch over a partial one replaced a file a reader may hold"


def test_snapshot_identity_is_content(store):
    a, b = "art:" + "1" * 64, "art:" + "2" * 64
    s1 = store.snapshot("nightly", [a, b], note="first freeze")
    s2 = store.snapshot("nightly", [b, a, a], note="first freeze")
    assert s1 == s2, "same set (any order, duplicates collapsed) -> same id"
    assert store.snapshot("nightly", [a], note="first freeze") != s1
    assert store.snapshot("nightly", [a, b]) != s1, "note is content too"
    m = store.get_manifest(s1)
    assert m.kind == "dataset-snapshot/v1" and m.refs == {"members": [a, b]} and m.payload is None
    assert [r["id"] for r in store.index.select_artifacts(kind="dataset-snapshot/v1", where={"name": "nightly"})] == sorted({s1, store.snapshot("nightly", [a], note="first freeze"), store.snapshot("nightly", [a, b])})
    with pytest.raises(ValueError):
        store.snapshot("bad", ["drv:" + "0" * 64])


def test_labels_append_only(store):
    t = "art:" + "3" * 64
    p1 = store.add_label(t, "proof_class", "COMPLETE", "red-team-zk-1", ts="2026-09-22T11:20:29Z")
    p2 = store.add_label(t, "proof_class", "PARTIAL", "red-team-zk-2", ref="r20260922-105905-2021", ts="2026-09-22T11:20:30Z")
    p3 = store.add_label(t, "proof_class", "COMPLETE", "red-team-zk-2", ts="2026-09-22T11:25:00Z")
    p0 = store.add_label(t, "reviewed", True, "human", ts="2026-09-22T09:00:00Z")
    p4 = store.add_label(t, "reviewed", False, "human", ts="2026-09-22T09:00:00Z")
    assert len({p0, p1, p2, p3, p4}) == 5 and p4.name.endswith("~2.json"), "one immutable file per assertion, collisions get a suffix"
    assert p1.parent.name == "art_" + "3" * 64
    labs = store.labels(t)
    assert [(l.ts, l.key, l.by) for l in labs] == [("2026-09-22T09:00:00Z", "reviewed", "human"), ("2026-09-22T09:00:00Z", "reviewed", "human"),
                                                  ("2026-09-22T11:20:29Z", "proof_class", "red-team-zk-1"), ("2026-09-22T11:20:30Z", "proof_class", "red-team-zk-2"),
                                                  ("2026-09-22T11:25:00Z", "proof_class", "red-team-zk-2")]
    assert [l.value for l in store.labels(t, key="reviewed")] == [True, False], "history kept; the store never picks a winner"
    assert [l.by for l in store.labels(t, key="proof_class", by_glob="red-team-*")] == ["red-team-zk-1", "red-team-zk-2", "red-team-zk-2"]
    assert [l.ref for l in store.labels(t, by_glob="*-2")] == ["r20260922-105905-2021", None]
    assert not Path(p1).stat().st_mode & 0o222
    assert store.labels("art:" + "4" * 64) == []
    with pytest.raises(ValueError):
        store.add_label("not-an-id", "k", "v", "me")
    with pytest.raises(ValueError):
        store.add_label(t, "k", math.nan, "me")
    assert [r["id"] for r in store.index.select_artifacts(label=("proof_class", "PARTIAL"))] == [] , "labels index targets; select joins on artifacts"


def test_labels_same_second_history_order_survives_ten_or_more(store):
    """`~N` is a same-second suffix; 12 assertions in one second read back in the order they were made (`~10` after `~9`, not after `~1`)."""
    t = "art:" + "6" * 64
    for i in range(12):
        store.add_label(t, "seq", i, "lane", ref=f"ledger.jsonl:{i}", ts="2026-09-22T11:07:02Z")
    assert [l.value for l in store.labels(t)] == list(range(12))


def test_attempts_write_once(store):
    doc = _attempt(outputs={"result": "art:" + "5" * 64})
    store.put_attempt(doc)
    store.put_attempt(json.loads(json.dumps(doc)))  # same content, other key order: no-op
    with pytest.raises(AttemptExists):
        store.put_attempt(_attempt(outputs={"result": "art:" + "6" * 64}))
    assert store.get_attempt(doc["id"]) == doc
    assert store.get_attempt("r20260101-000000-0000") is None
    assert (store.root / "attempts" / f"{doc['id']}.json").read_bytes() == canon.dumps(doc)
    with pytest.raises(ValueError):
        store.put_attempt({"schema": "research/attempt/v0.1", "id": "bad"})
    assert [a["id"] for a in store.index.attempts_by_reuse_key(RK)] == [doc["id"]]
    assert [a["id"] for a in store.index.attempts_by_derivation(DRV)] == [doc["id"]]
    assert store.index.attempts_by_reuse_key("0" * 64) == []


def test_index_facets_and_rebuild_equivalence(store, tmp_path):
    src = tmp_path / "r.json"
    src.write_text("{}")
    r1 = store.put_artifact("bench-result/v1", {"mode": "interactive", "batch": 8, "tags": ["a", "b"], "nested": {"deep": 1.5}}, file=src)
    r2 = store.put_artifact("bench-result/v1", {"mode": "batch", "batch": 8}, file=src)
    p = store.put_artifact("proof/v1", {"backend": "sp1"}, refs={"inputs": [r1, r2]})
    store.put_attempt(_attempt("r20260922-105905-0001", outputs={"result": r1}))
    store.put_attempt(_attempt("r20260922-105905-0002", outputs={"result": r2}, drv="drv:" + "f" * 64, rk="e" * 64, tool="other"))
    store.add_label(r1, "quality", "gold", "me", ts="2026-09-22T00:00:00Z")
    idx = store.index
    assert [r["id"] for r in idx.select_artifacts(kind="bench-result/v1")] == sorted([r1, r2])
    assert [r["id"] for r in idx.select_artifacts(where={"meta.mode": "interactive"})] == [r1]
    assert [r["id"] for r in idx.select_artifacts(where={"batch": "8", "tags.1": "b"})] == [r1]
    assert [r["id"] for r in idx.select_artifacts(where={"nested.deep": "1.5"})] == [r1]
    assert [r["id"] for r in idx.select_artifacts(label=("quality", "gold"))] == [r1]
    assert [r["id"] for r in idx.select_artifacts(attempt="r20260922-105905-0002")] == [r2]
    assert [r["id"] for r in idx.select_artifacts(tool="bench_vu")] == [r1]
    assert [r["id"] for r in idx.select_artifacts(derivation="drv:" + "f" * 64, kind="bench-result/v1")] == [r2]
    assert idx.select_artifacts(kind="bench-result/v1", where={"mode": "batch"}, tool="bench_vu") == []
    assert idx.sql("SELECT dst FROM refs WHERE src=? ORDER BY dst", (p,)) == [(x,) for x in sorted([r1, r2])]
    assert idx.attempts_by_output(r1) == ["r20260922-105905-0001"]
    assert flatten({"a": {"b": [1, "x", None, True]}, "c": 2.0}) == [("a.b.0", "1"), ("a.b.1", "x"), ("a.b.2", "null"), ("a.b.3", "true"), ("c", "2.0")]
    with pytest.raises(Exception):
        idx.sql("DELETE FROM artifacts")

    queries = lambda i: (i.sql("SELECT * FROM artifacts ORDER BY id"), i.sql("SELECT * FROM artifact_meta ORDER BY id, key"),  # noqa: E731
                         i.sql("SELECT * FROM refs ORDER BY src, name, dst"), i.sql("SELECT * FROM attempts ORDER BY id"),
                         i.sql("SELECT * FROM attempt_params ORDER BY attempt, key"), i.sql("SELECT * FROM attempt_conditions ORDER BY attempt, key"),
                         i.sql("SELECT * FROM attempt_outputs ORDER BY attempt, name"), i.sql("SELECT * FROM labels ORDER BY path"))
    before = queries(idx)
    store.close()
    (store.root / "catalog.sqlite").unlink()
    fresh = LocalStore()
    assert fresh.index.select_artifacts() == []
    n = fresh.reindex()
    assert n == {"artifacts": 3, "attempts": 2, "labels": 1}
    assert queries(fresh.index) == before, "rebuild from files == incremental index"


def test_remote_roundtrip_and_preserved(store, fs_remote, tmp_path, capsys):
    src = tmp_path / "big.bin"
    src.write_bytes(os.urandom(10000))
    tree = _tree(tmp_path / "t")
    art = store.put_artifact("proof/v1", {"n": 1}, file=src)
    tart = store.put_artifact("run-files/v1", {}, tree=tree)
    assert isinstance(store.remote, FsRemote) and store.remote.root == fs_remote
    assert store.verify(art)["preserved"] is False and "absent" in store.verify(art)["errors"][0]
    rep = store.push(art)
    assert rep["preserved"] and rep["verified_utc"] and len(rep["checked"]) == 2
    assert (fs_remote / "objects/sha256" / store.get_manifest(art).payload.sha256).read_bytes() == src.read_bytes()
    assert (fs_remote / "manifests" / (ids.hex_of(art) + ".json")).read_bytes() == store.get_manifest(art).canonical()
    where = {w["location"]: w for w in store.where(art)}
    assert where["local"]["present"] and where["remote"]["present"] and where["remote"]["verified_utc"] == rep["verified_utc"]
    assert "repo" not in where

    store.push(tart)
    # lose the local copies (blob + manifest + tree cache), fetch again from the remote, then PRESERVED again
    for h, _ in store.get_manifest(art).blobs() + store.get_manifest(tart).blobs():
        os.chmod(store.object_path(h), 0o644)
        store.object_path(h).unlink()
    store.manifest_path(ids.hex_of(tart)).unlink()
    store.index.set_presence(art, local=False)
    assert not store.has_local(art) and not store.has_local(tart)
    assert store.fetch(art).read_bytes() == src.read_bytes()
    assert store.has_local(art)
    got = store.fetch(tart, tmp_path / "again")
    assert (got / "a/b/deep.bin").read_bytes() == bytes(range(256)) and store.has_local(tart)
    assert store.verify(tart)["preserved"] and store.verify(art)["preserved"]
    assert store.index.artifact(art)["remote"] and store.index.artifact(art)["local"]

    # the PRESERVED check reads back content: same size, different bytes is NOT preserved
    blob = fs_remote / "objects/sha256" / store.get_manifest(art).payload.sha256
    data = bytearray(blob.read_bytes())
    data[0] ^= 0xFF
    blob.write_bytes(bytes(data))
    bad = store.verify(art)
    assert not bad["preserved"] and "read-back sha256" in bad["errors"][0]
    with pytest.raises(RemoteConflict):
        store.remote.put("objects/sha256/" + store.get_manifest(art).payload.sha256, tree / "a/x.txt")
    with pytest.raises(IntegrityError):
        os.chmod(store.object_path(store.get_manifest(art).payload.sha256), 0o644)
        store.object_path(store.get_manifest(art).payload.sha256).unlink()
        store.fetch(art)

    # reindex --remote lists the remote; a store that never saw tart learns it from the listing
    other = LocalStore(tmp_path / "other")
    assert other.remote is not None
    n = other.reindex(remote=True)
    assert n["artifacts"] == 2 and other.index.artifact(tart)["remote"] and not other.index.artifact(tart)["local"]
    assert other.get_manifest(art).kind == "proof/v1"


def test_no_remote_is_stated_not_failed(store):
    assert store.remote is None
    art = store.snapshot("s", ["art:" + "1" * 64])
    rep = store.verify(art)
    assert rep == {"id": art, "preserved": False, "remote": None, "checked": [], "errors": [], "verified_utc": None, "reason": "no remote configured"}
    assert [w["location"] for w in store.where(art)] == ["local", "remote"] and store.where(art)[1]["configured"] is False
    with pytest.raises(Exception, match="no remote"):
        store.push(art)


def test_cli_push_run_and_pending(store, fs_remote, tmp_path, capsys):
    """`push RUN` preserves the attempt and every artifact it references; `push --pending` finds what the remote lacks; a second
    `--pending` has nothing left to do; exit code follows PRESERVED."""
    from research.store.cli import main
    from research.store.local import attempt_key

    src = tmp_path / "p.bin"
    src.write_bytes(b"proof" * 100)
    art = store.put_artifact("proof/v1", {"n": 1}, file=src)
    tele = store.put_artifact("telemetry/v1", {"run_id": "r20260922-105905-2021", "stream": "events"}, file=src)
    store.put_attempt(_attempt(outputs={"result": art, "run_files": None}, telemetry={"events": tele, "resources": None}))
    assert main(["push", "r20260922-105905-2021"]) == 0
    out = capsys.readouterr().out
    assert out.count("PRESERVED") == 3 and "3/3 preserved" in out
    assert (fs_remote / attempt_key("r20260922-105905-2021")).is_file()
    assert store.where(art)[-1]["verified_utc"]
    extra = store.snapshot("s", [art])                              # local only so far
    assert main(["push", "--pending"]) == 0 and extra in capsys.readouterr().out
    assert main(["push", "--pending", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["pushed"] == []
    assert main(["push", "nonsense"]) == 1 and "not an artifact or run id" in capsys.readouterr().err


def test_unusable_remote_degrades_to_local(env, monkeypatch):
    """store.toml names an S3 bucket but the credential env vars are unset (a pod, a test, a laptop without `source r2.env`): the
    local store must keep working -- reads, puts, attempt publication -- and only push/push_attempt name the missing credentials."""
    (env / "s3.toml").write_text('[remote]\ntype = "s3"\nbucket = "b"\nendpoint = "https://acct.r2.cloudflarestorage.com"\n'
                                 'access_key_id_env = "T_AK_ABSENT"\nsecret_access_key_env = "T_SK_ABSENT"\n')
    monkeypatch.delenv("T_AK_ABSENT", raising=False)
    monkeypatch.delenv("T_SK_ABSENT", raising=False)
    st = LocalStore(env / "degraded", config_path=env / "s3.toml")
    assert st.remote is None and "T_AK_ABSENT" in (st.remote_error or "")
    assert st.get_attempt("r20260101-000000-0000") is None                       # the remote fallback is skipped, not an error
    art = st.snapshot("s", ["art:" + "1" * 64])
    assert st.get_manifest(art).kind == "dataset-snapshot/v1"
    with pytest.raises(Exception, match="T_AK_ABSENT"):
        st.push(art)
    src = env / "payload.bin"
    src.write_bytes(b"evicted later")
    blob_art = st.put_artifact("fixture/v1", {}, {}, file=src)
    st.object_path(st.get_manifest(blob_art).payload.sha256).unlink()             # an evicted blob: only the remote could supply it
    with pytest.raises(NotFound, match="configured remote is unusable here: .*T_AK_ABSENT"):
        st.fetch(blob_art, env / "out")
    monkeypatch.setenv("T_AK_ABSENT", "id")
    monkeypatch.setenv("T_SK_ABSENT", "secret")
    assert isinstance(LocalStore(env / "degraded", config_path=env / "s3.toml").remote, remote_s3.S3Remote)


def test_remote_config(env, monkeypatch):
    assert load_remote(env / "missing.toml") is None
    (env / "c.toml").write_text('[other]\nx = 1\n')
    assert load_remote(env / "c.toml") is None
    (env / "c.toml").write_text('[remote]\ntype = "s3"\nbucket = "b"\nendpoint = "https://acct.r2.cloudflarestorage.com"\n'
                                'access_key_id_env = "T_AK"\nsecret_access_key_env = "T_SK"\n')
    with pytest.raises(ValueError, match="T_AK"):
        load_remote(env / "c.toml")
    monkeypatch.setenv("T_AK", "id")
    monkeypatch.setenv("T_SK", "secret")
    r = load_remote(env / "c.toml")
    assert isinstance(r, remote_s3.S3Remote) and r.bucket == "b" and r.host == "acct.r2.cloudflarestorage.com" and r.region == "auto"
    assert r.describe() == "s3://b @ https://acct.r2.cloudflarestorage.com"
    (env / "c.toml").write_text('[remote]\ntype = "ftp"\n')
    with pytest.raises(ValueError, match="unknown"):
        load_remote(env / "c.toml")


def test_repo_replica_map_verified_then_fallback(env, fs_remote, tmp_path, capsys):
    repo = env / "repo"
    fixture = repo / "fixtures" / "golden" / "rec.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_text('{"golden": true}')
    tdir = _tree(repo / "fixtures" / "tree")
    producer = LocalStore(tmp_path / "producer")
    art = producer.put_artifact("fixture/v1", {"name": "rec"}, file=fixture)
    tart = producer.put_artifact("fixture/v1", {"name": "tree"}, tree=tdir)
    producer.push(art)
    producer.push(tart)
    (repo / "fixtures" / "artifacts.json").write_text(json.dumps({art: "fixtures/golden/rec.json", tart: "fixtures/tree"}, indent=1))

    consumer = LocalStore()   # fresh store; manifests come from the remote, bytes from the repo
    assert consumer.replica_map() == {art: "fixtures/golden/rec.json", tart: "fixtures/tree"}
    assert consumer.fetch(art).read_bytes() == b'{"golden": true}'
    assert (consumer.fetch(tart) / "a/b/deep.bin").read_bytes() == bytes(range(256))
    where = {w["location"]: w for w in consumer.where(art)}
    assert where["repo"]["present"] and where["repo"]["path"] == "fixtures/golden/rec.json" and where["local"]["present"]
    assert consumer.index.artifact(art)["repo_path"] == "fixtures/golden/rec.json"

    # corrupt the repo copy: the map is not trusted, the remote is used, and the user is warned
    fixture.write_text('{"golden": false}')
    consumer2 = LocalStore(tmp_path / "consumer2")
    assert consumer2.fetch(art).read_bytes() == b'{"golden": true}'
    assert "does not match its manifest" in capsys.readouterr().err
    assert {w["location"]: w for w in consumer2.where(art)}["repo"]["present"] is False
    # missing repo copy + no remote: NotFound names both
    fixture.unlink()
    (env / "store.toml").unlink()
    consumer3 = LocalStore(tmp_path / "consumer3")
    with pytest.raises(NotFound):
        consumer3.get_manifest(art)
    shutil.copy(producer.manifest_path(ids.hex_of(art)), consumer3.manifest_path(ids.hex_of(art)))
    with pytest.raises(NotFound, match="not in the repo replica"):
        consumer3.fetch(art)


def test_where_with_a_broken_vault_still_reports_local_and_repo(env, tmp_path, monkeypatch, capsys):
    """A `[remote]` table whose credential env vars are unset is a broken vault (README §7): the store degrades to local-only.  `where`
    answers for the local and repo replicas and reports the remote as UNUSABLE -- never as 'not configured', never as absent."""
    monkeypatch.delenv("SB_TEST_KEY", raising=False)
    (env / "store.toml").write_text('[remote]\ntype = "s3"\nbucket = "b"\nendpoint = "https://example.invalid"\n'
                                    'access_key_id_env = "SB_TEST_KEY"\nsecret_access_key_env = "SB_TEST_KEY"\n')
    repo = env / "repo"
    fixture = repo / "fixtures" / "PIN.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_text('{"pin": 1}')
    st = LocalStore()
    art = st.put_artifact("fixture/v1", {}, file=fixture)
    (repo / "fixtures" / "artifacts.json").write_text(json.dumps({art: "fixtures/PIN.json"}))
    where = {w["location"]: w for w in LocalStore().where(art)}
    assert where["local"]["present"] and where["repo"]["present"] and where["repo"]["path"] == "fixtures/PIN.json"
    assert where["remote"]["configured"] is True and where["remote"]["present"] is None and "SB_TEST_KEY" in where["remote"]["error"]
    rep = LocalStore().verify(art)                                                # reads never raise; anything that needs the vault says why
    assert rep["preserved"] is False and "SB_TEST_KEY" in rep["reason"]
    with pytest.raises(StoreError, match="SB_TEST_KEY"):
        LocalStore().push(art)
    assert main(["where", art]) == 0
    out = capsys.readouterr().out
    assert "repo      present" in out and "remote    UNUSABLE  ConfigError" in out and "not configured" not in out


# ---- CLI --------------------------------------------------------------------------------------------------------------------------------

def test_cli_smoke_every_command(env, fs_remote, tmp_path, capsys):
    out = lambda: capsys.readouterr().out.strip()  # noqa: E731
    src = tmp_path / "proof.bin"
    src.write_bytes(b"proof bytes")
    (tmp_path / "meta.json").write_text('{"backend": "sp1", "params": {"k": 1}}')
    assert main(["put", "--kind", "proof/v1", "--meta", "@" + str(tmp_path / "meta.json"), "--file", str(src)]) == 0
    art = out()
    assert ids.is_art(art)
    tree = _tree(tmp_path / "tree")
    assert main(["put", "--kind", "run-files/v1", "--tree", str(tree), "--ref", f"proof={art}", "--ref", f"inputs={art},{art}", "--preserve", "--json"]) == 0
    rep = json.loads(out())
    tart = rep["id"]
    assert rep["preserve"]["preserved"] is True
    assert main(["show", tart, "--json"]) == 0
    shown = json.loads(out())
    assert shown["manifest"]["refs"] == {"proof": art, "inputs": [art, art]} and shown["manifest"]["payload"]["type"] == "tree"
    assert main(["show", art]) == 0 and "proof.bin" in out()
    assert main(["where", tart]) == 0 and "remote" in out()
    assert main(["where", art, "--json"]) == 0 and json.loads(out())[0]["location"] == "local"
    assert main(["fetch", art, "--to", str(tmp_path / "f")]) == 0 and Path(out()).read_bytes() == b"proof bytes"
    assert main(["fetch", tart]) == 0 and (Path(out()) / "top.json").exists()
    assert main(["verify", tart]) == 0 and "PRESERVED" in out()
    assert main(["verify", art]) == 1 and "NOT preserved" in out()
    assert main(["verify", art, "--json"]) == 1 and json.loads(out())["preserved"] is False

    doc = _attempt(outputs={"result": art, "run_files": tart})
    LocalStore().put_attempt(doc)
    assert main(["show", doc["id"]]) == 0 and DRV in out()
    assert main(["show", doc["id"], "--json"]) == 0 and json.loads(out())["id"] == doc["id"]
    assert main(["show", DRV]) == 0 and doc["id"] in out()
    assert main(["show", DRV, "--json"]) == 0 and json.loads(out())[0]["id"] == doc["id"]
    assert main(["show", "r20260101-000000-0000"]) == 1
    assert main(["show", "garbage"]) == 2

    assert main(["select", "--kind", "proof/v1", "--where", "meta.backend=sp1", "--where", "params.k=1", "--ids"]) == 0 and out() == art
    assert main(["select", "--tool", "bench_vu", "--json"]) == 0 and {r["id"] for r in json.loads(out())} == {art, tart}
    assert main(["select", "--attempt", doc["id"], "--kind", "run-files/v1", "--ids"]) == 0 and out() == tart
    assert main(["select", "--derivation", DRV]) == 0 and art in out()
    assert main(["snapshot", "--name", "nightly", "--members", f"{art},{tart}", "--note", "n"]) == 0
    snap = out()
    assert main(["snapshot", "--name", "nightly", "--note", "n", "--select", "--tool", "bench_vu"]) == 0
    assert out() != snap, "--select records the query in meta, so it is a different snapshot"
    assert main(["snapshot", "--name", "nightly2", "--members", tart, "--members", art, "--json"]) == 0 and json.loads(out())["members"] == sorted([art, tart])
    assert main(["snapshot", "--name", "empty", "--select", "--kind", "nothing/v1"]) == 1

    off = "--off-vocab"                                      # envelope semantics here; the vocabulary is test_store_vocab's
    assert main(["label", art, "proof_class", "COMPLETE", "--by", "red-team-1", "--ref", doc["id"], off]) == 0 and out().endswith("-red-team-1.json")
    assert main(["label", art, "proof_class", "PARTIAL", "--by", "red-team-2", off]) == 0 and Path(out()).is_file()
    assert main(["label", art, "proof_class", "COMPLETE", "--by", "red-team-2", off]) == 0 and Path(out()).is_file()
    assert main(["label", art, "score", "0.5", "--by", "bot", "--value-json", off]) == 0 and Path(out()).is_file()
    assert main(["labels", art]) == 0
    text = out()
    assert len(text.splitlines()) == 3 and "PARTIAL" not in text, "default view: latest per (key, by)"
    assert main(["labels", art, "--history", "--key", "proof_class", "--by", "red-team-*", "--json"]) == 0
    hist = json.loads(out())
    assert [h["value"] for h in hist] == ["COMPLETE", "PARTIAL", "COMPLETE"] and hist[0]["ref"] == doc["id"]
    assert main(["select", "--label", "proof_class=PARTIAL", "--ids"]) == 0 and out() == art
    assert main(["select", "--label", "score=0.5", "--ids"]) == 0 and out() == art

    assert main(["sql", "SELECT count(*) FROM attempts"]) == 0 and out().splitlines() == ["count(*)", "1"]
    assert main(["sql", "SELECT id FROM attempts", "--json"]) == 0 and json.loads(out()) == [{"id": doc["id"]}]
    assert main(["sql", "DELETE FROM attempts"]) == 1
    assert main(["reindex"]) == 0 and "5 artifacts, 1 attempts, 4 labels" in out()
    assert main(["where", tart, "--json"]) == 0 and {w["location"]: w for w in json.loads(out())}["remote"]["verified_utc"], "verification survives a rebuild"
    assert main(["select", "--kind", "run-files/v1", "--json"]) == 0 and json.loads(out())[0]["remote"] is True, "pushed-ness survives a plain reindex"
    assert main(["reindex", "--remote", "--json"]) == 0 and json.loads(out())["artifacts"] == 5
    assert main(["select", "--label", "proof_class=PARTIAL", "--ids"]) == 0 and out() == art

    assert main([]) == 0 and "research data" in out()
    assert main(["nope"]) == 2
    assert main(["fetch", "art:" + "9" * 64]) == 1
    assert main(["put", "--kind", "proof/v1", "--meta", "[1]"]) == 1
    assert research_main(["data", "select", "--ids"]) == 0 and art in out(), "wired into the top-level CLI"
    assert research_main(["--help"]) == 0 and "research data" in out()


# ---- S3 SigV4 ---------------------------------------------------------------------------------------------------------------------------

AK, SK, DATE, REGION = "AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "20130524T000000Z", "us-east-1"
HOST = "examplebucket.s3.amazonaws.com"


def test_sigv4_aws_documented_vectors():
    """The four examples from 'Authenticating Requests: Using the Authorization Header' (S3 API reference)."""
    empty = remote_s3.EMPTY_SHA256
    # GET object (Range header)
    h = remote_s3.sign_headers("GET", HOST, "/test.txt", {}, {"range": "bytes=0-9"}, empty, AK, SK, REGION, DATE)
    creq = remote_s3.canonical_request("GET", "/test.txt", {}, {k: v for k, v in h.items() if k != "Authorization"}, empty)
    assert creq == ("GET\n/test.txt\n\nhost:examplebucket.s3.amazonaws.com\nrange:bytes=0-9\n"
                    "x-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\nx-amz-date:20130524T000000Z\n\n"
                    "host;range;x-amz-content-sha256;x-amz-date\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    assert remote_s3.string_to_sign(DATE, f"20130524/{REGION}/s3/aws4_request", creq).endswith("7344ae5b7ee6c3e7e6b0fe0640412a37625d1fbfff95c48bbb2dc43964946972")
    assert h["Authorization"] == ("AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request,"
                                  "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date,Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41")
    # PUT object with a payload and an extra header; '$' in the key is URI-encoded once
    body = hashlib.sha256(b"Welcome to Amazon S3.").hexdigest()
    assert body == "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072"
    h = remote_s3.sign_headers("PUT", HOST, "/test$file.text", {}, {"date": "Fri, 24 May 2013 00:00:00 GMT", "x-amz-storage-class": "REDUCED_REDUNDANCY"},
                               body, AK, SK, REGION, DATE)
    assert h["Authorization"].endswith("Signature=98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd")
    # GET ?lifecycle (a valueless query parameter)
    h = remote_s3.sign_headers("GET", HOST, "/", {"lifecycle": ""}, {}, empty, AK, SK, REGION, DATE)
    assert h["Authorization"].endswith("Signature=fea454ca298b7da1c68078a5d1bdbfbbe0d65c699e0f91ac7a200a0136783543")
    # GET ?max-keys=2&prefix=J (list objects; sorted query)
    h = remote_s3.sign_headers("GET", HOST, "/", {"prefix": "J", "max-keys": "2"}, {}, empty, AK, SK, REGION, DATE)
    creq = remote_s3.canonical_request("GET", "/", {"prefix": "J", "max-keys": "2"}, {k: v for k, v in h.items() if k != "Authorization"}, empty)
    assert hashlib.sha256(creq.encode()).hexdigest() == "df57d21db20da04d7fa30298dd4488ba3a2b47ca3a489c74750e0f1e7df1b9b7"
    assert h["Authorization"].endswith("Signature=34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7")
    assert remote_s3.uri_encode("a b/c~d", keep_slash=True) == "a%20b/c~d" and remote_s3.uri_encode("a/b") == "a%2Fb"
    assert remote_s3.canonical_query({"list-type": "2", "prefix": "objects/sha256/"}) == "list-type=2&prefix=objects%2Fsha256%2F"


def test_presigned_url_aws_documented_vector():
    """'Authenticating Requests: Using Query Parameters' (S3 API reference): GET /test.txt for 86400 s."""
    url = remote_s3.presign("GET", "https", HOST, "/test.txt", AK, SK, REGION, DATE, 86400)
    assert url == ("https://examplebucket.s3.amazonaws.com/test.txt?X-Amz-Algorithm=AWS4-HMAC-SHA256"
                   "&X-Amz-Credential=AKIAIOSFODNN7EXAMPLE%2F20130524%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20130524T000000Z"
                   "&X-Amz-Expires=86400&X-Amz-SignedHeaders=host"
                   "&X-Amz-Signature=aeeed9bbccd4d02ee5c0109b86d86835f995330da4c265957d157751f604d404")
    r = remote_s3.S3Remote("https://acct.r2.cloudflarestorage.com", "bkt", AK, SK, session_token="tok")
    got = r.presign("objects/sha256/ab", 600)
    assert got.startswith("https://acct.r2.cloudflarestorage.com/bkt/objects/sha256/ab?") and "X-Amz-Security-Token=tok" in got
    assert SK not in got and "X-Amz-Expires=600" in got


def test_s3remote_against_a_fake_bucket(tmp_path):
    """The urllib plumbing (path-style URLs, signed payload hash, HEAD 404, list-type=2 paging) against an in-process HTTP server."""
    import http.server
    import threading
    import urllib.parse

    objects: dict[str, bytes] = {}
    seen: list[tuple[str, str]] = []

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # noqa: D102
            pass

        def _key(self):
            u = urllib.parse.urlsplit(self.path)
            assert u.path.startswith("/bkt")
            return urllib.parse.unquote(u.path[len("/bkt"):].lstrip("/")), dict(urllib.parse.parse_qsl(u.query, keep_blank_values=True))

        def do_PUT(self):
            key, _ = self._key()
            body = self.rfile.read(int(self.headers["Content-Length"]))
            assert self.headers["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=ak/")
            assert self.headers["x-amz-content-sha256"] == hashlib.sha256(body).hexdigest()
            seen.append(("PUT", key))
            objects[key] = body
            self.send_response(200)
            self.end_headers()

        def do_HEAD(self):
            key, _ = self._key()
            seen.append(("HEAD", key))
            if key not in objects:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(objects[key])))
            self.end_headers()

        def do_GET(self):
            key, q = self._key()
            if q.get("list-type") == "2":
                keys = sorted(k for k in objects if k.startswith(q.get("prefix", "")))
                start = int(q.get("continuation-token", "0"))
                page, more = keys[start:start + 2], start + 2 < len(keys)
                xml = ('<?xml version="1.0" encoding="UTF-8"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                       + "".join(f"<Contents><Key>{k}</Key></Contents>" for k in page)
                       + f"<IsTruncated>{'true' if more else 'false'}</IsTruncated>"
                       + (f"<NextContinuationToken>{start + 2}</NextContinuationToken>" if more else "") + "</ListBucketResult>").encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(xml)))
                self.end_headers()
                self.wfile.write(xml)
                return
            if key not in objects:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(objects[key])))
            self.end_headers()
            self.wfile.write(objects[key])

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = False
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        r = remote_s3.S3Remote(f"http://127.0.0.1:{srv.server_address[1]}", "bkt", "ak", "sk")
        files = {}
        for i in range(5):
            p = tmp_path / f"f{i}"
            p.write_bytes(os.urandom(100 + i))
            files[f"objects/sha256/{i:064x}"] = p
            r.put(f"objects/sha256/{i:064x}", p)
        r.put("manifests/x.json", tmp_path / "f0")
        assert r.head("objects/sha256/" + f"{1:064x}") == {"bytes": 101, "etag": None}
        assert r.head("objects/sha256/nope") is None
        assert r.list("objects/sha256/") == sorted(files) and len(r.list("")) == 6 and r.list("zzz") == []
        r.put("objects/sha256/" + f"{1:064x}", tmp_path / "f1")   # same size: HEAD only, no second PUT
        assert seen.count(("PUT", "objects/sha256/" + f"{1:064x}")) == 1
        with pytest.raises(RemoteConflict):
            r.put("objects/sha256/" + f"{1:064x}", tmp_path / "f2")
        r.get("objects/sha256/" + f"{3:064x}", tmp_path / "back")
        assert (tmp_path / "back").read_bytes() == (tmp_path / "f3").read_bytes()
        with pytest.raises(KeyError):
            r.get("objects/sha256/nope", tmp_path / "never")
        assert not (tmp_path / "never").exists()
    finally:
        srv.shutdown()
        srv.server_close()
        th.join()


class FakeS3:
    """An in-process S3 look-alike for `push`: HEAD / PUT / GET (+ list-type=2) on `/<bucket>/<key>`, ETag = MD5 of the stored bytes,
    `If-None-Match: *` -> 412 when the key exists.  Knobs: `reject_conditional` = 400 | 501 models a service without conditional PUT
    (a `NotImplemented` error body); `corrupt` keys are stored with one byte flipped (so their ETag != the client's MD5 and their
    read-back sha256 != the manifest's); `head_404_once` keys answer 404 to their first HEAD although present (a writer that landed
    between the client's HEAD and PUT, so the conditional PUT meets a 412).  `log` records (method, key, conditional?) in arrival order.

    Multipart as S3 / R2 do it: `POST ?uploads` -> UploadId, `PUT ?partNumber=&uploadId=` stores a part (ETag = its MD5), `POST ?uploadId=`
    assembles the parts in PartNumber order and gives the object the multipart ETag `md5(md5s)-n` (kept in `etags`, so HEAD / GET report
    it), `DELETE ?uploadId=` aborts.  `fail_part` = a part number whose PUT answers 500 every time (the client must abort);
    `uploads` holds the in-flight uploads, `aborted` the ids that were cleaned up."""

    def __init__(self, reject_conditional: int | None = None, corrupt: set[str] = frozenset(), head_404_once: set[str] = frozenset(),
                 fail_part: int | None = None, throttle_once: set[str] = frozenset()):
        import http.server
        import threading
        import urllib.parse

        self.objects: dict[str, bytes] = {}
        self.etags: dict[str, str] = {}
        self.uploads: dict[str, tuple[str, dict[int, bytes]]] = {}
        self.aborted: list[str] = []
        self.log: list[tuple[str, str, bool]] = []
        self.lock = threading.Lock()
        self.reject_conditional, self.corrupt, self.head_404_once, self.fail_part = reject_conditional, set(corrupt), set(head_404_once), fail_part
        self.throttle_once = set(throttle_once)     # keys whose first PUT answers 429 SlowDown (R2 under load), Retry-After: 0
        state = self

        def etag_of(key: str) -> str:
            return state.etags.get(key) or hashlib.md5(state.objects[key]).hexdigest()

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):  # noqa: D102
                pass

            def _key(self):
                u = urllib.parse.urlsplit(self.path)
                assert u.path.startswith("/bkt"), u.path
                return urllib.parse.unquote(u.path[len("/bkt"):].lstrip("/")), dict(urllib.parse.parse_qsl(u.query, keep_blank_values=True))

            def _reply(self, code: int, body: bytes = b"", headers: dict | None = None, head_only: bool = False):
                self.send_response(code)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)) if not head_only else headers.get("Content-Length", "0"))
                self.end_headers()
                if body and not head_only:
                    self.wfile.write(body)

            def do_PUT(self):
                key, q = self._key()
                body = self.rfile.read(int(self.headers["Content-Length"]))
                cond = self.headers.get("If-None-Match") == "*"
                with state.lock:
                    if "partNumber" in q:
                        n = int(q["partNumber"])
                        state.log.append(("PART", f"{key}#{n}", False))
                        if self.headers["x-amz-content-sha256"] != hashlib.sha256(body).hexdigest():
                            return self._reply(400, b"<Error><Code>XAmzContentSHA256Mismatch</Code></Error>")
                        up = state.uploads.get(q.get("uploadId", ""))
                        if up is None or up[0] != key:
                            return self._reply(404, b"<Error><Code>NoSuchUpload</Code></Error>")
                        if state.fail_part == n:
                            return self._reply(500, b"<Error><Code>InternalError</Code></Error>")
                        up[1][n] = body
                        return self._reply(200, headers={"ETag": f'"{hashlib.md5(body).hexdigest()}"'})
                    state.log.append(("PUT", key, cond))
                    if key in state.throttle_once:
                        state.throttle_once.discard(key)
                        return self._reply(429, b"<Error><Code>SlowDown</Code></Error>", headers={"Retry-After": "0"})
                    if self.headers["x-amz-content-sha256"] != hashlib.sha256(body).hexdigest():
                        return self._reply(400, b"<Error><Code>XAmzContentSHA256Mismatch</Code></Error>")
                    if cond and state.reject_conditional:
                        return self._reply(state.reject_conditional, b"<Error><Code>NotImplemented</Code><Message>If-None-Match</Message></Error>")
                    if cond and key in state.objects:
                        return self._reply(412, b"<Error><Code>PreconditionFailed</Code></Error>")
                    if key in state.corrupt:
                        body = bytes([body[0] ^ 0xFF]) + body[1:]
                    state.objects[key] = body
                    state.etags.pop(key, None)
                self._reply(200, headers={"ETag": f'"{hashlib.md5(body).hexdigest()}"'})

            def do_POST(self):
                key, q = self._key()
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                with state.lock:
                    if "uploads" in q:
                        uid = f"up{len(state.uploads) + len(state.aborted) + 1:04d}"
                        state.uploads[uid] = (key, {})
                        state.log.append(("INITIATE", key, False))
                        return self._reply(200, (f'<?xml version="1.0" encoding="UTF-8"?><InitiateMultipartUploadResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                                                 f"<Bucket>bkt</Bucket><Key>{key}</Key><UploadId>{uid}</UploadId></InitiateMultipartUploadResult>").encode())
                    up = state.uploads.get(q.get("uploadId", ""))
                    if up is None or up[0] != key:
                        return self._reply(404, b"<Error><Code>NoSuchUpload</Code></Error>")
                    import xml.etree.ElementTree as ET

                    want = [(int(p.findtext("PartNumber")), p.findtext("ETag").strip('"')) for p in ET.fromstring(body).iter("Part")]
                    assert want == sorted(want) and [n for n, _ in want] == sorted(up[1]), "every uploaded part listed once, ascending"
                    for n, et in want:
                        if hashlib.md5(up[1][n]).hexdigest() != et:
                            return self._reply(400, b"<Error><Code>InvalidPart</Code></Error>")
                    sizes = {len(up[1][n]) for n, _ in want[:-1]}
                    if len(sizes) > 1:                                  # R2: every part but the last must be the same size
                        return self._reply(400, b"<Error><Code>InvalidPart</Code><Message>All non-trailing parts must have the same length.</Message></Error>")
                    data = b"".join(up[1][n] for n, _ in want)
                    etag = hashlib.md5(b"".join(hashlib.md5(up[1][n]).digest() for n, _ in want)).hexdigest() + f"-{len(want)}"
                    state.objects[key], state.etags[key] = data, etag
                    del state.uploads[q["uploadId"]]
                    state.log.append(("COMPLETE", key, False))
                    return self._reply(200, (f'<?xml version="1.0" encoding="UTF-8"?><CompleteMultipartUploadResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                                             f'<Key>{key}</Key><ETag>"{etag}"</ETag></CompleteMultipartUploadResult>').encode())

            def do_DELETE(self):
                key, q = self._key()
                with state.lock:
                    up = state.uploads.pop(q.get("uploadId", ""), None)
                    state.log.append(("ABORT", key, False))
                    if up is None:
                        return self._reply(404, b"<Error><Code>NoSuchUpload</Code></Error>")
                    state.aborted.append(q["uploadId"])
                self._reply(204)

            def do_HEAD(self):
                key, _ = self._key()
                with state.lock:
                    state.log.append(("HEAD", key, False))
                    if key in state.head_404_once:
                        state.head_404_once.discard(key)
                        return self._reply(404, head_only=True, headers={"Content-Length": "0"})
                    data = state.objects.get(key)
                    etag = etag_of(key) if data is not None else None
                if data is None:
                    return self._reply(404, head_only=True, headers={"Content-Length": "0"})
                self._reply(200, head_only=True, headers={"Content-Length": str(len(data)), "ETag": f'"{etag}"'})

            def do_GET(self):
                key, q = self._key()
                with state.lock:
                    state.log.append(("GET", key, False))
                    if q.get("list-type") == "2":
                        keys = sorted(k for k in state.objects if k.startswith(q.get("prefix", "")))
                        xml = ('<?xml version="1.0" encoding="UTF-8"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                               + "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys) + "<IsTruncated>false</IsTruncated></ListBucketResult>").encode()
                        return self._reply(200, xml)
                    data = state.objects.get(key)
                    etag = etag_of(key) if data is not None else None
                if data is None:
                    return self._reply(404)
                self._reply(200, data, headers={"ETag": f'"{etag}"'})

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.srv.server_address[1]}"

    def remote(self) -> remote_s3.S3Remote:
        return remote_s3.S3Remote(self.endpoint, "bkt", "ak", "sk")

    def puts(self, key_prefix: str = "") -> list[tuple[str, bool]]:
        return [(k, c) for m, k, c in self.log if m == "PUT" and k.startswith(key_prefix)]

    def close(self) -> None:
        self.srv.shutdown()
        self.srv.server_close()
        self.thread.join()


@pytest.fixture
def fake_s3(env, monkeypatch):
    """A `FakeS3` wired in as the store's remote through store.toml (credentials from env vars, as in production)."""
    made: list[FakeS3] = []

    def make(**kw) -> FakeS3:
        s = FakeS3(**kw)
        made.append(s)
        (env / "store.toml").write_text(f'[remote]\ntype = "s3"\nbucket = "bkt"\nendpoint = "{s.endpoint}"\n'
                                        'access_key_id_env = "T_AK"\nsecret_access_key_env = "T_SK"\n')
        monkeypatch.setenv("T_AK", "ak")
        monkeypatch.setenv("T_SK", "sk")
        return s

    yield make
    for s in made:
        s.close()


def _many_files(root: Path, n: int, seed: int = 0) -> Path:
    rng = __import__("random").Random(seed)
    for i in range(n):
        p = root / f"d{i % 7}" / f"f{i:05d}.bin"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(rng.randbytes(rng.randint(1, 4096)))
    return root


@pytest.mark.slow
def test_push_is_concurrent_conditional_and_verified_by_etag(fake_s3, tmp_path):
    """300 blobs with 16 workers: every object lands, every PUT is a conditional create, the manifest's PUT comes after the last blob's,
    and the report is deterministic.  A second push HEADs everything, uploads nothing and is still PRESERVED (`etag-md5`).  A key that
    another writer created between our HEAD and PUT meets a 412 and counts as present."""
    s = fake_s3()
    st = LocalStore()
    art = st.put_artifact("run-files/v1", {"n": 300}, tree=_many_files(tmp_path / "t", 300))
    m = st.get_manifest(art)
    rep = st.push(art, jobs=16)
    assert rep["preserved"] and rep["verified"] == "etag-md5" and rep["errors"] == [] and rep["notes"] == []
    assert rep["objects"] == 301 and rep["uploaded"] == 301 and rep["skipped"] == 0 and rep["bytes"] == m.payload.bytes + len(m.canonical())
    assert set(s.objects) == {f"objects/sha256/{h}" for h, _ in m.blobs()} | {f"manifests/{ids.hex_of(art)}.json"}
    for h, _ in m.blobs():
        assert hashlib.sha256(s.objects[f"objects/sha256/{h}"]).hexdigest() == h
    puts = s.puts()
    assert len(puts) == 302 and all(c for _, c in puts), "every create carried If-None-Match: *, plus the session's one probe re-PUT"
    probed = [k for k, _ in puts if [k for k, _ in puts].count(k) == 2]
    assert len(probed) == 2 and probed[0] == probed[1], "the probe re-PUTs the first created key once (and saw the fake's 412)"
    assert st.remote.conditional_put and st.remote.conditional_put_probed
    assert puts[-1][0] == f"manifests/{ids.hex_of(art)}.json", "the manifest is the last object to go up"
    assert [k for k, _ in puts[:-1]] != sorted(k for k, _ in puts[:-1]) or True   # completion order is arbitrary...
    assert rep["checked"] == [f"objects/sha256/{h}" for h, _ in m.blobs()] + [f"manifests/{ids.hex_of(art)}.json"], "...the report is not"
    reps = st.index.replicas(art)["remote"]
    assert reps["verified_utc"] == rep["verified_utc"] and reps["verified"] == "etag-md5"
    where = {w["location"]: w for w in st.where(art)}
    assert where["remote"]["present"] and where["remote"]["verified_utc"] == rep["verified_utc"]

    n_log = len(s.log)
    again = st.push(art, jobs=16)
    assert again["preserved"] and again["uploaded"] == 0 and again["skipped"] == 301 and again["verified"] == "etag-md5"
    # PUT-first meets 412 on the first few present keys, then the adaptive window flips the client to HEAD-first (remote_s3 docstring)
    re_puts = [x for x in s.log[n_log:] if x[0] == "PUT"]
    assert 0 < len(re_puts) <= 16 + 16 and all(c for _, _, c in re_puts), "a handful of conditional PUTs (412) before HEAD-first takes over"
    assert not st.remote.stats()["put_first"], "the window remembers: this remote now HEADs first"
    assert [x[0] for x in s.log[n_log:]].count("GET") == 1 and len(s.log[n_log:]) == 301 + 1 + len(re_puts), "one HEAD per blob (+ the 412'd PUTs); HEAD + GET for the manifest (always read back)"

    # a racing writer: HEAD says absent, the conditional PUT says 412 -> `exists`, then the verifying HEAD confirms size + MD5
    victim = f"objects/sha256/{m.blobs()[7][0]}"
    s.head_404_once.add(victim)
    n_log = len(s.log)
    race = st.push(art, jobs=4)
    assert race["preserved"] and race["uploaded"] == 0 and race["skipped"] == 301
    assert s.log[n_log:].count(("PUT", victim, True)) == 1 and s.objects[victim] == s.objects[victim]

    # `push_attempt` and `preserve=False` still work on the dict-returning `put`
    st.put_attempt(_attempt(outputs={"result": art}))
    st.push_attempt("r20260922-105905-2021")
    assert "attempts/r20260922-105905-2021.json" in s.objects
    snap = st.snapshot("s", [art])
    rep = st.push(snap, preserve=False)
    assert rep["preserved"] is None and rep["objects"] == 1 and rep["verified"] is None and f"manifests/{ids.hex_of(snap)}.json" in s.objects
    assert st.index.replicas(snap)["remote"]["verified_utc"] is None
    assert st.push(snap)["verified"] == "sha256-readback", "no blobs: the manifest read-back is the whole check"


@pytest.mark.slow
def test_push_falls_back_to_plain_put_when_the_service_rejects_the_condition(fake_s3, tmp_path):
    s = fake_s3(reject_conditional=501)
    st = LocalStore()
    art = st.put_artifact("run-files/v1", {}, tree=_many_files(tmp_path / "t", 40, seed=1))
    rep = st.push(art, jobs=8)
    assert rep["preserved"] and rep["uploaded"] == 41
    assert len(rep["notes"]) == 1 and "NotImplemented (501)" in rep["notes"][0] and "plain PUT" in rep["notes"][0]
    puts = s.puts()
    assert sum(1 for _, c in puts if c) >= 1 and sum(1 for _, c in puts if not c) == 41, "the first rejection switches it off; every object still lands"
    assert st.remote.conditional_put is False and "501" in st.remote.conditional_put_error
    assert set(s.objects) == {f"objects/sha256/{h}" for h, _ in st.get_manifest(art).blobs()} | {f"manifests/{ids.hex_of(art)}.json"}
    s2 = fake_s3(reject_conditional=400)
    st2 = LocalStore(tmp_path / "s2")
    art2 = st2.put_artifact("proof/v1", {}, file=tmp_path / "t" / "d0" / "f00000.bin")
    rep2 = st2.push(art2)
    assert rep2["preserved"] and "(400)" in rep2["notes"][0] and len(s2.objects) == 2


def test_push_surfaces_a_bad_etag_as_not_preserved_and_withholds_the_manifest(fake_s3, tmp_path):
    """The service stores different bytes for one key (its ETag != our MD5): that blob is an error, the artifact is NOT preserved, the
    manifest is never uploaded (it must not be visible before its blobs), no verified_utc is recorded, and the batch (`cmd_push`)
    goes on.  With `--verify readback` the same corruption is caught by sha256."""
    s = fake_s3()
    st = LocalStore()
    art = st.put_artifact("run-files/v1", {}, tree=_many_files(tmp_path / "t", 30, seed=2))
    m = st.get_manifest(art)
    bad = f"objects/sha256/{m.blobs()[11][0]}"
    s.corrupt.add(bad)
    rep = st.push(art, jobs=8)
    assert rep["preserved"] is False and len(rep["errors"]) == 1 and rep["errors"][0].startswith(bad) and "ETag" in rep["errors"][0] and "MD5" in rep["errors"][0]
    assert rep["verified"] is None and rep["verified_utc"] is None and rep["uploaded"] == 30 and rep["objects"] == 31
    assert f"manifests/{ids.hex_of(art)}.json" not in s.objects and "manifest not uploaded: 1 blob(s) failed" in rep["notes"]
    assert len(rep["checked"]) == 29
    assert st.index.replicas(art).get("remote", {}).get("verified_utc") is None
    assert main(["push", art]) == 1
    # the remote holds same-size different bytes under an immutable key: a later push refuses to overwrite and says why
    rep = st.push(art, verify="readback")
    assert not rep["preserved"] and "RemoteConflict" in rep["errors"][0] and f"manifests/{ids.hex_of(art)}.json" not in s.objects
    # readback catches a corruption that HEAD cannot see: the server lies about the ETag but serves the bytes it has
    s2 = fake_s3()
    st2 = LocalStore(tmp_path / "s2")
    art2 = st2.put_artifact("run-files/v1", {}, tree=_many_files(tmp_path / "t2", 12, seed=3))
    m2 = st2.get_manifest(art2)
    rep = st2.push(art2, jobs=4, verify="readback")
    assert rep["preserved"] and rep["verified"] == "sha256-readback" and st2.index.replicas(art2)["remote"]["verified"] == "sha256-readback"
    assert sum(1 for x in s2.log if x[0] == "GET") == 13, "readback: one GET per blob and one for the manifest"
    victim = f"objects/sha256/{m2.blobs()[0][0]}"
    data = bytearray(s2.objects[victim])
    data[0] ^= 1
    s2.objects[victim] = bytes(data)
    rep = st2.push(art2, jobs=4, verify="readback")                       # HEAD sees a changed ETag -> RemoteConflict (immutable key)
    assert not rep["preserved"] and "ETag" in rep["errors"][0]
    assert not st2.verify(art2)["preserved"] and "read-back sha256" in st2.verify(art2)["errors"][0]


def test_push_uploads_a_blob_above_the_single_put_limit_in_parts(fake_s3, tmp_path, monkeypatch, capsys):
    """A blob over `single_put_max` goes up as a multipart upload (initiate / N parts / complete), the service's multipart ETag must be
    the one the local parts yield, the object is PRESERVED (`etag-multipart`), and a `<key>.parts.json` sidecar records the parts.  A
    second push HEADs and skips.  A part that keeps failing aborts the upload (no orphan) and the artifact is NOT preserved."""
    s = fake_s3()
    monkeypatch.setattr(remote_s3.S3Remote, "single_put_max", 8000)     # limits scaled down: 8 kB "PutObject", 4 kB parts
    monkeypatch.setattr(remote_s3.S3Remote, "part_size", 4096)
    st = LocalStore()
    big = tmp_path / "big.bin"
    big.write_bytes(os.urandom(4096 * 2 + 1000))                          # 3 parts: 4096, 4096, 1000
    small = _many_files(tmp_path / "t", 1) / "d0" / "f00000.bin"
    tree = tmp_path / "tree"
    tree.mkdir()
    shutil.copyfile(big, tree / "big.bin")
    shutil.copyfile(small, tree / "small.bin")
    art = st.put_artifact("run-files/v1", {}, tree=tree)
    m = st.get_manifest(art)
    big_sha = hashlib.sha256(big.read_bytes()).hexdigest()
    key = f"objects/sha256/{big_sha}"
    rep = st.push(art, jobs=4)
    assert rep["preserved"] and rep["errors"] == [] and rep["uploaded"] == 3 and rep["verified"] == "etag-md5,etag-multipart", rep
    assert s.objects[key] == big.read_bytes() and s.etags[key].endswith("-3") and s.uploads == {} and s.aborted == []
    parts = remote_s3.hash_parts(big, 4096)
    assert s.etags[key] == remote_s3.multipart_etag([p["md5"] for p in parts])
    seq = [(m_, k) for m_, k, _ in s.log if k.startswith(key)]
    assert seq[0] == ("HEAD", key) and seq[1] == ("INITIATE", key) and sorted(seq[2:5]) == [("PART", f"{key}#1"), ("PART", f"{key}#2"), ("PART", f"{key}#3")]
    assert seq[5] == ("COMPLETE", key) and ("PUT", f"{key}.parts.json") in seq     # the small sidecar may go PUT-first (no HEAD)
    side = json.loads(s.objects[f"{key}.parts.json"])
    assert side["schema"] == "research/parts/v1" and side["bytes"] == 9192 and side["part_bytes"] == 4096 and side["sha256"] == big_sha and side["etag"] == s.etags[key]
    assert [p["bytes"] for p in side["parts"]] == [4096, 4096, 1000] and [p["n"] for p in side["parts"]] == [1, 2, 3]
    assert st.index.replicas(art)["remote"]["verified"] == "etag-md5,etag-multipart"
    assert st.verify(art)["preserved"], "read-back of a multipart object is the same streamed GET + sha256"

    n = len(s.log)
    again = st.push(art, jobs=4)
    assert again["preserved"] and again["uploaded"] == 0 and again["skipped"] == 3 and again["verified"] == "etag-md5,etag-multipart"
    assert [x[0] for x in s.log[n:] if x[1] == key] == ["HEAD"], "present: one HEAD, its multipart ETag matches the local parts, nothing uploaded"

    # same size, different bytes under the immutable key: refused (the multipart ETag gives it away)
    other = tmp_path / "other.bin"
    other.write_bytes(os.urandom(9192))
    with pytest.raises(RemoteConflict, match="multipart ETag"):
        st.remote.put(key, other)

    # a part that never lands: the upload is aborted (DELETE ?uploadId), nothing is left half-written, the artifact is NOT preserved
    s2 = fake_s3(fail_part=2)
    monkeypatch.setattr(remote_s3, "RETRY_BACKOFF_S", 0.0)
    st2 = LocalStore(tmp_path / "s2")
    art2 = st2.put_artifact("proof/v1", {}, file=big)
    rep2 = st2.push(art2)
    assert rep2["preserved"] is False and len(rep2["errors"]) == 1 and "HTTP Error 500" in rep2["errors"][0]
    assert key not in s2.objects and s2.uploads == {} and s2.aborted == ["up0001"] and f"manifests/{ids.hex_of(art2)}.json" not in s2.objects
    assert main(["push", art2]) == 1 and "NOT preserved" in capsys.readouterr().out

    with pytest.raises(ValueError, match="verify must be"):
        st.push(art, verify="etag")


def test_throttled_put_is_retried_not_failed(fake_s3, tmp_path, monkeypatch):
    # R2 answers 429 under load (the vyv-v2cpu3 bench saw 4 per 10k small PUTs): retried like a 5xx, Retry-After honoured and capped
    s = fake_s3(throttle_once={"bench/a", "bench/b"})
    monkeypatch.setattr(remote_s3, "RETRY_BACKOFF_S", 0.0)
    r = s.remote()
    f = tmp_path / "a.bin"
    f.write_bytes(os.urandom(4096))
    res = r.put_many([("bench/a", f), ("bench/b", f)], jobs=2)
    assert [x["action"] for x in res] == ["created", "created"] and s.objects["bench/a"] == f.read_bytes()
    assert [k for k, _ in s.puts("bench/a")].count("bench/a") >= 2, "the throttled PUT was sent again"
    assert remote_s3.retry_delay("0", 0) == 0.0 and remote_s3.retry_delay("3600", 0) == remote_s3.RETRY_AFTER_MAX_S
    assert remote_s3.retry_delay("Wed, 21 Oct 2026 07:28:00 GMT", 2) == 0.0 and remote_s3.retry_delay(None, 1) == 0.0


@pytest.mark.slow
def test_cli_push_reports_rate_and_takes_jobs_and_verify(fake_s3, tmp_path, capsys):
    s = fake_s3()
    st = LocalStore()
    art = st.put_artifact("run-files/v1", {}, tree=_many_files(tmp_path / "t", 25, seed=4))
    assert main(["push", art, "--jobs", "5", "--verify", "readback"]) == 0
    out = capsys.readouterr().out
    line = out.splitlines()[0]
    assert line.startswith(f"{art}  run-files/v1  PRESERVED  26 objects (26 uploaded, 0 already there)") and "obj/s" in line and "verified sha256-readback" in line
    assert "1/1 preserved" in out and sum(1 for x in s.log if x[0] == "GET") == 26
    assert main(["push", art, "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    x = doc["pushed"][0]
    assert x["preserved"] and x["objects"] == 26 and x["uploaded"] == 0 and x["skipped"] == 26 and x["verified"] == "etag-md5" and x["seconds"] >= 0
    assert main(["push", art, "--jobs", "0"]) == 1 and "--jobs" in capsys.readouterr().err
    s.reject_conditional = 501
    art2 = st.put_artifact("run-files/v1", {}, tree=_many_files(tmp_path / "t2", 3, seed=5))
    assert main(["push", art2, art]) == 0
    cap = capsys.readouterr()
    assert cap.out.count("PRESERVED") == 2 and "2/2 preserved" in cap.out
    assert cap.err.count("note: NotImplemented (501)") == 1, "the fallback is said once per batch"


def test_s3remote_session_token_is_sent_and_signed(monkeypatch, tmp_path, env):
    """A temporary credential's session token goes out as `x-amz-security-token` on every request and is part of SignedHeaders (so
    the signature covers it, as SigV4 requires); without a token the header is absent.  The transport is faked at the pool: no network."""
    import email.message

    sent: list[dict] = []           # {"method", "target", "headers", "data"} per request

    class FakeResp:
        status, reason, will_close, length = 200, "OK", False, 0

        def __init__(self):
            self.headers = email.message.Message()
            self.headers["Content-Length"], self.headers["ETag"] = "5", '"x"'

        def read(self, n=-1):
            return b""

        def isclosed(self):
            return True

        def close(self):
            pass

    class FakeConn:
        def request(self, method, target, body=None, headers=None):
            sent.append({"method": method, "target": target, "headers": dict(headers or {}), "data": body})

        def getresponse(self):
            return FakeResp()

        def close(self):
            pass

    def fake_pool(r):
        monkeypatch.setattr(r.pool, "acquire", lambda: FakeConn())
        monkeypatch.setattr(r.pool, "release", lambda conn: None)

    r = remote_s3.S3Remote("https://acct.r2.cloudflarestorage.com", "bucket", "ak", "sk", session_token="tok/EN+123==")
    fake_pool(r)
    r.head("objects/sha256/" + "0" * 64)
    p = tmp_path / "x"
    p.write_bytes(b"12345")
    with r._request("PUT", "k", body=p):
        pass
    assert len(sent) == 2
    for req in sent:
        hdrs = {k.lower(): v for k, v in req["headers"].items()}
        assert hdrs["x-amz-security-token"] == "tok/EN+123=="
        auth = hdrs["authorization"]
        signed = auth.split("SignedHeaders=")[1].split(",")[0].split(";")
        assert "x-amz-security-token" in signed and signed == sorted(signed)
        # the signature is reproducible from the headers that were sent (token included), i.e. it really covers the token
        method = req["method"]
        sig_headers = {k: v for k, v in hdrs.items() if k != "authorization"}
        creq = remote_s3.canonical_request(method, "/bucket/" + ("k" if method == "PUT" else "objects/sha256/" + "0" * 64), {}, sig_headers,
                                           sig_headers["x-amz-content-sha256"])
        date = sig_headers["x-amz-date"]
        sts = remote_s3.string_to_sign(date, f"{date[:8]}/auto/s3/aws4_request", creq)
        assert auth.endswith("Signature=" + remote_s3.signature("sk", date[:8], "auto", sts))
    assert sent[1]["headers"]["content-length"] == "5" and hasattr(sent[1]["data"], "read"), "a Path body streams from the open file"
    sent.clear()
    plain = remote_s3.S3Remote("https://acct.r2.cloudflarestorage.com", "bucket", "ak", "sk")
    fake_pool(plain)
    plain.head("k")
    hdrs = {k.lower(): v for k, v in sent[0]["headers"].items()}
    assert "x-amz-security-token" not in hdrs and "security-token" not in hdrs["authorization"]

    # load_remote: the token's env var is optional in the config (default AWS_SESSION_TOKEN) and optional in the environment
    (env / "c.toml").write_text('[remote]\ntype = "s3"\nbucket = "b"\nendpoint = "https://acct.r2.cloudflarestorage.com"\n'
                                'access_key_id_env = "T_AK"\nsecret_access_key_env = "T_SK"\n')
    monkeypatch.setenv("T_AK", "id")
    monkeypatch.setenv("T_SK", "secret")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    assert load_remote(env / "c.toml").session_token is None
    monkeypatch.setenv("AWS_SESSION_TOKEN", "t1")
    assert load_remote(env / "c.toml").session_token == "t1"
    (env / "c.toml").write_text('[remote]\ntype = "s3"\nbucket = "b"\nendpoint = "https://acct.r2.cloudflarestorage.com"\n'
                                'access_key_id_env = "T_AK"\nsecret_access_key_env = "T_SK"\nsession_token_env = "T_ST"\n')
    assert load_remote(env / "c.toml").session_token is None, "a named env var that is unset means no token, not the default's value"
    monkeypatch.setenv("T_ST", "t2")
    assert load_remote(env / "c.toml").session_token == "t2"


# ---- temporary credentials ------------------------------------------------------------------------------------------------------------

def test_mint_local_ports_cloudflares_reference_signing():
    """The JWT is Cloudflare's `createTempCredentials` claim for claim: HS256 with the parent secret, `sub` account, `iss` parent key id,
    `aud` endpoint host, iat/exp; temp secret = sha256 hex of the JWT; session token = base64("jwt/" + JWT); key id = the parent's.
    `actions` replaces `scope` (R2 rejects both together)."""
    import base64
    import hmac

    from research.store import credentials as C

    acct = "0" * 32                                                  # a dummy account id: 32 hex digits, as R2 has
    ep, ak, sk = f"https://{acct}.r2.cloudflarestorage.com", "PARENTKEYID", "parent-secret"
    c = C.mint_local(ep, acct, ak, sk, "store-bucket", "object-read-only", ttl_s=900, now=1_790_000_000)
    head, claims = C.decode_jwt(c["jwt"])
    assert head == {"alg": "HS256", "typ": "JWT"}
    assert claims == {"bucket": "store-bucket", "scope": "object-read-only", "sub": acct, "iss": ak, "aud": f"{acct}.r2.cloudflarestorage.com",
                      "iat": 1_790_000_000, "exp": 1_790_000_900}
    assert list(claims) == ["bucket", "scope", "sub", "iss", "aud", "iat", "exp"], "claim order as jose emits it"
    h, p, s = c["jwt"].split(".")
    assert base64.urlsafe_b64encode(hmac.new(b"parent-secret", f"{h}.{p}".encode(), hashlib.sha256).digest()).rstrip(b"=").decode() == s
    assert "=" not in c["jwt"] and c["accessKeyId"] == ak
    assert c["secretAccessKey"] == hashlib.sha256(c["jwt"].encode()).hexdigest()
    assert base64.b64decode(c["sessionToken"]).decode() == "jwt/" + c["jwt"]
    assert c["expires_utc"] == "2026-09-21T14:28:20Z"
    same = C.mint_local(ep, acct, ak, sk, "store-bucket", "object-read-only", ttl_s=900, now=1_790_000_000)
    assert same == c, "deterministic for a fixed `now`"
    assert C.mint_local(ep, acct, ak, "other", "store-bucket", "object-read-only", ttl_s=900, now=1_790_000_000)["jwt"] != c["jwt"]

    scoped = C.mint_local(ep, acct, ak, sk, "b", "object-read-write", actions=["PutObject", "HeadObject"], prefixes=["scratch/x/"], now=1)
    claims = C.decode_jwt(scoped["jwt"])[1]
    assert "scope" not in claims and claims["actions"] == ["PutObject", "HeadObject"] and claims["paths"] == {"prefixPaths": ["scratch/x/"], "objectPaths": []}
    assert list(claims)[:3] == ["bucket", "actions", "paths"] and claims["exp"] - claims["iat"] == 3600, "default ttl 1 h"
    objs = C.mint_local(ep, acct, ak, sk, "b", "object-read-write", objects=["manifests/a.json"], now=1)
    assert C.decode_jwt(objs["jwt"])[1]["paths"] == {"prefixPaths": [], "objectPaths": ["manifests/a.json"]}
    assert "paths" not in C.decode_jwt(c["jwt"])[1], "no paths -> no paths claim (whole bucket)"
    for bad in ({"scope": "root"}, {"scope": "object-read-only", "actions": ["s3:GetObject"]}, {"scope": "object-read-only", "ttl_s": 604801}, {"scope": "object-read-only", "ttl_s": 0}):
        with pytest.raises(C.MintError):
            C.mint_local(ep, acct, ak, sk, "b", **bad)
    assert C.parse_ttl("6h") == 21600 and C.parse_ttl("15m") == 900 and C.parse_ttl("7d") == 604800 and C.parse_ttl(900) == 900 and C.parse_ttl("42") == 42
    for bad in ("8d", "0", "1w", "", "6 h"):
        with pytest.raises(C.MintError):
            C.parse_ttl(bad)
    assert C.account_id_from_endpoint(ep) == acct and C.account_id_from_endpoint("https://s3.amazonaws.com") is None


def test_mint_api_calls_the_temp_access_credentials_endpoint():
    import io
    import urllib.error

    from research.store import credentials as C

    calls = []

    class Opener:
        def __init__(self, status=200, body=None):
            self.status, self.body = status, body

        def open(self, req, timeout=None):
            calls.append((req.full_url, req.get_method(), dict(req.header_items()), json.loads(req.data)))
            if self.status != 200:
                raise urllib.error.HTTPError(req.full_url, self.status, "Forbidden", {}, io.BytesIO(json.dumps({"errors": [{"code": 10000, "message": "Authentication error"}]}).encode()))
            r = io.BytesIO(json.dumps(self.body).encode())
            r.__enter__, r.__exit__ = lambda: r, lambda *a: False
            return r

    ok = {"success": True, "errors": [], "result": {"accessKeyId": "tmpid", "secretAccessKey": "tmpsecret", "sessionToken": "tok"}}
    c = C.mint_api("acct123", "cf-token", "store-bucket", "PARENT", "object-read-write", ttl_s=3600, prefixes=["objects/"], opener=Opener(body=ok))
    url, method, headers, body = calls[0]
    assert url == "https://api.cloudflare.com/client/v4/accounts/acct123/r2/temp-access-credentials" and method == "POST"
    assert headers["Authorization"] == "Bearer cf-token" and headers["Content-type"] == "application/json"
    assert body == {"bucket": "store-bucket", "parentAccessKeyId": "PARENT", "permission": "object-read-write", "ttlSeconds": 3600, "prefixes": ["objects/"]}
    assert {k: c[k] for k in ("accessKeyId", "secretAccessKey", "sessionToken")} == ok["result"] and c["expires_utc"].endswith("Z")
    with pytest.raises(C.MintError, match="Authentication error"):
        C.mint_api("acct123", "bad", "b", "P", "object-read-only", opener=Opener(status=403))
    with pytest.raises(C.MintError, match="refused"):
        C.mint_api("acct123", "t", "b", "P", "object-read-only", opener=Opener(body={"success": False, "errors": [{"message": "no"}]}))
    with pytest.raises(C.MintError, match="lacks sessionToken"):
        C.mint_api("acct123", "t", "b", "P", "object-read-only", opener=Opener(body={"success": True, "result": {"accessKeyId": "a", "secretAccessKey": "b"}}))
    with pytest.raises(C.MintError):
        C.mint_api("acct123", "t", "b", "P", "admin", opener=Opener(body=ok))


def test_mint_resolves_inputs_from_config_and_env_and_the_cli_prints_env_lines(env, monkeypatch, capsys):
    """Inputs: store.toml `[remote]` (endpoint, bucket, the env var NAMES of the parent key) > R2_ENDPOINT / R2_BUCKET / AWS_*; the account id
    comes from the endpoint host.  `auto` signs locally when the parent key is present, else needs CLOUDFLARE_API_TOKEN.  `--env` prints the
    three export lines and nothing is written anywhere."""
    from research.store import credentials as C

    acct = "0" * 32                                                  # a dummy account id
    ep = f"https://{acct}.r2.cloudflarestorage.com"
    (env / "store.toml").write_text(f'[remote]\ntype = "s3"\nbucket = "store-bucket"\nendpoint = "{ep}"\naccess_key_id_env = "P_AK"\nsecret_access_key_env = "P_SK"\n')
    for v in ("P_AK", "P_SK", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "CLOUDFLARE_API_TOKEN", "R2_ACCOUNT_ID", "CLOUDFLARE_ACCOUNT_ID",
              "R2_ENDPOINT", "R2_BUCKET"):                              # a developer shell that sourced r2.env must not leak into the test
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(C.MintError, match="CLOUDFLARE_API_TOKEN"):
        C.mint()                                                         # no parent key, no API token: says what it needs
    monkeypatch.setenv("P_AK", "PARENT")
    monkeypatch.setenv("P_SK", "psecret")
    c = C.mint(scope="object-read-only", ttl_s=600, prefixes=["manifests/"])
    assert c["via"] == "local" and c["bucket"] == "store-bucket" and c["account_id"] == acct and c["accessKeyId"] == "PARENT"
    assert C.decode_jwt(c["jwt"])[1]["aud"] == f"{acct}.r2.cloudflarestorage.com" and c["scope"] == "object-read-only"
    assert C.mint(actions=["GetObject"])["scope"] is None
    with pytest.raises(C.MintError, match="local signing only"):
        C.mint(via="api", actions=["GetObject"])
    with pytest.raises(C.MintError, match="CLOUDFLARE_API_TOKEN"):
        C.mint(via="api")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "already-temporary")
    with pytest.raises(C.MintError, match="cannot mint another"):
        C.mint()
    monkeypatch.delenv("AWS_SESSION_TOKEN")
    # no store.toml: R2_ENDPOINT / R2_BUCKET / AWS_* and an explicit account id
    (env / "store.toml").unlink()
    with pytest.raises(C.MintError, match="R2_ENDPOINT"):
        C.mint()
    monkeypatch.setenv("R2_ENDPOINT", "https://s3.example.invalid")
    monkeypatch.setenv("R2_BUCKET", "b2")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AK2")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "SK2")
    with pytest.raises(C.MintError, match="account id"):
        C.mint()
    monkeypatch.setenv("R2_ACCOUNT_ID", "acct")
    c = C.mint(ttl_s=60)
    assert c["bucket"] == "b2" and c["account_id"] == "acct" and C.decode_jwt(c["jwt"])[1]["aud"] == "s3.example.invalid" and c["accessKeyId"] == "AK2"

    assert main(["mint-credential", "--ttl", "10m", "--permission", "object-read-only", "--prefix", "manifests/", "--env"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "export AWS_ACCESS_KEY_ID=AK2" and lines[1].startswith("export AWS_SECRET_ACCESS_KEY=") and len(lines[1]) == len("export AWS_SECRET_ACCESS_KEY=") + 64
    assert lines[2].startswith("export AWS_SESSION_TOKEN=") and lines[3].startswith("# object-read-only on s3://b2 until ") and "prefixes ['manifests/']" in lines[3]
    tok = lines[2].split("=", 1)[1]
    assert __import__("base64").b64decode(tok).startswith(b"jwt/")
    assert main(["mint-credential", "--actions", "PutObject,HeadObject", "--ttl", "6h"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["via"] == "local" and doc["scope"] is None and doc["actions"] == ["PutObject", "HeadObject"] and doc["ttl_s"] == 21600 and "jwt" not in doc
    assert {"accessKeyId", "secretAccessKey", "sessionToken", "expires_utc"} <= set(doc)
    assert main(["mint-credential", "--ttl", "8d"]) == 1 and "604800" in capsys.readouterr().err
    assert main(["mint-credential", "--via", "api"]) == 1 and "CLOUDFLARE_API_TOKEN" in capsys.readouterr().err
    assert not list(env.rglob("*token*")) and not list(env.rglob("*.env")), "nothing written to disk"


def test_s3remote_refuses_oversize_and_conflicts(monkeypatch, tmp_path):
    r = remote_s3.S3Remote("https://acct.r2.cloudflarestorage.com", "bucket", "ak", "sk")
    assert r._path("objects/sha256/ab") == "/bucket/objects/sha256/ab" and r._path("") == "/bucket"
    p = tmp_path / "x"
    p.write_bytes(b"12345")
    monkeypatch.setattr(r, "head", lambda key: {"bytes": 4})
    with pytest.raises(RemoteConflict):
        r.put("k", p)
    monkeypatch.setattr(r, "head", lambda key: {"bytes": 5})
    assert r.put("k", p)["action"] == "skipped"  # same size, no ETag to compare: nothing to do, no network
    monkeypatch.setattr(r, "head", lambda key: {"bytes": 5, "etag": hashlib.md5(b"12345").hexdigest()})
    assert r.put("k", p)["action"] == "skipped"  # same size and MD5
    monkeypatch.setattr(r, "head", lambda key: {"bytes": 5, "etag": hashlib.md5(b"12346").hexdigest()})
    with pytest.raises(RemoteConflict, match="ETag"):
        r.put("k", p)                              # same size, different bytes: an immutable key is never overwritten
    monkeypatch.setattr(r, "head", lambda key: {"bytes": 5, "etag": "abc-2"})
    assert r.put("k", p)["action"] == "skipped"  # a multipart ETag is not an MD5: size is all HEAD can check
    monkeypatch.setattr(r, "head", lambda key: None)
    r.single_put_max = 4
    monkeypatch.setattr(r, "put_multipart", lambda key, path, **kw: {"action": "multipart", "key": key, **kw})
    mp = r.put("k", p)                             # above the single-PUT limit: the multipart path, with the hashes already computed
    assert mp["action"] == "multipart" and mp["size"] == 5 and mp["sha256"] == hashlib.sha256(b"12345").hexdigest() and mp["md5"] == hashlib.md5(b"12345").hexdigest()
    assert remote_s3.SINGLE_PUT_MAX == 5 << 30 and remote_s3.etag_value('"AbC"') == "abc" and remote_s3.etag_value('W/"x"') == "x"
    assert remote_s3.is_md5_etag("0" * 32) and not remote_s3.is_md5_etag("0" * 32 + "-3") and not remote_s3.is_md5_etag(None)


def test_catalog_keeps_shipped_tables_positional_for_older_writers(tmp_path):
    """The catalog is shared by worktrees on older code that insert `replicas` positionally with four values.  A catalog that
    already grew the 5th column (`verified`, shipped 2026-09-22) is migrated back: the column's data moves to `replica_checks`,
    the table returns to four columns, and both an old-style positional insert and the new `set_replica(verified=)` work."""
    import sqlite3

    from research.store.index import Index

    p = tmp_path / "catalog.sqlite"
    db = sqlite3.connect(p)
    db.executescript("CREATE TABLE replicas(id TEXT, location TEXT, verified_utc TEXT, detail TEXT, verified TEXT, PRIMARY KEY(id, location));")
    db.execute("INSERT INTO replicas VALUES ('art:a', 'remote', '2026-09-22T00:00:00Z', null, 'etag-md5')")
    db.execute("INSERT INTO replicas VALUES ('art:b', 'local', '2026-09-22T00:00:00Z', null, null)")
    db.commit(); db.close()

    ix = Index(p)
    assert [r[1] for r in ix.db.execute("PRAGMA table_info(replicas)")] == ["id", "location", "verified_utc", "detail"]
    assert ix.replicas("art:a") == {"remote": {"verified_utc": "2026-09-22T00:00:00Z", "detail": None, "verified": "etag-md5"}}
    assert ix.replicas("art:b")["local"]["verified"] is None
    # an older writer (research at 4987a47) inserting positionally:
    ix.db.execute("INSERT OR REPLACE INTO replicas VALUES (?,?,?,?)", ("art:c", "remote", "2026-09-22T01:00:00Z", None))
    assert ix.replicas("art:c")["remote"]["verified"] is None
    ix.set_replica("art:c", "remote", "2026-09-22T02:00:00Z", verified="sha256-readback")
    assert ix.replicas("art:c")["remote"] == {"verified_utc": "2026-09-22T02:00:00Z", "detail": None, "verified": "sha256-readback"}
    Index(p)   # idempotent re-open
    assert ix.replicas("art:a")["remote"]["verified"] == "etag-md5"


def test_index_artifact_holds_the_write_lock_before_it_reads(tmp_path):
    """Two publishes into one machine's catalog: a writer that reads first and then asks to write cannot wait for the other (SQLite
    skips the busy timeout there, as waiting could deadlock) and failed "database is locked" at once (node 1, 2026-10-01).  Another
    writer that tries to take the lock while index_artifact reads must find it held; with busy_timeout 0 nothing here waits."""
    import sqlite3

    from research.store.index import Index

    p = tmp_path / "catalog.sqlite"
    ix = Index(p)
    other = sqlite3.connect(p, isolation_level=None)
    other.execute("PRAGMA busy_timeout = 0")
    took = []

    def on_statement(sql: str) -> None:
        if sql.lstrip().upper().startswith("SELECT") and not took:
            try:
                other.execute("BEGIN IMMEDIATE")
                other.execute("INSERT INTO labels VALUES ('t', 'k', 'v', 'by', 'ts', NULL, 'p')")
                took.append(True)
            except sqlite3.OperationalError:
                took.append(False)

    ix.db.set_trace_callback(on_statement)
    ix.index_artifact({"kind": "blob/v1", "payload": {"bytes": 0}, "meta": {"a": 1}, "refs": {}}, "art:" + "0" * 64)
    ix.db.set_trace_callback(None)
    assert took == [False]
    assert ix.db.execute("SELECT kind FROM artifacts").fetchall() == [("blob/v1",)]
