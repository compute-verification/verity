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
from research.store.local import AttemptExists, IntegrityError, LocalStore, NotFound
from research.store.model import ArtifactManifest, Payload, TreeFile, attempt_errors, validate_attempt_doc
from research.store.remote import FsRemote, RemoteConflict, load_remote

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
    assert not os.access(store.object_path(m.payload.sha256), os.W_OK), "blobs are read-only"

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
    assert not os.access(p1, os.W_OK)
    assert store.labels("art:" + "4" * 64) == []
    with pytest.raises(ValueError):
        store.add_label("not-an-id", "k", "v", "me")
    with pytest.raises(ValueError):
        store.add_label(t, "k", math.nan, "me")
    assert [r["id"] for r in store.index.select_artifacts(label=("proof_class", "PARTIAL"))] == [] , "labels index targets; select joins on artifacts"


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

    assert main(["label", art, "proof_class", "COMPLETE", "--by", "red-team-1", "--ref", doc["id"]]) == 0 and out().endswith("-red-team-1.json")
    assert main(["label", art, "proof_class", "PARTIAL", "--by", "red-team-2"]) == 0 and Path(out()).is_file()
    assert main(["label", art, "proof_class", "COMPLETE", "--by", "red-team-2"]) == 0 and Path(out()).is_file()
    assert main(["label", art, "score", "0.5", "--by", "bot", "--value-json"]) == 0 and Path(out()).is_file()
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
        th.join(5)


def test_s3remote_refuses_oversize_and_conflicts(monkeypatch, tmp_path):
    r = remote_s3.S3Remote("https://acct.r2.cloudflarestorage.com", "bucket", "ak", "sk")
    assert r._path("objects/sha256/ab") == "/bucket/objects/sha256/ab" and r._path("") == "/bucket"
    p = tmp_path / "x"
    p.write_bytes(b"12345")
    monkeypatch.setattr(r, "head", lambda key: {"bytes": 4})
    with pytest.raises(RemoteConflict):
        r.put("k", p)
    monkeypatch.setattr(r, "head", lambda key: {"bytes": 5})
    r.put("k", p)  # same size: nothing to do, no network
    monkeypatch.setattr(r, "head", lambda key: None)
    monkeypatch.setattr(Path, "stat", lambda self: type("S", (), {"st_size": remote_s3.SINGLE_PUT_MAX + 1})())
    with pytest.raises(NotImplementedError, match="parts.json"):
        r.put("k", p)
