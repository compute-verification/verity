"""Store honing (2026-09-23): eviction, partial fetch / listing, labels durability, catalog schema versions, vocabulary hygiene,
tar-streamed pulls, dead-pod errors, multipart upload, parallel `where`.  Temp stores and an `FsRemote` directory only; no network."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from research.store import ids
from research.store.cli import main
from research.store.local import LocalStore, NotFound
from research.store.remote import FsRemote

RUN = "r20260922-105905-2021"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    monkeypatch.setenv("RESEARCH_REPO", str(tmp_path / "repo"))
    monkeypatch.setenv("RESEARCH_RUNS", str(tmp_path / "runs"))
    (tmp_path / "repo").mkdir()
    return tmp_path


@pytest.fixture
def fs_remote(env):
    (env / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{env / "vault"}"\n')
    return env / "vault"


@pytest.fixture
def store(env) -> LocalStore:
    return LocalStore()


def _attempt(run_id=RUN, outputs=None, state="done", start="2026-09-22T10:59:05Z") -> dict:
    return {"schema": "research/attempt/v0.1", "id": run_id, "tool": {"name": "bench_vu", "version": "1"}, "derivation": "drv:" + "a" * 64,
            "reuse_key": None, "argv": ["x"], "params": {}, "key_params": [], "inputs": {}, "conditions": {}, "key_conditions": [],
            "source": {"commit": "c" * 40, "dirty": False}, "closure": None, "execution": {"start_utc": start, "host": "h", "project": "verity", "campaign": "r21-x"},
            "cost": None, "status": {"state": state, "rc": 0, "validation": "passed"}, "outputs": outputs or {}, "telemetry": None}


def _tree(root: Path, files: dict[str, bytes]) -> Path:
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return root


# ---- item 1: evict ----------------------------------------------------------------------------------------------------------------------

def test_evict_store_only_verified_blobs_largest_first_and_honest_presence(store, fs_remote, tmp_path, capsys):
    from research.store import evict as EV

    big = _tree(tmp_path / "big", {"a.bin": os.urandom(3000), "b.bin": os.urandom(2000), "small.txt": b"tiny"})
    verified = store.put_artifact("run-files/v1", {"n": 1}, tree=big)
    pending = store.put_artifact("proof/v1", {"n": 2}, file=_tree(tmp_path / "p", {"p.bin": os.urandom(2500)}) / "p.bin")
    shared_src = tmp_path / "shared.bin"
    shared_src.write_bytes(os.urandom(2200))
    shared_a = store.put_artifact("proof/v1", {"n": 3}, file=shared_src)            # same blob held by a verified and an unverified artifact
    shared_b = store.put_artifact("proof/v1", {"n": 4}, file=shared_src)
    assert store.push(verified)["preserved"] and store.push(shared_a)["preserved"]
    store.push(shared_b, preserve=False)                                               # flagged remote, never verified -> held back
    store.fetch(verified)                                                              # a trees/ cache entry
    assert (store.root / "trees" / ids.hex_of(verified)).is_dir()

    free = {"n": 1000}
    plan = EV.plan_store(store, min_bytes=1000)
    assert [c[0] for c in plan["candidates"]] == [3000, 2000], "only blobs held by verified artifacts, largest first; the 4-byte file is below min_bytes"
    assert plan["held_back_bytes"] == 2200 and plan["trees_bytes"] == 5004

    dry = EV.evict_store(store, target_free_bytes=8000, dry_run=True, min_bytes=1000, free_fn=lambda: free["n"])
    assert dry["dry_run"] and dry["trees_removed"] and dry["evicted_bytes"] == 3000 and dry["free_after"] == 1000 + 5004 + 3000 and dry["reached_target"]
    assert (store.root / "trees" / ids.hex_of(verified)).is_dir() and store.has_local(verified), "a dry run removes nothing"

    def free_fn():
        return free["n"] + (0 if (store.root / "trees" / ids.hex_of(verified)).is_dir() else 5004) + sum(
            n for n in (3000, 2000) if not any(store.object_path(h).exists() and s == n for h, s in store.get_manifest(verified).blobs()))

    rep = EV.evict_store(store, target_free_bytes=1000 + 5004 + 3000 + 1, min_bytes=1000, free_fn=free_fn)
    assert not (store.root / "trees" / ids.hex_of(verified)).exists()
    assert [b["bytes"] for b in rep["evicted_blobs"]] == [3000, 2000] and rep["artifacts_touched"] == [verified] and rep["reached_target"]
    assert not store.has_local(verified) and store.index.artifact(verified)["local"] is False, "presence is refreshed for touched artifacts"
    assert store.has_local(pending) and store.has_local(shared_a) and store.has_local(shared_b), "pending and held-back blobs are never evicted"
    assert store.object_path(store.get_manifest(verified).payload.files[2].sha256).exists(), "small.txt was below min_bytes"
    assert (store.root / "labels").is_dir() and store.manifest_path(ids.hex_of(verified)).exists() and (store.root / "catalog.sqlite").exists()
    assert (store.fetch(verified, tmp_path / "back") / "a.bin").read_bytes() == (big / "a.bin").read_bytes(), "an evicted blob comes back from the remote"
    assert store.has_local(verified)

    assert main(["evict", "--target-free-gb", "0.000001", "--dry-run", "--min-mb", "0.001"]) == 0
    assert "would evict" in capsys.readouterr().out
    assert main(["evict", "--target-free-gb", "99999", "--dry-run", "--json"]) == 1, "an unreachable target is exit 1"
    assert json.loads(capsys.readouterr().out)["store"]["reached_target"] is False


def test_evict_runs_only_preserved_terminal_quiet_runs_and_leaves_a_restore_record(store, fs_remote, tmp_path, capsys):
    from research.store import evict as EV

    runs = tmp_path / "runs"
    files = {"proofs/sub_00.proof": os.urandom(4000), "proofs/sub_01.proof": os.urandom(3000), "result.json": b"{}"}
    rd = _tree(runs / RUN, files)
    art = store.put_artifact("run-files/v1", {"run": RUN}, tree=rd)
    store.put_attempt(_attempt(outputs={"run_files": art}))
    other = _tree(runs / "r20260922-105906-0001", {"proofs/x.proof": os.urandom(4000)})          # no attempt record
    store.put_attempt(_attempt(run_id="r20260922-105907-0002", outputs={"run_files": art}, state="running"))
    _tree(runs / "r20260922-105907-0002", {"proofs/y.proof": os.urandom(4000)})
    changed = _tree(runs / "r20260922-105908-0003", {"proofs/sub_00.proof": os.urandom(4000)})  # same size, different bytes vs the artifact
    store.put_attempt(_attempt(run_id="r20260922-105908-0003", outputs={"run_files": art}))
    old = time.time() - 3600
    for d in (rd, other, changed, runs / "r20260922-105907-0002"):
        for p in d.rglob("*"):
            os.utime(p, (old, old))

    assert EV.plan_runs(store, runs, min_bytes=1000) == [], "nothing is evictable before the artifact is PRESERVED"
    assert store.push(art)["preserved"]
    plan = EV.plan_runs(store, runs, min_bytes=1000, log=print)
    log = capsys.readouterr().out
    assert [p["run"] for p in plan] == [RUN] and [f["path"] for f in plan[0]["files"]] == ["proofs/sub_00.proof", "proofs/sub_01.proof"][::-1], "largest first"
    assert "no attempt record" in log and "state 'running'" in log and "sha256 differs" in log
    fresh = _tree(runs / "r20260922-105909-0004", {"proofs/z.proof": os.urandom(4000)})
    store.put_attempt(_attempt(run_id="r20260922-105909-0004", outputs={"run_files": art}))
    assert [p["run"] for p in EV.plan_runs(store, runs, min_bytes=1000)] == [RUN], "a run dir touched in the last 30 min is skipped"
    assert fresh.exists()

    free = {"n": 0}
    rep = EV.evict_runs(store, runs, target_free_bytes=5000, dry_run=True, min_bytes=1000, free_fn=lambda: free["n"])
    assert rep["evicted_bytes"] == 7000 and (rd / "proofs/sub_00.proof").exists()
    rep = EV.evict_runs(store, runs, target_free_bytes=5000, min_bytes=1000, free_fn=lambda: free["n"])
    assert rep["runs"] == [{"run": RUN, "artifact": art, "files": 2, "bytes": 7000}]
    assert not (rd / "proofs/sub_00.proof").exists() and (rd / "result.json").exists()
    rec = json.loads((rd / EV.EVICTED_FILE).read_text())
    assert rec["artifact"] == art and rec["restore"] == f"research data fetch {art} --to {rd}" and len(rec["files"]) == 2
    assert EV.plan_runs(store, runs, min_bytes=1000) == [], "an evicted run dir is not planned twice"
    got = store.fetch(art, rd, paths=["proofs/*"])
    assert (got / "proofs/sub_00.proof").read_bytes() == files["proofs/sub_00.proof"], "the restore command brings the bytes back in place"

    assert main(["evict", "--target-free-gb", "0", "--runs", "--runs-dir", str(runs), "--min-mb", "0.001", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["runs"]["runs_root"] == str(runs) and out["store"]["reached_target"]


# ---- item 1: fetch --path / --list -----------------------------------------------------------------------------------------------------------

def test_fetch_subtree_and_listing(store, fs_remote, tmp_path, capsys):
    files = {"rep1/sub_00.proof": b"p0", "rep1/sub_01.proof": b"p1", "rep2/sub_00.proof": b"q0", "system.bin": b"sys", "manifest.json": b"{}"}
    art = store.put_artifact("run-files/v1", {}, tree=_tree(tmp_path / "t", files))
    store.push(art)
    for h, _ in store.get_manifest(art).blobs():                                     # everything evicted: bytes come from the remote
        os.chmod(store.object_path(h), 0o644)
        store.object_path(h).unlink()
    rows = store.listing(art)
    assert [r["path"] for r in rows] == sorted(files) and all(not r["local"] and r["remote"] for r in rows)
    assert [r["path"] for r in store.listing(art, paths=["rep1/*"])] == ["rep1/sub_00.proof", "rep1/sub_01.proof"]
    assert [r["path"] for r in store.listing(art, paths=["rep2"])] == ["rep2/sub_00.proof"], "a bare directory name is its subtree"
    assert [r["path"] for r in store.listing(art, paths=["*.bin", "manifest.json"])] == ["manifest.json", "system.bin"]

    got = store.fetch(art, tmp_path / "out", paths=["rep1/*"])
    assert sorted(p.relative_to(got).as_posix() for p in got.rglob("*") if p.is_file()) == ["rep1/sub_00.proof", "rep1/sub_01.proof"]
    assert (got / "rep1/sub_01.proof").read_bytes() == b"p1"
    assert not store.has_local(art) and store.index.artifact(art)["local"] is False, "a partial fetch does not claim the whole tree is local"
    assert [r["local"] for r in store.listing(art)] == [False, True, True, False, False]
    with pytest.raises(NotFound, match="no file matches"):
        store.fetch(art, tmp_path / "out2", paths=["nope/*"])

    cached = store.fetch(art, paths=["system.bin"])                                  # partial into the trees/ cache ...
    assert cached == store.root / "trees" / ids.hex_of(art) and (cached / "system.bin").exists() and not (cached / "manifest.json").exists()
    full = store.fetch(art)                                                            # ... which a full fetch completes rather than trusts
    assert full == cached and (full / "manifest.json").exists() and store.has_local(art) and store.index.artifact(art)["local"]

    assert main(["fetch", art, "--list"]) == 0
    out, err = capsys.readouterr()
    assert out.count("local") == 5 and "5 file(s), 11 bytes; 5 local" in err
    assert main(["fetch", art, "--list", "--path", "rep2/*", "--json"]) == 0
    assert [r["path"] for r in json.loads(capsys.readouterr().out)] == ["rep2/sub_00.proof"]
    assert main(["fetch", art, "--to", str(tmp_path / "cli"), "--path", "rep1", "--path", "system.bin"]) == 0
    assert Path(capsys.readouterr().out.strip()) == tmp_path / "cli"
    assert sorted(p.relative_to(tmp_path / "cli").as_posix() for p in (tmp_path / "cli").rglob("*") if p.is_file()) == ["rep1/sub_00.proof", "rep1/sub_01.proof", "system.bin"]


# ---- item 2: labels durability -------------------------------------------------------------------------------------------------------------------

def _labels_by_hash(st: LocalStore) -> dict[str, dict]:
    from research.store import labels_sync as LS

    return {h: lab.to_doc() for h, (_, lab) in LS.local_labels(st).items()}


def test_labels_sync_is_a_union_that_never_overwrites(env, fs_remote, tmp_path, capsys):
    from research.store import labels_sync as LS

    a, b = LocalStore(tmp_path / "A"), LocalStore(tmp_path / "B")
    art = "art:" + "1" * 64
    a.add_label(art, "status", "verified", "alice", ts="2026-09-23T01:00:00Z")
    a.add_label(art, "note", "first", "alice", ts="2026-09-23T01:00:00Z")            # same second, same by -> ~2 locally
    b.add_label(art, "status", "rejected", "alice", ts="2026-09-23T01:00:00Z")        # same second, same by, DIFFERENT assertion on another machine
    b.add_label(RUN, "grade", "A", "bob")
    assert (a.labels_dir(art) / "20260923T010000Z-alice.json").exists() and (b.labels_dir(art) / "20260923T010000Z-alice.json").exists()

    ra = LS.sync(a)
    assert ra["local"] == 2 and ra["remote_before"] == 0 and len(ra["pushed"]) == 2 and ra["pulled"] == [] and ra["bad"] == []
    keys = sorted(p.relative_to(fs_remote).as_posix() for p in fs_remote.rglob("*.json"))
    assert len(keys) == 2 and all(LS.KEY_RE.match(k) and k.startswith(f"labels/art_{'1' * 64}/") for k in keys), keys
    rb = LS.sync(b)
    assert len(rb["pushed"]) == 2 and len(rb["pulled"]) == 2 and rb["already"] == 0
    ra2 = LS.sync(a)
    assert ra2["pushed"] == [] and len(ra2["pulled"]) == 2 and ra2["already"] == 2
    assert _labels_by_hash(a) == _labels_by_hash(b) and len(_labels_by_hash(a)) == 4, "both stores hold the union"
    assert sorted(p.relative_to(fs_remote).as_posix() for p in fs_remote.rglob("*.json")).__len__() == 4, "one immutable key per assertion"
    assert [(l.value, l.by) for l in a.labels(art, key="status")] == [("verified", "alice"), ("rejected", "alice")]
    assert [(l.value, l.by) for l in b.labels(art, key="status")] == [("rejected", "alice"), ("verified", "alice")], "history order: ts, then arrival"
    assert a.index.labels(art, key="status").__len__() == 2 and b.index.labels(RUN)[0]["value"] == "A", "pulled labels are indexed"
    assert LS.sync(a) == {**LS.sync(b), "local": 4} | {"pushed": [], "pulled": [], "already": 4, "bad": [], "local": 4, "remote_before": 4, "remote": f"fs:{fs_remote}"}, "idempotent"
    assert a.labels(RUN)[0].ts == b.labels(RUN)[0].ts and a.labels(RUN)[0].ref is None

    # a hand-formatted local file hashes like its canonical form and goes up as canonical bytes; a tampered remote key is refused, not landed
    (a.labels_dir(RUN)).mkdir(exist_ok=True)
    (a.labels_dir(RUN) / "20260923T020000Z-carol.json").write_text(json.dumps(
        {"schema": "research/label/v0.1", "target": RUN, "key": "note", "value": "pretty", "by": "carol", "ts": "2026-09-23T02:00:00Z", "ref": None}, indent=2))
    ra3 = LS.sync(a)
    assert len(ra3["pushed"]) == 1
    up = fs_remote / LS.PREFIX / RUN / f"{ra3['pushed'][0]}.json"
    assert up.read_bytes() == json.dumps({"by": "carol", "key": "note", "ref": None, "schema": "research/label/v0.1", "target": RUN, "ts": "2026-09-23T02:00:00Z",
                                          "value": "pretty"}, separators=(",", ":")).encode()
    bad = fs_remote / LS.PREFIX / RUN / f"{'f' * 64}.json"
    bad.write_bytes(json.dumps({"schema": "research/label/v0.1", "target": RUN, "key": "grade", "value": "F", "by": "mallory", "ts": "2026-09-23T03:00:00Z", "ref": None}).encode())
    rb2 = LS.sync(b)
    assert len(rb2["pulled"]) == 1 and len(rb2["bad"]) == 1 and "not its name" in rb2["bad"][0]
    assert [l.value for l in b.labels(RUN, key="grade")] == ["A"], "the tampered assertion never lands"

    assert main(["labels-sync", "--store", str(tmp_path / "A")]) == 1, "bad remote keys -> exit 1"
    out, err = capsys.readouterr()
    assert "5 local, 6 on fs:" in out and "1 bad remote key" in out and "not its name" in err
    assert main(["labels-sync", "--store", str(tmp_path / "A"), "--pull-only", "--json"]) == 1
    rep = json.loads(capsys.readouterr().out)
    assert rep["pushed"] == [] and rep["pulled"] == [] and rep["already"] == 5


def test_push_pending_carries_labels_and_reindex_remote_pulls_them(env, fs_remote, tmp_path, capsys):
    a = LocalStore(tmp_path / "A")
    art = a.put_artifact("proof/v1", {"n": 1}, file=_tree(tmp_path / "p", {"p.bin": b"proof"}) / "p.bin")
    a.add_label(art, "status", "verified", "alice", ref=RUN)
    a.add_label(RUN, "note", "x", "alice")
    assert main(["push", "--store", str(tmp_path / "A"), "--pending", "--no-labels"]) == 0
    assert "labels:" not in capsys.readouterr().out and not (fs_remote / "labels").exists()
    assert main(["push", "--store", str(tmp_path / "A"), "--pending"]) == 0
    out = capsys.readouterr().out
    assert "labels: 2 pushed, 0 already there" in out
    assert main(["push", "--store", str(tmp_path / "A"), "--pending", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["labels"]["already"] == 2

    c = LocalStore(tmp_path / "C")                                                    # a fresh machine: reindex --remote brings labels down with the manifests
    assert c.labels(art) == []
    assert main(["reindex", "--store", str(tmp_path / "C"), "--remote"]) == 0
    assert "2 labels (2 pulled from the remote)" in capsys.readouterr().out
    assert [(l.key, l.value, l.ref) for l in c.labels(art)] == [("status", "verified", RUN)] and c.get_manifest(art).kind == "proof/v1"
    assert c.index.labels(art)[0]["value"] == "verified"


def test_labels_backup_tarball_and_directory_import_by_union(env, tmp_path, capsys):
    import tarfile

    from research.store import labels_sync as LS

    a = LocalStore(tmp_path / "A")
    art = "art:" + "2" * 64
    a.add_label(art, "status", "verified", "alice", ts="2026-09-23T01:00:00Z")
    a.add_label(art, "note", "n", "alice", ts="2026-09-23T01:00:00Z")
    a.add_label(RUN, "grade", "B", "bob")
    tgz = tmp_path / "labels-backup.tgz"
    with tarfile.open(tgz, "w:gz") as tf:
        tf.add(a.root / "labels", arcname="labels")
    backup = a.put_artifact("labels-backup/v1", {"files": 3, "targets": 2, "taken_utc": "2026-09-23T01:38:50Z", "host": "laptop"}, file=tgz)

    d = LocalStore(tmp_path / "D")
    d.add_label(RUN, "grade", "B", "bob", ts=a.labels(RUN)[0].ts)                      # the same assertion already here -> skipped, not duplicated
    d.add_label(RUN, "grade", "C", "dave")
    (d.root / "manifests").mkdir(exist_ok=True)
    import shutil
    shutil.copy(a.manifest_path(ids.hex_of(backup)), d.manifest_path(ids.hex_of(backup)))
    shutil.copy(a.object_path(a.get_manifest(backup).payload.sha256), d.object_path(a.get_manifest(backup).payload.sha256))
    d.reindex()
    rep = LS.import_backup(d, backup)
    assert rep == {"seen": 3, "added": 2, "already": 1, "bad": 0}
    assert _labels_by_hash(d).keys() >= _labels_by_hash(a).keys() and len(_labels_by_hash(d)) == 4
    assert [l.value for l in d.labels(RUN, key="grade")].__len__() == 2
    assert LS.import_backup(d, backup) == {"seen": 3, "added": 0, "already": 3, "bad": 0}, "idempotent"
    with pytest.raises(ValueError, match="not labels-backup/v1"):
        LS.import_backup(d, d.put_artifact("proof/v1", {}, file=tgz))

    e = LocalStore(tmp_path / "E")
    (a.labels_dir(art) / "junk.json").write_text("{not json")
    assert main(["labels-sync", "--store", str(tmp_path / "E"), "--from-dir", str(a.root / "labels")]) == 0
    out, err = capsys.readouterr()
    assert "4 in " in out and "3 added, 0 already here, 1 unreadable" in out and "junk.json" in err
    assert _labels_by_hash(e).keys() == _labels_by_hash(a).keys()


# ---- item 3: catalog schema version --------------------------------------------------------------------------------------------------------------

def test_catalog_schema_version_forward_migration_and_refusal_of_newer(env, tmp_path, monkeypatch, capsys):
    import sqlite3

    from research.store import index as IX

    p = tmp_path / "store" / "catalog.sqlite"
    p.parent.mkdir(parents=True)
    db = sqlite3.connect(p)                                                                     # a v0 catalog as the 2026-09-22 code left it
    db.executescript(IX.SCHEMA.replace("CREATE TABLE IF NOT EXISTS replicas(id TEXT, location TEXT, verified_utc TEXT, detail TEXT, PRIMARY KEY(id, location));",
                                       "CREATE TABLE IF NOT EXISTS replicas(id TEXT, location TEXT, verified_utc TEXT, detail TEXT, verified TEXT, PRIMARY KEY(id, location));"))
    db.execute("INSERT INTO replicas VALUES ('art:a', 'remote', '2026-09-22T00:00:00Z', 'fs:x', 'etag-md5')")
    db.execute("INSERT INTO replicas VALUES ('art:b', 'remote', NULL, 'fs:x', NULL)")
    db.execute("INSERT INTO artifacts VALUES ('art:a', 'proof/v1', 1, 'file', '{}', 1, 1, NULL)")
    db.commit()
    assert IX.catalog_version(db) == 0
    db.close()

    ix = IX.Index(p)
    assert ix.migration == {"from": 0, "to": IX.SCHEMA_VERSION, "steps": ["v0->v1"]} and ix.version == IX.SCHEMA_VERSION == 1
    assert [r[1] for r in ix.db.execute("PRAGMA table_info(replicas)")] == ["id", "location", "verified_utc", "detail"]
    assert ix.replicas("art:a") == {"remote": {"verified_utc": "2026-09-22T00:00:00Z", "detail": "fs:x", "verified": "etag-md5"}}
    assert ix.replicas("art:b")["remote"]["verified"] is None and ix.artifact("art:a")["kind"] == "proof/v1"
    info = ix.info()
    assert info["schema_version"] == 1 and info["code_version"] == 1 and info["tables"]["replicas"] == 2 and info["tables"]["replica_checks"] == 1
    ix.close()
    ix2 = IX.Index(p)
    assert ix2.migration == {"from": 1, "to": 1, "steps": []}, "a current catalog is left alone"
    ix2.close()
    assert main(["catalog", "--store", str(p.parent)]) == 0
    out = capsys.readouterr().out
    assert "schema v1 (code v1); current" in out and "replicas                    2 rows" in out

    fresh = IX.Index(tmp_path / "fresh" / "catalog.sqlite")                                     # a brand-new catalog is stamped, not "migrated"
    assert fresh.version == IX.SCHEMA_VERSION and fresh.migration["from"] == 0
    fresh.close()

    db = sqlite3.connect(p)                                                                     # a catalog written by code from the future
    db.execute(f"PRAGMA user_version = {IX.SCHEMA_VERSION + 1}")
    db.execute("CREATE TABLE future_only(x)")
    db.commit()
    db.close()
    with pytest.raises(IX.CatalogVersionError, match=rf"schema v{IX.SCHEMA_VERSION + 1}; this `research` code knows v{IX.SCHEMA_VERSION}.*OLDER.*RESEARCH_CATALOG_IGNORE_VERSION"):
        IX.Index(p)
    assert main(["catalog", "--store", str(p.parent)]) == 3 and main(["select", "--store", str(p.parent), "--kind", "proof/v1"]) == 3
    err = capsys.readouterr().err
    assert "research data catalog:" in err and "Traceback" not in err and str(p) in err
    monkeypatch.setenv(IX.IGNORE_VERSION_ENV, "1")
    ix3 = IX.Index(p)
    assert ix3.version == IX.SCHEMA_VERSION + 1 and ix3.migration.get("ignored") and ix3.artifact("art:a")["kind"] == "proof/v1", "escape hatch: opened, not downgraded"
    ix3.close()
    assert main(["catalog", "--store", str(p.parent)]) == 0 and "opened under RESEARCH_CATALOG_IGNORE_VERSION" in capsys.readouterr().out

    monkeypatch.delenv(IX.IGNORE_VERSION_ENV)
    monkeypatch.setattr(IX, "SCHEMA_VERSION", 3)                                                # a future code version chains the steps
    monkeypatch.setitem(IX.MIGRATIONS, 1, lambda db: db.execute("CREATE TABLE side_one(x)"))
    monkeypatch.setitem(IX.MIGRATIONS, 2, lambda db: db.execute("CREATE TABLE side_two(x)"))
    db = sqlite3.connect(p)
    db.execute("PRAGMA user_version = 1")
    db.commit()
    db.close()
    ix4 = IX.Index(p)
    assert ix4.migration == {"from": 1, "to": 3, "steps": ["v1->v2", "v2->v3"]} and "side_two" in ix4.info()["tables"]
    ix4.close()
    monkeypatch.setitem(IX.MIGRATIONS, 3, lambda db: db.execute("CREATE TABLE boom(x); INSERT INTO nope VALUES (1)"))
    monkeypatch.setattr(IX, "SCHEMA_VERSION", 4)
    with pytest.raises(sqlite3.Error):
        IX.Index(p)
    db = sqlite3.connect(p)
    assert IX.catalog_version(db) == 3 and not db.execute("SELECT name FROM sqlite_master WHERE name='boom'").fetchall(), "a failed step rolls back whole"
    db.close()


# ---- item 8: parallel HEADs in `where` -----------------------------------------------------------------------------------------------------------

def test_where_heads_in_parallel(store, fs_remote, tmp_path, monkeypatch):
    import threading

    art = store.put_artifact("run-files/v1", {}, tree=_tree(tmp_path / "t", {f"f{i:03d}": bytes([i]) * (i + 1) for i in range(40)}))
    store.push(art)
    r = store.remote
    assert isinstance(r, FsRemote)
    live, peak, lock = {"n": 0}, {"n": 0}, threading.Lock()
    real_head = r.head

    def slow_head(key):
        with lock:
            live["n"] += 1
            peak["n"] = max(peak["n"], live["n"])
        try:
            time.sleep(0.02)
            return real_head(key)
        finally:
            with lock:
                live["n"] -= 1

    monkeypatch.setattr(r, "head", slow_head)
    t0 = time.monotonic()
    rows = {w["location"]: w for w in store.where(art, jobs=8)}
    dt = time.monotonic() - t0
    assert rows["remote"]["present"] and rows["remote"]["missing"] == 0
    assert peak["n"] > 1 and dt < 41 * 0.02, f"41 HEADs of 20 ms each took {dt:.2f}s with peak concurrency {peak['n']}"
    (fs_remote / "objects/sha256" / store.get_manifest(art).payload.files[3].sha256).unlink()
    rows = {w["location"]: w for w in store.where(art, jobs=8)}
    assert rows["remote"]["present"] is False and rows["remote"]["missing"] == 1
