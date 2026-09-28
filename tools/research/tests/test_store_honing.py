"""Store honing (2026-09-23): eviction, partial fetch / listing, labels durability, catalog schema versions, vocabulary hygiene,
tar-streamed pulls, dead-pod errors, multipart upload, parallel `where`.  Temp stores and an `FsRemote` directory only; no network."""
from __future__ import annotations

import json
import os
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
    os.utime(store.root / "trees" / ids.hex_of(verified), (0, 0))                     # fetched long ago

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


def test_evict_keeps_every_blob_a_registry_artifact_holds(store, fs_remote, tmp_path, capsys):
    """Node 1's runs fetch fixtures only from the public export, which lacked rec-coins on 11 Oct: a blob any artifact of the
    registry holds stays, even when another, unlisted artifact shares it."""
    from research.store import evict as EV

    blob = tmp_path / "coins.bin"
    blob.write_bytes(os.urandom(3000))
    fixture = store.put_artifact("proof/v1", {"n": 1}, file=blob)
    other = store.put_artifact("proof/v1", {"n": 2}, file=blob)
    loose = store.put_artifact("proof/v1", {"n": 3}, file=_tree(tmp_path / "l", {"l.bin": os.urandom(2000)}) / "l.bin")
    assert all(store.push(a)["preserved"] for a in (fixture, other, loose))
    plan = EV.plan_store(store, min_bytes=1000, keep=frozenset({fixture}))
    assert [c[0] for c in plan["candidates"]] == [2000] and plan["kept_bytes"] == 3000
    reg = tmp_path / "artifacts.json"
    reg.write_text(json.dumps({fixture: "verity/ml/flock/tests/fixtures/rec-coins", "previous": {}}))
    assert main(["evict", "--target-free-gb", "99999", "--min-mb", "0.001", "--keep-registry", str(reg), "--json"]) == 1
    rep = json.loads(capsys.readouterr().out)["store"]
    assert [b["artifacts"] for b in rep["evicted_blobs"]] == [[loose]] and rep["kept_bytes"] == 3000
    assert store.has_local(fixture) and store.has_local(other) and not store.has_local(loose)


def test_evict_leaves_a_tree_someone_fetched_within_a_day(store, tmp_path):
    """Node 1's hourly eviction runs below its mark all day, so clearing trees/ whole took the files a run's view links into
    from under the run (the vLLM regression's #11 rebuild, r20261003-173418-89e5): only entries nobody fetched within
    TREE_KEEP_S go, and a fetch, a cache hit too, refreshes its entry."""
    from research.store import evict as EV

    old = store.put_artifact("run-files/v1", {"n": 1}, tree=_tree(tmp_path / "old", {"a.bin": b"a" * 10}))
    new = store.put_artifact("run-files/v1", {"n": 2}, tree=_tree(tmp_path / "new", {"b.bin": b"b" * 20}))
    t_old, t_new = store.fetch(old), store.fetch(new)
    os.utime(t_old, (0, 0))
    os.utime(t_new, (0, 0))
    assert store.fetch(new) == t_new and t_new.stat().st_mtime > 0, "a cache hit refreshes its entry"
    fetched = t_new.stat().st_mtime
    rep = EV.evict_store(store, target_free_bytes=1, min_bytes=1, free_fn=lambda: 0, now=fetched + EV.TREE_KEEP_S - 1)
    assert rep["trees_removed"] and rep["trees_kept"] == 1
    assert not t_old.exists() and (t_new / "b.bin").read_bytes() == b"b" * 20
    rep = EV.evict_store(store, target_free_bytes=1, min_bytes=1, free_fn=lambda: 0, now=fetched + EV.TREE_KEEP_S + 1)
    assert rep["trees_removed"] and rep["trees_kept"] == 0 and not t_new.exists()


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
    for d in (rd, other, changed, runs / "r20260922-105907-0002"):
        for p in d.rglob("*"):
            os.utime(p, (1_600_000_000, 1_600_000_000))           # quiet since 2020

    assert EV.plan_runs(store, runs, min_bytes=1000) == [], "nothing is evictable before the artifact is PRESERVED"
    assert store.push(art)["preserved"]
    plan = EV.plan_runs(store, runs, min_bytes=1000, log=print)
    log = capsys.readouterr().out
    assert [p["run"] for p in plan] == [RUN] and [f["path"] for f in plan[0]["files"]] == ["proofs/sub_00.proof", "proofs/sub_01.proof"], "largest first"
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


def test_evict_runs_takes_the_files_of_a_preserved_tree_output_from_under_its_tree_path(store, fs_remote, tmp_path, capsys):
    """Node 1's dispatcher runs (2026-10-02): 465 GB of Build, verdict and slim-replay outputs, every artifact verified on the
    remote, and none evictable, because only the `run_files` tree was matched against the run dir."""
    from research.store import evict as EV

    runs = tmp_path / "runs"
    build = {"weights/w0.bin": os.urandom(5000), "weights/w1.bin": os.urandom(4000), "context.json": b"{}"}
    rd = _tree(runs / RUN, {**{f"outputs/build/{k}": v for k, v in build.items()}, "logs/stdout.log": os.urandom(2000)})
    (rd / "outputs.json").write_text(json.dumps({"schema": "research/outputs/v0.1", "outputs": [
        {"name": "build", "kind": "vllm-build/v1", "meta": {}, "tree": "outputs/build"}]}))
    build_art = store.put_artifact("vllm-build/v1", {"run": RUN}, tree=rd / "outputs/build")
    logs_art = store.put_artifact("run-files/v1", {"run": RUN}, tree=_tree(tmp_path / "rf", {"logs/stdout.log": (rd / "logs/stdout.log").read_bytes()}))
    store.put_attempt(_attempt(outputs={"run_files": logs_art, "build": build_art}))
    for p in rd.rglob("*"):
        os.utime(p, (1_600_000_000, 1_600_000_000))

    assert EV.plan_runs(store, runs, min_bytes=1000, log=print) == []
    assert "not verified on the remote" in capsys.readouterr().out
    assert store.push(build_art)["preserved"]
    (plan,) = EV.plan_runs(store, runs, min_bytes=1000)
    assert [(f["path"], f["artifact"]) for f in plan["files"]] == [("outputs/build/weights/w0.bin", build_art), ("outputs/build/weights/w1.bin", build_art)], \
        "the output's files, under its tree path, while the unpushed run_files tree keeps the log"
    assert plan["artifact"] == build_art and plan["restore"] == f"research data fetch {build_art} --to {rd / 'outputs/build'}"

    assert store.push(logs_art)["preserved"]
    (plan,) = EV.plan_runs(store, runs, min_bytes=1000)
    assert {f["path"] for f in plan["files"]} == {"outputs/build/weights/w0.bin", "outputs/build/weights/w1.bin", "logs/stdout.log"}
    rep = EV.evict_runs(store, runs, target_free_bytes=1, min_bytes=1000, free_fn=lambda: 0)
    assert rep["evicted_bytes"] == 11000 and not (rd / "outputs/build/weights/w0.bin").exists() and (rd / "outputs/build/context.json").exists()
    rec = json.loads((rd / EV.EVICTED_FILE).read_text())
    assert rec["artifact"] == logs_art and rec["restore"] == (f"research data fetch {logs_art} --to {rd} && "
                                                              f"research data fetch {build_art} --to {rd / 'outputs/build'}")
    store.fetch(build_art, rd / "outputs/build", paths=["weights/*"])
    assert (rd / "outputs/build/weights/w0.bin").read_bytes() == build["weights/w0.bin"], "the restore brings the output back in place"


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


# ---- item 4: vocab-check --------------------------------------------------------------------------------------------------------------------------

def test_vocab_check_reads_the_files_and_reports_every_category(store, tmp_path, capsys):
    from research.store import vocab_check as VC

    art, art2 = "art:" + "3" * 64, "art:" + "4" * 64
    store.add_label(art, "zk", "true", "b-sweep", ts="2026-09-22T10:00:00Z")                  # the word, not the bool (an old writer)
    store.add_label(art, "zk", True, "b-sweep", ts="2026-09-22T10:00:01Z")                    # fine
    store.add_label(art, "model_agrees", "false", "tc-probe-fp8")
    store.add_label(art2, "independently_verified", "true", "h100-bestvsbest")
    store.add_label(art, "candidate", "A", "a-lane")                                          # old spelling
    store.add_label(art, "candidate", "A-GKR", "a-lane")
    store.add_label(art, "batch", 4096, "fill-a100")                                          # old key
    store.add_label(art, "checker", "v1", "fill-a100")
    store.add_label(art, "notes", "x", "red-team-zk")
    store.add_label(art, "verify_note", "y", "verify")
    store.add_label(art, "verified", "ligero-verify", "b-sweep")                              # off the enum
    store.add_label(art, "path", "graphed", "h100-bestvsbest")                                # a ref as a label
    store.add_label(art2, "result_artifact", art, "fill-a100")
    store.add_label(art, "self_proved", True, "fill-a100")                                    # unknown key
    store.add_label(art, "subbatches", 32, "coordinator")                                     # a measurement
    store.add_label(art, "K", 1536, "a-lane")
    (store.labels_dir(art) / "zzz.json").write_text("{broken")
    rep = VC.audit(store)
    assert rep["labels"] == 17 and rep["findings"] == 14
    cats = rep["by_category"]
    assert list(cats["string_typed"]) == ["independently_verified (bool held as a string: 'true')", "model_agrees (bool held as a string: 'false')",
                                          "zk (bool held as a string: 'true')"], "by count, then name"
    assert cats["string_typed"]["zk (bool held as a string: 'true')"] == {"count": 1, "by": {"b-sweep": 1}, "examples": [art]}
    assert cats["alias_values"] == {"candidate=A -> A-GKR": {"count": 1, "by": {"a-lane": 1}, "examples": [art]}}
    assert set(cats["alias_keys"]) == {"batch -> B", "checker -> verifier", "notes -> note", "verify_note -> note"}
    assert cats["off_enum"] == {"verified=ligero-verify": {"count": 1, "by": {"b-sweep": 1}, "examples": [art]}}
    assert set(cats["ref_keys"]) == {"path", "result_artifact"} and cats["ref_keys"]["result_artifact"]["examples"] == [art2]
    assert cats["unknown_keys"] == {"self_proved": {"count": 1, "by": {"fill-a100": 1}, "examples": [art]}}
    assert cats["measurement_keys"] == {"subbatches": {"count": 1, "by": {"coordinator": 1}, "examples": [art]}}
    assert list(cats["unreadable"]) == ["JSONDecodeError"]
    assert main(["vocab-check"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("17 assertion file(s); 14 off the vocabulary") and "\nstring_typed (3):" in out and "zk (bool held as a string: 'true')" in out and "by b-sweep" in out
    assert main(["vocab-check", "--strict"]) == 1
    capsys.readouterr()
    assert main(["vocab-check", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["by_category"]["alias_keys"]["batch -> B"]["count"] == 1
    clean = LocalStore(tmp_path / "clean")
    clean.add_label(art, "zk", True, "x")
    assert main(["vocab-check", "--store", str(tmp_path / "clean"), "--strict"]) == 0
    assert capsys.readouterr().out.strip() == "1 assertion file(s); 0 off the vocabulary"


# ---- item 5: `data pull` of a run-files tree with thousands of small files -------------------------------------------------------------------------

def _fake_ssh(monkeypatch, tmp_path, log: list):
    """`pull --from m` against a "machine" whose store is `tmp_path/machine/store`, over a local shell instead of ssh."""
    import subprocess

    from research.store import cli_prov

    def ssh_run(ssh, remote_argv, *, stdin=None, timeout=120):
        log.append((remote_argv, len(stdin or b"")))
        if remote_argv[0] == "cat":
            p = Path(remote_argv[1])
            return subprocess.CompletedProcess(remote_argv, 0 if p.is_file() else 1, p.read_bytes() if p.is_file() else b"", b"" if p.is_file() else b"no such file")
        assert remote_argv[:4] == ["tar", "-cf", "-", "-C"] and remote_argv[-2:] == ["-T", "-"], "blob paths travel on tar's stdin, not in the argv"
        assert sum(len(a) for a in remote_argv) < 4096
        return subprocess.run(remote_argv, input=stdin, capture_output=True)

    monkeypatch.setattr(cli_prov.remote, "ssh_run", ssh_run)
    monkeypatch.setattr(cli_prov.remote, "resolve_machine", lambda name, project, machines=None: {"name": name, "root": str(tmp_path / "machine"), "python": "python3"})
    monkeypatch.setattr(cli_prov.remote, "pod_target", lambda m: {"host": "h", "port": 22})
    monkeypatch.setattr(cli_prov.remote, "ssh_argv", lambda host, port, **k: ["ssh", host])
    return cli_prov


@pytest.mark.slow
def test_pull_streams_ten_thousand_small_files_in_a_few_tars_with_identical_layout(tmp_path, monkeypatch, capsys):
    from research.store import cli_prov as CP

    N = 10_000
    machine = LocalStore(tmp_path / "machine" / "store")
    tree = _tree(tmp_path / "tree", {f"rep{i % 4}/proof-{i:05d}.bin": f"proof {i}\n".encode() * (1 + i % 3) for i in range(N)} | {"summary.json": b"{}"})
    art = machine.put_artifact("run-files/v1", {"run": RUN}, tree=tree)
    machine.put_attempt(_attempt(outputs={"run_files": art}))
    log: list = []
    cli_prov = _fake_ssh(monkeypatch, tmp_path, log)
    monkeypatch.setattr(CP.A, "open_store", lambda store=None, root=None: LocalStore(tmp_path / "local"))
    monkeypatch.setattr(CP, "PULL_BATCH_FILES", 3000)
    rc = cli_prov.main(["pull", "--from", "m", RUN, "--project", "verity", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0, out
    tars = [a for a, _ in log if a[0] == "tar"]
    p = out["pulled"][0]
    assert p["id"] == art and p["blobs"] == N + 1 and p["batches"] == len(tars) - 1 == 4 and p["fetched"] == N + 1 and p["reused"] == 0, p   # 10001 distinct blobs / 3000 per tar, + one tar for the labels
    assert out["labels"] == {"seen": 0, "added": 0, "already": 0, "bad": 0}, "no assertions on the machine: nothing pulled, nothing invented"
    local = LocalStore(tmp_path / "local")
    got = local.fetch(art)
    assert sorted(q.relative_to(got).as_posix() for q in got.rglob("*") if q.is_file()) == sorted(q.relative_to(tree).as_posix() for q in tree.rglob("*") if q.is_file())
    assert (got / "rep3/proof-09999.bin").read_bytes() == (tree / "rep3/proof-09999.bin").read_bytes() and (got / "summary.json").read_bytes() == b"{}"
    assert local.get_attempt(RUN) == machine.get_attempt(RUN) and local.has_local(art)
    assert not list((tmp_path / "local" / "scratch").iterdir()), "the pull's staging dir under scratch/ is gone"

    # a second artifact sharing most blobs: only the new ones cross the wire
    tree2 = _tree(tmp_path / "tree2", {f"rep0/proof-{i:05d}.bin": f"proof {i}\n".encode() * (1 + i % 3) for i in range(100)} | {"new.bin": b"new"})
    art2 = machine.put_artifact("run-files/v1", {"run": RUN + "x"}, tree=tree2)
    machine.put_attempt(_attempt(run_id="r20260922-105906-2022", outputs={"run_files": art2}))
    log.clear()
    rc = cli_prov.main(["pull", "--from", "m", "r20260922-105906-2022", "--project", "verity", "--json"])
    p = json.loads(capsys.readouterr().out)["pulled"][0]
    assert rc == 0 and p["fetched"] == 1 and p["reused"] == 100 and p["batches"] == 1
    assert [a for a, _ in log if a[0] == "tar"][0][-2:] == ["-T", "-"]
    assert LocalStore(tmp_path / "local").fetch(art2).joinpath("new.bin").read_bytes() == b"new"


def test_pull_batches_by_count_and_by_bytes():
    from research.store import cli_prov as CP

    hexes = [f"{i:064x}" for i in range(10)]
    assert CP._batches([], None) == []
    assert CP._batches(hexes, None) == [hexes]
    sizes = {h: 200 * 2**20 for h in hexes}               # 200 MiB each: two per 512 MiB batch
    assert [len(b) for b in CP._batches(hexes, sizes)] == [2, 2, 2, 2, 2]
    sizes[hexes[0]] = 5 * 2**30                           # one blob over the byte cap still travels, alone
    assert [len(b) for b in CP._batches(hexes, sizes)] == [1, 2, 2, 2, 2, 1]
    assert [h for b in CP._batches(hexes, sizes) for h in b] == hexes, "order preserved"


# ---- item 6: a dead pod is a refusal, not a SystemExit -------------------------------------------------------------------------------------------

def _http_error(code: int, body: bytes = b"") -> Exception:
    import io
    import urllib.error

    return urllib.error.HTTPError("https://rest.runpod.io/v1/pods/p1", code, "x", {}, io.BytesIO(body))


def test_dead_pod_is_a_clean_refusal_with_the_pod_name(tmp_path, monkeypatch, capsys):
    import urllib.error

    from research import remote
    from research.cli import main as research_main
    from research.pods import part, runpod

    monkeypatch.setattr(runpod, "CONFIG", tmp_path / "config.toml")
    (tmp_path / "config.toml").write_text('apikey = "k"\napiurl = "https://api.runpod.io/graphql"\n')
    calls: list[tuple[str, str]] = []

    def urlopen(request, timeout=0):
        calls.append((request.get_method(), request.full_url))
        raise _http_error(404, b'{"error":"pod not found"}')

    monkeypatch.setattr(runpod.urllib.request, "urlopen", urlopen)
    with pytest.raises(runpod.PodError) as ei:
        runpod.get("p1")
    assert ei.value.code == 404 and ei.value.pod_id == "p1" and "HTTP 404" in str(ei.value) and "terminated" in str(ei.value)
    assert not isinstance(ei.value, SystemExit) and calls == [("GET", "https://rest.runpod.io/v1/pods/p1")]

    machine = {"provider": "runpod", "pod_id": "p1", "name": "pod-x"}
    with pytest.raises(remote.Refused, match=r"pod p1 \(machine pod-x\) no longer exists on runpod \(HTTP 404"):
        remote.pod_target(machine)
    with pytest.raises(remote.Refused, match=r"pod p1: GET /pods/p1 -> HTTP 404"):
        part.check_pod("p1")

    def urlopen_down(request, timeout=0):
        raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))

    monkeypatch.setattr(runpod.urllib.request, "urlopen", urlopen_down)
    with pytest.raises(remote.Refused, match=r"pod p1 \(machine pod-x\): runpod API unavailable: GET /pods/p1: URLError"):
        remote.pod_target(machine)
    monkeypatch.setattr(runpod, "CONFIG", tmp_path / "absent.toml")
    with pytest.raises(remote.Refused, match="no RunPod API config"):
        remote.pod_target(machine)
    monkeypatch.setattr(runpod, "CONFIG", tmp_path / "config.toml")
    monkeypatch.setattr(runpod.urllib.request, "urlopen", urlopen)

    # the CLI: `research run --on <dead pod>` is exit 3 with the pod's name on stderr and no traceback; the run is recorded as refused
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    (tmp_path / "machines.toml").write_text('[machines.pod-x]\nprovider = "runpod"\npod_id = "p1"\nproject = "verity"\n')
    rc = research_main(["run", "--runs-dir", str(tmp_path / "runs"), "--on", "pod-x", "--project", "verity", "--no-custody-r2", "--", "true"])
    err = capsys.readouterr().err
    assert rc == 3 and "refused: pod p1 (machine pod-x) no longer exists on runpod (HTTP 404: terminated?)" in err and "Traceback" not in err
    runs = list((tmp_path / "runs").iterdir())
    assert len(runs) == 1 and json.loads((runs[0] / "remote.json").read_text())["phase"] == "refused"

    # the standalone runpod CLI keeps its one-line exit (SystemExit raised there, and only there)
    monkeypatch.setattr("sys.argv", ["runpod", "get", "p1"])
    with pytest.raises(SystemExit) as se:
        runpod.main()
    assert "runpod: GET /pods/p1 -> HTTP 404" in str(se.value)


# ---- item 7: multipart upload > 5 GiB: the chunking, pure ------------------------------------------------------------------------------------------

def test_multipart_chunking_is_pure_and_matches_the_service_limits(tmp_path):
    import hashlib

    from research.store import remote_s3 as S

    GiB, MiB = 1 << 30, 1 << 20
    assert S.SINGLE_PUT_MAX == 5 * GiB and S.PART_SIZE == 256 * MiB and S.MAX_PARTS == 10_000 and 5 * MiB <= S.PART_SIZE
    # part size: 256 MiB unless that would exceed 10 000 parts, then grown in whole MiB
    assert S.choose_part_size(5 * GiB + 1) == 256 * MiB and S.choose_part_size(2500 * GiB) == 256 * MiB   # 256 MiB x 10 000 = 2500 GiB
    assert S.choose_part_size(2500 * GiB + 1) == 257 * MiB and S.choose_part_size(2500 * GiB + 1) * S.MAX_PARTS >= 2500 * GiB + 1
    assert S.choose_part_size(10 * MiB, part_size=4096) == 4096 and S.choose_part_size(100 * MiB, part_size=4096) == 1 * MiB, "grown in whole MiB"
    # the plan: every part but the last is `part_size`, numbered from 1, offsets contiguous, lengths summing to the size
    plan = S.plan_parts(5 * GiB + 1, 256 * MiB)
    assert len(plan) == 21 and plan[0] == (1, 0, 256 * MiB) and plan[-1] == (21, 20 * 256 * MiB, 1) and sum(n for _, _, n in plan) == 5 * GiB + 1
    assert [p[0] for p in plan] == list(range(1, 22)) and all(plan[i][1] + plan[i][2] == plan[i + 1][1] for i in range(20))
    assert S.plan_parts(8, 4) == [(1, 0, 4), (2, 4, 4)] and S.plan_parts(10, 4) == [(1, 0, 4), (2, 4, 4), (3, 8, 2)] and S.plan_parts(0, 4) == [(1, 0, 0)]
    # the multipart ETag is md5 of the raw part MD5s, dash, part count -- what S3 and R2 return
    a, b = b"a" * 4, b"bb" * 3
    md5s = [hashlib.md5(a).hexdigest(), hashlib.md5(b).hexdigest()]
    assert S.multipart_etag(md5s) == hashlib.md5(hashlib.md5(a).digest() + hashlib.md5(b).digest()).hexdigest() + "-2"
    assert S.parts_of(S.multipart_etag(md5s)) == 2 and S.parts_of("0" * 32) is None and S.parts_of(None) is None and S.parts_of("abc-x") is None
    assert not S.is_md5_etag(S.multipart_etag(md5s))
    # hash_parts reads each part once and agrees with the plan and with whole-file hashes
    f = tmp_path / "f"
    f.write_bytes(a + b)
    parts = S.hash_parts(f, 4)
    assert [(p["n"], p["offset"], p["bytes"]) for p in parts] == [(1, 0, 4), (2, 4, 4), (3, 8, 2)]
    assert parts[0]["md5"] == md5s[0] and parts[0]["sha256"] == hashlib.sha256(a).hexdigest() and parts[2]["sha256"] == hashlib.sha256(b"bb").hexdigest()
    side = S.parts_sidecar("objects/sha256/x", 10, hashlib.sha256(a + b).hexdigest(), 4, parts)
    assert side["schema"] == "research/parts/v1" and side["etag"] == S.multipart_etag([p["md5"] for p in parts]) and side["part_bytes"] == 4
    assert [set(p) for p in side["parts"]] == [{"n", "bytes", "sha256", "md5"}] * 3 and S.sidecar_key("objects/sha256/x") == "objects/sha256/x.parts.json"


def test_multipart_upload_is_aborted_when_the_local_file_changes_under_it(tmp_path, monkeypatch):
    """`_upload_part` re-hashes the slice it reads: a file rewritten mid-upload is a MultipartError, and the upload is aborted."""
    from research.store import remote_s3 as S

    r = S.S3Remote("https://acct.r2.cloudflarestorage.com", "bucket", "ak", "sk")
    f = tmp_path / "f"
    f.write_bytes(b"x" * 10)
    calls: list[tuple[str, str, dict]] = []

    class _Resp:
        def __init__(self, body: bytes = b"", etag: str | None = None):
            self.body, self.headers = body, {"ETag": f'"{etag}"'} if etag else {}

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_request(method, key, query=None, body=None, headers=None, payload_hash=None, stall_window=None):
        calls.append((method, key, dict(query or {})))
        if method == "POST" and "uploads" in (query or {}):
            return _Resp(b'<InitiateMultipartUploadResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><UploadId>U1</UploadId></InitiateMultipartUploadResult>')
        if method == "PUT":
            import hashlib

            f.write_bytes(b"y" * 10)                                    # the file changes after the parts were hashed: the NEXT part's slice differs
            return _Resp(etag=hashlib.md5(body).hexdigest())
        return _Resp()

    monkeypatch.setattr(r, "_request", fake_request)
    monkeypatch.setattr(r, "head", lambda key: None)
    with pytest.raises(S.MultipartError, match="changed while uploading"):
        r.put_multipart("k", f, part_size=4, jobs=1)
    assert calls[0] == ("POST", "k", {"uploads": ""}) and calls[1] == ("PUT", "k", {"partNumber": "1", "uploadId": "U1"}), "initiated, part 1 up"
    assert calls[-1] == ("DELETE", "k", {"uploadId": "U1"}) and len(calls) == 3, "part 2's slice no longer hashes as planned: aborted, never completed"

    # a service whose part ETag is not the part's MD5 is caught the same way
    f.write_bytes(b"x" * 10)
    calls.clear()

    def bad_etag(method, key, query=None, body=None, headers=None, payload_hash=None, stall_window=None):
        calls.append((method, key, dict(query or {})))
        if method == "POST" and "uploads" in (query or {}):
            return _Resp(b"<InitiateMultipartUploadResult><UploadId>U2</UploadId></InitiateMultipartUploadResult>")
        return _Resp(etag="0" * 32) if method == "PUT" else _Resp()

    monkeypatch.setattr(r, "_request", bad_etag)
    with pytest.raises(S.MultipartError, match="service ETag 0{32} != local MD5"):
        r.put_multipart("k", f, part_size=4, jobs=1)
    assert calls[-1] == ("DELETE", "k", {"uploadId": "U2"})


# ---- item 8: parallel HEADs in `where` -----------------------------------------------------------------------------------------------------------

def test_where_heads_in_parallel(store, fs_remote, tmp_path, monkeypatch):
    import threading

    art = store.put_artifact("run-files/v1", {}, tree=_tree(tmp_path / "t", {f"f{i:03d}": bytes([i]) * (i + 1) for i in range(40)}))
    store.push(art)
    r = store.remote
    assert isinstance(r, FsRemote)
    calls, lock = {"n": 0}, threading.Lock()
    together = threading.Barrier(2)                               # the first two HEADs get past it only if both are in flight at once
    real_head = r.head

    def meeting_head(key):
        with lock:
            calls["n"] += 1
            first_two = calls["n"] <= 2
        if first_two:
            together.wait()
        return real_head(key)

    monkeypatch.setattr(r, "head", meeting_head)
    rows = {w["location"]: w for w in store.where(art, jobs=8)}
    assert rows["remote"]["present"] and rows["remote"]["missing"] == 0 and calls["n"] == 41
    (fs_remote / "objects/sha256" / store.get_manifest(art).payload.files[3].sha256).unlink()
    rows = {w["location"]: w for w in store.where(art, jobs=8)}
    assert rows["remote"]["present"] is False and rows["remote"]["missing"] == 1
