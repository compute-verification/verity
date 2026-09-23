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
