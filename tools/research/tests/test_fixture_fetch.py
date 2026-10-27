"""The fetcher (`fetchset.py`), the registry's fetched entries (`fixtures.fetch_registry`, `hydrate_stubs`) and the public export
(`fixtures.export`, `verify_url`, `remote.load_export`): every byte checked against its id, whatever a source returns."""
from __future__ import annotations

import http.server
import subprocess
import threading
from functools import partial
from pathlib import Path

import pytest

from research.store import fetchset, fixtures as F
from research.store.cli import main as data_main
from research.store.local import LocalStore, manifest_key, object_key, sha256_file
from research.store.remote import ConfigError, FsRemote, HttpRemote, load_export


@pytest.fixture
def served(tmp_path):
    """A loopback HTTP server over a directory: (root dir, base URL)."""
    root = tmp_path / "public"
    root.mkdir()
    handler = partial(_Quiet, directory=str(root))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    t.start()
    yield root, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def _store(tmp_path: Path, name: str) -> LocalStore:
    return LocalStore(tmp_path / name)


def _publish(src: LocalStore, art: str, dest: FsRemote) -> None:
    """Copy one artifact's manifest and blobs from a store into a directory remote, as `export` lays them out."""
    hex_ = art.split(":", 1)[1]
    m = src.get_manifest(art)
    for h, _ in m.blobs():
        dest.put(object_key(h), src.object_path(h))
    dest.put(manifest_key(hex_), src.manifest_path(hex_))


def _put_file(st: LocalStore, path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return st.put_artifact(F.FIXTURE_KIND, {}, None, file=path)


# ---- fetchset ---------------------------------------------------------------------------------------------------------------------

def test_fetch_set_takes_any_ids_and_checks_every_byte(tmp_path):
    origin = _store(tmp_path, "origin")
    a = _put_file(origin, tmp_path / "src" / "a.bin", b"a" * 1000)
    tree = tmp_path / "src" / "t"
    (tree / "sub").mkdir(parents=True)
    (tree / "x.json").write_text("{}")
    (tree / "sub" / "y.bin").write_bytes(b"y" * 5000)
    t = origin.put_artifact(F.FIXTURE_KIND, {}, None, tree=tree)
    remote = FsRemote(tmp_path / "remote")
    for art in (a, t):
        _publish(origin, art, remote)
    st = _store(tmp_path, "fresh")
    ms, rep = fetchset.fetch_set(st, [a, t, a], [remote], jobs=4)
    assert set(ms) == {a, t} and not rep["missing"]
    assert rep["manifests"]["fetched"] == 2 and rep["blobs"]["fetched"] == 3
    assert st.has_local(a) and st.has_local(t)
    _, again = fetchset.fetch_set(st, [a, t], [remote])
    assert again["manifests"]["local"] == 2 and again["blobs"]["local"] == 3 and again["blobs"]["fetched"] == 0


def test_a_tampered_copy_is_refused_and_the_next_source_is_tried(tmp_path, served):
    root, url = served
    origin = _store(tmp_path, "origin")
    a = _put_file(origin, tmp_path / "src" / "a.bin", b"honest bytes")
    good = FsRemote(tmp_path / "good")
    _publish(origin, a, good)
    _publish(origin, a, FsRemote(root))
    h = origin.get_manifest(a).payload.sha256
    blob = root / object_key(h)
    blob.chmod(0o644)
    blob.write_bytes(b"forged bytes")
    st = _store(tmp_path, "one")
    _, rep = fetchset.fetch_set(st, [a], [HttpRemote(url)])
    assert rep["missing"] and rep["missing"][0]["sha256"] == h and "hashing to" in rep["missing"][0]["tried"][0]
    assert not st.object_path(h).exists()
    st2 = _store(tmp_path, "two")
    _, rep2 = fetchset.fetch_set(st2, [a], [HttpRemote(url), good])
    assert not rep2["missing"] and st2.object_path(h).read_bytes() == b"honest bytes"


def test_a_manifest_that_does_not_hash_to_its_id_is_refused(tmp_path):
    origin = _store(tmp_path, "origin")
    a = _put_file(origin, tmp_path / "src" / "a.bin", b"a")
    remote = FsRemote(tmp_path / "remote")
    _publish(origin, a, remote)
    (tmp_path / "remote" / manifest_key(a.split(":")[1])).write_bytes(b'{"forged": true}')
    ms = fetchset.fetch_manifests(_store(tmp_path, "fresh"), [a], [remote])
    assert ms == {}


def test_the_http_source_reads_absent_as_absent(served):
    root, url = served
    (root / "objects").mkdir()
    r = HttpRemote(url)
    assert r.head("objects/sha256/" + "0" * 64) is None
    with pytest.raises(KeyError):
        r.get("objects/sha256/" + "0" * 64, root / "out")
    with pytest.raises(NotImplementedError):
        r.put("k", root / "out")


# ---- the registry ---------------------------------------------------------------------------------------------------------------

def _repo(tmp_path: Path, origin: LocalStore, remote: FsRemote) -> tuple[Path, dict[str, str]]:
    """A git repo whose registry holds a tracked entry, a fetched file and a fetched tree (gitignored)."""
    repo = tmp_path / "repo"
    (repo / "fixtures").mkdir(parents=True)
    small = repo / "pkg" / "tests" / "fixtures" / "small.json"
    small.parent.mkdir(parents=True)
    small.write_text('{"n": 1}\n')
    tracked = F.fixture_id(small)
    big = _put_file(origin, tmp_path / "src" / "big.proof", bytes(range(256)) * 400)
    tree = tmp_path / "src" / "set"
    tree.mkdir()
    (tree / "sub_00.proof").write_bytes(b"p" * 3000)
    (tree / "sub_00.stmt").write_bytes(b"s" * 30)
    tset = origin.put_artifact(F.FIXTURE_KIND, {}, None, tree=tree)
    for art in (big, tset):
        _publish(origin, art, remote)
    doc = {tracked: "pkg/tests/fixtures/small.json", big: "pkg/tests/fixtures/big.proof", tset: "pkg/tests/fixtures/set"}
    (repo / "fixtures" / "artifacts.json").write_text(F.dump_map(doc))
    (repo / ".gitignore").write_text("pkg/tests/fixtures/big.proof\npkg/tests/fixtures/set/\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"], check=True)
    return repo, {"tracked": tracked, "big": big, "set": tset}


def test_fetch_registry_materialises_fetched_entries_and_leaves_tracked_ones(tmp_path):
    origin, remote = _store(tmp_path, "origin"), FsRemote(tmp_path / "remote")
    repo, a = _repo(tmp_path, origin, remote)
    small = repo / "pkg" / "tests" / "fixtures" / "small.json"
    small.write_text('{"n": 2}\n')                                    # a local edit of a tracked fixture is never overwritten
    st = _store(tmp_path, "dev")
    rep = F.fetch_registry(st, repo, srcs=[remote], jobs=4)
    by = {e["path"]: e["status"] for e in rep["entries"]}
    assert rep["ok"] and by == {"pkg/tests/fixtures/small.json": "tracked", "pkg/tests/fixtures/big.proof": "fetched",
                                "pkg/tests/fixtures/set": "fetched"}
    assert small.read_text() == '{"n": 2}\n'
    assert F.fixture_id(repo / "pkg/tests/fixtures/big.proof") == a["big"]
    assert F.fixture_id(repo / "pkg/tests/fixtures/set") == a["set"]
    again = F.fetch_registry(st, repo, srcs=[])                        # offline once fetched
    assert again["ok"] and {e["status"] for e in again["entries"]} == {"tracked", "present"}


def test_a_stale_fetched_copy_is_restored_exactly(tmp_path):
    origin, remote = _store(tmp_path, "origin"), FsRemote(tmp_path / "remote")
    repo, a = _repo(tmp_path, origin, remote)
    st = _store(tmp_path, "dev")
    F.fetch_registry(st, repo, srcs=[remote])
    d = repo / "pkg" / "tests" / "fixtures" / "set"
    (d / "sub_00.stmt").write_bytes(b"edited")
    (d / "leftover.bin").write_bytes(b"x")
    chk = F.fetch_registry(st, repo, ["pkg/tests/fixtures/set"], srcs=[remote], check_only=True)
    assert not chk["ok"] and chk["entries"][0]["status"] == "stale"
    assert (d / "leftover.bin").exists()                              # --check writes nothing
    rep = F.fetch_registry(st, repo, ["pkg/tests/fixtures/set/sub_00.stmt"], srcs=[remote])
    assert rep["ok"] and [e["status"] for e in rep["entries"]] == ["fetched"]
    assert F.fixture_id(d) == a["set"] and not (d / "leftover.bin").exists()


def test_a_shipped_tree_with_no_source_counts_its_exact_entries_as_present(tmp_path):
    """A `git archive` (no .git) on a machine with no store credentials and no public URL: tracked entries arrive with the tree, and
    their bytes alone show they are exact; only an absent one is missing."""
    origin, remote = _store(tmp_path, "origin"), FsRemote(tmp_path / "remote")
    repo, a = _repo(tmp_path, origin, remote)
    F.fetch_registry(_store(tmp_path, "dev"), repo, srcs=[remote])
    subprocess.run(["rm", "-rf", str(repo / ".git")], check=True)
    rep = F.fetch_registry(_store(tmp_path, "pod"), repo, srcs=[])
    assert rep["ok"] and {e["path"]: e["status"] for e in rep["entries"]} == {
        "pkg/tests/fixtures/small.json": "present", "pkg/tests/fixtures/big.proof": "present", "pkg/tests/fixtures/set": "present"}
    (repo / "pkg/tests/fixtures/big.proof").unlink()
    rep = F.fetch_registry(_store(tmp_path, "pod2"), repo, srcs=[])
    assert not rep["ok"] and [e["status"] for e in rep["entries"] if e["path"] == "pkg/tests/fixtures/big.proof"] == ["missing"]


def test_a_fixture_no_source_holds_is_reported_missing(tmp_path):
    origin, remote = _store(tmp_path, "origin"), FsRemote(tmp_path / "remote")
    repo, _ = _repo(tmp_path, origin, remote)
    rep = F.fetch_registry(_store(tmp_path, "offline"), repo, srcs=[])
    assert not rep["ok"] and {e["status"] for e in rep["entries"] if e["path"] != "pkg/tests/fixtures/small.json"} == {"missing"}


class _Held:
    """A source that counts its downloads and holds the first one until `n` fetches have reached the manifests: every fetch is
    in flight at once while the first is mid-download."""

    def __init__(self, inner: FsRemote, n: int):
        self.inner, self.n, self.arrived, self.gets = inner, n, 0, {}
        self.cv = threading.Condition()

    def arrive(self) -> None:
        with self.cv:
            self.arrived += 1
            self.cv.notify_all()

    def get(self, key: str, dst: Path) -> None:
        with self.cv:
            self.gets[key] = self.gets.get(key, 0) + 1
            self.cv.wait_for(lambda: self.arrived >= self.n)
        self.inner.get(key, dst)

    def describe(self) -> str:
        return "held"


def test_concurrent_first_fetches_of_one_fixture_all_succeed_with_one_download_each(tmp_path, monkeypatch):
    """A check's pytest and lean-suites steps fetch into one fresh tree and one cold store at once (r20261025-194220-d9fa): each
    fetch must succeed with the exact bytes, and a fetch that finds another mid-download waits for it and uses its copy."""
    origin, remote = _store(tmp_path, "origin"), FsRemote(tmp_path / "remote")
    repo, a = _repo(tmp_path, origin, remote)
    n = 4
    src = _Held(remote, n)
    real = fetchset.fetch_manifests

    def arriving(*args, **kw):
        src.arrive()
        return real(*args, **kw)
    monkeypatch.setattr(fetchset, "fetch_manifests", arriving)
    start = threading.Barrier(n)
    reps: list = [None] * n

    def one(i: int) -> None:
        start.wait()
        reps[i] = F.fetch_registry(LocalStore(tmp_path / "cold"), repo, srcs=[src], jobs=4)

    ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert all(r is not None and r["ok"] for r in reps), reps
    for path in ("pkg/tests/fixtures/big.proof", "pkg/tests/fixtures/set"):
        assert {e["status"] for r in reps for e in r["entries"] if e["path"] == path} <= {"fetched", "present"}
    assert F.fixture_id(repo / "pkg/tests/fixtures/big.proof") == a["big"]
    assert F.fixture_id(repo / "pkg/tests/fixtures/set") == a["set"]
    assert sorted(p.name for p in (repo / "pkg/tests/fixtures/set").iterdir()) == ["sub_00.proof", "sub_00.stmt"]
    assert not list((repo / "pkg/tests/fixtures").glob(".tmp-*"))
    assert src.gets and set(src.gets.values()) == {1}, src.gets


def test_history_stubs_are_restored(tmp_path):
    origin, remote = _store(tmp_path, "origin"), FsRemote(tmp_path / "remote")
    repo, a = _repo(tmp_path, origin, remote)
    m = origin.get_manifest(a["big"])
    stub = repo / "pkg" / "tests" / "fixtures" / "old.proof"
    stub.write_bytes(f"verity-fixture/v1 sha256:{m.payload.sha256} bytes:{m.payload.bytes}\n"
                     "fetch: uv run research data fetch-fixtures --stubs\n".encode())
    subprocess.run(["git", "-C", str(repo), "add", "pkg/tests/fixtures/old.proof"], check=True)
    rep = F.hydrate_stubs(_store(tmp_path, "dev"), repo, srcs=[remote])
    assert rep["ok"] and rep["hydrated"] == ["pkg/tests/fixtures/old.proof"]
    assert sha256_file(stub) == (m.payload.sha256, m.payload.bytes)


def test_the_cli_fetches_through_the_public_url_alone(tmp_path, served):
    root, url = served
    origin = _store(tmp_path, "origin")
    repo, a = _repo(tmp_path, origin, FsRemote(root))
    rc = data_main(["fetch-fixtures", "--repo", str(repo), "--store", str(tmp_path / "cli"), "--public-url", url, "--no-private", "--json"])
    assert rc == 0 and F.fixture_id(repo / "pkg/tests/fixtures/big.proof") == a["big"]
    assert data_main(["fetch-fixtures", "--repo", str(repo), "--store", str(tmp_path / "cli"), "--check", "--public-url", ""]) == 0


# ---- the public export ----------------------------------------------------------------------------------------------------------

def test_export_copies_every_entry_and_verify_url_reads_it_all_back(tmp_path, served):
    root, url = served
    origin, private = _store(tmp_path, "origin"), FsRemote(tmp_path / "private")
    repo, a = _repo(tmp_path, origin, private)
    st = _store(tmp_path, "exporter")
    rep = F.export(st, repo, FsRemote(root), srcs=[private], jobs=4)
    assert rep["ok"] and rep["ids"] == 3 and rep["uploaded"] == rep["blobs"] + 3
    for art in a.values():
        assert (root / manifest_key(art.split(":")[1])).is_file()
    again = F.export(st, repo, FsRemote(root), srcs=[private])
    assert again["ok"] and again["uploaded"] == 0 and again["skipped"] == rep["uploaded"]
    ver = F.verify_url(repo, url)
    assert ver["ok"] and ver["manifests"] == 3 and not ver["missing"]
    obj = next((root / "objects" / "sha256").iterdir())
    obj.chmod(0o644)
    obj.write_bytes(b"tampered")
    assert not F.verify_url(repo, url)["ok"]


def test_export_also_copies_an_artifact_it_is_named_and_reports_one_nobody_holds(tmp_path):
    origin, private = _store(tmp_path, "origin"), FsRemote(tmp_path / "private")
    repo, _ = _repo(tmp_path, origin, private)
    doc = tmp_path / "status" / "status.json"
    doc.parent.mkdir()
    doc.write_text('{"schema": "verity/status/v0"}')
    extra = origin.put_artifact("verity/status/v0", {"commit": "ab" * 20}, None, file=doc)
    _publish(origin, extra, private)
    rep = F.export(_store(tmp_path, "exporter"), repo, FsRemote(tmp_path / "public"), srcs=[private], arts=[extra])
    assert rep["ok"] and rep["ids"] == 4
    assert (tmp_path / "public" / manifest_key(extra.split(":")[1])).is_file()
    nobody = "art:" + "2" * 64
    rep = F.export(_store(tmp_path, "again"), repo, FsRemote(tmp_path / "public"), srcs=[private], arts=[nobody])
    assert not rep["ok"] and any(nobody in e for e in rep["errors"])


def test_export_recovers_superseded_ids_from_history_and_warns_on_lost_ones(tmp_path):
    origin, private = _store(tmp_path, "origin"), FsRemote(tmp_path / "private")
    repo, a = _repo(tmp_path, origin, private)
    small = repo / "pkg" / "tests" / "fixtures" / "small.json"
    old = a["tracked"]
    small.write_text('{"n": 3}\n')
    new = F.fixture_id(small)
    lost = "art:" + "1" * 64
    doc = F.load_map(repo)
    doc[new] = doc.pop(old)
    doc["previous"] = {new: [lost, old]}
    (repo / "fixtures" / "artifacts.json").write_text(F.dump_map(doc))
    dest = FsRemote(tmp_path / "public")
    rep = F.export(_store(tmp_path, "exporter"), repo, dest, srcs=[private])
    assert rep["ok"] and not rep["errors"] and len(rep["warnings"]) == 1 and lost in rep["warnings"][0]
    assert (tmp_path / "public" / manifest_key(old.split(":")[1])).is_file()        # the committed old version, from git history


def test_the_export_is_never_the_stores_own_bucket(tmp_path, monkeypatch):
    env = {"RESEARCH_EXPORT_BUCKET": "research-store", "R2_BUCKET": "research-store", "R2_ENDPOINT": "https://example.invalid",
           "AWS_ACCESS_KEY_ID": "dummy", "AWS_SECRET_ACCESS_KEY": "dummy"}
    with pytest.raises(ConfigError, match="store's own bucket"):
        load_export(tmp_path / "absent.toml", environ=env)
    assert load_export(tmp_path / "absent.toml", environ={}) is None
    assert load_export(tmp_path / "absent.toml", environ={"RESEARCH_EXPORT_DIR": str(tmp_path)}).describe() == f"fs:{tmp_path}"
    cfg = tmp_path / "store.toml"
    cfg.write_text('[remote]\ntype = "fs"\npath = "/x"\n[export]\ntype = "fs"\npath = "/y"\n')
    assert load_export(cfg, environ={}).describe() == "fs:/y"
