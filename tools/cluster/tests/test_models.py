"""`cluster models`: the per-node model-checkpoint cache, offline. A fake Hugging Face (metadata in the shape
`/api/models/<repo>/revision/<rev>?blobs=true` returns, files served from memory) stands in for the network, and the disk,
the clock and the /proc scan are passed in."""
from __future__ import annotations

import hashlib
import io
import json
import os
import urllib.request
from pathlib import Path

import pytest

from cluster import cli, model, models
from cluster.models import GB, Refused, Scan

T = model.instant("2026-10-01T02:00:00Z")
REV = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
REPO = "org/tiny"
NAME = f"org--tiny@{REV}"
WEIGHTS = bytes(range(256)) * 512
FILES = {"config.json": b'{"model_type": "llama"}\n', "model.safetensors": WEIGHTS, "tok/tokenizer.json": b'{"v": [1, 2]}'}
LFS = {"model.safetensors"}
CFG = models.node_config({"quota_gb": {"default": 200, "circuits": 500}, "max_use_pct": 85})


def disk(used_gb: float, avail_gb: float):
    """A filesystem with this much used and available, as `df` counts them."""
    return lambda root: (int(used_gb * GB), int(avail_gb * GB))


ROOMY = disk(10, 90)


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def blob(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


class Hub:
    """Hugging Face as the cache sees it: `metadata` and `get(path, start)`, serving `serve` (corruptible), cutting each
    stream in `cut` short once, at that byte, and recording every request."""
    authenticated = False

    def __init__(self, files=FILES, lfs=LFS, rev=REV, gated=False, serve=None, cut=None, ignore_range=False, drop=()):
        self.files, self.lfs, self.rev, self.gated = dict(files), set(lfs), rev, gated
        self.serve, self.cut, self.ignore_range, self.drop = dict(serve or files), dict(cut or {}), ignore_range, set(drop)
        self.gets: list[tuple[str, int]] = []
        self.during = None

    def metadata(self, repo, rev):
        sib = []
        for p, d in self.files.items():
            s = {"rfilename": p, "size": len(d), "blobId": blob(d)}
            if p in self.lfs:
                s.update(blobId="1" * 40, lfs={"sha256": sha256(d), "size": len(d), "pointerSize": 134})
            if p in self.drop:
                s.pop("blobId"), s.pop("lfs", None)
            sib.append(s)
        return {"sha": self.rev, "gated": self.gated, "siblings": sib}

    def get(self, repo, rev, path, start):
        self.gets.append((path, start))
        if self.during:
            self.during()
        data, start = self.serve[path], 0 if self.ignore_range else start
        if path in self.cut:
            return start, io.BytesIO(data[start:self.cut.pop(path)])
        return start, io.BytesIO(data[start:])


def fetch(root, hub=None, *, owner="circuits", now=T, disk=ROOMY, **kw):
    kw.setdefault("phase", "A")
    kw.setdefault("declared_gb", 1)
    return models.fetch(root, models.ref(f"{REPO}@{REV}"), owner=owner, now=now, hub=hub or Hub(), cfg=CFG, disk=disk,
                        sleep=lambda s: None, **kw)


def entry(root: Path, repo: str, rev: str, *, owner: str, gb: float, expires: int, expires_h: float = 72) -> Path:
    """An entry written as a fetch leaves one, holding one small file but claiming `gb`."""
    d = root / models.Ref(repo, rev).name
    d.mkdir(parents=True)
    (d / "config.json").write_bytes(b"{}")
    models.write_entry(d, {"schema": models.SCHEMA, "repo": repo, "revision": rev, "owner": owner, "phase": "A",
                           "files": {"config.json": {"sha256": sha256(b"{}"), "bytes": 2}}, "total_bytes": int(gb * GB),
                           "declared_gb": gb, "expires_h": expires_h, "include": None, "created": models.iso(T),
                           "last_used": models.iso(T), "expires": models.iso(expires), "fetched_by": None})
    return d


def leftovers(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if ".locks" not in p.parts)


def test_a_fetch_takes_a_pinned_commit_or_nothing():
    for bad in (f"{REPO}@main", f"{REPO}@v1.0", f"{REPO}@{REV[:12]}", f"{REPO}@{REV.upper()}", REPO, f"tiny@{REV}",
                f"a--b/c@{REV}", f"../x@{REV}"):
        with pytest.raises(Refused) as e:
            models.ref(bad)
        assert e.value.code == 2, bad
    assert "pinned or nothing" in str(pytest.raises(Refused, models.ref, f"{REPO}@main").value)
    assert models.ref(f"{REPO}@{REV}").name == NAME


def test_the_cli_refuses_a_branch_in_one_line(tmp_path, capsys):
    assert cli.main(["models", "fetch", f"{REPO}@main", "--owner", "circuits", "--phase", "A", "--declared-gb", "1",
                     "--root", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "isn't a commit" in err and "pinned or nothing" in err
    assert not list(tmp_path.iterdir())


def test_a_fetch_writes_the_repos_files_as_plain_files_and_its_entry_last(tmp_path):
    hub = Hub()
    placed = []
    hub.during = lambda: placed.append((tmp_path / NAME).exists())
    e, how = fetch(tmp_path, hub, fetched_by="r20261001-020000-abcd")
    assert how == "fetched" and placed and not any(placed)
    d = tmp_path / NAME
    assert sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()) == \
        [".entry.json", "config.json", "model.safetensors", "tok/tokenizer.json"]
    assert not any(p.is_symlink() for p in d.rglob("*"))
    assert leftovers(tmp_path) == [".tmp", NAME, *(f"{NAME}/{p}" for p in (".entry.json", "config.json", "model.safetensors",
                                                                            "tok", "tok/tokenizer.json"))]
    assert json.loads((d / ".entry.json").read_text()) == e
    assert e["schema"] == "verity/model-entry/v1" and (e["repo"], e["revision"]) == (REPO, REV)
    assert e["files"] == {p: {"sha256": sha256(b), "bytes": len(b)} for p, b in FILES.items()}
    assert e["total_bytes"] == sum(map(len, FILES.values()))
    assert (e["owner"], e["phase"], e["declared_gb"], e["expires_h"], e["fetched_by"]) == ("circuits", "A", 1, 72, "r20261001-020000-abcd")
    assert (e["created"], e["last_used"], e["expires"]) == ("2026-10-01T02:00:00Z", "2026-10-01T02:00:00Z", "2026-10-04T02:00:00Z")
    assert (d / "model.safetensors").read_bytes() == WEIGHTS
    assert oct((d / "model.safetensors").stat().st_mode & 0o777) == "0o644"


def test_the_fetch_holds_the_entrys_lock_while_it_downloads(tmp_path):
    hub = Hub()
    seen = []
    hub.during = lambda: seen.append(models._held(tmp_path, NAME))
    fetch(tmp_path, hub)
    assert seen and all(seen) and not models._held(tmp_path, NAME)


def test_include_fetches_only_the_matching_files(tmp_path):
    e, _ = fetch(tmp_path, include=("*.json",))
    assert sorted(e["files"]) == ["config.json", "tok/tokenizer.json"] and e["include"] == ["*.json"]
    with pytest.raises(Refused, match="no files match"):
        fetch(tmp_path / "other", include=("*.bin",))


def test_a_file_without_a_checksum_or_metadata_of_another_commit_is_refused_and_nothing_is_written(tmp_path):
    with pytest.raises(Refused, match=r"tok/tokenizer.json has no checksum") as e:
        fetch(tmp_path, Hub(drop={"tok/tokenizer.json"}))
    assert e.value.code == 3
    with pytest.raises(Refused, match=r"model.safetensors has no checksum"):
        fetch(tmp_path, Hub(drop={"model.safetensors"}))
    with pytest.raises(Refused, match="metadata is of commit"):
        fetch(tmp_path, Hub(rev="0" * 40))
    assert leftovers(tmp_path) == []


@pytest.mark.parametrize("path", ["model.safetensors", "config.json"])
def test_a_checksum_mismatch_deletes_the_download(tmp_path, path):
    bad = bytearray(FILES[path])
    bad[3] ^= 1
    with pytest.raises(models.Mismatch, match=("sha256" if path in LFS else "git blob") + ".*deleted the download"):
        fetch(tmp_path, Hub(serve={**FILES, path: bytes(bad)}))
    assert leftovers(tmp_path) == [".tmp"] and models.entries(tmp_path) == {}


def test_a_file_longer_than_listed_is_a_mismatch(tmp_path):
    with pytest.raises(models.Mismatch, match="longer than"):
        fetch(tmp_path, Hub(serve={**FILES, "config.json": FILES["config.json"] + b" "}))
    assert leftovers(tmp_path) == [".tmp"]


def test_a_listed_total_over_the_declared_gb_is_refused(tmp_path):
    with pytest.raises(Refused, match=r"over the declared 0.0001 GB") as e:
        fetch(tmp_path, declared_gb=0.0001)
    assert e.value.code == 3 and leftovers(tmp_path) == []


def test_an_owners_quota_counts_its_entries_and_its_running_fetches_on_the_node(tmp_path):
    entry(tmp_path, "org/big", "1" * 40, owner="circuits", gb=499.9999, expires=T + 3600)
    entry(tmp_path, "org/theirs", "2" * 40, owner="infra", gb=150, expires=T + 3600)
    with pytest.raises(Refused, match=r"circuits's entries on this node would hold 500.0 GB .* over its quota of 500 GB") as e:
        fetch(tmp_path)
    assert e.value.code == 3 and not (tmp_path / NAME).exists() and list((tmp_path / ".tmp").iterdir()) == []
    with pytest.raises(Refused, match="over its quota of 200 GB"):
        _infra_over(tmp_path)
    assert fetch(tmp_path, owner="pous")[1] == "fetched"


def _infra_over(root: Path):
    """infra's 150 GB entry plus a 50.1 GB fetch of infra's still running (it holds its lock) leave no room for this one."""
    run = root / ".tmp" / f"org--run@{'3' * 40}-4242"
    run.mkdir(parents=True)
    (run / ".fetch.json").write_text(json.dumps({"owner": "infra", "total_bytes": int(50.1 * GB)}))
    with models._lock(root, f"org--run@{'3' * 40}"):
        fetch(root, owner="infra")


def test_a_fetch_is_refused_that_would_fill_the_filesystem_past_its_max(tmp_path):
    total = sum(map(len, FILES.values()))
    with pytest.raises(Refused, match=r"would be 85.0% used after this fetch .* over its max of 85%") as e:
        fetch(tmp_path, disk=lambda root: (85 * GB - total + 1, 15 * GB + total - 1))
    assert e.value.code == 3 and leftovers(tmp_path) == [".tmp"]
    assert fetch(tmp_path, disk=lambda root: (85 * GB - total, 15 * GB + total))[1] == "fetched"


def test_a_running_fetchs_remaining_bytes_count_against_the_disk(tmp_path):
    run = tmp_path / ".tmp" / f"org--run@{'3' * 40}-4242"
    run.mkdir(parents=True)
    (run / ".fetch.json").write_text(json.dumps({"owner": "infra", "total_bytes": 6 * GB}))
    with models._lock(tmp_path, f"org--run@{'3' * 40}"):
        with pytest.raises(Refused, match=r"other fetches' included"):
            fetch(tmp_path, disk=disk(80, 20))
    assert fetch(tmp_path, disk=disk(80, 20))[1] == "fetched"


def test_a_fetch_of_a_cached_entry_by_its_owner_only_bumps_it_and_another_owner_is_refused(tmp_path):
    first, _ = fetch(tmp_path)
    hub = Hub()
    again, how = fetch(tmp_path, hub, now=T + 3600, phase="B", expires_h=10)
    assert how == "cached" and hub.gets == [] and again["created"] == first["created"] and again["phase"] == "B"
    assert (again["last_used"], again["expires"], again["expires_h"]) == ("2026-10-01T03:00:00Z", "2026-10-01T13:00:00Z", 10)
    with pytest.raises(Refused, match="cached here for circuits, not infra") as e:
        fetch(tmp_path, owner="infra")
    assert e.value.code == 3
    with pytest.raises(Refused, match="cached with other files"):
        fetch(tmp_path, Hub(files={**FILES, "extra.bin": b"x"}), include=("*",))


def test_a_cached_entry_with_a_file_gone_is_fetched_again(tmp_path):
    fetch(tmp_path)
    (tmp_path / NAME / "model.safetensors").unlink()
    hub = Hub()
    assert fetch(tmp_path, hub, now=T + 60)[1] == "fetched" and ("model.safetensors", 0) in hub.gets


def test_an_interrupted_stream_resumes_with_a_range_and_a_killed_fetchs_download_is_taken_over(tmp_path):
    hub = Hub(cut={"model.safetensors": 1000})
    e, _ = fetch(tmp_path, hub)
    assert [g for g in hub.gets if g[0] == "model.safetensors"] == [("model.safetensors", 0), ("model.safetensors", 1000)]
    assert e["files"]["model.safetensors"]["sha256"] == sha256(WEIGHTS)

    other = tmp_path / "other"
    killed = other / ".tmp" / f"{NAME}-99999"
    (killed / "tok").mkdir(parents=True)
    (killed / "model.safetensors").write_bytes(WEIGHTS[:5000])
    (killed / "tok" / "tokenizer.json").write_bytes(FILES["tok/tokenizer.json"])
    hub = Hub()
    e, _ = fetch(other, hub)
    assert sorted(hub.gets) == [("config.json", 0), ("model.safetensors", 5000)]
    assert e["files"]["model.safetensors"]["sha256"] == sha256(WEIGHTS) and list((other / ".tmp").iterdir()) == []


def test_a_server_that_ignores_the_range_restarts_the_file(tmp_path):
    hub = Hub(cut={"model.safetensors": 1000}, ignore_range=True)
    e, _ = fetch(tmp_path, hub)
    assert e["files"]["model.safetensors"]["sha256"] == sha256(WEIGHTS)


def test_a_gated_repo_without_the_token_is_refused_naming_where_the_token_goes(tmp_path):
    with pytest.raises(Refused) as e:
        fetch(tmp_path, Hub(gated="manual"), token_path="/etc/verity/hf-token")
    assert (e.value.code, str(e.value)) == (3, "gated repo: needs the HF token at /etc/verity/hf-token (not installed)")

    class Denied(Hub):
        def get(self, repo, rev, path, start):
            raise models.HubError(401, "HTTP 401 from huggingface.co")
    with pytest.raises(Refused, match=r"gated repo: needs the HF token at /etc/verity/hf-token \(not readable by this user\)"):
        fetch(tmp_path, Denied(), token_why="not readable by this user")
    assert models.entries(tmp_path) == {}


def test_the_token_goes_to_huggingface_alone_and_never_to_the_cdn(tmp_path):
    fake = "hf_FAKE_not_a_credential"
    (tmp_path / "hf-token").write_text(fake + "\n")
    assert models.token(str(tmp_path / "hf-token")) == (fake, "empty")
    assert models.token(str(tmp_path / "missing")) == (None, "not installed")
    h = models._Redirect(f"Bearer {fake}")
    req = urllib.request.Request("https://huggingface.co/org/tiny/resolve/x/model.safetensors", headers={"Range": "bytes=5-"})
    req.add_unredirected_header("Authorization", f"Bearer {fake}")
    cdn = h.redirect_request(req, None, 302, "Found", {}, "https://cas-bridge.xethub.hf.co/xet-bridge-us/abc?sig=1")
    assert cdn.get_header("Authorization") is None and cdn.get_header("Range") == "bytes=5-"
    same = h.redirect_request(req, None, 307, "Temporary Redirect", {}, "https://huggingface.co/api/resolve-cache/models/x")
    assert same.get_header("Authorization") == f"Bearer {fake}"
    assert fake not in str(models.HubError(401, "HTTP 401 from huggingface.co"))


def test_use_bumps_last_used_and_pushes_expires_by_the_entrys_hours(tmp_path):
    fetch(tmp_path, expires_h=10)
    e = models.use(tmp_path, models.ref(f"{REPO}@{REV}"), T + 5 * 3600)
    assert (e["last_used"], e["expires"]) == ("2026-10-01T07:00:00Z", "2026-10-01T17:00:00Z")
    assert models.read_entry(tmp_path / NAME) == e
    with pytest.raises(Refused, match="isn't cached on this node") as x:
        models.use(tmp_path, models.ref(f"org/other@{REV}"), T)
    assert x.value.code == 3


def test_an_entry_without_its_entry_file_is_invisible(tmp_path):
    d = tmp_path / NAME
    d.mkdir()
    (d / "model.safetensors").write_bytes(WEIGHTS)
    (tmp_path / f"org--wrong@{'4' * 40}").mkdir()
    (tmp_path / f"org--wrong@{'4' * 40}" / ".entry.json").write_text(json.dumps({"schema": models.SCHEMA, "repo": "org/else",
                                                                                 "revision": "4" * 40}))
    assert models.entries(tmp_path) == {}
    with pytest.raises(Refused, match="isn't cached"):
        models.use(tmp_path, models.ref(f"{REPO}@{REV}"), T)
    inc = {Path(x["path"]).name: x for x in models.incomplete(tmp_path)}
    assert set(inc) == {NAME, f"org--wrong@{'4' * 40}"} and inc[NAME]["bytes"] == len(WEIGHTS) and not inc[NAME]["running"]
    assert models.parse_entries('{"schema": "other"}\n{"schema": "verity/model-entry/v1", "owner": "x"} {broken') == \
        [{"schema": "verity/model-entry/v1", "owner": "x"}]
    assert fetch(tmp_path)[1] == "fetched" and leftovers(tmp_path)[:2] == [".tmp", NAME]


def test_release_is_the_owners_alone_and_removes_the_entry_file_before_the_files(tmp_path, monkeypatch):
    fetch(tmp_path)
    r = models.ref(f"{REPO}@{REV}")
    with pytest.raises(Refused, match="is circuits's: only its owner releases it") as e:
        models.release(tmp_path, r, "infra", scan_=Scan)
    assert e.value.code == 3 and models.read_entry(tmp_path / NAME)
    seen = []
    real = models.shutil.rmtree
    monkeypatch.setattr(models.shutil, "rmtree", lambda p, **kw: (seen.append((tmp_path / NAME / ".entry.json").exists()),
                                                                  real(p, **kw)))
    models.release(tmp_path, r, "circuits", scan_=Scan)
    assert seen == [False] and not (tmp_path / NAME).exists()


def test_release_refuses_while_a_process_holds_a_file_of_the_entry(tmp_path):
    fetch(tmp_path)
    st = (tmp_path / NAME / "model.safetensors").stat()
    held = Scan(inodes={(st.st_dev, st.st_ino): {4321}})
    with pytest.raises(Refused, match="in use by pid 4321"):
        models.release(tmp_path, models.ref(f"{REPO}@{REV}"), "circuits", scan_=lambda: held)
    assert models.read_entry(tmp_path / NAME)


def test_evict_deletes_expired_entries_and_stale_downloads_but_nothing_in_use_or_locked(tmp_path):
    gone = entry(tmp_path, "org/old", "1" * 40, owner="circuits", gb=1, expires=T - 1)
    live = entry(tmp_path, "org/new", "2" * 40, owner="circuits", gb=1, expires=T + 1)
    mapped = entry(tmp_path, "org/mapped", "3" * 40, owner="infra", gb=1, expires=T - 1)
    locked = entry(tmp_path, "org/locked", "4" * 40, owner="infra", gb=1, expires=T - 1)
    stale = tmp_path / ".tmp" / f"org--stale@{'5' * 40}-111"
    fresh = tmp_path / ".tmp" / f"org--fresh@{'6' * 40}-222"
    for d, t in ((stale, T - 2 * models.STALE_S), (fresh, T - 60)):
        d.mkdir(parents=True)
        (d / "part").write_bytes(b"x" * 10)
        os.utime(d / "part", (t, t))
        os.utime(d, (t, t))
    st = (mapped / "config.json").stat()
    scan = Scan(inodes={(st.st_dev, st.st_ino): {77}})
    with models._lock(tmp_path, locked.name):
        dry = models.evict(tmp_path, T, lambda: scan, dry_run=True)
        assert all(d.exists() for d in (gone, mapped, locked, stale))
        rep = models.evict(tmp_path, T, lambda: scan)
    assert dry["deleted"] == rep["deleted"]
    assert [Path(x["path"]).name for x in rep["deleted"]] == [gone.name, stale.name]
    assert {Path(x["path"]).name: x["why"] for x in rep["kept"]} == {
        mapped.name: "in use by pid 77", locked.name: "locked: a fetch, use or release of it is running"}
    assert not gone.exists() and not stale.exists() and all(d.exists() for d in (live, mapped, locked, fresh))
    idle = models.evict(tmp_path, T - 10, lambda: pytest.fail("nothing is due, so nothing is scanned"))
    assert (idle["deleted"], idle["hidden"]) == ([], None) and (dry["hidden"], rep["hidden"]) == (0, 0)


def test_evict_refuses_to_delete_when_it_cannot_see_every_process(tmp_path):
    d = entry(tmp_path, "org/old", "1" * 40, owner="circuits", gb=1, expires=T - 1)
    blind = Scan(hidden=[1, 2])
    with pytest.raises(Refused, match="can't see 2 processes' open and mapped files, so deleting nothing") as e:
        models.evict(tmp_path, T, lambda: blind)
    assert e.value.code == 3 and d.exists()
    dry = models.evict(tmp_path, T, lambda: blind, dry_run=True)
    assert dry["hidden"] == 2 and [Path(x["path"]).name for x in dry["deleted"]] == [d.name] and d.exists()


def test_the_scan_finds_open_and_mapped_files_by_inode_and_by_path(tmp_path):
    fetch(tmp_path / "models")
    f = tmp_path / "models" / NAME / "model.safetensors"
    st = f.stat()
    proc = tmp_path / "proc"
    (proc / "10" / "fd").mkdir(parents=True)
    (proc / "10" / "maps").write_text("")
    os.symlink(f, proc / "10" / "fd" / "3")
    os.symlink("socket:[123]", proc / "10" / "fd" / "4")
    (proc / "11" / "fd").mkdir(parents=True)
    (proc / "11" / "maps").write_text(f"7f00-7f10 r--s 00000000 {os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x} "
                                      f"{st.st_ino}                    /models/{NAME}/model.safetensors\n"
                                      "7f20-7f30 r-xp 00000000 00:00 0 \n")
    (proc / "self").mkdir()
    s = models.scan(proc)
    assert s.hidden == [] and s.holders(tmp_path / "models" / NAME) == {10, 11}
    assert s.holders(tmp_path / "models") == {10, 11} and s.holders(tmp_path / "proc") == set()


def test_ls_lists_entries_by_owner_and_incomplete_downloads(tmp_path, capsys):
    entry(tmp_path, "org/a", "1" * 40, owner="circuits", gb=120.3, expires=T + 3600)
    entry(tmp_path, "org/b", "2" * 40, owner="infra", gb=21.7, expires=T + 7200)
    run = tmp_path / ".tmp" / f"org--c@{'3' * 40}-9"
    run.mkdir(parents=True)
    (run / ".fetch.json").write_text(json.dumps({"owner": "infra", "total_bytes": 4 * GB}))
    assert cli.main(["models", "ls", "--root", str(tmp_path), "--now", "2026-10-01T02:00:00Z"]) == 0
    out = capsys.readouterr().out
    assert "2 entries, 142.0 GB (circuits 120.3/500 GB, infra 21.7/200 GB), next expiry 2026-10-01T03:00Z" in out
    assert f"org/a@{'1' * 40}  circuits  A      120.30" in out
    assert f"incomplete: .tmp/org--c@{'3' * 40}-9 (infra, 0.00 of 4.00 GB, untouched since" in out
    assert cli.main(["models", "ls", "--json", "--owner", "infra", "--root", str(tmp_path), "--now", "2026-10-01T02:00:00Z"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["schema"] == "cluster/models/v1" and [e["repo"] for e in doc["entries"]] == ["org/b"]
    assert doc["summary"]["owners"] == {"infra": {"entries": 1, "bytes": int(21.7 * GB), "gb": 21.7, "quota_gb": 200}}


def test_the_cli_uses_releases_and_evicts(tmp_path, capsys):
    d = entry(tmp_path, "org/a", "1" * 40, owner="circuits", gb=1, expires=T + 3600, expires_h=24)
    args = ["--root", str(tmp_path), "--now", "2026-10-01T04:00:00Z"]
    assert cli.main(["models", "use", f"org/a@{'1' * 40}", *args]) == 0
    assert capsys.readouterr().out == f"{d}\n" and models.read_entry(d)["expires"] == "2026-10-02T04:00:00Z"
    assert cli.main(["models", "release", f"org/a@{'1' * 40}", "--owner", "infra", *args]) == 3
    assert "is circuits's" in capsys.readouterr().err
    assert cli.main(["models", "evict", "--dry-run", *args]) == 0
    assert json.loads(capsys.readouterr().out)["deleted"] == []
    assert cli.main(["models", "use", f"org/a@{'1' * 40}", "--root", str(tmp_path), "--now", "2026-10-01"]) == 2


def test_the_status_line_sums_each_owner_against_its_quota_and_names_the_next_expiry(tmp_path):
    a = entry(tmp_path, "org/a", "1" * 40, owner="circuits", gb=100.1, expires=T + 3600)
    entry(tmp_path, "org/b", "2" * 40, owner="circuits", gb=20.2, expires=T - 60)
    entry(tmp_path, "org/c", "3" * 40, owner="infra", gb=21.7, expires=T + 7200)
    text = "".join((d / ".entry.json").read_text() for d in sorted(tmp_path.iterdir()) if not d.name.startswith("."))
    s = models.summary(models.parse_entries(text), CFG, T)
    assert models.line(s) == ("models: 3 entries, 142.0 GB (circuits 120.3/500 GB, infra 21.7/200 GB), "
                              "next expiry 2026-10-01T03:00Z; 1 past expiry, awaiting evict")
    assert s["items"][0] == {"repo": "org/a", "revision": "1" * 40, "owner": "circuits", "phase": "A", "gb": 100.1,
                             "last_used": "2026-10-01T02:00:00Z", "expires": models.read_entry(a)["expires"]}
    assert models.line(models.summary([], CFG, T)) == "models: no entries in /workspace/models"


def test_the_configuration_sets_circuits_quota_and_finds_each_node_by_name_or_host():
    cfg = models.load_config()
    n2 = models.node_config(cfg, hostname=cfg["nodes"]["vy-nebius-2"]["hostname"])
    assert n2["node"] == "vy-nebius-2" and n2["root"] == "/workspace/models" and n2["expires_h"] == 72
    assert models.quota_gb(n2, "circuits") == 500 and models.quota_gb(n2, "proofs") == 200
    assert models.node_config(cfg, node="vy-nebius-1")["node"] == "vy-nebius-1"
    assert models.node_config({"nodes": {"n": {"max_use_pct": 70}}}, node="n")["max_use_pct"] == 70
    assert models.node_config({}, hostname="elsewhere")["node"] is None
