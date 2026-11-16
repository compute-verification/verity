"""The labels pack: every assertion of the remote's `labels/` as one tarball at a fixed key, which a sync that lacks most of them
takes in one request before it pulls the rest.  The pack is a cache of `labels/`, never its source: a member lands only under a
name the remote lists and with content that hashes to it, and a pack that is absent, corrupt or unreadable costs nothing but the
requests it would have saved.  Temp stores and an `FsRemote` directory only; no network."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from research.store import canon
from research.store import cli
from research.store import labels_sync as LS
from research.store import local as L
from research.store.cli import main
from research.store.local import LocalStore
from research.store.model import Label
from research.store.remote import FsRemote

ART = "art:" + "3" * 64
PACK_KEYS = [LS.PACK_INFO_KEY, LS.PACK_KEY]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    monkeypatch.setenv("RESEARCH_REPO", str(tmp_path / "repo"))
    monkeypatch.setenv("RESEARCH_RUNS", str(tmp_path / "runs"))
    (tmp_path / "repo").mkdir()
    monkeypatch.setattr(LS, "PACK_MIN_MISSING", 4)
    return tmp_path


class Seen:
    """The configured remote, recording the key of every GET and every replaced key."""

    def __init__(self, inner):
        self.inner = inner
        self.gets: list[str] = []
        self.replaced: list[str] = []

    def get(self, key, path):
        self.gets.append(key)
        return self.inner.get(key, path)

    def replace(self, key, path):
        self.replaced.append(key)
        return self.inner.replace(key, path)

    def __getattr__(self, name):
        return getattr(self.inner, name)


@pytest.fixture
def remote(env, monkeypatch) -> Seen:
    seen = Seen(FsRemote(env / "vault"))
    monkeypatch.setattr(L, "load_remote", lambda path=None: seen)
    return seen


def _assert(st: LocalStore, n: int, first: int = 0) -> None:
    for i in range(first, first + n):
        st.add_label(ART if i % 2 else f"r20260922-105905-{i:04d}", "note", f"n{i}", "alice", ts="2026-09-23T01:00:00Z")


def _hashes(st: LocalStore) -> set[str]:
    return set(LS.local_labels(st))


def _label_gets(remote: Seen) -> list[str]:
    return [k for k in remote.gets if k.startswith(LS.PREFIX)]


def _put_pack(vault: Path, members: list[tuple[str, bytes]], sha256: str | None = None) -> None:
    """A pack written by hand: `members` as (name, bytes), and an info naming `sha256` (default: the tarball's own)."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tf:
        for name, data in members:
            mem = tarfile.TarInfo(name)
            mem.size = len(data)
            tf.addfile(mem, io.BytesIO(data))
    (vault / LS.PACK_KEY).parent.mkdir(parents=True, exist_ok=True)
    (vault / LS.PACK_KEY).write_bytes(buf.getvalue())
    (vault / LS.PACK_INFO_KEY).write_bytes(canon.dumps({"schema": LS.PACK_SCHEMA, "sha256": sha256 or hashlib.sha256(buf.getvalue()).hexdigest()}))


def test_a_fresh_store_takes_the_pack_in_one_fetch_and_pulls_only_the_difference(env, remote, tmp_path, capsys):
    a = LocalStore(tmp_path / "A")
    _assert(a, 12)
    LS.sync(a)
    info = LS.publish_pack(LocalStore(tmp_path / "P"))
    assert info["labels"] == 12 and info["pulled"] == 12 and info["bad"] == 0 and info["schema"] == LS.PACK_SCHEMA
    assert remote.replaced == [LS.PACK_KEY, LS.PACK_INFO_KEY], "the tarball first: the info never names one that isn't there yet"
    assert info["sha256"] == hashlib.sha256((env / "vault" / LS.PACK_KEY).read_bytes()).hexdigest()
    assert json.loads((env / "vault" / LS.PACK_INFO_KEY).read_bytes()) == {k: info[k] for k in ("schema", "sha256", "bytes", "labels", "built_utc")}
    _assert(a, 3, first=12)                                               # asserted after the pack was built: on the remote, not in the pack
    LS.sync(a)

    b = LocalStore(tmp_path / "B")
    remote.gets.clear()
    lines: list[str] = []
    rep = LS.sync(b, push=False, log=lines.append)
    assert len(rep["pulled"]) == 15 and rep["pack"] == 12 and rep["bad"] == [] and lines == []
    assert remote.gets[:2] == PACK_KEYS and len(remote.gets) == 5, "one fetch of the pack, then one GET for each assertion it lacks"
    assert sorted(_label_gets(remote)) == sorted(LS.label_key(lab, h) for h, (_, lab) in LS.local_labels(a).items() if h not in rep["pulled"][:12])
    assert _hashes(b) == _hashes(a) and len(_hashes(b)) == 15
    assert len(b.index.labels(ART)) == 7 and [l.value for l in b.labels("r20260922-105905-0000")] == ["n0"], "pack members are indexed like pulled ones"
    again = LS.sync(b, push=False)
    assert again["pulled"] == [] and again["pack"] == 0 and again["already"] == 15

    remote.gets.clear()
    assert main(["labels-sync", "--store", str(tmp_path / "C"), "--pull-only"]) == 0
    assert "15 pulled (12 from the labels pack), 0 already on both" in capsys.readouterr().out
    assert _hashes(LocalStore(tmp_path / "C")) == _hashes(a)


def test_a_store_that_lacks_few_assertions_never_asks_for_the_pack(env, remote, tmp_path):
    a = LocalStore(tmp_path / "A")
    _assert(a, 4)
    LS.sync(a)
    LS.publish_pack(LocalStore(tmp_path / "P"))
    remote.gets.clear()
    rep = LS.sync(LocalStore(tmp_path / "B"), push=False)
    assert len(rep["pulled"]) == 4 and rep["pack"] == 0, "4 missing is not more than the threshold of 4"
    assert len(remote.gets) == 4 and remote.gets == _label_gets(remote)


def test_with_no_pack_a_sync_pulls_each_assertion_as_before(env, remote, tmp_path, capsys):
    a = LocalStore(tmp_path / "A")
    _assert(a, 9)
    LS.sync(a)
    remote.gets.clear()
    lines: list[str] = []
    b = LocalStore(tmp_path / "B")
    rep = LS.sync(b, push=False, log=lines.append)
    assert rep == {"remote": remote.describe(), "local": 0, "remote_before": 9, "pushed": [], "pulled": sorted(_hashes(a)), "pack": 0,
                   "already": 0, "bad": []}
    assert lines == [] and remote.gets == _label_gets(remote) and len(remote.gets) == 9, "the same GETs as with no pack to ask for"
    assert _hashes(b) == _hashes(a)
    assert not (env / "vault" / "packs").exists() and not list((b.root / "scratch").iterdir())
    assert main(["labels-sync", "--store", str(tmp_path / "C"), "--pull-only"]) == 0
    assert "9 pulled, 0 already on both" in capsys.readouterr().out


@pytest.mark.parametrize("damage", ["bytes", "sha256", "truncated", "not a tarball", "info"])
def test_a_pack_that_fails_its_sha256_or_cannot_be_read_is_ignored_with_one_line(env, remote, tmp_path, damage):
    a = LocalStore(tmp_path / "A")
    _assert(a, 9)
    LS.sync(a)
    LS.publish_pack(LocalStore(tmp_path / "P"))
    tar, info = env / "vault" / LS.PACK_KEY, env / "vault" / LS.PACK_INFO_KEY
    doc = json.loads(info.read_bytes())
    if damage == "bytes":
        data = bytearray(tar.read_bytes())
        data[len(data) // 2] ^= 0xFF
        tar.write_bytes(bytes(data))
    elif damage == "sha256":
        info.write_bytes(canon.dumps(doc | {"sha256": "0" * 64}))
    elif damage == "truncated":                                           # cut short, under an info that names the cut file
        tar.write_bytes(tar.read_bytes()[: tar.stat().st_size // 2])
        info.write_bytes(canon.dumps(doc | {"sha256": hashlib.sha256(tar.read_bytes()).hexdigest()}))
    elif damage == "not a tarball":
        tar.write_bytes(b"not a tarball")
        info.write_bytes(canon.dumps(doc | {"sha256": hashlib.sha256(b"not a tarball").hexdigest()}))
    else:
        info.write_bytes(b"{not json")
    remote.gets.clear()
    lines: list[str] = []
    b = LocalStore(tmp_path / "B")
    rep = LS.sync(b, push=False, log=lines.append)
    assert len(lines) == 1 and "ignoring the labels pack" in lines[0], lines
    if damage in ("bytes", "sha256"):
        assert "sha256" in lines[0] and rep["pack"] == 0
    assert len(rep["pulled"]) == 9 and rep["bad"] == [] and _hashes(b) == _hashes(a), "every assertion still arrives, from labels/"
    assert len(_label_gets(remote)) == 9 - rep["pack"]
    assert not list((b.root / "scratch").iterdir()), "the downloaded pack is not left behind"


def test_a_pack_member_lands_only_under_a_name_the_remote_lists_and_its_content_hashes_to(env, remote, tmp_path):
    a = LocalStore(tmp_path / "A")
    _assert(a, 6)
    LS.sync(a)
    real = {h: lab for h, (_, lab) in LS.local_labels(a).items()}
    forged_h, *honest = sorted(real)
    forged = Label.from_doc(real[forged_h].to_doc() | {"value": "forged"})
    unlisted = Label(target=ART, key="grade", value="A", by="mallory", ts="2026-09-23T02:00:00Z", ref=None).validate()
    members = [(LS.label_key(real[h], h), canon.dumps(real[h].to_doc())) for h in honest]
    members += [(LS.label_key(real[forged_h], forged_h), canon.dumps(forged.to_doc())),     # a listed name over another assertion's content
                (LS.label_key(unlisted), canon.dumps(unlisted.to_doc())),                   # a well-named assertion labels/ doesn't hold
                (LS.label_key(real[honest[0]], honest[0]), canon.dumps(real[honest[0]].to_doc())),          # one assertion twice
                (LS.label_key(real[honest[1]], honest[1]).replace("labels/", "labels/../labels/"), b"{}"),
                ("README", b"not an assertion")]
    _put_pack(env / "vault", members)

    remote.gets.clear()
    lines: list[str] = []
    b = LocalStore(tmp_path / "B")
    rep = LS.sync(b, push=False, log=lines.append)
    assert rep["pack"] == 5 and sorted(rep["pulled"][:5]) == honest and rep["pulled"][5:] == [forged_h] and rep["bad"] == []
    assert _label_gets(remote) == [LS.label_key(real[forged_h], forged_h)], "what the pack didn't give honestly comes from labels/"
    assert len(lines) == 1 and "1 member(s)" in lines[0] and "not their name" in lines[0]
    assert _hashes(b) == set(real), "neither the forged content nor the unlisted assertion landed"
    assert "forged" not in {l.value for t in (ART, *{lab.target for lab in real.values()}) for l in b.labels(t)}
    assert [l.by for l in b.labels(ART, key="grade")] == []
    assert sorted(p.name for p in (b.root / "labels").rglob("*.json")) == sorted(p.name for p in (a.root / "labels").rglob("*.json"))


def test_publishing_packs_what_the_remote_lists_and_writes_nothing_under_labels(env, remote, tmp_path, capsys, monkeypatch):
    a, p = LocalStore(tmp_path / "A"), LocalStore(tmp_path / "P")
    _assert(a, 5)
    LS.sync(a)
    p.add_label(ART, "note", "only here", "pat", ts="2026-09-23T03:00:00Z")           # the publisher's own, never pushed
    before = sorted(k for k in remote.list(LS.PREFIX))
    first = LS.publish_pack(p)
    assert first["labels"] == 5 and sorted(remote.list(LS.PREFIX)) == before, "labels/ is read, never written"
    with tarfile.open(env / "vault" / LS.PACK_KEY, "r:gz") as tf:
        names = tf.getnames()
        assert names == before and tf.extractfile(names[0]).read() == (env / "vault" / names[0]).read_bytes()
    assert LS.publish_pack(p)["sha256"] == first["sha256"], "the same assertions give the same bytes"

    _assert(a, 2, first=5)
    LS.sync(a)
    assert main(["labels-pack", "--store", str(tmp_path / "P")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("labels-pack: 7 labels in ") and f"at {LS.PACK_KEY}" in out and "2 pulled first" in out
    assert json.loads((env / "vault" / LS.PACK_INFO_KEY).read_bytes())["labels"] == 7, "the fixed key is replaced, not added to"
    assert sorted(p.name for p in (env / "vault" / "packs" / "labels").iterdir()) == ["latest.json", "latest.tar.gz"]

    naps: list[float] = []

    class Enough(Exception):
        pass

    def nap(s: float) -> None:
        naps.append(s)
        if len(naps) == 2:
            raise Enough

    monkeypatch.setattr(cli.time, "sleep", nap)
    with pytest.raises(Enough):
        main(["labels-pack", "--store", str(tmp_path / "P"), "--every", "3600", "--json"])
    passes = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert naps == [3600.0, 3600.0] and [r["labels"] for r in passes] == [7, 7], "one pass, then the wait, then the next"

    def refuse(key, path):
        raise OSError(f"injected: PUT {key} refused")

    monkeypatch.setattr(remote, "replace", refuse)
    assert main(["labels-pack", "--store", str(tmp_path / "P"), "--every", "3600"]) == 1, "a pass that fails ends the loop"
    assert "refused" in capsys.readouterr().err and len(naps) == 2
    assert not list((p.root / "scratch").iterdir())


def test_a_remote_that_cannot_replace_a_key_refuses_to_publish(env, tmp_path, monkeypatch):
    class Immutable:
        def __init__(self, inner):
            self.list, self.get, self.describe = inner.list, inner.get, inner.describe

    monkeypatch.setattr(L, "load_remote", lambda path=None: Immutable(FsRemote(env / "vault")))
    with pytest.raises(NotImplementedError, match="cannot replace a key"):
        LS.publish_pack(LocalStore(tmp_path / "P"))
