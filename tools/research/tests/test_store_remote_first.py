"""The remote-first catalog (README §7.2): write-through `label` / `put` with a visible pending queue, preservation verdicts that do
not depend on which machine uploaded, incremental refresh, an atomic rebuild, and the content parity check against a store built
from the remote alone."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from research import notes
from research.store import labels_sync as LS
from research.store import parity as PA
from research.store import preserved as P
from research.store.cli import main
from research.store.index import Index
from research.store.local import LocalStore, write_through_wanted
from research.store.remote import FsRemote

RUN = "r20260922-105905-2021"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    monkeypatch.setenv("RESEARCH_REPO", str(tmp_path / "repo"))
    monkeypatch.delenv("RESEARCH_WRITE_THROUGH", raising=False)
    (tmp_path / "repo").mkdir()
    (tmp_path / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\nwrite_through = true\n')
    return tmp_path


def _file(d: Path, name: str, data: bytes) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(data)
    return d / name


def _attempt(run_id: str, outputs: dict) -> dict:
    return {"schema": "research/attempt/v0.1", "id": run_id, "tool": {"name": "bench_vu", "version": "1"}, "derivation": "drv:" + "a" * 64,
            "reuse_key": None, "argv": ["x"], "params": {}, "key_params": [], "inputs": {}, "conditions": {}, "key_conditions": [],
            "source": {"commit": "c" * 40, "dirty": False}, "closure": None,
            "execution": {"start_utc": "2026-09-22T10:59:05Z", "host": "h", "project": "verity", "campaign": "x"},
            "cost": None, "status": {"state": "done", "rc": 0, "validation": "passed"}, "outputs": outputs, "telemetry": None}


class Broken:
    """The configured remote with its endpoint down: every call raises."""

    def __init__(self, inner):
        self.inner = inner

    def describe(self):
        return self.inner.describe()

    def __getattr__(self, name):
        def fail(*a, **k):
            raise OSError("injected: endpoint unreachable")
        return fail


def test_write_through_is_off_on_the_laptop_until_asked(tmp_path, monkeypatch):
    cfg = tmp_path / "store.toml"
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    monkeypatch.delenv("RESEARCH_WRITE_THROUGH", raising=False)
    assert write_through_wanted() is False
    cfg.write_text('[remote]\ntype = "fs"\npath = "/x"\n')
    assert write_through_wanted() is False                          # the laptop's store.toml today
    cfg.write_text('[remote]\ntype = "fs"\npath = "/x"\nwrite_through = true\n')
    assert write_through_wanted() is True
    monkeypatch.setenv("RESEARCH_WRITE_THROUGH", "0")
    assert write_through_wanted() is False
    monkeypatch.delenv("RESEARCH_STORE_CONFIG")
    monkeypatch.delenv("RESEARCH_WRITE_THROUGH")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setenv("R2_ENDPOINT", "https://acct.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_BUCKET", "b")
    assert write_through_wanted() is True                           # a cloud machine: the environment names the remote


def test_d1_a_label_is_on_the_remote_when_the_command_returns(env, capsys):
    st = LocalStore()
    art = st.put_artifact("fixture/v1", {"n": 1}, file=_file(env / "in", "a.bin", b"a"))
    assert main(["label", art, "verified", "accepted", "--by", "red-team-1"]) == 0
    assert "research data label: on fs:" in capsys.readouterr().err
    assert main(["labels", art, "--remote"]) == 0 and "both" in capsys.readouterr().out
    os.environ["RESEARCH_WRITE_THROUGH"] = "0"
    try:
        assert main(["label", art, "note", "later", "--by", "x", "--off-vocab"]) == 0
        assert "local only (write-through off); pending: `research data push --pending`" in capsys.readouterr().err
    finally:
        del os.environ["RESEARCH_WRITE_THROUGH"]


def test_d2_a_broken_endpoint_keeps_writes_local_pending_and_push_pending_drains_them(env, monkeypatch, capsys):
    real = LocalStore.remote.fget
    monkeypatch.setattr(LocalStore, "remote", property(lambda self: Broken(real(self))))
    assert main(["put", "--kind", "fixture/v1", "--file", str(_file(env / "in", "p.bin", b"payload"))]) == 0
    out = capsys.readouterr()
    art = out.out.strip()
    assert "research data put: NOT preserved (" in out.err and "pending" in out.err
    assert main(["label", art, "verified", "accepted", "--by", "red-team-1"]) == 0
    assert "research data label: NOT preserved (" in capsys.readouterr().err
    monkeypatch.setattr(LocalStore, "remote", property(real))
    st = LocalStore()
    q = st.pending()
    assert q["artifacts"] == [art] and [x["target"] for x in q["labels"]] == [art]
    assert main(["pending"]) == 1
    assert main(["push", "--pending"]) == 0
    capsys.readouterr()
    assert main(["pending"]) == 0 and LocalStore().pending() == {**q, "artifacts": [], "labels": []}
    assert main(["preserved", art]) == 0


def test_d3_a_pod_upload_is_preserved_everywhere_and_short_ids_resolve_on_the_remote(env, capsys):
    pod = LocalStore(env / "pod-store")
    arts = [pod.put_artifact("fixture/v1", {"i": i}, file=_file(env / "pod", f"{i}.bin", bytes([i]) * 100)) for i in range(4)]
    for a in arts:
        assert pod.push(a)["preserved"]
    here = LocalStore()
    mine = [here.put_artifact("fixture/v1", {"j": j}, file=_file(env / "mine", f"{j}.bin", bytes([j + 9]) * 50)) for j in range(4)]
    for a in mine:
        assert here.push(a)["preserved"]
    short = arts[0][:12]
    assert here.index.resolve_art_prefix(short) == [] and here.resolve_art(short) == [arts[0]]
    assert main(["preserved", short]) == 0
    assert main(["preserved", arts[1], "--mode", "recorded"]) == 0
    assert "PRESERVED" in capsys.readouterr().out
    assert here.index.replicas(arts[1])["remote"]["verified_utc"], "the check that verified it recorded it"
    assert not P.check(here, [arts[2]], mode="recorded", head_fallback=False)["preserved"], "offline recorded stays offline"
    ok, why = notes._custody([a[4:14] for a in arts + mine])      # the verifier-cost FINAL replay: 8/8, not 4/8
    assert (ok, why) == (True, "8 cited artifacts preserved (recorded remote replicas, else verified on the remote now)")


def test_d5_a_rebuild_killed_partway_leaves_the_previous_catalog_answering(env, monkeypatch):
    st = LocalStore()
    art = st.put_artifact("fixture/v1", {"n": 1}, file=_file(env / "in", "a.bin", b"a"))
    st.push(art)
    before = st.index.artifact(art)

    def dies(self, arts, attempts, labels):
        self.index_artifact(*arts[0][:2])
        raise KeyboardInterrupt("killed")

    with monkeypatch.context() as m, pytest.raises(KeyboardInterrupt):
        m.setattr(Index, "rebuild", dies)
        LocalStore().reindex()
    fresh = LocalStore(os.environ["RESEARCH_STORE"])
    assert fresh.index.artifact(art) == before and fresh.index.replicas(art)["remote"]["verified_utc"]
    assert not list(Path(os.environ["RESEARCH_STORE"]).glob(".catalog.sqlite.rebuild-*"))
    assert LocalStore().reindex()["artifacts"] == 1
    assert st.index.replicas(art)["remote"]["verified_utc"], "replicas survive the swap; an open store sees the new file"


def test_d6_refresh_picks_up_pod_writes_without_wiping(env):
    here = LocalStore()
    local_only = here.put_artifact("fixture/v1", {"local": True}, file=_file(env / "in", "l.bin", b"l"))
    pod = LocalStore(env / "pod-store")
    art = pod.put_artifact("fixture/v1", {"pod": True}, file=_file(env / "pod", "p.bin", b"p"))
    pod.put_attempt(_attempt(RUN, {"result": art}))
    pod.add_label(art, "verified", "accepted", "red-team-1")
    pod.push(art)
    pod.push_attempt(RUN)
    pod.push_labels([art])
    rep = here.refresh()
    assert (rep["manifests"], rep["attempts"], rep["labels"], rep["errors"]) == (1, 1, 1, [])
    assert here.index.artifact(art)["remote"] and here.index.attempt(RUN)["id"] == RUN and here.index.labels(art)[0]["by"] == "red-team-1"
    assert here.index.artifact(local_only) is not None
    assert here.refresh()["manifests"] == 0


def test_parity_explains_every_difference_and_fails_on_content(env):
    here = LocalStore()
    shared = here.put_artifact("fixture/v1", {"s": 1}, file=_file(env / "in", "s.bin", b"shared"))
    here.push(shared)
    here.put_attempt(_attempt(RUN, {"result": shared}))
    here.push_attempt(RUN)
    here.add_label(shared, "verified", "accepted", "red-team-1")
    here.push_labels([shared])
    pending = here.put_artifact("fixture/v1", {"s": 2}, file=_file(env / "in", "p.bin", b"pending"))
    pod = LocalStore(env / "pod-store")
    pulled = pod.put_artifact("fixture/v1", {"s": 3}, file=_file(env / "pod", "q.bin", b"pod"))
    pod.push(pulled)

    rep = PA.run(here.root, env / "fresh-1")
    assert rep["ok"], rep["unexplained"]
    assert rep["artifacts"]["only_here"] == {pending[4:]: "pending here"} and rep["artifacts"]["not_pulled"] == [pulled[4:]]
    assert rep["attempts"]["both"] == 1 and rep["labels"]["both"] == 1 and rep["replicas"] == {"rows": 1, "confirmed": 1, "not_confirmed": {},
                                                                                  "blobs_etag_verified": 1, "blobs_remote_only_size_verified": 0}
    lines = PA.format_report(rep)
    assert lines[-1] == "PARITY OK: every difference is explained" and "1 pending here" in lines[1]

    vault = FsRemote(env / "vault")
    blob = next(k for k in vault.list("objects/") if k.endswith(json.loads(here.manifest_path(shared[4:]).read_text())["payload"]["sha256"]))
    (env / "vault" / blob).chmod(0o644)
    (env / "vault" / blob).write_bytes(b"truncated")
    rep = PA.run(here.root, env / "fresh-2")
    assert not rep["ok"] and any(u.startswith(f"replica {shared}: objects/") and "remote size" in u for u in rep["unexplained"])
    with pytest.raises(ValueError, match="separate directory"):
        PA.run(here.root, here.root / "sub")


def test_parity_cli_writes_the_report(env, capsys):
    here = LocalStore()
    here.push(here.put_artifact("fixture/v1", {"s": 1}, file=_file(env / "in", "s.bin", b"x")))
    assert main(["parity", "--fresh", str(env / "fresh"), "--json-out", str(env / "parity.json")]) == 0
    assert capsys.readouterr().out.strip().endswith("PARITY OK: every difference is explained")
    assert json.loads((env / "parity.json").read_text())["ok"] is True
    assert LS.local_labels(here) == {}


def test_parity_never_downloads_an_evicted_blob_and_catches_a_same_size_corruption_by_etag(env, monkeypatch):
    here = LocalStore()
    art = here.put_artifact("fixture/v1", {"s": 1}, file=_file(env / "in", "s.bin", b"payload-bytes"))
    here.push(art)
    blob = json.loads(here.manifest_path(art[4:]).read_text())["payload"]["sha256"]
    here.object_path(blob).chmod(0o644)
    here.object_path(blob).unlink()                                   # evicted on the laptop
    gets: list[str] = []
    real_get = FsRemote.get
    monkeypatch.setattr(FsRemote, "get", lambda self, key, path: gets.append(key) or real_get(self, key, path))
    rep = PA.run(here.root, env / "fresh-1")
    assert rep["ok"] and rep["replicas"]["blobs_remote_only_size_verified"] == 1 and rep["replicas"]["blobs_etag_verified"] == 0
    assert not [k for k in gets if k.startswith("objects/")], "the comparison fetched blob bytes"
    assert "1 remote-only, size-verified; nothing downloaded" in "\n".join(PA.format_report(rep))

    other = here.put_artifact("fixture/v1", {"s": 2}, file=_file(env / "in", "t.bin", b"AAAA"))
    here.push(other)
    ob = env / "vault" / "objects" / "sha256" / json.loads(here.manifest_path(other[4:]).read_text())["payload"]["sha256"]
    ob.chmod(0o644)
    ob.write_bytes(b"BBBB")                                           # same size, different bytes
    rep = PA.run(here.root, env / "fresh-2")
    assert not rep["ok"] and any("ETag" in u and "local blob's MD5" in u for u in rep["unexplained"])


def test_the_render_goes_through_a_view_that_writes_nothing_to_the_store_under_test(env, monkeypatch):
    here = LocalStore()
    art = here.put_artifact("fixture/v1", {"s": 1}, file=_file(env / "in", "s.bin", b"rendered"))
    here.push(art)
    blob = json.loads(here.manifest_path(art[4:]).read_text())["payload"]["sha256"]
    here.object_path(blob).chmod(0o644)
    here.object_path(blob).unlink()
    before = sorted((p.relative_to(here.root).as_posix(), p.stat().st_mtime_ns) for p in here.root.rglob("*") if p.is_file())
    gets: list[str] = []
    real_get = FsRemote.get
    monkeypatch.setattr(FsRemote, "get", lambda self, key, path: gets.append(key) or real_get(self, key, path))

    def fake_render(root, timeout):                                   # what bench.tables does: catalog query + payload reads
        st = LocalStore(root)
        return "".join(st.fetch(a["id"]).read_text() for a in st.index.artifacts()), None

    monkeypatch.setattr(PA, "_render", fake_render)
    rep = PA.run(here.root, env / "fresh", render=True)
    assert rep["ok"] and rep["render"]["identical"], rep["unexplained"]
    assert gets.count(f"objects/sha256/{blob}") == 1, "each blob the render reads is downloaded once, for both renders"
    after = sorted((p.relative_to(here.root).as_posix(), p.stat().st_mtime_ns) for p in here.root.rglob("*") if p.is_file())
    assert after == before, "the store under test was written"


def test_progress_lines_come_on_schedule(capsys):
    lines: list[str] = []
    prog = PA.Progress(lines.append, every=0.05)
    prog.phase("replicas", 10)
    prog.tick(3)
    import time as _t
    _t.sleep(0.2)
    prog.stop()
    assert lines[0].startswith("parity: replicas (10) at ") and any(x.startswith("parity: replicas 3/10 (") for x in lines[1:])


def test_the_render_sees_remote_only_records_on_both_sides(env, monkeypatch):
    here = LocalStore()
    here.push(here.put_artifact("fixture/v1", {"s": 1}, file=_file(env / "in", "s.bin", b"mine")))
    pod = LocalStore(env / "pod-store")
    art = pod.put_artifact("fixture/v1", {"s": 2}, file=_file(env / "pod", "p.bin", b"pods"))
    pod.add_label(art, "verified", "accepted", "red-team-1")
    pod.push(art)
    pod.push_labels([art])

    def fake_render(root, timeout):                                   # catalog rows and labels: what a table is built from
        st = LocalStore(root)
        return "\n".join(f"{a['id']} {[(l['key'], l['value']) for l in st.index.labels(a['id'])]}" for a in st.index.artifacts()), None

    monkeypatch.setattr(PA, "_render", fake_render)
    rep = PA.run(here.root, env / "fresh", render=True)
    assert rep["artifacts"]["not_pulled"] == [art[4:]] and rep["labels"]["not_pulled"]
    assert rep["ok"] and rep["render"]["identical"] and rep["render_view_remote_records"] == 2, rep["unexplained"]
