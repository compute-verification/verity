"""The remote-first catalog (README §7.2): write-through `label` / `put` with a visible pending queue, preservation verdicts that do
not depend on which machine uploaded, incremental refresh, an atomic rebuild, and the content parity check against a store built
from the remote alone."""
from __future__ import annotations

import json
import os
import threading
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
    assert main(["label", art, "verified", "accepted", "--by", "red-team-1"]) == 1, "a label left local by a failed upload exits 1"
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


def test_refresh_fetches_the_small_payloads_of_the_named_kinds(env):
    """A render machine reads some payloads (instance-equiv/v1 documents): `refresh(payload_kinds=...)` makes the file payloads
    of those kinds local -- up to the size cap, new and already-catalogued artifacts alike -- and nothing else."""
    pod = LocalStore(env / "pod-store")
    doc = pod.put_artifact("instance-equiv/v1", {"lane": "x"}, file=_file(env / "pod", "e.json", b'{"schema": "instance-equiv/v1"}'))
    big = pod.put_artifact("instance-equiv/v1", {"lane": "y"}, file=_file(env / "pod", "big.json", b"x" * 64))
    other = pod.put_artifact("fixture/v1", {"pod": True}, file=_file(env / "pod", "p.bin", b"p"))
    for a in (doc, big, other):
        pod.push(a)
    here = LocalStore()
    assert here.refresh()["manifests"] == 3 and "payloads" not in here.refresh()
    blob = lambda a: here.object_path(here.get_manifest(a).payload.sha256)  # noqa: E731
    assert not blob(doc).exists()
    rep = here.refresh(payload_kinds=["instance-equiv/v1"], payload_max_bytes=40)
    assert (rep["payloads"], rep["errors"]) == (1, [])
    assert blob(doc).exists() and not blob(big).exists() and not blob(other).exists()
    assert here.index.artifact(doc)["local"]
    assert here.refresh(payload_kinds=["instance-equiv/v1"], payload_max_bytes=40)["payloads"] == 0


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


def test_progress_lines_come_on_schedule(capsys):
    lines: list[str] = []
    ticked = threading.Event()

    def log(line: str) -> None:
        lines.append(line)
        if line.startswith("parity: replicas 3/10 ("):
            ticked.set()

    prog = PA.Progress(log, every=0.05)
    prog.phase("replicas", 10)
    prog.tick(3)
    ticked.wait()                                                  # a periodic line reports the phase's progress
    prog.stop()
    assert any(x.startswith("parity: replicas (10) at ") for x in lines)


def test_a_divergent_local_attempt_is_named_refused_by_preserved_and_reconciled_with_an_audit_log(env, capsys):
    pod = LocalStore(env / "pod-store")
    full = {**_attempt(RUN, {}), "telemetry": {"events": None, "resources": None}}
    full["execution"] = {**full["execution"], "load_end": None, "host": "pod"}
    pod.put_attempt(full)
    pod.push_attempt(RUN)                                              # the machine that ran it publishes first
    here = LocalStore()
    here.put_attempt(_attempt(RUN, {}))                                # a second, leaner publish where the remote was not visible
    rep = PA.run(here.root, env / "fresh-1")
    assert rep["attempts"]["mismatched"] == [RUN] and not rep["ok"]
    assert f"reconcile-attempts {RUN}" in rep["unexplained"][0]
    assert main(["preserved", RUN]) == 1 and "DIFFERS from the remote's copy" in capsys.readouterr().out
    assert main(["reconcile-attempts", "--all"]) == 1
    out = capsys.readouterr()
    assert out.out.startswith(f"{RUN}  local ") and "remote only: execution.host" not in out.out and "execution.load_end" in out.out
    local_before = here.attempt_path(RUN).read_bytes()
    assert main(["reconcile-attempts", RUN, "--apply"]) == 0
    st = LocalStore()
    assert json.loads(st.attempt_path(RUN).read_bytes())["execution"]["host"] == "pod" and st.index.attempt(RUN)["execution"]["host"] == "pod"
    aside = list((st.root / "attempts" / ".superseded").glob(f"{RUN}.*.json"))
    assert len(aside) == 1 and aside[0].read_bytes() == local_before
    log = [json.loads(x) for x in (st.root / "reconcile.log").read_text().splitlines()]
    assert log[0]["run"] == RUN and log[0]["action"] == "superseded-by-remote" and "write-once" in log[0]["reason"]
    assert main(["reconcile-attempts", "--all"]) == 0
    assert PA.run(st.root, env / "fresh-2")["ok"]


def test_replica_rows_are_confirmed_from_the_listing_not_a_head_per_blob(env, monkeypatch):
    here = LocalStore()
    tree = env / "big-tree"
    for i in range(300):
        _file(tree / f"d{i % 7}", f"f{i}.bin", f"blob {i}".encode())
    art = here.put_artifact("fixture/v1", {"tree": True}, tree=tree)
    assert here.push(art)["preserved"]
    heads: list[str] = []
    real_head = FsRemote.head
    monkeypatch.setattr(FsRemote, "head", lambda self, key: heads.append(key) or real_head(self, key))
    rep = PA.run(here.root, env / "fresh-1")
    assert rep["ok"] and rep["replicas"]["confirmed"] == 1 and rep["replicas"]["blobs_etag_verified"] == 300
    assert not [k for k in heads if k.startswith("objects/")], "a HEAD per blob: a 259k-blob tree takes hours that way"

    real_list = FsRemote.list_meta
    monkeypatch.setattr(FsRemote, "list_meta", lambda self, prefix, progress=None, keep=None:
                        {k: v for k, v in real_list(self, prefix, progress, keep).items() if not k.endswith("0")})   # keys the listing missed
    heads.clear()
    rep = PA.run(here.root, env / "fresh-2")
    assert rep["ok"] and heads and all(k.endswith("0") for k in heads if k.startswith("objects/"))


def test_a_replica_head_that_fails_every_try_is_named_by_key(env, monkeypatch):
    here = LocalStore()
    art = here.put_artifact("fixture/v1", {"s": 1}, file=_file(env / "in", "s.bin", b"x"))
    here.push(art)
    monkeypatch.setattr(FsRemote, "list_meta", lambda self, prefix, progress=None, keep=None: {})     # nothing listed: every key goes to HEAD
    real_head = FsRemote.head

    def head(self, key):
        if key.startswith("objects/"):
            raise TimeoutError(f"HEAD {key}: no response within the 30s deadline on all 2 tries")
        return real_head(self, key)
    monkeypatch.setattr(FsRemote, "head", head)
    rep = PA.run(here.root, env / "fresh")
    why = rep["replicas"]["not_confirmed"][art]
    assert not rep["ok"] and why.startswith("objects/sha256/") and "HEAD failed after its retries" in why


def test_labels_pulls_the_targets_assertions_first_and_an_unreachable_remote_answers_from_local(env, monkeypatch, capsys):
    head = "0123456789abcdef" * 2 + "01234567"
    pr12, pr13 = f"pr:12@{head}", f"pr:13@{head}"
    reviewer = LocalStore(env / "reviewer")
    for t in (pr12, pr13):
        reviewer.add_label(t, "grant", "red-team", "red-team-flock-3", ts="2026-09-29T05:30:00Z")
    assert LS.push_targets(reviewer, [pr12, pr13])["failed"] == []
    LocalStore().add_label(pr12, "ready", True, "lane-12", ts="2026-09-29T05:31:00Z")         # asserted here, never pushed
    assert main(["labels", pr12]) == 0
    out = capsys.readouterr()
    assert out.out.splitlines() == ["2026-09-29T05:30:00Z  grant=red-team  by red-team-flock-3", "2026-09-29T05:31:00Z  ready=true  by lane-12"]
    assert out.err == ""
    assert main(["labels", pr12, "--json"]) == 0 and len(json.loads(capsys.readouterr().out)) == 2, "pulled once, nothing doubled"
    assert sorted(lab.key for lab in LocalStore().labels(pr12)) == ["grant", "ready"] and LocalStore().labels(pr13) == []

    real = LocalStore.remote.fget
    monkeypatch.setattr(LocalStore, "remote", property(lambda self: Broken(real(self))))
    assert main(["labels", pr13]) == 0
    err = capsys.readouterr().err
    assert f"labels of {pr13}: pulling from fs:" in err and "failed (OSError: injected: endpoint unreachable); answering from the local labels" in err
    assert f"no labels on {pr13}" in err
    monkeypatch.setattr(LocalStore, "remote", property(real))
    (env / "store.toml").write_text("[remote\n")
    assert main(["labels", pr12]) == 0
    out = capsys.readouterr()
    assert "the remote is unusable here (" in out.err and "ready=true  by lane-12" in out.out
