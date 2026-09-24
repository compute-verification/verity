"""Labels are durable with what they describe (r20: assertions made on pods lived only in the pod's store; `push` / `publish` moved
manifests and blobs and nothing under `labels/`).  `push RUN` / `publish` / `preserved` cover the attempt, its derivation and its
outputs' assertions; a failed label push makes the attempt NOT preserved and the gate names the leg; a wiped index is rebuilt from
the remote listing with provenance intact; conflicting assertions coexist."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from research.store import attempt as A
from research.store import ids
from research.store import labels_sync as LS
from research.store import local as L
from research.store import preserved as P
from research.store.cli import main
from research.store.local import LocalStore
from research.store.remote import FsRemote

RUN = "r20260922-105905-2021"
FIXTURES = Path(__file__).parent / "fixtures"
RUN_ID = "r20260922-120000-abcd"                  # the run_v0.2 fixture's id


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    monkeypatch.setenv("RESEARCH_REPO", str(tmp_path / "repo"))
    monkeypatch.setenv("RESEARCH_RUNS", str(tmp_path / "runs"))
    (tmp_path / "repo").mkdir()
    (tmp_path / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    return tmp_path


class LabelsFail:
    """The configured remote, except that every write under labels/ fails (a scoped credential, a network blip on that key)."""

    def __init__(self, inner):
        self.inner = inner
        self.refused: list[str] = []

    def put(self, key, path):
        if key.startswith(LS.PREFIX):
            self.refused.append(key)
            raise OSError(f"injected: PUT {key} refused")
        return self.inner.put(key, path)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _attempt(run_id=RUN, outputs=None, drv="a" * 64) -> dict:
    return {"schema": "research/attempt/v0.1", "id": run_id, "tool": {"name": "bench_vu", "version": "1"}, "derivation": "drv:" + drv,
            "reuse_key": None, "argv": ["x"], "params": {}, "key_params": [], "inputs": {}, "conditions": {}, "key_conditions": [],
            "source": {"commit": "c" * 40, "dirty": False}, "closure": None, "execution": {"start_utc": "2026-09-22T10:59:05Z", "host": "h", "project": "verity", "campaign": "r21-x"},
            "cost": None, "status": {"state": "done", "rc": 0, "validation": "passed"}, "outputs": outputs or {}, "telemetry": None}


def _art(st: LocalStore, tmp: Path, name: str, data: bytes) -> str:
    p = tmp / name
    p.write_bytes(data)
    return st.put_artifact("proof/v1", {"n": name}, file=p)


def _remote_keys(vault: Path, target: str) -> list[str]:
    d = vault / "labels" / ids.target_dirname(target)
    return sorted(p.name for p in d.glob("*.json")) if d.is_dir() else []


def test_push_run_carries_labels_and_a_wiped_index_is_rebuilt_from_the_remote_with_provenance(env, tmp_path, capsys):
    st = LocalStore()
    art = _art(st, tmp_path, "p.bin", b"proof")
    doc = _attempt(outputs={"result": art})
    st.put_attempt(doc)
    st.add_label(RUN, "class", "GREEN", "fleet-agent", ref="note:r20-int-3", ts="2026-09-23T14:00:00Z")
    st.add_label(RUN, "preserved", True, "fleet-agent", ts="2026-09-23T14:00:01Z")
    st.add_label(doc["derivation"], "decision", "accept", "daniel", ref="F-r19-int-24", ts="2026-09-23T14:00:02Z")
    st.add_label(art, "verified", "accepted", "red-team-1", ts="2026-09-23T14:00:03Z")
    st.add_label("art:" + "9" * 64, "unrelated", "x", "someone")                        # not this attempt's: stays where it is
    assert LS.attempt_targets(doc) == [RUN, doc["derivation"], art]
    assert main(["labels", RUN, "--remote"]) == 1, "before the push: local only -> not durable"
    out, err = capsys.readouterr()
    assert out.count("LOCAL ONLY") == 2 and "NOT durable" in err

    assert main(["push", RUN]) == 0
    out = capsys.readouterr().out
    assert f"{RUN}  attempt  PRESERVED" in out and "labels: 4 pushed, 0 already there" in out
    vault = env / "vault"
    for t, n in ((RUN, 2), (doc["derivation"], 1), (art, 1)):
        keys = _remote_keys(vault, t)
        assert len(keys) == n and all(LS.KEY_RE.match(f"labels/{ids.target_dirname(t)}/{k}") for k in keys), (t, keys)
    assert _remote_keys(vault, "art:" + "9" * 64) == [], "an unrelated target's assertion is not the attempt's business"
    h = LS.label_hash(st.labels(RUN, key="class")[0])
    assert (vault / "labels" / RUN / f"{h}.json").read_bytes() == json.dumps(
        {"by": "fleet-agent", "key": "class", "ref": "note:r20-int-3", "schema": "research/label/v0.1", "target": RUN, "ts": "2026-09-23T14:00:00Z", "value": "GREEN"},
        separators=(",", ":")).encode(), "content key over canonical bytes"
    assert main(["push", RUN]) == 0 and "labels: 0 pushed, 4 already there" in capsys.readouterr().out, "idempotent"
    assert main(["labels", RUN, "--remote", "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["counts"] == {"both": 2, "local": 0, "remote": 0} and {x["label"]["key"] for x in rep["labels"]} == {"class", "preserved"}
    assert main(["preserved", RUN]) == 0
    out = capsys.readouterr().out
    assert "labels                 DURABLE  4 local, 4 on remote" in out and out.strip().splitlines()[-1].startswith("PRESERVED on fs:")

    # the pod dies: a fresh machine rebuilds everything, assertions included, from the remote listing
    st.close()
    shutil.rmtree(env / "store")
    fresh = LocalStore()
    assert fresh.labels(RUN) == [] and fresh.index.labels(RUN) == []
    assert main(["reindex", "--remote"]) == 0
    assert "4 labels (4 pulled from the remote)" in capsys.readouterr().out
    labs = fresh.labels(RUN)
    assert [(l.key, l.value, l.by, l.ts, l.ref) for l in labs] == [("class", "GREEN", "fleet-agent", "2026-09-23T14:00:00Z", "note:r20-int-3"),
                                                                    ("preserved", True, "fleet-agent", "2026-09-23T14:00:01Z", None)]
    assert fresh.labels(doc["derivation"])[0].ref == "F-r19-int-24" and fresh.labels(art)[0].by == "red-team-1"
    assert fresh.index.labels(RUN, key="class")[0]["by"] == "fleet-agent" and fresh.index.select_artifacts(label=[("verified", "accepted")])[0]["id"] == art
    assert fresh.get_attempt(RUN) == doc and fresh.get_manifest(art).kind == "proof/v1"
    assert main(["preserved", RUN]) == 0, "the rebuilt store is preserved too (local == remote)"
    capsys.readouterr()


def test_preserved_refuses_and_names_the_labels_leg_when_a_label_push_fails(env, tmp_path, monkeypatch, capsys):
    vault = env / "vault"
    failing = LabelsFail(FsRemote(vault))
    monkeypatch.setattr(L, "load_remote", lambda path=None: failing)
    st = LocalStore()
    art = _art(st, tmp_path, "p.bin", b"proof")
    doc = _attempt(outputs={"result": art})
    st.put_attempt(doc)
    st.add_label(RUN, "class", "GREEN", "fleet-agent")
    st.add_label(art, "verified", "accepted", "red-team-1")

    rc = main(["push", RUN])
    out = capsys.readouterr().out
    assert rc == 1, out
    assert f"{RUN}  attempt  NOT preserved: labels: 2 assertion(s) not pushed (labels/{RUN}/" in out
    assert "0/2 preserved" in out and "2 NOT pushed" in out
    assert (vault / "attempts" / f"{RUN}.json").exists() and (vault / "manifests").is_dir(), "manifests and payloads did land; the labels leg is what failed"
    assert not (vault / "labels").exists() and len(failing.refused) == 2
    rc = main(["preserved", RUN])
    out = capsys.readouterr().out
    assert rc == 1 and "labels                 NOT durable: 2 local assertion(s) absent on the remote  2 local, 0 on remote" in out
    assert "NOT PRESERVED: 1 item(s) not verifiably on the remote  (legs failing: labels)" in out
    rep = P.check(st, [RUN])
    assert rep["preserved"] is False and rep["targets"][0]["legs"] == {"manifest": True, "artifacts": True, "labels": False}
    assert [x["kind"] for x in rep["not_preserved"]] == ["labels"] and all(x.endswith(".json: absent on the remote") for x in rep["not_preserved"][0]["errors"])
    # the artifact alone: its own assertion is its leg
    assert main(["preserved", art]) == 1 and "legs failing: labels" in capsys.readouterr().out
    # a publish in the same condition says so (never raises into the runner)
    d = tmp_path / "runs" / RUN_ID
    shutil.copytree(FIXTURES / "run_v0.2", d)
    (d / "state.json").rename(d / "status.json")
    st.add_label(RUN_ID, "class", "GREEN", "fleet-agent")                              # asserted before (re-)publication
    pub = A.publish(d, store=st)
    assert pub["published"] and pub["preserved"] is False and pub["preserve_error"].startswith("labels: 1 assertion(s) not pushed")
    assert pub["labels"]["local"] == 1 and pub["labels"]["failed"] and not pub["labels"]["pushed"]
    assert all(pub["preserved_artifacts"].values()), "the artifacts leg passed; the labels leg is the reason"

    # the fault clears: the same commands repair and the gate opens
    monkeypatch.setattr(L, "load_remote", lambda path=None: FsRemote(vault))
    assert main(["push", RUN]) == 0
    assert "labels: 2 pushed, 0 already there" in capsys.readouterr().out
    assert main(["preserved", RUN]) == 0
    out = capsys.readouterr().out
    assert "DURABLE  2 local, 2 on remote" in out and "legs failing" not in out
    st2 = LocalStore()
    assert P.check(st2, [RUN])["targets"][0]["legs"] == {"manifest": True, "artifacts": True, "labels": True}
    assert P.check(st2, [RUN], mode="recorded")["targets"][0]["legs"]["labels"] is None, "offline: not checked, not counted"
    # --no-labels is explicit about leaving the leg out: the push reports no labels line and the gate stays shut for a new assertion
    st2.add_label(RUN, "note", "late", "alice")
    assert main(["push", RUN, "--no-labels"]) == 0 and "labels:" not in capsys.readouterr().out
    assert main(["preserved", RUN]) == 1 and "1 local assertion(s) absent" in capsys.readouterr().out


def test_conflicting_assertions_coexist_on_both_sides_with_latest_and_history_views(env, tmp_path, capsys):
    a, b = LocalStore(tmp_path / "A"), LocalStore(tmp_path / "B")
    for s in (a, b):
        s.put_attempt(_attempt())
    a.add_label(RUN, "class", "GREEN", "alice", ts="2026-09-23T01:00:00Z", ref="note:1")
    b.add_label(RUN, "class", "RED", "bob", ts="2026-09-23T01:00:05Z", ref="note:2")           # another machine disagrees
    b.add_label(RUN, "class", "AMBER", "alice", ts="2026-09-23T01:00:09Z")                     # alice changes her mind on machine B
    assert main(["push", "--store", str(a.root), RUN]) == 0 and main(["push", "--store", str(b.root), RUN]) == 0
    capsys.readouterr()
    assert len(_remote_keys(env / "vault", RUN)) == 3, "three assertions, three immutable keys, nothing overwritten"
    assert main(["labels", "--store", str(a.root), RUN, "--remote"]) == 0
    out, err = capsys.readouterr()
    assert out.count("both ") == 1 and out.count("REMOTE ONLY") == 2 and "1 on both, 0 local only, 2 remote only" in err

    c = LocalStore(tmp_path / "C")
    assert main(["reindex", "--store", str(c.root), "--remote"]) == 0
    capsys.readouterr()
    hist = c.labels(RUN, key="class")
    assert [(l.value, l.by) for l in hist] == [("GREEN", "alice"), ("RED", "bob"), ("AMBER", "alice")], "full history in ts order; the store picks no winner"
    assert main(["labels", "--store", str(c.root), RUN, "--key", "class"]) == 0
    latest = capsys.readouterr().out.strip().splitlines()
    assert len(latest) == 2 and "class=RED  by bob" in latest[0] and "class=AMBER  by alice" in latest[1], "latest per (key, by)"
    assert main(["labels", "--store", str(c.root), RUN, "--key", "class", "--history"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 3
    assert [r["value"] for r in c.index.labels(RUN, key="class")] == ["GREEN", "RED", "AMBER"]
    assert main(["preserved", "--store", str(c.root), RUN]) == 0
    assert "DURABLE  3 local, 3 on remote" in capsys.readouterr().out
    # A still holds only its own: durable (its one is up), and it sees the other two as asserted elsewhere
    rep = P.check(a, [RUN])
    assert rep["preserved"] and rep["targets"][0]["labels"] == {**rep["targets"][0]["labels"], "local": 1, "remote": 3, "missing": 0}


def test_pull_from_a_machine_brings_the_labels_of_the_attempt_its_derivation_and_outputs(env, tmp_path, monkeypatch, capsys):
    from research.store import cli_prov

    machine = LocalStore(tmp_path / "machine" / "store")
    art = _art(machine, tmp_path, "p.bin", b"proof")
    doc = _attempt(outputs={"result": art})
    machine.put_attempt(doc)
    machine.add_label(RUN, "class", "GREEN", "fleet-agent", ref="note:pod", ts="2026-09-23T14:00:00Z")
    machine.add_label(doc["derivation"], "decision", "accept", "daniel", ts="2026-09-23T14:00:01Z")
    machine.add_label(art, "verified", "accepted", "red-team-1", ts="2026-09-23T14:00:02Z")
    machine.add_label("art:" + "9" * 64, "unrelated", "x", "someone")
    log: list = []

    def ssh_run(ssh, remote_argv, *, stdin=None, timeout=120):
        log.append(remote_argv)
        if remote_argv[0] == "cat":
            p = Path(remote_argv[1])
            return subprocess.CompletedProcess(remote_argv, 0 if p.is_file() else 1, p.read_bytes() if p.is_file() else b"", b"" if p.is_file() else b"no such file")
        assert remote_argv[:4] == ["tar", "-cf", "-", "-C"] and remote_argv[-2:] == ["-T", "-"]
        return subprocess.run(remote_argv, input=stdin, capture_output=True)

    monkeypatch.setattr(cli_prov.remote, "ssh_run", ssh_run)
    monkeypatch.setattr(cli_prov.remote, "resolve_machine", lambda name, project, machines=None: {"name": name, "root": str(tmp_path / "machine"), "python": "python3"})
    monkeypatch.setattr(cli_prov.remote, "pod_target", lambda m: {"host": "h", "port": 22})
    monkeypatch.setattr(cli_prov.remote, "ssh_argv", lambda host, port, **k: ["ssh", host])
    local = LocalStore(tmp_path / "local")
    monkeypatch.setattr(cli_prov.A, "open_store", lambda store=None, root=None: local)
    local.add_label(RUN, "class", "GREEN", "fleet-agent", ref="note:pod", ts="2026-09-23T14:00:00Z")   # already here: skipped, not duplicated
    assert cli_prov.main(["pull", "--from", "m", RUN, "--project", "verity"]) == 0
    out = capsys.readouterr().out
    assert "labels: 2 added, 1 already here" in out
    assert [(l.key, l.value, l.ref) for l in local.labels(RUN)] == [("class", "GREEN", "note:pod")]
    assert local.labels(doc["derivation"])[0].by == "daniel" and local.labels(art)[0].value == "accepted"
    assert local.labels("art:" + "9" * 64) == [], "only the attempt's closure travels"
    assert local.index.labels(art)[0]["by"] == "red-team-1"
    assert cli_prov.main(["pull", "--from", "m", RUN, "--project", "verity", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["labels"] == {"seen": 3, "added": 0, "already": 3, "bad": 0}, "idempotent"
    assert sum(1 for a in log if a[0] == "tar") == 3, "first pull: one tar of blobs + one of labels; second: the blob is already here, one tar of labels"


def test_labels_only_push_moves_just_the_targets_closure_and_rechecks_no_blob(env, tmp_path, monkeypatch, capsys):
    """r21 close-out: a pod-side PRESERVED left the laptop's assertions behind, and the only repairs were `data push art:` (every blob
    re-verified, 259k of them) or `labels-sync --push-only` (the whole store).  `push --labels-only` is the scoped one."""
    st = LocalStore()
    art = _art(st, tmp_path, "p.bin", b"proof")
    doc = _attempt(outputs={"result": art})
    st.put_attempt(doc)
    assert main(["push", RUN, "--no-labels"]) == 0
    capsys.readouterr()
    st.add_label(RUN, "class", "GREEN", "laptop")
    st.add_label(doc["derivation"], "decision", "accept", "daniel")
    st.add_label(art, "verified", "accepted", "red-team-1")
    other = _art(st, tmp_path, "q.bin", b"other")
    st.add_label(other, "note", "x", "laptop")
    st.add_label("art:" + "9" * 64, "unrelated", "x", "someone")
    assert P.check(st, [RUN])["targets"][0]["legs"] == {"manifest": True, "artifacts": True, "labels": False}
    vault = env / "vault"
    touched: list[str] = []
    real = st.remote.__class__.head

    def spy_head(self, key):
        touched.append(key)
        return real(self, key)

    monkeypatch.setattr(st.remote.__class__, "head", spy_head)
    assert main(["push", RUN, "--labels-only"]) == 0
    out = capsys.readouterr().out
    assert f"{RUN}  labels  DURABLE: 3 local assertion(s) over 3 target(s), 3 pushed, 0 already there" in out and "1/1 with durable labels" in out
    assert touched and all(k.startswith("labels/") for k in touched), "only assertions are HEADed: no blob, manifest or attempt re-check"
    assert [len(_remote_keys(vault, t)) for t in (RUN, doc["derivation"], art, other, "art:" + "9" * 64)] == [1, 1, 1, 0, 0]
    assert P.check(st, [RUN])["preserved"] is True, "the gate now passes: labels durable"
    # an artifact's own assertions only; idempotent; JSON
    assert main(["push", other, "--labels-only", "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["durable"] and rep["targets"][other] == {"label_targets": [other], "local": 1, "pushed": 1, "already": 0, "failed": [], "missing": [],
                                                        "errors": [], "durable": True}
    assert main(["push", RUN, other, "--labels-only"]) == 0 and "0 pushed, 3 already there" in capsys.readouterr().out
    # misuse is refused by name; an unknown attempt is named
    assert main(["push", "--pending", "--labels-only"]) == 1 and "--labels-only takes run / art: ids" in capsys.readouterr().err
    assert main(["push", "r20260101-000000-dead", "--labels-only"]) == 1 and "not in the store" in capsys.readouterr().err


def test_labels_only_push_that_fails_is_not_durable(env, tmp_path, monkeypatch, capsys):
    failing = LabelsFail(FsRemote(env / "vault"))
    monkeypatch.setattr(L, "load_remote", lambda path=None: failing)
    st = LocalStore()
    art = _art(st, tmp_path, "p.bin", b"proof")
    st.add_label(art, "verified", "accepted", "red-team-1")
    rep = LS.push_closure(st, [art])
    assert rep["durable"] is False and rep["targets"][art]["failed"] and rep["targets"][art]["missing"]
    assert main(["push", art, "--labels-only"]) == 1 and "NOT durable" in capsys.readouterr().out
