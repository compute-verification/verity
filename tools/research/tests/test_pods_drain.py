"""`research pods unpreserved|drain <machine>`: pod teardown gated on the store's `preserved` predicate (pods/drain.py), and the
machine -> Attempt link (`exec.machine` / `execution.machine`, env.KEYS + store.attempt)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research import env as E
from research.pods import drain as D
from research.pods import part
from research.store import attempt as A
from research.store import preserved as P
from research.store.local import LocalStore

from test_store import _attempt, _tree


@pytest.fixture
def machines(tmp_path, monkeypatch):
    p = tmp_path / "machines.toml"
    p.write_text('[machines.vy-tp2]\nprovider = "runpod"\npod_id = "pod-tp2"\nproject = "verity"\nroot = "/workspace/research"\n'
                 '[machines.vy-other]\nprovider = "runpod"\npod_id = "pod-other"\nproject = "verity"\n')
    monkeypatch.setenv("RESEARCH_MACHINES", str(p))
    monkeypatch.setenv("RESEARCH_DRAIN_LOG", str(tmp_path / "drain.log"))
    return p


@pytest.fixture
def fs_store(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    (tmp_path / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    return LocalStore()


def _on(st: LocalStore, run_id: str, art: str | None, **exe) -> str:
    ex = {"start_utc": "2026-09-23T10:00:00Z", "end_utc": "2026-09-23T10:00:01Z", "wall_s": 1.0, "host": "c0ffee", "project": "verity", **exe}
    doc = _attempt(run_id=run_id, outputs={"result": art} if art else {}, execution=ex)
    doc["conditions"] = {**doc["conditions"], "exec.machine": exe.get("machine")}
    st.put_attempt(doc)
    return run_id


def test_exec_machine_is_a_condition_filled_from_the_launch_record_and_indexed(tmp_path):
    assert "exec.machine" in E.KEYS
    job = {"run": {"id": "r20260923-100000-0001", "remote": {"machine": "vy-tp2", "pod_id": "pod-tp2"}}, "ownership": {"project": "verity"}}
    assert E.conditions_from_job(job)["exec.machine"] == "vy-tp2"
    assert E.conditions_from_job({"run": {}})["exec.machine"] is None
    # attempt.build: conditions recorded at run time (no exec.machine yet) are completed from job.json run.remote.machine
    d = tmp_path / "runs" / "r20260923-100000-0001"
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({"schema": "research/job/v0.2", **job, "store": {"conditions": {"hw.family": "cpu"}, "tool": {"name": "t", "version": "1"}}}))
    (d / "status.json").write_text(json.dumps({"state": "exited", "rc": 0}))
    m = A.build(d)
    assert m["conditions"]["exec.machine"] == "vy-tp2" and m["execution"]["machine"] == "vy-tp2"
    st = LocalStore(tmp_path / "st")
    st.put_attempt(m)
    assert st.index.sql("SELECT attempt FROM attempt_conditions WHERE key='exec.machine' AND value='vy-tp2'") == [(m["id"],)]


def test_unpreserved_lists_attempts_on_the_machine_that_the_gate_refuses(fs_store, machines, tmp_path):
    st = fs_store
    pushed = st.put_artifact("run-files/v1", {}, tree=_tree(tmp_path / "t1"))
    st.push(pushed)
    local_only = st.put_artifact("run-files/v1", {"never": "pushed"}, tree=_tree(tmp_path / "t2"))
    ok = _on(st, "r20260923-100000-0001", pushed, machine="vy-tp2")
    st.push_attempt(ok)
    bad = _on(st, "r20260923-100000-0002", local_only, machine="vy-tp2")            # artifact not on the remote, manifest not pushed
    legacy = _on(st, "r20260923-100000-0003", pushed, machine=None, remote={"machine": "vy-tp2", "pod_id": "pod-tp2"})   # pre-exec.machine manifest
    st.push_attempt(legacy)
    by_pod = "r20260923-100000-0004"                                                  # started ON the pod: only cost.pod_id ties it
    st.put_attempt(_attempt(run_id=by_pod, execution={"start_utc": "2026-09-23T10:00:00Z", "host": "c0ffee"}, cost={"provider": "runpod", "pod_id": "pod-tp2"}))
    st.push_attempt(by_pod)
    elsewhere = _on(st, "r20260923-100000-0005", local_only, machine="vy-other", host="other-host")
    linked = D.attempts_on(st, "vy-tp2", "pod-tp2", hostname="c0ffee")
    assert set(linked) == {ok, bad, legacy, by_pod}, "vy-other's attempt is not on this machine"
    assert linked[ok] == ["exec.machine", "execution.machine", "hw.hostname"] and linked[legacy] == ["execution.remote.machine", "cost.pod_id", "hw.hostname"]
    assert linked[by_pod] == ["cost.pod_id", "hw.hostname"]
    rep = D.report(st, "vy-tp2", probe=False)
    assert rep["machine"] == "vy-tp2" and rep["pod_id"] == "pod-tp2" and rep["n_attempts"] == 4 and rep["hostname"] is None
    assert rep["unpreserved"] == [bad] and set(rep["preserved"]) == {ok, legacy, by_pod}
    assert rep["attempts"][bad]["missing_names"] == ["outputs.result"] and rep["attempts"][by_pod]["preserved"] is True
    assert rep["check"]["preserved"] is False
    # the same predicate as `research data preserved`
    assert P.check(st, [bad])["preserved"] is False and P.check(st, [ok, legacy])["preserved"] is True
    lines = D.format_report(rep)
    assert "1 NOT preserved, 3 preserved" in lines[0] and f"  NOT preserved  {bad}  (exec.machine, execution.machine)  failing: manifest, artifacts" in lines
    assert f"      manifest  attempts/{bad}.json: absent on the remote" in lines
    assert any(l.startswith(f"      artifacts outputs.result {local_only}: manifests/") for l in lines)
    # by pod id, and an unknown target
    assert D.report(st, "pod-tp2", probe=False)["machine"] == "vy-tp2"
    assert D.report(st, "abcdefghijkl", probe=False)["n_attempts"] == 0 and D.report(st, "not-a-machine", probe=False)["pod_id"] is None
    assert D.report(st, "vy-tp2", mode="recorded", probe=False)["unpreserved"] == [bad], "offline from the catalog"


def test_report_names_the_failing_leg_and_what_it_concerns_not_a_blanket_manifest_reason(fs_store, machines, tmp_path):
    """r21 close-out: drain said "attempt manifest not on the remote" for attempts whose manifest WAS there; the labels leg failed."""
    st = fs_store
    art = st.put_artifact("run-files/v1", {}, tree=_tree(tmp_path / "t1"))
    st.push(art)
    aid = _on(st, "r20260923-100000-0001", art, machine="vy-tp2")
    st.push_attempt(aid)
    st.add_label(aid, "note", "laptop-only", by="laptop")
    st.add_label(art, "note", "laptop-only too", by="laptop")
    rep = D.report(st, "vy-tp2", probe=False)
    a = rep["attempts"][aid]
    assert rep["unpreserved"] == [aid] and a["legs"] == {"manifest": True, "artifacts": True, "labels": False}
    assert [(f["leg"], f["what"]) for f in a["failures"]] == [("labels", aid), ("labels", art)]
    assert all("1 of 1 local assertion(s) absent on the remote (first: labels/" in f["why"] for f in a["failures"])
    text = "\n".join(D.format_report(rep))
    assert f"NOT preserved  {aid}  (exec.machine, execution.machine)  failing: labels" in text and "manifest" not in text
    # the manifest leg, when it is the one failing, is named with its key; a blob missing is named by output name, art id and object
    (tmp_path / "t2").mkdir()
    (tmp_path / "t2" / "only.bin").write_bytes(b"a blob no other artifact shares")
    st2_art = st.put_artifact("run-files/v1", {"n": 2}, tree=tmp_path / "t2")
    st.push(st2_art)
    Path(st.remote.root / "objects" / "sha256" / st.get_manifest(st2_art).blobs()[0][0]).unlink()
    aid2 = _on(st, "r20260923-100000-0002", st2_art, machine="vy-tp2")
    a2 = D.report(st, "vy-tp2", probe=False)["attempts"][aid2]
    assert a2["legs"] == {"manifest": False, "artifacts": False, "labels": True}
    assert [(f["leg"], f["what"]) for f in a2["failures"]] == [("manifest", f"attempts/{aid2}.json"), ("artifacts", f"outputs.result {st2_art}")]
    assert a2["failures"][1]["why"].startswith("objects/sha256/") and a2["failures"][1]["why"].endswith(": absent")
    # `research data preserved` names the same failures (one predicate)
    assert P.failures(P.check(st, [aid])["targets"][0]) == a["failures"]


def test_drain_refuses_while_anything_is_unpreserved_and_logs_a_forced_override(fs_store, machines, tmp_path):
    st = fs_store
    killed: list[str] = []
    pushed = st.put_artifact("run-files/v1", {}, tree=_tree(tmp_path / "t1"))
    st.push(pushed)
    local_only = st.put_artifact("run-files/v1", {"never": "pushed"}, tree=_tree(tmp_path / "t2"))
    ok = _on(st, "r20260923-100000-0001", pushed, machine="vy-tp2")
    st.push_attempt(ok)
    bad = _on(st, "r20260923-100000-0002", local_only, machine="vy-tp2")
    log = tmp_path / "drain.log"

    rc, rep = D.drain(st, "vy-tp2", probe=False, terminate=killed.append, log_path=log)
    assert rc == 1 and killed == [] and rep["decision"]["action"] == "refused" and bad in rep["decision"]["why"] or "not preserved" in rep["decision"]["why"]
    rc, rep = D.drain(st, "vy-tp2", probe=False, force=True, terminate=killed.append, log_path=log)
    assert rc == 2 and killed == [] and "--reason" in rep["decision"]["why"], "an override without a reason is not an override"
    rc, rep = D.drain(st, "vy-tp2", probe=False, force=True, reason="evidence copied to /vault by hand", terminate=killed.append, log_path=log)
    assert rc == 0 and killed == ["pod-tp2"] and rep["decision"] == {"action": "terminated", "forced": True, "reason": "evidence copied to /vault by hand",
                                                                     "why": "FORCED over 1 unpreserved attempt(s): evidence copied to /vault by hand"}
    entries = [json.loads(l) for l in log.read_text().splitlines()]
    assert [e["action"] for e in entries] == ["refused", "terminated"] and entries[1]["forced"] and entries[1]["unpreserved"] == [bad]
    # once everything is preserved, drain terminates without force
    st.push(local_only)
    st.push_attempt(bad)
    rc, rep = D.drain(st, "vy-tp2", probe=False, terminate=killed.append, log_path=log)
    assert rc == 0 and killed == ["pod-tp2", "pod-tp2"] and rep["decision"] == {"action": "terminated", "forced": False, "reason": None, "why": "all 2 attempt(s) preserved"}
    # nothing recorded on a machine: terminate, and say so
    rc, rep = D.drain(st, "vy-other", probe=False, terminate=killed.append, log_path=log)
    assert rc == 0 and killed[-1] == "pod-other" and rep["decision"]["why"] == "nothing recorded on the machine"
    # unknown target: refused, nothing terminated
    rc, rep = D.drain(st, "not-a-machine", probe=False, terminate=killed.append, log_path=log)
    assert rc == 2 and killed[-1] == "pod-other" and "neither" in rep["decision"]["why"]
    # the terminate API failing is reported, not hidden

    def boom(pod_id):
        raise RuntimeError("HTTP 500")

    rc, rep = D.drain(st, "vy-tp2", probe=False, terminate=boom, log_path=log)
    assert rc == 2 and rep["decision"]["action"] == "error" and "HTTP 500" in rep["decision"]["why"]


def test_pods_cli_routes_unpreserved_and_drain_and_require_preserved(fs_store, machines, tmp_path, capsys, monkeypatch):
    st = fs_store
    local_only = st.put_artifact("run-files/v1", {"never": "pushed"}, tree=_tree(tmp_path / "t2"))
    _on(st, "r20260923-100000-0002", local_only, machine="vy-tp2")
    assert part.main(["unpreserved", "vy-tp2", "--no-probe"]) == 1
    out = capsys.readouterr().out
    assert "NOT preserved  r20260923-100000-0002" in out
    assert part.main(["unpreserved", "vy-tp2", "--no-probe", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["unpreserved"] == ["r20260923-100000-0002"]
    assert part.main(["unpreserved", "not-a-machine", "--no-probe"]) == 2
    assert part.main(["unpreserved", "abcdefghijkl", "--no-probe"]) == 0, "a raw pod id nothing was recorded on"
    killed: list[str] = []
    monkeypatch.setattr("research.pods.runpod.terminate", killed.append)
    assert part.main(["drain", "vy-tp2", "--no-probe"]) == 1 and killed == []
    assert "REFUSED" in capsys.readouterr().out
    assert part.main(["terminate", "pod-tp2", "--require-preserved", "--no-probe"]) == 1 and killed == []
    assert part.main(["drain", "vy-tp2", "--no-probe", "--force", "--reason", "copied by hand"]) == 0 and killed == ["pod-tp2"]
    assert "TERMINATED: FORCED" in capsys.readouterr().out
    assert Path(tmp_path / "drain.log").exists()
