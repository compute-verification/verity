"""`research inspect|status|fetch RUN` for a run another VM launched (`remote.adopt`): the run's machine is found among the ssh
machines, a local remote.json is written as `launch` would have, and the records are fetched.  The machines are local directories
reached over `sh -c` (the same argv `ssh_run` sends), so no network is used."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research import cli, remote

RUN = "r20261002-061029-1beb"
LIVE = "r20261002-071855-0131"


def _run_on(root: Path, run_id: str, state: str) -> Path:
    rd = root / "runs" / run_id
    rd.mkdir(parents=True)
    trans = [{"state": "running", "t_utc": "2026-10-02T06:10:30Z"}] + ([{"state": state, "t_utc": "2026-10-02T06:40:00Z"}] if state != "running" else [])
    (rd / "status.json").write_text(json.dumps({"schema": "telemetry/status/v0.1", "state": state, "rc": 0 if state == "done" else None,
                                                "transitions": trans}))
    (rd / "stdout.log").write_text("big output stays on the machine\n")
    return rd


@pytest.fixture
def machines(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    ms = {"vy-a": {"provider": "ssh", "host": "a", "user": "research", "root": str(tmp_path / "a"), "project": "verity"},
          "vy-b": {"provider": "ssh", "host": "b", "user": "research", "root": str(tmp_path / "b"), "project": "verity"},
          "pod": {"provider": "runpod", "pod_id": "p1", "root": "/workspace/research", "project": "verity"},
          "vy-pod": {"provider": "runpod", "pod_id": "p2", "host": "p", "user": "root", "project": "verity"}}
    probed: list[str] = []
    monkeypatch.setattr(remote, "load_machines", lambda path=None: ms)
    monkeypatch.setattr(remote, "pod_target", lambda m: probed.append(m["host"]) or {"host": m["host"], "port": 22, "user": m["user"]})
    monkeypatch.setattr(remote, "ssh_argv", lambda host, port, **kw: ["sh", "-c"])
    _run_on(tmp_path / "b", RUN, "done")
    _run_on(tmp_path / "a", LIVE, "running")
    return {"local": tmp_path / "here", "probed": probed, "tmp": tmp_path}


def test_inspect_finds_a_run_another_vm_launched_on_the_machine_that_holds_it_and_fetches_its_records(machines, capsys):
    here = machines["local"]
    assert cli.main(["inspect", RUN, "--runs-dir", str(here)]) == 0
    out = capsys.readouterr().out
    assert "exited    done  rc=0" in out and "observed  phase=adopted" in out
    rec = json.loads((here / RUN / "remote.json").read_text())
    assert (rec["machine"], rec["phase"], rec["remote_run_dir"]) == ("vy-b", "adopted", str(machines["tmp"] / "b" / "runs" / RUN))
    assert rec["observed_utc"] and rec["last_observed"]["state"] == "done"
    assert (here / RUN / "status.json").exists() and not (here / RUN / "stdout.log").exists(), "the records, not the run's files"
    assert machines["probed"] == ["a", "b"], "each ssh machine in name order, never the pod"

    assert cli.main(["status", RUN, "--runs-dir", str(here)]) == 0
    (row,) = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith(RUN)]
    assert " done " in row and " vy-b " in row, "status RUN shows the adopted run and where it is"
    assert machines["probed"] == ["a", "b"], "a local, finished run is not fetched again"


def test_status_run_adopts_a_running_run_and_on_names_the_only_machine_asked(machines, capsys):
    here = machines["local"]
    assert cli.main(["status", LIVE, "--on", "vy-a", "--runs-dir", str(here), "--json"]) == 0
    (s,) = json.loads(capsys.readouterr().out)["runs"]
    assert s["id"] == LIVE and s["state"] == "running" and machines["probed"] == ["a"], "only the machine --on names is asked"


def test_a_pod_registered_without_a_root_is_asked_under_the_default_one(machines, monkeypatch, capsys):
    """`research pods register` writes no root, which `resolve_machine` defaults: adopting from such a pod uses that default
    rather than failing on the missing key."""
    monkeypatch.setattr(remote, "DEFAULT_ROOT", str(machines["tmp"] / "pod"))
    _run_on(machines["tmp"] / "pod", "r20261002-081500-0a0a", "running")
    here = machines["local"]
    assert cli.main(["status", "r20261002-081500-0a0a", "--on", "vy-pod", "--runs-dir", str(here), "--json"]) == 0
    (s,) = json.loads(capsys.readouterr().out)["runs"]
    assert s["state"] == "running" and machines["probed"] == ["p"]
    rec = json.loads((here / "r20261002-081500-0a0a" / "remote.json").read_text())
    assert rec["root"] == str(machines["tmp"] / "pod") and rec["machine"] == "vy-pod"


def test_a_run_no_machine_holds_names_each_machines_answer_and_the_store_attempt(machines, capsys):
    from research.store.local import LocalStore

    here, gone = machines["local"], "r20261002-000000-dead"
    assert cli.main(["inspect", gone, "--runs-dir", str(here)]) == 1
    err = capsys.readouterr().err
    assert f"vy-a: no {machines['tmp'] / 'a' / 'runs' / gone}" in err and f"vy-b: no {machines['tmp'] / 'b' / 'runs' / gone}" in err
    assert "research data show" not in err and not (here / gone).exists()

    LocalStore().put_attempt({"schema": "research/attempt/v0.1", "id": gone, "tool": {"name": "t", "version": "1"},
                              "derivation": "drv:" + "a" * 64, "reuse_key": None, "argv": ["x"], "params": {}, "key_params": [],
                              "inputs": {}, "conditions": {}, "key_conditions": [], "source": {"commit": "c" * 40, "dirty": False},
                              "closure": None, "execution": {"start_utc": "2026-10-02T00:00:00Z", "host": "h", "project": "verity",
                                                             "campaign": "c"}, "cost": None,
                              "status": {"state": "done", "rc": 0, "validation": "passed"}, "outputs": {}, "telemetry": None})
    assert cli.main(["fetch", gone, "--runs-dir", str(here)]) == 1
    assert f"the store has its attempt: `research data show {gone}`" in capsys.readouterr().err


def test_the_machines_are_asked_at_once_and_the_first_that_holds_the_run_answers(machines, monkeypatch, capsys):
    """One at a time, a run on a late machine waited behind every machine before it, a stopped one costing its whole connect
    timeout. vy-a's probe here returns only once vy-b's has started, which a walk one machine at a time never reaches; the
    probes connect with ADOPT_CONNECT_S, and one line says what the wait is and how to skip it."""
    import threading

    both = threading.Barrier(2)
    seen = []
    real = remote.ssh_run

    def ssh_run(ssh, argv, **kw):
        seen.append(kw.get("timeout"))
        both.wait()
        return real(ssh, argv, **kw)

    argvs = []
    monkeypatch.setattr(remote, "ssh_run", ssh_run)
    monkeypatch.setattr(remote, "ssh_argv", lambda host, port, **kw: argvs.append(kw) or ["sh", "-c"])
    local = remote.adopt(machines["local"], RUN)
    assert json.loads((local / "remote.json").read_text())["machine"] == "vy-b"
    assert seen == [remote.ADOPT_PROBE_S] * 2 and [kw.get("connect_timeout") for kw in argvs[:2]] == [remote.ADOPT_CONNECT_S] * 2
    assert f"research: looking for {RUN} on 2 machines (--on NAME skips this)" in capsys.readouterr().err


def test_a_cancel_asks_every_machine_and_refuses_a_run_two_hold(machines):
    _run_on(machines["tmp"] / "a", RUN, "running")
    with pytest.raises(remote.Refused, match="vy-a and vy-b each hold a run"):
        remote.adopt(machines["local"], RUN, unique=True)


def test_a_prefix_is_never_looked_up_on_the_machines(machines, capsys):
    assert cli.main(["inspect", "r20261002", "--runs-dir", str(machines["local"])]) == 1
    assert "no run matching 'r20261002'" in capsys.readouterr().err and machines["probed"] == []
