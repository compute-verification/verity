"""A launch whose run id is another launch's on the machine: it overwrites nothing, never says launched, and a picked id is
picked again.  The machine is a directory reached through `sh -c`, and its `launch-request` is this package's own."""
from __future__ import annotations

import io
import json
import subprocess
import sys
from contextlib import redirect_stderr
from pathlib import Path

import pytest

from research import cli, remote
from research import runs as R
from waits import wait_runner

PY = sys.executable
SRC = str(Path(remote.__file__).resolve().parents[1])
A, B = "r20000101-000000-aaaa", "r20000101-000000-bbbb"
SSH = ["sh", "-c"]


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """The machine `m`'s root."""
    root = tmp_path / "m"
    monkeypatch.setattr(remote, "resolve_machine", lambda name, project: {"root": str(root), "python": PY})
    monkeypatch.setattr(remote, "pod_target", lambda m: {"host": "m", "port": 22, "user": "research", "pod_name": "n1",
                                                         "cost_per_hr": 0, "rate_observed_utc": "t"})
    monkeypatch.setattr(remote, "ssh_argv", lambda host, port, **kw: SSH)
    monkeypatch.setattr(remote, "ship_tool", lambda *a: SRC)
    monkeypatch.setattr(remote, "launch_pause", lambda attempt: 0)
    return root


def _another_launchs_run(root: Path, run_id: str) -> Path:
    """A run dir under `root` that no request of this test's launches claimed."""
    d = root / "runs" / run_id
    d.mkdir(parents=True)
    (d / "launch.json").write_text(json.dumps({"run_id": run_id, "runner_pid": 4242}))
    return d


def _launch(tmp_path: Path, run_id: str) -> dict:
    return remote.launch(run_id=run_id, local_dir=tmp_path / "here" / run_id, machine_name="m", project="p", command=[PY, "-c", "pass"],
                         harness_args=["--stage", "t", "--no-sampler"], source=None, sends=[], cwd=None, env={}, exclusive=False)


def test_a_request_another_launch_staged_is_never_overwritten(machine):
    """Rule: staging is a claim; weakening: `mkdir -p` of the request dir, which overwrites the first launch's request and key."""
    first = remote.stage_request(SSH, str(machine), A, {"by": "first"}, [], private=[("custody/cred.json", b"first's key")])
    with pytest.raises(remote.RunIdTaken, match=f"run id {A} is another launch's on the machine .*launch again"):
        remote.stage_request(SSH, str(machine), A, {"by": "second"}, [], private=[("custody/cred.json", b"second's key")])
    assert json.loads(Path(first).read_text()) == {"by": "first"}
    assert (Path(first).parent / "custody" / "cred.json").read_bytes() == b"first's key"


def test_an_existing_run_its_request_did_not_claim_is_refused_and_what_it_staged_is_removed(tmp_path, machine):
    """Rule: `existing` is this launch's only when its own request claimed it; weakening: any `existing` reads as launched."""
    theirs = _another_launchs_run(machine, A)
    with pytest.raises(remote.RunIdTaken, match=f"run id {A} is another launch's on m .*launch again"):
        _launch(tmp_path, A)
    assert json.loads((tmp_path / "here" / A / "remote.json").read_text())["phase"] == "refused"
    assert not (machine / "requests" / A).exists(), "nothing it staged stays"
    assert json.loads((theirs / "launch.json").read_text())["runner_pid"] == 4242 and sorted(p.name for p in theirs.iterdir()) == ["launch.json"]


def test_a_redelivery_that_finds_the_run_its_own_request_claimed_says_launched(tmp_path, machine, monkeypatch):
    """Rule: a re-delivery after an ssh drop reports its own run, as today; weakening: refusing every `existing`."""
    real, answered = remote.ssh_run, []

    def dropped_once(ssh, argv, **kw):
        r = real(ssh, argv, **kw)
        if "launch-request" in argv and not answered:
            answered.append(json.loads(r.stdout.decode().strip().splitlines()[-1]))
            return subprocess.CompletedProcess(r.args, 255, b"", b"connection reset")
        return r

    monkeypatch.setattr(remote, "ssh_run", dropped_once)
    rec = _launch(tmp_path, A)
    wait_runner(machine / "runs" / A)
    assert answered[0]["started"] and rec["phase"] == "launched"
    assert rec["launch"]["existing"] and rec["launch"]["this_request"] and rec["launch"]["runner_pid"] == answered[0]["runner_pid"]
    assert R.summarize(machine / "runs" / A)["rc"] == 0


@pytest.mark.parametrize("dropped, phase", [(False, None), (True, "launched")])
def test_a_machine_that_predates_the_mark_is_answered_by_whether_an_earlier_delivery_may_have_reached_it(tmp_path, machine,
                                                                                                          monkeypatch, dropped, phase):
    """Rule: with no `this_request` in the answer, only a delivery after one that may have reached the machine is this launch's;
    weakening: reading a missing `this_request` as either answer."""
    real, deliveries = remote.ssh_run, []
    existing = {"claimed": False, "run_id": A, "existing": True, "state": "running", "runner_pid": 4242}

    def old_machine(ssh, argv, **kw):
        if "launch-request" not in argv:
            return real(ssh, argv, **kw)
        deliveries.append(argv)
        if dropped and len(deliveries) == 1:
            return subprocess.CompletedProcess(argv, 255, b"", b"connection reset")
        return subprocess.CompletedProcess(argv, 0, json.dumps(existing).encode() + b"\n", b"")

    monkeypatch.setattr(remote, "ssh_run", old_machine)
    if phase is None:
        with pytest.raises(remote.RunIdTaken, match=f"run id {A}"):
            _launch(tmp_path, A)
    else:
        assert _launch(tmp_path, A)["phase"] == phase


def _run(here: Path, *flags: str) -> tuple[int, str]:
    err = io.StringIO()
    with redirect_stderr(err):
        rc = cli.main(["run", "--runs-dir", str(here), "--on", "m", "--project", "p", "--no-custody-r2", "--no-sampler", *flags, "--",
                       PY, "-c", "pass"])
    return rc, err.getvalue()


def test_a_launch_that_picked_its_id_picks_a_new_one_when_the_machine_has_it_and_one_given_its_id_is_refused(tmp_path, machine,
                                                                                                           monkeypatch):
    """Rule: a picked id met on the machine is picked again, and a given one is refused; weakening: no new pick, or one for --id."""
    handed = []

    def ids():
        handed.append(B if handed else A)
        return handed[-1]

    monkeypatch.setattr(R, "new_run_id", ids)
    _another_launchs_run(machine, A)
    here = tmp_path / "here"
    rc, err = _run(here)
    wait_runner(machine / "runs" / B)
    assert rc == 0 and handed == [A, B], err
    assert f"run id {A} is another launch's there; launching again with a new id" in err
    assert sorted(p.name for p in here.iterdir()) == [B], "the dir of the id that was the other launch's is not kept here"
    assert json.loads((here / B / "remote.json").read_text())["phase"] == "launched" and R.summarize(machine / "runs" / B)["rc"] == 0

    handed.clear()
    rc, err = _run(here, "--id", A)
    assert rc == 3 and handed == [] and "launching again" not in err, err
    assert f"refused: run id {A} is another launch's on m" in err and "launch again" in err
    assert sorted(p.name for p in here.iterdir()) == [B]
