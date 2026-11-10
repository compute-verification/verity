"""pods/nebius/wake.py, the control pod's side of `research run --on`'s wake, and remote.wake's fallback to it: a lane's VM holds no
Nebius key, so the control pod starts a stopped VM for it, but only one the registry names, before its deadline, under a live
budget line. No cloud and no ssh: the Nebius CLI and the transport to the pod are injected."""
from __future__ import annotations

import io
import json
import os
import subprocess
from contextlib import redirect_stderr, redirect_stdout

import pytest

from research import remote
from research.pods import budgets
from research.pods.nebius import wake as W

IID = "computeinstance-e00ea3tycx9kk3dafr"
NOW = budgets.parse_time("2026-10-19T18:30:00Z")


def guard_state(expires="2026-10-24T21:00:00Z", last_ok=NOW - 30) -> dict:
    return {"last_ok_poll": last_ok, "lines": {},
            "budgets": {"source": "budgets.toml", "lines": {"vy-glm53-": {"prefix": "vy-glm53-", "cap_usd": 24000, "expires": expires,
                                                                           "max_pod_hours": 192}}}}


@pytest.fixture
def pod(tmp_path, monkeypatch):
    """The pod's files: the steward's machines.d, the guard's state and the key file."""
    d = tmp_path / "machines.d"
    d.mkdir()
    (d / "vy-glm53-eu.toml").write_text(f'provider = "ssh"\nhost = "203.0.113.7"\nproject = "verity"\nlane = "circuits"\n'
                                        f'nebius_instance_id = "{IID}"\ndeadline = "2026-10-23T21:00:00Z"\n')
    (d / "vy-dc-h200.toml").write_text('provider = "ssh"\nhost = "203.0.113.8"\nproject = "verity"\n')
    gs = tmp_path / "guard-budgets.json"
    gs.write_text(json.dumps(guard_state()))
    key = tmp_path / "nebius-wake.env"
    monkeypatch.setattr(W, "MACHINES_D", d)
    monkeypatch.setattr(W, "GUARD_STATE", gs)
    monkeypatch.setattr(W, "WAKE_ENV", key)
    return d, gs, key


class Nebius:
    def __init__(self, state="STOPPED"):
        self.state, self.calls = state, []

    def __call__(self, *args):
        self.calls.append(args[2])
        return {"status": {"state": self.state}} if args[2] == "get" else {}


def run_pod(argv, nebius) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = W.main(argv, nebius=nebius, now=NOW)
    return rc, out.getvalue(), err.getvalue()


def test_the_pod_starts_a_stopped_registered_instance_and_leaves_a_running_one(pod):
    nb = Nebius()
    rc, out, _ = run_pod(["start", IID], nb)
    assert rc == 0 and json.loads(out) == {"name": "vy-glm53-eu", "instance": IID, "was": "STOPPED", "started": True}
    assert nb.calls == ["get", "start"]
    nb = Nebius("RUNNING")
    rc, out, _ = run_pod(["start", IID], nb)
    assert rc == 0 and json.loads(out)["started"] is False and nb.calls == ["get"]


def test_the_pod_starts_only_an_instance_exactly_one_registry_entry_names(pod):
    """The allowlist: an id no entry names, or two entries name, is refused before any Nebius call; so is a malformed one."""
    d, _, _ = pod
    for iid, why in (("computeinstance-e00notregistered00", "named by 0 machines.d entries"),
                     ("computeinstance-x; rm -rf /", "is not a Nebius instance id")):
        nb = Nebius()
        rc, _, err = run_pod(["start", iid], nb)
        assert rc == 2 and why in err and nb.calls == []
    (d / "vy-glm53-eu-copy.toml").write_text((d / "vy-glm53-eu.toml").read_text())
    rc, _, err = run_pod(["start", IID], Nebius())
    assert rc == 2 and "named by 2 machines.d entries (vy-glm53-eu, vy-glm53-eu-copy)" in err


def test_the_pod_refuses_past_the_vms_deadline_or_with_none_recorded(pod):
    """The VM stops itself 5 min before /etc/research/deadline; a start then would only bill until it stops again."""
    d, _, _ = pod
    f = d / "vy-glm53-eu.toml"
    text = f.read_text()
    f.write_text(text.replace("2026-10-23T21:00:00Z", "2026-10-19T18:34:00Z"))
    rc, _, err = run_pod(["start", IID], nb := Nebius())
    assert rc == 2 and "deadline 2026-10-19T18:34:00Z has passed" in err and nb.calls == []
    f.write_text("\n".join(ln for ln in text.splitlines() if not ln.startswith("deadline")) + "\n")
    rc, _, err = run_pod(["start", IID], nb)
    assert rc == 2 and "no recorded deadline" in err and nb.calls == []


def test_the_pod_refuses_when_the_guard_finds_no_live_budget_line(pod):
    """budgets.gate on the guard's own state, as `pods create` judges a new pod: an expired line, a stale guard, no state."""
    _, gs, _ = pod
    for st, why in ((guard_state(expires="2026-10-19T18:00:00Z"), "expired at 2026-10-19T18:00:00Z"),
                    (guard_state(last_ok=NOW - 600), "last completed a poll 10.0 min ago")):
        gs.write_text(json.dumps(st))
        rc, _, err = run_pod(["start", IID], nb := Nebius())
        assert rc == 2 and why in err and nb.calls == []
    gs.unlink()
    rc, _, err = run_pod(["start", IID], Nebius())
    assert rc == 2 and "cannot read the budgets guard's state" in err


def test_the_pod_refuses_without_a_key_that_may_start_vms_and_reads_one_from_its_file(pod, monkeypatch):
    _, _, key = pod
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = W.main(["start", IID], now=NOW)
    assert rc == 2 and "holds no Nebius key that may start a VM" in err.getvalue()
    for k in ("NEBIUS_SA_ID", "NEBIUS_BIN"):    # recorded, so what load_key sets is undone after the test
        monkeypatch.setenv(k, "before")
        monkeypatch.delenv(k)
    key.write_text("# start-capable\nNEBIUS_SA_ID=serviceaccount-dummy\nNEBIUS_BIN='/data/nebius/bin/nebius'\n")
    W.load_key(key)
    assert os.environ["NEBIUS_SA_ID"] == "serviceaccount-dummy" and os.environ["NEBIUS_BIN"] == "/data/nebius/bin/nebius"


def test_a_dry_run_checks_and_starts_nothing(pod):
    rc, out, _ = run_pod(["start", IID, "--dry-run"], nb := Nebius())
    assert rc == 0 and json.loads(out)["dry_run"] is True and nb.calls == []


# ---- the client: remote.wake ------------------------------------------------------------------------------------------------------------

def machine():
    return {"name": "vy-glm53-eu", "nebius_instance_id": IID}


def test_with_no_nebius_key_here_run_on_asks_the_control_pod_and_waits_for_ssh(pod, monkeypatch):
    """A lane's VM (no NEBIUS_SA_*): `research run --on` asks the control pod, which here is wake.py run in-process."""
    for k in remote.NEBIUS_KEY_ENV:
        monkeypatch.delenv(k, raising=False)
    answers = iter([1, 0, 0])
    monkeypatch.setattr(remote, "ssh_run", lambda ssh, argv, timeout=120: subprocess.CompletedProcess(argv, next(answers)))
    nb, asked, t = Nebius(), [], [0.0]

    def pod_side(argv, **kw):
        asked.append(argv)
        rc, out, err = run_pod(["start", argv[-1].split()[-1]], nb)
        return subprocess.CompletedProcess(argv, rc, out, err)

    monkeypatch.setattr("research.pods.connect.ssh_base", lambda host: ["ssh", f"root@{host}"])

    def sleep(s):
        t[0] += s

    got = remote.wake(machine(), ["ssh"], control=lambda iid: remote.wake_on_control(iid, run=pod_side), sleep=sleep,
                      clock=lambda: t[0])
    assert got == "started from STOPPED" and nb.calls == ["get", "start"] and t[0] == remote.WAKE_POLL_S
    assert asked == [["ssh", "root@vy-control-verity", f"{remote.CONTROL_WAKE} {IID}"]]


def test_the_default_without_a_key_is_the_control_pod_and_its_refusal_names_the_reason(monkeypatch):
    for k in remote.NEBIUS_KEY_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(remote, "ssh_run", lambda ssh, argv, timeout=120: subprocess.CompletedProcess(argv, 1))
    monkeypatch.setattr("research.pods.connect.ssh_base", lambda host: ["ssh", f"root@{host}"])
    refusal = "wake.py: refused: vy-glm53-eu's deadline 2026-10-19T16:00:00Z has passed (it stops itself 5 min before)"
    monkeypatch.setattr(remote.subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 2, "", refusal + "\n"))
    with pytest.raises(remote.Refused, match="started by the control pod: the control pod vy-control-verity refused: wake.py: "
                                             "refused: vy-glm53-eu's deadline 2026-10-19T16:00:00Z has passed"):
        remote.wake(machine(), ["ssh"])


def test_an_unreachable_control_pod_is_a_refusal_not_a_crash(monkeypatch):
    def no_runpod(host):
        raise remote.Refused("runpod API unavailable: no RUNPOD_API_KEY")

    monkeypatch.setattr("research.pods.connect.ssh_base", no_runpod)
    with pytest.raises(remote.Refused, match="the control pod vy-control-verity can't be reached .*no RUNPOD_API_KEY"):
        remote.wake_on_control(IID)
