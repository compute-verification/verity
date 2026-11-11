"""The control pod's door (infra/control/vy-control-gate, door_setup.sh) and `research` through it (research.pods.door): the gate
runs one verb with checked arguments or refuses, no request reaches a shell, and a lane's VM that holds neither root's key there
nor the RunPod account's still passes `pods create`'s budget gate, wakes a node and lists pods. Nothing here logs in anywhere:
the transport to the pod is a function that hands the request to the gate, and RunPod and the pod's programs are fakes."""
from __future__ import annotations

import argparse
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from research import remote
from research.pods import budgets, door, runpod
from verity_ops import deploy

ROOT = Path(__file__).resolve().parents[3]
CONTROL = ROOT / "infra" / "control"
NOW = budgets.parse_time("2026-10-19T18:30:00Z")
IID = "computeinstance-e00fakeinstance0id"
SHA = "0123456789abcdef0123456789abcdef01234567"
CONTROL_POD, LANE_POD, OTHER_POD = "p1ugdh9hfgqe5d", "abc123def45678", "mmm000nnn11122"
KEY_A = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI" + "DummyAgentsKeyDummyAgentsKeyDummyAgents"
KEY_CI = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI" + "DummyCiKeyDummyCiKeyDummyCiKeyDummyCiKe"
KEY_ADMIN = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI" + "DummyAdminKeyDummyAdminKeyDummyAdminKey"
RUNPOD_REQUEST = runpod._request  # a lane's VM's, while the pod's is its fake


def guard_state() -> dict:
    line = {"prefix": "vy-a-", budgets.CAP_KEYS[0]: 3.0, "expires": "2026-10-24T21:00:00Z", "max_pod_hours": 4.0}
    return {"last_ok_poll": NOW - 30, "lines": {}, "budgets": {"source": "budgets.toml", "lines": {"vy-a-": line}}}


class GateSource(importlib.machinery.SourceFileLoader):
    """The gate's file, compiled from its source each time: it has no suffix, and no bytecode cache is looked for or written."""

    def get_code(self, fullname):
        return self.source_to_code(self.get_data(self.path), self.path)


class Pod:
    """What the gate reaches on the pod, faked: RunPod's API, and the programs it starts (recorded, never run)."""

    def __init__(self):
        self.calls, self.ran = [], []
        self.pods = {CONTROL_POD: {"id": CONTROL_POD, "name": "vy-control-verity", "env": {"PUBLIC_KEY": KEY_A}},
                     LANE_POD: {"id": LANE_POD, "name": "vy-a-1-veritor-campaign", "env": {"TOKEN": "x"}, "desiredStatus": "RUNNING"},
                     OTHER_POD: {"id": OTHER_POD, "name": "plain-name"}}

    def api(self, method, path, body=None):
        self.calls.append((method, path, body))
        if (method, path) == ("GET", "/pods"):
            return list(self.pods.values())
        if (method, path) == ("POST", "/pods"):
            return {"id": "new000new00000", **body}
        if path[len("/pods/"):] not in self.pods:
            raise runpod.PodError(f"{method} {path} -> HTTP 404: not found (terminated, or never existed)", code=404)
        return self.pods[path[len("/pods/"):]] if method == "GET" else None

    def run(self, argv, env):
        self.ran.append((argv, env))
        return 0


@pytest.fixture
def gate(tmp_path, monkeypatch):
    """The gate as the pod runs it, on a fake pod: its state file, its log and its clock are the test's."""
    loader = GateSource("vy_control_gate", str(CONTROL / "vy-control-gate"))
    G = importlib.util.module_from_spec(importlib.util.spec_from_loader("vy_control_gate", loader))
    loader.exec_module(G)
    (tmp_path / "state.json").write_text(json.dumps(guard_state()))
    G.pod = Pod()
    for name, value in (("STATE", tmp_path / "state.json"), ("LOG", tmp_path / "door" / "gate.log"), ("clock", lambda: NOW),
                        ("GUARD_SRC", str(ROOT / "tools/research/src")), ("run", G.pod.run)):
        monkeypatch.setattr(G, name, value)
    monkeypatch.setattr(runpod, "_request", G.pod.api)     # the pod holds the account's key: its research reaches RunPod
    monkeypatch.setenv("RESEARCH_GUARD_HOST", "")          # the gate sets it for the code it loads: undone with the test
    return G


def serve(G, request, lane="-", stdin=b""):
    old, sys.stdin = sys.stdin, argparse.Namespace(buffer=io.BytesIO(stdin))
    try:
        return G.serve(request, lane)
    finally:
        sys.stdin = old


def create_body(**flags) -> dict:
    """The body `research pods create` sends for these flags: pods/runpod.py's own `create`, its request captured."""
    ap = argparse.ArgumentParser()
    runpod.create_args(ap)
    ns = ap.parse_args(["--name", "vy-a-1", *[w for k, v in flags.items() for w in (f"--{k.replace('_', '-')}", str(v))]])
    ns.lease_expires = int(NOW + 2 * 3600)
    sent = {}
    with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(io.StringIO()):
        mp.setattr(runpod, "public_key", lambda: f"{KEY_A} agents")
        mp.setattr(runpod, "_request", lambda method, path, body=None: sent.update(body) or {})
        runpod.create(ns)
    return sent


# ---- the gate ----------------------------------------------------------------------------------------------------------------------

def test_each_verb_runs_what_the_gate_builds_and_nothing_of_the_callers(gate, capsys):
    assert serve(gate, "true") == 0 and serve(gate, "hostname") == 0
    assert capsys.readouterr().out.strip() == gate.socket.gethostname()
    assert serve(gate, "budgets-state") == 0
    body, _, clock = capsys.readouterr().out.rstrip().rpartition("\n")          # as budgets.read_guard_state reads it
    assert json.loads(body) == guard_state() and float(clock) == int(NOW)
    assert serve(gate, f"wake {IID}") == 0
    assert gate.pod.ran == [([gate.PYTHON, "-m", "research.pods.nebius.wake", "start", IID], {**gate.ENV, "PYTHONPATH": gate.WAKE_SRC})]
    assert serve(gate, "pods-list") == 0
    listed = json.loads(capsys.readouterr().out)
    assert [p["id"] for p in listed] == [CONTROL_POD, LANE_POD, OTHER_POD] and not any("env" in p for p in listed), \
        "a pod's environment is its creator's: no caller reads it"
    assert serve(gate, f"pods-get {LANE_POD}") == 0
    assert json.loads(capsys.readouterr().out) == {"id": LANE_POD, "name": "vy-a-1-veritor-campaign", "desiredStatus": "RUNNING"}
    assert serve(gate, f"pods-terminate {LANE_POD}") == 0 and gate.pod.calls[-1] == ("DELETE", f"/pods/{LANE_POD}", None)
    log = gate.LOG.read_text().splitlines()
    assert len(log) == 7 and log[3] == f"2026-10-19T18:30:00Z - rc=0 {f'wake {IID}'!r}", "every call is one line of the log"


INJECTIONS = [
    "true; id", "true && id", "true | id", "true\nid", "true $(id)", "true `id`", "true > /data/infra/drive.sh", "hostname -f",
    "budgets-state /etc/shadow", "budgets-state; cat /root/.config/verity/nebius-wake.env",
    f"wake {IID};id", f"wake {IID} --force", "wake $(id)", "wake `id`", "wake ../../etc/passwd", "wake computeinstance-A", "wake",
    f"wake '{IID} x'", f"pods-get {LANE_POD};id", "pods-get ../pods", "pods-get", f"pods-get {LANE_POD} {CONTROL_POD}",
    "pods-terminate 'a b'", "pods-terminate ALL", "pods-list --all", "pods-create --network-volume 4ue28s1c6g",
    # near misses of the two requests an older checkout sends
    "cat /etc/shadow && echo && date +%s", "cat /root/.research/pods/guard-budgets.json && echo && date +%s; id",
    "cat /root/.research/pods/guard-budgets.json /root/.runpod/config.toml && echo && date +%s",
    f"PYTHONPATH=/tmp/x python3.12 -m research.pods.nebius.wake start {IID}",
    f"PYTHONPATH=/data/infra/store-index-src/src python3.12 -m research.pods.nebius.wake start {IID}; id",
    f"PYTHONPATH=/data/infra/store-index-src/src python3.12 -c 'import os' start {IID}",
    # what a shell account would take
    "", "   ", "bash", "sh -c id", "id", "scp -t /tmp", "rsync --server -e.LsfxC . /data", "env X=1 true", "/bin/true", "TRUE",
    "sudo -n true", "test -d /workspace/research/runs/r1", "true 'unbalanced", "true \x07", "true caf\u00e9", "true " + "x" * 2000,
]


@pytest.mark.parametrize("request_", INJECTIONS)
def test_a_request_that_is_not_a_verb_with_its_arguments_is_refused_and_nothing_runs(gate, capsys, request_):
    for lane in ("-", "ci"):
        assert serve(gate, request_, lane, stdin=json.dumps(create_body()).encode()) == 2
        assert capsys.readouterr().err.startswith("vy-control-gate: refused: ")
    assert gate.pod.ran == [] and gate.pod.calls == [], "nothing was started and RunPod was not asked"
    assert gate.LOG.read_text().count(" refused ") == 2


def test_an_unknown_verb_is_refused_with_the_verbs_named(gate, capsys):
    assert serve(gate, "nvidia-smi -L") == 2
    err = capsys.readouterr().err
    assert "'nvidia-smi' is no verb of the gate" in err and all(v in err for v in gate.VERBS)
    assert serve(gate, "true", lane="ci; id") == 2 and "lane" in capsys.readouterr().err, "the key's lane is checked too"


def test_the_two_requests_an_older_checkout_sends_are_taken_as_their_verbs(gate, capsys):
    """Every checkout older than the door sends root's shell these two strings: once the registry names the gated account they
    arrive at the gate, which must take them or every such lane's `pods create` and wake stop."""
    assert serve(gate, f"cat {budgets.GUARD_STATE} && echo && date +%s") == 0
    assert json.loads(capsys.readouterr().out.rstrip().rpartition("\n")[0]) == guard_state()
    assert serve(gate, f"{remote.CONTROL_WAKE} {IID}") == 0 and gate.pod.ran[0][0][-2:] == ["start", IID]


def test_drive_is_cis_key_only_and_only_its_listed_forms(gate, capsys):
    forms = ["status", "start", "stop", "once --dry-run", f"deploy {SHA}", "deploy main", "queue promote",
             f"queue baseline {SHA} --by ci", f"queue baseline {SHA[:12]} --by ci --steps suites,lean-check",
             f"queue accept-checked {SHA} --by ci", f"queue drive-shard-add {SHA} 3 20261019T1830Z-check-k3",
             "queue eject 6861 --by ci --reason 'red on checked: suites (tools/research)'"]
    for form in forms:
        assert serve(gate, f"drive {form}", "ci") == 0, form
    assert [a for a, _ in gate.pod.ran][-1] == [gate.DRIVE_SH, "queue", "eject", "6861", "--by", "ci", "--reason",
                                                "red on checked: suites (tools/research)"], "one word a caller's word: no shell"
    assert all(a[0] == gate.DRIVE_SH and e == gate.ENV for a, e in gate.pod.ran) and len(gate.pod.ran) == len(forms)
    gate.pod.ran.clear()
    for lane in ("-", "infra"):
        assert serve(gate, "drive status", lane) == 2 and "`drive` is for the key of ci" in capsys.readouterr().err
    for bad in ["", "merge --push", "merge 6861", "queue sync", "queue drive", "once", "once --first 12", "deploy HEAD", "deploy main; id",
                f"queue baseline {SHA} --by infra", f"queue baseline {SHA}", "queue baseline main --by ci",
                f"queue baseline {SHA} --by ci --steps 'a b'", "queue eject 12 --by ci --reason '$(id)'",
                "queue eject 12 --by ci --reason --land", "queue eject 0 --by ci --reason x", f"queue accept-checked {SHA} --by ci --force"]:
        assert serve(gate, f"drive {bad}", "ci") == 2, bad
    assert gate.pod.ran == []


def test_the_control_pod_and_pods_outside_the_project_are_never_terminated(gate, capsys):
    assert serve(gate, f"pods-terminate {CONTROL_POD}") == 2 and "never the control pod" in capsys.readouterr().err
    assert serve(gate, f"pods-terminate {OTHER_POD}") == 2
    assert not [c for c in gate.pod.calls if c[0] == "DELETE"]
    assert serve(gate, "pods-terminate zzz999yyy88877") == 1 and "HTTP 404" in capsys.readouterr().err, "RunPod's answer, not a refusal"


def test_pods_create_takes_the_body_research_sends_under_a_live_budget_line_and_refuses_any_other(gate, capsys):
    body = create_body()
    assert serve(gate, "pods-create", stdin=json.dumps(body).encode()) == 0
    assert gate.pod.calls == [("POST", "/pods", body)] and json.loads(capsys.readouterr().out)["id"] == "new000new00000"
    for flags in ({"cpu": "cpu3m", "vcpu": 32}, {"gpu_count": 2, "min_vcpu": 8, "min_ram": 64, "data_center": "EU-RO-1", "expose_port": 8443}):
        assert serve(gate, "pods-create", stdin=json.dumps(create_body(**flags)).encode()) == 0, flags
    gate.pod.calls.clear()

    def refused(change, why):
        doc = json.loads(json.dumps(body))
        change(doc)
        assert serve(gate, "pods-create", stdin=json.dumps(doc).encode()) == 2, why
        assert why in capsys.readouterr().err

    refused(lambda b: b.update(name="vy-b-1-veritor-campaign"), "no budget line covers 'vy-b-1-veritor-campaign'")
    refused(lambda b: b["env"].update({gate.LEASE_ENV: str(int(NOW + 9 * 3600))}), "max_pod_hours")
    refused(lambda b: b["env"].update({gate.LEASE_ENV: str(int(NOW - 60))}), "already expired")
    refused(lambda b: b.update(networkVolumeId="4ue28s1c6g", volumeMountPath="/workspace"), "no network volume through the door")
    refused(lambda b: b.update(dockerStartCmd=["bash", "-c", "id"]), "no start command through the door")
    refused(lambda b: b["env"].update(RESEARCH_LEASE_SH="id"), "no start command through the door")
    refused(lambda b: b["env"].update(LD_PRELOAD="/tmp/x.so"), "the create body's env")
    refused(lambda b: b["env"].update(PUBLIC_KEY='command="id" ' + KEY_A), "the create body's env")
    refused(lambda b: b.update(name="vy-control-2-veritor-campaign"), "the control pod is not created from here")
    refused(lambda b: b.update(name="other-veritor-campaign"), "the create body's name")
    refused(lambda b: b.update(dockerArgs="id", templateId="x"), "keys the gate does not take: dockerArgs, templateId")
    refused(lambda b: b.update(gpuCount=True), "the create body's gpuCount")
    refused(lambda b: b.pop("env"), "the create body lacks env")
    for raw, why in ((b"not json", "is not JSON"), (b"[]", "not a JSON object"), (b" " * (gate.MAX_BODY + 1), "bytes")):
        assert serve(gate, "pods-create", stdin=raw) == 2 and why in capsys.readouterr().err
    stale = guard_state() | {"last_ok_poll": NOW - 3600}
    gate.STATE.write_text(json.dumps(stale))
    assert serve(gate, "pods-create", stdin=json.dumps(body).encode()) == 2 and "last completed a poll" in capsys.readouterr().err
    assert gate.pod.calls == [], "RunPod was asked for none of them"


def test_what_the_gate_loads_never_asks_the_door_again_and_a_failure_is_told_without_a_traceback(gate, monkeypatch, capsys):
    assert serve(gate, "pods-list") == 0 and budgets.guard_host() == "local"
    monkeypatch.setattr(runpod, "CONFIG", gate.STATE.parent / "absent-config.toml")
    with pytest.raises(runpod.PodError, match="^no RunPod API config at .*absent-config.toml and RUNPOD_API_KEY is not set$"):
        RUNPOD_REQUEST("GET", "/pods")          # a pod that lost its key says so: it does not ssh to itself

    def broken(method, path, body=None):
        raise ValueError("unexpected reply from /root/.runpod")

    monkeypatch.setattr(runpod, "_request", broken)
    capsys.readouterr()
    assert serve(gate, "pods-list") == 1
    assert capsys.readouterr().err == "vy-control-gate: ValueError: unexpected reply from /root/.runpod\n"


def test_a_key_that_is_not_root_reaches_the_gate_only_through_sudo_with_the_request_as_one_argument(gate, monkeypatch, capsys):
    ran = []
    monkeypatch.setattr(gate.os, "geteuid", lambda: 999)
    monkeypatch.setattr(gate.os, "execv", lambda path, argv: ran.append((path, argv)))
    monkeypatch.setenv("SSH_ORIGINAL_COMMAND", "drive status; id")
    gate.main(["ci"])
    monkeypatch.delenv("SSH_ORIGINAL_COMMAND")
    gate.main([])
    assert ran == [(gate.SUDO, [gate.SUDO, "-n", gate.GATE, "--ssh", "drive status; id", "--key", "ci"]),
                   (gate.SUDO, [gate.SUDO, "-n", gate.GATE, "--ssh", "", "--key", "-"])]
    monkeypatch.setenv("SSH_ORIGINAL_COMMAND", "x" * 5000)
    assert gate.main([]) == 2 and len(ran) == 2
    monkeypatch.setattr(gate.os, "geteuid", lambda: 0)
    monkeypatch.setenv("HOME", "/var/lib/vy-control-door")
    assert gate.main(["--ssh", "true", "--key", "-"]) == 0 and gate.main(["--ssh", "id", "--key", "-"]) == 2
    assert str(Path.home()) == "/root", "the code the gate loads reads root's files, whatever sudo kept"
    assert gate.main(["--ssh", "true"]) == 2 and len(ran) == 2
    sudoers = [ln for ln in (CONTROL / "door_setup.sh").read_text().splitlines() if "NOPASSWD" in ln]
    assert sudoers == ['  "research ALL=(root) NOPASSWD: $GATE --ssh *" > "$s.tmp"'], "the account's one rule is the gate"
    assert f"GATE={gate.GATE}\n" in (CONTROL / "door_setup.sh").read_text()


# ---- its setup ---------------------------------------------------------------------------------------------------------------------

def authorized_keys(tmp_path, gate_keys, admin_keys=None):
    (tmp_path / "gate-keys").write_text(gate_keys)
    argv = ["bash", str(CONTROL / "door_setup.sh"), "--authorized-keys", str(tmp_path / "gate-keys")]
    if admin_keys is not None:
        (tmp_path / "admin-keys").write_text(admin_keys)
        argv.append(str(tmp_path / "admin-keys"))
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def test_setup_forces_every_gated_key_to_the_gate_and_keeps_roots_keys_apart(tmp_path, gate):
    r = authorized_keys(tmp_path, f"# the door's keys\n- {KEY_A} every agent\n\nci {KEY_CI} ci@vm\n", f"{KEY_ADMIN} infra's lead\n")
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [f'restrict,command="{gate.GATE}" {KEY_A} every agent',
                                     f'restrict,command="{gate.GATE} ci" {KEY_CI} ci@vm', "--", f"{KEY_ADMIN} infra's lead"]
    for keys, why in ((f'- command="bash" {KEY_A}\n', "not LANE TYPE BASE64"), (f"- no-pty {KEY_A}\n", "not LANE TYPE BASE64"),
                      (f"{KEY_A}\n", "not LANE TYPE BASE64"), (f"ci;id {KEY_CI}\n", "not LANE TYPE BASE64"),
                      (f"root {KEY_A}\n", "root is an account here"), (f"research {KEY_A}\n", "research is an account here"),
                      (f"- {KEY_A}\nci {KEY_A} again\n", "one key on two lines"), ("# nothing\n", "holds no key"),
                      ("-----BEGIN OPENSSH " + "PRIVATE KEY-----\n", "a file of public keys")):
        r = authorized_keys(tmp_path, keys)
        assert r.returncode == 1 and why in r.stderr and "restrict" not in r.stdout, keys
    for admin, why in ((f"{KEY_A} the agents' key\n", "a key agents hold is not root's"), ("", "root would have no way in"),
                       (f'from="*" {KEY_ADMIN}\n', "not TYPE BASE64")):
        r = authorized_keys(tmp_path, f"- {KEY_A}\n", admin)
        assert r.returncode == 1 and why in r.stderr, admin


def test_the_door_is_made_again_after_a_reset_and_deploy_installs_its_two_files():
    boot = (CONTROL / "post_start.sh").read_text()
    assert "[ ! -s /data/boot/door/gate-keys ] || { /data/infra/vy-control-setup;" in boot and " sudo\n" in boot
    node = deploy.load()["vy-control-verity"]
    assert node.admin, "root there is the admin key's: the relays' drift, which holds the agents' key, passes it by"
    assert {e.path: e.src or e.by for e in node.entries} == {
        "/data/infra/vy-control-gate": "infra/control/vy-control-gate", "/data/infra/vy-control-setup": "infra/control/door_setup.sh",
        "/usr/local/sbin/vy-control-gate": "infra/control/door_setup.sh"}
    assert all(os.access(CONTROL / f, os.X_OK) for f in ("vy-control-gate", "door_setup.sh")), "installed with the repository's mode"


# ---- research through it -----------------------------------------------------------------------------------------------------------

@pytest.fixture
def lane_vm(gate, tmp_path, monkeypatch):
    """A lane's VM after the cut: the registry names the pod's gated account, no RunPod key and no Nebius key are here, and ssh to
    that account is the gate itself, as the key every agent holds (`-`)."""
    (tmp_path / "machines.toml").write_text('[machines.vy-control-verity]\nprovider = "ssh"\nhost = "control.invalid"\nport = 40022\n'
                                            'user = "research"\nproject = "verity"\n')
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    monkeypatch.delenv("RESEARCH_GUARD_HOST")
    monkeypatch.setattr(runpod, "CONFIG", tmp_path / "absent-config.toml")
    monkeypatch.setattr(runpod, "_request", RUNPOD_REQUEST)
    for k in remote.NEBIUS_KEY_ENV:
        monkeypatch.delenv(k, raising=False)
    sent = []

    def ssh(argv, capture_output=True, text=True, timeout=None, input=None):
        assert argv[-2] == "research@control.invalid" and argv[argv.index("-p") + 1] == "40022", "the gated account, never root"
        sent.append(argv[-1])
        out, err = io.StringIO(), io.StringIO()
        with monkeypatch.context() as pod, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            pod.setattr(runpod, "_request", gate.pod.api)
            pod.setenv("RESEARCH_GUARD_HOST", "")            # the gate's own process, which ends with the call
            rc = serve(gate, argv[-1], "-", stdin=(input or "").encode())
        return subprocess.CompletedProcess(argv, rc, out.getvalue(), err.getvalue())

    monkeypatch.setattr(door.subprocess, "run", ssh)
    return sent


def test_pods_create_passes_or_refuses_on_its_budget_through_the_door(lane_vm, gate):
    assert budgets.check_create("vy-a-1", 1) == (None, "vy-a-")
    why, line = budgets.check_create("vy-b-1", 1)
    assert "no budget line covers 'vy-b-1'" in why and "cannot read" not in why and line is None
    assert lane_vm == ["budgets-state", "budgets-state"]


def test_a_stopped_node_is_woken_through_the_door(lane_vm, gate, monkeypatch):
    monkeypatch.setattr(gate, "run", lambda argv, env: gate.pod.ran.append(argv) or print(json.dumps({"was": "STOPPED"})) or 0)
    assert remote.wake_on_control(IID) == "STOPPED" and lane_vm == [f"wake {IID}"]
    assert gate.pod.ran == [[gate.PYTHON, "-m", "research.pods.nebius.wake", "start", IID]]


def test_with_no_runpod_key_pods_are_listed_created_and_terminated_through_the_door_but_never_the_control_pod(lane_vm, gate):
    assert [p["id"] for p in runpod._request("GET", "/pods")] == [CONTROL_POD, LANE_POD, OTHER_POD]
    assert runpod.get(LANE_POD)["name"] == "vy-a-1-veritor-campaign"
    body = create_body()
    assert runpod._request("POST", "/pods", body)["id"] == "new000new00000" and gate.pod.calls[-1] == ("POST", "/pods", body)
    runpod.terminate(LANE_POD)
    assert gate.pod.calls[-1] == ("DELETE", f"/pods/{LANE_POD}", None)
    with pytest.raises(runpod.PodError, match="through vy-control-verity's door: vy-control-gate: refused: .*never the control pod"):
        runpod.terminate(CONTROL_POD)
    with pytest.raises(runpod.PodError, match="HTTP 404") as gone:
        runpod.get("zzz999yyy88877")
    assert gone.value.code == 404 and gone.value.pod_id == "zzz999yyy88877", "a caller still tells a pod that is gone"
    assert lane_vm == ["pods-list", f"pods-get {LANE_POD}", "pods-create", f"pods-terminate {LANE_POD}",
                       f"pods-terminate {CONTROL_POD}", "pods-get zzz999yyy88877"]


def test_a_shell_command_from_a_lanes_vm_is_refused_by_the_gate(lane_vm, gate):
    from research.pods import connect

    for command in ("id", "cat /data/state/config-verity/r2.env", "/data/infra/drive.sh queue baseline main --by ci"):
        r = door.subprocess.run([*connect.ssh_base("vy-control-verity"), command], capture_output=True, text=True)
        assert r.returncode == 2 and r.stderr.startswith("vy-control-gate: refused: "), command
    assert gate.pod.ran == [] and gate.pod.calls == []
