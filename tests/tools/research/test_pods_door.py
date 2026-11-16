"""pods/door.py, this side of the control pod's door: a verb to the gated account once the machine registry names it, the shell
command root took before where the pod has no door yet (said in one line), and RunPod asked through the door only on a machine
that holds no RunPod key. No ssh: the transport is a function that records what it was sent. The gate's side, and both together,
are infra/ops's tests/test_control_door.py."""
from __future__ import annotations

import json
import subprocess

import pytest

from research import remote
from research.pods import budgets, door, runpod

IID = "computeinstance-e00fakeinstance0id"
STATE = {"last_ok_poll": 100.0, "budgets": {"lines": {}}}


@pytest.fixture
def sent(monkeypatch, tmp_path):
    """What ssh was asked to run, each call answered as the pod's gate answers `budgets-state` and `wake`."""
    calls = []

    def ssh(argv, capture_output=True, text=True, timeout=None, **kw):
        calls.append((argv[:-1], argv[-1], kw))
        out = json.dumps({"was": "STOPPED"}) if "wake" in argv[-1] else f"{json.dumps(STATE)}\n130"
        return subprocess.CompletedProcess(argv, 0, out + "\n", "")

    monkeypatch.setattr(door.subprocess, "run", ssh)
    monkeypatch.setattr(door, "_told", set())
    monkeypatch.setattr(runpod, "CONFIG", tmp_path / "absent-config.toml")
    for k in remote.NEBIUS_KEY_ENV:
        monkeypatch.delenv(k, raising=False)
    return calls


@pytest.fixture
def no_door(monkeypatch):
    """Before the window: the registry has no ssh entry for the pod, which is reached as root after a RunPod lookup."""
    monkeypatch.setattr("research.pods.connect.ssh_base", lambda host: ["ssh", f"root@{host}"])


@pytest.fixture
def the_door(monkeypatch, tmp_path):
    (tmp_path / "machines.toml").write_text('[machines.vy-control-verity]\nprovider = "ssh"\nhost = "control.invalid"\nport = 40022\n'
                                            'user = "research"\nproject = "verity"\n')
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))


def test_a_pod_without_its_door_is_asked_as_before_and_one_line_says_so(sent, no_door, capsys):
    assert budgets.read_guard_state() == (STATE, 130.0)
    assert remote.wake_on_control(IID) == "STOPPED"
    assert [(a, c) for a, c, _ in sent] == [(["ssh", "root@vy-control-verity"], f"cat {budgets.GUARD_STATE} && echo && date +%s"),
                                           (["ssh", "root@vy-control-verity"], f"{remote.CONTROL_WAKE} {IID}")]
    assert capsys.readouterr().err.splitlines() == ["research: vy-control-verity has no door yet (the machine registry has no ssh "
                                                    "entry for it): asking its root shell, as before"]


def test_a_pod_with_its_door_is_sent_verbs_at_its_gated_account(sent, the_door, capsys):
    assert budgets.read_guard_state() == (STATE, 130.0)
    assert remote.wake_on_control(IID) == "STOPPED"
    assert [c for _, c, _ in sent] == ["budgets-state", f"wake {IID}"]
    for argv, _, _ in sent:
        assert argv[-1] == "research@control.invalid" and argv[argv.index("-p") + 1] == "40022"
    assert capsys.readouterr().err == ""


def test_a_checkout_without_the_entry_is_told_to_sync_once_root_refuses_it(sent, no_door, monkeypatch):
    denied = lambda argv, **kw: subprocess.CompletedProcess(argv, 255, "", "root@h: Permission denied (publickey).\n")  # noqa: E731
    monkeypatch.setattr(door.subprocess, "run", denied)
    why, _ = budgets.check_create("vy-a-1", 1)
    assert "Permission denied" in why and "may have its door now: `research notes sync` brings its registry entry" in why
    with pytest.raises(remote.Refused, match="research notes sync"):
        remote.wake_on_control(IID)


def test_runpod_is_asked_through_the_door_only_where_no_key_is_held(sent, the_door, monkeypatch, tmp_path):
    answers = iter([(0, '[{"id": "abc123def45678"}]', ""), (0, '{"id": "abc123def45678"}', ""), (0, '{"id": "new000new00000"}', ""),
                    (0, "null\n", ""), (2, "", "vy-control-gate: refused: only the project's pods\n"),
                    (1, "", "vy-control-gate: GET /pods/zzz999yyy88877 -> HTTP 404: not found (terminated, or never existed)\n")])
    seen = []

    def ssh(argv, capture_output=True, text=True, timeout=None, **kw):
        seen.append((argv[-1], kw))
        rc, out, err = next(answers)
        return subprocess.CompletedProcess(argv, rc, out, err)

    monkeypatch.setattr(door.subprocess, "run", ssh)
    assert runpod._request("GET", "/pods") == [{"id": "abc123def45678"}]
    assert runpod.get("abc123def45678") == {"id": "abc123def45678"}
    assert runpod._request("POST", "/pods", {"name": "vy-a-1-veritor-campaign"}) == {"id": "new000new00000"}
    assert runpod._request("DELETE", "/pods/abc123def45678") is None
    with pytest.raises(runpod.PodError, match="DELETE /pods/p1ugdh9hfgqe5d through vy-control-verity's door: vy-control-gate: refused"):
        runpod._request("DELETE", "/pods/p1ugdh9hfgqe5d")
    with pytest.raises(runpod.PodError) as gone:
        runpod.get("zzz999yyy88877")
    assert gone.value.code == 404 and gone.value.pod_id == "zzz999yyy88877"
    assert seen == [("pods-list", {}), ("pods-get abc123def45678", {}), ("pods-create", {"input": '{"name": "vy-a-1-veritor-campaign"}'}),
                    ("pods-terminate abc123def45678", {}), ("pods-terminate p1ugdh9hfgqe5d", {}), ("pods-get zzz999yyy88877", {})]
    with pytest.raises(runpod.PodError, match="no RunPod API config .*, and vy-control-verity's door has no verb for PATCH /pods/x"):
        runpod._request("PATCH", "/pods/x", {})
    assert len(seen) == 6

    (tmp_path / "config.toml").write_text('apikey = "' + "dummy" + '-key"\napiurl = "https://api.invalid/graphql"\n')
    monkeypatch.setattr(runpod, "CONFIG", tmp_path / "config.toml")
    monkeypatch.setattr(runpod.urllib.request, "urlopen", lambda request, timeout: pytest.fail("no network in a test"))
    monkeypatch.setattr(door, "api", lambda *a, **kw: pytest.fail("a machine that holds the key asks RunPod itself"))
    with pytest.raises(BaseException, match="no network in a test"):
        runpod._request("GET", "/pods")


def test_with_neither_a_key_nor_a_door_the_refusal_names_both(sent, no_door, monkeypatch):
    with pytest.raises(runpod.PodError, match="no RunPod API config at .* and RUNPOD_API_KEY is not set, and vy-control-verity has no "
                                              "door to ask"):
        runpod._request("GET", "/pods")
    monkeypatch.setenv("RESEARCH_GUARD_HOST", "local")
    with pytest.raises(runpod.PodError, match="RUNPOD_API_KEY is not set$"):
        runpod._request("GET", "/pods")
    assert sent == []
