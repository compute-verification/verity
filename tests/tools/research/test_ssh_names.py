"""ssh's stderr names machines, never their addresses: `remote.name_addresses`, which `ssh_run` and `--on auto`'s reading of a
machine's share put every error through.  Every address here is a documentation one."""
from __future__ import annotations

import subprocess

import pytest

from research import auto, remote

DENIED = "Permission denied (publickey)."
NAMES = {"192.0.2.7": "vy-nebius-2", "2001:db8::7": "vy-six", "box.example.invalid": "vy-box"}


@pytest.mark.parametrize("said, named", [
    (f"research@192.0.2.7: {DENIED}", f"vy-nebius-2: {DENIED}"),
    (f"research@2001:db8::7: {DENIED}", f"vy-six: {DENIED}"),
    (f"research@box.example.invalid: {DENIED}", f"vy-box: {DENIED}"),
    ("ssh: connect to host 192.0.2.7 port 22: Connection timed out", "ssh: connect to host vy-nebius-2 port 22: Connection timed out"),
    ("Connection closed by [2001:db8::7]:22 and 192.0.2.7:22", "Connection closed by [vy-six]:22 and vy-nebius-2:22"),
    (f"root@192.0.2.99: {DENIED}", f"<redacted>: {DENIED}"),
    ("Connection closed by 2001:db8::99 port 22", "Connection closed by <redacted> port 22"),
])
def test_an_address_and_its_user_are_given_as_the_machines_name(said, named):
    assert remote.name_addresses(said, NAMES) == named


def test_text_with_no_address_is_unchanged():
    said = "exit 1: python 3.11.10 at 12:30:45 said: no module named slot (see 1.2)."
    assert remote.name_addresses(said, NAMES) == said


def test_an_unregistered_address_is_given_as_the_machine_ssh_was_asked_to_reach():
    assert remote.name_addresses(f"research@198.51.100.4: {DENIED}", {}, machine="vy-pod") == f"vy-pod: {DENIED}"


def test_ssh_run_gives_its_stderr_with_names(monkeypatch):
    monkeypatch.setattr(remote, "_registered_names", lambda: dict(NAMES))
    r = remote.ssh_run(["sh", "-c"], ["sh", "-c", f"echo 'research@192.0.2.7: {DENIED}' >&2; exit 255"])
    assert (r.returncode, r.stderr.decode()) == (255, f"vy-nebius-2: {DENIED}\n")


def test_on_autos_placement_says_why_a_share_was_unread_by_the_machines_name(tmp_path, monkeypatch):
    """Rule: the share's error, which a run's request.json and the stranded report keep, names the machine; weakening: ssh's
    raw `user@ADDRESS`."""
    monkeypatch.setattr(remote, "_registered_names", lambda: {})
    monkeypatch.setattr(auto, "spill", lambda: ())
    (tmp_path / auto.SLOT_PY).parent.mkdir(parents=True)
    (tmp_path / auto.SLOT_PY).write_text("")

    def run(argv, payload):
        return subprocess.CompletedProcess(argv, 255, b"", f"research@192.0.2.7: {DENIED}\n".encode())

    docs = auto.read(tmp_path, 4, reach=lambda name: [name], run=run)
    assert auto.off(docs[auto.AUTO_NODE]) == f"{auto.AUTO_NODE}'s share wasn't read (exit 255: {auto.AUTO_NODE}: {DENIED})"
