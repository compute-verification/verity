"""`research` tells an agent, once a session, that no shell hook runs on its VM (`research.guard_notice`). The agent guard's
`inbox.py guard` decides that and words the line; here a stand-in answers for it, so nothing is started and nothing waits."""
from __future__ import annotations

import io
import os
import subprocess
import sys

from research import cli, guard_notice

A = "bc-12345678-1234-4234-8234-123456789abc"
LINE = f"no shell hook has run for {A} on this VM, so the guard and the inbox are not running here. Why. The fix"


class Guard:
    """`inbox.py guard`, answering with each of `answers` in turn: an exit code and its output, or an exception to raise."""

    def __init__(self, tmp_path, *answers):
        self.tool, self.marks, self.answers, self.calls, self.out = tmp_path / "inbox.py", tmp_path / "marks", list(answers), [], io.StringIO()
        self.tool.touch()

    def run(self, argv, **kw):
        self.calls.append((argv, kw))
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        code, stdout, stderr = answer
        return subprocess.CompletedProcess(argv, code, stdout, stderr)

    def notice(self, **env):
        return guard_notice.notice(env, tool=self.tool, marks=self.marks, run=self.run, out=self.out)


def test_an_agent_is_told_once_a_session_that_no_shell_hook_runs_on_its_vm(tmp_path):
    """A VM whose branch's hooks.json Cursor couldn't load ran every command with no guard and no inbox, and said nothing.
    The agent's first `research` command there prints the guard's line; its later ones ask nothing and print nothing."""
    other = "bc-87654321-4321-4321-8321-cba987654321"
    guard = Guard(tmp_path, (1, LINE + "\n", ""), (1, LINE.replace(A, other) + "\n", "inbox: a note of its own\n"))
    assert guard.notice(CURSOR_CONVERSATION_ID=A, HOME="/h") == f"research: {LINE}"
    assert guard.out.getvalue() == f"research: {LINE}\n"
    (argv, kw), = guard.calls
    assert argv == [sys.executable, str(guard.tool), "guard", A, "--repo", "."]
    assert kw["env"] == {"CURSOR_CONVERSATION_ID": A, "HOME": "/h"} and kw["timeout"] == guard_notice.ASK_TIMEOUT_S
    for _ in range(3):
        assert guard.notice(CURSOR_CONVERSATION_ID=A) is None
    assert len(guard.calls) == 1 and guard.out.getvalue().count("\n") == 1, "once a session"
    # another agent on the VM is another session, and the agent VM's `research` names the checkout
    assert guard.notice(CURSOR_CONVERSATION_ID=other, RESEARCH_CHECKOUT="/agent/repos/verity") == f"research: {LINE.replace(A, other)}"
    assert guard.calls[1][0][-3:] == [other, "--repo", "/agent/repos/verity"]


def test_a_vm_whose_shell_hook_runs_is_asked_once_and_told_nothing(tmp_path):
    guard = Guard(tmp_path, (0, "", ""))
    assert guard.notice(CURSOR_CONVERSATION_ID=A) is None and guard.notice(CURSOR_CONVERSATION_ID=A) is None
    assert len(guard.calls) == 1 and guard.out.getvalue() == ""
    assert (guard.marks / A).exists()


def test_only_an_agent_is_asked(tmp_path):
    """A person, a pod and a node have no conversation id, and an id the hook would never deliver to has no answer."""
    guard = Guard(tmp_path)
    assert guard.notice() is None
    assert guard.notice(CURSOR_CONVERSATION_ID="") is None
    assert guard.notice(CURSOR_CONVERSATION_ID="../../etc/passwd") is None
    assert guard.notice(CURSOR_CONVERSATION_ID=A[:11]) is None
    guard.tool.unlink()
    assert guard.notice(CURSOR_CONVERSATION_ID=A) is None, "a tree without the guard's tool"
    assert guard.calls == [] and guard.out.getvalue() == "" and not guard.marks.exists()


def test_an_answer_that_could_not_be_had_is_said_and_the_next_command_asks_again(tmp_path):
    """A failed ask must not read as a guard that runs: it is said, nothing is marked, and the next command asks again."""
    guard = Guard(tmp_path, subprocess.TimeoutExpired("inbox.py", 20), OSError("no such interpreter"),
                  (2, "", "inbox.py: error: argument cmd: invalid choice: 'guard'\n"), (1, "", "inbox: 'x' is no full agent id\n"),
                  (0, "something on stdout\n", ""), (1, LINE + "\n", ""))
    said = [guard.notice(CURSOR_CONVERSATION_ID=A) for _ in range(5)]
    assert all(s.startswith("research: could not ask whether the agent guard runs on this VM") for s in said), said
    assert "timed out" in said[0] and "no such interpreter" in said[1] and "exited 2: inbox.py: error" in said[2]
    assert "exited 1: inbox: 'x' is no full agent id" in said[3] and "exited 0: something on stdout" in said[4]
    assert not (guard.marks / A).exists()
    assert guard.notice(CURSOR_CONVERSATION_ID=A) == f"research: {LINE}" and (guard.marks / A).exists()
    assert guard.notice(CURSOR_CONVERSATION_ID=A) is None and len(guard.calls) == 6


def test_research_asks_before_a_command_but_not_for_its_version_and_never_under_a_test(monkeypatch, capsys):
    """A test's `research`, and each `python -m research` it starts, reads only what its suite declares and says nothing of
    the VM the test happens to run on; outside a test every command asks first, with the process's whole environment."""
    events = []
    monkeypatch.setattr(guard_notice, "notice", lambda env: events.append(("asked", env is os.environ)))
    monkeypatch.setattr(cli, "cmd_status", lambda rest: events.append(("ran", rest)) or 0)
    assert cli.main(["status"]) == 0 and events == [("ran", [])]
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    monkeypatch.delenv("CLOUD_AGENT_INJECTED_SECRET_NAMES", raising=False)      # no secret to take from a tmux server
    assert cli.main(["--version"]) == 0 and events == [("ran", [])]
    assert cli.main(["status", "--json"]) == 0
    assert events[1:] == [("asked", True), ("ran", ["--json"])], "asked first, with the process's own environment"
    capsys.readouterr()
