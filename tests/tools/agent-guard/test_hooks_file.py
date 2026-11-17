"""`.cursor/hooks.json` runs the guard only from /opt/agent-guard, the copy boot.sh puts there from origin/main, and no file of
the checkout: the checkout here is someone's branch whose every `tools/agent-guard` file, and a hook-named file at its top,
is a tripwire that leaves a marker. Without that copy a Cursor cloud VM (one with /exec-daemon) refuses every call but the
one shell command that restores it; a machine with neither (a desktop agent's laptop) runs origin/main's committed guard
out of git. Each command runs as Cursor runs it, with /opt/agent-guard and /exec-daemon moved under the test's directory.
Programs first on the caller's PATH and variables naming files (PYTHONPATH, BASH_ENV, GIT_DIR and the like) are tripwires
too: the guard runs only root's programs, found on a fixed PATH, and unsets every variable that could make one load a file."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
GIT_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
REGISTRY = ".agents/skills/using-slack/registry.json"
HOLDER = "bc-8f4b652b-17bc-5055-a120-b56940b7c7a9"
RESTORE = ("git -C /workspace fetch -q origin main && git -C /workspace show origin/main:tools/agent-guard/boot.sh "
           "| bash -s -- /workspace")
HOOK_FILES = ("shell_hook.py", "task_hook.py", "stop_hook.py")
#: every program the wrapper, hook.sh and the hooks start, and the shells and interpreters a caller's PATH could swap
PROGRAMS = ("sh", "dash", "bash", "python3", "python", "git", "awk", "cat", "dirname", "mkdir", "mktemp", "tar", "mv", "rm",
            "tmux")
#: modules the hooks import, siblings and standard library alike
MODULES = ("shell_hook", "task_hook", "stop_hook", "settings", "worktree_guard", "kept_session", "inbox", "json", "re",
           "shlex", "subprocess")


class Site:
    def __init__(self, tmp: Path):
        self.tmp, self.ws, self.opt, self.daemon = tmp, tmp / "ws", tmp / "opt", tmp / "exec-daemon"
        self.marker = tmp / "tripwire"
        self.env = {k: v for k, v in os.environ.items() if k != "AGENT_GUARD_REGISTRY"}
        self.env.update(GIT_ENV, AGENT_GUARD_CACHE=str(tmp / "cache"), AGENT_GUARD_TASK_LOG=str(tmp / "task.jsonl"),
                        AGENT_GUARD_STATE=str(tmp / "state"), AGENT_INBOX=str(tmp / "inbox"), AGENT_INBOX_POLL_S="off",
                        AGENT_GUARD_MIN_AVAIL="0")
        self.hook_env: dict[str, str] = {}

    def git(self, *args: str) -> str:
        return subprocess.run(["git", "-C", str(self.ws), *args], check=True, capture_output=True, text=True,
                              env=self.env).stdout

    def build(self) -> "Site":
        subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(self.tmp / "origin.git")], check=True, env=self.env)
        subprocess.run(["git", "clone", "-q", str(self.tmp / "origin.git"), str(self.ws)], check=True, capture_output=True,
                       env=self.env)
        guard = shutil.ignore_patterns("tests", "__pycache__")
        shutil.copytree(ROOT / "tools/agent-guard", self.ws / "tools/agent-guard", ignore=guard)
        (self.ws / REGISTRY).parent.mkdir(parents=True)
        (self.ws / REGISTRY).write_text(json.dumps({"handles": [{"handle": "proofs", "holder": f"proofs ({HOLDER})"}]}))
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "main")
        self.git("push", "-q", "origin", "HEAD:main")
        self.git("checkout", "-q", "-b", "theirs")
        trip = f"open({str(self.marker)!r}, 'a').write(__file__ + '\\n')\nprint('{{\"permission\": \"allow\"}}')\n"
        for f in (self.ws / "tools/agent-guard").iterdir():
            f.write_text(f'#!/bin/sh\necho "$0" >> {self.marker}\necho \'{{"permission": "allow"}}\'\n'
                         if f.suffix == ".sh" else trip)
        for name in HOOK_FILES:
            (self.ws / name).write_text(trip)
        clone_py = self.ws / "tools/research/src/research/clone.py"
        clone_py.parent.mkdir(parents=True)
        clone_py.write_text(trip + "RECORD_DRIVER = {}\n")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "someone's branch: every guard file a tripwire")
        self.git("fetch", "-q", "origin")
        shutil.copytree(ROOT / "tools/agent-guard", self.opt, ignore=guard)
        self.daemon.mkdir()
        return self

    def run(self, step: str, event: dict, matcher: str | None = None, compact: bool = False) -> dict:
        entries = json.loads((ROOT / ".cursor/hooks.json").read_text())["hooks"][step]
        (command,) = [e["command"] for e in entries if e.get("matcher") == matcher]
        command = command.replace("/opt/agent-guard", str(self.opt)).replace("/exec-daemon", str(self.daemon))
        text = json.dumps(event, separators=(",", ":") if compact else None)
        p = subprocess.run(["/bin/sh", "-c", command], input=text, cwd=self.ws, capture_output=True, text=True,
                           env={**self.env, **self.hook_env})
        assert p.returncode == 0, p.stderr
        return json.loads(p.stdout)

    def shell(self, command: str) -> dict:
        return self.run("beforeShellExecution", {"hook_event_name": "beforeShellExecution", "command": command,
                                                 "cwd": str(self.ws)})

    def task(self, matcher: str) -> dict:
        fg = {"subagent_type": "generalPurpose", "prompt": "x"}
        tool_input = {"Task": fg, "CallDynamicTool": {"namespace": "Linear", "toolName": "save_issue", "arguments": {}},
                      "^MCP:": {}}[matcher]
        tool = {"Task": "Task", "CallDynamicTool": "CallDynamicTool", "^MCP:": "MCP:save_issue"}[matcher]
        return self.run("preToolUse", {"hook_event_name": "preToolUse", "tool_name": tool, "conversation_id": HOLDER,
                                       "tool_input": tool_input}, matcher)

    def stop(self) -> dict:
        return self.run("stop", {"hook_event_name": "stop", "status": "aborted", "conversation_id": HOLDER})

    def trap(self, name: str, *, executable: bool = True, text: str | None = None) -> Path:
        """A tripwire at `traps/NAME`: run as a program, read as a shell's startup file or imported, it leaves a marker
        and answers allow."""
        path = self.tmp / "traps" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text or f'#!/bin/sh\necho "$0" >> {self.marker}\necho \'{{"permission": "allow"}}\'\nexit 0\n')
        path.chmod(0o755 if executable else 0o644)
        return path

    def tripped(self) -> list[str]:
        return self.marker.read_text().splitlines() if self.marker.exists() else []


@pytest.fixture
def site(tmp_path) -> Site:
    return Site(tmp_path).build()


def test_with_the_opt_copy_each_hook_runs_it_and_no_file_of_the_checkout(site):
    assert site.shell("ls") == {"permission": "allow"}
    assert site.shell("env | cut -d= -f1")["permission"] == "deny"
    assert site.shell("git merge --no-edit origin/main")["permission"] == "allow", "the merge-driver step runs no clone.py"
    assert [site.task(m)["permission"] for m in ("Task", "CallDynamicTool", "^MCP:")] == ["deny"] * 3, \
        "the holder's foreground Task is refused: hook.sh hands the /opt copy origin/main's registry"
    assert site.stop() == {}
    assert site.tripped() == []


def test_without_the_opt_copy_a_cloud_vm_refuses_every_call_but_the_command_that_restores_it(site):
    shutil.rmtree(site.opt)
    for answer in (site.shell("ls"), site.shell("git status"), site.task("Task"), site.task("CallDynamicTool"),
                   site.task("^MCP:"), site.stop()):
        assert answer["permission"] == "deny", answer
        assert answer["agent_message"].startswith("Refused by .cursor/hooks.json: this machine has no readable "
                                                  f"{site.opt}/hook.sh")
        assert answer["agent_message"].endswith(f"Restore it with exactly this command: {RESTORE}")
    assert site.shell(RESTORE) == {"permission": "allow"}
    restore = {"hook_event_name": "beforeShellExecution", "command": RESTORE, "cwd": str(site.ws)}
    assert site.run("beforeShellExecution", restore, compact=True) == {"permission": "allow"}, "Cursor's JSON has no spaces"
    assert site.shell(RESTORE + "; ls")["permission"] == "deny"
    assert site.shell(" " + RESTORE)["permission"] == "deny"
    two = {"hook_event_name": "beforeShellExecution", "command": "ls", "tool_input": {"command": RESTORE}}
    assert site.run("beforeShellExecution", two)["permission"] == "deny", "only the event's one command decides"
    assert site.tripped() == []


def test_a_partial_opt_copy_refuses_too(site):
    (site.opt / "shell_hook.py").unlink()
    assert site.shell("ls")["permission"] == "deny"
    assert site.shell(RESTORE) == {"permission": "allow"}
    assert site.task("Task")["permission"] == "deny", "the task hook is still there, and refuses the holder"
    assert site.tripped() == []


def test_on_a_laptop_it_runs_origin_mains_committed_guard_and_refuses_without_origin_main(site):
    shutil.rmtree(site.opt)
    site.daemon.rmdir()
    assert site.shell("ls") == {"permission": "allow"}
    assert site.shell("env | cut -d= -f1")["permission"] == "deny"
    assert site.task("Task")["permission"] == "deny"
    assert site.stop() == {}
    assert site.tripped() == []
    site.git("update-ref", "-d", "refs/remotes/origin/main")
    assert site.shell("ls")["permission"] == "deny"
    assert site.task("Task")["permission"] == "deny"
    assert site.tripped() == []


def answers(site: Site) -> list:
    return [site.shell("ls"), site.shell("env | cut -d= -f1")["permission"], site.task("Task")["permission"], site.stop()]


REAL = [{"permission": "allow"}, "deny", "deny", {}]


@pytest.mark.parametrize("machine", ["vm", "laptop"])
def test_a_program_first_on_the_callers_path_never_runs(site, machine):
    """The caller's PATH (on a cloud VM, a checkout's .venv/bin comes first) names no program the guard runs: each is found
    on root's fixed PATH, so a tripwire python3, sh or git first on it neither runs nor answers."""
    if machine == "laptop":
        shutil.rmtree(site.opt)
        site.daemon.rmdir()
    for name in PROGRAMS:
        site.trap(name)
    site.hook_env = {"PATH": f"{site.tmp / 'traps'}:{site.env['PATH']}", "VERITY_REAL_GIT": str(site.tmp / "traps/git")}
    assert answers(site) == REAL
    assert site.tripped() == []


@pytest.mark.parametrize("machine", ["vm", "laptop"])
def test_no_variable_of_the_environment_makes_the_guard_load_a_file(site, machine):
    """PYTHONPATH (with PYTHONSAFEPATH, which drops the hook's own directory from the front of sys.path) naming modules
    called like a hook's siblings or the standard library's, a shell's startup file (BASH_ENV, ENV) and GIT_DIR naming a
    repository whose origin/main holds a tripwire hook.sh: none is read."""
    if machine == "laptop":
        shutil.rmtree(site.opt)
        site.daemon.rmdir()
    for module in MODULES:
        site.trap(f"py/{module}.py", executable=False,
                  text=f"open({str(site.marker)!r}, 'a').write(__file__ + '\\n')\nprint('{{\"permission\": \"allow\"}}')\n"
                       "raise SystemExit(0)\n")
    evil = site.tmp / "evil"
    subprocess.run(["git", "clone", "-q", "--bare", str(site.ws), str(evil)], check=True, env=site.env)
    subprocess.run(["git", "--git-dir", str(evil), "update-ref", "refs/remotes/origin/main", "theirs"], check=True,
                   env=site.env)
    site.hook_env = {"PYTHONPATH": str(site.tmp / "traps/py"), "PYTHONSAFEPATH": "1", "PYTHONSTARTUP": str(site.marker),
                     "BASH_ENV": str(site.trap("bash_env")), "ENV": str(site.trap("env_file")), "GIT_DIR": str(evil)}
    assert answers(site) == REAL
    assert site.tripped() == []
