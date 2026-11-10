"""`tools/agent-guard/boot.sh` puts the shell hook on a VM whose branch predates it: cloud agents read project hooks only, so
a VM on a branch without `.cursor/hooks.json` would run no hook."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BOOT = ROOT / "tools/agent-guard/boot.sh"
GIT_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
sys.path.insert(0, str(BOOT.parent))
import inbox  # noqa: E402

sys.path.remove(str(BOOT.parent))


def cursor_reads(path: Path) -> dict:
    """A hooks.json boot wrote, as Cursor reads it (comments stripped), which must be what its JSON says."""
    text = path.read_text()
    assert inbox.as_cursor_reads(text) == json.loads(text), f"Cursor strips a `/*` or `//` in {path} as a comment"
    return json.loads(text)


def test_boot_installs_the_hook_on_an_old_branch_and_steps_aside_for_a_tracked_one(tmp_path):
    env = {**os.environ, **GIT_ENV, "AGENT_GUARD_DEST": str(tmp_path / "opt"), "AGENT_GUARD_SKIP_INSTALL": "1",
           "AGENT_GUARD_BASHRC": str(tmp_path / "bashrc"), "AGENT_GUARD_ROOTS": str(tmp_path / "agent")}

    def git(*a, cwd=tmp_path / "ws"):
        return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True, env=env).stdout

    def boot():
        return subprocess.run(["bash", str(BOOT), str(tmp_path / "ws")], capture_output=True, text=True, env=env)

    git("init", "-q", "--bare", "-b", "main", str(tmp_path / "origin.git"), cwd=tmp_path)
    git("clone", "-q", str(tmp_path / "origin.git"), str(tmp_path / "ws"), cwd=tmp_path)
    (tmp_path / "ws" / "a").write_text("old\n")
    git("add", "a")
    git("commit", "-q", "-m", "old")
    git("push", "-q", "origin", "HEAD:old")
    shutil.copytree(ROOT / "tools/agent-guard", tmp_path / "ws/tools/agent-guard",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / ".cursor", tmp_path / "ws/.cursor")
    git("add", "tools", ".cursor")
    git("commit", "-q", "-m", "guard")
    git("push", "-q", "origin", "HEAD:main")
    git("checkout", "-q", "old")
    assert not (tmp_path / "ws/.cursor").exists()

    p = boot()
    assert p.returncode == 0 and "wrote" in p.stdout, (p.stdout, p.stderr)
    hooks = cursor_reads(tmp_path / "ws/.cursor/hooks.json")["hooks"]
    (entry,) = hooks["beforeShellExecution"]
    assert entry == {"command": f"python3 {tmp_path / 'opt'}/shell_hook.py", "timeout": 10, "failClosed": True}
    assert hooks["stop"] == [{"command": f"python3 {tmp_path / 'opt'}/stop_hook.py", "timeout": 60}]
    assert git("status", "--porcelain") == ""
    hook = subprocess.run(entry["command"].split(), input=json.dumps({"hook_event_name": "beforeShellExecution",
                                                                      "command": "env | cut -d= -f1", "cwd": "/"}),
                          capture_output=True, text=True, check=True)
    assert json.loads(hook.stdout)["permission"] == "deny"

    git("remote", "set-url", "origin", str(tmp_path / "gone.git"))
    p = boot()
    assert p.returncode == 0 and "using the copy already in" in p.stderr, (p.stdout, p.stderr)

    git("remote", "set-url", "origin", str(tmp_path / "origin.git"))
    git("checkout", "-q", "main")
    assert json.loads((tmp_path / "ws/.cursor/hooks.json").read_text()) == json.loads((ROOT / ".cursor/hooks.json").read_text())
    p = boot()
    assert p.returncode == 0 and "tracks .cursor/hooks.json" in p.stdout, (p.stdout, p.stderr)
    assert git("status", "--porcelain") == ""
    path_lines = [ln for ln in (tmp_path / "bashrc").read_text().splitlines() if "/workspace/.venv/bin" in ln]
    assert path_lines == ["export PATH=/workspace/.git/verity-auth/bin:/workspace/.venv/bin:$PATH"], "research on PATH, once"


def test_boot_refuses_without_origin_or_an_earlier_copy(tmp_path):
    env = {**os.environ, **GIT_ENV, "AGENT_GUARD_DEST": str(tmp_path / "opt"), "AGENT_GUARD_SKIP_INSTALL": "1",
           "AGENT_GUARD_BASHRC": str(tmp_path / "bashrc"), "AGENT_GUARD_ROOTS": str(tmp_path / "agent")}
    subprocess.run(["git", "init", "-q", str(tmp_path / "ws")], check=True, env=env)
    p = subprocess.run(["bash", str(BOOT), str(tmp_path / "ws")], capture_output=True, text=True, env=env)
    assert p.returncode == 1 and "no guard on this VM" in p.stderr, p.stderr
    (entry,) = json.loads((tmp_path / "ws/.cursor/hooks.json").read_text())["hooks"]["beforeShellExecution"]
    assert entry["failClosed"] is True and not (tmp_path / "opt/shell_hook.py").exists()


def test_boot_puts_the_hook_at_a_task_launched_agents_root_above_its_checkout(tmp_path):
    """A Task-launched cloud agent's workspace root is /agent with the repository under /agent/repos/, and Cursor reads hooks
    from the root only: Task-launched workers ran no hook, so no inbox message and no secret refusal reached them. The
    root's hooks.json runs the checkout's hook.sh, or DEST's copy while there is no checkout, and a root that is itself a
    checkout gets none."""
    env = {**os.environ, **GIT_ENV, "AGENT_GUARD_DEST": str(tmp_path / "opt"), "AGENT_GUARD_SKIP_INSTALL": "1",
           "AGENT_GUARD_BASHRC": str(tmp_path / "bashrc"),
           "AGENT_GUARD_ROOTS": f"{tmp_path / 'agent'} {tmp_path / 'ws'} {tmp_path / 'absent'}"}

    def git(*a, cwd=tmp_path / "ws"):
        return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True, env=env).stdout

    git("init", "-q", "--bare", "-b", "main", str(tmp_path / "origin.git"), cwd=tmp_path)
    git("clone", "-q", str(tmp_path / "origin.git"), str(tmp_path / "ws"), cwd=tmp_path)
    shutil.copytree(ROOT / "tools/agent-guard", tmp_path / "ws/tools/agent-guard",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / ".cursor", tmp_path / "ws/.cursor")
    git("add", "tools", ".cursor")
    git("commit", "-q", "-m", "guard")
    git("push", "-q", "origin", "HEAD:main")
    git("clone", "-q", str(tmp_path / "origin.git"), str(tmp_path / "agent/repos/verity"), cwd=tmp_path)

    p = subprocess.run(["bash", str(BOOT), str(tmp_path / "ws")], capture_output=True, text=True, env=env)
    assert p.returncode == 0 and f"wrote {tmp_path / 'agent'}/.cursor/hooks.json" in p.stdout, (p.stdout, p.stderr)
    assert not (tmp_path / "absent").exists()
    assert json.loads((tmp_path / "ws/.cursor/hooks.json").read_text()) == json.loads((ROOT / ".cursor/hooks.json").read_text())
    hooks = cursor_reads(tmp_path / "agent/.cursor/hooks.json")["hooks"]
    (entry,) = hooks["beforeShellExecution"]
    assert entry["failClosed"] is True and entry["command"].endswith(" guard shell_hook.py")
    assert [h["matcher"] for h in hooks["preToolUse"]] == ["Task", "CallDynamicTool", "^MCP:"]
    assert [h["command"].rsplit(" ", 1)[1] for h in hooks["stop"]] == ["stop_hook.py"]

    def verdict():
        out = subprocess.run(entry["command"], shell=True, cwd=tmp_path / "agent", capture_output=True, text=True, check=True,
                             input=json.dumps({"hook_event_name": "beforeShellExecution", "command": "env | cut -d= -f1",
                                               "cwd": str(tmp_path / "agent"), "workspace_roots": [str(tmp_path / "agent")]}),
                             env={**env, "AGENT_GUARD_CACHE": str(tmp_path / "cache"), "AGENT_INBOX_POLL_S": "off"})
        return json.loads(out.stdout)["permission"]

    assert verdict() == "deny"
    assert any((tmp_path / "cache").iterdir()), "the checkout's hook.sh ran origin/main's guard"
    shutil.rmtree(tmp_path / "agent/repos")
    assert verdict() == "deny", "DEST's copy runs while there is no checkout"
