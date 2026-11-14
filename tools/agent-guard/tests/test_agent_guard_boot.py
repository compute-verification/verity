"""`tools/agent-guard/boot.sh` puts the guard's files on a VM and gives a checkout that tracks no `.cursor/hooks.json` a
fail-closed one. It can't give a branch a hook: a session runs the hooks of the branch its VM boots on, checked out after
the environment's `start`, so boot leaves a tracked hooks.json alone and writes none anywhere else."""
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


def test_boot_gives_a_checkout_that_tracks_no_hooks_file_a_fail_closed_one_and_leaves_a_tracked_one_alone(tmp_path):
    env = {**os.environ, **GIT_ENV, "AGENT_GUARD_DEST": str(tmp_path / "opt"), "AGENT_GUARD_SKIP_INSTALL": "1",
           "AGENT_GUARD_BASHRC": str(tmp_path / "bashrc")}

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
    assert p.returncode == 0 and "tracks .cursor/hooks.json, left as it is" in p.stdout, (p.stdout, p.stderr)
    assert "its hook runs" not in p.stdout, "boot can't know: Cursor may read no hook from a tracked file"
    assert git("status", "--porcelain") == ""
    path_lines = [ln for ln in (tmp_path / "bashrc").read_text().splitlines() if "/workspace/.venv/bin" in ln]
    assert path_lines == ["export PATH=/workspace/.git/verity-auth/bin:/workspace/.venv/bin:$PATH"], "research on PATH, once"


def test_boot_refuses_without_origin_or_an_earlier_copy(tmp_path):
    env = {**os.environ, **GIT_ENV, "AGENT_GUARD_DEST": str(tmp_path / "opt"), "AGENT_GUARD_SKIP_INSTALL": "1",
           "AGENT_GUARD_BASHRC": str(tmp_path / "bashrc")}
    subprocess.run(["git", "init", "-q", str(tmp_path / "ws")], check=True, env=env)
    p = subprocess.run(["bash", str(BOOT), str(tmp_path / "ws")], capture_output=True, text=True, env=env)
    assert p.returncode == 1 and "no guard on this VM" in p.stderr, p.stderr
    (entry,) = json.loads((tmp_path / "ws/.cursor/hooks.json").read_text())["hooks"]["beforeShellExecution"]
    assert entry["failClosed"] is True and not (tmp_path / "opt/shell_hook.py").exists()


def test_boot_writes_no_hooks_file_above_the_checkout(tmp_path):
    """A Task-launched cloud agent's workspace root is /agent with the repository under /agent/repos/, and Cursor reads the
    checkout's hooks.json, never one at the root: a file boot wrote there ran no hook in any session. So boot writes a hooks
    file nowhere but in WORKSPACE, and the variable that named such roots does nothing."""
    ws = tmp_path / "agent/repos/verity"
    env = {**os.environ, **GIT_ENV, "AGENT_GUARD_DEST": str(tmp_path / "opt"), "AGENT_GUARD_SKIP_INSTALL": "1",
           "AGENT_GUARD_BASHRC": str(tmp_path / "bashrc"), "AGENT_GUARD_ROOTS": str(tmp_path / "agent")}

    def git(*a, cwd=ws):
        return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True, env=env).stdout

    git("init", "-q", "--bare", "-b", "main", str(tmp_path / "origin.git"), cwd=tmp_path)
    git("clone", "-q", str(tmp_path / "origin.git"), str(ws), cwd=tmp_path)
    shutil.copytree(ROOT / "tools/agent-guard", ws / "tools/agent-guard", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / ".cursor", ws / ".cursor")
    git("add", "tools", ".cursor")
    git("commit", "-q", "-m", "guard")
    git("push", "-q", "origin", "HEAD:main")

    p = subprocess.run(["bash", str(BOOT), str(ws)], capture_output=True, text=True, env=env)
    assert p.returncode == 0, (p.stdout, p.stderr)
    assert (tmp_path / "opt/shell_hook.py").is_file(), "the guard's files are on the VM"
    assert sorted(tmp_path.rglob("hooks.json")) == [ws / ".cursor/hooks.json"], "the checkout's tracked file, and no other"
    assert git("status", "--porcelain") == ""
