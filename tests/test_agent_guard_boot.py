"""`tools/agent-guard/boot.sh` puts the shell hook on a VM whose branch predates it: cloud agents read project hooks only, and
on 10 Oct top's and ci's VMs ran no hook because their branches (5 Oct) had no `.cursor/hooks.json`."""
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOT = ROOT / "tools/agent-guard/boot.sh"
GIT_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def test_boot_installs_the_hook_on_an_old_branch_and_steps_aside_for_a_tracked_one(tmp_path):
    env = {**os.environ, **GIT_ENV, "AGENT_GUARD_DEST": str(tmp_path / "opt"), "AGENT_GUARD_SKIP_INSTALL": "1"}

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
    shutil.copytree(ROOT / "tools/agent-guard", tmp_path / "ws/tools/agent-guard")
    shutil.copytree(ROOT / ".cursor", tmp_path / "ws/.cursor")
    git("add", "tools", ".cursor")
    git("commit", "-q", "-m", "guard")
    git("push", "-q", "origin", "HEAD:main")
    git("checkout", "-q", "old")
    assert not (tmp_path / "ws/.cursor").exists()

    p = boot()
    assert p.returncode == 0 and "wrote" in p.stdout, (p.stdout, p.stderr)
    (entry,) = json.loads((tmp_path / "ws/.cursor/hooks.json").read_text())["hooks"]["beforeShellExecution"]
    assert entry == {"command": f"python3 {tmp_path / 'opt'}/shell_hook.py", "timeout": 10, "failClosed": True}
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


def test_boot_refuses_without_origin_or_an_earlier_copy(tmp_path):
    env = {**os.environ, **GIT_ENV, "AGENT_GUARD_DEST": str(tmp_path / "opt"), "AGENT_GUARD_SKIP_INSTALL": "1"}
    subprocess.run(["git", "init", "-q", str(tmp_path / "ws")], check=True, env=env)
    p = subprocess.run(["bash", str(BOOT), str(tmp_path / "ws")], capture_output=True, text=True, env=env)
    assert p.returncode == 1 and "no guard on this VM" in p.stderr, p.stderr
    (entry,) = json.loads((tmp_path / "ws/.cursor/hooks.json").read_text())["hooks"]["beforeShellExecution"]
    assert entry["failClosed"] is True and not (tmp_path / "opt/shell_hook.py").exists()
