"""`tools/agent-guard/boot.sh` puts compute-verification/infra's agent guard on a VM, as the environment's `start` runs it:
it clones infra to INFRA and runs infra's boot.sh, which installs the guard in DEST; when infra can't be fetched it runs
INFRA's earlier checkout, else Verity's last guard. A branch without `.cursor/hooks.json` gets one that runs DEST's hook
fail-closed, and no path ends with a hook in DEST that lets `env` through. Infra's boot.sh is infra's to test: here it is a
stub that installs the hook beside it."""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOT = ROOT / "tools/agent-guard/boot.sh"
GIT_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
DENY = 'import json, sys\njson.load(sys.stdin)\nprint(json.dumps({"permission": "deny"}))\n'
ALLOW = DENY.replace("deny", "allow")
INFRA_BOOT = ('set -e\nmkdir -p "$AGENT_GUARD_DEST"\ncp "$(dirname "$0")/shell_hook.py" "$AGENT_GUARD_DEST/"\n'
              'echo "infra boot.sh {} ran for $1"\n')
LAST_BOOT = 'echo "Verity\'s last boot.sh ran for $1"\n'
ENV_EVENT = json.dumps({"hook_event_name": "beforeShellExecution", "command": "env | cut -d= -f1", "cwd": "/"})


def git(env, cwd, *a) -> str:
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True, env=env).stdout.strip()


def commit(env, repo, files: dict, message: str) -> str:
    """Write `files` (None deletes one), commit them and push to origin's main."""
    for name, text in files.items():
        if text is None:
            (repo / name).unlink()
        else:
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_text(text)
    git(env, repo, "add", "-A")
    git(env, repo, "commit", "-q", "-m", message)
    git(env, repo, "push", "-q", "origin", "HEAD:main")
    return git(env, repo, "rev-parse", "HEAD")


def infra_guard(version: str, hook: str = DENY) -> dict:
    return {"tools/agent-guard/boot.sh": INFRA_BOOT.format(version),
            "tools/agent-guard/shell_hook.py": f"{hook}# infra {version}\n"}


def vm(tmp_path, infra_hook: str | None = DENY):
    """A VM: WORKSPACE, a Verity clone on a branch older than the guard, whose origin/main had Verity's last guard and then
    moved it to infra; infra's repository with a stub guard whose hook is `infra_hook` (none when None); DEST and INFRA
    empty."""
    env = {**os.environ, **GIT_ENV, "AGENT_GUARD_DEST": str(tmp_path / "opt"), "AGENT_GUARD_INFRA": str(tmp_path / "infra"),
           "AGENT_GUARD_INFRA_URL": str(tmp_path / "infra.git"), "AGENT_GUARD_SKIP_INSTALL": "1",
           "AGENT_GUARD_BASHRC": str(tmp_path / "bashrc")}
    ws = tmp_path / "ws"
    git(env, tmp_path, "init", "-q", "--bare", "-b", "main", "origin.git")
    git(env, tmp_path, "clone", "-q", "origin.git", "ws")
    commit(env, ws, {"a": "old\n"}, "old")
    git(env, ws, "branch", "old")
    commit(env, ws, {"tools/agent-guard/boot.sh": LAST_BOOT, "tools/agent-guard/shell_hook.py": DENY + "# Verity's last\n"},
           "Verity's last guard")
    commit(env, ws, {"tools/agent-guard/boot.sh": BOOT.read_text(), "tools/agent-guard/shell_hook.py": None,
                     ".cursor/hooks.json": (ROOT / ".cursor/hooks.json").read_text()}, "the guard moves to infra")
    git(env, ws, "checkout", "-q", "old")
    if infra_hook is not None:
        git(env, tmp_path, "init", "-q", "--bare", "-b", "main", "infra.git")
        git(env, tmp_path, "clone", "-q", "infra.git", "infra-src")
        commit(env, tmp_path / "infra-src", infra_guard("v1", infra_hook), "v1")
    return env, ws


def boot(env, ws) -> subprocess.CompletedProcess:
    """The environment's `start`: origin/main's boot.sh on bash's stdin."""
    return subprocess.run(["bash", "-s", "--", str(ws)], input=git(env, ws, "show", "origin/main:tools/agent-guard/boot.sh"),
                          capture_output=True, text=True, env=env)


def refuses_env(dest: Path) -> bool:
    p = subprocess.run(["python3", str(dest / "shell_hook.py")], input=ENV_EVENT, capture_output=True, text=True, check=True)
    return json.loads(p.stdout)["permission"] == "deny"


def test_boot_clones_infra_runs_its_boot_sh_and_brings_it_to_infras_main_on_the_next_boot(tmp_path):
    env, ws = vm(tmp_path)
    p = boot(env, ws)
    assert p.returncode == 0 and "wrote" in p.stdout and f"infra boot.sh v1 ran for {ws}" in p.stdout, (p.stdout, p.stderr)
    (entry,) = json.loads((ws / ".cursor/hooks.json").read_text())["hooks"]["beforeShellExecution"]
    assert entry == {"command": f"python3 {tmp_path / 'opt'}/shell_hook.py", "timeout": 10, "failClosed": True}
    assert git(env, ws, "status", "--porcelain") == ""
    assert refuses_env(tmp_path / "opt") and (tmp_path / "opt/shell_hook.py").read_text().endswith("# infra v1\n")

    v2 = commit(env, tmp_path / "infra-src", infra_guard("v2"), "v2")
    (tmp_path / "infra/tools/agent-guard/boot.sh").write_text("exit 1\n")
    p = boot(env, ws)
    assert p.returncode == 0 and "infra boot.sh v2 ran" in p.stdout, (p.stdout, p.stderr)
    assert git(env, tmp_path / "infra", "rev-parse", "HEAD") == v2
    assert git(env, tmp_path / "infra", "branch", "--show-current") == "main"


def test_boot_runs_infras_earlier_checkout_when_infra_cant_be_fetched(tmp_path):
    env, ws = vm(tmp_path)
    assert boot(env, ws).returncode == 0
    env["AGENT_GUARD_INFRA_URL"] = str(tmp_path / "gone.git")
    p = boot(env, ws)
    assert p.returncode == 0 and f"can't be fetched; running {tmp_path / 'infra'}'s checkout" in p.stderr, (p.stdout, p.stderr)
    assert "infra boot.sh v1 ran" in p.stdout and refuses_env(tmp_path / "opt")


def test_without_infra_verity_s_last_guard_runs_and_a_guard_dest_has_stays(tmp_path):
    env, ws = vm(tmp_path, infra_hook=None)
    p = boot(env, ws)
    assert p.returncode == 0 and "put Verity's last guard" in p.stderr, (p.stdout, p.stderr)
    assert f"Verity's last boot.sh ran for {ws}" in p.stdout
    assert (tmp_path / "opt/shell_hook.py").read_text().endswith("# Verity's last\n")

    (tmp_path / "opt/shell_hook.py").write_text(DENY + "# an earlier boot's\n")
    p = boot(env, ws)
    assert p.returncode == 0 and "keeping the guard in" in p.stderr and "Verity's last boot.sh ran" in p.stdout, p.stderr
    assert (tmp_path / "opt/shell_hook.py").read_text().endswith("# an earlier boot's\n")


def test_a_guard_from_infra_that_lets_env_through_gives_way_to_verity_s_last(tmp_path):
    env, ws = vm(tmp_path, infra_hook=ALLOW)
    p = boot(env, ws)
    assert p.returncode == 0 and "infra boot.sh v1 ran" in p.stdout and "put Verity's last guard" in p.stderr, p.stderr
    assert refuses_env(tmp_path / "opt")


def test_with_no_guard_to_install_the_fail_closed_hook_refuses_every_command(tmp_path):
    env, ws = vm(tmp_path, infra_hook=None)
    git(env, ws, "update-ref", "-d", "refs/remotes/origin/main")
    (tmp_path / "opt").mkdir()
    (tmp_path / "opt/shell_hook.py").write_text(ALLOW)
    p = subprocess.run(["bash", str(BOOT), str(ws)], capture_output=True, text=True, env=env)
    assert p.returncode == 1 and "no guard on this VM" in p.stderr, (p.stdout, p.stderr)
    (entry,) = json.loads((ws / ".cursor/hooks.json").read_text())["hooks"]["beforeShellExecution"]
    assert entry["failClosed"] is True and not (tmp_path / "opt/shell_hook.py").exists()


def test_a_last_guard_whose_boot_sh_is_this_one_stops_instead_of_running_itself_forever(tmp_path):
    env, ws = vm(tmp_path, infra_hook=None)
    git(env, ws, "checkout", "-q", "-b", "beside", "origin/main")
    commit(env, ws, {"tools/agent-guard/shell_hook.py": ALLOW}, "a hook beside the shim")
    p = boot(env, ws)
    assert f"{ws} tracks .cursor/hooks.json" in p.stdout
    assert p.returncode == 1 and "is this boot.sh; stopping" in p.stderr and "no guard on this VM" in p.stderr, p.stderr
    assert not (tmp_path / "opt/shell_hook.py").exists()


def test_tracked_hooks_run_the_installed_guard_fail_closed():
    hooks = json.loads((ROOT / ".cursor/hooks.json").read_text())["hooks"]
    (entry,) = hooks["beforeShellExecution"]
    assert entry == {"command": "python3 /opt/agent-guard/shell_hook.py", "timeout": 10, "failClosed": True}
    assert {e["command"] for e in hooks["preToolUse"]} == {"python3 /opt/agent-guard/task_hook.py"}
    assert sorted(e["matcher"] for e in hooks["preToolUse"]) == ["CallDynamicTool", "Task", "^MCP:"]


def test_a_session_whose_hooks_ran_through_hook_sh_runs_the_installed_hook(tmp_path):
    (tmp_path / "shell_hook.py").write_text("import json, os, sys\n"
                                            "print(json.dumps({'registry': os.environ['AGENT_GUARD_REGISTRY'], "
                                            "'event': json.load(sys.stdin)}))\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("AGENT_GUARD_")} | {"AGENT_GUARD_DEST": str(tmp_path)}
    p = subprocess.run(["sh", "tools/agent-guard/hook.sh", "shell_hook.py"], cwd=ROOT, input=ENV_EVENT, capture_output=True,
                       text=True, check=True, env=env)
    out = json.loads(p.stdout)
    assert out["event"] == json.loads(ENV_EVENT)
    assert (ROOT / out["registry"]).resolve() == ROOT / ".agents/skills/using-slack/registry.json"


def test_verity_holds_only_the_boot_shim():
    files = sorted(p.name for p in (ROOT / "tools/agent-guard").iterdir() if p.is_file())
    assert files == ["boot.sh", "hook.sh"], ("the agent guard lives in compute-verification/infra (tools/agent-guard): change "
                                             f"it there, not here: {files}")
