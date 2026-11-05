"""`tools/agent-guard/boot.sh` puts compute-verification/infra's agent guard on a VM, as the environment's `start` runs it:
it clones infra to INFRA and runs infra's boot.sh, which installs the guard in DEST; when infra can't be fetched it runs
INFRA's earlier checkout, else Verity's last guard, and leaves one retry behind, which takes the same steps once the fetch
works. A branch without `.cursor/hooks.json` gets one that runs DEST's hook fail-closed, and no path ends with a hook in DEST
that lets `env` through. The tracked hooks file runs DEST's hooks through `hook.sh`, which names the Slack registry for them:
Verity's last Task hook, taken from this checkout's history, finds none beside it in DEST. When DEST has no hook, `hook.sh`
runs Verity's last guard from origin/main's history, and with none there the hook fails closed. Infra's boot.sh is infra's to
test: here it is a stub that installs the hook beside it. Here the retry waits no time between tries, and a git on PATH
fails infra's fetch as many times as a test asks."""
import fcntl
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOT = ROOT / "tools/agent-guard/boot.sh"
GIT_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
REGISTRY = ".agents/skills/using-slack/registry.json"
HOLDER = "bc-8f4b652b-17bc-5055-a120-b56940b7c7a9"
FORMER = "bc-00000000-0000-0000-0000-000000000000"
DENY = 'import json, sys\njson.load(sys.stdin)\nprint(json.dumps({"permission": "deny"}))\n'
ALLOW = DENY.replace("deny", "allow")
INFRA_BOOT = ('set -e\nmkdir -p "$AGENT_GUARD_DEST"\ncp "$(dirname "$0")/shell_hook.py" "$AGENT_GUARD_DEST/"\n'
              'echo "infra boot.sh {} ran for $1"\n')
LAST_BOOT = 'echo "Verity\'s last boot.sh ran for $1"\n'
ENV_EVENT = json.dumps({"hook_event_name": "beforeShellExecution", "command": "env | cut -d= -f1", "cwd": "/"})
# git, but a clone or fetch of infra fails until it has been tried more than INFRA_FAILS times; with INFRA_HOLD set, every
# try but the first waits until the test lets go of its lock on INFRA_HOLD
FLAKY_GIT = """#!/bin/bash
case " $* " in *" $AGENT_GUARD_INFRA_URL "*)
  echo >> "$INFRA_TRIES"
  n=$(wc -l < "$INFRA_TRIES")
  [ "$n" = 1 ] || [ -z "$INFRA_HOLD" ] || flock "$INFRA_HOLD" true
  [ "$n" -gt "$INFRA_FAILS" ] || exit 128
esac
exec {git} "$@"
"""


def git(env, cwd, *a) -> str:
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True, env=env).stdout.strip()


def commit(env, repo, files: dict, message: str, branch: str = "main") -> str:
    """Write `files` (None deletes one), commit them and push to origin's `branch`."""
    for name, text in files.items():
        if text is None:
            (repo / name).unlink()
        else:
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_text(text)
    git(env, repo, "add", "-A")
    git(env, repo, "commit", "-q", "-m", message)
    git(env, repo, "push", "-q", "origin", f"HEAD:{branch}")
    return git(env, repo, "rev-parse", "HEAD")


def infra_guard(version: str, hook: str = DENY) -> dict:
    return {"tools/agent-guard/boot.sh": INFRA_BOOT.format(version),
            "tools/agent-guard/shell_hook.py": f"{hook}# infra {version}\n"}


def vm(tmp_path, infra_hook: str | None = DENY):
    """A VM: WORKSPACE, a Verity clone on a branch older than the guard, whose origin/main changed the guard after the branch
    that moves it to infra started, then merged that branch, first parent main, as the queue lands a PR; infra's repository
    with a stub guard whose hook is `infra_hook` (none when None); DEST and INFRA empty."""
    env = {**os.environ, **GIT_ENV, "AGENT_GUARD_DEST": str(tmp_path / "opt"), "AGENT_GUARD_INFRA": str(tmp_path / "infra"),
           "AGENT_GUARD_INFRA_URL": str(tmp_path / "infra.git"), "AGENT_GUARD_SKIP_INSTALL": "1",
           "AGENT_GUARD_BASHRC": str(tmp_path / "bashrc"), "AGENT_GUARD_RETRY_LOG": str(tmp_path / "retry.log"),
           "AGENT_GUARD_RETRY_ATTEMPTS": "0", "AGENT_GUARD_RETRY_WAIT": "0"}
    ws = tmp_path / "ws"
    git(env, tmp_path, "init", "-q", "--bare", "-b", "main", "origin.git")
    git(env, tmp_path, "clone", "-q", "origin.git", "ws")
    commit(env, ws, {"a": "old\n"}, "old")
    git(env, ws, "branch", "old")
    commit(env, ws, {"tools/agent-guard/boot.sh": LAST_BOOT, "tools/agent-guard/shell_hook.py": DENY + "# Verity's earlier\n"},
           "an earlier guard")
    git(env, ws, "checkout", "-q", "-b", "move")
    commit(env, ws, {"tools/agent-guard/boot.sh": BOOT.read_text(), "tools/agent-guard/shell_hook.py": None,
                     ".cursor/hooks.json": (ROOT / ".cursor/hooks.json").read_text()}, "the guard moves to infra", "move")
    git(env, ws, "checkout", "-q", "main")
    commit(env, ws, {"tools/agent-guard/shell_hook.py": DENY + "# Verity's last\n"}, "Verity's last guard")
    assert subprocess.run(["git", "merge", "-q", "--no-ff", "move"], cwd=ws, env=env, capture_output=True).returncode
    git(env, ws, "rm", "-q", "tools/agent-guard/shell_hook.py")
    git(env, ws, "commit", "-q", "--no-edit")
    git(env, ws, "push", "-q", "origin", "HEAD:main")
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


def flaky(env, tmp_path, fails: int, attempts: int) -> dict:
    """`env` with FLAKY_GIT on PATH, failing infra's fetch `fails` times, and a boot.sh that retries it `attempts` times."""
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/git").write_text(FLAKY_GIT.format(git=shutil.which("git")))
    (tmp_path / "bin/git").chmod(0o755)
    return env | {"PATH": f"{tmp_path / 'bin'}:{env['PATH']}", "INFRA_TRIES": str(tmp_path / "tries"),
                  "INFRA_FAILS": str(fails), "AGENT_GUARD_RETRY_ATTEMPTS": str(attempts)}


def retried(env) -> str:
    """The retry's log once the retry has ended: it holds a lock on the log while it runs."""
    with open(env["AGENT_GUARD_RETRY_LOG"], "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        return f.read()


def tries(tmp_path) -> int:
    return len((tmp_path / "tries").read_text().splitlines())


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


def test_a_boot_that_cant_fetch_infra_puts_verity_s_last_guard_and_its_retry_puts_infra_s(tmp_path):
    env, ws = vm(tmp_path)
    env = flaky(env, tmp_path, fails=3, attempts=5)
    p = boot(env, ws)
    assert p.returncode == 0 and "put Verity's last guard" in p.stderr, (p.stdout, p.stderr)
    assert "retrying infra's main in the background every 0s, up to 5 times" in p.stderr
    log = retried(env)
    assert "fetched infra's main on retry 3" in log and f"infra boot.sh v1 ran for {ws}" in log, log
    assert tries(tmp_path) == 4
    assert refuses_env(tmp_path / "opt") and (tmp_path / "opt/shell_hook.py").read_text().endswith("# infra v1\n")
    p = boot(env, ws)
    assert p.returncode == 0 and "infra boot.sh v1 ran" in p.stdout and "retry" not in p.stderr, (p.stdout, p.stderr)


def test_a_retry_that_never_fetches_infra_leaves_verity_s_last_guard(tmp_path):
    env, ws = vm(tmp_path)
    env = flaky(env, tmp_path, fails=100, attempts=3)
    assert boot(env, ws).returncode == 0
    log = retried(env)
    assert "still can't be fetched after 3 retries" in log and "fetched infra's main" not in log, log
    assert tries(tmp_path) == 4
    assert refuses_env(tmp_path / "opt") and (tmp_path / "opt/shell_hook.py").read_text().endswith("# Verity's last\n")


def test_a_boot_that_leaves_no_guard_still_leaves_a_retry(tmp_path):
    env, ws = vm(tmp_path)
    env = flaky(env, tmp_path, fails=1, attempts=1)
    git(env, ws, "update-ref", "-d", "refs/remotes/origin/main")
    p = subprocess.run(["bash", str(BOOT), str(ws)], capture_output=True, text=True, env=env)
    assert p.returncode == 1 and "no guard on this VM" in p.stderr, (p.stdout, p.stderr)
    assert "retrying infra's main in the background" in p.stderr
    assert "fetched infra's main on retry 1" in retried(env) and refuses_env(tmp_path / "opt")


def test_a_boot_while_a_retry_runs_leaves_no_second_retry(tmp_path):
    env, ws = vm(tmp_path)
    env = flaky(env, tmp_path, fails=100, attempts=1)
    with open(tmp_path / "hold", "w") as hold:
        fcntl.flock(hold, fcntl.LOCK_EX)
        first = boot(env | {"INFRA_HOLD": hold.name}, ws)
        second = boot(env, ws)
    assert "retrying infra's main in the background" in first.stderr, first.stderr
    assert second.returncode == 0 and "no retry started" in second.stderr, second.stderr
    log = retried(env)
    assert log.count("retrying infra's main every") == 1 and "after 1 retries" in log, log
    assert tries(tmp_path) == 3


def hook_env(tmp_path) -> dict:
    return {k: v for k, v in os.environ.items() if not k.startswith("AGENT_GUARD_")} | GIT_ENV | {
        "AGENT_GUARD_DEST": str(tmp_path / "opt"), "AGENT_GUARD_CACHE": str(tmp_path / "cache"),
        "AGENT_GUARD_TASK_LOG": str(tmp_path / "task.jsonl")}


def shim(tmp_path, *history: dict) -> tuple[dict, Path]:
    """WORKSPACE: a clone whose origin/main has this hook.sh, then each commit of ``history`` (a file None deletes); DEST and
    the cache empty."""
    env = hook_env(tmp_path)
    ws = tmp_path / "ws"
    git(env, tmp_path, "init", "-q", "--bare", "-b", "main", "origin.git")
    git(env, tmp_path, "clone", "-q", "origin.git", "ws")
    commit(env, ws, {"tools/agent-guard/hook.sh": (ROOT / "tools/agent-guard/hook.sh").read_text()}, "the shim")
    for i, files in enumerate(history):
        commit(env, ws, files, f"guard {i}")
    return env, ws


def shell_hook(env, ws) -> subprocess.CompletedProcess:
    """The tracked shell hook, as Cursor runs it in WORKSPACE, on `env | cut -d= -f1`."""
    (entry,) = json.loads((ROOT / ".cursor/hooks.json").read_text())["hooks"]["beforeShellExecution"]
    return subprocess.run(shlex.split(entry["command"]), cwd=ws, input=ENV_EVENT, capture_output=True, text=True, env=env)


def named(name: str) -> str:
    """A shell hook that answers with its name and the registry it was given."""
    return f'import json, os\nprint(json.dumps({{"hook": "{name}", "registry": os.environ["AGENT_GUARD_REGISTRY"]}}))\n'


def last_guard(name: str) -> str:
    """NAME in Verity's last guard, the tree boot.sh falls back to, found as boot.sh finds it in origin/main's history but in
    HEAD's, which is origin/main's once this lands: the last first-parent commit that touched the shell hook, or its parent
    when that commit deleted it."""
    last = subprocess.run(["git", "rev-list", "-1", "--first-parent", "HEAD", "--", "tools/agent-guard/shell_hook.py"],
                          cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    if subprocess.run(["git", "cat-file", "-e", f"{last}:tools/agent-guard/shell_hook.py"], cwd=ROOT).returncode:
        last += "^"
    return subprocess.run(["git", "show", f"{last}:tools/agent-guard/{name}"], cwd=ROOT, check=True, capture_output=True,
                          text=True).stdout


def test_tracked_hooks_run_the_installed_guard_through_hook_sh_fail_closed(tmp_path):
    hooks = json.loads((ROOT / ".cursor/hooks.json").read_text())["hooks"]
    (entry,) = hooks["beforeShellExecution"]
    assert entry == {"command": "sh tools/agent-guard/hook.sh shell_hook.py", "timeout": 10, "failClosed": True}
    assert {e["command"] for e in hooks["preToolUse"]} == {"sh tools/agent-guard/hook.sh task_hook.py"}
    assert sorted(e["matcher"] for e in hooks["preToolUse"]) == ["CallDynamicTool", "Task", "^MCP:"]
    p = shell_hook(*shim(tmp_path))
    assert p.returncode != 0 and p.stdout == "", (p.stdout, p.stderr)


def test_with_no_hook_in_dest_hook_sh_runs_the_shell_hook_origin_main_had_before_it_was_deleted(tmp_path):
    """A VM that hasn't booted the guard since it moved has no DEST. Its shell hook is the one origin/main had before the
    commit that deleted it, extracted once into the cache under that commit, where a later call finds it with no history to
    read. Once DEST has the hook, DEST's runs."""
    env, ws = shim(tmp_path, {"tools/agent-guard/shell_hook.py": named("last")}, {"tools/agent-guard/shell_hook.py": None})
    p = shell_hook(env, ws)
    assert p.returncode == 0 and json.loads(p.stdout) == {"hook": "last", "registry": f"{ws.resolve()}/{REGISTRY}"}, p.stderr
    assert (tmp_path / "cache/last").resolve() == (tmp_path / "cache" / git(env, ws, "rev-parse", "HEAD^")).resolve()
    git(env, ws, "update-ref", "-d", "refs/remotes/origin/main")
    p = shell_hook(env, ws)
    assert p.returncode == 0 and json.loads(p.stdout)["hook"] == "last", (p.stdout, p.stderr)
    (tmp_path / "opt").mkdir()
    (tmp_path / "opt/shell_hook.py").write_text(named("dest"))
    p = shell_hook(env, ws)
    assert p.returncode == 0 and json.loads(p.stdout)["hook"] == "dest", (p.stdout, p.stderr)


def test_verity_s_last_task_hook_refuses_a_holder_s_foreground_task_through_the_hooks_file(tmp_path):
    """In DEST, Verity's last Task hook looks for the Slack registry beside its own file, where there is none. Through the
    hooks file it gets origin/main's from hook.sh, which knows the handle's holder registered after the branch was cut, and
    the branch's when origin/main can't be read."""
    env = hook_env(tmp_path)
    (tmp_path / "opt").mkdir()
    (tmp_path / "opt/task_hook.py").write_text(last_guard("task_hook.py"))
    ws = tmp_path / "ws"
    git(env, tmp_path, "init", "-q", "--bare", "-b", "main", "origin.git")
    git(env, tmp_path, "clone", "-q", "origin.git", "ws")

    def registry(holder):
        return json.dumps({"handles": [{"handle": "proofs", "holder": f"proofs ({holder})"}]})

    commit(env, ws, {"tools/agent-guard/hook.sh": (ROOT / "tools/agent-guard/hook.sh").read_text(), REGISTRY: registry(FORMER)},
           "a branch is cut")
    git(env, ws, "branch", "cut")
    commit(env, ws, {REGISTRY: registry(HOLDER)}, "the handle's next holder")
    git(env, ws, "checkout", "-q", "cut")
    hooks = json.loads((ROOT / ".cursor/hooks.json").read_text())["hooks"]
    (task,) = [e["command"] for e in hooks["preToolUse"] if e["matcher"] == "Task"]

    def verdict(agent):
        event = {"hook_event_name": "preToolUse", "tool_name": "Task", "conversation_id": agent,
                 "tool_input": {"subagent_type": "generalPurpose", "prompt": "x"}}
        p = subprocess.run(shlex.split(task), cwd=ws, input=json.dumps(event), capture_output=True, text=True, env=env,
                           check=True)
        return json.loads(p.stdout)["permission"]

    assert (verdict(HOLDER), verdict(FORMER)) == ("deny", "allow")
    git(env, ws, "update-ref", "-d", "refs/remotes/origin/main")
    assert (verdict(HOLDER), verdict(FORMER)) == ("allow", "deny")


def test_verity_holds_only_the_boot_shim():
    files = sorted(p.name for p in (ROOT / "tools/agent-guard").iterdir() if p.is_file())
    assert files == ["boot.sh", "hook.sh"], ("the agent guard lives in compute-verification/infra (tools/agent-guard): change "
                                             f"it there, not here: {files}")
