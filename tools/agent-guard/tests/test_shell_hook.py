"""`.cursor/hooks.json`'s shell hook refuses commands that print the environment and lets everything else through."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
HOOK = ROOT / "tools/agent-guard/shell_hook.py"
GIT_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def verdict(command: str, **env: str) -> str:
    return answer(command, **env)["permission"]


def answer(command: str, **env: str) -> dict:
    env = {"AGENT_GUARD_MIN_AVAIL": "0", **env}           # the memory gate only where a test sets it, whatever this VM has free
    out = subprocess.run([sys.executable, HOOK], input=json.dumps({"hook_event_name": "beforeShellExecution",
                                                                     "command": command, "cwd": str(ROOT)}),
                         capture_output=True, text=True, env={**os.environ, **env}, check=True).stdout
    return json.loads(out)


def test_cursor_runs_the_hook_on_every_shell_command_and_blocks_if_it_fails():
    (entry,) = json.loads((ROOT / ".cursor/hooks.json").read_text())["hooks"]["beforeShellExecution"]
    assert entry["command"] == "sh tools/agent-guard/hook.sh shell_hook.py" and entry["failClosed"] is True
    assert "matcher" not in entry


@pytest.mark.parametrize("command", [
    "env | cut -d= -f1",                                  # a multi-line value's later lines print as names
    "env",
    "env | sort",
    "env -0 | cut -z -d= -f1 | tr '\\0' '\\n'",
    "env -u HOME",
    "/usr/bin/env | grep -i nebius",
    "sudo -u research env",
    "env 2>/dev/null | head",
    "printenv",
    "printenv | cut -d= -f1",
    "printenv RUNPOD_API_KEY",
    "set",
    "set | grep NEBIUS",
    "export",
    "export -p",
    "declare -p",
    "declare -x",
    "typeset",
    "cat /proc/self/environ | tr '\\0' '\\n'",
    "tr '\\0' '\\n' < /proc/1/environ",
    "echo \"$(env)\" > /tmp/x",
    "echo `printenv`",
    "diff <(env) /tmp/before",
    "cd /workspace && env | wc -l",
    "true; env",
    "{ env; } | cut -d= -f1",
    "bash -c 'env | cut -d= -f1'",
    "ssh n1 env",
    "ssh -o BatchMode=yes n1 'env | cut -d= -f1'",
    "research pods ssh vy-mig-check-20 -- 'printenv'",
    "timeout 5 env",
    "env FOO=1 env",
    "xargs -0 env",
    "ps aux",                                             # a daemon's token in its argv
    "ps auxww | head -50",
    "ps -ef",
    "ps -eF | sort",
    "ps -eo pid,args",
    "ps -A -o pid,rss,command",
    "ps axo pid,args",
    "ps --everyone --format=pid,cmd",
    "sudo ps aux | tail",
    "ssh n1 'ps -ef'",
    "ps -o pid,etime,cmd --ppid 4242",                    # a git child's token-bearing URL
    "ps -f -u research",
    "ps -o args -C git",
    "ps -p 1,2 -o args",
    "cat /proc/*/cmdline",
    "tmux -f /exec-daemon/tmux.portal.conf show-environment -g | cut -d= -f1",   # a multi-line key's lines
    "tmux show-environment",
    "tmux showenv -t main",
    "tmux show-environment -g NEBIUS_SA_PRIVATE_KEY",
    "ssh n1 'tmux show-env -g'",
])
def test_a_command_that_would_print_the_environment_is_refused(command):
    assert verdict(command) == "deny"


@pytest.mark.parametrize("command", [
    "compgen -e",
    "compgen -e | rg NEBIUS",
    "[ -n \"${NEBIUS_SA_PRIVATE_KEY_B64:-}\" ] && echo set",
    "env FOO=1 python3 -c 'print(1)'",
    "env -u PYTHONPATH uv run pytest tests",
    "/usr/bin/env bash -lc 'uv sync'",
    "printenv HOME",
    "printenv PATH | tr : '\\n'",
    "set -euo pipefail",
    "set -x; make",
    "export PYTHONPATH=/tmp/x",
    "export -n FOO",
    "declare -f my_function",
    "declare -A table",
    "local x=1",
    "uv run --env-file .env pytest tests/test_x.py",
    "python -m venv env && source env/bin/activate",
    "cat .env.example",
    "git commit -m 'never use env | cut, use compgen -e'",
    "rg -n 'env|printenv' tools",
    "ssh n1 'df -i /workspace'",
    "research pods ssh vy-mig-check-20 -- 'nvidia-smi'",
    "cat > /tmp/f <<'EOF'\nenv\nEOF",
    "echo \"$(git rev-parse HEAD)\"",
    "ls 2>/dev/null",
    "timeout 30 make test",
    "ps -eo pid,stat,etimes,comm",
    "ps -e",
    "ps axc",
    "ps ax -o pid,comm",
    "ps",
    "ps -p 1234 -o pid,args",
    "ps -o pid,etime,comm --ppid 4242",
    "ps -u research",
    "ps -eo pid,args | grep \"[p]ytest\"",
    "ps aux | rg -i lake",
    "ssh n1 'ps -eo pid,args | grep \"[s]lot.py\"'",
    "pgrep -af quick_tier",
    "tmux show-environment -g DISPLAY",
    "tmux -f /exec-daemon/tmux.portal.conf send-keys -t build:0.0 'make' C-m",
    "cat /proc/1234/cmdline | tr '\\0' ' '",
    "",
])
def test_everything_else_passes(command):
    assert verdict(command) == "allow"


def test_printing_a_variable_cursor_lists_as_a_secret_is_refused():
    assert verdict("printenv DUMMY_VALUE") == "allow"
    assert verdict("printenv DUMMY_VALUE", CLOUD_AGENT_ALL_SECRET_NAMES="OTHER,DUMMY_VALUE") == "deny"


def test_a_command_the_parser_cannot_read_is_judged_on_its_text():
    assert verdict("echo 'unbalanced; env | cut -d= -f1") == "deny"
    assert verdict("echo 'unbalanced; ls") == "allow"


def test_the_refusal_tells_the_agent_how_to_list_names_and_input_that_is_not_json_is_still_judged():
    out = subprocess.run([sys.executable, HOOK], input=json.dumps({"command": "env | cut -d= -f1"}),
                         capture_output=True, text=True, check=True).stdout
    assert "compgen -e" in json.loads(out)["agent_message"]
    out = subprocess.run([sys.executable, HOOK], input="env", capture_output=True, text=True, check=True).stdout
    assert json.loads(out)["permission"] == "deny"


def test_builds_and_tests_are_refused_on_an_agent_vm_and_light_ones_only_while_it_is_short_of_memory(tmp_path):
    """Remote by default for every handle: OOM kills on shared agent VMs took down every worker on them.
    A broad pytest, `lake build`, cargo, nvcc and `circuit-check --all` are refused here always, pointing to `research run --on
    auto`; a pytest of named files, `lake env`, the suites and `verity check` only under a quarter of the memory available. Through
    `research run`, over ssh, or with AGENT_GUARD_LOCAL_OK=1 any passes, as does work that is neither."""
    meminfo = tmp_path / "meminfo"
    cgroup = tmp_path / "cg"
    cgroup.mkdir()
    (cgroup / "memory.max").write_text("max\n")
    gate = {"AGENT_GUARD_MEMINFO": str(meminfo), "AGENT_GUARD_MIN_AVAIL": "0.25", "AGENT_GUARD_CGROUP": str(cgroup)}
    always = ("uv run pytest -q tests/", "cd tools/research && python -m pytest -x", "lake build Security", "cargo test --release",
              "uv run circuit-check --all", "nvcc -O3 k.cu", "timeout 600 /workspace/.venv/bin/python -m pytest")
    light = ("uv run pytest -q tests/test_x.py", "python -m pytest tests/test_x.py::test_y", "pytest -k mem tools/research/tests",
             "uv run tools/verity/check/suites.py --quick", "uv run --extra torch-cpu verity check",
             "cd Security && lake env lean X.lean")
    passes = ("research run --on auto -- uv run pytest -q", "uv run research run --on m --source . -- lake build",
              "ssh n1 'cd /w && pytest'", "git status", "AGENT_GUARD_LOCAL_OK=1 uv run pytest -q tests/", "rg pytest tools/", "ls")
    meminfo.write_text("MemTotal:       16000000 kB\nMemFree:          100000 kB\nMemAvailable:    6000000 kB\n")
    for command in always:
        out = answer(command, **gate)
        assert out["permission"] == "deny", command
        assert "research run --on auto" in out["agent_message"] and "AGENT_GUARD_LOCAL_OK=1" in out["agent_message"]
    for command in light + passes:
        assert verdict(command, **gate) == "allow", command
    meminfo.write_text("MemTotal:       16000000 kB\nMemFree:          100000 kB\nMemAvailable:    2000000 kB\n")
    for command in light:
        out = answer(command, **gate)
        assert out["permission"] == "deny", command
        assert "1.9 GiB of 15.3 GiB available" in out["agent_message"] and "research run --on auto" in out["agent_message"]
    for command in passes:
        assert verdict(command, **gate) == "allow", command
    meminfo.write_text("MemTotal:       16000000 kB\nMemFree:          100000 kB\nMemAvailable:    6000000 kB\n")
    (cgroup / "memory.max").write_text(f"{8 * 2**30}\n")         # a cgroup limit tighter than the VM: its headroom counts
    (cgroup / "memory.current").write_text(f"{7 * 2**30}\n")
    assert verdict("uv run pytest -q tests/test_x.py", **gate) == "deny"


@pytest.mark.parametrize("command", [
    "sleep 1201", "sleep 30m", "sleep 2h", "sleep 1d", "sleep 1h 30m", "sleep 1500s", "sleep infinity",
    "cd /w && sleep 3600 && research status r1", "while true; do sleep 1800; research status r1; done",
    "timeout 3h sleep 2h", "bash -c 'sleep 3000; ls'", 'echo "$(sleep 3000)"', "ssh n1 sleep 3000",
    "research run --on auto --wait --wait-max 3000 -- true", "research run --on auto --wait --wait-max=1201 -- true",
])
def test_a_command_that_plainly_blocks_past_twenty_minutes_is_refused(command):
    """No tool call may block more than 20 minutes: agents that blocked in one long call looked silent and missed
    their wake-ups."""
    out = answer(command)
    assert out["permission"] == "deny", command
    assert ("no tool call may block more than 20 minutes (Daniel, 19 Oct); end your turn and let the run's completion or a timer "
            "subscription wake you") in out["agent_message"]


@pytest.mark.parametrize("command", [
    "sleep 30", "sleep 20m", "sleep 1200", "sleep $DELAY", "timeout 60 sleep 5000", "sleep 5000 &", "nohup sleep 9000 &",
    "tail --pid=4242 -f /dev/null", "while kill -0 4242; do sleep 30; done", "research run --on auto --wait -- sleep 5000",
    "research run --on auto --wait --wait-max 600 -- true", "tmux send-keys -t s:0.0 'sleep 3000' C-m", "rg 'sleep 3000' tools/",
])
def test_a_wait_within_the_cap_or_off_this_shell_passes(command):
    assert verdict(command) == "allow", command


INBOX_TOOL = ROOT / "tools/agent-guard/inbox.py"


def hook(command: str, conversation: str, inbox: Path) -> dict:
    out = subprocess.run([sys.executable, HOOK], input=json.dumps({"hook_event_name": "beforeShellExecution",
                                                                     "command": command, "conversation_id": conversation}),
                         capture_output=True, text=True, env={**os.environ, "AGENT_INBOX": str(inbox)}, check=True).stdout
    return json.loads(out)


def queue(agent: str, text: str, inbox: Path) -> None:
    subprocess.run([sys.executable, INBOX_TOOL, "send", agent, "--text", text], env={**os.environ, "AGENT_INBOX": str(inbox)},
                   check=True, capture_output=True)


def test_a_queued_message_refuses_the_agents_next_command_once_with_the_message_as_its_text(tmp_path):
    queue("bc-worker", "rebase on main first", tmp_path)
    queue("bc-worker", "then mark nothing", tmp_path)
    first = hook("ls", "bc-worker", tmp_path)
    assert first["permission"] == "deny"
    assert "re-run it exactly as it was" in first["user_message"]
    assert "rebase on main first" in first["user_message"] and "then mark nothing" in first["user_message"]
    assert hook("ls", "bc-worker", tmp_path) == {"permission": "allow"}
    assert "rebase on main first" in (tmp_path / "bc-worker.delivered").read_text()


def test_a_message_is_only_its_agents_and_a_secret_refusal_keeps_it_queued(tmp_path):
    queue("bc-worker", "hello", tmp_path)
    assert hook("ls", "bc-other", tmp_path) == {"permission": "allow"}
    refused = hook("env | cut -d= -f1", "bc-worker", tmp_path)
    assert refused["permission"] == "deny" and "hello" not in refused["user_message"]
    assert "hello" in hook("ls", "bc-worker", tmp_path)["user_message"]


def test_the_inbox_list_says_since_when_a_message_is_pending(tmp_path):
    queue("bc-worker", "first", tmp_path)
    queue("bc-worker", "second", tmp_path)
    queue("bc-done", "delivered", tmp_path)
    hook("ls", "bc-done", tmp_path)
    listed = subprocess.run([sys.executable, INBOX_TOOL, "list"], env={**os.environ, "AGENT_INBOX": str(tmp_path)},
                            check=True, capture_output=True, text=True).stdout.splitlines()
    stamp = (tmp_path / "bc-worker.md").read_text().splitlines()[0].strip("[]")
    assert any(x.startswith("pending   bc-worker ") and x.endswith(f"bytes  since {stamp}") for x in listed)
    assert any(x.startswith("delivered bc-done ") and "since" not in x for x in listed)


def test_an_unreadable_inbox_or_a_path_as_an_id_never_blocks_a_command(tmp_path):
    inbox = tmp_path / "not-a-dir"
    inbox.write_text("")
    assert hook("ls", "bc-worker", inbox) == {"permission": "allow"}
    assert hook("ls", "../etc/passwd", tmp_path) == {"permission": "allow"}


def test_one_agent_per_checkout_switching_another_agents_worktree_or_pushing_its_branch_is_refused(tmp_path):
    """A worker that switches its coordinator's checkout drags every editable install in it to that branch. A worktree belongs to the first
    agent that runs git in it; another agent may read it and check out files, but not switch its branch, and may not push a
    branch another agent created or holds, until it is handed over."""
    env = {"AGENT_GUARD_STATE": str(tmp_path / "state"), "AGENT_GUARD_MIN_AVAIL": "0", **GIT_ENV}
    ws, origin = tmp_path / "ws", tmp_path / "origin.git"
    for args in (["init", "-q", "--bare", "-b", "main", str(origin)], ["clone", "-q", str(origin), str(ws)]):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, env={**os.environ, **GIT_ENV})
    for args in (["commit", "-q", "--allow-empty", "-m", "a"], ["branch", "lane"], ["checkout", "-q", "-b", "coord"]):
        subprocess.run(["git", *args], cwd=ws, check=True, capture_output=True, env={**os.environ, **GIT_ENV})

    def ask(command, agent, cwd=ws):
        out = subprocess.run([sys.executable, HOOK], input=json.dumps({"hook_event_name": "beforeShellExecution", "command": command,
                                                                         "conversation_id": agent, "cwd": str(cwd)}),
                             capture_output=True, text=True, env={**os.environ, **env}, check=True).stdout
        return json.loads(out)

    assert ask("git status", "bc-coord")["permission"] == "allow"                   # the coordinator claims ws, and branch coord
    out = ask("git checkout lane", "bc-worker")
    assert out["permission"] == "deny" and "bc-coord's checkout" in out["agent_message"] and "worktree add" in out["agent_message"]
    for command in ("git switch lane", "git checkout -b mine"):
        assert ask(command, "bc-worker")["permission"] == "deny", command
    for command in (f"cd {ws} && git checkout lane", f"git -C {ws} switch -c x"):         # from outside, naming ws
        assert ask(command, "bc-worker", cwd=tmp_path)["permission"] == "deny", command
    for command in ("git checkout -- README", "git checkout lane -- README", "git log -1", "git diff"):
        assert ask(command, "bc-worker")["permission"] == "allow", command
    assert ask("git checkout lane", "bc-coord")["permission"] == "allow"
    out = ask("git push origin coord", "bc-worker")
    assert out["permission"] == "deny" and "belongs to agent bc-coord" in out["agent_message"]
    assert ask("git worktree add ../wt -b wbranch", "bc-worker")["permission"] == "allow"
    subprocess.run(["git", "worktree", "add", "-q", str(tmp_path / "wt"), "-b", "wbranch"], cwd=ws, check=True, capture_output=True,
                   env={**os.environ, **GIT_ENV})
    assert ask("git push -u origin wbranch", "bc-worker", cwd=tmp_path / "wt")["permission"] == "allow"
    assert ask("git push origin wbranch", "bc-coord")["permission"] == "deny"
    assert ask("git checkout lane", "bc-coord", cwd=tmp_path / "wt")["permission"] == "deny", "wt is the worker's"
    subprocess.run([sys.executable, str(ROOT / "tools/agent-guard/worktree_guard.py"), "hand", "coord", "bc-worker"], check=True,
                   capture_output=True, env={**os.environ, **env})
    assert ask("git push origin coord", "bc-worker")["permission"] == "allow"
    assert ask("git checkout lane", "")["permission"] == "allow", "no conversation id: nothing to judge"


def test_a_push_that_would_update_main_or_next_and_a_config_that_prints_tokens_are_refused_for_any_agent(tmp_path):
    """Only drive moves `main` and `next`, so an agent's push to either is refused; and git config values can hold a GitHub
    installation token, so printing them is refused."""
    env = {"AGENT_GUARD_STATE": str(tmp_path / "state"), "AGENT_GUARD_MIN_AVAIL": "0", **GIT_ENV}
    ws, origin = tmp_path / "ws", tmp_path / "origin.git"
    for args in (["init", "-q", "--bare", "-b", "main", str(origin)], ["clone", "-q", str(origin), str(ws)]):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, env={**os.environ, **GIT_ENV})
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "a"], cwd=ws, check=True, capture_output=True,
                   env={**os.environ, **GIT_ENV})

    def ask(command, agent="bc-w"):
        out = subprocess.run([sys.executable, HOOK], input=json.dumps({"hook_event_name": "beforeShellExecution", "command": command,
                                                                         "conversation_id": agent, "cwd": str(ws)}),
                             capture_output=True, text=True, env={**os.environ, **env}, check=True).stdout
        return json.loads(out)

    for command in ("git push", "git push origin", "git push origin main", "git push -u origin HEAD:main", "git push origin +main",
                    "git push origin HEAD:refs/heads/main", "git push --all origin", "git push --mirror origin",
                    "git push origin --delete main", "git push origin :main", f"git -C {ws} push origin main",
                    "git push origin next", "git push origin HEAD:refs/heads/next", "git push --force origin +HEAD:next"):
        out = ask(command)
        assert out["permission"] == "deny" and "HEAD:refs/heads/<your-branch>" in out["agent_message"], command
    assert ask("git push origin main", "")["permission"] == "deny", "whoever runs it"
    for command in ("git push origin HEAD:refs/heads/cursor/mine", "git push -u origin cursor/mine", "git log -1"):
        assert ask(command)["permission"] == "allow", command
    repair = "AGENT_GUARD_PUSH_NEXT_OK='restore next to abc under the queue lock' git push --force-with-lease=next:bad origin abc:next"
    assert ask(repair)["permission"] == "allow"
    assert json.loads((tmp_path / "state" / "push-next-ok.jsonl").read_text())["reason"] == "restore next to abc under the queue lock"
    for command in ("AGENT_GUARD_PUSH_NEXT_OK=x git push origin main", "AGENT_GUARD_PUSH_NEXT_OK=x git push origin next main",
                    "AGENT_GUARD_PUSH_NEXT_OK=x git push --all origin", "AGENT_GUARD_PUSH_NEXT_OK= git push origin next",
                    "AGENT_GUARD_PUSH_NEXT_OK=x true; git push origin next", "export AGENT_GUARD_PUSH_NEXT_OK=x; git push origin next"):
        assert ask(command)["permission"] == "deny", command
    for command in ("git config --list", "git config -l", "git config --get-regexp http", "git config --show-origin --list",
                    "git config --get http.https://github.com/.extraheader", "git config http.https://github.com/.extraheader",
                    "git config --name-only --list", "git config --name-only --show-origin --list",
                    "git config --name-only --get-regexp url"):
        out = ask(command)
        assert out["permission"] == "deny" and "url.*.insteadof's name" in out["agent_message"], command
    for command in ("git config user.name", "git config user.name t", "git config --get remote.origin.fetch"):
        assert ask(command)["permission"] == "allow", command
