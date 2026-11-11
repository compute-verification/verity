"""`.cursor/hooks.json`'s shell hook refuses commands that print the environment and lets everything else through."""
import importlib.util
import json
import re
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
HOOK = ROOT / "tools/agent-guard/shell_hook.py"
GIT_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


@pytest.fixture(autouse=True)
def no_inbox_poll(monkeypatch):
    """The hook's background pull would ask the real origin; a test that wants it sets AGENT_INBOX_POLL_S itself."""
    monkeypatch.setenv("AGENT_INBOX_POLL_S", "off")


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
    assert entry["command"].startswith("sh -c 'h=tools/agent-guard/hook.sh;") and entry["command"].endswith(" guard shell_hook.py")
    assert entry["failClosed"] is True and "matcher" not in entry


def guard_module(name: str, monkeypatch):
    monkeypatch.syspath_prepend(str(HOOK.parent))
    return importlib.import_module(name)


def test_cursor_reads_the_hooks_file_as_its_json_says(monkeypatch):
    """Cursor reads hooks.json as JSONC, stripping `//...` and `/*...*/` inside strings too. A cache glob `.../*/` in each
    command deleted the text between the shell hook's `/*` and the next `*/`: Cursor then ran no shell hook (no secret
    refusal, no inbox delivery) and no Task hook, and ran task_hook.py on shell commands that named Task."""
    inbox = guard_module("inbox", monkeypatch)
    text = (ROOT / ".cursor/hooks.json").read_text()
    assert inbox.as_cursor_reads(text) == json.loads(text)
    assert sorted(json.loads(text)["hooks"]) == ["beforeShellExecution", "preToolUse", "stop"]
    old = json.dumps({"version": 1, "hooks": {
        "beforeShellExecution": [{"command": "ls -td c/*/ | head -1; run shell_hook.py", "failClosed": True}],
        "preToolUse": [{"command": "ls -td c/*/ | head -1; run task_hook.py", "matcher": m}
                       for m in ("Task", "CallDynamicTool", "^MCP:")]}})
    assert inbox.as_cursor_reads(old) == {"version": 1, "hooks": {"beforeShellExecution": [
        {"command": "ls -td c | head -1; run task_hook.py", "matcher": m} for m in ("Task", "^MCP:")]}}


def test_the_shell_hook_judges_a_pretooluse_shell_event_like_a_beforeshellexecution_one():
    def ask(command):
        event = {"hook_event_name": "preToolUse", "tool_name": "Shell", "tool_input": {"command": command, "cwd": str(ROOT)}}
        out = subprocess.run([sys.executable, HOOK], input=json.dumps(event), capture_output=True, text=True, check=True,
                             env={**os.environ, "AGENT_GUARD_MIN_AVAIL": "0"}).stdout
        return json.loads(out)["permission"]

    assert ask("env | cut -d= -f1") == "deny" and ask("ls") == "allow"


def test_each_hook_run_records_its_events_key_names_and_never_a_value(event_log):
    W = "bc-44444444-4444-4444-8444-444444444444"
    subprocess.run([sys.executable, HOOK], input=json.dumps({"hook_event_name": "beforeShellExecution", "conversation_id": W,
                                                             "command": "echo not-to-be-copied", "cwd": "/"}),
                   capture_output=True, text=True, check=True)
    subprocess.run([sys.executable, HOOK], input="not json", capture_output=True, text=True, check=True)
    text = event_log.read_text()
    first, second = (json.loads(ln) for ln in text.splitlines())
    assert "not-to-be-copied" not in text
    assert (first["hook"], first["event"], first["conversation_id"]) == ("shell_hook.py", "beforeShellExecution", W)
    assert first["keys"] == ["command", "conversation_id", "cwd", "hook_event_name"]
    assert second["keys"] == "str" and second["conversation_id"] is None


def test_a_checkout_without_the_guard_runs_the_cached_copy_so_a_branch_switch_cant_lock_the_agent_out(tmp_path):
    """Cursor keeps the hooks it read at the session's start. A branch without tools/agent-guard (checked out later) would make
    every command, the checkout back included, fail the fail-closed hook; the command falls back to the cached origin/main copy."""
    (entry,) = json.loads((ROOT / ".cursor/hooks.json").read_text())["hooks"]["beforeShellExecution"]
    ws, cache = tmp_path / "ws", tmp_path / "cache"
    ws.mkdir()
    shutil.copytree(ROOT / "tools/agent-guard", cache / "0123abcd")

    def ask(command):
        event = json.dumps({"hook_event_name": "beforeShellExecution", "command": command, "cwd": str(ws)})
        out = subprocess.run(["sh", "-c", entry["command"]], input=event, cwd=ws, capture_output=True, text=True, check=True,
                             env={**os.environ, "AGENT_GUARD_CACHE": str(cache), "AGENT_GUARD_MIN_AVAIL": "0"}).stdout
        return json.loads(out)["permission"]

    assert ask("git checkout main") == "allow"
    assert ask("env | cut -d= -f1") == "deny"


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
    "ps -ef --forest | grep -v -E 'xfce|dbus' | tail -30",  # an inverted filter prints every other line
    "ps -eo pid,etime,args | grep -v kworker",
    "ps aux | grep -iv lake",
    "ps aux | rg --invert-match lake",
    "ps aux | grep --inv lake",
    "ssh n1 'ps -ef | grep -v grep'",
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
    "ps -ef | grep -v grep | grep \"[p]ython\"",
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
    for command in ("cd tools/research && python -m pytest -x", "cd t && uv run pytest tests; echo done"):
        advice = re.search(r"`(research run --on auto -- [^`]+)`", answer(command, **gate)["agent_message"]).group(1)
        assert advice.startswith("research run --on auto -- bash -lc ") and verdict(advice, **gate) == "allow", \
            f"the advice for a compound command passes the hook: {advice}"
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
    (tmp_path / "vmstat").write_text(f"balloon_inflate {3 * 2**20}\nballoon_deflate {2**20}\n")     # 8 GiB in the balloon
    assert verdict("uv run pytest -q tests/test_x.py", **gate) == "allow", \
        "memory the balloon holds is available: the VM gets it back as it asks"


def cgroup_gate(tmp_path: Path, stat: str | None) -> dict[str, str]:
    """The memory gate's overrides for a VM with most of its 15.3 GiB available, under a cgroup limit of 8 GiB with 7 GiB in use
    whose memory.stat is `stat` (None: there is none)."""
    meminfo, cgroup = tmp_path / "meminfo", tmp_path / "cg"
    cgroup.mkdir()
    meminfo.write_text("MemTotal:       16000000 kB\nMemFree:          100000 kB\nMemAvailable:   14000000 kB\n")
    (cgroup / "memory.max").write_text(f"{8 * 2**30}\n")
    (cgroup / "memory.current").write_text(f"{7 * 2**30}\n")
    if stat is not None:
        (cgroup / "memory.stat").write_text(stat)
    return {"AGENT_GUARD_MEMINFO": str(meminfo), "AGENT_GUARD_MIN_AVAIL": "0.25", "AGENT_GUARD_CGROUP": str(cgroup)}


def test_under_a_cgroup_limit_its_inactive_file_cache_counts_as_available(tmp_path):
    """Under a cgroup limit tighter than the VM, the headroom is taken over the cgroup's working set: memory.current less
    memory.stat's inactive_file, the file cache the kernel reclaims on demand (as container runtimes count it), so a cache that
    fills the cgroup doesn't refuse a one-file test run while most of its memory is free; active_file still counts as used."""
    gate = cgroup_gate(tmp_path, f"anon {2 * 2**30}\nfile {5 * 2**30}\ninactive_file {4 * 2**30}\nactive_file {2**30}\n")
    assert verdict("uv run pytest -q tests/test_x.py", **gate) == "allow"
    (tmp_path / "cg" / "memory.stat").write_text(f"anon {2 * 2**30}\nfile {5 * 2**30}\ninactive_file {2**29}\nactive_file {9 * 2**29}\n")
    out = answer("uv run pytest -q tests/test_x.py", **gate)
    assert out["permission"] == "deny" and "1.5 GiB of 8.0 GiB available" in out["agent_message"]


def test_under_a_cgroup_limit_without_memory_stat_all_of_memory_current_counts(tmp_path):
    """Where memory.stat can't be read, or names no inactive_file, the headroom is the limit less all of memory.current."""
    gate = cgroup_gate(tmp_path, None)
    out = answer("uv run pytest -q tests/test_x.py", **gate)
    assert out["permission"] == "deny" and "1.0 GiB of 8.0 GiB available" in out["agent_message"]
    (tmp_path / "cg" / "memory.stat").write_text(f"anon {7 * 2**30}\nfile 0\n")
    out = answer("uv run pytest -q tests/test_x.py", **gate)
    assert out["permission"] == "deny" and "1.0 GiB of 8.0 GiB available" in out["agent_message"]


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
    subprocess.run([sys.executable, INBOX_TOOL, "send", agent, "--local", "--text", text],
                   env={**os.environ, "AGENT_INBOX": str(inbox)}, check=True, capture_output=True)


def test_a_message_pushed_from_one_vm_is_pulled_and_delivered_on_another_once(tmp_path):
    """Two clones of one origin stand in for two VMs. The coordinator's `send` pushes to origin's `inbox/<agent>`; the
    worker's pull (what its hook starts) moves the messages into its VM's inbox and empties the branch, so the next command
    delivers them, and neither a repeated pull nor a third VM gets them again."""
    origin, coord, worker, other = (tmp_path / n for n in ("origin.git", "coord", "worker", "other"))
    env = {**os.environ, **GIT_ENV}
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True, capture_output=True, env=env)
    for clone in (coord, worker, other):
        subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True, capture_output=True, env=env)

    W = "bc-22222222-2222-4222-8222-222222222222"

    def tool(*args, inbox=tmp_path / "unused"):
        return subprocess.run([sys.executable, INBOX_TOOL, *args], env={**env, "AGENT_INBOX": str(inbox)}, check=True,
                              capture_output=True, text=True).stdout

    tool("send", W, "--text", "restack on main", "--repo", str(coord))
    tool("send", W, "--text", "then mark it ready", "--repo", str(coord))
    box = tmp_path / "worker-inbox"
    assert "2 new message(s)" in tool("pull", W, "--repo", str(worker), inbox=box)
    assert "0 new message(s)" in tool("pull", W, "--repo", str(worker), inbox=box)
    first = hook("ls", W, box)
    assert first["permission"] == "deny"
    assert "restack on main" in first["user_message"] and "then mark it ready" in first["user_message"]
    assert hook("ls", W, box) == {"permission": "allow"}
    assert "0 new message(s)" in tool("pull", W, "--repo", str(other), inbox=tmp_path / "other-inbox")
    tool("send", W, "--text", "and one more", "--repo", str(coord))
    assert "1 new message(s)" in tool("pull", W, "--repo", str(worker), inbox=box)
    assert "and one more" in hook("ls", W, box)["user_message"]


def test_a_sender_learns_when_a_target_hook_is_not_taking_its_messages_and_the_target_can_check_its_own_vm(tmp_path, monkeypatch):
    """One branch held every message for three hours while each `send` said only "pushed". `status` and a later `send` say a
    message has waited past STALE_S unpulled; `check` on the target's VM says whether its hook polls. Every `send` says a
    target whose turn ended (one finished before its stop messages reached it) takes nothing until resumed."""
    origin, coord, worker = (tmp_path / n for n in ("origin.git", "coord", "worker"))
    env = {**os.environ, **GIT_ENV}
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True, capture_output=True, env=env)
    for clone in (coord, worker):
        subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True, capture_output=True, env=env)
    W = "bc-33333333-3333-4333-8333-333333333333"

    def tool(*args, inbox=tmp_path / "unused", poll="30"):
        return subprocess.run([sys.executable, INBOX_TOOL, *args], env={**env, "AGENT_INBOX": str(inbox), "AGENT_INBOX_POLL_S": poll},
                              capture_output=True, text=True)

    assert "is empty" in tool("status", W, "--repo", str(coord)).stdout
    sent = tool("send", W, "--text", "fresh", "--repo", str(coord))
    assert "FINISHED or IDLE" in sent.stdout and "resume it with Task" in sent.stdout   # MEMPOOL: a finished target never pulls
    assert "earlier" not in sent.stderr
    second = tool("send", W, "--text", "fresh too", "--repo", str(coord))
    assert "1 earlier message(s) uncollected since" in second.stderr and "NOT DELIVERED" not in second.stderr
    fresh = tool("status", W, "--repo", str(coord))
    assert fresh.returncode == 0 and "within its hook's poll interval" in fresh.stdout
    spec = importlib.util.spec_from_file_location("inbox_tool", INBOX_TOOL)
    sys.path.insert(0, str(INBOX_TOOL.parent))
    try:
        inbox = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inbox)
    finally:
        sys.path.remove(str(INBOX_TOOL.parent))
    for k, v in GIT_ENV.items():
        monkeypatch.setenv(k, v)
    old = "[2026-01-01T00:00Z] abcdefabcdef\nthe 04:33Z heads-up\n\n"
    inbox.commit(coord, W, old, inbox.tip(coord, W), "an old message")
    stale = tool("status", W, "--repo", str(coord))
    assert stale.returncode == 1 and "NOT DELIVERED" in stale.stdout and "2026-01-01T00:00Z" in stale.stdout
    assert "resumed with Task" in stale.stdout
    again = tool("send", W, "--text", "one more", "--repo", str(coord))
    assert again.returncode == 0
    assert "NOT DELIVERED YET: 1 earlier message(s) uncollected since 2026-01-01T00:00Z: this agent's VM is not pulling" in again.stderr
    box = tmp_path / "worker-inbox"
    assert tool("pull", W, "--repo", str(worker), inbox=box).returncode == 0
    assert "is empty" in tool("status", W, "--repo", str(coord)).stdout
    shutil.copytree(ROOT / ".cursor", worker / ".cursor")
    off = tool("check", W, "--repo", str(worker), inbox=tmp_path / "quiet-inbox")
    assert off.returncode == 1 and "NOT POLLING" in off.stdout and f"inbox/{W}" in off.stdout
    (box / f"{W}.polled").touch()
    unseen = tool("check", W, "--repo", str(worker), inbox=box)
    assert unseen.returncode == 1 and f"PROBLEM: no shell hook has run for {W}" in unseen.stdout
    hook("ls", W, box)
    assert tool("check", W, "--repo", str(worker), inbox=box).returncode == 0
    assert "AGENT_INBOX_POLL_S is off" in tool("check", W, "--repo", str(worker), inbox=box, poll="off").stdout


def test_a_coordinators_self_check_sends_itself_a_message_its_next_commands_deliver_and_names_a_hooks_file_cursor_misreads(tmp_path):
    """`selftest` is delivery end to end: it pushes a message to the agent's own inbox branch and clears its poll mark, so its
    next command's hook pulls it and the one after is refused with it. It also says, at once, when Cursor reads no shell hook
    here or this agent's hook has never run."""
    origin, vm = tmp_path / "origin.git", tmp_path / "vm"
    env = {**os.environ, **GIT_ENV}
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True, capture_output=True, env=env)
    subprocess.run(["git", "clone", "-q", str(origin), str(vm)], check=True, capture_output=True, env=env)
    A, box = "bc-55555555-5555-4555-8555-555555555555", tmp_path / "inbox"
    shutil.copytree(ROOT / ".cursor", vm / ".cursor")
    box.mkdir()
    (box / f"{A}.polled").touch()

    def selftest(*args):
        return subprocess.run([sys.executable, INBOX_TOOL, "selftest", *args, "--repo", str(vm)], capture_output=True,
                              text=True, env={**env, "AGENT_INBOX": str(box), "CURSOR_CONVERSATION_ID": A})

    hook("true", A, box)
    ok = selftest()
    token = re.search(r"self-check ([0-9a-f]{8}) pushed", ok.stdout)[1]
    assert ok.returncode == 0 and "PROBLEM" not in ok.stdout, ok.stdout + ok.stderr
    assert not (box / f"{A}.polled").exists(), "the next command's hook pulls at once"
    subprocess.run([sys.executable, INBOX_TOOL, "pull", A, "--repo", str(vm)], check=True, capture_output=True,
                   env={**env, "AGENT_INBOX": str(box)})
    delivered = hook("true", A, box)
    assert delivered["permission"] == "deny" and f"inbox self-check {token}" in delivered["agent_message"]

    hooks = vm / ".cursor/hooks.json"
    hooks.write_text(hooks.read_text().replace("/[0-9a-f]*/", "/*/"))
    other = "bc-66666666-6666-4666-8666-666666666666"
    bad = selftest(other)
    assert bad.returncode == 1 and f"pushed to origin's inbox/{other}" in bad.stdout
    assert f"PROBLEM: Cursor reads {hooks.resolve()} otherwise than its JSON says" in bad.stdout
    assert "the steps beforeShellExecution, stop" in bad.stdout and "PROBLEM: Cursor reads no shell hook" in bad.stdout
    assert f"PROBLEM: no shell hook has run for {other}" in bad.stdout


STOP_HOOK = ROOT / "tools/agent-guard/stop_hook.py"


def stop(conversation: str, inbox: Path, status: str = "completed", roots: tuple[str, ...] = (), poll: str = "off") -> dict:
    event = {"hook_event_name": "stop", "conversation_id": conversation, "status": status, "loop_count": 0,
             "workspace_roots": list(roots)}
    out = subprocess.run([sys.executable, STOP_HOOK], input=json.dumps(event), capture_output=True, text=True, check=True,
                         env={**os.environ, **GIT_ENV, "AGENT_INBOX": str(inbox), "AGENT_INBOX_POLL_S": poll}).stdout
    return json.loads(out)


def test_a_message_still_queued_when_a_turn_completes_is_its_follow_up_once(tmp_path):
    """A worker that ends its turn with no further shell command never met the shell hook's refusal, so a message sent after
    its last command waited until it was resumed by hand. The stop hook hands it to Cursor as the turn's follow-up."""
    W = "bc-77777777-7777-4777-8777-777777777777"
    queue(W, "park the PR", tmp_path)
    assert stop(W, tmp_path, status="aborted") == {}
    assert stop("bc-other", tmp_path) == {}
    first = stop(W, tmp_path)
    assert "park the PR" in first["followup_message"] and "stop hook" in first["followup_message"]
    assert stop(W, tmp_path) == {}
    assert hook("ls", W, tmp_path) == {"permission": "allow"}, "delivered once, by whichever hook took it"
    assert "park the PR" in (tmp_path / f"{W}.delivered").read_text()


def test_the_stop_hook_pulls_a_message_pushed_from_another_vm_first(tmp_path):
    origin, coord, worker = (tmp_path / n for n in ("origin.git", "coord", "agent/repos/verity"))
    env = {**os.environ, **GIT_ENV}
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True, capture_output=True, env=env)
    for clone in (coord, worker):
        subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True, capture_output=True, env=env)
    W = "bc-88888888-8888-4888-8888-888888888888"
    subprocess.run([sys.executable, INBOX_TOOL, "send", W, "--text", "stop and report", "--repo", str(coord)], check=True,
                   capture_output=True, env=env)
    got = stop(W, tmp_path / "inbox", roots=(str(tmp_path / "agent"),), poll="30")
    assert "stop and report" in got["followup_message"]


def test_the_hook_starts_a_pull_at_most_once_per_interval_and_never_when_off(tmp_path):
    def ask(every):
        out = subprocess.run([sys.executable, HOOK], input=json.dumps({"hook_event_name": "beforeShellExecution", "command": "ls",
                                                                         "conversation_id": "bc-w", "cwd": str(tmp_path / "nowhere")}),
                             capture_output=True, text=True, check=True,
                             env={**os.environ, "AGENT_INBOX": str(tmp_path), "AGENT_INBOX_POLL_S": every}).stdout
        return json.loads(out)

    mark = tmp_path / "bc-w.polled"
    assert ask("off") == {"permission": "allow"} and not mark.exists()
    assert ask("30") == {"permission": "allow"} and mark.exists() and (tmp_path / "bc-w.pull.log").exists()
    os.utime(mark, (10**10, 10**10))
    assert ask("30") == {"permission": "allow"} and mark.stat().st_mtime == 10**10, "polled again within the interval"


def test_the_pull_runs_from_the_checkout_under_a_root_that_is_not_one(tmp_path):
    """A Task-launched cloud agent's workspace root is /agent, its checkout /agent/repos/verity: a pull from the root itself
    fails in git, so that agent's messages never arrive."""
    (tmp_path / "agent/repos/verity/.git").mkdir(parents=True)
    (tmp_path / "ws/.git").mkdir(parents=True)

    def checkout(root, cwd):
        return subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import shell_hook; "
                               "print(shell_hook.checkout(*sys.argv[2:]))", str(HOOK.parent), root, cwd],
                              capture_output=True, text=True, check=True).stdout.strip()

    assert checkout(str(tmp_path / "agent"), "/tmp") == str(tmp_path / "agent/repos/verity")
    assert checkout(str(tmp_path / "ws"), "/tmp") == str(tmp_path / "ws")
    assert checkout(str(tmp_path / "empty"), "/tmp") == "/tmp"
    assert checkout("", "/tmp") == "/tmp"


def test_a_queued_message_refuses_the_agents_next_command_once_with_the_message_as_its_text(tmp_path):
    queue("bc-11111111-1111-4111-8111-111111111111", "rebase on main first", tmp_path)
    queue("bc-11111111-1111-4111-8111-111111111111", "then mark nothing", tmp_path)
    first = hook("ls", "bc-11111111-1111-4111-8111-111111111111", tmp_path)
    assert first["permission"] == "deny"
    assert "re-run it exactly as it was" in first["user_message"]
    assert "rebase on main first" in first["user_message"] and "then mark nothing" in first["user_message"]
    assert hook("ls", "bc-11111111-1111-4111-8111-111111111111", tmp_path) == {"permission": "allow"}
    assert "rebase on main first" in (tmp_path / "bc-11111111-1111-4111-8111-111111111111.delivered").read_text()


def test_a_message_is_only_its_agents_and_a_secret_refusal_keeps_it_queued(tmp_path):
    queue("bc-11111111-1111-4111-8111-111111111111", "hello", tmp_path)
    assert hook("ls", "bc-22222222-2222-4222-8222-222222222222", tmp_path) == {"permission": "allow"}
    refused = hook("env | cut -d= -f1", "bc-11111111-1111-4111-8111-111111111111", tmp_path)
    assert refused["permission"] == "deny" and "hello" not in refused["user_message"]
    assert "hello" in hook("ls", "bc-11111111-1111-4111-8111-111111111111", tmp_path)["user_message"]


def test_the_inbox_list_says_since_when_a_message_is_pending(tmp_path):
    queue("bc-11111111-1111-4111-8111-111111111111", "first", tmp_path)
    queue("bc-11111111-1111-4111-8111-111111111111", "second", tmp_path)
    queue("bc-33333333-3333-4333-8333-333333333333", "delivered", tmp_path)
    hook("ls", "bc-33333333-3333-4333-8333-333333333333", tmp_path)
    listed = subprocess.run([sys.executable, INBOX_TOOL, "list"], env={**os.environ, "AGENT_INBOX": str(tmp_path)},
                            check=True, capture_output=True, text=True).stdout.splitlines()
    stamp = (tmp_path / "bc-11111111-1111-4111-8111-111111111111.md").read_text().splitlines()[0].strip("[]")
    assert any(x.startswith("pending   bc-11111111-1111-4111-8111-111111111111 ") and x.endswith(f"bytes  since {stamp}") for x in listed)
    assert any(x.startswith("delivered bc-33333333-3333-4333-8333-333333333333 ") and "since" not in x for x in listed)


def test_send_refuses_an_id_the_hook_would_never_deliver_to(tmp_path):
    """The hook reads the whole conversation id, so a message queued under an id prefix was never delivered."""
    inbox, repo = tmp_path / "inbox", tmp_path / "not-a-checkout"
    repo.mkdir()
    for agent in ("bc-prefix", "bc-worker", "../etc/passwd", ""):
        for where in (["--local"], ["--repo", str(repo)]):
            p = subprocess.run([sys.executable, INBOX_TOOL, "send", agent, "--text", "x", *where], capture_output=True,
                               text=True, env={**os.environ, "AGENT_INBOX": str(inbox)})
            assert p.returncode != 0 and "full agent id" in p.stderr, (agent, where)
    assert not inbox.exists() and not any(repo.iterdir())


def test_an_unreadable_inbox_or_a_path_as_an_id_never_blocks_a_command(tmp_path):
    inbox = tmp_path / "not-a-dir"
    inbox.write_text("")
    assert hook("ls", "bc-11111111-1111-4111-8111-111111111111", inbox) == {"permission": "allow"}
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
    hand = "python3 tools/agent-guard/worktree_guard.py hand coord bc-third"
    out = ask(hand, "bc-coord")
    assert out["permission"] == "deny" and "belongs to agent bc-worker" in out["agent_message"], "only the owner hands a branch on"
    assert ask(hand, "bc-worker")["permission"] == "allow"
    assert ask("python3 tools/agent-guard/worktree_guard.py hand unregistered bc-third", "bc-coord")["permission"] == "allow"
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
    for command in ("B=main; git push origin HEAD:refs/heads/$B", 'git push origin "HEAD:$B"', "git push origin $(echo main)",
                    "git -C $D push origin cursor/mine", "git switch $B", "git checkout -b ${B}x", "git worktree add ../wt -b $B"):
        out = ask(command)
        assert out["permission"] == "deny" and "expands only after the guard" in out["agent_message"], command
    assert ask("git push origin `echo cursor/mine`")["permission"] == "deny"
    assert ask("git push origin HEAD:refs/heads/$B", "")["permission"] == "deny", "whoever runs it"
    for command in ('git checkout -- "$F"', 'git log -1 "$REV"', "git push origin HEAD:refs/heads/cursor/mine && echo $?"):
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


def test_a_merge_on_a_vm_that_never_ran_boot_sh_first_gets_the_record_merge_drivers(tmp_path):
    """A cloud worker's VM never runs boot.sh, so its trial `git merge-tree` of a PR onto `next` merged each lean-audit.json
    line by line and showed 359 conflicts the queue never saw."""
    ws = tmp_path / "ws"
    clone_py = ws / "tools/research/src/research/clone.py"
    clone_py.parent.mkdir(parents=True)
    clone_py.write_text('RECORD_DRIVER = {"merge.lean-audit.name": "n", "merge.lean-audit.driver": "python3 m.py %O %A %B %P"}\n')
    subprocess.run(["git", "init", "-q", str(ws)], check=True, env={**os.environ, **GIT_ENV})

    def ask(command: str) -> str:
        out = subprocess.run([sys.executable, HOOK], input=json.dumps({"hook_event_name": "beforeShellExecution",
                                                                         "command": command, "cwd": str(ws)}),
                             capture_output=True, text=True, env={**os.environ, **GIT_ENV, "AGENT_GUARD_MIN_AVAIL": "0"},
                             check=True).stdout
        return json.loads(out)["permission"]

    def driver() -> str:
        return subprocess.run(["git", "-C", str(ws), "config", "--get", "merge.lean-audit.driver"], capture_output=True,
                              text=True, env={**os.environ, **GIT_ENV}).stdout.strip()

    assert ask("git status") == "allow" and driver() == "", "only a merge sets them"
    assert ask("git merge-tree --write-tree --name-only origin/next FETCH_HEAD") == "allow"
    assert driver() == "python3 m.py %O %A %B %P"
    subprocess.run(["git", "-C", str(ws), "config", "merge.lean-audit.driver", "mine"], check=True, env={**os.environ, **GIT_ENV})
    assert ask("git merge origin/main") == "allow" and driver() == "mine", "without origin/main, one already set stays"


def test_a_merge_sets_origin_main_s_merge_drivers_over_an_older_command(tmp_path):
    """Git runs a driver from the worktree of the side merged into, and a VM keeps the command it was given: one whose
    command ran that worktree's merge.py merged main's lock into an old branch by the branch's older copy, which collapsed
    Security's lock to one `reads` group.  So where the repository has origin/main, each record driver whose value differs
    from origin/main's RECORD_DRIVER is set to it, whatever the checkout's own clone.py says."""
    ws = tmp_path / "ws"
    clone_py = ws / "tools/research/src/research/clone.py"
    clone_py.parent.mkdir(parents=True)
    env = {**os.environ, **GIT_ENV}

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(ws), *args], capture_output=True, text=True, env=env, check=True).stdout.strip()
    git("init", "-q")
    clone_py.write_text('RECORD_DRIVER = {"merge.lean-audit.name": "n", "merge.lean-audit.driver": "python3 main.py %O %A %B %P"}\n')
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "main")
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    clone_py.write_text('RECORD_DRIVER = {"merge.lean-audit.driver": "python3 worktree.py %O %A %B %P"}\n')
    git("config", "merge.lean-audit.driver", "python3 old.py %O %A %B %P")
    git("config", "merge.lean-audit.name", "n")
    out = subprocess.run([sys.executable, HOOK], input=json.dumps({"hook_event_name": "beforeShellExecution",
                                                                     "command": "git merge --no-edit origin/main",
                                                                     "cwd": str(ws)}),
                         capture_output=True, text=True, env={**env, "AGENT_GUARD_MIN_AVAIL": "0"}, check=True)
    assert json.loads(out.stdout)["permission"] == "allow", out.stderr
    assert git("config", "--get", "merge.lean-audit.driver") == "python3 main.py %O %A %B %P"
