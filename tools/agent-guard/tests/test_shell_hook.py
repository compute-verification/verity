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
    "env -u PYTHONPATH uv sync",
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
    "ps -o pid,etime,comm --ppid 4242",
    "ps -u research",
    "ssh n1 'ps -eo pid,args | grep \"[s]lot.py\"'",
    "pgrep -l quick_tier",
    "tmux show-environment -g DISPLAY",
    "tmux -f /exec-daemon/tmux.portal.conf send-keys -t build:0.0 'make' C-m",
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
    assert f"PROBLEM: no shell hook has run for {W}" in off.stdout, "neither an event of it nor a poll mark"
    (box / f"{W}.polled").touch()
    polled = tool("check", W, "--repo", str(worker), inbox=box)
    # hook.sh runs origin/main's copy of the hook: one from before the event log polls and delivers, and records no event
    assert polled.returncode == 0 and "PROBLEM" not in polled.stdout and "hook polls for" in polled.stdout, polled.stdout
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


def test_guard_is_one_line_saying_what_is_off_why_and_the_fix_only_where_no_shell_hook_has_run_for_the_agent(tmp_path, event_log):
    """A VM whose branch's hooks.json Cursor can't load runs no shell hook: no secret refusal, no inbox, and nothing says so.
    `guard` prints nothing where a shell hook has run for the agent (its event, or the poll mark a copy from before the event
    log leaves), and otherwise one line: what is off, how Cursor misreads which file, and the fix, which depends on whether
    origin/main's file loads. The Task hook's event counts for nothing: from a misread file Cursor runs it on shell commands."""
    origin, vm, box = tmp_path / "origin.git", tmp_path / "vm", tmp_path / "inbox"
    env = {**os.environ, **GIT_ENV}
    A, other = "bc-77777777-7777-4777-8777-777777777777", "bc-99999999-9999-4999-8999-999999999999"

    def git(*args):
        subprocess.run(["git", "-C", str(vm), *args], check=True, capture_output=True, env=env)

    def guard(*args, repo=vm):
        return subprocess.run([sys.executable, INBOX_TOOL, "guard", *args, "--repo", str(repo)], capture_output=True, text=True,
                              env={**env, "AGENT_INBOX": str(box), "CURSOR_CONVERSATION_ID": A})

    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True, capture_output=True, env=env)
    subprocess.run(["git", "clone", "-q", str(origin), str(vm)], check=True, capture_output=True, env=env)
    hooks = vm / ".cursor/hooks.json"
    hooks.parent.mkdir()
    loads = (ROOT / ".cursor/hooks.json").read_text()
    misread = loads.replace("/[0-9a-f]*/", "/*/")
    assert misread != loads
    hooks.write_text(loads)
    git("add", ".cursor")
    git("commit", "-q", "-m", "hooks Cursor reads as written")
    git("push", "-q", "origin", "HEAD:main")
    git("fetch", "-q", "origin")

    hooks.write_text(misread)                 # the branch this VM booted on predates main's fix
    off = guard()
    assert off.returncode == 1 and off.stdout.count("\n") == 1, off.stdout + off.stderr
    assert off.stdout.startswith(f"no shell hook has run for {A} on this VM, so the guard that refuses a command printing a "
                                 "secret and the inbox are not running here.")
    assert f"Cursor reads {hooks.resolve()} otherwise than its JSON says" in off.stdout
    assert "Merge origin/main, whose hooks.json loads, and push: a VM booted from the branch after that runs them" in off.stdout
    (vm / "sub").mkdir()
    assert guard(A, repo=vm / "sub").stdout == off.stdout, "the checkout's file, from anywhere inside it"

    event_log.write_text(json.dumps({"hook": "task_hook.py", "event": "beforeShellExecution", "conversation_id": A}) + "\n"
                         + json.dumps({"hook": "shell_hook.py", "event": "beforeShellExecution", "conversation_id": other}) + "\n")
    assert guard().stdout == off.stdout, "the Task hook's event, and another agent's shell hook, show nothing of this one's"

    git("commit", "-q", "-am", "main's file doesn't load either")
    git("push", "-q", "origin", "HEAD:main")
    git("fetch", "-q", "origin")
    both = guard()
    assert both.returncode == 1 and "origin/main's does not load either, as of this checkout's last fetch" in both.stdout
    assert "Merge origin/main, whose" not in both.stdout

    hooks.write_text(loads)
    late = guard()
    assert late.returncode == 1 and "this branch's loads as written now, so it changed after this VM booted" in late.stdout
    (tmp_path / "nowhere").mkdir()
    lost = guard(repo=tmp_path / "nowhere")
    assert lost.returncode == 1 and lost.stdout.count("\n") == 1 and "in your checkout for why, and the fix" in lost.stdout

    box.mkdir()
    (box / f"{A}.polled").touch()
    assert (guard().returncode, guard().stdout) == (0, ""), "a hook from before the event log polls, and records no event"
    (box / f"{A}.polled").unlink()
    assert guard().returncode == 1
    hook("true", A, box)
    assert (guard().returncode, guard().stdout) == (0, ""), "this hook's own event"


def test_send_wait_tells_the_sender_whether_a_vm_took_this_message(tmp_path, monkeypatch, capsys):
    """`send` says "pushed" whether or not anything will ever pull the message, and its warning that a target isn't pulling
    comes only with a later send, minutes on. `send --wait S` looks at origin until a VM has pulled this message: taken, and
    after how long, or NOT TAKEN and exit 1 after S seconds, when the sender has to stop the work itself. The clock and the
    sleep are the test's: the worker's pull happens inside the sender's second sleep."""
    origin, coord, worker, box = (tmp_path / n for n in ("origin.git", "coord", "worker", "worker-inbox"))
    env = {**os.environ, **GIT_ENV}
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True, capture_output=True, env=env)
    for clone in (coord, worker):
        subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True, capture_output=True, env=env)
    for k, v in GIT_ENV.items():
        monkeypatch.setenv(k, v)
    inbox = guard_module("inbox", monkeypatch)
    W = "bc-88888888-8888-4888-8888-888888888888"
    now, naps, pull_in_nap = [0.0], [], []

    def sleep(seconds):
        naps.append(seconds)
        now[0] += seconds
        if len(naps) in pull_in_nap:
            inbox.pull(W, worker, box)

    def send(text, wait):
        code = inbox.main(["send", W, "--text", text, "--wait", wait, "--repo", str(coord)], sleep=sleep, clock=lambda: now[0])
        return code, capsys.readouterr()

    code, said = send("pause: nobody pulls this", "12")
    assert code == 1 and naps == [5.0, 5.0, 2.0], naps
    assert f"pushed for {W}" in said.out and "taken" not in said.out
    assert "inbox: NOT TAKEN in 12 s: no VM pulled" in said.err and "cancel its runs yourself" in said.err
    assert inbox.unpulled(coord, W)[0] == 1, "it stays on the branch for a later pull"

    del naps[:]
    pull_in_nap.append(2)
    code, said = send("pause: the worker's hook pulls this", "60")
    assert code == 0 and naps == [5.0, 5.0], naps
    assert f"inbox: taken 10 s after the push: {W}'s VM pulled" in said.out and "Taken is not acted on" in said.out
    assert "NOT TAKEN" not in said.err and inbox.unpulled(coord, W)[0] == 0
    delivered = hook("ls", W, box)
    assert delivered["permission"] == "deny" and "pause: the worker's hook pulls this" in delivered["user_message"]

    for wait in ("0", "1201"):
        with pytest.raises(SystemExit, match="--wait .* is not within 0 to 1200 seconds"):
            send("never pushed", wait)
    assert inbox.unpulled(coord, W)[0] == 0
    del naps[:]
    assert inbox.wait_taken(tmp_path / "no-checkout", W, "0" * 12, 5, sleep, lambda: now[0]) is None and naps == [5.0]
    assert "could not read origin's inbox" in capsys.readouterr().err, "a look git couldn't make is not a message taken"


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
    env = {"AGENT_GUARD_STATE": str(tmp_path / "state"), "AGENT_GUARD_MIN_AVAIL": "0", **GIT_ENV,
           "AGENT_GUARD_PUSH_REMOTES": str(tmp_path / "origin.git")}            # origin stands in for the queue's repository
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


#: a GitHub installation token's shape, from parts, as the URL a Cursor VM's insteadOf rewrites to carries one
TOKEN = "gh" "s_" "dummyDUMMYdummy0123456789abcdefABCDEF"
QUEUE_URLS = ["https://github.com/compute-verification/verity.git", "https://github.com/compute-verification/verity",
              "git@github.com:compute-verification/verity.git", "ssh://git@github.com/compute-verification/verity",
              f"https://x-access-token:{TOKEN}@github.com/compute-verification/verity", "https://github.com/compute-verification/infra.git"]
OTHER_URLS = ["https://github.com/compute-verification/research-notes.git", "https://github.com/someone/verity-fork",
              "https://github.com/compute-verification/verity-notes", "https://github.com/someone/compute-verification/verity"]


def test_the_push_rules_cover_the_queues_repositories_and_no_other(tmp_path):
    """`main`, `checked` and `next` are the queue's branches of verity and infra, and the registry names a branch without its
    repository; every agent's `research notes sync` pushes the notes' `main`, and a lane's notes branch may share a name with a
    registered verity branch. So a typed push to any other repository meets neither the three branches' rule nor the owner rule."""
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "branches.json").write_text(json.dumps({"lanes/infra": "bc-other"}))
    ws = tmp_path / "notes"
    for args in (["init", "-q", "-b", "main", str(ws)], ["-C", str(ws), "commit", "-q", "--allow-empty", "-m", "a"],
                 ["-C", str(ws), "remote", "add", "origin", OTHER_URLS[0]],
                 *(["-C", str(ws), "remote", "add", f"q{i}", u] for i, u in enumerate(QUEUE_URLS)),
                 *(["-C", str(ws), "remote", "add", f"o{i}", u] for i, u in enumerate(OTHER_URLS))):
        subprocess.run(["/usr/bin/git", *args], check=True, capture_output=True, env={**os.environ, **GIT_ENV})
    env = {k: v for k, v in os.environ.items() if k != "AGENT_GUARD_PUSH_REMOTES"}

    def ask(command):
        event = {"hook_event_name": "beforeShellExecution", "command": command, "conversation_id": "bc-me", "cwd": str(ws)}
        out = subprocess.run([sys.executable, HOOK], input=json.dumps(event), capture_output=True, text=True, check=True,
                             env={**env, **GIT_ENV, "AGENT_GUARD_STATE": str(tmp_path / "state"), "AGENT_GUARD_MIN_AVAIL": "0"})
        return json.loads(out.stdout)

    for command in ("git push origin main", "git push", "git push origin HEAD:refs/heads/next", "git push origin lanes/infra",
                    "git push origin HEAD:refs/heads/$B", *(f"git push o{i} main" for i in range(len(OTHER_URLS))),
                    *(f"git push o{i} lanes/infra" for i in range(len(OTHER_URLS)))):
        assert ask(command) == {"permission": "allow"}, command
    for i in range(len(QUEUE_URLS)):
        for command, said in ((f"git push q{i} main", "only drive and the lander move"),
                              (f"git push q{i} HEAD:refs/heads/checked", "only drive and the lander move"),
                              (f"git push q{i} lanes/infra", "belongs to agent bc-other"),
                              (f"git push q{i} HEAD:refs/heads/$B", "expands only after")):
            out = ask(command)
            assert out["permission"] == "deny" and said in out["agent_message"] and TOKEN not in out["agent_message"], command
    for command in ("git push https://github.com/compute-verification/verity.git main", "git push $R main"):
        assert ask(command)["permission"] == "deny", command


@pytest.fixture
def clones(tmp_path):
    """The hook's answer to a command agent bc-me types in a clone of verity, of infra or of the notes (each origin's URL;
    nothing is fetched or pushed), each on `main` with branches `lane/mine` and `theirs`, `theirs` bc-other's."""
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "branches.json").write_text(json.dumps({"theirs": "bc-other"}))
    urls = {"verity": QUEUE_URLS[0], "infra": QUEUE_URLS[5], "notes": OTHER_URLS[0]}
    for name, url in urls.items():
        d = str(tmp_path / name)
        for args in (["init", "-q", "-b", "main", d], ["-C", d, "commit", "-q", "--allow-empty", "-m", "a"],
                     ["-C", d, "branch", "lane/mine"], ["-C", d, "branch", "theirs"], ["-C", d, "remote", "add", "origin", url]):
            subprocess.run(["/usr/bin/git", *args], check=True, capture_output=True, env={**os.environ, **GIT_ENV})
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AGENT_GUARD_", "GIT_CONFIG"))}
    env["AGENT_GUARD_EVENT_LOG"] = str(tmp_path / "events.jsonl")     # the hook's record of this test's runs, not the machine's

    def ask(form: str, clone: str) -> dict:
        command = form
        for key, value in (("{VURL}", urls["verity"]), ("{NURL}", urls["notes"]), ("{V}", tmp_path / "verity"),
                           ("{N}", tmp_path / "notes")):
            command = command.replace(key, str(value))
        event = {"hook_event_name": "beforeShellExecution", "command": command, "conversation_id": "bc-me",
                 "cwd": str(tmp_path / clone)}
        out = subprocess.run([sys.executable, HOOK], input=json.dumps(event), capture_output=True, text=True, check=True,
                             env={**env, **GIT_ENV, "AGENT_GUARD_STATE": str(tmp_path / "state"), "AGENT_GUARD_MIN_AVAIL": "0"})
        return json.loads(out.stdout)
    return ask


#: pushes of `main` or of bc-other's branch with options, push's or git's own, that take a value in the word after them or in
#: their own: each goes to the clone's origin
VALUED_OPTION_PUSHES = [
    "git push -o ci.skip origin main", "git push --push-option x origin main", "git push --push-option=x origin main",
    "git push --repo origin", "git push --repo=origin", "git -C . push -o x origin HEAD:main",
    "git push --receive-pack git-receive-pack origin main", "git push --exec git-receive-pack origin main",
    "git push --recurse-submodules check origin main", "git push -uo x origin main", "git push -oci.skip origin main",
    "git push --push-opt x origin main", "git push origin main -o x", "git push -o x origin theirs",
    "bash -c 'git push -o ci.skip origin main'", "tmux send-keys 'git push --push-option x origin main' C-m",
    "tmux send-keys -t w 'git push --receive-pack P origin main' Enter", "bash -lc 'git push -o x origin theirs'",
    "git --attr-source HEAD push origin main", "git --config-env user.name=HOME push origin main",
]


@pytest.mark.parametrize("form", VALUED_OPTION_PUSHES)
def test_a_pushs_options_that_take_a_value_are_read_as_git_reads_them(clones, form):
    """The typed rule took the first word that isn't an option for the remote, so `git push -o ci.skip origin main` read as a
    push to a repository `ci.skip`, not verity, and passed; `main`'s guard, which judges every push of `main`, refused it."""
    for clone in ("verity", "infra"):
        out = clones(form, clone)
        assert out["permission"] == "deny" and ("only drive and the lander move" in out["agent_message"]
                                                or "belongs to agent bc-other" in out["agent_message"]), (form, clone, out)
    assert clones(form, "notes") == {"permission": "allow"}, form


def test_a_repo_option_names_the_repository_unless_a_word_does(clones):
    for clone in ("verity", "infra", "notes"):
        assert clones("git push --repo={VURL}", clone)["permission"] == "deny", clone
        assert clones("git push --repo {VURL}", clone)["permission"] == "deny", clone
        assert clones("git push --repo={NURL}", clone) == {"permission": "allow"}, clone
    assert clones("git push --repo={VURL} origin main", "notes") == {"permission": "allow"}, "git pushes to origin, the notes"


@pytest.mark.parametrize("form, clone", [
    ("git push nosuchremote main", "notes"),                             # neither a remote nor a URL nor a path
    ("git push --repo origin main", "notes"),                            # git takes `main` for the repository
    ("git push --frobnicate x origin main", "notes"),                    # no option of git push
    ("GIT_DIR={V}/.git git push origin main", "notes"),
    ("GIT_DIR={N}/.git git push origin main", "notes"),                  # named by GIT_DIR, whichever it is
    ("GIT_WORK_TREE={N} git push origin main", "notes"),
    ("env GIT_DIR={V}/.git git push origin main", "notes"),
    ("export GIT_DIR={V}/.git; git push origin main", "notes"),
    ("git push origin main; export GIT_DIR={V}/.git", "notes"),           # a loop or a function may run it first
    ("git --git-dir={V}/.git push origin main", "notes"),
    ("git --git-dir {V}/.git push origin main", "notes"),
    ("git --work-tree={N} push origin main", "notes"),
    ("git -c remote.origin.pushurl={VURL} push origin main", "notes"),
])
def test_a_push_whose_repository_the_guard_cant_tell_is_judged_as_one_to_the_queues(clones, form, clone):
    """Where the words may be misread, or the command names its repository or changes the configuration git reads, the push is
    judged as one to verity or infra, never passed; the refusal says how to name the repository so the guard can tell it."""
    out = clones(form, clone)
    assert out["permission"] == "deny" and "only drive and the lander move" in out["agent_message"], (form, out)


#: commands with a `cd` that never runs, or that runs only where the push doesn't: the shell pushes from the clone it started in
NEVER_RUN_CDS = [
    "true || cd {T}; git push origin main", "false && cd {T}; git push origin main", "if false; then cd {T}; fi; git push origin main",
    "while false; do cd {T}; done; git push origin main", "f() { cd {T}; }; git push origin main",
]
#: and commands whose `cd` the guard would have to follow, and doesn't
UNFOLLOWED_CDS = [
    *(f.replace("cd {T}", "pushd {T}") for f in NEVER_RUN_CDS),
    "function f { cd {T}; }; git push origin main", "(cd {T} && true); git push origin main", "bash -c 'cd {T}'; git push origin main",
    "cd {T} | true; git push origin main", "case x in x) cd {T};; esac; git push origin main", "C=cd; $C {T}; git push origin main",
    "git push origin main; cd {T}", "popd; git push origin main", ". ./env.sh; git push origin main",
    "cd {T} && git push origin main", "cd {T}; git push --repo=origin", "(cd {T} && git push origin main)",
    "cd {T} && git push origin HEAD:main", "cd {T} && git -C . push origin main", "cd {T}/.. && git -C ../.. push origin main",
    "cd {T} && git -C ~ push origin main", "source ./env.sh && git -C {T} push origin main", ". ./env.sh; git -C {T} push origin main",
    "C=cd; $C /tmp; git -C {T} push origin main", "cd /tmp && GIT_DIR={T}/.git git -C {T} push origin main",
    "cd /tmp && git --git-dir={T}/.git -C {T} push origin main",
]
#: and commands whose git names its checkout by an absolute `-C`: it runs there wherever a `cd`, `pushd` or `popd` leaves the shell
ABSOLUTE_C = [
    "cd {T} && git -C {T} push origin main", "cd /tmp && git -C {T} push origin main", "cd /tmp; git -C {T} push origin HEAD:main",
    "pushd /tmp && git -C {T} push origin main && popd", "(cd /tmp && git -C {T} push origin main)", "popd; git -C {T} push origin main",
    "false && cd /tmp; git -C {T} push origin main", "cd /tmp && git -C /nowhere -C {T} push origin main",
    "cd /tmp && git -C {T}/.. -C {T} push origin main", "cd /tmp && git -c x.y=z -C {T} push -o x origin main",
    "bash -c 'cd /tmp && git -C {T} push origin main'", "tmux send-keys -t w 'cd /tmp && git -C {T} push origin main' C-m",
]


@pytest.mark.parametrize("form", NEVER_RUN_CDS + UNFOLLOWED_CDS)
def test_a_cd_anywhere_in_the_command_leaves_a_push_unplaced(clones, form):
    """The rule followed a `cd` as if it held, so `false && cd NOTES; git push origin main` in a verity clone passed as a push to
    the notes, and the shell pushed verity's `main`; `main`'s guard, which judges every push of `main`, refused it. The guard
    now places a push by the hook's cwd, git's `-C` and a URL alone: any `cd` in the command, run or not, leaves it unplaced,
    and a push it can't place is judged as one to verity or infra, its current branch any branch."""
    for clone, target in (("verity", "{N}"), ("infra", "{N}"), ("notes", "{V}")):
        out = clones(form.replace("{T}", target), clone)
        assert out["permission"] == "deny" and "with no `cd` in the command, `git -C <checkout> push" in out["agent_message"], (
            form, clone, out)


@pytest.mark.parametrize("form", ABSOLUTE_C)
def test_an_absolute_git_C_places_a_push_whatever_cd_the_command_holds(clones, form):
    """`cd /tmp && git -C /abs/checkout push ...` was refused as unplaced, though git runs in `/abs/checkout` wherever the shell
    is. It is now judged as the push it is: of the notes' `main` it passes from any clone, of verity's it is refused as `main`'s,
    with no word about a `cd`."""
    for clone in ("verity", "infra", "notes"):
        assert clones(form.replace("{T}", "{N}"), clone) == {"permission": "allow"}, (form, clone)
        out = clones(form.replace("{T}", "{V}"), clone)
        assert out["permission"] == "deny" and "only drive and the lander move" in out["agent_message"], (form, clone, out)
        assert "The guard judged it as a push to verity or infra" not in out["agent_message"], (form, clone, out)


def test_an_absolute_git_C_places_the_checkout_a_push_or_switch_runs_in(tmp_path):
    """The push an agent types from a tool shell that starts elsewhere, `cd /tmp && git -C WT push -u origin HEAD`, pushes WT's
    branch: placed by its absolute `-C`, it is held to that branch's owner like any other, and a branch switch to WT's owner.
    A relative `-C`, a `source`, a `.`, a command or a `-C` the shell names only later, and a repository named by `GIT_DIR` or
    `--git-dir` leave it unplaced, as before."""
    env = {"AGENT_GUARD_STATE": str(tmp_path / "state"), "AGENT_GUARD_MIN_AVAIL": "0", **GIT_ENV,
           "AGENT_GUARD_PUSH_REMOTES": str(tmp_path / "origin.git")}            # origin stands in for the queue's repository
    ws, wt, origin = tmp_path / "ws", tmp_path / "wt", tmp_path / "origin.git"
    for args in (["init", "-q", "--bare", "-b", "main", str(origin)], ["clone", "-q", str(origin), str(ws)],
                 ["-C", str(ws), "commit", "-q", "--allow-empty", "-m", "a"],
                 ["-C", str(ws), "worktree", "add", "-q", str(wt), "-b", "mine"]):
        subprocess.run(["/usr/bin/git", *args], check=True, capture_output=True, env={**os.environ, **GIT_ENV})

    def ask(command, agent="bc-worker", cwd=ws):
        event = {"hook_event_name": "beforeShellExecution", "command": command, "conversation_id": agent, "cwd": str(cwd)}
        out = subprocess.run([sys.executable, HOOK], input=json.dumps(event), capture_output=True, text=True,
                             env={**os.environ, **env}, check=True).stdout
        return json.loads(out)

    assert ask("git status", "bc-coord", ws) == {"permission": "allow"}
    assert ask("git status", "bc-worker", wt) == {"permission": "allow"}
    placed = [f"cd /tmp && git -C {wt} push -u origin HEAD", f"cd {wt} && git -C {wt} push", f"cd /tmp; git -C {wt} push origin @",
              f"pushd /tmp && git -C {wt} push -u origin HEAD && popd", f"(cd /tmp && git -C {wt} push -u origin HEAD)",
              f"cd /tmp && git -C {ws} -C {wt} push -u origin HEAD", f"cd /tmp && git -C {wt} -C . push -u origin HEAD",
              f"bash -c 'cd /tmp && git -C {wt} push -u origin HEAD'",
              f"tmux send-keys -t w 'cd /tmp && git -C {wt} push -u origin HEAD' C-m"]
    for command in placed:
        for cwd in (ws, wt, tmp_path):
            assert ask(command, cwd=cwd) == {"permission": "allow"}, (command, cwd)
        out = ask(command, "bc-coord")
        assert out["permission"] == "deny" and "branch mine belongs to agent bc-worker" in out["agent_message"], (command, out)
        out = ask(command.replace(str(wt), str(ws)))                   # the same push of the coordinator's checkout: `main`'s
        assert out["permission"] == "deny" and out["agent_message"].startswith(
            "Refused by tools/agent-guard/shell_hook.py (.cursor/hooks.json): this push would update `main`"), (command, out)

    unplaced = [(f"cd {wt} && git push -u origin HEAD", "a `cd`"), (f"cd {tmp_path} && git -C wt push -u origin HEAD", "a `cd`"),
                (f"cd {wt}/.. && git -C ./wt push -u origin HEAD", "a `cd`"), ("cd /tmp && git -C ~ push -u origin HEAD", "a `cd`"),
                (f"pushd {tmp_path} && git -C wt push -u origin HEAD", "a `pushd`"),
                (f"source ./env.sh && git -C {wt} push -u origin HEAD", "a `source` in it, which can change what git reads"),
                (f". ./env.sh; git -C {wt} push -u origin HEAD", "runs `.`, which may move it to another directory or change"),
                (f"C=cd; $C /tmp; git -C {wt} push -u origin HEAD", "runs `$C`"),
                (f"cd /tmp && GIT_DIR={ws}/.git git -C {wt} push -u origin HEAD", "the command sets `GIT_DIR`"),
                (f"cd /tmp && git --git-dir={ws}/.git -C {wt} push -u origin HEAD", "names its repository with `--git-dir`")]
    for command, why in unplaced:
        out = ask(command, cwd=wt)
        assert out["permission"] == "deny" and "the guard can't tell which checkout this push runs in" in out["agent_message"], (
            command, out)
        assert why in out["agent_message"], (command, out)
        assert "by its absolute path or with no `cd` in the command, `git -C <checkout> push -u origin HEAD`" in out["agent_message"]
    out = ask(f"D={wt}; cd /tmp && git -C $D push -u origin HEAD", cwd=wt)
    assert out["permission"] == "deny" and "expands only after the guard has read it" in out["agent_message"], out

    for command in (f"cd /tmp && git -C {wt} switch -c x", f"pushd /tmp; git -C {wt} checkout -b y", f"(cd /tmp && git -C {wt} switch mine)"):
        assert ask(command) == {"permission": "allow"}, command
        out = ask(command, "bc-coord", wt)
        assert out["permission"] == "deny" and f"{wt} is agent bc-worker's checkout" in out["agent_message"], (command, out)
    for command in (f"cd {tmp_path} && git -C wt switch -c z", f"source ./env.sh; git -C {wt} switch -c z",
                    f"cd /tmp && git --git-dir={ws}/.git -C {wt} switch -c z"):
        out = ask(command)
        assert out["permission"] == "deny" and "can't tell whose checkout" in out["agent_message"], (command, out)
        assert "by its absolute path or with no `cd` in the command: `git -C <checkout> " in out["agent_message"], out


def test_a_push_whose_branch_the_guard_cant_tell_is_told_so_and_what_it_accepts(clones):
    """`cd WT && git push -u origin HEAD` updates whatever WT has checked out, not `main`: its refusal leads with what the guard
    can't tell and why, and names the push it accepts. A push that names `main` in its own words leads with the protected
    branches whatever the `cd`, and then says how to name another repository's checkout."""
    prefix = "Refused by tools/agent-guard/shell_hook.py (.cursor/hooks.json): "
    for form, why in (("cd {T} && git push -u origin HEAD", "the command has a `cd` in it"), ("cd {T} && git push", "a `cd`"),
                      ("pushd {T} && git push origin @", "a `pushd`"), ("GIT_DIR={T}/.git git push", "the command sets `GIT_DIR`")):
        for clone, target in (("verity", "{N}"), ("infra", "{N}"), ("notes", "{V}")):
            out = clones(form.replace("{T}", target), clone)
            message = out["agent_message"]
            assert out["permission"] == "deny" and message.startswith(
                prefix + "the guard can't tell which checkout this push runs in, so which branch it pushes, because "), message
            assert why in message and "`git -C <checkout> push -u origin HEAD`" in message, message
            assert "`git push -u origin <your-branch>`" in message and "only drive and the lander move" not in message, message
    out = clones("cd {N} && git push origin main", "verity")
    assert out["agent_message"].startswith(prefix + "this push would update `main`, `checked` or `next`"), out
    assert "because the command has a `cd` in it" in out["agent_message"], out
    assert "`git -C <checkout> push <remote> <refspec>`" in out["agent_message"], out


def test_what_the_guard_can_place_in_another_repository_still_passes(clones):
    out = clones("git push nosuchremote main", "notes")
    assert "git -C <checkout> push <remote> <refspec>" in out["agent_message"] and "is no remote" in out["agent_message"]
    out = clones("cd {N} && git push origin main", "verity")
    assert out["permission"] == "deny" and "has a `cd` in it" in out["agent_message"], out
    for form, clone in (("git push origin main", "notes"), ("git -C {N} push -o x origin main", "verity"),
                        ("git -C {N} push origin main", "infra"), ("git push {NURL} main", "verity"),
                        ("git push --repo={NURL}", "infra"), ("git push nosuchremote lane/mine", "notes")):
        assert clones(form, clone) == {"permission": "allow"}, (form, clone)


@pytest.mark.parametrize("form", [
    "cd {T} && git push origin lane/mine", "false && cd {T}; git push -u origin lane/mine",
    "pushd {T} && git push origin HEAD:refs/heads/lane/mine", "cd {T} && git push --force-with-lease origin lane/mine",
])
def test_a_lanes_own_branch_still_passes_after_a_cd(clones, form):
    """Where a push goes matters only for `main`, `checked`, `next` and another agent's branches: a lane's branch, named, passes
    from wherever the shell pushes it."""
    for clone, target in (("verity", "{N}"), ("infra", "{N}"), ("notes", "{V}")):
        assert clones(form.replace("{T}", target), clone) == {"permission": "allow"}, (form, clone)


def test_a_branch_switch_the_guard_cant_place_is_refused(tmp_path):
    """The branch-switch rule followed a `cd` too: from the coordinator's checkout, `false && cd MINE; git switch lane` read as
    a switch of the worker's own checkout and passed, and the shell switched the coordinator's. A switch in a command with a
    `cd`, or naming its repository by `GIT_DIR` or `--git-dir`, is refused; `git -C` places it."""
    env = {"AGENT_GUARD_STATE": str(tmp_path / "state"), "AGENT_GUARD_MIN_AVAIL": "0", **GIT_ENV}
    ws, wt = tmp_path / "ws", tmp_path / "wt"
    for args in (["init", "-q", "-b", "main", str(ws)], ["-C", str(ws), "commit", "-q", "--allow-empty", "-m", "a"],
                 ["-C", str(ws), "branch", "lane"], ["-C", str(ws), "worktree", "add", "-q", str(wt), "-b", "mine"]):
        subprocess.run(["/usr/bin/git", *args], check=True, capture_output=True, env={**os.environ, **GIT_ENV})

    def ask(command, agent, cwd):
        event = {"hook_event_name": "beforeShellExecution", "command": command, "conversation_id": agent, "cwd": str(cwd)}
        out = subprocess.run([sys.executable, HOOK], input=json.dumps(event), capture_output=True, text=True,
                             env={**os.environ, **env}, check=True).stdout
        return json.loads(out)

    assert ask("git status", "bc-coord", ws) == {"permission": "allow"}
    assert ask("git status", "bc-worker", wt) == {"permission": "allow"}
    lies = [f.replace("{T}", str(wt)).replace("git push origin main", "git switch lane") for f in NEVER_RUN_CDS]
    for command in (*lies, *(c.replace("cd ", "pushd ") for c in lies), f"cd {wt} && git switch -c x", f"cd {wt} && git checkout lane",
                    f"GIT_DIR={ws}/.git git switch lane", f"git --git-dir={ws}/.git switch lane"):
        out = ask(command, "bc-worker", ws if "GIT_DIR" not in command and "git-dir" not in command else wt)
        assert out["permission"] == "deny" and "can't tell whose checkout" in out["agent_message"], (command, out)
        assert "`git -C <checkout> " in out["agent_message"], out
    for command in (f"git -C {wt} switch -c x", f"cd {wt} && git checkout -- README", f"cd {ws} && git log -1"):
        assert ask(command, "bc-worker", ws) == {"permission": "allow"}, command


@pytest.mark.parametrize("form", [
    "git push --all origin", "git push --al origin", "git push --branches origin", "git push --bran origin",
    "git push --mirror origin", "git push origin 'refs/heads/*:refs/heads/*'", "git push origin :",
    "git push origin HEAD:heads/main", "git push origin @:main", "git push origin 'refs/heads/the*:refs/heads/the*'",
])
def test_every_branch_and_main_by_another_name_are_judged(clones, form):
    for clone in ("verity", "infra"):
        assert clones(form, clone)["permission"] == "deny", (form, clone)
    assert clones(form, "notes") == {"permission": "allow"}, form


@pytest.mark.parametrize("form", [
    "git push -o ci.skip origin lane/mine", "git push -u origin lane/mine", "git push --force-with-lease origin lane/mine",
    "git push --force-with-lease=lane/mine:abc origin +lane/mine", "git push --push-option=x origin HEAD:refs/heads/lane/mine",
    "git push origin --delete lane/mine", "git push --no-verify -q origin lane/mine", "bash -c 'git push -o x origin lane/mine'",
    "tmux send-keys 'git push -u origin lane/mine' C-m", "git push -- origin lane/mine",
])
def test_a_lanes_pushes_of_its_own_branch_still_pass(clones, form):
    for clone in ("verity", "infra", "notes"):
        assert clones(form, clone) == {"permission": "allow"}, (form, clone)


def test_a_push_that_would_update_main_or_next_and_a_config_that_prints_tokens_are_refused_for_any_agent(tmp_path):
    """Only drive moves `main` and `next`, so an agent's push to either is refused; and git config values can hold a GitHub
    installation token, so printing them is refused."""
    env = {"AGENT_GUARD_STATE": str(tmp_path / "state"), "AGENT_GUARD_MIN_AVAIL": "0", **GIT_ENV,
           "AGENT_GUARD_PUSH_REMOTES": str(tmp_path / "origin.git")}            # origin stands in for the queue's repository
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
