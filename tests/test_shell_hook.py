"""`.cursor/hooks.json`'s shell hook refuses commands that print the environment and lets everything else through."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "tools/agent-guard/shell_hook.py"


def verdict(command: str, **env: str) -> str:
    out = subprocess.run([sys.executable, HOOK], input=json.dumps({"hook_event_name": "beforeShellExecution",
                                                                     "command": command, "cwd": str(ROOT)}),
                         capture_output=True, text=True, env={**os.environ, **env}, check=True).stdout
    return json.loads(out)["permission"]


def test_cursor_runs_the_hook_on_every_shell_command_and_blocks_if_it_fails():
    (entry,) = json.loads((ROOT / ".cursor/hooks.json").read_text())["hooks"]["beforeShellExecution"]
    assert entry["command"] == "python3 tools/agent-guard/shell_hook.py" and entry["failClosed"] is True
    assert "matcher" not in entry


@pytest.mark.parametrize("command", [
    "env | cut -d= -f1",                                  # the leak, five times by 6 Oct
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
    "ps aux",                                             # a daemon's token in its argv, 6 Oct
    "ps auxww | head -50",
    "ps -ef",
    "ps -eF | sort",
    "ps -eo pid,args",
    "ps -A -o pid,rss,command",
    "ps axo pid,args",
    "ps --everyone --format=pid,cmd",
    "sudo ps aux | tail",
    "ssh n1 'ps -ef'",
    "cat /proc/*/cmdline",
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
    "uv run --env-file .env pytest",
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
    "ps -eo pid,args | grep \"[p]ytest\"",
    "ps aux | rg -i lake",
    "ssh n1 'ps -eo pid,args | grep \"[s]lot.py\"'",
    "pgrep -af quick_tier",
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
