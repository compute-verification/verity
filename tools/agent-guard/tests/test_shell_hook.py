"""`.cursor/hooks.json`'s shell hook refuses commands that print the environment and lets everything else through."""
import importlib.util
import json
import re
import os
import shutil
import subprocess
import sys
import uuid
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


def test_each_hook_records_its_events_key_names_and_of_its_values_only_the_model_and_the_turns_id(tmp_path, event_log):
    """Cursor's payload names a model and a turn (`generation_id`) beside the command, the account and paths. Each hook's
    line keeps those two where the payload has them as strings, cut at 64 characters, and no other value, so which model a
    turn's events name is read on the agent's own VM (`inbox.py turns`)."""
    W, turn = "bc-44444444-4444-4444-8444-444444444444", "0f0f0f0f-0f0f-4f0f-8f0f-0f0f0f0f0f0f"
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"handles": []}))
    env = {**os.environ, "AGENT_INBOX": str(tmp_path / "inbox"), "AGENT_GUARD_REGISTRY": str(registry),
           "AGENT_GUARD_TASK_LOG": str(tmp_path / "task.jsonl")}
    sent = {"conversation_id": W, "generation_id": turn, "model": "a-model-name", "cursor_version": "0.0-not-to-be-copied",
            "sandbox": False, "session_id": "session-not-to-be-copied", "tool_use_id": "tool-not-to-be-copied",
            "transcript_path": "/not-to-be-copied/transcript.jsonl", "user_email": "an-address-not-to-be-copied",
            "workspace_roots": ["/not-to-be-copied"]}
    shell = {**sent, "hook_event_name": "beforeShellExecution", "command": "echo not-to-be-copied", "cwd": "/"}
    task = {**sent, "hook_event_name": "preToolUse", "tool_name": "Task",
            "tool_input": {"subagent_type": "explore", "prompt": "not-to-be-copied", "model": "not-to-be-copied"}}
    stop = {**sent, "hook_event_name": "stop", "status": "completed", "loop_count": 0}

    def run(name, event):
        subprocess.run([sys.executable, HOOK.parent / name], input=event if isinstance(event, str) else json.dumps(event),
                       capture_output=True, text=True, check=True, env=env)

    for name, event in (("shell_hook.py", shell), ("task_hook.py", task), ("stop_hook.py", stop)):
        run(name, event)
    run("shell_hook.py", {**shell, "conversation_id": "bc-45454545-4545-4545-8545-454545454545", "model": "m" * 500,
                          "generation_id": "g" * 500})
    run("shell_hook.py", {**shell, "model": {"name": "not-to-be-copied"}, "generation_id": 7})
    run("shell_hook.py", {k: v for k, v in shell.items() if k not in ("model", "generation_id")})
    run("shell_hook.py", "not json")
    text = event_log.read_text()
    lines = [json.loads(ln) for ln in text.splitlines()]
    assert "not-to-be-copied" not in text
    assert [(e["hook"], e["event"], e["conversation_id"], e["generation_id"], e["model"]) for e in lines[:3]] == [
        ("shell_hook.py", "beforeShellExecution", W, turn, "a-model-name"),
        ("task_hook.py", "preToolUse", W, turn, "a-model-name"), ("stop_hook.py", "stop", W, turn, "a-model-name")]
    assert [e["keys"] for e in lines[:3]] == [sorted(shell), sorted(task), sorted(stop)]
    assert lines[1]["tool_input_keys"] == ["model", "prompt", "subagent_type"]
    assert sorted(lines[0]) == ["conversation_id", "event", "generation_id", "hook", "keys", "model", "t", "tool_input_keys",
                                "tool_name"], "the two values, and nothing else new"
    cut, typed, without, unread = lines[3:]
    assert (cut["model"], cut["generation_id"]) == ("m" * 64, "g" * 64)
    assert not {"model", "generation_id"} & (set(typed) | set(without) | set(unread)), "kept only where the payload has a string"
    assert typed["keys"] == sorted(shell) and unread["keys"] == "str" and unread["conversation_id"] is None
    listed = subprocess.run([sys.executable, INBOX_TOOL, "turns", W], capture_output=True, text=True, check=True, env=env).stdout
    assert re.search(rf"^{turn}  \S+  \S+ +3  a-model-name$", listed, re.M), listed


@pytest.fixture(scope="module")
def before_any_test():
    """What a module's fixture has before any per-test fixture has run: a hook it started, and a copy of the environment."""
    event = {"hook_event_name": "a module's fixture", "command": "true", "cwd": "/"}
    subprocess.run([sys.executable, HOOK], input=json.dumps(event), capture_output=True, text=True, check=True)
    return dict(os.environ)


def test_no_hook_a_test_starts_records_in_the_machines_log_whatever_environment_it_is_handed(before_any_test, scratch_event_log,
                                                                                            event_log):
    """A module's fixture runs before conftest's per-test log is named in the environment, and a test that builds a hook's
    environment from nothing, or takes every `AGENT_GUARD_*` variable out of it, drops the name. One fixture (conftest's
    `scratch_event_log`) gives each such hook a log under the test session's own directory: in the machine's log, a run's
    hundreds of events would push real agents' events out as it rotates."""
    def run(name, env):
        event = {"hook_event_name": name, "command": "true", "cwd": "/"}
        subprocess.run([sys.executable, HOOK], input=json.dumps(event), capture_output=True, text=True, check=True, env=env)

    def events(log):
        return [json.loads(ln)["event"] for ln in log.read_text().splitlines()]

    run("the test's own", None)
    run("a module's copy", before_any_test)
    run("from nothing", {"PATH": os.defpath})
    run("stripped", {k: v for k, v in os.environ.items() if not k.startswith("AGENT_GUARD_")})
    assert events(event_log) == ["the test's own"]
    elsewhere = ["a module's fixture", "a module's copy", "from nothing", "stripped"]
    assert [e for e in events(scratch_event_log) if e in elsewhere] == elsewhere


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
    # with a comment or a continued line in it as without one
    assert verdict("echo 'unbalanced # a note; env") == "deny"
    assert verdict("echo $'it\\'s' # a note\nenv; echo 'unbalanced'") == "deny"
    assert verdict("ls \\\n  -l; echo 'unbalanced") == "allow"


#: one refused command of each kind of rule, the same command continued over a line's end, and words of its refusal: the
#: environment, a secret file, a build, a long wait, the VM's tmux server and, asked in a clone of verity, a push of `main`,
#: a push of another agent's branch and a `git config` that prints tokens
REFUSED_KINDS = [
    ("env", "env \\\n  | cut -d= -f1", "would print environment variables' values"),
    ("cat deploy.env", "cat \\\n  deploy.env", "would print an env or credentials file"),
    ("cargo build", "cargo \\\n  build", "builds and tests run on our nodes"),
    ("sleep 3000", "sleep \\\n  3000", "no tool call may block more than 20 minutes"),
    ("tmux kill-server", "tmux \\\n  kill-server", "would end this VM's tmux server"),
    ("git push origin main", "git push origin \\\n  main", "only drive and the lander move"),
    ("git push origin theirs", "git push origin \\\n  theirs", "belongs to agent bc-other"),
    ("git config --list", "git config \\\n  --list", "would print a GitHub token"),
]


@pytest.mark.parametrize("beside", [
    "# a note\n{C}", "{C}  # a note", "echo issue#12; {C}", "curl -s https://example.com/#top && {C}",
    "echo '# not a comment'; {C}", "{CONTINUED}", "{C} # a note; echo \"two\nlines\"",
    "bash -c '# a note\n{C}'", "bash -c '{CONTINUED}'", 'bash -c "{CONTINUED}"',
])
@pytest.mark.parametrize("command, continued, said", REFUSED_KINDS, ids=[kind[0] for kind in REFUSED_KINDS])
def test_no_comment_and_no_continued_line_hides_a_refused_command(request, command, continued, said, beside):
    """Every rule reads a command as the shell runs it: a comment hides the rest of its own line and nothing after it, a `#`
    inside a word or a quoted string hides nothing, and a `\\` at a line's end joins the next line to it, in the command as
    typed and in what a carrier has run. Read as shlex reads a `#`, a comment from wherever one stands to the text's end,
    each rule passed whatever stood after one; and a continued line read as two commands, neither the one the shell runs.
    A comment that hides a quote's opening leaves a quote nothing closes: what stands before it is read all the same."""
    text = beside.replace("{C}", command).replace("{CONTINUED}", continued)
    out = request.getfixturevalue("clones")(text, "verity") if command.startswith("git ") else answer(text)
    assert out["permission"] == "deny" and said in out["agent_message"], (text, out)


@pytest.mark.parametrize("command, said", [
    ("ssh n1 '# a note\ncat deploy.env'", "would print an env or credentials file"),
    ("eval '# a note\nsleep 3000'", "no tool call may block more than 20 minutes"),
    ("tmux send-keys -t w '# a note' Enter 'env' Enter", "would print environment variables' values"),
    ("tmux send-keys -t w 'env \\' Enter '| cut -d= -f1' Enter", "would print environment variables' values"),
    ("ssh n1 \"bash -c '# a note\nenv'\"", "would print environment variables' values"),
    ("bash -c 'bash -c \"sleep \\\\\n3000\"'", "no tool call may block more than 20 minutes"),
    ("bash -lc \"sleep \\\\\n3000\"", "no tool call may block more than 20 minutes"),
    ("bash -lc \"sleep 1 \\\\\\\\\nsleep 3000\"", "no tool call may block more than 20 minutes"),
    ("tmux if-shell true 'display-message ok\nkill-server'  # a note", "would end this VM's tmux server"),
])
def test_what_a_carrier_runs_is_read_with_its_own_comments_and_continued_lines(command, said):
    """A quoted string reaches whatever reads it next with its lines as the shell hands them on: a comment in it ends at
    its own line, a `\\` a pane is sent before `Enter` continues the line typed, and inside double quotes the shell hands
    on one `\\` of each two, so `\\\\` at a line's end there continues the inner shell's line and four of them end it."""
    out = answer(command)
    assert out["permission"] == "deny" and said in out["agent_message"], (command, out)


@pytest.mark.parametrize("command", [
    "# a note", "# env | cut -d= -f1\n  # git push origin main\n# sleep 3000",
    "# where things stand\ngit status --short", "ls -l tools  # a note", "curl -sI https://example.com/guide#install",
    "# list, then search\nls tools\nrg -n words_of tools/agent-guard  # the reader\n"
    "uv run pytest tools/agent-guard/tests/test_envnames.py -k names",
    "cat > /tmp/note.md <<'EOF'\n# Title\nenv\nsleep 3000\ngit push origin main\nEOF",
    "python3 - <<'PY'\n# count the names\nimport os  # the environment\nprint(len(os.environ))  # no value is printed\nPY",
    "# don't forget\ngit log --oneline -5  # what's recent", "ls  # `env` is refused", "sleep 30  # not sleep 3000",
    "git commit -m 'guard: read past a comment (#12)' -m '# no comment, and env is a word here'",
    "echo \"$#\" \"${#HOME}\" \"${HOME##*/}\"", "rg -n '^#' README.md; sed -n '/^#/p' README.md",
    "rg -n \\\n  --glob '*.py' \\\n  words_of tools  # continued", "bash -lc '\n# sync first\nuv sync -q\n# then list\nls\n'",
    "tmux -f /exec-daemon/tmux.portal.conf send-keys -t build:0.0 'make # quick' C-m",
    # one command to the shell, and one no rule refuses: the lines joined, and a word that only begins like `env`
    "timeout 60 \\\n  sleep 5000", "research run --on auto --cwd source -- \\\n  uv run pytest -q", "env#x", "sleep 3000#x",
])
def test_a_comment_a_continued_line_and_a_hash_in_a_word_refuse_nothing_the_shell_would_not_run(command):
    assert answer(command) == {"permission": "allow"}, command


def test_a_comment_and_a_continued_line_are_read_as_bash_reads_them(monkeypatch):
    """`words_of` against bash itself: the simple commands bash runs for a text, and each one's words. Nothing runs: every
    command is `qa` or `qb`, which no PATH holds, and bash hands a command it cannot find to a function that prints its
    words."""
    hook = guard_module("shell_hook", monkeypatch)
    handler = r'''command_not_found_handle() { printf '%s\0' "$@" $'\1'; }; eval "$1"'''
    for text in ("# a note\nqa", "qa b # a note\nqb", "qa b # a note; qa c\nqb", "qa a#b; qb", "qa https://example.com/#top && qb",
                 "qa '# b' \"# c\" 'd'#e; qb", "qa a#b#c #d\nqb", "(qa) # a note\nqb", "qa;#b\nqb", "qa b \\\n  c; qb",
                 "qa a\\\n#b; qb", "qa a \\\n# b\nqb", "qa \\ #b; qb", "qa a \\# b; qb", "# a note \\\nqa", "qa b # a note \\\nqb",
                 "qa a\\\\\nqb", "qa a\\\\\\\nqb", "qa \"a\\\nb\"; qb", "qa 'a\\\nb'; qb", "qa \"a # b\\\n # c\"; qb",
                 "qa \"a\\\\\\\nb\" 'c\\\\\nd'; qb", "qa b # it's\nqb"):
        out = subprocess.run([shutil.which("bash"), "--norc", "--noprofile", "-c", handler, "bash", text], capture_output=True,
                             check=True, env={"PATH": "/nonexistent"}).stdout.decode()
        ran = [command.split("\0")[:-1] for command in out.split("\1\0")[:-1]]
        assert ran and [words for words, _ in hook.words_of(text)] == ran, text


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
    message has waited past STALE_S unpulled; `check` on the target's VM says whether its hook polls. Every `send`, pushed or
    `--local`, says in its one line that the target's next shell command delivers the message, that nothing delivers it at a
    turn's end, and that a target whose turn ended (one finished before its stop messages reached it) is resumed with the
    text itself: what its VM held unshown may be gone by its next turn."""
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
    told = (f"{W}'s next shell command delivers it, and nothing delivers it at a turn's end. If {W} has ended its turn "
            "(FINISHED or IDLE), resume it with the text itself (Task or a follow-up): a message its VM holds unshown when "
            "its turn ends may be gone by its next turn")
    assert re.fullmatch(re.escape(f"pushed for {W}: origin's inbox/{W} at ") + "[0-9a-f]{12}" + re.escape(
        f"; {told}. `--wait S` says whether its VM took it within S seconds\n"), sent.stdout), sent.stdout
    assert "earlier" not in sent.stderr
    here = tool("send", W, "--local", "--text", "on this VM", inbox=tmp_path / "local-inbox")
    assert here.stdout == f"queued for {W}: {tmp_path / 'local-inbox' / f'{W}.md'}; {told}\n", here.stdout
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
    after how long, or NOT TAKEN and exit 1 after S seconds, when the sender has to stop the work itself. Taken is said as
    neither shown nor acted on: a turn that ends before another command may lose it. The clock and the sleep are the
    test's: the worker's pull happens inside the sender's second sleep."""
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
    pushed, took = said.out.splitlines()
    assert pushed.startswith(f"pushed for {W}: ")
    assert re.fullmatch(re.escape(f"inbox: taken 10 s after the push: {W}'s VM pulled ") + "[0-9a-f]{12}" + re.escape(
        ". Taken is neither shown nor acted on: its next shell command shows it, and if its turn ends first it may be gone by "
        f"its next turn (then resume {W} with the text itself); its runs and its issue show whether it acted"), took), took
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


def test_the_stop_hook_offers_a_message_still_queued_when_a_turn_completes_as_its_follow_up_once(tmp_path):
    """A worker that ends its turn with no further shell command never meets the shell hook's refusal, so a message sent after
    its last command is not shown in that turn and may be gone by its next. The stop hook offers it to Cursor as the turn's
    follow-up, once, for a runtime that honours one; no delivery relies on it, and the shell hook does not deliver again a
    message the stop hook took."""
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


def test_turns_prints_the_model_each_turn_of_an_agent_names_from_both_files_of_the_event_log(event_log):
    """`turns` reads this VM's event log, the file it rotated to first: a line per generation id of the agent, oldest first,
    with when its first and last event were recorded, how many it has and the model they name, so a rule about models can
    be checked by turn. `-` stands for what an event doesn't name, a generation id whose events name two models is a line
    for each, and only printable characters of the log are printed."""
    A, other = "bc-12121212-1212-4212-8212-121212121212", "bc-34343434-3434-4434-8434-343434343434"
    one, two, three = (f"{n * 8}-0000-4000-8000-000000000000" for n in "123")

    def line(t, agent=A, hook="shell_hook.py", **kept):
        return json.dumps({"t": t, "hook": hook, "event": "beforeShellExecution", "conversation_id": agent, **kept}) + "\n"

    def turns(*args, **env):
        return subprocess.run([sys.executable, INBOX_TOOL, "turns", *args], capture_output=True, text=True,
                              env={**os.environ, **env})

    Path(f"{event_log}.1").write_text(
        line(1_000_000_000) + line(1_000_000_060, generation_id=one, model="model-a")
        + line(1_000_000_090, other, generation_id=three, model="model-c"))
    event_log.write_text(
        line(1_000_000_120, hook="task_hook.py", generation_id=one, model="model-a") + "not json\n"
        + line(1_000_003_600, generation_id=two, model="model-b")
        + line(1_000_003_660, hook="stop_hook.py", generation_id=two, model="model-b")
        + line(1_000_003_720, generation_id=two, model="model-c\x1b[2J") + line(1_000_007_200, generation_id=three))
    listed = turns(A)
    assert listed.returncode == 0 and listed.stdout.splitlines() == [
        "generation_id                         first_event           last_event            events  model",
        "-                                     2001-09-09T01:46:40Z  2001-09-09T01:46:40Z       1  -",
        f"{one}  2001-09-09T01:47:40Z  2001-09-09T01:48:40Z       2  model-a",
        f"{two}  2001-09-09T02:46:40Z  2001-09-09T02:47:40Z       2  model-b",
        f"{two}  2001-09-09T02:48:40Z  2001-09-09T02:48:40Z       1  model-c?[2J",
        f"{three}  2001-09-09T03:46:40Z  2001-09-09T03:46:40Z       1  -"], listed.stdout
    assert listed.stderr.startswith("inbox: `-` stands for what a turn's events don't name")
    mine = turns(CURSOR_CONVERSATION_ID=other)
    assert (mine.returncode, mine.stdout.splitlines()[1:], mine.stderr) == (
        0, [f"{three}  2001-09-09T01:48:10Z  2001-09-09T01:48:10Z       1  model-c"], ""), "this shell's agent by default"
    nobody = turns("bc-nobody")
    assert nobody.returncode == 1 and nobody.stdout == (
        f"inbox: {event_log} holds no event of bc-nobody: no hook has run for it on this VM, or its events have rotated out\n")


ALLOW = {"permission": "allow"}
LAUNCH, OTHER = "launch-model", "other-model"
AGENT, PEER = "bc-56565656-5656-4565-8565-565656565656", "bc-67676767-6767-4676-8676-676767676767"
TURNS = [f"{n * 8}-1111-4111-8111-111111111111" for n in "123456789"]


def on_model(command: str, agent: str, turn: str | None, model: str | None, inbox: Path, **env: str) -> dict:
    """The hook's verdict on `command` in the turn `turn` of `agent` on `model`; a None leaves that key out of the payload."""
    event = {"hook_event_name": "beforeShellExecution", "command": command, "conversation_id": agent,
             **({"generation_id": turn} if turn is not None else {}), **({"model": model} if model is not None else {})}
    out = subprocess.run([sys.executable, HOOK], input=json.dumps(event), capture_output=True, text=True, check=True,
                         env={**os.environ, "AGENT_INBOX": str(inbox), "AGENT_GUARD_MIN_AVAIL": "0", **env}).stdout
    return json.loads(out)


def settled(log: Path, event: str = "model_refusal") -> list[dict]:
    return [e for e in map(json.loads, log.read_text().splitlines()) if e.get("event") == event]


def logged(t: int, agent: str, **kept: str) -> str:
    return json.dumps({"t": t, "hook": "shell_hook.py", "event": "beforeShellExecution", "conversation_id": agent,
                       **kept}) + "\n"


def test_a_turn_on_another_model_than_the_launchs_is_refused_once_at_its_first_command_with_both_models_and_both_ways_on(
        tmp_path, event_log):
    """A turn an agent's own timer or a subscription starts can take another model than its launcher named, and nothing in
    the turn says so. Its first command is refused once, naming both models and the two ways on; the rest of the turn
    passes, and so does every turn back on the launch model. Each turn on another model is told, not only the first."""
    one, two, three, four = TURNS[:4]
    assert on_model("ls", AGENT, one, LAUNCH, tmp_path) == ALLOW
    assert on_model("pwd", AGENT, one, LAUNCH, tmp_path) == ALLOW
    first = on_model("ls", AGENT, two, OTHER, tmp_path)
    text = first["user_message"]
    assert first["permission"] == "deny" and text == first["agent_message"]
    assert f"this turn runs on `{OTHER}`, and you were launched on `{LAUNCH}`" in text
    assert "Task and `model: claude-opus-5-5-high` (or `claude-sonnet-5-5-high`" in text and "never Fable 5.2" in text
    assert f"model: {LAUNCH}" not in text and "resumed by your lead" in text and "go by your brief" in text
    assert on_model("ls", AGENT, two, OTHER, tmp_path) == ALLOW
    assert on_model("pwd", AGENT, two, OTHER, tmp_path) == ALLOW
    (line,) = settled(event_log)
    assert sorted(line) == ["conversation_id", "event", "generation_id", "hook", "launch_model", "model", "t"]
    assert (line["hook"], line["conversation_id"], line["generation_id"], line["model"], line["launch_model"]) == (
        "shell_hook.py", AGENT, two, OTHER, LAUNCH)
    assert on_model("ls", AGENT, three, LAUNCH, tmp_path) == ALLOW
    assert on_model("pwd", AGENT, three, LAUNCH, tmp_path) == ALLOW
    assert on_model("ls", AGENT, four, OTHER, tmp_path)["permission"] == "deny"
    assert on_model("ls", AGENT, four, OTHER, tmp_path) == ALLOW
    assert [e["generation_id"] for e in settled(event_log)] == [two, four], "once a turn, not once an agent"


def test_an_agent_whose_turns_all_run_on_its_launch_model_is_never_refused(tmp_path, event_log):
    """The launch model is each agent's own: another agent's turns on another model, in the same log and the first in it,
    refuse nothing of this one's."""
    assert on_model("ls", PEER, TURNS[5], OTHER, tmp_path) == ALLOW
    for mine, peers in zip(TURNS[:3], TURNS[5:8]):
        for command in ("ls", "pwd"):
            assert on_model(command, AGENT, mine, LAUNCH, tmp_path) == ALLOW
            assert on_model(command, PEER, peers, OTHER, tmp_path) == ALLOW
    assert settled(event_log) == []


def test_models_are_compared_by_family_never_by_effort(tmp_path, event_log, registry):
    """A lead moved from Extra High to Medium runs the model it was launched on: a launch on Fable 5.2 Extra High and a
    turn on Fable 5.2 Medium pass, and so does Opus at another effort; a turn on another family is told, whatever its
    effort, and its refusal names no Fable child."""
    one, two, three, four = TURNS[:4]
    assert on_model("ls", HOLDER, one, "claude-fable-5-2-xhigh", tmp_path, **registry) == ALLOW
    assert on_model("ls", HOLDER, two, "claude-fable-5-2-medium", tmp_path, **registry) == ALLOW
    assert on_model("ls", HOLDER, three, "claude-fable-5-2", tmp_path, **registry) == ALLOW
    told = on_model("ls", HOLDER, four, "claude-opus-5-5-medium", tmp_path, **registry)
    assert told["permission"] == "deny" and "you were launched on `claude-fable-5-2-xhigh`" in told["user_message"]
    assert "model: claude-fable" not in told["user_message"]
    assert on_model("ls", PEER, one, "claude-opus-5-5-high", tmp_path, **registry) == ALLOW
    assert on_model("ls", PEER, two, "claude-opus-5-5-medium", tmp_path, **registry) == ALLOW
    assert [e["conversation_id"] for e in settled(event_log)] == [HOLDER]


def test_no_launch_model_on_record_refuses_nothing(tmp_path, event_log):
    """A payload that names no model or no turn, an agent whose first line names no turn or whose first turn names no
    model, and a local session pass on any model. The launch model is the first model a line of the first turn names,
    though that turn's first line names none."""
    one, two, three = TURNS[:3]
    assert on_model("ls", AGENT, one, LAUNCH, tmp_path) == ALLOW
    assert on_model("ls", AGENT, two, None, tmp_path) == ALLOW
    assert on_model("ls", AGENT, None, OTHER, tmp_path) == ALLOW
    assert on_model("ls", AGENT, two, OTHER, tmp_path)["permission"] == "deny", "the turn the two payloads would be in"

    untold, unnamed = "bc-78787878-7878-4787-8787-787878787878", "bc-89898989-8989-4898-8898-898989898989"
    with event_log.open("a") as log:
        log.write(logged(1_000_000_000, untold, model="model-before-turns"))
        log.write(logged(1_000_000_000, unnamed, generation_id=one))
    for turn, model in ((one, LAUNCH), (two, OTHER), (three, LAUNCH)):
        assert on_model("ls", untold, turn, model, tmp_path) == ALLOW, "its first line names no turn"
    for turn, model in ((two, LAUNCH), (three, OTHER)):
        assert on_model("ls", unnamed, turn, model, tmp_path) == ALLOW, "its first turn names no model"
    local = "56565656-5656-4565-8565-565656565656"
    assert on_model("ls", local, one, LAUNCH, tmp_path) == ALLOW
    assert on_model("ls", local, two, OTHER, tmp_path) == ALLOW, "a local session's user picks its model turn by turn"

    named_later = "bc-90909090-9090-4909-8909-909090909090"
    with event_log.open("a") as log:
        log.write(logged(1_000_000_000, named_later, generation_id=one))
    assert on_model("ls", named_later, one, LAUNCH, tmp_path) == ALLOW
    refused = on_model("ls", named_later, two, OTHER, tmp_path)
    assert refused["permission"] == "deny" and f"launched on `{LAUNCH}`" in refused["user_message"]
    assert [e["conversation_id"] for e in settled(event_log)] == [AGENT, named_later]


def test_the_setting_in_a_turns_first_command_passes_it_and_settles_the_turn(tmp_path, event_log):
    """`AGENT_GUARD_MODEL_OK=1` in the first command of a turn on another model passes it and settles the turn as a refusal
    would, with a `model_ok` line; it settles only its own turn, and in a turn already refused it adds nothing."""
    one, two, three = TURNS[:3]
    assert on_model("ls", AGENT, one, LAUNCH, tmp_path) == ALLOW
    assert on_model("AGENT_GUARD_MODEL_OK=1 ls", AGENT, two, OTHER, tmp_path) == ALLOW
    assert on_model("ls", AGENT, two, OTHER, tmp_path) == ALLOW
    assert [e["generation_id"] for e in settled(event_log, "model_ok")] == [two] and settled(event_log) == []
    assert on_model("ls", AGENT, three, OTHER, tmp_path)["permission"] == "deny"
    assert on_model("AGENT_GUARD_MODEL_OK=1 ls", AGENT, three, OTHER, tmp_path) == ALLOW
    assert len(settled(event_log, "model_ok")) == 1 and [e["generation_id"] for e in settled(event_log)] == [three]


def test_another_refusal_comes_first_and_a_waiting_message_after(tmp_path, event_log):
    """A command the hook refuses for another reason is refused with that reason and leaves the turn unsettled, so the
    next command it would pass is the one the model rule refuses; a waiting inbox message goes to the command after."""
    one, two = TURNS[:2]
    inbox = tmp_path / "inbox"
    assert on_model("ls", AGENT, one, LAUNCH, inbox) == ALLOW
    environment = on_model("env", AGENT, two, OTHER, inbox)
    assert environment["permission"] == "deny" and "environment variables' values" in environment["user_message"]
    assert "Refused once" not in environment["user_message"] and settled(event_log) == []
    queue(AGENT, "a waiting message", inbox)
    told = on_model("ls", AGENT, two, OTHER, inbox)
    assert told["permission"] == "deny" and told["user_message"].startswith("Refused once")
    assert "a waiting message" not in told["user_message"]
    delivered = on_model("ls", AGENT, two, OTHER, inbox)
    assert delivered["permission"] == "deny" and "a waiting message" in delivered["user_message"]
    assert on_model("ls", AGENT, two, OTHER, inbox) == ALLOW


def test_the_launch_model_is_the_first_turns_in_the_file_the_log_rotated_to_and_a_log_that_starts_late_names_what_it_holds(
        tmp_path, event_log):
    """The launch turn may sit in the file the log rotated to. A log that starts after the launch (rotated out, or a new
    machine a wake booted) holds a later turn first, and that turn's model is on record: a turn on any other is refused
    naming it, and the refusal says to go by the brief."""
    one, two, three, five, six = (TURNS[i] for i in (0, 1, 2, 4, 5))
    Path(f"{event_log}.1").write_text(logged(1_000_000_000, AGENT, generation_id=one, model=LAUNCH))
    event_log.write_text(logged(1_000_003_600, AGENT, generation_id=two, model=OTHER))
    refused = on_model("ls", AGENT, three, OTHER, tmp_path)
    assert refused["permission"] == "deny" and f"launched on `{LAUNCH}`" in refused["user_message"]

    late = "bc-a1a1a1a1-a1a1-4a1a-8a1a-a1a1a1a1a1a1"
    assert on_model("ls", late, five, "late-model", tmp_path) == ALLOW
    told = on_model("ls", late, six, LAUNCH, tmp_path)["user_message"]
    assert f"this turn runs on `{LAUNCH}`, and you were launched on `late-model`" in told and "go by your brief" in told


def test_a_log_the_rule_cannot_write_refuses_nothing_and_says_so(tmp_path, event_log):
    """A guard that can't record that a turn was told can't keep the telling to once, so it refuses nothing, and says so:
    with the launch turn on record in the file the log rotated to and the log itself unwritable, a turn on another model
    passes, each command saying on stderr why."""
    Path(f"{event_log}.1").write_text(logged(1_000_000_000, AGENT, generation_id=TURNS[0], model=LAUNCH))
    event_log.mkdir()
    event = {"hook_event_name": "beforeShellExecution", "command": "ls", "conversation_id": AGENT,
             "generation_id": TURNS[1], "model": OTHER}
    for _ in range(2):
        run = subprocess.run([sys.executable, HOOK], input=json.dumps(event), capture_output=True, text=True, check=True,
                             env={**os.environ, "AGENT_INBOX": str(tmp_path / "inbox"), "AGENT_GUARD_MIN_AVAIL": "0"})
        assert json.loads(run.stdout) == ALLOW
        assert "the model rule refuses nothing, since it can't record a turn as told in" in run.stderr


FABLE, OPUS, SONNET = "claude-fable-5-2-xhigh", "claude-opus-5-5-high", "claude-sonnet-5-5-high"
HOLDER = "bc-12121212-1212-4121-8121-121212121212"


def coordinator(handle: str, generation: int) -> str:
    """The id `infra/cursor/launch_coordinator.py` gives `handle`'s coordinator at `generation`."""
    namespace = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/compute-verification/verity/coordinators")
    return f"bc-{uuid.uuid5(namespace, f'{handle}/{generation}')}"


@pytest.fixture
def registry(tmp_path) -> dict[str, str]:
    """A Slack registry with two handles, `infra` held by HOLDER and `vllm` held by no one, as the hook's environment."""
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"handles": [{"handle": "infra", "holder": HOLDER}, {"handle": "vllm"}]}))
    return {"AGENT_GUARD_REGISTRY": str(path)}


def test_every_command_of_a_workers_turn_on_fable_is_refused_each_time_with_the_way_on(tmp_path, event_log, registry):
    """No worker runs Fable 5.2: every command of its turn on it, at any effort, is refused, not only the first, with the
    way on; its turns on Opus or Sonnet pass, and so does a turn on Opus after a launch on Fable, which the model rule
    would otherwise tell to go back to Fable."""
    for turn, model in ((TURNS[0], FABLE), (TURNS[1], "claude-fable-5-2-medium"), (TURNS[2], "claude-fable-5-2")):
        for command in ("ls", "echo pushed", "AGENT_GUARD_MODEL_OK=1 ls"):
            refused = on_model(command, AGENT, turn, model, tmp_path, **registry)
            text = refused["user_message"]
            assert refused["permission"] == "deny" and text == refused["agent_message"], (model, command)
            assert f"this turn runs on `{model}`, and no worker runs Fable 5.2" in text
            assert "`claude-opus-5-5-high`, or `claude-sonnet-5-5-high` for a fully specified task" in text
            assert "End this turn now" in text and "what started it" in text and "naming `claude-opus-5-5-high`" in text
    for turn, model in ((TURNS[3], OPUS), (TURNS[4], SONNET), (TURNS[5], OPUS)):
        assert on_model("ls", AGENT, turn, model, tmp_path, **registry) == ALLOW, model
        assert on_model("pwd", AGENT, turn, model, tmp_path, **registry) == ALLOW, model
    assert settled(event_log) == [] and settled(event_log, "model_ok") == []


def test_a_leads_turn_on_fable_passes_and_so_does_a_later_generation_of_a_registered_handle(tmp_path, event_log, registry):
    """A lead is a registry holder, or launch_coordinator's id for a registered handle, held or not, at a generation from 1
    to 50; it runs Fable at any effort. A generation past 50, an unregistered handle's coordinator and a lead's child are
    workers."""
    for agent in (HOLDER, coordinator("infra", 1), coordinator("vllm", 7), coordinator("infra", 50)):
        for turn, model in ((TURNS[0], FABLE), (TURNS[1], "claude-fable-5-2-medium")):
            for command in ("ls", "pwd"):
                assert on_model(command, agent, turn, model, tmp_path, **registry) == ALLOW, (agent, model)
    for agent in (coordinator("infra", 51), coordinator("infra", 0), coordinator("comms", 1), PEER):
        assert on_model("ls", agent, TURNS[0], FABLE, tmp_path, **registry)["permission"] == "deny", agent


def test_a_turn_whose_model_the_hook_cannot_read_and_a_registry_it_cannot_read_refuse_nothing(tmp_path, event_log, registry):
    """The rule must never lock an agent out: a payload with no model, or one that isn't a string, passes; so does a local
    session on Fable; and while the registry can't be read, a lead can't be told from a worker, so a turn on Fable passes
    and the hook says why on stderr."""
    assert on_model("ls", AGENT, TURNS[0], None, tmp_path, **registry) == ALLOW
    event = {"hook_event_name": "beforeShellExecution", "command": "ls", "conversation_id": AGENT,
             "generation_id": TURNS[1], "model": ["claude-fable-5-2-xhigh"]}
    run = subprocess.run([sys.executable, HOOK], input=json.dumps(event), capture_output=True, text=True, check=True,
                         env={**os.environ, "AGENT_INBOX": str(tmp_path), "AGENT_GUARD_MIN_AVAIL": "0", **registry})
    assert json.loads(run.stdout) == ALLOW
    assert on_model("ls", "56565656-5656-4565-8565-565656565656", TURNS[2], FABLE, tmp_path, **registry) == ALLOW
    for missing in (tmp_path / "no-registry.json", tmp_path):
        event = {**event, "generation_id": TURNS[3], "model": FABLE}
        for _ in range(2):
            run = subprocess.run([sys.executable, HOOK], input=json.dumps(event), capture_output=True, text=True,
                                 check=True, env={**os.environ, "AGENT_INBOX": str(tmp_path / "inbox"),
                                                  "AGENT_GUARD_MIN_AVAIL": "0", "AGENT_GUARD_REGISTRY": str(missing)})
            assert json.loads(run.stdout) == ALLOW, missing
            assert "the Fable rule refuses nothing, since it can't read the registry" in run.stderr


def test_the_guards_lead_ids_are_launch_coordinators(monkeypatch):
    """The guard can't import `infra/`, so it keeps launch_coordinator's namespace and id form itself: they agree."""
    source = (ROOT / "infra/cursor/launch_coordinator.py").read_text()
    (url,) = re.findall(r'^NAMESPACE = uuid\.uuid5\(uuid\.NAMESPACE_URL, "([^"]+)"\)$', source, re.M)
    assert "    return f\"bc-{uuid.uuid5(NAMESPACE, f'{handle}/{generation}')}\"\n" in source
    task_hook = guard_module("task_hook", monkeypatch)
    assert task_hook.COORDINATORS == uuid.uuid5(uuid.NAMESPACE_URL, url)
    assert coordinator("infra", 3) == f"bc-{uuid.uuid5(task_hook.COORDINATORS, 'infra/3')}"


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


LEAD, PAST, WORKER = (f"bc-{d * 8}-{d * 4}-4{d * 3}-8{d * 3}-{d * 12}" for d in "abc")
WORKTREE_GUARD = HOOK.parent / "worktree_guard.py"


@pytest.fixture
def vm(tmp_path, event_log):
    """A coordinator's VM: its checkout `ws` on branch `lead`, a worker's worktree `wt` on `work` and one nobody has run git
    in, `fresh`; LEAD holds a handle in the registry the hooks read. `ask` is the hook's answer to a command an agent types,
    `reclaim` the command as that agent's shell runs it, and `long_ago` moves every hook run recorded so far back to 1970."""
    env = {**os.environ, **GIT_ENV, "AGENT_GUARD_STATE": str(tmp_path / "state"), "AGENT_GUARD_MIN_AVAIL": "0",
           "AGENT_GUARD_REGISTRY": str(tmp_path / "registry.json"), "AGENT_GUARD_PUSH_REMOTES": str(tmp_path / "origin.git")}
    (tmp_path / "registry.json").write_text(json.dumps({"handles": [{"handle": "infra", "holder": LEAD}]}))
    ws, origin = tmp_path / "ws", tmp_path / "origin.git"
    for args in (["init", "-q", "--bare", "-b", "main", str(origin)], ["clone", "-q", str(origin), str(ws)],
                 ["-C", str(ws), "checkout", "-q", "-b", "lead"], ["-C", str(ws), "commit", "-q", "--allow-empty", "-m", "a"],
                 ["-C", str(ws), "worktree", "add", "-q", str(tmp_path / "wt"), "-b", "work"],
                 ["-C", str(ws), "worktree", "add", "-q", str(tmp_path / "fresh"), "-b", "unused"]):
        subprocess.run(["/usr/bin/git", *args], check=True, capture_output=True, env=env)

    class VM:
        root, environment = tmp_path, env

        @staticmethod
        def ask(command: str, agent: str, cwd: Path = ws) -> dict:
            event = {"hook_event_name": "beforeShellExecution", "command": command, "conversation_id": agent, "cwd": str(cwd)}
            return json.loads(subprocess.run([sys.executable, HOOK], input=json.dumps(event), capture_output=True, text=True,
                                             env=env, check=True).stdout)

        @staticmethod
        def reclaim(directory: Path, agent: str) -> subprocess.CompletedProcess:
            return subprocess.run([sys.executable, WORKTREE_GUARD, "reclaim", str(directory)], capture_output=True, text=True,
                                  env={**env, "CURSOR_CONVERSATION_ID": agent})

        @staticmethod
        def long_ago() -> None:
            event_log.write_text(re.sub(r'"t": \d+', '"t": 1000', event_log.read_text()))

        @staticmethod
        def owner(directory: Path) -> str:
            gitdir = subprocess.run(["/usr/bin/git", "-C", str(directory), "rev-parse", "--absolute-git-dir"], check=True,
                                    capture_output=True, text=True, env=env).stdout.strip()
            stamp = Path(gitdir) / "agent-owner"
            return stamp.read_text().strip() if stamp.exists() else ""

        @staticmethod
        def branches() -> dict:
            return json.loads((tmp_path / "state" / "branches.json").read_text())
    return VM


def test_the_vms_coordinator_takes_back_a_checkout_and_its_branches_from_an_owner_that_is_gone(vm):
    """A checkout belongs to the first agent that runs git in it, and the stamp outlived the agent: a coordinator whose
    workspace a past worker had claimed could neither switch its branch nor push it, and the refusals' advice was another
    checkout and another branch. Its refusals now name `reclaim`, which gives it the checkout and the branches the gone owner
    held."""
    ws, command = vm.root / "ws", f"python3 tools/agent-guard/worktree_guard.py reclaim {vm.root / 'ws'}"
    assert vm.ask("git status", PAST) == {"permission": "allow"}             # the past worker claims ws, and its branch
    assert (vm.owner(ws), vm.branches()) == (PAST, {"lead": PAST})
    vm.long_ago()
    taking = f"take back {ws} and the branches {PAST} holds with `{command}`"
    for refused, said in (("git switch -c other", f"{ws} is agent {PAST}'s checkout"),
                          ("git push origin lead", f"branch lead belongs to agent {PAST}"),
                          ("python3 tools/agent-guard/worktree_guard.py hand lead bc-third", "only that agent can hand it on")):
        out = vm.ask(refused, LEAD)
        assert out["permission"] == "deny" and said in out["agent_message"] and taking in out["agent_message"], out
        assert "you hold @infra" in out["agent_message"] and "If it is gone, push a branch of your own" not in out["agent_message"]
        out = vm.ask(refused, WORKER)
        assert out["permission"] == "deny" and said in out["agent_message"] and "worktree_guard.py reclaim" not in out["agent_message"], out
    assert "If it is gone, push a branch of your own." in vm.ask("python3 worktree_guard.py hand lead bc-third", WORKER)["agent_message"]

    assert vm.ask(command, LEAD) == {"permission": "allow"}
    done = vm.reclaim(ws, LEAD)
    assert done.returncode == 0 and f"{ws} is now {LEAD}'s (@infra); it was {PAST}'s" in done.stdout, done.stdout + done.stderr
    assert f"So are the branches {PAST} held: lead" in done.stdout
    assert (vm.owner(ws), vm.branches()) == (LEAD, {"lead": LEAD})
    for command in ("git switch -c other", "git push origin lead"):
        assert vm.ask(command, LEAD) == {"permission": "allow"}, command
    assert vm.ask("git switch lead", WORKER)["permission"] == "deny", "and it is the coordinator's against every other agent"
    again = vm.reclaim(ws, LEAD)
    assert again.returncode == 0 and "is already" in again.stdout


def test_reclaim_takes_nothing_from_a_live_owner_for_a_caller_without_a_handle_or_from_a_checkout_no_one_owns(vm):
    """What `reclaim` must not do: evict a worker that is running, on a guess; hand a checkout to a worker, whatever it says
    it is; or claim a checkout that has no owner. Each refusal says why, and what to do instead."""
    ws, wt, fresh = (vm.root / d for d in ("ws", "wt", "fresh"))
    typed = "python3 tools/agent-guard/worktree_guard.py reclaim"
    assert vm.ask("git status", PAST) == {"permission": "allow"}
    vm.long_ago()
    assert vm.ask("git status", WORKER, wt) == {"permission": "allow"}           # a worker claims wt, and is running
    assert (vm.owner(ws), vm.owner(wt), vm.owner(fresh)) == (PAST, WORKER, "")

    # a caller that holds no handle: the hook refuses what it types, whoever its command line says it is, and so does the command
    for command in (f"{typed} {ws}", f"CURSOR_CONVERSATION_ID={LEAD} {typed} {ws}", f"cd {ws} && {typed} .",
                    f"bash -c '{typed} {ws}'", f"tmux send-keys '{typed} {ws}' C-m"):
        out = vm.ask(command, WORKER)
        assert out["permission"] == "deny" and "only the VM's coordinator takes a checkout back" in out["agent_message"], command
        assert f"{WORKER} holds no handle in the Slack registry" in out["agent_message"], out
    assert f"ask your coordinator to run `{typed} {ws}`" in vm.ask(f"{typed} {ws}", WORKER)["agent_message"]
    for agent in (WORKER, PAST, ""):
        out = vm.reclaim(ws, agent)
        assert out.returncode == 1 and "only the VM's coordinator takes a checkout back" in out.stderr, out.stderr
    assert vm.owner(ws) == PAST

    # a live owner: the coordinator's command is let run, and refuses
    assert "worktree_guard.py reclaim" not in vm.ask("git switch lead", LEAD, wt)["agent_message"], "nothing to take back from a running worker"
    assert vm.ask(f"{typed} {wt}", LEAD) == {"permission": "allow"}
    out = vm.reclaim(wt, LEAD)
    assert out.returncode == 1 and f"{wt} stays agent {WORKER}'s: agent {WORKER} ran a command on this VM" in out.stderr, out.stderr
    assert "so it is not gone (gone is no hook run in 2 h)" in out.stderr and "work in a checkout of your own" in out.stderr
    assert (vm.owner(wt), vm.branches()["work"]) == (WORKER, WORKER)

    # a checkout with no stamp, and a directory in no checkout
    out = vm.reclaim(fresh, LEAD)
    assert out.returncode == 1 and f"{fresh} has no owner to take it back from" in out.stderr, out.stderr
    assert f"run `git -C {fresh} status` and it is yours" in out.stderr and vm.owner(fresh) == ""
    out = vm.reclaim(vm.root, LEAD)
    assert out.returncode == 1 and "is in no checkout" in out.stderr, out.stderr

    assert vm.reclaim(ws, LEAD).returncode == 0, "and the gone owner's still goes"


def test_the_command_reads_who_holds_a_handle_from_origin_mains_registry_as_the_hook_does(vm):
    """`hook.sh` hands the hook origin/main's registry, and the command an agent runs judges by the same one: a checkout's own
    copy can be older than its coordinator's handle, and a worker's edit to it makes the worker no coordinator."""
    ws, repo = vm.root / "ws", vm.root / "repo"
    shutil.copytree(HOOK.parent, repo / "tools/agent-guard", ignore=shutil.ignore_patterns("tests", "__pycache__"))
    own = repo / ".agents/skills/using-slack/registry.json"
    own.parent.mkdir(parents=True)
    own.write_text(json.dumps({"handles": [{"handle": "infra", "holder": LEAD}]}))
    for args in (["init", "-q", "-b", "main"], ["add", "."], ["commit", "-q", "-m", "a"],
                 ["update-ref", "refs/remotes/origin/main", "HEAD"]):
        subprocess.run(["/usr/bin/git", "-C", str(repo), *args], check=True, capture_output=True, env=vm.environment)
    own.write_text(json.dumps({"handles": [{"handle": "infra", "holder": WORKER}]}))      # the checkout's copy, edited

    def reclaim(agent: str) -> subprocess.CompletedProcess:
        env = {k: v for k, v in vm.environment.items() if k != "AGENT_GUARD_REGISTRY"}
        return subprocess.run([sys.executable, repo / "tools/agent-guard/worktree_guard.py", "reclaim", str(ws)],
                              capture_output=True, text=True, env={**env, "CURSOR_CONVERSATION_ID": agent})

    assert vm.ask("git status", PAST) == {"permission": "allow"}
    vm.long_ago()
    assert vm.ask("git status", LEAD, vm.root / "fresh") == {"permission": "allow"}
    out = reclaim(WORKER)
    assert out.returncode == 1 and f"{WORKER} holds no handle in the Slack registry (origin/main's .agents/skills/" in out.stderr
    assert vm.owner(ws) == PAST
    subprocess.run(["/usr/bin/git", "-C", str(repo), "update-ref", "-d", "refs/remotes/origin/main"], check=True,
                   env=vm.environment)
    out = reclaim(LEAD)                                                         # no origin/main to read: the checkout's own
    assert out.returncode == 1 and f"{LEAD} holds no handle in the Slack registry ({own})" in out.stderr, out.stderr
    subprocess.run(["/usr/bin/git", "-C", str(repo), "update-ref", "refs/remotes/origin/main", "HEAD"], check=True,
                   env=vm.environment)
    out = reclaim(LEAD)
    assert out.returncode == 0 and f"{ws} is now {LEAD}'s (@infra)" in out.stdout and vm.owner(ws) == LEAD, out.stderr


def test_no_test_judges_by_the_checkouts_slack_registry_or_origin_mains(before_any_test, scratch_registry, tmp_path):
    """Who holds a handle today is no input of a test, and where `origin/main` can be read `worktree_guard` reads its
    registry through `git show`, which the suite's read guard can't see. One fixture (conftest's `scratch_registry`) names a
    registry that holds no handle in the session's environment, so in a module's copy of it as in a test's own: `reclaim`,
    which says which registry it judged by, names that one."""
    for env in (os.environ, before_any_test):
        out = subprocess.run([sys.executable, WORKTREE_GUARD, "reclaim", str(tmp_path)], capture_output=True, text=True,
                             env={**env, "CURSOR_CONVERSATION_ID": LEAD})
        assert out.returncode == 1 and f"{LEAD} holds no handle in the Slack registry ({scratch_registry})" in out.stderr, out.stderr


def test_gone_is_read_from_a_record_that_reaches_back_and_that_the_shell_hook_is_writing(vm, event_log):
    """Gone is no hook run in two hours by the VM's record of hook runs. An owner the record holds no run of is gone only if
    the record reaches back that far (it rotates by size), and only if it holds the caller's own shell-hook run: where no
    shell hook runs, no agent's commands are recorded, and every one of them would read as gone. A run of the caller's by
    another hook shows nothing of that: where Cursor hands shell commands to the Task hook and to no shell hook, the record
    holds the caller's runs and none of an owner that is at work."""
    ws, typed = vm.root / "ws", "python3 tools/agent-guard/worktree_guard.py reclaim"
    assert vm.ask("git status", PAST) == {"permission": "allow"}
    event_log.write_text(event_log.read_text().replace('"hook": "shell_hook.py"', '"hook": "task_hook.py"'))
    vm.long_ago()                                    # the record: one Task-hook run of the past worker's, long ago
    out = vm.reclaim(ws, LEAD)
    assert out.returncode == 1 and "holds no shell-hook run of yours in the last 2 h" in out.stderr, out.stderr
    assert "`python3 tools/agent-guard/inbox.py guard` says why" in out.stderr and vm.owner(ws) == PAST

    typed_here = {"hook_event_name": "beforeShellExecution", "command": f"{typed} {ws}", "conversation_id": LEAD, "cwd": str(ws)}
    subprocess.run([sys.executable, HOOK.parent / "task_hook.py"], input=json.dumps(typed_here), check=True, capture_output=True,
                   text=True, env=vm.environment)    # the caller's command, handed to the Task hook: its one run, just now
    runs = [json.loads(line) for line in event_log.read_text().splitlines()]
    assert [(run["hook"], run["conversation_id"]) for run in runs] == [("task_hook.py", PAST), ("task_hook.py", LEAD)]
    out = vm.reclaim(ws, LEAD)
    assert out.returncode == 1 and "holds no shell-hook run of yours in the last 2 h" in out.stderr, out.stdout + out.stderr
    assert vm.owner(ws) == PAST, "a Task-hook run of the caller's evicts no one"

    event_log.unlink()
    assert vm.ask(f"{typed} {ws}", LEAD) == {"permission": "allow"}             # the record starts with this run
    out = vm.reclaim(ws, LEAD)
    assert out.returncode == 1 and "reaches back only " in out.stderr and "Try again after " in out.stderr, out.stderr
    assert f"{PAST} may have run a command before that" in out.stderr and vm.owner(ws) == PAST
    assert "worktree_guard.py reclaim" not in vm.ask("git switch -c other", LEAD)["agent_message"], "nor does a refusal name it yet"

    vm.long_ago()
    assert vm.ask(f"{typed} {ws}", LEAD) == {"permission": "allow"}
    assert vm.reclaim(ws, LEAD).returncode == 0 and vm.owner(ws) == LEAD


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


@pytest.mark.parametrize("form", [
    'git worktree remove "$p"', 'for p in /tmp/wt-a /tmp/wt-b; do git worktree remove --force "$p"; done',
    "for p in $(git worktree list --porcelain | awk '/^worktree/{print $2}'); do git worktree remove \"$p\"; done",
    'git worktree prune --expire "$WHEN"', "git worktree prune -v $DRY", "git worktree list $FLAGS",
    'git -C "$R" worktree list --porcelain', 'git worktree move "$a" "$b"', 'git worktree lock "$p"',
    'git worktree repair "$p"', "bash -c 'git worktree remove \"$p\"'",
])
def test_a_worktree_command_that_adds_none_passes_over_a_variable(clones, form):
    """The rule on words the shell has yet to expand is about a branch the guard can't name: of `git worktree`, only `add`
    creates or checks one out. The rule refused every `git worktree` with a `$` in it, so worktrees were removed one literal
    command at a time."""
    for clone in ("verity", "notes"):
        assert clones(form, clone) == {"permission": "allow"}, (form, clone)


@pytest.mark.parametrize("form", [
    "git worktree add ../wt -b $B", 'git worktree add "$D" lane/mine', "git worktree add -b x ../wt $REV",
    "git worktree add ../wt $(cat branch.txt)", "git worktree $SUB ../wt -b x", 'git -C "$R" worktree add ../wt lane/mine',
    'for b in a b; do git worktree add "../wt-$b" -b "$b"; done',
])
def test_a_worktree_add_over_a_variable_is_still_refused(clones, form):
    for clone in ("verity", "notes"):
        out = clones(form, clone)
        assert out["permission"] == "deny" and "expands only after the guard has read it" in out["agent_message"], (form, out)
        assert "Name it literally (`git worktree add ../wt-<name> -b <branch>`)" in out["agent_message"], (form, out)


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
    for command in ("git config user.name", "git config core.abbrev 12", "git config --get remote.origin.fetch"):
        assert ask(command)["permission"] == "allow", command


#: git commands that name an author or committer of their own, by each form the rule reads: git's own `-c` and `--config-env`,
#: a variable assigned ahead of the command, `commit --author`, and a `git config` that writes the checkout's own
BY_OPTION = [
    "git -c user.name=a -c user.email=a@a commit -m x", "git -c user.email=a@a commit -m x", "git -c User.Name=a commit -m x",
    "git -c author.name=a commit -m x", "git -c committer.email=a@a revert --no-edit HEAD",
    "git --config-env=user.name=NAME commit -m x", "git --config-env user.email=ADDRESS commit -m x",
    "git add -A && git -c user.name=a -c user.email=a@a commit -m x", "bash -lc 'git -c user.name=a commit -m x'",
]
BY_VARIABLE = [
    "GIT_AUTHOR_NAME=a GIT_AUTHOR_EMAIL=a@a GIT_COMMITTER_NAME=a GIT_COMMITTER_EMAIL=a@a git commit -m x",
    "GIT_AUTHOR_NAME=a git commit -m x", "GIT_AUTHOR_EMAIL=a@a git commit -m x", "GIT_COMMITTER_NAME=a git commit -m x",
    "GIT_COMMITTER_EMAIL=a@a git cherry-pick lane/mine", "env GIT_AUTHOR_EMAIL=a@a git commit -m x",
    "tmux send-keys 'GIT_AUTHOR_NAME=a git commit -m x' C-m",
]
BY_AUTHOR = ["git commit --author='A <a@a>' -m x", "git commit -m x --author 'A <a@a>'",
             "git commit --amend --no-edit --author='A <a@a>'"]
BY_CONFIG = ["git config user.name a", "git config user.email a@a", "git config --local user.name a",
             "git config --worktree user.email a@a", "git config set user.email a@a", "git config Author.Name a"]
OWN_IDENTITY = BY_OPTION + BY_VARIABLE + BY_AUTHOR + BY_CONFIG
IDENTITY_FIX = "commit with the machine's configured identity; pass no name or address"
#: and what both refusals about a variable say clears one a shell already holds
UNSET = "`unset GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL GIT_COMMITTER_NAME GIT_COMMITTER_EMAIL` clears"


@pytest.mark.parametrize("form", OWN_IDENTITY)
def test_a_git_that_names_an_author_or_committer_of_its_own_is_refused_in_a_checkout_of_verity_or_infra(clones, form):
    """A commit in the queue's repositories carries the machine's configured identity, by which their history shows who wrote
    what. A commit made under a name and an address passed on the command line shows an author nobody can place, so a git
    command that passes one there is refused, and the refusal says how to commit."""
    for clone in ("verity", "infra"):
        out = clones(form, clone)
        assert out["permission"] == "deny" and IDENTITY_FIX in out["agent_message"] and UNSET in out["agent_message"], (form, clone, out)


@pytest.mark.parametrize("form", OWN_IDENTITY)
def test_a_scratch_repository_and_any_other_take_any_identity(clones, tmp_path, form):
    """Tests commit in throwaway repositories under a temporary directory all the time, under any name: a repository with no
    `origin`, or one that is not the queue's, takes the same commands."""
    subprocess.run(["/usr/bin/git", "init", "-q", str(tmp_path / "scratch")], check=True, capture_output=True,
                   env={**os.environ, **GIT_ENV})
    for clone in ("scratch", "notes"):
        assert clones(form, clone) == {"permission": "allow"}, (form, clone)


def test_the_identity_rule_judges_the_repository_the_command_runs_in(clones, tmp_path):
    """The rule places a command by the directory it starts in and git's own `-C`: a worktree of a verity checkout is one, and
    a scratch repository named from inside one is not. A command the guard can't place, as one that names its repository with
    `GIT_DIR` or `--git-dir`, passes: the rule refuses only where it can tell."""
    scratch, env = tmp_path / "scratch", {**os.environ, **GIT_ENV}
    for args in (["init", "-q", str(scratch)], ["-C", str(tmp_path / "verity"), "worktree", "add", "-q", str(tmp_path / "wt"), "-b", "wt"]):
        subprocess.run(["/usr/bin/git", *args], check=True, capture_output=True, env=env)
    commit = "-c user.name=a -c user.email=a@a commit -m x"
    for form, clone in ((f"git {commit}", "wt"), (f"git -C {{V}} {commit}", "notes"), (f"git -C {{V}} {commit}", "scratch"),
                        (f"git -C ../verity {commit}", "scratch"), (f"cd /tmp && git -C {{V}} {commit}", "notes"),
                        (f"git -C {tmp_path}/wt {commit}", "notes")):
        out = clones(form, clone)
        assert out["permission"] == "deny" and IDENTITY_FIX in out["agent_message"], (form, clone, out)
    for form in (f"git -C {scratch} {commit}", f"git -C ../scratch {commit}", f"git -C {{N}} {commit}",
                 f"cd /tmp && git -C {scratch} {commit}", f"cd {scratch} && git {commit}",
                 f"cd {scratch} && git init -q . && git {commit}", f"GIT_DIR={scratch}/.git git {commit}",
                 f"git --git-dir={scratch}/.git {commit}", f"git init -q {tmp_path}/new && git -C {tmp_path}/new {commit}"):
        assert clones(form, "verity") == {"permission": "allow"}, form


COMMIT = "-c user.name=a -c user.email=a@a commit -m x"
#: `cd DIR && git ...` that names an identity, with the clone it is typed in: DIR a checkout of verity or infra, a worktree of
#: one or a directory inside one ({T} is the directory the clones are in, {S} a scratch repository, {V}/sub a directory)
CD_INTO_A_CHECKOUT = [
    (f"cd {{V}} && git {COMMIT}", "notes"), (f"cd {{V}} && git {COMMIT}", "scratch"), (f"cd {{T}}/infra && git {COMMIT}", "notes"),
    (f"cd {{T}}/wt && git {COMMIT}", "scratch"), (f"cd ../verity && git {COMMIT}", "notes"), (f"cd sub && git {COMMIT}", "verity"),
    (f"cd {{V}}/sub && git {COMMIT}", "scratch"), (f"cd {{V}}; git {COMMIT}", "notes"), (f"cd {{V}}\ngit {COMMIT}", "notes"),
    (f"cd {{V}} 2>/dev/null && git {COMMIT}", "notes"), (f"cd {{V}} && git add -A && git {COMMIT}", "notes"),
    (f"cd {{V}} && git diff --cached --quiet || git {COMMIT}", "notes"), (f"cd {{V}} && git status | tail -1; git {COMMIT}", "notes"),
    (f"cd {{V}} && git status |& tail -1; git {COMMIT}", "notes"), (f"cd {{V}} && env LC_ALL=C git add -A && git {COMMIT}", "notes"),
    ("cd {V} && git commit --author='A <a@a>' -m x", "notes"), ("cd {V} && GIT_AUTHOR_NAME=a git merge lane/mine", "notes"),
    ("cd {V} && env GIT_COMMITTER_EMAIL=a@a git commit -m x", "notes"), ("cd {V} && git config user.email a@a", "notes"),
    (f"cd {{S}} && cd {{V}} && git {COMMIT}", "notes"), (f"cd {{T}} && cd verity && git {COMMIT}", "scratch"),
    (f"cd {{T}} && git -C verity {COMMIT}", "scratch"), (f"(cd {{V}} && git {COMMIT})", "notes"),
    (f"cd {{S}} && (cd {{V}} && git {COMMIT})", "notes"), (f"bash -lc 'cd {{V}} && git {COMMIT}'", "notes"),
    (f"tmux send-keys 'cd {{V}} && git {COMMIT}' C-m", "notes"),
]
#: the same after a `cd` into a scratch repository or the notes, typed in a checkout
CD_OUT_OF_A_CHECKOUT = [
    (f"cd {{S}} && git {COMMIT}", "verity"), (f"cd {{S}} && git {COMMIT}", "infra"), (f"cd {{S}} && git {COMMIT}", "wt"),
    (f"cd {{N}} && git {COMMIT}", "verity"), (f"cd ../scratch && git {COMMIT}", "verity"), (f"cd {{S}}; git {COMMIT}", "verity"),
    (f"cd {{V}} && cd {{S}} && git {COMMIT}", "verity"), (f"cd {{S}} && git init -q . && git {COMMIT}", "verity"),
    (f"(cd {{S}} && git {COMMIT})", "verity"), (f"bash -lc 'cd {{S}} && git {COMMIT}'", "verity"),
]
#: and where the `cd`s leave the directory in doubt, each typed where a wrong guess would refuse it: a directory that does not
#: exist, one the shell names later, a `cd` that may not run or runs in a shell of its own, and what the guard follows for no rule
CD_IN_DOUBT = [
    (f"cd {{T}}/nowhere && git {COMMIT}", "verity"), (f"cd {{V}}/nowhere; cd ..; git {COMMIT}", "scratch"),
    (f'cd "$D" && git {COMMIT}', "verity"), (f"cd $(mktemp -d) && git init -q . && git {COMMIT}", "verity"),
    (f"cd ~ && git {COMMIT}", "verity"), (f"(cd {{V}} && git status); git {COMMIT}", "scratch"),
    (f"cd {{V}} & git {COMMIT}", "scratch"), (f"true || cd {{V}}; git {COMMIT}", "scratch"),
    (f"if false; then true; cd {{V}}; fi; git {COMMIT}", "scratch"), (f"f() {{ cd {{V}}; }}; git {COMMIT}", "scratch"),
    (f"X=`cd {{V}} && git rev-parse HEAD`; git {COMMIT}", "scratch"), (f"echo $(cd {{V}} && pwd); git {COMMIT}", "scratch"),
    (f"nohup cd {{V}}; git {COMMIT}", "scratch"), (f"eval 'cd {{S}}' && git {COMMIT}", "verity"),
    (f"pushd {{S}} && git {COMMIT}", "verity"), (f"source ./env.sh && cd {{V}} && git {COMMIT}", "notes"),
    (f"cd {{V}} && GIT_DIR={{S}}/.git git {COMMIT}", "notes"), (f"cd {{V}} && git --git-dir={{S}}/.git {COMMIT}", "notes"),
]


def test_the_identity_rule_follows_a_cd_written_out_and_passes_what_a_cd_leaves_in_doubt(clones, tmp_path, monkeypatch):
    """`cd CHECKOUT && git ...` is how a command is usually typed, so the identity rule, and no other, judges the git after a
    plain `cd` where the `cd` leaves the shell: refused in a checkout of verity or infra, a worktree of one or a directory
    inside one, passed in a scratch repository or the notes. Where the `cd`s leave the directory in doubt, the command passes."""
    env, hook = {**os.environ, **GIT_ENV}, guard_module("shell_hook", monkeypatch)
    for args in (["init", "-q", str(tmp_path / "scratch")], ["-C", str(tmp_path / "verity"), "worktree", "add", "-q", str(tmp_path / "wt"), "-b", "wt"]):
        subprocess.run(["/usr/bin/git", *args], check=True, capture_output=True, env=env)
    (tmp_path / "verity" / "sub").mkdir()

    def typed(form: str) -> str:
        return form.replace("{T}", str(tmp_path)).replace("{S}", str(tmp_path / "scratch"))
    passed = [(form, clone) for form, clone in CD_INTO_A_CHECKOUT
              if (out := clones(typed(form), clone))["permission"] != "deny" or IDENTITY_FIX not in out["agent_message"]]
    refused = [(form, clone) for form, clone in CD_OUT_OF_A_CHECKOUT + CD_IN_DOUBT if clones(typed(form), clone) != {"permission": "allow"}]
    assert not passed and not refused, json.dumps({"passed": passed, "refused": refused})
    for form, _ in CD_INTO_A_CHECKOUT + CD_OUT_OF_A_CHECKOUT + CD_IN_DOUBT:      # one directory for each command `flattened` gives
        assert len(hook.cd_dirs(typed(form), str(tmp_path))[0]) == len(hook.flattened(typed(form))), form


#: `git config --global` that writes the machine's identity, and the reads and other writes beside it
GLOBAL_IDENTITY = [
    "git config --global user.name a", "git config --global user.email a@a", "git config --global author.name a",
    "git config --global committer.email a@a", "git config set --global user.email a@a", "git config --global User.Email a@a",
    "git config --global user.name 'A B'", "/usr/bin/git config --global user.email a@a", "git -C {N} config --global user.name a",
    "cd /tmp && git config --global user.name a", "bash -lc 'git config --global user.email a@a'",
    "GIT_DIR=/nowhere git config --global user.name a", "XDG_CONFIG_HOME=/tmp/x git config --global user.name a",
]
#: what its refusal says to do, which holds on a machine that has an identity and on one that was given none
GLOBAL_FIX = ("`git config --global user.name` and `git config --global user.email` read the identity the machine has; if either "
              "prints nothing, the machine was not given it: stop and tell your coordinator rather than set one.")


@pytest.mark.parametrize("form", GLOBAL_IDENTITY)
def test_a_git_config_that_writes_the_machines_global_identity_is_refused_wherever_it_is_typed(clones, tmp_path, form):
    """The identity every checkout on a machine commits with sits in its global configuration, so one `git config --global
    user.name VALUE` typed anywhere changes the author of verity's and infra's commits. It is refused in every repository and
    outside one. The refusal says how to read the identity the machine has, to stop and tell the coordinator where it was given
    none rather than set one, and where a scratch repository's name goes instead."""
    subprocess.run(["/usr/bin/git", "init", "-q", str(tmp_path / "scratch")], check=True, capture_output=True,
                   env={**os.environ, **GIT_ENV})
    for clone in ("verity", "infra", "notes", "scratch", "state"):
        out = clones(form, clone)
        assert out["permission"] == "deny" and GLOBAL_FIX in out["agent_message"], (form, clone, out)
        assert "git -C <scratch> config user.name" in out["agent_message"] and "git -C <scratch> -c user.name=" in out["agent_message"]


@pytest.mark.parametrize("form", [
    "git config --global --get user.email", "git config --global user.name", "git config get --global user.email",
    "git config --global core.abbrev 12", "git config --global user.signingkey K", "git config --global --unset user.name",
    "git config --system user.name a", "git config --file /tmp/x user.name a", "git config -f /tmp/x user.email a@a",
    "HOME=/tmp/home git config --global user.name a", "GIT_CONFIG_GLOBAL=/tmp/x git config --global user.email a@a",
])
def test_reading_the_global_identity_and_writing_any_other_global_key_passes(clones, form):
    """`git config --global user.name` and `--get` read the machine's identity, a `--global` write of another key leaves it
    alone, and a command that names another file as the global configuration (`HOME=DIR git config --global ...`) writes
    there. `--system` and `--file` are not judged."""
    for clone in ("verity", "infra", "notes"):
        assert clones(form, clone) == {"permission": "allow"}, (form, clone)


@pytest.mark.parametrize("form", [
    "git log --author=NAME -3", "git shortlog -sn --author NAME", "git rev-list --count --author=a HEAD",
    "git config --get user.email", "git config user.email", "git config --local user.name", "git var GIT_AUTHOR_IDENT",
    "git config --unset user.name", "git config core.abbrev 12", "git commit -m x", "git commit -m 'refuse --author on a commit'",
    "git commit --amend --no-edit --reset-author", "git -c commit.gpgsign=false commit -m x", "git -c user.signingkey=K commit -S -m x",
    "GIT_AUTHOR_DATE=@1 GIT_COMMITTER_DATE=@1 git commit -m x",
    "unset GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL; git commit -m x",
])
def test_reading_an_author_or_the_configured_identity_passes_in_a_checkout_of_verity_or_infra(clones, form):
    """`--author` selects commits in `git log`, `git shortlog` and `git rev-list`; `git config KEY`, `git config --get KEY` and
    `git var GIT_AUTHOR_IDENT` read the identity; and a commit that passes none carries the machine's."""
    for clone in ("verity", "infra"):
        assert clones(form, clone) == {"permission": "allow"}, (form, clone)


#: commands that leave a variable naming a commit's author or committer set in the shell that runs them ({S} is a scratch
#: repository): an `export`, a `declare` or `typeset`, an assignment with no command after it, each also in what a body, a
#: subshell or a carrier runs, after a comment, and in a text the hook can't split
SET_IN_THE_SHELL = [
    "export GIT_AUTHOR_NAME=x GIT_AUTHOR_EMAIL=x@x GIT_COMMITTER_NAME=x GIT_COMMITTER_EMAIL=x@x",
    "export GIT_AUTHOR_NAME=x", "export GIT_AUTHOR_EMAIL=x@x", "export GIT_COMMITTER_NAME=x", "export GIT_COMMITTER_EMAIL=x@x",
    "export GIT_AUTHOR_NAME='A B'", 'export GIT_COMMITTER_NAME="$USER"', "export GIT_AUTHOR_DATE=@0 GIT_COMMITTER_EMAIL=x@x",
    "export PATH=/x:$PATH GIT_AUTHOR_NAME=x", "export GIT_AUTHOR_NAME", "export GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL",
    "GIT_AUTHOR_NAME=x; export GIT_AUTHOR_NAME", "declare -x GIT_AUTHOR_NAME=x", "typeset -x GIT_COMMITTER_EMAIL=x@x",
    "declare -gx GIT_COMMITTER_NAME=x", "declare GIT_AUTHOR_NAME=x", "readonly GIT_AUTHOR_EMAIL=x@x", "export -p GIT_AUTHOR_NAME",
    "GIT_AUTHOR_NAME=x", "GIT_AUTHOR_NAME=x GIT_AUTHOR_EMAIL=x@x", "X=1 GIT_COMMITTER_NAME=x",
    "GIT_COMMITTER_EMAIL=x@x; git -C {S} commit -m x", "set -a; GIT_AUTHOR_NAME=x; set +a",
    "export GIT_AUTHOR_NAME=x && git -C {S} commit --allow-empty -m x",
    "export GIT_AUTHOR_NAME=x GIT_AUTHOR_EMAIL=x@x; git -C {S} commit -m x; unset GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL",
    "cd {S} && export GIT_AUTHOR_NAME=x && git commit -m x", "command export GIT_AUTHOR_NAME=x",
    "if true; then export GIT_AUTHOR_NAME=x; fi", "for n in a b; do export GIT_AUTHOR_NAME=$n; done",
    "f() { export GIT_AUTHOR_NAME=x; }; f", "(export GIT_AUTHOR_NAME=x; git -C {S} commit -m x)",
    "bash -c 'export GIT_AUTHOR_NAME=x; git -C {S} commit -m x'", "bash -lc 'GIT_COMMITTER_NAME=x'",
    "eval 'export GIT_AUTHOR_NAME=x'", "tmux send-keys 'export GIT_AUTHOR_NAME=x' C-m",
    "# a throwaway name\nexport GIT_AUTHOR_NAME=x", "# a throwaway name\nGIT_AUTHOR_NAME=x",
    "git -C {S} status # first\nexport GIT_AUTHOR_NAME=x", "export GIT_AUTHOR_NAME=x # for the scratch commits",
    "# a throwaway name\ncd {S} && export GIT_AUTHOR_NAME=x", "# in a pane\ntmux send-keys 'export GIT_AUTHOR_NAME=x' C-m",
    "export GIT_AUTHOR_NAME=$'it\\'s'", "echo $'it\\'s' && export GIT_AUTHOR_NAME=x", "echo $'it\\'s'\nexport GIT_AUTHOR_NAME=x",
]
#: and commands that leave none there: the names ahead of the one command, on its line or on continued ones, which the rule
#: above judges by where its git runs; an `unset`; a read; an `export` of any other variable; and an export that is only
#: words, in a quoted string, a comment or a heredoc's body
NOT_SET_IN_THE_SHELL = [
    "GIT_AUTHOR_NAME=a GIT_AUTHOR_EMAIL=a@a GIT_COMMITTER_NAME=a GIT_COMMITTER_EMAIL=a@a git -C {S} commit --allow-empty -m x",
    "env GIT_AUTHOR_NAME=a GIT_AUTHOR_EMAIL=a@a git -C {S} commit -m x", "git -C {S} -c user.name=a -c user.email=a@a commit -m x",
    "cd {S} && GIT_AUTHOR_NAME=a GIT_COMMITTER_NAME=a git commit -m x", "GIT_AUTHOR_NAME=a GIT_COMMITTER_NAME=a python3 fixture.py",
    "unset GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL GIT_COMMITTER_NAME GIT_COMMITTER_EMAIL", "unset GIT_AUTHOR_NAME && git -C {S} status",
    'echo "$GIT_AUTHOR_NAME"', 'echo "${GIT_COMMITTER_EMAIL-unset}"', '[ -n "${GIT_AUTHOR_NAME+x}" ] && echo set',
    "declare -p GIT_AUTHOR_NAME", "compgen -e | rg GIT_", "git -C {S} var GIT_AUTHOR_IDENT",
    "export GIT_AUTHOR_DATE=@0 GIT_COMMITTER_DATE=@0", "GIT_AUTHOR_DATE=@0", "export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0",
    "export PATH=$HOME/bin:$PATH", "export NAME=GIT_AUTHOR_NAME",
    'git -C {S} commit --allow-empty -m "export GIT_AUTHOR_NAME=x is refused"', 'rg -n "export GIT_AUTHOR_NAME" {S}',
    "echo 'GIT_AUTHOR_NAME=x; export GIT_AUTHOR_NAME'", "# export GIT_AUTHOR_NAME=x\ngit -C {S} status",
    "# a throwaway commit\nGIT_AUTHOR_NAME=a GIT_AUTHOR_EMAIL=a@a git -C {S} commit --allow-empty -m x",
    "# a throwaway commit\nGIT_AUTHOR_NAME='A (B)' GIT_AUTHOR_EMAIL=a@a git -C {S} commit --allow-empty -m 'x; GIT_AUTHOR_NAME=y'",
    "GIT_AUTHOR_NAME=a GIT_AUTHOR_EMAIL=a@a \\\n  GIT_COMMITTER_NAME=a GIT_COMMITTER_EMAIL=a@a \\\n  git -C {S} commit --allow-empty -m x",
    "# a throwaway commit\nGIT_AUTHOR_NAME=a GIT_AUTHOR_EMAIL=a@a \\\n  git -C {S} commit --allow-empty -m x",
    "cat > {S}/note <<'EOF'\n# how not to\nexport GIT_AUTHOR_NAME=x\nEOF",
]
#: what its refusal says to do: where a scratch repository's name goes instead (and UNSET, for one a shell already holds)
SHELL_FIX = ("put the names ahead of it, `GIT_AUTHOR_NAME=NAME GIT_AUTHOR_EMAIL=ADDRESS GIT_COMMITTER_NAME=NAME "
             "GIT_COMMITTER_EMAIL=ADDRESS git -C <scratch> commit ...`")


@pytest.fixture
def scratch(tmp_path):
    """A scratch repository beside the clones, and a form with its path for {S}."""
    subprocess.run(["/usr/bin/git", "init", "-q", str(tmp_path / "scratch")], check=True, capture_output=True,
                   env={**os.environ, **GIT_ENV})
    return lambda form: form.replace("{S}", str(tmp_path / "scratch"))


@pytest.mark.parametrize("form", SET_IN_THE_SHELL)
def test_a_command_that_leaves_an_author_or_committer_set_in_the_shell_is_refused_wherever_it_is_typed(clones, scratch, form):
    """A shell keeps what it is told to export, and git reads `GIT_AUTHOR_NAME` and its three kin from its environment in every
    repository: a name exported for a commit in a scratch repository is the author of the next commit in verity's checkout too.
    The hook reads a command and never a shell's environment, so the command that puts the name there is the one it can
    refuse: in every repository and outside one, and, since a wrong pass can't be taken back, also where the name would die
    with a subshell or a carrier. The refusal says where a scratch repository's name goes and what clears one already set."""
    for clone in ("verity", "infra", "notes", "scratch", "state"):
        out = clones(scratch(form), clone)
        assert out["permission"] == "deny" and SHELL_FIX in out["agent_message"] and UNSET in out["agent_message"], (form, clone, out)
        assert "-c user.name=NAME -c user.email=ADDRESS" in out["agent_message"]


@pytest.mark.parametrize("form", NOT_SET_IN_THE_SHELL)
def test_naming_an_author_ahead_of_one_command_unsetting_or_reading_one_and_any_other_export_pass(clones, scratch, form):
    """The names ahead of one command reach that command alone, which the rule for a git that names an identity judges by
    where the git runs; `unset` is the cure; a read changes nothing; `GIT_AUTHOR_DATE` and every other variable are not an
    identity; and an export that is only the words of a message, a search, a comment or a file being written sets nothing."""
    for clone in ("verity", "infra", "notes", "scratch", "state"):
        assert clones(scratch(form), clone) == {"permission": "allow"}, (form, clone)


def test_a_name_exported_for_throwaway_commits_is_stopped_before_a_later_merge_in_the_checkout_carries_it(clones, scratch):
    """Three commands, each typed on its own: an export of a made-up author and committer, a throwaway commit in a scratch
    repository, and later a merge in verity's checkout. The shell that ran the first still held the names at the third, whose
    merge commit went into the history under them. The second and third name no identity and pass as typed, as they must:
    the hook can't see what a shell holds. So the first is refused, and the form its refusal gives for the throwaway commit,
    the names ahead of that one command, passes."""
    export = ("export GIT_AUTHOR_NAME=x GIT_AUTHOR_EMAIL=x@x GIT_COMMITTER_NAME=x GIT_COMMITTER_EMAIL=x@x "
              "GIT_AUTHOR_DATE=@0 GIT_COMMITTER_DATE=@0")
    commit, merge = scratch("git -C {S} commit -q --allow-empty -m throwaway"), "git merge --no-edit lane/mine"
    out = clones(export, "verity")
    assert out["permission"] == "deny" and SHELL_FIX in out["agent_message"] and UNSET in out["agent_message"], out
    assert clones(commit, "verity") == {"permission": "allow"} and clones(merge, "verity") == {"permission": "allow"}
    ahead = "GIT_AUTHOR_NAME=x GIT_AUTHOR_EMAIL=x@x GIT_COMMITTER_NAME=x GIT_COMMITTER_EMAIL=x@x GIT_AUTHOR_DATE=@0 GIT_COMMITTER_DATE=@0"
    assert clones(f"{ahead} {commit}", "verity") == {"permission": "allow"}


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


def test_a_command_the_hook_cant_split_weighs_nothing_and_says_so(monkeypatch, capsys):
    shell_hook = guard_module("shell_hook", monkeypatch)
    assert shell_hook.weight("echo 'unclosed") == 0 and shell_hook.blocks_text("echo 'unclosed") == 0.0
    err = capsys.readouterr().err
    assert "shell_hook: a command the hook can't split weighs nothing (No closing quotation)" in err
    assert "shell_hook: a command the hook can't split blocks for no time (No closing quotation)" in err


def test_unreadable_memory_is_not_known_and_said_while_an_empty_inbox_says_nothing(tmp_path, monkeypatch, capsys):
    shell_hook = guard_module("shell_hook", monkeypatch)
    monkeypatch.setenv("AGENT_GUARD_MEMINFO", str(tmp_path / "meminfo"))
    assert shell_hook.available() is None
    assert f"shell_hook: {tmp_path / 'meminfo'} unread, so memory is not known (FileNotFoundError" in capsys.readouterr().err
    monkeypatch.setattr(shell_hook, "INBOX", str(tmp_path))
    assert shell_hook.delivery("bc-x") is None and capsys.readouterr().err == ""


def test_a_branch_registry_that_does_not_parse_reads_as_none_registered_and_says_so(tmp_path, monkeypatch, capsys):
    worktree_guard = guard_module("worktree_guard", monkeypatch)
    monkeypatch.setattr(worktree_guard, "STATE", tmp_path)
    assert worktree_guard.registry() == {} and capsys.readouterr().err == ""
    (tmp_path / "branches.json").write_text("{")
    assert worktree_guard.registry() == {}
    assert "branches.json unreadable, read as no branch registered (JSONDecodeError" in capsys.readouterr().err
