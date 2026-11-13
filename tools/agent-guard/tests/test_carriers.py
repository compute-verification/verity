"""What a command has another program run meets every rule a typed command meets: what tmux types or starts (`send-keys`,
`new-session`, `new-window`, `split-window`, `respawn-pane`, `run-shell`, `-c`, with or without `-f CONFIG` and `-t TARGET`), and
what `bash -c`, `nohup`, `setsid`, `timeout`, `env`, `xargs`, `eval` and `watch` run. For each carrier one inner command per rule
is refused (a build, a process listing, the environment, a secret file, a push to another agent's branch) and one passes."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parents[1]
HOOK = GUARD / "shell_hook.py"
GIT_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
ME, OTHER = "bc-me", "bc-other"

#: each carrier, with `{}` where the inner command goes: quoted as one word when the carrier takes it so, else as its words
CARRIERS = [
    "tmux send-keys -t w:0.0 {q} C-m",
    "tmux -f /exec-daemon/tmux.portal.conf send-keys -t w:0.0 {q} Enter",
    "tmux send -l {q}",
    "tmux new-session -d -s s {q}",
    "tmux -f /exec-daemon/tmux.portal.conf new-session -d -s s -- {w}",
    "tmux new -d -s s -c . {q}",
    "tmux new-window -t s {q}",
    "tmux neww {q}",
    "tmux split-window -h -t s:0 {q}",
    "tmux splitw -v {w}",
    "tmux respawn-pane -k -t s:0.1 {q}",
    "tmux run-shell {q}",
    "tmux run -b -t s {q}",
    "tmux new -d \\; send-keys {q} C-m",
    "tmux -c {q}",
    "bash -c {q}",
    "bash -lc {q}",
    "sh -c {q}",
    "nohup {w} &",
    "setsid {w}",
    "timeout 60 {w}",
    "env FOO=1 {w}",
    "xargs {w}",
    "eval {q}",
    "watch -n 5 {q}",
]
#: rule -> (an inner command it refuses, what its refusal says, an inner command it lets through)
RULES = {
    "build": ("lake build", "builds and tests run on our nodes", "make docs"),
    "listing": ("ps aux", "full command lines", "ps -eo pid,etimes,comm"),
    "environment": ("printenv", "environment variable", "compgen -e"),
    "secret": ("cat ~/.ssh/id_rsa", "would print a private key", "ls ~/.ssh"),
    "push": ("git push origin HEAD:refs/heads/theirs", f"belongs to agent {OTHER}", "git push origin HEAD:refs/heads/mine"),
}


@pytest.fixture(scope="module")
def ask(tmp_path_factory):
    """The shell hook's verdict on a command agent ME runs in a clone of verity (its origin's URL; nothing is fetched or
    pushed) where branch `theirs` is OTHER's."""
    tmp = tmp_path_factory.mktemp("carriers")
    state, ws = tmp / "state", tmp / "ws"
    state.mkdir()
    (state / "branches.json").write_text(json.dumps({"theirs": OTHER}))
    for args in (["init", "-q", "--bare", "-b", "main", str(tmp / "o.git")], ["clone", "-q", str(tmp / "o.git"), str(ws)],
                 ["-C", str(ws), "commit", "-q", "--allow-empty", "-m", "a"],
                 ["-C", str(ws), "remote", "set-url", "origin", "https://github.com/compute-verification/verity.git"]):
        subprocess.run(["/usr/bin/git", *args], check=True, capture_output=True, env={**os.environ, **GIT_ENV})
    # set here, since a module's fixture is built before conftest's per-test ones: the hook's record and tmux socket directory
    env = {**os.environ, **GIT_ENV, "AGENT_GUARD_STATE": str(state), "AGENT_GUARD_MIN_AVAIL": "0", "AGENT_INBOX_POLL_S": "off",
           "AGENT_INBOX": str(tmp / "inbox"), "AGENT_GUARD_EVENT_LOG": str(tmp / "events.jsonl"), "TMUX_TMPDIR": str(tmp)}
    env.pop("TMUX", None)

    def run(command: str, cwd: Path = ws) -> dict:
        event = {"hook_event_name": "beforeShellExecution", "command": command, "conversation_id": ME, "cwd": str(cwd)}
        out = subprocess.run([sys.executable, HOOK], input=json.dumps(event), capture_output=True, text=True, check=True,
                             env=env).stdout
        return json.loads(out)
    return run


def fill(carrier: str, inner: str) -> str:
    return carrier.format(q="'" + inner + "'", w=inner)


@pytest.mark.parametrize("carrier", CARRIERS)
def test_each_rule_holds_through_each_carrier(ask, carrier):
    for rule, (bad, said, good) in RULES.items():
        out = ask(fill(carrier, bad))
        assert out["permission"] == "deny" and said in out["agent_message"], (rule, fill(carrier, bad), out)
        assert ask(fill(carrier, good)) == {"permission": "allow"}, (rule, fill(carrier, good))


def test_the_build_rule_is_about_where_a_build_runs_so_a_scratch_copy_is_no_exception(ask, tmp_path):
    for command in ("tmux send-keys -t w 'cd /tmp/scratch && lake build' C-m", "cd /tmp/scratch && bash -lc 'cargo test'",
                    "tmux new-session -d -c /tmp/scratch -- cargo build"):
        assert ask(command)["permission"] == "deny", command
        assert ask(command, cwd=tmp_path)["permission"] == "deny", command
    assert ask("tmux send-keys -t w 'AGENT_GUARD_LOCAL_OK=1 lake build' C-m") == {"permission": "allow"}, "its opt-in still works"


@pytest.mark.parametrize("command", [
    "tmux run -C 'new-window \"lake build\"'",
    "tmux if-shell true 'new-window \"ps aux\"'",
    "tmux if-shell 'printenv' 'display ok'",
    "tmux display-popup -E 'cargo build'",
    "tmux new-s -d 'lake build'",
    "tmux send-keys -t w ps Space aux Enter",
    "tmux send-keys -t w env Enter",
    "tmux send-keys -t w 'git push origin HEAD:refs/heads/main' Enter",
    "tmux send-keys -t w 'ssh n1 ps aux' Enter",
    "tmux send-keys -t w 'bash -lc \"lake build\"' Enter",
    "bash -o pipefail -c 'lake build'",
    "env -S 'lake build'",
    "echo x | xargs -n 1 -I{} lake build",
    "echo x | xargs -i ps aux",                        # -e, -i and -l take a value only in their own word
    "echo x | xargs -l printenv",
    "echo x | xargs -e lake build",
])
def test_nested_and_spelled_out_forms_are_read_too(ask, command):
    assert ask(command)["permission"] == "deny", command


@pytest.mark.parametrize("command", [
    "tmux display 'lake build'",                       # display-message prints its argument
    "tmux send-keys -X copy-mode",
    "tmux send-keys -t w 'sleep 3000' C-m",           # a pane's command blocks no tool call
    "tmux new-session -d -s s 'sleep 3000'",
    "tmux send-keys -t w 'ssh n1 \"ps aux | grep [s]lot\"' C-m",
    "tmux send-keys -t w 'research run --on auto -- lake build' C-m",
    "tmux kill-session -t s",
    "tmux show-environment -g DISPLAY",
    "tmux source-file ~/.tmux.conf",
])
def test_what_a_typed_command_may_do_tmux_may_do(ask, command):
    assert ask(command) == {"permission": "allow"}, command


def test_a_sleep_past_twenty_minutes_still_blocks_through_a_shell(ask):
    for command in ("bash -lc 'sleep 3000'", "env FOO=1 sleep 3000", "timeout 4000 bash -c 'sleep 3000'"):
        assert "would block" in ask(command)["agent_message"], command


@pytest.fixture(scope="module")
def hook():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "dont_write_bytecode", True)      # no __pycache__ in tools/agent-guard, which boot.sh copies file by file
        for name in ("settings", "shell_hook"):
            spec = importlib.util.spec_from_file_location(f"{name}_carriers", GUARD / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            mp.setitem(sys.modules, name if name == "settings" else spec.name, module)
            spec.loader.exec_module(module)
        yield module


@pytest.mark.parametrize("words, texts", [
    (["tmux", "send-keys", "-t", "w", "lake", "Space", "build", "C-m", "ls", "Enter"], ["lake build\nls\n"]),
    (["tmux", "send-keys", "-l", "C-m"], ["C-m"]),
    (["tmux", "-f", "x.conf", "-L", "sock", "new-session", "-d", "-s", "s", "-c", "/w", "--", "a", "b c"], ["cd /w && a 'b c'"]),
    (["tmux", "split-window", "-l", "20", "-t", "s", "top"], ["top"]),
    (["tmux", "run-shell", "-d", "3", "make"], ["make"]),
    (["tmux", "if-shell", "-F", "#{x}", "new-window ls"], ["ls"]),
    (["tmux", "display-message", "x"], []),
    (["send-keys", "-t", "w", "make", "Enter"], ["make\n"]),
    (["bash", "-e", "-xc", "make"], ["make"]),
    (["bash", "script.sh"], []),
    (["env", "-u", "X", "A=1", "make", "-j"], ["A=1 make -j"]),
    (["env"], []),
    (["xargs", "-0", "-n", "1", "rm"], ["rm"]),
    (["xargs", "-i", "env"], ["env"]),
    (["xargs", "-l", "-e", "env"], ["env"]),
    (["xargs", "-i{}", "-E", "END", "echo", "{}"], ["echo '{}'"]),
    (["watch", "-n", "2", "ls", "-l"], ["ls -l"]),
])
def test_carried_reads_each_carrier(hook, words, texts):
    assert hook.carried(words) == texts
