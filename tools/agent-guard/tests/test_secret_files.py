"""The shell hook refuses a command that would print a key, an env or credentials file, or a file of a secrets directory,
wherever the reading command sits in the line, and names what to run instead (`envnames.py` for an env file's names)."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parents[1]
HOOK = GUARD / "shell_hook.py"
INBOX_TOOL = GUARD / "inbox.py"


@pytest.fixture(scope="module")
def hook():
    """`shell_hook.py` as a module: the rule is a function of the command's text, so the tables ask it in this process."""
    def load(name):
        spec = importlib.util.spec_from_file_location(f"{name}_under_test", GUARD / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "dont_write_bytecode", True)      # no __pycache__ in tools/agent-guard, which boot.sh copies file by file
        mp.setitem(sys.modules, "settings", load("settings"))
        yield load("shell_hook")


REFUSED = [
    # each reading command, on each kind of file
    "cat key.pem",
    "head -3 ~/.ssh/id_rsa",
    "tail -n 2 /home/ubuntu/.ssh/id_ed25519",
    "tail -f prod.env",
    "less id_ecdsa",
    "more server.key",
    "tac prod.env",
    "nl .env",
    "sed 's/=.*//' /root/.config/verity/r2.env",          # the filter that prints a key file's every line
    "sed -n 1,5p x.env",
    "sed -e 's/a/b/' x.env",
    "awk -F= '{print $1}' /etc/vy/env/root.env",
    "awk -f names.awk x.env",
    "cut -d= -f1 relay.env",
    "xxd key.pem | head",
    "od -c x.key",
    "strings x.pem",
    "base64 -w0 key.pem",
    "grep TOKEN /root/.config/verity/relay.env",
    "grep -E 'KEY|TOKEN' x.env",
    "grep -e TOKEN -e KEY x.env",
    "grep -f patterns.txt x.env",
    "egrep -i secret credentials",
    "grep aws ~/.aws/credentials.json",
    "rg -n . ~/.netrc",
    "rg -L KEY x.env",                                    # rg's -L follows symlinks; only grep's lists files
    "cat a.txt x.env",
    "cat 'my key.pem'",
    "cat x.env > /tmp/copy",
    # a secrets directory, and a glob that matches the key beside the env files
    "cat /root/.config/verity/*",
    "cat ~/.config/verity/*.env | cut -d= -f1",
    "sed 's/=.*//' /data/state/secrets/*",
    "cat $HOME/.ssh/*",
    "head -1 /home/research/.ssh/vy-cluster",
    "grep -r TOKEN ~/.config/verity",
    "cat /srv/keys/*.pem",
    "cat deploy/id_rsa*",
    # as the command's input, through a loop's variable, as find's -exec
    "cut -d= -f1 < /root/.config/verity/r2.env",
    "for f in /root/.config/verity/*; do echo \"== $f\"; sed 's/=.*//' \"$f\"; done",
    "for f in $(ls /data/state/secrets/*); do cut -d= -f1 ${f}; done",
    "find /root/.config/verity -type f -exec cat {} +",
    "find / -name '*.pem' -exec head -2 {} \\;",
    # behind wrappers, in subshells and substitutions
    "sudo -u research cat /home/research/.ssh/id_ed25519",
    "timeout 5 cat x.key",
    "cd /root && cat .config/verity/r2.env",
    "bash -c 'cat x.pem'",
    "echo \"$(cat x.env)\"",
    "echo `cat x.pem`",
    "KEY=$(cat x.pem) ./run",
    # over ssh
    "ssh n1 cat /root/.config/verity/r2.env",
    "ssh -o BatchMode=yes n1 'cat /root/.config/verity/*.env | sed \"s/=.*//\"'",
    "ssh n1 \"sudo head -1 /etc/vy/env/root.env\"",
    "ssh n1 'for f in /data/state/secrets/*; do cut -d= -f1 $f; done'",
    "ssh n1 cat < ~/.ssh/id_rsa",
    "research pods ssh vy-control-verity -- 'cat /data/state/secrets/*'",
    "research pods ssh vy-control-verity -- cat /root/.config/verity/panels.key",
    "research run --on auto -- bash -lc 'cat ~/.ssh/id_rsa'",
    # inside a quoted string handed to a wrapper script
    "./on_node.sh n1 'cat /root/.config/verity/*.env'",
    "bash infra/control/as_root.sh \"sed 's/=.*//' /data/state/secrets/msg.env\"",
    "tmux send-keys -t s:0.0 'cat x.env' C-m",
    "docker exec c cat /run/secrets/db.key",
    "./w.sh \"don't cat x.env\"",                         # a string the shell parser can't read is judged on its words
    "echo 'unbalanced; cat x.pem",
    # the local-work opt-in is not this rule's
    "AGENT_GUARD_LOCAL_OK=1 cat x.pem",
    "AGENT_GUARD_LOCAL_OK=1 ssh n1 'cat /root/.config/verity/*'",
]

ALLOWED = [
    # commands that show no contents
    "ls -la ~/.ssh /root/.config/verity",
    "stat -c '%a %U %n' /root/.config/verity/*",
    "wc -l /root/.config/verity/r2.env",
    "sha256sum ~/.ssh/id_rsa key.pem",
    "test -f x.env && echo there",
    "[ -r /data/state/secrets/verity-drive.pem ] && echo readable",
    "chmod 600 ~/.ssh/id_ed25519 x.pem",
    "cp a.env b.env",
    "mv old.key new.key",
    "rm -f /tmp/throwaway.pem",
    # writing, appending, editing in place, sourcing
    "echo FOO=1 > x.env",
    "echo FOO=1 >> /root/.config/verity/extra.env",
    "cat > x.env <<'EOF'\nFOO=1\nEOF",
    "cat defaults.txt >> x.env",
    "printf '%s\\n' \"$KEY\" > key.pem",
    "base64 -d <<< \"$KEY_B64\" > key.pem",
    "sed -i 's/^FOO=.*/FOO=2/' /etc/vy/env/root.env",
    "ssh n1 'umask 077; cat > ~/.ssh/vy-cluster' < ~/.ssh/vy-cluster",
    ". x.env",
    "source ~/.config/verity/r2.env",
    "set -a; . /root/.config/verity/relay.env; set +a",
    # a key's public half and its fingerprint
    "openssl pkey -in key.pem -pubout",
    "openssl rsa -in key.pem -pubout -outform DER | sha256sum",
    "openssl pkey -in /data/state/secrets/verity-drive.pem -pubout | head -3",
    "ssh-keygen -l -f ~/.ssh/id_ed25519",
    "ssh-keygen -lf key.pem",
    "ssh-keygen -y -f ~/.ssh/id_rsa > ~/.ssh/id_rsa.pub",
    "cat ~/.ssh/id_ed25519.pub",
    "cat key-pub.pem",
    "head -2 ~/.ssh/authorized_keys",
    "grep github ~/.ssh/known_hosts",
    "cat ~/.ssh/config",
    "grep -qF \"$(cat ~/.ssh/vy-pool.pub)\" ~/.ssh/authorized_keys",
    "cat ~/.ssh/vy-pool.pub >> ~/.ssh/authorized_keys",
    "for f in ~/.ssh/*.pub; do cat \"$f\"; done",
    # examples, templates and a tests/ directory's fixtures
    "cat .env.example",
    "sed -n 1,20p deploy/prod.env.template",
    "cat tools/research/tests/fixtures/dummy.pem",
    "head -3 tests/data/sample.env",
    "grep -n KEY /agent/repos/verity/tools/x/tests/creds/credentials.json",
    # a module or a doc named for what it handles
    "sed -n 1,40p tools/research/src/research/store/credentials.py",
    "rg -n mint tools/research/src/research/store/credentials.py docs/credentials.md",
    # the names-only reader, however it is run
    "python3 tools/agent-guard/envnames.py /root/.config/verity/*",
    "ssh n1 python3 - '/root/.config/verity/*' '/data/state/secrets/*' < tools/agent-guard/envnames.py",
    "research pods ssh vy-control-verity -- python3 - /root/.config/verity/r2.env < tools/agent-guard/envnames.py",
    "python3 tools/agent-guard/envnames.py x.env | head -5",
    "for f in /root/.config/verity/*; do python3 tools/agent-guard/envnames.py \"$f\"; done",
    "sed -n 1,40p tools/agent-guard/envnames.py",
    # a grep that lists files, counts or only tests
    "grep -l TOKEN /root/.config/verity/*",
    "grep -c . x.env",
    "grep -q SLACK_BOT_TOKEN /root/.config/verity/relay.env && echo set",
    "grep -L TOKEN *.env",
    "grep -rl KEY ~/.config/verity",
    "rg -l TOKEN ~/.config/verity",
    "rg -c . x.env",
    "rg --files-with-matches KEY x.env",
    "ssh n1 'grep -c \"\" /etc/vy/env/*.env'",
    "find /root/.config/verity -name '*.env' -exec grep -l X {} +",
    # a key or env file that is used, not read
    "ssh -i ~/.ssh/research_key n1 'tail -5 /var/log/syslog'",
    "ssh -i key.pem ubuntu@host cat /etc/hostname",
    "scp -i ~/.ssh/id_ed25519 n1:/tmp/x .",
    "uv run --env-file .env pytest tests/test_x.py",
    "docker run --env-file prod.env img cat /etc/hosts",
    "for f in /root/.config/verity/*; do stat -c %n \"$f\"; done",
    # a pattern or prose that names such a file is not a file
    "rg -n '\\.pem|\\.env' infra/",
    "grep -rn id_rsa tools/research/src",
    "grep -e '\\.env' README.md",
    "rg -n 'cat .*\\.pem' tools/",
    "git commit -m 'hook: refuse cat x.pem and head ~/.ssh/id_rsa'",
    "git grep -n cat -- '*.env'",
    "research msg linear comment INF-1 'the hook refuses `cat x.pem`' --worker W",
    "research queue open --head B --sha S --title 'refuse cat of *.pem' --body-file F --issue INF-1 --by infra",
    # reading anything else
    "cat README.md | head",
    "tail -f /var/log/x.log",
    "cut -d= -f1 names.txt",
    "find . -name '*.py' -exec head -1 {} \\;",
    "cat $KEYFILE",
    "systemctl cat vy-pool.service",
    "compgen -e | rg NEBIUS",
    "",
]


@pytest.mark.parametrize("command", REFUSED)
def test_a_command_that_would_print_a_secret_file_is_refused_with_the_fix(hook, command):
    reason = hook.secret_file_reason(command)
    assert reason is not None, command
    assert "python3 tools/agent-guard/envnames.py FILE..." in reason
    assert "ssh HOST python3 - FILE... < tools/agent-guard/envnames.py" in reason
    assert "`stat`, `sha256sum` or `openssl pkey -in FILE -pubout`" in reason
    assert "AGENT_GUARD_LOCAL_OK=1 does not lift this" in reason


@pytest.mark.parametrize("command", ALLOWED)
def test_a_command_that_shows_no_secret_files_contents_passes(hook, command):
    assert hook.secret_file_reason(command) is None, command


@pytest.mark.parametrize("command, reader, path, kind", [
    ("ssh n1 'head -1 /data/state/secrets/verity-drive.pem'", "head", "/data/state/secrets/verity-drive.pem", "a private key"),
    ("cut -d= -f1 /root/.config/verity/relay.env", "cut", "/root/.config/verity/relay.env", "an env or credentials file"),
    ("for f in /root/.config/verity/*; do sed 's/=.*//' $f; done", "sed", "/root/.config/verity/*", "a file of a secrets directory"),
])
def test_the_refusal_says_what_it_matched(hook, command, reader, path, kind):
    assert f"this command would print {kind}: `{reader}` reads `{path}`." in hook.secret_file_reason(command)


def test_a_bug_in_the_parse_still_refuses_what_the_text_shows(hook, monkeypatch):
    def broken(text):
        raise IndexError(text)

    monkeypatch.setattr(hook, "secret_read", broken)
    assert "`cat` reads `/root/.config/verity/r2.env`" in hook.secret_file_reason("ssh n1 'cat /root/.config/verity/r2.env'")
    assert hook.secret_file_reason("ls -la ~/.ssh") is None
    assert hook.secret_file_reason("cat README.md") is None


def ask(command: str, **env: str) -> dict:
    event = {"hook_event_name": "beforeShellExecution", "command": command, "conversation_id": "bc-11111111-1111-4111-8111-111111111111"}
    out = subprocess.run([sys.executable, HOOK], input=json.dumps(event), capture_output=True, text=True, check=True,
                         env={**os.environ, "AGENT_INBOX_POLL_S": "off", "AGENT_GUARD_MIN_AVAIL": "0", **env}).stdout
    return json.loads(out)


def test_the_hook_denies_it_keeps_a_queued_message_and_its_reason_comes_before_a_gates(tmp_path):
    inbox = {"AGENT_INBOX": str(tmp_path)}
    subprocess.run([sys.executable, INBOX_TOOL, "send", "bc-11111111-1111-4111-8111-111111111111", "--local", "--text", "hello"],
                   env={**os.environ, **inbox}, check=True, capture_output=True)
    out = ask("ssh n1 'cat /root/.config/verity/*.env | cut -d= -f1'", **inbox)
    assert out["permission"] == "deny" and "hello" not in out["agent_message"]
    assert out["agent_message"].startswith("Refused by tools/agent-guard/shell_hook.py (.cursor/hooks.json): this command would "
                                           "print an env or credentials file: `cat` reads `/root/.config/verity/*.env`.")
    assert "hello" in ask("python3 tools/agent-guard/envnames.py /root/.config/verity/*", **inbox)["agent_message"]
    assert ask("python3 tools/agent-guard/envnames.py /root/.config/verity/*", **inbox) == {"permission": "allow"}
    blocked = ask("AGENT_GUARD_LOCAL_OK=1 ssh n1 'cat ~/.ssh/id_rsa; sleep 3000'")
    assert blocked["permission"] == "deny" and "would print a private key" in blocked["agent_message"]
