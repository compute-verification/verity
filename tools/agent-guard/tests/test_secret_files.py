"""The shell hook refuses a command that would print a key, an env or credentials file, or a file of a secrets directory,
wherever the reading command sits in the line, and names what to run instead (`envnames.py` for an env file's names)."""
import importlib.util
import json
import os
import shlex
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
    # a grep's option that takes a value is neither its pattern nor a file
    "grep -m 1 TOKEN x.env",
    "grep -A 2 -e KEY x.env",
    "grep -e KEY -e TOKEN x.env y.env",
    "grep -rn --include '*.py' KEY secrets/",
    "rg -t py KEY x.env",
    "rg --type=py -n KEY /root/.config/verity/r2.env",
    "rg -g '*.env' -e TOKEN ~/.ssh/id_rsa",
    "rg -C3 KEY x.env",
    "rg -m1 -- -KEY x.env",
    # the other commands that write out what they read
    "sort x.env",
    "sort -u /root/.config/verity/*.env",
    "uniq x.env",
    "paste -sd, x.env",
    "column -t -s= x.env",
    "fold -w 64 key.pem",
    "rev key.pem",
    "hexdump -C key.pem",
    "diff a.env b.env",
    "diff -u /etc/vy/env/research.env ~/.config/verity/research.env",
    "comm a.env b.env",
    "cmp -l a.key b.key",
    "git diff --no-index /dev/null ~/.ssh/id_rsa",
    "tr -d '\\n' < key.pem",
    "tee < x.env",
    "tee /dev/stderr < x.env",
    "dd if=key.pem bs=64 count=1",
    "dd if=/root/.ssh/id_ed25519 2>/dev/null",
    "dd < key.pem",
    # openssl writes out the private key it reads
    "openssl pkey -in key.pem",
    "openssl rsa -in ~/.ssh/id_rsa -text -noout",
    "openssl ec -in ec.key -outform DER | base64 -w0",
    "openssl rsa -in mykey",
    "openssl pkey < key.pem",
    "openssl pkcs8 -topk8 -nocrypt -in key.pem",
    "openssl dsa -in dsa.pem",
    "sudo openssl rsa -in /etc/ssl/private/server.key",
    "ssh n1 'openssl pkey -in /root/.config/verity/nebius-wake-key.pem -text'",
    # behind xargs, what the commands piped into it name
    "ls /root/.config/verity/* | xargs cat",
    "find ~/.ssh -type f | xargs head -1",
    "find / -name '*.pem' | xargs -n1 cat",
    "find /etc/vy/env -name '*.env' -print0 | xargs -0 cut -d= -f1",
    "printf '%s\\n' a.env b.env | xargs sed 's/=.*//'",
    "ls ~/.ssh | grep -v pub | xargs -I{} cat ~/.ssh/{}",
    "grep -rl KEY /root/.config/verity | xargs cat",
    # the shell reading the file itself
    "while read l; do echo \"$l\"; done < x.env",
    "while IFS= read -r line; do echo \"${line%%=*}\"; done < /root/.config/verity/r2.env",
    "ssh n1 'while read l; do echo ${l%%=*}; done < /etc/vy/env/research.env'",
    "echo \"$(<x.pem)\"",
    "echo $( < x.env )",
    "KEY=$(<~/.ssh/id_rsa)",
    "K=$(< key.pem); echo ok",
    # an interpreter's code that names the file and opens, reads or prints
    "python3 -c \"print(open('/root/.config/verity/r2.env').read())\"",
    "python3 -c \"import os; os.system('cat x.env')\"",
    "python -c 'import sys; print(open(sys.argv[1]).read())' key.pem",
    "python3 -c \"import pathlib; print(pathlib.Path('~/.ssh/id_rsa').expanduser().read_text())\"",
    "python3 -c 'import sys; print(sys.stdin.read())' < x.env",
    "ssh n1 \"python3 -c 'print(open(\\\"/etc/vy/env/r2.env\\\").read())'\"",
    "perl -pe 's/=.*//' x.env",
    "perl -ne 'print if /KEY/' /etc/vy/env/research.env",
    "ruby -e 'puts File.read(\"credentials.json\")'",
    # a copy between machines by way of the transcript
    "research pods ssh vy-nebius-2 -- 'cat ~/.ssh/vy-cluster' | research pods ssh vy-nebius-3 -- 'umask 077; cat > ~/.ssh/vy-cluster'",
    "research pods ssh vy-nebius-1 -- 'sudo cat /etc/verity/lean-cache.env' | research pods ssh vy-nebius-3 -- 'sudo install -D -m 600 "
    "/dev/stdin /etc/verity/lean-cache.env'",
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
    # a grep's option that takes a value is neither its pattern nor a file
    "rg -t py credentials tools/",
    "rg -tpy credentials tools/",
    "rg --type py credentials tools/",
    "rg --type=py credentials tools/",
    "rg -g '*.py' -n credentials .",
    "rg -T md -C 2 id_rsa tools/",
    "rg -e credentials -e id_rsa docs/",
    "grep -rn -m 3 credentials docs/",
    "grep --max-count 2 credentials README.md",
    "grep -A 2 -e credentials README.md",
    "grep -rn --include '*.md' credentials .",
    "grep -q -m 1 KEY x.env",
    "grep -rlZ KEY /root/.config/verity",
    "rg -lm1 KEY x.env",
    "rg --count-matches KEY x.env",
    # a comparison that shows no line, a sort into a file, a copy
    "diff -q a.env b.env",
    "diff --brief a.env b.env",
    "cmp a.key b.key",
    "cmp -s a.key b.key",
    "sort -o x.env x.env",
    "dd if=key.pem of=/tmp/k.bak",
    "echo FOO=1 | sudo tee -a /etc/vy/env/extra.env",
    "echo \"$X\" | tr -d '\\n' > x.env",
    "ssh n1 'sudo tee /etc/verity/lean-cache.env >/dev/null' < /etc/verity/lean-cache.env",
    "research pods ssh vy-nebius-2 -- 'tar -C ~/.ssh -cf - vy-cluster vy-cluster.pub' | research pods ssh vy-nebius-3 -- 'umask 077; tar "
    "-C ~/.ssh -xf -'",
    "research pods ssh vy-nebius-1 -- 'sudo tar -C /etc/verity -cf - lean-cache.env' | research pods ssh vy-nebius-3 -- 'sudo mkdir -p "
    "/etc/verity && sudo tar -C /etc/verity --no-same-owner -xf - && sudo chmod 600 /etc/verity/lean-cache.env'",
    # openssl on a certificate, on a public key, checking a key, or writing a file
    "openssl x509 -in cert.pem -noout -text",
    "openssl x509 -in /etc/ssl/certs/ca.pem -noout -fingerprint -sha256",
    "openssl x509 -in cert.pem -noout -subject -enddate",
    "openssl pkey -in key.pem -noout",
    "openssl rsa -in key.pem -check -noout",
    "openssl pkey -in key.pem -text_pub -noout",
    "openssl pkey -pubin -in key-pub.pem -text",
    "openssl rsa -in key.pem -out key2.pem",
    "openssl genpkey -algorithm ed25519 -out key.pem",
    "openssl dgst -sha256 -sign key.pem -out sig.bin file",
    "openssl req -new -key key.pem -out req.csr",
    # xargs behind commands that name no such file, or before a command that shows no contents
    "find . -name '*.py' | xargs grep -n foo",
    "find ~/.ssh -type f | xargs grep -l BEGIN",
    "rg -l credentials tools/ | xargs sed -n 1p",
    "grep -rl id_rsa docs | xargs head -3",
    "git ls-files '*.env' | xargs cat",
    "ls *.env | xargs wc -l",
    # the shell reading another file, an interpreter that reads none
    "while read f; do echo \"$f\"; done < files.txt",
    "while read l; do echo $l; done < <(ls ~/.ssh)",
    "echo \"$(<README.md)\"",
    "python3 -c 'print(1+1)'",
    "python3 script.py --key key.pem",
    "python3 -m pytest tests/test_secret_files.py -q",
    "perl -pi -e 's/^FOO=.*/FOO=2/' x.env",
    "docker run python cat /etc/hostname",
    "gh pr diff 8496",
    "git diff HEAD~1 -- x.env",
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


def test_an_openssl_refusal_names_the_options_that_keep_the_key_in(hook):
    reason = hook.secret_file_reason("openssl rsa -in deploy.key")
    assert "this command would print a private key: `openssl rsa` reads `deploy.key`." in reason
    assert "add `-pubout` for its public half or `-noout` to only check it" in reason
    assert hook.secret_file_reason("openssl rsa -in deploy.key -noout -text").count("-text_pub") == 1


def test_a_capture_into_a_variable_is_refused_and_told_where_to_read_the_file(hook):
    for command in ("KEY=$(cat x.pem)", "export TOKEN=\"$(cat ~/.config/verity/r2.env)\"", "KEY=$(<x.pem); ./run"):
        assert "reads the file inside the program that needs it" in hook.secret_file_reason(command), command


@pytest.mark.parametrize("command", [
    "for f in /root/.config/verity/*; do echo \"== $f\"; sed 's/=.*//' \"$f\"; done",
    "for f in /root/.config/verity/*; do echo \"== $f\"; cut -d= -f1 \"$f\"; done",
    "for f in /root/.config/verity/*; do echo \"== $f\"; grep -o '^[A-Z_]*' \"$f\"; done",
    "for f in /root/.config/verity/*; do awk -F= '{print FILENAME \": \" $1}' \"$f\"; done",
    "cut -d= -f1 /root/.config/verity/*",
    "sed 's/=.*//' /root/.config/verity/*",
    "grep -o '^[A-Z_]*' /root/.config/verity/*",
    "cat /root/.config/verity/* | cut -d= -f1 | sort -u",
    "sudo sh -c \"for f in /root/.config/verity/*; do cut -d= -f1 \\$f; done\"",
])
def test_the_names_only_idiom_over_a_secrets_directory_is_refused_on_the_control_plane(hook, command):
    """The incident: a loop or a glob over the control plane's secrets directory, each line cut at its `=` to list variable
    names. The glob matches the key file beside the env files, and no line of a key holds an `=`."""
    for line in (f"research pods ssh vy-control-verity -- {shlex.quote(command)}", f"ssh root@control {shlex.quote(command)}", command):
        reason = hook.secret_file_reason(line)
        assert reason and "would print a file of a secrets directory" in reason and "`/root/.config/verity/*`" in reason, line
    assert hook.secret_file_reason("research pods ssh vy-control-verity -- python3 - '/root/.config/verity/*' < tools/agent-guard/"
                                   "envnames.py") is None


GIT_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A checkout with settings files named like secrets, committed, and beside them a changed one, an untracked one and a
    link."""
    for name, value in GIT_ENV.items():
        monkeypatch.setenv(name, value)
    for name in ("conf/env/a.env", "conf/env/b.env", "conf/settings.key", "changed.env", "sub/c.env"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("NAME=value\n")
    (tmp_path / "link.env").symlink_to(tmp_path / "outside.txt")
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "settings"]):
        subprocess.run(["git", "-C", tmp_path, *args], check=True, capture_output=True)
    (tmp_path / "outside.txt").write_text("TOKEN=dummy\n")
    (tmp_path / "changed.env").write_text("TOKEN=dummy\n")
    (tmp_path / "untracked.env").write_text("TOKEN=dummy\n")
    (tmp_path / "conf/env/new.key").write_text("dummy\n")
    return str(tmp_path)


TRACKED = [
    "cat conf/env/a.env",
    "cat conf/env/a.env conf/env/b.env",
    "cat {here}/conf/env/a.env",
    "cat conf/env/*.env",
    "head -3 conf/settings.key",
    "grep -n NAME conf/env/a.env | head -3",
    "sort conf/env/a.env | uniq -c",
    "diff conf/env/a.env conf/env/b.env",
    "tr a-z A-Z < conf/env/a.env",
    "openssl pkey -in conf/settings.key",
    "sudo cat conf/env/a.env",
    "timeout 5 cat conf/env/a.env",
    "cd conf && cat env/a.env",
    "cd conf/env; grep -n NAME a.env b.env",
    "cat sub/c.env && cd sub && cat c.env",
    "bash -c 'cat conf/env/a.env'",
    "bash -lc 'cd conf && cut -d= -f1 env/*.env'",
    "for f in conf/env/*.env; do echo \"== $f\"; cut -d= -f1 \"$f\"; done",
    "while read l; do echo \"$l\"; done < conf/env/a.env",
    "python3 -c \"print(open('conf/env/a.env').read())\"",
]

NOT_TRACKED = [
    # not the committed bytes
    "cat changed.env",
    "cat untracked.env",
    "cat link.env",
    "cat missing.env",
    "cat conf/env/a.env untracked.env",
    "cat *.env",
    "cat conf/*.key conf/env/*.key",
    "openssl pkey -in conf/env/new.key",
    "for f in conf/env/a.env untracked.env; do cat $f; done",
    "cat conf/env/a.env; cd sub; cat ../changed.env",
    # not on this machine, or not known to be
    "ssh n1 cat conf/env/a.env",
    "ssh n1 'cat conf/env/a.env'",
    "research pods ssh vy-nebius-1 -- cat conf/env/a.env",
    "tools/infra/on-node.sh 'cat conf/env/a.env'",
    "docker run -v $PWD:/w img cat /w/conf/env/a.env",
    "ls conf/env/*.env | xargs cat",
    "find conf -name '*.env' -exec cat {} +",
    # not in this directory, or not known to be
    "cat ~/conf/env/a.env",
    "cd /etc/vy && cat conf/env/a.env",
    "cd sub && cat conf/env/a.env",
    "cd $HOME && cat conf/env/a.env",
    "cd conf | cat conf/env/a.env",
    "(cd sub); cat c.env",
    "pushd conf; cat env/a.env",
    "sudo -i cat conf/env/a.env",
    "sudo -i bash -c 'cat conf/env/a.env'",
]


@pytest.mark.parametrize("command", TRACKED)
def test_a_file_git_has_committed_is_read_freely_in_its_checkout(hook, checkout, command):
    command = command.format(here=checkout)
    assert hook.secret_file_reason(command, checkout) is None, command
    assert hook.secret_file_reason(command) is not None, command                      # the directory unknown: refused as before
    assert hook.secret_file_reason(command, str(Path(checkout).parent)) is not None or checkout in command, command


@pytest.mark.parametrize("command", NOT_TRACKED)
def test_a_changed_untracked_linked_or_remote_file_of_that_name_is_still_refused(hook, checkout, command):
    assert hook.secret_file_reason(command, checkout) is not None, command


def test_without_git_no_file_is_taken_for_committed(hook, checkout, monkeypatch):
    monkeypatch.setenv("PATH", "")
    assert hook.secret_file_reason("cat conf/env/a.env", checkout) is not None


def test_a_bug_in_the_parse_still_refuses_what_the_text_shows(hook, monkeypatch):
    def broken(text, where=None):
        raise IndexError(text)

    monkeypatch.setattr(hook, "secret_read", broken)
    assert "`cat` reads `/root/.config/verity/r2.env`" in hook.secret_file_reason("ssh n1 'cat /root/.config/verity/r2.env'")
    assert hook.secret_file_reason("ls -la ~/.ssh") is None
    assert hook.secret_file_reason("cat README.md") is None


def ask(command: str, cwd: str = "", **env: str) -> dict:
    event = {"hook_event_name": "beforeShellExecution", "command": command, "conversation_id": "bc-11111111-1111-4111-8111-111111111111",
             **({"cwd": cwd} if cwd else {})}
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


def test_the_hook_reads_the_commands_directory_from_the_event_and_guesses_none(checkout):
    assert ask("cat conf/env/a.env", cwd=checkout) == {"permission": "allow"}
    assert ask("cd conf && cut -d= -f1 env/*.env", cwd=checkout) == {"permission": "allow"}
    for command, cwd in (("cat changed.env", checkout), ("ssh n1 cat conf/env/a.env", checkout), ("cat conf/env/a.env", "")):
        out = ask(command, cwd=cwd)
        assert out["permission"] == "deny" and "would print an env or credentials file" in out["agent_message"], command
    stated_nowhere = subprocess.run([sys.executable, HOOK], input=json.dumps({"command": "cat conf/env/a.env", "workspace_roots": [checkout]}),
                                    capture_output=True, text=True, check=True, cwd=checkout,
                                    env={**os.environ, "AGENT_INBOX_POLL_S": "off", "AGENT_GUARD_MIN_AVAIL": "0"}).stdout
    assert json.loads(stated_nowhere)["permission"] == "deny"
