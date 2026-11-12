"""A VM keeps one tmux session, `vm-secrets`, so that its tmux server, whose environment holds the secrets it started with,
never ends (`kept_session.py`). The shell hook refuses `tmux kill-server` and a `tmux kill-session` that would end that session
or the server's last one, and lets every other kill through; `kept_session.py start`, which boot.sh runs, starts the session
only where the server it starts would hold the secrets.

The hook's tests give it a stand-in `tmux` that lists the sessions a test names; the one test of a real tmux runs it on a
socket directory of its own. Neither asks the tmux server of the machine the tests run on."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parents[1]
HOOK, KEPT_SESSION = GUARD / "shell_hook.py", GUARD / "kept_session.py"
KEPT = "vm-secrets"
STAND_IN = """#!/bin/sh
echo "$*" >> "$0.calls"
[ "$1" = list-sessions ] || { echo "the hook asks tmux only for its sessions: $*" >&2; exit 64; }
[ -f "$0.broken" ] && { echo "lost server" >&2; exit 1; }
[ -f "$0.sessions" ] || { echo "no server running on /nowhere/default" >&2; exit 1; }
cat "$0.sessions"
"""

#: with `vm-secrets`, `build` and `mine` there: what ends the server
KILL_SERVER = [
    "tmux kill-server",
    "tmux kill-ser",                                            # tmux takes a command's prefix
    "tmux -f /exec-daemon/tmux.portal.conf kill-server",
    "tmux -2u kill-server",
    "tmux -L default kill-server",                              # the default socket, named
    "tmux -S /tmp/tmux-1000/default kill-server",
    "cd /tmp && tmux kill-server",
    "tmux kill-server; tmux new-session -d -s fresh",
    "tmux new-session -d -s x \\; kill-server",
    "tmux list-sessions ';' kill-server",
    "tmux if-shell true 'kill-server'",
    "tmux run-shell -C 'kill-server'",
    "bash -lc 'tmux kill-server'",
    "timeout 5 tmux kill-server || true",
    "tmux send-keys -t build 'tmux kill-server' Enter",
    "tmux -L mine send-keys -t x 'tmux kill-server' Enter",     # typed into a pane: the hook can't see which server its tmux is
    "tmux kill-server 'an unbalanced quote",
]
#: what ends the kept session, by any way tmux reads a session's name
KILL_KEPT = [
    f"tmux kill-session -t {KEPT}",
    f"tmux kill-ses -t {KEPT}",
    f"tmux kill-session -t={KEPT}",
    f"tmux kill-session -t ={KEPT}",
    f"tmux kill-session -t {KEPT}:",
    f"tmux kill-session -t {KEPT}:0.0",
    "tmux kill-session -t vm",                                  # the start of its name
    "tmux kill-session -t 'vm-*'",                              # a pattern
    f"tmux -f /exec-daemon/tmux.portal.conf kill-session -t {KEPT}",
    f"tmux kill-session -t build \\; kill-session -t {KEPT}",
    f"tmux kill-session -t build; tmux kill-session -t {KEPT}",
    f"bash -c 'tmux kill-session -t {KEPT}'",
    f"tmux send-keys -t build 'tmux kill-session -t {KEPT}' C-m",
    f"tmux if-shell true 'kill-session -t {KEPT}'",
]
#: what doesn't say, in words the hook can read, which session it ends or spares
UNNAMED = [
    "tmux kill-session",
    "tmux kill-session -a",
    'tmux kill-session -t "$SESSION"',
    "tmux kill-session -t $(tmux display-message -p '#S')",
    'tmux kill-session -a -t "$SESSION"',
    "tmux kill-session -t '$0'",                                # a session's id: the hook reads no `$`
    "tmux kill-session -t build.0",                             # a window and a pane, in a session tmux picks
    "tmux kill-session -t :1",
]
#: what ends every session but one that is not the kept one
ALL_BUT_ONE = ["tmux kill-session -a -t build", "tmux kill-session -at build", "tmux kill-session -a -t =mine"]
#: what ends neither the kept session nor the server
ALLOWED = [
    "tmux kill-session -t build",
    "tmux kill-ses -t build",
    "tmux kill-session -t =mine",
    "tmux kill-session -t bui",                                 # the start of another's name
    "tmux kill-session -t build:0.0",
    "tmux kill-session -t build \\; kill-session -t mine",       # the kept session is left
    "tmux kill-session -t build; tmux kill-session -t mine",
    "tmux -f /exec-daemon/tmux.portal.conf kill-session -t build",
    "tmux send-keys -t build 'tmux kill-session -t mine' Enter",
    "tmux kill-session -t nosuch",                              # tmux finds no such session and ends nothing
    "tmux kill-session -t 'm*' ; tmux kill-session -t '*i*'",   # one name each, neither the kept one's (`build`, `mine`)
    f"tmux kill-session -C -t {KEPT}",                          # -C clears alerts and ends nothing
    f"tmux kill-session -a -t {KEPT}",                          # every session but the kept one
    f"tmux has-session -t {KEPT} || tmux new-session -d -s {KEPT}",
    "tmux list-sessions",
    "tmux kill-window -t build:1",
    "echo tmux kill-server is refused",
    "git commit -m 'the hook refuses tmux kill-server'",
]
#: a server of the command's own, whatever it ends there
ANOTHER_SERVER = [
    "tmux -L mine kill-server",
    "tmux -Lmine kill-server",
    "tmux -S /tmp/pytest/sock kill-server",
    'tmux -S "$TMP/sock" kill-server',
    "tmux -L mine new-session -d -s x \\; kill-server",
    f"tmux -L mine kill-session -t {KEPT}",
    "tmux -S /tmp/pytest/sock kill-session",
    "tmux -L mine kill-session -a",
]


@pytest.fixture
def vm(tmp_path):
    """The shell hook's verdict on a command, on a VM whose tmux lists the sessions `vm.sessions(...)` names (none: no
    server runs; `vm.broken()`: tmux gives no answer), and each question the hook put to tmux (`vm.asked()`)."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    tmux = bin_ / "tmux"
    tmux.write_text(STAND_IN)
    tmux.chmod(0o755)
    env = {"PATH": f"{bin_}{os.pathsep}{os.defpath}", "HOME": str(tmp_path), "TMUX_TMPDIR": str(tmp_path),
           "AGENT_GUARD_EVENT_LOG": str(tmp_path / "events.jsonl"), "AGENT_INBOX": str(tmp_path / "inbox"),
           "AGENT_INBOX_POLL_S": "off", "AGENT_GUARD_STATE": str(tmp_path / "state"), "AGENT_GUARD_MIN_AVAIL": "0"}

    class VM:
        def sessions(self, *names):
            Path(f"{tmux}.sessions").write_text("".join(f"${i} {n}\n" for i, n in enumerate(names)))
            return self

        def broken(self):
            Path(f"{tmux}.broken").write_text("")
            return self

        def asked(self):
            calls = Path(f"{tmux}.calls")
            return calls.read_text().splitlines() if calls.exists() else []

        def __call__(self, command):
            event = {"hook_event_name": "beforeShellExecution", "command": command, "conversation_id": "bc-me", "cwd": str(tmp_path)}
            out = subprocess.run([sys.executable, HOOK], input=json.dumps(event), capture_output=True, text=True, check=True,
                                 env=env).stdout
            return json.loads(out)

        def refused(self, command, *said):
            out = self(command)
            assert out["permission"] == "deny", command
            for text in said:
                assert text in out["agent_message"], (command, text, out["agent_message"])
            return out["agent_message"]

        def allowed(self, command):
            assert self(command) == {"permission": "allow"}, command
    return VM()


@pytest.fixture(scope="module")
def kept():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "dont_write_bytecode", True)      # no __pycache__ in tools/agent-guard, which boot.sh copies file by file
        for name in ("settings", "kept_session"):
            spec = importlib.util.spec_from_file_location(name if name == "settings" else f"{name}_tests", GUARD / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            mp.setitem(sys.modules, spec.name, module)
            spec.loader.exec_module(module)
        yield module


@pytest.mark.parametrize("command", KILL_SERVER)
def test_ending_the_vms_tmux_server_is_refused_with_what_to_do_instead(vm, command):
    vm.sessions(KEPT, "build", "mine").refused(command, "`tmux kill-server` would end this VM's tmux server and every session on it",
                                               "holds in its environment the secrets it started with",
                                               "Close your own sessions by name instead: `tmux kill-session -t NAME`",
                                               "`tmux -L NAME kill-server`")
    assert vm.asked() == [], "ending the server is refused whatever sessions there are"


def test_ending_the_server_is_refused_where_no_server_runs_and_where_tmux_gives_no_answer(vm):
    vm.refused("tmux kill-server", "would end this VM's tmux server")
    vm.broken().refused("tmux kill-server", "would end this VM's tmux server")


@pytest.mark.parametrize("command", KILL_KEPT)
def test_ending_the_kept_session_is_refused_however_the_command_names_it(vm, command):
    said = vm.sessions(KEPT, "build", "mine").refused(command, f"`{KEPT}`, the session this VM keeps so that its tmux server never ends",
                                                      "holds in its environment the secrets it started with",
                                                      "Close your own sessions by name instead: `tmux kill-session -t NAME`")
    assert "would end" in said or "doesn't say which session it ends" in said, said


@pytest.mark.parametrize("command", UNNAMED)
def test_a_kill_that_names_no_session_the_hook_can_read_is_refused(vm, command):
    vm.sessions(KEPT, "build", "mine").refused(command, "doesn't say which", f"`{KEPT}`, the session this VM keeps",
                                               "Close your own sessions by name instead")


@pytest.mark.parametrize("command", ALL_BUT_ONE)
def test_a_kill_of_every_session_but_one_is_refused_while_the_kept_session_is_among_them(vm, command):
    vm.sessions(KEPT, "build", "mine").refused(command, "would end every session but `", f"`{KEPT}`, the session this VM keeps",
                                               "and other agents'", "Close your own sessions by name instead")
    vm.sessions("build", "mine").allowed(command)       # no kept session there: the one spared keeps the server


@pytest.mark.parametrize("command", ALLOWED)
def test_a_kill_of_a_session_that_is_neither_the_kept_one_nor_the_last_passes(vm, command):
    vm.sessions(KEPT, "build", "mine").allowed(command)


@pytest.mark.parametrize("command", ANOTHER_SERVER)
def test_a_command_on_a_tmux_server_of_its_own_passes_and_asks_nothing_of_the_vms(vm, command):
    vm.sessions(KEPT).allowed(command)
    assert vm.asked() == []


def test_a_kill_of_the_servers_last_session_is_refused_until_the_kept_session_is_there(vm):
    start = f"Start the session this VM keeps first, `tmux new-session -d -s {KEPT}`; this command then passes."
    vm.sessions("build")
    for command in ("tmux kill-session -t build", "tmux kill-session -t bui", "tmux kill-session -t =build", "tmux kill-session",
                    'tmux kill-session -t "$S"'):
        vm.refused(command, "would end `build`, the last session on this VM's tmux server, and the server with it", start)
    vm.allowed("tmux kill-session -a -t build")         # it ends every other session, and there is none
    vm.allowed("tmux kill-session -a")
    vm.sessions("build", "mine")
    vm.allowed("tmux kill-session -t build")
    vm.allowed("tmux kill-session -t mine")
    for command in ("tmux kill-session -t build \\; kill-session -t mine", "tmux kill-session -t build; tmux kill-session -t mine",
                    "tmux kill-session -t build && tmux kill-ses -t =mine"):
        vm.refused(command, "`tmux kill-session -t ", "would end `mine`, the last session on this VM's tmux server", start)
    vm.sessions("build", "mine", KEPT)
    vm.allowed("tmux kill-session -t build \\; kill-session -t mine")
    vm.sessions(KEPT)
    vm.refused("tmux kill-session", f"would end `{KEPT}`, the session this VM keeps", "Leave it there.")
    vm.allowed("tmux kill-session -a")


def test_where_no_server_runs_there_is_nothing_to_end(vm):
    for command in (f"tmux kill-session -t {KEPT}", "tmux kill-session", "tmux kill-session -a", "tmux kill-session -a -t build"):
        vm.allowed(command)


def test_where_tmux_gives_no_answer_the_command_s_text_decides(vm):
    vm.sessions(KEPT, "build").broken()
    cant = "the hook can't ask it which are there (`tmux list-sessions` exits 1)"
    for command in (f"tmux kill-session -t {KEPT}", f"tmux kill-session -t ={KEPT}", "tmux kill-session -t vm", "tmux kill-session -t 'vm-*'",
                    "tmux kill-session -t '*'"):
        vm.refused(command, f"may end `{KEPT}`, the session this VM keeps", cant, "Leave it there.")
    vm.refused("tmux kill-session", "doesn't say which session it ends")
    vm.refused("tmux kill-session -a -t build", "would end every session but one, and the hook can't tell that",
               "(`tmux list-sessions` exits 1)")
    for command in ("tmux kill-session -t build", "tmux kill-session -t =vm", f"tmux kill-session -a -t {KEPT}",
                    f"tmux kill-session -a -t ={KEPT}", "tmux kill-session -t 'an unbalanced quote"):
        vm.allowed(command)


def test_the_hook_asks_tmux_once_a_command_and_only_for_a_kill_session_on_the_vms_server(vm):
    vm.sessions(KEPT, "a", "b", "c")
    vm.allowed("tmux kill-session -t a \\; kill-session -t b; tmux kill-session -t c")
    assert vm.asked() == ["list-sessions -F #{session_id} #{session_name}"]
    for command in ("tmux new-session -d -s x", "tmux send-keys -t a ls Enter", "tmux kill-window -t a:1", "ls", "echo kill-session"):
        vm.allowed(command)
    assert len(vm.asked()) == 1


@pytest.mark.parametrize("target, sessions, hit", [
    ("build", ["build", "builder"], "build"),                   # a whole name before the start of another
    ("bui", ["build", "mine"], "build"),                        # the start of one name
    ("b", ["build", "bench"], None),                            # the start of two: tmux finds none
    ("m*", ["build", "mine"], "mine"),                          # a pattern
    ("*", ["build", "mine"], None),                             # a pattern two match
    ("=bui", ["build"], None),                                  # `=`: the whole name only
    ("=build", ["build", "builder"], "build"),
    ("$1", ["build", "mine"], "mine"),                          # an id
    ("$7", ["build", "mine"], None),
    ("nosuch", ["build"], None),
    ("vm", ["vm-secrets", "build"], "vm-secrets"),
])
def test_a_target_names_the_session_tmux_would_find(kept, target, sessions, hit):
    assert kept.named(target, [(f"${i}", n) for i, n in enumerate(sessions)]) == hit


def test_the_session_part_of_a_target_is_what_stands_before_its_colon(kept):
    assert [kept.session_part(t) for t in ("build", "build:", "build:1.0", "=build:1", ":1", "1.0", "")] == \
        ["build", "build", "build", "=build", "", "", ""]


@pytest.mark.skipif(shutil.which("tmux") is None, reason="no tmux on this machine")
def test_start_starts_the_kept_session_once_and_only_on_a_server_that_would_hold_the_secrets(tmp_path_factory):
    """Against a real tmux, on a socket directory of this test's own ($TMUX_TMPDIR, with no $TMUX): what a stand-in can't
    show is that the server `start` starts holds the environment it was started from, that a later session takes its agent's
    id from the shell that starts it, and that the hook's rule reads a real tmux's answers."""
    tmp = tmp_path_factory.mktemp("s")                          # a short path: a socket's is capped near 100 bytes
    path = f"{os.path.dirname(shutil.which('tmux'))}{os.pathsep}{os.defpath}"
    base = {"PATH": path, "HOME": str(tmp), "LANG": "C.UTF-8", "SHELL": "/bin/sh", "TMUX_TMPDIR": str(tmp),
            "AGENT_GUARD_EVENT_LOG": str(tmp / "events.jsonl"), "AGENT_INBOX": str(tmp / "inbox"), "AGENT_INBOX_POLL_S": "off",
            "AGENT_GUARD_STATE": str(tmp / "state"), "AGENT_GUARD_MIN_AVAIL": "0"}
    injected = {"CLOUD_AGENT_INJECTED_SECRET_NAMES": "JOBS_TOKEN,STORE_KEY", "JOBS_TOKEN": "dummy-a", "STORE_KEY": "dummy-b"}
    socket = tmp / f"tmux-{os.getuid()}" / "default"

    def start(**env):
        p = subprocess.run([sys.executable, KEPT_SESSION, "start"], capture_output=True, text=True, env={**base, **env})
        return p.returncode, p.stdout + p.stderr

    def tmux(*args, **env):
        p = subprocess.run(["tmux", "-S", str(socket), *args], capture_output=True, text=True, env={**base, **env})
        return p.returncode, p.stdout.strip()

    def hook(command):
        event = {"hook_event_name": "beforeShellExecution", "command": command, "conversation_id": "bc-me", "cwd": str(tmp)}
        out = subprocess.run([sys.executable, HOOK], input=json.dumps(event), capture_output=True, text=True, check=True, env=base)
        return json.loads(out.stdout)["permission"]

    try:
        code, said = start()
        assert code == 1 and "this environment holds no secret Cursor injects" in said and not socket.exists(), said
        code, said = start(**{**injected, "STORE_KEY": ""})
        assert code == 1 and "lacks 1 of the 2 secrets this VM injects" in said and not socket.exists(), said
        assert f"From a shell that has them: `tmux new-session -d -s {KEPT}`" in said
        assert hook(f"tmux kill-session -t {KEPT}") == "allow", "no server runs: nothing to end"

        code, said = start(**injected)
        assert (code, said) == (0, f"agent-guard: tmux session `{KEPT}` started; its server holds the 2 secrets this VM injects\n")
        assert tmux("list-sessions", "-F", "#{session_name}") == (0, KEPT)
        assert tmux("show-environment", "-g", "JOBS_TOKEN") == (0, "JOBS_TOKEN=dummy-a"), "the server holds what it started with"
        assert start() == (0, f"agent-guard: tmux session `{KEPT}` is already there\n"), "whatever this environment holds"
        assert tmux("list-sessions", "-F", "#{session_name}") == (0, KEPT)

        assert tmux("new-session", "-d", "-s", "work", CURSOR_CONVERSATION_ID="bc-worker")[0] == 0
        assert tmux("show-environment", "-t", "=work:", "CURSOR_CONVERSATION_ID") == (0, "CURSOR_CONVERSATION_ID=bc-worker")
        assert tmux("show-environment", "-t", f"={KEPT}:", "CURSOR_CONVERSATION_ID")[0] == 1, "the kept session is no agent's"

        assert [hook(c) for c in ("tmux kill-server", f"tmux kill-session -t {KEPT}", "tmux kill-session -t vm", "tmux kill-session",
                                  "tmux kill-session -a -t work")] == ["deny"] * 5
        assert hook("tmux kill-session -t work") == "allow"
        assert tmux("kill-session", "-t", "=work")[0] == 0 and tmux("list-sessions", "-F", "#{session_name}") == (0, KEPT)

        assert tmux("new-session", "-d", "-s", "other")[0] == 0 and tmux("kill-session", "-t", f"={KEPT}")[0] == 0
        assert hook("tmux kill-session -t other") == "deny", "the last session"
        assert start() == (0, f"agent-guard: tmux session `{KEPT}` started on the server already running\n")
        assert hook("tmux kill-session -t other") == "allow"
    finally:
        if socket.exists():
            subprocess.run(["tmux", "-S", str(socket), "kill-server"], capture_output=True, env=base)


def test_research_names_the_session_and_the_command_that_starts_it_as_the_guard_does(kept):
    research = (GUARD.parents[1] / "tools/research/src/research/secret_hint.py").read_text()
    assert f'KEPT_SESSION = "{kept.NAME}"' in research and kept.NAME == KEPT
    assert 'START_KEPT = f"tmux new-session -d -s {KEPT_SESSION}"' in research and kept.START == f"tmux new-session -d -s {KEPT}"


def test_boot_starts_the_kept_session_with_the_script_it_copies():
    boot = (GUARD / "boot.sh").read_text()
    assert 'python3 "$dest/kept_session.py" start || true' in boot, "boot goes on when no session could be started"
