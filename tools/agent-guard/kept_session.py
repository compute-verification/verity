#!/usr/bin/env python3
"""The tmux session a VM keeps, `vm-secrets`, so that its tmux server never ends: `boot.sh` starts it, and the shell hook
refuses what would end it or the server.

A Cursor VM's secrets reach a process only through its environment, and a long-lived shell can lose them. A tmux server's
environment is fixed when the server starts, every shell on its panes inherits it, and `research` takes a secret its shell
lost back from it (`research.secret_hint.recover`). The server lives only while it has a session: when the last one ends,
that copy of the secrets ends with it, and the next server holds whatever the shell that starts it still has.

    python3 tools/agent-guard/kept_session.py start     # what boot.sh runs
    tmux new-session -d -s vm-secrets                   # the same by hand (START), from a shell that has the secrets

`start` starts one detached session of a login shell, NAME, unless one is there. Where no server runs yet the session starts
one, so `start` does that only from an environment that holds every secret the VM injects (settings.INJECTED): a server
started without them would hand every later tmux shell on the VM an environment without them, and be kept. A server it
starts reads Cursor's tmux configuration where the VM has one, as an agent's own tmux does, and copies PASSED_ON from the
shell that starts each later session into that session, so a pane's commands run as the agent that started its session and
not as nobody: the boot's environment names no agent. The kept session itself has none, and is no place to work.

`reason` is the shell hook's rule (`shell_hook.kept_session_reason` reads the command; this judges it). On the VM's own
server it refuses `tmux kill-server`, and a `tmux kill-session` that would end NAME (named exactly, by the start of its name
or by a pattern, as tmux reads a target), that would end the server's last session, or that names no session the hook can
read (no `-t`, a variable, a window or a pane). A `kill-session -a` ends every session but one, whoever's they are, and the
hook can't tell other agents' sessions from the caller's: it is refused whatever its target while the server has more than
one session. For a `kill-session` it asks tmux which sessions there are (`sessions`), once; where tmux gives no answer it
judges by the command's text alone and refuses what could name NAME, and every `-a`. `kill-session -C` ends nothing and
passes, as does a kill that finds no session. What it can't see: which server a pane's own tmux talks to (a `tmux
kill-server` typed into a pane is judged as the VM's), a `TMUX` or `TMUX_TMPDIR` the command sets, a script's contents, and
a session ended any other way (its last window or pane killed, its shell's `exit`, a signal).
"""
from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
import sys

from settings import INJECTED, INJECTED_UNSET

NAME = "vm-secrets"
START = f"tmux new-session -d -s {NAME}"
#: Cursor's tmux configuration, which a server reads only when it starts
PORTAL_CONF = "/exec-daemon/tmux.portal.conf"
#: what a session takes from the shell that starts it, on a server `start` started (tmux's `update-environment`)
PASSED_ON = ("CURSOR_*", "PATH")
#: what tmux says when no server listens on its socket
NO_SERVER = re.compile(r"no server running|error connecting to|server exited unexpectedly")
TIMEOUT_S = 3
UNEXPANDED = ("$", "`")

WHY = ("This VM's tmux server holds in its environment the secrets it started with, and a shell that lost them takes them "
       "back from it (`research` does, before every command); the server ends with its last session.")
BY_NAME = "Close your own sessions by name instead: `tmux kill-session -t NAME` (`tmux list-sessions` names them)."
KEEPS = f"`{NAME}`, the session this VM keeps so that its tmux server never ends"
BUT_ONE = "ends every session on this VM's tmux server but one, and the hook can't tell other agents' sessions from yours"


def tmux(*args: str) -> subprocess.CompletedProcess | str:
    """`tmux ARGS` on the VM's own server (the default socket), or why tmux gave no answer."""
    try:
        return subprocess.run([shutil.which("tmux") or "tmux", *args], capture_output=True, text=True, timeout=TIMEOUT_S,
                              stdin=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        return f"tmux gave no answer ({type(e).__name__})"


def sessions() -> list[str] | str:
    """The name of each session of the VM's own tmux server, none where no server runs; or why tmux couldn't say."""
    p = tmux("list-sessions", "-F", "#{session_name}")
    if isinstance(p, str):
        return p
    if p.returncode:
        return [] if NO_SERVER.search(p.stderr) else f"`tmux list-sessions` exits {p.returncode}"
    return [line for line in p.stdout.splitlines() if line]


def session_part(target: str) -> str:
    """The session `target` names as tmux reads a target-session, or "" when it names a window or a pane and leaves the
    session to tmux (`:1`, `0.1`)."""
    return target.split(":", 1)[0] if ":" in target else "" if "." in target else target


def named(part: str, names: list[str]) -> str | None:
    """The one session `part` names among `names`, as tmux finds it: by its whole name after `=`, else by its name, by the
    start of its name, or as a pattern. None when it names none or more than one: tmux then ends nothing. (tmux also takes
    a session's id, `$3`; the hook reads no `$`, so `unread` refuses one before it gets here.)"""
    if part.startswith("="):
        return part[1:] if part[1:] in names else None
    if part in names:
        return part
    for hits in ([n for n in names if n.startswith(part)], [n for n in names if fnmatch.fnmatchcase(n, part)]):
        if hits:
            return hits[0] if len(hits) == 1 else None
    return None


def could_name(part: str) -> bool:
    """Whether `part` may name NAME, whatever other sessions there are."""
    return part[1:] == NAME if part.startswith("=") else NAME.startswith(part) or fnmatch.fnmatchcase(NAME, part)


def unread(given: dict[str, str]) -> str:
    """Why the hook can't read which session a `kill-session` with these options ends, or ""."""
    if not given.get("t"):
        return "it has no `-t`, so tmux picks the session used last"
    if any(c in given["t"] for c in UNEXPANDED):
        return f"`-t {given['t']}` names it through a variable, which the hook can't read"
    if not session_part(given["t"]).lstrip("="):
        return f"`-t {given['t']}` names a window or a pane and leaves the session to tmux"
    return ""


def reason(kills: list[tuple[str, dict[str, str]]], listed: list[str] | str | None = None) -> str | None:
    """Why the command giving the VM's own tmux server these commands may not run, or None. `kills`: each `kill-server` and
    `kill-session` it gives, in order, with the options it has (a letter to its value); `listed`: `sessions()`'s answer, asked
    for when a `kill-session` first needs it."""
    for name, given in kills:
        if name == "kill-server":
            return (f"`tmux kill-server` would end this VM's tmux server and every session on it, other agents' among them. "
                    f"{WHY} {BY_NAME} A server of your own (`tmux -L NAME new-session ...`) ends with `tmux -L NAME kill-server`.")
        if "C" in given:
            continue
        listed = sessions() if listed is None else listed
        if listed == []:
            continue
        but_one, what = "a" in given, unread(given)
        typed = f"`tmux kill-session{' -a' if but_one else ''}{' -t ' + given['t'] if given.get('t') else ''}`"
        part = session_part(given.get("t", ""))
        if but_one and isinstance(listed, str):
            return f"{typed} {BUT_ONE}, nor ask tmux which are there ({listed}). {BY_NAME}"
        if but_one and len(listed) == 1:
            continue                                    # the one session there is the one it spares, or it finds none
        if but_one:
            spared = None if what else named(part, listed)
            kept = "." if spared == NAME or NAME not in listed else f", {KEEPS}, {'is' if spared else 'may be'} among them. {WHY}"
            return f"{typed} {BUT_ONE}: {len(listed) - 1} would end{kept} {BY_NAME}"
        if what and isinstance(listed, list) and len(listed) == 1:
            hit = listed[0]                             # the one session there is the one tmux picks
        elif what:
            return (f"{typed} doesn't say which session it ends ({what}), and that may be {KEEPS}, or another agent's. {WHY} "
                    f"{BY_NAME}")
        elif isinstance(listed, str):
            if could_name(part):
                return (f"{typed} may end {KEEPS}: tmux takes the start of a name, or a pattern, for a session, and the hook "
                        f"can't ask it which are there ({listed}). {WHY} Leave it there. {BY_NAME}")
            continue
        elif (hit := named(part, listed)) is None:
            continue
        if hit == NAME:
            return f"{typed} would end {KEEPS}. {WHY} Leave it there. {BY_NAME}"
        if len(listed) == 1:
            return (f"{typed} would end `{hit}`, the last session on this VM's tmux server, and the server with it. {WHY} "
                    f"Start the session this VM keeps first, `{START}`; this command then passes.")
        listed = [n for n in listed if n != hit]
    return None


def start() -> tuple[int, str]:
    """Start NAME unless it is there: (exit code, what happened)."""
    listed = sessions()
    if isinstance(listed, str):
        return 1, f"no tmux session started: {listed}"
    if NAME in listed:
        return 0, f"tmux session `{NAME}` is already there"
    if not listed and (INJECTED_UNSET or not INJECTED):
        lacks = f"lacks {len(INJECTED_UNSET)} of the {len(INJECTED)} secrets this VM injects" if INJECTED else "holds no secret Cursor injects"
        return 1, (f"no tmux session started: no tmux server runs, and this environment {lacks}. A server started from it would "
                   f"hand every later tmux shell an environment without them. From a shell that has them: `{START}`")
    conf = ["-f", PORTAL_CONF] if not listed and os.path.isfile(PORTAL_CONF) else []
    made = tmux(*conf, "new-session", "-d", "-s", NAME, "-c", os.path.expanduser("~"))
    if isinstance(made, str) or made.returncode:
        return 1, f"no tmux session started: `{START}` " + (made if isinstance(made, str) else f"exits {made.returncode}")
    if listed:
        return 0, f"tmux session `{NAME}` started on the server already running"
    passed = [tmux("set-option", "-g", "-a", "update-environment", name) for name in PASSED_ON]
    failed = next((f"; `tmux set-option update-environment` failed, so a later session takes nothing from the shell that "
                   f"starts it ({p if isinstance(p, str) else f'exit {p.returncode}'})"
                   for p in passed if isinstance(p, str) or p.returncode), "")
    return 0, f"tmux session `{NAME}` started; its server holds the {len(INJECTED)} secrets this VM injects{failed}"


def main(argv: list[str]) -> int:
    if argv[1:] != ["start"]:
        print(f"usage: {os.path.basename(argv[0])} start", file=sys.stderr)
        return 2
    code, text = start()
    print(f"agent-guard: {text}", file=sys.stderr if code else sys.stdout)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
