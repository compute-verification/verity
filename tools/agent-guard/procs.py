#!/usr/bin/env python3
"""`procs.py [PATTERN] [-p PID]`: this machine's processes, one a line as pid, seconds running, name and command line, with
whatever may be a credential in the command line replaced by [REDACTED].

    python3 tools/agent-guard/procs.py PATTERN                  # those whose masked command line matches the regex PATTERN
    python3 tools/agent-guard/procs.py -p PID                   # one process (`-p PID,PID`: several); no argument: all
    ssh HOST python3 - PATTERN < tools/agent-guard/procs.py     # on another machine

A command line holds whatever its program was handed, and some programs are handed tokens (Cursor's exec daemon holds three).
`ps` with a command-line column, `pgrep -a` and `/proc/PID/cmdline` print them as they are, a filter after them keeps the
line that holds one, and the shell hook (shell_hook.py) therefore refuses them and names this instead. Masked here:

- the value of a flag, a variable, a field or a header whose name holds key, token, secret, passw, credential, private or
  auth (NAMES): `--flag VALUE`, `--flag=VALUE`, `NAME=VALUE`, `"name": "value"`, `Name: value`, to the end of its argument;
- everything after `Authorization:`, and the word after `Bearer`;
- a URL's userinfo, which is what the git guard takes off a remote (git_guard.sh's CRED);
- a JWT, and a run of 32 or more hex digits;
- an opaque word: a run of OPAQUE (20) or more characters other than `/ . : , @ =` that holds a digit or both cases, a word
  of base64 that holds a digit and both cases (its `/` would cut it into shorter runs), and any word of LONG (100) or more
  characters, of which a leading `--flag=` stays.

That masks ids, hashes and long names too, on purpose: a reader wants the program and its subcommand, and one mask too many
costs nothing. PATTERN is searched in the masked line, never in the raw one, so the first characters of a secret match
nothing. Exits 0 when a process was listed, 1 when none was, 2 on a usage error.

install.sh installs this file as `procs` and, as `ps` and `pgrep`, ahead of /usr/bin. Under those two names it runs the real
program on the same arguments and masks each line it prints, so a VM whose hook never loaded prints no token through them.

It guards against accidents and is no boundary. The masks are patterns: a short secret under a name that doesn't say so
(`sshpass -p VALUE`, curl's `-u USER:PASSWORD`) shows, and a program that opens `/proc/PID/cmdline` itself reads everything.
The fix is for no program to take a token in its arguments. Standard library only, one file: install.sh copies it alone, and
it runs from stdin on a machine without a checkout.
"""
import os
import re
import signal
import subprocess
import sys

USAGE = ("usage: python3 procs.py [PATTERN] [-p PID[,PID...]]    "
         "(on another machine: ssh HOST python3 - PATTERN < tools/agent-guard/procs.py)")
REAL = {"ps": "/usr/bin/ps", "pgrep": "/usr/bin/pgrep"}
MASK = "[REDACTED]"
#: what a secret's name holds: env_guard.py's PATTERN, and `auth`
NAMES = "KEY|TOKEN|SECRET|PASSW|CREDENTIAL|PRIVATE|AUTH"
OPAQUE, LONG = 20, 100
NAME = rf"[\w.-]{{0,80}}?(?:{NAMES})[\w.-]{{0,80}}"
#: a secret's name with its value after it: as the start of an argument, whose rest is the value, and anywhere in a word
START = re.compile(rf"""({NAME}["']?[=:])(.*)""", re.I | re.S)
NAMED = re.compile(rf"""(.*?{NAME}["']?[=:])(.*)""", re.I | re.S)
#: a flag that names a secret, whose value is the next word unless that is an option too
FLAG = re.compile(rf"--?{NAME}", re.I)
OPTION = re.compile(r"--?[A-Za-z][A-Za-z-]*(=.*)?", re.S)
AUTHORIZATION = re.compile(r"authorization:", re.I)
#: a URL's userinfo as git_guard.sh's CRED reads it: `user:password@`, a GitHub token, or a user name of 30 characters or more
USERINFO = re.compile(r"://(?:[^/@\s]*:[^/@\s]*|(?:gh[pousr]|github_pat)_[\w.-]+|[\w.-]{30,})@")
JWT = re.compile(r"eyJ[\w-]*\.[\w.-]+")
HEX = re.compile(r"[0-9a-fA-F]{32,}")
BASE64 = re.compile(rf"(?<![^=])[A-Za-z0-9+/_-]{{{OPAQUE},}}=*$")
RUN = re.compile(rf"[^\s/.:,@=]{{{OPAQUE},}}")
LONG_FLAG = re.compile(r"(--?[\w-]{1,40}=)?")


def kinds(text: str) -> int:
    """How many of a digit, a small letter and a capital `text` holds."""
    return sum(bool(re.search(kind, text)) for kind in (r"\d", "[a-z]", "[A-Z]"))


def shapes(word: str) -> str:
    """`word` with each part of it that is shaped like a credential masked (the module doc's last three kinds)."""
    for shape, to in ((USERINFO, f"://{MASK}@"), (JWT, MASK), (HEX, MASK)):
        word = shape.sub(to, word)
    word = BASE64.sub(lambda m: MASK if kinds(m[0]) == 3 else m[0], word)
    return RUN.sub(lambda m: MASK if re.search(r"\d", m[0]) or kinds(m[0]) == 2 else m[0], word)


def masked_line(line: str) -> str:
    """`line` with every word of it that may be a credential masked, its spacing kept. A line's words are all there is to go
    by, so a secret's value is the word after its flag or name, and everything after `Authorization:` is one."""
    parts, hide = re.split(r"(\s+)", line), ""          # hide: the "next" word is a value, or the "rest" of the line is
    for i in range(0, len(parts), 2):
        word = parts[i]
        if not word:
            continue
        if hide == "rest" or (hide == "next" and not OPTION.fullmatch(word)):
            parts[i], hide = MASK, hide if hide == "rest" else ""
        elif len(word) >= LONG:
            parts[i], hide = LONG_FLAG.match(word)[0] + MASK, ""
        else:
            named = NAMED.fullmatch(word)
            head, value = (named[1], named[2]) if named else (word, "")
            hide = ("rest" if AUTHORIZATION.search(head) else
                    "next" if (named and not value) or FLAG.fullmatch(word) or word.lower() == "bearer" else "")
            parts[i] = shapes(head) + (MASK if value else "")
    return "".join(parts)


def masked(argv: list[str]) -> str:
    """The command line `argv` on one line, masked. An argument is known whole here, which a line's words are not: the one
    after a flag that names a secret is its value, and so is the rest of one that starts with a secret's name, spaces and
    all. Any other argument is masked as a line of words (a `bash -c` script, a title a daemon wrote over its arguments)."""
    out, value = [], False
    for arg in argv:
        start = START.fullmatch(arg)
        if value and not OPTION.fullmatch(arg):
            out.append(MASK)
        elif start and start[2].strip():
            out.append(shapes(start[1]) + MASK)
        else:
            out.append(masked_line(arg))
        value = bool(FLAG.fullmatch(arg) or (start and not start[2].strip()) or arg.lower() == "bearer")
    return "".join(c if c.isprintable() else "?" for c in " ".join(out))


def processes():
    """(pid, seconds running, name, arguments) of every process but this one, by pid; one that ends meanwhile is left out."""
    with open("/proc/uptime") as f:
        up = float(f.read().split()[0])
    for pid in sorted(int(d) for d in os.listdir("/proc") if d.isdigit()):
        if pid == os.getpid():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = [a.decode("utf-8", "replace") for a in f.read().split(b"\0")]
            with open(f"/proc/{pid}/stat", "rb") as f:
                name, _, rest = f.read().decode("utf-8", "replace").partition("(")[2].rpartition(")")
        except OSError:
            continue
        while argv and not argv[-1]:
            argv.pop()
        yield pid, max(0, int(up - int(rest.split()[19]) / os.sysconf("SC_CLK_TCK"))), name, argv


def filtered(tool: str, args: list[str]) -> int:
    """Stand in for the real `tool`: run it on `args` and write what it prints, each line masked; its exit status.

    This process must not show in what it stands in for. A `pgrep` not asked to list (`-a`, `-l`) prints pids alone, so the
    real one takes this process's place. A `pgrep` that lists would match this process too, its pattern being in this
    command line: that line is left out, and with it the status 0 when it was the only match. And this process goes by the
    tool's name, not python's, so `ps -C python3` and `pgrep python3` find what they did."""
    try:
        if tool == "pgrep" and not any(re.fullmatch(r"-[A-Za-z]*[al][A-Za-z]*|--list-.*", a) for a in args):
            os.execv(REAL[tool], [tool, *args])
        with open("/proc/self/comm", "w") as comm:
            comm.write(tool)
        real = subprocess.Popen([REAL[tool], *args], stdout=subprocess.PIPE)
    except OSError as e:
        print(f"{tool}: {REAL[tool]}: {e.strerror}", file=sys.stderr)
        return 127
    shown = 0
    for line in real.stdout:
        text = line.decode("utf-8", "replace")
        if tool != "pgrep" or text.split(None, 1)[:1] != [str(os.getpid())]:
            sys.stdout.write(masked_line(text))
            shown += 1
    sys.stdout.flush()
    status = real.wait()
    return 128 - status if status < 0 else 1 if tool == "pgrep" and not status and not shown else status


def main(argv: list[str]) -> int:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)        # `procs | head` ends quietly, as `ps | head` does
    if hasattr(sys.stdout, "reconfigure"):               # a name the terminal's encoding lacks must not end the listing
        sys.stdout.reconfigure(errors="backslashreplace")
    if os.path.basename(argv[0]) in REAL:
        return filtered(os.path.basename(argv[0]), argv[1:])
    pids, pattern, args = set(), None, argv[1:]
    try:
        while args:
            a = args.pop(0)
            if a in ("-p", "--pid") and args:
                pids |= {int(p) for p in args.pop(0).split(",")}
            elif a.startswith("-") or pattern is not None:
                raise ValueError(a)
            else:
                pattern = re.compile(a)
    except (ValueError, re.error) as e:
        print(f"procs: {e}\n{USAGE}", file=sys.stderr)
        return 2
    found = 0
    for pid, seconds, name, arguments in processes():
        line = masked(arguments) or f"[{name}]"
        if (not pids or pid in pids) and (pattern is None or pattern.search(line)):
            print(f"{pid:>7} {seconds:>8} {name:<15} {line}")
            found += 1
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
