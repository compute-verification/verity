#!/usr/bin/env python3
"""`procs.py [PATTERN]`: this machine's processes, each as its pid, the seconds since it started, its command's name and its
command line with every token-like run cut out, and never an environment.

    python3 tools/agent-guard/procs.py [PATTERN]
    ssh HOST python3 - [PATTERN] < tools/agent-guard/procs.py      # on another machine

The shell hook refuses every listing of full command lines, since a daemon's token sits in its argv; this is the form it
points to. PATTERN is a Python regular expression searched in each whole command line (a kernel thread's: its name), as
`pgrep -f` matches, and only matching processes are listed, never this one; what is printed is the cut form (`cut`). It reads
/proc/PID/stat and /proc/PID/cmdline, never /proc/PID/environ. Exits 0 when a process was listed, 1 when none matched, 2 on a
PATTERN that doesn't compile. Standard library only, one file, so it runs from stdin on a machine without a checkout.

Cut, each to `[cut]`: the value after an option or a `KEY=` or `KEY:` whose name holds `token`, `secret`, `passw`, `auth`,
`key`, `credential`, `private` or `cookie` (`--token X`, `--auth=X`, `api_key=X`, `"password": "X"`, `Authorization: Bearer
X`), unless the name ends in `file`, `path` or `dir`; what follows `Bearer ` or `Basic `; a URL's password (`://USER:X@`); a
token with a known prefix (`ghp_`, `github_pat_`, `hf_`, `sk-`, `xoxb-`, `AKIA` and the like); and any run of 20 or more
characters of the base64, hex or URL-safe alphabets that holds a digit or both cases (a git sha, a uuid, a key). A run with `/`
in it is read as a path: each part between its slashes that is such a run, holds a digit and both cases in 8 or more
characters, or holds a `+` is cut, and the whole run when no other part is a lower-case word (`cut_run`). Kept: the program, its
options' names, words of lower-case letters and `-` or `_` (`--disable-log-requests`), and paths.
"""
import os
import re
import sys
from dataclasses import dataclass

USAGE = "usage: python3 procs.py [PATTERN]    (on another machine: ssh HOST python3 - [PATTERN] < tools/agent-guard/procs.py)"
CUT = "[cut]"
SECRET_NAME = r"[\w.-]*(?:token(?!s)|secret|passw|auth|key|credential|private|cookie)[\w.-]*"
NOT_SECRET = re.compile(r"(file|path|dir)$", re.I)
OPTION = re.compile(rf"--?{SECRET_NAME}", re.I)
ASSIGNED = re.compile(rf"""(?<![\w-])(-{{0,2}}{SECRET_NAME})(["']?\s*[=:]\s*["']?)((?:(?:bearer|basic)\s+)?[^\s"'&,;{{}}]+)""", re.I)
SCHEME = re.compile(r"\b(bearer|basic)(\s+)(?!\[cut\])([^\s\"']+)", re.I)
USERINFO = re.compile(r"(\w+://[^/\s:@]*:)[^/\s@]+@")
PREFIXED = re.compile(r"(?<![\w-])(?:gh[pousr]_|github_pat_|hf_|sk-|xox[a-z]-|glpat-|AKIA|ya29\.)[\w.-]{8,}")
RUN = re.compile(r"[A-Za-z0-9_+/-]{20,}={0,2}")


@dataclass
class Proc:
    pid: int
    elapsed: int
    name: str
    argv: list[str]


def mixed(word: str) -> bool:
    return bool(re.search(r"\d", word)) or (word.lower() != word and word.upper() != word)


def cut_run(run: str) -> str:
    """`run`, a run of 20 or more characters of the base64, hex or URL-safe alphabets, with what reads as a token cut: all of
    it when it holds a digit or both cases, or a `+`; when it holds a `/`, each part between slashes that does so in 20 or more
    characters, holds a digit and both cases in 8 or more, or holds a `+`, and all of it when no other part is a lower-case
    word (`cache`, `agent-guard`), as a path's parts are."""
    if "/" not in run:
        return CUT if mixed(run) or "+" in run else run
    parts = run.split("/")
    hit = [len(p) >= 20 and mixed(p) or "+" in p or len(p) >= 8 and bool(re.search(r"\d", p)) and p.lower() != p != p.upper()
           for p in parts]
    if any(hit) and not any(re.fullmatch(r"[a-z][a-z_-]{2,}", p) for p, h in zip(parts, hit) if not h):
        return CUT
    return "/".join(CUT if h else p for p, h in zip(parts, hit))


def cut(argv: list[str]) -> str:
    """`argv` as one line, with every token-like run cut out (module doc)."""
    words = list(argv)
    for i, w in enumerate(words[:-1]):
        if OPTION.fullmatch(w) and not NOT_SECRET.search(w) and not words[i + 1].startswith("-"):
            words[i + 1] = CUT
    line = " ".join(words)
    line = USERINFO.sub(lambda m: m[1] + CUT + "@", line)
    line = ASSIGNED.sub(lambda m: m[0] if NOT_SECRET.search(m[1]) or m[3].startswith(CUT) else m[1] + m[2] + CUT, line)
    line = SCHEME.sub(lambda m: m[1] + m[2] + CUT, line)
    line = PREFIXED.sub(CUT, line)
    return RUN.sub(lambda m: cut_run(m[0]), line)


def processes(root: str = "/proc"):
    """Each process under `root`, from its `stat` and `cmdline` (and `uptime`), skipping one that ends while it is read."""
    ticks = os.sysconf("SC_CLK_TCK")
    with open(os.path.join(root, "uptime")) as f:
        uptime = float(f.read().split()[0])
    for entry in os.listdir(root):
        if not entry.isdigit():
            continue
        try:
            with open(os.path.join(root, entry, "stat"), "rb") as f:
                stat = f.read().decode("utf-8", "replace")
            with open(os.path.join(root, entry, "cmdline"), "rb") as f:
                raw = f.read()
        except OSError as e:                             # most often: it ended between the listing and the read
            print(f"procs.py: {entry} not listed: {e.strerror}", file=sys.stderr)
            continue
        name, _, rest = stat[stat.index("(") + 1:].rpartition(")")
        started = int(rest.split()[19]) / ticks
        argv = [a.decode("utf-8", "replace") for a in raw.rstrip(b"\0").split(b"\0")] if raw else []
        yield Proc(int(entry), max(0, int(uptime - started)), name, argv)


def main(argv: list[str], source=processes) -> int:
    if len(argv) > 1 or argv[:1] in (["-h"], ["--help"]):
        print(USAGE, file=sys.stderr)
        return 2
    try:
        pattern = re.compile(argv[0]) if argv else None
    except re.error as e:
        print(f"procs.py: PATTERN doesn't compile: {e}", file=sys.stderr)
        return 2
    shown = 0
    for p in sorted(source(), key=lambda p: p.pid):
        whole = " ".join(p.argv) if p.argv else p.name
        if p.pid == os.getpid() or (pattern is not None and not pattern.search(whole)):
            continue
        if not shown:
            print(f"{'PID':>7} {'ELAPSED':>8} {'COMMAND':<15} ARGS")
        print(f"{p.pid:>7} {p.elapsed:>8} {cut([p.name])[:15]:<15} {cut(p.argv) if p.argv else '[' + cut([p.name]) + ']'}")
        shown += 1
    return 0 if shown else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
