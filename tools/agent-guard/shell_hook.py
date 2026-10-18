#!/usr/bin/env python3
"""Cursor's `beforeShellExecution` hook (`.cursor/hooks.json`): refuses shell commands that would print secrets wholesale.

Cursor runs every agent shell command through it, on any VM, so it holds where the redacting `env` (`install.sh`) was never
installed. A multi-line secret defeats `env | cut -d= -f1`: its later lines print as if they were names, which is how a line
of the Nebius key reached agents' transcripts at least five times by 6 Oct. Refused, anywhere in the command (pipelines,
`$(...)`, `bash -c`, `ssh HOST CMD`, `research pods ssh POD -- CMD`): `env` or `printenv` with nothing to run, `printenv` of a
secret's name, a bare `set`, `export` or `export -p`, `declare`/`typeset` listing variables, reading `/proc/*/environ`, and
`tmux show-environment` (`showenv`) with no name or a secret's (11 Oct: piping tmux's global environment through `cut -d= -f1`
printed the Nebius key's lines again).
`compgen -e` lists names only and passes. Also refused: full command lines of every process or of any selection beyond one pid
(`ps aux`, `ps -ef`, `ps -eo args`, `ps -o cmd --ppid N`, `ps -f -u USER`, `/proc/*/cmdline`), where a daemon's or a git child's
token can sit in argv; one process (`ps -p PID -o args`), a listing piped straight into
`grep`/`rg`, and `pgrep -af PATTERN` pass. Reads Cursor's hook JSON on stdin and prints its verdict as JSON on stdout.

It also refuses a heavy local command (pytest, the check's suites, `verity check`, `lake build|env|exe`, `cargo build|test|run|check`,
nvcc) while the VM has less than $AGENT_GUARD_MIN_AVAIL (default 0.25) of its memory available (MemAvailable, or its cgroup's
limit less its use when that is less): on a VM several agents share, one more such job is how the kernel OOM-kills another
agent's processes (circuits' coordinator VM, 15 and 18 Oct). The same command through `research run --on NODE`, or over ssh,
runs elsewhere and passes; `AGENT_GUARD_LOW_MEM_OK=1` anywhere in the command passes it too.
"""
import json
import os
import re
import shlex
import sys

PATTERN = re.compile(r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL|PRIVATE", re.I)
WRAPPERS = {"sudo", "doas", "command", "builtin", "exec", "nohup", "time", "nice", "stdbuf", "timeout", "setsid", "taskset"}
WRAPPER_ARG_OPTIONS = {"sudo": {"-u", "-g", "-C", "-h", "-p", "-U", "-r", "-t"}, "doas": {"-u", "-C"}, "nice": {"-n"},
                       "timeout": {"-s", "-k", "--signal", "--kill-after"}, "taskset": {"-c", "-p"}}
KEYWORDS = {"{", "}", "!", "if", "then", "elif", "else", "do", "while", "until"}
SHELLS = {"bash", "sh", "zsh", "dash"}
SSH_ARG_OPTIONS = set("bcDEeFIiJLlmOopQRSWw")
TMUX_ARG_OPTIONS = set("cfLST")
ENVIRON = re.compile(r"/proc/[^\s'\"]*environ|/proc/\*/cmdline")
FILTERS = {"grep", "egrep", "fgrep", "rg"}
FALLBACK = re.compile(r"(^|[\s|;&(`])(env|printenv|set|export|declare|typeset)(\s+-\S+)*\s*($|[|;&)>`])|show-?env", re.M)
HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2[ \t]*(?=\n|$)", re.S)

REASON = ("this command would print environment variables' values (a multi-line secret's later lines print even through "
          "`cut -d= -f1`) or every process's full command line (tokens sit in some daemons' argv). List variable names with "
          "`compgen -e` (filter with `compgen -e | rg NAME`); check one variable with `[ -n \"${NAME:-}\" ] && echo set`. "
          "List processes with `ps -eo pid,stat,etimes,comm`, or filter full command lines with "
          "`ps -eo pid,args | grep \"[p]attern\"` or `pgrep -af PATTERN`.")


def secret(name: str) -> bool:
    listed = set(re.split(r"[,\s]+", os.environ.get("CLOUD_AGENT_ALL_SECRET_NAMES", ""))) - {""}
    return name in listed or bool(PATTERN.search(name))


def words_of(text: str) -> list[tuple[list[str], str]]:
    """`text`'s simple commands, each as its words and the operator after it, split at pipes, lists, subshells, newlines and
    backticks."""
    text = HEREDOC.sub(" ", text).replace("`", " ; ").replace("\n", " ; ")
    text = re.sub(r"(?<![\w-])\d+(?=[<>])", " ", text)      # `2>/dev/null`: the fd number is not an argument
    lex = shlex.shlex(text, posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    commands, current = [], []
    for tok in lex:
        if tok and set(tok) <= set("|&;()<>") and (set(tok) & set("()") or not set(tok) & set("<>")):
            commands.append((current, tok))
            current = []
        else:
            current.append(tok)
    commands.append((current, ""))
    return [c for c in commands if c[0]]


def strip_redirections(words: list[str]) -> list[str]:
    """`words` without redirection operators and their targets (`> f`, `>& 1`, `<<< s`)."""
    out, skip = [], False
    for w in words:
        if skip:
            skip = False
        elif w and set(w) <= set("&;<>|") and set(w) & set("<>"):
            skip = True
        else:
            out.append(w)
    return out


def env_command(args: list[str]) -> list[str]:
    """What `env ARGS` runs (empty when it would print the environment instead)."""
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-u", "--unset", "-C", "--chdir", "-S", "--split-string"):
            if a in ("-S", "--split-string"):
                return args[i + 1:i + 2]
            i += 1
        elif a == "--":
            return args[i + 1:]
        elif a.startswith("-") or ("=" in a and not a.startswith("=")):
            pass
        else:
            return args[i:]
        i += 1
    return []


def unwrap(words: list[str]) -> list[str]:
    """`words` without leading assignments and wrappers (`sudo -u USER`, `timeout 30`, `nice -n 5`, `command`, ...)."""
    while words:
        if words[0] in KEYWORDS or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[0], re.S):
            words = words[1:]
        elif os.path.basename(words[0]) in WRAPPERS:
            wrapper, words = os.path.basename(words[0]), words[1:]
            while words and words[0].startswith("-"):
                words = words[2:] if words[0] in WRAPPER_ARG_OPTIONS.get(wrapper, ()) else words[1:]
            if wrapper == "timeout":
                words = words[1:]
        else:
            return words
    return words


def one_pid(args: list[str]) -> bool:
    """Whether `ps ARGS` selects exactly one process by pid (`-p PID`, `--pid PID`) and nothing else."""
    sel = [(a, args[i + 1]) for i, a in enumerate(args[:-1]) if a in ("-p", "--pid", "-q", "--quick-pid")]
    others = any(a in ("--ppid", "-C", "-u", "-U", "-g", "-G", "-t", "--user", "--group", "--sid", "-s") or a.startswith("--ppid=")
                 for a in args)
    return len(sel) == 1 and re.fullmatch(r"\d+", sel[0][1]) is not None and not others


def every_argv(args: list[str]) -> bool:
    """Whether `ps ARGS` prints full command lines of more than one named process: of every process, or of a selection such as
    `--ppid N`, `-C NAME` or `-u USER` (17 Oct, INF-52: `ps -o pid,etime,cmd --ppid N` printed a git child's token-bearing URL)."""
    fmt = next((args[i + 1] for i, a in enumerate(args[:-1])
                if a == "--format" or re.fullmatch(r"-?[a-zA-Z]*[oO]", a)), None)
    fmt = fmt or next((a.split("=", 1)[1] for a in args if a.startswith("--format=")), None)
    shown = fmt is None or bool(re.search(r"\b(args|cmd|command)\b", fmt))
    bsd = [a for i, a in enumerate(args) if not a.startswith("-") and re.fullmatch(r"[a-zA-Z]+", a) and a != fmt
           and not (i and args[i - 1].startswith("-"))]
    short = "".join(a[1:] for a in args if a.startswith("-") and not a.startswith("--"))
    if any(set(b) & set("ax") for b in bsd):
        return shown and not any("c" in b for b in bsd)
    if one_pid(args):
        return False
    return bool(set(short) & set("fF")) if fmt is None else shown


def refused(words: list[str], filtered: bool = False) -> bool:
    if any(refused_text(w.split("$(", 1)[1]) for w in words if "$(" in w):
        return True
    words = unwrap(strip_redirections(words))
    if not words:
        return False
    name, args = os.path.basename(words[0]), words[1:]
    opts = [a for a in args if a.startswith(("-", "+"))]
    names = [a for a in args if not a.startswith(("-", "+"))]
    if name == "env":
        run = env_command(args)
        if not run:
            return True
        return refused_text(run[0]) if len(run) == 1 and " " in run[0] else refused(run)
    if name == "printenv":
        return not names or any(secret(n) for n in names)
    if name == "set":
        return not args
    if name == "export":
        return all(a == "-p" for a in args)
    if name in ("declare", "typeset", "local", "readonly"):
        if not names:
            return not opts or any(set(o.lstrip("-+")) - set("fF") for o in opts)
        return any("p" in o for o in opts) and any(secret(n) for n in names)
    if name in SHELLS and "-c" in args[:-1]:
        return refused_text(args[args.index("-c") + 1])
    if name == "ssh":
        i = 0
        while i < len(args) and args[i].startswith("-"):
            i += 2 if args[i][-1] in SSH_ARG_OPTIONS and len(args[i]) == 2 else 1
        return refused_text(" ".join(args[i + 1:])) if i + 1 < len(args) else False
    if name == "research" and "ssh" in args and "--" in args:
        return refused_text(" ".join(args[args.index("--") + 1:]))
    if name == "ps":
        return not filtered and every_argv(args)
    if name == "tmux":
        i = 0
        while i < len(args) and args[i].startswith("-"):
            i += 2 if args[i][-1] in TMUX_ARG_OPTIONS and len(args[i]) == 2 else 1
        if i < len(args) and (args[i] == "showenv" or args[i].startswith("show-env")):      # tmux takes a command's prefix
            rest, shown = args[i + 1:], []
            while rest:
                a, rest = rest[0], rest[1:]
                if a == "-t":
                    rest = rest[1:]
                elif not a.startswith("-"):
                    shown.append(a)
            return not shown or any(secret(n) for n in shown)
        return False
    if name == "xargs":
        rest = [a for a in args if not a.startswith("-")]
        return bool(rest) and refused(rest)
    return False


def filter_cmd(words: list[str]) -> bool:
    words = unwrap(strip_redirections(words))
    return bool(words) and os.path.basename(words[0]) in FILTERS


def refused_text(text: str) -> bool:
    if ENVIRON.search(text):
        return True
    try:
        commands = words_of(text)
        return any(refused(w, op == "|" and i + 1 < len(commands) and filter_cmd(commands[i + 1][0]))
                   for i, (w, op) in enumerate(commands))
    except ValueError:                                   # unbalanced quotes: judge the raw text
        return bool(FALLBACK.search(text))


HEAVY_PROGRAMS = {"pytest": None, "py.test": None, "nvcc": None, "suites.py": None, "lake": {"build", "env", "exe"},
                  "cargo": {"build", "test", "run", "check"}, "verity": {"check"}}
UV_RUN_VALUED = {"--extra", "--with", "--package", "-p", "--python", "--directory", "--project", "--group", "--env-file", "--index"}
LOW_MEM_OK = "AGENT_GUARD_LOW_MEM_OK=1"


def heavy_words(words: list[str]) -> bool:
    """Whether one simple command runs a heavy program here, seen through `uv run [OPTS]`, `python -m MOD` and `python SCRIPT`."""
    words = unwrap(strip_redirections(words))
    while words and re.fullmatch(r"\w+=.*", words[0]):
        words = words[1:]
    if not words:
        return False
    prog = os.path.basename(words[0])
    if prog == "uv" and words[1:2] == ["run"]:
        rest, i = words[2:], 0
        while i < len(rest) and rest[i].startswith("-"):
            i += 2 if rest[i] in UV_RUN_VALUED else 1
        return heavy_words(rest[i:])
    if re.fullmatch(r"python[\d.]*", prog):
        args = words[1:]
        if args[:1] == ["-m"]:
            return heavy_words(args[1:])
        script = next((a for a in args if not a.startswith("-")), None)
        return script is not None and heavy_words([script, *args[args.index(script) + 1:]])
    if prog not in HEAVY_PROGRAMS:
        return False
    subs = HEAVY_PROGRAMS[prog]
    return subs is None or bool(set(words[1:2]) & subs)


def heavy(text: str) -> bool:
    """Whether a simple command of `text` runs heavy work on this VM: a command through `research run` or over ssh runs elsewhere."""
    try:
        commands = words_of(text)
    except ValueError:
        return False
    return any(heavy_words(words) for words, _ in commands)


def available() -> tuple[float, float] | None:
    """(available, total) bytes of this VM's memory: /proc/meminfo's MemAvailable and MemTotal, or this process's cgroup's limit and
    its headroom where that limit is tighter."""
    try:
        info = dict(re.findall(r"^(\w+):\s+(\d+) kB", open(os.environ.get("AGENT_GUARD_MEMINFO", "/proc/meminfo")).read(), re.M))
        avail, total = int(info["MemAvailable"]) * 1024, int(info["MemTotal"]) * 1024
    except (OSError, KeyError, ValueError):
        return None
    try:
        rel = open("/proc/self/cgroup").read().split("::", 1)[1].strip()
        cg = os.environ.get("AGENT_GUARD_CGROUP") or f"/sys/fs/cgroup{rel}"
        limit = open(f"{cg}/memory.max").read().strip()
        if limit != "max" and int(limit) < total:
            total, avail = int(limit), min(avail, int(limit) - int(open(f"{cg}/memory.current").read()))
    except (OSError, IndexError, ValueError):
        pass
    return avail, total


def low_memory_reason(command: str) -> str | None:
    if LOW_MEM_OK in command or not heavy(command):
        return None
    mem = available()
    floor = float(os.environ.get("AGENT_GUARD_MIN_AVAIL", "0.25"))
    if mem is None or mem[0] >= floor * mem[1]:
        return None
    return (f"this VM has {mem[0] / 2**30:.1f} GiB of {mem[1] / 2**30:.1f} GiB available (under {floor:.0%}), and this command runs "
            f"heavy work here: on a VM other agents share, the kernel would OOM-kill some agent's processes. Run it on a CPU node "
            f"(`research run --on vy-nebius-cpu-1 --pool research --source . --mem-gb G -- CMD`, or your lane's pool), or wait "
            f"for memory; for a small run you know fits, put {LOW_MEM_OK} in the command.")


def main() -> int:
    raw = sys.stdin.read()
    try:
        command = json.loads(raw).get("command") or ""
    except (ValueError, AttributeError):
        command = raw
    try:
        deny = refused_text(command)
    except Exception:                                   # a bug here must not allow what the fallback would refuse
        deny = bool(FALLBACK.search(command) or ENVIRON.search(command))
    try:
        low = None if deny else low_memory_reason(command)
    except Exception:                                   # a bug in the memory gate must not stop every command
        low = None
    if deny or low:
        reason = REASON if deny else low
        print(json.dumps({"permission": "deny",
                          "user_message": "Refused by tools/agent-guard/shell_hook.py: " + reason,
                          "agent_message": "Refused by tools/agent-guard/shell_hook.py (.cursor/hooks.json): " + reason}))
    else:
        print(json.dumps({"permission": "allow"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
