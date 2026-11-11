#!/usr/bin/env python3
"""Cursor's `beforeShellExecution` hook (`.cursor/hooks.json`): refuses shell commands that would print secrets wholesale.

Cursor runs every agent shell command through it, on any VM, so it holds where the redacting `env` (`install.sh`) was never
installed. A multi-line secret defeats `env | cut -d= -f1`: its later lines print as if they were names, and so reach an
agent's transcript. Refused, anywhere in the command (pipelines,
`$(...)`, `bash -c`, `ssh HOST CMD`, `research pods ssh POD -- CMD`): `env` or `printenv` with nothing to run, `printenv` of a
secret's name, a bare `set`, `export` or `export -p`, `declare`/`typeset` listing variables, reading `/proc/*/environ`, and
`tmux show-environment` (`showenv`) with no name or a secret's (tmux's global environment through `cut -d= -f1` prints a
multi-line key's lines too).
`compgen -e` lists names only and passes. Also refused: full command lines of every process or of any selection beyond one pid
(`ps aux`, `ps -ef`, `ps -eo args`, `ps -o cmd --ppid N`, `ps -f -u USER`, `/proc/*/cmdline`), where a daemon's or a git child's
token can sit in argv; one process (`ps -p PID -o args`), a listing piped straight into a `grep`/`rg` that keeps only matching
lines (an inverted one, `grep -v`, prints the rest, so `ps -ef | grep -v grep` is refused and `ps -ef | grep -v grep | grep
NAME` passes), and `pgrep -af PATTERN` pass. Reads Cursor's hook JSON on stdin, `beforeShellExecution`'s (`command`, `cwd`) or a
`preToolUse` event's for the Shell tool (`tool_input.command`), and prints its verdict as JSON on stdout; each event's key
names go to $AGENT_GUARD_EVENT_LOG (`settings.record_event`).

A secret file is refused the same way: a reading command (`cat`, `head`, `tail`, `less`, `more`, `tac`, `nl`, `sed`, `awk`,
`cut`, `xxd`, `od`, `strings`, `base64`, and a `grep`/`rg` that prints lines, one without `-l`, `-L`, `-c` or `-q`) applied to a
private key (`*.pem`, `*.key`, `id_rsa*`, `id_ed25519*`, `id_ecdsa*`), an env or credentials file (`*.env`, `.netrc`,
`credentials`, `credentials.*`) or anything in a secrets directory (`.ssh/`, `.config/verity/`, a `secrets/`), named as an
argument, a glob, its input (`< FILE`) or a `for` loop's variable, wherever the command sits: over ssh, after `research pods ssh
POD --`, in a quoted string handed to a wrapper script. A glob over a secrets directory matches the key files beside the env
files, and a filter that cuts each line at its `=` prints every line of a key. `envnames.py FILE...` lists an env file's
variable names instead and prints nothing of a file that isn't one. What shows no contents passes (`ls`, `stat`, `wc`,
`sha256sum`, `cp`, writing or sourcing the file, `sed -i`, `openssl pkey -pubout`, `ssh-keygen -l`), as do public halves
(`*.pub`, `*-pub.pem`, `authorized_keys`, `known_hosts`, `.ssh/config`), `*.example` and `*.template` files, a module or a doc
(`credentials.py`) and the fixtures under a `tests/` directory. `AGENT_GUARD_LOCAL_OK=1` does not lift it.

Builds and tests run on our nodes, not on agent VMs: on a shared VM one OOM kill takes down every worker on it. So it refuses,
always, a local pytest that names no test file or `-k` selection, `lake build|exe`, `cargo build|test|run|check`, nvcc and
`circuit-check --all`, pointing to `research run --on auto`; and while the VM has less than $AGENT_GUARD_MIN_AVAIL (default
0.25) of its memory available (MemAvailable, or, when that is less, its cgroup's limit less its working set: its use less its
inactive file cache, which the kernel reclaims on demand) it also refuses the lighter ones: a pytest of named files, `lake env`,
the check's suites and `verity check`. The same command through `research run`, or over ssh, runs elsewhere and passes;
`AGENT_GUARD_LOCAL_OK=1` anywhere in the command passes it too.

No tool call may block more than 20 minutes: an agent blocked in one long call looks silent and misses its wake-ups. So it
refuses a command that plainly would: a `sleep` past BLOCK_CAP_S (`sleep 3000`, `sleep 30m`, `sleep 2h`, `sleep infinity`),
there or in `bash -c`, `$(...)` or over ssh, unless a `timeout` around it ends it sooner or it runs in the background (`&`), and
`research run --wait-max S` past the cap. A polling loop is judged only by the sleeps it names.

One agent per checkout (worktree_guard.py): it refuses an agent's branch switch in a worktree another agent owns, and its push of
a branch registered to another agent.

It also carries messages to running agents, the one channel that reaches a subagent mid-run, on any VM (`inbox.py send
AGENT_ID`). A message waits in `$AGENT_INBOX` (default `/tmp/agent-inbox`) as `<conversation_id>.md`. The agent's next
allowed command is refused once, with the message as the refusal's text, which is what Cursor shows the agent (an
`agent_message` on an allowed command reaches no one). The agent then re-runs the command. A refused secret-printing command
keeps the message queued, and the delivered text is appended to `<conversation_id>.delivered`. Messages pushed from other
VMs reach the inbox through `inbox.py pull`, which the hook starts in the background, detached, at most every
$AGENT_INBOX_POLL_S seconds per agent (default 30; `off` stops it), so the hook itself never waits on the network. A message
still queued when the agent's turn ends is delivered by the stop hook (`stop_hook.py`) instead.

Before an allowed `git merge` or `git merge-tree`, it sets the repository's merge drivers (`.gitattributes` names them; git keeps
them only in a clone's config) from origin/main's `research.clone.RECORD_DRIVER`, each whose value differs, as boot.sh and
every run clone do; without origin/main, from the checkout's own when lean-audit's is unset. A VM that never ran boot.sh (a
cloud worker's) otherwise merges each lean-audit.json line by line and shows conflicts the queue never sees: 359 in one trial
merge of a PR onto `next`; and one that kept an older command runs the merged-into branch's own merge.py, which collapsed
Security's lock to one `reads` group when an old branch merged main.
"""
import fnmatch
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from settings import INBOX, INBOX_POLL_S, record_event

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
          "`ps -eo pid,args | grep \"[p]attern\"` (not `grep -v`, which prints every other line) or `pgrep -af PATTERN`.")


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


def strip_redirections(words: list[str], inputs: bool = False) -> list[str]:
    """`words` without redirection operators and their targets (`> f`, `>& 1`, `<<< s`); with `inputs`, the file a command
    reads as its input (`< f`) stays, as an argument."""
    out, skip = [], False
    for w in words:
        if skip:
            skip = False
        elif w and set(w) <= set("&;<>|") and set(w) & set("<>"):
            skip = not (inputs and w == "<")
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


def unwrap(words: list[str], timeouts: list[str] | None = None) -> list[str]:
    """`words` without leading assignments and wrappers (`sudo -u USER`, `timeout 30`, `nice -n 5`, `command`, ...); each
    `timeout`'s duration goes to `timeouts`."""
    while words:
        if words[0] in KEYWORDS or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[0], re.S):
            words = words[1:]
        elif os.path.basename(words[0]) in WRAPPERS:
            wrapper, words = os.path.basename(words[0]), words[1:]
            while words and words[0].startswith("-"):
                words = words[2:] if words[0] in WRAPPER_ARG_OPTIONS.get(wrapper, ()) else words[1:]
            if wrapper == "timeout":
                if timeouts is not None:
                    timeouts.extend(words[:1])
                words = words[1:]
        else:
            return words
    return words


def assigned(words: list[str], name: str) -> str:
    """The value `words` assigns to `name` ahead of the command they run (`NAME=VALUE cmd ...`), or ""."""
    for w in words[:len(words) - len(unwrap(words))]:
        if w.startswith(name + "="):
            return w[len(name) + 1:]
    return ""


def one_pid(args: list[str]) -> bool:
    """Whether `ps ARGS` selects exactly one process by pid (`-p PID`, `--pid PID`) and nothing else."""
    sel = [(a, args[i + 1]) for i, a in enumerate(args[:-1]) if a in ("-p", "--pid", "-q", "--quick-pid")]
    others = any(a in ("--ppid", "-C", "-u", "-U", "-g", "-G", "-t", "--user", "--group", "--sid", "-s") or a.startswith("--ppid=")
                 for a in args)
    return len(sel) == 1 and re.fullmatch(r"\d+", sel[0][1]) is not None and not others


def every_argv(args: list[str]) -> bool:
    """Whether `ps ARGS` prints full command lines of more than one named process: of every process, or of a selection such as
    `--ppid N`, `-C NAME` or `-u USER` (`ps -o pid,etime,cmd --ppid N` prints a git child's token-bearing URL)."""
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


def ssh_command(args: list[str]) -> str:
    """What `ssh ARGS` runs on the far side ("" for a login shell)."""
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i][-1] in SSH_ARG_OPTIONS and len(args[i]) == 2 else 1
    return " ".join(args[i + 1:])


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
        return refused_text(ssh_command(args))
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


def inverted(words: list[str]) -> bool:
    """Whether a `grep`/`rg` prints the lines that don't match: `-v`, alone or in a cluster such as `-iv`, or `--invert-match`,
    which GNU grep takes from `--inv` on."""
    return any(a.startswith("--inv") and "--invert-match".startswith(a) or re.fullmatch(r"-\w*v\w*", a)
               for a in unwrap(strip_redirections(words))[1:])


def filtered(commands: list[tuple[list[str], str]], i: int) -> bool:
    """Whether command `i` pipes straight into a `grep`/`rg` that keeps only matching lines, read past inverted ones
    (`ps -ef | grep -v grep | grep NAME`)."""
    while commands[i][1] == "|" and i + 1 < len(commands) and filter_cmd(commands[i + 1][0]):
        i += 1
        if not inverted(commands[i][0]):
            return True
    return False


def refused_text(text: str) -> bool:
    if ENVIRON.search(text):
        return True
    try:
        commands = words_of(text)
        return any(refused(w, filtered(commands, i)) for i, (w, _) in enumerate(commands))
    except ValueError:                                   # unbalanced quotes: judge the raw text
        return bool(FALLBACK.search(text))


#: commands that write what they read to their output
READERS = {"cat", "head", "tail", "less", "more", "tac", "nl", "sed", "awk", "cut", "xxd", "od", "strings", "base64"} | FILTERS
#: readers whose first argument is a pattern or a script, not a file, unless one of SCRIPT_OPTIONS gives it
SCRIPTED = {"sed", "awk"} | FILTERS
SCRIPT_OPTIONS = ("-e", "-f", "--regexp", "--file", "--expression")
QUIET_GREP = {"--files-with-matches", "--files-without-match", "--count", "--count-matches", "--quiet", "--silent", "--files"}
#: what a file's name, written out or as a glob, says it holds
SECRET_FILES = {"a private key": ("*.pem", "*.key", "id_rsa*", "id_ed25519*", "id_ecdsa*"),
                "an env or credentials file": ("*.env", ".netrc", "credentials", "credentials.*")}
#: names that hold no secret: a key's public half, ssh's public files, a template, a module or a doc (`credentials.py`)
PUBLIC_FILES = ("*.pub", "*-pub.pem", "authorized_keys*", "known_hosts*", "*.example", "*.template", "*.py", "*.md")
#: directories where every file counts, so a glob over one does (`/root/.config/verity/*` matches the key beside the env files)
SECRET_DIRS = re.compile(r"(^|/)(\.ssh|\.config/verity|secrets)(/|$)")
#: what a reader's line names when the parse itself fails: cruder, and it errs toward refusing
SECRET_FALLBACK = re.compile(r"\b(%s)\b[^|;&\n]*?([^\s'\"]*(\.pem|\.key|\.env|\.netrc|credentials|id_rsa|id_ed25519|id_ecdsa|"
                             r"\.ssh/|\.config/verity/|secrets/)[^\s'\"]*)" % "|".join(sorted(READERS)))
ENVNAMES = "tools/agent-guard/envnames.py"


def secret_file(path: str) -> str | None:
    """What the file name or glob `path` says it holds, when that is secret: a private key, an env or credentials file, or a
    file of a secrets directory (`.ssh/`, `.config/verity/` under any home, a `secrets/`). None for PUBLIC_FILES, `.ssh/config`
    and the fixtures under a `tests/` directory."""
    parts = path.rstrip("/").split("/")
    if any(fnmatch.fnmatchcase(parts[-1], p) for p in PUBLIC_FILES) or "tests" in parts[:-1] or parts[-2:] == [".ssh", "config"]:
        return None
    for kind, patterns in SECRET_FILES.items():
        if any(fnmatch.fnmatchcase(parts[-1], p) for p in patterns):
            return kind
    return "a file of a secrets directory" if SECRET_DIRS.search(path) else None


def prints_no_line(reader: str, args: list[str]) -> bool:
    """Whether `reader ARGS` shows no line of what it reads: a grep that lists files, counts or only tests (`-l`, `-L`, `-c`,
    `-q`, alone or in a cluster such as `-rl`), or a sed that edits in place (`-i`)."""
    if reader == "sed":
        return any(a.startswith(("-i", "--in-place")) for a in args)
    short = "lcq" if reader == "rg" else "lcqL"          # rg's -L follows symlinks
    return reader in FILTERS and any(a in QUIET_GREP or re.fullmatch(rf"-[a-dg-zA-Z]*[{short}][a-zA-Z]*", a) for a in args)


def files_read(reader: str, args: list[str]) -> list[str]:
    """The files `reader ARGS` names: its arguments that are not options, less a grep's, sed's or awk's pattern or script (the
    word after each of SCRIPT_OPTIONS, else the first)."""
    if reader not in SCRIPTED:
        return [a for a in args if not a.startswith("-")]
    files = [a for i, a in enumerate(args) if not a.startswith("-") and not (i and args[i - 1] in SCRIPT_OPTIONS)]
    return files if any(a.startswith(SCRIPT_OPTIONS) for a in args) else files[1:]


def secret_read(text: str) -> tuple[str, str, str] | None:
    """(reader, file, what the file holds) of the first command in `text` that would print a secret file, or None. In each simple
    command the first of READERS, wherever it stands (after `sudo`, as ssh's remote command, after `research pods ssh POD --`,
    as `find`'s `-exec`), is judged on the files it names, its input (`< FILE`) and the variable of a `for` loop over such
    files; the quoted strings before it are read as commands in turn, which is how a wrapper script's argument and `bash -c`'s
    are seen. `git`'s arguments and those of a `research` command that runs nothing (no `--`) are prose and are not read."""
    try:
        commands = [words for words, _ in words_of(text)]
    except ValueError:                                   # unbalanced quotes: the words between the shell's operators, unquoted
        commands = [re.findall(r"[<>]+|[^\s'\"<>]+", part) for part in re.split(r"[|;&()`\n]", text)]
    loops = {m[1]: next((w for w in re.split(r"[\s'\"`$()]+", m[2]) if secret_file(w)), "")
             for m in re.finditer(r"\bfor\s+(\w+)\s+in\s([^;\n]*)", text)}
    for words in commands:
        words = strip_redirections(words, inputs=True)
        run = unwrap(words)
        prog = os.path.basename(run[0]) if run else ""
        if prog == "git" or (prog == "research" and "--" not in run):
            continue
        at = next((i for i, w in enumerate(words) if os.path.basename(w) in READERS), len(words))
        reader, args = os.path.basename(words[at]) if at < len(words) else "", words[at + 1:]
        if reader and not prints_no_line(reader, args):
            for a in files_read(reader, args) + (words[:at] if "{}" in args else []):
                var = re.fullmatch(r"\$\{?(\w+)\}?", a)
                a = loops.get(var[1], "") if var else a
                if kind := secret_file(a):
                    return reader, a, kind
        for w in words[:at]:
            if re.search(r"\s", w) and (found := secret_read(w)):
                return found
    return None


def secret_file_reason(command: str) -> str | None:
    """Why this command may not run (it would print a key, an env or credentials file, or a file of a secrets directory), or
    None. AGENT_GUARD_LOCAL_OK=1 does not lift it: that opt-in is the local-work gate's alone."""
    try:
        found = secret_read(command)
    except (ValueError, IndexError, KeyError, TypeError, AttributeError, RecursionError):   # a bug here must not allow what the text shows
        m = SECRET_FALLBACK.search(command)
        found = (m[1], m[2], "a secret file") if m else None
    if found is None:
        return None
    reader, path, kind = found
    return (f"this command would print {kind}: `{reader}` reads `{path}`. A filter after it (`cut -d= -f1`, `sed 's/=.*//'`) "
            "still prints a key file's lines and a multi-line value's later ones, which hold no `=`. List an env file's variable "
            f"names with `python3 {ENVNAMES} FILE...` (on another machine: `ssh HOST python3 - FILE... < {ENVNAMES}`), which "
            "prints nothing of a file that isn't one; look at a key with `stat`, `sha256sum` or `openssl pkey -in FILE -pubout`. "
            "A command that needs the contents takes the file itself (`. FILE`, `CMD < FILE`, `cp`). "
            f"{LOCAL_OK} does not lift this.")


ALWAYS, PRESSURE = 2, 1
#: program -> {first argument: weight} (None: any): ALWAYS is refused locally, PRESSURE only while memory is short
HEAVY_PROGRAMS = {"nvcc": {None: ALWAYS}, "suites.py": {None: PRESSURE}, "check.py": {None: PRESSURE},
                  "lake": {"build": ALWAYS, "exe": ALWAYS, "env": PRESSURE}, "verity": {"check": PRESSURE},
                  "cargo": {"build": ALWAYS, "test": ALWAYS, "run": ALWAYS, "check": ALWAYS}}
UV_RUN_VALUED = {"--extra", "--with", "--package", "-p", "--python", "--directory", "--project", "--group", "--env-file", "--index"}
#: the opt-in of the local-work gate, and of nothing else: no refusal of a command that prints secrets reads it
LOCAL_OK = "AGENT_GUARD_LOCAL_OK=1"


def weight_words(words: list[str]) -> int:
    """How heavy one simple command is here (0, PRESSURE or ALWAYS), seen through `uv run [OPTS]`, `python -m MOD` and `python
    SCRIPT`. A pytest naming test files or a `-k` selection is PRESSURE, any other ALWAYS; `circuit-check --all` is ALWAYS."""
    words = unwrap(strip_redirections(words))
    while words and re.fullmatch(r"\w+=.*", words[0]):
        words = words[1:]
    if not words:
        return 0
    prog, args = os.path.basename(words[0]), words[1:]
    if prog == "uv" and args[:1] == ["run"]:
        rest, i = args[1:], 0
        while i < len(rest) and rest[i].startswith("-"):
            i += 2 if rest[i] in UV_RUN_VALUED else 1
        return weight_words(rest[i:])
    if re.fullmatch(r"python[\d.]*", prog):
        if args[:1] == ["-m"]:
            return weight_words(args[1:])
        script = next((a for a in args if not a.startswith("-")), None)
        return weight_words([script, *args[args.index(script) + 1:]]) if script is not None else 0
    if prog in ("pytest", "py.test"):
        named = any(a.endswith(".py") or "::" in a for a in args) or "-k" in args
        return PRESSURE if named else ALWAYS
    if prog == "circuit-check":
        return ALWAYS if "--all" in args else PRESSURE
    table = HEAVY_PROGRAMS.get(prog) or {}
    return table.get(None) or table.get(args[0] if args else "", 0)


def weight(text: str) -> int:
    """The heaviest simple command of `text` run on this VM: a command through `research run` or over ssh runs elsewhere."""
    try:
        commands = words_of(text)
    except ValueError:
        return 0
    return max((weight_words(words) for words, _ in commands), default=0)


def available() -> tuple[float, float] | None:
    """(available, total) bytes of this VM's memory: /proc/meminfo's MemAvailable and MemTotal, plus what a hypervisor balloon holds
    (/proc/vmstat beside it), or, where this process's cgroup's limit is tighter, that limit and the lesser of that and what the
    limit leaves over the cgroup's working set. The working set is memory.current less memory.stat's inactive_file, the file cache
    the kernel reclaims on demand, as container runtimes count it; all of memory.current where memory.stat can't be read."""
    meminfo = Path(os.environ.get("AGENT_GUARD_MEMINFO", "/proc/meminfo"))
    try:
        info = dict(re.findall(r"^(\w+):\s+(\d+) kB", meminfo.read_text(), re.M))
        avail, total = int(info["MemAvailable"]) * 1024, int(info["MemTotal"]) * 1024
    except (OSError, KeyError, ValueError):
        return None
    # the balloon's pages are counted as used, but the hypervisor hands them back as the VM asks for memory
    if (vmstat := meminfo.with_name("vmstat")).is_file():
        v = dict(re.findall(r"^(balloon_inflate|balloon_deflate) (\d+)$", vmstat.read_text(), re.M))
        avail = min(total, avail + max(0, int(v.get("balloon_inflate", 0)) - int(v.get("balloon_deflate", 0))) * 4096)
    try:
        rel = open("/proc/self/cgroup").read().split("::", 1)[1].strip()
        cg = os.environ.get("AGENT_GUARD_CGROUP") or f"/sys/fs/cgroup{rel}"
        limit = open(f"{cg}/memory.max").read().strip()
        if limit != "max" and int(limit) < total:
            meminfo_avail, used, total = avail, int(open(f"{cg}/memory.current").read()), int(limit)
            avail = min(meminfo_avail, total - used)        # stands when memory.stat can't be read
            cache = int(dict(re.findall(r"^(\w+) (\d+)$", open(f"{cg}/memory.stat").read(), re.M)).get("inactive_file", 0))
            avail = min(meminfo_avail, total - max(0, used - cache))
    except (OSError, IndexError, ValueError):
        pass
    return avail, total


#: a command this hook would read past its first part: on a node it runs whole only inside one `bash -lc`
COMPOUND = re.compile(r"&&|\|\||;|\|")


def local_work_reason(command: str) -> str | None:
    """Why this command may not run on this VM, or None."""
    w = 0 if LOCAL_OK in command else weight(command)
    cmd = f"bash -lc {shlex.quote(command)}" if COMPOUND.search(command) else "CMD"
    remote = (f"Run it on a node: `research run --on auto -- {cmd}` (placed in a CPU node's research pool from this checkout: 4 CPUs, 16 GB "
              f"and 60 min unless --cpus, --mem-gb or --timeout say otherwise; add --wait to block until it ends, for at most 20 "
              f"minutes). For a small local run "
              f"you know fits, put {LOCAL_OK} in the command.")
    if w == ALWAYS:
        return ("builds and tests run on our nodes, not on an agent VM (every handle, at Daniel's request, 18 Oct: OOM kills on "
                "shared VMs took down every worker on them). A pytest of named test files or a `-k` selection, or `lake env lean` "
                "on one file, still runs here while memory allows. " + remote)
    if w != PRESSURE:
        return None
    mem = available()
    floor = float(os.environ.get("AGENT_GUARD_MIN_AVAIL", "0.25"))
    if mem is None or mem[0] >= floor * mem[1]:
        return None
    return (f"this VM has {mem[0] / 2**30:.1f} GiB of {mem[1] / 2**30:.1f} GiB available (under {floor:.0%}), and this command runs "
            f"tests or a build here: on a VM other agents share, the kernel would OOM-kill some agent's processes. " + remote)


#: no tool call may block longer: an agent blocked in one long call looks silent and misses its wake-ups
BLOCK_CAP_S = 1200
DURATION = re.compile(r"(\d+(?:\.\d*)?|\.\d+)([smhd]?)")
UNIT_S = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}


def duration(args: list[str]) -> float | None:
    """Seconds `sleep ARGS` lasts (GNU sleep sums its arguments; `timeout`'s DURATION reads the same), or None when an argument
    isn't a literal one (`$N`)."""
    total = 0.0
    for a in args:
        if a in ("inf", "infinity"):
            return math.inf
        if not (m := DURATION.fullmatch(a)):
            return None
        total += float(m[1]) * UNIT_S[m[2]]
    return total if args else None


def blocks_words(words: list[str]) -> float:
    """Seconds one simple command plainly blocks the shell for: a `sleep`, `research run --wait-max S`, or what `bash -c` or ssh
    runs, each bounded by a `timeout` around it; 0 when it states no literal time."""
    s = max((blocks_text(w.split("$(", 1)[1]) for w in words if "$(" in w), default=0.0)
    timeouts: list[str] = []
    words = unwrap(strip_redirections(words), timeouts)
    bound = min((d for t in timeouts if (d := duration([t])) is not None), default=math.inf)
    name, args = (os.path.basename(words[0]), words[1:]) if words else ("", [])
    if name == "sleep":
        s = max(s, duration(args) or 0.0)
    elif name in SHELLS and "-c" in args[:-1]:
        s = max(s, blocks_text(args[args.index("-c") + 1]))
    elif name == "ssh":
        s = max(s, blocks_text(ssh_command(args)))
    elif name == "research" and args[:1] == ["run"]:
        opts = args[:args.index("--")] if "--" in args else args
        given = [opts[i + 1] for i, a in enumerate(opts[:-1]) if a == "--wait-max"]
        given += [a.split("=", 1)[1] for a in opts if a.startswith("--wait-max=")]
        s = max([s, *(float(g) for g in given if DURATION.fullmatch(g) and not g[-1].isalpha())])
    return min(s, bound)


def blocks_text(text: str) -> float:
    """The longest any simple command of `text` plainly blocks for; one sent to the background (`&`) blocks nothing."""
    try:
        commands = words_of(text)
    except ValueError:
        return 0.0
    return max((blocks_words(words) for words, op in commands if op != "&"), default=0.0)


def block_reason(command: str) -> str | None:
    """Why this command may not run as one tool call (it would block past BLOCK_CAP_S), or None."""
    s = blocks_text(command)
    if s <= BLOCK_CAP_S:
        return None
    return (f"this command would block for {'ever' if s == math.inf else f'{s:g} s'}, past {BLOCK_CAP_S} s: no tool call may block "
            "more than 20 minutes (Daniel, 19 Oct); end your turn and let the run's completion or a timer subscription wake you. "
            f"`research run --wait` stops after {BLOCK_CAP_S} s on its own and names the command to check back with.")


def checkout_reason(command: str, conversation: str, cwd: str) -> str | None:
    """One agent per checkout (worktree_guard.py): why this command's git may not run here, or None."""
    sys.dont_write_bytecode = True                      # no __pycache__ beside the hook: boot.sh copies its directory file by file
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import worktree_guard

    try:
        raw = [strip_redirections(words) for words, _ in words_of(command)]
    except ValueError:
        return None
    next_ok = [assigned(w, "AGENT_GUARD_PUSH_NEXT_OK") for w in raw]
    return worktree_guard.reason([unwrap(w) for w in raw], conversation, cwd, next_ok)


def delivery(conversation: str) -> str | None:
    """The message queued for agent `conversation` (`inbox.py send`), taken out of its inbox, or None."""
    if not conversation or "/" in conversation or conversation.startswith("."):
        return None
    box = os.path.join(INBOX, conversation + ".md")
    taking = box + ".taking"
    try:
        os.replace(box, taking)
    except OSError:
        return None
    try:
        with open(taking) as f:
            text = f.read()
        with open(os.path.join(INBOX, conversation + ".delivered"), "a") as log:
            log.write(text)
    finally:
        os.remove(taking)
    return text if text.strip() else None


def checkout(root: str, cwd: str) -> str:
    """The checkout an inbox pull runs from: the workspace root, or when the root isn't one (a Task-launched cloud agent's
    `/agent`), the first checkout under its `repos/`, else the command's directory."""
    if root and not os.path.exists(os.path.join(root, ".git")):
        found = sorted(Path(root).glob("repos/*/.git"))
        root = str(found[0].parent) if found else ""
    return root or cwd


def poll(conversation: str, repo: str) -> bool:
    """Start `inbox.py pull` for agent `conversation` from checkout `repo`, detached, unless one started within the poll
    interval; whether it started one."""
    every = INBOX_POLL_S
    if every == "off" or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", conversation) or not repo:
        return False
    os.makedirs(INBOX, exist_ok=True)
    mark = os.path.join(INBOX, conversation + ".polled")
    if os.path.exists(mark) and time.time() - os.stat(mark).st_mtime < float(every):
        return False
    with open(mark, "a"):
        os.utime(mark)
    with open(os.path.join(INBOX, conversation + ".pull.log"), "a") as log:
        subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "inbox.py"), "pull",
                          conversation, "--repo", repo], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                         env={**os.environ, "AGENT_INBOX": INBOX}, start_new_session=True)
    return True


MERGE = re.compile(r"\bgit\b.*\bmerge(-tree)?\b")
CLONE_PY = "tools/research/src/research/clone.py"


def merge_drivers(command: str, cwd: str) -> None:
    """Set the record merge drivers in ``cwd``'s repository before a merge there: origin/main's, each that differs, else the
    checkout's when lean-audit's is unset (module doc)."""
    if not MERGE.search(command):
        return
    import importlib.util
    import subprocess
    import tempfile

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True, timeout=3)

    if not (top := git("rev-parse", "--show-toplevel").stdout.strip()):
        return
    main = git("show", f"origin/main:{CLONE_PY}")
    if main.returncode and git("config", "--get", "merge.lean-audit.driver").stdout.strip():
        return
    with tempfile.TemporaryDirectory(prefix="agent-guard-clone-") as tmp:
        path = os.path.join(top, CLONE_PY)
        if not main.returncode:
            with open(path := os.path.join(tmp, "clone.py"), "w") as f:
                f.write(main.stdout)
        spec = importlib.util.spec_from_file_location("agent_guard_clone", path)
        if spec is None or spec.loader is None:
            return
        clone = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(clone)
    have = dict(e.split("\n", 1) for e in git("config", "-z", "--get-regexp", r"^merge\.").stdout.split("\0") if "\n" in e)
    for k, v in clone.RECORD_DRIVER.items():
        if have.get(k) != v:
            git("config", k, v)


def main() -> int:
    raw = sys.stdin.read()
    try:
        event = json.loads(raw)
        shell = event.get("tool_input") if isinstance(event.get("tool_input"), dict) else {}
        command = event.get("command") or shell.get("command") or ""
        conversation = str(event.get("conversation_id") or "")
        root = next(iter(event.get("workspace_roots") or []), "")
        cwd = event.get("cwd") or shell.get("cwd") or root or os.getcwd()
        record_event("shell_hook.py", event)
    except (ValueError, AttributeError):
        record_event("shell_hook.py", raw)
        command, conversation, cwd, root = raw, "", os.getcwd(), ""
    try:
        deny = refused_text(command)
    except Exception:                                   # a bug here must not allow what the fallback would refuse
        deny = bool(FALLBACK.search(command) or ENVIRON.search(command))
    try:
        low = None if deny else block_reason(command)
    except Exception:                                   # a bug in the block gate must not stop every command
        low = None
    try:
        low = low or (None if deny else local_work_reason(command))
    except Exception:                                   # a bug in the local-work gate must not stop every command
        low = None
    try:
        low = low or (None if deny or ("git" not in command and "worktree_guard" not in command)
                      else checkout_reason(command, conversation, cwd))
    except Exception:                                   # nor one in the checkout guard
        pass
    low = secret_file_reason(command) or low            # ahead of the gates' reasons: the one about a secret is the one to act on
    if deny or low:
        reason = REASON if deny else low
        print(json.dumps({"permission": "deny",
                          "user_message": "Refused by tools/agent-guard/shell_hook.py: " + reason,
                          "agent_message": "Refused by tools/agent-guard/shell_hook.py (.cursor/hooks.json): " + reason}))
    else:
        try:
            poll(conversation, checkout(root, cwd))
        except (OSError, ValueError) as e:              # nor a failed poll: the next command's hook tries again
            print(f"shell_hook: inbox poll failed: {e}", file=sys.stderr)
        try:
            message = delivery(conversation)
        except Exception:                               # an unreadable inbox must not block the agent's command
            message = None
        try:
            merge_drivers(command, cwd)
        except (OSError, ImportError, SyntaxError, AttributeError,
                subprocess.SubprocessError) as e:       # nor a merge driver git couldn't set
            print(f"shell_hook: merge drivers not set in {cwd}: {type(e).__name__}: {e}", file=sys.stderr)
        if message:
            note = ("MESSAGE FROM YOUR COORDINATOR, delivered by the shell hook (tools/agent-guard/inbox.py). This command was "
                    "refused only to deliver it: nothing is wrong with it, so re-run it exactly as it was, then act on the "
                    "message.\n\n" + message)
            print(json.dumps({"permission": "deny", "user_message": note, "agent_message": note}))
        else:
            print(json.dumps({"permission": "allow"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
