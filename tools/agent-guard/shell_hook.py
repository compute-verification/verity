#!/usr/bin/env python3
"""Cursor's `beforeShellExecution` hook (`.cursor/hooks.json`): refuses shell commands that would print secrets wholesale.

Cursor runs every agent shell command through it, on any VM, so it holds where the redacting `env` (`install.sh`) was never
installed. A multi-line secret defeats `env | cut -d= -f1`: its later lines print as if they were names, and so reach an
agent's transcript. Refused, anywhere in the command (pipelines,
`$(...)`, what a carrier runs (`bash -c`, tmux; below), `ssh HOST CMD`, `research pods ssh POD -- CMD`): `env` or `printenv` with nothing to run, `printenv` of a
secret's name, a bare `set`, `export` or `export -p`, `declare`/`typeset` listing variables, reading `/proc/*/environ`, and
`tmux show-environment` (`showenv`) with no name or a secret's (tmux's global environment through `cut -d= -f1` prints a
multi-line key's lines too).
`compgen -e` lists names only and passes.

Also refused, run on this machine: every listing that prints a process's full command line, whatever it is piped into, since a
daemon's or a git child's token sits in its argv and a filter for the daemon's name keeps exactly the lines that hold it
(`argv_listing`): `ps` with a format naming `args`, `cmd`, `command` or `%a` (`-o`, `-O`, `o`, `--format`, each one read, as in
`ps -o pid= -o args=`), with `-f` or `-F` in any bundle (`-efww`), or in BSD form without `c` or a format (`ps aux`, `ps ax`, `ps
x`, a bare pid), and `ps e`, which adds each process's environment; `PS_FORMAT` naming one of those or `PS_PERSONALITY` set at
all; `pgrep -a` or `--list-full`; `top -c`, `htop` and `btop`; `pstree -a`; and reading a `/proc/PID/cmdline` (`cat`, `tr '\\0'
' ' < FILE`, `xargs -0 < FILE`, a `$P/cmdline`, a `cd /proc/PID` then `cmdline`, psutil's `cmdline()`). What says whether a
process runs and since when, with no arguments, passes: `ps -eo pid,etimes,comm`, `pgrep -l PATTERN`, and `procs.py` beside this
hook, which prints each command line with every token-like run cut out. A listing that another machine runs (`ssh HOST CMD`,
`research pods ssh POD -- CMD`) prints that machine's argv and is judged as it always was (`every_argv`): refused when it prints
the full command lines of every process or of a selection beyond one pid, unless piped straight into a `grep`/`rg` that keeps
only matching lines (an inverted one, `grep -v`, prints the rest); `/proc/*/cmdline` is refused there too.

Every rule here (the environment, listings, secret files, builds, pushes and branch switches) reads, beside the command as
typed, what it has another program run (`carried`, `flattened`): what tmux types into a pane or starts (`send-keys ... C-m`,
`new-session`, `new-window`, `split-window`, `respawn-pane`, `run-shell`, `if-shell`, `pipe-pane`, `display-popup` and tmux's
own `-c`, by name, alias or prefix, with any `-f CONFIG`, `-L`, `-S` and `-t TARGET`, and after `\\;`), `bash|sh|zsh|dash -c`
in any bundle (`-lc`), `env [VAR=VALUE...] CMD`, `xargs CMD`, `eval` and `watch`, as it reads through `nohup`, `setsid`,
`timeout`, `sudo`, `nice` and the like (`unwrap`). Not read: a script file's contents (`bash FILE`, `tmux source-file`), a
heredoc fed to a shell, `find -exec`, `su -c`, `flock`, `script -c` and `parallel`. A pane's command blocks no tool call, so
the block rule leaves tmux alone.

Reads Cursor's hook JSON on stdin, `beforeShellExecution`'s (`command`, `cwd`) or a `preToolUse` event's for the Shell tool
(`tool_input.command`), and prints its verdict as JSON on stdout; each event's key names go to $AGENT_GUARD_EVENT_LOG
(`settings.record_event`).

A secret file is refused the same way: a command that writes what it reads to its output (READERS: `cat`, `head`, `tail`,
`sed`, `awk`, `cut`, `sort`, `diff`, `xxd`, `base64` and the like, `zcat` and its kin, and a `grep`/`rg` that prints lines, one
without `-l`, `-L`, `-c` or `-q`) applied to a private key (`*.pem`, `*.key`, `id_rsa*`, `id_ed25519*`, `id_ecdsa*`), an env or
credentials file (`*.env`, `.netrc`, `credentials`, `credentials.*`), a compressed or archived copy of one (`x.env.gz`) or
anything in a secrets directory (`.ssh/`, `.config/verity/`, a `secrets/`), named as an argument, a glob, its input
(`< FILE`), a `for` loop's variable or, behind `xargs`, by the commands piped into it, wherever the command sits: over ssh,
after `research pods ssh POD --`, in a quoted string handed to a wrapper script. Every word that names such a command is
judged, so a host, an image or a directory of that name hides no reader after it. So are `tr` and `tee` with such a file as
their input, `dd if=FILE` with no `of=`, the shell's own reads (`while read ...; done < FILE`, `read KEY < FILE`, `$(<FILE)`),
the code of `python -c`, `perl -e` or `ruby -e` that names such a file and opens, reads or prints, `openssl enc -in FILE` with
no `-out`, and `openssl pkey|rsa|ec|dsa|pkcs8 -in KEY` with neither `-pubout` nor `-noout`, or with `-text`, which writes the
private key out. An archiver or a compressor that writes to its output is such a command too: `tar -c` with `-f -` or no `-f`
(only a terminal stops it, and an agent's shell has none), `tar -x` with `-O`, `zip - FILE`, `unzip -p`, `cpio -o` and `gzip
-c`, `bzip2 -c`, `xz -c`, `zstd -c` and their kin. One pipe passes, the copy between machines: `tar -cf - FILE... | ssh HOST tar
-xf - -C DIR`, where the archive goes straight into a `tar` that extracts from its input to disk, each end run here or by
`ssh HOST` or `research pods ssh POD --`, the far one beside nothing but `umask`, `mkdir`, `chmod`, `chown` and `cd`. That list
is closed: any other consumer (`cat`, `base64`, `gzip`, a redirect, a subshell) may write the archive on. A glob over a secrets
directory matches the key files beside the env files, and a filter that cuts each line at its `=` prints every line of a key.
`envnames.py FILE...` lists an env file's variable names instead and prints nothing of a file that isn't one. What shows no
contents passes (`ls`, `stat`, `wc`, `sha256sum`, `cp`, writing or sourcing the file, `sed -i`, `diff -q`, `cmp`, `tar -cf
OUT.tar`, `gzip -k`, `openssl pkey -pubout`, `openssl x509`, `ssh-keygen -l`), as do public halves (`*.pub`, `*-pub.pem`,
`authorized_keys`, `known_hosts`, `.ssh/config`), `*.example` and `*.template` files, a module or a doc (`credentials.py`), the
fixtures under a `tests/` directory, and a file git tracks in the checkout, unchanged, read there by the command itself: every
agent can read it already, while over ssh the hook can't know. `AGENT_GUARD_LOCAL_OK=1` does not lift it.

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
a branch registered to another agent or of `main`, `checked` or `next`, to the repositories whose queue drive runs
(`compute-verification/verity` and `compute-verification/infra`) and no other: the notes' `main` is every agent's to push.
It places a git command by the event's cwd and its own `-C` alone: a `cd` anywhere in the command (`flattened` holds what
carriers run too) or a `GIT_DIR` the command sets (`git_env`) leaves it unplaced, and an unplaced push is judged as theirs.

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
import glob
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
LISTING_FALLBACK = re.compile(r"(^|[\s|;&(`'\"])(ps\s+[^|;&\n]*(\b(args|cmd|command)\b|-\w*[fF]|(^|\s)[a-zA-Z]+(\s|$))|"
                              r"pgrep\s+[^|;&\n]*(-\w*a|--list-f)|htop|btop)|cmdline", re.M)
HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2[ \t]*(?=\n|$)", re.S)

#: what a refused command would print: `refused_text`'s verdicts, each with its refusal
ENVIRONMENT, ARGV = "environment", "argv"
PROCS = "tools/agent-guard/procs.py"
REASONS = {
    ENVIRONMENT: ("this command would print environment variables' values (a multi-line secret's later lines print even through "
                  "`cut -d= -f1`). List variable names with `compgen -e` (filter with `compgen -e | rg NAME`); check one "
                  "variable with `[ -n \"${NAME:-}\" ] && echo set`."),
    ARGV: ("this command would print processes' full command lines, and a daemon's token sits in its argv: a filter for the "
           "daemon's name keeps exactly the lines that hold it. Whether a process runs, and since when: `ps -eo pid,etimes,comm` "
           f"or `pgrep -l PATTERN`. Its arguments with every token-like run cut out: `python3 {PROCS} [PATTERN]` (on another "
           f"machine: `ssh HOST python3 - [PATTERN] < {PROCS}`)."),
}
#: `ps` formats that print the command line, and options of `ps`, `pgrep`, `top` and `pstree` that take a value (the rest
#: of the word, else the next word)
SHOWS_ARGV = re.compile(r"\b(args|cmd|command)\b|%a")
PS_VALUED = {"unix": set("CGgOopqstUu"), "bsd": set("OoptUk"),
             "long": {"--format", "--sort", "--pid", "--ppid", "--quick-pid", "--user", "--User", "--group", "--Group", "--sid",
                      "--tty", "--cols", "--columns", "--width", "--rows", "--lines"}}
LISTER_VALUED = {"pgrep": set("dgGPstuUFr"), "top": set("deEnopuU"), "pstree": set("HNC")}
#: reading a process's command line from /proc, in a word (a path, a variable's, Python code) or by psutil
PROC_ARGV = re.compile(r"/proc/\S*cmdline|\$\{?\w+\}?/cmdline\b|\.cmdline\(|psutil\S*(cmdline|environ|as_dict)")


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


def ps_prints(args: list[str], ps_format: str = "") -> str:
    """What `ps ARGS` prints beyond each process's name: ENVIRONMENT (BSD `e`), ARGV (its command line) or "". It prints the
    command line under a format naming it (every `-o`, `-O`, `o`, `O` and `--format` is read, and `ps_format`, the PS_FORMAT it
    runs under), with `-f` or `-F`, and in BSD form (any option without a dash, or a bare pid) unless `c` or a format is given."""
    formats, unix, bsd, i = [ps_format] if ps_format else [], set(), set(), 0
    while i < len(args):
        a, i = args[i], i + 1
        if a.startswith("--"):
            name, eq, value = a.partition("=")
            if name == "--format":
                formats.append(value if eq else args[i] if i < len(args) else "")
            i += name in PS_VALUED["long"] and not eq
        elif a.startswith("-") and len(a) > 1 or not re.fullmatch(r"[\d,]+", a):
            dash = a.startswith("-")
            letters, valued = (a[1:], PS_VALUED["unix"]) if dash else (a, PS_VALUED["bsd"])
            for at, letter in enumerate(letters):
                (unix if dash else bsd).add(letter)
                if letter in valued:
                    value = letters[at + 1:] or (args[i] if i < len(args) else "")
                    i += not letters[at + 1:]
                    if letter in "oO":
                        formats.append(value)
                    break
        else:
            bsd.add("p")
    if "e" in bsd:
        return ENVIRONMENT
    if formats:
        return ARGV if any(SHOWS_ARGV.search(f) for f in formats) or unix & {"f", "F"} else ""
    return ARGV if unix & {"f", "F"} or (bsd and "c" not in bsd) else ""


def letters_before_value(args: list[str], valued: set[str]) -> set[str]:
    """The single-letter options `ARGS` gives a program whose options in `valued` take a value."""
    out, skip = set(), False
    for a in args:
        if skip:
            skip = False
        elif re.fullmatch(r"-[^-].*", a):
            for at, letter in enumerate(a[1:], 2):
                out.add(letter)
                if letter in valued:
                    skip = at == len(a)
                    break
    return out


def argv_listing(words: list[str]) -> str:
    """What the simple command `words` (unwrapped) prints of processes beyond their names, run on this machine: ENVIRONMENT,
    ARGV or "" (module doc)."""
    if not words:
        return ""
    name, args = os.path.basename(words[0]), words[1:]
    if name == "ps":
        return ps_prints(args)
    if name == "pgrep":
        full = any(a.startswith("--list-f") and "--list-full".startswith(a) for a in args)
        return ARGV if full or "a" in letters_before_value(args, LISTER_VALUED["pgrep"]) else ""
    if name == "top":
        return ARGV if "c" in letters_before_value(args, LISTER_VALUED["top"]) else ""
    if name == "pstree":
        return ARGV if "--arguments" in args or "a" in letters_before_value(args, LISTER_VALUED["pstree"]) else ""
    return ARGV if name in ("htop", "btop") else ""


def ps_variables(words: list[str]) -> str:
    """ARGV when `words` set PS_FORMAT to a format that prints the command line, or PS_PERSONALITY at all (as an assignment
    ahead of the command or by `export`), which change what a later `ps` prints; else ""."""
    run = unwrap(words)
    sets = words[:len(words) - len(run)] + (run[1:] if run[:1] == ["export"] else [])
    for w in sets:
        name, _, value = w.partition("=")
        if name == "PS_PERSONALITY" or name == "PS_FORMAT" and SHOWS_ARGV.search(value):
            return ARGV
    return ""


def ssh_words(args: list[str]) -> list[str]:
    """The words of what `ssh ARGS` runs on the far side (none for a login shell)."""
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i][-1] in SSH_ARG_OPTIONS and len(args[i]) == 2 else 1
    return args[i + 1:]


#: tmux commands that run a shell command -> (the shortest prefix tmux takes for it, its options that take a value, the option
#: naming the directory the command runs in)
TMUX_CARRIERS = {"new-session": ("new-s", "cefFnstxy", "c"), "new-window": ("new-w", "ceFnt", "c"),
                 "split-window": ("spl", "ceFlpt", "c"), "respawn-pane": ("respawn-p", "cet", "c"),
                 "respawn-window": ("respawn-w", "cet", "c"), "run-shell": ("run-", "cdt", "c"), "if-shell": ("if-", "t", ""),
                 "pipe-pane": ("pip", "t", ""), "display-popup": ("display-po", "bcdehsStTwxy", "d"),
                 "send-keys": ("send-k", "cNTt", "")}
TMUX_ALIASES = {"new": "new-session", "neww": "new-window", "splitw": "split-window", "respawnp": "respawn-pane",
                "respawnw": "respawn-window", "run": "run-shell", "if": "if-shell", "pipep": "pipe-pane", "popup": "display-popup",
                "send": "send-keys"}
#: names that start a simple command of their own once the shell has split `tmux A \; B` at the `;`
TMUX_BARE = {n for n in TMUX_CARRIERS if "-" in n} | {"neww", "splitw", "respawnp", "respawnw", "pipep"}
#: send-keys: keys that end a typed line, and keys that type a blank
TMUX_ENTER, TMUX_BLANK = {"Enter", "C-m", "C-j", "KPEnter", "^M", "^J"}, {"Space", "Tab"}
TMUX_KEY = re.compile(r"([CMS]-)+\S+|\^.|Escape|BTab|BSpace|Up|Down|Left|Right|Home|End|IC|DC|PageUp|PgUp|PageDown|PgDn|NPage|"
                      r"PPage|F\d+|KP\w+")
#: xargs: its options that take a value in their word or the next, and those that take one only in their word (`-i{}`)
XARGS_VALUED, XARGS_STUCK = set("adEILnPs"), set("eil")
SHELL_CODE = re.compile(r"-[a-zA-Z]*c[a-zA-Z]*")


def options_end(args: list[str], valued: str | set[str], stuck: str | set[str] = "") -> tuple[int, dict[str, str]]:
    """Where `args`'s single-letter options end (past a `--`), and the value each valued one took: a `valued` one the rest of
    its word or the next word, a `stuck` one only the rest of its word."""
    i, given = 0, {}
    while i < len(args) and args[i].startswith("-") and len(args[i]) > 1:
        a, i = args[i], i + 1
        if a == "--":
            break
        for at, letter in enumerate(a[1:], 2):
            if letter in valued or letter in stuck:
                given[letter] = a[at:] or (args[i] if i < len(args) and letter in valued else "")
                i += not a[at:] and letter in valued
                break
            given.setdefault(letter, "")
    return i, given


def typed(key: str, literal: bool) -> str:
    """What `tmux send-keys` types for `key`: itself, unless it names a key (`-l`: never); a blank for `Space` and `Tab`, and a
    line's end for `Enter`, `C-m` and any other key, which may run or cut the line typed so far."""
    if literal or not (key in TMUX_ENTER or key in TMUX_BLANK or TMUX_KEY.fullmatch(key)):
        return key
    return " " if key in TMUX_BLANK else "\n"


def tmux_carried(args: list[str]) -> list[str]:
    """The shell commands `tmux ARGS` has run (or types into a pane, `send-keys`), each as one text: global `-c CMD`, and each
    command of TMUX_CARRIERS, by its name, alias or a prefix that names one; `run-shell -C` and `if-shell`'s branches are tmux
    commands, read again. Several words after the options are one command's argv, joined; a start directory becomes a `cd`."""
    i, given = options_end(args, "cfLST")
    if "c" in given:
        return [given["c"]]
    if i >= len(args):
        return []
    name, rest = args[i], args[i + 1:]
    canon = TMUX_ALIASES.get(name) or next((c for c, (least, _, _) in TMUX_CARRIERS.items()
                                            if c.startswith(name) and name.startswith(least)), "")
    if not canon:
        return []
    _, valued, dir_option = TMUX_CARRIERS[canon]
    j, given = options_end(rest, valued)
    rest = rest[j:]
    if canon == "send-keys":
        if "X" in given or "H" in given:
            return []
        text = "".join(typed(k, "l" in given) for k in rest)
        return [text] if text.strip() else []
    if canon == "run-shell" and "C" in given:
        return [t for c in rest for t in tmux_carried(shlex.split(c))]
    if canon == "if-shell":
        tmux = [t for c in rest[1:] for t in tmux_carried(shlex.split(c))]
        return tmux if "F" in given else rest[:1] + tmux
    if not rest:
        return []
    text = rest[0] if len(rest) == 1 else shlex.join(rest)
    where = given.get(dir_option, "") if dir_option else ""
    return [f"cd {shlex.quote(where)} && {text}" if where else text]


def carried(words: list[str]) -> list[str]:
    """The shell commands the simple command `words` (unwrapped) has run, each as one text, beyond the wrappers `unwrap`
    strips: `bash|sh|zsh|dash -c CODE` (any bundle with `c`, `-lc`), `env [VAR=VALUE...] CMD` (the assignments kept), `xargs
    CMD`, `eval WORDS`, `watch CMD`, and what tmux runs (`tmux_carried`), also as a command of its own after `\\;`. A script
    file's contents are not read."""
    if not words:
        return []
    name, args = os.path.basename(words[0]), words[1:]
    if name in SHELLS:
        at = next((k for k, a in enumerate(args[:-1]) if SHELL_CODE.fullmatch(a)), None)
        return [args[at + 1]] if at is not None else []
    if name == "env":
        run = env_command(args)
        if len(run) == 1 and " " in run[0]:
            return run
        assignments = [a for a in args[:len(args) - len(run)] if "=" in a and not a.startswith("-")]
        return [shlex.join(assignments + run)] if run else []
    if name == "xargs":
        i, _ = options_end([a for a in args if not a.startswith("--")], XARGS_VALUED, XARGS_STUCK)
        rest = [a for a in args if not a.startswith("--")][i:]
        return [shlex.join(rest)] if rest else []
    if name == "eval":
        return [" ".join(args)] if args else []
    if name == "watch":
        i, _ = options_end([a for a in args if not a.startswith("--")], "n")
        rest = [a for a in args if not a.startswith("--")][i:]
        return [" ".join(rest)] if rest else []
    if name == "tmux":
        return tmux_carried(args)
    if name in TMUX_BARE:
        return tmux_carried(words)
    return []


def prose(words: list[str]) -> bool:
    """Whether the simple command `words` (unwrapped) only carries text: `git`'s arguments (a commit message) and those of a
    `research` command that runs nothing (no `--`)."""
    prog = os.path.basename(words[0]) if words else ""
    return (prog == "git" and "--no-index" not in words) or (prog == "research" and "--" not in words)


def refused(words: list[str], filtered: bool = False, remote: bool = False, text: str = "") -> str:
    """What the simple command `words` would print that it may not (ENVIRONMENT, ARGV) or "". `filtered`: its output goes
    straight into a filter that keeps only matching lines; `remote`: another machine runs it; `text`: the whole command."""
    for w in words:
        if "$(" in w and (kind := refused_text(w.split("$(", 1)[1], remote)):
            return kind
    run = unwrap(strip_redirections(words))
    if not remote and not prose(run) and os.path.basename(run[0] if run else "") not in ("ssh", "research"):
        if any(PROC_ARGV.search(w) or w.endswith("/cmdline") or (w == "cmdline" and "/proc" in text) for w in words):
            return ARGV
        if kind := ps_variables(strip_redirections(words)) or argv_listing(run):
            return kind
    words = run
    if not words:
        return ""
    name, args = os.path.basename(words[0]), words[1:]
    opts = [a for a in args if a.startswith(("-", "+"))]
    names = [a for a in args if not a.startswith(("-", "+"))]
    if name == "env":
        run = env_command(args)
        if not run:
            return ENVIRONMENT
        if len(run) == 1 and " " in run[0]:
            return refused_text(run[0], remote)
        assignments = [a for a in args[:len(args) - len(run)] if "=" in a and not a.startswith("-")]
        return refused(assignments + run, remote=remote, text=text)
    if name == "printenv":
        return ENVIRONMENT if not names or any(secret(n) for n in names) else ""
    if name == "set":
        return "" if args else ENVIRONMENT
    if name == "export":
        return ENVIRONMENT if all(a == "-p" for a in args) else ""
    if name in ("declare", "typeset", "local", "readonly"):
        if not names:
            return ENVIRONMENT if not opts or any(set(o.lstrip("-+")) - set("fF") for o in opts) else ""
        return ENVIRONMENT if any("p" in o for o in opts) and any(secret(n) for n in names) else ""
    if texts := carried(words):
        return next((kind for t in texts if (kind := refused_text(t, remote))), "")
    if name == "ssh":
        return refused_text(" ".join(ssh_words(args)), True)
    if name == "research" and "ssh" in args and "--" in args:
        return refused_text(" ".join(args[args.index("--") + 1:]), True)
    if name == "ps":
        return ARGV if remote and not filtered and every_argv(args) else ""
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
            return ENVIRONMENT if not shown or any(secret(n) for n in shown) else ""
    return ""


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


def refused_text(text: str, remote: bool = False) -> str:
    """What `text` would print that it may not (ENVIRONMENT, ARGV) or "": `remote` when another machine runs it."""
    if m := ENVIRON.search(text):
        return ENVIRONMENT if m[0].endswith("environ") else ARGV
    try:
        commands = words_of(text)
    except ValueError:                                   # unbalanced quotes: judge the raw text
        return ENVIRONMENT if FALLBACK.search(text) else ARGV if LISTING_FALLBACK.search(text) else ""
    return next((kind for i, (w, _) in enumerate(commands) if (kind := refused(w, filtered(commands, i), remote, text))), "")


#: commands that write what they read to their output
READERS = {"cat", "head", "tail", "less", "more", "tac", "nl", "sed", "awk", "cut", "xxd", "od", "hexdump", "strings", "base64",
           "base32", "basenc", "uuencode", "sort", "uniq", "paste", "column", "fold", "rev", "diff", "comm", "cmp"} | FILTERS
#: commands that print only their input (`tr a-z A-Z < FILE`): `tr`'s arguments are sets, and `tee`'s are files it writes
INPUT_PRINTERS = {"tr", "tee"}
#: archivers and compressors, which write what they pack to their output only when told to (`packed`)
TARS = {"tar", "gtar", "bsdtar"}
PACKERS = TARS | {"zip", "unzip", "cpio", "gzip", "gunzip", "bzip2", "bunzip2", "xz", "unxz", "lzma", "unlzma", "zstd", "unzstd",
                  "lz4", "unlz4", "pigz", "unpigz", "pbzip2", "compress", "uncompress"}
PRINTERS = READERS | INPUT_PRINTERS | PACKERS | {"dd", "openssl"}
#: `zcat`, `bzgrep`, `xzless` and the like read a compressed file, or a plain one, as the tool they wrap does
WRAPS = re.compile(r"^(g?z|bz|xz|lz4?|zstd)(?=(cat|less|more|grep|egrep|fgrep|diff|cmp)$)")
#: readers whose first argument is a pattern or a script, not a file, unless an option of SCRIPT_GIVEN gave it
SCRIPTED = {"sed", "awk"} | FILTERS
SCRIPT_GIVEN = {"e", "f", "regexp", "file", "expression", "source"}
#: each reader's options that take a value, the short ones as letters and the long ones by name: the word after one is
#: neither its pattern nor a file (`rg -t py PATTERN DIR`)
GREP_VALUED = ("efABCmdD", {"regexp", "file", "after-context", "before-context", "context", "max-count", "include", "exclude",
                            "exclude-dir", "exclude-from", "label", "binary-files", "devices", "directories"})
VALUED = {"grep": GREP_VALUED, "egrep": GREP_VALUED, "fgrep": GREP_VALUED,
          "rg": ("efABCmtTgjMrEd", {"regexp", "file", "after-context", "before-context", "context", "max-count", "type", "type-not",
                                    "type-add", "type-clear", "glob", "iglob", "threads", "max-columns", "replace", "encoding",
                                    "max-depth", "max-filesize", "sort", "sortr", "color", "colors", "context-separator",
                                    "field-match-separator", "field-context-separator", "path-separator", "pre", "pre-glob",
                                    "engine", "ignore-file", "dfa-size-limit", "regex-size-limit"}),
          "sed": ("efl", {"expression", "file", "line-length"}),
          "awk": ("fFveil", {"file", "field-separator", "assign", "source", "include", "load"}),
          "sort": ("koStT", {"key", "output", "buffer-size", "field-separator", "temporary-directory"}),
          "diff": ("UCIxXLFWD", set()), "zip": ("bnPtO", set()),
          "tar": ("bCfFgHIKLNTVX", {"file", "directory", "files-from", "exclude", "exclude-from", "transform", "xform", "owner",
                                    "group", "mode", "mtime", "newer", "format", "label", "sort", "strip-components",
                                    "use-compress-program", "to-command", "checkpoint-action", "info-script", "rsh-command"})}
#: the options under which a reader shows no line of what it reads: a grep that lists files, counts or only tests (rg's -L
#: follows symlinks), a sed that edits in place, a diff that only says whether the files differ, a sort into a file
GREP_SILENT = {"l", "c", "q", "files-with-matches", "files-without-match", "count", "quiet", "silent"}
SILENT = {"grep": GREP_SILENT | {"L"}, "egrep": GREP_SILENT | {"L"}, "fgrep": GREP_SILENT | {"L"},
          "rg": GREP_SILENT | {"count-matches", "files"}, "sed": {"i", "in-place"}, "diff": {"q", "brief"}, "sort": {"o", "output"}}
#: `cmp` says only where two files first differ, unless one of these has it print their bytes
CMP_BYTES = {"l", "b", "verbose", "print-bytes"}
#: an archive's name that puts it on `tar`'s output or a terminal, not in a file (so does no `-f` at all), and `tar`'s options
#: that write a member to its output or run a program: without them a `tar` extracting from its input only unpacks to disk
ON_OUTPUT = re.compile(r"-$|/(dev|proc)/(?!null$)")
TAR_SHOWS = {"O", "to-stdout", "to-command", "I", "use-compress-program", "checkpoint-action", "F", "info-script",
             "new-volume-script", "rsh-command"}
#: `tar`'s long options this rule reads, each of which `tar` also takes cut short (`--extr`, `--to-stdo`)
TAR_READ = {"create", "extract", "get", "file", "files-from", "directory"} | {name for name in TAR_SHOWS if len(name) > 1}
#: what the far side of a copy may run beside the `tar` that unpacks it
SETUP = {"umask", "mkdir", "chmod", "chown", "cd"}
#: the openssl commands that write out the private key they read, unless `-pubout` or `-noout` is given; `enc` writes out
#: any file it reads, encoded or enciphered
KEY_COMMANDS = {"pkey", "rsa", "ec", "dsa", "pkcs8"}
#: the shell's own readers of a command's input, which put it in a variable: one `echo` from the transcript
SHELL_READS = {"read", "mapfile", "readarray"}
#: interpreters whose `-c CODE` or `-e CODE` is judged, and what in CODE says it shows a file it names
INTERPRETER = re.compile(r"python[\d.]*|perl|ruby")
CODE_OPTION = re.compile(r"-\w*[ce]")
CODE_READS = re.compile(r"open\(|read|print|cat")
#: perl's and ruby's option that edits the files in place, alone or after `-p`, `-n` and the like: nothing is printed
IN_PLACE = re.compile(r"-[lnpaw0-9]*i(\.\w+|~)?")
#: what a file's name, written out or as a glob, says it holds
SECRET_FILES = {"a private key": ("*.pem", "*.key", "id_rsa*", "id_ed25519*", "id_ecdsa*"),
                "an env or credentials file": ("*.env", ".netrc", "credentials", "credentials.*")}
#: names that hold no secret: a key's public half, ssh's public files, a template, a module or a doc (`credentials.py`)
PUBLIC_FILES = ("*.pub", "*-pub.pem", "authorized_keys*", "known_hosts*", "*.example", "*.template", "*.py", "*.md")
#: the suffix of a compressed or archived copy, which holds what the file named without it holds
PACKED = re.compile(r"((\.tar)?\.(gz|bz2|xz|zst|lz4|lzma|Z)|\.tgz|\.tar|\.zip)$")
#: directories where every file counts, so a glob over one does (`/root/.config/verity/*` matches the key beside the env files)
SECRET_DIRS = re.compile(r"(^|/)(\.ssh|\.config/verity|secrets)(/|$)")
#: what a reader's line names when the parse itself fails: cruder, and it errs toward refusing
SECRET_FALLBACK = re.compile(r"\b(%s)\b[^|;&\n]*?([^\s'\"]*(\.pem|\.key|\.env|\.netrc|credentials|id_rsa|id_ed25519|id_ecdsa|"
                             r"\.ssh/|\.config/verity/|secrets/)[^\s'\"]*)" % "|".join(sorted(READERS)))
ENVNAMES = "tools/agent-guard/envnames.py"


def public_file(path: str) -> bool:
    """Whether the name `path` says it holds no secret: one of PUBLIC_FILES, `.ssh/config`, or a fixture under a `tests/`
    directory."""
    parts = path.rstrip("/").split("/")
    return any(fnmatch.fnmatchcase(parts[-1], p) for p in PUBLIC_FILES) or "tests" in parts[:-1] or parts[-2:] == [".ssh", "config"]


def secret_file(path: str) -> str | None:
    """What the file name or glob `path` says it holds, when that is secret: a private key, an env or credentials file, or a
    file of a secrets directory (`.ssh/`, `.config/verity/` under any home, a `secrets/`), under its own name or the one a
    compressed or archived copy is a copy of (PACKED). None for a `public_file`."""
    for whole in dict.fromkeys((path, PACKED.sub("", path.rstrip("/")))):
        if public_file(whole):
            continue
        name = whole.rstrip("/").split("/")[-1]
        for kind, patterns in SECRET_FILES.items():
            if any(fnmatch.fnmatchcase(name, p) for p in patterns):
                return kind
        if SECRET_DIRS.search(whole):
            return "a file of a secrets directory"
    return None


def tracked(path: str, where: str) -> bool:
    """Whether `path`, read from the directory `where` on this machine, names only files whose bytes are the ones git has
    committed there: every agent can read those already (`git show`). A glob is expanded as the shell would expand it, and one
    changed file, untracked file or link among its matches keeps the refusal; so does a glob over more than two levels of
    directories, which could walk a whole disk."""
    files = glob.glob(os.path.join(where, path)) if len(re.findall(r"[^/]*[*?\[][^/]*/", path)) <= 2 else []
    if not files or not all(os.path.isfile(f) for f in files):
        return False
    there = os.path.dirname(files[0])
    try:
        committed, here = (subprocess.run(["git", "-C", there, *args], capture_output=True, text=True, timeout=3) for args in (
            ["rev-parse", *("HEAD:./" + os.path.relpath(f, there) for f in files)], ["hash-object", "--no-filters", "--", *files]))
    except (OSError, ValueError, subprocess.SubprocessError):   # no git, no answer or no such path: not known to be committed
        return False
    return committed.returncode == here.returncode == 0 and committed.stdout == here.stdout != ""


def operands(reader: str, args: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """(the options given, the other arguments) of `reader ARGS`: short options as letters and long ones by name, each with
    the values given it when it takes one (VALUED: the rest of its word, else the next word), which are no other argument."""
    shorts, longs = VALUED.get(reader, ("", ()))
    given, rest, i = {}, [], 0
    while i < len(args):
        a = args[i]
        i += 1
        if a == "--":
            return given, rest + args[i:]
        if a.startswith("--"):
            name, valued, value = a[2:].partition("=")
            given.setdefault(name, []).extend([value] if valued else args[i:i + 1] if name in longs else [])
            i += name in longs and not valued
        elif a.startswith("-") and len(a) > 1:
            for at, letter in enumerate(a[1:], 2):
                given.setdefault(letter, [])
                if letter in shorts:
                    given[letter] += [a[at:]] if at < len(a) else args[i:i + 1]
                    i += at == len(a)
                    break
        else:
            rest.append(a)
    return given, rest


def files_printed(reader: str, args: list[str], inputs: list[str]) -> list[str] | None:
    """The files `reader ARGS` writes to its output, `inputs` being what it reads as its own input (`< FILE`), or None when
    it shows no line of them (SILENT, `cmp` without CMP_BYTES, `dd` with an `of=`). A grep's, sed's or awk's first argument
    is its pattern or script, unless an option gave that (SCRIPT_GIVEN)."""
    if reader in INPUT_PRINTERS:
        return inputs
    if reader == "dd":
        return None if any(a.startswith("of=") for a in args) else [a[3:] for a in args if a.startswith("if=")] + inputs
    given, rest = operands(reader, args)
    if given.keys() & SILENT.get(reader, set()) or (reader == "cmp" and not given.keys() & CMP_BYTES):
        return None
    return (rest[1:] if reader in SCRIPTED and not given.keys() & SCRIPT_GIVEN else rest) + inputs


def tar_options(args: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """`operands` of `tar ARGS`, whose first word may be its options without a dash, their values after it (`tar czf OUT DIR`).
    A long option cut short counts as every name of TAR_READ it starts: `tar` takes it for one of them or refuses it."""
    if args and not args[0].startswith("-"):
        spelled, values = [], args[1:]
        for letter in args[0]:
            spelled += ["-" + letter, *values[:letter in VALUED["tar"][0]]]
            values = values[letter in VALUED["tar"][0]:]
        args = spelled + values
    given, rest = operands("tar", args)
    for short in [g for g in given if len(g) > 1 and g not in TAR_READ | VALUED["tar"][1]]:
        for name in [n for n in TAR_READ if n.startswith(short)]:
            given.setdefault(name, []).extend(given[short])
    return given, rest


def packed(tool: str, args: list[str], inputs: list[str], piped: list[str], sunk: bool) -> list[str] | None:
    """The files the archiver or compressor `tool ARGS` writes to its output, or None when it writes none there. `tar`
    creating an archive on its output (ON_OUTPUT, or no `-f`: only a terminal stops it) writes the files it names, each also
    as a member of each `-C DIR`, and with `-T -` the names piped in, unless that output is `sunk`, handed to a command that
    only unpacks it (`unpacks`), and the `tar` itself runs no program (TAR_SHOWS). `tar -x` writes them with `-O`. So do
    `zip` with `-` for its archive, `unzip -p|-c`, `cpio -o` with no archive file, of the names piped in, and a compressor
    with `-c`, or of its input."""
    given, rest = tar_options(args) if tool in TARS else operands(tool, args)
    if tool in TARS:
        archives = given.get("f", []) + given.get("file", [])
        if given.keys() & {"x", "extract", "get"}:
            return archives + rest + inputs if given.keys() & {"O", "to-stdout"} else None
        if not given.keys() & {"c", "create"} or (archives and not any(map(ON_OUTPUT.match, archives))):
            return None
        if sunk and set(archives) <= {"-"} and not given.keys() & TAR_SHOWS:
            return None
        names = rest + (piped if "-" in given.get("T", []) + given.get("files-from", []) else [])
        return names + [os.path.join(d, n) for d in given.get("C", []) + given.get("directory", []) for n in names]
    if tool == "zip":
        return rest[1:] + (piped if "@" in given else []) if rest[:1] == ["-"] else None
    if tool == "unzip":
        return rest if given.keys() & {"p", "c"} else None
    if tool == "cpio":
        return piped if given.keys() & {"o", "create"} and not given.keys() & {"F", "O", "file"} else None
    return (rest if given.keys() & {"c", "stdout", "to-stdout"} else []) + inputs


def far_words(run: list[str]) -> list[str]:
    """The last words of the command `run` when they are what it has another machine's shell run on `run`'s own input and
    output: ssh's remote command or `research pods ssh POD --`'s. None for any other command."""
    prog = os.path.basename(run[0]) if run else ""
    if prog == "ssh":
        return ssh_words(run[1:])
    return run[run.index("--") + 1:] if prog == "research" and run[1:3] == ["pods", "ssh"] and "--" in run else []


def unpacks(words: list[str]) -> bool:
    """Whether the command `words` does nothing with its input but unpack it to disk, so that an archive piped into it shows
    nowhere: a `tar` that extracts from its input under none of TAR_SHOWS, run here (after `sudo` and the like) or as the
    command ssh or `research pods ssh` has the far side run (`far_words`), where commands of SETUP may stand beside it,
    joined by `;` or `&&`. The list is closed: any other program may write what it reads to its output."""
    run = unwrap(strip_redirections(words))
    if run[:1] and os.path.basename(run[0]) in TARS:
        given, _ = tar_options(run[1:])
        return bool(given.keys() & {"x", "extract", "get"}) and not given.keys() & TAR_SHOWS and set(
            given.get("f", []) + given.get("file", [])) <= {"-"}
    try:
        commands = words_of(" ".join(far_words(run)))
    except ValueError:
        return False
    steps = [unwrap(strip_redirections(w)) for w, _ in commands]
    return (all(op in (";", "&&", "") for _, op in commands) and any(map(unpacks, steps))
            and all(os.path.basename(s[0]) in SETUP or unpacks(s) for s in steps if s))


def key_printed(args: list[str], inputs: list[str]) -> list[str] | None:
    """The file `openssl ARGS` writes to its output, or None: one of KEY_COMMANDS writes out the private key it reads (`-in
    FILE`, else its input) unless `-pubout` or `-noout` is given, and the key's private numbers with `-text` even then; `enc`
    writes out the file it reads. `-out FILE` writes a file instead, and `-pubin` reads a public key."""
    given = set(args)
    if not (KEY_COMMANDS | {"enc"}) & set(args[:1]) or given & {"-out", "-pubin"}:
        return None
    if "-text" not in given and given & {"-pubout", "-noout"}:
        return None
    return [args[i + 1] for i, a in enumerate(args[:-1]) if a == "-in"] or inputs


def code_printed(interpreter: str, args: list[str], inputs: list[str]) -> list[str] | None:
    """The files the code of `python -c CODE`, `perl -e CODE` or `ruby -e CODE` could print, or None: when CODE opens, reads or
    prints (CODE_READS), or perl's `-p` prints each line for it, the path-like words of its string literals, the arguments
    after it and its input. `perl -pi -e CODE FILE` edits FILE in place (IN_PLACE) and prints nothing."""
    at = next((i for i, a in enumerate(args[:-1]) if CODE_OPTION.fullmatch(a)), None)
    if at is None or not (CODE_READS.search(args[at + 1]) or "p" in args[at]):
        return None
    if not interpreter.startswith("python") and any(IN_PLACE.fullmatch(a) for a in args[:at]):
        return None
    literals = [w for quoted in re.findall(r"""['"]([^'"]+)['"]""", args[at + 1]) for w in quoted.split() if re.search(r"[/.]", w)]
    return literals + args[at + 2:] + inputs


def piped_names(run: list[str]) -> list[str]:
    """The words of the command `run` that can name files to the `xargs` it is piped into: all of them, less a grep's, sed's
    or awk's options and pattern (`rg -l credentials tools/ | xargs sed -n 1p` names `tools/`)."""
    prog = os.path.basename(run[0]) if run else ""
    if prog not in SCRIPTED:
        return run
    given, rest = operands(prog, run[1:])
    return rest if given.keys() & SCRIPT_GIVEN else rest[1:]


def printed(words: list[str], at: int, inputs: list[str], piped: list[str], sunk: bool) -> tuple[str, list[str] | None, list[str]]:
    """(the command as a refusal names it, the files it writes to its output or None, those of them that are keys whatever
    their names) for the printer or the interpreter `words[at]`, judged on the words after it, its input `inputs` and, behind
    `xargs` or as an archiver's list, the names `piped` into it."""
    said, args = os.path.basename(words[at]), words[at + 1:]
    name, keys = WRAPS.sub("", said), None
    if name == "openssl":                                # the file a key command names is a key, whatever its name
        said, files = " ".join([said, *args[:1]]), key_printed(args, inputs)
        keys = files if args[:1] != ["enc"] else None
    elif name in PACKERS:
        files = packed(name, args, inputs, piped, sunk)
    else:
        files = files_printed(name, args, inputs) if name in PRINTERS else code_printed(name, args, inputs)
    if files is not None:                                # `find -exec READER {} +` reads what find names, `xargs READER` what is piped in
        files = files + (words[:at] if "{}" in args else []) + (piped if "xargs" in map(os.path.basename, words[:at]) else [])
    return said, files, keys or []


def secret_read(text: str, where: str | None = None, sunk: bool = False) -> tuple[str, str, str] | None:
    """(command, file, what the file holds) of the first command in `text` that would print a secret file, or None. In each
    simple command every word that names one of PRINTERS, or INTERPRETER given code, wherever it stands (after `sudo`, as
    ssh's remote command, after `research pods ssh POD --`, behind `xargs`, as `find`'s `-exec`), is judged on the files it
    names, its input (`< FILE`), the variable of a `for` loop over such files and, behind `xargs`, the names the commands
    piped into it carry; so are a loop's or `read`'s input (`done < FILE`) and `$(<FILE)`. A word that only shares a
    printer's name (a host, an image, a directory) therefore hides no reader after it. A grep, sed or awk that is the
    command itself, here or the one ssh or `research pods ssh` has the far side run, is the last word judged: what follows
    it is its pattern, its script and its files. The quoted strings up to it are read as commands in turn, which is how a
    wrapper script's argument, `bash -c`'s and an interpreter's code are seen. `git`'s arguments and those of a `research`
    command that runs nothing (no `--`) are prose and are not read. `where` is the directory the commands run in when they
    run on this machine, None on another or when they may move (`worktree_guard.moves`: any `cd` among them, run or not):
    there a command that reads a file itself (behind nothing but `sudo`, `timeout` and the like without options, or in such
    a `bash -c`) reads one git has committed freely (`tracked`). `sunk` says the output of `text` as a whole goes to a
    command that only unpacks it. Inside `text` a command's does when a `|` hands it to one (`unpacks`) that shares the pipe with no other command (no subshell or group),
    or `text`'s own does and the command is piped nowhere, and it is redirected nowhere but to `/dev/null`; so then does the
    output of the command it has another machine run (`far_words`). Only a `tar` that is the command itself, here or there,
    is let through on it."""
    try:
        commands = words_of(text)
        if where is not None and guard().moves([unwrap(w) for w in flattened(text)]):
            where = None
    except ValueError:                                   # unbalanced quotes: the words between the shell's operators, unquoted
        commands = [(re.findall(r"[<>]+|[^\s'\"<>]+", part), "") for part in re.split(r"[|;&()`\n]", text)]
        where, sunk = None, False
    plain = "`" not in text and not any(set(op) & set("()") for _, op in commands)
    loops = {m[1]: [w for w in re.split(r"[\s'\"`$()]+", m[2]) if secret_file(w)]
             for m in re.finditer(r"\bfor\s+(\w+)\s+in\s([^;\n]*)", text)}
    piped, before = [], ""
    for n, (words, op) in enumerate(commands):
        inputs = [words[i + 1] for i, w in enumerate(words[:-1]) if w == "<"]
        whole = all(to == "/dev/null" for w, to in zip(words, words[1:] + [""]) if ">" in w and set(w) <= set("&<>|"))
        taker = commands[n + 1][0] if op in ("|", "|&") and n + 1 < len(commands) else None
        into = whole and (plain and taker[0] != "{" and unpacks(taker) if taker else sunk and op not in ("|", "|&"))
        words, upstream, opened = strip_redirections(words), piped, before.endswith("(<")
        run, before = unwrap(words), op
        prog, first, far = os.path.basename(run[0]) if run else "", len(words) - len(run), far_words(run)
        direct = {first, len(words) - len(unwrap(far))} if far else {first}   # where the command stands, here or on the far side
        prose = (prog == "git" and "--no-index" not in run) or (prog == "research" and "--" not in run)
        piped = upstream + piped_names(run) if op in ("|", "|&") and not prose else []
        if prose:
            continue
        ats = [i for i, w in enumerate(words) if WRAPS.sub("", os.path.basename(w)) in PRINTERS
               or INTERPRETER.fullmatch(os.path.basename(w)) and any(CODE_OPTION.fullmatch(a) for a in words[i + 1:-1])]
        last = next((i for i in ats if i in direct and os.path.basename(words[i]) in SCRIPTED), len(words))
        reads = [(at, *printed(words, at, inputs, upstream, into and at in direct)) for at in ats if at <= last]
        if opened or (inputs and not words):             # `$(<FILE)`: the shell itself reads the file into the command line
            reads.append((first, "$(<", words[:1] + inputs, []))
        elif inputs and (words[:1] == ["done"] or prog in SHELL_READS):   # `while read ...; done < FILE`, `read -r KEY < FILE`
            reads.append((first, "while read" if words[:1] == ["done"] else prog, inputs, []))
        here = where if where is not None and not any(w.startswith("-") for w in words[:first]) else None
        for at, name, files, keys in reads:
            for a in files or []:
                var = re.fullmatch(r"\$\{?(\w+)\}?", a)
                for f in var and loops.get(var[1]) or [a]:
                    kind = "a private key" if a in keys and not public_file(f) else secret_file(f)
                    if kind and not (here is not None and at == first and tracked(f, where)):
                        return name, f, kind
        for i, w in enumerate(words[:last]):             # a shell's `-c CODE` runs here; any other string may run anywhere
            code = prog in SHELLS and i and re.fullmatch(r"-\w*c\w*", words[i - 1])
            if re.search(r"\s|\$\(", w) and (found := secret_read(w, here if code else None, into and w in far and "$(" not in w)):
                return found
    return None


def stated_cwd(raw: str) -> str:
    """The directory Cursor's event `raw` says its command runs in, "" when it names none: a guess at it (the workspace's
    root) could take a file of another directory for the checkout's."""
    try:
        return str(json.loads(raw).get("cwd") or "")
    except (ValueError, AttributeError):
        return ""


def secret_file_reason(command: str, cwd: str = "") -> str | None:
    """Why this command, run in `cwd`, may not run (it would print a key, an env or credentials file, or a file of a secrets
    directory), or None. AGENT_GUARD_LOCAL_OK=1 does not lift it: that opt-in is the local-work gate's alone."""
    try:
        found = secret_read(command, cwd or None)
    except (ValueError, IndexError, KeyError, TypeError, AttributeError, RecursionError):   # a bug here must not allow what the text shows
        m = SECRET_FALLBACK.search(command)
        found = (m[1], m[2], "a secret file") if m else None
    if found is None:
        return None
    reader, path, kind = found
    why = ("An archive, a compressed or an encoded copy on the output is the file itself: `tar` and `gzip` stop only at a "
           "terminal, and an agent's shell has none. Write it to a file (`tar -cf OUT.tar FILE...`, `gzip -k FILE`, `openssl enc "
           "-out OUT`), or copy it to another machine with `tar -cf - FILE... | ssh HOST tar -xf - -C DIR`: a `tar` that only "
           "unpacks its input is the one place such an archive may go. "
           if reader.split()[0] in PACKERS or reader == "openssl enc"
           else "`openssl` writes out the private key it reads: add `-pubout` for its public half or `-noout` to only check it, and "
           "never `-text`, which prints the private numbers (`-text_pub` prints the public ones). " if reader.startswith("openssl")
           else "A filter after it (`cut -d= -f1`, `sed 's/=.*//'`) still prints a key file's lines and a multi-line value's later "
           "ones, which hold no `=`. ")
    return (f"this command would print {kind}: `{reader}` reads `{path}`. {why}List an env file's variable "
            f"names with `python3 {ENVNAMES} FILE...` (on another machine: `ssh HOST python3 - FILE... < {ENVNAMES}`), which "
            "prints nothing of a file that isn't one; look at a key with `stat`, `sha256sum` or `openssl pkey -in FILE -pubout`. "
            "A command that needs the contents reads the file inside the program that needs it (give it the path, `. FILE`, "
            "`CMD < FILE`, `cp`): a value captured in a shell variable (`KEY=$(cat FILE)`) is one `echo` from the transcript. "
            f"A file git tracks in the checkout, unchanged, reads freely there, named from the directory the command starts in "
            f"and with no `cd` in the command. {LOCAL_OK} does not lift this.")


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
    if texts := carried(words):
        return max(weight(text) for text in texts)
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
    """Seconds one simple command plainly blocks the shell for: a `sleep`, `research run --wait-max S`, or what ssh or a
    carrier other than tmux (`carried`: `bash -c`, `env`, `xargs`, ...) runs, each bounded by a `timeout` around it; 0 when it
    states no literal time."""
    s = max((blocks_text(w.split("$(", 1)[1]) for w in words if "$(" in w), default=0.0)
    timeouts: list[str] = []
    words = unwrap(strip_redirections(words), timeouts)
    bound = min((d for t in timeouts if (d := duration([t])) is not None), default=math.inf)
    name, args = (os.path.basename(words[0]), words[1:]) if words else ("", [])
    if name == "sleep":
        s = max(s, duration(args) or 0.0)
    elif name != "tmux" and name not in TMUX_BARE and (texts := carried(words)):    # a tmux pane's command blocks no call
        s = max(s, *(blocks_text(text) for text in texts))
    elif name == "ssh":
        s = max(s, blocks_text(" ".join(ssh_words(args))))
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


def flattened(text: str) -> list[list[str]]:
    """`text`'s simple commands without redirections, each followed by those it carries (`carried`), in the order they run."""
    out = []
    for words, _ in words_of(text):
        words = strip_redirections(words)
        out.append(words)
        out.extend(w for t in carried(unwrap(words)) for w in flattened(t))
    return out


#: the variables that change which repository or configuration git reads, and the builtins that set or unset one
GIT_VARIABLES = re.compile(r"GIT_\w+|HOME|XDG_CONFIG_HOME")
SETTERS = {"export", "declare", "typeset", "readonly", "local", "unset"}


def git_env(raw: list[list[str]]) -> list[dict[str, str]]:
    """The GIT_VARIABLES each of the simple commands `raw` runs under, as far as the command shows: those assigned ahead of a
    `git` (`GIT_DIR=D git push`), and `$` for each that any other command in it sets or unsets (an assignment alone or ahead of
    another program, `export`, `declare`, `unset`), wherever it sits, since a loop or a function may run it first."""
    ahead, elsewhere = [], set()
    for words in raw:
        run = unwrap(words)
        mine = dict(w.split("=", 1) for w in words[:len(words) - len(run)] if "=" in w and GIT_VARIABLES.fullmatch(w.split("=")[0]))
        if not (run and os.path.basename(run[0]) == "git"):
            elsewhere |= set(mine)
        if run[:1] and run[0] in SETTERS:
            elsewhere |= {n for a in run[1:] if GIT_VARIABLES.fullmatch(n := a.split("=", 1)[0])}
        ahead.append(mine)
    return [{**dict.fromkeys(elsewhere, "$"), **mine} for mine in ahead]


def guard():
    """worktree_guard, beside this hook, imported when a rule first needs it."""
    sys.dont_write_bytecode = True                      # no __pycache__ beside the hook: boot.sh copies its directory file by file
    if (here := os.path.dirname(os.path.abspath(__file__))) not in sys.path:
        sys.path.insert(0, here)
    import worktree_guard
    return worktree_guard


def checkout_reason(command: str, conversation: str, cwd: str) -> str | None:
    """One agent per checkout (worktree_guard.py): why this command's git may not run here, or None."""
    worktree_guard = guard()
    try:
        raw = flattened(command)
    except ValueError:
        return None
    next_ok = [assigned(w, "AGENT_GUARD_PUSH_NEXT_OK") for w in raw]
    return worktree_guard.reason([unwrap(w) for w in raw], conversation, cwd, next_ok, git_env(raw))


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
        deny = (ENVIRONMENT if FALLBACK.search(command) or re.search(r"/proc/\S*environ", command)
                else ARGV if LISTING_FALLBACK.search(command) else "")
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
    low = secret_file_reason(command, stated_cwd(raw)) or low   # ahead of the gates' reasons: the one about a secret is the one to act on
    if deny or low:
        reason = REASONS[deny] if deny else low
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
