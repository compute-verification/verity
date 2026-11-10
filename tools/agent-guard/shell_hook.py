#!/usr/bin/env python3
"""Cursor's `beforeShellExecution` hook (`.cursor/hooks.json`): refuses shell commands that would print secrets wholesale.

Cursor runs every agent shell command through it, on any VM, so it holds where the redacting `env` (`install.sh`) was never
installed. A multi-line secret defeats `env | cut -d= -f1`: its later lines print as if they were names, and so reach an
agent's transcript. Refused, anywhere in the command (pipelines,
`$(...)`, `bash -c`, `ssh HOST CMD`, `research pods ssh POD -- CMD`): `env` or `printenv` with nothing to run, `printenv` of a
secret's name, a bare `set`, `export` or `export -p`, `declare`/`typeset` listing variables, reading `/proc/*/environ`, and
`tmux show-environment` (`showenv`) with no name or a secret's (tmux's global environment through `cut -d= -f1` prints a
multi-line key's lines too).
`compgen -e` lists names only and passes. Reads Cursor's hook JSON on stdin and prints its verdict as JSON on stdout.

A process's command line is refused the same way, whoever's it is and however few are asked for: a daemon's or a git child's
token sits in its arguments, and a filter after the listing keeps the very line that holds it, as does asking for the one
pid. Refused: `ps` unless it surely shows no arguments and no environment (`ps_shows`: `ps aux`, `ps -ef`, `ps -eo pid,args |
grep NAME`, `ps -p PID -o args`, `ps -fp $(pgrep NAME)`, `ps PID`, and what it reads as those), `pgrep -a` and `pgrep -fl`,
`pstree -a`, `top -c`, behind `xargs` too, and any path `/proc/.../cmdline` or `/proc/.../environ`, for a pid, `self`, a
variable or a glob. What names processes without their arguments passes: `ps -eo pid,stat,etimes,comm`, `ps -o pid,etimes,comm
-p PID`, `pgrep -f PATTERN`. For the command lines themselves there is `procs.py [PATTERN] [-p PID]` beside this file, which
prints them with whatever may be a credential masked and matches PATTERN against the masked line; install.sh puts it on PATH
as `procs` and, as a filter, in front of `ps` and `pgrep`, for the VM whose hook didn't load. This guards against accidents and
is no boundary: a program that opens `/proc/PID/cmdline` itself (a script, `python3 -c`, a debugger) is not seen, nor is
another tool that prints arguments (`pmap`, `htop`, `systemctl status`), and what would close it is for no program to take a
token in its arguments, which for the exec daemon is Cursor's to change.

A secret file is refused the same way: a command that writes what it reads to its output (READERS: `cat`, `head`, `tail`,
`sed`, `awk`, `cut`, `sort`, `diff`, `xxd`, `base64` and the like, `zcat` and its kin, and a `grep`/`rg` that prints lines, one
without `-l`, `-L`, `-c` or `-q`) applied to a private key (`*.pem`, `*.key`, `id_rsa*`, `id_ed25519*`, `id_ecdsa*`), an env or
credentials file (`*.env`, `.netrc`, `credentials`, `credentials.*`), a compressed or archived copy of one (`x.env.gz`) or
anything in a secrets directory (`.ssh/`, `.config/verity/`, a `secrets/`), named as an argument, a glob, its input
(`< FILE`), a `for` loop's variable or, behind `xargs`, by the commands piped into it, wherever the command sits: over ssh,
after `research pods ssh POD --`, in a quoted string handed to a wrapper script. Every word that names such a command is
judged, so a host, an image or a directory of that name hides no reader after it. So are `tr` and `tee` with such a file as
their input, `dd if=FILE` with no `of=`, a file a `sed` or `awk` program itself names (sed's `r FILE`, `R FILE` and `e
COMMAND`; awk's string literals and `-v` values when it reads with `getline`, `system`, a pipe or `ARGV`), the shell's own
reads (`while read ...; done < FILE`, `read KEY < FILE`, `$(<FILE)`), the code of `python -c`, `perl -e` or `ruby -e` that
names such a file and opens, reads or prints, `openssl enc -in FILE` with no `-out`, and `openssl pkey|rsa|ec|dsa|pkcs8 -in
KEY` with neither `-pubout` nor `-noout`, or with `-text`, which writes the private key out. An archiver or a compressor
that writes to its output is such a command too: `tar -c` with `-f -` or no `-f`
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
a branch registered to another agent.

It also carries messages to running agents, the one channel that reaches a subagent mid-run, on any VM (`inbox.py send
AGENT_ID`). A message waits in `$AGENT_INBOX` (default `/tmp/agent-inbox`) as `<conversation_id>.md`. The agent's next
allowed command is refused once, with the message as the refusal's text, which is what Cursor shows the agent (an
`agent_message` on an allowed command reaches no one). The agent then re-runs the command. A refused secret-printing command
keeps the message queued, and the delivered text is appended to `<conversation_id>.delivered`. Messages pushed from other
VMs reach the inbox through `inbox.py pull`, which the hook starts in the background, detached, at most every
$AGENT_INBOX_POLL_S seconds per agent (default 30; `off` stops it), so the hook itself never waits on the network.

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

from settings import INBOX, INBOX_POLL_S

PATTERN = re.compile(r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL|PRIVATE", re.I)
WRAPPERS = {"sudo", "doas", "command", "builtin", "exec", "nohup", "time", "nice", "stdbuf", "timeout", "setsid", "taskset"}
WRAPPER_ARG_OPTIONS = {"sudo": {"-u", "-g", "-C", "-h", "-p", "-U", "-r", "-t"}, "doas": {"-u", "-C"}, "nice": {"-n"},
                       "timeout": {"-s", "-k", "--signal", "--kill-after"}, "taskset": {"-c", "-p"}}
KEYWORDS = {"{", "}", "!", "if", "then", "elif", "else", "do", "while", "until"}
SHELLS = {"bash", "sh", "zsh", "dash"}
SSH_ARG_OPTIONS = set("bcDEeFIiJLlmOopQRSWw")
TMUX_ARG_OPTIONS = set("cfLST")
#: a process's environment or arguments as a path, whatever names the process: a pid, `self`, `"$pid"`, `$(pgrep X)`, a glob,
#: or a variable for its whole directory (`for p in /proc/[0-9]*; do cat $p/cmdline`)
ENVIRON = re.compile(r"""/proc/(?:[^\s|;&]|\$\([^)]*\))*?(?:environ|/cmdline)|\$\{?\w+\}?["']?/(?:environ|cmdline)(?![\w.-])""")
SUBSTITUTION = re.compile(r"\$\(([^()]*)\)")
FILTERS = {"grep", "egrep", "fgrep", "rg"}
FALLBACK = re.compile(r"(^|[\s|;&(`])(env|printenv|set|export|declare|typeset)(\s+-\S+)*\s*($|[|;&)>`])|show-?env", re.M)
HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2[ \t]*(?=\n|$)", re.S)

PROCS = "tools/agent-guard/procs.py"
REASON = ("this command would print environment variables' values (a multi-line secret's later lines print even through "
          "`cut -d= -f1`) or a process's command line or environment (tokens sit in some daemons' arguments, and a filter "
          "after the listing keeps the line that holds one). List variable names with `compgen -e` (filter with "
          "`compgen -e | rg NAME`); check one variable with `[ -n \"${NAME:-}\" ] && echo set`. List processes with "
          "`ps -eo pid,stat,etimes,comm`, find one with `pgrep -f PATTERN` and look at it with `ps -o pid,etimes,comm -p PID`. "
          f"For command lines, `python3 {PROCS} PATTERN` prints the matching ones with credentials masked (one process: "
          f"`python3 {PROCS} -p PID`; on another machine: `ssh HOST python3 - PATTERN < {PROCS}`).")


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


#: `ps`'s options that show no arguments and no environment, the only ones it passes on. `ps` reads its arguments twice: as
#: written and, when that reading fails (an option it lacks, a user it doesn't know, two options at odds), all of them over
#: again as BSD's, where `-e` is `e`, the environment, and no format means full command lines (`ps -aux`, `ps -e -x`, `ps -e
#: -T --forest`). So none is here whose failure the second reading survives: not `-c`, `-M`, `-H` or `-T`.
PS_FLAGS = set("AadeNwljyLmZP")
PS_VALUED = set("CDGgopqstUu")
#: the valued ones that stand alone in BSD's reading (`-u` is its `u`): each takes a word of its own, written out, since a
#: value `ps` can't use, or none (`-u $EMPTY`), leaves the BSD listing; and `-D`, which reads a value in its own word as
#: further options (`-Df` is `-D f -f`)
PS_NAMED = set("CDgstu")
PS_LONG = {"cumulative", "deselect", "forest", "header", "headers", "heading", "headings", "help", "info", "no-header",
           "no-headers", "no-heading", "no-headings", "noheader", "noheaders", "noheading", "noheadings", "signames", "version"}
PS_LONG_VALUED = {"Group", "User", "cols", "columns", "date-format", "format", "group", "lines", "pid", "ppid", "quick-pid",
                  "rows", "sid", "sort", "tty", "user", "width"}
#: BSD's options (no dash) that take the rest of their word, else the next word, as their value
PS_BSD_VALUED = set("kopqtU")
#: the columns that hold a process's arguments
PS_ARGV = {"args", "cmd", "command"}


def ps_shows(args: list[str]) -> bool:
    """Whether `ps ARGS` may print a command line or an environment: True unless it surely doesn't. It doesn't when every
    option is one of PS_FLAGS, PS_VALUED, PS_LONG or PS_LONG_VALUED and either each format given (`-o`, `--format`) names no
    column of PS_ARGV, no macro of several columns (`DefBSD`: every one has a capital) and no AIX `%a`, or none is given and
    nothing brings BSD's defaults, whose format ends in the arguments: a BSD option (`ps ax`), a pid after the options (`ps
    1234`), or a `$VARIABLE` or `$(...)`, which may hold several pids (`ps -p $(pgrep X)` is `ps -p 12 34`). BSD's `c` (`ps
    axc`) keeps the name alone; its `e` adds the environment even then, and its `O`, like `-O`, a format that ends in the
    command."""
    formats, bsd, names_only, i = [], False, False, 0
    while i < len(args):
        word = args[i]
        i += 1
        if word.startswith("--"):
            name, sep, value = re.fullmatch(r"--([^=:]*)([=:]?)(.*)", word, re.S).groups()
            if name in PS_LONG:
                continue
            if name not in PS_LONG_VALUED or (not sep and i == len(args)):
                return True
            value, i = (value, i) if sep else (args[i], i + 1)
            bsd |= "$" in value
            formats += [value] if name == "format" else []
        elif re.match(r"-[A-Za-z]", word):
            for at, letter in enumerate(word[1:], 2):
                if letter in PS_FLAGS:
                    continue
                if letter not in PS_VALUED or (at == len(word) and i == len(args)):
                    return True
                value, i = (word[at:], i) if at < len(word) else (args[i], i + 1)
                if letter in PS_NAMED and (not value.isdigit() if at < len(word) else value.startswith("-") or "$" in value):
                    return True
                bsd |= "$" in value
                formats += [value] if letter == "o" else []
                break
        elif re.match(r"[A-Za-z]", word):
            bsd = True
            for at, letter in enumerate(word, 1):
                if letter in "eO":
                    return True
                names_only |= letter == "c"
                if letter in PS_BSD_VALUED:
                    value, i = (word[at:], i) if at < len(word) or i == len(args) else (args[i], i + 1)
                    formats += [value] if letter == "o" else []
                    break
        elif re.match(r"[-+]?\d|\$", word):               # pids, to the last word: no option after them counts
            bsd = True
            break
        else:
            return True
    columns = [re.split(r"[=:]", column)[0] for f in formats for column in re.split(r"[,\s]+", f)]
    if formats:
        return any(c in PS_ARGV or c != c.lower() or ("%" in c and c not in ("%cpu", "%mem")) for c in columns)
    return bsd and not names_only


#: the option of each of these that prints processes' arguments, as (program, letter, long name, the fewest of its
#: characters getopt takes it by)
ARGV_OPTIONS = (("pgrep", "a", "list-full", 6), ("pstree", "a", "arguments", 2), ("top", "c", "cmdline-toggle", 1))
#: xargs's own options that take the next word as their value
XARGS_VALUED = ("adEILnPs", {"arg-file", "delimiter", "max-args", "max-procs", "max-chars", "process-slot-var"})


def asked(given: dict[str, list[str]], letter: str, name: str, least: int) -> bool:
    """Whether the options `given` (`operands`) hold `-LETTER` or `--NAME`, the long one cut as short as `least` characters."""
    return letter in given or any(least <= len(g) > 1 and name.startswith(g) for g in given)


def xargs_command(args: list[str]) -> tuple[list[str], bool]:
    """(the command `xargs ARGS` runs, after xargs's own options; whether xargs puts what it reads in place of a word of it
    (`-I {}`) and not after its last one)."""
    i, replaces = 0, False
    while i < len(args) and args[i].startswith("-") and len(args[i]) > 1:
        word = args[i]
        i += 1
        if word == "--":
            break
        if word.startswith("--"):
            name, valued, _ = word[2:].partition("=")
            replaces |= len(name) > 1 and "replace".startswith(name)
            i += not valued and any(long.startswith(name) for long in XARGS_VALUED[1])
            continue
        for at, letter in enumerate(word[1:], 2):
            replaces |= letter in "Ii"
            if letter in XARGS_VALUED[0] + "eil":           # `-e`, `-i` and `-l` take a value only in their own word
                i += letter in XARGS_VALUED[0] and at == len(word)
                break
    return args[i:], replaces


def ssh_words(args: list[str]) -> list[str]:
    """The words of what `ssh ARGS` runs on the far side (none for a login shell)."""
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i][-1] in SSH_ARG_OPTIONS and len(args[i]) == 2 else 1
    return args[i + 1:]


def refused(words: list[str], fed: bool = False) -> bool:
    """Whether the simple command `words` would print the environment or a process's command line. `fed` says xargs runs it,
    with what it reads as further words after the last."""
    if any(refused_text(w.split("$(", 1)[1]) for w in words if "$(" in w):
        return True
    given = strip_redirections(words)
    words = unwrap(given)
    if not words:
        return False
    name, args = os.path.basename(words[0]), words[1:]
    opts = [a for a in args if a.startswith(("-", "+"))]
    names = [a for a in args if not a.startswith(("-", "+"))]
    if name == "env":
        run = env_command(args)
        if not run:
            return True
        ps_settings = [a for a in args[:len(args) - len(run)] if a.startswith("PS_")]
        return refused_text(run[0]) if len(run) == 1 and " " in run[0] else refused(ps_settings + run)
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
        return refused_text(" ".join(ssh_words(args)))
    if name == "research" and "ssh" in args and "--" in args:
        return refused_text(" ".join(args[args.index("--") + 1:]))
    if name == "ps":                                     # $PS_FORMAT is the format when none is given, $PS_PERSONALITY the reading
        return any(w.startswith("PS_") for w in given[:len(given) - len(words)]) or ps_shows(args + ["$"] if fed else args)
    for program, letter, long, least in ARGV_OPTIONS:
        if name == program:
            options, _ = operands(name, args)
            return asked(options, letter, long, least) or (
                name == "pgrep" and asked(options, "l", "list-name", 6) and asked(options, "f", "full", 1))
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
        run, replaces = xargs_command(args)
        return bool(run) and refused(run, not replaces)
    return False


def refused_text(text: str) -> bool:
    """Whether a command of `text` is `refused`. Each `$(...)` is judged as text of its own, and where it stood is a lone `$`,
    so the command it stands in keeps the words after it: `ps -p $(pgrep X) -o args` is one `ps`."""
    if ENVIRON.search(text):
        return True
    parts = [text]
    while (inner := SUBSTITUTION.search(parts[0])):
        parts[0] = parts[0][:inner.start()] + "$" + parts[0][inner.end():]
        parts.append(inner[1])
    try:
        return any(refused(words) for part in parts for words, _ in words_of(part))
    except ValueError:                                   # unbalanced quotes: judge the raw text
        return bool(FALLBACK.search(text))


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
#: the options of sed and awk whose value is the program itself, not a file that holds it
PROGRAM_GIVEN = ("e", "expression", "source")
#: sed's commands that put a file they name, or the output of a command they run, on sed's own output, with the rest of the
#: line, which names it: `r FILE`, `R FILE` and GNU's `e COMMAND`, after an address or a `;`, never inside a word or as `\r`
SED_READS = re.compile(r"(?<![A-Za-z\\])[rRe][ \t]*([^\n]*)")
#: what in an awk program reads a file or runs a command it names; without one, a name in it is text or a file it writes
AWK_READS = re.compile(r"getline|system|ARGV|\||@include")
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
          "pgrep": ("dFgGOPrstuU", {"delimiter", "pidfile", "pgroup", "group", "older", "parent", "runstates", "session",
                                    "terminal", "euid", "uid", "signal", "cgroup", "ns", "nslist"}),
          "pstree": ("HN", {"highlight-pid", "ns-sort"}),
          "top": ("dEenopUu", {"delay", "scale-summary-mem", "scale-task-mem", "iterations", "sort-override", "pid",
                               "filter-any-user", "filter-only-euser"}),
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


def literals(code: str) -> list[str]:
    """The path-like words of the string literals of `code`."""
    return [w for quoted in re.findall(r"""['"]([^'"]+)['"]""", code) for w in quoted.split() if re.search(r"[/.]", w)]


def program_files(reader: str, programs: list[str], given: dict[str, list[str]]) -> list[str]:
    """The files the sed or awk `programs` themselves name and write to the output: sed's after an `r`, `R` or `e` (SED_READS),
    each word to the end of the line, since `e` runs a command that may name several; awk's, when the program reads at all
    (AWK_READS), the path-like words of its string literals and the values `-v` assigns. A grep's pattern names none."""
    if reader == "sed":
        return [word.strip(";{}") for program in programs for read in SED_READS.finditer(program) for word in read[1].split()]
    if reader == "awk" and any(map(AWK_READS.search, programs)):
        return [word for program in programs for word in literals(program)] + [
            assignment.partition("=")[2] for assignment in given.get("v", []) + given.get("assign", [])]
    return []


def files_printed(reader: str, args: list[str], inputs: list[str]) -> list[str] | None:
    """The files `reader ARGS` writes to its output, `inputs` being what it reads as its own input (`< FILE`), or None when
    it shows no line of them (SILENT, `cmp` without CMP_BYTES, `dd` with an `of=`). A grep's, sed's or awk's first argument
    is its pattern or script, unless an option gave that (SCRIPT_GIVEN), and a sed or awk script is read for the files it
    names itself (`program_files`)."""
    if reader in INPUT_PRINTERS:
        return inputs
    if reader == "dd":
        return None if any(a.startswith("of=") for a in args) else [a[3:] for a in args if a.startswith("if=")] + inputs
    given, rest = operands(reader, args)
    if given.keys() & SILENT.get(reader, set()) or (reader == "cmp" and not given.keys() & CMP_BYTES):
        return None
    first = 1 if reader in SCRIPTED and not given.keys() & SCRIPT_GIVEN else 0
    programs = rest[:first] + [program for option in PROGRAM_GIVEN for program in given.get(option, [])]
    return rest[first:] + inputs + program_files(reader, programs, given)


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
    return literals(args[at + 1]) + args[at + 2:] + inputs


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
    run on this machine, None on another or once a `cd` can't be followed: there a command that reads a file itself (behind
    nothing but `sudo`, `timeout` and the like without options, or in such a `bash -c`) reads one git has committed freely
    (`tracked`). `sunk` says the output of `text` as a whole goes to a command that only unpacks it. Inside `text` a
    command's does when a `|` hands it to one (`unpacks`) that shares the pipe with no other command (no subshell or group),
    or `text`'s own does and the command is piped nowhere, and it is redirected nowhere but to `/dev/null`; so then does the
    output of the command it has another machine run (`far_words`). Only a `tar` that is the command itself, here or there,
    is let through on it."""
    try:
        commands = words_of(text)
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
        words, upstream, opened, after = strip_redirections(words), piped, before.endswith("(<"), before
        run, before = unwrap(words), op
        prog, first, far = os.path.basename(run[0]) if run else "", len(words) - len(run), far_words(run)
        direct = {first, len(words) - len(unwrap(far))} if far else {first}   # where the command stands, here or on the far side
        prose = (prog == "git" and "--no-index" not in run) or (prog == "research" and "--" not in run)
        piped = upstream + piped_names(run) if op in ("|", "|&") and not prose else []
        if prose:
            continue
        if where is not None and prog in ("cd", "pushd", "popd"):   # followed only where the next command runs in the new directory
            followed = (plain and words[0] == "cd" and len(words) == 2 and re.fullmatch(r"[\w./@+][\w./@+-]*", words[1])
                        and op in ("&&", ";") and after in ("", "&&", ";"))
            where = os.path.join(where, words[1]) if followed else None
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
            f"A file git tracks in the checkout, unchanged, reads freely there. {LOCAL_OK} does not lift this.")


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


def local_work_reason(command: str) -> str | None:
    """Why this command may not run on this VM, or None."""
    w = 0 if LOCAL_OK in command else weight(command)
    remote = (f"Run it on a node: `research run --on auto -- CMD` (placed in a CPU node's research pool from this checkout: 4 CPUs, 16 GB "
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
        command = event.get("command") or ""
        conversation = str(event.get("conversation_id") or "")
        root = next(iter(event.get("workspace_roots") or []), "")
        cwd = event.get("cwd") or root or os.getcwd()
    except (ValueError, AttributeError):
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
    low = secret_file_reason(command, stated_cwd(raw)) or low   # ahead of the gates' reasons: the one about a secret is the one to act on
    if deny or low:
        reason = REASON if deny else low
        print(json.dumps({"permission": "deny",
                          "user_message": "Refused by tools/agent-guard/shell_hook.py: " + reason,
                          "agent_message": "Refused by tools/agent-guard/shell_hook.py (.cursor/hooks.json): " + reason}))
    else:
        try:
            poll(conversation, root or cwd)
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
