#!/usr/bin/env python3
"""Cursor's `beforeShellExecution` hook (`.cursor/hooks.json`): refuses a shell command that would print the environment.

Cursor runs every agent shell command through it, on any VM, so it holds where the redacting `env` (`install.sh`) was never
installed. A multi-line secret defeats `env | cut -d= -f1`: its later lines print as if they were names, which is how a line
of the Nebius key reached agents' transcripts at least five times by 6 Oct. Refused, anywhere in the command (pipelines,
`$(...)`, `bash -c`, `ssh HOST CMD`, `research pods ssh POD -- CMD`): `env` or `printenv` with nothing to run, `printenv` of a
secret's name, a bare `set`, `export` or `export -p`, `declare`/`typeset` listing variables, and reading `/proc/*/environ`.
`compgen -e` lists names only and passes. Reads Cursor's hook JSON on stdin and prints its verdict as JSON on stdout.
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
ENVIRON = re.compile(r"/proc/[^\s'\"]*environ")
FALLBACK = re.compile(r"(^|[\s|;&(`])(env|printenv|set|export|declare|typeset)(\s+-\S+)*\s*($|[|;&)>`])", re.M)
HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2[ \t]*(?=\n|$)", re.S)

REASON = ("this command would print environment variables' values, and a multi-line secret's later lines print even "
          "through `cut -d= -f1`. List names with `compgen -e` (filter with `compgen -e | rg NAME`); check one variable "
          "with `[ -n \"${NAME:-}\" ] && echo set`.")


def secret(name: str) -> bool:
    listed = set(re.split(r"[,\s]+", os.environ.get("CLOUD_AGENT_ALL_SECRET_NAMES", ""))) - {""}
    return name in listed or bool(PATTERN.search(name))


def words_of(text: str) -> list[list[str]]:
    """`text`'s simple commands, each as its words, split at pipes, lists, subshells, newlines and backticks."""
    text = HEREDOC.sub(" ", text).replace("`", " ; ").replace("\n", " ; ")
    text = re.sub(r"(?<![\w-])\d+(?=[<>])", " ", text)      # `2>/dev/null`: the fd number is not an argument
    lex = shlex.shlex(text, posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    commands, current = [], []
    for tok in lex:
        if tok and set(tok) <= set("|&;()<>") and (set(tok) & set("()") or not set(tok) & set("<>")):
            commands.append(current)
            current = []
        else:
            current.append(tok)
    commands.append(current)
    return [c for c in commands if c]


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


def refused(words: list[str]) -> bool:
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
    if name == "xargs":
        rest = [a for a in args if not a.startswith("-")]
        return bool(rest) and refused(rest)
    return False


def refused_text(text: str) -> bool:
    if ENVIRON.search(text):
        return True
    try:
        return any(refused(w) for w in words_of(text))
    except ValueError:                                   # unbalanced quotes: judge the raw text
        return bool(FALLBACK.search(text))


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
    if deny:
        print(json.dumps({"permission": "deny",
                          "user_message": "Refused by tools/agent-guard/shell_hook.py: " + REASON,
                          "agent_message": "Refused by tools/agent-guard/shell_hook.py (.cursor/hooks.json): " + REASON}))
    else:
        print(json.dumps({"permission": "allow"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
