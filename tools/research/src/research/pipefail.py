"""A workload's shell pipe without `pipefail`: such a script exits with its pipe's last command's status (often `tee`'s), so a
run ends `done`, rc 0, whatever failed before it. `research run` refuses one (`refusal`) unless `--allow-pipe` says the pipe's
own exit code is meant.

The shell is found anywhere in the command, behind any wrapper (`env K=V`, `nice -n 19`, `timeout`, `uv run`, slot.py ... --):
the first word named bash, sh, dash or zsh (by any path). Only one given a script with `-c` is judged. Its script is
tokenized as a shell does (`shlex` with punctuation, quotes kept whole), so `||`, `>|` and a `|` inside quotes, a word, a
`case` pattern, `[[ ]]`, `(( ))` or a here-document's body are no pipe, and `|&` is one. The script is safe when the shell is
started with `-o pipefail`, or a `set` command turns it on (`set -o pipefail`, `set -euo pipefail`) before the first pipe.
A script shlex can't parse (bash nests quotes inside `"$( )"`, shlex doesn't) is refused only if it holds a bare `|` with
no `set ... -o pipefail` before it. Stdlib only.
"""
from __future__ import annotations

import io
import os
import re
import shlex

SHELLS = frozenset({"bash", "sh", "dash", "zsh"})
#: a shell's long options that take an argument
_LONG_WITH_ARG = frozenset({"--rcfile", "--init-file"})
#: shell operators, longest first, to split a run of punctuation shlex returns as one token (`|(`, `;\n`)
_OPS = (";;&", "<<<", "<<-", "&>>", "||", "|&", "&&", ";;", ";&", ">|", ">>", ">&", "<&", "<<", "<>", "&>", "((", "))")
_SEPARATORS = frozenset({";", "&", "&&", "||", "\n", "(", ")", "((", "))", "|", "|&", ";;", ";&", ";;&"})
_COMMAND_START = frozenset({"then", "do", "else", "elif", "{", "}", "!", "if", "while", "until", "time"})
_HEREDOC = re.compile(r"(?<!<)<<(-?)[ \t]*(?:'([^'\n]*)'|\"([^\"\n]*)\"|\\?([A-Za-z_0-9][\w.-]*))")
_BARE_PIPE = re.compile(r"(?<!\|)\|(?!\|)")
_SET_PIPEFAIL = re.compile(r"\bset\b[^\n;|&]*?\s-[a-zA-Z]*o[a-zA-Z]*\s+pipefail\b")

CURE = ("start the script with `set -o pipefail;` (or invoke the shell with `-o pipefail`), or drop the `| tee FILE` (the run "
        "already keeps the workload's stdout in its log), or pass --allow-pipe when the pipe's own exit code is meant")


def refusal(command: list[str]) -> str | None:
    """Why `research run` refuses ``command`` (a shell script that pipes without `pipefail`), or None."""
    found = shell_script(command)
    if found is None:
        return None
    shell, script, pipefail = found
    if pipefail or not unsafe_pipe(script):
        return None
    return (f"the workload's {shell} script pipes without `pipefail`, so the run would end with the exit code of the pipe's "
            f"last command, whatever failed before it: {CURE}")


def shell_script(command: list[str]) -> tuple[str, str, bool] | None:
    """The first shell in ``command`` with its `-c` script: (its name, the script, whether it was started with `-o pipefail`).
    None when no shell is there, or the first one runs a file or reads stdin."""
    for at, word in enumerate(command):
        if os.path.basename(word) in SHELLS:
            return _invocation(os.path.basename(word), command[at + 1:])
    return None


def _invocation(shell: str, args: list[str]) -> tuple[str, str, bool] | None:
    has_c = pipefail = False
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("--", "-"):
            i += 1
            break
        if a.startswith("--"):
            i += 2 if a in _LONG_WITH_ARG else 1
            continue
        if len(a) < 2 or a[0] not in "-+":
            break
        i += 1
        for ch in a[1:]:
            if ch in "oO" and i < len(args):
                pipefail = pipefail or (ch == "o" and a[0] == "-" and args[i] == "pipefail")
                i += 1
            elif ch == "c":
                has_c = True
    if not has_c or i >= len(args):
        return None
    return shell, args[i], pipefail


def unsafe_pipe(script: str) -> bool:
    """Whether ``script`` pipes before any `set` command turns `pipefail` on."""
    try:
        tokens = _tokens(_without_heredoc_bodies(script))
    except ValueError:
        pipe = _BARE_PIPE.search(script)
        return bool(pipe) and not _SET_PIPEFAIL.search(script, 0, pipe.start())
    prev: str | None = None
    case: list[str] = []          # per open `case`: "word" (before its `in`), "pattern" or "body"
    test = arith = 0
    in_set = False
    for k, (tok, op) in enumerate(tokens):
        at_start = prev is None or prev in _SEPARATORS or prev in _COMMAND_START or (case and case[-1] == "body" and prev == ")")
        if op:
            if test or arith:
                arith -= tok == "))" and arith > 0
            elif case and case[-1] == "pattern":
                if tok == ")":
                    case[-1] = "body"
            elif case and case[-1] == "body" and tok in (";;", ";&", ";;&"):
                case[-1] = "pattern"
            elif tok in ("|", "|&"):
                return True
            elif tok == "((":
                arith += 1
            if tok in _SEPARATORS:
                in_set = False
        elif test:
            test -= tok == "]]"
        elif case and case[-1] == "word":
            if tok == "in":
                case[-1] = "pattern"
        elif case and case[-1] == "pattern" and tok == "esac":
            case.pop()
        elif case and case[-1] == "body" and tok == "esac" and at_start:
            case.pop()
        elif at_start and tok == "case":
            case.append("word")
        elif at_start and tok == "[[":
            test += 1
        elif at_start and tok == "set":
            in_set = True
        elif in_set and tok.startswith("-") and not tok.startswith("--"):
            names = tokens[k + 1:k + 1 + tok.count("o")]
            if any(not o and n == "pipefail" for n, o in names):
                return False
        prev = tok
    return False


class _Lexer(shlex.shlex):
    """shlex that says whether its last token was an operator: a run of punctuation it met unquoted (its state `c`), where a
    quoted `'|'` comes back as the same text."""

    operator = False

    @property
    def state(self) -> str | None:
        return self._state

    @state.setter
    def state(self, value: str | None) -> None:
        self._state = value
        self.operator = self.operator or value == "c"


class _Lines(io.StringIO):
    """A comment is read with `readline`; this one leaves the newline, which ends the command, to be read."""

    def readline(self, size: int = -1) -> str:
        line = super().readline(size)
        if line.endswith("\n"):
            self.seek(self.tell() - 1)
            return line[:-1]
        return line


def _tokens(script: str) -> list[tuple[str, bool]]:
    """``script``'s words and operators, each (text, is an operator); a newline is an operator, as it separates commands."""
    lex = _Lexer(_Lines(script), posix=True, punctuation_chars="();<>|&\n")
    lex.whitespace = " \t\r"
    lex.whitespace_split = True
    out: list[tuple[str, bool]] = []
    while True:
        lex.operator = False
        tok = lex.get_token()
        if tok is None:
            return out
        out += [(op, True) for op in _split_ops(tok)] if lex.operator else [(tok, False)]


def _split_ops(run: str) -> list[str]:
    out: list[str] = []
    while run:
        op = next((o for o in _OPS if run.startswith(o)), run[0])
        out.append(op)
        run = run[len(op):]
    return out


def _without_heredoc_bodies(script: str) -> str:
    """``script`` with each here-document's body dropped, its operator and delimiter kept: a body is data, not shell."""
    out: list[str] = []
    ends: list[tuple[str, bool]] = []
    for line in script.split("\n"):
        if ends:
            word, tabs = ends[0]
            if (line.lstrip("\t") if tabs else line) == word:
                ends.pop(0)
            continue
        out.append(line)
        for m in _HEREDOC.finditer(line):
            ends.append((next(g for g in m.groups()[1:] if g is not None), m[1] == "-"))
    return "\n".join(out)
