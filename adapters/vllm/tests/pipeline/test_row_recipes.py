"""Every recipe that runs a command of the row driver parses: a script's line for `verity-vllm row run`, `row stage`, `epoch ...`
or another of `COMMANDS` is given to the one command-line parser (`cli.parse`) as the shell would give it, so a recipe that an
option's change left stale fails here and not on a pod. A `row run` or `row stage` line is also read by the run store's Tools
(`verity_vllm.pipeline.research_tools.parse`, whose reading keys the stage's derivation), and the two readings name one row, one
checkpoint and one stage.

A recipe is a command in a script of this tree (`SCRIPTS`: a `.sh`, `.yaml`, `.yml` or `.toml` file outside the tests) whose
launcher is `verity-vllm`, `python -m verity_vllm.pipeline.cli` or `"$VV"`, the name the epoch's scripts give the launcher they
write. Its words are the shell's (`shlex`), up to the first operator. A number that touches the redirect after it (`2>/dev/null`) is
that redirect's file descriptor and no word; one a space parts from it (`--config-run 1 > log`) is a word. `${NAME:+WORDS}` gives WORDS
and `${NAME:-WORD}` gives WORD; any other variable gives `STAND_IN`. Words the line doesn't spell (`"$@"`, `"${NAME[@]}"`, `$(...)`) and a line whose quotes
`shlex` can't follow are failures that name the line (`UNREAD`), never a recipe passed over.
"""
from __future__ import annotations

import re
import shlex
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from verity_vllm.pipeline import cli
from verity_vllm.pipeline.research_tools import STAGES
from verity_vllm.pipeline.research_tools import parse as tool_parse

TREE = Path(__file__).resolve().parents[2]
SCRIPTS = (".sh", ".yaml", ".yml", ".toml")
#: the row driver's modules that are commands
MODULES = ("verity_vllm.pipeline.epoch", "verity_vllm.pipeline.launch_context", "verity_vllm.pipeline.research_outputs",
           "verity_vllm.pipeline.research_result", "verity_vllm.pipeline.row", "verity_vllm.pipeline.timeline",
           "verity_vllm.pipeline.verdict_record")
COMMANDS = frozenset(name for name, (module, _) in cli.COMMANDS.items() if module in MODULES)
STAND_IN = "1"

#: a launcher and one of COMMANDS, however the line quotes them: each is a recipe `recipes` must give
MENTION = re.compile(r"(?:verity-vllm|verity_vllm\.pipeline\.cli|\$VV)[\"']?\s+(?:" + "|".join(sorted(map(re.escape, COMMANDS)))
                     + r")(?![\w-])")
_COMMENT = re.compile(r"(^|\s)#\s.*$")
_IF_SET = re.compile(r"\$\{\w+:\+((?:[^{}]|\$\{[^{}]*\})*)\}")
_DEFAULT = re.compile(r"\$\{\w+:-([^{}]*)\}")
_UNSPELLED = re.compile(r"\$[@*]|\$\{\w+\[[@*]\]|\$$")
_VARIABLE = re.compile(r"\$\{[^{}]*\}|\$\w+")
_OPERATOR = re.compile(r"[();<>|&]+")
#: a redirect's file descriptor: a number that is a whole word and touches the operator after it. The first group is text the shell
#: quotes, where a number is never one.
_DESCRIPTOR = re.compile(r"""('[^']*'|"(?:\\.|[^"\\])*"|\\.)|(?<![^\s;|&()<>])\d+(?=[<>])""", re.DOTALL)


class Unread(Exception):
    """A line that names a command of COMMANDS in a shape `recipes` doesn't read."""


def lines(text: str) -> Iterator[tuple[int, str]]:
    """A script's commands, each with its first line's number: a line ending in a backslash goes on in the next, and a line that is
    a comment is none."""
    held, first = "", 0
    for n, line in enumerate(text.split("\n"), 1):
        first = first or n
        if line.endswith("\\"):
            held += line[:-1] + " "
            continue
        if not (held + line).lstrip().startswith("#"):
            yield first, held + line
        held, first = "", 0
    if held:
        yield first, held


def recipes(line: str) -> Iterator[tuple[str, list[str]]]:
    """Each (command, its arguments) of COMMANDS the line runs. A word that holds a whole command (`bash -c '...'`, `"$(...)"`) is
    read as a line."""
    try:
        spelled = _DESCRIPTOR.sub(lambda m: m.group(1) or "", _IF_SET.sub(r"\1", line))
        lex = shlex.shlex(spelled, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        toks = list(lex)
    except ValueError as e:
        raise Unread(f"`shlex` can't split it ({e})") from e
    for i, tok in enumerate(toks):
        if " " in tok and MENTION.search(tok):
            yield from recipes(tok)
        elif (tok.endswith("verity-vllm") or tok in ("verity_vllm.pipeline.cli", "$VV")) and i + 1 < len(toks) and toks[i + 1] in COMMANDS:
            end = next((j for j in range(i + 2, len(toks)) if _OPERATOR.fullmatch(toks[j])), len(toks))
            argv = toks[i + 2:end]
            if unspelled := [a for a in argv if _UNSPELLED.search(a)]:
                raise Unread(f"{' '.join(unspelled)}: words the line doesn't spell")
            yield toks[i + 1], [_VARIABLE.sub(STAND_IN, _DEFAULT.sub(r"\1", a)) for a in argv]


def found() -> tuple[list[tuple[str, int, str, list[str]]], list[str]]:
    """(script, line number, command, arguments) of every recipe in the tree, and each line that names a command and isn't read."""
    out, unread = [], []
    for path in sorted(p for p in TREE.rglob("*") if p.suffix in SCRIPTS and p.is_file() and "tests" not in p.relative_to(TREE).parts):
        where = path.relative_to(TREE).as_posix()
        for n, line in lines(path.read_text(encoding="utf-8")):
            if not (mentions := len(MENTION.findall(_COMMENT.sub("", line)))):
                continue
            try:
                got = list(recipes(line))
            except Unread as e:
                unread.append(f"{where}:{n}: {e}")
                continue
            if len(got) < mentions:
                unread.append(f"{where}:{n}: names {mentions} commands, of which {len(got)} are read")
            out += [(where, n, command, argv) for command, argv in got]
    return out, unread


RECIPES, UNREAD = found()
ROW_RECIPES = [(where, n, argv) for where, n, command, argv in RECIPES if command == "row" and argv[:1] in (["run"], ["stage"])]


def parsed(command: str, argv: list[str], capsys: pytest.CaptureFixture[str]) -> Any:
    try:
        return cli.parse(command, argv, {})
    except SystemExit:
        pytest.fail(f"`verity-vllm {command} {shlex.join(argv)}` is not a command line of this tree: "
                    + capsys.readouterr().err.strip().rpartition("\n")[2])


def test_every_line_that_names_a_command_is_read():
    assert not UNREAD, "teach `recipes` these lines' shapes (an array: where the script fills it):\n" + "\n".join(UNREAD)
    assert ROW_RECIPES and {"epoch", "row"} <= {command for _, _, command, _ in RECIPES}, "the scripts run the row and the epoch's commands"


@pytest.mark.parametrize("line, expected", [
    ('PAIRS=1 "${PY%/*}/verity-vllm" row run "$row" R "$repo" "$rev" --stages build,match > "$CD/$name.log" 2>&1; local rc=$?',
     [("row", ["run", "1", "R", "1", "1", "--stages", "build,match"])]),
    ('read -r A B < <("$VV" epoch job "$RUN" --store-reserve-s "${RESERVE:-2400}" ${END:+--job-deadline "$END"} 2>/dev/null)',
     [("epoch", ["job", "1", "--store-reserve-s", "2400", "--job-deadline", "1"])]),
    ('eval "$("$VV" epoch custody-env "$C")" && "$PY" -m verity_vllm.pipeline.cli epoch word-check "$ROW" "$D" --out "$EV/w.json" \\',
     [("epoch", ["custody-env", "1"]), ("epoch", ["word-check", "1", "1", "--out", "1/w.json"])]),
    ("    bash -c 'cd \"$SRC\" && python -m verity_vllm.pipeline.cli row stage build \"$ROW\" \"$ROLE\" \"$REPO\" \"$REV\" --config-run 1'",
     [("row", ["stage", "build", "1", "1", "1", "1", "--config-run", "1"])]),
    ('stage() {  # stage NAME: `verity-vllm row run --stages NAME` under its deadline', []),
    ('log "ran verity-vllm commit and verity-vllm rows"; "$VV" timeline report "$T" --md "${MD:-}"', [("timeline", ["report", "1", "--md", ""])]),
    ("verity-vllm row stage build R ROLE REPO REV --config-run 1 > log", [("row", ["stage", "build", "R", "ROLE", "REPO", "REV", "--config-run", "1"])]),
    ("verity-vllm row stage build R ROLE REPO REV --config-run 1 2>/dev/null",
     [("row", ["stage", "build", "R", "ROLE", "REPO", "REV", "--config-run", "1"])]),
    ("verity-vllm row stage build R ROLE REPO REV 2>&1 | tee log", [("row", ["stage", "build", "R", "ROLE", "REPO", "REV"])]),
    ('"$VV" epoch job "$RUN" --store-reserve-s 2400 >out 2> err', [("epoch", ["job", "1", "--store-reserve-s", "2400"])]),
    ('"$VV" epoch job "$RUN" --store-reserve-s=2400>out', [("epoch", ["job", "1", "--store-reserve-s=2400"])]),
    ('"$VV" timeline report "$T" --md "a 2>b" 1>out', [("timeline", ["report", "1", "--md", "a 2>b"])]),
    ("bash -c 'verity-vllm row stage build R ROLE REPO REV --config-run 1 2>/dev/null' 2>&1",
     [("row", ["stage", "build", "R", "ROLE", "REPO", "REV", "--config-run", "1"])]),
])
def test_a_line_is_read_as_the_shell_gives_it(line: str, expected: list[tuple[str, list[str]]]):
    assert [r for _, text in lines(line) for r in recipes(text)] == expected


@pytest.mark.parametrize("line", ['"$VV" row run "$ROW" "$ROLE" "$REPO" "$REV" "$@"', 'verity-vllm row stage commit "${ARGS[@]}"',
                                  '"$VV" epoch digests $(ls -d "$BASE"/*) --out "$O"', "verity-vllm row run 'unclosed"])
def test_words_a_line_does_not_spell_are_named_not_passed_over(line: str):
    with pytest.raises(Unread):
        list(recipes(line))


@pytest.mark.parametrize("where, n, command, argv", RECIPES, ids=[f"{where}:{n}" for where, n, _, _ in RECIPES])
def test_a_recipe_parses(where: str, n: int, command: str, argv: list[str], capsys: pytest.CaptureFixture[str]):
    parsed(command, argv, capsys)


@pytest.mark.parametrize("where, n, argv", ROW_RECIPES, ids=[f"{where}:{n}" for where, n, _ in ROW_RECIPES])
def test_the_run_stores_tools_read_a_row_recipe_as_the_parser_does(where: str, n: int, argv: list[str], capsys: pytest.CaptureFixture[str]):
    """The Tool's row, checkpoint and stage are the parser's; a `row run` that names no stages is the Tool's own stage."""
    o = parsed("row", argv, capsys)
    named = o.stage if argv[0] == "stage" else o.stages if any(a.split("=")[0] == "--stages" for a in argv) else None
    for launcher in (["verity-vllm"], ["/venv/bin/python", "-m", "verity_vllm.pipeline.cli"]):
        for stage in STAGES:
            p = tool_parse([*launcher, "row", *argv], stage=stage)
            assert (p["row"], p["role"], p["model"], p["revision"], p["stage"]) == (o.row, o.role, o.repo, o.rev, named or stage)


ROW = "llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager"
NAMED = (ROW, "R", "org/repo", "rev1")


@pytest.mark.parametrize("argv, stage, named", [
    (["verity-vllm", "row", "stage", "match", *NAMED, "--lp", "8"], "match", NAMED),
    (["/venv/bin/python", "-m", "verity_vllm.pipeline.cli", "row", "stage", "commit", *NAMED], "commit", NAMED),
    (["bash", "/tree/run_row_v2.sh", "stage", "match", *NAMED], "match", NAMED),
    (["bash", "/tree/row_pod.sh", *NAMED, "match,commit"], "match,commit", NAMED),
    (["bash", "/tree/tp_stage.sh", *NAMED, "commit"], "commit", NAMED),
    (["bash", "/tree/row_pod.sh", "stage", "match", ROW, "R"], "build", ("stage", "match", ROW, "R")),   # no stage line of its own
    (["verity-vllm", "row", *NAMED, "--stages", "commit"], "commit", NAMED),
    (["verity-vllm", "row", *NAMED], "build", NAMED),
    (["verity-vllm", "row", "stage"], "build", ("stage", None, None, None)),                            # `stage` and no stage after it
])
def test_the_run_stores_tools_read_a_stage_line_and_an_earlier_runners_argv(argv: list[str], stage: str, named: tuple):
    """`row stage STAGE ...` names its stage before the row, as the earlier `run_row_v2.sh stage` did; `row_pod.sh` and `tp_stage.sh`
    named their stages after the revision; any other line's stage is its `--stages`, else the Tool's own (here `build`)."""
    p = tool_parse(argv, stage="build")
    assert (p["stage"], (p["row"], p["role"], p["model"], p["revision"])) == (stage, named)
