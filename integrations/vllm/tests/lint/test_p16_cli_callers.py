"""P16: every CLI command has a caller.

A command in ``verity_vllm.pipeline.cli.COMMANDS`` is called when one of these runs it:

* code: an argv with ``"verity_vllm.pipeline.cli", "NAME"`` or ``"verity-vllm", "NAME"`` side by side (a stage, or a process
  a module starts, its own module included), or a stage driver's ``cli("NAME", ...)``;
* a script: a line of a ``.sh``, ``.toml`` or ``.yaml`` file that isn't a comment, or a string in another module's code (not
  a docstring), reading ``verity-vllm NAME`` or ``verity_vllm.pipeline.cli NAME``, options allowed in between;
* a public command: one run from outside ``integrations/vllm``, which this suite doesn't read, that the ``public`` table of
  ``allowlists/p16_cli_callers.json`` names with the file that runs it.  It is a declaration rather than an exception, as the
  environment-read lint's settings-registry marks are, so no ratchet counts it: the ratchet gate
  (``integrations/vllm/tools/ratchet_gate.py``) accepts one a change adds, or gives another caller, when that file, in the
  change's tree, runs the command (``runs``, as this module's base copy finds a call), and refuses it otherwise.

Scanned: ``integrations/vllm`` outside ``tests`` directories, ``test_*`` files and ``cli.py`` (whose table names every
command). A test is not a caller, and neither is Markdown or a command's own help text.  Checked today, against
``allowlists/p16_cli_callers.json``: an ``uncalled-command``, keyed by the file of the module it runs.  The gate loads this
module from a change's base without running a test, so it imports only the standard library and reads no file as it loads.
"""
from __future__ import annotations

import ast
import json
import re
from collections.abc import Collection
from functools import lru_cache
from pathlib import Path, PurePosixPath

from . import _ratchet as R

NAME = "p16_cli_callers"
RULE = "P16 (every CLI command has a caller)"
CLI = R.PKG / "pipeline" / "cli.py"
LAUNCHERS = frozenset({"verity-vllm", "verity_vllm.pipeline.cli"})
SCRIPTS = frozenset({".sh", ".toml", ".yaml", ".yml"})
PUBLIC = "public"


@lru_cache(maxsize=None)
def public_commands() -> dict[str, str]:
    """The list's ``public`` table: each command run from outside integrations/vllm, with the file that runs it."""
    table = json.loads((R.ALLOWLISTS / f"{NAME}.json").read_text(encoding="utf-8")).get(PUBLIC, {})
    assert isinstance(table, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in table.items()), \
        f"allowlists/{NAME}.json: {PUBLIC!r} is a table of command -> the file that runs it: {table}"
    return table


@lru_cache(maxsize=None)
def commands() -> dict[str, str]:
    """``COMMANDS``: name -> the module it runs."""
    for node in R.parse(CLI).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "COMMANDS":
            return {k.value: v.elts[0].value for k, v in zip(node.value.keys, node.value.values)}
    raise AssertionError(f"{R.rel(CLI)}: no COMMANDS table")


@lru_cache(maxsize=None)
def invocation(names: frozenset[str]) -> re.Pattern:
    alt = "|".join(sorted(map(re.escape, names), key=len, reverse=True))
    return re.compile(r"(?:\bverity-vllm|verity_vllm\.pipeline\.cli)[\"']?\s+(?:--?\w[\w-]*\s+)*(" + alt + r")(?![\w-])")


def py_callers(src: str, names: frozenset[str], own: Collection[str] = ()) -> set[str]:
    """The commands a module's code runs; ``own``: the commands that run this module, whose names in its strings don't count."""
    tree = R.parse_src(src)
    docs = R.docstring_ids(tree)
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)):
            vals = [e.value if R.is_str(e) else None for e in node.elts]
            out |= {b for a, b in zip(vals, vals[1:]) if a in LAUNCHERS and b in names}
        elif isinstance(node, ast.Call) and node.args and R.is_str(node.args[0]) and node.args[0].value in names \
                and (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)) == "cli":
            out.add(node.args[0].value)
        elif R.is_str(node) and id(node) not in docs:
            out |= {m.group(1) for m in invocation(names).finditer(node.value)} - set(own)
    return out


def script_callers(text: str, names: frozenset[str]) -> set[str]:
    rx = invocation(names)
    return {m.group(1) for line in text.splitlines() if not line.lstrip().startswith("#") for m in rx.finditer(line)}


def runs(path: str, text: str, command: str) -> bool:
    """Whether the file ``path``, holding ``text``, runs ``command`` as a scanned file would: a public command's caller."""
    suffix, names = PurePosixPath(path).suffix, frozenset({command})
    if suffix == ".py":
        return command in py_callers(text, names)
    return suffix in SCRIPTS and command in script_callers(text, names)


def module_file(module: str) -> str:
    p = R.PKG.parent / Path(*module.split("."))
    return R.rel(p / "__init__.py") if p.is_dir() else R.rel(p.with_suffix(".py"))


def scanned() -> list[Path]:
    out = []
    for p in sorted(R.ROOT.rglob("*")):
        parts = p.relative_to(R.ROOT).parts
        if p.suffix in SCRIPTS | {".py"} and p != CLI and p.is_file() and not p.name.startswith("test_") \
                and not any(d in R.SKIP_DIRS or d.startswith(".") for d in parts[:-1]):
            out.append(p)
    return out


@lru_cache(maxsize=None)
def scan() -> tuple[R.Violation, ...]:
    cmds = commands()
    names = frozenset(cmds)
    by_module: dict[str, list[str]] = {}
    for c, m in cmds.items():
        by_module.setdefault(module_file(m), []).append(c)
    called: set[str] = set(public_commands())
    for p in scanned():
        if p.suffix == ".py":
            called |= py_callers(R.source(p), names, by_module.get(R.rel(p), ()))
        else:
            called |= script_callers(R.source(p), names)
    return tuple(R.Violation(module_file(m), "uncalled-command", "<module>", c) for c, m in sorted(cmds.items()) if c not in called)


def test_no_new_violations() -> None:
    R.assert_no_new(NAME, scan(), RULE,
                    "A command nothing runs is dead: delete it from COMMANDS (and its module, if nothing else imports it), "
                    "or run it from the stage or script it belongs to. A command run from outside integrations/vllm goes "
                    f"in the {PUBLIC!r} table of tests/lint/allowlists/{NAME}.json with the file that runs it, which the "
                    "ratchet gate checks.")


def test_allowlist_has_no_stale_entries() -> None:
    R.assert_no_stale(NAME, scan(), RULE)


def test_public_commands_are_commands() -> None:
    gone = sorted(set(public_commands()) - set(commands()))
    assert not gone, f"{RULE}: allowlists/{NAME}.json's {PUBLIC!r} table names commands COMMANDS no longer has; delete them: " \
                     f"{gone}"


def test_detectors_on_examples() -> None:
    names = frozenset({"row", "row-tp", "build", "advice", "glue", "twins", "served"})
    src = (
        "'''Run `verity-vllm twins` by hand.'''\n"
        "import sys\n"
        "ADVICE = [sys.executable, '-m', 'verity_vllm.pipeline.cli', 'advice']\n"
        "def stage(self):\n"
        "    '''`verity-vllm served` is only prose here.'''\n"
        "    self.cli('build', '--model', 'm')\n"
        "    return f'run `verity-vllm --quiet row-tp {self.x}` first'\n"
        "HELP = 'see verity-vllm glue'\n"
    )
    assert py_callers(src, names) == {"advice", "build", "row-tp", "glue"}
    assert py_callers(src, names, own=("glue", "build")) == {"advice", "build", "row-tp"}
    sh = ('# verity-vllm twins (a comment)\n'
          '"${PY%/*}/verity-vllm" row --x 1\n'
          'python -m verity_vllm.pipeline.cli served-pin x\n'
          'verity-vllm rows\n')
    assert script_callers(sh, names) == {"row"}
    assert runs("benchmarks/x/run.sh", sh, "row") and not runs("benchmarks/x/run.sh", sh, "twins")
    assert runs("benchmarks/x/e2e.py", src, "advice") and not runs("benchmarks/x/e2e.py", src, "twins")
    assert not runs("benchmarks/x/README.md", sh, "row")
