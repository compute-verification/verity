"""P6: one CLI over the public API; no ``__main__``, heredocs or ``python -c`` in library code or ``ops/``.

Checked today, against ``allowlists/p06_one_cli.json``:

* ``main-block``: an ``if __name__ == "__main__":`` block, or a ``__main__.py`` module.
* ``argparse``: ``import argparse`` / ``from argparse import ...``.
* ``python-m``: a shell script running ``python -m verity_vllm...`` (keyed by module) instead of ``verity-vllm``.
* ``python-c``: a shell script running ``python -c`` (or ``$PY -c``).
* ``heredoc``: a ``<<`` heredoc in a shell script (keyed by the command it feeds).

Shell checks cover every ``*.sh`` under ``verity_vllm/`` (``ops/`` and the tap build scripts); counts are per script.
``pipeline/cli.py`` does not exist yet, so every ``main-block`` and ``argparse`` is a violation today; once it exists,
it is the one file in ``CLI_OWNERS``.  Not checked: the ``[project.scripts]`` entry that installs the CLI.
"""
from __future__ import annotations

import ast
import re
from functools import lru_cache

from . import _ratchet as R

NAME = "p06_one_cli"
RULE = "P6 (one CLI; no __main__, heredocs or python -c)"
CLI_OWNERS: frozenset[str] = frozenset()      # verity_vllm/pipeline/cli.py, once it exists
PY = r"""(?:\bpython[0-9.]*|\$\{?PY[A-Z0-9_]*\}?|"\$\{?PY[A-Z0-9_]*\}?")"""
PYTHON_M = re.compile(PY + r"""\s+(?:-[A-Za-z]+\s+)*-m\s+(verity_vllm[\w.]*)""")
PYTHON_C = re.compile(PY + r"""\s+(?:-[A-Za-z]+\s+)*-c\b""")
HEREDOC = re.compile(r"""(?<!<)<<-?(?!<)\s*(['"]?)([A-Za-z_][A-Za-z0-9_]*)\1""")


def _is_main_test(test: ast.AST) -> bool:
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
        sides = [test.left, test.comparators[0]]
        return any(isinstance(s, ast.Name) and s.id == "__name__" for s in sides) \
            and any(isinstance(s, ast.Constant) and s.value == "__main__" for s in sides)
    return False


def py_violations(src: str, rel: str) -> list[R.Violation]:
    if rel in CLI_OWNERS:
        return []
    tree = R.parse_src(src)
    out: list[R.Violation] = []
    if rel.endswith("/__main__.py"):
        out.append(R.Violation(rel, "main-block", "<module>", "__main__.py", 1))
    for node, symbol in R.walk(tree):
        if isinstance(node, ast.If) and _is_main_test(node.test):
            out.append(R.Violation(rel, "main-block", symbol, 'if __name__ == "__main__"', node.lineno))
        elif isinstance(node, ast.Import) and any(a.name == "argparse" for a in node.names):
            out.append(R.Violation(rel, "argparse", symbol, "import argparse", node.lineno))
        elif isinstance(node, ast.ImportFrom) and node.module == "argparse" and node.level == 0:
            out.append(R.Violation(rel, "argparse", symbol, "from argparse import", node.lineno))
    return out


def sh_violations(text: str, rel: str) -> list[R.Violation]:
    out: list[R.Violation] = []
    ends: list[str] = []
    for i, line in enumerate(text.splitlines(), 1):
        if ends:
            if line.strip() == ends[0]:
                ends.pop(0)
            continue
        code = line.split("#", 1)[0] if line.lstrip().startswith("#") else line
        if not code.strip():
            continue
        for m in PYTHON_M.finditer(code):
            out.append(R.Violation(rel, "python-m", "<script>", m.group(1), i))
        for _ in PYTHON_C.finditer(code):
            out.append(R.Violation(rel, "python-c", "<script>", "python -c", i))
        for m in HEREDOC.finditer(code):
            if "$((" in code[: m.start()] and "))" in code[m.end():]:
                continue
            head = code[: m.start()].split("|")[-1].split("$(")[-1].strip().split()
            cmd = head[0] if head else "?"
            out.append(R.Violation(rel, "heredoc", "<script>", f"{cmd} <<", i))
            ends.append(m.group(2))
    return out


@lru_cache(maxsize=None)
def scan() -> tuple[R.Violation, ...]:
    out: list[R.Violation] = []
    for p in R.library_files():
        out += py_violations(R.source(p), R.rel(p))
    for p in R.library_files(".sh"):
        out += sh_violations(R.source(p), R.rel(p))
    return tuple(out)


def test_no_new_violations() -> None:
    R.assert_no_new(NAME, scan(), RULE,
                    "Expose the entry point as a `verity-vllm` subcommand over the public API; scripts call "
                    "`verity-vllm`, not python code.")


def test_allowlist_has_no_stale_entries() -> None:
    R.assert_no_stale(NAME, scan(), RULE)


def test_detectors_on_examples() -> None:
    py = "import argparse\ndef main(): pass\nif __name__ == '__main__':\n    main()\n"
    assert sorted(v.kind for v in py_violations(py, "verity_vllm/x.py")) == ["argparse", "main-block"]
    sh = (
        "# $PY -m verity_vllm.not_code in a comment\n"
        "$PY312 -m verity_vllm.pipeline.commit --case B0\n"
        "B=$(python3 -c \"import json;print(1)\")\n"
        "N=$((SNAP_MB << 20))\n"
        "$PY - \"$1\" <<'EOF' >> log\n"
        "x = 1 << 20  # python -c inside a heredoc body is not a script line\n"
        "EOF\n"
        "verity-vllm row R --config row.toml\n"
    )
    got = sorted((v.kind, v.detail) for v in sh_violations(sh, "verity_vllm/ops/x.sh"))
    assert got == [("heredoc", "$PY <<"), ("python-c", "python -c"), ("python-m", "verity_vllm.pipeline.commit")], got
