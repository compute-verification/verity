"""Ruff's findings on every tracked Python file outside ``archive/`` that the vLLM integration's ruff lint doesn't hold, held to
``allowlists/repo_ruff.json``: ``ruff check`` under the root ``ruff.toml``, with the ruff the root ``dev`` group pins.

The integration's lint (``integrations/vllm/tests/lint/test_ruff.py``) holds ``integrations/vllm`` and ``verity/ml/definitions``
to its own list, and ``archive/`` is kept but not developed (``archive/AGENTS.md``), so neither is held here (``LEFT_OUT``); this
lint holds the rest of ``git ls-files``' ``*.py`` and ``*.pyi``.  A finding is keyed ``(file, kind, symbol, detail)`` as there:
the file relative to the repository, the rule's code, the enclosing ``Class.function`` (``<module>`` at top level) and ruff's
message with ``from line N`` dropped, so edits elsewhere in a file don't churn the list.  An entry allows ``count`` findings of
its key (1 when absent).  A finding the list doesn't allow fails, and so does an entry that allows more than is found, so the
list only shrinks.  ``ruff check --fix PATH`` fixes some (F401, F541, RUF100's stale ``# noqa``); delete the entry each fix
makes stale.

The ratchet gate (``integrations/vllm/tools/ratchet_gate.py``) refuses a change that grows the list, and judges a change to this
module, to ``ruff.toml`` or to the ruff pin by running this lint as the change's base has it on the change's tree.  So it imports
only the standard library and pytest, and ``test_no_new_findings`` takes no fixture.

    python3 tests/lint/test_repo_ruff.py --write   # the list as today's findings, for a move: the gate refuses one that grows it
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

NAME = "repo_ruff"
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "ruff.toml"
LIST = Path(__file__).resolve().parent / "allowlists" / f"{NAME}.json"
#: what the integration's ruff lint holds to its own list (its `ROOT` and `DEFINITIONS`), and the archive
LEFT_OUT = ("integrations/vllm/", "verity/ml/definitions/", "archive/")
LINE_REF = re.compile(r" from line \d+")
FIELDS = ("file", "kind", "symbol", "detail")
FIX = ("Fix the code (`ruff check --fix PATH` fixes some), or, where the rule is wrong for that line, say why in a "
       "`# noqa: CODE` on it. If an allowed finding only moved (a rename, a file move), move its entry in the same change: "
       "`python3 tests/lint/test_repo_ruff.py --write` rewrites the list, and the ratchet gate refuses one that grows.")


@lru_cache(maxsize=None)
def scopes(path: str) -> tuple[tuple[int, int, str], ...]:
    """(first line, last line, qualified name) of every def and class in ``path``, in source order.  A def is a statement, so
    only statement bodies are walked."""
    out: list[tuple[int, int, str]] = []

    def visit(stmts: list[ast.AST], prefix: str) -> None:
        for s in stmts:
            name = prefix
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}.{s.name}" if prefix else s.name
                out.append((s.lineno, s.end_lineno, name))
            for field in ("body", "orelse", "finalbody", "handlers", "cases"):
                visit(getattr(s, field, []), name)

    visit(ast.parse((ROOT / path).read_text(encoding="utf-8")).body, "")
    return tuple(out)


def enclosing(path: str, line: int) -> str:
    """The innermost def or class of ``path`` whose lines hold ``line`` (``<module>`` outside any)."""
    return ([name for first, last, name in scopes(path) if first <= line <= last] or ["<module>"])[-1]


def files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "--", "*.py", "*.pyi"], cwd=ROOT, capture_output=True, check=True).stdout
    return sorted(p for p in out.decode().split("\0") if p and not p.startswith(LEFT_OUT))


@lru_cache(maxsize=None)
def scan() -> tuple[tuple[tuple[str, str, str, str], int], ...]:
    """Each finding's key and line."""
    cmd = [sys.executable, "-m", "ruff", "check", "--config", str(CONFIG), "--no-cache", "--exit-zero", "--output-format",
           "json", "--", *files()]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, (f"ruff check exited {r.returncode} (no ruff? `uv sync --all-packages`: the root `dev` group pins "
                               f"it):\n{r.stderr[-2000:]}")
    out = []
    for f in json.loads(r.stdout):
        path, line, code = Path(f["filename"]).relative_to(ROOT).as_posix(), f["location"]["row"], f["code"] or "syntax-error"
        symbol = "<module>" if code == "syntax-error" else enclosing(path, line)
        out.append(((path, code, symbol, LINE_REF.sub("", f["message"])), line))
    return tuple(out)


def allowed() -> Counter:
    doc = json.loads(LIST.read_text(encoding="utf-8"))
    assert isinstance(doc, dict) and isinstance(doc.get("entries"), list), f"{LIST.name}: {{'entries': [...]}} expected"
    out: Counter = Counter()
    for e in doc["entries"]:
        assert set(FIELDS) <= set(e) <= {*FIELDS, "count"}, f"{LIST.name}: malformed entry {e}"
        k = tuple(e[f] for f in FIELDS)
        assert k not in out, f"{LIST.name}: duplicate entry {e}"
        assert isinstance(e.get("count", 1), int) and e.get("count", 1) >= 1, f"{LIST.name}: count must be a positive int: {e}"
        out[k] = e.get("count", 1)
    return out


def entry_line(key: tuple[str, str, str, str], count: int = 1) -> str:
    return json.dumps({**dict(zip(FIELDS, key)), **({"count": count} if count != 1 else {})}, ensure_ascii=False)


def test_no_new_findings() -> None:
    found, lines, have = Counter(k for k, _ in scan()), {}, allowed()
    for k, line in scan():
        lines.setdefault(k, []).append(line)
    new = [f"  {k[0]}:{','.join(map(str, sorted(lines[k])))} [{k[1]}] {k[2]}: {k[3]}"
           + (f"  ({n} found, {have[k]} allowed)" if have[k] else "") for k, n in sorted(found.items()) if n > have[k]]
    assert not new, (f"ruff (ruff.toml): {len(new)} finding(s) not in tests/lint/allowlists/{NAME}.json. {FIX}\n"
                     + "\n".join(new))


def test_the_list_has_no_stale_entries() -> None:
    found = Counter(k for k, _ in scan())
    stale = [("  delete " if found[k] == 0 else f"  lower count to {found[k]}: ") + entry_line(k, n)
             for k, n in sorted(allowed().items()) if found[k] < n]
    assert not stale, (f"ruff (ruff.toml): {len(stale)} entr(y/ies) in tests/lint/allowlists/{NAME}.json no longer match "
                       "anything: the finding is gone, so delete the entry (the list only shrinks):\n" + "\n".join(stale))


def write() -> None:
    found = Counter(k for k, _ in scan())
    lines = [f"    {entry_line(k, n)}" for k, n in sorted(found.items())]
    LIST.write_text('{\n  "rule": "ruff (ruff.toml)",\n  "entries": [\n' + ",\n".join(lines) + "\n  ]\n}\n", encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit(__doc__.rsplit("\n\n", 1)[-1])
    write()
