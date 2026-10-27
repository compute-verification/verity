"""The one table of domain separators (`verity.ml.commitments.domains`, Lean `Identity.Domains`): the two are equal, entry for
entry; the protocols that read it (``READERS``) carry no literal of it; and every entry is read, or still written somewhere
by an owner whose code doesn't read the table yet."""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

from verity.ml.commitments import domains

ROOT = Path(__file__).resolve().parents[1]
LEAN = ROOT / "verity/core/primitives/identity/Identity/Domains.lean"
PYTHON = ROOT / "verity/ml/commitments/domains.py"
#: The protocols whose code reads every separator from the table, Python and Lean, outside their tests and scripts. It
#: only grows: a protocol joins in the change that makes its code read the table. A Lean definition a lock reads keeps
#: its literal until its record next changes (the record hashes the value as written), tied to its entry by an
#: `example : … Domains.NAME … := rfl` in its file (`TIE`).
READERS: tuple[str, ...] = ("verity/core/protocols/pouw",)
SKIPPED_PARTS = {"tests", "scripts"}
ABBREV = re.compile(r'abbrev ([A-Z][A-Z0-9_]*) : String :=\s*"([^"\\]*)"')
TIE = re.compile(r"^example : (.*) := rfl$", re.M)


def lean_entries() -> list[tuple[str, str]]:
    body = re.sub(r"/-.*?-/", "", LEAN.read_text(), flags=re.S)
    body = re.sub(r"--[^\n]*", "", body)
    rest = ABBREV.sub("", body).replace("namespace Domains", "").replace("end Domains", "")
    assert not rest.strip(), f"{LEAN.relative_to(ROOT)} holds only `abbrev NAME : String := \"…\"` lines, found: {rest.strip()[:200]}"
    return ABBREV.findall(body)


def test_lean_and_python_are_one_table():
    lean = lean_entries()
    assert len({n for n, _ in lean}) == len(lean), "a name appears twice in the Lean table"
    assert lean == list(domains.TABLE.items()), (
        "the Lean table (Identity/Domains.lean) and the Python one (verity/ml/commitments/domains.py) differ: write each "
        "entry in both, in the same order")


def test_entries_are_distinct_and_bare():
    values = list(domains.TABLE.values())
    assert len(set(values)) == len(values), "two names hold the same bytes: keep one"
    bad = {n: v for n, v in domains.TABLE.items() if not v or not v.isascii() or any(c in v for c in '\0"\\ \n')}
    assert not bad, f"an entry is printable ASCII with no terminator, quote, backslash or space: {bad}"


def _python_literals(path: Path):
    tree = ast.parse(path.read_text())
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body
            and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, (str, bytes)) and id(n) not in docs:
            v = n.value.decode("latin-1") if isinstance(n.value, bytes) else n.value
            yield n.lineno, v


CHAR = re.compile(r"'(\\.|[^\\'\n])'")


def _lean_code(text: str, i: int, out: list, interpolated: bool = False) -> int:
    """Scan Lean code from `i`, appending each string literal's (offset, text) to `out`, through the `}` that closes an
    interpolation when `interpolated`; comments, character literals and nested `s!` interpolations are skipped as Lean does."""
    braces = 0
    while i < len(text):
        c = text[i]
        if text.startswith("/-", i):
            depth, i = 1, i + 2
            while depth and i < len(text):
                depth += text.startswith("/-", i) - text.startswith("-/", i)
                i += 2 if text.startswith(("/-", "-/"), i) else 1
        elif text.startswith("--", i):
            end = text.find("\n", i)
            i = len(text) if end < 0 else end
        elif c == "'" and (m := CHAR.match(text, i)) and not (i and (text[i - 1].isalnum() or text[i - 1] in "_'!?")):
            i = m.end()
        elif c == '"':
            i = _lean_string(text, i, out)
        elif interpolated and c == "{":
            braces, i = braces + 1, i + 1
        elif interpolated and c == "}":
            if not braces:
                return i + 1
            braces, i = braces - 1, i + 1
        else:
            i += 1
    return i


def _lean_string(text: str, i: int, out: list) -> int:
    start, s, i = i, [], i + 1
    interpolating = text[start - 2:start] == "s!"
    while text[i] != '"':
        if text[i] == "\\":
            s.append("\0" if text[i + 1] == "0" else text[i:i + 2])
            i += 2
        elif interpolating and text[i] == "{":
            s.append("\ufffd")
            i = _lean_code(text, i + 1, out, interpolated=True)
        else:
            s.append(text[i])
            i += 1
    out.append((start, "".join(s)))
    return i + 1


def _lean_literals(path: Path):
    text, out = path.read_text(), []
    _lean_code(text, 0, out)
    tied = {domains.TABLE.get(n) for t in TIE.findall(text) for n in re.findall(r"\bDomains\.([A-Z][A-Z0-9_]*)", t)}
    for offset, s in out:
        if s.rstrip("\0") not in tied:
            yield text.count("\n", 0, offset) + 1, s


def test_readers_carry_no_literal_of_the_table():
    by_value = {v: n for n, v in domains.TABLE.items()}
    found = []
    for reader in READERS:
        for path in sorted((ROOT / reader).rglob("*")):
            rel = path.relative_to(ROOT)
            if SKIPPED_PARTS & set(rel.parts) or ".lake" in rel.parts or path.suffix not in (".py", ".lean"):
                continue
            literals = _python_literals(path) if path.suffix == ".py" else _lean_literals(path)
            found += [f"{rel}:{line}: {v!r} is `{by_value[v.rstrip(chr(0))]}`" for line, v in literals
                      if v.rstrip("\0") in by_value]
    assert not found, ("a protocol that reads the table writes a separator of it as a literal; read it from "
                       "`verity.ml.commitments.domains` (Python) or `Domains` (Lean, `import Identity.Domains`; a "
                       "definition a lock reads keeps its literal, tied in its file by `example : … Domains.NAME … := "
                       "rfl`):\n"
                       + "\n".join(found))


def test_every_entry_is_read_or_still_written():
    files = subprocess.run(["git", "ls-files", "-z", "verity"], cwd=ROOT, check=True, capture_output=True).stdout
    text = []
    for rel in files.decode().split("\0"):
        path = ROOT / rel
        if rel and path.suffix in (".py", ".lean", ".c", ".cu", ".cuh", ".h", ".rs", ".json") and path not in (LEAN, PYTHON):
            text.append(path.read_text(errors="replace"))
    corpus = "\n".join(text)
    names = set(re.findall(r"\b[A-Z][A-Z0-9_]*\b", corpus))
    unused = [n for n, v in domains.TABLE.items() if n not in names and f'"{v}' not in corpus]
    assert not unused, f"nothing reads these entries and no code writes their bytes; delete them: {unused}"
