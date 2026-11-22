"""The one table of domain separators (Lean `Identity.Domains`, which `verity.commitments.domains` reads):
what the module reads is the Lean table, entry for entry, from a checkout and from the wheel's layout, and a name it lacks fails to import; the protocols that read it (``READERS``) carry no literal of it; and every entry is read, or still written somewhere
by an owner whose code doesn't read the table yet."""

from __future__ import annotations

import ast
import importlib.util
import re
import shutil
import subprocess
import tomllib
from fnmatch import fnmatch
from pathlib import Path

import pytest

from verity.commitments import domains

ROOT = Path(__file__).resolve().parents[1]
LEAN = ROOT / "verity/commitments/identity/Identity/Domains.lean"
PYTHON = ROOT / "verity/commitments/domains.py"
#: The protocols whose code reads every separator from the table, Python and Lean, outside their tests and scripts. It
#: only grows: a protocol joins in the change that makes its code read the table. A Lean definition a lock reads keeps
#: its literal until its record next changes (the record hashes the value as written), tied to its entry by an
#: `example : … Domains.NAME … := rfl` in its file (`TIE`).
READERS: tuple[str, ...] = ("verity/protocols/accounting/compute/pouw/verifier",)
SKIPPED_PARTS = {"tests", "scripts"}
#: Files that run on the standard library alone, so keep an entry's bytes under its own name rather than import the table.
STDLIB_COPIES: dict[str, tuple[str, ...]] = {
    "verity/protocols/proof/flock/prover/agreement/agree.py": ("FLOCK_NETLIST",),
}
ABBREV = re.compile(r'abbrev ([A-Z][A-Z0-9_]*) : String :=\s*"([^"\\]*)"')
TIE = re.compile(r"^example : (.*) := rfl$", re.M)
#: the module header every Lean file has (`tools/verity/lean/modules.py`)
MODULE = re.compile(r"^(?:module|@\[expose\] public section)$", re.M)
#: a part of the table, `Identity/Domains/NAME.lean`, which `LEAN` imports ahead of its own entries
PART = re.compile(r"^public import [\w.]+\.([A-Za-z]+)$", re.M)
SCOPE = re.compile(r"^(?:namespace|end) \w+$", re.M)


def lean_files(top: Path = LEAN) -> list[Path]:
    """The Lean table's files in its order: the parts `top` imports, in its order, then `top`."""
    return [top.with_suffix("") / f"{p}.lean" for p in PART.findall(top.read_text())] + [top]


def lean_entries(top: Path = LEAN) -> list[tuple[str, str]]:
    entries = []
    for path in lean_files(top):
        body = re.sub(r"/-.*?-/", "", path.read_text(), flags=re.S)
        body = re.sub(r"--[^\n]*", "", body)
        rest = SCOPE.sub("", PART.sub("", MODULE.sub("", ABBREV.sub("", body))))
        assert not rest.strip(), (f"{path} holds only its module header, the table's parts it imports "
                                  f"and `abbrev NAME : String := \"…\"` lines, found: {rest.strip()[:200]}")
        entries += ABBREV.findall(body)
    return entries


def assert_one_table(lean: list[tuple[str, str]], read: dict[str, str]) -> None:
    assert len({n for n, _ in lean}) == len(lean), "a name appears twice in the Lean table"
    dropped, added = [e for e in lean if e not in read.items()], [e for e in read.items() if e not in lean]
    assert lean == list(read.items()), (
        "verity/commitments/domains.py's reading of the Lean table (Identity/Domains.lean, after the parts it imports from "
        f"Identity/Domains/) differs from the table: it drops {dropped} and adds {added}; make the reader read each entry, "
        "or refuse its line by name")


def test_the_module_reads_the_lean_table():
    assert_one_table(lean_entries(), domains.TABLE)


def _synthetic(tmp_path: Path, extra: str = "", package: str = "Identity", namespace: str = "Domains") -> Path:
    """A table of two files in the Lean's layout, `package/Domains.lean` in `namespace`: a part, then the top, with an
    entry on one line, one on two, a comment block and a line comment."""
    top = tmp_path / f"{package}/Domains.lean"
    (top.parent / "Domains").mkdir(parents=True)
    (top.parent / "Domains/Part.lean").write_text(
        f'module\n\n/-! A part. -/\n\n@[expose] public section\n\nnamespace {namespace}\n\n-- one\n'
        f'abbrev PART_ONE : String := "part/one/v1"\nabbrev PART_TWO : String :=\n  "part/two/v1"\n\nend {namespace}\n')
    top.write_text(f'module\n\npublic import {package}.Domains.Part\n\n/-!\nThe top.\n-/\n\n@[expose] public section\n\n'
                   f'namespace {namespace}\n\nabbrev TOP : String := "top/v1"\n{extra}\nend {namespace}\n')
    return top


@pytest.mark.parametrize(("package", "namespace"), [("Identity", "Domains"), ("Randomness", "Separators")])
def test_the_reader_reads_a_synthetic_table(tmp_path, package, namespace):
    """The reader takes the parts' module from the table's path and the namespace from its file, so moving or renaming
    either needs no change to it."""
    top = _synthetic(tmp_path, package=package, namespace=namespace)
    assert lean_entries(top) == [("PART_ONE", "part/one/v1"), ("PART_TWO", "part/two/v1"), ("TOP", "top/v1")]
    assert_one_table(lean_entries(top), domains._read(top))


def test_an_entry_the_reader_drops_fails_the_table_test(tmp_path):
    """The table gains an entry; a reader that drops it fails `assert_one_table`, naming the entry."""
    top = _synthetic(tmp_path, 'abbrev NEW : String := "new/v1"\n')
    dropping = {n: v for n, v in domains._read(top).items() if n != "NEW"}
    with pytest.raises(AssertionError, match=r"drops \[\('NEW', 'new/v1'\)\]"):
        assert_one_table(lean_entries(top), dropping)


@pytest.mark.parametrize("line", ['abbrev NEW : String := "new/v1" -- a note', 'abbrev NEW : String := "a\\"b"',
                                  'def NEW : String := "new/v1"', '/- a -/ abbrev NEW : String := "new/v1"',
                                  'public import Other.Domains.Part', 'end Other'])
def test_the_reader_refuses_a_line_it_cannot_read(tmp_path, line):
    top = _synthetic(tmp_path, line + "\n")
    with pytest.raises(ValueError, match=r"Domains\.lean:\d+: "):
        domains._read(top)


def test_a_name_the_table_lacks_fails_to_import():
    with pytest.raises(ImportError, match="NOT_A_DOMAIN"):
        from verity.commitments.domains import NOT_A_DOMAIN  # noqa: F401
    with pytest.raises(AttributeError, match="has no entry NOT_A_DOMAIN"):
        _ = domains.NOT_A_DOMAIN
    assert sorted(domains.__all__) == sorted([*domains.TABLE, "TABLE"]) and set(domains.__all__) <= set(dir(domains))


def test_the_wheel_ships_the_table(tmp_path):
    """The wheel holds `verity/` less its excludes and what git ignores (hatch's selection), so the Lean table's files are in
    it; and the module, laid out as the wheel installs it apart from the checkout, reads the table."""
    wheel = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["hatch"]["build"]["targets"]["wheel"]
    shipped = [str(p.relative_to(ROOT)) for p in (PYTHON, *lean_files())]
    assert all(r.split("/")[0] in wheel["packages"] for r in shipped)
    assert not [r for r in shipped for x in wheel.get("exclude", []) if fnmatch(r, x) or fnmatch(r, x + "/*")]
    ignored = subprocess.run(["git", "check-ignore", *shipped], cwd=ROOT, capture_output=True, text=True).stdout
    assert not ignored, f"git ignores these, so the wheel leaves them out: {ignored}"
    for r in shipped:
        (tmp_path / r).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / r, tmp_path / r)
    spec = importlib.util.spec_from_file_location("installed_domains", tmp_path / "verity/commitments/domains.py")
    installed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installed)
    assert installed.TABLE == domains.TABLE and installed._LEAN.is_relative_to(tmp_path)


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
                       "`verity.commitments.domains` (Python) or `Domains` (Lean, `import Identity.Domains.Pouw`; a "
                       "definition a lock reads keeps its literal, tied in its file by `example : … Domains.NAME … := "
                       "rfl`):\n"
                       + "\n".join(found))


def test_a_standard_library_files_copy_is_its_entry():
    unequal = []
    for rel, names in STDLIB_COPIES.items():
        assigned = {t.id: n.value.value for n in ast.parse((ROOT / rel).read_text()).body if isinstance(n, ast.Assign)
                    and isinstance(n.value, ast.Constant) for t in n.targets if isinstance(t, ast.Name)}
        unequal += [f"{rel}: {n} = {assigned.get(n)!r}, the table's is {domains.TABLE.get(n)!r}" for n in names
                    if n not in domains.TABLE or assigned.get(n) != domains.TABLE[n]]
    assert not unequal, "a standard-library file's copy of a table entry differs from the table; copy its bytes:\n" + "\n".join(unequal)


def test_every_entry_is_read_or_still_written():
    files = subprocess.run(["git", "ls-files", "-z", "verity", "adapters/vllm/verity_vllm", "tests/verity", "tests/adapters/vllm/verity_vllm"], cwd=ROOT, check=True, capture_output=True).stdout
    text, table = [], {*lean_files(), PYTHON}
    for rel in files.decode().split("\0"):
        path = ROOT / rel
        if rel and path.suffix in (".py", ".lean", ".c", ".cu", ".cuh", ".h", ".rs", ".json") and path not in table:
            text.append(path.read_text(errors="replace"))
    corpus = "\n".join(text)
    names = set(re.findall(r"\b[A-Z][A-Z0-9_]*\b", corpus))
    unused = [n for n, v in domains.TABLE.items() if n not in names and f'"{v}' not in corpus]
    assert not unused, f"nothing reads these entries and no code writes their bytes; delete them: {unused}"

