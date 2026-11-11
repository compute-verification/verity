"""The test-file family: one task per test file of every suite, which `runners/test_file.py` runs alone under the suites' read guard.

Suites are what `tools/verity/check/suites.py` finds (`discover`): each directory whose `pyproject.toml` configures pytest, the
repository's own at the root, but those check runs after its Lean steps (`build.lean_suites`, `suites.AFTER_LEAN`), which the
build family's lean-suites tasks run whole once lean-build has built what they need. A suite's test files are its `test_*.py`
under its `testpaths`, outside a data directory
(`DATA_DIRS`) and the suites nested in it (`suites.test_files`). A task's inputs are what running its one file reads, found from
the commit's files alone:

- the test file and each `conftest.py` from its directory up to its suite's, with the `__init__.py` files pytest's import of them
  runs (its default `prepend` mode: a module in a package is imported from the first directory above it that isn't one), and
  each `__init__.py` from its directory up to its suite's (pytest imports a Package node's to set it up);
- their import closure over the repository's Python (`Imports`): each `import` and `from ... import` in a file, wherever it is
  (a function's, a `TYPE_CHECKING` block's), relative ones from the file's package, absolute ones where the workspace's packages
  put their names (`Layout`), where pytest puts them (each imported file's directory above its packages, the suite's
  `pythonpath`), where a file of the closure puts them on `sys.path` (`_Paths`: a directory it builds from `__file__`, string
  parts and its module-level names; where it builds one otherwise, the directories it builds anywhere and those above it serve
  its own imports), and a name none has from the importing file's own directory; a constant name given to `import_module`,
  `__import__`, `importorskip` or `run_module`, and each module of the package an f-string's constant head or a module's
  `__name__` names (`f"verity.ml.registry.{m.name}"`), given or assigned to the name given; a dotted target given to
  `monkeypatch.setattr` or `mock.patch`, which imports its module; a conftest's `pytest_plugins`; a `.py` file a path loader is given
  that the call's strings name, in one string or in parts joined by `/` (from the file's directory or one above it), and
  where the path is a variable (a helper's parameter), each `.py` file the file's strings name; where it builds the path from
  a name (`_load("gamma")`), each `.py` file of its directory a string of it names; each module of a package that imports them
  all (`pkgutil.iter_modules(__path__)`); and the settings registry `verity_config.settings.of(__file__)` loads. Another
  computed name can't be followed (`Info.unfollowed`); what it loads is an undeclared read when the runner meets it;
- the closures of what the test file and its conftests start or load: each `.py` file a string in them names (a script a test
  runs), each module they run as `python -m NAME`, each absolute import a string of them starts or holds after a newline or a
  `;` (the program they hand `python -c`), and where they load a module by a path they build, each name a string gives that a
  `.py` file has there or above;
- the data beside the closure's modules (`Tree.beside`): each one's directory's files that aren't Python, and the files under
  its directories that hold no Python (`tables/`); and every pyproject.toml when a module of it lists them all
  (`suites.metadata`) or reads each workspace member's (its strings hold `members` and `pyproject.toml`: `research.pythonpath`);
- the files under each path the test file, its conftests and the helpers it imports (a module of the closure under the suite's
  test paths that is no `test_*.py`) name: string parts joined by `/` or one string holding one, and parts joined to a path
  they build (`VERIFIER / "agree.py"`);
- the suite's declared inputs (`[tool.verity.tests] inputs`, a directory as its files) less their Python, which the closure has
  where it is imported, but for the whole tree (`.`), and its own directory's files but their Python and the suites nested in
  it (`suites.inputs_of` always holds it); less Markdown under a directory unless the suite lists a `.md` (`suites.markdown`);
  the fixture registry when an entry of it that git doesn't hold (a fetched fixture) is under the suite's directory or its
  inputs;
- what its suite's `[[tool.verity.tests.reads]]` say it reads whole (`walks`): each entry names test files (fnmatch patterns
  from the suite's directory) and `paths` (from the root: a file, a directory as its files, or an fnmatch pattern), Python
  included, for what no rule above follows: a tree a test walks, a module it loads by a name it computes, what a program it
  starts reads. An entry that names no test file or a path that matches no file is refused;
- for a test file its suite's `[tool.verity.tests] untraced` names (fnmatch patterns from the suite's directory), one that
  starts a program the guard can't trace (git, lake, a compiler, a shell, a Python run with `-I`), what `suites.py` keys the
  suite on (`whole`), with `vars` `untraced` = `suite`; a task whose closure holds a module that builds native code (`_NATIVE`)
  has `untraced` = `compilers`, since a compiler reads the sources beside it; the runner makes any other untraced program an
  error;
- each `pyproject.toml` from the test file's directory up to the root;
- the read guard the runner loads into pytest (`GUARD`); the core adds the runner itself.

A task names its suite's directory in `vars` (`suite`), where the runner runs pytest. Its memory is the suite's `[tool.verity.tests]
worker_gb` and its timeout the suite's `timeout_s` where it declares them, else `[family.test-file]`'s.
"""
from __future__ import annotations

import ast
import bisect
import fnmatch
import json
import math
import posixpath
import re
import sys
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

KIND = "test-file"
#: the read guard the runner loads into pytest, the one `suites.py` loads
GUARD = ("tools/verity/check/guard/verity_suite_guard.py", "tools/verity/check/guard/child/sitecustomize.py",
         "tools/verity/check/guard/child/verity_suite_reads.py")
REGISTRY = "fixtures/artifacts.json"
#: directories whose `test_*.py` files are data a test hands on, not tests (`suites._DATA_DIRS`)
DATA_DIRS = frozenset({"fixtures", "data", "expected", "testdata"})
STDLIB = frozenset(sys.stdlib_module_names)

#: the start of an import statement on a line of its own (or after `try:`, `else:`, `finally:`), read to the end of its logical
#: line (`_statement`): parsing every file whole takes 20 s for the tree, reading its import statements well under one
_STMT = re.compile(r"^[ \t]*(?:(?:try|else|finally)[ \t]*:[ \t]*)?(from[ \t]+\.*[\w.]*[ \t]+import\b|import[ \t]+(?=[\w.]))", re.M)
_NAME = r"[A-Za-z_]\w*"
_DOTTED = rf"{_NAME}(?:\s*\.\s*{_NAME})*"
_AS = rf"(?:\s+as\s+{_NAME})?"
_IMPORT = re.compile(rf"import\s+({_DOTTED}{_AS}(?:\s*,\s*{_DOTTED}{_AS})*)\s*")
_FROM = re.compile(rf"from\s+(\.*)\s*({_DOTTED})?\s+import\s+(\*|\(?\s*{_NAME}{_AS}(?:\s*,\s*{_NAME}{_AS})*\s*,?\s*\)?)\s*")
_AS_SPLIT = re.compile(r"\s+as\s+")
#: calls that import a module by its name, and those that load one from its path
_BY_NAME = ("import_module", "__import__", "importorskip", "run_module")
_CALLS = (*_BY_NAME, "spec_from_file_location", "run_path", "SourceFileLoader")
_CALL = re.compile(rf"\b({'|'.join(_CALLS)})[ \t]*\(")
_CONSTANT = re.compile(r"""\s*[rRuU]?(['"])([\w.]+)\1\s*[,)]""")
#: a string naming a `.py` file, and a `.py` path of string parts (`ROOT / "tools" / "check.py"`)
_PYFILE = re.compile(r"""["']([\w./-]*[\w-]\.py)["']""")
_PARTS = re.compile(r"""(?:["'][\w.-]+["']\s*/\s*)+["'][\w.-]*[\w-]\.py["']""")
#: a module a test runs as `python -m NAME`, in a command's list
_DASH_M = re.compile(r"""["']-m["']\s*,\s*[rRuU]?["']([A-Za-z_][\w.]*)["']""")
#: an import statement a string starts, or one after a `\n` or a `;` in it: the program a test hands `python -c`
_CODE_IMPORT = re.compile(r"""(?:["']|\\n|;)[ \t]*(?:from[ \t]+([A-Za-z_][\w.]*)[ \t]+import[ \t]+([A-Za-z_][\w \t,]*)"""
                          r"""|import[ \t]+([A-Za-z_][\w.]*(?:[ \t]*,[ \t]*[A-Za-z_][\w.]*)*))""")
#: a loader's path built from a name (`HERE / f"{name}.py"`, `name + ".py"`), and the names a file's strings give
_BUILT = re.compile(r"""[fF][rR]?["'][^"'\n]*\{[^}]+\}\.py["']|\+\s*["']\.py["']""")
_BARE = re.compile(r"""["']([A-Za-z_]\w*)["']""")
_PLUGINS = re.compile(r"^pytest_plugins[ \t]*=", re.M)
#: a module name built in an f-string (`f"verity.definitions.ops.boolean.{m.name}"`, `f"{__name__}.{n}"`), or a name holding one
_FSTRING = re.compile(r"""\s*[fF](['"])([^'"\n]*\{[^'"\n]*)\1""")
_ARG_NAME = re.compile(r"\s*([A-Za-z_]\w*)\s*[,)]")
#: a dotted target string handed to pytest's `monkeypatch.setattr` or to `mock.patch`, which imports the module it names
_PATCH = re.compile(r"""\b(?:setattr|delattr|patch)\(\s*[rRuU]?["']([A-Za-z_]\w*(?:\.\w+)+)["']""")
#: a path joined of string parts to a name or `Path(__file__)` (`HERE.parent / "harness"`)
_BUILT_PATH = re.compile(r"""(?:\bPath\(__file__\)|\b[A-Za-z_]\w*)(?:\.\w+|\[\d+\]|\(\))*(?:\s*/\s*["'][\w.-]+["'])+""")
_QUOTED_SLASH = re.compile(r"""/\s*["']""")
#: a call that puts a directory on `sys.path`
_SYSPATH = re.compile(r"\bsys\.path\.(?:insert|append|extend)\(|\bsys\.path\[[^\]\n]*\]\s*=|\bsyspath_prepend\(|\baddsitedir\(")
#: the calls and methods a path expression passes its path through, and those taking its directory
_KEEP = frozenset({"Path", "PurePath", "PurePosixPath", "str", "fspath", "abspath", "realpath", "normpath"})
_SAME = frozenset({"resolve", "absolute", "expanduser"})
#: a repository path a test names: string parts joined by `/` (`ROOT / "verity" / "core"`), or one string holding a `/`
_DATA_PARTS = re.compile(r"""(?:["'][\w.-]+["']\s*/\s*)+["'][\w.-]+["']""")
_SLASHED = re.compile(r"""["']([\w.-]+(?:/[\w.-]+)+)/?["']""")
#: a module that lists every pyproject.toml of the tree (`suites.metadata`), or reads each workspace member's from the root's
#: `[tool.uv.workspace] members` (`research.pythonpath`)
_METADATA = re.compile(r"""["']\*/pyproject\.toml["']""")
_MEMBERS = re.compile(r"""["']members["']""")
_PYPROJECT = re.compile(r"""["']pyproject\.toml["']""")
#: a module that builds native code: it runs a compiler the guard can't trace whenever its build cache misses, on the sources
#: beside it (`Tree.beside`), so a task whose closure holds one lets its runs start a compiler (`vars` `untraced` = `compilers`)
_NATIVE = re.compile(r"""cpp_extension|load_inline|["'](?:g\+\+|gcc|c\+\+|cc|clang\+?\+?|nvcc)["']""")
#: what any match of `_NATIVE` holds: a scan for these is a fraction of the expression's
_NATIVE_HINTS = ("cpp_extension", "load_inline", "cc'", 'cc"', "+'", '+"', "clang'", 'clang"')


def _parent(path: str) -> str:
    """The directory holding ``path``, "" for the root."""
    return path.rpartition("/")[0]


def _join(d: str, name: str) -> str:
    return f"{d}/{name}" if d else name


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _up(path: str | None, n: int) -> str | None:
    """``n`` directories above ``path``, None above the root."""
    for _ in range(n):
        if not path:
            return None
        path = _parent(path)
    return path


def _above(d: str) -> list[str]:
    """The directories above ``d``, nearest first, the root last."""
    out = []
    while d:
        d = _parent(d)
        out.append(d)
    return out


def _alias(text: str, name: str) -> str | None:
    """The module ``name`` stands for in ``text``: `import a.b as name`, `from a import b as name`, `from a import name`, or
    `name = "a.b"`."""
    if not re.fullmatch(_NAME, name):
        return None
    n = re.escape(name)
    if m := re.search(rf"^[ \t]*import[ \t]+([\w.]+)[ \t]+as[ \t]+{n}\b", text, re.M):
        return m.group(1)
    if m := re.search(rf"^[ \t]*from[ \t]+([\w.]+)[ \t]+import[ \t]+(?:[^\n#]*?[ \t,(])?(\w+)[ \t]+as[ \t]+{n}\b", text, re.M):
        return f"{m.group(1)}.{m.group(2)}"
    if m := re.search(rf"^[ \t]*from[ \t]+([\w.]+)[ \t]+import[ \t]+(?:[^\n#]*?[ \t,(])?{n}\b(?![ \t]+as)", text, re.M):
        return f"{m.group(1)}.{name}"
    if m := re.search(rf"""^{n}[ \t]*(?::[^=\n]*)?=[ \t]*[rRuU]?(['"])([\w.]+)\1""", text, re.M):
        return m.group(2)
    return None


def _sub(path: str | None, part: str | None) -> str | None:
    """``part`` under ``path``, None outside the tree."""
    if path is None or part is None:
        return None
    p = posixpath.normpath(_join(path, part))
    return None if p.startswith(("..", "/")) else "" if p == "." else p


def _ast(source: str, mode: str = "exec") -> ast.Module | ast.Expression | Exception:
    """``source`` parsed, or the error parsing it raised, for the caller's fallback: a fragment that is no code alone (a
    statement's `else:` arm), a file Python can't import, or nesting past the parser's depth."""
    try:
        return ast.parse(source, mode=mode)
    except (SyntaxError, ValueError, RecursionError) as e:
        return e


def _built_paths(text: str) -> Iterator[re.Match]:
    """`_BUILT_PATH.finditer(text)`, trying the expression only before each `/` a string follows: a match's head holds no
    whitespace and no `/` and only whitespace parts it from its first such `/`, so it starts in the run of non-whitespace before
    the first one at or after where the search is, or nowhere before that `/`. Trying it at every name is the slow way."""
    pos = 0
    while s := _QUOTED_SLASH.search(text, pos):
        slash = lo = s.start()
        while lo > pos and text[lo - 1].isspace():
            lo -= 1
        while lo > pos and not text[lo - 1].isspace():
            lo -= 1
        for at in range(lo, slash):
            if m := _BUILT_PATH.match(text, at):
                yield m
                pos = m.end()
                break
        else:
            pos = slash + 1


def _parse(text: str, at: int) -> ast.Module | Exception:
    """The logical line of ``text`` from the start of the line holding offset ``at``, parsed alone: parsing whole files to
    find the few statements that touch `sys.path` would take most of the listing's time."""
    return _ast(_statement(text, text.rfind("\n", 0, at) + 1).strip())


class _Paths:
    """The repository paths file ``path``'s expressions build from `__file__`, string constants and its module-level names
    (`Path(__file__).resolve().parents[1]`, `HERE.parent / "x"`, `os.path.dirname(__file__)`): `where`, None for one it builds
    otherwise or outside the tree."""

    def __init__(self, path: str, text: str):
        self.path, self.text = path, text
        self._defs: dict[str, ast.expr | None] = {}

    def value(self, name: str) -> ast.expr | None:
        """What module-level name ``name`` is assigned."""
        if name not in self._defs:
            self._defs[name] = None
            m = re.search(rf"^{re.escape(name)}[ \t]*(?::[^=\n]*)?=(?!=)", self.text, re.M)
            body = tree.body if m and isinstance(tree := _parse(self.text, m.start()), ast.Module) else []
            if body and isinstance(body[0], (ast.Assign, ast.AnnAssign)):
                self._defs[name] = body[0].value
        return self._defs[name]

    def string(self, node: ast.expr, seen: tuple[str, ...] = ()) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id not in seen and (v := self.value(node.id)) is not None:
            return self.string(v, (*seen, node.id))
        return None

    def where(self, node: ast.expr, seen: tuple[str, ...] = ()) -> str | None:
        if isinstance(node, ast.Name):
            if node.id == "__file__":
                return self.path
            return self.where(v, (*seen, node.id)) if node.id not in seen and (v := self.value(node.id)) is not None else None
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            return _up(self.where(node.value, seen), 1)
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "parents"
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int)):
            return _up(self.where(node.value.value, seen), node.slice.value + 1)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return _sub(self.where(node.left, seen), self.string(node.right, seen))
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
            if name in _SAME and isinstance(f, ast.Attribute):
                return self.where(f.value, seen)
            if not node.args:
                return None
            if name == "dirname":
                return _up(self.where(node.args[0], seen), 1)
            if name == "join":
                p = self.where(node.args[0], seen)
                for a in node.args[1:]:
                    p = _sub(p, self.string(a, seen))
                return p
            if name in _KEEP:
                return self.where(node.args[0], seen)
        return None

    def built(self) -> list[str]:
        """Each path the file builds by joining string parts to a path (`HERE.parent / "harness"`)."""
        out = []
        for m in _built_paths(self.text):
            if isinstance(e := _ast(m.group(0), "eval"), ast.Expression) and (p := self.where(e.body)) is not None:
                out.append(p)
        return out

    def syspath(self) -> tuple[list[str], bool]:
        """The directories the file puts on `sys.path`, and whether it puts one there that it builds otherwise."""
        out, blind = [], False
        for m in _SYSPATH.finditer(self.text):
            if not isinstance(stmt := _parse(self.text, m.start()), ast.Module):
                blind = True
                continue
            for n in ast.walk(stmt):
                for e in _put(n):
                    if (d := self.where(e)) is not None:
                        out.append(d)
                    else:
                        blind = True
        return list(dict.fromkeys(out)), blind


def _is_syspath(node: ast.expr) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "path" and isinstance(node.value, ast.Name) and node.value.id == "sys"


def _items(node: ast.expr) -> list[ast.expr]:
    return list(node.elts) if isinstance(node, (ast.List, ast.Tuple)) else [node]


def _put(node: ast.AST) -> list[ast.expr]:
    """What a statement puts on `sys.path`."""
    if isinstance(node, ast.Assign):
        return _items(node.value) if any(isinstance(t, ast.Subscript) and _is_syspath(t.value) for t in node.targets) else []
    if not isinstance(node, ast.Call) or not node.args:
        return []
    f = node.func
    name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
    if name in ("syspath_prepend", "addsitedir"):
        return [node.args[0]]
    if name not in ("insert", "append", "extend") or not isinstance(f, ast.Attribute) or not _is_syspath(f.value):
        return []
    if name == "insert":
        return node.args[1:2]
    return [node.args[0]] if name == "append" else _items(node.args[0])


@dataclass(frozen=True)
class Suite:
    name: str
    path: str                      # its directory, "." for the repository's own
    testpaths: tuple[str, ...]     # root-relative
    pythonpath: tuple[str, ...]    # root-relative: `[tool.pytest.ini_options] pythonpath`, from its rootdir
    inputs: tuple[str, ...]        # `[tool.verity.tests] inputs`
    timeout_s: int | None          # `[tool.verity.tests] timeout_s`
    mem_gb: float | None           # `[tool.verity.tests] worker_gb`: one test process's peak
    reads: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = ()  # `[[tool.verity.tests.reads]]`: (tests, paths) each
    untraced: tuple[str, ...] = ()  # `[tool.verity.tests] untraced`: test files that start programs the guard can't trace

    @property
    def dir(self) -> str:
        return "" if self.path == "." else self.path

    @property
    def markdown(self) -> bool:
        """Whether its tests read Markdown, which it says by listing a `.md` (`suites.markdown`)."""
        return any(p.endswith(".md") for p in self.inputs)


class Tree:
    """The commit's files, read from ``root``: which are files, which directories, what each directory holds."""

    def __init__(self, root: Path, files: list[str]):
        self.root = root
        self.sorted = sorted(files)
        self.files = frozenset(self.sorted)
        self.children: dict[str, set[str]] = {}
        for f in self.sorted:
            d, _, name = f.rpartition("/")
            while True:
                kids = self.children.setdefault(d, set())
                if name in kids:
                    break
                kids.add(name)
                if not d:
                    break
                d, _, name = d.rpartition("/")
        self.dirs = frozenset(self.children)
        code: set[str] = set()                  # the directories holding Python or a pyproject.toml at any depth
        for f in self.sorted:
            if f.endswith(".py") or f.rpartition("/")[2] == "pyproject.toml":
                d = _parent(f)
                while d not in code:
                    code.add(d)
                    if not d:
                        break
                    d = _parent(d)
        self.code = frozenset(code)
        self.pyprojects = tuple(f for f in self.sorted if f.rpartition("/")[2] == "pyproject.toml")
        self._beside: dict[str, list[str]] = {}
        self._toml: dict[str, dict] = {}

    def beside(self, d: str) -> list[str]:
        """The data beside the modules of directory ``d``: its files that aren't Python, and every file under those of its
        directories that hold no Python and no pyproject.toml (`tables/`, `pins/`). The root's directories are no module's."""
        if (got := self._beside.get(d)) is None:
            got = self._beside[d] = []
            for n in sorted(self.children.get(d, ())):
                p = _join(d, n)
                if p in self.files:
                    if not n.endswith(".py"):
                        got.append(p)
                elif d and p not in self.code:
                    got += self.under(p)
        return got

    def text(self, path: str) -> str:
        return (self.root / path).read_bytes().decode("utf-8", "replace")

    def toml(self, path: str) -> dict:
        """``path`` parsed, once: the caller reads it and never changes it."""
        if (got := self._toml.get(path)) is None:
            got = self._toml[path] = tomllib.loads(self.text(path))
        return got

    def under(self, d: str) -> list[str]:
        """The files at or under ``d`` ("" or "." for the root)."""
        if d in ("", "."):
            return self.sorted
        if d in self.files:
            return [d]
        lo = bisect.bisect_left(self.sorted, d + "/")
        hi = bisect.bisect_left(self.sorted, d + "0")         # "0" sorts just after "/"
        return self.sorted[lo:hi]

    def has(self, base: str, name: str) -> bool:
        """Whether a directory on `sys.path` serves import name ``name``: a module file or a directory (a package, or a namespace
        package's portion)."""
        return _join(base, name + ".py") in self.files or _join(base, name) in self.dirs

    def names(self, base: str) -> set[str]:
        """The import names directory ``base`` serves."""
        out = set()
        for n in self.children.get(base, ()):
            if n.endswith(".py"):
                out.add(n[:-3])
            elif _join(base, n) in self.dirs:
                out.add(n)
        return out


def _wheel(data: dict) -> dict:
    hatch = data.get("tool", {}).get("hatch", {}).get("build", {})
    return {**hatch, **hatch.get("targets", {}).get("wheel", {})}


class Layout:
    """Where the workspace's packages put import names in the environment `uv sync` installs (each member editable, by hatch): an
    exact editable (`dev-mode-exact`) serves its packages alone, by name; any other puts each package's parent directory (or its
    `dev-mode-dirs`) on `sys.path`, which then serves every name in it. A virtual member (`[tool.uv] package = false`) serves none."""

    def __init__(self, tree: Tree):
        self.tree = tree
        root = tree.toml("pyproject.toml")
        members = [""] + [m.strip("/") for m in root.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])]
        self.exact: dict[str, list[str]] = {}
        paths: set[str] = set()
        for m in members:
            data = tree.toml(_join(m, "pyproject.toml"))
            if data.get("tool", {}).get("uv", {}).get("package") is False or "build-system" not in data or "project" not in data:
                continue
            wheel = _wheel(data)
            module = _norm(data["project"]["name"]).replace("-", "_")
            packages = wheel.get("packages") or [p for p in (f"src/{module}", module) if _join(_join(m, p), "__init__.py")
                                                  in tree.files][:1]
            full = [_join(m, p.strip("/")) for p in packages]
            if wheel.get("dev-mode-exact"):
                for p in full:
                    self.exact.setdefault(p.rpartition("/")[2].removesuffix(".py"), []).append(_parent(p))
            elif wheel.get("dev-mode-dirs"):
                paths.update(_join(m, d.strip("/")).rstrip("/") for d in wheel["dev-mode-dirs"] if d.strip("/") != ".")
                paths.update(m for d in wheel["dev-mode-dirs"] if d.strip("/") == ".")
            else:
                paths.update(_parent(p) for p in full)
        self.paths = tuple(sorted(paths))

    def bases(self, top: str) -> list[str]:
        """The `sys.path` directories that serve top-level name ``top``."""
        return [*self.exact.get(top, ()), *(b for b in self.paths if self.tree.has(b, top))]

    def serves(self, top: str) -> bool:
        return top in self.exact or any(self.tree.has(b, top) for b in self.paths)


def _statement(text: str, start: int) -> str:
    """The logical line from ``start``: to a newline outside brackets that no backslash continues, comments left out."""
    end = text.find("\n", start)
    end = len(text) if end < 0 else end
    line = text[start:end]
    if "(" not in line and "#" not in line and not line.endswith("\\"):
        return line
    out, depth, i, n = [], 0, start, len(text)
    while i < n:
        c = text[i]
        if c == "#":
            i = text.find("\n", i)
            if i < 0:
                break
            continue
        if c == "\\" and text.startswith("\n", i + 1):
            out.append(" ")
            i += 2
            continue
        if c == "\n" and depth <= 0:
            break
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        out.append(" " if c == "\n" else c)
        i += 1
    return "".join(out)


def _pyfiles(text: str) -> list[str]:
    """The `.py` paths the strings of ``text`` name: each `.py` string, and each path of string parts joined by `/`."""
    return [*_PYFILE.findall(text), *("/".join(re.findall(r"""["']([\w.-]+)["']""", m)) for m in _PARTS.findall(text))]


def _requests(stmt: str) -> list[tuple[int, str, tuple[str, ...]]] | None:
    """The imports one statement makes, each (level, module, names imported from it), or None when it isn't an import statement
    of the forms the regular expressions read (the caller parses it)."""
    out = []
    for part in stmt.split(";"):
        part = part.strip()
        if not part:
            continue
        if part.startswith("import"):
            if not (m := _IMPORT.fullmatch(part)):
                return None
            out += [(0, re.sub(r"\s", "", _AS_SPLIT.split(a)[0]), ()) for a in re.split(r"\s*,\s*", m.group(1))]
        elif part.startswith("from"):
            if not (m := _FROM.fullmatch(part)):
                return None
            names = () if m.group(3) == "*" else tuple(
                _AS_SPLIT.split(n.strip())[0] for n in m.group(3).strip("() \t").split(",") if n.strip())
            out.append((len(m.group(1)), re.sub(r"\s", "", m.group(2) or ""), names))
        elif out:
            break                                           # code after the imports on their line
        else:
            return None
    return out


def _parsed(text: str) -> list[tuple[int, str, tuple[str, ...]]]:
    """Every import statement of ``text`` (a file, or a statement of one), by its syntax tree: none where Python can't parse it,
    which no import of the file could run."""
    if not isinstance(tree := _ast(text), ast.Module):
        return []
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out += [(0, a.name, ()) for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            out.append((n.level, n.module or "", tuple(a.name for a in n.names if a.name != "*")))
    return out


@dataclass
class Info:
    """What one file imports: the files that resolves to wherever it is imported (`fixed`), and its absolute imports of names the
    workspace doesn't serve (`local`: top-level name -> [(module, names)]), which the directories pytest or another file puts on
    `sys.path` may (`paths`, those this one puts there); what it may start: the `.py` files its strings name (`scripts`), the
    modules it runs as `python -m NAME` (`modules`) and the absolute imports of the code its strings hold (`code`: a `python -c`
    program); and whether it lists every pyproject.toml (`metadata`)."""
    fixed: tuple[str, ...]
    local: dict[str, list[tuple[str, tuple[str, ...]]]]
    unfollowed: int
    scripts: tuple[str, ...]
    modules: tuple[str, ...]
    paths: tuple[str, ...]
    metadata: bool
    code: tuple[tuple[str, tuple[str, ...]], ...] = ()


class Imports:
    """The repository's Python files' imports, each file read once, and the closure of a task's files."""

    def __init__(self, tree: Tree, layout: Layout):
        self.tree, self.layout = tree, layout
        self._info: dict[str, Info] = {}
        self._serves: dict[str, bool] = {}
        self._walked: dict[tuple[str, str, tuple[str, ...]], tuple[str, ...]] = {}
        self._closed: dict[str, frozenset[str]] = {}            # file -> what its `fixed` edges reach, itself included
        self._paths: dict[str, list[str]] = {}                  # file -> the files under the paths it names (`named`)
        self.metadata: set[str] = set()                         # the files read so far that list every pyproject.toml
        self.native: set[str] = set()                           # those that build native code (`_NATIVE`)
        self._by_top: dict[str, set[str]] = {}                  # top-level name -> the files importing it absolutely
        self.fallbacks = 0              # files whose statements the regular expressions couldn't read, parsed whole

    def serves(self, top: str) -> bool:
        if top not in self._serves:
            self._serves[top] = self.layout.serves(top)
        return self._serves[top]

    def walk(self, base: str, module: str, names: tuple[str, ...]) -> tuple[str, ...]:
        """The files importing ``module`` from ``base`` (a `sys.path` directory, or a package's for a relative import) runs: each
        package's `__init__.py` on the way and the module's file, then each of ``names`` that is a submodule (`("*",)`: each of
        its modules)."""
        if (got := self._walked.get(at := (base, module, names))) is None:
            got = self._walked[at] = tuple(self._walk(base, module, names))
        return got

    def _walk(self, base: str, module: str, names: tuple[str, ...]) -> list[str]:
        out, d = [], base
        parts = [p for p in module.split(".") if p]
        for p in parts:
            d = _join(d, p)
            if d in self.tree.dirs:
                if (init := d + "/__init__.py") in self.tree.files:
                    out.append(init)
                if d + ".py" in self.tree.files:            # a namespace directory beside a module of its name
                    out.append(d + ".py")
            elif d + ".py" in self.tree.files:
                out.append(d + ".py")
                return out
            else:
                return out
        if names == ("*",):
            names = tuple(sorted(self.tree.names(d) - {"__init__"}))
        for n in names:
            sub = _join(d, n)
            if sub + ".py" in self.tree.files:
                out.append(sub + ".py")
            elif (init := sub + "/__init__.py") in self.tree.files:
                out.append(init)
        return out

    def _named(self, path: str, name: str, dirs: bool = False) -> str | None:
        """The repository file (or with ``dirs`` directory) a string in ``path`` names: from its directory, or the nearest one
        above it."""
        d = _parent(path)
        while True:
            p = posixpath.normpath(_join(d, name))
            if not p.startswith("../") and (p in self.tree.files or dirs and p in self.tree.dirs):
                return p
            if not d:
                return None
            d = _parent(d)

    def named(self, path: str) -> list[str]:
        """The repository files under the paths file ``path`` names: string parts joined by `/` or one string holding one
        (`"verity" / "core"`, `"tools/verity/check"`), each from its directory or the nearest one above it, and string parts
        joined to a path it builds (`VERIFIER / "agree.py"`, `VERIFIER = HERE.parent / "verifier"`)."""
        if (got := self._paths.get(path)) is None:
            text = self.tree.text(path)
            parts = ("/".join(re.findall(r"""["']([\w.-]+)["']""", m)) for m in _DATA_PARTS.findall(text))
            found = [p for s in [*parts, *_SLASHED.findall(text)] if (p := self._named(path, s, dirs=True)) is not None]
            found += [p for p in _Paths(path, text).built() if p in self.tree.files or p in self.tree.dirs]
            got = self._paths[path] = sorted({f for p in found for f in self.tree.under(p)})
        return got

    def _registry(self, path: str) -> str | None:
        """The settings registry `verity_config.settings.of(path)` loads: the `settings.py` beside the nearest pyproject.toml
        above ``path``, else beside ``path``."""
        d = _parent(path)
        while True:
            if _join(d, "pyproject.toml") in self.tree.files:
                return p if (p := _join(d, "settings.py")) in self.tree.files else None
            if not d:
                break
            d = _parent(d)
        return p if (p := _join(_parent(path), "settings.py")) in self.tree.files else None

    @staticmethod
    def _fstring(text: str, at: int) -> str | None:
        """The f-string a call's first argument at offset ``at`` is, or the one a module-level or local name it passes is
        assigned (`module = f"{__name__}.{ADAPTERS[name]}"`)."""
        if f := _FSTRING.match(text, at):
            return f.group(2)
        if (n := _ARG_NAME.match(text, at)) and (a := re.search(rf"""^[ \t]*{n.group(1)}[ \t]*=[ \t]*[fF](['"])([^'"\n]*\{{[^'"\n]*)\1""",
                                                                 text, re.M)):
            return a.group(2)
        return None

    def _package(self, path: str, text: str, name: str) -> tuple[int, str] | None:
        """The package an f-string module name ``name`` in file ``path`` names a module of, as an import's (level, module), or
        None: the constant dotted head before its first `{` (`f"verity.ml.registry.{m}"`), else the package `{__name__}` or
        `{__package__}` is, or the module `{X.__name__}` or a constant `{X}` names, with what follows it up to the next `{`."""
        head, _, rest = name.partition("{")
        if head:
            level = len(head) - len(head.lstrip("."))
            return level, head[level:].rpartition(".")[0]
        inner, _, after = rest.partition("}")
        if not after.startswith("."):
            return None
        tail = after[1:].partition("{")[0].rpartition(".")[0]
        if inner in ("__name__", "__package__"):
            own = "" if inner == "__package__" or path.rpartition("/")[2] == "__init__.py" else path.rpartition("/")[2][:-3]
            return 1, ".".join(p for p in (own, tail) if p)
        module = _alias(text, inner.removesuffix(".__name__"))
        return (0, ".".join(p for p in (module, tail) if p)) if module else None

    def info(self, path: str) -> Info:
        if (got := self._info.get(path)) is not None:
            return got
        text = self.tree.text(path)
        reqs: list[tuple[int, str, tuple[str, ...]]] = []
        for m in _STMT.finditer(text):
            r = _requests(_statement(text, m.start(1)))
            if r is None:
                r = _parsed(_statement(text, m.start(1)).strip())
                if not r:
                    self.fallbacks += 1
                    reqs = _parsed(text)
                    break
            reqs += r
        unfollowed, by_path, blind = 0, False, False
        fixed: list[str] = []
        for m in _CALL.finditer(text) if any(k in text for k in _CALLS) else ():
            call = _statement(text, m.end() - 1)
            if m.group(1) in _BY_NAME:
                if c := _CONSTANT.match(text, m.end()):
                    reqs.append((0, c.group(2), ()))
                elif (f := self._fstring(text, m.end())) and (each := self._package(path, text, f)):
                    reqs.append((*each, ("*",)))
                else:
                    unfollowed += 1
            elif loaded := [p for s in _pyfiles(call) if (p := self._named(path, s))]:
                fixed += loaded
            else:
                unfollowed, by_path, blind = unfollowed + 1, by_path or bool(_BUILT.search(call)), True
        for line in _PLUGINS.finditer(text) if "pytest_plugins" in text else ():
            for s in re.findall(r"""["']([\w.]+)["']""", _statement(text, line.start())):
                reqs.append((0, s, ()))
        reqs += [(0, s, ()) for s in _PATCH.findall(text)] if "setattr(" in text or "delattr(" in text or "patch(" in text else []
        local: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
        here = _parent(path)
        put = "sys.path" in text or "syspath_prepend" in text or "addsitedir" in text
        built = _Paths(path, text) if put and _SYSPATH.search(text) else None
        dirs, lost = built.syspath() if built else ([], False)
        dirs = [d for d in dirs if d in self.tree.dirs]
        guesses = [d for d in built.built() if d in self.tree.dirs] if lost else []
        search = [*dirs, here, *guesses, *(_above(here) if lost else ())]   # where it builds them otherwise: any it builds
        for level, module, names in reqs:
            if level:
                base = here
                for _ in range(level - 1):
                    base = _parent(base)
                fixed += self.walk(base, module, names)
                continue
            top = module.partition(".")[0]
            if not top or top in STDLIB or top == "__future__":
                continue
            local.setdefault(top, []).append((module, names))
            if self.serves(top):
                for b in self.layout.bases(top):
                    fixed += self.walk(b, module, names)
            elif (b := next((b for b in search if self.tree.has(b, top)), None)) is not None:
                fixed += self.walk(b, module, names)
        if "iter_modules(__path__)" in text:            # a package that imports each of its modules
            fixed += [f for f in self.tree.under(here) if _parent(f) == here and f.endswith(".py")]
        if "of(__file__)" in text and (registry := self._registry(path)):        # `verity_config.settings.of`
            fixed.append(registry)
        scripts = {p for s in _pyfiles(text) if (p := self._named(path, s))} if ".py'" in text or '.py"' in text else set()
        if blind:                       # a path loader handed a path it doesn't name (a helper's parameter): each `.py` it names
            fixed += scripts
        if by_path:                     # a module loaded by a path it builds from a name (`_load("gamma")`): each name with a file
            bare = set(_BARE.findall(text))
            fixed += [p for s in bare if (p := _join(here, s + ".py")) in self.tree.files]
            scripts |= {p for s in bare if (p := self._named(path, s + ".py"))}
        modules = set(_DASH_M.findall(text)) if "-m'" in text or '-m"' in text else set()
        code = set()
        for m in _CODE_IMPORT.finditer(text) if "import" in text else ():
            if m.group(1):
                code.add((m.group(1), tuple(n for n in re.split(r"[\s,]+", m.group(2)) if n and n != "as")))
            else:
                code.update((n, ()) for n in re.split(r"\s*,\s*", m.group(3)))
        got = self._info[path] = Info(tuple(sorted(set(fixed) - {path})), local, unfollowed, tuple(sorted(scripts - {path})),
                                      tuple(sorted(modules)), tuple(dirs), bool("*/pyproject.toml" in text and _METADATA.search(text)
                                                                                or "members" in text and _MEMBERS.search(text)
                                                                                and _PYPROJECT.search(text)),
                                      tuple(sorted(c for c in code if c[0].partition(".")[0] not in STDLIB)))
        if got.metadata:
            self.metadata.add(path)
        if any(k in text for k in _NATIVE_HINTS) and _NATIVE.search(text):
            self.native.add(path)
        for top in local:
            self._by_top.setdefault(top, set()).add(path)
        return got

    def reached(self, start: str) -> frozenset[str]:
        """What ``start``'s `fixed` edges reach, itself included: one set per strongly connected component (Tarjan's, iterative),
        each from its members and its successors' sets, so every file's is computed once."""
        if (got := self._closed.get(start)) is not None:
            return got
        index: dict[str, int] = {}
        low: dict[str, int] = {}
        stack: list[str] = []
        on: set[str] = set()
        work = [(start, iter(self.info(start).fixed))]
        index[start] = low[start] = 0
        stack.append(start)
        on.add(start)
        while work:
            v, edges = work[-1]
            for w in edges:
                if w in self._closed:
                    continue
                if w not in index:
                    index[w] = low[w] = len(index)
                    stack.append(w)
                    on.add(w)
                    work.append((w, iter(self.info(w).fixed)))
                    break
                if w in on:
                    low[v] = min(low[v], index[w])
            else:
                work.pop()
                if work:
                    u = work[-1][0]
                    low[u] = min(low[u], low[v])
                if low[v] == index[v]:
                    members = set()
                    while True:
                        w = stack.pop()
                        on.discard(w)
                        members.add(w)
                        if w == v:
                            break
                    out = set(members)
                    for m in members:
                        for w in self.info(m).fixed:
                            if w not in members:
                                out |= self._closed[w]
                    done = frozenset(out)
                    for m in members:
                        self._closed[m] = done
        return self._closed[start]

    def closure(self, seeds: list[str], bases: list[str], starters: list[str]) -> set[str]:
        """The Python files importing ``seeds`` with ``bases`` on `sys.path` (pytest's) can run, ``seeds`` among them, and those
        the scripts, modules and `python -c` programs ``starters`` (a test file and its conftests) name can (`Info.scripts`,
        `Info.modules`, `Info.code`): what their
        `fixed` edges reach (`reached`), and what those files' absolute imports of a name one of ``bases`` serves reach from it,
        or one of the directories a file of the closure puts on `sys.path` (`Info.paths`): the process's `sys.path` is one.
        What a module further in starts isn't followed: across the tree that joins most of it into one closure."""
        roots = [s for s in seeds if s.endswith(".py")]
        for s in starters:
            info = self.info(s)
            roots += info.scripts
            for m, names in [*((m, ("__main__",)) for m in info.modules), *info.code]:
                top = m.partition(".")[0]
                for b in dict.fromkeys([*self.layout.bases(top), *(b for b in bases if self.tree.has(b, top))]):
                    roots += self.walk(b, m, names)
        out: set[str] = set()
        grown = set().union(*(self.reached(s) for s in roots))
        known: set[str] = set(self.layout.paths)
        served: dict[str, list[str]] = {}
        while grown:
            out |= grown
            for b in dict.fromkeys([*bases, *(d for f in sorted(grown) for d in self._info[f].paths)]):
                if b not in known:
                    known.add(b)
                    for n in self.tree.names(b):
                        served.setdefault(n, []).append(b)
            bases = []
            new = set()
            for top in served.keys() & self._by_top.keys():
                for f in self._by_top[top] & out:
                    for b in served[top]:
                        for module, names in self._info[f].local[top]:
                            new.update(self.walk(b, module, names))
            grown = set().union(*(self.reached(g) for g in new - out)) - out
        return out


def discover(tree: Tree) -> list[Suite]:
    """Every suite, as `suites.discover` finds them: each directory whose `pyproject.toml` configures pytest."""
    out = []
    for f in tree.sorted:
        if f != "pyproject.toml" and not f.endswith("/pyproject.toml"):
            continue
        data = tree.toml(f)
        if "pytest" not in data.get("tool", {}):
            continue
        path = _parent(f) or "."
        base = "" if path == "." else path
        project = data.get("project", {})
        cfg = data["tool"]["pytest"].get("ini_options", {})
        tests = data["tool"].get("verity", {}).get("tests", {})
        testpaths = tuple(_join(base, t.strip("/")) if t not in (".", "") else base for t in cfg.get("testpaths", ["."]))
        pythonpath = tuple("" if (p := posixpath.normpath(_join(base, x))) == "." else p for x in cfg.get("pythonpath", []))
        out.append(Suite("repository" if path == "." else _norm(project["name"]) if project else path, path, testpaths,
                         pythonpath, tuple(tests.get("inputs", [])), math.ceil(tests["timeout_s"]) if tests.get("timeout_s") else None,
                         tests.get("worker_gb") or None, _reads(f, tests.get("reads", [])), _strings(f, tests, "untraced")))
    return out


def _strings(where: str, tests: dict, key: str) -> tuple[str, ...]:
    got = tests.get(key, [])
    if not isinstance(got, list) or not all(isinstance(s, str) and s for s in got):
        raise ValueError(f"{where}: [tool.verity.tests] {key} is a list of test files (fnmatch patterns) from the suite's directory")
    return tuple(got)


def _reads(where: str, entries: object) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    """A suite's `[[tool.verity.tests.reads]]`: each entry's `tests` (its test files, or fnmatch patterns over them, from its
    directory) and `paths` (what they read whole, Python included, from the root: a file, a directory or an fnmatch pattern)."""
    def strings(v: object) -> bool:
        return isinstance(v, list) and bool(v) and all(isinstance(s, str) and s for s in v)

    if not isinstance(entries, list) or not all(isinstance(e, dict) and set(e) == {"tests", "paths"} and strings(e["tests"])
                                                and strings(e["paths"]) for e in entries):
        raise ValueError(f"{where}: each [[tool.verity.tests.reads]] holds `tests` (test files from the suite's directory) and "
                         "`paths` (what they read whole, from the root), each a non-empty list of strings")
    return tuple((tuple(e["tests"]), tuple(e["paths"])) for e in entries)


def _under(path: str, d: str) -> bool:
    return d in ("", ".") or path == d or path.startswith(d + "/")


def suite_tests(tree: Tree, suite: Suite, suites: list[Suite]) -> list[str]:
    """``suite``'s test files: its `test_*.py` under its test paths, outside a data directory and the suites nested in it."""
    nested = [s.dir for s in suites if s.path != suite.path and suite.path != "." and _under(s.dir, suite.dir)]
    out = []
    for t in suite.testpaths:
        for f in tree.under(t):
            parts = f.split("/")
            if (parts[-1].startswith("test_") and parts[-1].endswith(".py") and not DATA_DIRS.intersection(parts[:-1])
                    and not any(_under(f, n) for n in nested)):
                out.append(f)
    return sorted(set(out))


def declared(tree: Tree, suite: Suite, suites: list[Suite], fixtures: list[str]) -> list[str]:
    """The suite's declared inputs less their Python, an import list the closure replaces, and its own directory's files but
    their Python and the suites nested in it (`suites.inputs_of` always holds the directory); Markdown under a directory only
    when it lists a `.md`; and the fixture registry when a fetched fixture is under the suite or one of its inputs. The whole
    tree (`.`) is no import list: a suite declares it whose tests read every file (the root's lints, the graph's listing)."""
    out: set[str] = set()
    for p in suite.inputs:
        p = p.strip("/")
        if p in tree.files:
            if not p.endswith(".py"):
                out.add(p)
            continue
        out.update(f for f in tree.under(p) if (p in ("", ".") or not f.endswith(".py")) and (suite.markdown or not f.endswith(".md")))
    if suite.path != ".":
        nested = [s.dir for s in suites if s.path != suite.path and _under(s.dir, suite.dir)]
        out.update(f for f in tree.under(suite.dir) if not f.endswith(".py") and (suite.markdown or not f.endswith(".md"))
                   and not any(_under(f, n) for n in nested))
    if fetches(suite, fixtures):
        out.add(REGISTRY)
    return sorted(out)


def walks(tree: Tree, suite: Suite, tests: list[str]) -> dict[str, set[str]]:
    """{test file: the files its suite's `[[tool.verity.tests.reads]]` say it reads whole}: what it reads that no listing can
    follow (a tree it walks, a module it loads by a name it computes, what a program it starts reads). An entry whose `tests`
    match none of ``tests`` or whose `paths` match no file of the tree is refused, so none outlives what it declares."""
    where = _join(suite.dir, "pyproject.toml")
    out: dict[str, set[str]] = {}
    for pats, paths in suite.reads:
        files: set[str] = set()
        for p in paths:
            got = ([f for f in tree.sorted if fnmatch.fnmatchcase(f, p)] if any(c in p for c in "*?[")
                   else tree.under(p.strip("/")) if p.strip("/") not in ("", ".") else [])
            if not got:
                raise ValueError(f"{where}: [[tool.verity.tests.reads]] path {p!r} is no file or directory of the tree: drop it, "
                                 "or name what the tests read")
            files.update(got)
        for t in _named_tests(suite, tests, pats, "[[tool.verity.tests.reads]] tests"):
            out.setdefault(t, set()).update(files)
    return out


def _named_tests(suite: Suite, tests: list[str], pats: tuple[str, ...], what: str) -> set[str]:
    """The test files of ``tests`` that the fnmatch patterns ``pats`` (from the suite's directory) name; a pattern naming none is
    refused."""
    out: set[str] = set()
    for pat in pats:
        hit = [t for t in tests if fnmatch.fnmatchcase(t[len(suite.dir) + 1:] if suite.dir else t, pat)]
        if not hit:
            raise ValueError(f"{_join(suite.dir, 'pyproject.toml')}: {what} {pat!r} names none of the suite's test files: drop it, "
                             "or name them from the suite's directory")
        out.update(hit)
    return out


def whole(tree: Tree, suite: Suite, suites: list[Suite]) -> list[str]:
    """What `suites.py` keys the suite on (`suites.inputs_of`, less `suites.dropped`): the files of its directory but the suites
    nested in it that no other path of it covers, of each workspace package it depends on (its dependencies, extras and
    dependency groups, and theirs but groups: the package's `[tool.verity.tests] exports`, else its directory), but with
    `inputs_only = true`, and of its declared inputs. A test file that starts a program the guard can't trace is keyed on it."""
    root = tree.toml("pyproject.toml")
    pkgs, data = {}, {}
    for m in ["", *(m.strip("/") for m in root.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", []))]:
        d = data[m] = tree.toml(_join(m, "pyproject.toml"))
        if "project" in d and (m or ("build-system" in d and d.get("tool", {}).get("uv", {}).get("package") is not False)):
            pkgs[_norm(d["project"]["name"])] = m

    def requires(d: dict, groups: bool) -> list[str]:
        project = d.get("project", {})
        reqs = [*project.get("dependencies", []), *(r for e in project.get("optional-dependencies", {}).values() for r in e),
                *(r for g in d.get("dependency-groups", {}).values() if groups for r in g if isinstance(r, str))]
        return [n for r in reqs if (n := _norm(re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", r).group(1))) in pkgs]

    own = tree.toml(_join(suite.dir, "pyproject.toml"))
    paths = {suite.dir, *(p.strip("/") for p in suite.inputs)}
    if own.get("tool", {}).get("verity", {}).get("tests", {}).get("inputs_only") is not True:
        seen, todo = set(), requires(own, True)
        while todo:
            if (name := todo.pop()) in seen:
                continue
            seen.add(name)
            m = pkgs[name]
            todo += requires(data[m], False)
            if m == suite.dir:
                continue
            exports = data[m].get("tool", {}).get("verity", {}).get("tests", {}).get("exports")
            paths.update(exports if exports is not None else [m] if m else _wheel(data[m]).get("packages", []))
    other = [p for p in paths if p != suite.dir]
    cut = [s.dir for s in suites if s.path != suite.path and suite.path != "." and _under(s.dir, suite.dir)
           and not any(_under(s.dir, o) for o in other)]
    out = {f for p in paths for f in tree.under(p)}
    for c in cut:
        out -= set(tree.under(c)) - {f for o in other if _under(o, c) for f in tree.under(o)}
    return sorted(out)


def fetches(suite: Suite, fixtures: list[str]) -> list[str]:
    """The fetched fixtures under ``suite``'s directory or its declared inputs, the root's aside (`suites.py`: the repository's
    own suite reads tracked files only)."""
    dirs = [d for d in (p.strip("/") for p in (suite.dir, *suite.inputs)) if d not in ("", ".")]
    return [f for f in fixtures if any(_under(f, d) for d in dirs)]


def _basedir(tree: Tree, path: str) -> tuple[str, list[str]]:
    """Where pytest's `prepend` import puts module ``path`` on `sys.path`, the first directory above it that isn't a package,
    and the packages' `__init__.py` files the import runs."""
    d, inits = _parent(path), []
    while d and (init := d + "/__init__.py") in tree.files:
        inits.append(init)
        d = _parent(d)
    return d, inits


def task(tree: Tree, imports: Imports, suite: Suite, path: str, data: list[str], walked: set[str] | frozenset[str] = frozenset()) -> dict:
    conftests, d = [], _parent(path)
    while True:
        if (c := _join(d, "conftest.py")) in tree.files:
            conftests.append(c)
        if d == suite.dir or not d:
            break
        d = _parent(d)
    packages, d = [], _parent(path)
    while True:                                 # pytest's Package nodes: it imports each one's `__init__.py` to set it up
        if (init := _join(d, "__init__.py")) in tree.files:
            packages.append(init)
        if d == suite.dir or not d:
            break
        d = _parent(d)
    seeds, bases = [path, *conftests, *packages], list(suite.pythonpath)
    for f in [path, *conftests, *packages]:
        base, inits = _basedir(tree, f)
        bases.append(base)
        seeds += inits
    d, pyprojects = _parent(path), []
    while True:
        if (p := _join(d, "pyproject.toml")) in tree.files:
            pyprojects.append(p)
        if not d:
            break
        d = _parent(d)
    code, md = imports.closure(seeds, bases, [path, *conftests]), suite.markdown
    beside = {f for d in {_parent(f) for f in code} for f in tree.beside(d) if md or not f.endswith(".md")}
    if not imports.metadata.isdisjoint(code):
        pyprojects += tree.pyprojects
    every, under = any(t in ("", ".") for t in suite.testpaths), tuple(t + "/" for t in suite.testpaths)
    helpers = [f for f in code if (every or f.startswith(under) or f in suite.testpaths)
               and not f.rpartition("/")[2].startswith("test_")]
    paths = {f for s in {path, *conftests, *helpers} for f in imports.named(s) if md or not f.endswith(".md")}
    inputs = {*code, *beside, *paths, *seeds, *data, *pyprojects, *walked, *GUARD}
    out = {"id": f"{KIND}:{path}", "vars": {"suite": suite.path}, "inputs": sorted(inputs)}
    if not imports.native.isdisjoint(code):
        out["vars"]["untraced"] = "compilers"
    if suite.mem_gb:
        out["resources"] = {"mem_gb": suite.mem_gb}
    if suite.timeout_s:
        out["timeout_s"] = suite.timeout_s
    return out


def fetched(tree: Tree) -> list[str]:
    """The fixture registry's paths that git doesn't hold: fetched before a test reads them (`suites.fetched_fixtures`)."""
    if REGISTRY not in tree.files:
        return []
    reg = json.loads(tree.text(REGISTRY))
    return sorted(v for k, v in reg.items() if k.startswith("art:") and isinstance(v, str) and v not in tree.files)


def tasks(root: Path, files: list[str]) -> list[dict]:
    from .build import lean_suites  # build imports this module's

    tree = Tree(root, files)
    imports = Imports(tree, Layout(tree))
    suites = discover(tree)
    fixtures = fetched(tree)
    after = set(lean_suites(root, files))
    out = []
    for s in suites:
        if not after.isdisjoint((s.name, s.path)):
            continue
        data = declared(tree, s, suites, fixtures)
        tests = suite_tests(tree, s, suites)
        walked = walks(tree, s, tests)
        untraced = _named_tests(s, tests, s.untraced, "[tool.verity.tests] untraced")
        every = whole(tree, s, suites) if untraced else []
        for f in tests:
            t = task(tree, imports, s, f, data, walked.get(f, set()))
            if f in untraced:
                t["inputs"], t["vars"]["untraced"] = sorted({*t["inputs"], *every}), "suite"
            out.append(t)
    return sorted(out, key=lambda t: t["id"])
