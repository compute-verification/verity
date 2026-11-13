"""`public_boundary`: no file the public monorepo exports needs a file it doesn't.

    uv run tools/verity/check/public_boundary.py                # the edges ALLOWED doesn't list; exit 1 when there is one
    uv run tools/verity/check/public_boundary.py --all          # every edge, by the public group its file is in
    uv run tools/verity/check/public_boundary.py --public       # the public set: the files the export copies
    uv run tools/verity/check/public_boundary.py --json         # the public set, every edge, what moving its target moves
    uv run tools/verity/check/public_boundary.py --prune        # drop from ALLOWED the edges no tree given has
    uv run tools/verity/check/public_boundary.py --record       # ALLOWED is the edges of the trees given
    uv run tools/verity/check/public_boundary.py --root DIR ... # judge other checkouts (a worktree of `next`), one or more

The public monorepo is exported one way from this repository, which stays the only source of truth; `public_boundary.toml`
says which files it holds. A file needs another when it:

* imports it: a Lean `import` (any of `public`, `meta`, `all`), resolved through the Lake packages' sources and roots
  (`tools/verity/lean/module_keys.py`); a Python import, a string an import call takes or a ``"module:attr"`` string
  (`tests/test_boundaries.py`'s ``_imports``, and each `a` of a `from M import a` that is a module ``M.a``), resolved to a
  repository path through the file's own directory (the first
  above it outside a package, as a script or pytest puts it on the path), its suite's `pythonpath`, the workspace's
  distributions and the root, with the `__init__.py` of each package above the target;
* is a lakefile whose package `require`s the other's by path, or a pyproject whose distribution depends on the other's or
  whose script runs a module of it;
* reaches it by a path anchored on its own place (`anchors.py`: a load, a `sys.path` entry, a data file, a script's
  `$(dirname "$0")/..`) or on a package's (`resources.files("pkg")`, the directory an import of `pkg` from the file
  finds), or a shell script names it by a path from the root;
* runs a command the other's distribution provides (`research`, `cluster`): a shell command, or a Python subprocess call
  or `shutil.which`;
* is a Lean module, of its package's lakefile; a lakefile, of its manifest and toolchain; a Python file, of its nearest
  pyproject and the `conftest.py` above it in its tests; a pyproject, of the workspace's and its lock.

A target is a tracked file, or a directory holding one (a path anchored past what git tracks). An edge from a public file
to a target that isn't public is a file the export would ship broken. `public_boundary_allowed.txt` lists the edges the
check allows, one `SOURCE -> TARGET` a line, and only shrinks: an edge it doesn't list fails. Cut an edge (import from a
public module instead, move the lemma, or ask to move a file across), then `--prune`. A listed edge no tree has is reported,
not refused: one list serves every checkout the landing queue holds, and an edge one of them cut another may still have.
Lists of files a manifest builds (a lakefile's `roots`, the workspace's `members`) are not edges: the export keeps the
public entries. A path `require`, an extra's entry, a dependency group's and a console script (`EXPORT_CUTS`) are edges
the export's manifest rewrite drops: closed by the export, not open, so the list holds none of them, while
`public_export.py --build` still builds what it copies without them.
"""
from __future__ import annotations

import argparse
import ast
import functools
import importlib.util
import json
import multiprocessing
import os
import re
import shlex
import subprocess
import sys
import tomllib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BOUNDARY = HERE / "public_boundary.toml"
ALLOWED = HERE / "public_boundary_allowed.txt"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


#: the one reading of a Lean header and of a lakefile, the anchored-path reader and the Python import reader
HEADERS = _load("public_boundary_headers", ROOT / "tools" / "verity" / "lean" / "headers.py")
MODULE_KEYS = _load("public_boundary_module_keys", ROOT / "tools" / "verity" / "lean" / "module_keys.py")
ANCHORS = _load("public_boundary_anchors", HERE / "anchors.py")
IMPORTS = _load("public_boundary_imports", ROOT / "tests" / "test_boundaries.py")

#: calls that run a command line: its first word is the command
RUNNERS = {"run", "check_call", "check_output", "call", "Popen", "system", "popen", "getoutput", "getstatusoutput",
           "execv", "execvp", "execvpe", "execl", "execlp", "spawnv", "spawnvp", "create_subprocess_exec",
           "create_subprocess_shell", "which"}
#: `uv run`'s options that take a value, skipped to reach the command it runs
UV_VALUED = {"--extra", "--package", "--with", "--group", "--directory", "--project", "--python", "-p", "--index",
             "--env-file", "--with-requirements", "--only-group"}
#: what may stand before a command in a shell line
SH_STARTS = {";", "&", "&&", "||", "|", "(", "`", "if", "while", "until", "then", "do", "else", "exec", "time", "env", "sudo",
             "xargs", "!", "{"}
#: a path from the root, as a shell script names one
SH_PATH = re.compile(r"(?<![\w./$-])(?:\./)?((?:verity|tools|infra|benchmarks|integrations|examples|tests|Security|security|"
                     r"fixtures|\.agents|\.cursor)/[\w./@+-]*)")
FIX = ("A public file needs one the export doesn't copy (tools/verity/check/public_boundary.toml). Cut the edge: import "
       "from a public module instead, move the lemma or definition to the public side, or state an interface the public "
       "side owns; or move the needed file to the public side in public_boundary.toml, a boundary decision. Never add a "
       "line to public_boundary_allowed.txt: it only shrinks (`uv run tools/verity/check/public_boundary.py --prune`).")


def _expand(glob: str) -> list[str]:
    m = re.search(r"\{([^{}]*)\}", glob)
    return [glob] if not m else [g for alt in m.group(1).split(",")
                                 for g in _expand(glob[:m.start()] + alt + glob[m.end():])]


def _regex(glob: str) -> str:
    """A glob's regular expression: it covers the path it names and everything under it."""
    segs, out = glob.strip("/").split("/"), []
    for i, s in enumerate(segs):
        last = i == len(segs) - 1
        if s == "**":
            out.append(".*" if last else "(?:[^/]+/)*")
        else:
            out.append("".join("[^/]*" if c == "*" else "[^/]" if c == "?" else re.escape(c) for c in s) + ("" if last else "/"))
    return "".join(out) + "(?:/.*)?"


def globs(patterns: Iterable[str]) -> re.Pattern[str]:
    alts = [_regex(g) for p in patterns for g in _expand(p)]
    return re.compile("|".join(f"(?:{a})" for a in alts) if alts else "(?!)")


class Needed(NamedTuple):
    why: str
    paths: re.Pattern[str]
    #: the public groups whose files may pull these in, or None for any
    source: frozenset[str] | None
    #: the kinds of edge that may pull these in (`Edge.kind`), or None for any
    via: frozenset[str] | None = None


class Residue(NamedTuple):
    what: str
    #: what marks a site of it in a file's text
    pattern: re.Pattern[str]
    paths: re.Pattern[str]
    #: the change that removes it
    until: str


class Boundary(NamedTuple):
    groups: dict[str, re.Pattern[str]]
    raw: dict[str, list[str]]
    private: re.Pattern[str]
    needed: tuple[Needed, ...]
    when_closed: re.Pattern[str]
    residue: tuple[Residue, ...] = ()
    #: the fetched fixtures the export ships as tracked files (`public_export.fixtures`)
    fixtures: re.Pattern[str] = re.compile("(?!)")

    @classmethod
    def load(cls, doc: dict | Path = BOUNDARY) -> Boundary:
        if isinstance(doc, Path):
            doc = tomllib.loads(doc.read_text(encoding="utf-8"))
        if other := sorted(set(doc) - {"public", "private", "needed", "when_closed", "residue", "fixtures"}):
            raise ValueError(f"the boundary has keys it doesn't read: {other}")
        residue = []
        for e in doc.get("residue", []):
            if sorted(e) != ["paths", "pattern", "until", "what"]:
                raise ValueError(f"residue {e.get('what')!r} has keys {sorted(e)}; each has exactly what, pattern, paths "
                                 "and until")
            residue.append(Residue(e["what"], re.compile(e["pattern"]), globs(e["paths"]), e["until"]))
        groups = {name: globs(ps) for name, ps in doc.get("public", {}).items()}
        needed = []
        for e in doc.get("needed", []):
            if other := sorted(set(e) - {"why", "paths", "from", "via"}):
                raise ValueError(f"needed {e.get('why')!r} has keys it doesn't read: {other}; a key after a `[[needed]]` "
                                 "entry is that entry's, so a top-level key goes above `[public]`")
            src, via = e.get("from", "*"), e.get("via", "*")
            if src != "*" and not set(src) <= set(groups):
                raise ValueError(f"needed {e.get('why')!r}: from names {sorted(set(src) - set(groups))}, no public group")
            if via != "*" and not set(via) <= KINDS:
                raise ValueError(f"needed {e.get('why')!r}: via names {sorted(set(via) - KINDS)}, no kind of edge")
            needed.append(Needed(e.get("why", ""), globs(e["paths"]), None if src == "*" else frozenset(src),
                                 None if via == "*" else frozenset(via)))
        return cls(groups, dict(doc.get("public", {})), globs(p for ps in doc.get("private", {}).values() for p in ps),
                   tuple(needed), globs(doc.get("when_closed", [])), tuple(residue), globs(doc.get("fixtures", [])))

    def group(self, path: str) -> str | None:
        return next((g for g, r in self.groups.items() if r.fullmatch(path)), None)

    def is_private(self, path: str) -> bool:
        return bool(self.private.fullmatch(path))


#: the files whose needs are read: what else a public file holds (data, docs, Rust) needs nothing of the tree here
PARSED = (".lean", ".py", ".sh", ".bash", "lakefile.toml", "pyproject.toml")
#: the kinds of edge: what in the source needs the target (`python extra`: an extra's or a dependency group's entry;
#: `python script`: a console script's module)
KINDS = frozenset({"lean import", "lake require", "python import", "python dependency", "python extra", "python script",
                   "path", "command", "package"})
#: the kinds the export's manifest rewrite drops (`public_export.py`): closed by the export, not open, while its build of
#: what it copies still passes
EXPORT_CUTS = frozenset({"lake require", "python extra", "python script"})


class Edge(NamedTuple):
    source: str
    target: str
    line: int
    #: one of ``KINDS``; `package` is a file's need of its own package's files
    kind: str
    what: str

    @property
    def key(self) -> str:
        return f"{self.source} -> {self.target}"

    def __str__(self) -> str:
        return f"{self.source}:{self.line}: {self.kind} {self.what} -> {self.target}"


def _git_files(root: Path) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout.decode()
    return [f for f in out.split("\0") if f]


def _rel(path: str) -> str:
    """A normalized path from the root, `""` for the root."""
    p = Path(os.path.normpath(path)).as_posix()
    return "" if p == "." else p


def _norm_dist(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(req: str) -> str:
    return _norm_dist(re.split(r"[\s\[<>=!~;@(]", req.strip(), maxsplit=1)[0])


class Tree:
    """The tracked files of one checkout, and what each needs."""

    def __init__(self, root: Path = ROOT, files: Iterable[str] | None = None):
        self.root = root.resolve()
        self.files = frozenset(f for f in (_git_files(self.root) if files is None else files)
                               if (self.root / f).is_file() and not (self.root / f).is_symlink())
        self.anchored = ANCHORS.Tree(self.files)
        self.dirs = self.anchored.dirs
        self._edges: dict[str, list[Edge]] = {}

    @functools.cache
    def text(self, rel: str) -> str:
        return (self.root / rel).read_text(encoding="utf-8", errors="replace")

    def toml(self, rel: str) -> dict:
        try:
            return tomllib.loads(self.text(rel))
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"public_boundary: {rel} isn't TOML ({e}), so what it needs can't be read") from e

    def under(self, d: str) -> list[str]:
        return self.anchored.under(d)

    # Lean

    @functools.cached_property
    def lake(self) -> tuple[dict[str, str], dict[str, str], dict[str, object]]:
        """({module: its file}, {Lean file: its package's lakefile}, {lakefile: its package}) of every Lake package the tree
        tracks a `lakefile.toml` of."""
        pkgs, unread = {}, []
        for f in sorted(self.files):
            if f.rpartition("/")[2] == "lakefile.lean":
                unread.append(f"{f}: a lakefile.lean, whose package this reading doesn't follow")
            elif f.rpartition("/")[2] == "lakefile.toml":
                p = MODULE_KEYS.package(self.root / os.path.dirname(f))
                if isinstance(p, str):
                    unread.append(p)
                else:
                    pkgs[f] = p
        if unread:
            raise ValueError("public_boundary: a Lake package it can't read, whose imports it would miss; make its lakefile "
                             "one `tools/verity/lean/module_keys.py` reads:\n" + "\n".join(unread))
        srcs = {lf: _rel(os.path.relpath(p.src, self.root)) for lf, p in pkgs.items()}
        roots = {lf: tuple(HEADERS.name(r) for r in p.roots) for lf, p in pkgs.items()}
        modules: dict[str, str] = {}
        owner: dict[str, str] = {}
        claimed: dict[str, str] = {}
        for f in sorted(x for x in self.files if x.endswith(".lean")):
            for lf, src in srcs.items():
                if src and not f.startswith(src + "/"):
                    continue
                mod = f[len(src) + 1 if src else 0:].removesuffix(".lean").replace("/", ".")
                modules.setdefault(mod, f)
                if any(mod == r or mod.startswith(r + ".") for r in roots[lf]) or mod in pkgs[lf].exes:
                    claimed.setdefault(f, lf)
                    modules[mod] = f
        for f in (x for x in self.files if x.endswith(".lean")):
            owner[f] = claimed.get(f) or max((lf for lf in pkgs if f.startswith(os.path.dirname(lf) + "/")
                                              or os.path.dirname(lf) == ""), key=len, default=None)
        return modules, {f: lf for f, lf in owner.items() if lf}, pkgs

    def _lean(self, f: str) -> list[Edge]:
        modules, owner, _ = self.lake
        src = self.text(f)
        h, starts = HEADERS.header(src), HEADERS.starts(src)
        if h is None or starts is None:
            raise ValueError(f"public_boundary: {f}'s header is cut off (a comment left open, or an import without a "
                             "module's name), so what it imports can't be read")
        out = [Edge(f, owner[f], 1, "package", "its Lake package")] if f in owner else []
        for imp, at in zip(h.imports, starts):
            if (t := modules.get(imp.module)) is not None and t != f:
                out.append(Edge(f, t, src.count("\n", 0, at) + 1, "lean import", imp.module))
        return out

    def _lakefile(self, f: str) -> list[Edge]:
        d = os.path.dirname(f)
        out = [Edge(f, t, 1, "package", "its package's files") for t in (f"{d}/lake-manifest.json" if d else "lake-manifest.json",
                                                                       f"{d}/lean-toolchain" if d else "lean-toolchain")
               if t in self.files]
        text = self.text(f)
        for r in self.toml(f).get("require", []):
            if not isinstance(r, dict) or not isinstance(r.get("path"), str):
                continue
            t = Path(os.path.normpath(Path(d) / r["path"] / "lakefile.toml")).as_posix()
            m = re.search(rf'name\s*=\s*"{re.escape(str(r.get("name", "")))}"', text)
            if t in self.files:
                out.append(Edge(f, t, text.count("\n", 0, m.start()) + 1 if m else 1, "lake require", str(r.get("name"))))
        return out

    # Python

    @functools.cached_property
    def python(self) -> tuple[dict[str, list[str]], dict[str, str], dict[str, str], dict[str, list[str]]]:
        """({import name: its distributions' source roots}, {distribution: its pyproject}, {command: its distribution's
        pyproject}, {pyproject: the roots its suite's `pythonpath` adds})."""
        names: dict[str, list[str]] = defaultdict(list)
        dists, commands, paths = {}, {}, {}
        for f in sorted(x for x in self.files if x.rpartition("/")[2] == "pyproject.toml"):
            doc = self.toml(f)
            d = os.path.dirname(f)
            project, tool = doc.get("project", {}), doc.get("tool", {})
            if isinstance(project.get("name"), str):
                dists[_norm_dist(project["name"])] = f
            for p in tool.get("hatch", {}).get("build", {}).get("targets", {}).get("wheel", {}).get("packages", []):
                at = _rel(os.path.join(d, p))
                names[at.rpartition("/")[2]].append(at.rpartition("/")[0])
            for c in project.get("scripts", {}):
                commands.setdefault(c, f)
            pp = tool.get("pytest", {}).get("ini_options", {}).get("pythonpath", [])
            paths[f] = [r for e in pp if isinstance(e, str) and not (r := _rel(os.path.join(d, e))).startswith("..")]
        return dict(names), dists, commands, paths

    def _pyproject(self, f: str) -> str | None:
        d = os.path.dirname(f)
        while True:
            p = f"{d}/pyproject.toml" if d else "pyproject.toml"
            if p in self.files:
                return p
            if not d:
                return None
            d = os.path.dirname(d)

    def _roots(self, f: str, top: str) -> list[str]:
        d = os.path.dirname(f)
        while d and f"{d}/__init__.py" in self.files:
            d = os.path.dirname(d)
        names, _, _, paths = self.python
        out = [d]
        if (p := self._pyproject(f)) is not None:
            out += paths.get(p, [])
        return out + names.get(top, []) + [""]

    def _module(self, name: str, f: str, more: Iterable[str] = ()) -> list[str]:
        """What a Python import of ``name`` from ``f`` names: its module (or package's `__init__.py`, or a namespace
        package's directory), under the first root holding its top package: the file's own roots, then ``more``.  The
        packages above it are that file's own `package` edges."""
        parts = name.split(".")
        if not parts[0] or parts[0] in sys.stdlib_module_names:
            return []
        for r in (*self._roots(f, parts[0]), *more):
            base = f"{r}/" if r else ""
            if f"{base}{parts[0]}.py" not in self.files and f"{base}{parts[0]}" not in self.dirs:
                continue
            for k in range(len(parts), 0, -1):
                p = base + "/".join(parts[:k])
                hit = next((t for t in (f"{p}.py", f"{p}/__init__.py") if t in self.files), None)
                if hit is None and p in self.dirs and k == len(parts):
                    hit = p
                if hit is not None:
                    return [hit]
            return []
        return []

    def _package(self, name: str, f: str) -> str | None:
        """The directory `importlib.resources.files(name)` is from ``f``: the package's own, or the one holding the module;
        None when the tree holds no module of that whole name."""
        hit = next(iter(self._module(name, f)), None)
        if hit is None or not hit.removesuffix("/__init__.py").removesuffix(".py").endswith(name.replace(".", "/")):
            return None
        return os.path.dirname(hit) if hit in self.files else hit

    def _packages(self, f: str) -> list[str]:
        """The `__init__.py` of each package above ``f``, which importing it runs."""
        out, d = [], os.path.dirname(f)
        while d and (i := f"{d}/__init__.py") in self.files:
            if i != f:
                out.append(i)
            d = os.path.dirname(d)
        return out

    def _command(self, words: list[str]) -> tuple[str, str] | None:
        """(kind, name) of what a command line runs, past `uv run`'s options and `python -m`: a command or a module."""
        i = 0
        while i < len(words):
            w = words[i]
            if w in ("uv", "uvx") and i + 1 < len(words) and (w == "uvx" or words[i + 1] == "run"):
                i += 1 if w == "uvx" else 2
                while i < len(words) and words[i].startswith("-"):
                    i += 2 if words[i] in UV_VALUED else 1
                continue
            if re.fullmatch(r"python[\d.]*", w):
                if i + 2 < len(words) and words[i + 1] == "-m":
                    return "module", words[i + 2]
                return None
            return "command", w
        return None

    def _ran(self, f: str, line: int, words: list[str]) -> list[Edge]:
        got = self._command(words)
        if got is None:
            return []
        kind, name = got
        if kind == "module":
            return [Edge(f, t, line, "command", f"python -m {name}") for t in self._module(name, f)[:1]]
        _, _, commands, _ = self.python
        return [Edge(f, commands[name], line, "command", name)] if name in commands else []

    def _python(self, f: str) -> list[Edge]:
        text = self.text(f)
        try:
            tree = ast.parse(text, filename=f)
        except SyntaxError as e:
            raise ValueError(f"public_boundary: {f} isn't Python ({e}), so what it imports can't be read") from e
        nodes = list(ast.walk(tree))
        out = [Edge(f, i, 1, "package", "a package above it") for i in self._packages(f)]
        if (p := self._pyproject(f)) is not None:
            out.append(Edge(f, p, 1, "package", "its pyproject"))
            if "tests" in f.split("/")[:-1]:
                suite, d = os.path.dirname(p), os.path.dirname(f)
                while True:
                    if (c := f"{d}/conftest.py" if d else "conftest.py") in self.files and c != f:
                        out.append(Edge(f, c, 1, "package", "a conftest.py above it"))
                    if d == suite or not d:
                        break
                    d = os.path.dirname(d)
        for name, line, _ in IMPORTS._imports(self.root / f, self.root, (), nodes):
            out += [Edge(f, t, line, "python import", name) for t in self._module(name, f) if t != f]
        # `from M import a` also loads M.a when that is a module, which a private file of a public package can be
        for n in nodes:
            if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                out += [Edge(f, t, n.lineno, "python import", f"from {n.module} import {a.name}")
                        for a in n.names for t in self._module(f"{n.module}.{a.name}", f)[:1]
                        if t != f and t.endswith((f"/{a.name}.py", f"/{a.name}/__init__.py", f"/{a.name}"))]
        for line, reached in ANCHORS.joins(f, text, lambda name: self._package(name, f)):
            out += self._path(f, line, reached)
        _, _, commands, _ = self.python
        if not any(c in text for c in commands) and not ("python" in text and "-m" in text):
            return out
        for n in nodes:
            if not isinstance(n, ast.Call) or not n.args:
                continue
            fn = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
            if fn not in RUNNERS:
                continue
            a = n.args[0]
            if isinstance(a, (ast.List, ast.Tuple)):
                words = [e.value for e in a.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            elif isinstance(a, ast.Constant) and isinstance(a.value, str):
                words = a.value.split()
            else:
                continue
            out += self._ran(f, n.lineno, words)
        return out

    def _path(self, f: str, line: int, reached: str) -> list[Edge]:
        if reached == ANCHORS.OUTSIDE or reached.startswith(ANCHORS.OUTSIDE + "/"):
            return []
        t = self.anchored.tracked_prefix(reached)
        return [Edge(f, t, line, "path", reached)] if t else []

    def _shell(self, f: str) -> list[Edge]:
        text, out = self.text(f), []
        for line, reached in ANCHORS.shell_joins(f, text):
            out += self._path(f, line, reached)
        lines, at = text.splitlines(), 0
        while at < len(lines):
            n, raw = at + 1, lines[at]
            while raw.endswith("\\") and at + 1 < len(lines):
                at += 1
                raw = raw[:-1] + " " + lines[at]
            at += 1
            if raw.lstrip().startswith("#"):
                continue
            # a path from the root only when the tree has exactly it: a script's relative path is often from its own place
            out += [Edge(f, p, n, "path", p) for m in SH_PATH.finditer(raw)
                    if (p := m.group(1).rstrip("/.")) in self.files or p in self.dirs]
            lex = shlex.shlex(raw.replace("$(", " ( "), posix=True, punctuation_chars=";&|()`")
            lex.whitespace_split, lex.commenters = True, "#"
            try:
                words = list(lex)
            except ValueError:
                # a quote this line leaves open (a string over several lines): its words as they stand
                words = re.findall(r"[;&|()`]+|[^\s;&|()`'\"]+", raw.replace("$(", " ( "))
            for i, w in enumerate(words):
                if i == 0 or words[i - 1] in SH_STARTS:
                    out += self._ran(f, n, words[i:])
        return out

    def _manifest(self, f: str) -> list[Edge]:
        doc = self.toml(f)
        _, dists, _, _ = self.python
        out = [Edge(f, t, 1, "package", "the workspace's") for t in ("pyproject.toml", "uv.lock") if t in self.files and t != f]
        project = doc.get("project", {})
        reqs = [*((r, "python dependency") for r in project.get("dependencies", [])),
                *((r, "python extra") for rs in project.get("optional-dependencies", {}).values() for r in rs),
                *((r, "python extra") for rs in doc.get("dependency-groups", {}).values() for r in rs if isinstance(r, str))]
        text = self.text(f)
        for r, kind in reqs:
            if (t := dists.get(_requirement_name(r))) is not None and t != f:
                at = text.find(f'"{r}"')
                out.append(Edge(f, t, text.count("\n", 0, at) + 1 if at >= 0 else 1, kind, r))
        for name, entry in project.get("scripts", {}).items():
            if isinstance(entry, str):
                at = text.find(f'"{entry}"')
                out += [Edge(f, t, text.count("\n", 0, at) + 1 if at >= 0 else 1, "python script", f"{name} = {entry}")
                        for t in self._module(entry.partition(":")[0].strip(), f)]
        return out

    def prefetch(self, files: Iterable[str], workers: int = 4) -> None:
        """Read what each of ``files`` needs in forked workers, each over a share of them: one file's edges read nothing of
        another's, and a Python file's parse is most of the time a judgment takes."""
        todo = sorted(f for f in files if f not in self._edges)
        n = min(workers, os.cpu_count() or 1, len(todo) // 100)
        if n < 2 or "fork" not in multiprocessing.get_all_start_methods():
            return
        ctx, procs = multiprocessing.get_context("fork"), []
        for i in range(n):
            r, w = ctx.Pipe(duplex=False)
            p = ctx.Process(target=self._send, args=(todo[i::n], w), daemon=True)
            p.start()
            w.close()
            procs.append((p, r))
        for p, r in procs:
            got = r.recv()
            p.join()
            if p.exitcode != 0 or not isinstance(got, list):
                raise RuntimeError(f"public_boundary: a worker failed ({p.exitcode}): {got}")
            self._edges |= {f: [Edge(*e) for e in es] for f, es in got}

    def _send(self, files: list[str], w) -> None:
        try:
            w.send([(f, [tuple(e) for e in self.edges(f)]) for f in files])
        except BaseException as e:
            w.send(f"{type(e).__name__}: {e}")
            raise
        finally:
            w.close()

    def edges(self, f: str) -> list[Edge]:
        """What ``f`` needs, each target once with its first site, one the export's rewrite doesn't cut if it has one."""
        if f not in self._edges:
            name = f.rpartition("/")[2]
            got = (self._lean(f) if f.endswith(".lean") else self._lakefile(f) if name == "lakefile.toml"
                   else self._manifest(f) if name == "pyproject.toml" else self._python(f) if f.endswith(".py")
                   else self._shell(f) if f.endswith((".sh", ".bash")) else [])
            first: dict[str, Edge] = {}
            for e in got:
                if e.target != f and (e.target not in first or first[e.target].kind in EXPORT_CUTS
                                      and e.kind not in EXPORT_CUTS):
                    first[e.target] = e
            self._edges[f] = sorted(first.values(), key=lambda e: (e.line, e.target))
        return self._edges[f]


class Judgment(NamedTuple):
    #: {public file: why it is public: its group, or the entry that pulled it in}
    public: dict[str, str]
    #: the open edges
    edges: list[Edge]
    #: the edges of a kind the export's rewrite drops (`EXPORT_CUTS`)
    cut: list[Edge] = []


def judge(tree: Tree, boundary: Boundary) -> Judgment:
    """The public set and its edges into what isn't public."""
    why: dict[str, str] = {}
    origin: dict[str, int] = {}
    for f in sorted(tree.files):
        if not boundary.is_private(f) and (g := boundary.group(f)) is not None:
            why[f] = g
    tree.prefetch(f for f in tree.files if f.rpartition("/")[2].endswith(PARSED) and not boundary.is_private(f)
                  and (f in why or boundary.when_closed.fullmatch(f) or any(n.paths.fullmatch(f) for n in boundary.needed)))

    def pullable(t: str, e: Edge) -> int | None:
        if boundary.is_private(t):
            return None
        for i, n in enumerate(boundary.needed):
            if n.paths.fullmatch(t) and (n.via is None or e.kind in n.via) \
                    and (n.source is None or why.get(e.source) in n.source or origin.get(e.source) == i):
                return i
        return None

    def close(todo: list[str]) -> None:
        while todo:
            f = todo.pop()
            for e in tree.edges(f):
                ts = [e.target] if e.target in tree.files else tree.under(e.target)
                if e.target not in tree.files and (e.target in why or any(t in why for t in ts)):
                    continue
                for t in ts:
                    if t not in why and (i := pullable(t, e)) is not None:
                        why[t], origin[t] = f"needed: {boundary.needed[i].why}", i
                        todo.append(t)

    close(list(why))
    closed = {f for f in tree.files if f not in why and boundary.when_closed.fullmatch(f) and not boundary.is_private(f)}
    while True:
        keep = {f for f in closed if all(e.target in why or e.target in closed or pullable(e.target, e) is not None
                                         or e.target not in tree.files and any(t in why or t in closed for t in tree.under(e.target))
                                         for e in tree.edges(f))}
        if keep == closed:
            break
        closed = keep
    why |= dict.fromkeys(closed, "when closed")
    close(sorted(closed))
    edges = [e for f in sorted(why) for e in tree.edges(f)
             if not (e.target in why if e.target in tree.files else any(t in why for t in tree.under(e.target)))]
    # `from M import a`'s edge to the module M.a is one more to cut only while the file's edge to M isn't open
    open_ = {(e.source, e.target) for e in edges}
    edges = [e for e in edges if not (e.kind == "python import" and e.what.startswith("from ")
                                      and (e.source, next(iter(tree._module(e.what.split()[1], e.source)), "")) in open_)]
    return Judgment(why, [e for e in edges if e.kind not in EXPORT_CUTS], [e for e in edges if e.kind in EXPORT_CUTS])


def drags(tree: Tree, boundary: Boundary, judgment: Judgment, target: str) -> tuple[set[str], set[str]]:
    """(what moving ``target`` to the public side would move with it: the files neither public nor `private` it needs,
    itself included, directly or through one another; the `private` files among what they need, which can't move)."""
    seen, stuck, todo = set(), set(), [target]
    while todo:
        t = todo.pop()
        for x in [t] if t in tree.files else tree.under(t):
            if x in seen or x in stuck or x in judgment.public:
                continue
            if boundary.is_private(x):
                stuck.add(x)
                continue
            seen.add(x)
            todo += [e.target for e in tree.edges(x)]
    return seen, stuck


def read_allowed(path: Path = ALLOWED) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


HEADER = ("# Each edge from a public file to one the export doesn't copy, `SOURCE -> TARGET`, that the boundary check\n"
          "# (`public_boundary.py`) allows. It only shrinks: cut an edge, then `--prune`.\n")


def write_allowed(edges: Iterable[str], path: Path = ALLOWED) -> None:
    path.write_text(HEADER + "".join(f"{e}\n" for e in sorted(set(edges))), encoding="utf-8")


def _area(path: str) -> str:
    parts = path.split("/")
    return "/".join(parts[:3] if parts[0] in ("security", "verity") else parts[:2])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="public_boundary", description=__doc__.split("\n\n")[0])
    p.add_argument("--root", action="append", type=Path, help="a checkout to judge (repeatable; default: this one)")
    p.add_argument("--boundary", type=Path, default=BOUNDARY)
    p.add_argument("--allowed", type=Path, default=ALLOWED)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="every edge, by the public group its file is in")
    mode.add_argument("--public", action="store_true", help="the public set, one file a line")
    mode.add_argument("--json", action="store_true", help="the public set, every edge, and what moving its target moves")
    mode.add_argument("--prune", action="store_true", help="drop from ALLOWED the edges no tree given has")
    mode.add_argument("--record", action="store_true", help="ALLOWED is the edges of the trees given")
    ns = p.parse_args(argv)
    boundary = Boundary.load(ns.boundary)
    trees = [Tree(r) for r in ns.root or [ROOT]]
    judged = [(t, judge(t, boundary)) for t in trees]
    found = {e.key for _, j in judged for e in j.edges}
    allowed = read_allowed(ns.allowed)
    if ns.record or ns.prune:
        write_allowed(found if ns.record else set(allowed) & found, ns.allowed)
        print(f"public_boundary: {ns.allowed.name} lists {len(found if ns.record else set(allowed) & found)} edges")
        return 0
    if ns.public:
        print("\n".join(sorted(set().union(*(j.public for _, j in judged)))))
        return 0
    if ns.json:
        out = []
        for t, j in judged:
            t.prefetch(f for f in t.files if f.rpartition("/")[2].endswith(PARSED))
            moved = {x: drags(t, boundary, j, x) for x in {e.target for e in j.edges}}
            out.append({"root": str(t.root), "public": j.public,
                        "edges": [e._asdict() | {"private": boundary.is_private(e.target),
                                                 "group": j.public[e.source].split(":")[0],
                                                 "moves": sorted(moved[e.target][0]),
                                                 "blocked_by": sorted(moved[e.target][1])} for e in j.edges],
                        "cut_by_export": [e._asdict() | {"group": j.public[e.source].split(":")[0]} for e in j.cut]})
        print(json.dumps(out, indent=1))
        return 0
    if ns.all:
        for t, j in judged:
            by: dict[str, list[Edge]] = defaultdict(list)
            for e in j.edges:
                by[j.public[e.source].split(":")[0]].append(e)
            print(f"{t.root}: {len(j.public)} public files, {len(j.edges)} edges, {len(j.cut)} more the export's rewrite cuts")
            for g, es in sorted(by.items()):
                print(f"\n{g}: {len(es)}")
                print("\n".join(f"  {e}" for e in es))
            if j.cut:
                print(f"\ncut by the export's rewrite (closed while its build passes): {len(j.cut)}")
                print("\n".join(f"  {e}" for e in j.cut))
        return 0
    allowed_set = set(allowed)
    new = [e for _, j in judged for e in j.edges if e.key not in allowed_set]
    stale = len(allowed_set - found)
    if new:
        print(FIX + "\n" + "\n".join(map(str, new)))
    print(f"public_boundary: {len(found)} edges, {len(new)} new; {stale} listed edges no tree given has (--prune)")
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main())
