"""`tracked_tree`: the tracked files of one checkout, and where a Python import and a Lean import of one of them resolve.

`tests/test_boundaries.py` reads the import order through it: `Tree.lake` gives every Lake package the tree tracks a
`lakefile.toml` of, with each Lean file's package, and `Tree._lakefile` the packages a lakefile requires by path; `Tree._module`
resolves a Python import of a name from a file through the file's own directory (the first above it outside a package, as a
script or pytest puts it on the path), its suite's `pythonpath`, the workspace's distributions and the root.
"""
from __future__ import annotations

import functools
import importlib.util
import os
import re
import subprocess
import sys
import tomllib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


#: the one reading of a Lean header and of a lakefile, and the anchored-path reader
HEADERS = _load("tracked_tree_headers", ROOT / "tools" / "verity" / "lean" / "headers.py")
MODULE_KEYS = _load("tracked_tree_module_keys", ROOT / "tools" / "verity" / "lean" / "module_keys.py")
ANCHORS = _load("tracked_tree_anchors", HERE / "anchors.py")


class Require(NamedTuple):
    """A lakefile's `require` of another tracked package by path."""
    source: str
    target: str
    line: int
    kind: str
    what: str


def _git_files(root: Path) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout.decode()
    return [f for f in out.split("\0") if f]


def _rel(path: str) -> str:
    """A normalized path from the root, `""` for the root."""
    p = Path(os.path.normpath(path)).as_posix()
    return "" if p == "." else p


class Tree:
    """The tracked files of one checkout."""

    def __init__(self, root: Path = ROOT, files: Iterable[str] | None = None):
        self.root = root.resolve()
        self.files = frozenset(f for f in (_git_files(self.root) if files is None else files)
                               if (self.root / f).is_file() and not (self.root / f).is_symlink())
        self.dirs = ANCHORS.Tree(self.files).dirs

    @functools.cache
    def text(self, rel: str) -> str:
        return (self.root / rel).read_text(encoding="utf-8", errors="replace")

    def toml(self, rel: str) -> dict:
        try:
            return tomllib.loads(self.text(rel))
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"tracked_tree: {rel} isn't TOML ({e}), so what it names can't be read") from e

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
            raise ValueError("tracked_tree: a Lake package it can't read, whose imports it would miss; make its lakefile "
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

    def _lakefile(self, f: str) -> list[Require]:
        """The tracked packages the lakefile ``f`` requires by path."""
        d = os.path.dirname(f)
        out = []
        text = self.text(f)
        for r in self.toml(f).get("require", []):
            if not isinstance(r, dict) or not isinstance(r.get("path"), str):
                continue
            t = Path(os.path.normpath(Path(d) / r["path"] / "lakefile.toml")).as_posix()
            m = re.search(rf'name\s*=\s*"{re.escape(str(r.get("name", "")))}"', text)
            if t in self.files:
                out.append(Require(f, t, text.count("\n", 0, m.start()) + 1 if m else 1, "lake require", str(r.get("name"))))
        return out

    # Python

    @functools.cached_property
    def python(self) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """({import name: its distributions' source roots}, {pyproject: the roots its suite's `pythonpath` adds})."""
        names: dict[str, list[str]] = defaultdict(list)
        paths = {}
        for f in sorted(x for x in self.files if x.rpartition("/")[2] == "pyproject.toml"):
            doc = self.toml(f)
            d = os.path.dirname(f)
            tool = doc.get("tool", {})
            for p in tool.get("hatch", {}).get("build", {}).get("targets", {}).get("wheel", {}).get("packages", []):
                at = _rel(os.path.join(d, p))
                names[at.rpartition("/")[2]].append(at.rpartition("/")[0])
            pp = tool.get("pytest", {}).get("ini_options", {}).get("pythonpath", [])
            paths[f] = [r for e in pp if isinstance(e, str) and not (r := _rel(os.path.join(d, e))).startswith("..")]
        return dict(names), paths

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
        names, paths = self.python
        out = [d]
        if (p := self._pyproject(f)) is not None:
            out += paths.get(p, [])
        return out + names.get(top, []) + [""]

    def _module(self, name: str, f: str, more: Iterable[str] = ()) -> list[str]:
        """What a Python import of ``name`` from ``f`` names: its module (or package's `__init__.py`, or a namespace
        package's directory), under the first root holding its top package: the file's own roots, then ``more``."""
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
