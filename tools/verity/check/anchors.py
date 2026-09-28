"""`anchors`: every repository path a Python or shell file reaches from its own place still reaches it after a move.

    uv run python tools/verity/check/anchors.py BASE HEAD [--json]

A layout move renames files and rewrites names, and `Path(__file__).resolve().parents[3] / "data"` is neither: a file renamed
to another depth, or away from the files beside it, reaches somewhere else, and only a test that reads that path notices.
For each Python file at BASE that HEAD still has (where git's rename detection put it, or in place), whatever a join anchored
on `__file__` reached at BASE, some join of the file at HEAD reaches where it went: the tracked part of the path (its
longest prefix that is a tracked file or a directory holding one) where those files went, the rest (a build output, a
fetched fixture) unchanged below it. A join that left the tree or reached nothing tracked at BASE is not judged, and a
change that drops a reference on purpose is listed too, so run it on the move commit alone. Exit 0 when every target is
still reached, 1 listing those that aren't, 2 on a git error.

Anchored means, in Python, a chain on `Path(__file__)`: `.resolve()`, `.absolute()`, `.parent`, `.parents[N]`, `/ "lit"`,
`.joinpath("lit", ...)`, `.with_name("lit")`, and names bound to one (`os.path` arithmetic on `__file__` isn't read); in a
`*.sh`, `$(dirname "$0")/path` or `${BASH_SOURCE[0]}`'s, and `"$VAR/path"` for a `VAR=$(cd "<one of those>/.." && pwd)`
(a relative path after a bare `cd "$(dirname "$0")"` isn't read).
"""
from __future__ import annotations

import argparse
import ast
import bisect
import functools
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
#: what `_up` returns for a path above the repository root
OUTSIDE = ".."
SOURCES = (".py", ".sh")


def _up(path: str, n: int = 1) -> str:
    for _ in range(n):
        if path == OUTSIDE or path.startswith(OUTSIDE + "/"):
            return OUTSIDE
        path = path.rpartition("/")[0] if "/" in path else ("" if path else OUTSIDE)
    return path


def _join(path: str, *parts: str) -> str:
    if path == OUTSIDE:
        return OUTSIDE
    out = [p for p in path.split("/") if p]
    for part in "/".join(parts).split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not out:
                return OUTSIDE
            out.pop()
        else:
            out.append(part)
    return "/".join(out)


def anchor(node: ast.AST, here: str, names: dict[str, str]) -> str | None:
    """The repository path (relative to the root, `""` for the root, `OUTSIDE` above it) the expression reaches when
    ``here`` is the file's path; None when it isn't anchored on `__file__`."""
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Name) and f.id == "Path" and len(node.args) == 1 and isinstance(node.args[0], ast.Name) \
                and node.args[0].id == "__file__":
            return here
        if isinstance(f, ast.Attribute) and f.attr in ("resolve", "absolute") and not node.args:
            return anchor(f.value, here, names)
        lits = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if isinstance(f, ast.Attribute) and f.attr == "joinpath" and node.args and len(lits) == len(node.args):
            a = anchor(f.value, here, names)
            return None if a is None else _join(a, *lits)
        if isinstance(f, ast.Attribute) and f.attr == "with_name" and len(node.args) == 1 and lits:
            a = anchor(f.value, here, names)
            return None if a is None else _join(_up(a), lits[0])
        return None
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        a = anchor(node.value, here, names)
        return None if a is None else _up(a)
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "parents" \
            and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
        a = anchor(node.value.value, here, names)
        return None if a is None else _up(a, node.slice.value + 1)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and isinstance(node.right, ast.Constant) \
            and isinstance(node.right.value, str):
        a = anchor(node.left, here, names)
        return None if a is None else _join(a, node.right.value)
    if isinstance(node, ast.Name):
        return names.get(node.id)
    return None


def bindings(tree: ast.AST, here: str) -> dict[str, str]:
    """Each name some assignment in the file binds to an anchored path (the last one wins; a name rebound to two paths in
    different scopes is read as one, which a move's pairing keeps the same before and after)."""
    names: dict[str, str] = {}
    assigns = [n for n in ast.walk(tree) if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None]
    for _ in range(3):
        for node in assigns:
            if (a := anchor(node.value, here, names)) is not None:
                for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                    if isinstance(t, ast.Name):
                        names[t.id] = a
    return names


def joins(here: str, text: str) -> list[tuple[int, str]]:
    """(line, reached path) for each outermost anchored expression of the file other than a bare name, in source order."""
    if "__file__" not in text:
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    names, out = bindings(tree, here), []

    def visit(node: ast.AST) -> None:
        if not isinstance(node, ast.Name) and (a := anchor(node, here, names)) is not None:
            out.append((node.lineno, node.col_offset, a))
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return [(line, a) for line, _, a in sorted(out)]


_SH_DIR = r'\$\(\s*dirname\s+"?\$(?:0|\{BASH_SOURCE\[0\]\}|\{BASH_SOURCE\}|BASH_SOURCE)"?\s*\)'
_SH_BASE = rf'(?:{_SH_DIR}|\$\{{?(?P<var>\w+)\}}?)'
_SH_REL = r'(?P<rel>(?:/[\w.@+-]+)*)'
#: `VAR=$(cd "$(dirname "$0")/.." && pwd)`, `VAR="$(cd "$HERE/../.." && pwd)"`
_SH_ASSIGN = re.compile(rf'(?:^|[;\s(])(?:local\s+|export\s+|readonly\s+)?(?P<name>\w+)="?\$\(\s*cd\s+"?{_SH_BASE}{_SH_REL}"?\s*'
                        rf'(?:&>\s*/dev/null\s*|>\s*/dev/null\s*)?&&\s*pwd(?:\s+-P)?\s*\)"?')
#: `$HERE/x/y`, `"${ROOT}/x"`, `$(dirname "$0")/x`
_SH_REF = re.compile(rf'{_SH_BASE}{_SH_REL}')


def shell_joins(here: str, text: str) -> list[tuple[int, str]]:
    """(line, reached path) for each path a shell script names from its own directory: `$(dirname "$0")` or
    `${BASH_SOURCE[0]}`, and variables `cd`-assigned from one."""
    if "dirname" not in text:
        return []
    names: dict[str, str] = {}
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        code = line.split(" #")[0] if not line.lstrip().startswith("#") else ""
        assigns = list(_SH_ASSIGN.finditer(code))
        refs = [m for m in _SH_REF.finditer(code) if not any(a.start() <= m.start() < a.end() for a in assigns)]
        for m in sorted(assigns + refs, key=lambda m: m.start()):
            base = names.get(m["var"]) if m["var"] else _up(here)
            if base is None:
                continue
            if m.re is _SH_ASSIGN:
                names[m["name"]] = _join(base, m["rel"])
            elif m["rel"] or m["var"]:
                out.append((n, _join(base, m["rel"])))
    return out


def reaches(here: str, text: str) -> list[tuple[int, str]]:
    return shell_joins(here, text) if here.endswith(".sh") else joins(here, text)


class Tree:
    """The tracked files of one revision."""

    def __init__(self, files: frozenset[str]):
        self.files = files
        self.sorted = sorted(files)

    def under(self, prefix: str) -> list[str]:
        lo = bisect.bisect_left(self.sorted, prefix + "/")
        return self.sorted[lo:bisect.bisect_left(self.sorted, prefix + "0")]  # "0" sorts just after "/"

    @functools.cached_property
    def dirs(self) -> frozenset[str]:
        return frozenset(_up(f, i) for f in self.files for i in range(1, f.count("/") + 1))

    def has(self, path: str) -> bool:
        return path == "" or path in self.files or path in self.dirs

    def tracked_prefix(self, path: str) -> str | None:
        """The longest prefix of ``path`` that is tracked here, `""` being the root."""
        if path == OUTSIDE or path.startswith(OUTSIDE + "/"):
            return None
        while path and not self.has(path):
            path = _up(path)
        return path


def destinations(prefix: str, base: Tree, moved: dict[str, str]) -> set[str]:
    """Where a tracked BASE path went: a file to its new path; a directory to each place its files went with their path
    below it unchanged (more than one when the move split it); the root to itself."""
    if prefix == "":
        return {""}
    if prefix in base.files:
        return {moved[prefix]} if prefix in moved else set()
    out = set()
    for f in base.under(prefix):
        if f in moved and moved[f].endswith(f[len(prefix):]):
            out.add(moved[f][:len(moved[f]) - len(f) + len(prefix)])
    return out


def check(base_files: frozenset[str], moved: dict[str, str], read_base, read_head) -> list[str]:
    """What each anchored join of a Python or shell file at BASE fails to reach at HEAD. ``moved`` maps each BASE file HEAD still has
    to its HEAD path; ``read_base`` and ``read_head`` return a file's text at each."""
    base, problems = Tree(base_files), []
    for old in sorted(f for f in base_files if f.endswith(SOURCES) and f in moved):
        new = moved[old]
        before = [(line, was, prefix) for line, was in reaches(old, read_base(old))
                  if (prefix := base.tracked_prefix(was)) is not None]
        if not before:
            continue
        reached = {now for _, now in reaches(new, read_head(new))}
        for line, was, prefix in before:
            rest = was[len(prefix):].lstrip("/")
            want = {_join(d, rest) for d in destinations(prefix, base, moved)}
            if not want & reached:
                where = " or ".join(sorted(w or "." for w in want)) if want else "nowhere (HEAD has none of its files)"
                problems.append(f"{old}:{line} (now {new}): reached {was or '.'}, which went to {where}; no join there reaches it")
    return list(dict.fromkeys(problems))


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


def renames(base: str, head: str, cwd: Path = ROOT) -> tuple[frozenset[str], dict[str, str]]:
    base_files = frozenset(_git("ls-tree", "-r", "--name-only", "-z", base, cwd=cwd).split("\0")) - {""}
    head_files = frozenset(_git("ls-tree", "-r", "--name-only", "-z", head, cwd=cwd).split("\0")) - {""}
    moved = {f: f for f in base_files & head_files}
    out = _git("diff", "--name-status", "-z", "-M", base, head, cwd=cwd).split("\0")
    i = 0
    while i < len(out) - 1:
        status = out[i]
        if status.startswith(("R", "C")):
            moved[out[i + 1]] = out[i + 2]
            i += 3
        else:
            i += 2
    return base_files, moved


def texts(rev: str, paths: list[str], cwd: Path = ROOT) -> dict[str, str]:
    """Each file's text at ``rev``, through one `git cat-file --batch`."""
    out = subprocess.run(["git", "cat-file", "--batch"], cwd=cwd, input="".join(f"{rev}:{p}\n" for p in paths).encode(),
                         capture_output=True, check=True).stdout
    got, i = {}, 0
    for p in paths:
        header_end = out.index(b"\n", i)
        size = int(out[i:header_end].split()[2])
        got[p] = out[header_end + 1:header_end + 1 + size].decode("utf-8", "replace")
        i = header_end + 1 + size + 1
    return got


def run(base: str, head: str, cwd: Path = ROOT) -> list[str]:
    base_files, moved = renames(base, head, cwd)
    olds = sorted(f for f in base_files if f.endswith(SOURCES) and f in moved)
    before, after = texts(base, olds, cwd), texts(head, [moved[f] for f in olds], cwd)
    return check(base_files, moved, before.__getitem__, after.__getitem__)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="anchors", description=__doc__.split("\n\n")[0])
    p.add_argument("base")
    p.add_argument("head")
    p.add_argument("--json", action="store_true")
    ns = p.parse_args(argv)
    try:
        problems = run(ns.base, ns.head, ROOT)
    except subprocess.CalledProcessError as e:
        print(f"anchors: git failed: {e.stderr.strip()}", file=sys.stderr)
        return 2
    if ns.json:
        print(json.dumps({"ok": not problems, "problems": problems}, indent=2))
    else:
        print("\n".join(problems) or f"anchors: every anchored path from {ns.base} still reaches its target at {ns.head}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
