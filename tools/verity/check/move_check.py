"""`move-check`: certify that BASE..HEAD is a pure move, so the lander may schedule it first in a quiet window, alone in its
train (Daniel, 4 Oct). It relaxes no gate: a certified move still goes through `check`.

    uv run python tools/verity/check/move_check.py BASE HEAD [--map OLD=NEW ...] [--json]

Every path git sees changed must be one of:
- a rename with byte-identical content and mode (git's 100% rename);
- an edit, in place or with a rename, to an allowlisted path-bearing file (`PATH_FILES`, `PATH_NAMES`, a skill's `SKILL.md`)
  that leaves it unchanged once each old path in it is rewritten to its new one, a quoted relative path (`".."`,
  `"../level3"`) being resolved against the file's own directory first;
- an edit to a `*.py` file whose changed lines only rewrite the module of an `import M` / `from M import` line, from where
  a renamed module was to where it is now; in `PY_PATH_FILES`, also a path (a pathlib chain `ROOT / "a" / "b"` read as
  `"a/b"`) or a dotted name (`verity.ml.flock.tool:FLOCK_PURE`); in `research`'s sources, also the module of a `-m` line;
- an added line in `REGISTRIES` that registers exactly a package the range created: a root `pyproject.toml` workspace
  member (or its `tool.uv.sources` entry) for a directory that is new at HEAD, a `tests/test_lean_packages.py` entry for a
  new Lake package with its `lean-audit.json`;
- an edit to a `*.lean` file whose changed lines only rewrite an `import M` to the new module of a `.lean` file renamed in
  the range (a module is its path below its Lake package, `/` as `.`, without `.lean`), as when a Lean split gives the
  proof modules a root of their own while their declarations keep their namespace;
- an edit to a `lean-audit.json` that keeps `pins` (the guarantees' records; `guarantees` too, if a lock has it) and
  `reads` byte for byte, may change the structural `roots` and `proved_in` (each `proved_in` entry a Lake package at HEAD),
  and changes any other field only by moved paths and module names; a `runs` entry may move to another lock the range
  touches (a proofs package's new one), its runner module renamed and its package-relative paths following the move, and
  must end up in exactly one, otherwise unchanged (`runs_moved`);
- a new `lakefile.toml`, `lakefile.lean`, `lake-manifest.json`, `lean-audit.json` or `lean-toolchain` at the root of a new
  Lake package that the range moved `.lean` files into from one source package, which is a copy of the source package's
  file (`copy_change`): a lakefile may differ only in package and library names, `roots`, `globs`, `defaultTargets`,
  comments and `require`s by relative path to a Lake package at HEAD; a manifest only in its name, the moved paths of
  its path packages, and new path packages; a lock lists no guarantees and keeps each field it keeps; a toolchain is
  identical.
Anything else is refused, with what differed and its first offending hunk. Exit 0 certified, 1 refused, 2 on a usage or
git error.

The path map comes from the renames. Each rename contributes the shortest prefix pair (old directory, new directory) whose
whole subtree in BASE moved there file by file. When only `verity/core/protocols/pous` moved, `protocols -> security_proofs` is never
inferred. Renames are git's exact ones, and its similar ones (`-M`) as candidates, each edit then judged as above.
`--map OLD=NEW` adds an entry, and a dotted one (no `/`) also maps a Python or Lean module and those below it.
"""
from __future__ import annotations

import argparse
import bisect
import difflib
import functools
import json
import posixpath
import re
import subprocess
import sys
import tomllib

PATH_FILES = {"tools/verity/check/ci.toml", "AGENTS.md", "fixtures/artifacts.json", "tools/research/src/research/pods/nebius/deploy.toml"}
#: Python files that name the tree in strings: each may rewrite a path (a `ROOT / "a" / "b"` chain read as `"a/b"`) or a
#: dotted name (`verity.ml.flock.tool:FLOCK_PURE`, a module) to where it moved, besides its imports
PY_PATH_FILES = {
    "tests/test_lean_packages.py",                                        # the Lean packages
    "tools/research/src/research/store/tools_registry.py",                # REGISTRY: tool -> dotted repo path
    "tools/research/src/research/store/vocab.py",                         # its mirror of PROOF_CLASSES names their source
    "benchmarks/numerical/python/verity_numerical/bench/lowerings.py",      # DIRECTORIES
}
#: `research`'s sources, whose `python -m M` / `"-m", "M"` lines may rewrite M to where the module moved
MODULE_ARG_PREFIX = "tools/research/src/research/"
#: the files where a new package is registered by an added line (`registration`)
REGISTRIES = {"pyproject.toml": "workspace", "tests/test_lean_packages.py": "lean"}
PATH_NAMES = {"pyproject.toml", "lakefile.lean", "lakefile.toml", "lake-manifest.json", ".gitignore"}
#: files whose quoted relative paths name another directory of the tree (a Lake `require`, a manifest's `dir`, a uv source)
RELATIVE_NAMES = {"pyproject.toml", "lakefile.lean", "lakefile.toml", "lake-manifest.json"}
LAKEFILES = ("lakefile.toml", "lakefile.lean")
LAKE_COPIES = {*LAKEFILES, "lake-manifest.json", "lean-audit.json", "lean-toolchain"}
#: a lock's records of what its guarantees say, and its structural fields, which a split changes
PROTECTED, STRUCTURAL = ("pins", "guarantees", "reads"), ("roots", "proved_in")
IMPORT = re.compile(r"^(\s*from\s+)([\w.]+)(\s+import\b.*)$|^(\s*import\s+)([\w.]+)(.*)$")
LEAN_IMPORT = re.compile(r"^(\s*(?:(?:public|private|meta)\s+)*import\s+(?:all\s+)?)([\w.]+)(\s*(?:--.*)?)$")
RELATIVE = re.compile(r"""(["'])(\.\.?(?:/[^"'\s]*)?)\1""")
#: the lines a destination package's lakefile may change from the source package's
LAKE_LINE = re.compile(r"""^\s*(
    | \#.* | --.*
    | \[\[(require|lean_lib)\]\]
    | (name|roots|globs|defaultTargets|path) \s*=.*
    | package \s+\S+ (\s+where)?
    | (@\[default_target\]\s*)? lean_lib \s+\S+ (\s+where)?
    | require \s+\S+ \s+from\s+"[^"]*"
    | (roots|globs|defaultTargets|name) \s*:=.*
    )\s*$""", re.X)
REQUIRE_PATH = re.compile(r"^\s*(path\s*=|require\s)")
MANIFEST_PATH_KEYS = {"type", "scope", "name", "manifestFile", "inherited", "dir", "configFile"}
HUNK_LINES = 8


def git(*args: str, cwd: str | None = None) -> bytes:
    return subprocess.run(["git", "-c", "diff.renameLimit=1000000", *args], cwd=cwd, check=True, capture_output=True).stdout


def kind(path: str) -> str | None:
    name = posixpath.basename(path)
    if name == "lean-audit.json":
        return "lock"
    if path in PATH_FILES or path in PY_PATH_FILES or name in PATH_NAMES or (path.startswith(".agents/skills/") and name == "SKILL.md"):
        return "path"
    return {".py": "import", ".lean": "lean"}.get(posixpath.splitext(path)[1])


def raw_diff(base: str, head: str, cwd: str | None, renames: str = "-M100%") -> list[dict]:
    """`git diff --raw -z`, parsed. At `-M100%` only byte-identical renames are paired; a moved file with an edit is a
    delete and an add, which `certify` pairs through the path map or git's similar renames."""
    fields = git("diff", "--raw", "-z", renames, "--no-abbrev", base, head, cwd=cwd).split(b"\0")
    out, i = [], 0
    while i < len(fields) and fields[i]:
        src_mode, dst_mode, src_sha, dst_sha, status = fields[i].decode()[1:].split(" ")
        n = 2 if status[0] in "RC" else 1
        paths = [p.decode("utf-8", "surrogateescape") for p in fields[i + 1:i + 1 + n]]
        out.append({"status": status[0], "score": int(status[1:] or 0), "old": paths[0], "new": paths[-1],
                    "modes": (src_mode, dst_mode), "shas": (src_sha, dst_sha)})
        i += 1 + n
    return out


class Tree:
    """One side of the range: its files, its Lake packages and its Lean modules."""

    def __init__(self, rev: str, cwd: str | None):
        self.rev, self.cwd = rev, cwd
        self.files = sorted(p.decode("utf-8", "surrogateescape")
                            for p in git("ls-tree", "-r", "-z", "--name-only", rev, cwd=cwd).split(b"\0") if p)
        self.set = set(self.files)

    def has_dir(self, d: str) -> bool:
        i = bisect.bisect_left(self.files, d + "/")
        return i < len(self.files) and self.files[i].startswith(d + "/")

    def is_package(self, d: str) -> bool:
        return any(posixpath.join(d, f) in self.set for f in LAKEFILES)

    def package(self, path: str) -> str | None:
        d = posixpath.dirname(path)
        while True:
            if self.is_package(d):
                return d
            if not d:
                return None
            d = posixpath.dirname(d)

    @functools.cache
    def source_dirs(self, pkg: str) -> list[str]:
        """The directories ``pkg``'s module names are relative to, deepest first: each `lean_lib`'s `srcDir` under the
        package's (`lakefile.toml`; Lake's default `.`), so a nested package with `srcDir = ".."` names its modules from the
        directory above it."""
        try:
            cfg = tomllib.loads(self.text(posixpath.join(pkg, "lakefile.toml")) or "")
        except tomllib.TOMLDecodeError:
            cfg = {}
        top = posixpath.normpath(posixpath.join(pkg, cfg.get("srcDir", ".")))
        dirs = {top, *(posixpath.normpath(posixpath.join(top, lib.get("srcDir", "."))) for lib in cfg.get("lean_lib", []))}
        return sorted(dirs, key=lambda d: (d != ".") * (d.count("/") + 1), reverse=True)

    def module(self, path: str) -> str | None:
        pkg = self.package(path)
        if pkg is None:
            return None
        src = next((s for s in self.source_dirs(pkg) if s == "." or path.startswith(s + "/")), pkg)
        return (path[len(src) + 1:] if src not in ("", ".") else path)[:-len(".lean")].replace("/", ".")

    def text(self, path: str) -> str | None:
        try:
            return git("cat-file", "blob", f"{self.rev}:{path}", cwd=self.cwd).decode("utf-8")
        except (subprocess.CalledProcessError, UnicodeDecodeError):
            return None


def derive_map(renames: list[tuple[str, str]], base_files: list[str], deleted: set[str], added: set[str]) -> dict[str, str]:
    """Each rename's shortest (old prefix, new prefix), sharing the rest of the path, whose subtree all moved there: each
    file in it renamed there, or deleted with its counterpart there added (a moved file with an edit, judged on its own)."""
    moved = dict(renames)
    sound: dict[tuple[str, str], bool] = {}

    def there(f: str, target: str) -> bool:
        return moved.get(f) == target or (f in deleted and target in added)

    def whole(p_old: str, p_new: str) -> bool:
        if (p_old, p_new) not in sound:
            lo = bisect.bisect_left(base_files, p_old + "/")
            hi = bisect.bisect_left(base_files, p_old + "0")         # "0" sorts right after "/"
            sound[(p_old, p_new)] = all(there(f, p_new + f[len(p_old):]) for f in base_files[lo:hi])
        return sound[(p_old, p_new)]

    out: dict[str, str] = {}
    for old, new in renames:
        o, n = old.split("/"), new.split("/")
        k = 0
        while k < min(len(o), len(n)) - 1 and o[-1 - k] == n[-1 - k]:
            k += 1
        for keep in range(k, 0, -1):
            p_old, p_new = "/".join(o[:-keep]), "/".join(n[:-keep])
            if whole(p_old, p_new):
                out[p_old] = p_new
                break
        else:
            out[old] = new
    return out


def map_path(p: str, pmap: dict[str, str]) -> str:
    for old in sorted(pmap, key=len, reverse=True):
        if p == old or p.startswith(old + "/"):
            return pmap[old] + p[len(old):]
    return p


def substitute(text: str, pmap: dict[str, str]) -> str:
    if not pmap:
        return text
    olds = sorted(pmap, key=len, reverse=True)
    pattern = re.compile(r"(?:(?<![\w.\-/])|(?<=^/))(" + "|".join(map(re.escape, olds)) + r")(?![\w.\-])", re.M)
    return pattern.sub(lambda m: pmap[m.group(1)], text)


def module_sub(text: str, mods: dict[str, str]) -> str:
    """Each whole moved module name rewritten: `A.B` but not a declaration `A.B.c` below it, which keeps its namespace."""
    if not mods:
        return text
    pattern = re.compile(r"(?<![\w.])(" + "|".join(map(re.escape, sorted(mods, key=len, reverse=True))) + r")(?![\w.])")
    return pattern.sub(lambda m: mods[m.group(1)], text)


def resolve_relative(text: str, path: str, pmap: dict[str, str]) -> str:
    """Each quoted relative path, as `<root>/` and the repository path it names from `path`'s directory, mapped by `pmap`."""
    here = posixpath.dirname(path)

    def repl(m: re.Match) -> str:
        p = posixpath.normpath(posixpath.join(here, m.group(2)))
        return m.group(0) if p.startswith("..") else f"{m.group(1)}<root>/{map_path(p, pmap)}{m.group(1)}"
    return RELATIVE.sub(repl, text)


def by_extra(a: str, b: str, extra: dict[str, str]) -> bool:
    return any((a == o or a.startswith(o + ".")) and b == n + a[len(o):] for o, n in extra.items())


class Modules:
    """Whether a renamed module justifies rewriting a Python `import A` to `import B`."""

    def __init__(self, renames: list[tuple[str, str]], extra: dict[str, str]):
        self.files = [(self._mod(o), self._mod(n)) for o, n in renames if o.endswith(".py") and n.endswith(".py")]
        self.extra = extra

    @staticmethod
    def _mod(path: str) -> list[str]:
        parts = path[:-3].split("/")
        return parts[:-1] if parts[-1] == "__init__" else parts

    def moved(self, a: str, b: str) -> bool:
        if by_extra(a, b, self.extra):
            return True
        A, B = a.split("."), b.split(".")
        for o, n in self.files:
            for r in range(min(len(o), len(n)) + 1):
                if r and o[-r] != n[-r]:
                    break
                oo, nn = o[:len(o) - r], n[:len(n) - r]
                if len(oo) >= len(A) and len(nn) >= len(B) and oo[len(oo) - len(A):] == A and nn[len(nn) - len(B):] == B:
                    return True
        return False


def import_rewrite(old: str, new: str, modules: Modules) -> bool:
    mo, mn = IMPORT.match(old), IMPORT.match(new)
    if not (mo and mn):
        return False
    go, gn = mo.groups(), mn.groups()
    i = 0 if go[0] is not None else 3
    if (gn[0] is not None) != (i == 0):
        return False
    return go[i] == gn[i] and go[i + 2] == gn[i + 2] and modules.moved(go[i + 1], gn[i + 1])


CHAIN = re.compile(r'"[\w.\-]+"(?:\s*/\s*"[\w.\-]+")+')
DOTTED = re.compile(r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)")
MODULE_ARG = re.compile(r"""python3?\s+-m\s|["']-m["']\s*,""")


def join_chains(text: str) -> str:
    """Each pathlib chain of string parts (`ROOT / "a" / "b"`) as one path string (`ROOT / "a/b"`)."""
    return CHAIN.sub(lambda m: '"' + "/".join(re.findall(r'"([^"]+)"', m.group(0))) + '"', text)


def dotted_rewrite(old: str, new: str, modules: Modules) -> bool:
    """Whether `new` is `old` with dotted names (modules, or repository paths with `.` for `/`) rewritten to where they moved."""
    a, b = DOTTED.split(old), DOTTED.split(new)
    if len(a) != len(b) or a[0::2] != b[0::2]:
        return False
    return all(x == y or modules.moved(x, y) for x, y in zip(a[1::2], b[1::2]))


def list_addition(old: str, new: str, entry) -> bool:
    """Whether `new` is the one-line list `old` with entries added, each a string that `entry` accepts."""
    xs, ys = re.findall(r'"([^"]*)"', old), re.findall(r'"([^"]*)"', new)
    it = iter(ys)
    if not all(x in it for x in xs) or len(ys) <= len(xs):            # `old`'s entries, in order, within `new`'s
        return False
    rest = list(ys)
    for x in xs:
        rest.remove(x)
    stripped = new
    for e in rest:
        if not entry(e):
            return False
        for form in (f'"{e}", ', f', "{e}"', f'"{e}",', f'"{e}"'):
            if form in stripped:
                stripped = stripped.replace(form, "", 1)
                break
    return stripped == old


def lean_import_rewrite(old: str, new: str, lmods: dict[str, str], extra: dict[str, str]) -> bool:
    mo, mn = LEAN_IMPORT.match(old), LEAN_IMPORT.match(new)
    if not (mo and mn) or mo.group(1) != mn.group(1) or mo.group(3) != mn.group(3):
        return False
    a, b = mo.group(2), mn.group(2)
    return lmods.get(a) == b or by_extra(a, b, extra)


def hunk(a: list[str], b: list[str], i1: int, i2: int, j1: int, j2: int) -> str:
    lines = [f"@@ -{i1 + 1},{i2 - i1} +{j1 + 1},{j2 - j1} @@"]
    lines += ["-" + x for x in a[i1:i2][:HUNK_LINES]] + ["+" + x for x in b[j1:j2][:HUNK_LINES]]
    return "\n".join(lines)


def first_unexplained(a: list[str], b: list[str], na: list[str], nb: list[str], pair=None, line=None,
                      added=None) -> str | None:
    """The first hunk of `na` -> `nb` (normalized, line for line `a` and `b`) whose changed lines are neither each a `pair`
    rewrite, nor each a `line`, nor (lines only added) each an `added` line, in the file's own lines; None when every change
    is explained."""
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, na, nb, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if pair and tag == "replace" and i2 - i1 == j2 - j1 and all(pair(x, y) for x, y in zip(na[i1:i2], nb[j1:j2])):
            continue
        if line and all(line(x) for x in na[i1:i2] + nb[j1:j2]):
            continue
        if added and tag == "insert" and all(added(y) for y in nb[j1:j2]):
            continue
        return hunk(a, b, i1, i2, j1, j2)
    return None


def top_level(text: str) -> dict[str, str] | None:
    """A JSON object's top-level fields as their raw text, or None when the text is not one."""
    dec, out = json.JSONDecoder(), {}
    ws = re.compile(r"\s*")
    try:
        i = ws.match(text, 0).end()
        if text[i] != "{":
            return None
        i = ws.match(text, i + 1).end()
        while text[i] != "}":
            key, i = dec.raw_decode(text, i)
            i = ws.match(text, i).end()
            if text[i] != ":":
                return None
            start = ws.match(text, i + 1).end()
            _, i = dec.raw_decode(text, start)
            out[key] = text[start:i]
            i = ws.match(text, i).end()
            if text[i] == ",":
                i = ws.match(text, i + 1).end()
            elif text[i] != "}":
                return None
        return out
    except (ValueError, IndexError):
        return None


def field_hunk(k: str, old: str | None, new: str | None) -> str:
    """The first entry of field `k` that differs, as `-`/`+` lines."""
    try:
        a, b = json.loads(old) if old else None, json.loads(new) if new else None
    except ValueError:
        a = b = None
    if isinstance(a, dict) and isinstance(b, dict):
        for e in sorted(set(a) | set(b)):
            if e not in a or e not in b or a[e] != b[e] or json.dumps(a[e]) != json.dumps(b[e]):
                return "\n".join(f"{s}{k}.{e}: {json.dumps(v[e])[:200]}" for s, v in (("-", a), ("+", b)) if e in v)
    return "\n".join(f"{s}{k}: {v[:200]}" for s, v in (("-", old), ("+", new)) if v is not None)


def lock_change(old: str, new: str, new_path: str, pmap: dict[str, str], lmods: dict[str, str], head: Tree,
                copy: bool = False) -> tuple[str, str | None] | None:
    """Why a `lean-audit.json` edit (or, `copy`, a destination package's new lock copied from `old`) is more than a move."""
    a, b = top_level(old), top_level(new)
    if a is None or b is None:
        return "not a JSON object", None
    for k in PROTECTED:
        if copy and k in b:
            return f"`{k}` in a new package's lock: a proofs package's lock lists no guarantees, the spec's keeps them", field_hunk(k, None, b[k])
        if not copy and (a.get(k) != b.get(k) or (k in a and json.loads(a[k]) != json.loads(b[k]))):
            return f"`{k}` changed: a pure move keeps the guarantees' records byte for byte", field_hunk(k, a.get(k), b.get(k))
    for k in sorted(set(a) | set(b)):
        if k in PROTECTED or k in STRUCTURAL or k == "runs":                # `runs` entries are judged across locks
            continue
        if k not in a:
            return f"`{k}` added", field_hunk(k, None, b[k])
        if k not in b:
            if copy:
                continue
            return f"`{k}` removed", field_hunk(k, a[k], None)
        if module_sub(substitute(a[k], pmap), lmods) != b[k]:
            return f"`{k}` changed beyond moved paths and module names", field_hunk(k, a[k], b[k])
    here = posixpath.dirname(new_path)
    for k in STRUCTURAL:
        if k in b:
            v = json.loads(b[k])
            if not (isinstance(v, list) and all(isinstance(x, str) for x in v)):
                return f"`{k}` is not a list of strings", field_hunk(k, a.get(k), b[k])
            if k == "proved_in":
                for p in v:
                    d = posixpath.normpath(posixpath.join(here, p))
                    if d.startswith("..") or not head.is_package(d):
                        return f"`proved_in` entry {p!r} is not a Lake package at HEAD ({d})", field_hunk(k, a.get(k), b[k])
    return None


def manifest_change(old: str, new: str, src_path: str, new_path: str, pmap: dict[str, str], head: Tree) -> tuple[str, str | None] | None:
    """Why a destination package's new `lake-manifest.json` is more than a copy of the source package's."""
    try:
        a, b = json.loads(old), json.loads(new)
        pa, pb = {p["name"]: p for p in a["packages"]}, {p["name"]: p for p in b["packages"]}
    except (ValueError, KeyError, TypeError):
        return "not a Lake manifest", None
    for k in sorted(set(a) | set(b)):
        if k not in ("name", "packages") and a.get(k) != b.get(k):
            return f"manifest field `{k}` differs from the source package's", field_hunk(k, json.dumps(a.get(k)), json.dumps(b.get(k)))
    src_dir, new_dir = posixpath.dirname(src_path), posixpath.dirname(new_path)
    for name, p in pa.items():
        q = pb.get(name)
        if q is None:
            return f"manifest drops package `{name}`", field_hunk("packages", json.dumps(p), None)
        if p.get("type") == "path" and q.get("type") == "path":
            pd = map_path(posixpath.normpath(posixpath.join(src_dir, p.get("dir", ""))), pmap)
            if pd != posixpath.normpath(posixpath.join(new_dir, q.get("dir", ""))):
                return f"manifest package `{name}`'s dir names another directory", field_hunk("packages", json.dumps(p), json.dumps(q))
            p, q = {**p, "dir": None}, {**q, "dir": None}
        if p != q:
            return f"manifest package `{name}` differs from the source package's", field_hunk("packages", json.dumps(p), json.dumps(q))
    for name in pb.keys() - pa.keys():
        q = pb[name]
        d = posixpath.normpath(posixpath.join(new_dir, str(q.get("dir", ""))))
        if q.get("type") != "path" or set(q) - MANIFEST_PATH_KEYS or d.startswith("..") or not head.is_package(d):
            return f"manifest adds package `{name}`, not a path package at HEAD", field_hunk("packages", None, json.dumps(q))
    return None


def lakefile_change(old: str, new: str, src_path: str, new_path: str, pmap: dict[str, str], lmods: dict[str, str],
                    head: Tree) -> tuple[str, str | None] | None:
    """Why a destination package's new lakefile is more than a copy of the source package's."""
    na = module_sub(substitute(resolve_relative(old, src_path, pmap), pmap), lmods).splitlines()
    nb = resolve_relative(new, new_path, {}).splitlines()

    def allowed(line: str) -> bool:
        if not LAKE_LINE.match(line):
            return False
        if REQUIRE_PATH.match(line):
            targets = re.findall(r"<root>/([^\"']*)", line)
            return bool(targets) and all(head.is_package(t) for t in targets)
        return True
    h = first_unexplained(old.splitlines(), new.splitlines(), na, nb, line=allowed)
    if h is None:
        return None
    return ("differs from the source package's lakefile beyond package and library names, roots, globs, defaultTargets, "
            "comments and requires by relative path to a Lake package at HEAD"), h


def copy_change(name: str, old: str, new: str, src_path: str, new_path: str, pmap: dict[str, str], lmods: dict[str, str],
                head: Tree) -> tuple[str, str | None] | None:
    if name == "lean-toolchain":
        return None if old == new else ("differs from the source package's lean-toolchain", f"-{old.strip()}\n+{new.strip()}")
    if name == "lean-audit.json":
        return lock_change(old, new, new_path, pmap, lmods, head, copy=True)
    if name == "lake-manifest.json":
        return manifest_change(old, new, src_path, new_path, pmap, head)
    return lakefile_change(old, new, src_path, new_path, pmap, lmods, head)


def certify(base: str, head: str, extra: dict[str, str] | None = None, cwd: str | None = None) -> dict:
    extra = extra or {}
    dotted = {o: n for o, n in extra.items() if "/" not in o}
    B, H = Tree(base, cwd), Tree(head, cwd)
    entries = raw_diff(base, head, cwd)
    exact = [(e["old"], e["new"]) for e in entries
             if e["status"] == "R" and e["score"] == 100 and e["modes"][0] == e["modes"][1]]
    dels = {e["old"]: e for e in entries if e["status"] == "D"}
    adds = {e["new"]: e for e in entries if e["status"] == "A"}
    similar = {e["old"]: e["new"] for e in raw_diff(base, head, cwd, "-M")
               if e["status"] == "R" and e["old"] in dels and e["new"] in adds} if dels and adds else {}
    pmap = {**derive_map(exact + list(similar.items()), B.files, set(dels), set(adds)), **extra}
    for old, d in list(dels.items()):
        new = map_path(old, pmap)
        if new == old or new not in adds:
            new = similar.get(old)
        if new in adds:
            a = adds.pop(new)
            entries.remove(a)
            d.update(status="R", score=0, new=new, modes=(d["modes"][0], a["modes"][1]), shas=(d["shas"][0], a["shas"][1]))
    renamed = [(e["old"], e["new"]) for e in entries if e["status"] == "R"]
    modules = Modules(renamed, dotted)
    lmods, dests = {}, {}
    for old, new in renamed:
        if old.endswith(".lean") and new.endswith(".lean"):
            mo, mn = B.module(old), H.module(new)
            if mo and mn and mo != mn:
                lmods[mo] = mn
            bp, hp = B.package(old), H.package(new)
            if bp is not None and hp is not None and not B.is_package(hp):
                dests.setdefault(hp, set()).add(bp)
    report = {"base": base, "head": head, "map": dict(sorted(pmap.items())), "lean_modules": dict(sorted(lmods.items())),
              "renames": [], "edits": [], "refusals": []}

    def refuse(e: dict, reason: str, h: str | None = None) -> None:
        report["refusals"].append({"path": e["new"], **({"old": e["old"]} if e["old"] != e["new"] else {}),
                                   "status": e["status"], "reason": reason, "hunk": h})

    def text(e: dict, i: int) -> str | None:
        try:
            return git("cat-file", "blob", e["shas"][i], cwd=cwd).decode("utf-8")
        except UnicodeDecodeError:
            return None

    def new_dir(p: str) -> bool:
        return H.has_dir(p) and not B.has_dir(p)

    def new_lake(p: str) -> bool:
        return new_dir(p) and H.is_package(p) and posixpath.join(p, "lean-audit.json") in H.set

    new_projects = set()
    for e in entries:
        if e["status"] == "A" and posixpath.basename(e["new"]) == "pyproject.toml" and new_dir(posixpath.dirname(e["new"])):
            try:
                new_projects.add(tomllib.loads(H.text(e["new"]) or "").get("project", {}).get("name"))
            except tomllib.TOMLDecodeError:
                pass

    def py_pair(path: str, x: str, y: str) -> bool:
        if not path.endswith(".py"):
            return False
        return (import_rewrite(x, y, modules) or (path in PY_PATH_FILES and dotted_rewrite(x, y, modules))
                or (path.startswith(MODULE_ARG_PREFIX) and bool(MODULE_ARG.search(x)) and bool(MODULE_ARG.search(y))
                    and dotted_rewrite(x, y, modules)))

    def registry_pair(path: str, x: str, y: str) -> bool:
        reg = REGISTRIES.get(path)
        return bool(reg) and list_addition(x, y, new_dir if reg == "workspace" else new_lake)

    def registered(path: str, y: str) -> bool:
        """Whether an added line registers exactly a package the range created: a workspace member or its uv source, or a
        Lake package with its lock."""
        reg = REGISTRIES.get(path)
        if reg == "lean":
            m = re.match(r'^\s*(?:\w+\s*/\s*)?"([^"]+)"\s*,?\s*(#.*)?$', y)
            return bool(m) and new_lake(m.group(1))
        if reg == "workspace":
            if m := re.match(r'^\s*"([^"]+)"\s*,?\s*$', y):
                return new_dir(m.group(1))
            if m := re.match(r'^\s*([\w.\-]+)\s*=\s*\{\s*workspace\s*=\s*true\s*\}\s*$', y):
                return m.group(1) in new_projects
            if m := re.match(r'^\s*[\w.\-]+\s*=\s*\{\s*path\s*=\s*"([^"]+)"\s*\}\s*$', y):
                return new_dir(m.group(1))
        return False

    for e in entries:
        st, old, new = e["status"], e["old"], e["new"]
        name = posixpath.basename(new)
        if st == "A" and name in LAKE_COPIES:
            d = posixpath.dirname(new)
            srcs = dests.get(d, set())
            if len(srcs) != 1:
                refuse(e, "added: not at the root of a new Lake package that the range moved .lean files into from one "
                          f"source package (sources: {sorted(srcs) or 'none'})")
                continue
            src_path = posixpath.join(next(iter(srcs)), name)
            a, b = B.text(src_path), text(e, 1)
            if a is None or b is None:
                refuse(e, f"added: the source package has no {name} to copy ({src_path})" if a is None else "not UTF-8 text")
                continue
            why = copy_change(name, a, b, src_path, new, pmap, lmods, H)
            if why:
                refuse(e, f"not a copy of {src_path}: {why[0]}", why[1])
            else:
                report["edits"].append({"path": new, "copy_of": src_path, "kind": "copy"})
            continue
        if st not in "RM":
            refuse(e, {"A": "added", "D": "deleted", "C": "copied", "T": "type changed"}.get(st, f"status {st}"))
            continue
        if e["modes"][0] != e["modes"][1]:
            refuse(e, f"mode {e['modes'][0]} -> {e['modes'][1]}")
            continue
        k = kind(new)
        if (k is None or kind(old) != k) and st == "R" and e["score"] == 100:
            report["renames"].append({"old": old, "new": new})
            continue
        a, b = text(e, 0), text(e, 1)
        if a is None or b is None:
            refuse(e, "edited, not UTF-8 text")
            continue
        al, bl = a.splitlines(), b.splitlines()
        why: tuple[str, str | None] | None = None
        if k is None or kind(old) != k:
            why = "edited, not an allowlisted path-only file", first_unexplained(al, bl, al, bl)
        elif k == "lock":
            why = None if a == b else lock_change(a, b, new, pmap, lmods, H)
        elif k == "lean":
            h = first_unexplained(al, bl, al, bl, pair=lambda x, y: lean_import_rewrite(x, y, lmods, dotted))
            why = h and ("edit beyond import lines that name a module renamed in the range", h)
        else:
            na, nb = al, bl
            if k == "path":
                rel, chains = name in RELATIVE_NAMES, new in PY_PATH_FILES
                na_text = resolve_relative(a, old, pmap) if rel else join_chains(a) if chains else a
                na = (na_text if a == b else substitute(na_text, pmap)).splitlines()
                nb = (resolve_relative(b, new, {}) if rel else join_chains(b) if chains else b).splitlines()
            h = first_unexplained(al, bl, na, nb, pair=lambda x, y: py_pair(new, x, y) or registry_pair(new, x, y),
                                  added=lambda y: registered(new, y))
            why = h and ("relative path now resolves elsewhere" if a == b
                         else f"edit beyond {'path strings' if k == 'path' else 'import lines'}", h)
        if why:
            refuse(e, *why)
        elif a == b:
            report["renames"].append({"old": old, "new": new})
        else:
            report["edits"].append({"path": new, **({"old": old} if old != new else {}), "kind": k})
    runs_moved(entries, B, H, pmap, lmods, report["refusals"])
    report["certified"] = not report["refusals"]
    return report


def lock_runs(tree: Tree, path: str) -> dict:
    try:
        runs = json.loads(tree.text(path) or "{}").get("runs", {})
    except (ValueError, AttributeError):
        return {}
    return runs if isinstance(runs, dict) else {}


def move_run(entry, old_dir: str, new_dir: str, pmap: dict[str, str], base: Tree):
    """A `runs` entry as it reads from `new_dir` once its package-relative paths (`scripts/x.py`, `{difftest/a.jsonl}`, `data`'s
    keys) follow the move map; a path neither moved nor in the tree (a file the run generates) stays as written."""
    def path(s: str) -> str:
        braced = s.startswith("{") and s.endswith("}")
        inner = s[1:-1] if braced else s
        if "/" not in inner:
            return s
        p = posixpath.normpath(posixpath.join(old_dir, inner))
        m = map_path(p, pmap)
        if m == p and not (old_dir != new_dir and (p in base.set or base.has_dir(p))):
            return s
        r = posixpath.relpath(m, new_dir or ".")
        return "{" + r + "}" if braced else r
    if isinstance(entry, str):
        return path(entry)
    if isinstance(entry, list):
        return [move_run(x, old_dir, new_dir, pmap, base) for x in entry]
    if isinstance(entry, dict):
        return {path(k): move_run(v, old_dir, new_dir, pmap, base) for k, v in entry.items()}
    return entry


def runs_moved(entries: list[dict], B: Tree, H: Tree, pmap: dict[str, str], lmods: dict[str, str], refusals: list) -> None:
    """Each `runs` entry of a lock the range touches must be in exactly one touched lock at HEAD, under its runner module's
    new name, with nothing changed but the paths the move rewrites: a run may move to the proofs package's lock, never vanish,
    be duplicated or change its data."""
    olds = [e["old"] for e in entries if e["status"] in "RMD" and posixpath.basename(e["old"]) == "lean-audit.json"]
    news = [e["new"] for e in entries if e["status"] in "RMA" and posixpath.basename(e["new"]) == "lean-audit.json"]
    head = {p: lock_runs(H, p) for p in news}
    claimed = set()

    def refuse(path: str, reason: str, h: str | None = None) -> None:
        refusals.append({"path": path, "status": "M", "reason": reason, "hunk": h})
    for lock in olds:
        for name, entry in lock_runs(B, lock).items():
            new_name = lmods.get(name, name)
            hits = [p for p, runs in head.items() if new_name in runs]
            if not hits:
                refuse(lock, f"`runs` entry {name} is in no lock at HEAD: a move keeps every run", field_hunk("runs", json.dumps({name: entry}), None))
            elif len(hits) > 1:
                refuse(hits[0], f"`runs` entry {name} is in {len(hits)} locks at HEAD ({', '.join(hits)}): a move puts it in one")
            else:
                claimed.add((hits[0], new_name))
                want = move_run(entry, posixpath.dirname(lock), posixpath.dirname(hits[0]), pmap, B)
                got = head[hits[0]][new_name]
                if want != got:
                    refuse(hits[0], f"`runs` entry {name} changed beyond moved paths and its module's name",
                           field_hunk("runs", json.dumps({new_name: want}), json.dumps({new_name: got})))
    for p, runs in head.items():
        for name in runs:
            if (p, name) not in claimed:
                refuse(p, f"`runs` entry {name} added: a move only moves runs from a lock it touches")


def summary(r: dict) -> str:
    span = f"{r['base']}..{r['head']}"
    counts = f"{len(r['renames'])} renames, {len(r['edits'])} path-only edits"
    if r["certified"]:
        return f"move-check: certified {span} is a pure move ({counts})"
    return f"move-check: refused {span}: {len(r['refusals'])} paths are not part of a pure move ({counts})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="move-check", description=__doc__.split("\n\n")[0])
    ap.add_argument("base")
    ap.add_argument("head")
    ap.add_argument("--map", action="append", default=[], metavar="OLD=NEW", help="an extra path (or dotted module) mapping")
    ap.add_argument("--json", action="store_true", help="print the report as JSON; the summary goes to stderr")
    args = ap.parse_args(argv)
    extra = {}
    for m in args.map:
        old, sep, new = m.partition("=")
        if not (sep and old and new):
            ap.error(f"--map {m!r}: expected OLD=NEW")
        extra[old.strip("/")] = new.strip("/")
    try:
        r = certify(args.base, args.head, extra)
    except subprocess.CalledProcessError as e:
        print(f"move-check: git {' '.join(e.cmd[3:])}: {e.stderr.decode(errors='replace').strip()}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(r, indent=2))
        print(summary(r), file=sys.stderr)
    else:
        for f in r["refusals"]:
            print(f"{f['path']}: {f['reason']}" + (f" (from {f['old']})" if "old" in f else ""))
            if f["hunk"]:
                print("    " + f["hunk"].replace("\n", "\n    "))
        print(summary(r))
    return 0 if r["certified"] else 1


if __name__ == "__main__":
    sys.exit(main())
