"""`move-check`: certify that BASE..HEAD is a pure move, so the lander may schedule it first in a quiet window, alone in its
train. It relaxes no gate: a certified move still goes through `check`.

    uv run python tools/verity/check/move_check.py BASE HEAD [--map OLD=NEW ...] [--json]

Every path git sees changed must be one of:
- a rename with byte-identical content and mode (git's 100% rename);
- an edit, in place or with a rename, to a text file that leaves it unchanged once each moved path in it (from the root, also
  after a variable such as `$REPO/`) is rewritten to where it is now or, in a line, left as it was (a top-level directory's
  name is also a word: `examples` in prose, a JSON key, a function's name), in `RELATIVE_NAMES` a quoted relative path (`".."`, `"../level3"`) resolved against
  the file's own directory first, in a `*.py` file a pathlib chain `ROOT / "a" / "b"` read as `"a/b"`. Each changed
  line left may only rewrite a moved path as written below a directory its old and new places share
  (`verity_vllm/program/predict/`), or a dotted name (`verity.ml.flock.tool:FLOCK_PURE`, `verity_vllm.protocols.remote_bandwidth`'s) from where a
  renamed module was to where it is now; in any of them, the modules a run of import lines binds each name to
  (`from a import m as M` to `from b import m2 as M`, a line split in two, a `from a import (x,` whose names go on
  past the change; in a heredoc, or in a string literal `"from a import m\n"` whose quotes the move keeps); in a
  `*.py` file, also the module of an `import M` /
  `from M import` line (a relative one, `.m`, read from the file's own place), or a `Path(__file__).parents[N]` anchor
  (or `Path(m.__file__)`'s, and the path joined to it) recounted to name, from where it now is, what the old one named,
  moved; in a shell script, likewise a path from its own directory (`$(dirname "$0")/../..`, or after
  `${BASH_SOURCE[0]}` or a variable it sets to its own path, `SH_SELF`);
- an added line in `REGISTRIES` that registers exactly a package the range created: a root `pyproject.toml` workspace
  member (or its `tool.uv.sources` entry) for a directory that is new at HEAD, a `tests/test_lean_packages.py` entry for a
  new Lake package with its `lean-audit.json`; and in any `pyproject.toml`, an added line of a new Lake package's own
  files (its lakefile, manifest, lock or toolchain), as a suite's inputs list what Lake reads;
- an edit to a `*.lean` file whose changed lines only rewrite an `import M` to the new module of a `.lean` file renamed in
  the range (a module is its path below its Lake package, `/` as `.`, without `.lean`), as when a Lean split gives the
  proof modules a root of their own while their declarations keep their namespace;
- an edit to a `lean-audit.json` that keeps `pins` (the guarantees' records; `guarantees` too, if a lock has it) and
  `reads` byte for byte (a `reads` key, a module's name, may follow the module's move), may change the structural `roots`,
  `proved_in` (each entry a Lake package at HEAD) and `dependencies` (the fetched modules the package imports, which a
  split shares out and the Lean check holds to the build), and changes any other field only by moved paths and module
  names; a `runs` entry may move to another lock the range touches (a proofs package's new one), its runner module
  renamed and its package-relative paths following the move, and must end up in exactly one, otherwise unchanged
  (`runs_moved`);
- a new `lakefile.toml`, `lakefile.lean`, `lake-manifest.json`, `lean-audit.json` or `lean-toolchain` at the root of a new
  Lake package that the range moved `.lean` files into from one source package, which is a copy of the source package's
  file (`copy_change`): a lakefile may differ only in package and library names, `roots`, `globs`, `defaultTargets`,
  comments and `require`s by relative path to a Lake package at HEAD; a manifest only in its name, the moved paths of
  its path packages, and new path packages; a lock lists no guarantees (an empty record at most) and keeps each field it
  keeps; a toolchain is identical;
- in place, a `lakefile.toml` whose library `roots` lose the modules the range moved out of its package and gain those it
  moved in, that adds a `require` of a new Lake package, or adds comment lines (`lakefile_in_place`), and a
  `lake-manifest.json` that adds that package (`manifest_change`);
- a `TRUSTED.sha256` line rewritten from a file's digest at BASE to its digest at HEAD, under its path moved by the map;
- an entry appended to research's `moves.json` whose paths agree with the range's (`moves_change`);
- any of these after the move scripts' word maps (`tools/move/*.toml` with `[words]`, read at BASE) are applied, as
  `tools/move/rename.py` applies them, with only the words that rename what the range moved: a path the map moves to
  where the range moved it, or a Lean module likewise (`word_maps`). A text file the words alone rewrite into its new
  text is an edit of that kind, whatever the file, and so is one that also drops each list entry the words made a
  repeat of the one before it (`merged`).
Anything else is refused, with what differed and its first offending hunk. Exit 0 certified, 1 refused, 2 on a usage or
git error.

The path map comes from the renames. Each rename contributes the shortest prefix pair (old directory, new directory) whose
whole subtree in BASE moved there file by file. When only `verity/core/protocols/pous` moved, `protocols -> security_proofs` is never
inferred. Renames are git's exact ones, and its similar ones (`-M`) as candidates, each edit then judged as above; a file
below a directory that moved whole is paired with its place there, whichever identical file git paired it with. A deleted
path is paired with the added one a word map at BASE moves it to, and an in-place edit of a path a word map moves to a path
new at HEAD is that move and another file landing on its path (`landings`), each pair again judged as above. The entry
the range appends to `moves.json` adds each of its paths that the range bears out on its own (`declared`): a directory
that moved whole but for files a longer path of the entry sends elsewhere, which no rename pairs, and a file whose edit
leaves it below git's similarity, each pair again judged as above.
`--map OLD=NEW` adds an entry, and a dotted one (no `/`) also maps a Python or Lean module and those below it.
"""
from __future__ import annotations

import argparse
import bisect
import difflib
import functools
import hashlib
import importlib.util
import json
import posixpath
import re
import subprocess
import sys
import tomllib
from pathlib import Path

#: research's record of the moves, to which a move appends its own (`moves_change`)
MOVES = "tools/research/src/research/store/moves.json"
#: the files where a new package is registered by an added line (`registration`)
REGISTRIES = {"pyproject.toml": "workspace", "tests/test_lean_packages.py": "lean"}
#: files whose quoted relative paths name another directory of the tree (a Lake `require`, a manifest's `dir`, a uv source)
RELATIVE_NAMES = {"pyproject.toml", "lakefile.lean", "lakefile.toml", "lake-manifest.json"}
LAKEFILES = ("lakefile.toml", "lakefile.lean")
LAKE_COPIES = {*LAKEFILES, "lake-manifest.json", "lean-audit.json", "lean-toolchain"}
#: a lock's records of what its guarantees say, and its structural fields, which a split changes
PROTECTED, STRUCTURAL = ("pins", "guarantees", "reads"), ("roots", "proved_in", "dependencies")
#: the move scripts' word maps, each a `*.toml` directly in it with `[words]` (`tools/move/rename.py`'s format)
MAPS = "tools/move/"
#: the files that pin other files' digests (`<sha256>  <relative path>` lines)
PINS = "TRUSTED.sha256"
PIN_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")
ROOT = Path(__file__).resolve().parents[3]
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


def kind(path: str) -> str:
    name = posixpath.basename(path)
    if name == "lean-audit.json":
        return "lock"
    if path == MOVES:
        return "moves"
    if name in RELATIVE_NAMES:
        return "path"
    return {".py": "python", ".lean": "lean"}.get(posixpath.splitext(path)[1], "text")


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

    def blob(self, path: str) -> bytes | None:
        if path not in self.set:
            return None
        try:
            return git("cat-file", "blob", f"{self.rev}:{path}", cwd=self.cwd)
        except subprocess.CalledProcessError:
            return None

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
    files: dict[str, str] = {}
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
            files[old] = new
    # git pairs identical files (empty `__init__.py`s) across directories at random: a directory that moved whole keeps
    # its files, whatever git paired them with
    for old, new in files.items():
        to = map_path(old, out)
        if to == old or not there(old, to):
            out[old] = new
    return out


def map_path(p: str, pmap: dict[str, str]) -> str:
    for old in sorted(pmap, key=len, reverse=True):
        if p == old or p.startswith(old + "/"):
            return pmap[old] + p[len(old):]
    return p


@functools.lru_cache(maxsize=8)
def _paths_pattern(olds: tuple[str, ...]) -> re.Pattern:
    """A moved path from the root: alone, at a line's `/`, or after a variable holding the checkout (`$REPO/`, `${ROOT}/`),
    as `tools/move/layout.py` rewrites it; never the tail of a longer path."""
    return re.compile(r"(?:(?<![\w.\-/])|(?<=^/)|(?<![\w.\-/$])(\$\{?\w+\}?/))(" + "|".join(map(re.escape, olds))
                      + r")(?![\w.\-])", re.M)


def substitute(text: str, pmap: dict[str, str]) -> str:
    if not pmap:
        return text
    return _paths_pattern(tuple(sorted(pmap, key=len, reverse=True))).sub(lambda m: (m.group(1) or "") + pmap[m.group(2)],
                                                                          text)


def partly(old: str, new: str, pmap: dict[str, str]) -> bool:
    """Whether the line `new` is `old` with each moved path in it either rewritten to where it is now or left as it was: a
    top-level directory's name is also a word (`examples` in prose, a JSON key, a function's name) that the move leaves."""
    if not pmap:
        return False
    parts, at = [], 0
    for m in _paths_pattern(tuple(sorted(pmap, key=len, reverse=True))).finditer(old):
        parts += [re.escape(old[at:m.start()]), f"(?:{re.escape(m[0])}|{re.escape((m[1] or '') + pmap[m[2]])})"]
        at = m.end()
    return bool(parts) and re.fullmatch("".join(parts) + re.escape(old[at:]), new) is not None


def short_forms(pmap: dict[str, str], base: Tree) -> dict[str, str]:
    """Each moved path as written below a directory its old and new paths share (`verity_vllm/program/predict` for
    `integrations/vllm/verity_vllm/program/predict`), keeping at least one shared part; dropped where two moves disagree or where it
    names a path of BASE's tree from the root."""
    out: dict[str, str] = {}
    bad: set[str] = set()
    for old, new in pmap.items():
        o, n = old.split("/"), new.split("/")
        c = 0
        while c < min(len(o), len(n)) - 1 and o[c] == n[c]:
            c += 1
        for k in range(1, c):
            so, sn = "/".join(o[k:]), "/".join(n[k:])
            if so in pmap or so in base.set or base.has_dir(so) or out.setdefault(so, sn) != sn:
                bad.add(so)
    return {k: v for k, v in out.items() if k not in bad}


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


def absolute(module: str, path: str) -> str:
    """A relative import's module as the dotted path of the file it names (`.b` in `a/x.py` is `a.b`)."""
    dots = len(module) - len(module.lstrip("."))
    if not dots:
        return module
    parts = [p for p in posixpath.dirname(path).split("/") if p]
    if dots - 1 > len(parts):
        return module
    return ".".join(parts[:len(parts) - (dots - 1)] + ([module[dots:]] if module[dots:] else []))


def import_rewrite(old: str, new: str, modules: Modules, old_path: str = "", new_path: str = "") -> bool:
    mo, mn = IMPORT.match(old), IMPORT.match(new)
    if not (mo and mn):
        return False
    go, gn = mo.groups(), mn.groups()
    i = 0 if go[0] is not None else 3
    if (gn[0] is not None) != (i == 0):
        return False
    return go[i] == gn[i] and go[i + 2] == gn[i + 2] and modules.moved(absolute(go[i + 1], old_path),
                                                                      absolute(gn[i + 1], new_path))


#: an import line, also one inside a string literal (`"from a import b\n"`), whose wrapper the move leaves as it was
_OPEN, _CLOSE = r"""(\s*(?:[rbfu]*["'])?)""", r"""(#.*|(?:\\n)?["'][\s,)]*)?"""
FROM = re.compile(rf"^{_OPEN}from\s+([\w.]+)\s+import\s+([\w\s,]+?)\s*{_CLOSE}$")
IMPORT_AS = re.compile(rf"^{_OPEN}import\s+([\w.]+)(?:\s+as\s+(\w+))?\s*{_CLOSE}$")
#: `from a import (x, y,`, its names continued on lines the move may leave as they were, or `from a import (x, y)`
FROM_PAREN = re.compile(r"^(\s*)from\s+([\w.]+)\s+import\s+\(([\w\s,]*?),?\s*\)?\s*(#.*)?$")


def bindings(lines: list[str], path: str) -> list[tuple[str, str, str, str]] | None:
    """Each name the import lines bind, as (indent, comment, the dotted name it binds, the name), or None if a line isn't
    one such import. A parenthesized `from a import (` also binds `a` itself, under the name `(`: its continuation lines
    may be unchanged context, so `a`'s rewrite must be checked even when the line names nothing."""
    out = []
    for line in lines:
        if (m := FROM.match(line)) or (m := FROM_PAREN.match(line)):
            if m.re is FROM_PAREN:
                out.append((m.group(1), m.group(4) or "", absolute(m.group(2), path), "("))
            for part in filter(str.strip, m.group(3).split(",")) if m.re is FROM_PAREN else m.group(3).split(","):
                name, _, alias = (w.strip() for w in part.partition(" as "))
                if not name.isidentifier() or (alias and not alias.isidentifier()):
                    return None
                out.append((m.group(1), m.group(4) or "", f"{absolute(m.group(2), path)}.{name}", alias or name))
        elif m := IMPORT_AS.match(line):
            out.append((m.group(1), m.group(4) or "", m.group(2), m.group(3) or m.group(2)))
        else:
            return None
    return out


def import_block(old: list[str], new: list[str], modules: Modules, old_path: str, new_path: str) -> bool:
    """Whether the import lines `new` bind the names `old` did, each to where its module moved (`from a import m as M`
    to `from b import m2 as M`, one line split in two)."""
    a, b = bindings(old, old_path), bindings(new, new_path)
    if a is None or b is None or len(a) != len(b):
        return False
    rest = list(b)
    for indent, comment, x, name in a:
        y = next((t for t in rest if t[:2] == (indent, comment) and (t[3] == name or name == x)
                  and (t[2] == x or moved_name(x, t[2], modules))), None)
        if y is None:
            return False
        rest.remove(y)
    return True


#: `Path(__file__).parents[N]` or `.parent.parent` (or a module's, `Path(m.__file__)`), the directory a file names its
#: paths from, and the path joined to it (`ROOT / "a/b"`, a chain joined by `join_chains`)
ANCHOR = re.compile(r"(?P<head>Path\((?:__file__|(?P<mod>[A-Za-z_]\w*)\.__file__)\)(?:\.resolve\(\)|\.absolute\(\))?)"
                    r"(?:\.parents\[(?P<n>\d+)\]|(?P<p>(?:\.parent\b)+))(?:\s*/\s*\"(?P<rest>[^\"]+)\")?")


def anchored(m: re.Match, path: str | None) -> str | None:
    if path is None:
        return None
    up = int(m.group("n")) if m.group("n") is not None else m.group("p").count(".parent") - 1
    dirs = [p for p in posixpath.dirname(path).split("/") if p]
    if up > len(dirs):
        return None
    d = "/".join(dirs[:len(dirs) - up])
    return posixpath.normpath(posixpath.join(d, m.group("rest"))) if m.group("rest") else d


def module_file(tree: Tree, text: str, alias: str) -> str | None:
    """The file of the module `text` imports as `alias`, when exactly one file of `tree` is that module."""
    m = re.search(rf"^\s*from\s+([\w.]+)\s+import\s+(\w+)\s+as\s+{alias}\b|^\s*import\s+([\w.]+)\s+as\s+{alias}\b", text, re.M)
    if m is None:
        return None
    tail = (f"{m.group(1)}.{m.group(2)}" if m.group(1) else m.group(3)).replace(".", "/")
    hits = [f for f in tree.files if ("/" + f).endswith(("/" + tail + ".py", "/" + tail + "/__init__.py"))]
    return hits[0] if len(hits) == 1 else None


def anchor_rewrite(old: str, new: str, at_old, at_new, pmap: dict[str, str]) -> bool:
    """Whether `new` is `old` with each anchor recounted so that, from where it now is (`at_new(module alias or None)`),
    it names the path the old one named, moved by `pmap`."""
    a, b = list(ANCHOR.finditer(old)), list(ANCHOR.finditer(new))
    if not a or len(a) != len(b) or ANCHOR.sub("\0", old) != ANCHOR.sub("\0", new):
        return False
    for x, y in zip(a, b):
        dx, dy = anchored(x, at_old(x.group("mod"))), anchored(y, at_new(y.group("mod")))
        if x.group("head") != y.group("head") or dx is None or dy is None or map_path(dx, pmap) != dy:
            return False
    return True


#: a shell script's own directory and the path joined to it (`$(dirname "$0")/../..`), as `tools/move/layout.py` writes it
SH_ANCHOR = re.compile(r'(?P<head>\$\(dirname\s+"(?P<ref>\$(?:\{[A-Za-z_0-9]\w*(?:\[0\])?\}|[A-Za-z_0-9]\w*))"\))'
                       r'(?P<rel>(?:/[\w.\-]+)+)')
#: a variable a shell script sets to its own path (`SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")`)
SH_SELF = re.compile(r'^\s*(?:export\s+|readonly\s+|local\s+)?(?P<v>[A-Za-z_]\w*)=(?P<q>"?)'
                     r'(?:\$\(cd\s+"\$\(dirname\s+"(?P<a>\$0|\$\{BASH_SOURCE\[0\]\})"\)"\s+&&\s+pwd\)/\$\(basename\s+"(?P=a)"\)'
                     r'|\$\((?:realpath|readlink\s+-f)\s+"(?:\$0|\$\{BASH_SOURCE\[0\]\})"\))(?P=q)\s*$', re.M)
SH_OWN = {"0", "BASH_SOURCE", "BASH_SOURCE[0]"}


def is_shell(path: str, text: str) -> bool:
    return path.endswith(".sh") or re.match(r"#!\S*(?:/|\s)(?:ba|da|z)?sh\b", text) is not None


def sh_self_names(text: str) -> set[str]:
    """The names whose `dirname` is a shell script's own directory: its own path's, and each variable it sets to that."""
    return SH_OWN | {m.group("v") for m in SH_SELF.finditer(text)}


def sh_anchor_rewrite(old: str, new: str, old_path: str, new_path: str, names_old: set[str], names_new: set[str],
                      pmap: dict[str, str]) -> bool:
    """Whether `new` is `old` with each of a shell script's paths from its own directory recounted so that, from the script's
    new place, it names the path the old one named, moved by `pmap`."""
    a, b = list(SH_ANCHOR.finditer(old)), list(SH_ANCHOR.finditer(new))
    if not a or len(a) != len(b) or SH_ANCHOR.sub("\0", old) != SH_ANCHOR.sub("\0", new):
        return False
    for x, y in zip(a, b):
        ref = x.group("ref")[1:].strip("{}")
        if x.group("ref") != y.group("ref") or ref not in names_old or ref not in names_new:
            return False
        dx = posixpath.normpath(posixpath.join(posixpath.dirname(old_path), x.group("rel")[1:]))
        dy = posixpath.normpath(posixpath.join(posixpath.dirname(new_path), y.group("rel")[1:]))
        if dx.startswith("..") or dy.startswith("..") or map_path(dx, pmap) != dy:
            return False
    return True


CHAIN = re.compile(r'"[\w.\-]+"(?:\s*/\s*"[\w.\-]+")+')
DOTTED = re.compile(r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)")


def join_chains(text: str) -> str:
    """Each pathlib chain of string parts (`ROOT / "a" / "b"`) as one path string (`ROOT / "a/b"`)."""
    return CHAIN.sub(lambda m: '"' + "/".join(re.findall(r'"([^"]+)"', m.group(0))) + '"', text)


def dotted_rewrite(old: str, new: str, modules: Modules) -> bool:
    """Whether `new` is `old` with dotted names (modules, or repository paths with `.` for `/`) rewritten to where they moved."""
    a, b = DOTTED.split(old), DOTTED.split(new)
    if len(a) != len(b) or a[0::2] != b[0::2]:
        return False
    return all(x == y or moved_name(x, y, modules) for x, y in zip(a[1::2], b[1::2]))


def moved_name(x: str, y: str, modules: Modules) -> bool:
    """Whether `y` is `x` with its module moved, the names below it (`remote.Session`'s `Session`) kept."""
    a, b = x.split("."), y.split(".")
    for j in range(min(len(a), len(b))):
        if j and a[-j] != b[-j]:
            return False
        if modules.moved(".".join(a[:len(a) - j]), ".".join(b[:len(b) - j])):
            return True
    return False


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
                      added=None, block=None, left=None) -> str | None:
    """The first hunk of `na` -> `nb` (normalized, line for line `a` and `b`) whose changed lines are neither each a `pair`
    rewrite or (`left(i, y)`) line i with moved paths left as they were, nor each a `line`, nor (lines only added) each an
    `added` line, nor together a `block` rewrite, in the file's own lines; None when every change is explained."""
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, na, nb, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if pair and tag == "replace" and i2 - i1 == j2 - j1 and all(
                pair(x, y) or (left and left(i, y)) for i, x, y in zip(range(i1, i2), na[i1:i2], nb[j1:j2])):
            continue
        if block and tag == "replace" and block(na[i1:i2], nb[j1:j2]):
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


def same_value(old: str, new: str) -> bool:
    """Whether two JSON texts hold the same value, keys in the same order, however each is wrapped: the lock's writer
    wraps a list that a renamed module lengthened past its line."""
    try:
        return json.dumps(json.loads(old)) == json.dumps(json.loads(new))
    except ValueError:
        return old == new


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
        if copy and k in b and json.loads(b[k]):
            return f"`{k}` in a new package's lock: a proofs package's lock lists no guarantees, the spec's keeps them", field_hunk(k, None, b[k])
        if copy or (a.get(k) == b.get(k)) or (k == "reads" and k in a and k in b and reads_moved(a[k], b[k], lmods)):
            continue
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
        if not same_value(module_sub(substitute(a[k], pmap), lmods), b[k]):
            return f"`{k}` changed beyond moved paths and module names", field_hunk(k, a[k], b[k])
    here = posixpath.dirname(new_path)
    for k in ("roots", "proved_in"):
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


def reads_moved(old: str, new: str, lmods: dict[str, str]) -> bool:
    """Whether `reads` (keyed by the module a definition is in) is ``old``'s with each key renamed as its module moved and
    each record byte for byte the same."""
    try:
        a, b = json.loads(old), json.loads(new)
    except ValueError:
        return False
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return False
    moved = {lmods.get(m, m): v for m, v in a.items()}
    return len(moved) == len(a) and moved.keys() == b.keys() and all(json.dumps(v) == json.dumps(b[m]) for m, v in moved.items())


def manifest_change(old: str, new: str, src_path: str, new_path: str, pmap: dict[str, str], head: Tree,
                    added=None) -> tuple[str, str | None] | None:
    """Why a destination package's new `lake-manifest.json` is more than a copy of the source package's, or (in place,
    ``added`` the packages it may add) why a manifest's edit is more than adding them."""
    added = added or head.is_package
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
        if q.get("type") != "path" or set(q) - MANIFEST_PATH_KEYS or d.startswith("..") or not added(d):
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


def lakefile_in_place(old: str, new: str, path: str, moves: dict[str, tuple[str, str]], new_lake) -> tuple[str, str | None] | None:
    """Why an in-place `lakefile.toml` edit is more than moving `roots` with their modules (``moves``: a module, by its old
    name and its new, to its package at BASE and at HEAD): dropping those moved out of this package and adding those moved
    into it, requiring a new Lake package (``new_lake``) by its relative path, and adding comment lines."""
    try:
        a, b = tomllib.loads(old), tomllib.loads(new)
    except tomllib.TOMLDecodeError:
        return "not TOML", None
    here = posixpath.dirname(path)
    out = {m for m, (bp, hp) in moves.items() if bp == here != hp}
    into = {m for m, (bp, hp) in moves.items() if hp == here != bp}
    ra, rb = a.pop("require", []), b.pop("require", [])
    it = iter(rb)
    if not all(r in it for r in ra):
        return "changes or drops a `require`", field_hunk("require", json.dumps(ra), json.dumps(rb))
    for r in rb:
        if r not in ra and (set(r) != {"name", "path"}
                            or not new_lake(posixpath.normpath(posixpath.join(here, str(r["path"]))))):
            return "adds a `require` that is not of a new Lake package by its relative path", field_hunk("require", None, json.dumps(r))
    la, lb = a.pop("lean_lib", []), b.pop("lean_lib", [])
    if len(la) != len(lb):
        return "adds or drops a `lean_lib`", None
    for x, y in zip(la, lb):
        rx, ry = x.pop("roots", []), y.pop("roots", [])
        if x != y or [r for r in rx if r not in out] != [r for r in ry if r not in into]:
            return (f"`lean_lib` {x.get('name')} changes beyond the roots of modules moved out of this package or into it",
                    field_hunk("roots", json.dumps(rx), json.dumps(ry)))
    if a != b:
        return "changes a field other than `require`, `roots` and comments", None
    ca, cb = ([ln for ln in t.splitlines() if ln.lstrip().startswith("#")] for t in (old, new))
    if h := first_unexplained(ca, cb, ca, cb, added=lambda y: True):
        return "changes a comment beyond adding lines", h
    return None


def pins_change(a: list[str], b: list[str], old: str, new: str, base: Tree, head: Tree, pmap: dict[str, str]) -> str | None:
    """The first hunk of a `TRUSTED.sha256` edit whose lines are not each a file's digest at BASE rewritten to its digest at
    HEAD, under its path moved by the map, in any order (the command that writes the file sorts the paths, and a moved
    path sorts anew); None when there is none."""
    digests: dict[tuple[int, str], str | None] = {}

    def digest(tree: Tree, path: str) -> str | None:
        if (id(tree), path) not in digests:
            data = tree.blob(path)
            digests[id(tree), path] = None if data is None else hashlib.sha256(data).hexdigest()
        return digests[id(tree), path]

    def pair(x: str, y: str) -> bool:
        mx, my = PIN_LINE.match(x), PIN_LINE.match(y)
        if not (mx and my):
            return False
        po = posixpath.normpath(posixpath.join(posixpath.dirname(old), mx[2]))
        pn = posixpath.normpath(posixpath.join(posixpath.dirname(new), my[2]))
        return map_path(po, pmap) == pn and digest(base, po) == mx[1] and digest(head, pn) == my[1]
    hunk = first_unexplained(a, b, a, b, pair=pair)
    if hunk is None or len(a) != len(b):
        return hunk
    left = list(a)
    for y in b:
        if (x := next((x for x in left if pair(x, y)), None)) is None:
            return hunk
        left.remove(x)
    return None


@functools.cache
def _rename():
    """`tools/move/rename.py`, whose `Words` applies a word map as the move did."""
    spec = importlib.util.spec_from_file_location("move_rename", ROOT / "tools" / "move" / "rename.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def spelled(word: str) -> str:
    """The path or module ``word`` spells: a pathlib chain of string parts (`"a" / "b"`) as `a/b`, a regex's escaped dots
    (`a\\.b`) as `a.b`."""
    joined = join_chains(word)
    if joined != word and joined.startswith('"') and joined.endswith('"'):
        joined = joined[1:-1]
    return joined.replace("\\.", ".")


def base_maps(base: Tree) -> list[tuple[dict, object]]:
    """The move scripts' word maps at BASE, each as its data and its whole `Words`."""
    R, out = _rename(), []
    for p in base.files:
        if not (p.startswith(MAPS) and p.endswith(".toml") and "/" not in p[len(MAPS):]):
            continue
        try:
            data = tomllib.loads(base.text(p) or "")
            if data.get("words"):
                out.append((data, R.Words(data)))
        except (tomllib.TOMLDecodeError, SystemExit, KeyError, TypeError, AttributeError):
            continue
    return out


def landings(entries: list[dict], base: Tree, head: Tree) -> list[dict]:
    """``entries`` with each in-place edit of a path that a word map at BASE moves to a path new at HEAD split into its
    delete and an add: its file moved out and another moved in (`Area.lean` to `Area/Check.lean` while `Area/Check.lean`
    goes to `Area/Check/Check.lean`). `certify` pairs both through the maps."""
    whole = [w for _, w in base_maps(base)]
    out = []
    for e in entries:
        p = e["old"]
        if e["status"] == "M" and any((n := w.path(p)) != p and n in head.set and n not in base.set for w in whole):
            out.append({**e, "status": "D", "modes": (e["modes"][0], "0" * 6), "shas": (e["shas"][0], "0" * 40)})
            out.append({**e, "status": "A", "modes": ("0" * 6, e["modes"][1]), "shas": ("0" * 40, e["shas"][1])})
        else:
            out.append(e)
    return out


def word_maps(base: Tree, renamed: list[tuple[str, str]], lmods: dict[str, str], pmap: dict[str, str]) -> list:
    """The move scripts' word maps at BASE, each cut to the words that rename what the range moved: a word is kept when it
    renames a path, a directory the range moved whole (as `dir/`) or a Lean module that its map, applied whole, renames to
    where the range moved it, or when it spells a kept word's rename as a pathlib chain or a regex (`spelled`)."""
    R = _rename()
    names = [*renamed, *lmods.items(), *((o + "/", n + "/") for o, n in pmap.items())]
    out = []
    for data, whole in base_maps(base):
        moved = {a: b for a, b in names if whole.path(a) == b and a != b}
        kept = {o: n for o, n in data["words"].items()
                if any(R.Words({"words": {o: n}}).path(a) != a for a in moved)}
        kept |= {o: n for o, n in data["words"].items() if (spelled(o), spelled(n)) in kept.items()}
        if kept:
            out.append(R.Words({**data, "words": kept, "words_in": []}))
    return out


ENTRY = re.compile(r'\s*"[^"\n]*",?\s*')


def merged(old: str, new: str) -> str:
    """``new`` (``old`` worded line for line) without each list entry the words made a repeat of the entry before it: two
    directories the range merged into one, both listed."""
    o, w = old.splitlines(keepends=True), new.splitlines(keepends=True)
    if len(o) != len(w):
        return new
    return "".join(w[i] for i in range(len(w))
                   if not (i and w[i] == w[i - 1] and o[i] != o[i - 1] and ENTRY.fullmatch(w[i])))


def worded(text: str, path: str, maps: list, lock: bool = False) -> str:
    """``text`` with each word map applied as `tools/move/rename.py` applies it: not to a file the map excludes, and in a
    lock, not to the records."""
    for w in maps:
        if not w.excluded(path):
            text = w.sub(text, path, _rename().record_spans(text) if lock else [])
    return text


def copy_change(name: str, old: str, new: str, src_path: str, new_path: str, pmap: dict[str, str], lmods: dict[str, str],
                head: Tree) -> tuple[str, str | None] | None:
    if name == "lean-toolchain":
        return None if old == new else ("differs from the source package's lean-toolchain", f"-{old.strip()}\n+{new.strip()}")
    if name == "lean-audit.json":
        return lock_change(old, new, new_path, pmap, lmods, head, copy=True)
    if name == "lake-manifest.json":
        return manifest_change(old, new, src_path, new_path, pmap, head)
    return lakefile_change(old, new, src_path, new_path, pmap, lmods, head)


def appended(old: str, new: str) -> tuple[dict[str, str] | None, str | None]:
    """The `paths` of the one entry `moves.json`'s `new` text appends to its `old` one, each old path's new path, or why
    `new` isn't `old` with one such entry appended."""
    try:
        a, b = json.loads(old), json.loads(new)
    except ValueError:
        return None, "not JSON"
    am, bm = a.get("moves"), b.get("moves")
    if not (isinstance(am, list) and isinstance(bm, list)) or len(bm) != len(am) + 1 or bm[:len(am)] != am or \
            {k: v for k, v in a.items() if k != "moves"} != {k: v for k, v in b.items() if k != "moves"}:
        return None, "changes more than one appended entry"
    entry = bm[-1]
    if not (isinstance(entry, dict) and set(entry) == {"paths"} and isinstance(entry["paths"], dict)
            and all(isinstance(v, str) for v in (*entry["paths"], *entry["paths"].values()))):
        return None, "the appended entry isn't one `paths` map"
    return {o.rstrip("/"): n.rstrip("/") for o, n in entry["paths"].items()}, None


def moves_change(old: str, new: str, pmap: dict[str, str], base: Tree) -> tuple[str, str | None] | None:
    """Why `moves.json` at HEAD isn't BASE's with one entry appended whose `paths` agree with the range: each BASE file
    below an entry's old path (and below no longer one) moved to the same place below its new path. An entry may name a
    path BASE doesn't have (a file an open branch adds), which no file contradicts."""
    declared, why = appended(old, new)
    if declared is None:
        return why, None
    for f in base.files:
        o = max((d for d in declared if f == d or f.startswith(d + "/")), key=len, default=None)
        if o is not None and map_path(f, pmap) != declared[o] + f[len(o):]:
            return f"the appended entry moves {f} to {declared[o] + f[len(o):]}, the range to {map_path(f, pmap)}", None
    return None


def declared(base: Tree, head: Tree) -> dict[str, str]:
    """The paths of the entry the range appends to `moves.json` that the range bears out without them: each names a file or
    directory of BASE, and every BASE file below it (and below no longer path of the entry) is gone at HEAD, with its place
    below the entry's new path new at HEAD. Whether its content moved unchanged is then judged like any pair's. A range
    that appends no such entry declares nothing, and `moves_change` refuses its `moves.json` edit, if any, by name."""
    decl, _ = appended(base.text(MOVES) or "{}", head.text(MOVES) or "{}")
    out: dict[str, str] = {}
    if decl is None:
        return out
    for o, n in decl.items():
        mine = [f for f in base.files if (f == o or f.startswith(o + "/"))
                and max((d for d in decl if f == d or f.startswith(d + "/")), key=len) == o]
        if mine and all(f not in head.set and (t := n + f[len(o):]) in head.set and t not in base.set for f in mine):
            out[o] = n
    return out


def certify(base: str, head: str, extra: dict[str, str] | None = None, cwd: str | None = None) -> dict:
    extra = extra or {}
    dotted = {o: n for o, n in extra.items() if "/" not in o}
    B, H = Tree(base, cwd), Tree(head, cwd)
    extra = {**declared(B, H), **extra}
    entries = landings(raw_diff(base, head, cwd), B, H)
    whole = [w for _, w in base_maps(B)]
    exact = [(e["old"], e["new"]) for e in entries
             if e["status"] == "R" and e["score"] == 100 and e["modes"][0] == e["modes"][1]]
    dels = {e["old"]: e for e in entries if e["status"] == "D"}
    adds = {e["new"]: e for e in entries if e["status"] == "A"}
    similar = {e["old"]: e["new"] for e in raw_diff(base, head, cwd, "-M")
               if e["status"] == "R" and e["old"] in dels and e["new"] in adds} if dels and adds else {}
    pmap = {**derive_map(exact + list(similar.items()), B.files, B.set - H.set, H.set - B.set), **extra}
    for e in list(entries):
        to = map_path(e["old"], pmap)
        if e["status"] == "R" and to not in (e["old"], e["new"]) and to in H.set and to not in B.set:
            entries.remove(e)
            dels[e["old"]] = {**e, "status": "D", "new": e["old"]}
            adds[e["new"]] = {**e, "status": "A", "old": e["new"]}
            entries += [dels[e["old"]], adds[e["new"]]]
    for old, d in list(dels.items()):
        new = map_path(old, pmap)
        if new == old or new not in adds:
            if (worded_to := next((n for w in whole if (n := w.path(old)) in adds), None)) is not None:
                pmap[old] = worded_to
            new = worded_to or similar.get(old)
        if new in adds:
            a = adds.pop(new)
            entries.remove(a)
            d.update(status="R", score=100 if d["shas"][0] == a["shas"][1] else 0, new=new,
                     modes=(d["modes"][0], a["modes"][1]), shas=(d["shas"][0], a["shas"][1]))
    short = short_forms(pmap, B)
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
    maps = word_maps(B, renamed, lmods, pmap)
    moves: dict[str, tuple[str, str]] = {}                  # a module that changed package: (its package at BASE, at HEAD)
    for o, n in renamed:
        if o.endswith(".lean") and n.endswith(".lean"):
            bp, hp = B.package(o), H.package(n)
            if bp is not None and hp is not None and bp != hp:
                moves.update(dict.fromkeys(filter(None, (B.module(o), H.module(n))), (bp, hp)))
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

    def registry_pair(path: str, x: str, y: str) -> bool:
        reg = REGISTRIES.get(path)
        return bool(reg) and list_addition(x, y, new_dir if reg == "workspace" else new_lake)

    def registered(path: str, y: str) -> bool:
        """Whether an added line registers exactly a package the range created: a workspace member or its uv source, or a
        Lake package with its lock; or, in any `pyproject.toml`, a line of quoted paths each a Lake file at the root of a new
        Lake package (a suite's inputs naming what Lake reads to load it)."""
        if posixpath.basename(path) == "pyproject.toml" and re.fullmatch(r'\s*("[^"]+"\s*,\s*)*"[^"]+"\s*,?\s*', y):
            files = re.findall(r'"([^"]+)"', y)
            if all(posixpath.basename(f) in LAKE_COPIES and new_lake(posixpath.dirname(f)) for f in files):
                return True
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
            a = worded(a, src_path, maps, lock=name == "lean-audit.json")
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
        if (k == "text" or kind(old) != k) and st == "R" and e["score"] == 100:
            report["renames"].append({"old": old, "new": new})
            continue
        a, b = text(e, 0), text(e, 1)
        if a is None or b is None:
            refuse(e, "edited, not UTF-8 text")
            continue
        same, pins = a == b, name == PINS and posixpath.basename(old) == PINS
        if not same and not pins and old != MOVES:
            a, unworded = worded(a, old, maps, lock=k == "lock"), a
            if b in (a, merged(unworded, a)):
                report["edits"].append({"path": new, **({"old": old} if old != new else {}), "kind": "words"})
                continue
        al, bl = a.splitlines(), b.splitlines()
        why: tuple[str, str | None] | None = None
        if pins:
            h = pins_change(al, bl, old, new, B, H, pmap)
            why = h and ("a line beyond a file's digest at BASE rewritten to its digest at HEAD, under its moved path", h)
        elif kind(old) != k:
            why = f"edited, and renamed from a {kind(old)} file to a {k} one", first_unexplained(al, bl, al, bl)
        elif k == "lock":
            why = None if same else lock_change(a, b, new, pmap, lmods, H)
        elif k == "moves":
            why = None if same else moves_change(a, b, pmap, B)
        elif k == "lean":
            h = first_unexplained(al, bl, al, bl, pair=lambda x, y: lean_import_rewrite(x, y, lmods, dotted))
            why = h and ("edit beyond import lines that name a module renamed in the range", h)
        else:
            rel, py = name in RELATIVE_NAMES, new.endswith(".py")
            na_text = resolve_relative(a, old, pmap) if rel else join_chains(a) if py else a
            n0 = na_text.splitlines()
            na = n0 if same else substitute(na_text, pmap).splitlines()
            nb = (resolve_relative(b, new, {}) if rel else join_chains(b) if py else b).splitlines()

            def at(tree: Tree, path: str, text: str):
                return lambda mod: path if mod is None else module_file(tree, text, mod)

            sh = is_shell(old, a) and is_shell(new, b)
            sh_names = (sh_self_names(a), sh_self_names(b)) if sh else (set(), set())

            def pair(x: str, y: str, at_old=at(B, old, a), at_new=at(H, new, b)) -> bool:
                return any((py and (import_rewrite(z, y, modules, old, new) or anchor_rewrite(z, y, at_old, at_new, pmap)))
                           or (sh and sh_anchor_rewrite(z, y, old, new, *sh_names, pmap))
                           or dotted_rewrite(z, y, modules) or registry_pair(new, z, y)
                           for z in dict.fromkeys((x, substitute(x, short))))

            def block(xs: list[str], ys: list[str]) -> bool:
                return import_block(xs, ys, modules, old, new)
            h = first_unexplained(al, bl, na, nb, pair=pair, added=lambda y: registered(new, y), block=block,
                                  left=lambda i, y: partly(n0[i], y, pmap))
            why = h and ("relative path now resolves elsewhere" if same else "edit beyond moved paths and module names", h)
            if why and not same and old == new and name == "lakefile.toml":
                why = lakefile_in_place(a, b, new, moves, new_lake) and why
            elif why and not same and old == new and name == "lake-manifest.json":
                why = manifest_change(a, b, old, new, pmap, H, added=new_lake) and why
        if why:
            refuse(e, *why)
        elif same:
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
