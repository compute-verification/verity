"""`move-check`: certify that BASE..HEAD is a pure move: drive's queue lands a layout move it regenerated on `next` only once this
certifies it. It relaxes no gate: a certified move still goes through `check`.

    uv run python tools/verity/check/move_check.py BASE HEAD [--map OLD=NEW ...] [--words MAP] [--json]

Every path git sees changed must be one of:
- a rename with byte-identical content and mode (git's 100% rename);
- an edit, in place or with a rename, to a text file that leaves it unchanged once each moved path in it (from the root, also
  after a variable such as `$REPO/`) is rewritten to where it is now or, in a line, left as it was (a top-level directory's
  name is also a word: `examples` in prose, a JSON key, a function's name), in `RELATIVE_NAMES` a quoted relative path (`".."`, `"../level3"`) resolved against
  the file's own directory first (in a `*.md` file, one in backticks or a link's target; in a Lake file, a requirement's
  `path` and a path package's `dir` also where written with no leading dot, `"composition"`), in a `*.py` file a pathlib
  chain `ROOT / "a" / "b"` read as `"a/b"`. A changed line may also, the rest of it kept byte for byte, rewrite each
  place it names from the file's own to name, from where the file now is, where that place moved (`placed`): a relative path from the file's directory (`../../verity/x/`, in a
  comment, prose or code), a path below a moved directory the file is in and stays in (`rho/circuits.py` in a file of
  `verity/protocols/inference_only/rho/`, a path BASE has), or in a `*.py` file a pathlib join on a name the file binds only to its own place
  (`HERE.parent / "rho"` after `HERE = Path(__file__).resolve().parent`). Each changed line left may only rewrite a
  moved path as written below a directory its old and new places share (`verity_vllm/program/predict/`), or a dotted
  name (`verity.protocols.flock.tool:FLOCK_PURE`, `verity_vllm.protocols.remote_bandwidth`'s) from where a renamed module was
  to where it is now; in any of them, the modules a run of import lines binds each name to
  (`from a import m as M` to `from b import m2 as M`, a line split in two, a `from a import (x,` whose names go on
  past the change, or a `from a import x, \\` (`rules`'s `continued`); in a heredoc, or in a string literal
  `"from a import m\n"` whose quotes the move keeps); in a `*.py` file, also the module of an `import M` /
  `from M import` line (a relative one, `.m`, read from the file's own place), or a `Path(__file__).parents[N]` anchor
  (or `Path(m.__file__)`'s, and the path joined to it) recounted to name, from where it now is, what the old one named,
  moved (one with no path, in a file that left the directory it names, joins its way back in parentheses:
  `(Path(__file__).parents[2] / "old/dir")`); in a shell script, likewise a path from its own directory
  (`$(dirname "$0")/../..`, or after `${BASH_SOURCE[0]}` or a variable it sets to its own path, `SH_SELF`);
- an added line in `REGISTRIES` that registers exactly a package the range created: a root `pyproject.toml` workspace
  member (or its `tool.uv.sources` entry) for a directory that is new at HEAD, a `tests/test_lean_packages.py` entry for a
  new Lake package with its `lean-audit.json`; and in any `pyproject.toml`, an added line of a new Lake package's own
  files (its lakefile, manifest, lock or toolchain), as a suite's inputs list what Lake reads;
- an edit to a `*.lean` file whose changed lines only rewrite an `import M` to the new module of a `.lean` file renamed in
  the range (a module is its path below its Lake package, `/` as `.`, without `.lean`), as when a Lean split gives the
  proof modules a root of their own while their declarations keep their namespace, or a moved path from the root (a doc
  comment's), as in a text file;
- an edit to a `lean-audit.json` that keeps `pins` (the guarantees' records; `guarantees` too, if a lock has it) and
  `reads` byte for byte (a `reads` key, a module's name, may follow the module's move), may change the structural `roots`,
  `proved_in` (each entry a Lake package at HEAD) and `dependencies` (the fetched modules the package imports, which a
  split shares out and the Lean check holds to the build), and changes any other field only by moved paths and module
  names, a moved path there too left where it is a word (a reason that calls a conclusion `security`, when the directory
  of that name moves); a `runs` entry may move to another lock the range touches (a proofs package's new one), its runner module
  renamed, its package-relative paths following the move and its `why` changed only by moved paths from the root and
  the dotted names of moved Python modules (`rules`'s `why` in the `--json` report lists the locks whose `why` a dotted
  name changed; a string of it may also change as a line may by the map's rows, below), and must end up in exactly
  one, otherwise unchanged (`runs_moved`);
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
- in a JSON pin of `tools/verity/check/pins.py`, a Lake manifest's digest rewritten likewise, since a move changes a
  manifest only in where its path packages sit, which no pinned output reads (`manifest_pin`);
- an entry appended to research's `moves.json` whose paths agree with the range's (`moves_change`);
- a new, empty `__init__.py` in a directory at or above a `.py` file the range moved there out of a package (its old
  directory had an `__init__.py`) or from outside every package (none at or above it; `rules`'s `level`) to a directory
  not run by path (`tests/`, `lean/`), and no higher than the root of that file's package after the move (`[roots] new`
  of a path map under `tools/move/` at BASE): the package level `tools/move/layout.py` writes so the module imports by
  its new name. And a
  deleted `__init__.py` of a package the range's moves took away (files left its directory by a rename and HEAD has none
  at or below it; `rules`'s `emptied` in the `--json` report) that holds nothing but whitespace and comments at BASE, read
  by the mover's own `only_comments`: one with a docstring, code, or text that doesn't parse is refused, and it gets a row
  in the map saying where it goes or a hand commit before the move that deletes it;
- an edit to a shell script that recounts a path after a variable the script sets once to its own directory
  (`here=$(cd "$(dirname "$0")" && pwd)`, then `$here/../..`), as a path from its own directory is (`rules`'s `anchor`);
- a list of paths rewritten as a cover (`rules`'s `cover`): a list `tools/move/layout.py` reads as one (its `lists`: a
  Python list, tuple or set of string literals, a TOML or JSON array of strings, the objects of `git rev-parse` and the
  members of `tar`) with an entry that names what went to no one place, a directory whose files the range sent apart or
  a file it deleted. Each new entry is an old one as the range sent it whole or as its row of the map sends it; or a
  part, which holds, of HEAD's files that the range did not add, only files the old list's paths held at BASE, one of
  them a file of an entry the list no longer has; or, where the old list named the `__init__.py` of a package the range
  took away, a package level above an entry the list did not have. The new list holds every file of each old entry it no
  longer has, in any order and at any level; nothing else of the list changes. An `__init__.py` git pairs with one where
  no other file of its directory went is deleted and added, not moved;
- on a line none of these explains, what the move's own path map says beyond the renames (`Rows`): the map each commit of
  the range names (`Layout-move: tools/move/NAME.toml`, `tools/move/stage.py`'s line), read as BASE has it. A reference to
  a directory the map splits may go where its `[tokens]` row sends it, or stay, when the range bears the row out (part of
  the directory is now below the row's): the directory itself (`verity/protocols/flock/prover/`, also as a pathlib chain or a quoted
  relative path), a pattern below it, and a path below it that names nothing at BASE; a path below it that BASE has
  follows only its own move. In a `*.py` file an anchor that names the directory (`Path(__file__).parents[1]`) may be
  recounted to name, from where the file now is, where the row sends it. On such a line a moved path, or such a
  reference, is also read after each prefix
  `tools/move/layout.py` reads as the root (its `ROOT_PREFIX`: `<tree>/`, `{r}/`, `$T/wt-$c/`, `~/verity/`, a pod's
  `/workspace/`; and as a shell variable's default, `${TESTS:-verity/x.py}`), one pattern for both, and after each prefix
  the map lists for the root (`[rewrite] root_prefixes`: a pod's install directory). A file the map lists
  under `[drop] paths` may be deleted. Where a `[split]` row names the part one file means by such a directory
  (`"old file: old directory" = { to = "..." }`), that file's references to it go there instead, borne out the same
  way. And a path the line names from its file's directory, or else from its package's
  root (the directory of the nearest `pyproject.toml` above the file), may be based again: written from where that
  directory went (the file's new directory; the directory the `pyproject.toml` moved to, or where the package
  directory's `[tokens]` row sends a reference to it, one of the two for the whole line) to where the path went
  (`ops/README.md` to `../definitions/ops/README.md`); a plain path the root also has is a path from the root. `--json`
  lists under `rules` the paths each of these took (`tokens`, `split`, `prefixes`, `root_prefixes`, `drop`, `relative`);
- any of these after the move scripts' word maps (`tools/move/*.toml` with `[words]`, read at BASE) are applied, as
  `tools/move/rename.py` applies them, with only the words that rename what the range moved: a path the map moves to
  where the range moved it, or a Lean module likewise; or a Lean declaration of a `.lean` file the range edits, to a name
  BASE never spells in a file the map rewrites (outside a `.lean` file's comments), so that the rename merges no two
  names (a `[[words_in]]` entry's words
  in the files its globs match; `word_maps`). A text file the words alone rewrite into its new text is an edit of that
  kind, whatever the file, and so is one that also drops each list entry the words made a repeat of the one before it
  (`merged`); a JSON file may come back with each object whose keys were sorted and that renamed keys leave out of
  order sorted again, as `rename.py` writes it (`resorted`).
A move by a word map with `scope = "prose"` (`--words MAP`, else the map HEAD's `Move: tools/move/NAME.toml` line names, as
BASE has it) is judged by that map instead (`Terms`): each file it edits is BASE's text with some of the map's words
replaced, each in prose (a comment, a docstring, Markdown's text); or in Python or Lean code, as part of a name it renames to
one the file doesn't spell at BASE, the same in every file, and not one a lock's records name; or in a literal or a data
file, as a path or module the range moved. Otherwise a string literal and a data file keep their bytes, and a file a JSON
pin records stays as it was.
Anything else is refused, with what differed and its first offending hunk. Exit 0 certified, 1 refused, 2 on a usage or
git error.

The path map comes from the renames. Each rename contributes the shortest prefix pair (old directory, new directory) whose
whole subtree in BASE moved there file by file. When only `verity/protocols/pous` moved, `protocols -> security_proofs` is never
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
import collections
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
#: the pins of Lean sources, some of them JSON ({path: digest} under a key, `json_pins`)
PINS_PY = "tools/verity/check/pins.py"
PIN_ENTRY = re.compile(r'^\s*"([^"]+)"\s*:\s*"([0-9a-f]{64})",?\s*$')
ROOT = Path(__file__).resolve().parents[3]
IMPORT = re.compile(r"^(\s*from\s+)([\w.]+)(\s+import\b.*)$|^(\s*import\s+)([\w.]+)(.*)$")
LEAN_IMPORT = re.compile(r"^(\s*(?:(?:public|private|meta)\s+)*import\s+(?:all\s+)?)([\w.]+)(\s*(?:--.*)?)$")
RELATIVE = re.compile(r"""(["'])(\.\.?(?:/[^"'\s]*)?)\1""")
#: a place a line names from its own file's (`placed`): a pathlib join on a name (`HERE.parent / "rho"`), a relative path
#: (`../../verity/x/`), or a path with a `/` in it (`rho/circuits.py`)
SPOT = re.compile(r'(?<![\w.])(?P<var>[A-Za-z_]\w*)(?P<up>(?:\.parent\b)*|\.parents\[\d+\])\s*/\s*"(?P<rest>[^"]+)"'
                  r"|(?<![\w.\-/~$*:@\\])(?P<path>(?:\.\.?/)+(?:[\w.\-*]*[\w\-*](?:/[\w.\-*]*[\w\-*])*/?)?"
                  r"|[\w\-*][\w.\-*]*(?:/[\w.\-*]*[\w\-*])+/?)")
#: a module-level binding, which `places` reads for a name bound to the file's own place
BIND = re.compile(r"^([A-Za-z_]\w*)\s*=\s*(.+?)\s*(?:#.*)?$", re.M)
MD_RELATIVE = re.compile(r"(?:(?<=`)|(?<=\]\())\.\.?/[^`)\s]*(?=[`)])")
#: a Lean name where it is declared: after a declaring keyword, as a constructor (`| c`, `| c : T`) or a field (`  f : T`);
#: and the scopes that qualify it (`namespace N`, `section`, `mutual`, each closed by an `end`)
LEAN_DECL = re.compile(r"""^[ \t]*(?:
      (?:@\[[^\]\n]*\][ \t]*)*(?:(?:private|protected|public|noncomputable|partial|unsafe|nonrec|scoped|local|meta)[ \t]+)*
        (def|theorem|lemma|abbrev|instance|opaque|axiom|structure|inductive|class(?:[ \t]+inductive)?)[ \t]+
        ([^\s:({\[⦃]+)
    | \|[ \t]*([\w'.]+)(?=[ \t]*(?:[:({\[⦃]|$))
    | [ \t]+([\w'.]+)(?=[ \t]*:(?!=))
    | (?:(?:noncomputable|public|private)[ \t]+)*(namespace|section|mutual|end)\b[ \t]*([^\s]*)
    )""", re.M | re.X)
LEAN_COMMENT = re.compile(r"/-.*?-/|--[^\n]*", re.S)
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
#: a layout move's line naming its path map, as `tools/move/stage.py` writes it
LAYOUT_LINE = re.compile(r"^Layout-move: (tools/move/[\w.-]+\.toml)(?: --no-lock)?$", re.M)
#: what a line goes on to name below a directory: `/tests/test_x.py`
BELOW = re.compile(r"(?:/[\w.\-]+)*")
#: the one prefix `_paths_pattern` reads as the root: a variable holding the checkout
CHECKOUT = re.compile(r"(?<![\w.\-/$])\$\{?\w+\}?/")
#: a path a line may name from its file's directory or its package's root, as `tools/move/layout.py` reads one: after a
#: variable holding a directory (`$PWD/../../x`), from a dot (`../x`, `./x`) or plain (`x/y`)
REBASED = re.compile(r"(?P<v>\$\{?\w+\}?/)(?P<p>(?:\.\./)+[\w.\-]+(?:/[\w.\-]+)*/?)"
                     r"|(?<![\w.\-/$}])(?P<q>(?:\.\.?/)+[\w.\-]+(?:/[\w.\-]+)*/?|[\w\-][\w.\-]*(?:/[\w.\-]+)+/?)")
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

    def below(self, p: str) -> list[str]:
        """The files path `p` names: itself, or those below it."""
        return self.files[bisect.bisect_left(self.files, p + "/"):bisect.bisect_left(self.files, p + "0")] or [p] * (p in self.set)

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

    @functools.cache
    def digest(self, path: str) -> str | None:
        data = self.blob(path)
        return None if data is None else hashlib.sha256(data).hexdigest()


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


def either(old: str, new: str, refs: list[tuple]) -> list[tuple] | None:
    """The ``refs`` ((start, end, what it becomes[, the rules that read it, whether it must change])) that `new` rewrites,
    when it is `old` with each either rewritten or, unless it must change, left as it was; else None."""
    parts, at = [], 0
    for s, e, to, *more in refs:
        was = [] if more[1:] and more[1] else [re.escape(old[s:e])]
        parts += [re.escape(old[at:s]), f"({'|'.join([*was, *map(re.escape, (to,) if isinstance(to, str) else to)])})"]
        at = e
    m = re.fullmatch("".join(parts) + re.escape(old[at:]), new) if parts else None
    return m and [r for r, got in zip(refs, m.groups()) if got != old[r[0]:r[1]]]


def partly(old: str, new: str, pmap: dict[str, str]) -> bool:
    """Whether the line `new` is `old` with each moved path in it either rewritten to where it is now or left as it was: a
    top-level directory's name is also a word (`examples` in prose, a JSON key, a function's name) that the move leaves."""
    if not pmap:
        return False
    return either(old, new, [(m.start(2), m.end(2), pmap[m[2]])
                             for m in _paths_pattern(tuple(sorted(pmap, key=len, reverse=True))).finditer(old)]) is not None


def short_forms(pmap: dict[str, str], base: Tree) -> dict[str, str]:
    """Each moved path as written below a directory its old and new paths share (`verity_vllm/program/predict` for
    `adapters/vllm/verity_vllm/program/predict`), keeping at least one shared part; dropped where two moves disagree or where it
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
    """Each quoted relative path (in Markdown, in backticks or a link's target), as `<root>/` and the repository path it names
    from `path`'s directory, mapped by `pmap`. In a Lake file, a path to another package is one with no leading dot too
    (`path = "composition"`, a package inside the requiring one): Lake reads it from the package's directory, and
    `tools/move/layout.py` rewrites it as one (its `lake_spans`)."""
    here = posixpath.dirname(path)

    def rooted(rel: str) -> str | None:
        p = posixpath.normpath(posixpath.join(here, rel))
        return None if p.startswith("..") else f"<root>/{map_path(p, pmap)}"
    if path.endswith(".md"):
        return MD_RELATIVE.sub(lambda m: rooted(m.group(0)) or m.group(0), text)
    for a, b in reversed(_rename().L.lake_spans(posixpath.basename(path), text)):
        if not text[a:b].startswith(".") and (r := rooted(text[a:b])):
            text = text[:a] + r + text[b:]
    return RELATIVE.sub(lambda m: (r := rooted(m.group(2))) and f"{m.group(1)}{r}{m.group(1)}" or m.group(0), text)


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
#: `from a import x, y, \`, its names continued likewise
FROM_SLASH = re.compile(r"^(\s*)from\s+([\w.]+)\s+import\s+([\w\s,]*?),?\s*\\()$")


def bindings(lines: list[str], path: str) -> list[tuple[str, str, str, str]] | None:
    """Each name the import lines bind, as (indent, comment, the dotted name it binds, the name), or None if a line isn't
    one such import. A parenthesized `from a import (`, or one continued after `\\`, also binds `a` itself, under the name
    `(`: its continuation lines may be unchanged context, so `a`'s rewrite must be checked even when the line names nothing."""
    out = []
    for line in lines:
        if (m := FROM.match(line)) or (m := FROM_PAREN.match(line)) or (m := FROM_SLASH.match(line)):
            if m.re is not FROM:
                out.append((m.group(1), m.group(4) or "", absolute(m.group(2), path), "("))
            for part in filter(str.strip, m.group(3).split(",")) if m.re is not FROM else m.group(3).split(","):
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


def anchors_out(text: str, anchors: list[re.Match], joined: list[bool]) -> str:
    """`text` with each anchor a NUL, and with it the parentheses around one that `joined` marks."""
    out, at = [], 0
    for m, j in zip(anchors, joined):
        s, e = m.span()
        if j and text[s - 1:s] == "(" and text[e:e + 1] == ")":
            s, e = s - 1, e + 1
        out += [text[at:s], "\0"]
        at = e
    return "".join(out) + text[at:]


def anchor_rewrite(old: str, new: str, at_old, at_new, pmap: dict[str, str]) -> bool:
    """Whether `new` is `old` with each anchor recounted so that, from where it now is (`at_new(module alias or None)`),
    it names the path the old one named, moved by `pmap`. An anchor with no path that gains one may gain parentheses
    with it, as `tools/move/layout.py` writes the anchor of a file that left the directory it names
    (`Path(__file__).resolve().parent` to `(Path(__file__).resolve().parents[2] / "old/dir")`); the parentheses the
    line already had around it, as a call's own (`str(Path(__file__).resolve().parents[2] / "old/dir")`), are not ones
    it gained."""
    a, b = list(ANCHOR.finditer(old)), list(ANCHOR.finditer(new))
    if not a or len(a) != len(b):
        return False
    joined = [x.group("rest") is None and y.group("rest") is not None for x, y in zip(a, b)]
    plain = ANCHOR.sub("\0", old)
    if plain not in (ANCHOR.sub("\0", new), anchors_out(new, b, joined)):
        return False
    for x, y in zip(a, b):
        dx, dy = anchored(x, at_old(x.group("mod"))), anchored(y, at_new(y.group("mod")))
        if x.group("head") != y.group("head") or dx is None or dy is None or map_path(dx, pmap) != dy:
            return False
    return True


def anchor_rows(line: str, at_old, rows: Rows) -> dict[str, str]:
    """Each directory an anchor of `line` names that the range splits, to where its `[tokens]` row sends a reference to
    it (`Rows.went`): what `anchor_rewrite` may recount such an anchor by, beside the path map."""
    out = {}
    for x in ANCHOR.finditer(line):
        d = anchored(x, at_old(x.group("mod")))
        if d is not None and (to := rows.went(d))[1]:
            out[d] = to[0]
    return out


def rebinds(text: str, v: str) -> int:
    """How often ``text`` binds the name ``v``: an assignment (also in a tuple or a call's keyword), a loop, `with` or
    import target, a `global`, a parameter."""
    e = re.escape(v)
    return len(re.findall(rf"(?<![\w.]){e}\s*(?:,[\w\s,]*)?(?::=|[-+*/%&|^@]?=(?!=))"
                          rf"|\b(?:as|for|global|nonlocal|import)\s+(?:[\w\s,]*,\s*)?{e}\b"
                          rf"|\bdef\s+\w+\s*\([^)]*?(?<![\w.]){e}\b|\blambda\b[^:]*?(?<![\w.]){e}\b", text))


def places(text: str, path: str) -> dict[str, str]:
    """The names ``text`` (the file at ``path``) binds at module level to a directory from its own place
    (`HERE = Path(__file__).resolve().parent`) and nowhere else, each to that directory."""
    out = {}
    for v, rhs in BIND.findall(text):
        m = ANCHOR.fullmatch(rhs)
        if m and m.group("mod") is None and rebinds(text, v) == 1 and (d := anchored(m, path)) is not None:
            out[v] = d
    return out


def ancestor(d: str, n: int) -> str | None:
    parts = [p for p in d.split("/") if p]
    return None if n > len(parts) else "/".join(parts[:len(parts) - n])


def spot_moved(x: re.Match, y: re.Match, old: str, new: str, pmap: dict[str, str], base: Tree,
               binds: tuple[dict[str, str], dict[str, str]]) -> bool:
    """Whether the place ``y`` names from ``new`` is where the place ``x`` named from ``old`` moved (`placed`)."""
    if x["var"] is not None or y["var"] is not None:
        if x["var"] is None or y["var"] is None or x[0][:x.start("rest") - x.start()] != y[0][:y.start("rest") - y.start()]:
            return False
        n = int(x["up"][9:-1]) + 1 if x["up"].startswith(".parents[") else x["up"].count(".parent")
        do, dn = (b.get(x["var"]) for b in binds)
        do, dn = (None if d is None else ancestor(d, n) for d in (do, dn))
        if do is None or dn is None:
            return False
        p, q = (posixpath.normpath(posixpath.join(d, m["rest"])) for d, m in ((do, x), (dn, y)))
        return not p.startswith("..") and map_path(p, pmap) == q and (p != q or do != dn)
    t, u = x["path"], y["path"]
    if t.endswith("/") != u.endswith("/"):
        return False
    if substitute(t, pmap) == u:
        return True
    if t.startswith(("./", "../")):
        if not u.startswith(("./", "../")):
            return False
        p, q = (posixpath.normpath(posixpath.join(posixpath.dirname(f), s)) for f, s in ((old, t), (new, u)))
        return not p.startswith("..") and map_path(p, pmap) == q and (p != q or old != new)
    d = posixpath.dirname(old)
    while d:
        to = map_path(d, pmap)
        p, q = posixpath.normpath(posixpath.join(d, t)), posixpath.normpath(posixpath.join(to, u))
        if (to != d and (new.startswith(to + "/")) and p.startswith(d + "/") and q.startswith(to + "/")
                and (p in base.set or base.has_dir(p)) and map_path(p, pmap) == q):
            return True
        d = posixpath.dirname(d)
    return False


def placed(x: str, y: str, old: str, new: str, pmap: dict[str, str], base: Tree,
           binds: tuple[dict[str, str], dict[str, str]]) -> bool:
    """Whether the line ``y`` (of ``new``) is the line ``x`` (of ``old``) with nothing changed but places named from the file's
    own (`SPOT`), each rewritten to name, from ``new``, where the place it named from ``old`` moved: a pathlib join on a name
    the file binds to its own place (``binds``, `places` at BASE and at HEAD), a relative path from its directory, a path
    below a moved directory the file is in and still is (a file of the package that moved, which BASE has), or a moved path
    from the root. Any other change to the line, a place that moved elsewhere or one that did not move, is refused."""
    a, b = list(SPOT.finditer(x)), list(SPOT.finditer(y))
    if not a or len(a) != len(b) or SPOT.sub("\0", x) != SPOT.sub("\0", y):
        return False
    diff = [(s, t) for s, t in zip(a, b) if s[0] != t[0]]
    return bool(diff) and all(spot_moved(s, t, old, new, pmap, base, binds) for s, t in diff)


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
                      pmap: dict[str, str], here: frozenset[str] = frozenset()) -> bool:
    """Whether `new` is `old` with each of a shell script's paths from its own directory recounted so that, from the script's
    new place, it names the path the old one named, moved by `pmap`: a path after `$(dirname "$0")`, and one after a
    variable of `here` (`$here/../..`), which the script sets to its own directory before and after the move
    (`tools/move/layout.py`'s `SH_HERE`)."""
    anchor = _rename().L.sh_anchor(here) if here else SH_ANCHOR
    a, b = list(anchor.finditer(old)), list(anchor.finditer(new))
    if not a or len(a) != len(b) or anchor.sub("\0", old) != anchor.sub("\0", new):
        return False
    for x, y in zip(a, b):
        ref = (x.group("ref") or "")[1:].strip("{}")
        if x.group("head") != y.group("head") or ref and (ref not in names_old or ref not in names_new):
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


def between(text: str, span: tuple[int, int], entries: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Where each word of a list's text (`span`) outside `entries` is: its brackets, its comments' words, a command's
    other words and the entries that are no path, without the quotes, commas and white space around them."""
    cuts = [span[0], *(i for s, t in entries for i in (s, t)), span[1]]
    return [(i + m.start(), i + m.end()) for i, j in zip(cuts[::2], cuts[1::2]) for m in re.finditer(r"[^\s,\"']+", text[i:j])]


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
        if not same_value(module_sub(substitute(a[k], pmap), lmods), b[k]) and not partly(a[k], b[k], pmap):
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
    def pair(x: str, y: str) -> bool:
        mx, my = PIN_LINE.match(x), PIN_LINE.match(y)
        if not (mx and my):
            return False
        po = posixpath.normpath(posixpath.join(posixpath.dirname(old), mx[2]))
        pn = posixpath.normpath(posixpath.join(posixpath.dirname(new), my[2]))
        return map_path(po, pmap) == pn and base.digest(po) == mx[1] and head.digest(pn) == my[1]
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
def json_pins() -> dict[str, str]:
    """`tools/verity/check/pins.py`'s pins that record {path: digest} under a JSON key: {pin: the directory its paths are
    below}."""
    spec = importlib.util.spec_from_file_location("move_check_pins", ROOT / PINS_PY)
    mod = sys.modules[spec.name] = importlib.util.module_from_spec(spec)       # its dataclass looks its module up
    spec.loader.exec_module(mod)
    return {p.path: p.under for p in mod.PINS if p.key is not None}


@functools.cache
def json_pin_keys() -> dict[str, tuple[str, str]]:
    """`json_pins`, each with its JSON key: {pin: (key, the directory its paths are below)}."""
    json_pins()
    return {p.path: (p.key, p.under) for p in sys.modules["move_check_pins"].PINS if p.key is not None}


def manifest_pin(x: str, y: str, under: str, base: Tree, head: Tree, pmap: dict[str, str]) -> bool:
    """Whether a JSON pin's line ``y`` is its line ``x`` at BASE with a Lake manifest's digest at BASE rewritten to its digest
    at HEAD, under its path moved by the map."""
    mx, my = PIN_ENTRY.match(x), PIN_ENTRY.match(y)
    if not (mx and my):
        return False
    po, pn = (posixpath.normpath(posixpath.join(under, m[1])) for m in (mx, my))
    return (posixpath.basename(po) == "lake-manifest.json" and map_path(po, pmap) == pn
            and base.digest(po) == mx[2] and head.digest(pn) == my[2])


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
            if (data.get("words") or data.get("words_in")) and not data.get("scope"):
                out.append((data, R.Words(data)))
        except (tomllib.TOMLDecodeError, SystemExit, KeyError, TypeError, AttributeError):
            continue
    return out


def package_roots(base: Tree) -> set[str]:
    """The directory of each import package after a move, as the path maps at BASE give it (`[roots] new` of each `.toml`
    under `tools/move/`, at any depth: a layout's maps may sit in a directory of their own): `tools/move/layout.py` writes
    package levels from a moved module up to its root and no higher."""
    out = set()
    for p in base.files:
        if not (p.startswith(MAPS) and p.endswith(".toml")):
            continue
        try:
            roots = tomllib.loads(base.text(p) or "").get("roots")
        except tomllib.TOMLDecodeError as e:
            raise SystemExit(f"move-check: {p} doesn't parse at BASE, so the package roots it may name can't be read: {e}") from e
        new = roots.get("new") if isinstance(roots, dict) else None
        if isinstance(new, dict):
            out |= {d.strip("/") for d in new.values() if isinstance(d, str)}
    return out


def layout_maps(base: Tree, head: str, cwd: str | None) -> list[dict]:
    """The path maps the range's layout moves ran by (each commit's `Layout-move: tools/move/NAME.toml` line), as BASE has
    them: what a map says beyond the renames is read from the map BASE holds, never from the range's own files."""
    log = git("log", "--format=%B", f"{base.rev}..{head}", cwd=cwd).decode(errors="replace")
    out = []
    for path in dict.fromkeys(LAYOUT_LINE.findall(log)):
        try:
            out.append(tomllib.loads(base.text(path) or ""))
        except tomllib.TOMLDecodeError as e:
            raise SystemExit(f"move-check: the move's path map {path} doesn't parse at BASE: {e}") from e
    return out


class Rows:
    """What the move's maps (`layout_maps`) say beyond the renames, read for a line nothing else explains and never added to
    the path map, so no rename is paired by it, and an anchor is recounted by it only where the directory it names is a
    `[tokens]` row's own (`anchor_rows`); and for such a line, a moved path after each prefix the mover reads as the
    root (`pattern`).

    `[tokens]`: where a reference to a directory the map splits goes. A row counts only where the range bears it out: a
    directory of BASE that didn't move whole, part of it now below the row's directory; two maps that disagree on a
    directory name it nowhere. It reads the directory itself (`verity/protocols/flock/prover/`, `verity/protocols/flock/prover/*.patch`) and a path
    below it that names nothing at BASE (a file a test makes up); a path below it that BASE has follows only its own move.

    `[split]`: for one file (by its path at BASE) and one such directory, the part the file means by it (`{ to = "..." }`),
    which that file's references to the directory follow in place of the `[tokens]` row. It counts as a `[tokens]` row
    does: the directory didn't move whole and part of it is now below the row's.

    `[drop] paths`: the files the move deletes.

    `[rewrite] root_prefixes`: what stands for the root before a path, beyond what the mover reads so in every tree (a
    pod's install directory, `/opt/vy-verifier/current/`), each as it is written.

    A relative path based again (`relative`): a path the line names from its file's directory, else from its package's
    root (the nearest `pyproject.toml` above the file at BASE), written from where that directory went (the file's new
    directory; where the package's `pyproject.toml` moved, or, every such path of the line, where the package
    directory's `[tokens]` row sends a reference to it, as the mover bases one) to where the path went. A plain path the
    root also has is a path from the root, and a directory the range splits with no row went nowhere one path names."""

    def __init__(self, maps: list[dict], pmap: dict[str, str], base: Tree, head: Tree):
        self.pmap, self.base, self.head, self.tokens, self._went = pmap, base, head, {}, {}
        self.roots = sorted({p for data in maps if isinstance(rw := data.get("rewrite"), dict)
                             and isinstance(ps := rw.get("root_prefixes"), list) for p in ps if _rename().L.root_prefix_ok(p)})
        self.drop = {p for data in maps if isinstance(d := data.get("drop"), dict) and isinstance(ps := d.get("paths"), list)
                     for p in ps if isinstance(p, str)}
        bad = set()
        for data in maps:
            rows = data.get("tokens")
            for d, n in rows.items() if isinstance(rows, dict) else ():
                d, n = d.rstrip("/"), str(n).rstrip("/")
                lo, hi = bisect.bisect_left(base.files, d + "/"), bisect.bisect_left(base.files, d + "0")
                if d not in pmap and any(map_path(f, pmap).startswith(n + "/") for f in base.files[lo:hi]) \
                        and self.tokens.setdefault(d, n) != n:
                    bad.add(d)
        for d in bad:
            del self.tokens[d]
        self.parts, bad = {}, set()
        for data in maps:
            rows = data.get("split")
            for k, v in rows.items() if isinstance(rows, dict) else ():
                f, _, d = k.rpartition(": ")
                if f and isinstance(v, dict) and isinstance(to := v.get("to"), str):
                    d, to = d.strip("/"), to.rstrip("/")
                    if d not in pmap and any(map_path(x, pmap).startswith(to + "/") for x in base.below(d)) \
                            and self.parts.setdefault((f, d), to) != to:
                        bad.add((f, d))
        for k in bad:
            del self.parts[k]

    @functools.cached_property
    def pattern(self) -> re.Pattern:
        """`_paths_pattern` over the moved paths and the split directories, after whatever `tools/move/layout.py` reads as
        the root (its `ROOT_PREFIX`, the `<root>/` of a relative path `resolve_relative` read among them, each prefix the
        maps list for the root, and a shell variable's default, `${DEFS:-verity/x.json}`)."""
        L = _rename().L
        root = "|".join((L.root_prefix(self.roots), L.SH_DEFAULT.pattern.removesuffix(r"\Z")))
        keys = sorted({*self.pmap, *self.tokens, *(d for _, d in self.parts)}, key=len, reverse=True)
        return re.compile(rf"(?:(?<![\w.\-/])|(?<=^/)|({root}))(" + "|".join(map(re.escape, keys)) + r")(?![\w.\-])", re.M)

    def references(self, text: str, resolved: bool = False,
                   file: str | None = None) -> list[tuple[int, int, str, tuple[str, ...], bool]]:
        """Each moved path and each reference to a split directory in ``text``, a line of ``file`` at BASE: (start, end,
        what it becomes, the rules that read it beyond `_paths_pattern`'s moved path, whether it must change: in
        ``resolved`` text, a relative path `resolve_relative` read, which names its place)."""
        out = []
        for m in self.pattern.finditer(text):
            key, must = m[2], resolved and text.endswith("<root>/", 0, m.start(2))
            plain = m[1] is None or must or ((c := CHECKOUT.match(text, m.start(1))) is not None and c.end() == m.end(1))
            rules = () if plain else ("root_prefixes",) if m[1] in self.roots else ("prefixes",)
            if key in self.pmap:
                out.append((m.start(2), m.end(2), self.pmap[key], rules, must))
                continue
            below = key + BELOW.match(text, m.end(2))[0].rstrip(".")
            if below == key or not (below in self.base.set or self.base.has_dir(below)):
                if (file, key) in self.parts:
                    out.append((m.start(2), m.end(2), self.parts[file, key], ("split", *rules), must))
                elif key in self.tokens:
                    out.append((m.start(2), m.end(2), self.tokens[key], ("tokens", *rules), must))
        return out

    def went(self, p: str) -> tuple[str | None, bool]:
        """Where the BASE path ``p`` went and whether a `[tokens]` row says so; None for a directory the range splits
        with no row, and for a place HEAD doesn't have."""
        if p not in self._went:
            q, row = map_path(p, self.pmap), False
            if q == p and p not in self.base.set:
                lo, hi = bisect.bisect_left(self.base.files, p + "/"), bisect.bisect_left(self.base.files, p + "0")
                if p in self.tokens:
                    q, row = self.tokens[p], True
                elif any(map_path(f, self.pmap) != f for f in self.base.files[lo:hi]):
                    q = None
            self._went[p] = (q, row) if q is not None and (q in self.head.set or self.head.has_dir(q)) else (None, False)
        return self._went[p]

    def relative(self, text: str, old: str, new: str, resolved: bool = False, row: bool = False) -> list[tuple]:
        """Each path ``text`` (a line of the file at ``old``, now at ``new``) names from the file's directory or its
        package's root, where based again it is written otherwise: as `references` gives one, what it becomes also as
        `resolve_relative` reads it in ``resolved`` text. ``row``: the package's root is where its directory's `[tokens]`
        row sends a reference to it, where it has one."""
        def named(p: str) -> bool:
            return p != "." and not p.startswith("..") and (p in self.base.set or self.base.has_dir(p))
        out, here = [], posixpath.dirname(old)
        for m in REBASED.finditer(text):
            g = "p" if m["v"] is not None else "q"
            t = m[g]
            while t.endswith(("/", ".")) and not t.endswith(".."):
                t = t[:-1]
            if not t or not t.startswith(".") and named(t):
                continue
            at, by_row = None, False
            if g == "q" and named(p := posixpath.normpath(posixpath.join(here, t))):
                at = posixpath.dirname(new)
            else:
                d = here
                while d and f"{d}/pyproject.toml" not in self.base.set:
                    d = posixpath.dirname(d)
                if d and named(p := posixpath.normpath(posixpath.join(d, t))):
                    at, by_row = (self.tokens[d], True) if row and d in self.tokens else \
                        (posixpath.dirname(map_path(f"{d}/pyproject.toml", self.pmap)), False)
            q, sent = self.went(p) if at is not None else (None, False)
            if q is None:
                continue
            to = posixpath.relpath(q, at or ".")
            if t.startswith("./") and not to.startswith("."):
                to = "./" + to
            if to != t:
                end, forms = m.start(g) + len(t), (to,)
                if resolved and to.startswith("."):     # `resolve_relative` read it from the new file's directory, to no slash
                    slash = "/" * text.startswith("/", end)
                    end, forms = end + len(slash), (to + slash, "<root>/" + posixpath.normpath(posixpath.join(posixpath.dirname(new), to)))
                out.append((m.start(g), end, forms, ("relative", *(("tokens",) if sent or by_row else ())), False))
        return out

    def reads(self, old: str, new: str, pair=None, resolved: bool = False, at: tuple[str, str] | None = None) -> set[str]:
        """The rules of the references ``new`` rewrites, when it is ``old`` with each reference rewritten or left, or
        (``pair``) what `pair` takes ``old`` with every one rewritten to; empty when a moved path alone would do, which is
        not these rows' to say. ``at``: the line's file at BASE and at HEAD, for a relative path based again, which is
        read only where the line is not explained without it."""
        refs = self.references(old, resolved, at and at[0]) if self.pmap or self.tokens or self.parts else []
        more = [[r for r in self.relative(old, *at, resolved, row) if not any(r[0] < e and s < r[1] for s, e, *_ in refs)]
                for row in (False, True)] if at else []
        for each in (refs, *(sorted(refs + m) for i, m in enumerate(more) if m and m not in more[:i])):
            took = either(old, new, each)
            if took is None and pair and each and pair(rewritten(old, each), new):
                took = each
            if took:
                return {r for _, _, _, rs, _ in took for r in rs}
        return set()


def rewritten(text: str, refs: list[tuple]) -> str:
    """``text`` with each of ``refs`` ((start, end, what it becomes or the forms it may take, ...), in order) rewritten."""
    out, at = [], 0
    for s, e, to, *_ in refs:
        out += [text[at:s], to if isinstance(to, str) else to[0]]
        at = e
    return "".join(out) + text[at:]


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


def lean_declared(text: str) -> list[str]:
    """The names ``text`` (a `.lean` file) declares, as written where it declares them and in full: below its namespaces,
    and a field's or a constructor's below the structure or inductive type before it."""
    out, scopes, owner = [], [], ""
    for m in LEAN_DECL.finditer(LEAN_COMMENT.sub("", text)):
        kw, decl, ctor, field, scope, arg = m.groups()
        if scope == "end":
            scopes = scopes[:-1]
        elif scope in ("section", "mutual"):
            scopes.append("")
        elif name := arg if scope else decl or ctor or field:
            full = ".".join(filter(None, [*scopes, owner if not kw and not scope else "", name.removeprefix("_root_.")]))
            out += dict.fromkeys((name, full))
            if scope:
                scopes.append(name)
            elif kw:
                owner = name if kw.split()[-1] in ("structure", "inductive", "class") else ""
    return out


def renaming(base: Tree, table: dict[str, str], declared: list[str], now: set[str], rewrites,
             whole: bool = False) -> dict[str, str]:
    """The entries of ``table`` (a map's words, or with ``whole`` a `[[words_in]]` entry's) that make a rename the range
    made: with them applied, a name of ``declared`` (the edited `.lean` files' at BASE) becomes one of ``now`` (theirs at
    HEAD), where it no longer stands. Each must rename to a name BASE never spells in a file the map rewrites
    (``rewrites``; a `.lean` file's comments only mention names, so prose there may already use the new word), and no
    two to one, so that the rename merges no two names. A word of an earlier move that the range
    didn't make (a stale one, whose rename a later move undid) is never kept."""
    R = _rename()

    def words(t: dict[str, str]):
        return R.Words({"words": {}, "words_in": [{"files": ["*"], "words": t}]} if whole else {"words": t})
    names = "\n".join(declared)
    if not (table := {o: n for o, n in table.items() if words({o: n}).sub(names, "-") != names}):
        return {}
    every, hits = words(table), {}
    for d in declared:
        if (n := every.sub(d, "-")) != d and n in now and d not in now:
            hits |= {o: m for o, m in table.items() if o not in hits and words({o: m}).sub(d, "-") != d}
    if not hits:
        return {}
    grep = subprocess.run(["git", "grep", "-l", "-z", "-F", *(a for n in hits.values() for a in ("-e", n)), base.rev, "--"],
                          cwd=base.cwd, capture_output=True)
    if grep.returncode not in (0, 1):  # 1: nothing matches
        raise subprocess.CalledProcessError(grep.returncode, grep.args, grep.stdout, grep.stderr)
    for p in (p.decode("utf-8", "surrogateescape").partition(":")[2] for p in grep.stdout.split(b"\0") if p):
        if not rewrites(p) or (text := base.text(p)) is None:
            continue
        if p.endswith(".lean"):
            text = LEAN_COMMENT.sub("", text)
        for o, n in list(hits.items()):
            if n in text and words({n: o}).sub(text, p) != text:
                del hits[o]
    seen = [n for n in hits.values()]
    return {o: n for o, n in hits.items() if seen.count(n) == 1}


def word_maps(base: Tree, renamed: list[tuple[str, str]], lmods: dict[str, str], pmap: dict[str, str],
              lean_edited: dict[str, str] | None = None, head: Tree | None = None) -> list:
    """The move scripts' word maps at BASE, each cut to the words that rename what the range moved: a word is kept when it
    renames a path, a directory the range moved whole (as `dir/`) or a Lean module that its map, applied whole, renames to
    where the range moved it, or when it spells a kept word's rename as a pathlib chain or a regex (`spelled`); and a
    word, or a `[[words_in]]` entry's word in the files its globs match, when it renames a Lean declaration of a `.lean`
    file the range edits (``lean_edited``, each BASE path's HEAD path) as the range renamed it, to a name BASE never
    spells (`renaming`)."""
    R = _rename()
    names = [*renamed, *lmods.items(), *((o + "/", n + "/") for o, n in pmap.items())]
    lean_edited = lean_edited or {}
    declared = {p: lean_declared(base.text(p) or "") for p in lean_edited}
    now = {d for q in lean_edited.values() for d in lean_declared(head.text(q) or "")} if head else set()
    out = []
    for data, whole in base_maps(base):
        moved = {a: b for a, b in names if whole.path(a) == b and a != b}
        kept = {o: n for o, n in data["words"].items()
                if any(R.Words({"words": {o: n}}).path(a) != a for a in moved)}
        kept |= {o: n for o, n in data["words"].items() if (spelled(o), spelled(n)) in kept.items()}
        mine = [d for p, ds in declared.items() if not whole.excluded(p) for d in ds]
        kept |= renaming(base, {o: n for o, n in data["words"].items() if o not in kept}, mine, now,
                         lambda p: not whole.excluded(p))
        kept_in = []
        for s in data.get("words_in", []):
            def scoped(p: str, files=s["files"]) -> bool:
                return not whole.excluded(p) and any(R._match(p, g) for g in files)
            if w := renaming(base, s["words"], [d for p, ds in declared.items() if scoped(p) for d in ds], now, scoped,
                             whole=True):
                kept_in.append({"files": s["files"], "words": w})
        if kept or kept_in:
            out.append(R.Words({**data, "words": kept, "words_in": kept_in}))
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
    """``text`` with each word map applied as `tools/move/rename.py` applies it: not to a file the map excludes, in a
    lock not to the records, and in a JSON file with each sorted object whose keys it renames sorted again (`resorted`)."""
    R = _rename()
    for w in maps:
        if not w.excluded(path):
            text = R.resorted(text, w.sub(text, path, R.record_spans(text) if lock else []), path)
    return text


#: a move commit's line naming its word map (`research.queue`'s `MOVE_TRAILER_RE`)
MOVE_LINE = re.compile(r"^Move: (tools/move/[\w.-]+\.toml)$", re.M)
#: a name as Python or Lean spells it, one part of a dotted name
NAME = {"python": re.compile(r"[^\W\d]\w*"), "lean": re.compile(r"[^\W\d][\w'!?]*")}
NAME_CHAR = {"python": re.compile(r"\w"), "lean": re.compile(r"[\w'!?]")}
#: what a path or a dotted module in a literal may hold
TOKEN_CHAR = re.compile(r"[^\s\"'`()\[\]{}<>,;=]")


def scoped_map(base: Tree, head: str, cwd: str | None, path: str | None = None) -> tuple[str, dict] | None:
    """The word map a move with `scope = "prose"` renames by (``path``, else the `Move:` line of HEAD's message), as BASE has
    it; None for any other move."""
    if path is None:
        m = MOVE_LINE.search(git("log", "-1", "--format=%B", head, cwd=cwd).decode(errors="replace"))
        path = m and m.group(1)
    try:
        data = tomllib.loads(base.text(path) or "") if path else {}
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"move-check: the move's word map {path} doesn't parse at BASE: {e}") from e
    return (path, data) if data.get("scope") == "prose" and data.get("words") else None


class Terms:
    """The certificate of a move by a word map with `scope = "prose"`: each file it edits must be BASE's text with some of the
    map's words (as `tools/move/rename.py` matches them in that file) replaced, and each one replaced must stand in prose (a
    comment, a docstring, Markdown's text, `rename.regions`); or in Python or Lean code, as a part of a name renamed to one
    the file doesn't spell at BASE, the same name in every file, and not a Lean name a lock records (its record follows a
    rename only by `check.py --update --moved`, which a regenerated move doesn't run); or in a literal or a data file, as a
    path or module the range moved. A string literal and a data file otherwise keep their bytes, so a wire format, a
    record key or a message stays as it was. A file a JSON pin records is not edited at all: the pin's command, which a
    move doesn't run, regenerates it."""

    def __init__(self, base: Tree, map_path: str, data: dict, pmap: dict[str, str], lmods: dict[str, str]):
        R = _rename()
        self.R, self.map, self.words, self.pmap, self.lmods = R, map_path, R.Words(data), pmap, lmods
        self.records = set()
        for p in base.files:
            if posixpath.basename(p) == "lean-audit.json":
                self.records |= set(NAME["lean"].findall(base.text(p) or ""))
        self.pinned: dict[str, str] = {}
        for pin, (key, under) in json_pin_keys().items():
            for rel in json.loads(base.text(pin) or "{}").get(key, {}):
                self.pinned[posixpath.normpath(posixpath.join(under, rel))] = pin
        self.names: dict[str, dict[str, list[str]]] = {}

    def change(self, a: str, b: str, old: str, new: str) -> tuple[str, str | None] | None:
        """Why ``b`` (file ``new`` at HEAD) isn't a certified edit of ``a`` (``old`` at BASE), or None."""
        if old in self.pinned:
            return (f"the JSON pin {self.pinned[old]} records this file's digest: its command regenerates the pin, which a move "
                    "doesn't run, so the move leaves the file as it is", None)
        if self.words.excluded(old):
            return f"{self.map} excludes this file", None
        pattern, repl = self.words.rules(old, a)
        sites = [(m.start(), m.end(), self.R._rewrapped(m.group(0), repl[m.lastindex - 1]) if "\n" in m.group(0)
                  else repl[m.lastindex - 1]) for m in pattern.finditer(a)]
        applied = aligned(a, b, sites)
        if applied is None:
            al, bl = a.splitlines(), b.splitlines()
            return f"edit beyond the words of {self.map}", first_unexplained(al, bl, al, bl)
        spans, lang = self.R.regions(old, a), self.R.language(old, a)
        spelled = None
        for s, e, at, to in applied:
            kinds = {self.R.kind_at(spans, i) for i in range(s, e) if not a[i].isspace()}
            if kinds == {self.R.PROSE}:
                continue
            if kinds == {self.R.CODE} and lang in NAME:
                was, now = around(a, s, e, NAME_CHAR[lang]), around(b, at, at + len(to), NAME_CHAR[lang])
                if spelled is None:
                    spelled = names_outside_prose(a, spans, lang, self.R)
                if now in spelled:
                    return (f"renames {was} to {now}, which {old} spells at BASE: the two names would merge",
                            line_hunk(a, b, s, at))
                if lang == "lean" and (set(was.split(".")) & self.records):
                    return (f"renames {was}, which a lock's records name: a record follows a rename only by "
                            "`check.py --update --moved`, which a regenerated move doesn't run", line_hunk(a, b, s, at))
                self.names.setdefault(was, {}).setdefault(now, []).append(new)
                continue
            if kinds <= {self.R.LITERAL, self.R.CODE}:
                ta, tb = around(a, s, e, TOKEN_CHAR), around(b, at, at + len(to), TOKEN_CHAR)
                if ta != tb and (substitute(ta, self.pmap) == tb or module_sub(ta, self.lmods) == tb):
                    continue
            where = "a string literal or a data file" if self.R.LITERAL in kinds else f"{lang} code"
            return (f"renames {a[s:e]!r} to {to!r} in {where}, which keeps its bytes but for moved paths",
                    line_hunk(a, b, s, at))
        return None

    def conflicts(self) -> list[tuple[str, str]]:
        """(path, why) for each file that renames a name otherwise than another file does."""
        out = []
        for was, nows in sorted(self.names.items()):
            if len(nows) > 1:
                told = ", ".join(f"{n} in {fs[0]}" for n, fs in sorted(nows.items()))
                out += [(f, f"renames {was} to different names in different files ({told})")
                        for fs in nows.values() for f in dict.fromkeys(fs)]
        return out


def names_outside_prose(text: str, spans: list, lang: str, R) -> set[str]:
    """The names ``text`` spells in its code and literals."""
    out, i = set(), 0
    for s, e, k in [*spans, (len(text), len(text), R.PROSE)]:
        if k == R.PROSE:
            out |= set(NAME[lang].findall(text, i, s))
            i = e
    return out


def around(text: str, s: int, e: int, char: re.Pattern) -> str:
    """``text[s:e]`` widened to the run of ``char`` it stands in."""
    while s > 0 and char.match(text[s - 1]):
        s -= 1
    while e < len(text) and char.match(text[e]):
        e += 1
    return text[s:e]


def line_hunk(a: str, b: str, s: int, at: int) -> str:
    """The line of ``a`` at ``s`` and of ``b`` at ``at``, as a hunk."""
    def line(t: str, i: int) -> str:
        return t[t.rfind("\n", 0, i) + 1:(t.find("\n", i) + 1 or len(t) + 1) - 1]
    return f"-{line(a, s)}\n+{line(b, at)}"


def aligned(a: str, b: str, sites: list[tuple[int, int, str]]) -> list[tuple[int, int, int, str]] | None:
    """The ``sites`` of ``a`` ((start, end, replacement), in order, apart) that ``b`` replaces, each with where its
    replacement starts in ``b``, when ``b`` is ``a`` with some of them replaced; None when it isn't."""
    frontier: dict[int, None] = {0: None}
    back: list[dict[int, tuple[int, bool]]] = []
    i = 0
    for s, e, to in sites:
        gap, nxt = a[i:s], {}
        for j in frontier:
            if b.startswith(gap, j):
                k = j + len(gap)
                if b.startswith(a[s:e], k):
                    nxt.setdefault(k + e - s, (j, False))
                if b.startswith(to, k):
                    nxt.setdefault(k + len(to), (j, True))
        if not nxt:
            return None
        back.append(nxt)
        frontier, i = nxt, e
    end = next((j for j in frontier if len(b) - j == len(a) - i and b.endswith(a[i:])), None)
    if end is None:
        return None
    out = []
    for (s, e, to), step in zip(reversed(sites), reversed(back)):
        j, used = step[end]
        if used:
            out.append((s, e, end - len(to), to))
        end = j
    return out[::-1]


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


def certify(base: str, head: str, extra: dict[str, str] | None = None, cwd: str | None = None,
            words: str | None = None) -> dict:
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
    rows = Rows(layout_maps(B, head, cwd), pmap, B, H)
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
    maps = word_maps(B, renamed, lmods, pmap, {e["old"]: e["new"] for e in entries if e["status"] in "RM"
                                               and e["old"].endswith(".lean") and e["shas"][0] != e["shas"][1]}, H)
    moves: dict[str, tuple[str, str]] = {}                  # a module that changed package: (its package at BASE, at HEAD)
    for o, n in renamed:
        if o.endswith(".lean") and n.endswith(".lean"):
            bp, hp = B.package(o), H.package(n)
            if bp is not None and hp is not None and bp != hp:
                moves.update(dict.fromkeys(filter(None, (B.module(o), H.module(n))), (bp, hp)))
    report = {"base": base, "head": head, "map": dict(sorted(pmap.items())), "lean_modules": dict(sorted(lmods.items())),
              "renames": [], "edits": [], "refusals": [], "rules": {}}
    scoped = scoped_map(B, head, cwd, words)
    terms = Terms(B, *scoped, pmap, lmods) if scoped else None
    if scoped:
        report["words"] = scoped[0]

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

    roots = sorted(package_roots(B), key=len, reverse=True)

    def package_level(d: str) -> str | None:
        """Whether an empty `__init__.py` added in `d` is a package level the move needs: a module the range moved to `d` or
        below it is its package's there (`tools/move/layout.py`'s `module_there`: it was a package's at BASE, or outside
        every package and now outside a directory run by path), so it is importable by its new name only if each
        directory from it up to its package's root is a package (the mover writes these, and none above the root). It is
        "package" where one of those modules was a package's, "outside" where each came from outside every package."""
        by = None
        for o, n in renamed:
            if n.endswith(".py") and n.startswith(d + "/"):
                root = next((r for r in roots if n.startswith(r + "/")), None)
                if root is not None and (d == root or d.startswith(root + "/")) and _rename().L.module_there(o, n, root, B.set):
                    if posixpath.join(posixpath.dirname(o), "__init__.py") in B.set:
                        return "package"
                    by = "outside"
        return by

    def emptied(d: str) -> bool:
        """Whether the range's moves took the package at `d` away: files left it by a rename, and HEAD has none at or below
        it, so the `__init__.py` that marked it has nothing to mark (the mover deletes it if it holds nothing but comments
        and whitespace, and refuses the move otherwise)."""
        return not H.has_dir(d) and any(o.startswith(d + "/") for o, _ in renamed)

    went, created = dict(renamed), {e["new"] for e in entries if e["status"] == "A"}
    sides = collections.Counter((posixpath.dirname(o), posixpath.dirname(n)) for o, n in renamed)
    for e in entries:
        # git pairs identical files across directories at random: an `__init__.py` that went where no other file of its
        # directory did is one the range deleted, and the one git pairs it with is one the range added
        if (e["status"] == "R" and e["score"] == 100 and posixpath.basename(e["old"]) == posixpath.basename(e["new"]) == "__init__.py"
                and pmap.get(e["old"]) == e["new"] and sides[posixpath.dirname(e["old"]), posixpath.dirname(e["new"])] == 1):
            del went[e["old"]]
            created.add(e["new"])

    def now(f: str) -> str | None:
        """Where BASE's file `f` is at HEAD: where the range moved it, itself if it stayed, None if the range deleted it."""
        return went.get(f, f if f in H.set and f not in created else None)

    def cover(a: str, b: str, old: str, new: str) -> tuple[str, set[str]]:
        """`a` with each list of paths that `b` writes as a cover of it put as `b` has it, and the rules that took them.
        The lists are those `tools/move/layout.py` rewrites (its `lists`), paired in the file's order. A pair is a cover
        when an entry of the old list names what went to no one place (a directory whose files the range sent apart, or a
        file it deleted) and each entry of the new list is one of four. An old entry as the range sent it whole. An old
        entry as its row of the maps sends it (`Rows`: `[tokens]`, or this file's `[split]` row). A part: of HEAD's files
        that the range did not add it holds only ones the old list's paths held at BASE, one of them a file of an old
        entry that the new list no longer has, whose prefix and suffix (`HEAD:`, `/`, `/**`) it bears; an entry whose
        files went apart, which the list has as the range or its row sends it and every other file of it too, is one it
        no longer has, and that place is a part.
        A level: the `__init__.py` of a package directory above an entry the old list did not have, itself in a package,
        where the old list named a package's `__init__.py` that the range deleted with its package. And the new list
        still holds every file of each old entry it no longer has, one at least or a level. The list's other words (its
        comments', a command's, the entries that are no path) stay `a`'s, one for one, so a change to one is judged as
        any line's is; a list with more or fewer of them is no cover."""
        lists, entry = _rename().L.lists, _rename().L.ENTRY
        took: set[str] = set()

        def form(m: re.Match, p: str) -> tuple[str, str, str]:
            return m["pre"], p, m["post"] or ""
        la, lb = lists(old, a), lists(new, b)
        for (_, (a0, a1), ea), (_, (b0, b1), eb) in reversed(list(zip(la, lb))) if len(la) == len(lb) else ():
            px = [m for s, t in ea if (m := entry.fullmatch(a, s, t)) and (m["p"] in B.set or B.has_dir(m["p"]))]
            py = [m for s, t in eb if (m := entry.fullmatch(b, s, t)) and (m["p"] in H.set or H.has_dir(m["p"]))]
            wa, wb = between(a, (a0, a1), [m.span() for m in px]), between(b, (b0, b1), [m.span() for m in py])
            if len(wa) != len(wb):
                continue
            if all(now(f) == map_path(m["p"], pmap) + f[len(m["p"]):] for m in px for f in B.below(m["p"])):
                continue
            files = [{n for f in B.below(m["p"]) if (n := now(f)) is not None} for m in px]
            held, was, news = set().union(*files), {form(m, m["p"]) for m in px}, {form(y, y["p"]) for y in py}
            sent = [{form(m, map_path(m["p"], pmap)): "whole",
                     **({form(m, rows.parts[old, m["p"]]): "split"} if (old, m["p"]) in rows.parts else
                        {form(m, rows.tokens[m["p"]]): "tokens"} if m["p"] in rows.tokens else {})} for m in px]
            has = {f for y in py for f in H.below(y["p"])}
            # an old entry is gone unless the list has it as the range or its row sends it and not its other files
            gone = [not news & to.keys() or (fs <= has and any(now(f) != w + f[len(m["p"]):] for f in B.below(m["p"])))
                    for m, fs, to in zip(px, files, sent) for w in [map_path(m["p"], pmap)]]
            anew = [y["p"] for y in py if form(y, y["p"]) not in was]
            lost = any(x["p"].endswith("/__init__.py") and now(x["p"]) is None and emptied(posixpath.dirname(x["p"])) for x in px)

            def part(y: re.Match) -> bool:
                fs, below = {f for f in H.below(y["p"]) if f not in created}, H.has_dir(y["p"])
                return bool(fs) and fs <= held and any(fs & got and x["pre"] == y["pre"] and (y["post"] or "") == (
                    (x["post"] or "") if below else "") for x, got, g in zip(px, files, gone) if g)

            def level(y: re.Match) -> bool:
                d = posixpath.dirname(y["p"])
                return lost and y["p"].endswith("/__init__.py") and y["p"] in H.set and not y["post"] and any(
                    c != y["p"] and c.startswith(d + "/") and f"{posixpath.dirname(c)}/__init__.py" in H.set for c in anew)
            whole = {f for to in sent for f, k in to.items() if k == "whole"}
            by = {f: k for to in sent for f, k in to.items() if k != "whole"}
            kinds = ["whole" if (f := form(y, y["p"])) in whole else "level" if level(y) else "part" if part(y) else by.get(f)
                     for y in py]
            if None not in kinds and (any(gone) or "level" in kinds) and all(fs <= has for fs, g in zip(files, gone) if g):
                as_b = b[b0:b1]
                for (s, t), (u, v) in zip(reversed(wb), reversed(wa)):
                    as_b = as_b[:s - b0] + a[u:v] + as_b[t - b0:]
                a = a[:a0] + as_b + a[a1:]
                took |= {"cover", *kinds} - {"whole", "part", "level"}
        return a, took

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
        took: set[str] = set()                              # the maps' rows this path's edit needed (`Rows`)

        def by_rows(x: str, y: str, pair=None, resolved: bool = False, old=old, new=new) -> bool:
            took.update(r := rows.reads(x, y, pair, resolved, (old, new)))
            return bool(r)
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
        if st == "A" and name == "__init__.py" and text(e, 1) == "" and (by := package_level(posixpath.dirname(new))):
            report["edits"].append({"path": new, "kind": "package level"})
            if by == "outside":
                report["rules"].setdefault("level", []).append(new)
            continue
        if st == "D" and old in rows.drop:
            report["edits"].append({"path": old, "kind": "dropped"})
            report["rules"].setdefault("drop", []).append(old)
            continue
        if st == "D" and posixpath.basename(old) == "__init__.py" and emptied(posixpath.dirname(old)):
            if not _rename().L.only_comments(B.text(old)):
                refuse(e, "deleted: the `__init__.py` of a package the range emptied, which holds more than comments and "
                          "whitespace at BASE: give it a row in the map saying where it goes, or delete it in a hand commit "
                          "before the move")
                continue
            report["edits"].append({"path": old, "kind": "emptied package"})
            report["rules"].setdefault("emptied", []).append(old)
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
        if terms is not None and not same and not pins and k not in ("lock", "moves"):
            if why := terms.change(a, b, old, new):
                refuse(e, *why)
            else:
                report["edits"].append({"path": new, **({"old": old} if old != new else {}), "kind": "words"})
            continue
        if not same and not pins and k == kind(old) and k not in ("lock", "moves", "lean"):
            a, covered = cover(a, b, old, new)
            took |= covered
        at_base = a.splitlines()
        if not same and not pins and old != MOVES:
            a, unworded = worded(a, old, maps, lock=k == "lock"), a
            if not took and b in (a, merged(unworded, a)):
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
            h = first_unexplained(al, bl, substitute(a, pmap).splitlines(), bl,
                                  pair=lambda x, y: lean_import_rewrite(x, y, lmods, dotted),
                                  left=lambda i, y: partly(al[i], y, pmap) or by_rows(al[i], y))
            why = h and ("edit beyond import lines that name a module renamed in the range and moved paths", h)
        else:
            rel, py = name in RELATIVE_NAMES or new.endswith(".md"), new.endswith(".py")
            na_text = resolve_relative(a, old, pmap) if rel else join_chains(a) if py else a
            n0 = na_text.splitlines()
            na = n0 if same else substitute(na_text, pmap).splitlines()
            nb = (resolve_relative(b, new, {}) if rel else join_chains(b) if py else b).splitlines()

            def at(tree: Tree, path: str, text: str):
                return lambda mod: path if mod is None else module_file(tree, text, mod)

            sh = is_shell(old, a) and is_shell(new, b)
            sh_names = (sh_self_names(a), sh_self_names(b)) if sh else (set(), set())
            sh_here = _rename().L.sh_here_names(a) & _rename().L.sh_here_names(b) if sh else frozenset()

            def pair(x: str, y: str, at_old=at(B, old, a), at_new=at(H, new, b)) -> bool:  # noqa: B008  (this pair's, bound now)
                return any((py and (import_rewrite(z, y, modules, old, new) or anchor_rewrite(z, y, at_old, at_new, pmap)
                                    or ((by := anchor_rows(z, at_old, rows)) and anchor_rewrite(
                                        z, y, at_old, at_new, {**pmap, **by}) and not took.add("tokens"))))
                           or (sh and sh_anchor_rewrite(z, y, old, new, *sh_names, pmap))
                           or (sh_here and sh_anchor_rewrite(z, y, old, new, *sh_names, pmap, sh_here) and not took.add("anchor"))
                           or dotted_rewrite(z, y, modules) or registry_pair(new, z, y)
                           for z in dict.fromkeys((x, substitute(x, short))))

            def block(xs: list[str], ys: list[str]) -> bool:
                return import_block(xs, ys, modules, old, new) and not (
                    any(FROM_SLASH.match(z) for z in xs + ys) and took.add("continued"))
            binds = (places(a, old), places(b, new)) if py else ({}, {})
            under = json_pins().get(old, json_pins().get(new))

            def left(i: int, y: str) -> bool:
                return (partly(n0[i], y, pmap) or placed(n0[i], y, old, new, pmap, B, binds)
                        or (under is not None and manifest_pin(at_base[i], y, under, B, H, pmap))
                        or by_rows(resolve_relative(al[i], old, {}) if rel else n0[i], y, pair, rel))
            h = first_unexplained(al, bl, na, nb, pair=pair, added=lambda y: registered(new, y), block=block, left=left)
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
            for rule in sorted(took):
                report["rules"].setdefault(rule, []).append(new)
    for path, why in terms.conflicts() if terms else ():
        report["refusals"].append({"path": path, "status": "M", "reason": why, "hunk": None})

    def run_string(field: str, w: str, g: str) -> set[str]:
        """The rules by which a string of a `runs` entry may change beyond its package-relative paths: its `why` is prose,
        where each moved path is rewritten and each dotted name follows its moved Python module; any string is read as a
        line is (`Rows`: a generator's `{repo}/verity/x.py`)."""
        return {"why"} if field == "why" and dotted_rewrite(substitute(w, pmap), g, modules) else rows.reads(w, g)
    runs_moved(entries, B, H, pmap, lmods, report["refusals"], run_string, report["rules"])
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


def but_strings(want, got, take, field: str = "") -> bool:
    """Whether ``got`` is ``want`` but for strings `take(the top-level field they are in, old, new)` takes."""
    if isinstance(want, str) and isinstance(got, str):
        return want == got or take(field, want, got)
    if isinstance(want, list) and isinstance(got, list):
        return len(want) == len(got) and all(but_strings(w, g, take, field) for w, g in zip(want, got))
    if isinstance(want, dict) and isinstance(got, dict):
        return want.keys() == got.keys() and all(but_strings(want[k], got[k], take, field or k) for k in want)
    return want == got


def runs_moved(entries: list[dict], B: Tree, H: Tree, pmap: dict[str, str], lmods: dict[str, str], refusals: list,
               string=None, rules: dict | None = None) -> None:
    """Each `runs` entry of a lock the range touches must be in exactly one touched lock at HEAD, under its runner module's
    new name, with nothing changed but the paths the move rewrites: a run may move to the proofs package's lock, never vanish,
    be duplicated or change its data. A string that still differs is ``string``'s to take (the top-level field it is in,
    old, new, to the rules that take it), and each lock a rule took goes under the rule's name in ``rules``."""
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
                if (isinstance(want, dict) and isinstance(got, dict) and isinstance(w := want.get("why"), str)
                        and isinstance(g := got.get("why"), str) and partly(w, g, pmap)):
                    want = {**want, "why": g}           # its prose's moved paths from the root, rewritten or left
                took: set[str] = set()

                def take(field: str, w: str, g: str, took=took) -> bool:
                    took.update(r := string(field, w, g) if string else ())
                    return bool(r)
                if not but_strings(want, got, take):
                    refuse(hits[0], f"`runs` entry {name} changed beyond moved paths and its module's name",
                           field_hunk("runs", json.dumps({new_name: want}), json.dumps({new_name: got})))
                    took.clear()
                for rule in sorted(took) if rules is not None else ():
                    if hits[0] not in rules.setdefault(rule, []):
                        rules[rule].append(hits[0])
    for p, runs in head.items():
        for name in runs:
            if (p, name) not in claimed:
                refuse(p, f"`runs` entry {name} added: a move only moves runs from a lock it touches")


def summary(r: dict) -> str:
    span = f"{r['base']}..{r['head']}"
    edits = f"edits by {r['words']}'s words" if r.get("words") else "path-only edits"
    counts = f"{len(r['renames'])} renames, {len(r['edits'])} {edits}"
    if r["certified"]:
        return f"move-check: certified {span} is a pure move ({counts})"
    return f"move-check: refused {span}: {len(r['refusals'])} paths are not part of a pure move ({counts})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="move-check", description=__doc__.split("\n\n")[0])
    ap.add_argument("base")
    ap.add_argument("head")
    ap.add_argument("--map", action="append", default=[], metavar="OLD=NEW", help="an extra path (or dotted module) mapping")
    ap.add_argument("--json", action="store_true", help="print the report as JSON; the summary goes to stderr")
    ap.add_argument("--words", metavar="MAP", help="the word map, at BASE, a move with `scope = \"prose\"` renames by "
                    "(default: the `Move:` line of HEAD's message)")
    args = ap.parse_args(argv)
    extra = {}
    for m in args.map:
        old, sep, new = m.partition("=")
        if not (sep and old and new):
            ap.error(f"--map {m!r}: expected OLD=NEW")
        extra[old.strip("/")] = new.strip("/")
    try:
        r = certify(args.base, args.head, extra, words=args.words)
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
