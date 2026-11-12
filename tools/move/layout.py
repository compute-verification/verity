#!/usr/bin/env python3
"""The repository's layout move: move every file `layout_map.toml` names and rewrite every reference to it.

    python3 tools/move/layout.py              move and rewrite, then `uv lock` (the uv pod_setup.sh pins); prints what it changed
    python3 tools/move/layout.py --dry-run    print what it would change, and change nothing
    python3 tools/move/layout.py --modules    print the module renames that [paths] and [roots] imply ([modules] must equal it)
    python3 tools/move/layout.py --resolve M  the name module M has after the move (for readers of recorded data)

It is idempotent and reruns on any later main: a file still at an old path moves, a reference to an old path or module is
rewritten, and what is already moved is left alone. It refuses a dirty worktree, and it stops, with an exit of its own
(`NO_ROW`) and each file's name, when a tracked file under a path the map retires ([retired]) has no row, so the file gets
a row rather than a guess: in the map when it is main's, and when an open branch adds it, in the branch's rows file.

The rows file of `NAME.toml` is `NAME.rows.toml` beside it, a file the branch commits, with [paths] alone. It is read
after the map: a row there moves its file, or every file under its directory, and rewrites each reference to it, as a row
of the map does, and the module renames it implies are derived, not declared. The script deletes the rows file once it has
read it, so the moved tree holds none, and leaves its rows out of `moves.json`, which records the move's own. It refuses
(exit 1) a rows file that would change what the map says: a row for a path the map sends elsewhere or deletes, a directory
row over a path the map names, a module the map renames another way; and a row no tracked file is at or under.

A text file is one with no NUL in its first 8 KiB. One of more than `TEXT_LIMIT` bytes is not read, and the move stops and
names it rather than leave its lines as they were, until the map excludes it; a Lean lock is read whatever its size.

What it rewrites, in every tracked text file the map doesn't exclude:
- paths: root-relative ones (`packages/verity/src/verity/ir/cut.py`, also after `{ROOT}/`, `$ROOT/` or `~/verity/`, and
  after each prefix the map lists as standing for the root: [rewrite] root_prefixes, `/opt/vy-verifier/current/`),
  ones relative to the file (`../../packages/verity/lean` in a lakefile, `include_str!`), ones relative to an old package
  root (`verity/ir/cut.py`, `verity_pouw/schemes/ncp.py`) when that file exists, Python joins (`ROOT / "census"`), the
  `Path(__file__).parents[N]` anchors of a moved file, and a shell script's paths from its own directory
  (`$(dirname "$0")/../..`, also through `${BASH_SOURCE[0]}` or a variable the script sets to its own path or its own
  directory). A path below a directory the map splits that names no tracked file (a file a test makes up, a build's
  output) follows the directory's [tokens] row;
- lists of paths (a Python list, tuple or set of strings, a TOML or JSON array, the objects of `git rev-parse`, the
  members of `tar`): an entry that names what has no one place after the move, a directory the map splits or empties or
  a file it deletes, becomes the fewest directories and files that hold where the list's files went and no others; and
  where the list named the `__init__.py` of a package the move deletes, the package levels above those. A list of
  strings with an entry of one name that no path of the tree ends in holds names (suites, a command's words), and a
  directory's in it is one more: it follows the [tokens] row, which [split] must acknowledge;
- modules: import statements (split when their names now live in different modules, each statement keeping the comment
  that ended a one-line import, and one of them the lines a longer import went on over, as written; made absolute when a
  relative one no longer resolves), `from M import n` in strings, and every dotted name (`-m`, `import_module`,
  `module:attr`, docs). A module whose own row keeps its name (a [paths] row with old == new) stays where it is when its
  package moves;
- pyprojects: the distributions folded into `verity` ([distributions]), the workspace members, the suites' inputs.
A `lean-audit.json`'s `guarantees` and `reads` never change, and neither do the digests a path-keyed pin file records. An
old path that names something outside the tree ([rewrite] keep: the frozen evidence artifact's layout) is never rewritten,
and neither is a dotted name that only looks like a module ([rewrite] keep_dotted: a torch op's qualified name).
A string literal in Rust or Lean is protocol text (a domain tag, a statement's field, a JSON key) where only a path changes:
no dotted name and no lone quoted word in one is rewritten. In Python too a lone quoted word is a name (`"census"`: a key),
unless the map lists the directory under [rewrite] words (`"Security"`: no key and no run's directory has that name): then
its name alone between quotes or backticks is rewritten in every file but Rust and Lean; a name that is a key in some
file and the directory in another is listed for the files where it is the directory ([rewrite] words_in:
`{glob = [directories]}`). Where a key is made of another
word and a package's path (`"INLINE Security"`), [rewrite] word_ends lists the file (`{glob = [directories]}`): there the
name is rewritten also as the last word of a double-quoted string. A tracked file that a `.gitignore` matches stays tracked where
it moves. A `.py` file moved below a package root from a package, or from outside every package to a directory that is
not run by path (`tests/`, `lean/`), gets an `__init__.py` in each directory from it up to the root; and an `__init__.py`
left where a package held more and now holds only such files is deleted with the package. It appends [paths] to research's
`moves.json` (once), each row whose path the tree has, and never rewrites the moves recorded there.

It fails on a rewritten reference to a directory the map splits across destinations (`verity/ml/flock/` to
`verity/protocols/flock/prover/` with its `verifier/` to `verity/protocols/flock/agreement/`): the rewrite follows the
larger part, so a gate that named the
whole would no longer cover the part that left. [split] acknowledges each one it may let through, by file and directory,
with the reason; one that no longer occurs is an error too. Where the file means another part, the row names it
(`"old file: old directory" = { to = "verity/protocols/flock/agreement", why = "..." }`), and the file's references to
the directory go there.
"""
from __future__ import annotations

import argparse
import ast
import bisect
import fnmatch
import functools
import hashlib
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
_ROOT_JOIN = "\x00root\x00"
IDENT = re.compile(r"[A-Za-z_]\w*\Z")
#: the largest text file the rewrite reads, a Lean lock apart, which it reads whatever its size
TEXT_LIMIT = 4 << 20
LOCK = "lean-audit.json"
#: old paths that the move empties, when the map has no [retired]; a tracked file left under one has no row in the map
RETIRED = ("packages/", "protocols/", "census/", "backends/gkr/", "backends/direct/", "backends/ligero-verify/",
           "backends/ligerito-verify/", "backends/sp1/", "backends/vole/", "backends/redteam/", "backends/shared/",
           "backends/numerical/", "backends/README.md", "backends/AGENTS.md", "benchmarks/dot_product/",
           "benchmarks/ir_call/")
#: the exit when a tracked file under a retired path has no row: stderr then holds the fix in its first line and each
#: file on a line of its own, indented by two spaces (`restack.py` reads both)
NO_ROW = 3
#: what a map's rows file is named by, beside the map: `NAME.rows.toml` for `NAME.toml`
ROWS_SUFFIX = ".rows.toml"
#: a dotted name ending in one of these is a file name (`verity_pouw.json`), not a module
MOVES = "tools/research/src/research/store/moves.json"   # each layout move's [paths], for reading a tool from an older tree
FILE_SUFFIXES = {"py", "json", "toml", "md", "txt", "lean", "sh", "rs", "cu", "cuh", "yaml", "yml", "gz", "jsonl", "npy", "html"}
#: file types whose plain relative paths (`fixtures/x.json`, with no `./`) are resolved against the file's directory
RELATIVE_TYPES = {".py", ".toml", ".json", ".sh", ".md", ".rs", ".cu", ".cuh", ".yaml", ".yml", ".txt", ".cfg", ".lean"}
#: a distribution's directory as `uv.lock` writes it: a package's `source` and a `requires-dist` or manifest entry
_LOCK_SOURCE = re.compile(r'\b(editable|virtual|directory) = "([^"\n]+)"')
#: a shell script's own directory and the path joined to it (`$(dirname "$0")/../..`); `ref` is what `dirname` takes
_SH_DIRNAME = r'\$\(dirname\s+"(?P<ref>\$(?:\{[A-Za-z_0-9]\w*(?:\[0\])?\}|[A-Za-z_0-9]\w*))"\)'
_SH_REL = r"(?P<rel>(?:/[\w.\-]+)+)"
SH_ANCHOR = re.compile(rf"(?P<head>{_SH_DIRNAME}){_SH_REL}")
#: a variable a shell script sets to its own directory (`here=$(cd "$(dirname "$0")" && pwd)`)
SH_HERE = re.compile(r'(?:^|[;\s])(?P<v>[A-Za-z_]\w*)=(?P<q>"?)\$\(cd\s+"\$\(dirname\s+"(?:\$0|\$\{BASH_SOURCE\[0\]\})"\)"'
                     r"\s+&&\s+pwd(?:\s+-P)?\)(?P=q)(?=[;\s]|$)", re.M)
#: a variable a shell script sets to its own path (`SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")`)
SH_SELF = re.compile(r'^\s*(?:export\s+|readonly\s+|local\s+)?(?P<v>[A-Za-z_]\w*)=(?P<q>"?)'
                     r'(?:\$\(cd\s+"\$\(dirname\s+"(?P<a>\$0|\$\{BASH_SOURCE\[0\]\})"\)"\s+&&\s+pwd\)/\$\(basename\s+"(?P=a)"\)'
                     r'|\$\((?:realpath|readlink\s+-f)\s+"(?:\$0|\$\{BASH_SOURCE\[0\]\})"\))(?P=q)\s*$', re.M)
SH_OWN = {"0", "BASH_SOURCE", "BASH_SOURCE[0]"}
#: what stands before a shell variable's default (`${DEFS:-verity/x.json}`): the `-` there is no part of a longer word
SH_DEFAULT = re.compile(r"\$\{\w+:?-\Z")
#: what stands for the root before a path from it: what closes a placeholder or a command (`{ROOT}/`, `<tree>/`,
#: `$(pwd)/`), a shell variable, the home's checkout, a pod's. `tools/verity/check/move_check.py` reads a path as this does.
ROOT_PREFIX = r"[})>]/|\$\{?\w+\}?/|~/(?:verity/)?|/workspace/"


def root_prefix_ok(x) -> bool:
    """Whether a map may list `x` as standing for the root before a path ([rewrite] root_prefixes)."""
    return isinstance(x, str) and x.endswith("/") and x.strip("/") != ""


def root_prefix(more=()) -> str:
    """The pattern of what stands for the root before a path: `ROOT_PREFIX`, and each prefix of `more` as it is written,
    where no longer path holds it."""
    return "|".join([ROOT_PREFIX, *(rf"(?<![\w.\-/]){re.escape(x)}" for x in more)])
#: an entry of a list that is one path: a git revision before it (`HEAD:`), `/` or all below it (`/**`) after it
ENTRY = re.compile(r"(?P<pre>(?:HEAD:)?)(?P<p>[\w\-][\w.\-]*(?:/[\w.\-]+)*)(?P<post>/\*\*|/)?")
_STRING = r'"(?:[^"\\\n]|\\.)*"|\'[^\'\\\n]*\''
#: a TOML or JSON array of strings that is a key's value, with the comments between them
ARRAY = re.compile(rf"[=:]\s*(\[\s*(?:(?:#[^\n]*|(?:{_STRING})(?:\s*,)?)\s*)*\])")
#: the shell commands whose words are a list of paths: `git rev-parse`'s objects (`HEAD:path`) and `tar`'s members. Any
#: other command's word may be the one directory it takes (`cd`, a destination), so no list is read there.
SH_LIST = re.compile(r"(?<![\w.\-/])(?P<cmd>git\s+rev-parse|tar)(?=[ \t])(?P<args>[^|;&()\n]*)")
#: where `_covers` took an entry out of a list, until the list is closed up around it
GONE = "\ue002"
#: a pyproject's `members`: the workspace's distributions, each the directory its `pyproject.toml` is in
MEMBERS = re.compile(r"(^\s*members\s*=\s*\[)(?P<body>[^\]]*)(\])", re.M)
#: directories whose Python is run by path and is no module of the package around it
BY_PATH = {"tests", "lean"}


def is_shell(path: str, text: str) -> bool:
    """A shell script: `*.sh`, or a file whose first line is a shell's `#!`."""
    return path.endswith(".sh") or re.match(r"#!\S*(?:/|\s)(?:ba|da|z)?sh\b", text) is not None


def sh_ref(ref: str) -> str:
    """`$0`, `${BASH_SOURCE[0]}` or `$SELF` as the name it reads."""
    return ref[1:].strip("{}")


def sh_self_names(text: str) -> set[str]:
    """The names whose `dirname` is a shell script's own directory: its own path's, and each variable it sets to that."""
    return SH_OWN | {m.group("v") for m in SH_SELF.finditer(text)}


def sh_here_names(text: str) -> frozenset[str]:
    """The variables a shell script sets to its own directory (`SH_HERE`) and nowhere sets again."""
    return frozenset(v for m in SH_HERE.finditer(text) if len(re.findall(rf"(?:^|[;\s]){(v := m['v'])}=", text, re.M)) == 1)


@functools.cache
def sh_anchor(here: frozenset[str] = frozenset()) -> re.Pattern:
    """A path a shell script names from its own directory: after `$(dirname "$0")` (`SH_ANCHOR`, whose `ref` is the name
    read), or after a variable of `here` (`$here/../..`), which the script set to that directory."""
    if not here:
        return SH_ANCHOR
    names = "|".join(sorted(map(re.escape, here)))
    return re.compile(rf"(?P<head>{_SH_DIRNAME}|\$(?:\{{(?:{names})\}}|(?:{names})(?!\w))){_SH_REL}")


def git(root: Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


class NoRow(SystemExit):
    """Tracked files under a path the map retires that no row moves (`files`, every one; the message a caller that lets
    it through exits with names the first of them)."""

    def __init__(self, files: list[str]):
        super().__init__("tracked files under a retired path with no row in the map:\n  " + "\n  ".join(files[:50]))
        self.files = files


# ---------------------------------------------------------------------------------------------------------------- the map

class Layout:
    """The map: path moves (longest prefix wins), module renames (longest dotted prefix wins) and package roots."""

    def __init__(self, data: dict):
        self.data = data
        paths = data["paths"]
        self.files = {k: v for k, v in paths.items() if not k.endswith("/")}
        self.dirs = sorted(((k.rstrip("/"), v.rstrip("/")) for k, v in paths.items() if k.endswith("/")),
                           key=lambda kv: -len(kv[0]))
        self.tokens = {k.rstrip("/"): v.rstrip("/") for k, v in data.get("tokens", {}).items()}
        self.old_roots = data["roots"]["old"]
        self.new_roots = data["roots"]["new"]
        self.modules = dict(data.get("modules", {}))
        self.modules.update(data.get("aliases", {}))
        rw = data.get("rewrite", {})
        self.exclude, self.include = rw.get("exclude", []), rw.get("include", [])
        self.keep = rw.get("keep", [])
        self.keep_in = rw.get("keep_in", {})
        self.keep_dotted = tuple(rw.get("keep_dotted", []))
        #: moved directories whose name alone, between quotes or backticks, is ours wherever it is written ([rewrite] words)
        self.words = {w.rstrip("/") for w in rw.get("words", [])}
        #: per file glob, more such directories, each a name that is ours in those files and a key in others ([rewrite] words_in)
        self.words_in = {g: {w.rstrip("/") for w in ws} for g, ws in rw.get("words_in", {}).items()}
        #: per file glob, moved directories whose name is ours there also as the last word of a double-quoted string
        #: ([rewrite] word_ends: `"INLINE Security"`, a key made of another word and the package's path)
        self.word_ends = {g: {w.rstrip("/") for w in ws} for g, ws in rw.get("word_ends", {}).items()}
        self.merged = set(data.get("distributions", {}).get("merged", []))
        self.retired = tuple(data.get("retired", {}).get("paths", RETIRED))
        self.drop = set(data.get("drop", {}).get("paths", []))
        #: "old file: old directory" -> why the file's reference to a directory the map splits may follow the larger part
        self.split = dict(data.get("split", {}))
        #: of those rows, the ones that name the part the file means by the directory (`{ to = "new directory", why = "..." }`)
        self.split_to = {k: v["to"].rstrip("/") for k, v in self.split.items() if isinstance(v, dict)}
        #: what stands for the root before a path, beyond `ROOT_PREFIX` ([rewrite] root_prefixes: a pod's install directory)
        self.root_prefixes = rw.get("root_prefixes", [])
        if bad := [x for x in self.root_prefixes if not root_prefix_ok(x)]:
            raise SystemExit(f"[rewrite] root_prefixes: each is what a path is written after, ending in `/`: {bad}")

    # paths
    def file(self, p: str) -> str | None:
        """Where file `p` goes, or None when it stays."""
        if p in self.drop:
            return None
        if p in self.files:
            return self.files[p]
        for o, n in self.dirs:
            if p.startswith(o + "/"):
                return n + p[len(o):]
        return None

    def directory(self, d: str) -> tuple[str | None, bool]:
        """Where directory `d` goes, and whether part of it goes elsewhere (the rewrite is then approximate)."""
        d = d.rstrip("/")
        new = None
        for o, n in self.dirs:
            if d == o or d.startswith(o + "/"):
                new = n + d[len(o):]
                break
        if new is None and d in self.tokens:
            new = self.tokens[d]
        if new is None:
            return None, any(k.startswith(d + "/") for k in (*self.files, *(o for o, _ in self.dirs)))
        diverts = any(k.startswith(d + "/") and not (self.file(k) or "").startswith(new + "/")
                      for k in self.files) or any(o.startswith(d + "/") and not n.startswith(new + "/") and n != new
                                                  for o, n in self.dirs)
        return new, diverts

    def path(self, p: str, is_dir: bool) -> tuple[str | None, bool]:
        if not is_dir and (f := self.file(p)) is not None:
            return f, False
        return self.directory(p)

    def prefixes(self) -> list[str]:
        return sorted({*self.files, *(o for o, _ in self.dirs), *self.tokens}, key=len, reverse=True)

    # modules
    def module(self, name: str) -> str:
        parts = name.split(".")
        for i in range(len(parts), 0, -1):
            head = ".".join(parts[:i])
            if head in self.modules:
                return ".".join([self.modules[head], *parts[i:]])
        return name

    def renamed_prefix(self, name: str) -> int:
        """How many leading segments of `name` an entry renames (0: none)."""
        parts = name.split(".")
        for i in range(len(parts), 0, -1):
            if ".".join(parts[:i]) in self.modules:
                return i
        return 0

    def kept(self, p: str, where: str = "") -> bool:
        """Whether old path `p` is a key that outlives the tree (a path inside a store artifact), never rewritten; in file
        `where`, also a path that `keep_in` keeps for the files its glob matches (a run directory's own `census/`)."""
        p = p.strip("/")
        pats = [*self.keep, *(q for g, qs in self.keep_in.items() if where and fnmatch.fnmatchcase(where, g) for q in qs)]
        return any(fnmatch.fnmatchcase(p, pat) for pat in pats)

    def ends(self, word: str, where: str) -> bool:
        """Whether file `where` is one in which `word`, a moved directory's name, is ours as a string's last word too."""
        return any(word in ws and fnmatch.fnmatchcase(where, g) for g, ws in self.word_ends.items())

    def ours(self, word: str, where: str) -> bool:
        """Whether `word`, a moved directory's name alone between quotes, is the directory in file `where`."""
        return word in self.words or any(word in ws and fnmatch.fnmatchcase(where, g) for g, ws in self.words_in.items())

    def excluded(self, *paths: str) -> bool:
        def hit(pats, p):
            return any(fnmatch.fnmatchcase(p, pat) or (pat.startswith("**/") and fnmatch.fnmatchcase(p, pat[3:]))
                       for pat in pats)
        if MOVES in paths:              # each move's old paths, kept as they were: rewritten, a rerun would record it again
            return True
        if any(hit(self.include, p) for p in paths if p):
            return False
        return any(hit(self.exclude, p) for p in paths if p)


def module_name(path: str, roots: dict[str, str]) -> str | None:
    """The import name of the module or package at `path` (a `.py` file or a directory), under `roots` or else by its path."""
    p = path[:-3] if path.endswith(".py") else path.rstrip("/")
    if p.endswith("/__init__"):
        p = p[: -len("/__init__")]
    for name, d in sorted(roots.items(), key=lambda kv: -len(kv[1])):
        if p == d or p.startswith(d + "/"):
            rest = p[len(d):].strip("/")
            parts = [name, *(rest.split("/") if rest else [])]
            return ".".join(parts) if all(IDENT.match(x) for x in parts) else None
    parts = p.split("/")
    return ".".join(parts) if all(IDENT.match(x) for x in parts) else None


def module_there(old: str, new: str, root: str, tracked) -> bool:
    """Whether the `.py` file moved from `old` to `new`, below the package root `root`, is a module of that package: it was a
    package's (its directory had an `__init__.py`, among the `tracked` files), or it was outside every package (no
    `__init__.py` at or above its directory) and lands outside each directory run by path (`BY_PATH`)."""
    if posixpath.join(posixpath.dirname(old), "__init__.py") in tracked:
        return True
    return not any(f"{d}/__init__.py" in tracked for d in _ancestors(old)) and not BY_PATH & set(new[len(root) + 1:].split("/")[:-1])


def braces(token: str) -> list[str]:
    """Each path a brace list names: `a/{x,y/z}.py` is `a/x.py` and `a/y/z.py`; a token with no list is itself."""
    m = re.search(r"\{([^{}]*)\}", token)
    if not m:
        return [token]
    return [e for alt in m.group(1).split(",") for e in braces(token[:m.start()] + alt + token[m.end():])]


def parsed(text: str) -> ast.Module | None:
    """`text` as Python; None where it doesn't parse, and a reader then finds nothing in it."""
    try:
        return ast.parse(text)
    except (SyntaxError, ValueError):
        return None


def lists(path: str, text: str) -> list[tuple[str, tuple[int, int], list[tuple[int, int]]]]:
    """The lists `text` holds whose entries may each be a path, as (what parts one entry from the next, the list's span,
    each entry's span). In Python a list, tuple or set of nothing but string literals and `*names`; in TOML and JSON an
    array of strings; either only as a value of its own: one that is an entry of another, or whose entries an assignment
    binds to names one by one, is a record. In a shell script the words of a command that takes a list (`SH_LIST`:
    `HEAD:path` after `git rev-parse`; after `tar` a word that is no option's value)."""
    out = []
    if path.endswith(".py"):
        if (tree := parsed(text)) is None:
            return out
        lines, starts = text.splitlines(keepends=True), [0]
        for line in lines:
            starts.append(starts[-1] + len(line))

        def offset(lineno: int, col: int) -> int:
            return starts[lineno - 1] + len(lines[lineno - 1].encode("utf-8")[:col].decode("utf-8", "replace"))
        many = (ast.List, ast.Tuple, ast.Set)
        records = {id(e) for node in ast.walk(tree) if isinstance(node, many) for e in node.elts}
        records |= {id(node.value) for node in ast.walk(tree)
                    if isinstance(node, ast.Assign) and any(isinstance(t, many) for t in node.targets)}
        for node in ast.walk(tree):
            if not isinstance(node, many) or id(node) in records or not all(
                    isinstance(e, ast.Starred) or isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts):
                continue
            entries = []
            for e in node.elts:
                if isinstance(e, ast.Constant) and e.lineno == e.end_lineno:
                    a, b = offset(e.lineno, e.col_offset), offset(e.end_lineno, e.end_col_offset)
                    if text[a] in "\"'" and text[a + 1:b - 1] == e.value:
                        entries.append((a + 1, b - 1))
            if entries:
                out.append((", ", (offset(node.lineno, node.col_offset), offset(node.end_lineno, node.end_col_offset)), entries))
    elif path.endswith((".toml", ".json")):
        for m in ARRAY.finditer(text):
            at = m.start(1)
            out.append((", ", m.span(1), [(at + s.start() + 1, at + s.end() - 1)
                                          for s in re.finditer(rf"#[^\n]*|{_STRING}", m[1]) if s[0][0] != "#"]))
    elif is_shell(path, text):
        for m in SH_LIST.finditer(text):
            entries, before = [], ""
            for w in re.finditer(r"\S+", m["args"]):
                a, b = m.start("args") + w.start(), m.start("args") + w.end()
                if b - a > 1 and text[a] == '"' == text[b - 1]:
                    a, b = a + 1, b - 1
                rev = text.startswith("HEAD:", a)
                if rev if m["cmd"] != "tar" else not rev and (not before.startswith("-") or before == "-" or "=" in before):
                    entries.append((a, b))
                before = w[0]
            out.append((" ", m.span("args"), entries))
    return out


def derived_modules(layout: Layout) -> dict[str, str]:
    """The module renames that [paths] and [roots] imply: one per path entry whose old and new names differ."""
    out = {}
    for old, new in sorted(layout.data["paths"].items()):
        if not (old.endswith("/") or old.endswith(".py")):
            continue
        a, b = module_name(old, layout.old_roots), module_name(new, layout.new_roots)
        if a and b and (a != b or old == new):     # a row that keeps a directory keeps its modules' names too
            out[a] = b
        if old.endswith("/") and (r := next((n for n, d in layout.old_roots.items() if d == old.rstrip("/")), None)):
            if b and r != b:
                out[r] = b
    return out


def alternation(words) -> str:
    """A regex for any one of `words`, the longest that matches first, as `"|".join(map(re.escape, sorted(words, key=len,
    reverse=True)))` is. `re` tries that list word by word at every character of every file, which with a map of hundreds
    of rows is most of a run; this is the same set as a trie, one branch for each next character."""
    trie: dict = {}
    for w in words:
        node = trie
        for c in w:
            node = node.setdefault(c, {})
        node[""] = {}                               # a word ends here

    def regex(node: dict) -> str:
        longer = [re.escape(c) + regex(sub) for c, sub in node.items() if c]
        if not longer:
            return ""
        if "" in node:
            return "(?:" + "|".join(longer) + ")?"  # greedy: a longer word before the one that ends here
        return "(?:" + "|".join(longer) + ")" if len(longer) > 1 else longer[0]
    return regex(trie)


def oversize(src: Path) -> bool:
    """Whether `src` is left unread for its size alone: more than `TEXT_LIMIT` bytes of text (no NUL in its first 8 KiB) that
    is no Lean lock. A move never leaves such a file as it is without saying so: it stops and names it (`refuse_unread`)."""
    if src.name == LOCK or src.stat().st_size <= TEXT_LIMIT:
        return False
    with open(src, "rb") as f:
        return b"\0" not in f.read(8192)


def refuse_unread(unread: dict[str, int]):
    """Stop a move that would leave `unread` (path -> bytes, the text files `oversize` names) with every line as it was."""
    if unread:
        raise SystemExit(f"text files over {TEXT_LIMIT} bytes, which the move would leave unread, with every path and name in "
                         "them as it was:\n" + "".join(f"  {p} ({n} bytes)\n" for p, n in sorted(unread.items()))
                         + "exclude in the map one that names nothing the move changes, or raise TEXT_LIMIT in "
                         "tools/move/layout.py")


# ------------------------------------------------------------------------------------------------------------- the run

class Run:
    def __init__(self, root: Path, layout: Layout, dry: bool):
        self.root, self.L, self.dry = root, layout, dry
        self.changes: Counter = Counter()          # kind -> number of references rewritten
        self.files_changed: dict[str, Counter] = defaultdict(Counter)
        self.notes: dict[str, list[str]] = defaultdict(list)    # kind -> things for a person to look at
        self.moved: dict[str, str] = {}             # new path -> old path, for files this run moves
        self.texts: dict[str, str] = {}             # new path -> text as it will be written
        self.dropped: list[str] = []                # old paths of the files [drop] deletes
        self.leftover: list[str] = []               # the `__init__.py` files left alone in the packages the move empties
        self.splits: dict[str, str] = {}            # "old file: old directory" -> a rewrite of a split directory's reference
        self.unread: dict[str, int] = {}            # old path -> bytes, for the text files left unread for their size alone

    def note(self, kind: str, what: str):
        self.notes[kind].append(what)

    def split_reference(self, path: str, directory: str, what: str):
        """File `path` (new) refers to old `directory`, which the map splits, by a reference the rewrite changes (`what`)."""
        self.note("approximate directory paths", what)
        self.splits.setdefault(f"{self.old_of.get(path, path)}: {directory.strip('/')}", what)

    def _part(self, path: str, directory: str, new: str | None = None, approx: bool = False) -> tuple[str | None, bool]:
        """Where file `path`'s (new) reference to old `directory` goes, and whether [split] must acknowledge it: `new` and
        `approx` as the map's rows give them, unless the file's [split] row names the part the file means by the
        directory (`to`), a directory that a file of `directory` goes to or below."""
        key = f"{self.old_of.get(path, path)}: {directory}"
        to = self.L.split_to.get(key)
        if to is None:
            return new, approx
        if not any(self.new_of[f].startswith(to + "/") for f in self._under(directory)):
            raise SystemExit(f"[split] \"{key}\" names {to}, where no file of {directory} goes: name a directory one goes to or below")
        return to, True

    def unacknowledged_splits(self) -> tuple[list[str], list[str]]:
        """The split references [split] doesn't acknowledge, and the acknowledgements no reference needs any more."""
        return ([f"{k} ({w})" for k, w in sorted(self.splits.items()) if k not in self.L.split],
                sorted(k for k in self.L.split if k not in self.splits))

    # ---------------------------------------------------------------------------------------------- moves

    def plan(self) -> list[tuple[str, str]]:
        tracked = [p for p in git(self.root, "ls-files", "-z").split("\0") if p]
        self.tracked_old = set(tracked)
        try:                                        # fetched fixtures: files of the tree that git ignores
            fetched = {v for k, v in json.loads((self.root / "fixtures" / "artifacts.json").read_text()).items()
                       if k.startswith("art:") and isinstance(v, str)}
        except (OSError, ValueError):
            fetched = set()
        self.known_old = self.tracked_old | fetched
        self.binds = {}                             # old package -> names its __init__ binds to something not its submodule
        for p in tracked:
            if p.endswith("/__init__.py") and (pkg := module_name(p, self.L.old_roots)) and self.L.renamed_prefix(pkg):
                try:
                    body = ast.parse((self.root / p).read_text()).body
                except (OSError, SyntaxError, UnicodeDecodeError):
                    continue
                self.binds[pkg] = {a.asname or a.name for n in body if isinstance(n, ast.ImportFrom) and n.module
                                   for a in n.names} | {n.name for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        moves, left, self.dropped = [], [], []
        for p in tracked:
            if p in self.L.drop:
                self.dropped.append(p)
                continue
            new = self.L.file(p)
            if new is None:
                if p.startswith(self.L.retired) or p in self.L.retired:
                    left.append(p)
                continue
            moves.append((p, new))
        if left:
            raise NoRow(left)
        clash = Counter(n for _, n in moves)
        dup = [n for n, c in clash.items() if c > 1] + [n for _, n in moves if n in self.tracked_old and self.L.file(n) is None]
        if dup:
            raise SystemExit("two files would land at one path:\n  " + "\n  ".join(sorted(set(dup))[:50]))
        self.old_dirs = {posixpath.dirname(p) for p in tracked}
        self.old_dirs |= {d for p in (*tracked, *fetched) for d in _ancestors(p)}
        self.leftover = self._leftover(tracked, dict(moves))
        return moves

    def _leftover(self, tracked: list[str], dest: dict[str, str]) -> list[str]:
        """Each `__init__.py` that stays in a directory which held more and, after the move, holds nothing but such files,
        or below one: the package is gone, and a file that only marked it goes with it ([create] keeps one it names)."""
        before = sorted(tracked)
        after = sorted({dest.get(p, p) for p in tracked if p not in self.L.drop} | set(self.L.data.get("create", {})))

        def only_marks(files: list[str], d: str) -> bool:
            lo, hi = bisect.bisect_left(files, d + "/"), bisect.bisect_left(files, d + "0")
            return all(f.endswith("/__init__.py") for f in files[lo:hi])
        return [p for p in before if p.endswith("/__init__.py") and dest.get(p, p) == p and p not in self.L.drop
                and p not in self.L.data.get("create", {})
                and any(only_marks(after, d) and not only_marks(before, d) for d in _ancestors(p))]

    def move(self, moves: list[tuple[str, str]]):
        landing = {n for _, n in moves}
        aside = {o: f"{o}.moving" for o, _ in moves if o in landing and not self.dry}
        for old, tmp in aside.items():                      # a file another one lands on leaves first
            os.rename(self.root / old, self.root / tmp)
        for old, new in moves:
            self.moved[new] = old
            if self.dry:
                continue
            dst = self.root / new
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.rename(self.root / aside.get(old, old), dst)
        if not self.dry:
            for p in (*self.dropped, *self.leftover):
                (self.root / p).unlink()
        self.changes["files dropped"] = len(self.dropped)
        for p in self.leftover:
            self.note("`__init__.py` left alone in a package the move empties, deleted", p)
        if not self.dry:
            for d in sorted({posixpath.dirname(o) for o in (*(o for o, _ in moves), *self.dropped, *self.leftover)},
                            key=len, reverse=True):
                for a in [d, *_ancestors(d)]:
                    try:
                        (self.root / a).rmdir()
                    except OSError:
                        break
        self.changes["files moved"] = len(moves)

    def create(self):
        """[create] files, and an `__init__.py` in each new directory between a module moved into a package (`module_there`)
        and its root."""
        made = dict(self.L.data.get("create", {}))
        for new, old in self.moved.items():
            if not new.endswith(".py"):
                continue
            root = next((d for d in sorted(self.L.new_roots.values(), key=len, reverse=True)
                         if new.startswith(d + "/")), None)
            if root is None or not module_there(old, new, root, self.tracked_old):
                continue
            d = posixpath.dirname(new)
            while d and (d == root or d.startswith(root + "/")):
                made.setdefault(f"{d}/__init__.py", "")
                d = posixpath.dirname(d)
        given = self.L.data.get("create", {})
        for path, text in sorted(made.items()):
            if self._exists_new(path) and (path not in given or self._read(path) == text):
                continue
            self.changes["files created" if not self._exists_new(path) else "files written again from [create]"] += 1
            self.texts[path] = text
            self.moved.setdefault(path, path)
            if not self.dry:
                (self.root / path).parent.mkdir(parents=True, exist_ok=True)
                (self.root / path).write_text(text)

    def _exists_old(self, p: str) -> bool:
        return p in self.tracked_old

    def _exists_new(self, p: str) -> bool:
        return p in self.new_files or (not self.dry and (self.root / p).exists())

    # ------------------------------------------------------------------------------------------- the tree after

    def index(self, moves: list[tuple[str, str]]):
        dest = dict(moves)
        kept = self.tracked_old - {*self.dropped, *self.leftover}
        self.new_files = {dest.get(p, p) for p in kept}
        self.old_of = {dest.get(p, p): p for p in kept}
        self.new_of = {o: n for n, o in self.old_of.items()}
        self.kept = sorted(kept)
        self.holds = Counter(d for p in self.new_files for d in _ancestors(p))     # directory -> the files the moved tree has in it
        self.new_dirs = set(self.holds)
        mods = set()
        for p in self.new_files | set(self.L.data.get("create", {})):
            if p.endswith(".py"):
                if (m := module_name(p, self.L.new_roots)) is not None:
                    mods.add(m)
                    mods.update(".".join(m.split(".")[:i]) for i in range(1, len(m.split("."))))
        self.new_modules = mods

    def old_module_of(self, new_path: str) -> str | None:
        old = self.old_of.get(new_path, new_path)
        if old == new_path and (here := module_name(new_path, self.L.new_roots)) is not None and \
                any(new_path.startswith(d + "/") for d in self.L.new_roots.values()):
            return here                            # already at its new place (a rerun): its name doesn't change
        return module_name(old, self.L.old_roots)

    # ------------------------------------------------------------------------------------------------ rewriting

    def rewrite_all(self):
        lean_before = {}
        for p in sorted(self.new_files | set(self.texts)):
            if p.endswith("lean-audit.json"):
                lean_before[p] = self._lean_records(self._read(p))
        pins_before = {p: re.findall(r"\b[0-9a-f]{64}\b", self._read(p) or "")
                       for p in self.new_files if self.L.excluded(self.old_of.get(p, p)) is False
                       and any(fnmatch.fnmatchcase(p, pat) for pat in self.L.include)}
        created = set(self.L.data.get("create", {}))
        for p in sorted(self.new_files - created, key=lambda p: (p == "pyproject.toml", p)):   # the root's members read the rest
            if p == "pyproject.toml":
                self._dependents()
            old = self.old_of.get(p, p)
            if self.L.excluded(old, p):
                continue
            text = self._read(p)
            if text is None:
                continue
            new = self.rewrite(p, old, text)
            if new != text:
                self.texts[p] = new
        self.rekey_lean_deps()
        for p, before in lean_before.items():
            if self._lean_records(self.texts.get(p) or self._read(p)) != before:
                raise SystemExit(f"{p}: the rewrite would change a guarantee or reads record")
        registered = {v for k, v in json.loads(self._read("fixtures/artifacts.json") or "{}").items()
                      if k.startswith("art:") and isinstance(v, str)}
        for p in sorted(registered & set(self.texts)):
            if p in self.new_files:
                self.note("registered fixtures rewritten (`research data refresh-fixtures --preserve` re-registers them)", p)
        for p, before in pins_before.items():
            if re.findall(r"\b[0-9a-f]{64}\b", self.texts.get(p) or self._read(p) or "") != before:
                raise SystemExit(f"{p}: the rewrite would change a pinned digest")

    def rekey_lean_deps(self, pin: str = "tools/lean/lean-deps.json"):
        """Each pinned dependency bundle is keyed by its package's manifest and toolchain (`lean_audit.deps_key`), and a moved
        package's manifest names its path requirements' new directories. A bundle holds only the fetched packages, so when the
        manifest changed in nothing else the same bundle is pinned under the new key."""
        pin = self.L.file(pin) or pin
        text = self._read(pin)
        if text is None:
            return
        doc = json.loads(text)

        def key(manifest: str, toolchain: str) -> str:
            return hashlib.sha256(manifest.encode() + b"\0" + toolchain.encode()).hexdigest()

        def fetched(manifest: str) -> list:
            return [d for d in json.loads(manifest).get("packages", []) if d.get("type") != "path"]

        deps = {}
        for k, v in doc["deps"].items():
            pkg = v["package"]
            old_pkg = posixpath.dirname(self.old_of.get(f"{pkg}/lake-manifest.json", f"{pkg}/lake-manifest.json"))
            old_m = git(self.root, "show", f"HEAD:{old_pkg}/lake-manifest.json", check=False)
            new_m, tc = self._read(f"{pkg}/lake-manifest.json"), self._read(f"{pkg}/lean-toolchain")
            if not old_m or new_m is None or tc is None or key(old_m, tc) != k or fetched(old_m) != fetched(new_m):
                deps[k] = v
                if new_m is None or tc is None or key(new_m, tc) != k:
                    self.note("lean dependency bundles left under a stale key", f"{pkg} ({k[:12]})")
                continue
            deps[key(new_m, tc)] = v
            if key(new_m, tc) != k:
                self._count(pin, "lean dependency bundles rekeyed")
        new = json.dumps({**doc, "deps": deps}, indent=1, sort_keys=True) + "\n"
        if new != text:
            self.texts[pin] = new

    @staticmethod
    def _lean_records(text):
        d = json.loads(text) if text else {}
        return json.dumps([d.get("guarantees"), d.get("reads")], sort_keys=True)

    def _read(self, p: str) -> str | None:
        """The text of `p` as it will be written, or None for a file that isn't text (a NUL in its first 8 KiB, or not UTF-8)
        or is over `TEXT_LIMIT` and no Lean lock: one `oversize` names goes to `unread`, unless the map excludes it."""
        if p in self.texts:
            return self.texts[p]
        src = self.root / (p if not self.dry else self.old_of.get(p, p))
        try:
            if (size := src.stat().st_size) > TEXT_LIMIT and src.name != LOCK:
                if oversize(src) and not self.L.excluded(old := self.old_of.get(p, p), p):
                    self.unread[old] = size
                return None
            raw = src.read_bytes()
        except OSError:
            return None
        if b"\0" in raw[:8192]:
            return None
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def rewrite(self, path: str, old: str, text: str) -> str:
        out, covers = self._covers(path, text)
        out = self._rewrite(path, old, out)
        for mark, entries in covers.items():
            out = out.replace(mark, entries)
        return out

    def _rewrite(self, path: str, old: str, text: str) -> str:
        out = text
        if posixpath.basename(path) == ".gitignore":
            out = self._gitignore(path, out)
        else:
            out = self._paths(path, old, out)
        if path.endswith(".py"):
            out = self._anchors(path, old, out)
            out = self._joins(path, old, out)
            out = self._imports(path, old, out)
        if is_shell(path, out):
            out = self._sh_anchors(path, old, out)
        if not path.endswith(".lean"):
            out = self._from_strings(path, out)
            out = self._dotted(path, out)
        if posixpath.basename(path) == "pyproject.toml":
            out = self._pyproject(path, out)
        return out

    def _count(self, path: str, kind: str, n: int = 1):
        self.changes[kind] += n
        self.files_changed[path][kind] += n

    # covers -------------------------------------------------------------------------------------------------------

    def cover(self, olds) -> dict[str, str]:
        """For each of the files `olds` (old paths the move keeps), the directory or file of the moved tree that stands for
        it in their cover: the fewest directories and files that hold these files and no others. A directory stands for
        its files when every file the moved tree has in it is one of them (an `__init__.py` the move creates counts as
        none): the topmost such, named at its lowest level that holds them all."""
        news = {o: self.new_of[o] for o in olds}
        mine = Counter(d for n in news.values() for d in _ancestors(n))
        out = {}
        for o, n in news.items():
            up = list(_ancestors(n))
            top = next((d for d in reversed(up) if mine[d] == self.holds[d]), None)
            out[o] = n if top is None else next(d for d in up if self.holds[d] == self.holds[top])
        return out

    def _parts(self, d: str) -> bool:
        """Whether old path `d` stands for no one place after the move: a directory the map splits (a row says where it goes,
        and not every file of it keeps its place below there) or empties (no row, and the moved tree has no file there), or
        a file the move deletes."""
        if d not in self.old_dirs:
            return d in self.dropped or d in self.leftover
        new, approx = self.L.directory(d)
        if new is None:
            return approx and d not in self.new_dirs
        return any(self.new_of[f] != new + f[len(d):] for f in self._under(d))

    def _covers(self, path: str, text: str) -> tuple[str, dict[str, str]]:
        """In a list of paths (`lists`), an entry that names what stands for no one place after the move (`_parts`) becomes
        where its files went: the directories and files of the moved tree that hold the files the list's entries held, all
        of them together, and no others (`cover`). Each is written once, at the first such entry that held a file of it,
        with what the entry had around its path, and one that an entry the list keeps already holds is not written. The
        first entry that names a package's `__init__.py` the move deletes becomes the package levels above the entries
        the move wrote anew (`_levels`), those no entry holds: the list named the levels above what it holds, and an
        import still runs each. An entry left with nothing to write goes, and a line left empty goes with it. A directory
        with a row, and another entry of the list above or below it, is not read so: such a list names levels, and the
        row says which the directory is. Neither is a workspace's `members`, each one distribution's directory, which
        `_members` keeps, nor any list of a file under a `tests/` directory, where a list is as often the files of a
        stand-in tree as the tree's own and nothing in it says which. Nor is a list of strings (Python, TOML, JSON) with
        an entry of one name, no `/` in it, that no path of the tree ends in: its entries are names (suites, each named
        by its distribution or else by its directory; a command's words), a directory's among them is the name of one
        thing, and the other rewrites send it where the `[tokens]` row says, which `[split]` must acknowledge. A name
        some path ends in is a pattern for those paths (`lean-audit.json` in a gate), a path the tree no longer has is
        still a path, and a shell command of `SH_LIST` says itself that its words are paths: those lists are read as
        before. Returns the text with a mark where each rewritten
        entry was, and what each mark stands for, so that no later rewrite reads the new paths as old ones."""
        if not hasattr(self, "_parts_re"):
            names = [p for p in (*self.old_dirs, *self.dropped, *self.leftover) if self._parts(p)]
            self._parts_re = re.compile(alternation(names)) if names else None
            self._last_names = {posixpath.basename(p) for p in (*self.known_old, *self.old_dirs)}
        if self._parts_re is None or "tests" in path.split("/")[:-1] or not self._parts_re.search(text):
            return text, {}
        edits, marks = [], {}
        members = {m.start("body") - 1 for m in MEMBERS.finditer(text)} if posixpath.basename(path) == "pyproject.toml" else ()
        strings = path.endswith((".py", ".toml", ".json"))
        for sep, (start, _), spans in lists(path, text):
            if start in members:
                continue
            formed = [(a, b, m) for a, b in spans if (m := ENTRY.fullmatch(text, a, b))]
            if strings and any("/" not in m["p"] and m["p"] not in self._last_names for _, _, m in formed):
                continue
            named = [(a, b, m) for a, b, m in formed
                     if (m["p"] in self.old_dirs or m["p"] in self.tracked_old) and not self.L.kept(m["p"], path)]
            names = {m["p"] for _, _, m in named}
            # entries one above the other name levels, not the files below them: those are left to the other rewrites
            # where a row gives the directory a place, and one whose [split] row names the part the file means is that part
            todo = {p for p in names if self._parts(p) and not self._part(path, p)[0] and not (
                p in self.old_dirs and self.L.directory(p)[0] is not None and any(
                    q.startswith(p + "/") or p.startswith(q + "/") for q in names))}
            if not todo:
                continue
            files = {p: self._under(p) for p in names}
            cover = self.cover({f for fs in files.values() for f in fs})
            held = [self._part(path, p, *self.L.path(p, p in self.old_dirs))[0] or p for p in names - todo]

            def new(parts) -> list[str]:
                return [c for c in sorted(parts) if not any(c == h or c.startswith(h + "/") for h in held)]
            written = {}
            for a, b, m in named:
                if m["p"] in todo:
                    written[a] = new({cover[f] for f in files[m["p"]]})
                    held += written[a]
            if (lost := next((a for a, _, m in named if m["p"] in self.leftover), None)) is not None:
                written[lost] = new({level for c in held if c not in names for level in self._levels(c)})
            for a, b, m in named:
                if m["p"] not in todo:
                    continue
                parts = written[a]
                q = text[a - 1] if text[a - 1] in "\"'" and text[b:b + 1] == text[a - 1] else ""
                self._count(path, "list entries covered")
                self.note("list entries written as where their files went", f"{path}: {m[0]} -> {' '.join(parts) or 'nothing'}")
                if parts:
                    marks[mark := f"\ue000{len(marks)}\ue001"] = (q + sep + q).join(
                        m["pre"] + c + ((m["post"] or "") if c in self.new_dirs else "") for c in parts)
                    edits.append((a, b, mark))
                else:
                    edits.append((a - len(q), b + len(q), GONE))
        out = text
        for a, b, s in sorted(set(edits), reverse=True):
            out = out[:a] + s + out[b:]
        if GONE in out:
            if path.endswith(".json"):                  # JSON takes no comma before the bracket that closes a list
                while (less := re.sub(rf",\s*{GONE}(?=\s*\])", "", out)) != out:
                    out = less
            lines = []
            for line in out.split("\n"):
                if GONE in line:
                    for around in (rf"{GONE}[ \t]*,[ \t]*", rf"(?<=\S)[ \t]+{GONE}", rf"{GONE}[ \t]*"):
                        line = re.sub(around, "", line)     # with its comma; else with the space before it; else alone
                    line = line.rstrip()
                    if not line:
                        continue
                lines.append(line)
            out = "\n".join(lines)
        return out, marks

    def _levels(self, p: str) -> list[str]:
        """The `__init__.py` of each package directory above path `p` of the moved tree, where `p` is in a package."""
        def package(d: str) -> bool:
            return f"{d}/__init__.py" in self.new_files or f"{d}/__init__.py" in self.texts
        return [f"{d}/__init__.py" for d in _ancestors(p) if package(d)] if package(posixpath.dirname(p)) else []

    def _under(self, p: str) -> list[str]:
        """The files the move keeps that old path `p` names: itself, or those below it."""
        lo, hi = bisect.bisect_left(self.kept, p + "/"), bisect.bisect_left(self.kept, p + "0")
        return self.kept[lo:hi] or [f for f in (p,) if f in self.new_of]

    # paths --------------------------------------------------------------------------------------------------------

    def _paths(self, path: str, old: str, text: str) -> str:
        if not hasattr(self, "_path_re"):
            alts = alternation(self.L.prefixes())
            self._path_re = re.compile(rf"(?P<pre>{root_prefix(self.L.root_prefixes)})?(?P<p>{alts})(?![\w\-]|\.\w)"
                                       rf"(?P<rest>(?:/[\w.\-*{{}},]*)*)")
            self._rel_re = re.compile(r"(?<![\w.\-/$}])(?P<v>)(?P<p>(?:\.\.?/)+[\w.\-]+(?:/[\w.\-]+)*/?)"
                                      r"|(?P<v2>\$\{?\w+\}?/)(?P<p2>(?:\.\./)+[\w.\-]+(?:/[\w.\-]+)*/?)")
            roots = "|".join(re.escape(n) for n in sorted(self.L.old_roots, key=len, reverse=True))
            self._short_re = re.compile(rf"(?<![\w.\-/])(?P<p>(?:{roots})/[\w.\-]+(?:/[\w.\-]+)*/?)")
            self._plain_re = re.compile(r"(?<![\w.\-/$}])(?P<p>[\w\-][\w.\-]*(?:/[\w.\-]+)+/?)")

        def root_relative(m: re.Match) -> str:
            pre, p, rest = m.group("pre") or "", m.group("p"), m.group("rest")
            start = m.start("p") if not pre else m.start()
            before = text[start - 1] if start > 0 else ""
            if not pre and before and (before.isalnum() or before in "_.-/") and not SH_DEFAULT.search(text, 0, start):
                return m.group(0)
            token = p + rest
            trail = ""
            while token and (token[-1] in ".," or (token[-1] == "}" and token.count("}") > token.count("{"))):
                token, trail = token[:-1], token[-1] + trail    # a `}` no `{` of the token opens closes what holds the path
            if "/" not in token:
                after = text[m.end("p"):m.end("p") + 1]
                ours = self.L.ours(token, path)         # [rewrite] words: a name no key or run directory shares
                lone = before in ("\"'`" if ours else "\"'") and after == before
                last = before == " " and after == '"' and self.L.ends(token, path)      # [rewrite] word_ends
                if not last and (not lone or path.endswith((".rs", ".lean")) or path.endswith(".py") and not ours):
                    return m.group(0)                   # in code a lone quoted word is a name ("census": a key); joins are `_joins`'s
            new = self._map_token(token, path)
            if new is None or new == token:
                return m.group(0)
            self._count(path, "paths")
            if new.rstrip("/") == "." and pre.endswith("/"):
                return pre[:-1] + trail                 # `$REPO/packages/verity/src` is `$REPO` now
            return pre + new + trail

        text = self._path_re.sub(root_relative, text)

        def relative(m: re.Match) -> str:
            var, token = (m.group("v"), m.group("p")) if m.group("p") is not None else (m.group("v2"), m.group("p2"))
            new = self._map_relative(token, path, old, from_member=bool(var))
            if new is None or new == token:
                return m.group(0)
            self._count(path, "relative paths")
            self.note("relative paths rewritten", f"{path}: {var}{token} -> {var}{new}")
            return var + new

        text = self._rel_re.sub(relative, text)

        def short(m: re.Match) -> str:
            token = m.group("p")
            name, _, rest = token.partition("/")
            full = f"{self.L.old_roots[name]}/{rest}".rstrip("/")
            if self.L.kept(full, path) or not (full in self.known_old or full in self.old_dirs):
                return token
            new, approx = self._part(path, full, *self.L.path(full, full in self.old_dirs))
            if new is None:
                return token
            out = self._short_form(new)
            if out == token.rstrip("/"):
                return token
            if approx:
                self.split_reference(path, full, f"{path}: {token} -> {out}")
            self._count(path, "package-relative paths")
            return out + ("/" if token.endswith("/") else "")

        text = self._short_re.sub(short, text)
        if path != old and posixpath.splitext(path)[1] in RELATIVE_TYPES:
            def plain(m: re.Match) -> str:
                token = m.group("p")
                stripped = token.rstrip("/.,")
                first = stripped.split("/")[0]
                if first in {"packages", "protocols", "backends", "benchmarks", "census", "fixtures", "tools",
                             "integrations", "verity", "catalog", "kernels", "experimental", "archive", "infra"}:
                    return token
                if first == "tests" and self._member_of(old) is None:
                    return token                        # the root's tests/; a distribution's own is relative to it
                new = self._map_relative(stripped, path, old, plain=True)
                if new is None or new == stripped:
                    return token
                self._count(path, "relative paths")
                self.note("relative paths rewritten", f"{path}: {stripped} -> {new}")
                return new + token[len(stripped):]
            text = self._plain_re.sub(plain, text)
        return text

    def _short_form(self, new: str) -> str:
        """A new path as a package-relative one when it lands under a new root, else root-relative."""
        for name, d in sorted(self.L.new_roots.items(), key=lambda kv: -len(kv[1])):
            if new == d or new.startswith(d + "/"):
                return name + new[len(d):] if d != name else new
        return new

    def _map_token(self, token: str, path: str) -> str | None:
        slash = token.endswith("/")
        t = token.rstrip("/")
        if self.L.kept(t, path):
            return None
        if any(c in t for c in "*{},"):
            parts = t.split("/")
            i = next(i for i, x in enumerate(parts) if any(c in x for c in "*{},"))
            head, tail = "/".join(parts[:i]), "/".join(parts[i:])
            if not head:
                return None
            new, approx = self._part(path, head, *self.L.path(head, True))
            if "{" in tail and "*" not in t and (each := self._brace_list(t, tail, new, path)):
                return each + ("/" if slash else "")
            if new is None:
                if approx:
                    self.note("unmapped directory paths", f"{path}: {token}")
                return None
            if approx:
                self.split_reference(path, head, f"{path}: {token} -> {new}/{tail}")
            elif "{" in tail:
                self.note("approximate directory paths", f"{path}: {token} -> {new}/{tail}")
            return f"{new}/{tail}" + ("/" if slash else "")
        is_dir = t in self.old_dirs or slash or (t not in self.known_old and self.L.file(t) is None)
        new, approx = self._part(path, t, *self.L.path(t, is_dir))
        if new is None and (k := self._below_split(t)):
            new = self._part(path, k, self.L.tokens[k])[0] + t[len(k):]
            self.split_reference(path, k, f"{path}: {token} -> {new}")
            self.note("paths that name no tracked file, below a directory the map splits, sent where its [tokens] row goes",
                      f"{path}: {token} -> {new}")
            return new + ("/" if slash else "")
        if new is None:
            if approx:
                self.note("unmapped directory paths", f"{path}: {token}")
            return None
        if approx:
            self.split_reference(path, t, f"{path}: {token} -> {new}")
        return new + ("/" if slash else "")

    def _below_split(self, t: str) -> str | None:
        """The directory with a `[tokens]` row that `t` is a path below, where `t` names nothing the tree tracks and its
        first name below that directory is none the tree has there: no file or directory, and up to its first `.` no
        file's name before its suffix (`circuits/boolean/forms` is in `boolean.py`, which has its own row). What is left
        is a file of a test's own tree, or the start of a name a variable ends (`verity/ml/flock/flock-$p.patch`). Such a
        path follows the row, as a reference to the directory does."""
        k = next((k for k in sorted(self.L.tokens, key=len, reverse=True) if t.startswith(k + "/")), None)
        if k is None or t in self.known_old or t in self.old_dirs:
            return None
        first = f"{k}/{t[len(k) + 1:].split('/')[0]}"
        stem = f"{k}/{t[len(k) + 1:].split('/')[0].split('.')[0]}."
        at = bisect.bisect_left(self.kept, stem)
        if first in self.old_dirs or first in self.tracked_old or at < len(self.kept) and self.kept[at].startswith(stem):
            return None
        return k

    def _brace_list(self, t: str, tail: str, new: str | None, path: str) -> str | None:
        """Brace list `t` (`flock/{tool.py,verifier/tool.py}`) as a list of its paths, each mapped alone, where every one
        is in the old tree and they don't all go where the directory before the list goes (`new`); else None, and that
        directory's row carries the list as written: so does a pattern, one path of which names nothing. Each path is
        written whole, so a later map still finds one that this map leaves where it is."""
        olds = braces(t)
        if not all(o in self.known_old or o in self.old_dirs for o in olds):
            return None
        news = [self._map_token(o, path) or o for o in olds]
        if news == braces(f"{new}/{tail}" if new is not None else t):
            return None
        out = "{" + ",".join(news) + "}"
        self.note("brace lists split", f"{path}: {t} -> {out}")
        return out

    def _map_relative(self, token: str, path: str, old: str, plain: bool = False, from_member: bool = False) -> str | None:
        slash = token.endswith("/")
        t = token.rstrip("/")
        dot = t.endswith("/.") and t != "."
        if dot:
            t = t[:-2]                                  # `v1/.` (copy the contents): normpath would drop it
        base_new = posixpath.dirname(path)
        target = posixpath.normpath(posixpath.join(posixpath.dirname(old), t))
        if from_member:
            target = ".."                               # `$SP1_SRC/../..`: a variable names a directory, not this file's
        if not (target in self.known_old or target in self.old_dirs) and (member := self._member_of(old)):
            # relative to the distribution's directory, where its commands run (`../../packages/verity/src` in vLLM's pod
            # scripts and `cwd=` calls)
            alt = posixpath.normpath(posixpath.join(member, t))
            if alt in self.known_old or alt in self.old_dirs:
                target, base_new = alt, self.L.directory(member)[0] or member
        if target.startswith("..") or target == "." or self.L.kept(target, path):
            return None
        is_dir = target in self.old_dirs
        if not (target in self.known_old or is_dir):
            return None
        if is_dir:
            new_target, approx = self._part(path, target, *self.L.directory(target))
            if new_target is None:
                new_target = target            # it stays, though part of it may move
                if approx and path != old:
                    self.split_reference(path, target, f"{path}: {token} (part of {target} moves)")
            elif approx:
                self.split_reference(path, target, f"{path}: {token} -> {new_target}")
        else:
            new_target = self.L.file(target) or target
        if path == old and new_target == target and not self._through_moved(posixpath.dirname(old), t):
            return None                                 # neither end moved: the token stays as written
        rel = posixpath.relpath(new_target, base_new or ".")
        if t.startswith("./") and not rel.startswith("."):
            rel = "./" + rel
        if plain and rel.startswith(".."):
            self.note("relative paths that now leave their directory", f"{path}: {t} -> {rel}")
        return rel + ("/." if dot else "") + ("/" if slash else "")

    def _through_moved(self, base: str, t: str) -> bool:
        """Whether relative path `t`, written from `base`, climbs back out of a directory the move takes away (a Lake
        manifest's inherited `lms/lean/../../../../../core/service/flock`): as written it would stop resolving, so it is
        rewritten to its normal form even when neither end moved."""
        segs, cur = t.split("/"), base
        for i, seg in enumerate(segs[:-1]):
            cur = posixpath.normpath(posixpath.join(cur, seg))
            if seg != ".." and ".." in segs[i + 1:] and self.L.directory(cur)[0] not in (None, cur):
                return True
        return False

    def _member_of(self, old: str) -> str | None:
        """The directory of the distribution (a pyproject.toml below the root) that file `old` belongs to."""
        d = posixpath.dirname(old)
        while d:
            if f"{d}/pyproject.toml" in self.tracked_old:
                return d
            d = posixpath.dirname(d)
        return None

    def _gitignore(self, path: str, text: str) -> str:
        if posixpath.dirname(path):
            return text
        out = []
        for line in text.splitlines(keepends=True):
            body = line.rstrip("\n")
            neg = body.startswith("!")
            core = body[1:] if neg else body
            lead = "/" if core.startswith("/") else ""
            pat = core[len(lead):]
            if pat and not pat.startswith("#") and "/" in pat.rstrip("/"):
                new = self._map_token(pat, path)
                if new and new != pat:
                    self._count(path, "paths")
                    line = ("!" if neg else "") + lead + new + ("\n" if line.endswith("\n") else "")
            out.append(line)
        return "".join(out)

    # Python anchors and joins -----------------------------------------------------------------------------------------

    _ANCHOR = re.compile(r"(?P<head>Path\((?:__file__|(?P<mod>[A-Za-z_][\w.]*)\.__file__)\))(?P<r>\.resolve\(\)|\.absolute\(\))?"
                         r"(?:\.parents\[(?P<n>\d+)\]|(?P<p>(?:\.parent)+))(?P<chain>(?:\s*/\s*(?P<q>[\"'])[\w.\-]+(?P=q))*)")

    def _module_file(self, name: str) -> str | None:
        """The old path of module `name`'s file, when it is a file of this tree."""
        for root, d in sorted(self.L.old_roots.items(), key=lambda kv: -len(kv[1])):
            if name == root or name.startswith(root + "."):
                base = d + name[len(root):].replace(".", "/")
                break
        else:
            base = name.replace(".", "/")
        return next((f for f in (base + ".py", base + "/__init__.py") if f in self.tracked_old), None)

    def _aliases(self, old: str, text: str) -> dict[str, str]:
        """The names this file binds to modules by importing them: {local name: old module name}."""
        if (tree := parsed(text)) is None:
            return {}
        me, pkg, out = module_name(old, self.L.old_roots), old.endswith("__init__.py"), {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    out[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
            elif isinstance(node, ast.ImportFrom):
                base = self._resolve_relative(me, pkg, node.level, node.module) if node.level else node.module
                for a in node.names:
                    if base and self._module_file(f"{base}.{a.name}"):
                        out[a.asname or a.name] = f"{base}.{a.name}"
        return out

    def _anchors(self, path: str, old: str, text: str) -> str:
        """`Path(__file__).parents[N] / "a" / "b"` in a file that moved, or `Path(m.__file__)...` for a module m that moved:
        the path it names, from the anchor file's new place."""
        aliases = None

        def fix(m: re.Match) -> str:
            nonlocal aliases
            if m.group("mod"):
                if aliases is None:
                    aliases = self._aliases(old, text)
                name = aliases.get(m.group("mod"))
                file_old = self._module_file(name) if name else None
                if file_old is None:
                    return m.group(0)
                return self._anchor(m, path, self.L.file(file_old) or file_old, file_old)
            return self._anchor(m, path, path, old)

        return self._ANCHOR.sub(fix, text)

    def _anchor(self, m: re.Match, where: str, path: str, old: str) -> str:
        """Anchor `m` (in file `where`) at the file `old`, which is at `path` after the move."""
        # how many directories the anchor climbs above the file's own: parents[n] climbs n, .parent x k climbs k - 1
        up = int(m.group("n")) if m.group("n") is not None else m.group("p").count(".parent") - 1
        old_dirs = old.split("/")[:-1]
        if up > len(old_dirs):
            return m.group(0)
        anchor = "/".join(old_dirs[: len(old_dirs) - up])
        segs = re.findall(r"""["']([\w.\-]+)["']""", m.group("chain"))
        target = posixpath.normpath(posixpath.join(anchor, *segs)) if segs else anchor
        target = "" if target == "." else target
        if target and self.L.kept(target, where):
            return m.group(0)
        if target and (target in self.known_old or target in self.old_dirs):
            new_target = self.L.file(target) if target in self.known_old else None
            if new_target is None:
                new_target, approx = self.L.directory(target)
                if new_target is None:
                    new_target = target
                    if approx and segs:
                        self.note("anchors to a directory that splits", f"{where}: {m.group(0)} (was {target})")
        elif not segs:
            new_target = self.L.directory(anchor)[0] or anchor if anchor else ""
        else:
            return self._anchor_only(m, where, path, old, anchor, up)
        if path == old and new_target == target:
            return m.group(0)
        new_dirs = path.split("/")[:-1]
        tparts = new_target.split("/") if new_target else []
        common = 0
        while common < min(len(new_dirs), len(tparts)) and new_dirs[common] == tparts[common]:
            common += 1
        new_up = len(new_dirs) - common
        rest = tparts[common:]
        r = m.group("r") or ""
        base = f"{m.group('head')}{r}.parents[{new_up}]" if m.group("n") is not None else \
            f"{m.group('head')}{r}" + ".parent" * (new_up + 1)
        q = m.group("q") or '"'
        out = base + "".join(f" / {q}{s}{q}" for s in rest)
        if not segs and rest:
            out = f"({out})"                        # the file left the directory it anchored at: join its way back
        if out == m.group(0) or (new_up == up and rest == segs):
            return m.group(0)
        self._count(where, "anchors")
        return out

    def _anchor_only(self, m: re.Match, where: str, path: str, old: str, anchor: str, up: int) -> str:
        """An anchor whose joined path isn't in the tree: keep the directory it names."""
        if path == old:
            return m.group(0)
        new_anchor = (self.L.directory(anchor)[0] or anchor) if anchor else ""
        new_dirs = path.split("/")[:-1]
        if new_anchor and not path.startswith(new_anchor + "/"):
            self.note("anchors not recomputed", f"{where}: {m.group(0)} (was {anchor or 'the root'})")
            return m.group(0)
        new_up = len(new_dirs) - (len(new_anchor.split("/")) if new_anchor else 0)
        if new_up == up:
            return m.group(0)
        self._count(where, "anchors")
        r = m.group("r") or ""
        base = f"{m.group('head')}{r}.parents[{new_up}]" if m.group("n") is not None else \
            f"{m.group('head')}{r}" + ".parent" * (new_up + 1)
        return base + m.group("chain")

    def _sh_anchors(self, path: str, old: str, text: str) -> str:
        """`$(dirname "$0")/../..` in a shell script, or after `${BASH_SOURCE[0]}` or a variable the script sets to its own
        path (`SH_SELF`), and `$here/../..` after a variable it sets to its own directory (`SH_HERE`): the path it names
        from the script's directory, from the script's new place."""
        names = sh_self_names(text)
        here_old, here_new = posixpath.dirname(old), posixpath.dirname(path)

        def fix(m: re.Match) -> str:
            if m.group("ref") is not None and sh_ref(m.group("ref")) not in names:
                return m.group(0)
            rel = m.group("rel")[1:]
            target = posixpath.normpath(posixpath.join(here_old, rel))
            if target.startswith(".."):
                return m.group(0)
            target = "" if target == "." else target
            if not target:
                new_target = ""
            elif self.L.kept(target, path):
                return m.group(0)
            elif target in self.known_old:
                new_target = self.L.file(target) or target
            elif target in self.old_dirs:
                new_target, approx = self.L.directory(target)
                if approx:
                    self.note("shell anchors to a directory that splits", f"{path}: {m.group(0)} (was {target})")
                new_target = new_target or target
            else:
                if path != old and (m.group("ref") is not None or rel.startswith("../")):   # `$here/out.json` goes with the script
                    self.note("shell anchors not recomputed", f"{path}: {m.group(0)} (names {target}, not in the tree)")
                return m.group(0)
            if path == old and new_target == target:
                return m.group(0)
            new_rel = posixpath.relpath(new_target or ".", here_new or ".")
            if new_rel == rel:
                return m.group(0)
            self._count(path, "shell anchors")
            return f"{m.group('head')}/{new_rel}"

        return sh_anchor(sh_here_names(text)).sub(fix, text)

    _JOIN = re.compile(r"""(?P<chain>(?:(?P<q>["'])[\w.\-]+(?P=q)\s*/\s*)*(?P<q2>["'])[\w.\-]+(?P=q2))""")

    def _joins(self, path: str, old: str, text: str) -> str:
        """`ROOT / "packages" / "verity" / "lean"`: a chain of string segments joined by `/` that names an old path."""
        def fix(m: re.Match) -> str:
            chain = m.group("chain")
            before = text[: m.start()].rstrip()
            if not before.endswith("/") or before.endswith("//"):
                return chain
            if re.search(r"Path\((?:__file__|[A-Za-z_][\w.]*\.__file__)\)(?:\.resolve\(\)|\.absolute\(\))?"
                         r"(?:\.parents\[\d+\]|(?:\.parent)+)?\s*$", before[:-1].rstrip()):
                return chain                            # an anchor's own chain, which `_anchors` resolves from the file
            segs = re.findall(r"""["']([\w.\-]+)["']""", chain)
            joined = "/".join(segs)
            if len(segs) == 1:
                return chain                            # `run / "census"`: one word names a run's own directory as often as ours
            if not (joined in self.known_old or joined in self.old_dirs or joined in self.L.files
                    or joined in dict(self.L.dirs) or joined in self.L.tokens or self._below_split(joined)):
                return chain
            new = self._map_token(joined, path)
            if new is None or new == joined:
                return chain
            q = m.group("q2")
            self._count(path, "path joins")
            self.note("path joins rewritten", f"{path}: {joined} -> {new}")
            if new.strip("/") == ".":
                return _ROOT_JOIN                       # the repository root: the join itself goes
            return " / ".join(f"{q}{s}{q}" for s in new.strip("/").split("/"))

        def fix_path(m: re.Match) -> str:
            segs = [m.group("first"), *re.findall(r"""["']([\w.\-]+)["']""", m.group("rest"))]
            joined = "/".join(segs)
            if len(segs) == 1 or not (joined in self.known_old or joined in self.old_dirs or joined in self.L.files
                                      or joined in dict(self.L.dirs) or joined in self.L.tokens or self._below_split(joined)):
                return m.group(0)
            new = self._map_token(joined, path)
            if new is None or new == joined or new.strip("/") == ".":
                return m.group(0)
            q = m.group("q")
            self._count(path, "path joins")
            self.note("path joins rewritten", f"{path}: {joined} -> {new}")
            first, *rest = new.strip("/").split("/")
            return f"{m.group('head')}{q}{first}{q})" + "".join(f" / {q}{s}{q}" for s in rest)

        text = self._PATH_JOIN.sub(fix_path, text)
        return re.sub(r"\s*/\s*" + _ROOT_JOIN, "", self._JOIN.sub(fix, text))

    #: `Path("packages") / "verity" / "src"`: a join whose first segment is a Path of one string
    _PATH_JOIN = re.compile(r"""(?P<head>(?<![\w.])(?:Pure(?:Posix)?)?Path\()(?P<q>["'])(?P<first>[\w.\-]+)(?P=q)\)"""
                            r"""(?P<rest>(?:\s*/\s*(?P<q3>["'])[\w.\-]+(?P=q3))*)""")

    # imports ----------------------------------------------------------------------------------------------------------

    def _imports(self, path: str, old: str, text: str) -> str:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            if any(n in text for n in self.L.modules):
                self.note("files that don't parse (imports left to the dotted rewrite)", path)
            return text
        lines = text.splitlines(keepends=True)
        starts = [0]
        for line in lines:
            starts.append(starts[-1] + len(line))

        def offset(lineno: int, col: int) -> int:
            line = lines[lineno - 1]
            return starts[lineno - 1] + len(line.encode("utf-8")[:col].decode("utf-8", "replace"))

        me_old = self.old_module_of(path)
        me_new = module_name(path, self.L.new_roots)
        pkg_old, pkg_new = old.endswith("__init__.py"), path.endswith("__init__.py")
        edits = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                rep = self._import_from(node, me_old, me_new, (pkg_old, pkg_new), path)
            elif isinstance(node, ast.Import):
                rep = self._import(node)
            else:
                continue
            if rep is None:
                continue
            a, b = offset(node.lineno, node.col_offset), offset(node.end_lineno, node.end_col_offset)
            src = text[a:b]
            if isinstance(rep, tuple):           # only the module token changes: keep the statement's own formatting
                old_mod, new_mod = rep
                new_src = re.sub(r"(\bfrom\s+)" + re.escape(old_mod) + r"(\s*import\b)",
                                 lambda mm: mm.group(1) + new_mod + mm.group(2), src, count=1) \
                    if isinstance(node, ast.ImportFrom) else _replace_names(src, old_mod, new_mod)
            else:
                indent = " " * node.col_offset
                # each statement a one-line import splits into keeps the comment that ended the line (a linter's `noqa`), which
                # the last already has; the lines an import went on over stay with the statement that has their names
                comment = re.match(r"[ \t]*#[^\n]*", text[b:]) if "\n" not in src else None
                parts = _continued(src, rep, indent) or [_wrap(s, indent) for s in rep]
                new_src = ((comment[0] if comment else "") + "\n" + indent).join(parts)
            if new_src != src:
                edits.append((a, b, new_src))
        out = text
        for a, b, s in sorted(edits, reverse=True):
            out = out[:a] + s + out[b:]
        if edits:
            try:
                ast.parse(out)
            except SyntaxError as e:
                self.note("imports left as they were (the rewrite didn't parse)", f"{path}: {e}")
                return text
            self._count(path, "imports", len(edits))
        return out

    def _resolve_relative(self, me: str | None, is_pkg: bool, level: int, module: str | None) -> str | None:
        if me is None:
            return None
        parts = me.split(".")
        if not is_pkg:
            parts = parts[:-1]
        if level - 1 > len(parts):
            return None
        base = parts[: len(parts) - (level - 1)]
        return ".".join([*base, *([module] if module else [])]) or None

    def _import_from(self, node: ast.ImportFrom, me_old, me_new, pkg: tuple[bool, bool], path: str):
        if node.level:
            m_old = self._resolve_relative(me_old, pkg[0], node.level, node.module)
            if m_old is None:
                return None
        else:
            m_old = node.module
        groups: dict[str, list[str]] = {}
        for a in node.names:
            if a.name == "*":
                groups.setdefault(self.L.module(m_old), []).append("*")
                continue
            full = f"{m_old}.{a.name}"
            nf = self.L.module(full)
            if (nf != full or full in self.L.modules) and nf in self.new_modules and a.name not in self.binds.get(m_old, ()):
                parent, _, leaf = nf.rpartition(".")
                asname = a.asname or (a.name if leaf != a.name else None)
                if not parent:
                    self.note("imports to check", f"{path}: from {m_old} import {a.name} (now top-level {nf})")
                    return None
                groups.setdefault(parent, []).append(leaf + (f" as {asname}" if asname else ""))
            else:
                if self.L.renamed_prefix(m_old) and self.L.module(m_old) not in self.new_modules:
                    self.note("imports to check", f"{path}: from {m_old} import {a.name}")
                groups.setdefault(self.L.module(m_old), []).append(a.name + (f" as {a.asname}" if a.asname else ""))
        if node.level:
            here = self._resolve_relative(me_new, pkg[1], node.level, node.module)
            names = [a.name + (f" as {a.asname}" if a.asname else "") for a in node.names]
            if list(groups) == [here] and groups[here] == names:
                return None
            if len(groups) == 1 and next(iter(groups.values())) == names:
                return ("." * node.level + (node.module or ""), next(iter(groups)))
            return [f"from {mod} import {', '.join(ns)}" for mod, ns in groups.items()]
        names = [a.name + (f" as {a.asname}" if a.asname else "") for a in node.names]
        if list(groups) == [m_old] and groups[m_old] == names:
            return None
        if len(groups) == 1 and next(iter(groups.values())) == names:
            return (m_old, next(iter(groups)))
        return [f"from {mod} import {', '.join(ns)}" for mod, ns in groups.items()]

    def _import(self, node: ast.Import):
        changed = [(a.name, self.L.module(a.name)) for a in node.names if self.L.module(a.name) != a.name]
        if not changed:
            return None
        if len(changed) == 1:
            return changed[0]
        return [("import " + ", ".join(self.L.module(a.name) + (f" as {a.asname}" if a.asname else "")
                                      for a in node.names))]

    # module strings ---------------------------------------------------------------------------------------------------

    _FROM_STR = re.compile(r"\bfrom\s+(?P<m>[A-Za-z_][\w.]*)\s+import\s+(?P<n>[A-Za-z_]\w*(?:\s+as\s+\w+)?(?:\s*,\s*[A-Za-z_]\w*(?:\s+as\s+\w+)?)*)")

    def _from_strings(self, path: str, text: str) -> str:
        """`from M import n` written in a string or a document: each name to the module it now lives in."""
        def fix(m: re.Match) -> str:
            mod = m.group("m")
            items = [x.strip() for x in m.group("n").split(",")]
            if not any(self.L.renamed_prefix(n) for n in (mod, *(f"{mod}.{it.partition(' as ')[0].strip()}" for it in items))):
                return m.group(0)
            groups: dict[str, list[str]] = {}
            for it in items:
                name, _, alias = it.partition(" as ")
                name, alias = name.strip(), alias.strip()
                full = self.L.module(f"{mod}.{name}")
                if (full != f"{mod}.{name}" or f"{mod}.{name}" in self.L.modules) and full in self.new_modules:
                    parent, _, leaf = full.rpartition(".")
                    asn = alias or (name if leaf != name else "")
                    groups.setdefault(parent, []).append(leaf + (f" as {asn}" if asn else ""))
                else:
                    groups.setdefault(self.L.module(mod), []).append(it)
            if len(groups) == 1:
                (newmod, ns), = groups.items()
                if ns == items:
                    return m.group(0).replace(mod, newmod, 1)
            line_start = text.rfind("\n", 0, m.start()) + 1
            indent = re.match(r"\s*", text[line_start:m.start()]).group(0)
            sep = "\n" + indent if text[line_start:m.start()].strip() == "" else "; "
            self._count(path, "imports in strings")
            return sep.join(f"from {k} import {', '.join(v)}" for k, v in groups.items())

        return self._FROM_STR.sub(fix, text)

    def _dotted(self, path: str, text: str) -> str:
        if not hasattr(self, "_dot_re"):
            self._dot_re = re.compile(r"(?<![\w.$@/\-])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)(?![\w/\-])")
            singles = [n for n in self.L.modules if "." not in n and n.startswith("verity_")]
            # `verity_pouw::ncp_linear` is a torch op's namespace, not the module, and a namespace holds no dot
            self._single_re = re.compile(r"(?<![\w.$@/\-])(" + "|".join(map(re.escape, singles)) + r")(?![\w/\-]|::)") if singles else None
        spans = rust_literals(text) if path.endswith(".rs") else []

        def fix(m: re.Match) -> str:
            name = m.group(1)
            if _inside(spans, m.start()):
                return name
            if name.rpartition(".")[2] in FILE_SUFFIXES and name not in self.L.modules:
                return name
            if any(name == d or name.startswith(d + ".") for d in self.L.keep_dotted):
                return name
            k = self.L.renamed_prefix(name)
            if k == 0:
                return name
            head = ".".join(name.split(".")[:k])
            if "." not in head and not head.startswith("verity_"):
                return name
            # `from pkg import facts`, which the import passes left naming `pkg` because `pkg.facts` is still there
            if re.search(r"(?:^|[^\w.])from\s+$", m.string[max(0, m.start() - 40):m.start()]) and \
                    (first := re.match(r"\s+import\s+\(?\s*([A-Za-z_]\w*)", m.string[m.end():])) and \
                    f"{name}.{first.group(1)}" in self.new_modules:
                return name
            new = self.L.module(name)
            if new == name:
                return name
            self._count(path, "dotted names")
            return new

        text = self._dot_re.sub(fix, text)
        if self._single_re is not None:
            spans = rust_literals(text) if path.endswith(".rs") else []

            def one(m: re.Match) -> str:
                at = m.start()
                if _inside(spans, at):
                    return m.group(1)
                if any(text.startswith(d, at) and not re.match(r"\w", text[at + len(d):at + len(d) + 1]) for d in self.L.keep_dotted):
                    return m.group(1)
                self._count(path, "dotted names")
                return self.L.module(m.group(1))
            text = self._single_re.sub(one, text)
        return text

    # pyprojects -------------------------------------------------------------------------------------------------------

    def _pyproject(self, path: str, text: str) -> str:
        merged = self.L.merged
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            self.note("pyprojects that don't parse", f"{path}: {e}")
            return text
        name = data.get("project", {}).get("name")
        out = text
        if name in merged:
            out = self._suite_only(path, out)
            self._count(path, "distributions folded into verity")
        conf = self.L.data.get("pyproject", {}).get(path, {})
        for header, body in conf.get("replace", {}).items():
            out = _replace_section(out, header, body)
        if scripts := conf.get("scripts"):
            out = _add_table_entries(out, "project.scripts", {k: json.dumps(v) for k, v in scripts.items()}, before="build-system")
        if add := conf.get("add_inputs"):
            out = _add_inputs(out, add)
        for old, new in conf.get("testpaths_replace", {}).items():
            out = _replace_list_item(out, "testpaths", old, new)
        for n in merged:
            dep = re.compile(r'"' + re.escape(n) + r'(\[[^\]]*\])?"')
            if dep.search(out):
                out = dep.sub('"verity"', out)
                self._count(path, "dependencies on a folded distribution")
            src = re.compile(r"^" + re.escape(n) + r"\s*=\s*\{[^}]*\}\s*\n", re.M)
            out = src.sub("", out)
        out = _dedupe_quoted_in_arrays(out, "verity")
        if path == "pyproject.toml":
            for n in conf.get("sources_drop", []):
                out = _drop_table_entry(out, "tool.uv.sources", n)
            new = self.L.data.get("distributions", {}).get("new", {})
            out = _add_table_entries(out, "tool.uv.sources", {n: "{ workspace = true }" for n in new}, before="dependency-groups")
            # the members the workspace root depended on, which `uv run` installed with it, become the default group's: the
            # root is now `verity`, which they depend on
            was = [_requirement(r) for r in data.get("project", {}).get("dependencies", [])]
            members = [n for n in was if n != "verity" and n not in merged] + [n for n in new if n not in was]
            if members:
                out = _add_group_items(out, "dev", members)
                self._count(path, "members in the default dependency group", len(members))
            return self._members(out, list(new.values()))
        if '"verity"' in out and "[tool.uv.sources]" in out and not re.search(r"^verity\s*=", out, re.M):
            out = out.replace("[tool.uv.sources]\n", "[tool.uv.sources]\nverity = { workspace = true }\n", 1)
        return out

    def _dependents(self):
        """A pyproject whose directory holds Python that imports a new distribution's package depends on it ([distributions]
        new), or, a suite-only one, takes its directory as an input. [create]'s pyprojects are the map's own, so one that lacks
        a distribution it imports is reported instead."""
        new = self.L.data.get("distributions", {}).get("new", {})
        names = {imp: dist for dist, d in new.items() for imp, r in self.L.new_roots.items() if r.startswith(d + "/")}
        if not names:
            return
        pat = re.compile(r"^\s*(?:from|import)\s+(" + "|".join(map(re.escape, sorted(names))) + r")\b", re.M)
        files = self.new_files | set(self.texts)
        owners = sorted((posixpath.dirname(p) for p in files if posixpath.basename(p) == "pyproject.toml"), key=len, reverse=True)
        need: dict[str, set[str]] = defaultdict(set)
        for f in sorted(files):
            if not f.endswith(".py") or not (text := self._read(f)) or not (hits := pat.findall(text)):
                continue
            owner = next((o for o in owners if o and f.startswith(o + "/")), "")
            need[owner] |= {names[h] for h in hits}
        created = self.L.data.get("create", {})
        for owner, dists in sorted(need.items()):
            if not owner:
                continue                                # the repository's own suite reads the whole tree
            path = f"{owner}/pyproject.toml"
            text = self._read(path) or ""
            data = tomllib.loads(text)
            project, tests = data.get("project", {}), data.get("tool", {}).get("verity", {}).get("tests", {})
            have = {_requirement(r) for r in project.get("dependencies", [])} | set(tests.get("inputs", []))
            lacking = sorted(x for x in dists if x != project.get("name") and new[x] != owner and x not in have
                             and new[x] not in have)
            if not lacking:
                continue
            if path in created:
                self.note("[create] pyprojects that import a distribution they don't name", f"{path}: {', '.join(lacking)}")
                continue
            if project:
                out = _add_dependencies(text, lacking)
            elif "pytest" in data.get("tool", {}):
                out = _add_inputs(text, [new[x] for x in lacking])
            else:
                continue
            if out != text:
                self.texts[path] = out
                self._count(path, "dependencies on a new distribution", len(lacking))

    def _suite_only(self, path: str, text: str) -> str:
        keep = []
        for header, body in _sections(text):
            if header is None or header.startswith(("tool.pytest", "tool.verity")):
                keep.append((header, body))
        out = "".join(body for _, body in keep)
        out = "# A suite of `verity`'s tests: they run in the workspace's environment, against the editable `verity`.\n" + \
            out.lstrip("\n")
        return _add_inputs(out, ["verity"], first=True)

    def _members(self, text: str, add: list[str] = ()) -> str:
        m = MEMBERS.search(text)
        if not m:
            return text
        items = re.findall(r'"([^"]+)"', m.group("body"))
        items_in = list(items)
        items += [a for a in add if a not in items]
        keep = []
        for it in items:
            pp = f"{it}/pyproject.toml"
            body = self.texts.get(pp) or self._read(pp)
            if body is None:
                continue
            try:
                proj = tomllib.loads(body).get("project")
            except tomllib.TOMLDecodeError as e:
                self.note("pyprojects that don't parse", f"{pp}: {e}")
                keep.append(it)
                continue
            if proj and proj.get("name") not in self.L.merged:
                keep.append(it)
        if keep == items_in:
            return text
        self._count("pyproject.toml", "workspace members", len(set(items_in) ^ set(keep)))
        body = ", ".join(f'"{k}"' for k in keep)
        return text[: m.start("body")] + body + text[m.end("body"):]

    def record(self):
        """Append this move's [paths] to research's `moves.json` once, so `tools_registry` still finds a tool's declaration in a
        tree from before it."""
        path = self.L.file(MOVES) or MOVES
        if (text := self._read(path)) is None:
            raise SystemExit(f"{path}: research's record of the layout moves is missing")
        def here(old: str, new: str) -> bool:
            if old.endswith("/"):
                return any(p.startswith(old) for p in self.tracked_old) or any(p.startswith(new) for p in self.new_files)
            return old in self.tracked_old or new in self.new_files
        # a row for a file an open branch adds (its restack needs the row) names a path this tree lacks until it lands; a row
        # of a branch's rows file is no row of the move: an entry that differed by branch would conflict where two land
        own = self.L.data.get("rows", ())
        doc, entry = json.loads(text), {"paths": {o: n for o, n in self.L.data["paths"].items() if o not in own and here(o, n)}}
        if entry not in doc["moves"]:
            doc["moves"].append(entry)
            self.texts[path] = json.dumps(doc, indent=2) + "\n"
            self._count(path, "layout moves recorded")

    def lock(self):
        """Name each moved distribution by its new directory in `uv.lock`. `uv lock` doesn't: where a map moves a
        distribution and no member's name changes, it reads the locked one's metadata at the old directory and fails
        (`Distribution not found at`), and so does `uv sync` on the moved tree."""
        if (text := self._read("uv.lock")) is None:
            return
        def moved(m: re.Match) -> str:
            if (new := self.L.file(f"{m.group(2)}/pyproject.toml")) is None:
                return m.group(0)
            self._count("uv.lock", "distribution directories")
            return f'{m.group(1)} = "{posixpath.dirname(new)}"'
        new = _LOCK_SOURCE.sub(moved, text)
        if new != text:
            self.texts["uv.lock"] = new

    # ---------------------------------------------------------------------------------------------- writing

    def write(self):
        for p, t in sorted(self.texts.items()):
            if self.dry:
                continue
            (self.root / p).write_text(t)


def _ancestors(p: str):
    d = posixpath.dirname(p)
    while d:
        yield d
        d = posixpath.dirname(d)


_RAW = re.compile(r'(?<!\w)b?r(#*)"')
_CHAR = re.compile(r"'(?:\\.[^'\n]*|[^'\\\n])'")


def rust_literals(text: str) -> list[tuple[int, int]]:
    """The (start, end) offsets of each string literal in Rust source, quotes included: plain, byte and raw ones. A quote in a
    comment or a char literal opens none; a `'` that closes no char literal is a lifetime's."""
    spans, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            depth, i = 1, i + 2
            while i < n and depth:
                if text.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif text.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
        elif c in "br" and (m := _RAW.match(text, i)):
            j = text.find('"' + m.group(1), m.end())
            end = n if j < 0 else j + 1 + len(m.group(1))
            spans.append((i, end))
            i = end
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            spans.append((i, min(j + 1, n)))
            i = j + 1
        elif c == "'" and (m := _CHAR.match(text, i)):
            i = m.end()
        else:
            i += 1
    return spans


def _inside(spans: list[tuple[int, int]], at: int) -> bool:
    k = bisect.bisect_right(spans, (at, float("inf"))) - 1
    return k >= 0 and spans[k][0] <= at < spans[k][1]


def _wrap(stmt: str, indent: str, width: int = 120) -> str:
    """`from m import a, b` over several lines in parentheses when it is longer than `width`."""
    if len(indent) + len(stmt) <= width or not stmt.startswith("from "):
        return stmt
    head, _, names = stmt.partition(" import ")
    inner = indent + "    "
    return f"{head} import (\n" + "".join(f"{inner}{n.strip()},\n" for n in names.split(",")) + f"{indent})"


def _continued(src: str, stmts: list[str], indent: str) -> list[str] | None:
    """The statements `stmts` that the `from` import `src` splits into, where `src` goes on over more lines (in brackets
    or after `\\`): those lines are kept as they were, after the one statement whose names end with theirs as written,
    which goes last. None when `src` is one line or no one statement has them."""
    first, _, rest = src.partition("\n")
    names = [n.strip() for line in rest.split("\n") for n in re.sub(r"#.*|[()\\]", "", line).split(",") if n.strip()]
    bracket = first.count("(") == 1 and ")" not in first
    if len(stmts) < 2 or not names or "#" in first or bracket == first.rstrip().endswith("\\"):
        return None
    for i, stmt in enumerate(stmts):
        head, _, mine = stmt.partition(" import ")
        mine = [n.strip() for n in mine.split(",")]
        if mine[-len(names):] == names:
            lead = ", ".join(mine[:-len(names)])
            line = f"{head} import ({lead}{',' * bool(lead)}" if bracket else f"{head} import {lead}{', ' * bool(lead)}\\"
            return [*(_wrap(t, indent) for t in stmts[:i] + stmts[i + 1:]), f"{line}\n{rest}"]
    return None


def _replace_names(src: str, old: str, new: str) -> str:
    return re.sub(r"(?<![\w.])" + re.escape(old) + r"(?![\w])", new, src, count=1)


def _sections(text: str):
    """[(header or None for the preamble, its text including the header line)]"""
    out, header, buf = [], None, []
    for line in text.splitlines(keepends=True):
        m = re.match(r"^\[\s*([^\[\]]+?)\s*\]\s*(#.*)?$", line.rstrip("\n"))
        if m:
            out.append((header, "".join(buf)))
            header, buf = m.group(1), [line]
        else:
            buf.append(line)
    out.append((header, "".join(buf)))
    return [(h, b) for h, b in out if h is not None or b.strip()]


def _replace_section(text: str, header: str, body: str) -> str:
    secs = _sections(text)
    new_body = f"[{header}]\n{body.rstrip()}\n\n"
    if not any(h == header for h, _ in secs):
        return text
    current = next(b for h, b in secs if h == header)
    if current.strip() == new_body.strip():
        return text
    first = body.split("\n[", 1)
    extra = re.findall(r"^\[([^\]]+)\]", body, re.M)
    out = []
    for h, b in secs:
        if h == header:
            out.append(new_body)
        elif h in extra:
            continue
        else:
            out.append(b)
    return "".join(out).rstrip("\n") + "\n"


def _add_table_entries(text: str, header: str, entries: dict[str, str], before: str) -> str:
    secs = _sections(text)
    if any(h == header for h, _ in secs):
        body = next(b for h, b in secs if h == header)
        add = [f"{k} = {v}\n" for k, v in entries.items() if not re.search(rf"^{re.escape(k)}\s*=", body, re.M)]
        if not add:
            return text
        return text.replace(body, body.rstrip("\n") + "\n" + "".join(add) + "\n", 1)
    block = f"[{header}]\n" + "".join(f"{k} = {v}\n" for k, v in entries.items()) + "\n"
    m = re.search(rf"^\[{re.escape(before)}\]", text, re.M)
    return text[: m.start()] + block + text[m.start():] if m else text.rstrip("\n") + "\n\n" + block


def _add_inputs(text: str, add: list[str], first: bool = False) -> str:
    secs = _sections(text)
    if not any(h == "tool.verity.tests" for h, _ in secs):
        items = ", ".join(json.dumps(a) for a in add)
        return text.rstrip("\n") + f"\n\n[tool.verity.tests]\ninputs = [{items}]\n"
    body = next(b for h, b in secs if h == "tool.verity.tests")
    m = re.search(r"^inputs\s*=\s*\[(?P<items>[^\]]*)\]", body, re.M)
    if not m:
        items = ", ".join(json.dumps(a) for a in add)
        return text.replace(body, body.rstrip("\n") + f"\ninputs = [{items}]\n\n", 1)
    have = re.findall(r'"([^"]+)"', m.group("items"))
    missing = [a for a in add if a not in have]
    if not missing:
        return text
    ins = ", ".join(json.dumps(a) for a in missing)
    items = m.group("items")
    new_items = (ins + (", " + items.lstrip() if items.strip() else "")) if first else \
        (items.rstrip() + (", " if items.strip() else "") + ins)
    new_body = body[: m.start("items")] + new_items + body[m.end("items"):]
    return text.replace(body, new_body, 1)


def _replace_list_item(text: str, key: str, old: str, new: list[str]) -> str:
    """Item `old` of every `key = [...]` array, replaced by the items `new`."""
    def fix(m: re.Match) -> str:
        if f'"{old}"' not in m.group("items"):
            return m.group(0)
        return m.group("head") + m.group("items").replace(f'"{old}"', ", ".join(json.dumps(n) for n in new), 1) + "]"
    return re.sub(rf"^(?P<head>{key}\s*=\s*\[)(?P<items>[^\]]*)\]", fix, text, flags=re.M)


def _requirement(r: str) -> str:
    return re.split(r"[\s\[<>=!~;@]", r.strip(), maxsplit=1)[0]


def _drop_table_entry(text: str, header: str, key: str) -> str:
    secs = _sections(text)
    if not any(h == header for h, _ in secs):
        return text
    body = next(b for h, b in secs if h == header)
    return text.replace(body, re.sub(rf"^{re.escape(key)}\s*=.*\n", "", body, count=1, flags=re.M), 1)


def _add_group_items(text: str, group: str, add: list[str]) -> str:
    """`add` appended to [dependency-groups] `group`, where it isn't there already."""
    secs = _sections(text)
    body = next((b for h, b in secs if h == "dependency-groups"), None)
    m = re.search(rf"^{re.escape(group)}\s*=\s*\[(?P<items>[^\]]*)\]", body or "", re.M)
    if m is None:
        return _add_table_entries(text, "dependency-groups", {group: json.dumps(add)}, before="tool.pytest.ini_options")
    have = {_requirement(x) for x in re.findall(r'"([^"]+)"', m.group("items"))}
    missing = [a for a in add if a not in have]
    if not missing:
        return text
    old = m.group("items")
    new_items = (old.rstrip().rstrip(",") + ", " if old.strip() else "") + ", ".join(json.dumps(a) for a in missing)
    return text.replace(body, body[: m.start("items")] + new_items + body[m.end("items"):], 1)


def _add_dependencies(text: str, add: list[str]) -> str:
    """`add` appended to [project] dependencies (the key made when it has none)."""
    secs = _sections(text)
    body = next(b for h, b in secs if h == "project")
    items = ", ".join(json.dumps(a) for a in add)
    m = re.search(r"^dependencies\s*=\s*\[(?P<items>[^\]]*)\]", body, re.M)
    if m is None:
        new_body = body.rstrip("\n") + f"\ndependencies = [{items}]\n" + body[len(body.rstrip("\n")) + 1:]
    else:
        old = m.group("items")
        if "\n" in old.strip("\n ") or old.endswith("\n"):          # one item per line: append in that style
            indent = re.search(r"\n([ \t]*)\S", old).group(1)
            head = old.rstrip()
            added = "".join(f"\n{indent}{json.dumps(a)}," for a in add)
            new_items = (head if head.endswith(",") else head + ",") + added + old[len(head):]
        else:
            new_items = (old.rstrip().rstrip(",") + ", " if old.strip() else "") + items
        new_body = body[: m.start("items")] + new_items + body[m.end("items"):]
    return text.replace(body, new_body, 1)


def _dedupe_quoted_in_arrays(text: str, name: str) -> str:
    def fix(m: re.Match) -> str:
        body = m.group(0)
        if body.count(f'"{name}"') < 2:
            return body
        first = body.index(f'"{name}"') + len(name) + 2
        rest = re.sub(r'\s*,\s*"' + re.escape(name) + r'"(?=\s*[,\]])', "", body[first:])
        rest = re.sub(r'"' + re.escape(name) + r'"\s*,\s*', "", rest)
        return body[:first] + rest
    return re.sub(r"\[[^\[\]]*\]", fix, text)


# -------------------------------------------------------------------------------------------------------------- main

def load(path: Path) -> Layout:
    return Layout(tomllib.loads(path.read_text()))


def rows_path(map_path: Path) -> Path:
    """The rows file of the map at `map_path`: beside it, `NAME.rows.toml` for `NAME.toml`."""
    return map_path.with_name(map_path.name.removesuffix(".toml") + ROWS_SUFFIX)


def with_rows(data: dict, rows_file: Path, root: Path) -> dict:
    """The map `data` with the rows of `rows_file`, a tracked file of the checkout at `root`, read after it: [paths] gains
    each row the map lacks ([rows] names them), [modules] is what the paths now imply, and [drop] gains the rows file itself,
    which is spent once read. It refuses a rows file that is not the checkout's own, that holds more than [paths], or a row
    of which changes what the map alone says or moves nothing."""
    tracked = {p for p in git(root, "ls-files", "-z").split("\0") if p}
    here = rows_file.resolve()
    rel = here.relative_to(root).as_posix() if here.is_relative_to(root) else None
    if rel not in tracked:
        raise SystemExit(f"{rows_file}: a rows file is its branch's own, committed beside the map in the checkout that moves")
    try:
        doc = tomllib.loads(rows_file.read_text())
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"{rows_file}: {e}") from e
    rows = doc.get("paths", {})
    if set(doc) - {"paths"} or not isinstance(rows, dict):
        raise SystemExit(f"{rows_file}: a rows file holds [paths] alone, not {sorted(set(doc) - {'paths'}) or 'a key named paths'}: "
                         "what else a move needs is its map's")
    lay = Layout(data)
    named = [k.rstrip("/") for k in (*data["paths"], *data.get("tokens", {}), *lay.drop)]
    bad = []
    for old, new in rows.items():
        o, is_dir = old.rstrip("/"), old.endswith("/")
        if not isinstance(new, str) or not o or not new.strip("/") or is_dir != new.endswith("/"):
            bad.append(f'{old!r} = {new!r}: a file to a file, or a directory to a directory, each ending in "/"')
            continue
        if data["paths"].get(old) == new:           # the map's own row, said again
            continue
        if not (any(p.startswith(old) for p in tracked) if is_dir else old in tracked):
            bad.append(f"{old!r}: no tracked file is {'under' if is_dir else 'at'} it")
            continue
        if is_dir:                                  # where the map alone sends it ("": it deletes it), and what it names there
            to = next((f"{n}{o[len(k):]}/" for k, n in lay.dirs if o == k or o.startswith(k + "/")), None)
            under = sorted(k for k in named if k == o or k.startswith(o + "/"))
        else:
            to, under = "" if old in lay.drop else lay.file(old), []
        if to == "":
            bad.append(f"{old!r}: the map deletes it")
        elif to is not None and to != new:
            bad.append(f"{old!r} = {new!r}: the map sends it to {to!r}")
        elif to is None and under:
            bad.append(f"{old!r}: the map names {len(under)} path{'s' * (len(under) != 1)} at or under it ({under[0]}"
                       f"{', ...' * (len(under) > 1)}), so it speaks for that directory: give each file a row of its own")
    own = {o: n for o, n in rows.items() if data["paths"].get(o) != n}
    merged = {**data, "paths": {**data["paths"], **own}, "rows": sorted(own),
              "drop": {"paths": sorted(lay.drop | {rel})}}
    if not bad:
        modules = derived_modules(Layout(merged))
        bad = [f"module {k}: the map renames it {v}, and these rows {modules.get(k) or 'not at all'}"
               for k, v in sorted(data.get("modules", {}).items()) if modules.get(k) != v]
    if bad:
        raise SystemExit(f"{rows_file}: rows that change what the map says, or move nothing:\n  " + "\n  ".join(bad))
    return {**merged, "modules": modules}


def unpinned_uv(root: Path) -> str | None:
    """Why `uv lock` here wouldn't be the pinned uv's (`tools/check/pod_setup.sh`), or None.  Another uv writes another lock
    header, so the move would differ from a rerun of the script with the pinned one, which `restack.py` refuses."""
    setup = next((p for p in (root / "tools" / "check" / "pod_setup.sh", root / "tools" / "verity" / "check" / "pod_setup.sh")
                  if p.exists()), None)
    m = setup and re.search(r"^UV_VERSION=\$\{UV_VERSION:-([^}]+)\}", setup.read_text(), re.M)
    if not m:
        return "pod_setup.sh pins no UV_VERSION"
    if not shutil.which("uv"):
        return f"uv isn't on the PATH (pod_setup.sh pins {m.group(1)})"
    got = subprocess.run(["uv", "--version"], capture_output=True, text=True).stdout.split()[1:2]
    return None if got == [m.group(1)] else f"uv is {got[0] if got else 'broken'}, not the {m.group(1)} pod_setup.sh pins"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", type=Path, default=HERE / "layout_map.toml")
    ap.add_argument("--root", type=Path, default=None, help="the checkout to move (default: this one)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-lock", action="store_true", help="skip `uv lock`")
    ap.add_argument("--modules", action="store_true", help="print the module renames [paths] implies, as TOML")
    ap.add_argument("--resolve", metavar="MODULE")
    ap.add_argument("--report", type=Path, help="write every change and note as JSON here")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--report-splits", action="store_true",
                    help="note references to a split directory that [split] doesn't acknowledge, instead of failing (restack.py "
                         "on a branch: its own references are its owner's to check)")
    a = ap.parse_args(argv)
    layout = load(a.map)
    if a.resolve:
        print(layout.module(a.resolve))
        return 0
    derived = derived_modules(layout)
    if a.modules:
        for k, v in sorted(derived.items()):
            print(f'"{k}" = "{v}"')
        return 0
    declared = layout.data.get("modules", {})
    if derived != declared:
        missing = {k: v for k, v in derived.items() if declared.get(k) != v}
        extra = {k: v for k, v in declared.items() if k not in derived}
        raise SystemExit(f"[modules] disagrees with what [paths] implies (`--modules` prints it):\n"
                         f"  implied but not declared: {missing}\n  declared but not implied: {extra}")
    root = (a.root or Path(git(HERE, "rev-parse", "--show-toplevel").strip())).resolve()
    if not a.allow_dirty and git(root, "status", "--porcelain").strip():
        raise SystemExit(f"{root}: the worktree isn't clean")
    if not a.dry_run and not a.no_lock and (why := unpinned_uv(root)):
        raise SystemExit(f"{why}: put the pinned uv first on the PATH, or pass --no-lock")
    base = git(root, "rev-parse", "HEAD").strip()
    if (rows := rows_path(a.map)).is_file():
        layout = Layout(with_rows(layout.data, rows, root))
    run = Run(root, layout, a.dry_run)
    try:
        moves = run.plan()
    except NoRow as e:
        print(f"no row for {len(e.files)} tracked file{'s' * (len(e.files) != 1)} under a path the map retires: give each a "
              f"row, in the map when main tracks the file, else in {rows} on the branch that adds it ([paths], \"old path\" = "
              f"\"new path\", a directory ending in \"/\"; committed, it is read after the map), or delete the file\n  "
              + "\n  ".join(e.files), file=sys.stderr)
        return NO_ROW
    run.index(moves)
    for old, new in moves:
        if old.endswith(".py") and (mo := module_name(old, layout.old_roots)) and (mn := module_name(new, layout.new_roots)):
            if layout.module(mo) != mn:
                raise SystemExit(f"{old} -> {new}: its module {mo} would be renamed {layout.module(mo)}, not {mn}")
    run.move(moves)
    run.create()
    run.rewrite_all()
    refuse_unread(run.unread)
    unacknowledged, stale = run.unacknowledged_splits()
    if a.report_splits:
        for x in unacknowledged:
            run.note("references to a split directory [split] doesn't acknowledge", x)
    elif unacknowledged or stale:
        raise SystemExit("references to a directory the map splits, which [split] doesn't acknowledge:\n"
                         + "".join(f"  {x}\n" for x in unacknowledged)
                         + "".join(f"  [split] acknowledges what no reference needs any more: {k}\n" for k in stale))
    run.record()
    run.lock()
    run.write()
    renamed_modules = sum(1 for o, n in moves if o.endswith(".py") and (mo := module_name(o, layout.old_roots))
                          and mo != module_name(n, layout.new_roots))
    run.changes["modules renamed"] = renamed_modules
    if "rows" in layout.data:
        run.changes[f"rows read from {rows}"] = len(layout.data["rows"])
    if not a.dry_run:
        git(root, "add", "-A")
        if moves:
            subprocess.run(["git", "--literal-pathspecs", "add", "-f", "--pathspec-from-file=-", "--pathspec-file-nul"], cwd=root,
                           input="\0".join(n for _, n in moves).encode(), check=True, capture_output=True)
        if not a.no_lock:
            r = subprocess.run(["uv", "lock"], cwd=root, capture_output=True, text=True)
            if r.returncode:
                run.note("uv lock failed", r.stderr.strip()[-2000:])
            else:
                git(root, "add", "uv.lock")
    print(f"layout move at base {base}{' (dry run)' if a.dry_run else ''}")
    for k, v in sorted(run.changes.items()):
        print(f"  {k}: {v}")
    print(f"  files rewritten: {len(run.files_changed)}")
    for kind, items in sorted(run.notes.items()):
        print(f"  {kind}: {len(items)}")
        for it in items[:8]:
            print(f"    {it}")
        if len(items) > 8:
            print(f"    ... {len(items) - 8} more")
    if a.report:
        a.report.write_text(json.dumps({"base": base, "changes": run.changes, "notes": run.notes,
                                        "files": {p: dict(c) for p, c in sorted(run.files_changed.items())},
                                        "moves": dict(moves)}, indent=1, sort_keys=True))
    if "uv lock failed" in run.notes:
        print("`uv lock` failed, so uv.lock doesn't match the moved tree and `uv sync` fails on it: fix what uv names "
              "above and run the move again", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
