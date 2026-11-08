"""A Lean file's import header, read as Lean 4.34 reads it: the one reading for every tool that needs a file's imports (the
per-module cache keys, `module_keys.py`, and the task graph's `lean-module` family) or rewrites them (the codemod,
`modules.py`, at `starts`; the merge driver; the lift). Stdlib only, loaded by path.

The header is `module`, then `prelude`, each optional and in that order, then imports, each `[public] [meta] import [all] M`
with its words in that order. Spaces, line breaks and comments (`--` to the end of the line, nested `/- -/`) may sit between
any two tokens, and several imports may share a line. The header ends at the first other token: a declaration, `public
section`, a word out of order (`meta public import`, `prelude module`), or a doc comment (`/--`, `/-!`), which Lean reads as
a command, so an `import` after one is refused. Lean refuses a tab and a byte-order mark, so neither is a space here.
`public import all` is read, and Lean refuses it after reading it, as it refuses `public`, `meta` and `all` without
`module`."""
from __future__ import annotations

import functools
import re
from typing import NamedTuple

#: the words of a header, none of which names a module there
WORDS = ("module", "prelude", "public", "meta", "import", "all")
#: a module's name: components joined by dots, none empty (`import A.` is refused), each `«»`-quoted or bare
NAME = re.compile(r"(?:«[^»]*»|[^.«»]+)(?:\.(?:«[^»]*»|[^.«»]+))*")


@functools.lru_cache(maxsize=None)
def components(mod: str) -> tuple[str, ...]:
    """A module name's components, each without the `«»` that quotes it, kept per name: the task graph's listing asks for
    each imported module's some hundred times (`module_keys.Package.claims`, once per library root of each package)."""
    return tuple(p.strip("«»") for p in re.findall(r"«[^»]*»|[^.«»]+", mod))


def parts(mod: str) -> list[str]:
    """A module name's components, each without the `«»` that quotes it."""
    return list(components(mod))


@functools.lru_cache(maxsize=None)
def name(mod: str) -> str:
    return ".".join(components(mod))


class Import(NamedTuple):
    module: str
    public: bool = False
    meta: bool = False
    all: bool = False


class Header(NamedTuple):
    module: bool
    prelude: bool
    imports: tuple[Import, ...]
    #: where the header ends in the source: just past its last token (`module`, `prelude` or an import's module), 0 without one
    end: int


def skip(src: str, i: int) -> int:
    """Past the spaces and comments at ``i`` in ``src``; -1 when a comment is left open."""
    n = len(src)
    while i < n:
        if src[i] in " \r\n":
            i += 1
        elif src.startswith("--", i):
            j = src.find("\n", i)
            i = n if j < 0 else j + 1
        elif src.startswith("/-", i) and not src.startswith(("/--", "/-!"), i):
            depth, i = 1, i + 2
            while i < n and depth:
                if src.startswith("/-", i):
                    depth, i = depth + 1, i + 2
                elif src.startswith("-/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            if depth:
                return -1
        else:
            break
    return i


def _token(src: str, i: int) -> tuple[str, int]:
    """The token at ``i`` (up to a space or a comment, a `«»` span whole) and where it ends."""
    j, n = i, len(src)
    while j < n and src[j] not in " \r\n" and not src.startswith(("--", "/-"), j):
        if src[j] == "«":
            k = src.find("»", j)
            j = n if k < 0 else k + 1
        else:
            j += 1
    return src[i:j], j


def header(src: str) -> Header | None:
    """``src``'s header, or None when it is cut off: a comment left open, or an import without a module's name (`NAME`)."""
    r = _read(src)
    return None if r is None else r[0]


def starts(src: str) -> tuple[int, ...] | None:
    """Where each import of ``src``'s header starts (its first word: `public`, `meta` or `import`), in `header`'s order, or
    None when the header is cut off: where a tool that rewrites a header (`modules.py`, the codemod) edits it."""
    r = _read(src)
    return None if r is None else r[1]


def _read(src: str) -> tuple[Header, tuple[int, ...]] | None:
    flags, i, end, at = {}, 0, 0, []
    for word in ("module", "prelude"):
        if (k := skip(src, i)) < 0:
            return None
        t, j = _token(src, k)
        flags[word] = t == word
        if flags[word]:
            i = end = j
    out: list[Import] = []
    while True:
        if (start := skip(src, i)) < 0:
            return None
        words, k = set(), start
        for word in ("public", "meta", "import"):
            t, j = _token(src, k)
            if t == word:
                words.add(word)
                if (k := skip(src, j)) < 0:
                    return None
        if "import" not in words:
            return Header(flags["module"], flags["prelude"], tuple(out), end), tuple(at)
        t, j = _token(src, k)
        if every := t == "all":
            if (k := skip(src, j)) < 0:
                return None
            t, j = _token(src, k)
        if t in WORDS or not NAME.fullmatch(t):
            return None
        out.append(Import(name(t), "public" in words, "meta" in words, every))
        at.append(start)
        i = end = j


def imports(src: str) -> list[str] | None:
    """The modules ``src``'s header imports, in order, or None when it is cut off (`header`)."""
    h = header(src)
    return None if h is None else [i.module for i in h.imports]
