#!/usr/bin/env python3
"""The rename move: rename every word a word map names, in the tree's paths, modules, Lean names and prose.

    python3 tools/move/rename.py --map MAP              rename and move, then stage it all; prints what it changed
    python3 tools/move/rename.py --map MAP --dry-run    print what it would change, and change nothing
    python3 tools/move/rename.py --map MAP --moves      print the declaration renames that the Lean locks' records need
                                                        ({old: new}), the map `tools/verity/lean/check.py --update --moved`
                                                        takes; run it on the tree you re-record, since a branch's records
                                                        hold names no other lock has

What it does, on the tree as it is (so it reruns on a later main, and `restack.py --script tools/move/rename.py` carries a
branch across it):
1. renames each [words] entry in every tracked text file the map doesn't exclude, except inside what [keep] keeps (a domain
   tag, a format id, an id; a name in a Python module's `__all__` only as it keeps the name, never as a quoted string,
   since `__all__` names what the module exports; the name of a file the second step leaves in place, `Words.hold`) and,
   in the files a [keep_in] glob matches, the words it lists; in a JSON file other than a lock, a Lake manifest or
   `moves.json`, it sorts again each object whose keys were sorted and that renamed keys leave out of order, keeping the
   file's lines but for their order and trailing commas, or else leaves the renamed text as it is (`resorted`);
2. moves each tracked file whose path the same words rename, with `layout.py`'s machinery (`layout.Run.move`), but for the
   words a [keep_in] or [keep_paths] glob lists for it ([keep_paths] holds a path and not the file's text); a module's
   name is its path's, so the first step has already renamed every import of it;
3. in a `lean-audit.json`, renames module names and prose but not the records the Lean check's `--update` writes (`guarantees`,
   `reads`, `dependencies`), so that `--update --moved` compares the build with them under their old names;
4. rewrites a `TRUSTED.sha256` line whose file the move changed, when it held that file's digest before, and likewise a
   Lake manifest's digest in a JSON pin of `tools/verity/check/pins.py`: a move changes a manifest only in where its path
   packages sit, which no pinned output reads (a pin of any other file the move changes is left for its command);
5. appends its renamed paths to research's `moves.json` once, as `layout.py` does, each moved path up to the end of each
   run of renamed parts (`.../warden/`, not each file under it, and `.../warden/tests/test_warden.py` besides; a directory
   only when it moves whole, else the file), so that a branch carried across the move appends the same entry unless it
   adds a path that renames a part below a renamed one.
It refuses a dirty worktree, and a second run changes nothing. The first step runs in `--jobs` forked processes (by
default the cores it may use, at most 32), since `restack.py` runs it once for each branch it carries; a file's new text
reads only that file and the map, so the result is one process's.

A map with `scope = "prose"` renames only prose, the part of a move `tools/verity/check/move_check.py` certifies without a
build: a word that stands as a word in a comment, a docstring or Markdown's text (`regions`, `prose_word`), never one in
code, a string literal, a data file, backticks, a name, a path or a constant's name, nor in a file a JSON pin records (its
command regenerates the pin). Such a move moves no path, renames no declaration (`--moves` prints `{}`) and leaves every
lock and pin as it was.

Every mode refuses a map whose [[words_in]] files, [keep_in] or [keep_paths] name a path or glob that no tracked file
matches (`Words.unmatched`), naming each, rather than skip the renames or keeps it holds for a file since moved or deleted.
"""
from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import fnmatch
import hashlib
import importlib.util
import json
import multiprocessing
import os
import posixpath
import re
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layout as L  # noqa: E402

POLICY = "lean-audit.json"
RECORDS = ("guarantees", "reads", "dependencies")   # what the Lean check's `--update` writes from the build
PINS = "TRUSTED.sha256"
PIN = re.compile(r"^([0-9a-f]{64})  (.+)$", re.M)
PINS_PY = "tools/verity/check/pins.py"              # the pins of Lean sources, some of them JSON (`json_pins`)
MANIFEST = "lake-manifest.json"
MOVES = L.MOVES                                     # research's record of each move's renamed paths
JOBS, CHUNK = 32, 8                                 # at most this many processes rewrite files, this many files a task
EXPORTS = re.compile(r"^__all__\s*\+?=\s*[\[(][^\])]*[\])]", re.M)   # a Python module's `__all__`: names, never data
# a space of a phrase: one, a line break and what starts the next line, or a break between two string literals Python joins
GAP = r"""(?: ["']\n[ \t]*[rbfRBF]{0,2}["']|["']\n[ \t]*[rbfRBF]{0,2}["'] | |\n[ \t]*(?:(?:///?|--|#|\*) )?)"""


def _match(path: str, glob: str) -> bool:
    return fnmatch.fnmatchcase(path, glob) or (glob.startswith("**/") and fnmatch.fnmatchcase(path, glob[3:]))


def _word(w: str, whole: bool = False) -> str:
    """`w` as it stands as a word or as one part of an identifier, or (``whole``) as a whole word, and a namespace (``X.``)
    only at the start of a name (the rules at the top of `tests/fixtures/rename_map.toml`)."""
    if whole:
        left = (r"(?<![\w.])" if w.endswith(".") else r"(?<!\w)") if re.match(r"\w", w[0]) else ""
        right = r"(?!\w)" if re.match(r"\w", w[-1]) else ""
    else:
        if w[0] == "'":
            left = ""
        elif re.search(r"[ ./]", w):
            left = r"(?<![\w-])"
        elif w[0].islower():
            left = r"(?<![A-Za-z0-9])"
        else:
            left = r"(?<![A-Z])"
        right = r"(?![a-z])" if w[-1].islower() else r"(?![A-Za-z])"
    return left + re.escape(w).replace(r"\ ", GAP) + right


def _suffixes(p: str) -> list[str]:
    parts = p.split("/")
    return ["/".join(parts[i:]) for i in range(len(parts))]


def _rewrapped(old: str, new: str) -> str:
    """`new` for a phrase `old` that a line break splits: the break, and what starts the next line, at the same space of
    `new`, or at its last; a one-word `new` joins the lines, but where they are two string literals `old` stands, since
    joining them could move words into a literal of another kind (an f-string's)."""
    gaps = re.findall(GAP, old)
    k = next(i for i, g in enumerate(gaps) if "\n" in g)
    parts = new.split(" ")
    if len(parts) == 1:
        return old if re.search("[\"']", gaps[k]) else new
    j = min(k, len(parts) - 2)
    return " ".join(parts[:j + 1]) + gaps[k] + " ".join(parts[j + 1:])


# What each part of a file is, for a map with `scope = "prose"` and for `move_check.py`'s certificate of its move: prose
# (comments, docstrings, Markdown's text), a literal (a string literal, a character, Markdown's code and link targets, and
# the whole of a data file, every byte of which some reader may parse) or code (the rest).
PROSE, LITERAL, CODE = "prose", "literal", "code"
DATA = (".json", ".jsonl", ".toml", ".yaml", ".yml", ".lock", ".csv", ".tsv", ".sha256", ".txt", ".template", ".cfg", ".ini")
C_LIKE = (".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh", ".rs", ".js", ".ts", ".tsx", ".go", ".java")
CODE_SPAN = re.compile(r"(`+)(?!`)(?:.|\n(?!\s*\n))+?(?<!`)\1(?!`)")
#: a Lean string literal whose braces are code: `s!`, `m!`, `f!` and the elaborator's message commands
LEAN_INTERPOLATED = re.compile(r"(?:\b[smf]!|\b(?:throwError|logInfo|logWarning|logError|dbg_trace))\s*$")
#: a new word that spells a name (`zk_served`, `Flock.Firewall`), which prose doesn't take in place of a word
NAMED = re.compile(r"_|\w\.\w")


def language(path: str, text: str) -> str:
    """The lexer `regions` reads ``path`` with: python, lean, shell, c, markdown, or data for anything else."""
    name = posixpath.basename(path)
    if name.endswith(".py"):
        return "python"
    if name.endswith(".lean"):
        return "lean"
    if name.endswith(".sh"):
        return "shell"
    if name.endswith(C_LIKE):
        return "c"
    if name.endswith((".md", ".markdown")):
        return "markdown"
    first = text.split("\n", 1)[0]
    if "." not in name and first.startswith("#!"):
        return "python" if "python" in first else "shell" if re.search(r"\b(?:ba|z|)sh\b", first) else "data"
    return "data"


def regions(path: str, text: str) -> list[tuple[int, int, str]]:
    """The prose and literal spans of ``text`` (file ``path``), sorted and disjoint; what they leave is code. A file no lexer
    here reads, or one its lexer can't read, is a literal throughout. In prose, a span in backticks is a literal, since it
    quotes code."""
    lang = language(path, text)
    if lang == "data":
        return [(0, len(text), LITERAL)] if text else []
    try:
        spans = {"python": _python, "lean": _lean, "shell": _shell, "c": _c_like, "markdown": _markdown}[lang](text)
    except (SyntaxError, ValueError):                    # unread, so nothing in it is renamed
        return [(0, len(text), LITERAL)]
    out = []
    for s, e, k in sorted(spans):
        if k != PROSE or lang == "markdown":
            out.append((s, e, k))
            continue
        i = s
        for m in CODE_SPAN.finditer(text, s, e):
            out += [(i, m.start(), PROSE)] if m.start() > i else []
            out.append((m.start(), m.end(), LITERAL))
            i = m.end()
        out += [(i, e, PROSE)] if e > i else []
    return out


def kind_at(spans: list[tuple[int, int, str]], i: int) -> str:
    """What position ``i`` is, in the `regions` ``spans``."""
    k = bisect.bisect_right(spans, (i, float("inf"), "")) - 1
    return spans[k][2] if k >= 0 and spans[k][0] <= i < spans[k][1] else CODE


def prose_word(text: str, spans: list[tuple[int, int, str]], s: int, e: int) -> bool:
    """Whether the word at ``text[s:e]`` stands in prose as a word, which a map with `scope = "prose"` renames: each of its
    characters but the spaces is prose, and it is no part of a name, a path, a flag or a call (`unit_draw`, `.draw`,
    `draw.py`, `--draw`, `draw(`), nor a constant's or a camel-case name (`DRAW`, `DrawnUnit`), which prose names as the code
    spells it. `Words.sub` also keeps a word whose new word is a name (`NAMED`)."""
    w = text[s:e]
    if any(not text[i].isspace() and kind_at(spans, i) != PROSE for i in range(s, e)):
        return False
    if re.search(r"[_./]|[a-z][A-Z]", w) or (sum(c.isalpha() for c in w) > 1 and w.upper() == w):
        return False
    left, right, after = text[s - 1:s], text[e:e + 1], text[e + 1:e + 2]
    if left and (left.isalnum() or left in "_./\\:@$`{<" or (left == "-" and text[s - 2:s - 1] in ("", "-", " ", "\t", "\n", "(", "`"))):
        return False
    return not (right and (right.isalnum() or right in "_(`}>"
                           or (right in "./:\\" and after and (after.isalnum() or after in "_/"))))


def _python(text: str) -> list[tuple[int, int, str]]:
    import ast
    import io
    import tokenize
    starts = [0] + [m.end() for m in re.finditer("\n", text)]
    lines = text.split("\n")
    docs = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            d = node.body[0]
            if isinstance(d, ast.Expr) and isinstance(d.value, ast.Constant) and isinstance(d.value.value, str):
                row = d.value.lineno
                docs.add((row, len(lines[row - 1].encode()[:d.value.col_offset].decode(errors="replace"))))
    out, opened, names = [], [], []
    f_start, f_end = getattr(tokenize, "FSTRING_START", None), getattr(tokenize, "FSTRING_END", None)

    def at(rc: tuple[int, int]) -> int:
        return starts[rc[0] - 1] + rc[1]
    try:
        for t in tokenize.generate_tokens(io.StringIO(text).readline):
            if t.type == tokenize.COMMENT:
                out.append((at(t.start), at(t.end), PROSE))
            elif t.type == tokenize.STRING and not opened:
                out.append((at(t.start), at(t.end), PROSE if t.start in docs else LITERAL))
            elif t.type == f_start:
                opened.append(at(t.start))
            elif t.type == f_end and opened:
                s = opened.pop()
                if not opened:
                    out.append((s, at(t.end), LITERAL))
            elif t.type == tokenize.NAME and opened:
                names.append((at(t.start), at(t.end)))
    except tokenize.TokenError as e:
        raise ValueError(e) from e
    split = []                                  # an f-string is a literal but for the names in its fields, which are code
    for s, e, k in sorted(out):
        j = bisect.bisect_left(names, (s, s)) if k == LITERAL else len(names)
        while j < len(names) and names[j][1] <= e:
            a, b = names[j]
            split += [(s, a, k)] if a > s else []
            s, j = b, j + 1
        split += [(s, e, k)] if e > s else []
    return split


def _lean(text: str) -> list[tuple[int, int, str]]:
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("--", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append((i, j, PROSE))
        elif text.startswith("/-", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith("/-", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("-/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            out.append((i, j, PROSE))
        elif c == '"' or (c == "r" and re.match(r'r#*"', text[i:i + 8]) and not re.match(r"[\w'.!?]", text[i - 1:i])):
            j = _lean_string(text, i, out)
        elif c == "'" and not re.match(r"[\w'.!?»₀-₉]", text[i - 1:i]) and \
                (m := re.compile(r"'(?:\\(?:u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2}|.)|[^\\'\n])'").match(text, i)):
            j = m.end()
            out.append((i, j, LITERAL))
        elif c == "«":
            j = text.find("»", i)
            j = n if j < 0 else j + 1
        else:
            j = i + 1
        i = j
    return out


def _lean_string(text: str, i: int, out: list) -> int:
    """The end of the Lean string literal at ``i``, its spans appended to ``out``: a literal, but in an interpolated one
    (`LEAN_INTERPOLATED`) the braces' insides, which are code."""
    n = len(text)
    if text[i] == "r":
        hashes = len(re.match(r"r(#*)", text[i:]).group(1))
        close = '"' + "#" * hashes
        j = text.find(close, i + 2 + hashes)
        j = n if j < 0 else j + len(close)
        out.append((i, j, LITERAL))
        return j
    interpolated = bool(LEAN_INTERPOLATED.search(text, max(0, i - 40), i))
    j, s = i + 1, i
    while j < n and text[j] != '"':
        if text[j] == "\\":
            j += 2
        elif text[j] == "{" and interpolated:
            out.append((s, j + 1, LITERAL))
            depth, j = 1, j + 1
            while j < n and depth:
                depth += {"{": 1, "}": -1}.get(text[j], 0)
                j = j + 1 if depth else j
            s = j
        else:
            j += 1
    j = min(n, j + 1)
    out.append((s, j, LITERAL))
    return j


def _shell(text: str) -> list[tuple[int, int, str]]:
    out, i, n = [], 0, len(text)
    heredocs: list[str] = []
    while i < n:
        c = text[i]
        if c == "\n" and heredocs:
            end, j = heredocs.pop(0), i + 1
            while j < n:
                k = text.find("\n", j)
                k = n if k < 0 else k
                if text[j:k].strip() == end:
                    break
                j = k + 1
            out.append((i + 1, j, LITERAL))
            i = j
            continue
        if c == "#" and (i == 0 or text[i - 1] in " \t\n;(|&"):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append((i, j, PROSE))
        elif c in "'\"":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if c == '"' and text[j] == "\\" else 1
            j = min(n, j + 1)
            out.append((i, j, LITERAL))
        elif m := re.compile(r"<<-?\s*(['\"]?)(\w+)\1").match(text, i):
            heredocs.append(m.group(2))
            j = m.end()
        elif c == "\\":
            j = i + 2
        else:
            j = i + 1
        i = j
    return out


def _c_like(text: str) -> list[tuple[int, int, str]]:
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append((i, j, PROSE))
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append((i, j, PROSE))
        elif (m := re.compile(r'r(#*)"').match(text, i)) and not re.match(r"\w", text[i - 1:i]):
            close = '"' + m.group(1)
            j = text.find(close, m.end())
            j = n if j < 0 else j + len(close)
            out.append((i, j, LITERAL))
        elif c == '"':
            j = i + 1
            while j < n and text[j] not in '"\n':
                j += 2 if text[j] == "\\" else 1
            j = min(n, j + 1)
            out.append((i, j, LITERAL))
        elif c == "'" and (m := re.compile(r"'(?:\\[^\n]{1,10}?|[^\\'\n])'").match(text, i)):
            j = m.end()
            out.append((i, j, LITERAL))
        else:
            j = i + 1
        i = j
    return out


def _markdown(text: str) -> list[tuple[int, int, str]]:
    out, i, fence = [], 0, None
    lines = text.splitlines(keepends=True)
    prose_from = 0
    for line in lines:
        m = re.match(r" {0,3}(`{3,}|~{3,})", line)
        if fence is None and m:
            out += _markdown_text(text, prose_from, i)
            fence, fence_at = m.group(1), i
        elif fence is not None and m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) and \
                not line.strip().strip(fence[0]):
            out.append((fence_at, i + len(line), LITERAL))
            fence, prose_from = None, i + len(line)
        i += len(line)
    if fence is not None:
        out.append((fence_at, len(text), LITERAL))
    else:
        out += _markdown_text(text, prose_from, len(text))
    return out


def _markdown_text(text: str, s: int, e: int) -> list[tuple[int, int, str]]:
    """Markdown's text in ``text[s:e]``: prose, but for its code spans, link targets and autolinks, which are literals."""
    out, i = [], s
    for m in re.compile(r"(`+)(?!`)(?:.|\n(?!\s*\n))+?(?<!`)\1(?!`)|\]\([^)\s]*(?:\s+\"[^\"]*\")?\)|<[a-z]+://[^>\s]+>|https?://\S+"
                        ).finditer(text, s, e):
        a = m.start() + 1 if m.group(0).startswith("]") else m.start()
        out += [(i, a, PROSE)] if a > i else []
        out.append((a, m.end(), LITERAL))
        i = m.end()
    out += [(i, e, PROSE)] if e > i else []
    return out


class Words:
    def __init__(self, data: dict):
        self.words: dict[str, str] = dict(data["words"])
        self.scoped: list[tuple[list[str], dict[str, str]]] = [(s["files"], dict(s["words"]))
                                                              for s in data.get("words_in", [])]
        for table in (self.words, *(t for _, t in self.scoped)):
            if bad := [o for o, n in table.items() if not o or not n or o == n]:
                raise SystemExit(f"the word map: entries that rename nothing: {bad}")
        self.exclude = data.get("rewrite", {}).get("exclude", [])
        keep = data.get("keep", {})
        self.keep = [re.compile(p) for p in keep.get("patterns", [])]
        self.keep += [re.compile(re.escape(s)) for s in keep.get("strings", [])]
        self.keep_in: dict[str, list[str]] = data.get("keep_in", {})
        self.keep_paths: dict[str, list[str]] = data.get("keep_paths", {})
        self.scope: str | None = data.get("scope")
        if self.scope not in (None, "prose"):
            raise SystemExit(f"the word map: scope {self.scope!r}: the one scope is \"prose\"")
        self._rules: dict[tuple[int, ...], list[tuple[list[str], str, str]]] = {}
        self._patterns: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[re.Pattern, list[str]]] = {}

    def excluded(self, path: str) -> bool:
        return path == MOVES or any(_match(path, g) for g in self.exclude)

    def rules(self, path: str, text: str) -> tuple[re.Pattern, list[str]]:
        """The words for file `path`: those of each [[words_in]] whose files match it (whole words, first), then [words],
        each only if `text` holds every one of its parts (no other can match). The pattern has one group per rule, and the
        list holds each group's new word."""
        which = tuple(i for i, (globs, _) in enumerate(self.scoped) if path and any(_match(path, g) for g in globs))
        if which not in self._rules:
            scoped: dict[str, str] = {}
            for i in which:
                for k, v in self.scoped[i][1].items():
                    scoped.setdefault(k, v)
            self._rules[which] = ([(w.split(" "), _word(w, whole=True), n) for w, n in scoped.items()] +
                                  [(w.split(" "), _word(w), n) for w, n in self.words.items()])
        rules = self._rules[which]
        live = tuple(i for i, (parts, _, _) in enumerate(rules) if all(p in text for p in parts))
        if (which, live) not in self._patterns:
            first = "".join(sorted({re.escape(rules[i][0][0][0]) for i in live}))
            pattern = f"(?=[{first}])(?:" + "|".join(f"({rules[i][1]})" for i in live) + ")" if live else "(?!)"
            self._patterns[which, live] = (re.compile(pattern), [rules[i][2] for i in live])
        return self._patterns[which, live]

    def sub(self, text: str, path: str = "", keep: list[tuple[int, int]] = (), counts: Counter | None = None,
            kept: Counter | None = None, protect: bool = True) -> str:
        """`text` (of file `path`) with every word renamed, except inside the spans [keep] (and `keep`) cover and the words
        [keep_in] lists for `path`. ``counts`` and ``kept`` count the words renamed (by old and new) and the words left
        standing; without ``protect`` (a path), [keep] keeps nothing and [keep_paths] holds its words as [keep_in] does.
        Under `scope = "prose"` a file with no prose (a data file, a lock) is returned as it is, its words not counted:
        finding each one in Security's lock takes most of a 32-process move."""
        pattern, new = self.rules(path, text)
        if pattern.search(text) is None:
            return text
        parts = regions(path, text) if self.scope and protect else None
        if parts is not None and all(k != PROSE for _, _, k in parts):
            return text
        held = [(m.start(), m.end()) for k in self.keep if protect for m in k.finditer(text)]
        if held and path.endswith(".py") and (names := [(m.start(), m.end()) for m in EXPORTS.finditer(text)]):
            exported = [any(a <= s and e <= b for a, b in names) for s, e in held]
            rest = [h for h, x in zip(held, exported) if not x]

            def free(name: str) -> bool:                            # the name stands somewhere outside every kept span
                return any(not any(s <= m.start() and m.end() <= e for s, e in rest)
                           for m in re.finditer(rf"(?<![\w'\"]){re.escape(name)}(?![\w'\"])", text))

            held = [(s, e) for (s, e), x in zip(held, exported)
                    if not (x and (text[e - 1] in "\"'" or free(text[s:e])))]
        spans = sorted([*keep, *held])
        starts, reach, far = [], [], 0
        for s, e in spans:                                          # reach[i]: the furthest end of spans[:i + 1]
            starts.append(s)
            far = max(far, e)
            reach.append(far)
        held_in = [*self.keep_in.items(), *(() if protect else self.keep_paths.items())]
        skip = {w for g, ws in held_in if path and _match(path, g) for w in ws}

        def one(m: re.Match) -> str:
            w = m.group(0)
            i = bisect.bisect_left(starts, m.end()) - 1               # the last span starting before the match ends
            to = _rewrapped(w, new[m.lastindex - 1]) if "\n" in w else new[m.lastindex - 1]
            if to == w or w in skip or (i >= 0 and reach[i] > m.start()) or \
                    (parts is not None and (not prose_word(text, parts, m.start(), m.end()) or NAMED.search(to))):
                if kept is not None:
                    kept[re.sub(GAP, " ", w)] += 1
                return w
            if counts is not None:
                counts[re.sub(GAP, " ", w), new[m.lastindex - 1]] += 1
            return to
        return pattern.sub(one, text)

    def path(self, p: str) -> str:
        return p if self.scope else self.sub(p, p, protect=False)

    def hold(self, files: list[str]) -> list[str]:
        """Keep, in every file, the names of the files among `files` that [keep_in] or [keep_paths] hold where they are
        and the words would otherwise move: each one's path, and each shorter suffix of it (its name included) that
        names no other file. Returns those paths."""
        held = [p for p in files if not self.excluded(p) and self.path(p) == p and self.sub(p, protect=False) != p]
        named = Counter(s for p in files for s in _suffixes(p))
        names = [s for p in held for s in _suffixes(p) if s == p or named[s] == 1 and self.sub(s, protect=False) != s]
        if names:
            self.keep.append(re.compile(r"(?<![\w./-])(?:" + "|".join(map(re.escape, sorted(names, key=len, reverse=True)))
                                        + r")(?![\w/-])"))
        return held

    def unmatched(self, files: list[str]) -> list[str]:
        """The paths and globs of [[words_in]]'s files, [keep_in] and [keep_paths] that match none of `files`, each after its
        table. Each names files the map renames or keeps apart, so one that matches none (a file since moved or deleted) is
        a rename or a keep the move would skip; [rewrite] exclude may name what a tree lacks (`archive/**`, a file that
        lands later), since it only leaves files as they are."""
        named = [("[[words_in]] files", g) for globs, _ in self.scoped for g in globs]
        named += [(t, g) for t, held in (("[keep_in]", self.keep_in), ("[keep_paths]", self.keep_paths)) for g in held]
        tracked = set(files)
        return [f"{t}: {g}" for t, g in dict.fromkeys(named) if g not in tracked and not any(_match(p, g) for p in files)]

    def name(self, n: str, path: str = "") -> str:
        """A Lean or Python dotted name, as it is renamed where it stands in file `path`: not where [keep] keeps it (a
        namespace that shares a module's name)."""
        return self.sub(n, path)


def record_spans(text: str) -> list[tuple[int, int]]:
    """The spans of a `lean-audit.json`'s top-level values that the Lean check's `--update` writes (`RECORDS`)."""
    dec, gap, spans = json.JSONDecoder(), re.compile(r"[\s,]*"), []
    i = gap.match(text, text.index("{") + 1).end()
    while text[i] != "}":
        key, i = json.decoder.scanstring(text, i + 1)
        i = re.compile(r"\s*:\s*").match(text, i).end()
        _, end = dec.raw_decode(text, i)
        if key in RECORDS:
            spans.append((i, end))
        i = gap.match(text, end).end()
    return spans


JSON_GAP = re.compile(r"[ \t\n\r]*")


def resorted(old: str, new: str, path: str) -> str:
    """``new``, the text ``old`` becomes under the words, with each JSON object whose keys were sorted in ``old`` and that
    renamed keys leave out of order sorted again, as a writer that keeps its keys sorted (`unread.py`'s caps) writes it:
    in a `*.json` file other than a lock, a Lake manifest or `moves.json`, when its lines stay ``new``'s but for their
    order and trailing commas. Otherwise (a JSON text on one line, a duplicate key, text that isn't JSON) ``new`` as it
    is."""
    if (old == new or not path.endswith(".json") or posixpath.basename(path) in (POLICY, MANIFEST) or path == MOVES
            or not (_is_json(old) and _is_json(new))):
        return new

    def again(value: tuple, was: tuple) -> str | None:
        """``new``'s span of ``value`` with each object sorted again that was sorted in ``was`` (``old``'s) and isn't now:
        its members change places and the text between them stays. None when the two differ in shape."""
        kind, start, end, parts = value
        if (kind, len(parts)) != (was[0], len(was[3])):
            return None
        if not kind:
            return new[start:end]
        inner = [again(v, w) for (_, _, v), (_, _, w) in zip(parts, was[3])]
        if None in inner:
            return None
        members = [new[at:v[1]] + s for (_, at, v), s in zip(parts, inner)]
        keys, before = [k for k, _, _ in parts], [k for k, _, _ in was[3]]
        if kind == "{" and before == sorted(before) and keys != sorted(keys):
            members = [m for _, m in sorted(zip(keys, members))]
        out, at = [], start
        for (_, a, v), m in zip(parts, members):
            out += [new[at:a], m]
            at = v[2]
        return "".join(out) + new[at:end]

    o, n = (_value(t, JSON_GAP.match(t).end()) for t in (old, new))
    body = again(n, o)
    out = new if body is None else new[:n[1]] + body + new[n[2]:]
    lines = [sorted(x.removesuffix(",") for x in t.splitlines()) for t in (out, new)]
    return out if lines[0] == lines[1] else new


def _unique(pairs: list) -> list:
    if len({k for k, _ in pairs}) < len(pairs):
        raise ValueError("a key twice in one object")
    return pairs


def _is_json(text: str) -> bool:
    """Whether ``text`` is JSON with no key twice in an object."""
    try:
        json.loads(text, object_pairs_hook=_unique)
    except ValueError:
        return False
    return True


def _value(text: str, i: int) -> tuple:
    """The value at ``text[i]`` of a JSON text with no key twice in an object, as (kind, start, end, parts): an object's
    (`{`) parts its members (key, the key's start, value), a list's (`[`) its values (None, start, value), a scalar's
    none."""
    if text[i] not in "{[":
        return "", i, json.JSONDecoder().raw_decode(text, i)[1], []
    kind, start, parts = text[i], i, []
    i = JSON_GAP.match(text, i + 1).end()
    while text[i] not in "}]":
        key, at = None, i
        if kind == "{":
            key, i = json.decoder.scanstring(text, i + 1)
            i = JSON_GAP.match(text, JSON_GAP.match(text, i).end() + 1).end()
        value = _value(text, i)
        parts.append((key, at, value))
        i = JSON_GAP.match(text, value[2]).end()
        if text[i] == ",":
            i = JSON_GAP.match(text, i + 1).end()
    return kind, start, i + 1, parts


def renamed(name: str, moved: dict[str, str]) -> str:
    """`tools/verity/lean/check.py`'s `renamed`: `name` with its longest moved prefix replaced."""
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        if (p := ".".join(parts[:i])) in moved:
            return ".".join([moved[p], *parts[i:]])
    return name


def module_files(root: Path, files: list[str]) -> dict[str, list[tuple[str, str | None]]]:
    """Each Lean module's files with their packages ({module: [(path, package directory)]}, one entry unless two packages
    define the module). A module is its file's path below its package's source directory: the nearest `lakefile.toml` at or
    above the file is its package, whose `srcDir` (its own directory by default) is that directory, so `Security/Proofs`'s
    `Properties/X.lean` is `Proofs.Properties.X` beside `Security/Properties`'s `Properties.X`. A file under no
    `lakefile.toml` is each dotted tail of its path that no other such file's path ends in, with no package. A test's
    fixture is no package's module."""
    lakes = {posixpath.dirname(p): posixpath.normpath(posixpath.join(posixpath.dirname(p),
                                                                      tomllib.loads((root / p).read_text()).get("srcDir", ".")))
             for p in files if posixpath.basename(p) == "lakefile.toml" and "/tests/fixtures/" not in f"/{p}"}
    out: dict[str, list[tuple[str, str | None]]] = {}
    tails: dict[str, list[str]] = {}
    for p in files:
        if not p.endswith(".lean") or "/tests/fixtures/" in f"/{p}":
            continue
        pkg = posixpath.dirname(p)
        while pkg not in lakes and pkg:
            pkg = posixpath.dirname(pkg)
        if pkg not in lakes:
            parts = p[:-len(".lean")].split("/")
            for i in range(len(parts)):
                tails.setdefault(".".join(parts[i:]), []).append(p)
        elif not (rel := posixpath.relpath(p, lakes[pkg])).startswith("../"):
            out.setdefault(rel[:-len(".lean")].replace("/", "."), []).append((p, pkg))
    for m, ps in tails.items():
        if len(ps) == 1:
            out.setdefault(m, []).append((ps[0], None))
    return out


def lock_moves(root: Path, words: Words, files: list[str]) -> dict[str, str]:
    """The declaration renames the locks' records need: for each guarantee and read definition whose name the words rename
    where it is declared (a read definition in its module's file, a guarantee in its lock), its first renamed prefix, and
    the full name too when that prefix alone would rename it otherwise; none implied by another. A prefix with such a name
    under it is listed as its names in full instead, since `check.py --moved` reads no move under another move's new name
    that maps back elsewhere (its `move_errors`). A lock under a `tests/fixtures/` directory is a test's data, in whatever
    form its test needs, and no package's record, so it is not read: `check.py --update --moved` re-records packages.

    A module two packages define (`module_files`) is read from the lock's own package when that package is one of them;
    otherwise every file defining it must rename the definition alike, and a map that renames it apart in them is refused
    before any records run, naming the lock, the definition and each file's rename."""
    want: dict[str, str] = {}
    where = module_files(root, files)
    for p in files:
        if posixpath.basename(p) != POLICY or words.excluded(p) or "/tests/fixtures/" in f"/{p}":
            continue
        d = json.loads((root / p).read_text())
        own = posixpath.dirname(p)
        names = {n: [p] for n in d.get("guarantees", {})}
        for m, g in d.get("reads", {}).items():
            fs = where.get(m, [])
            fs = [f for f, pkg in fs if pkg == own] or [f for f, _ in fs] or [p]
            names |= {n: fs for n in g.get("definitions", {})}
        for n, fs in sorted(names.items()):
            got = {f: n if words.excluded(f) else words.name(n, f) for f in fs}
            if len(set(got.values())) > 1:
                raise SystemExit(f"{p}: {n} is read from a module that {len(fs)} packages define, none of them this lock's, "
                                 "and the map renames it apart in their files: "
                                 + "; ".join(f"{f}: {new}" for f, new in sorted(got.items()))
                                 + ". Rename it alike in each, or keep it in each")
            if (new := got[fs[0]]) != n and want.setdefault(n, new) != new:
                raise SystemExit(f"{n} is renamed {want[n]} in one lock and {new} in {p}")
    want = dict(sorted(want.items()))
    moves: dict[str, str] = {}
    for old, new in want.items():
        a, b = old.split("."), new.split(".")
        i = next(i for i in range(len(a)) if a[i] != b[i])
        moves.setdefault(".".join(a[:i + 1]), ".".join(b[:i + 1]))
    for old, new in want.items():
        if renamed(old, moves) != new:
            moves[old] = new
    while nested := sorted({p for p, q in moves.items() for o, n in moves.items()
                            if n.startswith(q + ".") and o != p + n[len(q):]}):
        if nested[0] in want:
            raise SystemExit(f"{nested[0]} is renamed {moves[nested[0]]} and a name under it is renamed elsewhere: "
                             "`check.py --moved` can't read both back, so rename them in two moves")
        del moves[nested[0]]
        moves |= {o: n for o, n in want.items() if o.startswith(nested[0] + ".")}
    for old in sorted(moves, key=lambda k: -k.count(".")):
        rest = {k: v for k, v in moves.items() if k != old}
        if renamed(old, rest) == moves[old]:
            del moves[old]
    return dict(sorted(moves.items()))


def repin(root: Path, words: Words, pins: str, texts: dict[str, str], dest: dict[str, str]) -> str | None:
    """The `TRUSTED.sha256` at `pins` after the move: each line whose file the move changes or moves, and whose digest was
    that file's, gets the file's new digest and path. None when no line changes."""
    text = (root / pins).read_text()
    here = posixpath.dirname(pins)

    def line(m: re.Match) -> str:
        digest, rel = m.groups()
        target = posixpath.normpath(posixpath.join(here, rel))
        try:
            before = (root / target).read_bytes()
        except OSError:
            return m.group(0)
        after = texts[target].encode() if target in texts else before
        if hashlib.sha256(before).hexdigest() != digest or (after == before and target not in dest):
            return m.group(0)
        return f"{hashlib.sha256(after).hexdigest()}  {words.path(rel)}"
    new = PIN.sub(line, text)
    return None if new == text else new


def json_pins(root: Path) -> list[tuple[str, str, str]]:
    """The pins `tools/verity/check/pins.py` lists that record {path: digest} under a JSON key, as (pin, key, the directory
    its paths are below); none in a tree without it."""
    if not (root / PINS_PY).is_file():
        return []
    spec = importlib.util.spec_from_file_location("rename_move_pins", root / PINS_PY)
    mod = sys.modules[spec.name] = importlib.util.module_from_spec(spec)       # its dataclass looks its module up
    spec.loader.exec_module(mod)
    return [(p.path, p.key, p.under) for p in mod.PINS if p.key is not None]


def pinned(root: Path) -> set[str]:
    """The files a JSON pin records the digest of: its command regenerates it, which a move doesn't run, so a move that
    renames only prose leaves them as they are."""
    out = set()
    for pin, key, under in json_pins(root):
        if (root / pin).is_file():
            out |= {posixpath.normpath(posixpath.join(under, p)) for p in json.loads((root / pin).read_text()).get(key, {})}
    return out


def repin_manifests(root: Path, words: Words, pin: str, key: str, under: str, texts: dict[str, str]) -> str | None:
    """The JSON pin at `pin` after the move: each Lake manifest it records under `key` whose digest it held, and which the
    move changes, gets the manifest's new digest. None when no digest changes."""
    before = (root / pin).read_text()
    text = new = texts.get(pin, before)
    for rel, digest in json.loads(before).get(key, {}).items():
        target = posixpath.normpath(posixpath.join(under, rel))
        if posixpath.basename(target) != MANIFEST or target not in texts:
            continue
        if hashlib.sha256((root / target).read_bytes()).hexdigest() != digest:
            continue
        written = rel if words.excluded(pin) else words.sub(rel, pin)
        after = hashlib.sha256(texts[target].encode()).hexdigest()
        new = re.sub(rf'("{re.escape(written)}"\s*:\s*"){digest}"', lambda m, h=after: f'{m[1]}{h}"', new)
    return None if new == text else new


def renamed_paths(moves: list[tuple[str, str]], present: list[str] = ()) -> dict[str, str]:
    """Each moved path up to the end of each of its runs of renamed parts ({old: new}; a directory ends in "/"), so that
    every file under one renamed directory gives the same entries, a path below it that renames another part
    (`warden/tests/test_warden.py`) is recorded besides, since `move_check.py` and research's `tools_registry` read the
    longest entry that holds a path, and a swap (`Properties/Proofs/` to `Proofs/Properties/`) is recorded whole. A
    directory in which a file of ``present`` that no longer entry holds stays, or lands elsewhere than below its new path,
    didn't move whole (a word that names one file: `identity/Identity/Domains.lean`): the files that moved out of it are
    recorded instead, as `move_check.py`'s `moves_change` reads the entry."""
    out: dict[str, str] = {}
    for o, n in moves:
        a, b = o.split("/"), n.split("/")
        spans = [(o, n)] if len(a) != len(b) else []
        i = 0 if len(a) == len(b) else len(a)
        while i < len(a):
            if a[i] != b[i]:
                while i + 1 < len(a) and a[i + 1] != b[i + 1]:
                    i += 1
                end = "/" if i < len(a) - 1 else ""
                spans.append(("/".join(a[:i + 1]) + end, "/".join(b[:i + 1]) + end))
            i += 1
        for old, new in spans:
            if out.setdefault(old, new) != new:
                raise SystemExit(f"{old} would move to both {out[old]} and {new}")
    dest = dict(moves)

    def holder(f: str) -> str | None:
        return max((k for k in out if f == k or (k.endswith("/") and f.startswith(k))), key=len, default=None)
    while part := next((d for d in out if d.endswith("/") and any(
            holder(f) == d and dest.get(f, f) != out[d] + f[len(d):] for f in present if f.startswith(d))), None):
        del out[part]
        for o, n in moves:
            if o.startswith(part) and ((h := holder(o)) is None or out[h] + o[len(h):] != n):
                out[o] = n
    return dict(sorted(out.items()))


def record(root: Path, moves: list[tuple[str, str]], present: list[str] = ()) -> str | None:
    """research's `moves.json` with this move's `renamed_paths` appended, or None when it holds them already or nothing
    moves."""
    if not moves:
        return None
    if not (root / MOVES).is_file():
        raise SystemExit(f"{MOVES}: research's record of the moves is missing")
    doc, entry = json.loads((root / MOVES).read_text()), {"paths": renamed_paths(moves, present)}
    if entry in doc["moves"]:
        return None
    doc["moves"].append(entry)
    return json.dumps(doc, indent=2) + "\n"


#: what the workers `rewrite` forks read: (root, words, run)
_FORKED: tuple = ()


def _rewrite(files: list[str]) -> tuple[dict[str, str], Counter, Counter]:
    """The new text of each of `files` that the words change, the words renamed (by old and new) and the words left standing
    (by file and word)."""
    root, words, run = _FORKED
    texts, counts, kept = {}, Counter(), Counter()
    for p in files:
        lock = posixpath.basename(p) == POLICY
        text = run._read(p)
        if text is None:
            continue
        here = Counter()
        new = resorted(text, words.sub(text, p, record_spans(text) if lock else [], counts, here), p)
        kept.update({(p, w): n for w, n in here.items()})
        if new != text:
            texts[p] = new
    return texts, counts, kept


def rewrite(root: Path, words: Words, run: L.Run, files: list[str], jobs: int) -> tuple[dict[str, str], Counter, Counter]:
    """`_rewrite` of `files`, in `jobs` forked processes when that is more than one, largest files first: a file's new text
    reads that file and the map alone, so it is what one process writes, in the files' order."""
    global _FORKED
    _FORKED = (root, words, run)
    if jobs > 1 and len(files) > CHUNK:
        big = sorted(files, key=lambda p: -os.path.getsize(root / p))
        with concurrent.futures.ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("fork")) as ex:
            parts = list(ex.map(_rewrite, [big[i:i + CHUNK] for i in range(0, len(big), CHUNK)]))
    else:
        parts = [_rewrite(files)]
    got, counts, kept = {}, Counter(), Counter()
    for t, c, k in parts:
        got.update(t)
        counts.update(c)
        kept.update(k)
    return {p: got[p] for p in files if p in got}, counts, kept


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", type=Path, required=True, help="the word map (`tests/fixtures/rename_map.toml` is an example)")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--moves", action="store_true", help="print the declaration renames this tree's locks need, for the Lean check's `--update --moved`")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--jobs", type=int, default=min(JOBS, len(os.sched_getaffinity(0))),
                    help=f"the processes that rewrite files (default: the cores this process may use, at most {JOBS})")
    a = ap.parse_args(argv)
    root = (a.root or Path(L.git(HERE, "rev-parse", "--show-toplevel").strip())).resolve()
    words = Words(tomllib.loads(a.map.read_text()))
    files = [p for p in L.git(root, "ls-files", "-z").split("\0") if p]
    if missing := words.unmatched(files):
        raise SystemExit(f"{a.map}: it names paths no tracked file matches, whose renames or keeps the move would skip; point "
                         "each at its file's path on this tree, or drop it:\n  " + "\n  ".join(missing))
    words.hold(files)
    if a.moves:
        print(json.dumps({} if words.scope else lock_moves(root, words, files), indent=1))
        return 0
    if not a.dry_run and not a.allow_dirty and L.git(root, "status", "--porcelain").strip():
        raise SystemExit(f"{root}: the worktree isn't clean")
    fixed = pinned(root) if words.scope else set()
    run = L.Run(root, L.Layout({"roots": {"old": {}, "new": {}}, "paths": {}}), a.dry_run)
    run.old_of = {}
    present = [p for p in files if os.path.lexists(root / p)]
    moves = [(p, n) for p in present if not words.excluded(p) and (n := words.path(p)) != p]
    dest = dict(moves)
    landing = Counter(dest.get(p, p) for p in present)
    if clash := sorted(n for n, c in landing.items() if c > 1):
        raise SystemExit("two files would land at one path:\n  " + "\n  ".join(clash[:50]))
    folded = Counter(dest.get(p, p).lower() for p in present)
    if fold := sorted(n for n in dest.values() if folded[n.lower()] > 1):
        raise SystemExit("paths that would differ only in case:\n  " + "\n  ".join(fold[:50]))

    todo = [p for p in present if not (words.excluded(p) or os.path.islink(root / p) or p in fixed)]
    L.refuse_unread({p: os.path.getsize(root / p) for p in todo if L.oversize(root / p)})
    texts, counts, kept = rewrite(root, words, run, todo, a.jobs)
    pins = {p: t for p in present if posixpath.basename(p) == PINS and (t := repin(root, words, p, texts, dest)) is not None}
    tracked = set(present)
    pins |= {p: t for p, key, under in json_pins(root)
             if p in tracked and (t := repin_manifests(root, words, p, key, under, texts)) is not None}
    recorded = {MOVES: t} if (t := record(root, moves, present)) is not None else {}

    print(f"rename move at {L.git(root, 'rev-parse', 'HEAD').strip()}{' (dry run)' if a.dry_run else ''}")
    print(f"  files rewritten: {len(texts)}, files moved: {len(moves)}, pins repinned: {len(pins)}")
    if recorded:
        print(f"  {MOVES}: the renamed paths recorded, {len(renamed_paths(moves, present))} of them")
    for (w, new), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {n:5d}  {w!r} -> {new!r}")
    if kept:
        print(f"  left standing ([keep], [keep_in] or a lock's records): {sum(kept.values())}")
        for (p, w), n in sorted(kept.items()):
            print(f"    {n:5d}  {w!r} in {p}")
    if a.dry_run:
        for o, n in moves:
            print(f"    move {o} -> {n}")
        for p in sorted(pins):
            print(f"    repin {p}")
        for o, n in renamed_paths(moves, present).items() if recorded else ():
            print(f"    record {o} -> {n}")
        return 0
    for p, t in {**texts, **pins, **recorded}.items():
        with open(root / p, "w", encoding="utf-8", newline="") as f:
            f.write(t)
    run.move(moves)
    L.git(root, "add", "-A")
    if moves:
        subprocess.run(["git", "--literal-pathspecs", "add", "-f", "--pathspec-from-file=-", "--pathspec-file-nul"],
                       cwd=root, input="\0".join(n for _, n in moves).encode(), check=True, capture_output=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
