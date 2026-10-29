"""Lane text in the tree's prose, held to ``allowlists/repo_lane_text.json``: S5 of the quality rollout, a tree that says what
is true now.  Prose is every comment and docstring in code (Python's comments and string statements, read by ``tokenize``;
Lean's ``--``, ``/- -/`` and ``/-- -/`` comments; the ``#`` comments of shell, TOML, YAML and unit files; the ``//`` and
``/* */`` comments of Rust, C, C++ and CUDA) and every Markdown file, whole.  Code's string literals are not prose and are
not read.  Lane text is what records how the tree got here or who is working on it, which the tree can't keep true:

* ``attribution``: a parenthetical holding a date, ``(Daniel, 18 Oct)``, ``(12 Oct 06:20Z)``, ``(2026-10-12)``, or a date
  that follows ``Daniel``;
* ``date``: an ISO date, ``2026-10-12``;
* ``note``, ``notion``, ``linear``, ``pr``: a ``note:`` id, a Notion URL, a Linear issue id (``VLLM-136``, the workspace's
  team keys) or URL, and a pull request (``#1234``, or its GitHub URL);
* ``agent``, ``slack``: a Cursor agent id (``bc-`` and hex) and a Slack timestamp or archive URL;
* ``codename``: a worker's codename in prose: a board id (``M-1234``, ``F-r16``), a round (``R10``..``R19``), ``[lane `` or
  ``lane NAME`` with a hyphenated lane name, and a handle (``@vllm``), except where AGENTS.md and the skills name a handle
  as an owner.

``art:`` ids are allowed: they name evidence, which the store keeps.  Declared data isn't read at all (``*.json``, ``*.csv``
and every other file not named above: ``fixtures/artifacts.json``, each ``lean-audit.json``'s records, the handle registry,
the manifests), nor what ``SKIP`` names, each with its reason.  This lint takes over P11's docstring patterns
(``integrations/vllm/tests/lint/test_p11_names.py``: board ids, ``[lane ``, rounds and dates), for the whole tree.

Sites are counted per file, the key the ratchet gate counts by too: an entry ``{"file", "count"}`` allows ``count`` sites in
its file.  A file with more than its entry allows, or with one and no entry, fails, and the files ``PINNED`` names
(AGENTS.md, ARCHITECTURE.md, GLOSSARY.md and the skills) hold none and may have no entry.  An entry that allows more than
is found doesn't fail: the list can run ahead of a tree, as each area cleans its own, and ``--prune`` deletes and lowers
what this tree no longer finds.  A file that moves takes its entry with it, in the same change (``--move``).  The ratchet
gate (``integrations/vllm/tools/ratchet_gate.py``) refuses a change that adds a site to a file it touches, or that grows
this list or leaves a renamed file's entry behind, and judges a change to this module by running it as the change's base
has it on the change's tree.  So it imports only the standard library, ``test_no_new_lane_text`` takes no fixture, and the
gate reads ``LEFT_OUT``, ``FIX``, ``holds``, ``sites`` and ``parse`` from the base's copy of this module.

    python3 tests/lint/test_repo_lane_text.py --move OLD NEW   # a file (or directory) moved: its entry follows it
    python3 tests/lint/test_repo_lane_text.py --prune          # delete and lower what this tree no longer finds
    python3 tests/lint/test_repo_lane_text.py --record         # the list is what this tree finds (the gate holds a rise)
"""
from __future__ import annotations

import bisect
import io
import json
import re
import subprocess
import sys
import tokenize
from collections import Counter
from functools import lru_cache
from pathlib import Path

NAME = "repo_lane_text"
ROOT = Path(__file__).resolve().parents[2]
LIST = Path(__file__).resolve().parent / "allowlists" / f"{NAME}.json"
#: the archive isn't developed
LEFT_OUT = ("archive/",)
#: prose that is data, by path prefix (or a whole path), each with its reason
SKIP = {
    "fixtures/": "replicated fixtures and their notes are data, named by what they hold",
    "benchmarks/census/": "data only (AGENTS.md: verity never reads it)",
    f"tests/lint/test_{NAME}.py": "its examples are the lane text it refuses",
}
#: test fixtures: a path with one of these directories is data
FIXTURE_DIRS = ("fixtures", "testdata", "golden")
#: the files that hold no lane text and may have no entry
PINNED = ("AGENTS.md", "ARCHITECTURE.md", "GLOSSARY.md", ".agents/skills/")
#: where AGENTS.md and the skills name handles as owners
OWNERS = ("AGENTS.md", ".agents/skills/")
FIX = ("Say what is true now, and put the rest where it lives: history in git (the commit message), work in Linear (the "
       "issue), rulings in Notion (the decisions log), results in the store (`research data put`, cited as `art:<id>`). "
       f"If the file moved, move its entry in the same change: `python3 tests/lint/test_{NAME}.py --move OLD NEW`.")

PY = (".py", ".pyi")
LEAN = (".lean",)
MARKDOWN = (".md",)
HASH = (".sh", ".bash", ".toml", ".yaml", ".yml", ".service", ".timer", ".conf", ".cfg", ".ini")
SLASH = (".rs", ".c", ".h", ".cc", ".cpp", ".hpp", ".cu", ".cuh", ".ts", ".js", ".go")

MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
DATE = rf"(?:\b\d{{1,2}} {MONTH}(?![\w-])|\b{MONTH} \d{{1,2}}\b(?!:)|\b20\d\d-[01]\d-[0-3]\d\b)"
TEAMS = "TRN|CMP|TOP|CI|PRF|CIR|INF|NET|HASH|MEM|VLLM|LEAN|BENCH|COMM|CON|DAN|LAND"
HANDLES = ("compute-accounting", "compute-research", "memory-accounting", "network-accounting", "benchmarks", "circuits",
           "soundness", "training", "console", "kernels", "proofs", "infra", "comms", "lean", "vllm", "top", "ci")
WORD = re.compile(r"\w")
#: (kind, pattern, the characters that may not precede a match): each pattern starts with a literal or a small class, which
#: ``re`` searches for fast, so the tree reads in seconds; where two overlap, the one that starts first wins, then the one
#: listed first, so an attribution's date counts once
PATTERNS = tuple((kind, re.compile(p), before and re.compile(before)) for kind, p, before in (
    ("attribution", rf"\((?:[^()\n]{{0,80}}?[\s,;:(])?{DATE}[^()\n]{{0,120}}\)?", None),
    ("attribution", rf"Daniel(?:'s)?\b[^()\n.]{{0,40}}?{DATE}", r"\w"),
    ("date", r"20\d\d-[01]\d-[0-3]\d\b", r"\w"),
    ("note", r"note:[0-9A-Za-z][\w.:-]*", r"\w"),
    ("notion", r"notion\.(?:so|site|com)\b\S*", None),
    ("linear", rf"(?:{TEAMS})-\d+\b", r"\w"),
    ("linear", r"linear\.app/\S*", r"\w"),
    ("pr", r"#\d{3,5}\b", r"[\w&/#]"),
    ("pr", r"github\.com/[\w.-]+/[\w.-]+/(?:pull|issues)/\d+", None),
    ("agent", r"bc-[0-9a-f]{6,}", r"\w"),
    ("slack", r"1[5-9]\d{8}\.\d{6}\b", r"\w"),
    ("slack", r"slack\.com/archives/\S*", None),
    ("codename", r"M-\d{4}\b|F-r\d+|R1\d\b", r"\w"),
    ("codename", r"\[lane ", None),
    ("codename", r"[Ll]ane[: ]+`?[a-z][a-z0-9]*(?:-[a-z0-9]+)+", r"\w"),
    ("codename", rf"@(?:{'|'.join(HANDLES)})(?![\w-])", r"[\w.@/+-]"),
))
HANDLE = re.compile(rf"^@(?:{'|'.join(HANDLES)})$")
PY_LEX = re.compile(r'#[^\n]*|"""(?:[^"\\]|\\.|"(?!""))*"""|\'\'\'(?:[^\'\\]|\\.|\'(?!\'\'))*\'\'\''
                    r'|"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'', re.DOTALL)
#: after a string statement: the line's end, or a comment, or ``;``
PY_TAIL = re.compile(r"[ \t]*(?:[#;\n]|\\?$)")
LEAN_TOKEN = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])\'|--|/-|-/')
LEAN_BLOCK = re.compile(r"/-|-/")
SLASH_COMMENT = re.compile(r"(?<![:\\])//[^\n]*|/\*.*?\*/", re.DOTALL)
HASH_COMMENT = re.compile(r"(?:^|(?<=\s))#[^\n]*", re.MULTILINE)


def holds(path: str) -> bool:
    """Whether this lint reads ``path``: prose it holds, outside ``LEFT_OUT``, ``SKIP`` and test fixtures."""
    if path.startswith(LEFT_OUT) or any(path == s or s.endswith("/") and path.startswith(s) for s in SKIP):
        return False
    if any(d in FIXTURE_DIRS for d in path.split("/")[:-1]):
        return False
    return path.endswith(PY + LEAN + MARKDOWN + HASH + SLASH)


def found_in(prose: list[tuple[int, str]], owners: bool = False) -> list[tuple[int, str]]:
    """(line, ``kind: text``) of each site in ``prose``, its chunks' (first line, text); ``owners``: handles name owners
    here."""
    text = "\n".join(chunk for _, chunk in prose)
    if not text:
        return []
    hits = []
    for rank, (kind, pattern, before) in enumerate(PATTERNS):
        for m in pattern.finditer(text):
            if not (before and m.start() and before.match(text, m.start() - 1)):
                hits.append((m.start(), rank, m.end(), kind, m.group()))
    starts, at = [], 0
    for _, chunk in prose:
        starts.append(at)
        at += len(chunk) + 1
    out, end = [], 0
    for start, _, stop, kind, what in sorted(hits):
        if start < end or owners and kind == "codename" and HANDLE.match(what):
            continue
        i = bisect.bisect_right(starts, start) - 1
        out.append((prose[i][0] + text.count("\n", starts[i], start), f"{kind}: {what.strip()[:80]}"))
        end = stop
    return out


def python_prose(text: str) -> list[tuple[int, str]]:
    """(first line, text) of each comment and string statement (a docstring, or a string that documents what precedes it),
    read by one regular expression over comments and strings, the code between them counted for bracket depth: as
    ``tokenized`` reads them, which is the reference and five times slower."""
    out: list[tuple[int, str]] = []
    depth, at, line = 0, 0, 1
    for m in PY_LEX.finditer(text):
        code = text[at:m.start()]
        depth = max(0, depth + sum(map(code.count, "([{")) - sum(map(code.count, ")]}")))
        line += text.count("\n", at, m.start())
        at = m.end()
        s = m.group()
        if s[0] == "#":
            out.append((line, s))
        elif depth == 0 and not text[text.rfind("\n", 0, m.start()) + 1:m.start()].strip(" \trRuU") \
                and PY_TAIL.match(text, m.end()):
            out.append((line, s))
        line += s.count("\n")
    return out


def tokenized(text: str) -> list[tuple[int, str]]:
    """``python_prose`` by ``tokenize``."""
    out: list[tuple[int, str]] = []
    strings: list[tokenize.TokenInfo] = []
    start, depth = True, 0
    for t in tokenize.generate_tokens(io.StringIO(text).readline):
        if t.type == tokenize.COMMENT or t.type == tokenize.NL and depth == 0 and not strings:
            out += [(t.start[0], t.string)] * (t.type == tokenize.COMMENT)
            continue
        if t.type == tokenize.STRING and depth == 0 and (start or strings):
            strings.append(t)
            continue
        if strings and t.type in (tokenize.NEWLINE, tokenize.ENDMARKER, tokenize.NL):
            out += [(s.start[0], s.string) for s in strings]
        if t.type == tokenize.OP and t.string in "([{":
            depth += 1
        elif t.type == tokenize.OP and t.string in ")]}":
            depth = max(0, depth - 1)
        strings = []
        start = depth == 0 and t.type in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.NL)
    return out


def lean_prose(text: str) -> list[tuple[int, str]]:
    """(first line, text) of each Lean comment: ``--`` to the line's end, and ``/- -/`` (``/--``, ``/-!``), nested."""
    out, at, line, counted = [], 0, 1, 0
    while m := LEAN_TOKEN.search(text, at):
        if m.group() == "--":
            end = text.find("\n", m.start())
            end = len(text) if end < 0 else end
        elif m.group() == "/-":
            depth, end = 1, m.end()
            while depth and (b := LEAN_BLOCK.search(text, end)):
                depth, end = depth + (1 if b.group() == "/-" else -1), b.end()
            end = len(text) if depth else end
        else:
            at = m.end()
            continue
        line, counted = line + text.count("\n", counted, m.start()), m.start()
        out.append((line, text[m.start():end]))
        at = end
    return out


def comments(text: str, pattern: re.Pattern) -> list[tuple[int, str]]:
    out, line, counted = [], 1, 0
    for m in pattern.finditer(text):
        line, counted = line + text.count("\n", counted, m.start()), m.start()
        out.append((line, m.group()))
    return out


def sites(text: str, path: str) -> list[tuple[int, str]]:
    """(line, ``kind: text``) of each site of lane text in ``path``'s prose, ``text`` its content."""
    owners = path == "AGENTS.md" or path.startswith(OWNERS[1:])
    if path.endswith(MARKDOWN):
        prose = [(1, text)]
    elif path.endswith(PY):
        prose = python_prose(text)
    elif path.endswith(LEAN):
        prose = lean_prose(text)
    elif path.endswith(HASH):
        prose = comments(text, HASH_COMMENT)
    elif path.endswith(SLASH):
        prose = comments(text, SLASH_COMMENT)
    else:
        return []
    return sorted(found_in(sorted(prose, key=lambda c: c[0]), owners), key=lambda s: s[0])


def parse(doc: object) -> Counter:
    """Each file's allowed sites, from a list's document."""
    assert isinstance(doc, dict) and isinstance(doc.get("entries"), list), f"{LIST.name}: {{'entries': [...]}} expected"
    counts: Counter = Counter()
    for e in doc["entries"]:
        assert isinstance(e, dict) and set(e) == {"file", "count"} and isinstance(e["file"], str), \
            f"{LIST.name}: an entry is {{file, count}}: {e}"
        assert type(e["count"]) is int and e["count"] >= 1, f"{LIST.name}: count must be a positive int: {e}"
        assert e["file"] not in counts, f"{LIST.name}: duplicate entry {e}"
        assert not pinned(e["file"]), f"{LIST.name}: {e['file']} is pinned at zero (PINNED), so it has no entry: {e}"
        counts[e["file"]] = e["count"]
    return counts


def pinned(path: str) -> bool:
    return any(path == p or p.endswith("/") and path.startswith(p) for p in PINNED)


def files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    return sorted(p for p in out.decode().split("\0") if p and holds(p))


@lru_cache(maxsize=None)
def scan() -> tuple[tuple[str, int, str], ...]:
    """Each site's file, line and text."""
    return tuple((f, line, what) for f in files() if (ROOT / f).is_file()
                 for line, what in sites((ROOT / f).read_text(encoding="utf-8", errors="replace"), f))


def allowed() -> Counter:
    return parse(json.loads(LIST.read_text(encoding="utf-8")))


def found() -> Counter:
    return Counter(f for f, *_ in scan())


def test_no_new_lane_text() -> None:
    have = allowed()
    where: dict[str, list[tuple[int, str]]] = {}
    for f, line, what in scan():
        where.setdefault(f, []).append((line, what))
    new = [f"  {f}: {n} found, {have[f]} allowed" + (" (pinned at zero)" if pinned(f) else "") + "\n"
           + "\n".join(f"    {f}:{line} {what}" for line, what in where[f]) for f, n in sorted(found().items()) if n > have[f]]
    assert not new, (f"lane text: {len(new)} file(s) hold more than tests/lint/allowlists/{NAME}.json allows. {FIX}\n"
                     + "\n".join(new))


def test_what_is_lane_text_and_what_is_not() -> None:
    py = ('"""Built for M-0451 (Daniel, 18 Oct), see #1826 and VLLM-136."""\n'
          "import re  # bc-5c763a25: from note:20261016T0219Z-report-premise-ledger\n"
          "X = '(12 Oct) #1234 is data in a literal'\n"
          '"""(the vLLM coordinator, 27 Oct 06:20Z, CI-198) R17 and [lane tp2]."""\n'
          "def f():\n"
          "    '''2026-10-12, https://www.notion.so/x, 1729000000.123456, lane vllm-quality-rollout, @vllm.'''\n"
          "    return SHA_512  # SHA-512, F-r16, art:87f1, R1, @property, a@vllm.dev\n")
    assert [(line, w.split(":")[0]) for line, w in sites(py, "a.py")] == [
        (1, "codename"), (1, "attribution"), (1, "pr"), (1, "linear"), (2, "agent"), (2, "note"), (4, "attribution"),
        (4, "codename"), (4, "codename"), (6, "date"), (6, "notion"), (6, "slack"), (6, "codename"), (6, "codename"),
        (7, "codename")]
    assert sites("m = (\n    '(12 Oct) in a call'\n    'and #1234'\n)\nf(x,\n  '''VLLM-1''')\n", "b.py") == []
    lean = ("/-- Lean's (Daniel, 8 Oct)\n  /- nested -/ still (12 Oct) -/\n"
            'def s := "-- (12 Oct) is a string"  -- PRF-12\n/-! #4040 -/\n')
    assert [(line, w.split(":")[0]) for line, w in sites(lean, "A.lean")] == [
        (1, "attribution"), (2, "attribution"), (3, "linear"), (4, "pr")]
    md = "# Owners\n\n@vllm owns it (Daniel, 8 Oct); see https://linear.app/x and #5040.\n"
    assert [w.split(":")[0] for _, w in sites(md, "README.md")] == ["codename", "attribution", "linear", "pr"]
    assert [w.split(":")[0] for _, w in sites(md, ".agents/skills/x/SKILL.md")] == ["attribution", "linear", "pr"]
    assert sites("int x;  // INF-71 (Oct 12)\nchar *u = \"https://x\";\n", "k.cu") == [
        (1, "linear: INF-71"), (1, "attribution: (Oct 12)")]
    assert sites("x = 1  # (12 Oct)\n", "a.toml") == [(1, "attribution: (12 Oct)")]
    for src in (py, Path(__file__).read_text(encoding="utf-8"), (ROOT / "integrations/vllm/tools/ratchet_gate.py").read_text()):
        assert found_in(python_prose(src)) == found_in(tokenized(src))
    assert sites("#5040 (12 Oct)", "data.json") == []
    assert not holds("archive/x.md") and not holds("fixtures/a/README.md") and not holds("x/tests/fixtures/y.py")
    assert holds("x/tests/test_y.py") and holds("Security/A.lean") and not holds("x.csv")


def test_pinned_files_hold_none() -> None:
    assert not [f for f in found() if pinned(f)], [s for s in scan() if pinned(s[0])]


def test_the_list_is_well_formed_and_an_entry_moves_with_its_file() -> None:
    assert parse({"entries": [{"file": "a.py", "count": 2}]}) == Counter({"a.py": 2})
    for bad in ({"file": "a.py", "count": 0}, {"file": "a.py", "count": True}, {"file": "a.py", "registry": True},
                {"file": "a.py", "count": 1, "rule": "x"}, {"file": "AGENTS.md", "count": 1},
                {"file": ".agents/skills/x/SKILL.md", "count": 1}):
        try:
            parse({"entries": [bad]})
        except AssertionError as e:
            assert str(e).startswith(f"{LIST.name}: "), e
        else:
            raise AssertionError(f"{bad} parsed")
    assert moved(Counter({"a/x.py": 2, "a/y.md": 1, "b.py": 1}), "a", "c") == Counter({"c/x.py": 2, "c/y.md": 1, "b.py": 1})
    assert moved(Counter({"a.py": 2, "b.py": 1}), "a.py", "b.py") == Counter({"b.py": 3})


def moved(counts: Counter, old: str, new: str) -> Counter:
    """``counts`` with the file ``old``'s, or the directory ``old``'s files', under ``new``: a count already there adds."""
    old, new = old.rstrip("/"), new.rstrip("/")
    out: Counter = Counter()
    for f, n in counts.items():
        out[new + f[len(old):] if f == old or f.startswith(f"{old}/") else f] += n
    return out


def dump(counts: Counter) -> None:
    lines = [f"    {json.dumps({'file': f, 'count': n}, ensure_ascii=False)}" for f, n in sorted(counts.items()) if n > 0]
    LIST.write_text(f'{{\n  "rule": "lane text in prose (tests/lint/test_{NAME}.py)",\n  "entries": [\n'
                    + ",\n".join(lines) + "\n  ]\n}\n", encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--move"] and len(sys.argv) == 4:
        have = allowed()
        after = moved(have, *sys.argv[2:])
        dump(after)
        print(f"{sum(1 for f in have if f not in after)} entries moved from {sys.argv[2]} to {sys.argv[3]}")
    elif sys.argv[1:] == ["--prune"]:
        dump(allowed() & found())
    elif sys.argv[1:] == ["--record"]:
        dump(Counter({f: n for f, n in found().items() if not pinned(f)}))
    else:
        sys.exit(__doc__.rsplit("\n\n", 1)[-1])
