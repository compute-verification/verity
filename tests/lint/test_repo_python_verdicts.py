"""The places where Python can accept, held to ``allowlists/repo_python_verdicts.json``: a reading's list, at a commit, of
where a Python check's accept is what gets acted on, kept so that each place stays where it was read until its owner
switches its verdict to the Lean verifier and deletes its entry.

A place is one or more sites, and a site is a file, the function a line is in and that line's text: an entry ``{file, kind,
symbol, detail, owner, what, digest}``.  ``kind`` names the place, and the entries of one place share ``owner`` (a handle of
``.agents/skills/using-slack/registry.json``) and ``what`` (what Python decides there).  ``symbol`` is the innermost ``def``
around the line, dotted through its classes and outer functions (``Verifier.session``), or ``""`` where no ``def`` is
around it or the file isn't Python.  ``detail`` is the line, stripped at both ends.  ``digest`` is the first 16 hex digits
of SHA-256 over the function's lines, from its ``def`` line to its last (no decorator), each stripped on the right and
joined by newlines, or over ``detail`` where ``symbol`` is ``""``.  The first four fields are the key the lists' merge
driver reads (``adapters/vllm/tests/lint/merge_allowlist.py``), so two changes that each delete an entry merge.

Each entry's line is one of its tracked file's lines exactly once, or the suite fails: the list only shrinks, and a line
that left its file says nothing about whether Python still decides there, which only a reader of the change can say.  A
listed function that is no longer the code that was read (its ``symbol`` or ``digest`` differs) warns, as bookkeeping: it
means "read this again", never "Python no longer decides here".  The ratchet gate (``adapters/vllm/tools/ratchet_gate.py``)
reports a change that grows the list and holds nothing; so this module imports only the standard library and
``test_no_newly_missing_site`` takes no fixture.

What it does not see: it looks for no place outside the list, so a Python check that accepts elsewhere, or a new caller of
a listed check, passes.  The list is what a reading found, from a text search and the entry points it followed, and is not
shown complete.  Deleting an entry doesn't show its place gone, and a listed entry doesn't show its place correct, safe,
or the only Python on its verdict's path.  It reads text, not what runs.

    python3 tests/lint/test_repo_python_verdicts.py --record [PLACE...]   # symbol and digest from this tree; sorts the list
    python3 tests/lint/test_repo_python_verdicts.py --move OLD NEW        # a file (or directory) moved: its entries follow it
    python3 tests/lint/test_repo_python_verdicts.py --prune               # delete the entries whose file is no longer tracked
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

NAME = "repo_python_verdicts"
ROOT = Path(__file__).resolve().parents[2]
LIST = Path(__file__).resolve().parent / "allowlists" / f"{NAME}.json"
REGISTRY = ROOT / ".agents/skills/using-slack/registry.json"
RULE = f"places where Python can accept (tests/lint/test_{NAME}.py)"
FIELDS = ("file", "kind", "symbol", "detail", "owner", "what", "digest")
SELF = f"python3 tests/lint/test_{NAME}.py"
FIX = (f"tests/lint/allowlists/{NAME}.json lists where Python can accept, and only shrinks. If the Lean verifier's verdict "
       "is now the one acted on here, delete the entry and name in the pull request the executable that decides. If the "
       "line was reworded and Python still decides, put its new text in the entry and run "
       f"`{SELF} --record`. If the file moved: `{SELF} --move OLD NEW`. If the file was deleted: `{SELF} --prune`.")

Read = Callable[[str], str | None]


def tracked(root: Path = ROOT) -> set[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout
    return {p for p in out.decode().split("\0") if p}


def reader(rev: str | None = None, root: Path = ROOT) -> Read:
    """The text of a tracked file, or None: in the worktree, or at ``rev`` through ``git show``."""
    if rev is None:
        files = tracked(root)
        return lambda p: (root / p).read_text(encoding="utf-8", errors="replace") if p in files and (root / p).is_file() \
            else None

    def at(p: str) -> str | None:
        r = subprocess.run(["git", "show", f"{rev}:{p}"], cwd=root, capture_output=True)
        return r.stdout.decode("utf-8", errors="replace") if r.returncode == 0 else None
    return at


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def functions(source: str) -> list[tuple[int, int, str]]:
    """(first line, last line, dotted name) of each ``def`` in ``source``."""
    spans, todo = [], [(ast.parse(source), ())]
    while todo:
        node, outer = todo.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                spans.append((child.lineno, child.end_lineno, ".".join((*outer, child.name))))
                todo.append((child, (*outer, child.name)))
            elif isinstance(child, ast.ClassDef):
                todo.append((child, (*outer, child.name)))
            else:
                todo.append((child, outer))
    return spans


def sites(source: str, path: str, text: str) -> list[tuple[int, str | None, str | None]]:
    """(line, symbol, digest) of each line of ``source`` whose stripped text is ``text``; symbol and digest are None in a
    Python file that doesn't parse."""
    lines = source.split("\n")
    hits = [i + 1 for i, line in enumerate(lines) if line.strip() == text]
    try:
        spans = functions(source) if path.endswith(".py") else []
    except (SyntaxError, ValueError):
        return [(h, None, None) for h in hits]
    out = []
    for h in hits:
        around = [s for s in spans if s[0] <= h <= s[1]]
        if around:
            first, last, name = min(around, key=lambda s: s[1] - s[0])
            out.append((h, name, digest("\n".join(line.rstrip() for line in lines[first - 1:last]))))
        else:
            out.append((h, "", digest(text)))
    return out


def parse(doc: object, handles: set[str]) -> list[dict]:
    """The list's entries, from its document; refuses a malformed one with a message that starts with the list's name."""
    assert isinstance(doc, dict) and set(doc) == {"rule", "entries"} and isinstance(doc["entries"], list), \
        f"{LIST.name}: {{'rule': ..., 'entries': [...]}} expected"
    assert doc["rule"] == RULE, f"{LIST.name}: the rule is {RULE!r}, not {doc['rule']!r}"
    seen: set[tuple[str, str]] = set()
    places: dict[str, tuple[str, str]] = {}
    for e in doc["entries"]:
        assert isinstance(e, dict) and set(e) == set(FIELDS) and all(isinstance(e[f], str) for f in FIELDS), \
            f"{LIST.name}: an entry is {{{', '.join(FIELDS)}}}, all strings: {e}"
        assert all(e[f] for f in FIELDS if f != "symbol"), f"{LIST.name}: only `symbol` may be empty: {e}"
        assert re.fullmatch(r"[0-9a-f]{16}", e["digest"]), f"{LIST.name}: `digest` is 16 lower-case hex digits: {e}"
        assert (e["file"], e["detail"]) not in seen, f"{LIST.name}: duplicate entry {e}"
        seen.add((e["file"], e["detail"]))
        assert places.setdefault(e["kind"], (e["owner"], e["what"])) == (e["owner"], e["what"]), \
            f"{LIST.name}: the entries of the place {e['kind']!r} differ in `owner` or `what`: {e}"
        assert e["owner"] in handles, f"{LIST.name}: `owner` {e['owner']!r} is no handle of {REGISTRY.name}: {e}"
    return doc["entries"]


def registry_handles() -> set[str]:
    return {h["handle"] for h in json.loads(REGISTRY.read_text(encoding="utf-8"))["handles"]}


def load() -> list[dict]:
    return parse(json.loads(LIST.read_text(encoding="utf-8")), registry_handles())


def missing(entries: list[dict], read: Read) -> list[str]:
    """Each entry whose line is not one of its tracked file's lines exactly once, with why."""
    out = []
    for e in entries:
        source = read(e["file"])
        n = None if source is None else sum(1 for line in source.split("\n") if line.strip() == e["detail"])
        if n != 1:
            where = "is in no tracked file" if n is None else f"is in the file {n} times, not once"
            out.append(f"  {e['file']}: the line `{e['detail']}` of the place `{e['kind']}` ({e['owner']}'s: {e['what']}) "
                       f"{where}.")
    return out


def changed(entries: list[dict], read: Read) -> list[str]:
    """Each entry whose line stands once and whose function is not the code that was read."""
    out = []
    for e in entries:
        source = read(e["file"])
        got = [] if source is None else sites(source, e["file"], e["detail"])
        if len(got) != 1:
            continue
        _, symbol, now = got[0]
        if (symbol, now) != (e["symbol"], e["digest"]):
            name = e["symbol"] or "the line"
            moved = "" if symbol == e["symbol"] else f", now in {symbol!r}" if symbol is not None else ", which doesn't parse"
            out.append(f"  {e['kind']} ({e['owner']}): {name} in {e['file']} is not the code that was read (digest "
                       f"{e['digest']}, now {now}{moved}).")
    return out


def record(entries: list[dict], read: Read, places: set[str] | None = None) -> tuple[list[dict], list[str]]:
    """(the entries with ``symbol`` and ``digest`` from ``read`` for every place, or those named, sorted by owner, place,
    file and line; each entry whose line isn't in its file once, left as it was)."""
    out, bad = [], []
    for e in entries:
        source = read(e["file"])
        got = [] if source is None else sites(source, e["file"], e["detail"])
        if len(got) != 1 or got[0][1] is None:
            bad.append(f"  {e['file']}: `{e['detail']}` ({e['kind']}): "
                       + ("unparseable" if got and got[0][1] is None else f"{len(got)} times, not once"))
        elif places is None or e["kind"] in places:
            e = {**e, "symbol": got[0][1], "digest": got[0][2]}
        out.append((got[0][0] if len(got) == 1 else 0, e))
    out.sort(key=lambda pair: (pair[1]["owner"], pair[1]["kind"], pair[1]["file"], pair[0]))
    return [e for _, e in out], bad


def dotted(path: str) -> str:
    return path.removesuffix(".py").replace("/", ".")


def moved(entries: list[dict], old: str, new: str) -> list[dict]:
    """``entries`` with the file ``old``, or the directory ``old``'s files, under ``new``, and ``old`` inside each
    ``detail`` too, as a path and as a dotted module name."""
    old, new = old.rstrip("/"), new.rstrip("/")
    out = []
    for e in entries:
        f = new + e["file"][len(old):] if e["file"] == old or e["file"].startswith(f"{old}/") else e["file"]
        d = e["detail"]
        for a, b in ((old, new), (dotted(old), dotted(new))):
            d = re.sub(rf"(?<![\w./]){re.escape(a)}(?!\w)", lambda _m, b=b: b, d)
        out.append({**e, "file": f, "detail": d})
    return out


def pruned(entries: list[dict], files: set[str]) -> list[dict]:
    """``entries`` less those whose file is not in ``files``: a line gone from a file that stands is never pruned."""
    return [e for e in entries if e["file"] in files]


def dump(entries: list[dict]) -> str:
    lines = [f"    {json.dumps({f: e[f] for f in FIELDS}, ensure_ascii=False)}" for e in entries]
    return f'{{\n  "rule": {json.dumps(RULE)},\n  "entries": [\n' + ",\n".join(lines) + "\n  ]\n}\n"


def test_no_newly_missing_site() -> None:
    bad = missing(load(), reader())
    assert not bad, f"places where Python can accept: {len(bad)} listed site(s) don't stand. {FIX}\n" + "\n".join(bad)


def test_no_listed_function_changed_since_it_was_read() -> None:
    bad = changed(load(), reader())
    assert not bad, (f"places where Python can accept: {len(bad)} listed function(s) changed since they were read. Nothing "
                     "fails. Its owner or lean reads each again: if Python no longer decides there, delete the entry; else "
                     f"`{SELF} --record PLACE`.\n" + "\n".join(bad))


def test_the_list_is_well_formed_and_an_entry_moves_with_its_file() -> None:
    good = {"file": "a/b.py", "kind": "p", "symbol": "", "detail": "x = 1", "owner": "lean", "what": "w",
            "digest": "0123456789abcdef"}
    assert parse({"rule": RULE, "entries": [good]}, {"lean"}) == [good]
    for bad in ({"rule": RULE}, {"rule": "x", "entries": [good]}, {"rule": RULE, "entries": [good], "more": 1},
                {"rule": RULE, "entries": [{**good, "count": 1}]}, {"rule": RULE, "entries": [{**good, "what": ""}]},
                {"rule": RULE, "entries": [{**good, "digest": "0123456789ABCDEF"}]},
                {"rule": RULE, "entries": [{**good, "symbol": None}]}, {"rule": RULE, "entries": [good, good]},
                {"rule": RULE, "entries": [good, {**good, "detail": "y = 2", "what": "v"}]},
                {"rule": RULE, "entries": [{**good, "owner": "nobody"}]}):
        try:
            parse(bad, {"lean"})
        except AssertionError as e:
            assert str(e).startswith(f"{LIST.name}: "), e
        else:
            raise AssertionError(f"{bad} parsed")
    parse(json.loads(LIST.read_text(encoding="utf-8")), registry_handles())
    assert dump(parse(json.loads(dump([good])), {"lean"})) == dump([good])

    es = [{**good, "file": "a/b.py", "detail": 'run("a/b.py")'}, {**good, "file": "a/c.sh", "detail": "from a.b import f"},
          {**good, "file": "a/bc.py", "detail": "a/bc.py a.bc"}, {**good, "file": "d.py", "detail": "x"}]
    assert [(e["file"], e["detail"]) for e in moved(es, "a/b.py", "z/y.py")] == [
        ("z/y.py", 'run("z/y.py")'), ("a/c.sh", "from z.y import f"), ("a/bc.py", "a/bc.py a.bc"), ("d.py", "x")]
    assert [(e["file"], e["detail"]) for e in moved(es, "a/", "q")] == [
        ("q/b.py", 'run("q/b.py")'), ("q/c.sh", "from q.b import f"), ("q/bc.py", "q/bc.py q.bc"), ("d.py", "x")]
    assert pruned(es, {"a/b.py", "a/bc.py", "d.py"}) == [es[0], es[2], es[3]]
    gone = {**good, "file": "d.py", "detail": "y = 2"}
    assert pruned([gone], {"d.py"}) == [gone] and missing([gone], {"d.py": "x = 1\n"}.get) != []


def test_what_a_site_is() -> None:
    src = ("import os\n"
           "X = 1\n"
           "class C:\n"
           "    def f(self):\n"
           "        return X  # one\n"
           "def f():\n"
           "    def g():\n"
           "        return 2\n"
           "    return g\n"
           "async def h():\n"
           "    await go()\n"
           "@decorated\n"
           "def k():\n"
           "    twice()\n"
           "    twice()\n"
           "def m():\n"
           "    if True:\n"
           "        indented = 1\n")
    one = sites(src, "a.py", "return X  # one")
    assert [(n, s) for n, s, _ in one] == [(5, "C.f")]
    assert [(n, s) for n, s, _ in sites(src, "a.py", "return 2")] == [(8, "f.g")]
    assert [(n, s) for n, s, _ in sites(src, "a.py", "await go()")] == [(11, "h")]
    assert [(n, s) for n, s, _ in sites(src, "a.py", "twice()")] == [(14, "k"), (15, "k")]
    assert [(n, s) for n, s, _ in sites(src, "a.py", "indented = 1")] == [(18, "m")]
    assert sites(src, "a.py", "X = 1") == [(2, "", digest("X = 1"))]
    assert sites(src, "a.py", "X = 2") == []
    k = sites(src, "a.py", "twice()")[0][2]
    assert k == digest("def k():\n    twice()\n    twice()")
    assert sites("echo hi\nexit 0\n", "a.sh", "exit 0") == [(2, "", digest("exit 0"))]
    assert sites("def f(:\n    return 1\n", "a.py", "return 1") == [(2, None, None)]

    entry = {"file": "a.py", "kind": "p", "symbol": "C.f", "detail": "return X  # one", "owner": "lean", "what": "w",
             "digest": one[0][2]}
    assert changed([entry], {"a.py": src}.get) == [] and missing([entry], {"a.py": src}.get) == []
    edited_inside = src.replace("        return X  # one", "        # a comment\n        return X  # one")
    assert len(changed([entry], {"a.py": edited_inside}.get)) == 1 and missing([entry], {"a.py": edited_inside}.get) == []
    edited_outside = src.replace("X = 1", "X = 2")
    assert changed([entry], {"a.py": edited_outside}.get) == []
    assert "now in 'f'" in changed([{**entry, "detail": "return g"}], {"a.py": src}.get)[0]
    unparsed = src + "def broken(:\n"
    assert "doesn't parse" in changed([entry], {"a.py": unparsed}.get)[0] and missing([entry], {"a.py": unparsed}.get) == []
    assert "2 times" in missing([{**entry, "detail": "twice()"}], {"a.py": src}.get)[0]
    assert "0 times" in missing([{**entry, "detail": "thrice()"}], {"a.py": src}.get)[0]
    assert "no tracked file" in missing([entry], {}.get)[0]
    again, bad = record([{**entry, "symbol": "", "digest": "0" * 16}], {"a.py": src}.get)
    assert again == [entry] and bad == []


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--record"]:
        after, bad = record(load(), reader(), set(args[1:]) or None)
        LIST.write_text(dump(after), encoding="utf-8")
        print(f"{len(after) - len(bad)} entries recorded" + (f"; {len(bad)} left as they were, not once in their file:\n"
                                                              + "\n".join(bad) if bad else ""))
        sys.exit(1 if bad else 0)
    elif args[:1] == ["--move"] and len(args) == 3:
        before = load()
        after = moved(before, *args[1:])
        LIST.write_text(dump(after), encoding="utf-8")
        print(f"{sum(a != b for a, b in zip(before, after))} entries moved from {args[1]} to {args[2]}")
    elif args == ["--prune"]:
        before = load()
        after = pruned(before, tracked())
        LIST.write_text(dump(after), encoding="utf-8")
        print(f"{len(before) - len(after)} entries pruned")
    else:
        sys.exit(__doc__.rsplit("\n\n", 1)[-1])
