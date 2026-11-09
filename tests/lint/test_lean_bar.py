"""Mathlib's bar for the tree's Lean, held to ``allowlists/lean_bar.json``: what Mathlib's style linters refuse of a file
(its ``header``, ``longFile`` and ``maxHeartbeats`` linters), found here by reading the text, with no Lean and no build.

Read: every ``.lean`` file under ``Security/`` (the security analysis) and ``verity/core/`` (the code packages), found by a
walk that leaves out ``.lake`` (``BUILT``: Lake's build trees and the packages it fetches, which git never tracks), so in a
clean checkout the tracked ones.  The walk starts no program, so the suite's read guard traces all this lint reads.  The
Lean check's own fixtures (``tools/verity/lean/tests/fixtures``) lie outside both trees, as do the Lean tools' files, the
benchmarks' and ``verity/ml``'s; under the two trees nothing is left out, a package's test modules included.

The rules, each a finding in code only, never in a comment or a string literal:

* ``module-doc``: the file has no module docstring (``/-! … -/``), which says what the file holds.
* ``long-file``: the file has more than ``LONG`` lines.  ``tests/test_lean_packages.py``'s ``OVERLONG`` holds another
  measure, the lines past 2,000 of each package summed, so that no long module grows; this rule holds how many files
  stand past Mathlib's 1,500, the files between the two lengths among them.
* ``heartbeats-unlimited``: ``maxHeartbeats`` set to ``0`` (``synthInstance.maxHeartbeats`` and any other option of that
  name too), which lets one declaration hang a build.
* ``heartbeats-uncommented``: ``maxHeartbeats`` set to any number with no comment that says why, on its line or on the
  line above, which then holds nothing but the comment.

Not held here: theorem names (Mathlib's snake_case is a rename campaign of its own) and ``linter.missingDocs`` (an option of
every package's lakefile, not a reading of the text).

Findings are counted per ``(file, rule)``, the key the ratchet gate (``integrations/vllm/tools/ratchet_gate.py``) pools a
list's counts by, so each rule's total is held apart.  An entry ``{"file", "rule", "count"}`` allows ``count`` findings of its
pair; ``module-doc`` and ``long-file`` find one a file.  The tree is held to the list and to nothing else: no base commit
is read.  A pair found more often than its entry allows, or found with no entry, fails, naming each site and its fix.  An
entry that allows more than is found only warns (``Stale``), naming the count to lower it to, so a change that fixes a
finding never fails for the entry it leaves behind, and ``--prune`` deletes and lowers what this tree no longer finds.

The list is written by this module, never by hand.  ``--record`` rewrites it to the tree's counts, for a merge that brings
files written before the bar held them: the merge commits the rewritten list once.  It lists no new ``maxHeartbeats 0``
(``HARD``): ``heartbeats-unlimited`` is allowed only where the list already has it, at most as often, so a new one fails
until it has a bound.  A file that moves takes its entries with it, in the same change (``--move``).

The ratchet gate runs ``test_no_new_findings`` as a change's base has it, outside pytest, when the change touches this
module or its list.  So the module imports only the standard library and that test takes no fixture.

    python3 tests/lint/test_lean_bar.py --counts          # each rule's findings, by package
    python3 tests/lint/test_lean_bar.py --move OLD NEW    # a file (or directory) moved: its entries follow it
    python3 tests/lint/test_lean_bar.py --prune           # delete and lower what this tree no longer finds
    python3 tests/lint/test_lean_bar.py --record          # the list is what this tree finds, but no new `maxHeartbeats 0`
"""
from __future__ import annotations

import json
import os
import re
import sys
import warnings
from collections import Counter
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

NAME = "lean_bar"
ROOT = Path(__file__).resolve().parents[2]
LIST = Path(__file__).resolve().parent / "allowlists" / f"{NAME}.json"
#: the trees held: the security analysis and the code packages
TREES = ("Security", "verity/core")
#: the directories a walk leaves out: Lake's build trees and fetched packages, Mathlib's sources among them
BUILT = (".lake",)
#: the lines a file may run to, Mathlib's ``linter.style.longFile``
LONG = 1500
FIELDS = ("file", "rule", "count")
#: the rule `--record` never adds an entry of, or raises one of
HARD = "heartbeats-unlimited"
#: each rule and its fix
RULES = {
    "module-doc": "Open the file with a module docstring after its imports: `/-! # Title`, then what the file holds and "
                  "why, closed by `-/`.",
    "long-file": f"Split it: move a group of its declarations to a module beside it and import that, so no file runs past "
                 f"{LONG} lines; a generator splits its output.",
    "heartbeats-unlimited": "Give it a bound: measure the declaration (`set_option profiler true in`, or `#count_heartbeats "
                            "in` where Mathlib is imported), fix the proof (`simp only [...]`, a split-out lemma, a smaller "
                            "`decide`) and set the number it then needs. No command lists a new one.",
    "heartbeats-uncommented": "Say why the declaration needs the bound in a `--` comment on the option's line or the line "
                              "above it.",
}
FIX = (f"Fix each as its rule says. Never edit tests/lint/allowlists/{NAME}.json by hand. If a merge brought these files, "
       f"`python3 tests/lint/test_{NAME}.py --record` rewrites the list to this tree's counts: commit it with the merge. It "
       f"lists no new `maxHeartbeats 0` ({HARD}), which fails until it has a bound. If a file moved, move its entries in "
       f"the same change: `python3 tests/lint/test_{NAME}.py --move OLD NEW`.")

#: where a comment or a string literal opens; a character literal of a quote opens none
_OPEN = re.compile(r"""--|/-|'"'|'\\"'|(?<![\w'.])r#*"|["]""")
_NEST = re.compile(r"/-|-/")
_QUOTED = re.compile(r'(?:[^"\\]|\\.)*"', re.DOTALL)
_INK = re.compile(r"[^\n]")
#: `maxHeartbeats`, or an option of that name under a prefix, given a number: `set_option NAME N`, a context's `NAME := N`
_OVERRIDE = re.compile(r"(?<![\w.])((?:\w+\.)*maxHeartbeats)(?:\s*:=\s*|\s+)(\d[\d_]*)")


class Stale(UserWarning):
    """An entry that allows more than the tree holds: a later change lowers it (``--prune``)."""


def spans(src: str) -> Iterator[tuple[str, int, int]]:
    """(kind, start, end) of each comment and string literal of Lean source ``src``, in order: ``line`` (``--`` to the
    line's end), ``block`` (``/- -/``, nested), ``doc`` (``/--``), ``module`` (``/-!``) and ``string`` (``"…"`` with its
    escapes, and a raw ``r"…"`` or ``r#"…"#``).  One left open runs to the end."""
    at, n = 0, len(src)
    while (m := _OPEN.search(src, at)) is not None:
        start, at = m.start(), m.end()
        if m[0] == "--":
            at = n if (end := src.find("\n", at)) < 0 else end
            yield "line", start, at
        elif m[0] == "/-":
            depth = 1
            while depth and (b := _NEST.search(src, at)) is not None:
                depth, at = depth + (1 if b[0] == "/-" else -1), b.end()
            at = n if depth else at
            yield ("doc" if src.startswith("/--", start) else "module" if src.startswith("/-!", start) else "block"), start, at
        elif m[0] == '"':
            at = n if (q := _QUOTED.match(src, at)) is None else q.end()
            yield "string", start, at
        elif m[0].startswith("r"):
            close = '"' + m[0][1:-1]
            at = n if (end := src.find(close, at)) < 0 else end + len(close)
            yield "string", start, at


def read(src: str) -> tuple[list[str], set[int]]:
    """The lines of ``src`` with its comments and string literals blanked to spaces, and the lines (from 1) a comment says
    something on: an empty ``--`` or the bare close of a block comment says nothing."""
    code, said, at, line, counted = [], set(), 0, 1, 0
    for kind, start, end in spans(src):
        code += [src[at:start], _INK.sub(" ", src[start:end])]
        at = end
        if kind != "string":
            line, counted = line + src.count("\n", counted, start), start
            said.update(line + k for k, text in enumerate(src[start:end].split("\n")) if text.strip(" \t-/!"))
    code.append(src[at:])
    return "".join(code).split("\n"), said


def findings(src: str) -> list[tuple[str, int, str]]:
    """(rule, line, what) of each finding in Lean source ``src``."""
    out = []
    lines = src.count("\n") + (1 if src and not src.endswith("\n") else 0)
    if "/-!" not in src or not any(kind == "module" for kind, _, _ in spans(src)):
        out.append(("module-doc", 1, "no `/-!` module docstring"))
    if lines > LONG:
        out.append(("long-file", LONG + 1, f"{lines} lines, {lines - LONG} past {LONG}"))
    if "maxHeartbeats" in src:
        code, said = read(src)
        for line, text in enumerate(code, 1):
            for m in _OVERRIDE.finditer(text) if "maxHeartbeats" in text else ():
                what = f"{m[1]} {m[2]}"
                if int(m[2].replace("_", "")) == 0:
                    out.append(("heartbeats-unlimited", line, what))
                if line not in said and not (line - 1 in said and not code[line - 2].strip()):
                    out.append(("heartbeats-uncommented", line, what))
    return out


def walk(root: Path) -> Iterator[tuple[str, list[str]]]:
    """(directory, its files' names) of each directory under ``TREES`` in ``root``, outside ``BUILT``, from ``root``."""
    for tree in TREES:
        for d, dirs, names in os.walk(root / tree):
            dirs[:] = sorted(x for x in dirs if x not in BUILT)
            yield Path(d).relative_to(root).as_posix(), names


def files(root: Path = ROOT) -> list[str]:
    """Each ``.lean`` file under ``TREES`` in ``root``, outside ``BUILT``."""
    return sorted(f"{d}/{n}" for d, names in walk(root) for n in names if n.endswith(".lean"))


@lru_cache(maxsize=None)
def scan(root: Path = ROOT) -> tuple[tuple[str, str, int, str], ...]:
    """Each finding's file, rule, line and text."""
    return tuple((f, rule, line, what) for f in files(root) if (root / f).is_file()
                 for rule, line, what in findings((root / f).read_text(encoding="utf-8", errors="replace")))


def parse(doc: object) -> Counter:
    """Each (file, rule)'s allowed findings, from a list's document."""
    assert isinstance(doc, dict) and isinstance(doc.get("entries"), list), f"{LIST.name}: {{'entries': [...]}} expected"
    out: Counter = Counter()
    for e in doc["entries"]:
        assert isinstance(e, dict) and set(e) == set(FIELDS) and isinstance(e["file"], str), \
            f"{LIST.name}: an entry is {{{', '.join(FIELDS)}}}: {e}"
        assert e["rule"] in RULES, f"{LIST.name}: no rule {e['rule']!r} ({', '.join(RULES)}): {e}"
        assert type(e["count"]) is int and e["count"] >= 1, f"{LIST.name}: count must be a positive int: {e}"
        k = (e["file"], e["rule"])
        assert k not in out, f"{LIST.name}: duplicate entry {e}"
        out[k] = e["count"]
    return out


def allowed() -> Counter:
    return parse(json.loads(LIST.read_text(encoding="utf-8")))


def found(root: Path = ROOT) -> Counter:
    return Counter((f, rule) for f, rule, *_ in scan(root))


def held(sites: tuple[tuple[str, str, int, str], ...], have: Counter) -> None:
    """The findings ``sites`` (`scan`) held to the list ``have``: a (file, rule) pair found more often than its entry
    allows, or with no entry, fails with its sites and its rule's fix; an entry that allows more than is found only warns
    (`Stale`), naming the count to lower it to."""
    got = Counter((f, rule) for f, rule, *_ in sites)
    stale = [f"  {f} [{rule}]: {n} allowed, {got[f, rule]} found: "
             + (f"lower its count to {got[f, rule]}" if got[f, rule] else "delete the entry")
             for (f, rule), n in sorted(have.items()) if n > got[f, rule]]
    if stale:
        warnings.warn(Stale(f"the Lean bar: {len(stale)} of tests/lint/allowlists/{NAME}.json's entries allow more than the "
                            f"tree holds; `python3 tests/lint/test_{NAME}.py --prune` lowers them:\n" + "\n".join(stale)),
                      stacklevel=2)
    where: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for f, rule, line, what in sites:
        where.setdefault((f, rule), []).append((line, what))
    new = [f"  {f} [{rule}]: {n} found, {have[f, rule]} allowed. {RULES[rule]}\n"
           + "\n".join(f"    {f}:{line} {what}" for line, what in sorted(where[f, rule]))
           for (f, rule), n in sorted(got.items()) if n > have[f, rule]]
    assert not new, (f"the Lean bar: {len(new)} (file, rule) pair(s) found more often than tests/lint/allowlists/{NAME}.json "
                     f"allows. {FIX}\n" + "\n".join(new))


def test_no_new_findings() -> None:
    held(scan(), allowed())


def _rules(src: str) -> list[tuple[str, int]]:
    return [(rule, line) for rule, line, _ in findings(src)]


def test_what_each_rule_finds() -> None:
    doc = "/-! # A file\nWhat it holds. -/\n"
    assert _rules(doc + "def a := 1\n") == []
    assert _rules("module\npublic import B\n\n/-- a -/\ndef a := 1\n") == [("module-doc", 1)]
    assert _rules('-- /-! not one -/\ndef s := "/-! nor this -/"\n/- /-! nor a nested one -/ -/\n') == [("module-doc", 1)]
    assert _rules(doc + "x\n" * (LONG - 2)) == [] and _rules(doc + "x\n" * (LONG - 2) + "x") == [("long-file", LONG + 1)]
    assert findings(doc + "x\n" * LONG)[0][2] == f"{LONG + 2} lines, 2 past {LONG}"
    src = (doc + "def z := 0\n"
           + "set_option maxHeartbeats 400000 in\n"                                   # 4: no comment
           + "theorem a : True := trivial\n"
           + "-- the kernel evaluates the whole table\n"
           + "set_option maxHeartbeats 400000 in\n"                                   # 7: a comment above
           + "theorem b : True := trivial\n"
           + "set_option synthInstance.maxHeartbeats 100000 -- instance search over the product\n"   # 9: one on its line
           + "set_option maxHeartbeats 0 in\n"                                        # 10: unlimited; the line above holds code
           + "theorem c : True := trivial\n"
           + "/- the walk steps through every\n   branch of the parser -/\n"
           + "set_option synthInstance.maxHeartbeats 0 in\n"                          # 14: unlimited, with a comment above
           + "theorem d : True := trivial\n"
           + "--\n"
           + "set_option maxHeartbeats 1_000_000 in\n"                                # 17: an empty comment says nothing
           + "theorem e : True := trivial\n"
           + "-- why\n\n"
           + "set_option maxHeartbeats 200000 in\n"                                   # 21: a comment two lines above
           + "theorem f : True := trivial\n")
    assert _rules(src) == [("heartbeats-uncommented", 4), ("heartbeats-unlimited", 10), ("heartbeats-uncommented", 10),
                           ("heartbeats-unlimited", 14), ("heartbeats-uncommented", 17), ("heartbeats-uncommented", 21)]
    assert [w for _, _, w in findings(src)][1:4] == ["maxHeartbeats 0", "maxHeartbeats 0", "synthInstance.maxHeartbeats 0"]
    quiet = (doc + "-- set_option maxHeartbeats 0 in\n/-- as `set_option maxHeartbeats 0` would -/\n"
             'def s := "set_option maxHeartbeats 0"\ndef r := r#"a "quoted" maxHeartbeats 0"#\n'
             "def c := '\"'\ndef maxHeartbeatsOf := 0\n")
    assert _rules(quiet) == []
    assert _rules(doc + "def f := withReader ({ · with maxHeartbeats := 0 })  -- unbounded on purpose\n") == [
        ("heartbeats-unlimited", 3)]


def test_it_reads_comments_and_strings_as_lean_does() -> None:
    src = 'a -- b "c\n/- d /- e -/ f -/ g "h -- i \\" j" k /-- l -/ r#"m " n"# o /-! p'
    assert [(kind, src[s:e]) for kind, s, e in spans(src)] == [
        ("line", '-- b "c'), ("block", "/- d /- e -/ f -/"), ("string", '"h -- i \\" j"'), ("doc", "/-- l -/"),
        ("string", 'r#"m " n"#'), ("module", "/-! p")]
    assert read("a -- b\n/- c\n-/\nd\n--\n") == (["a     ", "    ", "  ", "d", "  ", ""], {1, 2})


def _tree(root: Path) -> None:
    for path, src in (("Security/Proofs/A.lean", "set_option maxHeartbeats 0 in\ntheorem a : True := trivial\n"),
                      ("Security/Proofs/B.lean", "/-! # B -/\n" + "x\n" * LONG),
                      ("verity/core/c/C.lean", "/-! # C -/\n-- the table is large\nset_option maxHeartbeats 9 in\ndef c := 1\n"),
                      ("verity/core/c/.lake/packages/m/M.lean", "set_option maxHeartbeats 0\n"),
                      ("verity/ml/D.lean", "set_option maxHeartbeats 0\n"),
                      ("verity/core/c/README.md", "set_option maxHeartbeats 0\n")):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(src, encoding="utf-8")


def test_an_unlisted_finding_fails_a_listed_one_passes_and_a_stale_entry_only_warns(tmp_path: Path) -> None:
    _tree(tmp_path)
    a, b = "Security/Proofs/A.lean", "Security/Proofs/B.lean"
    assert files(tmp_path) == [a, b, "verity/core/c/C.lean"]
    exact = Counter({(a, "module-doc"): 1, (a, "heartbeats-unlimited"): 1, (a, "heartbeats-uncommented"): 1, (b, "long-file"): 1})
    assert found(tmp_path) == exact
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        held(scan(tmp_path), exact)

    def fails(have: Counter) -> str:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                held(scan(tmp_path), have)
        except AssertionError as e:
            return str(e)
        raise AssertionError(f"{dict(have)} passed")
    for rule in RULES:
        (k,) = [k for k in exact if k[1] == rule]
        why = fails(exact - Counter({k: 1}))
        assert "1 (file, rule) pair(s)" in why and f"  {k[0]} [{rule}]: 1 found, 0 allowed. {RULES[rule]}" in why, why
        assert "by hand" in why and f"test_{NAME}.py --record" in why and f"test_{NAME}.py --move OLD NEW" in why, why
    assert f"    {a}:1 maxHeartbeats 0" in fails(exact - Counter({(a, "heartbeats-unlimited"): 1}))
    assert f"    {b}:{LONG + 1} {LONG + 1} lines, 1 past {LONG}" in fails(exact - Counter({(b, "long-file"): 1}))

    loose = exact + Counter({(a, "heartbeats-uncommented"): 2, ("verity/core/c/C.lean", "module-doc"): 1,
                             ("Security/Gone.lean", "long-file"): 1})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        held(scan(tmp_path), loose)
    (w,) = caught
    assert w.category is Stale and str(w.message).splitlines()[1:] == [
        "  Security/Gone.lean [long-file]: 1 allowed, 0 found: delete the entry",
        f"  {a} [heartbeats-uncommented]: 3 allowed, 1 found: lower its count to 1",
        "  verity/core/c/C.lean [module-doc]: 1 allowed, 0 found: delete the entry"], str(w.message)
    assert f"test_{NAME}.py --prune" in str(w.message)
    assert loose & found(tmp_path) == exact, "--prune leaves what the tree holds"

    # --record lists what a merge brought and lowers what it fixed, but a `maxHeartbeats 0` the list doesn't have still fails
    assert recorded(loose, found(tmp_path)) == exact and recorded(exact - Counter({(b, "long-file"): 1}), exact) == exact
    unlimited = Counter({(a, HARD): 1})
    assert recorded(Counter(), found(tmp_path)) == exact - unlimited == recorded(exact - unlimited, found(tmp_path))
    assert recorded(exact, exact + unlimited) == exact and recorded(exact + unlimited, exact) == exact
    assert f"  {a} [{HARD}]: 1 found, 0 allowed" in fails(recorded(Counter(), found(tmp_path)))


def test_the_list_is_well_formed_and_an_entry_moves_with_its_file() -> None:
    assert parse({"entries": [{"file": "A.lean", "rule": "long-file", "count": 1}]}) == Counter({("A.lean", "long-file"): 1})
    for bad in ({"file": "A.lean", "rule": "long-file"}, {"file": "A.lean", "rule": "long-file", "count": 0},
                {"file": "A.lean", "rule": "long-file", "count": True}, {"file": "A.lean", "rule": "naming", "count": 1},
                {"file": "A.lean", "count": 1}):
        try:
            parse({"entries": [bad]})
        except AssertionError as e:
            assert str(e).startswith(f"{LIST.name}: "), e
        else:
            raise AssertionError(f"{bad} parsed")
    try:
        parse({"entries": [{"file": "A.lean", "rule": "long-file", "count": 1}] * 2})
    except AssertionError as e:
        assert "duplicate entry" in str(e), e
    else:
        raise AssertionError("a duplicate parsed")
    have = Counter({("a/X.lean", "module-doc"): 1, ("a/Y.lean", "heartbeats-uncommented"): 2, ("B.lean", "module-doc"): 1})
    assert moved(have, "a", "c") == Counter({("c/X.lean", "module-doc"): 1, ("c/Y.lean", "heartbeats-uncommented"): 2,
                                             ("B.lean", "module-doc"): 1})
    assert moved(have, "a/X.lean", "B.lean") == Counter({("B.lean", "module-doc"): 2, ("a/Y.lean", "heartbeats-uncommented"): 2})


def recorded(have: Counter, got: Counter) -> Counter:
    """The list `--record` writes over ``have`` for a tree that finds ``got``: ``got``, but of ``HARD`` only what ``have``
    already allows."""
    return Counter({k: n for k, n in got.items() if k[1] != HARD}) + Counter({k: min(n, have[k]) for k, n in got.items()
                                                                              if k[1] == HARD})


def moved(entries: Counter, old: str, new: str) -> Counter:
    """``entries`` with the file ``old``'s, or the directory ``old``'s files', under ``new``: a pair already there adds."""
    old, new = old.rstrip("/"), new.rstrip("/")
    out: Counter = Counter()
    for (f, rule), n in entries.items():
        out[(new + f[len(old):] if f == old or f.startswith(f"{old}/") else f, rule)] += n
    return out


def dump(entries: Counter) -> None:
    lines = [f"    {json.dumps(dict(zip(FIELDS, (*k, n))), ensure_ascii=False)}" for k, n in sorted(entries.items()) if n > 0]
    LIST.write_text(f'{{\n  "rule": "the Lean bar (tests/lint/test_{NAME}.py)",\n  "entries": [\n' + ",\n".join(lines)
                    + "\n  ]\n}\n", encoding="utf-8")


def counts(root: Path = ROOT) -> str:
    """Each rule's findings in ``root``, in all and by Lake package (a file's is the nearest directory above it that holds
    a lakefile)."""
    packages = sorted((d for d, names in walk(root) if {"lakefile.toml", "lakefile.lean"} & set(names)), key=len, reverse=True)
    by: dict[str, Counter] = {}
    for f, rule, *_ in scan(root):
        package = next((p for p in packages if f.startswith(f"{p}/")), "(no package)")
        by.setdefault(package, Counter())[rule] += 1
    total = sum(by.values(), Counter())
    rows = [("package", *RULES), ("all", *(str(total[r]) for r in RULES)),
            *((p, *(str(c[r]) for r in RULES)) for p, c in sorted(by.items()))]
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(cell.ljust(w) if i == 0 else cell.rjust(w) for i, (cell, w) in enumerate(zip(r, widths)))
                     for r in rows)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--move"] and len(sys.argv) == 4:
        have = allowed()
        dump(after := moved(have, *sys.argv[2:]))
        print(f"{sum(1 for k in have if k not in after)} entries moved from {sys.argv[2]} to {sys.argv[3]}")
    elif sys.argv[1:] == ["--prune"]:
        dump(allowed() & found())
    elif sys.argv[1:] == ["--record"]:
        dump(after := recorded(allowed(), got := found()))
        if left := sorted(k[0] for k in got - after):
            sys.exit("--record lists no new `maxHeartbeats 0`; these files hold more than the list allows, so give the "
                     "new ones a bound:\n" + "\n".join(
                f"  {f}:{line} {what}" for f, rule, line, what in scan() if rule == HARD and f in left))
    elif sys.argv[1:] == ["--counts"]:
        print(counts())
    else:
        sys.exit(__doc__.rsplit("\n\n", 1)[-1])
