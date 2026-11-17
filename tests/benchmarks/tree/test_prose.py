"""The prose numbers (`prose.py`) on made-up Python and Lean sources."""
from __future__ import annotations

import json
from pathlib import Path

import prose as P

RUN = "r20000101-000000-aaaa"
STAMP = {"card": "none", "gpus": [], "driver": None, "cuda": None, "node": "test", "commit": "d" * 40}
HANDLES = [{"handle": "top", "owns": ["tests/"]}, {"handle": "lean", "owns": ["security/"]}, {"handle": "comms"}]
NONE = (set(), set())

PY = '''#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One sentence. And a second."""
import os  # noqa: F401
import sys  # why sys


def short():
    """One.

    Two.
    """
    # a
    # b
    return 1


def long():
    """One.
    Two.
    Three.
    Four.
    """
    # a
    # b
    # c
    x = 1  # type: ignore
    return x
'''

LEAN = '''/-! The first. The second. -/
namespace Foo

/-! A later one.
Two.
Three. -/

/-- One.
Two.
Three.
Four. -/
theorem listed : True := trivial

/-- One.
Two.
Three.
Four. -/
def read : Nat := 1

/-- Short. -/
def other : String := "-- not a comment /- nor this"

-- a
-- b
-- c
def x := s!"{"--"}" ++ «a--b».c ++ '"'.toString -- trailing
/- outer /- nested -/ still -/
end Foo
'''


def kinds(items: list[P.Item]) -> list[tuple[str, int, bool]]:
    return [(it.kind, it.line, P.over_length(it)) for it in items]


def test_python_items_and_their_lengths() -> None:
    n, prose, items = P.python_prose(PY)
    assert n == 28
    assert prose == {3, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26}
    assert kinds(items) == [("file", 3, True), ("declaration", 9, False), ("declaration", 19, True),
                            ("trailing", 5, False), ("comment", 13, False), ("comment", 24, True)]


def test_a_module_docstring_that_is_help_text_counts_only_past_one_paragraph() -> None:
    one = '"""First. Second."""\nprint(__doc__)\n'
    two = '"""First.\n\nSecond."""\nprint(__doc__)\n'
    assert [P.over_length(it) for it in P.python_prose(one)[2]] == [False]
    assert [P.over_length(it) for it in P.python_prose(two)[2]] == [True]
    usage = '"""What it does\n\n    prog ARGS\n"""\nprint(__doc__)\n'
    assert [P.over_length(it) for it in P.python_prose(usage)[2]] == [True]


def test_a_run_of_comments_ends_where_its_column_changes() -> None:
    src = "if x:\n    # a\n    # b\n# c\n# d\n    pass\n"
    assert [(it.line, it.body) for it in P.python_prose(src)[2]] == [(2, ["a", "b"]), (4, ["c", "d"])]
    lean = "-- a\n-- b\n  -- c\n"
    assert [(it.line, it.body) for it in P.lean_prose(lean, NONE, NONE)[2]] == [(1, ["a", "b"]), (3, ["c"])]


def test_a_lean_blocks_blank_lines_are_its_lines() -> None:
    assert P.lean_prose("/-! One.\n\nTwo. -/\ndef x := 1\n", NONE, NONE)[1] == {1, 2, 3}


def test_lean_spans_know_nesting_strings_chars_and_names() -> None:
    spans = P.lean_spans(LEAN)
    comments = [LEAN[a:b] for k, a, b in spans if k != "string"]
    assert "/- outer /- nested -/ still -/" in comments
    assert "-- trailing" in comments
    assert not any("not a comment" in c or "a--b" in c for c in comments)
    assert sum(k == "string" for k, _, _ in spans) == 3


def test_lean_items_their_lengths_and_the_locks() -> None:
    audit, reads = P.tails({"Foo.listed"}), P.tails({"Foo.read"})
    n, prose, items = P.lean_prose(LEAN, audit, reads)
    assert n == 28
    assert prose == {1, 4, 5, 6, 8, 9, 10, 11, 14, 15, 16, 17, 20, 23, 24, 25, 27}
    assert kinds(items) == [("file", 1, True), ("comment", 4, True), ("declaration", 8, False),
                            ("declaration", 14, True), ("declaration", 20, False), ("comment", 23, True),
                            ("trailing", 26, False), ("comment", 27, False)]
    assert [it.reads for it in items if it.kind == "declaration"] == [False, True, False]


def test_an_unclosed_lean_block_is_not_parsed() -> None:
    sums, broken = P.measure({"security/A.lean": b"/- open\ndef x := 1\n", "tests/a.py": b"x = (\n"}, HANDLES,
                             NONE, NONE)
    assert broken == ["security/A.lean (ValueError)", "tests/a.py (SyntaxError)"]
    assert not +sums[None]


def test_memory_kinds() -> None:
    hit = {
        "date": "as of Mon 16 Nov, and 2026-11-16",
        "issue": "see TOP-433",
        "pull request": "landed in PR #10590",
        "run": "run r20261116-220000-abcd",
        "artifact or note": "art:cc59294a",
        "commit": "vLLM at d9105ea80",
        "machine": "on node 2, and vy-nebius-cpu-4",
        "person": "as Daniel asked",
    }
    for kind, text in hit.items():
        assert {k for k, rx in P.MEMORY_KINDS.items() if rx.search(text)} == {kind}, text
    for text in ("SHA-512, FP-8 and ISO-8601", "root node 1 reads node 2", "the 16 bits", "vy-admin and vy-sudo",
                 "CHAIN-12"):
        assert not any(rx.search(text) for rx in P.MEMORY_KINDS.values()), text


def test_skipped_paths_and_generated_files() -> None:
    for path in ("archive/a.py", "x/.lake/b.lean", "tests/x/fixtures/c.py", "a/vendored/d.py", "b/upstream/e.lean"):
        assert P.SKIP.search(path), path
    assert not P.SKIP.search("tests/x/test_fixtures.py")
    lean = b"import A\nimport B\nimport C\nimport D\nimport E\n/-! Written by a script; do not edit. -/\ndef x := 1\n"
    sums, _ = P.measure({"tests/g.py": b"# @generated by a tool\nx = 1\n", "security/G.lean": lean}, HANDLES, NONE, NONE)
    assert not +sums[None]
    sums, _ = P.measure({"security/H.lean": b"import A\n/-! The vectors a script generates. -/\n"}, HANDLES, NONE, NONE)
    assert sums[None]["lines", "Lean"] == 2


FILES = {
    "tests/a.py": b'"""One."""\nx = 1\n# Daniel, 16 Nov\n# two\n# three\n',
    "security/A.lean": b"/-! One. -/\ndef a := 1\n",
    "elsewhere/b.py": b"x = 1\n",
}


def test_each_lead_gets_its_own_files_and_the_total_is_written_last(tmp_path: Path) -> None:
    sums, broken = P.measure(FILES, HANDLES, NONE, NONE)
    assert list(sums) == ["lean", "top", P.UNOWNED, None]
    assert {m: P.share(c) for m, c in sums.items()} == {"lean": 50.0, "top": 80.0, P.UNOWNED: 0.0, None: 62.5}
    P.write(P.kit.Writer(tmp_path, RUN, STAMP), sums, broken)
    doc = json.loads((tmp_path / P.kit.RESULT).read_text(encoding="utf-8"))
    got = {(m["name"], m.get("member")): (m["value"], m["target"], m["unit"], m["mark"]) for m in doc["measurements"]}
    assert got[(P.SHARE, None)] == (62.5, 10, "percent", "measured")
    assert got[(P.OVER, "top")] == (1, 0, "items", "measured") and got[(P.OVER, "lean")][0] == 0
    assert got[(P.MEMORY, None)] == (1, 0, "items", "measured") and got[(P.MEMORY, P.UNOWNED)][0] == 0
    for name in (P.SHARE, P.OVER, P.MEMORY):
        mine = [m for m in doc["measurements"] if m["name"] == name]
        assert [m.get("member") for m in mine] == ["lean", "top", P.UNOWNED, None]
    assert "Python 66.67% of 6 lines, Lean 50.0% of 2 lines" in doc["validation"]["detail"]


def test_a_file_not_parsed_makes_every_reading_partial(tmp_path: Path) -> None:
    sums, broken = P.measure({**FILES, "tests/bad.py": b"def (\n"}, HANDLES, NONE, NONE)
    P.write(P.kit.Writer(tmp_path, RUN, STAMP), sums, broken)
    doc = json.loads((tmp_path / P.kit.RESULT).read_text(encoding="utf-8"))
    assert {m["mark"] for m in doc["measurements"]} == {"partial"}
    assert {tuple(m["inputs"]) for m in doc["measurements"]} == {("partial", "not parsed: tests/bad.py (SyntaxError)")}
    assert "Not parsed, so not counted: tests/bad.py (SyntaxError)" in doc["validation"]["detail"]
