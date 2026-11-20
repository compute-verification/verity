"""No pin is stale, every file that records a Lean source's digest is a pin with its command, and every test module that rewrites
its fixture with ``--write`` is a writer whose failure names its command (`tools/check/pins.py`)."""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("repo_pins_under_test", ROOT / "tools/check/pins.py")
PINS = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = PINS
spec.loader.exec_module(PINS)
WRITES = re.compile(r"""__name__ == ["']__main__["'][\s\S]*["']--write["']""")


def test_no_pin_is_stale():
    rows = PINS.stale(ROOT)
    assert not rows, "regenerate each stale pin with its command, never by hand:\n" + PINS.report(rows, ROOT)


def test_every_file_that_records_a_lean_sources_digest_is_a_pin():
    found = PINS.unlisted(ROOT)
    assert not found, ("these record Lean sources' digests: list each in tools/check/pins.py `PINS` with the command "
                       f"that regenerates it: {found}")


def test_every_test_that_rewrites_its_fixture_is_a_writer():
    listed = {w.module for w in PINS.WRITERS}
    found = sorted(str(p.relative_to(ROOT)) for d in PINS.WRITER_DIRS for p in (ROOT / d).rglob("test_*.py")
                   if WRITES.search(p.read_text()))
    unlisted = [m for m in found if m not in listed]
    assert not unlisted, ("these rewrite a fixture with --write: list each in tools/check/pins.py `WRITERS` with its "
                          f"fixture, and name the `Writer`'s command in the test's failure: {unlisted}")


def test_every_writer_s_test_names_its_command():
    for w in PINS.WRITERS:
        assert (ROOT / w.path).is_file() and (ROOT / w.module).is_file(), f"{w}: no such fixture or test module"
        text = (ROOT / w.module).read_text()
        assert WRITES.search(text), f"{w.module}: no `--write` under `__main__`: drop its `Writer`"
        assert w.command in text, f"{w.module}: its staleness failure names `{w.command}`, the one command that rewrites {w.path}"


def test_every_inline_pin_is_stated_where_its_writer_reads_it():
    for e in PINS.INLINE:
        text = (ROOT / e.module).read_text()
        assert e.hook in PINS.HOOKS and re.search(rf"\b{e.hook}\(", text), f"{e}: its module calls no `{e.hook}`"
        if e.name is None:
            assert re.search(rf"parametrize\([^)]*\b{e.by}\b", text), f"{e}: no test takes a `{e.by}` argument"
        else:
            assert re.search(rf"^{e.name}\b[^=\n]*= \{{", text, re.M), f"{e}: no module-level dict `{e.name}`"
    for r in PINS.REPINS:
        assert (ROOT / r.path).is_file() and "VERITY_REPIN" in (ROOT / r.module).read_text(), r


def _entry():
    return PINS.Inline("m.py", "standalone_call", "target", "PINS")


def _pin(old, new, pinned=None, counts=()):
    return {"old": old, "pinned": pinned, "new": list(new), "counts": [list(c) for c in counts], "facts": []}


def test_settle_rewrites_only_a_pin_its_test_recomputed_with_the_pinned_counts():
    a, b, c, d, e = (ch * 128 for ch in "abcde")
    text = f'PINS = {{"x": (1, 2, 3, "{a}"), "y": (4, 5, 6, "{b}"), "z": (7, 8, 9, "{c}"), "w": (1, 1, 1, "{d}"), "v": "{e}"}}\n'
    got = {"x": _pin(a, [e[:-1] + "f"], [1, 2, 3], [(1, 2, 3)]), "y": _pin(b, ["0" * 128], [4, 5, 6], [(4, 5, 7)]),
           "z": _pin(c, []), "w": _pin(d, ["1" * 128, "2" * 128]), "v": _pin(e, [e])}
    out, rows = PINS.settle(text, [_entry()], [got])
    by = {r["key"]: r for r in rows}
    assert by["x"].get("rewritten") and e[:-1] + "f" in out and a not in out
    assert by["y"]["refused"].startswith("counts moved") and b in out
    assert by["z"]["refused"] == "not reached" and by["w"]["refused"] == "two digests" and by["v"].get("kept")


def test_settle_refuses_a_digest_that_is_not_one_literal():
    a = "a" * 128
    out, rows = PINS.settle(f'X = "{a}"\nY = "{a}"\n', [_entry()], [{"k": _pin(a, ["b" * 128])}])
    assert rows[0]["refused"] == "not one literal" and out.count(a) == 2
