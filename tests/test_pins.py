"""No pin is stale, every file that records a Lean source's digest is a pin with its command, and every test module that rewrites
its fixture with ``--write`` is a writer whose failure names its command (`tools/verity/check/pins.py`)."""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("repo_pins_under_test", ROOT / "tools/verity/check/pins.py")
PINS = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = PINS
spec.loader.exec_module(PINS)
WRITES = re.compile(r"""__name__ == ["']__main__["'][\s\S]*["']--write["']""")


def test_no_pin_is_stale():
    rows = PINS.stale(ROOT)
    assert not rows, "regenerate each stale pin with its command, never by hand:\n" + PINS.report(rows, ROOT)


def test_every_file_that_records_a_lean_sources_digest_is_a_pin():
    found = PINS.unlisted(ROOT)
    assert not found, ("these record Lean sources' digests: list each in tools/verity/check/pins.py `PINS` with the command "
                       f"that regenerates it: {found}")


def test_every_test_that_rewrites_its_fixture_is_a_writer():
    listed = {w.module for w in PINS.WRITERS}
    found = sorted(str(p.relative_to(ROOT)) for d in PINS.WRITER_DIRS for p in (ROOT / d).rglob("test_*.py")
                   if WRITES.search(p.read_text()))
    unlisted = [m for m in found if m not in listed]
    assert not unlisted, ("these rewrite a fixture with --write: list each in tools/verity/check/pins.py `WRITERS` with its "
                          f"fixture, and name the `Writer`'s command in the test's failure: {unlisted}")


def test_every_writer_s_test_names_its_command():
    for w in PINS.WRITERS:
        assert (ROOT / w.path).is_file() and (ROOT / w.module).is_file(), f"{w}: no such fixture or test module"
        text = (ROOT / w.module).read_text()
        assert WRITES.search(text), f"{w.module}: no `--write` under `__main__`: drop its `Writer`"
        assert w.command in text, f"{w.module}: its staleness failure names `{w.command}`, the one command that rewrites {w.path}"
