"""`c2_replay.rss_kb`: this process's resident set and its high-water mark in kB, read from ``/proc/self/status``, and
``(None, None)`` where that file can't be read or a line of it can't be parsed."""
from __future__ import annotations

from pathlib import Path

import pytest

from verity_vllm.pipeline.c2_replay import rss_kb


def status(monkeypatch: pytest.MonkeyPatch, text: str | OSError) -> None:
    def read_text(self: Path, *a: object, **k: object) -> str:
        assert str(self) == "/proc/self/status"
        if isinstance(text, OSError):
            raise text
        return text
    monkeypatch.setattr(Path, "read_text", read_text)


def test_it_reads_the_resident_set_and_its_high_water_mark(monkeypatch: pytest.MonkeyPatch) -> None:
    status(monkeypatch, "Name:\tpython\nVmHWM:\t    2048 kB\nVmRSS:\t    1024 kB\nThreads:\t1\n")
    assert rss_kb() == (1024, 2048)


def test_a_field_the_file_does_not_hold_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    status(monkeypatch, "Name:\tpython\nVmRSS:\t    1024 kB\n")
    assert rss_kb() == (1024, None)


@pytest.mark.parametrize("fault", [FileNotFoundError("no /proc here"), PermissionError("not this process's to read")])
def test_both_are_none_where_the_file_cannot_be_read(monkeypatch: pytest.MonkeyPatch, fault: OSError) -> None:
    status(monkeypatch, fault)
    assert rss_kb() == (None, None)


@pytest.mark.parametrize("line", ["VmRSS:", "VmHWM:\tmany kB"])
def test_both_are_none_on_a_line_that_does_not_parse(monkeypatch: pytest.MonkeyPatch, line: str) -> None:
    status(monkeypatch, f"VmRSS:\t    1024 kB\n{line}\n")
    assert rss_kb() == (None, None)
