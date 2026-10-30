"""A recorded check refuses before it starts when its cache's disk is nearly full, rather than failing a build hours in."""
from __future__ import annotations

import importlib.util
import shutil
from collections import namedtuple
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("repo_check_cache_room", ROOT / "tools" / "verity" / "check" / "check.py")
CHECK = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CHECK)

Usage = namedtuple("Usage", "total used free")


def _disk(monkeypatch, tmp_path, free_gb):
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path / "not-yet" / "verity-check"))
    seen = []

    def usage(path):
        seen.append(Path(path))
        return Usage(0, 0, int(free_gb * 2**30))

    monkeypatch.setattr(shutil, "disk_usage", usage)
    return seen


def test_room_is_read_on_the_nearest_existing_ancestor(monkeypatch, tmp_path):
    seen = _disk(monkeypatch, tmp_path, 500)
    assert CHECK.cache_room() is None
    assert seen == [tmp_path]


def test_a_full_disk_names_the_cache_and_the_fix(monkeypatch, tmp_path):
    _disk(monkeypatch, tmp_path, 26)
    why = CHECK.cache_room()
    assert "26 GB free, under 60 GB" in why and "verity-check" in why and "CHECK_CACHE" in why


def test_a_recorded_check_refuses_on_a_full_disk(monkeypatch, tmp_path, capsys):
    _disk(monkeypatch, tmp_path, 10)
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20990101-000000-0000")
    assert CHECK.main(["--keep-going"]) == 2
    assert "under 60 GB" in capsys.readouterr().err
