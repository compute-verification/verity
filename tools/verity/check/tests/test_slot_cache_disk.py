"""The check cache's disk blocks a slot as /workspace does: a cache left on the root disk filled it mid-build."""
from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("repo_slot_cache_disk", ROOT / "tools" / "verity" / "check" / "slot.py")
SLOT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SLOT)
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_the_cache_disk_is_its_nearest_existing_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path / "not-yet" / "verity-check"))
    assert SLOT.cache_disk() == tmp_path
    (tmp_path / "not-yet" / "verity-check").mkdir(parents=True)
    assert SLOT.cache_disk() == tmp_path / "not-yet" / "verity-check"


def test_a_full_cache_disk_blocks_every_slot(monkeypatch, tmp_path):
    slot = SLOT.parse("a 0-1")[0]
    monkeypatch.setattr(SLOT, "DISK", tmp_path / "no-disk")
    monkeypatch.setattr(SLOT, "memory_gb", lambda root=None: 10**6)
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path / "verity-check"))
    assert SLOT.blocked(slot, NOW) is None
    monkeypatch.setattr(SLOT, "DISK_MIN_GB", 1e18)
    assert SLOT.blocked(slot, NOW).startswith(f"{tmp_path} has ") and "GB free, under" in SLOT.blocked(slot, NOW)
