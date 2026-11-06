"""drive.toml parses strictly with this tree's research: a key it doesn't read (a typo, or a key landed before the code that
reads it) fails here, since clients ignore such keys and only drive itself would refuse them."""
from __future__ import annotations

from pathlib import Path

from research.queue import DRIVE_FLAGS, DriveFlags

ROOT = Path(__file__).resolve().parents[1]


def test_drive_toml_parses_strictly():
    DriveFlags.parse((ROOT / DRIVE_FLAGS).read_text(), DRIVE_FLAGS, strict=True)
