"""drive.toml parses strictly with this tree's research: a key it doesn't read (a typo, or a key landed before the code that
reads it) fails here, since its readers (the verified queue's loop and the clients) ignore such keys and nothing else would
refuse them.  A key whose code was deleted (`DELETED_KEYS`), which even the strict parse ignores so that code deployed ahead of
`main` runs on it, fails here too."""
from __future__ import annotations

import tomllib
from pathlib import Path

from research.queue import DELETED_KEYS, DRIVE_FLAGS, DriveFlags

ROOT = Path(__file__).resolve().parents[1]


def test_drive_toml_parses_strictly():
    text = (ROOT / DRIVE_FLAGS).read_text()
    DriveFlags.parse(text, DRIVE_FLAGS, strict=True)
    assert not (gone := sorted(set(tomllib.loads(text)) & set(DELETED_KEYS))), f"{DRIVE_FLAGS} sets {gone}, whose code was deleted"
