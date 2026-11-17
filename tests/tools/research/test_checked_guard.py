"""`checked`, the old path's frozen landing branch, which the verified queue reads as landed, is guarded as `main` is: only the
live queue (the verity-drive App) pushes it, and no sweep deletes it."""
from __future__ import annotations

import pytest

from research.merge import CHECKED, KEEP_BRANCHES
from research.queue import QueueError
from test_queue import World, world  # noqa: F401


def test_only_the_live_queue_pushes_checked_and_the_sweep_keeps_it(world: World, monkeypatch):
    assert CHECKED in KEEP_BRANCHES
    q = world.queue()
    q._drive_only(CHECKED, "once", False)
    monkeypatch.delenv("RESEARCH_DRIVE", raising=False)
    with pytest.raises(QueueError, match="`checked` is moved only by the verity-drive App"):
        q._drive_only(CHECKED, "once", False)
