"""`checked`, the frozen landing branch, is guarded as `main` and `next` are: only the verified queue's loop on the control pod
pushes it, and no sweep deletes it."""
from __future__ import annotations

import pytest

from research.merge import CHECKED, KEEP_BRANCHES
from research.queue import QueueError
from test_queue import World, world  # noqa: F401


def test_only_the_control_pod_pushes_checked_and_the_sweep_keeps_it(world: World, monkeypatch):
    assert CHECKED in KEEP_BRANCHES
    q = world.queue()
    q._drive_only(CHECKED, "once", False)
    monkeypatch.delenv("RESEARCH_DRIVE", raising=False)
    with pytest.raises(QueueError, match="`checked` is moved only by the verified queue's loop on the control pod"):
        q._drive_only(CHECKED, "once", False)
