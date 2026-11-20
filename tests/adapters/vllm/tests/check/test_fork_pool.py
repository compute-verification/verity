"""A forked child dies with its parent (`die_with_parent`)."""
from __future__ import annotations

import os
import sys

import pytest

from verity_vllm.check.replay.fork import die_with_parent

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")


def test_a_child_forked_after_its_parent_died_exits_at_once():
    pid = os.fork()
    if pid == 0:
        die_with_parent(os.getppid() + 1)   # not this child's parent: as if the parent died before the child ran
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 1
