"""`verity_vllm.acquire.h1_commit.RankCommit.h1rank_commit` on a rank that holds no H1 window: the Commit is refused before
the rank's model is read."""
from __future__ import annotations

import pytest

from verity_vllm.acquire import h1_commit as HC


class Rank(HC.RankCommit):
    """A worker whose model must not be read."""

    def _h1rank_layers(self):
        raise AssertionError("the Commit read the model of a rank with no H1 window")


def test_a_rank_whose_window_was_released_refuses_its_commit():
    """`RankExtension.h1rank_release` leaves the rank's state None."""
    rank = Rank()
    rank._h1rank = None
    with pytest.raises(RuntimeError, match="holds no H1 window"):
        rank.h1rank_commit()
