"""CPU test for ``tp-commit --replay-ranks-at-once K`` (`pipeline.tp.rank_replay.host_slot`): rank r replays in slot r % K, so ranks of
one slot never hold their Programs at once while ranks of different slots do, and without K no rank waits."""
from __future__ import annotations

import fcntl

from verity_vllm.pipeline.tp import rank_replay as RR


def _held(path) -> bool:
    with open(path, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
        return False


def test_a_rank_replays_in_slot_rank_mod_k_beside_the_ranks_of_other_slots(tmp_path):
    cfg = {"ranks_at_once": 2, "slot_dir": str(tmp_path)}
    with RR.host_slot(cfg, 4) as s4:
        assert s4["slot"] == 0 and s4["ranks_at_once"] == 2
        assert _held(tmp_path / "replay_slot0.lock")          # ranks 0, 2 and 6 wait for rank 4
        with RR.host_slot(cfg, 7) as s7:                       # slot 1 is free: rank 7 replays beside rank 4
            assert s7["slot"] == 1 and _held(tmp_path / "replay_slot1.lock")
        assert not _held(tmp_path / "replay_slot1.lock")
    assert not _held(tmp_path / "replay_slot0.lock")
    with RR.host_slot({}, 3) as none:
        assert none is None
