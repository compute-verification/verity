"""`tp-commit`'s schedule of measured runs (`pipeline.tp.commit.schedule`), from which its fail-fast names the runs it did not run:
each pair's two arms, their order reversed on odd pairs; ``--only-arm`` keeps that arm of each pair."""
from __future__ import annotations

from verity_vllm.pipeline.tp import commit as TPCM


def test_the_schedule_alternates_the_arms_and_keeps_only_arm():
    assert TPCM.schedule(3, None) == [(0, "control"), (0, "instrumented"), (1, "instrumented"), (1, "control"), (2, "control"), (2, "instrumented")]
    assert TPCM.schedule(2, "instrumented") == [(0, "instrumented"), (1, "instrumented")]
