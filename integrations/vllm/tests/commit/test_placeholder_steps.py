"""A run with a placeholder step (root b"") finalizes without a run root instead of crashing in the run-root fold.

Bounded staging's warm-up is a learn-only pass: a learning step over the transient budget records its plan, discards its windows and
commits a placeholder root b"" (`native_collect._commit_step_windowed`).  The run-root fold takes full-length step roots only, so folding
those roots raised at the warm-up's finalize (#74, --bounded-staging --retain-exclude fa3_hidden_m1) before any record run.  A run whose
every step has a root folds exactly as before.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random

import pytest

pytest.importorskip("torch")

from verity_vllm.acquire.committer_api import StepCommitment
from verity_vllm.commit import step_rows
from verity_vllm.commit.committer.native_host import NativeHostCommitter

PROGRAM = hashlib.sha256(b"program").digest()


def stream(steps: int = 3) -> list[list[tuple[str, bytes]]]:
    rng = random.Random(5)
    return [[(f"layer{k}.op", rng.randbytes(30000 + 7 * k)) for k in range(3)] for _ in range(steps)]


class _LearnOnly(NativeHostCommitter):
    """Step 1 is discarded the way a learn-only pass discards a learning step over budget."""

    def _commit_step(self, job) -> None:
        if job.step != 1:
            return super()._commit_step(job)
        self._layouts[job.step] = job.metas
        self._done[job.step] = StepCommitment(job.step, b"", b"", 0, sum(m.nbytes for m in job.metas), len(job.metas), b"", job.t_step_end,
                                              job.t_step_end, 0)
        return None


def test_a_learn_only_pass_finalizes_without_a_run_root():
    whole = NativeHostCommitter(None, run_id="r", hash_threads=1, program_digest=PROGRAM)
    full = whole.commit_offline(stream())
    learner = _LearnOnly(None, run_id="r", hash_threads=1, program_digest=PROGRAM)
    learner.replay_row_keys = dict(whole.row_keys)
    part = learner.commit_offline(stream())
    assert part.run_root == b""
    assert part.steps[1].root == b"" and [s.root for s in part.steps[::2]] == [s.root for s in full.steps[::2]]


def test_a_run_whose_steps_all_have_roots_folds_as_before():
    c = NativeHostCommitter(None, run_id="r", hash_threads=1, program_digest=PROGRAM)
    run = c.commit_offline(stream())
    assert len(run.run_root) == step_rows.ROOT_BYTES
    assert run.run_root == step_rows.run_root(PROGRAM, c._geo_digest, [s.root for s in run.steps])


def test_a_run_without_a_run_root_verifies_no_opening():
    """The placeholder path can never pass: every opening is verified up to the run root, and an empty run root matches nothing -- not
    even the empty root recomputed from the placeholder step, nor a forged full-length run root over it."""
    c = _LearnOnly(None, run_id="r", hash_threads=1, program_digest=PROGRAM)
    run = c.commit_offline(stream())
    ops = c.open([(s, i) for s in (0, 2) for i in range(run.steps[s].n_leaves)])
    assert ops and not any(c.verify(run, o) for o in ops)
    forged = dataclasses.replace(run, run_root=step_rows.run_root(PROGRAM, c._geo_digest, [s.root or bytes(step_rows.ROOT_BYTES) for s in run.steps]))
    assert not any(c.verify(forged, o) for o in ops)
    whole = NativeHostCommitter(None, run_id="r", hash_threads=1, program_digest=PROGRAM)
    full = whole.commit_offline(stream())
    assert all(whole.verify(full, o) for o in whole.open([(0, 0), (2, 1)]))


def test_a_run_without_a_run_root_verifies_no_range_opening():
    """The range verifier (`verify_range`) is held to the same run root: a placeholder run's recomputed run root is None, never the empty
    root, so no range of a real step passes -- with or without a forged run root, and not through the per-step roots cache either."""
    c = _LearnOnly(None, run_id="r", hash_threads=1, program_digest=PROGRAM)
    run = c.commit_offline(stream())
    ros = [c.open_range(0, 0, 3), c.open_range(2, 16, 20)]
    assert all(ro is not None for ro in ros)
    forged = dataclasses.replace(run, run_root=step_rows.run_root(PROGRAM, c._geo_digest, [s.root or bytes(step_rows.ROOT_BYTES) for s in run.steps]))
    for r in (run, forged):
        cache: dict[int, bytes] = {}
        assert not any(c.verify_range(r, ro) for ro in ros) and not any(c.verify_range(r, ro, cache) for ro in ros) and cache == {}
    whole = NativeHostCommitter(None, run_id="r", hash_threads=1, program_digest=PROGRAM)
    full = whole.commit_offline(stream())
    assert all(whole.verify_range(full, whole.open_range(s, lo, hi)) for s, lo, hi in ((0, 0, 3), (2, 16, 20)))
