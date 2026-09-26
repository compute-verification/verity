"""A run with a placeholder step (root b"") finalizes without a run root instead of crashing in the run-root fold.

Bounded staging's warm-up is a learn-only pass: a learning step over the transient budget records its plan, discards its windows and
commits a placeholder root b"" (`native_collect._commit_step_windowed`).  Core's fold takes 32-byte digests only, so folding those roots
raised `InvalidArtifact: leaf must be a 32-byte digest` at the warm-up's finalize (#74, --bounded-staging --retain-exclude fa3_hidden_m1)
before any record run.  A run whose every step has a root folds exactly as before.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
import time

from verity_vllm.acquire.committer_api import StepCommitment
from verity_vllm.commit import scheme
from verity_vllm.commit.committer.native_host import NativeHostCommitter

CHUNK = 64
PROGRAM = hashlib.sha256(b"program").digest()


def stream(steps: int = 3) -> list[list[tuple[str, bytes]]]:
    rng = random.Random(5)
    return [[(f"layer{k}.op", rng.randbytes(900 + 7 * k)) for k in range(3)] for _ in range(steps)]


class _LearnOnly(NativeHostCommitter):
    """Step 1 is discarded the way a learn-only pass discards a learning step over budget."""

    def _commit_step(self, job) -> None:
        if job.step != 1:
            return super()._commit_step(job)
        self._layouts[job.step] = job.metas
        self._done[job.step] = StepCommitment(job.step, b"", b"", 0, sum(m.nbytes for m in job.metas), len(job.metas), b"", job.t_step_end,
                                              time.perf_counter(), 0)
        return None


def test_a_learn_only_pass_finalizes_without_a_run_root():
    full = NativeHostCommitter(None, run_id="r", chunk=CHUNK, hash_threads=1, program_digest=PROGRAM).commit_offline(stream())
    part = _LearnOnly(None, run_id="r", chunk=CHUNK, hash_threads=1, program_digest=PROGRAM).commit_offline(stream())
    assert part.run_root == b""
    assert part.steps[1].root == b"" and [s.root for s in part.steps[::2]] == [s.root for s in full.steps[::2]]


def test_a_run_whose_steps_all_have_roots_folds_as_before():
    c = NativeHostCommitter(None, run_id="r", chunk=CHUNK, hash_threads=1, program_digest=PROGRAM)
    run = c.commit_offline(stream())
    assert len(run.run_root) == 32
    assert run.run_root == scheme.run_root(PROGRAM, c._geo_digest, [s.root for s in run.steps])


def test_a_run_without_a_run_root_verifies_no_opening():
    """The placeholder path can never pass: every opening is verified up to the run root, and an empty run root matches nothing -- not
    even the empty root recomputed from the placeholder step, nor a forged 32-byte run root over it."""
    c = _LearnOnly(None, run_id="r", chunk=CHUNK, hash_threads=1, program_digest=PROGRAM)
    run = c.commit_offline(stream())
    ops = c.open([(s, i) for s in (0, 2) for i in range(run.steps[s].n_leaves)])
    assert ops and not any(c.verify(run, o) for o in ops)
    forged = dataclasses.replace(run, run_root=scheme.run_root(PROGRAM, c._geo_digest, [s.root or b"\0" * 32 for s in run.steps]))
    assert not any(c.verify(forged, o) for o in ops)
    whole = NativeHostCommitter(None, run_id="r", chunk=CHUNK, hash_threads=1, program_digest=PROGRAM)
    full = whole.commit_offline(stream())
    assert all(whole.verify(full, o) for o in whole.open([(0, 0), (2, 1)]))
