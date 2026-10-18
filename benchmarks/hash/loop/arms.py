"""The hash loop's sibling arm: circuits' rows arm (``served_commit.prof_arm.RowsArm``) committing under ``rows-blake3``
(``verity_vllm.commit.step_rows_b3``: the same step with keyed BLAKE3 in every hash position).  Research only: BLAKE3 never serves.
``job.sh`` names it to prof.sh as ``served-commit-rows-blake3``."""
from __future__ import annotations

from served_commit.prof_arm import RowsArm


class RowsBlake3Arm(RowsArm):
    commit_scheme = "rows-blake3"


SERVED_COMMIT_ROWS_BLAKE3 = RowsBlake3Arm("served-commit-rows-blake3", retain="none")
