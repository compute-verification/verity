"""The two-stage audit's environment variables, each declared once (`verity_config.settings`): the command that runs the
guard's Lean model (`tests/TwoGuard.lean`), which `tests/guard_node.py --pytest` makes and `tests/test_guard.py` reads
(``of(__file__)``). `guard_node.py` reads the run's output directory from the shared ones.
"""
from __future__ import annotations

from verity_config.settings import WHERE, Registry, Setting as S

REGISTRY = Registry(
    __file__, ("VERITY_TWO_STAGE_",),
    S("VERITY_TWO_STAGE_GUARD", str, None, WHERE,
      "the command that runs TwoGuard.lean on a run's guard export (guard_node.py --pytest makes it and sets it for "
      "test_guard.py, which skips while it is unset)"),
)
