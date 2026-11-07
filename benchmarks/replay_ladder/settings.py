"""The replay ladder's environment variables, each declared once (`verity_config.settings`): the arm run.sh serves and
replays a model in, and where a replay's processes log Inductor's run-time choices. ladder.py reads them (``of(__file__)``)
and hands them to inductor_pin, which reads none itself: serving loads it from a site directory outside the tree.
"""
from __future__ import annotations

from verity_config.settings import VALUE, WHERE, Registry, Setting as S, choice

REGISTRY = Registry(
    __file__, ("LADDER_",),
    S("LADDER_ARM", choice("pinned", "rblock", "unpinned"), "unpinned", VALUE,
      "the arm run.sh serves and replays a model in: pinned keeps Inductor's first candidate at each on-device choice "
      "(inductor_pin), rblock runs with TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0, unpinned with neither"),
    S("LADDER_TILINGS", str, None, WHERE, "the directory a replay's engine processes log Inductor's run-time choices to "
                                          "(ladder.py sets it for the processes a replay starts)"),
)
