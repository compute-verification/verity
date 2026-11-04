"""The grader suite's environment variables, each declared once in a `verity_config.settings.Registry`."""
from __future__ import annotations

from verity_config.settings import WHERE, Registry, Setting

REGISTRY = Registry(__file__, ("VERITY_GRADER_",),
    Setting("VERITY_GRADER_SLOWDOWN_CODE", str, None, WHERE,
            "DIR@COMMIT, the directory training's slowdown code's package sits in and the commit it is at: the suite reads "
            "the tiny steps beside it (tests/test_slowdown_code.py, about a minute); unset, that test skips"))
