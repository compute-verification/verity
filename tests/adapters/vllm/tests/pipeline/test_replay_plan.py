"""The slim bundle's planner process (`pipeline.replay_plan`): a failure in it reaches the Commit by name (which then replays in process)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from verity_vllm.pipeline import replay_plan as RP


def test_a_planner_that_cannot_prepare_names_why():
    p = RP.Planner(SimpleNamespace(**({k: None for k in RP.ARGS} | {"workload": "/nonexistent/workload.json"})))
    with pytest.raises(RuntimeError, match="replay planner: FileNotFoundError"):
        p.ready()
    p.close()
