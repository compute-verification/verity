"""`research move`: a pool run still waiting in its line is resent with what it was sent with, and a started one is refused."""
from datetime import UTC, datetime
from pathlib import Path

import pytest

from research import move

POOL_CMD = ["env", "CHECK_SLOTS=/workspace/research/locks/research/slots", "python3", "tools/verity/check/slot.py", "--by", "proofs",
            "--wait-max", "14400", "--run-max", "3600", "--", "nice", "-n", "19", "bash", "-c", "uv run lake build"]
HARNESS = ["--stage", "run", "--project", "verity", "--timeout", "3600", "--timeout", "18000", "--cwd", "clone", "--tree-sha", "abc",
           "--source-identity", '{"commit": "abc"}', "--declared-output", "out/*.json", "--no-sampler"]


def test_the_resend_keeps_the_runs_lane_timeout_memory_environment_and_harness_options():
    """Lanes cancelled and resent runs stuck behind full lines by hand, a dozen an hour, some with the wrong options."""
    req = {"command": POOL_CMD, "harness_args": HARNESS, "env": {"CHECK_RUN_MEM_GB": "48", "LAKE_JOBS": "8"}, "mem_gb": None}
    argv = move.argv_for(req, source=Path("/tmp/tree"), to="auto")
    assert argv == ["research", "run", "--on", "auto", "--project", "verity", "--source", "/tmp/tree", "--as", "proofs",
                    "--mem-gb", "48", "--env", "LAKE_JOBS=8", "--timeout", "3600", "--cwd", "clone",
                    "--declared-output", "out/*.json", "--no-sampler", "--", "bash", "-c", "uv run lake build"]
    assert move.argv_for(req, source=Path("/t"), to="vy-nebius-cpu-12", cpus=8)[2:4] == ["--on", "vy-nebius-cpu-12"]
    assert "--cpus" in move.argv_for(req, source=Path("/t"), cpus=8)


def test_a_run_no_pool_line_wraps_is_refused():
    with pytest.raises(move.Unmovable, match="isn't a pool run"):
        move.argv_for({"command": ["bash", "-c", "make"], "harness_args": []}, source=Path("/t"))
    with pytest.raises(move.Unmovable, match="wraps no command"):
        move.unwrap(POOL_CMD[:11] + ["nice", "-n", "19"])


def test_only_a_run_still_waiting_in_its_line_counts_as_movable():
    now = datetime(2026, 10, 29, 15, 0, tzinfo=UTC)
    assert move.waited_s({"state": "waiting", "since": "2026-10-29T14:30:00Z"}, now) == 1800
    assert move.waited_s({"state": "held", "since": "2026-10-29T14:30:00Z", "took": "2026-10-29T14:31:00Z"}, now) is None
    assert move.waited_s({}, now) is None
