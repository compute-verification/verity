"""`check.py --record --on M --pool NAME`: a lane's own recorded check in a slot of M's NAME pool, `research run --pool`'s pools,
since the CPU nodes' merge-check slots are only=trains. And `--record` inside a `research run`, which can't launch one of its own, names the command that
records it instead."""
from __future__ import annotations

import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from research import auto
from research.merge import post_landing, unlike
from research.pods.nebius import slot_only

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("repo_check_record_pool", ROOT / "tools" / "verity" / "check" / "check.py")
CHECK = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CHECK)
TOOL = ["uv", "run", "--locked", "--all-packages", "--extra", "torch-cpu", "python", "tools/verity/check/check.py"]
CPU1 = "vy-nebius-cpu-1"


@pytest.fixture
def launched(monkeypatch):
    real, out = subprocess.run, []

    def run(cmd, *a, **k):
        if cmd[:4] == [sys.executable, "-m", "research", "run"]:
            out.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)
        return real(cmd, *a, **k)

    monkeypatch.setattr(CHECK.subprocess, "run", run)
    monkeypatch.setattr(CHECK, "landable", lambda base: True)
    monkeypatch.setattr(CHECK, "lean_deps_url_file", lambda d: None)
    monkeypatch.setattr(CHECK, "gather_verdicts", lambda d, given=None: None)
    monkeypatch.setenv("RESEARCH_LANE", "ci")
    monkeypatch.delenv("RESEARCH_RUN_ID", raising=False)
    return out


def _split(cmd: list[str]) -> tuple[list[str], list[str]]:
    i = cmd.index("--")
    return cmd[:i], cmd[i + 1:]


def test_a_pooled_record_waits_in_the_pools_slots_and_stays_a_check_the_gate_reads(launched):
    assert CHECK.record(["--base", "HEAD", "--on", CPU1, "--pool", "research", "--keep-going"]) == 0
    [cmd] = launched
    head, tail = _split(cmd)
    assert head[head.index("--env") + 1] == f"CHECK_SLOTS={slot_only.POOLS['research']}"
    assert head[head.index("--on") + 1] == CPU1 and head[head.index("--cwd") + 1] == "clone"
    assert head[head.index("--tool") + 1] == "check" and "--pool" not in head
    assert tail == ["python3", "tools/verity/check/slot.py", "--by", "ci", "--", *TOOL, "--keep-going"]
    assert slot_only.through_slot(tail) and unlike(tail, TOOL) is None


def test_a_pooled_deferred_record_is_a_post_landing_run_of_those_steps(launched):
    assert CHECK.record(["--base", "HEAD", "--on", CPU1, "--pool", "research", "--deferred", "lean-agreement"]) == 0
    [cmd] = launched
    head, tail = _split(cmd)
    assert "--tool" not in head and head[head.index("--env") + 1] == f"CHECK_SLOTS={slot_only.POOLS['research']}"
    assert tail[:5] == ["python3", "tools/verity/check/slot.py", "--by", "ci", "--"]
    assert tail[-2:] == ["--deferred", "lean-agreement"] and post_landing(tail, TOOL) == "lean-agreement"


def test_a_pool_a_node_keeps_of_its_own_is_taken_by_name(launched):
    """node 1's leanrec pool is no pool every node has, but a records run takes it by name, as slot.py's line there."""
    assert CHECK.record(["--base", "HEAD", "--on", CPU1, "--pool", "leanrec", "--deferred", "lean-agreement"]) == 0
    [cmd] = launched
    assert "CHECK_SLOTS=/workspace/research/locks/leanrec/slots" in cmd


def test_a_records_timeout_is_its_runs_not_the_checks(launched):
    assert CHECK.record(["--base", "HEAD", "--on", CPU1, "--pool", "research", "--timeout", "14400", "--deferred", "lean-agreement"]) == 0
    [cmd] = launched
    head, tail = _split(cmd)
    assert head[head.index("--timeout") + 1] == "14400" and "--timeout" not in tail


@pytest.mark.parametrize("argv, why", [
    (["--on", CPU1, "--pool", "../nope"], "../nope: no such pool; a pool is a name of letters, digits"),
    (["--pool", "research"], f"needs --on MACHINE, the one machine whose pool it waits in (`--on {CPU1} --pool research`)"),
    (["--on", "auto", "--pool", "research"], "needs --on MACHINE"),
    (["--on", f"{CPU1},vy-nebius-cpu-5", "--pool", "research"], "needs --on MACHINE"),
    (["--on", CPU1, "--pool", "research", "--priority"], "takes no --priority"),
])
def test_a_pool_that_cannot_be_taken_so_is_refused_before_anything_launches(launched, capsys, argv, why):
    assert CHECK.record(["--base", "HEAD", *argv]) == 2 and launched == []
    assert why in capsys.readouterr().err


def test_record_inside_a_research_run_is_refused_with_the_command_that_records_it(launched, monkeypatch, capsys):
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261027-020000-abcd")
    assert CHECK.main(["--record", "--deferred", "lean-agreement"]) == 2 and launched == []
    err = capsys.readouterr().err
    want = f"uv run python tools/verity/check/check.py --record --on {CPU1} --pool research --deferred lean-agreement"
    assert "check: --record inside a recorded run (r20261027-020000-abcd)" in err and f"`{want}`" in err
    assert CHECK.main(["--record", "--on", "vy-nebius-2", "--priority"]) == 2
    assert "`uv run python tools/verity/check/check.py --record --on vy-nebius-2 --priority`" in capsys.readouterr().err
    assert CHECK.main(["--pool", "proofs", "--record"]) == 2
    assert f"`uv run python tools/verity/check/check.py --record --on {CPU1} --pool proofs`" in capsys.readouterr().err
    monkeypatch.delenv("RESEARCH_RUN_ID")
    assert CHECK.main(shlex.split(want)[4:] + ["--base", "HEAD"]) == 0          # the named command records it
    [cmd] = launched
    assert cmd[cmd.index("--on") + 1] == CPU1 and f"CHECK_SLOTS={slot_only.POOLS['research']}" in cmd


def test_the_lane_pool_is_one_research_run_on_auto_spills_to():
    assert CHECK.LANE_POOL in auto.SPILL and CHECK.LANE_POOL[1] in slot_only.POOLS
