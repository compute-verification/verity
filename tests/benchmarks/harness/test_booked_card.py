from pathlib import Path

import pytest

from verity_bench import jobs
from verity_bench.cli import main
from verity_bench.tool import VLLM_BENCH_POINT


def test_a_vllm_point_on_a_booked_card_leases_that_card_on_its_cores(monkeypatch, capsys):
    """A booked card and its cores are the launch's `--env GPU_LEASE_ON=N --env GPU_LEASE_CORES=CPUS`: the point's lease
    asks for that card alone and pins the point to those CPUs; the flags of the launch before go."""

    from verity_bench import cli

    root = Path(__file__).resolve().parents[3]
    name = "llama32-1b-fp8-1gpu-bandwidth-served-h200"
    monkeypatch.delenv("GPU_LEASE_UUID", raising=False)
    monkeypatch.setenv("GPU_LEASE_ON", "2")
    monkeypatch.setenv("GPU_LEASE_CORES", "20-39")
    monkeypatch.setattr(jobs, "lease_host", lambda: True)
    plan = jobs.make_plan(name, root=root, commit="a" * 40, lookup=lambda identity, commit: None)
    commands = []
    monkeypatch.setattr(cli, "_make_plan", lambda args: plan)
    dispatch = jobs.dispatch
    monkeypatch.setattr(jobs, "dispatch", lambda *a, **k: dispatch(
        *a, **k, execute=lambda command, cwd: commands.append(list(command)) or 0))
    for item in plan.points:
        assert main(["run", name, "--point", item.point.id]) == 0
    assert len(commands) == len(plan.points) == 2
    for command, item in zip(commands, plan.points, strict=True):
        assert command[:command.index("--")] == ["gpu-lease", "1", "--wait", "--on", "2", "--cores", "20-39", "--max-min",
                                                 "120", "--preemptible"]
        assert command[command.index("--") + 1:] == [
            "env", "VERITY_VLLM_BENCH_KV_RECEIPTS=on", "VERITY_BENCH_SETUP=benchmarks/harness/setup/vllm_encoded.sh",
            "bash", "benchmarks/harness/run_point.sh", "--spec", f"benchmarks/harness/specs/{name}.toml", "--point",
            item.point.id, "--identity", item.identity]
        assert VLLM_BENCH_POINT.parse(command)["identity"] == item.identity
    for argv in (["--direct", "vy-spot-glm53", "--card", "2"], ["--cores", "20-39"]):
        with pytest.raises(SystemExit):
            main(["run", name, *argv])
        assert "unrecognized arguments" in capsys.readouterr().err
    assert len(commands) == 2


def test_a_booked_point_s_research_run_flags_name_its_slot_cards_end_and_bound_and_are_timed(tmp_path):
    """A booked period's point (`jobs.booked`): its slot of the pool, the spec's max_minutes as the bound `--end` must fit,
    `--timed`; a vLLM point's run leases the booked cards, a steady-state point's names its one card to its node side, which
    leases it; a booking of the wrong number of cards, or a spec with no bound, is refused."""

    from dataclasses import replace

    from steady_standins import tree
    from verity_bench import steady_plan
    from verity_bench.vllm.presets import Admission

    root = Path(__file__).resolve().parents[3]
    spec = jobs.make_plan("llama32-1b-fp8-1gpu-bandwidth-served-h200", root=root, commit="a" * 40,
                          lookup=lambda identity, commit: None).spec
    end = "2026-11-19T05:00:00Z"
    assert jobs.booked(spec, "gpu", "g2", end, [2]) == [
        "--pool", "gpu", "--slot", "g2", "--card", "2", "--end", end, "--timeout", "120m", "--timed"]
    with pytest.raises(ValueError, match=r"runs on 1 card\(s\), and the booking names 2: \[2, 3\]"):
        jobs.booked(spec, "gpu", "g2", end, [2, 3])
    with pytest.raises(ValueError, match="has no admission max_minutes"):
        jobs.booked(replace(spec, admission=Admission(None, None, None)), "gpu", "g2", end, [2])
    steady_root, path = tree(tmp_path)
    steady = steady_plan.make_plan(steady_root / path, root=steady_root, lookup=lambda s, i, identity: (None, None)).spec
    minutes = steady.admission.max_minutes
    assert jobs.booked(steady, "gpu", "g1", end, [3]) == [
        "--pool", "gpu", "--slot", "g1", "--env", "GPU_LEASE_ON=3", "--end", end, "--timeout", f"{minutes}m", "--timed"]
