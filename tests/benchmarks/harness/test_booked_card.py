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
    name = "llama32-1b-fp8-1gpu-bandwidth-served"
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
