from pathlib import Path

import pytest

from verity_bench import jobs
from verity_bench.cli import main
from verity_bench.tool import VLLM_BENCH_POINT


def test_a_vllm_point_on_a_booked_card_takes_the_booked_slot_and_leases_that_card_on_its_cores(monkeypatch, capsys):
    from research.pods.nebius.slot_only import through_slot

    from verity_bench import cli

    root = Path(__file__).resolve().parents[3]
    name = "llama32-1b-fp8-1gpu-bandwidth-served-h200"
    monkeypatch.delenv("VERITY_BENCH_VENV", raising=False)
    plan = jobs.make_plan(name, root=root, commit="a" * 40, lookup=lambda identity, commit: None)
    commands = []
    monkeypatch.setattr(cli, "_make_plan", lambda args: plan)
    dispatch = jobs.dispatch
    monkeypatch.setattr(jobs, "dispatch", lambda *a, **k: dispatch(
        *a, **k, execute=lambda command, cwd: commands.append(list(command)) or 0))
    assert main(["run", name, "--direct", "vy-spot-glm53", "--card", "2", "--cores", "20-39"]) == 0
    assert len(commands) == len(plan.points) == 2
    for command, item in zip(commands, plan.points, strict=True):
        flags, workload = command[:command.index("--")], command[command.index("--") + 1:]
        assert flags[:4] == ["research", "run", "--on", "vy-spot-glm53"]
        assert [flags[i + 1] for i, t in enumerate(flags) if t == "--env"] == [
            "VERITY_VLLM_BENCH_KV_RECEIPTS=on", "VERITY_BENCH_SETUP=benchmarks/harness/setup/vllm_encoded.sh"]
        assert workload[:3] == ["taskset", "-c", "20-39"] and through_slot(workload[3:])
        lease = workload[workload.index("gpu-lease"):]
        assert lease[:lease.index("--")] == ["gpu-lease", "1", "--wait", "--on", "2", "--cores", "20-39", "--max-min",
                                             "120", "--preemptible"]
        assert lease[lease.index("--") + 1:] == ["bash", "benchmarks/harness/run_point.sh", "--spec",
                                                 f"benchmarks/harness/specs/{name}.toml", "--point", item.point.id,
                                                 "--identity", item.identity]
        assert VLLM_BENCH_POINT.parse(command)["identity"] == item.identity
    for argv, why in ((["--direct", "vy-spot-glm53", "--card", "2"], "--card 2 needs --cores"),
                      (["--cores", "20-39"], "need --direct NODE")):
        with pytest.raises(SystemExit):
            main(["run", name, *argv])
        assert why in capsys.readouterr().err
    assert len(commands) == 2
