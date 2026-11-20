"""The recursion bench's CLI (`run.py`): the job each `run` launches (its node, queue, lease, limit, steps and
environment), and the commands that read records back (`result`, `table`, `store`). No node or store: `run` prints
its command (`--dry-run`), and the records are made here (`test_recursion_e2e.job`)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import rec_profile as RP
import run as RUN
from test_recursion_e2e import job


def test_result_command_writes_both_files(tmp_path: Path) -> None:
    o = job(tmp_path)
    assert RUN.main(["result", "--dir", str(o), "--out", str(tmp_path / "p.json"), "--md", str(tmp_path / "t.md")]) == 0
    doc = json.loads((tmp_path / "p.json").read_text())
    assert doc["schema"] == RP.SCHEMA and (tmp_path / "t.md").read_text().startswith("Recursion bench: K = 4096")


def test_a_gpu_run_is_a_gpu_host_job_research_run_places_or_on_the_named_machines_queue(capsys) -> None:
    assert RUN.main(["run", "--dry-run", "--N", "16", "--steps", "lean,build", "--modes", "dev"]) == 0
    cmd = capsys.readouterr().out.split()
    assert cmd[:5] == ["research", "run", "--on", "auto", "--gpu-host"] and "--queue" not in cmd
    assert "--gpus" not in cmd                              # 85 refuses a job holding a lease: each prove leases its own
    assert cmd[cmd.index("--") + 1:][:2] == ["bash", "benchmarks/recursion_e2e/job.sh"]
    assert "N=16" in cmd and "STEPS=build,lean" in cmd     # the pipeline's order, whatever order --steps gives
    assert "--preemptible" not in cmd
    assert RUN.main(["run", "--dry-run", "--node", "gpu-node-2", "--preemptible", "--oprove-gpu", "1"]) == 0
    cmd = capsys.readouterr().out.split()
    assert cmd[2:6] == ["--queue", "--on", "gpu-node-2", "--gpu-host"] and "--preemptible" in cmd[:cmd.index("--")]


def test_env_passes_through_and_must_be_key_value(capsys) -> None:
    assert RUN.main(["run", "--dry-run", "--env", "LEVELS=L0", "--env", "GPU_PREEMPTIBLE="]) == 0
    tail = capsys.readouterr().out.split(" -- ")[1].split()
    assert tail[-2:] == ["LEVELS=L0", "GPU_PREEMPTIBLE="]
    with pytest.raises(SystemExit, match="not KEY=VALUE"):
        RUN.main(["run", "--dry-run", "--env", "levels"])


def test_unknown_step_is_refused() -> None:
    with pytest.raises(SystemExit, match="ostage"):
        RUN.main(["plan", "--steps", "build,ostage"])


def test_a_run_id_whose_records_arent_fetched_is_refused_with_the_fetch(tmp_path: Path, monkeypatch) -> None:
    """`load` reads the records `research fetch` put under RESEARCH_RUNS_DIR and fetches nothing itself."""
    monkeypatch.setenv("RESEARCH_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(RUN.subprocess, "run", lambda *a, **k: pytest.fail("load ran a command"))
    with pytest.raises(SystemExit, match=r"r20261024-000000-beef: .*runs/r20261024-000000-beef/out holds neither "
                                         r"recursion_e2e\.json nor steps/.*`research fetch --all r20261024-000000-beef`"):
        RUN.load("r20261024-000000-beef")
    job(tmp_path / "runs" / "r20261024-000000-beef")
    assert RUN.load("r20261024-000000-beef")["run"] == "r20261024-000000-beef"


def test_table_refuses_another_schema(tmp_path: Path) -> None:
    (tmp_path / "x.json").write_text(json.dumps({"schema": "proof-throughput/v1"}))
    with pytest.raises(SystemExit, match="not recursion-e2e/v1"):
        RUN.main(["table", str(tmp_path / "x.json")])


def test_a_direct_run_behind_the_zk_server_is_a_plain_run_on_the_cpu_node_it_names(capsys) -> None:
    """--direct is a plain run on --on MACHINE, pinned to --cores; it names its machine and has no GPU host. Which machines
    take plain runs is research run's launcher's to refuse, so no machine is named here."""
    assert RUN.main(["run", "--dry-run", "--direct", "--on", "cpu-node-2", "--steps", "vcap", "--repeat", "1"]) == 0
    cmd = capsys.readouterr().out.split()
    head, job = cmd[:cmd.index("--")], cmd[cmd.index("--") + 1:]
    assert head[:4] == ["research", "run", "--on", "cpu-node-2"] and "--queue" not in head and "--gpu-host" not in head
    assert head[head.index("--timeout") + 1] == f"{RUN.SETUP_MIN + 1}m"     # a plain run's limit is its --timeout
    assert job[:5] == ["taskset", "-c", "0-31", "bash", "benchmarks/recursion_e2e/job.sh"]
    assert "GPU=0" in job and f"FLOCK_WORK={RUN.CPU_WORK}" in job
    assert RUN.main(["run", "--dry-run", "--direct", "--on", "cpu-node-5", "--cores", "32-63", "--env", "FLOCK_WORK=/x"]) == 0
    cmd = capsys.readouterr().out.split()
    assert cmd[3] == "cpu-node-5" and cmd[cmd.index("taskset") + 2] == "32-63" and cmd[-1] == "FLOCK_WORK=/x"
    with pytest.raises(SystemExit, match="names its machine"):
        RUN.main(["run", "--dry-run", "--direct"])
    with pytest.raises(SystemExit, match="no GPU host"):
        RUN.main(["run", "--dry-run", "--direct", "--on", "cpu-node-2", "--oprove-gpu", "1"])


#: a plain run on a CPU node, whose limit is its --timeout
PLAIN = ("--direct", "--on", "cpu-node-2")


WINDOW = ["--window", "/workspace/kept/serving-rows", "--units", "0-15", "--subcircuit", "rope-head/d64/neox-bf16",
          "--serves", "llama-3.2-1b-h200"]


def test_a_window_run_stages_on_the_machine_that_keeps_the_window_and_sends_nothing(capsys) -> None:
    """--window is stage.sh over the window where it lies: with --direct a plain run on its calendar row's cores, the
    job's own work tree (no CPU node's cache), and no file sent with the run."""
    assert RUN.main(["run", "--dry-run", "--direct", "--on", "window-node", "--cores", "10-41", *WINDOW]) == 0
    cmd = capsys.readouterr().out.split()
    head, job = cmd[:cmd.index("--")], cmd[cmd.index("--") + 1:]
    assert head[:4] == ["research", "run", "--on", "window-node"] and "--queue" not in head and "--send" not in head
    assert job[:5] == ["taskset", "-c", "10-41", "bash", "benchmarks/recursion_e2e/stage.sh"]
    assert {"WINDOW=/workspace/kept/serving-rows", "MEMBER=0", "UNITS=0-15", "SUBCIRCUIT=rope-head/d64/neox-bf16",
            "SERVES=llama-3.2-1b-h200", "GPU=0"} <= set(job) and not any(e.startswith("FLOCK_WORK=") for e in job)
    with pytest.raises(SystemExit, match="calendar row"):
        RUN.main(["run", "--dry-run", "--direct", "--on", "window-node", *WINDOW])
    with pytest.raises(SystemExit, match="never copied"):
        RUN.main(["run", "--dry-run", *WINDOW])
    with pytest.raises(SystemExit, match="never copied"):
        RUN.main(["run", "--dry-run", "--on", "auto", *WINDOW])
    with pytest.raises(SystemExit, match="CPU prover"):
        RUN.main(["run", "--dry-run", "--direct", "--on", "window-node", "--cores", "10-41", "--oprove-gpu", "1", *WINDOW])
    with pytest.raises(SystemExit, match="absolute path"):
        RUN.main(["run", "--dry-run", "--direct", "--on", "window-node", "--cores", "10-41", "--window", "rows",
                  *WINDOW[2:]])
