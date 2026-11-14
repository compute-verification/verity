"""The recursion bench's CLI (`run.py`): the job each `run` launches (its node, queue, lease, limit, steps and
environment), and the commands that read records back (`result`, `table`, `store`). No node or store: `run` prints
its command (`--dry-run`), and the records are made here (`test_recursion_e2e.job`)."""
from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest
import rec_profile as RP
import run as RUN
from test_recursion_e2e import job, row


def test_result_command_writes_both_files(tmp_path: Path) -> None:
    o = job(tmp_path)
    assert RUN.main(["result", "--dir", str(o), "--out", str(tmp_path / "p.json"), "--md", str(tmp_path / "t.md")]) == 0
    doc = json.loads((tmp_path / "p.json").read_text())
    assert doc["schema"] == RP.SCHEMA and (tmp_path / "t.md").read_text().startswith("Recursion bench: K = 4096")


def test_a_gpu_run_is_a_gpu_host_job_on_node_1s_queue(capsys) -> None:
    assert RUN.main(["run", "--dry-run", "--N", "16", "--steps", "lean,build", "--modes", "dev"]) == 0
    cmd = capsys.readouterr().out.split()
    assert cmd[:6] == ["research", "run", "--queue", "--on", "vy-nebius-1", "--gpu-host"]
    assert "--gpus" not in cmd                              # 85 refuses a job holding a lease: each prove leases its own
    assert cmd[cmd.index("--") + 1:][:2] == ["bash", "benchmarks/recursion_e2e/job.sh"]
    assert "N=16" in cmd and "STEPS=build,lean" in cmd     # the pipeline's order, whatever order --steps gives
    assert "--preemptible" not in cmd
    assert RUN.main(["run", "--dry-run", "--node", "vy-nebius-2", "--preemptible", "--oprove-gpu", "1"]) == 0
    cmd = capsys.readouterr().out.split()
    assert cmd[4] == "vy-nebius-2" and "--preemptible" in cmd[:cmd.index("--")]


def test_env_passes_through_and_must_be_key_value(capsys) -> None:
    assert RUN.main(["run", "--dry-run", "--env", "LEVELS=L0", "--env", "GPU_PREEMPTIBLE="]) == 0
    tail = capsys.readouterr().out.split(" -- ")[1].split()
    assert tail[-2:] == ["LEVELS=L0", "GPU_PREEMPTIBLE="]
    with pytest.raises(SystemExit, match="not KEY=VALUE"):
        RUN.main(["run", "--dry-run", "--env", "levels"])


def test_unknown_step_is_refused() -> None:
    with pytest.raises(SystemExit, match="ostage"):
        RUN.main(["plan", "--steps", "build,ostage"])


def test_a_run_id_is_read_from_its_records_not_its_result_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(RUN.subprocess, "run", lambda *a, **k: pytest.fail("load ran a command"))
    out = tmp_path / ".research" / "runs" / "r20261024-000000-abcd"
    job(out)
    (out / "out" / "recursion_e2e.json").write_text(json.dumps({"schema": RP.SCHEMA, "rows": [], "stale": True}))
    doc = RUN.load("r20261024-000000-abcd")
    assert "stale" not in doc and doc["run"] == "r20261024-000000-abcd" and row(doc, "inner M0")["ok"] is True


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


def test_a_firewalled_run_is_a_plain_run_on_a_cpu_node(capsys) -> None:
    assert RUN.main(["run", "--dry-run", "--steps", "vcap", "--repeat", "1"]) == 0
    cmd = capsys.readouterr().out.split()
    head, job = cmd[:cmd.index("--")], cmd[cmd.index("--") + 1:]
    assert head[:4] == ["research", "run", "--on", "vy-nebius-cpu-2"] and "--queue" not in head and "--gpu-host" not in head
    assert head[head.index("--timeout") + 1] == f"{RUN.SETUP_MIN + 1}m"     # a plain run's limit is its --timeout
    assert job[:5] == ["taskset", "-c", "0-31", "bash", "benchmarks/recursion_e2e/job.sh"]
    assert "GPU=0" in job and f"FLOCK_WORK={RUN.CPU_WORK}" in job
    assert RUN.main(["run", "--dry-run", "--on", "vy-nebius-cpu-5", "--cores", "32-63", "--env", "FLOCK_WORK=/x"]) == 0
    cmd = capsys.readouterr().out.split()
    assert cmd[3] == "vy-nebius-cpu-5" and cmd[cmd.index("taskset") + 2] == "32-63" and cmd[-1] == "FLOCK_WORK=/x"
    with pytest.raises(SystemExit, match="only through its slots"):
        RUN.main(["run", "--dry-run", "--on", "vy-nebius-cpu-1"])
    with pytest.raises(SystemExit, match="has no GPU"):
        RUN.main(["run", "--dry-run", "--on", "vy-nebius-cpu-2", "--oprove-gpu", "1"])


def test_a_cpu_run_on_auto_or_a_named_cpu_node_is_a_command_research_run_takes(capsys) -> None:
    """`--on auto` is research run's queue itself, which refuses --queue: a CPU run there is the queue's options without
    --queue or --gpu-host, its job on a CPU node's GPU=0 and work tree, since the queue may place it on one. A named CPU
    node is a plain run there; a GPU run on auto keeps its GPU host."""
    assert RUN.main(["run", "--dry-run", "--on", "auto", "--steps", "vcap", "--repeat", "1"]) == 0
    cmd = capsys.readouterr().out.split()
    head, job = cmd[:cmd.index("--")], cmd[cmd.index("--") + 1:]
    assert head[:4] == ["research", "run", "--on", "auto"] and "--queue" not in head and "--gpu-host" not in head
    assert head[head.index("--max-min") + 1] == str(RUN.SETUP_MIN + 1) and "--cpus" in head and "--timeout" not in head
    assert job[:2] == ["bash", "benchmarks/recursion_e2e/job.sh"] and "GPU=0" in job and f"FLOCK_WORK={RUN.CPU_WORK}" in job
    assert RUN.main(["run", "--dry-run", "--on", "auto", "--cpu", "--K", "64", "--N", "16", "--modes", "dev"]) == 0
    cmd = capsys.readouterr().out.split()
    assert cmd[:4] == ["research", "run", "--on", "auto"] and "--queue" not in cmd and "--gpu-host" not in cmd
    assert "GPU=0" in cmd
    assert RUN.main(["run", "--dry-run", "--on", "vy-nebius-cpu-3"]) == 0
    cmd = capsys.readouterr().out.split()
    head, job = cmd[:cmd.index("--")], cmd[cmd.index("--") + 1:]
    assert head[:4] == ["research", "run", "--on", "vy-nebius-cpu-3"] and "--queue" not in head and "--gpu-host" not in head
    assert job[:3] == ["taskset", "-c", "0-31"] and "GPU=0" in job
    assert RUN.main(["run", "--dry-run", "--on", "auto", "--oprove-gpu", "1"]) == 0
    cmd = capsys.readouterr().out.split()
    assert cmd[:5] == ["research", "run", "--on", "auto", "--gpu-host"] and "--queue" not in cmd and "GPU=0" not in cmd


def test_e2e_is_one_pass_with_its_negative(capsys) -> None:
    assert RUN.main(["run", "--dry-run", "--e2e", "--cpu", "--K", "64", "--N", "16", "--modes", "dev"]) == 0
    cmd = capsys.readouterr().out.split()
    assert "--gpu-host" not in cmd and cmd[3] == "vy-nebius-cpu-2"
    assert "STEPS=build,inner,vstage,vcap,oprove,lean,negative" in cmd and "REPEAT=1" in cmd and "GPU=0" in cmd
    assert "WARM=0" in cmd and "RUNS=1" in cmd and "NEG=challenge,query" in cmd


def test_release_is_one_pass_whose_limit_holds_every_window(capsys) -> None:
    assert RUN.main(["run", "--dry-run", "--steps", "oprove"]) == 0
    plain = capsys.readouterr().out.split()
    assert RUN.main(["run", "--dry-run", "--steps", "oprove", "--release", "--env", "TREE_FROM=local"]) == 0
    cmd = capsys.readouterr().out.split()
    assert "RELEASE=1" in cmd and "REPEAT=1" in cmd and "WARM=0" in cmd and "RUNS=1" in cmd and cmd[-1] == "TREE_FROM=local"
    assert "RELEASE_BUDGET=1" in cmd
    limit = {c: int(x[x.index("--timeout") + 1][:-1]) for c, x in (("plain", plain), ("release", cmd))}
    assert limit["release"] - RUN.RELEASE_LEAD_MIN > 1.9 * limit["plain"] - 2 * RUN.SETUP_MIN


def test_store_puts_each_result_the_merge_and_the_table(tmp_path: Path, monkeypatch, capsys) -> None:
    (tmp_path / "a.json").write_text(json.dumps(RP.read(job(tmp_path / "a")) | {"ref": "rA"}))
    monkeypatch.setattr(RUN.subprocess, "run", lambda *a, **k: pytest.fail("store ran a command"))
    art = tmp_path / "art"
    assert RUN.main(["store", str(tmp_path / "a.json"), "--merge", "--dir", str(art)]) == 0
    cmd = shlex.split(capsys.readouterr().out)
    assert cmd[:3] == ["research", "data", "put"] and cmd[cmd.index("--kind") + 1] == RUN.KIND
    assert cmd[cmd.index("--tree") + 1] == str(art) and "--preserve" in cmd
    assert sorted(p.name for p in art.iterdir()) == ["merged.json", "rA.json", "table.md"]
    meta = json.loads(cmd[cmd.index("--meta") + 1])
    assert meta["benchmark"] == "recursion_e2e" and meta["results"] == ["rA"]
    with pytest.raises(SystemExit, match="not empty"):
        RUN.main(["store", str(tmp_path / "a.json"), "--dir", str(art)])


def test_forge_after_runs_after_lean_whenever_its_kinds_are_forged(capsys) -> None:
    def job_env(*argv: str) -> list[str]:
        assert RUN.main(["run", "--dry-run", *argv]) == 0
        cmd = capsys.readouterr().out.split()
        return cmd[cmd.index("--") + 1:]

    env = job_env("--forge", "coef,sum")
    assert "STEPS=build,inner,icheck,vstage,vcap,oprove,oreplay,lean,forge-after,negative" in env
    assert "FORGE=coef,sum" in env and "FORGE_AFTER=coef" in env
    assert "STEPS=build,inner,icheck,vstage,vcap,oprove,oreplay,lean,negative" in job_env()   # no forgery, no forge-after
    assert "FORGE=" in job_env() and not any(e.startswith("FORGE_AFTER") for e in job_env("--forge", "sum"))
    assert "forge-after" not in " ".join(job_env("--forge", "coef", "--no-forge-after"))
    assert "forge-after" not in " ".join(job_env("--forge", "coef", "--steps", "build,inner"))   # nothing to forge after
    # asked for by name, it forges every kind it knows (job.sh's FORGE_AFTER default)
    assert "STEPS=lean,forge-after" in job_env("--steps", "forge-after,lean") and "FORGE_AFTER=" in job_env("--steps", "lean,forge-after")
    def limit(*argv: str) -> int:
        assert RUN.main(["run", "--dry-run", *argv]) == 0
        cmd = capsys.readouterr().out.split()
        return int(cmd[cmd.index("--timeout") + 1][:-1])

    assert limit("--forge", "coef") - limit() == pytest.approx(RUN.MARGIN * RUN.FW_S["L0"] / 60 + RUN.OTHER_MIN["forge-after"], abs=1)
    # the message forgery proves alg-p1, which it leaves honest, beside alg-p0
    alg = RUN.FW_S["alg-p0"] + RUN.FW_S["alg-p1"]
    assert limit("--forge", "message") - limit() == pytest.approx(RUN.MARGIN * alg / 60 + RUN.OTHER_MIN["forge-after"], abs=1)
    assert limit("--forge", "sum") - limit() == pytest.approx(RUN.MARGIN * (RUN.FW_S["L0"] + RUN.FW_S["L3"] + alg) / 60, abs=1)
    assert all(RUN.statements(name) for v in RUN.FORGED.values() for name in v.split(","))   # each names a measured statement
    with pytest.raises(SystemExit, match="--forge: nope"):
        RUN.main(["run", "--dry-run", "--forge", "coef,nope"])
    assert RUN.main(["run", "--dry-run", "--steps", "lean,forge-after", "--repeat", "1"]) == 0
    cmd = capsys.readouterr().out.split()
    assert cmd[cmd.index("--timeout") + 1] == f"{RUN.SETUP_MIN + RUN.OTHER_MIN['lean'] + RUN.OTHER_MIN['forge-after']}m"


def test_a_wired_run_budgets_its_link_part_and_its_forgery_as_the_largest_statement(capsys) -> None:
    def launched(*argv: str) -> list[str]:
        assert RUN.main(["run", "--dry-run", *argv]) == 0
        return capsys.readouterr().out.split()

    def limit(*argv: str) -> int:
        cmd = launched(*argv)
        return int(cmd[cmd.index("--timeout") + 1][:-1])

    assert "REC_LINKS=1" in launched("--env", "REC_LINKS=1")
    big = max(RUN.FW_S.values())
    assert limit("--env", "REC_LINKS=1") - limit() == pytest.approx(RUN.MARGIN * 2 * big / 60, abs=1)
    assert limit("--env", "REC_LINKS=1", "--oprove-runs", "2") - limit("--oprove-runs", "2") == pytest.approx(
        RUN.MARGIN * 3 * big / 60, abs=1)
    assert limit("--env", "REC_LINKS=1", "--env", "REC_LINKS=0") == limit()
