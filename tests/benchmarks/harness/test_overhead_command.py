"""`verity bench overhead`'s command: each step a research run, and the table under one."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest
from overhead_standins import CUBIN, REPO, Store, stored_points, timed
from overhead_standins import scheme as scheme
from overhead_standins import stamped as stamped
from verity_bench import kit
from verity_bench import overhead as O
from verity_bench import overhead_records as R
from verity_bench.cli import main

SMALL = "llama3.2-1b"

# ---- the command ------------------------------------------------------------------------------------------------------------------


def job(argv: list[str], minutes: str, *lease: str) -> list[str]:
    """The node job's arguments in a launch: the command after `research run`'s `--` is the node job under a lease of one
    GPU (with the gpu-lease options `lease`) for the run's bound, which the job takes itself (a node whose GPUs gpu-lease
    doesn't hand out refuses `--gpus`)."""

    cmd = argv[argv.index("--") + 1:]
    head = ["gpu-lease", "1", "--wait", *lease, "--max-min", minutes, "--", "bash"]
    assert cmd[:len(head)] == head and cmd[len(head)] == R.NODE_JOB
    return cmd[len(head) + 1:]


def test_the_command_launches_each_step_as_a_research_run_on_one_gpu_and_ships_the_points(tmp_path, monkeypatch, capsys):
    s = Store(tmp_path)
    art = stored_points(tmp_path, s, "r20261104-020000-0001")
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "producer"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(s.config))
    monkeypatch.delenv("VERITY_BENCH_VENV", raising=False)
    launched = []

    def execute(argv, root):
        argv = list(argv)
        sent = [Path(argv[i + 1]) for i, a in enumerate(argv) if a == "--send"]
        launched.append((argv, [p.read_bytes() for p in sent]))
        return 0

    monkeypatch.setattr(R, "_execute", execute)
    assert main(["overhead", "points", "--card", "h200"]) == 0
    assert main(["overhead", "run", "--scheme", "stock", "--point", "L", "--local-only"]) == 0
    assert main(["overhead", "run", "--scheme", "pearl-plus", "--cubin", s.cubin, "--local-only"]) == 0
    (points, _), (run, sent), (armed, armed_sent) = launched
    assert job(armed, "60") == ["run", "--card", "h200", "--scheme", "pearl-plus", "--points", "points.json",
                                             "--points-art", art, "--point", "all", "--cubin", "arm.cubin",
                                             "--cubin-art", s.cubin]
    assert armed_sent == [Path(s.store.fetch(art)).read_bytes(), CUBIN]
    with pytest.raises(SystemExit):
        main(["overhead", "run", "--scheme", "stock", "--cubin", s.cubin, "--local-only"])
    assert "stock has no arm to load it" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["overhead", "run", "--scheme", "pearl-plus", "--cubin", "art:" + "f" * 64, "--local-only"])
    assert "isn't in the store" in capsys.readouterr().err
    (tmp_path / "cubins").mkdir()
    (tmp_path / "cubins" / "a.cubin").write_bytes(CUBIN)
    tree = s.store.put_artifact("cubin/v1", {}, tree=tmp_path / "cubins")
    with pytest.raises(SystemExit):
        main(["overhead", "run", "--scheme", "pearl-plus", "--cubin", tree, "--local-only"])
    assert "isn't one file" in capsys.readouterr().err
    for argv in (points, run):
        assert argv[:4] == ["research", "run", "--on", "vy-dc-h200"] and "--require-result" in argv
        assert argv[argv.index("--cwd") + 1] == "source" and "--gpus" not in argv
    assert job(points, "150") == ["points", "--card", "h200", "--model", O.KERNEL_MODEL, "--precision", "fp8"]
    assert job(run, "60") == ["run", "--card", "h200", "--scheme", "stock", "--points", "points.json",
                              "--points-art", art, "--point", "L"]
    assert sent == [Path(s.store.fetch(art)).read_bytes()] and run[run.index("--timeout") + 1] == "60m"
    with pytest.raises(SystemExit):
        main(["overhead", "points", "--card", "rtx-pro-6000"])
    assert "has no default node" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["overhead", "run", "--card", "h200", "--local-only"])
    assert "--scheme S, or --record RUN" in capsys.readouterr().err


def test_verity_bench_overhead_is_the_door_s_runner(tmp_path, monkeypatch, capsys):
    """`verity bench overhead` runs benchmarks/overhead/run.py, which is `verity-bench overhead`."""

    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "empty"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(Store(tmp_path).config))
    monkeypatch.setattr(sys, "argv", ["run.py", "table", "--json", "--local-only"])
    with pytest.raises(SystemExit) as done:
        runpy.run_path(str(REPO / "benchmarks" / "overhead" / "run.py"), run_name="__main__")
    assert done.value.code == 0
    table = json.loads(capsys.readouterr().out)
    assert table["rows"] == [] and table["census"]["world_flop_per_year"] == 9.36e30


def test_under_a_research_run_the_table_records_its_metrics_and_prints_the_same(tmp_path, monkeypatch, capsys, scheme):
    """Under a research run, `verity bench overhead table` writes run.py's metrics into the run's result through the kit:
    each the γ of record of its row, measured, with its interval, the scheme gate as its target, its run and points' art,
    and its note; with no γ of record, no value and why, and a failed validation naming it. What it prints is the same in
    a run as outside one, where it writes nothing."""

    s = Store(tmp_path)
    art = stored_points(tmp_path, s, "r20261104-020000-0001")
    run = timed(tmp_path, s, art, "pearl-plus", "r20261104-030000-0002")
    capsys.readouterr()
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "producer"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(s.config))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261114-000000-0000")
    runner = REPO / "benchmarks" / "overhead" / "run.py"
    metrics = runpy.run_path(str(runner))["METRICS"]
    assert metrics == {"benchmarks/pearl-plus-gamma-c-h200": "pearl-plus/h200/C",
                       "benchmarks/pearl-plus-gamma-l-h200": "pearl-plus/h200/L"}

    def table(within: Path | None = None) -> str:
        if within is None:
            monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
        else:
            monkeypatch.setenv("RESEARCH_RUN_DIR", str(within))
        monkeypatch.setattr(sys, "argv", ["run.py", "table", "--local-only"])
        with pytest.raises(SystemExit) as done:
            runpy.run_path(str(runner), run_name="__main__")
        assert done.value.code == 0
        return capsys.readouterr().out

    def result(within: Path) -> dict:
        return json.loads((within / kit.RESULT).read_text())

    before = tmp_path / "run-1"
    printed = table(before)
    assert printed == table() and printed.endswith(f"\n## {SMALL}\n\nNo labelled run times {SMALL} "
                                                   "(`run_of_record=overhead`): it has no γ.\n")
    got = result(before)
    assert [(m["name"], m.get("member")) for m in got["measurements"]] == [(n, w) for n in metrics for w in (None, SMALL)]
    for m, point in zip(got["measurements"], [p for p in O.POINTS for _ in (0, 1)]):
        assert m["value"] is None and m["mark"] == "measured" and m["target"] == O.GAMMA_MAX and m["verdict"] is None
        on = f" on {SMALL}" if m.get("member") else ""
        assert m["why"] == f"no γ of record: no labelled run times pearl-plus at h200's {point}{on} (`run_of_record=overhead`)"
    assert got["validation"]["status"] == "failed" and got["validation"]["evidence"] == [R.TABLE]
    assert got["validation"]["detail"].startswith("0 of 2 metrics have a γ of record")

    assert R.record_run(s.store, run) == []
    after = tmp_path / "run-2"
    printed = table(after)
    assert printed == table() and "| pearl-plus | h200 | C |" in printed
    got, kept = result(after), json.loads((after / R.TABLE).read_text())
    rows = {r["point"]: r for r in kept["rows"]}
    assert got["workload_fingerprint"]["runs"] == [run] and got["measurement_files"] == [R.TABLE]
    own = [m for m in got["measurements"] if "member" not in m]
    assert [m["value"] for m in got["measurements"] if "member" in m] == [None, None]
    for m, point in zip(own, O.POINTS):
        g = rows[point]["gamma"]
        assert (m["value"], m["ci95"], m["unit"], m["mark"]) == (g["gamma"], g["ci95"], R.GAMMA_UNIT, "measured")
        assert m["verdict"] == "within target" and m["inputs"] == ["measured", run, art] and "why" not in m
        assert m["note"] == O.WITHOUT_Y and m["covers"] == O.covers()
    assert got["validation"] == {"status": "passed", "checker": "verity_bench.overhead_records.of_record",
                                 "detail": "2 of 2 metrics have a γ of record (the source check, `of_record`)",
                                 "evidence": [R.TABLE]}


def test_llama32_1b_s_linears_are_its_pinned_config_s():
    hf = json.loads((REPO / "adapters/vllm/verity_vllm/engine/profiles/hf_configs/LLAMA32_1B.config.json").read_text())
    d, q, kv = hf["hidden_size"], hf["num_attention_heads"] * hf["head_dim"], hf["num_key_value_heads"] * hf["head_dim"]
    assert O.MODELS[SMALL] == {"qkv": (q + 2 * kv, d), "o": (d, q), "gate_up": (2 * hf["intermediate_size"], d),
                               "down": (d, hf["intermediate_size"])}


def test_the_1b_points_launch_on_one_named_gpu_and_cpus_and_their_gamma_is_each_metric_s_member(tmp_path, monkeypatch,
                                                                                             capsys, scheme):
    """Card 6 of the H200 node and CPUs 72-95 (its NUMA node): the sweep and the timed run launch with gpu-lease's `--on`
    and `--cores`, which the source check accepts; a 1B run times at the 1B points of record, not the 70B ones; and its γ
    is printed under its own heading and written as each metric's `member`, the 70B reading unchanged beside it."""

    s = Store(tmp_path)
    big = stored_points(tmp_path, s, "r20261104-020000-0001")
    small = stored_points(tmp_path, s, "r20261104-020000-0003", SMALL)
    assert s.store.get_manifest(small).meta["model"] == SMALL and R.current_points(s.store, "h200") == big
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "producer"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(s.config))
    monkeypatch.delenv("VERITY_BENCH_VENV", raising=False)
    launched = []
    monkeypatch.setattr(R, "_execute", lambda argv, root: launched.append(list(argv)) or 0)
    on = ["--on", "vy-dc-h200", "--gpu", "6", "--cores", "72-95", "--model", SMALL, "--local-only"]
    assert main(["overhead", "points", *on]) == 0
    assert main(["overhead", "run", "--scheme", "pearl-plus", "--cubin", s.cubin, *on]) == 0
    with pytest.raises(SystemExit):
        main(["overhead", "run", "--scheme", "pearl-plus", "--points", big, *on])
    assert f"{big} holds the points of the h200 and llama3.1-70b, not the h200 and {SMALL}" in capsys.readouterr().err
    points, run = launched
    lease = ("--on", "6", "--cores", "72-95")
    assert job(points, "150", *lease) == ["points", "--card", "h200", "--model", SMALL, "--precision", "fp8"]
    assert job(run, "60", *lease)[:8] == ["run", "--card", "h200", "--scheme", "pearl-plus", "--points", "points.json",
                                          "--points-art"] and job(run, "60", *lease)[8] == small
    for argv in (points, run):
        assert R.door_why({"argv": argv[argv.index("--") + 1:]}, "r", argv[argv.index("bash") + 2]) is None

    seventy = timed(tmp_path, s, big, "pearl-plus", "r20261104-030000-0002")
    one = timed(tmp_path, s, small, "pearl-plus", "r20261104-030000-0004")
    assert R.record_run(s.store, seventy) == [] and R.record_run(s.store, one) == []
    table = R.build(s.store, local_only=True)
    printed = R.render(table)
    head, _, tail = printed.partition(f"\n## {SMALL}\n\n")
    assert f"| {seventy} |" in head and one not in head and f"| {one} |" in tail and O.covers(SMALL) in tail
    rows = {(r["model"], r["point"]): r for r in table["rows"]}
    written = R.measurements(table, {"benchmarks/pearl-plus-gamma-c-h200": "pearl-plus/h200/C"})
    assert [(m.get("member"), m["value"], m["inputs"]) for m in written] == [
        (None, rows["llama3.1-70b", "C"]["gamma"]["gamma"], [seventy, big]),
        (SMALL, rows[SMALL, "C"]["gamma"]["gamma"], [one, small])]
    assert written[1]["extra"]["covers"] == O.covers(SMALL) and written[0]["extra"]["covers"] == table["covers"]
