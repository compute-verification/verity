"""The Match driver's plan (`verity_vllm.pipeline.match`), each case in this process: every stage's command whole, the stages of a
phase in their order, the weights' files a `--checkpoint-dir` names and the paths of a capture given by its log.  A dry run prints
each stage's line and writes nothing.  A GPU phase takes its lease before its first stage, runs every stage though one fails and
writes their statuses beside the run; a stage name without a command leaves no record, and a gates stage that raises is a failed
stage.  The gates stage writes the gates, the card, its copy under `--cards-dir` and its summary, and prints the card's head and
the gates' table.  The driver gives other code what its `__all__` lists and hands on no other module's name."""
from __future__ import annotations

import ast
import json
import re
import shlex
import sys
from pathlib import Path

import pytest

from verity_vllm.pipeline import cli
from verity_vllm.pipeline import match as RC
from verity_vllm.pipeline import report as CARD

RUN = "/r/RUN"
LOG = f"{RUN}/capture/log.jsonl.gz"
GPU = ["/pod/py311", "-m", cli.__name__]
CPU = ["/pod/py312", "-m", cli.__name__]
CPU_STAGES = ["resolve", "census_reconcile", "nonint_compare", "compare", "fold", "fold_2", "root_policy", "golden", "wiring", "replay_a",
              "replay_chain", "gates"]
NO_COMMAND = ("root_policy", "wiring", "replay_a", "replay_chain", "gates")


def _workload(tmp_path, **engine):
    path = tmp_path / "wl.json"
    path.write_text(json.dumps({"requests": [{"request_id": "r0", "prompt_len": 5, "max_tokens": 4}], **({"engine": engine} if engine else {})}))
    return str(path)


def _options(workload, *given, out=RUN):
    return cli.parse("match", ["--config-id", "cid", "--profile", "prof", "--workload", workload, "--case", "B1", "--out", out,
                               "--python-gpu", GPU[0], "--python-cpu", CPU[0], *given], {})


def test_each_stage_s_command_is_its_python_the_door_one_of_its_commands_and_the_run_s_own_paths(tmp_path):
    wl = _workload(tmp_path)
    plan = RC.Plan(_options(wl, "--snapshot-steps", "0,1,2"))
    served = ["--workload", wl, "--case", "B1", "--max-num-seqs", "8", "--checkpoints", "manifests/checkpoints.json"]
    fold = [*CPU, "match", "--worker", "fold", "--log", LOG, "--profile", "prof", "--out"]
    commands = {
        "capture": [*GPU, "m1-capture", *served, "--out", f"{RUN}/capture", "--snapshot-steps", "0,1,2", "--verify-weights"],
        "control": [*GPU, "m1-capture", *served, "--out", f"{RUN}/control", "--no-observer"],
        "census": [*GPU, "census", *served, "--out", f"{RUN}/census", "--no-traces"],
        "nonint": [*GPU, "noninterference", *served, "--out", f"{RUN}/nonint", "--python", GPU[0], "--accepted-ctl", f"{RUN}/control/tokens.json"],
        "resolve": [*CPU, "resolve-log", LOG, "--profile", "prof", "--out", f"{RUN}/resolution", "--quiet"],
        "census_reconcile": [*CPU, "census", "--reconcile", f"{RUN}/census/census.json", "--log", LOG, "--profile", "prof",
                             "--report", f"{RUN}/census_reconcile"],
        "nonint_compare": [*CPU, "noninterference", "--compare", "--out", f"{RUN}/nonint", "--workload", wl, "--case", "B1",
                           "--accepted-ctl", f"{RUN}/control/tokens.json"],
        "compare": [*CPU, "fold-compare", "--log", LOG, "--snapshots", f"{RUN}/capture", "--profile", "prof", "--out", f"{RUN}/compare.json"],
        "fold": [*fold, f"{RUN}/fold_summary.json", "--persist-dir", RUN],
        "fold_2": [*fold, f"{RUN}/fold_summary_2.json"],
        "golden": [*CPU, "golden", "--check", "--out", f"{RUN}/golden.json"],
    }
    assert RC.DEFAULT_CHECKPOINTS == "manifests/checkpoints.json" and set(commands) == {*RC.GPU_STAGES, *CPU_STAGES} - set(NO_COMMAND)
    assert {st: plan.cmd(st) for st in commands} == commands
    assert [plan.cmd(st) for st in (*NO_COMMAND, "fold_other", "no-such-stage")] == [None] * 7      # no second capture: no fold of one
    for cmd in commands.values():                                               # the door parses each as it is written
        assert cmd[3] in cli.COMMANDS and cli.parse(cmd[3], cmd[4:], {}) is not None


def test_a_run_s_options_reach_the_stages_that_read_them(tmp_path, monkeypatch):
    wl = _workload(tmp_path, args={"max_num_batched_tokens": 64, "enable_prefix_caching": False}, target={"moe_construction": "padded"})
    log, ctl, other = tmp_path / "old" / "log.jsonl.gz", tmp_path / "ctl.json", str(tmp_path / "other.jsonl.gz")
    monkeypatch.chdir(tmp_path)                                                 # a log and a control named from here are whole paths in a command
    plan = RC.Plan(_options(wl, "--max-tokens", "6", "--max-num-seqs", "2", "--checkpoints", "ck.json", "--engine-arg", "enable_prefix_caching=true",
                            "--engine-arg", "seed=7", "--engine-arg", "name=x", "--snapshot-max-bytes", "777", "--snapshot-max-bytes-op", "55",
                            "--log", "old/log.jsonl.gz", "--control-tokens", "ctl.json", "--other-log", other, "--checkpoint-dir", "/ck/snap",
                            "--keep-traces", "--no-verify-weights"))
    # the workload's engine arguments, a command line's over them and after them, each value as JSON (a text that is none, quoted)
    served = ["--workload", wl, "--case", "B1", "--max-num-seqs", "2", "--checkpoints", "ck.json", "--max-tokens", "6",
              "--engine-arg=max_num_batched_tokens=64", "--engine-arg=enable_prefix_caching=true", "--engine-arg=seed=7", '--engine-arg=name="x"']
    assert plan.cmd("capture") == [*GPU, "m1-capture", *served, "--out", f"{RUN}/capture", "--snapshot-steps", "0,1,2,3,4,5",
                                   "--snapshot-max-bytes", "777", "--snapshot-max-bytes-op", "55"]
    assert plan.cmd("control") == [*GPU, "m1-capture", *served, "--out", f"{RUN}/control", "--no-observer"]
    assert plan.cmd("census") == [*GPU, "census", *served, "--out", f"{RUN}/census"]
    assert plan.cmd("nonint") == [*GPU, "noninterference", *served, "--out", f"{RUN}/nonint", "--python", GPU[0], "--accepted-ctl", str(ctl)]
    assert plan.cmd("nonint_compare")[-2:] == ["--accepted-ctl", str(ctl)]
    assert plan.cmd("resolve")[:5] == [*CPU, "resolve-log", str(log)] and plan.cmd("census_reconcile")[6:8] == ["--log", str(log)]
    assert plan.cmd("compare") == [*CPU, "fold-compare", "--log", str(log), "--snapshots", str(log.parent), "--profile", "prof",
                                   "--out", f"{RUN}/compare.json", "--safetensors", "/ck/snap/model*.safetensors"]
    fold, declared = [*CPU, "match", "--worker", "fold", "--log", str(log), "--profile", "prof", "--out"], ["--moe-construction", "padded"]
    assert plan.cmd("fold") == [*fold, f"{RUN}/fold_summary.json", *declared, "--persist-dir", RUN]
    assert plan.cmd("fold_2") == [*fold, f"{RUN}/fold_summary_2.json", *declared]
    assert plan.cmd("fold_other") == [*CPU, "match", "--worker", "fold", "--log", other, "--profile", "prof",
                                      "--out", f"{RUN}/fold_summary_other.json", *declared]
    assert plan.stage_list("cpu") == [*CPU_STAGES[:6], "fold_other", *CPU_STAGES[6:]]


def test_a_phase_s_stages_are_the_gpu_s_four_then_the_cpu_s_twelve(tmp_path):
    plan = RC.Plan(_options(_workload(tmp_path)))
    assert plan.stage_list("gpu") == ["capture", "control", "census", "nonint"] == list(RC.GPU_STAGES)
    assert plan.stage_list("cpu") == CPU_STAGES and plan.stage_list("all") == [*RC.GPU_STAGES, *CPU_STAGES]


def test_a_capture_given_by_its_log_is_recorded_by_the_paths_of_its_files(tmp_path, monkeypatch):
    wl = _workload(tmp_path)
    log, snaps, ctl = tmp_path / "old" / "log.jsonl.gz", tmp_path / "snaps", tmp_path / "ctl.json"
    monkeypatch.chdir(tmp_path)

    def recorded(*given):
        return RC.Plan(_options(wl, *given)).artifacts_override()

    def files(log, directory):
        return {"log": str(log), "snapshots": str(directory / "snapshots.json"), "tokens": str(directory / "tokens.json"),
                "versions": str(directory / "versions.json"), "stats": str(directory / "stats.json"), "cos_sin": str(directory / "cos_sin_cache.npy")}

    assert recorded() == {}
    assert recorded("--log", str(log)) == recorded("--log", "old/log.jsonl.gz") == files(log, log.parent)          # beside the log
    assert recorded("--log", str(log), "--snapshots", "snaps") == files(log, snaps)
    assert recorded("--snapshots", str(snaps)) == files(LOG, snaps)                                              # the run's own log
    assert recorded("--control-tokens", "ctl.json") == {"control_tokens": str(ctl)}
    assert recorded("--log", str(log), "--control-tokens", str(ctl)) == {**files(log, log.parent), "control_tokens": str(ctl)}


def test_the_weights_files_are_a_pattern_as_it_is_given_a_directory_s_model_files_or_the_one_file(tmp_path):
    wl = _workload(tmp_path)
    (tmp_path / "snap.v1").mkdir()

    def weights(*given):
        return RC.Plan(_options(wl, *given)).safetensors_glob()

    assert weights() is None and weights("--checkpoint-dir", "") is None
    for pattern in ("/ck/*/model-0000?.safetensors", "/ck/snap/model-[0-9].safetensors", "/ck/snap?", "/ck/*"):
        assert weights("--checkpoint-dir", pattern) == pattern
    assert weights("--checkpoint-dir", "/ck/snap") == "/ck/snap/model*.safetensors"               # no suffix: a directory, there or not
    assert weights("--checkpoint-dir", str(tmp_path / "snap.v1")) == str(tmp_path / "snap.v1" / "model*.safetensors")
    assert weights("--checkpoint-dir", "/ck/snap/model.safetensors") == "/ck/snap/model.safetensors"


@pytest.mark.parametrize("steps", ["0,2", "1,2", "1", "", ",", "2,1,0", "0,0"])
def test_snapshot_steps_that_do_not_count_up_from_0_are_refused(tmp_path, steps):
    refusal = f"--snapshot-steps must be contiguous from 0 (got {steps}): replay of step N needs every earlier KV producer"
    with pytest.raises(SystemExit, match=f"^{re.escape(refusal)}$"):
        RC.Plan(_options(_workload(tmp_path), "--snapshot-steps", steps))


@pytest.mark.parametrize("given, steps", [("0", [0]), ("0,1", [0, 1]), (" 0, 1 ,2", [0, 1, 2]), ("0,,1,", [0, 1]), (" ALL ", [0, 1, 2, 3])])
def test_snapshot_steps_are_a_comma_list_from_0_or_every_step_of_the_workload(tmp_path, given, steps):
    assert RC.Plan(_options(_workload(tmp_path), "--snapshot-steps", given)).steps == steps


@pytest.mark.parametrize("given, processes", [
    ((), 4), (("--parallel", "8"), 8), (("--parallel", "2"), 2), (("--parallel", "1"), 1), (("--parallel", "0"), 1), (("--parallel", "-2"), 1),
    (("--laptop",), 1), (("--laptop", "--parallel", "8"), 1)])
def test_a_run_replays_in_four_processes_or_as_many_as_asked_and_in_one_on_a_laptop(tmp_path, given, processes):
    assert RC.Plan(_options(_workload(tmp_path), *given)).parallel == processes


def test_a_dry_run_prints_each_stage_s_line_in_the_phase_s_order_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(RC, "profile_config", lambda name: None)
    run, wl = tmp_path / "run", _workload(tmp_path)
    given = ["--dry-run", "--skip", "nonint,golden,wiring", "--other-log", "/x/other.jsonl.gz", "--cards-dir", "cards", "--engine-arg", "name=x"]
    a = _options(wl, *given, out=str(run))
    assert RC.main(a) == 0
    plan = RC.Plan(a)

    def line(stage):
        return f"[{stage}] {shlex.join(plan.cmd(stage))}"

    cpu = [line("resolve"), line("census_reconcile"), "[nonint_compare] SKIP (--skip)", line("compare"), line("fold"), line("fold_2"),
           line("fold_other"), "[golden] SKIP (--skip)", "[wiring] SKIP (--skip)",
           f"[gates] in-process: gates.evaluate_gates({run}) -> {run}/gates.json; card -> {run}/card.json, {run}/CARD.md, cards/cid.json"]
    assert capsys.readouterr().out.splitlines() == [
        f"# run_config --dry-run  RUN_DIR={run}  phase=all  snapshot_steps=[0, 1, 2, 3]  laptop=False",
        line("capture"), line("control"), line("census"), "[nonint] SKIP (--skip)", *cpu,
        f"[gpu_done] write {run}/gpu_done.json with per-stage timings"]
    # the capture's line carries the limit of one tensor's snapshot that the driver derived (with no checkpoint to read, the least
    # one, 32 MiB), and a word a shell would split or unquote is quoted
    assert plan.cmd("capture")[-2:] == ["--snapshot-max-bytes", str(32 << 20)]
    assert """ '--engine-arg=name="x"' """ in line("capture")
    assert RC.main(_options(wl, *given, "--phase", "cpu", "--laptop", out=str(run))) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"# run_config --dry-run  RUN_DIR={run}  phase=cpu  snapshot_steps=[0, 1, 2, 3]  laptop=True", *cpu]
    assert RC.main(_options(wl, *given, "--phase", "gpu", out=str(run))) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"# run_config --dry-run  RUN_DIR={run}  phase=gpu  snapshot_steps=[0, 1, 2, 3]  laptop=False",
        line("capture"), line("control"), line("census"), "[nonint] SKIP (--skip)", f"[gpu_done] write {run}/gpu_done.json with per-stage timings"]
    assert not run.exists()


def test_a_gpu_phase_takes_its_lease_first_runs_every_stage_though_one_fails_and_writes_their_statuses(tmp_path, monkeypatch):
    monkeypatch.setattr(RC, "profile_config", lambda name: None)
    events: list[str] = []

    def cmd(self, stage):
        events.append(stage)
        return [sys.executable, "-c", "raise SystemExit(3)" if stage in failing else "pass"]

    monkeypatch.setattr(RC.Plan, "cmd", cmd)
    monkeypatch.setattr(RC.GL, "acquire", lambda: events.append("lease"))
    for name, failing, code in (("one-fails", {"census"}, 1), ("all-pass", set(), 0)):
        run = tmp_path / name
        events.clear()
        assert RC.main(_options(_workload(tmp_path), "--phase", "gpu", out=str(run))) == code
        assert events == ["lease", *RC.GPU_STAGES]
        stages = json.loads((run / "stages.json").read_text())["stages"]
        assert [(s["stage"], s["status"], s["returncode"]) for s in stages] == [
            (st, "failed" if st in failing else "ok", 3 if st in failing else 0) for st in RC.GPU_STAGES]
        done = json.loads((run / "gpu_done.json").read_text())
        assert sorted(done) == ["config_id", "seconds", "stages", "utc"] and done["config_id"] == "cid"
        assert (run / "gpu_done.json").read_text() == json.dumps(done, indent=1) + "\n" and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", done["utc"])
        assert done["stages"] == {s["stage"]: {"status": s["status"], "seconds": s["seconds"]} for s in stages}
        assert sorted(p.name for p in (run / "logs").iterdir()) == sorted(f"{st}.log" for st in RC.GPU_STAGES)


def _stages_stood_in(monkeypatch, events):
    """The lease, each stage's run and the gates stage stood in: `events` takes the lease's word and then each stage's run whole."""
    def run_stage(stage, cmd, log, *, max_rss_gb, timeout_s):
        events.append((stage, cmd, log, max_rss_gb, timeout_s))
        return {"stage": stage, "status": "ok", "returncode": 0, "seconds": 0.25, "log": str(log), "cmd": shlex.join(cmd), "reason": None}

    def gates_stage(plan, a):
        events.append("gates")
        return {"stage": "gates", "status": "ok", "returncode": 0, "seconds": 0.5, "log": None, "cmd": "in-process", "reason": None}

    monkeypatch.setattr(RC, "profile_config", lambda name: None)
    monkeypatch.setattr(RC.GL, "acquire", lambda: events.append("lease"))
    monkeypatch.setattr(RC, "run_stage", run_stage)
    monkeypatch.setattr(RC, "gates_stage", gates_stage)


@pytest.mark.parametrize("given, limits", [
    ((), (None, None)), (("--laptop",), (RC.LAPTOP_MAX_RSS_GB, RC.LAPTOP_STAGE_TIMEOUT_S)), (("--max-rss-gb", "3", "--stage-timeout", "9"), (3.0, 9.0)),
    (("--laptop", "--max-rss-gb", "0", "--stage-timeout", "0"), (0.0, 0.0))])
def test_each_stage_runs_its_command_under_the_limits_given_a_laptop_s_when_none_is_and_none_otherwise(tmp_path, monkeypatch, given, limits):
    events: list = []
    _stages_stood_in(monkeypatch, events)
    run = tmp_path / "run"
    a = _options(_workload(tmp_path), "--phase", "gpu", *given, out=str(run))
    assert RC.main(a) == 0
    plan = RC.Plan(a)
    assert events == ["lease", *((st, plan.cmd(st), run / "logs" / f"{st}.log", *limits) for st in RC.GPU_STAGES)]
    assert (RC.LAPTOP_MAX_RSS_GB, RC.LAPTOP_STAGE_TIMEOUT_S) == (1.5, 180.0)


def test_a_whole_run_takes_the_lease_times_the_gpu_s_stages_alone_and_records_a_skipped_stage_whole(tmp_path, monkeypatch):
    events: list = []
    _stages_stood_in(monkeypatch, events)
    run = tmp_path / "run"
    assert RC.main(_options(_workload(tmp_path), "--skip", "census,golden,replay_a", out=str(run))) == 0
    ran = ["capture", "control", "nonint", "resolve", "nonint_compare", "compare", "fold", "fold_2", "gates"]      # `census` names two stages
    assert [e if isinstance(e, str) else e[0] for e in events] == ["lease", *ran]
    stages = json.loads((run / "stages.json").read_text())["stages"]
    assert [(s["stage"], s["status"]) for s in stages] == [
        ("capture", "ok"), ("control", "ok"), ("census", "skipped"), ("nonint", "ok"), ("resolve", "ok"), ("census_reconcile", "skipped"),
        ("nonint_compare", "ok"), ("compare", "ok"), ("fold", "ok"), ("fold_2", "ok"), ("golden", "skipped"), ("replay_a", "skipped"), ("gates", "ok")]
    assert [s for s in stages if s["status"] == "skipped"] == [
        {"stage": st, "status": "skipped", "returncode": None, "seconds": 0.0, "log": None, "cmd": None, "reason": "--skip"}
        for st in ("census", "census_reconcile", "golden", "replay_a")]
    done = json.loads((run / "gpu_done.json").read_text())
    assert done["stages"] == {"capture": {"status": "ok", "seconds": 0.25}, "control": {"status": "ok", "seconds": 0.25},
                              "census": {"status": "skipped", "seconds": 0.0}, "nonint": {"status": "ok", "seconds": 0.25}}


def test_a_stage_name_without_a_command_leaves_no_record_and_a_gates_stage_that_raises_is_a_failed_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(RC, "profile_config", lambda name: None)
    leases: list[int] = []
    monkeypatch.setattr(RC.GL, "acquire", lambda: leases.append(1))

    def no_card(plan, a):
        raise ValueError("no card")

    monkeypatch.setattr(RC, "gates_stage", no_card)
    run, skipped = tmp_path / "run", [st for st in CPU_STAGES if st not in NO_COMMAND]
    assert RC.main(_options(_workload(tmp_path), "--phase", "cpu", "--skip", ",".join(skipped), out=str(run))) == 1
    stages = json.loads((run / "stages.json").read_text())["stages"]
    assert [(s["stage"], s["status"]) for s in stages] == [*((st, "skipped") for st in skipped), ("gates", "failed")]
    assert stages[-1] == {"stage": "gates", "status": "failed", "returncode": None, "seconds": 0.0, "log": None, "cmd": "in-process",
                          "reason": "ValueError: no card"}
    assert leases == [] and not (run / "gpu_done.json").exists()                # a CPU phase takes no GPU and times none


def test_the_gates_stage_writes_the_gates_the_card_twice_and_its_summary_and_prints_the_card_s_head_and_the_gates(tmp_path, capsys):
    run, cards = tmp_path / "run", tmp_path / "cards"
    run.mkdir()
    a = _options(_workload(tmp_path), "--cards-dir", str(cards), out=str(run))
    rec = RC.gates_stage(RC.Plan(a), a)
    seconds = rec.pop("seconds")
    assert rec == {"stage": "gates", "status": "ok", "returncode": 0, "log": None, "cmd": "in-process", "reason": None,
                   "verdict": "INCOMPLETE", "card": str(cards / "cid.json")}       # a run directory with no stage's output in it
    assert isinstance(seconds, float) and seconds >= 0
    assert sorted(p.name for p in run.iterdir()) == ["CARD.md", "card.json", "gates.json"] and [p.name for p in cards.iterdir()] == ["cid.json"]
    card = CARD.load_card(run / "card.json")
    assert (card.config_id, card.verdict) == ("cid", "INCOMPLETE") and (cards / "cid.json").read_bytes() == (run / "card.json").read_bytes()
    assert (run / "CARD.md").read_text() == CARD.summarize_md(card)
    results = RC.G.evaluate_gates(run)
    written = json.loads((run / "gates.json").read_text())                       # but for the second it was written in
    assert {**written, "utc": None} == {**RC.G.gates_json(results, run, RC.G.diagnostics(run)), "utc": None}
    assert capsys.readouterr().out == f"{CARD.top_section(card)}\n{RC.G.gates_markdown(results)}\n"


def test_the_driver_hands_a_fold_worker_s_options_to_the_worker_before_it_asks_for_its_own():
    with pytest.raises(SystemExit, match="^--worker fold needs --log$"):
        RC.main(cli.parse("match", ["--worker", "fold"], {}))


def test_the_driver_binds_no_name_of_another_module_that_its_own_text_does_not_read():
    """What the driver gives other code is its `__all__`.  Every other name it imports from a module is one its own text reads, so
    a reader of such a name takes it from the module that defines it and not through the driver."""
    tree = ast.parse(Path(RC.__file__).read_text())
    bound = {a.asname or a.name for n in tree.body if isinstance(n, ast.ImportFrom) and n.module != "__future__" for a in n.names}
    read = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert RC.__all__ == ["Options", "fold_construction_args", "main"]
    assert sorted(bound - read - set(RC.__all__)) == []
