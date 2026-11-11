"""A stopped job continues from its kept state: what it keeps, what it refuses, and that its record over two segments
is the unbroken job's."""
import io
import json
import os
import signal

import pytest
import torch
from synthetic import tiny_registry, write_config

from training_slowdown import VERSION, config, run, starts
from training_slowdown.data import Data, Stream

#: what depends on when a job ran, not on what it computed: its times, and the flags read from them
TIMES = {"cpu_s", "wall_s", "gpu_s", "process_cpu_s", "process_wall_s", "children_cpu_s", "self_cpu_s", "elapsed_s",
         "flags"}
BUDGET = {"steps": 6, "batch": 2}
KEY = "00112233445566778899aabbccddeeff"
META = {"config": {"digest": "c" * 64}}


def timeless(x, keyed: bool = True):
    """`x` without its times and, unless `keyed`, with each keyed weight digest as a count: two jobs digest under
    two keys."""
    if isinstance(x, dict):
        return {k: (len(v) if k == "weights" and not keyed else timeless(v, keyed)) for k, v in x.items()
                if k not in TIMES}
    return [timeless(v, keyed) for v in x] if isinstance(x, list) else x


def assert_continues(whole: dict, cut: dict, at: int, keyed: bool = True) -> None:
    """`cut`, a job stopped and continued from step `at`, records what `whole`, the unbroken one, does: every step's
    count, every checkpoint's digests and score, and everything else but its times, its segments and the route's
    build, which the second segment ran and counted again."""
    again = cut["setup"]["macs"] - whole["setup"]["macs"]
    assert [(s["first"], s["steps"]) for s in cut["segments"]] == [(0, at), (at, len(whole["steps"]) - at)]
    assert [(s["first"], s["steps"]) for s in whole["segments"]] == [(0, len(whole["steps"]))]
    assert sum(s["macs"] for s in cut["segments"]) == cut["totals"]["macs"] == whole["totals"]["macs"] + again
    assert again == whole["setup"]["macs"] > 0
    a, b = timeless(whole, keyed), timeless(cut, keyed)
    assert b["steps"] == a["steps"] and [s["step"] for s in b["steps"]] == list(range(len(b["steps"])))
    assert [cp | {"macs": cp["macs"] - again * (cp["step"] >= at)} for cp in b["checkpoints"]] == a["checkpoints"]
    assert b["structures"].items() >= a["structures"].items()

    def rest(rec):
        return {k: v for k, v in rec.items() if k not in ("steps", "checkpoints", "segments", "setup", "structures",
                                                          "totals", *(() if keyed else ("digest_key",)))}
    assert rest(b) == rest(a)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("resume")
    reg = tiny_registry(root / "reg")
    return root, reg, write_config(root / "a.toml", budget=BUDGET)


def segment(tiny, kept=None, state=None, stop_after=None):
    """One segment of the tiny config's job in this process, as `run.inside` runs it but for the sandbox: its record
    when it ends, or what it kept last (its tensors as they are read back, its record so far) when it is stopped
    after the checkpoint of step `stop_after`."""
    root, reg, path = tiny
    registry = starts.registry(reg)
    cfg = config.load(path, registry)
    route = config.route(cfg.route)
    theta = starts.load(cfg.start, reg.parent / "starts")
    last = {}

    def keep(trained):
        raw = io.BytesIO()
        torch.save(run.resume_state(trained, optimizers, stream), raw)
        last.update(state=torch.load(io.BytesIO(raw.getvalue()), weights_only=True), record=te.so_far(META))
        if last["record"]["checkpoints"][-1]["step"] == stop_after:
            raise run.Stopped

    torch.manual_seed(0)
    te = run.Harness(theta, root / "out", start=cfg.start.entry(), clock=lambda: 0.0, keep=keep,
                     kept={"digest_key": KEY} | (kept or {}))
    stream = Stream(Data.of(cfg.start.data, registry), 0, ctx=int(cfg.start.shape["ctx"]))
    try:
        with te.counter:
            model, optimizers = route.build(cfg.start.shape, cfg.dials)
            run.train(te, cfg, route, stream, model, optimizers, state)
    except run.Stopped:
        return last
    return te.finish(META)


def test_a_segment_from_a_kept_state_runs_the_steps_that_remain_as_the_unbroken_job_does(tiny):
    """The harness alone, in one process: a job stopped after the checkpoint of step 2 keeps the tensors and the
    record so far; a second harness, told the next step and the digest key, loads θ, takes the tensors and runs steps
    3 to 5; the two records joined are the unbroken job's, with every keyed weight digest. A kept state without its
    optimizers' state gives other weights."""
    whole = run.joined(None, segment(tiny))
    assert [cp["step"] for cp in whole["checkpoints"]] == [0, 1, 2, 3, 4, 5]
    first = segment(tiny, stop_after=2)
    assert len(first["record"]["steps"]) == 3 and first["state"]["stream"] == 3 * BUDGET["batch"]
    cut = run.joined(run.joined(None, first["record"]), segment(tiny, {"step": 3}, first["state"]))
    assert_continues(whole, cut, at=3)

    digests = [cp["output_sha256"] for cp in whole["checkpoints"]]
    fresh = first["state"] | {"optimizers": [{"state": {}, "param_groups": o["param_groups"]}
                                             for o in first["state"]["optimizers"]]}
    assert [cp["output_sha256"] for cp in segment(tiny, {"step": 3}, fresh)["checkpoints"]] != digests[3:]


def test_the_generators_a_step_may_read_come_back_as_they_were_kept():
    import random

    import numpy as np

    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    raw = io.BytesIO()
    torch.save(run.plain(run.generators()), raw)
    drawn = (torch.rand(3).tolist(), np.random.rand(3).tolist(), random.random())
    run.set_generators(torch.load(io.BytesIO(raw.getvalue()), weights_only=True))
    assert (torch.rand(3).tolist(), np.random.rand(3).tolist(), random.random()) == drawn


def test_a_segments_clock_starts_where_the_kept_states_stood(tmp_path):
    """`budget.minutes` runs over every segment: a job that used its minutes before it stopped is past them when it
    continues."""
    te = run.Harness({}, tmp_path, clock=lambda: 10.0, budget_s=40.0, kept={"elapsed_s": 50.0})
    assert te.elapsed() == 50.0 and te.time_left() == -10.0
    with pytest.raises(run.BudgetExceeded):
        te._after_step(0)
    assert run.Harness({}, tmp_path, clock=lambda: 10.0, budget_s=40.0).elapsed() == 0.0


def kept_state(tmp_path, tiny, **over):
    """A state directory holding what the tiny job kept after step 2, with `over` in its record."""
    first = segment(tiny, stop_after=2)
    record = run.joined(None, first["record"] | {"seed": 0, "split": "final"}) | over
    tensors = tmp_path / "handed.pt"
    torch.save(first["state"], tensors)
    state = run.Kept(tmp_path / "state")
    state.write(tensors, record, 1.5)
    return state, record


def test_a_job_keeps_one_state_and_continues_only_its_own(tmp_path, tiny):
    """The state directory holds the last state alone, the tensors moved in and named by sha256. A state of another
    VERSION, seed or split, with tensors that aren't the ones it names or a model that isn't its record's last
    checkpoint's, is refused by what differs, and left as it was."""
    state, record = kept_state(tmp_path, tiny)
    assert sorted(p.name for p in state.dir.iterdir()) == ["state.json", "tensors-0000002.pt"]
    assert not (tmp_path / "handed.pt").exists()
    later = tmp_path / "later.pt"
    later.write_bytes((state.dir / "tensors-0000002.pt").read_bytes())
    state.write(later, record | {"checkpoints": record["checkpoints"] + [record["checkpoints"][-1] | {"step": 3}]}, 2.0)
    assert sorted(p.name for p in state.dir.iterdir()) == ["state.json", "tensors-0000003.pt"]
    state.write(state.dir / "tensors-0000003.pt", record, 1.5)
    kept = state.load(0, "final")
    assert kept["record"] == json.loads(json.dumps(record)) and kept["elapsed_s"] == 1.5
    assert (kept["tensors"], kept["sha256"]) == ("tensors-0000002.pt", run.file_sha256(state.dir / kept["tensors"]))

    for seed, split, why in ((1, "final", "its seed is 0 and this job's is 1"),
                             (0, "search", "its split is 'final' and this job's is 'search'")):
        with pytest.raises(SystemExit, match=f"isn't one this job continues: {why}. Remove {state.dir} to start"):
            state.load(seed, split)
    manifest = state.dir / "state.json"
    whole = manifest.read_text()

    def refused(text: str, why: str) -> None:
        manifest.write_text(text)
        with pytest.raises(SystemExit, match=why):
            state.load(0, "final")
        manifest.write_text(whole)

    other = json.loads(whole)
    refused(json.dumps(other | {"record": other["record"] | {"training_slowdown": "0.0.0"}}),
            f"its training_slowdown VERSION is '0.0.0' and this job's is '{VERSION}'")
    refused(json.dumps(other | {"schema": "training-slowdown/state/v0"}), "its schema is 'training-slowdown/state/v0'")
    refused(whole[:-9], "its state.json doesn't parse")
    refused(json.dumps({k: v for k, v in other.items() if k != "elapsed_s"}),
            "its state.json lacks what a state holds .KeyError: 'elapsed_s'.")
    refused(json.dumps(other | {"sha256": "0" * 64}), f"its tensors, tensors-0000002.pt, have sha256 {other['sha256']}")
    earlier = other["record"] | {"checkpoints": other["record"]["checkpoints"][:2]}
    refused(json.dumps(other | {"record": earlier}),
            "its model hashes to .* its record's last checkpoint, of step 1, to")
    assert state.load(0, "final") == kept

    (state.dir / "tensors-0000002.pt").unlink()
    with pytest.raises(SystemExit, match="its tensors, tensors-0000002.pt, are missing"):
        state.load(0, "final")
    state.clear()
    assert not state.dir.exists() and run.Kept(state.dir).load(0, "final") is None


def test_a_segment_that_names_another_job_than_the_kept_record_is_refused(tiny):
    stopped = segment(tiny, stop_after=2)
    first, rest = run.joined(None, stopped["record"]), segment(tiny, {"step": 3}, stopped["state"])
    for over, why in (({"config": {"digest": "b" * 64}}, "config digest"), ({"seed": 5}, "seed"),
                      ({"steps": rest["steps"][1:]}, "first step")):
        with pytest.raises(SystemExit, match=f"the kept record's {why} is .* no record is written"):
            run.joined(first, rest | over)


def test_a_route_that_names_a_signal_is_refused():
    assert run.screen("import torch\n") == []
    assert run.screen("import os\nos.kill(os.getppid(), 15)\n") and run.screen("import signal\n")


@pytest.mark.slow
def test_a_job_stopped_after_a_checkpoint_and_run_again_writes_the_unbroken_jobs_record(tmp_path, tiny, monkeypatch):
    """The job as a node runs it, four times, each two processes, so slow: what the quick tests above don't reach is
    the job's own process keeping the state, the stop, and the sandbox. An unbroken job; the same job told to stop
    (SIGTERM) once it has kept its state after step 2, which ends with `STOPPED`, no record and the state left; a
    job of another config on that state, refused by both digests before it trains, the state left; and the
    stopped job again, which continues while its state directory, hidden from the route's process, still holds the
    state, writes the unbroken job's record in two segments and removes the state."""
    root, reg, path = tiny

    def job(out, state, cfg=path):
        return run.main(["--config", cfg, "--seed", 0, "--split", "final", "--out", tmp_path / out, "--starts", reg,
                         "--device", "cpu", "--state", tmp_path / state])

    assert job("whole", "state-whole") == 0 and not (tmp_path / "state-whole").exists()
    whole = json.loads((tmp_path / "whole" / "run.json").read_text())
    assert whole["invalid"] == [] and len(whole["segments"]) == 1

    write = run.Kept.write

    def stop_after(self, tensors, record, elapsed_s):
        write(self, tensors, record, elapsed_s)
        if record["checkpoints"][-1]["step"] == 2:
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(run.Kept, "write", stop_after)
    assert job("cut", "state") == run.STOPPED and not (tmp_path / "cut" / "run.json").exists()
    monkeypatch.setattr(run.Kept, "write", write)
    kept = sorted(p.name for p in (tmp_path / "state").iterdir())
    assert kept == ["state.json", "tensors-0000002.pt"]
    so_far = json.loads((tmp_path / "state" / "state.json").read_text())["record"]
    assert [cp["score"]["split"] for cp in so_far["checkpoints"]] == ["final"] * 3 and len(so_far["steps"]) == 3

    other = write_config(tmp_path / "other.toml", budget=BUDGET | {"steps": 7})
    with pytest.raises(SystemExit, match=f"its config digest is {so_far['config']['digest']} and this job's is "
                                         f"{config.load(other, starts.registry(reg)).digest}"):
        job("other", "state", other)
    assert sorted(p.name for p in (tmp_path / "state").iterdir()) == kept and not (tmp_path / "other").exists()

    assert job("cut", "state") == 0 and not (tmp_path / "state").exists()
    cut = json.loads((tmp_path / "cut" / "run.json").read_text())
    assert cut["invalid"] == [] and cut["digest_key"] == so_far["digest_key"] != whole["digest_key"]
    assert_continues(whole, cut, at=3, keyed=False)
    assert [cp["score"] for cp in cut["checkpoints"]] == [cp["score"] for cp in whole["checkpoints"]]
    assert sorted(p.name for p in (tmp_path / "cut").iterdir()) == ["e", "run.json"]
