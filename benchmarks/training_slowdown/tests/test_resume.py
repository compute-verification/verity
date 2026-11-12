"""A stopped job continues from its kept state: what it keeps, what it refuses, and that its record over two segments
is the unbroken job's."""
import argparse
import contextlib
import copy
import hashlib
import io
import json
import os
import signal

import pytest
import torch
from synthetic import SCORE_REL, tiny_registry, write_config

from training_slowdown import NOTHING, TOLD, VERSION, config, grade, run, starts, stop_record, told
from training_slowdown.data import Data

#: what depends on when a job ran, not on what it computed: its times, and the flags read from them
TIMES = {"cpu_s", "wall_s", "gpu_s", "process_cpu_s", "process_wall_s", "children_cpu_s", "self_cpu_s", "elapsed_s",
         "flags"}
BUDGET = {"steps": 4, "batch": 2}
KEY = "00112233445566778899aabbccddeeff"
META = {"config": {"digest": "c" * 64}}


def timeless(x, keyed: bool = True):
    """`x` without its times and, unless `keyed`, with each keyed weight digest as a count: two jobs digest under
    two keys."""
    if isinstance(x, dict):
        return {k: (len(v) if k == "weights" and not keyed else timeless(v, keyed)) for k, v in x.items()
                if k not in TIMES}
    return [timeless(v, keyed) for v in x] if isinstance(x, list) else x


def scored(cp: dict) -> dict:
    """A checkpoint to compare another job's with: its scores, the numbers the evaluator's forward passes gave (its
    bpb and each float of its `score`), to within what two passes of one model may differ by (`SCORE_REL`), and every
    other field, the digests of its state among them, as it is."""
    def near(x):
        return pytest.approx(x, rel=SCORE_REL) if isinstance(x, float) else x
    return cp | {"bpb": near(cp["bpb"]), "score": cp["score"] and {k: near(v) for k, v in cp["score"].items()}}


def assert_continues(whole: dict, cut: dict, at: int, keyed: bool = True) -> None:
    """`cut`, a job stopped and continued from step `at`, records what `whole`, the unbroken one, does: every step's
    count, every checkpoint's digests, its score to within `SCORE_REL` (`scored`), and everything else but its
    times, its segments and what it counted outside its steps: the route's build, which the second segment ran and
    counted again, and under a route's `output` that computes (`lora`'s) the output at the checkpoint the job
    stopped at. The unbroken job counts that in step `at`; the stopped one counted it after its last kept step, so
    `cut` holds it outside the steps and its step `at` is short of exactly it. It is what each checkpoint's output
    counts, the last one's among them: the unbroken job's tail."""
    build, output = whole["setup"]["macs"], whole["tail"]
    assert [(s["first"], s["steps"]) for s in cut["segments"]] == [(0, at), (at, len(whole["steps"]) - at)]
    assert [(s["first"], s["steps"]) for s in whole["segments"]] == [(0, len(whole["steps"]))]
    assert sum(s["macs"] for s in cut["segments"]) == cut["totals"]["macs"] == whole["totals"]["macs"] + build
    assert cut["setup"]["macs"] == 2 * build + output["macs"] and build > 0
    a, b = timeless(whole, keyed), timeless(cut, keyed)
    if output["macs"]:
        short, full = b["steps"][at], a["steps"][at]
        counts, by = ("macs", "matmul_macs", "elementwise_macs", "approx_macs"), ("by", "device_macs")
        assert {k: short[k] + output[k] for k in counts} == {k: full[k] for k in counts}
        assert {k: run._added(short[k], output[k]) for k in by} == {k: full[k] for k in by}
        assert short["struct"] != full["struct"]
        b["steps"][at] = short | {k: full[k] for k in (*counts, *by, "struct")}
    assert b["steps"] == a["steps"] and [s["step"] for s in b["steps"]] == list(range(len(b["steps"])))
    assert [cp | {"macs": cp["macs"] - build * (cp["step"] >= at)} for cp in b["checkpoints"]] == [
        scored(cp) for cp in a["checkpoints"]]
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
    stream = run.stream_of(cfg, Data.of(cfg.start.data, registry), 0, reg.parent / "starts")
    te = run.Harness(theta, root / "out", start=cfg.start.entry(), clock=lambda: 0.0, keep=keep,
                     kept={"digest_key": KEY} | (kept or {}), stream=stream)
    with contextlib.nullcontext() if stop_after is None else pytest.raises(run.Stopped), te.counter:
        model, optimizers = route.build(cfg.start.shape, cfg.dials)
        run.train(te, cfg, route, stream, model, optimizers, state)
    return te.finish(META) if stop_after is None else last


@pytest.fixture(scope="module")
def stopped(tiny):
    """What the tiny job kept when it was stopped after the checkpoint of step 1."""
    return segment(tiny, stop_after=1)


@pytest.fixture(scope="module")
def rest(tiny, stopped):
    """The record of the segment that continues `stopped`: steps 2 and 3."""
    return segment(tiny, {"step": 2}, stopped["state"])


def test_a_segment_from_a_kept_state_runs_the_steps_that_remain_as_the_unbroken_job_does(tiny, stopped, rest):
    """The harness alone, in one process: a job stopped after the checkpoint of step 1 keeps the tensors and the
    record so far; a second harness, told the next step and the digest key, loads θ, takes the tensors and runs steps
    2 and 3; the two records joined are the unbroken job's, with every keyed weight digest."""
    whole = run.joined(None, segment(tiny))
    assert [cp["step"] for cp in whole["checkpoints"]] == [0, 1, 2, 3]
    assert len(stopped["record"]["steps"]) == 2 and stopped["state"]["stream"] == 2 * BUDGET["batch"]
    assert_continues(whole, run.joined(run.joined(None, stopped["record"]), rest), at=2)


def test_two_segments_op_logs_are_added_whether_their_rows_name_a_dtype_or_none(stopped, rest):
    """What two segments counted outside their steps is added row by row (`run.outside_steps`), a row being one op at
    one shape in one dtype. The counter's rows name each op's dtype, a seventh field; a record written before rows
    did has six. Both are read: the two segments joined without their dtypes are the two joined with them, the
    dtypes then left out. An op at one shape in two dtypes stays two rows."""
    named = run.joined(run.joined(None, stopped["record"]), rest)
    bare = run.joined(run.joined(None, grade.without_dtypes(stopped["record"])), grade.without_dtypes(rest))
    seven = named["structures"][named["setup"]["struct"]]["log"]
    six = bare["structures"][bare["setup"]["struct"]]["log"]
    assert seven and {len(row) for row in seven} == {7} and {len(row) for row in six} == {6}
    assert six == grade.without_dtypes(named)["structures"][named["setup"]["struct"]]["log"]
    counts = ("tokens", "macs", "matmul_macs", "elementwise_macs", "approx_macs", "by", "device_macs")
    assert {k: bare["setup"][k] for k in counts} == {k: named["setup"][k] for k in counts}
    assert sum(row[5] for row in six) == sum(row[5] for row in seven) == named["setup"]["macs"] > 0

    first = seven[0]
    structures = named["structures"] | {"other": {"updated": [], "log": [[*first[:6], "float64"]]}}
    both = run.outside_steps(structures, [named["setup"], named["setup"] | {"struct": "other"}])
    assert [row for row in structures[both["struct"]]["log"] if row[:4] == first[:4]] == sorted(
        [first, [*first[:6], "float64"]])


def test_a_kept_state_without_its_optimizers_state_continues_to_other_weights(tiny, stopped, rest):
    fresh = stopped["state"] | {"optimizers": [{"state": {}, "param_groups": o["param_groups"]}
                                               for o in stopped["state"]["optimizers"]]}
    again = segment(tiny, {"step": 2}, fresh)
    assert [cp["step"] for cp in again["checkpoints"]] == [cp["step"] for cp in rest["checkpoints"]] == [2, 3]
    assert all(a["output_sha256"] != b["output_sha256"] for a, b in zip(again["checkpoints"], rest["checkpoints"]))


def test_a_kept_state_with_other_buffers_than_the_route_built_is_refused(tiny, stopped):
    """The model's buffers outside its state dict come back by name: a state that holds others than `build`
    registered is another model's."""
    assert sorted(stopped["state"]["buffers"]) == ["cos", "sin"]
    for buffers, held in (({}, r"\[\]"), (stopped["state"]["buffers"] | {"average_0": torch.zeros(2)},
                                           r"\['average_0', 'cos', 'sin'\]")):
        with pytest.raises(SystemExit, match=rf"the kept state holds the buffers {held} outside the model's state "
                                             rf"dict and the route built \['cos', 'sin'\]"):
            segment(tiny, {"step": 2}, stopped["state"] | {"buffers": buffers})


#: the routes with an `output`, each with dials for the tiny start
OUTPUTS = {"ema": {"trainable": "blocks.0.w_*", "lr": 1e-2, "ema_decay": 0.9},
           "lora": {"targets": "blocks.0.w_*.weight", "rank": 2, "lr": 1e-2},
           "qat": {"targets": "blocks.0.w*.weight", "keep": 0.5, "lr": 1e-2}}


def bare(state: dict) -> dict:
    """A kept state without what a route with an `output` carries beside the weights θ holds: `ema`'s averages,
    `lora`'s B, and `qat`'s latents and its codes (`scales`, `plus`, `minus`), each zeros, as `build` leaves it."""
    def zeros(tensors: dict, named) -> dict:
        return {k: torch.zeros_like(v) if named(k) else v for k, v in tensors.items()}
    return state | {"buffers": zeros(state["buffers"], lambda k: k.startswith("average_")
                                     or k.endswith((".scales", ".plus", ".minus"))),
                    "model": zeros(state["model"], lambda k: k.endswith((".b", ".v")))}


@pytest.mark.parametrize("route", sorted(OUTPUTS))
def test_a_job_of_a_route_with_an_output_keeps_the_model_it_trains_and_continues_as_the_unbroken_job(
        tmp_path, route):
    """The harness alone, in one process. Under a route's `output` a checkpoint scores the output model (`ema` an
    average of the weights, `lora` its adapters merged into theirs), which is not what the route's next step reads.
    What the job keeps is the model the route trains, whose digest each checkpoint states beside its output's, and
    the model's buffers outside its state dict, where `ema` carries its averages. The segment that continues from
    it, served fresh windows from the kept position, ends at the unbroken job's outputs bit for bit, and the two
    records joined are the unbroken job's (`assert_continues`); without what the route carries beside θ's weights
    (`bare`) it ends at other outputs."""
    reg = tiny_registry(tmp_path / "reg", read=(1,))
    tiny = tmp_path, reg, write_config(tmp_path / "a.toml", route=route, budget=BUDGET, windows="fresh",
                                       dials=OUTPUTS[route])
    whole = run.joined(None, segment(tiny))
    assert whole["invalid"] == [] and len({cp["output_sha256"] for cp in whole["checkpoints"]}) == 4
    cut = segment(tiny, stop_after=1)
    kept, last = cut["state"], cut["record"]["checkpoints"][-1]
    assert last["step"] == 1 and run.theta_sha256(kept["model"]) == last["trained_sha256"] != last["output_sha256"]
    assert [k for k in kept["buffers"] if k.startswith("average_")] == ["average_0", "average_1", "average_2"] * (
        route == "ema")
    rest = segment(tiny, {"step": 2}, kept)
    both = run.joined(run.joined(None, cut["record"]), rest)
    assert both["invalid"] == [] and both["windows"] == whole["windows"]
    assert_continues(whole, both, at=2)
    assert (whole["tail"]["macs"] > 0) == (route == "lora")
    other = segment(tiny, {"step": 2}, bare(kept))
    assert all(a["output_sha256"] != b["output_sha256"] for a, b in zip(other["checkpoints"], rest["checkpoints"]))


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


def kept_state(tmp_path, stopped):
    """A state directory holding what the tiny job kept after step 1."""
    record = run.joined(None, stopped["record"] | {"seed": 0, "split": "final"})
    tensors = tmp_path / "handed.pt"
    torch.save(stopped["state"], tensors)
    state = run.Kept(tmp_path / "state")
    state.write(tensors, record, 1.5)
    return state, record


def changed(state: dict, part: str) -> dict:
    """`state` with one bit of `part` flipped: of its first tensor's first byte, or of the stream's position."""
    other = copy.deepcopy(state)
    if part == "stream":
        other[part] ^= 1
    else:
        first = next(t for t in run._tensors(other[part]) if t.numel())
        first.reshape(-1).view(torch.uint8)[0] ^= 1
    return other


def test_each_part_of_a_kept_state_has_a_sha512_of_its_own(stopped):
    """One bit of a part changed moves that part's digest and no other's; a mapping's order and a tuple written as
    a list move none."""
    whole = run.kept_sha512(stopped["state"])
    assert sorted(whole) == ["buffers", "generators", "model", "optimizers", "stream"]
    assert all(len(d) == 128 and int(d, 16) >= 0 for d in whole.values()) and len(set(whole.values())) == 5
    for part in whole:
        got = run.kept_sha512(changed(stopped["state"], part))
        assert [k for k in whole if got[k] != whole[k]] == [part]
    again = {k: stopped["state"][k] for k in reversed(stopped["state"])}
    again["model"] = dict(reversed(again["model"].items()))
    again["generators"] = again["generators"] | {"python": tuple(again["generators"]["python"])}
    assert run.kept_sha512(again) == whole
    with pytest.raises(TypeError, match="a kept state holds plain values, and this is a set"):
        run.kept_sha512({"stream": {1}})


def test_a_job_keeps_one_state_and_continues_only_its_own(tmp_path, stopped):
    """The state directory holds the last state alone, the tensors moved in, the SHA-512 of each of their parts in
    the record's last segment. A state of another VERSION, seed or split, with a part of its tensors that doesn't
    hash to what its record states or a model that isn't its record's last checkpoint's, is refused by what
    differs, and left as it was."""
    state, record = kept_state(tmp_path, stopped)
    assert sorted(p.name for p in state.dir.iterdir()) == ["state.json", "tensors-0000001.pt"]
    assert not (tmp_path / "handed.pt").exists()
    later = tmp_path / "later.pt"
    later.write_bytes((state.dir / "tensors-0000001.pt").read_bytes())
    state.write(later, record | {"checkpoints": record["checkpoints"] + [record["checkpoints"][-1] | {"step": 2}]}, 2.0)
    assert sorted(p.name for p in state.dir.iterdir()) == ["state.json", "tensors-0000002.pt"]
    state.write(state.dir / "tensors-0000002.pt", record, 1.5)
    kept = state.load(0, "final")
    stated = run.kept_sha512(stopped["state"])
    assert "kept" not in record["segments"][-1] and kept["record"] == json.loads(json.dumps(
        record | {"segments": [record["segments"][-1] | {"kept": stated}]}))
    assert (kept["tensors"], kept["elapsed_s"]) == ("tensors-0000001.pt", 1.5) and "sha256" not in kept

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
    segment = other["record"]["segments"][-1]
    bare = {k: v for k, v in segment.items() if k != "kept"}
    refused(json.dumps(other | {"record": other["record"] | {"segments": [bare]}}),
            "its state.json lacks what a state holds .KeyError: 'kept'.")
    moved = segment | {"kept": stated | {"stream": "0" * 128}}
    refused(json.dumps(other | {"record": other["record"] | {"segments": [moved]}}),
            f"the SHA-512 of its stream, as tensors-0000001.pt holds it, is {stated['stream']} and the record of the "
            f"segment that kept it states {'0' * 128}")
    earlier = other["record"] | {"checkpoints": other["record"]["checkpoints"][:1]}
    refused(json.dumps(other | {"record": earlier}),
            "its model hashes to .* its record's last checkpoint, of step 0, to")
    assert state.load(0, "final") == kept

    tensors = state.dir / "tensors-0000001.pt"
    as_kept = tensors.read_bytes()
    for part in stated:
        flipped = changed(stopped["state"], part)
        torch.save(flipped, tensors)
        with pytest.raises(SystemExit, match=f"the SHA-512 of its {part}, as tensors-0000001.pt holds it, is "
                                             f"{run.kept_sha512(flipped)[part]} and the record of the segment that "
                                             f"kept it states {stated[part]}. Remove {state.dir} to start"):
            state.load(0, "final")
    tensors.write_bytes(as_kept[:-9])
    with pytest.raises(SystemExit, match="its tensors, tensors-0000001.pt, don't load as a state's"):
        state.load(0, "final")
    tensors.write_bytes(as_kept)
    assert state.load(0, "final") == kept

    tensors.unlink()
    with pytest.raises(SystemExit, match="its tensors, tensors-0000001.pt, are missing"):
        state.load(0, "final")
    state.clear()
    assert not state.dir.exists() and run.Kept(state.dir).load(0, "final") is None


def test_a_state_rewritten_whole_is_refused_against_the_stopped_runs_record(tmp_path, stopped):
    """A writer of the state directory writes `state.json` with the tensors, so a state rewritten whole, its
    record's digests matching its tensors, is one the directory alone can't tell from the kept one. `--continues`
    is the sha256 of `state.json` as the stopped run's record holds it: the kept state continues under it, the
    rewritten one is refused by both digests, and under `nothing` any state is. A directory that holds no state
    starts from θ under either."""
    state, _ = kept_state(tmp_path, stopped)
    manifest, tensors = state.dir / "state.json", state.dir / "tensors-0000001.pt"
    stored = hashlib.sha256(manifest.read_bytes()).hexdigest()
    kept = state.load(0, "final")
    assert state.load(0, "final", stored) == kept

    flipped = changed(stopped["state"], "optimizers")
    torch.save(flipped, tensors)
    whole = json.loads(manifest.read_text())
    whole["record"]["segments"][-1]["kept"] = run.kept_sha512(flipped)
    manifest.write_text(json.dumps(whole))
    rewritten = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert state.load(0, "final")["record"]["segments"][-1]["kept"] == run.kept_sha512(flipped) != kept["record"][
        "segments"][-1]["kept"]
    with pytest.raises(SystemExit, match=f"isn't one this job continues: its state.json hashes to {rewritten} "
                                         f".sha256. and the one this job was started to continue from, in the "
                                         f"stopped run's record, to {stored}. Remove {state.dir} to start"):
        state.load(0, "final", stored)
    with pytest.raises(SystemExit, match="whoever started this job found no state for it to continue from"):
        state.load(0, "final", NOTHING)
    assert json.loads(manifest.read_text()) == whole
    assert run.kept_sha512(torch.load(tensors, weights_only=True)) == run.kept_sha512(flipped)

    empty = run.Kept(tmp_path / "empty")
    assert empty.load(0, "final", stored) is None and empty.load(0, "final", NOTHING) is None
    assert run.continued(stored) == stored and run.continued(NOTHING) == NOTHING
    for text in ("", stored[:-1], stored.upper(), "none"):
        with pytest.raises(argparse.ArgumentTypeError, match="is neither 'nothing' nor a sha256 in hex"):
            run.continued(text)


def test_a_segment_that_names_another_job_than_the_kept_record_is_refused(stopped, rest):
    first = run.joined(None, stopped["record"])
    for over, why in (({"config": {"digest": "b" * 64}}, "config digest"), ({"seed": 5}, "seed"),
                      ({"steps": rest["steps"][1:]}, "first step")):
        with pytest.raises(SystemExit, match=f"the kept record's {why} is .* no record is written"):
            run.joined(first, rest | over)


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    """The tiny config asking for fresh windows, of a start that names the stream's middle part as read: five of
    the stream's six windows are fresh, so the job's four steps of two run out of them once, after step 1."""
    root = tmp_path_factory.mktemp("fresh")
    reg = tiny_registry(root / "reg", read=(1,))
    return root, reg, write_config(root / "a.toml", budget=BUDGET, windows="fresh")


def test_a_job_served_fresh_windows_is_served_after_a_stop_what_the_unbroken_job_is(fresh):
    """The harness alone, in one process: a job that asked for fresh windows is stopped after the checkpoint of
    step 1, and the segment that continues it is served, from the kept position, the windows the unbroken job's
    steps 2 and 3 are, the order starting again where the unbroken job's does. The two records joined state the
    unbroken job's windows; a segment that left other parts out is refused."""
    whole = run.joined(None, segment(fresh))
    cut = segment(fresh, stop_after=1)
    rest = segment(fresh, {"step": 2}, cut["state"])
    served = whole["windows"]["served"]
    assert len(served) == 4 and whole["windows"]["rounds"] == 1 and not {2} & {i for step in served for i in step}
    assert cut["state"]["stream"] == 2 * BUDGET["batch"]
    assert (cut["record"]["windows"]["served"], cut["record"]["windows"]["rounds"]) == (served[:2], 0)
    assert (rest["windows"]["served"], rest["windows"]["rounds"]) == (served[2:], 1)
    first = run.joined(None, cut["record"])
    both = run.joined(first, rest)
    assert both["windows"] == whole["windows"]
    assert_continues(whole, both, at=2)
    other = rest["windows"] | {"fresh": 4, "excluded": [e | {"part": 0} for e in rest["windows"]["excluded"]]}
    every = {k: v for k, v in rest.items() if k != "windows"}
    for segment_, states in ((rest | {"windows": other}, r"\[4, \[0\]\]"), (every, "None")):
        with pytest.raises(SystemExit, match=rf"the kept record's fresh windows .how many, and the parts left out. "
                                             rf"is \[5, \[1\]\] and the segment that continued it states {states}: "
                                             rf"no record is written"):
            run.joined(first, segment_)


def stop_once_kept(monkeypatch, step: int):
    """Tell the job to stop (SIGTERM) once its own process has kept its state after `step`."""
    write = run.Kept.write

    def stop_after(self, tensors, record, elapsed_s):
        write(self, tensors, record, elapsed_s)
        if record["checkpoints"][-1]["step"] == step:
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(run.Kept, "write", stop_after)


@pytest.mark.slow
def test_a_job_that_asked_for_fresh_windows_stopped_and_run_again_is_served_the_unbroken_jobs_windows(
        tmp_path, fresh, monkeypatch):
    """The job as a node runs it, three times, each two processes, so slow: what the quick test above doesn't reach
    is the windows served going from the route's process into the state the job's own process keeps, and from that
    state into the record of the job that continues it. An unbroken job that asked for fresh windows; the same job
    told to stop (SIGTERM) once it has kept its state after step 1, whose state holds the windows of steps 0 and 1;
    and the stopped job again, which continues, and whose record states the windows the unbroken job's does, step
    for step, none of them one the start read, with every checkpoint's state the unbroken job's bit for bit."""
    root, reg, path = fresh

    def job(out, state):
        return run.main(["--config", path, "--seed", 0, "--split", "final", "--out", tmp_path / out, "--starts", reg,
                         "--device", "cpu", "--state", tmp_path / state])

    assert job("whole", "state-whole") == 0
    whole = json.loads((tmp_path / "whole" / "run.json").read_text())
    served = whole["windows"]["served"]
    assert whole["invalid"] == [] and len(served) == 4 and whole["windows"]["rounds"] == 1
    assert [e["windows"] for e in whole["windows"]["excluded"]] == [[2, 3]]
    assert not {2} & {i for step in served for i in step}

    with monkeypatch.context() as stopping:
        stop_once_kept(stopping, 1)
        assert job("cut", "state") == run.STOPPED
    so_far = json.loads((tmp_path / "state" / "state.json").read_text())["record"]
    assert (so_far["windows"]["served"], so_far["windows"]["rounds"]) == (served[:2], 0) and len(so_far["steps"]) == 2

    assert job("cut", "state") == 0 and not (tmp_path / "state").exists()
    cut = json.loads((tmp_path / "cut" / "run.json").read_text())
    assert cut["invalid"] == [] and cut["windows"] == whole["windows"]
    assert_continues(whole, cut, at=2, keyed=False)
    states = [(cp["output_sha256"], cp["e_sha256"]) for cp in whole["checkpoints"]]
    assert [(cp["output_sha256"], cp["e_sha256"]) for cp in cut["checkpoints"]] == states and len(set(states)) == 4


@pytest.mark.slow
@pytest.mark.parametrize("route", sorted(OUTPUTS))
def test_a_stopped_job_of_a_route_with_an_output_continues_and_writes_the_unbroken_jobs_record(
        tmp_path, fresh, monkeypatch, route):
    """The job as a node runs it, three times, each two processes, so slow: what the quick test of the harness alone
    doesn't reach is the kept model on its way through the job's own process, which holds it to the record's last
    checkpoint by the digest of the model the route trains and not of the output it scored (`Kept.load`), and into
    a route's process that built its model anew. A route with an `output` hands out its output model at a
    checkpoint (`ema` an average, `lora` its adapters merged into their weights). So: an unbroken job that asked
    for fresh windows; the same job told to stop (SIGTERM) once it has kept its state after step 1, whose state
    holds the trained model, not the output; and the stopped job again, which continues in a second segment and
    writes the unbroken job's record: every step's count (`assert_continues`), every checkpoint's output bit for
    bit, and the windows served step for step, none of them one the start read."""
    _, reg, _ = fresh
    path = write_config(tmp_path / "a.toml", route=route, budget=BUDGET, windows="fresh", dials=OUTPUTS[route])

    def job(out, state):
        return run.main(["--config", path, "--seed", 0, "--split", "final", "--out", tmp_path / out, "--starts", reg,
                         "--device", "cpu", "--state", tmp_path / state])

    assert job("whole", "state-whole") == 0 and not (tmp_path / "state-whole").exists()
    whole = json.loads((tmp_path / "whole" / "run.json").read_text())
    served = whole["windows"]["served"]
    assert whole["invalid"] == [] and len(whole["segments"]) == 1
    assert len(served) == 4 and whole["windows"]["rounds"] == 1 and not {2} & {i for step in served for i in step}

    with monkeypatch.context() as stopping:
        stop_once_kept(stopping, 1)
        assert job("cut", "state") == run.STOPPED
    assert told(json.loads((tmp_path / "cut" / "run.json").read_text()))
    so_far = json.loads((tmp_path / "state" / "state.json").read_text())["record"]
    last, kept = so_far["checkpoints"][-1], torch.load(tmp_path / "state" / "tensors-0000001.pt", weights_only=True)
    assert last["step"] == 1 and run.theta_sha256(kept["model"]) == last["trained_sha256"] != last["output_sha256"]
    assert so_far["segments"][-1]["kept"] == run.kept_sha512(kept) and sorted(kept) == [
        "buffers", "generators", "model", "optimizers", "stream"]
    assert (so_far["windows"]["served"], so_far["windows"]["rounds"]) == (served[:2], 0)

    assert job("cut", "state") == 0 and not (tmp_path / "state").exists()
    cut = json.loads((tmp_path / "cut" / "run.json").read_text())
    assert cut["invalid"] == [] and cut["windows"] == whole["windows"]
    assert_continues(whole, cut, at=2, keyed=False)
    states = [(cp["output_sha256"], cp["e_sha256"]) for cp in whole["checkpoints"]]
    assert [(cp["output_sha256"], cp["e_sha256"]) for cp in cut["checkpoints"]] == states and len(set(states)) == 4


def test_a_route_that_names_a_signal_is_refused():
    assert run.screen("import torch\n") == []
    assert run.screen("import os\nos.kill(os.getppid(), 15)\n") and run.screen("import signal\n")


def test_a_job_told_to_stop_writes_the_record_of_its_stop_whatever_it_is_told_meanwhile(tmp_path, tiny, monkeypatch):
    """A job told to stop (SIGTERM) writes `run.json` before it ends: whose it is and that it was told to stop, with
    no step or checkpoint for a grade to score. That record is all that says the run was stopped, so a second
    SIGTERM doesn't interrupt it: `timeout` signals the job and then its process group. Here the job is told as it
    reads its state directory, before it starts the route's process, and again as it makes the record."""
    root, reg, path = tiny
    make, before = run.stop_record, signal.getsignal(signal.SIGTERM)

    def told_again(seed, split):
        os.kill(os.getpid(), signal.SIGTERM)
        return make(seed, split)

    monkeypatch.setattr(run, "stop_record", told_again)
    monkeypatch.setattr(run.Kept, "load", lambda *a: os.kill(os.getpid(), signal.SIGTERM))
    assert run.main(["--config", path, "--seed", 0, "--split", "final", "--out", tmp_path / "out", "--starts", reg,
                     "--state", tmp_path / "state"]) == run.STOPPED
    assert signal.getsignal(signal.SIGTERM) is before
    rec = json.loads((tmp_path / "out" / "run.json").read_text())
    assert rec == stop_record(0, "final") == {
        "schema": "training-slowdown/run/v1", "training_slowdown": VERSION, "seed": 0, "split": "final",
        "stopped": TOLD, "steps": [], "checkpoints": [], "invalid": ["the job was told to stop before its last step"]}
    assert told(rec) and grade.rows_of(rec) == []
    assert not told(rec | {"stopped": "12.0 s past the tier's 10.0 s budget"}) and not told(None) and not told([TOLD])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out", "state"] and not any((tmp_path / "state").iterdir())


@pytest.mark.slow
def test_a_job_stopped_after_a_checkpoint_and_run_again_writes_the_unbroken_jobs_record(tmp_path, tiny, monkeypatch):
    """The job as a node runs it, four times, each two processes, so slow: what the quick tests above don't reach is
    the job's own process keeping the state, the stop, and the sandbox. An unbroken job; the same job told to stop
    (SIGTERM) once it has kept its state after step 1, and told again as it ends the route's process, which ends
    with `STOPPED`, the record of its stop and the state left; a
    job of another config on that state, refused by both digests before it trains, the state left; the stopped job
    on its state with one bit of the optimizers' part flipped, in the state directory and then only in the copy
    handed to the route's process, refused each time by that part's two SHA-512; the stopped job told to continue
    from nothing, and from a `state.json` of another sha256, refused each time; and the stopped job again, told
    the sha256 of its `state.json`, which continues while its state directory, hidden from the route's process,
    still holds the state, writes the unbroken job's record in two segments, the first with the digests of the
    state it kept, and removes the state. Every checkpoint's state is the unbroken job's bit for bit; its score is
    another forward pass's, so within `SCORE_REL`."""
    root, reg, path = tiny

    def job(out, state, cfg=path, *more):
        return run.main(["--config", cfg, "--seed", 0, "--split", "final", "--out", tmp_path / out, "--starts", reg,
                         "--device", "cpu", "--state", tmp_path / state, *more])

    assert job("whole", "state-whole") == 0 and not (tmp_path / "state-whole").exists()
    whole = json.loads((tmp_path / "whole" / "run.json").read_text())
    assert whole["invalid"] == [] and len(whole["segments"]) == 1

    write = run.Kept.write

    def stop_after(self, tensors, record, elapsed_s):
        write(self, tensors, record, elapsed_s)
        if record["checkpoints"][-1]["step"] == 1:
            os.kill(os.getpid(), signal.SIGTERM)

    terminate, ended = run.subprocess.Popen.terminate, []

    def told_again(self):
        # the route's process is still there, and the record of the stop already is
        ended.append((self.poll(), json.loads((tmp_path / "cut" / "run.json").read_text())))
        os.kill(os.getpid(), signal.SIGTERM)
        terminate(self)

    with monkeypatch.context() as stopping:
        stopping.setattr(run.Kept, "write", stop_after)
        stopping.setattr(run.subprocess.Popen, "terminate", told_again)
        assert job("cut", "state") == run.STOPPED and ended == [(None, stop_record(0, "final"))]
    assert told(json.loads((tmp_path / "cut" / "run.json").read_text()))
    kept = sorted(p.name for p in (tmp_path / "state").iterdir())
    assert kept == ["state.json", "tensors-0000001.pt"]
    so_far = json.loads((tmp_path / "state" / "state.json").read_text())["record"]
    assert [cp["score"]["split"] for cp in so_far["checkpoints"]] == ["final"] * 2 and len(so_far["steps"]) == 2

    other = write_config(tmp_path / "other.toml", budget=BUDGET | {"steps": 5})
    with pytest.raises(SystemExit, match=f"its config digest is {so_far['config']['digest']} and this job's is "
                                         f"{config.load(other, starts.registry(reg)).digest}"):
        job("other", "state", other)
    assert sorted(p.name for p in (tmp_path / "state").iterdir()) == kept and not (tmp_path / "other").exists()

    stated = so_far["segments"][-1]["kept"]
    tensors = tmp_path / "state" / "tensors-0000001.pt"
    as_kept = tensors.read_bytes()
    flipped = changed(torch.load(tensors, weights_only=True), "optimizers")
    refusal = (f"the SHA-512 of its optimizers, as {{}}, is {run.kept_sha512(flipped)['optimizers']} and the record "
               f"of the segment that kept it states {stated['optimizers']}")
    torch.save(flipped, tensors)
    with pytest.raises(SystemExit, match=refusal.format("tensors-0000001.pt holds it")):
        job("changed", "state")
    tensors.write_bytes(as_kept)
    with monkeypatch.context() as on_the_way:
        on_the_way.setattr(run.shutil, "copyfile", lambda src, dst: torch.save(flipped, dst))
        with pytest.raises(SystemExit, match=refusal.format("the route's process read it")):
            job("changed", "state")
    assert sorted(p.name for p in (tmp_path / "state").iterdir()) == kept and tensors.read_bytes() == as_kept
    assert not (tmp_path / "changed").exists()

    stored = hashlib.sha256((tmp_path / "state" / "state.json").read_bytes()).hexdigest()
    with pytest.raises(SystemExit, match="whoever started this job found no state for it to continue from"):
        job("changed", "state", path, "--continues", NOTHING)
    with pytest.raises(SystemExit, match=f"its state.json hashes to {stored} .sha256. and the one this job was "
                                         f"started to continue from, in the stopped run's record, to {'0' * 64}"):
        job("changed", "state", path, "--continues", "0" * 64)
    assert sorted(p.name for p in (tmp_path / "state").iterdir()) == kept and not (tmp_path / "changed").exists()

    assert job("cut", "state", path, "--continues", stored) == 0 and not (tmp_path / "state").exists()
    cut = json.loads((tmp_path / "cut" / "run.json").read_text())
    assert cut["invalid"] == [] and cut["digest_key"] == so_far["digest_key"] != whole["digest_key"]
    assert [s.get("kept") for s in cut["segments"]] == [stated, None] and "kept" not in whole["segments"][0]
    assert_continues(whole, cut, at=2, keyed=False)
    # the states are the unbroken job's bit for bit; their scores are two forward passes' each
    states = [(cp["output_sha256"], cp["e_sha256"]) for cp in whole["checkpoints"]]
    assert [(cp["output_sha256"], cp["e_sha256"]) for cp in cut["checkpoints"]] == states and len(set(states)) == 4
    assert [cp["score"] for cp in cut["checkpoints"]] == [scored(cp)["score"] for cp in whole["checkpoints"]]
    assert all(cp["score"]["bpb"] == cp["bpb"] > 0 for cp in cut["checkpoints"])
    assert sorted(p.name for p in (tmp_path / "cut").iterdir()) == ["e", "run.json"]
