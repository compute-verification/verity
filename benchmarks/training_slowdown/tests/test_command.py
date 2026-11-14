import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import synthetic
from synthetic import tiny_registry, write_config

from training_slowdown import STOPPED, TOLD, config, grade, starts, stop_record

COMMAND = Path(__file__).resolve().parents[1] / "run.py"
_spec = importlib.util.spec_from_file_location("training_slowdown_command", COMMAND)
cmd = sys.modules[_spec.name] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmd)

ART = "art:" + "e" * 64
#: the store's sha256 of the state.json a record holds
STATED = "5" * 64


def env(job) -> str:
    """How a queued job's script starts: a directory of the run's own under /tmp and the directory a state is fetched
    into, both removed when its shell ends; a SIGTERM that writes the job's record of the stop where the job wrote
    none and ends the script with the stopped command's status; and the job's Python environment in that directory."""
    told = shlex.quote(json.dumps(stop_record(job.seed, job.split)))
    return ('tmp=$(mktemp -d /tmp/training-slowdown.XXXXXX) && fetching="$(dirname "$OUT")/fetching" && '
            'trap \'rm -rf "$tmp" "$fetching"\' EXIT && stopped=' + told + ' && '
            'trap \'status=$?; [ -e "$OUT/run.json" ] || { mkdir -p "$OUT" && printf "%s\\n" "$stopped" > '
            '"$OUT/run.json"; [ "$status" -ne 0 ] || status=143; }; exit $status\' TERM && '
            'export UV_PROJECT_ENVIRONMENT="$tmp/venv"')


class Research:
    """`research` as the command calls it: labels, attempts and the runs it starts, all in memory."""

    def __init__(self):
        self.labels: list[tuple[str, str, str]] = []
        self.attempts: dict[str, dict] = {}
        self.runs: list[list[str]] = []
        self.sync_rc = 0
        #: the records that hold a job's state, and the sha256 the store lists for each one's state.json
        self.states: set[str] = set()
        self.sha256 = STATED
        self.shas: dict[str, str] = {}
        #: the text of the run.json, and of the state.json, each record holds
        self.said: dict[str, str] = {}
        self.kept: dict[str, str] = {}
        #: the records a file was fetched from, in order
        self.fetched: list[str] = []

    def __call__(self, argv):
        assert argv[0] == "research", argv
        if argv[1] == "run":
            self.runs.append(argv)
            return SimpleNamespace(returncode=0, stdout=f"queued r20990101-000000-{len(self.runs):04x}\n", stderr="")
        verb = argv[2]
        if verb == "fetch" and "--to" in argv:
            assert argv[4] == "--path" and argv[6] == "--to" and len(argv) == 8, argv
            self.fetched.append(argv[3])
            texts = {"out/run.json": self.said, cmd.STATE_JSON: self.kept}[argv[5]]
            if argv[3] not in texts:
                return SimpleNamespace(returncode=1, stdout="", stderr="no file matches")
            (Path(argv[7]) / argv[5]).parent.mkdir(parents=True)
            (Path(argv[7]) / argv[5]).write_text(texts[argv[3]])
            return SimpleNamespace(returncode=0, stdout=argv[7], stderr="")
        if verb == "fetch":
            assert argv[4:] == ["--list", "--json", "--path", "out/state/state.json"], argv
            held = argv[3] in self.states
            return SimpleNamespace(returncode=0 if held else 1, stderr="" if held else "no file matches",
                                   stdout=json.dumps([{"path": "out/state/state.json",
                                                       "sha256": self.shas.get(argv[3], self.sha256)}])
                                   if held else "")
        if verb == "labels-sync":
            return SimpleNamespace(returncode=self.sync_rc, stdout="", stderr="unreachable")
        if verb == "sql":
            assert f"key = '{config.LABEL_KEY}'" in argv[3]
            rows = [{"target": t, "value": v} for t, k, v in self.labels if k == config.LABEL_KEY]
            return SimpleNamespace(returncode=0, stdout=json.dumps(rows), stderr="")
        if verb == "show":
            a = self.attempts.get(argv[3])
            return SimpleNamespace(returncode=0 if a else 1, stdout=json.dumps(a) if a else "", stderr="")
        if verb == "label":
            assert argv[6:] == ["--by", cmd.BY], argv
            self.labels.append((argv[3], argv[4], argv[5]))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(argv)

    def run_of(self, label):
        return next(t for t, _, v in self.labels if v == label and cmd.RUN_RE.fullmatch(t))


@pytest.fixture(scope="module")
def reg_path(tmp_path_factory):
    return tiny_registry(tmp_path_factory.mktemp("reg"))


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A checkout whose `benchmarks/training_slowdown/attacks/` holds the test's configs."""
    root = tmp_path.resolve() / "tree"
    here = root / "benchmarks" / "training_slowdown"
    monkeypatch.setattr(cmd, "HERE", here)
    monkeypatch.setattr(cmd, "ROOT", root)
    return here / "attacks"


def command(research, *argv):
    return cmd.main(list(argv), store=cmd.Store(research))


def test_each_missing_job_is_queued_once_and_labelled(tree, reg_path, capsys):
    path = write_config(tree / "a.toml", seeds={"search": [0], "final": [101, 102]})
    cfg = config.load(path, starts.registry(reg_path))
    r = Research()
    assert command(r, "a", "--starts", str(reg_path)) == 3
    assert len(r.runs) == 3
    for argv, job in zip(r.runs, cfg.jobs()):
        assert argv[:14] == ["research", "run", "--queue", "--kind", "training-attack", "--preemptible", "--project",
                             "verity", "--source", str(tree.parents[2]), "--cwd", "source", "--gpus", "1"]
        script = argv[-1]
        assert argv[-3:-1] == ["bash", "-lc"] and script.startswith(
            f'{env(job)} && {cmd.SYNC} && cd benchmarks/training_slowdown && "$UV_PROJECT_ENVIRONMENT/bin/python" '
            f'-m training_slowdown.run ')
        assert (f"--config attacks/a.toml --seed {job.seed} --split {job.split} --starts {reg_path.resolve()}"
                in script and script.endswith(f'--state "$OUT/state" --continues {cmd.NOTHING} --out "$OUT"'))
        assert ".venv" not in script and "exec " not in script
        assert (r.run_of(job.label), config.LABEL_KEY, job.label) in r.labels
    assert (cmd.TMP, cmd.SYNC) == ("/tmp", "uv sync -q --frozen --all-packages --extra torch")
    assert "3 of 3 jobs pending" in capsys.readouterr().out
    assert command(r, "a", "--starts", str(reg_path)) == 3 and len(r.runs) == 3  # pending: nothing queued again


def test_a_done_job_labels_its_record_and_a_failed_one_is_never_run_again(tree, reg_path, capsys):
    path = write_config(tree / "a.toml", seeds={"search": [0], "final": [101]})
    search, final = config.load(path, starts.registry(reg_path)).jobs()
    r = Research()
    command(r, str(path), "--starts", str(reg_path))
    r.attempts[r.run_of(search.label)] = {"status": {"rc": 0}, "outputs": {"run_record": ART}}
    r.attempts[r.run_of(final.label)] = {"status": {"rc": 1}, "outputs": {}}
    assert command(r, str(path), "--starts", str(reg_path)) == 1
    assert (ART, config.LABEL_KEY, search.label) in r.labels and len(r.runs) == 2
    out = capsys.readouterr().out
    assert "done" in out and "failed" in out and "rc 1, no record" in out and "never run again" in out
    r.attempts[r.run_of(final.label)] = {"status": {"rc": 0}, "outputs": {"run_record": ART}}
    assert command(r, str(path), "--starts", str(reg_path)) == 0 and len(r.runs) == 2
    assert r.labels.count((ART, config.LABEL_KEY, search.label)) == 1
    assert "every job has its record (2)" in capsys.readouterr().out


def last_run(r: Research, job) -> str:
    return sorted(t for t, _, v in r.labels if v == job.label and cmd.RUN_RE.fullmatch(t))[-1]


def stop(r: Research, job, record=ART, machine="node-7", rc=124) -> str:
    """The job's last run ended as one a lease out of time stopped: `timeout`'s status, and a record whose run.json
    is the job's `stop_record`; its id."""
    run = last_run(r, job)
    r.attempts[run] = {"status": {"rc": rc}, "outputs": {"run_record": record}, "execution": {"machine": machine}}
    r.said[record] = json.dumps(stop_record(job.seed, job.split))
    return run


def test_a_stop_is_read_from_the_runs_record_and_never_from_its_exit_status(tree, reg_path, capsys):
    """A run that ended with any status but 0 is stopped when the run.json in its record says the job was told to
    stop, and the command's next run continues it: under `timeout`'s 124, which a lease out of time reports whatever
    the job exits with, a signal's 143 or 137, or the job's own. With no record, a record with no run.json or one
    that doesn't read, or a run.json that doesn't say so, the job's last record with the budget's `stopped` among
    them, it is failed and never run again, the job's own `STOPPED` status included."""
    path = write_config(tree / "a.toml")
    (job,) = config.load(path, starts.registry(reg_path)).jobs()
    args = ("a", "--starts", str(reg_path))
    for rc in (124, 143, 137, STOPPED):
        r = Research()
        command(r, *args)
        first = stop(r, job, rc=rc)
        assert command(r, *args, "--dry-run") == 3 and f"stopped  {first}" in capsys.readouterr().out
        assert command(r, *args) == 3 and len(r.runs) == 2 and f"continues {first}" in capsys.readouterr().out
        assert last_run(r, job) != first

    last = stop_record(job.seed, job.split) | {"stopped": "12.0 s past the tier's 10.0 s budget", "invalid": []}
    assert TOLD == "told to stop" and not cmd.told(last) and cmd.told(stop_record(0, "final"))
    unsaid = "a record that doesn't say the job was told to stop"
    for rc in (124, STOPPED):
        for record, said, why in ((None, None, "no record"), (ART, None, "no record"),
                                  (ART, json.dumps(stop_record(job.seed, job.split))[:-1], "no record"),
                                  (ART, json.dumps([TOLD]), unsaid), (ART, json.dumps(last), unsaid),
                                  (ART, json.dumps(last | {"stopped": ""}), unsaid)):
            r = Research()
            command(r, *args)
            r.attempts[last_run(r, job)] = {"status": {"rc": rc}, "outputs": {"run_record": record}}
            if said is not None:
                r.said[record] = said
            capsys.readouterr()
            assert command(r, *args) == 1 and len(r.runs) == 1
            out, err = capsys.readouterr()
            assert f"failed   {last_run(r, job)} (rc {rc}, {why})" in out and "never run again" in out
            # only a run.json that doesn't read is said: one a record doesn't hold is the usual failed job
            unread = said is not None and why == "no record"
            assert (f"{ART} holds no run.json that reads (JSONDecodeError: " in err) == unread


def test_a_stopped_job_is_run_again_from_the_state_its_run_kept(tree, reg_path, capsys):
    """A job whose run's record says it was told to stop is started again by the command's next run, on whichever
    machine the queue gives it: where the stopped run's record holds a state, the new run fetches it into its own $OUT before
    the job starts, with the store's read key, and the job is told the sha256 the store lists for that record's
    state.json, to continue from no other; where it holds none, or the store lists no sha256 for it, the job starts
    again from θ and is told to continue from nothing."""
    path = write_config(tree / "a.toml", seeds={"search": [0], "final": [101]})
    search, final = config.load(path, starts.registry(reg_path)).jobs()
    r = Research()
    command(r, "a", "--starts", str(reg_path))
    r.states.add(ART)
    first = stop(r, search, ART)
    stop(r, final, "art:" + "f" * 64)
    assert command(r, "a", "--starts", str(reg_path), "--dry-run") == 3 and len(r.runs) == 2
    out = capsys.readouterr().out
    assert f"stopped  {first} (its record holds its state)" in out
    assert "(it kept no state: it starts again from θ)" in out
    assert "2 of 2 jobs pending, 2 of them stopped: run the command again to continue them" in out
    assert out.count("would run: research run --queue") == 2 and out.count("--store-read") == 1

    assert command(r, "a", "--starts", str(reg_path)) == 3 and len(r.runs) == 4
    again, fresh = r.runs[2:]
    assert again[:6] == fresh[:6] == ["research", "run", "--queue", "--kind", "training-attack", "--preemptible"]
    assert again[-5] == "--store-read" and "--store-read" not in fresh and "--on" not in again
    assert again[-1].startswith(f'{env(search)} && {cmd.SYNC} && "$UV_PROJECT_ENVIRONMENT/bin/research" data fetch '
                                f'{ART} --path "out/state/*" --to "$fetching" && mkdir -p "$OUT" && mv '
                                f'"$fetching/out/state" "$OUT/state" && cd benchmarks/training_slowdown '
                                f'&& "$UV_PROJECT_ENVIRONMENT/bin/python" -m ')
    assert "--threads" not in again[-1]
    assert "research data fetch" not in fresh[-1] and fresh[-1] == r.runs[1][-1]
    assert again[-1].endswith(f'--state "$OUT/state" --continues {STATED} --out "$OUT"')
    assert fresh[-1].endswith('--state "$OUT/state" --continues nothing --out "$OUT"')
    assert f"continues {first}" in capsys.readouterr().out
    assert r.run_of(search.label) == first and len([t for t, _, v in r.labels if v == search.label]) == 2
    assert command(r, "a", "--starts", str(reg_path)) == 3 and len(r.runs) == 4   # pending: nothing queued again
    r.attempts[sorted(t for t, _, v in r.labels if v == search.label)[-1]] = {"status": {"rc": 0},
                                                                              "outputs": {"run_record": ART}}
    assert "done" in (command(r, "a", "--starts", str(reg_path)), capsys.readouterr().out)[1]

    r = Research()
    r.sha256 = '$(touch x)"'
    command(r, "a", "--starts", str(reg_path))
    r.states.add(ART)
    stop(r, search, ART)
    capsys.readouterr()
    assert command(r, "a", "--starts", str(reg_path), "--dry-run") == 3
    assert "(it kept no state: it starts again from θ)" in capsys.readouterr().out
    assert command(r, "a", "--starts", str(reg_path)) == 3 and len(r.runs) == 3
    assert r.runs[2][-1] == r.runs[0][-1] and "--store-read" not in r.runs[2] and "touch" not in r.runs[2][-1]


def test_a_run_stopped_before_its_job_was_up_loses_no_earlier_state(tree, reg_path, capsys):
    """A run stopped before its job was up kept no state of its own: the job goes on from the latest state an earlier
    run of it kept, told the store's sha256 for it and the CPU thread count that state's record states."""
    path = write_config(tree / "a.toml")
    (job,) = config.load(path, starts.registry(reg_path)).jobs()
    args = ("a", "--starts", str(reg_path))
    r = Research()
    command(r, *args)
    r.states.add(ART)
    r.kept[ART] = json.dumps({"record": {"threads": 3}})
    first = stop(r, job, ART)
    assert command(r, *args) == 3 and len(r.runs) == 2
    continued = r.runs[1][-1]
    assert f"data fetch {ART} " in continued
    assert f' --threads 3 --state "$OUT/state" --continues {STATED} ' in continued
    second = stop(r, job, "art:" + "f" * 64)
    capsys.readouterr()
    assert command(r, *args, "--dry-run") == 3
    assert f"stopped  {second} (it kept no state: it goes on from the state {first} kept)" in capsys.readouterr().out
    assert command(r, *args) == 3 and len(r.runs) == 3 and r.runs[2][-1] == continued


def test_a_job_stopped_every_time_before_its_next_checkpoint_stalls_until_told_to_continue(tree, reg_path, capsys):
    """A job whose last `STALLS` runs each kept no newer state than the one before them, none or the one they were
    started on, is stalled: the command names those runs and continues it only under `--continue-stalled`, which
    allows `STALLS` more such stops; a run that keeps a new state starts the count again. A failed job is never run
    again, under `--continue-stalled` too."""
    path = write_config(tree / "a.toml")
    (job,) = config.load(path, starts.registry(reg_path)).jobs()
    args = ("a", "--starts", str(reg_path))
    none, same = "art:" + "f" * 64, "art:" + "a" * 64
    r = Research()
    command(r, *args)
    r.states.update({ART, same})
    stop(r, job, ART)
    assert command(r, *args) == 3 and len(r.runs) == 2
    continued = r.runs[1][-1]
    stalled = [stop(r, job, none)]
    assert command(r, *args) == 3 and len(r.runs) == 3
    stalled.append(stop(r, job, none))
    assert command(r, *args) == 3 and len(r.runs) == 4
    stalled.append(stop(r, job, same))
    capsys.readouterr()
    assert cmd.STALLS == 3 and command(r, *args) == 4 and len(r.runs) == 4
    out = capsys.readouterr().out
    assert (f"stalled  {stalled[-1]} (its last 3 runs, {', '.join(stalled)}, were each stopped and kept no newer "
            "state than the one before them: it is stopped every time before its next checkpoint, and "
            "`--continue-stalled` continues it for 3 more such stops)" in out)
    assert "1 of 1 jobs stalled: run the command with --continue-stalled to continue them" in out
    assert command(r, *args) == 4 and len(r.runs) == 4
    assert command(r, *args, "--continue-stalled") == 3 and len(r.runs) == 5
    assert r.runs[4][-1] == continued.replace(ART, same)
    for n in range(cmd.STALLS - 1):
        stop(r, job, none)
        assert command(r, *args) == 3 and len(r.runs) == 6 + n
    stop(r, job, none)
    assert command(r, *args) == 4 and len(r.runs) == 6 + n
    assert command(r, *args, "--continue-stalled") == 3
    newer = "art:" + "b" * 64
    r.states.add(newer)
    r.shas[newer] = "6" * 64
    stop(r, job, newer)
    assert command(r, *args) == 3
    stop(r, job, none)
    assert command(r, *args) == 3 and f"data fetch {newer} " in r.runs[-1][-1] and "6" * 64 in r.runs[-1][-1]

    r = Research()
    command(r, *args)
    for _ in range(cmd.STALLS - 1):
        stop(r, job, none)
        assert command(r, *args) == 3
    stop(r, job, none)
    assert command(r, *args) == 4 and len(r.runs) == cmd.STALLS
    r.attempts[last_run(r, job)] = {"status": {"rc": 1}, "outputs": {}}
    assert command(r, *args, "--continue-stalled") == 1 and len(r.runs) == cmd.STALLS


def test_a_state_kept_on_the_nodes_disk_sends_the_job_back_to_its_machine(tree, reg_path, monkeypatch):
    """`NODE_STATE` is the one line that places a job's state: a directory of the nodes' own disks keeps each job's
    state there under its label, and the job's next run asks the queue for the stopped run's machine and fetches
    nothing."""
    monkeypatch.setattr(cmd, "NODE_STATE", "/scratch/training-slowdown")
    path = write_config(tree / "a.toml")
    (job,) = config.load(path, starts.registry(reg_path)).jobs()
    r = Research()
    command(r, "a", "--starts", str(reg_path))
    assert r.runs[0][-1].endswith(f'--state /scratch/training-slowdown/{job.label} --out "$OUT"')
    stop(r, job, ART, machine="node-7")
    assert command(r, "a", "--starts", str(reg_path)) == 3
    assert r.runs[1][2:8] == ["--queue", "--kind", "training-attack", "--preemptible", "--project", "verity"]
    assert r.runs[1][-6:-4] == ["--on", "node-7"] and r.runs[1][-1] == r.runs[0][-1]
    assert "--continues" not in r.runs[1][-1]
    assert "--store-read" not in r.runs[1]


def test_a_jobs_run_gets_the_time_its_configs_budget_gives_it(tree, reg_path):
    """`budget.minutes` is the job's own limit, over every run it continues in; a run gets `BUDGET_SLACK` times it
    and `OUTSIDE_MIN`, on the queue and on a named machine. A budget with no minutes leaves the kind's time."""
    write_config(tree / "a.toml", budget={"steps": 3, "batch": 2, "minutes": 51})
    write_config(tree / "b.toml")
    queued, on, plain = Research(), Research(), Research()
    command(queued, "a", "--starts", str(reg_path))
    command(on, "a", "--starts", str(reg_path), "--on", "node-1")
    command(plain, "b", "--starts", str(reg_path))
    assert cmd.OUTSIDE_MIN == 30 and queued.runs[0][-6:-4] == ["--max-min", "92"]
    assert on.runs[0][2:4] == ["--on", "node-1"] and on.runs[0][-6:-4] == ["--timeout", "92m"]
    assert "--max-min" not in plain.runs[0] and "--timeout" not in plain.runs[0]


#: of each start of starts.toml: the GB a job's first run asks the queue for, and a run that fetched its state
DISK_GB = {"s-t20": (22, 22), "s-t200": (22, 22), "m-t20": (23, 24), "m-t200": (23, 24), "l16-t20": (29, 32),
           "xl-t20": (44, 52)}


def test_a_queued_run_asks_for_the_disk_its_starts_size_needs(tree, reg_path, monkeypatch):
    """A run's disk is its start's size and nothing measured: 4 bytes a parameter of starts.toml, `STATE_OVER_MODEL`
    models a state, `STATE_COPIES` states at once and a fourth in a run that fetched its state, and `OUTSIDE_GB`.
    The figures are pinned for every start of the registry; the queue is asked for a job's, and a named machine,
    which declares nothing to it, for none."""
    reg = starts.registry()
    assert set(reg["starts"]) == set(DISK_GB)
    assert (cmd.STATE_OVER_MODEL, cmd.STATE_COPIES, cmd.OUTSIDE_GB) == (3, 3, 20)
    for name, (first, fetched) in DISK_GB.items():
        s = starts.start(name, reg)
        state = cmd.STATE_OVER_MODEL * 4 * s.shape["parameters"]
        assert 3 * state <= (first - cmd.OUTSIDE_GB) * 10**9 < 3 * state + 10**9, name
        assert 4 * state <= (fetched - cmd.OUTSIDE_GB) * 10**9 < 4 * state + 10**9, name
        assert (cmd.run_disk_gb(s), cmd.run_disk_gb(s, fetched=True)) == (first, fetched), name
    xl = starts.start("xl-t20", reg)
    assert (cmd.run_disk_gb(xl, weights=True), cmd.run_disk_gb(xl, weights=True, fetched=True)) == (47, 55)

    write_config(tree / "a.toml")
    (job,) = config.load(tree / "a.toml", starts.registry(reg_path)).jobs()
    tiny = cmd.run_disk_gb(job.config.start)
    assert tiny == cmd.OUTSIDE_GB + 1
    r, on = Research(), Research()
    command(r, "a", "--starts", str(reg_path))
    assert r.runs[0][14:16] == ["--disk-gb", str(tiny)]
    r.states.add(ART)
    stop(r, job, ART)
    command(r, "a", "--starts", str(reg_path))
    assert r.runs[1][14:16] == ["--disk-gb", str(tiny)] and r.runs[1][-5] == "--store-read"
    monkeypatch.setattr(cmd, "run_disk_gb", lambda start, weights, fetched: f"{start.name}/{weights}/{fetched}")
    r = Research()
    command(r, "a", "--starts", str(reg_path), "--weights")
    stop(r, job, ART)
    r.states.add(ART)
    command(r, "a", "--starts", str(reg_path))
    assert [argv[14:16] for argv in r.runs] == [["--disk-gb", "tiny/True/False"], ["--disk-gb", "tiny/False/True"]]
    command(on, "a", "--starts", str(reg_path), "--on", "node-1")
    assert "--disk-gb" not in on.runs[0]


def test_the_honest_configs_jobs_come_with_it_and_on_names_the_machine(tree, reg_path):
    write_config(tree / "ref.toml", dials={"trainable": "*"})
    write_config(tree / "a.toml", honest="ref")
    r = Research()
    assert command(r, "a", "--starts", str(reg_path), "--on", "node-1", "--pool", "p") == 3
    assert [argv[2:6] for argv in r.runs] == [["--on", "node-1", "--pool", "p"]] * 2
    assert any("attacks/ref.toml" in argv[-1] for argv in r.runs)


def test_a_job_keeps_its_weights_only_under_the_commands_one_flag(tree, reg_path):
    write_config(tree / "a.toml")
    r = Research()
    assert command(r, "a", "--starts", str(reg_path)) == 3 and "--keep-weights" not in r.runs[0][-1]
    r = Research()
    assert command(r, "a", "--starts", str(reg_path), "--weights") == 3
    assert f"--starts {reg_path.resolve()} --keep-weights --state" in r.runs[0][-1]


def test_a_dry_run_starts_and_labels_nothing(tree, reg_path, capsys):
    write_config(tree / "a.toml")
    r = Research()
    r.sync_rc = 1
    assert command(r, "a", "--starts", str(reg_path), "--dry-run") == 3
    assert r.runs == [] and r.labels == []
    assert "would run: research run --queue" in capsys.readouterr().out


@pytest.mark.parametrize("name, jobs", [("s-t20-1.1x-shallow-strideshare", 8), ("s-t20-1.1x-ref-optim", 4)])
def test_the_trees_configs_dry_run_against_the_registry(name, jobs, capsys):
    """Each config of `attacks/` loads against starts.toml, its honest config beside it, and a dry run names every
    job it would queue: one per seed and split of the config and of its honest config."""
    r = Research()
    assert command(r, name, "--dry-run") == 3
    assert r.runs == [] and r.labels == []
    out = capsys.readouterr().out
    assert out.count("would run: research run --queue --kind training-attack") == jobs
    assert out.count(f"--config attacks/{name}.toml") == 4 and f"{jobs} of {jobs} jobs pending" in out
    cfg = config.load(cmd.HERE / "attacks" / f"{name}.toml", starts.registry())
    assert (cfg.start.name, cfg.route, cfg.target, cfg.policy) == ("s-t20", "subset", {"gain": 1.1}, "console")
    assert cfg.seeds == {"search": [0], "final": [0, 101, 102]} and "minutes" not in cfg.budget
    honest = config.honest(cfg, starts.registry())
    assert (honest.name if honest else None) == {8: "s-t20-1.1x-ref-optim", 4: None}[jobs]


def test_a_queued_job_reads_its_config_from_the_tree(tree, reg_path, tmp_path):
    path = write_config(tmp_path / "elsewhere" / "a.toml")
    with pytest.raises(SystemExit, match="is outside"):
        command(Research(), str(path), "--starts", str(reg_path))


def test_local_records_each_jobs_exit_and_never_reruns_a_failed_job(tmp_path, reg_path, monkeypatch, capsys):
    path = write_config(tmp_path / "a.toml", seeds={"search": [0, 1]})
    calls = []

    def fake(argv, cwd):
        calls.append(argv)
        return SimpleNamespace(returncode=1 if "1" == argv[argv.index("--seed") + 1] else 0)
    monkeypatch.setattr(cmd.subprocess, "run", fake)
    root = tmp_path / "jobs"
    assert cmd.main([str(path), "--local", "--root", str(root), "--starts", str(reg_path)]) == 1
    assert len(calls) == 2 and all(c[0] == sys.executable for c in calls)
    assert cmd.main([str(path), "--local", "--root", str(root), "--starts", str(reg_path)]) == 1 and len(calls) == 2
    done = next(c for c in calls if c[c.index("--seed") + 1] == "0")
    assert done[done.index("--state") + 1] == done[done.index("--out") + 1] + "/state"
    monkeypatch.delenv("OUT", raising=False)
    with pytest.raises(SystemExit, match="--root DIR"):
        cmd.main([str(path), "--local", "--starts", str(reg_path)])


def test_local_runs_a_stopped_job_again_on_its_own_state(tmp_path, reg_path, monkeypatch, capsys):
    """Here too a stop is read from the job's run.json and not from its exit status: a job whose record says it was
    told to stop is run again, under any status, and one that then fails, or that exits with `STOPPED` and no such
    record, is failed. The stopped run's record is gone before the job runs again, so a run that writes none
    isn't read as that stop."""
    path = write_config(tmp_path / "a.toml")
    (job,) = config.load(path, starts.registry(reg_path)).jobs()
    calls = []

    def fake(argv, cwd):
        calls.append(argv)
        rc, record = ends[len(calls) - 1]
        out = Path(argv[argv.index("--out") + 1])
        assert not (out / "run.json").exists()
        if record is not None:
            (out / "run.json").write_text(record if isinstance(record, str) else json.dumps(record))
        return SimpleNamespace(returncode=rc)
    monkeypatch.setattr(cmd.subprocess, "run", fake)
    told, last = stop_record(job.seed, job.split), {"stopped": "12.0 s past the tier's 10.0 s budget"}
    ends = [(124, told), (STOPPED, told), (0, last)]
    argv = [str(path), "--local", "--root", str(tmp_path / "jobs"), "--starts", str(reg_path)]
    assert cmd.main(argv) == 3
    out = capsys.readouterr().out
    assert "1 of 1 jobs pending, 1 of them stopped" in out and "(it kept no state: it starts again from θ)" in out
    kept = tmp_path / "jobs" / job.label / "state"
    kept.mkdir(parents=True, exist_ok=True)
    (kept / "state.json").write_text("{}")
    assert cmd.main(argv) == 3 and "(its state is in state/)" in capsys.readouterr().out
    assert cmd.main(argv) == 0 and cmd.main(argv) == 0
    assert len(calls) == 3 and calls[0] == calls[1] == calls[2]

    unread = json.dumps(told)[:-1]
    for n, then in enumerate(((1, None), (STOPPED, None), (124, last), (124, unread))):
        calls.clear()
        ends = [(143, told), then]
        argv = [str(path), "--local", "--root", str(tmp_path / f"jobs-{n}"), "--starts", str(reg_path)]
        assert cmd.main(argv) == 3 and cmd.main(argv) == 1 and cmd.main(argv) == 1 and len(calls) == 2
        out, err = capsys.readouterr()
        assert f"(rc {then[0]})" in out
        # a record that doesn't read is said each time the command reads it, here twice; a job with none is not
        assert err.count("holds no run.json that reads (JSONDecodeError: ") == (2 if then[1] == unread else 0)


#: what the command prints before a grade's block
ALL_DONE = "every job has its record ({})\n"
#: the line of a config that names no honest config
NO_HONEST = ("names no honest config, so it has no slowdown: the grade divides the config's count by honest "
             'training\'s, and `honest = "NAME"` in the config names one')


def pair(where: Path, reg: Path, seeds=None) -> tuple[Path, config.Config, config.Config]:
    """An attack config of the tiny start, `a`, that names an honest config, `ref`, which trains every weight: the
    attack's path, and both as loaded on `reg`."""
    seeds = seeds or {"search": [0], "final": [1]}
    write_config(where / "ref.toml", seeds=seeds, dials={"trainable": "*", "lr": 1e-2})
    path = write_config(where / "a.toml", seeds=seeds, honest="ref")
    cfg = config.load(path, starts.registry(reg))
    return path, cfg, config.honest(cfg, starts.registry(reg))


@pytest.fixture(scope="module")
def graded(tmp_path_factory):
    """The command as a user starts it, `--local --device cpu`, through the jobs' own processes, on `pair`'s two
    configs of the tiny start on the graded registry (`synthetic.graded_registry`), twice: the two runs' processes,
    and the `slowdown.json` each left."""
    tmp = tmp_path_factory.mktemp("graded")
    reg = synthetic.graded_registry(tmp / "reg")
    path, cfg, honest = pair(tmp / "attacks", reg)
    root = tmp / "jobs"
    argv = [sys.executable, str(COMMAND), str(path), "--local", "--root", str(root), "--starts", str(reg),
            "--device", "cpu"]
    runs, wrote = [], []
    for _ in range(2):
        runs.append(subprocess.run(argv, capture_output=True, text=True))
        wrote.append((root / cfg.digest / "slowdown.json").read_bytes())
    return SimpleNamespace(reg=reg, path=path, cfg=cfg, honest=honest, root=root, runs=runs, wrote=wrote)


def block(out: str, jobs: int) -> str:
    """The grade's block the command printed: all it prints after it says every job has its record."""
    assert out.count(ALL_DONE.format(jobs)) == 1, out[-3000:]
    return out.split(ALL_DONE.format(jobs))[1]


@pytest.mark.slow
def test_local_runs_a_config_end_to_end_grades_it_and_a_second_run_runs_nothing(graded):
    """The command as a user starts it, through the jobs' own processes: what an in-process test can't show is that
    a job's process finds the route, the start and the data from its argv alone and leaves its record where the
    command looks, and that the grade reads those records as the jobs wrote them. Every job of the config and of
    its honest config done, the command grades them: its block states a slowdown as a number, since the attack and
    the honest config each reach the gain and the search record's step is measured, and `slowdown.json` is in the
    config's directory of jobs. A second run runs no job and prints and writes the same."""
    first, again = graded.runs
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-4000:]
    for job in graded.cfg.jobs() + graded.honest.jobs():
        rec = json.loads((graded.root / job.label / "run.json").read_bytes())
        assert rec["invalid"] == [] and rec["split"] == job.split and rec["label"].endswith(job.label)
        assert all(cp["e_bytes"] and cp["score"]["split"] == job.split for cp in rec["checkpoints"])
    s = json.loads(graded.wrote[0])
    assert s["schema"] == grade.SCHEMA and s["config"] == "a" and s["honest"]["config"] == "ref"
    assert s["S"]["why"] is None and s["rho"]["rho"] is not None and s["basis"]["split"] == "final"
    assert all(s["H"][k]["C"] and isinstance(s["S"][k], float) and s["S"][k] > 0 for k in ("any", "honest"))
    shown = block(first.stdout, 4)
    assert shown.startswith("SLOWDOWN a   start tiny, gain 1.1x, policy console ")
    assert f"against the honest route: S {s['S']['honest']:,.6g} = p " in shown
    assert not (graded.root / graded.honest.digest / "slowdown.json").exists()
    assert {x["source"] for x in s["records"]} == {
        str(graded.root / j.label / "run.json") for j in graded.cfg.jobs() + graded.honest.jobs()}

    assert again.returncode == 0 and again.stdout.startswith(f"a: digest {graded.cfg.digest[:16]}")
    assert block(again.stdout, 4) == shown and graded.wrote[1] == graded.wrote[0]


@pytest.mark.slow
def test_the_commands_grade_is_the_grades_own_commands_on_the_same_records(graded, tmp_path):
    """The block the command prints and the slowdown.json it writes are, byte for byte, what the grade's own
    command prints and writes handed the two directories of jobs."""
    out = tmp_path / "out"
    p = subprocess.run([sys.executable, "-m", "training_slowdown.grade", "--config", str(graded.path), "--starts",
                        str(graded.reg), "--records", str(graded.root / graded.cfg.digest), "--honest",
                        str(graded.root / graded.honest.digest), "--out", str(out)],
                       cwd=COMMAND.parent, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-4000:]
    assert block(graded.runs[0].stdout, 4) == p.stdout and p.stdout.startswith("SLOWDOWN a ")
    assert (out / "slowdown.json").read_bytes() == graded.wrote[0]


@pytest.mark.slow
def test_local_grades_the_configs_own_jobs_and_names_a_record_at_a_seed_it_doesnt_list(graded, tmp_path, capsys):
    """A config's digest leaves out its seeds, so its directory of jobs may hold a record at a seed it no longer
    lists, which would move the grade's mean over seeds: the command doesn't read it, names it, and grades as it does
    without it. The grade's own command, handed the whole directory, would read it."""
    root = tmp_path / "jobs"
    shutil.copytree(graded.root, root)
    extra = root / graded.cfg.digest / "7" / "final"
    shutil.copytree(root / graded.cfg.digest / "1" / "final", extra)
    assert cmd.main([str(graded.path), "--local", "--root", str(root), "--starts", str(graded.reg)]) == 0
    assert block(capsys.readouterr().out, 4) == (
        f"training-slowdown: not graded, at a seed or split its config doesn't list: {extra / 'run.json'}\n"
        + block(graded.runs[0].stdout, 4))
    assert (root / graded.cfg.digest / "slowdown.json").read_bytes() == graded.wrote[0].replace(
        str(graded.root).encode(), str(root).encode())
    assert [r["seed"] for r in grade.read(root / graded.cfg.digest) if r["split"] == "final"] == [1, 7]


def done(r: Research, *cfgs: config.Config) -> dict[str, str]:
    """Every job of `cfgs` done on the fake store, each with a record of its own: the records by label."""
    records = {}
    for n, j in enumerate(j for c in cfgs for j in c.jobs()):
        run, records[j.label] = f"r20980101-000000-{n:04x}", "art:" + f"{n:x}" * 64
        r.labels.append((run, config.LABEL_KEY, j.label))
        r.attempts[run] = {"status": {"rc": 0}, "outputs": {"run_record": records[j.label]}}
    return records


@pytest.mark.slow
def test_from_the_store_the_grade_reads_each_jobs_record_and_says_as_local_does(graded, tmp_path, monkeypatch,
                                                                                 capsys):
    """Without `--local` the grade reads each done job's run.json in its run's record, with the record's id as its
    source: on the records `--local` graded, the block is the same and so is `slowdown.json` but for those sources,
    written under `$OUT` at the place `--local` writes it in its root."""
    r = Research()
    records = done(r, graded.cfg, graded.honest)
    for j in graded.cfg.jobs() + graded.honest.jobs():
        r.said[records[j.label]] = (graded.root / j.label / "run.json").read_text()
    monkeypatch.setenv("OUT", str(tmp_path / "out"))
    assert command(r, str(graded.path), "--starts", str(graded.reg)) == 0
    assert block(capsys.readouterr().out, 4) == block(graded.runs[0].stdout, 4)
    assert sorted(r.fetched) == sorted(records.values())
    got, local = (json.loads(b) for b in ((tmp_path / "out" / graded.cfg.digest / "slowdown.json").read_bytes(),
                                          graded.wrote[0]))
    by_record = {v: str(graded.root / label / "run.json") for label, v in records.items()}
    assert sorted(by_record[x["source"]] for x in got["records"]) == sorted(x["source"] for x in local["records"])
    unsourced = [{**x, "source": None} for x in got["records"]], [{**x, "source": None} for x in local["records"]]
    assert unsourced[0] == unsourced[1] and {**got, "records": None} == {**local, "records": None}

    monkeypatch.delenv("OUT")
    r.fetched.clear()
    assert command(r, str(graded.path), "--starts", str(graded.reg)) == 0
    assert block(capsys.readouterr().out, 4) == block(graded.runs[0].stdout, 4) and len(r.fetched) == 4


@pytest.mark.parametrize("state, rc", [("pending", 3), ("stopped", 3), ("stalled", 4), ("failed", 1)])
def test_a_job_without_its_record_leaves_no_grade_and_fetches_no_record_for_one(state, rc, tree, reg_path, capsys):
    """While a job of the config or of its honest config is pending, stopped, stalled or failed, the command grades
    nothing: it prints no block, and fetches the record of no done job. A stopped job is started again, and is then
    pending."""
    path, cfg, honest = pair(tree, reg_path)
    r = Research()
    records = done(r, cfg, honest)
    job = honest.jobs()[-1]
    r.labels = [x for x in r.labels if x[2] != job.label]
    command(r, str(path), "--starts", str(reg_path))
    if state == "failed":
        r.attempts[last_run(r, job)] = {"status": {"rc": 1}, "outputs": {}}
    elif state in ("stopped", "stalled"):
        for _ in range(cmd.STALLS - 1 if state == "stalled" else 0):
            stop(r, job, "art:" + "f" * 64)
            command(r, str(path), "--starts", str(reg_path))
        stop(r, job, "art:" + "f" * 64)
    capsys.readouterr()
    assert command(r, str(path), "--starts", str(reg_path)) == rc
    out = capsys.readouterr().out
    shown = {"stopped": f"pending  {last_run(r, job)} (continues "}.get(state, f"{state:8} {last_run(r, job)}")
    assert f"  {job.split:6} {job.seed:<5} {shown}" in out
    assert "SLOWDOWN" not in out and "every job has its record" not in out
    assert not set(r.fetched) & set(records.values())


def done_locally(root: Path, *cfgs: config.Config) -> None:
    """Every job of `cfgs` done under `root` as `--local` leaves one, its record an empty one."""
    for j in (j for c in cfgs for j in c.jobs()):
        (root / j.label).mkdir(parents=True)
        (root / j.label / "job.json").write_text('{"rc": 0}\n')
        (root / j.label / "run.json").write_text("{}\n")


def test_a_grade_that_refuses_says_why_and_exits_with_its_own_status_and_a_dry_run_grades_nothing(tmp_path, reg_path,
                                                                                               capsys):
    """Every job done, a grade that refuses the records is the command's own status, `REFUSED`, which no job's
    state gives, with the grade's reason: here the tiny registry's one start on its data, which gives the grade's fit
    no frontier; from the store, a done job whose record holds no run.json that reads. A dry run of the same grades
    nothing and reads no record: it exits as one with every record does."""
    path, cfg, honest = pair(tmp_path / "attacks", reg_path)
    root = tmp_path / "jobs"
    done_locally(root, cfg, honest)
    local = [str(path), "--local", "--root", str(root), "--starts", str(reg_path)]
    assert cmd.REFUSED == 5 and cmd.REFUSED not in (0, 1, 3, 4)
    assert cmd.main(local) == cmd.REFUSED
    out = capsys.readouterr().out
    assert out.endswith("training-slowdown: no slowdown, the grade refuses the records: grade: the 1 starts on data "
                        "'tiny' give no fit: it takes 3 frontier points spanning a factor 10 in C0\n")
    assert ALL_DONE.format(4) in out and not (root / cfg.digest / "slowdown.json").exists()

    assert cmd.main(local + ["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.endswith(ALL_DONE.format(4)) and "would run" not in out
    r = Research()
    done(r, cfg, honest)
    assert command(r, str(path), "--starts", str(reg_path), "--dry-run") == 0
    assert capsys.readouterr().out.endswith(ALL_DONE.format(4)) and r.fetched == []
    assert command(r, str(path), "--starts", str(reg_path)) == cmd.REFUSED and len(r.fetched) == 1
    assert "holds no run.json that reads" in capsys.readouterr().out


def test_a_config_that_names_no_honest_config_has_no_grade(tmp_path, reg_path, capsys):
    """The grade divides a config's count by honest training's: a config that names no honest config is never graded
    against itself, and the command says how one is named and exits as one whose every job has its record."""
    path = write_config(tmp_path / "a.toml", seeds={"search": [0], "final": [1]})
    root = tmp_path / "jobs"
    done_locally(root, config.load(path, starts.registry(reg_path)))
    assert cmd.main([str(path), "--local", "--root", str(root), "--starts", str(reg_path)]) == 0
    assert capsys.readouterr().out.endswith(ALL_DONE.format(2) + f"a {NO_HONEST}\n")
    assert list(root.glob("*/slowdown.json")) == []
