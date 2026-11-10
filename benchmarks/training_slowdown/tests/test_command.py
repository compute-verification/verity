import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from synthetic import tiny_registry, write_config

from training_slowdown import config, starts

COMMAND = Path(__file__).resolve().parents[1] / "run.py"
_spec = importlib.util.spec_from_file_location("training_slowdown_command", COMMAND)
cmd = sys.modules[_spec.name] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmd)

ART = "art:" + "e" * 64


class Research:
    """`research` as the command calls it: labels, attempts and the runs it starts, all in memory."""

    def __init__(self):
        self.labels: list[tuple[str, str, str]] = []
        self.attempts: dict[str, dict] = {}
        self.runs: list[list[str]] = []
        self.sync_rc = 0

    def __call__(self, argv):
        assert argv[0] == "research", argv
        if argv[1] == "run":
            self.runs.append(argv)
            return SimpleNamespace(returncode=0, stdout=f"queued r20990101-000000-{len(self.runs):04x}\n", stderr="")
        verb = argv[2]
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
        assert argv[-3:-1] == ["bash", "-lc"] and script.startswith(cmd.SYNC + " && cd benchmarks/training_slowdown")
        assert (f"--config attacks/a.toml --seed {job.seed} --split {job.split} --starts {reg_path.resolve()}"
                in script and script.endswith('--out "$OUT"'))
        assert (r.run_of(job.label), config.LABEL_KEY, job.label) in r.labels
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


def test_the_honest_configs_jobs_come_with_it_and_on_names_the_machine(tree, reg_path):
    write_config(tree / "ref.toml", dials={"trainable": "*"})
    write_config(tree / "a.toml", honest="ref")
    r = Research()
    assert command(r, "a", "--starts", str(reg_path), "--on", "node-1", "--pool", "p") == 3
    assert [argv[2:6] for argv in r.runs] == [["--on", "node-1", "--pool", "p"]] * 2
    assert any("attacks/ref.toml" in argv[-1] for argv in r.runs)


def test_a_dry_run_starts_and_labels_nothing(tree, reg_path, capsys):
    write_config(tree / "a.toml")
    r = Research()
    r.sync_rc = 1
    assert command(r, "a", "--starts", str(reg_path), "--dry-run") == 3
    assert r.runs == [] and r.labels == []
    assert "would run: research run --queue" in capsys.readouterr().out


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
    monkeypatch.delenv("OUT", raising=False)
    with pytest.raises(SystemExit, match="--root DIR"):
        cmd.main([str(path), "--local", "--starts", str(reg_path)])


@pytest.mark.slow
def test_local_runs_a_config_end_to_end_and_a_second_run_runs_nothing(tmp_path, reg_path):
    """The command as a user starts it, through the jobs' own processes: what an in-process test can't show is that
    a job's process finds the route, the start and the data from its argv alone and leaves its record where the
    command looks."""
    path = write_config(tmp_path / "a.toml", seeds={"search": [0], "final": [1]})
    cfg = config.load(path, starts.registry(reg_path))
    root = tmp_path / "jobs"
    argv = [sys.executable, str(COMMAND), str(path), "--local", "--root", str(root), "--starts", str(reg_path),
            "--device", "cpu"]
    p = subprocess.run(argv, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-4000:]
    recs = {}
    for job in cfg.jobs():
        raw = (root / job.label / "run.json").read_bytes()
        rec = recs[job.label] = json.loads(raw)
        assert rec["invalid"] == [] and rec["split"] == job.split and rec["label"].endswith(job.label)
        assert all(cp["e_bytes"] and cp["score"]["split"] == job.split for cp in rec["checkpoints"])
    again = subprocess.run(argv, capture_output=True, text=True)
    assert again.returncode == 0 and "every job has its record (2)" in again.stdout
    assert all(json.loads((root / label / "run.json").read_bytes()) == rec for label, rec in recs.items())
