"""`research run` takes its run dir by making it (`runs.take_run_dir`): a fresh id whose dir exists, which two runs in one
second pick once in 65,536 pairs, is picked again a bounded number of times; an id the caller gives whose dir exists is refused
as before. Each test fixes the id source (`runs.new_run_id`), so none depends on the clock or on chance."""
from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path

from research import cli, remote
from research import runs as R

A, B = "r20000101-000000-aaaa", "r20000101-000000-bbbb"
GIVEN = "research run: {} already exists; a rerun gets a fresh id"


def _ids(monkeypatch, *ids: str) -> list[str]:
    """Fix the id source to return ``ids`` in order (the last one ever after); returns the list of ids it handed out. Asked for
    more than one past the bound, it raises, so a lost bound fails the test rather than hanging it."""
    handed: list[str] = []

    def new_run_id(now=None):
        if len(handed) > R.FRESH_ID_TRIES:
            raise AssertionError(f"picked {len(handed) + 1} fresh ids for one run, past FRESH_ID_TRIES = {R.FRESH_ID_TRIES}")
        handed.append(ids[min(len(handed), len(ids) - 1)])
        return handed[-1]

    monkeypatch.setattr(R, "new_run_id", new_run_id)
    return handed


def _run(runs: Path, *flags: str) -> int:
    return cli.main(["run", "--runs-dir", str(runs), "--no-sampler", *flags, "--", sys.executable, "-c", "pass"])


def test_two_runs_that_pick_the_same_fresh_id_both_start_with_different_ids(tmp_path, monkeypatch, capsys):
    handed = _ids(monkeypatch, A, A, B)
    runs = tmp_path / "runs"
    assert _run(runs) == 0 and _run(runs) == 0
    assert handed == [A, A, B], "the second run met the first's dir and picked again"
    assert sorted(d.name for d in runs.iterdir()) == sorted([A, B])
    assert {json.loads((runs / i / "job.json").read_text())["run"]["id"] for i in (A, B)} == {A, B}
    assert "already exists" not in capsys.readouterr().err


def test_the_claim_is_the_make_itself_so_a_dir_made_after_a_check_still_counts(tmp_path, monkeypatch):
    """Another process can make A's dir between a check and a make. Here every check says A's dir is not there while it is,
    so only a take that claims by its `mkdir` alone still picks again."""
    handed = _ids(monkeypatch, A, B)
    runs = tmp_path / "runs"
    (runs / A).mkdir(parents=True)
    real_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda p, *a, **kw: False if p == runs / A else real_exists(p, *a, **kw))
    assert R.take_run_dir(runs) == runs / B and handed == [A, B]


def test_an_id_the_caller_gives_whose_dir_exists_is_refused_here_on_a_machine_and_on_the_queue(tmp_path, monkeypatch, capsys):
    handed = _ids(monkeypatch, A, B)
    runs = tmp_path / "runs"
    assert _run(runs) == 0
    job = (runs / A / "job.json").read_text()
    capsys.readouterr()
    launched, placed = [], []
    monkeypatch.setattr(remote, "launch", lambda **kw: launched.append(kw))
    monkeypatch.setattr(remote, "queue_placement", lambda *a: placed.append(a))
    for where in ([], ["--on", "m", "--project", "p", "--no-custody-r2"],
                  ["--queue", "--project", "p", "--source", str(tmp_path), "--max-min", "1"]):
        assert _run(runs, "--id", A, *where) == 2, where
        assert GIVEN.format(runs / A) in capsys.readouterr().err, where
    assert handed == [A], "a refused --id picks no fresh id in its place"
    assert launched == placed == [] and (runs / A / "job.json").read_text() == job, "nothing launched, the first run's dir untouched"


def test_fresh_ids_are_picked_a_bounded_number_of_times_then_the_run_is_refused(tmp_path, monkeypatch, capsys):
    handed = _ids(monkeypatch, A)
    runs = tmp_path / "runs"
    (runs / A).mkdir(parents=True)
    assert _run(runs) == 2
    err = capsys.readouterr().err
    assert (f"research run: refused: {R.FRESH_ID_TRIES} fresh run ids in a row each named a run dir that already exists under "
            f"{runs} ({A}), so nothing ran") in err
    assert len(handed) == R.FRESH_ID_TRIES and [d.name for d in runs.iterdir()] == [A]


def test_a_launch_gets_its_fresh_dir_made_and_empty_and_a_refused_one_leaves_none(tmp_path, monkeypatch, capsys):
    """`--on` and `--queue` take the dir as a local run does: the queue books the job under the id it took, the launch writes
    into that dir, and a run refused after the take leaves no dir behind."""
    _ids(monkeypatch, A, A, B)
    runs = tmp_path / "runs"
    (runs / A).mkdir(parents=True)
    seen = []

    def launch(**kw):
        seen.append((kw["run_id"], kw["local_dir"], kw["taken"], kw["local_dir"].is_dir() and not any(kw["local_dir"].iterdir())))
        return {"phase": "launched", "pod_id": "p1", "remote_run_dir": "/r", "launch": {"runner_pid": 1}}

    def place(source, run_id, spec):
        seen.append(run_id)
        raise remote.Refused("placement refused")

    monkeypatch.setattr(remote, "launch", launch)
    monkeypatch.setattr(remote, "queue_placement", place)
    assert _run(runs, "--on", "m", "--project", "p", "--no-custody-r2") == 0
    assert seen == [(B, runs / B, True, True)]
    assert _run(runs, "--queue", "--project", "p", "--source", str(tmp_path), "--max-min", "1") == 3
    assert seen[1] == B and "placement refused" in capsys.readouterr().err
    assert [d.name for d in runs.iterdir()] == [A], "neither run wrote in its dir (the launch is a stand-in), so neither stays"


#: a launcher held inside its run's window: its dir taken, nothing written in it yet (a stand-in `_prepare_store` says so on
#: stdout, then waits on stdin)
HELD = f"""
import sys
from research import cli
from research import runs as R

R.new_run_id = lambda now=None: {A!r}

def held(*a, **kw):
    print("held", flush=True)
    sys.stdin.read()

cli._prepare_store = held
sys.exit(cli.main(["run", "--runs-dir", sys.argv[1], "--no-sampler", "--", sys.executable, "-c", "pass"]))
"""


def test_a_launcher_stopped_by_sigterm_inside_the_window_leaves_no_dir(tmp_path):
    runs = tmp_path / "runs"
    child = subprocess.Popen([sys.executable, "-c", HELD, str(runs)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    assert child.stdout.readline() == "held\n", child.communicate()[1][-2000:]
    assert [d.name for d in runs.iterdir()] == [A], "the dir is taken before the store is prepared"
    child.send_signal(signal.SIGTERM)
    _, err = child.communicate()
    assert child.returncode == 128 + signal.SIGTERM, err[-2000:]
    assert list(runs.iterdir()) == [], "the taken dir held nothing, so it is removed"


def test_a_run_takes_sigterm_as_its_exit_while_it_runs_and_puts_the_handler_before_back(tmp_path, monkeypatch):
    def mine(signum, frame):
        raise AssertionError("not delivered here")

    inside = []
    prepare = cli._prepare_store
    monkeypatch.setattr(cli, "_prepare_store", lambda *a, **kw: inside.append(signal.getsignal(signal.SIGTERM)) or prepare(*a, **kw))
    before = signal.signal(signal.SIGTERM, mine)
    try:
        assert _run(tmp_path / "runs") == 0
        assert signal.getsignal(signal.SIGTERM) is mine
    finally:
        signal.signal(signal.SIGTERM, before)
    assert inside == [cli._exit_on_sigterm]
