"""`research queue drive --config PATH`: drive's flags of record (`DriveFlags`), read fresh each tick from PATH at `main` in the
queue's clone, so a change to them is a reviewed commit and the loop running drive needs no restart.  A flag the command line
gives overrides the file's; without --config nothing changes."""
from __future__ import annotations

from pathlib import Path

import pytest

import research.drive as D
import research.queue as Q
from research.queue import DRIVE_FLAGS, DriveFlags, QueueError
from test_queue import World, _git

FLAGS = """on = ["m1", "m2"]
marked_after = "2026-10-12T06:00:00Z"
regenerate_locks = true
first = [1858]
land = "review"
after_landing_no_cache = true
landing_gates = ["gates/g.py"]
max_lean_checks = 3
nonlean_first = false
reviewed_first = false
stale_days = 3
stale_close_days = 7
shards = 2
"""


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


@pytest.fixture
def driven(monkeypatch) -> list[dict]:
    """Each Driver the command builds, as its queue's `first` and the flags it was given; its tick does nothing."""
    built: list[dict] = []

    class Driver:
        def __init__(self, q, **kw):
            assert callable(kw.pop("reviews")) and callable(kw.pop("ready_by"))
            built.append({"first": q.first, "nonlean_first": q.nonlean_first, "reviewed_first": q.reviewed_first, **kw})

        def tick(self, dry_run=False):
            return ""

    monkeypatch.setattr(D, "Driver", Driver)
    return built


def drive(world: World, *argv: str) -> int:
    return Q.main(["drive", "--workdir", str(world.tmp / "q" / "verity"), "--remote", str(world.origin), "--repo", "o/r",
                   "--store", str(world.tmp / "store"), *argv])


def on_main(world: World, text: str) -> str:
    sha = world.change({DRIVE_FLAGS: text})
    _git(world.dev, "push", "-q", "origin", f"{sha}:refs/heads/main")
    return sha


def test_the_flags_parse_with_their_types_and_absent_keys_take_the_commands_defaults():
    f = DriveFlags.parse(FLAGS)
    assert f == DriveFlags(on="m1,m2", marked_after="2026-10-12T06:00:00Z", regenerate_locks=True, first=(1858,),
                           land="review", after_landing_no_cache=True,
                           landing_gates=("gates/g.py",), max_lean_checks=3, nonlean_first=False, reviewed_first=False,
                           stale_days=3, stale_close_days=7, shards=2)
    assert DriveFlags.parse("") == DriveFlags() == DriveFlags(None, None, False, (), "review", False)
    assert f.line() == ("on=m1,m2 marked_after=2026-10-12T06:00:00Z regenerate_locks=True first=1858 land=review "
                        "after_landing_no_cache=True landing_gates=gates/g.py max_lean_checks=3 nonlean_first=False reviewed_first=False stale_days=3 "
                        "stale_close_days=7 stale_keep= shards=2")
    for text, why in (('on = "m1,m2"', 'on = \'m1,m2\': a list of names, as ["a", "b"]'),
                      ("on = []", "on = []: at least one machine"),
                      ("max_lean_checks = 0", "max_lean_checks = 0: a whole number, 1 or more"),
                      ("shards = 0", "shards = 0: a whole number, 1 or more"),
                      ('shards = "5"', "shards = '5': a whole number, 1 or more"),
                      ('marked_after = "12 Oct"', "marked_after = '12 Oct': a UTC time as YYYY-MM-DDTHH:MM:SSZ"),
                      ('regenerate_locks = "yes"', "regenerate_locks = 'yes': true or false"),
                      ("nonlean_first = 1", "nonlean_first = 1: true or false"),
                      ("reviewed_first = 1", "reviewed_first = 1: true or false"),
                      ('first = ["1858"]', "first = ['1858']: a list of PR numbers"),
                      ('land = "fast"', "land = 'fast': one of review"),
                      ('land = "check"', "land = 'check': train landing was deleted with its code (#5070): drive lands on "
                                         "review only"),
                      ("after_landing_no_cache = 1", "after_landing_no_cache = 1: true or false"),
                      ('landing_gates = "g.py"', "landing_gates = 'g.py': a list of Python scripts by their path"),
                      ('landing_gates = ["../g.py"]', "landing_gates = ['../g.py']: a list of Python scripts"),
                      ('landing_gates = ["/g.py"]', "landing_gates = ['/g.py']: a list of Python scripts"),
                      ('landing_gates = ["g.sh"]', "landing_gates = ['g.sh']: a list of Python scripts"),
                      ("stale_days = -1", "stale_days = -1: a whole number of days, 0 for off"),
                      ("stale_days = 3.5", "stale_days = 3.5: a whole number of days"),
                      ("stale_close_days = 7", "stale_close_days = 7: 0, or at least stale_days, which must be on"),
                      ("stale_days = 7\nstale_close_days = 3", "stale_close_days = 3: 0, or at least stale_days"),
                      ("on = [", "not TOML")):
        with pytest.raises(QueueError, match="^drive.toml: ") as e:
            DriveFlags.parse(text)
        assert why in str(e.value)


def test_drive_refuses_an_unknown_key_naming_the_keys_it_reads_and_the_one_it_likely_meant():
    with pytest.raises(QueueError) as e:
        DriveFlags.parse("max_lean_check = 4\n", "d.toml", strict=True)
    assert str(e.value) == ("d.toml: unknown key 'max_lean_check' (did you mean 'max_lean_checks'?): drive reads only on, "
                            "marked_after, regenerate_locks, first, land, after_landing_no_cache, landing_gates, max_lean_checks, "
                            "nonlean_first, reviewed_first, stale_days, stale_close_days, stale_keep, shards; rename or remove it, or "
                            "update this research")


def test_a_client_ignores_every_key_it_doesnt_read_a_near_miss_too_so_it_parses_a_newer_mains_flags(capsys):
    assert DriveFlags.parse("max_running = 4\nmax_lean_slots = 2\nland = \"review\"\n", "d.toml").land == "review"
    err = capsys.readouterr().err
    assert "d.toml: ignoring key 'max_running'" in err and "d.toml: ignoring key 'max_lean_slots'" in err


def test_drive_reads_the_flags_at_main_in_its_clone_never_the_callers_tree_and_takes_a_new_main_without_a_restart(world, driven, monkeypatch, capsys):
    """The caller's working tree (here `dev`, checked out with other flags) is whatever its VM last pulled; drive reads main's."""
    sha = on_main(world, FLAGS)
    monkeypatch.chdir(world.dev)
    (world.dev / DRIVE_FLAGS).write_text('on = ["stale"]\nmax_lean_checks = 1\n')
    assert drive(world, "--config", DRIVE_FLAGS) == 0
    assert driven[-1] == {"first": (1858,), "nonlean_first": False, "reviewed_first": False, "on": "m1,m2", "regenerate_locks": True,
                          "after_landing_no_cache": True, "max_lean_checks": 3, "stale_days": 3, "stale_close_days": 7, "stale_keep": (),
                          "shards": 2}
    lines = [x for x in capsys.readouterr().out.splitlines() if "flags from" in x]
    assert lines == [f"research queue drive: flags from {DRIVE_FLAGS} at main {sha[:12]}: on=m1,m2 "
                     "marked_after=2026-10-12T06:00:00Z regenerate_locks=True first=1858 land=review "
                     "after_landing_no_cache=True landing_gates=gates/g.py max_lean_checks=3 nonlean_first=False reviewed_first=False stale_days=3 "
                     "stale_close_days=7 stale_keep= shards=2"]

    on_main(world, FLAGS.replace("max_lean_checks = 3", "max_lean_checks = 6"))
    assert drive(world, "--config", DRIVE_FLAGS) == 0
    assert driven[-1]["max_lean_checks"] == 6

    (world.dev / DRIVE_FLAGS).write_text('on = ["stale"]\nmax_lean_checks = 1\n')
    assert drive(world, "--config", DRIVE_FLAGS, "--config-from-tree") == 0
    assert driven[-1] == {"first": (), "nonlean_first": True, "reviewed_first": True, "on": "stale", "regenerate_locks": False,
                          "after_landing_no_cache": False, "max_lean_checks": 1, "stale_days": 0,
                          "stale_close_days": 0, "stale_keep": (), "shards": None}
    assert f"flags from {DRIVE_FLAGS}: on=stale " in capsys.readouterr().out


def test_a_flag_given_on_the_command_line_overrides_the_files(world, driven, monkeypatch):
    on_main(world, FLAGS)
    monkeypatch.chdir(world.dev)
    assert drive(world, "--config", DRIVE_FLAGS, "--on", "m9", "--no-regenerate-locks", "--first", "7,8",
                 "--no-after-landing-no-cache") == 0
    assert driven[-1] == {"first": (7, 8), "nonlean_first": False, "reviewed_first": False, "on": "m9", "regenerate_locks": False,
                          "after_landing_no_cache": False, "max_lean_checks": 3, "stale_days": 3, "stale_close_days": 7, "stale_keep": (),
                          "shards": 2}


def test_without_config_drive_takes_its_flags_as_before(world, driven, monkeypatch, capsys):
    rules = world.tmp / "rules.toml"
    rules.write_text('hold_label = "hold"\nmarked_after = "2026-10-12T06:00:00Z"\n')
    monkeypatch.chdir(world.dev)
    assert drive(world, "--on", "m1", "--rules", str(rules)) == 0
    assert driven[-1] == {"first": (), "nonlean_first": None, "reviewed_first": None, "on": "m1", "regenerate_locks": False,
                          "after_landing_no_cache": False, "max_lean_checks": 2, "stale_days": 0,
                          "stale_close_days": 0, "stale_keep": (), "shards": None}
    assert drive(world, "--on", "m1", "--rules", str(rules), "--regenerate-locks", "--first", "5") == 0
    assert driven[-1] == {"first": (5,), "nonlean_first": None, "reviewed_first": None, "on": "m1", "regenerate_locks": True,
                          "after_landing_no_cache": False, "max_lean_checks": 2, "stale_days": 0, "stale_close_days": 0, "stale_keep": (),
                          "shards": None}
    assert "flags from" not in capsys.readouterr().out


def test_the_deleted_train_flags_parse_and_are_ignored_with_a_note(world, driven, monkeypatch, capsys):
    """Drive's own args on the control pod may still pass `--max-running 10`: refusing it would stop drive."""
    rules = world.tmp / "rules.toml"
    rules.write_text('hold_label = "hold"\nmarked_after = "2026-10-12T06:00:00Z"\n')
    monkeypatch.chdir(world.dev)
    assert drive(world, "--on", "m1", "--rules", str(rules)) == 0
    plain = driven[-1]
    assert "ignored" not in capsys.readouterr().out
    assert drive(world, "--on", "m1", "--rules", str(rules), "--max-running", "10", "--train-size", "4") == 0
    assert driven[-1] == plain
    assert ("research queue drive: --max-running and --train-size ignored: train landing was deleted (#5070); drop them from "
            "drive's args") in capsys.readouterr().out
    assert drive(world, "--on", "m1", "--rules", str(rules), "--max-running", "10") == 0
    assert driven[-1] == plain and "drive: --max-running ignored: " in capsys.readouterr().out


def test_config_refusals_name_the_fix(world, driven, monkeypatch, capsys):
    monkeypatch.chdir(world.dev)
    assert drive(world, "--config", DRIVE_FLAGS) == 1
    assert f"--config {DRIVE_FLAGS}: `main` (" in (err := capsys.readouterr().err) and "has no such file: land it first" in err
    on_main(world, 'max_lean_checks = 4\n')
    assert drive(world, "--config", DRIVE_FLAGS) == 1
    assert "drive needs a machine for the checks it launches: set `on` in" in capsys.readouterr().err
    on_main(world, 'on = ["m1"]\nmax_lean_check = 4\n')
    assert drive(world, "--config", DRIVE_FLAGS) == 1
    assert "unknown key 'max_lean_check' (did you mean 'max_lean_checks'?)" in capsys.readouterr().err
    assert drive(world, "--config", str(world.dev / DRIVE_FLAGS)) == 1
    assert "a path from the repository's root, read at `main` (or --config-from-tree" in capsys.readouterr().err
    assert driven == []
    for argv, why in ((["sync", "--config", DRIVE_FLAGS], "--config and --config-from-tree are for drive only"),
                      (["drive", "--on", "m1", "--config-from-tree"], "--config-from-tree needs --config PATH")):
        with pytest.raises(SystemExit) as e:
            Q.main(argv)
        assert e.value.code == 2 and why in capsys.readouterr().err


def test_the_gates_are_read_whatever_other_keys_drive_toml_holds(tmp_path):
    """A branch's older queue.py reads main's drive.toml for the preflight's gates; a key added since doesn't stop it."""
    from research.queue import gates_at

    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / DRIVE_FLAGS).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / DRIVE_FLAGS).write_text('landing_gates = ["gates/g.py"]\nmax_lean_checks_of_the_future = 9\n')
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
    assert gates_at(tmp_path, "main") == ("gates/g.py",)
    (tmp_path / DRIVE_FLAGS).write_text('landing_gates = "gates/g.py"\n')
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "y")
    with pytest.raises(QueueError, match="landing_gates"):
        gates_at(tmp_path, "main")
