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
max_running = 4
train_size = 15
marked_after = "2026-10-12T06:00:00Z"
defer = ["circuit-check", "pytest"]
regenerate_locks = true
first = [1858]
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
            built.append({"first": q.first, **kw})

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
    assert f == DriveFlags(on="m1,m2", max_running=4, train_size=15, marked_after="2026-10-12T06:00:00Z",
                           defer=("circuit-check", "pytest"), regenerate_locks=True, first=(1858,))
    assert DriveFlags.parse("") == DriveFlags() == DriveFlags(None, 2, 4, None, (), False, ())
    assert f.line() == ("on=m1,m2 max_running=4 train_size=15 marked_after=2026-10-12T06:00:00Z defer=circuit-check,pytest "
                        "regenerate_locks=True first=1858")
    for text, why in (('on = "m1,m2"', 'on = \'m1,m2\': a list of names, as ["a", "b"]'),
                      ("on = []", "on = []: at least one machine"),
                      ("max_running = true", "max_running = True: a whole number, 1 or more"),
                      ("train_size = 0", "train_size = 0: a whole number, 1 or more"),
                      ('marked_after = "12 Oct"', "marked_after = '12 Oct': a UTC time as YYYY-MM-DDTHH:MM:SSZ"),
                      ('defer = "pytest"', "defer = 'pytest': a list of names"),
                      ('regenerate_locks = "yes"', "regenerate_locks = 'yes': true or false"),
                      ('first = ["1858"]', "first = ['1858']: a list of PR numbers"),
                      ("on = [", "not TOML")):
        with pytest.raises(QueueError, match="^drive.toml: ") as e:
            DriveFlags.parse(text)
        assert why in str(e.value)


def test_an_unknown_key_is_refused_naming_the_keys_drive_reads_and_the_one_it_likely_meant():
    with pytest.raises(QueueError) as e:
        DriveFlags.parse("max_runing = 4\n", "d.toml")
    assert str(e.value) == ("d.toml: unknown key 'max_runing' (did you mean 'max_running'?): drive reads only on, max_running, "
                            "train_size, marked_after, defer, regenerate_locks, first; rename or remove it")
    with pytest.raises(QueueError, match="unknown key 'shadow': drive reads only on, "):
        DriveFlags.parse("shadow = true\n")


def test_drive_reads_the_flags_at_main_in_its_clone_never_the_callers_tree_and_takes_a_new_main_without_a_restart(world, driven, monkeypatch, capsys):
    """The caller's working tree (here `dev`, checked out with other flags) is whatever its VM last pulled; drive reads main's."""
    sha = on_main(world, FLAGS)
    monkeypatch.chdir(world.dev)
    (world.dev / DRIVE_FLAGS).write_text('on = ["stale"]\ntrain_size = 1\n')
    assert drive(world, "--config", DRIVE_FLAGS) == 0
    assert driven[-1] == {"first": (1858,), "on": "m1,m2", "max_running": 4, "train_size": 15,
                          "defer": frozenset({"circuit-check", "pytest"}), "regenerate_locks": True}
    lines = [x for x in capsys.readouterr().out.splitlines() if "flags from" in x]
    assert lines == [f"research queue drive: flags from {DRIVE_FLAGS} at main {sha[:12]}: on=m1,m2 max_running=4 train_size=15 "
                     "marked_after=2026-10-12T06:00:00Z defer=circuit-check,pytest regenerate_locks=True first=1858"]

    on_main(world, FLAGS.replace("train_size = 15", "train_size = 6"))
    assert drive(world, "--config", DRIVE_FLAGS) == 0
    assert driven[-1]["train_size"] == 6

    (world.dev / DRIVE_FLAGS).write_text('on = ["stale"]\ntrain_size = 1\n')
    assert drive(world, "--config", DRIVE_FLAGS, "--config-from-tree") == 0
    assert driven[-1] == {"first": (), "on": "stale", "max_running": 2, "train_size": 1, "defer": frozenset(), "regenerate_locks": False}
    assert f"flags from {DRIVE_FLAGS}: on=stale " in capsys.readouterr().out


def test_a_flag_given_on_the_command_line_overrides_the_files(world, driven, monkeypatch):
    on_main(world, FLAGS)
    monkeypatch.chdir(world.dev)
    assert drive(world, "--config", DRIVE_FLAGS, "--on", "m9", "--max-running", "1", "--defer", "", "--no-regenerate-locks",
                 "--first", "7,8") == 0
    assert driven[-1] == {"first": (7, 8), "on": "m9", "max_running": 1, "train_size": 15, "defer": frozenset(),
                          "regenerate_locks": False}


def test_without_config_drive_takes_its_flags_as_before(world, driven, monkeypatch, capsys):
    rules = world.tmp / "rules.toml"
    rules.write_text('hold_label = "hold"\nmarked_after = "2026-10-12T06:00:00Z"\n')
    monkeypatch.chdir(world.dev)
    assert drive(world, "--on", "m1", "--rules", str(rules)) == 0
    assert driven[-1] == {"first": (), "on": "m1", "max_running": 2, "train_size": 4, "defer": frozenset(), "regenerate_locks": False}
    assert drive(world, "--on", "m1", "--rules", str(rules), "--max-running", "4", "--train-size", "15", "--defer",
                 "circuit-check,pytest", "--regenerate-locks", "--first", "5") == 0
    assert driven[-1] == {"first": (5,), "on": "m1", "max_running": 4, "train_size": 15,
                          "defer": frozenset({"circuit-check", "pytest"}), "regenerate_locks": True}
    assert "flags from" not in capsys.readouterr().out


def test_config_refusals_name_the_fix(world, driven, monkeypatch, capsys):
    monkeypatch.chdir(world.dev)
    assert drive(world, "--config", DRIVE_FLAGS) == 1
    assert f"--config {DRIVE_FLAGS}: `main` (" in (err := capsys.readouterr().err) and "has no such file: land it first" in err
    on_main(world, 'max_running = 4\n')
    assert drive(world, "--config", DRIVE_FLAGS) == 1
    assert "drive needs a machine for the checks it launches: set `on` in" in capsys.readouterr().err
    on_main(world, 'on = ["m1"]\nmax_runing = 4\n')
    assert drive(world, "--config", DRIVE_FLAGS) == 1
    assert "unknown key 'max_runing' (did you mean 'max_running'?)" in capsys.readouterr().err
    assert drive(world, "--config", str(world.dev / DRIVE_FLAGS)) == 1
    assert "a path from the repository's root, read at `main` (or --config-from-tree" in capsys.readouterr().err
    assert driven == []
    for argv, why in ((["sync", "--config", DRIVE_FLAGS], "--config and --config-from-tree are for drive only"),
                      (["drive", "--on", "m1", "--config-from-tree"], "--config-from-tree needs --config PATH")):
        with pytest.raises(SystemExit) as e:
            Q.main(argv)
        assert e.value.code == 2 and why in capsys.readouterr().err
