"""Drive's flags (`DriveFlags`): tools/verity/check/drive.toml at `main` in the queue's clone, read fresh each tick, their one
source, so a change to them is a reviewed commit and the loop running drive needs no restart.  A `--config FILE` is refused,
naming each key, unless every key it sets has main's value; a command-line flag overrides its key only where drive.toml's
`command_line` lets it.  Before it syncs, a tick refuses a name its flags use that doesn't exist (`drive_names`)."""
from __future__ import annotations

import subprocess
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
"""


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


@pytest.fixture(autouse=True)
def machines(monkeypatch) -> tuple[set[str], dict[str, str]]:
    """The machine registry drive resolves `on` against (`drive_machines`): the names it knows, and those it refuses with why,
    which a test may change."""
    known, refused = {"m1", "m2", "m9"}, {}
    monkeypatch.setattr(Q, "drive_machines", lambda: (known, refused))
    return known, refused


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
    sha = world.change({DRIVE_FLAGS: text, "gates/g.py": "import sys\nsys.exit(0)\n"})
    _git(world.dev, "push", "-q", "origin", f"{sha}:refs/heads/main")
    return sha


def test_the_flags_parse_with_their_types_and_absent_keys_take_the_commands_defaults():
    f = DriveFlags.parse(FLAGS)
    assert f == DriveFlags(on="m1,m2", marked_after="2026-10-12T06:00:00Z", regenerate_locks=True, first=(1858,),
                           land="review", after_landing_no_cache=True,
                           landing_gates=("gates/g.py",), max_lean_checks=3, nonlean_first=False, reviewed_first=False,
                           stale_days=3, stale_close_days=7)
    assert DriveFlags.parse("") == DriveFlags() == DriveFlags(None, None, False, (), "review", None, False)
    assert DriveFlags.parse("after_landing_shards = 3\n").after_landing_shards == 3
    assert f.line() == ("on=m1,m2 marked_after=2026-10-12T06:00:00Z regenerate_locks=True first=1858 land=review "
                        "land_into=None after_landing_no_cache=True landing_gates=gates/g.py max_lean_checks=3 nonlean_first=False reviewed_first=False stale_days=3 "
                        "stale_close_days=7 stale_keep= after_landing_shards=1 command_line=on,first")
    for text, why in (('on = "m1,m2"', 'on = \'m1,m2\': a list of names, as ["a", "b"]'),
                      ("on = []", "on = []: at least one machine"),
                      ("max_lean_checks = 0", "max_lean_checks = 0: a whole number, 1 or more"),
                      ("after_landing_shards = 0", "after_landing_shards = 0: a whole number, 1 or a shard plan's N"),
                      ('marked_after = "12 Oct"', "marked_after = '12 Oct': a UTC time as YYYY-MM-DDTHH:MM:SSZ"),
                      ('regenerate_locks = "yes"', "regenerate_locks = 'yes': true or false"),
                      ("nonlean_first = 1", "nonlean_first = 1: true or false"),
                      ("reviewed_first = 1", "reviewed_first = 1: true or false"),
                      ('first = ["1858"]', "first = ['1858']: a list of PR numbers"),
                      ('land = "fast"', "land = 'fast': one of review"),
                      ('land = "check"', "land = 'check': train landing was deleted with its code (#5070): drive lands on "
                                         "review only"),
                      ('land_into = "next"', "land_into = 'next': \"checked\" (main follows it to fully checked commits) or "
                                             "\"main\""),
                      ("after_landing_no_cache = 1", "after_landing_no_cache = 1: true or false"),
                      ('landing_gates = "g.py"', "landing_gates = 'g.py': a list of Python scripts by their path"),
                      ('landing_gates = ["../g.py"]', "landing_gates = ['../g.py']: a list of Python scripts"),
                      ('landing_gates = ["/g.py"]', "landing_gates = ['/g.py']: a list of Python scripts"),
                      ('landing_gates = ["g.sh"]', "landing_gates = ['g.sh']: a list of Python scripts"),
                      ("stale_days = -1", "stale_days = -1: a whole number of days, 0 for off"),
                      ("stale_days = 3.5", "stale_days = 3.5: a whole number of days"),
                      ("stale_close_days = 7", "stale_close_days = 7: 0, or at least stale_days, which must be on"),
                      ("stale_days = 7\nstale_close_days = 3", "stale_close_days = 3: 0, or at least stale_days"),
                      ('command_line = ["max_lean_checks"]', "command_line = ['max_lean_checks']: a list of the keys a "
                                                             "command-line flag sets, of on, marked_after"),
                      ("on = [", "not TOML")):
        with pytest.raises(QueueError, match="^drive.toml: ") as e:
            DriveFlags.parse(text)
        assert why in str(e.value)


def test_drive_refuses_an_unknown_key_naming_the_keys_it_reads_and_the_one_it_likely_meant():
    with pytest.raises(QueueError) as e:
        DriveFlags.parse("max_lean_check = 4\n", "d.toml", strict=True)
    assert str(e.value) == ("d.toml: unknown key 'max_lean_check' (did you mean 'max_lean_checks'?): drive reads only on, "
                            "marked_after, regenerate_locks, first, land, land_into, after_landing_no_cache, landing_gates, max_lean_checks, "
                            "nonlean_first, reviewed_first, stale_days, stale_close_days, stale_keep, after_landing_shards, command_line; "
                            "rename or remove it, or update this research")


def test_a_client_ignores_every_key_it_doesnt_read_a_near_miss_too_so_it_parses_a_newer_mains_flags(capsys):
    assert DriveFlags.parse("max_running = 4\nmax_lean_slots = 2\nland = \"review\"\n", "d.toml").land == "review"
    err = capsys.readouterr().err
    assert "d.toml: ignoring key 'max_running'" in err and "d.toml: ignoring key 'max_lean_slots'" in err


#: what `driven` records of a Driver built from `FLAGS`
FROM_FLAGS = {"first": (1858,), "nonlean_first": False, "reviewed_first": False, "on": "m1,m2", "regenerate_locks": True,
              "after_landing_no_cache": True, "max_lean_checks": 3, "stale_days": 3, "stale_close_days": 7, "stale_keep": (),
              "after_landing_shards": 1}


def test_drive_reads_the_flags_at_main_in_its_clone_never_the_callers_tree_and_takes_a_new_main_without_a_restart(world, driven, monkeypatch, capsys):
    """The caller's working tree (here `dev`, checked out with other flags) is whatever its VM last pulled; drive reads main's,
    with --config or without it."""
    sha = on_main(world, FLAGS)
    monkeypatch.chdir(world.dev)
    (world.dev / DRIVE_FLAGS).write_text('on = ["stale"]\n')
    assert drive(world) == 0
    assert driven[-1] == FROM_FLAGS
    lines = [x for x in capsys.readouterr().out.splitlines() if "flags from" in x]
    assert lines == [f"research queue drive: flags from {DRIVE_FLAGS} at main {sha[:12]}: on=m1,m2 "
                     "marked_after=2026-10-12T06:00:00Z regenerate_locks=True first=1858 land=review "
                     "land_into=None after_landing_no_cache=True landing_gates=gates/g.py max_lean_checks=3 nonlean_first=False reviewed_first=False stale_days=3 "
                     "stale_close_days=7 stale_keep= after_landing_shards=1 command_line=on,first"]
    assert drive(world, "--config", DRIVE_FLAGS) == 0 and driven[-1] == FROM_FLAGS

    on_main(world, FLAGS.replace("max_lean_checks = 3", "max_lean_checks = 6"))
    assert drive(world, "--config", DRIVE_FLAGS) == 0
    assert driven[-1]["max_lean_checks"] == 6


def test_a_pod_config_whose_keys_all_have_mains_values_is_accepted_and_changes_nothing(world, driven, monkeypatch):
    on_main(world, FLAGS)
    pod = world.tmp / "drive-tonight.toml"
    for text in ("", "# emptied once its keys landed in drive.toml\n", 'on = ["m1", "m2"]\nmax_lean_checks = 3\nnonlean_first = false\n'
                 'reviewed_first = false\nland = "review"\nafter_landing_shards = 1\n'):
        pod.write_text(text)
        assert drive(world, "--config", str(pod), "--config-from-tree") == 0
        assert driven[-1] == FROM_FLAGS


def test_a_pod_config_key_that_differs_from_mains_refuses_the_tick_naming_each_key_both_values_and_the_fix(world, driven, monkeypatch, capsys):
    """Twice a file on the control pod silently overrode drive.toml at `main` (`landing_gates = []`; `land_into` left to `main`):
    now such a tick does nothing, and says what to change."""
    sha = on_main(world, FLAGS + 'land_into = "main"\n')
    pod = world.tmp / "drive-tonight.toml"
    pod.write_text('land_into = "checked"\nmax_lean_checks = 3\nafter_landing_shards = 3\nlanding_gates = []\n')
    assert drive(world, "--config", str(pod), "--config-from-tree") == 1
    assert capsys.readouterr().err == (
        f"research queue: --config {pod} differs from {DRIVE_FLAGS} at main {sha[:12]}, drive's one source of flags: "
        'land_into = "checked" there, "main" at main; after_landing_shards = 3 there, 1 at main; landing_gates = [] there, '
        f'["gates/g.py"] at main. Commit the change to {DRIVE_FLAGS}, or remove the key from {pod}\n')
    assert driven == []
    on_main(world, FLAGS)
    pod.write_text("land_into = \"main\"\n")                  # a key `main` leaves unset differs from every value
    assert drive(world, "--config", str(pod), "--config-from-tree") == 1
    assert 'land_into = "main" there, unset at main.' in capsys.readouterr().err and driven == []
    world.change({"pod.toml": "max_lean_checks = 4\n"})
    _git(world.dev, "push", "-q", "origin", "HEAD:refs/heads/main")
    assert drive(world, "--config", "pod.toml") == 1             # a file at `main` is held to drive.toml's values as well
    assert "max_lean_checks = 4 there, 3 at main." in capsys.readouterr().err and driven == []


def test_the_command_line_overrides_only_the_keys_drive_toml_lets_it(world, driven, monkeypatch, capsys):
    """The control pod's args pass `--on` and `--first` (drive.toml's `command_line`); any other flag refuses the tick."""
    sha = on_main(world, FLAGS)
    monkeypatch.chdir(world.dev)
    assert drive(world, "--on", "m9", "--first", "7,8") == 0
    assert driven[-1] == {**FROM_FLAGS, "first": (7, 8), "on": "m9"}
    assert drive(world, "--on", "m9", "--no-regenerate-locks", "--marked-after", "2026-10-13T06:00:00Z") == 1
    assert capsys.readouterr().err == (
        f"research queue: --marked-after and --regenerate-locks: {DRIVE_FLAGS} at main {sha[:12]} lets drive's command line set "
        f"only on, first (its `command_line`): commit the change to {DRIVE_FLAGS}, or drop the flag from drive's args\n")
    assert len(driven) == 1
    on_main(world, FLAGS + 'command_line = ["on", "first", "regenerate_locks", "after_landing_no_cache"]\n')
    assert drive(world, "--no-regenerate-locks", "--no-after-landing-no-cache") == 0
    assert driven[-1] == {**FROM_FLAGS, "regenerate_locks": False, "after_landing_no_cache": False}
    on_main(world, FLAGS + "command_line = []\n")
    assert drive(world, "--on", "m9") == 1 and "set only nothing (its `command_line`)" in capsys.readouterr().err


def test_without_drive_toml_at_main_drive_takes_the_defaults_and_its_on_and_first(world, driven, monkeypatch, capsys):
    rules = world.tmp / "rules.toml"
    rules.write_text('hold_label = "hold"\nmarked_after = "2026-10-12T06:00:00Z"\n')
    monkeypatch.chdir(world.dev)
    assert drive(world, "--on", "m1", "--rules", str(rules), "--first", "5") == 0
    assert driven[-1] == {"first": (5,), "nonlean_first": True, "reviewed_first": True, "on": "m1", "regenerate_locks": False,
                          "after_landing_no_cache": False, "max_lean_checks": 2, "stale_days": 0, "stale_close_days": 0,
                          "stale_keep": (), "after_landing_shards": 1}
    assert f"flags from the defaults ({DRIVE_FLAGS} isn't at main " in capsys.readouterr().out
    assert drive(world, "--on", "m1", "--rules", str(rules), "--regenerate-locks") == 1
    assert "--regenerate-locks: the defaults" in capsys.readouterr().err


def test_a_missing_landing_branch_refuses_the_tick_with_the_fix(world, driven, monkeypatch, capsys):
    """`land_into = "checked"` once went live with no `checked` on origin."""
    main = on_main(world, FLAGS + 'land_into = "checked"\n')
    assert drive(world) == 1
    assert capsys.readouterr().err == (
        "research queue: drive's flags name what doesn't exist, so this tick does nothing: origin has no branch `checked` "
        "(drive.toml's `land_into`): create it at `main` as drive's App (`git push origin refs/remotes/origin/main:"
        f"refs/heads/checked` in drive's checkout on the control pod), or commit `land_into = \"main\"` to {DRIVE_FLAGS}\n")
    assert driven == []
    _git(world.origin, "update-ref", "refs/heads/checked", main)
    assert drive(world) == 0 and len(driven) == 1


def test_a_landing_branch_drive_cant_push_refuses_the_tick(world, driven, monkeypatch, capsys):
    """`git push --dry-run` asks the remote for write access; a refusal names the credential's fix.  A dry run doesn't push,
    so it doesn't ask."""
    on_main(world, FLAGS)
    real = Q.Queue._remote

    def remote(self, cwd, *args, ok=(0,)):
        if "push" in args:
            assert "--dry-run" in args and args[-1] == "refs/remotes/origin/main:refs/heads/main"
            return subprocess.CompletedProcess(args, 128, "", "remote: Permission to o/r.git denied to verity-drive[bot].")
        return real(self, cwd, *args, ok=ok)

    monkeypatch.setattr(Q.Queue, "_remote", remote)
    assert drive(world) == 1
    assert ("drive's credential can't push `main` (`git push --dry-run`: remote: Permission to o/r.git denied to "
            "verity-drive[bot].): give drive's App contents write") in capsys.readouterr().err and driven == []
    assert drive(world, "--dry-run") == 0 and len(driven) == 1


def test_a_landing_gate_missing_at_main_refuses_the_tick(world, driven, monkeypatch, capsys):
    on_main(world, FLAGS.replace('landing_gates = ["gates/g.py"]', 'landing_gates = ["gates/g.py", "gates/gone.py"]'))
    assert drive(world) == 1
    assert capsys.readouterr().err == (
        "research queue: drive's flags name what doesn't exist, so this tick does nothing: landing gate gates/gone.py isn't at "
        f"`main`: land it first, or remove it from `landing_gates` in {DRIVE_FLAGS}\n")
    assert driven == []


def test_a_machine_that_doesnt_resolve_refuses_the_tick(world, driven, machines, monkeypatch, capsys):
    """Each machine of `on` (and of --on) must resolve as train.sh's `research run --on` resolves it; `auto` is train.sh's own."""
    on_main(world, FLAGS.replace('on = ["m1", "m2"]', 'on = ["m1", "gone", "auto"]'))
    known, refused = machines
    assert drive(world) == 1
    assert capsys.readouterr().err == (
        "research queue: drive's flags name what doesn't exist, so this tick does nothing: machine 'gone' of `on` doesn't "
        "resolve (not in machines.toml or machines.d): register it (`research pods register gone ...`), or remove it from `on` "
        f"in {DRIVE_FLAGS} (or drive's --on)\n")
    known.discard("m2")
    refused["m2"] = "machines.toml binds it to pod a and machines.d/m2.toml to pod b; refused until one of them is removed"
    assert drive(world, "--on", "m2") == 1
    assert "machine 'm2' of `on` doesn't resolve (machines.toml binds it to pod a and machines.d/m2.toml to pod b" in \
        capsys.readouterr().err
    assert driven == []
    assert drive(world, "--on", "m1,auto") == 0 and driven[-1]["on"] == "m1,auto"


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
    assert drive(world, "--config", str(world.dev / DRIVE_FLAGS)) == 1
    assert "a path from the repository's root, read at `main` (or --config-from-tree" in capsys.readouterr().err
    assert drive(world, "--config", str(world.tmp / "absent.toml"), "--config-from-tree") == 1
    assert f"--config {world.tmp / 'absent.toml'}: No such file or directory" in capsys.readouterr().err
    on_main(world, 'on = ["m1"]\nmax_lean_check = 4\n')
    assert drive(world) == 1
    assert "unknown key 'max_lean_check' (did you mean 'max_lean_checks'?)" in capsys.readouterr().err
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
