"""drive.toml's flags (`DriveFlags`), which the verified queue's loop reads at `main`: they parse with their types, absent
keys take their defaults, the strict parse refuses an unknown key naming the keys it reads, a reader ignores a key it doesn't
read so it parses a newer `main`'s flags, the gates read whatever other keys the file holds, `land_into` is `checked` or
`main` and only drive's App pushes `checked`, and the low-risk class (`lowrisk`) is no file the excludes name."""
from __future__ import annotations

from pathlib import Path

import pytest

from research.queue import DRIVE_FLAGS, DriveFlags, QueueError, lowrisk
from test_queue import World, _git

FLAGS = """on = ["m1", "m2"]
marked_after = "2026-10-12T06:00:00Z"
regenerate_locks = true
land = "review"
after_landing_no_cache = true
landing_gates = ["gates/g.py"]
verdict_first = false
stale_days = 3
stale_close_days = 7
close_unlandable = "log"
bottom_ring = ["@ci"]
bottom_wait_min = 45
"""


def test_the_flags_parse_with_their_types_and_absent_keys_take_the_commands_defaults():
    f = DriveFlags.parse(FLAGS)
    assert f == DriveFlags(on="m1,m2", marked_after="2026-10-12T06:00:00Z", regenerate_locks=True,
                           land="review", after_landing_no_cache=True, landing_gates=("gates/g.py",), verdict_first=False,
                           stale_days=3, stale_close_days=7, close_unlandable="log", bottom_ring=("@ci",), bottom_wait_min=45)
    assert DriveFlags.parse("") == DriveFlags() == DriveFlags(None, None, False, "review", None, False)
    assert (DriveFlags().bottom_ring, DriveFlags().bottom_wait_min) == (("@ci", "@top"), 30)
    assert DriveFlags.parse("after_landing_shards = 3\n").after_landing_shards == 3
    assert DriveFlags.parse("restack_jobs = 1\n").restack_jobs == 1
    assert DriveFlags.parse('warn_steps = ["ratchet-gate"]\n').warn_steps == ("ratchet-gate",)
    assert DriveFlags.parse('lowrisk_excludes = ["*.lean", "verity/", "verity/__init__.py"]\n').lowrisk_excludes == (
        "*.lean", "verity/", "verity/__init__.py")
    assert [DriveFlags.parse(f'after_landing_shards = "{v}"\n').after_landing_shards for v in ("auto", "auto:4")] == ["auto", "auto:4"]
    assert f.line() == ("on=m1,m2 marked_after=2026-10-12T06:00:00Z regenerate_locks=True land=review "
                        "land_into=None after_landing_no_cache=True landing_gates=gates/g.py warn_gates= warn_steps= verdict_first=False "
                        "stale_days=3 stale_close_days=7 stale_keep= close_unlandable=log after_landing_shards=1 command_line=on lowrisk_lane=True "
                        "lowrisk_per_tick=20 lowrisk_excludes= executor=off executor_enqueue= verdict_carry=off bottom_ring=@ci "
                        "bottom_wait_min=45 signed_verdicts=off restack_jobs=3")
    assert [DriveFlags.parse(f'executor = "{v}"\n').executor for v in ("off", "shadow", "on")] == ["off", "shadow", "on"]
    assert [DriveFlags.parse(f'verdict_carry = "{v}"\n').verdict_carry for v in ("off", "shadow", "on")] == ["off", "shadow", "on"]
    assert [DriveFlags.parse(f'close_unlandable = "{v}"\n').close_unlandable for v in ("off", "log", "on")] == ["off", "log", "on"]
    assert DriveFlags.parse('executor_enqueue = ["python3", "enqueue.py"]\nstale_keep = [2692, 2693]\n') == DriveFlags(
        executor_enqueue=("python3", "enqueue.py"), stale_keep=(2692, 2693))
    for text, why in (('on = "m1,m2"', 'on = \'m1,m2\': a list of names, as ["a", "b"]'),
                      ("executor_enqueue = []", "executor_enqueue = []: a command as a non-empty list of its words"),
                      ('executor_enqueue = ["python3", ""]', "executor_enqueue = ['python3', '']: a command as a non-empty"),
                      ('executor_enqueue = "enqueue.py"', "executor_enqueue = 'enqueue.py': a command as a non-empty"),
                      ("stale_keep = [0]", "stale_keep = [0]: a list of PR numbers, as [2692, 2693]"),
                      ('stale_keep = ["2692"]', "stale_keep = ['2692']: a list of PR numbers"),
                      ("stale_keep = 2692", "stale_keep = 2692: a list of PR numbers"),
                      ('executor = "yes"', "executor = 'yes': one of off, shadow, on"),
                      ("verdict_carry = true", "verdict_carry = True: one of off, shadow, on"),
                      ('close_unlandable = "yes"', "close_unlandable = 'yes': one of off, log, on"),
                      ("on = []", "on = []: at least one machine"),
                      ("after_landing_shards = 0", "after_landing_shards = 0: a whole number, 1 or a shard plan's N"),
                      ('after_landing_shards = "auto:0"', "after_landing_shards = 'auto:0': a whole number, 1 or a shard plan's N"),
                      ('after_landing_shards = "3"', "after_landing_shards = '3': a whole number, 1 or a shard plan's N"),
                      ('marked_after = "12 Oct"', "marked_after = '12 Oct': a UTC time as YYYY-MM-DDTHH:MM:SSZ"),
                      ('regenerate_locks = "yes"', "regenerate_locks = 'yes': true or false"),
                      ("verdict_first = 1", "verdict_first = 1: true or false"),
                      ("lowrisk_lane = 1", "lowrisk_lane = 1: true or false"),
                      ("lowrisk_per_tick = 0", "lowrisk_per_tick = 0: a whole number, 1 or more"),
                      ("restack_jobs = 0", "restack_jobs = 0: a whole number, 1 or more"),
                      ("restack_jobs = 2.5", "restack_jobs = 2.5: a whole number, 1 or more"),
                      ("lowrisk_excludes = []", "lowrisk_excludes = []: a non-empty list of directories"),
                      ('lowrisk_excludes = ["/verity/core/"]', "lowrisk_excludes = ['/verity/core/']: a non-empty list"),
                      ('lowrisk_excludes = ["../x/"]', "lowrisk_excludes = ['../x/']: a non-empty list"),
                      ('lowrisk_excludes = ["verity/*/x.py"]', "lowrisk_excludes = ['verity/*/x.py']: a non-empty list"),
                      ('lowrisk_excludes = ["x/../y.py"]', "lowrisk_excludes = ['x/../y.py']: a non-empty list"),
                      ("bottom_ring = []", "bottom_ring = []: a non-empty list of handles, each `@` and its lower-case name, "
                                           "as [\"@ci\"]"),
                      ('bottom_ring = "@ci"', "bottom_ring = '@ci': a non-empty list of handles"),
                      ('bottom_ring = ["ci"]', "bottom_ring = ['ci']: a non-empty list of handles"),
                      ('bottom_ring = ["@ci @top"]', "bottom_ring = ['@ci @top']: a non-empty list of handles"),
                      ('bottom_ring = ["@CI"]', "bottom_ring = ['@CI']: a non-empty list of handles"),
                      ("bottom_wait_min = 0", "bottom_wait_min = 0: a whole number of minutes, 1 or more"),
                      ("bottom_wait_min = 0.5", "bottom_wait_min = 0.5: a whole number of minutes"),
                      ('bottom_wait_min = "30"', "bottom_wait_min = '30': a whole number of minutes"),
                      ('land = "fast"', "land = 'fast': one of review"),
                      ('land = "check"', "land = 'check': train landing was deleted with its code: drive lands on "
                                         "review only"),
                      ('land_into = "next"', "land_into = 'next': \"checked\" (main follows it to fully checked commits) or "
                                             "\"main\""),
                      ("after_landing_no_cache = 1", "after_landing_no_cache = 1: true or false"),
                      ('landing_gates = "g.py"', "landing_gates = 'g.py': a list of Python scripts by their path"),
                      ('landing_gates = ["../g.py"]', "landing_gates = ['../g.py']: a list of Python scripts"),
                      ('landing_gates = ["/g.py"]', "landing_gates = ['/g.py']: a list of Python scripts"),
                      ('landing_gates = ["g.sh"]', "landing_gates = ['g.sh']: a list of Python scripts"),
                      ('warn_gates = "g.py"', "warn_gates = 'g.py': a list of Python scripts by their path"),
                      ('warn_gates = ["../g.py"]', "warn_gates = ['../g.py']: a list of Python scripts"),
                      ('warn_steps = "ratchet-gate"', "warn_steps = 'ratchet-gate': a list of the check's steps, of "
                                                      "preflight-lock, "),
                      ('warn_steps = ["ratchet"]', "warn_steps = ['ratchet']: a list of the check's steps"),
                      ("stale_days = -1", "stale_days = -1: a whole number of days, 0 for off"),
                      ("stale_days = 3.5", "stale_days = 3.5: a whole number of days"),
                      ("stale_close_days = 7", "stale_close_days = 7: 0, or at least stale_days, which must be on"),
                      ("stale_days = 7\nstale_close_days = 3", "stale_close_days = 3: 0, or at least stale_days"),
                      ('command_line = ["stale_days"]', "command_line = ['stale_days']: a list of the keys a "
                                                        "command-line flag sets, of on, marked_after"),
                      ('command_line = ["on", "first"]', "command_line = ['on', 'first']: a list of the keys a command-line "
                                                         "flag sets, of on, marked_after, regenerate_locks, "
                                                         "after_landing_no_cache"),
                      ("on = [", "not TOML")):
        with pytest.raises(QueueError, match="^drive.toml: ") as e:
            DriveFlags.parse(text)
        assert why in str(e.value)


def test_the_strict_parse_refuses_an_unknown_key_naming_the_keys_it_reads_and_the_one_it_likely_meant():
    with pytest.raises(QueueError) as e:
        DriveFlags.parse("stale_day = 4\n", "d.toml", strict=True)
    assert str(e.value) == ("d.toml: unknown key 'stale_day' (did you mean 'stale_days'?): this research reads only on, "
                            "marked_after, regenerate_locks, land, land_into, after_landing_no_cache, landing_gates, "
                            "warn_gates, warn_steps, verdict_first, stale_days, stale_close_days, stale_keep, close_unlandable, "
                            "after_landing_shards, command_line, lowrisk_lane, "
                            "lowrisk_per_tick, lowrisk_excludes, executor, executor_enqueue, verdict_carry, bottom_ring, "
                            "bottom_wait_min, signed_verdicts, restack_jobs; "
                            "rename or remove it, or update this research")


def test_a_client_ignores_every_key_it_doesnt_read_a_near_miss_too_so_it_parses_a_newer_mains_flags(capsys):
    assert DriveFlags.parse("max_running = 4\nmax_lean_slots = 2\nland = \"review\"\n", "d.toml").land == "review"
    err = capsys.readouterr().err
    assert "d.toml: ignoring key 'max_running'" in err and "d.toml: ignoring key 'max_lean_slots'" in err


def test_a_key_whose_code_was_deleted_is_ignored_so_a_main_that_still_sets_it_parses(capsys):
    """Code deployed ahead of `main` reads `main`'s drive.toml, which keeps a deleted key until `main` reaches the commit that
    removed it: refusing it would stop every tick."""
    f = DriveFlags.parse(FLAGS + "max_lean_checks = 10\nnonlean_first = false\nreviewed_first = false\n", "d.toml", strict=True)
    assert f == DriveFlags.parse(FLAGS)
    err = capsys.readouterr().err
    assert all(f"d.toml: ignoring key {k!r}, which this research doesn't read (its code was deleted: remove it)" in err
               for k in ("max_lean_checks", "nonlean_first", "reviewed_first")), err


def test_the_gates_are_read_whatever_other_keys_drive_toml_holds(tmp_path):
    """A branch's older queue.py reads main's drive.toml for the preflight's gates; a key added since doesn't stop it."""
    from research.queue import gates_at

    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / DRIVE_FLAGS).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / DRIVE_FLAGS).write_text('landing_gates = ["gates/g.py"]\nwarn_gates = ["gates/w.py"]\n'
                                        'max_lean_checks_of_the_future = 9\n')
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
    assert gates_at(tmp_path, "main") == ("gates/g.py",) and gates_at(tmp_path, "main", "warn_gates") == ("gates/w.py",)
    (tmp_path / DRIVE_FLAGS).write_text('landing_gates = "gates/g.py"\n')
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "y")
    with pytest.raises(QueueError, match="landing_gates"):
        gates_at(tmp_path, "main")


def test_land_into_takes_checked_or_main_and_only_drives_app_pushes_checked(tmp_path, monkeypatch):
    assert DriveFlags.parse('land_into = "checked"\n').land_into == "checked" and DriveFlags().land_into is None
    assert DriveFlags.parse('on = ["m"]\n').land_into is None
    with pytest.raises(QueueError, match="land_into"):
        DriveFlags.parse('land_into = "staging"\n')
    monkeypatch.delenv("RESEARCH_DRIVE", raising=False)
    with pytest.raises(QueueError, match="`checked` is moved only by drive's App"):
        World(tmp_path).queue(land_into="checked")._drive_only("checked", "once", False)


def test_the_low_risk_lane_excludes_what_runs_with_the_control_pods_or_every_agents_credentials():
    """A change to infra, the research tool, the move scripts and maps (which the loop runs, and certifies a move against), the
    agent guard and its hooks, or the environment runs on the control pod or on every agent once it is on main, so it lands
    only on a review, never on a ready mark alone."""
    import tomllib

    root = Path(__file__).resolve().parents[3]
    excludes = tomllib.loads((root / "tools/verity/check/drive.toml").read_text())["lowrisk_excludes"]
    for path in ("infra/", "tools/research/", "tools/move/", "tools/agent-guard/", ".cursor/", "pyproject.toml", "uv.lock"):
        assert path in excludes, f"drive.toml's lowrisk_excludes lacks {path}"


#: drive.toml's `lowrisk_excludes`
EXCLUDES = ("*.lean", "lean-audit.json", "lakefile.*", "lean-toolchain", "lake-manifest.json", "TRUSTED.sha256",
            "verity/core/", "verity/circuits/", "verity/definitions/ops/", "verity/definitions/kernels/", "verity/commitments/", "verity/commitments/crypto/",
            "verity/commitments/randomness/")


@pytest.mark.parametrize("files, low", [
    (["tools/research/src/research/landing.py", "README.md"], True),
    (["adapters/vllm/verity_vllm/serve.py", "verity/ml/flock/x.py", "verity/ml/definitions/y.py"], True),
    (["security/Proofs/Core/X.lean"], False),
    (["docs/a.md", "verity/protocols/pous/lean/lean-audit.json"], False),
    (["verity/protocols/flock/verifier/lakefile.toml"], False),
    (["x/lean-toolchain"], False), (["x/lake-manifest.json"], False), (["x/TRUSTED.sha256"], False),
    (["verity/core/claims.py"], False),
    (["verity/definitions/ops/add.py"], False), (["verity/definitions/kernels/k.cu"], False),
    (["verity/commitments/c.py"], False), (["verity/commitments/crypto/c.py"], False), (["verity/commitments/randomness/r.py"], False),
    (["verity/ml/circuitsx/a.py"], True),                   # a directory entry is a whole path component
    ([], False),                                            # nothing to land
])
def test_the_class_is_no_file_drive_toml_excludes(files, low):
    assert lowrisk(files, EXCLUDES) is low


def test_no_excludes_is_no_low_risk_change():
    assert not lowrisk(["README.md"], ())


def test_an_entry_with_a_slash_before_its_end_keeps_that_one_file_off():
    """A package's `__init__.py` runs on every import below it, so it is kept off by its path, without the rest of its
    directory; such an entry names no other file of that name, and no path it only begins or ends."""
    excludes = ("verity/protocols/__init__.py", "verity/hardware/record.py")
    for f in excludes:
        assert not lowrisk(["README.md", f], excludes), f
    assert lowrisk(["verity/protocols/pouw/__init__.py", "verity/protocols/reporting.py", "adapters/vllm/record.py",
                    "x/verity/hardware/record.py", "verity/hardware/record.pyc"], excludes)
