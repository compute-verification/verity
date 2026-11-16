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
    assert DriveFlags.parse('verdict_excludes = ["*.lean", "tools/verity/check/", "a/queue.py"]\n') == DriveFlags(
        verdict_excludes=("*.lean", "tools/verity/check/", "a/queue.py"))
    assert [DriveFlags.parse(f'after_landing_shards = "{v}"\n').after_landing_shards for v in ("auto", "auto:4")] == ["auto", "auto:4"]
    assert f.line() == ("on=m1,m2 marked_after=2026-10-12T06:00:00Z regenerate_locks=True land=review "
                        "land_into=None after_landing_no_cache=True landing_gates=gates/g.py warn_gates= warn_steps= verdict_first=False "
                        "stale_days=3 stale_close_days=7 stale_keep= close_unlandable=log after_landing_shards=1 command_line=on lowrisk_lane=True "
                        "lowrisk_per_tick=20 lowrisk_excludes= verdict_excludes= verdict_plain= executor=off executor_enqueue= verdict_carry=off bottom_ring=@ci "
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
                      ("verdict_excludes = []", "verdict_excludes = []: a non-empty list of directories"),
                      ('verdict_excludes = ["/tools/"]', "verdict_excludes = ['/tools/']: a non-empty list"),
                      ("verdict_plain = []", "verdict_plain = []: a non-empty list of directories"),
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
                            "lowrisk_per_tick, lowrisk_excludes, verdict_excludes, verdict_plain, executor, executor_enqueue, verdict_carry, bottom_ring, "
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
    excludes = tomllib.loads((root / "tools/check/drive.toml").read_text())["lowrisk_excludes"]
    for path in ("infra/", "tools/research/", "tools/move/", "tools/agent-guard/", ".cursor/", "pyproject.toml", "uv.lock"):
        assert path in excludes, f"drive.toml's lowrisk_excludes lacks {path}"


def test_the_land_verdicts_gate_reads_lowrisk_excludes_but_verdict_plain_and_adds_verdict_excludes():
    """Where the land verdicts gate (`research queue vqueue --gate verdict`), what keeps a change on the whole check
    (`DriveFlags.plain_excludes`) is `lowrisk_excludes` without the entries `verdict_plain` lists, and then
    `verdict_excludes`, each entry once; with `lowrisk_excludes` or `verdict_excludes` absent, nothing, whatever
    `verdict_plain` holds.  Over this tree's drive.toml, a change to Lean, a lock, a Lake file or the trusted hashes, the
    check, the research tool (the code that lands and reverts among it), the Lean check, the circuit language, a verifier or
    a device takes the whole check before it lands, whatever its land verdicts."""
    flags = DriveFlags.parse('lowrisk_excludes = ["*.lean", "a/"]\nverdict_excludes = ["a/", "b/x.py"]\n')
    assert flags.plain_excludes == ("*.lean", "a/", "b/x.py")
    flags = DriveFlags.parse('lowrisk_excludes = ["*.lean", "a/", "c/", "x.toml"]\nverdict_plain = ["c/", "x.toml"]\n'
                             'verdict_excludes = ["b/"]\n')
    assert flags.verdict_plain == ("c/", "x.toml") and flags.plain_excludes == ("*.lean", "a/", "b/")
    assert flags.lowrisk_excludes == ("*.lean", "a/", "c/", "x.toml") and not lowrisk(["c/y.py"], flags.lowrisk_excludes)
    assert lowrisk(["c/y.py", "d/x.toml"], flags.plain_excludes) and not lowrisk(["c/y.py", "a/z.py"], flags.plain_excludes)
    for text in ('lowrisk_excludes = ["a/"]\n', 'verdict_excludes = ["a/"]\n', "",
                 'lowrisk_excludes = ["a/", "c/"]\nverdict_plain = ["c/"]\n'):
        assert DriveFlags.parse(text).plain_excludes == () and not lowrisk(["README.md"], DriveFlags.parse(text).plain_excludes)
    root = Path(__file__).resolve().parents[3]
    flags = DriveFlags.parse((root / "tools/check/drive.toml").read_text())
    for path in ("security/Proofs/Core/X.lean", "tools/pous_grader/lean-audit.json", "security/lakefile.toml",
                 "security/lean-toolchain", "security/lake-manifest.json", "x/TRUSTED.sha256", ".gitattributes",
                 "tools/check/check.py", "tools/check/drive.toml", "tools/check/policy.toml",
                 *(f"tools/research/src/research/{m}.py" for m in ("queue", "landing", "vqueue", "vqueue_loop", "vqueue_slim",
                                                                    "cli", "merge", "carry", "labelauth")),
                 "tools/verity/lean/check.py", "verity/circuits/program.py", "verity/commitments/x.py",
                 "verity/protocols/resource_accounting/compute/pouw/verifier/audit.py",
                 "verity/protocols/compose/verifier/live_challenger/auditor.py"):
        assert not lowrisk([path, "README.md"], flags.plain_excludes), f"drive.toml's excludes let {path} land on a verdict"
    for path in ("GLOSSARY.md", "benchmarks/census/README.md"):
        assert lowrisk([path], flags.plain_excludes), f"drive.toml's excludes keep {path} on the whole check"


@pytest.mark.parametrize("path, entry", [
    ("tests/test_tree.py", "tests/"),
    ("pyproject.toml", "pyproject.toml"),
    ("adapters/vllm/verity_vllm/definitions/__init__.py", "__init__.py"),
    ("tools/agent-guard/shell_hook.py", "tools/agent-guard/"),
    ("infra/README.md", "infra/"),
    ("benchmarks/one_stage/a5.py", None),
    ("verity/protocols/resource_accounting/network/attester/network_attester_serve.py", None),
    ("benchmarks/remote_session/auditor_line.py", None),
    ("tools/move/restack.py", None),
    ("verity/circuits/program.py", None),
    ("tools/research/src/research/vqueue_loop.py", None),
    ("tools/check/drive.toml", None),
    ("verity/protocols/resource_accounting/compute/pouw/verifier/__init__.py", None),
    ("tests/verity/protocols/compose/verifier/lean/pyproject.toml", None),
], ids=lambda x: x or "-")
def test_each_verdict_plain_entry_frees_its_files_for_the_gate_and_the_rest_stay_on_the_whole_check(path, entry):
    """Over this tree's drive.toml and files: a file kept off the low-risk lane only by an entry of `verdict_plain` (``entry``)
    is plain where the land verdicts gate, and still off the low-risk lane; the code an auditor's device runs, the move tool,
    the circuit language, the research tool, the check, a verifier's `__init__.py` and a Lean package's `pyproject.toml`,
    which entries the gate reads keep off, are not plain."""
    root = Path(__file__).resolve().parents[3]
    flags = DriveFlags.parse((root / DRIVE_FLAGS).read_text())
    assert (root / path).is_file() and not lowrisk([path], flags.lowrisk_excludes)
    assert lowrisk([path], flags.plain_excludes) is (entry is not None)
    if entry is not None:
        assert entry in flags.verdict_plain and lowrisk([path], [x for x in flags.lowrisk_excludes if x != entry])


@pytest.mark.parametrize("text, why", [
    ('lowrisk_excludes = ["a/", "b/"]\nverdict_plain = ["a/", "a/"]\nverdict_excludes = ["d/"]',
     "verdict_plain = ['a/', 'a/']: a/ written more than once: keep each entry once"),
    ('lowrisk_excludes = ["a/", "b/"]\nverdict_plain = ["a"]\nverdict_excludes = ["d/"]',
     "verdict_plain = ['a']: a is no entry of lowrisk_excludes: each entry is one of lowrisk_excludes, written character "
     "for character as it is there; copy it from there, or remove it"),
    ('lowrisk_excludes = ["a/", "b/"]\nverdict_plain = ["a/x/"]\nverdict_excludes = ["d/"]',
     "verdict_plain = ['a/x/']: a/x/ is no entry of lowrisk_excludes"),
    ('verdict_plain = ["a/"]\nverdict_excludes = ["d/"]', "verdict_plain = ['a/']: a/ is no entry of lowrisk_excludes"),
    ('lowrisk_excludes = ["a/", "d/"]\nverdict_plain = ["d/"]\nverdict_excludes = ["d/"]',
     "verdict_plain = ['d/']: d/ is also an entry of verdict_excludes, which keeps it on the whole check: remove it from "
     "one of the two"),
    ('lowrisk_excludes = ["a/"]\nverdict_plain = ["../a/"]', "verdict_plain = ['../a/']: a non-empty list of directories"),
], ids=["a duplicate", "not character for character", "narrower than its entry", "lowrisk_excludes absent",
        "also in verdict_excludes", "malformed"])
def test_the_parse_refuses_a_verdict_plain_entry_that_is_no_lowrisk_exclude_one_in_verdict_excludes_and_a_duplicate(text, why):
    """`verdict_plain` can't go stale or widen silently: each entry is one of `lowrisk_excludes` as written there (so a
    renamed or deleted entry fails the parse, and none frees what that list doesn't name), none is in `verdict_excludes`,
    and none is written twice."""
    with pytest.raises(QueueError, match="^drive.toml: ") as e:
        DriveFlags.parse(text)
    assert why in str(e.value)


#: drive.toml's `lowrisk_excludes`
EXCLUDES = ("*.lean", "lean-audit.json", "lakefile.*", "lean-toolchain", "lake-manifest.json", "TRUSTED.sha256",
            "verity/core/", "verity/circuits/", "verity/circuits/ops/", "verity/circuits/kernels/", "verity/commitments/", "verity/commitments/crypto/",
            "verity/commitments/randomness/")


@pytest.mark.parametrize("files, low", [
    (["tools/research/src/research/landing.py", "README.md"], True),
    (["adapters/vllm/verity_vllm/serve.py", "verity/ml/flock/x.py", "verity/ml/definitions/y.py"], True),
    (["security/Proofs/Core/X.lean"], False),
    (["docs/a.md", "tools/pous_grader/lean-audit.json"], False),
    (["verity/protocols/proofs/flock/verifier/lakefile.toml"], False),
    (["x/lean-toolchain"], False), (["x/lake-manifest.json"], False), (["x/TRUSTED.sha256"], False),
    (["verity/core/claims.py"], False),
    (["verity/circuits/ops/add.py"], False), (["verity/circuits/kernels/k.cu"], False),
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
    excludes = ("verity/protocols/__init__.py", "verity/channel.py")
    for f in excludes:
        assert not lowrisk(["README.md", f], excludes), f
    assert lowrisk(["verity/protocols/resource_accounting/compute/pouw/__init__.py", "verity/protocols/reporting.py", "adapters/vllm/channel.py",
                    "x/verity/channel.py", "verity/channel.pyc"], excludes)


def test_a_directory_entry_holds_its_tests_under_the_roots_tests_and_tests_holds_only_the_repository_suites_own():
    """An entry `D/` keeps `D`'s tests at `tests/D/` off too, and `tests/` keeps off only the repository suite's own: the
    files of `tests/` itself and of its own directories (`suite_layout.ROOT_OWN`), not the tests of another directory."""
    excludes = ("verity/core/", "tests/", "tools/check/", "verity/protocols/__init__.py", "*.lean")
    for f in ("tests/verity/core/test_claims.py", "tests/verity/core/sub/test_x.py", "tests/tools/check/test_suites.py",
              "tests/test_boundaries.py", "tests/lint/test_repo_ruff.py", "tests/conftest.py", "tests/verity/X.lean"):
        assert not lowrisk(["README.md", f], excludes), f
    assert lowrisk(["tests/verity/circuits/test_ops.py", "tests/verity/protocols/__init__.py", "tests/verity/corex/test_a.py",
                    "tests/benchmarks/pouw/test_a.py", "tests/tools/research/test_queue.py", "tests/verity/circuits/test_x.py"],
                   excludes)
    assert lowrisk(["tests/lint/test_y.py"], ("lint/",)) and not lowrisk(["tests/verity/core/test_claims.py"], ("verity/",))
    assert not lowrisk(["tests/verity/core/test_claims.py"], ("tests/verity/",))
    assert not lowrisk(["tests/test_boundaries.py"], ("tests/",)) and lowrisk(["tests/verity/test_a.py"], ("tests/",))
