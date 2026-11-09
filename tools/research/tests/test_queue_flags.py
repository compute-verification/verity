"""drive.toml's flags (`DriveFlags`), read at `main` (`flags_at`) by the verified queue's loop, `research queue ready`, `review`
and `status`: their types and the defaults absent keys take, a malformed value refused naming the fix, a key the code doesn't
read refused by the strict parse (the tree's own file, `tests/test_drive_toml.py`) and ignored by every reader, so a reader
older than `main` keeps working; the gates read whatever else the file holds; and the low-risk class."""
from __future__ import annotations

import pytest

from research.queue import DRIVE_FLAGS, DriveFlags, QueueError, gates_at, lowrisk
from test_queue import _git

FLAGS = """on = ["m1", "m2"]
marked_after = "2026-10-12T06:00:00Z"
land_into = "checked"
landing_gates = ["gates/g.py"]
warn_gates = ["gates/w.py"]
stale_days = 3
stale_keep = [2692, 2693]
lowrisk_excludes = ["*.lean", "verity/core/"]
"""
#: drive.toml's `lowrisk_excludes`, but the guards of the landing rules and what runs with everyone's credentials
EXCLUDES = ("*.lean", "lean-audit.json", "lakefile.*", "lean-toolchain", "lake-manifest.json", "TRUSTED.sha256",
            "verity/core/", "verity/ml/circuits/", "verity/ml/kernels/", "verity/ml/commitments/", "verity/ml/crypto/",
            "verity/ml/randomness/")


def test_the_flags_parse_with_their_types_and_absent_keys_take_their_defaults():
    assert DriveFlags.parse(FLAGS) == DriveFlags(on="m1,m2", marked_after="2026-10-12T06:00:00Z", land_into="checked",
                                                 landing_gates=("gates/g.py",), warn_gates=("gates/w.py",), stale_days=3,
                                                 stale_keep=(2692, 2693), lowrisk_excludes=("*.lean", "verity/core/"))
    assert DriveFlags.parse("") == DriveFlags()
    d = DriveFlags()
    assert (d.on, d.land_into, d.verdict_first, d.lowrisk_lane, d.signed_verdicts, d.restack_jobs) == (None, None, True, True,
                                                                                                        False, 3)
    for text, why in (('on = "m1,m2"', 'on = \'m1,m2\': a list of names, as ["a", "b"]'),
                      ("on = []", "on = []: at least one machine"),
                      ('marked_after = "12 Oct"', "marked_after = '12 Oct': a UTC time as YYYY-MM-DDTHH:MM:SSZ"),
                      ('land_into = "next"', "land_into = 'next': \"checked\" (frozen, while `main` doesn't hold it) or \"main\""),
                      ('landing_gates = "g.py"', "landing_gates = 'g.py': a list of Python scripts by their path"),
                      ('landing_gates = ["../g.py"]', "landing_gates = ['../g.py']: a list of Python scripts"),
                      ('landing_gates = ["g.sh"]', "landing_gates = ['g.sh']: a list of Python scripts"),
                      ('warn_gates = ["/g.py"]', "warn_gates = ['/g.py']: a list of Python scripts"),
                      ("verdict_first = 1", "verdict_first = 1: true or false"),
                      ("lowrisk_lane = 1", "lowrisk_lane = 1: true or false"),
                      ("restack_jobs = 0", "restack_jobs = 0: a whole number, 1 or more"),
                      ("lowrisk_excludes = []", "lowrisk_excludes = []: a non-empty list of directories"),
                      ('lowrisk_excludes = ["../x/"]', "lowrisk_excludes = ['../x/']: a non-empty list"),
                      ('lowrisk_excludes = ["verity/*/x.py"]', "lowrisk_excludes = ['verity/*/x.py']: a non-empty list"),
                      ("stale_days = -1", "stale_days = -1: a whole number of days, 0 for off"),
                      ('stale_keep = ["2692"]', "stale_keep = ['2692']: a list of PR numbers"),
                      ("on = [", "not TOML")):
        with pytest.raises(QueueError, match="^drive.toml: ") as e:
            DriveFlags.parse(text)
        assert why in str(e.value)


def test_signed_verdicts_needs_every_guard_of_the_labels_off_the_low_risk_lane():
    """A change to who holds a handle or how a label is judged landing on its ready mark alone could make a lane another's
    holder, so signed verdicts are refused until the low-risk lane excludes each guard or is off."""
    text = 'signed_verdicts = true\nlowrisk_excludes = ["tools/research/", ".agents/"]\n'
    assert DriveFlags.parse(text).signed_verdicts
    with pytest.raises(QueueError, match="tools/research/src/research/vqueue_loop.py"):
        DriveFlags.parse('signed_verdicts = true\nlowrisk_excludes = [".agents/"]\n')
    assert DriveFlags.parse('signed_verdicts = true\nlowrisk_lane = false\n').signed_verdicts


def test_the_strict_parse_refuses_an_unknown_key_naming_the_keys_it_reads_and_the_one_it_likely_meant():
    with pytest.raises(QueueError) as e:
        DriveFlags.parse("stale_day = 4\n", "d.toml", strict=True)
    assert str(e.value) == ("d.toml: unknown key 'stale_day' (did you mean 'stale_days'?): the queue reads only on, "
                            "marked_after, land_into, landing_gates, warn_gates, verdict_first, stale_days, stale_keep, "
                            "lowrisk_lane, lowrisk_excludes, signed_verdicts, restack_jobs; rename or remove it, or update this "
                            "research")


def test_a_reader_ignores_every_key_it_doesnt_read_a_near_miss_too_so_it_parses_a_newer_mains_flags(capsys):
    assert DriveFlags.parse('max_running = 4\nstale_day = 2\nland = "review"\nstale_days = 1\n', "d.toml").stale_days == 1
    err = capsys.readouterr().err
    assert all(f"d.toml: ignoring key {k!r}" in err for k in ("max_running", "stale_day", "land")), err


def test_the_gates_are_read_whatever_other_keys_drive_toml_holds(tmp_path):
    """A branch's older queue.py reads main's drive.toml for the preflight's gates; a key added since doesn't stop it."""
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


@pytest.mark.parametrize("files, low", [
    (["tools/research/src/research/landing.py", "README.md"], True),
    (["integrations/vllm/verity_vllm/serve.py", "verity/ml/flock/x.py", "verity/ml/definitions/y.py"], True),
    (["Security/Proofs/Core/X.lean"], False),
    (["docs/a.md", "verity/core/protocols/pous/lean/lean-audit.json"], False),
    (["verity/core/service/flock/lakefile.toml"], False),
    (["x/lean-toolchain"], False), (["x/lake-manifest.json"], False), (["x/TRUSTED.sha256"], False),
    (["verity/core/claims.py"], False),
    (["verity/ml/circuits/ops/add.py"], False), (["verity/ml/kernels/k.cu"], False),
    (["verity/ml/commitments/c.py"], False), (["verity/ml/crypto/c.py"], False), (["verity/ml/randomness/r.py"], False),
    (["verity/ml/circuitsx/a.py"], True),                   # a directory entry is a whole path component
    ([], False),                                            # nothing to land
])
def test_the_low_risk_class_is_no_file_the_excludes_name(files, low):
    assert lowrisk(files, EXCLUDES) is low


def test_no_excludes_is_no_low_risk_change():
    assert not lowrisk(["README.md"], ())
