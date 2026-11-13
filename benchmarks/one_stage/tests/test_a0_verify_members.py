"""a0 hands a run's members to one kept verifier (`a0.VerifyMembers`, `a0.lean_verify`'s ``members``), on a stand-in for
Lean's verifier: each member gets the verdict `verify` gives its session as a process of its own, equal but for the
seconds, and its log reads as that process's does; a member the kept process gives no verdict is refused, and so is every
member after it, with no second process and no second try."""
from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import a0  # noqa: E402
from test_a0_archive_once import window  # noqa: E402


def kept_window(tmp_path: Path) -> tuple[str, list[tuple[dict, Path]], a0.VerifyMembers]:
    """`window`'s three members, their files put once, and a kept verifier on their program."""
    lean, _, members, archived = window(tmp_path)
    archive = tmp_path / "lean-verify" / "archive"
    keys = a0.archive_put(str(lean), archive, [f for st, _ in members for f in a0.archive_files(st, 8)],
                          tmp_path / "lean-verify" / "archive-put.log", archived)
    kept = a0.VerifyMembers(str(lean), archive, keys[members[0][0]["program_file"]], tmp_path / "lean-verify" / "verify-members.log")
    return str(lean), members, kept


def verify_line(log: Path) -> list[str]:
    """The arguments after `verify` on the command line a verify log opens with, as the two-stage audit reads them."""
    argv = shlex.split(next(x for x in log.read_text().splitlines() if x.startswith("$ "))[2:])
    return argv[argv.index("verify") + 1:]


def verdict_lines(log: Path) -> list[str]:
    return [x for x in log.read_text().splitlines() if x.startswith("VERDICT ")]


def test_each_member_gets_the_verdict_verify_gives_its_session_alone(tmp_path):
    lean, members, kept = kept_window(tmp_path)
    for i, (st, out) in enumerate(members):
        got = a0.lean_verify(lean, st, 8, out, zk=True, members=kept)
        alone = a0.lean_verify(lean, st, 8, out, zk=True, log="lean-verify-alone")
        assert got["accepted"] and got["digest"] is not None
        assert {**got, "verify_s": None} == {**alone, "verify_s": None}
        log, own = out / "lean-verify.log", out / "lean-verify-alone.log"
        # the member's log is a `verify` process's: its command line, its seconds, its verdict and its standard error
        assert verify_line(log) == verify_line(own) and "--zk" in verify_line(log)
        assert verdict_lines(log) == verdict_lines(own) and len(verdict_lines(log)) == 1
        assert log.read_text().split("--- stderr\n")[1] == own.read_text().split("--- stderr\n")[1] != ""
        head = log.read_text().split("--- stdout")[0].splitlines()
        assert re.fullmatch(r"# rc 0 in \d+\.\d{3} s", head[1]) and head[2] == f"# run as member {i} of: {' '.join(kept.cmd)}"
    assert kept.close() == 0 and kept.members == 3
    text = (tmp_path / "lean-verify" / "verify-members.log").read_text()
    assert text.splitlines()[0] == f"$ {lean} verify-members --archive {tmp_path / 'lean-verify' / 'archive'} --program {kept.cmd[-1]}"
    assert "# 3 members handed in; closed after its last member (it exited 0)" in text
    assert [x for x in text.splitlines() if x.startswith("MEMBER ")] == [f'MEMBER {{"exit":0,"index":{i}}}' for i in range(3)]
    # every member's standard error is in the process's log once, and the file it was read from is gone
    assert text.split("--- stderr\n")[1].count("inputs: 5 ms") == 3 and not kept.stderr.exists()


def test_a_member_verify_refuses_or_stops_on_is_that_members_alone(tmp_path):
    """Member 1's public object is not its key's bytes, so `verify` refuses it; member 2 stops at its usage, with no verdict.
    Each is refused as `verify` alone refuses it, and the kept process is still the one that answered."""
    lean, members, kept = kept_window(tmp_path)
    (st1, out1), (_, out2) = members[1], members[2]
    key = a0.archive_put(lean, out1 / "archive", [st1["dir"] / "pub-8.bin"], tmp_path / "again.log", st1["archived"])[st1["dir"] / "pub-8.bin"]
    (out1 / "archive" / "sha512" / key).unlink()
    (out1 / "archive" / "sha512" / key).write_bytes(b"other bytes")
    (out2 / "sessions" / "s" / "mode").write_text("usage")
    got = [a0.lean_verify(lean, st, 8, out, zk=True, members=kept) for st, out in members]
    alone = [a0.lean_verify(lean, st, 8, out, zk=True, log="lean-verify-alone") for st, out in members]
    assert [(v["accepted"], v["why"]) for v in got] == [(True, None), (False, "archive"),
                                                        (False, "no verdict (rc 2): usage: verify ...\n")]
    assert [{**v, "verify_s": None} for v in got] == [{**v, "verify_s": None} for v in alone]
    assert a0.lean_refused(got[1]) and not a0.lean_refused(got[2]) and kept.ended is None
    assert kept.close() == 0


@pytest.mark.parametrize("mode, why", [
    ("dies", "the kept verifier ended before member 1's MEMBER line (it exited 3)"),
    ("another place", 'the kept verifier answered member 1 with MEMBER {"exit":0,"index":2} (it exited -9)'),
    ("no JSON", "the kept verifier answered member 1 with MEMBER no JSON (it exited -9)")])
def test_a_member_the_kept_process_does_not_answer_is_refused_and_so_is_every_member_after_it(tmp_path, mode, why):
    """The process ends at member 1, or answers it with another place or no status: member 1 has no verdict whatever
    `VERDICT` line came before, member 2 is never handed in, and neither is tried again."""
    lean, members, kept = kept_window(tmp_path)
    (members[1][1] / "sessions" / "s" / "mode").write_text(mode)
    got = [a0.lean_verify(lean, st, 8, out, zk=True, members=kept) for st, out in members]
    assert [v["accepted"] for v in got] == [True, False, False]
    assert all(v["digest"] is None and v["why"].startswith(f"no verdict: {why}: ") and not a0.lean_refused(v) for v in got[1:])
    assert kept.ended == why and kept.members == 3
    one, two = (out / "lean-verify.log" for _, out in members[1:])
    # what the process printed for member 1 is kept, and is no verdict to a reader of its log
    assert verdict_lines(one) == [] and "not taken: VERDICT " in one.read_text() and "VERDICT" not in two.read_text()
    for i, log in ((1, one), (2, two)):
        head = log.read_text().splitlines()
        assert re.fullmatch(r"# rc None in \d+\.\d{3} s", head[1])
        assert head[2:4] == [f"# run as member {i} of: {' '.join(kept.cmd)}", f"# {why}"]
    assert verify_line(two)[-2:] == ["--session", str(members[2][1] / "sessions" / "s")]
    assert kept.close() == kept.proc.returncode != 0
    assert f"# 3 members handed in; {why}" in (tmp_path / "lean-verify" / "verify-members.log").read_text()


def test_a_kept_process_that_cannot_read_its_program_gives_no_member_a_verdict(tmp_path):
    lean, _, members, archived = window(tmp_path)
    kept = a0.VerifyMembers(str(lean), tmp_path / "lean-verify" / "archive", "", tmp_path / "verify-members.log")
    got = [a0.lean_verify(str(lean), st, 8, out, zk=True, members=kept) for st, out in members]
    assert [v["accepted"] for v in got] == [False] * 3 and kept.proc.returncode == 2
    assert all("(it exited 2)" in v["why"] for v in got) and "program: no such object" in got[0]["why"]
    assert kept.close() == 2


def test_a_kept_process_that_writes_nothing_in_its_wait_is_ended_and_answers_no_member(tmp_path, monkeypatch):
    """The wait is read on a0's clock, which here passes an hour and more at every reading: the process is ended, by its
    PID, before its first line is read."""
    ticks = iter(range(10**6))
    monkeypatch.setattr(a0, "monotonic", lambda: next(ticks) * 2 * a0.MEMBER_WAIT)
    lean, members, kept = kept_window(tmp_path)
    got = [a0.lean_verify(lean, st, 8, out, zk=True, members=kept) for st, out in members]
    assert [v["accepted"] for v in got] == [False] * 3
    assert kept.ended == "the kept verifier wrote nothing for 3600 s before member 0's MEMBER line (it exited -9)"
    assert kept.close() == -9
