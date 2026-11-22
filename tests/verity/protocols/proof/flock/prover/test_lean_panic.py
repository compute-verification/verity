"""Flock's distribution and its agreement scripts read a Lean executable's answer only where it wrote no panic line and
ended with a status its caller documents, and start it with the runtime's panic switch."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from verity_flock import lean_child as LC, rec_live as RL

POD = Path(__file__).resolve().parents[6] / "verity" / "protocols" / "proof" / "flock" / "prover" / "pod"
sys.path[:0] = [str(POD), str(POD.parent / "agreement")]
import agree as AG  # noqa: E402
import lean_run as LR  # noqa: E402
import lean_verdicts as LV  # noqa: E402
import qcall_agree as QA  # noqa: E402
import qword_agree as QW  # noqa: E402

PANIC_LINE = "PANIC at Flock.verify Flock.Verify:1:2: index out of bounds"
#: the case's stand-in's lines before its answer, its exit, and the refusal's words
CASES = {
    "panic": (f"print({PANIC_LINE!r}, file=sys.stderr)\n", 0, f"{LC.LEAN_PANIC}: {PANIC_LINE}"),
    "index out of bounds": ("print('Error: index out of bounds', file=sys.stderr)\n", 0,
                            f"{LC.LEAN_PANIC}: Error: index out of bounds"),
    "silent abort": ("", 134, f"{LC.LEAN_EXIT}: status 134 (abort)"),
}


def _fake(tmp_path: Path, body: str) -> Path:
    exe = tmp_path / "fake-lean"
    exe.write_text(f"#!{sys.executable}\nimport json, os, sys\n{body}")
    exe.chmod(0o755)
    return exe


def _answering(tmp_path: Path, case: str, answer: str) -> Path:
    """A stand-in that checks the panic switch, writes the case's lines, prints ``answer`` (an accept) and exits as the case
    says."""
    before, rc, _why = CASES[case]
    return _fake(tmp_path, f"""assert os.environ.get({LC.ABORT_ON_PANIC!r}) == "1"
{before}{'' if rc else answer}sys.exit({rc})
""")


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("home", [LC, LR], ids=["verity_flock", "agreement"])
def test_an_accept_beside_a_panic_or_an_abort_is_refused_by_name(tmp_path, home, case):
    exe = _answering(tmp_path, case, "print('VERDICT ' + json.dumps({'accepted': True}))\n")
    with pytest.raises(home.LeanPanic) as e:
        home.run([str(exe)])
    assert str(e.value) == CASES[case][2]


@pytest.mark.parametrize("home", [LC, LR], ids=["verity_flock", "agreement"])
def test_the_child_starts_with_the_panic_switch(tmp_path, monkeypatch, home):
    monkeypatch.setenv(home.ABORT_ON_PANIC, "0")
    exe = _fake(tmp_path, f"print(os.environ.get({home.ABORT_ON_PANIC!r}))\n")
    assert home.run([str(exe)]).stdout == "1\n"
    assert home.run([str(exe)], env={"PATH": "/usr/bin"}).stdout == "1\n"


@pytest.mark.parametrize("case", CASES)
def test_the_agreement_harnesses_refuse_by_name(tmp_path, case):
    exe = _answering(tmp_path, case, "print('QCALLPROGRAM ' + json.dumps({'ok': True}))\n"
                                     "print('EVALUATE ' + json.dumps({'file': 'f', 'ok': True}))\n")
    with pytest.raises(LR.LeanPanic, match=f"^{re.escape(CASES[case][2])}$"):
        QA.run(SimpleNamespace(lean=str(exe)), "qcall-program", "QCALLPROGRAM")
    with pytest.raises(LR.LeanPanic, match=f"^{re.escape(CASES[case][2])}$"):
        QW.lean(str(exe), ["f"])
    p = LR.start([str(exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p.stdin.close()
    with pytest.raises(LR.LeanPanic, match=f"^{re.escape(CASES[case][2])}$"):
        LR.ended(p, LR.REFUSES)


def test_a_harness_reads_a_documented_refusal(tmp_path):
    exe = _fake(tmp_path, "print('QCALLPROGRAM ' + json.dumps({'ok': False}))\nsys.exit(1)\n")
    assert QA.run(SimpleNamespace(lean=str(exe)), "qcall-program", "QCALLPROGRAM") == {"ok": False}


@pytest.mark.parametrize("case", CASES)
def test_the_live_zk_servers_answer_beside_a_panic_or_an_abort_is_refused_by_name(tmp_path, case):
    before, rc, why = CASES[case]
    exe = _fake(tmp_path, f"""assert sys.argv[1:] == ["live"] and os.environ.get({LC.ABORT_ON_PANIC!r}) == "1"
for line in sys.stdin:
    {before.strip() or 'pass'}
    sys.stderr.flush()
    if {rc}:
        sys.exit({rc})
    print(json.dumps({{"ok": True}}), flush=True)
""")
    with pytest.raises(ValueError, match=f"{re.escape(why)}$"):
        RL.LeanZkServer(exe, {})


def _live(tmp_path: Path, after: str, requests: int | None = None) -> Path:
    """A live ZK server that accepts the schedule and answers each request with nothing sent, then runs ``after``: at the
    end of its input, or after its ``requests``-th answer with its input still open."""
    return _fake(tmp_path, f"""assert sys.argv[1:] == ["live"] and os.environ.get({LC.ABORT_ON_PANIC!r}) == "1"
print(json.dumps({{"ok": True}}), flush=True)
n = 0
for line in iter(sys.stdin.readline, ""):
    print(json.dumps({{"sent": [], "salts": []}}), flush=True)
    n += 1
    if n == {requests}:
        break
{after}
""")


@pytest.mark.parametrize("after, why", [
    ("sys.exit(134)", f"{LC.LEAN_EXIT}: status 134 (abort)"),
    (f"print({PANIC_LINE!r}, file=sys.stderr)", f"{LC.LEAN_PANIC}: {PANIC_LINE}"),
], ids=["abort", "panic"])
def test_the_live_zk_servers_close_refuses_its_accepts_by_name(tmp_path, after, why):
    """A kept live ZK server that accepted a request and then aborted or wrote a panic line at the end of its input: its
    close raises the refusal by name, so the answers it gave stand on nothing."""
    lean = RL.LeanZkServer(_live(tmp_path, after), {})
    assert lean.step(b"\x00") == ([], [])
    with pytest.raises(ValueError, match=f"did not end as documented at its close: {re.escape(why)}$"):
        lean.close()


def test_the_live_zk_server_ending_between_requests_refuses_the_next_by_name(tmp_path):
    lean = RL.LeanZkServer(_live(tmp_path, "sys.exit(0)", requests=1), {})
    with pytest.raises(ValueError, match=f"{re.escape(LC.LEAN_EXIT)}: status 0$"):
        lean.step(b"\x00")
        lean.step(b"\x00")


def test_the_live_zk_server_ending_at_its_documented_status_closes(tmp_path):
    lean = RL.LeanZkServer(_live(tmp_path, "sys.exit(0)"), {})
    assert lean.step(b"\x00") == ([], [])
    lean.close()


@pytest.mark.parametrize("case", CASES)
def test_a_composition_accepted_beside_a_panic_or_an_abort_is_neither_accepted_nor_refused(tmp_path, case):
    exe = _answering(tmp_path, case, "print('REC-COMPOSE ' + json.dumps({'accepted': True}))\n")
    assert LV.compose(str(exe), tmp_path, []) == (None, CASES[case][2])


def _agreement(tmp_path, monkeypatch, case):
    exe = _answering(tmp_path, case, "for s in sys.argv[1:]:\n"
                                     "    print('VERDICT ' + json.dumps({'session': s, 'accepted': True, 'why': ''}))\n")
    session = tmp_path / "s"
    session.mkdir()
    (session / "case.json").write_text('{"expect": "accept", "live": {"accepted": true, "why": ""}}')
    monkeypatch.setattr(AG, "groups", lambda a: {("pub.bin", None): [str(session)]})
    monkeypatch.setattr(AG, "lean_cmd", lambda a, public, registered=None: [str(exe)])
    return exe, str(session)


@pytest.mark.parametrize("case", CASES)
def test_the_agreements_lean_verdict_beside_a_panic_or_an_abort_is_not_read(tmp_path, monkeypatch, case):
    _exe, session = _agreement(tmp_path, monkeypatch, case)
    a = SimpleNamespace(sessions=[session], upstream=AG.LIVE, zk=True, jobs=1, out=tmp_path)
    got, _up = AG.verdicts(a)
    assert got == {}
    assert a.lean_refused == [CASES[case][2]]
    assert CASES[case][2] in (tmp_path / "lean.err").read_text()


@pytest.mark.parametrize("case", CASES)
def test_the_agreement_step_fails_naming_the_refusal(tmp_path, monkeypatch, capsys, case):
    exe, session = _agreement(tmp_path, monkeypatch, case)
    out = tmp_path / "out"
    rc = AG.main(["--lean", str(exe), "--upstream", AG.LIVE, "--circuit", "c", "--public", "p", "--zk", "--out", str(out),
                  "--sessions", session], quiet=True)
    assert rc == 1
    assert f"agree: FAILED: {CASES[case][2]}" in capsys.readouterr().err.splitlines()


ACCEPT = "print('VERDICT ' + json.dumps({'accepted': True}))\n"


@pytest.mark.parametrize("case", [*CASES, "undocumented status"])
def test_a_shell_jobs_verdict_beside_a_panic_an_abort_or_an_undocumented_status_is_not_read(tmp_path, capsys, case):
    """`lean_run.py SET OUT -- CMD`, a shell job's launch (`zk_server_finish_agree.sh`'s `verify`): an accept beside a
    panic line, after an abort or at a status SET doesn't hold leaves OUT.out empty and exits 3, naming the refusal."""
    if case in CASES:
        exe, why = _answering(tmp_path, case, ACCEPT), CASES[case][2]
    else:
        exe = _fake(tmp_path, f"assert os.environ.get({LR.ABORT_ON_PANIC!r}) == '1'\n{ACCEPT}sys.exit(3)\n")
        why = f"{LR.LEAN_EXIT}: status 3"
    assert LR.main(["VERIFY_EXITS", str(tmp_path / "v"), "--", str(exe)]) == 3
    assert (tmp_path / "v.out").read_text() == "" and capsys.readouterr().err == f"{why}\n"


def test_a_shell_job_reads_a_documented_refusal(tmp_path):
    exe = _fake(tmp_path, "print('VERDICT ' + json.dumps({'accepted': False}))\nprint('setup: 1 ms', file=sys.stderr)\n"
                          "sys.exit(1)\n")
    assert LR.main(["VERIFY_EXITS", str(tmp_path / "v"), "--", str(exe)]) == 0
    assert (tmp_path / "v.out").read_text().startswith("VERDICT ") and (tmp_path / "v.err").read_text() == "setup: 1 ms\n"
    assert LR.main(["NO_SUCH_SET", str(tmp_path / "v"), "--", str(exe)]) == 2


def test_the_finish_agreements_lean_verify_starts_under_lean_run():
    lines = (POD.parent / "agreement" / "zk_server_finish_agree.sh").read_text().splitlines()
    at = [i for i, x in enumerate(lines) if '"$LEAN" verify' in x]
    assert at and all("/agreement/lean_run.py\" VERIFY_EXITS " in lines[i - 1] and lines[i - 1].endswith("-- \\") for i in at)
