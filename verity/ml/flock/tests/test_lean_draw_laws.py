"""Draw laws (PROTOCOL.md §16.15, D12): `verify` refuses a session drawn by a law no soundness claim draws by (a bernoulli
draw, a work draw with or without a closure) at setup, whatever flags it holds, and passes a stratified draw, and at
hidden outputs a subset draw, on to the session's checks. The statement's form is checked last, once every check has
passed, and a session without a draw is refused after that. The statement is
`test_lean_registered_reads.py`'s honest one (`stage`, every port registered): its program and partition are the ones a
work draw's law derives from, so a work draw sets up there, where any other statement refuses it at setup before its law
is read. Each session is a record that names its draw and no proof files: a draw that passes the law check sets up and
meets the session's files, which it has none of, and so does a session without a draw, in either mode, since its
refusal comes after every check (its own refusal shows on the statements whose form is covered: `test_lean_zk.py`,
`test_lean_live_os.py`, `test_lean_hidden_outputs.py`). Each draw is its own setup (about 10 s), and staging composes the
circuit (about 30 s), so the module is slow; it is the only test that reaches the law check with a work draw."""
from __future__ import annotations

import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest
from verity_flock.record_text import session_text

import test_lean_registered_reads as R

BIN = R.BIN
TEMPLATE = "RoPEHead_v1{D=64}"

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


def _law(why: str) -> str:
    return f"refused: {why}: no soundness claim draws by this law"


@pytest.fixture(scope="module")
def verdicts(tmp_path_factory):
    """Each draw's verdict by name, without `--zk` and with it, the two runs side by side."""
    s = R.stage(tmp_path_factory.mktemp("registered"))
    public = R._stage(s, "honest")
    d = tmp_path_factory.mktemp("laws")
    (d / "work.json").write_text(json.dumps({TEMPLATE: 1}))
    (d / "closure.json").write_text(json.dumps({"0": [1]}))
    work = ["--work", "2", "--work-table", str(d / "work.json")]

    def draw(*law: str) -> dict:
        p = subprocess.run([str(BIN), "draw", *s.bound, *law, "--stream", "00112233445566778899aabbccddeeff" * 2],
                           capture_output=True, text=True, check=True)
        return json.loads(p.stdout.removeprefix("DRAW "))

    widened = draw(*work, "--closure", str(d / "closure.json"))
    assert widened["units"] == [0, 4] and widened["closure"] == [1], widened
    draws = {"work": {**draw(*work), "units": [2, 6]}, "closure": widened,
             "bernoulli": {"law": "bernoulli", "population": 8, "p": "1/2", "units": [1, 6]},
             "subset": {"law": "subset", "population": 8, "k": 2, "units": [1, 6]},
             "stratified": draw("--stratified", "2"), "none": None}
    sessions = {}
    for name, law in draws.items():
        sessions[name] = d / name
        sessions[name].mkdir()
        (sessions[name] / "session.json").write_text(session_text({} if law is None else {"unit_draw": law}))
    flags = [*s.bound, "--registered", str(s.dir / "own.json"), *work, "--stratified", "2", "--closure", str(d / "closure.json")]

    def verify(zk: bool, names: list[str]) -> dict:
        p = subprocess.run([str(BIN), "verify", "--statement", R.STATEMENT, "--circuit", str(s.dir / "circuit.txt"),
                            "--public", str(public), *flags, *(["--zk"] if zk else []),
                            "--session", *(str(sessions[n]) for n in names)], capture_output=True, text=True, check=False)
        lines = [json.loads(x.split(" ", 1)[1]) for x in p.stdout.splitlines() if x.startswith("VERDICT ")]
        assert len(lines) == len(names) and p.returncode == 1, p.stderr
        return dict(zip(names, lines))

    with ThreadPoolExecutor(max_workers=2) as ex:
        plain = ex.submit(verify, False, list(draws))
        zk = ex.submit(verify, True, ["work", "closure", "bernoulli", "subset", "none"])
        return {False: plain.result(), True: zk.result()}


@pytest.mark.parametrize("zk", [False, True])
@pytest.mark.parametrize("name, why", [("work", "a work draw"), ("closure", "a work draw with a closure"),
                                       ("bernoulli", "a bernoulli draw")])
def test_no_claim_draws_by_these_laws_in_either_mode(verdicts, zk, name, why):
    """At the verifier's own K, work table, program, partition and closure map the work draws set up, and are refused for
    their law; so is a bernoulli draw."""
    v = verdicts[zk][name]
    assert not v["accepted"] and v["why"] == _law(why), v["why"]


def test_a_subset_draw_at_hidden_outputs_passes_the_law_in_either_mode(verdicts):
    """At hidden outputs a plain claim draws a subset (`SoundExecH`), as `--zk` does: the subset draw sets up and meets
    the session's files in either mode, as a stratified draw does without `--zk`. At public outputs, without `--zk`, it is
    refused for its law (`test_lean_session_tables.py`'s `SUBSET`)."""
    for zk in (False, True):
        assert verdicts[zk]["subset"]["why"].startswith("session files: "), (zk, verdicts[zk]["subset"]["why"])
    assert verdicts[False]["stratified"]["why"].startswith("session files: "), verdicts[False]["stratified"]["why"]


def test_a_session_without_a_draw_is_refused_after_every_check(verdicts):
    """No soundness claim covers a session without a draw (D12), and `verify` refuses one last, after every check has
    passed, so a session that fails a check keeps that check's reason: here, in either mode, its missing files."""
    for zk in (False, True):
        assert verdicts[zk]["none"]["why"].startswith("session files: "), (zk, verdicts[zk]["none"]["why"])
