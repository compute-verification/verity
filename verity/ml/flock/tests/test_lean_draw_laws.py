"""Draw laws (PROTOCOL.md §16.15, D12): `verify` refuses a session drawn by a law no soundness claim draws by (a bernoulli
draw, a work draw with or without a closure, a subset draw without `--zk` at public outputs) at setup, whatever flags it
holds, and passes a stratified draw, and under `--zk` a subset draw, on to the session's checks. The statement's form is
checked last, once every check has passed, and a session without a draw is refused after that. The sessions are the registered-reads set's honest
session (`FIXTURE`, vectors.json's set 16) with its record's draw replaced: its statement is the corpus's one whose program
and partition a work draw's law derives from, so a work draw sets up there, where any other statement refuses it at setup
before its law is read. A draw that passes the law check sets up and meets the session's checks, which the edited record
fails at S2: its Hello is the undrawn session's, and under `--zk` not a `--zk` session's. Its public file has shared rows,
which the form check refuses, so the session as recorded, without a draw and without `--zk`, passes every check and meets
that refusal. Upstream's seed-injection build proved it, with no zk patch, so its plain files are upstream's form, without
the `zk_level0 = None` tag the deployed prover writes, and the run without `--zk` reads them under `--upstream-plain`
(PROTOCOL.md §8.1). Each draw is its own setup (about 10 s), so the module is slow; it is the only test that reaches the law
check with a work draw."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from verity_flock.record_text import session_text

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
VERIFIER = Path(__file__).resolve().parents[1] / "verifier"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
FIXTURE = "art:bf70d8eb"
STATEMENT = "verity/flock-circuit@967b8d06+seed-injection"
TEMPLATE = "RoPEHead_v1{D=64}"
SHARED = "refused: a shared-row public file (header shared_rows): a statement form the soundness proof does not cover"

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


def _law(why: str) -> str:
    return f"refused: {why}: no soundness claim draws by this law"


@pytest.fixture(scope="module")
def verdicts(tmp_path_factory):
    """Each draw's verdict by name, without `--zk` and with it, the two runs side by side."""
    sys.path.insert(0, str(VERIFIER))
    import ci

    try:
        root = ci.fetch_sessions(FIXTURE, tmp_path_factory.mktemp("store"))
    except (subprocess.CalledProcessError, OSError) as e:
        pytest.skip(f"the evidence store is not reachable: {e}")
    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)
    d = tmp_path_factory.mktemp("laws")
    (d / "work.json").write_text(json.dumps({TEMPLATE: 1}))
    (d / "closure.json").write_text(json.dumps({"0": [1]}))
    bound = ["--program", str(root / "stage" / "program.json"), "--partition", str(root / "stage" / "partition.json")]
    work = ["--work", "2", "--work-table", str(d / "work.json")]

    def draw(*law: str) -> dict:
        p = subprocess.run([str(BIN), "draw", *bound, *law, "--stream", "00112233445566778899aabbccddeeff" * 2],
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
        s = shutil.copytree(root / "vectors" / "honest", d / name)
        rec = json.loads((s / "session.json").read_text())
        (s / "session.json").write_text(session_text({**rec, "unit_draw": law}))
        sessions[name] = s
    flags = [*bound, "--registered", str(root / "registered" / "own.json"), *work, "--stratified", "2",
             "--closure", str(d / "closure.json")]

    def verify(zk: bool, names: list[str]) -> dict:
        p = subprocess.run([str(BIN), "verify", "--statement", STATEMENT, "--circuit", str(root / "stage" / "circuit.txt"),
                            "--public", str(root / "registered" / "pub-8.bin"), *flags, "--zk" if zk else "--upstream-plain",
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


def test_a_subset_draw_needs_zk(verdicts):
    """Without `--zk` a subset draw is refused for its law; under `--zk` it sets up and meets the session's checks, as a
    stratified draw does without `--zk`."""
    assert verdicts[False]["subset"]["why"] == _law("a subset draw without --zk")
    assert verdicts[True]["subset"]["why"].startswith("S2/R7"), verdicts[True]["subset"]["why"]
    assert verdicts[False]["stratified"]["why"].startswith("S2/R7"), verdicts[False]["stratified"]["why"]


def test_a_session_without_a_draw_is_refused_after_every_check(verdicts):
    """No soundness claim covers a session without a draw (D12), and `verify` refuses one last, after every check has
    passed and after the form check, so a session that fails a check keeps that check's reason: without `--zk`, the
    statement's form; under `--zk`, Hello's coin check. Its own refusal shows on the statements whose form is covered
    (`test_lean_zk.py`, `test_lean_live_os.py`, `test_lean_hidden_outputs.py`)."""
    assert verdicts[False]["none"]["why"] == SHARED
    assert verdicts[True]["none"]["why"].startswith("S2/R7"), verdicts[True]["none"]["why"]
