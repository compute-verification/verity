"""Forged proofs: the Lean verifier refuses a well-formed `--zk` proof of a false statement in the proof's own checks, and
accepts the honest one.

A tampered output word is refused first by M0's prover, whose self-check finds the witness false and sends no proof, so
Lean refuses that session for want of its proof files (`session files: ...`), whatever its proof checks are. These sessions
come from a prover that skipped that self-check (`FC_ZK_SELF_CHECK=skip`) and sent its proofs of the false witness. They are
M0 `--zk` sessions of one small A8 unit, LlBf16DotprodDot_v1 at K=128, staged by `UnitStager` bound to a template query's
partition, with four instances and every unit drawn (subset:4). Bit 0 of the first output word is flipped before
commitment, on every instance (`wrong-output-word`) and on one alone (`wrong-output-word-one-unit`). Each forged proof
parses whole as a `--zk` proof and is the one its record digests, so only a proof check can refuse it. The inner proof's
batched constraint does, as M0's server did live. The honest session, on the same circuit and draw law, is accepted.

It is the one test in `check` that hands the Lean verifier a proof of a false statement that the prover made in full:
`test_lean_zk.py`'s mutants edit an honest proof, whose statement is true. So it catches a verifier that accepts a proof
whose transcript is consistent and whose witness is false, which those can't.

The sessions are a fetched fixture (`fixtures/artifacts.json`), written by `benchmarks/one_stage/forged_proofs.sh` (run
r20261027-022307-6c8f), which `suites.py` fetches before the suite runs, so a missing fixture fails these tests rather than
skipping them."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
VERIFIER = Path(__file__).resolve().parents[1] / "verifier"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
#: ``stage/`` (circuit.txt, partition.json, program.json), per case ``<case>/pub-4.bin`` and ``<case>/session/``, and
#: ``cases.json`` (each case's flips, the prover's LIVE line, M0's verdict and Lean's at generation): `art:31c671f6`
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "forged-proofs"
STATEMENT = "verity/flock-circuit"
N = 4
FORGED = ("wrong-output-word", "wrong-output-word-one-unit")
#: the inner proof's refusal of a witness whose output rows are not the unit's outputs
BATCHED = "zk inner: the batched constraint fails (⟨c, Y⟩ ≠ t + β τ)"

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    assert (FIXTURE / "cases.json").is_file(), (
        f"{FIXTURE} is not fetched: `uv run research data fetch-fixtures verity/ml/flock/tests/fixtures/forged-proofs`")
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    sys.path.insert(0, str(VERIFIER))
    import ci

    return ci.canonical_tree(FIXTURE, tmp_path_factory.mktemp("store") / "forged-proofs")


def _verify(root: Path, case: str, session: Path | None = None) -> dict:
    s = root / "stage"
    p = subprocess.run([str(BIN), "verify", "--statement", STATEMENT, "--circuit", str(s / "circuit.txt"), "--public",
                        str(root / case / f"pub-{N}.bin"), "--partition", str(s / "partition.json"), "--program",
                        str(s / "program.json"), "--zk", "--session", str(session or root / case / "session")],
                       capture_output=True, text=True)
    [v] = [json.loads(ln[8:]) for ln in p.stdout.splitlines() if ln.startswith("VERDICT ")]
    return v


def _proofs(session: Path) -> dict[str, Path]:
    rec = json.loads((session / "session.json").read_text())
    return {e["stream"]: session / f"{e['stream'].replace('/', '.')}.bin" for e in rec["proof_sha512"]}


@pytest.mark.slow
@pytest.mark.parametrize("case", FORGED)
def test_a_forged_session_holds_well_formed_proofs_of_its_false_statement(store, case):
    """The prover sent every proof its record names, each the bytes the record digests and a whole `--zk` proof with the
    honest proof's layout, field for field, of a public file that is not the honest one; M0's server refused it in the inner
    proof."""
    sys.path.insert(0, str(VERIFIER))
    import redigest

    got = json.loads((store / "cases.json").read_text())["cases"][case]
    assert got["flip"] and got["m0"]["accepted"] is False and got["m0"]["why"] == BATCHED, got
    assert (store / case / f"pub-{N}.bin").read_bytes() != (store / "honest" / f"pub-{N}.bin").read_bytes()
    rec = json.loads((store / case / "session" / "session.json").read_text())
    proofs, honest = _proofs(store / case / "session"), _proofs(store / "honest" / "session")
    assert set(proofs) == set(honest) and proofs, proofs
    for e in rec["proof_sha512"]:
        b, h = proofs[e["stream"]].read_bytes(), honest[e["stream"]].read_bytes()
        assert (len(b), hashlib.sha512(b).hexdigest()) == (e["bytes"], e["sha512"]), e["stream"]
        assert b != h and redigest.layout(b, zk=True) == redigest.layout(h, zk=True), e["stream"]


@pytest.mark.slow
def test_lean_accepts_the_honest_session_and_refuses_each_forged_proof_in_the_proof(store):
    """The honest drawn session is accepted; each forged one is refused by the inner proof's batched constraint, the reason
    M0's server gave live, never by a check of the session's files or setup."""
    cases = ("honest", *FORGED)
    with ThreadPoolExecutor(max_workers=len(cases)) as ex:
        got = dict(zip(cases, ex.map(lambda c: _verify(store, c), cases)))
    assert got["honest"]["accepted"], got["honest"]["why"]
    assert {c: (got[c]["accepted"], got[c]["why"]) for c in FORGED} == dict.fromkeys(FORGED, (False, BATCHED)), got


@pytest.mark.slow
def test_a_forged_session_without_its_proofs_is_refused_for_want_of_them(store, tmp_path):
    """The refusal A8's tampered member met, where the prover's self-check sent no proof: the session's files, before any
    proof check, a reason other than the forged proof's."""
    s = shutil.copytree(store / FORGED[0] / "session", tmp_path / "session")
    for f in _proofs(s).values():
        f.unlink()
    v = _verify(store, FORGED[0], s)
    assert not v["accepted"] and v["why"].startswith("session files: "), v["why"]
