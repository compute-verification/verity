"""The honest plain session: the compiled Lean verifier accepts a plain (non-zk) session that the unmodified prover made
in full against the rec proxy's coin server (`--coins os`), with the sample held to its own law (`--stratified 16`), and
refuses the same session with one coin changed.

It checks the compiled `flock-verify` only, on one record. It makes no Lean premise hold and proves nothing about the
records a theorem quantifies over.

The session is rope-head/neox-bf16/pair at n=16, m=25, k_log=22 under `hm96-sha512/v1`, every unit drawn, written by
`verity/ml/flock/pod/80-rec-inner.sh N=16` (run r20261026-212829-1d52, commit 4331ec922a), whose live verdicts are
`run.json`'s. It is a fetched fixture (`fixtures/artifacts.json`), which `suites.py` fetches before the suite runs, so
a missing fixture fails these tests rather than skipping them. It catches a verifier that refuses an honest plain
session in full, which `test_lean_forged_proofs.py`'s `--zk` sessions of four units can't."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
#: ``stage/`` (circuit.txt, pub-16.bin, partition.json, program.json, draw.json), ``session/`` (session.json and rep 0's
#: and rep 1's proofs) and ``run.json`` (the run, its verify command and its verdicts): `art:f9bfe23d`
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "honest-plain"
N = 16
STREAM = "circuit/rep0"

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


@pytest.fixture(scope="module")
def store():
    assert (FIXTURE / "run.json").is_file(), (
        f"{FIXTURE} is not fetched: `uv run research data fetch-fixtures verity/ml/flock/tests/fixtures/honest-plain`")
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    return FIXTURE


def _verify(root: Path, session: Path | None = None) -> dict:
    s = root / "stage"
    p = subprocess.run([str(BIN), "verify", "--statement", "verity/flock-circuit", "--circuit", str(s / "circuit.txt"),
                        "--public", str(s / f"pub-{N}.bin"), "--partition", str(s / "partition.json"), "--program",
                        str(s / "program.json"), "--stratified", str(N), "--coins", "os", "--session",
                        str(session or root / "session")], capture_output=True, text=True)
    [v] = [json.loads(ln[8:]) for ln in p.stdout.splitlines() if ln.startswith("VERDICT ")]
    return v


@pytest.mark.slow
def test_lean_accepts_the_honest_plain_session(store):
    """Accepted with no reason, as the run's Lean and upstream's replay accepted it live."""
    run = json.loads((store / "run.json").read_text())
    assert run["lean"]["accepted"] and run["replay"]["accepted"], run
    v = _verify(store)
    assert (v["accepted"], v["why"]) == (True, ""), v


@pytest.mark.slow
def test_lean_refuses_the_session_with_its_first_challenge_changed(store, tmp_path):
    """Rep 0's first coin after the binding round, its low bit flipped (`rec_inner.py forge --what challenge`): the
    zerocheck's final sum fails, as it did live, so the acceptance above rests on the proof's checks."""
    s = shutil.copytree(store / "session", tmp_path / "session")
    rec = json.loads((s / "session.json").read_text())
    rd = next(r for r in next(x for x in rec["streams"] if x["stream"] == STREAM)["rounds"] if r["k"] > 0 and r["n"])
    c = rd["coins"][0]
    rd["coins"][0] = c[:-1] + format(int(c[-1], 16) ^ 1, "x")
    (s / "session.json").write_text(json.dumps(rec))
    v = _verify(store, s)
    assert (v["accepted"], v["why"]) == (False, "zerocheck: SumcheckFinalFailed"), v
