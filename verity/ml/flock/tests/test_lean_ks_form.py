"""`one_stage.proofs.ks_form_why`, the Python mirror of `verify --zk`'s refusal (`Flock.KsScope.filesWhy`) of a
statement form outside the file preconditions the sampled audit's knowledge soundness takes (`SampledKSLiveV3`'s, and
`SampledKSLiveV5`'s at a shared-row public file; the planned `SampledKSLiveV4`, not stated yet, is to discharge V3's),
held to `flock-verify ks-form`, which runs that check alone, on statements the prover wrote:
M0's flat hidden-output sets, RoPE (its unit returns its output) and GEMM (its unit reads two input groups and a tail
stage returns the outputs, which both refuse, `SCOPE_WHY`), and the served registered RoPE stage (a shared-row public
file in the form `SampledKSLiveV5` covers, which both pass). Each case's answer is Lean's and the mirror's alike.

The hidden-output sets are a fetched fixture (`fixtures/artifacts.json`), as `test_lean_hidden_outputs.py` reads them; the
served stage is `test_lean_served_registered.py`'s row/v1 fixture, fetched from the evidence store."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from verity.core.protocols.sampled_proofs.one_stage import proofs as PF

HERE = Path(__file__).resolve().parent
LEAN = HERE.parents[2] / "core" / "service" / "flock"
VERIFIER = HERE.parent / "verifier"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
HIDDEN = HERE / "fixtures" / "hidden-outputs" / "flat"
FLAT = "verity/flock-circuit@f5df3bbc+seed-injection"
SERVED = "art:88544efa5af6f50ab24ad04a8e8d3118075e9f62898e41bbfb45ec3f3bc3d85b"
SCOPE = f"refused: {PF.SCOPE_WHY}: a statement form the soundness proof does not cover"

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


@pytest.fixture(scope="module")
def built():
    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)


def _lean(statement: str, stage: Path, public: str) -> str | None:
    tables = ["--tables", str(stage / "tables")] if (stage / "tables").exists() else []
    p = subprocess.run([str(BIN), "ks-form", "--statement", statement, "--circuit", str(stage / "circuit.txt"), "--public",
                        str(stage / public), *tables], capture_output=True, text=True)
    (line,) = [ln[len("KS-FORM "):] for ln in p.stdout.splitlines() if ln.startswith("KS-FORM ")]
    assert p.returncode == (0 if line == "ok" else 1), p
    return None if line == "ok" else line


@pytest.mark.parametrize("name, want", [("rope-head", None), ("gemm-coordinate", SCOPE)])
def test_lean_and_its_mirror_give_a_flat_hidden_output_set_one_answer(built, name, want):
    stage = HIDDEN / name / "stage"
    assert (stage / "circuit.txt").is_file(), (
        f"{stage} is not fetched: `uv run research data fetch-fixtures verity/ml/flock/tests/fixtures/hidden-outputs`")
    got = (_lean(FLAT, stage, "pub-4.bin"), PF.ks_form_why(stage / "circuit.txt", stage / "pub-4.bin"))
    assert got == (want, want), got


def test_lean_and_its_mirror_both_pass_the_served_registered_shared_row_file(built, tmp_path):
    sys.path.insert(0, str(VERIFIER))
    import ci

    try:
        root = ci.fetch_sessions(SERVED, tmp_path / "store")
    except (subprocess.CalledProcessError, OSError) as e:
        pytest.skip(f"the evidence store is not reachable: {e}")
    stage = root / "stage"
    got = (_lean(PF.STATEMENT, stage, "pub-32.bin"), PF.ks_form_why(stage / "circuit.txt", stage / "pub-32.bin"))
    assert got == (None, None), got
