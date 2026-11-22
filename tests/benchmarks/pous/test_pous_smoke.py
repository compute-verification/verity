"""The storage piece's smallest honest run that reaches `pous-run-protocol`, on a processor: the piece's
sessions over the tiny int8 checkpoint on the CPU, the parties played in one process (so nothing here certifies a
deployment), the signer the PoUS tests' own, and the Lean the built `pous-run-protocol` under `--test`, which takes that
signer's schemes where the deployed protocol takes the attester's. The honest case is accepted by it. It costs about a minute: what it catches that the suite's stand-in judge
cannot is a verdict that has stopped reaching the real executable, or a case the real one no longer reads as the piece writes
it."""
from __future__ import annotations

import hashlib
import json
import re

import pytest
import settings
from test_pous_served_piece import RUN_EXE, TEST_PROTOCOL, _run, run_exe, served  # noqa: F401  (fixtures)

from verity.protocols.accounting.memory.pous.verifier import period


def _digest(exe) -> str:
    """The SHA-512 of the Lean file the script of `run_exe` execs."""
    real = re.search(r"exec (\S+)", exe.read_text()).group(1)
    return hashlib.sha512(open(real, "rb").read()).hexdigest()


@pytest.mark.slow
def test_the_honest_case_is_accepted_by_its_lean(
        served, tmp_path, monkeypatch, run_exe):
    answer, _ = _run(served, tmp_path, monkeypatch, case_exe=str(run_exe))
    assert answer["verdict"] == "accept", answer["reason"]
    doc = json.loads((served.out / "pous.json").read_text())
    assert doc["decided_by"]["program"] == RUN_EXE
    case = json.loads((served.private / "case.json").read_text())
    honest = period.serve(run_exe, [case])[0]
    assert honest["verdict"] == "accept" and honest["protocol"] == TEST_PROTOCOL
    line = f"SMOKE storage honest: {({k: honest[k] for k in ('verdict', 'protocol')})} by {RUN_EXE} sha512 {_digest(run_exe)}"
    if out := settings.smoke_out():
        with open(out, "a") as f:
            f.write(line + "\n")

