"""The storage piece's smallest runs that reach `pous-run-protocol`, honest and dishonest, on a processor: the piece's
sessions over the tiny int8 checkpoint on the CPU, the parties played in one process (so nothing here certifies a
deployment), the signer the PoUS tests' own, and the Lean the built `pous-run-protocol` under `--test`, which takes that
signer's schemes where the deployed protocol takes the attester's. The honest case is accepted by it; the same case with
its trusted key another's is refused by it, by name."""
from __future__ import annotations

import hashlib
import json
import re

import pytest
from test_pous_served_piece import RUN_EXE, TEST_PROTOCOL, _run, run_exe, served  # noqa: F401  (fixtures)

from verity.protocols.resource_accounting.memory.pous.verifier import period

NOT_TRUSTED = "the session is signed by a key that is not a trusted memory challenger's"


def _digest(exe) -> str:
    """The SHA-512 of the Lean file the script of `run_exe` execs."""
    real = re.search(r"exec (\S+)", exe.read_text()).group(1)
    return hashlib.sha512(open(real, "rb").read()).hexdigest()


@pytest.mark.slow
def test_the_honest_case_is_accepted_and_the_same_case_under_another_key_is_refused_by_its_lean(
        served, tmp_path, monkeypatch, run_exe, capsys):
    answer, _ = _run(served, tmp_path, monkeypatch, case_exe=str(run_exe))
    assert answer["verdict"] == "accept", answer["reason"]
    doc = json.loads((served.out / "pous.json").read_text())
    assert doc["decided_by"]["program"] == RUN_EXE
    case = json.loads((served.private / "case.json").read_text())
    honest = period.serve(run_exe, [case])[0]
    assert honest["verdict"] == "accept" and honest["protocol"] == TEST_PROTOCOL
    other = case | {"params": case["params"] | {"trusted": ["00" * 64]}}
    refused = period.serve(run_exe, [other])[0]
    assert refused["verdict"] == "reject" and refused["reason"]
    with capsys.disabled():
        print("SMOKE storage honest:", {k: honest[k] for k in ("verdict", "protocol")},
              "dishonest (trusted key swapped):", {k: refused[k] for k in ("verdict", "reason")},
              "by", RUN_EXE, "sha512", _digest(run_exe)[:32], flush=True)
