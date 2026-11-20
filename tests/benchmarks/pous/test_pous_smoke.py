"""The storage piece's smallest runs that reach `pous-run-protocol`, honest and dishonest, on a processor: the piece's
sessions over the tiny int8 checkpoint on the CPU, the parties played in one process (so nothing here certifies a
deployment), the signer the PoUS tests' own, and the Lean the built `pous-run-protocol` under `--test`, which takes that
signer's schemes where the deployed protocol takes the attester's. The honest case is accepted by it; the same case with
its trusted key another's is refused by it, by name."""
from __future__ import annotations

import hashlib
import json
import os
import re

import pytest
from test_pous_served_piece import RUN_EXE, TEST_PROTOCOL, _run, case_exe, run_exe, served  # noqa: F401  (fixtures)

from verity.protocols.resource_accounting.memory.pous.verifier import period

NOT_TRUSTED = "the session is signed by a key that is not a trusted memory challenger's"


def _digest(exe) -> str:
    """The SHA-512 of the Lean file the script of `run_exe` execs."""
    real = re.search(r"exec (\S+)", exe.read_text()).group(1)
    return hashlib.sha512(open(real, "rb").read()).hexdigest()


@pytest.mark.slow
def test_the_honest_case_is_accepted_and_the_same_case_under_another_key_is_refused_by_its_lean(
        served, tmp_path, monkeypatch, run_exe):
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
    line = (f"SMOKE storage honest: {({k: honest[k] for k in ('verdict', 'protocol')})} dishonest (trusted key swapped): "
            f"{({k: refused[k] for k in ('verdict', 'reason')})} by {RUN_EXE} sha512 {_digest(run_exe)}")
    if out := os.environ.get("SMOKE_OUT"):
        with open(out, "a") as f:
            f.write(line + "\n")


@pytest.mark.slow
def test_the_bandwidth_audits_honest_session_is_accepted_and_two_dishonest_are_refused_by_its_lean(tmp_path, case_exe):
    """`bandwidth.py session` over the toy's store on the CPU, against `p2_party_at_leaf.py` in the auditor gate's shape
    (`python -S -E`, its close signed at the deployed parameters under a test key made without a keygen): the parties in one
    machine, so it certifies no deployment. The Lean is the built `poub-p2-protocol`. The honest session is accepted by it;
    a prover that skips a layer is rejected by it for its receipts; and the same honest session under a party trusting another
    key is refused by it, by name."""
    import hashlib as H
    import sys
    from pathlib import Path

    import test_pous_p2_bandwidth as T

    reg = T.BT.toy(tmp_path / "toy")
    (tmp_path / "challenger").mkdir()
    T.mc.SessionLog.create(tmp_path / "challenger" / "sessions")
    lines = []

    def run(name: str, trusted: str, more: list[str], expect: str) -> dict:
        session = H.sha512(name.encode()).hexdigest()[:64]
        party = (f"cd {T.ROOT} && exec {sys.executable} -S -E {Path(T.__file__).resolve().parent / 'p2_party_at_leaf.py'} "
                 f"--device {tmp_path / ('party-' + name)} --listen 127.0.0.1:0 "
                 f"--registration {tmp_path / 'toy' / 'registration.json'} {T.held_to(1, 1, 8)} "
                 f"--challenger {tmp_path / 'challenger'} --session {session} --trusted-sha512 {trusted}")
        private = tmp_path / "private" / name
        assert T.B.main(["session", "--store", str(tmp_path / "toy" / "store"), "--layout", str(tmp_path / "toy" / "layout.json"),
                         "--node", reg.node.decode(), "--party", party, "--party-host", "127.0.0.1", "--device", "cpu",
                         "--ref-decode", "--samples", "8", "--windows", "1", "--passes", "1", "--window-s", "0.001",
                         "--deadline-s", "0.001", "--workers", "1", "--out", str(tmp_path / name), "--expect", expect,
                         "--case-dir", str(private), "--case-exe", str(case_exe), *more]) == 0
        answer = json.loads((private / T.p2_case.ANSWER).read_text())
        lines.append(f"SMOKE bandwidth {name}: {({k: answer.get(k) for k in ('verdict', 'reason', 'protocol')})}")
        return answer

    honest = run("honest", T.AT_LEAF, [], "accept")
    assert honest["verdict"] == "accept"
    cheat = run("cheat", T.AT_LEAF, ["--skip-layer", "1"], "reject")
    assert cheat["verdict"] == "reject"
    swapped = run("swapped", "00" * 64, [], "accept")
    assert swapped["verdict"] in ("reject", "undecided") and swapped["reason"]
    if out := os.environ.get("SMOKE_OUT"):
        with open(out, "a") as f:
            f.write("\n".join(lines) + f"\nby poub-p2-protocol sha512 {hashlib.sha512(open(case_exe, 'rb').read()).hexdigest()}\n")
