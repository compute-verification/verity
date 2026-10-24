"""The proof service's own run of C-Flock's Lean verifier of record (`one_stage.proofs`), on a pinned `--zk --registered`
session: a RoPE d64 statement of 32 units, each reading its `cs` row from a registered served tree, proved by C-Flock in
one loopback session under a subset draw of 16 (`verity/ml/flock/pod/89-served-registered.sh`'s row/v1 fixture). The
verifier's own program, partition and `--registered` file are the fixture's `stage/`, the ones
`verity/ml/flock/tests/test_lean_served_registered.py` holds to `register_served`.

Each verify sets up an m = 25 table (about 10 s), so the module is slow. It is the one test that a protocol's verdict comes
from core's own verifier run: the honest session is accepted with its program, registered finding and units read off that
run, a circuit file other than the committed one is refused before the verifier runs, and tampered proof bytes are
rejected by the verifier itself, with the record's proof digests kept or rewritten to match."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from verity.core.protocols.sampled_proofs.one_stage import proofs as PF

#: `test_lean_served_registered.py`'s row/v1 fixture, fetched from the evidence store as that test fetches it (its records
#: re-rendered in the canonical text the verifier requires)
FIXTURE = "art:eeb1ed6ad88c683f6dfd3b0ffd2e9941c47f2fcb6f39c097e8f2f20720c6ce4a"
VERIFIER = Path(__file__).resolve().parents[4] / "ml" / "flock" / "verifier"
#: the SHA-512 of the fixture's circuit file, as an auditor would commit it (its public file's `circuit_sha512` pin)
CIRCUIT = ("7b9e46f755bced2c6e00a7706f3fca90631864ca22a2d08b51c8b82260a7bd9a"
           "9ae7a191c0eabe2993a6fbf686b832c440f2a78dad65fa3b13b3cf5bd47284e0")
CS_ROOT = ("2deda0ffb88e7c0e56693ec7e41b16c21bca80726956209c5c56505c922b5ac1"
           "a176ca20f9238c9298ef77b0678e94681352a92ec2f95059a9f92e6a56054001")
#: the instances the session's own draw drew, which are the units, the public file's indices being 0..31
DRAWN = (1, 2, 3, 6, 8, 10, 11, 13, 16, 17, 21, 22, 24, 26, 28, 30)

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


@pytest.fixture(scope="module")
def fx(tmp_path_factory) -> Path:
    sys.path.insert(0, str(VERIFIER))
    import ci

    try:
        root = ci.fetch_sessions(FIXTURE, tmp_path_factory.mktemp("store"))
    except (subprocess.CalledProcessError, OSError) as e:
        pytest.skip(f"the evidence store is not reachable: {e}")
    PF.executable()
    return root


def _verify(fx: Path, session: Path, circuit: str = CIRCUIT):
    st = fx / "stage"
    return PF.verify(PF.Proof(st / "circuit.txt", st / "pub-32.bin", (session,)), circuit_sha512=circuit,
                     program=(st / "program.json").read_bytes(),
                     partition=json.loads((st / "partition.json").read_text()),
                     registered=json.loads((st / "registered.json").read_text()))


def test_the_honest_session_is_accepted_by_cores_own_verifier_run(fx):
    got, why = _verify(fx, fx / "session")
    assert why is None, why
    (f,) = got
    partition = json.loads((fx / "stage" / "partition.json").read_text())
    assert f.accepted is True and f.why == "", f
    assert f.program == partition["program"]
    assert f.registered == {"cs": {"value": "cs_table", "root": CS_ROOT}}
    assert f.units == DRAWN


def test_a_circuit_file_other_than_the_committed_one_is_refused_before_the_verifier_runs(fx, monkeypatch):
    def never():
        raise AssertionError("the verifier ran on a circuit file the auditor didn't commit")

    monkeypatch.setattr(PF, "executable", never)
    got, why = _verify(fx, fx / "session", circuit="ab" * 64)
    assert got is None and why[0] == "circuit" and "is not the one the auditor committed" in why[1], why


def _tampered(fx: Path, to: Path, rehash: bool) -> Path:
    s = shutil.copytree(fx / "session", to / "session")
    rep = s / "circuit.rep0.bin"
    b = bytearray(rep.read_bytes())
    b[len(b) // 2] ^= 1
    rep.write_bytes(bytes(b))
    if rehash:
        rec = json.loads((s / "session.json").read_text())
        for p in rec["proof_sha512"]:
            if p["stream"] == "circuit/rep0":
                p["sha512"] = hashlib.sha512(bytes(b)).hexdigest()
        (s / "session.json").write_text(json.dumps(rec))
    return s


@pytest.mark.parametrize("rehash", [False, True], ids=["record-kept", "record-rehashed"])
def test_tampered_proof_bytes_are_rejected_by_the_verifier_itself(fx, tmp_path, rehash):
    """One bit of the first rep's proof flipped: with the record's proof digest kept the verifier refuses the file the
    record doesn't bind, and with it rewritten to the flipped bytes it refuses the proof itself. Either way the finding is
    the verifier's reject, with no units proved."""
    got, why = _verify(fx, _tampered(fx, tmp_path, rehash))
    assert why is None, why
    (f,) = got
    assert f.accepted is False and f.why and f.units == (), f


def test_missing_proof_files_are_refused(fx, tmp_path):
    st = fx / "stage"
    got, why = PF.verify(PF.Proof(st / "circuit.txt", st / "pub-32.bin", (tmp_path / "none",)), circuit_sha512=CIRCUIT,
                         program=b"{}", partition={}, registered={})
    assert got is None and why[0] == "proof", why
