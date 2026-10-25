"""A plain hidden-output session, end to end (PROTOCOL.md §16.15, `plain_hidden.py`): a RoPE d64 statement with hidden
outputs whose 32 units read their own rows (no shared rows, no registered value), proved by C-Flock in one loopback
session without `--zk` under a subset draw of 16 on OS coins (`pod/91-plain-hidden.sh`, `FIXTURE`). `flock-verify verify
--coins os` accepts it with the verifier's own program and partition, the form `flock_verify_sound`'s plain hidden-output
branch covers at the subset law (`FailClosed.SoundExecH` at its subset draw design), and refuses it with a byte of a
proof changed, or with its record's draw a bernoulli draw of the same units (no claim draws by that law). It is the one
test of a plain session at the subset law on the verifier of record; the `#guard`s of `Flock/Test/PlainHidden.lean` hold
the rule itself. Each verify sets up an m = 22 table, so the module is slow."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from verity.ml.circuits.codec import canonical_json, encode_program

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
VERIFIER = Path(__file__).resolve().parents[1] / "verifier"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
#: `pod/91-plain-hidden.sh`'s fixture (run r20261025-033320-4832): stage/ (circuit.txt, pub-32.bin, program.json,
#: partition.json), session/ (the drawn plain session's record and proofs), live.json (the prover's LIVE record)
FIXTURE = "art:06435437"
STATEMENT = "verity/flock-circuit"

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    pytest.importorskip("verity_numerical")
    sys.path.insert(0, str(VERIFIER))
    import ci

    try:
        root = ci.fetch(FIXTURE, tmp_path_factory.mktemp("store"))
    except (subprocess.CalledProcessError, OSError) as e:
        pytest.skip(f"the evidence store is not reachable: {e}")
    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)
    return root


def _verify(fx: Path, session: Path) -> dict:
    st = fx / "stage"
    p = subprocess.run([str(BIN), "verify", "--statement", STATEMENT, "--coins", "os", "--circuit", str(st / "circuit.txt"),
                        "--public", str(st / "pub-32.bin"), "--partition", str(st / "partition.json"), "--program",
                        str(st / "program.json"), "--session", str(session)], capture_output=True, text=True, check=False)
    (line,) = [json.loads(x.split(" ", 1)[1]) for x in p.stdout.splitlines() if x.startswith("VERDICT ")]
    return line


def test_the_plain_session_at_a_subset_draw_is_accepted(fx):
    import plain_hidden as PH

    program, _ = PH.program()
    assert canonical_json(encode_program(program)) == (fx / "stage" / "program.json").read_bytes()
    draw = json.loads((fx / "session" / "session.json").read_text())["unit_draw"]
    assert (draw["law"], draw["population"], draw["k"], len(draw["units"])) == ("subset", 32, 16, 16)
    assert json.loads((fx / "live.json").read_text())["accepted"] is True
    ok = _verify(fx, fx / "session")
    assert ok["accepted"] is True, ok
    assert ok["program_sha512"] == json.loads((fx / "stage" / "partition.json").read_text())["program"]


def test_a_changed_proof_or_an_uncovered_law_is_refused(fx, tmp_path):
    flipped = shutil.copytree(fx / "session", tmp_path / "flipped")
    rep = sorted(flipped.glob("*.bin"))[0]
    b = bytearray(rep.read_bytes())
    b[len(b) // 2] ^= 1
    rep.write_bytes(bytes(b))
    bad = _verify(fx, flipped)
    assert bad["accepted"] is False and not bad["why"].startswith("setup:"), bad
    bern = shutil.copytree(fx / "session", tmp_path / "bernoulli")
    rec = json.loads((bern / "session.json").read_text())
    d = rec["unit_draw"]
    rec["unit_draw"] = {"law": "bernoulli", "population": d["population"], "p": "1/2", "units": d["units"]}
    (bern / "session.json").write_text(json.dumps(rec))
    bad = _verify(fx, bern)
    assert bad == {**bad, "accepted": False, "why": "refused: a bernoulli draw: no soundness claim draws by this law"}, bad
