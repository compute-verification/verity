"""A served tree registered as a port, end to end (`one_stage.served.register_served`, `verity_flock.served_registered`): a
RoPE d64 statement whose 32 units each read their `cs` row from the served tree `cs_table`, proved by C-Flock in one
loopback `--zk` session under a subset draw of 16 (`pod/89-served-registered.sh`, `FIXTURE`). The verifier writes its own
`--registered` file from the tree as the service holds it (`served.json`), with `register_served` on its own program, and
`flock-verify verify --zk --registered` accepts the session on it and refuses it on the same file with the tree's root
changed. Each verify sets up an m = 25 table (about 10 s), so the module is slow; it is the one test of `register_served`'s
file on a proved session, which the Python round trip (`verity/core/protocols/sampled_proofs/tests/test_served.py`) can't
reach."""
from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from verity.ml.circuits.codec import canonical_json, encode_program
from verity.core.protocols.sampled_proofs.one_stage.served import ServedTree, register_served

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
VERIFIER = Path(__file__).resolve().parents[1] / "verifier"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
#: `pod/89-served-registered.sh`'s fixture: stage/ (circuit.txt, pub-32.bin, program.json, partition.json, served.json,
#: registered.json, tables/), session/ (the drawn --zk session's record and proofs), live.json (the prover's LIVE record)
FIXTURE = "art:PENDING"
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


def _verify(fx: Path, registered: Path) -> dict:
    st = fx / "stage"
    p = subprocess.run([str(BIN), "verify", "--zk", "--statement", STATEMENT, "--circuit", str(st / "circuit.txt"), "--public",
                        str(st / "pub-32.bin"), "--partition", str(st / "partition.json"), "--program",
                        str(st / "program.json"), "--registered", str(registered), "--session", str(fx / "session")],
                       capture_output=True, text=True, check=False)
    (line,) = [json.loads(x.split(" ", 1)[1]) for x in p.stdout.splitlines() if x.startswith("VERDICT ")]
    return line


def test_a_session_verified_on_the_file_register_served_wrote_is_accepted_and_a_changed_root_refused(fx, tmp_path):
    from verity_flock import served_registered as SR

    program, query = SR.program()
    assert canonical_json(encode_program(program)) == (fx / "stage" / "program.json").read_bytes()
    tree = ServedTree(**json.loads((fx / "stage" / "served.json").read_text()))
    sr = register_served(program, query, [tree])
    assert sr.file_bytes() == (fx / "stage" / "registered.json").read_bytes()
    ok = _verify(fx, sr.write(tmp_path / "registered.json"))
    assert ok["accepted"] is True, ok
    assert ok["registered"] == {"cs": {"value": SR.VALUE, "root": tree.root}}
    moved = register_served(program, query, [dataclasses.replace(tree, root="00" * 64)])
    bad = _verify(fx, moved.write(tmp_path / "registered-root.json"))
    assert bad["accepted"] is False and f"does not open {SR.VALUE}'s registered root" in bad["why"], bad
