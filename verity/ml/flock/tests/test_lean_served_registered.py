"""A served tree registered as a port, end to end (`one_stage.served.register_served`, `served_registered.py`): a
RoPE d64 statement whose 32 units each read their `cs` row from the served tree `cs_table`, proved by C-Flock in one
loopback `--zk` session under a subset draw of 16 (`pod/89-served-registered.sh`, `FIXTURES`), once for each row format a
served tree may be: `row/v1`, and `row/v2` and `row-seg/v1` bit rows (the circuit's rows bit rows of their words, `cs` a
v2 or a segmented row). The verifier writes its own `--registered` file from the tree as the service holds it
(`served.json`), with `register_served` on its own program, and `flock-verify verify --zk --registered` accepts the session
on it, its verdict naming that program's digest (the partition's), and refuses it on the same file with the tree's root changed, or a bit row's port with its words or width changed.
A plain session of the row/v1 stage (`PLAIN`: no `--zk`, the verifier's own stratified draw) is accepted on that file too,
and refused with the root changed and without `--registered`.
Each verify sets up an m = 25 table (about 10 s), so the module is slow; it is the one test of `register_served`'s file on
a proved session, which the Python round trip (`verity/core/protocols/sampled_proofs/tests/test_served.py`) can't
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
from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity.core.protocols.sampled_proofs.one_stage.served import ServedTree, register_served

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
VERIFIER = Path(__file__).resolve().parents[1] / "verifier"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
#: `pod/89-served-registered.sh`'s fixture for each row format (`SCHEMA=`): stage/ (circuit.txt, pub-32.bin, program.json,
#: partition.json, served.json, registered.json, tables/), session/ (the drawn --zk session's record and proofs), live.json
#: (the prover's LIVE record)
FIXTURES = {
    G.SCHEMA: "art:88544efa5af6f50ab24ad04a8e8d3118075e9f62898e41bbfb45ec3f3bc3d85b",
    G.SCHEMA_V2: "art:eedab8e57d1f9c380472fe263ab38d00dc8ec68f1a7e94b035de01d0cb2416db",
    G.SCHEMA_SEG: "art:d42ad7aedb6b15f351dcddd65fd51ab0e0f91afbbfa48451d192d87273b926e6",
}
#: `pod/89-served-registered.sh ZK=0`'s fixture (row/v1): the same stage, with the verifier's own stratified draw of `PLAIN_K`
#: (stage/draw.json), and session/ a plain session (no --zk, OS coins) drawn by it; vectors/ is vectors.json's live set
PLAIN = "art:88fd3f12b8f363acb27415b58e38e43f04431ce61e198b081d25c33f5b8524b9"
PLAIN_K = 16
STATEMENT = "verity/flock-circuit"

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


@pytest.fixture(scope="module")
def built():
    pytest.importorskip("verity_numerical")
    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)


@pytest.fixture(scope="module", params=list(FIXTURES), ids=["v1", "v2", "seg"])
def fx(request, built, tmp_path_factory):
    sys.path.insert(0, str(VERIFIER))
    import ci

    try:
        root = ci.fetch_sessions(FIXTURES[request.param], tmp_path_factory.mktemp("store"))
    except (subprocess.CalledProcessError, OSError) as e:
        pytest.skip(f"the evidence store is not reachable: {e}")
    return request.param, root


def _verify(fx: Path, registered: Path | None, mode: tuple[str, ...] = ("--zk",)) -> dict:
    st = fx / "stage"
    reads = ["--registered", str(registered)] if registered else []
    p = subprocess.run([str(BIN), "verify", *mode, "--statement", STATEMENT, "--circuit", str(st / "circuit.txt"), "--public",
                        str(st / "pub-32.bin"), "--partition", str(st / "partition.json"), "--program",
                        str(st / "program.json"), *reads, "--session", str(fx / "session")],
                       capture_output=True, text=True, check=False)
    (line,) = [json.loads(x.split(" ", 1)[1]) for x in p.stdout.splitlines() if x.startswith("VERDICT ")]
    return line


def test_a_session_verified_on_the_file_register_served_wrote_is_accepted_and_a_changed_root_refused(fx, tmp_path):
    import served_registered as SR

    schema, fx = fx
    program, query = SR.program()
    assert canonical_json(encode_program(program)) == (fx / "stage" / "program.json").read_bytes()
    tree = ServedTree(**json.loads((fx / "stage" / "served.json").read_text()))
    assert tree.row_schema == schema
    sr = register_served(program, query, [tree])
    assert sr.file_bytes() == (fx / "stage" / "registered.json").read_bytes()
    assert sr.ports["cs"].get(G.SCHEMA_KEY, G.SCHEMA) == schema
    assert sr.partition == json.loads((fx / "stage" / "partition.json").read_text())
    ok = _verify(fx, sr.write(tmp_path / "registered.json"))
    assert ok["accepted"] is True, ok
    assert ok["registered"] == {"cs": {"value": SR.VALUE, "root": tree.root}}
    assert ok["program_sha512"] == sr.partition["program"]
    moved = register_served(program, query, [dataclasses.replace(tree, root="00" * 64)])
    bad = _verify(fx, moved.write(tmp_path / "registered-root.json"))
    assert bad["accepted"] is False and f"does not open {SR.VALUE}'s registered root" in bad["why"], bad


def test_a_bit_row_port_whose_words_and_width_are_not_its_rows_is_refused(fx, tmp_path):
    """A bit row is exactly `words * word_bits` bits: `register_served` refuses a tree whose words or width are not the
    parameter's rows, and `flock-verify` refuses the session on a file whose `cs` port names 63 or 65 16-bit words, or 64
    32-bit ones, for its 1024-bit rows (`Port.fits`)."""
    import served_registered as SR

    schema, fx = fx
    if schema == G.SCHEMA:
        pytest.skip("a row/v1 row pads to whole blocks: its width check is test_lean_registered_reads.py's")
    program, query = SR.program()
    tree = ServedTree(**json.loads((fx / "stage" / "served.json").read_text()))
    for change in ({"words": 63}, {"words": 65}, {"word_bits": 32}):
        with pytest.raises(ValueError):
            register_served(program, query, [dataclasses.replace(tree, **change)])
    honest = json.loads(register_served(program, query, [tree]).file_bytes())
    for name, width, want in (("w63", {"words": 63}, "rows of 63 16-bit leaves"),
                              ("w65", {"words": 65}, "rows of 65 16-bit leaves"),
                              ("w64x32", {"word_bits": 32}, "rows of 64 32-bit leaves")):
        doc = json.loads(json.dumps(honest))
        doc["ports"]["cs"].update(width)
        (tmp_path / f"{name}.json").write_text(json.dumps(doc))
        bad = _verify(fx, tmp_path / f"{name}.json")
        assert bad["accepted"] is False and f"not {SR.VALUE}'s registered {want}" in bad["why"], (name, bad)


@pytest.fixture(scope="module")
def plain(built, tmp_path_factory):
    sys.path.insert(0, str(VERIFIER))
    import ci

    try:
        return ci.fetch(PLAIN, tmp_path_factory.mktemp("store"))
    except (subprocess.CalledProcessError, OSError) as e:
        pytest.skip(f"the evidence store is not reachable: {e}")


def test_a_plain_session_on_registered_reads_is_accepted_and_refused_without_its_reads(plain, tmp_path):
    """Without `--zk`, a shared-row public file at hidden outputs and `hm96-sha512/v1` leaves is a covered form when the
    verifier holds its own registered reads (`ProvedScope.sharedRowsOk`, `FailClosed.SoundExecHR`): the plain session drawn by
    the verifier's own stratified law is accepted on `register_served`'s file, and refused on it with the tree's root changed
    (the registered check opens no row) and without `--registered`."""
    import served_registered as SR

    program, query = SR.program()
    tree = ServedTree(**json.loads((plain / "stage" / "served.json").read_text()))
    mode = ("--stratified", str(PLAIN_K), "--coins", "os")
    ok = _verify(plain, register_served(program, query, [tree]).write(tmp_path / "registered.json"), mode)
    assert ok["accepted"] is True, ok
    assert ok["registered"] == {"cs": {"value": SR.VALUE, "root": tree.root}}
    moved = register_served(program, query, [dataclasses.replace(tree, root="00" * 64)])
    bad = _verify(plain, moved.write(tmp_path / "registered-root.json"), mode)
    assert bad["accepted"] is False and f"does not open {SR.VALUE}'s registered root" in bad["why"], bad
    none = _verify(plain, None, mode)
    assert none["accepted"] is False and "the verifier holds no registered roots" in none["why"], none
