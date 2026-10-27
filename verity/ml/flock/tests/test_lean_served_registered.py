"""A served tree registered as a port, end to end (`one_stage.served.register_served`, `served_registered.py`): a
RoPE d64 statement whose 32 units each read their `cs` row from the served tree `cs_table`, proved by C-Flock in one
loopback `--zk` session under a subset draw of 16 (`pod/89-served-registered.sh`, `FIXTURES`), once for each row format a
served tree may be: `row/v1`, and `row/v2` and `row-seg/v1` bit rows (the circuit's rows bit rows of their words, `cs` a
v2 or a segmented row). The verifier writes its own `--registered` file from the tree as the service holds it
(`served.json`), with `register_served` on its own program, and `flock-verify verify --zk --registered` passes every check
of the session on it but the last, its verdict naming that program's digest (the partition's), and refuses it on the same
file with the tree's root changed, or a bit row's port with its words or width changed, at that check.
`--zk` holds the files to the form `SampledKSLiveV4` covers (`Flock.PortRoots`, `Flock.KsScope`). The rows of all three
are in it, but their public file is a shared-row file, which `KsScope` refuses under `--zk` until a proof covers it, so
each session is refused for that alone (`TABLES`).
A plain session of the row/v1 stage (`PLAIN`: no `--zk`, the verifier's own stratified draw) is accepted on that file too,
and refused with the root changed and without `--registered`.
Each verify sets up an m = 25 table (about 10 s), so the module is slow; it is the one test of `register_served`'s file on
a proved session, which the Python round trip (`verity/core/protocols/sampled_proofs/tests/test_served.py`) can't
reach."""
from __future__ import annotations

import dataclasses
import hashlib
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
    G.SCHEMA: "art:48ed507aa908f8a220405467097af6ce986fab36ba32d2d7aede8b7cbc5213d5",
    G.SCHEMA_V2: "art:2425aa5eff7e781331bda47d22e1fcd7eb813cbd7e64e39965d851b6f843b0cd",
    G.SCHEMA_SEG: "art:1a17e14138bb6ab7412ad3ee01c62e49c9f84c666ca0fd9848c2688d6311f403",
}
#: `pod/89-served-registered.sh ZK=0`'s fixture (row/v1): the same stage, with the verifier's own stratified draw of `PLAIN_K`
#: (stage/draw.json), and session/ a plain session (no --zk, OS coins) drawn by it; vectors/ is vectors.json's live set
PLAIN = "art:88fd3f12b8f363acb27415b58e38e43f04431ce61e198b081d25c33f5b8524b9"
PLAIN_K = 16
STATEMENT = "verity/flock-circuit"
#: the last refusal of `verify --zk`, a shared-row public file (`KsScope.TABLES_WHY`)
TABLES = ("refused: a shared-row public file (header shared_rows) under --zk: a statement form the soundness proof does "
          "not cover")

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


def test_a_session_verified_on_the_file_register_served_wrote_passes_every_check_and_a_changed_root_is_refused(fx, tmp_path):
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
    assert (ok["accepted"], ok["why"]) == (False, TABLES), ok
    assert ok["registered"] == {"cs": {"value": SR.VALUE, "root": tree.root}}
    assert ok["program_sha512"] == sr.partition["program"]
    moved = register_served(program, query, [dataclasses.replace(tree, root="00" * 64)])
    bad = _verify(fx, moved.write(tmp_path / "registered-root.json"))
    assert bad["accepted"] is False and f"does not open {SR.VALUE}'s registered root" in bad["why"], bad


def test_the_provers_record_carries_the_sha512_of_the_public_file_the_verifier_reads(fx):
    """`live.json`'s ``public_file_sha512`` is the SHA-512 of ``stage/pub-32.bin``, the file `verify --public` reads. Its
    ``public_sha512`` is Σ's public digest, over the drawn statement's header, which no file holds."""
    _, fx = fx
    live = json.loads((fx / "live.json").read_text())
    assert live.get("public_file_sha512") == hashlib.sha512((fx / "stage" / "pub-32.bin").read_bytes()).hexdigest()
    assert live["public_sha512"] != live["public_file_sha512"]


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


def test_ks_form_is_the_default_under_zk_and_every_row_format_passes_its_rows_check(fx, tmp_path):
    """`--zk` holds the files to the form `SampledKSLiveV4` covers (`Flock.PortRoots.checkOf`, then `Flock.KsScope.checkOf`),
    whose rows every served row format is in: the session at a `row/v1` circuit, at `row/v2` rows and with `cs` a
    `row-seg/v1` row passes the rows check and is refused at the next, its shared-row public file, and `--ks-form`, the
    check's old flag, gives the same verdict. `--ks-form` without `--zk` is a usage error."""
    import served_registered as SR

    _, fx = fx
    program, query = SR.program()
    tree = ServedTree(**json.loads((fx / "stage" / "served.json").read_text()))
    reg = register_served(program, query, [tree]).write(tmp_path / "registered.json")
    v = _verify(fx, reg, ("--zk", "--ks-form"))
    assert (v["accepted"], v["why"]) == (False, TABLES), v
    assert v == _verify(fx, reg), v
    p = subprocess.run([str(BIN), "verify", "--ks-form", "--statement", STATEMENT, "--circuit", str(fx / "stage" / "circuit.txt"),
                        "--public", str(fx / "stage" / "pub-32.bin"), "--session", str(fx / "session")],
                       capture_output=True, text=True, check=False)
    assert p.returncode == 2 and "--ks-form goes with --zk" in p.stderr, p.stderr


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
