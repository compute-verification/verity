"""A served tree registered as a port, end to end (`one_stage.served.register_served`, `served_registered.py`): a
RoPE d64 statement whose 32 units each read their `cs` row from the served tree `cs_table`, proved by Flock in one
loopback `--zk` session under a subset sample of 16 (`../pod/89-served-registered.sh`, `FIXTURES`), once for each row format a
served tree may be: `row/v1`, and `row/v2` and `row-seg/v1` bit rows (the circuit's rows bit rows of their words, `cs` a
v2 or a segmented row). Each unit's `x` row and output row are registered values too, so every port of the circuit is
registered (`Flock.Registered.checkComplete`). The verifier writes its own `--registered` file from the trees as the
service holds them (`served.json`, `served-x.json`), with `register_served` on its own program, and the output rows' port
(`out-port.json`; `served_registered.registered_file`), and `flock-verify verify --zk --registered` passes every check of
the session on it, its verdict naming that program's digest (the partition's), and refuses it on the same file with the
tree's root changed, or a bit row's port with its words or width changed.
`--zk` holds the files to the form `SampledKSLiveV4` covers, or at a shared-row public file `SampledKSLiveV5`'s
(`Flock.PortRoots`, `Flock.KsScope`). The rows of all three and their shared-row public files are in it, so each session
is accepted.
A plain session of the row/v1 stage (`PLAIN`: no `--zk`, the verifier's own stratified sample, on the challenger's coin tree,
its coins the ones the challenger's release log shows) is accepted on that file too, and refused with the root changed and
without `--registered`.
Each verify sets up an m = 25 table (about 10 s), so the module is slow; it is the one test of `register_served`'s file on
a proved session, which the Python round trip (`verity/protocols/spot/verifier/tests/test_served.py`) can't
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

from verity.circuits.codec import canonical_json, encode_program
from verity.protocols.spot.verifier.one_stage import registered as G
from verity.protocols.spot.verifier.one_stage.served import register_served

LEAN = Path(__file__).resolve().parents[2] / "verifier"
VERIFIER = Path(__file__).resolve().parents[2] / "agreement"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
#: `../pod/89-served-registered.sh`'s fixture for each row format (`SCHEMA=`): stage/ (circuit.txt, pub-32.bin, program.json,
#: partition.json, served.json, registered.json, tables/), session/ (the sampled --zk session's record and proofs), live.json
#: (the prover's LIVE record)
FIXTURES = {
    G.SCHEMA: "art:54456e32d1e801018ba6c9da92548482a0e9c8c8fd5c2fc956086cb6e3d8bd4b",
    G.SCHEMA_V2: "art:f14126710f23f62410d0387454ab8c69b236f75f5b92df10ddc214ebc45f37e6",
    G.SCHEMA_SEG: "art:ae8f5cf2b166710fd2418eea10dddc217102587c75d00db85abf2474b2d0864c",
}
#: `../pod/89-served-registered.sh ZK=0`'s fixture (row/v1): the same stage, with the verifier's own stratified sample of `PLAIN_K`
#: (stage/draw.json), session/ a plain session (no --zk) sampled by it on the challenger's coin tree, and challenger/ its
#: record and release log; vectors/ is vectors.json's live set
PLAIN = "art:e43917db0ffa59b34e6beae04917b27a70489a4d7ab778d734ddb10c85c5af7b"
PLAIN_K = 16
STATEMENT = "verity/flock-circuit"

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


@pytest.fixture(scope="module")
def built():
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


def _verify(fx: Path, registered: Path | None, mode: tuple[str, ...] = ("--zk",), public: Path | None = None) -> dict:
    st = fx / "stage"
    reads = ["--registered", str(registered)] if registered else []
    p = subprocess.run([str(BIN), "verify", *mode, "--statement", STATEMENT, "--circuit", str(st / "circuit.txt"), "--public",
                        str(public or st / "pub-32.bin"), "--partition", str(st / "partition.json"), "--program",
                        str(st / "program.json"), *reads, "--session", str(fx / "session")],
                       capture_output=True, text=True, check=False)
    (line,) = [json.loads(x.split(" ", 1)[1]) for x in p.stdout.splitlines() if x.startswith("VERDICT ")]
    return line


def test_a_session_verified_on_the_file_register_served_wrote_is_accepted_and_a_changed_root_is_refused(fx, tmp_path):
    import served_registered as SR

    schema, fx = fx
    program, query = SR.program()
    assert canonical_json(encode_program(program)) == (fx / "stage" / "program.json").read_bytes()
    tree, x_tree, out_port = SR.load(fx / "stage")
    assert tree.row_schema == schema
    doc = SR.registered_file(program, query, tree, x_tree, out_port)
    assert canonical_json(doc) == (fx / "stage" / "registered.json").read_bytes()
    assert doc["ports"]["cs"].get(G.SCHEMA_KEY, G.SCHEMA) == schema
    sr = register_served(program, query, [tree, x_tree])
    assert sr.partition == json.loads((fx / "stage" / "partition.json").read_text())
    ok = _verify(fx, SR.write_registered(doc, tmp_path / "registered.json"))
    assert (ok["accepted"], ok["why"]) == (True, ""), ok
    assert ok["registered"] == {"cs": {"value": SR.VALUE, "root": tree.root}, "x": {"value": "x", "root": x_tree.root},
                                "out": {"value": SR.OUT_VALUE, "root": out_port["root"]}}
    assert ok["program_sha512"] == sr.partition["program"]
    moved = SR.registered_file(program, query, dataclasses.replace(tree, root="00" * 64), x_tree, out_port)
    bad = _verify(fx, SR.write_registered(moved, tmp_path / "registered-root.json"))
    assert bad["accepted"] is False and f"does not open {SR.VALUE}'s registered root" in bad["why"], bad


def test_the_verdict_names_the_files_and_the_sample_it_checked_the_session_against(fx, tmp_path):
    """`verify`'s verdict names the public file it checked the session against by its header's digest
    (``public_header_sha512``, the SHA-512 of the header's canonical JSON, `publicHeader`: the population's header, which
    the sampled statement is drawn from, never the drawn one), the registered reads by the digest of their file
    (``registered_sha512``, the SHA-512 of its canonical JSON), the session's sample (``unit_draw``, its record's) and
    its tables (``session_tables``, one here), which the second stage's audit holds each statement to. On a public file
    whose header is another's (one more key), or a registered file at another root, the session is refused, and its
    verdict names that file, never the one it was proved on."""
    import served_registered as SR

    def digest(j) -> str:
        return hashlib.sha512(canonical_json(j)).hexdigest()

    _, fx = fx
    program, query = SR.program()
    tree, x_tree, out_port = SR.load(fx / "stage")
    doc = SR.registered_file(program, query, tree, x_tree, out_port)
    reads = SR.write_registered(doc, tmp_path / "registered.json")
    pub = (fx / "stage" / "pub-32.bin").read_bytes()
    nl = pub.index(b"\n")
    header = json.loads(pub[:nl])
    ok = _verify(fx, reads)
    assert ok["accepted"] is True, ok
    assert ok["public_header_sha512"] == digest(header) and ok["registered_sha512"] == digest(doc)
    assert ok["unit_draw"] == json.loads((fx / "session" / "session.json").read_text())["unit_draw"]
    assert ok["session_tables"] == 1
    other = {**header, "staged_for": "another file"}
    (tmp_path / "pub-32.bin").write_bytes(json.dumps(other).encode() + pub[nl:])
    bad = _verify(fx, reads, public=tmp_path / "pub-32.bin")
    assert bad["accepted"] is False, bad
    assert bad["public_header_sha512"] == digest(other) != ok["public_header_sha512"]
    moved = SR.registered_file(program, query, dataclasses.replace(tree, root="00" * 64), x_tree, out_port)
    bad = _verify(fx, SR.write_registered(moved, tmp_path / "registered-root.json"))
    assert bad["accepted"] is False and bad["registered_sha512"] == digest(moved) != ok["registered_sha512"], bad


def test_the_provers_record_carries_the_sha512_of_the_public_file_the_verifier_reads(fx):
    """`live.json`'s ``public_file_sha512`` is the SHA-512 of ``stage/pub-32.bin``, the file `verify --public` reads. Its
    ``public_sha512`` is Σ's public digest, over the sampled statement's header, which no file holds."""
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
    tree, x_tree, out_port = SR.load(fx / "stage")
    for change in ({"words": 63}, {"words": 65}, {"word_bits": 32}):
        with pytest.raises(ValueError):
            register_served(program, query, [dataclasses.replace(tree, **change)])
    honest = SR.registered_file(program, query, tree, x_tree, out_port)
    for name, width, want in (("w63", {"words": 63}, "rows of 63 16-bit leaves"),
                              ("w65", {"words": 65}, "rows of 65 16-bit leaves"),
                              ("w64x32", {"word_bits": 32}, "rows of 64 32-bit leaves")):
        doc = json.loads(json.dumps(honest))
        doc["ports"]["cs"].update(width)
        (tmp_path / f"{name}.json").write_text(json.dumps(doc))
        bad = _verify(fx, tmp_path / f"{name}.json")
        assert bad["accepted"] is False and f"not {SR.VALUE}'s registered {want}" in bad["why"], (name, bad)


def test_ks_form_is_the_default_under_zk_and_every_row_format_passes_its_rows_check(fx, tmp_path):
    """`--zk` holds the files to the form `SampledKSLiveV4` covers, or at a shared-row public file `SampledKSLiveV5`'s
    (`Flock.PortRoots.checkOf`, then `Flock.KsScope.checkOf`), which every served row format is in: the session at a
    `row/v1` circuit, at `row/v2` rows and with `cs` a `row-seg/v1` row is accepted, and `--ks-form`, the check's old
    flag, gives the same verdict. `--ks-form` without `--zk` is a usage error."""
    import served_registered as SR

    _, fx = fx
    program, query = SR.program()
    reg = SR.write_registered(SR.registered_file(program, query, *SR.load(fx / "stage")), tmp_path / "registered.json")
    v = _verify(fx, reg, ("--zk", "--ks-form"))
    assert (v["accepted"], v["why"]) == (True, ""), v
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


def test_the_plain_sessions_coins_are_the_challengers(plain):
    """The plain session's coin tree is one the challenger's release log shows it sampled, with the rounds it opened; with
    one of its link points changed, it is refused at them."""
    from verity.protocols.spot.device import challenger as C, remote as R

    trees = C.coin_trees_of(R.log_lines(plain / "challenger" / R.RELEASES))
    rec = json.loads((plain / "session" / "session.json").read_text())
    tables = dict.fromkeys(rec["link"]["root_b"])
    assert len(trees) == 1 and C.sessions_tree_reason(trees, [("session", tables, rec)]) is None
    c = rec["link"]["points"][0][0]
    rec["link"]["points"][0][0] = ("0" if c[0] != "0" else "1") + c[1:]
    assert "link points are not the coins" in C.sessions_tree_reason(trees, [("session", tables, rec)])


def test_a_plain_session_on_registered_reads_is_accepted_and_refused_without_its_reads(plain, tmp_path):
    """Without `--zk`, a shared-row public file at hidden outputs and `hm96-sha512/v1` leaves is a covered form when the
    verifier holds its own registered reads (`ProvedScope.sharedRowsOk`, `FailClosed.SoundExecHR`): the plain session sampled by
    the verifier's own stratified sampling design is accepted on `register_served`'s file, and refused on it with the tree's root changed
    (the registered check opens no row) and without `--registered`."""
    import served_registered as SR

    program, query = SR.program()
    tree, x_tree, out_port = SR.load(plain / "stage")
    mode = ("--stratified", str(PLAIN_K), "--coins", "tree")
    ok = _verify(plain, SR.write_registered(SR.registered_file(program, query, tree, x_tree, out_port),
                                            tmp_path / "registered.json"), mode)
    assert ok["accepted"] is True, ok
    assert ok["registered"]["cs"] == {"value": SR.VALUE, "root": tree.root}
    assert set(ok["registered"]) == {"cs", "x", "out"}, ok
    moved = SR.registered_file(program, query, dataclasses.replace(tree, root="00" * 64), x_tree, out_port)
    bad = _verify(plain, SR.write_registered(moved, tmp_path / "registered-root.json"), mode)
    assert bad["accepted"] is False and f"does not open {SR.VALUE}'s registered root" in bad["why"], bad
    none = _verify(plain, None, mode)
    assert none["accepted"] is False and "the verifier holds no registered roots" in none["why"], none
