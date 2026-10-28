"""The proof service's own run of C-Flock's Lean verifier of record (`one_stage.proofs`), on a pinned `--zk --registered`
session: a RoPE d64 statement of 32 proof blocks, each reading its `cs` row from a registered served tree, proved by C-Flock in
one loopback session under a subset draw of 16 (`verity/ml/flock/pod/89-served-registered.sh`'s row/v1 fixture). The
verifier's own program, partition and `--registered` file are the fixture's `stage/`, the ones
`verity/ml/flock/tests/test_lean_served_registered.py` holds to `register_served`.

Each verify sets up an m = 25 table (about 10 s), so the fixture's tests are slow. They are the one test that a protocol's
verdict comes from core's own verifier run: the honest session passes every check with its program and registered
finding read off that run, and is refused for its shared-row public file alone, which `verify --zk` refuses until a proof
covers it (`Flock.KsScope`, `SampledKSLiveV4`), so it proves no proof block; a circuit file other than the committed one
is refused before the verifier runs, tampered proof bytes
are rejected by the verifier itself, with the record's proof digests kept or rewritten to match, and the statement, whose
``cs`` reads are registered by the verifier's own file, is refused by Lean without that file, whether its public file
names them or not.

Without registered reads (``registered`` None), `verify` refuses a program with an input bound at registration (a
`RegisteredInput`) before Lean runs (fast tests, no fixture): Lean demands registered reads only where the public file's
``registered``, which the prover writes, names them. The fixture's program binds ``cs`` as a run input, so that refusal
doesn't apply to it."""
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
FIXTURE = "art:48ed507aa908f8a220405467097af6ce986fab36ba32d2d7aede8b7cbc5213d5"
VERIFIER = Path(__file__).resolve().parents[4] / "ml" / "flock" / "verifier"
#: the SHA-512 of the fixture's circuit file, as an auditor would commit it (its public file's `circuit_sha512` pin)
CIRCUIT = ("7b9e46f755bced2c6e00a7706f3fca90631864ca22a2d08b51c8b82260a7bd9a"
           "9ae7a191c0eabe2993a6fbf686b832c440f2a78dad65fa3b13b3cf5bd47284e0")
CS_ROOT = ("8c31089e5864c2e01b73d983ea3bcfa704e49e961466476987a2a4d908a87ef3"
           "d65628f58e1ac7eb5f3d86da77250db500a01d1a4c9cf77e0be83ca3aae5413a")
#: the instances the session's own draw drew, which are the proof blocks, the public file's indices being 0..31
DRAWN = (0, 3, 4, 5, 8, 11, 12, 13, 17, 18, 23, 26, 28, 29, 30, 31)
#: `verify --zk`'s last refusal, of a shared-row public file (`KsScope.TABLES_WHY`)
TABLES = ("refused: a shared-row public file (header shared_rows) under --zk: a statement form the soundness proof does "
          "not cover")

#: a test that runs Lean on the fixture
lean = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
#: descriptors that read registered values, or can't say they don't, and one that reads none
REGISTERED = {"registered input": json.dumps({"definitions": {"RegisteredInput16_v1": {}, "Input16_v1": {}}}).encode(),
              "not json": b"definitions", "no definitions": b"{}", "definitions not an object": b'{"definitions": []}'}
RUN_ONLY = json.dumps({"definitions": {"Input16_v1": {}, "Main_v1": {}}}).encode()


def _never():
    raise AssertionError("the verifier ran")


@pytest.mark.parametrize("program", REGISTERED.values(), ids=REGISTERED.keys())
def test_without_registered_reads_a_program_that_reads_registered_values_is_refused_before_lean_runs(
        tmp_path, monkeypatch, program):
    monkeypatch.setattr(PF, "executable", _never)
    got, why = PF.verify(PF.Proof(tmp_path / "circuit.txt", tmp_path / "pub.bin", (tmp_path / "session",)),
                         circuit_sha512="ab" * 64, program=program, partition={}, registered=None)
    assert got is None and why[0] == "registered", why
    assert PF.reads_registered(program)


@pytest.mark.parametrize("program, registered", [(RUN_ONLY, None), (REGISTERED["registered input"], {})],
                         ids=["reads none", "registered reads given"])
def test_a_program_without_registered_reads_or_given_them_goes_on_to_its_files(tmp_path, monkeypatch, program,
                                                                               registered):
    monkeypatch.setattr(PF, "executable", _never)
    got, why = PF.verify(PF.Proof(tmp_path / "circuit.txt", tmp_path / "pub.bin", (tmp_path / "session",)),
                         circuit_sha512="ab" * 64, program=program, partition={}, registered=registered)
    assert got is None and why[0] == "proof", why


def _files(d: Path, groups: int = 1, meta: dict | None = None, header: dict | None = None) -> tuple[Path, Path]:
    """A circuit file's head (its META and a ``unit`` circuit of `groups` input groups) and a public file's header."""
    unit = f"flock-ir-unit/v2 u 1 2 {groups}" + " 0 1 1 16" * groups + " 1 1 1 1 16\n0 0\n"
    circuit, public = d / "circuit.txt", d / "pub.bin"
    circuit.write_text(f"flock-circuit/v1 u\nMETA {json.dumps(meta or {})}\nCIRCUIT unit 2\n{unit}")
    public.write_bytes(json.dumps(header or {}).encode() + b"\n")
    return circuit, public


SCOPE = f"refused: {PF.SCOPE_WHY}: a statement form the soundness proof does not cover"
FORMS = {"unit form, own rows": ({}, None), "two input groups": ({"groups": 2}, SCOPE),
         "a tail stage returns the outputs": ({"meta": {"out_net": "stage0"}}, SCOPE),
         "shared rows": ({"header": {"shared_rows": {"x": 2}}}, TABLES),
         "the scope before the rows": ({"groups": 2, "header": {"shared_rows": {"x": 2}}}, SCOPE)}


@pytest.mark.parametrize("form, want", FORMS.values(), ids=FORMS.keys())
def test_the_mirror_of_the_ks_form_refuses_what_lean_refuses(tmp_path, form, want):
    """`ks_form_why` on files' heads: `KsScope.filesWhy`'s refusals with its reasons, the scope's first
    (`verity/ml/flock/tests/test_lean_ks_form.py` holds it to `flock-verify ks-form` on the prover's files)."""
    assert PF.ks_form_why(*_files(tmp_path, **form)) == want


def test_a_session_lean_accepts_in_a_form_the_mirror_refuses_is_refused(tmp_path, monkeypatch):
    circuit, public = _files(tmp_path, groups=2)
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "session.json").write_text("{}")
    verdict = {"session": str(tmp_path / "s"), "accepted": True, "why": "", "program_sha512": "p"}
    monkeypatch.setattr(PF, "executable", lambda: Path("flock-verify"))
    monkeypatch.setattr(PF.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 0, f"VERDICT {json.dumps(verdict)}\n", ""))
    got, why = PF.verify(PF.Proof(circuit, public, (tmp_path / "s",)), circuit_sha512=PF.file_sha512(circuit),
                         program=RUN_ONLY, partition={"program": "p"}, registered=None)
    assert got is None and why[0] == "form" and why[1].endswith(SCOPE), why


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


def _verify(fx: Path, session: Path, circuit: str = CIRCUIT, public: str = "pub-32.bin", own: bool = True):
    st = fx / "stage"
    return PF.verify(PF.Proof(st / "circuit.txt", st / public, (session,)), circuit_sha512=circuit,
                     program=(st / "program.json").read_bytes(),
                     partition=json.loads((st / "partition.json").read_text()),
                     registered=json.loads((st / "registered.json").read_text()) if own else None)


@lean
@pytest.mark.slow
def test_the_honest_session_passes_every_check_of_cores_own_verifier_run_but_its_shared_rows(fx):
    """The honest session's draw is `DRAWN` (the record's `unit_draw`), and its verdict names the partition's program and
    the registered root; the verifier refuses it for its shared-row public file alone, so it proves no proof block."""
    got, why = _verify(fx, fx / "session")
    assert why is None, why
    (f,) = got
    partition = json.loads((fx / "stage" / "partition.json").read_text())
    assert (f.accepted, f.why) == (False, TABLES), f
    assert PF.ks_form_why(fx / "stage" / "circuit.txt", fx / "stage" / "pub-32.bin") == TABLES
    assert f.program == partition["program"]
    assert f.registered == {"cs": {"value": "cs_table", "root": CS_ROOT}}
    assert f.units == ()
    draw = json.loads((fx / "session" / "session.json").read_text())["unit_draw"]
    assert tuple(draw["units"]) == DRAWN, draw


def _unnamed(fx: Path) -> str:
    """The fixture's public file with its header's ``registered`` taken out, as a prover that names no registered read
    would write it; its tables (the registered port's shared rows) kept."""
    st = fx / "stage"
    head, rest = (st / "pub-32.bin").read_bytes().split(b"\n", 1)
    header = json.loads(head)
    del header["registered"]
    (st / "pub-unnamed.bin").write_bytes(json.dumps(header).encode() + b"\n" + rest)
    return "pub-unnamed.bin"


@lean
@pytest.mark.slow
@pytest.mark.parametrize("unnamed", [False, True], ids=["public file names its reads", "public file names none"])
def test_lean_without_registered_reads_refuses_the_statement_that_reads_them(fx, unnamed):
    """The program binds ``cs`` as a run input (`Input16_v1`), registered only by the verifier's own file, so `verify`
    without that file goes on to Lean: the honest public file, which names the ``cs`` read, is refused by
    `Registered.check`; with that name taken out and the shared rows kept, the session is still not accepted. Neither
    finding proves a proof block."""
    assert not PF.reads_registered((fx / "stage" / "program.json").read_bytes())
    got, why = _verify(fx, fx / "session", public=_unnamed(fx) if unnamed else "pub-32.bin", own=False)
    assert why is None, why
    (f,) = got
    assert f.accepted is False and f.units == (), f
    if not unnamed:
        assert "the verifier holds no registered roots" in f.why, f


@lean
@pytest.mark.slow
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


@lean
@pytest.mark.slow
@pytest.mark.parametrize("rehash", [False, True], ids=["record-kept", "record-rehashed"])
def test_tampered_proof_bytes_are_rejected_by_the_verifier_itself(fx, tmp_path, rehash):
    """One bit of the first rep's proof flipped: with the record's proof digest kept the verifier refuses the file the
    record doesn't bind, and with it rewritten to the flipped bytes it refuses the proof itself. Either way the finding is
    the verifier's reject, with no proof blocks proved."""
    got, why = _verify(fx, _tampered(fx, tmp_path, rehash))
    assert why is None, why
    (f,) = got
    assert f.accepted is False and f.why and f.units == (), f


@lean
@pytest.mark.slow
def test_missing_proof_files_are_refused(fx, tmp_path):
    st = fx / "stage"
    got, why = PF.verify(PF.Proof(st / "circuit.txt", st / "pub-32.bin", (tmp_path / "none",)), circuit_sha512=CIRCUIT,
                         program=b"{}", partition={}, registered={})
    assert got is None and why[0] == "proof", why


def _v2(fx: Path, session: Path, circuit: str = CIRCUIT, public: str = "pub-32.bin", own: bool = True):
    st = fx / "stage"
    return PF.verdicts_v2(PF.Proof(st / "circuit.txt", st / public, (session,)), circuit_sha512=circuit,
                          program=json.loads((st / "program.json").read_bytes()),
                          partition=json.loads((st / "partition.json").read_text()),
                          registered=json.loads((st / "registered.json").read_text()) if own else None)


def _same(v1, v2):
    """The two runs' findings agree on all but the session's name (a path for `verify`, a name for the v2 audit)."""
    (a,), (b,) = v1, v2
    assert (a.accepted, a.why, a.program, a.registered, a.units) == (b.accepted, b.why, b.program, b.registered,
                                                                    b.units), (a, b)


@lean
@pytest.mark.slow
def test_the_v2_audits_own_run_on_blobs_agrees_with_verify(fx, tmp_path):
    """The v2 audit (`SampledProofs.OneStageZk`) runs `flock-verify verify --zk`'s one-stage path inside its own Lean, on
    the files as blobs: on the honest session (refused for its shared rows alone, proving no proof block), a session
    whose proof bytes were tampered with (record kept or rehashed), and the honest session without the verifier's
    registered file, its finding is `verify`'s, reason and proof blocks included."""
    for session, own in ((fx / "session", True), (_tampered(fx, tmp_path / "kept", False), True),
                         (_tampered(fx, tmp_path / "rehashed", True), True), (fx / "session", False)):
        got, why = _verify(fx, session, own=own)
        assert why is None, why
        mine, refused = _v2(fx, session, own=own)
        assert refused is None, refused
        _same(got, mine)
        if own and session == fx / "session":
            (f,) = mine
            assert (f.accepted, f.why, f.units) == (False, TABLES, ()), f


@lean
@pytest.mark.slow
def test_the_v2_audit_refuses_a_circuit_no_blob_holds(fx):
    """The circuit file's blob is found only by the SHA-512 of its own bytes, so a digest other than the file's is a
    refusal before any session is set up."""
    got, why = _v2(fx, fx / "session", circuit="ab" * 64)
    assert got is None and why.startswith("circuit: no blob has the SHA-512"), why
