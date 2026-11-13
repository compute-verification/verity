"""The proof system's own run of Flock's Lean verifier (`one_stage.proofs`), on a pinned `--zk --registered`
session: a RoPE d64 statement of 32 proof blocks, each reading its `cs` row from a registered served tree, its `x` row and
output row registered too, proved by Flock in one loopback session under a subset sample of 16
(`verity/protocols/flock/pod/89-served-registered.sh`'s row/v1 fixture). The verifier's own program, partition and `--registered`
file are the fixture's `stage/`, the ones
`verity/protocols/flock/prover/tests/test_lean_served_registered.py` holds to `register_served`.

Each verify sets up an m = 25 table (about 10 s), so the fixture's tests are slow. They are the one test that a protocol's
verdict comes from core's own verifier run: the honest session, whose public file is a shared-row file in the form
`SampledKSLiveV5` covers (`Flock.KsScope.sharedWhy`), is accepted with its program and registered finding read off that
run, and proves the proof blocks its sample holds; a circuit file other than the committed one
is refused before the verifier runs, tampered proof bytes
are rejected by the verifier itself, with the record's proof digests kept or rewritten to match, and the statement, whose
``cs`` reads are registered by the verifier's own file, is refused by Lean without that file, whether its public file
names them or not. With it, a public file whose refs point one unit at another unit's rows is refused at its registered
port. Under the registration's strata (`STRATA`) its public file's units are the population in order, and an
accepted session under another population is rejected (`population_why`). A registration whose partition object queries
the program for another population is rejected at setup, by the v2 audit and by `verify` alike: the circuit file binds the
digest of its own partition object.

Without registered reads (``registered`` None), `verify` refuses a program with an input bound at registration (a
`RegisteredInput`) before Lean runs (fast tests, no fixture): Lean demands registered reads only where the public file's
``registered``, which the prover writes, names them. The fixture's program binds ``cs`` as a run input, so that refusal
doesn't apply to it.

`printed` prints Lean's own `printT` vectors as Lean does, and `fits_why` and `draw_fits_why` refuse a length at their
limit and not below it (fast tests; the sampled-proofs service's difftest holds them to Lean's `fits` piece)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from verity.protocols.spot.verifier.one_stage import produced as PRODUCED, proofs as PF

#: `test_lean_served_registered.py`'s row/v1 fixture, fetched from the evidence store as that test fetches it (its records
#: re-rendered in the canonical text the verifier requires)
FIXTURE = "art:54456e32d1e801018ba6c9da92548482a0e9c8c8fd5c2fc956086cb6e3d8bd4b"
VERIFIER = Path(__file__).resolve().parents[3] / "flock" / "agreement"
#: the SHA-512 of the fixture's circuit file, as an auditor would commit it (its public file's `circuit_sha512` pin)
CIRCUIT = ("7b9e46f755bced2c6e00a7706f3fca90631864ca22a2d08b51c8b82260a7bd9a"
           "9ae7a191c0eabe2993a6fbf686b832c440f2a78dad65fa3b13b3cf5bd47284e0")
#: the registration's strata over the fixture's program and query (`partition.strata`): one template's 32 instances
STRATA = (("RoPEHead_v1{D=64}", ((0, 32),)),)
#: the fixture's registered roots: every port of the circuit is registered (`Flock.Registered.checkComplete`)
REGISTERED_ROOTS = {
    "cs": {"value": "cs_table", "root": "e06311d0afa0492b635c50a6a983041b1ed4a577f27209d705aebcb1d5690820"
                                        "97d7da63c77fb2c614f5838501f22e2a557f838d48fa0c143a43751f4593296e"},
    "x": {"value": "x", "root": "504f74c35b38fceacb9485685bf8a2fa753254d9e47eab2cd788584f4f2d417c"
                                "bde4dc5c56538d21d7047a2b834c0c7ef816e3b8b68378738e38820ad0e3fe70"},
    "out": {"value": "rope_out", "root": "cffcc31602347cbafa44022c617bae5506de1f8f800b5b0c4b75559020f3c50a"
                                         "2960e8ff8e0968d9d4adf7281c8739366165795a5a72c1dd55d6798aaf1dd4fb"},
}
#: the instances the session's own sample holds, which are the proof blocks, the public file's indices being 0..31
DRAWN = (1, 4, 6, 7, 9, 14, 16, 18, 19, 20, 21, 22, 23, 24, 27, 29)

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


@pytest.mark.parametrize("indices, strata, ok", [
    ([0, 1, 2], [("a", [[0, 2]]), ("b", [[2, 3]])], True),
    ([], [], True),
    ([0, 1], [("a", [[0, 3]])], False),
    ([0, 1, 2], [("a", [[0, 2]])], False),
    ([1, 0], [("a", [[0, 2]])], False),
    ([1, 2], [("a", [[0, 2]])], False),
    ([0, 1, 2], [("a", [[0, 1], [2, 3]]), ("b", [[1, 2]])], False),
    ([0, 1], [("a", [[1, 2], [0, 1]])], False),
], ids=["two strata in order", "empty", "a unit short", "a unit over", "two swapped", "shifted",
        "strata interleaved", "ranges out of order"])
def test_the_population_is_the_public_files_units_in_order(indices, strata, ok):
    """`population_why` (`FilePart.why`) passes only where the strata, read stratum by stratum, are the units 0, 1, …
    in order and the public file's units are exactly those."""
    why = PF.population_why(indices, strata)
    assert (why is None) == ok, why
    if not ok:
        assert why.startswith("partition: "), why


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
         "shared rows, Lean's to refuse": ({"header": {"shared_rows": {"x": 2}}}, None),
         "the scope at shared rows": ({"groups": 2, "header": {"shared_rows": {"x": 2}}}, SCOPE)}


@pytest.mark.parametrize("form, want", FORMS.values(), ids=FORMS.keys())
def test_the_mirror_of_the_ks_form_refuses_what_lean_refuses(tmp_path, form, want):
    """`ks_form_why` on files' heads: `KsScope.filesWhy`'s scope refusal with its reason, a shared-row file left to
    Lean's `sharedWhy` (`verity/protocols/flock/prover/tests/test_lean_ks_form.py` holds it to `flock-verify ks-form` on the prover's
    files)."""
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


@pytest.mark.parametrize("own, want", [(None, "-1"), ("0", "0")], ids=["unset", "the caller's own"])
def test_the_verifier_runs_under_its_allocator_setting_unless_the_caller_sets_its_own(tmp_path, monkeypatch, own, want):
    circuit, public = _files(tmp_path)
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "session.json").write_text("{}")
    verdict = {"session": str(tmp_path / "s"), "accepted": False, "why": "refused"}
    children = []

    def child(cmd, **kw):
        children.append(kw["env"])
        return subprocess.CompletedProcess(cmd, 0, f"VERDICT {json.dumps(verdict)}\n", "")

    assert PF.VERIFIER_ENV == {"MIMALLOC_PURGE_DELAY": "-1"}
    if own is None:
        monkeypatch.delenv("MIMALLOC_PURGE_DELAY", raising=False)
    else:
        monkeypatch.setenv("MIMALLOC_PURGE_DELAY", own)
    monkeypatch.setattr(PF, "executable", lambda: Path(PF.EXE))
    monkeypatch.setattr(PF.subprocess, "run", child)
    got, why = PF.verify(PF.Proof(circuit, public, (tmp_path / "s",)), circuit_sha512=PF.file_sha512(circuit),
                         program=RUN_ONLY, partition={"program": "p"}, registered=None)
    assert why is None and [(f.accepted, f.why) for f in got] == [(False, "refused")]
    assert children == [{**os.environ, "MIMALLOC_PURGE_DELAY": want}]


#: `Flock/Test/RecordText.lean`'s `printed` vectors and a text it `accepts`, which is its own printing, verbatim (Lean's
#: string escapes read as Python's): a JSON text and `Flock.RecordText.printT` of the value Lean parses from it
PRINTED = {
    "escapes": ("{\"s\": \"a\\\"b\\\\c\\nd\\re\\tf\\u0001g\\u001fh/i\x7fjék l😀m\", \"e\": \"\", \"kéy\": 1}",
                "{\"e\":\"\",\"kéy\":1,\"s\":\"a\\\"b\\\\c\\nd\\re\\u0009f\\u0001g\\u001fh/i\x7fjék l😀m\"}"),
    "order": ("{\"b\": 1, \"a\": 2, \"B\": 3, \"aa\": 4, \"a_b\": 5, \"é\": 6, \"z\": {\"y\": 1, \"x\": [3, {\"q\": null, \"p\": true}]}}",
              "{\"B\":3,\"a\":2,\"a_b\":5,\"aa\":4,\"b\":1,\"z\":{\"x\":[3,{\"p\":true,\"q\":null}],\"y\":1},\"é\":6}"),
    "literals": ("{\"n\": null, \"t\": true, \"f\": false, \"arr\": [], \"obj\": {}, \"nested\": [[], [[]], {}]}",
                 "{\"arr\":[],\"f\":false,\"n\":null,\"nested\":[[],[[]],{}],\"obj\":{},\"t\":true}"),
    "integers": ("{\"z\": 0, \"neg\": -3, \"big\": 12345678901234567890, \"u64\": 18446744073709551615, \"i64\": -9223372036854775808}",
                 "{\"big\":12345678901234567890,\"i64\":-9223372036854775808,\"neg\":-3,\"u64\":18446744073709551615,\"z\":0}"),
    "verdict": ("{\"soundness\": {\"per_proof_log2\": -132.02, \"terms\": {\"coin_commitment_hiding\": -191.0, \"irs_field\": -169.441}}, \"accepted\": true, \"secs\": 0.53}",
                "{\"accepted\":true,\"secs\":0.53,\"soundness\":{\"per_proof_log2\":-132.02,\"terms\":{\"coin_commitment_hiding\":-191,\"irs_field\":-169.441}}}"),
    "accepted": ("{\"a\":\"\\u0009\",\"b\":0.00001}", "{\"a\":\"\\u0009\",\"b\":0.00001}"),
}


@pytest.mark.parametrize("text, want", PRINTED.values(), ids=PRINTED.keys())
def test_printed_is_lean_printT_on_its_own_vectors(text, want):
    assert PF.printed(json.loads(text)) == want
    assert PF.printed_size(json.loads(text)) == len(want.encode())


#: one field at its own size at a time, the rest empty: (field, value, size, the refusal naming it)
FIELDS = [("circuit", b"abc", 3, "the circuit file"), ("public", b"abc", 3, "the public file"),
          ("tables", [("", b"")] * 3, 3, "the number of lookup tables"),
          ("tables", [("é", b"")], 2, "a lookup table"), ("tables", [("", b"abcd")], 4, "a lookup table"),
          ("program", b"abcde", 5, "the program"), ("partition", {"a": 1}, 7, "the partition object"),
          ("registered", ["é"], 6, "the registered reads")]


@pytest.mark.parametrize("field, value, size, want", FIELDS, ids=[f"{f} {w}" for f, _, _, w in FIELDS])
def test_fits_why_refuses_a_field_at_the_limit_and_not_below_it(field, value, size, want):
    inputs = {"circuit": b"", "public": b"", "tables": [], "program": None, "partition": None, "registered": None}
    inputs[field] = value
    assert PF.fits_why(size, **inputs) == want
    assert PF.fits_why(size + 1, **inputs) is None


def test_fits_why_names_the_first_field_past_the_limit_and_the_draw_its_own():
    assert PF.fits_why(1, b"x", b"y", [("t", b"z")], b"p", {}, []) == "the circuit file"
    assert PF.fits_why(3, b"", b"", [], b"", {"a": 1}, ["é"]) == "the partition object"
    assert PF.fits_why(PF.WORD, b"x", b"y", [("t", b"z")], b"p", {}, []) is None
    assert PF.draw_fits_why(7, {"a": 1}) == "the draw" and PF.draw_fits_why(8, {"a": 1}) is None
    assert PF.draw_fits_why(0, None) is None


def _template(d: Path, const: int, meta: dict) -> Path:
    """A circuit file's head: META and a ``unit`` circuit of one input group of one word, its constant at row
    `const`."""
    circuit = d / "circuit.txt"
    circuit.write_text(f"flock-circuit/v1 u\nMETA {json.dumps(meta)}\nCIRCUIT unit 2\n"
                       f"flock-ir-unit/v2 u {const + 1} {const} 1 0 1 1 16 1 1 1 1 16\n0 0\n")
    return circuit


TEMPLATES = {"one unit, 16 bits": (129, {"units_per_vu": 1, "leaves_out": [[0]]}, None),
             "one unit, no computed row of its net's own": (128, {"units_per_vu": 1, "leaves_out": [[0]]}, None),
             "no unit": (129, {"units_per_vu": 0, "leaves_out": []}, PF.NO_ROW_WHY),
             "the widest class": (129, {"units_per_vu": 2, "leaves_out": [list(range(8192))] * 2}, None),
             "one leaf past it": (129, {"units_per_vu": 2, "leaves_out": [list(range(8192)), list(range(8193))]},
                                  PF.WIDE_WHY),
             "a leaf map past units_per_vu is no unit's": (
                 129, {"units_per_vu": 1, "leaves_out": [[0], list(range(16384))]}, None)}


@pytest.mark.parametrize("const, meta, want", TEMPLATES.values(), ids=TEMPLATES.keys())
def test_the_mirror_of_the_template_check_refuses_what_lean_refuses(tmp_path, const, meta, want):
    """`template_why` on a circuit file's head: `OneStageZk.templateOk`'s two refusals with their reasons, at the widest
    class's edge, and only over the ``units_per_vu`` units' leaf maps."""
    assert PF.template_why(_template(tmp_path, const, meta)) == want


def test_a_session_lean_accepts_on_a_template_the_mirror_refuses_is_rejected_with_its_reason(tmp_path, monkeypatch):
    """`verify` rejects a session the verifier accepts on a circuit `template_why` refuses, with v2's reason, as a
    finding rather than a refusal of the run."""
    circuit = _template(tmp_path, 129, {"units_per_vu": 0, "leaves_out": []})
    public = tmp_path / "pub.bin"
    public.write_bytes(b"{}\n")
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "session.json").write_text("{}")
    verdict = {"session": str(tmp_path / "s"), "accepted": True, "why": "", "program_sha512": "p"}
    monkeypatch.setattr(PF, "executable", lambda: Path("flock-verify"))
    monkeypatch.setattr(PF.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 0, f"VERDICT {json.dumps(verdict)}\n", ""))
    got, why = PF.verify(PF.Proof(circuit, public, (tmp_path / "s",)), circuit_sha512=PF.file_sha512(circuit),
                         program=RUN_ONLY, partition={"program": "p"}, registered=None)
    assert why is None, why
    (f,) = got
    assert (f.accepted, f.why, f.units) == (False, PF.NO_ROW_WHY, ()), f


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


def _verify(fx: Path, session: Path, circuit: str = CIRCUIT, public: str = "pub-32.bin", own: bool = True,
            strata=STRATA, partition=None):
    st = fx / "stage"
    return PF.verify(PF.Proof(st / "circuit.txt", st / public, (session,)), circuit_sha512=circuit,
                     program=(st / "program.json").read_bytes(),
                     partition=partition or json.loads((st / "partition.json").read_text()),
                     registered=json.loads((st / "registered.json").read_text()) if own else None, strata=strata)


@lean
@pytest.mark.slow
def test_the_honest_session_is_accepted_by_cores_own_verifier_run_at_its_shared_rows(fx):
    """The honest session's sample is `DRAWN` (the record's `unit_draw`), and its verdict names the partition's program and
    the registered root; its shared-row public file is in the form `SampledKSLiveV5` covers, so the verifier accepts it
    and it proves the sampled proof blocks."""
    got, why = _verify(fx, fx / "session")
    assert why is None, why
    (f,) = got
    partition = json.loads((fx / "stage" / "partition.json").read_text())
    assert (f.accepted, f.why) == (True, ""), f
    assert PF.ks_form_why(fx / "stage" / "circuit.txt", fx / "stage" / "pub-32.bin") is None
    assert f.program == partition["program"]
    assert f.registered == REGISTERED_ROOTS
    assert f.units == DRAWN
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


def _pointed(fx: Path) -> str:
    """The fixture's public file with the first drawn unit's ref at port 0 (the row it reads of that port's shared table)
    pointed at the next instance's, as a prover pointing a drawn unit at another unit's rows would write it; its header
    and tables kept. A drawn unit's, since the session's statement is the drawn units' (`Flock.Draw`); port 0, an input,
    since a circuit's inputs precede its outputs, whose refs setup holds to each instance's own row."""
    st = fx / "stage"
    head, body = (st / "pub-32.bin").read_bytes().split(b"\n", 1)
    header = json.loads(head)
    n, np_ = header["instances"], len(header["shared_rows"])
    at = PRODUCED.BC_BYTES * sum(header["shared_rows"].values())
    refs = bytearray(body[at:at + 4 * n * np_])
    i = header["units"]["indices"].index(DRAWN[0])
    a, b = 4 * i * np_, 4 * ((i + 1) % n) * np_
    assert refs[a:a + 4] != refs[b:b + 4], "the drawn unit and the next read one row at port 0: nothing to point"
    refs[a:a + 4] = refs[b:b + 4]
    (st / "pub-pointed.bin").write_bytes(head + b"\n" + body[:at] + bytes(refs) + body[at + 4 * n * np_:])
    return "pub-pointed.bin"


@lean
@pytest.mark.slow
def test_a_ref_pointed_at_another_units_rows_is_refused_at_its_registered_port(fx):
    """Every port of the fixture's circuit is registered (`REGISTERED_ROOTS`), so the session's verifier holds each
    instance's ref to the row at the position its unit's read names (`Flock.Registered.check`): a drawn unit's input ref
    pointed at the next instance's row is refused at setup, before any proof is read, and proves no proof block. Round 2
    requires every port a second-stage header shares rows at to be registered (`second-refs`), so its sessions' refs are
    held this way."""
    got, why = _verify(fx, fx / "session", public=_pointed(fx))
    assert why is None, why
    (f,) = got
    assert f.accepted is False and f.units == (), f
    assert f.why.startswith("setup: registered reads: port ") and " reads position " in f.why, f


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


def _v2(fx: Path, session: Path, circuit: str = CIRCUIT, public: str = "pub-32.bin", own: bool = True, partition=None):
    st = fx / "stage"
    return PF.verdicts_v2(PF.Proof(st / "circuit.txt", st / public, (session,)), circuit_sha512=circuit,
                          program=json.loads((st / "program.json").read_bytes()),
                          partition=partition or json.loads((st / "partition.json").read_text()),
                          registered=json.loads((st / "registered.json").read_text()) if own else None)


def _same(v1, v2):
    """The two runs' findings agree on all but the session's name (a path for `verify`, a name for the v2 audit)."""
    (a,), (b,) = v1, v2
    assert (a.accepted, a.why, a.program, a.registered, a.units) == (b.accepted, b.why, b.program, b.registered,
                                                                    b.units), (a, b)


@lean
@pytest.mark.slow
def test_the_v2_audits_own_run_on_blobs_agrees_with_verify(fx, tmp_path):
    """The v2 audit (`Spot.OneStageZk`) runs `flock-verify verify --zk`'s one-stage path inside its own Lean, on
    the files as blobs: on the honest session (accepted at its shared rows, proving its sampled proof blocks), a session
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
            assert (f.accepted, f.why, f.units) == (True, "", DRAWN), f


@lean
@pytest.mark.slow
def test_an_accepted_session_of_another_population_is_rejected(fx):
    """Under a registration of 31 units, the honest session that the verifier accepts on its public file of 32 is
    rejected with `population_why`'s reason: its file's unit 31 is no slot of the registration's."""
    (f,), why = _verify(fx, fx / "session", strata=(("RoPEHead_v1{D=64}", ((0, 31),)),))
    assert why is None, why
    assert (f.accepted, f.units) == (False, ()) and f.why == PF.population_why(range(32), [("t", [[0, 31]])]), f


@lean
@pytest.mark.slow
def test_a_registration_whose_partition_disagrees_with_the_circuit_file_is_refused_by_the_v2_audit(fx):
    """A registration whose partition object asks the fixture's program for its `Q_call` units at X = 16, the RoPE
    heads' calls (2048 of them, `partition.strata`), rather than the 32 template instances the circuit file holds: the
    v2 audit's own run rejects the honest session at setup, since the circuit file binds another partition object's
    digest (`Flock.checkPartitionWith`), before any verdict reaches `FilePart.check`; `verify`, handed that
    registration's strata, rejects it the same way. The other tests here run the v2 audit only on the circuit file's own
    partition object."""
    from verity.protocols.spot.verifier.one_stage import partition as PA
    from verity.circuits.codec import decode_program
    from verity.definitions.registry.catalog import load

    st = fx / "stage"
    part = {**json.loads((st / "partition.json").read_text()),
            "query": {"name": "Q_call", "version": 1, "params": {"X": 16}}}
    strata = PA.strata(decode_program(json.loads((st / "program.json").read_text()), load()[0]), part["query"])
    why = PF.population_why(range(32), strata)
    assert why is not None and why.startswith("partition: the public file's 32 units are not"), (why, strata)
    mine, refused = _v2(fx, fx / "session", partition=part)
    assert refused is None, refused
    (f,) = mine
    assert (f.accepted, f.why, f.units) == (
        False, "setup: partition: the circuit's partition digest is not the verifier's partition object's", ()), f
    got, refused = _verify(fx, fx / "session", strata=strata, partition=part)
    assert refused is None, refused
    _same(got, mine)


@lean
@pytest.mark.slow
def test_the_v2_audit_refuses_a_circuit_no_blob_holds(fx):
    """The circuit file's blob is found only by the SHA-512 of its own bytes, so a digest other than the file's is a
    refusal before any session is set up."""
    got, why = _v2(fx, fx / "session", circuit="ab" * 64)
    assert got is None and why.startswith("circuit: no blob has the SHA-512"), why
