"""`verify --zk` itself refuses a statement outside the form the sampled audit's knowledge soundness covers
(`Flock.KsScope`, `Flock.ProvedScope`), on sessions the prover made in full, and so does the v2 audit's own run of it.

The sessions are M0 `--zk` sessions of four instances, every unit sampled, bound to its template query's partition, written
by `benchmarks/one_stage/form_refusals.sh`: ``honest``, a small A8 unit (LlBf16DotprodDot_v1 at K=128) flat;
``templated``, the same unit as its structural template (`verity/flock-circuit/tmpl`, the rows not in the file), which
`verify` takes (`ProvedScope.served`) in the form `ProvedScope.scopeOkT` covers; and ``tail``, a toy with its MUFU reads
in lookup slots and its last stage returning the outputs, which `ProvedScope.check` admits and the knowledge-soundness
form (`KsScope.filesWhy`) refuses. `verify` gives that reason only once every earlier check, the proof's included, has
passed, so it says the session was otherwise accepted. The honest session under a statement its files don't fit
(`WRONG_TAGS`: typed, public outputs, plain leaves, templated) is refused as at generation.

`Flock.KsScope`'s own tests run its check functions on Lean values, and `flock-verify ks-form` runs the check alone; these are
the tests that `verify --zk`'s verdict calls them. Each verify sets up an m = 25 table (about 10 s), so they are slow.

The sessions are a fetched fixture (`artifacts.json`), which `suites.py` fetches before the suite runs, so a
missing fixture fails these tests rather than skipping them."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from verity.circuits import partition_object as PO
from verity.protocols.proofs.spot.verifier.one_stage import proofs as PF

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "form-refusals"
VERIFIER = Path(__file__).resolve().parents[3] / "flock" / "prover" / "agreement"
ROOT = Path(__file__).resolve().parents[6]
#: Flock's public-input sessions' stage (its `test_lean_public_inputs.py`), a fetched fixture under this suite's inputs
PUBLIC_INPUTS = Path(__file__).resolve().parents[3] / "flock" / "prover" / "tests" / "fixtures" / "public-inputs"
#: a descriptor of one primitive Definition, which reads no registered value
DESCRIPTOR = {"schema": "verity-ir/descriptor/v1", "callees": [], "definitions": {"Main_v1": {"form": "primitive"}},
              "root": "Main_v1"}
#: the hm96 leaf scheme's salt source `Flock.Tags` pins, which the public-input fixture's circuit predates
SALT_SOURCE = "os bytes (getrandom), 192 per leaf, each tree's read once, never reused or stored"
#: META's bindings, which a circuit's class leaves out (`HmRow.BINDING_KEYS`)
BINDING_KEYS = ("partition", "program_sha512", "program_digests")
CHECK_OWN = ("public inputs: the circuit names public input ports; pass the verifier's own recomputation "
             "(--public-inputs F)")
N = 4
WRONG_TAGS = ("verity/flock-circuit/types", "verity/flock-circuit@eb90718f", "verity/flock-circuit+plain-leaves",
              "verity/flock-circuit/tmpl")
FORM = ": a statement form the soundness proof does not cover"

pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")]


@pytest.fixture(scope="module")
def fx(tmp_path_factory) -> Path:
    assert (FIXTURE / "cases.json").is_file(), (
        f"{FIXTURE} is not fetched: `uv run research data fetch-fixtures {FIXTURE.relative_to(ROOT)}`")
    sys.path.insert(0, str(VERIFIER))
    import ci

    root = ci.canonical_tree(FIXTURE, tmp_path_factory.mktemp("store") / "form-refusals")
    PF.executable()
    return root


def _cases(fx: Path) -> dict:
    return json.loads((fx / "cases.json").read_text())["cases"]


def _tables(fx: Path, case: str) -> list[Path]:
    """The case's MUFU tables; a case without lookups has none, and the stored tree no ``tables/``."""
    d = fx / case / "tables"
    return sorted(f for f in d.iterdir() if f.is_file()) if d.is_dir() else []


def _verify(fx: Path, case: str, statement: str | None = None) -> tuple[dict | None, int]:
    """Lean's VERDICT on the case's session under ``statement`` (the case's own by default), or None, and the exit code."""
    s, c = fx / case / "stage", fx / case
    tables = ["--tables", str(c / "tables")] if _tables(fx, case) else []
    p = subprocess.run([str(PF.executable()), "verify", "--zk", "--statement", statement or _cases(fx)[case]["statement"],
                        "--circuit", str(s / "circuit.txt"), "--public", str(c / f"pub-{N}.bin"), *tables,
                        "--partition", str(s / "partition.json"), "--program", str(s / "program.json"),
                        "--session", str(c / "session")], capture_output=True, text=True)
    line = next((x for x in p.stdout.splitlines() if x.startswith("VERDICT ")), None)
    return (json.loads(line[len("VERDICT "):]) if line else None), p.returncode


def _v2(fx: Path, case: str) -> PF.Finding:
    s, c = fx / case / "stage", fx / case
    got, refused = PF.verdicts_v2(PF.Proof(s / "circuit.txt", c / f"pub-{N}.bin", (c / "session",)),
                                  circuit_sha512=PF.file_sha512(s / "circuit.txt"),
                                  program=json.loads((s / "program.json").read_bytes()),
                                  partition=json.loads((s / "partition.json").read_text()),
                                  tables=_tables(fx, case))
    assert refused is None, refused
    (finding,) = got
    return finding


def test_verify_zk_accepts_the_honest_session(fx):
    v, _ = _verify(fx, "honest")
    assert v is not None and v.get("accepted") is True, v


def test_verify_zk_accepts_the_templated_session(fx):
    v, _ = _verify(fx, "templated")
    assert v is not None and v.get("accepted") is True and _cases(fx)["templated"]["lean"]["accepted"] is True, v


def test_verify_zk_refuses_a_tail_returning_the_outputs_for_its_form(fx):
    """The session's proof checks and every check before the form pass; the form's check refuses its circuit."""
    v, _ = _verify(fx, "tail")
    assert v is not None and v.get("accepted") is False and str(v.get("why")).endswith(FORM), v
    assert v["why"] == _cases(fx)["tail"]["lean"]["why"], v


def test_the_v2_audits_own_run_accepts_the_honest_session_and_refuses_the_tail_for_its_form(fx):
    """v2 verifies under the flat statement only, so of the templated and tail sessions it runs the tail's."""
    honest, tail = _v2(fx, "honest"), _v2(fx, "tail")
    assert honest.accepted and honest.units == tuple(range(N)), honest
    assert (tail.accepted, tail.why) == (False, _cases(fx)["tail"]["lean"]["why"]), tail


def _circuit_text(head: str, meta: dict, rest: str) -> str:
    return head + "\nMETA " + json.dumps(meta, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n" + rest


def _public_inputs_files(to: Path) -> tuple[Path, Path, dict]:
    """Flock's public-input fixture (`PUBLIC_INPUTS`: a RoPE head whose `cs` is a public input port) under v2's
    statement: its leaf scheme's salt source the one the statement pins (`Flock.Tags`), its partition a
    `verity/partition/v1` object over its program, and its public file pinned to that circuit and its class
    (`HmRow.classOf`: the file without META's bindings); the circuit, the public file and the partition object."""
    assert (PUBLIC_INPUTS / "stage" / "circuit.txt").is_file(), (
        f"{PUBLIC_INPUTS} is not fetched: `uv run research data fetch-fixtures {PUBLIC_INPUTS.relative_to(ROOT)}`")
    head, meta_line, rest = (PUBLIC_INPUTS / "stage" / "circuit.txt").read_text().split("\n", 2)
    meta = json.loads(meta_line[5:])
    assert meta["public_inputs"], meta
    meta["leaf_scheme"]["salt_source"] = SALT_SOURCE
    partition = {"format": PO.FORMAT, "program": meta["program_sha512"], "query": PO.call_query()}
    meta["partition"] = {"digest": PO.digest(partition), "rule": PO.FORMAT}
    circuit = to / "circuit.txt"
    circuit.write_text(_circuit_text(head, meta, rest))
    unbound = _circuit_text(head, {k: v for k, v in meta.items() if k not in BINDING_KEYS}, rest)
    h, body = (PUBLIC_INPUTS / "stage" / "pub-4.bin").read_bytes().split(b"\n", 1)
    header = {**json.loads(h), "circuit_sha512": PF.file_sha512(circuit)}
    header["units"]["classes"] = [hashlib.sha512(unbound.encode()).hexdigest()] * len(header["units"]["classes"])
    public = to / "pub-4.bin"
    public.write_bytes(json.dumps(header, sort_keys=True).encode() + b"\n" + body)
    return circuit, public, partition


def test_the_v2_audit_refuses_a_circuit_with_a_public_input_port(tmp_path):
    """v2 sets its session up without the verifier's own recomputation of public input words (`OneStageZk.buildSessionOf`
    passes no `--public-inputs`), so a circuit naming a public input port is refused at setup (`HmIn.checkOwn`), before
    any proof is read, on the audit's own run and not only at `flock-verify`'s command; the session is a record and no
    proof file. `Verity.Sampled.pub_of_setup` reads off that refusal that every session v2 accepts has no public input
    port."""
    circuit, public, partition = _public_inputs_files(tmp_path)
    (tmp_path / "session").mkdir()
    (tmp_path / "session" / "session.json").write_text("{}")
    got, refused = PF.verdicts_v2(PF.Proof(circuit, public, (tmp_path / "session",)),
                                  circuit_sha512=PF.file_sha512(circuit), program=DESCRIPTOR, partition=partition)
    assert refused is None, refused
    (f,) = got
    assert (f.accepted, f.why, f.units) == (False, f"setup: {CHECK_OWN}", ()), f


@pytest.mark.parametrize("statement", WRONG_TAGS)
def test_verify_zk_refuses_the_honest_session_under_a_statement_its_files_dont_fit(fx, statement):
    v, rc = _verify(fx, "honest", statement)
    at_generation = _cases(fx)["honest"]["wrong_tags"][statement]
    assert not at_generation["accepted"], at_generation
    if at_generation["digest"] is None:
        assert v is None and rc != 0, (v, rc)
    else:
        assert v is not None and v.get("accepted") is False and v.get("why") == at_generation["why"], v
