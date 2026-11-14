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

`ks-form` then holds the public file's header and rows as a session's setup takes them, with no session
(`Flock.PortRoots.headerOf`), which the per-member form's run reads for every member's files (`members.flock_form`):
it passes the honest files, and refuses the honest public file with a header naming ``public_inputs`` (the circuit's
META names none), with a row changed, and with another ``frame``, ``row_key_sha512`` or ``schemas`` entry.

The sessions are a fetched fixture (`artifacts.json`), which `suites.py` fetches before the suite runs, so a
missing fixture fails these tests rather than skipping them."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from verity.protocols.proofs.spot.verifier.one_stage import members as M, proofs as PF

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "form-refusals"
VERIFIER = Path(__file__).resolve().parents[3] / "flock" / "prover" / "agreement"
ROOT = Path(__file__).resolve().parents[6]
N = 4
WRONG_TAGS = ("verity/flock-circuit/types", "verity/flock-circuit@eb90718f", "verity/flock-circuit+plain-leaves",
              "verity/flock-circuit/tmpl")
FORM = ": a statement form the soundness proof does not cover"
#: `Flock.PortRoots.INPUTS_WHY` and `HEADER_WHY`
INPUTS_WHY = "a public file whose header names public_inputs at a circuit whose META names none"
HEADER_WHY = "a public file whose rows are not its header's"
#: `Flock.HmOut.checkPublic`'s refusal of another frame or row key
NOT_FRAMED = "the instance file's rows are not frame-v3-sha512 hm96 leaves under the default key"

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


def _public(fx: Path, at: Path, header=None, rows=None) -> Path:
    """The honest case's public file written to ``at``, its header changed by ``header`` and its rows through
    ``rows``."""
    line, body = (fx / "honest" / f"pub-{N}.bin").read_bytes().split(b"\n", 1)
    h = json.loads(line)
    if header is not None:
        header(h)
    at.write_bytes(json.dumps(h, separators=(",", ":"), sort_keys=True).encode() + b"\n"
                   + (body if rows is None else rows(body)))
    return at


def test_the_form_check_holds_the_public_files_header_and_rows_with_no_session(fx, tmp_path):
    """`members.flock_form`, the executable's ``ks-form``, on the honest circuit: the honest public file passes, as
    written and with its header rewritten; a header naming ``public_inputs``, a changed row, and another frame, row
    key or schema are each refused with `Flock.PortRoots.headerOf`'s reason."""
    circuit = fx / "honest" / "stage" / "circuit.txt"

    def form(k: str, **change) -> str | None:
        return M.flock_form(circuit, _public(fx, tmp_path / k, **change), [])

    assert M.flock_form(circuit, fx / "honest" / f"pub-{N}.bin", []) is None
    assert form("rewritten") is None
    assert form("inputs", header=lambda h: h.update(public_inputs=[])) == f"refused: {INPUTS_WHY}{FORM}"
    row = form("row", rows=lambda body: bytes([body[0] ^ 1]) + body[1:])
    assert row is not None and row.startswith(f"refused: {HEADER_WHY} (row ") and row.endswith(
        f": the rows' b || c do not hash to the header's frame-v3-sha512 root){FORM}"), row
    assert form("frame", header=lambda h: h["frame_v3"].update(frame="frame-v3")) == (
        f"refused: {HEADER_WHY} ({NOT_FRAMED}){FORM}")
    assert form("row-key", header=lambda h: h["frame_v3"].update(row_key_sha512="00" * 64)) == (
        f"refused: {HEADER_WHY} ({NOT_FRAMED}){FORM}")
    assert form("schema", header=lambda h: h["frame_v3"]["schemas"].update(r0="hm96-sha512/row/v1")) == (
        f"refused: {HEADER_WHY} (row r0: schema is not hm96-sha512/row/v2){FORM}")


@pytest.mark.parametrize("statement", WRONG_TAGS)
def test_verify_zk_refuses_the_honest_session_under_a_statement_its_files_dont_fit(fx, statement):
    v, rc = _verify(fx, "honest", statement)
    at_generation = _cases(fx)["honest"]["wrong_tags"][statement]
    assert not at_generation["accepted"], at_generation
    if at_generation["digest"] is None:
        assert v is None and rc != 0, (v, rc)
    else:
        assert v is not None and v.get("accepted") is False and v.get("why") == at_generation["why"], v
