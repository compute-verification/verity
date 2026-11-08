"""`verify --zk` itself refuses a statement outside the form the sampled audit's knowledge soundness covers
(`Flock.KsScope`, `Flock.ProvedScope`), on sessions the prover made in full, and so does the v2 audit's own run of it.

The sessions are M0 `--zk` sessions of one small A8 unit (LlBf16DotprodDot_v1 at K=128, four instances, every unit
sampled, bound to its template query's partition), written by `benchmarks/one_stage/form_refusals.sh`: ``honest``, x then
w in one input row, and ``two-groups``, x and w in two, so its unit reads two input groups. `ProvedScope.check` admits
both; only `KsScope.filesWhy` refuses the second, and `verify` gives a form's reason only once every other check, the
proof's included, has passed, so that reason says the session was otherwise accepted. The honest session under a
statement `verify --zk` must refuse (`WRONG_TAGS`: typed, public outputs, plain leaves, templated) is refused, each by the
check that catches it as at generation.

`Flock.KsScope`'s own tests run its check functions on Lean values, and `flock-verify ks-form` runs the check alone; these are
the tests that `verify --zk`'s verdict calls them. Each verify sets up an m = 25 table (about 10 s), so they are slow.

The sessions are a fetched fixture (`fixtures/artifacts.json`), which `suites.py` fetches before the suite runs, so a
missing fixture fails these tests rather than skipping them."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from verity.core.protocols.sampled_proofs.one_stage import proofs as PF

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "form-refusals"
VERIFIER = Path(__file__).resolve().parents[4] / "ml" / "flock" / "verifier"
N = 4
WRONG_TAGS = ("verity/flock-circuit/types", "verity/flock-circuit@eb90718f", "verity/flock-circuit+plain-leaves",
              "verity/flock-circuit/tmpl")
SCOPE_REFUSAL = PF._refusal(PF.SCOPE_WHY)

pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")]


@pytest.fixture(scope="module")
def fx(tmp_path_factory) -> Path:
    assert (FIXTURE / "cases.json").is_file(), (
        f"{FIXTURE} is not fetched: `uv run research data fetch-fixtures {FIXTURE.relative_to(FIXTURE.parents[6])}`")
    sys.path.insert(0, str(VERIFIER))
    import ci

    root = ci.canonical_tree(FIXTURE, tmp_path_factory.mktemp("store") / "form-refusals")
    PF.executable()
    return root


def _cases(fx: Path) -> dict:
    return json.loads((fx / "cases.json").read_text())["cases"]


def _verify(fx: Path, case: str, statement: str = PF.STATEMENT) -> dict:
    s = fx / case / "stage"
    p = subprocess.run([str(PF.executable()), "verify", "--zk", "--statement", statement, "--circuit", str(s / "circuit.txt"),
                        "--public", str(fx / case / f"pub-{N}.bin"), "--partition", str(s / "partition.json"),
                        "--program", str(s / "program.json"), "--session", str(fx / case / "session")],
                       capture_output=True, text=True)
    (line,) = [x for x in p.stdout.splitlines() if x.startswith("VERDICT ")] or [f"VERDICT {json.dumps({'why': p.stderr[-400:]})}"]
    return json.loads(line[len("VERDICT "):])


def _v2(fx: Path, case: str) -> PF.Finding:
    s = fx / case / "stage"
    got, refused = PF.verdicts_v2(PF.Proof(s / "circuit.txt", fx / case / f"pub-{N}.bin", (fx / case / "session",)),
                                  circuit_sha512=PF.file_sha512(s / "circuit.txt"),
                                  program=json.loads((s / "program.json").read_bytes()),
                                  partition=json.loads((s / "partition.json").read_text()))
    assert refused is None, refused
    (finding,) = got
    return finding


def test_verify_zk_accepts_the_honest_session(fx):
    v = _verify(fx, "honest")
    assert v.get("accepted") is True, v


def test_verify_zk_refuses_a_unit_of_two_input_groups_for_its_form(fx):
    """The session's proof checks and every check before the form pass; `KsScope.filesWhy` refuses its circuit."""
    v = _verify(fx, "two-groups")
    assert v.get("accepted") is False and v.get("why") == SCOPE_REFUSAL, v


def test_the_v2_audits_own_run_accepts_the_honest_session_and_refuses_the_other_for_its_form(fx):
    honest, two = _v2(fx, "honest"), _v2(fx, "two-groups")
    assert honest.accepted and honest.units == tuple(range(N)), honest
    assert (two.accepted, two.why) == (False, SCOPE_REFUSAL), two


@pytest.mark.parametrize("statement", WRONG_TAGS)
def test_verify_zk_refuses_the_honest_session_under_a_statement_outside_the_form(fx, statement):
    v = _verify(fx, "honest", statement)
    assert v.get("accepted") is False, v
    assert v.get("why") == _cases(fx)["honest"]["wrong_tags"][statement]["why"], v
