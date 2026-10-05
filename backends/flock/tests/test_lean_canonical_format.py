"""The canonical I/O format (verifier PROTOCOL.md §16.16, V1): under `--zk`, the form check `verify` ends a session's verdict
with (`ProvedScope.check`) passes a statement with hidden outputs whose unit returns the outputs and whose rows are all
unsegmented `sha512/row/v2` rows, each output row exactly its ports' words (`ProvedScope.canonicalOk`). Without `--zk` the
same circuit is refused by the v1-row rule. Under it a circuit outside the format is refused for the first conjunct it
fails (`ProvedScope.canonicalWhy`), and a shared-row public file, the form registered reads take, is refused before
either. A stratified draw at a statement only the format admits is refused (`ProvedScope.canonicalLaw`). The statements
are `test_lean_verifier`'s toys, none with a proof, and `verify` reaches the form check only in a session that passed every
other check (§16.15), so the check is run alone (`flock-verify scope`), and a canonical toy's session of its record alone
shows that the statement sets up under `--zk` (its verdict is the missing proof file's). The live server's
`proved_scope::check` gives the same verdicts (`proved_scope.rs`'s `the_canonical_format_is_accepted_under_zk_alone`). Each
toy takes about 10 s to stage and 6 s to set up, so the module is slow."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_lean_verifier import LEAN, _bit_row_toy, _seg_writer_toy  # noqa: E402

BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
STATEMENT = "verity/flock-circuit"
V1 = "a row that is not a sha512/row/v1 row of 16 · words bits in at least one chunk"

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


def _refusal(why: str) -> str:
    return f"refused: {why}: a statement form the soundness proof does not cover"


@pytest.fixture(scope="module", autouse=True)
def built():
    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)


def _files(d: Path, zk: bool) -> list[str]:
    return ["--statement", STATEMENT, "--circuit", str(d / "circuit.txt"), "--public", str(d / "pub.bin"),
            *(["--zk"] if zk else [])]


def _scope(d: Path, zk: bool) -> dict:
    """`flock-verify scope` at the statement staged in ``d``: the form check's ``ok`` and ``why``, and ``canonical``."""
    p = subprocess.run([str(BIN), "scope", *_files(d, zk)], capture_output=True, text=True)
    (v,) = [json.loads(ln[6:]) for ln in p.stdout.splitlines() if ln.startswith("SCOPE ")]
    assert p.returncode == (0 if v["ok"] else 1), p.stdout + p.stderr
    return v


def _why(d: Path, zk: bool) -> str:
    """The reason `verify` refuses a session of an empty record at the statement staged in ``d``."""
    s = d / "session"
    s.mkdir(exist_ok=True)
    (s / "session.json").write_text("{}")
    p = subprocess.run([str(BIN), "verify", *_files(d, zk), "--session", str(s)], capture_output=True, text=True)
    (v,) = [json.loads(ln[8:]) for ln in p.stdout.splitlines() if ln.startswith("VERDICT ")]
    assert p.returncode == 1 and not v["accepted"], p.stdout + p.stderr
    return v["why"]


@pytest.mark.parametrize("out_words, in_bits", [([16], 1024), ([256, 32, 64], 1024), ([64], 1009)],
                         ids=["one-port", "a-run", "a-short-row"])
def test_a_canonical_statement_passes_the_form_check_under_zk_alone(tmp_path, out_words, in_bits):
    """An exact v2 output row of one port (PoUW's 16-word tile digest) or of a run of them (the nvfp4-p W row), and an input
    row of 1009 bits that every output word reads (the composer folds the bits past the row's to zeros, so the unit reads
    port bits only): under `--zk` the form check passes it by the format alone, and it sets up, so a session meets its own
    checks, here its missing proof file; without `--zk` the check refuses it by the v1-row rule."""
    _seg_writer_toy(out_words, tmp_path, seg=False, in_bits=in_bits)
    assert _scope(tmp_path, zk=True) == {"ok": True, "why": "", "canonical": True}
    assert _scope(tmp_path, zk=False) == {"ok": False, "why": _refusal(V1), "canonical": False}
    why = _why(tmp_path, zk=True)
    assert why.startswith("session files: ") and why.endswith("/session/circuit.rep0.bin"), why


#: a statement outside the canonical format, as ``stage(d)`` stages it, and the reason the form check under `--zk` refuses it
#: for
OUTSIDE = {
    "padded-output-row": (lambda d: _bit_row_toy([1024, 512], d), "an output row that is not exactly its output ports' bits"),
    "segmented-row": (lambda d: _bit_row_toy([1024, 512], d, seg=frozenset({"p0"})),
                      "a row that is not an unsegmented sha512/row/v2 row, in a circuit with sha512/row/v2 rows"),
    "shared-rows": (lambda d: _seg_writer_toy([16], d, seg=False, share_rows=True),
                    "a shared-row public file (header shared_rows)"),
}


@pytest.mark.parametrize("name", list(OUTSIDE))
def test_a_statement_outside_the_format_is_refused_for_its_first_failure(tmp_path, name):
    stage, why = OUTSIDE[name]
    stage(tmp_path)
    v = _scope(tmp_path, zk=True)
    assert (v["ok"], v["why"]) == (False, _refusal(why)), v


CANONICAL_LAW = r"""import Flock
open Flock

def law (s : String) : Option Lean.Json := (Lean.Json.parse s).toOption

def verdict : Except String Unit → String
  | .ok () => "ok"
  | .error e => e

def stratified := law "{\"law\":\"stratified\",\"population\":4,\"k\":2,\"strata\":[{\"template\":\"t\",\"units\":[[0,4]],\"k\":2}],\"units\":[1,3]}"
def subset := law "{\"law\":\"subset\",\"population\":4,\"k\":2,\"units\":[1,3]}"

#eval IO.println (Lean.toJson [verdict (ProvedScope.canonicalLaw true stratified), verdict (ProvedScope.canonicalLaw true subset),
  verdict (ProvedScope.canonicalLaw true none), verdict (ProvedScope.canonicalLaw false stratified)]).compress
"""


def test_a_stratified_draw_at_a_canonical_statement_is_refused(tmp_path):
    """Where `canonicalAt` holds a stratified draw is refused, and a subset draw, or none, passes on. No toy sets up a
    stratified draw (its unit cover names no strata), so the law is run on its own; `canonicalAt` is `!scopeOk &&
    canonicalOk`, which the form tests above pin at each canonical toy (`scope`'s ``canonical``)."""
    f = tmp_path / "CanonicalLaw.lean"
    f.write_text(CANONICAL_LAW)
    p = subprocess.run(["lake", "env", "lean", str(f)], cwd=LEAN, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert json.loads(p.stdout) == ["refused: a stratified draw at a canonical statement: no soundness claim draws by this law",
                                    "ok", "ok", "ok"]
