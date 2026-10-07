"""The canonical I/O format (verifier PROTOCOL.md §16.16, V1): under `--zk`, the form check `verify` ends a session's verdict
with (`ProvedScope.check`) passes a statement with hidden outputs whose unit returns the outputs and whose rows are all
unsegmented `sha512/row/v2` rows, each output row exactly its ports' words (`ProvedScope.canonicalOk`). Without `--zk` the
same circuit is refused by the v1-row rule. Under it a circuit outside the format is refused for the first conjunct it
fails (`ProvedScope.canonicalWhy`), and a shared-row public file, the form registered reads take, is refused before
either. A canonical statement with a public input port sets up under the verifier's own `--public-inputs` file and passes
under `--zk` alone; a file that is not the verifier's own is refused at setup. A stratified
draw at a statement only the format admits is refused (`ProvedScope.canonicalLaw`). The statements are
`test_lean_verifier`'s toys, none with a proof, and `verify` reaches the form check only in a session that passed every
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


def _files(d: Path, zk: bool, own: Path | None = None) -> list[str]:
    return ["--statement", STATEMENT, "--circuit", str(d / "circuit.txt"), "--public", str(d / "pub.bin"),
            *(["--zk"] if zk else []), *(["--public-inputs", str(own)] if own else [])]


def _scope(d: Path, zk: bool, own: Path | None = None) -> dict:
    """`flock-verify scope` at the statement staged in ``d``, ``own`` the verifier's `--public-inputs` file: the form check's
    ``ok`` and ``why``, and ``canonical``."""
    p = subprocess.run([str(BIN), "scope", *_files(d, zk, own)], capture_output=True, text=True)
    (v,) = [json.loads(ln[6:]) for ln in p.stdout.splitlines() if ln.startswith("SCOPE ")]
    assert p.returncode == (0 if v["ok"] else 1), p.stdout + p.stderr
    return v


def _why(d: Path, zk: bool, own: Path | None = None) -> str:
    """The reason `verify` refuses a session of an empty record at the statement staged in ``d``, ``own`` the verifier's
    `--public-inputs` file."""
    s = d / "session"
    s.mkdir(exist_ok=True)
    (s / "session.json").write_text("{}")
    p = subprocess.run([str(BIN), "verify", *_files(d, zk, own), "--session", str(s)], capture_output=True, text=True)
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


#: a public input port (PROTOCOL.md §16.10) of 8 words, which the units read as their leaf group's word 0 (leaves 0 to 7)
X = {"port": "x", "words": 8, "seed": "toy/x", "derivation": "toy/x/v1"}


def _public_word_0(meta: dict) -> None:
    meta["public_inputs"] = [X]
    meta["leaves_public"] = [[j if j < 8 else -1 for j in range(len(li))] for li in meta["leaves_in"]]
    meta["leaves_in"] = [[-1 if j < 8 else w for j, w in enumerate(li)] for li in meta["leaves_in"]]


@pytest.fixture(scope="module")
def public_input_toy(tmp_path_factory):
    """The one-port canonical toy with its leaf group's word 0 read from port ``x`` in place of its row's first 8 words, and
    ``x``'s words those 8 words (the toy's seed 1), staged with the verifier's own file ``own.bin``. The composer makes no
    such statement (`compose` puts no public input beside a bit row), so META is edited as `test_lean_public_inputs` does."""
    import numpy as np

    from verity_flock import circuit as FC

    d = tmp_path_factory.mktemp("public-input")
    words = np.random.default_rng(1).integers(0, 1 << 16, (4, 64), dtype=np.uint64).astype(np.uint16)[:, :8]
    text, _, _ = _seg_writer_toy([16], d, seg=False, edit=_public_word_0, public_words={"x": words})
    FC.write_public_inputs(text, {"x": words}, 4, d / "own.bin")
    return d


def test_a_canonical_statement_with_a_public_input_port_passes_under_zk_alone(public_input_toy):
    """With the verifier's own file the statement sets up (a session meets its own checks, here its missing proof file),
    and the form check passes it under `--zk` by the format alone (`ZkPubIn.zk_session_sound_canonicalP` is the claim at
    it) and refuses it for its public input ports without: no claim covers public inputs outside the canonical format.
    The live server gives the same verdicts (`the_canonical_format_is_accepted_under_zk_alone`). Without the file, setup
    refuses it."""
    d = public_input_toy
    assert _scope(d, zk=True, own=d / "own.bin") == {"ok": True, "why": "", "canonical": True}
    v = _scope(d, zk=False, own=d / "own.bin")
    assert (v["ok"], v["why"]) == (False, _refusal("public input ports (--public-inputs)")), v
    why = _why(d, zk=True, own=d / "own.bin")
    assert why.startswith("session files: ") and why.endswith("/session/circuit.rep0.bin"), why
    assert _why(d, zk=True) == ("setup: public inputs: the circuit names public input ports; pass the verifier's own "
                                "recomputation (--public-inputs F)")


#: a `--public-inputs` file that is not the verifier's own, as ``edit(header, body)`` makes it from the own file, and why
#: setup refuses it
NOT_OWN = {
    "a-word-changed": (lambda h, b: (h, bytes([b[0] ^ 1]) + b[1:]),
                       "public input x: word 0 of instance 0 is not the verifier's own recomputation (seed toy/x, "
                       "derivation toy/x/v1)"),
    "another-derivation": (lambda h, b: ({**h, "ports": [{**X, "derivation": "toy/x/v0"}]}, b),
                           "--public-inputs: its ports (names, words, seeds, derivations) are not the circuit's "
                           "public_inputs"),
}


@pytest.mark.parametrize("name", list(NOT_OWN))
def test_a_public_inputs_file_not_the_verifiers_own_is_refused(public_input_toy, name, tmp_path):
    edit, why = NOT_OWN[name]
    head, body = (public_input_toy / "own.bin").read_bytes().split(b"\n", 1)
    head, body = edit(json.loads(head), body)
    own = tmp_path / "own.bin"
    own.write_bytes(json.dumps(head, sort_keys=True).encode() + b"\n" + body)
    assert _why(public_input_toy, zk=True, own=own) == f"setup: {why}"


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
