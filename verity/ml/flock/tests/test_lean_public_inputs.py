"""Public input ports (PROTOCOL.md §16.10): the Lean verifier reads a hidden-output statement whose unit reads a public
input port (RoPE's `cs`, its cos/sin table) from in regions, as the Rust prover writes it, on its recorded CPU selftest
sessions, with the public words in one aligned block of the unit's leaf group and in two, and holds the public file's words to
the verifier's own recomputation (`--public-inputs F`). Public input ports are a form the soundness proof does not cover
(§16.15), so once the statement sets up `verify` refuses every session of it, for that once every check has passed.

It is the one test in `check` of public input ports against the prover's sessions: check's agreement step replays only
upstream's pinned builds, none of which has them. So it catches a verifier that disagrees with the prover on the in regions
or on the public file's tail (the statement's Hello is the prover's), which no statement-only test sees, and it runs each
refusal of the port's condition (only the verifier's own recomputation, named by seed and derivation, is public) on a real
statement. Each case sets up the statement once (seconds); the cases run in parallel workers.

The set is a fetched fixture (`fixtures/artifacts.json`), which `suites.py` fetches before the suite runs and fails the suite
when it cannot, so a missing set fails this test rather than skipping it."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
VERIFIER = Path(__file__).resolve().parents[1] / "verifier"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
#: `stage/`: `circuit.stage(IL.lowering(rope_head.definition(sub), rope_head.UNIT + "/x4", group=4), set, 0, 4, out,
#: public_inputs=PUBLIC)` on `generate(rope-head/d64/neox-bf16, 4, 20260926)`; `records-cpu/`: `flock-circuit selftest
#: --instances inst-4.bin --circuit circuit.txt --record-dir` (CPU, non-zk) at 54146af61
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "public-inputs"
#: the same set under the x8 lowering (`cs` in words 2..3 of a unit's leaf group), its META edited so that each unit's second
#: x word (leaves 8..15) is a second public port `xh` (`XH`): public words 1..3, two in regions of blocks (1, 0) and (2, 1)
#: (`circuit.aligned_blocks`), staged with `circuit.write(..., public_words=)` and `write_public_inputs`; records as above at
#: 4d53e4749
BLOCKS = Path(__file__).resolve().parent / "fixtures" / "public-inputs-blocks"
PUBLIC = {"cs": {"seed": "rope/positions", "derivation": "verity/rope/neox-bf16/cos-sin/v1"}}
XH = {"port": "xh", "words": 32, "seed": "test/rope-x", "derivation": "test/rope-x/second-word"}
STATEMENT = "verity/flock-circuit@f5df3bbc+seed-injection"
CASES = {"honest": True, "public_input_claim_false": False, "output_claim_false": False, "digest_claim_false": False,
         "output_row_committed_false": False}
OWN_REFUSAL = "is not the verifier's own recomputation (seed rope/positions, derivation verity/rope/neox-bf16/cos-sin/v1)"
FORM_REFUSAL = "refused: public input ports (--public-inputs): a statement form the soundness proof does not cover"

pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")]


@pytest.fixture(scope="module")
def stores():
    for d in (FIXTURE, BLOCKS):
        assert (d / "stage" / "circuit.txt").is_file(), (
            f"{d} is not fetched: `uv run research data fetch-fixtures verity/ml/flock/tests/fixtures/{d.name}`")
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    return {"one-block": FIXTURE, "two-blocks": BLOCKS}


@pytest.fixture(scope="module")
def store(stores):
    return stores["one-block"]


def _split(f: Path) -> tuple[dict, bytes]:
    head, body = f.read_bytes().split(b"\n", 1)
    return json.loads(head), body


def _join(head: dict, body: bytes, f: Path) -> Path:
    f.write_bytes(json.dumps(head, sort_keys=True).encode() + b"\n" + body)
    return f


def _words(d: Path) -> int:
    meta = json.loads((d / "stage" / "circuit.txt").read_text().split("\n", 2)[1][5:])
    return 2 * 4 * sum(p["words"] for p in meta["public_inputs"])


def _recircuit(d: Path, tmp: Path, edit) -> tuple[Path, Path]:
    """The stage's circuit with META edited by `edit`, and its public file's header naming it (pin and class)."""
    from verity_flock import circuit as FC
    head, meta_line, rest = (d / "stage" / "circuit.txt").read_text().split("\n", 2)
    meta = json.loads(meta_line[5:])
    edit(meta)
    text = head + "\nMETA " + json.dumps(meta, sort_keys=True, separators=(",", ":")) + "\n" + rest
    (tmp / "circuit.txt").write_text(text)
    h, body = _split(d / "stage" / "pub-4.bin")
    h["circuit_sha512"] = FC.digest(text)
    h["units"]["classes"] = [FC.class_digest(text)] * len(h["units"]["classes"])
    if "public_inputs" not in meta:
        h.pop("public_inputs")
        body = body[:-_words(d)]
    return tmp / "circuit.txt", _join(h, body, tmp / "pub.bin")


def _statement(circuit: Path, pub: Path, own: Path | None, statement: str = STATEMENT) -> subprocess.CompletedProcess:
    flags = ["--public-inputs", str(own)] if own else []
    return subprocess.run([str(BIN), "statement", "--statement", statement, "--circuit", str(circuit), "--public", str(pub),
                           *flags], capture_output=True, text=True)


def test_the_files_name_the_ports_recomputation(store):
    meta = json.loads((store / "stage" / "circuit.txt").read_text().split("\n", 2)[1][5:])
    pins = [{"port": p, "words": 64, **r} for p, r in PUBLIC.items()]
    assert meta["public_inputs"] == pins and [p[0] for p in meta["ports"]] == ["x", "out"]
    head, body = _split(store / "stage" / "public-inputs-4.bin")
    assert head == {"format": "verity/flock-public-inputs/v1", "instances": 4, "ports": pins}
    pub_head, pub_body = _split(store / "stage" / "pub-4.bin")
    assert pub_head["public_inputs"] == pins and pub_body.endswith(body) and len(body) == _words(store)


@pytest.mark.parametrize("layout", ["one-block", "two-blocks"])
@pytest.mark.parametrize("case", list(CASES))
def test_it_refuses_the_provers_sessions(stores, layout, case, tmp_path):
    """Each session under the verifier's own recomputation of its public file's words, with the public words in one aligned
    block and in two: the statement sets up, a false public word's too once the verifier's file says the same word, the
    honest session meets the refusal of its form, and each false statement the check that catches it, not its parameters."""
    store = stores[layout]
    rec = store / "records-cpu"
    subprocess.run([sys.executable, str(VERIFIER / "selftest_records.py"), str(rec), str(tmp_path)], check=True,
                   capture_output=True)
    s = tmp_path / case
    assert json.loads((s / "case.json").read_text())["live"]["accepted"] == CASES[case]
    pub = s / "pub.bin" if (s / "pub.bin").exists() else store / "stage" / "pub-4.bin"
    head, _ = _split(store / "stage" / "public-inputs-4.bin")
    own = _join(head, pub.read_bytes()[-_words(store):], tmp_path / "own.bin")
    p = subprocess.run([str(BIN), "verify", "--statement", STATEMENT, "--circuit", str(store / "stage" / "circuit.txt"),
                        "--public", str(pub), "--public-inputs", str(own), "--session", str(s)], capture_output=True, text=True)
    [v] = [json.loads(ln[8:]) for ln in p.stdout.splitlines() if ln.startswith("VERDICT ")]
    assert not v["accepted"] and (v["why"] == FORM_REFUSAL) == CASES[case], v
    if case != "honest":
        assert not v["why"].startswith(("refused:", "setup:", "S2")), v


def test_a_session_that_fails_a_check_keeps_its_reason(store, tmp_path):
    """The form's refusal comes once every check has passed (§16.15): in one verifier run, the honest session meets it, and
    the same session with one byte of a proof changed meets the check that catches the change instead."""
    subprocess.run([sys.executable, str(VERIFIER / "selftest_records.py"), str(store / "records-cpu"), str(tmp_path / "s")],
                   check=True, capture_output=True)
    honest = tmp_path / "s" / "honest"
    edited = shutil.copytree(honest, tmp_path / "edited")
    proof = bytearray((edited / "circuit.rep0.bin").read_bytes())
    proof[len(proof) // 2] ^= 1
    (edited / "circuit.rep0.bin").write_bytes(bytes(proof))
    head, _ = _split(store / "stage" / "public-inputs-4.bin")
    own = _join(head, (store / "stage" / "pub-4.bin").read_bytes()[-_words(store):], tmp_path / "own.bin")
    p = subprocess.run([str(BIN), "verify", "--statement", STATEMENT, "--circuit", str(store / "stage" / "circuit.txt"),
                        "--public", str(store / "stage" / "pub-4.bin"), "--public-inputs", str(own),
                        "--session", str(honest), str(edited)], capture_output=True, text=True)
    good, bad = [json.loads(ln[8:]) for ln in p.stdout.splitlines() if ln.startswith("VERDICT ")]
    assert (good["accepted"], good["why"]) == (False, FORM_REFUSAL), good
    assert not bad["accepted"] and not bad["why"].startswith(("refused:", "setup:")), bad


def _missing(d, tmp):
    return d / "stage" / "circuit.txt", d / "stage" / "pub-4.bin", None


def _public_word_changed(d, tmp):
    false = d / "records-cpu" / "public_input_claim_false.pub.bin"
    return d / "stage" / "circuit.txt", false, d / "stage" / "public-inputs-4.bin"


def _own_word_changed(d, tmp):
    head, body = _split(d / "stage" / "public-inputs-4.bin")
    return (d / "stage" / "circuit.txt", d / "stage" / "pub-4.bin",
            _join(head, body[:-2] + bytes([body[-2] ^ 1, body[-1]]), tmp / "own.bin"))


def _own_derivation(d, tmp):
    head, body = _split(d / "stage" / "public-inputs-4.bin")
    head["ports"][0]["derivation"] = "verity/rope/neox-bf16/cos-sin/v0"
    return d / "stage" / "circuit.txt", d / "stage" / "pub-4.bin", _join(head, body, tmp / "own.bin")


def _own_instances(d, tmp):
    head, body = _split(d / "stage" / "public-inputs-4.bin")
    head["instances"] = 8
    return d / "stage" / "circuit.txt", d / "stage" / "pub-4.bin", _join(head, body + body, tmp / "own.bin")


def _header_unnamed(d, tmp):
    head, body = _split(d / "stage" / "pub-4.bin")
    head.pop("public_inputs")
    return d / "stage" / "circuit.txt", _join(head, body, tmp / "pub.bin"), d / "stage" / "public-inputs-4.bin"


def _meta_edit(edit):
    def case(d, tmp):
        return (*_recircuit(d, tmp, edit), d / "stage" / "public-inputs-4.bin")
    return case


def _no_seed(meta):
    del meta["public_inputs"][0]["seed"]


def _empty_derivation(meta):
    meta["public_inputs"][0]["derivation"] = ""


def _no_port(meta):
    meta.pop("public_inputs")
    meta.pop("leaves_public")


def _leaves_only(meta):
    meta.pop("public_inputs")


def _row_beside_public(meta):
    # unit 0's leaf 7, a word of x, read from the public words instead: word 0 then holds both rows and a public word
    meta["leaves_in"][0][7], meta["leaves_public"][0][7] = -1, 0


def _non_hidden(d, tmp):
    return d / "stage" / "circuit.txt", d / "stage" / "pub-4.bin", d / "stage" / "public-inputs-4.bin"


#: each refusal of the port's condition: the case's (circuit, public file, the verifier's own file), and the refusal
REFUSALS = {
    "missing_public_inputs": (_missing, "public inputs: the circuit names public input ports; pass the verifier's own "
                                        "recomputation (--public-inputs F)"),
    "public_word_changed": (_public_word_changed, "public input cs: word 0 of instance 0 " + OWN_REFUSAL),
    "own_word_changed": (_own_word_changed, "public input cs: word 63 of instance 3 " + OWN_REFUSAL),
    "own_derivation_differs": (_own_derivation, "--public-inputs: its ports (names, words, seeds, derivations) are not the "
                                                "circuit's public_inputs"),
    "own_instances_differ": (_own_instances, "--public-inputs: not the population's 4 instances"),
    "header_names_no_port": (_header_unnamed, "public file: header public_inputs is not the circuit's"),
    "meta_names_no_seed": (_meta_edit(_no_seed), "META: a public input names no port, words, or recomputation (a non-empty "
                                                 "seed and derivation)"),
    "meta_empty_derivation": (_meta_edit(_empty_derivation), "META: a public input names no port, words, or recomputation "
                                                             "(a non-empty seed and derivation)"),
    "circuit_names_no_port": (_meta_edit(_no_port), "--public-inputs: the circuit names no public input port"),
    "leaves_without_ports": (_meta_edit(_leaves_only), "META: leaves_public without public_inputs"),
    "row_word_beside_public": (_meta_edit(_row_beside_public), "public inputs: leaf 0 reads a row word inside word 0, which "
                                                               "holds a public word"),
}


@pytest.mark.parametrize("name", list(REFUSALS))
def test_the_statement_refuses_what_is_not_the_verifiers_own(store, name, tmp_path):
    build, why = REFUSALS[name]
    p = _statement(*build(store, tmp_path))
    assert p.returncode == 1 and p.stdout.strip() == f"setup: {why}", (p.stdout, p.stderr)


def test_a_statement_without_hidden_outputs_takes_no_public_inputs(store):
    for statement in ("verity/flock-circuit@967b8d06+seed-injection", "verity/flock-circuit@967b8d06"):
        p = _statement(*_non_hidden(store, None), statement=statement)
        assert p.returncode == 1 and p.stdout.startswith(f"setup: --public-inputs: a {statement} statement has no public "
                                                         "input ports"), (p.stdout, p.stderr)


@pytest.mark.parametrize("layout, regions", [("one-block", 3), ("two-blocks", 4)])
def test_the_honest_statement_is_the_provers(stores, layout, regions):
    """The statement's Hello is the prover's: two digest regions (x and out), then an in region per block."""
    store = stores[layout]
    p = _statement(store / "stage" / "circuit.txt", store / "stage" / "pub-4.bin", store / "stage" / "public-inputs-4.bin")
    st = json.loads(p.stdout)
    record = json.loads((store / "records-cpu" / "honest.json").read_text())["record"]
    assert st["regions"] == regions and st["hello"] == record["hello"] and st["sigma"] in st["hello"]


def test_the_two_block_files_name_both_ports(stores):
    from verity_flock import circuit as FC
    meta = json.loads((stores["two-blocks"] / "stage" / "circuit.txt").read_text().split("\n", 2)[1][5:])
    assert meta["public_inputs"] == [{"port": p, "words": 64, **r} for p, r in PUBLIC.items()] + [XH]
    words = {16 * j // 128 for u in meta["leaves_public"] for j, q in enumerate(u) if q >= 0}
    assert FC.aligned_blocks(sorted(words)) == [(1, 0), (2, 1)]
    head, body = _split(stores["two-blocks"] / "stage" / "public-inputs-4.bin")
    assert head["ports"] == meta["public_inputs"] and len(body) == _words(stores["two-blocks"])
