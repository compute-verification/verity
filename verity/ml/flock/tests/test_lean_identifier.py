"""A statement's identifier (PROTOCOL.md §16.10): the public header's `identifier`, a JSON object C-Flock does not read, must be
the verifier's own `--identifier F` in RFC 8785 form, or the Lean verifier refuses the statement; a header with one and no
`F`, and `F` with a header that names none, refuse too. The public digest binds the header, so Σ binds the identifier, and a
session's verdict reports it. The fixture's public input ports are a form the soundness proof does not cover (§16.15), so
`verify` refuses its sessions, for that once every check has passed.

The statement is `test_lean_public_inputs.py`'s fetched fixture (RoPE x4, its recorded CPU selftest sessions), with the
identifier added to its public file's header. It catches what `test_circuit.py` can't: the executable verifier's own
refusals, on the statement's and the session's setup paths. Each case sets up the statement once (seconds)."""
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
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "public-inputs"
STATEMENT = "verity/flock-circuit@f5df3bbc+seed-injection"
#: PoUW's (H, s, U) stand-in: C-Flock reads none of it
IDENTIFIER = {"scheme": "test/identifier/v0", "H": "ab" * 32, "s": 7, "U": [3, 1, 2], "nested": {"b": None, "a": True}}

OUTSIDE = "identifier: --identifier F is outside canonical JSON's RFC 8785 domain: "
FORM_REFUSAL = "refused: public input ports (--public-inputs): a statement form the soundness proof does not cover"

pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")]


@pytest.fixture(scope="module")
def store():
    assert (FIXTURE / "stage" / "circuit.txt").is_file(), (
        f"{FIXTURE} is not fetched: `uv run research data fetch-fixtures verity/ml/flock/tests/fixtures/{FIXTURE.name}`")
    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)
    return FIXTURE


def _pub(store: Path, tmp: Path, identifier=None, name: str = "pub.bin") -> Path:
    head, body = (store / "stage" / "pub-4.bin").read_bytes().split(b"\n", 1)
    head = json.loads(head)
    if identifier is not None:
        head["identifier"] = identifier
    (tmp / name).write_bytes(json.dumps(head, sort_keys=True).encode() + b"\n" + body)
    return tmp / name


def _own(tmp: Path, text: str, name: str = "identifier.json") -> Path:
    (tmp / name).write_text(text)
    return tmp / name


def _flags(store: Path, pub: Path, own: Path | None) -> list[str]:
    return ["--statement", STATEMENT, "--circuit", str(store / "stage" / "circuit.txt"), "--public", str(pub),
            "--public-inputs", str(store / "stage" / "public-inputs-4.bin"), *(["--identifier", str(own)] if own else [])]


def _statement(store: Path, pub: Path, own: Path | None) -> subprocess.CompletedProcess:
    return subprocess.run([str(BIN), "statement", *_flags(store, pub, own)], capture_output=True, text=True)


def _verdict(store: Path, pub: Path, own: Path | None, tmp: Path) -> dict:
    subprocess.run([sys.executable, str(VERIFIER / "selftest_records.py"), str(store / "records-cpu"), str(tmp / "s")],
                   check=True, capture_output=True)
    p = subprocess.run([str(BIN), "verify", *_flags(store, pub, own), "--session", str(tmp / "s" / "honest")],
                       capture_output=True, text=True)
    [v] = [json.loads(ln[8:]) for ln in p.stdout.splitlines() if ln.startswith("VERDICT ")]
    return v


def test_the_verifiers_own_identifier_is_accepted_in_any_json_form_and_bound_by_sigma(store, tmp_path):
    """F's key order and whitespace don't matter (RFC 8785 form); the statement reports the identifier, and its public digest
    and Σ are not the header's without it."""
    own = _own(tmp_path, json.dumps(dict(reversed(list(IDENTIFIER.items()))), indent=2))
    p = _statement(store, _pub(store, tmp_path, IDENTIFIER), own)
    assert p.returncode == 0, p.stdout
    st = json.loads(p.stdout)
    plain = json.loads(_statement(store, store / "stage" / "pub-4.bin", None).stdout)
    assert st["identifier"] == IDENTIFIER and plain["identifier"] is None
    assert st["public_sha512"] != plain["public_sha512"] and st["sigma"] != plain["sigma"]


def test_the_verdict_reports_the_identifier(store, tmp_path):
    """The prover's honest session, over the header without an identifier and with one: both set up. Over the header it was
    proved against, the session passes every check and `verify` refuses it for its form (public input ports, §16.15); over
    the header with the identifier its Hello's Σ is not the statement's (S2: Σ binds the identifier, as the test above shows
    on the statement), and the verdict reports the identifier."""
    plain = _verdict(store, store / "stage" / "pub-4.bin", None, tmp_path)
    assert (plain["accepted"], plain["why"]) == (False, FORM_REFUSAL) and "identifier" not in plain, plain["why"]
    v = _verdict(store, _pub(store, tmp_path, IDENTIFIER), _own(tmp_path, json.dumps(IDENTIFIER)), tmp_path)
    assert not v["accepted"] and v["why"].startswith("S2/R7") and v["identifier"] == IDENTIFIER, v["why"]


#: each refusal: the header's identifier (None: no key), F's text (None: no flag), and the setup refusal
REFUSALS = {
    "mismatch": (IDENTIFIER, json.dumps(dict(IDENTIFIER, s=8)),
                 "identifier: the public header's identifier is not the verifier's own (--identifier F)"),
    "identifier_without_flag": (IDENTIFIER, None,
                                "identifier: the public header names an identifier; pass the verifier's own (--identifier F)"),
    "flag_without_identifier": (None, json.dumps(IDENTIFIER),
                                "identifier: the verifier holds an identifier (--identifier F), and the public header names none"),
    "not_an_object": ([1, 2], json.dumps(IDENTIFIER), "identifier: the public header's identifier is not a JSON object"),
    "outside_rfc8785": ({"s": 2 ** 53}, json.dumps({"s": 2 ** 53 - 1}),
                        "identifier: the public header's identifier is outside canonical JSON's RFC 8785 domain: identifier.s: "
                        "integer 9007199254740992 exceeds 2^53 - 1 in magnitude"),
}


@pytest.mark.parametrize("name", list(REFUSALS))
def test_the_statement_refuses_an_identifier_that_is_not_the_verifiers_own(store, name, tmp_path):
    header, own, why = REFUSALS[name]
    p = _statement(store, _pub(store, tmp_path, header), None if own is None else _own(tmp_path, own))
    assert p.returncode == 1 and p.stdout.strip() == f"setup: {why}", (p.stdout, p.stderr)


@pytest.mark.parametrize("name", ["mismatch", "identifier_without_flag", "flag_without_identifier"])
def test_the_session_is_refused_at_setup(store, name, tmp_path):
    header, own, why = REFUSALS[name]
    v = _verdict(store, _pub(store, tmp_path, header), None if own is None else _own(tmp_path, own), tmp_path)
    assert not v["accepted"] and v["why"] == f"setup: {why}", v["why"]


@pytest.mark.parametrize("text, why", [
    ("[1, 2]", "identifier: not a JSON object"),
    ('{"s": 1.5}', f"{OUTSIDE}identifier.s: a float is not a JSON integer, string, boolean, null, array or object"),
    ('{"s": "caf\\u00e9"}', f"{OUTSIDE}identifier.s: string is not ASCII without DEL"),
    ("{", None),
])
def test_the_verifiers_own_file_is_an_rfc8785_object(store, text, why, tmp_path):
    p = _statement(store, _pub(store, tmp_path, IDENTIFIER), _own(tmp_path, text))
    assert p.returncode == 2 and p.stdout == "", p.stdout
    assert p.stderr.strip() == why if why else p.stderr.startswith("identifier: not JSON"), p.stderr
