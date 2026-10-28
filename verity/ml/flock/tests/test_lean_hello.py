"""A firewalled session's `Hello` in Lean (`Flock/Firewall/Hello.lean`, `flock-firewall hello`), the one the ZK server's Lean
session loop sends in the prover's place (`Session.helloOf`): held byte for byte to the `Hello`s Rust's `SessionConfig::hello_with_nonce`
sent in the recorded `--zk` sessions of `test_lean_zk.py`'s fixture (one table, two tables, m = 26 and a drawn session), at
each record's nonce and draw; at a nonce of its own, a fresh one each session; and its protocol's refusals. A request is
`draw LEN [NONCE]` and `LEN` bytes of the verifier's sample; a statement the setup refuses is answered `refused WHY` and the
session goes on, a malformed request `err WHY`, which ends it with exit 1."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-firewall"
#: test_lean_zk.py's: RoPE d64 with hidden outputs and the live build's --zk sessions, under the statement it was staged for; a
#: fetched fixture (`fixtures/artifacts.json`), which `suites.py` fetches before the suite runs: `art:b9cc903c`
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "zk-sessions"
STATEMENT = "verity/flock-circuit+seed-injection"
NONCE = re.compile(r'"nonce":"([0-9a-f]{64})"')


@pytest.fixture(scope="module", autouse=True)
def built():
    if shutil.which("lake") is None:
        pytest.skip("no Lean toolchain")
    subprocess.run(["lake", "build", "flock-firewall"], cwd=LEAN, check=True, capture_output=True)


@pytest.fixture(scope="module")
def zk():
    assert (FIXTURE / "stage" / "circuit.txt").is_file(), (
        f"{FIXTURE} is not fetched: `uv run research data fetch-fixtures verity/ml/flock/tests/fixtures/zk-sessions`")
    return FIXTURE


def _request(draw: str, nonce: str | None = None) -> bytes:
    b = draw.encode()
    return f"draw {len(b)}{'' if nonce is None else ' ' + nonce}\n".encode() + b


def _hello(args: list[str], stdin: bytes) -> tuple[list[str], int]:
    """The answers, each `ok` with its `Hello`'s text, and the exit code."""
    p = subprocess.run([str(BIN), "hello", *args], input=stdin, capture_output=True)
    out, answers = p.stdout, []
    while out:
        line, _, out = out.partition(b"\n")
        text = line.decode()
        if text.startswith("ok "):
            n = int(text[3:])
            text, out = f"ok {out[:n].decode()}", out[n:]
        answers.append(text)
    return answers, p.returncode


def _args(root: Path, public: str, tables: int = 1) -> list[str]:
    return ["--statement", STATEMENT, "--circuit", str(root / "stage" / "circuit.txt"), "--public", str(root / "stage" / public),
            "--session-tables", str(tables)]


def _record(root: Path, name: str) -> tuple[str, str, str]:
    """A recorded session's `Hello`, its nonce and its sample's JSON (`null` without one): `selftest/CASE`, or a set's one
    session."""
    d = root / "sessions" / name
    if "/" not in name:
        [d] = [p for p in d.iterdir() if p.is_dir()]
    rec = json.loads((d / "session.json").read_text())
    return rec["hello"], NONCE.search(rec["hello"])[1], json.dumps(rec.get("unit_draw"))


@pytest.mark.slow
@pytest.mark.parametrize("name, public, tables", [("selftest/honest", "pub-16.bin", 1), ("m26", "pub-32.bin", 1),
                                                   ("j2", "pub-32.bin", 2), ("drawn", "pub-32.bin", 1)])
def test_the_recorded_hello_at_its_nonce(zk, name, public, tables):
    """At a record's nonce and draw, Lean's `Hello` is the one Rust sent, byte for byte."""
    hello, nonce, draw = _record(zk, name)
    answers, rc = _hello(_args(zk, public, tables), _request(draw, nonce))
    assert rc == 0 and answers == [f"ok {hello}"], answers


@pytest.mark.slow
def test_one_process_serves_every_draw(zk):
    """One process answers sessions at several draws, each sample's statement set up once: the population's, a sample's, and the
    population's again at another nonce."""
    m26, n26, null = _record(zk, "m26")
    drawn, nd, draw = _record(zk, "drawn")
    other = "ab" * 32
    answers, rc = _hello(_args(zk, "pub-32.bin"), _request(null, n26) + _request(draw, nd) + _request(null, other))
    assert rc == 0 and answers == [f"ok {m26}", f"ok {drawn}", f"ok {m26.replace(n26, other)}"], answers


@pytest.mark.slow
def test_a_nonce_of_its_own(zk):
    """Without a nonce in the request, each session's is 32 bytes the command draws: two sessions' differ, and each `Hello` is
    the recorded one at its nonce."""
    hello, nonce, draw = _record(zk, "selftest/honest")
    answers, rc = _hello(_args(zk, "pub-16.bin"), _request(draw) + _request(draw))
    assert rc == 0 and len(answers) == 2, answers
    got = [NONCE.search(a)[1] for a in answers]
    assert got[0] != got[1] and answers == [f"ok {hello.replace(nonce, g)}" for g in got], answers


@pytest.mark.slow
def test_a_draw_the_setup_refuses_and_the_session_goes_on(zk):
    """The drawn session's sample over the 16-unit population's public file names another population: refused, and the next
    request is answered."""
    hello, nonce, null = _record(zk, "selftest/honest")
    _, _, draw = _record(zk, "drawn")
    answers, rc = _hello(_args(zk, "pub-16.bin"), _request(draw, nonce) + _request(null, nonce))
    assert rc == 0 and answers[0].startswith("refused setup: ") and answers[1:] == [f"ok {hello}"], answers


@pytest.fixture
def junk(tmp_path):
    """Files no statement sets up from: requests are refused before any setup reads them, a setup refuses them."""
    (tmp_path / "c").write_bytes(b"x")
    (tmp_path / "p").write_bytes(b"x")
    return ["--circuit", str(tmp_path / "c"), "--public", str(tmp_path / "p")]


def test_end_of_input_ends_the_session(junk):
    assert _hello(junk, b"") == ([], 0)


def test_usage():
    p = subprocess.run([str(BIN), "hello"], input=b"", capture_output=True)
    assert p.returncode == 2 and p.stderr.decode().startswith("usage: flock-firewall hello")


def test_unknown_statement(junk):
    p = subprocess.run([str(BIN), "hello", "--statement", "no-such", *junk], input=b"", capture_output=True)
    assert p.returncode == 2 and p.stderr.decode().startswith("unknown statement")


def test_plain_leaves_have_no_zk_session(junk):
    p = subprocess.run([str(BIN), "hello", "--statement", "verity/flock-circuit+plain-leaves", *junk], input=b"", capture_output=True)
    assert p.returncode == 2 and "plain SHA-512 leaves" in p.stderr.decode()


@pytest.mark.parametrize("line", [b"draw\n", b"draw x\n", b"draws 4\n", b"draw 4 ab cd\n"])
def test_malformed_line(junk, line):
    answers, rc = _hello(junk, line + b"null")
    assert rc == 1 and answers == [f"err a request `{line.decode().strip()}`"], answers


def test_a_draw_that_is_not_json(junk):
    answers, rc = _hello(junk, _request("{"))
    assert rc == 1 and len(answers) == 1 and answers[0].startswith("err the draw is not JSON"), answers


@pytest.mark.parametrize("nonce", ["ab" * 31, "AB" * 32, "zz" * 32])
def test_a_nonce_that_is_not_64_hex_digits(junk, nonce):
    answers, rc = _hello(junk, _request("null", nonce))
    assert rc == 1 and answers == [f"err a nonce `{nonce}` that is not 64 lowercase hex digits"], answers


def test_a_statement_the_setup_refuses(junk):
    """Refused, and the session goes on to its next request."""
    answers, rc = _hello(junk, _request("null") + b"draw\n")
    assert rc == 1 and answers[0].startswith("refused setup: ") and answers[1] == "err a request `draw`", answers


#: a stratified sample of one of two units, one template's
STRATIFIED = json.dumps({"law": "stratified", "population": 2, "k": 1, "strata": [{"template": "t", "units": [[0, 2]], "k": 1}],
                         "units": [0]})


@pytest.mark.parametrize("law, why", [([], "U2: a stratified draw needs the verifier's own K, program and partition"),
                                      (["--stratified", "2"], "U2: the draw's K is 1, not the verifier's 2")])
def test_a_stratified_draw_at_the_verifiers_own_k(junk, tmp_path, law, why):
    """A stratified sample sets up only at the verifier's own K, program and partition (`Stmt.setupTables`, U2), so
    `--stratified` reaches the setup: without it the sample is refused for want of it, and at another K for its K."""
    (tmp_path / "partition.json").write_text("{}")
    (tmp_path / "program").write_bytes(b"x")
    args = [*junk, "--partition", str(tmp_path / "partition.json"), "--program", str(tmp_path / "program"), *law]
    answers, rc = _hello(args, _request(STRATIFIED))
    assert rc == 0 and len(answers) == 1 and answers[0].startswith(f"refused setup: {why}"), answers


@pytest.mark.parametrize("law, why", [(["--stratified", "x"], "--stratified K: K is a natural number"),
                                      (["--work", "x", "--work-table", "table.json"], "--work K: K is a natural number"),
                                      (["--work", "2"], "--work needs --work-table"),
                                      (["--work-table", "table.json"], "--work-table needs --work"),
                                      (["--closure", "junk"], "not JSON")])
def test_a_draw_law_input_it_cannot_read(junk, tmp_path, law, why):
    (tmp_path / "table.json").write_text('{"t": 1}')
    (tmp_path / "junk").write_text("{")
    law = [str(tmp_path / x) if x in ("table.json", "junk") else x for x in law]
    p = subprocess.run([str(BIN), "hello", *junk, *law], input=b"", capture_output=True)
    assert p.returncode == 2 and p.stderr.decode().startswith("hello: ") and why in p.stderr.decode(), p.stderr
