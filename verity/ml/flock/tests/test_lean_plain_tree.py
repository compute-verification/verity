"""A plain session on the keyed coin tree (`--coins tree`, `Flock.CoinTree`): the compiled Lean verifier accepts
a plain (non-zk) session whose coins the coin server committed at `Hello` in a `coin-tree/hm96-sha512/v2` tree
(`flock-circuit serve` and `prove` under `FC_COINS=tree`, the prover checking each opening), and refuses it on any other
coins at S2/R7, and a second time in one run for its reused session key.

The session is `pod/91-plain-hidden.sh COINS=tree`'s: the RoPE d64 statement with hidden outputs of
`test_lean_plain_hidden.py`, 32 units under a subset sample of 16. It is a fetched fixture (`fixtures/artifacts.json`),
which `suites.py` fetches before the suite runs, so a missing fixture fails these tests rather than skipping them. Each
verify sets up an m = 22 table, so the module is slow. It catches a verifier that refuses an honest plain session on the
tree, or takes one on coins it doesn't name, which no `--zk` fixture can."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
#: ``stage/`` (circuit.txt, pub-32.bin, program.json, partition.json), ``session/`` (the drawn plain session's
#: record and proofs) and ``live.json`` (the prover's LIVE record)
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "plain-tree"
SPEC = {"block": 512, "rounds_log": 9, "scheme": "coin-tree/hm96-sha512/v2", "streams": ["session", "circuit/rep0", "circuit/rep1"]}

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


@pytest.fixture(scope="module")
def fx():
    assert (FIXTURE / "live.json").is_file(), (
        f"{FIXTURE} is not fetched: `uv run research data fetch-fixtures verity/ml/flock/tests/fixtures/plain-tree`")
    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)
    return FIXTURE


def _verify(fx: Path, coins: str, *sessions: Path) -> tuple[int, list[dict]]:
    st = fx / "stage"
    p = subprocess.run([str(BIN), "verify", "--statement", "verity/flock-circuit", "--coins", coins, "--circuit",
                        str(st / "circuit.txt"), "--public", str(st / "pub-32.bin"), "--partition", str(st / "partition.json"),
                        "--program", str(st / "program.json"), "--session", *map(str, sessions or [fx / "session"])],
                       capture_output=True, text=True, check=False)
    return p.returncode, [json.loads(x.split(" ", 1)[1]) for x in p.stdout.splitlines() if x.startswith("VERDICT ")]


def test_the_plain_session_on_the_tree_is_accepted(fx):
    rec = json.loads((fx / "session" / "session.json").read_text())
    assert {k: v for k, v in json.loads(rec["hello"])["coins"].items() if k != "nonce"} == SPEC
    assert rec["coin_tree"]["spec"] == SPEC and rec["coin_tree"]["nonce"] == json.loads(rec["hello"])["coins"]["nonce"]
    live = json.loads((fx / "live.json").read_text())
    assert (live["accepted"], live["step0_checked"]) == (True, True), live
    rc, [v] = _verify(fx, "tree")
    assert (rc, v["accepted"], v["why"]) == (0, True, ""), v
    assert v["program_sha512"] == json.loads((fx / "stage" / "partition.json").read_text())["program"]


@pytest.mark.parametrize("coins", ["os", "seed"])
def test_other_coins_refuse_it_at_hello(fx, coins):
    rc, [v] = _verify(fx, coins)
    assert (rc, v["accepted"], v["why"]) == (1, False, "S2/R7: the session parameters are not the verifier's"), v


def test_its_session_key_is_used_once(fx, tmp_path):
    again = shutil.copytree(fx / "session", tmp_path / "again")
    rc, [first, second] = _verify(fx, "tree", fx / "session", again)
    assert (rc, first["accepted"], second["accepted"]) == (1, True, False), (first, second)
    assert second["why"] == "S2: coin_tree.key is an earlier session's; a session key is never reused", second


def test_coins_it_does_not_know_are_a_usage_error(fx):
    rc, verdicts = _verify(fx, "urandom")
    assert (rc, verdicts) == (2, [])
