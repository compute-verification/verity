"""A plain hidden-output session on the keyed coin tree, end to end (`Flock.ProvedScope`, `plain_hidden.py`, `--coins
tree`, `Flock.CoinTree`): a RoPE d64 statement with hidden outputs whose 32 units read their own rows (no shared rows, no
registered value), proved by Flock in one loopback session without `--zk` under a subset sample of 16
(`../pod/91-plain-hidden.sh`), its coins committed at `Hello` in a `coin-tree/hm96-sha512/v2` tree that the auditor's
challenger sampled and opened (`flock-circuit serve --tree-from`, the prover checking each opening).

The session's coins are the ones the challenger's release log shows it sampled and opened. The compiled Lean verifier
accepts it with the verifier's own program and partition, the form `flock_verify_sound`'s plain hidden-output branch
covers at the subset sampling design (`FailClosed.SoundExecH` at its subset sample design), and refuses it on seed coins at S2/R7
(`--coins` names no other source), a second time in one run for its reused session key, with a byte of a proof changed, or with its record's sample a
bernoulli sample of the same units (no claim samples by that sampling design). It is the test of a plain session at the subset sampling design on
the verifier; the `#guard`s of `Flock/Test/PlainHidden.lean` hold the rule itself.

It is a fetched fixture (`fixtures/artifacts.json`), which `suites.py` fetches before the suite runs, so a missing fixture
fails these tests rather than skipping them. Each verify sets up an m = 22 table, so the module is slow. It catches a
verifier that refuses an honest plain session on the tree, or takes one on coins it doesn't name, which no `--zk`
fixture can."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from verity.circuits.codec import canonical_json, encode_program

LEAN = Path(__file__).resolve().parents[2] / "verifier"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
#: ``stage/`` (circuit.txt, pub-32.bin, program.json, partition.json), ``session/`` (the sampled plain session's
#: record and proofs), ``live.json`` (the prover's LIVE record) and ``challenger/`` (its record and release log)
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "plain-tree"
SPEC = {"block": 512, "rounds_log": 9, "scheme": "coin-tree/hm96-sha512/v2", "streams": ["session", "circuit/rep0", "circuit/rep1"]}

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


@pytest.fixture(scope="module")
def fx():
    assert (FIXTURE / "live.json").is_file(), (
        f"{FIXTURE} is not fetched: `uv run research data fetch-fixtures verity/protocols/flock/prover/tests/fixtures/plain-tree`")
    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)
    return FIXTURE


def _verify(fx: Path, coins: str | None, *sessions: Path) -> tuple[int, list[dict]]:
    st = fx / "stage"
    p = subprocess.run([str(BIN), "verify", "--statement", "verity/flock-circuit", *(("--coins", coins) if coins else ()),
                        "--circuit", str(st / "circuit.txt"), "--public", str(st / "pub-32.bin"), "--partition", str(st / "partition.json"),
                        "--program", str(st / "program.json"), "--session", *map(str, sessions or [fx / "session"])],
                       capture_output=True, text=True, check=False)
    return p.returncode, [json.loads(x.split(" ", 1)[1]) for x in p.stdout.splitlines() if x.startswith("VERDICT ")]


def test_the_sessions_coins_are_the_challengers(fx):
    """Its coin tree is one the challenger's release log shows it sampled, with the rounds it opened; with one of its
    link points changed, it is refused at them."""
    from verity.protocols.spot.device import challenger as C, remote as R

    trees = C.coin_trees_of(R.log_lines(fx / "challenger" / R.RELEASES))
    rec = json.loads((fx / "session" / "session.json").read_text())
    tables = dict.fromkeys(rec["link"]["root_b"])
    assert len(trees) == 1 and C.sessions_tree_reason(trees, [("session", tables, rec)]) is None
    c = rec["link"]["points"][0][0]
    rec["link"]["points"][0][0] = ("0" if c[0] != "0" else "1") + c[1:]
    assert "link points are not the coins" in C.sessions_tree_reason(trees, [("session", tables, rec)])


def test_the_plain_session_on_the_tree_at_a_subset_draw_is_accepted(fx):
    import plain_hidden as PH

    program, _ = PH.program()
    assert canonical_json(encode_program(program)) == (fx / "stage" / "program.json").read_bytes()
    rec = json.loads((fx / "session" / "session.json").read_text())
    draw = rec["unit_draw"]
    assert (draw["law"], draw["population"], draw["k"], len(draw["units"])) == ("subset", 32, 16, 16)
    assert {k: v for k, v in json.loads(rec["hello"])["coins"].items() if k != "nonce"} == SPEC
    assert rec["coin_tree"]["spec"] == SPEC and rec["coin_tree"]["nonce"] == json.loads(rec["hello"])["coins"]["nonce"]
    live = json.loads((fx / "live.json").read_text())
    assert (live["accepted"], live["step0_checked"]) == (True, True), live
    rc, [v] = _verify(fx, "tree")
    assert (rc, v["accepted"], v["why"]) == (0, True, ""), v
    assert v["program_sha512"] == json.loads((fx / "stage" / "partition.json").read_text())["program"]


def test_seed_coins_refuse_it_at_hello(fx):
    rc, [v] = _verify(fx, None)
    assert (rc, v["accepted"], v["why"]) == (1, False, "S2/R7: the session parameters are not the verifier's"), v


def test_its_session_key_is_used_once(fx, tmp_path):
    again = shutil.copytree(fx / "session", tmp_path / "again")
    rc, [first, second] = _verify(fx, "tree", fx / "session", again)
    assert (rc, first["accepted"], second["accepted"]) == (1, True, False), (first, second)
    assert second["why"] == "S2: coin_tree.key is an earlier session's; a session key is never reused", second


def test_a_changed_proof_or_an_uncovered_law_is_refused(fx, tmp_path):
    flipped = shutil.copytree(fx / "session", tmp_path / "flipped")
    rep = sorted(flipped.glob("*.bin"))[0]
    b = bytearray(rep.read_bytes())
    b[len(b) // 2] ^= 1
    rep.write_bytes(bytes(b))
    _, [bad] = _verify(fx, "tree", flipped)
    assert bad["accepted"] is False and not bad["why"].startswith("setup:"), bad
    bern = shutil.copytree(fx / "session", tmp_path / "bernoulli")
    rec = json.loads((bern / "session.json").read_text())
    d = rec["unit_draw"]
    rec["unit_draw"] = {"law": "bernoulli", "population": d["population"], "p": "1/2", "units": d["units"]}
    (bern / "session.json").write_text(json.dumps(rec))
    _, [bad] = _verify(fx, "tree", bern)
    assert bad == {**bad, "accepted": False, "why": "refused: a bernoulli draw: no soundness claim draws by this law"}, bad


@pytest.mark.parametrize("coins", ["os", "seed", "urandom"])
def test_coins_other_than_the_tree_are_a_usage_error(fx, coins):
    rc, verdicts = _verify(fx, coins)
    assert (rc, verdicts) == (2, [])
