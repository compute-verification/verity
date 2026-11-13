"""Plain sessions of several tables at hidden outputs (`Flock.ProvedScope`, `--plain-tables` with `SoundExecHJ`):
`plain_hidden.py`'s statement (RoPE d64 instances with hidden outputs, hm96-sha512 leaves), proved without `--zk` on the
keyed coin tree, sampled and opened by the auditor's challenger, in loopback sessions of two tables under a subset sample
(`verity/protocols/flock/pod/91-plain-hidden-tables.sh`). The sessions' coins are the ones the challenger's release log shows it
sampled and opened; under `--plain-tables` with the verifier's own program and partition, both sessions are accepted and a
tampered one keeps its check's reason; verified against another program it is refused; without the flag, several tables
without `--zk` are refused, once every check has passed."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[2] / "verifier"
VERIFIER = Path(__file__).resolve().parents[2] / "agreement"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
#: plain_hidden.py's 32 instances, two loopback sessions of J = 2 tables at `--draw subset:16`, and their challenger's
#: record and release log
FIXTURE = "art:8e0fd487e4f5853cb77664d09397f9aea25cfc937a4a293bece2b73d6bced855"
N = 32

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    sys.path.insert(0, str(VERIFIER))
    import ci

    try:
        r = ci.fetch(FIXTURE, tmp_path_factory.mktemp("store"))
    except (subprocess.CalledProcessError, OSError) as e:
        pytest.skip(f"the evidence store is not reachable: {e}")
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    return r


def _sessions(root: Path) -> list[Path]:
    return sorted(p.parent for p in (root / "sessions").glob("*/session.json"))


def _meta(root: Path) -> dict:
    return json.loads((root / "stage" / "circuit.txt").read_text().split("\n", 2)[1][len("META "):])


def _verify(root: Path, *sessions: Path, program: Path | None = None, flags: tuple = ("--plain-tables",)) -> list[dict]:
    st = root / "stage"
    p = subprocess.run([str(BIN), "verify", "--statement", "verity/flock-circuit", "--coins", "tree", "--circuit",
                        str(st / "circuit.txt"), "--public", str(st / f"pub-{N}.bin"), "--session-tables", "2",
                        "--partition", str(st / "partition.json"), "--program", str(program or st / "program.json"),
                        *flags, "--session", *map(str, sessions)],
                       capture_output=True, text=True, check=False)
    return [json.loads(line.split(" ", 1)[1]) for line in p.stdout.splitlines() if line.startswith("VERDICT ")]


@pytest.fixture(scope="module")
def plain(root, tmp_path_factory):
    """`--plain-tables` verdicts by name, in one verifier run: both honest sessions and the first with table 1's root gone."""
    s, t = _sessions(root)[:2]
    gone = tmp_path_factory.mktemp("plain") / "root"
    shutil.copytree(s, gone)
    rec = json.loads((gone / "session.json").read_text())
    rec["link"]["root_b"].pop("circuit-01")
    (gone / "session.json").write_text(json.dumps(rec))
    dirs = {"honest": s, "honest, the other session": t, "a missing root": gone}
    return dict(zip(dirs, _verify(root, *dirs.values())))


def test_the_fixture_is_a_hidden_output_statement(root):
    assert _meta(root)["leaf_scheme"]["id"] == "hm96-sha512/v1"
    assert len(_sessions(root)) == 2


def test_the_sessions_coins_are_the_challengers(root):
    """Each session's coin tree is one the challenger's release log shows it sampled, with the rounds it opened; with the
    two sessions' trees swapped, every tree the log shows is still a session's, and the first is refused at its coins."""
    from verity.protocols.spot.device import challenger as C, remote as R

    trees = C.coin_trees_of(R.log_lines(root / "challenger" / R.RELEASES))
    records = [(s.name, json.loads((s / "session.json").read_text())) for s in _sessions(root)]
    (a, ra), (b, rb) = records
    ta, tb = dict.fromkeys(ra["link"]["root_b"]), dict.fromkeys(rb["link"]["root_b"])
    assert len(trees) == 2 and C.sessions_tree_reason(trees, [(a, ta, ra), (b, tb, rb)]) is None
    swapped = [(a, ta, {**ra, "coin_tree": rb["coin_tree"]}), (b, tb, {**rb, "coin_tree": ra["coin_tree"]})]
    assert "link points are not the coins" in C.sessions_tree_reason(trees, swapped)


@pytest.mark.slow
def test_plain_tables_accepts_honest_hidden_output_sessions(root, plain):
    """Both sessions pass every check and the form `--plain-tables` covers at hidden outputs, against the verifier's own
    program."""
    for name in ("honest", "honest, the other session"):
        v = plain[name]
        assert v["accepted"] is True, (name, v["why"])
        assert v["program_sha512"] == _meta(root)["program_sha512"], v


@pytest.mark.slow
def test_plain_tables_refuses_a_tampered_session_at_its_check(plain):
    v = plain["a missing root"]
    assert not v["accepted"] and not v["why"].startswith("refused: "), v["why"]


@pytest.mark.slow
def test_plain_tables_holds_the_program_to_the_circuit(root, tmp_path):
    """Verified against another program (a RoPE d64 instance's alone), whose digest is not META's `program_sha512`, an honest
    session is refused (at setup, where the partition's program is checked, or by `ProvedScope.programP`)."""
    from verity_flock import subcircuit_templates as TM

    from verity.circuits.codec import canonical_json, encode_program
    from verity.circuits.program import Program

    other = tmp_path / "other.json"
    other.write_bytes(canonical_json(encode_program(Program(TM.rope_head(64).definition()))))
    [v] = _verify(root, _sessions(root)[0], program=other)
    assert not v["accepted"] and "program_sha512" not in v and "not held to the circuit" in v["program_note"], v


@pytest.mark.slow
def test_without_plain_tables_two_tables_are_refused(root):
    [v] = _verify(root, _sessions(root)[0], flags=())
    assert v["why"] == "refused: 2 tables in a session without --zk: a statement form the soundness proof does not cover", v
