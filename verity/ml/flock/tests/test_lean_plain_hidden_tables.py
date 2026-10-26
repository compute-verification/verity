"""Plain sessions of several tables at hidden outputs (PROTOCOL.md §16.15, `--plain-tables` with `SoundExecHJ`):
`plain_hidden.py`'s statement (RoPE d64 instances with hidden outputs, hm96-sha512 leaves), proved without `--zk` on OS
coins in loopback sessions of two tables under a subset draw (`verity/ml/flock/pod/91-plain-hidden-tables.sh`). Under
`--plain-tables` with the verifier's own program and partition, both sessions are accepted and a tampered one keeps its
check's reason; verified against another program it is refused; without the flag, several tables without `--zk` are
refused, once every check has passed."""
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
#: plain_hidden.py's 32 instances, two loopback sessions of J = 2 tables at `--draw subset:16`
FIXTURE = "art:b0aa23bbf9fcb1900d926d0b9f9fd18b7e0b075d042cf923a1c02b1e50b1070c"
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
    p = subprocess.run([str(BIN), "verify", "--statement", "verity/flock-circuit", "--coins", "os", "--circuit",
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
    pytest.importorskip("verity_numerical")
    from verity_numerical.bench import templates as TM

    from verity.ml.circuits.codec import canonical_json, encode_program
    from verity.ml.circuits.program import Program

    other = tmp_path / "other.json"
    other.write_bytes(canonical_json(encode_program(Program(TM.rope_head(64).definition()))))
    [v] = _verify(root, _sessions(root)[0], program=other)
    assert not v["accepted"] and v["program_sha512"] != _meta(root)["program_sha512"], v


@pytest.mark.slow
def test_without_plain_tables_two_tables_are_refused(root):
    [v] = _verify(root, _sessions(root)[0], flags=())
    assert v["why"] == "refused: 2 tables in a session without --zk: a statement form the soundness proof does not cover", v
