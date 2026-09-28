"""Sessions of several tables (PROTOCOL.md §16.11, #306's `--session-tables J`, N4): the Lean verifier builds each table's
statement from its part of the session's draw (without one, of the population's units), never from anything in the record,
and runs §5.3 over every table. The sessions are #306's live ones on RoPE, all accepted by its server."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[1] / "verifier" / "lean"
VERIFIER = LEAN.parent
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
#: RoPE d64 (8 instances) with #306's loopback sessions: J = 2 without a draw, J = 2 with `--draw subset:4` (two), J = 4
FIXTURE = "art:425f6860"
STATEMENT = "verity/flock-circuit@967b8d06+seed-injection"

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


@pytest.fixture(scope="module")
def rope(tmp_path_factory):
    sys.path.insert(0, str(VERIFIER))
    import ci  # noqa: PLC0415

    try:
        root = ci.fetch(FIXTURE, tmp_path_factory.mktemp("store"))
    except (subprocess.CalledProcessError, OSError) as e:
        pytest.skip(f"the evidence store is not reachable: {e}")
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    return root


def _session(root: Path, name: str, k: int = 0) -> Path:
    return sorted(p for p in (root / "sessions" / name).iterdir() if p.is_dir())[k]


def _verify(root: Path, tables: int, *sessions: Path, statement: str = STATEMENT) -> list[dict]:
    p = subprocess.run([str(BIN), "verify", "--statement", statement, "--circuit", str(root / "stage" / "circuit.txt"),
                        "--public", str(root / "stage" / "pub-8.bin"), "--coins", "seed", "--session-tables", str(tables),
                        "--session", *map(str, sessions)], capture_output=True, text=True)
    return [json.loads(line.split(" ", 1)[1]) for line in p.stdout.splitlines() if line.startswith("VERDICT ")]


def _copy(src: Path, dst: Path, edit=None) -> Path:
    shutil.copytree(src, dst)
    if edit:
        rec = json.loads((dst / "session.json").read_text())
        edit(rec)
        (dst / "session.json").write_text(json.dumps(rec))
    return dst


def test_two_tables_are_accepted(rope):
    [v] = _verify(rope, 2, _session(rope, "j2"))
    assert v["accepted"], v["why"]


def test_two_tables_of_a_drawn_session_are_accepted(rope):
    [v] = _verify(rope, 2, _session(rope, "j2-draw"))
    assert v["accepted"], v["why"]


@pytest.mark.slow
def test_every_live_session_is_accepted(rope):
    """The other drawn session and J = 4: about two minutes, so with `--slow`."""
    assert [v["accepted"] for v in _verify(rope, 2, _session(rope, "j2-draw", 1))] == [True]
    assert [v["accepted"] for v in _verify(rope, 4, _session(rope, "j4"))] == [True]


@pytest.mark.parametrize("tables", [2, 3])
def test_another_table_count_is_refused(rope, tables):
    """J is the verifier's: a J = 4 session read as J = 2 (or J = 3, a count that splits 8 units unevenly) is refused."""
    [v] = _verify(rope, tables, _session(rope, "j4"))
    assert not v["accepted"]
    assert v["why"].startswith("S2/R7" if tables == 2 else "setup: tables: 8 units do not split into 3 equal tables"), v["why"]


def test_a_changed_draw_is_refused(rope, tmp_path):
    """The parts come from the record's draw, and Σ covers each: another draw gives another session, refused at S2."""
    def edit(rec):
        rec["unit_draw"]["units"] = [0, 4, 5, 6]
    [v] = _verify(rope, 2, _copy(_session(rope, "j2-draw"), tmp_path / "s", edit))
    assert not v["accepted"] and v["why"].startswith("S2/R7"), v["why"]


def test_a_tables_part_is_not_the_sessions_draw(rope, tmp_path):
    """Table 0's part as the record's draw is one table's statement, not the session's."""
    def edit(rec):
        d = rec["unit_draw"]
        rec["unit_draw"] = {"law": "subset", "population": d["population"], "k": 2, "units": d["units"][:2]}
    [v] = _verify(rope, 2, _copy(_session(rope, "j2-draw"), tmp_path / "s", edit))
    assert not v["accepted"] and v["why"].startswith("S2/R7"), v["why"]


@pytest.mark.parametrize("draw, why", [
    ({"law": "bernoulli", "population": 8, "p": "1/1", "units": list(range(8))}, "setup: tables: several tables need a subset draw"),
    ({"law": "subset", "population": 8, "k": 3, "units": [0, 1, 2]}, "setup: tables: 3 units do not split into 2 equal tables"),
    ({"law": "subset", "population": 9, "k": 4, "units": [3, 4, 5, 6]}, "setup: U: the draw's population is not this registered population"),
])
def test_draws_the_tables_cannot_split_are_refused(rope, tmp_path, draw, why):
    def edit(rec):
        rec["unit_draw"] = draw
    [v] = _verify(rope, 2, _copy(_session(rope, "j2-draw"), tmp_path / "s", edit))
    assert not v["accepted"] and v["why"].startswith(why), v["why"]


def test_a_missing_tables_root_is_refused(rope, tmp_path):
    def edit(rec):
        del rec["link"]["root_b"]["circuit-01"]
    [v] = _verify(rope, 2, _copy(_session(rope, "j2"), tmp_path / "s", edit))
    assert not v["accepted"] and "circuit-01 missing" in v["why"], v["why"]


def test_y_is_two_words_per_table(rope, tmp_path):
    def edit(rec):
        rec["link"]["y"] = rec["link"]["y"][:2]
    [v] = _verify(rope, 2, _copy(_session(rope, "j2"), tmp_path / "s", edit))
    assert not v["accepted"] and v["why"].startswith("S10"), v["why"]


def test_proofs_are_their_tables(rope, tmp_path):
    """Table 1's proofs in table 0's files: S17 refuses the files; with the record's digests swapped too, table 0's streams
    refuse table 1's messages."""
    s = _copy(_session(rope, "j2"), tmp_path / "files")
    for r in (0, 1):
        a, b = s / f"circuit-00.rep{r}.bin", s / f"circuit-01.rep{r}.bin"
        x = a.read_bytes()
        a.write_bytes(b.read_bytes())
        b.write_bytes(x)
    [v] = _verify(rope, 2, s)
    assert not v["accepted"] and v["why"].startswith("S17"), v["why"]

    def edit(rec):
        for p in rec["proof_sha512"]:
            t, r = p["stream"].split("/")
            p["stream"] = f"{'circuit-01' if t == 'circuit-00' else 'circuit-00'}/{r}"
    rec = json.loads((s / "session.json").read_text())
    edit(rec)
    (s / "session.json").write_text(json.dumps(rec))
    [v] = _verify(rope, 2, s)
    assert not v["accepted"] and v["why"].startswith("R2"), v["why"]


def test_statements_without_several_tables_refuse_them(rope):
    [v] = _verify(rope, 2, _session(rope, "j2"), statement="verity/flock-circuit+seed-injection")
    assert not v["accepted"] and v["why"].startswith("setup: verity/flock-circuit+seed-injection: one table per session"), v["why"]
    [v] = _verify(rope, 65, _session(rope, "j2"))
    assert not v["accepted"] and v["why"].startswith("setup: tables: at most 64 per session"), v["why"]
