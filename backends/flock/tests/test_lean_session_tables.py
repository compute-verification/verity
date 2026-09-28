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
    import ci

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
                        "--session", *map(str, sessions)], capture_output=True, text=True, check=False)
    return [json.loads(line.split(" ", 1)[1]) for line in p.stdout.splitlines() if line.startswith("VERDICT ")]


def _copy(src: Path, dst: Path, edit=None) -> Path:
    shutil.copytree(src, dst)
    if edit:
        rec = json.loads((dst / "session.json").read_text())
        edit(rec)
        (dst / "session.json").write_text(json.dumps(rec))
    return dst


def _swap_tables(rec):
    for p in rec["proof_sha512"]:
        t, r = p["stream"].split("/")
        p["stream"] = f"{'circuit-01' if t == 'circuit-00' else 'circuit-00'}/{r}"


@pytest.fixture(scope="module")
def refused(rope, tmp_path_factory):
    """The refusals of sessions without a draw, in one verifier run (one setup): their verdicts by name."""
    w = tmp_path_factory.mktemp("refused")
    j2 = _session(rope, "j2")
    dirs = {"another J": _session(rope, "j4"),
            "a missing root": _copy(j2, w / "root", lambda rec: rec["link"]["root_b"].pop("circuit-01")),
            "a short y": _copy(j2, w / "y", lambda rec: rec["link"].update(y=rec["link"]["y"][:2]))}
    files = _copy(j2, w / "files")
    for r in (0, 1):
        a, b = files / f"circuit-00.rep{r}.bin", files / f"circuit-01.rep{r}.bin"
        x = a.read_bytes()
        a.write_bytes(b.read_bytes())
        b.write_bytes(x)
    dirs["swapped files"] = files
    dirs["swapped files and digests"] = _copy(files, w / "both", _swap_tables)
    return dict(zip(dirs, _verify(rope, 2, *dirs.values())))


def _draw(**kw):
    return lambda rec: rec["unit_draw"].update(kw)


#: the drawn session's draw (units 3..6 of 8) edited, and the refusal it meets
DRAWS = {
    "a changed draw": (_draw(units=[0, 4, 5, 6]), "S2/R7"),
    "a table's part as the session's draw": (_draw(k=2, units=[3, 4]), "S2/R7"),
    "a Bernoulli draw": (lambda rec: rec.update(unit_draw={"law": "bernoulli", "population": 8, "p": "1/1", "units": list(range(8))}),
                         "setup: tables: several tables need a subset draw"),
    "an odd count": (_draw(k=3, units=[0, 1, 2]), "setup: tables: 3 units do not split into 2 equal tables"),
    "another population": (_draw(population=9), "setup: U: the draw's population is not this registered population"),
}


@pytest.fixture(scope="module")
def refused_draws(rope, tmp_path_factory):
    w = tmp_path_factory.mktemp("draws")
    dirs = [_copy(_session(rope, "j2-draw"), w / str(i), edit) for i, (edit, _) in enumerate(DRAWS.values())]
    return dict(zip(DRAWS, _verify(rope, 2, *dirs)))


def test_two_tables_of_a_drawn_session_are_accepted(rope):
    [v] = _verify(rope, 2, _session(rope, "j2-draw"))
    assert v["accepted"], v["why"]


@pytest.mark.slow
@pytest.mark.skipif("not os.environ.get('FLOCK_VERIFIER_SLOW') and not config.getoption('--slow', False)",
                    reason="about three minutes: set FLOCK_VERIFIER_SLOW=1 (or pass --slow)")
def test_every_live_session_is_accepted(rope):
    """J = 2 without a draw, the other drawn session, and J = 4: only when asked for."""
    assert [v["accepted"] for v in _verify(rope, 2, _session(rope, "j2"), _session(rope, "j2-draw", 1))] == [True, True]
    assert [v["accepted"] for v in _verify(rope, 4, _session(rope, "j4"))] == [True]


def test_another_table_count_is_refused(refused):
    """J is the verifier's: a J = 4 session read as J = 2 names other tables and another Σ."""
    v = refused["another J"]
    assert not v["accepted"] and v["why"].startswith("S2/R7"), v["why"]


def test_a_missing_tables_root_is_refused(refused):
    v = refused["a missing root"]
    assert not v["accepted"] and "circuit-01 missing" in v["why"], v["why"]


def test_y_is_two_words_per_table(refused):
    v = refused["a short y"]
    assert not v["accepted"] and v["why"].startswith("S10"), v["why"]


def test_proofs_are_their_tables(refused):
    """Table 1's proofs in table 0's files: S17 refuses the files; with the record's digests swapped too, table 0's streams
    refuse table 1's messages."""
    v = refused["swapped files"]
    assert not v["accepted"] and v["why"].startswith("S17"), v["why"]
    v = refused["swapped files and digests"]
    assert not v["accepted"] and v["why"].startswith("R2"), v["why"]


@pytest.mark.parametrize("name", list(DRAWS))
def test_the_parts_come_from_the_sessions_draw(refused_draws, name):
    """Σ covers each table's part, so another draw is another session (S2); a draw the tables cannot split is refused."""
    v = refused_draws[name]
    assert not v["accepted"] and v["why"].startswith(DRAWS[name][1]), v["why"]


def test_statements_without_several_tables_refuse_them(rope):
    [v] = _verify(rope, 2, _session(rope, "j2"), statement="verity/flock-circuit+seed-injection")
    assert not v["accepted"] and v["why"].startswith("setup: verity/flock-circuit+seed-injection: one table per session"), v["why"]
    [v] = _verify(rope, 65, _session(rope, "j2"))
    assert not v["accepted"] and v["why"].startswith("setup: tables: at most 64 per session"), v["why"]
