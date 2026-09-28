"""Sessions of several tables (PROTOCOL.md §16.12, #306's `--session-tables J`, N4): the Lean verifier builds each table's
statement from its part of the session's draw (without one, of the population's units), never from anything in the record.
The sessions are #306's live ones on RoPE, all accepted by its server, and none is `--zk`: several tables without `--zk` are a
form the soundness proof does not cover (§16.15), so the verifier refuses every session, for that once every check has
passed: a drawn one for its subset draw without `--zk` at setup first, and a draw it sets no statement up from keeps its
setup's reason."""
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


def _refused(tables: int) -> str:
    return f"refused: {tables} tables in a session without --zk: a statement form the soundness proof does not cover"


#: a drawn session's refusal: no soundness claim draws a subset without `--zk`
SUBSET = "refused: a subset draw without --zk: no soundness claim draws by this law"


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


def _draw(**kw):
    return lambda rec: rec["unit_draw"].update(kw)


#: the drawn session's draw (units 3..6 of 8) edited, and the refusal it meets
DRAWS = {
    "a table's part as the session's draw": (_draw(k=2, units=[3, 4]), SUBSET),
    "a Bernoulli draw": (lambda rec: rec.update(unit_draw={"law": "bernoulli", "population": 8, "p": "1/1", "units": list(range(8))}),
                         "setup: tables: several tables need a subset draw"),
    "an odd count": (_draw(k=3, units=[0, 1, 2]), "setup: tables: 3 units do not split into 2 equal tables"),
    "another population": (_draw(population=9), "setup: U: the draw's population is not this registered population"),
}


@pytest.fixture(scope="module")
def verdicts(rope, tmp_path_factory):
    """The drawn J = 2 session and its refusals, in one verifier run (each statement set up once): verdicts by name."""
    w = tmp_path_factory.mktemp("sessions")
    s = _session(rope, "j2-draw")
    dirs = {"honest": s,
            "a missing root": _copy(s, w / "root", lambda rec: rec["link"]["root_b"].pop("circuit-01")),
            "a short y": _copy(s, w / "y", lambda rec: rec["link"].update(y=rec["link"]["y"][:2]))}
    files = _copy(s, w / "files")
    for r in (0, 1):
        a, b = files / f"circuit-00.rep{r}.bin", files / f"circuit-01.rep{r}.bin"
        x = a.read_bytes()
        a.write_bytes(b.read_bytes())
        b.write_bytes(x)
    dirs["swapped files"] = files
    dirs["swapped files and digests"] = _copy(files, w / "both", _swap_tables)
    dirs.update({name: _copy(s, w / f"draw{i}", edit) for i, (name, (edit, _)) in enumerate(DRAWS.items())})
    return dict(zip(dirs, _verify(rope, 2, *dirs.values())))


@pytest.mark.slow
@pytest.mark.parametrize("name", ["honest", "a missing root", "a short y", "swapped files", "swapped files and digests"])
def test_a_drawn_session_of_two_tables_is_refused(verdicts, name):
    """Honest or tampered, the drawn session's statements set up and it meets the refusal of its draw's law."""
    v = verdicts[name]
    assert not v["accepted"] and v["why"] == SUBSET, v["why"]


@pytest.mark.slow
@pytest.mark.skipif("not os.environ.get('FLOCK_VERIFIER_SLOW') and not config.getoption('--slow', False)",
                    reason="about three minutes: set FLOCK_VERIFIER_SLOW=1 (or pass --slow)")
def test_every_live_session_is_refused(rope):
    """J = 2 without a draw, the other drawn session, and J = 4: only when asked for."""
    assert [v["why"] for v in _verify(rope, 2, _session(rope, "j2"), _session(rope, "j2-draw", 1))] == [_refused(2), SUBSET]
    assert [v["why"] for v in _verify(rope, 4, _session(rope, "j4"))] == [_refused(4)]


@pytest.mark.slow
def test_another_table_count_is_refused(rope, tmp_path):
    """J is the verifier's: a J = 2 session read as one table names other tables and another Σ (S2), before any proof (a
    session without a draw: a drawn one meets its law's refusal first)."""
    s = _copy(_session(rope, "j2"), tmp_path / "s")
    for r in (0, 1):
        shutil.copy(s / f"circuit-00.rep{r}.bin", s / f"circuit.rep{r}.bin")
    [v] = _verify(rope, 1, s)
    assert not v["accepted"] and v["why"].startswith("S2/R7"), v["why"]


@pytest.mark.slow
@pytest.mark.parametrize("name", list(DRAWS))
def test_the_parts_come_from_the_sessions_draw(verdicts, name):
    """A table's part as the session's draw sets up and meets its law's refusal; a draw the tables cannot split is refused
    at setup."""
    v = verdicts[name]
    assert not v["accepted"] and v["why"].startswith(DRAWS[name][1]), v["why"]


def test_statements_without_several_tables_refuse_them(rope):
    [v] = _verify(rope, 2, _session(rope, "j2"), statement="verity/flock-circuit@e51e2b86+seed-injection")
    assert not v["accepted"] and v["why"].startswith("setup: verity/flock-circuit@e51e2b86+seed-injection: one table per session"), v["why"]
    [v] = _verify(rope, 65, _session(rope, "j2"))
    assert not v["accepted"] and v["why"].startswith("setup: tables: at most 64 per session"), v["why"]
