"""`verify --zk` (verifier PROTOCOL.md §16.14, `Flock.Zk`): the Lean verifier on `flock-circuit --zk` sessions, the masked
protocol (M1) on the keyed coin tree (M2). The sessions are the live build's on RoPE d64 (m = 25 and 26), each with the
verdict its server reached: a `selftest --zk --record-dir` run (`selftest_records.py`), and `serve`/`prove --zk` sessions of
two tables, of one m = 26 table and of a drawn one. Tampered copies come from `zk_mutants.py`; `agree.py --zk` compares all of
them with upstream's offline verdict (§18). Only the drawn session is accepted: the others have no draw, which `verify` refuses
after every check has passed (§16.15, D12), so a session the server accepted gets that refusal (`DRAWLESS`) and every other
session its own reason."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
VERIFIER = Path(__file__).resolve().parents[1] / "verifier"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
#: RoPE d64 staged by verity_flock.circuit with hidden outputs (circuit_sha512 5c33464c…) with the live build's --zk sessions:
#: sessions/selftest (one table, m = 25, each case's live verdict in case.json), sessions/j2 (two tables of 16 instances),
#: sessions/m26 (one table of 32) and sessions/drawn (a subset draw of 16 of 32), each with its server's index.jsonl (all
#: accepted); its identity's refused_coins name the zerocheck outer coin r_j = 1 (run r20261004-033801-14b6)
FIXTURE = "art:5542740a"
STATEMENT = "verity/flock-circuit@f5df3bbc+seed-injection"
#: the refusal of a session without a draw, which `verify` reaches only when every check has passed
DRAWLESS = "refused: a session without a draw: every soundness claim is about a drawn session (D12)"

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


@pytest.fixture(scope="module")
def zk(tmp_path_factory):
    sys.path.insert(0, str(VERIFIER))
    import ci

    try:
        root = ci.fetch(FIXTURE, tmp_path_factory.mktemp("store"))
    except (subprocess.CalledProcessError, OSError) as e:
        pytest.skip(f"the evidence store is not reachable: {e}")
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    return root


def _run(root: Path, *sessions: Path, public: str | Path = "pub-16.bin", tables: int = 1, zk: bool = True,
         extra: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    return subprocess.run([str(BIN), "verify", "--statement", STATEMENT, "--circuit", str(root / "stage" / "circuit.txt"),
                           "--public", str(root / "stage" / public), "--session-tables", str(tables), *(["--zk"] if zk else []),
                           *extra, "--session", *map(str, sessions)], capture_output=True, text=True, check=False)


def _verify(root: Path, *sessions: Path, **kw) -> list[dict]:
    return [json.loads(line.split(" ", 1)[1]) for line in _run(root, *sessions, **kw).stdout.splitlines()
            if line.startswith("VERDICT ")]


def _passes(v: dict) -> bool:
    """A session without a draw passed every check: it is refused for that alone."""
    assert not v["accepted"], v
    return v["why"] == DRAWLESS


def _one(root: Path, name: str, k: int = 0) -> Path:
    return sorted(p for p in (root / "sessions" / name).iterdir() if p.is_dir())[k]


def _selftest(root: Path, keep) -> None:
    """The selftest cases `keep` admits, in one run, a case with its own false statement (`pub.bin`) in a run of its own:
    each gets its live verdict, and an inner proof's refusal its reason."""
    found = sorted(p for p in (root / "sessions" / "selftest").iterdir() if p.is_dir() and keep(p.name))
    shared = [d for d in found if not (d / "pub.bin").exists()]
    own = [d for d in found if (d / "pub.bin").exists()]
    cases = shared + own
    verdicts = _verify(root, *shared) + [v for d in own for v in _verify(root, d, public=d / "pub.bin")]
    assert len(verdicts) == len(cases)
    for d, v in zip(cases, verdicts):
        live = json.loads((d / "case.json").read_text())["live"]
        assert _passes(v) == live["accepted"], (d.name, v["why"], live["why"])
        if live["why"].startswith("zk inner"):
            assert v["why"] == live["why"], d.name


@pytest.mark.slow
def test_the_zk_selftest_sessions_get_their_live_verdicts(zk):
    """The two honest sessions pass every check; level 0's hiding forged four ways, the pads not the committed ones (with and
    without `final_c'` re-solved) and an internal bit flipped, each refused by its check."""
    _selftest(zk, lambda n: n.startswith(("honest", "zk_")) or n == "unit_internal_bit_flipped")


@pytest.mark.slow
def test_final_c_solved_after_the_batching_coins_is_refused(zk):
    """`final_c'` is a constant of two rows, so a prover who fixed it after the inner proof's batching coins could cancel any
    failing row; it is absorbed before `ε` (live PROTOCOL.md §5), so the re-solved value is not the round's bytes."""
    d = zk / "sessions" / "selftest" / "zk_final_c_solved_after_the_batching_coins"
    live = json.loads((d / "case.json").read_text())["live"]
    [v] = _verify(zk, d)
    assert not live["accepted"] and live["why"].startswith("R2: "), live["why"]
    assert not v["accepted"] and v["why"].startswith("R2: round "), v["why"]


@pytest.mark.slow
@pytest.mark.skipif("not os.environ.get('FLOCK_VERIFIER_SLOW') and not config.getoption('--slow', False)",
                    reason="about two minutes: set FLOCK_VERIFIER_SLOW=1 (or pass --slow)")
def test_every_selftest_session_gets_its_live_verdict(zk):
    """The session-level forgeries too (aborted sessions, swapped or missing proofs, other statements): only when asked for."""
    _selftest(zk, lambda n: True)


@pytest.mark.slow
def test_a_session_key_is_used_once(zk):
    """The coin tree's key is fresh (§7.4): a session read twice in one run is refused the second time."""
    honest = zk / "sessions" / "selftest" / "honest"
    first, second = _verify(zk, honest, honest)
    assert _passes(first), first["why"]
    assert not second["accepted"] and second["why"].startswith("S2: coin_tree.key is an earlier session's"), second["why"]


@pytest.mark.slow
def test_zk_is_the_verifiers_setting(zk):
    """Without `--zk` the statement is the plain one: another identity, so another digest and another `Hello` (S2)."""
    [v] = _verify(zk, zk / "sessions" / "selftest" / "honest", zk=False)
    assert not v["accepted"] and v["why"].startswith("S2"), v["why"]


@pytest.mark.slow
def test_zk_takes_no_coins_flag(zk):
    """A `--zk` session's coins are its keyed coin tree, never `os` or `seed` coins (the live build reads no `FC_COINS` under
    `--zk`, and its identity never names live-OS coins), so `--zk` with either `--coins` is refused before any setup."""
    for coins in ("os", "seed"):
        p = _run(zk, zk / "sessions" / "selftest" / "honest", extra=("--coins", coins))
        assert p.returncode == 2 and "VERDICT" not in p.stdout, (coins, p.stdout)
        assert p.stderr.startswith("verify --zk: a --zk session's coins are its keyed coin tree"), (coins, p.stderr)


@pytest.mark.slow
def test_a_draw_is_accepted_and_two_tables_and_m26_pass_every_check(zk):
    """The drawn session is accepted; the sessions of two tables and of one m = 26 table, which have no draw, pass every check
    and are refused for that alone."""
    [j2] = _verify(zk, _one(zk, "j2"), public="pub-32.bin", tables=2)
    m26, drawn = _verify(zk, _one(zk, "m26"), _one(zk, "drawn"), public="pub-32.bin")
    assert _passes(j2) and _passes(m26), (j2["why"], m26["why"])
    assert drawn["accepted"], drawn["why"]


@pytest.mark.slow
def test_another_table_count_is_refused(zk, tmp_path):
    """J is the verifier's: a two-table session read as one table, and a one-table session read as two, each with its proofs
    in the files the other count names, meet other session parameters (S2/R7) before any proof."""
    as1 = shutil.copytree(_one(zk, "j2"), tmp_path / "as1")
    as2 = shutil.copytree(_one(zk, "m26"), tmp_path / "as2")
    for r in (0, 1):
        shutil.copy(as1 / f"circuit-00.rep{r}.bin", as1 / f"circuit.rep{r}.bin")
        for t in ("circuit-00", "circuit-01"):
            shutil.copy(as2 / f"circuit.rep{r}.bin", as2 / f"{t}.rep{r}.bin")
    for s, tables in ((as1, 1), (as2, 2)):
        [v] = _verify(zk, s, public="pub-32.bin", tables=tables)
        assert not v["accepted"] and v["why"].startswith("S2/R7"), v["why"]


#: one tampered copy per check that catches it (`zk_mutants.py`'s names)
TAMPERED = ["zk-zc-final-a", "zk-lig-lane-message", "zk-l0-ybar", "zk-l0-missing", "zk-lig-row-extra-lane", "zk-pads-root",
            "zk-tau", "zk-inner-opened", "zk-coin-tree-nonce", "zk-coin-tree-key", "zk-trailing-bytes"]


@pytest.mark.slow
def test_tampered_sessions_meet_their_checks(zk, tmp_path):
    """A tampered copy keeps its original's key, so each runs alone; each gets the verdict and the reason `zk_mutants.py`
    expects of the Lean verifier, a copy it expects accepted passing every check (the honest session has no draw)."""
    sys.path.insert(0, str(VERIFIER))
    import zk_mutants

    zk_mutants.main(zk / "sessions" / "selftest" / "honest", tmp_path)
    expect = {e["name"]: e["expect"] for e in json.loads((tmp_path / "zk_mutants.json").read_text())}
    with ThreadPoolExecutor(max_workers=4) as ex:
        verdicts = dict(zip(TAMPERED, ex.map(lambda n: _verify(zk, tmp_path / n)[0], TAMPERED)))
    for name, v in verdicts.items():
        e = expect[name]
        assert _passes(v) if e["accepted"] else not _passes(v) and v["why"].startswith(e["why"]), (name, v["why"], e)


@pytest.mark.slow
def test_an_initial_cap_other_than_the_commitments_is_refused(zk, tmp_path):
    """One bit of rep 0's `initial_cap` flipped in its proof file alone, with the rep's `proof_sha512` rebuilt and no retained
    round edited, is refused by `ZkLigerito.ligerito`'s cap check. Nothing else reads the field, so a `zk_mutants.py` proof
    edit can't stand in: it carries the new bytes into round 0, and both verifiers then refuse at R2."""
    sys.path.insert(0, str(VERIFIER))
    import redigest
    import zk_mutants

    s = shutil.copytree(zk / "sessions" / "selftest" / "honest", tmp_path / "icap")
    rec = json.loads((s / "session.json").read_text())
    table = rec["config"]["tables"][0]["name"]
    f = s / f"{table}.rep0.bin"
    b = bytearray(f.read_bytes())
    b[redigest.layout(bytes(b), zk=True)["lig.icap"][0]] ^= 1
    f.write_bytes(b)
    zk_mutants.set_proof_digest(rec, f"{table}/rep0", b)
    (s / "session.json").write_text(json.dumps(rec))
    [v] = _verify(zk, s)
    assert v["why"] == "ligerito: initial_cap differs from the commitment's cap", v["why"]
