"""`pod/challenger.sh`'s audits of served sessions against their challenger's release log, run as the pods run them, on the
plain-tree fixture's challenger and session (`test_lean_plain_tree.py`): `challenger_audit` through `challengers.end`, as the
sweeps' servers and eight pod jobs run it, and `challenger_cases`, 93-zk-tree-from.sh's. Both hold each record at the tables
its link's `root_b` keys, in Python (`challenger.sessions_tree_reason`) and in Lean (`sampled-proofs-difftest`'s
`challenger_session_trees`), so either side called with (name, record) alone fails here. It builds `sampled-proofs-difftest`,
which no other test of this suite runs."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import verity_difftest as D
from verity.protocols.spot.device import challenger as C

from verity_flock import challengers as CH

SP = Path(__file__).resolve().parents[3] / "spot" / "verifier" / "lean"
#: ``challenger/`` (its record and release log) and ``session/`` (the plain session it sampled and opened the tree of)
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "plain-tree"

pytestmark = [pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain"), pytest.mark.slow]


class _Exited:
    def wait(self, timeout=None):
        return 0


def _bumped(h: str) -> str:
    return h[:-1] + "0123456789abcdef"[("0123456789abcdef".index(h[-1]) + 1) % 16]


def _session(d: Path, rec: dict) -> Path:
    d.mkdir(parents=True)
    (d / "session.json").write_text(json.dumps(rec))
    return d


@pytest.fixture(scope="module")
def built():
    assert (FIXTURE / "challenger" / "releases.jsonl").is_file(), (
        f"{FIXTURE} is not fetched: `uv run research data fetch-fixtures verity/protocols/flock/prover/tests/fixtures/plain-tree`")
    D.build(SP, "sampled-proofs-difftest")


@pytest.fixture
def ch(built, tmp_path: Path) -> Path:
    """The fixture's challenger directory, copied, since the audit writes its audit.json there."""
    d = tmp_path / "challenger"
    shutil.copytree(FIXTURE / "challenger", d)
    (d / "audit.json").unlink(missing_ok=True)
    return d


@pytest.fixture
def rec() -> dict:
    return json.loads((FIXTURE / "session" / "session.json").read_text())


def test_challenger_audit_passes_the_served_session_and_refuses_it_with_a_link_point_changed(ch, rec, tmp_path):
    honest = _session(tmp_path / "sessions" / "honest", rec)
    assert CH.end(ch, _Exited(), [str(honest)]) == {"exit": 0, "sessions": 1, "audit": True, "ok": True}
    assert json.loads((ch / "audit.json").read_text()) == {"sessions": ["honest"], "python": None, "lean": None, "ok": True}
    rec["link"]["points"][0][0] = _bumped(rec["link"]["points"][0][0])
    bent = _session(tmp_path / "sessions" / "bent", rec)
    assert CH.end(ch, _Exited(), [str(bent)])["ok"] is False
    audit = json.loads((ch / "audit.json").read_text())
    assert audit["python"] == audit["lean"] and "link points are not the coins" in audit["python"], audit


def test_challenger_cases_passes_the_tree_session_and_refuses_the_rest_for_the_leans_reason(ch, rec, tmp_path):
    """93-zk-tree-from.sh's cases, its own-tree session one whose coin tree's root no release names."""
    tree = _session(tmp_path / "tree-from", rec)
    own = _session(tmp_path / "own-tree", {**rec, "coin_tree": {**rec["coin_tree"], "root": _bumped(rec["coin_tree"]["root"])}})
    out = tmp_path / "audit.json"
    p = subprocess.run(["bash", "-c", '. "$0" && challenger_cases "$@"', str(CH.SH), str(ch), str(tree), str(own), str(out)],
                       capture_output=True, text=True, env=CH._env(ch))
    assert p.returncode == 0, p.stderr[-2000:]
    audit = json.loads(out.read_text())
    assert (audit["trees"], audit["hellos_refused"]) == (1, 0) and all(n > 0 for n in audit["opened"].values()), audit
    own_why = f"session own-tree's coins are not the challenger's: {C.NO_TREE}"
    tree_why = "session tree-from's coins are not the challenger's: its "
    assert audit["cases"] == {name: {"python": why, "lean": why} for name, why in {
        "tree-from": None, "own-tree": own_why, "both": own_why,
        "coin-changed": tree_why + "circuit/rep0 round 0 is not coins the challenger opened under its tree after that round's message",
        "point-changed": tree_why + "link points are not the coins the challenger opened as its tree's session round 0 after its "
                                    "link's roots"}.items()}, audit["cases"]
