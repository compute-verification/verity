"""A live H1 window's salt from a local gate (`window_salt.issue`): the session `core_check.window_salt` accepts, and only
for the window's own weights, activations and epoch."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.sampled_proofs import challenger as Ch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import window_salt as WS  # noqa: E402

STUB = WS.TREE / "benchmarks" / "live_challenger" / "tests" / "stub_serve.py"


def group(program: str, seed: int, *names: str) -> dict:
    """A group of committed values' entries, as the torch loop's registration names them (`loop.activations`)."""
    return {"program": program, "values": {n: {"value": n, "root": bytes([seed + i] * 64).hex(), "leaves": 8, "words": 2,
                                               "domain": bytes([seed + i + 100] * 64).hex()}
                                           for i, n in enumerate(names)}}


WEIGHTS = group("torch-loop/mlp-w8a8-h128/seed0", 1, "sw", "w")
ACTS = group("torch-loop/mlp-w8a8-h128/seed0", 7, "sx", "x")
REC = WS.anchors(WEIGHTS, ACTS)


@pytest.fixture
def gate(tmp_path: Path, monkeypatch):
    """A local gate under `tmp_path`, pinned on this tree's commit, its serve.py under live_challenger's stub of the Lean
    verifier's coins, so the test builds no `sampled-proofs-verifier`."""
    AC = WS.a5_coins()
    A = AC._module("a5_auditor", WS.TREE / "benchmarks" / "live_challenger" / "auditor.py")
    real = A.serve_argv
    monkeypatch.setattr(A, "serve_argv", lambda *a, **k: [sys.executable, str(STUB), *real(*a, **k)[2:]])
    monkeypatch.setenv("COINS_LOG", str(tmp_path / "coins.log"))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    main = [AC.on_main([])]
    return WS.local(tmp_path / "host", main), main


def params(live: dict) -> dict:
    return {"check": {Wd.COINS: "live"}, "live": live}


def test_the_window_salt_is_the_sessions_and_only_its_windows(gate, tmp_path):
    """The salt session opens on the epoch's root over W and X, releases its one coin, and `window_salt` accepts what
    `issue` returns; the same answer is refused for other weights, another epoch, or a seeded window."""
    g, main = gate
    got = WS.issue(g, REC, 3, main, tmp_path / "out")
    assert (tmp_path / "coins.log").read_text(), "the session's coins came from the stub, not a Lean build"
    assert got.salt.hex() == got.answer["salt"] and len(got.salt) == 64 and got.answer["epoch"] == 3
    assert CC.window_salt(params(got.live), REC, got.answer) is None
    assert CC.window_salt(params(got.live), WS.anchors(group(WEIGHTS["program"], 3, "sw", "w"), ACTS), got.answer) \
        is not None
    assert CC.window_salt(params(got.live), REC, {**got.answer, "epoch": 4}) is not None
    assert CC.window_salt({"check": {}}, REC, got.answer) == CC.SEEDED_SALT


def test_activations_committed_after_the_salt_are_refused(gate, tmp_path):
    """A prover that takes the salt first and commits its activations after it (x chosen knowing the salt) registers
    activations whose root X is not the one the salt session was opened on: the ledger opened no session on the
    registration's root, so `window_salt` refuses it, whether the session was opened on other activations or on none."""
    g, main = gate
    early = WS.issue(g, WS.anchors(WEIGHTS, group(ACTS["program"], 40, "sx", "x")), 0, main, tmp_path / "early")
    why = CC.window_salt(params(early.live), REC, early.answer)
    assert why is not None and why.startswith("the challenger's session is not trusted"), why
    with pytest.raises(Ch.CoinError, match="commits no activations"):
        WS.issue(g, {"roots": [], "window": {"registered": WEIGHTS}}, 1, main, tmp_path / "none")
    assert CC.window_salt(params(early.live), {"roots": [], "window": {"registered": WEIGHTS}}, early.answer) \
        == CC.WINDOW_NO_X


def test_a_window_names_one_weights_root():
    """W is the window's ``W`` root or its registered weights' root, never both, and each group is ``{program, values}``
    with entries naming their value, domain and root."""
    w_root = {"port": "W", "root": "ab" * 64, "leaves": 1, "domain": "cd" * 64}
    assert CC.window_anchors({"roots": [w_root], "window": {"activations": ACTS}})[0] == bytes.fromhex("ab" * 64)
    assert CC.window_anchors({"roots": [w_root], "window": {"activations": ACTS, "registered": WEIGHTS}}) == CC.WINDOW_NO_W
    assert CC.window_anchors({"roots": [], "window": {"activations": ACTS}}) == CC.WINDOW_NO_W
    w, x = CC.window_anchors(REC)
    assert (w, x) == (CC.values_root(WEIGHTS), CC.values_root(ACTS)) and w != x
    renamed = {**ACTS, "values": {"x": {**ACTS["values"]["x"], "value": "sx"}}}
    for bad in (None, {}, {"program": "p", "values": {}}, {**ACTS, "extra": 1}, renamed,
                {**ACTS, "values": {"x": {**ACTS["values"]["x"], "root": "AB" * 64}}}):
        assert CC.values_root(bad) is None, bad
    moved = {**ACTS, "values": {"x": {**ACTS["values"]["x"], "domain": ACTS["values"]["sx"]["domain"]},
                                "sx": ACTS["values"]["sx"]}}
    assert len({CC.values_root(ACTS), CC.values_root(moved), CC.values_root({**ACTS, "program": "q"})}) == 3


def test_a_root_salts_once(gate, tmp_path):
    """The gate opens the epoch's root once, ever: a second session for the same W, X and epoch is refused before any
    coin."""
    g, main = gate
    WS.issue(g, REC, 0, main, tmp_path / "a")
    with pytest.raises(Ch.CoinError):
        WS.issue(g, REC, 0, main, tmp_path / "b")
