"""A live H1 window's salt from a local gate (`window_salt.issue`): the session `core_check.window_salt` accepts, and only
for the window's own weights and epoch."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

from verity.ml.pouw import core_check as CC
from verity.ml.pouw import window as Wd
from verity.core.protocols.sampled_proofs import challenger as Ch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import window_salt as WS  # noqa: E402

STUB = WS.TREE / "benchmarks" / "live_challenger" / "tests" / "stub_serve.py"
W = bytes(range(64))


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
    """The salt session opens on the epoch's root over W, releases its one coin, and `window_salt` accepts what `issue`
    returns; the same answer is refused for another W, another epoch, or a seeded window."""
    g, main = gate
    got = WS.issue(g, W, 3, main, tmp_path / "out")
    assert (tmp_path / "coins.log").read_text(), "the session's coins came from the stub, not a Lean build"
    assert got.salt.hex() == got.answer["salt"] and len(got.salt) == 64 and got.answer["epoch"] == 3
    assert CC.window_salt(params(got.live), W.hex(), got.answer) is None
    assert CC.window_salt(params(got.live), bytes(64).hex(), got.answer) is not None
    assert CC.window_salt(params(got.live), W.hex(), {**got.answer, "epoch": 4}) is not None
    assert CC.window_salt({"check": {}}, W.hex(), got.answer) == CC.SEEDED_SALT


def test_a_root_salts_once(gate, tmp_path):
    """The gate opens the epoch's root once, ever: a second session for the same W and epoch is refused before any coin."""
    g, main = gate
    WS.issue(g, W, 0, main, tmp_path / "a")
    with pytest.raises(Ch.CoinError):
        WS.issue(g, W, 0, main, tmp_path / "b")
