"""round_coins: each drawing round's coins from a session of its own of the auditor's challenger, on a root that binds
the run's base, the step and the round. The fake challenger is ``local`` on live_challenger's stub of the Lean
verifier's coins (`fake_challenger`, as `test_a5_reserve` runs it), so nothing builds `sampled-proofs-verifier`. A
round's draw holds against its session's evidence (`Session.held`: `coins.coins_reason` on the round's root and its
commitment, each `a5_coins.negatives` refused, the bytes read the ones released); a draw held to other bytes, another
commitment or another root is refused; a round that never planned its draw opens nothing, and a session the challenger
refuses fails the round, with nothing drawn on other coins. Flags that can't name a challenger are refused by name."""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.sampled_proofs import service as S

HERE = Path(__file__).resolve().parents[1]
TREE = HERE.parents[1]
STUB = TREE / "benchmarks" / "live_challenger" / "tests" / "stub_serve.py"
sys.path.insert(0, str(HERE))
import round_coins as RC  # noqa: E402

N, K = 26, 8


def fake_challenger(tmp_path: Path, monkeypatch, **flags) -> SimpleNamespace:
    """``--challenger local`` on the stub of the Lean verifier's coins, its scratch roots under `tmp_path`, pinned on a
    main of this tree's own newest commit: the challenger flags a driver reads."""
    AC = RC.a5_coins()
    A = AC._module("a5_auditor", AC.TREE / "benchmarks" / "live_challenger" / "auditor.py")
    real = A.serve_argv
    monkeypatch.setattr(A, "serve_argv", lambda *a, **k: [sys.executable, str(STUB), *real(*a, **k)[2:]])
    monkeypatch.setenv("COINS_LOG", str(tmp_path / "coins.log"))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    main = tmp_path / "main.txt"
    main.write_text(AC.on_main([]) + "\n")
    return SimpleNamespace(**{"challenger": "local", "main": str(main), "gate_key": None, "peer": None,
                              "challenger_key": None, **flags})


def draw(s: RC.Session, commitment: bytes) -> tuple[bytes, dict | None]:
    """The round's `subset:K` of N on its session's coins once committed to `commitment`: (the bytes read, the
    segment its turn released)."""
    s.rank.plan(RC.subset_need(K, N))
    s.rank.commit(commitment)
    read = bytearray()
    rule = S.Law(f"subset:{K}")
    S.select_draw(rule, S.selection_law(rule, N), N, lambda m: read.extend(b := s.rank(m)) or b)
    return bytes(read), s.rank.drawn()


def test_each_rounds_root_binds_the_base_the_step_and_the_round():
    base = "ab" * 64
    roots = {RC.root(base, step, name) for step in ("scan", "n2") for name in ("honest", "tampered")}
    assert len(roots) == 4 and all(len(r) == 128 for r in roots)
    assert RC.root(base, "scan", "honest") == RC.root(base, "scan", "honest") != RC.root("cd" * 64, "scan", "honest")
    assert RC.digest({"a": 1, "b": [2]}) == RC.digest({"b": [2], "a": 1}) != RC.digest({"a": 1, "b": [3]})


@pytest.mark.parametrize("flags, why", [
    ({}, "nothing draws on the verifier's own coins"),
    ({"challenger": "remote"}, "not 'remote'"),
    ({"challenger": "gate:research@h", "peer": "p"}, "needs --peer and --challenger-key"),
    ({"challenger": "local", "main": None}, "needs --main"),
])
def test_flags_that_cant_name_a_challenger_are_refused_by_name(flags, why):
    a = SimpleNamespace(**{"challenger": None, "main": "m.txt", "peer": None, "challenger_key": None, **flags})
    assert why in RC.refusal(a)
    assert RC.refusal(SimpleNamespace(challenger="local", main="m.txt", peer=None, challenger_key=None)) is None


def test_a_rounds_draw_holds_on_its_own_session_and_other_bytes_a_commitment_or_a_root_do_not(tmp_path, monkeypatch):
    a = fake_challenger(tmp_path, monkeypatch)
    root, after = RC.root("ab" * 64, "scan", "honest"), hashlib.sha512(b"registration").digest()
    s = RC.Session(a, root, tmp_path / "honest" / "challenger")
    try:
        read, segment = draw(s, after)
    finally:
        s.close()
    assert (tmp_path / "coins.log").read_text(), "the session's coins came from the stub, not a Lean build"
    held = s.held(after, read, segment)
    assert held["ok"] and held["coins_reason"] is None and held["read_is_released"], held
    assert held["source"] == "challenger:local" and held["root"] == root and held["session"]
    assert held["negatives_refused"] and all(held["negatives_refused"].values()), held["negatives_refused"]
    assert held["read_bytes"] == len(read) > 0 and (tmp_path / "honest" / "challenger" / "coins.json").is_file()
    flipped = bytes([read[0] ^ 1]) + read[1:]
    assert not s.held(after, flipped, segment)["ok"]
    other = s.held(hashlib.sha512(b"another registration").digest(), read, segment)
    assert not other["ok"] and other["coins_reason"], other
    s.root = RC.root("ab" * 64, "scan", "chain")
    moved = s.held(after, read, segment)
    assert not moved["ok"] and moved["coins_reason"], moved


def test_a_round_that_never_planned_opens_nothing_and_a_refused_session_fails_closed(tmp_path, monkeypatch):
    a = fake_challenger(tmp_path, monkeypatch)
    s = RC.Session(a, RC.root("ab" * 64, "scan", "chain"), tmp_path / "chain" / "challenger")
    published = s.close()
    assert "was not opened" in published["error"] and published["plan"].get("session") is None
    held = s.held(b"", b"", None)
    assert not held["ok"] and held["session"] is None and held["error"] == published["error"]
    assert not (tmp_path / "coins.log").exists() and not (tmp_path / "chain" / "challenger" / "coins.json").exists()

    AC = RC.a5_coins()
    monkeypatch.setattr(AC, "reserve_reason", lambda layout, needs: "the challenger refuses the reserve")
    s = RC.Session(a, RC.root("ab" * 64, "scan", "tampered"), tmp_path / "tampered" / "challenger")
    try:
        with pytest.raises(AC.Ch.CoinError, match="the challenger refuses the reserve"):
            s.rank.plan(RC.subset_need(K, N))
    finally:
        s.close()
    held = s.held(b"", b"", None)
    assert not held["ok"] and "the challenger refuses the reserve" in held["error"]
    assert not (tmp_path / "coins.log").exists()
