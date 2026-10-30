"""The challenger's entropy, `pouw.verifier.coins`: bytes `pouw-check-difftest` reads from the operating system
(`IO.getRandomBytes`, A3), which a served `coins.Challenger` and `window.live_challenger` draw every coin and hm96 salt
from, in order. A count the Lean can't answer is refused before it, and a Lean that refuses, answers other than the bytes
asked or can't run gives no coins, so no commitment."""
from __future__ import annotations

import shutil

import pytest

from verity.core.protocols.pouw import coins as Co
from verity.core.protocols.pouw import verifier as V
from verity.core.protocols.pouw import window as Wd

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


def _spy(monkeypatch) -> list[bytes]:
    answered, real = [], V._server.call

    def spy(fn, value):
        kind, body = real(fn, value)
        if fn == "coins":
            answered.append(bytes.fromhex(body))
        return kind, body
    monkeypatch.setattr(V._server, "call", spy)
    return answered


def test_a_served_challengers_coins_are_the_leans(monkeypatch):
    """Every byte a challenger on `verifier.coins` commits to is a `coins` answer of the Lean, in order: a challenger
    replaying exactly those answers commits to the same root, for an epoch's seeded coins and a live epoch's."""
    a, b = V.coins(64), V.coins(64)
    assert len(a) == len(b) == 64 and a != b and V.coins(0) == b""
    answered = _spy(monkeypatch)
    ch = Co.Challenger(3, V.coins)
    assert [len(x) for x in answered] == [Co.COIN_BYTES, Co.HM96.salt_bytes] * len(Co.LABELS)
    replay = iter(list(answered))
    assert Co.Challenger(3, lambda n: next(replay)).commitment == ch.commitment
    answered.clear()
    live = Wd.live_challenger(3, 2, V.coins)
    assert len(answered) == 2 * (1 + 2)
    replay = iter(list(answered))
    assert Wd.live_challenger(3, 2, lambda n: next(replay)).commitment == live.commitment


def test_coins_the_lean_cannot_give_are_no_coins(monkeypatch):
    for n in (-1, V.COINS_MAX + 1, 2.0, True):
        with pytest.raises(ValueError, match="coins"):
            V.coins(n)
    assert V._server.call("coins", "64") == ("reject", "coins: the input is a number of bytes")
    assert V._server.call("coins", V.COINS_MAX + 1) == ("reject", f"coins: {V.COINS_MAX + 1} bytes in one request, "
                                                                  f"past {V.COINS_MAX}")
    with pytest.raises(V.LeanError, match="unknown fn"):
        V._server.call("no such fn", 0)
    for answer in (("reject", "no"), ("output", "zz"), ("output", "00" * 7)):
        monkeypatch.setattr(V._server, "call", lambda fn, value, answer=answer: answer)
        with pytest.raises(V.LeanError):
            V.coins(8)

    def broken(fn, value):
        raise V.LeanError("lake is not on the PATH")
    monkeypatch.setattr(V._server, "call", broken)
    with pytest.raises(V.LeanError, match="lake"):
        Co.Challenger(0, V.coins)
