"""The refresh rule replayed over a memory map (`benchmarks/memory/refresh.py`): with the setting off the replay is
today's, byte for byte; probation's inequality has no slack; today's re-key keeps its payload's birth; and with a
constant data life the replay comes to f = max(0, 1 − P/D) within one window."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.memory import refresh as R  # noqa: E402
from verity.protocols.resource_accounting.memory.poub.verifier import memory_map as MM  # noqa: E402

N = 4


def d(i: int) -> bytes:
    return hashlib.sha512(i.to_bytes(8, "big")).digest()


def small() -> tuple[MM.Header, list[MM.Step]]:
    """Writes over a registered slot and beside the store, a free, a withdrawal and a write over a written block."""
    return MM.Header(bytes(64), N), [
        MM.Step(0, (), (MM.Written(0, N, d(0)), MM.Written(N, N + 1, d(1)))),
        MM.Step(2, (N,), ()),
        MM.Step(3, (), (MM.Written(N + 1, N + 3, d(3)),), (N + 2,)),
        MM.Step(5, (1,), (MM.Written(0, N + 4, d(4)),)),
    ]


def changes_digest(changes: list[MM.Change]) -> str:
    doc = [[c.step, c.first, [x.hex() for x in c.digests], c.released, c.withdrawn] for c in changes]
    return hashlib.sha256(json.dumps(doc).encode()).hexdigest()


def test_setting_off_is_todays_replay():
    head, steps = small()
    today = MM.Replay(head)
    expected = [today.apply(s) for s in steps]
    changes, lives = R.replay(head, steps)
    assert lives is None and changes == expected
    assert changes_digest(changes) == "685ff5bc4a887861d1d84a7f15a388f7f9d4710a63e9d54f896ffe0ee6fdab39"
    on, lives = R.replay(head, steps, R.Refresh(2, 30.0, 1.0))
    assert on == expected and lives is not None


def test_lives_of_the_small_map():
    head, steps = small()
    _, lives = R.replay(head, steps, R.Refresh(1, 30.0, 1.0))
    assert lives.end == 6.0
    assert sorted(lives.registered_ends) == [1.0, 6.0]
    assert sorted((g.start, g.end, g.censored) for g in lives.written) == [
        (1.0, 3.0, False), (1.0, 6.0, False), (4.0, 6.0, True), (6.0, 6.0, True)]


def one_block(life_steps: int, lifetime: int, born_step: int = 0) -> dict:
    head = MM.Header(bytes(64), 1)
    steps = [MM.Step(0, (), ())] * (born_step > 0) + [
        MM.Step(born_step, (), (MM.Written(1, 1, d(1)),)), MM.Step(born_step + life_steps, (1,), ())]
    setting = R.Refresh(lifetime, 30.0, 1.0)
    _, lives = R.replay(head, steps, setting)
    return R.credit(lives, setting)


def test_probation_has_no_slack():
    """Born at the end of step 29, t = 30 s, window 1: credited from window 1 + L, at 30 + 30·L s."""
    at = one_block(life_steps=60, lifetime=2, born_step=29)
    assert at["written_share_bytes"] == 0 and at["written_share_byte_seconds"] == 0
    past = one_block(life_steps=61, lifetime=2, born_step=29)
    assert past["written_share_bytes"] == 1 and math.isclose(past["written_share_byte_seconds"], 1 / 61)


def test_todays_rekey_keeps_its_payloads_birth():
    head = MM.Header(bytes(64), 1)
    steps = [MM.Step(0, (), (MM.Written(1, 1, d(7)),)), MM.Step(40, (1,), ()),
             MM.Step(41, (), (MM.Written(1, 2, d(7)),)), MM.Step(99, (1,), ())]
    setting = R.Refresh(2, 30.0, 1.0)
    _, lives = R.replay(head, steps, setting)
    assert [(g.born, g.start, g.last) for g in lives.written] == [(1.0, 1.0, False), (1.0, 42.0, True)]
    a = R.appraise(lives, setting, {})
    assert a["written"] == 1 and a["rekeyed"] == 1 and a["life_median_s"] == 99.0
    assert math.isclose(R.credit(lives, setting)["written_share_byte_seconds"], (100 - 60) / (40 + 58))


def test_constant_life_comes_to_the_formula():
    """Every written block lives D = 600 s; one is born a second, so births fill each window evenly."""
    life, count, lifetime, window = 600, 900, 4, 30.0
    steps = [MM.Step(s, (1 + s - life,) if s >= life else (), (MM.Written(1 + s, 1 + s, d(s)),) if s < count else ())
             for s in range(count + life)]
    setting = R.Refresh(lifetime, window, 1.0)
    _, lives = R.replay(MM.Header(bytes(64), 1), steps, setting)
    assert not any(g.censored for g in lives.written)
    f = R.credit(lives, setting)["written_share_byte_seconds"]
    period = lifetime * window
    assert 1 - period / life <= f <= 1 - (period - window) / life
    assert math.isclose(f, 1 - (period - (window - 1) / 2) / life)


def test_credit_starts_where_the_rota_says(monkeypatch):
    base = one_block(life_steps=100, lifetime=2)
    monkeypatch.setattr(R, "ROTA", lambda born_window, lifetime: born_window + lifetime + 1)
    later = one_block(life_steps=100, lifetime=2)
    assert math.isclose(base["written_share_byte_seconds"], (101 - 60) / 100)
    assert math.isclose(later["written_share_byte_seconds"], (101 - 90) / 100)


def test_the_command_prices_at_a_given_rate(tmp_path, capsys):
    path = tmp_path / "map.jsonl"
    path.write_text(MM.dumps(*small()))
    out = tmp_path / "out.json"
    assert R.main([str(path), "--step-s", "1", "--rate", "h200=deployed:1.249e10", "--json", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["rota"] == "benchmarks.memory.refresh.uniform_rota"
    rates = doc["maps"][str(path)]["1/map"]["rates"]
    assert list(rates) == ["h200"] and rates["h200"]["codec"] == R.codec.DEFAULT.name
    assert "rota: benchmarks.memory.refresh.uniform_rota" in capsys.readouterr().out


def test_a_share_affords_a_period():
    head = MM.Header(bytes(64), 1000)
    setting = R.Refresh(1, 30.0, 1.0)
    _, lives = R.replay(head, [MM.Step(0, (), ())], setting)
    rate = 1e6 * R.BLOCK / R.codec.DEFAULT.encode_squarings
    a = R.appraise(lives, setting, {"x": (R.codec.DEFAULT, 1e6)}, shares=(0.05,))
    work = 1000 * R.BLOCK
    assert math.isclose(a["rates"]["x"]["share_needed"], work / (30.0 * rate))
    assert a["rates"]["x"]["shares"]["0.05"]["lifetime_windows"] == math.ceil(work / (0.05 * rate * 30.0))
    card = R.appraise(lives, setting, {"x": (R.codec.DEFAULT, 1e6)}, shares=(0.05,), store_bytes=96 * 2**30)
    assert card["rates"]["x"]["shares"]["0.05"]["lifetime_windows"] == math.ceil(96 * 2**30 / (0.05 * rate * 30.0))
