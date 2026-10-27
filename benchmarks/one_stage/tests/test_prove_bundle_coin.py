"""`prove_bundle.held_coin`: a delivered step's drawn indices are re-derived from the challenger's live coin over the step
registration's digest, not taken from ``draw.json``. The coin here is an in-process `LiveChallenger`'s under m3coin's
scheme, label and domain; its session has no ledger, so ``coin:session`` is refused in every case."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prove_bundle as PB  # noqa: E402

sys.path.insert(0, str(PB.REPO / "benchmarks" / "e2e_h200"))
import m3coin as COIN  # noqa: E402
from verity.core.protocols.sampled_proofs import challenger as Ch  # noqa: E402

REG = {"format": PB.RB.STEP_FORMAT, "law": {"law": "subset", "k": 8}, "population": 3_100_000_000, "run": "m3-test"}


def _entropy(n: int, _state=[0]) -> bytes:
    _state[0] += 1
    return hashlib.shake_256(f"test entropy {_state[0]}".encode()).digest(n)


def _tree(tmp_path: Path, after: str | None = None, **edits) -> tuple[Path, dict]:
    digest = PB.RB.step_digest(REG)
    ch = Ch.LiveChallenger(COIN.SCHEME, 0, [(COIN.LABEL, 1)], _entropy)
    opening = ch.open(COIN.LABEL, 0, bytes.fromhex(after or digest))
    units = sorted(COIN.draw(opening.coin, after or digest, REG["law"]["k"], REG["population"]))
    coin = {"registration": digest, "scheme": COIN.SCHEME, "families": [[COIN.LABEL, 1]], "domain": COIN.DOMAIN,
            "law": REG["law"], "population": REG["population"], "commitment": ch.commitment.root.hex(),
            "opening": opening.to_json(), "units": units, "reserve": {"session": "s20261027T001504Z-test"}, **edits}
    (tmp_path / "coin.json").write_text(json.dumps(coin))
    return tmp_path, {"law": "subset", "k": 8, "population": REG["population"], "units": units}


def test_an_honest_coin_re_derives_the_draw(tmp_path) -> None:
    d, draw = _tree(tmp_path)
    held, derived, info = PB.held_coin(d, REG, draw, None)
    assert derived == draw["units"] and len(derived) == 8
    assert {k: held[k] for k in ("coin:registration", "coin:layout", "coin:units")} == dict.fromkeys(
        ("coin:registration", "coin:layout", "coin:units"), True)
    assert held["coin:session"] is False and "main" in info["why"]


def test_a_wrong_index_in_the_draw_is_refused(tmp_path) -> None:
    d, draw = _tree(tmp_path)
    wrong = sorted({*draw["units"][1:], (draw["units"][0] + 1) % REG["population"]})
    held, derived, _ = PB.held_coin(d, REG, {**draw, "units": wrong}, None)
    assert derived == draw["units"] != wrong and held["coin:units"] is False


@pytest.mark.parametrize("why, kw", [
    ("opened after another root", {"after": "ab" * 64}),
    ("another domain", {"domain": "verity/e2e-h200-m3/draw/v2"}),
    ("another registration", {"registration": "cd" * 64}),
])
def test_a_coin_not_this_steps_is_refused(tmp_path, why, kw) -> None:
    after = kw.pop("after", None)
    d, draw = _tree(tmp_path, after, **kw)
    held, _, _ = PB.held_coin(d, REG, draw, None)
    assert not all(held[k] for k in ("coin:registration", "coin:layout", "coin:units")), why


def test_no_coin_refuses_the_step(tmp_path) -> None:
    assert PB.held_coin(tmp_path, REG, {"units": []}, None)[0] == {"coin": False}
