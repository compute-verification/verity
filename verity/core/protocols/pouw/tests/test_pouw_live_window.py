"""PoUW's live window, the proof service's challenger's first client (`window.live_challenger`, `rounds_reason`,
`round_key`, `mode`): the epoch's coins are committed before it and each round's coin opens once, after the verifier
recorded that round's root; the verifier refuses a round coin opened twice, one opened before its round's root was
recorded and a foreign epoch's; each round's noise key is its own coin's and root's; the check item's ``coins`` names
the window's mode, and a seeded window is as it was."""

from __future__ import annotations

import dataclasses
import random

import pytest

from verity.core.protocols.pouw import coins as Co
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.sampled_proofs import challenger as Ch


def entropy(seed: str):
    return random.Random(seed).randbytes


def window(rounds: int = 4, k: int | None = None, epoch: int = 7, seed: str = "a"
           ) -> tuple[Ch.LiveChallenger, list[bytes], list[Ch.LiveOpening]]:
    """A live epoch's challenger and a window that recorded `k` round roots, each round's coin opened after its root."""
    ch = Wd.live_challenger(epoch, rounds, entropy(seed))
    rec = Ch.LiveRecord(ch, Wd.ROUND)
    for r in range(rounds if k is None else k):
        rec.record(bytes([r]) * 64)
    return ch, rec.roots, rec.openings


def test_an_honest_window_is_accepted_round_by_round():
    ch, roots, ops = window()
    assert ch.commitment.families == ((Wd.SALT, 1), (Wd.DRAW, 1), (Wd.ROUND, 4))
    assert ch.commitment.scheme == Wd.LIVE_SCHEME and Wd.live_epoch_reason(ch.commitment) is None
    for k in range(5):
        assert Wd.rounds_reason(ch.commitment, roots[:k], ops[:k]) is None
    assert [(o.label, o.index, o.after) for o in ops] == [(Wd.ROUND, r, roots[r]) for r in range(4)]


def test_a_round_coin_opened_twice_is_refused():
    ch, roots, ops = window()
    with pytest.raises(Ch.CoinError, match="pouw/round coin 1 is already open: a retry never redraws"):
        ch.open(Wd.ROUND, 1, roots[1])
    assert Wd.rounds_reason(ch.commitment, roots, [ops[0], ops[1], ops[1], ops[2]]) == \
        "the epoch's pouw/round coin 1 is opened twice: a retry never redraws"


def test_a_round_coin_opened_before_its_rounds_root_was_recorded_is_refused():
    ch, roots, ops = window(k=2)
    early = ch.open(Wd.ROUND, 2, roots[1])
    assert Wd.rounds_reason(ch.commitment, roots, ops + [early]) == \
        "pouw/round coin 2 was opened before its root was recorded"
    assert Wd.rounds_reason(ch.commitment, roots + [bytes([9]) * 64], ops + [early]) == \
        "pouw/round coin 2: coin was not opened after what the verifier recorded"


def test_a_foreign_epochs_round_coin_is_refused():
    ch, roots, _ = window(epoch=7)
    _, _, foreign = window(epoch=8)
    assert Wd.rounds_reason(ch.commitment, roots, foreign) == "pouw/round coin 0: coin is epoch 8's, not epoch 7's"
    assert Wd.rounds_reason(ch.commitment, roots, [dataclasses.replace(o, epoch=7) for o in foreign]) == \
        "pouw/round coin 0: coin does not open against the epoch's coin commitment"


def test_only_a_live_epochs_coins_carry_a_window():
    vllm = Ch.LiveChallenger("verity/vllm/commit-coins/v1", 7, Wd.live_families(2), entropy("v"))
    assert Wd.rounds_reason(vllm.commitment, [], []) == \
        f"the coin commitment's scheme is verity/vllm/commit-coins/v1, not {Wd.LIVE_SCHEME}"
    swapped = Ch.LiveChallenger(Wd.LIVE_SCHEME, 7, ((Wd.DRAW, 1), (Wd.SALT, 1), (Wd.ROUND, 2)), entropy("s"))
    assert Wd.rounds_reason(swapped.commitment, [], []) == \
        "the coin commitment's coins are not a live epoch's: one salt, one draw, then the rounds'"


def test_the_round_key_is_its_own_coin_and_root():
    ch, roots, ops = window()
    keys = [Wd.round_key(ch.commitment, o.coin, r, roots[r]).value for r, o in enumerate(ops)]
    assert len(set(keys)) == 4 and all(len(k) == 32 for k in keys)
    assert Wd.round_key(ch.commitment, ops[0].coin, 0, roots[1]).value != keys[0]
    assert Wd.round_key(ch.commitment, ops[1].coin, 0, roots[0]).value != keys[0]
    other, _, _ = window(seed="b")
    assert Wd.round_key(other.commitment, ops[0].coin, 0, roots[0]).value != keys[0]


def test_core_checks_rounds_reads_the_challengers_json():
    ch, roots, ops = window()
    args = ch.commitment.to_json(), [r.hex() for r in roots], [o.to_json() for o in ops]
    assert CC.rounds(*args) is None
    seeded = Co.Challenger(7, entropy("s")).commitment.to_json()
    assert CC.rounds(seeded, *args[1:]) == ("coin", f"the verifier's coins are malformed: {Ch.COMMITMENT_FORM}")
    assert CC.rounds(args[0], ["AB" * 64] + args[1][1:], args[2]) == \
        ("rounds", "the round roots are not a list of SHA-512 digests")


def _params(check: dict) -> dict:
    template, tile = next(iter(CC.TILES.items()))
    return {"check": {"entries": 4, "scheme": CC.SCHEME, "template": template, "tile_work": tile.tile_work, **check},
            "context": "served-zk/run-1/ab", "delta": "1/100", "eps": "1/10", "grid_bits": 2, "query": tile.query,
            "range": [1, 64], "shape": [{"m": 16, "n": 1, "tile0": 0}]}


def test_the_check_item_names_the_windows_coin_mode():
    assert (Wd.mode({}), Wd.mode({"coins": "live"}), Wd.mode({"coins": "seeded"})) == ("seeded", "live", "seeded")
    for coins in ({}, {"coins": "live"}, {"coins": "seeded"}):
        assert CC.params(_params(coins))["check"] == _params(coins)["check"]
    with pytest.raises(CC.Refused, match="^check.coins is not live or seeded$"):
        CC.params(_params({"coins": "fiat-shamir"}))
    with pytest.raises(CC.Refused, match="^check's keys are not entries, scheme, template, tile_work, and optionally coins$"):
        CC.params(_params({"noise": "live"}))


def test_a_seeded_window_is_as_it_was():
    assert (Co.SCHEME, Co.LABELS) == ("verity/pouw/epoch-coins/v1", ("salt", "draw"))
    seeded = Co.Challenger(7, entropy("s"))
    assert Co.coin_reason(seeded.commitment, seeded.open("draw", b"t"), "draw", b"t") is None
    assert "coins" not in CC.expected_items(CC.params(_params({})), 0, Wd.Work.at(0, 17), 1, None)["check"]
