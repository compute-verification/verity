"""Live coins by label and index (`coins.LiveChallenger`, `live_reason`, `order_reason`) and PoUW's live epoch
(`live_challenger`, `rounds_reason`, `round_key`): the challenger commits to every coin before the epoch and opens each
once; the verifier refuses a coin opened twice, one opened before its round's root was recorded, a foreign epoch's or
challenger's, and every recorded root whose coin was never opened; each round's noise key is its own coin's and root's;
and the seeded scheme is unchanged beside it."""

from __future__ import annotations

import dataclasses
import random

import pytest

from verity.core.protocols.pouw import coins as C
from verity.core.protocols.pouw.protocol import DomainError


def entropy(seed: str):
    return random.Random(seed).randbytes


def window(rounds: int = 4, k: int | None = None, epoch: int = 7, seed: str = "a"
           ) -> tuple[C.LiveChallenger, list[bytes], list[C.LiveOpening]]:
    """A live epoch's challenger and a window that recorded `k` round roots, each round's coin opened after its root."""
    ch = C.live_challenger(epoch, rounds, entropy(seed))
    rec = C.LiveRecord(ch, C.ROUND)
    for r in range(rounds if k is None else k):
        rec.record(bytes([r]) * 64)
    return ch, rec.roots, rec.openings


def test_an_honest_window_is_accepted_round_by_round():
    ch, roots, ops = window()
    assert C.rounds_reason(ch.commitment, roots, ops) is None
    for k in range(5):
        assert C.rounds_reason(ch.commitment, roots[:k], ops[:k]) is None
    assert [(o.label, o.index, o.after) for o in ops] == [(C.ROUND, r, roots[r]) for r in range(4)]
    assert C.live_epoch_reason(ch.commitment) is None and ch.commitment.families == (("salt", 1), ("draw", 1), ("round", 4))


def test_a_coin_opened_twice_is_refused():
    ch, roots, ops = window()
    with pytest.raises(DomainError, match="'round' coin 1 is already open: a retry never redraws"):
        ch.open(C.ROUND, 1, roots[1])
    assert C.rounds_reason(ch.commitment, roots, [ops[0], ops[1], ops[1], ops[2]]) == \
        "the epoch's 'round' coin 1 is opened twice: a retry never redraws"
    assert C.rounds_reason(ch.commitment, roots, ops + [ops[0]]) == \
        "the epoch's 'round' coin 0 is opened twice: a retry never redraws"


def test_a_coin_opened_before_its_rounds_root_was_recorded_is_refused():
    ch, roots, ops = window(k=2)
    early = ch.open(C.ROUND, 2, roots[1])
    assert C.rounds_reason(ch.commitment, roots, ops + [early]) == "'round' coin 2 was opened before its root was recorded"
    assert C.rounds_reason(ch.commitment, roots[:1], ops) == "'round' coin 1 was opened before its root was recorded"
    late_root = bytes([9]) * 64
    assert C.rounds_reason(ch.commitment, roots + [late_root], ops + [early]) == \
        "'round' coin 2: coin was not opened after what the verifier recorded"


def test_a_foreign_epochs_coin_is_refused():
    ch, roots, ops = window(epoch=7)
    _, _, foreign = window(epoch=8)
    assert C.rounds_reason(ch.commitment, roots, foreign) == "'round' coin 0: coin is epoch 8's, not epoch 7's"
    relabelled = [dataclasses.replace(o, epoch=7) for o in foreign]
    assert C.rounds_reason(ch.commitment, roots, relabelled) == \
        "'round' coin 0: coin does not open against the epoch's coin commitment"
    _, _, other = window(epoch=7, seed="another challenger")
    assert C.rounds_reason(ch.commitment, roots, other) == \
        "'round' coin 0: coin does not open against the epoch's coin commitment"


def test_every_recorded_root_has_its_coin_and_the_coins_their_rounds():
    ch, roots, ops = window()
    assert C.rounds_reason(ch.commitment, roots, ops[:3]) == "'round' coin 3 was never opened"
    assert C.rounds_reason(ch.commitment, roots + [roots[0]], ops) == "5 roots recorded, more than the epoch's 4 'round' coins"
    assert C.rounds_reason(ch.commitment, roots, [ops[1], ops[0]] + ops[2:]) == \
        "'round' coin 0: coin is the epoch's 'round' coin 1, not its 'round' coin 0"
    salt = ch.open("salt", 0, roots[0])
    assert C.rounds_reason(ch.commitment, roots[:1], [salt]) == \
        "'round' coin 0: coin is the epoch's 'salt' coin 0, not its 'round' coin 0"
    assert C.rounds_reason(ch.commitment, roots, ops[:1] + [dataclasses.replace(ops[1], salt=bytes(192))] + ops[2:]) == \
        "'round' coin 1: coin does not open against the epoch's coin commitment"
    assert C.rounds_reason(ch.commitment, roots, ops[:1] + [dataclasses.replace(ops[1], coin=ops[1].coin[:31])] + ops[2:]) \
        == "'round' coin 1: coin opening is malformed"


def test_the_record_asks_for_each_coin_only_after_recording_its_root():
    ch = C.live_challenger(3, 2, entropy("r"))
    rec = C.LiveRecord(ch, C.ROUND)
    first = rec.record(b"\x01" * 64)
    assert (first.index, first.after, rec.roots) == (0, b"\x01" * 64, [b"\x01" * 64])
    rec.record(b"\x02" * 64)
    with pytest.raises(DomainError, match="every 'round' coin of the epoch is open"):
        rec.record(b"\x03" * 64)
    assert rec.reason() is None
    with pytest.raises(DomainError, match="no 'noise' family"):
        C.LiveRecord(ch, "noise")


def test_the_round_key_is_its_own_coin_and_root():
    ch, roots, ops = window()
    keys = [C.round_key(ch.commitment, o.coin, r, roots[r]).value for r, o in enumerate(ops)]
    assert len(set(keys)) == 4 and all(len(k) == 32 for k in keys)
    assert C.round_key(ch.commitment, ops[0].coin, 0, roots[1]).value != keys[0]
    assert C.round_key(ch.commitment, ops[1].coin, 0, roots[0]).value != keys[0]
    with pytest.raises(TypeError):
        C.round_key(ch.commitment, b"", 0, roots[0])


def test_a_layout_by_label_and_index_serves_any_client():
    """The live coins are named by no protocol: a client's layout (here one family of commit coins) commits and opens as
    PoUW's does, and is not a live PoUW epoch's."""
    ch = C.LiveChallenger("verity/vllm/commit-coins/v1", 2, [("commit", 3)], entropy("v"))
    o = ch.open("commit", 2, b"x" * 64)
    assert C.live_reason(ch.commitment, o, "commit", 2, b"x" * 64) is None
    assert C.order_reason(ch.commitment, "commit", [], []) is None
    assert C.rounds_reason(ch.commitment, [], []) == \
        "the coin commitment's scheme is 'verity/vllm/commit-coins/v1', not 'verity/pouw/epoch-coins/live/v1'"
    with pytest.raises(DomainError, match="no 'commit' coin 3"):
        ch.open("commit", 3, b"")
    assert C.LiveCommitment.from_json(ch.commitment.to_json()) == ch.commitment
    assert C.LiveOpening.from_json(o.to_json()) == o


@pytest.mark.parametrize("scheme, epoch, families, why", [
    ("", 0, (("a", 1),), "a live coin scheme is"), ("Verity/x", 0, (("a", 1),), "a live coin scheme is"),
    ("v/x", -1, (("a", 1),), "an epoch is"), ("v/x", True, (("a", 1),), "an epoch is"),
    ("v/x", 0, (), "a live coin layout is"), ("v/x", 0, (("a", 0),), "a live coin layout is"),
    ("v/x", 0, (("a/b", 1),), "a live coin layout is"), ("v/x", 0, (("a", True),), "a live coin layout is"),
    ("v/x", 0, (("a", 1), ("a", 2)), "a live coin layout's labels are distinct"),
    ("v/x", 0, (("a", C.MAX_COINS), ("b", 1)), "a live coin layout holds at most"),
])
def test_a_layout_is_checked(scheme, epoch, families, why):
    assert C.layout_problem(scheme, epoch, families).startswith(why)
    with pytest.raises(DomainError, match=f"^{why}"):
        C.LiveChallenger(scheme, epoch, families, entropy("l"))


def test_the_commitment_names_the_epochs_mode_and_the_seeded_scheme_is_unchanged():
    assert (C.mode(C.SCHEME), C.mode(C.LIVE_SCHEME)) == ("seeded", "live") and C.MODES == ("live", "seeded")
    with pytest.raises(DomainError):
        C.mode("verity/pouw/epoch-coins/v2")
    assert (C.SCHEME, C.LABELS) == ("verity/pouw/epoch-coins/v1", ("salt", "draw"))
    seeded = C.Challenger(7, entropy("s"))
    assert C.coin_reason(seeded.commitment, seeded.open("draw", b"t"), "draw", b"t") is None


def test_a_fresh_live_challenger_draws_from_the_operating_system():
    a, b = C.live_challenger(7, 2), C.live_challenger(7, 2)
    assert a.commitment.root != b.commitment.root and len(a.commitment.root) == 64
    with pytest.raises(ValueError, match="returned other than"):
        C.live_challenger(7, 2, lambda n: bytes(n - 1))
