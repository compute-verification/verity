"""The proof service's challenger (`challenger`): it commits to every coin of a layout before the epoch and opens each
once, after the root the verifier recorded for it; the verifier refuses a coin opened twice, one opened before its root
was recorded, a foreign epoch's or another challenger's, a coin that isn't the one asked for, and a recorded root whose
coin was never opened.  The layouts are any client's: named by no protocol."""

from __future__ import annotations

import dataclasses
import random

import pytest

from verity.core.protocols.sampled_proofs import challenger as C

LABEL = "commit"


def entropy(seed: str):
    return random.Random(seed).randbytes


def family(count: int = 4, k: int | None = None, epoch: int = 7, seed: str = "a", families=None
           ) -> tuple[C.LiveChallenger, list[bytes], list[C.LiveOpening]]:
    """A challenger over `families` (one family of `count` commit coins by default) and the verifier's record of `k` of
    its roots, each coin opened after its root."""
    ch = C.LiveChallenger("verity/test/coins/v1", epoch, families or [("other", 2), (LABEL, count)], entropy(seed))
    rec = C.LiveRecord(ch, LABEL)
    for i in range(count if k is None else k):
        rec.record(bytes([i]) * 64)
    return ch, rec.roots, rec.openings


def test_an_honest_record_is_accepted_coin_by_coin():
    ch, roots, ops = family()
    assert C.order_reason(ch.commitment, LABEL, roots, ops) is None
    for k in range(5):
        assert C.order_reason(ch.commitment, LABEL, roots[:k], ops[:k]) is None
    assert [(o.label, o.index, o.after) for o in ops] == [(LABEL, i, roots[i]) for i in range(4)]
    assert ch.commitment.count == 6 and ch.commitment.position(LABEL, 1) == 3 and ch.commitment.position(LABEL, 4) is None


def test_a_coin_opened_twice_is_refused():
    ch, roots, ops = family()
    with pytest.raises(C.CoinError, match="commit coin 1 is already open: a retry never redraws"):
        ch.open(LABEL, 1, roots[1])
    assert C.order_reason(ch.commitment, LABEL, roots, [ops[0], ops[1], ops[1], ops[2]]) == \
        "the epoch's commit coin 1 is opened twice: a retry never redraws"
    assert C.order_reason(ch.commitment, LABEL, roots, ops + [ops[0]]) == \
        "the epoch's commit coin 0 is opened twice: a retry never redraws"


def test_a_coin_opened_before_its_root_was_recorded_is_refused():
    ch, roots, ops = family(k=2)
    early = ch.open(LABEL, 2, roots[1])
    assert C.order_reason(ch.commitment, LABEL, roots, ops + [early]) == "commit coin 2 was opened before its root was recorded"
    assert C.order_reason(ch.commitment, LABEL, roots[:1], ops) == "commit coin 1 was opened before its root was recorded"
    assert C.order_reason(ch.commitment, LABEL, roots + [bytes([9]) * 64], ops + [early]) == \
        "commit coin 2: coin was not opened after what the verifier recorded"


def test_a_foreign_epochs_coin_is_refused():
    ch, roots, ops = family(epoch=7)
    _, _, foreign = family(epoch=8)
    assert C.order_reason(ch.commitment, LABEL, roots, foreign) == "commit coin 0: coin is epoch 8's, not epoch 7's"
    relabelled = [dataclasses.replace(o, epoch=7) for o in foreign]
    assert C.order_reason(ch.commitment, LABEL, roots, relabelled) == \
        "commit coin 0: coin does not open against the epoch's coin commitment"
    _, _, other = family(epoch=7, seed="another challenger")
    assert C.order_reason(ch.commitment, LABEL, roots, other) == \
        "commit coin 0: coin does not open against the epoch's coin commitment"


def test_every_recorded_root_has_its_coin_and_each_coin_its_place():
    ch, roots, ops = family()
    assert C.order_reason(ch.commitment, LABEL, roots, ops[:3]) == "commit coin 3 was never opened"
    assert C.order_reason(ch.commitment, LABEL, roots + [roots[0]], ops) == \
        "5 roots recorded, more than the epoch's 4 commit coins"
    assert C.order_reason(ch.commitment, LABEL, roots, [ops[1], ops[0]] + ops[2:]) == \
        "commit coin 0: coin is the epoch's commit coin 1, not its commit coin 0"
    other = ch.open("other", 0, roots[0])
    assert C.order_reason(ch.commitment, LABEL, roots[:1], [other]) == \
        "commit coin 0: coin is the epoch's other coin 0, not its commit coin 0"
    assert C.order_reason(ch.commitment, LABEL, roots, ops[:1] + [dataclasses.replace(ops[1], salt=bytes(192))] + ops[2:]) \
        == "commit coin 1: coin does not open against the epoch's coin commitment"
    for short in (ops[1].coin[:32], ops[1].coin[:-1]):
        assert C.order_reason(ch.commitment, LABEL, roots, ops[:1] + [dataclasses.replace(ops[1], coin=short)]
                              + ops[2:]) == "commit coin 1: coin opening is malformed"
    with pytest.raises(C.CoinError, match="no noise family"):
        C.order_reason(ch.commitment, "noise", [], [])
    with pytest.raises(C.CoinError, match="no commit coin 4"):
        C.live_reason(ch.commitment, ops[0], LABEL, 4, roots[0])


def test_the_record_asks_for_each_coin_only_after_recording_its_root():
    ch = C.LiveChallenger("verity/test/coins/v1", 3, [(LABEL, 2)], entropy("r"))
    rec = C.LiveRecord(ch, LABEL)
    first = rec.record(b"\x01" * 64)
    assert (first.index, first.after, rec.roots) == (0, b"\x01" * 64, [b"\x01" * 64])
    rec.record(b"\x02" * 64)
    with pytest.raises(C.CoinError, match="every commit coin of the epoch is open"):
        rec.record(b"\x03" * 64)
    assert rec.reason() is None
    with pytest.raises(C.CoinError, match="no noise family"):
        C.LiveRecord(ch, "noise")


def test_the_record_logs_each_release_after_the_root_it_follows():
    """`LiveRecord.log` is the release order on the record's clock: root i's receipt, then coin i's release.  That the
    challenger released coin i only after receiving root i is trusted (`trusted-nebius-host`), not checked: the log
    records the order, and `order_reason` reads only the roots the openings name."""
    ticks = iter(range(100))
    ch = C.LiveChallenger("verity/test/coins/v1", 3, [("other", 1), (LABEL, 3)], entropy("t"))
    rec = C.LiveRecord(ch, LABEL, clock=lambda: float(next(ticks)))
    roots = [bytes([i + 1]) * 64 for i in range(3)]
    ops = [rec.record(r) for r in roots]
    assert [(e["event"], e["label"], e["index"]) for e in rec.log] == \
        [(event, LABEL, i) for i in range(3) for event in ("root", "release")]
    assert [e["received_at"] if e["event"] == "root" else e["released_at"] for e in rec.log] == [0., 1., 2., 3., 4., 5.]
    assert [e["root"] for e in rec.log[::2]] == [r.hex() for r in roots] == [e["after"] for e in rec.log[1::2]]
    assert rec.reason() is None and all(len(o.coin) == C.COIN_BYTES == 64 for o in ops)
    with pytest.raises(C.CoinError, match="every commit coin of the epoch is open"):
        rec.record(b"\x04" * 64)
    assert len(rec.log) == 6

    ch = C.LiveChallenger("verity/test/coins/v1", 3, [(LABEL, 2)], entropy("u"))
    early = C.LiveRecord(ch, LABEL, clock=lambda: float(next(ticks)))
    ch.open(LABEL, 0, b"\x09" * 64)
    with pytest.raises(C.CoinError, match="commit coin 0 is already open"):
        early.record(b"\x01" * 64)
    assert [e["event"] for e in early.log] == ["root"] and early.reason() == "commit coin 0 was never opened"


def test_one_coin_and_a_family_of_many_open_alike():
    """A tree of one coin (no path) and families past a power of two (padded trees) open as the others do."""
    for families in ([(LABEL, 1)], [("other", 5), (LABEL, 3)], [(LABEL, 17)]):
        ch, roots, ops = family(count=dict(families)[LABEL], families=families)
        assert C.order_reason(ch.commitment, LABEL, roots, ops) is None
    assert family(count=1, families=[(LABEL, 1)])[2][0].path == ()


def test_the_json_forms_round_trip_and_refuse_what_they_dont_read():
    ch, roots, ops = family()
    assert C.LiveCommitment.from_json(ch.commitment.to_json()) == ch.commitment
    assert C.LiveOpening.from_json(ops[1].to_json()) == ops[1]
    doc = ch.commitment.to_json()
    for bad in ({**doc, "extra": 1}, {**doc, "families": [("commit", 1)]}, {**doc, "root": doc["root"].upper()},
                {**doc, "root": doc["root"][:-1]}, [doc]):
        with pytest.raises(C.CoinError, match="^a live coin commitment is"):
            C.LiveCommitment.from_json(bad)
    with pytest.raises(C.CoinError, match="^a coin commitment's root is 64 bytes"):
        C.LiveCommitment.from_json({**doc, "root": doc["root"][:-2]})
    with pytest.raises(C.CoinError, match="^a live coin layout's labels are distinct"):
        C.LiveCommitment.from_json({**doc, "families": [["a", 1], ["a", 1]]})
    o = ops[1].to_json()
    for bad in ({**o, "epoch": True}, {**o, "index": "1"}, {**o, "label": 1}, {**o, "coin": " " + o["coin"]},
                {**o, "path": o["path"] + ["00" * 32]}, {**o, "path": "00"}, {k: v for k, v in o.items() if k != "salt"}):
        with pytest.raises(C.CoinError, match="^a live coin opening is"):
            C.LiveOpening.from_json(bad)


@pytest.mark.parametrize("scheme, epoch, families, why", [
    ("", 0, (("a", 1),), "a live coin scheme is"), ("Verity/x", 0, (("a", 1),), "a live coin scheme is"),
    ("v/x", -1, (("a", 1),), "an epoch is"), ("v/x", True, (("a", 1),), "an epoch is"),
    ("v/x", 0, (), "a live coin layout is"), ("v/x", 0, (("a", 0),), "a live coin layout is"),
    ("v/x", 0, (("a b", 1),), "a live coin layout is"), ("v/x", 0, (("a", True),), "a live coin layout is"),
    ("v/x", 0, [("a", 1)], "a live coin layout is"),
    ("v/x", 0, (("a", 1), ("a", 2)), "a live coin layout's labels are distinct"),
    ("v/x", 0, (("a", C.MAX_COINS), ("b", 1)), "a live coin layout holds at most"),
])
def test_a_layout_is_checked(scheme, epoch, families, why):
    assert C.layout_problem(scheme, epoch, families).startswith(why)
    if type(families) is tuple:
        with pytest.raises(C.CoinError, match=f"^{why}"):
            C.LiveChallenger(scheme, epoch, families, entropy("l"))


def test_a_label_may_name_its_client():
    """A client names its coins in its own label space, as PoUW's round coins are ``pouw/round``."""
    ch = C.LiveChallenger("verity/test/coins/v1", 0, [("pouw/round", 2), ("vllm/positions", 1)], entropy("n"))
    o = ch.open("vllm/positions", 0, b"r" * 64)
    assert C.live_reason(ch.commitment, o, "vllm/positions", 0, b"r" * 64) is None


def test_a_fresh_challenger_draws_from_the_operating_system():
    a, b = (C.LiveChallenger("verity/test/coins/v1", 7, [(LABEL, 2)]) for _ in range(2))
    assert a.commitment.root != b.commitment.root and len(a.commitment.root) == 64
    with pytest.raises(ValueError, match="returned other than"):
        C.LiveChallenger("verity/test/coins/v1", 7, [(LABEL, 2)], lambda n: bytes(n - 1))
