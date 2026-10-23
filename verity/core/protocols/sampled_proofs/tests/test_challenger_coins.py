"""A draw's coins from the auditor's challenger (`coins.ChallengerCoins`): the stream is one family's coins opened after
the prover's commitment, concatenated in order, extended with the family's next coins, each checked against the
challenger's commitment before its bytes are used; the verifier replays it from the record (`challenger_stream`,
`read_reason`) against the roots the challenger's signed log shows. A bad opening, a commitment drawn twice, coins the
log doesn't show released after the commitment, coins of another commitment and a coin opened but not read are refused,
in process, over a served challenger, and through the proof service's `Stream.select`."""

from __future__ import annotations

import dataclasses
import itertools

import pytest

from verity.core.protocols.sampled_proofs import challenger as C, coins as T, remote as R

from test_challenger import entropy
from test_remote import Host, signer  # noqa: F401  (the challenger key's signer fixture)

SCHEME = "verity/test/coins/v1"
FAMILIES = [("other", 2), ("draw", 80)]
ROOT = bytes([0xab]) * 64
SUBSET = {"law": "subset", "k": 4}
#: `draw_first`'s stream for subset:4 over 678 units: 2 k · width(678) + 64 bytes, width 10
NEED = 2 * 4 * 10 + 64


def _coins(challenger: C.LiveChallenger | None = None) -> tuple[C.LiveChallenger, T.ChallengerCoins]:
    ch = challenger or C.LiveChallenger(SCHEME, 0, FAMILIES, entropy("host"))
    return ch, T.ChallengerCoins(C.LiveRecord(ch, "draw", itertools.count().__next__))


def _reading(coins):
    read = bytearray()

    def source(n: int) -> bytes:
        b = coins(n)
        read.extend(b)
        return b
    return source, read


def test_the_draw_reads_the_coins_opened_after_the_commitment_in_order():
    """subset:4 over 678 reads 144 bytes, three coins, opened in one batch after the commitment; the draw is
    `draw_first` on their concatenation, and the verifier replays it from the record."""
    ch, coins = _coins()
    segment = coins.commit(ROOT)
    source, read = _reading(coins)
    draw = T.draw_first(T.OwnRandomness(source), SUBSET, 678)
    assert len(read) == NEED and [o.index for o in segment.openings] == [0, 1, 2]
    assert coins.record.roots == [ROOT] * 3 and coins.record.reason() is None
    stream = b"".join(o.coin for o in segment.openings)
    assert bytes(read) == stream[:NEED]
    assert draw == T.draw_first(T.OwnRandomness(T._on(stream)), SUBSET, 678)
    doc = segment.to_json()
    assert T.challenger_stream(ch.commitment, doc, ROOT, coins.record.roots) == stream
    assert T.read_reason(ch.commitment, doc, ROOT, bytes(read), coins.record.roots) is None


def test_a_rerun_extends_with_the_familys_next_coins_only():
    """A second read past the opened coins opens only what is missing, the next indices, after the same commitment."""
    ch, coins = _coins()
    segment = coins.commit(ROOT)
    first, second = coins(NEED), coins(NEED)
    assert [o.index for o in segment.openings] == [0, 1, 2, 3, 4]
    stream = b"".join(o.coin for o in segment.openings)
    assert first + second == stream[:2 * NEED]
    assert T.read_reason(ch.commitment, segment.to_json(), ROOT, first + second, coins.record.roots) is None
    assert "a coin opened and not read" in T.read_reason(ch.commitment, segment.to_json(), ROOT, first,
                                                         coins.record.roots)


def test_more_coins_than_a_batch_open_in_batches():
    ch, coins = _coins()
    segment = coins.commit(ROOT)
    assert len(coins((R.MAX_BATCH + 3) * 64)) == (R.MAX_BATCH + 3) * 64
    assert [o.index for o in segment.openings] == list(range(R.MAX_BATCH + 3))


class _Tampered(C.LiveChallenger):
    def open(self, label: str, index: int, after: bytes) -> C.LiveOpening:
        o = super().open(label, index, after)
        return dataclasses.replace(o, coin=bytes(64)) if index == 1 else o


def test_a_bad_opening_is_refused_before_its_bytes_are_used_and_every_read_after():
    _, coins = _coins(_Tampered(SCHEME, 0, FAMILIES, entropy("host")))
    coins.commit(ROOT)
    with pytest.raises(C.CoinError, match="draw coin 1: coin does not open"):
        coins(NEED)
    with pytest.raises(C.CoinError, match="a retry never redraws"):
        coins(1)
    with pytest.raises(C.CoinError, match="a retry never redraws"):
        coins.commit(bytes(64))


@pytest.mark.parametrize("change, why", [
    (lambda d: {**d, "openings": [{**d["openings"][0], "coin": "00" * 64}, *d["openings"][1:]]}, "does not open"),
    (lambda d: {**d, "openings": [{**o, "after": "cd" * 64} for o in d["openings"]]}, "not opened after"),
    (lambda d: {**d, "start": 1}, "not its draw coin 1"),
    (lambda d: {**d, "openings": d["openings"][1:], "start": 1}, "not the ones the challenger's log shows"),
    (lambda d: {**d, "after": "cd" * 64}, "not opened after the prover's commitment"),
    (lambda d: {**d, "label": "other"}, "not its other coin"),
    (lambda d: {k: v for k, v in d.items() if k != "start"}, "a draw's coins are"),
])
def test_the_verifier_refuses_coins_other_than_those_released_after_the_commitment(change, why):
    ch, coins = _coins()
    segment = coins.commit(ROOT)
    read = coins(NEED)
    reason = T.read_reason(ch.commitment, change(segment.to_json()), ROOT, read, coins.record.roots)
    assert reason is not None and why in reason, reason


def test_a_commitment_is_drawn_once_and_a_read_needs_one():
    """A second draw for the same commitment would take the family's later coins, which the prover could pick among:
    refused. A read before any commitment opens nothing."""
    _, coins = _coins()
    with pytest.raises(C.CoinError, match="only after the prover's commitment"):
        coins(1)
    assert coins.record.roots == []
    coins.commit(ROOT)
    coins(1)
    with pytest.raises(C.CoinError, match="drawn once"):
        coins.commit(ROOT)
    other = coins.commit(bytes([1]) * 64)
    assert coins(64) == other.openings[0].coin and other.start == 1


def test_coins_skipped_after_the_commitment_are_caught_by_the_log():
    """A client that opened a coin after the commitment and then presents a later one: the challenger's log shows the
    first coin released after the commitment, so the record's coins are not all of them."""
    ch, coins = _coins()
    record = coins.record
    record.record(ROOT)
    skipped = T.Segment("draw", 1, ROOT, [record.record(ROOT)])
    read = skipped.openings[0].coin
    assert "not the ones the challenger's log shows" in T.read_reason(ch.commitment, skipped.to_json(), ROOT, read,
                                                                      record.roots)
    assert T.read_reason(ch.commitment, skipped.to_json(), ROOT, read) is None


def test_a_served_challengers_coins_replay_against_its_signed_log(tmp_path, signer):  # noqa: F811
    """Over a challenger served on loopback (`remote.serve`): the draw opens its coins in one round trip, and the
    verifier replays it against the commitment and roots it trusts from the challenger's directory and the auditor's
    ledger (`remote.trusted_session`)."""
    host = Host(tmp_path, signer)
    client = R.RemoteChallenger.connect(host.address)
    coins = T.ChallengerCoins(C.LiveRecord(client, "positions", itertools.count().__next__))
    segment = coins.commit(ROOT)
    source, read = _reading(coins)
    law = {"law": "subset", "k": 1}
    draw = T.draw_first(T.OwnRandomness(source), law, 6)
    client.close()
    assert host.ended() is None
    commitment, roots = host.trusted()
    assert roots["positions"] == (ROOT, ROOT) and len(read) == 2 * 9 + 64
    doc = segment.to_json()
    assert T.read_reason(commitment, doc, ROOT, bytes(read), roots["positions"]) is None
    assert draw == T.draw_first(T.OwnRandomness(T._on(T.challenger_stream(commitment, doc, ROOT))), law, 6)


def test_the_proof_service_draws_from_the_challengers_coins_after_the_receipt():
    """`Stream.select` with the coins as its `coins`, committed to the registration's digest after its receipt: the
    draw both verifiers of record agree on, from bytes the verifier replays from the record. Without the commitment the
    select reads no coin."""
    from test_service import _ports, _record, _stream

    ch, coins = _coins()
    stream = _stream(coins=coins)
    reg = stream.register(_record(), _ports(6))
    with pytest.raises(C.CoinError, match="only after the prover's commitment"):
        stream.select(reg)
    stream = _stream(coins=coins)
    reg = stream.register(_record(), _ports(6))
    segment = coins.commit(bytes.fromhex(reg.digest))
    draw = stream.select(reg)
    read = stream._entries[reg.order].read
    assert len(draw.units) == 2 and read
    assert T.read_reason(ch.commitment, segment.to_json(), bytes.fromhex(reg.digest), read, coins.record.roots) is None
