"""A draw's coins from the auditor's challenger (`coins.ChallengerCoins`): the stream is one family's coins opened after
the prover's commitment, concatenated in order, extended with the family's next coins, each checked against the
challenger's commitment before its bytes are used; the verifier replays it from the record (`challenger_stream`,
`read_reason`) against the roots the challenger's signed log shows. A bad opening, a commitment drawn twice, coins the
log doesn't show released after the commitment, coins of another commitment and a coin opened but not read are refused,
in process, over a served challenger, and through the proof system's `Stream.select`; and the Lean verifier's
run refuses an outcome whose coins' evidence fails (`Stream.outcome(coins=...)`)."""

from __future__ import annotations

import dataclasses
import itertools
import json

import pytest

from verity.core.protocols.sampled_proofs import challenger as C, coins as T, remote as R, service as S

from test_challenger import entropy
from test_remote import Host, signer  # noqa: F401  (the challenger key's signer fixture)

SCHEME = "verity/test/coins/v1"
FAMILIES = [("other", 2), ("draw", 80)]
ROOT = bytes([0xab]) * 64
SUBSET = {"law": "subset", "k": 4}
#: `draw_first`'s stream for subset:4 over 678 proof blocks: 2 k · width(678) + 64 bytes, width 10
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


def test_the_proof_system_draws_from_the_challengers_coins_after_the_receipt():
    """`Stream.select` with the coins as its `coins`, committed to the registration's digest after its receipt: the
    draw both verifiers agree on, from bytes the verifier replays from the record. Without the commitment the
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


#: The draws `coin_outcomes` runs: the honest one, then each the Lean run refuses at its select (`coins`).
#: `other-window` stays last: a second host serves it.
COIN_CASES = ["honest", "skipped", "wrong", "before", "missing", "empty", "no-window", "off-main", "other-window"]


def coin_outcomes(tmp_path, signer, monkeypatch, register, positions: int = 16):  # noqa: F811
    """Each of `COIN_CASES`' outcomes over a served challenger: `register(coins, i, window_root)` is case i's stream on
    the challenger's `coins` and its registration under `window_root`, drawn after the registration's digest, its proof
    delivered, and its outcome run with the case's evidence (`coins_reason`'s: the challenger key, the auditor's
    ledger, the verifier's copy of main's first-parent history, the host's pins log, the segment and the session's
    files): the honest one; a segment that skips a coin opened after the digest, one with a wrong opening, one that
    adds an earlier registration's coins; no evidence, empty evidence; a registration without its window root; a
    session whose pin isn't on the verifier's copy of main (`pinned_session`); and coins a session of another window
    released after the same digest. Each session's challenger holds `positions` coins of the draws' family.
    -> ({case: (its stream, its outcome, its audit record)}, {case: its segment})."""
    import test_remote
    from test_service import SESSION, _lean

    monkeypatch.setattr(test_remote, "FAMILIES", [("positions", positions), ("replay", 2)])
    window, other = test_remote.WINDOW, {"root": "22" * 64}
    draws = {}

    def serve(host: Host, served: list[str]) -> None:
        client = R.RemoteChallenger.connect(host.address)
        coins = T.ChallengerCoins(C.LiveRecord(client, "positions", itertools.count().__next__))
        for name in served:
            stream, reg = register(coins, COIN_CASES.index(name), None if name == "no-window" else window["root"])
            assert stream.coin_source == "challenger"
            segment = coins.commit(bytes.fromhex(reg.digest))
            draws[name] = (stream, stream.select(reg), segment.to_json())
        client.close()
        assert host.ended() is None

    host = Host(tmp_path, signer)
    serve(host, COIN_CASES[:-1])
    alt = Host(tmp_path, signer, session="s-2", seed="alt", ledger=host.ledger, window=other)
    serve(alt, COIN_CASES[-1:])
    ledger = R.ledger_lines(host.ledger.path)
    files = {name: R.session_files((alt if name == "other-window" else host).directory) for name in COIN_CASES}
    segments = {name: d[2] for name, d in draws.items()}
    s = segments["skipped"]
    segments["skipped"] = {**s, "openings": s["openings"][1:], "start": s["start"] + 1}
    s = segments["wrong"]
    segments["wrong"] = {**s, "openings": [{**s["openings"][0], "coin": "00" * 64}, *s["openings"][1:]]}
    s = segments["before"]
    earlier = [{**o, "after": s["after"]} for o in draws["wrong"][2]["openings"]]
    segments["before"] = {**s, "openings": earlier + s["openings"], "start": s["start"] - len(earlier)}
    out = {}
    for name, (stream, draw, _) in draws.items():
        evidence = {"key": test_remote.pinned_key().to_json(), "ledger": ledger,
                    "main": [] if name == "off-main" else test_remote.MAIN, "pins": test_remote.HOST_PINS,
                    "segment": segments[name], "session": files[name]}
        evidence = {"missing": None, "empty": {}}.get(name, evidence)
        assert stream.deliver(draw, [SESSION]) is None
        out[name] = (stream, stream.outcome(draw, _lean(), {"statement": "stub"}, coins=evidence),
                     stream.records[-1]["audit"])
    return out, segments


def assert_only_honest_coins_accepted(outcomes) -> None:
    """The honest case's outcome is accepted with no `run` failure, and every other is a reject whose run the Lean
    refused at its select step (`coins`)."""
    for name, (_, out, audit) in outcomes.items():
        if name == "honest":
            assert out.accepted and "run" not in audit["checks"], audit["checks"].get("run")
        else:
            assert not out.accepted and out.verdict.detail["reject"]["checks"] == ["run"], name
            lean = audit["checks"]["run"]["detail"]["disagree"]["lean"]
            assert (lean["step"], lean["codes"]) == ("select", ["coins"]), (name, lean)


def test_the_lean_run_refuses_a_draw_whose_coins_the_challenger_did_not_release_after_the_registration(
        tmp_path, signer, monkeypatch):  # noqa: F811
    """On a stream drawing from the challenger (`coin_source` `challenger`, which its digest names), every outcome's run
    by the verifier (`Run.run`) checks the draw's coins: with their evidence for the registration's window
    (`window_root`) it accepts the honest draw, and it rejects, failing `run` with the select step's `coins`, each of
    `coin_outcomes`' other cases (a prover picking the window among them). A sampler beside the challenger's coins, and
    evidence for a draw that read none, are refused."""
    import test_remote
    from test_service import SESSION, _lean, _ports, _record, _stream

    def register(coins, i, window_root):
        stream = _stream(coins=coins)
        return stream, stream.register(_record(6 + i), _ports(6 + i), window_root=window_root)

    outcomes, segments = coin_outcomes(tmp_path, signer, monkeypatch, register)
    assert_only_honest_coins_accepted(outcomes)
    coins = T.ChallengerCoins(C.LiveRecord(C.LiveChallenger(test_remote.SCHEME, 0, test_remote.FAMILIES,
                                                            entropy("x")), "positions", itertools.count().__next__))
    assert "coins" not in json.dumps(_stream().digest) and outcomes["honest"][0].digest != _stream().digest
    with pytest.raises(ValueError, match="no sampler"):
        _stream(coins=coins, sampler=lambda law, n: [0, 1])
    sampled = _stream(sampler=lambda law, n: [0, 1])
    draw = sampled.select(sampled.register(_record(), _ports(6)))
    assert sampled.deliver(draw, [SESSION]) is None
    with pytest.raises(S.Refused, match="coins") as refused:
        sampled.outcome(draw, _lean(), {"statement": "stub"}, coins={"segment": segments["honest"]})
    assert "read none" in refused.value.verdict.detail["coins"]
