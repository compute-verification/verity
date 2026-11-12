"""A sample's coins from the auditor's challenger (`coins.ChallengerCoins`): the stream is one family's coins opened after
the prover's commitment, concatenated in order, extended with the family's next coins, each checked against the
challenger's commitment before its bytes are used; the verifier replays it from the record (`challenger_stream`,
`read_reason`) against the roots the challenger's signed log shows. A bad opening, a commitment sampled twice, coins the
log doesn't show released after the commitment, coins of another commitment and a coin opened but not read are refused,
in process, over a served challenger, and through the proof system's `Stream.select`; and the Lean verifier's
run refuses an outcome whose coins' evidence fails (`Stream.outcome(coins=...)`). A family's coins follow one
commitment: the client takes no second one in a family (`ChallengerCoins(record, *more)`: a family for each), and both
verifiers refuse a family whose log shows another root (`FAMILY_ROOT`, the run's `family`), so a prover that reads a
sample and registers again has no sample a verifier takes. A window's seed salt
(`coins.draw_seed_salt`) is its family's first coin after the window's open root, drawn once, which the verifier reads
back from the challenger's log (`coins.seed_salt_of`), refusing a second coin, a forged one and another window's."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import re

import pytest

from verity.protocols.spot.device import challenger as C, coins as T, remote as R
from verity.protocols.spot.verifier import service as S

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
    """subset:4 over 678 reads 144 bytes, three coins, opened in one batch after the commitment; the sample is
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


def test_a_family_is_drawn_once_and_a_read_needs_a_commitment():
    """A second sample in a family, for the same commitment or another, would take the family's later coins, which the
    prover could pick among once it has read the first: refused. A read before any commitment opens nothing. Coins over
    several families take a commitment each, in order, every one from its family's first coin."""
    ch, coins = _coins()
    with pytest.raises(C.CoinError, match="only after the prover's commitment"):
        coins(1)
    assert coins.record.roots == [] and coins.label == "draw"
    coins.commit(ROOT)
    coins(1)
    for again in (ROOT, bytes([1]) * 64):
        with pytest.raises(C.CoinError, match="drawn once"):
            coins.commit(again)
    assert coins.record.roots == [ROOT] and len(coins.segments) == 1
    ch = C.LiveChallenger(SCHEME, 0, [("draw/0", 4), ("draw/1", 4)], entropy("host"))
    coins = T.ChallengerCoins(*(C.LiveRecord(ch, f"draw/{i}", itertools.count().__next__) for i in range(2)))
    first = coins.commit(ROOT)
    read = coins(64)
    other = coins.commit(bytes([1]) * 64)
    assert coins.label == "draw/1" and coins(64) == other.openings[0].coin != read
    assert [(g.label, g.start) for g in (first, other)] == [("draw/0", 0), ("draw/1", 0)]
    with pytest.raises(C.CoinError, match="drawn once"):
        coins.commit(bytes([2]) * 64)


def test_a_family_whose_log_shows_another_root_is_refused():
    """A prover's client that opens a coin of the sample's family after another root, before the sample's coins or
    after them: the challenger's log shows it, and the verifier refuses the family (`FAMILY_ROOT`) though the segment's
    coins are every coin the log shows released after the commitment, consecutive and all read. Without the log the
    same segment replays."""
    for before in (True, False):
        ch, coins = _coins()
        segment = coins.commit(ROOT)
        if before:
            coins.record.record(bytes([0xcd]) * 64)
            segment.start = 1
        read = coins(NEED)
        if not before:
            coins.record.record(bytes([0xcd]) * 64)
        assert coins.record.reason() is None and len(set(coins.record.roots)) == 2
        assert T.read_reason(ch.commitment, segment.to_json(), ROOT, read, coins.record.roots) == T.FAMILY_ROOT
        with pytest.raises(C.CoinError, match=re.escape(T.FAMILY_ROOT)):
            T.challenger_stream(ch.commitment, segment.to_json(), ROOT, coins.record.roots)
        assert T.read_reason(ch.commitment, segment.to_json(), ROOT, read) is None


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
    """Over a challenger served on loopback (`remote.serve`): the sample opens its coins in one round trip, and the
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
    sample both verifiers agree on, from bytes the verifier replays from the record. Without the commitment the
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


#: The samples `coin_outcomes` runs: the honest one, then each the Lean run refuses at its select (`coins`, or the code
#: `COIN_CODES` names). `other-window` stays last: a second host serves it.
COIN_CASES = ["honest", "skipped", "wrong", "before", "missing", "empty", "no-window", "off-main", "second-root",
              "other-family", "other-window"]
#: the select step's code for the cases it refuses under another than `coins`
COIN_CODES = {"second-root": "family"}


def coin_outcomes(tmp_path, signer, monkeypatch, register, positions: int = 16):  # noqa: F811
    """Each of `COIN_CASES`' outcomes over a served challenger: `register(coins, i, window_root)` is case i's stream on
    the challenger's `coins` and its registration under `window_root`, sampled after the registration's digest, its proof
    delivered, and its outcome run with the case's evidence (`coins_reason`'s: the challenger key, the auditor's
    ledger, the verifier's copy of main's first-parent history, the host's pins log, the segment and the session's
    files): the honest one; a segment that skips a coin opened after the digest, one with a wrong opening, one that
    adds another registration's coins in front; no evidence, empty evidence; a registration without its window root; a
    session whose pin isn't on the verifier's copy of main (`pinned_session`); a family in which the client opened a
    coin after another root once it had read the sample (the run's `family`); another case's segment, a family the
    stream did not read; and coins a session of another window released after the same digest. Each case's
    registration has a family of its own (`positions/<case>`, `positions` coins), since a family's coins follow one.
    -> ({case: (its stream, its outcome, its audit record)}, {case: its segment})."""
    import test_remote
    from test_service import SESSION, _lean

    monkeypatch.setattr(test_remote, "FAMILIES", [*((f"positions/{name}", positions) for name in COIN_CASES),
                                                  ("replay", 2)])
    window, other = test_remote.WINDOW, {"root": "22" * 64}
    draws = {}

    def serve(host: Host, served: list[str]) -> None:
        client = R.RemoteChallenger.connect(host.address)
        coins = T.ChallengerCoins(*(C.LiveRecord(client, f"positions/{name}", itertools.count().__next__)
                                    for name in served))
        for name in served:
            stream, reg = register(coins, COIN_CASES.index(name), None if name == "no-window" else window["root"])
            segment = coins.commit(bytes.fromhex(reg.digest))
            draws[name] = (stream, stream.select(reg), segment)
            if name == "second-root":
                coins.record.record(bytes([0xcd]) * 64)
        client.close()
        coins.settle()
        for name in served:
            stream, draw, segment = draws[name]
            draws[name] = (stream, draw, segment.to_json())
        assert host.ended() is None

    host = Host(tmp_path, signer, withhold=True)
    serve(host, COIN_CASES[:-1])
    alt = Host(tmp_path, signer, session="s-2", seed="alt", ledger=host.ledger, window=other, withhold=True)
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
    segments["before"] = {**s, "openings": earlier + s["openings"]}
    segments["other-family"] = draws["honest"][2]
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
    refused at its select step (`coins`; `family` for a family whose log shows another root)."""
    for name, (_, out, audit) in outcomes.items():
        if name == "honest":
            assert out.accepted and "run" not in audit["checks"], audit["checks"].get("run")
        else:
            assert not out.accepted and out.verdict.detail["reject"]["checks"] == ["run"], name
            lean = audit["checks"]["run"]["detail"]["disagree"]["lean"]
            assert (lean["step"], lean["codes"]) == ("select", [COIN_CODES.get(name, "coins")]), (name, lean)


def test_the_lean_run_refuses_a_draw_whose_coins_the_challenger_did_not_release_after_the_registration(
        tmp_path, signer, monkeypatch):  # noqa: F811
    """On a stream sampling from the challenger (which its digest names), every outcome's run
    by the verifier (`Run.run`) checks the sample's coins: with their evidence for the registration's window
    (`window_root`) it accepts the honest sample, and it rejects, failing `run` with the select step's `coins`, each of
    `coin_outcomes`' other cases (a prover picking the window among them, or the family), but for the family holding a
    second root, which it rejects with the select step's `family`. A sampler beside the challenger's coins, and
    evidence for a sample that read none, are refused."""
    import test_remote
    from test_service import SESSION, _lean, _ports, _rated, _record, _stream

    def register(coins, i, window_root):
        stream = _stream(coins=coins)
        rec = _rated(stream.spec.selection.text, 6 + i)
        return stream, stream.register(rec, _ports(6 + i), window_root=window_root)

    outcomes, segments = coin_outcomes(tmp_path, signer, monkeypatch, register, positions=32)
    assert_only_honest_coins_accepted(outcomes)
    coins = T.ChallengerCoins(C.LiveRecord(C.LiveChallenger(test_remote.SCHEME, 0, test_remote.FAMILIES, entropy("x")),
                                           "positions/honest", itertools.count().__next__))
    assert outcomes["honest"][0].digest == _stream().digest != _stream(sampler=lambda law, n: [0, 1]).digest
    with pytest.raises(ValueError, match="no sampler"):
        _stream(coins=coins, sampler=lambda law, n: [0, 1])
    sampled = _stream(sampler=lambda law, n: [0, 1])
    draw = sampled.select(sampled.register(_record(), _ports(6)))
    assert sampled.deliver(draw, [SESSION]) is None
    with pytest.raises(S.Refused, match="coins") as refused:
        sampled.outcome(draw, _lean(), {"statement": "stub"}, coins={"segment": segments["honest"]})
    assert "read none" in refused.value.verdict.detail["coins"]


def test_the_lean_run_accepts_a_draw_whose_session_released_its_coins_in_several_frames(tmp_path, signer, monkeypatch):
    """A window's withheld session that opened `test_remote.RELEASED` coins of another family before the draw, so
    many that its release at the close is past `remote.MAX_HEAD` as one frame (which cut the close, so its coins never
    settled) and takes several (`remote.release_frames`): the draw's coins, its own family's from the first, settle
    from them, and the Lean run accepts the draw on that evidence. The evidence with the draw's coins as their raw
    bytes alone, as a cut close left them, is refused: an opening without its salt is malformed."""
    import test_remote
    from test_challenger import lms_verify
    from test_service import SESSION, _lean, _ports, _record, _stream

    monkeypatch.setattr(test_remote, "FAMILIES", [("positions", 64), ("replay", test_remote.RELEASED)])
    host = Host(tmp_path, signer, withhold=True)
    client = R.RemoteChallenger.connect(host.address)
    earlier = C.LiveRecord(client, "replay", itertools.count().__next__)
    for k in range(0, test_remote.RELEASED, R.MAX_BATCH):
        C.record_many([(earlier, bytes([0xcd]) * 64)] * min(R.MAX_BATCH, test_remote.RELEASED - k))
    coins = T.ChallengerCoins(C.LiveRecord(client, "positions", itertools.count().__next__))
    stream = _stream(coins=coins)
    window = test_remote.WINDOW["root"]
    reg = stream.register(_record(), _ports(6), window_root=window)
    segment = coins.commit(bytes.fromhex(reg.digest))
    draw = stream.select(reg)
    raw = segment.to_json()
    client.close()
    assert host.ended() is None
    coins.settle()
    assert len(client.opened) > test_remote.RELEASED and segment.start == 0 and stream.family(draw) == "positions"
    assert len(R.release_frames(client.commitment, list(client.opened.values()))) > 1
    read = stream._entries[reg.order].read
    evidence = {"key": test_remote.pinned_key().to_json(), "ledger": R.ledger_lines(host.ledger.path),
                "main": test_remote.MAIN, "pins": test_remote.HOST_PINS, "segment": segment.to_json(),
                "session": R.session_files(host.directory)}
    after = bytes.fromhex(reg.digest)
    assert T.coins_reason(evidence, window, "positions", after, read, lms_verify) is None
    assert "coin opening is malformed" in T.coins_reason({**evidence, "segment": raw}, window, "positions", after, read,
                                                         lms_verify)
    assert stream.deliver(draw, [SESSION]) is None
    out = stream.outcome(draw, _lean(), {"statement": "stub"}, coins=evidence)
    assert out.accepted and "run" not in stream.records[-1]["audit"]["checks"], stream.records[-1]["audit"]["checks"]


class Redraws(T.ChallengerCoins):
    """A prover's own client: it commits again in a family whose coins it has read, each segment starting where the
    family's coins stand, which `ChallengerCoins` refuses to."""

    def commit(self, after: bytes) -> T.Segment:
        roots, self.record.roots = self.record.roots, []
        try:
            segment = super().commit(after)
        finally:
            self.record.roots[:0] = roots
        segment.start = len(roots)
        return segment


def test_a_registration_made_after_an_earlier_ones_sample_is_read_is_refused_for_its_family(
        tmp_path, signer, monkeypatch):
    """A second registration in one family of one window's withheld session is refused by both verifiers. The
    session's client (`Redraws`) registers six proof blocks, commits the family's coins to that registration's digest
    and reads its sample: a withheld session answers each open with the coin's raw bytes, so the client holds the
    sample before it sends anything more. It then registers the same six proof blocks again, each time on a stream of
    its own with an empty log and under roots of other bytes (a fresh commitment's), so under another digest, and
    commits the family's next coins to that digest, until a sample leaves out a proof block of the first one. The
    verifier is handed the last registration alone: its record, its segment and the session's signed files, whose log
    lists every earlier digest's coins before its own. The coins after the last digest are consecutive in the family
    and all read, and the family's log shows the earlier roots: `coins_reason` refuses for that (`FAMILY_ROOT`), and
    the outcome is a reject whose run the Lean (`Run.run`) refused at its select under the code `family`. So is the
    first registration's, whose coins are the family's first: the family holds later roots. One-stage v2's read of the
    same log at the parameters' family (`draw_coins`, `OneStage.coinsOf`'s) refuses every one of them alike. The
    honest client takes no second commitment in the family (`ChallengerCoins.commit`)."""
    import test_remote
    from test_challenger import lms_verify
    from test_service import SESSION, _lean, _ports, _record, _stream
    from verity.commitments import coin_tree as CT

    def record(i: int) -> dict:
        return _record(roots=[{"port": p, "root": hashlib.sha512(f"{i}/{p}".encode()).hexdigest(), **layout}
                              for p, layout in _ports(6).items()])

    monkeypatch.setattr(test_remote, "FAMILIES", [("positions", 64), ("replay", 2)])
    host = Host(tmp_path, signer, withhold=True)
    client = R.RemoteChallenger.connect(host.address)
    coins = Redraws(C.LiveRecord(client, "positions", itertools.count().__next__))
    window = test_remote.WINDOW["root"]
    tries = []
    for i in range(16):
        stream = _stream(coins=coins)
        reg = stream.register(record(i), _ports(6), window_root=window)
        segment = coins.commit(bytes.fromhex(reg.digest))
        tries.append((stream, reg, segment, stream.select(reg)))
        if i and tries[0][3].units[0] not in tries[-1][3].units:
            break
    with pytest.raises(C.CoinError, match="follow one commitment, drawn once"):
        T.ChallengerCoins(coins.record).commit(bytes(64))
    _, tree, _ = client.tree(test_remote.TREE_SPEC, test_remote.NONCE)
    client.tree_open(tree, CT.SESSION, 0, 2, C.link_after(test_remote.ROOT_F, test_remote.ROOT_B))
    client.tree_open(tree, "t/rep0", 0, 3, test_remote.root(30))
    client.close()
    assert host.ended() is None
    coins.settle()
    stream, reg, segment, draw = tries[-1]
    assert len(tries) > 1 and tries[0][3].units[0] not in draw.units, [t[3].units for t in tries]
    assert len({t[1].digest for t in tries}) == len(tries) and all(len(t[0]._entries) == 1 for t in tries)
    _, roots = host.trusted()
    assert list(dict.fromkeys(r.hex() for r in roots["positions"])) == [t[1].digest for t in tries]
    assert segment.start == sum(len(t[2].openings) for t in tries[:-1]) > 0 == tries[0][2].start
    evidence = {"key": test_remote.pinned_key().to_json(), "ledger": R.ledger_lines(host.ledger.path),
                "main": test_remote.MAIN, "pins": test_remote.HOST_PINS, "segment": segment.to_json(),
                "session": R.session_files(host.directory)}
    mine = {k: v for k, v in evidence.items() if k != "key"}
    for t in tries:
        after, read = bytes.fromhex(t[1].digest), t[0]._entries[t[1].order].read
        assert t[0].family(t[3]) == "positions"
        assert T.coins_reason({**evidence, "segment": t[2].to_json()}, window, "positions", after, read,
                              lms_verify) == T.FAMILY_ROOT
        with pytest.raises(C.CoinError, match=re.escape(T.FAMILY_ROOT)):
            T.draw_coins(test_remote.pinned_key(), window, "positions", after, {**mine, "segment": t[2].to_json()},
                         lms_verify)
    for t in (tries[0], tries[-1]):
        assert t[0].deliver(t[3], [SESSION]) is None
        out = t[0].outcome(t[3], _lean(), {"statement": "stub"}, coins={**evidence, "segment": t[2].to_json()})
        assert not out.accepted and out.verdict.detail["reject"]["checks"] == ["run"], out.verdict
        lean = t[0].records[-1]["audit"]["checks"]["run"]["detail"]["disagree"]["lean"]
        assert (lean["step"], lean["codes"]) == ("select", ["family"]), lean


def _salt_coins(challenger: C.LiveChallenger | None = None) -> tuple[C.LiveChallenger, T.ChallengerCoins]:
    ch = challenger or C.LiveChallenger(SCHEME, 0, [(T.SEED_SALT_FAMILY, 2), *FAMILIES], entropy("salt"))
    return ch, T.ChallengerCoins(C.LiveRecord(ch, T.SEED_SALT_FAMILY, itertools.count().__next__))


def test_a_windows_seed_salt_is_its_familys_first_coin_after_the_open_root_drawn_once():
    """The seed salt is the seed-salt family's first coin, opened after the window's open root and read whole, with no
    derivation, which the verifier replays from the record. The draw refuses coins of another family, a window root in
    another form, and a family with a coin open already: nothing redraws a window's salt."""
    window = "a1" * 64
    ch, coins = _salt_coins()
    salt = T.draw_seed_salt(coins, window)
    (segment,) = coins.segments
    assert salt == segment.openings[0].coin and len(salt) == C.COIN_BYTES
    assert (segment.label, segment.start, segment.after.hex(), len(segment.openings)) == (T.SEED_SALT_FAMILY, 0, window, 1)
    assert coins.record.roots == [bytes.fromhex(window)] and coins.record.reason() is None
    assert T.challenger_stream(ch.commitment, segment.to_json(), bytes.fromhex(window), coins.record.roots) == salt
    with pytest.raises(C.CoinError, match="drawn once"):
        T.draw_seed_salt(coins, window)
    with pytest.raises(C.CoinError, match="drawn once"):
        T.draw_seed_salt(T.ChallengerCoins(coins.record), "22" * 64)
    _, draw = _coins(ch)
    with pytest.raises(C.CoinError, match="seed-salt family"):
        T.draw_seed_salt(draw, window)
    for bad in (window.upper(), window[:-2], window + "00", "", None, bytes.fromhex(window)):
        with pytest.raises(C.CoinError, match=re.escape(T.SEED_SALT_WINDOW)):
            T.draw_seed_salt(_salt_coins()[1], bad)
    assert draw.record.roots == []


def test_a_windows_seed_salt_is_read_back_from_its_challengers_log(tmp_path, signer, monkeypatch):  # noqa: F811
    """Over a challenger served on loopback for the window, withheld: the stream draws the window's seed salt at its
    open (`Stream.seed_salt`), and the verifier reads the same salt back from the challenger's log under its own key
    (`coins.seed_salt_of`: the session the auditor's ledger opens for the window, its one seed-salt coin after the
    window's open root). A session whose challenger released a second coin after the root (a server picking its salt
    between two) is refused, read alone or both, and so are a forged coin, another family's coins and another window's."""
    import test_remote
    from test_challenger import lms_verify
    from test_service import _stream

    monkeypatch.setattr(test_remote, "FAMILIES", [(T.SEED_SALT_FAMILY, 2), ("positions", 2)])
    window, other = test_remote.WINDOW["root"], "22" * 64
    host = Host(tmp_path, signer, withhold=True)
    alt = Host(tmp_path, signer, session="s-2", seed="alt", ledger=host.ledger, window={"root": other}, withhold=True)
    stream, salts, segments, second = _stream(), {}, {}, None
    for h, root in ((host, window), (alt, other)):
        client = R.RemoteChallenger.connect(h.address)
        coins = T.ChallengerCoins(C.LiveRecord(client, T.SEED_SALT_FAMILY, itertools.count().__next__))
        salts[root] = stream.seed_salt(coins, root)
        if h is alt:
            coins.record.record(bytes.fromhex(root))
        client.close()
        coins.settle()
        assert h.ended() is None
        segments[root] = coins.segments[0].to_json()
        second = client.opened.get((T.SEED_SALT_FAMILY, 1), second)
    assert [e["window_root"] for e in stream.epoch_log] == [window, other]
    ledger = R.ledger_lines(host.ledger.path)
    files = {window: R.session_files(host.directory), other: R.session_files(alt.directory)}

    def read(root: str, segment: dict | None = None, at: str | None = None) -> bytes | str:
        """The seed salt the verifier reads on window `at` (`root`'s, unless given) from `root`'s session's evidence."""
        evidence = {"ledger": ledger, "main": test_remote.MAIN, "pins": test_remote.HOST_PINS,
                    "segment": segments[root] if segment is None else segment, "session": files[root]}
        try:
            return T.seed_salt_of(test_remote.pinned_key(), at or root, evidence, lms_verify)
        except C.CoinError as e:
            return str(e)

    assert read(window) == salts[window] and len(salts[window]) == C.COIN_BYTES
    assert "not the ones the challenger's log shows" in read(other)
    assert second is not None and read(other, {**segments[other], "openings": [*segments[other]["openings"],
                                                                                second.to_json()]}) == T.SEED_SALT_ONE
    forged = {**segments[window], "openings": [{**segments[window]["openings"][0], "coin": "5a" * 64}]}
    assert "does not open" in read(window, forged)
    assert "family's" in read(window, {**segments[window], "label": "positions"})
    assert isinstance(read(window, at=other), str) and isinstance(read(window, at="33" * 64), str)
