"""Pearl+'s verifier's coin from a live draw session (`pearl_c.draw`): reserved withheld through infra's gate with the
draw coin alone, opened on the run's commitment, its coin released after that root, so the coin record it writes holds
to the auditor's ledger and the session's own files (`draw_reason`). A second draw on one commitment is the gate's
refusal, a gate that reserves the session plain is refused before any open, and a record whose session, lane, layout,
end, opening or source isn't the ledger's is refused, naming why.

CPU only: the gate and the challenger are `test_commit_coins._Auditor`'s, in this process.
"""
from __future__ import annotations

import hashlib
import shlex

import pytest
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs.challenger import CoinError
from verity_vllm.pipeline import commit_coins as CC
from verity_vllm.protocols.pouw.h1 import challenger as PHC
from verity_vllm.protocols.pouw.pearl_c import draw as PD

from tests.pipeline.test_commit_coins import GATE, KEY, PEER, _Auditor
from tests.protocols import challenger_host as CH

ROOT = hashlib.sha512(b"Pearl+'s run commitment").digest()


def _drawn(auditor: _Auditor, root: bytes = ROOT) -> dict:
    """The coin record `served_passes.py verify --draw-gate` writes over a run whose commitment is `root` (the fields
    `draw_reason` reads), its coin a draw session's on `auditor`'s gate, the session ended."""
    coin, drawn = PD.draw(CC.ChallengerGate(GATE, auditor.run).call, PEER, root)
    auditor.end(drawn["session"]["session"])
    return {"commitment": root.hex(), "source": coin.hex(), "draw": drawn}


def _reason(auditor: _Auditor, record: dict, lane: str = "vllm", trees: bool = True) -> str | None:
    """`draw_reason` over `auditor`'s ledger as infra publishes it, each session's tree where the gate's fetch gives it
    (none without `trees`)."""
    ledger = PHC.Ledger.load(auditor.heads(), KEY, pins=[CH.write_pins(auditor.tmp / "pins.jsonl")], main=list(CH.MAIN))
    return PD.draw_reason(record, ledger, lambda s: auditor.tmp / s if trees and s in auditor.sessions else None, lane)


def test_a_draw_on_the_runs_commitment_is_its_draw_sessions_coin(tmp_path, monkeypatch):
    """`draw` reserves a draw session withheld with the draw coin alone as this peer, the gate opens it withheld on the
    run's commitment, and its coin, released after that root, is the source; the record carries the session without its
    hello secret, its reserve, the coin's opening and the commitment its challenger released at its close, and
    `draw_reason` finds it the ledger's."""
    auditor = _Auditor(tmp_path, monkeypatch)
    record = _drawn(auditor)
    sid = record["draw"]["session"]["session"]
    assert _reason(auditor, record) is None
    words = [shlex.split(a[-1]) for a in auditor.argv]
    assert words == [["reserve", "--peer", PEER, "--scheme", PHC.SCHEME, "--families", PHC.draw_families(), "--withhold"],
                     ["open", "--session", sid, "--root", ROOT.hex()]]
    assert auditor.gate._opens[sid]["withheld"] is True and auditor.sessions[sid]["root"] == ROOT.hex()
    assert "secret" not in record["draw"]["session"] and record["draw"]["reserved"]["session"] == sid
    commitment = Ch.LiveCommitment.from_json(record["draw"]["commitment"])
    opening = Ch.LiveOpening.from_json(record["draw"]["opening"])
    assert Wd.draw_epoch_reason(commitment) is None and commitment.root.hex() == auditor.sessions[sid]["commitment"]
    assert Ch.live_reason(commitment, opening, Wd.DRAW, 0, ROOT) is None and opening.coin.hex() == record["source"]


def test_a_second_draw_on_one_commitment_and_a_plain_reserve_are_refused(tmp_path, monkeypatch):
    """The gate opens one session on a root, so a second draw on the run's commitment (a retry, or a prover grinding the
    coin) is the gate's refusal before any coin; a gate that reserves the draw session plain is refused before its
    open."""
    auditor = _Auditor(tmp_path, monkeypatch)
    _drawn(auditor)
    with pytest.raises(CoinError, match="the challenger gate refused open"):
        PD.draw(CC.ChallengerGate(GATE, auditor.run).call, PEER, ROOT)
    plain = CC.ChallengerGate(GATE, lambda argv: auditor.run([*argv[:-1], argv[-1].replace(" --withhold", "")]))
    with pytest.raises(CoinError, match=r"the gate reserved draw session \S+ plain, not withheld"):
        PD.draw(plain.call, PEER, hashlib.sha512(b"another run").digest())
    assert [shlex.split(a[-1])[0] for a in auditor.argv] == ["reserve", "open", "reserve", "open", "reserve"]


def test_a_coin_record_that_isnt_its_draw_sessions_is_refused(tmp_path, monkeypatch):
    """`draw_reason` holds the record to the ledger, never its word: a record drawn on the node (no draw), naming its
    hello secret or another session, over a commitment the ledger opens no session on, with an opening or a source that
    isn't its session's coin, read for another lane or with no tree of the session, is refused, naming why."""
    auditor = _Auditor(tmp_path, monkeypatch)
    record = _drawn(auditor)
    draw, sid = record["draw"], record["draw"]["session"]["session"]
    other = _drawn(auditor, hashlib.sha512(b"another run").digest())
    for given, lane, trees, why in (
            ({k: v for k, v in record.items() if k != "draw"}, "vllm", True, PD.NO_DRAW),
            ({**record, "commitment": ROOT.hex().upper()}, "vllm", True, "the coin record's run commitment is not 128 "
                                                                         "lowercase hex"),
            ({**record, "draw": {**draw, "session": {**draw["session"], "secret": "ab" * 32}}}, "vllm", True,
             "the coin record names its draw session's hello secret, which a record never carries"),
            ({**record, "draw": other["draw"]}, "vllm", True, f"the coin record's draw session is not {sid}, the one the "
                                                              "auditor's ledger opened on the run commitment"),
            ({**record, "commitment": "ab" * 64}, "vllm", True, "the auditor's ledger opens no session on the run "
                                                                "commitment, so the coin is no draw session's"),
            (record, "other", True, f"draw session {sid} was not reserved for the lane 'other': 'vllm'"),
            (record, "vllm", False, f"draw session {sid} is not trusted: session {sid} has no tree on the auditor's side"),
            ({**record, "draw": {**draw, "opening": {**draw["opening"], "coin": other["source"]}}}, "vllm", True,
             "the verifier's coin: "),
            ({**record, "source": other["source"]}, "vllm", True, "the coin record's source is not its draw session's coin")):
        got = _reason(auditor, given, lane, trees)
        assert got is not None and got.startswith(why), (why, got)


@pytest.mark.parametrize("families, withheld, why", [
    (Wd.SALT_FAMILIES, True, f"was reserved with [[{Wd.SALT!r}, 1]], not the draw coin alone"),
    (Wd.DRAW_FAMILIES, False, Ch.WITHHELD),
    (Wd.DRAW_FAMILIES, True, "never ended with its client closing it"),
])
def test_a_session_on_the_commitment_that_isnt_a_closed_withheld_draw_session_is_refused(tmp_path, monkeypatch, families,
                                                                                         withheld, why):
    """The session the ledger opens on the run's commitment is a draw session only if it was reserved with the draw coin
    alone, opened withheld and ended closed, whatever the record says."""
    auditor = _Auditor(tmp_path, monkeypatch)
    sid, root = CH.session_id(1), hashlib.sha512(b"a run").hexdigest()
    auditor.gate.reserve(sid, families, **({"withheld": True} if withheld else {}))
    auditor.gate.open(sid, root)
    got = _reason(auditor, {"commitment": root, "source": "5e" * 32, "draw": {"session": {"session": sid}}})
    assert got == f"draw session {sid}" + (f": {why}" if why == Ch.WITHHELD else f" {why}"), got
