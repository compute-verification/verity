"""Pearl+'s verifier's coin from a draw session of its own (`window.DRAW_FAMILIES`, the draw coin alone) on the auditor
host's challenger, reserved withheld through infra's gate under the lane's gate key (the ledger's reserve names the lane)
and opened on the run's commitment (`verify_run.run_commitment`, over every retained pass's roots and salt), so its
challenger releases the coin only after that root, and the gate opens no second session on it: `draw` is the
verifier's side, called once the passes are fixed, and `draw_reason` the check of what the coin record carries, against
the auditor's ledger and the session's own files, never the record's word."""
from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping
from typing import Any

from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs.challenger import CoinError, LiveOpening, live_reason

from verity_vllm.protocols.pouw.h1 import challenger as PHC
from verity_vllm.protocols.pouw.h1 import window as PHW

#: the entry the draw session's refusals name
ENTRY = "pouw:pearl-plus"
#: where the auditor host's challenger reaches this node: its 127.0.0.1, at the port the gate's open answers
AT = "127.0.0.1:0"
#: why a coin record has no draw session to be checked against
NO_DRAW = ("the verifier's coin was drawn on the node, not from a live draw session reserved on the gate under the lane's "
           "key (served_passes.py verify --draw-gate --draw-peer)")
_ROOT = re.compile(r"^[0-9a-f]{128}$")


def draw(call: Callable[..., Mapping[str, Any]], peer: str, root: bytes,
         sleep: Callable[[float], Any] = time.sleep) -> tuple[bytes, dict[str, Any]]:
    """The coin of a draw session reserved now at infra's gate, as words (`call`, `commit_coins.ChallengerGate.call` over
    its ssh command under the lane's key), as peer `peer`, withheld with the draw coin alone (`pouw_h1_window.reserve_draw`),
    and opened on `root` (`pouw_h1_window.DrawSession`, once: a retry never redraws), and what a coin record carries of
    it: ``session`` (the gate's open answer without its secret), ``reserved`` (its reserve answer), ``opening`` (the
    coin's opening) and ``commitment`` (the coin commitment the session released at its close). `CoinError` when the gate
    or the challenger refuses, or the gate reserves the session plain."""
    reserved = PHW.reserve_draw(call, peer)
    session = PHW.DrawSession(ENTRY, AT, reserved, PHW.gate_of(call), sleep)
    opening = session.record(root)
    return opening.coin, {"session": session.session, "reserved": reserved, "opening": opening.to_json(),
                          "commitment": session.commitment.to_json()}


def draw_reason(record: Mapping[str, Any], ledger: Any, fetch: PHC.Fetch, lane: str,
                verify: Any = PHC.lms_verify) -> str | None:
    """Why the verifier's coin record `record` (`verify_run`'s ``coin.json``, its ``draw`` `draw`'s) isn't its draw
    session's: the one session the auditor's `ledger` (`pouw_h1_challenger.Ledger`) opens on the record's run commitment,
    reserved for `lane` with the draw coin alone and opened withheld (`challenger.withheld_on`), ended closed having
    released that coin and nothing else, its files as `fetch` gives them (`pouw_h1_challenger.evidence`): its signed
    commitment a draw session's (`window.draw_epoch_reason`), its signed log showing the run commitment alone as the
    root its coin followed, the record's ``opening`` that coin 0 opened after it (`challenger.live_reason`) and its
    ``source`` that coin. None when it is; fails closed."""
    root, got = record.get("commitment"), record.get("draw")
    if not (isinstance(root, str) and _ROOT.match(root)):
        return "the coin record's run commitment is not 128 lowercase hex"
    if not isinstance(got, Mapping):
        return NO_DRAW
    if isinstance(got.get("session"), Mapping) and "secret" in got["session"]:
        return "the coin record names its draw session's hello secret, which a record never carries"
    window = ledger.window(root)
    if window is None:
        return "the auditor's ledger opens no session on the run commitment, so the coin is no draw session's"
    sid, reserve, end = window["open"]["session"], window["reserve"], window["end"]
    if (got.get("session") or {}).get("session") != sid:
        return f"the coin record's draw session is not {sid}, the one the auditor's ledger opened on the run commitment"
    if reserve is None or Ch.lane_of(reserve) != lane:
        return f"draw session {sid} was not reserved for the lane {lane!r}: {None if reserve is None else Ch.lane_of(reserve)!r}"
    if json.dumps(reserve["body"].get("families")) != json.dumps([list(f) for f in Wd.DRAW_FAMILIES]):
        return f"draw session {sid} was reserved with {reserve['body'].get('families')}, not the draw coin alone"
    if not Ch.withheld_on(list(ledger.lines), root):
        return f"draw session {sid}: {Ch.WITHHELD}"
    if end is None or end["body"].get("closed") is not True:
        return f"draw session {sid} never ended with its client closing it"
    if json.dumps(end["body"].get("released"), sort_keys=True) != json.dumps({Wd.DRAW: 1}):
        return f"draw session {sid} released {end['body'].get('released')}, not its one draw coin"
    try:
        _, trusted, received = PHC.evidence(ledger, root, fetch, verify)
    except CoinError as e:
        return f"draw session {sid} is not trusted: {e}"
    if (why := Wd.draw_epoch_reason(trusted)) is not None:
        return f"draw session {sid}'s signed coins: {why}"
    if dict(received) != {Wd.DRAW: (bytes.fromhex(root),)}:
        return f"draw session {sid}'s signed log does not show the run commitment alone as the root its coin followed"
    try:
        opening = LiveOpening.from_json(got.get("opening"))
    except CoinError as e:
        return f"the coin record's opening is malformed ({e})"
    if (why := live_reason(trusted, opening, Wd.DRAW, 0, bytes.fromhex(root))) is not None:
        return f"the verifier's coin: {why}"
    if opening.coin.hex() != record.get("source"):
        return "the coin record's source is not its draw session's coin"
    return None


__all__ = ["AT", "ENTRY", "NO_DRAW", "draw", "draw_reason"]
