"""The served C2 replay's challenge from the auditor host's challenger (RT4-INT-21): its ``replay`` coin 0, opened after
the Commit's run root (`after`), in a session of a challenger started over `SCHEME` with a ``replay`` family. Without
the Commit's ``--replay-challenger`` the challenge is ``--replay-challenge`` or one this process draws, and the record
says so (``source``); only a ``challenger`` source is the auditor's draw.

The Commit carries its session, the challenger gate's reply (``--replay-session``, `pouw_h1_challenger.session_of`), in
the record. `reason` is the verifier's check of the record against the commitment the challenger signed (read on the
auditor's side, `pouw_h1_challenger.trusted_rounds_reason`'s way): the coin is replay coin 0 of that commitment, opened
after this run root, and the challenge is the coin. `session_reason` first holds the record's session to the one the
auditor's ledger opened for its Commit root (`pouw_h1_challenger.held`), and reads that commitment from the session's
sealed tree.
"""
from __future__ import annotations

import hashlib
import json
import secrets
from typing import Any, Mapping

from verity.core.protocols.sampled_proofs import remote
from verity.core.protocols.sampled_proofs.challenger import (CoinError, LiveCommitment, LiveOpening, live_reason,
                                                             trusted_commitment)

from verity_vllm.protocol_options import pouw_h1_challenger as PHC

SCHEME = "verity/vllm/commit-coins/v1"
FAMILY = "replay"
TAG = b"verity/vllm/replay-challenge/v1\0"


def after(run_root: bytes) -> bytes:
    """The 64-byte root the replay coin opens after: the Commit's run root under `TAG`."""
    return hashlib.sha512(TAG + bytes(run_root)).digest()


def challenge(at: str, run_root: bytes, session: Any = None) -> tuple[str, dict[str, Any]]:
    """The replay challenge (the coin, hex) from the challenger at `at`, in the session the gate's answer `session` names
    (`pouw_h1_challenger.session_of`; its hello carries the answer's secret, and a challenger of another commitment is
    refused before the coin, `pouw_h1_challenger.connect`), and its record, which carries the session without its secret;
    ValueError (a malformed session, another commitment than the answer's or a challenger of another layout) or
    `CoinError` (the challenger's refusal), with the session closed either way."""
    given = PHC.session_of(session)
    client = remote.RemoteChallenger.connect(at) if given is None else PHC.connect(at, given)
    try:
        c = client.commitment
        if c.scheme != SCHEME or not c.family(FAMILY):
            raise ValueError(f"the challenger at {at} is {c.scheme} with families {list(c.families)}, not {SCHEME} with a "
                             f"{FAMILY} family")
        opening = client.open(FAMILY, 0, after(run_root))
    finally:
        try:
            client.close()
        except CoinError:
            pass
    return opening.coin.hex(), {"source": "challenger", "at": at, "session": given and PHC.carried(given),
                                "commitment": c.to_json(), "opening": opening.to_json()}


def local(given: str | None, drawn: str) -> tuple[str, dict[str, Any]]:
    """The challenge without a challenger: ``--replay-challenge`` (`given`) or the one this process drew."""
    return (given, {"source": "flag"}) if given else (drawn, {"source": "in-process"})


def session_option(challenger: str | None, given: str | None, session: str | None) -> dict[str, Any] | None:
    """The Commit's ``--replay-session`` (`session`, the gate's answer as JSON) parsed, given its ``--replay-challenger``
    (`challenger`) and ``--replay-challenge`` (`given`); `SystemExit` when they don't go together."""
    if challenger and given:
        raise SystemExit("--replay-challenger and --replay-challenge are two sources of the C2 replay's challenge: give one")
    if bool(challenger) != bool(session):
        raise SystemExit("--replay-challenger and --replay-session go together: the challenger and the gate's answer for "
                         "the session it serves the Commit's replay coin in")
    if not session:
        return None
    try:
        return PHC.session_of(json.loads(session))
    except ValueError as e:
        raise SystemExit(f"--replay-session: {e}") from e


def drawn(challenger: str | None, run_root: bytes, session: Any, given: str | None) -> tuple[str, dict[str, Any]]:
    """The C2 replay's challenge and its record, drawn after `run_root`: the auditor host's replay coin (`challenge`)
    with a `challenger`, else `local` (``--replay-challenge`` `given`, or one this process draws now)."""
    if challenger:
        return challenge(challenger, run_root, session)
    return local(given, secrets.token_hex(16))


def reason(record: Mapping[str, Any], challenge_hex: str, run_root: bytes, trusted: LiveCommitment) -> str | None:
    """Why the replay challenge `challenge_hex` isn't replay coin 0 of `trusted` (the commitment the auditor's challenger
    signed) opened after `run_root`; None when it is."""
    if record.get("source") != "challenger":
        return f"the replay challenge's source is {record.get('source')!r}, not the auditor host's challenger"
    try:
        c, o = LiveCommitment.from_json(record.get("commitment")), LiveOpening.from_json(record.get("opening"))
    except CoinError as e:
        return f"the replay challenge's record is malformed ({e})"
    if c != trusted:
        return "the replay challenge's coins are not the commitment the auditor's challenger signed"
    try:
        why = live_reason(trusted, o, FAMILY, 0, after(run_root))
    except CoinError as e:
        why = str(e)
    if why is not None:
        return f"the replay coin: {why}"
    return None if o.coin.hex() == challenge_hex else "the replay challenge is not the replay coin"


def session_reason(record: Mapping[str, Any], challenge_hex: str, run_root: bytes, ledger: PHC.Ledger, fetch: PHC.Fetch,
                   commit: str | None = None) -> str | None:
    """Why the replay challenge isn't the replay coin of the one session the auditor's `ledger` opened for the record's
    Commit root (`commit`, the Commit root the verifier holds, when given), ended closed and complete under the record's
    commitment (`pouw_h1_challenger.held`), read from that session's tree as `fetch` gives it
    (`pouw_h1_challenger.evidence`); None when it is."""
    if record.get("source") != "challenger":
        return f"the replay challenge's source is {record.get('source')!r}, not the auditor host's challenger"
    why, window = PHC.held(record.get("session"), ledger, commit)
    if why is not None:
        return f"the replay challenge: {why}"
    try:
        out = PHC.evidence(window, fetch, ledger.key)
        trusted = trusted_commitment(out / remote.RECORD, ledger.key, PHC.lms_verify)
    except CoinError as e:
        return f"the replay challenge's session {window['open']['session']} is not trusted: {e}"
    return reason(record, challenge_hex, run_root, trusted)


__all__ = ["FAMILY", "SCHEME", "TAG", "after", "challenge", "drawn", "local", "reason", "session_option",
           "session_reason"]
