"""The served C2 replay's challenge from the auditor host's challenger (RT4-INT-21): its ``replay`` coin 0, opened after
the Commit's run root (`after`), in a session of a challenger started over `SCHEME` with a ``replay`` family. Without
the Commit's ``--replay-challenger`` the challenge is ``--replay-challenge`` or one this process draws, and the record
says so (``source``); only a ``challenger`` source is the auditor's draw.

`reason` is the verifier's check of the record against the commitment the challenger signed (read on the auditor's side,
`pouw_h1_challenger.trusted_rounds_reason`'s way): the coin is replay coin 0 of that commitment, opened after this run
root, and the challenge is the coin.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

from verity.core.protocols.sampled_proofs import remote
from verity.core.protocols.sampled_proofs.challenger import CoinError, LiveCommitment, LiveOpening, live_reason

SCHEME = "verity/vllm/commit-coins/v1"
FAMILY = "replay"
TAG = b"verity/vllm/replay-challenge/v1\0"


def after(run_root: bytes) -> bytes:
    """The 64-byte root the replay coin opens after: the Commit's run root under `TAG`."""
    return hashlib.sha512(TAG + bytes(run_root)).digest()


def challenge(at: str, run_root: bytes) -> tuple[str, dict[str, Any]]:
    """The replay challenge (the coin, hex) from the challenger at `at`, and its record; ValueError (a challenger of another
    layout) or `CoinError` (the challenger's refusal), with the session closed either way."""
    client = remote.RemoteChallenger.connect(at)
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
    return opening.coin.hex(), {"source": "challenger", "at": at, "commitment": c.to_json(), "opening": opening.to_json()}


def local(given: str | None, drawn: str) -> tuple[str, dict[str, Any]]:
    """The challenge without a challenger: ``--replay-challenge`` (`given`) or the one this process drew."""
    return (given, {"source": "flag"}) if given else (drawn, {"source": "in-process"})


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


__all__ = ["FAMILY", "SCHEME", "TAG", "after", "challenge", "local", "reason"]
