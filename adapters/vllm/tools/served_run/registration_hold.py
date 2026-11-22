"""The launcher's hold between the deployment's registration and its first request: with REGISTRATION_LANE it serves only
once the auditor's challenger has signed that registration for the lane (`challenger.registration_head`), and with none it
says first that no one signed it (the development path).

    registration_hold.py hold --registration DREG [--lane LANE --key KEY (--gate CMD | --ledger PATH)]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from verity.commitments import docket_digest as DD
from verity.commitments.live_coin import CoinError
from verity.protocols.accounting.compute.pouw.prover.serving.coins import window as PHW
from verity.protocols.proof.spot.verifier.challenger import challenger as Ch

DEVELOPMENT = ("REGISTRATION signed by no one: the development path (no REGISTRATION_LANE), so no challenger holds this "
               "run's epochs to the registration")
#: the refusals, each before any request
ALREADY_REGISTERED = ("REGISTRATION REFUSED already-registered: the lane already carries a registration, and a lane takes "
                      "one, ever; ours would be a second (SECOND_REGISTRATION): name a fresh REGISTRATION_LANE")
OTHER_DIGEST = ("REGISTRATION REFUSED other-digest: the challenger signed another registration for the lane than the one "
                "this serve wrote")
SECOND = "REGISTRATION REFUSED second-registration: the lane's ledger signs more than one registration"
PERIOD_CLOSED = ("REGISTRATION REFUSED period-closed: the period whose open signed this registration is not the lane's "
                 "open period, so no epoch can join it")
#: seconds between reads of the ledger while it signs no registration for the lane, the tied close's interval
POLL_S = 5


class Refused(Exception):
    pass


def digest_of(dreg: Path) -> str:
    return DD.digest(json.loads(Path(dreg).read_text()))


def period_open_line(lane: str, digest: str, circuit: str | None) -> str:
    c = circuit or "<SHA-512 of the period's circuit, which the auditor names: not known before the serve>"
    return (f"vy-challenger-gate period-open --name <PERIOD> --lane {lane} --cap <EPOCHS> --circuit {c} "
            f"--registration {digest}")


def hold(digest: str, lane: str, read: Callable[[], list[str]], head: Callable[[list[str]], str],
         wait: Callable[[], Any]) -> dict[str, Any]:
    """The lane's open period's ``period-open`` body once `head` (the lane's one signed registration in the ledger `read`
    gives, `challenger.registration_head`) is `digest`, read again after each `wait` while it signs none; `Refused`
    naming why otherwise. A registration on the first read that isn't ours was there before ours was written."""
    first = True
    while True:
        try:
            lines = read()
            got = head(lines)
        except CoinError as e:
            if str(e) == Ch.NO_REGISTRATION:
                got = None
            elif str(e).startswith(Ch.SECOND_REGISTRATION):
                raise Refused(f"{SECOND}: {e}") from None
            else:
                raise Refused(f"REGISTRATION REFUSED ledger: {e}") from None
        if got is not None:
            if got != digest:
                raise Refused(f"{ALREADY_REGISTERED if first else OTHER_DIGEST} ({got[:16]}..., ours {digest[:16]}...)")
            opened = PHW.open_period(lines, lane)
            if opened is None or opened.get("registration") != digest:
                raise Refused(PERIOD_CLOSED)
            return opened
        first = False
        wait()


def _reader(a: argparse.Namespace) -> Callable[[], list[str]]:
    if a.gate:
        from verity_vllm.pipeline.commit_coins import ChallengerGate
        return ChallengerGate(a.gate).ledger
    from verity.protocols.proof.spot.verifier.challenger import remote as SPR
    return lambda: SPR.ledger_lines(a.ledger)


def main(argv: Sequence[str] | None = None, *, read: Callable[[], list[str]] | None = None,
         head: Callable[[list[str]], str] | None = None, wait: Callable[[], Any] | None = None) -> int:
    p = argparse.ArgumentParser(prog="registration_hold.py")
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hold")
    h.add_argument("--registration", required=True, help="the serve's deployment-registration.json")
    h.add_argument("--lane", default="", help="REGISTRATION_LANE; empty: the development path")
    h.add_argument("--key", default="", help="the challenger's key record (REGISTRATION_KEY, else PP_CHALLENGER_KEY)")
    h.add_argument("--gate", default="", help="the challenger gate's command, whose `ledger` answers the auditor's ledger")
    h.add_argument("--ledger", default="", help="else the auditor's ledger as a file (REGISTRATION_LEDGER)")
    h.add_argument("--circuit", default="", help="the period's circuit digest, where it is known before the serve")
    a = p.parse_args(argv)
    if not a.lane:
        print(DEVELOPMENT, flush=True)
        return 0
    if head is None and not a.key or read is None and not (a.gate or a.ledger):
        print("REGISTRATION REFUSED preflight: REGISTRATION_LANE names a lane, so the hold reads the challenger's ledger "
              "(PP_CHALLENGER=gate:... or REGISTRATION_LEDGER) under its key record (REGISTRATION_KEY or "
              "PP_CHALLENGER_KEY); one is unset", flush=True)
        return 2
    digest = digest_of(Path(a.registration))
    print(f"REGISTRATION {digest} on lane {a.lane}; the epoch's circuit "
          + (a.circuit if a.circuit else "is not known before the serve (window-commit writes circuit.json after it)"),
          flush=True)
    print(f"REGISTRATION HOLD: the first request waits for this on the challenger's host: "
          f"{period_open_line(a.lane, digest, a.circuit or None)}", flush=True)
    if head is None:
        from verity.protocols.proof.spot.verifier.challenger import challenger_session as CS
        key = Ch.key_of_record(json.loads(Path(a.key).read_text()))
        head = lambda lines: Ch.registration_head(lines, a.lane, key, CS.lms_verify)  # noqa: E731
    try:
        opened = hold(digest, a.lane, read or _reader(a), head, wait or (lambda: time.sleep(POLL_S)))
    except Refused as e:
        print(e, flush=True)
        return 3
    print(f"REGISTRATION SIGNED: period {opened.get('period')} of lane {a.lane} signs it; serving", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
