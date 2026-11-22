"""The auditor's ledger kept current on the serving node: each look fetches it from the challenger gate's ``ledger`` and
appends its new lines to OUT, which `served_verdicts.py registration-hold` follows and tests under the pinned key.

    ledger_fetch.py --gate CMD --out PATH
"""
from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

#: seconds between fetches, the tied close's interval between reads of the ledger
POLL_S = 5


class Refused(Exception):
    pass


def append_new(fetched: list[str], out: Path) -> int:
    """`fetched`'s lines past those `out` holds, appended; `Refused` where `out` is not a prefix of it (the gate's ledger
    is append-only, so a copy that isn't is not the gate's). A shorter fetch appends nothing. -> lines appended."""
    held = out.read_text().splitlines() if out.exists() else []
    if fetched[:len(held)] != held[:len(fetched)]:
        raise Refused(f"LEDGER-FETCH refused: {out} is not a prefix of the gate's ledger")
    new = fetched[len(held):]
    if new:
        with out.open("a") as f:
            f.write("".join(x + "\n" for x in new))
    return len(new)


def follow(fetch: Callable[[], list[str]], out: Path, wait: Callable[[], Any], rounds: int | None = None) -> None:
    """`append_new` of each `fetch`, `wait` between, for `rounds` looks or until killed."""
    out.touch()
    n = 0
    while rounds is None or n < rounds:
        append_new(fetch(), out)
        n += 1
        if rounds is None or n < rounds:
            wait()


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ledger_fetch.py")
    p.add_argument("--gate", required=True, help="the challenger gate's command (ssh ... research@HOST)")
    p.add_argument("--out", required=True, type=Path, help="REGISTRATION_LEDGER")
    a = p.parse_args(argv)
    from verity_vllm.pipeline.commit_coins import ChallengerGate
    try:
        follow(ChallengerGate(a.gate).ledger, a.out, lambda: time.sleep(POLL_S))
    except Refused as e:
        print(e, flush=True)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
