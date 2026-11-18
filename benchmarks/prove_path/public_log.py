"""The lines of a prove path log that may leave the node: its tagged result lines without the advice's length or any
figure computed from it (`HELD`), and its bare error lines; every other line stays in the kept log.

    python3 public_log.py LOG > PUBLIC_LOG
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any

#: the result lines `verity bench prove layers` (`proof_layers.read_prove_path`) reads
TAGS = ("PROVE-PATH ", "RECORDS ", "A5 ", "PROOF-EDITS ")
#: the advice's true length (`advice_bits`, `leak`'s count against it) and the charge computed from it: private, the cap
#: public, until the charge is the cap
HELD = frozenset({"advice_bits", "advice", "charge", "leak"})
#: a traceback's last line, which a forged arm's reader names its failure by
ERROR = re.compile(r"\w+(Error|Exception): ")


def held_out(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: held_out(v) for k, v in o.items() if k not in HELD}
    if isinstance(o, list):
        return [held_out(v) for v in o]
    return o


def public_line(line: str) -> str | None:
    """``line`` as it may leave, or None. A tagged line whose body isn't JSON is held."""
    for tag in TAGS:
        if line.startswith(tag):
            try:
                body = json.loads(line[len(tag):])
            except ValueError:
                return None
            return tag + json.dumps(held_out(body), sort_keys=True, default=str)
    return line if ERROR.match(line) else None


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    with open(argv[0], errors="replace") as f:
        for line in f:
            if (out := public_line(line.rstrip("\n"))) is not None:
                print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
