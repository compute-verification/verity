"""The lines of a prove path log that may leave the node: of each tagged result line, only the fields its reader reads
(`ALLOWED`), and of each bare error line, only its exception's class; every other line stays in the kept log.

    python3 public_log.py LOG > PUBLIC_LOG
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any

#: what `verity bench prove layers` (`proof_layers.read_prove_path`) reads of each tagged line, by key path, ``*`` any
#: key, True a scalar. ``population`` is held: on a served scope it counts the execution phase's tokens. ``leak``,
#: ``charge`` and ``advice_bits`` are held: they are the advice's length or counted against it, not its cap.
ALLOWED: dict[str, dict[str, Any]] = {
    "PROVE-PATH ": {"forged": True, "ok": True, "proof": True, "coins": True, "ranks": {"*": {
        "accepted": True, "draw_counts": {"*": True},
        "per_unit": {"*": dict.fromkeys(("units", "proof_bytes", "prove_s_per_unit", "lean_verify_s_per_unit"), True)}}}},
    "RECORDS ": {"ok": True, "accepted": True, "negatives_refused": True, "draw_counts": {"*": True}},
    "A5 ": {"audit": {"accepted": True, "verdicts": {"*": True}}},
}
#: a traceback's last line, which a forged arm's reader names its failure by
ERROR = re.compile(r"(\w+(Error|Exception)): ")
HELD = "held in the kept log"


def allowed(o: Any, spec: Any) -> Any:
    """``o`` with only what ``spec`` allows; a value of another shape than ``spec``'s is `HELD`."""
    if spec is True:
        return o if o is None or isinstance(o, (bool, int, float, str)) else HELD
    if not isinstance(o, dict):
        return HELD
    return {k: allowed(v, spec[k] if k in spec else spec["*"]) for k, v in o.items() if k in spec or "*" in spec}


def public_line(line: str) -> str | None:
    """``line`` as it may leave, or None. A tagged line whose body isn't JSON leaves as its tag and why it is held."""
    for tag, spec in ALLOWED.items():
        if line.startswith(tag):
            try:
                body = allowed(json.loads(line[len(tag):]), spec)
            except ValueError as e:
                body = {"held": f"not JSON ({type(e).__name__}): the line stays in the kept log"}
            return tag + json.dumps(body, sort_keys=True, default=str)
    m = ERROR.match(line)
    return f"{m.group(1)}: {HELD}" if m else None


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
