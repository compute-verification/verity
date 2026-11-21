"""The lines of a prove path log that may leave the node: of each tagged result line, only the fields its reader reads
(`ALLOWED`), and of each bare error line, only its exception's class; every other line stays in the kept log. With
``summaries``, an audit's summaries kept under KEPT as they may leave into OUT: only the fields their reader reads
(`SUMMARIES`); every other file of the audit stays in the keep.

    python3 public_log.py LOG > PUBLIC_LOG
    python3 public_log.py summaries KEPT OUT
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

#: what `verity bench prove layers` (`proof_layers.read_prove_path`) reads of each tagged line, by key path, ``*`` any
#: key, True a scalar. ``population`` is held: on a served scope it counts the execution phase's tokens. ``leak``,
#: ``charge`` and ``advice_bits`` are held: they are the advice's length or counted against it, not its cap.
#: ``draw_counts``, ``per_unit``'s ``units`` and its means over them are held: a template proved whole counts every unit
#: the compiler made of the advice, so the count grows with the advice and the means divide the public totals by it.
ALLOWED: dict[str, dict[str, Any]] = {
    "PROVE-PATH ": {"ok": True, "proof": True, "coins": True, "ranks": {"*": {
        "accepted": True, "per_unit": {"*": {"proof_bytes": True}}}}},
    "RECORDS ": {"ok": True, "accepted": True, "negatives_refused": True},
    "A5 ": {"audit": {"accepted": True, "verdicts": {"*": True}}},
}
#: what `proof_layers.read_prove_path` reads of an audit's summaries, by the file's path under the audit: the
#: arm's whether it is plain and whose coins, a rank's (or a records audit's window's) source and its members' timing.
#: ``advice``, ``requests``, ``charge``, ``binding``, ``draw_counts`` (as in `ALLOWED`) and the rest are held.
SUMMARIES: dict[str, dict[str, Any]] = {
    "summary.json": {"plain": {"*": True}, "coins": {"challenger": True}},
    "rank*/summary.json": {"source": True, "timing": {"*": {
        "timing": {"proof_bytes": [True], "prove_total_s": True}, "lean_verify_s": True, "gpus": True, "device": True}}},
}
SUMMARIES["window/summary.json"] = SUMMARIES["rank*/summary.json"]
#: a traceback's last line, which leaves as its exception's class alone
ERROR = re.compile(r"(\w+(Error|Exception)): ")
HELD = "held in the kept log"


def allowed(o: Any, spec: Any) -> Any:
    """``o`` with only what ``spec`` allows; a value of another shape than ``spec``'s is `HELD`."""
    if spec is True:
        return o if o is None or isinstance(o, (bool, int, float, str)) else HELD
    if isinstance(spec, list):
        return [allowed(v, spec[0]) for v in o] if isinstance(o, list) else HELD
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


def summaries(kept: Path, out: Path) -> None:
    """Each summary `SUMMARIES` names under ``kept`` written under ``out`` in its public form."""
    for pattern, spec in SUMMARIES.items():
        for f in sorted(kept.glob(pattern)):
            (out / f.relative_to(kept)).parent.mkdir(parents=True, exist_ok=True)
            try:
                body = allowed(json.loads(f.read_text()), spec)
            except ValueError as e:
                body = {"held": f"not JSON ({type(e).__name__}): the file stays in the keep"}
            (out / f.relative_to(kept)).write_text(json.dumps(body, sort_keys=True))


def main(argv: list[str]) -> int:
    if argv[:1] == ["summaries"] and len(argv) == 3:
        summaries(Path(argv[1]), Path(argv[2]))
        return 0
    if len(argv) != 1:
        print("\n".join(x.strip() for x in __doc__.strip().splitlines()[-2:]), file=sys.stderr)
        return 2
    with open(argv[0], errors="replace") as f:
        for line in f:
            if (out := public_line(line.rstrip("\n"))) is not None:
                print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
