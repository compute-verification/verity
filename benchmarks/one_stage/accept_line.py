"""The one-stage audit's accept lines from what the verifiers printed: the Lean members check's verdict of record (its
object, the outcome's line as the executable printed it and that line's SHA-512), soundness's sentence from the
verifier's audit record (`wrong_units`: its δ, population and bound) and C's AND gates (`verity.circuits.count`), and
the two δ the road holds, compared. The record's `registered` check is no holding of the reads: every member's ports
are held by `OneStageZk.headerHeldAt` in the members check.

    accept_line.py SUMMARY_JSON C_JSON DELTA
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

from verity.circuits.codec import descriptor_id
from verity.circuits.count import count
from verity.circuits.restrict import program
from verity.protocols.proofs.spot.verifier.one_stage.verifier import REGISTRATION_DELTA


def lines(summary: dict, c: bytes, delta: str) -> list[str]:
    ml = summary["audit"]["members_lean"]
    lean = (ml.get("detail") or {}).get("lean") or {}
    w = summary["audit"]["wrong_units"]
    n, bound, d = int(w["of"]), int(w["bound"]), Fraction(str(w["delta"]))
    k = len(summary["draw"])
    P = program(c)
    declared = count(P)[descriptor_id(P.root)].ands
    if declared % n:
        raise SystemExit(f"C's {declared} AND gates are not {n} equal units: no per-unit figure")
    per = declared // n
    held = bound * per
    return [
        "ACCEPT verdict of record: the Lean members check (OneStageZk.checkMembersAt), which holds every member's ports "
        "(headerHeldAt): " + json.dumps({x: ml.get(x) for x in ("ran", "ok", "codes")}, sort_keys=True)
        + f" output {lean.get('output')} sha512 {lean.get('output_sha512')}; this road makes no stream entry",
        f"ACCEPT one window, {n} units, {k} drawn: except with probability {d}, at most {held:,} AND gates of its "
        f"{declared:,} declared ({100 * held / declared:.1f}%) are in blocks that are not correct "
        f"(the verifier's bound: {bound} of {n} units, each {per:,} AND gates by verity.circuits.count)",
        f"ACCEPT δ: the verifier's audit record bounds at {d} (wrong_units.delta), the command passed {delta}; the "
        f"registration path's parameters carry REGISTRATION_DELTA = {REGISTRATION_DELTA}, which R1-R8 do not read; "
        f"compared: record {'=' if d == Fraction(delta) else '!='} command, record "
        f"{'=' if d == Fraction(REGISTRATION_DELTA) else '!='} REGISTRATION_DELTA",
    ]


if __name__ == "__main__":
    s, c, delta = sys.argv[1:4]
    print("\n".join(lines(json.loads(Path(s).read_text()), Path(c).read_bytes(), delta)))
