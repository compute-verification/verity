"""The one-stage audit's accept lines from what the verifiers printed: the Lean members check's verdict of record (its
object, the outcome's line as the executable printed it and that line's SHA-512), soundness's sentence from the
verifier's audit record (`wrong_units`: its δ, population and bound), in C's AND gates (`verity.circuits.count`) where
C is given, and the δ the road holds, compared. The record's `registered` check is no holding of the reads: every
member's ports are held by `OneStageZk.headerHeldAt` in the members check. A summary that is no accept gets one
``NO ACCEPT`` line naming the record's failed checks, and the script exits 1.

    accept_line.py SUMMARY_JSON [C_JSON|-] [DELTA|-]
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

from one_linear_arms import refusal
from verity.protocols.proof.spot.verifier.one_stage.verifier import REGISTRATION_DELTA

def lines(summary: dict, c: bytes | None, delta: str | None) -> list[str]:
    if (why := refusal(summary)) is not None:
        return ["NO ACCEPT " + why]
    ml = summary["audit"]["members_lean"]
    lean = (ml.get("detail") or {}).get("lean") or {}
    w = summary["audit"]["wrong_units"]
    n, bound, d = int(w["of"]), int(w["bound"]), Fraction(str(w["delta"]))
    k = len(summary["draw"])
    out = ["ACCEPT verdict of record: the Lean members check (OneStageZk.checkMembersAt), which holds every member's "
           "ports (headerHeldAt): " + json.dumps({x: ml.get(x) for x in ("ran", "ok", "codes")}, sort_keys=True)
           + f" output {lean.get('output')} sha512 {lean.get('output_sha512')}; this road makes no stream entry"]
    if c is None:
        out.append(f"ACCEPT one window, {n} units, {k} drawn: except with probability {d}, at most {bound} of its {n} "
                   "units are in blocks that are not correct (no C given, so no AND-gate figure)")
    else:
        from verity.circuits.codec import descriptor_id
        from verity.circuits.count import count
        from verity.circuits.restrict import program
        P = program(c)
        declared = count(P)[descriptor_id(P.root)].ands
        if declared % n:
            raise SystemExit(f"C's {declared} AND gates are not {n} equal units: no per-unit figure")
        per = declared // n
        held = bound * per
        out.append(f"ACCEPT one window, {n} units, {k} drawn: except with probability {d}, at most {held:,} AND gates "
                   f"of its {declared:,} declared ({100 * held / declared:.1f}%) are in blocks that are not correct "
                   f"(the verifier's bound: {bound} of {n} units, each {per:,} AND gates by verity.circuits.count)")
    told = "the command passed none" if delta is None else f"the command passed {delta}"
    vs = "" if delta is None else f"record {'=' if d == Fraction(delta) else '!='} command, "
    out.append(f"ACCEPT δ: the verifier's audit record bounds at {d} (wrong_units.delta), {told}; the registration "
               f"path's parameters carry REGISTRATION_DELTA = {REGISTRATION_DELTA}, which R1-R8 do not read; compared: "
               f"{vs}record {'=' if d == Fraction(REGISTRATION_DELTA) else '!='} REGISTRATION_DELTA")
    return out


if __name__ == "__main__":
    args = [*sys.argv[1:], "-", "-"][:3]
    s, c, delta = args
    got = lines(json.loads(Path(s).read_text()), None if c == "-" else Path(c).read_bytes(),
                None if delta == "-" else delta)
    print("\n".join(got))
    sys.exit(1 if got[0].startswith("NO ACCEPT") else 0)
