"""Prices from a probes.sh run of copy.cu, branch.cu, brx.cu, br2.cu and lut.cu, in W1's units: an SM-clock is 4096
units (the floor, 32 a register, is 128 registers an SM-clock).

copy       a role's words an SM-clock (probe.py's rate x words a warp instruction), 4096 / that a word, and its share of
           the floor's 128 words an SM-clock; beside honest (hs_*), the same for the role's words while the honest warps
           ran, the honest MMA's rate as a share of its rate beside idle warps (honest_b, honest_l, honest_w), and the SM time
           honest lost a word placed: (1 - kept) x 4096 / words an SM-clock.
decisions  4096 x the median block's cycles / (iters x warps x decisions a trip) units a warp decision, its coin bits
           (bits a decision: one for the two-way forms, 4 for BRX16, a digit for a lookup), units a bit read by the
           warp and a lane-bit for the forms whose lanes read their own coins; the overhead is over the arms alone
           (`arm`, `larm`), scaled by decisions a trip where the arms differ.

  python prices.py OUT_DIR   # OUT_DIR: a probes.sh run's out/ (one directory a binary); prints JSON lines
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

SM_CLOCK = 4096
FLOOR_WORDS = 128
WARPS = 32  # the decision probes' blocks: 1024 threads

# decisions a trip, coin bits a decision, lanes read their own coins, the arms-alone probe
DECIDE = {
    "arm": (8, 1, False, None), "bru": (8, 1, False, "arm"), "brv": (8, 1, False, "arm"),
    "brw": (8, 1, False, "arm"), "brd": (8, 1, True, "arm"), "sel": (8, 1, False, "arm"),
    "lutb": (4, 8, False, None), "lutl": (4, 8, True, None),
    "brxu": (2, 4, False, "arm"), "brxv": (2, 4, False, "arm"), "brxw": (2, 4, False, "arm"),
    "brxd": (2, 4, True, "arm"),
    "x2u": (8, 1, False, "arm"), "x2w": (8, 1, False, "arm"), "x2d": (8, 1, True, "arm"),
    "lu": (4, 1, False, "larm"), "lw": (4, 1, False, "larm"), "ld": (4, 1, True, "larm"), "larm": (4, 1, False, None),
    "lut8": (4, 8, True, None), "lut10": (3, 10, True, None), "lut12": (2, 12, True, None),
}
HONEST = {"hs_g2s1k": "honest_b", "hs_g2s4k": "honest_b", "hs_g2s4kh": "honest_b", "hs_s2g4k": "honest_b",
          "hs_g2s16k": "honest_b", "hs_g2s4kz": "honest_b", "hs_g2s16kz": "honest_b", "hs_s2g4kz": "honest_b",
          "hw_g2s4k": "honest_w", "hw_g2s16k": "honest_w", "hw_g2s4kz": "honest_w", "hw_g2s4khz": "honest_w",
          "hw_s2g4kz": "honest_w",
          "hs_ldgsts4": "honest_l", "hs_ldgsts16": "honest_l", "hs_ldssts4": "honest_l", "hs_ldssts16": "honest_l"}


def load(d: Path) -> tuple[dict, dict]:
    rows, times = {}, {}
    for r in (json.loads(x) for x in (d / "rows.jsonl").read_text().splitlines() if x.strip()):
        rows[r["probe"]] = r
    for t in (json.loads(x) for x in (d / "times.jsonl").read_text().splitlines() if x.strip()):
        if t.get("op") == "time":
            times.setdefault(t["probe"], []).append(t)
    return rows, times


def imma(r: dict) -> float:
    return next(v for k, v in r["rates"].items() if "MMA" in k)


def main(out: Path) -> None:
    rows, times = {}, {}
    for d in sorted(p for p in out.iterdir() if (p / "rows.jsonl").exists()):
        r, t = load(d)
        rows.update(r)
        times.update(t)
    per_trip = {}
    for name in DECIDE:
        if name in times:
            per_trip[name] = statistics.median(SM_CLOCK * statistics.median(t["cyc"]) / (t["iters"] * WARPS)
                                               for t in times[name])
    for name, r in sorted(rows.items()):
        rec = {"probe": name, "run": r["run"], "sm_mhz": r["sm_mhz"], "spread": round(r["spread"], 4),
               "tags": r["tags"]}
        if name in DECIDE:
            d, bits, lanes, base = DECIDE[name]
            u = per_trip[name] / d
            rec.update(kind="decision", units_a_warp_decision=round(u, 1), bits_a_decision=bits,
                       units_a_bit=round(u / bits, 1))
            if lanes:
                rec["units_a_lane_bit"] = round(u / bits / 32, 2)
            if base and base in per_trip:
                bd = DECIDE[base][0]
                rec["overhead_a_warp_decision"] = round((per_trip[name] - per_trip[base] * d / bd) / d, 1)
        elif r.get("words"):
            w = r["rate"] * r["words"]
            rec.update(kind="copy", words_an_sm_clock=round(w, 3), units_a_word=round(SM_CLOCK / w, 1),
                       share_of_floor=round(w / FLOOR_WORDS, 5))
            if name in HONEST and HONEST[name] in rows:
                kept = imma(r) / rows[HONEST[name]]["rate"]
                rec["honest_kept"] = round(kept, 4)
                rec["lost_units_a_word"] = round((1 - kept) * SM_CLOCK / w, 1)
                rec["stop_gap"] = r.get("stop_gap")
        else:
            rec.update(kind="honest", imma_rate=round(r["rate"], 4))
        print(json.dumps(rec))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
