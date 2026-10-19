"""The sm_120 unit's floor under CP-W, against #2482's headline (RT2-120): the Lean bound's break line, and what the
per-set checkpoint adds to the floor of every program `PouwWarpCheck_v2` accepts at FOLD fx1.

The bound (`lean_headline.branches` on `Definitions/Accounting/Headline/Sm120.lean` at a495f9c5dd) reads a cheater's
floor a unit as N_w (w* + φ*) = 17: an IMMA.16832 writes 128 words a slot, and fx1's fold is 8 ALU + 8 IMAD lane
instructions a unit. ♭'s bound B adds the operand floor (LDSM_X4 a 39th of a unit) and ε_skip's NOROW. The bound takes
its unfolded branch 1 − Y (1 − 41/S), Y = 17 / B, rounded up to 656/100,000, so it fails for a floor below
B (1 − ε) / (1 − 41/S), a hair under 17.

CP-W (decision 104) commits, after every set (32 words a lane, 8 units a warp), the XOR over the warp's 32 lanes and 8
chains of y and of z (`segment.warp_checkpoint`), and `PouwWarpCheck_v2` reads both words of every set. Neither ♭ nor
the Lean floor has this work. Its cost a set-warp on DS-120 (every class 1 on DS_issue), by route:

- merged by issued instructions: 2 (256 − 1) = 510 merges, and no class merges more than 64 a warp slot (LOP3: 2 a lane;
  RED/ATOMS.XOR.64: a lane's 2 words into memory; REDUX: 31), so at least 8 slots;
- moved out of the registers and merged off the issue path (STSM_X4 is issue only on DS-120, as are the TMA reductions
  UTMA*/TMAred512): 512 words, and no class takes more than 4 words a lane out of the registers, so at least 4 slots.

`main` prints the bound, the routes' slot counts and the floor each implies, and checks on `segment` that the 64-bit
route (each lane's y XOR and z XOR as one word pair, XORed over lanes) is `warp_checkpoint` set by set.

    uv run python benchmarks/pouw/theory/cpw_floor.py
"""
from __future__ import annotations

import json
import random
import sys
from fractions import Fraction

from verity.core.protocols.pouw.circuit import segment as SG

#: Sm120.lean at a495f9c5dd: N_w, ♭'s word floor (IMMA16832, LDSM_X4), fold (ALU, IMAD) and skip (NOROW), S
NW = 128
FLAT_WORDS = {"IMMA16832": Fraction(1), "LDSM_X4": Fraction(25641, 1000000)}
FLAT_FOLD = {"ALU": Fraction(8), "IMAD": Fraction(8)}
FLAT_SKIP = {"NOROW": Fraction(14325412520309335111003, 199158597757329117000000)}
CHECKS = 48000
#: the statement's ε and the brief's break line (1 − 0.599%) B
EPS = Fraction(656, 100000)
BRIEF_BREAK = Fraction(16995, 1000)
#: lean_headline's w* (an IMMA.16832's slot over its 128 words) and φ* (fx1's 4 lane instructions a word over 32 lanes)
W_STAR, PHI_STAR = Fraction(1, 128), Fraction(1, 8)
LN2 = Fraction(6931471808, 10**10)
LN_P = Fraction(1, 8)

UNITS_A_SET = 8
LANES = 32
#: XOR merges a warp slot: LOP3 (3 inputs, 2 merges a lane), 64-bit RED/ATOMS XOR (a lane's 2 words into memory), REDUX
MERGES = {"LOP3": 2 * LANES, "RED.XOR.64": 2 * LANES, "REDUX.XOR": LANES - 1}
#: lane words a warp slot takes out of the registers (STSM_X4, STS.128: 4 a lane)
EGRESS = 4 * LANES


def bound() -> dict:
    B0 = sum(FLAT_WORDS.values()) + sum(FLAT_FOLD.values())
    B = B0 + sum(FLAT_SKIP.values())
    floor = NW * (W_STAR + PHI_STAR)
    Y = floor / B
    folded = 1 - Y + NW * W_STAR / B * 41 * LN2 / (LN_P * CHECKS)
    unfolded = 1 - Y * (1 - Fraction(41, CHECKS))
    brk = B * (1 - EPS) / (1 - Fraction(41, CHECKS))
    return {"B_flat_no_skip": float(B0), "B_flat": float(B), "lean_floor": float(floor), "Y": float(Y),
            "eps_folded": float(folded), "eps_unfolded": float(unfolded), "eps_stated": float(EPS),
            "break_line": float(brk), "break_slack_slots": float(floor - brk),
            "brief_break_line": float(BRIEF_BREAK), "brief_break_is": "(1 - 0.599%) B_flat, an epsilon without luck",
            "brief_break_rel_B": float(1 - BRIEF_BREAK / B)}


def checkpoint() -> dict:
    words = 2 * SG.NCH * LANES
    merges = 2 * (SG.NCH * LANES - 1)
    lane_xor = (2 * SG.NCH - 2) // 2
    routes = {
        "redux (7 LOP3, 2 REDUX, 1 STG)": lane_xor + 2 + 1,
        "shfl butterfly (7 LOP3, 10 SHFL, 10 LOP3, 1 STG)": lane_xor + 10 + 10 + 1,
        "red64 (7 LOP3, 1 RED.XOR.64)": lane_xor + 1,
        "atoms64 (7 LOP3, 1 ATOMS.XOR.64; flush amortized)": lane_xor + 1,
        "egress (4 STSM_X4; TMA XOR tree amortized)": 4,
    }
    lb_issue = -(-merges // max(MERGES.values()))
    lb_egress = -(-words // EGRESS)
    b = bound()
    out = {"set_words_a_lane": 32, "units_a_set_warp": UNITS_A_SET, "state_words_a_set_warp": words,
           "xor_merges_a_set_warp": merges, "routes_slots_a_set_warp": routes,
           "routes_slots_a_unit": {k: v / UNITS_A_SET for k, v in routes.items()},
           "lower_bound_issue_merged": lb_issue, "lower_bound_egress": lb_egress}
    for name, lb in (("issue_merged", lb_issue), ("egress", lb_egress)):
        f = b["lean_floor"] + lb / UNITS_A_SET
        out[f"floor_{name}"] = f
        out[f"time_over_credit_{name}"] = f / b["B_flat"]
        out[f"fx1_must_save_{name}"] = (f - b["break_line"]) / float(NW * PHI_STAR)
    return out


def pair_route_is_warp_checkpoint(sets: int = 4, seed: int = 120) -> bool:
    """Each set's 64-bit route (lane XOR of y in the low word, of z in the high, XORed over lanes) against
    `warp_checkpoint`, on random states and words through `segment.absorb('fx1', ...)`."""
    rnd = random.Random(seed)
    states = [[rnd.getrandbits(32) for _ in range(2 * SG.NCH)] for _ in range(LANES)]
    for _ in range(sets):
        states = [SG.absorb("fx1", st, [rnd.getrandbits(32) for _ in range(32)]) for st in states]
        acc = 0
        for st in states:
            ly = lz = 0
            for j in range(SG.NCH):
                ly ^= st[j]
                lz ^= st[SG.NCH + j]
            acc ^= (lz << 32) | ly
        if [acc & SG.M32, acc >> 32] != SG.warp_checkpoint(states):
            return False
    return True


def main() -> int:
    doc = {"bound": bound(), "cpw": checkpoint(), "pair_route_is_warp_checkpoint": pair_route_is_warp_checkpoint()}
    print(json.dumps(doc, indent=1))
    return 0 if doc["pair_route_is_warp_checkpoint"] else 1


if __name__ == "__main__":
    sys.exit(main())
