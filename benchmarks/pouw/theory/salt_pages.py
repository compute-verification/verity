"""Prices for two decisions on hiding: the keyed leaf against hm96's forms under L1 (one hm96 instance a call and one a
forward), and L2 (private roots, one hm96 instance a window), at the five points, read from N-HIDE's prices
(np_hide_price.json) and N-SALT's (salt.json). Stdlib only.

    python3 benchmarks/pouw/theory/salt_pages.py NP_HIDE_PRICE_JSON SALT_JSON OUT_JSON

L2's opening climbs ceil(log2(65 F)) levels above a call's subtree, F the window's forwards, two compressions a level on
each of the unit's two paths. np_hide_price.py takes F = 6,000 (a step of at most 0.1 s), which bounds the hiding share from
above but the depth from below. The depth here also takes F at its cap, the most forwards one H200 can run in a window:
every forward reads each int8 linear weight once, so it takes at least max(rows · MACs a token / peak, weight bytes /
bandwidth), both from the data sheet.

The forced mask's check reads its salt's committed partials, 512 × 48 int32 words, so a proof that checks them hashes
their 98,304 bytes in-circuit beside the 37 ANDs a partial; a drawn unit checks the salts of both instances it opens.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

POINTS = {"T 512, 64 rows": ("T 512, 64 real rows", 64), "T 512, 512 rows": ("T 512, 512 real rows", 512),
          "T 8,192": ("T 8,192", 8192), "decode, 16 rows": ("decode, 16 rows", 16), "decode, 1 row": ("decode, 1 row", 1)}
ANDS_COMPRESSION = 58_120
CALLS = 64
WINDOW_S = 600.0
STEP_S_NHIDE = 0.1
H200_PEAK_MACS = 9.89e14
H200_HBM_BYTES_S = 4.8e12
FM_PARTIAL_BYTES = 512 * 48 * 4
TARGET = 0.001
T1 = 0.01


def forwards_cap(rows: int, macs_token: int) -> float:
    return WINDOW_S / max(rows * macs_token / H200_PEAK_MACS, macs_token / H200_HBM_BYTES_S)


def levels(forwards: float) -> int:
    return math.ceil(math.log2((CALLS + 1) * forwards))


def main() -> None:
    nh = json.loads(Path(sys.argv[1]).read_text())
    salt = json.loads(Path(sys.argv[2]).read_text())
    lead = nh["designs"]["ssn_1l"]
    macs_token = lead["bytes"]["macs"]
    fm_check = salt["in_circuit_ands"]["hm96 + (a) + FM"]["a digest"] - salt["in_circuit_ands"]["hm96 + (a)"]["a digest"]
    fm_digest = fm_check + math.ceil((FM_PARTIAL_BYTES + 17) / 128) * ANDS_COMPRESSION
    up_nh = levels(WINDOW_S / STEP_S_NHIDE)
    out: dict = {"inputs": {"macs_token": macs_token, "weight_bytes": macs_token, "fm_check_ands_a_salt": fm_check,
                            "fm_ands_a_salt_with_its_partials_hashed": fm_digest,
                            "l2_levels_at_6000_forwards": up_nh}, "points": {}}
    for name, (nh_name, rows) in POINTS.items():
        pt = lead["points"][nh_name]
        hid, opn = pt["hiding"], pt["opening_ands"]
        cap = forwards_cap(rows, macs_token)
        extra = levels(cap) - up_nh
        l2 = {m: opn["L2, a window"][m]["paths and gadget"] + extra * 4 * ANDS_COMPRESSION for m in ("unforced", "forced (a)")}
        fm = salt["shares"]["hm96 + (a) + FM, interface uncommitted, L1"][name]["credited"]
        rows_out = {
            "keyed leaf (L1)": {"share": salt["shares"]["keyed leaf (P3), L1"][name]["credited"],
                                "opening": opn["L1"]["keyed leaf"]["paths and gadgets"]},
            "hm96 + (a) (L1)": {"share": {p: hid["L1"]["forced (a)"][p] for p in ("floor", "alone")},
                                "opening": opn["L1"]["forced (a)"]["paths and gadgets"]},
            "hm96 + (a) + FM (L1)": {"share": fm, "opening": opn["L1"]["forced (a)"]["paths and gadgets"] + 2 * fm_digest},
            "L2, OS salt": {"share": {p: hid["L2, a window"]["unforced"][p] for p in ("floor", "alone")},
                            "opening": l2["unforced"],
                            "opening at 6,000 forwards": opn["L2, a window"]["unforced"]["paths and gadget"]},
            "L2 + (a)": {"share": {p: hid["L2, a window"]["forced (a)"][p] for p in ("floor", "alone")},
                         "opening": l2["forced (a)"],
                         "opening at 6,000 forwards": opn["L2, a window"]["forced (a)"]["paths and gadget"]},
        }
        for r in rows_out.values():
            r["passes target"] = r["share"]["alone"] <= TARGET
            r["passes T1"] = r["share"]["alone"] <= T1
        out["points"][name] = {"rows": rows, "forwards cap": cap, "l2 levels": levels(cap), "extra levels": extra,
                               "designs": rows_out}
    stock = nh["periods"]["stock_ands"]
    out["stock_ands"] = stock
    out["pooled_floor"] = {f"{d}, {p}": {s: {per: v * out["points"][p]["designs"][d]["share"]["floor"]
                                             for per, v in stock[s].items()} for s in stock}
                           for d in ("hm96 + (a) (L1)", "hm96 + (a) + FM (L1)", "L2, OS salt", "keyed leaf (L1)")
                           for p in ("T 512, 64 rows", "decode, 1 row")}
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
    for name, pt in out["points"].items():
        print(f"{name}: forwards cap {pt['forwards cap']:.3g}, L2 levels {pt['l2 levels']} (+{pt['extra levels']})")
        for d, r in pt["designs"].items():
            s = r["share"]
            print(f"  {d:24s} {100 * s['floor']:.4g}% / {100 * s['alone']:.4g}%  opening {r['opening']:,}"
                  + (f" ({r['opening at 6,000 forwards']:,} at 6,000)" if "opening at 6,000 forwards" in r else "")
                  + f"  target {r['passes target']} T1 {r['passes T1']}")
    for k, v in out["pooled_floor"].items():
        print(k, {s: " / ".join(f"{v[s][per]:.3g}" for per in ("window", "week", "month", "year")) for s in v})


if __name__ == "__main__":
    main()
