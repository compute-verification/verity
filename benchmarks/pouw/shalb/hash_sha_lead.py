"""The lead's hashing baseline from N-CARD's calculator, at the three Llama-3.2-1B points, for hash_sha.py.

    PYTHONPATH=CARD_DIR:EXACT_KIT_DIR python3 hash_sha_lead.py OUT.json

CARD_DIR holds `pouw_space.py` and `pouw_space.toml` (benchmarks/pouw/theory on the card's branch); EXACT_KIT_DIR is
benchmarks/pouw/exact_kit. It builds the card's design space once (about three minutes) and evaluates the lead at each
point: h (hash_units_per_mac), the hashed bytes a MAC, r (the honest MACs a stock MAC), T1's terms and T2's parts.
"""
from __future__ import annotations

import json
import sys

LEAD = "tag-blc4a-t64-rot/served/s8/k32/w/l1w/l1a/min"
POINTS = ("t512", "t512r512", "t8192")


def main() -> None:
    out_path = sys.argv[1]
    sys.argv = sys.argv[:1]
    import pouw_space as P

    card = P.load(P.CARD, [])
    designs, _, _ = P.space(card)
    d = designs[LEAD]
    out = {"design": LEAD, "points": {}}
    for p in POINTS:
        r = P.evaluate(card, d, p, 0.0)
        t2 = r["t2"]
        out["points"][p] = {
            "title": card["point"][p]["title"], "h": r["hash_units_per_mac"], "bytes_per_mac": r["hash_bytes_per_mac"],
            "r": r["w_mac"] - r["hash_units_per_mac"], "stock_macs": r["stock_macs_forward"], "t1": r["t1"],
            "terms": {k: v for k, v in r["terms"].items() if v}, "t2": {e: v["ratio"] for e, v in t2["engines"].items()},
            "stock_ms": {e: v["stock_ms"] for e, v in t2["engines"].items()}, "hash_ms": t2["hash_ms"],
            "extra_parts_ms": t2["extra_parts_ms"], "bytes_forward": t2["bytes_forward"]}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
