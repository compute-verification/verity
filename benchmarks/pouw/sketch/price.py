"""What each sketch of the committed words costs against the lead's SHA-512 of every word: h, the opening's AND gates,
T2 and the per-period totals: stdlib.

    python3 benchmarks/pouw/sketch/price.py --basis BASIS.json [--words-per-value G] [--json OUT]

- **Basis** (`--basis`, the lead's measured and stated figures, kept with the run, not in the tree): `volume` (useful
  MACs a committed 32-bit word is 32 / volume), `h_lead` and `h_lead_alt` (W1 of hashing a useful MAC under the two
  readings), `h_a` (the part of h that A's pre-noise rows spend, which a sketch of the words leaves alone), `t2_lead`
  (name: the lead's measured T2 at that shape), `mac_rate` (one GPU's useful MACs a second), `dc_gpus` and
  `world_macs_per_year`. One dense int8 MAC is one W1 unit and a register write 32, so one register a committed word is
  `volume` W1 a useful MAC, and the lead spends (h_lead - h_a) / volume registers a word on hashing.
- **A design** replaces SHA-512 of every word by a sketch: `alu` instructions a word (each one register), or `tc`
  tensor-pipe registers a word, then the sketch values, one 32-bit value a G words, committed exactly as the lead
  commits words (hm96 rows, frame-v3-sha512 trees, the same hiding), so the seal costs the lead's hashing a word divided
  by G.
- **Opening.** The AND gates a drawn unit's proof spends a committed word beyond recomputing it (130 AND gates a MAC):
  SHA-512 58,120 a compression of 32 words, and each design's own circuit beside it.
- **T2** (estimate): 1 plus each word's ALU registers at the lead's marginal, (T2 - 1) over its hashing registers a
  word. A sketch on the tensor pipe instead adds its MMA share of stock's time.
"""
from __future__ import annotations

import argparse
import json

AND_SHA_WORD = 58_120 / 32
AND_PER_MAC = 130
PERIODS = {"week": 7 * 86400, "month": 30 * 86400, "year": 365 * 86400}

# name: (ALU registers a word, tensor-pipe registers a word, opening AND gates a word, folds?, premise, rank)
DESIGNS = {
    "lead: SHA-512 of every word": (0.0, 0.0, None, False, "ShaWrites + CarriedWords", "4a (hashing part)"),
    "linear: R·w, Carter-Wegman, Freivalds, MLE, count-sketch": (None, None, None, True,
                                                                 "none holds (the linear fold)", "-"),
    "nh2: IMAD of adjacent accumulators, S += w(j)·w(j+1)": (None, None, None, True,
                                                             "none holds: on the lead's separable words each product "
                                                             "expands into row sums and a GEMM over the shared column "
                                                             "terms (nh_attack, from G 64)", "-"),
    "nh: IMAD of consecutive steps, S += w_(2k-1)·w_(2k)": (None, None, None, True,
                                                            "none holds on the lead's words (nh_attack, from G 32); "
                                                            "with one S an output on mirror-π noise, NHWrites", "5"),
    "xor: LOP3, S ^= w ^ w'": (None, None, None, True,
                               "none holds: the lead's narrow words pack two or three to a register and XOR "
                               "lane by lane", "-"),
    "chain: LOP3 and IADD3 alternating, S = (S ^ w ^ w') + w'' + w'''": (None, None, None, True,
                                                                          "none holds: the added words enter linearly "
                                                                          "and fold (synth, T 4)", "-"),
    "sumsq: IMAD, S += w·w": (None, None, None, True,
                              "none holds: the square expands on the lead's separable words, as nh2 does", "-"),
    "tc-bytes: int8 MMA on the accumulators' bytes, N = 8": (0.0, 1.0, 32 * AND_PER_MAC, False, "BytesWrites", "4"),
    "lookups: four byte tables a word": (10.0, 0.0, 1_100.0, False, "LookupWrites (histogram fold risk)", "4-5"),
    # GD-SIS's opening is priced by its own method (a multiple of SHA-512's rows); it is not recomputed here
    "GD-SIS: A·x mod 2^32, binding (for reference)": (0.0, 3584 / 32, None, False, "M-SIS + cost", "5"),
}


def price(B, G):
    macs_per_word = 32 / B["volume"]
    word_regs_lead = (B["h_lead"] - B["h_a"]) / B["volume"]
    rows = []
    for name, (alu, tc, and_w, folds, prem, rank) in DESIGNS.items():
        if folds:
            rows.append({"design": name, "folds": True, "premise": prem, "rank": rank})
            continue
        lead = name.startswith("lead")
        if lead:
            h, regs_alu_word, seal, and_word = B["h_lead"], word_regs_lead, 0.0, AND_SHA_WORD
        else:
            seal = word_regs_lead / G
            h = B["h_a"] + B["volume"] * (alu + tc + seal)
            regs_alu_word = alu + seal
            and_word = None if and_w is None else and_w + AND_SHA_WORD / G
        t2 = {k: round(1 + (v - 1) / word_regs_lead * regs_alu_word + (tc * 32 / macs_per_word if tc else 0.0), 3)
              for k, v in B["t2_lead"].items()}
        rows.append({
            "design": name, "folds": False, "premise": prem, "rank": rank,
            "h": round(h, 3), "h_reduction": round(B["h_lead"] / h, 1),
            "h_at_alt": B["h_lead_alt"] if lead else
            round((B["h_a"] + B["volume"] * seal) * B["h_lead_alt"] / B["h_lead"] + B["volume"] * (alu + tc), 3),
            "opening_and_per_word": None if and_word is None else round(and_word, 1),
            "opening_vs_sha": None if and_word is None else round(and_word / AND_SHA_WORD, 4),
            "opening_with_words_vs_lead": None if and_word is None else
            round((AND_PER_MAC * macs_per_word + and_word) / (AND_PER_MAC * macs_per_word + AND_SHA_WORD), 3),
            "t2": t2,
        })
    return rows


def periods(B, rows):
    macs_per_word = 32 / B["volume"]
    scopes = [("one GPU", B["mac_rate"]), (f"datacenter ({B['dc_gpus']:,} GPUs)", B["mac_rate"] * B["dc_gpus"]),
              ("world", B["world_macs_per_year"] / PERIODS["year"])]
    out = []
    for r in rows:
        if r["folds"]:
            continue
        for scope, p, macs in [(scope, p, rate * s) for scope, rate in scopes for p, s in PERIODS.items()]:
            out.append({"design": r["design"], "scope": scope, "period": p, "useful_macs": macs,
                        "stock_and": macs * AND_PER_MAC, "hashing_w1": macs * r["h"],
                        "opening_and_if_all_opened": None if r["opening_and_per_word"] is None else
                        macs / macs_per_word * r["opening_and_per_word"]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--basis", required=True, help="JSON of the lead's figures (see the docstring)")
    ap.add_argument("--words-per-value", type=int, default=1024, help="G: words a 32-bit sketch value covers")
    ap.add_argument("--json")
    a = ap.parse_args()
    with open(a.basis) as f:
        B = json.load(f)
    rows = price(B, a.words_per_value)
    print(f"## Designs at G = {a.words_per_value} words a sketch value (h: W1 of hashing a useful MAC;"
          f" lead {B['h_lead']})")
    for r in rows:
        if r["folds"]:
            print(f"- {r['design']}: folds; {r['premise']}")
            continue
        print(f"- {r['design']}: h {r['h']} ({r['h_reduction']}x under the lead; {r['h_at_alt']} on the"
              f" {B['h_lead_alt']} reading); opening {r['opening_and_per_word']} AND a word ({r['opening_vs_sha']}x"
              f" SHA-512's; {r['opening_with_words_vs_lead']}x the lead's with the words' recomputation); T2 {r['t2']};"
              f" premise {r['premise']}, rank {r['rank']}")
    sweep = {G: [(r["design"].split(":")[0], r["h"]) for r in price(B, G) if not r["folds"]]
             for G in (64, 256, 1024, 4096)}
    print("## h against G")
    for G, hs in sweep.items():
        print(f"- G {G}: " + ", ".join(f"{n} {h}" for n, h in hs))
    per = periods(B, rows)
    print("## Per period (hashing W1, stock AND gates beside it)")
    for row in per:
        if row["design"].startswith(("lead", "tc-bytes")):
            opened = row["opening_and_if_all_opened"]
            print(f"- {row['design'].split(':')[0]}, {row['scope']}, {row['period']}: hashing {row['hashing_w1']:.3g} W1,"
                  f" {opened:.3g} AND gates as a circuit; stock {row['stock_and']:.3g} AND gates")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"basis": B, "G": a.words_per_value, "designs": rows, "sweep": sweep, "periods": per}, f, indent=1)


if __name__ == "__main__":
    main()
