"""Hiding and zero knowledge for the proposals that reduce PoUW's hashing, priced: what each costs to hide, as a share of
h (honest hashing over stock in W1) and of T1's budget, and what it leaks if it is published.

    uv run python benchmarks/pouw/theory/hash_hide.py OUT_JSON

Five parts:

- ``ands``: the in-circuit AND gates a drawn unit spends a committed 32-bit word, counted by compiling the tree's own
  Boolean Definitions: one SHA-512 compression (32 words) and the keyed folds li, duo, fx1 and xin (32 words each, the
  compute protocol's receipts, the one-hash rule's named Duo exception). Beside them, by the textbook counts (an estimate),
  a keyed 32-bit multiply, a carry-less 64-bit multiply over GF(2^64), a multiply mod 2^61 - 1 and a random byte lookup
  (one-hot decoder; its output XORs are free).
- ``per_value``: the hashing a value costs when it is hidden on its own (its own published digest) rather than inside the
  forward's words tree under one hm96 instance (layout L1): hm96 with the forced salt derivation (a), the pad
  b = x xor SHA-512(ks || salt || pos), the keyed digest, and the keyed leaf (SHA-512 started from the window's secret
  midstate S), each with the value's own frame leaf where the mechanism needs one. As a share of the lead's hashing a
  useful MAC, one hidden value every M useful MACs costs 992 c / M for c compressions, since the lead hashes 4 bytes a
  word and a word carries 31 useful MACs.
- ``sketch``: h over the lead's h for a b-byte sketch committed in place of every s words, inside the words tree (no
  hiding cost beyond its bytes), and with each sketch hidden on its own under each mechanism of ``per_value``.
- ``rounds``: chained rounds, R commit-then-coin rounds a call. Each round publishes one hiding root before its coin: under
  L1 + (a) one hm96 instance, so the hiding term (T1's honest-only part) is R times L1's term at one instance a call (the
  inputs are L1's term at each point); under the keyed leaf every root is a keyed digest and costs nothing.
- ``leak``: a published sketch. A linear sketch s = sum_t r_t w_t (mod p) with public coins r is a public linear function
  of the running words; S sketches with independent r over an S-step chain recover every word, hence every step's
  partial product (checked exactly here on random int8 data), and its bits count in leakage as a share of a unit's
  262,144 output bits.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

R_WORDS = 3515.38                    # ShaWrites' registers a compression for the words' form
H_LEAD = {"low": 119.5, "high": 233.0}   # the lead's h, both readings of stock in W1
WORD_BYTES, MACS_WORD, BLOCK = 4, 31, 128
LEAD_C_MAC = WORD_BYTES / BLOCK / MACS_WORD   # compressions a useful MAC in the lead: 1/992
PRF_SHARE = 53_568 / 58_120          # a compression from round 8 over a whole one (hm96's PRF)
MASK_REGS = 16 * (12 * (20 + 10 + 3) + 37 + 32)   # M_key·y in the H200 byte-table form, registers written a salt
PAD_XOR_REGS = 16
FRAME_LEAF_PREFIX = 33 + 4 + 4 + (8 + 64) + 2 * (8 + 3) + (8 + 22) + 8
NODE = 2                             # a binary frame-v3 node's compressions past its midstate
UNIT_WORDS = 262_144 // 32           # a drawn unit's words at the largest public size class
ANDS_MAC = 2079 / 16
# L1 + (a)'s hiding term at one instance a call, % of stock (W1 floor / alone): the inputs to `rounds`
L1_TERM = {"decode, 1 row": (1.6070, 3.2200), "decode, 16 rows": (0.1004, 0.2012), "T 512, 64 rows": (0.0251, 0.0503)}
LEAD_T1 = {"decode, 1 row": None, "decode, 16 rows": None, "T 512, 64 rows": 0.066}
CALLS = 64
TARGET, BUDGET = 0.1, 1.0            # % of stock
P61 = (1 << 61) - 1


def blocks(n: int) -> int:
    return (n + 17 + 127) // 128


def ands() -> dict:
    from verity.core.protocols.pouw.circuit import segment as SG
    from verity.ml.circuits.evaluation import bits as BS
    from verity.ml.commitments.gates import lms

    sha = BS.compile(lms.Sha512Compress).counts["and"]
    out = {"SHA-512 compression": {"ands_32_words": sha, "ands_a_word": sha / 32, "label": "checked"}}
    for fold in ("Li", "Duo", "Fx1", "Xin"):
        n = BS.compile(getattr(SG, f"Pouw{fold}Absorb")).counts["and"]
        out[f"{fold.lower()} fold"] = {"ands_32_words": n, "ands_a_word": n / 32, "label": "checked"}
    est = {"keyed u32 multiply (both operands private)": 32 * 33 // 2 + 31 * 31 // 2 + 31,
           "GF(2^64) carry-less multiply, per 32-bit word": 64 * 64 // 2,
           "multiply mod 2^61 - 1 (60 data bits a coefficient), per 32-bit word": (61 * 61 + 60 * 61 + 61) * 32 // 60,
           "random byte lookup x4, per 32-bit word": 4 * (2 * (2 * 4 + 16) + 256)}
    out.update({k: {"ands_a_word": v, "label": "estimate"} for k, v in est.items()})
    for v in out.values():
        v["over_sha"] = v["ands_a_word"] / (sha / 32)
    unit = {"words recomputed (31 MACs a word)": UNIT_WORDS * MACS_WORD * ANDS_MAC,
            "SHA-512 over the unit's words": UNIT_WORDS * sha / 32}
    unit.update({f"{k} over the unit's words": UNIT_WORDS * v["ands_a_word"] for k, v in out.items() if "fold" in k})
    unit.update({f"a {c // 1024} KiB chunk opened whole": blocks(c) * sha for c in (16384, 65536, 262144)})
    return {"a_word": out, "a_drawn_unit": unit}


def per_value(n_bytes: int = 4) -> dict:
    """Compressions (register-equivalent at R_WORDS) a value hidden on its own: total honest hashing, and the honest-only
    part that no premise credits."""
    leaf = blocks(FRAME_LEAF_PREFIX + n_bytes)
    mask = MASK_REGS / R_WORDS
    hm = 3 * PRF_SHARE + 2
    mech = {
        "hm96 + (a) over its own frame leaf": {"total": leaf + hm + mask, "honest_only": mask},
        "hm96 unforced over its own frame leaf": {"total": leaf + hm + mask, "honest_only": leaf + hm + mask},
        "pad (b = x xor PRF), x <= 64 bytes": {"total": 1 + PAD_XOR_REGS / R_WORDS, "honest_only": PAD_XOR_REGS / R_WORDS},
        "keyed digest f(S, pos || x)": {"total": 1.0, "honest_only": 0.0},
        "keyed leaf (the value's leaf from S)": {"total": float(blocks(FRAME_LEAF_PREFIX + n_bytes) - 1), "honest_only": 0.0},
        "own frame leaf, unhidden, in a binary tree": {"total": float(leaf + NODE), "honest_only": 0.0},
    }
    for v in mech.values():
        v["macs_for_1pct_of_lead_h"] = 100 * v["total"] / LEAD_C_MAC
    return {"value_bytes": n_bytes, "lead_compressions_a_mac": LEAD_C_MAC, "mechanisms": mech}


def sketch(pv: dict) -> dict:
    out = {}
    for b in (1, 4, 8):
        for s in (1, 2, 4, 8, 10, 16, 32, 64, 256):
            base = b / (WORD_BYTES * s)
            row = {"in the words tree (L1)": base}
            for name, v in pv["mechanisms"].items():
                if name.startswith("own frame leaf") or name.startswith("hm96 unforced"):
                    continue
                row[f"each hidden on its own: {name}"] = base + v["total"] / (s * MACS_WORD) / LEAD_C_MAC
            out[f"{b} B a sketch, one sketch every {s} words"] = {
                "h_over_lead": row, "h": {k: {r: x * h for r, h in H_LEAD.items()} for k, x in row.items()}}
    return out


def rounds() -> dict:
    out = {}
    for point, (fl, al) in L1_TERM.items():
        per_r = {R: {"floor": fl * (R * CALLS + 1) / (CALLS + 1), "alone": al * (R * CALLS + 1) / (CALLS + 1)}
                 for R in (1, 2, 4, 8, 16, 64)}
        lead = LEAD_T1[point]
        room = None if lead is None else BUDGET - lead
        out[point] = {"L1 + (a), % of stock": per_r, "keyed leaf, % of stock": 0.0,
                      "max R at 0.1% (floor / alone)": [max(0, int((TARGET * (CALLS + 1) / x - 1) / CALLS)) for x in (fl, al)],
                      "max R in T1's room beside the lead (floor / alone)":
                          None if room is None else [max(0, int((room * (CALLS + 1) / x - 1) / CALLS)) for x in (fl, al)]}
    return out


def _solve_mod(m: list[list[int]], y: list[int], p: int) -> list[int] | None:
    n = len(m)
    a = [row[:] + [y[i]] for i, row in enumerate(m)]
    for c in range(n):
        piv = next((r for r in range(c, n) if a[r][c] % p), None)
        if piv is None:
            return None
        a[c], a[piv] = a[piv], a[c]
        inv = pow(a[c][c], p - 2, p)
        a[c] = [x * inv % p for x in a[c]]
        for r in range(n):
            if r != c and a[r][c]:
                f = a[r][c]
                a[r] = [(x - f * z) % p for x, z in zip(a[r], a[c])]
    return [a[i][n] for i in range(n)]


def leak(seed: int = 7, trials: int = 8) -> dict:
    """Recover an S-step chain's running words, and its partials, from S published linear sketches with public coins."""
    rng = random.Random(seed)
    res = []
    for t in range(trials):
        S = (8, 16, 32, 64)[t % 4]
        a = [[rng.randint(-128, 127) for _ in range(32)] for _ in range(S)]
        b = [[rng.randint(-128, 127) for _ in range(32)] for _ in range(S)]
        kappa = [sum(x * y for x, y in zip(a[i], b[i])) for i in range(S)]
        w, acc = [], 0
        for k in kappa:
            acc += k
            w.append(acc)
        r = [[rng.randrange(P61) for _ in range(S)] for _ in range(S)]
        sk = [sum(ri * wi for ri, wi in zip(row, w)) % P61 for row in r]
        got = _solve_mod(r, sk, P61)
        signed = None if got is None else [g - P61 if g > P61 // 2 else g for g in got]
        rec = None if signed is None else [signed[0]] + [signed[i] - signed[i - 1] for i in range(1, S)]
        res.append({"steps": S, "words_recovered": signed == w, "partials_recovered": rec == kappa})
    per_unit = {f"{b} B a sketch every {s} words": {"bits_a_unit": UNIT_WORDS // s * 8 * b,
                                                    "share_of_unit_output_bits": (UNIT_WORDS // s * 8 * b) / 262_144}
                for b in (4, 8) for s in (1, 8, 32, 256)}
    return {"recovery": res, "all_recovered": all(x["words_recovered"] and x["partials_recovered"] for x in res),
            "published_bits": per_unit}


def main() -> None:
    out_path = Path(sys.argv[1])
    pv = per_value()
    out = {"ands": ands(), "per_value": pv, "sketch": sketch(pv), "rounds": rounds(), "leak": leak()}
    out_path.write_text(json.dumps(out, indent=1, sort_keys=False))
    a = out["ands"]["a_word"]
    print("in-circuit ANDs a 32-bit word (x SHA-512):")
    for k, v in a.items():
        print(f"  {k:48s} {v['ands_a_word']:9.1f}  {v['over_sha']:.3f}  {v['label']}")
    print("drawn unit (8,192 words), ANDs:")
    for k, v in out["ands"]["a_drawn_unit"].items():
        print(f"  {k:48s} {v:14,.0f}")
    print("a 4-byte value hidden on its own, compressions (total / honest-only) and MACs a value for 1% of the lead's h:")
    for k, v in pv["mechanisms"].items():
        print(f"  {k:48s} {v['total']:6.3f} / {v['honest_only']:6.3f}  {v['macs_for_1pct_of_lead_h']:12,.0f}")
    print("sketch h over the lead's (4 B a sketch):")
    for k, v in out["sketch"].items():
        if k.startswith("4 B"):
            print(f"  {k}: " + ", ".join(f"{n.split(': ')[-1][:18]} {x:.3f}" for n, x in v["h_over_lead"].items()))
    print("chained rounds:", json.dumps(out["rounds"], indent=None)[:2000])
    print("leak: all recovered =", out["leak"]["all_recovered"])


if __name__ == "__main__":
    main()
