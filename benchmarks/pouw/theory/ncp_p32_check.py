#!/usr/bin/env python3
"""P32's CPU check: the fewest issue slots an off-tensor instruction route needs for one of NCP's k32 checkpoint words,
against the counts `ncp_time.py`'s `products` reading charges (IDP.4A 8 a word, DFMA 16, IMAD 16, FFMA 32, HFMA2 64).

NCP's operands (`verity/core/protocols/pouw/schemes/ncp.py`): A and B int7 in [−64, 63]; E₁ and F₁ in [−31, 32]; R a
signed permutation; E₂ = E₁R, F₂ = −RF₁, E₃ = E₁ + E₂, F₃ = F₁ + F₂; X = [A + E₁ | A + E₂ | −(A + E₃)] and
Y = [B + F₁; B + F₂; B + F₃]. A word is a dot product's running sum, read every 32 along the stacked k.

Three searches, each able to refute a count:

1. Byte range. Which ±1 forms of two operand entries fit an IDP.4A byte (s8, or u8 after route U's β = 128) over the
   whole domain. None does, so an IDP.4A factor is one entry, and no bilinear saving (Strassen, Winograd) fits it. Forms
   over the data alone fit a byte, but a route that takes A·Bᵀ from advice still pays two noise products a position,
   (A + E)·F and E·B, which no number of Strassen levels inside a k32 step brings under one.
2. Packing. The most products f a wide instruction makes exactly in a chain of L, packing f operands 2^s apart: DFMA's
   53 bits, FFMA's 24, IMAD's 32 bits mod 2^32 (the top field read modulo), and an HFMA2 lane's 11.
3. Floors. The fewest issue slots a word over f, L and Strassen levels l inside the k32 step. Each floor counts:
   - the product instructions, (32/f)·(7/8)^l;
   - at least one write a field a chain, plus a conversion unless the chain starts at a magic constant (two bits less
     headroom);
   - Strassen's C-side combination, at least one add a C entry a level, packed when the fields hold a k32 sum.
   Strassen's operand sums are made once a row or column tile and shared along the other side, so they count nothing.
   A floor under the products reading's count refutes it as a lower bound; `ncp_time.py`'s `floor` reading takes these
   floors.

  uv run python benchmarks/pouw/theory/ncp_p32_check.py [--json OUT] [--mutations] [--verdict PATH]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from exact_kit.checks import Checks

INT7 = (-64, 63)
NOISE = (-31, 32)
WORD_K = 32
#: The products reading's issue slots a word (`ncp_time.py` ROUTES[...]["products"]).
READING = {"idp4a": 8, "dfma": 16, "ffma": 32, "imad": 16, "hfma2": 64}
#: Exact integer magnitudes: DFMA 2^53, FFMA 2^24, an HFMA2 lane 2^11; IMAD is exact mod 2^32.
BITS = {"dfma": 53, "ffma": 24, "hfma2": 11}


def add(u, v):
    return (u[0] + v[0], u[1] + v[1])


def neg(u):
    return (-u[1], -u[0])


def signed_perm(u):
    return (min(u[0], -u[1]), max(u[1], -u[0]))


def blocks(data=INT7, noise=NOISE):
    """X's and Y's three blocks as intervals (the scheme's definitions; Y's mirror X's under F₂ = −RF₁)."""
    e2 = signed_perm(noise)
    e3 = add(noise, e2)
    x = [add(data, noise), add(data, e2), neg(add(data, e3))]
    y = [add(data, noise), add(data, neg(e2)), add(data, add(noise, neg(e2)))]
    return x, y


def mag(u) -> int:
    return max(abs(u[0]), abs(u[1]))


def fits_byte(u) -> bool:
    """Some byte reading holds every value: s8, or u8 after a fixed bias."""
    return u[1] - u[0] + 1 <= 256


def max_fields(kind: str, v: int, chain: int, magic: bool = False) -> tuple[int, int]:
    """(f, s): the most products a `kind` instruction holds exactly, f partial sums of `chain` products each of
    magnitude <= v, packed 2^s apart (0 fields when one product doesn't fit). `magic`: the chain starts at 1.5·2^(m−1),
    so its bits are the packed integer with no conversion, at two bits less headroom."""
    total = v * chain
    s = total.bit_length() + 1
    best = (0, 0)
    for f in range(1, 9):
        if kind == "imad":
            ok = (32 - s * (f - 1) >= s) and (s * (f - 1) + 8 <= 32)
        else:
            ok = total * sum(2 ** (s * i) for i in range(f)) <= 2 ** (BITS[kind] - (2 if magic else 0))
        if not ok:
            break
        best = (f, s if f > 1 else 0)
    return best


def floor_route(kind: str, p0: int, levels=range(0, 6)) -> dict:
    """The fewest issue slots a word for `kind`, over Strassen levels l, fields f and chains (see the module docstring)."""
    best = None
    word_sum = WORD_K * p0
    for l in levels:
        p = p0 * 4 ** l
        tile = WORD_K // 2 ** l
        if tile < 1:
            break
        m_per_word = (7 / 4) ** l
        combos = sum((7 / 4) ** (j - 1) for j in range(1, l + 1))
        for chain, magic in ((c, m) for c in (1, 2, 4, 8, 16, 32) if c <= tile for m in (False, True)):
            if magic and kind == "imad":
                continue
            f, s = max_fields(kind, p, chain, magic)
            if f == 0:
                continue
            prod = (WORD_K / f) * (7 / 8) ** l
            conv = 0 if magic or kind == "imad" else 1
            whole = chain == tile and (f == 1 or s >= word_sum.bit_length() + 1)
            if kind == "imad" and f == 1:
                extract = 0.0
            elif whole:
                extract = (conv + f) / f
            else:
                extract = m_per_word * (tile / chain) * (conv + f) / f
            comb = combos / f if whole else combos
            n = prod + extract + comb
            if best is None or n < best["slots"] - 1e-12:
                best = {"slots": n, "levels": l, "fields": f, "spacing": s, "chain": chain, "magic": magic,
                        "products": prod, "extract": extract, "combine": comb}
    return best


def floors(x, y) -> dict:
    p0 = max(mag(u) for u in x) * max(mag(v) for v in y)
    out = {"idp4a": {"slots": 8.0, "levels": 0, "fields": 4, "why": "4 byte products an IDP.4A; its sums leave the byte"}}
    for kind in ("dfma", "ffma", "imad"):
        out[kind] = floor_route(kind, p0)
    nib = max(mag(v) for v in y) * 15
    out["hfma2"] = {"slots": 1.5 * WORD_K, "levels": 0, "fields": 0,
                    "why": f"an int8 product is two nibble lanes (|x_lo·y| <= {nib} < 2^11, a chain of 1): one HFMA2 a "
                           "product, and at least one two-lane accumulate every two"}
    return out


def main() -> int:
    checks = Checks()
    out = checks.args[checks.args.index("--json") + 1] if "--json" in checks.args else None
    x, y = blocks()
    checks.check("NCP's blocks: X [-95,95] [-96,95] [-127,127] and Y alike; every entry is s8, and X + 128 is u8",
                 x == [(-95, 95), (-96, 95), (-127, 127)] and y == x
                 and all(-128 <= u[0] and u[1] <= 127 for u in x + y) and all(0 <= u[0] + 128 and u[1] + 128 <= 255
                                                                              for u in x), f"X {x}, Y {y}")
    p0 = max(mag(u) for u in x) * max(mag(v) for v in y)
    checks.check("a k32 step's partial sum stays under 2^19 (exact in FP32, and 20 bits a packed field)",
                 WORD_K * p0 < 2 ** 19, f"32 x {p0} = {WORD_K * p0}")

    pairs = [(su, sv, i, j) for i in range(3) for j in range(3) for su in (1, -1) for sv in (1, -1)]
    fit = [(i, j) for su, sv, i, j in pairs for blk in (x, y)
           if fits_byte(add(blk[i] if su > 0 else neg(blk[i]), blk[j] if sv > 0 else neg(blk[j])))]
    checks.check("no ±u ± v of two operand entries fits an IDP.4A byte over the domain, so no bilinear saving fits it",
                 not fit, f"narrowest: {min(add(b[i], b[j])[1] - add(b[i], b[j])[0] + 1 for b in (x, y) for i in range(3) for j in range(3))} values")
    data_sum = add(INT7, INT7)
    checks.check("data-only sums fit a byte, but the advice route still pays two noise products a position, more than "
                 "one even after five Strassen levels", fits_byte(data_sum) and 2 * (7 / 8) ** 5 > 1,
                 f"a + a' in {data_sum}; 2 x (7/8)^5 = {2 * (7 / 8) ** 5:.3f}")

    table = {kind: {L: max_fields(kind, p0, L) for L in (1, 2, 4, 8, 16, 32)} for kind in ("dfma", "ffma", "imad")}
    nib = max(mag(v) for v in y) * 15
    table["hfma2"] = {1: max_fields("hfma2", p0, 1)}
    for kind, row in table.items():
        print(f"  {kind:6s} fields (spacing) by chain: " + ", ".join(f"L{L} {f} ({s})" for L, (f, s) in row.items()))
    checks.check("DFMA holds 3 exact products in chains up to 8 (spacing 2^18) and 2 in a whole k32 chain: P32's '2 a "
                 "DFMA' is not a product bound", table["dfma"][8][0] == 3 and table["dfma"][16][0] == 2
                 and table["dfma"][32][0] == 2, f"L8 {table['dfma'][8]}, L32 {table['dfma'][32]}")
    checks.check("FFMA holds one product, IMAD two in chains up to 2, an HFMA2 lane no whole int8 product",
                 all(f == 1 for f, _ in table["ffma"].values()) and table["imad"][2][0] == 2
                 and table["imad"][4][0] == 1 and table["hfma2"][1][0] == 0 and nib < 2 ** 11,
                 f"imad {table['imad']}, an HFMA2 nibble product <= {nib}")

    fl = floors(x, y)
    for kind, f in fl.items():
        print(f"  floor {kind:6s} {f['slots']:6.2f} issue slots a word (reading {READING[kind]})  "
              + ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in f.items() if k != "slots"))
    checks.check("IDP.4A's floor is the reading's 8 a word", fl["idp4a"]["slots"] == READING["idp4a"])
    checks.check("DFMA's floor is under the reading's 16 (Strassen inside the k32 step, two packed fields): the reading "
                 "is not a lower bound", fl["dfma"]["slots"] < READING["dfma"] - 0.5,
                 f"{fl['dfma']['slots']:.2f} at {fl['dfma']['levels']} levels, {fl['dfma']['fields']} fields")
    checks.check("FFMA's floor is under the reading's 32 (Strassen inside the k32 step)",
                 fl["ffma"]["slots"] < READING["ffma"] - 2, f"{fl['ffma']['slots']:.2f} at {fl['ffma']['levels']} levels")
    checks.check("IMAD's floor is above IDP.4A's on the same pipe, so IMAD stays dominated",
                 fl["imad"]["slots"] > fl["idp4a"]["slots"], f"{fl['imad']['slots']:.2f}")
    no_strassen = {k: floor_route(k, p0, levels=[0])["slots"] for k in ("dfma", "ffma")}
    checks.check("without Strassen, DFMA's floor is 3-field packing in chains of 8 at 16.0 (the reading's 16) and "
                 "FFMA's is 33", abs(no_strassen["dfma"] - 16.0) < 1e-9 and abs(no_strassen["ffma"] - 33.0) < 1e-9,
                 f"{no_strassen}")

    if checks.mutations:
        xs, ys = blocks(data=(-32, 31), noise=(-16, 16))
        fit2 = [(i, j) for b in (xs, ys) for i in range(3) for j in range(3) if fits_byte(add(b[i], b[j]))]
        checks.mutation("int6 data with [-16, 16] noise lets two-entry sums fit a byte (the claim's range bites)",
                        bool(fit2), f"{len(fit2)} block pairs fit")
        checks.mutation("a mantissa 3 bits short of FP64's holds only 2 fields in chains of 8: the 3-field claim rests "
                        "on the 53 bits", _fields_at(50, p0, 8) == 2, f"{_fields_at(50, p0, 8)} fields")
    if out:
        Path(out).write_text(json.dumps({"blocks": {"X": x, "Y": y}, "max_product": p0, "packing": table,
                                         "floors": fl, "no_strassen": no_strassen, "reading": READING},
                                        indent=1) + "\n")
    return checks.finish()


def _fields_at(bits: int, v: int, chain: int) -> int:
    old = BITS["dfma"]
    BITS["dfma"] = bits
    try:
        return max_fields("dfma", v, chain)[0]
    finally:
        BITS["dfma"] = old


if __name__ == "__main__":
    sys.exit(main())
