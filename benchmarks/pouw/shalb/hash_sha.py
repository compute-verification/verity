"""HASH's SHA lane: h's floor under ShaWrites, the SHA candidates priced at the lead's points, and exact M0 adversaries.

    python3 hash_sha.py LEAD.json [--verdict PATH] [--json OUT]

LEAD.json is hash_sha_lead.py's output (the lead's h, r, T1 terms and T2 parts at the three Llama-3.2-1B points).
Units: W1, 1/32 of a register write; stock is 1 W1 a useful MAC; h is honest hashing W1 a stock MAC; a word is one
32-step accumulator value (one register write of the IMMA).

- F. The floor. Under ShaWrites a compression absorbs at most 128 message bytes and costs at least R registers (3,510
  from a constant midstate, 3,532 chained), so a committed bit costs at least R/32 W1 of hashing. If the only credit
  past hashing is DistinctWrites' (32 W1 a committed distinct 32-bit register, so at most 1 W1 a committed bit) and the
  prover may pick the data (M0), T1 <= 1% needs (r - 0.01) W1 a stock MAC of credit, hence h >= (R/32)(r - 0.01).
  Chunk shapes move R between 3,515 (16 KiB) and 3,532.
- P. The candidates at the three points: committed bits a word, h, honest and credited writes a word under
  DistinctWrites alone and under each candidate's premise, T1, and T2 (sync, async; an estimate: the lead's measured and
  estimated parts with the words' export and hashing scaled by the committed bits, and the honest's added integer
  writes at H200's 64 int32 lanes a clock an SM).
- L. Additive combining is linear: the sum of two rows' words is the word of the summed rows, one IMMA output.
  xorrot-m commits the XOR of the m words each rotated by a distinct public amount (the first by 0): honest adds an
  SHF a word but the first, which a packed lane can't take.
- X. XOR of m consecutive words of a cell (xor-m) at M0, the lead's later steps' increments: an exact adversary that
  packs three cells' low 21 bits in 64 bits around a per-block base and reads the high bits off the lanes' bit 20.
  Bit for bit against honest, with its register writes counted instruction by instruction against honest's
  m + ceil((m - 1)/2).
- T. Low-k truncation (trunc-k) at M0 for k <= 4: an exact adversary in four 8-bit lanes a register, bit for bit, with
  its writes against honest's IMMA write and its LEA tree packing 32/k residues into a message register.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exact_kit"))
from exact_kit.checks import Checks  # noqa: E402

M32 = (1 << 32) - 1
M64 = (1 << 64) - 1
W1_PER_REG = 32
BLOCK = 128
R_FIRST, R_CHAIN, R_PAD = 3510, 3532, 2254
CHUNK_K, CHUNK_EXTRA, CHUNK_HEAD = 2213, 5, 23
T1_BAR = 0.01
INT_RATE = 64 * 132 * 1.98e9
POINTS = ("t512", "t512r512", "t8192")
PERIODS = {"week": 7 * 86400, "month": 30 * 86400, "year": 365 * 86400}
H200_MACS = 9.89e14 * 0.4
DC_GPUS = 100_000
WORLD_MACS_PER_YEAR = 4.68e30


def r_row(row: int) -> float:
    """ShaWrites' registers a compression credited for a words chunk of `row` bytes (the card's fit of the forms)."""
    return R_CHAIN - CHUNK_K / (row / BLOCK + CHUNK_EXTRA)


def w1_per_bit(regs: float) -> float:
    return regs * W1_PER_REG / (BLOCK * 8)


def part_f(checks: Checks, lead: dict) -> dict:
    p = lead["points"]["t8192"]
    r, h = p["r"], p["h"]
    words_b = r * 4 / 32
    rest_b = p["bytes_per_mac"] - words_b
    floor_first = w1_per_bit(R_FIRST) * (r - T1_BAR)
    floor_chain = w1_per_bit(R_CHAIN) * (r - T1_BAR)
    floor_r1 = w1_per_bit(R_FIRST) * (1 - T1_BAR)
    rows = {}
    for kib in (16, 64, 256, 1024, 4096, 16384):
        row = kib * 1024
        hw = r_row(row) * W1_PER_REG / BLOCK
        rows[kib] = {"R": r_row(row), "h_words": words_b * hw,
                     "chunk_head_t1": words_b / row * CHUNK_HEAD * W1_PER_REG}
    checks.check("F: the lead's h at 256 KiB words is its words' bytes at the card's 256 KiB form plus A's pre-noise "
                 "bytes", abs(rows[256]["h_words"] - h) / h < 0.003,
                 f"h {h:.3f}, words {rows[256]['h_words']:.3f} ({words_b:.5f} B a MAC at r {r:.4f}), "
                 f"pre-noise {rest_b:.2e} B a MAC")
    checks.check("F: a committed bit costs at least R/32 W1 of hashing: 109.69 from a constant midstate, 110.38 chained",
                 abs(w1_per_bit(R_FIRST) - 109.6875) < 1e-9 and abs(w1_per_bit(R_CHAIN) - 110.375) < 1e-9,
                 f"{w1_per_bit(R_FIRST):.4f}, {w1_per_bit(R_CHAIN):.4f}")
    checks.check("F: the floor h >= (R/32)(r - 1%) at the lead's r is within 1.2% of the lead; at r = 1 it is 108.6, "
                 "so no design in the class cuts h by even 1.2x, let alone 10x",
                 h / floor_chain < 1.012 and h / floor_r1 < 1.18 and floor_r1 > 10 * 10.0,
                 f"floor {floor_first:.2f} (first-block R) / {floor_chain:.2f} (chained) at r {r:.4f}; lead {h:.2f} "
                 f"= {h / floor_chain:.4f}x chained; r = 1: {floor_r1:.2f} ({h / floor_r1:.3f}x below the lead)")
    span = max(v["h_words"] for v in rows.values()) / min(v["h_words"] for v in rows.values())
    checks.check("F: chunk shapes from 16 KiB to 16 MiB move h by at most 0.5% and from 256 KiB by at most 0.04%",
                 span < 1.005 and abs(rows[16384]["h_words"] / rows[256]["h_words"] - 1) < 0.0004,
                 ", ".join(f"{k} KiB {v['h_words']:.2f} (R {v['R']:.1f}, chunk_head {v['chunk_head_t1']:.4%})"
                           for k, v in rows.items()))
    return {"r": r, "h": h, "words_bytes_per_mac": words_b, "rest_bytes_per_mac": rest_b, "floor_first": floor_first,
            "floor_chain": floor_chain, "floor_r1": floor_r1, "rows": rows}


def trunc_honest(k: int) -> float:
    """Honest's writes a word for trunc-k: the IMMA's, then the message register's share. At 16 bits one PRMT takes
    two words' low halves; at 8 a PRMT tree takes four words' low bytes in three; below, each word is masked and a LEA
    tree merges 32/k of them in 32/k - 1."""
    fields = 32 // k
    if k == 16:
        return 1 + 1 / 2
    if k == 8:
        return 1 + 3 / 4
    return 1 + 1 + (fields - 1) / fields


CANDIDATES = {
    "lead": {"bits": 32, "honest": 1.0, "credit_dw": 1.0, "credit_p": 1.0, "premise": "DistinctWrites"},
    **{f"trunc-{k}": {"bits": k, "honest": trunc_honest(k), "credit_dw": k / 32, "credit_p": 1 + k / 32,
                      "premise": f"TruncWrites_{k}"} for k in (16, 8, 4)},
    **{f"xor-{m}": {"bits": 32 / m, "honest": 1 + math.ceil((m - 1) / 2) / m, "credit_dw": 1 / m,
                    "credit_p": 1 + math.ceil((m - 1) / 2) / m, "premise": f"XorWrites_{m}"}
       for m in (2, 4, 8, 11, 16, 24, 31)},
    **{f"xorrot-{m}": {"bits": 32 / m, "honest": 1 + (m - 1) / m + math.ceil((m - 1) / 2) / m, "credit_dw": 1 / m,
                       "credit_p": 1 + (m - 1) / m + math.ceil((m - 1) / 2) / m, "premise": f"RotXorWrites_{m}"}
       for m in (11, 16, 24)},
    "sum-2": {"bits": 16, "honest": 1.5, "credit_dw": 0.5, "credit_p": 0.5, "premise": "none (linear, part L)"},
}


def part_p(checks: Checks, lead: dict, f: dict) -> dict:
    out = {}
    for name, c in CANDIDATES.items():
        out[name] = {"bits": c["bits"], "premise": c["premise"], "points": {}}
        for pt in POINTS:
            p = lead["points"][pt]
            r = p["r"]
            frac = c["bits"] / 32
            h_words = r * 4 / 32 * r_row(256 * 1024) * W1_PER_REG / BLOCK
            h = p["h"] - (1 - frac) * h_words
            base = p["t1"] - p["terms"].get("chunk_head", 0.0)
            chunk = p["terms"].get("chunk_head", 0.0) * frac
            t1_dw = base + chunk + r * (c["honest"] - c["credit_dw"])
            t1_p = base + chunk + r * (c["honest"] - c["credit_p"])
            words = p["stock_macs"] * r / 32
            int_ms = words * (c["honest"] - 1.0) / INT_RATE * 1e3
            parts = dict(p["extra_parts_ms"])
            extra = sum(parts.values()) - parts["export"] * (1 - frac) + int_ms
            t2 = {e: (s + extra + p["hash_ms"] * frac) / s for e, s in p["stock_ms"].items()}
            hashing = {per: {"h200": h * H200_MACS * sec, "datacenter": h * H200_MACS * sec * DC_GPUS,
                             "world": h * WORLD_MACS_PER_YEAR * sec / PERIODS["year"]} for per, sec in PERIODS.items()}
            out[name]["points"][pt] = {"h": h, "cut": p["h"] / h, "t1_dw": t1_dw, "t1_premise": t1_p,
                                       "int_ms": int_ms, "t2": t2, "h200_hash_w1": hashing}
    lead_row = out["lead"]["points"]
    checks.check("P: the lead row reproduces the card's T1 and T2 at every point",
                 all(abs(lead_row[pt]["t1_dw"] - lead["points"][pt]["t1"]) < 1e-12
                     and all(abs(lead_row[pt]["t2"][e] - lead["points"][pt]["t2"][e]) < 1e-9
                             for e in lead["points"][pt]["t2"]) for pt in POINTS),
                 "; ".join(f"{pt} T1 {lead_row[pt]['t1_dw']:.4%} T2 {lead_row[pt]['t2']['sync']:.2f}/"
                           f"{lead_row[pt]['t2']['async']:.2f}" for pt in POINTS))
    fails = [n for n, v in out.items() if n != "lead" and all(v["points"][pt]["t1_dw"] > T1_BAR for pt in POINTS)]
    checks.check("P: under DistinctWrites alone every candidate that commits fewer bits a word fails T1 at every point",
                 sorted(fails) == sorted(n for n in out if n != "lead"),
                 ", ".join(f"{n} {out[n]['points']['t8192']['t1_dw']:.1%}" for n in out if n != "lead"))
    tr = [n for n in out if n.startswith("trunc-") and n != "trunc-16"]
    checks.check("P: trunc-k below 16 bits fails T1 even when its premise credits the word: honest's LEA tree "
                 "packing 32/k residues is uncredited", all(out[n]["points"][pt]["t1_premise"] > T1_BAR
                                                             for n in tr for pt in POINTS),
                 ", ".join(f"{n} {out[n]['points']['t8192']['t1_premise']:.1%}" for n in tr))
    ten = [n for n in out if n.startswith("xor") and out[n]["points"]["t8192"]["cut"] >= 10]
    checks.check("P: xor-m and xorrot-m cut h 10x from m = 11, holding T1 and T2 at every point if their premise holds",
                 ten and min(int(n.split("-")[1]) for n in ten) == 11
                 and all(out[n]["points"][pt]["t1_premise"] <= T1_BAR and max(out[n]["points"][pt]["t2"].values()) <= 100
                         for n in ten for pt in POINTS),
                 "; ".join(f"{n} h {out[n]['points']['t8192']['h']:.2f} ({out[n]['points']['t8192']['cut']:.1f}x), "
                           f"T2 {out[n]['points']['t8192']['t2']['sync']:.2f}/{out[n]['points']['t8192']['t2']['async']:.2f}"
                           for n in ten))
    return out


def part_l(checks: Checks, rng: np.random.Generator) -> None:
    a = rng.integers(-127, 128, size=(4, 64), dtype=np.int64)
    b = rng.integers(-127, 128, size=(64, 8), dtype=np.int64)
    w = (a @ b) & M32
    summed = ((a[0] + a[1]) @ b) & M32
    checks.check("L: the sum of two rows' words is one IMMA output on the summed rows (mod 2^32), so sum-m credits one "
                 "write for m words", np.array_equal((w[0] + w[1]) & M32, summed), None)


# The lead's later k32 steps at M0 (data 0): lanes 28-30 hold three odd coins in ±127, weighted A' = (127, 1, 1) on even
# rows and (1, 127, 1) on odd, a (64-row tile, step, column); lane 31 is 1 + b_t(i), a coin bit a row and step, against
# the column code, modelled as uniform s8. The first step (row tags and coins) only sets the base W0, uniform in Z/2^32.
A_ROT = np.array([[127, 1, 1], [1, 127, 1]], dtype=np.int64)


def m0_draw(rng: np.random.Generator, rows: int, cols: int, steps: int) -> dict:
    s = 2 * rng.integers(-64, 64, size=(steps, 3, cols), dtype=np.int64) + 1
    g = rng.integers(-127, 128, size=(steps, cols), dtype=np.int64)
    b = rng.integers(0, 2, size=(steps, rows), dtype=np.int64)
    w0 = rng.integers(0, 1 << 32, size=(rows, cols), dtype=np.int64)
    coin = np.einsum("pl,tlj->tpj", A_ROT, s)
    par = np.arange(rows) % 2
    delta = coin[:, par, :] + (1 + b)[:, :, None] * g[:, None, :]
    words = (w0[None] + np.cumsum(delta, axis=0)) & M32
    return {"s": s, "g": g, "b": b, "w0": w0, "coin": coin, "delta": delta, "words": words, "par": par}


DELTA_MAX = 127 * 127 + 127 + 127 + 2 * 127


def xor_honest_writes(m: int) -> int:
    return m + math.ceil((m - 1) / 2)


def xor_adversary(d: dict, m: int) -> tuple[np.ndarray, dict]:
    """The packed M0 adversary for xor-m on three cells (one row, three columns) a 64-bit pair, block by block."""
    rows, cols = d["w0"].shape
    steps = d["delta"].shape[0]
    assert cols % 3 == 0 and steps % m == 0 and m * DELTA_MAX < 1 << 19
    grp = cols // 3
    lanes = np.arange(3)
    pk = d["delta"].reshape(steps, rows, grp, 3)
    packed = (pk * (1 << (21 * lanes))).sum(axis=-1).view(np.uint64)
    base = d["w0"].reshape(rows, grp, 3).astype(np.uint64)
    out = np.zeros((steps // m, rows, grp, 3), dtype=np.uint64)
    mask20, mask21 = np.uint64((1 << 20) - 1), np.uint64((1 << 21) - 1)
    for blk in range(steps // m):
        lo = base & mask20
        beta = ((~base) & np.uint64(1 << 19)) << np.uint64(1)
        u = lo + beta
        g2 = ((base + np.uint64(1 << 19)) & np.uint64(M32)) >> np.uint64(20)
        q = (u << (np.uint64(21) * lanes.astype(np.uint64))).sum(axis=-1, dtype=np.uint64)
        y = np.zeros_like(q)
        for t in range(blk * m, (blk + 1) * m):
            q = q + packed[t]
            y ^= q
        ylanes = (y[..., None] >> (np.uint64(21) * lanes.astype(np.uint64))) & mask21
        par = (ylanes >> np.uint64(20)) & np.uint64(1)
        par_other = par ^ np.uint64(m & 1)
        hi = (np.where(par == 1, g2, 0) ^ np.where(par_other == 1, (g2 - np.uint64(1)) & np.uint64(M32), 0))
        out[blk] = ((hi << np.uint64(20)) & np.uint64(M32)) | (ylanes & mask20)
        last = (q[..., None] >> (np.uint64(21) * lanes.astype(np.uint64))) & mask21
        top = np.where(((last >> np.uint64(20)) & np.uint64(1)) == 1, g2, (g2 - np.uint64(1)) & np.uint64(M32))
        base = ((top << np.uint64(20)) & np.uint64(M32)) | (last & mask20)
    writes = xor_adversary_writes(m)
    return out.reshape(steps // m, rows, cols), writes


def xor_adversary_writes(m: int) -> dict:
    """Register writes of xor_adversary a group of three cells and a block (a 64-bit operation is two; a predicate
    write counts as one), and the per-(tile, step, three columns) forming of its four packed increments (loads,
    coin bytes, both parities' coin terms, both b variants, packing), amortized over the tile's 64 rows.

    init: a lane's Lo, beta and U (3), its G'' = (base + 2^19) >> 20 (2), and the pair (3). end: three lanes out of the
    pair (5), then a lane's low 20 bits, bit-20 predicate, G'' - 1, the high part and the merge (6 at even m, 5 at odd).
    rebase: the last pair's lanes (5) and a lane's predicate, select, mask and merge (4)."""
    init = 3 * 3 + 3 * 2 + 3
    steps = 2 * m
    fold = 2 * math.ceil((m - 1) / 2)
    end = 5 + 3 * (6 if m % 2 == 0 else 5)
    rebase = 5 + 3 * 4
    forming = 3 + 9 + 3 + 12 + 12 + 16
    per_group = init + steps + fold + end + rebase
    per_register = per_group / 3 + forming * m / 64 / 3
    return {"init": init, "steps": steps, "fold": fold, "end": end, "rebase": rebase, "forming_per_tile_step": forming,
            "per_register": per_register, "honest_per_register": xor_honest_writes(m),
            "ratio": per_register / xor_honest_writes(m)}


def part_x(checks: Checks, rng: np.random.Generator) -> dict:
    out = {}
    for m in (4, 8, 16, 24, 30, 31):
        steps = m * (60 // m if m <= 30 else 2)
        d = m0_draw(rng, rows=64, cols=24, steps=steps)
        honest = np.bitwise_xor.reduce(d["words"].reshape(steps // m, m, 64, 24), axis=1)
        adv, w = xor_adversary(d, m)
        ok = np.array_equal(adv.astype(np.int64), honest)
        out[m] = w | {"exact": bool(ok), "blocks": steps // m}
        checks.check(f"X: xor-{m} at M0, the packed adversary matches honest's XOR bit for bit "
                     f"({steps // m} blocks x 64 rows x 24 columns)", ok,
                     f"{w['per_register']:.2f} writes a register against honest's {w['honest_per_register']} "
                     f"({w['ratio']:.2f}x)")
    checks.check("X: the lanes hold every block: m * max|delta| < 2^19 up to m = 31", 31 * DELTA_MAX < 1 << 19,
                 f"max|delta| {DELTA_MAX}, 31x = {31 * DELTA_MAX}")
    beat = [m for m in out if out[m]["ratio"] < 1]
    checks.check("X: this adversary pays more than honest at every m it can pack (m <= 31), so it does not refute "
                 "XorWrites_m", not beat, ", ".join(f"m {m}: {out[m]['ratio']:.2f}x" for m in out))
    running = 2 / 3 + out[16]["forming_per_tile_step"] / 64 / 3
    checks.check("X: its running sums alone cost less than a write a word, so a premise that charges each uncommitted "
                 "word a register write (DistinctWrites extended to intermediates) is false at M0; XorWrites_m must "
                 "charge the whole XOR", running < 1, f"{running:.3f} writes a word")
    return out


TRUNC_FORMING = 4 + 4 + 4 * 4 + 16 + 16 + 12


def trunc_adversary(d: dict, k: int) -> tuple[np.ndarray, dict]:
    """The M0 adversary for low-k residues: four cells a 32-bit register in 8-bit lanes, an add and a lane mask a step
    (each lane stays below 2^(k+1) before the mask, so none carries into the next), and the residues of 32/k cells
    merged into a message register by 32/k/4 - 1 LEAs. Its forming a (tile, step, four columns): the coin and code
    loads (4 + 4), coin bytes (16), both parities' coin terms (16), both b variants (16) and four byte packings (12),
    amortized over the tile's 64 rows."""
    rows, cols = d["w0"].shape
    steps = d["delta"].shape[0]
    assert cols % 4 == 0 and k <= 4
    mk = (1 << k) - 1
    lane_mask = sum(mk << (8 * i) for i in range(4))
    pk = (d["delta"] & mk).reshape(steps, rows, cols // 4, 4)
    packed = (pk << (8 * np.arange(4))).sum(axis=-1)
    reg = ((d["w0"] & mk).reshape(rows, cols // 4, 4) << (8 * np.arange(4))).sum(axis=-1)
    out = np.zeros((steps, rows, cols), dtype=np.int64)
    for t in range(steps):
        reg = (reg + packed[t]) & lane_mask
        out[t] = ((reg[..., None] >> (8 * np.arange(4))) & mk).reshape(rows, cols)
    fields = 32 // k
    per_word = 2 / 4 + (fields // 4 - 1) / fields + TRUNC_FORMING / 64 / 4
    return out, {"per_word": per_word, "honest_per_word": trunc_honest(k), "credit_premise": 1 + k / 32}


def part_t(checks: Checks, rng: np.random.Generator) -> dict:
    out = {}
    for k in (1, 2, 4):
        d = m0_draw(rng, rows=64, cols=32, steps=60)
        adv, w = trunc_adversary(d, k)
        ok = np.array_equal(adv, d["words"] & ((1 << k) - 1))
        out[k] = w | {"exact": bool(ok)}
        checks.check(f"T: trunc-{k} at M0, the 8-bit-lane adversary matches honest's low {k} bits bit for bit, at "
                     f"{w['per_word']:.3f} writes a word against TruncWrites_{k}'s credit {w['credit_premise']:.3f}",
                     ok and w["per_word"] < w["credit_premise"] - 0.01,
                     f"honest {w['honest_per_word']:.3f}; refutes TruncWrites_{k}")
    return out


def main() -> int:
    checks = Checks()
    args = checks.args
    lead = json.loads(Path(args[0]).read_text())
    rng = np.random.default_rng(20)
    f = part_f(checks, lead)
    p = part_p(checks, lead, f)
    part_l(checks, rng)
    x = part_x(checks, rng)
    t = part_t(checks, rng)
    if "--json" in args:
        Path(args[args.index("--json") + 1]).write_text(json.dumps(
            {"floor": f, "candidates": p, "xor": {str(k): v for k, v in x.items()},
             "trunc": {str(k): v for k, v in t.items()}}, indent=1, sort_keys=True, default=float) + "\n")
    print("\ncandidate      bits  h@t8192  cut   T1 DW-only  T1 premise  T2 t512 s/a     T2 r512 s/a     T2 t8192 s/a")
    for n, v in p.items():
        q = v["points"]
        print(f"{n:<13} {v['bits']:5.2f} {q['t8192']['h']:8.2f} {q['t8192']['cut']:5.1f}x {q['t8192']['t1_dw']:10.3%} "
              f"{q['t8192']['t1_premise']:10.3%}  " + "  ".join(
                  f"{q[pt]['t2']['sync']:6.2f}/{q[pt]['t2']['async']:6.2f}" for pt in POINTS))
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
