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
- R. A bit-sliced M0 adversary for xorrot-m and xor-m, bit for bit, its writes counted operation by operation: 32
  cells a register, so rotation is free; an add over the low bits only; the step's coins uniform across the warp, so
  each increment bit is a predicated LOP3 case (the warp's uniform writes shared by its 32 threads, as at T 8,192)
  and the b bits one register a step.
- Y. A packed M0 adversary for xor-m centred once a segment instead of each block, bit for bit, its writes modelled
  part by part with predicate writes free (W1 today) and priced as register writes: the price of a predicate decides
  XorWrites_m.
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


def rot_amount(t: int) -> int:
    return (7 * t) & 31


def xorrot_honest_per_word(m: int) -> float:
    return 1 + (m - 1) / m + math.ceil((m - 1) / 2) / m


def xor_fold_writes(n: int) -> int:
    """Writes to fold n registers into one by 3-input LOP3s: the first takes three, each later one two."""
    return 0 if n == 0 else 1 if n <= 3 else 1 + math.ceil((n - 3) / 2)


LO = 21


def uniform_per_step(lo_bits: int) -> int:
    """The warp's uniform writes a step, a column and a parity: both increments, then each low bit's two case
    predicates."""
    return 4 + 2 * lo_bits


class Regs:
    """Bit-sliced registers of one thread, 32 cells a register, for every group at once; `w` counts one write."""

    def __init__(self) -> None:
        self.writes = 0

    def w(self, x: np.ndarray) -> np.ndarray:
        self.writes += 1
        return x & np.uint64(M32)


def bitslice(v: np.ndarray, bits: range) -> list[np.ndarray]:
    """v: (32 cells, groups) -> one register a bit, bit c of a register the cell c's."""
    cells = np.arange(32, dtype=np.uint64)[:, None]
    return [(((v.astype(np.uint64) >> np.uint64(i)) & np.uint64(1)) << cells).sum(axis=0, dtype=np.uint64)
            for i in bits]


def transpose32(r: Regs, a: list[np.ndarray]) -> list[np.ndarray]:
    """A 32x32 bit transpose by block swaps (register k's bit c to register c's bit k): at block size j, register k's
    high-j bits trade places with register k + j's low-j bits. PRMT swaps the 16- and 8-bit blocks (one write a
    register), a shift and a LOP3 select the smaller ones (two)."""
    a = list(a)
    j, msk = 16, 0x0000FFFF
    while j:
        mj = np.uint64(msk)
        nm = ~mj & np.uint64(M32)
        k = 0
        while k < 32:
            x, y = a[k], a[k + j]
            nx = (x & mj) | ((y << np.uint64(j)) & nm)
            ny = (y & nm) | ((x >> np.uint64(j)) & mj)
            if j < 8:
                r.writes += 2
            a[k], a[k + j] = r.w(nx), r.w(ny)
            k = ((k | j) + 1) & ~j
        j >>= 1
        msk ^= msk << j
    return a


def block_reach(d: dict, m: int) -> np.ndarray:
    """The largest |partial sum| of a block's increments, over its cells and blocks, a column."""
    steps = d["delta"].shape[0]
    run = np.cumsum(d["delta"].reshape(steps // m, m, *d["delta"].shape[1:]), axis=1)
    return np.abs(run).max(axis=(0, 1, 2))


def lo_bits_for(reach: int) -> int:
    """The narrowest low part whose middle half holds a block: reach < 2^(L - 2)."""
    return max(4, int(reach).bit_length() + 2)


def bitsliced_adversary(d: dict, m: int, lo_bits: int = LO, rotate: bool = True) -> tuple[np.ndarray, dict]:
    """The bit-sliced M0 adversary for xorrot-m (rotate) or xor-m. A group is 32 rows of one parity in one column, one cell a bit of a
    register; rotation is free. With L = lo_bits, a block's state is (H', lo', beta): W = H' 2^L + lo' - beta 2^(L-1),
    with lo' in the middle half of [0, 2^L) at the block's start, so the block's L-bit adds never wrap while each
    partial sum of the block's increments stays below 2^(L-2). The low L bits of W_t are lo' with bit L-1 flipped
    where beta, and its high 32 - L bits are H' - (beta and not lo'_(L-1)), H' xor (not lo'_(L-1) and R), where R is
    beta and the borrow chain of H'. A step: the b-bit register, the L-bit add (each bit's increment is 0, ~0, B or ~B
    by the step's coins and code, a predicated LOP3 a case; sum and carry), and the 32 - L high terms. A block: the
    rebase, the R chain, the fold of the low and constant terms (3-input LOP3s), and the transpose. Unrotated (xor-m),
    every step's high terms land on the same bits, so a block takes them at once: the XOR P of the block's lo'_(L-1),
    which the fold of bit L-1 computes anyway, gives each high bit j as (P' and R_j) xor H'_j, one write. Bit 0 of an
    increment that is odd for both b values adds as ~a with carry a, one write."""
    rows, cols = d["w0"].shape
    steps = d["delta"].shape[0]
    lo_n, hi_n = lo_bits, 32 - lo_bits
    assert rows == 64 and steps % m == 0 and m <= 32 and block_reach(d, m).max() < 1 << (lo_n - 2)
    top, sub = lo_n - 1, lo_n - 2
    cells = lambda x: x.reshape(32, 2 * cols)  # noqa: E731
    r = Regs()
    lo = bitslice(cells(d["w0"] & ((1 << lo_n) - 1)), range(lo_n))
    hi = bitslice(cells(d["w0"] >> lo_n), range(hi_n))
    beta = np.zeros(2 * cols, dtype=np.uint64)
    ones = np.uint64(M32)
    b_cells = d["b"].reshape(steps, 32, 2)
    out = np.zeros((steps // m, rows, cols), dtype=np.int64)
    per_step = per_block = 0.0
    for blk in range(steps // m):
        start = r.writes
        wtop = r.w(lo[top] ^ beta)
        beta_new = r.w(~(lo[top] ^ beta ^ lo[sub]))
        dn = r.w(beta & ~lo[top])
        up = r.w(beta_new & wtop)
        lo[top] = r.w(wtop ^ beta_new)
        c = r.w(up ^ dn)
        for j in range(hi_n):
            hj = hi[j]
            hi[j] = r.w(hj ^ c)
            if j < hi_n - 1:
                c = r.w(c & (hj ^ dn))
        beta = beta_new
        rch = [beta]
        for j in range(hi_n - 1):
            rch.append(r.w(rch[j] & ~hi[j]))
        terms: list[list[np.ndarray]] = [[] for _ in range(32)]
        const: list[dict] = [{} for _ in range(32)]
        hiterm = [0] * 32
        x = [np.zeros(2 * cols, dtype=np.uint64) for _ in range(32)]
        steps_start = r.writes
        for t in range(m):
            step = blk * m + t
            rho = rot_amount(t) if rotate else 0
            bb = r.w((b_cells[step].astype(np.uint64) << np.arange(32, dtype=np.uint64)[:, None]).sum(
                axis=0, dtype=np.uint64))
            bmask = np.concatenate([np.repeat(bb[p], cols) for p in range(2)])
            dv0 = np.concatenate([d["coin"][step, p, :] + d["g"][step] for p in range(2)])
            dv1 = np.concatenate([d["coin"][step, p, :] + 2 * d["g"][step] for p in range(2)])
            bcell = cells(d["b"][step][:, None].repeat(cols, axis=1))
            assert np.array_equal(cells(d["delta"][step]), np.where(bcell == 1, dv1[None], dv0[None]))
            carry = None
            for j in range(lo_n):
                e0 = ((dv0 >> j) & 1).astype(bool)
                e1 = ((dv1 >> j) & 1).astype(bool)
                dj = np.where(e0 & e1, ones, np.where(e1, bmask, np.where(e0, ~bmask & ones, np.uint64(0))))
                a = lo[j]
                if carry is None:
                    lo[j] = r.w(a ^ dj)
                    carry = a & dj
                    r.writes += float(np.mean(~(e0 & e1)))
                elif j < lo_n - 1:
                    lo[j] = r.w(a ^ dj ^ carry)
                    carry = r.w((a & dj) | (a & carry) | (dj & carry))
                else:
                    lo[j] = r.w(a ^ dj ^ carry)
            for j in range(lo_n):
                terms[(j + rho) & 31].append(lo[j])
            k = (top + rho) & 31
            const[k]["beta"] = const[k].get("beta", 0) ^ 1
            for j in range(hi_n):
                k = (lo_n + j + rho) & 31
                const[k][j] = const[k].get(j, 0) ^ 1
                if rotate:
                    x[k] = r.w(x[k] ^ (~lo[top] & rch[j]))
                    hiterm[k] += 1
        per_step = (r.writes - steps_start) / m
        fold = 0
        if not rotate:
            ptop = np.bitwise_xor.reduce(np.stack(terms[top]), axis=0)
            e = ptop ^ (ones if m & 1 else np.uint64(0))
            if m & 1:
                r.writes += xor_fold_writes(m) + 1 - xor_fold_writes(m + 1)
            for j in range(hi_n):
                k = lo_n + j
                x[k] = r.w((e & rch[j]) ^ (hi[j] if const[k].pop(j, 0) else np.uint64(0)))
                hiterm[k] = -1
        for k in range(32):
            cs = [beta if v == "beta" else hi[v] for v, odd in const[k].items() if odd]
            for reg in terms[k] + cs:
                x[k] = x[k] ^ reg
            n = len(terms[k]) + len(cs)
            fold += 0 if n == 0 else math.ceil(n / 2) if hiterm[k] else xor_fold_writes(n)
        r.writes += fold
        y = np.stack(transpose32(r, x))
        per_block = r.writes - start - per_step * m
        for c in range(32):
            for p in range(2):
                out[blk, 2 * c + p] = y[c, p * cols:(p + 1) * cols].astype(np.int64)
    per_word = r.writes / (steps // m) / (32 * m)
    uni = uniform_per_step(lo_n)
    honest = xorrot_honest_per_word(m) if rotate else xor_honest_writes(m) / m
    return out, {"lo_bits": lo_n, "per_step": per_step, "per_block": per_block, "per_word": per_word,
                 "uniform_per_word_t8192": uni / 32 / 32, "uniform_per_word_t512": uni / 8 / 32,
                 "honest_per_word": honest, "ratio": (per_word + uni / 32 / 32) / honest}


def bitsliced_adaptive(d: dict, m: int, rotate: bool = True) -> tuple[np.ndarray, dict]:
    """bitsliced_adversary with each column's low part as narrow as its blocks allow (the coins are uniform across the
    warp, so it knows each block's reach before the block): the columns run at their own width and the writes are
    their mean."""
    lb = np.array([lo_bits_for(v) for v in block_reach(d, m)])
    out = np.zeros((d["delta"].shape[0] // m, *d["w0"].shape), dtype=np.int64)
    per_word = uniform = 0.0
    widths = {}
    for width in np.unique(lb):
        sel = np.flatnonzero(lb == width)
        sub = {k: (v[..., sel] if k in ("s", "g", "w0", "coin", "delta", "words") else v) for k, v in d.items()}
        o, info = bitsliced_adversary(sub, m, int(width), rotate)
        out[..., sel] = o
        share = len(sel) / len(lb)
        per_word += info["per_word"] * share
        uniform += info["uniform_per_word_t8192"] * share
        widths[int(width)] = len(sel)
    honest = xorrot_honest_per_word(m) if rotate else xor_honest_writes(m) / m
    return out, {"widths": widths, "per_word": per_word, "uniform_per_word_t8192": uniform, "honest_per_word": honest,
                 "ratio": (per_word + uniform) / honest}


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


def part_r(checks: Checks, rng: np.random.Generator, x: dict) -> dict:
    out: dict = {"xorrot": {}, "xor": {}}
    for rotate, name in ((True, "xorrot"), (False, "xor")):
        for m in (8, 11, 16, 24, 31):
            d = m0_draw(rng, rows=64, cols=16, steps=4 * m)
            rho = np.array([rot_amount(t) if rotate else 0 for t in range(m)], dtype=np.int64)[None, :, None, None]
            w = d["words"].reshape(4, m, 64, 16)
            honest = np.bitwise_xor.reduce(np.where(rho == 0, w, ((w << rho) | (w >> ((32 - rho) & 31))) & M32), axis=1)
            fixed, fi = bitsliced_adversary(d, m, LO, rotate)
            adapt, ai = bitsliced_adaptive(d, m, rotate)
            ok = np.array_equal(fixed, honest) and np.array_equal(adapt, honest)
            out[name][m] = {"fixed": fi, "adaptive": ai, "exact": bool(ok)}
            checks.check(f"R: {name}-{m} at M0, the bit-sliced adversary matches honest's XOR bit for bit (4 blocks x "
                         f"64 rows x 16 columns), with a 21-bit low part and with each column's own", ok,
                         f"{ai['per_word'] + ai['uniform_per_word_t8192']:.3f} writes a word against honest's "
                         f"{ai['honest_per_word']:.3f} ({ai['ratio']:.3f}x; widths {ai['widths']}); at 21 bits "
                         f"{fi['ratio']:.3f}x")
    rot = {m: v["adaptive"]["ratio"] for m, v in out["xorrot"].items()}
    xr = {m: v["adaptive"]["ratio"] for m, v in out["xor"].items()}
    checks.check("R: rotation is free to a bit-sliced program: it comes within 15% of honest's writes on xorrot-m from "
                 "m = 16 and within 1% at m = 31, below its margin on xor-m at every m, so rotating weakens the premise",
                 rot[16] < 1.15 and rot[31] < 1.01 and all(rot[m] < xr[m] for m in rot),
                 ", ".join(f"m {m}: xorrot {rot[m]:.3f}x, xor {xr[m]:.3f}x" for m in rot))
    best = {m: min(xr[m], x[m]["ratio"]) if m in x else xr[m] for m in xr}
    checks.check("R: on xor-m it beats the packed adversary at m = 16 and still pays more than honest at every m <= 31, "
                 "so XorWrites_m stands against both",
                 xr[16] < x[16]["ratio"] and all(v > 1 for v in best.values()),
                 ", ".join(f"m {m}: bit-sliced {xr[m]:.3f}x" + (f", packed {x[m]['ratio']:.3f}x" if m in x else "")
                           for m in xr))
    return out


LAYOUTS = ((16, 1), (8, 2))
FORMING, FORMING_PREDS, FORMING_ROWS = 36, 4, 256


def packed_segment_writes(m: int, steps: int, priced: bool, later: bool, rows: int, pairs: int) -> dict:
    """Writes a word of packed_segment_adversary, by part, a thread holding `pairs` pairs (three cells of a row, lanes
    at 0, L and 2L of 64 bits) in each of `rows` rows of one parity: a pair's two running sums, its fold and a cell's
    one or two constants are 9 to 12 registers, so rows * pairs = 16 fits the register file.

    A pair a step: the 64-bit add (two IADD3s and their carry predicate) of the variant its row's b predicate picks; the
    row's b predicate (one a row and step) and the b register (one a thread and step); the two variants' four loads,
    shared by the thread's rows; and their forming once a (step, three columns, parity) for a parity's 256 rows (T 512):
    three loads (a column's four coin bytes), twelve byte extractions, a column's two variants (IMAD, IADD3, IADD3) and
    two packings (six and two carry predicates each). A pair a block: the fold of its two registers by 3-input LOP3s,
    the block's first sum written into the fold's register. A cell a block: its lane out of the pair (lane 0 none, the
    others a shift or a funnel) and the LOP3 merging the constant its lane's top bit picks (a predicate). A cell a
    segment: W0 (the first step's IMMA), W0 + 2^(L-2), beta 2^(L-1), v, H' 2^L (an IADD3 and a mask), H' 2^L - 2^L,
    the constants (one at even m, two at odd), its share of building the pair (three a pair) and, priced, beta's
    predicate; a later segment takes its W back out of the pair (a lane and an IADD3) instead of an IMMA."""
    pred = 1 if priced else 0
    step = (2 + pred) + pred / pairs + 1 / (rows * pairs) + 4 / rows + (FORMING + pred * FORMING_PREDS) / FORMING_ROWS
    fold = 2 * xor_fold_writes(m)
    end = sum(1 + lane + pred for lane in (0, 1, 1)) / 3
    setup = 9 + (m & 1) + pred + (1 if later else 0)
    blocks = steps // m
    parts = {"add": step / 3, "fold": fold * blocks / 3 / steps, "end": end * blocks / steps, "setup": setup / steps}
    return parts | {"per_word": sum(parts.values()), "rows": rows, "pairs": pairs}


def best_layout(m: int, steps: int, priced: bool, later: bool) -> dict:
    return min((packed_segment_writes(m, steps, priced, later, r, p) for r, p in LAYOUTS), key=lambda v: v["per_word"])


def packed_segment_adversary(d: dict, m: int) -> tuple[np.ndarray, dict]:
    """A packed M0 adversary for xor-m, centred once a segment (the draw's steps) instead of each block: three cells of
    a row a 64-bit pair, in lanes of L bits, L the narrowest that holds the segment's reach of its three columns (the
    coins are known before the segment). A cell's state is W = H' 2^L + v - beta 2^(L-1), with v in the middle half of
    [0, 2^L) at the segment's start, so no lane wraps in the segment: the pair adds the row's packed increment (the
    variant for its b bit, by predicate) each step and XORs into its fold; a block's XOR is the lane's L bits with bit
    L-1 flipped where beta at odd m, and high bits from beta, m's parity and the lane's top bit."""
    rows, cols = d["w0"].shape
    steps = d["delta"].shape[0]
    assert cols % 3 == 0 and steps % m == 0
    reach = block_reach(d, steps)
    out = np.zeros((steps // m, rows, cols), dtype=np.int64)
    widths = []
    for g in range(cols // 3):
        cs = range(3 * g, 3 * g + 3)
        width = lo_bits_for(max(reach[c] for c in cs))
        assert 3 * width <= 64
        widths.append(width)
        top = 1 << (width - 1)
        mask = (1 << width) - 1
        for i in range(rows):
            w0 = [int(d["w0"][i, c]) for c in cs]
            beta = [int(((w >> (width - 1)) & 1) == ((w >> (width - 2)) & 1)) for w in w0]
            v = [(w & mask) ^ (b * top) for w, b in zip(w0, beta)]
            hs = [(w + b * top) & M32 & ~mask for w, b in zip(w0, beta)]
            assert all((h + x - b * top) & M32 == w for h, x, b, w in zip(hs, v, beta, w0))
            q = sum(x << (width * j) for j, x in enumerate(v))
            for blk in range(steps // m):
                y = 0
                for t in range(blk * m, (blk + 1) * m):
                    q += sum(int(d["delta"][t, i, c]) << (width * j) for j, c in enumerate(cs))
                    assert 0 <= q < 1 << 64
                    y ^= q
                for j, c in enumerate(cs):
                    f = (y >> (width * j)) & mask
                    p = f >> (width - 1)
                    if not beta[j]:
                        hi = hs[j] if m & 1 else 0
                    else:
                        f ^= top if m & 1 else 0
                        hi = (hs[j] if p else 0) ^ (((hs[j] - (1 << width)) & M32) if (m ^ p) & 1 else 0)
                    out[blk, i, c] = f ^ hi
    honest = xor_honest_writes(m) / m
    model = {name: best_layout(m, steps, priced, False) for name, priced in
             (("w1_today", False), ("predicates_priced", True))}
    return out, {"widths": sorted(set(widths)), "honest_per_word": honest,
                 **{k: v | {"ratio": v["per_word"] / honest} for k, v in model.items()},
                 "later_segment": {k: best_layout(m, steps, k == "predicates_priced", True)["per_word"] / honest
                                   for k in model}}


def part_y(checks: Checks, rng: np.random.Generator, r: dict) -> dict:
    out = {}
    for m in (8, 11, 16):
        steps = m * (64 // m)
        d = m0_draw(rng, rows=64, cols=24, steps=steps)
        honest = np.bitwise_xor.reduce(d["words"].reshape(steps // m, m, 64, 24), axis=1)
        adv, info = packed_segment_adversary(d, m)
        ok = np.array_equal(adv, honest)
        out[m] = info | {"exact": bool(ok), "steps": steps}
        checks.check(f"Y: xor-{m} at M0, the packed adversary centred once a {steps}-step segment matches honest's XOR "
                     f"bit for bit ({steps // m} blocks x 64 rows x 24 columns)", ok,
                     f"lanes of {info['widths']} bits; {info['w1_today']['per_word']:.3f} writes a word with "
                     f"predicates free ({info['w1_today']['ratio']:.3f}x honest's {info['honest_per_word']:.3f}), "
                     f"{info['predicates_priced']['per_word']:.3f} priced ({info['predicates_priced']['ratio']:.3f}x)")
    checks.check("Y: with predicate writes free (today's W1) it pays at least 5% less than honest at m = 16 in every "
                 "segment, so XorWrites_16 is false unless predicate writes are priced; at m = 11 it ties honest within "
                 "2%, and at m = 8 it pays no less",
                 out[16]["w1_today"]["ratio"] < 0.95 and out[16]["later_segment"]["w1_today"] < 0.95
                 and abs(out[11]["w1_today"]["ratio"] - 1) < 0.02 and abs(out[11]["later_segment"]["w1_today"] - 1) < 0.02
                 and out[8]["w1_today"]["ratio"] > 1 - 1e-9,
                 ", ".join(f"m {m}: {v['w1_today']['ratio']:.4f}x (a later segment {v['later_segment']['w1_today']:.4f}x)"
                           for m, v in out.items()))
    best = {m: min(out[m]["predicates_priced"]["ratio"], r["xor"][m]["adaptive"]["ratio"]) for m in out}
    checks.check("Y: with predicate writes priced as register writes, it and the bit-sliced adversary both pay more "
                 "than honest at m = 8, 11 and 16, so XorWrites_m stands there",
                 all(v > 1 for v in best.values()),
                 ", ".join(f"m {m}: packed {out[m]['predicates_priced']['ratio']:.3f}x, bit-sliced "
                           f"{r['xor'][m]['adaptive']['ratio']:.3f}x" for m in out))
    return out


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
    r = part_r(checks, rng, x)
    y = part_y(checks, rng, r)
    if "--json" in args:
        Path(args[args.index("--json") + 1]).write_text(json.dumps(
            {"floor": f, "candidates": p, "xor": {str(k): v for k, v in x.items()},
             "trunc": {str(k): v for k, v in t.items()},
             "bitsliced": {n: {str(k): v for k, v in g.items()} for n, g in r.items()},
             "packed_segment": {str(k): v for k, v in y.items()}},
            indent=1, sort_keys=True, default=float) + "\n")
    print("\ncandidate      bits  h@t8192  cut   T1 DW-only  T1 premise  T2 t512 s/a     T2 r512 s/a     T2 t8192 s/a")
    for n, v in p.items():
        q = v["points"]
        print(f"{n:<13} {v['bits']:5.2f} {q['t8192']['h']:8.2f} {q['t8192']['cut']:5.1f}x {q['t8192']['t1_dw']:10.3%} "
              f"{q['t8192']['t1_premise']:10.3%}  " + "  ".join(
                  f"{q[pt]['t2']['sync']:6.2f}/{q[pt]['t2']['async']:6.2f}" for pt in POINTS))
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
