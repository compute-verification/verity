"""ShaLB's numbers: SHA-512's registers written a compression, the bounds below it, and what each leaves of T1: stdlib.

    python3 benchmarks/pouw/shalb/shalb.py [--counts COUNTS_JSON] [--minimal MINIMAL_JSON] [--json OUT]

- **The structural count.** One compression in the minimal kernel's dataflow (sha512_min.cuh), each 64-bit word tagged
  variable or compile-time constant and constants propagated: Σ/σ 8 registers (6 SHF, 2 LOP3), Ch and Maj 2 (a LOP3 a
  half), an add of m terms 2·⌈(m − 1)/2⌉ (IADD3 / IADD3.X; constants fold into one immediate), each round's T1 either
  materialized or folded into e′ and a′, whichever writes fewer, and e₈₀ and a₈₀ folded into the feed-forward. COUNTS_JSON
  (shalb_sass.sh's counts.json) gives the compiled counts beside it.
- **hm96's leaf** (verity/ml/commitments/hm96/PROTOCOL.md), compression by compression and form by form.
- **The bound ladder**: what each rank's premise proves about any program's registers written a compression, and the
  check its fan-in rungs read: every output register of the compression depends on every input register (a witness
  pair each, from a stdlib SHA-512 held to hashlib).
- **The bitsliced count**, the alternative representation every per-function argument has to cover.
- **T1**: the hashing W1 a useful MAC at SSN's (`ssn_1l`, α 1.044, every k32) and NCP's volumes, the δ T1 allows, the
  term each rung of the ladder leaves, the frontier (bytes a MAC each rung allows), and the pooled figures for a week, a
  month and a year: one H200, the reference datacenter, the world.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROUNDS, SCHEDULE = 80, 64
W1_PER_REG = 32
BLOCK = 128
AND_PER_ADD64 = 63
AND_PER_MAC = 130
H200_MACS_PER_S = 9.9e14
DC_GPUS = 100_000
WINDOW_S = 600
WORLD_MACS_PER_YEAR = 4.68e30
PERIODS = {"week": 7 * 86400, "month": 30 * 86400, "year": 365 * 86400}
VOLUMES = {"SSN ssn_1l (α 1.044, every k32)": 1.044 * 4 / 32, "NCP k32": 0.375}


def add_cost(n_var: int, has_const: bool) -> int:
    """Registers an add of n_var variable 64-bit terms and (if has_const) one folded constant writes."""
    m = n_var + (1 if has_const and n_var else 0)
    return 0 if m <= 1 else 2 * math.ceil((m - 1) / 2)


class Count:
    def __init__(self):
        self.shf = self.lop3 = self.add = 0

    def sigma(self, var: bool):
        if var:
            self.shf += 6
            self.lop3 += 2
        return var

    def bitwise(self, *vs: bool):
        if any(vs):
            self.lop3 += 2
        return any(vs)

    @property
    def total(self) -> int:
        return self.shf + self.lop3 + self.add


def compression(cv_var: bool, words_var: list[bool]) -> dict:
    """Registers written by one compression whose chaining value is variable (or constant) and whose 16 message words
    are variable or constant as `words_var` says."""
    c = Count()
    w = list(words_var)
    for t in range(16, ROUNDS):
        s1, s0 = c.sigma(w[t - 2]), c.sigma(w[t - 15])
        terms = [s1, w[t - 7], s0, w[t - 16]]
        c.add += add_cost(sum(terms), not all(terms))
        w.append(any(terms))
    st = [cv_var] * 8
    for t in range(ROUNDS):
        a, b, cc, d, e, f, g, h = st
        s1 = c.sigma(e)
        ch = c.bitwise(e, f, g)
        s0 = c.sigma(a)
        mj = c.bitwise(a, b, cc)
        n1 = sum([h, w[t], s1, ch])
        last = t == ROUNDS - 1
        e_extra = [d] + ([cv_var] if last else [])
        a_extra = [s0, mj] + ([cv_var] if last else [])
        if n1:
            materialized = (add_cost(n1, True) + add_cost(1 + sum(e_extra), not all(e_extra))
                            + add_cost(1 + sum(a_extra), not all(a_extra)))
            folded = add_cost(n1 + sum(e_extra), True) + add_cost(n1 + sum(a_extra), True)
            c.add += min(materialized, folded)
        else:
            c.add += add_cost(sum(e_extra), True) + add_cost(sum(a_extra), True)
        t1 = n1 > 0
        st = [t1 or any(a_extra[:2]), a, b, cc, t1 or d, e, f, g]
    for i in (1, 2, 3, 5, 6, 7):
        c.add += add_cost(st[i] + cv_var, True)
    return {"SHF": c.shf, "LOP3": c.lop3, "adds": c.add, "registers": c.total}


def and_gates() -> int:
    """AND gates of one compression as a Boolean circuit (Ch and Maj one AND a bit, an n-bit add n − 1)."""
    return ROUNDS * (7 * AND_PER_ADD64 + 2 * 64) + SCHEDULE * 3 * AND_PER_ADD64 + 8 * AND_PER_ADD64


def bitsliced() -> dict:
    """LOP3s a compression, bitsliced 32 instances to a register: rotations free, a three-way XOR or a Ch/Maj one LOP3 a
    bit, a two-term add two a bit but its top bit's carry, transposes left out."""
    add = 2 * 64 - 1
    per_round = 4 * 64 + 7 * add
    per_word = 2 * 64 + 3 * add
    total = ROUNDS * per_round + SCHEDULE * per_word + 8 * add
    return {"LOP3 per 32 instances": total, "per instance": total / 32}


def hm96_leaf(row_bytes: int) -> list[dict]:
    """hm96-sha512's compressions for one leaf of `row_bytes` value bytes behind a block-aligned prefix, by form."""
    var16 = [True] * 16
    blocks = math.ceil((row_bytes + 17) / BLOCK)
    tail_words = (row_bytes % BLOCK) // 8
    last = [i < tail_words for i in range(16)] if row_bytes % BLOCK <= BLOCK - 17 else var16
    rows = [
        {"what": "x: first value block (constant midstate of the prefix)", "n": 1, **compression(False, var16)},
        {"what": "x: middle value blocks", "n": max(blocks - 2, 0), **compression(True, var16)},
        {"what": "x: last block (value tail and padding)", "n": 1 if blocks > 1 else 0,
         **compression(True, last)},
        {"what": "c: salt bytes 0–127 (constant midstate of the salt prefix)", "n": 1, **compression(False, var16)},
        {"what": "c: salt bytes 128–191 and padding", "n": 1,
         **compression(True, [i < 8 for i in range(16)])},
        {"what": "leaf: b ‖ c (constant midstate of leaf_prefix(key))", "n": 1, **compression(False, var16)},
        {"what": "leaf: padding block", "n": 1, **compression(True, [False] * 16)},
    ]
    return rows


M64 = (1 << 64) - 1


def _primes(n: int) -> list[int]:
    ps: list[int] = []
    c = 2
    while len(ps) < n:
        if all(c % p for p in ps):
            ps.append(c)
        c += 1
    return ps


def _icbrt(n: int) -> int:
    x = 1 << -(-n.bit_length() // 3)
    while (y := (2 * x + n // (x * x)) // 3) < x:
        x = y
    while x ** 3 > n:
        x -= 1
    return x


SHA_K = [_icbrt(p << 192) & M64 for p in _primes(80)]
SHA_IV = [math.isqrt(p << 128) & M64 for p in _primes(8)]


def _rotr(x: int, n: int) -> int:
    return (x >> n | x << (64 - n)) & M64


def sha512_compress(cv: list[int], block: list[int]) -> list[int]:
    """SHA-512's compression (FIPS 180-4), its constants computed from the primes."""
    w = list(block)
    for t in range(16, ROUNDS):
        s0 = _rotr(w[t - 15], 1) ^ _rotr(w[t - 15], 8) ^ w[t - 15] >> 7
        s1 = _rotr(w[t - 2], 19) ^ _rotr(w[t - 2], 61) ^ w[t - 2] >> 6
        w.append((s1 + w[t - 7] + s0 + w[t - 16]) & M64)
    a, b, c, d, e, f, g, h = cv
    for t in range(ROUNDS):
        t1 = (h + (_rotr(e, 14) ^ _rotr(e, 18) ^ _rotr(e, 41)) + (e & f ^ ~e & g) + SHA_K[t] + w[t]) & M64
        t2 = ((_rotr(a, 28) ^ _rotr(a, 34) ^ _rotr(a, 39)) + (a & b ^ a & c ^ b & c)) & M64
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & M64, c, b, a, (t1 + t2) & M64
    return [(x + y) & M64 for x, y in zip(cv, (a, b, c, d, e, f, g, h))]


def sha512(msg: bytes) -> bytes:
    data = msg + b"\x80" + bytes(-(len(msg) + 17) % BLOCK) + (8 * len(msg)).to_bytes(16, "big")
    cv = list(SHA_IV)
    for i in range(0, len(data), BLOCK):
        cv = sha512_compress(cv, [int.from_bytes(data[j:j + 8], "big") for j in range(i, i + BLOCK, 8)])
    return b"".join(x.to_bytes(8, "big") for x in cv)


def dependence(seed: int = 1, max_trials: int = 8) -> dict:
    """Checks that each of the compression's 16 output registers depends on each of its 48 input registers (8 chaining
    words and 16 block words, lo and hi): for every pair, an input differing in that register alone whose output
    differs in that output register. A witness proves the dependence; `sha512` is first held to hashlib."""
    import hashlib
    import random
    for n in (0, 3, 111, 112, 127, 128, 200, 1000):
        msg = bytes((i * 151 + 7) % 256 for i in range(n))
        assert sha512(msg) == hashlib.sha512(msg).digest(), n
    rng = random.Random(seed)
    halves = lambda ws: [x >> 32 * s & 0xFFFFFFFF for x in ws for s in (0, 1)]
    need = {(o, i) for o in range(16) for i in range(48)}
    trials = 0
    while need and trials < max_trials:
        trials += 1
        words = [rng.getrandbits(64) for _ in range(24)]
        base = halves(sha512_compress(words[:8], words[8:]))
        for i in range(48):
            flipped = list(words)
            flipped[i // 2] ^= 1 << (32 * (i % 2) + rng.randrange(32))
            out = halves(sha512_compress(flipped[:8], flipped[8:]))
            need -= {(o, i) for o in range(16) if out[o] != base[o]}
    return {"pairs": 16 * 48, "witnessed": 16 * 48 - len(need), "trials": trials, "missing": sorted(need)}


def ladder(honest: int) -> list[dict]:
    """Each rung: the premise, its rank, and the registers any program writes a compression under it."""
    inputs, outputs = 2 * (16 + 8), 16
    fan_in = lambda d: math.ceil((inputs - 1) / (d - 1)) + outputs - 1
    materialized = 2 * (2 * ROUNDS + SCHEDULE) + outputs - 4
    dataflow = ("a dataflow refinement of MH100w (rank 3): each value written is a function of the values its "
                "instruction reads, and a predicate write is priced as a register write (under today's W1 it is free, "
                "and free predicate chains fold any number of registers into one value read, so every L1 falls to L0). "
                "Every output register depends on all 48 input registers (checked, rank 2), so the first output "
                "written has a cone of ⌈47/(d − 1)⌉ writes and the other 15 outputs lie outside it")
    rungs = [
        ("L0", "1",
         "H32 and DistinctWrites (MH100w): the 16 output registers are distinct, priced writes. MH100w attains it: "
         "an int instruction writes any function, so 16 of them write the digest, and nothing stated in MH100w gives "
         "more", outputs),
        ("L1-isa", "3", f"{dataflow}; sm_90's whole ISA, d = 32 (REDUX, VOTE and MATCH read a value from every lane)",
         fan_in(32)),
        ("L1-5", "3", f"{dataflow}; instructions reading at most 3 registers and 2 predicates, d = 5 (the integer "
         "lane without warp-wide instructions or tensor cores)", fan_in(5)),
        ("L1-3", "3", f"{dataflow}; at most 3 values read, d = 3", fan_in(3)),
        ("L2", "4a",
         "materialization: every program holds each a_t, e_t (t = 1..80), each W_t (t = 16..79) and each output word "
         "in registers (a_80 and e_80 folded into H0 and H4). A claim about every algorithm, a weak form of L3",
         materialized),
        ("L3", "4a (per function checked)",
         "every Σ/σ half at its minimal 4, every Ch/Maj half 1, every add at its fan-in minimum, and no two of them "
         "share work: the honest count", honest),
    ]
    return [{"rung": r, "rank": k, "premise": p, "registers": n, "share_of_honest": n / honest,
             "residual": 1 - n / honest} for r, k, p, n in rungs]


BEST_RANK3 = "L0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="")
    ap.add_argument("--minimal", default="")
    ap.add_argument("--row-kib", type=int, default=256)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    out: dict = {}
    forms = {"variable chaining value and block": compression(True, [True] * 16),
             "constant chaining value, variable block": compression(False, [True] * 16),
             "variable chaining value, constant block (padding)": compression(True, [False] * 16)}
    out["structural"] = forms
    honest = forms["variable chaining value and block"]["registers"]
    print("## Structural count (derived)")
    for k, v in forms.items():
        print(f"- {k}: {v['registers']} registers ({v['SHF']} SHF, {v['LOP3']} LOP3, {v['adds']} adds)")
    if args.counts:
        counts = json.loads(Path(args.counts).read_text())
        out["compiled"] = {k: v["compression"]["registers_written"] for k, v in counts.items()}
        print("## Compiled (SASS, sm_90a)")
        for k, v in out["compiled"].items():
            print(f"- {k}: {v:.0f} registers written a compression")
    ands = and_gates()
    bs = bitsliced()
    out["and_gates"], out["bitsliced"] = ands, bs
    print(f"## AND gates a compression: {ands}; bitsliced: {bs['per instance']:.0f} LOP3 an instance "
          f"({bs['per instance'] / honest - 1:+.1%} against {honest})")

    leaf = hm96_leaf(args.row_kib * 1024)
    out["hm96_leaf"] = leaf
    tot = sum(r["n"] * r["registers"] for r in leaf)
    bulk = args.row_kib * 1024 / BLOCK * honest
    print(f"## hm96-sha512 leaf, {args.row_kib} KiB rows: {tot:,} registers ({tot / bulk - 1:+.3%} against "
          f"{args.row_kib} KiB at {honest} a block); the salt PRF (3 compressions) and M·y are N-HIDE's")
    for r in leaf:
        print(f"- {r['what']}: {r['n']} × {r['registers']}")

    dep = dependence()
    out["dependence"] = dep
    print(f"## Dependence (checked): {dep['witnessed']} of {dep['pairs']} (output, input) register pairs witnessed in "
          f"{dep['trials']} trial(s); sha512 matches hashlib")
    lad = ladder(honest)
    out["ladder"] = lad
    rung = {r["rung"]: r for r in lad}
    print("## The ladder: registers any program writes a compression")
    for r in lad:
        print(f"- {r['rung']} (rank {r['rank']}): {r['registers']} ({r['share_of_honest']:.2%} of {honest}; "
              f"residual {r['residual']:.4f}): {r['premise']}")

    w1_byte = honest * W1_PER_REG / BLOCK
    out["w1_per_byte"] = w1_byte
    print(f"## T1 (W1 floor {W1_PER_REG} a register; {w1_byte:.1f} W1 a hashed byte)")
    t1 = []
    for vol, bpm in VOLUMES.items():
        hw = bpm * w1_byte
        delta = 0.01 / hw
        row = {"volume": vol, "bytes_per_mac": bpm, "hash_w1_per_mac": hw, "delta": delta,
               "delta_registers": delta * honest, "rungs": {}}
        for r in lad:
            term = r["residual"] * hw
            row["rungs"][r["rung"]] = {"t1_term_share_of_stock": term,
                                       "bytes_per_mac_allowed": 0.01 / (r["residual"] * w1_byte) if r["residual"] else
                                       math.inf}
        t1.append(row)
        print(f"- {vol}: {bpm:.4f} B/MAC, hashing {hw:.1f} W1 a MAC ({hw:.0f}× stock); δ = {delta:.2e} "
              f"({delta * honest:.2f} registers a compression)")
        for k, v in row["rungs"].items():
            print(f"  - {k}: T1 term {v['t1_term_share_of_stock']:.3g}× stock; allows "
                  f"{v['bytes_per_mac_allowed']:.3g} B/MAC")
    out["t1"] = t1

    print("## Frontier: residual reached → bytes a MAC T1 allows")
    frontier = []
    for r in [rung[n]["residual"] for n in ("L0", "L1-isa", "L1-5", "L2")] + [0.1, 0.01, 1e-3, 1e-4,
                                                                             t1[0]["delta"], t1[1]["delta"]]:
        bpm = 0.01 / (r * w1_byte)
        frontier.append({"residual": r, "bytes_per_mac": bpm})
        print(f"- residual {r:.4g}: {bpm:.3g} B/MAC")
    out["frontier"] = frontier

    print(f"## Per monitoring period: the unproved hashing (rung {BEST_RANK3}'s residual, the best at rank ≤ 3 under "
          "today's W1) against T1's 1% of stock")
    pooled = []
    for vol_row in t1:
        hw, res = vol_row["hash_w1_per_mac"], rung[BEST_RANK3]["residual"]
        for scope, rate in (("one H200", H200_MACS_PER_S), ("datacenter (100,000 H200)", H200_MACS_PER_S * DC_GPUS)):
            for p, s in {"window (600 s)": WINDOW_S, **PERIODS}.items():
                macs = rate * s
                comp = macs * vol_row["bytes_per_mac"] / BLOCK
                pooled.append({"volume": vol_row["volume"], "scope": scope, "period": p, "stock_macs": macs,
                               "compressions": comp, "unproved_w1": res * hw * macs, "t1_budget_w1": 0.01 * macs,
                               "unproved_and": res * comp * ands, "stock_and": macs * AND_PER_MAC})
        macs = WORLD_MACS_PER_YEAR
        comp = macs * vol_row["bytes_per_mac"] / BLOCK
        pooled.append({"volume": vol_row["volume"], "scope": "world", "period": "year", "stock_macs": macs,
                       "compressions": comp, "unproved_w1": res * hw * macs, "t1_budget_w1": 0.01 * macs,
                       "unproved_and": res * comp * ands, "stock_and": macs * AND_PER_MAC})
    for r in pooled:
        print(f"- {r['volume']}, {r['scope']}, {r['period']}: {r['compressions']:.3g} compressions; unproved "
              f"{r['unproved_w1']:.3g} W1 against T1's {r['t1_budget_w1']:.3g} ({r['unproved_and']:.3g} AND gates "
              f"against stock's {r['stock_and']:.3g})")
    out["pooled"] = pooled

    if args.minimal:
        out["minimal"] = json.loads(Path(args.minimal).read_text())["summary"]
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
