"""hm96's hiding work and the hashing of served sparse noise (SSN) in Llama-3.2-1B's served step on the H200, priced: what the
honest prover alone does to hide its committed values, as a share of stock at T 8,192 and T 512 (64 and 512 rows), under each
commitment layout and each way of forcing it, the proof's cost to open what it draws, and SSN's bytes, t_H/t_S and ShaSpeed's
precision. Stdlib only, no GPU.

    python3 benchmarks/served_commit/np_hide_price.py SASS_JSON BEST_SASS_JSON ROWS_JSON OUT_JSON [LATENCY_JSON [MASK_SASS_JSON]]

SASS_JSON and ROWS_JSON are sw_hash_price.py's inputs: a SHA-512 compression's W1 work in the committer's build, and each row
size's leaf kernels' GPU time (the salt kernel's derived and given salts, a compression's time). BEST_SASS_JSON is the build
with the fewest registers written a compression: ShaSpeed can credit no more than it, so the committer's excess over it is
honest-only work, and each register a compression is priced as a share of stock at SSN's volume. LATENCY_JSON, if given, is
sha512_latency.cu's output: a call's seed path (A's hash, the seed and its first noise block, which the call's GEMM waits on)
is priced at a compression's latency on one thread and at the build's issue floor (an estimate), for each of CHAINS' layouts
of A's leaves and subtree. GEMM j can't start before seed_j and A_j is call j−1's output, so the chain bounds T2 from below at
1 + its time over the stock step, and it enters every T2 in place of the coins' floor (a launch and a noise block), which
the two sheet-clock readings keep. MASK_SASS_JSON, if given, is mask_sass.py's output: M_key·y's W1 a salt from its SASS, the
committer's rolled build as the price and the unrolled build beside it.

An hm96 instance is one salt: y = PRF_ks(pos) (three compressions from round 8), c = SHA-512(salt_prefix ‖ y) (two) and
b = x ⊕ M_key·y. A prover that reuses one salt skips all three, and nothing the gadget checks changes, so they are honest-only
work, priced a salt:
  unforced: the PRF, c and M_key·y; at the floor and alone each compression at its SASS W1, M_key·y at its registers written
    (the H200 byte-table form: from MASK_SASS_JSON, else counted from the source at 32 (floor) or one LOP3's measured price
    (alone)), and with the unrolled mask build when MASK_SASS_JSON is given; "time" is the salt
    kernel's measured time a salt less the frame leaf's first block, in MAC-times;
  forced (a): the drawn unit's proof checks y = PRF_ks(pos) against a committed ks and checks c, so ShaSpeed credits both
    and M_key·y alone remains (time: the salt kernel from given salts less c and the frame leaf's first block);
  SHA-512 mask (e2): M_key·y replaced by SHA-512(mask_prefix ‖ y), credited with the PRF and c under (a), so nothing remains;
  b1 mask (an estimate): M_key·y bit for bit on the tensor cores (mma m16n8k256 .b1 .and.popc, M's fragments resident in
    eight warps' registers), its registers written counted from the shape, not compiled.
The term is a salt's price times the salts a useful MAC, and the layout sets the salts:
  L0, today: one hm96 leaf per activation row of A per call (row-seg) and one per R-byte row of committed words;
  L1, a leaf over a subtree: one hm96 instance per call over A's chunks and one per forward over its committed words, each
    inner digest a frame-v3-sha512 tree, so A's public root stays a hiding root while openings cost a chunk and its path;
    no seed reads the words' root, so their hash stays off every call's seed path;
  L2, private roots: the call's roots enter the noise seed inside the drawn unit's proof only, so one instance hides a
    forward, or a window (WINDOW_STEP_S bounds a step's time from above, so a window holds at least 600 s over it forwards).
The proof opens an instance by hashing what it covers in-circuit: an L0 row whole, an L1 chunk and its path (two compressions a
level), plus c (and the PRF when forced). In AND gates (hm96's model) the hiding work is the PRF and c; M_key·y is XORs.

SSN (ssn_1l and ssn_a) commits every k32 accumulator word of its stacked product, S words a dot product, and A, whose root
seeds the call's noise. t_H/t_S, an estimate on measured parts, adds to the stock step the stacked width's extra GEMM work,
the hash at hm96 rows' measured rate alone (16 or 64 KiB rows, low and high reading; L1 at 16 KiB chunks scales 16 KiB's by
the leaf kernels' time without the salt side and the leaf hash), the words' move out of the accumulator (an in-register
fold, or HBM stores), and each call's coin expansion at its floor. Overlap is the larger of everything but the hash, and the
hash: hashing beside the step is cheaper only while the step runs. "At the bar" prices every compression at the bar's.
ShaSpeed's precision δ is the T1 budget over the hash's W1 a useful MAC. The hashing frontier is the bytes a useful MAC at
which the hash alone, at the bar, fits in T2 stock steps overlapped (where the serial part fits) or T2 − 1 serially, at
T2's bar (T2_BAR) and at 2 and 1.1. The V4-like layers (`v4`) get the hiding term under each layout and the seed path. Hashing before the noise is A's commitment: under L0 a
row-seg row with its hm96 leaf and tree share (its salt is in the hiding term); under L1 a row-seg or flat SHA-512 row and its
tree share, or segments as frame-v3 leaves of the call's own domain. A row digest is a function of the row alone, so a
repeated A_j or row reuses it whatever hides it; only the segment leaves read a per-call value (the domain id, rank and
position), so only their hashing is distinct in every call. Tree costs are frame-v3-sha512's (FRAME_LEAF_PREFIX, NODE),
whose node is binary.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncp_k32_price as N  # noqa: E402
import sw_hash_price as S  # noqa: E402

CLOCK = 1.98e9                                      # the hashing measurements' SM clock
UNITS_S = N.RULER * CLOCK                           # MAC-times a second at it
SHEET_MAC_S = 989.5e12                              # the H200's data-sheet dense int8 MACs a second
MFU_SERVE = 0.4                                     # stock work held fixed: benchmarks/rho/data/reference.toml fleet.world-h200
PRF_SHARE = S.ANDS_FROM_8 / S.ANDS_COMPRESSION      # a compression from round 8 over a whole one (W1 taken in the same ratio)
# M_key·y in the H200 split form (byte table in shared memory), registers written a salt: 16 passes of 12 salt bytes, each
# 5 LDS.128 (20 registers), 10 three-input XORs and 3 for the byte and its address; each pass 37 conditional word shifts and
# 16 funnel shifts with their 16 XORs
MASK_REGS = 16 * (12 * (20 + 10 + 3) + 37 + 32)
# the b1 form a salt: 24 mma (512 x 1536 bits over 16 x 256 tiles, eight salts a column set) writing 4 accumulator registers in
# each of 32 lanes, y's fragments (2 registers a k-step in each of eight warps), and two instructions an output bit for parity
MASK_REGS_B1 = 24 * 4 * 32 + 8 * 6 * 2 * 32 // 8 + 2 * 512
PRICE_REG = {"floor": 32.0, "alone": S.PRICE["alone"]}
# a frame-v3-sha512 leaf's message before its value: the frame string, the tag, the 64-byte domain id, rank and position
# (3 bytes each), a schema as long as hm96-sha512/row-seg/v1's, and the value's length, each part length-framed. Its first
# block holds the rank, so no midstate skips it
FRAME_LEAF_PREFIX = 33 + 4 + 4 + (8 + 64) + 2 * (8 + 3) + (8 + 22) + 8
HM96_LEAF = 2                                       # H(leaf_prefix ‖ b ‖ c): 128 bytes and the padding block
# a tree node's compressions past its (domain, level) midstate: the index, the children's digests and the padding. frame-v3's
# node is binary; a 4-ary node is a frame change
NODE = {2: 2, 4: 3}
TREE_LEAF = S.blocks(FRAME_LEAF_PREFIX + 64) + NODE[2]   # a row digest's frame leaf and its share of the binary nodes
# the seed path's layouts: A's leaves (row-seg rows, flat rows, or segments of so many bytes) and the subtree's arity
CHAINS = {"row-seg rows, binary (L1)": ("row-seg", 2), "flat rows, binary": ("flat", 2),
          "1 KiB segments as leaves, binary": (1024, 2), "256 B segments as leaves, binary": (256, 2),
          "row-seg rows, 4-ary (a frame change)": ("row-seg", 4), "1 KiB segments as leaves, 4-ary (a frame change)": (1024, 4),
          "256 B segments as leaves, 4-ary (a frame change)": (256, 4)}
WINDOW_STEP_S = 0.1
HASH_ALONE = {16384: (2043, 2159), 65536: (1928, 2010)}   # hm96 rows hashed alone, units a byte (low, high readings)
STEP_BESIDE = 981                                   # beside the served step's GEMM, units a byte, while the step runs
EXPORT_WORD = {"fold": 189, "stg": 848}             # extra units a word out of the accumulator at L 32: in-register fold, HBM
COIN_CALL_US = 9.3                                  # a call's coin expansion at its latency floor
CALLS = 4 * N.LAYERS
UNIT_OUT_BYTES = 262_144 // 8                       # a proof block's outputs at the largest public size class
BUDGET = 0.01
TARGET = 0.001
T2_BAR = 100
# V4-like shapes: a dense layer (K = n = 7,168) and a routed expert pair (gate-up n 4,096 K 7,168; down n 7,168 K 2,048),
# each expert reading TOP_K / EXPERTS of the rows; the H200's data-sheet HBM bandwidth bounds a call's weight read
V4_DENSE = {"dense": (7168, 7168)}
V4_EXPERT = {"gate-up": (4096, 7168), "down": (7168, 2048)}
EXPERTS, TOP_K = 256, 8
HBM_B_S = 4.8e12
SSN = {"ssn_1l": lambda k: 1 + math.ceil((k - 28) / 31), "ssn_a": lambda k: math.ceil(k / 28)}
SSN_T1 = {"ssn_1l": {"credited": 0.00049, "uncredited": 0.00255}, "ssn_a": {"credited": 0.00054, "uncredited": 0.00944}}
POINTS = {**N.POINTS, "decode, 16 rows": (16, 64), "decode, 1 row": (1, 64)}   # (real rows, the GEMM table's m)
MACS_TOKEN = N.LAYERS * sum(n * k for n, k in N.LINEARS.values())
HIDING_ANDS = 3 * S.ANDS_FROM_8 + 2 * S.ANDS_COMPRESSION


def segs(k: int) -> int:
    """A row-seg row of k bytes: its segments' compressions and the outer hash's."""
    s = -(-k // 1024)
    return 9 * s + (64 * s + 17 + 127) // 128


def instance_prices(sass: dict, rows: S.Rows, mask_sass: dict | None = None) -> dict:
    """W1 units a salt: the honest-only work, unforced and with the PRF and c forced."""
    w1c = {"floor": sass["w1_floor"], "alone": sass["w1_alone"]}
    r = rows.by[S.ROWS_16K]
    mt = N.RULER * r["mhz"] * 1e6 * 1e-9
    prf = {p: 3 * PRF_SHARE * w for p, w in w1c.items()}
    c = {p: 2 * w for p, w in w1c.items()}
    if mask_sass:
        mask = {p: mask_sass["rolled"][f"w1_{p}"] for p in w1c}
        best = {p: mask_sass["unrolled"][f"w1_{p}"] for p in w1c}
        regs = mask_sass["rolled"]["registers_written"]
    else:
        mask, best, regs = {p: MASK_REGS * PRICE_REG[p] for p in w1c}, None, MASK_REGS
    b1 = {p: MASK_REGS_B1 * PRICE_REG[p] for p in w1c}
    by_best = {"unforced, unrolled mask": {p: prf[p] + c[p] + best[p] for p in w1c},
               "forced (a), unrolled mask": dict(best)} if best else {}
    return {
        "parts": {"prf": prf, "c": c, "mask": mask, "mask_registers": regs, "b1_mask": b1, "b1_mask_registers": MASK_REGS_B1,
                  "mask_time": (r["salt_given_ns_leaf"] - (2 + 65 / 80) * r["c_ns"]) * mt,
                  "salt_kernel_ns": r["salt_ns_leaf"], "given_ns": r["salt_given_ns_leaf"], "compression_ns": r["c_ns"]},
        "unforced": {**{p: prf[p] + c[p] + mask[p] for p in w1c}, "time": (r["salt_ns_leaf"] - 65 / 80 * r["c_ns"]) * mt},
        "forced (a)": {**mask, "time": (r["salt_given_ns_leaf"] - (2 + 65 / 80) * r["c_ns"]) * mt},
        "unforced, b1 mask (estimate)": {p: prf[p] + c[p] + b1[p] for p in w1c},
        "forced (a), b1 mask (estimate)": dict(b1),
        "forced (a), SHA-512 mask (e2)": {"floor": 0.0, "alone": 0.0, "time": 0.0},
        **by_best,
        "compression": {**w1c, "time": r["c_ns"] * mt},
    }


def ssn_bytes(design: str) -> dict:
    """SSN a token: committed words' bytes, A's bytes, the linears' MACs and the stacked width α."""
    s = SSN[design]
    words = N.LAYERS * sum(n * 4 * s(k) for n, k in N.LINEARS.values())
    width = N.LAYERS * sum(n * 32 * s(k) for n, k in N.LINEARS.values())
    a = N.LAYERS * sum(k for _, k in N.LINEARS.values())
    return {"words": words, "a": a, "macs": MACS_TOKEN, "alpha": width / MACS_TOKEN,
            "words_bpm": words / MACS_TOKEN, "a_bpm": a / MACS_TOKEN, "bpm": (words + a) / MACS_TOKEN,
            "calls": {x: {"words_row": n * 4 * s(k), "a_row": k} for x, (n, k) in N.LINEARS.items()}}


def salts_per_mac(layout: str, rows: int, b: dict, R: int) -> dict:
    fwd = rows * MACS_TOKEN
    if layout == "L0":
        return {"A rows": CALLS * rows / fwd, "word rows": b["words_bpm"] / R}
    if layout == "L1":
        return {"calls": (CALLS + 1) / fwd}
    if layout == "L2, a window":
        return {"window": WINDOW_STEP_S / S.CENSUS["window s"] / fwd}
    return {"forward": 1 / fwd}


def opening_ands(layout: str, b: dict, rows: int, R: int, forced: bool) -> dict:
    """In-circuit ANDs to open what a drawn unit reads of one instance, beyond the bytes it reads itself."""
    # c, and when forced the PRF and ks's own opening (its inner digest and c; the verifier computes its leaf)
    extra = 2 * S.ANDS_COMPRESSION + (3 * S.ANDS_FROM_8 + 3 * S.ANDS_COMPRESSION if forced else 0)
    if layout == "L0":
        return {"words row, whole": S.blocks(R) * S.ANDS_COMPRESSION + extra,
                "words row over the unit's own outputs": max(0, S.blocks(R) - S.blocks(UNIT_OUT_BYTES)) * S.ANDS_COMPRESSION + extra}
    big = max(rows * c["words_row"] for c in b["calls"].values())
    depth = math.ceil(math.log2(max(2, big / UNIT_OUT_BYTES)))
    if layout.startswith("L2"):
        # up from the call's subtree to the instance: the forward's calls, and for a window its forwards
        span = CALLS + 1 if layout == "L2" else (CALLS + 1) * S.CENSUS["window s"] / WINDOW_STEP_S
        depth += math.ceil(math.log2(span))
    return {"path depth": depth, "path and gadget": 2 * depth * S.ANDS_COMPRESSION + extra}


def frame_leaf(n: int) -> int:
    """A frame-v3-sha512 leaf's compressions over an n-byte value."""
    return S.blocks(FRAME_LEAF_PREFIX + n)


def levels(leaves: int, arity: int) -> int:
    n, d = 1, 0
    while n < max(2, leaves):
        n, d = n * arity, d + 1
    return d


def seed_path(rows: int, k: int, row: str | int = "row-seg", arity: int = 2) -> int:
    """Compressions one after another from a call's A (rows of k bytes) to its first noise block under L1: A's leaf digests
    (a row-seg row: its segments side by side, then their outer hash, then the subtree's leaf; a flat SHA-512 row and its
    leaf; or, for an int `row`, segments of that many bytes as the subtree's leaves), the subtree (NODE[arity] a level),
    the instance's leaf (its salt, c and M_key·y precomputed), the seed past its fixed prefix, and one noise block."""
    if row == "row-seg":
        leaf, leaves = 9 + -(-(64 * -(-k // 1024) + 17) // 128) + frame_leaf(64), rows
    elif row == "flat":
        leaf, leaves = S.blocks(k) + frame_leaf(64), rows
    else:
        leaf, leaves = frame_leaf(row), rows * -(-k // row)
    return leaf + NODE[arity] * levels(leaves, arity) + HM96_LEAF + 2 + 1


def v4(price: dict, lat: dict | None, sizes: list[int]) -> dict:
    """ssn_1l's hiding work on the V4-like layers at each point's rows m: the dense layer; the experts with a seed group per
    expert call, or one per linear over the layer's experts (calls committed together share a slot of `pouw.roots`). A
    layer's L1 instances are its seed groups (the forward's one extra amortizes over its layers). With a latency, each
    linear's seed path in one stream over its GEMM alone (an estimate: the larger of its MACs at the sheet's rate and its
    weights at HBM_B_S; the layer's other operations would lower the ratio)."""
    s = SSN["ssn_1l"]
    out: dict = {}
    for pname, (m, _gm) in POINTS.items():
        m_e = max(1.0, m * TOP_K / EXPERTS)
        active = min(EXPERTS, m * TOP_K)
        cases = {"V4 dense": (V4_DENSE, m, 1, {"one per call": 1}),
                 "V4 experts": (V4_EXPERT, m * TOP_K, active, {"one per expert call": active * len(V4_EXPERT),
                                                               "one per linear over the experts": len(V4_EXPERT)})}
        for case, (lins, rows_l, n_exp, groups) in cases.items():
            macs = rows_l * sum(n * k for n, k in lins.values())
            words = rows_l * sum(n * 4 * s(k) for n, k in lins.values())
            sp: dict = {f"L0 {R // 1024} KiB rows": rows_l * len(lins) + words / R for R in sizes}
            sp.update({f"L1, {g}": c for g, c in groups.items()})
            e: dict = {"rows_a_linear": rows_l, "macs": macs, "bytes_a_mac": words / macs, "salts": sp, "hiding": {
                lay: {f: {p: n * v / macs for p, v in price[f].items() if p != "time"} for f in ("unforced", "forced (a)")}
                for lay, n in sp.items()}}
            if lat:
                e["seed_path"] = {}
                for g in groups:
                    span = rows_l if "over" in g or case == "V4 dense" else m_e
                    tot = gemm = 0.0
                    for n, k in lins.values():
                        chain = seed_path(math.ceil(span), k) * lat["tree_ns"] * 1e-9 + lat["launch_us"] * 1e-6
                        g_s = max(rows_l * n * k / SHEET_MAC_S, n_exp * n * k / HBM_B_S)
                        tot, gemm = tot + g_s + chain, gemm + g_s
                    e["seed_path"][g] = {"us": (tot - gemm) * 1e6, "gemm_us": gemm * 1e6, "t2_floor_over_gemm": tot / gemm}
            out[f"{case}, {pname}"] = e
    return out


def t2(design: str, b: dict, pname: str, rows: int, m: int, rates: dict, chain_ms: float | None = None) -> dict:
    """t_H/t_S for each engine and hash rate. What the GEMMs wait on serially is the calls' coin floor, or, given
    `chain_ms`, the seed path, which contains it."""
    g = N.GEMM[min(N.GEMM, key=lambda x: abs(x - m))]
    gemm_ms = (b["alpha"] - 1) * g["plain_us"] * 1e-3 * rows / m
    B = (b["words"] + b["a"]) * rows
    words = b["words"] / 4 * rows
    export = {e: words * u / UNITS_S * 1e3 for e, u in EXPORT_WORD.items()}
    coin_floor = CALLS * COIN_CALL_US * 1e-3
    coins = coin_floor if chain_ms is None else chain_ms
    out = {"GB_forward": B / 1e9, "gemm_extra_ms": gemm_ms, "export_ms": export, "coins_ms": coins,
           "coins_src": "coin floor a call" if chain_ms is None else "seed path, row-seg rows, binary (L1), measured", "hash_ms": {},
           "engines": {}}
    for rn, u in rates.items():
        out["hash_ms"][rn] = B * u / UNITS_S * 1e3
    engines = N.ENGINES.get(pname)
    if engines is None:
        return out
    fwd_macs = rows * MACS_TOKEN
    bar = rates["at the bar, L1 16 KiB chunks"]
    for eng, (stock, _gap, src) in engines.items():
        e = {"stock_ms": stock, "src": src, "stock_mac_times_a_useful_mac": stock * 1e-3 * UNITS_S / fwd_macs,
             "bytes a useful MAC for T2 at the bar, L1 16 KiB chunks": {
                 f"{sched} ≤ {x}": max(0.0, budget) * 1e-3 * UNITS_S / per_byte / fwd_macs for x in (T2_BAR, 2, 1.1)
                 for sched, budget, per_byte in (("overlap", x * stock if stock + gemm_ms + coins <= x * stock else 0.0, bar),
                                                 ("serial, HBM export", (x - 1) * stock - gemm_ms - coins,
                                                  bar + EXPORT_WORD["stg"] / 4))}}
        for rn, h in out["hash_ms"].items():
            e[rn] = {"serial": (stock + gemm_ms + h + export["stg"] + coins) / stock,
                     "fused": (stock + gemm_ms + h + export["fold"] + coins) / stock,
                     "overlap": max(stock + gemm_ms + export["fold"] + coins, h) / stock}
        e["reading: beside-the-step price on every byte, sheet clock"] = (
            stock + gemm_ms + B * (STEP_BESIDE + EXPORT_WORD["stg"] / 4) / SHEET_MAC_S * 1e3 + coin_floor) / stock
        e["reading: alone 64 KiB high, sheet clock, HBM export"] = (
            stock + gemm_ms + B * (HASH_ALONE[65536][1] + EXPORT_WORD["stg"] / 4) / SHEET_MAC_S * 1e3 + coin_floor) / stock
        out["engines"][eng] = e
    return out


def latency(path: Path) -> dict:
    """sha512_latency.cu's output: a compression's latency on one thread (a row's chain, mode 0; a tree path's, mode 1) from
    the SM clock's slope between 16 and 256 compressions at CLOCK, and a launch's floor (one compression's wall time less
    the compression)."""
    d = json.loads(path.read_text())
    one = {(r["mode"], r["n"]): r for r in d["rows"] if r["grid"] == 1 and r["block"] == 1}
    ns = {m: (one[(m, 256)]["cycles_max"] - one[(m, 16)]["cycles_max"]) / 240 / CLOCK * 1e9 for m in (0, 1)}
    return {"src": {"sha_roll": d["sha_roll"]}, "row_ns": ns[0], "tree_ns": ns[1],
            "launch_us": one[(0, 1)]["wall_us_best"] - ns[0] * 1e-3, "rows": d["rows"]}


def main() -> None:
    sass = json.loads(Path(sys.argv[1]).read_text())["compression"]
    best = json.loads(Path(sys.argv[2]).read_text())["compression"]
    rows = S.Rows(Path(sys.argv[3]))
    mask_sass = json.loads(Path(sys.argv[6]).read_text()) if len(sys.argv) > 6 else None
    price = instance_prices(sass, rows, mask_sass)
    excess = {"registers": sass["registers_written"] - best["registers_written"],
              "floor": sass["w1_floor"] - best["w1_floor"], "alone": sass["w1_alone"] - best["w1_alone"]}
    lat = latency(Path(sys.argv[5])) if len(sys.argv) > 5 else None
    if lat:
        lat["per_compression_ns"] = {"measured": lat["tree_ns"], "issue floor (estimate)": sass["instructions"] / CLOCK * 1e9}
    r16 = rows.by[S.ROWS_16K]
    l1_ratio = (r16["leaf_ns"] - r16["salt_ns_leaf"] + 65 / 80 * r16["c_ns"] + (3 - 2) * r16["c_ns"]) / (r16["leaf_ns"] + 3 * r16["c_ns"])
    rates = {"16 KiB low": HASH_ALONE[16384][0], "16 KiB high": HASH_ALONE[16384][1],
             "64 KiB low": HASH_ALONE[65536][0], "64 KiB high": HASH_ALONE[65536][1],
             "L1 16 KiB chunks low": HASH_ALONE[16384][0] * l1_ratio, "L1 16 KiB chunks high": HASH_ALONE[16384][1] * l1_ratio,
             "at the bar, 64 KiB rows": S.BAR_MAC_TIMES / 128 * (S.blocks(65536) + 12) / (65536 / 128),
             "at the bar, L1 16 KiB chunks": S.BAR_MAC_TIMES / 128 * (S.blocks(16384) + 5) / (16384 / 128)}
    comps_byte = {"L0 16 KiB rows": (S.blocks(16384) + 12) / 16384, "L0 64 KiB rows": (S.blocks(65536) + 12) / 65536,
                  "L1 16 KiB chunks": (S.blocks(16384) + 5) / 16384}
    out: dict = {"inputs": {"sass": sass, "best_sass": best, "committer_excess_a_compression": excess, "rows_16k": r16,
                            "l1_time_ratio": l1_ratio, "rates_units_byte": rates, "mask_registers": price["parts"]["mask_registers"],
                            "mask_registers_b1_estimate": MASK_REGS_B1, "units_s": UNITS_S, "latency": lat},
                 "instance": price, "designs": {}}
    sizes = sorted(rows.by)
    for design in SSN:
        b = ssn_bytes(design)
        d: dict = {"bytes": {k: v for k, v in b.items() if k != "calls"}, "calls": b["calls"], "points": {}}
        for pname, (real, m) in POINTS.items():
            pt: dict = {"hiding": {}, "opening_ands": {}}
            for layout in ("L0", "L1", "L2", "L2, a window"):
                for R in (sizes if layout == "L0" else [S.ROWS_16K]):
                    sp = salts_per_mac(layout, real, b, R)
                    n = sum(sp.values())
                    key = f"{layout} {R // 1024} KiB rows" if layout == "L0" else layout
                    pt["hiding"][key] = {
                        "salts_per_mac": sp,
                        **{f: {p: n * v for p, v in pr.items()} for f, pr in price.items() if f not in ("parts", "compression")},
                        "by part, unforced floor": {k: v * price["unforced"]["floor"] for k, v in sp.items()},
                        "ands_of_stock": n * HIDING_ANDS / S.ANDS_MAC}
                    pt["opening_ands"][key] = {f: opening_ands(layout, b, real, R, f) for f in (False, True)}
            need = {p: b["words_bpm"] * v / TARGET for p, v in price["unforced"].items()}
            need_a = {p: b["words_bpm"] * v / TARGET for p, v in price["forced (a)"].items()}
            pt["flat rows for words at the target"] = {
                "bytes unforced": need, "bytes forced (a)": need_a,
                "opening ANDs unforced floor": S.blocks(int(need["floor"])) * S.ANDS_COMPRESSION,
                "opening ANDs forced (a) floor": S.blocks(int(need_a["floor"])) * S.ANDS_COMPRESSION}
            chain = None
            if lat:
                pt["seed_path"] = {}
                launches = CALLS * lat["launch_us"] * 1e-3
                for lay, (row, arity) in CHAINS.items():
                    comps = N.LAYERS * sum(seed_path(real, k, row, arity) for _, k in N.LINEARS.values())
                    v: dict = {"compressions_a_forward": comps, "compressions_a_call": comps / CALLS, "by_latency": {}}
                    for ln, ns in lat["per_compression_ns"].items():
                        ms = comps * ns * 1e-6
                        v["by_latency"][ln] = {"ms": ms, "ms_with_a_launch_a_call": ms + launches, "t2_floor": {
                            eng: (st + ms + launches) / st for eng, (st, _g, _s) in N.ENGINES.get(pname, {}).items()}}
                    pt["seed_path"][lay] = v
                chain = pt["seed_path"]["row-seg rows, binary (L1)"]["by_latency"]["measured"]["ms_with_a_launch_a_call"]
            pt["t2"] = t2(design, b, pname, real, m, rates, chain)
            if lat:
                for e in pt["t2"]["engines"].values():
                    e["compressions a call for the seed path's T2 floor"] = {
                        f"≤ {x}, {ln}": max(0.0, ((x - 1) * e["stock_ms"] * 1e3 / CALLS - lat["launch_us"]) / (ns * 1e-3))
                        for x in (2, 1.1) for ln, ns in lat["per_compression_ns"].items()}
            hash_w1 = {k: {p: b["bpm"] * cb * w for p, w in (("floor", sass["w1_floor"]), ("alone", sass["w1_alone"]))}
                       for k, cb in comps_byte.items()}
            pt["shaspeed"] = {k: {"hash_w1_mac": v,
                                  "compressions_a_useful_mac": b["bpm"] * comps_byte[k],
                                  **{f"delta_{bud}": {p: (BUDGET - t1) / w for p, w in v.items()} for bud, t1 in
                                     (("whole budget", 0.0), ("after SSN's terms, coins credited", SSN_T1[design]["credited"]),
                                      ("after SSN's terms, coins uncredited", SSN_T1[design]["uncredited"]))},
                                  "instructions_a_compression_floor": BUDGET / v["floor"] * sass["instructions"],
                                  "of_stock_a_register_a_compression_floor": b["bpm"] * comps_byte[k] * PRICE_REG["floor"],
                                  "committer_excess_of_stock": {p: b["bpm"] * comps_byte[k] * excess[p] for p in ("floor", "alone")}}
                              for k, v in hash_w1.items()}
            # each layout's compressions an activation row a layer, and whether its bulk hashing reads a per-call value: a row
            # digest reads only the row, so a repeated A_j (or row) reuses it whatever wraps it; a frame-v3 leaf reads its
            # domain id, rank and position, so leaves of the call's own domain (bound to pouw.salt, the forward and the call)
            # differ in every call and every row
            a_comps = {"L0 (row-seg, a leaf a row)": (sum(segs(k) + HM96_LEAF + TREE_LEAF for _, k in N.LINEARS.values()), False),
                       "L1 (row-seg rows in the call's subtree)": (sum(segs(k) + TREE_LEAF for _, k in N.LINEARS.values()), False),
                       "L1 (flat SHA-512 rows in the call's subtree)": (sum(S.blocks(k) + TREE_LEAF for _, k in N.LINEARS.values()), False),
                       **{f"L1 ({seg} B frame-v3 segment leaves of the call's domain)":
                          (sum(-(-k // seg) * (frame_leaf(seg) + NODE[2]) for _, k in N.LINEARS.values()), True)
                          for seg in (1024, 256)}}
            stock = N.ENGINES.get(pname, {}).get("sync", (None,))[0]
            pre = {}
            for lay, (c_layer, per_call) in a_comps.items():
                comps = N.LAYERS * c_layer
                ms = comps * real * r16["c_ns"] * 1e-6
                pre[lay] = {"compressions_token": comps, "reads a per-call value": per_call,
                            "w1": {p: comps * w / MACS_TOKEN for p, w in (("floor", sass["w1_floor"]), ("alone", sass["w1_alone"]))},
                            "ands_of_stock": comps * S.ANDS_COMPRESSION / MACS_TOKEN / S.ANDS_MAC,
                            "ms": ms, "of_stock_step": ms / stock if stock else None,
                            "share_of_hashed_bytes": b["a_bpm"] / b["bpm"]}
            pt["pre_noise"] = pre
            d["points"][pname] = pt
        out["designs"][design] = d
    out["v4"] = v4(price, lat, sizes)
    h200_mac_s = S.CENSUS["H200 int8 dense ops/s"] / 2 * MFU_SERVE
    periods = {"window": S.CENSUS["window s"], **S.PERIODS}
    scopes = {"one H200": {p: h200_mac_s * s for p, s in periods.items()},
              "datacenter": {p: S.CENSUS["datacenter GPUs"] * h200_mac_s * s for p, s in periods.items()},
              "world": {p: S.CENSUS["world FLOP a year"] / 2 * s / S.PERIODS["year"] for p, s in periods.items()}}
    pooled: dict = {"stock_macs": scopes, "stock_ands": {sc: {p: m * S.ANDS_MAC for p, m in ps.items()} for sc, ps in scopes.items()}}
    for key in ("L0 16 KiB rows", "L0 256 KiB rows", "L1", "L2", "L2, a window"):
        for pname in ("T 8,192", "T 512, 64 real rows", "T 512, 512 real rows"):
            h = out["designs"]["ssn_1l"]["points"][pname]["hiding"][key]
            pooled[f"hiding {key}, {pname}"] = {
                f: {sc: {p: {"share_floor": h[f]["floor"], "ands_floor": mac * S.ANDS_MAC * h[f]["floor"],
                             "ands_alone": mac * S.ANDS_MAC * h[f]["alone"]}
                         for p, mac in ps.items()} for sc, ps in scopes.items()}
                for f in ("unforced", "forced (a)", "unforced, unrolled mask", "forced (a), unrolled mask") if f in h}
    pt = out["designs"]["ssn_1l"]["points"]["T 8,192"]
    for name, share in (("hashing before the noise, L1 flat rows", pt["pre_noise"]["L1 (flat SHA-512 rows in the call's subtree)"]["w1"]),
                        ("hashing before the noise, L1 1 KiB segment leaves of the call's domain",
                         pt["pre_noise"]["L1 (1024 B frame-v3 segment leaves of the call's domain)"]["w1"])):
        pooled[name] = {sc: {p: {"share_floor": share["floor"], "ands_floor": mac * S.ANDS_MAC * share["floor"]}
                             for p, mac in ps.items()} for sc, ps in scopes.items()}
    out["periods"] = pooled
    Path(sys.argv[4]).write_text(json.dumps(out, indent=1))

    p = price
    print(f"a salt, W1 floor / alone / time: unforced {p['unforced']['floor']:,.0f} / {p['unforced']['alone']:,.0f} / "
          f"{p['unforced']['time']:,.0f}; forced (a) {p['forced (a)']['floor']:,.0f} / {p['forced (a)']['alone']:,.0f} / "
          f"{p['forced (a)']['time']:,.0f}; M_key·y {p['parts']['mask_registers']} registers (b1 estimate {MASK_REGS_B1}); "
          f"L1 time a byte {l1_ratio:.4f} of 16 KiB rows'; committer's build over the best {excess['registers']} registers, "
          f"{excess['floor']:,.0f} / {excess['alone']:,.0f} W1 a compression")
    shown = [f for f in ("unforced", "unforced, unrolled mask", "unforced, b1 mask (estimate)", "forced (a)",
                         "forced (a), unrolled mask") if f in p]
    for design, d in out["designs"].items():
        bb = d["bytes"]
        print(f"\n{design}: α {bb['alpha']:.4f}, words {bb['words_bpm']:.5f} + A {bb['a_bpm']:.6f} = {bb['bpm']:.5f} B a useful MAC")
        for pname, pt in d["points"].items():
            print(f"  {pname}")
            for key, h in pt["hiding"].items():
                print(f"    hiding {key:<16} " + "; ".join(
                    f"{f} {h[f]['floor']:.4%} / {h[f]['alone']:.4%}" + (f" / {h[f]['time']:.4%}" if "time" in h[f] else "")
                    for f in shown) + f"; ANDs {h['ands_of_stock']:.5%}")
            t = pt["t2"]
            for eng, e in t["engines"].items():
                cells = "; ".join(f"{rn} {x['serial']:.1f}/{x['fused']:.1f}/{x['overlap']:.1f}" for rn, x in e.items()
                                  if isinstance(x, dict) and "serial" in x and rn in ("16 KiB low", "L1 16 KiB chunks low",
                                                                                      "at the bar, L1 16 KiB chunks"))
                fr = e["bytes a useful MAC for T2 at the bar, L1 16 KiB chunks"]
                print(f"    T2 {eng} (serial/fused/overlap; {t['coins_src']} {t['coins_ms']:.2f} ms): {cells}; readings: beside-step "
                      f"{e['reading: beside-the-step price on every byte, sheet clock']:.1f}, alone "
                      f"{e['reading: alone 64 KiB high, sheet clock, HBM export']:.1f}; stock "
                      f"{e['stock_mac_times_a_useful_mac']:.2f} MAC-times a useful MAC; B/MAC for T2 "
                      + ", ".join(f"{k} {v:.2e}" for k, v in fr.items()))
            for k, v in pt["shaspeed"].items():
                print(f"    δ {k}: {v['delta_whole budget']['floor']:.2e} / {v['delta_whole budget']['alone']:.2e} "
                      f"({v['instructions_a_compression_floor']:.3f} instructions); a register a compression "
                      f"{v['of_stock_a_register_a_compression_floor']:.2%} of stock; committer's excess "
                      f"{v['committer_excess_of_stock']['floor']:.1%} / {v['committer_excess_of_stock']['alone']:.1%}")
            for k, v in pt["pre_noise"].items():
                print(f"    pre-noise {k}{'' if v['reads a per-call value'] else ', replayable'}: W1 {v['w1']['floor']:.3f} / "
                      f"{v['w1']['alone']:.3f} of stock, ANDs {v['ands_of_stock']:.4%}"
                      + (f", {v['ms']:.2f} ms ({v['of_stock_step']:.1%} of the stock step)" if v["of_stock_step"] else ""))
            for k, v in pt.get("seed_path", {}).items():
                print(f"    seed path, {k}: {v['compressions_a_forward']} compressions a forward one after another "
                      f"({v['compressions_a_call']:.2f} a call); " + "; ".join(
                          f"{ln} {x['ms']:.2f} ms ({x['ms_with_a_launch_a_call']:.2f} with a launch a call)"
                          + "".join(f", T2 floor {eng} {f:.2f}" for eng, f in x["t2_floor"].items())
                          for ln, x in v["by_latency"].items()))
            for eng, e in pt["t2"]["engines"].items():
                if "compressions a call for the seed path's T2 floor" in e:
                    print(f"    T2 {eng}: compressions a call for the seed path's T2 floor " + ", ".join(
                        f"{k} {n:.1f}" for k, n in e["compressions a call for the seed path's T2 floor"].items()))
    print("\nssn_1l on the V4-like layers (hiding unforced, then forced (a), floor / alone)")
    for name, e in out["v4"].items():
        shown_l = [k for k in e["hiding"] if not k.startswith("L0") or k in ("L0 16 KiB rows", "L0 1024 KiB rows")]
        print(f"  {name}: {e['rows_a_linear']:g} rows a linear, {e['bytes_a_mac']:.4f} B a MAC; " + "; ".join(
            f"{k} {e['hiding'][k]['unforced']['floor']:.4%} / {e['hiding'][k]['unforced']['alone']:.4%}, "
            f"{e['hiding'][k]['forced (a)']['floor']:.4%} / {e['hiding'][k]['forced (a)']['alone']:.4%}" for k in shown_l))
        for g, v in e.get("seed_path", {}).items():
            print(f"    seed path, {g}: {v['us']:.0f} us a layer beside {v['gemm_us']:.1f} us of GEMM, "
                  f"T2 floor over the GEMM {v['t2_floor_over_gemm']:.1f}")


if __name__ == "__main__":
    main()
