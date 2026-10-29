"""hm96's hiding work and the hashing of served sparse noise (SSN) in Llama-3.2-1B's served step on the H200, priced: what the
honest prover alone does to hide its committed values, as a share of stock at T 8,192 and T 512 (64 and 512 rows), under each
commitment layout and each way of forcing it, the proof's cost to open what it draws, and SSN's bytes, t_H/t_S and ShaSpeed's
precision. Stdlib only, no GPU.

    python3 benchmarks/served_commit/np_hide_price.py SASS_JSON BEST_SASS_JSON ROWS_JSON OUT_JSON

SASS_JSON and ROWS_JSON are sw_hash_price.py's inputs: a SHA-512 compression's W1 work in the committer's build, and each row
size's leaf kernels' GPU time (the salt kernel's derived and given salts, a compression's time). BEST_SASS_JSON is the build
with the fewest registers written a compression: ShaSpeed can credit no more than it, so the committer's excess over it is
honest-only work, and each register a compression is priced as a share of stock at SSN's volume.

An hm96 instance is one salt: y = PRF_ks(pos) (three compressions from round 8), c = SHA-512(salt_prefix ‖ y) (two) and
b = x ⊕ M_key·y. A prover that reuses one salt skips all three, and nothing the gadget checks changes, so they are honest-only
work, priced a salt:
  unforced: the PRF, c and M_key·y; at the floor and alone each compression at its SASS W1, M_key·y at its registers written
    (the H200 byte-table form, counted from the source) at 32 (floor) or one LOP3's measured price (alone); "time" is the salt
    kernel's measured time a salt less the frame leaf's first block, in MAC-times;
  forced (a): the drawn unit's proof checks y = PRF_ks(pos) against a committed ks and checks c, so ShaSpeed credits both
    and M_key·y alone remains (time: the salt kernel from given salts less c and the frame leaf's first block);
  SHA-512 mask (e2): M_key·y replaced by SHA-512(mask_prefix ‖ y), credited with the PRF and c under (a), so nothing remains;
  b1 mask (an estimate): M_key·y bit for bit on the tensor cores (mma m16n8k256 .b1 .and.popc, M's fragments resident in
    eight warps' registers), its registers written counted from the shape, not compiled.
The term is a salt's price times the salts a useful MAC, and the layout sets the salts:
  L0, today: one hm96 leaf per activation row of A per call (row-seg) and one per R-byte row of committed words;
  L1, a leaf over a subtree: one hm96 instance per call, its inner digest a frame-v3-sha512 tree over the call's chunks
    (the words and the next call's A), so A's public root stays a hiding root while openings cost a chunk and its path;
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
which the hash alone, at the bar, fits in (T2 − 1) stock steps. Hashing before the noise is A's commitment: under L0 a
row-seg row with its hm96 leaf and tree share (its salt is in the hiding term); under L1 a row-seg or flat SHA-512 row and its
tree share.
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
PRF_SHARE = S.ANDS_FROM_8 / S.ANDS_COMPRESSION      # a compression from round 8 over a whole one (W1 taken in the same ratio)
# M_key·y in the H200 split form (byte table in shared memory), registers written a salt: 16 passes of 12 salt bytes, each
# 5 LDS.128 (20 registers), 10 three-input XORs and 3 for the byte and its address; each pass 37 conditional word shifts and
# 16 funnel shifts with their 16 XORs
MASK_REGS = 16 * (12 * (20 + 10 + 3) + 37 + 32)
# the b1 form a salt: 24 mma (512 x 1536 bits over 16 x 256 tiles, eight salts a column set) writing 4 accumulator registers in
# each of 32 lanes, y's fragments (2 registers a k-step in each of eight warps), and two instructions an output bit for parity
MASK_REGS_B1 = 24 * 4 * 32 + 8 * 6 * 2 * 32 // 8 + 2 * 512
PRICE_REG = {"floor": 32.0, "alone": S.PRICE["alone"]}
TREE_LEAF = 3                                       # sw_hash_price's tree: compressions a leaf beside the row digest
HM96_LEAF = 2                                       # H(leaf_prefix ‖ b ‖ c): 128 bytes and the padding block
WINDOW_STEP_S = 0.1
HASH_ALONE = {16384: (2043, 2159), 65536: (1928, 2010)}   # hm96 rows hashed alone, units a byte (low, high readings)
STEP_BESIDE = 981                                   # beside the served step's GEMM, units a byte, while the step runs
EXPORT_WORD = {"fold": 189, "stg": 848}             # extra units a word out of the accumulator at L 32: in-register fold, HBM
COIN_CALL_US = 9.3                                  # a call's coin expansion at its latency floor
CALLS = 4 * N.LAYERS
UNIT_OUT_BYTES = 262_144 // 8                       # a proof block's outputs at the largest public size class
BUDGET = 0.01
TARGET = 0.001
SSN = {"ssn_1l": lambda k: 1 + math.ceil((k - 28) / 31), "ssn_a": lambda k: math.ceil(k / 28)}
SSN_T1 = {"ssn_1l": {"credited": 0.00049, "uncredited": 0.00255}, "ssn_a": {"credited": 0.00054, "uncredited": 0.00944}}
POINTS = {**N.POINTS, "decode, 16 rows": (16, 64), "decode, 1 row": (1, 64)}   # (real rows, the GEMM table's m)
MACS_TOKEN = N.LAYERS * sum(n * k for n, k in N.LINEARS.values())
HIDING_ANDS = 3 * S.ANDS_FROM_8 + 2 * S.ANDS_COMPRESSION


def segs(k: int) -> int:
    """A row-seg row of k bytes: its segments' compressions and the outer hash's."""
    s = -(-k // 1024)
    return 9 * s + (64 * s + 17 + 127) // 128


def instance_prices(sass: dict, rows: S.Rows) -> dict:
    """W1 units a salt: the honest-only work, unforced and with the PRF and c forced."""
    w1c = {"floor": sass["w1_floor"], "alone": sass["w1_alone"]}
    r = rows.by[S.ROWS_16K]
    mt = N.RULER * r["mhz"] * 1e6 * 1e-9
    prf = {p: 3 * PRF_SHARE * w for p, w in w1c.items()}
    c = {p: 2 * w for p, w in w1c.items()}
    mask = {p: MASK_REGS * PRICE_REG[p] for p in w1c}
    b1 = {p: MASK_REGS_B1 * PRICE_REG[p] for p in w1c}
    return {
        "parts": {"prf": prf, "c": c, "mask": mask, "mask_registers": MASK_REGS, "b1_mask": b1, "b1_mask_registers": MASK_REGS_B1,
                  "mask_time": (r["salt_given_ns_leaf"] - (2 + 65 / 80) * r["c_ns"]) * mt,
                  "salt_kernel_ns": r["salt_ns_leaf"], "given_ns": r["salt_given_ns_leaf"], "compression_ns": r["c_ns"]},
        "unforced": {**{p: prf[p] + c[p] + mask[p] for p in w1c}, "time": (r["salt_ns_leaf"] - 65 / 80 * r["c_ns"]) * mt},
        "forced (a)": {**mask, "time": (r["salt_given_ns_leaf"] - (2 + 65 / 80) * r["c_ns"]) * mt},
        "unforced, b1 mask (estimate)": {p: prf[p] + c[p] + b1[p] for p in w1c},
        "forced (a), b1 mask (estimate)": dict(b1),
        "forced (a), SHA-512 mask (e2)": {"floor": 0.0, "alone": 0.0, "time": 0.0},
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
    extra = 2 * S.ANDS_COMPRESSION + (3 * S.ANDS_FROM_8 if forced else 0)
    if layout == "L0":
        return {"words row, whole": S.blocks(R) * S.ANDS_COMPRESSION + extra,
                "words row over the unit's own outputs": max(0, S.blocks(R) - S.blocks(UNIT_OUT_BYTES)) * S.ANDS_COMPRESSION + extra}
    big = max(rows * c["words_row"] for c in b["calls"].values())
    depth = math.ceil(math.log2(max(2, big / UNIT_OUT_BYTES)))
    return {"path depth": depth, "path and gadget": 2 * depth * S.ANDS_COMPRESSION + extra}


def t2(design: str, b: dict, pname: str, rows: int, m: int, rates: dict) -> dict:
    g = N.GEMM[min(N.GEMM, key=lambda x: abs(x - m))]
    gemm_ms = (b["alpha"] - 1) * g["plain_us"] * 1e-3 * rows / m
    B = (b["words"] + b["a"]) * rows
    words = b["words"] / 4 * rows
    export = {e: words * u / UNITS_S * 1e3 for e, u in EXPORT_WORD.items()}
    coins = CALLS * COIN_CALL_US * 1e-3
    out = {"GB_forward": B / 1e9, "gemm_extra_ms": gemm_ms, "export_ms": export, "coins_ms": coins, "hash_ms": {}, "engines": {}}
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
                 f"{sched} ≤ {x}": max(0.0, budget) * 1e-3 * UNITS_S / per_byte / fwd_macs for x in (2, 1.1)
                 for sched, budget, per_byte in (("overlap", x * stock, bar),
                                                 ("serial, HBM export", (x - 1) * stock - gemm_ms - coins,
                                                  bar + EXPORT_WORD["stg"] / 4))}}
        for rn, h in out["hash_ms"].items():
            e[rn] = {"serial": (stock + gemm_ms + h + export["stg"] + coins) / stock,
                     "fused": (stock + gemm_ms + h + export["fold"] + coins) / stock,
                     "overlap": max(stock + gemm_ms + export["fold"] + coins, h) / stock}
        e["reading: beside-the-step price on every byte, sheet clock"] = (
            stock + gemm_ms + B * (STEP_BESIDE + EXPORT_WORD["stg"] / 4) / SHEET_MAC_S * 1e3 + coins) / stock
        e["reading: alone 64 KiB high, sheet clock, HBM export"] = (
            stock + gemm_ms + B * (HASH_ALONE[65536][1] + EXPORT_WORD["stg"] / 4) / SHEET_MAC_S * 1e3 + coins) / stock
        out["engines"][eng] = e
    return out


def main() -> None:
    sass = json.loads(Path(sys.argv[1]).read_text())["compression"]
    best = json.loads(Path(sys.argv[2]).read_text())["compression"]
    rows = S.Rows(Path(sys.argv[3]))
    price = instance_prices(sass, rows)
    excess = {"registers": sass["registers_written"] - best["registers_written"],
              "floor": sass["w1_floor"] - best["w1_floor"], "alone": sass["w1_alone"] - best["w1_alone"]}
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
                            "l1_time_ratio": l1_ratio, "rates_units_byte": rates, "mask_registers": MASK_REGS,
                            "mask_registers_b1_estimate": MASK_REGS_B1, "units_s": UNITS_S},
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
            pt["t2"] = t2(design, b, pname, real, m, rates)
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
            a_comps = {"L0 (row-seg, a leaf a row)": sum(segs(k) + HM96_LEAF + TREE_LEAF for _, k in N.LINEARS.values()),
                       "L1 (row-seg rows in the call's subtree)": sum(segs(k) + TREE_LEAF for _, k in N.LINEARS.values()),
                       "L1 (flat SHA-512 rows in the call's subtree)": sum(S.blocks(k) + TREE_LEAF for _, k in N.LINEARS.values())}
            stock = N.ENGINES.get(pname, {}).get("sync", (None,))[0]
            pre = {}
            for lay, c_layer in a_comps.items():
                comps = N.LAYERS * c_layer
                ms = comps * real * r16["c_ns"] * 1e-6
                pre[lay] = {"compressions_token": comps,
                            "w1": {p: comps * w / MACS_TOKEN for p, w in (("floor", sass["w1_floor"]), ("alone", sass["w1_alone"]))},
                            "ands_of_stock": comps * S.ANDS_COMPRESSION / MACS_TOKEN / S.ANDS_MAC,
                            "ms": ms, "of_stock_step": ms / stock if stock else None,
                            "share_of_hashed_bytes": b["a_bpm"] / b["bpm"]}
            pt["pre_noise"] = pre
            d["points"][pname] = pt
        out["designs"][design] = d
    dc_mac_s = S.CENSUS["datacenter GPUs"] * S.CENSUS["H200 int8 dense ops/s"] / 2
    scopes = {"datacenter": {"window": dc_mac_s * S.CENSUS["window s"], **{p: dc_mac_s * s for p, s in S.PERIODS.items()}},
              "world": {p: S.CENSUS["world FLOP a year"] / 2 * s / S.PERIODS["year"] for p, s in S.PERIODS.items()}}
    pooled: dict = {"stock_macs": scopes}
    for key in ("L0 16 KiB rows", "L0 256 KiB rows", "L1", "L2", "L2, a window"):
        for pname in ("T 8,192", "T 512, 64 real rows", "T 512, 512 real rows"):
            h = out["designs"]["ssn_1l"]["points"][pname]["hiding"][key]
            pooled[f"hiding {key}, {pname}"] = {
                f: {sc: {p: {"w1_floor": mac * h[f]["floor"], "w1_alone": mac * h[f]["alone"],
                             "ands": mac * S.ANDS_MAC * h["ands_of_stock"] if f == "unforced" else None}
                         for p, mac in ps.items()} for sc, ps in scopes.items()}
                for f in ("unforced", "forced (a)")}
    pt = out["designs"]["ssn_1l"]["points"]["T 8,192"]
    for name, share in (("committer's SHA-512 build over the best, L1 16 KiB chunks",
                         pt["shaspeed"]["L1 16 KiB chunks"]["committer_excess_of_stock"]),
                        ("hashing before the noise, L1 flat rows", pt["pre_noise"]["L1 (flat SHA-512 rows in the call's subtree)"]["w1"])):
        pooled[name] = {sc: {p: {"w1_floor": mac * share["floor"], "w1_alone": mac * share["alone"]} for p, mac in ps.items()}
                        for sc, ps in scopes.items()}
    out["periods"] = pooled
    Path(sys.argv[4]).write_text(json.dumps(out, indent=1))

    p = price
    print(f"a salt, W1 floor / alone / time: unforced {p['unforced']['floor']:,.0f} / {p['unforced']['alone']:,.0f} / "
          f"{p['unforced']['time']:,.0f}; forced (a) {p['forced (a)']['floor']:,.0f} / {p['forced (a)']['alone']:,.0f} / "
          f"{p['forced (a)']['time']:,.0f}; M_key·y {MASK_REGS} registers (b1 estimate {MASK_REGS_B1}); "
          f"L1 time a byte {l1_ratio:.4f} of 16 KiB rows'; committer's build over the best {excess['registers']} registers, "
          f"{excess['floor']:,.0f} / {excess['alone']:,.0f} W1 a compression")
    shown = ("unforced", "unforced, b1 mask (estimate)", "forced (a)")
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
                print(f"    T2 {eng} (serial/fused/overlap): {cells}; readings: beside-step "
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
                print(f"    pre-noise {k}: W1 {v['w1']['floor']:.3f} / {v['w1']['alone']:.3f} of stock, ANDs {v['ands_of_stock']:.4%}"
                      + (f", {v['ms']:.2f} ms ({v['of_stock_step']:.1%} of the stock step)" if v["of_stock_step"] else ""))


if __name__ == "__main__":
    main()
