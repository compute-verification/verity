"""Committed bits per GEMM coordinate and session overhead for C-Flock's layout today against the restructurings in the
proofs-arch study, from measured inputs only:

* M0 #20 (`r20260930-184956-61fe`, run files `art:6a148912557d56a09258d3713668f7284e0a2ecd19d8b4f83d2f667547d9a066`): the
  layouts, prove times (`slowdown.json`), native times and the loopback verifier's `live.verify_s` (`results.jsonl`); the
  verify_s medians over M0's 31 runs of Sep 30 (`MEDIANS`, context only: the hosts were loaded and the spread is 2x).
* `sha512_circuit.compression3`'s census (83,451 rows a compression, three to a 2^18 slot) and P1's step template
  (`step_template.py`: 7,050 rows, decode 60 ANDs a word).
* the uniform-copies bench (`uniform_copies.rs`, JSON lines, `--bench`): the lincheck verifier per rep at today's k_log with a
  unit type of today's nnz, in C-Flock's `BlockCircuit` form and template-aware.

Session overhead is (prove + verify) per coordinate over native, serial (one statement at a time) or overlapped (the verifier
checks statement i while the prover proves i + 1, so the slower side sets the pace).

At a fixed m = 35 the prover's time per statement is today's, scaled on its witness share by ANDs per statement; a layout
that needs fewer committed bits per coordinate packs more coordinates into the same statement.
"""
from __future__ import annotations

import argparse
import json
import math

M_STATEMENT = 35
SHA_SLOT_BITS = (1 << 18) / 3
SHA_ANDS = 81_402
STEP_SLOT = 1 << 13
STEP_ANDS = 6_313
DECODE_BITS = 16 + 60 + 21
DECODE_ANDS = 60
WITNESS_SHARE = 0.1164 / 0.7672
REP1_SHARE = 0.2748 / 0.7672
#: M0 #20's GPU zerocheck per rep at m = 35 (both shapes' four buckets agree to 3%)
GPU_ZEROCHECK_S = 0.109
#: P1's step with products, alignment, max exponent and normalization as lookups: 16 + 32 + 15 + 2 lookups leave about
#: 2,450 committed bits, a 2^12 slot
LOOKUPS_PER_STEP = 65

TODAY = {
    2048: {"block_log": 26, "tile": (4, 4), "unit_slot_log": 21, "unit_ands": 1_103_522, "prove_s": 0.7798, "blocks": 512,
           "verify_s": 10.4628, "native_shape": "256x16384x2048", "coords": 256 * 16384, "native_s": 8.025376319885255e-05,
           "bench_unit_log": 20},
    8192: {"block_log": 25, "tile": (1, 1), "unit_slot_log": 23, "unit_ands": 4_414_754, "prove_s": 0.7514, "blocks": 1024,
           "verify_s": 27.4158, "native_shape": "256x2048x8192", "coords": 256 * 2048, "native_s": 3.9102079868316644e-05,
           "bench_unit_log": 22},
}
#: verify_s medians over M0's Sep 30 runs, by (K, k_log, nbl): n runs, median, min, max
MEDIANS = {(2048, 26, 9): (7, 10.84, 9.97, 13.10), (2048, 26, 7): (11, 8.40, 6.43, 14.65), (2048, 24, 11): (8, 9.26, 5.97, 10.06),
           (8192, 25, 10): (19, 23.02, 16.88, 38.93), (8192, 25, 9): (12, 15.91, 8.99, 29.10)}


def compressions(K: int) -> int:
    return K // 64 + 3


def today(K: int) -> dict:
    t = TODAY[K]
    P, C = t["tile"]
    sha = (P + C) * compressions(K)
    per_block_bits = 1 << t["block_log"]
    unit_bits = P * C * (1 << t["unit_slot_log"])
    sha_bits = math.ceil(sha / 3) * (1 << 18)
    return {"bits_per_coord": per_block_bits / (P * C), "coords_per_statement": t["blocks"] * P * C,
            "ands_per_coord": t["unit_ands"] + sha * SHA_ANDS / (P * C),
            "hash_share_bits": sha_bits / per_block_bits, "unit_share_bits": unit_bits / per_block_bits,
            "hash_share_ands": sha * SHA_ANDS / (sha * SHA_ANDS + P * C * t["unit_ands"])}


def unit_step_bits(K: int, step_slot: int) -> int:
    """A coordinate as K/16 uniform step slots plus one slot for its final conversion."""
    return (K // 16 + 1) * step_slot


def per_tile(K: int, step_slot: int = STEP_SLOT) -> dict:
    """V1: step slots, operands decoded once per tile row/column, rows hashed per tile as today; the best tile in a 2^26 block."""
    best = None
    for P in range(1, 33):
        for C in range(1, 33):
            bits = P * C * unit_step_bits(K, step_slot) + (P + C) * K * DECODE_BITS + math.ceil((P + C) * compressions(K) / 3) * (1 << 18)
            if bits > 1 << 26:
                continue
            per = (1 << 26) / (P * C)
            if best is None or per < best["bits_per_coord"]:
                ands = P * C * (K // 16) * STEP_ANDS + (P + C) * K * DECODE_ANDS + (P + C) * compressions(K) * SHA_ANDS
                best = {"tile": [P, C], "bits_per_coord": per, "coords_per_statement": (1 << (M_STATEMENT - 26)) * P * C,
                        "ands_per_coord": ands / (P * C)}
    return best


def per_statement(K: int, step_slot: int = STEP_SLOT, weights_bound: bool = False, max_rows: int = 256,
                  hashed_in_circuit: bool = True) -> dict:
    """V2/V3: rows hashed and decoded once per statement (a rows table glued to the units table); a P x C grid in 2^35 bits.
    Step slots need not fill a power of two, so C is any count. `weights_bound`: the weight columns are bound once at
    registration and hashed in no statement. `hashed_in_circuit=False`: no row is hashed in the circuit (every row bound by a
    reusable commitment)."""
    unit = unit_step_bits(K, step_slot)

    def hashed_rows(P: int, C: int) -> int:
        return 0 if not hashed_in_circuit else P + (0 if weights_bound else C)

    def bits(P: int, C: int) -> int:
        hashed = hashed_rows(P, C)
        return P * C * unit + (P + C) * K * DECODE_BITS + math.ceil(hashed * compressions(K) / 3) * (1 << 18)

    best = None
    for P in range(1, max_rows + 1):
        lo, hi = 0, 1 << 20
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if bits(P, mid) <= 1 << M_STATEMENT else (lo, mid - 1)
        C = lo
        if C == 0 or (best is not None and P * C <= best["coords_per_statement"]):
            continue
        hashed = hashed_rows(P, C)
        ands = P * C * (K // 16) * STEP_ANDS + (P + C) * K * DECODE_ANDS + hashed * compressions(K) * SHA_ANDS
        sha = math.ceil(hashed * compressions(K) / 3) * (1 << 18)
        best = {"grid": [P, C], "coords_per_statement": P * C, "bits_per_coord": (1 << M_STATEMENT) / (P * C),
                "ands_per_coord": ands / (P * C), "fill": bits(P, C) / (1 << M_STATEMENT), "hash_share_bits": sha / bits(P, C)}
    return best


def prove_per_statement(K: int, ands_per_statement: float, one_rep: bool = False) -> float:
    t = TODAY[K]
    base = today(K)
    ratio = ands_per_statement / (base["ands_per_coord"] * base["coords_per_statement"])
    s = t["prove_s"] * ((1 - WITNESS_SHARE) + WITNESS_SHARE * ratio)
    if one_rep:
        # rep 1 reuses rep 0's witness and commitment; a single rep at a higher target needs ~1.28x Ligerito queries
        s *= (1 - REP1_SHARE) + 0.28 * 0.1009 / 0.7672
    return s


def lincheck_saving(bench: list[dict], K: int) -> dict | None:
    """Seconds per statement (two reps) the template-aware verifier saves on the lincheck at today's k_log and a unit type of
    today's nnz, over every bench run of that point (any thread count; the VM is shared, so absolute times vary by run).
    `low` is the conservative figure the model uses."""
    t = TODAY[K]
    get = (lambda v: v["min"] if isinstance(v, dict) else v)
    runs = [r for r in bench if r.get("k_log") == t["block_log"] and r.get("unit_log") == t["bench_unit_log"]]
    if not runs:
        return None
    save = [2 * (get(r["verify_cflock_unit_s"]) - get(r["verify_tmpl_step_s"])) for r in runs]
    kept = [get(r["verify_tmpl_step_s"]) / get(r["verify_cflock_unit_s"]) for r in runs]
    scattered = [2 * (get(r["verify_cflock_unit_scattered_s"]) - get(r["verify_tmpl_step_s"])) for r in runs
                 if "verify_cflock_unit_scattered_s" in r]
    return {"low": min(save), "high": max(save), "runs": len(runs), "template_over_today": [min(kept), max(kept)],
            "scattered_unit": scattered,
            "today_per_rep_s": [get(r["verify_cflock_unit_s"]) for r in runs],
            "template_per_rep_s": [get(r["verify_tmpl_step_s"]) for r in runs]}


def overhead(K: int, s_per_coord: float) -> float:
    t = TODAY[K]
    return s_per_coord * t["coords"] / t["native_s"]


def lookup_price(lookup: list[dict]) -> dict | None:
    """A lookup's prover cost from `lookup_cost.rs` (CPU, largest sizes): half a batched product-GKR leaf, in committed bits'
    worth of zerocheck work, and in GPU seconds by the CPU ratio applied to M0 #20's GPU zerocheck."""
    zc = max((r for r in lookup if r.get("kind") == "zerocheck"), key=lambda r: r["m"], default=None)
    gkr = max((r for r in lookup if r.get("kind") == "product_gkr_batched"), key=lambda r: r["mu"], default=None)
    if zc is None or gkr is None:
        return None
    bits = (gkr["s_per_leaf"] / 2) / zc["s_per_bit"]
    return {"zerocheck_bits_per_lookup": bits, "gpu_s_per_lookup": bits * GPU_ZEROCHECK_S / (1 << M_STATEMENT),
            "cpu_zerocheck_s_per_bit": zc["s_per_bit"], "cpu_gkr_s_per_leaf": gkr["s_per_leaf"], "zc_m": zc["m"], "gkr_mu": gkr["mu"]}


def model(bench: list[dict], lookup: list[dict] | None = None) -> dict:
    out = {}
    price = lookup_price(lookup or [])
    for K in (2048, 8192):
        t = TODAY[K]
        base = today(K)
        saving = lincheck_saving(bench, K)
        save = saving["low"] if saving else None
        rows = {"V0 today": dict(base, prove_s=t["prove_s"], verify_s=t["verify_s"])}
        rows["V0 + template-aware lincheck verifier"] = dict(base, prove_s=t["prove_s"],
                                                             verify_s=t["verify_s"] - save if save is not None else None)
        v1 = per_tile(K)
        rows["V1 step slots"] = dict(v1)
        v2 = per_statement(K)
        rows["V2 + rows once per statement"] = dict(v2)
        v3 = per_statement(K, weights_bound=True)
        rows["V3 + weights bound at registration"] = dict(v3)
        rows["V2 - no row hashed in the circuit (needs a decision)"] = per_statement(K, hashed_in_circuit=False)
        rows["V2 + lookups (2^12 step, model)"] = per_statement(K, step_slot=1 << 12)
        for name, r in rows.items():
            if name.startswith("V0"):
                continue
            r["prove_s"] = prove_per_statement(K, r["ands_per_coord"] * r["coords_per_statement"])
            r["verify_s"] = max(t["verify_s"] - save, 0.0) if save is not None else None
        for name, r in rows.items():
            if name.startswith(("V2", "V3")):
                # A shared row reaches every block, which `I ⊗ A_0` cannot wire: a lincheck over the whole 2^35 (a closed-form
                # broadcast matrix for the verifier), priced at one zerocheck pass a rep
                r["glue_prove_s"] = 2 * GPU_ZEROCHECK_S
                r["prove_s"] += r["glue_prove_s"]
        lk = rows["V2 + lookups (2^12 step, model)"]
        if price is not None:
            lk["lookups_per_statement"] = lk["coords_per_statement"] * (K // 16) * LOOKUPS_PER_STEP
            lk["lookup_prove_s"] = 2 * lk["lookups_per_statement"] * price["gpu_s_per_lookup"]
            lk["prove_s"] += lk["lookup_prove_s"]
        one = dict(rows["V2 + rows once per statement"])
        one["glue_prove_s"] = GPU_ZEROCHECK_S
        one["prove_s"] = prove_per_statement(K, one["ands_per_coord"] * one["coords_per_statement"], one_rep=True) + GPU_ZEROCHECK_S
        one["verify_s"] = one["verify_s"] / 2 if one.get("verify_s") is not None else None
        rows["V2 + one rep (needs a target decision)"] = one
        for name, r in rows.items():
            n = r["coords_per_statement"]
            r["prover_overhead"] = overhead(K, r["prove_s"] / n)
            if r.get("verify_s") is not None:
                r["session_overhead_serial"] = overhead(K, (r["prove_s"] + r["verify_s"]) / n)
                r["session_overhead_overlapped"] = overhead(K, max(r["prove_s"], r["verify_s"]) / n)
            r["bits_per_mac"] = r["bits_per_coord"] / K
            r["gain_bits_vs_today"] = base["bits_per_coord"] / r["bits_per_coord"]
        out[K] = {"shape": t["native_shape"], "lincheck_saving": saving, "today_census": base, "rows": rows}
    out["lookup_price"] = price
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="append", default=[], help="uniform_copies JSON lines")
    ap.add_argument("--lookup", action="append", default=[], help="lookup_cost JSON lines")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    read = (lambda paths: [json.loads(line) for p in paths for line in open(p) if line.startswith("{")])
    res = {"question": "how many committed bits per GEMM coordinate, and what session overhead, does each restructuring leave?",
           "medians_verify_s": {f"K={k[0]} k_log={k[1]} nbl={k[2]}": v for k, v in MEDIANS.items()},
           "model": model(read(a.bench), read(a.lookup))}
    fmt = (lambda v: "-" if v is None else f"{v:.3g}")
    print(f"lookup price: {res['model']['lookup_price']}")
    for K, m in res["model"].items():
        if K == "lookup_price":
            continue
        s = m["lincheck_saving"]
        print(f"K={K} ({m['shape']}), lincheck saving per statement {fmt(s and s['low'])}-{fmt(s and s['high'])} s over "
              f"{s and s['runs']} runs; census {json.dumps({k: round(v, 3) for k, v in m['today_census'].items()})}")
        for name, r in m["rows"].items():
            print(f"  {name:40s} bits/MAC {r['bits_per_mac']:6.0f} x{r['gain_bits_vs_today']:5.2f} coords/stmt {r['coords_per_statement']:6d}"
                  f" prove {r['prove_s']:.3f}s verify {fmt(r.get('verify_s'))}s overhead: prover {fmt(r['prover_overhead'])}"
                  f" session serial {fmt(r.get('session_overhead_serial'))} overlapped {fmt(r.get('session_overhead_overlapped'))}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
