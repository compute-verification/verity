"""N-EGRESS: coin layouts whose network term is at most 1% of stock, priced on N-CARD's basis.

DitherInstance's guarantee (Security/Definitions/Pouw/Guarantees/DitherInstance.lean) charges n_max W_mac E in its
slack: E the bits the network certifier counts (capped at served traffic, the coin stream excluded), W_mac the honest
block's priced cost a MAC (its hashing and answers included) and n_max the most MACs one exchanged bit carries, through
OutsourceWithinEgress at beta = n_max W_mac. As a share of stock (W1: a MAC 1, a register write 32) the term is
n_max W_mac E / MACs. Each layout is priced on Llama-3.2-1B's linears at each point's rows: its coin bits, n_max, the
term at each cap and hashing price, the attack that outsources whole forwards, the coins' forming on the GPU, the
certifier's stream, T2 and the served noise for T3. Inputs: ``egress_inputs.json`` beside ``alt_roads_inputs.json``.
Writes ``egress.json`` and ``table.txt`` to ``--out``; exits non-zero if a self-check fails.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import alt_roads

HERE = Path(__file__).resolve().parent
POINTS = {"decode1": None, "t64": "t512", "t512": "t512r512", "t8192": "t8192"}
LEAD_T2 = "ssn-1l-tag/served/s8/k32/w/r256k/cert/min"


def steps(k: int, c: int) -> int:
    return 1 + math.ceil(max(0, k - (32 - c)) / 31)


def steps4(k: int, data_lanes: int) -> int:
    return 1 + math.ceil(max(0, k - 28) / data_lanes)


def row_bits(k: int, c: int) -> int:
    return 32 * math.ceil((steps(k, c) - 1 + c) / 32)


def odd_mean_square(e: int) -> float:
    """E[s^2] for s uniform over the odd values in +-(2^e - 1)."""
    return (4 ** e - 1) / 3.0


def call(design: dict, m: int, k: int, n: int, lead: dict, lay: dict, reg: float) -> dict:
    """One call's coins: coin bits, streamed bytes, forming in W1 units, words against the lead's, served noise rms."""
    c = lead["code_lanes"]
    s, rb = steps(k, c), row_bits(k, c)
    rows = {"bits": m * rb, "bytes": m * rb / 8.0, "forming": m * rb / 32.0 * reg}
    kind = design["kind"]
    if kind == "rows":
        return {**rows, "words": 1.0, "noise_rms": 0.0}
    if kind == "blc1":
        tile = design["tile"] or m
        coin_steps = math.ceil((s - 1) * min(1.0, m / tile))
        entries = math.ceil(m / tile) * n * coin_steps
        return {"bits": rows["bits"] + entries * design["e"], "bytes": rows["bytes"] + entries,
                "forming": rows["forming"] + entries / 4.0 * reg, "words": 1.0, "entries": entries,
                "noise_rms": math.sqrt(2.5 * coin_steps * odd_mean_square(design["e"]))}
    if kind == "blc4":
        l4, d4 = lay["lanes4"]["lanes"], lay["lanes4"]["data_lanes"]
        s4 = steps4(k, d4)
        entries = math.ceil(m / design["tile"]) * n * l4 * math.ceil((s4 - 1) * min(1.0, m / design["tile"]))
        return {"bits": rows["bits"] + entries * design["e"], "bytes": rows["bytes"] + entries,
                "forming": rows["forming"], "words": s4 / s, "entries": entries,
                "noise_rms": math.sqrt((2.5 + (l4 - 1)) * (s4 - 1) * odd_mean_square(design["e"]))}
    if kind == "cinit":
        e = math.ceil(k / lay["cinit_n_target"])
        return {"bits": rows["bits"] + m * n * e, "bytes": rows["bytes"] + m * n * math.ceil(e / 8),
                "forming": rows["forming"] + m * n * reg, "words": 1.0,
                "noise_rms": math.sqrt((4 ** e - 1) / 12.0)}
    if kind == "dither_a":
        return {"bits": m * k * design["bits_per_entry"], "bytes": m * k * design["bits_per_entry"] / 8.0,
                "forming": float("nan"), "words": design.get("words", 1.0), "noise_rms": float("nan")}
    raise ValueError(kind)


def designs(lay: dict) -> list[dict]:
    out = [{"id": "rows", "kind": "rows",
            "title": "the lead's row coins streamed raw (R1b): ssn-1l-tag's S - 1 + c bits a row in whole words"},
           {"id": "dither-a1", "kind": "dither_a", "bits_per_entry": 1,
            "title": "the locked Dither instance's layout: one coin bit an A entry (n_max = N), for reference"},
           {"id": "os2", "kind": "dither_a", "bits_per_entry": 3, "words": 2.0,
            "title": "SW-NOISE's os2: three coin bits an A entry, alpha 2, for reference"},
           {"id": "cinit", "kind": "cinit",
            "title": f"an accumulator-init coin an output element, ceil(K/{lay['cinit_n_target']}) bits, one init write"},
           {"id": "blc1-call-e1", "kind": "blc1", "tile": None, "e": 1,
            "title": "the B' lane's coins a (step, column) a call, shared by every row (ssn2-like two-sided)"}]
    for t in lay["tile_rows"]:
        for e in lay["lane_entropy_bits"]:
            out.append({"id": f"blc1-t{t}-e{e}", "kind": "blc1", "tile": t, "e": e,
                        "title": f"the lead's B' lane, a {e}-bit odd coin a ({t}-row tile, step, column), patched"})
    for e in (2, 7):
        out.append({"id": f"blc4-t64-e{e}", "kind": "blc4", "tile": 64, "e": e,
                    "title": f"four B' coin lanes a step (28 data lanes), {e}-bit odd coins a (64-row tile, step, "
                             "column), cp.async"})
    return out


def w_mac(base: dict, words: float, units_per_byte: float, answers: float) -> float:
    return base["alpha"] * words + base["hash_bytes_per_mac"] * words * units_per_byte + answers


def price_design(design: dict, inp: dict, einp: dict, ll: dict) -> dict:
    cfg = inp["llama_3_2_1b"]
    lead, lay, pr = einp["lead"], einp["layouts"], einp["prices"]
    caps = inp["egress"]["cap_bits_per_token"]
    sha = inp["sha512"]
    units = {"floor": sha["floor_units_per_byte"],
             "floor_chained": sha["floor_units_per_byte"] * pr["sha_registers_per_compression"]["chained"]
             / pr["sha_registers_per_compression"]["locked"],
             "bar": sha["bar_units_per_byte"]}
    base = {"alpha": inp["ssn_1l"]["alpha"], "hash_bytes_per_mac": inp["egress"]["credited_hash_bytes_per_mac"]}
    pts = {}
    for key, space_key in POINTS.items():
        m = inp["points"][key]["m"]
        calls = [(k, n, call(design, m, k, n, lead, lay, pr["reg_units"])) for k, n in cfg["linears"]]
        macs = cfg["layers"] * sum(m * k * n for k, n, _ in calls)
        bits = cfg["layers"] * sum(c["bits"] for _, _, c in calls)
        byts = cfg["layers"] * sum(c["bytes"] for _, _, c in calls)
        n_max = max(m * k * n / c["bits"] for k, n, c in calls)
        words = cfg["layers"] * sum(m * k * n * c["words"] for k, n, c in calls) / macs
        forming = cfg["layers"] * sum(c["forming"] for _, _, c in calls) / macs
        entries = cfg["layers"] * sum(c.get("entries", 0) for _, _, c in calls)
        wm = {p: w_mac(base, words, u, pr["answers_units_per_mac"]) for p, u in units.items()}
        row = {"m": m, "coin_bits_per_token": bits / m, "stream_bytes_per_token": byts / m, "n_max": n_max,
               "macs_per_coin_bit_mean": macs / bits, "words_vs_lead": words, "w_mac": wm,
               "forming_share_of_stock": forming,
               "noise_rms_lsb_by_k": {str(k): c["noise_rms"] for k, _, c in calls},
               "caps": {str(cap): {"stated": {p: n_max * w * cap / ll["macs_per_token"] for p, w in wm.items()},
                                   "honest_share": n_max * cap / ll["macs_per_token"],
                                   "attack_whole_forwards": {p: w * cap / (bits / m + 32) for p, w in wm.items()}}
                        for cap in caps}}
        if space_key:
            row["t2"] = t2_at(inp, einp, ll, space_key, byts, entries, words)
        pts[key] = row
    return {"id": design["id"], "title": design["title"], "points": pts}


def t2_at(inp: dict, einp: dict, ll: dict, space_key: str, step_bytes: float, step_entries: float,
          words: float) -> dict:
    """The coin stream beside N-CARD's honest step for the lead (hashing-bound, scaled by the words), the lane patch's
    byte stores, and R1b's arrival-order coins (alt_roads.r1b_arrival_t2)."""
    v = inp["space_t2"]["designs"][LEAD_T2][space_key]
    h = inp["h200"]
    patch_ms = step_entries / (h["sms"] * einp["prices"]["lane_store_bytes_per_sm_clk"] * h["fmax_hz"]) * 1e3
    honest_ms = v["t_h_ms"] * words + patch_ms
    arrival = alt_roads.r1b_arrival_t2(inp, ll)["designs"][LEAD_T2][space_key]["arrival_t2"]
    links = {"census_100g": inp["links"]["census_reference_bytes_per_s"],
             "gpu_400g": inp["links"]["per_gpu_400g_bytes_per_s"]}
    out = {"honest_ms": honest_ms, "patch_ms": patch_ms, "stock_ms": v["stock_ms"],
           "lead_judged": v["judged"], "stream_bytes_per_s": step_bytes / (honest_ms * 1e-3),
           "certifier_cores": [step_bytes / (honest_ms * 1e-3) / r
                               for r in einp["certifier"]["sha512_bytes_per_s_core"]][::-1]}
    for ln, rate in links.items():
        link_ms = step_bytes / rate * 1e3
        out[ln] = {"port_share": step_bytes / (honest_ms * 1e-3) / rate, "link_ms": link_ms,
                   "t2": max(honest_ms, link_ms) / v["stock_ms"],
                   "t2_arrival": [max(a * words + patch_ms / v["stock_ms"], link_ms / v["stock_ms"])
                                  for a in arrival[ln]]}
    return out


def lower_bound(inp: dict, ll: dict, res: dict) -> dict:
    """Any layout: the whole-forward attack is W_mac cap / (coin bits + 32), so 1% needs coin bits a token of at least
    100 W_mac cap - 32; the stated term needs n_max <= 0.01 MACs / (W_mac cap), and n_max >= MACs / coin bits.
    A B-shared lane: a coin entry is read by the tile's T rows over a step's d data lanes, so n_max >= T d / (L e)."""
    rows = res["designs"][0]["points"]["t8192"]
    out = {}
    for cap in inp["egress"]["cap_bits_per_token"]:
        out[str(cap)] = {}
        for p, w in rows["w_mac"].items():
            need = 0.01 * ll["macs_per_token"] / (w * cap)
            out[str(cap)][p] = {"n_max_at_most": need, "coin_bits_per_token_at_least": 100.0 * w * cap - 32,
                                "coin_bits_per_mac_at_least": (100.0 * w * cap - 32) / ll["macs_per_token"],
                                "blc_tile_e_lanes_that_meet_it": [
                                    {"tile": t, "lanes": lanes, "e": e, "n": t * d / (lanes * e)}
                                    for t in (16, 64, 128) for lanes, d in ((1, 31), (4, 28)) for e in (1, 2, 4, 7)
                                    if t * d / (lanes * e) <= need]}
    return out


def statement_check(inp: dict, ll: dict, res: dict) -> dict:
    """The term under the statements as locked: Accel.ext and CertifierSeesImports exclude only the coin stream, so
    R1b's A_j to the certifier (arrival order), or a 512-bit root(A_j) a call in its place, and the locked Dither
    layout's n_max = N, are counted as they read."""
    cfg = inp["llama_3_2_1b"]
    calls_per_forward = cfg["layers"] * len(cfg["linears"])
    out = {}
    for did in ("rows", res["closing"]["id"]):
        out[did] = {}
        for key, p in res["designs_by_id"][did]["points"].items():
            extra = {"a_j_to_the_certifier": 8.0 * ll["a_bytes_per_token"],
                     "root_a_j_a_call": 512.0 * calls_per_forward / p["m"]}
            out[did][key] = {name: {p_: p["n_max"] * w * (32 + bits) / ll["macs_per_token"]
                                    for p_, w in p["w_mac"].items()} | {"bits_per_token": bits}
                             for name, bits in extra.items()}
    return out


def per_period(inp: dict, shares: list[tuple[str, float]]) -> list[dict]:
    acc = inp["accounting"]
    periods = {"window": acc["window_s"], "week": acc["week_s"], "month": acc["month_s"], "year": acc["year_s"]}
    h200 = alt_roads.peak(inp) * acc["mfu_serve"]
    rates = {"one H200": h200, "datacenter": h200 * acc["dc_gpus"],
             "world": acc["world_flop_per_year"] / 2.0 / acc["year_s"]}
    return [{"bound": name, "share_of_stock": share,
             "and_gates": {sc: {p: share * r * s * acc["and_per_int8_mac"] for p, s in periods.items()}
                           for sc, r in rates.items()}} for name, share in shares]


def distinct_check(s_zero: bool, seed: int) -> bool:
    """ssn-1l-tag's words with blc1's lane as affine functions of a row's coin bits (step 1's four code bits against
    the column code, each later step's bit b_t against the lane's coin s_t[j]): pairwise distinct coefficient vectors
    over rows, columns and steps for any nonzero s. With s_zero one coin is 0, so that word's step adds no bit."""
    rng = random.Random(seed)
    m, n, steps_ = 3, 5, 4
    code = [[(j // 13 ** d) % 13 - 6 for d in range(4)] for j in range(1, n + 1)]
    s = [[rng.choice([v for v in range(-127, 128, 2)]) for _ in range(n)] for _ in range(steps_)]
    if s_zero:
        s[2][1] = 0
    seen = set()
    for i in range(m):
        for j in range(n):
            for t in range(steps_):
                coef = [0] * (m * (4 + steps_))
                for d in range(4):
                    coef[i * (4 + steps_) + d] = code[j][d]
                for u in range(1, t + 1):
                    coef[i * (4 + steps_) + 4 + u] = s[u][j]
                key = tuple(coef)
                if key in seen or not any(coef):
                    return False
                seen.add(key)
    return True


def noise_mc(e: int, s_count: int, trials: int, seed: int) -> float:
    rng = random.Random(seed)
    odd = list(range(-(2 ** e - 1), 2 ** e, 2))
    tot = 0.0
    for _ in range(trials):
        x = sum((1 + rng.randint(0, 1)) * rng.choice(odd) for _ in range(s_count))
        tot += x * x
    return math.sqrt(tot / trials)


def checks(inp: dict, einp: dict, ll: dict, res: dict) -> list[tuple[str, bool, str]]:
    d = {x["id"]: x for x in res["designs"]}
    ar = alt_roads.egress_outsourcing(inp, ll)
    rows8 = d["rows"]["points"]["t8192"]
    best = d[res["closing"]["id"]]["points"]
    blc = call({"kind": "blc1", "tile": 128, "e": 2}, 8192, 2048, 16384, einp["lead"], einp["layouts"], 32)
    brute = 8192 * 96 + sum(16384 * 66 * 2 for _ in range(0, 8192, 128))
    return [
        ("The lead's coins: 9,216 bits a token and n_max gate-up's 2,048 x 16,384 / 96, as alt_roads",
         rows8["coin_bits_per_token"] == ar["coin_bits_per_token"] and abs(rows8["n_max"] - ar["n_max"]) < 1e-6,
         f"{rows8['coin_bits_per_token']:.0f}, {rows8['n_max']:,.1f}"),
        ("The lead's stated term at a 32-bit cap reproduces alt_roads (137.6% floor, 267.8% bar)",
         abs(rows8["caps"]["32"]["stated"]["floor"] - ar["share_of_stock_at_cap"]["32"]["r1b_live_coins"]
             ["floor_units_per_byte"]) < 1e-9 and
         abs(rows8["caps"]["32"]["stated"]["bar"] - ar["share_of_stock_at_cap"]["32"]["r1b_live_coins"]
             ["bar_units_per_byte"]) < 1e-9,
         f"{rows8['caps']['32']['stated']['floor']:.4f}, {rows8['caps']['32']['stated']['bar']:.4f}"),
        ("blc1's coin bits at gate-up, 8,192 rows, 128-row tiles, 2 bits: the formula equals a tile-by-tile count",
         blc["bits"] == brute, f"{blc['bits']:,.0f}"),
        ("Every layout's n_max is at least its mean MACs a coin bit (the lower bound's step)",
         all(p["n_max"] >= p["macs_per_coin_bit_mean"] * (1 - 1e-12) for x in res["designs"]
             for p in x["points"].values()), "ok"),
        ("The closing layout's stated term is at most 1% of stock at a 32-bit cap, at both prices, at every point",
         all(p["caps"]["32"]["stated"][q] <= 0.01 for p in best.values() for q in ("floor", "bar")),
         ", ".join(f"{k} {p['caps']['32']['stated']['bar']:.4f}" for k, p in best.items())),
        ("The closing layout's stream is under 10% of a 400 Gb/s port at the honest pace at every timed point",
         all(p["t2"]["gpu_400g"]["port_share"] < 0.10 for p in best.values() if "t2" in p),
         ", ".join(f"{p['t2']['gpu_400g']['port_share']:.4f}" for p in best.values() if "t2" in p)),
        ("blc1's served noise rms matches a seeded Monte Carlo within 3% (e 2, 66 steps)",
         abs(noise_mc(2, 66, 20000, 7) / math.sqrt(2.5 * 66 * odd_mean_square(2)) - 1) < 0.03,
         f"{noise_mc(2, 66, 20000, 7):.2f} vs {math.sqrt(2.5 * 66 * odd_mean_square(2)):.2f}"),
        ("blc1 keeps the tag lead's words pairwise distinct as functions of the row's coin bits for nonzero lane coins",
         all(distinct_check(False, sd) for sd in range(20)), "20 seeds"),
        ("Mutation: a zero lane coin makes two words of one element coincide, and the check catches it",
         not any(distinct_check(True, sd) for sd in range(20)), "20 seeds"),
        ("Per period: N-CARD's stock work reproduces (one H200 window 3.09e19 AND gates)",
         abs(res["period"][0]["and_gates"]["one H200"]["window"] / res["period"][0]["share_of_stock"] / 3.086e19 - 1)
         < 1e-3, f"{res['period'][0]['and_gates']['one H200']['window'] / res['period'][0]['share_of_stock']:.4e}"),
    ]


def table(res: dict) -> str:
    lines = ["N-EGRESS: OutsourceWithinEgress's term n_max W_mac E / stock MACs on Llama-3.2-1B (T 8,192 unless stated)",
             "  design: coin bits a token, n_max, stream B a token; term at cap 32 / 320, floor / floor chained / bar; "
             "whole-forward attack at 32 (floor); forming (T1 uncredited); words"]
    for x in res["designs"]:
        p = x["points"]["t8192"]
        c32, c320 = p["caps"]["32"], p["caps"]["320"]
        f = lambda c: " / ".join(f"{c['stated'][q]:.4%}" for q in ("floor", "floor_chained", "bar"))
        lines.append(f"  {x['id']:<16} {p['coin_bits_per_token']:>12,.0f} {p['n_max']:>12,.1f} "
                     f"{p['stream_bytes_per_token']:>11,.0f}  32: {f(c32)}  320: {f(c320)}  attack "
                     f"{c32['attack_whole_forwards']['floor']:.4%}  forming {p['forming_share_of_stock']:.4%}  "
                     f"words x{p['words_vs_lead']:.3f}")
    cl = res["closing"]
    lines.append(f"Closing layout: {cl['id']} ({cl['title']})")
    for k, p in res["designs_by_id"][cl["id"]]["points"].items():
        t2 = p.get("t2")
        s = (f"  {k:<8} m {p['m']:>5}: n_max {p['n_max']:,.1f}, term cap 32 {p['caps']['32']['stated']['floor']:.4%}"
             f"-{p['caps']['32']['stated']['bar']:.4%}, cap 320 {p['caps']['320']['stated']['floor']:.4%}-"
             f"{p['caps']['320']['stated']['bar']:.4%}; forming {p['forming_share_of_stock']:.4%}; noise rms "
             + ", ".join(f"K {k_}: {v:.1f}" for k_, v in p["noise_rms_lsb_by_k"].items()))
        if t2:
            s += (f"; T2 {t2['gpu_400g']['t2']:.2f} (lead {t2['lead_judged']:.2f}), arrival 400G "
                  f"{t2['gpu_400g']['t2_arrival'][0]:.2f}-{t2['gpu_400g']['t2_arrival'][1]:.2f}, 100G "
                  f"{t2['census_100g']['t2_arrival'][0]:.2f}-{t2['census_100g']['t2_arrival'][1]:.2f}; stream "
                  f"{t2['stream_bytes_per_s'] / 1e9:.2f} GB/s ({t2['gpu_400g']['port_share']:.2%} of 400G, "
                  f"{t2['census_100g']['port_share']:.2%} of 100G), certifier {t2['certifier_cores'][0]:.2f}-"
                  f"{t2['certifier_cores'][1]:.2f} Genoa cores, patch {t2['patch_ms']:.3f} ms")
        lines.append(s)
    lines.append("Lower bound (any layout): coin bits a token >= 100 W_mac cap - 32; n_max <= 0.01 MACs / (W_mac cap)")
    for cap, ps in res["lower_bound"].items():
        for p, v in ps.items():
            ok = ", ".join(f"T{r['tile']}/L{r['lanes']}/e{r['e']} ({r['n']:.0f})"
                           for r in v["blc_tile_e_lanes_that_meet_it"]) or "none"
            lines.append(f"  cap {cap} {p}: n_max <= {v['n_max_at_most']:,.1f}, coin bits a token >= "
                         f"{v['coin_bits_per_token_at_least']:,.0f} ({v['coin_bits_per_mac_at_least']:.2e} a MAC); "
                         f"B-shared lanes that meet it: {ok}")
    lines.append("Under the statements as locked (only the coin stream excluded from E), cap 32, floor / bar:")
    for did, pts in res["statement_check"].items():
        for key, v in pts.items():
            lines.append(f"  {did} {key}: " + "; ".join(
                f"{name} ({x['bits_per_token']:,.0f} bits a token) {x['floor']:.4%} / {x['bar']:.4%}"
                for name, x in v.items()))
    lines.append("Per period (N-CARD's basis, stock work held fixed), AND gates window / week / month / year:")
    for r in res["period"]:
        cells = "; ".join(f"{sc} " + " / ".join(f"{v:.2e}" for v in ps.values()) for sc, ps in r["and_gates"].items())
        lines.append(f"  {r['bound']} ({r['share_of_stock']:.4%}): {cells}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", type=Path, default=HERE / "alt_roads_inputs.json")
    ap.add_argument("--egress-inputs", type=Path, default=HERE / "egress_inputs.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    inp, einp = alt_roads.load(args.inputs), alt_roads.load(args.egress_inputs)
    ll = alt_roads.llama(inp)
    res = {"designs": [price_design(x, inp, einp, ll) for x in designs(einp["layouts"])]}
    res["designs_by_id"] = {x["id"]: x for x in res["designs"]}
    closing = [x for x in res["designs"] if x["id"].startswith("blc1-t")
               and all(p["caps"]["32"]["stated"]["bar"] <= 0.01 for p in x["points"].values())]
    best = min(closing, key=lambda x: (x["points"]["t8192"]["stream_bytes_per_token"],
                                       x["points"]["t8192"]["noise_rms_lsb_by_k"]["8192"]))
    res["closing"] = {"id": best["id"], "title": next(d["title"] for d in designs(einp["layouts"])
                                                      if d["id"] == best["id"])}
    res["lower_bound"] = lower_bound(inp, ll, res)
    res["statement_check"] = statement_check(inp, ll, res)
    p8 = best["points"]["t8192"]["caps"]
    lead8 = res["designs_by_id"]["rows"]["points"]["t8192"]["caps"]
    res["period"] = per_period(inp, [
        ("stock work", 1.0),
        ("the lead's network term, cap 32, floor", lead8["32"]["stated"]["floor"]),
        ("the lead's network term, cap 32, bar", lead8["32"]["stated"]["bar"]),
        (f"{best['id']}'s network term, cap 32, floor", p8["32"]["stated"]["floor"]),
        (f"{best['id']}'s network term, cap 32, bar", p8["32"]["stated"]["bar"]),
        (f"{best['id']}'s lane forming, uncredited", best["points"]["t8192"]["forming_share_of_stock"]),
        ("1% allowance", 0.01)])
    cks = checks(inp, einp, ll, res)
    res["checks"] = [{"name": n, "ok": ok, "value": v} for n, ok, v in cks]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "egress.json").write_text(json.dumps(res, indent=2, sort_keys=True))
    text = table(res)
    (args.out / "table.txt").write_text(text + "\n")
    print(text)
    for n, ok, v in cks:
        print(f"{'ok  ' if ok else 'FAIL'} {n} {v}")
    bad = sum(1 for _, ok, _ in cks if not ok)
    print(f"{len(cks) - bad}/{len(cks)} checks")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
