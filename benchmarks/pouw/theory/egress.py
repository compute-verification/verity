"""N-EGRESS: coin layouts whose network term is at most 1% of stock, priced on N-CARD's basis.

DitherInstance's guarantee (Security/Definitions/Pouw/Guarantees/DitherInstance.lean) charges n_max W_mac E in its
slack: E the bits the network certifier counts (capped at served traffic, the coin stream excluded), W_mac the honest
block's priced cost a MAC (its hashing and answers included) and n_max the most MACs one exchanged bit carries, through
OutsourceWithinEgress at beta = n_max W_mac. As a share of stock (W1: a MAC 1, a register write 32) the term is
n_max W_mac E / MACs. Each layout is priced on Llama-3.2-1B's linears at each point's rows: its coin bits, n_max, the
term at each cap and hashing price, the outsourcing provers that realize it (a call set when the served path is
coin-free, a prefix of the forward when it is noisy), the coins' forming on the GPU, the certifier's stream, T2 and the
served noise for T3, with whether N-EXACT's epilogue fold serves it exactly, and N-RED's A2c / A8 (one chosen data lane
cancelling a step's likeliest coin increment) under model P at each point. Inputs: ``egress_inputs.json`` beside
``alt_roads_inputs.json``.
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


def uniform_mean_square(bits: int) -> float:
    """E[r^2] for r uniform over the signed bits-bit integers."""
    h = 2 ** (bits - 1)
    return sum(v * v for v in range(-h, h)) / (2 * h)


def call(design: dict, m: int, k: int, n: int, lead: dict, lay: dict, reg: float) -> dict:
    """One call's coins: coin bits, streamed bytes, forming in W1 units, words against the lead's, served noise rms."""
    c = lead["code_lanes"]
    s, rb = steps(k, c), row_bits(k, c)
    rows = {"bits": m * rb, "bytes": m * rb / 8.0, "forming": m * rb / 32.0 * reg}
    kind = design["kind"]
    if kind == "rows":
        return {**rows, "words": 1.0, "noise_rms": 0.0, "later": s - 1, "coin_steps": s - 1}
    if kind == "blc1":
        tile = design["tile"] or m
        coin_steps = math.ceil((s - 1) * min(1.0, m / tile))
        entries = math.ceil(m / tile) * n * coin_steps
        return {"bits": rows["bits"] + entries * design["e"], "bytes": rows["bytes"] + entries,
                "forming": rows["forming"] + entries / 4.0 * reg, "words": 1.0, "entries": entries,
                "later": s - 1, "coin_steps": coin_steps,
                "noise_rms": math.sqrt(2.5 * coin_steps * odd_mean_square(design["e"]))}
    if kind in ("blc1a", "blc4a"):
        la, dl = (1, lay["lane_add"]["data_lanes"]) if kind == "blc1a" else \
            (lay["lanes4"]["lanes"] - 1, lay["lanes4"]["data_lanes"])
        sa = steps4(k, dl)
        tiles = math.ceil(m / design["tile"])
        coin_steps = math.ceil((sa - 1) * min(1.0, m / design["tile"]))
        fill, ef = sa - 1 - coin_steps, lay["lane_add"]["fill_row_coin_bits"]
        rba = 32 * math.ceil((coin_steps + fill * max(ef, 1) + c) / 32)
        cells = tiles * n * coin_steps
        corr = 4.0 * tiles * n if coin_steps > 1 else 0.0
        lane_bytes = cells if kind == "blc1a" else cells * lay["lanes4"]["lanes"]
        return {"bits": m * rba + cells * la * design["e"], "bytes": m * rba / 8.0 + lane_bytes + corr,
                "forming": m * rba / 32.0 * reg + (cells / 4.0 * reg if kind == "blc1a" else 0.0),
                "words": sa / s, "entries": lane_bytes, "later": sa - 1, "coin_steps": coin_steps, "fill": fill,
                "noise_rms": 0.0, "word_noise_rms": math.sqrt(la * coin_steps * odd_mean_square(design["e"])
                                                               + (fill * uniform_mean_square(ef) if ef else 0.0))}
    if kind == "blc4":
        l4, d4 = lay["lanes4"]["lanes"], lay["lanes4"]["data_lanes"]
        s4 = steps4(k, d4)
        coin_steps = math.ceil((s4 - 1) * min(1.0, m / design["tile"]))
        entries = math.ceil(m / design["tile"]) * n * l4 * coin_steps
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
    for t in lay["tile_rows"]:
        for e in lay["lane_entropy_bits"][1:]:
            out.append({"id": f"blc1a-t{t}-e{e}", "kind": "blc1a", "tile": t, "e": e,
                        "title": f"a coin lane against A' = 1 beside the lead's row lane (30 data lanes), a {e}-bit "
                                 f"odd coin a ({t}-row tile, step, column), patched, served exactly"})
    for e in (2, 7):
        out.append({"id": f"blc4-t64-e{e}", "kind": "blc4", "tile": 64, "e": e,
                    "title": f"four B' coin lanes a step (28 data lanes), {e}-bit odd coins a (64-row tile, step, "
                             "column), cp.async"})
    for t, e in ((64, 4), (64, 7), (128, 7)):
        out.append({"id": f"blc4a-t{t}-e{e}", "kind": "blc4a", "tile": t, "e": e,
                    "title": f"three coin lanes against A' = 1 beside the lead's row lane (28 data lanes), {e}-bit odd "
                             f"coins a ({t}-row tile, step, column), one 4-byte cp.async with the public weight, served "
                             "exactly"})
    return out


SERVED = {"rows": "either", "blc1a": "exact", "blc4a": "exact"}


def lane_sum(lanes: int, e: int) -> dict[int, float]:
    odd = range(-(2 ** e - 1), 2 ** e, 2)
    dist = {0: 1.0}
    for _ in range(lanes):
        nxt: dict[int, float] = {}
        for v, p in dist.items():
            for s in odd:
                nxt[v + s] = nxt.get(v + s, 0.0) + p / len(odd)
        dist = nxt
    return dist


def step_dists(design: dict, lay: dict, w: int) -> dict | None:
    """N-RED's A2c / A8 against a layout: the adversary fixes, before the coins, one data lane a later step (A = 1,
    B = -v*) so that a word repeats its previous step's whenever the step's coin increment is v*, the likeliest value
    of the step's type. The increments' laws: "coin", a step carrying the layout's lane coins (uniform odd in
    +-(2^e - 1)) beside the row lane (1 + b) w, b a uniform row bit and w public; "fill", a step of a tile with fewer
    rows than T that carries none (the lead's lane, or blc1a / blc4a's uniform row coin r_t(i) w)."""
    kind, e = design["kind"], design.get("e", 1)
    if kind not in ("rows", "blc1", "blc1a", "blc4a"):
        return None
    lead = {w: 0.5, 2 * w: 0.5}
    lanes = {"rows": 0, "blc1": 0, "blc1a": 1, "blc4a": lay["lanes4"]["lanes"] - 1}[kind]
    coin: dict[int, float] = {}
    for b in (0, 1):
        for v, p in lane_sum(lanes, e).items():
            for s, q in (lane_sum(1, e) if kind == "blc1" else {w: 1.0}).items():
                coin[(1 + b) * s + v] = coin.get((1 + b) * s + v, 0.0) + 0.5 * p * q
    ef = lay["lane_add"]["fill_row_coin_bits"]
    if kind in ("blc1a", "blc4a") and ef:
        h = 2 ** (ef - 1)
        fill = {r * w: 1.0 / (2 * h) for r in range(-h, h)}
    else:
        fill = lead
    return {"coin": coin, "fill": fill}


def coin_plan(later: int, coin_steps: int) -> list[bool]:
    """Which later steps carry coins when a tile carries them in coin_steps of its later steps, spread evenly."""
    return [(u + 1) * coin_steps // later > u * coin_steps // later for u in range(later)]


def a2c_point(design: dict, lay: dict, w: int, calls: list, m: int, reg: float) -> dict | None:
    """A8 under model P at a point, N-RED's count (nred_round3.at_scale) in expectation: each kept entry-step saves a
    word at reg units, one pointer a row-step is charged anyway, over the stock MACs m k n. Per step type the adversary
    uses the cancelling lane only where it gains."""
    dists = step_dists(design, lay, w)
    if dists is None:
        return None
    ps = {t: max(d.values()) for t, d in dists.items()}
    saved = macs = 0.0
    seen = set()
    for k, n, c in calls:
        counts = {"coin": c["coin_steps"], "fill": c["later"] - c["coin_steps"]}
        for t, cnt in counts.items():
            saved += max(0.0, reg * m * cnt * (n * ps[t] - 1.0))
            if cnt:
                seen.add(t)
        macs += m * k * n
    return {"pstar": max(ps[t] for t in seen), "pstar_by_type": ps, "types": sorted(seen),
            "model_p_share": saved / macs}


def a8_words(design: dict, lay: dict, w: int, m: int, n: int, later: int, seed: int, plan_shift: int = 0) -> dict:
    """A8 under model P word for word on a small instance (nred_round3.build_entries' adversary on this layout): chosen
    data, zero past step 1 except each later step's first lane with A = 1 and B = -v* of the step's type; honest
    words are the k32 partial sums of A B plus the step's coins; the adversary keeps an entry's previous word when the
    coin increment is v* and otherwise adds its data lane and the coins. plan_shift > 0 mutates the adversary's coin
    plan. Returns word equality, the kept entry-steps, their exact mean and standard deviation."""
    rng = random.Random(seed)
    kind, tile, e = design["kind"], design["tile"], design["e"]
    la, dl = (1, lay["lane_add"]["data_lanes"]) if kind == "blc1a" else \
        (lay["lanes4"]["lanes"] - 1, lay["lanes4"]["data_lanes"])
    d1, ef = 28, lay["lane_add"]["fill_row_coin_bits"]
    plan = coin_plan(later, math.ceil(later * min(1.0, m / tile)))
    dists = step_dists(design, lay, w)
    vstar = {t: max(d, key=d.get) for t, d in dists.items()}
    k = d1 + dl * later
    a = [[rng.randint(-60, 60) if x < d1 else 0 for x in range(k)] for _ in range(m)]
    bcols = [[rng.randint(-60, 60) if x < d1 else 0 for x in range(k)] for _ in range(n)]
    for u in range(later):
        lane = d1 + dl * u
        for i in range(m):
            a[i][lane] = 1
        for j in range(n):
            bcols[j][lane] = -vstar["coin" if plan[u] else "fill"]
    odd = list(range(-(2 ** e - 1), 2 ** e, 2))
    h = 2 ** (max(ef, 1) - 1)
    s = [[[sum(rng.choice(odd) for _ in range(la)) for _ in range(n)] for _ in range(later)]
         for _ in range(math.ceil(m / tile))]
    bt = [[rng.randint(0, 1) for _ in range(later)] for _ in range(m)]
    r = [[rng.randrange(-h, h) for _ in range(later)] for _ in range(m)]

    def coin(i: int, j: int, u: int, pl: list[bool]) -> int:
        if pl[u]:
            return (1 + bt[i][u]) * w + s[i // tile][u][j]
        return r[i][u] * w if ef else (1 + bt[i][u]) * w

    adv_plan = plan[plan_shift:] + plan[:plan_shift]
    equal, kept = True, 0
    for i in range(m):
        for j in range(n):
            hw = aw = sum(x * y for x, y in zip(a[i][:d1], bcols[j][:d1]))
            for u in range(later):
                lo = d1 + dl * u
                hw += sum(x * y for x, y in zip(a[i][lo:lo + dl], bcols[j][lo:lo + dl])) + coin(i, j, u, plan)
                c_adv = coin(i, j, u, adv_plan)
                v_adv = vstar["coin" if adv_plan[u] else "fill"]
                if c_adv == v_adv:
                    kept += 1
                else:
                    aw += a[i][lo] * bcols[j][lo] + c_adv
                equal &= aw == hw
    rows_tile = min(m, tile)
    mean = var = 0.0
    for u in range(later):
        if plan[u]:
            x_law = {(1 + b) * w: 0.5 for b in (0, 1)}
            q = {sv: sum(p for x, p in x_law.items() if x + sv == vstar["coin"]) for sv in lane_sum(la, e)}
            ps_ = lane_sum(la, e)
            eq = sum(ps_[sv] * q[sv] for sv in ps_)
            eq2 = sum(ps_[sv] * q[sv] ** 2 for sv in ps_)
            cells = math.ceil(m / tile) * n
            mean += cells * rows_tile * eq
            var += cells * (rows_tile * eq + rows_tile * (rows_tile - 1) * eq2 - (rows_tile * eq) ** 2)
        else:
            p = dists["fill"][vstar["fill"]]
            mean += m * n * p
            var += m * n * n * p * (1 - p)
    return {"equal": equal, "kept": kept, "mean": mean, "sd": math.sqrt(var), "entry_steps": m * n * later,
            "coin_steps": sum(plan), "later": later}


def outsource(calls: list, m: int, layers: int, served: str, wfun, alpha: float, out: dict) -> dict:
    """Priced work an outsourcing prover saves a bit it exchanges, under R1b (it keeps A_j for the certifier, so it
    computes the served path; the outside machine computes the coin-dependent words and hashes them, returning a root a
    call). Coin-free served path: any set of call types, every layer, beside the token (saving W - 1 a MAC). Noisy
    served path: a prefix of the forward, since each call's input carries every earlier call's coins (W - alpha words
    a MAC: the prover still runs the noisy MMA)."""
    tok, root = out["token_bits"], out["root_bits"]
    res = {}
    if served in ("exact", "either"):
        per = [(k * n * (wfun(c["words"]) - 1.0), c["bits"] / m + root / m) for k, n, c in calls]
        res["subset"] = max(layers * sum(per[i][0] for i in range(len(per)) if mask >> i & 1)
                            / (layers * sum(per[i][1] for i in range(len(per)) if mask >> i & 1) + tok)
                            for mask in range(1, 2 ** len(per)))
    if served in ("noisy", "either"):
        sv = bt = best = 0.0
        for _ in range(layers):
            for k, n, c in calls:
                sv += k * n * (wfun(c["words"]) - alpha * c["words"])
                bt += c["bits"] / m + root / m
                best = max(best, sv / (bt + tok))
        res["prefix"] = best
    return res


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
        served = SERVED.get(design["kind"], "noisy")
        osr = einp["outsource"]
        per_bit = {p: outsource(calls, m, cfg["layers"], served,
                                lambda wd, u=u: w_mac(base, wd, u, pr["answers_units_per_mac"]), base["alpha"], osr)
                   for p, u in units.items()}
        row = {"m": m, "coin_bits_per_token": bits / m, "stream_bytes_per_token": byts / m, "n_max": n_max,
               "macs_per_coin_bit_mean": macs / bits, "words_vs_lead": words, "w_mac": wm,
               "forming_share_of_stock": forming, "served": served,
               "noise_rms_lsb_by_k": {str(k): c["noise_rms"] for k, _, c in calls},
               "word_noise_rms_lsb_by_k": {str(k): c.get("word_noise_rms", c["noise_rms"]) for k, _, c in calls},
               "outsource_units_per_bit": per_bit,
               "caps": {str(cap): {"stated": {p: n_max * w * cap / ll["macs_per_token"] for p, w in wm.items()},
                                   "honest_share": n_max * cap / ll["macs_per_token"],
                                   "attack": {p: {a: {"whole_cap": r * cap / ll["macs_per_token"],
                                                      "slack": r * (cap - osr["honest_io_bits_per_token"])
                                                      / ll["macs_per_token"],
                                                      "of_stated": r / (n_max * wm[p])}
                                                  for a, r in per_bit[p].items()} for p in wm}}
                        for cap in caps}}
        if space_key:
            row["t2"] = t2_at(inp, einp, ll, space_key, byts, entries, words)
        if key in einp["nkernel_t2"]:
            row["t2_nkernel_scaled"] = [v * words for v in einp["nkernel_t2"][key]]
        row["a2c"] = a2c_point(design, lay, einp["nred"]["public_lane_weight"], calls, m, pr["reg_units"])
        pts[key] = row
    a2c = None
    if all(p["a2c"] for p in pts.values()):
        worst = max(pts, key=lambda q: pts[q]["a2c"]["model_p_share"])
        a2c = {"pstar": max(p["a2c"]["pstar"] for p in pts.values()),
               "model_p_share": pts[worst]["a2c"]["model_p_share"], "worst_point": worst}
    return {"id": design["id"], "title": design["title"], "points": pts, "a2c": a2c}


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
    """Any layout: the stated term needs n_max <= 0.01 MACs / (W_mac cap), and n_max >= MACs / coin bits, so coin bits
    a token of at least 100 W_mac cap. A B-shared lane: a coin entry is read by the tile's T rows over a step's d data
    lanes, so n_max >= T d / (L e)."""
    rows = res["designs"][0]["points"]["t8192"]
    out = {}
    for cap in inp["egress"]["cap_bits_per_token"]:
        out[str(cap)] = {}
        for p, w in rows["w_mac"].items():
            need = 0.01 * ll["macs_per_token"] / (w * cap)
            out[str(cap)][p] = {"n_max_at_most": need, "coin_bits_per_token_at_least": 100.0 * w * cap,
                                "coin_bits_per_mac_at_least": 100.0 * w * cap / ll["macs_per_token"],
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


def fold_check(kind: str, seed: int) -> bool:
    """A tile's final accumulators with the lead's row lane (1 + b_t(i)) against public w_t and blc1's coin in that lane,
    or blc1a's coin on its own lane against A' = 1 on alternate steps and its 8-bit row coin r_t(i) in the row lane on
    the others (a tile with r < T rows): True when the noise is r_i + c_j and subtracting r_i = sum_t a_t(i) w_t and
    c_j = sum_t s_t[j] (N-EXACT's IADD3 sources) leaves A B^T exactly. A noise whose second differences
    N[i,j] - N[i,0] - N[0,j] + N[0,0] aren't all 0 is no r_i + c_j, so no such fold removes it."""
    rng = random.Random(seed)
    m, n, k, st = 6, 5, 40, 4
    a = [[rng.randint(-127, 127) for _ in range(k)] for _ in range(m)]
    b = [[rng.randint(-127, 127) for _ in range(n)] for _ in range(k)]
    bits = [[rng.randint(0, 1) for _ in range(st)] for _ in range(m)]
    rowc = [[rng.randrange(-128, 128) for _ in range(st)] for _ in range(m)]
    w = [rng.randint(-127, 127) for _ in range(st)]
    s = [[rng.choice([v for v in range(-15, 16, 2)]) for _ in range(n)] for _ in range(st)]
    coined = [t % 2 == 0 or kind == "blc1" for t in range(st)]
    exact = [[sum(a[i][x] * b[x][j] for x in range(k)) for j in range(n)] for i in range(m)]
    def a31(i: int, t: int) -> int:
        return 1 + bits[i][t] if coined[t] else rowc[i][t]

    def lane(i: int, j: int, t: int) -> int:
        if kind == "blc1":
            return (1 + bits[i][t]) * s[t][j]
        return a31(i, t) * w[t] + (s[t][j] if coined[t] else 0)
    acc = [[exact[i][j] + sum(lane(i, j, t) for t in range(st)) for j in range(n)] for i in range(m)]
    noise = [[acc[i][j] - exact[i][j] for j in range(n)] for i in range(m)]
    if any(noise[i][j] - noise[i][0] - noise[0][j] + noise[0][0] for i in range(m) for j in range(n)):
        return False
    r = [sum(a31(i, t) * w[t] for t in range(st)) for i in range(m)]
    c = [sum(s[t][j] for t in range(st) if coined[t]) for j in range(n)]
    return all(acc[i][j] - r[i] - c[j] == exact[i][j] for i in range(m) for j in range(n))


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
    lead_a2c, nred_pt = d["rows"]["points"]["t8192"]["a2c"], einp["nred"]["model_p_saving_at_pstar"][0]
    lay, w, reg = einp["layouts"], einp["nred"]["public_lane_weight"], einp["prices"]["reg_units"]
    by_id = {x["id"]: x for x in designs(lay)}
    lay0 = json.loads(json.dumps(lay))
    lay0["lane_add"]["fill_row_coin_bits"] = 0
    old = {}
    for key in ("t64", "decode1"):
        m = inp["points"][key]["m"]
        calls = [(k, n, call(by_id["blc1a-t128-e7"], m, k, n, einp["lead"], lay0, reg))
                 for k, n in inp["llama_3_2_1b"]["linears"]]
        old[key] = a2c_point(by_id["blc1a-t128-e7"], lay0, w, calls, m, reg)["model_p_share"]
    a8 = {}
    for c in ("closing", "closing_320"):
        des = by_id[res[c]["id"]]
        t = des["tile"]
        a8[f"{des['id']} m {2 * t}"] = a8_words(des, lay, w, 2 * t, 128, 11, 71)
        a8[f"{des['id']} m {t // 2}"] = a8_words(des, lay, w, t // 2, 32, 128, 73)
    half = by_id[res["closing"]["id"]]
    a8_bad = a8_words(half, lay, w, half["tile"] // 2, 32, 128, 73, plan_shift=1)
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
        ("blc1a's served noise, its coin steps and a small tile's row-coined steps, is r_i + c_j and N-EXACT's two fold "
         "sources return A B^T exactly",
         all(fold_check("blc1a", sd) for sd in range(20)), "20 seeds"),
        ("Mutation: blc1's coin in the row lane isn't r_i + c_j, so no free fold removes it, and the check says so",
         not any(fold_check("blc1", sd) for sd in range(20)), "20 seeds"),
        ("The de-noised lead's call-set outsourcer saves at least 95% a bit of what the stated term charges at T 8,192",
         rows8["caps"]["32"]["attack"]["floor"]["subset"]["of_stated"] >= 0.95,
         f"{rows8['caps']['32']['attack']['floor']['subset']['of_stated']:.4f}"),
        ("No outsourcer saves more a bit than the stated term charges, at every design, point, price and cap",
         all(v["of_stated"] <= 1.0 + 1e-12 for x in res["designs"] for p in x["points"].values()
             for c in p["caps"].values() for a in c["attack"].values() for v in a.values()), "ok"),
        ("The closing layout serves exactly (its served noise folds)",
         all(p["served"] == "exact" for p in best.values()), res["closing"]["id"]),
        ("The lead's A2c by N-RED's count: p* = 1/2 and model P 51.6% at T 8,192, as N-RED measured",
         lead_a2c["pstar"] == nred_pt[0] and abs(lead_a2c["model_p_share"] - nred_pt[1]) < 0.001,
         f"{lead_a2c['pstar']}, {lead_a2c['model_p_share']:.4f}"),
        ("Both closing layouts' A2c / A8 (model P, N-RED's count) is at most 1% of stock at every point",
         all(p["a2c"]["model_p_share"] <= 0.01 for c in ("closing", "closing_320")
             for p in d[res[c]["id"]]["points"].values()),
         ", ".join(f"{res[c]['id']} " + "/".join(f"{p['a2c']['model_p_share']:.4f}"
                                                 for p in d[res[c]["id"]]["points"].values())
                   for c in ("closing", "closing_320"))),
        ("Interim 2's blc1a-t128-e7 (the lead's public lane on a small tile's coin-less steps) leaves A2c above 25% "
         "at T 512 with 64 rows and above 50% at decode",
         old["t64"] > 0.25 and old["decode1"] > 0.5, f"{old['t64']:.4f}, {old['decode1']:.4f}"),
        ("A8 word for word on both closing layouts, two tiles of T rows and one of T/2 (half its steps row-coined): "
         "the adversary's words equal the honest words, its kept entry-steps within 4 sd of the expectation",
         all(v["equal"] and abs(v["kept"] - v["mean"]) <= 4 * v["sd"] for v in a8.values()),
         "; ".join(f"{name} kept {v['kept']} (mean {v['mean']:.0f}, sd {v['sd']:.0f}) of {v['entry_steps']:,}"
                   for name, v in a8.items())),
        ("Mutation: an A8 adversary whose coin plan is one step off on the half tile no longer matches the honest words",
         not a8_bad["equal"], f"kept {a8_bad['kept']}"),
        ("The cap-320 closing layout serves exactly, its stated term at most 1% at cap 320 at both prices, every point",
         all(p["served"] == "exact" and p["caps"]["320"]["stated"][q] <= 0.01
             for p in d[res["closing_320"]["id"]]["points"].values() for q in ("floor", "bar")),
         ", ".join(f"{k} {p['caps']['320']['stated']['bar']:.4f}"
                   for k, p in d[res["closing_320"]["id"]]["points"].items())),
        ("Per period: N-CARD's stock work reproduces (one H200 window 3.09e19 AND gates)",
         abs(res["period"][0]["and_gates"]["one H200"]["window"] / res["period"][0]["share_of_stock"] / 3.086e19 - 1)
         < 1e-3, f"{res['period'][0]['and_gates']['one H200']['window'] / res['period'][0]['share_of_stock']:.4e}"),
    ]


def table(res: dict) -> str:
    lines = ["N-EGRESS: OutsourceWithinEgress's term n_max W_mac E / stock MACs on Llama-3.2-1B (T 8,192 unless stated)",
             "  design: coin bits a token, n_max, stream B a token; term at cap 32 / 320, floor / floor chained / bar; "
             "outsourcers at 32 (floor; call set if the served path is coin-free, else a prefix): saving at the whole cap"
             " / at its slack over 17 bits, and a bit against the stated term; forming (T1 uncredited); words; served"]
    for x in res["designs"]:
        p = x["points"]["t8192"]
        c32, c320 = p["caps"]["32"], p["caps"]["320"]
        def f(c: dict) -> str:
            return " / ".join(f"{c['stated'][q]:.4%}" for q in ("floor", "floor_chained", "bar"))
        att = "; ".join(f"{a} {v['whole_cap']:.4%} / {v['slack']:.4%} ({v['of_stated']:.3f})"
                        for a, v in c32["attack"]["floor"].items())
        a2c = ("A2c p* " + " / ".join(f"{q['a2c']['pstar']:.5f}" for q in x["points"].values()) + " model P "
               + " / ".join(f"{q['a2c']['model_p_share']:.3%}" for q in x["points"].values()) + " (" +
               " / ".join(x["points"]) + ")" if x["a2c"] else "A2c n/a")
        lines.append(f"  {x['id']:<16} {p['coin_bits_per_token']:>12,.0f} {p['n_max']:>12,.1f} "
                     f"{p['stream_bytes_per_token']:>11,.0f}  32: {f(c32)}  320: {f(c320)}  outsourcers {att}  "
                     f"forming {p['forming_share_of_stock']:.4%}  words x{p['words_vs_lead']:.3f}  {p['served']}  "
                     f"{a2c}")
    for label, cl in (("Closing layout", res["closing"]), ("Closing layout at cap 320", res["closing_320"])):
        lines += closing_lines(res, label, cl)
    lines.append("Lower bound (any layout): coin bits a token >= 100 W_mac cap; n_max <= 0.01 MACs / (W_mac cap)")
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


def closing_lines(res: dict, label: str, cl: dict) -> list[str]:
    lines = [f"{label}: {cl['id']} ({cl['title']})"]
    for k, p in res["designs_by_id"][cl["id"]]["points"].items():
        t2 = p.get("t2")
        s = (f"  {k:<8} m {p['m']:>5}: n_max {p['n_max']:,.1f}, term cap 32 {p['caps']['32']['stated']['floor']:.4%}"
             f"-{p['caps']['32']['stated']['bar']:.4%}, cap 320 {p['caps']['320']['stated']['floor']:.4%}-"
             f"{p['caps']['320']['stated']['bar']:.4%}; forming {p['forming_share_of_stock']:.4%}; words "
             f"x{p['words_vs_lead']:.3f}; served noise rms "
             + ", ".join(f"K {k_}: {v:.1f}" for k_, v in p["noise_rms_lsb_by_k"].items())
             + "; committed words' coin rms "
             + ", ".join(f"K {k_}: {v:.1f}" for k_, v in p["word_noise_rms_lsb_by_k"].items())
             + f"; A2c model P {p['a2c']['model_p_share']:.3%} (p* {p['a2c']['pstar']:.5f})")
        if t2:
            s += (f"; T2 {t2['gpu_400g']['t2']:.2f} (lead {t2['lead_judged']:.2f}), arrival 400G "
                  f"{t2['gpu_400g']['t2_arrival'][0]:.2f}-{t2['gpu_400g']['t2_arrival'][1]:.2f}, 100G "
                  f"{t2['census_100g']['t2_arrival'][0]:.2f}-{t2['census_100g']['t2_arrival'][1]:.2f}; stream "
                  f"{t2['stream_bytes_per_s'] / 1e9:.2f} GB/s ({t2['gpu_400g']['port_share']:.2%} of 400G, "
                  f"{t2['census_100g']['port_share']:.2%} of 100G), certifier {t2['certifier_cores'][0]:.2f}-"
                  f"{t2['certifier_cores'][1]:.2f} Genoa cores, patch {t2['patch_ms']:.3f} ms; N-KERNEL's ssn_1l "
                  f"x words {p['t2_nkernel_scaled'][0]:.1f} beside / {p['t2_nkernel_scaled'][1]:.1f} serial")
        lines.append(s)
    return lines


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
    closing = [x for x in res["designs"] if x["id"].startswith(("blc1-t", "blc1a-t"))
               and all(p["caps"]["32"]["stated"]["bar"] <= 0.01 for p in x["points"].values())]
    def a2c_open(x: dict) -> bool:
        return x["a2c"] is None or x["a2c"]["model_p_share"] > 0.01

    best = min(closing, key=lambda x: (x["points"]["t8192"]["served"] != "exact", a2c_open(x),
                                       x["points"]["t8192"]["stream_bytes_per_token"],
                                       x["points"]["t8192"]["caps"]["32"]["stated"]["bar"]))
    res["closing"] = {"id": best["id"], "title": next(d["title"] for d in designs(einp["layouts"])
                                                      if d["id"] == best["id"])}
    exact320 = [x for x in res["designs"] if x["points"]["t8192"]["served"] == "exact"
                and all(p["caps"]["320"]["stated"]["bar"] <= 0.01 for p in x["points"].values())]
    best320 = min(exact320, key=lambda x: (a2c_open(x), x["points"]["t8192"]["stream_bytes_per_token"],
                                           x["points"]["t8192"]["caps"]["320"]["stated"]["bar"]))
    res["closing_320"] = {"id": best320["id"], "title": next(d["title"] for d in designs(einp["layouts"])
                                                             if d["id"] == best320["id"])}
    res["lower_bound"] = lower_bound(inp, ll, res)
    res["statement_check"] = statement_check(inp, ll, res)
    p8 = best["points"]["t8192"]["caps"]
    lead8 = res["designs_by_id"]["rows"]["points"]["t8192"]["caps"]
    res["period"] = per_period(inp, [
        ("stock work", 1.0),
        ("the lead's network term, cap 32, floor", lead8["32"]["stated"]["floor"]),
        ("the lead's network term, cap 32, bar", lead8["32"]["stated"]["bar"]),
        ("the de-noised lead's call-set outsourcer, cap 32's slack, floor",
         lead8["32"]["attack"]["floor"]["subset"]["slack"]),
        (f"{best['id']}'s network term, cap 32, floor", p8["32"]["stated"]["floor"]),
        (f"{best['id']}'s network term, cap 32, bar", p8["32"]["stated"]["bar"]),
        (f"{best['id']}'s lane forming, uncredited", best["points"]["t8192"]["forming_share_of_stock"]),
        (f"{best['id']}'s A2c / A8 residual, model P, at its worst point ({best['a2c']['worst_point']})",
         best["a2c"]["model_p_share"]),
        (f"{best320['id']}'s A2c / A8 residual, model P, at its worst point ({best320['a2c']['worst_point']})",
         best320["a2c"]["model_p_share"]),
        (f"{best320['id']}'s network term, cap 320, floor", best320["points"]["t8192"]["caps"]["320"]["stated"]["floor"]),
        (f"{best320['id']}'s network term, cap 320, bar", best320["points"]["t8192"]["caps"]["320"]["stated"]["bar"]),
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
