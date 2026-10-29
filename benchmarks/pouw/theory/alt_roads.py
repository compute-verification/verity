"""Roads outside DistinctWrites for PoUW on one H200: each road's best design, its terms and its verdict.

Every term is a share of stock's useful int8 MACs (W1 units: a MAC is 1, a register write 32) unless it says it is a
time or a rate. Inputs and their sources are in ``alt_roads_inputs.json``. Writes ``alt_roads.json`` and
``table.txt`` to ``--out``; exits non-zero if a self-check fails.

The roads:

- R0, the commitment-volume barrier: what any design that forces work by counting committed words must hash, and
  what a commitment computed on the prover's GPU must cost, so whether hashing can ever leave the credit.
- R1, commitment by an auditor-trusted device: the words leave the GPU to a network certifier, which commits them.
- R2, memory- or bandwidth-forced work: storing the words and challenging their presence.
- R3, capacity (PoUS occupancy) beside PoUW: bounding advice and precomputation.
- R4, the capacity reading: time and the device's assumed peak capacity, per pipe.
- R5, energy as the price (outside the design rules; a what-if).
- R6, an algebraic commitment on the tensor pipe (outside the one-hash rule; a what-if).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINES = ("t_stock_s", "t_stock_async_s")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def llama(inp: dict) -> dict:
    cfg = inp["llama_3_2_1b"]
    macs = cfg["layers"] * sum(k * n for k, n in cfg["linears"])
    outs = cfg["layers"] * sum(n for _, n in cfg["linears"])
    a_bytes = cfg["layers"] * sum(k for k, _ in cfg["linears"])
    attn_per_tt = cfg["layers"] * cfg["heads"] * cfg["head_dim"]
    return {"macs_per_token": macs, "outs_per_token": outs, "a_bytes_per_token": a_bytes,
            "attn_macs_per_t_tp1": attn_per_tt}


def peak(inp: dict) -> float:
    h = inp["h200"]
    return h["tensor_units_per_sm_clk"] * h["sms"] * h["tensor_clk_hz"]


def words_bytes_per_mac(alpha: float) -> float:
    return 4.0 * alpha / 32.0


def t2_on_gpu(inp: dict, point: dict, macs: float, alpha: float, u: float) -> float:
    """SW-NOISE's T2: stock, the stack's extra GEMM, hashing and export of every k32 word, and serial coins."""
    s = inp["sha512"]
    byts = words_bytes_per_mac(alpha) * macs
    hash_s = byts * (u + s["export_units_per_word"] / 4.0) / peak(inp)
    t = point["t_stock_s"] + (alpha - 1.0) * point["t_gemm_s"] + hash_s + inp["ssn_1l"]["coin_latency_s"]
    return t / point["t_stock_s"]


def export_s(inp: dict, word_bytes: float) -> float:
    """The words' store to HBM at SW-GPU's export price, as time at the peak rate."""
    return word_bytes * inp["sha512"]["export_units_per_word"] / 4.0 / peak(inp)


def r0_barrier(inp: dict) -> dict:
    """Any design that forces work by counting committed words, with every committed byte hashed on the GPU."""
    w1, s = inp["w1"], inp["sha512"]
    # A commitment that depends on all N committed words ends with at most O components holding them, and an
    # instruction reading r source words merges at most r of them into one, while H32 charges each of its w registers.
    merge = []
    for name, r, w in inp["r0_merge_instructions"]["rows"]:
        merge.append({"instruction": name, "source_words": r, "registers_written": w,
                      "units_per_byte": w1["h32_units_per_write"] * w / (r - 1) / 4.0})
    p_floor = min(x["units_per_byte"] for x in merge)
    rows = []
    for credit, why in ((32, "H32 alone (rank 1)"), (64, "N-DENSE's target: a word every 64 fully credited at rank <= 3")):
        b_min = 4.0 * (1.0 - 0.01) / credit
        p_allowed = 0.01 / b_min
        rows.append({
            "credit_units_per_word": credit, "why": why,
            "min_bytes_per_useful_mac": b_min,
            "uncredited_price_allowed_units_per_byte": p_allowed,
            "cheapest_on_gpu_commitment_units_per_byte": p_floor,
            "factor_over_allowed": p_floor / p_allowed,
            "sha512_floor_factor": s["floor_units_per_byte"] / p_allowed,
            "hashing_share_of_stock_at_sha_floor": b_min * s["floor_units_per_byte"],
            "hashing_share_of_stock_at_bar": b_min * s["bar_units_per_byte"],
        })
    precision = []
    for name, b in (("ssn_1l", words_bytes_per_mac(inp["ssn_1l"]["alpha"])),
                    ("ncp3", words_bytes_per_mac(inp["ncp3"]["alpha"])),
                    ("rank-3 minimum", rows[1]["min_bytes_per_useful_mac"])):
        for pname in ("floor_units_per_byte", "bar_units_per_byte", "measured_alone_units_per_byte"):
            p = s[pname]
            precision.append({"design": name, "price": pname, "units_per_byte": p,
                              "relative_precision_needed": 0.01 / (b * p),
                              "absolute_slack_units_per_byte": 0.01 / b})
    # BamFloor credits 64 - 1024/L for a word every L MACs; the 1% budget needs 64 - 1024/L >= 0.99 L.
    disc = 64.0 ** 2 - 4 * 0.99 * 1024.0
    bam_window = [(64.0 - math.sqrt(disc)) / (2 * 0.99), (64.0 + math.sqrt(disc)) / (2 * 0.99)]
    return {"rows": rows, "merge_floor": merge, "credited_hash_precision": precision,
            "bamfloor_interval_within_1pct": bam_window}


def r1_egress(inp: dict, ll: dict) -> dict:
    """SSN's words and the operand values streamed to a network certifier, which commits them on auditor hardware."""
    alpha = inp["ssn_1l"]["alpha"]
    b = words_bytes_per_mac(alpha)
    out = []
    for key, point in inp["points"].items():
        macs = ll["macs_per_token"] * point["m"]
        byts = b * macs + ll["a_bytes_per_token"] * point["m"]
        extra = (alpha - 1.0) * point["t_gemm_s"] + inp["ssn_1l"]["coin_latency_s"] + export_s(inp, b * macs)
        base = point["t_stock_s"] + extra
        row = {"point": key, "label": point["label"], "bytes_per_step": byts,
               "rate_at_stock_speed_bytes_per_s": byts / point["t_stock_s"],
               "hbm_factor_at_stock_speed": byts / point["t_stock_s"] / inp["h200"]["hbm_bytes_per_s"]}
        slow = inp["links"]["gemm_slowdown_under_d2h_measured"][1]
        for lname, rate in (("census_100g", inp["links"]["census_reference_bytes_per_s"]),
                            ("gpu_400g", inp["links"]["per_gpu_400g_bytes_per_s"])):
            t_e = byts / rate
            judged = max(max(point[e] + extra + (slow - 1.0) * alpha * point["t_gemm_s"], t_e) / point[e]
                         for e in ENGINES)
            row[lname] = {"egress_s": t_e,
                          "t2_overlapped": max(base, t_e) / point["t_stock_s"],
                          "t2_serial": (base + t_e) / point["t_stock_s"],
                          "t2_judged": judged, "passes_t2": judged <= 100.0}
        if "t_stock_band_s" in point:
            row["t2_overlapped_band_gpu_400g"] = [max(base - point["t_stock_s"] + t, byts / inp["links"]["per_gpu_400g_bytes_per_s"]) / t
                                                  for t in point["t_stock_band_s"]]
        row["pcie_egress_s"] = byts / inp["links"]["pcie_d2h_measured_bytes_per_s"][0]
        out.append(row)
    sha_rate = inp["h200"]["sms"] * inp["h200"]["fmax_hz"] * 128.0 / inp["sha512"]["sm_clk_per_compression"]
    certifier = {"h200_sha512_bytes_per_s": sha_rate,
                 "certifier_h200s_per_prover_gpu_at_400g": inp["links"]["per_gpu_400g_bytes_per_s"] / sha_rate}
    # T2 <= x exactly when stock's useful MAC rate is at most x * link / bytes a MAC: a share of the tensor peak.
    links = {"census_100g": inp["links"]["census_reference_bytes_per_s"],
             "gpu_400g": inp["links"]["per_gpu_400g_bytes_per_s"],
             "nvlink_900g": inp["links"]["nvlink_domain_bytes_per_s"][0],
             "nvlink_1800g": inp["links"]["nvlink_domain_bytes_per_s"][1]}
    criterion = {name: {f"t2_le_{x}": x * rate / b / peak(inp) for x in (10, 100)} for name, rate in links.items()}
    stock_share = {key: ll["macs_per_token"] * p["m"] / min(p[e] for e in ENGINES) / peak(inp) for key, p in inp["points"].items()}
    p8 = inp["points"]["t8192"]
    t8192_nvlink = {name: b * ll["macs_per_token"] * p8["m"] / rate / min(p8[e] for e in ENGINES)
                    for name, rate in links.items() if name.startswith("nvlink")}
    rtt = {f"{r * 1e6:g}us": {key: 64 * r / min(p[e] for e in ENGINES) for key, p in inp["points"].items()}
           for r in (2e-6, 10e-6, 1e-3)}
    pcie = inp["links"]["pcie_d2h_measured_bytes_per_s"]
    return {
        "certifier_hashing": certifier,
        "pcie_egress_measured": {"d2h_bytes_per_s": pcie,
                                 "port_400g_share_of_pcie": [inp["links"]["per_gpu_400g_bytes_per_s"] / x for x in reversed(pcie)],
                                 "gemm_slowdown_under_d2h": inp["links"]["gemm_slowdown_under_d2h_measured"]},
        "criterion_max_stock_tensor_share": criterion,
        "stock_useful_tensor_share": stock_share,
        "t2_t8192_nvlink_domain_sensitivity": t8192_nvlink,
        "t2_added_by_coin_round_trips_64_calls": rtt,
        "design": "ssn_1l + certifier commit (every k32 word and every operand value leaves the GPU; the certifier commits under hiding leaves)",
        "t1_matmul_credited_coins": inp["ssn_1l"]["t1_credited_coins"],
        "t1_matmul_uncredited_coins": inp["ssn_1l"]["t1_uncredited_coins"],
        "t1_hashing_on_prover": 0.0,
        "t1_hiding_on_prover": 0.0,
        "t1_hashing_before_noise_on_prover": 0.0,
        "premises": ["H32 (1, measured)", "W1 (3)", "trusted-host (3, a named assumption the design rules admit)",
                     "SSN distinctness lemma (0 once locked)", "the link's bytes are the GPU's (PeriodInventory, 3)"],
        "points": out,
    }


def stock_step_s(inp: dict, m: float, field: str = "t_stock_s") -> float:
    """Stock's step at m rows, linear between the decode, 64-token and T 8,192 points (an estimate)."""
    pts = sorted(inp["points"].values(), key=lambda p: p["m"])
    for lo, hi in zip(pts, pts[1:]):
        if m <= hi["m"]:
            return lo[field] + (hi[field] - lo[field]) * (m - lo["m"]) / (hi["m"] - lo["m"])
    return pts[-1][field] * m / pts[-1]["m"]


def r1_shapes(inp: dict, ll: dict) -> dict:
    """R1's T2 against a stock at the HBM roofline (any model), Llama between its points, and V4's dense and experts."""
    alpha, hbm, pk = inp["ssn_1l"]["alpha"], inp["h200"]["hbm_bytes_per_s"], peak(inp)
    b = words_bytes_per_mac(alpha)
    links = {"census_100g": inp["links"]["census_reference_bytes_per_s"],
             "gpu_400g": inp["links"]["per_gpu_400g_bytes_per_s"]}
    # Against a stock that reads its int8 weights once a step at HBM's rate, r rows a weight byte make r MACs a weight
    # byte, so the link's time over stock's is (b + 1/n)·r·HBM/link until r reaches peak/HBM and stock turns compute-bound.
    roofline = {}
    for name, (k, n) in (("llama_min_n", (2048, 2048)), ("v4_dense", tuple(inp["v4"]["dense"]))):
        be = b + 1.0 / n
        roofline[name] = {ln: {"t2_per_row": be * hbm / rate, "rows_t2_le_100": 100.0 * rate / (be * hbm),
                               "rows_t2_le_10": 10.0 * rate / (be * hbm), "compute_bound_t2": be * pk / rate}
                          for ln, rate in links.items()}
    knee_rows = pk / hbm
    e, kk = inp["v4"]["experts"], inp["v4"]["top_k"]
    be_x = b + 1.0 / min(n for _, n in inp["v4"]["expert"])

    def rows_per_expert(m: int) -> float:
        return m * kk / (e * (1.0 - (1.0 - kk / e) ** m))

    experts = {}
    for ln, rate in links.items():
        cap = 100.0 * rate / (be_x * hbm)
        experts[ln] = {"max_tokens_t2_le_100": max((m for m in range(1, 65537) if rows_per_expert(m) <= cap), default=0),
                       "rows_per_expert_allowed": cap}
    per_row = b * ll["macs_per_token"] + ll["a_bytes_per_token"]
    extra = lambda m: ((alpha - 1.0) * stock_step_s(inp, m, "t_gemm_s") + inp["ssn_1l"]["coin_latency_s"]
                       + export_s(inp, b * ll["macs_per_token"] * m))
    llama = {}
    for ln, rate in links.items():
        t2 = lambda m: max(max(stock_step_s(inp, m, e) + extra(m), per_row * m / rate) / stock_step_s(inp, m, e)
                           for e in ENGINES)
        fit = max((m for m in range(1, 8193) if t2(m) <= 100.0), default=0)
        fast = min(stock_step_s(inp, fit, e) for e in ENGINES)
        llama[ln] = {"t2_judged_512_rows": t2(512), "t2_judged_256_rows": t2(256),
                     "max_rows_t2_le_100": fit, "stock_step_at_max_rows_s": fast,
                     "stock_share_at_max_rows": ll["macs_per_token"] * fit / fast / pk}
    return {"roofline": roofline, "roofline_knee_rows": knee_rows,
            "v4_experts_uniform_routing": experts, "v4_rows_per_expert_at": {m: rows_per_expert(m) for m in (64, 256, 512, 8192)},
            "llama_interpolated": llama}


def r1b_salt_and_arrival(inp: dict, ll: dict) -> dict:
    """The lead with its words hashed on the GPU, but each hm96 leaf's salt side (M·y and c) supplied by the network
    certifier, and each call's A_j sent to it, which releases the call's coins on arrival."""
    alpha, r = inp["ssn_1l"]["alpha"], inp["r1b"]
    b = words_bytes_per_mac(alpha)
    sha_compressions = inp["h200"]["sms"] * inp["h200"]["fmax_hz"] / inp["sha512"]["sm_clk_per_compression"]
    links = {"census_100g": inp["links"]["census_reference_bytes_per_s"],
             "gpu_400g": inp["links"]["per_gpu_400g_bytes_per_s"]}
    out = []
    for key, t2s in (("t8192", inp["ssn_1l"]["t2_t8192"]), ("t64", inp["ssn_1l"]["t2_t64"])):
        point = inp["points"][key]
        words = b * ll["macs_per_token"] * point["m"]
        a_bytes = ll["a_bytes_per_token"] * point["m"]
        step = t2s[1] * point["t_stock_s"]
        rows = {}
        for rname, rb in r["row_bytes"].items():
            leaves = words / rb
            rows[rname] = {"leaves_per_step": leaves,
                           "inbound_bytes_per_step": leaves * r["salt_side_bytes_per_leaf"],
                           "inbound_share_of_words": r["salt_side_bytes_per_leaf"] / rb,
                           "inbound_bytes_per_s": leaves * r["salt_side_bytes_per_leaf"] / step,
                           "certifier_compressions_per_s": leaves * r["salt_side_compressions_per_leaf"] / step,
                           "certifier_h200s_of_sha512": leaves * r["salt_side_compressions_per_leaf"] / step / sha_compressions}
        out.append({"point": key, "honest_step_overlapped_s": step, "a_bytes_per_step": a_bytes,
                    "outbound_bytes_per_s": a_bytes / step,
                    "outbound_link_share": {ln: a_bytes / step / rate for ln, rate in links.items()},
                    "t2_change_by_rtt": {f"{rtt * 1e6:g}us": (64 * rtt - inp["ssn_1l"]["coin_latency_s"]) / point["t_stock_s"]
                                         for rtt in (2e-6, 10e-6, 1e-3)},
                    "rows": rows})
    return {"design": "ssn-1l-tag (or ssn_1l_hw), words hashed on the GPU; salt sides from the certifier; A_j to it and coins back "
                      "on arrival",
            "points": out,
            "terms_removed_t8192": {"hiding_k32": r["hiding_term_t8192_k32"], "hashing_before_noise": r["prenoise_term_t8192"]},
            "premises_removed": ["atomic call + PeriodInventory for hashing A before its noise (3)", "the grinding bound (open)",
                                 "PoUW's Fiat-Shamir seed exception (the rule's, not a premise)"],
            "premises_added": ["trusted-host (3)"]}


def r1b_arrival_t2(inp: dict, ll: dict) -> dict:
    """N-CARD's T2 for R1b's designs with arrival-order coins: A's root chain before each call's noise leaves the prover,
    and each call waits instead for its A_j to reach the certifier and its coins to come back."""
    s = inp["space_t2"]
    links = {"census_100g": inp["links"]["census_reference_bytes_per_s"],
             "gpu_400g": inp["links"]["per_gpu_400g_bytes_per_s"]}
    out = {}
    for design, pts in s["designs"].items():
        out[design] = {}
        for key, v in pts.items():
            a_bytes = ll["a_bytes_per_token"] * v["m"]
            rest_ms = v["t_h_ms"] - v["a_root_ms"]
            out[design][key] = {
                "space_t2": v["t_h_ms"] / v["stock_ms"], "space_t2_judged": v["judged"],
                "a_root_share_of_honest_step": v["a_root_ms"] / v["t_h_ms"],
                "arrival_t2": {ln: [(rest_ms + (a_bytes / rate + s["serial_calls"] * rtt) * 1e3) / v["stock_ms"]
                                    for rtt in s["rtt_s"]] for ln, rate in links.items()}}
    return {"design": "R1b's arrival-order coins on N-CARD's T2 (async engine)", "rtt_s": s["rtt_s"], "designs": out}


def egress_outsourcing(inp: dict, ll: dict) -> dict:
    """OutsourceWithinEgress's term at caps of served traffic, as a share of stock: an outside machine does the work
    a counted bit carries, beta = n_max W_mac (DitherInstance), n_max the most MACs one call's coin bit carries (a node
    that forwards a call's coins and gets back the root has the call's MACs done) and W_mac with the block's hashing."""
    cfg, e, s = inp["llama_3_2_1b"], inp["egress"], inp["sha512"]
    alpha, c = inp["ssn_1l"]["alpha"], e["code_lanes"]
    steps = lambda k: 1 + math.ceil(max(0, k - (32 - c)) / 31)
    row_bits = lambda k: 32 * math.ceil((steps(k) + e["coin_bits_over_steps"]) / 32)
    coin_bits = cfg["layers"] * sum(row_bits(k) for k, _ in cfg["linears"])
    per_call = {f"k{k}_n{n}": k * n / row_bits(k) for k, n in cfg["linears"]}
    n_max = max(per_call.values())
    w_mac = {price: alpha + e["credited_hash_bytes_per_mac"] * s[price]
             for price in ("floor_units_per_byte", "bar_units_per_byte")}
    mac_per_word_bit = 1.0 / (8.0 * words_bytes_per_mac(alpha))
    t8192 = inp["space_t2"]["designs"]["ssn-1l-tag/served/s8/k32/w/r256k/cert/min"]["t8192"]
    tokens_per_s = {"honest": t8192["m"] / (t8192["t_h_ms"] * 1e-3), "stock": t8192["m"] / (t8192["stock_ms"] * 1e-3)}
    caps = {}
    for cap in e["cap_bits_per_token"]:
        n_req = {p: 0.01 * ll["macs_per_token"] / (cap * w) for p, w in w_mac.items()}
        elem_bits = {p: cfg["layers"] * sum(n * math.ceil(k / nr) for k, n in cfg["linears"]) for p, nr in n_req.items()}
        caps[str(cap)] = {
            "r1b_live_coins": {p: cap * n_max * w / ll["macs_per_token"] for p, w in w_mac.items()},
            "n_max_for_1pct": n_req,
            "row_coins_meet_it_up_to_columns": {p: min(nr * row_bits(k) / k for k, _ in cfg["linears"])
                                                for p, nr in n_req.items()},
            "element_coins_bits_per_token": elem_bits,
            "element_coins_bytes_per_s_t8192": {pace: {p: b / 8.0 * tps for p, b in elem_bits.items()}
                                                for pace, tps in tokens_per_s.items()},
            "r1_words_through_the_port": cap * mac_per_word_bit * alpha / ll["macs_per_token"]}
    return {"coin_bits_per_token": coin_bits, "macs_per_coin_bit_by_call": per_call, "n_max": n_max,
            "macs_per_coin_bit_mean": ll["macs_per_token"] / coin_bits,
            "w_mac": w_mac, "r1_macs_per_imported_bit": mac_per_word_bit,
            "seeded_lead": f"false: {e['seed_bits_per_window']} bits (the window's salt) carry every call's noise, "
                           "so one counted message outsources the window", "share_of_stock_at_cap": caps}


def r2_memory(inp: dict, ll: dict) -> dict:
    """Store every word in HBM and let a memory challenger demand random ones against a deadline."""
    alpha = inp["ssn_1l"]["alpha"]
    cfg = inp["llama_3_2_1b"]
    m = inp["points"]["t8192"]["m"]
    per_call = {f"k{k}_n{n}": m * n * (k * alpha / 32.0) * 4.0 for k, n in cfg["linears"]}
    step = words_bytes_per_mac(alpha) * ll["macs_per_token"] * m
    k = max(kk for kk, _ in cfg["linears"])
    # One word recomputed on the lanes: K/4 IDP4A over 32 lanes, then a 5-level shuffle reduction, ~5 clocks each.
    recompute_s = (k / 4.0 / 32.0 + 5.0) * 5.0 / inp["h200"]["fmax_hz"]
    return {"words_bytes_per_call_t8192": per_call, "words_bytes_per_step_t8192": step,
            "hbm_bytes": inp["h200"]["hbm_bytes"], "step_over_hbm": step / inp["h200"]["hbm_bytes"],
            "recompute_one_word_s": recompute_s,
            "census_rtt_s": 1e-3, "rtt_over_recompute": 1e-3 / recompute_s,
            "rdma_rtt_s": 2e-6, "rdma_rtt_over_recompute": 2e-6 / recompute_s,
            "stored_word_credit_if_store_forced_units": inp["sha512"]["export_units_per_word"]}


def r3_capacity_occupancy(inp: dict, ll: dict) -> dict:
    """PoUS occupancy bounds the adversary's advice; what advice is worth under DistinctWrites."""
    w1 = inp["w1"]
    weights = ll["macs_per_token"] + 2048 * 128256
    return {"llama_weight_bytes_int8": weights,
            "advice_bound_after_pous_bytes": inp["h200"]["hbm_bytes"] - weights,
            "advice_share_of_hbm": 1.0 - weights / inp["h200"]["hbm_bytes"],
            "lookup_over_k32_word_l1": w1["l1_lookup_units_per_word"] / w1["h32_units_per_write"],
            "lookup_over_k32_word_l2": w1["l2_lookup_units_per_word"] / w1["h32_units_per_write"]}


def stock_tensor_share(inp: dict, ll: dict, point: dict, with_attention: bool) -> float:
    m = point["m"]
    work = ll["macs_per_token"] * m
    if with_attention and m > 1:
        work += 2.0 * ll["attn_macs_per_t_tp1"] * m * (m + 1)
    return work / (peak(inp) * point["t_stock_s"])


def r4_capacity_reading(inp: dict, ll: dict) -> dict:
    """Time and each pipe's assumed peak: the tensor capacity the honest step leaves idle, beyond stock's own."""
    s, acc = inp["sha512"], inp["accounting"]
    rows = []
    for key, t2s in (("t8192", inp["ssn_1l"]["t2_t8192"]), ("t64", inp["ssn_1l"]["t2_t64"])):
        point = inp["points"][key]
        macs = ll["macs_per_token"] * point["m"]
        e_lin = stock_tensor_share(inp, ll, point, False)
        e_all = stock_tensor_share(inp, ll, point, True)
        for sched, t2 in zip(("serial", "overlapped"), t2s):
            alpha = inp["ssn_1l"]["alpha"]
            excess = ((t2 - 1.0) * point["t_stock_s"] * peak(inp) - (alpha - 1.0) * macs) / macs
            rows.append({"point": key, "schedule": sched, "t2": t2,
                         "stock_tensor_share_linears": e_lin, "stock_tensor_share_with_attention": e_all,
                         "stock_idle_tensor_share": [1.0 - e_all, 1.0 - e_lin],
                         "excess_idle_stock_macs_per_stock_mac": excess,
                         "over_budget": excess / 0.01,
                         "t2_allowed": 1.0 + (0.01 + alpha - 1.0) * e_lin,
                         "excess_at_co_run_rate": acc["co_run_peak_share_beside_hash"] * (t2 - 1.0) / e_lin})
    ratio_floor = 0.0619 * s["floor_units_per_byte"]
    ratio_sha = 0.0619 * s["sm_clk_per_compression"] * inp["h200"]["tensor_units_per_sm_clk"] / 128.0
    point = inp["points"]["t8192"]
    values_b = (ll["a_bytes_per_token"] + 2 * ll["outs_per_token"]) / ll["macs_per_token"]
    v4_values_b = 1.0 / 7168 + 2.0 / 7168
    # Decision 74 credits T_cred = B_flat/f_max, a time. With the words' floor priced on the tensor pipe alone, an
    # adversary writing a share of them on the lanes beside it frees s/(1+s) of the credited tensor time; a floor that
    # takes the maximum over pipes closes that, and then the honest kernel trails it by s unless it co-issues too.
    lane = inp["w1"]["lane_useful_share_of_tensor"]
    t_cred = {"on_gpu_hashing_over_credit_if_sha_speed_attained": {f"{r['point']}_{r['schedule']}": r["excess_idle_stock_macs_per_stock_mac"]
                                                                  for r in rows},
              "certifier_commit_over_credit_tensor_only_floor": lane / (1.0 + lane),
              "certifier_commit_honest_gap_co_issued_floor": lane}
    return {
        "rows": rows,
        "t_cred_reading": t_cred,
        "word_commit_hash_over_tensor_time_at_rank3_min": {"sha_floor": ratio_floor, "sha_measured_rate": ratio_sha,
                                                           "tensor_idle_share_at_least": 1.0 - 1.0 / ratio_floor},
        "values_only_bytes_per_mac_llama": values_b,
        "values_only_hash_peak_slots_per_mac_llama": values_b * s["bar_units_per_byte"],
        "stock_peak_slots_per_mac_t8192": 1.0 / stock_tensor_share(inp, ll, point, False),
        "values_only_bytes_per_mac_v4_dense": v4_values_b,
        "values_only_hash_peak_slots_per_mac_v4_dense": v4_values_b * s["bar_units_per_byte"],
    }


def certifier_capacity(inp: dict, ll: dict, r1: dict) -> list[dict]:
    """R4's reading of R1: the egress-bound step leaves the tensor pipe idle too."""
    alpha = inp["ssn_1l"]["alpha"]
    rows = []
    for p in r1["points"]:
        if p["point"] == "t8192":
            continue
        point = inp["points"][p["point"]]
        macs = ll["macs_per_token"] * point["m"]
        t2 = p["gpu_400g"]["t2_judged"]
        excess = ((t2 - 1.0) * min(point[e] for e in ENGINES) * peak(inp) - (alpha - 1.0) * macs) / macs
        rows.append({"point": p["point"], "t2": t2, "excess_idle_stock_macs_per_stock_mac": excess})
    return rows


def r6_algebraic(inp: dict) -> dict:
    b = words_bytes_per_mac(inp["ssn_1l"]["alpha"])
    rows = []
    for n_rows in (8, 64, 256, 512):
        hashing = b * n_rows
        rows.append({"rows": n_rows, "units_per_byte": n_rows, "hashing_share_of_stock": hashing,
                     "cheaper_than_sha_bar": inp["sha512"]["bar_units_per_byte"] / n_rows,
                     "idle_lane_capacity_stock_equiv": hashing * inp["w1"]["lane_useful_share_of_tensor"]})
    return {"ssn_bytes_per_mac": b, "rows": rows}


def per_period_bounds(inp: dict, res: dict) -> dict:
    """Each new bound as a share of the stock work per period (N-CARD's one basis, stock work held fixed): one H200
    serves stock at the tensor peak times mfu_serve, the datacenter is dc_gpus of them, the world serves
    world-inference-2027; T2 is the hardware beside the pool, never folded in."""
    acc = inp["accounting"]
    ands = acc["and_per_int8_mac"]
    periods = {"window": acc["window_s"], "week": acc["week_s"], "month": acc["month_s"], "year": acc["year_s"]}
    h200 = peak(inp) * acc["mfu_serve"]
    rates = {"one H200": h200, "datacenter": h200 * acc["dc_gpus"], "world": acc["world_flop_per_year"] / 2.0 / acc["year_s"]}
    stock = {scope: {p: rate * s * ands for p, s in periods.items()} for scope, rate in rates.items()}
    r1 = {p["point"]: p for p in res["R1"]["points"]}
    hw = inp["ssn_1l_hw"]
    r4 = next(r for r in res["R4"]["rows"] if r["point"] == "t8192" and r["schedule"] == "overlapped")
    t2 = lambda key: r1[key]["gpu_400g"]["t2_judged"]
    r1b_t2 = inp["space_t2"]["designs"]["ssn-1l-tag/served/s8/k32/w/r256k/cert/min"]["t8192"]["judged"]
    bounds = [
        ("R1 + ssn_1l_hw, T1 closed (forming and coin branch)", "T 512, 64 rows", t2("t64"), hw["t1_closed"]["t64"]),
        ("R1 + ssn_1l_hw, T1 closed (forming and coin branch)", "T 512, 512 rows", t2("t512"), hw["t1_closed"]["t512rows"]),
        ("R1 + ssn_1l_hw, T1 closed (forming and coin branch)", "T 8,192", t2("t8192"), hw["t1_closed"]["t8192"]),
        ("R1 + ssn_1l_hw, T1 with coins uncredited", "T 512, 64 rows", t2("t64"), hw["t1_coins_uncredited"]["t64"]),
        ("R1 + ssn_1l_hw, T1 with coins uncredited", "T 8,192", t2("t8192"), hw["t1_coins_uncredited"]["t8192"]),
        ("R1 under T_cred, over-credit with a tensor-only floor", "T 512, 64 rows", t2("t64"),
         res["R4"]["t_cred_reading"]["certifier_commit_over_credit_tensor_only_floor"]),
        ("R1b + ssn_1l_hw at 256 KiB rows: hiding term removed (W1 floor)", "T 8,192", r1b_t2, hw["hiding_salt_side"]["t8192"]),
        ("R1b + ssn_1l_hw: hashing A before its noise removed (W1 floor)", "T 8,192", r1b_t2, hw["prenoise"]["t8192"]),
        ("R4 elapsed-time reading: tensor pipe idle beyond stock's, SSN overlapped", "T 8,192", r4["t2"],
         r4["excess_idle_stock_macs_per_stock_mac"]),
        ("R1b egress: OutsourceWithinEgress at a 32-bit-a-token cap, W_mac at SHA-512's floor", "any", None,
         res["egress"]["share_of_stock_at_cap"]["32"]["r1b_live_coins"]["floor_units_per_byte"]),
        ("R1 egress: words imported through the port at a 32-bit-a-token cap", "any", None,
         res["egress"]["share_of_stock_at_cap"]["32"]["r1_words_through_the_port"]),
        ("1% allowance", "any", None, 0.01),
    ]
    rows = [{"bound": name, "point": point, "t2_beside": t2v, "share_of_stock": share,
             "and_gates": {scope: {p: share * v for p, v in ps.items()} for scope, ps in stock.items()}}
            for name, point, t2v, share in bounds]
    return {"basis": "N-CARD's one basis, stock work held fixed: one H200 at the tensor peak x mfu_serve "
                     f"{acc['mfu_serve']}, dc-12500-machines-600s's {acc['dc_gpus']:,} GPUs, world-inference-2027, "
                     f"{ands} AND gates an int8 MAC, window {acc['window_s']} s; T2 beside, not folded in",
            "stock_mac_per_s": rates, "stock_work_and_gates": stock, "rows": rows}


def checks(inp: dict, ll: dict, res: dict) -> list[tuple[str, bool, str]]:
    s = inp["sha512"]
    pts = inp["points"]
    macs8k = ll["macs_per_token"] * pts["t8192"]["m"]
    macs64 = ll["macs_per_token"] * pts["t64"]["m"]
    a = inp["ssn_1l"]["alpha"]
    t2 = [t2_on_gpu(inp, pts["t8192"], macs8k, a, s["measured_alone_units_per_byte"]),
          t2_on_gpu(inp, pts["t8192"], macs8k, a, s["measured_beside_units_per_byte"]),
          t2_on_gpu(inp, pts["t64"], macs64, a, s["measured_alone_units_per_byte"]),
          t2_on_gpu(inp, pts["t64"], macs64, a, s["measured_beside_units_per_byte"])]
    want = inp["ssn_1l"]["t2_t8192"] + inp["ssn_1l"]["t2_t64"]
    sha_bytes = s["sm_clk_per_compression"] * inp["h200"]["tensor_units_per_sm_clk"] / 128.0
    allowance = 0.01 * peak(inp) / 3.0 * inp["accounting"]["window_s"] * inp["accounting"]["and_per_int8_mac"]
    out = [
        ("peak is SW-NOISE's 989.5e12 units a second", abs(peak(inp) / 989.5e12 - 1) < 1e-3, f"{peak(inp):.4e}"),
        ("Llama-3.2-1B's linears are 973,078,528 MACs a token", ll["macs_per_token"] == 973_078_528, str(ll["macs_per_token"])),
        ("SSN commits 0.348 x NCP's 0.375 B a useful MAC", abs(words_bytes_per_mac(a) / 0.375 - 0.348) < 1e-3,
         f"{words_bytes_per_mac(a):.4f}"),
        ("SW-NOISE's T2 reproduces (serial / overlapped, T 8,192 and 64 tokens)",
         all(abs(x - y) <= 0.05 * y for x, y in zip(t2, want)), ", ".join(f"{x:.2f}" for x in t2)),
        ("58.1 SM-clocks a compression is within 10% of SW-GPU's measured price a byte",
         abs(sha_bytes / s["measured_alone_units_per_byte"] - 1) < 0.10, f"{sha_bytes:.0f}"),
        ("SW-PREMISES' T1 allowance per window reproduces (2.57e17 on its basis)",
         abs(allowance / 2.57e17 - 1) < 0.01, f"{allowance:.3e}"),
        ("R0: no on-GPU commitment can leave the credit at rank <= 3",
         all(r["factor_over_allowed"] > 1 for r in res["R0"]["rows"]),
         ", ".join(f"{r['factor_over_allowed']:.1f}x" for r in res["R0"]["rows"])),
        ("R1: certifier commit passes T2 at 64 tokens on a 400 Gb/s port and fails at T 8,192",
         next(p for p in res["R1"]["points"] if p["point"] == "t64")["gpu_400g"]["passes_t2"]
         and not next(p for p in res["R1"]["points"] if p["point"] == "t8192")["gpu_400g"]["passes_t2"], ""),
        ("R1: the stock-share criterion agrees with every T2 row on both links",
         all((res["R1"]["stock_useful_tensor_share"][p["point"]]
              <= res["R1"]["criterion_max_stock_tensor_share"][ln]["t2_le_100"]) == p[ln]["passes_t2"]
             for p in res["R1"]["points"] for ln in ("census_100g", "gpu_400g")), ""),
        ("R0: BamFloor's 1% interval holds L = 32 and not L = 64",
         res["R0"]["bamfloor_interval_within_1pct"][0] <= 32 <= res["R0"]["bamfloor_interval_within_1pct"][1] < 64,
         ", ".join(f"{x:.2f}" for x in res["R0"]["bamfloor_interval_within_1pct"])),
        ("R2: recomputing a word beats the census RTT",
         res["R2"]["rtt_over_recompute"] > 1, f"{res['R2']['rtt_over_recompute']:.0f}x"),
        ("R4: every SSN schedule leaves more than 1% of stock idle on the tensor pipe",
         all(r["over_budget"] > 1 for r in res["R4"]["rows"]), ""),
        ("R0: the merge floor's minimum is sparse mma.sync's 8/3 units a byte",
         abs(min(x["units_per_byte"] for x in res["R0"]["merge_floor"]) - 8.0 / 3.0) < 1e-9,
         f"{min(x['units_per_byte'] for x in res['R0']['merge_floor']):.3f}"),
        ("R1: the interpolated stock step meets every point it is drawn through",
         all(abs(stock_step_s(inp, p["m"]) - p["t_stock_s"]) < 1e-12 for p in pts.values()), ""),
        ("R1: Llama's interpolated fit at 400 Gb/s agrees with the measured-point rows (passes at 64, fails at 8,192)",
         64 <= res["R1"]["shapes"]["llama_interpolated"]["gpu_400g"]["max_rows_t2_le_100"] < 8192,
         str(res["R1"]["shapes"]["llama_interpolated"]["gpu_400g"]["max_rows_t2_le_100"])),
        ("R1: at the roofline's knee the per-row slope meets the compute-bound T2",
         all(abs(v["t2_per_row"] * res["R1"]["shapes"]["roofline_knee_rows"] / v["compute_bound_t2"] - 1) < 1e-9
             for d in res["R1"]["shapes"]["roofline"].values() for v in d.values()), ""),
        ("R1b: A_j is SW-HASH's 229,376 bytes a token", ll["a_bytes_per_token"] == 229_376, str(ll["a_bytes_per_token"])),
        ("R1b: N-CARD's judged T2 reproduces from its async parts (t_H / stock) at every point",
         all(abs(v["space_t2"] / v["space_t2_judged"] - 1) < 0.005
             for pts in res["R1b"]["arrival_t2"]["designs"].values() for v in pts.values()), ""),
        ("Egress: ssn_1l_hw's coins are N-PREM's 3 words a row at K 2,048 and 9 at K 8,192 (18 a layer, 9,216 bits a token)",
         res["egress"]["coin_bits_per_token"] == 9216, str(res["egress"]["coin_bits_per_token"])),
        ("Egress: n_max is gate-up's 2,048 x 16,384 MACs a row over its 96 coin bits",
         abs(res["egress"]["n_max"] - 2048 * 16384 / 96) < 1e-6, f"{res['egress']['n_max']:,.2f}"),
        ("Egress: the credited hashing at the bar is N-CARD's 232 units a MAC",
         abs(inp["egress"]["credited_hash_bytes_per_mac"] * inp["sha512"]["bar_units_per_byte"] / 232.0 - 1) < 0.002,
         f"{inp['egress']['credited_hash_bytes_per_mac'] * inp['sha512']['bar_units_per_byte']:.2f}"),
        ("R1: a 400 Gb/s port fits inside the measured PCIe Gen5 x16 device-to-host rate",
         max(res["R1"]["pcie_egress_measured"]["port_400g_share_of_pcie"]) < 1.0,
         ", ".join(f"{x:.3f}" for x in res["R1"]["pcie_egress_measured"]["port_400g_share_of_pcie"])),
        ("Per period: N-CARD's stock work reproduces (one H200 window 3.08e19, world year 6.08e32 AND gates)",
         abs(res["period_bounds"]["stock_work_and_gates"]["one H200"]["window"] / 3.08e19 - 1) < 0.01
         and abs(res["period_bounds"]["stock_work_and_gates"]["world"]["year"] / 6.08e32 - 1) < 0.01,
         f"{res['period_bounds']['stock_work_and_gates']['one H200']['window']:.3e}, "
         f"{res['period_bounds']['stock_work_and_gates']['world']['year']:.3e}"),
        ("R4 at C26's co-run share reproduces its ~51 stock forwards for NCP at 160x (T 8,192)",
         abs(inp["accounting"]["co_run_peak_share_beside_hash"] * 160.0 / stock_tensor_share(inp, ll, pts["t8192"], False) / 51.0 - 1) < 0.1,
         f"{inp['accounting']['co_run_peak_share_beside_hash'] * 160.0 / stock_tensor_share(inp, ll, pts['t8192'], False):.1f}"),
    ]
    return out


def table(res: dict) -> str:
    lines = []
    r0 = res["R0"]
    lines.append("R0  the commitment-volume barrier")
    for r in r0["rows"]:
        lines.append(f"    credit {r['credit_units_per_word']:>3} a word ({r['why']}): >= {r['min_bytes_per_useful_mac']:.4f} B a "
                     f"useful MAC; uncredited hashing needs <= {r['uncredited_price_allowed_units_per_byte']:.3f} units a byte; the "
                     f"cheapest on-GPU commitment is {r['cheapest_on_gpu_commitment_units_per_byte']:.2f} ({r['factor_over_allowed']:.0f}x); "
                     f"SHA-512 at its floor {r['sha512_floor_factor']:,.0f}x, hashing {r['hashing_share_of_stock_at_sha_floor']:.0f}-"
                     f"{r['hashing_share_of_stock_at_bar']:.0f}x stock")
    lines.append("    merge floor (H32 units a byte merged): " + "; ".join(
        f"{x['instruction'].split(' (')[0]} {x['units_per_byte']:.2f}" for x in r0["merge_floor"]))
    lo, hi = r0["bamfloor_interval_within_1pct"]
    lines.append(f"    BamFloor's 64 - 1024/L meets the 1% budget only for L in [{lo:.1f}, {hi:.1f}]")
    for r in r0["credited_hash_precision"]:
        if r["price"] == "bar_units_per_byte":
            lines.append(f"    credited at the bar, {r['design']}: SHA-512's price must hold to {r['relative_precision_needed']:.2e} "
                         f"({r['absolute_slack_units_per_byte']:.3f} units a byte)")
    r1 = res["R1"]
    lines.append("R1  ssn_1l + certifier commit (no hashing, hiding or pre-noise hashing on the prover)")
    for p in r1["points"]:
        lines.append(f"    {p['point']:>8}: {p['bytes_per_step']:.3e} B a step, {p['rate_at_stock_speed_bytes_per_s']:.3e} B/s at stock "
                     f"speed ({p['hbm_factor_at_stock_speed']:.2f}x HBM); T2 overlapped / serial: 100 Gb/s "
                     f"{p['census_100g']['t2_overlapped']:.1f} / {p['census_100g']['t2_serial']:.1f}, 400 Gb/s "
                     f"{p['gpu_400g']['t2_overlapped']:.1f} / {p['gpu_400g']['t2_serial']:.1f}; judged (the faster stock step, "
                     f"the worst measured slowdown) 100 Gb/s {p['census_100g']['t2_judged']:.1f}, 400 Gb/s {p['gpu_400g']['t2_judged']:.1f}"
                     + (f"; 400 Gb/s over a 0.3-3 ms stock step: {p['t2_overlapped_band_gpu_400g'][0]:.1f}-"
                        f"{p['t2_overlapped_band_gpu_400g'][1]:.1f}" if "t2_overlapped_band_gpu_400g" in p else ""))
    lines.append(f"    the certifier hashes 400 Gb/s with {r1['certifier_hashing']['certifier_h200s_per_prover_gpu_at_400g']:.3f} "
                 "H200s of SHA-512 a prover GPU")
    pe = r1["pcie_egress_measured"]
    lines.append(f"    measured PCIe Gen5 x16 device-to-host {pe['d2h_bytes_per_s'][0]:.3e}-{pe['d2h_bytes_per_s'][1]:.3e} B/s: a 400 Gb/s "
                 f"port is {pe['port_400g_share_of_pcie'][0]:.1%}-{pe['port_400g_share_of_pcie'][1]:.1%} of it; int8 GEMMs under the "
                 f"stream {pe['gemm_slowdown_under_d2h'][0]:.4f}-{pe['gemm_slowdown_under_d2h'][1]:.4f}x")
    crit = r1["criterion_max_stock_tensor_share"]
    lines.append("    fits when stock's useful tensor share is at most (T2 <= 100 / <= 10): "
                 + "; ".join(f"{k} {v['t2_le_100']:.2%} / {v['t2_le_10']:.3%}" for k, v in crit.items()))
    lines.append("    stock's useful tensor share: " + ", ".join(f"{k} {v:.3%}" for k, v in r1["stock_useful_tensor_share"].items()))
    lines.append("    T 8,192 over an NVLink domain spanning machines (what-if, not an HGX H200): "
                 + ", ".join(f"{k} T2 {v:.1f}" for k, v in r1["t2_t8192_nvlink_domain_sensitivity"].items()))
    lines.append("    T2 added by 64 serial coin round trips a step: "
                 + "; ".join(f"{r}: " + ", ".join(f"{k} +{v:.2f}" for k, v in d.items())
                             for r, d in r1["t2_added_by_coin_round_trips_64_calls"].items()))
    sh = r1["shapes"]
    for name, d in sh["roofline"].items():
        lines.append(f"    against a stock at the HBM roofline ({name}): " + "; ".join(
            f"{ln} T2 {v['t2_per_row']:.1f} a row, fits to {v['rows_t2_le_100']:.1f} rows (T2 <= 10 to {v['rows_t2_le_10']:.2f}), "
            f"{v['compute_bound_t2']:,.0f} once compute-bound" for ln, v in d.items())
                     + f"; the knee is {sh['roofline_knee_rows']:.0f} rows")
    lines.append("    V4 routed experts at the roofline, uniform top-8 of 256: " + "; ".join(
        f"{ln} fits to {v['max_tokens_t2_le_100']} tokens ({v['rows_per_expert_allowed']:.2f} rows an expert)"
        for ln, v in sh["v4_experts_uniform_routing"].items()))
    lines.append("    Llama on the interpolated stock steps, judged: " + "; ".join(
        f"{ln} T2 {v['t2_judged_256_rows']:.0f} at 256 rows and {v['t2_judged_512_rows']:.0f} at 512, fits to "
        f"{v['max_rows_t2_le_100']} rows (stock at {v['stock_share_at_max_rows']:.2%} of the peak there)"
        for ln, v in sh["llama_interpolated"].items()))
    for r in res["R4"]["certifier_rows"]:
        lines.append(f"    capacity reading of certifier commit at {r['point']} (400 Gb/s): T2 {r['t2']:.1f}, excess idle "
                     f"{r['excess_idle_stock_macs_per_stock_mac']:.1f} stock a stock MAC")
    r1b = res["R1b"]
    lines.append("R1b " + r1b["design"])
    for p in r1b["points"]:
        lines.append(f"    {p['point']:>6}: A_j out {p['outbound_bytes_per_s']:.2e} B/s at the lead's overlapped pace ("
                     + ", ".join(f"{k} {v:.1%}" for k, v in p["outbound_link_share"].items()) + " of the link); salt sides in: "
                     + "; ".join(f"{k} rows {v['inbound_share_of_words']:.2%} of the words' bytes, {v['inbound_bytes_per_s']:.2e} B/s, "
                                 f"certifier {v['certifier_h200s_of_sha512']:.3f} H200s of SHA-512" for k, v in p["rows"].items())
                     + "; T2 change by 64 round trips less SW-NOISE's coin time: "
                     + ", ".join(f"{k} {v:+.2f}" for k, v in p["t2_change_by_rtt"].items()))
    lines.append("    removes at T 8,192 (W1 floor / alone, x stock): hiding " + ", ".join(
        f"{k} {v[0]}/{v[1]}" for k, v in r1b["terms_removed_t8192"]["hiding_k32"].items()) + "; hashing before the noise "
        + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in r1b["terms_removed_t8192"]["hashing_before_noise"].items()))
    at = r1b["arrival_t2"]
    for design, pts in at["designs"].items():
        lines.append(f"    {design} on N-CARD's T2 (A's root chain's share of the honest step; T2 now -> with arrival coins at "
                     f"{at['rtt_s'][0] * 1e6:g}-{at['rtt_s'][1] * 1e6:g} us): " + "; ".join(
                         f"{k} ({v['a_root_share_of_honest_step']:.0%}) {v['space_t2']:.1f} -> " + ", ".join(
                             f"{ln} {x[0]:.1f}-{x[1]:.1f}" for ln, x in v["arrival_t2"].items()) for k, v in pts.items()))
    eg = res["egress"]
    lines.append(f"Egress (OutsourceWithinEgress at served-traffic caps, share of stock): ssn_1l_hw streams {eg['coin_bits_per_token']:,} "
                 f"coin bits a token; one call's coin bit carries up to n_max {eg['n_max']:,.0f} MACs (" + ", ".join(
                     f"{k} {v:,.0f}" for k, v in eg["macs_per_coin_bit_by_call"].items()) +
                 f"; mean {eg['macs_per_coin_bit_mean']:,.0f}); W_mac {eg['w_mac']['floor_units_per_byte']:.1f}-"
                 f"{eg['w_mac']['bar_units_per_byte']:.1f} (SHA floor - bar); R1 imports words at {eg['r1_macs_per_imported_bit']:.3f} MACs a bit")
    for cap, v in eg["share_of_stock_at_cap"].items():
        f, b = "floor_units_per_byte", "bar_units_per_byte"
        lines.append(f"    cap {cap} bits a token: R1b's live coins {v['r1b_live_coins'][f]:.1%}-{v['r1b_live_coins'][b]:.1%}; "
                     f"R1 {v['r1_words_through_the_port']:.2e}; the seeded lead {eg['seeded_lead']}")
        st = v["element_coins_bytes_per_s_t8192"]
        lines.append(f"        a 1% term needs n_max <= {v['n_max_for_1pct'][f]:,.0f}-{v['n_max_for_1pct'][b]:,.0f} MACs a bit: coins a row "
                     f"meet it only up to {v['row_coins_meet_it_up_to_columns'][f]:.0f}-{v['row_coins_meet_it_up_to_columns'][b]:.0f} "
                     f"columns; coins an output element need {v['element_coins_bits_per_token'][f]:,}-"
                     f"{v['element_coins_bits_per_token'][b]:,} bits a token, at T 8,192 {st['honest'][f]:.2e}-{st['honest'][b]:.2e} B/s "
                     f"at the honest pace and {st['stock'][f]:.2e}-{st['stock'][b]:.2e} at stock's")
    r2 = res["R2"]
    lines.append(f"R2  stored words: {r2['words_bytes_per_step_t8192']:.3e} B a step at T 8,192 ({r2['step_over_hbm']:.1f}x HBM); "
                 f"one word recomputes in {r2['recompute_one_word_s'] * 1e6:.2f} us: census RTT {r2['rtt_over_recompute']:,.0f}x, "
                 f"2 us RDMA {r2['rdma_rtt_over_recompute']:.0f}x")
    r3 = res["R3"]
    lines.append(f"R3  PoUS beside PoUW: Llama's weights leave {r3['advice_share_of_hbm']:.1%} of HBM as advice; a lookup costs "
                 f"{r3['lookup_over_k32_word_l1']:.0f}x (L1) to {r3['lookup_over_k32_word_l2']:.1f}x (L2) a k32 word's charge")
    r4 = res["R4"]
    lines.append("R4  capacity reading (tensor pipe idle beyond stock's own, in stock MACs a stock MAC)")
    for r in r4["rows"]:
        lines.append(f"    {r['point']:>6} {r['schedule']:>10}: T2 {r['t2']:.1f}, stock uses {r['stock_tensor_share_linears']:.1%}-"
                     f"{r['stock_tensor_share_with_attention']:.1%} of the tensor peak, excess idle {r['excess_idle_stock_macs_per_stock_mac']:.1f} "
                     f"({r['over_budget']:,.0f}x the 1% budget; {r['excess_at_co_run_rate']:.1f} at C26's co-run rate); "
                     f"T2 allowed {r['t2_allowed']:.4f}")
    tc = r4["t_cred_reading"]
    lines.append("    T_cred reading: on-GPU hashing over-credits the idle tensor pipe by "
                 + ", ".join(f"{k} {v:.1f}" for k, v in tc["on_gpu_hashing_over_credit_if_sha_speed_attained"].items())
                 + f" stock a stock MAC; certifier commit {tc['certifier_commit_over_credit_tensor_only_floor']:.2%} of its credit "
                 f"(tensor-only floor) or 0 with the honest {tc['certifier_commit_honest_gap_co_issued_floor']:.2%} behind (co-issued floor)")
    w = r4["word_commit_hash_over_tensor_time_at_rank3_min"]
    lines.append(f"    any word-commit design at rank <= 3: hashing / forced tensor time >= {w['sha_floor']:.0f} (SHA floor), "
                 f"{w['sha_measured_rate']:.0f} (measured rate); tensor idle >= {w['tensor_idle_share_at_least']:.1%}")
    lines.append(f"    values-only commitments: Llama {r4['values_only_bytes_per_mac_llama']:.2e} B a MAC, "
                 f"{r4['values_only_hash_peak_slots_per_mac_llama']:.2f} peak slots a MAC of lane hashing against stock's "
                 f"{r4['stock_peak_slots_per_mac_t8192']:.2f}; V4 dense {r4['values_only_hash_peak_slots_per_mac_v4_dense']:.2f}")
    lines.append("R6  algebraic commitment on the tensor pipe (one-hash rule what-if), SSN's bytes")
    for r in res["R6"]["rows"]:
        lines.append(f"    {r['rows']:>4} rows: {r['hashing_share_of_stock']:.1f}x stock, {r['cheaper_than_sha_bar']:.1f}x cheaper than "
                     f"SHA-512's bar; idle lanes {r['idle_lane_capacity_stock_equiv']:.2f} stock")
    pb = res["period_bounds"]
    lines.append("New bounds per monitoring period (" + pb["basis"] + "), AND gates")
    lines.append("    stock work: " + "; ".join(
        f"{scope} " + ", ".join(f"{p} {v:.2e}" for p, v in ps.items()) for scope, ps in pb["stock_work_and_gates"].items()))
    for r in pb["rows"]:
        t2 = f"T2 {r['t2_beside']:.1f} beside" if r["t2_beside"] is not None else "no T2"
        lines.append(f"    {r['bound']} at {r['point']} ({r['share_of_stock']:.4%} of stock, {t2}): " + "; ".join(
            f"{scope} " + ", ".join(f"{p} {v:.2e}" for p, v in ps.items()) for scope, ps in r["and_gates"].items()))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", type=Path, default=HERE / "alt_roads_inputs.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    inp = load(args.inputs)
    ll = llama(inp)
    res = {"R0": r0_barrier(inp), "R1": r1_egress(inp, ll), "R2": r2_memory(inp, ll),
           "R3": r3_capacity_occupancy(inp, ll), "R4": r4_capacity_reading(inp, ll), "R6": r6_algebraic(inp)}
    res["R1"]["shapes"] = r1_shapes(inp, ll)
    res["R1b"] = r1b_salt_and_arrival(inp, ll)
    res["R1b"]["arrival_t2"] = r1b_arrival_t2(inp, ll)
    res["egress"] = egress_outsourcing(inp, ll)
    res["R4"]["certifier_rows"] = certifier_capacity(inp, ll, res["R1"])
    res["period_bounds"] = per_period_bounds(inp, res)
    cks = checks(inp, ll, res)
    res["checks"] = [{"name": n, "ok": ok, "value": v} for n, ok, v in cks]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "alt_roads.json").write_text(json.dumps(res, indent=2, sort_keys=True))
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
