#!/usr/bin/env python3
"""Other cost bases and placements for PoUW's commitments, priced against the lead on Llama-3.2-1B's linears.

h is honest hashing over stock in W1 units a stock MAC, at SHA-512's H32 floor and at the GPU bar. For each design the
script gives h, the card's extra time, T2 (sync and async stock, judged on the larger), the coins in and the bytes out a
token, and the egress term at caps of 32 and 320 bits a token (n_max · W_mac · E / stock):

- the lead: every k32 word hashed on the card;
- R1: every word and operand shipped to the network certifier, which hashes them;
- the split: the card hashes a share f of the words and ships the rest, at the least f that keeps T2 <= 100, and at the
  f where the card's hashing and the link take equal time;
- truncated words (k bits of each word shipped), priced only under a compute-floor premise: under DistinctWrites a
  k-bit field credits k/32 of a word (packing), so the shipped bytes a credited MAC don't fall;
- a card-side non-linear sketch at 2, 4 or 6 registers a word, if a premise credits it like ShaWrites;
- outputs-only with dense binary noise on the activations, (A + R)·Bᵀ served noisy, which needs public, third-party
  weights (the prover chooses only the tokens), and the same with its outputs shipped.

The checks are exact integer arithmetic on small seeded instances: the routes an adversary has to R·Bᵀ at W1's prices
(Four Russians tables, b1 bit-planes, 2:4 sparse halves, per-tile plane counts on synthetic weights), the low-bits
identity behind truncation, the fold of a linear sketch keyed after the words, and the calculator against its inputs.

  uv run python benchmarks/pouw/theory/hash_alt.py [--json PATH] [--mutations] [--verdict PATH]
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np

from exact_kit.checks import Checks

HERE = Path(__file__).resolve().parent
INP = json.loads((HERE / "hash_alt_inputs.json").read_text())
T2_MAX = 100.0


def peak() -> float:
    h = INP["h200"]
    return h["tensor_units_per_sm_clk"] * h["sms"] * h["tensor_clk_hz"]


def llama() -> dict:
    c = INP["llama_3_2_1b"]
    lin = c["linears"]
    return {"macs": c["layers"] * sum(k * n for k, n in lin), "outs": c["layers"] * sum(n for _, n in lin),
            "a_elems": c["layers"] * sum(k for k, _ in lin), "weights_bytes": c["layers"] * sum(k * n for k, n in lin),
            "attn_macs_per_ctx": c["layers"] * c["heads"] * c["head_dim"] * 2,
            "kv_bytes_per_ctx": c["layers"] * 2 * c["kv_heads"] * c["head_dim"] * c["kv_bytes_per_value"],
            "max_n": max(n for _, n in lin), "linears": lin, "layers": c["layers"]}


LL = llama()
LEAD = INP["lead"]
B_LEAD = 4.0 * LEAD["words_per_mac"]


def points() -> dict:
    """Each point's rows, stock steps (sync, async) and GEMM time; decode's from the HBM roofline (an estimate)."""
    hbm = INP["h200"]["hbm_bytes_per_s"] * INP["h200"]["hbm_achieved_share"]
    out = {}
    for key, p in INP["points"].items():
        q = dict(p)
        m = p["m"]
        if p.get("decode"):
            t = (LL["weights_bytes"] + LL["kv_bytes_per_ctx"] * p["attn_ctx"] * m) / hbm
            q["t_stock_s"] = [t, t]
            q["t_gemm_s"] = max(LL["macs"] * m / peak(), LL["weights_bytes"] / hbm)
        out[key] = q
    return out


PTS = points()


def hash_part(key: str) -> float:
    """The part of the lead's extra time that scales with the card's hashed bytes: measured at the three points (both
    engines give it), and at decode the 64-row measurement scaled by rows (an estimate)."""
    p = PTS[key]
    if key in LEAD["t2_measured"]:
        ex = max((t2 - 1.0) * t for t2, t in zip(LEAD["t2_measured"][key], p["t_stock_s"]))
        return ex - (LEAD["alpha"] - 1.0) * p["t_gemm_s"] - LEAD["coin_latency_s"]
    return hash_part("t64") * p["m"] / 64.0


def lead_extra(key: str) -> float:
    """The lead's honest time beyond stock: its extra lanes, its coin round trips and its hashing."""
    return (LEAD["alpha"] - 1.0) * PTS[key]["t_gemm_s"] + LEAD["coin_latency_s"] + hash_part(key)


def export_s(byts: float) -> float:
    return byts * INP["sha512"]["export_units_per_word"] / 4.0 / peak()


def t2(key: str, extra: float, link_s: float = 0.0) -> list[float]:
    """T2 on each engine's stock step, the link overlapped with the card's step."""
    return [max(t + extra, link_s) / t for t in PTS[key]["t_stock_s"]]


def h_of(bytes_per_mac: float) -> dict:
    s = INP["sha512"]
    return {"floor": bytes_per_mac * s["floor_units_per_byte"], "bar": bytes_per_mac * s["bar_units_per_byte"]}


def w_mac_lead_card() -> float:
    return LEAD["alpha"] + LEAD["words_per_mac"] * LEAD["card_hash_registers_per_compression"]


def egress_lead_layout(w_mac: float) -> list[float]:
    """The lead's coin layout carries the same MACs a bit, so its term scales with the priced work a MAC."""
    return [x * w_mac / w_mac_lead_card() for x in LEAD["network_term_at_card_price"]]


def egress_per_element(n_max: float, w_mac: float) -> list[float]:
    return [n_max * w_mac * cap / LL["macs"] for cap in INP["egress"]["caps_bits_per_token"]]


def row(key: str, name: str, h: dict, extra: float, link_bytes_tok: float, coins_tok: float, egress: list[float],
        **more) -> dict:
    p = PTS[key]
    link_s = link_bytes_tok * p["m"] / INP["links"]["port_bytes_per_s"]
    t = t2(key, extra, link_s)
    return {"design": name, "point": key, "label": p["label"], "h_floor": h["floor"], "h_bar": h["bar"],
            "extra_s": extra, "link_s": link_s, "t2_sync": t[0], "t2_async": t[1], "t2_judged": max(t),
            "passes_t2": max(t) <= T2_MAX, "coins_in_bytes_per_token": coins_tok,
            "out_bytes_per_token": link_bytes_tok, "port_share_at_stock": link_bytes_tok * p["m"] / min(p["t_stock_s"])
            / INP["links"]["port_bytes_per_s"], "egress_cap32": egress[0], "egress_cap320": egress[1], **more}


def lead_row(key: str) -> dict:
    h = h_of(B_LEAD)
    return row(key, "lead", h, lead_extra(key), 0.0, LEAD["coin_stream_bytes_per_token"],
               egress_lead_layout(LEAD["alpha"] + h["floor"]))


def ship_extra(key: str, f_card: float, ship_bytes_tok: float) -> float:
    p = PTS[key]
    slow = (INP["links"]["gemm_slowdown_under_d2h"] - 1.0) * LEAD["alpha"] * p["t_gemm_s"] if ship_bytes_tok else 0.0
    return ((LEAD["alpha"] - 1.0) * p["t_gemm_s"] + LEAD["coin_latency_s"] + f_card * hash_part(key)
            + export_s(ship_bytes_tok * p["m"]) + slow)


def shipped(f: float, shipped_bytes_per_mac: float) -> float:
    """Bytes a token that leave the card when it hashes a share f of the words: the rest of the words and the operands."""
    return (1.0 - f) * shipped_bytes_per_mac * LL["macs"] + (LL["a_elems"] if f < 1.0 else 0.0)


def split_row(key: str, f: float, name: str, shipped_bytes_per_mac: float = B_LEAD) -> dict:
    """The card hashes a share f of the words; the rest leave as shipped_bytes_per_mac, with the operands. An exchanged
    bit replaces a card-hashed block's work when f > 0; at f = 0 an outsourced result returns as shipped words."""
    ship = shipped(f, shipped_bytes_per_mac)
    h = h_of(f * B_LEAD)
    if f > 0:
        eg = egress_lead_layout(LEAD["alpha"] + h_of(B_LEAD)["floor"])
    else:
        eg = egress_per_element(1.0 / (8.0 * shipped_bytes_per_mac), LEAD["alpha"])
    extra = ship_extra(key, f, ship)
    p = PTS[key]
    step_s = max(min(p["t_stock_s"]) + extra, ship * p["m"] / INP["links"]["port_bytes_per_s"])
    return row(key, name, h, extra, ship, LEAD["coin_stream_bytes_per_token"], eg, f_card=f,
               certifier_cores_per_gpu=ship * p["m"] / step_s / INP["links"]["certifier_core_bytes_per_s"])


def row_t2_tmp(key: str, f: float, ship: float) -> float:
    link_s = ship * PTS[key]["m"] / INP["links"]["port_bytes_per_s"]
    return max(t2(key, ship_extra(key, f, ship), link_s))


def f_least(key: str, shipped_bytes_per_mac: float = B_LEAD) -> float:
    """The least card share f with T2 <= 100 (T2 rises as f falls once the link binds)."""
    def ok(f):
        return row_t2_tmp(key, f, shipped(f, shipped_bytes_per_mac)) <= T2_MAX
    if ok(0.0):
        return 0.0
    if not ok(1.0):
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi


def f_balanced(key: str) -> float:
    """The share where the card's hashing time equals the link's: the least T2 the split reaches."""
    p = PTS[key]
    link_all = (B_LEAD * LL["macs"] + LL["a_elems"]) * p["m"] / INP["links"]["port_bytes_per_s"]
    e = hash_part(key)
    return link_all / (link_all + e)


def sketch_row(key: str, regs: float) -> dict:
    h = {"floor": regs * INP["w1"]["h32_units_per_write"] * LEAD["words_per_mac"]}
    h["bar"] = h["floor"]
    extra = ((LEAD["alpha"] - 1.0) * PTS[key]["t_gemm_s"] + LEAD["coin_latency_s"]
             + hash_part(key) * h["floor"] / h_of(B_LEAD)["floor"])
    sketch_bytes_tok = 8.0 * LL["outs"] / (min(PTS[key]["m"], 64) * 8)
    return row(key, f"sketch-{regs:g}r", h, extra, sketch_bytes_tok, LEAD["coin_stream_bytes_per_token"],
               egress_lead_layout(LEAD["alpha"] + h["floor"]))


def dout_copies(w_mac: float, target: float = 0.01) -> list[int]:
    """Independent noise copies a linear needs so one coin bit carries at most n_cap MACs (the cap-320 term <= target)."""
    cap = INP["egress"]["caps_bits_per_token"][1]
    n_cap = target * LL["macs"] / (cap * w_mac)
    return [math.ceil(n / n_cap) for _, n in LL["linears"]]


def dout_terms(price: str = "floor_units_per_byte") -> dict:
    s = INP["sha512"]
    out_bpm = 4.0 * LL["outs"] / LL["macs"]
    h = out_bpm * s[price]
    w = INP["w1"]["h32_units_per_write"]
    form1 = (w / 4.0) * LL["a_elems"] / LL["macs"]
    copies = dout_copies(1.0 + form1 + h)
    a_cop = LL["layers"] * sum(k * c for (k, _), c in zip(LL["linears"], copies))
    form = (w / 4.0) * a_cop / LL["macs"]
    w_mac = 1.0 + form + h
    n_max = max(n / c for (_, n), c in zip(LL["linears"], copies))
    return {"h": h, "forming_t1": form, "copies": copies, "coin_bits_per_token": a_cop, "n_max": n_max,
            "w_mac": w_mac, "out_bytes_per_mac": out_bpm}


def dout_row(key: str, ship: bool) -> dict:
    d, db = dout_terms(), dout_terms("bar_units_per_byte")
    p = PTS[key]
    form_s = d["forming_t1"] * LL["macs"] * p["m"] / peak()
    out_tok = d["out_bytes_per_mac"] * LL["macs"]
    if ship:
        h = {"floor": 0.0, "bar": 0.0}
        extra = LEAD["coin_latency_s"] + form_s + export_s(out_tok * p["m"]) + (
            INP["links"]["gemm_slowdown_under_d2h"] - 1.0) * p["t_gemm_s"]
        link = out_tok
        eg = egress_per_element(d["n_max"], 1.0 + d["forming_t1"])
    else:
        h = {"floor": d["h"], "bar": db["h"]}
        extra = LEAD["coin_latency_s"] + form_s + hash_part(key) * d["out_bytes_per_mac"] / B_LEAD
        link = 0.0
        eg = egress_per_element(d["n_max"], d["w_mac"])
    attn = LL["attn_macs_per_ctx"] * p["attn_ctx"]
    share = attn / (attn + LL["macs"])
    h_whole = (h["floor"] * LL["macs"] + h_of(B_LEAD)["floor"] * attn) / (LL["macs"] + attn)
    return row(key, "dout-ship" if ship else "dout", h, extra, link, d["coin_bits_per_token"] / 8.0, eg,
               t1_forming=d["forming_t1"], attn_mac_share=share, h_floor_whole_step_attn_at_lead=h_whole,
               n_max=d["n_max"], noise_copies=d["copies"])


def cost_bases() -> list[dict]:
    """Each base: the credit a committed 32-bit word can carry, the bound it needs, its rank, and h if it held."""
    w1 = INP["w1"]
    hb = peak() / INP["h200"]["hbm_bytes_per_s"]
    s_fast = 52428800 + INP["h200"]["sms"] * (232448 + 262144)
    hk = 1.0 / (2.0 * math.sqrt(2.0 * s_fast))
    gemm_hbm = {}
    for key in ("t64", "t512", "t8192"):
        p = PTS[key]
        byts = LL["weights_bytes"] + p["m"] * (LL["a_elems"] + 2 * LL["outs"])
        gemm_hbm[key] = byts / p["t_gemm_s"] / INP["h200"]["hbm_bytes_per_s"]
    mma = {t: w1["h32_units_per_write"] * w1["mma_c_words_m16n8"] / (16 * 8 * d) for t, d in w1["mma_depth"].items()}
    lead_h = h_of(B_LEAD)["floor"]
    return [
        {"base": "register writes (DistinctWrites under H32, ShaWrites for the hash)", "credit_units_per_word": 32,
         "bound": "H32 (a register written costs 32), and SHA-512 has no shortcut", "rank": "1 and 4a", "h": lead_h},
        {"base": "MMA issue slots / tensor-core cycles", "units_per_mac_by_type": mma,
         "bound": "the committed value needs its MMAs; in M0 at zero data the words are noise-only products that pack "
                  "into the narrowest MMA, so uncommitted accumulators need a 4b premise (PureLanes, SAK-U(f) and DCH)",
         "rank": "4b for any sparsity", "h": lead_h},
        {"base": "HBM traffic, priced additively at the tensor rate", "units_per_byte": hb,
         "credit_units_per_word_if_sound": 4 * hb, "h_if_sound": lead_h * 32 / (4 * hb),
         "stock_gemm_hbm_share": gemm_hbm,
         "bound": "no overlap of HBM traffic with tensor work (false on the roofline: stock's GEMMs leave the HBM share "
                  "above idle), and in M0 only coin-dependent bytes are forced", "rank": "refuted (estimate)", "h": lead_h},
        {"base": "memory-bandwidth floor (I/O complexity of classical matmul, L2 + shared + registers as fast memory)",
         "forced_bytes_per_mac": hk, "credit_units_per_mac": hk * hb,
         "bound": "Hong-Kung / Irony-Toledo-Tiskin, classical algorithm only", "rank": "3", "h": lead_h},
        {"base": "public third-party weights (a model change: the prover picks only the tokens)",
         "bound": "SAK-UF(B): R·Bᵀ for uniform binary R costs its MACs on the registered B", "rank": "4",
         "h": dout_terms()["h"]},
    ]


# ----------------------------------------------------------------------------------------------------- the checks


def four_russians(R, B, c):
    """R (m x K) binary times B (n x K)ᵀ by c-bit subset tables over each column's K: exact, and its add count."""
    m, K = len(R), len(R[0])
    n = len(B)
    groups = range(0, K, c)
    tables = [[[sum(B[j][g + t] for t in range(min(c, K - g)) if mask >> t & 1) for mask in range(1 << c)]
               for g in groups] for j in range(n)]
    C = [[0] * n for _ in range(m)]
    adds = 0
    for i in range(m):
        keys = [sum(R[i][g + t] << t for t in range(min(c, K - g))) for g in groups]
        for j in range(n):
            C[i][j] = sum(tables[j][gi][key] for gi, key in enumerate(keys))
            adds += len(keys)
    return C, adds


def matmul(R, B):
    return [[sum(a * b for a, b in zip(r, col)) for col in B] for r in R]


def bitplanes(B):
    """Two's-complement planes of int8 B: B = Σ_j w_j B_j, w_7 = -128."""
    return [[[(v & 0xFF) >> j & 1 for v in col] for col in B] for j in range(8)], [1 << j for j in range(7)] + [-128]


def split24(r):
    """Split a binary row into two halves, each with at most 2 ones in every group of 4."""
    a, b = [0] * len(r), [0] * len(r)
    for g in range(0, len(r), 4):
        ones = [g + t for t in range(4) if r[g + t]]
        for t in ones[:2]:
            a[t] = 1
        for t in ones[2:]:
            b[t] = 1
    return a, b


def tile_credit(B: np.ndarray, rows: int = 8) -> dict:
    """Per tile (8 output columns), the cheapest exact route to R·Bᵀ at H32: b1 planes (0.125 a plane), u4 nibbles
    (0.5 each) or s8 (1.0). A tile needs planes 0..p and a sign plane when its values fit [-2^p, 2^p - 1]."""
    n, K = B.shape
    out = {}
    for depth_name, depth in (("b1", 256), ("u4", 64)):
        costs = []
        for j in range(0, n, rows):
            for g in range(0, K, depth):
                t = B[j:j + rows, g:g + depth]
                lo, hi = int(t.min()), int(t.max())
                p = 0
                while not (-(1 << p) <= lo and hi <= (1 << p) - 1):
                    p += 1
                planes = p + 1
                b1 = 0.125 * planes
                u4 = 0.5 if planes <= 4 else 1.0
                costs.append(min(b1, u4, 1.0) if depth_name == "b1" else min(u4, 1.0))
        out[depth_name] = float(np.mean(costs))
    out["best"] = min(out.values())
    return out


_BITLEN = np.array([i.bit_length() for i in range(256)])


def planes_of(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Planes 0..p and a sign plane, for the least p with every value in [-2^p, 2^p - 1] (int8 values)."""
    lo, hi = lo.astype(np.int64), hi.astype(np.int64)
    return np.maximum(_BITLEN[np.clip(-lo - 1, 0, 255)], _BITLEN[np.clip(hi, 0, 255)]) + 1


_PLANES8 = planes_of(np.arange(-128, 128), np.arange(-128, 128)).astype(np.uint8)


SPARSE_HIGH = ((0.25, -8, 7), (0.5, -128, 127))


def sorted_tile_credit(B: np.ndarray, r_bits: int, group: int, cols: int = 8, block: int = 1,
                       split: bool = True) -> float:
    """The cheapest exact R·Bᵀ when the adversary reorders the reduction index once for every `group` tiles of 8
    columns, by their largest magnitude, so outliers share chunks. Per chunk it takes the least of b1 planes, u4
    nibbles (0.5 each, 64 deep, R of up to 4 bits) and s8 (1.0). The b1 route keeps q planes (r_bits · q · 0.125, 256
    deep) and adds the entries that need more as sparse corrections, one predicated add a row of R and noise bit with
    int16 partials two to a register (16 units), r_bits · count / 128 a MAC of the chunk. With `split` it may instead
    write B = B_lo + 2^q·B_hi, B_lo the signed residue in q planes and B_hi on a 2:4-sparse MMA (mma.sp: s4 at 0.25 a
    dense MAC, s8 at 0.5), with the entries past two in a group of 4, or outside the sparse type, as sparse
    corrections; combining the parts is not charged. A reordering of single indices rewrites R's block (m·K bits, 32 to
    a register at 32 units), 1/(8·group) units a MAC; one of whole 32-index blocks is a 4-byte copy, free in W1.
    Group 0 keeps K's order."""
    n, K = B.shape
    span = cols * (group or n // cols)
    tot = 0.0
    for j0 in range(0, n, span):
        blk = B[j0:j0 + span]
        if group:
            key = np.abs(blk).max(axis=0).reshape(-1, block).max(axis=1)
            idx = (np.argsort(key, kind="stable")[:, None] * block + np.arange(block)).ravel()
            blk = blk[:, idx]
        t = blk.reshape(blk.shape[0] // cols, cols, K // 64, 64)
        lo64, hi64 = t.min(axis=(1, 3)), t.max(axis=(1, 3))
        pe = _PLANES8[blk.astype(np.int16) + 128].reshape(blk.shape[0] // cols, cols, K // 256, 256)
        b1 = np.min([r_bits * (0.125 * q + (pe > q).sum(axis=(1, 3)) / 128.0) for q in range(1, 9)], axis=0)
        if split:
            g4 = pe.reshape(*pe.shape[:3], 64, 4)
            vals = blk.astype(np.int16).reshape(g4.shape)
            for q in range(1, 8):
                for hp, lo, hi in SPARSE_HIGH:
                    if r_bits * 0.125 * q + hp >= 1.0 or (hp == 0.25 and r_bits > 4):
                        continue
                    out = g4 > q
                    if lo > -128:
                        half = 1 << (q - 1)
                        b_hi = (vals - (((vals + half) & ((1 << q) - 1)) - half)) >> q
                        bad = out & ((b_hi < lo) | (b_hi > hi))
                        excess = bad.sum(axis=-1) + np.maximum((out & ~bad).sum(axis=-1) - 2, 0)
                    else:
                        excess = np.maximum(out.sum(axis=-1) - 2, 0)
                    b1 = np.minimum(b1, r_bits * (0.125 * q + excess.sum(axis=(1, 3)) / 128.0) + hp)
        u4 = (np.where(planes_of(lo64, hi64) <= 4, 0.5, 1.0).reshape(-1, K // 256, 4).mean(axis=2)
              if r_bits <= 4 else 1.0)
        tot += float(np.minimum(np.minimum(b1, u4), 1.0).sum()) * cols * 256
    return tot / B.size + (1.0 / (cols * group) if group and block == 1 else 0.0)


def spread_order(B: np.ndarray, planes: int = 5) -> np.ndarray:
    """One permutation of K for the whole matrix that deals the indices with the most entries past `planes` planes
    round-robin over the groups of 4, so a 2:4 high part fits."""
    heavy = (_PLANES8[B.astype(np.int16) + 128] > planes).sum(axis=0)
    rank = np.argsort(-heavy, kind="stable")
    K = B.shape[1]
    pos = (np.arange(K) % (K // 4)) * 4 + np.arange(K) // (K // 4)
    order = np.empty(K, dtype=np.int64)
    order[pos] = rank
    return order


def gather_tile_credit(B: np.ndarray, r_bits: int, group: int, s: int, cols: int = 8) -> float:
    """The adversary takes, for every `group` tiles of 8 columns, the s indices of K with the largest magnitude, does
    them densely on s8 (s/K a MAC) from R's columns gathered once (m·s bits, s/(8·group·K) a MAC), and the rest in K's
    order with those indices zeroed, by the per-tile routes."""
    n, K = B.shape
    span = cols * group
    Bg = B.reshape(n // span, span, K).copy()
    top = np.argpartition(-np.abs(Bg).max(axis=1), s - 1, axis=1)[:, :s]
    Bg[np.arange(n // span)[:, None], :, top] = 0
    return sorted_tile_credit(Bg.reshape(n, K), r_bits, 0, split=False) + s / K + s / (span * K)


def best_tile_credit(B: np.ndarray, r_bits: int, split: bool = True, gather: bool = True) -> dict:
    groups = [1 << i for i in range(int(math.log2(B.shape[0] // 8)) + 1)]
    by = {"none": sorted_tile_credit(B, r_bits, 0, split=split),
          "blocks32": sorted_tile_credit(B, r_bits, 1, block=32, split=split),
          **{f"single g{g}": sorted_tile_credit(B, r_bits, g, split=split) for g in groups}}
    if split:
        by["spread"] = sorted_tile_credit(B[:, spread_order(B)], r_bits, 0) + 1.0 / B.shape[0]
    if gather:
        for g in (g for g in groups if B.shape[0] % (8 * g) == 0):
            for s in (32, 64, 128, 256):
                by[f"gather g{g} s{s}"] = gather_tile_credit(B, r_bits, g, s)
    route = min(by, key=by.get)
    return {"credit": by[route], "route": route,
            "free_reorder_credit": sorted_tile_credit(B, r_bits, 1, split=split) - 1.0 / 8}


def synth_weights(rng: np.random.Generator, n: int, K: int, kind: str) -> np.ndarray:
    x = rng.standard_normal((n, K)) if kind == "gauss" else rng.standard_t(3, (n, K))
    s = np.abs(x).max(axis=1, keepdims=True) / 127.0
    return np.clip(np.rint(x / s), -127, 127).astype(np.int64)


def run_checks(ck: Checks, res: dict) -> None:
    rng = random.Random(20261029)
    # The calculator against its inputs.
    h = h_of(B_LEAD)
    ck.check("lead W_mac at the floor and the bar is 119.5 / 232.7 (alpha + h)",
             abs(LEAD["alpha"] + h["floor"] - 119.5) < 0.1 and abs(LEAD["alpha"] + h["bar"] - 232.7) < 0.1,
             f"{LEAD['alpha'] + h['floor']:.2f} / {LEAD['alpha'] + h['bar']:.2f}")
    lead = {r["point"]: r for r in res["rows"] if r["design"] == "lead"}
    ck.check("lead T2 reproduces its measured T2 at the three points",
             all(abs(lead[k]["t2_sync"] - LEAD["t2_measured"][k][0]) < 0.06
                 and abs(lead[k]["t2_async"] - LEAD["t2_measured"][k][1]) < 0.06 for k in LEAD["t2_measured"]),
             ", ".join(f"{k} {lead[k]['t2_sync']:.2f}/{lead[k]['t2_async']:.2f}" for k in LEAD["t2_measured"]))
    r1 = {r["point"]: r for r in res["rows"] if r["design"] == "r1"}
    want = {"t64": 97.5, "t512": 569.0, "t8192": 665.0}
    ck.check("R1's judged T2 at 400 Gb/s is 97.5 / 569 / 665 within 1%",
             all(abs(r1[k]["t2_judged"] / v - 1) < 0.01 for k, v in want.items()),
             ", ".join(f"{k} {r1[k]['t2_judged']:.1f}" for k in want))
    sp = [r for r in res["rows"] if r["design"] == "split-t2"]
    ck.check("split-t2 rows meet T2 <= 100, and 0.001 less card share breaks it", all(
        r["t2_judged"] <= T2_MAX + 1e-6 and (r["f_card"] == 0 or row_t2_tmp(
            r["point"], r["f_card"] - 1e-3, shipped(r["f_card"] - 1e-3, B_LEAD)) > T2_MAX) for r in sp),
        ", ".join(f"{r['point']} f {r['f_card']:.3f}" for r in sp))
    d = dout_terms()
    ck.check("D-OUT's cap-320 egress term is at most 1% after its noise copies",
             d["n_max"] * d["w_mac"] * 320 / LL["macs"] <= 0.01 + 1e-12,
             f"n_max {d['n_max']:.0f}, copies {d['copies']}, term {d['n_max'] * d['w_mac'] * 320 / LL['macs']:.4%}")

    # Four Russians on R·Bᵀ: exact, and dearer than the tensor route at H32 for every c < 32.
    m, K, n = 6, 48, 5
    R = [[rng.randrange(2) for _ in range(K)] for _ in range(m)]
    B = [[rng.randrange(-127, 128) for _ in range(K)] for _ in range(n)]
    ok = True
    for c in (2, 4, 8):
        C, adds = four_russians(R, B, c)
        ok &= C == matmul(R, B) and adds == m * n * math.ceil(K / c)
    ck.check("Four Russians tables give R·Bᵀ exactly with K/c adds an output", ok, "c 2, 4, 8")
    fr = {}
    for key in ("t64", "t512", "t8192"):
        mm = PTS[key]["m"]
        fr[key] = {c: 32.0 / c + 32.0 * (1 << c) / (c * mm) for c in (4, 8, 12, 16)}
    res["four_russians_units_per_mac"] = fr
    res["four_russians_table_bytes_c8"] = LL["weights_bytes"] / 8 * 256 * 2
    ck.check("Four Russians costs >= 2 units a MAC at H32 for c <= 16 at every point (tensor: 1)",
             all(v >= 2.0 for p in fr.values() for v in p.values()),
             ", ".join(f"{k} c8 {v[8]:.3f}" for k, v in fr.items()))

    # b1 bit-planes: exact; 8 planes at 0.125 a plane plus the combine.
    planes, wts = bitplanes(B)
    via = [[sum(w * x for w, x in zip(wts, (matmul(R, planes[j])[i][jj] for j in range(8)))) for jj in range(n)]
           for i in range(m)]
    ck.check("R·Bᵀ = Σ_j w_j R·B_jᵀ over two's-complement planes, exactly", via == matmul(R, B), None)
    b1_cost = 8 * 32 * 128 / (16 * 8 * 256) + 8 * 32 / 2048
    ck.check("b1 planes cost 1.125 units a MAC at K 2,048 (ties the tensor route only as K grows)",
             abs(b1_cost - 1.125) < 1e-12, f"{b1_cost}")

    # 2:4 halves: any binary row splits into two 2:4 rows; at density 1/2 the second half is never skippable.
    ok = True
    for _ in range(50):
        r = [rng.randrange(2) for _ in range(64)]
        a, b = split24(r)
        ok &= [x + y for x, y in zip(a, b)] == r and all(sum(a[g:g + 4]) <= 2 and sum(b[g:g + 4]) <= 2
                                                         for g in range(0, 64, 4))
    ck.check("any binary row is the sum of two 2:4-sparse rows", ok, None)
    def skip(p):
        q = sum(math.comb(4, t) * p ** t * (1 - p) ** (4 - t) for t in (3, 4))
        return (1 - q) ** (16 * 16)
    res["sp24_skip_prob"] = {"p_half": skip(0.5), "p_1_32": skip(1 / 32)}
    ck.check("at density 1/2 a 16x64 tile's second half is zero with p < 1e-40 (credit 1.0); at 1/32 with p > 0.9 "
             "(credit about 0.5)", skip(0.5) < 1e-40 and skip(1 / 32) > 0.9,
             f"{skip(0.5):.2e} / {skip(1 / 32):.3f}")

    # Low k bits of a running word need only k-bit operands; a register holds 32/k such fields.
    ok = True
    for k in (1, 4, 8):
        for _ in range(20):
            a = [rng.randrange(-128, 128) for _ in range(64)]
            b = [rng.randrange(-128, 128) for _ in range(64)]
            ok &= sum(x * y for x, y in zip(a, b)) % (1 << k) == sum((x % (1 << k)) * (y % (1 << k))
                                                                      for x, y in zip(a, b)) % (1 << k)
    ck.check("low k bits of Σ a·b equal those of Σ (a mod 2^k)(b mod 2^k), k 1, 4, 8", ok, None)
    nar = {k: (0.125 if k == 1 else 0.5 if k <= 4 else 1.0) for k in (1, 4, 8)}
    res["low_bits_units_per_mac"] = nar
    res["truncated_bytes_per_credited_mac"] = {k: (k / 8.0) / (32.0 * k / 32.0) for k in (4, 8, 16, 32)}
    ck.check("under DistinctWrites a k-bit field credits k units, so shipped bytes a credited MAC stay 1/8 at every k",
             all(abs(v - 0.125) < 1e-12 for v in res["truncated_bytes_per_credited_mac"].values()), None)

    # A linear sketch keyed after the words folds onto the step products.
    L = 64
    kap = [rng.randrange(-1000, 1000) for _ in range(L)]
    r = [rng.randrange(1 << 20) for _ in range(L)]
    w = [sum(kap[:t + 1]) for t in range(L)]
    rho = [sum(r[t:]) for t in range(L)]
    ck.check("Σ r_t w_t = Σ ρ_τ κ_τ with ρ the suffix sums of r (the late key still folds)",
             sum(x * y for x, y in zip(r, w)) == sum(x * y for x, y in zip(rho, kap)), None)

    # Outputs-only with noise on A: free at zero weights (M0), the product itself on fixed weights.
    Z = [[0] * K for _ in range(n)]
    A = [[rng.randrange(-127, 127) for _ in range(K)] for _ in range(m)]
    AR = [[x + y for x, y in zip(ra, rr)] for ra, rr in zip(A, R)]
    ck.check("(A + R)·Bᵀ is 0 at B = 0 for every R (M0 with prover weights: no credit)",
             all(v == 0 for rr in matmul(AR, Z) for v in rr), None)
    ck.check("(A + R)·Bᵀ = A·Bᵀ + R·Bᵀ (at zero data the forced work is R·Bᵀ)",
             matmul(AR, B) == [[x + y for x, y in zip(p, q)] for p, q in zip(matmul(A, B), matmul(R, B))], None)

    # Per-tile plane counts on synthetic per-channel int8 weights: the falsification test's prototype.
    nrng = np.random.default_rng(20261029)
    tc = {}
    for kind in ("gauss", "t3"):
        tc[kind] = {f"{n_}x{k_}": tile_credit(synth_weights(nrng, n_, k_, kind)) for n_, k_ in ((256, 2048), (256, 8192))}
    res["tile_credit_synthetic"] = tc
    ck.check("tile credit on Gaussian weights: the cheapest exact R·Bᵀ costs >= 0.99 a MAC",
             all(v["best"] >= 0.99 for v in tc["gauss"].values()),
             ", ".join(f"{k} {v['best']:.3f}" for k, v in tc["gauss"].items()))
    res["tile_credit_t3_best"] = {k: v["best"] for k, v in tc["t3"].items()}
    down = LL["linears"][3]
    res["tile_gap_t3_down_mac_weighted"] = (1.0 - tc["t3"]["256x8192"]["best"]) * down[0] * down[1] * LL["layers"] / LL["macs"]
    ck.check("on Student-t(3) weights at K 8,192 b1 planes undercut the tensor route (SAK-UF falsified there)",
             tc["t3"]["256x8192"]["best"] < 0.99,
             ", ".join(f"{k} {v['best']:.3f}" for k, v in tc["t3"].items())
             + f"; down's gap MAC-weighted {res['tile_gap_t3_down_mac_weighted']:.3%}")
    srt = {}
    for kind in ("gauss", "t3"):
        for n_, k_ in ((256, 2048), (256, 8192)):
            Bw = synth_weights(nrng, n_, k_, kind)
            srt[f"{kind} {n_}x{k_}"] = {f"r{b}": best_tile_credit(Bw, b) for b in (1, 2)}
    v = np.arange(-127, 128, dtype=np.int16)
    split_ok = True
    for q in range(1, 8):
        half = 1 << (q - 1)
        b_lo = ((v + half) & ((1 << q) - 1)) - half
        b_hi = (v - b_lo) >> q
        split_ok &= bool(np.all(b_lo + (b_hi << q) == v) and np.all((-half <= b_lo) & (b_lo < half))
                         and np.all((b_hi != 0) == (_PLANES8[v + 128] > q)))
    ck.check("the weight split is exact: v = lo + 2^q·hi with lo in q planes, hi nonzero exactly where v needs more",
             split_ok, None)
    res["sorted_tile_credit"] = srt
    macs_w = [k * n for k, n in LL["linears"]]
    gap = {}
    for kind in ("gauss", "t3"):
        for b in (1, 2):
            for field in ("credit", "free_reorder_credit"):
                per = [srt[f"{kind} 256x{2048 if k == 2048 else 8192}"][f"r{b}"][field] for k, _ in LL["linears"]]
                gap[f"{kind} r{b} {field}"] = max(0.0, sum((1 - c) * w for c, w in zip(per, macs_w)) / sum(macs_w))
    res["sorted_gap_mac_weighted"] = gap
    detail = ", ".join(f"{k} {v:.2%}" for k, v in gap.items())
    ck.check("with the reordering priced, b1 planes undercut 1-bit noise by > 1% on Student-t(3) weights", 
             gap["t3 r1 credit"] > 0.01, detail)
    ck.check("with 2-bit noise the priced routes stay within 1% on Gaussian weights",
             gap["gauss r2 credit"] < 0.01, ", ".join(f"{k} {v}" for k, v in srt.items()))

    if ck.mutations:
        with ck.mutant("Four Russians table off by one bit"):
            C, _ = four_russians(R, [[v + (1 if t == 0 else 0) for t, v in enumerate(col)] for col in B], 4)
            ck.check("tables exact", C == matmul(R, B))
        with ck.mutant("bit-plane sign weight +128"):
            wts2 = wts[:7] + [128]
            via2 = [[sum(w_ * x for w_, x in zip(wts2, (matmul(R, planes[j])[i][jj] for j in range(8))))
                     for jj in range(n)] for i in range(m)]
            ck.check("planes exact", via2 == matmul(R, B))
        with ck.mutant("fold with prefix sums for ρ"):
            rho2 = [sum(r[:t + 1]) for t in range(L)]
            ck.check("fold", sum(x * y for x, y in zip(r, w)) == sum(x * y for x, y in zip(rho2, kap)))
        with ck.mutant("low bits from operands mod 2^(k-1)"):
            pairs = [([rng.randrange(-128, 128) for _ in range(64)], [rng.randrange(-128, 128) for _ in range(64)])
                     for _ in range(20)]
            ck.check("low bits", all(sum(x * y for x, y in zip(a, b)) % 16 == sum((x % 8) * (y % 8)
                                                                                   for x, y in zip(a, b)) % 16
                                     for a, b in pairs))


def build() -> dict:
    rows = []
    for key in PTS:
        rows.append(lead_row(key))
        rows.append(split_row(key, 0.0, "r1"))
        rows.append(split_row(key, f_least(key), "split-t2"))
        rows.append(split_row(key, f_balanced(key), "split-balanced"))
        for k in (8, 4):
            bpm = B_LEAD * k / 32.0
            rows.append(split_row(key, f_least(key, bpm), f"trunc{k}-t2 (compute-floor premise)", bpm))
        for regs in (2, 4, 6):
            rows.append(sketch_row(key, regs))
        rows.append(dout_row(key, False))
        rows.append(dout_row(key, True))
    return {"peak_units_per_s": peak(), "llama": {k: v for k, v in LL.items() if k != "linears"},
            "points": {k: {"m": p["m"], "t_stock_s": p["t_stock_s"], "t_gemm_s": p["t_gemm_s"],
                           "lead_extra_s": lead_extra(k)} for k, p in PTS.items()},
            "cost_bases": cost_bases(), "dout": dout_terms(), "dout_bar": dout_terms("bar_units_per_byte"),
            "rows": rows}


def per_period(res: dict) -> list[dict]:
    """Pooled honest hashing (W1 units and AND gates) a period, at one H200, the datacenter and the world."""
    a = INP["accounting"]
    designs = {"lead": h_of(B_LEAD)["floor"], "sketch-4r": 4 * 32 * LEAD["words_per_mac"],
               "dout (linears)": dout_terms()["h"]}
    for key in ("t8192", "dec64_8k"):
        r = next(x for x in res["rows"] if x["design"] == "dout" and x["point"] == key)
        designs[f"dout + attention at the lead's density, {key}"] = r["h_floor_whole_step_attn_at_lead"]
        s = next(x for x in res["rows"] if x["design"] == "split-t2" and x["point"] == key)
        designs[f"split-t2, {key}"] = s["h_floor"]
    out = []
    for scope, macs_s in (("one H200", a["h200_macs_per_s"]), ("datacenter", a["h200_macs_per_s"] * a["dc_gpus"]),
                          ("world", a["world_macs_per_year"] / a["year_s"])):
        for period in ("window_s", "week_s", "month_s", "year_s"):
            macs = macs_s * a[period]
            out.append({"scope": scope, "period": period[:-2], "stock_macs": macs, "t1_allows_units": 0.01 * macs,
                        **{f"hash_units {d}": h * macs for d, h in designs.items()},
                        **{f"hash_and {d}": h * macs * a["and_per_int8_mac"] for d, h in designs.items()}})
    return out


def table(res: dict) -> str:
    lines = ["design | point | h floor / bar | T2 sync / async | out B/tok | coins in B/tok | egress 32 / 320 | f"]
    for r in res["rows"]:
        lines.append(f"{r['design']} | {r['point']} | {r['h_floor']:.2f} / {r['h_bar']:.2f} | {r['t2_sync']:.1f} / "
                     f"{r['t2_async']:.1f} | {r['out_bytes_per_token']:.3g} | {r['coins_in_bytes_per_token']:.3g} | "
                     f"{r['egress_cap32']:.3%} / {r['egress_cap320']:.3%} | {r.get('f_card', '')}")
    return "\n".join(lines)


def main() -> int:
    ck = Checks()
    res = build()
    run_checks(ck, res)
    res["per_period"] = per_period(res)
    print(table(res))
    if "--json" in ck.args:
        Path(ck.args[ck.args.index("--json") + 1]).write_text(json.dumps(res, indent=1) + "\n")
    return ck.finish()


if __name__ == "__main__":
    sys.exit(main())
