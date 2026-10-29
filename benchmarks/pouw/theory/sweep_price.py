#!/usr/bin/env python3
"""The PoUW sweep's calculator: a design, (noise and de-noise) × (what is committed and how densely) × (the premise set),
priced under CORE.md §2a on the H200, for Llama-3.2-1B int8 at T 512 (64 real tokens) and T 8,192 and for a
DeepSeek-V4-like shape, and the sweep's ranking.

T1 is the additive gap cost(H) − min cost(adv) as a share of stock's m·n·K, in W1 units (an int8 MAC one unit, a register
written `reg` = 32 under H32, measured on the H200). The honest program is the best possible, so every register it writes
costs `reg`. Under DistinctWrites each committed, pairwise-distinct, noise-dependent word costs the adversary at least
`reg`. The gap is honest's writes that no premise forces, by term:

- final word: each dot product's last word is A·Bᵀ, free advice until the final-word lemma (32/K at L 32);
- uncommitted steps: r·(1 − reg/L) between words committed every L, unless the k32 semantics make a word force its L/32
  steps; then what the best-known fast scheme (Strassen-Winograd, `price.fast_matmul`) saves on them;
- forming: the formed operand entries at reg/4 each, unless committed under `commit-forced` or charged by the counting
  lemma;
- noise derivation: the noise entries derived from XOF bytes (one LOP3 a packed register, CORE-SMELL C28), the same way;
- de-noise: the design's correction work, unless forced;
- pre-noise hashing: bytes hashed before the noise (A's commitment feeds the call's seed) are free advice until the bound
  charges the atomic call's own work (the final-word lemma); so are rows holding only final words (`final_rows`, C28);
- hashing slack: credited hashing under ShaSpeed, δ times its work (δ = 0 as §2a states it). δ* is the precision
  ShaSpeed needs for T1 to hold (CORE-SMELL C27).

T2 is t_H / t_S: the stock step, plus NCP's GEMM parts (GD-DENOISE's measured stand-ins, scaled by the design's counts),
plus the hashing at the measured committer's rate; it is judged on the faster (async) stock step. T3 is the served
format's measured perplexity rise. The rank is the strongest premise the matmul argument needs; ShaSpeed's rank 4 is shown
beside it, since every design that credits hashing needs it.

A design whose card entry leaves a count missing or "?" is refused and never ranked, so an unpriced term can't top the
table. Every number carries its card source: measured (a run id), estimate, or arithmetic.

The ranking, at each point: designs that meet T1 first (T1 is hard), then by the brief's order, assumption rank, T1, T2,
T3. Open premises don't move a row; they are what the row still needs.

  uv run python benchmarks/pouw/theory/sweep_price.py [--card PATH] [--out DIR] [--delta D] [--mutations] [--verdict PATH]
"""
from __future__ import annotations

import copy
import csv
import itertools
import json
import math
import os
import sys
import tomllib
from pathlib import Path

from exact_kit.checks import Checks

CARD = Path(__file__).with_name("sweep_price.toml")
TERMS = ("final_word", "uncommitted_steps", "forming", "noise_derivation", "denoise", "pre_noise_hashing", "final_rows",
         "hashing_slack")
NOISE_FIELDS = ("r", "formed_x", "formed_y", "noise_x", "noise_y", "denoise", "pre_noise_bytes", "operand_max", "served",
                "source")
STATUSES = ("proved", "measured", "ruled", "assumed", "scratch", "open", "refuted")
UNSETTLED = ("open", "scratch")
COMMIT_FIELDS = ("L", "word_bits", "operands", "rows_mix_final")
CLASSES = {"formed_x": "forming", "formed_y": "forming", "noise_x": "noise_derivation", "noise_y": "noise_derivation"}
REQUIRED = ("H32", "W1", "ShaSpeed")
TARGET = {"t1": 0.01, "t2": 100.0, "t3": 0.05}
WORD_BYTES = 4
MIN_ROWS = 64
PERIODS = {"window": None, "week": 7 * 86400, "month": 30 * 86400, "year": 365 * 86400}


class Refused(Exception):
    """A design or point the card doesn't price: its reasons, each naming the field."""

    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


def ev(s, **v) -> float:
    if s is None or s == "?":
        raise Refused(["unpriced"])
    if isinstance(s, (int, float)):
        return float(s)
    return float(eval(compile(str(s), "<card>", "eval"), {"__builtins__": {}}, v))


def load(path: Path = CARD) -> dict:
    card = tomllib.loads(path.read_text())
    for key, sw in card.get("sweep", {}).items():
        for L, (tag, ops), count, least in itertools.product(sw["L"], sw["operands"].items(), (False, True),
                                                               (False, True)):
            if (count and {"formed_x", "formed_y"} <= set(ops)) or (least and {"noise_x", "noise_y"} <= set(ops)):
                continue
            prem = list(sw["base"])
            if L > 32:
                prem += sw["steps"].get(str(L), sw["steps"]["above"])
            if ops:
                prem.append("commit-forced")
            prem += (sw["charge"]["count"] if count else []) + (sw["charge"]["min"] if least else [])
            what = ", ".join(ops) if ops else "words only"
            card["design"][f"{key}-k{L}-{tag}" + ("-count" if count else "") + ("-min" if least else "")] = {
                "title": f"{key.upper()}, a word every {L} steps, committed: {what}"
                         + ("; forming by the counting lemma" if count else "")
                         + ("; the derivation by OneInstructionMinimal" if least else ""),
                "noise": sw["noise"], "premises": list(dict.fromkeys(prem)), "source": sw["source"], "swept": True,
                "commit": {"L": L, "word_bits": 32, "operands": list(ops), "rows_mix_final": True}}
    return card


def validate(card: dict, name: str, lenient: bool = False) -> list[str]:
    """Why `name` can't be priced; empty when every term has a count. `lenient` (a mutation only) reads "?" as 0."""
    d = card["design"][name]
    out: list[str] = []
    noise = card.get("noise", {}).get(d.get("noise"))
    if noise is None:
        if not lenient:
            out.append(f"noise {d.get('noise')!r} has no structure in the card: forming, noise derivation, de-noise and "
                       f"pre-noise hashing are unpriced")
    else:
        for f in NOISE_FIELDS:
            if noise.get(f) in (None, "?"):
                if not lenient:
                    out.append(f"noise.{d['noise']}.{f} unpriced")
            elif f not in ("served", "source", "r"):
                try:
                    ev(noise[f], m=64, n=64, K=64)
                except Exception as e:  # noqa: BLE001 - the card's expression is the input under test
                    out.append(f"noise.{d['noise']}.{f} = {noise[f]!r} doesn't evaluate ({e})")
    c = d.get("commit", {})
    for f in COMMIT_FIELDS:
        if c.get(f) in (None, "?") and not lenient:
            out.append(f"commit.{f} unpriced")
    if isinstance(c.get("L"), int) and c["L"] % 32:
        out.append(f"commit.L {c['L']} is not a whole number of k32 steps")
    if isinstance(c.get("word_bits"), int) and c["word_bits"] != 32:
        out.append(f"a {c['word_bits']}-bit word is unpriced: DistinctWrites counts 32-bit registers, and no premise prices "
                   f"a narrower committed word")
    for o in c.get("operands", []):
        if o not in CLASSES:
            out.append(f"commit.operands {o!r} is not one of {sorted(CLASSES)}")
    prem = card.get("premise", {})
    ids = d.get("premises", [])
    for p in ids:
        if p not in prem:
            out.append(f"premise {p!r} is not in the card")
        elif prem[p].get("status") not in STATUSES:
            out.append(f"premise {p!r} has status {prem[p].get('status')!r}, not one of {STATUSES}")
        elif prem[p].get("status") == "refuted":
            out.append(f"premise {p!r} is refuted ({prem[p].get('source', '')})")
        else:
            for q in prem[p].get("requires", []):
                if q not in ids:
                    out.append(f"premise {p!r} requires {q!r}, which the design doesn't take")
    for p in REQUIRED:
        if p not in ids and not lenient:
            out.append(f"premise {p!r} missing: the DistinctWrites price at H32 in W1 with hashing credited needs it")
    if not d.get("source"):
        out.append("design.source missing")
    return out


def effects(card: dict, ids: list[str]) -> tuple[set[str], bool, int]:
    """The terms the premises close, whether committed registers are forced, and the largest L at which a committed
    word forces its L/32 steps (32: only at every k32 word)."""
    prem = card["premise"]
    closes = set().union(*(set(prem[p].get("closes", [])) for p in ids if p in prem))
    forced = any(prem[p].get("forces_committed", False) for p in ids if p in prem)
    steps = max([32] + [int(prem[p].get("word_forces_steps", 32)) for p in ids if p in prem])
    return closes, forced, steps


def calls(card: dict, point: str) -> list[dict]:
    pt = card["point"][point]
    model = card["model"][pt["model"]]
    return [{"name": nm, "n": n, "K": K, "count": cnt, "m": pt["tokens"] * frac}
            for nm, n, K, cnt, frac in model["matmuls"]]


def fast_saving(card: dict, L: int, m: float, n: float, operand_max: float) -> tuple[float, int]:
    """The best-known scheme's saving against the k32 counting, as a share of a segment's mnL tensor units, and its levels.

    A level of a <2,2,2; t> scheme runs t products of half the depth. A product's summed side is split into two s8 parts
    when two operands' range passes the tensor's, so it costs c = Σ_p split^(summed sides of p) / 8 of the tensor units
    (P1's floor). ℓ levels pay c^ℓ (a lower bound: sums of sums only split more), plus each level's output-block
    additions (registers at `reg`) and operand-block additions (entries at reg/4):
    saving_ℓ = 1 − c^ℓ − (out·reg/(4L) + (op/2)·(reg/epr)/4·(1/m + 1/n)) · Σ_{i<ℓ} (t/4)^i, with `op` the
    operand-block additions of both sides, each of a quarter of its operand."""
    f = card["price"]["fast_matmul"]
    reg, epr = card["price"]["reg"]["value"], card["price"]["entries_per_reg"]["value"]
    split = 1 if 2 * operand_max <= f["tensor_max"] else 2
    t, best = len(f["products"]), (0.0, 0)
    c = sum(split ** (a + b) for a, b in f["products"]) / 8
    per_level = f["output_adds"] * reg / (4 * L) + f["operand_adds"] / 2 * (reg / epr) / 4 * (1 / m + 1 / n)
    for lv in range(1, f["max_levels"] + 1):
        if L / 2 ** lv < 32 or min(m, n) / 2 ** lv < 1:
            break
        s = 1 - c ** lv - per_level * sum((t / 4) ** i for i in range(lv))
        best = max(best, (s, lv))
    return best


def call_units(card: dict, name: str, m: float, n: float, K: float, lenient: bool = False) -> dict:
    """One call's T1 terms in W1 units, the credited hashing's bytes and the T2 counts."""
    d = card["design"][name]
    noise = card["noise"].get(d["noise"]) if d.get("noise") in card.get("noise", {}) else None
    c = d["commit"]
    price = card["price"]
    reg, epr, hw = price["reg"]["value"], price["entries_per_reg"]["value"], price["hash_work"]["value"]
    closes, forced, steps = effects(card, d.get("premises", []))

    def get(f, default=0.0):
        if noise is None or noise.get(f) in (None, "?"):
            if lenient:
                return default
            raise Refused([f"noise.{d.get('noise')}.{f} unpriced"])
        return ev(noise[f], m=m, n=n, K=K)

    if noise is None and lenient:
        zero = {"terms": dict.fromkeys(TERMS, 0.0), "stock": m * n * K, "r": 1.0, "mK": m * K, "nK": n * K}
        return zero | dict.fromkeys(("credited_bytes", "hashed_bytes", "word_bytes", "operand_bytes", "pre_noise_bytes",
                                     "coin_bytes", "x_entries", "y_entries", "noise_y"), 0.0)
    r = get("r", 1.0)
    L = c["L"] if isinstance(c.get("L"), int) else (32 if lenient else None)
    if L is None:
        raise Refused(["commit.L unpriced"])
    if (r * K) % L:
        raise Refused([f"{r:g}K = {r * K:g} is not a whole number of words at L {L}"])
    if L >= r * K:
        raise Refused([f"a word every {L} steps commits only the final word of a {r * K:g}-deep dot product, and the "
                       f"final word isn't noise-dependent"])
    stock = m * n * K
    words = r * K / L
    steps = L <= steps
    u_word = reg * L / 32 if steps else reg
    final_free = "final_word" not in closes
    honest_tensor = r * m * n * K
    t = dict.fromkeys(TERMS, 0.0)
    t["final_word"] = m * n * u_word if final_free else 0.0
    t["uncommitted_steps"] = honest_tensor - m * n * words * u_word
    if steps and L > 32:
        t["uncommitted_steps"] += fast_saving(card, L, m, n, get("operand_max"))[0] * honest_tensor
    committed = set(c.get("operands", []))
    ent = {k: get(k) for k in CLASSES}
    for k, term in CLASSES.items():
        if not ((forced and k in committed) or term in closes):
            t[term] += ent[k] * reg / epr
    den = get("denoise")
    if "denoise" not in closes:
        t["denoise"] = den
    pre_b = get("pre_noise_bytes")
    word_b = m * n * words * WORD_BYTES
    op_b = sum(ent[k] for k in committed)
    if "pre_noise_hashing" not in closes:
        t["pre_noise_hashing"] = pre_b * hw
    mix = c.get("rows_mix_final", True)
    if mix is False and "final_rows" not in closes:
        t["final_rows"] = m * n * WORD_BYTES * hw
    coin_b = (ent["noise_x"] + ent["noise_y"]) * price["coin_bytes"]["value"]
    credited_b = word_b + op_b + coin_b + (pre_b if "pre_noise_hashing" in closes else 0.0)
    return {"stock": stock, "terms": t, "credited_bytes": credited_b, "hashed_bytes": word_b + op_b + pre_b,
            "word_bytes": word_b, "operand_bytes": op_b, "pre_noise_bytes": pre_b, "coin_bytes": coin_b, "r": r,
            "x_entries": ent["formed_x"] + ent["noise_x"], "y_entries": ent["formed_y"] + ent["noise_y"],
            "noise_y": ent["noise_y"], "mK": m * K, "nK": n * K}


def t3_of(card: dict, served: str | None, model: str) -> dict:
    e = card.get("t3", {}).get(served or "")
    if not e or ("models" in e and model not in e["models"]):
        return {"ctx2k": None, "ctx8k": None, "kind": "unmeasured", "source": f"no T3 for {served!r} on {model}"}
    return {"ctx2k": e["ctx2k"], "ctx8k": e["ctx8k"], "kind": e["kind"], "source": e["source"]}


def evaluate(card: dict, name: str, point: str, delta: float = 0.0, lenient: bool = False) -> dict:
    """The design at the point: T1's terms, T2, T3, rank and what's open. Raises Refused."""
    reasons = validate(card, name, lenient)
    if reasons:
        raise Refused(reasons)
    d, pt = card["design"][name], card["point"][point]
    cs = calls(card, point)
    small = [f"{x['name']} ({x['m']:g} x {x['n']})" for x in cs if min(x["m"], x["n"]) < MIN_ROWS]
    if small:
        raise Refused([f"outside the bound's domain: {', '.join(small)} below {MIN_ROWS} rows a call, where MH100's floor "
                       f"is shape-blind (CORE-SMELL C15)"])
    agg: dict = {"stock": 0.0, "credited_bytes": 0.0, "hashed_bytes": 0.0, "word_bytes": 0.0, "operand_bytes": 0.0,
                 "pre_noise_bytes": 0.0, "coin_bytes": 0.0, "x_entries": 0.0, "y_entries": 0.0, "noise_y": 0.0, "mK": 0.0, "nK": 0.0,
                 "terms": dict.fromkeys(TERMS, 0.0)}
    per_matmul = {}
    r = None
    for x in cs:
        u = call_units(card, name, x["m"], x["n"], x["K"], lenient)
        r = u["r"]
        w = x["count"]
        for k in agg:
            if k == "terms":
                for tk in TERMS:
                    agg["terms"][tk] += w * u["terms"][tk]
            else:
                agg[k] += w * u[k]
        per_matmul[x["name"]] = {"m": x["m"], "n": x["n"], "K": x["K"], "calls": w,
                                 "t1": sum(u["terms"].values()) / u["stock"],
                                 "terms": {k: v / u["stock"] for k, v in u["terms"].items()}}
    stock = agg["stock"]
    hw = card["price"]["hash_work"]["value"]
    hash_share = agg["credited_bytes"] * hw / stock
    terms = {k: v / stock for k, v in agg["terms"].items()}
    terms["hashing_slack"] = delta * hash_share
    rest = sum(v for k, v in terms.items() if k != "hashing_slack")
    t1 = rest + terms["hashing_slack"]
    delta_star = max(0.0, TARGET["t1"] - rest) / hash_share if hash_share else math.inf
    committer_gap = agg["credited_bytes"] * (card["price"]["hash_work_committer"]["value"] - hw) / stock
    t2 = t2_of(card, point, agg, r)
    noise = card["noise"].get(d["noise"], {})
    t3 = t3_of(card, noise.get("served"), pt["model"])
    prem = card["premise"]
    ids = d["premises"]
    rank = max(prem[p]["rank"] for p in ids if p != "ShaSpeed")
    rank_all = max(prem[p]["rank"] for p in ids)
    open_items = [{"premise": p, "owner": prem[p].get("owner", ""), "status": prem[p]["status"],
                   "rank": prem[p]["rank"], "source": prem[p]["source"]}
                  for p in ids if prem[p]["status"] in UNSETTLED]
    assumed = [{"premise": p, "rank": prem[p]["rank"], "source": prem[p]["source"]}
               for p in ids if prem[p]["status"] == "assumed" and p not in REQUIRED]
    t3w = max(v for v in (t3["ctx2k"], t3["ctx8k"]) if v is not None) if t3["ctx2k"] is not None else None
    return {"design": name, "title": d["title"], "point": point, "t1": t1, "terms": terms, "t1_pass": t1 <= TARGET["t1"],
            "delta": delta, "delta_star": delta_star, "hash_units_per_mac": hash_share,
            "hash_bytes_per_mac": agg["credited_bytes"] / stock, "coin_bytes_per_mac": agg["coin_bytes"] / stock,
            "committer_gap": committer_gap, "t2": t2, "t2_pass": t2["judged"] <= TARGET["t2"],
            "t3": t3, "t3_worst": t3w, "t3_pass": t3w is not None and t3w <= TARGET["t3"], "rank": rank,
            "rank_with_shaspeed": rank_all, "premises": ids, "open": open_items, "assumed": assumed,
            "per_matmul": per_matmul,
            "stock_macs_forward": stock, "commit": d["commit"]}


def t2_of(card: dict, point: str, agg: dict, r: float) -> dict:
    """t_H / t_S a forward: stock, plus NCP's measured GEMM parts scaled to the design's counts, plus hashing."""
    pt = card["point"][point]
    price = card["price"]
    src = card["point"][pt.get("gemm_from", point)]
    parts = src["gemm_parts"]
    rate = pt.get("hash_rate", price["hash_rate"]["value"])
    if pt.get("matmul_only"):
        src_macs = sum(x["count"] * x["m"] * x["n"] * x["K"] for x in calls(card, pt["gemm_from"]))
        plain = agg["stock"] / (src_macs / src["plain_gemm_ms"])
        stock = {"sync": plain, "async": plain}
    else:
        plain = src["plain_gemm_ms"]
        stock = dict(pt["stock_ms"])
    rx = agg["x_entries"] / (4 * agg["mK"])
    ry = agg["y_entries"] / (4 * agg["nK"])
    coins = agg["noise_y"] / 64 / price["coin_rate"]["value"] * 1e3
    extra_parts = {"wide": plain * parts["wide"] * (r - 1) / 2, "a_side": plain * parts["a_side"] * rx,
                   "noised": plain * parts["noised"], "w_side": plain * parts["w_side"] * ry, "f1_coins": coins}
    extra = sum(extra_parts.values())
    hash_ms = agg["hashed_bytes"] / rate * 1e3
    write_ms = (agg["word_bytes"] + agg["operand_bytes"]) / price["hbm_rate"]["value"] * 1e3
    eng = {e: {"stock_ms": s, "t_H_ms": s + extra + hash_ms, "ratio": (s + extra + hash_ms) / s,
               "hash_share": hash_ms / (s + extra + hash_ms)} for e, s in stock.items()}
    return {"engines": eng, "judged": max(v["ratio"] for v in eng.values()),
            "unfused": max((s + extra + hash_ms + write_ms) / s for s in stock.values()), "hbm_write_ms": write_ms,
            "extra_ms": extra, "extra_parts_ms": extra_parts,
            "hash_ms": hash_ms, "plain_gemm_ms": plain, "matmuls_only": (plain + extra + hash_ms) / plain,
            "without_hashing": max((s + extra) / s for s in stock.values()), "bytes_forward": agg["hashed_bytes"],
            "kind": "estimate on measured parts: " + (pt.get("gemm_source", "") if pt.get("matmul_only") else
                                                      f"{pt['stock_kind']} stock; GEMM parts {src['gemm_kind']}") +
                    "; no end-to-end honest program exists"}


def rank_key(row: dict) -> tuple:
    return (not row["t1_pass"], row["rank"], row["t1"], not row["t2_pass"], row["t2"]["judged"], not row["t3_pass"],
            row["t3_worst"] if row["t3_worst"] is not None else math.inf)


def ranking(card: dict, point: str, delta: float = 0.0, lenient: bool = False) -> tuple[list[dict], dict[str, list[str]]]:
    rows, refused = [], {}
    for name in card["design"]:
        try:
            rows.append(evaluate(card, name, point, delta, lenient))
        except Refused as e:
            refused[name] = e.reasons
    return sorted(rows, key=rank_key), refused


def opened(row: dict) -> str:
    return ", ".join(o["premise"] + (" (scratch)" if o["status"] == "scratch" else "") for o in row["open"]) or "-"


def needs(row: dict) -> list[str]:
    out = [f"{o['premise']} ({o['owner']}, rank {o['rank']} once settled; "
           f"{'proved outside the lock: lock it' if o['status'] == 'scratch' else 'open'}): {o['source']}"
           for o in row["open"]]
    out += [f"{a['premise']} (assumed, rank {a['rank']}): {a['source']}" for a in row["assumed"]]
    if not row["t1_pass"]:
        big = max(((k, v) for k, v in row["terms"].items()), key=lambda kv: kv[1])
        out.append(f"T1 {row['t1']:.2%} > 1%: its largest term is {big[0]} {big[1]:.2%}")
    if not row["t2_pass"]:
        out.append(f"T2 {row['t2']['judged']:.1f}x > 100x (hashing {row['t2']['hash_ms']:.1f} ms a forward); "
                   f"{row['hash_bytes_per_mac']:.3f} B of credited hashing a useful MAC")
    elif row["t2"]["unfused"] > TARGET["t2"]:
        out.append(f"a committer fused with the GEMM: unfused, writing the committed bytes to HBM first makes T2 "
                   f"{row['t2']['unfused']:.1f}x")
    if row["t3_worst"] is None:
        out.append(f"T3 unmeasured: {row['t3']['source']}")
    elif not row["t3_pass"]:
        out.append(f"T3 {row['t3_worst']:+.2%} > 5%")
    out.append(f"ShaSpeed (rank 4, CORE-SMELL C27) to within delta* = {row['delta_star']:.2e} of its "
               f"{row['hash_units_per_mac']:.0f} W1 units a useful MAC (if the honest hasher is our committer, not the "
               f"bar, its excess alone is {row['committer_gap']:.0f}x stock)")
    return out


def periods(card: dict, row: dict) -> dict:
    """Over each period, for one H200 (its honest rate at the point), the datacenter (that H200 times the census's GPUs,
    scaled) and the world (the census's inference compute, scaled): the useful compute covered, the gap (T1 of it),
    ShaSpeed's credited hashing and the slack delta* lets it leave, all in AND gates (W1 units x and_per_mac)."""
    sc = card["scale"]
    and_mac = card["price"]["and_per_mac"]["value"]
    tH = min(e["t_H_ms"] for e in row["t2"]["engines"].values()) / 1e3
    rate = row["stock_macs_forward"] / tH
    scopes = {"h200": rate, "datacenter": rate * sc["datacenter_gpus"],
              "world": sc["world_mac_per_year"] / PERIODS["year"]}
    out: dict = {"h200_useful_mac_per_s": rate,
                 "h200_share_of_peak": rate / card["price"]["mac_rate_peak"]["value"],
                 "proved": "T1's terms are the card's priced counts; the H200's rate is t_H's estimate on measured parts",
                 "scaled": "the datacenter is one H200 times datacenter_gpus; the world is world_mac_per_year"}
    for p, s in PERIODS.items():
        s = sc["window_s"] if s is None else s
        out[p] = {"seconds": s}
        for scope, macs_s in scopes.items():
            useful = macs_s * s * and_mac
            out[p][scope] = {"useful_ands": useful, "gap_ands": row["t1"] * useful,
                             "shaspeed_ands": row["hash_units_per_mac"] * useful,
                             "shaspeed_slack_ands": row["delta_star"] * row["hash_units_per_mac"] * useful}
    return out


def period_table(card: dict, row: dict) -> list[str]:
    per = periods(card, row)
    lines = [f"Per monitoring period, AND gates ({card['price']['and_per_mac']['value']:g} an int8 MAC, estimate); one "
             f"H200 at {per['h200_useful_mac_per_s']:.3g} useful MAC/s ({per['h200_share_of_peak']:.2%} of its tensor "
             f"peak), the datacenter {card['scale']['datacenter_gpus']:,} of them (scaled), the world's inference "
             f"(scaled). The gap is T1 x the useful compute; ShaSpeed's slack is delta* x its credited hashing, the most "
             f"it may err before T1 exceeds 1%.", "",
             "| period | scope | useful compute | gap (T1) | ShaSpeed-credited hashing | ShaSpeed slack at delta* |",
             "|---|---|---|---|---|---|"]
    for p in PERIODS:
        for scope in ("h200", "datacenter", "world"):
            v = per[p][scope]
            lines.append(f"| {p} ({per[p]['seconds']:,} s) | {scope} | {v['useful_ands']:.3e} | {v['gap_ands']:.3e} "
                         f"({row['t1']:.2%}) | {v['shaspeed_ands']:.3e} ({row['hash_units_per_mac']:.0f}x) | "
                         f"{v['shaspeed_slack_ands']:.3e} |")
    return lines


def if_refuted(card: dict, point: str, delta: float = 0.0) -> list[tuple[str, dict | None]]:
    """The best design at the point if each unsettled premise (open, scratch, or assumed beyond H32, W1 and ShaSpeed by a
    swept design) is refuted, and if all of them are."""
    swept = {p for d in card["design"].values() if d.get("swept") for p in d["premises"]}
    unsettled = [p for p, v in card["premise"].items() if v.get("status") in UNSETTLED
                 or (v.get("status") == "assumed" and p not in REQUIRED and p in swept)]
    out = []
    for drop in [[p] for p in unsettled] + [unsettled]:
        c = copy.deepcopy(card)
        for p in drop:
            c["premise"][p]["status"] = "refuted"
        rows, _ = ranking(c, point, delta)
        out.append((drop[0] if len(drop) == 1 else "every unsettled premise", rows[0] if rows else None))
    return out


def pct(x: float | None) -> str:
    return "?" if x is None else f"{x:.2%}"


def render(card: dict, results: dict) -> str:
    lines = ["# PoUW sweep ranking (sweep_price.py)", "",
             "T1: additive gap over stock in W1 units (H32 floor 32 a register, measured on the H200, run "
             "r20261029-021534-a5a3); terms are shares of stock. T2: t_H/t_S judged on the faster stock step. T3: worst of "
             "WikiText-2 2K and 8K over stock int8. Rank: the matmul argument's strongest premise; every row also needs "
             "ShaSpeed (rank 4, C27), whose precision delta* is given.", ""]
    for point, res in results.items():
        pt = card["point"][point]
        lines += [f"## {pt['title']}", "",
                  "| # | design | rank (w/ ShaSpeed) | T1 | final word | uncommitted | forming | derivation | de-noise | "
                  "pre-noise hash | final rows | T2 sync / async | T3 | delta* | open |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for i, row in enumerate(res["rows"], 1):
            t = row["terms"]
            e = row["t2"]["engines"]
            lines.append(
                f"| {i} | {row['design']} | {row['rank']} ({row['rank_with_shaspeed']}) | **{pct(row['t1'])}** "
                f"{'pass' if row['t1_pass'] else 'fail'} | {pct(t['final_word'])} | {pct(t['uncommitted_steps'])} | "
                f"{pct(t['forming'])} | {pct(t['noise_derivation'])} | {pct(t['denoise'])} | {pct(t['pre_noise_hashing'])} | "
                f"{pct(t['final_rows'])} | {e['sync']['ratio']:.1f}x / {e['async']['ratio']:.1f}x "
                f"{'pass' if row['t2_pass'] else 'fail'} | {pct(row['t3_worst'])} | {row['delta_star']:.1e} | "
                f"{opened(row)} |")
        lines += ["", "Refused (an unpriced term or outside the domain):", ""]
        lines += [f"- {k}: {'; '.join(v)}" for k, v in res["refused"].items()] or ["- none"]
        best = res["rows"][0] if res["rows"] else None
        if best:
            lines += ["", f"**Best: {best['design']}** ({best['title']}). It still needs:", ""]
            lines += [f"- {n}" for n in needs(best)]
            lines += ["", "If a premise fails:", "", "| refuted | best then | rank | T1 | T2 sync / async | still open |",
                      "|---|---|---|---|---|---|"]
            for drop, b in res["if_refuted"]:
                lines.append(f"| {drop} | - | - | - | - | - |" if b is None else
                             f"| {drop} | {b['design']} | {b['rank']} | {pct(b['t1'])} {'pass' if b['t1_pass'] else 'fail'} | "
                             f"{b['t2']['engines']['sync']['ratio']:.1f}x / {b['t2']['engines']['async']['ratio']:.1f}x | "
                             f"{opened(b)} |")
            lines += [""] + period_table(card, best)
            nxt = next((r for r in res["rows"] if r["t1_pass"] and r["rank"] > best["rank"]), None)
            if nxt:
                e = nxt["t2"]["engines"]
                lines += ["", f"Best at the next rank: {nxt['design']}, rank {nxt['rank']} "
                              f"({', '.join(a['premise'] for a in nxt['assumed']) or 'no assumed premise'}), T1 "
                              f"{pct(nxt['t1'])}, T2 {e['sync']['ratio']:.1f}x / {e['async']['ratio']:.1f}x."]
        clean = next((r for r in res["rows"] if not r["open"] and not r["assumed"]), None)
        if clean and clean is not best:
            lines += ["", f"Best with nothing open: {clean['design']}, T1 {pct(clean['t1'])}, rank {clean['rank']}.", ""]
            lines += period_table(card, clean)
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    checks = Checks()
    args = checks.args
    card_path = Path(args[args.index("--card") + 1]) if "--card" in args else CARD
    out = Path(args[args.index("--out") + 1]) if "--out" in args else Path(os.environ.get("OUT", "."))
    delta = float(args[args.index("--delta") + 1]) if "--delta" in args else 0.0
    card = load(card_path)
    results = {}
    for point in card["point"]:
        rows, refused = ranking(card, point, delta)
        results[point] = {"rows": rows, "refused": refused, "if_refuted": if_refuted(card, point, delta)}
    for point, res in results.items():
        print(f"== {card['point'][point]['title']}")
        for i, row in enumerate(res["rows"], 1):
            t = row["terms"]
            print(f"  {i:2d}. {row['design']:<22} rank {row['rank']} ({row['rank_with_shaspeed']})  T1 {row['t1']:8.3%}  "
                  + " ".join(f"{k[:6]} {v:.3%}" for k, v in t.items() if v)
                  + f"  T2 {row['t2']['engines']['sync']['ratio']:.1f}x/{row['t2']['engines']['async']['ratio']:.1f}x"
                  + f"  T3 {pct(row['t3_worst'])}  delta* {row['delta_star']:.1e}  open {opened(row)}")
        for k, v in res["refused"].items():
            print(f"   refused {k}: {'; '.join(v)}")

    def row(point: str, name: str) -> dict:
        return next(r for r in results[point]["rows"] if r["design"] == name)

    # (1) The calculator reproduces tonight's priced figures before it adds anything.
    for point, want in (("t8192", 0.0210), ("t512", 0.393), ("t512r512", 0.0649)):
        r = row(point, "ncp-k32")
        got = r["terms"]["final_word"] + r["terms"]["forming"]
        checks.check(f"ncp-k32 at {point}: final word + forming reproduce CORE-NCP §7's fused figure {want:.2%}",
                     abs(got - want) < 0.0006, f"{got:.3%}")
    r = row("t8192", "ncp-k32")
    checks.check("the final-word term over Llama-3.2-1B's linears is 32·Σn/ΣnK = 1.239% (CORE-TIME final)",
                 abs(r["terms"]["final_word"] - 32 * 23552 / 60817408) < 1e-12, f"{r['terms']['final_word']:.4%}")
    for point, want in (("t8192", (154.8, 186.4)), ("t512", (16.8, 33.4)), ("t512r512", (92.5, 162.4))):
        e = row(point, "ncp-k32")["t2"]["engines"]
        got = (e["sync"]["ratio"], e["async"]["ratio"])
        checks.check(f"ncp-k32's T2 at {point} reproduces CORE-NCP §6's {want[0]}x / {want[1]}x",
                     all(abs(g / w - 1) < 0.01 for g, w in zip(got, want)), f"{got[0]:.1f}x / {got[1]:.1f}x")
    checks.check("ncp-k32 hashes 0.375 B of k32 words a useful MAC (12/32, CORE-NCP §2)",
                 abs(row("t8192", "ncp-k32")["t2"]["bytes_forward"] / row("t8192", "ncp-k32")["stock_macs_forward"]
                     - (0.375 + 14336 / 60817408)) < 1e-9,
                 f"{row('t8192', 'ncp-k32')['t2']['bytes_forward'] / row('t8192', 'ncp-k32')['stock_macs_forward']:.4f} B "
                 f"(words 0.375 and A's commitment)")

    # (2) CORE-SMELL C28's terms, as the card prices them.
    pn = row("t8192", "ncp-k32")["terms"]["pre_noise_hashing"]
    checks.check("hashing A's commitment before the noise is 0.42x stock at T 8,192 (CORE-SMELL C28: 0.42-0.72x)",
                 0.40 < pn < 0.44, f"{pn:.3f}x")
    dv = row("t8192", "ncp-k32")["terms"]["noise_derivation"]
    checks.check("deriving E1 and F1 from XOF bytes is 8/n + 8/m: 0.29% fused at T 8,192 (CORE-SMELL C28)",
                 abs(dv - 0.0029) < 0.0002, f"{dv:.3%}")
    sh = row("t8192", "ncp-k32-xyef")
    checks.check("with every honest write forced, delta* is about 0.0015% (CORE-SMELL C27) at T 8,192",
                 1.3e-5 < sh["delta_star"] < 1.6e-5, f"delta* {sh['delta_star']:.2e}, {sh['hash_units_per_mac']:.0f} units a MAC")

    swept = [n for n, d in card["design"].items() if d.get("swept")]
    checks.check("the sweep generates every L x committed classes x forming charge x derivation charge, less a charge "
                 "beside its class committed: 7 x 9 = 63 designs", len(swept) == 63, f"{len(swept)} swept")
    s256, s63, s126 = (fast_saving(card, L, 8192, 8192, om) for L, om in ((256, 63), (512, 63), (512, 126)))
    checks.check("Strassen-Winograd on operands within +-63 saves nothing against the k32 counting at L 256 "
                 "(1/8 - 56/256 < 0) and 1/8 - 56/512 - 8(2/8192) = 1.37% at L 512, m = n = 8,192; on NCP's +-126 "
                 "operands its sums leave s8 and it spends 18/8 of the steps, saving nothing (P1's floor)",
                 s256[0] <= 0 and abs(s63[0] - (1 / 8 - 56 / 512 - 16 / 8192)) < 1e-12 and s63[1] == 1 and s126[0] == 0,
                 f"+-63: L 256 {s256[0]:.2%}, L 512 {s63[0]:.3%}; +-126: L 512 {s126[0]:.2%}")
    r64, r128 = row("t8192", "ncp-k64-xyef"), row("t8192", "ncp-k128-xyef")
    checks.check("a word every 64 steps rests on merge-k64 at rank 3 (with W1); every 128 or more on OwnProducts, rank 4",
                 r64["rank"] == 3 and r128["rank"] == 4, f"k64 rank {r64['rank']}, k128 rank {r128['rank']}")

    # (3) Refusals: what an unpriced term does.
    for name in ("dither-c", "fresh-u", "accel-d5-d6"):
        checks.check(f"{name} is refused at every point, never ranked",
                     all(name in res["refused"] for res in results.values()),
                     results["t8192"]["refused"].get(name, ["ranked"])[0])
    ranked = {r["design"] for res in results.values() for r in res["rows"]}
    checks.check("every ranked row has all of T1's terms priced, and a source for each design",
                 all(set(r["terms"]) == set(TERMS) for res in results.values() for r in res["rows"])
                 and all(card["design"][n].get("source") for n in ranked), f"{len(ranked)} designs ranked")
    checks.check("V4-Pro's experts (128 rows a call at 8,192 tokens) are inside the domain, and its T 512 isn't priced",
                 "v4pro" in results and results["v4pro"]["rows"], f"{len(results['v4pro']['rows'])} rows")

    if checks.mutations:
        def mutated(fn) -> dict:
            c = copy.deepcopy(card)
            fn(c)
            return c

        c = mutated(lambda c: c["noise"]["ncp"].pop("formed_y"))
        checks.mutation("an NCP card without its weight-side forming count is refused, not priced at 0",
                        all(n in ranking(c, "t8192")[1] for n in ("ncp-k32", "ncp-k32-xyef")),
                        ranking(c, "t8192")[1].get("ncp-k32", ["priced"])[0])
        c = mutated(lambda c: c["design"]["ncp-k32-xyef"]["commit"].__setitem__("word_bits", 16))
        checks.mutation("a 16-bit committed word is refused: no premise prices it",
                        "ncp-k32-xyef" in ranking(c, "t8192")[1], None)
        c = mutated(lambda c: c["premise"]["commit-forced"].__setitem__("status", "refuted"))
        checks.mutation("a refuted commit-forced (SW-LEAN's counterexample) refuses every design that takes it",
                        all(n in ranking(c, "t512")[1] for n in ("ncp-k32-xy", "ncp-k32-xyef")), None)
        lr, _ = ranking(card, "t512", lenient=True)
        top = [r["design"] for r in lr if r["t1_pass"]]
        checks.mutation("read '?' as 0 and CORE-RANK's unpriced designs enter T1's passing tier: the refusal is what keeps "
                        "them out", any(n in top for n in ("dither-c", "fresh-u", "accel-d5-d6")), f"passing tier {top}")
        nosem = row("t8192", "ncp-k64-nosem")
        checks.mutation("committing every 64 steps without the k32 semantics leaves r(1 - 32/64) = 150% of stock "
                        "uncommitted: the density needs candidate 2", nosem["terms"]["uncommitted_steps"] > 1.4,
                        f"{nosem['terms']['uncommitted_steps']:.0%}")
        c = mutated(lambda c: c["noise"]["ncp"].update(formed_x="m*K", formed_y="n*K"))
        a = evaluate(c, "ncp-k32-w", "t8192")["terms"]["forming"]
        b = row("t8192", "ncp-k32-w")["terms"]["forming"]
        checks.mutation("forming one copy a side (the certificate's 8/n + 8/m read over stock) would be a third of NCP's "
                        "24/n + 24/m: the stack's 3 copies decide the term", abs(3 * a - b) < 1e-12 and b > 0.0085,
                        f"{a:.3%} vs {b:.3%}")
        d1 = evaluate(card, "ncp-k32-xyef", "t8192", delta=1e-4)
        checks.mutation("a ShaSpeed slack of 0.01% breaks T1 for the all-forced design: hashing's work is 666x stock",
                        not d1["t1_pass"], f"T1 {d1['t1']:.2%}")
        c = mutated(lambda c: c["noise"]["ncp"].__setitem__("operand_max", 63))
        e = evaluate(c, "ncp-k512-xyef", "t8192")
        checks.mutation("NCP's operands at +-63 instead of +-126 let Strassen-Winograd's sums stay in s8, and "
                        "ncp-k512-xyef fails T1 at T 8,192 on its uncommitted steps: the operand range keeps it out",
                        not e["t1_pass"] and e["terms"]["uncommitted_steps"] > 0.01, f"T1 {e['t1']:.2%}")
        c = mutated(lambda c: c["point"]["t512"].__setitem__("tokens", 16))
        rows16, ref16 = ranking(c, "t512")
        priced = [r["design"] for r in results["t512"]["rows"]]
        checks.mutation("a decode batch of 16 rows is refused as outside the domain (C15), not priced",
                        not rows16 and all("domain" in ref16[n][0] for n in priced), ref16[priced[0]][0][:60])

    out.mkdir(parents=True, exist_ok=True)
    md = render(card, results)
    (out / "ranking.md").write_text(md + "\n")
    blob = {"schema": "sw-price/result/v1", "card": card, "delta": delta,
            "points": {p: {"rows": res["rows"], "refused": res["refused"],
                           "best": res["rows"][0]["design"] if res["rows"] else None,
                           "best_needs": needs(res["rows"][0]) if res["rows"] else [],
                           "best_periods": periods(card, res["rows"][0]) if res["rows"] else None,
                           "if_refuted": [{"refuted": drop, "best": b and b["design"], "t1": b and b["t1"],
                                           "t2": b and b["t2"]["judged"], "open": b and [o["premise"] for o in b["open"]]}
                                          for drop, b in res["if_refuted"]]}
                       for p, res in results.items()}}
    (out / "sweep_price.json").write_text(json.dumps(blob, indent=1, default=str) + "\n")
    with (out / "ranking.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["point", "position", "design", "rank", "rank_with_shaspeed", "t1", *TERMS, "t2_sync", "t2_async",
                    "t3_worst", "delta_star", "hash_bytes_per_mac", "open"])
        for p, res in results.items():
            for i, r in enumerate(res["rows"], 1):
                e = r["t2"]["engines"]
                w.writerow([p, i, r["design"], r["rank"], r["rank_with_shaspeed"], f"{r['t1']:.6g}",
                            *(f"{r['terms'][k]:.6g}" for k in TERMS), f"{e['sync']['ratio']:.4g}",
                            f"{e['async']['ratio']:.4g}", "" if r["t3_worst"] is None else f"{r['t3_worst']:.4g}",
                            f"{r['delta_star']:.4g}", f"{r['hash_bytes_per_mac']:.4g}",
                            " ".join(o["premise"] for o in r["open"])])
    print(f"wrote {out / 'ranking.md'}, {out / 'sweep_price.json'}, {out / 'ranking.csv'}")
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
