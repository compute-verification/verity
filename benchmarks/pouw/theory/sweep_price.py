#!/usr/bin/env python3
"""The PoUW sweep's calculator: a design, (noise and de-noise) × (what is committed, how densely, in which rows and
packing) × (the premise set), priced under CORE.md §2a on the H200, for Llama-3.2-1B int8 at T 512 (64 real tokens) and
T 8,192 and for DeepSeek-V4-like shapes, and the sweep's ranking.

T1 is the additive gap cost(H) − min cost(adv) as a share of stock's m·n·K, in W1 units (an int8 MAC one unit, a register
written `reg` = 32 under H32, measured on the H200). The honest program is the best possible, so every register it writes
costs `reg`. Under DistinctWrites each committed, pairwise-distinct, noise-dependent word costs the adversary at least
`reg`. The gap is honest's work that no premise forces, by term:

- final word: each dot product's last word is A·Bᵀ, free advice until the final-word lemma (32/K at L 32), unless the
  noise is served (`final_target`: the output carries it, so the last word is a target);
- uncommitted steps: r·(1 − u/L) between words committed every L, where a word is charged u = reg (MH100w's
  anyFunctionCeiling) or a premise's `word_units` (SW-SEM's BamFloor, 64 − 1024/L; PureLanes or CellPure, L to 128); under a premise
  that credits the k32 counting itself, plus what the best-known fast scheme (Strassen-Winograd, `price.fast_matmul`)
  saves on them;
- forming: the formed operand entries at reg/4 each, unless committed under `commit-forced` or charged by the counting
  lemma;
- lane fix: the route's second write to each register whose signed bytes cross zero under the noise (SW-SMELL S9), which
  no premise forces;
- noise derivation: the noise entries derived from XOF bytes (one LOP3 a packed register, CORE-SMELL C28), the same way,
  and the derivation's intermediates no commitment holds (SSN's shift before its LOP3), always;
- de-noise: the design's correction work, unless forced;
- pre-noise hashing: bytes hashed before the noise (A's commitment feeds the call's seed), credited once the call is
  atomic; so are rows holding only final words (`final_rows`, C28);
- hiding: each hm96 leaf's salt side (the PRF, M_key·y and c), which a prover with one fixed salt skips with nothing
  checked changing (SW-HASH §7), unless forced or ruled;
- hashing slack: credited hashing under ShaSpeed, δ times its work (δ = 0 as §2a states it). δ* is the precision
  ShaSpeed needs for T1 to hold (CORE-SMELL C27). The coins' expansion is credited hashing too (raw coin words are free
  in M0, so expanding them is honest work no word forces); T1 with it uncredited is reported beside T1 (SW-NOISE final).

T2 is t_H / t_S, a serial sum: the stock step, the stacked product's extra over the plain GEMM (NCP's 3K-wide call,
GD-DENOISE; or the plain GEMM's time a MAC on the extra width), the noise's coins a call and NCP's formed copies
(SW-GPU's coinsncp and coins), A's root before each call's noise (its chain on the critical
path, an estimate), exporting each committed word (SW-GPU's k32 rates), and hashing the committed words and operands at
the layout's measured rate. It is judged on the faster (async) stock step. T3 is the served format's measured perplexity
rise. The rank is the strongest premise the matmul argument needs; the overall rank, the larger of it and the hashing
credit's (ShaSpeed and its composition with DistinctWrites, rank 4), is shown beside it, since every design that credits
hashing needs both.

A design whose card entry leaves a count missing or "?" is refused and never ranked, so an unpriced term can't top the
table. Every number carries its card source: measured (a run id), estimate, or arithmetic.

The ranking, at each point: designs that meet T1 first (T1 is hard), then rank (an open premise counts as rank 5 until
settled, then its own), then T2's and T3's targets, then the fewest unsettled premises, then T1, T2 and T3.

  uv run python benchmarks/pouw/theory/sweep_price.py [--card PATH] [--out DIR] [--delta D] [--mutations] [--verdict PATH]
"""
from __future__ import annotations

import copy
import csv
import functools
import itertools
import json
import math
import os
import sys
import tomllib
from pathlib import Path

from exact_kit.checks import Checks

CARD = Path(__file__).with_name("sweep_price.toml")
TERMS = ("final_word", "uncommitted_steps", "forming", "lane_fix", "noise_derivation", "denoise", "pre_noise_hashing",
         "final_rows", "hiding", "hashing_slack")
NOISE_FIELDS = ("r", "formed_x", "formed_y", "noise_x", "noise_y", "derive_extra", "denoise", "pre_noise_bytes",
                "operand_max", "lane_fix", "coin_calls", "wide", "final_target", "served", "variant", "source")
WIDE = ("measured", "proportional")
STATUSES = ("proved", "measured", "ruled", "assumed", "scratch", "open", "refuted")
UNSETTLED = ("open", "scratch")
COMMIT_FIELDS = ("L", "word_bits", "operands", "rows_mix_final", "layout", "route")
ROUTES = {"U": ("U",), "mix": ("U", "U'"), "none": ("none",)}
CLASSES = {"formed_x": "forming", "formed_y": "forming", "noise_x": "noise_derivation", "noise_y": "noise_derivation"}
HASH_CREDIT = ("ShaSpeed", "hash-compose")
REQUIRED = ("H32", "W1") + HASH_CREDIT
TARGET = {"t1": 0.01, "t2": 100.0, "t3": 0.05}
WORD_BYTES = 4
BLOCK = 128
MIN_ROWS = 64
PERIODS = {"window": None, "week": 7 * 86400, "month": 30 * 86400, "year": 365 * 86400}
SHOWN = 25


class Refused(Exception):
    """A design or point the card doesn't price: its reasons, each naming the field."""

    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


@functools.lru_cache(maxsize=None)
def _code(s: str):
    return compile(s, "<card>", "eval")


def ev(s, **v) -> float:
    if s is None or s == "?":
        raise Refused(["unpriced"])
    if isinstance(s, (int, float)):
        return float(s)
    return float(eval(_code(str(s)), {"__builtins__": {}, "ceil": math.ceil}, v))  # noqa: S307 - the card's arithmetic


def interp(xs: list[float], ys: list[float], x: float) -> float:
    """Log-log between the table's points, its end values outside them."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = next(j for j in range(len(xs) - 1) if xs[j] <= x <= xs[j + 1])
    t = math.log(x / xs[i]) / math.log(xs[i + 1] / xs[i])
    if ys[i] <= 0 or ys[i + 1] <= 0:
        return ys[i] + t * (ys[i + 1] - ys[i])
    return math.exp(math.log(ys[i]) + t * math.log(ys[i + 1] / ys[i]))


def load(path: Path = CARD) -> dict:
    card = tomllib.loads(path.read_text())
    for sw in card.get("sweep", {}).values():
        charge = sw.get("charge", {})
        credits = sw.get("credit", {})
        for nz, L, (tag, ops), count, least, layout, (hid, hprem), (route, rprem), cred in itertools.product(
                sw["noise"], sw["L"], sw["operands"].items(), (False, True) if "count" in charge else (False,),
                (False, True) if "min" in charge else (False,), sw["layouts"], sw["hiding"].items(),
                sw["routes"].items(), [None] + list(credits)):
            if (count and {"formed_x", "formed_y"} <= set(ops)) or (least and {"noise_x", "noise_y"} <= set(ops)):
                continue
            if (L > 32) != (cred is not None):
                continue
            prem = list(sw["base"]) + (credits[cred] if cred else [])
            if ops:
                prem.append("commit-forced")
            prem += (charge["count"] if count else []) + (charge["min"] if least else []) + hprem + rprem
            what = ", ".join(ops) if ops else "words only"
            name = (f"{nz}-k{L}" + (f"-{cred}" if cred else "") + f"-{tag}" + ("-count" if count else "")
                    + ("-min" if least else "") + f"-{layout}" + ("" if hid == "open" else f"-{hid}")
                    + ("-u2" if route == "mix" else ""))
            card["design"][name] = {
                "title": f"{nz.upper()}, a word every {L} steps" + (f" credited by {cred}" if cred else "")
                         + f", committed: {what}"
                         + ("; forming by the counting lemma" if count else "")
                         + ("; the derivation by OneInstructionMinimal" if least else "")
                         + f"; {layout} rows; hiding {hid}; route {route}",
                "noise": nz, "premises": list(dict.fromkeys(prem)), "source": sw["source"], "swept": True,
                "commit": {"L": L, "word_bits": 32, "operands": list(ops), "rows_mix_final": True, "layout": layout,
                           "route": route}}
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
            if noise.get(f) in (None, "?", {}):
                if not lenient:
                    out.append(f"noise.{d['noise']}.{f} unpriced")
            elif f == "coin_calls":
                for table, exprs in noise[f].items():
                    if table not in card["price"] or "entries" not in card["price"][table]:
                        out.append(f"noise.{d['noise']}.coin_calls: no measured call table price.{table}")
                    for s in exprs:
                        try:
                            ev(s, m=64, n=64, K=64)
                        except Exception as e:  # noqa: BLE001 - the card's expression is the input under test
                            out.append(f"noise.{d['noise']}.coin_calls.{table} = {s!r} doesn't evaluate ({e})")
            elif f == "wide":
                if noise[f] not in WIDE:
                    out.append(f"noise.{d['noise']}.wide {noise[f]!r} is not one of {WIDE}")
            elif f not in ("served", "variant", "source", "final_target"):
                for k, s in (noise[f].items() if isinstance(noise[f], dict) else [(None, noise[f])]):
                    try:
                        ev(s, m=64, n=64, K=64)
                    except Exception as e:  # noqa: BLE001 - the card's expression is the input under test
                        out.append(f"noise.{d['noise']}.{f}{'.' + k if k else ''} = {s!r} doesn't evaluate ({e})")
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
    if c.get("layout") not in (None, "?") and c["layout"] not in card.get("layout", {}):
        out.append(f"commit.layout {c['layout']!r} has no rows in the card")
    if c.get("route") not in (None, "?"):
        need = ROUTES.get(c["route"])
        lanes = (noise or {}).get("lane_fix") or {}
        if need is None or any(k not in lanes for k in need):
            out.append(f"commit.route {c['route']!r}: noise.{d.get('noise')}.lane_fix prices {sorted(lanes)}")
        if c["route"] == "mix" and "route-u2" not in d.get("premises", []):
            out.append("route mix takes route U', whose checked values need route-u2")
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


def effects(card: dict, ids: list[str], L: int) -> tuple[set[str], bool, float, bool]:
    """The terms the premises close, whether committed registers are forced, the W1 units a committed word every L
    steps is charged (`reg` under DistinctWrites alone, MH100w's anyFunctionCeiling; more under a premise's
    `word_units`), and whether the best-known fast scheme's saving is charged against that credit (`fast_saving`: a
    premise that credits the k32 counting itself, not a floor over every program)."""
    prem = card["premise"]
    reg = card["price"]["reg"]["value"]
    closes = set().union(*(set(prem[p].get("closes", [])) for p in ids if p in prem))
    forced = any(prem[p].get("forces_committed", False) for p in ids if p in prem)
    u, fast = max([(reg, False)] + [(ev(prem[p]["word_units"], L=L, reg=reg), bool(prem[p].get("fast_saving")))
                                    for p in ids if p in prem and "word_units" in prem[p]])
    return closes, forced, u, fast


def calls(card: dict, point: str) -> list[dict]:
    pt = card["point"][point]
    model = card["model"][pt["model"]]
    return [{"name": e[0], "n": e[1], "K": e[2], "count": e[3], "m": pt["tokens"] * e[4],
             "serial": e[5] if len(e) > 5 else e[3]} for e in model["matmuls"]]


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


def leaf_time_ratio(card: dict, leaf: float) -> float:
    """hm96's hash time a byte at this leaf size against 16 KiB rows."""
    h = card["price"]["hash_rows"]
    r0 = h["rows"][0]
    if leaf < r0:
        c = h["fixed_compressions"]
        return (leaf / BLOCK + c) / leaf / ((r0 / BLOCK + c) / r0)
    return interp(h["rows"], h["time_ratio"], leaf)


def leaf_hiding(card: dict, leaf: float) -> tuple[float, float]:
    """A leaf's salt side in compressions' time: (PRF and c, M_key·y)."""
    h = card["price"]["hash_rows"]
    hide = interp(h["rows"], [n * s for n, s in zip(h["ns_leaf"], h["hide_share"])], leaf)
    mask = interp(h["rows"], [n * s for n, s in zip(h["ns_leaf"], h["mask_share"])], leaf)
    return (hide - mask) / h["comp_ns"], mask / h["comp_ns"]


def coinsncp_ms(card: dict, entries: float, table: str = "coinsncp") -> float:
    c = card["price"][table]
    if entries >= c["entries"][-1]:
        return c["ms"][-1] * entries / c["entries"][-1]
    return interp(c["entries"], c["ms"], entries)


def call_units(card: dict, name: str, m: float, n: float, K: float, g: float = 1.0, rate: float = 0.0,
               lenient: bool = False) -> dict:
    """One call's T1 terms in W1 units, the credited hashing's bytes, and its share of T2's per-call parts (ms) when `g`
    calls run side by side in one serial step and hashing runs at `rate` B/s on 16 KiB rows."""
    d = card["design"][name]
    noise = card["noise"].get(d["noise"]) if d.get("noise") in card.get("noise", {}) else None
    c = d["commit"]
    price = card["price"]
    reg, epr, hw = price["reg"]["value"], price["entries_per_reg"]["value"], price["hash_work"]["value"]

    def get(f, default=0.0):
        if noise is None or noise.get(f) in (None, "?"):
            if lenient:
                return default
            raise Refused([f"noise.{d.get('noise')}.{f} unpriced"])
        return ev(noise[f], m=m, n=n, K=K)

    if noise is None and lenient:
        return {"terms": dict.fromkeys(TERMS, 0.0), "stock": m * n * K, "r": 1.0, "mK": m * K, "nK": n * K} | dict.fromkeys(
            ("credited_bytes", "hashed_bytes", "word_bytes", "operand_bytes", "pre_noise_bytes", "coin_bytes", "x_entries",
             "y_entries", "noise_y", "words", "coinform_ms", "a_ms", "bulk_hash_ms", "sts_ms", "stg_ms", "a_chain",
             "extra_macs"), 0.0)
    r = get("r", 1.0)
    L = c["L"] if isinstance(c.get("L"), int) else (32 if lenient else None)
    if L is None:
        raise Refused(["commit.L unpriced"])
    depth = round(r * K, 6)
    if depth % L:
        raise Refused([f"{r:g}K = {depth:g} is not a whole number of words at L {L}"])
    if L >= depth:
        raise Refused([f"a word every {L} steps commits only the final word of a {depth:g}-deep dot product, and the "
                       f"final word isn't noise-dependent"])
    closes, forced, u_word, fast = effects(card, d.get("premises", []), L)
    stock = m * n * K
    words = depth / L
    final_free = "final_word" not in closes and not (noise or {}).get("final_target", False)
    honest_tensor = r * m * n * K
    t = dict.fromkeys(TERMS, 0.0)
    t["final_word"] = m * n * u_word if final_free else 0.0
    t["uncommitted_steps"] = honest_tensor - m * n * words * u_word
    if abs(t["uncommitted_steps"]) < 1e-9 * honest_tensor:
        t["uncommitted_steps"] = 0.0
    if fast and L > 32:
        t["uncommitted_steps"] += fast_saving(card, L, m, n, get("operand_max"))[0] * honest_tensor
    committed = set(c.get("operands", []))
    ent = {k: get(k) for k in CLASSES}
    for k, term in CLASSES.items():
        if not ((forced and k in committed) or term in closes):
            t[term] += ent[k] * reg / epr
    if "noise_derivation" not in closes:
        t["noise_derivation"] += get("derive_extra") * reg / epr
    lanes = noise["lane_fix"]
    t["lane_fix"] = min(ev(lanes[k], m=m, n=n, K=K) for k in ROUTES[c["route"]]) * reg / epr
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

    lay = card["layout"][c["layout"]]
    a_leaf = K if lay["a_row"] == "row" else min(lay["a_row"], pre_b)
    objects = [(pre_b, a_leaf), (word_b, min(lay["row"], word_b)), (op_b, min(lay["row"], op_b) if op_b else 1.0)]
    salt_c = mask_c = 0.0
    for b, leaf in objects:
        if b:
            s, k = leaf_hiding(card, leaf)
            salt_c += math.ceil(b / leaf) * s
            mask_c += math.ceil(b / leaf) * k
    unit_c = BLOCK * hw
    if "hiding" not in closes:
        t["hiding"] = ((0.0 if "hiding_salt" in closes else salt_c) + (0.0 if "hiding_mask" in closes else mask_c)) * unit_c
    salt_b = salt_c * BLOCK if ("hiding_salt" in closes and "hiding" not in closes) else 0.0
    xof = {table: [ev(s, m=m, n=n, K=K) for s in exprs] for table, exprs in noise["coin_calls"].items()}
    coin_b = sum(map(sum, xof.values())) * price["coin_bytes"]["value"]
    credited_b = word_b + op_b + coin_b + salt_b + (pre_b if "pre_noise_hashing" in closes else 0.0)

    lat = price["latency"]
    a_leaves = math.ceil(pre_b / a_leaf)
    depth = math.ceil(math.log2(a_leaves)) if a_leaves > 1 else 0
    chain = math.ceil(a_leaf / BLOCK) + 1 + 2 + 3 * depth + 1
    a_chain_s = chain * lat["compression_s"] + (2 + depth) * lat["launch_s"]
    a_bulk_s = g * pre_b * leaf_time_ratio(card, a_leaf) / rate if rate else 0.0
    ex = price["extract"]
    sts = ex["sts_units_per_word"] / (ex["units_per_clock"] * ex["sts_mhz"] * 1e6)
    stg = ex["stg_units_per_word"] / (ex["units_per_clock"] * ex["stg_mhz"] * 1e6)
    nwords = m * n * words
    bulk = ((word_b * leaf_time_ratio(card, objects[1][1]) + (op_b * leaf_time_ratio(card, objects[2][1]) if op_b else 0.0))
            / rate if rate else 0.0)
    return {"stock": stock, "terms": t, "credited_bytes": credited_b, "hashed_bytes": word_b + op_b + pre_b,
            "word_bytes": word_b, "operand_bytes": op_b, "pre_noise_bytes": pre_b, "coin_bytes": coin_b, "r": r,
            "x_entries": ent["formed_x"] + ent["noise_x"], "y_entries": ent["formed_y"] + ent["noise_y"],
            "noise_y": ent["noise_y"], "mK": m * K, "nK": n * K, "words": nwords, "extra_macs": (r - 1) * stock,
            "coinform_ms": sum(coinsncp_ms(card, b * g, table) for table, bs in xof.items() for b in bs) / g,
            "a_ms": max(a_chain_s, a_bulk_s) / g * 1e3, "a_chain": chain / g, "bulk_hash_ms": bulk * 1e3,
            "sts_ms": nwords * sts * 1e3, "stg_ms": nwords * stg * 1e3}


def t3_of(card: dict, served: str | None, model: str) -> dict:
    e = card.get("t3", {}).get(served or "")
    if not e or ("models" in e and model not in e["models"]) or e.get("kind") == "unmeasured":
        why = f": {e['source']}" if e and e.get("kind") == "unmeasured" else ""
        return {"ctx2k": None, "ctx8k": None, "kind": "unmeasured", "source": f"no T3 for {served!r} on {model}{why}"}
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
    rate = pt.get("hash_rate", card["price"]["hash_rate"]["value"])
    keys = ("stock", "credited_bytes", "hashed_bytes", "word_bytes", "operand_bytes", "pre_noise_bytes", "coin_bytes",
            "x_entries", "y_entries", "noise_y", "mK", "nK", "words", "coinform_ms", "a_ms", "a_chain", "bulk_hash_ms",
            "sts_ms", "stg_ms", "extra_macs")
    agg: dict = dict.fromkeys(keys, 0.0) | {"terms": dict.fromkeys(TERMS, 0.0)}
    per_matmul = {}
    r = None
    for x in cs:
        u = call_units(card, name, x["m"], x["n"], x["K"], x["count"] / x["serial"], rate, lenient)
        r = u["r"]
        w = x["count"]
        for k in keys:
            agg[k] += w * u[k]
        for tk in TERMS:
            agg["terms"][tk] += w * u["terms"][tk]
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
    noise = card["noise"].get(d["noise"], {})
    t2 = t2_of(card, point, agg, r, noise.get("wide", "measured"))
    t3 = t3_of(card, noise.get("served"), pt["model"])
    prem = card["premise"]
    ids = d["premises"]
    rank = max(prem[p]["rank"] for p in ids if p not in HASH_CREDIT)
    rank_all = max(prem[p]["rank"] for p in ids)
    open_items = [{"premise": p, "owner": prem[p].get("owner", ""), "status": prem[p]["status"],
                   "rank": prem[p]["rank"], "source": prem[p]["source"]}
                  for p in ids if prem[p]["status"] in UNSETTLED]
    assumed = [{"premise": p, "rank": prem[p]["rank"], "source": prem[p]["source"]}
               for p in ids if prem[p]["status"] == "assumed" and p not in REQUIRED]
    t3w = max(v for v in (t3["ctx2k"], t3["ctx8k"]) if v is not None) if t3["ctx2k"] is not None else None
    matmul_open = [o for o in open_items if o["premise"] not in HASH_CREDIT]
    return {"design": name, "title": d["title"], "point": point, "t1": t1, "terms": terms, "t1_pass": t1 <= TARGET["t1"],
            "t1_coins_uncredited": t1 + agg["coin_bytes"] * hw / stock,
            "delta": delta, "delta_star": delta_star, "hash_units_per_mac": hash_share,
            "hash_bytes_per_mac": agg["credited_bytes"] / stock, "coin_bytes_per_mac": agg["coin_bytes"] / stock,
            "pre_noise_units": agg["pre_noise_bytes"] * hw / stock,
            "committer_gap": committer_gap, "t2": t2, "t2_pass": t2["judged"] <= TARGET["t2"],
            "t3": t3, "t3_worst": t3w, "t3_pass": t3w is not None and t3w <= TARGET["t3"], "rank": rank,
            "rank_effective": 5 if matmul_open else rank, "rank_with_hash": rank_all, "premises": ids,
            "open": open_items, "matmul_open": len(matmul_open), "assumed": assumed, "per_matmul": per_matmul,
            "stock_macs_forward": stock, "commit": d["commit"], "a_chain_compressions": agg["a_chain"]}


def t2_of(card: dict, point: str, agg: dict, r: float, wide: str = "measured") -> dict:
    """t_H / t_S a forward: stock, plus the stacked product's extra (`measured`: GD-DENOISE's 3K-wide call;
    `proportional`: the plain GEMM's time per MAC on the extra width, an estimate), the coins and formed copies a call,
    A's root before each call's noise, the words' export, and hashing; and CORE-NCP's reading (GD-DENOISE's stand-ins
    for forming, NCP's only)."""
    pt = card["point"][point]
    price = card["price"]
    src = card["point"][pt.get("gemm_from", point)]
    parts = src["gemm_parts"]
    rate = pt.get("hash_rate", price["hash_rate"]["value"])
    if pt.get("matmul_only"):
        src_macs = sum(x["count"] * x["m"] * x["n"] * x["K"] for x in calls(card, pt["gemm_from"]))
        plain = pt.get("plain_gemm_ms", agg["stock"] / (src_macs / src["plain_gemm_ms"]))
        stock = {"sync": plain, "async": plain}
    else:
        plain = src["plain_gemm_ms"]
        stock = dict(pt["stock_ms"])
    if wide == "proportional":
        gemm = plain * agg["extra_macs"] / agg["stock"]
    else:
        gemm = plain * (parts["wide"] * (r - 1) / 2 + parts["noised"])
    extra_parts = {"wide_gemm": gemm, "coins_forming": agg["coinform_ms"], "a_root": agg["a_ms"],
                   "export": agg["sts_ms"]}
    extra = sum(extra_parts.values())
    hash_ms = agg["bulk_hash_ms"]
    eng = {e: {"stock_ms": s, "t_H_ms": s + extra + hash_ms, "ratio": (s + extra + hash_ms) / s,
               "hash_share": hash_ms / (s + extra + hash_ms)} for e, s in stock.items()}
    unfused = max((s + extra - agg["sts_ms"] + agg["stg_ms"] + hash_ms) / s for s in stock.values())
    rx = agg["x_entries"] / (4 * agg["mK"])
    ry = agg["y_entries"] / (4 * agg["nK"])
    old = (plain * parts["wide"] * (r - 1) / 2 + plain * parts["a_side"] * rx + plain * parts["noised"]
           + plain * parts["w_side"] * ry + agg["noise_y"] / 64 / price["coin_rate"]["value"] * 1e3
           + agg["hashed_bytes"] / rate * 1e3)
    return {"engines": eng, "judged": max(v["ratio"] for v in eng.values()), "unfused": unfused,
            "extra_ms": extra, "extra_parts_ms": extra_parts, "hash_ms": hash_ms, "plain_gemm_ms": plain,
            "core_ncp": {e: (s + old) / s for e, s in stock.items()},
            "without_hashing": max((s + extra) / s for s in stock.values()), "bytes_forward": agg["hashed_bytes"],
            "kind": "estimate on measured parts: " + (pt.get("gemm_source", "") if pt.get("matmul_only") else
                                                      f"{pt['stock_kind']} stock; 3K product {src['gemm_kind']}") +
                    "; coins and forming measured (coinsncp); A's root chain estimated; export measured on resident "
                    "tiles; no end-to-end honest program exists"}


def rank_key(row: dict) -> tuple:
    return (not row["t1_pass"], row["rank_effective"], row["rank"], not row["t2_pass"], not row["t3_pass"],
            len(row["open"]), row["t1"], row["t2"]["judged"],
            row["t3_worst"] if row["t3_worst"] is not None else math.inf)


def ranking(card: dict, point: str, delta: float = 0.0, lenient: bool = False,
            names: list[str] | None = None) -> tuple[list[dict], dict[str, list[str]]]:
    rows, refused = [], {}
    for name in names or card["design"]:
        try:
            rows.append(evaluate(card, name, point, delta, lenient))
        except Refused as e:
            refused[name] = e.reasons
    return sorted(rows, key=rank_key), refused


def opened(row: dict) -> str:
    return ", ".join(o["premise"] + (" (scratch)" if o["status"] == "scratch" else "") for o in row["open"]
                     if o["premise"] not in HASH_CREDIT) or "-"


def needs(row: dict) -> list[str]:
    out = [f"{o['premise']} ({o['owner']}, rank {o['rank']} once settled; "
           f"{'proved outside the lock: lock it' if o['status'] == 'scratch' else 'open'}): {o['source']}"
           for o in row["open"]]
    out += [f"{a['premise']} (assumed, rank {a['rank']}): {a['source']}" for a in row["assumed"]]
    if not row["t1_pass"]:
        big = max(((k, v) for k, v in row["terms"].items()), key=lambda kv: kv[1])
        out.append(f"T1 {row['t1']:.2%} > 1%: its largest term is {big[0]} {big[1]:.2%}")
    t2 = row["t2"]
    big = max(t2["extra_parts_ms"].items(), key=lambda kv: kv[1])
    if not row["t2_pass"]:
        out.append(f"T2 {t2['judged']:.1f}x > 100x (hashing {t2['hash_ms']:.1f} ms a forward, largest other part {big[0]} "
                   f"{big[1]:.1f} ms); {row['hash_bytes_per_mac']:.3f} B of credited hashing a useful MAC")
    elif t2["unfused"] > TARGET["t2"]:
        out.append(f"a committer fused with the GEMM: exporting the words to HBM instead of shared memory makes T2 "
                   f"{t2['unfused']:.1f}x")
    out.append(f"T2's parts measured where stated: A's root before each call's noise ({row['a_chain_compressions']:.0f} "
               f"compressions' chain a forward, {t2['extra_parts_ms']['a_root']:.2f} ms) is an estimate nobody has timed")
    if row["t3_worst"] is None:
        out.append(f"T3 unmeasured: {row['t3']['source']}")
    elif not row["t3_pass"]:
        out.append(f"T3 {row['t3_worst']:+.2%} > 5%")
    out.append(f"ShaSpeed (rank 4, CORE-SMELL C27) to within delta* = {row['delta_star']:.2e} of its "
               f"{row['hash_units_per_mac']:.0f} W1 units a useful MAC, and its composition with DistinctWrites "
               f"(hash-compose); if the honest hasher is our committer, not the bar, its excess alone is "
               f"{row['committer_gap']:.0f}x stock; with the coins' expansion uncredited T1 is "
               f"{row['t1_coins_uncredited']:.3%}")
    return out


def periods(card: dict, row: dict) -> dict:
    """Over each period, for one H200, the datacenter (that H200 times the census's GPUs, scaled) and the world (the
    census's inference compute, scaled), on two bases: `hardware` holds the scope's hardware fixed, which then covers
    stock's work over T2; `compute` holds the scope's stock compute fixed, which then takes T2 times the hardware. Each
    gives the useful compute covered, the gap (T1 of it), ShaSpeed's credited hashing and the slack delta* leaves it, in
    AND gates (W1 units x and_per_mac)."""
    sc = card["scale"]
    and_mac = card["price"]["and_per_mac"]["value"]
    tH = min(e["t_H_ms"] for e in row["t2"]["engines"].values()) / 1e3
    tS = min(e["stock_ms"] for e in row["t2"]["engines"].values()) / 1e3
    honest = row["stock_macs_forward"] / tH
    stock = row["stock_macs_forward"] / tS
    world = sc["world_mac_per_year"] / PERIODS["year"]
    scopes = {"h200": {"hardware": honest, "compute": stock},
              "datacenter": {"hardware": honest * sc["datacenter_gpus"], "compute": stock * sc["datacenter_gpus"]},
              "world": {"hardware": world * tS / tH, "compute": world}}
    out: dict = {"h200_useful_mac_per_s": honest, "h200_share_of_peak": honest / card["price"]["mac_rate_peak"]["value"],
                 "hardware_for_compute": tH / tS,
                 "proved": "T1's terms are the card's priced counts under the row's premises; the H200's rate is t_H's "
                           "estimate on measured parts",
                 "scaled": "the datacenter is one H200 times datacenter_gpus; the world is world_mac_per_year, its "
                           "hardware taken as running stock at that rate"}
    for p, s in PERIODS.items():
        s = sc["window_s"] if s is None else s
        out[p] = {"seconds": s}
        for scope, bases in scopes.items():
            for basis, macs_s in bases.items():
                useful = macs_s * s * and_mac
                out[p][f"{scope}/{basis}"] = {
                    "useful_ands": useful, "gap_ands": row["t1"] * useful,
                    "shaspeed_ands": row["hash_units_per_mac"] * useful,
                    "shaspeed_slack_ands": min(row["delta_star"], 1e9) * row["hash_units_per_mac"] * useful}
    return out


def period_table(card: dict, row: dict) -> list[str]:
    per = periods(card, row)
    lines = [f"Per monitoring period, AND gates ({card['price']['and_per_mac']['value']:g} an int8 MAC, estimate). One "
             f"H200 at its honest rate covers {per['h200_useful_mac_per_s']:.3g} useful MAC/s "
             f"({per['h200_share_of_peak']:.2%} of its tensor peak); the datacenter is {card['scale']['datacenter_gpus']:,} "
             f"of them and the world the census's inference compute (both scaled). Basis `hardware`: the scope's hardware "
             f"runs the design. Basis `compute`: the scope's stock compute is covered, on {per['hardware_for_compute']:.1f}x "
             f"its hardware. The gap is T1 x the useful compute; ShaSpeed's slack is delta* x its credited hashing, the "
             f"most it may err before T1 exceeds 1%.", "",
             "| period | scope | basis | useful compute | gap (T1) | ShaSpeed-credited hashing | ShaSpeed slack at delta* |",
             "|---|---|---|---|---|---|---|"]
    for p in PERIODS:
        for scope in ("h200", "datacenter", "world"):
            for basis in ("hardware", "compute"):
                v = per[p][f"{scope}/{basis}"]
                lines.append(f"| {p} ({per[p]['seconds']:,} s) | {scope} | {basis} | {v['useful_ands']:.3e} | "
                             f"{v['gap_ands']:.3e} ({row['t1']:.2%}) | {v['shaspeed_ands']:.3e} "
                             f"({row['hash_units_per_mac']:.0f}x) | {v['shaspeed_slack_ands']:.3e} |")
    return lines


def unsettled_premises(card: dict) -> list[str]:
    swept = {p for d in card["design"].values() if d.get("swept") for p in d["premises"]}
    return [p for p, v in card["premise"].items() if v.get("status") in UNSETTLED
            or (v.get("status") == "assumed" and p not in REQUIRED and p in swept)]


def if_refuted(card: dict, rows: list[dict]) -> list[tuple[str, dict | None]]:
    """The best design if each unsettled premise (open, scratch, or assumed beyond the required ones by a swept design)
    is refuted, and if all of them are but the hashing credit. Refuting a premise refuses exactly the designs that take
    it (a premise that requires it is taken with it) and changes no other row, so this filters the ranking."""
    uns = unsettled_premises(card)
    out = []
    for drop in [[p] for p in uns] + [[p for p in uns if p not in HASH_CREDIT]]:
        best = next((r for r in rows if not set(drop) & set(r["premises"])), None)
        out.append((drop[0] if len(drop) == 1 else "every unsettled premise but the hashing credit", best))
    return out


def alternates(rows: list[dict]) -> dict[str, dict | None]:
    """The best row that meets T1 and T2 at any rank, and the best if every unmeasured T3 came in under its target."""
    hopeful = sorted(rows, key=lambda r: rank_key(r | {"t3_pass": r["t3_pass"] or r["t3_worst"] is None}))
    return {"t1_t2_any_rank": next((r for r in rows if r["t1_pass"] and r["t2_pass"]), None),
            "if_unmeasured_t3_passes": hopeful[0] if hopeful else None}


def pct(x: float | None) -> str:
    return "?" if x is None else f"{x:.2%}"


def t2s(row: dict) -> str:
    e = row["t2"]["engines"]
    return f"{e['sync']['ratio']:.1f}x / {e['async']['ratio']:.1f}x"


def render(card: dict, results: dict) -> str:
    noise = card["noise"]["ncp"]
    ssn = [v["variant"] for k, v in card["noise"].items() if k.startswith("ssn")]
    lines = ["# PoUW sweep ranking (sweep_price.py)", "",
             "T1: additive gap over stock in W1 units (H32 floor 32 a register, measured on the H200, run "
             "r20261029-021534-a5a3); terms are shares of stock; '+coins' is T1 with the coins' expansion uncredited. "
             "T2: t_H/t_S judged on the faster stock step (unfused: the words exported to HBM). T3: worst of "
             "WikiText-2 2K and 8K over stock int8. Rank: the matmul argument's strongest premise (an open one counts 5 "
             "in the order until settled), then in brackets the overall rank with the hashing credit, ShaSpeed and "
             "hash-compose (rank 4, C27), whose precision delta* is given. NCP here is "
             f"{noise['variant']}, not the tree's ncp-v1; SSN is SW-NOISE's served sparse noise ({', '.join(ssn)}). "
             "Designs are named <noise>-k<L>[-<credit: bam BamFloor, pl PureLanes or CellPure>]-<classes committed>"
             "[-count][-min]-<rows>[-<hiding charge>][-u2].", ""]
    for point, res in results.items():
        pt = card["point"][point]
        lines += [f"## {pt['title']}", "",
                  f"{len(res['rows'])} designs priced, {len(res['refused'])} refused; the top {SHOWN} and the named "
                  "designs (the CSV has every row).", "",
                  "| # | design | rank matmul (overall) | T1 | +coins | final word | uncommitted | forming | lane fix | "
                  "derivation | hiding | T2 sync / async (unfused) | T3 | delta* | open (matmul) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for i, row in enumerate(res["rows"], 1):
            if i > SHOWN and card["design"][row["design"]].get("swept"):
                continue
            t = row["terms"]
            lines.append(
                f"| {i} | {row['design']} | {row['rank']} ({row['rank_with_hash']}) | **{pct(row['t1'])}** "
                f"{'pass' if row['t1_pass'] else 'fail'} | {pct(row['t1_coins_uncredited'])} | {pct(t['final_word'])} | "
                f"{pct(t['uncommitted_steps'])} | "
                f"{pct(t['forming'])} | {pct(t['lane_fix'])} | {pct(t['noise_derivation'])} | {pct(t['hiding'])} | "
                f"{t2s(row)} ({row['t2']['unfused']:.1f}x) {'pass' if row['t2_pass'] else 'fail'} | "
                f"{pct(row['t3_worst'])} | {row['delta_star']:.1e} | {row['matmul_open']}: {opened(row)} |")
        lines += ["", "Refused (an unpriced term or outside the domain):", ""]
        named = {k: v for k, v in res["refused"].items() if not card["design"][k].get("swept")}
        swept = len(res["refused"]) - len(named)
        lines += [f"- {k}: {'; '.join(v)}" for k, v in named.items()] or ["- none named"]
        if swept:
            lines.append(f"- {swept} swept designs (L at or past the stacked depth, or below a call's rows)")
        best = res["rows"][0] if res["rows"] else None
        if best:
            p = best["t2"]["extra_parts_ms"]
            lines += ["", f"**Best: {best['design']}** ({best['title']}). T2's parts a forward: stock "
                          f"{best['t2']['engines']['async']['stock_ms']:.2f} ms (async), "
                          + ", ".join(f"{k} {v:.2f} ms" for k, v in p.items())
                          + f", hashing {best['t2']['hash_ms']:.2f} ms. It still needs:", ""]
            lines += [f"- {n}" for n in needs(best)]
            lines += ["", "If a premise fails:", "", "| refuted | best then | rank | T1 | T2 sync / async | still open |",
                      "|---|---|---|---|---|---|"]
            for drop, b in res["if_refuted"]:
                lines.append(f"| {drop} | - | - | - | - | - |" if b is None else
                             f"| {drop} | {b['design']} | {b['rank']} | {pct(b['t1'])} {'pass' if b['t1_pass'] else 'fail'} | "
                             f"{t2s(b)} | {opened(b)} |")
            alt = alternates(res["rows"])
            if alt["t1_t2_any_rank"] is None:
                lines += ["", "No design meets T1 and T2 together at any rank."]
            for what, b in alt.items():
                if b is not None and b is not best:
                    lines += ["", f"Best {'meeting T1 and T2 at any rank' if what == 't1_t2_any_rank' else 'if every unmeasured T3 passes'}"
                                  f": {b['design']}, rank {b['rank']}, T1 {pct(b['t1'])}, T2 {t2s(b)}, T3 "
                                  f"{pct(b['t3_worst'])}, open: {opened(b)}."]
            lines += [""] + period_table(card, best)
            nxt = next((r for r in res["rows"] if r["t1_pass"] and r["rank"] > best["rank"]), None)
            if nxt:
                lines += ["", f"Best at the next rank: {nxt['design']}, rank {nxt['rank']} "
                              f"({', '.join(a['premise'] for a in nxt['assumed']) or 'no assumed premise'}), T1 "
                              f"{pct(nxt['t1'])}, T2 {t2s(nxt)}."]
        clean = next((r for r in res["rows"] if not r["matmul_open"] and not r["assumed"]), None)
        if clean and clean is not best:
            lines += ["", f"Best with nothing open but the hashing credit: {clean['design']}, T1 {pct(clean['t1'])}, rank "
                          f"{clean['rank']}, T2 {t2s(clean)}.", ""]
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
        results[point] = {"rows": rows, "refused": refused, "if_refuted": if_refuted(card, rows)}
    for point, res in results.items():
        print(f"== {card['point'][point]['title']} ({len(res['rows'])} priced, {len(res['refused'])} refused)")
        for i, row in enumerate(res["rows"][:12], 1):
            t = row["terms"]
            print(f"  {i:2d}. {row['design']:<36} rank {row['rank']} ({row['rank_with_hash']})  T1 {row['t1']:8.3%}  "
                  + " ".join(f"{k[:6]} {v:.3%}" for k, v in t.items() if v)
                  + f"  T2 {t2s(row)}  T3 {pct(row['t3_worst'])}  delta* {row['delta_star']:.1e}  open {opened(row)}")
        for k, v in res["refused"].items():
            if not card["design"][k].get("swept"):
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
        e = row(point, "ncp-k32")["t2"]["core_ncp"]
        got = (e["sync"], e["async"])
        checks.check(f"ncp-k32's T2 on CORE-NCP's parts at {point} reproduces CORE-NCP §6's {want[0]}x / {want[1]}x",
                     all(abs(g / w - 1) < 0.01 for g, w in zip(got, want)), f"{got[0]:.1f}x / {got[1]:.1f}x")
    checks.check("ncp-k32 hashes 0.375 B of k32 words a useful MAC (12/32, CORE-NCP §2)",
                 abs(row("t8192", "ncp-k32")["t2"]["bytes_forward"] / row("t8192", "ncp-k32")["stock_macs_forward"]
                     - (0.375 + 14336 / 60817408)) < 1e-9,
                 f"{row('t8192', 'ncp-k32')['t2']['bytes_forward'] / row('t8192', 'ncp-k32')['stock_macs_forward']:.4f} B "
                 f"(words 0.375 and A's commitment)")

    # (2) CORE-SMELL C28's terms, and the folded notes' figures, as the card prices them.
    pn = row("t8192", "ncp-k32")
    checks.check("hashing A's commitment before the noise is 0.42x stock at T 8,192 (CORE-SMELL C28: 0.42-0.72x), credited "
                 "under the atomic call (SW-SMELL P3), not left as gap", 0.40 < pn["pre_noise_units"] < 0.44
                 and pn["terms"]["pre_noise_hashing"] == 0, f"{pn['pre_noise_units']:.3f}x")
    dv = row("t8192", "ncp-k32")["terms"]["noise_derivation"]
    checks.check("deriving E1 and F1 from XOF bytes is 8/n + 8/m: 0.29% fused at T 8,192 (CORE-SMELL C28)",
                 abs(dv - 0.0029) < 0.0002, f"{dv:.3%}")
    sh = row("t8192", "ncp-k32-xyef-r16k-hr")
    rest = sum(v for k, v in sh["terms"].items() if k != "hashing_slack")
    checks.check("with every honest write forced and hiding ruled, credited hashing is about 666x stock at T 8,192 "
                 "(CORE-SMELL C27: delta* about 0.0015%), and delta* spends what route U's lane fix leaves of 1%",
                 660 < sh["hash_units_per_mac"] < 680 and abs(sh["delta_star"] * sh["hash_units_per_mac"] + rest - 0.01)
                 < 1e-12, f"delta* {sh['delta_star']:.2e}, {sh['hash_units_per_mac']:.0f} units a MAC, other terms {rest:.3%}")
    for point, want in (("t8192", 0.024), ("t512", 0.520)):
        t = row(point, "ncp-k32")["terms"]
        got = t["final_word"] + t["forming"] + t["noise_derivation"]
        checks.check(f"on settled premises ncp-k32's final word, forming and derivation are SW-SMELL P3's {want:.1%} at "
                     f"{point}", abs(got - want) < 0.001, f"{got:.2%}")
    lu, lm = row("t512", "ncp-k32-w-r16k-hr"), row("t8192", "ncp-k32-w-r16k-hr-u2")
    checks.check("route U's Y lane fix is 24/m (SW-SMELL S9): 37.5% of stock at 64 rows; the mixed route stays under route "
                 "U's 0.293% at T 8,192", abs(lu["terms"]["lane_fix"] - 0.375) < 1e-12
                 and lm["terms"]["lane_fix"] <= 24 / 8192 + 1e-12, f"{lu['terms']['lane_fix']:.2%}, mix {lm['terms']['lane_fix']:.3%}")
    hk = row("t8192", "ncp-k32-w-r16k")
    words_hide = hk["terms"]["hiding"] - 0.0553 * hk["pre_noise_units"]
    checks.check("the salt side's hiding on k32 words at 16 KiB rows lies between SW-HASH §7's floor and alone readings "
                 "(20.8x and 41.6x stock)", 20.7 < words_hide < 41.7, f"{words_hide:.1f}x")
    hb = card["price"]["extract"]["units_per_clock"] * 1980e6 / card["price"]["hash_rate"]["value"]
    checks.check("T2's hash rate at 16 KiB rows is SW-GPU final's measured 2,043-2,159 units a byte at 1,980 MHz alone "
                 "(r20261029-042705-c421, -045936-22a0, -050641-3a03, -051426-adb8)", 2043 <= hb <= 2159, f"{hb:.0f}")
    checks.check("coinsncp's interpolation returns the measured times at measured sizes, and the smallest call's time below",
                 abs(coinsncp_ms(card, 16777216) - 0.061657) < 1e-12 and coinsncp_ms(card, 1000) == 0.011485, None)
    a = call_units(card, "ncp-k32-w-r256k-hr", 64, 3072, 2048, 1, 510.48e9)
    checks.check("A's root at 64 rows by activation rows: 16 + 1 + 2 + 3·6 + 1 = 38 compressions before the noise",
                 a["a_chain"] == 38, f"{a['a_chain']:g}")

    swept = [n for n, d in card["design"].items() if d.get("swept")]
    checks.check("the sweep generates, for NCP, every L (L > 32 under each of 2 credits) x committed classes x forming "
                 "charge x derivation charge x layout x hiding charge x route, less a charge beside its class committed: "
                 "(1 + 6 x 2) x 9 x 3 x 4 x 2 = 2,808; for the two SSNs 2 x 2 x 3 x 4 = 48; for per2 3 x 3 x 4 = 36",
                 len(swept) == 2892, f"{len(swept)} swept")
    pw, px = row("t8192", "per2-k32-w-r16k-hr"), row("t8192", "per2-k32-xe-r16k-hr")
    pr = [call_units(card, "per2-k32-w-r16k-hr", 64, 2048, K)["r"] for K in (2048, 8192)]
    checks.check("per2 under the final-word lemma reproduces SW-NOISE final §3's 0.190% without candidate 1 (8/n + "
                 "256/(nK), plus V's derivation, 256/(nK)) and 0 with X and V committed, at widths 1 + 32·12/2,048 and "
                 "1 + 32·21/8,192 (its measured limbs)",
                 abs(pw["t1"] - 0.0019) < 0.00003 and px["t1"] < 1e-12 and pw["terms"]["final_word"] == 0
                 and abs(pr[0] - 1.1875) < 1e-12 and abs(pr[1] - (1 + 672 / 8192)) < 1e-12,
                 f"T1 {pw['t1']:.4%}, with C1 {px['t1']:.4%}, widths {pr[0]:.4f} / {pr[1]:.4f}")
    for nz, want in (("ssn-1l", (0.00049, 0.00025)), ("ssn-a", (0.00054, 0.00027))):
        s, se = row("t8192", f"{nz}-k32-w-r16k-hr"), row("t8192", f"{nz}-k32-e-r16k-hr")
        checks.check(f"{nz} with hiding ruled reproduces SW-NOISE final §3's {want[0]:.3%} without candidate 1 and "
                     f"{want[1]:.3%} with it at T 8,192 (a SHF and a LOP3 a row a step, the LOP3 forced when committed), "
                     "with no final word, forming, lane fix or uncommitted step",
                     abs(s["t1"] - want[0]) < 0.000006 and abs(se["t1"] - want[1]) < 0.000006
                     and s["terms"]["final_word"] == 0 and s["terms"]["forming"] == 0 and s["terms"]["lane_fix"] == 0
                     and abs(s["terms"]["uncommitted_steps"]) < 1e-12, f"T1 {s['t1']:.4%}, with C1 {se['t1']:.4%}")
    widths = {}
    for nz, want in (("ssn-1l", ((2144 / 2048, 70), (8480 / 8192, 268))), ("ssn-a", ((2368 / 2048, 296), (9376 / 8192, 1172)))):
        got = [call_units(card, f"{nz}-k32-w-r16k-hr", 64, 2048, K) for K in (2048, 8192)]
        widths[nz] = [(u["r"], u["coin_bytes"] / card["price"]["coin_bytes"]["value"] * 8 / 64) for u in got]
        checks.check(f"{nz}'s stacked width and coin bits a row a call are SW-NOISE final §2's: "
                     f"{want[0][0]:.4f} and {want[0][1]} at K 2,048, {want[1][0]:.4f} and {want[1][1]} at K 8,192",
                     all(abs(g[0] - w[0]) < 1e-12 and abs(g[1] - w[1]) < 1e-9 for g, w in zip(widths[nz], want)),
                     f"{widths[nz]}")
    b64 = row("t8192", "ncp-k64-bam-xyef-r16k-hr")
    checks.check("a word every 64 steps at rank <= 3 is credited BamFloor's 48 units, not 64 (SW-SEM): r(1 - 48/64) = 75% "
                 "of stock uncommitted, so it fails T1", abs(b64["terms"]["uncommitted_steps"] - 0.75) < 1e-9
                 and not b64["t1_pass"] and b64["rank"] == 3, f"{b64['terms']['uncommitted_steps']:.2%}, rank {b64['rank']}")
    checks.check("the 05:05Z best (ncp-k64-w-count-min on the k64 merge charge) is refused: SW-OLD final refutes it",
                 all("ncp-k64-merge-0505" in res["refused"] for res in results.values())
                 and "refuted" in results["t8192"]["refused"]["ncp-k64-merge-0505"][0],
                 results["t8192"]["refused"].get("ncp-k64-merge-0505", ["ranked"])[0][:70])
    s256, s63, s126 = (fast_saving(card, L, 8192, 8192, om) for L, om in ((256, 63), (512, 63), (512, 126)))
    checks.check("Strassen-Winograd on operands within +-63 saves nothing against the k32 counting at L 256 "
                 "(1/8 - 56/256 < 0) and 1/8 - 56/512 - 8(2/8192) = 1.37% at L 512, m = n = 8,192; on NCP's +-126 "
                 "operands its sums leave s8 and it spends 18/8 of the steps, saving nothing (P1's floor)",
                 s256[0] <= 0 and abs(s63[0] - (1 / 8 - 56 / 512 - 16 / 8192)) < 1e-12 and s63[1] == 1 and s126[0] == 0,
                 f"+-63: L 256 {s256[0]:.2%}, L 512 {s63[0]:.3%}; +-126: L 512 {s126[0]:.2%}")
    r64, r128, r256 = (row("t8192", f"ncp-k{L}-pl-xyef-r16k") for L in (64, 128, 256))
    checks.check("crediting a word every L > 32 with L units rests on PureLanes or CellPure, rank 4 (SW-SEM final), and "
                 "only to L 128: at L 256 Hopcroft-Kerr wins by 12.5%, so the word is credited BamFloor's 60 units and "
                 "3(1 - 60/256) = 229.7% of stock stays uncommitted",
                 r64["rank"] == 4 and r128["rank"] == 4 and r64["terms"]["uncommitted_steps"] == 0
                 and r128["terms"]["uncommitted_steps"] == 0
                 and abs(r256["terms"]["uncommitted_steps"] - 3 * (1 - 60 / 256)) < 1e-9,
                 f"k64 rank {r64['rank']}, k128 rank {r128['rank']}, k256 uncommitted {r256['terms']['uncommitted_steps']:.1%}")
    fewest = {}
    for p, res in results.items():
        b = res["rows"][0]
        peers = [r for r in res["rows"] if r["t1_pass"] and r["t2_pass"] and r["t3_pass"] == b["t3_pass"]
                 and r["rank"] == b["rank"]]
        fewest[p] = not b["t2_pass"] or len(b["open"]) == min(len(r["open"]) for r in peers)
    checks.check("the best meeting T1 and T2 at its rank takes the fewest unsettled premises of the rows that do with its "
                 "T3 verdict, whatever their T2 (SW-SMELL P1: an open premise is not free; an unmeasured T3 doesn't pass)",
                 all(fewest.values()), f"{fewest}")
    passing = {p: [r for r in res["rows"] if r["t1_pass"] and card["design"][r["design"]]["noise"] == "ncp"]
               for p, res in results.items()}
    checks.check("at T 512's 64 rows every NCP design passing T1 takes route U' (route-u2): route U's Y lane fix alone is "
                 "37.5% (SW-SMELL S9)", passing["t512"] and all("route-u2" in r["premises"] for r in passing["t512"]),
                 f"{len(passing['t512'])} pass")
    checks.check("at T 8,192 every NCP design passing T1 takes hiding-ruled or mask-optimal: the salt side's M_key·y "
                 "alone costs over 1% of stock in every layout (SW-HASH §7)",
                 passing["t8192"] and all({"hiding-ruled", "mask-optimal"} & set(r["premises"]) for r in passing["t8192"]),
                 f"{len(passing['t8192'])} pass")

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

        few = ["ncp-k32", "ncp-k32-xyef-r16k", "ncp-k32-xy-r16k", "ncp-k32-w-r16k", "ncp-k64-bam-w-count-min-r256k-hr-u2"]
        c = mutated(lambda c: c["noise"]["ncp"].pop("formed_y"))
        checks.mutation("an NCP card without its weight-side forming count is refused, not priced at 0",
                        all(n in ranking(c, "t8192", names=few)[1] for n in ("ncp-k32", "ncp-k32-xyef-r16k")),
                        ranking(c, "t8192", names=few)[1].get("ncp-k32", ["priced"])[0])
        c = mutated(lambda c: c["noise"]["ncp"]["lane_fix"].pop("U"))
        checks.mutation("an NCP card without route U's lane fix refuses route U's designs, not prices them at 0",
                        "ncp-k32" in ranking(c, "t8192", names=few)[1], None)
        c = mutated(lambda c: c["design"]["ncp-k32-xyef-r16k"]["commit"].__setitem__("word_bits", 16))
        checks.mutation("a 16-bit committed word is refused: no premise prices it",
                        "ncp-k32-xyef-r16k" in ranking(c, "t8192", names=few)[1], None)
        c = mutated(lambda c: c["premise"]["commit-forced"].__setitem__("status", "refuted"))
        checks.mutation("a refuted commit-forced (SW-LEAN's counterexample) refuses every design that takes it",
                        all(n in ranking(c, "t512", names=few)[1] for n in ("ncp-k32-xy-r16k", "ncp-k32-xyef-r16k")), None)
        lr, _ = ranking(card, "t512", lenient=True, names=["dither-c", "fresh-u", "accel-d5-d6", "ncp-k32"])
        top = [r["design"] for r in lr if r["t1_pass"]]
        checks.mutation("read '?' as 0 and CORE-RANK's unpriced designs enter T1's passing tier: the refusal is what keeps "
                        "them out", any(n in top for n in ("dither-c", "fresh-u", "accel-d5-d6")), f"passing tier {top}")
        nosem = row("t8192", "ncp-k64-nosem")
        checks.mutation("committing every 64 steps without the k32 semantics leaves r(1 - 32/64) = 150% of stock "
                        "uncommitted: the density needs candidate 2", nosem["terms"]["uncommitted_steps"] > 1.4,
                        f"{nosem['terms']['uncommitted_steps']:.0%}")
        c = mutated(lambda c: c["noise"]["ncp"].update(formed_x="m*K", formed_y="n*K"))
        a = evaluate(c, "ncp-k32-w-r16k", "t8192")["terms"]["forming"]
        b = row("t8192", "ncp-k32-w-r16k")["terms"]["forming"]
        checks.mutation("forming one copy a side (the certificate's 8/n + 8/m read over stock) would be a third of NCP's "
                        "24/n + 24/m: the stack's 3 copies decide the term", abs(3 * a - b) < 1e-12 and b > 0.0085,
                        f"{a:.3%} vs {b:.3%}")
        d1 = evaluate(card, "ncp-k32-xyef-r16k-hr", "t8192", delta=1e-4)
        checks.mutation("a ShaSpeed slack of 0.01% breaks T1 for the all-forced design: hashing's work is 666x stock",
                        not d1["t1_pass"], f"T1 {d1['t1']:.2%}")
        c = mutated(lambda c: c["premise"]["pure-lanes"].__setitem__("word_units", "reg*L/32"))
        e, e0 = evaluate(c, "ncp-k1024-pl-xyef-r256k-hr", "v4like"), row("v4like", "ncp-k1024-pl-xyef-r256k-hr")
        checks.mutation("credit PureLanes' full L at every L and ncp-k1024-pl-xyef passes T1 on V4-like (as it topped V4 "
                        "under NoSharingGain): the cap at L 128 from SW-SEM's margins keeps it out",
                        e["t1_pass"] and not e0["t1_pass"], f"T1 {e['t1']:.2%} uncapped, {e0['t1']:.0%} capped")
        c = mutated(lambda c: c["noise"]["ssn-1l"].__setitem__("derive_extra", "0"))
        e, e0 = evaluate(c, "ssn-1l-k32-e-r16k-hr", "t8192"), row("t8192", "ssn-1l-k32-e-r16k-hr")
        checks.mutation("drop SSN's shift and committing its noise registers would close the derivation to 0: the SHF no "
                        "commitment holds is what stays", e["terms"]["noise_derivation"] == 0
                        and e0["terms"]["noise_derivation"] > 0, f"{e0['terms']['noise_derivation']:.4%} with the SHF")
        c = mutated(lambda c: c["price"]["hash_rows"].__setitem__("hide_share", [0.0] * 4) or
                    c["price"]["hash_rows"].__setitem__("mask_share", [0.0] * 4))
        e = evaluate(c, "ncp-k64-pl-w-count-min-r256k-salt-u2", "t8192")
        checks.mutation("a card that prices the salt side at nothing lets a design without hiding-ruled or mask-optimal "
                        "pass T1 at T 8,192: the hiding term is what keeps it out", e["t1_pass"], f"T1 {e['t1']:.2%}")
        c = mutated(lambda c: c["point"]["t512"].__setitem__("tokens", 16))
        rows16, ref16 = ranking(c, "t512", names=few)
        checks.mutation("a decode batch of 16 rows is refused as outside the domain (C15), not priced",
                        not rows16 and all("domain" in ref16[n][0] for n in few), ref16[few[0]][0][:60])

    out.mkdir(parents=True, exist_ok=True)
    md = render(card, results)
    (out / "ranking.md").write_text(md + "\n")

    def kept(rows: list[dict]) -> list[dict]:
        return [r for i, r in enumerate(rows) if i < 50 or not card["design"][r["design"]].get("swept")]

    blob = {"schema": "sw-price/result/v2", "card": {k: v for k, v in card.items() if k != "design"},
            "designs_named": {k: v for k, v in card["design"].items() if not v.get("swept")}, "delta": delta,
            "points": {p: {"rows": kept(res["rows"]), "rows_total": len(res["rows"]),
                           "refused_named": {k: v for k, v in res["refused"].items()
                                             if not card["design"][k].get("swept")},
                           "refused_swept": len([k for k in res["refused"] if card["design"][k].get("swept")]),
                           "best": res["rows"][0]["design"] if res["rows"] else None,
                           "best_needs": needs(res["rows"][0]) if res["rows"] else [],
                           "alternates": {k: b and {"design": b["design"], "rank": b["rank"], "t1": b["t1"],
                                                    "t2": b["t2"]["judged"], "t3": b["t3_worst"], "needs": needs(b)}
                                          for k, b in alternates(res["rows"]).items()},
                           "best_periods": periods(card, res["rows"][0]) if res["rows"] else None,
                           "if_refuted": [{"refuted": drop, "best": b and b["design"], "t1": b and b["t1"],
                                           "t2": b and b["t2"]["judged"], "open": b and [o["premise"] for o in b["open"]]}
                                          for drop, b in res["if_refuted"]]}
                       for p, res in results.items()}}
    (out / "sweep_price.json").write_text(json.dumps(blob, indent=1, default=str) + "\n")
    with (out / "ranking.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["point", "position", "design", "rank", "rank_effective", "rank_with_hash", "t1",
                    "t1_coins_uncredited", *TERMS, "t2_sync", "t2_async", "t2_unfused", "t3_worst", "delta_star",
                    "hash_bytes_per_mac", "open"])
        for p, res in results.items():
            for i, r in enumerate(res["rows"], 1):
                e = r["t2"]["engines"]
                w.writerow([p, i, r["design"], r["rank"], r["rank_effective"], r["rank_with_hash"], f"{r['t1']:.6g}",
                            f"{r['t1_coins_uncredited']:.6g}",
                            *(f"{r['terms'][k]:.6g}" for k in TERMS), f"{e['sync']['ratio']:.4g}",
                            f"{e['async']['ratio']:.4g}", f"{r['t2']['unfused']:.4g}",
                            "" if r["t3_worst"] is None else f"{r['t3_worst']:.4g}",
                            f"{r['delta_star']:.4g}", f"{r['hash_bytes_per_mac']:.4g}",
                            " ".join(o["premise"] for o in r["open"])])
    print(f"wrote {out / 'ranking.md'}, {out / 'sweep_price.json'}, {out / 'ranking.csv'}")
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
