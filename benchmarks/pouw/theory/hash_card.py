"""The HASH card: every design that reduces PoUW's hashing, on one basis with the lead as its baseline.

h is the honest program's credited commitment work over stock, in W1 units a useful MAC (a register write 32, a stock
MAC 1): SHA-512 at ShaWrites' constant per form (pouw_space's `price.sha_forms`), plus any non-hash commitment work a
design adds. Beside h each design carries T1, T2, T3 and the rank of its assumptions; the frontier is the least h at each
rank among the designs meeting T1 <= 1%, T2 <= 100x and T3 <= 5%.

* The lead (`hash_card.toml` [basis].lead) is composed and priced by pouw_space (`compose`, `evaluate`), at the
  basis's Llama-3.2-1B points, and its h decomposed by part (words, operands, A, coins, bound blocks, salts), each at
  its own form's price. Its measured T2 and T3 are carried from the lanes.
* The readings in circulation (233, 119.5) are reproduced from their designs and prices.
* A design is priced by pouw_space when it composes (`compose` or `swap` on the lead): checked. Otherwise its h is
  taken as its lane states it and converted to the basis's price; every conversion, and every model, point, chunk,
  baseline or label that differs from the basis, is a flag.
* The rank axis is the larger of the design's hashing part (pouw_space's `ranks`) and every premise it adds; the
  matmul part's floor, common to every design that keeps the lead's noise, is reported beside it.

    python3 hash_card.py [--mutations] [--verdict PATH] [--out DIR] [--data hash_card.toml]

Writes hash_card.md, hash_card.json and hash_card.csv under --out.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pouw_space as ps  # noqa: E402
from exact_kit.checks import Checks  # noqa: E402

DATA = HERE / "hash_card.toml"
TARGET = ps.TARGET
SLOTS = ("noise", "denoise", "format", "density", "operands", "rows", "hiding", "final", "forming", "derivation")
GONE = ("refuted", "closed")
LABELS = ("proved", "checked", "measured", "estimate")
REQUIRE_FALSIFICATION = True
FOLD_WRITES = True
GRANT_ANY_REFUTED = False


def load(path: Path = DATA) -> dict:
    return tomllib.loads(path.read_text())


def choice(hc: dict, swap: dict | None = None, full: dict | None = None) -> dict:
    out = dict(full or hc["basis"]["lead"])
    out.update(swap or {})
    return {s: out[s] for s in SLOTS}


def word_price(card: dict, row: float) -> float:
    """W1 a words byte at a chunk of `row` bytes: ShaWrites' R for the chunk (pouw_space's fit) x reg / 128."""
    sha, reg = card["price"]["sha_forms"], card["price"]["reg"]["value"]
    return (sha["chain"] - sha["chunk_k"] / (row / ps.BLOCK + sha["chunk_extra"])) * reg / ps.BLOCK


def decompose(card: dict, d: dict, point: str, bound: bool = True) -> dict:
    """h at the point by part, each at its form's price, as pouw_space's `evaluate` sums it: the words and operands at
    their chunk's R, A's pre-noise bytes at its leaf's, coins and salts at a chained compression's, the bound blocks
    at a compression from a constant chaining value. Returns W1 a useful MAC by part and the credited bytes a MAC."""
    price = card["price"]
    hw, iv, floor = price["hash_work"]["value"], price["hash_work_iv"]["value"], price["hash_work_floor"]["value"]
    a_price = price["sha_forms"]["a_leaf"] * price["reg"]["value"] / ps.BLOCK
    rate = card["point"][point].get("hash_rate", price["hash_rate"]["value"])
    parts = dict.fromkeys(("words", "operands", "A", "coins", "bound", "salts"), 0.0)
    stock = credited = 0.0
    for x in ps.calls(card, point):
        u = ps.call_units(card, d, x["m"], x["n"], x["K"], x["count"] / x["serial"], rate)
        f = d["fields"].at(x["m"], x["n"], x["K"])
        w, off = x["count"], bool(f["offload"])
        closes = ps.effects(card, d["premises"], f["L"])[0]
        pre_cred = 0.0 if off or "pre_noise_hashing" not in closes else u["pre_noise_bytes"]
        hw_row = word_price(card, f["row"])
        b = u["bound_bytes"] if bound else 0.0
        salt_b = u["credited_bytes"] - u["word_bytes"] - u["operand_bytes"] - u["coin_bytes"] - u["bound_bytes"] - pre_cred
        if not off:
            parts["words"] += w * u["word_bytes"] * hw_row
            parts["operands"] += w * u["operand_bytes"] * hw_row
            parts["A"] += w * pre_cred * a_price
            parts["coins"] += w * u["coin_bytes"] * hw
            parts["bound"] += w * b * iv
            parts["salts"] += w * salt_b * hw
        stock += w * u["stock"]
        credited += w * u["credited_bytes"]
    fs = d.get("forward_salts", 0) if d.get("layout", "l0") != "l0" else 0
    if fs:
        closes = ps.effects(card, d["premises"], 32)[0]
        (sw,), _ = ps.salt_units(card, d.get("salt_basis", "w1"), [1.0], d.get("salt"))
        if "hiding_salt" in closes and "hiding" not in closes:
            parts["salts"] += fs * sw / floor * hw
            credited += fs * sw / floor
    out = {k: v / stock for k, v in parts.items()}
    out["h"] = sum(out.values())
    out["bytes_per_mac"] = credited / stock
    return out


def priced(card: dict, hc: dict, ch: dict) -> dict:
    """A composed design at every basis point: pouw_space's row, h's parts, and the worst of T1, T2 and T3."""
    d = ps.compose(card, ch)
    rows, parts, refused = {}, {}, {}
    for p in hc["basis"]["points"]:
        try:
            rows[p] = ps.evaluate(card, d, p)
            parts[p] = decompose(card, d, p)
        except ps.Refused as e:
            refused[p] = e.reasons
    worst = {}
    if rows:
        worst = {"h": max(r["hash_units_per_mac"] for r in rows.values()),
                 "t1": max(r["t1"] for r in rows.values()),
                 "t2": max(r["t2"]["judged"] for r in rows.values()),
                 "t3": max(r["t3_worst"] for r in rows.values()) if all(r["t3_worst"] is not None for r in rows.values())
                 else None}
    return {"design": d, "name": d["name"], "rows": rows, "parts": parts, "refused": refused, "worst": worst}


def with_premises(card: dict, adds: list[dict]) -> dict:
    c = copy.copy(card)
    c["premise"] = dict(card["premise"]) | {a["id"]: {"rank": a["rank"], "part": a.get("part", "hashing"),
                                                      "status": a.get("status", "open"), "source": a.get("source", "")}
                                            for a in adds}
    return c


def rank_axis(card: dict, premises: list[str], adds: list[dict], with_adds: bool = True) -> dict:
    """The design's ranks: pouw_space's by part, and the axis, the larger of the hashing part and every premise added."""
    c = with_premises(card, adds)
    ids = list(dict.fromkeys(premises + [a["id"] for a in adds]))
    r = ps.ranks(c, ids)
    added = max((ps.rank_value(a["rank"]) for a in adds), default=0.0) if with_adds else 0.0
    r["axis"] = max(r["hashing"], added)
    r["axis_effective"] = max(r["hashing_effective"], 5.0 if any(a.get("status", "open") in ps.UNSETTLED for a in adds)
                              else added)
    return r


def stated_h(hc: dict, card: dict, h: dict, convert: bool = True) -> tuple[float, list[str]]:
    """h as a lane states it, on the basis: converted from its price (and its words chunk, at the per-form price), with
    a flag for each conversion and each basis field that differs."""
    flags, value = [], float(h["value"])
    base_row = hc["basis"]["words_chunk"]
    pr = hc["price"][h.get("price", "w1-forms")]
    if h.get("price", "w1-forms") != "w1-forms":
        factor = word_price(card, base_row) / pr["per_byte"]
        flags.append(f"h stated at {h['price']} ({pr['kind']}, {pr['per_byte']:,} W1 a byte); x{factor:.3f} to the basis")
        if convert:
            value *= factor
    chunk = h.get("words_chunk", base_row)
    if chunk and chunk != base_row:
        factor = word_price(card, base_row) / word_price(card, chunk)
        flags.append(f"words in {chunk // 1024} KiB chunks, not {base_row // 1024} KiB: R moves {factor - 1:+.3%}")
        if convert:
            value *= factor
    if h.get("model", hc["basis"]["model"]) not in (hc["basis"]["model"], "any"):
        flags.append(f"h on {h['model']}, not {hc['basis']['model']}")
    if h.get("point", "any") not in hc["basis"]["points"] + ["any"]:
        flags.append(f"h at point {h['point']}, not one of {', '.join(hc['basis']['points'])}")
    if h.get("label", "estimate") == "estimate":
        flags.append("h is an estimate")
    return value, flags


def derived(card: dict, hc: dict, lead: dict, spec: dict, fold_writes: bool = True) -> dict:
    """A design the card derives from the lead's priced parts at every basis point: `combine` commits one register for m
    words (xor-m; xorrot-m with `rot`), `fold` folds every word into its cell's state and seals the state with the
    cell's final word (`seal_bytes_per_cell`). The words' hashing falls to the committed share `keep`, T1's chunk-head
    term with it; the fold's own register writes are added to h at W1; `t1_dw` is T1 with the uncommitted words and
    the fold's writes uncredited (DistinctWrites alone)."""
    pr = priced(card, hc, choice(hc, spec.get("swap")))
    cm, fo, mpw = spec.get("combine"), spec.get("fold"), lead["macs_per_word"]
    rows = {}
    for p in hc["basis"]["points"]:
        q, r = pr["parts"][p], pr["rows"][p]
        if cm:
            m = cm["m"]
            keep = 1 / m
            writes = cm.get("writes_per_word", (math.ceil((m - 1) / 2) + cm.get("rot", 0) * (m - 1)) / m)
        else:
            cs = ps.calls(card, p)
            cells = sum(x["count"] * x["m"] * x["n"] for x in cs)
            macs = sum(x["count"] * x["m"] * x["n"] * x["K"] for x in cs)
            keep, writes = fo["seal_bytes_per_cell"] * cells / macs / (4 / mpw), fo["writes_per_word"]
        fold = writes * 32 / mpw if fold_writes and FOLD_WRITES else 0.0
        t1 = r["t1"] - r["terms"]["chunk_head"] * (1 - keep)
        rows[p] = {"h": q["h"] - q["words"] * (1 - keep) + fold, "fold": fold, "keep": keep, "t1": t1,
                   "t1_dw": t1 + ((1 - keep) + writes) * 32 / mpw}
    return {"design": pr["design"], "name": pr["name"], "rows": rows,
            "worst": {k: max(r[k] for r in rows.values()) for k in ("h", "fold", "t1", "t1_dw")}}


def family_rank(hc: dict, add: dict) -> tuple[object, str | None]:
    """A premise two lanes state under different names is one family, ranked at the most conservative rank stated."""
    fam = hc.get("family", {}).get(add.get("family", ""))
    if not fam:
        return add["rank"], None
    top = max((s["rank"] for s in fam["stated"]), key=ps.rank_value)
    if ps.rank_value(top) > ps.rank_value(add["rank"]):
        said = ", ".join(f"{s['name']} {s['rank']} ({s['lane']})" for s in fam["stated"])
        return top, f"{add['id']} is family {add['family']}, stated {said}: ranked {top}"
    return add["rank"], None


def design_entry(card: dict, hc: dict, key: str, spec: dict, lead: dict, kind: str, convert: bool = True) -> dict:
    """One design on the basis: h, cut against the lead, T1/T2/T3 with labels, ranks, status and flags."""
    flags: list[str] = []
    adds = []
    refuted, refuting = [], []
    for a in spec.get("adds", []):
        rank, why = family_rank(hc, a)
        fam = hc.get("family", {}).get(a.get("family", ""), {})
        if fam.get("status") == "refuted":
            refuting.append(fam)
        adds.append(a | {"rank": rank} | ({"status": "refuted"} if fam.get("status") == "refuted" else {}))
        if why:
            flags.append(why)
        if a.get("rank_stated"):
            flags.append(f"{a['id']}: its lane states rank {a['rank_stated']}, not on the card's scale; carded at {rank}")
        if fam.get("status") == "refuted":
            refuted.append(f"{a['id']} (family {a['family']}) refuted: {fam['refuted_by']}"
                           + (f"; it would stand if {fam['stands_if']}" if fam.get("stands_if") else ""))
    flags += refuted
    e = {"id": key, "kind": kind, "title": spec.get("title", key), "lane": spec.get("lane", ""),
         "note": spec.get("note", ""), "status": spec.get("status", "proposed"), "rule_change": spec.get("rule_change", ""),
         "what": spec.get("what", ""), "extra_w1_per_mac": float(spec.get("extra_w1_per_mac", 0.0)), "adds": adds,
         "rule_question": spec.get("rule_question", "")}
    e["stands_if"] = ([f.get("stands_if", "") for f in refuting] if refuting and e["status"] not in GONE
                      and (GRANT_ANY_REFUTED or all(f.get("stands_if") for f in refuting)) else [])
    if refuted and e["status"] not in GONE:
        e["status"] = "refuted"
    if "combine" in spec or "fold" in spec:
        dv = derived(card, hc, lead, spec)
        w = dv["worst"]
        e["design"] = dv["name"] + "+" + spec.get("form", ("xorrot" if spec.get("combine", {}).get("rot") else "xor") + (
            f"-{spec['combine']['m']}" if "combine" in spec else "-fold"))
        e["h"] = w["h"] + e["extra_w1_per_mac"]
        e["h_label"] = spec.get("h_label", "checked" if "combine" in spec else "estimate")
        e["t1"] = (w["t1"], spec.get("t1_label", "checked" if "combine" in spec else "estimate"))
        e["t1_dw"] = w["t1_dw"]
        e["fold_w1"] = w["fold"]
        e["t3"] = (lead["t3_measured"], "measured")
        for t in ("t2", "t3"):
            if t in spec:
                e[t] = (float(spec[t]["value"]), spec[t].get("label", "estimate"))
        prem = [p for p in lead["premises"] if p not in spec.get("drops", [])]
        e["ranks"] = rank_axis(card, prem, adds)
        st = spec.get("stated", {})
        if "h" in st and abs(e["h"] - st["h"]) > 0.01 * max(st["h"], 1.0):
            flags.append(f"its lane states h {st['h']:g}; the card's {e['h']:.2f} counts the fold's own writes "
                         f"({w['fold']:.2f} W1 a MAC)" + (f" and seals {spec['fold']['seal_bytes_per_cell']:g} B a cell "
                                                          f"over the forward" if "fold" in spec else "") if w["fold"] else
                         f"its lane states h {st['h']:g}, the card {e['h']:.2f}")
        if "t1" in st and abs(e["t1"][0] - st["t1"]) > max(0.1 * st["t1"], 2e-6):
            flags.append(f"its lane states T1 {st['t1']:.4%} (worst), the card {e['t1'][0]:.4%}")
    elif "compose" in spec or "swap" in spec:
        try:
            pr = priced(card, hc, choice(hc, spec.get("swap"), spec.get("compose")))
        except ps.Refused as err:
            pr = {"name": "/".join(str(v) for v in spec.get("swap", {}).values()), "rows": {}, "refused": {"all": err.reasons},
                  "worst": {}, "design": None}
        e["design"] = pr["name"]
        if pr["refused"]:
            e["refused"] = {p: r[0][:200] for p, r in pr["refused"].items()}
        if pr["worst"]:
            w = pr["worst"]
            e["h"], e["h_label"] = w["h"] + e["extra_w1_per_mac"], "checked"
            e["t1"], e["t2"], e["t3"] = (w["t1"], "checked"), (w["t2"], "estimate"), (w["t3"], "measured")
            prem = pr["design"]["premises"]
            e["ranks"] = rank_axis(card, prem, adds)
            e["attacks"] = sorted({a for r in pr["rows"].values() for a in r["attacks"]})
        else:
            e["h"], e["h_label"] = None, "refused"
    else:
        if "bytes_per_mac" in spec:
            e["h"] = spec["bytes_per_mac"] * word_price(card, hc["basis"]["words_chunk"]) + e["extra_w1_per_mac"]
            e["h_label"] = spec.get("label", "estimate")
        elif "h" in spec and "ratio" in spec["h"]:
            base = spec["h"].get("baseline")
            e["h"] = spec["h"]["ratio"] * (lead["h"] if convert else float(base)) + e["extra_w1_per_mac"]
            e["h_label"] = spec["h"].get("label", "estimate")
            flags.append(f"stated as h / h_lead = {spec['h']['ratio']:.4g} against its own lead (h = {base:g}); applied "
                         f"to the basis lead's {lead['h']:.2f}, which holds only while the words are its h")
            if spec["h"].get("label", "estimate") == "estimate":
                flags.append("h is an estimate")
        elif "h" in spec:
            v, f = stated_h(hc, card, spec["h"], convert)
            if spec.get("per_byte"):
                v *= lead["bytes_per_mac"]
                flags.append(f"stated a committed byte; x{lead['bytes_per_mac']:.5f}, the lead's credited bytes a MAC")
            e["h"], e["h_label"] = v + e["extra_w1_per_mac"], spec["h"].get("label", "estimate")
            flags += f
            base = spec["h"].get("baseline")
            if isinstance(base, (int, float)) and abs(base / lead["h"] - 1) > 0.01:
                flags.append(f"its cut is against h = {base:g}, not the lead's {lead['h']:.1f} on the basis")
        else:
            e["h"], e["h_label"] = None, "unpriced"
        for t in ("t1", "t2", "t3"):
            if t in spec:
                e[t] = (float(spec[t]["value"]), spec[t].get("label", "estimate"))
        prem = [p for p in lead["premises"] if p not in spec.get("drops", [])]
        e["ranks"] = rank_axis(card, prem, adds)
    if e.get("h") is not None and e["h"] > 0:
        e["cut"] = lead["h"] / e["h"]
    elif e.get("h") == 0:
        e["cut"] = math.inf
    for t in ("t1", "t2", "t3"):
        if t not in e:
            flags.append(f"{t.upper()} unpriced")
        elif e[t][0] is not None and e[t][1] == "estimate" and t == "t2" and kind == "proposal":
            flags.append("T2 an estimate; the lead's is measured")
    mpw = spec.get("layout", {}).get("macs_per_word")
    if mpw and abs(mpw / lead["macs_per_word"] - 1) > 0.02:
        flags.append(f"its lead's layout is {mpw:g} useful MACs a word, the basis lead's {lead['macs_per_word']:.2f}")
    if e["rule_change"]:
        flags.append(f"needs a rule change: {e['rule_change']}")
    if spec.get("rule_question"):
        flags.append(f"rule question open: {spec['rule_question']}")
    flags += spec.get("caveats", [])
    for a in adds:
        if a.get("status", "open") in ps.UNSETTLED and not a.get("falsification"):
            flags.append(f"premise {a['id']} has no falsification test")
    e["passes"] = {t: (e[t][0] <= TARGET[t]) if t in e and e[t][0] is not None else None for t in ("t1", "t2", "t3")}
    e["meets"] = all(v is True for v in e["passes"].values())
    e["flags"] = flags
    e["done_missing"] = done_missing(e)
    return e


def done_missing(e: dict) -> list[str]:
    """What the brief's done-when still needs of a design: a 10x cut holding T1, T2 and T3, its credit checked on CPU,
    its cost measured or bounded, and a falsification test for each premise it adds."""
    out = []
    if not (e.get("cut") and e["cut"] >= 10):
        out.append("cut under 10x")
    if not e["meets"]:
        out.append("misses " + ", ".join(t.upper() for t, v in e["passes"].items() if v is not True))
    if e.get("t1") and e["t1"][1] not in ("checked", "proved", "measured"):
        out.append("credit (T1) not checked")
    if not (e.get("t2") and e["t2"][1] in ("measured", "bound")):
        out.append("cost (T2) neither measured nor bounded")
    out += [f"{a['id']}: no falsification test" for a in e.get("adds", [])
            if REQUIRE_FALSIFICATION and a.get("status", "open") in ps.UNSETTLED and not a.get("falsification")]
    if e.get("rule_change") or e.get("rule_question"):
        out.append("a ruling (" + ("rule change" if e.get("rule_change") else "rule question") + ")")
    if e["status"] in GONE:
        out.append(e["status"])
    return out


def frontier(entries: list[dict], enforce: bool = True, rules_stand: bool = False, stands_if: bool = False) -> list[dict]:
    """At each rank, the least-h design among those meeting every target, not refuted or closed, with the axis at most
    that rank; with `rules_stand`, only designs that need no ruling; with `stands_if`, a design refuted only by families
    that state the write model under which they'd stand counts as standing (the model granted)."""
    out = []
    for lv in sorted(set(ps.RANKS.values())):
        live = [e for e in entries if e.get("h") is not None and e.get("ranks") and e["ranks"]["axis"] <= lv
                and (not enforce or (e["meets"] and (e["status"] not in GONE or (stands_if and e.get("stands_if")))))
                and not (rules_stand and (e.get("rule_change") or e.get("rule_question")))]
        if live:
            best = min(live, key=lambda e: (e["h"], e["ranks"]["axis"], e["id"]))
            out.append({"rank": lv, "best": best["id"], "h": best["h"], "cut": best.get("cut")})
    return out


def lead_block(card: dict, hc: dict) -> dict:
    pr = priced(card, hc, choice(hc))
    if pr["refused"]:
        raise SystemExit(f"the lead is refused: {pr['refused']}")
    rows, parts = pr["rows"], pr["parts"]
    t2m, t3m = hc["lead"]["t2_measured"], hc["lead"]["t3_measured"]
    out = {"name": pr["name"], "premises": pr["design"]["premises"], "design": pr["design"],
           "h": max(r["hash_units_per_mac"] for r in rows.values()), "rows": rows, "parts": parts,
           "bytes_per_mac": max(q["bytes_per_mac"] for q in parts.values()),
           "macs_per_word": 4 * word_price(card, hc["basis"]["words_chunk"]) / parts["t8192"]["words"],
           "t1": max(r["t1"] for r in rows.values()),
           "t2_measured": max(max(t2m[p]) for p in hc["basis"]["points"]),
           "t2_card": max(r["t2"]["judged"] for r in rows.values()),
           "t3_measured": max(t3m["ctx2k"], t3m["ctx8k"]),
           "ranks": rank_axis(card, pr["design"]["premises"], []), "periods": ps.periods(card, rows["t8192"])}
    return out


def fmt(x, pct: bool = False, digits: int = 1) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    return f"{x:.{digits + 2}%}" if pct else f"{x:,.{digits}f}"


def rank_label(v: float) -> str:
    return ps.LABEL.get(v, f"{v:g}")


def render(hc: dict, lead: dict, readings: list[dict], entries: list[dict], fr: list[dict], fr_rules: list[dict],
           checks_line: str, fr_model: list[dict] | None = None) -> str:
    b = hc["basis"]
    L = [f"# The HASH card: h against assumption rank, the lead as baseline", "",
         f"**Basis.** {b['title']}. Model {b['model']}, points {', '.join(b['points'])}, the words in "
         f"{b['words_chunk'] // 1024} KiB chunks. {b['h_definition']}.", "",
         f"## The lead: `{lead['name']}`", "",
         "| point | h (W1 a MAC) | words | A | bound | salts | coins | credited bytes a MAC | T1 (checked) | T2 card / measured (sync, async) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for p in b["points"]:
        r, q = lead["rows"][p], lead["parts"][p]
        m = hc["lead"]["t2_measured"][p]
        L.append(f"| {p} | **{r['hash_units_per_mac']:.2f}** | {q['words']:.2f} | {q['A']:.3f} | {q['bound']:.3f} | "
                 f"{q['salts']:.3f} | {q['coins']:.3f} | {q['bytes_per_mac']:.5f} | {r['t1']:.4%} | "
                 f"{r['t2']['judged']:.1f}× / {m[0]:.2f}×, {m[1]:.2f}× |")
    rk = lead["ranks"]
    L += ["", f"- T3 measured: {hc['lead']['t3_measured']['ctx2k']:+.3%} / {hc['lead']['t3_measured']['ctx8k']:+.3%} "
          f"(2K / 8K). T2 measured at most {lead['t2_measured']:.2f}×.",
          f"- Layout: {lead['macs_per_word']:.2f} useful MACs a committed 4-byte word "
          f"({4 / lead['macs_per_word']:.5f} words bytes a MAC).",
          f"- Ranks (nominal / effective): hashing part {rank_label(rk['hashing'])} / {rank_label(rk['hashing_effective'])}, "
          f"matmul part {rank_label(rk['matmul'])} / {rank_label(rk['matmul_effective'])}, overall "
          f"{rank_label(rk['overall'])} / {rank_label(rk['overall_effective'])}. **The axis is the hashing part's "
          f"{rank_label(rk['axis'])}**; the matmul part's floor is common to every design on the lead's noise.", "",
          "## The h readings in circulation", "",
          "| reading | said | design | price | credited bytes a MAC | h at its price | h on the basis | reproduced |",
          "|---|---|---|---|---|---|---|---|"]
    for rd in readings:
        L.append(f"| {rd['value']:g} | {rd['said']} | `{rd['design']}` | {rd['price']} | {rd['bytes']:.5f} | "
                 f"{rd['at_price']:.2f} | {rd['on_basis']:.2f} | {'yes' if rd['ok'] else 'NO'} |")
    if hc.get("bound"):
        L += ["", "## No-go bounds the lanes have shown", "", "| bound | lane | class | what it says | label |", "|---|---|---|---|---|"]
        for k, v in hc["bound"].items():
            L.append(f"| {k} | {v['lane']} | {v['what']} | {v['value']} | {v['label']} |")
    L += ["", "## Per monitoring period: the lead's honest hashing (h × stock work)", "",
          "| period | scope | stock work (AND) | honest hashing, W1 at stock's AND rate | in SHA-512's own AND gates | share of the compute covered |",
          "|---|---|---|---|---|---|"]
    per = lead["periods"]
    for period in ("window", "week", "month", "year"):
        for scope in ("h200", "datacenter", "world"):
            v = per[period][scope]
            L.append(f"| {period} | {scope} | {v['stock_ands']:.3e} | {v['shaspeed_ands']:.3e} | "
                     f"{v['shaspeed_real_ands']:.3e} | {v['shaspeed_ands'] / v['stock_ands']:.1f}× |")
    L += ["", f"Scaled, not proved: {per['scaled']}.", "",
          "## The designs", "",
          "| design | lane | status | h on the basis | label | cut vs lead | T1 | T2 | T3 | axis rank (eff.) | meets all |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in entries:
        t = {k: (f"{fmt(e[k][0], pct=(k != 't2'), digits=2)}{'×' if k == 't2' else ''} ({e[k][1]})" if k in e and e[k][0] is not None else "—")
             for k in ("t1", "t2", "t3")}
        ax = f"{rank_label(e['ranks']['axis'])} ({rank_label(e['ranks']['axis_effective'])})" if e.get("ranks") else "—"
        L.append(f"| {e['id']}: {e['title']} | {e['lane'] or e['kind']} | {e['status']} | {fmt(e.get('h'), digits=2)} | {e['h_label']} | "
                 f"{fmt(e.get('cut'), digits=2)}{'×' if e.get('cut') is not None else ''} | {t['t1']} | {t['t2']} | {t['t3']} | {ax} | "
                 f"{'yes' if e['meets'] else 'no'} |")
    L += ["", "## What doesn't compare, design by design", ""]
    for e in entries:
        fl = e["flags"] + list((e.get("refused") or {}).values())[:1]
        if fl:
            L.append(f"- **{e['id']}**: " + "; ".join(fl) + ".")
    by = {e["id"]: e for e in entries}
    models = sorted({f.get("model", f["stands_if"]) for f in hc.get("family", {}).values() if f.get("stands_if")})
    tables = [("as the lanes rank them, rulings pending", fr), ("under the rules as they stand (no ruling needed)", fr_rules)]
    if fr_model is not None:
        tables.append(("under the rules as they stand, with each refuted family's write model granted ("
                       + " | ".join(models) + ")", fr_model))
    for title, f_ in tables:
        L += ["", f"## The frontier, {title}: the least h at each rank among designs meeting T1, T2 and T3", "",
              "| axis rank ≤ | best design | h | cut vs lead | effective rank | what done-when still needs |", "|---|---|---|---|---|---|"]
        for f in f_:
            e = by[f["best"]]
            L.append(f"| {rank_label(f['rank'])} | {f['best']} | {f['h']:.2f} | {fmt(f['cut'], digits=2)}× | "
                     f"{rank_label(e['ranks']['axis_effective'])} | {'; '.join(e['done_missing']) or 'nothing'} |")
    best = list(dict.fromkeys(["lead"] + [f["best"] for f in fr + fr_rules + (fr_model or [])]))
    L += ["", "## Per monitoring period: honest commitment work of the frontier's designs (h × stock, W1 at stock's AND rate)",
          "", "Pooled totals over each period and scope; the share of the compute covered is h itself (and T1's uncredited "
          "share beside it). The window is one 600 s window.", "",
          "| design | h (share of stock) | T1 | period | one H200 | datacenter | world |", "|---|---|---|---|---|---|---|"]
    for k in best:
        e = by[k]
        for period in ("window", "week", "month", "year"):
            s = {sc: per[period][sc]["stock_ands"] for sc in ("h200", "datacenter", "world")}
            L.append(f"| {k} | {e['h']:.2f}× | {fmt(e['t1'][0], pct=True, digits=2)} | {period} | "
                     + " | ".join(f"{e['h'] * s[sc]:.3e} (T1 {e['t1'][0] * s[sc]:.2e})" for sc in ("h200", "datacenter", "world")) + " |")
    done = [e for e in entries if not e["done_missing"]]
    L += ["", f"**Done-when (a 10× cut holding the targets, credit checked, cost measured or bounded, each new premise "
          f"with a falsification test, no ruling pending):** {'met by ' + ', '.join(e['id'] for e in done) if done else 'not met'}.",
          "", f"Checks: {checks_line}", ""]
    return "\n".join(L)


def reading(card: dict, hc: dict, key: str, rd: dict) -> dict:
    pr = priced(card, hc, choice(hc, full=rd["compose"]))
    bytes_now = max(q["bytes_per_mac"] for q in pr["parts"].values())
    per_byte = hc["price"][rd["price"]]["per_byte"]
    b = rd.get("bytes_per_mac_then", bytes_now)
    at_price = b * per_byte
    return {"id": key, "value": rd["value"], "said": rd["said"], "design": rd["design"], "price": rd["price"],
            "bytes": b, "bytes_now": bytes_now, "at_price": at_price, "on_basis": pr["worst"]["h"],
            "ok": abs(at_price / rd["value"] - 1) <= rd.get("tolerance", 0.002),
            "words_chunk": rd["words_chunk"]}


def ratio_check(card: dict, hc: dict, lead: dict, checks: Checks, prefix: str = "", convert: bool = True) -> None:
    seed = {"lane": "TEST", "note": "none", "title": "seeded", "status": "proposed",
            "h": {"ratio": 0.5, "baseline": 233.0, "label": "estimate"}, "layout": {"macs_per_word": 31}}
    e = design_entry(card, hc, "seeded", seed, lead, "proposal", convert=convert)
    fl = "; ".join(e["flags"])
    checks.check(f"{prefix}an h stated as a ratio to a lane's own lead (233) lands on the basis lead "
                 f"({lead['h']:.2f}), flagged with that baseline, and a 31-MAC layout is flagged against the basis "
                 f"lead's {lead['macs_per_word']:.2f}",
                 abs(e["h"] / (0.5 * lead["h"]) - 1) < 1e-12 and "h = 233" in fl and "31 useful MACs a word" in fl,
                 f"h {e['h']:.2f}; {fl[:160]}")


def run(card: dict, hc: dict, checks: Checks, prefix: str = "") -> dict:
    lead = lead_block(card, hc)
    pts = hc["basis"]["points"]
    hs = [lead["rows"][p]["hash_units_per_mac"] for p in pts]
    checks.check(f"{prefix}the lead composes as its name, {hc['basis']['lead_name']}", lead["name"] == hc["basis"]["lead_name"],
                 lead["name"])
    checks.check(f"{prefix}the lead's h is N-CARD final's 127.50 W1 a MAC at every Llama point",
                 all(abs(h - 127.50) < 0.01 for h in hs), ", ".join(f"{h:.3f}" for h in hs))
    sums = [abs(lead["parts"][p]["h"] / lead["rows"][p]["hash_units_per_mac"] - 1) for p in pts]
    checks.check(f"{prefix}h's parts (words, operands, A, coins, bound, salts) sum to pouw_space's h at every point",
                 max(sums) < 1e-9, f"largest relative residual {max(sums):.1e}")
    q = lead["parts"]["t8192"]
    checks.check(f"{prefix}the words are the lead's hashing: {q['words'] / q['h']:.2%} of h at T 8,192 (k32 words, "
                 f"4 B a 32-deep step)", q["words"] / q["h"] > 0.9, f"words {q['words']:.2f} of {q['h']:.2f}")
    t1 = [lead["rows"][p]["t1"] for p in pts]
    checks.check(f"{prefix}the lead's T1 is N-CARD final's 0.0656% / 0.0436% / 0.0407%",
                 all(abs(a - b) < 5e-7 for a, b in zip(t1, (0.000656, 0.000436, 0.000407))),
                 " / ".join(f"{x:.4%}" for x in t1))
    ratio = [lead["rows"][p]["t2"]["judged"] / max(hc["lead"]["t2_measured"][p]) for p in pts]
    checks.check(f"{prefix}the lead's T2 is measured under 100x, and the card's estimate within 0.88-1.0 of it",
                 lead["t2_measured"] <= TARGET["t2"] and all(0.88 <= r <= 1.0 for r in ratio),
                 f"measured at most {lead['t2_measured']:.2f}x; card / measured " + ", ".join(f"{r:.3f}" for r in ratio))
    per = lead["periods"]["year"]["world"]
    checks.check(f"{prefix}per period, the lead's honest hashing is h x stock work in every scope",
                 abs(per["shaspeed_ands"] / per["stock_ands"] / lead["rows"]["t8192"]["hash_units_per_mac"] - 1) < 1e-12,
                 f"world, year: {per['shaspeed_ands']:.3e} AND against {per['stock_ands']:.3e}")

    readings = [reading(card, hc, k, v) for k, v in hc["reading"].items()]
    r233 = next(r for r in readings if r["id"] == "brief-233")
    checks.check(f"{prefix}233 is the 08:32Z lead's 0.13104 B a MAC at the bar's time price (1,774.85 W1 a byte), and "
                 f"today's card prices the same design's bytes within 0.1%: the 233 is the price, not the bytes",
                 r233["ok"] and abs(r233["bytes_now"] / r233["bytes"] - 1) < 0.001,
                 f"{r233['bytes']:.5f} x 1,774.85 = {r233['at_price']:.2f}; today {r233['bytes_now']:.5f} B, "
                 f"h {r233['on_basis']:.2f} on the basis")
    r119 = next(r for r in readings if r["id"] == "brief-119_5")
    checks.check(f"{prefix}119.5 is ssn_1l at 256 KiB rows at the served committer's floor (908 W1 a byte), within 1%",
                 r119["ok"], f"{r119['bytes']:.5f} x 908 = {r119['at_price']:.2f}; on the basis {r119['on_basis']:.2f}")

    base = design_entry(card, hc, "lead", {"title": "the lead", "status": "baseline", "compose": hc["basis"]["lead"]},
                        lead, "baseline")
    base["t2"], base["t3"] = (lead["t2_measured"], "measured"), (lead["t3_measured"], "measured")
    base["lane"], base["done_missing"] = "baseline", done_missing(base)
    entries = [base]
    for k, v in hc.get("known", {}).items():
        entries.append(design_entry(card, hc, k, v, lead, "known"))
    for k, v in hc.get("proposal", {}).items():
        entries.append(design_entry(card, hc, k, v, lead, "proposal"))
    by = {e["id"]: e for e in entries}
    r16, r256 = by["r16k-words"], by["lead"]
    checks.check(f"{prefix}the words chunk moves h by under 1% (16 KiB against 256 KiB): the brief's 233 / 119.5 "
                 f"is not a chunk-size pair", r16.get("h") and abs(r16["h"] / r256["h"] - 1) < 0.01,
                 f"16 KiB {r16.get('h', 0):.2f}, 256 KiB {r256['h']:.2f}; T1 at 16 KiB {fmt(r16.get('t1', (None,))[0], True)}")
    r1 = by["r1"]
    checks.check(f"{prefix}R1 leaves the prover no hashing but fails T2 at every point",
                 r1.get("h") == 0 and r1["passes"]["t2"] is False,
                 f"h {r1.get('h')}, T2 {fmt(r1.get('t2', (None,))[0])}x")
    k64 = by["k64"]
    why = "; ".join(sorted(set((k64.get("refused") or {}).values())))
    checks.check(f"{prefix}the card prices no word every 64 steps on the lead's noise, for its layout, not the skip: "
                 f"the lead's depth isn't whole 64-step words, so road (b)'s verdict stays N-DENSE's, carried",
                 k64.get("h") is None and "is not a whole number of words at L 64" in why, why[:160])

    props = [e for e in entries if e["kind"] == "proposal"]
    bad = [e["id"] for e in props if not (e["lane"] and e["note"] and e["status"] and e["h_label"] in LABELS + ("refused", "unpriced"))]
    checks.check(f"{prefix}every proposal names its lane, its note, its status and h's label", not bad,
                 f"{len(props)} proposals" + (f"; missing: {', '.join(bad)}" if bad else ""))
    nofals = [(e, a["id"]) for e in props for a in e["adds"]
              if a.get("status", "open") in ps.UNSETTLED and not a.get("falsification")]
    held = all(f"{p}: no falsification test" in e["done_missing"] and f"premise {p} has no falsification test" in e["flags"]
               for e, p in nofals)
    checks.check(f"{prefix}every premise a proposal adds is ranked, and an open one without a falsification test is "
                 f"flagged and held off done-when",
                 held and all(a["rank"] in ps.RANKS for e in props for a in e["adds"]),
                 f"{len(nofals)} without a test: " + ", ".join(f"{e['id']}/{p}" for e, p in nofals) if nofals else "none")
    dv = [(e, hc["proposal"][e["id"]]) for e in props if "combine" in hc["proposal"][e["id"]] or "fold" in hc["proposal"][e["id"]]]
    rep = []
    for e, spec in dv:
        st = spec.get("stated", {})
        bare = derived(card, hc, lead, spec, fold_writes=False)["worst"]
        if "h_bare" in st:
            rep.append((e["id"], abs(bare["h"] - st["h_bare"]) < 0.01 and
                        ("t1_dw" not in st or abs(bare["t1_dw"] - st["t1_dw"]) < 0.01)))
    checks.check(f"{prefix}the card reproduces each derived design's h before the fold's writes (within 0.01) and its "
                 f"T1 under DistinctWrites alone (within 1 point) as its lane states them",
                 all(ok for _, ok in rep), ", ".join(f"{k} {'ok' if ok else 'NO'}" for k, ok in rep) or "none stated")
    nofold = [e["id"] for e, _ in dv if not (e["fold_w1"] > 0 and abs(
        e["h"] - e["extra_w1_per_mac"] - e["fold_w1"] - derived(card, hc, lead, hc["proposal"][e["id"]], False)["worst"]["h"]) < 1e-9)]
    checks.check(f"{prefix}every derived design's h counts the fold's own register writes at W1, as h's definition "
                 f"and the lead's word count require", not nofold and bool(dv),
                 ", ".join(nofold) or ", ".join(f"{e['id']} +{e['fold_w1']:.2f}" for e, _ in dv)[:200])
    below = [e["id"] for e in entries if e.get("ranks") and any(
        ps.rank_value(a["rank"]) > e["ranks"]["axis"] for a in (hc.get("proposal", {}).get(e["id"], {}).get("adds", [])))]
    checks.check(f"{prefix}no design's axis rank is below a premise it adds", not below, ", ".join(below) or "none")
    ratio_check(card, hc, lead, checks, prefix)
    fr = frontier(entries)
    mono = all(a["h"] >= b["h"] for a, b in zip(fr, fr[1:]))
    ok = all(by[f["best"]]["meets"] and by[f["best"]]["status"] not in GONE for f in fr)
    checks.check(f"{prefix}the frontier's h falls (weakly) as rank rises, and every design on it meets T1, T2 and T3 and "
                 f"stands", mono and ok and bool(fr), " → ".join(f"{rank_label(f['rank'])}: {f['best']} {f['h']:.1f}" for f in fr))
    fm = frontier(entries, rules_stand=True, stands_if=True)

    def granted(e: dict) -> bool:
        fams = [hc.get("family", {}).get(a.get("family", ""), {}) for a in hc.get("proposal", {}).get(e["id"], {}).get("adds", [])]
        return all(f.get("stands_if") for f in fams if f.get("status") == "refuted")
    ok = all(by[f["best"]]["meets"] and (by[f["best"]]["status"] not in GONE or granted(by[f["best"]])) for f in fm)
    checks.check(f"{prefix}with each refuted family's stated write model granted (rules otherwise as they stand), every "
                 f"design on the frontier meets T1, T2 and T3 and is refuted, if at all, only by such families",
                 ok and bool(fm), " → ".join(f"{rank_label(f['rank'])}: {f['best']} {f['h']:.1f}" for f in fm))
    return {"lead": lead, "readings": readings, "entries": entries, "frontier": fr,
            "frontier_rules": frontier(entries, rules_stand=True), "frontier_model": fm}


def mutations(card: dict, hc: dict, checks: Checks) -> None:
    def caught(name: str, card2: dict, hc2: dict, fn=None) -> None:
        with checks.mutant(name):
            if fn:
                fn(card2, hc2)
            else:
                run(card2, hc2, checks, prefix="[mutant] ")

    c = copy.deepcopy(card)
    c["price"]["sha_forms"]["chain"] = hc["price"]["bar"]["per_byte"] * ps.BLOCK / c["price"]["reg"]["value"]
    caught("the words priced at the bar's time price, as the card did until 09:58Z", c, hc)
    c = copy.deepcopy(card)
    c["price"]["sha_forms"]["chunk_k"] = 0
    caught("the words priced at a flat 3,532 a compression, not their chunk's R", c, hc)

    def no_bound(card2, hc2):
        lead = lead_block(card2, hc2)
        d = lead["design"]
        q = decompose(card2, d, "t8192", bound=False)
        checks.check("[mutant] h's parts sum to pouw_space's h without the bound blocks",
                     abs(q["h"] / lead["rows"]["t8192"]["hash_units_per_mac"] - 1) < 1e-9, f"{q['h']:.4f}")
    caught("h decomposed without the row-bound blocks", card, hc, no_bound)

    def unenforced(card2, hc2):
        res = run(card2, hc2, Checks(argv=[], out=open("/dev/null", "w")))
        by = {e["id"]: e for e in res["entries"]}
        fr = frontier(res["entries"], enforce=False)
        checks.check("[mutant] every design on an unenforced frontier meets the targets and stands",
                     all(by[f["best"]]["meets"] and by[f["best"]]["status"] not in GONE for f in fr),
                     ", ".join(f["best"] for f in fr))
    caught("a frontier that ignores T2 and status: R1 at h = 0", card, hc, unenforced)

    def switched(name, value):
        def fn(card2, hc2):
            old = globals()[name]
            globals()[name] = value
            try:
                run(card2, hc2, checks, prefix="[mutant] ")
            finally:
                globals()[name] = old
        return fn

    h2 = copy.deepcopy(hc)
    h2.setdefault("proposal", {})["mut-nofals"] = {
        "lane": "TEST", "note": "none", "title": "a seeded proposal", "status": "proposed",
        "h": {"value": 12.0, "price": "w1-forms", "label": "estimate"},
        "adds": [{"id": "seeded-premise", "rank": 5, "part": "matmul", "status": "open"}]}
    caught("done-when that ignores an open premise without a falsification test", card, h2,
           switched("REQUIRE_FALSIFICATION", False))
    caught("a derived design's h without the fold's own register writes", card, hc, switched("FOLD_WRITES", False))
    h3 = copy.deepcopy(hc)
    h3["family"]["mut-refuted"] = {"what": "seeded", "stated": [{"lane": "TEST", "name": "seeded", "rank": "4a"}],
                                   "status": "refuted", "refuted_by": "a seeded adversary"}
    h3["proposal"]["mut-refuted"] = {
        "lane": "TEST", "note": "none", "title": "a seeded refuted proposal", "status": "proposed",
        "h": {"value": 0.5, "price": "w1-forms", "label": "checked"}, "t1": {"value": 0.0001, "label": "checked"},
        "t2": {"value": 2.0, "label": "estimate"}, "t3": {"value": 0.0, "label": "measured"},
        "adds": [{"id": "seeded", "family": "mut-refuted", "rank": "4a", "part": "hashing", "status": "open",
                  "falsification": "x"}]}
    caught("a model-granted frontier that grants a family stating no write model", card, h3,
           switched("GRANT_ANY_REFUTED", True))

    def axis_without_adds(card2, hc2):
        spec = {"lane": "TEST", "note": "none", "title": "seeded", "status": "proposed",
                "h": {"value": 12.0, "price": "w1-forms", "label": "estimate"},
                "adds": [{"id": "seeded-premise", "rank": 5, "part": "matmul", "status": "open", "falsification": "x"}]}
        lead = lead_block(card2, hc2)
        e = design_entry(card2, hc2, "seeded", spec, lead, "proposal")
        e["ranks"] = rank_axis(card2, lead["premises"], spec["adds"], with_adds=False)
        checks.check("[mutant] a design's axis rank is at least every premise it adds",
                     e["ranks"]["axis"] >= 5.0, rank_label(e["ranks"]["axis"]))
    caught("the axis rank computed without the premises a design adds", card, hc, axis_without_adds)

    def unconverted(card2, hc2):
        lead = lead_block(card2, hc2)
        spec = {"lane": "TEST", "note": "none", "title": "seeded", "status": "proposed",
                "h": {"value": 233.0, "price": "bar", "label": "estimate"}}
        e = design_entry(card2, hc2, "seeded", spec, lead, "proposal", convert=False)
        checks.check("[mutant] the 08:32Z lead's 233 at the bar lands within 1% of its h on the basis",
                     abs(e["h"] / 115.73 - 1) < 0.01, f"{e['h']:.2f}")
    caught("an h stated at the bar taken unconverted", card, hc, unconverted)
    caught("a ratio applied to the lane's own lead's h, not the basis lead's", card, hc,
           lambda c2, h2: ratio_check(c2, h2, lead_block(c2, h2), checks, "[mutant] ", convert=False))


def main() -> int:
    checks = Checks()
    args = checks.args
    out = Path(args[args.index("--out") + 1]) if "--out" in args else Path(".")
    data = Path(args[args.index("--data") + 1]) if "--data" in args else DATA
    card, hc = ps.load(ps.CARD), load(data)
    res = run(card, hc, checks)
    if checks.mutations:
        mutations(card, hc, checks)
    n_ok = sum(c["ok"] for c in checks.checks)
    line = f"{n_ok}/{len(checks.checks)} checks" + (
        f", {sum(m['caught'] for m in checks.mutants)}/{len(checks.mutants)} mutations caught" if checks.mutants else "")
    out.mkdir(parents=True, exist_ok=True)
    (out / "hash_card.md").write_text(render(hc, res["lead"], res["readings"], res["entries"], res["frontier"],
                                             res["frontier_rules"], line, res["frontier_model"]))
    keep = ("id", "kind", "title", "lane", "note", "status", "design", "h", "h_label", "cut", "t1", "t1_dw", "fold_w1",
            "t2", "t3", "meets", "flags", "rule_change", "rule_question", "refused", "attacks", "done_missing", "adds",
            "stands_if")
    js = {"basis": hc["basis"], "lead": {k: res["lead"][k] for k in ("name", "h", "t1", "t2_measured", "t2_card", "t3_measured",
                                                                     "macs_per_word")}
          | {"parts": res["lead"]["parts"], "ranks": res["lead"]["ranks"]},
          "readings": res["readings"], "frontier": res["frontier"], "frontier_rules_stand": res["frontier_rules"],
          "frontier_model_granted": res["frontier_model"],
          "designs": [{k: e[k] for k in keep if k in e} | {"ranks": e.get("ranks")} for e in res["entries"]]}
    (out / "hash_card.json").write_text(json.dumps(js, indent=1, sort_keys=True, default=str) + "\n")
    with (out / "hash_card.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "kind", "status", "h", "h_label", "cut", "t1", "t2", "t3", "axis_rank", "axis_rank_effective",
                    "meets", "flags"])
        for e in res["entries"]:
            w.writerow([e["id"], e["kind"], e["status"], "" if e.get("h") is None else f"{e['h']:.4f}", e["h_label"],
                        "" if e.get("cut") is None else f"{e['cut']:.4f}",
                        *("" if t not in e or e[t][0] is None else f"{e[t][0]:.6g}" for t in ("t1", "t2", "t3")),
                        rank_label(e["ranks"]["axis"]) if e.get("ranks") else "",
                        rank_label(e["ranks"]["axis_effective"]) if e.get("ranks") else "",
                        e["meets"], " | ".join(e["flags"])])
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
