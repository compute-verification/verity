"""The honest lead kernel's price a committed word, from an np_kernel run.sh out/ (ssn.sass, time_m<M>.jsonl, smi.csv).

SASS (W1): register writes a warp-step of each ssn_k<BN, EXPORT>'s main loop, by path: the head (IGMMA, the export's
STGs, the skip-loads branch), the load head, the plain 16-byte path (stock, and step 0), the lane-coin path (later steps
under BLC), and the tail. A warp's GPR write counts 32, a uniform register or predicate 1 (W1 counts each as 1/32 of a register),
IGMMA.64xNx32 N/2 a lane. Predicated GPR writes count at half their lanes (pred 0.5: the guards test tid & 8) and at
all of them (pred 1.0). W1 a committed word = 32 x writes a warp-step / (16 x BN) words, averaged over a launch's S
steps (in-loop loads for steps 3..S-1, on the lane-coin path when --blc is 1 or 2; the last three iterations load
nothing); the prologue and epilogue are amortized over S, both arms counted. form_k, g_k and coins_k are loop-free:
writes a thread on the path that calls nothing, x threads / committed words. forward_w1 sums a forward's linears.

Time: units = ms x MHz x SMs x 4096 (the harness's f0); committed words M n S (ssn_*), k32 words M n K/32 (plain_gemm);
words and register writes an SM-clock; the export's HBM write rate.

  python lead_words.py OUT_DIR [--blc 1] > priced.jsonl
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

NO_DEST = {"ST", "STG", "STS", "STL", "RED", "REDUX", "BRA", "BRX", "JMP", "JMX", "CALL", "RET", "EXIT", "BAR", "NOP",
           "MEMBAR", "FENCE", "DEPBAR", "LDGDEPBAR", "LDGSTS", "WARPSYNC", "BSSY", "BSYNC", "YIELD", "WARPGROUP",
           "ERRBAR", "CCTL", "UTMALDG", "UTMASTG", "UBLKCP", "ARRIVES", "PREEXIT", "BPT", "KILL", "SYNCS", "NANOSLEEP"}
PRED2 = {"ISETP", "FSETP", "DSETP", "HSETP2", "PLOP3", "IADD3", "LEA", "LOP3", "IMAD", "UISETP", "UIADD3", "ULEA",
         "ULOP3", "UPLOP3", "SHFL", "VOTE", "R2P", "ATOMS", "UIMAD"}
INS = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
GUARD = re.compile(r"^@(!?)(U?P[T0-6])\s+")
FORMING = {"form_k": "_Z6form_k", "g_k": "_Z3g_k", "coins_k": "_Z7coins_k"}
SMS = 132


def width(op: str) -> int:
    base = op.split(".")[0]
    if base in ("IGMMA", "HGMMA", "QGMMA"):
        n = re.search(r"\.64x(\d+)x", op)
        return int(n.group(1)) // 2 if n else 0
    if ".WIDE" in op or (base == "CS2R" and ".32" not in op):
        return 2
    return 4 if ".128" in op else 2 if ".64" in op else 1


def writes(text: str, pred: float) -> tuple[str, float]:
    g = GUARD.match(text)
    if g and g.group(2) in ("PT", "UPT") and g.group(1) == "!":
        return "never", 0.0
    t = text[g.end():] if g else text
    parts = t.split(None, 1)
    op, base = parts[0], parts[0].split(".")[0]
    ops = [o.strip() for o in parts[1].split(",")] if len(parts) > 1 else []
    if base in NO_DEST or not ops:
        return op, 0.0
    w = 0.0
    if re.fullmatch(r"R\d+", ops[0]):
        w += 32 * width(op) * (pred if g and base != "IGMMA" else 1)
    elif re.fullmatch(r"UR\d+", ops[0]):
        w += width(op)
    elif re.fullmatch(r"P\d", ops[0]):
        w += 1
    elif re.fullmatch(r"UP\d", ops[0]):
        w += 1 / 32
    if len(ops) > 1 and base in PRED2:
        w += 1 if re.fullmatch(r"P\d", ops[1]) else 1 / 32 if re.fullmatch(r"UP\d", ops[1]) else 0
    return op, w


def functions(path: Path, prefixes: tuple[str, ...]) -> dict[str, list[tuple[int, str]]]:
    out, cur = {}, None
    for line in path.open():
        m = re.search(r"Function : (\S+)", line)
        if m:
            cur = m.group(1) if m.group(1).startswith(prefixes) else None
            if cur:
                out[cur] = []
        elif cur:
            m = INS.search(line)
            if m:
                out[cur].append((int(m.group(1), 16), m.group(2)))
    return out


def target(t: str) -> int | None:
    m = re.search(r"\bBRA(?:\s+!?P\d,)?\s+0x([0-9a-f]+)", t)
    return int(m.group(1), 16) if m else None


def ssn_loop(body: list[tuple[int, str]], pred: float) -> dict:
    lo, hi = next((target(t), a) for a, t in body if target(t) is not None and target(t) < a)
    loop = [(a, t) for a, t in body if lo <= a <= hi]
    mma = next(i for i, (_, t) in enumerate(loop) if "IGMMA" in t)
    skip = next(i for i in range(mma, len(loop)) if loop[i][1].startswith("@P") and target(loop[i][1]) is not None)
    tail = target(loop[skip][1])
    cv = next(i for i in range(skip + 1, len(loop)) if loop[i][1].startswith("@") and target(loop[i][1]) is not None)
    fork = target(loop[cv][1])

    def span(a0: int, a1: int) -> dict:
        ins = [(a, t) for a, t in loop if a0 <= a < a1]
        acc = sum(writes(t, pred)[1] for _, t in ins if "IGMMA" in t)
        word_copies = sum(bool(re.search(r"LDGSTS\.E \[", t)) for _, t in ins)
        return {"instr": len(ins), "acc": acc, "other": sum(writes(t, pred)[1] for _, t in ins) - acc,
                "word_copies": word_copies}

    near, far = span(loop[cv][0] + 1, fork), span(fork, tail)
    plain, lane = (near, far) if far["word_copies"] > near["word_copies"] else (far, near)
    parts = {"head": span(lo, loop[skip][0] + 1), "load_head": span(loop[skip][0] + 1, loop[cv][0] + 1),
             "plain": plain, "lane": lane, "tail": span(tail, hi + 1)}
    outside = sum(writes(t, pred)[1] for a, t in body if not lo <= a <= hi)
    shape = re.search(r"IGMMA\.(\S+)", loop[mma][1]).group(1)
    return {"loop": [hex(lo), hex(hi)], "igmma": shape, "parts": parts, "outside": outside}


def taken(body: list[tuple[int, str]], pred: float) -> float:
    """Writes a thread on the path that calls nothing: the operands fit 32 bits, so the 64-bit division's arm and the
    subroutine it calls are skipped."""
    skip = []
    for i, (a, t) in enumerate(body):
        if t.startswith("CALL"):
            sub = int(re.search(r"0x([0-9a-f]+)", t).group(1), 16)
            skip.append((sub, next(b for b, u in body if b >= sub and u.startswith("RET"))))
            g = max(b for b, u in body[:i] if u.startswith("@") and (target(u) or 0) > a)
            skip.append((g + 1, next(b for b, u in body[i:] if u.startswith("BRA"))))
    return sum(writes(t, pred)[1] for a, t in body if not any(lo <= a <= hi for lo, hi in skip)) / 32


def step_writes(p: dict, path: str | None) -> float:
    s = p["head"]["other"] + p["tail"]["other"]
    return s + (p["load_head"]["other"] + p[path]["other"] if path else 0)


def sass(out: Path, blc: int, preds: tuple[float, ...]) -> tuple[list[dict], dict]:
    names = functions(out / "ssn.sass", ("_Z5ssn_k",) + tuple(FORMING.values()))
    rows, per_word = [], {}
    for pred in preds:
        for name, body in names.items():
            m = re.match(r"_Z5ssn_kILi(\d+)ELb(\d)E", name)
            if not m:
                continue
            bn, export = int(m.group(1)), m.group(2) == "1"
            L = ssn_loop(body, pred)
            p, words, acc = L["parts"], 16 * bn, L["parts"]["head"]["acc"]
            for kind, S in (("lead", 74), ("lead", 293), ("stock", 64), ("stock", 256)):
                path = "lane" if kind == "lead" and blc else "plain"
                loads = S - 3
                avg = (loads * step_writes(p, path) + 3 * step_writes(p, None) + L["outside"]) / S
                wpw = (acc + avg) / words
                row = {"what": "sass_w1", "kernel": name, "BN": bn, "export": export, "pred": pred, "kind": kind,
                       "S": S, "igmma": L["igmma"], "acc_writes_warp_step": acc,
                       "other_writes_warp_step": {k: round(v["other"], 2) for k, v in p.items()},
                       "instr": {k: v["instr"] for k, v in p.items()}, "outside_writes_warp": round(L["outside"], 1),
                       "words_warp_step": words, "writes_per_word": round(wpw, 4), "w1_per_word": round(32 * wpw, 2)}
                rows.append(row)
                per_word[(bn, export, kind, S, pred)] = wpw
        for k, pre in FORMING.items():
            body = next(b for n, b in names.items() if n.startswith(pre))
            rows.append({"what": "sass_w1_forming", "kernel": k, "pred": pred, "instr": len(body),
                         "writes_per_thread": round(taken(body, pred), 2),
                         "writes_per_thread_both_arms": round(sum(writes(t, pred)[1] for _, t in body) / 32, 2)})
    return rows, per_word


def times(out: Path, per_word: dict, forming: dict) -> list[dict]:
    rows = []
    for f in sorted(out.glob("time_m*.jsonl"), key=lambda p: int(re.search(r"m(\d+)", p.name).group(1))):
        lines = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        by, fwd = {}, {}
        for r in lines:
            if r.get("linear") not in (None, "all"):
                by.setdefault(r["linear"], {})[r["what"]] = r
        for lin, d in by.items():
            ex, ssn, plain = d["ssn_gemm_export"], d["ssn_gemm"], d["plain_gemm"]
            M, n, K, S, BN, mhz = ex["M"], ex["n"], ex["K"], ex["S"], ex["BN"], ex["mhz"]
            clocks = lambda r: r["ms"] * 1e-3 * r["mhz"] * 1e6 * SMS  # noqa: E731
            words, k32 = M * n * S, M * n * K // 32
            wpw = {k: per_word[(BN, e, k2, S if k2 == "lead" else K // 32, 0.5)]
                   for k, e, k2 in (("export", True, "lead"), ("ssn", False, "lead"), ("plain", False, "stock"))}
            row = {"what": "time", "M": M, "linear": lin, "n": n, "K": K, "S": S, "BN": BN, "mhz": mhz,
                   "lane_overhead": round(S / (K / 32), 4),
                   "ms": {k: d[k]["ms"] for k in ("plain_gemm", "ssn_gemm", "ssn_gemm_export", "lane_link", "coins",
                                                    "forming", "g_draw", "coins_forming") if k in d}}
            for k, r, w in (("export", ex, words), ("ssn", ssn, words), ("plain", plain, k32)):
                per_clock = w / clocks(r)
                row[k] = {"units_per_word": round(4096 / per_clock, 2), "words_per_sm_clock": round(per_clock, 3),
                          "writes_per_sm_clock": round(per_clock * wpw[k], 3), "w1_per_word": round(32 * wpw[k], 2)}
            row["export"]["hbm_write_gb_s"] = round(4 * words / (ex["ms"] * 1e-3) / 1e9, 1)
            row["over_stock_time"] = {k: round(d[k]["ms"] / plain["ms"], 3)
                                      for k in ("ssn_gemm", "ssn_gemm_export", "lane_link", "coins_forming") if k in d}
            row["forming_units_per_word"] = {k: d[k]["units_per_word"] for k in ("coins", "forming", "g_draw",
                                                                                 "coins_forming", "lane_link") if k in d}
            threads = {"form_k": M * S, "g_k": n, "coins_k": d["coins"]["extra"]}
            w1 = {k: 32 * threads[k] * forming[k] / words for k in threads}
            row["forming_w1_per_word"] = {k: round(v, 3) for k, v in w1.items()}
            row["forming_w1_over_stock_w1"] = {k: round(v * words / (32 * wpw["plain"] * k32), 5) for k, v in w1.items()}
            rows.append(row)
            sums = {"floor": 32 * k32, "stock": 32 * wpw["plain"] * k32, "lead_gemm": 32 * wpw["export"] * words,
                    "lead_gemm_at_32": 32 * words, **{k: v * words for k, v in w1.items()},
                    **{f"ms_{k}": d[k]["ms"] for k in ("plain_gemm", "ssn_gemm", "ssn_gemm_export", "coins_forming")}}
            for k, v in sums.items():
                fwd[k] = fwd.get(k, 0.0) + v
        st = fwd["stock"]
        rows.append({"what": "forward_w1", "M": M, "linears": sorted(by),
                     "stock_over_floor": round(st / fwd["floor"], 4),
                     "lead_gemm_over_stock": round(fwd["lead_gemm"] / st, 4),
                     "lead_gemm_at_32_over_stock": round(fwd["lead_gemm_at_32"] / st, 4),
                     "form_g_over_stock": round((fwd["form_k"] + fwd["g_k"]) / st, 5),
                     "coins_over_stock": round(fwd["coins_k"] / st, 5),
                     "lead_over_stock": round((fwd["lead_gemm"] + fwd["form_k"] + fwd["g_k"] + fwd["coins_k"]) / st, 4),
                     "time_over_stock": {k: round(fwd[f"ms_{k}"] / fwd["ms_plain_gemm"], 3)
                                         for k in ("ssn_gemm", "ssn_gemm_export", "coins_forming")}})
    return rows


def clocks(out: Path) -> dict:
    rows = list(csv.reader((out / "smi.csv").open()))[1:]
    busy = [int(r[1].split()[0]) for r in rows if len(r) > 3 and r[3].strip().split()[0].isdigit()
            and int(r[3].split()[0]) > 0]
    return {"what": "smi", "busy_samples": len(busy), "min_mhz": min(busy), "max_mhz": max(busy),
            "share_at_max": round(sum(c == max(busy) for c in busy) / len(busy), 4)}


def main() -> None:
    out = Path(sys.argv[1])
    blc = int(sys.argv[sys.argv.index("--blc") + 1]) if "--blc" in sys.argv else 1
    rows, per_word = sass(out, blc, (0.5, 1.0))
    forming = {r["kernel"]: r["writes_per_thread"] for r in rows if r["what"] == "sass_w1_forming" and r["pred"] == 0.5}
    for r in rows + times(out, per_word, forming) + [clocks(out)]:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
