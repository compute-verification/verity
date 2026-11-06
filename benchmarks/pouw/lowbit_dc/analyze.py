"""fpx's records into the exactness tables.

  python analyze.py OUT [--json FILE]       # OUT holds fpx's records.jsonl, windows.jsonl, summary.json; markdown to stdout
  python analyze.py gemm DIR [--json FILE]  # DIR holds egemm's gate.jsonl, bench.jsonl and dgemm-bench.jsonl
  python analyze.py pgemm DIR [--json FILE] # DIR holds pgemm's gate.jsonl, bench.jsonl and pgemm.sass

Per chain config: the first K whose partials an FP32 can't hold (first_nonrep), the first K with a wrong output
(first_inexact), bad_rep (wrong while every partial was representable: must be 0 for "exact iff representable"), and
how many outputs a chain of one FP32 add per MMA, rounded toward zero, reproduces at every K (rz_all). Per window test:
the largest gap at which every output is exact, and what happens past it. Per emu8 route: wrong accumulators and words.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# routes whose every k32 word is checked: emu8's int8 limb routes and the restarted E4M3 grid
WORD_ROUTES = ("k3b", "k4b", "k3z", "k4z", "k3h", "e4r")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def per_config(records: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in records:
        by[r["cfg"]].append(r)
    rows = []
    for cfg, rs in sorted(by.items()):
        rs.sort(key=lambda r: r["K"])
        head = {k: rs[0][k] for k in ("cfg", "fmt", "grid", "s", "dist", "c0", "c0bits", "sa", "sb", "unit_num", "unit_e",
                                      "control")}
        nonrep = [r["K"] for r in rs if r["rep"] < r["n"]]
        inexact = [r["K"] for r in rs if r["inexact"] > 0]
        rows.append(head | {
            "K_max": rs[-1]["K"],
            "first_nonrep": nonrep[0] if nonrep else None,
            "first_inexact": inexact[0] if inexact else None,
            "bad_rep": sum(r["bad_rep"] for r in rs),
            "rz_all": all(r["rz"] == r["n"] for r in rs),
            "rn_all": all(r["rn"] == r["n"] for r in rs),
            "rz_frac": sum(r["rz"] for r in rs) / max(1, sum(r["n"] for r in rs)),
            "negzero": sum(r["negzero"] for r in rs),
            "bad_word": rs[-1]["bad_word"],
            "bad_acc": rs[-1]["bad_acc"],
            "maxerr": max(r["maxerr"] for r in rs),
        })
    return rows


def group_chains(rows: list[dict]) -> list[dict]:
    """One line per (format, grid, scales, C0) over its distributions."""
    by = defaultdict(list)
    for r in rows:
        if r["fmt"] in ("e4m3", "mxf4", "nvf4") and not r["control"]:
            by[(r["fmt"], r["grid"], r["s"], r["sa"], r["sb"], r["c0"])].append(r)
    out = []
    for (fmt, grid, s, sa, sb, c0), rs in sorted(by.items()):
        fi = [r["first_inexact"] for r in rs if r["first_inexact"] is not None]
        fn = [r["first_nonrep"] for r in rs if r["first_nonrep"] is not None]
        out.append({
            "fmt": fmt, "grid": grid, "s": s, "sa": sa, "sb": sb, "c0": c0, "c0bits": rs[0]["c0bits"],
            "unit": f"{rs[0]['unit_num']}*2^{rs[0]['unit_e']}", "configs": len(rs), "K_max": max(r["K_max"] for r in rs),
            "first_nonrep": min(fn) if fn else None, "first_inexact": min(fi) if fi else None,
            "bad_rep": sum(r["bad_rep"] for r in rs), "rz_all": all(r["rz_all"] for r in rs),
            "rn_all": all(r["rn_all"] for r in rs), "rz_frac": min(r["rz_frac"] for r in rs),
            "negzero": sum(r["negzero"] for r in rs),
        })
    return out


def windows(win: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for w in win:
        by[(w["fmt"], w["test"])].append(w)
    out = []
    for (fmt, test), ws in sorted(by.items()):
        ws.sort(key=lambda w: w["gap"])
        exact_gaps = [w["gap"] for w in ws if w["exact"] == w["n"]]
        bad = [w for w in ws if w["exact"] < w["n"]]
        out.append({
            "fmt": fmt, "test": test, "gaps": [ws[0]["gap"], ws[-1]["gap"]],
            "max_exact_gap": max((g for g in exact_gaps if all(x["exact"] == x["n"] for x in ws if x["gap"] <= g)),
                                 default=None),
            "first_bad_gap": bad[0]["gap"] if bad else None,
            "past": {k: sum(w[k] for w in bad) for k in ("n", "exact", "rz", "rn", "dropped", "other")},
        })
    return out


def gemm_summary(gates: list[dict], bench: list[dict], dbench: list[dict]) -> dict:
    """egemm's gates and its time. Per shape and emulated arm (k3*, k4*), streamed, resident (RES: operands loaded once,
    the on-chip pipeline alone) and pipelined (PIPE: resident with the next step's MMAs beside this step's absorb, against
    i8r, native int8 in the same shape): the time against native int8 at the emulated arm's tile (the like-for-like
    ratio), against int8 at dgemm's tile and against cuBLASLt int8 (dgemm bench), and committed words a SM a clock at
    the device's reported clock."""
    dg = {(d["m"], d["n"], d["k"]): d for d in dbench}
    rows, native = [], []
    for b in bench:
        arms = sorted({k[: -len("_fold_ms")] for k in b if k.endswith("_fold_ms")})
        words = b["m"] * b["n"] * b["k"] / 32
        clocks = b.get("sms", 0) * b.get("clock_khz", 0) * 1e3

        def rate(ms: float) -> float | None:
            return words / (ms * 1e-3 * clocks) if clocks else None

        d = dg.get((b["m"], b["n"], b["k"]), {})
        for emu in (a for a in arms if a.startswith(("k3", "k4"))):
            tile = emu.split("_")[1]
            same = f"i8_{tile}"
            other = "i8_bn256" if b.get("arch") == "sm_100a" else "i8_bn128"
            split = "split_x_k4s" if emu.startswith("k4s") else "split_x"
            row = {"m": b["m"], "n": b["n"], "k": b["k"], "emu": emu, "i8_same_tile": same, "i8_dgemm_tile": other,
                   "split_x_ms": b.get(f"{split}_ms")}
            for mode, sfx, base in (("streamed", "", same), ("resident", "_res", same),
                                    ("pipelined", "_pipe", f"i8r_{tile}")):
                if f"{emu}_fold{sfx}_ms" not in b or f"{base}_fold{sfx}_ms" not in b:
                    continue
                e, i8 = b[f"{emu}_fold{sfx}_ms"], b[f"{base}_fold{sfx}_ms"]
                ref = b[f"{other}_fold{'_res' if sfx else ''}_ms"]
                row[mode] = {
                    "base": base, "emu_fold_ms": e, "emu_nofold_ms": b[f"{emu}_nofold{sfx}_ms"], "i8_fold_ms": i8,
                    "i8_nofold_ms": b[f"{base}_nofold{sfx}_ms"], "i8_dgemm_tile_fold_ms": ref,
                    "fold_ratio_same_tile": e / i8,
                    "nofold_ratio_same_tile": b[f"{emu}_nofold{sfx}_ms"] / b[f"{base}_nofold{sfx}_ms"],
                    "fold_ratio_dgemm_tile": e / ref,
                    "emu_words_sm_clk": rate(e), "i8_words_sm_clk": rate(i8), "i8_dgemm_tile_words_sm_clk": rate(ref),
                }
            if "cublaslt_ms" in d:
                row |= {"cublaslt_ms": d["cublaslt_ms"], "emu_fold_over_cublaslt": b[f"{emu}_fold_ms"] / d["cublaslt_ms"]}
            if row["split_x_ms"] is not None:
                row["split_x_over_i8_fold"] = row["split_x_ms"] / b[f"{same}_fold_ms"]
            rows.append(row)
        for i8r in (a for a in arms if a.startswith("i8r_") and f"{a}_fold_pipe_ms" in b):
            same = "i8_" + i8r.split("_")[1]
            native.append({"m": b["m"], "n": b["n"], "k": b["k"], "arm": i8r,
                           "res_over_i8_res": b[f"{i8r}_fold_res_ms"] / b[f"{same}_fold_res_ms"],
                           "pipe_over_i8_res": b[f"{i8r}_fold_pipe_ms"] / b[f"{same}_fold_res_ms"],
                           "pipe_words_sm_clk": rate(b[f"{i8r}_fold_pipe_ms"])})
    return {
        "gates": len(gates),
        "gates_exact": sum(g["exact"] and g["control_rejected"] and g["split_bad"] == 0 for g in gates),
        "gate_arms": sorted({g["arm"] for g in gates}),
        "bench": rows,
        "i8r": native,
    }


def _f(x: float | None, spec: str) -> str:
    return "-" if x is None else format(x, spec)


def gemm_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="analyze.py gemm")
    ap.add_argument("out", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    s = gemm_summary(load(a.out / "gate.jsonl"), load(a.out / "bench.jsonl"), load(a.out / "dgemm-bench.jsonl"))
    print(f"# egemm: int8 emulated in FP8 against native int8\n\ngates exact (route, receipts, control rejected, split): "
          f"{s['gates_exact']} of {s['gates']} over arms {', '.join(s['gate_arms'])}\n")
    print("| m x n x k | arm | mode | emu fold ms | int8 same tile fold ms (arm) | ratio | nofold ratio | ratio to dgemm tile | "
          "emu words/SM/clk | i8 words/SM/clk | emu / cuBLASLt | split X / i8 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in s["bench"]:
        for mode in ("streamed", "resident", "pipelined"):
            if mode not in r:
                continue
            m = r[mode]
            print(f"| {r['m']}x{r['n']}x{r['k']} | {r['emu']} | {mode} | {m['emu_fold_ms']:.4f} | {m['i8_fold_ms']:.4f} "
                  f"({m['base']}) | {m['fold_ratio_same_tile']:.2f} | {m['nofold_ratio_same_tile']:.2f} | "
                  f"{m['fold_ratio_dgemm_tile']:.2f} | {_f(m['emu_words_sm_clk'], '.2f')} | "
                  f"{_f(m['i8_words_sm_clk'], '.2f')} | "
                  f"{_f(r.get('emu_fold_over_cublaslt') if mode == 'streamed' else None, '.1f')} | "
                  f"{_f(r.get('split_x_over_i8_fold') if mode == 'streamed' else None, '.2f')} |")
    if s["i8r"]:
        print("\nNative int8 in k4s's shape (i8r: fresh accumulator each step, word in registers), against i8 resident:\n")
        print("| m x n x k | arm | i8r resident / i8 resident | i8r pipelined / i8 resident | i8r pipelined words/SM/clk |")
        print("|---|---|---|---|---|")
        for r in s["i8r"]:
            print(f"| {r['m']}x{r['n']}x{r['k']} | {r['arm']} | {r['res_over_i8_res']:.2f} | {r['pipe_over_i8_res']:.2f} | "
                  f"{_f(r['pipe_words_sm_clk'], '.2f')} |")
    if a.json:
        a.json.write_text(json.dumps(s, indent=1))
    return 0 if s["gates"] and s["gates_exact"] == s["gates"] else 4


PG_ROUTES = ("k4s", "i8c", "i8r", "pf8", "pi8", "k4p", "k4b", "k4t")  # pgemm's Route and Mode enums, in order
PG_MODES = ("full", "ext", "ld", "mma", "foldonly", "mmares", "fullres")
PG_DITHER = ("k4s", "k4p", "k4b", "i8c", "i8r", "k4t")
# k4t: a 128 x 256 CTA, one fold warpgroup a 128-column tile (template FW 2, receipt tile 128: its arms are named fw1,
# and so are its time-split arms).
PG_SPLIT_FW = {"k4t": 1}


def pg_tile(route: str) -> int:
    """pgemm's CTA tile, BM x cta_n(route)."""
    return 128 * (256 if route == "k4t" else 128)
SM100_ALU, SM100_IMAD, SM100_ISSUE = 2.0, 2.0, 4.0  # warp instructions a SM a clock (dc/README.md, sm_100's census)
ALU_OPS = ("LOP3", "LEA", "IADD3", "SHF", "ISETP", "PLOP3", "SEL")
NOT_LANE = ("U", "BRA", "NOP", "SYNCS", "LDTM", "STTM", "BSSY", "BSYNC", "ELECT", "FENCE", "VOTEU", "S2UR", "LDCU", "UTC")


def fold_mix(sass: str) -> dict[str, dict]:
    """Per pgemm kernel (named as its arm), the instructions a word of its word loop: the smallest backward-branch loop
    holding a word's duo (BNR LEA.HI; FULL, FOLDONLY), its TMEM traffic (EXT, LD) or its buffer release (MMA). The model
    bound is 32 words a warp instruction over the busiest of the ALU pipe, the IMAD pipe and issue."""
    out = {}
    parts = re.split(r"\n\s*Function : (\S+)\n", sass)
    for name, body in zip(parts[1::2], parts[2::2]):
        m = re.match(r"_Z5pg100ILi(\d)ELi(\d)ELi(\d+)ELi(\d)E", name)
        if not m:
            continue
        r, md, L, fw = map(int, m.groups())
        if PG_ROUTES[r] == "k4t":
            fw = 1
        bnr = 128 // fw
        ins, loops = [], []
        for line in body.splitlines():
            mm = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(.*?);", line)
            if not mm:
                continue
            addr, text = int(mm.group(1), 16), re.sub(r"^@!?U?P\w+\s+", "", mm.group(2))
            ins.append((addr, text.split()[0]))
            tgt = re.search(r"0x([0-9a-f]+)", text)
            if text.startswith("BRA") and tgt and int(tgt.group(1), 16) < addr:
                loops.append([o for a, o in ins if int(tgt.group(1), 16) <= a <= addr])
        if PG_MODES[md] in ("full", "foldonly", "fullres"):
            cand = [lp for lp in loops if sum(o.startswith("LEA.HI") for o in lp) >= bnr]
        elif PG_MODES[md] in ("mma", "mmares"):
            cand = [lp for lp in loops if any(o.startswith("SYNCS.ARRIVE") for o in lp)]
        else:
            cand = [lp for lp in loops if any(o.startswith(("LDTM", "STTM")) for o in lp)]
        if not cand:
            continue
        ops = Counter(o.split(".")[0] for o in min(cand, key=len))
        per = bnr  # one trip of the word loop: a thread's BNR words of one row
        alu = sum(v for k, v in ops.items() if k in ALU_OPS) / per
        imad = ops.get("IMAD", 0) / per
        total = sum(ops.values()) / per
        bound = 32 * min(SM100_ALU / alu if alu else float("inf"), SM100_IMAD / imad if imad else float("inf"),
                         SM100_ISSUE / total)
        arm = f"{PG_ROUTES[r]}_L{L}_fw{fw}_{PG_MODES[md]}"
        out[arm] = {"instrs_word": total, "lane_ops_word": sum(v for k, v in ops.items() if not k.startswith(NOT_LANE))
                    / per, "alu_word": alu, "imad_word": imad, "model_words_sm_clk": bound,
                    "mix": dict(sorted(ops.items(), key=lambda kv: -kv[1]))}
    return out


SM100_F8_SLOTS = 256  # k32 slots (an output element's k32 step) of FP8 tcgen05 MMA a SM a clock, dense peak


def gamma_peak(words_sm_clk: float | None, L: int) -> float | None:
    """γ a clock against the FP8 tensor peak: plain FP8's time for the words' k over the route's."""
    return SM100_F8_SLOTS * 32 / (L * words_sm_clk) if words_sm_clk else None


def dc_best(dc: list[dict]) -> dict[int, dict]:
    """DC's pipelined int8 kernel (`pgemm rate`): per L, its fastest honest configuration (arm 2: load and fold)."""
    out = {}
    for r in dc:
        if r.get("mode") == "rate" and r.get("arm") == 2 and r["committed_words_per_clk"] > out.get(r["L"], {}).get(
                "committed_words_per_clk", 0):
            out[r["L"]] = r
    return out


def pgemm_summary(gates: list[dict], bench: list[dict], mix: dict[str, dict], dc: list[dict] | None = None) -> dict:
    """pgemm's gates and time. Per shape, spacing L and Dither route: the kernel of record (the faster receipt tile
    width), its words a SM a clock, γ against plain FP8 (cuBLASLt and the same pipeline's pf8) and plain int8, its ratio
    to int8 Dither (the faster of i8c and i8r), the time split at FW 2 (tensor: MMARES, the MMAs on resident operands;
    operand feed: MMA - MMARES; tensor memory loads: LD - MMA; extraction: EXT - LD; fold: FULL - EXT), the alone times
    of the facets (each the pipeline up to it: tensor, + operand feed, + TMEM traffic; and the fold's lane work without
    tensor or TMEM), the binding facet (the largest), and γ on resident operands (FULLRES over cuBLASLt FP8 and over
    pf8_res: the kernel as a perfect operand feed would leave it). Rates are at each arm's measured SM clock
    (`<arm>_mhz`) where the bench has it, else at the reported clock; the CTA rate is the tile's words over its median
    CTA's clocks (`<arm>_cta_clk`), DC's metric, and γ a clock against the FP8 tensor peak is taken at it. With DC's
    `pgemm rate` rows, each route's time against DC's honest int8 kernel (its fastest configuration at that L, on
    resident operands) a clock and a second, and DC's γ against cuBLASLt FP8 and the FP8 peak."""
    dcb = dc_best(dc or [])
    rows = []
    for b in bench:
        sms, nominal = b.get("sms", 0), b.get("clock_khz", 0) / 1e3
        plain = {p: b.get(f"{p}_ms") for p in ("cublaslt_f8", "pf8", "cublaslt_i8", "pi8", "pf8_res")}
        for L in (32, 64):
            words = b["m"] * b["n"] * b["k"] / L

            def rate(arm: str, ms: float | None) -> float | None:
                mhz = b.get(f"{arm}_mhz") or nominal
                return words / (ms * 1e-3 * sms * mhz * 1e6) if ms and sms and mhz else None

            routes = {}
            for r in PG_DITHER:
                fulls = {fw: b[f"{r}_L{L}_fw{fw}_full_ms"] for fw in (2, 1) if f"{r}_L{L}_fw{fw}_full_ms" in b}
                if not fulls:
                    continue
                fw = min(fulls, key=fulls.get)
                sfw = PG_SPLIT_FW.get(r, 2)
                t, t2 = fulls[fw], fulls.get(sfw)
                md = {x: b.get(f"{r}_L{L}_fw{sfw}_{x}_ms") for x in ("mma", "ld", "ext", "foldonly", "mmares",
                                                                     "fullres")}
                facet_arm = {"tensor": "mmares", "operand_feed": "mma", "tensor_memory": "ld", "fold": "foldonly"}
                alone = {f: md[x] for f, x in facet_arm.items()}
                bind = max((f for f in alone if alone[f]), key=lambda f: alone[f], default=None)
                split = None
                if t2 and None not in (md["mma"], md["ld"], md["ext"]):
                    tensor = md["mmares"] or md["mma"]
                    split = {"tensor": tensor, "operand_feed": md["mma"] - tensor,
                             "tensor_memory_loads": md["ld"] - md["mma"], "extraction": md["ext"] - md["ld"],
                             "fold": t2 - md["ext"]}
                arm = f"{r}_L{L}_fw{fw}_full"
                res = md["fullres"]
                cta = pg_tile(r) * b["k"] / L / b[f"{arm}_cta_clk"] if b.get(f"{arm}_cta_clk") else None
                d = dcb.get(L)
                over_dc = {}
                if d and cta:
                    over_dc = {"over_dc_int8_clk": d["committed_words_per_clk"] / cta,
                               "over_dc_int8_s": d["committed_words_per_clk"] * d["mhz"] / (cta * b[f"{arm}_mhz"])
                               if b.get(f"{arm}_mhz") else None}
                routes[r] = {"full_ms": t, "fw": fw, "full_ms_by_fw": fulls, "mhz": b.get(f"{arm}_mhz"),
                             "words_sm_clk": rate(arm, t), "cta_words_sm_clk": cta,
                             "gamma_fp8_peak_clk": gamma_peak(cta, L), **over_dc,
                             "split_ms": split, "alone_ms": alone,
                             "alone_words_sm_clk": {f: rate(f"{r}_L{L}_fw{sfw}_{facet_arm[f]}", v)
                                                    for f, v in alone.items()},
                             "binding": bind, "binding_of_full": alone[bind] / t2 if bind and t2 else None,
                             "issue": mix.get(f"{r}_L{L}_fw{sfw}_full"), "full_resident_ms": res,
                             "gamma_resident_cublaslt_f8": res / plain["cublaslt_f8"] if res and plain["cublaslt_f8"]
                             else None,
                             "gamma_resident_pf8_res": res / plain["pf8_res"] if res and plain["pf8_res"] else None,
                             **{f"gamma_{p}": t / v for p, v in plain.items() if v and p != "pf8_res"}}
            int8 = min((r for r in ("i8c", "i8r") if r in routes), key=lambda r: routes[r]["full_ms"], default=None)
            for r in routes:
                routes[r]["over_int8_dither"] = routes[r]["full_ms"] / routes[int8]["full_ms"] if int8 else None
            d = dcb.get(L)
            dc_ms = words / (d["committed_words_per_clk"] * sms * d["mhz"] * 1e3) if d and sms else None
            rows.append({"m": b["m"], "n": b["n"], "k": b["k"], "L": L, "plain_ms": plain,
                         "plain_words_sm_clk": {p: rate(p, v) for p, v in plain.items()},
                         "cublaslt_f8_out": b.get("cublaslt_f8_out"), "int8_dither": int8, "routes": routes,
                         "dc_int8": d and {"words_sm_clk": d["committed_words_per_clk"], "mhz": d["mhz"],
                                           "nb": d["nb"], "lw": d["lw"], "mi": d["mi"], "ms_at_shape": dc_ms,
                                           "gamma_fp8_peak_clk": gamma_peak(d["committed_words_per_clk"], L),
                                           "gamma_cublaslt_f8": dc_ms / plain["cublaslt_f8"]
                                           if dc_ms and plain["cublaslt_f8"] else None}})
    return {
        "gates": len(gates),
        "gates_exact": sum(g["exact"] and g["control_rejected"] for g in gates),
        "gate_arms": sorted({g["arm"] for g in gates}),
        "gate_shapes": sorted({(g["m"], g["n"], g["k"], g["dist"]) for g in gates}),
        "bench": rows,
        "fold_mix": mix,
        "dc_int8": dcb,
    }


def pgemm_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="analyze.py pgemm")
    ap.add_argument("out", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    sass = a.out / "pgemm.sass"
    s = pgemm_summary(load(a.out / "gate.jsonl"), load(a.out / "bench.jsonl"),
                      fold_mix(sass.read_text()) if sass.exists() else {}, load(a.out / "dc-rate.jsonl"))
    print(f"# pgemm: the k4s Dither GEMM pipelined on B200\n\ngates exact (C, receipts, control rejected): "
          f"{s['gates_exact']} of {s['gates']} over {len(s['gate_arms'])} arms and {len(s['gate_shapes'])} shapes\n")
    print("| m x n x k | L | route | fw | ms | MHz | words/SM/clk | γ cuBLASLt FP8 | γ pf8 | γ cuBLASLt i8 | / int8 Dither | "
          "split ms: tensor, operand feed, TMEM loads, extraction, fold | alone ms: tensor, +feed, +TMEM, fold | "
          "fold alone words/SM/clk | binds (of FULL) | resident γ: cuBLASLt FP8, pf8_res |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in s["bench"]:
        for r, d in row["routes"].items():
            sp, al = d["split_ms"], d["alone_ms"]
            sps = ", ".join(f"{sp[x]:.3f}" for x in ("tensor", "operand_feed", "tensor_memory_loads", "extraction",
                                                       "fold")) if sp else "-"
            als = ", ".join(_f(al[x], ".3f") for x in ("tensor", "operand_feed", "tensor_memory", "fold"))
            print(f"| {row['m']}x{row['n']}x{row['k']} | {row['L']} | {r} | {d['fw']} | {d['full_ms']:.4f} | "
                  f"{_f(d['mhz'], '.0f')} | {_f(d['words_sm_clk'], '.1f')} | {_f(d.get('gamma_cublaslt_f8'), '.2f')} | "
                  f"{_f(d.get('gamma_pf8'), '.2f')} | {_f(d.get('gamma_cublaslt_i8'), '.2f')} | "
                  f"{_f(d['over_int8_dither'], '.2f')} | {sps} | {als} | "
                  f"{_f(d['alone_words_sm_clk']['fold'], '.1f')} | {d['binding']} ({_f(d['binding_of_full'], '.2f')}) | "
                  f"{_f(d['gamma_resident_cublaslt_f8'], '.2f')}, {_f(d['gamma_resident_pf8_res'], '.2f')} |")
    print("\nPlain (ms; words/SM/clk at L = 32):\n")
    for row in (r for r in s["bench"] if r["L"] == 32):
        print(f"- {row['m']}x{row['n']}x{row['k']}: " + ", ".join(
            f"{p} {_f(v, '.4f')} ({_f(row['plain_words_sm_clk'][p], '.1f')})" for p, v in row["plain_ms"].items())
              + f"; cuBLASLt FP8 out {row['cublaslt_f8_out']}")
    if s["dc_int8"]:
        print("\nAgainst DC's honest pipelined int8 kernel (its fastest configuration, resident operands, one CTA a "
              "SM), words a SM a clock of the median CTA:\n")
        print("| m x n x k | L | route | CTA words/SM/clk | DC int8 words/SM/clk (MHz) | time / DC's a clock | "
              "time / DC's a second | DC's γ: cuBLASLt FP8, FP8 peak a clock | route's γ: cuBLASLt FP8 (resident), "
              "FP8 peak a clock |")
        print("|---|---|---|---|---|---|---|---|---|")
        for row in s["bench"]:
            dci = row["dc_int8"]
            for r, d in row["routes"].items() if dci else ():
                print(f"| {row['m']}x{row['n']}x{row['k']} | {row['L']} | {r} | {_f(d['cta_words_sm_clk'], '.1f')} | "
                      f"{dci['words_sm_clk']:.1f} ({dci['mhz']:.0f}) | {_f(d.get('over_dc_int8_clk'), '.2f')} | "
                      f"{_f(d.get('over_dc_int8_s'), '.2f')} | {_f(dci['gamma_cublaslt_f8'], '.2f')}, "
                      f"{dci['gamma_fp8_peak_clk']:.2f} | {_f(d.get('gamma_cublaslt_f8'), '.2f')} "
                      f"({_f(d['gamma_resident_cublaslt_f8'], '.2f')}), {_f(d['gamma_fp8_peak_clk'], '.2f')} |")
    if s["fold_mix"]:
        print("\nIssue cost (SASS, word loop, per word): instructions, lane ops, ALU pipe, IMAD pipe, model bound\n")
        print("| arm | instrs | lane ops | ALU | IMAD | bound words/SM/clk |")
        print("|---|---|---|---|---|---|")
        for arm, x in sorted(s["fold_mix"].items()):
            if "_fw2_" in arm or arm.startswith("k4t_"):
                print(f"| {arm} | {x['instrs_word']:.2f} | {x['lane_ops_word']:.2f} | {x['alu_word']:.2f} | "
                      f"{x['imad_word']:.2f} | {x['model_words_sm_clk']:.1f} |")
    if a.json:
        a.json.write_text(json.dumps(s, indent=1))
    return 0 if s["gates"] and s["gates_exact"] == s["gates"] and s["bench"] else 4


def main() -> int:
    if sys.argv[1:2] == ["gemm"]:
        return gemm_main(sys.argv[2:])
    if sys.argv[1:2] == ["pgemm"]:
        return pgemm_main(sys.argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    summary = json.loads((a.out / "summary.json").read_text()) if (a.out / "summary.json").exists() else {}
    rows = per_config(load(a.out / "records.jsonl"))
    chains = group_chains(rows)
    wins = windows(load(a.out / "windows.jsonl"))
    emu = [r for r in rows if r["fmt"] in WORD_ROUTES]
    controls = [r for r in rows if r["control"]]
    print(f"# fpx on {summary.get('device', '?')} (cc {summary.get('cc', '?')})\n")
    print(f"summary: {json.dumps(summary)}\n")
    print("## Chains (one line per format, grid, scales, C0; over its distributions)\n")
    print("| fmt | grid | s | sa | sb | C0 | unit | configs | K max | first non-rep K | first inexact K | bad_rep | "
          "RZ chain all | min RZ frac | -0 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in chains:
        print(f"| {c['fmt']} | {c['grid']} | {c['s']} | {c['sa']} | {c['sb']} | {c['c0']} {c['c0bits']} | {c['unit']} | "
              f"{c['configs']} | {c['K_max']} | {c['first_nonrep']} | {c['first_inexact']} | {c['bad_rep']} | "
              f"{c['rz_all']} | {c['rz_frac']:.4f} | {c['negzero']} |")
    print("\n## Windows\n")
    print("| fmt | test | gaps | largest all-exact gap | first bad gap | past it: n, exact, RZ, RN, dropped, other |")
    print("|---|---|---|---|---|---|")
    for w in wins:
        p = w["past"]
        print(f"| {w['fmt']} | {w['test']} | {w['gaps']} | {w['max_exact_gap']} | {w['first_bad_gap']} | "
              f"{p['n']}, {p['exact']}, {p['rz']}, {p['rn']}, {p['dropped']}, {p['other']} |")
    print("\n## Word routes (every k32 prefix word: emu8's int8 limbs, and e4r's restarted E4M3 grid)\n")
    print("| route | grid | dist | K | bad accumulators | bad words |")
    print("|---|---|---|---|---|---|")
    for r in emu:
        if not r["control"]:
            print(f"| {r['fmt']} | {r['grid'] if r['fmt'] == 'e4r' else 'int8'} | {r['dist']} | {r['K_max']} | "
                  f"{r['bad_acc']} | {r['bad_word']} |")
    print("\n## Controls (no MMA issued; each must count wrong outputs)\n")
    for r in controls:
        caught = r["bad_rep"] + r["bad_word"] + r["bad_acc"] + (0 if r["fmt"] in WORD_ROUTES else
                                                                  int(r["first_inexact"] is not None))
        print(f"- {r['fmt']} {r['dist']} C0 {r['c0']}: {'caught' if caught else 'MISSED'}")
    if a.json:
        a.json.write_text(json.dumps({"summary": summary, "chains": chains, "windows": wins, "emu": emu,
                                      "controls": controls, "configs": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
