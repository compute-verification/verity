"""fpx's records into the exactness tables (note:20261017T0643Z-draft-lowbit-dc).

  python analyze.py OUT [--json FILE]       # OUT holds fpx's records.jsonl, windows.jsonl, summary.json; markdown to stdout
  python analyze.py gemm DIR [--json FILE]  # DIR holds egemm's gate.jsonl, bench.jsonl and dgemm-bench.jsonl

Per chain config: the first K whose partials an FP32 can't hold (first_nonrep), the first K with a wrong output
(first_inexact), bad_rep (wrong while every partial was representable: must be 0 for "exact iff representable"), and
how many outputs a chain of one FP32 add per MMA, rounded toward zero, reproduces at every K (rz_all). Per window test:
the largest gap at which every output is exact, and what happens past it. Per emu8 route: wrong accumulators and words.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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


def main() -> int:
    if sys.argv[1:2] == ["gemm"]:
        return gemm_main(sys.argv[2:])
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
