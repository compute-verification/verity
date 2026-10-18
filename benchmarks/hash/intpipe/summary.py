"""A run's measurements as one JSON object a build (or a candidate), with the integer-issue ceiling derived from its SASS.

    python3 summary.py sha512 OUT     sha512_baseline.sh's run: per build, the raw rate and the 4 KiB row-seg leaves
    python3 summary.py hashes OUT     candidates.sh's run: per candidate, against full SHA-512 in the same rounds
    python3 summary.py table RUN...   fetched candidates.sh runs' directories (research fetch RUN --all): the hash x GPU table
    python3 summary.py hm96 OUT       hm96.sh's run: per candidate and size, GB/s (salts precomputed; the whole step)
    python3 summary.py hm96table RUN...   fetched hm96.sh runs: GB/s per primitive and GPU, and the GPU x candidate speedups

The ceiling: SMs x 4 sub-partitions x the median SM clock x 32 lanes / the warp cycles a block (sass_count.py's bound:
2 cycles an ALU or FMA instruction, 1 an issue).
"""
from __future__ import annotations

import json
import os
import statistics
import sys


def lines(path: str) -> list[dict]:
    return [json.loads(x) for x in open(path) if x.startswith("{")] if os.path.exists(path) else []


def med(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def sms(out: str) -> int:
    for name in sorted(os.listdir(out)):
        for r in lines(os.path.join(out, name)) if name.endswith(".jsonl") else []:
            for v in (r, r.get("sha", {}), r.get("meta", {})):
                if isinstance(v, dict) and "sms" in v:
                    return int(v["sms"])
    return 0


def gpu_name(out: str) -> str:
    p = os.path.join(out, "gpu.txt")
    if not os.path.exists(p):
        return ""
    rows = [x for x in open(p).read().splitlines()[1:] if x and not x.startswith("index")]
    return rows[0].split(", ")[2] if rows else ""


def ceiling(sass: dict, n_sm: int, mhz: float) -> float:
    return n_sm * 4 * mhz * 1e6 * 32 / sass["warp_cycles_per_block_bound"] * 1e-9 if sass and n_sm else float("nan")


def sha512(out: str) -> None:
    n_sm = sms(out)
    straight = {}
    p = os.path.join(out, "sass-a0r0.json")
    if os.path.exists(p):
        straight = json.load(open(p))
    for t in sorted(x[len("sha-"):-len(".jsonl")] for x in os.listdir(out) if x.startswith("sha-") and x.endswith(".jsonl")):
        raw = [r["sha"] for r in lines(os.path.join(out, f"sha-{t}.jsonl"))]
        mhz = med([r["mhz"] for r in raw])
        sass = json.load(open(os.path.join(out, f"sass-{t}.json"))) if os.path.exists(os.path.join(out, f"sass-{t}.json")) else {}
        rec = {"build": t, "gpu": gpu_name(out), "sms": n_sm, "rounds": len(raw), "gcomp_per_s": med([r["gcomp_per_s"] for r in raw]),
               "gcomp_per_s_all": [r["gcomp_per_s"] for r in raw], "gb_per_s_blocks": med([r["gb_per_s"] for r in raw]),
               "mhz": mhz, "sass_total_per_block": sass.get("total_per_block"), "sass_pipes_per_block": sass.get("pipes_per_block"),
               "sass_per_block": sass.get("per_block"), "ceiling_gcomp_per_s": ceiling(sass, n_sm, mhz),
               "ceiling_from_straight_line_gcomp_per_s": ceiling(straight, n_sm, mhz)}
        rec["fraction_of_ceiling"] = rec["gcomp_per_s"] / rec["ceiling_from_straight_line_gcomp_per_s"]
        by_n: dict[int, list[dict]] = {}
        for r in lines(os.path.join(out, f"bench-seg-{t}.jsonl")):
            if "bench" in r:
                by_n.setdefault(r["n"], []).append(r)
        rec["leaves_4k_rowseg"] = {
            str(n): {"operand_mib": rs[0]["operand_bytes"] / 2**20, "rounds": len(rs),
                     "data_phase_ms": med([r["commit_ms"] for r in rs]), "data_phase_gb_per_s": med([r["operand_gb_per_s"] for r in rs]),
                     "graph_ms": med([r["graph_ms"] for r in rs]), "graph_gb_per_s": med([r["operand_gb_per_s_graph"] for r in rs]),
                     "salt_ms": med([r["salt_ms"] for r in rs]),
                     "with_salt_gb_per_s": med([r["operand_bytes"] / (r["commit_ms"] + r["salt_ms"]) * 1e-6 for r in rs]),
                     "gcomp_per_s": med([r["gcomp_per_s"] for r in rs]), "mhz": med([r["mhz"] for r in rs])}
            for n, rs in sorted(by_n.items())}
        print(json.dumps(rec))


def ands_per_compression(kind: int, rounds: int) -> int:
    """hm96/PROTOCOL.md section 5's model: a w-bit addition w - 1 ANDs, Ch and Maj w each, XOR free."""
    if kind in (0, 1):
        return rounds * (7 * 63 + 2 * 64) + max(0, rounds - 16) * 3 * 63 + 8 * 63
    if kind == 2:
        return 64 * (7 * 31 + 2 * 32) + 48 * 3 * 31 + 8 * 31
    if kind == 5:
        return 0
    return 7 * 8 * 6 * 31


def hashes(out: str) -> None:
    by: dict[tuple[str, int, str, int], list[dict]] = {}
    for r in lines(os.path.join(out, "bench.jsonl")):
        if "candidate" in r:
            by.setdefault((r["candidate"], r["sha_roll"], r["mode"], r.get("leaves", 0)), []).append(r)
    for (cand, roll, mode, _), rs in sorted(by.items()):
        m = rs[0]
        if mode == "memread":
            print(json.dumps({"candidate": cand, "sha_roll": roll, "mode": mode, "gpu": m["gpu"], "leaves": m["leaves"],
                              "reps": len(rs), "gb_per_s": med([r["gb_per_s"] for r in rs]), "mhz": med([r["mhz"] for r in rs])}))
            continue
        mhz = med([r["mhz"] for r in rs])
        ands = ands_per_compression(m["cand_kind"], m["cand_r"])
        rec = {"candidate": cand, "sha_roll": roll, "mode": mode, "gpu": m["gpu"], "sms": m["sms"], "reps": len(rs),
               "gcomp_per_s": med([r["gcomp_per_s"] for r in rs]), "gb_per_s": med([r["gb_per_s"] for r in rs]),
               "base_gcomp_per_s": med([r["base_gcomp_per_s"] for r in rs]), "base_gb_per_s": med([r["base_gb_per_s"] for r in rs]),
               "ratio_to_base": med([r["gb_per_s"] / r["base_gb_per_s"] for r in rs]),
               "mhz": mhz, "base_mhz": med([r["base_mhz"] for r in rs]), "ands_per_compression": ands}
        if mode == "raw":
            rec["ands_per_byte"] = ands / m["block_bytes"]
            p = os.path.join(out, f"sass-{cand}-r{roll}.json")
            straight = os.path.join(out, f"sass-{cand}-r0.json")
            s = json.load(open(p)) if os.path.exists(p) else {}
            s0 = json.load(open(straight)) if os.path.exists(straight) else {}
            if s:
                rec["sass_total_per_block"] = s["total_per_block"]
                rec["sass_pipes_per_block"] = s["pipes_per_block"]
                rec["ceiling_gcomp_per_s"] = ceiling(s, m["sms"], mhz)
            if s0:
                rec["straight_sass_total_per_block"] = s0["total_per_block"]
                rec["straight_sass_pipes_per_block"] = s0["pipes_per_block"]
                rec["instructions_per_byte"] = s0["total_per_block"] / m["block_bytes"]
                rec["ceiling_from_straight_line_gcomp_per_s"] = ceiling(s0, m["sms"], mhz)
                rec["fraction_of_ceiling"] = rec["gcomp_per_s"] / rec["ceiling_from_straight_line_gcomp_per_s"]
        else:
            n = m["leaves"]
            rec["leaves"] = n
            rec["leaf_ms"] = med([r["leaf_ms"] for r in rs])
            rec["ms"] = med([r["ms"] for r in rs])
            rec["leaf_gb_per_s"] = med([r["leaf_gb_per_s"] for r in rs])
            rec["base_leaf_gb_per_s"] = med([r["base_leaf_gb_per_s"] for r in rs])
            rec["threads_per_leaf"] = m["threads_per_leaf"]
            rec["fetch"] = m.get("fetch", 128)
            rec["loader_async"] = m.get("loader_async", 0)
            rec["leaf_warps"] = m.get("leaf_warps", 4)
            rec["ands_per_byte"] = (m["leaf_comps"] * n + m["node_comps"] * (n - 1)) * ands / (4096 * n)
        print(json.dumps(rec))


ORDER = ["sha512", "sha512r64", "sha512r48", "sha512r40", "sha512r32", "sha512r24", "sha256", "blake3", "blake3chunks", "xor"]


def table(args: list[str]) -> None:
    """The hash x GPU table (Markdown) from fetched runs' directories, the rolled form; leaves at the loader's best FETCH."""
    raw: dict[tuple[str, str], tuple[dict, str]] = {}
    leaf: dict[tuple[str, str, int], tuple[dict, str]] = {}
    for d in args:
        run = os.path.basename(os.path.normpath(d))
        for root, _, files in os.walk(d):
            if "summary.jsonl" not in files:
                continue
            for r in lines(os.path.join(root, "summary.jsonl")):
                if r["sha_roll"] != 1:
                    continue
                if r["mode"] == "raw":
                    raw.setdefault((r["candidate"], r["gpu"]), (r, run))
                else:
                    k = (r["candidate"], r["gpu"], r["leaves"])
                    if k not in leaf or r["gb_per_s"] > leaf[k][0]["gb_per_s"]:
                        leaf[k] = (r, run)
    gpus = sorted({g for _, g in raw}, key=lambda g: ("RTX" not in g, g))
    print("| hash | GPU | G cmp/s | GB/s raw (x SHA-512) | % ceiling | GB/s 4 KiB leaves 256 MiB / 1 GiB (x SHA-512) | fetch B "
          "| ANDs/B raw / leaves | instr/B | runs |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for c in ORDER:
        for g in gpus:
            if (c, g) not in raw:
                continue
            r, rr = raw[(c, g)]
            small, sr = leaf.get((c, g, 65536), ({}, ""))
            big, br = leaf.get((c, g, 262144), ({}, ""))
            runs = " ".join(sorted({rr, sr, br} - {""}))
            rawc = ("— | — | —" if c == "xor" else   # the control's raw rate is a few XORs a block: no ceiling to read
                    f"{r['gcomp_per_s']:.2f} | {r['gb_per_s']:.0f} ({r['ratio_to_base']:.2f}) "
                    f"| {100 * r.get('fraction_of_ceiling', float('nan')):.0f}")
            print(f"| {c} | {g.replace('NVIDIA ', '').replace(' Blackwell Server Edition', '')} | {rawc} "
                  f"| {small.get('gb_per_s', float('nan')):.0f} / {big.get('gb_per_s', float('nan')):.0f} "
                  f"({big.get('ratio_to_base', float('nan')):.2f}) | {big.get('fetch', '')} "
                  f"| {r['ands_per_byte']:.0f} / {big.get('ands_per_byte', float('nan')):.0f} "
                  f"| {r.get('instructions_per_byte', float('nan')):.1f} | {runs} |")


def hm96(out: str) -> None:
    """hm96.sh's run: per candidate and size, the medians of the candidate's and base SHA-512's GB/s (data phase alone,
    and the whole step) and of the per-rep ratios."""
    by: dict[tuple[str, int], list[dict]] = {}
    for r in lines(os.path.join(out, "bench.jsonl")):
        if r.get("mode") == "hm96":
            by.setdefault((r["candidate"], r["leaves"]), []).append(r)
    for (cand, n), rs in sorted(by.items()):
        m = rs[0]
        rec = {"candidate": cand, "gpu": m["gpu"], "sms": m["sms"], "leaves": n, "bytes": m["bytes"], "reps": len(rs),
               "fetch": m["fetch"], "loader_async": m["loader_async"], "seg_warps": m["seg_warps"]}
        rec.update({k: m[k] for k in ("depth", "top_from", "data_launches") if k in m})
        for k in HM96_KEYS:
            if k in m:
                rec[k] = med([r[k] for r in rs])
                rec["base_" + k] = med([r["base_" + k] for r in rs])
        for mode in ("pre", "step", "pre_graph", "step_graph"):
            if mode + "_gb_per_s" in m:
                rec[mode + "_ratio"] = med([r[mode + "_gb_per_s"] / r["base_" + mode + "_gb_per_s"] for r in rs])
        print(json.dumps(rec))


HM96_KEYS = ("pre_gb_per_s", "step_gb_per_s", "pre_graph_gb_per_s", "step_graph_gb_per_s", "pre_ms", "step_ms", "salt_ms",
             "pre_graph_ms", "step_graph_ms", "seg_ms", "leaf_ms", "levels_ms", "top_ms", "mhz")


HM96_ORDER = ["sha512", "sha512r32", "sha512r24", "blake3k"]


def short(g: str) -> str:
    return g.replace("NVIDIA ", "").replace(" Blackwell Server Edition", "")


def hm96table(args: list[str]) -> None:
    """Fetched hm96.sh runs' directories: GB/s per candidate and GPU, then the speedup table (GPU x candidate)."""
    rec: dict[tuple[str, str, int], tuple[dict, str]] = {}
    for d in args:
        run = os.path.basename(os.path.normpath(d))
        for root, _, files in os.walk(d):
            if "summary.jsonl" in files:
                for r in lines(os.path.join(root, "summary.jsonl")):
                    rec[(r["candidate"], r["gpu"], r["leaves"])] = (r, run)
    gpus = sorted({g for _, g, _ in rec}, key=lambda g: ("RTX" not in g, g))
    sizes = sorted({n for _, _, n in rec})
    lab = {n: f"{n * 4096 / 1e6:.0f} MB" if n * 4096 < 2**30 else "1 GiB" for n in sizes}
    hdr = " | ".join(f"{m} {lab[n]}" for m in ("pre", "step") for n in sizes)
    print(f"| primitive | GPU | GB/s {hdr} | x SHA-512 {hdr} | MHz | run |")
    print("|---|---|" + "---|" * (2 * 2 * len(sizes) + 2))
    for c in HM96_ORDER:
        for g in gpus:
            rs = [rec.get((c, g, n), ({}, ""))[0] for n in sizes]
            if not any(rs):
                continue
            gb = " | ".join(f"{r.get(m + '_gb_per_s', float('nan')):.0f}" for m in ("pre", "step") for r in rs)
            x = " | ".join(f"{r.get(m + '_ratio', float('nan')):.2f}" for m in ("pre", "step") for r in rs)
            runs = " ".join(sorted({rec.get((c, g, n), ({}, ""))[1] for n in sizes} - {""}))
            print(f"| {c} | {short(g)} | {gb} | {x} | {med([r['mhz'] for r in rs if r]):.0f} | {runs} |")
    for g in gpus:
        rs = [rec.get(("sha512", g, n), ({}, ""))[0] for n in sizes]
        if any(rs):
            gb = " | ".join(f"{r.get('base_' + m + '_gb_per_s', float('nan')):.0f}" for m in ("pre", "step") for r in rs)
            print(f"| base sha512 (in the sha512 binary) | {short(g)} | {gb} | " + " | ".join(["1.00"] * 2 * len(sizes))
                  + f" | {med([r['base_mhz'] for r in rs if r]):.0f} | |")
    print()
    cands = [c for c in HM96_ORDER if c != "sha512"]
    big = sorted(sizes, reverse=True)
    print("| GPU | " + " | ".join(f"{c} {lab[n]}" for c in cands for n in big) + " |")
    print("|---|" + "---|" * (len(cands) * len(big)))
    for g in gpus:
        cells = []
        for c in cands:
            for n in big:
                r = rec.get((c, g, n), ({}, ""))[0]
                cells.append(f"{r['pre_ratio']:.2f} / {r['step_ratio']:.2f}" if r else "—")
        print(f"| {short(g)} | " + " | ".join(cells) + " |")


def hm96windows(args: list[str]) -> None:
    """Fetched hm96.sh runs at small windows: per (GPU, window, arm) GB/s and latency, launched kernel by kernel and as one
    CUDA graph, salts precomputed (pre) and on the step; the ratio over SHA-512 rep by rep; then the data phase by kernel."""
    rec: dict[tuple[str, str, int], tuple[dict, str]] = {}
    for d in args:
        run = os.path.basename(os.path.normpath(d))
        for root, _, files in os.walk(d):
            if "summary.jsonl" in files:
                for r in lines(os.path.join(root, "summary.jsonl")):
                    rec[(r["candidate"], r["gpu"], r["leaves"])] = (r, run)
    gpus = sorted({g for _, g, _ in rec}, key=lambda g: ("B200" not in g, "H200" not in g, g))
    sizes = sorted({n for _, _, n in rec})
    us = lambda r, k: f"{1e3 * r[k]:.0f}"
    print("| GPU | window | arm | GB/s pre / step | latency µs pre / step | x SHA-512 pre / step | graph: GB/s pre / step "
          "| graph: latency µs pre / step | graph: x SHA-512 pre / step | run |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for g in gpus:
        for n in sizes:
            for c in HM96_ORDER:
                r, run = rec.get((c, g, n), ({}, ""))
                if not r:
                    continue
                print(f"| {short(g)} | {r['bytes'] / 1e6:.1f} MB | {c} | {r['pre_gb_per_s']:.0f} / {r['step_gb_per_s']:.0f} | "
                      f"{us(r, 'pre_ms')} / {us(r, 'step_ms')} | {r['pre_ratio']:.2f} / {r['step_ratio']:.2f} | "
                      f"{r['pre_graph_gb_per_s']:.0f} / {r['step_graph_gb_per_s']:.0f} | "
                      f"{us(r, 'pre_graph_ms')} / {us(r, 'step_graph_ms')} | "
                      f"{r['pre_graph_ratio']:.2f} / {r['step_graph_ratio']:.2f} | {run} |")
    print()
    print("| GPU | window | arm | leaves (depth, level launches + top from) | segments µs | leaves µs | level launches µs "
          "| top CTA µs | data phase on the GPU µs | MHz |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for g in gpus:
        for n in sizes:
            for c in HM96_ORDER:
                r = rec.get((c, g, n), ({}, ""))[0]
                if not r:
                    continue
                tot = sum(r[k] for k in ("seg_ms", "leaf_ms", "levels_ms", "top_ms"))
                print(f"| {short(g)} | {r['bytes'] / 1e6:.1f} MB | {c} | {n} ({r['depth']}, {r['top_from'] - 1} + "
                      f"{r['top_from']}) | {us(r, 'seg_ms')} | {us(r, 'leaf_ms')} | {us(r, 'levels_ms')} | {us(r, 'top_ms')} | "
                      f"{1e3 * tot:.0f} | {r['mhz']:.0f} |")


if __name__ == "__main__":
    if sys.argv[1] in ("table", "hm96table", "hm96windows"):
        {"table": table, "hm96table": hm96table, "hm96windows": hm96windows}[sys.argv[1]](sys.argv[2:])
    else:
        {"sha512": sha512, "hashes": hashes, "hm96": hm96}[sys.argv[1]](sys.argv[2])
