"""run.sh's summary: the gates, each shape's hot items with each arm's ratio to plain cuBLASLt, the ratio table (each
GEMM kernel against plain and each edited call, editedf:K, against plainq, hot and interleaved), the parts at o_proj and
ncu's counters. python3 summary.py OUT_DIR"""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

LAYER = {(8192, 4096, 4096): "o_proj", (8192, 6144, 4096): "qkv", (8192, 28672, 4096): "gate_up", (8192, 4096, 14336): "down"}
KIND = {"e_": "duo32", "t_duo32": "duo32", "d64_": "duo64", "x_": "xin", "t4_": "tb4", "t4c_": "tb4c", "t2w_": "tb2w",
        "t1w_": "tb1w"}


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.startswith("{")] if p.exists() else []


def ncu(p: Path) -> dict[str, str]:
    if not p.exists():
        return {}
    lines = p.read_text().splitlines()
    start = next((i for i, x in enumerate(lines) if x.startswith('"ID"')), None)
    if start is None:
        return {}
    return {r["Metric Name"]: r["Metric Value"] for r in csv.DictReader(io.StringIO("\n".join(lines[start:])))}


def kind(k: str) -> str:
    if k == "e_plain":
        return ""
    return next((v for p, v in sorted(KIND.items(), key=lambda x: -len(x[0])) if k.startswith(p)), "")


def shape(r: dict) -> str:
    return LAYER.get((r["m"], r["n"], r["k"]), f"{r['m']}x{r['n']}x{r['k']}")


def f(x: float | None, d: int = 3) -> str:
    return "" if x is None else f"{x:.{d}f}"


def main(o: Path) -> None:
    print(f"# edited GEMM, cheaper receipts and Llama-8B's linears ({o})\n")
    print("## gates\n")
    for g in jl(o / "gate.jsonl"):
        bad = {k: v for k, v in g["device"].items() if v["y_bad"] or v["r_bad"] > 0}
        print(f"- device {g['m']} x {g['n']} x {g['k']}: device_ok {g['device_ok']}, control_rejected {g['control_rejected']}, "
              f"{len(g['device'])} checks, failing {bad or 'none'}")
    for r in jl(o / "ref.jsonl"):
        nr = sum(len(t["R"]) for t in r["tiles"].values())
        okr = sum(sum(t["R"].values()) for t in r["tiles"].values())
        other = all(all(v for v in t.values() if isinstance(v, bool)) for t in r["tiles"].values())
        print(f"- CPU ref {r['m']} x {r['n']} x {r['k']}: ok {r['ok']}; receipts {okr}/{nr} tile-kernels bit-exact; X, Xq, P_A, W', "
              f"P_B, peel, Y {other}; S range {[t['S_range'] for t in r['tiles'].values()]}")
    bench = {shape(b): b for b in jl(o / "bench.jsonl")}
    il: dict[tuple, dict] = {}
    for r in jl(o / "il.jsonl"):
        il[(shape(r), r["il"], r["base"], bool(r.get("cold")))] = r
    for sh, b in bench.items():
        a = b["arms"]
        pl, pq = a.get("plain", {}).get("ms"), a.get("plainq", {}).get("ms")
        print(f"\n## {sh} ({b['m']} x {b['n']} x {b['k']}): hot items (median of the reps, ms)\n")
        print("| arm | ms | / plain | / plainq | MHz |\n|---|---|---|---|---|")
        for name, v in a.items():
            print(f"| {name} | {v['ms']:.4f} | {f(v['ms'] / pl if pl else None)} | {f(v['ms'] / pq if pq else None)} | {v['mhz']:.0f} |")
    print("\n## the ratio table\n")
    print("GEMM: the kernel against plain cuBLASLt int8 on the same X, W; wedit + GEMM: edited:K (wedit, then K) against "
          "plain; edited call: editedf:K (qwe, then K) against plainq (the plain quantizer, then cuBLASLt). Hot: medians of "
          "interleaved items; IL: per call, alternating; cold: IL with L2 flushed before each call.\n")
    print("| shape | kernel | receipt | GEMM hot | GEMM IL | GEMM IL cold | wedit + GEMM hot | wedit + GEMM IL | edited hot | "
          "edited IL | edited IL cold |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for sh, b in bench.items():
        a = b["arms"]
        pl, pq = a.get("plain", {}).get("ms"), a.get("plainq", {}).get("ms")
        for k in [n for n in a if kind(n) or n == "e_plain"]:
            ew, ef = a.get(f"edited:{k}", {}).get("ms"), a.get(f"editedf:{k}", {}).get("ms")
            g = lambda arm, base, cold: il.get((sh, arm, base, cold), {}).get("ratio")  # noqa: E731
            print(f"| {sh} | {k} | {kind(k) or 'none'} | {f(a[k]['ms'] / pl if pl else None)} | {f(g(k, 'plain', False))} | "
                  f"{f(g(k, 'plain', True))} | {f(ew / pl if ew and pl else None)} | {f(g(f'edited:{k}', 'plain', False))} | "
                  f"{f(ef / pq if ef and pq else None)} | {f(g(f'editedf:{k}', 'plainq', False))} | "
                  f"{f(g(f'editedf:{k}', 'plainq', True))} |")
    a = bench.get("o_proj", {}).get("arms", {})
    g = lambda n: a.get(n, {}).get("ms")  # noqa: E731
    if g("plain") and g("e_plain"):
        print("\n## o_proj by part (hot items, ms)\n")
        print(f"- mainloop + 2-atom tail against cuBLASLt (e_plain - plain): {g('e_plain') - g('plain'):+.4f}")
        for v in [n for n in a if kind(n)]:
            print(f"- receipts in {v} ({kind(v)}; {v} - e_plain): {g(v) - g('e_plain'):+.4f}")
        if g("qdither") and g("qplain") and g("wedit"):
            print(f"- wedit {g('wedit'):.4f}; qdither - qplain {g('qdither') - g('qplain'):+.4f}; qwe {f(g('qwe'), 4)}")
    leftover = [r for k, r in il.items() if k[0] not in bench]
    for r in leftover:
        print(f"- IL {shape(r)} {r['il']} / {r['base']}: {r['ratio']:.4f}")
    for p in sorted(o.glob("ncu-*.csv")):
        d = ncu(p)
        if d:
            print(f"\n### {p.stem}\n\n" + "; ".join(f"{k} {v}" for k, v in d.items()))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
