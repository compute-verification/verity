"""Summarize a host_tiles run's out/ (run.sh's JSON lines) as markdown: weights covered per scheme, batch, k-slice and core
count, and the capacity term against ΣR.

    python summarize.py OUT_DIR [--sigma-r 10468] [--host-cores 96] [--dram GBPS]

NCP's domain is 16 | m: m = 1 and 8 run as m = 16 (padded rows cost as much as real ones, E₁ being absolute noise), so
their rows repeat m = 16's.  Pearl-C runs m as given.  The whole-host estimate is min(compute, DRAM): compute is the rate a
core sustained with every one of the run's cores busy, times --host-cores; DRAM is --dram (both sockets' read bandwidth,
GB/s; default: twice the run's best single-socket read) divided by the bytes a weight byte costs (NCP: Y's 3; Pearl-C: B̃'s
1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NCP_M = {1: 16, 8: 16, 16: 16, 32: 32, 64: 64, 128: 128}
MS = (1, 8, 16, 32, 64, 128)


def rows(path: Path) -> list[dict]:
    rs = [json.loads(line) for line in path.read_text().splitlines() if line.startswith("{")] if path.exists() else []
    for r in rs:
        if r.get("kernel") == "ncp_amx":
            r.setdefault("y", "stored")
    return rs


def best(rs: list[dict], **kw) -> dict | None:
    hits = [r for r in rs if all(r.get(k) == v for k, v in kw.items())]
    return max(hits, key=lambda r: r["weight_bytes_per_s"]) if hits else None


def gbps(r: dict | None, key: str = "weight_bytes_per_s") -> str:
    return f"{r[key] / 1e9:.1f}" if r else "–"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--sigma-r", type=float, default=8 * 1308.5)
    ap.add_argument("--host-cores", type=int, default=96)
    ap.add_argument("--dram", type=float, default=None)
    ns = ap.parse_args()
    out = ns.out
    stream = rows(out / "stream.jsonl")
    res, strm, pearl = rows(out / "ncp_resident.jsonl"), rows(out / "ncp_stream.jsonl"), rows(out / "pearl.jsonl")
    threads = sorted({r["threads"] for r in res + strm + pearl})
    one_per_core = [t for t in threads if t <= max(threads) // 2] or threads
    cores = max(one_per_core)
    sock = max((r["bytes_per_s"] for r in stream), default=0) / 1e9
    dram = ns.dram if ns.dram else 2 * sock
    print(f"DRAM read, this run's CPUs: {', '.join(f'{r['threads']} threads {r['bytes_per_s'] / 1e9:.0f}' for r in stream)} GB/s")
    print(f"whole-host DRAM assumed: {dram:.0f} GB/s; cores: {cores} measured, {ns.host_cores} extrapolated; ΣR = {ns.sigma_r:.0f} GB/s\n")

    stored = lambda rs: [r for r in rs if r["y"] == "stored"]  # noqa: E731
    formed = lambda rs: [r for r in rs if r["y"] == "formed"]  # noqa: E731
    for name, rs, ks_list, mmap in (("NCP resident (Y in L2)", stored(res), (0, 128, 64, 32, 16), NCP_M),
                                    ("NCP resident, Y formed from B and F₁ (both in L2)", formed(res), (128, 32), NCP_M),
                                    ("NCP streaming (Y from DRAM, 3 B a weight byte)", stored(strm), (0, 128, 32, 16), NCP_M),
                                    ("NCP streaming, Y formed (B and F₁ from DRAM, 2 B a weight byte)", formed(strm),
                                     (128, 32), NCP_M),
                                    ("Pearl-C (gauss data, B̃ codes from DRAM)", pearl, (128, 32), {m: m for m in MS})):
        if not rs:
            continue
        print(f"### {name}: GB/s of weights covered\n")
        print("| m | k_s | " + " | ".join(f"{t} thr" for t in threads) + " |")
        print("|---|---|" + "---|" * len(threads))
        for m in MS:
            if mmap[m] != m and rs is not pearl:
                continue
            for ks in ks_list:
                cells = [gbps(best(rs, m=mmap[m], ks=ks, threads=t, **({"data": "gauss"} if rs is pearl else {})))
                         for t in threads]
                if any(c != "–" for c in cells):
                    print(f"| {m} | {ks} | " + " | ".join(cells) + " |")
        print()

    print("### Capacity term γ_cpu = whole-host tile rate / ΣR\n")
    print(f"| scheme | m | k_s | per core, all busy (GB/s) | core-ns a weight byte | ops/s a core | compute × host cores | "
          f"DRAM cap | whole host | γ_cpu | measured, {cores} cores |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for scheme, rs, ms, ks_list, mmap, per_byte in (
            ("NCP, Y stored", (stored(res), stored(strm)), MS, (128, 32, 16, 0), NCP_M, 3.0),
            ("NCP, Y formed", (formed(res), formed(strm)), MS, (128, 32), NCP_M, 2.0),
            ("Pearl-C", (pearl, pearl), MS, (128, 32), {m: m for m in MS}, 1.0)):
        for m in ms:
            for ks in ks_list:
                kw = {"data": "gauss"} if scheme == "Pearl-C" else {}
                r = best(rs[0], m=mmap[m], ks=ks, threads=cores, **kw)
                if not r:
                    continue
                per_core = r["weight_bytes_per_s"] / 1e9 / cores
                compute = per_core * ns.host_cores
                cap = dram / per_byte
                host = min(compute, cap)
                ops = r["mac_per_s"] if "mac_per_s" in r else r["products_per_s"]
                print(f"| {scheme} | {m} | {ks} | {per_core:.2f} | {1 / per_core:.3f} | {ops / cores:.2e} | "
                      f"{compute:.0f} | {cap:.0f} | {host:.0f} | "
                      f"{100 * host / ns.sigma_r:.2f}% | {gbps(best(rs[1], m=mmap[m], ks=ks, threads=cores, **kw))} |")
    diag = rows(out / "pearl_diag.jsonl")
    if diag:
        print("\n### Pearl-C small m: prefetch distance and B̃ resident in the LLC (2 weights) against streamed (40)\n")
        print("| m | threads | weights | prefetch | fused | GB/s |")
        print("|---|---|---|---|---|---|")
        for r in diag:
            print(f"| {r['m']} | {r['threads']} | {r['weights']} | {r.get('prefetch', 0)} | {r.get('fused', 0)} | {gbps(r)} |")
    for r in rows(out / "shake.jsonl"):
        print(f"\nSHAKE256 (hashlib): {r['bytes_per_s'] / 1e9:.2f} GB/s" + (f" over {r['processes']} processes, "
              f"{r['per_core_min'] / 1e9:.2f} GB/s the slowest" if "processes" in r else " on one core"))
    lat = rows(out / "latency.jsonl")
    if lat:
        print("\n### vLLM-like host process, per step (µs): p50 / p99 / p99.9\n")
        print("| condition | step | scheduler | tokenizer |")
        print("|---|---|---|---|")
        for r in lat:
            f = lambda s: f"{s['p50']:.0f} / {s['p99']:.0f} / {s['p999']:.0f}"  # noqa: E731
            print(f"| {r['probe']} | {f(r['step_us'])} | {f(r['scheduler_us'])} | {f(r['tokenizer_us'])} |")


if __name__ == "__main__":
    main()
