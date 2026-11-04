"""T2 for ssn_1l from ssn.cu's timings: per linear and per forward, in milliseconds and MH100 units (one dense int8 MAC = 1).

    python3 t2.py OUT [--stock M:SYNC_MS[:ASYNC_MS][:GEMM_MS] ...] [--json FILE]

OUT is a run's out/ (time_m<M>[_cap<C>].jsonl). Each --stock gives, at M rows, the stock step's time (synchronous, and asynchronous
if known) and the stock GEMMs' time a forward, all in ms. T2 is the step with ssn_1l's linears over the stock step, two ways:
  own   (stock + forward - forward's plain GEMM on the same kernel) / stock: ssn_1l's extra over this kernel's own plain GEMM
  swap  (stock - stock GEMM + forward) / stock: this forward in place of the stock GEMMs
where forward is the 16 layers' four linears with root(A_j), the certifier's feed, forming, every k32 word exported and hashed, run serially (one
stream) or beside (each chunk's hash on a second stream while the next chunk's GEMM runs), for each hash scheme (hasher:
rowseg, rows<B> for the served committer's hm96 leaves of B-byte rows, or plain<B> for L1's plain frame-v3 leaves of B-byte
rows). beside_bare (BARE=1) is beside with nothing ahead of each call's GEMM, and beside_crit beside's GEMM stream alone (the
work ahead of each GEMM and the GEMMs, no hash): controls, not forwards of ssn_1l.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LINEARS = ("qkv", "o", "gate_up", "down")


def load(out: Path) -> dict[str, list[dict]]:
    runs = {}
    for f in sorted(out.glob("time_m*.jsonl"), key=lambda f: (int(f.stem.split("_")[1][1:]), f.stem)):
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.startswith("{")]
        if rows:
            runs[f.stem] = rows
    return runs


def summarize(rows: list[dict], stock: dict | None) -> dict:
    by = {(r["what"], r.get("linear"), r.get("hasher", "rowseg" if "hash" in r["what"] or r["what"][4:] in ("serial", "beside")
                                                    else "none")): r for r in rows if "linear" in r}
    hashers = sorted({h for (_, _, h) in by if h not in ("none", "l1")})
    lin = {}
    for name in LINEARS:
        g = lambda w, h="none": by.get((w, name, h))  # noqa: E731
        if not g("plain_gemm"):
            continue
        p, s, e = g("plain_gemm"), g("ssn_gemm"), g("ssn_gemm_export")
        words = e["words"]
        lin[name] = {
            "n": p["n"], "K": p["K"], "S": p["S"], "BN": p["BN"], "words": words, "stock_macs": p["stock_macs"],
            "plain_ms": p["ms"], "ssn_ms": s["ms"], "alpha_time": s["ms"] / p["ms"], "alpha_layout": 32 * p["S"] / p["K"],
            "export_ms": e["ms"], "export_units_per_word_over_ssn": (e["units"] - s["units"]) / words,
            "feed_us": 1e3 * g("feed")["ms"], "feed_bytes": int(g("feed")["extra"]), "forming_us": 1e3 * g("forming")["ms"],
            "feed_forming_us": 1e3 * g("feed_forming")["ms"],
            "hash": {h: {"ms": r["ms"], "units_per_byte": r["units_per_byte"], "gb_per_s": 4 * words / (r["ms"] * 1e6)}
                     for h in hashers if (r := g("hash_pass", h))},
            "tree_ms": {h: r["ms"] for h in hashers if (r := g("hash_tree", h))},
            "root_a_us": {h: 1e3 * r["ms"] for h in hashers if (r := g("root_a_hash", h))},
            "root_a_root_us": {h: 1e3 * r["ms"] for h in hashers if (r := g("root_a_root", h))},
            "g_copy_us": 1e3 * r["ms"] if (r := g("g_copy")) else None,
            "l1_root_us": 1e3 * r["ms"] if (r := g("l1_root", "l1")) else None,
            "l1_path_us": 1e3 * r["ms"] if (r := g("l1_path", "l1")) else None,
            "l1_root2_us": 1e3 * r["ms"] if (r := g("l1_root2", "l1")) else None,
            "l1_path2_us": 1e3 * r["ms"] if (r := g("l1_path2", "l1")) else None,
            "l1_cta2": int(r["extra"]) if (r := g("l1_root2", "l1")) else None,
            "lane_link_us": 1e3 * r["ms"] if (r := g("lane_link")) else None,
            "lane_coin_bytes": int(r["extra"]) if (r := g("lane_link")) else None,
        }
    fwd = {"plain_gemm": by.get(("fwd_plain_gemm", "all", "none"))}
    for h in hashers:
        for sched in ("serial", "beside", "beside_bare", "beside_crit"):
            if (r := by.get((f"fwd_{sched}", "all", h))):
                fwd[f"{sched}/{h}"] = r
    clock = next((r for r in rows if r["what"] == "clock"), {})
    first = next(r for r in rows if "mhz" in r)
    small = [r for r in rows if r["what"] == "small_leaves_check"]
    out = {"M": first["M"], "mhz": first["mhz"], "root_a_in_forward": clock.get("root_a_in_forward", 0),
           "root_a_rows": clock.get("root_a_rows", -1), "root_a_tree": clock.get("root_a_tree", 0),
           "words_tree": clock.get("words_tree", 0), "rowseg_leaf": clock.get("rowseg_leaf", "segment digests"),
           "prio": clock.get("prio", 0), "layout": clock.get("layout", "ssn_1l"), "l1": clock.get("l1", 0),
           "l1_cta": clock.get("l1_cta", 0), "split_sms": clock.get("split_sms", [0, 0]),
           "beside_buffers": clock.get("beside_buffers", 0), "blc_w": clock.get("blc_w", [1, 1, 1]),
           "small_threads": clock.get("small_threads", 0),
           "small_rows_max": clock.get("small_rows_max", 0),
           "small_leaves_check": {"cases": len(small), "leaves": sum(r["leaves"] for r in small),
                                  "mismatched": sum(r["mismatched_leaves"] for r in small)},
           "linears": lin,
           "forward": {k: {"ms": r["ms"], "units": r["units"], "units_per_stock_mac": r["units_per_stock_mac"],
                           "units_per_word": r["units_per_word"], "words": r["words"], "stock_macs": r["stock_macs"]}
                       for k, r in fwd.items() if r}}
    if stock and fwd.get("plain_gemm"):
        plain = fwd["plain_gemm"]["ms"]
        t2 = {}
        for k, r in fwd.items():
            if k == "plain_gemm" or k.startswith(("beside_bare/", "beside_crit/")) or not r:
                continue
            for kind in ("sync", "async"):
                st = stock.get(kind)
                if st is None:
                    continue
                t2[f"{k} {kind} own"] = (st + r["ms"] - plain) / st
                if stock.get("gemm") is not None:
                    t2[f"{k} {kind} swap"] = (st - stock["gemm"] + r["ms"]) / st
        out["stock"] = stock
        out["t2"] = t2
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--stock", action="append", default=[])
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    stocks = {}
    for s in a.stock:
        parts = s.split(":")
        v = [float(x) if x else None for x in parts[1:]] + [None] * 3
        stocks[int(parts[0])] = {"sync": v[0], "async": v[1], "gemm": v[2]}
    res = []
    for name, rows in load(a.out).items():
        res.append(summarize(rows, stocks.get(rows[0]["M"])) | {"file": name})
    for r in res:
        print(f"## {r['file']}: M {r['M']} (SM clock {r['mhz']:.0f} MHz)")
        print("| linear | S | BN | plain ms | ssn ms | α time / layout | export ms | export units a word | feed µs | "
              "forming µs | feed+forming+G µs | hash ms, units a byte, GB/s by scheme |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for n, x in r["linears"].items():
            hs = "; ".join(f"{h} {v['ms']:.3f} / {v['units_per_byte']:.0f} / {v['gb_per_s']:.0f}" for h, v in x["hash"].items())
            print(f"| {n} | {x['S']} | {x['BN']} | {x['plain_ms']:.4f} | {x['ssn_ms']:.4f} | {x['alpha_time']:.3f} / "
                  f"{x['alpha_layout']:.3f} | {x['export_ms']:.3f} | {x['export_units_per_word_over_ssn']:.0f} | "
                  f"{x['feed_us']:.1f} | {x['forming_us']:.1f} | {x['feed_forming_us']:.1f} | {hs} |")
        for n, x in r["linears"].items():
            if x["tree_ms"]:
                print(f"- the words' tree above their leaves, {n}: " + "; ".join(f"{h} {v:.3f} ms ({v / x['hash'][h]['ms']:.1%} "
                                                                                 "of the leaves)" for h, v in x["tree_ms"].items()))
            if x["root_a_us"]:
                print(f"- root(A_j) leaves, {n}: " + "; ".join(f"{h} {v:.1f} µs" for h, v in x["root_a_us"].items()))
            if x["root_a_root_us"]:
                print(f"- root(A_j) leaves and tree, {n}: " + "; ".join(f"{h} {v:.1f} µs" for h, v in x["root_a_root_us"].items()))
            print(f"- the certifier's feed, {n}: {x['feed_bytes']} bytes, {x['feed_us']:.1f} µs host to device")
            if x["g_copy_us"] is not None:
                print(f"- G's copy, {n}: {x['g_copy_us']:.1f} µs")
            if x["l1_root_us"] is not None:
                print(f"- L1 + (a), {n}: root(A_j) {x['l1_root_us']:.1f} µs; with root(A_j) up, the feed back, forming and G "
                      f"{x['l1_path_us']:.1f} µs")
            if x["l1_root2_us"] is not None:
                print(f"- L1 + (a) for latency (CTAs of {x['l1_cta2']}), {n}: root(A_j) {x['l1_root2_us']:.1f} µs; with "
                      f"root(A_j) up, the feed back, forming and G {x['l1_path2_us']:.1f} µs")
            if x["lane_link_us"] is not None:
                print(f"- blc4a-t64-e7's link, {n}: A_j to the host and the call's {x['lane_coin_bytes'] / 1e6:.2f} MB of lane "
                      f"coins back, {x['lane_link_us']:.1f} µs")
        print(f"- layout {r['layout']}" + (f", A' = ({', '.join(map(str, r['blc_w']))}) in the lane-coin lanes"
                                           if r["layout"].startswith("blc") else ""))
        if r["l1"] == 2:
            print(f"- the forwards below compute root(A_j) by L1 + (a) for latency (CTAs of at least {r['l1_cta']}), then "
                  "send it up and take the feed back, ahead of each call's forming and G")
        elif r["l1"]:
            print("- the forwards below compute root(A_j) by L1 + (a), then send it up and take the feed back, ahead of "
                  "each call's forming and G")
        elif r["root_a_in_forward"]:
            ra = r["root_a_rows"]
            by_scheme = "the words' scheme" if ra < 0 else f"rows{ra}" if ra else "rowseg"
            what = "root(A_j), leaves and tree," if r["root_a_tree"] else "A_j's leaves"
            print(f"- the forwards below compute {what} ahead of each call's feed, by {by_scheme}")
        print(f"- the forwards below hash the words' leaves{' and the tree above each chunk' if r['words_tree'] else ''}; "
              f"rowseg's leaf: {r['rowseg_leaf']}")
        if r["small_threads"]:
            c = r["small_leaves_check"]
            print(f"- leaves launches under {r['small_rows_max']} rows use the committer built with {r['small_threads']}-thread "
                  f"blocks; its leaves against the served build's: {c['mismatched']} of {c['leaves']} differ ({c['cases']} cases)")
        if r["prio"]:
            print("- the GEMM stream runs at the greatest stream priority, the hash streams at the least")
        if r["split_sms"][0]:
            print(f"- beside's GEMM stream runs in a green context of {r['split_sms'][0]} SMs, its hash streams in one of "
                  f"{r['split_sms'][1]} (no serial forward)")
        print(f"- beside's buffers, each with its own hash stream: {r['beside_buffers']}")
        for w, x in r["forward"].items():
            print(f"- forward {w}: {x['ms']:.3f} ms, {x['units_per_stock_mac']:.3f} units a stock MAC")
        for w, x in r["forward"].items():
            if w.startswith("beside_bare/") and (full := r["forward"].get("beside/" + w.split("/")[1])):
                print(f"- the work ahead of each call's GEMM costs beside/{w.split('/')[1]} {full['ms'] - x['ms']:.3f} ms a "
                      f"forward over beside_bare, which runs none")
        for k, v in r.get("t2", {}).items():
            print(f"- T2 {k}: {v:.2f}x")
        print()
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
