"""T2 for ssn_1l from ssn.cu's timings: per linear and per forward, in milliseconds and MH100 units (one dense int8 MAC = 1).

    python3 t2.py OUT [--stock M:SYNC_MS[:ASYNC_MS][:GEMM_MS] ...] [--json FILE]

OUT is a run's out/ (time_m<M>.jsonl). Each --stock gives, at M rows, the stock step's time (synchronous, and asynchronous
if known) and the stock GEMMs' time a forward, all in ms. T2 is the step with ssn_1l's linears over the stock step, two ways:
  own   (stock + forward - forward's plain GEMM on the same kernel) / stock: ssn_1l's extra over this kernel's own plain GEMM
  swap  (stock - stock GEMM + forward) / stock: this forward in place of the stock GEMMs
where forward is the 16 layers' four linears with coins, forming, every k32 word exported and hashed, run serially (one
stream) or beside (each chunk's hash on a second stream while the next chunk's GEMM runs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(out: Path) -> dict[int, list[dict]]:
    runs = {}
    for f in sorted(out.glob("time_m*.jsonl")):
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.startswith("{")]
        if rows:
            runs[rows[0]["M"]] = rows
    return runs


def summarize(rows: list[dict], stock: dict | None) -> dict:
    by = {(r["what"], r.get("linear")): r for r in rows if "what" in r}
    lin = {}
    for name in ("qkv", "o", "gate_up", "down"):
        g = lambda w: by.get((w, name))  # noqa: E731
        if not g("plain_gemm"):
            continue
        p, s, e, h = g("plain_gemm"), g("ssn_gemm"), g("ssn_gemm_export"), g("hash_pass")
        words = e["words"]
        lin[name] = {
            "n": p["n"], "K": p["K"], "S": p["S"], "BN": p["BN"], "words": words,
            "plain_ms": p["ms"], "ssn_ms": s["ms"], "alpha_time": s["ms"] / p["ms"], "alpha_layout": 32 * p["S"] / p["K"],
            "export_ms": e["ms"], "export_units_per_word_over_ssn": (e["units"] - s["units"]) / words,
            "export_units_per_stock_mac": e["units_per_stock_mac"],
            "hash_ms": h["ms"], "hash_units_per_byte": h["units_per_byte"], "hash_gb_per_s": 4 * words / (h["ms"] * 1e6),
            "coins_us": 1e3 * g("coins")["ms"], "forming_us": 1e3 * g("forming")["ms"],
            "coins_forming_us": 1e3 * g("coins_forming")["ms"],
        }
    f = {w: by[(w, "all")] for w in ("fwd_plain_gemm", "fwd_serial", "fwd_beside") if (w, "all") in by}
    out = {"M": rows[0]["M"], "mhz": rows[0]["mhz"], "linears": lin,
           "forward": {w: {"ms": r["ms"], "units_per_stock_mac": r["units_per_stock_mac"],
                           "units_per_word": r["units_per_word"], "words": r["words"]} for w, r in f.items()}}
    if stock and f:
        t2 = {}
        plain = f["fwd_plain_gemm"]["ms"]
        for sched in ("fwd_serial", "fwd_beside"):
            fw = f[sched]["ms"]
            for kind in ("sync", "async"):
                st = stock.get(kind)
                if st is None:
                    continue
                t2[f"{sched[4:]}_{kind}_own"] = (st + fw - plain) / st
                if stock.get("gemm") is not None:
                    t2[f"{sched[4:]}_{kind}_swap"] = (st - stock["gemm"] + fw) / st
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
    res = [summarize(rows, stocks.get(m)) for m, rows in sorted(load(a.out).items())]
    for r in res:
        print(f"## M {r['M']} (SM clock {r['mhz']:.0f} MHz at idle)")
        print("| linear | S | BN | plain ms | ssn ms | α time / layout | export ms | export units a word | hash ms | hash units a "
              "byte | hash GB/s | coins µs | forming µs | coins+forming µs |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for n, x in r["linears"].items():
            print(f"| {n} | {x['S']} | {x['BN']} | {x['plain_ms']:.4f} | {x['ssn_ms']:.4f} | {x['alpha_time']:.3f} / "
                  f"{x['alpha_layout']:.3f} | {x['export_ms']:.3f} | {x['export_units_per_word_over_ssn']:.0f} | "
                  f"{x['hash_ms']:.3f} | {x['hash_units_per_byte']:.0f} | {x['hash_gb_per_s']:.0f} | {x['coins_us']:.1f} | "
                  f"{x['forming_us']:.1f} | {x['coins_forming_us']:.1f} |")
        for w, x in r["forward"].items():
            print(f"- {w}: {x['ms']:.3f} ms, {x['units_per_stock_mac']:.3f} units a stock MAC, {x['words']:.4g} words")
        for k, v in r.get("t2", {}).items():
            print(f"- T2 {k}: {v:.2f}x")
        print()
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
