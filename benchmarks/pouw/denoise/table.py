"""denoise.cu's JSON lines as overheads over the plain int8 GEMM (cuBLASLt on A, B), per shape and for the model.

Every number is a fraction of the shape's `plain` arm, so 0.25 is +25% of the plain GEMM's time. Shapes are named
LINEAR-mM (denoise.sh): Llama-3.2-1B's qkv (3,072 x 2,048), o (2,048 x 2,048), gu (gate and up fused, 16,384 x 2,048) and
down (2,048 x 8,192), 16 layers of each, and the head (128,256 x 2,048) on its own row, out of the model's total (vLLM applies
it only to the positions it samples, and whether it is int8 isn't the served path's question here).

The structures, each a call's overhead (the weight side, W' and Y_W, is formed once a period, so it is 0 a call; its
per-call price is shown beside):
  periodic L', de-noised   coins_a(L') + form_a(L') + (noised - plain) + (ext(L') - noised) + wrap(L'): X with Z_X and the
                           wrap lists, the noised GEMM, the peel fused as more K, the sparse wrap correction
  periodic L', recompute   coins_a(L') + form_x + (noised - plain) + recompute: O from a plain GEMM on A + d_A
  fresh, recompute         bits_a + noise_vk + form_xk + (noised - plain) + recompute
  fresh, more GEMMs        bits_a + noise_vk + form_xk + (fresh3k - plain): a lower bound (no wrap digits, no wrap correction)
coins_a(L') is bits_a + noise_v(L'). T_s's own cost (seg, the prefix sums through a library GEMM) is shown apart: v31's
GEMM stores them from its registers.

The served call (`served-LP`, where the JSON has it) is measured whole: d_A's bits, V, X, the noised GEMM over k < (S - 1) L
at L = Lw with every T_s left in DRAM, a' and O's plain GEMM; its overhead is served - plain. Beside it: ts_srv (that noised
GEMM alone), the launch floor (nop), and the plain GEMM writing int8 or bf16 instead of int32.

    python3 table.py DENOISE.jsonl [--md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LAYERS = 16
LINEARS = ("qkv", "o", "gu", "down")
LPS = (32, 128, 512)


def parts(r: dict) -> dict[str, float]:
    """One shape's numerators, ms a call, by name (the table's columns and the structures' sums)."""
    a = {k: v["ms"] for k, v in r["arms"].items()}
    p, nz = a["plain"], a["noised"]
    out = {"plain": p, "noised_extra": nz - p, "form_x": a["form_x"], "bits_a": a["bits_a"],
           "w_call": a["bits_b"] + a["coins_w-128"] + a["form_w"], "recompute": a["recompute"],
           "fresh_a": a["bits_a"] + a["noise_vk"] + a["form_xk"], "fresh3k_extra": a["fresh3k"] - nz}
    for lp in LPS:
        coins = a["bits_a"] + a[f"noise_v-{lp}"]
        out[f"coins_a-{lp}"] = coins
        out[f"peel-{lp}"] = a[f"ext-{lp}"] - nz
        out[f"sep-{lp}"] = a[f"sep-{lp}"]
        out[f"zx-{lp}"] = a[f"form_a-{lp}"] - a["form_x"]
        out[f"wrap-{lp}"] = a[f"wrap-{lp}"]
        out[f"seg-{lp}"] = a[f"seg-{lp}"] - nz
        out[f"yw_call-{lp}"] = a["bits_b"] + a[f"coins_w-{lp}"] + a[f"form_b-{lp}"]
        out[f"per_denoise-{lp}"] = coins + a[f"form_a-{lp}"] + (nz - p) + (a[f"ext-{lp}"] - nz) + a[f"wrap-{lp}"]
        out[f"per_recompute-{lp}"] = coins + a["form_x"] + (nz - p) + a["recompute"]
    out["fresh_recompute"] = out["fresh_a"] + (nz - p) + a["recompute"]
    out["fresh_gemms"] = out["fresh_a"] + (a["fresh3k"] - p)
    if "ts_srv" in a:
        out["ts_srv"] = a["ts_srv"]
        out["nop"] = a["nop"]
        for lp in LPS:
            out[f"served-{lp}"] = a[f"served-{lp}"] - p
        for o in ("plain_i8", "plain_bf16"):
            if o in a:
                out[o] = a[o]
        out["ts_floor"] = (r["Sw"] - 1) * r["Lw"] / r["k"] * a.get("plain_i8", p)
        for lp in LPS:
            out[f"served_floor-{lp}"] = out[f"served-{lp}"] - a["ts_srv"] + out["ts_floor"]
    return out


def load(paths: list[str]) -> dict[tuple[str, int], dict]:
    rows = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.startswith("{"):
                r = json.loads(line)
                lin, m = r["shape"].rsplit("-m", 1)
                rows[(lin, int(m))] = r
    return rows


COLS = [("plain µs", lambda q: 1e3 * q["plain"], "{:.1f}"),
        ("coins_a 128", lambda q: q["coins_a-128"] / q["plain"], "{:.3f}"),
        ("X (form_x)", lambda q: q["form_x"] / q["plain"], "{:.3f}"),
        ("noised − plain", lambda q: q["noised_extra"] / q["plain"], "{:+.3f}"),
        *[(f"peel {lp}", (lambda lp: lambda q: q[f"peel-{lp}"] / q["plain"])(lp), "{:.3f}") for lp in LPS],
        *[(f"sep {lp}", (lambda lp: lambda q: q[f"sep-{lp}"] / q["plain"])(lp), "{:.3f}") for lp in LPS],
        *[(f"Z_X {lp}", (lambda lp: lambda q: q[f"zx-{lp}"] / q["plain"])(lp), "{:.3f}") for lp in LPS],
        *[(f"wrap {lp}", (lambda lp: lambda q: q[f"wrap-{lp}"] / q["plain"])(lp), "{:.3f}") for lp in LPS],
        *[(f"T_s lib {lp}", (lambda lp: lambda q: q[f"seg-{lp}"] / q["plain"])(lp), "{:.3f}") for lp in LPS],
        ("recompute", lambda q: q["recompute"] / q["plain"], "{:.3f}"),
        ("fresh A side", lambda q: q["fresh_a"] / q["plain"], "{:.3f}"),
        ("fresh 3K − noised", lambda q: q["fresh3k_extra"] / q["plain"], "{:.3f}"),
        ("W′ a call", lambda q: q["w_call"] / q["plain"], "{:.3f}")]
TOT = [*[(f"periodic {lp} de-noised", (lambda lp: lambda q: q[f"per_denoise-{lp}"] / q["plain"])(lp)) for lp in LPS],
       *[(f"periodic {lp} recompute", (lambda lp: lambda q: q[f"per_recompute-{lp}"] / q["plain"])(lp)) for lp in LPS],
       ("fresh recompute", lambda q: q["fresh_recompute"] / q["plain"]),
       ("fresh more GEMMs (lower bound)", lambda q: q["fresh_gemms"] / q["plain"])]


def model(rows: dict, m: int) -> dict[str, float] | None:
    """The model's sums at m: every numerator over the 16 layers' four linears."""
    qs = [parts(rows[(lin, m)]) for lin in LINEARS if (lin, m) in rows]
    if len(qs) != len(LINEARS):
        return None
    out = {k: LAYERS * sum(q[k] for q in qs) for k in qs[0] if all(k in q for q in qs)}
    if "nop" in out:
        out["nop"] /= LAYERS * len(qs)
    return out


SERVED = [*[(f"served call {lp}", (lambda lp: lambda q: q[f"served-{lp}"] / q["plain"])(lp), "{:.3f}") for lp in LPS],
          ("ts_srv", lambda q: q["ts_srv"] / q["plain"], "{:.3f}"),
          ("launch floor µs", lambda q: 1e3 * q["nop"], "{:.1f}"),
          ("plain int8 out", lambda q: q["plain_i8"] / q["plain"] if "plain_i8" in q else float("nan"), "{:.3f}"),
          ("plain bf16 out", lambda q: q["plain_bf16"] / q["plain"] if "plain_bf16" in q else float("nan"), "{:.3f}")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    rows = load(a.files)
    ms = sorted({m for _, m in rows})
    keyed = [((lin, m), parts(r)) for (lin, m), r in sorted(rows.items(), key=lambda x: (x[0][1], x[0][0]))]
    for m in ms:
        if (q := model(rows, m)) is not None:
            keyed.append((("model", m), q))
    print("Parts, each over the shape's plain GEMM (model: the 16 layers' sums)\n")
    print("| shape | m | " + " | ".join(c for c, _, _ in COLS) + " |")
    print("|---|---|" + "---|" * len(COLS))
    for (lin, m), q in keyed:
        print(f"| {lin} | {m} | " + " | ".join(f.format(fn(q)) for _, fn, f in COLS) + " |")
    print("\nA call's overhead over its plain GEMM, by structure (the weight side once a period)\n")
    print("| shape | m | " + " | ".join(c for c, _ in TOT) + " | best periodic 32 / 128 / 512 |")
    print("|---|---|" + "---|" * (len(TOT) + 1))
    for (lin, m), q in keyed:
        best = " / ".join(f"{min(q[f'per_denoise-{lp}'], q[f'per_recompute-{lp}']) / q['plain']:.3f}" for lp in LPS)
        print(f"| {lin} | {m} | " + " | ".join(f"{fn(q):.3f}" for _, fn in TOT) + f" | {best} |")
    if all("ts_srv" in q for _, q in keyed):
        lw = {r["Lw"] for r in rows.values()}
        print(f"\nThe served call, measured whole (L = {', '.join(map(str, sorted(lw)))}), over its plain GEMM; the GEMMs beside it\n")
        print("| shape | m | S | " + " | ".join(c for c, _, _ in SERVED) + " |")
        print("|---|---|---|" + "---|" * len(SERVED))
        for (lin, m), q in keyed:
            s = rows[(lin, m)]["Sw"] if lin != "model" else ""
            print(f"| {lin} | {m} | {s} | " + " | ".join(f.format(fn(q)) for _, fn, f in SERVED) + " |")
        print("\nEstimate, not measured: the served call with T_s hashed in the noised GEMM's epilogue (PROTOCOL.md) rather than left in "
              "DRAM, ts_srv replaced by (S - 1) L / K of the int8-out plain GEMM (of the int32 one where cuBLASLt has no int8 out); "
              "the hashing itself excluded\n")
        print("| shape | m | T_s floor | served call 32 | served call 128 | served call 512 |")
        print("|---|---|---|---|---|---|")
        for (lin, m), q in keyed:
            print(f"| {lin} | {m} | {q['ts_floor'] / q['plain']:.3f} | "
                  + " | ".join(f"{q[f'served_floor-{lp}'] / q['plain']:.3f}" for lp in LPS) + " |")
    print("\nRuns' conditions\n")
    print("| shape | m | GPU | SM MHz (plain) | steady | throttled arms | wrap rate A, B at L' 128 | flags |")
    print("|---|---|---|---|---|---|---|---|")
    for (lin, m), r in sorted(rows.items(), key=lambda x: (x[0][1], x[0][0])):
        thr = [k for k, v in r["arms"].items() if v.get("throttled")]
        lp = next(x for x in r["lp"] if x["lp"] == 128)
        flags = ",".join(str(x["flag"]) for x in r["lp"])
        print(f"| {lin} | {m} | {r['gpu']} | {r['arms']['plain']['mhz']:.0f} | {r['steady']} | {len(thr)} | "
              f"{lp['wrap_rate_A']:.4f}, {lp['wrap_rate_B']:.4f} | {flags} |")
    for m in ms:
        if (q := model(rows, m)) is not None:
            best = {lp: min(q[f"per_denoise-{lp}"], q[f"per_recompute-{lp}"]) / q["plain"] for lp in LPS}
            print(f"summary: model m {m}: plain {1e3 * q['plain'] / LAYERS:.1f} µs a layer; noise application "
                  f"(A side, L' 128) {(q['coins_a-128'] + q['form_x']) / q['plain']:.3f}; noised GEMM "
                  f"{q['noised_extra'] / q['plain']:+.3f}; periodic de-noise (peel + Z_X + wrap) "
                  + ", ".join(f"L' {lp} {(q[f'peel-{lp}'] + q[f'zx-{lp}'] + q[f'wrap-{lp}']) / q['plain']:.3f}" for lp in LPS)
                  + f"; plain recompute {q['recompute'] / q['plain']:.3f}; best call overhead periodic "
                  + ", ".join(f"L' {lp} {v:.3f}" for lp, v in best.items())
                  + f"; fresh {q['fresh_recompute'] / q['plain']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
