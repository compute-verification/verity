"""sw_gpu's measurements in MH100 units (one dense int8 MAC = 1; the H200's ruler 4,096 MACs an SM-clock x 132 SMs).

    python units.py [--bench OUT] [--k32 OUT] [--denoise OUT] [--hash OUT] [--json FILE]   markdown on stdout

Each OUT is a run's out/ directory: run.sh's (bench.jsonl, k32.jsonl and their SASS gates), denoise.sh's (denoise.jsonl) and
hash_cost.sh's (one directory an arm, each with hash_cost.json). A time t at SM clock f is t f x 540,672 MAC-times: the
work the H200's tensor cores could have done in it (time-equivalent units). Priced work is the H32 floor: 32 units a
register a thread writes, from the gated SASS loops (every instruction but stores, branches, compares and barriers).
A per-call figure is the measured call at that operand size; a forward sums Llama-3.2-1B's 16 layers of qkv, o, gu and
down (as denoise/table.py's model), and the DeepSeek-V4-like layer one call each of v4sq, v4gu and v4dn.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RULER = 4096 * 132
H32 = 32
NK = {"qkv": (3072, 2048), "o": (2048, 2048), "gu": (16384, 2048), "down": (2048, 8192), "head": (128256, 2048),
      "v4sq": (7168, 7168), "v4gu": (4096, 7168), "v4dn": (7168, 2048)}
MODELS = {"llama-3.2-1b forward": (16, ("qkv", "o", "gu", "down")), "v4-like layer": (1, ("v4sq", "v4gu", "v4dn"))}
MS = (64, 512, 8192)
UNPRICED = {"STG", "STS", "BRA", "ISETP", "EXIT", "NOP", "BSSY", "BSYNC", "WARPSYNC", "ENDCOLLECTIVE", "BAR", "MEMBAR", "FENCE",
            "WARPGROUP", "IGMMA"}


def units(ms: float, mhz: float) -> float:
    return ms * 1e-3 * mhz * 1e6 * RULER


def jl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.startswith("{")]


def priced(loop: dict) -> int:
    return sum(n for op, n in loop.items() if op not in UNPRICED and not op.startswith("U"))


def gate_loops(path: Path) -> dict[str, list[dict]]:
    return {r["kernel"]: r["loops"] for r in jl(path)}


def bench(out: Path, res: dict) -> list[str]:
    rows = jl(out / "bench.jsonl")
    loops = gate_loops(out / "gate_sw_bench.jsonl")

    def kern(sub: str) -> list[dict]:
        return next(v for k, v in loops.items() if sub in k)

    # priced work an entry: the loop a thread iterates (16 entries; a compression's 64 for the coins, its 16-round body 4 times)
    work = {"copy": priced(kern("_Z6copy_k")[1]) / 16, "form1": priced(kern("form1_kILb0E")[1]) / 16,
            "form1s": priced(kern("form1_kILb1E")[1]) / 16, "formncp": priced(kern("formncp_k")[1]) / 16}
    for item, sub, inner, outer in (("coins", "_Z7coins_k", 1, 2), ("coinsform1", "coinsform1_k", 1, 2),
                                    ("coinsncp", "coinsncp_k", 2, 3)):
        ls = kern(sub)
        work[item] = (4 * priced(ls[inner]) + priced(ls[outer])) / 64
    res["bench_work_per_entry"] = {k: H32 * v for k, v in work.items()}
    by = {(r["item"], r["entries"]): r for r in rows}
    sizes = sorted({r["entries"] for r in rows})
    items = list(dict.fromkeys(r["item"] for r in rows))
    out_md = ["### Forming and coins, whole calls (sw_bench): time-equivalent units an entry", "",
              "| item | priced work an entry (H32) | " + " | ".join(f"{s:,}" for s in sizes) + " |",
              "|---|---|" + "---|" * len(sizes)]
    for it in items:
        out_md.append(f"| {it} | {H32 * work[it]:.1f} | " + " | ".join(
            f"{by[(it, s)]['units_per_entry']:.1f}" if (it, s) in by else "" for s in sizes) + " |")
    out_md += ["", "Time a call (µs) and SM MHz at each size:", "", "| item | " + " | ".join(f"{s:,}" for s in sizes) + " |",
               "|---|" + "---|" * len(sizes)]
    for it in items:
        out_md.append(f"| {it} | " + " | ".join(f"{1e3 * by[(it, s)]['ms']:.1f} @{by[(it, s)]['mhz']:.0f}" if (it, s) in by else ""
                                              for s in sizes) + " |")
    res["bench"] = rows
    res["bench_by"] = {f"{it}:{s}": by[(it, s)] for it, s in by}
    return out_md


def per_call_bench(res: dict, plain: dict) -> list[str]:
    """The forming and coin items at each call's operand sizes, over the call's plain GEMM (denoise's plain arm)."""
    by = res.get("bench_by")
    if not by:
        return []
    md = ["### Per call and per forward: forming and coins for one side (X = A side, m x K; W = weight side, N x K) and both",
          "", "Units a call, and over the call's plain cuBLASLt GEMM (its own measured time and clock):", ""]
    md.append("| shape | m | plain units | MACs | coins X | coins W | form1 X | form1 W | coins+form1 fused X | fused W | "
              "NCP form X (3 copies) | NCP coins+form fused X | both sides unfused (coins+form1) | both fused |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    tot: dict = {}
    for lin, (n, k) in NK.items():
        for m in MS:
            p = plain.get((lin, m))
            if p is None:
                continue
            ex, ew = m * k, n * k
            g = lambda it, e: by.get(f"{it}:{e}", {}).get("units")  # noqa: E731
            vals = {"coins_x": g("coins", ex), "coins_w": g("coins", ew), "form1_x": g("form1", ex), "form1_w": g("form1", ew),
                    "cf1_x": g("coinsform1", ex), "cf1_w": g("coinsform1", ew), "ncp_x": g("formncp", ex),
                    "cncp_x": g("coinsncp", ex)}
            if any(v is None for v in vals.values()):
                continue
            vals["both_unfused"] = vals["coins_x"] + vals["form1_x"] + vals["coins_w"] + vals["form1_w"]
            vals["both_fused"] = vals["cf1_x"] + vals["cf1_w"]
            pu = p["units"]
            md.append(f"| {lin} | {m} | {pu:.3e} | {m * n * k:.3e} | " + " | ".join(
                f"{vals[c]:.3e} ({vals[c] / pu:.2f}×)" for c in ("coins_x", "coins_w", "form1_x", "form1_w", "cf1_x", "cf1_w",
                                                                 "ncp_x", "cncp_x", "both_unfused", "both_fused")) + " |")
            for name, (reps, lins) in MODELS.items():
                if lin in lins:
                    t = tot.setdefault((name, m), {"plain": 0.0, "macs": 0.0, **{c: 0.0 for c in vals}})
                    t["plain"] += reps * pu
                    t["macs"] += reps * m * n * k
                    for c, v in vals.items():
                        t[c] += reps * v
    md += ["", "Per forward (sums of the calls above):", "",
           "| model | m | plain units | MACs | coins X | coins W | form1 X | form1 W | fused X | fused W | NCP form X | NCP fused X | "
           "both unfused | both fused |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (name, m), t in sorted(tot.items()):
        md.append(f"| {name} | {m} | {t['plain']:.3e} | {t['macs']:.3e} | " + " | ".join(
            f"{t[c] / t['plain']:.3f}×" for c in ("coins_x", "coins_w", "form1_x", "form1_w", "cf1_x", "cf1_w", "ncp_x", "cncp_x",
                                                 "both_unfused", "both_fused")) + " |")
    res["per_forward_bench"] = {f"{n}:{m}": t for (n, m), t in tot.items()}
    return md


def k32(out: Path, res: dict) -> list[str]:
    rows = jl(out / "k32.jsonl")
    plain = {r["occ"]: r for r in rows if r["mode"] == "plain"}
    formed = {"forma": 64 * 32, "formb": 256 * 32, "formab": 320 * 32, "formars": 64 * 32}
    md = ["### Every-k32 commit and forming beside wgmma (k32): m64n256k32 s8, operands resident in shared memory", "",
          "A mode with a period L exports a word every L steps of K (every L/32-th wgmma); a word is one s32 accumulator word "
          "exported, and the extra is over plain at the same CTAs an SM.", "",
          "| mode | every L | CTAs an SM | MACs an SM-clock | units a MAC | units a word | over plain | extra units a word | "
          "extra units a MAC | extra units a formed entry | MHz |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        p = plain.get(r["occ"]) or plain[max(plain)]
        extra = r["units"] - p["units"] * r["steps"] / p["steps"] * r["ctas"] / p["ctas"]
        ent = r["ctas"] * r["steps"] * formed[r["mode"]] if r["mode"] in formed else None
        r["over_plain"] = r["units"] / (p["units"] * r["ctas"] / p["ctas"]) - 1
        r["extra_per_word"] = extra / r["words"]
        r["extra_per_mac"] = extra / r["macs"]
        r["extra_per_formed_entry"] = extra / ent if ent else None
        every = 32 * r.get("period", 1) if r["mode"] in ("wait0", "xor", "fold", "sts", "stg") else ""
        md.append(f"| {r['mode']} | {every} | {r['occ']} | {r['mac_per_sm_clk']:.0f} | {r['units_per_mac']:.3f} | "
                  f"{r['units_per_word']:.1f} | {r['over_plain']:+.3f} | {r['extra_per_word']:.1f} | {r['extra_per_mac']:+.4f} | "
                  + (f"{r['extra_per_formed_entry']:.2f}" if ent else "") + f" | {r['mhz']:.0f} |")
    res["k32"] = rows
    return md


def denoise(out: Path, res: dict) -> tuple[list[str], dict]:
    rows = jl(out / "denoise.jsonl")
    plain, md = {}, []
    md += ["### Dither's periodic-noise correction and the 3K-wide product (denoise.cu's arms), in units", "",
           "Per call: the plain GEMM's units and units a MAC (1 is the tensor cores' peak), then each part's units over the plain "
           "GEMM's. Periodic correction = peel (fused as more K) + Z_X (forming the tail rows) + wrap (the sparse correction); "
           "3K product = fresh3k, cuBLASLt [X | X | V][W' | -U | -B']^T at 3K (NCP's shape); coins = bits_a and bits_b (v3.1's "
           "SHA-512 coin streams for d_A and d_B), form = form_x and form_w (X and W' alone).", "",
           "| shape | m | plain units | units a MAC | corr 32 | corr 128 | corr 512 | 3K product | 3K − noised | bits_a | bits_b | "
           "form_x | form_w | MHz |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    agg: dict = {}
    for r in sorted(rows, key=lambda r: (r["m"], r["shape"])):
        lin, m = r["shape"].rsplit("-m", 1)
        m = int(m)
        a = {k: units(v["ms"], v["mhz"]) for k, v in r["arms"].items()}
        macs = r["m"] * r["n"] * r["k"]
        p = a["plain"]
        plain[(lin, m)] = {"units": p, "macs": macs, "ms": r["arms"]["plain"]["ms"], "mhz": r["arms"]["plain"]["mhz"]}
        corr = {lp: (a[f"ext-{lp}"] - a["noised"]) + (a[f"form_a-{lp}"] - a["form_x"]) + a[f"wrap-{lp}"] for lp in (32, 128, 512)}
        q = {"plain": p, "macs": macs, **{f"corr{lp}": v for lp, v in corr.items()}, "k3": a["fresh3k"],
             "k3x": a["fresh3k"] - a["noised"], "bits_a": a["bits_a"], "bits_b": a["bits_b"], "form_x": a["form_x"],
             "form_w": a["form_w"]}
        md.append(f"| {lin} | {m} | {p:.3e} | {p / macs:.3f} | " + " | ".join(
            f"{q[c] / p:.3f}" for c in ("corr32", "corr128", "corr512", "k3", "k3x", "bits_a", "bits_b", "form_x", "form_w"))
            + f" | {r['arms']['plain']['mhz']} |")
        for name, (reps, lins) in MODELS.items():
            if lin in lins:
                t = agg.setdefault((name, m), {c: 0.0 for c in q})
                for c, v in q.items():
                    t[c] += reps * v
    md += ["", "| model | m | plain units | units a MAC | corr 32 | corr 128 | corr 512 | 3K product | 3K − noised | bits_a | bits_b | "
           "form_x | form_w |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (name, m), t in sorted(agg.items()):
        md.append(f"| {name} | {m} | {t['plain']:.3e} | {t['plain'] / t['macs']:.3f} | " + " | ".join(
            f"{t[c] / t['plain']:.3f}" for c in ("corr32", "corr128", "corr512", "k3", "k3x", "bits_a", "bits_b", "form_x", "form_w"))
            + " |")
    res["denoise_models"] = {f"{n}:{m}": t for (n, m), t in agg.items()}
    res["plain"] = {f"{k[0]}:{k[1]}": v for k, v in plain.items()}
    return md, plain


def smi_clock(out: Path) -> list[tuple[float, float]]:
    """(UTC epoch s, SM MHz) from run.sh's or hash_cost.sh's nvidia-smi samples."""
    import datetime
    path, got = out / "smi.csv", []
    if not path.exists():
        return got
    for line in path.read_text().splitlines()[1:]:
        f = [x.strip() for x in line.split(",")]
        try:
            t = datetime.datetime.strptime(f[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc).timestamp()
            got.append((t, float(f[1].split()[0])))
        except (ValueError, IndexError):
            continue
    return got


def hashing(out: Path, res: dict) -> list[str]:
    md = ["### hm96-sha512 rows (the served committer's leaves): alone and beside the served int8 GEMM", "",
          "Alone at the data sheet's f_max (1,980 MHz), and at the SM clock nvidia-smi sampled over the alone leg; beside from the "
          "GEMM's loss in SM-cycles at its own clock. A word is 4 bytes (an s32 accumulator word); a plain k32 word is 32 MAC-times "
          "of tensor work.", "",
          "| arm | row | T | cap (CTAs) | hash GB/s | G compressions/s | GEMM loss (Mcycles) | units a byte | units a word | "
          "words' cost over their 32 MACs | kappa |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    hs = []
    clock = smi_clock(out)
    for d in sorted(p for p in out.iterdir() if (p / "hash_cost.json").exists()):
        j = json.loads((d / "hash_cost.json").read_text())
        marks = j.get("clock_marks", [])
        for rb, s in j["forms"]["t256"]["sizes"].items():
            h_alone = RULER * 1980e6 / (s["leaf_gbps"] * 1e9)
            i = next((n for n, (name, _) in enumerate(marks) if name == f"t256:{rb}:alone"), None)
            win = [f for t, f in clock if i is not None and i + 1 < len(marks) and marks[i][1] <= t < marks[i + 1][1]]
            f_alone = sorted(win)[len(win) // 2] if win else None
            h_meas = RULER * f_alone * 1e6 / (s["leaf_gbps"] * 1e9) if f_alone else None
            hs.append({"arm": d.name, "row": int(rb), "T": j["T"], "cap": "alone", "gbps": s["leaf_gbps"],
                       "gcomp": s["leaf_gcomp_per_s"], "h": h_alone, "h_sampled": h_meas, "mhz_sampled": f_alone})
            meas = f" ({h_meas:.0f} at {f_alone:.0f} MHz)" if h_meas else ""
            md.append(f"| {d.name} | {int(rb) // 1024} KiB | {j['T']} | alone | {s['leaf_gbps']} | {s['leaf_gcomp_per_s']} | | "
                      f"{h_alone:.0f}{meas} | {4 * h_alone:.0f} | {4 * h_alone / 32:.0f}× | |")
            for b in s.get("beside", []):
                h = b["delta_mcycles"] * 1e6 * RULER / b["bytes"]
                hs.append({"arm": d.name, "row": int(rb), "T": j["T"], "cap": b["cap"], "gbps": b["hash_gbps"], "h": h,
                           "kappa": b["kappa_window"], "delta_mcycles": b["delta_mcycles"], "bytes": b["bytes"]})
                md.append(f"| {d.name} | {int(rb) // 1024} KiB | {j['T']} | {b['cap'] or 'all'} | {b['hash_gbps']} | | "
                          f"{b['delta_mcycles']:.1f} | {h:.0f} | {4 * h:.0f} | {4 * h / 32:.0f}× | {b['kappa_window']} |")
    res["hash"] = hs
    return md


def main() -> int:
    ap = argparse.ArgumentParser()
    for a in ("bench", "k32", "denoise", "hash", "json"):
        ap.add_argument(f"--{a}")
    ns = ap.parse_args()
    res: dict = {}
    md: list[str] = []
    plain: dict = {}
    if ns.denoise:
        part, plain = denoise(Path(ns.denoise), res)
        md += part + [""]
    if ns.bench:
        md += bench(Path(ns.bench), res) + [""]
        if plain:
            md += per_call_bench(res, plain) + [""]
    if ns.k32:
        md += k32(Path(ns.k32), res) + [""]
    if ns.hash:
        md += hashing(Path(ns.hash), res) + [""]
    print("\n".join(md))
    if ns.json:
        Path(ns.json).write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
