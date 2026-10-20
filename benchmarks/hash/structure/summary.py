"""run.sh's tables from a run's outputs (OUT): fusion, leaf size, tree arity and salts.

    python3 summary.py OUT [--json]

Fusion columns (ms, medians of interleaved reps, each from a stand-in producer's end): lt is cuBLASLt's s8 GEMM (s32 out);
g this GEMM alone; gs g then the separate data phase (leaves, then nodes); f4 / f8 the fused GEMM with 4 / 8 hash warps,
then nodes; h the separate data phase alone; salt the salt phase. "sep +" and "fused +" are each over g; "hidden" is the
share of the separate pass's cost that fusion removes, (gs - f) / (gs - g), at the better of f4 and f8.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NAMES = {(6144, 4096): "qkv", (4096, 4096): "o", (24576, 4096): "gate_up", (4096, 12288): "down"}


def lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text().splitlines():
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    return out


def fusion(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r.get("bench") != "fuse":
            continue
        a = {k: v["ms"] for k, v in r["arms"].items()}
        f = min(a["f4"], a["f8"])
        best = "f4" if a["f4"] <= a["f8"] else "f8"
        sep = a["gs"] - a["g"]
        out.append({
            "gpu": r["gpu"], "gemm": NAMES.get((r["n"], r["k"]), f"{r['n']}x{r['k']}"), "m": r["m"], "n": r["n"], "k": r["k"],
            "r": r["r"], "leaf_bytes": r["leaf_bytes"], "leaves": r["leaves"], "mb": r["operand_bytes"] / 1e6,
            "lt": a["lt"], "g": a["g"], "gs": a["gs"], "sep_leaves": r["arms"]["gs"]["part2_ms"], "rec": a["rec"],
            "f4": a["f4"], "f8": a["f8"], "fused_kernel": r["arms"][best]["part1_ms"], "h": a["h"], "salt": a["salt"],
            "sep_over_g": sep / a["g"], "fused_over_g": (f - a["g"]) / a["g"], "hidden": (a["gs"] - f) / sep if sep > 0 else 0.0,
            "best": best, "mhz": r["arms"]["g"]["mhz"], "g_over_lt": a["g"] / a["lt"],
        })
    return out


def render(o: Path) -> str:
    fz = fusion(lines(o / "bench-fuse.jsonl"))
    s = [f"# {o}", ""]
    if fz:
        s += [f"GPU: {fz[0]['gpu']}, SM clock ~{fz[0]['mhz']:.0f} MHz (NVML, per rep in the JSON)", "",
              "| GEMM | M x N x K | leaf | MB | lt | g | g/lt | gs (sep leaves) | rec | f4 | f8 | fused kernel | h | salt | sep + | fused + | hidden |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in fz:
            s.append(f"| {r['gemm']} | {r['m']}x{r['n']}x{r['k']} | {r['leaf_bytes']} B | {r['mb']:.1f} | {r['lt']:.4f} | {r['g']:.4f} | "
                     f"{r['g_over_lt']:.2f} | {r['gs']:.4f} ({r['sep_leaves']:.4f}) | {r['rec']:.4f} | {r['f4']:.4f} | {r['f8']:.4f} | "
                     f"{r['fused_kernel']:.4f} | {r['h']:.4f} | {r['salt']:.4f} | {100 * r['sep_over_g']:+.1f}% | "
                     f"{100 * r['fused_over_g']:+.1f}% | {100 * r['hidden']:.0f}% |")
    tr = [x["tree"] for x in lines(o / "tree.jsonl") if "tree" in x]
    if tr:
        s += ["", "| tree leaves | binary ms (levels) | 4-ary | 8-ary | 16-ary | committer binary |", "|---|---|---|---|---|---|"]
        for t in tr:
            s.append(f"| {t['leaves']} | " + " | ".join(f"{t[k]['ms']:.4f} ({t[k]['levels']})" for k in ("a2", "a4", "a8", "a16", "committer_binary")) + " |")
    ch = [x["chacha"] for x in lines(o / "chacha.jsonl") if "chacha" in x]
    if ch:
        s += ["", "| salts: leaves | ChaCha20 keystream ms (GB/s) | salt phase on ChaCha20 y ms | salt phase on SHA-512 PRF y ms |", "|---|---|---|---|"]
        for c in ch:
            s.append(f"| {c['leaves']} | {c['keystream_ms']:.4f} ({c['keystream_gb_per_s']:.0f}) | {c['salt_phase_given_ms']:.4f} | {c['salt_phase_prf_ms']:.4f} |")
    lf = lines(o / "bench-leaf.jsonl")
    rows = [x for x in lf if "bench" in x]
    beside = [x["beside_gemm"] for x in lf if "beside_gemm" in x]
    if rows:
        s += ["", "| leaf (32 MiB operand) | digest | leaves | latency alone ms | GB/s | salt ms | GEMM ms | hi-prio root at ms | GEMM slowdown |",
              "|---|---|---|---|---|---|---|---|---|"]
        for x, b in zip(rows, beside + [None] * len(rows)):
            lb = x["lbits"] // 8
            gm = f"{b['gemm_ms']:.4f} | {b['priority']['commit_ms']:.4f} | {b['priority']['gemm_ms'] / (b['gemm_ms'] * b['gemms']) - 1:+.2f}" if b else "- | - | -"
            s.append(f"| {lb} B | {'row-seg' if x['bench'] == 'seg' else 'row/v2'} | {x['n']} | {x['graph_ms']:.4f} | "
                     f"{x['operand_gb_per_s_graph']:.0f} | {x['salt_ms']:.4f} | {gm} |")
    sha = [x["sha"] for x in lines(o / "sha.jsonl") if "sha" in x]
    if sha:
        s += ["", f"raw SHA-512: {sha[0]['gcomp_per_s']:.2f} G compressions/s ({sha[0]['gb_per_s']:.0f} GB/s of blocks) at {sha[0]['mhz']:.0f} MHz"]
    return "\n".join(s) + "\n"


if __name__ == "__main__":
    o = Path(sys.argv[1])
    if "--json" in sys.argv:
        print(json.dumps(fusion(lines(o / "bench-fuse.jsonl")), indent=1))
    else:
        print(render(o), end="")
