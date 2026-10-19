"""aprun.sh's V-producer profile: per shape (s tokens, KV heads) and arm (vedit, vsum, vpair, vfold, vfold_x), the hot
bench's time (vbench.jsonl) and ncu's one call (ncu_vp/vp-<s>-<arm>): executed warp instructions and their census classes
an entry (KV head x key x channel), threads' instructions an entry (32 a warp instruction), issue, DRAM and L2
throughput; and the sums' marginal, vfold less vfold_x (what a one-hot cheater skips), as seconds and instructions an
entry, with r2's vsum beside it. Prints Markdown; --json and --toml write the same numbers.

    python3 vprod_profile.py OUT_DIR [--json PATH] [--toml PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eps_profile import SCALE, by_class, first, raw_metrics, sass_counts  # noqa: E402

ARMS = ("vedit", "vsum", "vpair", "vfold", "vfold_x")


def ncu_one(base: Path, entries: int) -> dict | None:
    allc, _, _ = sass_counts(Path(str(base) + ".sass.csv"))
    raw, units = raw_metrics(Path(str(base) + ".raw.csv"))
    if not allc:
        return None
    cls, _ = by_class(allc, "UTMA16K")
    warp = sum(allc.values())
    t, tn = first(raw, "gpu__time_duration.sum")
    return {
        "warp_inst": warp,
        "warp_inst_per_entry": warp / entries,
        "thread_inst_per_entry": 32 * warp / entries,
        "class_per_entry": {k: v / entries for k, v in sorted(cls.items(), key=lambda x: -x[1])},
        "issue_pct": raw.get("smsp__issue_active.avg.pct_of_peak_sustained_active", float("nan")),
        "dram_pct": first(raw, "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                          "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed")[0],
        "l2_pct": first(raw, "lts__throughput.avg.pct_of_peak_sustained_elapsed")[0],
        "mem_pct": first(raw, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")[0],
        "dram_bytes": first(raw, "dram__bytes.sum")[0] * {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}.get(
            units.get("dram__bytes.sum", "byte"), float("nan")),
        "occupancy_pct": raw.get("sm__warps_active.avg.pct_of_peak_sustained_active", float("nan")),
        "ms": t * SCALE.get(units.get(tn, "ns"), float("nan")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--toml", type=Path)
    a = ap.parse_args()
    shapes = {}
    p = a.out / "vbench.jsonl"
    for line in (p.read_text().splitlines() if p.exists() else []):
        if line.startswith("{"):
            b = json.loads(line)
            shapes[(b["s"], b["kvh"])] = {"entries": b["entries"], "digits": b["digits"],
                                          "bench": {k: v["ms"] for k, v in b["arms"].items()}, "ncu": {}}
    for (s, kvh), sh in shapes.items():
        for arm in ARMS:
            r = ncu_one(a.out / "ncu_vp" / f"vp-{s}-{arm}", sh["entries"])
            if r:
                sh["ncu"][arm] = r
        b, n, e = sh["bench"], sh["ncu"], sh["entries"]
        m = {}
        if "vfold" in b and "vfold_x" in b:
            m["s_per_entry"] = (b["vfold"] - b["vfold_x"]) * 1e-3 / e
        if "vsum" in b:
            m["r2_vsum_s_per_entry"] = b["vsum"] * 1e-3 / e
        if "vfold" in n and "vfold_x" in n:
            m["warp_inst_per_entry"] = n["vfold"]["warp_inst_per_entry"] - n["vfold_x"]["warp_inst_per_entry"]
            m["thread_inst_per_entry"] = n["vfold"]["thread_inst_per_entry"] - n["vfold_x"]["thread_inst_per_entry"]
        sh["sums_marginal"] = m
    out = {f"{s}x{kvh}": sh for (s, kvh), sh in sorted(shapes.items())}
    lines = ["| s | arm | bench ms | ncu ms | warp inst / entry | thread inst / entry | issue % | DRAM % | L2 % |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key, sh in out.items():
        for arm in ARMS:
            if arm not in sh["bench"] and arm not in sh["ncu"]:
                continue
            n = sh["ncu"].get(arm, {})
            g = lambda k, d=3: f"{n[k]:.{d}f}" if k in n else "-"  # noqa: E731
            lines.append(f"| {key} | {arm} | {sh['bench'].get(arm, float('nan')):.4f} | {g('ms', 4)} | "
                         f"{g('warp_inst_per_entry', 4)} | {g('thread_inst_per_entry', 2)} | {g('issue_pct', 1)} | "
                         f"{g('dram_pct', 1)} | {g('l2_pct', 1)} |")
        m = sh["sums_marginal"]
        lines.append(f"| {key} | sums (vfold - vfold_x) | {m.get('s_per_entry', float('nan')):.3e} s/entry | | "
                     f"{m.get('warp_inst_per_entry', float('nan')):.4f} | {m.get('thread_inst_per_entry', float('nan')):.2f} "
                     f"| | | |  (r2 vsum {m.get('r2_vsum_s_per_entry', float('nan')):.3e} s/entry)")
    print("\n".join(lines))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))
    if a.toml:
        t = ["[v_producers]   # ms a call (hot medians); sums = vfold - vfold_x, s and instructions an entry (KV head x key x channel)"]
        for key, sh in out.items():
            b, n, m = sh["bench"], sh["ncu"], sh["sums_marginal"]
            t.append(f"s{key} = {{ entries = {sh['entries']}, " + ", ".join(f"{k} = {v:.5f}" for k, v in b.items())
                     + f", t_vsum = {m.get('s_per_entry', float('nan')):.3e}"
                     + f", sums_thread_inst = {m.get('thread_inst_per_entry', float('nan')):.2f}"
                     + "".join(f", {k}_issue_pct = {n[k]['issue_pct']:.1f}" for k in ("vfold", "vsum") if k in n) + " }")
        a.toml.write_text("\n".join(t) + "\n")
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())
