"""The hash loop's plot: iteration (x) against each measure (y), one line per (GPU, arm), read from the evidence store
(`research data select --label campaign=hash-loop --kind bench-result/v1`: each run's result, its measures per arm in its
measurements, its iteration and GPU in its labels). Results whose gates passed are plotted (a failed run stays in the
JSON); several at one (iteration, GPU) are averaged, each also drawn as a dot.

    python3 benchmarks/hash/loop/plot.py [--out DIR] [--measures a,b,...] [--table] [--no-refresh]

Refreshes this store's catalog from the remote first (`research data refresh`: incremental, though a fresh VM's first one
downloads every manifest and attempt). Writes DIR/hash-loop.png and DIR/hash-loop.json (the points plotted, with their result ids); DIR defaults to $OUT, else ./out.
--table prints the iteration table (markdown) as well. Needs matplotlib (`uv run --with matplotlib python3 ...`).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

MEASURES = ("tpot_overhead_pct", "throughput_overhead_pct", "prefill_step_ms", "decode_step_ms", "prefill_hash_gpu_ms",
            "decode_hash_gpu_ms", "prefill_serve_rise_pct", "decode_serve_rise_pct")
ARMS = ("stock", "sha512", "blake3")   # their order in the table; plot.py draws every arm it finds
HEADLINE = ("tpot_overhead_pct", "throughput_overhead_pct")


def research(*args: str) -> str:
    return subprocess.run(["research", *args], check=True, capture_output=True, text=True).stdout


def load(refresh: bool = True) -> list[dict]:
    """Every hash-loop result: {id, row, hardware, label, outcome, source, run, measures: {(arm, name): value}}."""
    if refresh:     # a run's result is published from its node; this store indexes it only once refreshed
        rc = subprocess.run(["research", "data", "refresh"], stdout=sys.stderr).returncode
        if rc:      # an unreadable record anywhere in the bucket fails it; what it did fetch is indexed
            print(f"plot.py: research data refresh exited {rc}; plotting what the catalog has", file=sys.stderr)
    ids = research("data", "select", "--label", "campaign=hash-loop", "--kind", "bench-result/v1", "--ids").split()
    out = []
    for art in ids:
        man = json.loads(research("data", "show", art, "--json"))
        labels: dict[str, str] = {}
        for lab in json.loads(research("data", "labels", art, "--json")):
            labels[lab["key"]] = str(lab["value"])
        meta = (man.get("manifest") or man).get("meta") or {}
        meas = {(m.get("arm", ""), m["name"]): m["value"] for m in meta.get("measurements") or [] if m.get("value") is not None}
        try:
            row = int(labels.get("row", ""))
        except ValueError:
            continue
        out.append({"id": art, "row": row, "hardware": labels.get("hardware", "?"), "label": labels.get("label", ""),
                    "outcome": labels.get("kernel_outcome", ""), "source": labels.get("source", ""), "run": meta.get("run_id"),
                    "valid": (meta.get("validation") or {}).get("status") == "passed", "measures": meas})
    return out


def series(results: list[dict], measure: str) -> dict[tuple[str, str], list[tuple[int, float, list[float]]]]:
    acc: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for r in results:
        if not r["valid"]:
            continue
        for (arm, name), v in r["measures"].items():
            if name == measure:
                acc[(r["hardware"], arm, r["row"])].append(float(v))
    out: dict[tuple[str, str], list[tuple[int, float, list[float]]]] = defaultdict(list)
    for (hw, arm, row), vs in sorted(acc.items()):
        out[(hw, arm)].append((row, statistics.fmean(vs), vs))
    return out


def table(results: list[dict]) -> str:
    lines = ["| N | hypothesis | GPU | arm | TPOT ovh % | tput ovh % | prefill hash ms | decode hash ms | runs | outcome |",
             "|---|---|---|---|---:|---:|---:|---:|---|---|"]
    by: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for r in results:
        if r["valid"]:
            by[(r["row"], r["hardware"])].append(r)
    for (row, hw), rs in sorted(by.items()):
        for arm in [*ARMS[1:], *sorted({a for x in rs for a, _ in x["measures"]} - set(ARMS))]:
            vals = [[x["measures"].get((arm, k)) for x in rs] for k in (*HEADLINE, "prefill_hash_gpu_ms", "decode_hash_gpu_ms")]
            if not any(v is not None for v in vals[0]):
                continue
            cell = lambda vs: " / ".join(f"{v:.1f}" for v in vs if v is not None)  # noqa: E731
            lines.append(f"| {row} | {rs[0]['label'][:60]} | {hw} | {arm} | {cell(vals[0])} | {cell(vals[1])} | {cell(vals[2])} | "
                         f"{cell(vals[3])} | {', '.join(str(x['run']) for x in rs)} | {rs[0]['outcome']} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("OUT", "out")))
    ap.add_argument("--measures", default=",".join(MEASURES))
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--no-refresh", action="store_true", help="read this store's catalog as it is (no `research data refresh`)")
    a = ap.parse_args()
    results = load(refresh=not a.no_refresh)
    if not results:
        print("no result labeled campaign=hash-loop in the store", file=sys.stderr)
        return 1
    measures = a.measures.split(",")
    a.out.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 2
    rows = (len(measures) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 3.6 * rows), squeeze=False)
    colors = {"H200": "tab:blue", "B200": "tab:red", "RTX PRO 6000": "tab:green"}
    styles = {"stock": ":", "sha512": "-", "blake3": "--"}
    markers = {"stock": "s", "sha512": "o", "blake3": "^"}
    dump = {}
    for ax, measure in zip(axes.flat, measures):
        ser = series(results, measure)
        dump[measure] = {f"{hw}/{arm}": [(row, mean) for row, mean, _ in pts] for (hw, arm), pts in ser.items()}
        for (hw, arm), pts in sorted(ser.items()):
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            style = styles.get(arm, "--" if "blake3" in arm else "-.")
            ax.plot(xs, ys, style, marker=markers.get(arm, "D"), color=colors.get(hw), label=f"{hw} {arm}")
            for row, _, vs in pts:
                if len(vs) > 1:
                    ax.scatter([row] * len(vs), vs, s=8, color=colors.get(hw), alpha=0.4)
        ax.set_title(measure)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
        top = max((r["row"] for r in results), default=0)
        ax.set_xticks(range(top + 1))
        ax.set_xlim(-0.5, top + 0.5)
    for ax in list(axes.flat)[len(measures):]:
        ax.axis("off")
    seen: dict[str, object] = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    labels = sorted(seen)
    handles = [seen[lab] for lab in labels]
    hyps = sorted({(r["row"], r["label"]) for r in results})
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=8)
    fig.text(0.01, 0.005, "\n".join(f"{row}: {h}" for row, h in hyps), fontsize=7, va="bottom")
    fig.tight_layout(rect=(0, 0.02 + 0.015 * len(hyps), 1, 0.95))
    png = a.out / "hash-loop.png"
    fig.savefig(png, dpi=110)
    (a.out / "hash-loop.json").write_text(json.dumps({"results": [{k: v for k, v in r.items() if k != "measures"} for r in results],
                                                      "series": dump}, indent=1) + "\n")
    print(png)
    if a.table:
        print(table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
