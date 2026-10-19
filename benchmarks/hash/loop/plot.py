"""The hash loop's plot: iteration (x) against each measure (y), one row of panels per workload (qwen3-8b first, then e.g. glm-5.3
or a PoUW workload), one line per (GPU, workload:arm, served-pin state), pinned lines with filled markers and unpinned hollow,
read from the evidence store
(`research data select --label campaign=hash-loop --kind bench-result/v1`: each run's result, its measures per arm in its
measurements, its iteration (`row`), GPU (`hardware`) and workload (`arm`, e.g. qwen3-8b, or llama8b-int8:commit+mem for a result
that is one arm) in its labels; a result with no `arm` label is qwen3-8b's; a PoUW result, `arm` pearl-c-*, is drawn as workload
pouw, its decode step with the hashing against without, `pouw_headline`). A maxtput row's throughput panel plots its loss
against its card's native reference run (`native_loss_pct`, `panel_measure`); the table keeps the same-cycle overhead beside it.
Results whose gates passed and that are labeled neither kernel_outcome=voided nor superseded_by are plotted (the rest stay in
the JSON); several at one (iteration, GPU, pin state) are averaged, each also drawn as a dot. A what-if row (not an integer,
e.g. what-if-a) is in the table only. Each table line names the served-pin states of the results it reads (pinned, unpinned,
or both as pinned/unpinned).

    python3 benchmarks/hash/loop/plot.py [--out DIR] [--measures a,b,...] [--table] [--no-refresh]   # measures: TPOT, throughput

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
            "decode_hash_gpu_ms", "prefill_serve_rise_pct", "decode_serve_rise_pct", "first_run_rise_pct", "first_run_copy_gb",
            "serve_lost_per_corun_ms", "prefill_gather_ms")
WORKLOAD = "qwen3-8b"
ARMS = ("stock", "sha512", "blake3", "sha512-layer")   # their order in the table; plot.py draws every arm it finds
HEADLINE = ("tpot_overhead_pct", "throughput_overhead_pct")
NATIVE = {"throughput_overhead_pct": "native_loss_pct"}   # what a maxtput row's panel plots instead (panel_measure)


def research(*args: str) -> str:
    return subprocess.run(["research", *args], check=True, capture_output=True, text=True).stdout


def pouw_headline(ms: list[dict]) -> dict[tuple[str, str], float]:
    """A PoUW result's headline (workload pearl-c-*): its decode B32 step with the hashing over the same step without it
    (pearlc vs pearlc-nohash; the whole-step window where the result has several), as pouw:sha512's tpot and throughput overhead."""
    step: dict[str, float] = {}
    for m in ms:
        name, arm, v = m.get("name", ""), str(m.get("arm") or ""), m.get("value")
        if v is None:
            continue
        if name.endswith(".decode_b32_step_s"):
            step.setdefault(name.split(".")[0], float(v))
        elif name == "decode_step_ms" and arm.startswith("whole:"):
            step[arm.split(":", 1)[1]] = float(v) / 1e3
    on, off = step.get("pearlc"), step.get("pearlc-nohash")
    if not (on and off):
        return {}
    return {("pouw:sha512", "tpot_overhead_pct"): round(100 * (on / off - 1), 3),
            ("pouw:sha512", "throughput_overhead_pct"): round(100 * (1 - off / on), 3)}


def load(refresh: bool = True) -> list[dict]:
    """Every hash-loop result: {id, row, hardware, label, outcome, source, run, pin, measures: {(workload:arm, name): value}}."""
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
        workload = labels.get("arm", WORKLOAD)
        if workload.startswith("pearl-c-"):
            meas = pouw_headline(meta.get("measurements") or [])
        else:
            meas = {(f"{workload}:{m['arm']}" if m.get("arm") else workload, m["name"]): m["value"]
                    for m in meta.get("measurements") or [] if m.get("value") is not None}
        row: int | str = labels.get("row", "")
        try:
            row = int(row)
        except ValueError:
            if not row:
                continue
        pin = ((meta.get("workload_fingerprint") or {}).get("pin") or {}).get("state", "unpinned")
        out.append({"id": art, "row": row, "pin": pin, "hardware": labels.get("hardware", "?"), "label": labels.get("label", ""),
                    "outcome": labels.get("kernel_outcome", ""), "source": labels.get("source", ""), "run": meta.get("run_id"),
                    "superseded_by": labels.get("superseded_by", ""),
                    "valid": (meta.get("validation") or {}).get("status") == "passed"
                    and labels.get("kernel_outcome") != "voided" and not labels.get("superseded_by"), "measures": meas})
    return out


def panel_measure(workload: str, measure: str) -> str:
    """The measure a workload's panel plots for ``measure``: a maxtput row's throughput is its loss against its card's native
    reference run (row.py's native_loss_pct), not against the same cycle's stock, which can run off native (a smaller KV pool,
    another attention backend)."""
    return NATIVE.get(measure, measure) if workload.startswith("maxtput") else measure


def series(results: list[dict], measure: str) -> dict[tuple[str, str, str], list[tuple[int, float, list[float]]]]:
    """(GPU, workload:arm, pin state) -> its points: pinned and unpinned results are separate lines, never averaged."""
    acc: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for r in results:
        if not r["valid"] or not isinstance(r["row"], int):
            continue
        for (arm, name), v in r["measures"].items():
            if name == panel_measure(arm.split(":")[0], measure):
                acc[(r["hardware"], arm, r["pin"], r["row"])].append(float(v))
    out: dict[tuple[str, str, str], list[tuple[int, float, list[float]]]] = defaultdict(list)
    for (hw, arm, pin, row), vs in sorted(acc.items()):
        out[(hw, arm, pin)].append((row, statistics.fmean(vs), vs))
    return out


def pins(results: list[dict]) -> str:
    return "/".join(sorted({r["pin"] for r in results}))


def table(results: list[dict]) -> str:
    lines = [("| N | hypothesis | GPU | arm | TPOT ovh % | tput ovh % | lost vs native % | prefill hash ms | decode hash ms | "
              "serve lost / co-run ms | runs | outcome |"), "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    by: dict[tuple[int | str, str], list[dict]] = defaultdict(list)
    for r in results:
        if r["valid"]:
            by[(r["row"], r["hardware"])].append(r)
    cols = (*HEADLINE, "native_loss_pct", "prefill_hash_gpu_ms", "decode_hash_gpu_ms", "serve_lost_per_corun_ms")
    for (row, hw), rs in sorted(by.items(), key=lambda kv: (isinstance(kv[0][0], str), str(kv[0][0]).zfill(4), kv[0][1])):
        first = [f"{WORKLOAD}:{a}" for a in ARMS[1:]]
        rest = {a for x in rs for a, _ in x["measures"]} - set(first)
        for arm in [*first, *sorted(a for a in rest if not a.endswith(":stock") and a != WORKLOAD)]:
            xs = [x for x in rs if any(a == arm for a, _ in x["measures"])]
            vals = [[x["measures"].get((arm, k)) for x in xs] for k in cols]
            if not any(v is not None for v in vals[0]):
                continue
            cell = lambda vs: " / ".join(f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}" for v in vs if v is not None)  # noqa: E731
            lines.append(f"| {row} | {xs[0]['label'][:60]} | {hw} ({pins(xs)}) | {arm} | "
                         + " | ".join(cell(v) for v in vals)
                         + f" | {', '.join(str(x['run']) for x in xs)} | {xs[0]['outcome']} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("OUT", "out")))
    ap.add_argument("--measures", default=",".join(HEADLINE), help=f"comma-separated, of {','.join(MEASURES)}")
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

    sers = {m: series(results, m) for m in measures}
    workloads = sorted({arm.split(":")[0] for s in sers.values() for _, arm, _ in s}, key=lambda w: (w != WORKLOAD, w))
    cols, rows = len(measures), max(len(workloads), 1)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 3.8 * rows), squeeze=False)
    colors = {"H200": "tab:blue", "B200": "tab:red", "RTX PRO 6000": "tab:green", "8x H200": "tab:purple"}
    styles = {"stock": ":", "sha512": "-", "blake3": "--", "sha512-layer": "-."}
    markers = {"stock": "s", "sha512": "o", "blake3": "^", "sha512-layer": "v"}
    top = max((r["row"] for r in results if isinstance(r["row"], int)), default=0)
    dump = {}
    for wi, workload in enumerate(workloads):
        for mi, measure in enumerate(measures):
            ax, ser = axes[wi][mi], sers[measure]
            shown = panel_measure(workload, measure)
            dump.setdefault(shown, {}).update({f"{hw}/{arm}/{pin}": [(row, mean) for row, mean, _ in pts]
                                               for (hw, arm, pin), pts in ser.items() if arm.split(":")[0] == workload})
            for (hw, arm, pin), pts in sorted(ser.items()):
                if arm.split(":")[0] != workload:
                    continue
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                kind = arm.split(":", 1)[1] if ":" in arm else arm
                style = styles.get(kind, "--" if "blake3" in kind else "-.")
                ax.plot(xs, ys, style, marker=markers.get(kind, "D"), color=colors.get(hw), label=f"{hw} {arm} ({pin})",
                        **({} if pin == "pinned" else {"markerfacecolor": "none"}))
                for row, _, vs in pts:
                    if len(vs) > 1:
                        ax.scatter([row] * len(vs), vs, s=8, color=colors.get(hw), alpha=0.4)
            ax.set_title(f"{workload}: {shown}")
            ax.set_xlabel("iteration")
            ax.grid(alpha=0.3)
            ax.set_xticks(range(top + 1))
            ax.set_xlim(-0.5, top + 0.5)
    seen: dict[str, object] = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    labels = sorted(seen)
    handles = [seen[lab] for lab in labels]
    hyps = sorted({(r["row"], r["label"]) for r in results if isinstance(r["row"], int) and r["valid"]})
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
