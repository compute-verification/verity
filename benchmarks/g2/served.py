"""Renders the served arm's runs (`benchmarks/served_commit/served_arm.sh`, which measures them): the Commit's overhead
(rows-sha512, 4 KiB rows) over stock as vLLM serves it (torch.compile and CUDA graphs), beside the batch-invariant eager stock
that GLM-5.3's eager-vs-eager points on G2's curve (`curve.py`) compare against, on one RTX PRO 6000 (sm_120).

One resident run interleaves, rep by rep and on one GPU, batch-invariant eager stock (`stock`, `stock-fp8`: the harness's
eager), compiled stock (`stock@compiled`: vLLM's serving defaults, a served variant) and the Commit, compiled or eager. The
Commit is an arm of COMMIT_MODULES:
  - the served-arm specs' (`benchmarks/specs/*-rtx-served-arm*.toml`, `served_commit/served_arm.py`): `served-commit-rows-4k`,
    compiled (ServedCompiledSource, the arm prof.sh ran for G2's TP1 points), and `commit-rows-4k-eager`, the harness's eager
    `commit` at 4 KiB chunk constants;
  - the harness's (`verity_bench.vllm.commit_arm`, `llama31-8b-fp8-1gpu-commit-served`): `commit`, eager, and `commit@compiled`.
An arm's execution is its own (`identity.arms`), else the spec's. A run holds any of these arms, at most one per role; every
ratio below whose two arms it holds is read from it.

Each ratio is one arm over another in the same run, paired by rep (rep k of every arm ran in the same interleaved round), as the
median with [min, max] over the pairs:
  served        compiled Commit / compiled stock: the overhead as served
  eager-served  eager Commit / compiled stock: the overhead as served for a Commit that runs only eager
  eager         eager Commit / batch-invariant eager stock: the mode of GLM-5.3's points on G2's curve
  compiled-bi   compiled Commit / batch-invariant eager stock
  f             compiled stock / batch-invariant eager stock, which turns an eager ratio into a served one
The metrics are curve.py's: prefill ttft (time to first token, the mean over the batch's requests), decode itl_p50 (per-token
latency p50) and run tpot (the run's wall over its generated tokens, prefill included). A ratio's gates pass when every gate on
either arm passed (the harness's: an arm's tokens equal its control's, its KV pool equals its control's, it closed clean).

A ratio counts by curve.py's rules (`curve.Measure.counts`; `of_record` for f, stock over stock): its gates passed, its Commit
is rows-sha512, the run's GPU is its preset's, and a ratio with a compiled arm ran pinned: the run's `out/pin_check.txt`
(`served_arm.sh` PIN_BUILD: every arm's process compiled into one Inductor dir seeded with the Build of record's autotune
winners) says the dir's winners are the Build's. An uncounted ratio is mechanism evidence, its reason in the tables. The
RTX PRO 6000's Builds of record are `integrations/vllm/verity_vllm/pins/served/*.NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition.json`'s
(Llama-3.2-1B, Qwen3-8B); the served Build has no FP8 path, so no FP8 compiled ratio counts.

    uv run --no-project --with matplotlib python3 benchmarks/g2/served.py RUN... [--tail RUN=FIXED]... [--runs-dir DIR] [--out DIR]

Reads each run's `out/result.json` and `out/pin_check.txt` from DIR/RUN (the run's directory, fetched whole), else from its
record in the evidence store.
Writes OUT/g2-served.md, OUT/g2-served.json (every measure with its pin and why it isn't counted, and each run's facts: timed
points, host load, failed gates, pin) and OUT/g2-served.png; OUT defaults to $OUT, else ./out.

``--tail RUN=FIXED`` reads RUN's compiled Commit run time with FIXED's tail (`retail`): for a run whose Commit arm did more
than commit between its last step and the clock stop (an older `prof_arm.RootsArm` wrote its profiling records there),
FIXED being a run of the same model whose arm doesn't. Its rows are labeled ``RUN (tail FIXED)``.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import curve
from curve import gpu_name, model_name, pin_state, research

RATIOS = {"served": ("commit-compiled", "stock-compiled"), "eager-served": ("commit-eager", "stock-compiled"),
          "eager": ("commit-eager", "stock-eager"), "compiled-bi": ("commit-compiled", "stock-eager"),
          "f": ("stock-compiled", "stock-eager")}
TITLES = {"served": "compiled Commit / compiled stock (served)", "eager-served": "eager Commit / compiled stock",
          "eager": "eager Commit / bi-eager stock (G2's eager mode)", "compiled-bi": "compiled Commit / bi-eager stock",
          "f": "compiled stock / bi-eager stock (f)"}
STOCK = ("stock", "stock-fp8")
COMMIT_MODULES = ("served_commit.served_arm:", "verity_bench.vllm.commit_arm:")
METRICS = {"prefill": ("ttft", "time_to_first_token_s"), "decode": ("itl_p50", "inter_token_latency_p50_s"),
           "run": ("tpot", "time_per_output_token_s")}


@dataclass(frozen=True)
class Measure:
    model: str
    gpu: str
    batch: int
    phase: str        # prefill | decode | run
    metric: str
    ratio: str        # a RATIOS key
    arm: str
    over: str
    median: float
    lo: float
    hi: float
    pairs: int
    arm_ms: float     # the arm's median over its reps
    over_ms: float
    gates: bool
    run: str
    pin: str = "unpinned"   # the run's (curve.pin_state), or eager for a ratio of eager arms
    counts: str = ""        # why the ratio isn't counted by curve.py's rules; empty when it is


def spec_execution(ident: dict) -> str:
    """The execution of an arm that names none: the spec's, as the harness writes it into the preset (curve.py's rule)."""
    return "eager" if (ident["preset"].get("engine_settings") or {}).get("enforce_eager") else "compiled"


def role(arm: dict, execution: str) -> str | None:
    """An arm's role in RATIOS: stock (the builtin or one of its served variants, `stock@compiled`) or the Commit (an arm of
    COMMIT_MODULES), in the arm's own execution, else `execution`."""
    compiled = (arm.get("execution") or execution) == "compiled"
    if arm["id"].split("@")[0] in STOCK:
        return "stock-compiled" if compiled else "stock-eager"
    if (arm.get("module_spec") or "").startswith(COMMIT_MODULES):
        return "commit-compiled" if compiled else "commit-eager"
    return None


def roles(ident: dict) -> dict[str, str]:
    """{role: arm id} over a result's arms (`identity`)."""
    execution, out = spec_execution(ident), {}
    for arm in ident["arms"]:
        if (r := role(arm, execution)) is None:
            continue
        if r in out:
            raise ValueError(f"arms {out[r]} and {arm['id']} are both {r}: a ratio reads one arm per role")
        out[r] = arm["id"]
    return out


def model_label(preset: dict) -> str:
    name = model_name(preset["model"])
    return f"{name} FP8" if str(preset.get("precision", "")).lower() == "fp8" else name


def gates_by_arm(doc: dict) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for g in doc.get("gates", ()):
        out[g["arm"]] = out.get(g["arm"], True) and bool(g.get("passed"))
    return out


def counts(m: Measure, preset_gpu: str | None) -> str:
    """Why curve.py's rules don't count `m` (its `Measure.counts`, or `of_record` for f, whose arms are both stock), else "".
    Every arm of COMMIT_MODULES commits rows-sha512 at 4 KiB rows, which `curve.classify` can't read from the harness's name
    (`commit`)."""
    c = curve.Measure(model=m.model, gpu=m.gpu, mode="eager" if m.pin == "eager" else "compiled", batch=m.batch, phase=m.phase,
                      metric=m.metric, arm=m.arm, scheme=curve.SCHEME_ROWS, acquisition="full", row_bytes=4096, ratio=m.median,
                      lo=m.lo, hi=m.hi, stock_ms=m.over_ms, run=m.run, source=m.run, pin=m.pin, retain=None, gates=m.gates,
                      preset_gpu=preset_gpu)
    return (c.of_record() if m.ratio == "f" else c.counts())[1]


def measures(doc: dict, *, run: str, pin: str = "unpinned") -> list[Measure]:
    """Every ratio RATIOS names whose two arms the doc (a verity-bench result, vllm-overhead/v1) holds, per point and metric;
    `pin` is the run's served-pin verdict (`curve.pin_state`), which a ratio with a compiled arm takes."""
    ident = doc["identity"]
    arms = roles(ident)
    gates = gates_by_arm(doc)
    preset_gpu = gpu_name(ident["preset"]["gpu_id"]) if ident["preset"].get("gpu_id") else None
    gpu = gpu_name(None, ident["gpu"]) if ident.get("gpu") else preset_gpu or "?"
    out = []
    for point in doc["points"]:
        batch = int(point["point"]["batch"])
        for name, (num, den) in RATIOS.items():
            if num not in arms or den not in arms:
                continue
            a, b = point["arms"][arms[num]]["reps"], point["arms"][arms[den]]["reps"]
            for phase, (metric, key) in METRICS.items():
                xs = [ra["metrics"][key] for ra in a]
                ys = [rb["metrics"][key] for rb in b]
                ratios = [x / y for x, y in zip(xs, ys)]
                m = Measure(model=model_label(ident["preset"]), gpu=gpu, batch=batch, phase=phase, metric=metric,
                            ratio=name, arm=arms[num], over=arms[den], median=statistics.median(ratios),
                            lo=min(ratios), hi=max(ratios), pairs=len(ratios), arm_ms=1e3 * statistics.median(xs),
                            over_ms=1e3 * statistics.median(ys),
                            gates=gates.get(arms[num], False) and gates.get(arms[den], False), run=run,
                            pin=pin if "compiled" in num + den else "eager")
                out.append(replace(m, counts=counts(m, preset_gpu)))
    return out


def retail(doc: dict, fixed: dict) -> dict:
    """``doc`` with each compiled Commit rep's run time per output token over its wall less its own device tail (from its last
    step's return to the clock stop, where the arm's ``end_run`` runs) plus ``fixed``'s compiled Commit's median tail at the same
    batch: the serving is the run's own, the tail the fixed arm's. Prefill and decode latency don't read the tail."""
    arm, fixed_arm = (roles(d["identity"])["commit-compiled"] for d in (doc, fixed))
    tails = {int(p["point"]["batch"]): statistics.median(r["host_timing"]["device_tail_s"] for r in p["arms"][fixed_arm]["reps"])
             for p in fixed["points"]}
    out = copy.deepcopy(doc)
    for point in out["points"]:
        tail = tails[int(point["point"]["batch"])]
        for r in point["arms"][arm]["reps"]:
            r["metrics"]["time_per_output_token_s"] = (r["wall_s"] - r["host_timing"]["device_tail_s"] + tail) / r["generated_tokens"]
    return out


def graph_steps(doc: dict) -> dict[int, tuple[float, float, float]]:
    """The compiled Commit's (replayed, eager, all) steps per batch: replayed steps ran its CUDA graphs."""
    arm = roles(doc["identity"]).get("commit-compiled")
    if arm is None:
        return {}
    out = {}
    for point in doc["points"]:
        ex = point["arms"][arm].get("extras") or {}
        out[int(point["point"]["batch"])] = tuple(float((ex.get(k) or {}).get("mean", 0))
                                                  for k in ("replayed_steps", "eager_steps", "steps"))
    return out


def run_facts(doc: dict) -> dict:
    """What a run's table needs besides its ratios: its model, its Commit arms, its timed points, the host's 1-minute load average
    over its timed reps (eager arms launch every kernel from the host, so a loaded host slows them and not the CUDA graphs) and each
    failed gate (a tokens gate with where its tokens first left its control's)."""
    loads = [r["host"]["loadavg_1m"] for p in doc["points"] for a in p["arms"].values() for r in a["reps"]
             if "loadavg_1m" in (r.get("host") or {})]
    failed = []
    for g in doc.get("gates", ()):
        if g.get("passed"):
            continue
        div = g.get("first_divergence") or {}
        why = (f", divergent requests {g['divergent_requests']} (first {div.get('request')} at token {div.get('position')})"
               if g.get("divergent_requests") else "")
        failed.append(f"{g['arm']} {g.get('kind')} {g.get('point') or ''} against {g.get('control') or g.get('against')}{why}")
    commits = [arm for r, arm in roles(doc["identity"]).items() if r.startswith("commit")]
    return {"model": model_label(doc["identity"]["preset"]), "commit": commits, "points": len(doc["points"]),
            "load": (min(loads), max(loads)) if loads else None, "failed": failed}


def cell(m: Measure | None) -> str:
    if m is None:
        return ""
    why = " (gates failed)" if not m.gates else " (not counted)" if m.counts else ""
    return f"{m.median:.3f}× [{m.lo:.3f}, {m.hi:.3f}]{why}"


def stock_ms(row: list[Measure | None]) -> tuple[str, str]:
    """(compiled stock, bi-eager stock) medians in ms, from whichever of a row's ratios reads them."""
    comp = [m.over_ms for m in row if m and m.ratio in ("served", "eager-served")] + [m.arm_ms for m in row if m and m.ratio == "f"]
    bi = [m.over_ms for m in row if m and m.ratio in ("eager", "compiled-bi", "f")]
    return tuple(f"{xs[0]:.3f}" if xs else "" for xs in (comp, bi))


def report(ms: list[Measure], steps: dict[str, dict[int, tuple[float, float, float]]], facts: dict[str, dict] | None = None) -> str:
    by = {(m.model, m.run, m.batch, m.phase, m.ratio): m for m in ms}
    groups = sorted({(m.model, m.run) for m in ms})
    batches = sorted({m.batch for m in ms})
    lines = ["# G2's served arm: the Commit (rows-sha512, 4 KiB rows) over stock as served, RTX PRO 6000", "",
             ("Ratios are arm / reference in the same run, paired by interleaved rep: median [min, max] over the pairs. "
              "Lower is better; 1.000× is no overhead. A ratio counts by curve.py's rules when its gates passed and, for a "
              "ratio with a compiled arm, the run's GPU is its preset's and every arm compiled the Build of record's kernels "
              "(pinned); one that doesn't is mechanism evidence, marked (not counted)."), "",
             "## Served overhead (compiled Commit / compiled stock)", "",
             ("| model | B | prefill ttft | decode itl_p50 | run tpot | compiled stock itl_p50 ms | Commit arm | run | "
              "counted |"), "|---|---:|---|---|---|---:|---|---|---|"]
    for model, run in groups:
        for b in batches:
            row = [by.get((model, run, b, p, "served")) for p in METRICS]
            if not all(row):
                continue
            dec = by[(model, run, b, "decode", "served")]
            lines.append(f"| {model} | {b} | {' | '.join(cell(m) for m in row)} | {dec.over_ms:.3f} | {dec.arm} | {run} | "
                         f"{'; '.join(sorted({m.counts for m in row if m.counts})) or 'yes'} |")
    lines += ["", "## Every ratio", "",
              (f"| model | run | B | phase | metric | {' | '.join(TITLES[r] for r in RATIOS)} | compiled stock ms | "
               "bi-eager stock ms |"), "|---|---|---:|---|---|" + "---|" * len(RATIOS) + "---:|---:|"]
    for model, run in groups:
        for b in batches:
            for phase, (metric, _) in METRICS.items():
                row = [by.get((model, run, b, phase, r)) for r in RATIOS]
                if not any(row):
                    continue
                comp, bi = stock_ms(row)
                lines.append(f"| {model} | {run} | {b} | {phase} | {metric} | {' | '.join(cell(m) for m in row)} | {comp} | {bi} |")
    lines += ["", "## The compiled Commit ran its CUDA graphs", "",
              ("Steps per timed rep (mean over the reps): replayed through the captured graphs, eager (outside every capture "
               "size), all."), "", "| run | B | replayed | eager | steps |", "|---|---:|---:|---:|---:|"]
    for run, per in sorted(steps.items()):
        for b, (rep, eag, n) in sorted(per.items()):
            lines.append(f"| {run} | {b} | {rep:.0f} | {eag:.0f} | {n:.0f} |")
    if facts:
        lines += ["", "## Runs", "",
                  "| run | model | Commit arms | timed points | host load (1 min) over its reps | failed gates | pin |",
                  "|---|---|---|---:|---|---|---|"]
        for run, f in facts.items():
            load = f"{f['load'][0]:.0f}–{f['load'][1]:.0f}" if f["load"] else ""
            lines.append(f"| {run} | {f['model']} | {', '.join(f['commit'])} | {f['points']} | {load} | "
                         f"{'; '.join(f['failed']) or 'none'} | {f.get('pin', '')} |")
    runs = sorted({m.run for m in ms})
    failed = sorted({m.run for m in ms if not m.gates})
    lines += ["", f"Runs: {', '.join(runs)}. Gates failed in: {', '.join(failed) or 'none'}."]
    return "\n".join(lines)


def plot(ms: list[Measure], png: Path) -> None:
    """Prefill, decode and run (columns): each model's served, eager-served and eager ratios by batch, [min, max] as bars, each
    from the first run listed that has it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first: dict[tuple, Measure] = {}
    for m in ms:
        first.setdefault((m.model, m.batch, m.phase, m.ratio), m)
    ms = list(first.values())
    models = sorted({m.model for m in ms})
    batches = sorted({m.batch for m in ms})
    styles = {"served": ("-", "o"), "eager-served": ("--", "s"), "eager": (":", "v")}
    colors = dict(zip(models, ("tab:blue", "tab:red", "tab:green", "tab:purple")))
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5.2 * len(METRICS), 4.4), squeeze=False)
    for ax, (phase, (metric, _)) in zip(axes[0], METRICS.items()):
        for model in models:
            for ratio, (ls, mk) in styles.items():
                pts = sorted((batches.index(m.batch), m) for m in ms
                             if m.model == model and m.phase == phase and m.ratio == ratio)
                if not pts:
                    continue
                xs = [x + 0.06 * (list(styles).index(ratio) - 1) for x, _ in pts]
                ys = [m.median for _, m in pts]
                err = [[m.median - m.lo for _, m in pts], [m.hi - m.median for _, m in pts]]
                ax.errorbar(xs, ys, yerr=err, ls=ls, marker=mk, color=colors[model], capsize=3,
                            label=f"{model}: {TITLES[ratio]}")
                hollow = [(x, m.median) for x, (_, m) in zip(xs, pts) if m.counts]
                if hollow:
                    ax.scatter(*zip(*hollow), marker=mk, facecolors="white", edgecolors=colors[model], zorder=3)
        ax.axhline(1.0, color="black", lw=0.5)
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        ticks = [t for t in (0.8, 1, 1.25, 1.5, 2, 3, 4, 6, 8) if lo <= t <= hi]
        ax.set_yticks(ticks, [f"{t:g}×" for t in ticks])
        ax.minorticks_off()
        ax.set_xticks(range(len(batches)), [f"B{b}" for b in batches])
        ax.set_ylabel("Commit / stock (log)")
        ax.set_title(f"{phase} {metric}")
        ax.grid(alpha=0.3)
    legend: dict[str, object] = {}
    for ax in axes.flat:
        for h, label in zip(*ax.get_legend_handles_labels()):
            legend.setdefault(label, h)
    fig.legend(list(legend.values()), list(legend), loc="lower center", ncol=3, fontsize=7)
    fig.suptitle("G2's served arm on RTX PRO 6000: the Commit (rows-sha512, 4 KiB rows) over compiled stock (served) and over "
                 "bi-eager stock; median, [min, max] over 3 interleaved pairs; hollow: not counted (curve.py's rules)", fontsize=9)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(png, dpi=110)


def load(run: str, runs_dir: Path | None, scratch: Path) -> tuple[dict, str]:
    """A run's result doc and its served-pin verdict (`curve.pin_state` of its `out/pin_check.txt`): from DIR/RUN/out, else
    from its record in the evidence store."""
    out = runs_dir / run / "out" if runs_dir is not None else None
    if out is None or not (out / "result.json").is_file():
        rec = (json.loads(research("data", "show", run, "--json")).get("outputs") or {}).get("run_record")
        if not rec:
            raise SystemExit(f"served.py: {run} has no run record in the store and no out/result.json under {runs_dir}")
        subprocess.run(["research", "data", "fetch", rec, "--to", str(scratch / run), "--path", "out/result.json", "--path",
                        "out/pin_check.txt"], check=True, capture_output=True, text=True)
        out = next((scratch / run).rglob("out/result.json")).parent
    pin = out / "pin_check.txt"
    return json.loads((out / "result.json").read_text()), pin_state(pin.read_text() if pin.is_file() else None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tail", action="append", default=[], metavar="RUN=FIXED",
                    help="RUN's compiled Commit run time with FIXED's tail (retail)")
    ap.add_argument("--runs-dir", type=Path, help="read DIR/RUN/out/result.json where it exists (the run's directory, fetched whole)")
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("OUT", "out")))
    a = ap.parse_args()
    tails = dict(t.split("=", 1) for t in a.tail)
    if unknown := sorted(set(tails) - set(a.runs)):
        ap.error(f"--tail names runs that aren't rendered: {', '.join(unknown)}")
    ms: list[Measure] = []
    steps: dict[str, dict[int, tuple[float, float, float]]] = {}
    facts: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="g2-served-") as tmp:
        for run in a.runs:
            doc, pin = load(run, a.runs_dir, Path(tmp))
            if run in tails:
                fixed, fixed_pin = load(tails[run], a.runs_dir, Path(tmp))
                doc, run = retail(doc, fixed), f"{run} (tail {tails[run]})"
                pin = pin if fixed_pin == "pinned" else fixed_pin
            ms += measures(doc, run=run, pin=pin)
            facts[run] = run_facts(doc) | {"pin": pin}
            if s := graph_steps(doc):
                steps[run] = s
    a.out.mkdir(parents=True, exist_ok=True)
    md = report(ms, steps, facts)
    (a.out / "g2-served.md").write_text(md + "\n")
    (a.out / "g2-served.json").write_text(json.dumps({"runs": facts, "measures": [asdict(m) for m in ms]}, indent=1) + "\n")
    plot(ms, a.out / "g2-served.png")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
