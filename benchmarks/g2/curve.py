"""G2: the served Commit's measured overhead over stock, per model, GPU, batch and phase, and the best achievable, rendered from
the evidence store only (CIR-26). G2 has no pass/fail number: it is this curve.

Where the numbers come from:
- Runs labeled campaign=g2 (`research data label RUN campaign g2 --by circuits --ref RUN`). The run's record
  (`outputs.run_record`) holds verity-bench result docs (`*bench/result.json`, schema verity-bench/vllm-overhead/v1, from
  served_commit's `run.sh` or `prof.sh`: TP1 engines under vLLM's serving defaults) or tp_decode summaries
  (`out/<arm>/summary.json`, `tp_decode.sh`: GLM-5.3 TP8 eager, the mode of GLM-5.3's row of record).
- The hash loop's results (campaign=hash-loop, kind bench-result/v1, `benchmarks/hash/loop/collect.py`'s): Qwen3-8B B32
  compiled, one result per iteration and GPU, from the loop branch. Only workload qwen3-8b is read: the loop's glm-5.3 rows
  are copies of campaign=g2 runs, and its other workloads aren't G2's models.

Each measure is a Commit arm's ratio to stock in the same run, with the harness's 95% interval over its interleaved reps,
tp_decode's min and max over its pairs, or the loop's interval where it has one:
  prefill  ttft     time to first token, the mean over the batch's requests
           step     the mean prefill engine step under nsys (hash loop: the arm's prefill_step_ms over stock-prof's)
  decode   itl_p50  per-token latency p50, what a streaming client sees after its first token (itl_p99, its p99)
           step     the mean decode step with every request decoding (tp_decode), or under nsys (hash loop)
  run      tpot     the run's wall over its generated tokens: prefill included, so not a decode number
           tput     stock's output tokens per second over the arm's

An arm's scheme follows its name (`classify`): rows-sha512 (served-commit-rows*, tp_decode's rows*, the loop's sha512*);
vllm-v1 (served-commit, served-commit-prof, served-commit-nativedrain*, ...: SHA-256 position leaves, kept in the table and
never counted, by the one-hash rule); rows-blake3 (the loop's research arm, never counted). `-layer` arms acquire only each
layer's residual stream, K and V (their roots are no scheme's: the per-layer unit is PRF-62's question, not the Commit of
record), and `-4k` arms hash 4 KiB rows; the loop's rows are 4 KiB from iteration 1 on (PRF-64), and every other rows arm
labeled so far is 256 B. A compiled measure counts only under its Build of record's compile artifact (`pin` pinned, from the
run's served-pin check): an unpinned one is mechanism evidence. A measure counts when its comparison's gates passed (the
harness's: every arm's tokens equal stock's; tp_decode's: tokens equal the control's, every rank's root, no collector error;
the loop's validation), its scheme is rows-sha512, and it is pinned or eager. The best per (model, GPU, mode, batch, phase,
metric) is the lowest counting ratio, at full acquisition and, apart, per layer.

    uv run --with matplotlib python3 benchmarks/g2/curve.py [--out DIR] [--no-refresh]

Writes DIR/g2-curve.md (the curve, then every measure), DIR/g2-curve.png and DIR/g2-curve.json (every measure); DIR defaults
to $OUT, else ./out. `research data refresh` first (a run publishes from its node), unless --no-refresh.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

MODELS = ("Llama-3.2-1B", "Qwen3-8B", "Qwen3-30B-A3B", "GLM-5.3")   # G2's models, narrowest first
GPUS = {"h200": "H200", "b200": "B200", "h100": "H100", "rtx-pro-6000": "RTX PRO 6000", "rtx6000": "RTX PRO 6000"}
PHASES = {"prefill": ("ttft", "step"), "decode": ("itl_p50", "itl_p99", "step"), "run": ("tpot", "tput")}
CURVE_METRICS = {"prefill": ("ttft", "step"), "decode": ("itl_p50", "step"), "run": ("tpot",)}
HARNESS = {"time_to_first_token_s": ("prefill", "ttft"), "inter_token_latency_p50_s": ("decode", "itl_p50"),
           "inter_token_latency_p99_s": ("decode", "itl_p99"), "time_per_output_token_s": ("run", "tpot"),
           "tokens_per_s": ("run", "tput")}
LOOP_ARMS = {"sha512": "served-commit-rows", "sha512-layer": "served-commit-rows-layer-4k", "blake3": "served-commit-rows-blake3"}
LOOP_4K_FROM = 1   # the loop's iteration 1 set rows-sha512's rows to 4 KiB (PRF-64)
SCHEME_ROWS, SCHEME_V1, SCHEME_B3 = "rows-sha512", "vllm-v1", "rows-blake3"
CAMPAIGN = "g2"


@dataclass(frozen=True)
class Measure:
    model: str
    gpu: str
    mode: str             # compiled | eager
    batch: int
    phase: str            # prefill | decode | run
    metric: str
    arm: str
    scheme: str
    acquisition: str      # full | per-layer
    row_bytes: int | None
    ratio: float          # arm / stock (tput: stock / arm), so 1.0 is no overhead
    lo: float | None
    hi: float | None
    stock_ms: float | None
    run: str
    source: str           # the art: id the numbers were read from
    pin: str              # pinned | unpinned | pin-failed | eager
    retain: str | None    # the arm's value retention (none: openings by replay)
    gates: bool
    committed_gb: float | None = None
    iteration: int | None = None

    @property
    def sha256(self) -> bool:
        return self.scheme == SCHEME_V1

    def counts(self) -> tuple[bool, str]:
        """Whether this measure is on G2's curve, and if not, why."""
        if not self.gates:
            return False, "gates failed"
        if self.scheme == SCHEME_V1:
            return False, "SHA-256 (vllm-v1): not the one hash"
        if self.scheme != SCHEME_ROWS:
            return False, f"{self.scheme}: not the one hash"
        if self.mode == "compiled" and self.pin != "pinned":
            return False, f"{self.pin}: not the Build of record's kernels"
        return True, ""


def model_name(repo: str) -> str:
    name = repo.rsplit("/", 1)[-1]
    for m in MODELS:
        if name.lower() == m.lower() or name.lower().startswith(m.lower() + "-instruct"):
            return m
    return name


def gpu_name(gpu_id: str | None, sku: str | None = None) -> str:
    if gpu_id:
        return GPUS.get(gpu_id.lower(), gpu_id.upper())
    return (sku or "?").removeprefix("NVIDIA ").strip()


def classify(arm: str, committer: str = "") -> tuple[str, str, int | None]:
    """(scheme, acquisition, row bytes) of a Commit arm, from its name (and tp_decode's committer string)."""
    a = arm.lower()
    if "blake3" in a:
        scheme = SCHEME_B3
    elif "rows" in a or a.startswith("sha512") or "rows-sha512" in committer:
        scheme = SCHEME_ROWS
    else:
        scheme = SCHEME_V1
    acquisition = "per-layer" if "layer" in a else "full"
    row_bytes = (4096 if "4k" in a else 256) if scheme == SCHEME_ROWS else None
    return scheme, acquisition, row_bytes


def retain_of(arm: str, module_spec: str | None) -> str | None:
    """The arm's retention: served_commit's prof_arm arms all retain none; arm.py's served-commit retains on the host."""
    spec = module_spec or ""
    if spec.startswith("served_commit.prof_arm:") or spec.startswith("hash.loop.arms:"):
        return "none"
    if spec.startswith("served_commit.arm:"):
        return "none" if arm.endswith("noretain") else "host"
    return None


def pin_state(text: str | None) -> str:
    """A served-pin check's verdict (`pin_check.txt`): pinned when the Inductor dir's winners equal the artifact's."""
    if text is None:
        return "unpinned"
    return "pinned" if "winners equal to artifact" in text else "pin-failed"


def _interval(r: dict) -> tuple[float | None, float | None]:
    ci = r.get("ci95") or []
    return (float(ci[0]), float(ci[1])) if len(ci) == 2 else (None, None)


def harness_measures(doc: dict, *, run: str, source: str, pin: str) -> list[Measure]:
    """Every Commit arm's measures in one verity-bench result doc (verity-bench/vllm-overhead/v1)."""
    ident = doc["identity"]
    preset = ident["preset"]
    eng = preset.get("engine_settings") or {}
    mode = "eager" if eng.get("enforce_eager") else "compiled"
    gates = bool(doc.get("gates")) and all(g.get("passed") for g in doc["gates"]) and bool(doc.get("points"))
    base = doc.get("baseline", "stock")
    specs = {a["id"]: a.get("module_spec") for a in ident.get("arms", [])}
    common = dict(model=model_name(preset["model"]), gpu=gpu_name(preset.get("gpu_id"), ident.get("gpu")), mode=mode, run=run,
                  source=source, pin=pin if mode == "compiled" else "eager", gates=gates)
    out = []
    for point in doc.get("points", ()):
        batch = int(point["point"]["batch"])
        stock = point["arms"][base]["metrics"]
        for arm, a in point["arms"].items():
            if arm == base or arm == "stock-prof":
                continue
            scheme, acq, rows = classify(arm)
            gb = (a.get("extras") or {}).get("committed_bytes", {}).get("mean")
            for key, (phase, metric) in HARNESS.items():
                m = a["metrics"].get(key)
                if not m or not (m.get("ratio_to_stock") or {}).get("mean"):
                    continue
                r = m["ratio_to_stock"]
                lo, hi = _interval(r)
                ratio = float(r["mean"])
                if metric == "tput":
                    ratio, lo, hi = 1 / ratio, (1 / hi if hi else None), (1 / lo if lo else None)
                sm = stock.get(key, {}).get("mean")
                out.append(Measure(**common, batch=batch, phase=phase, metric=metric, arm=arm, scheme=scheme, acquisition=acq,
                                   row_bytes=rows, ratio=ratio, lo=lo, hi=hi,
                                   stock_ms=None if sm is None or metric == "tput" else 1e3 * float(sm),
                                   retain=retain_of(arm, specs.get(arm)),
                                   committed_gb=None if gb is None else float(gb) / 1e9))
    return out


def tp_measures(summary: dict, *, run: str, source: str, gpu: str, model: str = "GLM-5.3", retain: str | None = "none") -> list[Measure]:
    """One tp_decode arm's measures (`out/<arm>/summary.json`): prefill as TTFT, decode as the mean decode step."""
    arm = summary["arm"]
    scheme, acq, _ = classify(arm, summary.get("committer", ""))
    rows = summary.get("row_bytes")
    out = []
    for b, rec in summary.get("batches", {}).items():
        gates = bool(summary.get("gates_passed")) and bool(rec.get("gates_passed")) and bool(rec.get("control_tokens_equal"))
        for phase, metric, key, stock_key in (("prefill", "ttft", "ttft_mean_ratio", "ttft_mean_s"),
                                              ("decode", "step", "tpot_ratio", "tpot_s")):
            r = rec.get(key)
            if not r:
                continue
            sm = (rec.get("control") or {}).get(stock_key)
            out.append(Measure(model=model, gpu=gpu, mode="eager", batch=int(b), phase=phase, metric=metric, arm=arm, scheme=scheme,
                               acquisition=acq, row_bytes=rows, ratio=float(r["median"]), lo=r.get("min"), hi=r.get("max"),
                               stock_ms=None if sm is None else 1e3 * float(sm), run=run, source=source, pin="eager",
                               retain=retain, gates=gates))
    return out


def loop_measures(meta: dict, labels: dict[str, str], *, source: str) -> list[Measure]:
    """One hash-loop result's measures (Qwen3-8B B32, compiled): run TPOT and throughput from the harness's interleaved ratios,
    and each phase's step from the arm's step time over stock-prof's."""
    if labels.get("arm", "qwen3-8b") != "qwen3-8b":
        return []
    try:
        iteration = int(labels.get("row", ""))
    except ValueError:
        return []           # a what-if row
    fp = meta.get("workload_fingerprint") or {}
    gates = (meta.get("validation") or {}).get("status") == "passed" and labels.get("kernel_outcome") != "voided" \
        and not labels.get("superseded_by")
    by: dict[tuple[str, str], dict] = {(m.get("arm") or "", m["name"]): m for m in meta.get("measurements", ())
                                       if m.get("value") is not None}
    common = dict(model="Qwen3-8B", gpu=labels.get("hardware", "?"), mode="compiled", batch=32, run=str(meta.get("run_id")),
                  source=source, pin=(fp.get("pin") or {}).get("state", "unpinned"), gates=gates, iteration=iteration)
    out = []
    for loop_arm, arm in LOOP_ARMS.items():
        scheme, acq, _ = classify(loop_arm)
        rows = 4096 if scheme == SCHEME_ROWS and (iteration >= LOOP_4K_FROM or acq == "per-layer") else (256 if scheme == SCHEME_ROWS else None)
        retain = (fp.get("retain") or {}).get(arm)
        kw = dict(common, arm=arm, scheme=scheme, acquisition=acq, row_bytes=rows, retain=retain)
        for name, metric, sign in (("tpot_overhead_pct", "tpot", 1), ("throughput_overhead_pct", "tput", -1)):
            m = by.get((loop_arm, name))
            if m is None:
                continue
            ratio = 1 + m["value"] / 100 if sign > 0 else 1 / (1 - m["value"] / 100)
            ci = m.get("ci95") or []
            lo, hi = ((1 + ci[0] / 100, 1 + ci[1] / 100) if sign > 0 else (1 / (1 - ci[0] / 100), 1 / (1 - ci[1] / 100))) \
                if len(ci) == 2 else (None, None)
            stock = by.get(("stock", "tpot_ms"), {}).get("value") if metric == "tpot" else None
            out.append(Measure(**kw, phase="run", metric=metric, ratio=ratio, lo=lo, hi=hi, stock_ms=stock))
        for phase in ("prefill", "decode"):
            x, s = by.get((loop_arm, f"{phase}_step_ms")), by.get(("stock", f"{phase}_step_ms"))
            if x and s and s["value"]:
                out.append(Measure(**kw, phase=phase, metric="step", ratio=x["value"] / s["value"], lo=None, hi=None,
                                   stock_ms=s["value"]))
    return out


def dedupe(measures: list[Measure]) -> list[Measure]:
    """One measure per (run, model, arm, batch, phase, metric): a run read twice (labeled and also a loop row) counts once."""
    seen: dict[tuple, Measure] = {}
    for m in measures:
        seen.setdefault((m.run, m.model, m.arm, m.batch, m.phase, m.metric), m)
    return list(seen.values())


def best(measures: list[Measure]) -> dict[tuple, dict[str, Measure]]:
    """(model, GPU, mode, batch, phase, metric) -> {"full": lowest counting full-acquisition measure, "per-layer": ...,
    "vllm-v1": lowest SHA-256 one (shown, never counted)}."""
    out: dict[tuple, dict[str, Measure]] = defaultdict(dict)
    for m in measures:
        k = (m.model, m.gpu, m.mode, m.batch, m.phase, m.metric)
        ok, _ = m.counts()
        col = m.acquisition if ok else (SCHEME_V1 if m.sha256 and m.gates and m.acquisition == "full" else None)
        if col and (col not in out[k] or m.ratio < out[k][col].ratio):
            out[k][col] = m
    return dict(out)


def _order(k: tuple) -> tuple:
    model, gpu, mode, batch, phase, metric = k
    return (MODELS.index(model) if model in MODELS else len(MODELS), model, gpu, mode, batch, list(PHASES).index(phase),
            PHASES[phase].index(metric) if metric in PHASES[phase] else 9)


def _cell(m: Measure | None, *, run: bool = True) -> str:
    if m is None:
        return "–"
    iv = f" [{m.lo:.2f}, {m.hi:.2f}]" if m.lo is not None and m.hi is not None else ""
    return f"{m.ratio:.2f}x{iv}" + (f" {m.arm.removeprefix('served-commit-')} {m.run}" if run else "")


def curve_table(measures: list[Measure]) -> str:
    """The curve: per model, GPU, mode, batch and phase, the best counting ratio at full acquisition and per layer, and the
    lowest SHA-256 one beside them."""
    b = best(measures)
    lines = ["| model | GPU | mode | B | phase | metric | stock ms | full Commit, rows-sha512 | per-layer, rows-sha512 | vllm-v1 (SHA-256, not counted) |",
             "|---|---|---|---:|---|---|---:|---|---|---|"]
    for k in sorted(b, key=_order):
        model, gpu, mode, batch, phase, metric = k
        if metric not in CURVE_METRICS[phase] or not (b[k].get("full") or b[k].get("per-layer")):
            continue
        cols = b[k]
        stock = next((m.stock_ms for m in cols.values() if m.stock_ms is not None), None)
        lines.append(f"| {model} | {gpu} | {mode} | {batch} | {phase} | {metric} | {'' if stock is None else f'{stock:.2f}'} | "
                     f"{_cell(cols.get('full'))} | {_cell(cols.get('per-layer'))} | {_cell(cols.get(SCHEME_V1))} |")
    return "\n".join(lines)


def measures_table(measures: list[Measure]) -> str:
    """Every measure, one row per run, model, arm and batch, with whether it counts and why not."""
    rows: dict[tuple, dict[tuple[str, str], Measure]] = defaultdict(dict)
    for m in measures:
        rows[(m.model, m.gpu, m.mode, m.run, m.arm, m.batch)][(m.phase, m.metric)] = m
    cols = [(p, x) for p in PHASES for x in PHASES[p]]
    head = " | ".join(f"{p} {x}" for p, x in cols)
    lines = [f"| model | GPU | mode | run | arm | scheme | rows | retain | pin | B | {head} | GB | counts |",
             "|---|---|---|---|---|---|---:|---|---|---:|" + "---:|" * len(cols) + "---:|---|"]
    for k in sorted(rows, key=lambda k: (MODELS.index(k[0]) if k[0] in MODELS else 9, k)):
        ms = rows[k]
        m0 = next(iter(ms.values()))
        ok, why = m0.counts()
        gb = next((m.committed_gb for m in ms.values() if m.committed_gb is not None), None)
        lines.append(f"| {m0.model} | {m0.gpu} | {m0.mode} | {m0.run} | {m0.arm} | {m0.scheme} | {m0.row_bytes or ''} | "
                     f"{m0.retain or '?'} | {m0.pin} | {m0.batch} | "
                     + " | ".join(_cell(ms.get(c), run=False) for c in cols)
                     + f" | {'' if gb is None else f'{gb:.1f}'} | {'yes' if ok else 'no: ' + why} |")
    return "\n".join(lines)


def report(measures: list[Measure]) -> str:
    counted = sorted({m.run for m in measures if m.counts()[0]})
    return "\n\n".join([
        "# G2: the served Commit's overhead over stock (rendered by benchmarks/g2/curve.py from the evidence store)",
        "Ratios are the Commit arm over stock in the same run (1.00x is no overhead); brackets are the 95% interval over "
        "interleaved reps, or the min and max over tp_decode's pairs. prefill ttft is time to first token; decode itl_p50 is "
        "per-token latency p50, and decode step the mean decode step; run tpot is the run's wall per output token, prefill "
        "included. Full Commit commits every value; per-layer commits each layer's residual stream, K and V only (PRF-62's "
        "unit, whose roots are no scheme's). A compiled measure counts only pinned to its Build of record.",
        "## The curve: best counting measure per model, GPU, mode, batch and phase", curve_table(measures),
        f"Runs on the curve: {', '.join(counted) or 'none'}.",
        "## Every measure", measures_table(measures)])


def plot(measures: list[Measure], png: Path) -> None:
    """Prefill and decode (columns) by batch (rows): each model's best counting ratio per GPU, full and per layer."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = best(measures)
    batches = sorted({k[3] for k in b if k[3] in (1, 8, 32)}) or [32]
    fig, axes = plt.subplots(len(batches), 2, figsize=(13, 3.6 * len(batches)), squeeze=False)
    colors = {"H200": "tab:blue", "B200": "tab:red", "RTX PRO 6000": "tab:green", "8x H200": "tab:purple"}
    styles = {"full": ("-", "o"), "per-layer": ("--", "v"), SCHEME_V1: (":", "x")}
    for bi, batch in enumerate(batches):
        for pi, phase in enumerate(("prefill", "decode")):
            ax = axes[bi][pi]
            series: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
            for k, cols in b.items():
                model, gpu, _, kb, kp, metric = k
                if kb != batch or kp != phase or metric not in CURVE_METRICS[phase][:1] + ("step",) or model not in MODELS:
                    continue
                gpu_line = "8x H200" if model == "GLM-5.3" else gpu
                for col, m in cols.items():
                    if metric == "step" and model != "GLM-5.3":
                        continue   # TP1 decode is ITL p50 and prefill TTFT; GLM-5.3's decode is its decode step
                    series[(gpu_line, col, metric)].append((MODELS.index(model), m.ratio))
            for (gpu, col, metric), pts in sorted(series.items()):
                pts.sort()
                ls, mk = styles[col]
                ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, marker=mk, color=colors.get(gpu, "tab:gray"),
                        label=f"{gpu} {col}", alpha=0.5 if col == SCHEME_V1 else 1)
            ax.axhline(1.0, color="black", lw=0.5)
            ax.set_xticks(range(len(MODELS)), MODELS)
            ax.set_xlim(-0.4, len(MODELS) - 0.6)
            ax.set_ylabel("Commit / stock")
            ax.set_title(f"B{batch} {phase} ({'TTFT' if phase == 'prefill' else 'ITL p50; GLM-5.3: decode step'})")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
    fig.suptitle("G2: served Commit over stock, best counting measure (rows-sha512); vllm-v1 SHA-256 dotted, not counted")
    fig.tight_layout()
    fig.savefig(png, dpi=110)


def research(*args: str) -> str:
    return subprocess.run(["research", *args], check=True, capture_output=True, text=True).stdout


def _labels(target: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for lab in json.loads(research("data", "labels", target, "--json")):
        out[lab["key"]] = str(lab["value"])
    return out


def _labeled_runs(key: str, value: str) -> list[str]:
    rows = json.loads(research("data", "sql", f"SELECT DISTINCT target FROM labels WHERE key = '{key}' AND value = '{value}'",
                               "--json") or "[]")
    return sorted(r["target"] for r in rows if str(r.get("target", "")).startswith("r"))


def run_measures(run: str, scratch: Path) -> list[Measure]:
    """A campaign=g2 run's measures, from the result docs and tp_decode summaries in its record."""
    att = json.loads(research("data", "show", run, "--json"))
    rec = (att.get("outputs") or {}).get("run_record")
    if not rec:
        print(f"curve.py: {run} has no run record", file=sys.stderr)
        return []
    to = scratch / run
    research("data", "fetch", rec, "--to", str(to), "--path", "*bench/result.json", "--path", "*pin_check.txt",
             "--path", "out/*/summary.json")
    cond = att.get("conditions") or {}
    out: list[Measure] = []
    for doc in sorted(to.rglob("bench/result.json")):
        pin_txt = doc.parent.parent / "pin_check.txt"
        out += harness_measures(json.loads(doc.read_text()), run=run, source=rec,
                                pin=pin_state(pin_txt.read_text() if pin_txt.is_file() else None))
    for summ in sorted(to.glob("out/*/summary.json")):
        s = json.loads(summ.read_text())
        if "batches" in s and "arm" in s:
            n = int(cond.get("hw.count") or s.get("world") or 1)
            gpu = gpu_name(None, cond.get("hw.sku"))
            out += tp_measures(s, run=run, source=rec, gpu=f"{n}x {gpu}" if n > 1 else gpu)
    return out


def load(refresh: bool = True) -> list[Measure]:
    if refresh:
        rc = subprocess.run(["research", "data", "refresh"], stdout=sys.stderr).returncode
        if rc:
            print(f"curve.py: research data refresh exited {rc}; rendering what the catalog has", file=sys.stderr)
    out: list[Measure] = []
    with tempfile.TemporaryDirectory(prefix="g2-curve-") as tmp:
        for run in _labeled_runs("campaign", CAMPAIGN):
            out += run_measures(run, Path(tmp))
    for art in research("data", "select", "--label", "campaign=hash-loop", "--kind", "bench-result/v1", "--ids").split():
        man = json.loads(research("data", "show", art, "--json"))
        out += loop_measures((man.get("manifest") or man).get("meta") or {}, _labels(art), source=art)
    return dedupe(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("OUT", "out")))
    ap.add_argument("--no-refresh", action="store_true", help="read this store's catalog as it is (no `research data refresh`)")
    a = ap.parse_args()
    measures = load(refresh=not a.no_refresh)
    if not measures:
        print("curve.py: no run labeled campaign=g2 and no hash-loop result in the store", file=sys.stderr)
        return 1
    a.out.mkdir(parents=True, exist_ok=True)
    md = report(measures)
    (a.out / "g2-curve.md").write_text(md + "\n")
    (a.out / "g2-curve.json").write_text(json.dumps([dict(asdict(m), counts=m.counts()[0], why=m.counts()[1])
                                                     for m in measures], indent=1) + "\n")
    plot(measures, a.out / "g2-curve.png")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
