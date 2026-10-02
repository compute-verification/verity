#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8"]
# ///
"""The overnight-sep30 morning dashboard, rendered from the evidence store (docs/overnight-objectives.md, four workstreams).

    cp scripts/overnight-dashboard.py /tmp/ && uv run --no-config --script /tmp/overnight-dashboard.py [--no-refresh] [--repo /workspace]

(run from a local copy with --no-config: the project store's mount answers EAGAIN at times, and uv probes it for uv.toml.)
Reads the `ov.*` labels (one point per labelled run, or per `--ref` when a run carries several rows) and the sweep's config records
(`config_record.json` / `sweep.json` inside the campaign's run trees), renders locally, then copies into the project store
    media/overnight-sep30/{build-throughput,build-peak-mem,build-commit-time,prover-overhead,pouw-overhead}.png, docs/overnight-results.md
each to a temp name renamed into place (retrying transient mount errors), and checks every one afterwards (exit 1 if a check fails).

Markers: filled = quiet, hollow = `ov.noisy true` (other jobs shared the machine), star = a quiet-hour re-measurement (a run wholly
inside 12:30-13:30Z that repeats an attempt number its line already has; it ends the line and is the line's latest value),
x = `ov.gate fail` (drawn, never connected). Points past a re-measured attempt stay as faint markers off the line. A missing
`ov.noisy` is read as quiet (the spec writes it only when true) and is listed under label problems. Units are normalised to the
plotted metric's canonical unit; a point in an unknown unit is left out and listed. The store code comes from the Verity checkout.
"""
from __future__ import annotations

import argparse
import errno
import json
import math
import os
import re
import textwrap
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

CAMPAIGN = "overnight-sep30"
QUIET = (datetime(2026, 9, 30, 12, 30, tzinfo=timezone.utc), datetime(2026, 9, 30, 13, 30, tzinfo=timezone.utc))
LAST_RENDER = datetime(2026, 9, 30, 14, 0, tzinfo=timezone.utc)
PT = ZoneInfo("America/Los_Angeles")
PHASES = ("prefill", "decode")
COMMIT = "commit"  # an `ov.phase` of its own; its rows split into prefill/decode series by `ov.shape`, else `--ref`, else `ov.note`
PHASE_ORDER = (*PHASES, COMMIT)
WS_OWNER = {
    "build": ("Speed up the vLLM Build step", "https://cursor.com/agents/bc-47d0a3ed-166c-5d3b-830d-892cdf106942"),
    "coverage": ("vLLM coordinator", "https://cursor.com/agents/bc-ecac3029-d77d-50d3-b80b-df419ba48ee1"),
    "prover": ("C-Flock circuit prover (M0)", "https://cursor.com/agents/bc-ff572e70-b0e7-5094-85be-13ff9ddc4d6a"),
    "security": ("Prove GEMM step relation in Lean", "https://cursor.com/agents/bc-590cc416-b61a-523d-84f5-280ce207a815"),
    "pouw": ("POUS (PoUW)", None),
}
QUIET_WS = {"pouw"}  # rendered, never listed under label problems
KNOWN = {"ov.ws", "ov.metric", "ov.value", "ov.unit", "ov.config", "ov.phase", "ov.line", "ov.attempt", "ov.gate", "ov.noisy", "ov.note",
         "ov.gpu", "ov.shape"}
RUN_ID = re.compile(r"^r\d{8}-\d{6}-[0-9a-f]{4}$")
# canonical unit first; the others convert into it
UNITS = {
    "throughput": {"gates/s": 1.0, "gate/s": 1.0, "gates/sec": 1.0, "kgates/s": 1e3, "mgates/s": 1e6, "ggates/s": 1e9},
    "peak-rss": {"GiB": 1.0, "gib": 1.0, "gb": 1e9 / 2**30, "mib": 1 / 1024, "mb": 1e6 / 2**30, "tib": 1024.0},
    "overhead": {"× native": 1.0, "x-native": 1.0, "x": 1.0, "×": 1.0, "x native": 1.0, "×native": 1.0, "xnative": 1.0},
    "commit-time": {"s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0, "ms": 1e-3, "min": 60.0, "minutes": 60.0, "h": 3600.0},
}
HIGHER_BETTER = {"throughput"}
# Workstream 1's fixed benchmark set (small dense, mid-size dense, MoE), as the Build owner measures it; build rows on any other
# config are tabled, never plotted, so every plotted attempt stays apples to apples.
BENCHMARK = {"build": ("llama32-1b", "llama32-1b-1k", "mistral-7b", "olmoe-1b-7b")}


@dataclass(frozen=True)
class Plot:
    file: str
    ws: str
    metric: str
    by_config: bool
    title: str
    columns: tuple[str, ...] = PHASES


PLOTS = (
    Plot("build-throughput.png", "build", "throughput", True, "Build throughput: circuit gates emitted per wall-clock second"),
    Plot("build-peak-mem.png", "build", "peak-rss", True, "Build peak memory: peak RSS"),
    Plot("build-commit-time.png", "build", "commit-time", True, "Commit time", columns=(COMMIT,)),
    Plot("prover-overhead.png", "prover", "overhead", False, "Prover overhead: prover time / native GEMM time, same GPU"),
    Plot("pouw-overhead.png", "pouw", "overhead", False, "PoUW overhead (POUS): measured, verifier-accepted PoUW panel rows"),
)
GATE_K = 460
RECORD_SCHEMA = "verity-vllm/config-record/v1"
EPOCH = datetime.min.replace(tzinfo=timezone.utc)


def retried(fn, *args, tries: int = 8, wait: float = 2.0):
    """`fn(*args)`, retrying the transient errors the project store's mount returns (EAGAIN and kin)."""
    for i in range(tries):
        try:
            return fn(*args)
        except OSError as e:
            if e.errno not in (errno.EAGAIN, errno.EBUSY, errno.EIO, errno.ETIMEDOUT) or i == tries - 1:
                raise
            time.sleep(wait)


def publish(local: Path, dest: Path) -> None:
    tmp = dest.with_name(f".{dest.name}.tmp")
    retried(shutil.copyfile, local, tmp)
    retried(os.replace, tmp, dest)


def utc(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def when(t: datetime) -> str:
    return f"{t.astimezone(PT):%-I:%M %p} PT ({t:%H:%MZ})"


def unit_of(metric: str) -> str | None:
    return next(iter(UNITS[metric])) if metric in UNITS else None


@dataclass(eq=False)
class Point:
    target: str
    by: str
    ws: str
    metric: str
    value: float
    unit: str | None
    config: str | None
    phase: str
    line: str
    attempt: int
    gate: str | None
    noisy: bool | None
    note: str | None
    run: str | None = None
    machine: str | None = None
    start: datetime | None = None
    end: datetime | None = None
    remeasure: bool = False
    gpu: str | None = None
    labelled: str = ""
    sub: str | None = None

    @property
    def quiet_window(self) -> bool:
        return bool(self.start and self.end and QUIET[0] <= self.start and self.end <= QUIET[1])


@dataclass
class Problems:
    items: dict[tuple[str, str, str], list[str]] = field(default_factory=lambda: defaultdict(list))

    def add(self, ws: str | None, by: str, what: str, target: str) -> None:
        if ws in QUIET_WS:
            return
        self.items[(ws if ws in WS_OWNER else "unassigned", by, what)].append(target)

    def lines(self) -> list[str]:
        out = []
        for ws in [*WS_OWNER, "unassigned"]:
            mine = sorted((k, v) for k, v in self.items.items() if k[0] == ws)
            if not mine:
                continue
            if ws not in WS_OWNER:
                owner = " (no `ov.ws`, owner unknown)"
            else:
                name, link = WS_OWNER[ws]
                owner = f" ([{name}]({link}))" if link else f" ({name})"
            out += ["", f"**{ws}**{owner}:"]
            for (_, by, what), targets in mine:
                shown = ", ".join(f"`{t}`" for t in targets[:6]) + (f" and {len(targets) - 6} more" if len(targets) > 6 else "")
                out.append(f"- `{by}`: {what} ({len(targets)}): {shown}")
        return out[1:]

    def flat(self) -> list[str]:
        return [f"{ws}/{by}: {what} ({len(t)})" for (ws, by, what), t in sorted(self.items.items())]


def attempt_of(idx, target: str) -> dict | None:
    if target.startswith("art:"):
        runs = idx.attempts_by_output(target)
        return idx.attempt(runs[0]) if runs else None
    if target.startswith("drv:"):
        docs = idx.attempts_by_derivation(target)
        return docs[-1] if docs else None
    return idx.attempt(target)


def result_meta(idx, doc: dict) -> dict:
    art = (doc.get("outputs") or {}).get("result")
    row = idx.artifact(art) if art else None
    return ((row or {}).get("manifest") or {}).get("meta") or {}


def superseded(idx) -> set[str]:
    """Runs and targets a lane has retired with `superseded_by <run>` (labels are append-only); only the replacement is shown."""
    return {t for (t,) in idx.sql("SELECT DISTINCT target FROM labels WHERE key = 'superseded_by'")}


def label_groups(idx, problems: Problems, retired: set[str]) -> list[tuple[str, str, dict[str, str]]]:
    """(target, asserters, {key: latest value}) per labelled row; lanes that disagree on a key are listed, the latest wins."""
    rows = idx.sql("SELECT target, key, value, by, ts, ref FROM labels WHERE key LIKE 'ov.%' ORDER BY ts, path")
    groups: dict[tuple[str, str | None], list[tuple]] = defaultdict(list)
    for r in rows:
        if r[0] not in retired and r[5] not in retired:
            groups[(r[0], r[5])].append(r)
    out = []
    for (target, ref), labs in groups.items():
        latest = {r[1]: r[2] for r in labs}
        latest["_ref"], latest["_ts"] = ref, max(r[4] for r in labs)
        latest["_vts"] = max((r[4] for r in labs if r[1] == "ov.value"), default=latest["_ts"])
        per_by = {(r[1], r[3]): r[2] for r in labs}
        who = ",".join(sorted({r[3] for r in labs}))
        ws = latest.get("ov.ws")
        for key in sorted({k for k, _ in per_by}):
            vals = {v for (k, _), v in per_by.items() if k == key}
            if len(vals) > 1:
                problems.add(ws, who, f"lanes disagree on `{key}` ({' vs '.join(sorted(vals))}); the latest is used", target)
        if "ov.slowdown" in latest:
            problems.add(ws, who, "uses the retired `ov.slowdown`; relabel with `ov.ws`, `ov.metric`, `ov.value`, `ov.unit`, `ov.config`", target)
        for key in sorted(set(latest) - KNOWN - {"ov.slowdown", "_ref", "_ts", "_vts"}):
            problems.add(ws, who, f"unknown key `{key}`", target)
        out.append((target, who, latest))
    return out


def load(idx, problems: Problems) -> tuple[list[Point], list[dict], int, int]:
    """Measurement points (workstreams build and prover; one per run and measurement, the latest assertion wins), coverage rows
    from labels (workstream coverage), the security-labelled count, and how many repeated assertions were merged."""
    points, cov, n_security = [], [], 0
    retired = superseded(idx)
    for target, who, v in label_groups(idx, problems, retired):
        ws = v.get("ov.ws")
        if ws is None:
            if "ov.slowdown" not in v:
                problems.add(None, who, "no `ov.ws`; not rendered", target)
            continue
        if ws not in WS_OWNER:
            problems.add(None, who, f"invalid `ov.ws {ws}`; not rendered", target)
            continue
        run_id = run_of(idx, target, v["_ref"])
        if run_id in retired:
            continue
        doc = attempt_of(idx, target) or (idx.attempt(run_id) if run_id else None)
        ex = (doc or {}).get("execution") or {}
        if run_id is None and doc is None:
            problems.add(ws, who, "no run: the target is not an attempt, and neither `--ref` nor the artifact names one", target)
        if ws == "security":
            n_security += 1
            continue
        if ws == "coverage":
            gate, cfg, note = v.get("ov.gate"), v.get("ov.config"), v.get("ov.note")
            if not cfg or gate not in ("pass", "fail", "unsupported"):
                problems.add(ws, who, "coverage row needs `ov.config` and `ov.gate pass|fail`; not rendered", target)
                continue
            if gate == "fail" and not note:
                problems.add(ws, who, "finding without a named cause (`ov.note`)", target)
            status = {"pass": "pass", "fail": "finding", "unsupported": "unsupported"}[gate]
            cov.append({"row": cfg, "status": status, "why": note or ("" if gate == "pass" else "no cause named"),
                        "run": run_id or target, "_when": ex.get("end_utc") or v["_ts"], "config": {}})
            continue
        missing = [k for k in ("ov.metric", "ov.value", "ov.line", "ov.attempt", "ov.phase") if k not in v]
        if missing:
            problems.add(ws, who, f"missing {', '.join('`' + k + '`' for k in missing)}; not plotted", target)
            continue
        bad = []
        metric = v["ov.metric"]
        if v["ov.phase"] not in PHASE_ORDER:
            bad.append(f"`ov.phase {v['ov.phase']}`")
        try:
            attempt = int(str(v["ov.attempt"]).strip())
        except ValueError:
            attempt = 0
            bad.append(f"`ov.attempt {v['ov.attempt']}`")
        try:
            value = float(str(v["ov.value"]).strip())
            if not math.isfinite(value) or value <= 0:
                raise ValueError
        except ValueError:
            value = 0.0
            bad.append(f"`ov.value {v['ov.value']}` (a positive number)")
        unit = v.get("ov.unit")
        if metric in UNITS:
            if unit is None:
                problems.add(ws, who, f"no `ov.unit`; read as {unit_of(metric)}", target)
                unit = unit_of(metric)
            table = {k.lower(): f for k, f in UNITS[metric].items()}
            factor = table.get(unit.strip().lower())
            if factor is None:
                bad.append(f"`ov.unit {unit}` for `{metric}` (expected {unit_of(metric)})")
            else:
                value, unit = value * factor, unit_of(metric)
        if bad:
            problems.add(ws, who, f"invalid {', '.join(bad)}; not plotted", target)
            continue
        cfg = v.get("ov.config")
        if cfg is None and ws == "build":
            problems.add(ws, who, "no `ov.config` (one of the three benchmark configs); plotted under `(no config)`", target)
        gate = v.get("ov.gate")
        if gate is not None and gate not in ("pass", "fail"):
            problems.add(ws, who, f"invalid `ov.gate {gate}`; plotted as counted", target)
            gate = None
        noisy_raw = v.get("ov.noisy")
        noisy = {"true": True, "false": False}.get(str(noisy_raw).strip().lower()) if noisy_raw is not None else None
        if noisy_raw is None:
            problems.add(ws, who, "no `ov.noisy`; plotted filled (quiet)", target)
        elif noisy is None:
            problems.add(ws, who, f"invalid `ov.noisy {noisy_raw}`; read as quiet", target)
        # A run's window places a measurement only when the whole run is on one side of the quiet hour; a long run (or one not
        # published yet) carries many measurements, and the `ov.value` label's timestamp, capped at the run's end (lanes label
        # after the fact), is the closest record of when each one finished.
        start, end = utc(ex.get("start_utc")), utc(ex.get("end_utc"))
        if not (start and end) or (start < QUIET[1] and end > QUIET[0] and not (QUIET[0] <= start and end <= QUIET[1])):
            finished = utc(v["_vts"])
            start = end = min(finished, end) if end and finished else finished
        p = Point(target, who, ws, metric, value, unit, cfg or ("(no config)" if ws == "build" else None), v["ov.phase"], v["ov.line"],
                  attempt, gate, noisy, v.get("ov.note"), run=run_id,
                  machine=ex.get("machine") or (ex.get("remote") or {}).get("machine") or ex.get("host"), start=start, end=end)
        p.gpu, p.labelled = v.get("ov.gpu"), v["_ts"]
        if metric == "commit-time" and p.phase in PHASES:
            p.phase, p.sub = COMMIT, p.phase
        elif p.phase == COMMIT:
            p.sub, why = split_of(v)
            if p.sub is None:
                problems.add(ws, who, f"commit row {why}; plotted as one series (label `ov.shape prefill` or `ov.shape decode`)", target)
        if p.quiet_window and noisy:
            problems.add(ws, who, "measurement inside the quiet hour is labelled `ov.noisy true`", target)
        points.append(p)
    kept: dict[tuple, Point] = {}
    for p in points:
        k = (p.run or p.target, p.ws, p.metric, p.config, p.phase, p.sub, p.line, p.attempt)
        if k not in kept or p.labelled >= kept[k].labelled:
            kept[k] = p
    merged = len(points) - len(kept)
    points = list(kept.values())
    for ws in ("build", "prover", "pouw"):
        if len({p.gpu for p in points if p.ws == ws and p.gpu}) > 1:
            for p in points:
                if p.ws == ws and p.gpu:
                    p.config = f"{p.config} @ {p.gpu}"
    return points, cov, n_security, merged


def split_of(v: dict) -> tuple[str | None, str]:
    """A commit row's series: `ov.shape` (its phase word, else the value itself), else the one phase word in `--ref`, else in
    `ov.note`.  Returns (series or None, why none)."""
    def word(text: str | None) -> str | None:
        hits = {ph for ph in PHASES if text and re.search(rf"(?<![a-z]){ph}(?![a-z])", text.lower())}
        return hits.pop() if len(hits) == 1 else None
    if v.get("ov.shape"):
        return word(v["ov.shape"]) or v["ov.shape"], ""
    from_ref, from_note = word(v["_ref"]), word(v.get("ov.note"))
    if from_ref and from_note and from_ref != from_note:
        return None, f"says {from_ref} in `--ref` but {from_note} in `ov.note`"
    return (from_ref or from_note), "names neither prefill nor decode in `ov.shape`, `--ref` or `ov.note`"


def run_of(idx, target: str, ref: str | None) -> str | None:
    """The run a labelled row measures: the target itself, its `--ref`, or the `run` an evidence artifact's meta names."""
    if RUN_ID.match(target):
        return target
    if ref and RUN_ID.match(ref):
        return ref
    if target.startswith("art:"):
        meta = (((idx.artifact(target) or {}).get("manifest")) or {}).get("meta") or {}
        if isinstance(meta.get("run"), str) and RUN_ID.match(meta["run"]):
            return meta["run"]
        runs = idx.attempts_by_output(target)
        return runs[0] if runs else None
    return None


@dataclass
class Series:
    ws: str
    metric: str
    config: str | None
    phase: str
    line: str
    points: list[Point]
    path: list[Point] = field(default_factory=list)
    off: list[Point] = field(default_factory=list)
    remeasure: Point | None = None
    sub: str | None = None

    @property
    def phase_label(self) -> str:
        return f"{self.phase} ({self.sub})" if self.sub else self.phase

    @property
    def rank(self) -> tuple:
        return (PHASE_ORDER.index(self.phase) if self.phase in PHASE_ORDER else 9,
                PHASES.index(self.sub) if self.sub in PHASES else 9, self.sub or "")

    @property
    def latest(self) -> Point | None:
        return self.path[-1] if self.path else None

    @property
    def best(self) -> Point | None:
        counted = [p for p in self.points if p.gate != "fail"]
        pick = max if self.metric in HIGHER_BETTER else min
        return pick(counted, key=lambda p: p.value) if counted else None


def build_series(points: list[Point]) -> dict[tuple, Series]:
    grouped: dict[tuple, list[Point]] = defaultdict(list)
    for p in points:
        grouped[(p.ws, p.metric, p.config or "", p.phase, p.line, p.sub or "")].append(p)
    out = {}
    for key, pts in sorted(grouped.items()):
        s = Series(key[0], key[1], key[2] or None, key[3], key[4], points=pts, sub=key[5] or None)
        counted = sorted((p for p in pts if p.gate != "fail"), key=lambda p: (p.attempt, p.end or EPOCH))
        earlier = {p.attempt for p in counted if not p.quiet_window}
        rems = [p for p in counted if p.quiet_window and not p.noisy and p.attempt in earlier]
        if rems:
            s.remeasure = max(rems, key=lambda p: p.end)
            s.remeasure.remeasure = True
        k = s.remeasure.attempt if s.remeasure else None
        per_attempt: dict[int, Point] = {}
        for p in counted:
            if p is not s.remeasure and (k is None or p.attempt < k):
                per_attempt[p.attempt] = p
        s.path = [per_attempt[a] for a in sorted(per_attempt)] + ([s.remeasure] if s.remeasure else [])
        s.off = [p for p in counted if p not in s.path and p.attempt != k]
        out[key] = s
    return out


def sci(v: float) -> str:
    if v <= 0:
        return str(v)
    e = int(math.floor(math.log10(v)))
    if 0 <= e <= 3:
        return f"{v:.4g}"
    return f"{v:.3g}" if -2 <= e < 0 else f"{v / 10 ** e:.2f}×10^{e}"


def sci_tex(v: float) -> str:
    return re.sub(r"×10\^(-?\d+)", r"$\\times10^{\1}$", sci(v))


def tick_tex(v: float) -> str:
    if v <= 0:
        return ""
    e = int(math.floor(math.log10(v) + 1e-9))
    if -2 <= e <= 3:
        return f"{v:g}"
    m = f"{v / 10 ** e:.2g}"
    return rf"$10^{{{e}}}$" if m == "1" else rf"${m}\times10^{{{e}}}$"


def render_plot(spec: Plot, series: dict, path: Path, now: datetime) -> None:
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.lines import Line2D  # noqa: PLC0415
    from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: PLC0415

    def minor_ticks(ax):
        def f(v, _):
            lo, hi = ax.get_ylim()
            return tick_tex(v) if 0 < lo < hi and math.log10(hi / lo) <= 1.3 else ""
        return FuncFormatter(f)

    cols = spec.columns
    mine = [s for s in series.values() if s.ws == spec.ws and s.metric == spec.metric and s.phase in cols]
    # by_config: one row per config, shared by every column; otherwise each phase's configs (GEMM shapes) stack down its column
    if spec.by_config:
        shared = sorted({s.config for s in mine}) or [None]
        column = {ph: shared for ph in cols}
    else:
        column = {ph: sorted({s.config for s in mine if s.phase == ph}) for ph in cols}
    n_rows = max(1, *(len(v) for v in column.values()))
    first_seen = {}
    for s in mine:
        for p in s.points:
            first_seen[s.line] = min(first_seen.get(s.line, p.labelled), p.labelled)
    lines = sorted(first_seen, key=lambda ln: (first_seen[ln], ln))  # oldest line keeps the first colour as lines are added
    palette = plt.get_cmap("tab10" if len(lines) <= 10 else "tab20")
    colour = {ln: palette(i % palette.N) for i, ln in enumerate(lines)}
    fig, axes = plt.subplots(n_rows, len(cols), figsize=(6.5 * len(cols) + (0 if len(cols) > 1 else 2.5), 1.3 + 3.6 * n_rows),
                             squeeze=False, constrained_layout=True)
    for r in range(n_rows):
        for c, phase in enumerate(cols):
            ax = axes[r][c]
            cfgs = column[phase]
            if mine and r >= len(cfgs):
                ax.set_visible(False)
                continue
            cfg = cfgs[r] if cfgs else None
            panel = [s for s in mine if s.phase == phase and (cfg is None or s.config == cfg)]
            ax.set_title(textwrap.fill(f"{cfg}: {phase}", 56) if cfg else phase.capitalize(), fontsize=11)
            ax.set_xlabel("optimization attempt #")
            ax.set_ylabel(f"{spec.metric} ({unit_of(spec.metric)})")
            ax.grid(True, which="both", alpha=0.25)
            ax.set_yscale("log")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if not panel:
                ax.text(0.5, 0.5, "no labelled runs yet", ha="center", va="center", transform=ax.transAxes, color="0.45", fontsize=12)
                ax.set_ylim(1, 10)
                ax.set_xlim(0.5, 3.5)
                continue
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: tick_tex(v)))
            ax.yaxis.set_minor_formatter(minor_ticks(ax))
            ax.margins(y=0.15)
            for s in panel:
                col = colour[s.line]
                if s.path:
                    ax.plot([p.attempt for p in s.path], [p.value for p in s.path], "--" if s.sub == "decode" else "-", color=col,
                            lw=1.8, alpha=0.9, zorder=2)
                for p in s.points:
                    if p.gate == "fail":
                        ax.plot(p.attempt, p.value, "x", color=col, ms=8, mew=2, zorder=3)
                    elif p.remeasure:
                        ax.plot(p.attempt, p.value, "*", color=col, ms=15, mec="black", mew=0.6, zorder=5)
                    else:
                        ax.plot(p.attempt, p.value, "o", ms=7, mew=1.6, mec=col, mfc="white" if p.noisy else col,
                                alpha=0.35 if p in s.off else 1.0, zorder=4)
                if s.latest:
                    ax.annotate(sci_tex(s.latest.value) + (f"  {s.sub}" if s.sub else ""), (s.latest.attempt, s.latest.value),
                                textcoords="offset points", xytext=(7, 4), fontsize=8.5, color=col)
            xs = [p.attempt for s in panel for p in s.points]
            ax.set_xlim(min(xs) - 0.5, max(xs) + 0.9)
            ax.set_xticks(range(min(xs), max(xs) + 1))
    handles = [Line2D([], [], color=colour[ln], lw=2, label=ln) for ln in lines]
    if any(s.sub for s in mine):
        handles += [Line2D([], [], color="0.3", ls="-", lw=1.6, label="prefill"), Line2D([], [], color="0.3", ls="--", lw=1.6, label="decode")]
    handles += [Line2D([], [], ls="", marker="o", color="0.3", mfc="0.3", label="quiet"),
                Line2D([], [], ls="", marker="o", color="0.3", mfc="white", mew=1.6, label="noisy (shared machine)"),
                Line2D([], [], ls="", marker="*", color="0.3", mec="black", ms=13, label="quiet-hour re-measurement"),
                Line2D([], [], ls="", marker="x", color="0.3", mew=2, label="gate fail (not counted)")]
    fig.legend(handles=handles, loc="outside right upper", fontsize=9, frameon=False)
    better = "higher is better" if spec.metric in HIGHER_BETTER else "lower is better"
    fig.suptitle(f"{spec.title}\n({better}; log scale)", fontsize=12)
    n = sum(len(s.points) for s in mine)
    fig.supxlabel(f"campaign {CAMPAIGN}, {n} labelled points; rendered {when(now)}", fontsize=8, color="0.4", x=0.01, ha="left")
    tmp = path.with_name(f".{path.name}.tmp")
    fig.savefig(tmp, dpi=120, format="png")
    plt.close(fig)
    os.replace(tmp, path)


def tree_files(manifest: dict) -> list[dict]:
    pay = manifest.get("payload") or {}
    return (pay.get("files") or []) if pay.get("type") == "tree" else []


def read_tree_json(store, art: str, f: dict) -> dict | None:
    try:
        with tempfile.TemporaryDirectory() as d:
            store.fetch(art, Path(d), paths=[f["path"]])
            return json.loads((Path(d) / f["path"]).read_text())
    except Exception:  # noqa: BLE001  one unreadable record is a note, not a failed render
        return None


def status_of_record(rec: dict) -> tuple[str, str]:
    rp = rec.get("replay") or {}
    if rec.get("outcome") != "PASS":
        return "finding", rec.get("why") or "FAIL"
    k = rp.get("evaluated") or rp.get("k")
    if not isinstance(k, int) or k < GATE_K:
        return "short replay", f"replay evaluated {k} units, the gate needs {GATE_K}"
    return "pass", f"{rp.get('equal')}/{k} units equal"


def load_coverage(store, idx, labelled: list[dict], problems: Problems) -> list[dict]:
    """One row per config: its latest record in the campaign (config_record.json, else sweep.json, else the result's stage
    summary, or an `ov.ws coverage` label)."""
    found: dict[str, dict] = {}

    def put(row: str, when_: str, cand: dict) -> None:
        if row not in found or (when_ or "") >= found[row]["_when"]:
            found[row] = {"config": {}, **cand, "row": row, "_when": when_ or ""}

    retired = superseded(idx)
    # a run's `campaign` label (the lane filing it after the fact) outranks the campaign it was launched under
    relabelled = {t: v for t, v, _ in idx.sql("SELECT target, value, max(ts) FROM labels WHERE key = 'campaign' GROUP BY target")}
    for (mj,) in idx.sql("SELECT manifest_json FROM attempts WHERE campaign=? OR id IN "
                         "(SELECT target FROM labels WHERE key = 'campaign' AND value = ?) ORDER BY start_utc", (CAMPAIGN, CAMPAIGN)):
        doc = json.loads(mj)
        run, t0 = doc.get("id"), (doc.get("execution") or {}).get("end_utc")
        if run in retired or relabelled.get(run, CAMPAIGN) != CAMPAIGN:
            continue
        got = False
        for _, art in sorted((doc.get("outputs") or {}).items()):
            man = ((idx.artifact(art) or {}).get("manifest")) or {}
            meta = man.get("meta") or {}
            if meta.get("schema") == RECORD_SCHEMA:
                st, why = status_of_record(meta)
                put((meta.get("config") or {}).get("row") or "?", t0, {"status": st, "why": why, "run": run, "config": meta.get("config") or {}})
                got = True
            for f in tree_files(man):
                base = f["path"].rsplit("/", 1)[-1]
                if base == "config_record.json":
                    rec = read_tree_json(store, art, f)
                    if rec is None:
                        problems.add("coverage", "sweep", f"unreadable `{f['path']}`", run)
                        continue
                    st, why = status_of_record(rec)
                    put((rec.get("config") or {}).get("row") or f["path"], t0, {"status": st, "why": why, "run": run, "config": rec.get("config") or {}})
                    got = True
                elif base == "sweep.json":
                    sw = read_tree_json(store, art, f) or {}
                    for r in sw.get("refused") or []:
                        put(r.get("row", "?"), t0, {"status": "unsupported", "why": r.get("why") or "refused by the sweep", "run": run})
                        got = True
                    for r in sw.get("done") or []:
                        rowid = r.get("row", "?")
                        if rowid not in found or found[rowid].get("_from") == "sweep":
                            st = "finding" if r.get("outcome") != "PASS" else "short replay"
                            why = r.get("why") or ("PASS, but no config record to read the replay count from" if st == "short replay" else "FAIL")
                            put(rowid, t0, {"status": st, "why": why, "run": run, "_from": "sweep"})
                            got = True
        if not got:
            meta = result_meta(idx, doc)
            fp, val = meta.get("workload_fingerprint") or {}, meta.get("validation") or {}
            if fp.get("row") and val:
                stages = [s.strip() for s in (val.get("detail") or "").split(";") if s.strip()]
                failed = [s for s in stages if s.endswith("FAIL")]
                passed = [s[:-len(" PASS")] for s in stages if s.endswith(" PASS")]
                if val.get("status") == "passed":
                    put(fp["row"], t0, {"status": "short replay", "why": "no config record; replay count unknown", "run": run})
                elif failed:
                    put(fp["row"], t0, {"status": "finding", "why": failed[0], "run": run})
                else:
                    stop = (doc.get("status") or {}).get("failure") or (doc.get("status") or {}).get("state") or "stopped"
                    put(fp["row"], t0, {"status": "not run", "run": run,
                                        "why": f"stopped after {passed[-1] if passed else 'launch'}: {stop}"})
    for r in labelled:
        put(r["row"], r["_when"], r)
    return [{**r, "config": r["config"] or parse_row(rowid)} for rowid, r in sorted(found.items())]


FAMILIES = (("tinyllama", "TinyLlama"), ("qwen3", "Qwen3"), ("qwen2.5", "Qwen2.5"), ("qwen2", "Qwen2"), ("llama", "Llama"),
            ("gemma", "Gemma"), ("phi", "Phi"), ("mistral", "Mistral"), ("mixtral", "Mistral"), ("pythia", "Pythia"),
            ("smollm", "SmolLM"), ("olmoe", "OLMoE"))


def family(model: str | None) -> str:
    m = (model or "").lower()
    return next((name for pre, name in FAMILIES if pre in m), "other")


def is_moe(model: str | None) -> bool:
    m = (model or "").lower()
    return "moe" in m or "mixtral" in m or bool(re.search(r"-a\d+(\.\d+)?b", m))


def parse_row(row: str) -> dict:
    parts = row.split("__")
    if len(parts) < 9:
        return {"row": row}
    d = dict(zip(("model", "dtype", "gpu", "tp", "batch", "input", "output", "arrivals", "sampling", "engine"), parts))
    for k, pre in (("tp", "tp"), ("batch", "b"), ("input", "i"), ("output", "o")):
        if isinstance(d.get(k), str) and d[k].startswith(pre) and d[k][len(pre):].isdigit():
            d[k] = int(d[k][len(pre):])
    return {"row": row, **d}


def copied_tables(src: Path, header_re: str, what: str, level: int) -> list[str]:
    """The tables of `src` whose header row matches `header_re` (else every table), each under its nearest heading at `level`."""
    try:
        raw = retried(src.read_text)
    except FileNotFoundError:
        return [f"Not written yet: `{src}` doesn't exist at this render. The next render picks it up."]
    text = re.sub(r"\A---\n.*?\n---\n", "", raw, flags=re.S)
    tables, cur, heading = [], [], None
    for ln in text.splitlines() + [""]:
        if ln.lstrip().startswith("|"):
            cur.append(ln)
            continue
        if cur:
            tables.append((heading, cur))
            cur = []
        m = re.match(r"^#{2,6}\s+(.*)", ln)
        if m:
            heading = m.group(1).strip()
    keep = [t for t in tables if re.search(header_re, t[1][0], re.I)] or tables
    if not keep:
        return [f"`{src}` exists but holds no {what} table yet; see the file itself."]
    out = [f"The tables of [{src.name}]({src}) as of this render; its prose, notes and gaps are in the file."]
    last = None
    for heading, t in keep:
        if heading and heading != last:
            out += ["", "#" * level + " " + heading]
            last = heading
        out += [""] + t
    return out


def cell(v) -> str:
    return str(v).replace("|", "\\|").replace("\n", " ") if v is not None else ""


def last_note(s: Series) -> str:
    return cell(max(s.points, key=lambda p: (p.attempt, p.end or EPOCH)).note)


def plot_table(spec: Plot, series: dict) -> list[str]:
    mine = [s for s in series.values() if s.ws == spec.ws and s.metric == spec.metric and s.phase in spec.columns]
    if not mine:
        return ["No labelled runs yet."]
    u = unit_of(spec.metric)
    out = ["| Config | Phase | Line | Points | Latest | Best (attempt) | Quiet re-measurement | Latest change |",
           "|---|---|---|---:|---:|---:|---|---|"]
    for s in sorted(mine, key=lambda s: (s.config or "", s.rank, s.line)):
        end, best, rem = s.latest, s.best, s.remeasure
        out.append("| " + " | ".join([
            cell(s.config), cell(s.phase_label), f"`{s.line}`", str(len(s.points)),
            (f"{sci(end.value)} {u}" + (" (noisy)" if end.noisy else "")) if end else "",
            f"{sci(best.value)} (#{best.attempt})" if best else "",
            f"#{rem.attempt}: {sci(rem.value)} (`{rem.run}`)" if rem else "", last_note(s)]) + " |")
    return out


def other_metrics_table(ws: str, series: dict, every_metric: bool = False) -> list[str]:
    plotted = set() if every_metric else {(p.metric, ph) for p in PLOTS if p.ws == ws for ph in p.columns}
    mine = [s for s in series.values() if s.ws == ws and (s.metric, s.phase) not in plotted]
    if not mine:
        return []
    out = ["| Metric | Config | Phase | Line | Points | Latest | Best (attempt) | Latest change |", "|---|---|---|---|---:|---:|---:|---|"]
    for s in sorted(mine, key=lambda s: (s.metric, s.config or "", s.rank, s.line)):
        end, best = s.latest, s.best
        out.append("| " + " | ".join([
            f"`{s.metric}`", cell(s.config), cell(s.phase_label), f"`{s.line}`", str(len(s.points)),
            f"{sci(end.value)} {cell(end.unit or '')}".strip() if end else "", f"{sci(best.value)} (#{best.attempt})" if best else "",
            last_note(s)]) + " |")
    return out


def coverage_tables(rows: list[dict]) -> list[str]:
    if not rows:
        return ["No config records or coverage labels in the campaign yet."]
    order = ("pass", "finding", "short replay", "unsupported", "not run")
    counts = {k: sum(1 for r in rows if r["status"] == k) for k in order}
    out = [", ".join(f"**{counts[k]} {k}**" for k in order) + f" of {len(rows)} configs. `pass` means built, committed and "
           f"{GATE_K} of {GATE_K} randomly drawn units replayed bit-exact; `short replay` passed with fewer than {GATE_K} units, "
           "which isn't the gate; `not run` stopped before any stage failed (killed or cancelled), so it says nothing about the config.", ""]
    gpus = sorted({str(r["config"].get("gpu") or "?") for r in rows})
    fams = sorted({family(r["config"].get("model")) for r in rows})
    out += ["| Family | " + " | ".join(gpus) + " |", "|---|" + "---|" * len(gpus)]
    for fam in fams:
        cells = []
        for g in gpus:
            here = [r for r in rows if family(r["config"].get("model")) == fam and str(r["config"].get("gpu") or "?") == g]
            cells.append(", ".join(f"{sum(1 for r in here if r['status'] == k)} {k}" for k in order
                                   if any(r["status"] == k for r in here)))
        out.append(f"| {fam} | " + " | ".join(cells) + " |")
    out += ["", "| Config | Family | Dense/MoE | dtype | Sampler | TP | Batch | In/out | Engine | GPU | Status | Cause | Run |",
            "|---|---|---|---|---|---:|---:|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (order.index(r["status"]) if r["status"] in order else 9, r["row"])):
        c = r["config"]
        out.append("| " + " | ".join(cell(x) for x in [
            f"`{r['row']}`", family(c.get("model")) if c.get("model") else "", ("MoE" if is_moe(c.get("model")) else "dense") if c.get("model") else "",
            c.get("dtype"), c.get("sampling"), c.get("tp"), c.get("batch"),
            f"{c.get('input')}/{c.get('output')}" if c.get("input") is not None else "", c.get("engine"), c.get("gpu"),
            r["status"], r["why"], f"`{r['run']}`"]) + " |")
    return out


def short(v: float) -> str:
    return re.sub(r"e([+-])0*(\d)", lambda m: "e" + ("-" if m.group(1) == "-" else "") + m.group(2), f"{v:.3g}")


def ends_quiet(s: Series) -> bool:
    """The line's latest point is a quiet-hour measurement: a re-measurement (starred) or a new attempt run in the quiet hour."""
    return bool(s.remeasure or (s.latest and s.latest.quiet_window and not s.latest.noisy))


def checkpoint_text(series: dict, n_points: int, now: datetime, doc_path: Path) -> str:
    """One line: the point count, and per line its attempts, how many plotted series end on a quiet re-measurement, each series'
    latest value (`q` = quiet re-measurement) and the series still without one."""
    plotted = {(p.ws, p.metric, ph) for p in PLOTS for ph in p.columns}
    by_line: dict[tuple[str, str], list[Series]] = defaultdict(list)
    for s in series.values():
        if (s.ws, s.metric, s.phase) in plotted and s.latest:
            by_line[(s.ws, s.line)].append(s)
    parts, missing = [], []
    for (ws, line), ss in sorted(by_line.items()):
        attempts = sorted({p.attempt for s in ss for p in s.points})
        quiet = [s for s in ss if ends_quiet(s)]
        vals = ", ".join(f"{s.metric}/{s.config}/{s.phase_label} {short(s.latest.value)}#{s.latest.attempt}"
                         + ("q" if s.remeasure else "q(new)" if ends_quiet(s) else "")
                         for s in sorted(ss, key=lambda s: (s.metric, s.config or "", s.rank)))
        parts.append(f"{line} ({ws}; attempts {attempts[0]}–{attempts[-1]}; quiet {len(quiet)}/{len(ss)}): {vals}")
        if len(quiet) < len(ss):
            missing.append(f"{line} ({'all' if not quiet else ', '.join(f'{s.metric}/{s.config}/{s.phase_label}' for s in ss if not ends_quiet(s))})")
    head = f"quiet-hour render at {now:%H:%M}Z: {n_points} points"
    tail = (f"no quiet re-measurement landed by {now:%H:%M}Z: " + "; ".join(missing)) if missing else "every plotted series ends on a quiet re-measurement"
    return f"{head}; " + " | ".join(parts) + f" | {tail}; page {doc_path}"


def write_checkpoint(lane_dir: Path, text: str, now: datetime) -> Path:
    """`CHECKPOINT none (HH:MMZ) [open] <text>` under the front matter of the lane's newest report (created if none), newest first."""
    lane = lane_dir.name
    retried(lane_dir.mkdir, 0o777, True, True)
    reports = sorted(retried(lambda: list(lane_dir.glob(f"*-report-{lane}.md"))))
    path = reports[-1] if reports else lane_dir / f"{now:%Y%m%dT%H%MZ}-report-{lane}.md"
    if not reports:
        retried(path.write_text, "---\ncursor:\n  subagentId: \"bc-14cfd836-f3c3-5765-b161-df07ef9ea438\"\n"
                f"lane: {lane}\nkind: report\ncampaign: {CAMPAIGN}\ncreated: {now:%Y-%m-%dT%H:%MZ}\nstatus: open\n---\n")
    doc = retried(path.read_text)
    m = re.match(r"\A---\n.*?\n---\n", doc, flags=re.S)
    front, body = (doc[:m.end()], doc[m.end():]) if m else ("", doc)
    line = f"CHECKPOINT none ({now:%H:%M}Z) [open] {' '.join(text.split())}\n"
    retried(path.write_text, front + "\n" + line + body.lstrip("\n"))
    if line not in retried(path.read_text):
        raise OSError(f"{path}: the checkpoint line is not there after writing")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=os.environ.get("VERITY_REPO", "/workspace"))
    ap.add_argument("--project-store", default=os.environ.get("OV_PROJECT_STORE", "/cursor/stores/bc-36415049-30db-4fff-a34b-81f0afc0124d"))
    ap.add_argument("--no-refresh", action="store_true", help="render from the local catalog as it is")
    ap.add_argument("--checkpoint", metavar="LANE_DIR", help="also write a CHECKPOINT line (points, each line's latest values, "
                    "which series lack a quiet re-measurement) to the lane report in LANE_DIR")
    ns = ap.parse_args(argv)
    sys.path.insert(0, str(Path(ns.repo) / "tools" / "research" / "src"))
    from research.store.local import LocalStore  # noqa: PLC0415

    project = Path(ns.project_store)
    media = project / "media" / CAMPAIGN
    retried(media.mkdir, 0o777, True, True)
    doc_path = project / "docs" / "overnight-results.md"
    local = Path(tempfile.mkdtemp(prefix="ov-dashboard-"))
    now = datetime.now(timezone.utc)
    store = LocalStore()
    refresh_note = "catalog not refreshed (--no-refresh)"
    if not ns.no_refresh:
        try:
            rep = store.refresh(jobs=32)
            refresh_note = (f"catalog refreshed from the remote (+{rep['attempts']} attempts, +{rep['labels']} labels)"
                            + (f", {len(rep['errors'])} fetch error(s)" if rep["errors"] else ""))
        except Exception as e:  # noqa: BLE001  a stale catalog still renders, and the page says so
            refresh_note = f"**refresh failed, catalog may be stale**: {type(e).__name__}: {e}"
    idx = store.index
    problems = Problems()
    points, cov_labels, n_security, merged = load(idx, problems)
    series = build_series(points)
    off_bench = {k: s for k, s in series.items() if s.ws in BENCHMARK and (s.config or "").split(" @ ")[0] not in BENCHMARK[s.ws]}
    for k, s in off_bench.items():
        del series[k]
        problems.add(s.ws, ",".join(sorted({p.by for p in s.points})),
                     f"row outside the benchmark configs ({', '.join(BENCHMARK[s.ws])}); tabled, not plotted", f"{s.line}: {s.config}")
    coverage = load_coverage(store, idx, cov_labels, problems)
    after_quiet = now >= QUIET[1]
    if after_quiet:
        for s in series.values():
            if not ends_quiet(s) and s.path and s.metric in {p.metric for p in PLOTS}:
                problems.add(s.ws, ",".join(sorted({p.by for p in s.points})), "no quiet-hour re-measurement",
                             f"{s.line}: {s.metric}/{s.config}/{s.phase}")

    pngs = {}
    for spec in PLOTS:
        pngs[spec.file] = media / spec.file
        render_plot(spec, series, local / spec.file, now)

    def img(name: str, alt: str) -> str:
        return f"![{alt}]({pngs[name]})"

    lean = project / "internal" / "lanes" / "lean-gemm-relation" / "theorems.md"
    by_file = {p.file: p for p in PLOTS}
    other_build = other_metrics_table("build", series)
    other_prover = other_metrics_table("prover", series)
    n_runs = len({p.run or p.target for p in points})
    doc = [
        "---", "cursor:", '  subagentId: "bc-14cfd836-f3c3-5765-b161-df07ef9ea438"', "---", "",
        "# Overnight results, Sep 30", "",
        f"Rendered {when(now)} from campaign `{CAMPAIGN}` in the evidence store: {len(points)} measurement points from {n_runs} runs "
        f"({merged} repeated assertions of the same measurement merged), {len(coverage)} configs with a coverage record, "
        f"{n_security} security-labelled runs; {refresh_note}.",
        ("This is the render after the quiet hour (5:30–6:30 AM PT, 12:30–13:30Z): each line with a quiet re-measurement ends on it."
         if after_quiet else
         f"Re-rendered every hour until {when(LAST_RENDER)}. Once the quiet hour (5:30–6:30 AM PT, 12:30–13:30Z) is over, each line "
         "ends on its quiet re-measurement."),
        "", "In the plots, filled markers are quiet measurements and hollow ones are `ov.noisy true` (other jobs shared the machine). "
        "A star is a quiet-hour re-measurement, which ends its line. A cross is a gate failure: drawn, but not counted. Faint markers "
        "are off the line: an earlier measurement of an attempt number that was measured again (the line uses the latest), or an "
        "attempt after the one a line re-measured. Every y-axis is log scale.", "",
        "## 1. Circuit building faster", "",
        f"Build and Commit on CPU, `vy-nebius-1`, 32 vCPU; one row of panels per benchmark config ({', '.join(BENCHMARK['build'])}).", "",
        img("build-throughput.png", "Build throughput"), "", *plot_table(by_file["build-throughput.png"], series), "",
        img("build-peak-mem.png", "Build peak memory"), "", *plot_table(by_file["build-peak-mem.png"], series), "",
        "Commit time (`ov.phase commit`, `ov.metric commit-time`): one panel per benchmark config, prefill and decode as separate "
        "series (solid and dashed), split by `ov.shape`, else `--ref`, else `ov.note`.", "",
        img("build-commit-time.png", "Commit time"), "", *plot_table(by_file["build-commit-time.png"], series), "",
        *(["Other build metrics (bytes per gate, commit time, …):", "", *other_build, ""] if other_build else []),
        *(["Build rows on configs outside the benchmark set (other lanes' experiments; not plotted):", "",
           *other_metrics_table("build", off_bench, every_metric=True), ""] if off_bench else []),
        "## 2. Coverage", "", *coverage_tables(coverage), "",
        "### Assumptions", "", *copied_tables(project / "docs" / "semantic-assumptions.md", r"assumption|rating", "assumptions", 4), "",
        "## 3. Prover faster", "", img("prover-overhead.png", "Prover overhead"), "", *plot_table(by_file["prover-overhead.png"], series), "",
        *(["Other prover metrics:", "", *other_prover, ""] if other_prover else []),
        "## 4. Prover is secure", "", *copied_tables(lean, r"theorem|status|sorry", "theorem", 3), "",
        "## PoUW overhead (POUS)", "",
        "POUS's measured, verifier-accepted PoUW panel rows (`ov.ws pouw`), kept apart from the Verity prover above.", "",
        img("pouw-overhead.png", "PoUW overhead"), "", *plot_table(by_file["pouw-overhead.png"], series), "",
        "## Label problems", "",
        "Grouped by workstream, with its owner; each line names the asserting lane (`--by`) and the labelled runs.", "",
        *(problems.lines() or ["None found."]), "",
        "## How this is rendered", "",
        "~~~text", f"cp {project / 'scripts' / 'overnight-dashboard.py'} /tmp/ && uv run --no-config --script /tmp/overnight-dashboard.py",
        "~~~", "",
    ]
    (local / doc_path.name).write_text("\n".join(doc))

    outputs = [(local / name, dest) for name, dest in pngs.items()] + [(local / doc_path.name, doc_path)]
    failed = []
    for src, dest in outputs:
        try:
            publish(src, dest)
        except OSError as e:
            failed.append(f"{dest}: not written ({e})")
    for src, dest in outputs:
        try:
            got = retried(dest.read_bytes)
        except OSError as e:
            failed.append(f"{dest}: unreadable after writing ({e})")
            continue
        if not got or got != src.read_bytes():
            failed.append(f"{dest}: empty or differs from what was rendered")
        elif dest.suffix == ".png" and got[:8] != b"\x89PNG\r\n\x1a\n":
            failed.append(f"{dest}: not a PNG")
    text = (local / doc_path.name).read_text()
    failed += [f"{doc_path}: does not embed {p}" for p in pngs.values() if str(p) not in text]
    written = [dest for _, dest in outputs]
    shutil.rmtree(local, ignore_errors=True)
    if ns.checkpoint:
        try:
            written.append(write_checkpoint(Path(ns.checkpoint), checkpoint_text(series, len(points), now, doc_path), now))
        except OSError as e:
            failed.append(f"checkpoint in {ns.checkpoint}: not written ({e})")
    print(json.dumps({"rendered": now.isoformat(timespec="seconds"), "points": len(points), "runs": n_runs, "configs": len(coverage),
                      "security_runs": n_security, "series": sorted("/".join(str(x) for x in k) for k in series), "refresh": refresh_note,
                      "problems": problems.flat(), "files": [str(p) for p in written], "verify": failed or "ok"}, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
