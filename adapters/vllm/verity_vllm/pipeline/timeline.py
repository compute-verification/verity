"""Automatic stage telemetry for the sweep runner.

One append-only JSONL file per row attempt (`<row>/timeline.jsonl`, path in `VERITY_TIMELINE`): every process the runner starts
(row_pod.sh itself, commit_delta, m1_capture, run_config's stages) appends `start` / `end` events with a wall-clock timestamp
(`time.time()` / `date +%s.%N`), its pid and a process label, so elapsed time per stage is MEASURED, never estimated.  Torch-free.

    {"t": 1789875600.123, "utc": "...", "proc": "commit_delta", "pid": 123, "stage": "engine.build", "ev": "end", "s": 41.2,
     "hot": false, "engine_reused": false, "restart_reason": "fresh process", ...}

Fields every runner log must carry: hot/cold process (`hot`), engine reused or restarted (`engine_reused`) + reason
(`restart_reason`), compile/warmup time (`prep.*` stages), workload/Commit time (`work.*`), offline validation time (`validate.*`).

    verity-vllm timeline report <timeline.jsonl> [--json out.json] [--md out.md]
    verity-vllm timeline mark <timeline.jsonl> <proc> <stage> <start|end|mark> [k=v ...]

`report` folds the events into the stage table (process/model startup · preparation · workload · validation · closure) from the
stage-name prefixes below, counts engine starts vs reuses and lists the restart reasons.  Overlapping spans are NOT summed into
elapsed time: every category reports both its summed span time and the union of its spans.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from verity_vllm import settings
from verity_vllm.config import option

SCHEMA = "verity-sweep/timeline/v1"

# stage categories <- stage-name prefixes (first match wins; the row-level `row`/`stage.*` wrappers are the envelope, not a category)
CATEGORIES: list[tuple[str, tuple[str, ...]]] = [
    ("startup", ("engine.build", "engine.load", "process.start", "startup.")),
    ("preparation", ("prep.", "warmup.", "compile.", "graph.", "committer.setup", "ext.", "native.")),
    ("workload", ("work.", "pair", "run.", "capture.run", "control.run")),
    ("validation", ("validate.", "match.", "build.", "fold", "compare", "global_match", "manifest", "verify", "openings", "replay", "oracle", "coverage", "weights")),
    ("closure", ("closure.", "summary", "transfer", "finalize", "release")),
]


def category_of(stage: str) -> str:
    s = stage.lower()
    for cat, prefixes in CATEGORIES:
        for p in prefixes:
            if s.startswith(p) or ("." + p) in s or s == p.rstrip("."):
                return cat
    return "other"


class Timeline:
    """Append-only JSONL writer.  `Timeline(None)` is a no-op sink so callers never branch on telemetry being enabled."""

    def __init__(self, path: str | os.PathLike | None, proc: str = "?", **defaults: Any) -> None:
        self.path = Path(path) if path else None
        self.proc = proc
        self.defaults = dict(defaults)
        self._open: dict[str, float] = {}
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, proc: str, **defaults: Any) -> "Timeline":
        return cls(settings.VERITY_TIMELINE, proc=proc, **defaults)

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def event(self, stage: str, ev: str = "mark", **fields: Any) -> dict[str, Any] | None:
        if self.path is None:
            return None
        now = time.time()
        rec: dict[str, Any] = {"t": round(now, 6), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), "schema": SCHEMA,
                               "proc": self.proc, "pid": os.getpid(), "stage": stage, "ev": ev}
        rec.update(self.defaults)
        rec.update({k: v for k, v in fields.items() if v is not None})
        if ev == "start":
            self._open[stage] = now
        elif ev == "end" and "s" not in rec:
            t0 = self._open.pop(stage, None)
            if t0 is not None:
                rec["s"] = round(now - t0, 6)
        try:
            with self.path.open("a") as f:
                f.write(json.dumps(rec, default=str, separators=(",", ":")) + "\n")
        except OSError:
            return None
        return rec

    def start(self, stage: str, **fields: Any) -> None:
        self.event(stage, "start", **fields)

    def end(self, stage: str, **fields: Any) -> None:
        self.event(stage, "end", **fields)

    @contextmanager
    def span(self, stage: str, **fields: Any) -> Iterator[dict[str, Any]]:
        """`with tl.span("validate.oracle_compare") as extra:` -- `extra` fields set inside the block land on the `end` event."""
        extra: dict[str, Any] = {}
        self.start(stage, **fields)
        ok = True
        try:
            yield extra
        except BaseException:
            ok = False
            raise
        finally:
            self.end(stage, ok=ok, **extra)


def read(path: str | os.PathLike) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    out.sort(key=lambda r: float(r.get("t") or 0))
    return out


def _union(spans: list[tuple[float, float]]) -> float:
    tot, cur = 0.0, None
    for a, b in sorted(spans):
        if cur is None:
            cur = [a, b]
        elif a <= cur[1]:
            cur[1] = max(cur[1], b)
        else:
            tot += cur[1] - cur[0]
            cur = [a, b]
    if cur is not None:
        tot += cur[1] - cur[0]
    return tot


def last_attempt(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """The file is append-only across attempts of the same row (reruns, retries, Commit-only repeats), and a report over
    all of them adds every attempt's build / replay together (a 2 h window with 7 engine starts for a 100 s Commit).  The LAST attempt is
    what the row's table is about: events from the last `row start` mark of row_pod on (the python stages write inside that window), plus
    nothing before it.  Returns (events, attempts_seen); a file without a row envelope is one attempt."""
    starts = [i for i, e in enumerate(events) if e.get("proc") == "row_pod" and e.get("stage") == "row" and e.get("ev") == "start"]
    if len(starts) <= 1:
        return events, max(1, len(starts))
    return events[starts[-1]:], len(starts)


def report(events: list[dict[str, Any]], *, attempt: str = "last") -> dict[str, Any]:
    """Pair start/end events per (proc, pid, stage) into spans; fold into the stage categories; count engine starts/reuses.
    `attempt="last"` (default) reports the last row attempt in an append-only file; `"all"` folds every attempt together."""
    attempts_seen = 1
    if attempt == "last":
        events, attempts_seen = last_attempt(events)
    open_: dict[tuple, dict[str, Any]] = {}
    spans: list[dict[str, Any]] = []
    for e in events:
        key = (e.get("proc"), e.get("pid"), e.get("stage"))
        if e.get("ev") == "start":
            open_[key] = e
        elif e.get("ev") == "end":
            st = open_.pop(key, None)
            t1 = float(e["t"])
            t0 = float(st["t"]) if st else (t1 - float(e.get("s") or 0.0))
            spans.append({"proc": e.get("proc"), "pid": e.get("pid"), "stage": e.get("stage"), "t0": t0, "t1": t1, "s": round(t1 - t0, 3),
                          "category": category_of(str(e.get("stage"))), "fields": {k: v for k, v in e.items() if k not in ("t", "utc", "schema", "proc", "pid", "stage", "ev", "s")}})
    unclosed = [{"proc": k[0], "pid": k[1], "stage": k[2], "t0": float(v["t"])} for k, v in open_.items()]
    ts = [float(e["t"]) for e in events if "t" in e]
    t_first, t_last = (min(ts), max(ts)) if ts else (None, None)
    cats: dict[str, dict[str, Any]] = {}
    # envelope stages (the whole row / whole stage wrapper) are reported separately, not folded into a category
    envelope = [s for s in spans if s["stage"] in ("row",) or s["stage"].startswith("stage.")]
    leaf = [s for s in spans if s not in envelope]
    # a category's time = union of its LEAF spans that have no finer child span of the same category inside them (avoid double counting
    # `commit.commit_delta` (bash, whole process) + `engine.build` (python, inside it)): keep spans not strictly containing another span
    # A span marked `par` (row_pod's concurrent Build derives) is a leaf by construction: it may overlap -- even strictly
    # contain -- a sibling that ran beside it, and must not be read as that sibling's wrapper.
    def contains(a: dict, b: dict) -> bool:
        if a["fields"].get("par"):
            return False
        return a is not b and a["t0"] <= b["t0"] and b["t1"] <= a["t1"] and (a["t1"] - a["t0"]) > (b["t1"] - b["t0"])
    finest = [s for s in leaf if not any(contains(s, o) for o in leaf)]
    for s in finest:
        c = cats.setdefault(s["category"], {"spans": 0, "sum_s": 0.0, "union_s": 0.0, "_iv": [], "stages": {}})
        c["spans"] += 1
        c["sum_s"] += s["s"]
        c["_iv"].append((s["t0"], s["t1"]))
        c["stages"][s["stage"]] = round(c["stages"].get(s["stage"], 0.0) + s["s"], 3)
    for c in cats.values():
        c["union_s"] = round(_union(c.pop("_iv")), 3)
        c["sum_s"] = round(c["sum_s"], 3)
    starts = [s for s in spans if s["stage"] == "engine.build"]
    reuses = [e for e in events if e.get("stage") == "engine.reuse" and e.get("ev") in ("mark", "end")]
    reasons: dict[str, int] = {}
    for s in starts:
        r = str(s["fields"].get("restart_reason") or "unrecorded")
        reasons[r] = reasons.get(r, 0) + 1
    hot = sorted({bool(e.get("hot")) for e in events if "hot" in e}, key=str)
    procs = sorted({(s["proc"], s["pid"]) for s in spans}, key=str)
    return {"schema": SCHEMA + "/report", "events": len(events), "spans": len(spans), "unclosed": unclosed,
            "attempt": attempt, "attempts_in_file": attempts_seen,
            "t_first_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_first)) if t_first else None,
            "t_last_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_last)) if t_last else None,
            "elapsed_s": round(t_last - t_first, 3) if ts else None,
            "envelope": [{"stage": s["stage"], "s": s["s"], "proc": s["proc"]} for s in envelope],
            "categories": {k: cats[k] for k in ("startup", "preparation", "workload", "validation", "closure", "other") if k in cats},
            "engine": {"starts": len(starts), "reuses": len(reuses), "restart_reasons": reasons,
                       "build_s": [s["s"] for s in starts], "hot_values": hot, "processes": len(procs)},
            "spans_detail": sorted(({"proc": s["proc"], "pid": s["pid"], "stage": s["stage"], "s": s["s"], "category": s["category"],
                                     "t0_utc": time.strftime("%H:%M:%S", time.gmtime(s["t0"])), **{k: v for k, v in s["fields"].items() if k in ("rc", "ok", "hot", "engine_reused", "restart_reason", "arm", "pair", "model", "n")}}
                                    for s in spans), key=lambda d: d["t0_utc"])}


def report_md(rep: dict[str, Any], title: str = "") -> str:
    lines = [f"### Runner timeline{(' — ' + title) if title else ''} [measured]", "",
             f"window {rep.get('t_first_utc')} → {rep.get('t_last_utc')} · elapsed {rep.get('elapsed_s')} s · {rep['spans']} spans in {rep['engine']['processes']} processes · "
             + (f"attempt {rep.get('attempt')} of {rep.get('attempts_in_file')} in the file · " if rep.get('attempts_in_file', 1) > 1 else "") +
             f"engine starts **{rep['engine']['starts']}** (build s {[round(x, 1) for x in rep['engine']['build_s']]}) · engine reuses **{rep['engine']['reuses']}** · "
             f"restart reasons {rep['engine']['restart_reasons']} · hot {rep['engine']['hot_values']}", "",
             "| stage | spans | summed s | union s | share of elapsed | finest stages (s) |", "|---|---:|---:|---:|---:|---|"]
    el = rep.get("elapsed_s") or 0
    for k, c in rep["categories"].items():
        st = ", ".join(f"{n} {v:.1f}" for n, v in sorted(c["stages"].items(), key=lambda kv: -kv[1])[:8])
        lines.append(f"| {k} | {c['spans']} | {c['sum_s']:.1f} | {c['union_s']:.1f} | {(c['union_s'] / el * 100) if el else 0:.0f} % | {st} |")
    if rep.get("envelope"):
        lines += ["", "envelope: " + "; ".join(f"{e['stage']} {e['s']:.1f} s ({e['proc']})" for e in rep["envelope"])]
    if rep.get("unclosed"):
        lines += ["", f"unclosed spans (process died / still running): {[(u['proc'], u['stage']) for u in rep['unclosed']]}"]
    return "\n".join(lines) + "\n"


_STAGE_RE = re.compile(r"^(build|match|commit)\s+(PASS|FAIL|SKIP)\s+(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\s+(.*)$")
_RELEASE_RE = re.compile(r"^release\s+(\S+)\s+(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)")


def _utc(s: str) -> float:
    import calendar

    return float(calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")))


def derive_from_logs(row_dir: Path) -> dict[str, Any] | None:
    """Rows run by a runner WITHOUT the timeline: the row's stages.txt is itself timestamped -- a `release <tag> <utc>`
    line at every attempt start, then `<stage> <verdict> <utc> ... wall=<N>s` per stage (row_pod verdict()). The LAST attempt is reported,
    plus commit.log's own measured lines (engine build, weight registration). Labelled `derived`."""
    stages_txt = row_dir / "stages.txt"
    if not stages_txt.exists():
        return None
    attempts: list[dict[str, Any]] = []
    for line in stages_txt.read_text(errors="replace").splitlines():
        r = _RELEASE_RE.match(line)
        if r:
            attempts.append({"release": r.group(1), "t_start": _utc(r.group(2)), "stages": {}})
            continue
        m = _STAGE_RE.match(line)
        if m:
            if not attempts:
                attempts.append({"release": None, "t_start": None, "stages": {}})
            w = re.search(r"\bwall=(\d+)s", m.group(4))
            attempts[-1]["stages"][m.group(1)] = {"verdict": m.group(2), "t_end": _utc(m.group(3)), "s": float(w.group(1)) if w else None}
    att = next((a for a in reversed(attempts) if a["stages"]), None)
    if att is None:
        return None
    t_start = att["t_start"] if att["t_start"] is not None else min(v["t_end"] - (v["s"] or 0) for v in att["stages"].values())
    t_end = max(v["t_end"] for v in att["stages"].values())
    env = {k: {"s": v["s"], "verdict": v["verdict"]} for k, v in att["stages"].items()}
    out: dict[str, Any] = {"label": "derived (stages.txt: release line = attempt start, per-stage verdict utc + wall)", "stages": env,
                           "release": att["release"], "elapsed_s": round(t_end - t_start, 1),
                           "t_first_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start))}
    cl = row_dir / "commit.log"
    if cl.exists():
        txt = cl.read_text(errors="replace")
        builds = re.findall(r"\[delta\] engine built in ([0-9.]+)s", txt)
        regs = re.findall(r"weight registration \(\S+\): \{[^}]*'wall_s': ([0-9.]+)", txt)
        out["commit_log"] = {"engine_builds": [float(x) for x in builds], "register_weights_s": [round(float(x), 1) for x in regs],
                             "engine_starts": len(builds)}
    return out


def collect(dirs: list[Path]) -> dict[str, Any]:
    """One table over many rows: per row the §5 categories, engine starts/reuses and hot/cold from `timeline*.jsonl` when the runner wrote
    one, else the log-derived envelopes; totals over the set (engine starts vs reuses as they accumulate)."""
    rows: list[dict[str, Any]] = []
    for d in dirs:
        if not d.is_dir():
            continue
        is_row = (d / "stages.txt").exists() or (d / "row.log").exists()
        for row_dir in sorted(p for p in ([d] if is_row else d.iterdir()) if p.is_dir()):
            tls = sorted(row_dir.glob("timeline*.jsonl"), key=lambda p: p.stat().st_mtime)
            if tls:
                for tl in tls:
                    rep = report(read(tl))            # last attempt in an append-only file
                    rows.append({"row": row_dir.name, "source": tl.name, "measured": True, "elapsed_s": rep["elapsed_s"],
                                 "attempts_in_file": rep["attempts_in_file"],
                                 "categories": {k: v["union_s"] for k, v in rep["categories"].items()}, "engine": rep["engine"],
                                 "t_first_utc": rep["t_first_utc"]})
            else:
                der = derive_from_logs(row_dir)
                if der:
                    rows.append({"row": row_dir.name, "source": "stages.txt", "measured": False, "elapsed_s": der["elapsed_s"],
                                 "stages": der["stages"], "commit_log": der.get("commit_log"), "t_first_utc": der["t_first_utc"]})
    starts = sum((r["engine"]["starts"] if r["measured"] else (r.get("commit_log") or {}).get("engine_starts", 0)) for r in rows)
    reuses = sum(r["engine"]["reuses"] for r in rows if r["measured"])
    reasons: dict[str, int] = {}
    for r in rows:
        if r["measured"]:
            for k, v in r["engine"]["restart_reasons"].items():
                reasons[k] = reasons.get(k, 0) + v
    return {"schema": SCHEMA + "/collect", "rows": rows, "n_rows": len(rows), "measured_rows": sum(1 for r in rows if r["measured"]),
            "engine_starts": starts, "engine_reuses": reuses, "restart_reasons": reasons,
            "totals_s": {k: round(sum((r.get("categories") or {}).get(k, 0.0) for r in rows if r["measured"]), 1)
                         for k in ("startup", "preparation", "workload", "validation", "closure", "other")}}


def collect_md(col: dict[str, Any], title: str = "") -> str:
    lines = [f"### Runner timelines{(' — ' + title) if title else ''}", "",
             f"{col['n_rows']} rows ({col['measured_rows']} with a runner timeline, {col['n_rows'] - col['measured_rows']} derived from logs) · "
             f"engine starts **{col['engine_starts']}** · engine reuses **{col['engine_reuses']}** · restart reasons {col['restart_reasons']} · "
             f"measured totals s {col['totals_s']}", "",
             "| row | source | start | elapsed s | startup | preparation | workload | validation | closure | engine starts / reuses | hot |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in col["rows"]:
        if r["measured"]:
            c = r["categories"]
            lines.append(f"| {r['row']} | {r['source']} | {r['t_first_utc']} | {r['elapsed_s']} | {c.get('startup', 0):.1f} | {c.get('preparation', 0):.1f} | "
                         f"{c.get('workload', 0):.1f} | {c.get('validation', 0):.1f} | {c.get('closure', 0):.1f} | {r['engine']['starts']} / {r['engine']['reuses']} | {r['engine']['hot_values']} |")
        else:
            st = "; ".join(f"{k} {v['s']} ({v['verdict']})" for k, v in r["stages"].items())
            cl = r.get("commit_log") or {}
            lines.append(f"| {r['row']} | {r['source']} (derived) | {r['t_first_utc']} | {r['elapsed_s']} | engine build {cl.get('engine_builds')} | "
                         f"register_weights {cl.get('register_weights_s')} | — | — | — | {cl.get('engine_starts', '?')} / 0 | stages: {st} |")
    return "\n".join(lines) + "\n"


def _kv(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep:
            continue
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


@dataclass(frozen=True, kw_only=True)
class MarkOptions:
    "append one event (bash callers)"

    path: str = option("path")
    proc: str = option("proc")
    stage: str = option("stage")
    ev: str = option("ev", choices=["start", "end", "mark"])
    kv: list[str] = option("kv", default=[], nargs="*", help="k=v fields (JSON values when parseable)")
    cmd: str = field(default="mark", init=False)


@dataclass(frozen=True, kw_only=True)
class ReportOptions:
    "fold a timeline into the stage table"

    path: str = option("path")
    json: str | None = option("--json", default=None)
    md: str | None = option("--md", default=None)
    title: str = option("--title", default="")
    all: bool = option("--all", default=False, action="store_true",
                       help="fold every attempt in the (append-only) file together; default: the last row attempt")
    cmd: str = field(default="report", init=False)


@dataclass(frozen=True, kw_only=True)
class CollectOptions:
    "one table over a sweep dir's rows (timelines where present, log-derived envelopes otherwise)"

    dirs: list[str] = option("dirs", nargs="+")
    json: str | None = option("--json", default=None)
    md: str | None = option("--md", default=None)
    title: str = option("--title", default="")
    cmd: str = field(default="collect", init=False)


COMMANDS = {"mark": MarkOptions, "report": ReportOptions, "collect": CollectOptions}


def main(a: MarkOptions | ReportOptions | CollectOptions) -> int:
    if a.cmd == "mark":
        Timeline(a.path, proc=a.proc).event(a.stage, a.ev, **_kv(a.kv))
        return 0
    if a.cmd == "collect":
        col = collect([Path(d) for d in a.dirs])
        if a.json:
            Path(a.json).write_text(json.dumps(col, indent=1, default=str) + "\n")
        md = collect_md(col, a.title)
        if a.md:
            Path(a.md).write_text(md)
        sys.stdout.write(md)
        return 0
    rep = report(read(a.path), attempt="all" if a.all else "last")
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    md = report_md(rep, a.title)
    if a.md:
        Path(a.md).write_text(md)
    sys.stdout.write(md)
    return 0
