#!/usr/bin/env python3
"""Verity's panels for the site's live console (`/admin/live`), recomputed from their live sources and sent in the site's panel
format (Project store `internal/live-console/panel-format.md`; the inventory is `internal/live-console/verity-panels.md`).

    verity_console.py --group store [--store DIR] [--git DIR] [--research-src DIR] [--out DIR] [--key FILE] [--no-refresh]
    verity_console.py --group node1 [--out DIR] [--key FILE]

- `store`, on the control pod: the evidence store's `ov.*` labels and the campaign's coverage records, read with the overnight
  dashboard's own loaders (`ovd.py` is a copy of the Project store's `scripts/overnight-dashboard.py`), and the merge trains on
  `main` (a commits-only clone, `--git`) with their `check` Attempts.
- `node1`, on vy-nebius-1: its Prometheus (DCGM and node-exporter, with the steward's definitions from `util_collect.py`, a copy of
  `internal/lanes/nebius-infra/tools/util_collect.py`) and the Kueue samples `vy-usage` appends to /workspace/usage/queues.jsonl.

Every panel is checked against the format (fields, lengths, cell types, 64 KiB) and written to OUT/<name>.json. When the key file
exists (default ~/.config/verity/panels.key) each valid panel is also PUT to the site; without it the run is a dry run. A panel
whose source fails this run is neither written nor sent, so the site keeps the last good one. Prints one JSON line per run; exit 1
when a panel failed, was invalid, or wasn't accepted.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = "https://website-docs-sage.vercel.app"
FIELDS = ("id", "title", "kind", "unit", "y_scale", "target", "target_label", "stacked", "note", "updated_at", "source", "columns", "rows")
REQUIRED = {"id", "title", "kind", "updated_at", "source", "columns", "rows"}
MAX_BODY = 64 * 1024
ID = re.compile(r"[a-z0-9-]+/[a-z0-9-]+")
ISO = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d(:\d\d(\.\d+)?)?(Z|[+-]\d\d:\d\d)")
STATUS_ORDER = ("pass", "finding", "short replay", "unsupported", "not run")
GPUS, CORES = 8, 192
GPU_TARGET, CPU_TARGET = 60, 70  # Daniel's node-1 targets (docs/gpu-utilization-postmortem.md)
PROM = "/api/v1/namespaces/skypilot/services/skypilot-prometheus-server:80/proxy/api/v1"
QUEUES = Path("/workspace/usage/queues.jsonl")


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def isoz(s: str | None) -> str | None:
    """A store timestamp (`…T12:54:39.764Z`) in the format's form, to the second."""
    if not s:
        return None
    try:
        return iso(datetime.fromisoformat(s.replace("Z", "+00:00")))
    except ValueError:
        return None


def sig(v, n: int = 4):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(f"{v:.{n}g}")


def ratio(a, b, n: int = 3):
    return sig(a / b, n) if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b else None


def panel(name: str, title: str, kind: str, columns: list, rows: list, *, source: str, updated_at: str, unit: str | None = None,
          note: str | None = None, y_scale: str | None = None, producer: str = "verity", target: float | None = None,
          target_label: str | None = None, stacked: bool | None = None) -> dict:
    p = {"id": f"{producer}/{name}", "title": title, "kind": kind, "unit": unit, "y_scale": y_scale, "target": target,
         "target_label": target_label, "stacked": stacked, "note": note,
         "updated_at": updated_at, "source": source, "columns": columns, "rows": rows}
    return {k: p[k] for k in FIELDS if p[k] is not None}


def problems(p: dict) -> list[str]:
    """Everything the site's validator would refuse (panel-format.md §1)."""
    out = [f"unknown field `{k}`" for k in p if k not in FIELDS]
    out += [f"missing `{k}`" for k in sorted(REQUIRED - set(p))]
    if out:
        return out

    def text(k: str, hi: int) -> None:
        if k in p and not (isinstance(p[k], str) and 1 <= len(p[k]) <= hi):
            out.append(f"`{k}` must be 1 to {hi} characters")

    if not (isinstance(p["id"], str) and ID.fullmatch(p["id"])):
        out.append("`id` must be producer/name in lowercase letters, digits and dashes")
    text("title", 120), text("unit", 20), text("note", 500), text("source", 300)
    if p["kind"] not in ("table", "line", "bar"):
        out.append("`kind` must be table, line or bar")
    if "y_scale" in p and (p["y_scale"] not in ("linear", "log") or p["kind"] == "table"):
        out.append("`y_scale` must be linear or log, on a line or bar chart")
    if "target" in p and (p["kind"] == "table" or isinstance(p["target"], bool) or not isinstance(p["target"], (int, float))):
        out.append("`target` must be a number, on a line or bar chart")
    if "target_label" in p and ("target" not in p or not (isinstance(p["target_label"], str) and 1 <= len(p["target_label"]) <= 40)):
        out.append("`target_label` must be 1 to 40 characters, with a target")
    if "stacked" in p and (p["kind"] != "bar" or not isinstance(p["stacked"], bool)):
        out.append("`stacked` must be true or false, on a bar chart")
    if not (isinstance(p["updated_at"], str) and ISO.fullmatch(p["updated_at"])):
        out.append("`updated_at` must be ISO 8601 with a zone")
    cols, rows = p["columns"], p["rows"]
    if not (isinstance(cols, list) and 1 <= len(cols) <= 20 and all(isinstance(c, str) and 1 <= len(c) <= 60 for c in cols)
            and len(set(cols)) == len(cols)):
        out.append("`columns` must be 1 to 20 distinct names of up to 60 characters")
    elif p["kind"] != "table" and len(cols) < 2:
        out.append("a chart needs at least 2 columns")
    if not isinstance(rows, list) or len(rows) > 1000:
        out.append("`rows` must be a list of at most 1000 rows")
    else:
        for i, r in enumerate(rows):
            if not isinstance(r, list) or len(r) != len(cols):
                out.append(f"row {i} has {len(r) if isinstance(r, list) else '?'} cells, not {len(cols)}")
                continue
            for j, c in enumerate(r):
                ok = c is None or (isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c)) or (
                    isinstance(c, str) and len(c) <= 200 and (j == 0 or p["kind"] == "table"))
                if not ok:
                    out.append(f"row {i} column {j}: {'a series cell must be a number or null' if isinstance(c, str) else 'bad cell'}")
    size = len(json.dumps(p).encode())
    if size > MAX_BODY:
        out.append(f"{size} bytes, over 64 KiB")
    return out


def fit(p: dict) -> dict:
    """A table over 64 KiB: its long strings cut shorter, then its rows cut, saying so in `note`."""
    if p["kind"] != "table":
        return p
    for cap in (200, 140, 100, 70):
        if len(json.dumps(p).encode()) <= MAX_BODY:
            return p
        p["rows"] = [[c[:cap - 1] + "…" if isinstance(c, str) and len(c) > cap else c for c in r] for r in p["rows"]]
    n = len(p["rows"])
    while len(json.dumps(p).encode()) > MAX_BODY and p["rows"]:
        p["rows"] = p["rows"][:int(len(p["rows"]) * 0.9)]
        p["note"] = (f"The first {len(p['rows'])} of {n} rows (the site's 64 KiB limit). " + p.get("note", ""))[:500]
    return p


def publish(p: dict, url: str, key: str) -> str:
    req = urllib.request.Request(f"{url}/api/panels/{p['id']}", data=json.dumps(p).encode(), method="PUT",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return str(r.status)
        except urllib.error.HTTPError as e:
            if e.code != 503 or attempt == 2:
                return f"{e.code} {e.read()[:300].decode(errors='replace')}"
        except (urllib.error.URLError, OSError) as e:
            if attempt == 2:
                return f"unreachable: {e}"
        time.sleep(5)
    return "unreachable"


def write(out: Path, p: dict) -> None:
    dest = out / f"{p['id'].split('/', 1)[1]}.json"
    tmp = dest.with_name(f".{dest.name}.tmp")
    tmp.write_text(json.dumps(p, indent=1))
    os.replace(tmp, dest)


# ---------------------------------------------------------------------------------------------------------------- store group

def store_panels(ns, now: datetime, errors: list) -> tuple[list[dict], dict]:
    sys.path[:0] = [str(ns.research_src), str(HERE)]
    import ovd  # noqa: PLC0415  the overnight dashboard's loaders
    from research.store.local import LocalStore  # noqa: PLC0415

    store = LocalStore(ns.store) if ns.store else LocalStore()
    info = {"refresh": "skipped (--no-refresh)"}
    if not ns.no_refresh:
        try:
            rep = store.refresh(jobs=8)
            info["refresh"] = f"+{rep['attempts']} attempts, +{rep['labels']} labels, {len(rep['errors'])} errors, {rep['seconds']:.0f} s"
        except Exception as e:  # noqa: BLE001  a stale catalog still renders; the run line says so
            info["refresh"] = f"failed: {type(e).__name__}: {e}"
    idx = store.index
    probs = ovd.Problems()
    points, cov_labels, _, _ = ovd.load(idx, probs)
    bench = ovd.BENCHMARK["build"]
    series = {k: s for k, s in ovd.build_series(points).items()
              if s.ws != "build" or (s.config or "").split(" @ ")[0] in bench}
    src = "evidence store (`research data`), ov.* labels: ov.ws {ws}, ov.metric {metric}"
    out = []

    def attempt_chart(name, title, ws, metric, phases, unit, note, y_scale=None):
        mine = sorted((s for s in series.values() if s.ws == ws and s.metric == metric and s.phase in phases and s.path),
                      key=lambda s: (s.config or "", s.rank, s.line))[:19]
        cols, by_col = ["attempt"], []
        for s in mine:
            label = " ".join(x for x in (s.config, s.sub or (s.phase if len(phases) > 1 else None), s.line) if x)[:60]
            while label in cols:
                label = label[:57] + "'"
            cols.append(label)
            by_col.append({p.attempt: p.value for p in s.path})
        attempts = sorted({a for c in by_col for a in c})
        rows = [[a, *[sig(c.get(a)) for c in by_col]] for a in attempts]
        when = max((p.labelled for s in mine for p in s.points), default=None)
        return panel(name, title, "line", cols, rows, unit=unit, note=note, updated_at=isoz(when) or iso(now),
                     source=src.format(ws=ws, metric=metric), y_scale=y_scale)

    how = ("x is the optimization attempt. A line's last point is its quiet-hour re-measurement when it has one; noisy runs are "
           "included as measured and gate failures are left out.")
    build = [
        ("build-throughput", "Build throughput: circuit gates emitted per wall-clock second", "throughput", ovd.PHASES, "gates/s",
         f"Per benchmark config, phase and optimization line; higher is better; log scale. {how}", "log"),
        ("build-peak-mem", "Build peak memory: peak RSS", "peak-rss", ovd.PHASES, "GiB",
         f"Per benchmark config, phase and optimization line; lower is better. {how}", None),
        ("build-commit-time", "Commit time on the RTX PRO 6000", "commit-time", (ovd.COMMIT,), "s",
         f"Commit stage seconds per benchmark config and phase; lower is better. {how}", None),
    ]
    for name, title, metric, phases, unit, note, y_scale in build:
        out.append(attempt_chart(name, title, "build", metric, phases, unit, note, y_scale))
    out.append(attempt_chart("pouw-overhead", "PoUW overhead (POUS): verifier-accepted PoUW panel rows", "pouw", "overhead",
                             ovd.PHASES, "× native", f"POUS's measured rows as labelled in Verity's evidence store; log scale. {how}",
                             "log"))
    out.append(build_speedups(ovd, series, now, src))
    cov = ovd.load_coverage(store, idx, cov_labels, probs)
    out += coverage_panels(ovd, cov, now)
    try:
        out += merge_panels(idx, ns, now)
    except Exception as e:  # noqa: BLE001  the store panels still go out
        errors.append(f"merge: {type(e).__name__}: {e}")
    info["points"], info["coverage_configs"] = len(points), len(cov)
    return out, info


def at0(s):
    return next((p for p in s.path if p.attempt == 0), None) if s else None


def measured(p) -> str:
    return "quiet re-measure" if p.remeasure else "noisy" if p.noisy else "quiet"


def build_speedups(ovd, series, now, src) -> dict:
    thr = {(s.config, s.phase, s.line): s for s in series.values()
           if s.ws == "build" and s.metric == "throughput" and s.phase in ovd.PHASES and s.path}
    mem = {(s.config, s.phase, s.line): s for s in series.values()
           if s.ws == "build" and s.metric == "peak-rss" and s.phase in ovd.PHASES and s.path}
    rows, when = [], []
    for (cfg, ph, line), s in sorted(thr.items(), key=lambda kv: (kv[0][0], ovd.PHASES.index(kv[0][1]), kv[0][2])):
        base = at0(s) or at0(thr.get((cfg, ph, "build-v1")))
        last, best = s.latest, s.best
        m = mem.get((cfg, ph, line))
        mbase, mlast = at0(m) or at0(mem.get((cfg, ph, "build-v1"))), m.latest if m else None
        mg = lambda p: sig(p.value / 1e6) if p else None  # noqa: E731
        rows.append([cfg, ph, line, mg(base), mg(last), last.attempt, measured(last), ratio(last.value, base.value if base else None),
                     mg(best), best.attempt if best else None, ratio(best.value if best else None, base.value if base else None),
                     sig(mbase.value, 3) if mbase else None, sig(mlast.value, 3) if mlast else None])
        when += [p.labelled for p in s.points]
    cols = ["Config", "Phase", "Line", "Baseline (Mgates/s)", "Latest (Mgates/s)", "Latest attempt", "Latest measured",
            "Speedup, latest (×)", "Best (Mgates/s)", "Best attempt", "Speedup, best (×)", "Peak GiB, baseline", "Peak GiB, latest"]
    return panel("build-speedups", "Build speedups against the baseline", "table", cols, rows,
                 updated_at=isoz(max(when, default=None)) or iso(now), source=src.format(ws="build", metric="throughput, peak-rss"),
                 note="Throughput is gates per wall-clock second at 32 vCPU on vy-nebius-1. The baseline is attempt 0 of the same line "
                      "(build-v1's when the line has none). The morning report's two inflated baselines (Llama prefill's first-run "
                      "bytecode compile, OLMoE decode's ~290 s lock wait) are not corrected here.")


PREMERGE = re.compile(r"^pre-merge( #\d+)* @ [0-9a-f]+;\s*")


def cause_of(why: str) -> str:
    """A coverage cell's cause with its run-specific parts dropped, so cells with one cause group together."""
    text = PREMERGE.sub("", why or "no cause named").split(";", 1)[0].strip()
    return re.sub(r"\b\d+\b", "N", text)[:140] or "no cause named"


def coverage_panels(ovd, cov: list[dict], now: datetime) -> list[dict]:
    when = isoz(max((r.get("_when") or "" for r in cov), default="")) or iso(now)
    src = "evidence store: campaign overnight-sep30 config records (config_record.json, sweep.json) and ov.ws coverage labels"
    gate = (f"pass = built, committed and {ovd.GATE_K} of {ovd.GATE_K} drawn units replayed bit-exact; short replay = passed "
            f"with fewer than {ovd.GATE_K}; not run = stopped before any stage failed. One row per config: its latest record.")
    gpus = sorted({str(r["config"].get("gpu") or "?") for r in cov})
    tally = []
    for g in gpus + ["all"]:
        here = [r for r in cov if g == "all" or str(r["config"].get("gpu") or "?") == g]
        c = Counter(r["status"] for r in here)
        tally.append([g, *[c.get(k, 0) for k in STATUS_ORDER], len(here)])
    tally.sort(key=lambda r: (r[0] == "all", -r[1], r[0]))
    out = [panel("coverage-tally", "vLLM coverage: configs by GPU and status", "table",
                 ["GPU", "Pass", "Finding", "Short replay", "Unsupported", "Not run", "Configs"], tally,
                 updated_at=when, source=src, note=gate)]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in cov:
        if r["status"] != "pass":
            groups[cause_of(r["why"])].append(r)
    causes = []
    for cause, rs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        st = Counter(r["status"] for r in rs)
        latest = max(rs, key=lambda r: r.get("_when") or "")
        causes.append([cause, len(rs), ", ".join(f"{st[k]} {k}" for k in STATUS_ORDER if st[k]),
                       ", ".join(sorted({str(r["config"].get("gpu") or "?") for r in rs})), latest["run"]])
    out.append(panel("coverage-causes", "vLLM coverage: why configs don't pass", "table",
                     ["Cause", "Configs", "Status", "GPUs", "Latest run"], causes, updated_at=when, source=src,
                     note="Every config that doesn't pass, grouped by its recorded cause (the pre-merge prefix and the text after "
                          "the first ';' dropped, standalone numbers shown as N). Fixes and owners are in the morning report."))
    fams = sorted({ovd.family(r["config"].get("model")) for r in cov})
    frows = []
    for fam in fams:
        cells = []
        for g in gpus:
            c = Counter(r["status"] for r in cov if ovd.family(r["config"].get("model")) == fam
                        and str(r["config"].get("gpu") or "?") == g)
            cells.append(", ".join(f"{c[k]} {k}" for k in STATUS_ORDER if c[k]) or None)
        frows.append([fam, *cells])
    out.append(panel("coverage-families", "vLLM coverage by model family and GPU", "table", ["Family", *gpus], frows,
                     updated_at=when, source=src, note=gate))
    crow = [[r["row"], str(r["config"].get("gpu") or ""), r["status"], r["why"][:200] or None, r["run"], isoz(r.get("_when"))]
            for r in sorted(cov, key=lambda r: (STATUS_ORDER.index(r["status"]) if r["status"] in STATUS_ORDER else 9, r["row"]))]
    out.append(fit(panel("coverage-configs", "vLLM coverage: every config", "table",
                         ["Config", "GPU", "Status", "Cause", "Run", "Recorded (UTC)"], crow, updated_at=when, source=src,
                         note=gate + " The config id is model__dtype__gpu__tp__batch__input__output__arrivals__sampler__engine.")))
    return out


def git(gitdir: Path, *args: str) -> str:
    return subprocess.run(["git", f"--git-dir={gitdir}", *args], check=True, capture_output=True, text=True, timeout=300).stdout


def machine_of(doc: dict) -> str | None:
    ex = doc.get("execution") or {}
    return ex.get("machine") or (ex.get("remote") or {}).get("machine") or ex.get("host")


def merge_panels(idx, ns, now: datetime) -> list[dict]:
    if not ns.no_fetch:
        git(ns.git, "fetch", "-q", "origin", f"+refs/heads/{ns.branch}:refs/heads/{ns.branch}")
    log = git(ns.git, "log", ns.branch, "--first-parent", "--merges", "-n", "600", "--format=%H%x1f%P%x1f%s%x1f%b%x1e")
    trains = []
    for rec in log.split("\x1e"):
        parts = rec.strip("\n").split("\x1f")
        if len(parts) < 4:
            continue
        h, parents, subj, body = parts
        m = re.match(r"Merge train (\S+) \(([0-9a-f]+)\)", subj)
        ps = parents.split()
        if m and len(ps) > 1:
            run = re.search(r"^Check: (r\d{8}-\d{6}-[0-9a-f]{4})", body, re.M)
            trains.append({"name": m[1], "merged": h, "first": ps[0], "head": ps[1], "run": run[1] if run else None})
    head_of = {t["head"]: t for t in trains}

    def attempt(run):
        rows = idx.sql("SELECT start_utc, wall_s, rc, state, manifest_json FROM attempts WHERE id = ?", (run,)) if run else []
        return rows[0] if rows else None

    rows, ends = [], []
    for t in trains[:30]:
        subjects = git(ns.git, "log", "--no-merges", "--format=%s", f"{t['first']}..{t['head']}").splitlines()
        prs = list(dict.fromkeys(re.findall(r"#(\d{2,4})\b", "\n".join(subjects))))
        a = attempt(t["run"])
        start, wall = (a[0], a[1]) if a else (None, None)
        if start and wall:
            ends.append(datetime.fromisoformat(start.replace("Z", "+00:00")) + timedelta(seconds=wall))
        rows.append([t["name"], " ".join(f"#{n}" for n in prs)[:200] or None, len(subjects), t["head"][:8], t["merged"][:9], t["run"],
                     isoz(start), sig(wall / 60, 3) if wall else None, machine_of(json.loads(a[4])) if a and a[4] else None])
    when = iso(max(ends)) if ends else iso(now)
    out = [panel("merge-trains", "Merge trains landed on main, newest first", "table",
                 ["Train", "PRs named in its commits", "Commits", "Head", "Merged as", "Check run", "Check started (UTC)",
                  "Check minutes", "Checked on"], rows, updated_at=when,
                 source=f"git: main's `Merge train` commits and their Check trailers; evidence store: those check Attempts",
                 note="The latest 30 trains. PRs are the #numbers in the carried commits' subjects, so a PR whose commits don't "
                      "name it is missing. main's commit dates are normalized, so the check's start stands in for when a train ran.")]
    since = iso(now - timedelta(hours=24))
    crows, cends = [], []
    for run, start, wall, rc, state, commit, mj in idx.sql(
            "SELECT id, start_utc, wall_s, rc, state, source_commit, manifest_json FROM attempts WHERE tool = 'check' AND start_utc >= ? "
            "ORDER BY start_utc DESC", (since,)):
        result = "pass" if state == "done" and rc == 0 else state if state in ("running", "cancelled") else f"fail (rc {rc})"
        t = head_of.get(commit or "")
        crows.append([run, isoz(start), sig(wall / 60, 3) if wall else None, result, (commit or "")[:8] or None,
                      t["name"] if t else None, machine_of(json.loads(mj)) if mj else None])
        if start and wall:
            cends.append(datetime.fromisoformat(start.replace("Z", "+00:00")) + timedelta(seconds=wall))
    out.append(panel("merge-checks", "Every `check` in the last 24 hours", "table",
                     ["Run", "Started (UTC)", "Minutes", "Result", "Commit", "Landed as train", "Machine"], crows,
                     updated_at=iso(max(cends)) if cends else iso(now), source="evidence store: tool=check Attempts; git: main's trains",
                     note="A check whose commit is a train's head on main shows that train; the others are checks of branches or of "
                          "trains that didn't land. Attempts are published when a check ends, so one in progress isn't listed."))
    return out


# ---------------------------------------------------------------------------------------------------------------- node1 group

def kraw(path: str) -> dict:
    env = dict(os.environ, KUBECONFIG=os.environ.get("KUBECONFIG", str(Path.home() / ".kube" / "config")))
    return json.loads(subprocess.run(["kubectl", "get", "--raw", path], check=True, capture_output=True, text=True, timeout=120,
                                     env=env).stdout)


def prom_range(query: str, start: float, end: float, step: int = 60) -> list[dict]:
    qs = urllib.parse.urlencode({"query": query, "start": int(start), "end": int(end), "step": step})
    return kraw(f"{PROM}/query_range?{qs}")["data"]["result"]


def prom_now(query: str) -> list[dict]:
    return kraw(f"{PROM}/query?{urllib.parse.urlencode({'query': query})}")["data"]["result"]


def node1_panels(ns, now: datetime, errors: list) -> tuple[list[dict], dict]:
    sys.path.insert(0, str(HERE))
    import util_collect as uc  # noqa: PLC0415  the steward's utilization definitions

    end = now.timestamp()
    start = (now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=23)).timestamp()
    prom = {k: prom_range(q, start, end) for k, q in uc.Q.items()}
    qlines = [json.loads(ln) for ln in QUEUES.read_text().splitlines() if ln.strip().startswith("{")] if QUEUES.exists() else []
    src = "vy-nebius-1: Prometheus (DCGM, node-exporter) and Kueue samples (vy-usage, /workspace/usage/queues.jsonl)"
    out = []

    summary = uc.summarize_n1(prom, start, end)
    alloc = uc.kueue_allocated(qlines, start)
    hours = summary["hours"]
    cur = uc.hour(end)
    hrows = []
    for h in sorted(hours, reverse=True):
        b = hours[h]
        cpu = 100 * b.get("cpu_core_h_busy", 0) / (b["cpu_h_observed"] * CORES) if b.get("cpu_h_observed") else None
        hrows.append([h + (" (so far)" if h == cur else ""), sig(b.get("gpu_h_observed"), 3), sig(alloc.get(h, 0.0), 3) if qlines else None,
                      sig(b.get("gpu_h_held_or_busy"), 3), sig(b.get("gpu_h_busy"), 3), sig(b.get("gpu_h_effective"), 3),
                      sig(b.get("gpu_h_idle"), 3), sig(cpu, 3), sig(b.get("ram_peak_gib"), 3)])
    out.append(panel("node1-hours", "vy-nebius-1 per hour: GPU-hours and CPU", "table",
                     ["Hour (UTC)", "GPU-h", "Kueue-allocated", "Held or busy", "Busy", "Util-weighted", "Idle", "CPU busy %",
                      "RAM peak GiB"], hrows, updated_at=iso(now), source=src,
                     note="The steward's definitions: busy = DCGM utilization above 0 in a minute; held = 1 GiB or more of GPU memory "
                          "in use; util-weighted = the sum of utilization; idle = GPU-hours neither held nor busy. Kueue-allocated is "
                          "GPU-hours admitted to jobs, from 5-minute samples. The last 24 hours, newest first."))

    tot = summary["total"]
    o, b, h = tot.get("gpu_h_observed", 0), tot.get("gpu_h_busy", 0), tot.get("gpu_h_held_or_busy", 0)
    a = sum(alloc.values())
    split = [("Busy", b), ("GPU memory held, not busy", h - b), ("Allocated by Kueue, no GPU memory held", max(a - h, 0)),
             ("Neither allocated nor held", o - max(a, h))]
    srows = [[k, sig(v, 3), sig(100 * v / o, 3) if o else None] for k, v in split] + [["All GPU-hours", sig(o, 3), 100.0 if o else None]]
    out.append(panel("node1-split", "vy-nebius-1: where the last 24 hours of GPU-hours went", "table",
                     ["Where the GPU-hours went", "GPU-h", "Share %"], srows, updated_at=iso(now), source=src,
                     note=f"Since {uc.hour(start)}, the postmortem's split. Held counts any process on a GPU, inside Kueue or not. "
                          "Busy is generous: utilization above 0 in a minute."))

    util = {r["metric"].get("gpu"): {int(float(t)): float(v) for t, v in r["values"]} for r in prom["gpu_util"]}
    fb = {r["metric"].get("gpu"): {int(float(t)): float(v) for t, v in r["values"]} for r in prom["gpu_fb_mib"]}
    cpu = {int(float(t)): float(v) for r in prom["cpu_busy"] for t, v in r["values"]}
    buckets: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for g, ser in util.items():
        for t, u in ser.items():
            bk = buckets[t // 300 * 300]
            bk["busy"].append(100.0 * (u > 0))
            bk["util"].append(u)
            bk["held"].append(100.0 * (u > 0 or fb.get(g, {}).get(t, 0.0) >= 1024))
    for t, c in cpu.items():
        buckets[t // 300 * 300]["cpu"].append(100 * c / CORES)
    for d in qlines:
        if d.get("t", 0) >= start:
            gpus = sum(q.get("usage", {}).get("nvidia.com/gpu", 0.0) for q in d.get("queues", {}).values())
            buckets[int(d["t"]) // 300 * 300]["alloc"].append(100 * gpus / GPUS)
    mean = lambda xs: sig(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    urows = [[iso(datetime.fromtimestamp(t, timezone.utc)), mean(v["busy"]), mean(v["util"]), mean(v["held"]), mean(v["alloc"]),
              mean(v["cpu"]), GPU_TARGET, CPU_TARGET] for t, v in sorted(buckets.items())]
    out.append(panel("node1-utilization", "vy-nebius-1 utilization, last 24 hours", "line",
                     ["time", "GPU busy %", "GPU util %", "GPU held %", "Kueue-allocated %", "CPU busy %", "GPU busy target %",
                      "CPU busy target %"], urows, unit="%", updated_at=iso(now), source=src,
                     note="5-minute means over the 8 GPUs and 192 vCPUs. GPU busy: utilization above 0 (the steward's measure, and "
                          "Daniel's 60% target); GPU util: mean utilization; held: 1 GiB or more of memory in use; Kueue-allocated: "
                          "GPUs admitted to jobs. CPU busy: non-idle, non-iowait cores (target 70%)."))

    if qlines:
        last = qlines[-1]
        qrows = []
        for q, v in sorted(last.get("queues", {}).items()):
            n, u, bo = v.get("nominal", {}), v.get("usage", {}), v.get("borrowed", {})
            gi = lambda d: sig(d.get("memory", 0) / 2**30, 3)  # noqa: E731
            qrows.append([q, u.get("nvidia.com/gpu"), n.get("nvidia.com/gpu"), bo.get("nvidia.com/gpu"), u.get("cpu"), n.get("cpu"),
                          gi(u), gi(n), v.get("admitted"), v.get("pending"), sig((v.get("oldest_pending_s") or 0) / 60, 3),
                          sig(v.get("gpu_util_mean"), 3)])
        out.append(panel("node1-queues", "vy-nebius-1 Kueue queues now", "table",
                         ["Queue", "GPUs used", "GPUs nominal", "GPUs borrowed", "CPUs used", "CPUs nominal", "Memory used GiB",
                          "Memory nominal GiB", "Admitted", "Pending", "Oldest pending (min)", "Pods' mean GPU util %"], qrows,
                         updated_at=iso(datetime.fromtimestamp(last["t"], timezone.utc)), source=src,
                         note="Each ClusterQueue's quota and use from vy-usage's latest sample (every 5 minutes)."))
    else:
        errors.append(f"node1-queues: no samples in {QUEUES}")

    try:
        pods = kraw("/api/v1/pods")["items"]
        queue_of = {(p["metadata"]["namespace"], p["metadata"]["name"]): (p["metadata"].get("labels") or {}).get("kueue.x-k8s.io/queue-name")
                    for p in pods}
    except Exception as e:  # noqa: BLE001  the GPU table goes out without queue names
        queue_of = {}
        errors.append(f"node1-gpus: pod list: {type(e).__name__}: {e}")
    now_q = {"util": "DCGM_FI_DEV_GPU_UTIL", "fb": "DCGM_FI_DEV_FB_USED", "engine": "DCGM_FI_PROF_GR_ENGINE_ACTIVE",
             "power": "DCGM_FI_DEV_POWER_USAGE"}
    gpu: dict[str, dict] = defaultdict(dict)
    for k, q in now_q.items():
        for r in prom_now(q):
            m = r["metric"]
            d = gpu[m.get("gpu", "?")]
            d[k] = float(r["value"][1])
            if m.get("pod"):
                d["pod"], d["ns"] = m["pod"], m.get("namespace")
    grows = []
    for g in sorted(gpu, key=lambda x: int(x) if x.isdigit() else 99):
        d = gpu[g]
        holder = d.get("pod") or ("(no pod: direct or leaked)" if d.get("fb", 0) >= 1024 else None)
        grows.append([int(g) if g.isdigit() else g, sig(d.get("util"), 3), sig(100 * d["engine"], 3) if "engine" in d else None,
                      sig(d.get("fb", 0) / 1024, 3), sig(d.get("power"), 3), holder, queue_of.get((d.get("ns"), d.get("pod")))])
    out.append(panel("node1-gpus", "vy-nebius-1 GPUs now", "table",
                     ["GPU", "Util %", "Engine active %", "Memory GiB", "Power W", "Held by pod", "Kueue queue"], grows,
                     updated_at=iso(now), source=src,
                     note="Each GPU's latest DCGM reading (1-minute scrapes) and the pod DCGM names as holding it. Engine active is "
                          "DCGM_FI_PROF_GR_ENGINE_ACTIVE, closer to real SM occupancy than utilization."))
    try:
        pool = json.loads(POOL.read_text()) if POOL.exists() else {}
        out += pool_panels(pool, n1_from_prom(summary, urows, grows, now), now, errors)
    except Exception as e:  # noqa: BLE001  the node-1 panels still go out
        errors.append(f"pool: {type(e).__name__}: {e}")
    try:
        out += hillclimb_panels(now, errors)
    except Exception as e:  # noqa: BLE001  the node-1 panels still go out
        errors.append(f"hillclimb: {type(e).__name__}: {e}")
    return out, {"queue_samples": len(qlines), "pool_file": POOL.exists(),
                 "hillclimb_files": len(list(HILLCLIMB.glob("*.json"))) if HILLCLIMB.is_dir() else 0}


# ---------------------------------------------------------------------------------------------------------------- proofs hill climb

HILLCLIMB = Path("/workspace/usage/hillclimb")  # proofs' roll-ups, one `verity/hillclimb-rollup/v0` file per subcircuit
SUBCIRCUIT = re.compile(r"[a-z0-9]+/[A-Za-z0-9_]+/K=\d+")
HILLCLIMB_COLUMNS = ["Step", "Label", "Commit", "Run", "Time (UTC)", "Overhead", "Throughput VU/s", "GPU util", "GPU-held s/VU",
                     "Byte-identical", "Flags", "VUs", "Session s/VU", "Setup s", "Verify s/statement", "Scheme", "GPU",
                     "vCPU", "Dot", "Prove-only s/VU"]


def hillclimb_panels(now: datetime, errors: list) -> list[dict]:
    """One table per subcircuit, a row per optimization step. The title is the subcircuit id, which the site parses."""
    out = []
    for f in sorted(HILLCLIMB.glob("*.json")) if HILLCLIMB.is_dir() else []:
        try:
            doc = json.loads(f.read_text())
            sub = doc["subcircuit"]
            sid = sub["id"]
            if doc.get("schema") != "verity/hillclimb-rollup/v0" or not SUBCIRCUIT.fullmatch(sid):
                raise ValueError(f"schema {doc.get('schema')!r}, subcircuit {sid!r}")
            text = lambda v, n=200: str(v)[:n] if v not in (None, "") else None  # noqa: E731
            rows = []
            for p in sorted(doc.get("points") or [], key=lambda p: p.get("step", 0)):
                hw = p.get("hardware") or {}
                rows.append([p.get("step"), text(p.get("label")), text(p.get("commit"), 12), text(p.get("run_id")),
                             isoz(p.get("time_utc")), sig(p.get("overhead")), sig(p.get("throughput_vu_per_s")),
                             sig(p.get("gpu_util")), sig(p.get("gpu_held_s_per_vu")),
                             {True: "yes", False: "no"}.get(p.get("byte_identical")), text(" ".join(p.get("flags") or [])),
                             p.get("vus") if isinstance(p.get("vus"), int) else None, sig(p.get("t_prove_s_per_vu")),
                             sig(p.get("setup_s")), sig(p.get("verify_s_per_statement")), text(p.get("scheme"), 60),
                             text(hw.get("gpu")), hw.get("vcpu") if isinstance(hw.get("vcpu"), int) else None,
                             text(sub.get("dot")), sig(p.get("t_prove_only_s_per_vu"))])
            name = "hillclimb-" + re.sub(r"[^a-z0-9]+", "-", sid.lower()).strip("-")
            when = isoz(doc.get("updated_utc")) or iso(datetime.fromtimestamp(f.stat().st_mtime, timezone.utc))
            out.append(fit(panel(name, sid, "table", HILLCLIMB_COLUMNS, rows, updated_at=when,
                                 source=f"proofs' hill-climb roll-up (vy-nebius-1 {f}), from hillclimb.json records",
                                 note="overhead = datasheet dense peak ÷ (2·K·VUs ÷ session time), the session being the whole interactive "
                                      "proof job, verifier round trips included (proofs, 4:47 PM PDT Sep 30). Prove-only s/VU is the "
                                      "prover buckets alone. One row per step.")))
        except Exception as e:  # noqa: BLE001  one bad file doesn't stop the others
            errors.append(f"hillclimb {f.name}: {type(e).__name__}: {e}")
    return out


# ---------------------------------------------------------------------------------------------------------------- pool (infra/*)

POOL = Path("/workspace/usage/infra-pool.json")  # infra's file: node 2 (and later node 1), every 5 minutes, written atomically
POOL_HISTORY = HERE / "pool-history.jsonl"      # this script's own 24 h of pool samples, one line a run
USEFUL_TARGET = 85  # Daniel, 2:06 PM PDT Sep 30: both nodes at >= 85% useful GPU busy by noon Oct 1 PDT
TARGET_BY = "noon Oct 1 PDT"
TARGET_END = datetime(2026, 10, 1, 19, 0, tzinfo=timezone.utc)
WATERMARK_GPU_H = 12
POOL_STALE_S = 900


def pct(v):
    return sig(100 * v, 3) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def n1_from_prom(summary: dict, util_rows: list, gpu_rows: list, now: datetime) -> dict:
    """Node 1 in the pool file's shape, from this script's own Prometheus reading, until infra's file carries n1."""
    recent = [r for r in util_rows if r[0] >= iso(now - timedelta(hours=1))]
    last = util_rows[-1] if util_rows else None
    mean = lambda j, rows: (sum(r[j] for r in rows if r[j] is not None) / max(1, sum(r[j] is not None for r in rows)) / 100) if rows else None  # noqa: E731
    return {"gpu_busy_5m": last[1] / 100 if last and last[1] is not None else None, "gpu_busy_1h": mean(1, recent),
            "gpu_held_1h": mean(3, recent), "useful_share_1h": None, "cpu_busy_5m": last[5] / 100 if last and last[5] is not None else None,
            "cpu_busy_1h": mean(5, recent),
            "gpus": [{"index": g[0], "busy_5m": (g[1] or 0) / 100, "holder": g[5], "class": g[6]} for g in gpu_rows],
            "_from": "prometheus"}


def held_idle(node: dict):
    """Held-idle share: infra's field when present, else GPUs with a holder and no work in the last 5 minutes."""
    if isinstance(node.get("held_idle_1h"), (int, float)):
        return node["held_idle_1h"]
    if isinstance(node.get("gpu_held_1h"), (int, float)) and isinstance(node.get("gpu_busy_1h"), (int, float)):
        return max(0.0, node["gpu_held_1h"] - node["gpu_busy_1h"])
    gpus = node.get("gpus") or []
    return sum(1 for g in gpus if g.get("holder") and (g.get("busy_5m") or 0) < 0.05) / len(gpus) if gpus else None


def useful(node: dict):
    b, u = node.get("gpu_busy_1h"), node.get("useful_share_1h")
    if not isinstance(b, (int, float)):
        return None
    return b * u if isinstance(u, (int, float)) else b


def pool_panels(pool: dict, n1: dict | None, now: datetime, errors: list) -> list[dict]:
    nodes = dict(pool.get("nodes") or {k: v for k, v in pool.items() if k in ("n1", "n2")})
    if "n1" not in nodes and n1:
        nodes["n1"] = n1
    gen = pool.get("generated_at") or iso(now)
    src = f"infra's {POOL} (generated {gen}) and vy-nebius-1's Prometheus"
    names = {"n1": "node 1", "n2": "node 2"}
    order = [k for k in ("n1", "n2") if k in nodes]
    out = []

    busy_rows = []
    for k in order:
        d = nodes[k]
        u, b, hi = useful(d), d.get("gpu_busy_1h"), held_idle(d)
        filler = (b - u) if isinstance(b, (int, float)) and isinstance(u, (int, float)) else None
        busy_rows.append([f"{names[k]} GPU", pct(u), pct(filler), pct(hi)])
    for k in order:
        d = nodes[k]
        busy_rows.append([f"{names[k]} CPU", pct(d.get("cpu_busy_1h", d.get("cpu_busy_5m"))), None, None])
    out.append(panel("pool-busy", "Both nodes, last hour: useful work, other busy, held idle", "bar", ["node", "useful", "busy, not useful", "held idle"],
                     busy_rows, producer="infra", unit="%", target=USEFUL_TARGET, target_label="useful GPU target", stacked=True,
                     updated_at=iso(now), source=src,
                     note="GPU: useful busy (busy times infra's useful share), busy on filler, and held idle (memory held or a lease, no work). "
                          "CPU: busy. Node 1's useful share is its busy share until infra's file carries n1."))

    hist = []
    if POOL_HISTORY.exists():
        cutoff = iso(now - timedelta(hours=24))
        hist = [h for h in (json.loads(ln) for ln in POOL_HISTORY.read_text().splitlines() if ln.strip()) if h.get("t", "") >= cutoff]
    hist.append({"t": iso(now), **{k: {"useful": useful(nodes[k]), "cpu": nodes[k].get("cpu_busy_5m")} for k in order}})
    tmp = POOL_HISTORY.with_name(f".{POOL_HISTORY.name}.tmp")
    tmp.write_text("".join(json.dumps(h) + "\n" for h in hist))
    os.replace(tmp, POOL_HISTORY)
    cols = ["time"] + [f"{names[k]} useful GPU %" for k in ("n1", "n2")] + [f"{names[k]} CPU %" for k in ("n1", "n2")]
    urows = [[h["t"]] + [pct((h.get(k) or {}).get("useful")) for k in ("n1", "n2")] + [pct((h.get(k) or {}).get("cpu")) for k in ("n1", "n2")]
             for h in hist]
    out.append(panel("pool-utilization", "Useful GPU busy and CPU busy, both nodes, last 24 hours", "line", cols, urows, producer="infra",
                     unit="%", target=USEFUL_TARGET, target_label="useful GPU target", updated_at=iso(now), source=src,
                     note="One point per 5-minute run: each node's last-hour useful GPU busy, and CPU busy. History starts when this panel did."))

    qrows = []
    for k in order:
        d = nodes[k]
        qrows.append([names[k], sig(d.get("ready_gpu_h"), 3), WATERMARK_GPU_H, d.get("waiting"), "yes" if d.get("timed_window") else "no"])
    tot = [n for n in (nodes[k].get("ready_gpu_h") for k in order) if isinstance(n, (int, float))]
    qrows.append(["both", sig(sum(tot), 3) if tot else None, 2 * WATERMARK_GPU_H,
                  sum(nodes[k].get("waiting") or 0 for k in order), ""])
    out.append(panel("pool-queue", "Work waiting: ready GPU-hours against the watermark", "table",
                     ["node", "ready GPU-h", "watermark GPU-h", "jobs waiting", "timed window now"], qrows, producer="infra",
                     updated_at=iso(now), source=src,
                     note=f"Ready GPU-h is queued work that could start now. Below the {WATERMARK_GPU_H} GPU-h watermark, a node can run dry."))

    grows = []
    for k in order:
        for g in nodes[k].get("gpus") or []:
            grows.append([names[k], g.get("index"), pct(g.get("busy_5m")), g.get("holder"), g.get("class")])
    out.append(panel("pool-gpus", "Every GPU now: busy and who holds it", "table", ["node", "GPU", "busy % (5 min)", "holder", "class"],
                     grows, producer="infra", updated_at=iso(now), source=src, note="The latest 5-minute reading per GPU."))

    krows = []
    for k in order:
        for r in nodes[k].get("kinds") or []:
            if str(r.get("kind", "")).startswith("timed:"):
                continue
            krows.append([names[k], r.get("kind"), (", ".join(o[:11] for o in r.get("owners") or []) or None) and ", ".join(o[:11] for o in r.get("owners") or [])[:200],
                          sig(r.get("leased_gpu_h"), 3), pct(r.get("efficiency")), sig(r.get("idle_gpu_h"), 3)])
    krows.sort(key=lambda r: -(r[5] or 0))
    if krows:
        out.append(panel("pool-kinds", "Job kinds, last 24 hours", "table",
                         ["node", "kind", "owners", "leased GPU-h", "efficiency %", "idle GPU-h"], krows[:200], producer="infra",
                         updated_at=iso(now), source=src))

    left_h = (TARGET_END - now).total_seconds() / 3600
    trows = []
    for k in order:
        u = useful(nodes[k])
        prev = [h for h in hist if h["t"] <= iso(now - timedelta(hours=1)) and (h.get(k) or {}).get("useful") is not None]
        trend = (u - prev[-1][k]["useful"]) if prev and isinstance(u, (int, float)) else None
        status = ("met" if u >= USEFUL_TARGET / 100 else "behind" if (trend or 0) <= 0 else "on track") if isinstance(u, (int, float)) else "no data"
        trows.append([f"{names[k]} useful GPU busy", f">= {USEFUL_TARGET}%", f"{pct(u)}%" if u is not None else None,
                      f"{'+' if (trend or 0) >= 0 else ''}{sig(100 * trend, 2)} pts/h" if trend is not None else None, status, TARGET_BY,
                      f"{sig(left_h, 3)} h" if left_h > 0 else "ended"])
    out.append(panel("targets", "Daniel's targets: how each is going", "table",
                     ["target", "goal", "now (last hour)", "change over the hour", "status", "by", "time left"], trows, producer="infra",
                     updated_at=iso(now), source=src,
                     note="Met: at or above the goal now. On track: below, and rising over the last hour. Behind: below and flat or falling."))
    age = (now - datetime.fromisoformat(gen.replace("Z", "+00:00"))).total_seconds() if isinstance(gen, str) else 0
    if age > POOL_STALE_S:
        errors.append(f"pool: {POOL} is {int(age)} s old")
    return out


# ---------------------------------------------------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--group", choices=("store", "node1"), required=True)
    ap.add_argument("--out", type=Path, default=HERE / "outbox", help="where each panel's JSON is written")
    ap.add_argument("--key", type=Path, default=Path("~/.config/verity/panels.key").expanduser(),
                    help="the panels:write key; absent means a dry run")
    ap.add_argument("--url", default=os.environ.get("PANELS_URL", URL))
    ap.add_argument("--store", type=Path, help="the evidence store's local root (default: $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--research-src", type=Path, default=HERE / "research-src", help="a directory holding the `research` package")
    ap.add_argument("--git", type=Path, default=HERE / "verity.git", help="a clone of Verity whose BRANCH is read for merge trains")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--no-refresh", action="store_true", help="read the store's catalog as it is")
    ap.add_argument("--no-fetch", action="store_true", help="read the clone's BRANCH as it is")
    ns = ap.parse_args(argv)
    t0, now, errors = time.time(), datetime.now(timezone.utc).replace(microsecond=0), []
    ns.out.mkdir(parents=True, exist_ok=True)
    try:
        panels, info = (store_panels if ns.group == "store" else node1_panels)(ns, now, errors)
    except Exception as e:  # noqa: BLE001  nothing is sent; the site keeps the last good panels
        panels, info = [], {}
        errors.append(f"{ns.group}: {type(e).__name__}: {e} | {traceback.format_exc(limit=3).splitlines()[-3:]}")
    key = ns.key.read_text().strip() if ns.key.is_file() else None
    sent: dict[str, str] = {}
    for p in panels:
        bad = problems(p)
        if bad:
            errors.append(f"{p.get('id')}: invalid: {'; '.join(bad)}")
            continue
        write(ns.out, p)
        if key:
            sent[p["id"]] = publish(p, ns.url, key)
    refused = {k: v for k, v in sent.items() if v not in ("200", "201")}
    line = {"at": iso(now), "group": ns.group, "seconds": round(time.time() - t0, 1), "panels": len(panels),
            "published": (f"{len(sent) - len(refused)} of {len(sent)}" if key else "dry run (no key)"), **info,
            "refused": refused or None, "errors": errors or None}
    (ns.out / f"_last-run-{ns.group}.json").write_text(json.dumps(line, indent=1))
    print(json.dumps(line), flush=True)
    return 1 if errors or refused else 0


if __name__ == "__main__":
    sys.exit(main())
