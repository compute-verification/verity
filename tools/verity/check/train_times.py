#!/usr/bin/env python3
"""`tools/verity/check/train_times.py [--last N] [--save DIR] [--json] [--no-refresh]`: where the train checks' wall time goes.

Reads the store's check Attempts (`tool = 'check'`) that drive launched on `next` (`train.sh`: `slot.py --priority`, its twins
grouped by their pair), keeps the last N whose check reached a verdict, and prints per step the median, p90 and max wall time
over them, and per train its critical path and the long pole on it.  A step's time is its `steps.jsonl` record's `seconds`
(the result's `steps`); a suite's is its row in the result's `reuse` (0 when reused); the slot wait is the run's `slot.json`
(`since` to `took`).  The phases around the steps:

    queued     the train's commit on `next` (its commit time) until `research run` registered the check (its run id's time)
    launch     registered until the run started on the machine
    slot wait  waiting in the machine's check line (`slot.py`)
    setup      the slot taken until the first step started (`uv run --locked` and the verdict import)
    finish     the last step's end until the run ended (the result and the verdict pack)

A train's critical path is those phases, the preflight, and the chain of steps ending at its last step to end, walked back
through its group in `ci.toml` (and a job's `after`); `other` is what the gaps between them add up to.  Its long pole is the
step on that chain with the most seconds.  `--save DIR` keeps each train's records there, one directory per run;
`--from DIR...` reads such directories (`steps.jsonl`, and when present `slot.json`, `suites.json`, `suites-lean.json` and
`train.json`), or a check's own run directory, without the store.  Stdlib only, apart from `research.store` for the store.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CI = Path(__file__).resolve().parent / "ci.toml"
#: a step that ended this long after the next one on its chain started still precedes it (records are to the second)
TOL_S = 2.0
PHASES = ("queued", "launch", "slot wait", "setup", "finish")
SUITE_STEPS = (("pytest", "suites.json"), ("lean-suites", "suites-lean.json"))
RUN_TIME = re.compile(r"^r(\d{8}-\d{6})-")


def utc(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def jobs(path: Path = CI) -> dict[str, dict]:
    """Each declared job's group, its place in that group, and the job it starts `after`."""
    out, place = {}, {}
    for j in tomllib.loads(path.read_text()).get("job", []):
        g = j.get("group")
        place[g] = place.get(g, -1) + 1
        out[j["name"]] = {"group": g, "order": place[g], "after": j.get("after")}
    return out


def load_dir(d: Path) -> dict:
    """One train from its directory: `steps.jsonl` (the last record per step), `slot.json`, the suites' rows and `train.json`."""
    steps: dict[str, dict] = {}
    for line in (d / "steps.jsonl").read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            steps[rec["step"]] = rec
    def doc(name: str):
        p = d / name
        return json.loads(p.read_text()) if p.is_file() else None
    suites = {}
    for step, name in SUITE_STEPS:
        rows = doc(name)
        if isinstance(rows, list):
            suites[step] = {r["suite"]: float(r.get("seconds") or 0) for r in rows if r.get("status") not in ("cached", "skipped")}
    return {**(doc("train.json") or {"run": d.name}), "slot": doc("slot.json") or {}, "steps": list(steps.values()),
            "suites": suites}


def costs(train: dict, js: dict[str, dict]) -> dict:
    """The train's phases, steps and suites (seconds), its critical path [(name, seconds)] and long pole."""
    spans = {}
    for r in train["steps"]:
        end = utc(r.get("ended_utc"))
        if end is not None:
            spans[r["step"]] = (end.timestamp() - float(r.get("seconds") or 0), end.timestamp(), float(r.get("seconds") or 0))
    pre = sorted((n for n in spans if n.startswith("preflight-")), key=lambda n: spans[n][1])
    rest = [n for n in spans if n not in pre]
    first = min((s for s, _, _ in spans.values()), default=None)
    last = max(rest or pre, key=lambda n: spans[n][1], default=None)
    chain = []
    cur = last if last in rest else None
    while cur is not None:
        chain.insert(0, cur)
        j = js.get(cur, {})
        group = [n for n in rest if n != cur and js.get(n, {}).get("group", n) == j.get("group", cur)
                 and js[n]["order"] < j.get("order", 0)]
        before = group + [a for n in group + [cur] if (a := js.get(n, {}).get("after")) in spans]
        before = [n for n in before if spans[n][1] <= spans[cur][0] + TOL_S and n not in chain]
        cur = max(before, key=lambda n: spans[n][1], default=None)
    t = {k: utc(train.get(k)) for k in ("committed_utc", "registered_utc", "start_utc", "end_utc")}
    since, took = utc(train["slot"].get("since")), utc(train["slot"].get("took"))
    start = since or t["start_utc"]

    def gap(a, b) -> float | None:
        a = a.timestamp() if isinstance(a, datetime) else a
        b = b.timestamp() if isinstance(b, datetime) else b
        return None if a is None or b is None else max(0.0, b - a)

    phases = {"queued": gap(t["committed_utc"], t["registered_utc"]), "launch": gap(t["registered_utc"], t["start_utc"]),
              "slot wait": gap(since, took) if took else (0.0 if start else None), "setup": gap(took or start, first),
              "finish": gap(spans[last][1] if last else None, t["end_utc"])}
    path = [(p, phases[p]) for p in PHASES[:4] if phases[p] is not None]
    path += [(n, spans[n][2]) for n in pre + chain] + ([("finish", phases["finish"])] if phases["finish"] is not None else [])
    total = gap(t["committed_utc"] or t["registered_utc"] or t["start_utc"], t["end_utc"])
    if total is not None:
        path.append(("other", max(0.0, total - sum(s for _, s in path))))
    pole = max(chain, key=lambda n: spans[n][2], default=None)
    times = {**{p: s for p, s in phases.items() if s is not None}, **{n: s for n, (_, _, s) in spans.items()}}
    for step, rows in train["suites"].items():
        times |= {f"{step}: {k}": v for k, v in rows.items()}
    return {"times": times, "path": path, "long_pole": pole, "check_s": gap(first, spans[last][1] if last else None),
            "total_s": total}


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[max(0, math.ceil(q * len(s)) - 1)]


def report(trains: list[dict], js: dict[str, dict]) -> dict:
    """Per name (phase, step, suite) its median, p90 and max over the trains, a suite counting 0 where its step ran without
    it (reused); each train's row; and the critical path's costs ranked by their median over the trains (0 off a train's path)."""
    rows = [{"run": t.get("run"), "commit": (t.get("commit") or "")[:12], "machine": t.get("machine"), "verdict": t.get("verdict"),
             **costs(t, js)} for t in trains]
    names: dict[str, list[float]] = {}
    for r in rows:
        for k, v in r["times"].items():
            names.setdefault(k, [])
    for k in names:
        step = k.split(": ", 1)[0] if ": " in k else None
        names[k] = [r["times"][k] if k in r["times"] else 0.0 for r in rows if k in r["times"] or (step and step in r["times"])]
    def stats(xs: list[float]) -> dict:
        return {"n": len(xs), "median": statistics.median(xs), "p90": pct(xs, 0.9), "max": max(xs)}
    per = {k: stats(v) for k, v in names.items() if v}
    on_path = sorted({n for r in rows for n, _ in r["path"]})
    path = {n: {**stats([dict(r["path"]).get(n, 0.0) for r in rows]),
                "long_pole_in": sum(r["long_pole"] == n for r in rows)} for n in on_path}
    return {"trains": rows, "steps": per, "path": dict(sorted(path.items(), key=lambda kv: (-kv[1]["median"], -kv[1]["max"])))}


def minutes(s: float | None) -> str:
    return "" if s is None else f"{s / 60:.1f}"


def render(rep: dict, note: str = "") -> str:
    rows = rep["trains"]
    out = [f"# Train checks: where the wall time goes ({len(rows)} trains, minutes)", ""] + ([note, ""] if note else [])
    out += ["## Per train", "", "| run | commit | machine | verdict | queued | launch | slot wait | setup | check | finish | total "
            "| long pole | critical path |", "|" + "---|" * 13]
    for r in rows:
        t = r["times"]
        crit = " → ".join(f"{n} {minutes(s)}" for n, s in r["path"] if s >= 30)
        out.append(f"| {r['run']} | {r['commit']} | {r['machine'] or ''} | {r['verdict'] or ''} | "
                   + " | ".join(minutes(t.get(p)) for p in PHASES[:4]) + f" | {minutes(r['check_s'])} | {minutes(t.get('finish'))} | "
                   f"{minutes(r['total_s'])} | {r['long_pole'] or ''} {minutes(t.get(r['long_pole']))} | {crit} |")
    out += ["", "## The critical path's costs (0 where off a train's path; long pole in: trains where it was the long pole)", "",
            "| cost | median | p90 | max | long pole in |", "|---|---|---|---|---|"]
    out += [f"| {n} | {minutes(s['median'])} | {minutes(s['p90'])} | {minutes(s['max'])} | {s['long_pole_in']} |"
            for n, s in rep["path"].items()]
    out += ["", "## Per step (a suite: 0 where its step reused it)", "", "| step | n | median | p90 | max |", "|---|---|---|---|---|"]
    order = list(PHASES) + sorted((k for k in rep["steps"] if k not in PHASES and ": " not in k), key=lambda k: -rep["steps"][k]["median"])
    order += sorted((k for k in rep["steps"] if ": " in k), key=lambda k: (-rep["steps"][k]["median"], -rep["steps"][k]["max"]))
    out += [f"| {k} | {s['n']} | {minutes(s['median'])} | {minutes(s['p90'])} | {minutes(s['max'])} |"
            for k in order if (s := rep["steps"].get(k)) and (": " not in k or s["max"] > 0)]
    return "\n".join(out) + "\n"


def committed(sha: str) -> str | None:
    """The commit's time, fetched by sha when this clone lacks it (GitHub serves a rebuilt `next`'s commits by sha)."""
    def show():
        r = subprocess.run(["git", "show", "-s", "--format=%cI", sha], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() or None if r.returncode == 0 else None
    if (got := show()) is None:
        subprocess.run(["git", "fetch", "-q", "origin", sha], cwd=ROOT, capture_output=True)
        got = show()
    return got


def from_store(last: int, save: Path, refresh: bool = True) -> tuple[list[Path], str]:
    """The last ``last`` train checks that reached a verdict, each written to a directory of ``save``; and a line on those left out."""
    from research.store.local import LocalStore

    store = LocalStore()
    if refresh and store.remote is not None:
        store.refresh(attempts_only=True)
    groups: dict[str, list[dict]] = {}
    for (doc,) in store.index.sql("SELECT manifest_json FROM attempts WHERE tool = 'check'"):
        a = json.loads(doc)
        params = a.get("params") or {}
        if params.get("priority") is True:
            groups.setdefault(str(params.get("twin") or a["id"]).split("@")[0], []).append(a)
    ran, stopped, none = [], 0, 0
    for members in groups.values():
        got = [a for a in members if (a.get("outputs") or {}).get("result")]
        verdict = got and (got[0].get("status") or {}).get("validation")
        if verdict in ("passed", "failed"):
            ran.append(got[0])
        else:
            stopped += bool(got)
            none += not got
    ran.sort(key=lambda a: RUN_TIME.match(a["id"])[1])
    dirs = []
    for a in ran[-last:]:
        d = save / a["id"]
        d.mkdir(parents=True, exist_ok=True)
        meta = store.get_manifest(a["outputs"]["result"]).meta
        (d / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in meta.get("steps") or []))
        for step, name in SUITE_STEPS:
            reran = ((meta.get("reuse") or {}).get("steps") or {}).get(step, {}).get("reran") or []
            (d / name).write_text(json.dumps([{"suite": x["suite"], "status": "passed", "seconds": x.get("seconds")}
                                              for x in reran if isinstance(x, dict)]))
        rec = (a.get("outputs") or {}).get("run_record")
        if rec and any(f["path"] == "slot.json" for f in store.listing(rec)):
            store.fetch(rec, d, paths=["slot.json"])
        ex, sha = a.get("execution") or {}, (a.get("source") or {}).get("commit") or ""
        reg = datetime.strptime(RUN_TIME.match(a["id"])[1], "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
        (d / "train.json").write_text(json.dumps({
            "run": a["id"], "commit": sha, "machine": ex.get("machine") or ex.get("host"), "verdict": a["status"]["validation"],
            "committed_utc": committed(sha) if sha else None, "registered_utc": reg.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "start_utc": ex.get("start_utc"), "end_utc": ex.get("end_utc")}, indent=1))
        dirs.append(d)
    note = (f"The last {len(dirs)} of {len(ran)} train checks in the store that reached a verdict; left out: {stopped} stopped "
            f"before one (cancelled), {none} whose check never ran (every twin left the line, or it was cancelled waiting).")
    return dirs, note


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="train_times.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--last", type=int, default=10)
    ap.add_argument("--save", type=Path)
    ap.add_argument("--from", dest="dirs", type=Path, nargs="+")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-refresh", action="store_true")
    a = ap.parse_args(argv)
    if a.dirs:
        dirs, note = a.dirs, ""
    else:
        dirs, note = from_store(a.last, a.save or Path(tempfile.mkdtemp(prefix="train-times-")), not a.no_refresh)
    if not dirs:
        print("train_times: no train check to read", file=sys.stderr)
        return 1
    rep = report([load_dir(d) for d in dirs], jobs())
    print(json.dumps({**rep, "note": note}, indent=1) if a.json else render(rep, note).rstrip("\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
