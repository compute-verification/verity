"""Corpus driver: the same builder, unchanged, on every record of an extracted archive tree — one roundtrip per record in
its own process (timeout, address-space cap, child peak RSS), results as one JSON line per record + a markdown table.

    python -m verity_capture.experimental.aot.corpus --records /workspace/cp/aot/corpus/records --out /workspace/cp/aot/runs \
        [--only NAME ...] [--skip NAME ...] [--as-gb 60] [--timeout 3600]

A record directory is `<name>/` holding either the run itself (`run.json`, `capture/log.jsonl.gz`) or `<name>/<name>/...`
(the small archives nest); a `program.json` next to either level (compact archives) is the accepted descriptor for the
full structural check, otherwise `fold_summary.json`'s digest is the (digest-only) reference.  Every failure mode is a
recorded result: `UNEXPRESSIBLE` (vocabulary), `INEXPRESSIBLE-SCHEDULE` (hints cannot express the observed schedule),
`DIFFERS` (built ≠ accepted, with the diff in check.json), `TIMEOUT` / `CRASH` (resource), never a silent skip.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path


def find_record(d: Path) -> Path | None:
    for cand in (d / d.name.replace(".run1", ""), d):
        if (cand / "run.json").exists():
            return cand
    hits = sorted(p.parent for p in d.rglob("run.json"))
    return hits[0] if hits else None


def find_accepted(d: Path, rec: Path) -> Path | None:
    for cand in (rec / "program.json", d / "program.json", rec / "fold_summary.json", d / "fold_summary.json"):
        if cand.exists():
            return cand
    return None


def run_one(name: str, rec: Path, accepted: Path | None, out: Path, *, as_gb: float | None, timeout: float, python: str,
            predictor: str | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [python, "-m", "verity_capture.experimental.aot.run", "roundtrip", "--record", str(rec), "--out", str(out)]
    if accepted is not None:
        cmd += ["--accepted", str(accepted)]
    if predictor:
        cmd += ["--predictor", predictor]

    def limits():
        if as_gb:
            b = int(as_gb * 1024 ** 3)
            resource.setrlimit(resource.RLIMIT_AS, (b, b))

    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    t0 = time.time()
    status = "ok"
    try:
        p = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[3]), capture_output=True, text=True, timeout=timeout,
                           preexec_fn=limits if sys.platform != "win32" else None)
        rc, tail = p.returncode, (p.stdout + p.stderr)[-4000:]
    except subprocess.TimeoutExpired as e:
        rc, tail, status = -1, ((e.stdout or b"") + (e.stderr or b""))[-4000:].decode(errors="replace") if isinstance(e.stdout, bytes) else str(e.stdout)[-4000:], "TIMEOUT"
    secs = round(time.time() - t0, 1)
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    child_rss_gb = round(after / (1024 ** 2), 3) if sys.platform != "darwin" else round(after / (1024 ** 3), 3)   # KiB on Linux, B on macOS
    (out / "driver.log").write_text(f"$ {' '.join(cmd)}\nrc={rc} seconds={secs} child_peak_rss_gb={child_rss_gb}\n{tail}\n")
    row = {"record": name, "rc": rc, "seconds": secs, "child_peak_rss_gb": child_rss_gb, "accepted": str(accepted) if accepted else None}
    rt = out / "roundtrip.json"
    if rt.exists():
        d = json.loads(rt.read_text())
        row.update({k: d.get(k) for k in ("verdict", "evidence", "hints_empty", "hints_naive_bits", "hints_json_bytes", "n_instances", "timings",
                                         "max_rss_gb", "descriptor_bytes", "digest_built", "digest_accepted", "reason", "encode_status", "observed")})
    else:
        row["verdict"] = status if status != "ok" else "CRASH"
        row["tail"] = tail[-1500:]
    if row.get("encode_status") == "inexpressible-schedule":
        row["verdict"] = "INEXPRESSIBLE-SCHEDULE"
        er = json.loads((out / "encode_report.json").read_text())
        row["reason"] = er.get("report", {}).get("error")
    return row


def markdown(rows: list[dict]) -> str:
    lines = ["| record | verdict | evidence | hints | naive bits | instances | build s | peak RSS GB | bytes | note |", "|---|---|---|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        t = r.get("timings") or {}
        note = (r.get("reason") or "")[:140]
        lines.append(f"| `{r['record']}` | {r.get('verdict')} | {r.get('evidence') or ''} | {'∅' if r.get('hints_empty') else ('h' if r.get('hints_empty') is False else '')} | "
                     f"{r.get('hints_naive_bits', '')} | {r.get('n_instances', '')} | {t.get('total_s', '')} | {r.get('max_rss_gb') or r.get('child_peak_rss_gb')} | "
                     f"{r.get('descriptor_bytes', '')} | {note} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--as-gb", type=float, default=None, help="RLIMIT_AS for each roundtrip process")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--predictor", default=None, help="predictor id passed to every roundtrip (default: run.py's default)")
    a = ap.parse_args(argv)
    root, out = Path(a.records), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(d.name for d in root.iterdir() if d.is_dir())
    if a.only:
        names = [n for n in names if n in set(a.only)]
    names = [n for n in names if n not in set(a.skip)]
    results = out / "corpus_results.jsonl"
    rows: list[dict] = []
    if results.exists():
        rows = [json.loads(l) for l in results.read_text().splitlines() if l.strip()]
    done = {r["record"] for r in rows}
    for n in names:
        if n in done:
            continue
        rec = find_record(root / n)
        if rec is None:
            row = {"record": n, "verdict": "NO-RUN-JSON"}
        else:
            print(f"[corpus] {n} ...", flush=True)
            row = run_one(n, rec, find_accepted(root / n, rec), out / n, as_gb=a.as_gb, timeout=a.timeout, python=a.python, predictor=a.predictor)
        print(f"[corpus] {n}: {row.get('verdict')} ({row.get('seconds')}s, rss {row.get('max_rss_gb') or row.get('child_peak_rss_gb')} GB) {row.get('reason') or ''}", flush=True)
        rows.append(row)
        with results.open("a") as f:
            f.write(json.dumps(row) + "\n")
        (out / "CORPUS.md").write_text(markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
