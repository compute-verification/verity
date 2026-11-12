"""table.py RUN...: one row per staged type of each fetched run (~/.research/runs/RUN/out/TAG): what stage.json and
stage.time record, and each Lean command's exit and words from verdicts.json. Prints TSV-ish lines and writes
/opt/cursor/artifacts/cir259/table.json."""
import json
import re
import sys
from pathlib import Path

RUNS = Path.home() / ".research/runs"
rows = []


def timed(p: Path):
    if not p.exists():
        return None, None
    t = p.read_text(errors="replace")
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", t)
    e = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", t)
    return (e[1] if e else None), (round(int(m[1]) / 2**20, 4) if m else None)


def words(v: dict | None) -> str:
    if v is None:
        return "-"
    if "verdict" in v:
        vd = v["verdict"]
        return "accepted" if vd.get("accepted") else f"refused: {vd.get('why')}"
    if "ks_form" in v:
        return f"KS-FORM {v['ks_form']}"
    if "setup" in v:
        return v["setup"]
    if "statement" in v:
        pc = (v["statement"] or {}).get("partition_check") or {}
        return f"ok, units {pc.get('units')}"
    return f"exit {v.get('exit')}: {v.get('said')}"


for run in sys.argv[1:]:
    out = RUNS / run / "out"
    log = (out / "job.log").read_text(errors="replace") if (out / "job.log").exists() else ""
    for d in sorted(p for p in out.iterdir() if p.is_dir() and (p / "B").is_dir()):
        tag = d.name
        row: dict = {"run": run, "tag": tag}
        sj = d / "B/stage.json"
        wall, peak = timed(d / "B/stage.time")
        row.update(stage_wall=wall, stage_peak_gb=peak)
        if sj.exists():
            r = json.loads(sj.read_text())
            row.update(staged=True, definition=r.get("census") or r.get("definition"), n=r["n"], ands=r["ands"], k_log=r["k_log"],
                       circuit_bytes=r["circuit_bytes"], statement=r.get("statement"), seconds=r.get("seconds"),
                       maxrss_gb=r.get("maxrss_gb"), lookups=bool(r.get("lookups")), unit=r.get("unit"), word=r.get("word"))
            pb = d / "B/program.bytes"
            row["program_bytes"] = int(pb.read_text()) if pb.exists() else None
        else:
            err = (d / "B/stage.err").read_text(errors="replace").strip().splitlines()
            row.update(staged=False, refusal=(err[-1] if err else "")[:600])
        vj = d / "verdicts.json"
        v = json.loads(vj.read_text()) if vj.exists() else {}
        row["lean"] = {k: {"says": words(x), "exit": x.get("exit"), "wall": x.get("wall"),
                           "peak_gb": round(x["maxrss_kb"] / 2**20, 4) if x.get("maxrss_kb") else None} for k, x in v.items()}
        held = v.get("B.statement.held") or {}
        pc = ((held.get("statement") or {}).get("partition_check") or {})
        row["units"] = pc.get("units")
        row["note"] = (pc.get("note") or "")[:200]
        pt = d / "B/prove.time"
        if pt.exists():
            try:
                j = json.loads(pt.read_text())
                row["prove"] = {k: j.get(k) for k in ("wall_s", "maxrss_kb", "rc", "peak_rss_kb", "seconds", "returncode")}
            except ValueError:
                row["prove"] = pt.read_text()[:200]
        st = d / "B/serve.time"
        if st.exists():
            try:
                j = json.loads(st.read_text())
                row["serve"] = {k: j.get(k) for k in ("wall_s", "maxrss_kb", "rc", "peak_rss_kb", "seconds", "returncode")}
            except ValueError:
                row["serve"] = st.read_text()[:200]
        m = re.search(rf"{re.escape(tag)}: --zk session FAILED: (.*)", log)
        if m:
            row["session_failed"] = m[1][:500]
        rows.append(row)
    for m in re.finditer(r"^\S+ (\S+): staging refused: rc (\d+)", log, re.M):
        if not any(r["run"] == run and r["tag"] == m[1] for r in rows):
            err = (out / m[1] / "B/stage.err")
            rows.append({"run": run, "tag": m[1], "staged": False,
                         "refusal": (err.read_text(errors="replace").strip().splitlines() or [""])[-1][:600] if err.exists() else ""})

A = Path("/opt/cursor/artifacts/cir259")
A.mkdir(parents=True, exist_ok=True)
(A / "table.json").write_text(json.dumps(rows, indent=1))
for r in rows:
    L = r.get("lean", {})
    g = lambda k: (L.get(k) or {}).get("says", "-")  # noqa: E731
    lp = max([x["peak_gb"] or 0 for x in L.values()] or [0])
    print(f"{r['run'][-4:]} {r['tag']}: {r.get('definition') or r.get('refusal')}")
    if r.get("staged"):
        print(f"    N={r['n']} ands={r['ands']:,} k_log={r['k_log']} circuit={r['circuit_bytes']:,} B program={r.get('program_bytes')} B "
              f"stage {r['stage_wall']} at {r['stage_peak_gb']} GB; reads {r['statement']}; lean peak {lp} GB")
        print(f"    statement held: {g('B.statement.held')} | other-def: {g('B.statement.other-def')[:90]} | no-program: {g('B.statement.no-program')} | ks-form: {g('B.ks-form')[:200]}")
        print(f"    verify held: {g('B.verify.held')[:200]} | other-def: {g('B.verify.other-def')[:90]} | no-program: {g('B.verify.no-program')[:60]}")
        if r.get("session_failed"):
            print(f"    SESSION FAILED: {r['session_failed'][:400]}")
        if r.get("prove"):
            print(f"    prove {r['prove']} serve {r.get('serve')}")
