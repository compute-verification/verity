"""cover_ksteady.py RUNS_DIR KS_DIR ARTS.txt OUT.json: HASH-98's Lean rehearsals (`benchmarks/prove/lean_verify.sh`) read.

ARTS.txt has a line ``LEAN_RUN ART`` per rehearsal: the proof archive it verified, a run record. RUNS_DIR/LEAN_RUN/out holds
the rehearsal's files (`research fetch LEAN_RUN --all`); KS_DIR/<last 4 of LEAN_RUN>/out the archive's stage and prove
timings (`research data fetch ART --path ...`). One record per rehearsal: the word, its Definition, the units, Lean's three
verdicts, what Lean was handed, the staging step's exit and peak, the prover's device and peak.
"""
import glob
import json
import re
import sys
from pathlib import Path

RUNS, KS, ARTS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])


def timed(path: str) -> dict:
    t = Path(path).read_text()
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", t)
    rc = re.search(r"Exit status: (\d+)", t)
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", t)
    return {"rc": int(rc.group(1)) if rc else None, "peak_gib": round(int(rss.group(1)) / 2**20, 2) if rss else None,
            "wall": wall.group(1) if wall else None}


out = []
for line in ARTS.read_text().split("\n"):
    if not line.strip():
        continue
    run, art = line.split()
    lv = json.loads((RUNS / run / "out" / "lean-verify.json").read_text())
    rec = {"run": run, "archive": art, "word": lv["block"]["word"], "definition": lv["block"]["definition"],
           "k_log": lv["block"]["k_log"], "blocks": lv["block"]["blocks"], "full": lv["block"]["full"],
           "accepted": lv["accepted"], "verdicts": [r["accepted"] for r in lv["verifier"]["runs"]],
           "lean_peak_gib": max(r["maxrss_gib"] for r in lv["verifier"]["runs"]), "label": lv["label"],
           "lean_commit": lv["host"]["commit"], "lean_node": lv["host"].get("node"), "lean_cpu": lv["host"].get("cpu_model"),
           "bound_program": lv["statement"]["bound"]["program_sha512"][:16],
           "handed_program": lv["statement"]["program"]["sha512"][:16],
           "program_held": lv["statement"]["bound"]["program_sha512"] == lv["statement"]["program"]["sha512"],
           "partition_handed": "--partition" in json.dumps(lv["verifier"].get("command"))}
    v0 = (RUNS / run / "out" / "verify-0.log").read_text()
    m = re.search(r"VERDICT (\{.*\})", v0)
    if m:
        d = json.loads(m.group(1))
        u = d.get("unit_draw") or {}
        rec["unit_draw"] = {"k": u.get("k"), "population": u.get("population"), "law": u.get("law")}
    k = KS / run[-4:] / "out"
    st = glob.glob(str(k / "stage" / "*" / "time-stage.txt"))
    sr = glob.glob(str(k / "stage" / "*" / "stage.record.json"))
    pv = glob.glob(str(k / "cpu" / "*" / "time-prove.txt")) + glob.glob(str(k / "gpu" / "*" / "time-prove.txt"))
    rec["stage"] = timed(st[0]) if st else None
    if sr:
        s = json.loads(Path(sr[0]).read_text())
        rec["stage"].update(definition=s.get("definition"), word=s.get("word"), census=s.get("census"), n=s.get("n"))
    rec["prove"] = timed(pv[0]) if pv else None
    if pv:
        rec["prove"]["device"] = Path(pv[0]).parent.parent.name
    cfg = k / "config.txt"
    rec["config"] = cfg.read_text().split("\n")[0] if cfg.exists() else None
    out.append(rec)
OUT.write_text(json.dumps(out, indent=1) + "\n")
for r in out:
    print(r["run"][-4:], r["accepted"], r["verdicts"], r["blocks"], (r["stage"] or {}).get("peak_gib"), (r["prove"] or {}).get("device"),
          (r["prove"] or {}).get("peak_gib"), r["program_held"], r["partition_handed"], r.get("unit_draw"), r["word"][:70])
