"""`run.sh`'s summary: per statement and mode (``gpu``, ``cpu``), the timed sessions' prove times from the prover's ``LIVE``
records, the GPU's busy share, tensor-pipe activity and power over the timed sessions' window, the prover's CPU seconds and
peak memory, and the staged statement's size.

    python summary.py OUT_DIR      prints the summary as JSON
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

#: a decimal number as ``float`` reads one; a sampler's other fields (``[N/A]``, units, headers) aren't samples
NUM = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


def text(p: Path) -> str:
    return p.read_text(errors="replace") if p.exists() else ""


def time_v(p: Path) -> dict:
    """GNU time -v: wall, user and system seconds, peak RSS (whole process: warm sessions and setup included)."""
    t = text(p)

    def num(key):
        m = re.search(key + r".*: (\S+)", t)
        return m.group(1) if m else None

    wall = num(r"Elapsed \(wall clock\) time")
    if wall:
        parts = [float(x) for x in wall.split(":")]
        wall = sum(x * 60 ** i for i, x in enumerate(reversed(parts)))
    rss = num(r"Maximum resident set size \(kbytes\)")
    user, sys_ = num("User time"), num("System time")
    return {"wall_s": wall, "user_s": float(user) if user else None, "sys_s": float(sys_) if sys_ else None,
            "peak_rss_gb": round(int(rss) / 2**20, 2) if rss else None}


def stamped(p: Path) -> list[tuple[float, str]]:
    out = []
    for line in text(p).splitlines():
        t, _, rest = line.partition("\t")
        if NUM.fullmatch(t):
            out.append((float(t), rest))
    return out


def window_mean(rows: list[tuple[float, list[float]]], t0: float, t1: float) -> list[float] | None:
    xs = [v for t, v in rows if t0 <= t <= t1 and v]
    if not xs:
        return None
    return [statistics.fmean(c[i] for c in xs) for i in range(min(map(len, xs)))]


def floats(s: str) -> list[float]:
    return [float(x) for x in re.split(r"[,\s]+", s.strip()) if NUM.fullmatch(x)]


def session_set(d: Path, gpu: bool = True) -> dict:
    """With ``gpu`` false (the CPU prover), the GPU's samples are left out: they show only other tenants of the card."""
    live = [(t, json.loads(l[5:])) for t, l in stamped(d / "prove.tsv") if l.startswith("LIVE")]
    if not live:
        return {"error": text(d / "prove.err")[-1500:]}
    warm = [(t, v) for t, v in live if v.get("warm")]
    timed = [(t, v) for t, v in live if not v.get("warm")]

    def med(f):
        return statistics.median(f(v) for _, v in timed) if timed else None

    t0 = warm[-1][0] if warm else None
    t1 = timed[-1][0] if timed else None
    s = {"sessions": len(live), "timed": len(timed), "accepted": [v.get("accepted") for _, v in live],
         "instances": live[0][1].get("instances"), "k_log": live[0][1].get("k_log"),
         "prove_total_s": med(lambda v: v.get("prove_total_s")),
         "prove_total_all_s": [v.get("prove_total_s") for _, v in live],
         "e2e_s": med(lambda v: v.get("e2e_s")), "e2e_all_s": [v.get("e2e_s") for _, v in live],
         "witness_s": med(lambda v: v.get("witness_s") or 0),
         "session_s": (t1 - t0) / len(timed) if t0 and timed else None,
         "proof_bytes": timed[-1][1].get("proof_bytes") if timed else None,
         "prove_s": timed[-1][1].get("prove_s") if timed else None,
         "buckets": [{k: x for k, x in b.items() if k.startswith("t.")} for b in (timed[-1][1].get("buckets") or [])] if timed else None,
         "process": time_v(d / "time-prove.txt")}
    if t0 and t1:
        smi = [(t, floats(r)) for t, r in stamped(d / "smi.tsv")]
        m = window_mean(smi, t0, t1) if gpu else None
        if m and len(m) >= 4:
            s["gpu_util_mean_pct"], s["gpu_mem_mean_mib"], s["sm_clock_mean_mhz"], s["power_mean_w"] = (round(x, 2) for x in m[:4])
            s["gpu_busy_s_per_session"] = round(m[0] / 100 * s["session_s"], 4)
        jif = [(t, v[4]) for t, v in smi if len(v) >= 5]
        before = [j for t, j in jif if t <= t0]
        after = [j for t, j in jif if t >= t1]
        if before and after:
            s["prover_cores_cpu_s_per_session"] = round((after[0] - before[-1]) / 100 / len(timed), 3)
        if not gpu:
            return s
        mem = [v[1] for _, v in smi if len(v) >= 2]
        s["gpu_mem_peak_mib"] = max(mem) if mem else None
        dcgm = [(t, floats(r.split("GPU", 1)[1])[1:] if "GPU" in r else []) for t, r in stamped(d / "dcgm.tsv")]
        md = window_mean(dcgm, t0, t1)
        if md and len(md) >= 3:
            s["dcgm_sm_active"], s["dcgm_tensor_active"], s["dcgm_dram_active"] = (round(x, 4) for x in md[:3])
    return s


def main(out: Path) -> dict:
    res = {"host": text(out / "host.txt"), "binary": text(out / "binary.sha256").strip(),
           "statements": {}}
    for st in sorted((out / "stage").glob("k*-n*")):
        rec = json.loads(text(st / "rec.json") or "{}")
        res["statements"][st.name] = {"definition": rec.get("definition"), "ands": rec.get("ands"), "in_bits": rec.get("in_bits"),
                                      "out_bits": rec.get("out_bits"), "stage": rec.get("stage"), "sizes": text(st / "sizes.txt"),
                                      "stage_process": time_v(st / "time-stage.txt")}
    for mode in ("gpu", "cpu", "gpu-nsys"):
        for d in sorted((out / mode).glob("k*-n*")):
            res["statements"].setdefault(d.name, {})[mode] = session_set(d, gpu=mode != "cpu")
    res["nsys"] = {f.name: nsys_total_s(f) for f in sorted((out / "nsys").glob("*.csv"))}
    return res


def nsys_total_s(p: Path) -> dict:
    """An `nsys stats` CSV's total time (s) and its five largest rows by time."""
    lines = text(p).splitlines()
    head = next((i for i, l in enumerate(lines) if l.startswith('"Time') or l.startswith("Time")), None)
    if head is None:
        return {"raw": text(p)[:500]}
    import csv
    rows = list(csv.DictReader(lines[head:]))
    total = sum(float(r.get("Total Time (ns)") or 0) for r in rows) / 1e9
    top = sorted(rows, key=lambda r: -float(r.get("Total Time (ns)") or 0))[:5]
    return {"total_s": round(total, 4), "top": [{k: r.get(k) for k in ("Name", "Operation", "Instances", "Count", "Total Time (ns)")
                                                 if r.get(k) is not None} for r in top]}


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=1, default=str))
