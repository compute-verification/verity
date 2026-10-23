"""A served decode step by part, from an sbench `prof` capture under DITHER_NSYS=prof (nsys SQLite export).  As
exactlin/steps.py: the step is the difference of the arm's two NVTX ranges (PROF_RANGE/<arm>/n<N1> and n<N2>, the same 32
requests generating N1 and then N2 tokens) over N2 - N1, only the pushing process's kernels and memory ops count, and each
gets a part.  Here a kernel's part is first the window phase (dither_one.PHASE_RANGE/<phase>) its launch call sat in on
the host thread (CUPTI's correlation id joins the launch to the kernel; a graph's nodes join to its cudaGraphLaunch,
outside every phase, so they are the forward's), then, in the forward, the part its name matches (PARTS, else
steps.CATEGORIES).  Each phase's host wall time a step is reported beside the parts, and the step's GPU idle time
(wall minus busy): where the GPU waits on the host.  A dither-a arm runs on several streams at once, so parts sum past the
wall; each stream's busy time a step and its parts are reported too, and the serving stream's busy time short of the wall
is how long it waits on the others and the host.

The units kernels carry the noise b, c and U, the MMAs, the receipts, the peel and y in one launch, so they are one part;
the peel's own cost is the peeled form's kernel time less the unpeeled form's (a kernel-level row).  The old quantizer
(`int8_exact.quantize_rows`, eager torch ops inside the graph) is most of the forward's at::native kernels; an -xl arm's
is exactlin_quant_rows.

    python stepprof.py gate.sqlite > stepprof.json
    python stepprof.py --table stepprof.json        # the parts as a text table (ms a step and share of the step's wall)
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from bisect import bisect_right
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "exactlin"))
import steps as ST

PHASE_RANGE = "verity/dither/phase/"
PHASE_PARTS = {"before": "A step (h1c_a_step) and the step's inputs", "admit_tables": "admit tables",
               "z_tables": "z tables", "step_trees": "round and step trees", "receipt_trees": "receipt trees",
               "root_read": "root read", "record_host": "record (host)", "forward_wait": "forward wait",
               "stash": "ring copies (A slots; under -ov the records)", "ring_wait": "-ov: the host's wait on the rings",
               "ov_leaf_tables": "-ov round: admit, z/y and served-y tables", "ov_round": "-ov round: step trees and round",
               "ov_units_dispatch": "-ov units (seeds, units, commit, receipt trees)"}
# dither_so -ov's sender and dispatcher threads: their launches run beside the next forward, not between steps
BESIDE = "ov_"
PARTS = [
    ("A table (h1c_a_call)", re.compile(r"h1c_a_call[23]?_k")),
    ("A step (h1c_a_step) and the step's inputs", re.compile(r"h1c_domains|h1c_a_pre_k|h1c_a_init_k")),
    ("units: seeds", re.compile(r"h1r_seeds|h1_seeds")),
    ("units: noise (separate launch)", re.compile(r"h1s_noise")),
    ("units (noise, MMAs, receipts, peel, z, y)", re.compile(r"h1o_units|h1r_units|\bh1[ukp]\b|\bh1um\b|kh_")),
    ("tree levels (h1c)", re.compile(r"h1c_leaves|h1c_segs|h1c_level")),
    ("quantizer (exactlin)", re.compile(r"exactlin_quant_rows")),
    ("GEMM (exactlin)", re.compile(r"exactlin_scaled_mm")),
    ("GEMM with the z/y epilogue (dithera_mm)", re.compile(r"dithera_mm")),
    ("receipt commit and fault (dithera_commit, dithera_fault)", re.compile(r"dithera_(commit|fault)")),
]


def part(name: str, phase: str | None) -> str:
    if phase is not None:
        return ("beside: " if phase.startswith(BESIDE) else "between: ") + PHASE_PARTS.get(phase, phase)
    for p, rx in PARTS:
        if rx.search(name):
            return p
    c = ST.category(name)
    return "quantizer (int8_exact, torch ops) and other torch ops: " + c if c.startswith("torch") else c


def analyse(path: str) -> dict:
    db = sqlite3.connect(path)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    gens: dict[str, dict[int, tuple[int, int, int]]] = {}
    phases: dict[int, list[tuple[int, int, str]]] = {}
    for start, end, text, tid, text_id in db.execute(
            "SELECT start, end, text, globalTid, textId FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        name = text if text is not None else strings.get(text_id)
        if not name:
            continue
        if name.startswith(ST.PROF_RANGE):
            arm, n = name.removeprefix(ST.PROF_RANGE).rsplit("/n", 1)
            gens.setdefault(arm, {})[int(n)] = (int(start), int(end), ST._pid(tid))
        elif name.startswith(PHASE_RANGE):
            phases.setdefault(int(tid), []).append((int(start), int(end), name.removeprefix(PHASE_RANGE)))
    starts = {tid: [s for s, _, _ in sorted(v)] for tid, v in phases.items()}
    phases = {tid: sorted(v) for tid, v in phases.items()}

    def phase_at(tid: int, t: int) -> str | None:
        v = phases.get(tid)
        if not v:
            return None
        i = bisect_right(starts[tid], t) - 1
        return v[i][2] if i >= 0 and v[i][0] <= t <= v[i][1] else None

    api: dict[tuple[int, int], tuple[int, int]] = {}
    for table in ("CUPTI_ACTIVITY_KIND_DRIVER", "CUPTI_ACTIVITY_KIND_RUNTIME"):
        try:
            api.update({(ST._pid(tid), int(c)): (int(tid), int(s)) for c, tid, s in db.execute(
                f"SELECT correlationId, globalTid, start FROM {table}")})
        except sqlite3.Error:
            pass
    ops = [(strings.get(n, str(n)), int(p), int(s), int(e), int(c), int(q)) for n, p, s, e, c, q in db.execute(
        "SELECT demangledName, globalPid, start, end, correlationId, streamId FROM CUPTI_ACTIVITY_KIND_KERNEL")]
    for table, label in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
        try:
            ops += [(f"[{label}]", int(p), int(s), int(e), int(c), int(q)) for p, s, e, c, q in db.execute(
                f"SELECT globalPid, start, end, correlationId, streamId FROM {table}")]
        except sqlite3.Error:
            pass
    doc: dict = {"arms": {}}
    for arm, by_n in gens.items():
        if len(by_n) != 2:
            doc["arms"][arm] = {"error": f"ranges {sorted(by_n)}"}
            continue
        (n1, r1), (n2, r2) = sorted(by_n.items())
        per: dict[int, dict] = {}
        for n, (start, end, pid) in ((n1, r1), (n2, r2)):
            rows: dict[tuple[str, str], list[int]] = {}
            iv = []
            by_stream: dict[int, list] = {}
            for name, gpid, s, e, corr, q in ops:
                if ST._pid(gpid) != pid or s < start or e > end:
                    continue
                tid, t = api.get((pid, corr), (None, None))
                key = (name, part(name, phase_at(tid, t) if tid is not None else None))
                r = rows.setdefault(key, [0, 0])
                r[0] += 1
                r[1] += e - s
                iv.append((s, e))
                sq = by_stream.setdefault(q, [[], {}])
                sq[0].append((s, e))
                sq[1][key[1]] = sq[1].get(key[1], 0) + e - s
            host: dict[str, int] = {}
            for v in phases.values():
                for s, e, ph in v:
                    if start <= s and e <= end:
                        host[ph] = host.get(ph, 0) + e - s
            per[n] = {"rows": rows, "busy": ST._busy(iv), "wall": end - start, "host": host,
                      "streams": {q: (ST._busy(v[0]), v[1]) for q, v in by_stream.items()}}
        steps = n2 - n1
        wall = (per[n2]["wall"] - per[n1]["wall"]) / steps
        kernels = []
        for key in set(per[n1]["rows"]) | set(per[n2]["rows"]):
            c1, t1 = per[n1]["rows"].get(key, [0, 0])
            c2, t2 = per[n2]["rows"].get(key, [0, 0])
            dc, dt = (c2 - c1) / steps, (t2 - t1) / steps
            if dc or dt:
                kernels.append({"name": ST.short(key[0]), "part": key[1], "launches_per_step": round(dc, 3),
                                "us_per_step": round(dt / 1e3, 2), "us_per_launch": round(dt / 1e3 / dc, 3) if dc else None})
        kernels.sort(key=lambda r: -r["us_per_step"])
        parts: dict[str, dict] = {}
        for r in kernels:
            p = parts.setdefault(r["part"], {"launches_per_step": 0.0, "us_per_step": 0.0})
            p["launches_per_step"] = round(p["launches_per_step"] + r["launches_per_step"], 3)
            p["us_per_step"] = round(p["us_per_step"] + r["us_per_step"], 2)
        for p in parts.values():
            p["share_of_wall"] = round(p["us_per_step"] * 1e3 / wall, 4) if wall else None
        busy = (per[n2]["busy"] - per[n1]["busy"]) / steps
        host = {ph: round((per[n2]["host"].get(ph, 0) - per[n1]["host"].get(ph, 0)) / steps / 1e6, 4)
                for ph in set(per[n1]["host"]) | set(per[n2]["host"])}
        streams = {}
        for q in set(per[n1]["streams"]) | set(per[n2]["streams"]):
            b1, p1 = per[n1]["streams"].get(q, (0, {}))
            b2, p2 = per[n2]["streams"].get(q, (0, {}))
            ps = {p: round((p2.get(p, 0) - p1.get(p, 0)) / steps / 1e6, 4) for p in set(p1) | set(p2)}
            if b2 - b1 > 0:
                streams[str(q)] = {"busy_ms_per_step": round((b2 - b1) / steps / 1e6, 4),
                                   "parts": dict(sorted(((p, v) for p, v in ps.items() if v > 0), key=lambda kv: -kv[1]))}
        doc["arms"][arm] = {
            "n": [n1, n2], "decode_steps": steps, "wall_ms_per_step": round(wall / 1e6, 4),
            "gpu_busy_ms_per_step": round(busy / 1e6, 4), "gpu_idle_ms_per_step": round((wall - busy) / 1e6, 4),
            "kernel_ms_per_step": round(sum(r["us_per_step"] for r in kernels) / 1e3, 4),
            "parts": dict(sorted(parts.items(), key=lambda kv: -kv[1]["us_per_step"])),
            "phase_host_ms_per_step": dict(sorted(host.items(), key=lambda kv: -kv[1])),
            "streams": dict(sorted(streams.items(), key=lambda kv: -kv[1]["busy_ms_per_step"])),
            "kernels": kernels}
    return doc


def table(doc: dict) -> str:
    out = []
    for arm, a in doc["arms"].items():
        if "error" in a:
            out.append(f"{arm}: {a['error']}")
            continue
        out.append(f"{arm}: wall {a['wall_ms_per_step']:.3f} ms/step, GPU busy {a['gpu_busy_ms_per_step']:.3f}, "
                   f"idle {a['gpu_idle_ms_per_step']:.3f}")
        for p, r in a["parts"].items():
            out.append(f"  {r['us_per_step'] / 1e3:8.3f} ms  {100 * (r['share_of_wall'] or 0):5.1f}%  "
                       f"{r['launches_per_step']:8.1f} launches  {p}")
        out.append("  host wall by phase (ms/step): " + ", ".join(f"{k} {v:.3f}" for k, v in a["phase_host_ms_per_step"].items()))
        streams = a.get("streams", {})
        big = {q: s for q, s in streams.items() if s["busy_ms_per_step"] >= 0.01 * a["wall_ms_per_step"]}
        for q, s in big.items():
            top = ", ".join(f"{p} {v:.3f}" for p, v in list(s["parts"].items())[:4])
            out.append(f"  stream {q}: busy {s['busy_ms_per_step']:.3f} ms/step ({top})")
        if len(streams) > len(big):
            rest = sum(s["busy_ms_per_step"] for q, s in streams.items() if q not in big)
            out.append(f"  {len(streams) - len(big)} more streams under 1% of the wall each: {rest:.3f} ms/step in all")
    return "\n".join(out)


if __name__ == "__main__":
    if sys.argv[1] == "--table":
        print(table(json.loads(Path(sys.argv[2]).read_text())))
    else:
        print(json.dumps(analyse(sys.argv[1]), indent=1))
