"""Where a served step's host time goes, from a `served_util.sh PYSAMPLE=1` capture: NVTX ranges and sampled call chains
(Python frames with nsys's `--python-sampling`) by thread, over the full steps the step trace names.

    python benchmarks/one_stage/host_profile.py --sqlite PROF.sqlite --trace ADVICE_TRACE.jsonl --T 8192 --out host.json

The window runs from the step before the first full step (T tokens) in the capture to the last one. A sample counts once
for each distinct frame on its stack (inclusive) and once for its innermost Python frame (self); per-step figures divide
by the window's full steps.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import served_util as su  # noqa: E402


def has(db: sqlite3.Connection, t: str) -> bool:
    return su._has(db, t)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--T", required=True, type=int)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    db = sqlite3.connect(f"file:{a.sqlite}?mode=ro", uri=True)
    z = su.utc0_ns(db)
    klo, khi = db.execute("SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    st = su.steps(a.trace)
    idx = su.steady(st, a.T, (klo + z) / 1e9, (khi + z) / 1e9)
    if len(idx) < 1:
        raise SystemExit("no full step in the capture")
    lo, hi = int(st[idx[0] - 1]["t"] * 1e9) - z, int(st[idx[-1]]["t"] * 1e9) - z
    n = len(idx)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    tname = {tid: strings.get(nid, str(nid)) for nid, _, tid in db.execute("SELECT nameId, priority, globalTid FROM ThreadNames")}
    pname = {gp: name for gp, _, name in db.execute("SELECT globalPid, pid, name FROM PROCESSES")}
    label = lambda tid: f"{pname.get(tid & ~0xFFFFFF, '?')}/{tname.get(tid, tid & 0xFFFFFF)}"  # noqa: E731
    out: dict = {"window_s": round((hi - lo) / 1e9, 3), "full_steps": n, "step_ms": round((hi - lo) / 1e6 / n, 2)}

    nv: dict[tuple, list] = defaultdict(lambda: [0, 0.0])
    if has(db, "NVTX_EVENTS"):
        cols = su._cols(db, "NVTX_EVENTS")
        text = "COALESCE(e.text, s.value)" if "textId" in cols else "e.text"
        join = "LEFT JOIN StringIds s ON s.id = e.textId" if "textId" in cols else ""
        q = (f"SELECT {text}, e.start, e.end, e.globalTid FROM NVTX_EVENTS e {join} "
             "WHERE e.end IS NOT NULL AND e.end > ? AND e.start < ?")
        for name, s, e, tid in db.execute(q, (lo, hi)):
            k = (str(name)[:100], label(tid or 0))
            nv[k][0] += 1
            nv[k][1] += (min(e, hi) - max(s, lo)) / 1e6
    out["nvtx"] = [{"name": k[0], "thread": k[1], "count": c, "ms_per_step": round(ms / n, 2)}
                   for k, (c, ms) in sorted(nv.items(), key=lambda kv: -kv[1][1])[:a.top * 2]]

    samples = {sid: tid for sid, tid in db.execute("SELECT id, globalTid FROM COMPOSITE_EVENTS WHERE start >= ? AND start < ?", (lo, hi))}
    frames: dict[int, list] = defaultdict(list)
    for sid, sym, mod, depth in db.execute("SELECT id, symbol, module, stackDepth FROM SAMPLING_CALLCHAINS"):
        if sid in samples:
            frames[sid].append((depth, strings.get(sym, str(sym)), strings.get(mod, str(mod))))
    per_thread = Counter(label(t) for t in samples.values())
    incl: dict[str, Counter] = defaultdict(Counter)
    self_: dict[str, Counter] = defaultdict(Counter)
    py = lambda m, s: m.endswith(".py") or ".py:" in s or s.endswith(".py")  # noqa: E731
    for sid, fr in frames.items():
        th = label(samples[sid])
        fr.sort()
        names = [f"{s} @ {os.path.basename(m)}" for _, s, m in fr]
        for f in set(names):
            incl[th][f] += 1
        pys = [f"{s} @ {os.path.basename(m)}" for _, s, m in fr if py(m, s)]
        self_[th][pys[0] if pys else names[0]] += 1
    out["sample_period_note"] = "samples are taken on running threads only"
    out["threads"] = []
    for th, c in per_thread.most_common(8):
        out["threads"].append({"thread": th, "samples": c, "samples_per_step": round(c / n, 1),
                               "inclusive": [[f, v, round(v / c, 3)] for f, v in incl[th].most_common(a.top)],
                               "self": [[f, v, round(v / c, 3)] for f, v in self_[th].most_common(a.top // 2)]})
    a.out.write_text(json.dumps(out, indent=1))
    print(f"HOST window {out['window_s']} s, {n} full steps of {out['step_ms']} ms")
    for r in out["nvtx"][:a.top]:
        print(f"  nvtx {r['ms_per_step']:9.2f} ms/step  {r['count']:6d}x  {r['thread'][:40]:40s} {r['name']}")
    for t in out["threads"]:
        print(f"THREAD {t['thread']}: {t['samples']} samples ({t['samples_per_step']} a step)")
        for f, v, sh in t["inclusive"][:a.top]:
            print(f"  incl {sh:6.1%} {f[:150]}")
        for f, v, sh in t["self"][:10]:
            print(f"  self {sh:6.1%} {f[:150]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
