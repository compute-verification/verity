"""One prover phase's time under Nsight Systems, from a `nsys export --type sqlite` of a warm `flock-circuit prove --zk --gpu`
process (`phase_profile.sh`'s `tl` step), split into what the GPU ran and what the host did while it waited.

    python benchmarks/one_stage/phase_profile.py --sqlite P.sqlite --prove-out prove.out [--phase t.ligerito]
        [--anchor lf_fold_base_pair] [--skip 2] [--json out.json]

A window is one rep's phase: it opens where the prover's thread returns from the last `cudaStreamSynchronize` before the
anchor kernel's launch (the phase mark's own synchronize) and lasts the rep's `--phase` seconds from the session records in
`--prove-out`, in order; `--skip` drops the first windows (the warm session's reps). Each window is cut at the thread's
synchronizes (the prover's transcript reads, `hc_wait`) into segments, and each segment's wall is split into GPU time by
kernel family, the host blocked in the synchronize while the GPU ran, the host's OS calls with the GPU idle (a live coin's
round trip to the verifier), its other CUDA calls, and the rest, host compute.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
from collections import defaultdict

FAMILIES = (
    ("fold", ("lf_fold",)),
    ("message", ("lf_msg", "lf_presplit_msg", "sumcheck_message", "combine_sumcheck")),
    ("ood/glue", ("basis_message", "combine_basis", "glue", "build_eq", "eq_", "lf_split_basis")),
    ("ntt", ("ntt", "lf_replicate")),
    ("merkle", ("sha512", "hm96", "merkle", "hash_")),
    ("open", ("gather", "open", "path", "salt", "scatter_query", "clear_field")),
    ("zk", ("lf_zk",)),
    ("hc_copy", ("hc_copy",)),
)
BLOCKING = re.compile(r"recv|read|poll|select|futex|pthread_cond|sem_wait|epoll|accept|write|send|sleep|sched_yield|pthread_mutex|pthread_join")


def family(name: str) -> str:
    n = name.lower()
    for f, keys in FAMILIES:
        if any(k in n for k in keys):
            return f
    return "other"


def tables(db) -> set[str]:
    return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def load(path: str):
    db = sqlite3.connect(path)
    have = tables(db)
    s = dict(db.execute("SELECT id, value FROM StringIds"))
    kcol = "demangledName" if "demangledName" in {r[1] for r in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")} else "shortName"
    gpu = [(a, b, s.get(n, "?"), c, "kernel", 0) for a, b, n, c in
           db.execute(f"SELECT start, end, {kcol}, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL")]
    kinds = {1: "memcpy HtoD", 2: "memcpy DtoH", 8: "memcpy DtoD", 10: "memcpy PtoP"}
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in have:
        gpu += [(a, b, kinds.get(k, f"memcpy kind {k}"), c, "memcpy", by) for a, b, k, c, by in
                db.execute("SELECT start, end, copyKind, correlationId, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY")]
    if "CUPTI_ACTIVITY_KIND_MEMSET" in have:
        gpu += [(a, b, "memset", c, "memset", by) for a, b, c, by in
                db.execute("SELECT start, end, correlationId, bytes FROM CUPTI_ACTIVITY_KIND_MEMSET")]
    gpu.sort()
    api = [(a, b, s.get(n, "?"), c, t) for a, b, n, c, t in
           db.execute("SELECT start, end, nameId, correlationId, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start")]
    osrt = []
    if "OSRT_API" in have:
        osrt = [(a, b, s.get(n, "?"), t) for a, b, n, t in db.execute("SELECT start, end, nameId, globalTid FROM OSRT_API ORDER BY start")]
    return gpu, api, osrt


def union(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def overlap(iv, a, b):
    return sum(max(0, min(y, b) - max(x, a)) for x, y in iv)


def complement(busy, a, b):
    gaps, at = [], a
    for x, y in busy:
        if y <= a or x >= b:
            continue
        if x > at:
            gaps.append((at, min(x, b)))
        at = max(at, y)
    if at < b:
        gaps.append((at, b))
    return gaps


def phase_seconds(prove_out: str, key: str) -> list[float]:
    """Each rep's ``key`` seconds, in order: the prover's `LIVE` session records, one bucket a rep."""
    vals = []
    for line in open(prove_out):
        if line.startswith("LIVE\t"):
            vals += [float(b[key]) for b in json.loads(line.split("\t", 1)[1]).get("buckets", []) if key in b]
    return vals


def window(gpu, api, osrt, tid, a, b):
    """One window's segments and totals; times in ns."""
    g = [x for x in gpu if x[1] > a and x[0] < b]
    busy = union([(max(x[0], a), min(x[1], b)) for x in g])
    gaps = complement(busy, a, b)
    th = [x for x in api if x[4] == tid and x[1] > a and x[0] < b]
    is_sync = lambda x: x[2].startswith(("cudaStreamSynchronize", "cudaDeviceSynchronize"))
    syncs = [x for x in th if is_sync(x)]
    other_api = [(x[0], x[1]) for x in th if not is_sync(x)]
    os_iv = [(x[0], x[1]) for x in osrt if x[3] == tid and x[1] > a and x[0] < b and BLOCKING.search(x[2])]
    os_names = defaultdict(int)
    for x in osrt:
        if x[3] == tid and x[1] > a and x[0] < b and BLOCKING.search(x[2]):
            os_names[x[2]] += 1
    os_u, api_u, sync_u = union(os_iv), union(other_api), union([(x[0], x[1]) for x in syncs])
    cuts = sorted({min(max(x[1], a), b) for x in syncs} | {a, b})
    segs = []
    for s0, s1 in zip(cuts, cuts[1:]):
        if s1 <= s0:
            continue
        seg_busy = overlap(busy, s0, s1)
        seg_gaps = [(max(x, s0), min(y, s1)) for x, y in gaps if y > s0 and x < s1]
        idle = sum(y - x for x, y in seg_gaps)
        os_idle = sum(overlap(os_u, x, y) for x, y in seg_gaps)
        api_idle = sum(overlap(api_u, x, y) for x, y in seg_gaps)
        sync_idle = sum(overlap(sync_u, x, y) for x, y in seg_gaps)
        ks = defaultdict(lambda: [0, 0])
        for x in g:
            d = overlap([(x[0], x[1])], s0, s1)
            if d:
                ks[x[2] if x[4] != "kernel" else re.sub(r"[<(].*", "", x[2])][0] += d
                ks[x[2] if x[4] != "kernel" else re.sub(r"[<(].*", "", x[2])][1] += 1
        top = sorted(ks.items(), key=lambda kv: -kv[1][0])[:3]
        segs.append({"t_us": round((s0 - a) / 1e3, 1), "wall_us": round((s1 - s0) / 1e3, 1), "gpu_us": round(seg_busy / 1e3, 1),
                     "idle_os_us": round(os_idle / 1e3, 1), "idle_api_us": round(api_idle / 1e3, 1),
                     "idle_sync_us": round(sync_idle / 1e3, 1), "idle_host_us": round((idle - os_idle - api_idle - sync_idle) / 1e3, 1),
                     "top": [(n, round(v[0] / 1e3, 1), v[1]) for n, v in top]})
    fam = defaultdict(float)
    cnt = defaultdict(int)
    by = defaultdict(int)
    for x in g:
        d = overlap([(x[0], x[1])], a, b)
        f = family(x[2]) if x[4] == "kernel" else x[2]
        fam[f] += d
        cnt[f] += 1
        if x[4] != "kernel":
            by[x[2]] += x[5]
    idle = sum(y - x for x, y in gaps)
    os_idle = sum(overlap(os_u, x, y) for x, y in gaps)
    api_idle = sum(overlap(api_u, x, y) for x, y in gaps)
    sync_idle = sum(overlap(sync_u, x, y) for x, y in gaps)
    return {"wall_ms": (b - a) / 1e6, "gpu_ms": sum(y - x for x, y in busy) / 1e6, "idle_ms": idle / 1e6,
            "idle_os_ms": os_idle / 1e6, "idle_api_ms": api_idle / 1e6, "idle_sync_ms": sync_idle / 1e6,
            "idle_host_ms": (idle - os_idle - api_idle - sync_idle) / 1e6, "syncs": len(syncs),
            "launches": sum(1 for x in th if "Launch" in x[2]), "os_calls": dict(os_names),
            "family_ms": {k: v / 1e6 for k, v in sorted(fam.items(), key=lambda kv: -kv[1])}, "family_n": dict(cnt),
            "copy_bytes": dict(by), "segments": segs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--prove-out", required=True)
    ap.add_argument("--phase", default="t.ligerito")
    ap.add_argument("--anchor", default="lf_fold_base_pair")
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--json")
    ap.add_argument("--segments", type=int, default=1, help="print the segments of this many windows")
    a = ap.parse_args()
    gpu, api, osrt = load(a.sqlite)
    secs = phase_seconds(a.prove_out, a.phase)
    corr = {x[3]: x for x in api}
    anchors = [corr[x[3]] for x in gpu if x[4] == "kernel" and a.anchor in x[2] and x[3] in corr]
    print(f"{len(anchors)} anchors ({a.anchor}), {len(secs)} {a.phase} records, {len(gpu)} GPU ops, {len(api)} CUDA calls, "
          f"{len(osrt)} OS calls")
    if len(anchors) != len(secs):
        print("anchors and records differ in number: windows paired in order, the shorter list's length")
    wins = []
    for i, (L, sec) in enumerate(zip(anchors, secs)):
        tid = L[4]
        before = [x[1] for x in api if x[4] == tid and x[1] <= L[0] and x[2].startswith("cudaStreamSynchronize")]
        if not before:
            continue
        s = max(before)
        e = s + int(sec * 1e9)
        after = [x[1] for x in api if x[4] == tid and x[2].startswith("cudaStreamSynchronize") and abs(x[1] - e) < 2_000_000]
        w = window(gpu, api, osrt, tid, s, e)
        w["rep"] = i
        w["end_vs_sync_us"] = round(min((x - e for x in after), key=abs) / 1e3, 1) if after else None
        wins.append(w)
    timed = wins[a.skip:]
    if not timed:
        print("no timed windows")
        return 1
    keys = ("wall_ms", "gpu_ms", "idle_ms", "idle_os_ms", "idle_api_ms", "idle_sync_ms", "idle_host_ms")
    med = {k: statistics.median(w[k] for w in timed) for k in keys}
    print(f"{a.phase} over {len(timed)} timed reps, median ms a rep: " + ", ".join(f"{k[:-3]} {v:.2f}" for k, v in med.items()))
    print(f"  syncs a rep {statistics.median(w['syncs'] for w in timed)}, launches {statistics.median(w['launches'] for w in timed)}, "
          f"window end against the nearest synchronize: {[w['end_vs_sync_us'] for w in timed]} us")
    fams = sorted({f for w in timed for f in w["family_ms"]}, key=lambda f: -statistics.median(w["family_ms"].get(f, 0) for w in timed))
    for f in fams:
        print(f"  {f:16s} {statistics.median(w['family_ms'].get(f, 0) for w in timed):8.3f} ms  "
              f"{statistics.median(w['family_n'].get(f, 0) for w in timed):6.0f} ops")
    print(f"  OS calls with the thread blocked (first timed rep): {timed[0]['os_calls']}")
    print(f"  copy bytes (first timed rep): {timed[0]['copy_bytes']}")
    for w in timed[:a.segments]:
        print(f"segments of rep {w['rep']}: t_us wall gpu | GPU idle: os api sync host | top GPU ops (name, us, n)")
        for sg in w["segments"]:
            print(f"  {sg['t_us']:9.1f} {sg['wall_us']:8.1f} {sg['gpu_us']:8.1f} | {sg['idle_os_us']:7.1f} {sg['idle_api_us']:7.1f} "
                  f"{sg['idle_sync_us']:7.1f} {sg['idle_host_us']:7.1f} | {sg['top']}")
    if a.json:
        json.dump({"phase": a.phase, "median": med, "windows": wins}, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
