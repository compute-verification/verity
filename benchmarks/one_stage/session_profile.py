"""Every phase of a session's prove time under Nsight Systems, ranked in ms a session, from the same trace and records as
`phase_profile.py` (`phase_profile.sh`'s `tl` step, then its `session` step).

    python benchmarks/one_stage/session_profile.py --sqlite P.sqlite --prove-out prove.out [--skip 1] [--json out.json]

A session's `prove_total_s` is its two reps' prove seconds. Each rep's device phases (the record's buckets ``t.witness``
through ``t.ligerito``) run back to back, so each window is placed backward from Ligerito's phase mark (`phase_profile.py`'s,
at ``lf_fold_base_pair``) by the record's seconds, and split into GPU time by kernel and the host's idle time. The
rest of a rep's prove seconds has no device window and is ranked from the records alone: the host around the device
prover (``t.upstream`` less ``t.total``), M1's finish (``t.prove`` less ``t.upstream``), and what the session adds to the
rep (rep 0: level 0's hiding and the ZK server's level-0 recheck; rep 1: its pads' commitment). ``--skip`` drops the first
sessions (the warm one)."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_profile import load, marks, window

DEVICE = ("t.witness", "t.encoding_commitment", "t.link", "t.zerocheck", "t.lincheck", "t.ring_switch", "t.ligerito")


def sessions(prove_out: str) -> list[dict]:
    return [json.loads(line.split("\t", 1)[1]) for line in open(prove_out) if line.startswith("LIVE\t")]


def host_parts(r: dict) -> dict[str, list[float]]:
    """A session's prove seconds that no device window holds, a list a phase (one entry a rep)."""
    out = defaultdict(list)
    for rep, (b, s) in enumerate(zip(r["buckets"], r["prove_s"])):
        out["host around the device prover"].append(b["t.upstream"] - b["t.total"])
        out["M1 finish (replay, inner proof)"].append(b["t.prove"] - b["t.upstream"])
        out["rep 0: level 0's hiding and recheck" if rep == 0 else "rep 1: its pads' commitment"].append(s - b["t.prove"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--prove-out", required=True)
    ap.add_argument("--anchor", default="lf_fold_base_pair")
    ap.add_argument("--skip", type=int, default=1)
    ap.add_argument("--json")
    a = ap.parse_args()
    gpu, api, osrt = load(a.sqlite)
    recs = sessions(a.prove_out)
    reps = [(si, b) for si, r in enumerate(recs) for b in r["buckets"]]
    ms = marks(gpu, api, a.anchor)
    print(f"{len(recs)} sessions, {len(reps)} reps, {len(ms)} Ligerito marks ({a.anchor}), {len(gpu)} GPU ops")
    if len(ms) != len(reps):
        print("marks and reps differ in number: paired in order, the shorter list's length")
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # session -> phase -> key -> value
    for (si, b), (s_lig, tid, _) in zip(reps, ms):
        end = s_lig + int(b["t.ligerito"] * 1e9)
        for k in reversed(DEVICE):
            start = end - int(b.get(k, 0.0) * 1e9)
            if end > start:
                w = window(gpu, api, osrt, tid, start, end)
                d = per[si][k[2:]]
                for key in ("wall_ms", "gpu_ms", "idle_os_ms", "idle_api_ms", "idle_sync_ms", "idle_host_ms"):
                    d[key] += w[key]
                for f, v in w["name_ms"].items():
                    d["op:" + f] += v
            end = start
    for si, r in enumerate(recs):
        for k, v in host_parts(r).items():
            per[si][k]["wall_ms"] += 1e3 * sum(v)
        per[si]["(prove_total_s)"]["wall_ms"] = 1e3 * r["prove_total_s"]
        per[si]["(session front: witness_s)"]["wall_ms"] = 1e3 * r.get("witness_s", 0.0)
        per[si]["(session e2e_s)"]["wall_ms"] = 1e3 * r.get("e2e_s", 0.0)
    timed = [per[si] for si in range(len(recs)) if si >= a.skip]
    if not timed:
        print("no timed sessions")
        return 1
    phases = sorted({p for s in timed for p in s})
    med = {p: {k: statistics.median(s[p].get(k, 0.0) for s in timed) for k in {k for s in timed for k in s[p]}} for p in phases}
    rank = sorted((p for p in phases if not p.startswith("(")), key=lambda p: -med[p]["wall_ms"])
    total = sum(med[p]["wall_ms"] for p in rank)
    print(f"over {len(timed)} timed sessions, median ms a session; prove_total_s {med['(prove_total_s)']['wall_ms']:.1f}, "
          f"the phases below sum to {total:.1f}; front (witness_s) {med['(session front: witness_s)']['wall_ms']:.1f}, "
          f"e2e {med['(session e2e_s)']['wall_ms']:.1f}")
    print(f"  {'phase':38s} {'wall':>7s} {'gpu':>7s} | idle: {'os':>6s} {'api':>6s} {'sync':>6s} {'host':>6s} | top GPU ops (ms)")
    for p in rank:
        m = med[p]
        fams = sorted(((k[3:], v) for k, v in m.items() if k.startswith("op:")), key=lambda kv: -kv[1])[:4]
        dev = "gpu_ms" in m
        print(f"  {p:38s} {m['wall_ms']:7.1f} " + (f"{m['gpu_ms']:7.1f} | {m['idle_os_ms']:12.1f} {m['idle_api_ms']:6.1f} "
              f"{m['idle_sync_ms']:6.1f} {m['idle_host_ms']:6.1f} | " + ", ".join(f"{f} {v:.1f}" for f, v in fams)
              if dev else f"{'-':>7s} | (host; no device window)"))
    if a.json:
        json.dump({"median": med, "rank": rank, "sessions": [{p: dict(d) for p, d in s.items()} for s in timed]}, open(a.json, "w"),
                  indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
