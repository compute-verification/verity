"""Every phase of a session's prove time under Nsight Systems, ranked in ms a session, from the same trace and records as
`phase_profile.py` (`phase_profile.sh`'s `tl` step, then its `session` step).

    python benchmarks/one_stage/session_profile.py --sqlite P.sqlite --prove-out prove.out [--skip 1] [--json out.json]
        [--timeline PHASE]   # also PHASE's GPU ops in start order, in the first timed session's first rep that has it
        [--gaps MS]          # also the prover thread's CUDA and OS calls and the GPU's ops in the MS before each timed rep's
                             # device windows and after them (the host around the device prover)

A session's `prove_total_s` is its two reps' prove seconds. Each rep's device phases (the record's buckets ``t.witness``
through ``t.ligerito``) run back to back, so each window is placed backward from Ligerito's phase mark (`phase_profile.py`'s,
at ``lf_fold_base_pair``) by the record's seconds, and split into GPU time by kernel and the host's idle time. The
rest of a rep's prove seconds has no device window and is ranked from the records alone: the host around the device
prover (``t.upstream`` less ``t.total``), Flock ZK's finish (``t.prove`` less ``t.upstream``), and what the session adds to the
rep (rep 0: Flock ZK's preparation past the witness build, where it runs beside it, else level 0's hiding and the ZK server's
level-0 recheck; rep 1: the rep's time outside its device call, its wait on rep 0's finish included). ``--skip`` drops the first
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
        out["Flock ZK finish (replay, inner proof)"].append(b["t.prove"] - b["t.upstream"])
        out["rep 0: Flock ZK's preparation past the witness" if rep == 0 else "rep 1: outside its device call"].append(s - b["t.prove"])
    return out


def timeline(gpu, api, span, phase: str) -> None:
    """The phase window's GPU ops in start order, a run of one name as one line: its offset from the window's start, count,
    summed and spanned time, bytes, and the runtime call that issued it."""
    if span is None:
        print(f"timeline: no timed {phase} window")
        return
    a, b = span
    call = {c: n for _, _, n, c, _ in api}
    rows = []
    for s, e, n, c, kind, by in gpu:
        if s < b and e > a:
            name = n.removeprefix("void ").split("<")[0].split("(")[0]
            via = call.get(c, "?").split("_v")[0]
            if rows and rows[-1][2] == name and rows[-1][6] == via:
                r = rows[-1]
                rows[-1] = (r[0], max(r[1], e), name, r[3] + 1, r[4] + (e - s), r[5] + by, via)
            else:
                rows.append((s, e, name, 1, e - s, by, via))
    print(f"timeline of {phase}: {(b - a) / 1e6:.2f} ms window, {len(rows)} runs of ops")
    print(f"  {'at ms':>8s} {'span':>7s} {'busy':>7s} {'n':>4s} {'MB':>8s}  op (via)")
    for s, e, name, n, busy, by, via in rows:
        mb = f"{by / 1e6:8.1f}" if by else f"{'':8s}"
        print(f"  {(s - a) / 1e6:8.2f} {(e - s) / 1e6:7.2f} {busy / 1e6:7.2f} {n:4d} {mb}  {name} ({via})")


def gaps(gpu, api, osrt, edges, span: int) -> None:
    """The prover thread's CUDA and OS calls and the GPU's ops in the `span` ns before each rep's first device window and after
    its Ligerito, summed over the reps (ms a rep), the largest first."""
    for side in ("before", "after"):
        calls, os_calls, ops = defaultdict(float), defaultdict(float), defaultdict(float)
        for _, start, end, tid in edges:
            a, b = (start - span, start) if side == "before" else (end, end + span)
            for s, e, n, _, t in api:
                if t == tid and s < b and e > a:
                    calls[n.split("_v")[0]] += (min(e, b) - max(s, a)) / 1e6
            for s, e, n, t in osrt:
                if t == tid and s < b and e > a:
                    os_calls[n] += (min(e, b) - max(s, a)) / 1e6
            for s, e, n, _, _, _ in gpu:
                if s < b and e > a:
                    ops[n.removeprefix("void ").split("<")[0].split("(")[0]] += (min(e, b) - max(s, a)) / 1e6
        k = max(len(edges), 1)
        print(f"{side} the device windows ({span / 1e6:.0f} ms, {len(edges)} reps), ms a rep:")
        print("  CUDA calls: " + ", ".join(f"{n} {v / k:.2f}" for n, v in sorted(calls.items(), key=lambda kv: -kv[1])[:12]))
        print("  OS calls: " + ", ".join(f"{n} {v / k:.2f}" for n, v in sorted(os_calls.items(), key=lambda kv: -kv[1])[:12]))
        print("  GPU ops: " + ", ".join(f"{n} {v / k:.2f}" for n, v in sorted(ops.items(), key=lambda kv: -kv[1])[:10]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--prove-out", required=True)
    ap.add_argument("--anchor", default="lf_fold_base_pair")
    ap.add_argument("--skip", type=int, default=1)
    ap.add_argument("--json")
    ap.add_argument("--timeline", help="a device phase (encoding_commitment, zerocheck, ...)")
    ap.add_argument("--gaps", type=float)
    a = ap.parse_args()
    gpu, api, osrt = load(a.sqlite)
    recs = sessions(a.prove_out)
    reps = [(si, b) for si, r in enumerate(recs) for b in r["buckets"]]
    ms = marks(gpu, api, a.anchor)
    print(f"{len(recs)} sessions, {len(reps)} reps, {len(ms)} Ligerito marks ({a.anchor}), {len(gpu)} GPU ops")
    if len(ms) != len(reps):
        print("marks and reps differ in number: paired in order, the shorter list's length")
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # session -> phase -> key -> value
    spans = []  # (session, phase, start, end)
    edges = []  # (session, rep's first device window's start, its Ligerito's end, the prover thread)
    for (si, b), (s_lig, tid, _) in zip(reps, ms):
        end = s_lig + int(b["t.ligerito"] * 1e9)
        for k in reversed(DEVICE):
            start = end - int(b.get(k, 0.0) * 1e9)
            if end > start:
                spans.append((si, k[2:], start, end))
                w = window(gpu, api, osrt, tid, start, end)
                d = per[si][k[2:]]
                for key in ("wall_ms", "gpu_ms", "idle_os_ms", "idle_api_ms", "idle_sync_ms", "idle_host_ms"):
                    d[key] += w[key]
                for f, v in w["name_ms"].items():
                    d["op:" + f] += v
            end = start
        edges.append((si, end, s_lig + int(b["t.ligerito"] * 1e9), tid))
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
    if a.gaps:
        gaps(gpu, api, osrt, [e for e in edges if e[0] >= a.skip], int(a.gaps * 1e6))
    if a.timeline:
        timeline(gpu, api, next(((s, e) for si, p, s, e in spans if p == a.timeline and si >= a.skip), None), a.timeline)
    if a.json:
        json.dump({"median": med, "rank": rank, "sessions": [{p: dict(d) for p, d in s.items()} for s in timed]}, open(a.json, "w"),
                  indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
