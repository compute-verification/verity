"""Configuration 2, recursive zero knowledge: what it is to report, and what of it runs today.

The design: workers prove the inner statement natively with ZK off and run the
outer proof's arithmetic unmasked; a ZK server, compiled Lean on a CPU node, draws the masks and salts, checks the outer
relations and releases the masked proof; the auditor runs the Lean verifier on what is released. The worker/server split
doesn't exist yet. What runs today, and what the ZK point (``job.sh ZK=1``, ``zk-point``) measures as its stand-in:

- ``flock-circuit prove --zk --gpu``: Flock's masked protocol (M1, ``verity/ml/flock/live/PROTOCOL.md``), the device
  proving and the host masking, with the session's B units sampled by the stratified law as in configuration 1;
- the ZK server (``FC_FIREWALL=1``, PROTOCOL §10): the Lean (``verity/core/service/flock/Flock/Firewall``, the
  ``flock-firewall`` processes the prover starts on its node), which commits level 0 under its own salts and computes the
  opening, the inner proof and the proof file, and ``flock-circuit``'s checks (the skeleton, the end check) on the
  prover's host threads. That is the ZK server's work today (the draft's §3), so the server's CPU is the prover process's
  CPU with the ZK server on less the same sessions' with it off, the ``flock-firewall`` processes counted once the prover
  has waited for them. The ZK server rechecks a witness the host assembled
  (``FC_HOST_WITNESS=1``, PROTOCOL §10.2), so both arms take the host's witness.

Each field names its source: ``config1`` is configuration 1 at the same B, ``stand-in`` the ZK point above.
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Any

FIELDS: tuple[tuple[str, str, str, str], ...] = (
    ("worker.vu_per_second", "VU/s", "config1",
     "the worker's inner proof is configuration 1's prover at best B; the unmasked outer arithmetic is not built, so this "
     "is an upper bound"),
    ("zk_server.cpu_s_per_session", "s", "stand-in",
     "prover.cpu_s_per_session with FC_FIREWALL=1 less with FC_FIREWALL=0, both --zk on the host's witness at the same B: "
     "the firewall's send check"),
    ("zk_server.cpu_s_per_vu", "s", "stand-in", "zk_server.cpu_s_per_session / B"),
    ("net.bytes_worker_to_server", "B", "stand-in",
     "the --zk session's prover-to-verifier bytes (net.bytes_out): the outer messages the worker would hand the server"),
    ("proof_bytes.released", "B", "stand-in", "the --zk session's proof, both reps, as the firewall lets it out"),
    ("zk_server.cores_per_gpu", "ratio", "stand-in",
     "zk_server.cpu_s_per_session / the --zk session's t.total: cores that keep up with one GPU; servers = that / cores each"),
)

STANDIN = ("today's --zk path under the ZK server on the prover's node (FC_FIREWALL=1: flock-circuit's checks and the Lean "
           "flock-firewall processes it starts), not the worker/server split: the masks and salts are drawn on the prover's "
           "node, the witness is the host's (FC_HOST_WITNESS=1), the server checks the device's values on the same node")


def describe() -> str:
    lines = [__doc__.strip(), "", "At configuration 1's best B:"]
    for name, unit, source, how in FIELDS:
        lines.append(f"  {name:<30} [{unit}] {source:<8} {how}")
    return "\n".join(lines)


def _jsonl(p: Path, prefix: str = "") -> list[dict[str, Any]]:
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines():
        if prefix and not ln.startswith(prefix):
            continue
        try:
            v = json.loads(ln[len(prefix):].lstrip("\t"))
        except ValueError:
            continue
        if isinstance(v, dict):
            out.append(v)
    return out


def _firewall(session: dict[str, Any]) -> dict[str, float]:
    """One session's ZK server times, summed over its reps' buckets."""
    got: dict[str, float] = {}
    for b in session.get("buckets") or []:
        for k, v in (b or {}).items():
            if (k.startswith("t.firewall") or k.startswith("cpu.firewall")) and isinstance(v, (int, float)):
                got[k] = got.get(k, 0.0) + float(v)
    return got


def point(d: Path, workload: Any, B: int, firewall: bool, run_id: str | None) -> dict[str, Any]:
    """The ZK point's ``research/result/v0.1`` document from the prover's records in ``d`` (results.jsonl, prove.out)."""
    rec = next((r for r in reversed(_jsonl(d / "results.jsonl")) if r.get("live")), None) or {}
    lv = rec.get("live") or {}
    sessions = _jsonl(d / "prove.out", "LIVE")
    timed = [s for s in sessions if not s.get("warm")] or sessions
    ms: list[dict[str, Any]] = []

    def m(name: str, value: Any, unit: str) -> None:
        if isinstance(value, (int, float)):
            ms.append({"name": name, "value": value, "unit": unit})

    t_total = lv.get("prove_total_s")
    m("t.total", t_total, "s")
    if t_total:
        m("vu.per_second", B / t_total, "VU/s")
    m("vu.count", B, "count")
    cpu, n = lv.get("prover_cpu_s"), len(sessions)
    m("prover.cpu_s", cpu, "s")
    if cpu is not None and n:
        m("prover.cpu_s_per_session", cpu / n, "s")
    fw = [_firewall(s) for s in timed]
    for k in sorted({k for f in fw for k in f}):
        m(k.replace("t.firewall", "firewall.t").replace("cpu.firewall", "firewall.cpu") + "_s",
          statistics.median(f.get(k, 0.0) for f in fw), "s")
    m("net.bytes_out", lv.get("wire_sent"), "B")
    m("net.bytes_in", lv.get("wire_recv"), "B")
    pb = lv.get("proof_bytes")
    m("proof_bytes", sum(pb) if isinstance(pb, list) else pb, "B")
    if lv.get("prover_peak_gb"):
        m("mem.peak_host_bytes", int(lv["prover_peak_gb"] * 2**30), "B")

    zk = bool(lv.get("zk")) and all((s.get("identity") or {}).get("hooks", {}).get("zk") is True for s in sessions)
    broke = ("no session record" if not lv else
             f"not accepted live: {lv.get('why')}" if lv.get("accepted") is not True else
             "the sessions were not --zk" if not zk else
             "FC_FIREWALL=1 but no firewall times in the sessions" if firewall and not any(fw) else None)
    return {
        "schema": "research/result/v0.1", "run_id": run_id,
        "workload_fingerprint": {"kind": "proof-throughput/zk-standin/v1", "workload": workload.name, "vu": workload.vu,
                                 "K": workload.K, "B": B, "zk": True, "firewall": firewall, "standin": STANDIN,
                                 "draw": {"law": "stratified", "k": B, "population": workload.n_max}},
        "validation": {"status": "failed" if broke else "passed",
                       "checker": "flock-circuit serve --zk (live, loopback) on every session; the Lean verifier not run",
                       "detail": broke or f"{len(timed)} timed and the warm --zk sessions accepted live"
                                           + (", under the firewall's send check" if firewall else ""),
                       "evidence": [p for p in ("out/results.jsonl", "out/prove.out", "out/prove.err") if (d / Path(p).name).exists()]},
        "measurements": ms,
        "measurement_files": [p for p in ("out/results.jsonl", "out/prove.out", "out/prove.err") if (d / Path(p).name).exists()],
    }


def firewall_on() -> bool:
    return os.environ.get("FC_FIREWALL") == "1"


__all__ = ["FIELDS", "STANDIN", "describe", "firewall_on", "point"]
