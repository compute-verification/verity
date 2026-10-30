"""A point's ``result.json``: the campaign contract's (``verity_numerical.bench.contract``), used as it is.

One point is sessions of one workload that each prove the B coordinates a loopback ``flock-circuit serve`` draws from the
workload's population by the stratified law (``instances``), proved by Flock's native prover with live coins, warm
``warm`` times and timed ``runs`` times, then each timed session verified by the compiled Lean verifier (``flock-verify
verify --stratified B``, its own law). The VU is one coordinate (K MACs, 2K FLOP), ``vu.count`` = B. The fingerprint's
``instances`` is null (the contract's names the B ids a statement proves, which a sample doesn't fix); ``population`` names the
frozen population and ``draw`` the law.

- ``t.total``: the median timed statement's prover seconds, the witness wait plus both reps (``statement_seconds``, the
  hill-climb's T_prove); its buckets are both reps' device phases. ``t.total_live`` adds the waits on the live verifier
  (the session's wall time).
- ``rate.*`` and ``overhead.vs_native_peak`` from ``contract.rate_measurements`` at the workload's profile, whose anchor's
  dense peak equals the H200's (README, "Native peak").
- ``mem.peak_host_bytes``: the prover process's peak RSS; ``mem.peak_device_bytes``: the leased GPU's framebuffer use,
  sampled every 200 ms (a lower bound on the device peak).
- ``transcript.bytes``: both directions of the session (``net.bytes_out`` the prover's, ``net.bytes_in`` the verifier's);
  ``proof_bytes`` the proof the prover sent, both reps.
- ``verifier.seconds``: the Lean verifier's wall seconds on one timed session (median), ``verify.cpu_s`` its CPU seconds,
  its setup (the drawn statement, once per distinct draw) included; the live verifier's own ``verify_s`` is
  ``verify.live_s``. ``verifier.cpu_seconds_per_vu`` and ``verifier.cores_per_gpu`` are the contract's ``verifier_rates``.
- ``fixed.setup_seconds``: staging the population (lowering, lanes, witness and public files), done once for every B.

The prover run's result passes when the live verifier accepted every session; the verifier run's, the point's result of
record, only when the Lean verifier also accepted every timed session. A session the Lean verifier refused for having no sample
alone (``DRAWLESS``) is labelled ``D12_LABEL`` and fails the point.
"""
from __future__ import annotations

import os
import statistics
import sys
from typing import Any, Mapping, Sequence

from verity_numerical.bench import contract

from . import instances as I
from .workloads import Workload

SECURITY = {"target": -100, "profile": "fast100", "reps": 2, "coins": "live-os: the verifier's OS draws, round by round"}
PREPROCESSING = ("the population is lowered and staged once for every B (fixed.setup_seconds); every timed session builds "
                 "its drawn statement and witness and proves both reps afresh")
CONFIGS = {"plain": "C-Flock native prover, ZK off (flock-circuit prove --gpu), live coins and the stratified draw of B units "
                    "from a loopback flock-circuit serve --draw-law; then the compiled Lean verifier (flock-verify verify "
                    "--coins os --stratified B) on each timed session"}


#: the prover's per-rep phases (``flock-circuit``'s ``gpu_buckets``) beyond the contract's buckets, each kept as ``prove.*_s``
PHASES = ("t.link", "t.zerocheck", "t.lincheck", "t.ring_switch", "t.ligerito", "h.setup", "h.stage", "h.decode")


def statement_seconds(lv: Mapping[str, Any]) -> float:
    """A statement's steady-state prover seconds from its median timed LIVE record: the witness wait plus both reps, or the
    pipelined witness builds' share when they are slower (``verity/ml/flock/pod/gemm_slowdown.statement_seconds``)."""
    full = (lv.get("witness_s") or 0.0) + lv["prove_total_s"]
    if lv.get("witness_prebuilt_s") is not None and lv.get("pipeline_depth"):
        full = max(full, lv["witness_prebuilt_s"] / lv["pipeline_depth"])
    return full


def _sum(reps: Sequence[Mapping[str, Any]] | None, key: str) -> float:
    return sum(r.get(key) or 0.0 for r in reps or [])


def prover_measurements(w: Workload, B: int, rec: Mapping[str, Any], gpu_peak_mib: float | None) -> list[dict[str, Any]]:
    """The prover's side of a point, from its class-sweep record (``results.jsonl``)."""
    lv = rec["live"]
    total = statement_seconds(lv)
    reps = lv.get("buckets") or []
    m = contract.measurement
    ms = [m("t.total", total), m("t.witness", _sum(reps, "t.witness")),
          m("t.encoding_commitment", _sum(reps, "t.encoding_commitment")), m("t.arithmetic", _sum(reps, "t.arithmetic")),
          m("t.serialization", _sum(reps, "h.serialize")),
          m("vu.count", B), m("vu.seconds_per_vu", total / B), m("vu.per_second", B / total),
          *contract.rate_measurements(total, B, w.K, w.profile)]
    if lv.get("e2e_s"):
        ms.append(m("t.total_live", max(total, lv["e2e_s"])))
    for name in PHASES:
        if any(name in r for r in reps):
            ms.append(m("prove." + ("host_" if name.startswith("h.") else "") + name[2:] + "_s", _sum(reps, name), "s"))
    if any("host_s" in r for r in reps):
        ms.append(m("prove.host_s", _sum(reps, "host_s"), "s"))
    if lv.get("witness_s") is not None:
        ms.append(m("prove.witness_wait_s", lv["witness_s"], "s"))
    if lv.get("prover_peak_gb"):
        ms.append(m("mem.peak_host_bytes", int(lv["prover_peak_gb"] * 2**30)))
    if gpu_peak_mib:
        ms.append(m("mem.peak_device_bytes", int(gpu_peak_mib * 2**20)))
    sent, recv = lv.get("wire_sent"), lv.get("wire_recv")
    if sent is not None and recv is not None:
        ms += [m("transcript.bytes", sent + recv), m("net.bytes_out", sent), m("net.bytes_in", recv)]
    pb = lv.get("proof_bytes")
    if pb:
        ms.append(m("proof_bytes", sum(pb) if isinstance(pb, list) else pb, "B"))
    if lv.get("verify_s") is not None:
        ms.append(m("verify.live_s", lv["verify_s"], "s"))
    if rec.get("stage_setup_s") is not None:
        ms.append(m("fixed.setup_seconds", rec["stage_setup_s"]))
    return ms


#: The Lean verifier's refusal of a session without a sample, issued only once every check has passed and the statement's form is
#: covered (verity/ml/flock/verifier/PROTOCOL.md §16.15, D12). A session counts toward throughput only when the Lean verifier
#: accepts it (the proofs ruling of 17 Oct): a point whose sessions meet this refusal fails, labelled ``D12_LABEL``, and gives
#: no headline. Its result still counts them (``verify.lean_drawless``).
DRAWLESS = "refused: a session without a draw: every soundness claim is about a drawn session (D12)"
D12_LABEL = "checks pass, refused under D12 (no draw)"


def lean_passed(v: Mapping[str, Any]) -> bool:
    return bool(v.get("accepted"))


def verifier_measurements(lean: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The Lean verifier's side: its median wall and CPU seconds over the timed sessions, its peak memory, and how many
    sessions it accepted and how many it refused for having no sample alone."""
    timed = [v for v in lean if v.get("wall_s") is not None]
    if not timed:
        return []
    m = contract.measurement
    wall = statistics.median(v["wall_s"] for v in timed)
    ms = [m("verifier.seconds", wall), m("verify.wall_s", wall),
          m("verify.lean_accepted", sum(1 for v in lean if v.get("accepted")), "count"),
          m("verify.lean_drawless", sum(1 for v in lean if v.get("why") == DRAWLESS), "count")]
    cpu = [v["cpu_s"] for v in timed if v.get("cpu_s") is not None]
    if cpu:
        ms.append(m("verify.cpu_s", statistics.median(cpu)))
    rss = [v["max_rss_kb"] for v in timed if v.get("max_rss_kb")]
    if rss:
        ms.append(m("verify.peak_rss_bytes", max(rss) * 1024, "B"))
    # the verifier's line for a session times it from before its setup, so the session's own seconds are that less the setup
    setup = [v["setup_ms"] for v in timed if v.get("setup_ms") is not None]
    if setup:
        ms.append(m("verify.setup_s", statistics.median(setup) / 1000, "s"))
    own = [v["session_ms"] - (v.get("setup_ms") or 0) for v in timed if v.get("session_ms") is not None]
    if own:
        ms.append(m("verify.session_s", statistics.median(own) / 1000, "s"))
    return ms


def fingerprint(w: Workload, B: int, rec: Mapping[str, Any], *, hardware: Mapping[str, Any], software: Mapping[str, Any],
                config: str = "plain", **extra: Any) -> dict[str, Any]:
    lv = rec.get("live") or {}
    return contract.fingerprint(
        profile=w.profile, domain=rec.get("domain") or "total", K=w.K, B=B, proof_class="NON_ZK_PROOF",
        authentication="included-hash", zk_mode=contract.ZK_NONE, security=SECURITY, hardware=hardware, software=software,
        preprocessing=PREPROCESSING, retention="retained", instances=None, population=I.population_ref(w),
        draw={"law": "stratified", "k": B, "population": w.n_max, "verifier": "flock-verify --stratified B, its own law"},
        workload=w.name, vu=w.vu, sites=w.sites, config=config, configuration=CONFIGS[config],
        statement_digest=lv.get("statement_digest"), k_log=lv.get("k_log"), dense_m=lv.get("dense_m"),
        coins=lv.get("coins"), **extra)


def software(rec: Mapping[str, Any], *, commit: str | None, dirty: bool | None, binary_sha256: str | None) -> dict[str, Any]:
    """The prover's software, as the prover run saw it (``software.json`` in the bundle); the verifier run fills ``verifier``."""
    lv = rec.get("live") or {}
    return {"backend": {"name": "flock-circuit", "commit": commit, "dirty": dirty, "binary_sha256": binary_sha256,
                        "prover": lv.get("prover"),
                        "env": {k: os.environ.get(k) for k in ("FC_PIPELINE_DEPTH", "FC_HOST_PREPIN", "FC_COINS", "FLOCK_OWN_ROWS")}},
            "verifier": {"name": "flock-verify", "language": "Lean (compiled)", "sha256": None, "build_key": None},
            "toolchain": {"python": sys.version.split()[0], "cuda": os.environ.get("PT_CUDA")}}


def point(w: Workload, B: int, rec: Mapping[str, Any], *, run_id: str, hardware: Mapping[str, Any],
          software: Mapping[str, Any], gpu_peak_mib: float | None, lean: Sequence[Mapping[str, Any]] | None,
          config: str = "plain", evidence: Sequence[str] = (), **extra: Any) -> dict[str, Any]:
    """The point's result: the prover's record and, when given, the Lean verifier's verdicts on its timed sessions."""
    lv = rec.get("live") or {}
    live_ok = bool(lv.get("accepted"))
    lean = list(lean or [])
    lean_ok = bool(lean) and all(lean_passed(v) for v in lean)
    ms = prover_measurements(w, B, rec, gpu_peak_mib) if lv.get("prove_total_s") is not None else []
    ms += verifier_measurements(lean)
    cpu, total = next((x["value"] for x in ms if x["name"] == "verify.cpu_s"), None), next((x["value"] for x in ms if x["name"] == "t.total"), None)
    if cpu and total:
        ms += contract.verifier_rates(cpu, B, total)
    why = []
    if not live_ok:
        why.append(f"live verifier: {lv.get('why') or rec.get('broke') or 'no record'}")
    accepted = sum(1 for v in lean if v.get("accepted"))
    drawless = sum(1 for v in lean if v.get("why") == DRAWLESS)
    if lean and not lean_ok:
        if drawless:
            why.append(f"Lean verifier: {accepted} of {len(lean)} sessions accepted; {drawless} {D12_LABEL}: no headline")
        why += [f"Lean {v.get('session')}: {v.get('why')}" for v in lean if not lean_passed(v) and v.get("why") != DRAWLESS][:4]
    status = "passed" if live_ok and (lean_ok or not lean) else "failed"
    checker = ("flock-verify (compiled Lean verifier) on every timed session; flock-circuit serve (live, loopback) on every session"
               if lean else "flock-circuit serve (live, loopback) on every session; the Lean verifier not run yet")
    lean_said = f"Lean verifier: {accepted} of {len(lean)} sessions accepted; " if lean else ""
    detail = "; ".join(why) or (lean_said + f"{lv.get('timed_runs')} timed and the warm sessions accepted live")
    fp = fingerprint(w, B, rec, hardware=hardware, software=software, config=config, **extra)
    doc = {"schema": contract.RESULT_SCHEMA, "run_id": run_id, "workload_fingerprint": fp,
           "validation": {"status": status, "checker": checker, "detail": detail[:900], "evidence": list(evidence)},
           "measurements": ms, "measurement_files": list(evidence), "artifacts": []}
    problems = contract.validate(doc)
    if problems:
        doc["validation"] = {"status": "failed", "checker": "verity_numerical.bench.contract.validate",
                             "detail": "; ".join([*why, "contract: " + "; ".join(problems)])[:900], "evidence": list(evidence)}
    return doc


def value(result: Mapping[str, Any], name: str) -> float | None:
    for m in result.get("measurements") or []:
        if m.get("name") == name:
            return m.get("value")
    return None


def dominant_bucket(result: Mapping[str, Any]) -> tuple[str, float] | None:
    """The largest part of ``t.total``, with its share: the contract's buckets, ``t.arithmetic`` split into the prover's phases
    when it reports them, the rep's host time and the wait on the witness."""
    total = value(result, "t.total")
    if not total:
        return None
    parts = {n: value(result, n) or 0.0 for n in contract.BUCKET_NAMES}
    split = {n: value(result, n) for n in ("prove.zerocheck_s", "prove.lincheck_s", "prove.ring_switch_s", "prove.ligerito_s")}
    if all(v is not None for v in split.values()):
        del parts["t.arithmetic"]
        parts |= split
    for n in ("prove.link_s", "prove.host_s", "prove.witness_wait_s"):
        parts[n] = value(result, n) or 0.0
    name = max(parts, key=parts.get)
    return name, parts[name] / total
