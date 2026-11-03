"""The recursive session's figures from one run's result (``rec_profile.read``): its cost, its knowledge-soundness error
and its stop-leak bound, for one window and pooled over the monitoring periods (`wipe_monitoring.PERIODS`: a week, a month,
a year), over one datacenter (``benchmarks/census/deployments.json``'s reference) and over the world's compute (the
adversary resource model, ``adversary.json``'s reference). Every figure says its basis: ``measured`` (this run),
``proved`` (a Lean theorem, by name, with what it assumes) or ``scaled`` (and what is scaled).

What is scaled is the same for every pooled figure: every machine of the fleet is audited as this run's session audited
its window, one recursive session a machine a window, each as this one; the world's compute counts as machines of the
datacenter's size (`gpus_per_machine`). The session covers its statement, the window's drawn blocks: its AND gates are the
inner statement's (its units times one unit's ANDs, from the circuit), the compute it proves, not the window's.

* **Cost** (measured, then scaled): the session's steps (``SESSION_STEPS``) end to end, their wall and CPU seconds (each
  step's ``/usr/bin/time -v``: the prover, the ZK server and Lean's verifier together), its proofs' bytes and its release
  schedule; the CPU cores it keeps busy a machine (CPU seconds over the window); pooled, the CPU seconds and proof bytes of
  every session, and as a share of the compute covered, CPU seconds per AND proved.
* **Knowledge soundness** (proved, at the adversary's SHA-512 calls): ``Flock.SecurityProofs.RecursiveKS`` (the lock, no
  assumption) is ``KSLin2On`` at the deployed dense statements; ``bitsLin2On_of_ksLin2On`` makes it
  ``Pr[acc ∧ ¬R] ≤ t·2^{-λ₂}`` for a prover making at most t SHA-512 calls a run, under the named assumptions
  ``StrictBirthday`` and ``ExpectedCR``, and ``RecDeploy.lamLin2_deployed_ge_dep`` (the lock) gives λ₂ ≥ 100 at the
  deployed records. Each session's calls are its own run's, so over a period they sum to at most the adversary's calls T,
  and the period's error is at most T·2^{-λ₂}, for the datacenter's sessions as for the world's. T is the adversary's: the
  world's GPUs (the adversary resource model) at ``HASH_RATE_LOG2`` compressions a second each, an assumed bound. Beside
  it, ``Claims.Recursion.lam``'s knowledge form, about ``LAM_KNOWLEDGE`` bits.
* **Stop leak** (proved, under open hypotheses): ``Flock.zk_server_lean``, one bit an outer session (its factor 2), under
  ``Claims.Hm96Hiding`` and the item-stream refinement (``ItemsRefine``), whose ``HelloEq``, ``LinkRoundEq`` and
  ``FinalRoundEq`` are not yet proved about the live path; over a stream of m outer sessions with at most b stops,
  log₂ Σ_{i≤b} C(m, i) bits (``stream_leak_deferred``). A recursive session's first stop ends it, so it is a stream of its
  S parts with at most one stop, log₂(1 + S) bits; a machine's stream over a period of R sessions has m = R·S and b = R, or
  b = ``budget`` under a failure budget. Pooled over the fleet, one stream a machine; as a share, bits per AND covered.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from verity.core.protocols.pous.wipe_monitoring import PERIODS  # noqa: E402

FORMAT = "recursion-e2e/figures/v1"
CENSUS = HERE.parent / "census"
YEAR_S = dict(PERIODS)["year"]
#: one recursive session's steps after its build: the inner session, V*'s staging, its outer proofs, Lean's verdicts
SESSION_STEPS = ("inner", "vstage", "vcap", "oprove", "lean")
#: λ₂'s floor at the deployed records (``RecDeploy.lamLin2_deployed_ge_dep``)
LAM2 = 100
#: ``Claims.Recursion.lam`` at the deployed shape, as its docstring evaluates it: the knowledge form's level
LAM_KNOWLEDGE = 65
#: the adversary's SHA-512 compressions a second a GPU, log₂: an assumed bound, above any GPU's measured commit rate
HASH_RATE_LOG2 = 33
KS = {"guarantee": "Flock.SecurityProofs.RecursiveKS", "level": "RecDeploy.lamLin2_deployed_ge_dep",
      "bits": "Claims.Recursion.bitsLin2On_of_ksLin2On", "assumes": ["StrictBirthday", "ExpectedCR"],
      "knowledge_form": "Claims.Recursion.lam"}
LEAK = {"theorem": "Flock.zk_server_lean", "stream": "Flock.ZkServer.stream_leak_deferred",
        "assumes": ["Claims.Hm96Hiding", "LeanRelease.ItemsRefine (HelloEq, LinkRoundEq, FinalRoundEq open on the live path)"]}


def log2_binom_sum(m: int, b: int) -> float:
    """log₂ Σ_{i ≤ b} C(m, i)."""
    b = min(b, m)
    if b == m:
        return float(m)
    terms = [(math.lgamma(m + 1) - math.lgamma(i + 1) - math.lgamma(m - i + 1)) / math.log(2) for i in range(b + 1)]
    top = max(terms)
    return top + math.log2(sum(2 ** (t - top) for t in terms))


def fleets(census: Path = CENSUS) -> dict[str, dict[str, Any]]:
    """The datacenter (the reference deployment) and the world (the reference adversary, as machines of its size)."""
    (dc,) = [d for d in json.loads((census / "deployments.json").read_text()) if d.get("role") == "reference"]
    (world,) = [d for d in json.loads((census / "adversary.json").read_text()) if d.get("role") == "reference"]
    g = dc["gpus_per_machine"]
    return {"datacenter": {"id": dc["id"], "machines": dc["machines"], "gpus": dc["machines"] * g, "window_s": dc["window_s"]},
            "world": {"id": world["id"], "machines": world["h100_equivalents"] / g, "gpus": world["h100_equivalents"],
                      "window_s": dc["window_s"], "flop_per_year": world["total_flop"]}}


def session(doc: dict[str, Any], times: dict[str, dict[str, float] | None]) -> dict[str, Any]:
    """The measured session: its statement's ANDs, each session step's wall and CPU seconds (``times``, each step's first
    invocation's ``/usr/bin/time -v``), its proofs and its schedule."""
    st = doc.get("staged") or {}
    n, a = st.get("n"), st.get("unit_ands")
    steps = {s: (t := times.get(s)) and {"wall_s": round(t["wall_s"], 3), "cpu_s": round(t["cpu_s"], 3)} for s in SESSION_STEPS}
    rows = doc.get("rows", [])
    proofs = sum(r.get("proof_bytes") or 0 for r in rows if r["step"] == "oprove" and not r["stage"].startswith("outer forged-"))
    inner = next((r.get("proof_bytes") for r in rows if r["stage"] == "inner M0"), None)
    rel = doc.get("release") or {}
    parts = len(rel.get("parts") or []) or len([r for r in rows if r["step"] == "oprove" and not r["stage"].startswith("outer forged-")])
    whole = all(steps.values())
    return {"statement": {"subcircuit": st.get("subcircuit"), "unit": st.get("unit"), "units": n, "unit_ands": a,
                          "ands": n * a if n and a else None},
            "parts": parts, "steps": steps, "complete": whole,
            "wall_s": round(sum(v["wall_s"] for v in steps.values() if v), 3) if whole else None,
            "cpu_s": round(sum(v["cpu_s"] for v in steps.values() if v), 3) if whole else None,
            "proof_bytes": {"outer": proofs or None, "inner": inner},
            "release": {"ok": rel.get("ok"), "schedule_s": rel["schedule_ms"] / 1000 if rel.get("schedule_ms") else None,
                        "slack_s": [round(p["slack_ms"] / 1000, 3) if p.get("slack_ms") is not None else None
                                    for p in rel.get("parts", [])]} if rel else None}


def figures(doc: dict[str, Any], times: dict[str, dict[str, float] | None], census: Path = CENSUS,
            budget: int | None = None) -> dict[str, Any]:
    """Every figure, per window and per period for each fleet (module docstring)."""
    s = session(doc, times)
    S, ands, cpu = s["parts"], s["statement"]["ands"], s["cpu_s"]
    out: dict[str, Any] = {"format": FORMAT, "session": s, "basis": {
        "cost": "measured: this run's session; scaled: every machine of the fleet audited so, a session a window",
        "ks": f"proved: {KS['guarantee']} (lock, no assumption), {KS['level']} (lock: λ₂ ≥ {LAM2}), {KS['bits']} "
              f"(under {', '.join(KS['assumes'])}); scaled: the adversary's SHA-512 calls, the world's GPUs at 2^{HASH_RATE_LOG2} a "
              "second each (assumed)",
        "leak": f"proved: {LEAK['theorem']} (one bit an outer session) and {LEAK['stream']}, under {'; '.join(LEAK['assumes'])}; "
                "scaled: a stream a machine, a session a window",
        "scaled": "every machine audited as this run's session audited its window, a session a machine a window; the world "
                  "as machines of the datacenter's size; covered compute is each session's statement's ANDs"},
        "levels": {"lam2_floor": LAM2, "lam_knowledge": LAM_KNOWLEDGE, "hash_rate_log2": HASH_RATE_LOG2}}
    leak_session = math.log2(1 + S) if S else None
    fl = {}
    fs = fleets(census)
    adversary = fs["world"]["gpus"]
    for name, f in fs.items():
        w = f["window_s"]
        t_window = math.log2(adversary * w) + HASH_RATE_LOG2
        per_window = {
            "cost": {"cpu_s": cpu, "cores_per_machine": cpu / w if cpu else None, "wall_s": s["wall_s"],
                     "fits_window": s["wall_s"] <= w if s["wall_s"] else None,
                     "cpu_s_per_and": cpu / ands if cpu and ands else None},
            "ks": {"sha512_calls_log2": round(t_window, 2), "error_log2": round(t_window - LAM2, 2),
                   "error_log2_knowledge_form": round(t_window - LAM_KNOWLEDGE, 2)},
            "leak": {"bits": leak_session, "bits_per_and": leak_session / ands if leak_session and ands else None}}
        periods = {}
        for p, secs in PERIODS:
            R = secs // w
            n = f["machines"] * R
            T = math.log2(adversary * secs) + HASH_RATE_LOG2
            per_stream = log2_binom_sum(R * S, R) if S else None
            budgeted = log2_binom_sum(R * S, budget) if S and budget else None
            covered = n * ands if ands else None
            periods[p] = {
                "seconds": secs, "sessions": n, "sessions_per_machine": R, "covered_ands": covered,
                "cost": {"cpu_s": n * cpu if cpu else None, "core_years": n * cpu / YEAR_S if cpu else None,
                         "proof_bytes": n * s["proof_bytes"]["outer"] if s["proof_bytes"]["outer"] else None,
                         "cpu_s_per_and": cpu / ands if cpu and ands else None},
                "ks": {"sha512_calls_log2": round(T, 2), "error_log2": round(T - LAM2, 2),
                       "error_log2_knowledge_form": round(T - LAM_KNOWLEDGE, 2)},
                "leak": {"bits_per_stream": per_stream, "bits": f["machines"] * per_stream if per_stream else None,
                         "bits_per_and": f["machines"] * per_stream / covered if per_stream and covered else None,
                         **({"budget": budget, "bits_per_stream_budgeted": budgeted,
                             "bits_budgeted": f["machines"] * budgeted} if budgeted is not None else {})}}
        fl[name] = {"fleet": f, "window": per_window, "periods": periods}
    out["fleets"] = fl
    return out


def _e(x: Any, unit: str = "") -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    return f"{x:.3g}{unit}"


def markdown(fig: dict[str, Any]) -> str:
    s = fig["session"]
    st = s["statement"]
    lines = ["", "**The session's figures.** One recursive session over "
             f"{st['units']} x {st['unit']} ({_e(st['ands'])} ANDs proved), {s['parts']} parts, "
             f"{_e(s['wall_s'])} s wall and {_e(s['cpu_s'])} CPU s over {', '.join(SESSION_STEPS)}"
             + (f", released on a {_e(s['release']['schedule_s'])} s schedule (ok: {_e(s['release']['ok'])})" if s.get("release") else "")
             + ".", "", f"- Cost: {fig['basis']['cost']}.", f"- Knowledge soundness: {fig['basis']['ks']}.",
             f"- Stop leak: {fig['basis']['leak']}.", "",
             "| fleet | span | sessions | ANDs covered | CPU s | core-years | CPU s per AND | log₂ KS error (λ₂ ≥ 100) | "
             "log₂ KS error (knowledge form) | stop leak bits | bits per AND |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, f in fig["fleets"].items():
        w = f["window"]
        lines.append(f"| {name} | window ({f['fleet']['window_s']} s, one session) | 1 | {_e(st['ands'])} | {_e(w['cost']['cpu_s'])} "
                     f"| — | {_e(w['cost']['cpu_s_per_and'])} | {w['ks']['error_log2']} | {w['ks']['error_log2_knowledge_form']} "
                     f"| {_e(w['leak']['bits'])} | {_e(w['leak']['bits_per_and'])} |")
        for p, r in f["periods"].items():
            lines.append(f"| {name} | {p} | {_e(r['sessions'])} | {_e(r['covered_ands'])} | {_e(r['cost']['cpu_s'])} | "
                         f"{_e(r['cost']['core_years'])} | {_e(r['cost']['cpu_s_per_and'])} | {r['ks']['error_log2']} | "
                         f"{r['ks']['error_log2_knowledge_form']} | {_e(r['leak']['bits'])} | {_e(r['leak']['bits_per_and'])} |")
    lines.append("")
    lines.append(f"Scaled: {fig['basis']['scaled']}. A log₂ error at or above 0 is no bound.")
    return "\n".join(lines) + "\n"
