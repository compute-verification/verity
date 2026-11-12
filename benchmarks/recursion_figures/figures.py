"""The recursive session's figures from one recursion bench result (``benchmarks/recursion_e2e``'s
``recursion_e2e.json``, ``rec_profile.read``'s, with its ``session_times``):

    python benchmarks/recursion_figures/figures.py RESULT.json [--json]

The bench is public and states what it measured; this side reads the census's fleets and the monitoring periods, which
stay private, and scales the measurement over them: its cost, its knowledge-soundness error
and its stop-leak bound, for one window and pooled over the monitoring periods (`reporting.PERIODS`: a week, a month,
a year), over one datacenter (``benchmarks/census/deployments.json``'s reference) and over the world's compute (the
adversary resource model, ``adversary.json``'s reference). Every figure says its basis: ``measured`` (this run),
``proved`` (a Lean theorem, by name, with what it assumes) or ``scaled`` (and what is scaled).

What is scaled is the same for every pooled figure: every machine of the fleet is audited as this run's session audited
its window, one recursive session a machine a window, each as this one; the world's compute counts as machines of the
datacenter's size (`gpus_per_machine`). The session covers its statement, the window's sampled blocks: its AND gates are the
inner statement's (its instances, each of ``units_per_vu`` units, times one unit's ANDs, from the circuit), the compute it
proves, not the window's.

* **Cost** (measured, then scaled): the session's steps (``SESSION_STEPS``) end to end, their wall and CPU seconds (each
  step's ``/usr/bin/time -v``: the prover, the ZK server and Lean's verifier together), its proofs' bytes and its release
  schedule; the CPU cores it keeps busy a machine (CPU seconds over the window); pooled, the CPU seconds and proof bytes of
  every session, and as a share of the compute covered, CPU seconds per AND proved.
* **Knowledge soundness** (κ and ε proved; the union over sessions scaled): ``RecDeploy.recursiveBitsK_deployed``
  (``Flock.SecurityProofs.RecursiveKS`` at the deployment's sizes) gives ``Pr[acc ∧ ¬R] ≤ κ + t·ε`` a session for a prover
  making at most t SHA-512 calls in its run, κ = 2^``KAPPA_LOG2`` and ε = 2^``EPS_LOG2``, under the SHA-512 collision
  assumptions ``StrictBirthday`` and ``ExpectedCR`` and hm96 hiding; ``Flock.Guarantees.RecursiveKSDeployed``'s knowledge
  form adds at most 2^``KNOWLEDGE_LOG2`` a session, counted in κ here. Over a period, by the union bound over its sessions
  (one a machine a window) whose calls sum to at most the adversary's calls T, the error is at most sessions·κ + T·ε.
  Two adversaries: the world's GPUs (the adversary resource model) and the datacenter's own, each at ``HASH_RATE_LOG2``
  SHA-512 calls a second a GPU, an assumed bound. As a share of the compute covered: the error over the sessions (one
  machine-window each) and over the ANDs they cover.
* **Stop leak** (proved, under open hypotheses): ``Flock.zk_server_lean``, one bit an outer session (its factor 2), under
  ``Claims.Hm96Hiding`` and the item-stream refinement (``ItemsRefine``), whose ``HelloEq``, ``LinkRoundEq`` and
  ``FinalRoundEq`` are not yet proved about the live path; over a stream of m outer sessions with at most b stops,
  log₂ Σ_{i≤b} C(m, i) bits (``stream_leak_deferred``). A recursive session's first stop ends it, so it is a stream of its
  S parts with at most one stop, log₂(1 + S) bits; a machine's stream over a period of R sessions has m = R·S and b = R, or
  b = ``budget`` under a failure budget. Pooled over the fleet, one stream a machine; as a share, bits per AND covered.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "recursion_e2e"))

from rec_profile import SESSION_STEPS  # noqa: E402
from verity.protocols.reporting import PERIODS  # noqa: E402

FORMAT = "recursion-e2e/figures/v2"
CENSUS = HERE.parent / "census"
YEAR_S = dict(PERIODS)["year"]
#: κ, the per-session term, and ε, the per-SHA-512-call term, log₂ (``RecDeploy.recursiveBitsK_deployed``)
KAPPA_LOG2 = -167.7
EPS_LOG2 = -174.95
#: the knowledge form's extraction term a session, log₂, a bound (``Flock.Guarantees.RecursiveKSDeployed``)
KNOWLEDGE_LOG2 = -182
#: the adversary's SHA-512 compressions a second a GPU, log₂: an assumed bound, above any GPU's measured commit rate
HASH_RATE_LOG2 = 33
#: the fleets whose GPUs bound the adversary's SHA-512 calls
ADVERSARIES = ("world", "datacenter")
KS = {"guarantee": "Flock.SecurityProofs.RecursiveKS", "level": "RecDeploy.recursiveBitsK_deployed",
      "knowledge": "Flock.Guarantees.RecursiveKSDeployed", "assumes": ["StrictBirthday", "ExpectedCR", "Claims.Hm96Hiding"]}


def log2_add(*xs: float) -> float:
    """log₂ Σ 2^x."""
    top = max(xs)
    return top + math.log2(sum(2 ** (x - top) for x in xs))


def ks_error(sessions: float, calls_log2: float, ands: float | None) -> dict[str, float | None]:
    """sessions·κ + T·ε (κ with the knowledge form's term), log₂, and its share of the sessions and of their ANDs."""
    e = log2_add(math.log2(sessions) + log2_add(KAPPA_LOG2, KNOWLEDGE_LOG2), calls_log2 + EPS_LOG2)
    share = e - math.log2(sessions)
    return {"sha512_calls_log2": round(calls_log2, 2), "error_log2": round(e, 2), "share_per_session_log2": round(share, 2),
            "share_per_and_log2": round(share - math.log2(ands), 2) if ands else None}
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
    # a stage.json without units_per_vu gives no count: one unit an instance would undercount a VU of several
    n, u, a = st.get("n"), st.get("units_per_vu"), st.get("unit_ands")
    steps = {s: (t := times.get(s)) and {"wall_s": round(t["wall_s"], 3), "cpu_s": round(t["cpu_s"], 3)} for s in SESSION_STEPS}
    rows = doc.get("rows", [])
    proofs = sum(r.get("proof_bytes") or 0 for r in rows if r["step"] == "oprove" and not r["stage"].startswith("outer forged-"))
    inner = next((r.get("proof_bytes") for r in rows if r["stage"] == "inner M0"), None)
    rel = doc.get("release") or {}
    parts = len(rel.get("parts") or []) or len([r for r in rows if r["step"] == "oprove" and not r["stage"].startswith("outer forged-")])
    whole = all(steps.values())
    return {"statement": {"subcircuit": st.get("subcircuit"), "unit": st.get("unit"), "units": n, "units_per_vu": u,
                          "unit_ands": a, "ands": n * u * a if n and u and a else None},
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
        "ks": f"proved: {KS['level']} ({KS['guarantee']} at the deployment's sizes: κ + t·ε a session, κ = 2^{KAPPA_LOG2}, "
              f"ε = 2^{EPS_LOG2} a SHA-512 call) and {KS['knowledge']} (at most 2^{KNOWLEDGE_LOG2} more a session), under "
              f"{', '.join(KS['assumes'])}; scaled: the union bound over the period's sessions (one a machine a window), "
              f"sessions·κ + T·ε, with T the adversary's SHA-512 calls, the world's or the datacenter's GPUs at "
              f"2^{HASH_RATE_LOG2} a second each (assumed)",
        "leak": f"proved: {LEAK['theorem']} (one bit an outer session) and {LEAK['stream']}, under {'; '.join(LEAK['assumes'])}; "
                "scaled: a stream a machine, a session a window",
        "scaled": "every machine audited as this run's session audited its window, a session a machine a window; the world "
                  "as machines of the datacenter's size; covered compute is each session's statement's ANDs"},
        "levels": {"kappa_log2": KAPPA_LOG2, "eps_log2": EPS_LOG2, "knowledge_log2": KNOWLEDGE_LOG2,
                   "hash_rate_log2": HASH_RATE_LOG2}}
    leak_session = math.log2(1 + S) if S else None
    fl = {}
    fs = fleets(census)
    adversaries = {a: fs[a]["gpus"] for a in ADVERSARIES}

    def ks(sessions: float, secs: int, covered: float | None) -> dict[str, Any]:
        return {a: ks_error(sessions, math.log2(g * secs) + HASH_RATE_LOG2, covered and covered / sessions)
                for a, g in adversaries.items()}

    for name, f in fs.items():
        w = f["window_s"]
        per_window = {
            "cost": {"cpu_s": cpu, "cores_per_machine": cpu / w if cpu else None, "wall_s": s["wall_s"],
                     "fits_window": s["wall_s"] <= w if s["wall_s"] else None,
                     "cpu_s_per_and": cpu / ands if cpu and ands else None},
            "ks": ks(1, w, ands),
            "leak": {"bits": leak_session, "bits_per_and": leak_session / ands if leak_session and ands else None}}
        periods = {}
        for p, secs in PERIODS:
            R = secs // w
            n = f["machines"] * R
            per_stream = log2_binom_sum(R * S, R) if S else None
            budgeted = log2_binom_sum(R * S, budget) if S and budget else None
            covered = n * ands if ands else None
            periods[p] = {
                "seconds": secs, "sessions": n, "sessions_per_machine": R, "covered_ands": covered,
                "cost": {"cpu_s": n * cpu if cpu else None, "core_years": n * cpu / YEAR_S if cpu else None,
                         "proof_bytes": n * s["proof_bytes"]["outer"] if s["proof_bytes"]["outer"] else None,
                         "cpu_s_per_and": cpu / ands if cpu and ands else None},
                "ks": ks(n, secs, covered),
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
             f"{st['units']} x {st['units_per_vu']} x {st['unit']} ({_e(st['ands'])} ANDs proved), {s['parts']} parts, "
             f"{_e(s['wall_s'])} s wall and {_e(s['cpu_s'])} CPU s over {', '.join(SESSION_STEPS)}"
             + (f", released on a {_e(s['release']['schedule_s'])} s schedule (ok: {_e(s['release']['ok'])})" if s.get("release") else "")
             + ".", "", f"- Cost: {fig['basis']['cost']}.", f"- Knowledge soundness: {fig['basis']['ks']}.",
             f"- Stop leak: {fig['basis']['leak']}.", "",
             "| fleet | span | sessions | ANDs covered | CPU s | core-years | CPU s per AND | log₂ KS error, world's calls | "
             "log₂ KS share a session | log₂ KS share an AND | log₂ KS error, datacenter's calls | stop leak bits | bits per AND |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def ks(k: dict[str, Any]) -> str:
        wk = k["world"]
        per_and = wk["share_per_and_log2"]
        return (f"{wk['error_log2']} | {wk['share_per_session_log2']} | {'—' if per_and is None else per_and} | "
                f"{k['datacenter']['error_log2']}")

    for name, f in fig["fleets"].items():
        w = f["window"]
        lines.append(f"| {name} | window ({f['fleet']['window_s']} s, one session) | 1 | {_e(st['ands'])} | {_e(w['cost']['cpu_s'])} "
                     f"| — | {_e(w['cost']['cpu_s_per_and'])} | {ks(w['ks'])} "
                     f"| {_e(w['leak']['bits'])} | {_e(w['leak']['bits_per_and'])} |")
        for p, r in f["periods"].items():
            lines.append(f"| {name} | {p} | {_e(r['sessions'])} | {_e(r['covered_ands'])} | {_e(r['cost']['cpu_s'])} | "
                         f"{_e(r['cost']['core_years'])} | {_e(r['cost']['cpu_s_per_and'])} | {ks(r['ks'])} | "
                         f"{_e(r['leak']['bits'])} | {_e(r['leak']['bits_per_and'])} |")
    lines.append("")
    lines.append(f"Scaled: {fig['basis']['scaled']}. KS: κ and ε proved, the union over sessions scaled; its share is the "
                 "pooled error over the period's sessions and over the ANDs they cover. A log₂ error at or above 0 is no bound.")
    return "\n".join(lines) + "\n"


def of_result(doc: dict[str, Any], census: Path = CENSUS) -> dict[str, Any]:
    """The figures of a result: its ``session_times`` and its release budget (``params``)."""
    if "session_times" not in doc:
        raise ValueError("the result has no session_times: reread its records with this tree's `run.py result --dir OUT`")
    budget = (doc.get("params") or {}).get("release_budget")
    return figures(doc, doc["session_times"], census, budget=int(budget) if budget else None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", type=Path, help="a recursion_e2e.json")
    ap.add_argument("--json", action="store_true", help="the figures as JSON, not the table")
    a = ap.parse_args(argv)
    fig = of_result(json.loads(a.result.read_text()))
    print(json.dumps(fig, indent=1) if a.json else markdown(fig), end="" if not a.json else "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
