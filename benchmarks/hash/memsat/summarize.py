"""One memsat run's table, by arm, from its out/ (run.sh's): node.py's prover.json (sessions, calibration), each engine's
engine.<g>.json and steps.jsonl (engine.py's phases and steps) and engine.<g>.latency.json (engine_int8.py's histograms),
and, given --verifier, the verifier run's verifier.json (γ). Writes OUT/memsat.json and, with --result, a
research/result/v0.1 result whose every measurement names its arm, so the hash loop's plot.py draws them as
llama8b-int8:<arm> once the run is labeled arm=llama8b-int8.

    python3 summarize.py OUT [--verifier VERIFIER_JSON] [--result PATH] [--window-ms 10000]

Arms (engine.py's phase tags; tokens a second are summed over the engines, each over its own steps in the arm):
  plain               plain-baseline and plain: the plain weights, before the prover touches a GPU
  codec               decode-only, between and after: the arenas encoded, every weight decoded at use, no pass
  receipts+codec      an unpaced session-K: the audit's passes back to back (sketch-sha512 receipts over every arena)
                      beside decode at use, as node.py runs them
  receipts+codec@R/s  a session node_paced.py paced to R passes a second
  calibrate           the calibration's back-to-back passes beside decode at use (reported, not an arm of the loop)
Overheads against plain in the same run: tpot_overhead_pct = 100 (TPOT_arm / TPOT_plain - 1) on the mean TPOT, and
throughput_overhead_pct = 100 (1 - tok/s_arm / tok/s_plain). A receipts arm's passes against the codec arm, per pass at
the arm's pass rate: receipts_overhead_pct = 100 (1 - tok/s_arm / tok/s_codec), tokens_lost_per_pass = (tok/s_codec -
tok/s_arm) / passes_per_s, and serving_ms_lost_per_pass = 1000 (1 - tok/s_arm / tok/s_codec) / passes_per_s, the node's
serving time a pass costs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

BIN_MS = 0.1
ARMS = {"plain": ("plain-baseline", "plain"), "codec": ("decode-only", "between", "after"), "calibrate": ("calibrate",)}


def session_arms(prover: dict) -> dict[int, str]:
    """Each session's arm: receipts+codec unpaced, receipts+codec@R/s at node_paced.py's R."""
    return {s["k"]: "receipts+codec" + (f"@{s['pace_passes_per_s']:g}/s" if s.get("pace_passes_per_s") else "")
            for s in prover.get("sessions", [])}


def arm_of(phase: str, sessions: dict[int, str]) -> str | None:
    if phase.startswith("session-"):
        k = phase.split("-")[1]
        return sessions.get(int(k), "receipts+codec") if k.isdigit() else None
    return next((arm for arm, tags in ARMS.items() if phase in tags), None)


def quantile(hist: Counter, q: float) -> float | None:
    n = sum(hist.values())
    if not n:
        return None
    k, seen = q * (n - 1), 0
    for b in sorted(hist):
        seen += hist[b]
        if seen > k:
            return round((b + 1) * BIN_MS, 3)
    return None


def mean(hist: Counter) -> float | None:
    n = sum(hist.values())
    return round(sum((b + 0.5) * BIN_MS * c for b, c in hist.items()) / n, 3) if n else None


def arms(out: Path, sessions: dict[int, str]) -> dict:
    """Each arm's tokens a second (summed over engines), busy fraction, batch, and merged latency histograms."""
    acc: dict[str, dict] = {}
    for f in sorted(out.glob("engine.*.latency.json")):
        lat = json.loads(f.read_text())
        for phase, p in lat["phases"].items():
            arm = arm_of(phase, sessions)
            if arm is None:
                continue
            a = acc.setdefault(arm, {"phases": set(), "engines": {}, "hist": {k: Counter() for k in ("itl", "ttft", "tpot")}})
            a["phases"].add(phase)
            e = a["engines"].setdefault(lat["gpu"], {"tokens": 0, "span_s": 0.0, "busy_s": 0.0, "steps": 0, "batch_sum": 0})
            for k in e:
                e[k] += p[k]
            for k, h in p["hist"].items():
                a["hist"][k].update({int(b): c for b, c in h.items()})
    res = {}
    for arm, a in acc.items():
        es = a["engines"].values()
        res[arm] = {
            "phases": sorted(a["phases"]), "engines": len(a["engines"]),
            "tokens_per_s": round(sum(e["tokens"] / e["span_s"] for e in es if e["span_s"]), 2),
            "span_s": round(max(e["span_s"] for e in es), 1),
            "busy": round(sum(e["busy_s"] for e in es) / sum(e["span_s"] for e in es), 4),
            "batch_mean": round(sum(e["batch_sum"] for e in es) / max(1, sum(e["steps"] for e in es)), 1),
            "steps_per_s": round(sum(e["steps"] / e["span_s"] for e in es if e["span_s"]) / len(a["engines"]), 3),
            **{f"{k}_{s}_ms": f(a["hist"][k]) for k in ("tpot", "itl", "ttft")
               for s, f in (("mean", mean), ("p50", lambda h: quantile(h, 0.5)), ("p99", lambda h: quantile(h, 0.99)))},
            "requests_finished": sum(a["hist"]["tpot"].values())}
    return res


def step_ms(out: Path, sessions: dict[int, str]) -> dict:
    """Each arm's engine step time, p50 and p99, over every step of every engine in the arm (engine.py's steps.jsonl)."""
    by: dict[str, list[float]] = {}
    for f in sorted(out.glob("engine.*.steps.jsonl")):
        for line in f.read_text().splitlines():
            s = json.loads(line)
            if arm := arm_of(s.get("phase", ""), sessions):
                by.setdefault(arm, []).append(s["ms"])
    q = lambda xs, p: round(sorted(xs)[min(len(xs) - 1, int(p * len(xs)))], 2)
    return {arm: {"step_ms_p50": q(xs, 0.5), "step_ms_p99": q(xs, 0.99)} for arm, xs in by.items() if xs}


def passes(prover: dict, window_ms: float | None, sessions: dict[int, str]) -> dict:
    """The calibration's pass rate, each session's, and each receipts arm's (the mean over its honest sessions)."""
    cal = prover.get("calibration") or {}
    out = {"calibrate": {"passes_per_s": round(1e9 / cal["pass_ns_median"], 2) if cal.get("pass_ns_median") else None,
                         "pass_ms_p50": round(cal["pass_ns_median"] / 1e6, 3) if cal.get("pass_ns_median") else None,
                         "pass_gpu_ms_p50": cal.get("pass_gpu_ms_median"), "read_ms": cal.get("read_ms"),
                         "slow_passes": cal.get("slow_passes"), "tau": cal.get("tau")}}
    rows = []
    for s in prover.get("sessions", []):
        n, w = sum(s["passes"]), len(s["passes"])
        host = sorted(s.get("pass_host_ns") or [0])
        rows.append({"k": s["k"], "arm": sessions[s["k"]], "mode": s["mode"], "accepted": s["accepted"], "passes": n,
                     "windows": w, "pace_passes_per_s": s.get("pace_passes_per_s", 0.0),
                     "passes_per_s": round(n / (w * window_ms / 1e3), 2) if window_ms else None,
                     "pass_host_ms_p50": round(host[len(host) // 2] / 1e6, 3),
                     "pass_gpu_ms_p50": (s.get("device") or {}).get("pass_gpu_ms_median")})
    out["sessions"] = rows
    for arm in sorted(set(sessions.values())):
        honest = [s for s in rows if s["arm"] == arm and s["mode"] == "honest" and s["passes_per_s"]]
        if honest:
            out[arm] = {"passes_per_s": round(sum(s["passes_per_s"] for s in honest) / len(honest), 2),
                        "pass_host_ms_p50": sorted(s["pass_host_ms_p50"] for s in honest)[len(honest) // 2]}
    return out


def gammas(verifier: dict) -> list[dict]:
    out = []
    for s in verifier.get("sessions", []):
        lg = s.get("ledger") or {}
        row = {"k": s.get("k"), "accepted": s.get("accepted")}
        for name in ("spec", "measured"):
            x = lg.get(name) or {}
            row[f"gamma_{name}"] = x.get("gamma")
            row[f"gamma_horizon_{name}"] = x.get("gamma_horizon")
            if name == "spec" and x.get("timing_parts"):
                row["timing_parts_spec"] = x["timing_parts"]
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--verifier", type=Path)
    ap.add_argument("--result", type=Path)
    ap.add_argument("--window-ms", type=float, default=float(os.environ.get("WINDOW_MS") or 0) or None)
    a = ap.parse_args()
    prover = json.loads((a.out / "prover.json").read_text())
    sessions = session_arms(prover)
    table = arms(a.out, sessions)
    for arm, s in step_ms(a.out, sessions).items():
        table.setdefault(arm, {}).update(s)
    pz = passes(prover, a.window_ms, sessions)
    for arm, t in table.items():
        t.update(pz.get(arm, {}))
    gz = gammas(json.loads(a.verifier.read_text())) if a.verifier else []
    for arm in set(sessions.values()) & set(table):
        for key in ("gamma_spec", "gamma_measured"):
            xs = [g[key] for g in gz if sessions.get(g["k"]) == arm and g.get(key) is not None]
            if xs:
                table[arm][key] = round(sum(xs) / len(xs), 5)
    base, codec = table.get("plain", {}), table.get("codec", {})
    for arm, t in table.items():
        if arm != "plain" and base.get("tpot_mean_ms") and t.get("tpot_mean_ms"):
            t["tpot_overhead_pct"] = round(100 * (t["tpot_mean_ms"] / base["tpot_mean_ms"] - 1), 2)
        if arm != "plain" and base.get("tokens_per_s") and t.get("tokens_per_s"):
            t["throughput_overhead_pct"] = round(100 * (1 - t["tokens_per_s"] / base["tokens_per_s"]), 2)
        if arm not in ("plain", "codec") and codec.get("tokens_per_s") and t.get("tokens_per_s") and t.get("passes_per_s"):
            lost = 1 - t["tokens_per_s"] / codec["tokens_per_s"]
            t["receipts_overhead_pct"] = round(100 * lost, 2)
            t["tokens_lost_per_pass"] = round((codec["tokens_per_s"] - t["tokens_per_s"]) / t["passes_per_s"], 2)
            t["serving_ms_lost_per_pass"] = round(1e3 * lost / t["passes_per_s"], 3)
    gates = prover.get("gates") or {}
    engines_ok = all(e.get("arena_decodes") and e.get("probe_tokens_equal") for e in gates.get("engines", [])) \
        and bool(gates.get("engines"))
    honest = [s for s in pz["sessions"] if s["mode"] == "honest"]
    doc = {"schema": "verity/hash-memsat/v0", "arms": table, "sessions": pz["sessions"], "gates": gates,
           "engines": [{k: e.get(k) for k in ("gpu", "uuid", "bytes", "weights_bytes", "weights_blocks", "params", "model")}
                       for e in prover.get("engines", [])], "error": prover.get("error")}
    if a.verifier:
        doc["gamma"] = gz
    (a.out / "memsat.json").write_text(json.dumps(doc, indent=1) + "\n")
    errors = []
    if prover.get("error"):
        errors.append(f"prover: {prover['error']}")
    if not engines_ok:
        errors.append("an engine's decode gate failed or is missing")
    if not honest or not all(s["accepted"] for s in honest):
        errors.append("an honest session missing or rejected")
    if "plain" not in table:
        errors.append("no plain phase")
    if a.result:
        meas = []
        for arm, t in table.items():
            for k, v in t.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    unit = "%" if k.endswith("_pct") else "ms" if k.endswith("_ms") else "1/s" if k.endswith("_per_s") else ""
                    meas.append({"name": k, "arm": arm, "value": v, "unit": unit})
        status = "passed" if not errors else "failed"
        res = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", "local"),
               "workload_fingerprint": {"bench": "benchmarks/hash/memsat/run.sh (remote_session SERVE=node)",
                                        "workload": "llama8b-int8", "model": (doc["engines"] or [{}])[0].get("model"),
                                        "engine_args": os.environ.get("ENGINE_ARGS"), "node_args": os.environ.get("NODE_ARGS"),
                                        "plan": os.environ.get("PLAN"), "window_ms": a.window_ms,
                                        "pin": {"state": "unpinned"}},
               "validation": {"status": status, "checker": "engines' decode gates (arena SHA-256, probe tokens), every "
                              "honest session accepted by the verifier", "detail": "; ".join(errors) or "gates passed",
                              "evidence": []},
               "measurements": meas, "measurement_files": ["out/memsat.json"]}
        a.result.write_text(json.dumps(res, indent=1) + "\n")
    for arm, t in table.items():
        print(arm, json.dumps({k: v for k, v in t.items() if not isinstance(v, list)}))
    print("errors:", errors or "none")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
