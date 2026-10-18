"""One memsat run's table, by arm, from its out/ (run.sh's): node.py's prover.json (sessions, calibration), each engine's
engine.<g>.json (engine.py's phases) and engine.<g>.latency.json (engine_int8.py's histograms), and, given --verifier, the
verifier run's verifier.json (γ). Writes OUT/memsat.json and, with --result, a research/result/v0.1 result whose every
measurement names its arm, so the hash loop's plot.py draws them as llama8b-int8:<arm> once the run is labeled
arm=llama8b-int8.

    python3 summarize.py OUT [--verifier VERIFIER_JSON] [--result PATH] [--window-ms 10000]

Arms (engine.py's phase tags; tokens a second are summed over the engines, each over its own steps in the arm):
  plain            plain-baseline and plain: the plain weights, before the prover touches a GPU
  codec            decode-only, between and after: the arenas encoded, every weight decoded at use, no pass
  receipts+codec   session-*: the audit's passes (sketch-sha512 receipts over every arena) beside decode at use
  calibrate        the calibration's back-to-back passes beside decode at use (reported, not an arm of the loop)
Overheads are against plain in the same run: tpot_overhead_pct = 100 (TPOT_arm / TPOT_plain - 1) on the mean TPOT, and
throughput_overhead_pct = 100 (1 - tok/s_arm / tok/s_plain).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

BIN_MS = 0.1
ARMS = {"plain": ("plain-baseline", "plain"), "codec": ("decode-only", "between", "after"),
        "receipts+codec": ("session-",), "calibrate": ("calibrate",)}


def arm_of(phase: str) -> str | None:
    for arm, tags in ARMS.items():
        if phase in tags or any(t.endswith("-") and phase.startswith(t) for t in tags):
            return arm
    return None


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


def arms(out: Path) -> dict:
    """Each arm's tokens a second (summed over engines), busy fraction, batch, and merged latency histograms."""
    acc: dict[str, dict] = {}
    for f in sorted(out.glob("engine.*.latency.json")):
        lat = json.loads(f.read_text())
        for phase, p in lat["phases"].items():
            arm = arm_of(phase)
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


def step_ms(out: Path) -> dict:
    """Each arm's engine step time, p50 and p99, as the median over engines of engine.py's per-phase percentiles."""
    by: dict[str, dict[str, list]] = {}
    for f in sorted(out.glob("engine.[0-9]*.json")):
        if f.name.endswith(".latency.json"):
            continue
        for phase, p in json.loads(f.read_text()).get("phases", {}).items():
            arm = arm_of(phase)
            if arm:
                x = by.setdefault(arm, {"p50": [], "p99": []})
                x["p50"].append(p["step_ms"]["p50"])
                x["p99"].append(p["step_ms"]["p99"])
    med = lambda xs: round(sorted(xs)[len(xs) // 2], 2) if xs else None
    return {arm: {"step_ms_p50": med(x["p50"]), "step_ms_p99": med(x["p99"])} for arm, x in by.items()}


def passes(prover: dict, window_ms: float | None) -> dict:
    cal = prover.get("calibration") or {}
    out = {"calibrate": {"passes_per_s": round(1e9 / cal["pass_ns_median"], 2) if cal.get("pass_ns_median") else None,
                         "pass_ms_p50": round(cal["pass_ns_median"] / 1e6, 3) if cal.get("pass_ns_median") else None,
                         "pass_gpu_ms_p50": cal.get("pass_gpu_ms_median"), "read_ms": cal.get("read_ms"),
                         "slow_passes": cal.get("slow_passes"), "tau": cal.get("tau")}}
    sessions = []
    for s in prover.get("sessions", []):
        n, w = sum(s["passes"]), len(s["passes"])
        host = sorted(s.get("pass_host_ns") or [0])
        sessions.append({"k": s["k"], "mode": s["mode"], "accepted": s["accepted"], "passes": n, "windows": w,
                         "passes_per_s": round(n / (w * window_ms / 1e3), 2) if window_ms else None,
                         "pass_host_ms_p50": round(host[len(host) // 2] / 1e6, 3),
                         "pass_gpu_ms_p50": (s.get("device") or {}).get("pass_gpu_ms_median")})
    out["sessions"] = sessions
    honest = [s for s in sessions if s["mode"] == "honest" and s["passes_per_s"]]
    out["receipts+codec"] = {"passes_per_s": round(sum(s["passes_per_s"] for s in honest) / len(honest), 2) if honest else None,
                             "pass_ms_p50": round(sorted(s["pass_host_ms_p50"] for s in honest)[len(honest) // 2], 3)
                             if honest else None}
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
    table = arms(a.out)
    for arm, s in step_ms(a.out).items():
        table.setdefault(arm, {}).update(s)
    pz = passes(prover, a.window_ms)
    for arm in ("calibrate", "receipts+codec"):
        if arm in table:
            table[arm].update({k: v for k, v in pz[arm].items()})
    base = table.get("plain", {})
    for arm, t in table.items():
        if arm != "plain" and base.get("tpot_mean_ms") and t.get("tpot_mean_ms"):
            t["tpot_overhead_pct"] = round(100 * (t["tpot_mean_ms"] / base["tpot_mean_ms"] - 1), 2)
        if arm != "plain" and base.get("tokens_per_s") and t.get("tokens_per_s"):
            t["throughput_overhead_pct"] = round(100 * (1 - t["tokens_per_s"] / base["tokens_per_s"]), 2)
    gates = prover.get("gates") or {}
    engines_ok = all(e.get("arena_decodes") and e.get("probe_tokens_equal") for e in gates.get("engines", [])) \
        and bool(gates.get("engines"))
    honest = [s for s in pz["sessions"] if s["mode"] == "honest"]
    doc = {"schema": "verity/hash-memsat/v0", "arms": table, "sessions": pz["sessions"], "gates": gates,
           "engines": [{k: e.get(k) for k in ("gpu", "uuid", "bytes", "weights_bytes", "weights_blocks", "params", "model")}
                       for e in prover.get("engines", [])], "error": prover.get("error")}
    if a.verifier:
        doc["gamma"] = gammas(json.loads(a.verifier.read_text()))
    (a.out / "memsat.json").write_text(json.dumps(doc, indent=1) + "\n")
    errors = []
    if prover.get("error"):
        errors.append(f"prover: {prover['error']}")
    if not engines_ok:
        errors.append("an engine's decode gate failed or is missing")
    if not honest or not all(s["accepted"] for s in honest):
        errors.append("no accepted honest session")
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
               "validation": {"status": status, "checker": "engines' decode gates (arena SHA-256, probe tokens), the honest "
                              "sessions accepted by the verifier", "detail": "; ".join(errors) or "gates passed", "evidence": []},
               "measurements": meas, "measurement_files": ["out/memsat.json"]}
        a.result.write_text(json.dumps(res, indent=1) + "\n")
    for arm, t in table.items():
        print(arm, json.dumps({k: v for k, v in t.items() if not isinstance(v, list)}))
    print("errors:", errors or "none")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
