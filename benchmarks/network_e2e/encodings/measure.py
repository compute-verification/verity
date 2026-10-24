"""Bits per window of each code on a corpus of traces: fit on a training split, measured on a test split.

    python measure.py synth --out DIR [--rates 0.5,2,6] [--windows 5] [--train 2] [--test 2]
    python measure.py corpus TRACE.jsonl ... --out DIR [--train-share 0.5]

Each run fits the codebook on the training traces (the step-time model, its phase, and every table), registers it
(`codebook.json`, with its SHA-512 digest), and on every test trace encodes both subjects (v0's fields and the admission
residual) under every code, decodes them back (a mismatch fails the run), and records each window's bits. It writes
`table.json` (per trace and window, and the summary) and `table.md`, for `research data put --kind advice-encoding`.

The worst case of each code, per window, is reported two ways: at the window's public counts (its certified arrivals,
with every advice field at its registered cap), and at the caps (`arrivals_per_bucket · T` arrivals too).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if f"enc_{name}" in sys.modules:
        return sys.modules[f"enc_{name}"]
    spec = importlib.util.spec_from_file_location(f"enc_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("code")

SUBJECTS = ("v0", "admit")


# ------------------------------------------------------------------------------------------------- the step time


def _solve3(m: list[list[float]], y: list[float]) -> list[float]:
    """Gaussian elimination on a 3×3 system."""
    a = [row[:] + [v] for row, v in zip(m, y)]
    for i in range(3):
        p = max(range(i, 3), key=lambda r: abs(a[r][i]))
        a[i], a[p] = a[p], a[i]
        if abs(a[i][i]) < 1e-12:
            raise ValueError("singular step-time fit")
        for r in range(3):
            if r != i:
                f = a[r][i] / a[i][i]
                a[r] = [x - f * z for x, z in zip(a[r], a[i])]
    return [a[i][3] / a[i][i] for i in range(3)]


def fit_steptime(samples: list[tuple[float, float, float, float]]) -> tuple[int, int, int]:
    """Least squares of `y ≈ a·x0 + b·x1 + c·x2` over samples `(x0, x1, x2, y)`, rounded to integer µs."""
    m = [[sum(s[i] * s[j] for s in samples) for j in range(3)] for i in range(3)]
    y = [sum(s[i] * s[3] for s in samples) for i in range(3)]
    a, b, c = _solve3(m, y)
    return max(int(round(a)), 1), max(int(round(b)), 0), max(int(round(c)), 0)


def steptime_samples(trace: dict) -> list[tuple[float, float, float, float]]:
    """From step durations when the trace has them (1, tokens, context/1024 → µs); else from the arrivals: each busy gap
    between admissions, `Σ_k (1, tokens_k, context_k/1024)` over its steps against the gap's bucket time."""
    adv, pub, admit = trace["advice"], trace["public"], trace["admit"]
    loads = [(sum(x.n for x in s.rows), sum(x.start + x.n for x in s.rows) / 1024) for s in adv.steps]
    if trace.get("durations"):
        return [(1.0, t, c, d) for (t, c), d in zip(loads, trace["durations"])]
    rep, out = C.TraceReplay(adv), []
    for i in range(1, len(admit)):
        g, db = admit[i] - admit[i - 1], pub.anchors[i] - pub.anchors[i - 1]
        if db and 0 < g <= rep.end(i) - admit[i - 1]:
            ks = range(admit[i - 1], admit[i])
            out.append((float(g), sum(loads[k][0] for k in ks), sum(loads[k][1] for k in ks), db * trace["tau_us"]))
    return out


# ---------------------------------------------------------------------------------------------------- fitting


def training_symbols(traces: list[dict], cb) -> dict[str, list]:
    out: dict[str, list] = {"static": [], "cond": [], "cond+adapt": []}
    for t in traces:
        rep = C.TraceReplay(t["advice"])
        for fam in ("static", "cond"):
            out[fam] += C.symbols_v0(t["advice"], t["public"], cb, fam)
        for fam in ("static", "cond", "cond+adapt"):
            out[fam] += C.symbols_admit(t["admit"], t["public"], cb, fam, rep)
    return out


def fit(traces: list[dict], caps, T: int, tau_us: int) -> object:
    samples = [s for t in traces for s in steptime_samples(t)]
    a, b, c = fit_steptime(samples)
    best = None
    for phase in sorted({0, 1000, 2000, 4000, 8000, a // 2, a}):
        st = C.StepTime(a, b, c, phase)
        cb0 = C.Codebook(caps, tau_us, st)
        syms = training_symbols(traces, cb0)
        cb = C.fit_codebook(caps, T, tau_us, st, syms)
        bits = sum(C.model_bits(w, cb, fam) for fam in ("cond", "cond+adapt") for w in syms[fam]
                   if any(n.startswith("admit") for n, *_ in w))
        if best is None or bits < best[0]:
            best = (bits, cb)
    return best[1]


# ---------------------------------------------------------------------------------------------- worst cases


def worst_admit_arrival(cb, T: int, family: str) -> float:
    """The most bits one arrival's admission step costs, over every feasible value at the caps."""
    cap = cb.caps.steps_run(T)
    if family == "fixed":
        return C.worst_bits([("admit", 0, 0, cap)], cb, family)
    if family in ("gamma", "static"):
        return C.worst_bits([("admit_gap", 0, 0, cap)], cb, family) - (2 if family == "static" else 0)
    tables = C._tables_for(cb, family)
    return max(tables[f"admit_res_{ctx}"].worst(-gh, cap - gh) for ctx in C.ADMIT_CONTEXTS
               for gh in (0, 1, 4, 16, 64, 1024, cap // 2)) + C.OVERHEAD_PER_SYMBOL


def worst_v0_window(cb, T: int, family: str, arrivals: int) -> float:
    """The most bits a window of `arrivals` requests costs under v0's fields, every step and row at its cap."""
    caps = cb.caps
    span = caps.prompt_max + caps.output_max - 1
    req = [("prompt_len", 0, 1, caps.prompt_max), ("output_len", 0, 1, caps.output_max)]
    S, R = caps.steps_window(T), caps.seqs_max
    if family == "fixed":
        row = [("request", 0, 0, caps.requests_max - 1), ("kind", 0, 0, 1), ("start", 0, 0, span),
               ("n", 0, 0, caps.tokens_max), ("cache_hit", 0, 0, caps.prompt_max), ("preempted", 0, 0, 1)]
        per_req = C.worst_bits(req + [("finish", 0, 0, 1)], cb, family)
        per_step = C.worst_bits([("rows", 0, 0, R), ("release", 0, 0, T - 1)], cb, family) + R * C.worst_bits(row, cb, family)
        return arrivals * per_req + C.worst_bits([("steps", 0, 0, S)], cb, family) + S * per_step
    fam = "cond" if family.startswith("cond") else family
    row = [("request_delta", 0, -caps.requests_max, caps.requests_max), ("kind", 0, 0, 1), ("start_delta", 0, -span, span),
           ("n_prefill", 0, 0, caps.tokens_max), ("cache_hit", 0, 0, caps.prompt_max), ("preempted", 0, 0, 1)]
    if fam == "cond":
        per_req = C.worst_bits(req + [("finish_at_cap", 0, 0, 1)], cb, fam) - 2
        rel = max(C.worst_bits([(f"release_res_{q}", 0, -g, T - 1 - g)], cb, fam)
                  for q in range(C.PHASES + 1) for g in (0, T - 1)) - 2
        per_step = C.worst_bits([("canon", 0, 0, 1), ("rows", 0, 0, R)], cb, fam) - 2 + rel
    else:
        per_req = C.worst_bits(req + [("finish", 0, 0, 1)], cb, fam) - (2 if fam == "static" else 0)
        per_step = C.worst_bits([("rows", 0, 0, R), ("release_gap", 0, 0, T - 1)], cb, fam) - (2 if fam == "static" else 0)
    per_row = C.worst_bits(row, cb, fam) - (2 if fam != "gamma" else 0)
    return arrivals * per_req + C.worst_bits([("steps", 0, 0, S)], cb, fam) + S * (per_step + R * per_row)


# ---------------------------------------------------------------------------------------------------- measure


def measure(test: list[dict], cb) -> dict:
    rows = []
    for t in test:
        adv, pub, admit = t["advice"], t["public"], t["admit"]
        rep = C.TraceReplay(adv)
        counts = [len(pub.requests_in(w)) for w in range(pub.W)]
        entry = {"name": t["name"], "windows": pub.W, "requests": len(adv.requests), "steps": len(adv.steps),
                 "rows": sum(len(s.rows) for s in adv.steps), "arrivals_per_window": counts, "bits": {}}
        for fam in C.FAMILIES:
            bits, lens = C.encode_v0(adv, pub, cb, fam)
            back = C.decode_v0(C.BitReader(bits.bits), pub, cb, fam)
            if back.steps != adv.steps or back.release != adv.release or \
                    [(r.prompt_len, r.output_len, r.finish) for r in back.requests] != \
                    [(r.prompt_len, r.output_len, r.finish) for r in adv.requests]:
                raise SystemExit(f"{t['name']}: v0 under {fam} does not round-trip")
            entry["bits"][f"v0/{fam}"] = lens
            bits, lens = C.encode_admit(admit, pub, cb, fam, rep)
            if C.decode_admit(C.BitReader(bits.bits), pub, cb, fam, rep) != list(admit):
                raise SystemExit(f"{t['name']}: admit under {fam} does not round-trip")
            entry["bits"][f"admit/{fam}"] = lens
        rows.append(entry)
    return {"traces": rows}


def summarize(result: dict, cb, T: int) -> list[dict]:
    """One row per subject and code over every test window with an arrival or a step."""
    out = []
    caps_arrivals = cb.caps.arrivals_per_bucket * T
    for subj in SUBJECTS:
        for fam in C.FAMILIES:
            vals, worst_pub = [], []
            for t in result["traces"]:
                for w, b in enumerate(t["bits"][f"{subj}/{fam}"]):
                    a = t["arrivals_per_window"][w]
                    if b == 0 and a == 0:
                        continue
                    vals.append(b)
                    worst_pub.append(a * worst_admit_arrival(cb, T, fam) + (2 if fam not in ("fixed", "gamma") else 0)
                                     if subj == "admit" else worst_v0_window(cb, T, fam, a))
            arr = [a for t in result["traces"] for w, a in enumerate(t["arrivals_per_window"])
                   if a or t["bits"][f"{subj}/{fam}"][w]]
            caps_worst = (caps_arrivals * worst_admit_arrival(cb, T, fam) if subj == "admit"
                          else worst_v0_window(cb, T, fam, caps_arrivals))
            out.append({
                "subject": subj, "code": fam, "windows": len(vals),
                "mean_bits": statistics.fmean(vals), "p50_bits": statistics.median(vals), "max_bits": max(vals),
                "mean_bits_per_arrival": sum(vals) / max(sum(arr), 1),
                "worst_bits_public_counts_max": max(worst_pub), "worst_bits_caps": caps_worst})
    return out


def render(summary: list[dict], meta: dict) -> str:
    lines = [f"# Advice bits per window ({meta['corpus']})", "",
             f"Codebook `{meta['codebook_digest'][:16]}…` (SHA-512), fit on {meta['train']}; measured on {meta['test']}. "
             f"T = {meta['T']} buckets of {meta['tau_us'] // 1000} ms. Step-time model {meta['steptime']}.", "",
             "| subject | code | windows | mean bits/window | p50 | max | bits/arrival | worst, public counts (max) | worst, caps |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in summary:
        lines.append(f"| {r['subject']} | {r['code']} | {r['windows']} | {r['mean_bits']:,.0f} | {r['p50_bits']:,.0f} | "
                     f"{r['max_bits']:,} | {r['mean_bits_per_arrival']:,.2f} | {r['worst_bits_public_counts_max']:,.0f} | "
                     f"{r['worst_bits_caps']:,.0f} |")
    lines += ["", f"Reference: the step clock coded per bucket at the caps is T·log2(s_max + 1) = "
                  f"{meta['T'] * math.log2(meta['steps_per_bucket'] + 1):,.0f} bits per window; the clock sync is "
                  f"log2 #Σ_sync = 7 bits per window."]
    return "\n".join(lines) + "\n"


def run(train: list[dict], test: list[dict], caps, T: int, tau_us: int, out: Path, corpus: str) -> dict:
    t0 = time.perf_counter()
    cb = fit(train, caps, T, tau_us)
    out.mkdir(parents=True, exist_ok=True)
    (out / "codebook.json").write_text(json.dumps(cb.to_json(), sort_keys=True) + "\n")
    result = measure(test, cb)
    summary = summarize(result, cb, T)
    meta = {"corpus": corpus, "codebook_digest": cb.digest(), "train": [t["name"] for t in train],
            "test": [t["name"] for t in test], "T": T, "tau_us": tau_us, "steptime": vars(cb.steptime),
            "steps_per_bucket": caps.steps_per_bucket, "caps": vars(caps), "seconds": round(time.perf_counter() - t0, 1)}
    (out / "table.json").write_text(json.dumps({"meta": meta, "summary": summary, **result}, indent=1) + "\n")
    (out / "table.md").write_text(render(summary, meta))
    return {"meta": meta, "summary": summary}


def synth_corpus(rates: list[float], windows: int, seeds: list[int]) -> list[dict]:
    S = _load("synth")
    out = []
    for rate in rates:
        for seed in seeds:
            p = S.Params(rate=rate, windows=windows, seed=seed)
            adv, pub, admit, dur = S.generate(p, C)
            out.append({"name": f"synth-r{rate:g}-s{seed}", "advice": adv, "public": pub, "admit": admit,
                        "durations": dur, "tau_us": p.tau_us})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--rates", default="0.5,2,6")
    s.add_argument("--windows", type=int, default=5)
    s.add_argument("--train", type=int, default=2)
    s.add_argument("--test", type=int, default=2)
    a = ap.parse_args(argv)
    if a.cmd == "synth":
        S = _load("synth")
        rates = [float(x) for x in a.rates.split(",")]
        train = synth_corpus(rates, a.windows, [1000 + i for i in range(a.train)])
        test = synth_corpus(rates, a.windows, list(range(a.test)))
        p = S.Params()
        caps = C.Caps(prompt_max=p.prompt_max, output_max=p.max_tokens, seqs_max=p.max_num_seqs,
                      tokens_max=p.max_batched_tokens, steps_per_bucket=32)
        res = run(train, test, caps, p.T, p.tau_us, a.out, f"synthetic vLLM-V1-like engine, rates {a.rates} req/s")
        print((a.out / "table.md").read_text())
        print(f"{res['meta']['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
