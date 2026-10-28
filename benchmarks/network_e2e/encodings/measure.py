"""Bits per window of each code on a corpus of traces: fit on a training split, measured on a test split.

    python measure.py synth --out DIR [--rates 0.5,2,6] [--windows 5] [--train 2] [--test 2]
    python measure.py corpus MODEL_DIR ... --out DIR [--folds 2]

`synth` fits on synthetic traces of other seeds; `corpus` reads the trace corpus's traces (`trace_corpus.py`, each under
its server's own caps) and measures each fold under a codebook fit on the other folds (`codebooks/`).

Each run fits the codebook on the training traces (the step-time model, its phase, and every table), registers it
(`codebook.json`, with its SHA-512 digest), and on every test trace encodes each subject (v0's fields, the admission
steps, and the residual) under every code, decodes it back (a mismatch fails the run), and records each window's bits,
with the static and conditional codes' bits per field. Where the advice module is installed it also builds each trace's
advice there (which checks its meaning) and records the advice module's own layout and charge as a reference row. It
writes `table.json` (per trace and window, and the summary) and `table.md`, which go into the store as one
`advice-encoding/v1` tree.

The worst case of each code, per window, is reported two ways: at the window's public counts (its certified arrivals
and replayed resumes, with every advice field at its registered cap), and at the caps (`arrivals_per_bucket · T`
arrivals too).
"""

from __future__ import annotations

import argparse
import dataclasses
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

SUBJECTS = ("v0", "admit", "residual")


def adv_spec():
    """The advice module (I1 v0's canonical layout and its charge), or None where it isn't installed."""
    try:
        import verity.core.protocols.certifier.advice as A
    except ImportError:
        return None
    return A


def to_adv_spec(A, adv):
    """`advice.Advice` for this module's, which checks the advice's meaning as it is built."""
    return A.Advice(tuple(A.RequestAdvice(q.id, q.prompt_len, q.output_len, q.finish) for q in adv.requests),
                    tuple(A.StepAdvice(tuple(A.Row(x.request, x.kind, x.start, x.n, x.cache_hit, x.preempted,
                                                   tuple(A.Source(r, n) for r, n in x.sources)) for x in s.rows))
                          for s in adv.steps), adv.release)


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


def _cb_for(cb, t: dict):
    """The codebook under a trace's own serving caps (its server's registered config), where the trace has them."""
    return dataclasses.replace(cb, caps=t["caps"]) if t.get("caps") else cb


def v0_symbols(traces: list[dict], cb) -> dict[str, list]:
    """Each family's windows of v0's symbols on the training traces (they read no step-time phase)."""
    return {fam: [w for t in traces for w in C.symbols_v0(t["advice"], t["public"], _cb_for(cb, t), fam)]
            for fam in ("static", "cond")}


def residual_symbols(traces: list[dict], cb) -> dict[str, list]:
    """Each family's windows of the residual's symbols on the training traces, without the fields v0 codes too (the
    lengths and sources) under the families v0 shares tables with, so that each value is fit once. The admit subject's
    symbols are the residual's admission symbols."""
    out: dict[str, list] = {"static": [], "cond": [], "cond+adapt": []}
    for t in traces:
        rep = C.TraceReplay(t["advice"], t["public"])
        res = C.residual_of(t["advice"], t["admit"])
        for fam in out:
            wins = C.symbols_residual(res, t["public"], _cb_for(cb, t), fam, rep)
            if fam != "cond+adapt":
                wins = [[x for x in win if not x[0].startswith(("prompt_len", "output_len", "source"))] for win in wins]
            out[fam] += wins
    return out


def fit(traces: list[dict], caps, T: int, tau_us: int) -> object:
    """The codebook: the step-time model by least squares, then for each candidate phase every table fit to the
    training symbols, keeping the phase whose conditional residual costs least."""
    samples = [s for t in traces for s in steptime_samples(t)]
    a, b, c = fit_steptime(samples)
    v0 = v0_symbols(traces, C.Codebook(caps, tau_us, C.StepTime(a, b, c)))
    best = None
    for phase in sorted({0, 1000, 2000, 4000, 8000, a // 2, a}):
        st = C.StepTime(a, b, c, phase)
        res = residual_symbols(traces, C.Codebook(caps, tau_us, st))
        syms = {fam: v0.get(fam, []) + res[fam] for fam in res}
        cb = C.fit_codebook(caps, T, tau_us, st, syms)
        bits = sum(C.model_bits(w, cb, fam) for fam in ("cond", "cond+adapt") for w in res[fam])
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
    return max(max(tables[f"admit_res_{ctx}"].worst(-gh, cap - gh) for ctx in C.ADMIT_CONTEXTS
                   for gh in (0, 1, 4, 16, 64, 1024, cap // 2)), tables["admit_gap"].worst(0, cap)) + C.OVERHEAD_PER_SYMBOL


def worst_admit_flag(cb, family: str) -> float:
    """The most bits a window's admission mode costs (`code._admit_mode`: the conditional codes' one flag)."""
    if not family.startswith("cond"):
        return 0.0
    return C._tables_for(cb, family)["admit_predicted"].worst(0, 1) + C.OVERHEAD_PER_SYMBOL


def _worst_source(cb, family: str) -> float:
    """The most bits one source request costs (a keyed one: a pick among the earlier sources, or a step), without the
    segment's tail."""
    Q, tail = cb.caps.requests_max, 2 if family not in ("fixed", "gamma") else 0
    if family == "fixed":
        return C.worst_bits([("source", 0, 0, Q - 1)], cb, family)
    step = C.worst_bits([("source_delta", 0, -Q, Q)], cb, family) - tail
    if family in ("gamma", "static"):
        return step
    seen = C._tables_for(cb, family)["source_seen"]
    rank = C.worst_bits([("source_rank", 0, 0, Q - 1)], cb, family) - tail
    return C.OVERHEAD_PER_SYMBOL + max(seen.cost(0, 0, 1) + step, seen.cost(1, 0, 1) + rank)


def _worst_hit(cb, family: str, name: str, known_max: int) -> float:
    """The most bits a cache hit below `known_max` tokens costs with its most sources, without the segment's tail."""
    caps = cb.caps
    H, top = caps.sources_max, (known_max - 1) // caps.block_size
    syms = [(name, 0, 0, top), ("sources_more", 0, 0, H - 1)] + H * [("source_n", 0, 0, caps.prompt_max - 1)]
    if family.startswith("cond"):
        syms += [(f"{name}_any", 0, 0, 1), ("source_hit_same", 0, 0, 1)]
    return C.worst_bits(syms, cb, family) - (2 if family not in ("fixed", "gamma") else 0) + H * _worst_source(cb, family)


def worst_residual_window(cb, T: int, family: str, arrivals: int, resumes: int) -> float:
    """The most bits a window of `arrivals` requests and `resumes` resumes (both public: the certified ingress and the
    replay) costs under the residual: each field at its cap, each hit with the most sources."""
    caps = cb.caps
    lengths = C.worst_bits([("prompt_len", 0, 1, caps.prompt_max), ("output_len", 0, 1, caps.output_max)], cb, family)
    lengths -= 2 if family not in ("fixed", "gamma") else 0
    per_arrival = lengths + _worst_hit(cb, family, "hit", caps.prompt_max) + worst_admit_arrival(cb, T, family)
    per_resume = _worst_hit(cb, family, "rehit", caps.prompt_max + caps.output_max)
    return (arrivals * per_arrival + resumes * per_resume + worst_admit_flag(cb, family) +
            (2 if family not in ("fixed", "gamma") else 0))


def worst_v0_window(cb, T: int, family: str, arrivals: int) -> float:
    """The most bits a window of `arrivals` requests costs under v0's fields: every step and row at its cap, each row
    keeping an output and hitting the cache with the most sources a hit can have."""
    caps = cb.caps
    fam = "cond" if family.startswith("cond") else family
    span, H, G, Q = caps.prompt_max + caps.output_max - 1, caps.prompt_max, caps.sources_max, caps.requests_max
    S, R, rel = caps.steps_window(T), caps.seqs_max, caps.windows_max * T - 1

    def seg(syms: list) -> float:  # a part of one segment: the arithmetic coder's two terminating bits are added once
        return C.worst_bits(syms, cb, fam) - (2 if fam not in ("fixed", "gamma") else 0)

    req = [("prompt_len", 0, 1, caps.prompt_max), ("output_len", 0, 1, caps.output_max)]
    if fam == "fixed":
        per_req = seg(req + [("finish", 0, 0, 1)])
        per_step = seg([("rows", 0, 0, R)])
        per_row = seg([("request", 0, 0, Q - 1), ("kind", 0, 0, 1), ("start", 0, 0, span), ("n", 0, 0, caps.tokens_max),
                       ("cache_hit", 0, 0, H), ("preempted", 0, 0, 1), ("sources_more", 0, 0, G - 1),
                       ("release", 0, 0, rel)]) + G * seg([("source", 0, 0, Q - 1), ("source_n", 0, 0, H - 1)])
    else:
        per_req = seg(req + [("finish_at_cap" if fam == "cond" else "finish", 0, 0, 1)])
        per_row = seg([("request_delta", 0, -Q, Q), ("kind", 0, 0, 1), ("start_delta", 0, -span, span),
                       ("n_prefill", 0, 0, caps.tokens_max), ("cache_hit", 0, 0, H), ("preempted", 0, 0, 1)])
        sources = seg([("sources_more", 0, 0, G - 1)]) + G * seg([("source_delta", 0, -Q, Q), ("source_n", 0, 0, H - 1)])
        if fam == "cond":
            per_step = seg([("canon", 0, 0, 1), ("rows", 0, 0, R)])
            new = [("new_rows_0", 0, 0, R), ("new_request", 0, 0, Q), ("chunk_new", 0, -caps.tokens_max, 0),
                   ("source_hit_same", 0, 0, 1)]
            sources = max(sources, seg([("sources_more", 0, 0, G - 1)]) + G * (seg([("source_n", 0, 0, H - 1)])
                                                                               + _worst_source(cb, fam)))
            per_row = max(per_row, seg([("cont_prefill", 0, 0, 1), ("chunk_cont", 0, -caps.tokens_max, 0)]),
                          seg(new + [("cache_hit_new_any", 0, 0, 1), ("cache_hit_new", 0, 0, H // caps.block_size)]),
                          seg(new + [("resumed", 0, 0, 1), ("rehit_new_any", 0, 0, 1),
                                     ("rehit_new", 0, 0, span // caps.block_size)])) + sources
            per_row += max(seg([(name, 0, -g, rel - g)]) for name in [f"release_res_{q}" for q in range(C.PHASES + 1)]
                           + ["release_same"] for g in (0, rel // 2, rel))
        else:
            per_step = seg([("rows", 0, 0, R)])
            per_row += sources + seg([("release_gap", 0, -rel, rel)])
    tail = 2 if fam not in ("fixed", "gamma") else 0
    return arrivals * per_req + seg([("steps", 0, 0, S)]) + S * (per_step + R * per_row) + tail


# ---------------------------------------------------------------------------------------------------- measure


def _field(name: str) -> str:
    for p in ("release_res_", "admit_res_", "new_rows_"):
        if name.startswith(p):
            return p[:-1]
    return name


def by_field(windows: list, cb, family: str) -> dict[str, float]:
    """The model's bits per field over these windows of symbols (the arithmetic coder comes within rounding of them)."""
    tables, out = C._tables_for(cb, family), {}
    for win in windows:
        for name, v, lo, hi in win:
            out[_field(name)] = out.get(_field(name), 0.0) + tables[name].cost(v, lo, hi)
    return out


FIELDS = (("v0", "cond"), ("residual", "static"), ("residual", "cond"))
#: The decomposition each code shares: gamma splits a window into static's symbols, cond+adapt into cond's.
DECOMPOSITION = {"fixed": "fixed", "gamma": "static", "static": "static", "cond": "cond", "cond+adapt": "cond"}


def _symbols(subj: str, fam: str, adv, pub, admit, res, rep, cb) -> list:
    if subj == "v0":
        return C.symbols_v0(adv, pub, cb, fam)
    if subj == "admit":
        return C.symbols_admit(admit, pub, cb, fam, rep)
    return C.symbols_residual(res, pub, cb, fam, rep)


def _timed(f, *args):
    t0 = time.perf_counter()
    out = f(*args)
    return out, time.perf_counter() - t0


def measure(test: list[dict], cb) -> dict:
    """Each test trace's bits per window under every subject and code, after decoding each back (and the decoder's
    seconds, this module's pure Python, which the verifier's port would run in a small fraction of)."""
    A = adv_spec()
    rows = []
    for t in test:
        adv, pub, admit, cbt = t["advice"], t["public"], t["admit"], _cb_for(cb, t)
        rep = C.TraceReplay(adv, pub)
        res = C.residual_of(adv, admit)
        entry = {"name": t["name"], "windows": pub.W, "requests": len(adv.requests), "steps": len(adv.steps),
                 "rows": sum(len(s.rows) for s in adv.steps), "outputs": len(adv.release),
                 "arrivals_per_window": [len(pub.requests_in(w)) for w in range(pub.W)],
                 "resumes_per_window": [len(rep.resumes_in(w)) for w in range(pub.W)], "bits": {}, "fields": {},
                 "decode_seconds": {}}
        for fam in C.FAMILIES:
            bits, lens = C.encode_v0(adv, pub, cbt, fam)
            back, dt = _timed(C.decode_v0, C.BitReader(bits.bits), pub, cbt, fam)
            if back.steps != adv.steps or back.release != adv.release or \
                    [(r.prompt_len, r.output_len, r.finish) for r in back.requests] != \
                    [(r.prompt_len, r.output_len, r.finish) for r in adv.requests]:
                raise SystemExit(f"{t['name']}: v0 under {fam} does not round-trip")
            entry["bits"][f"v0/{fam}"], entry["decode_seconds"][f"v0/{fam}"] = lens, dt
            bits, lens = C.encode_admit(admit, pub, cbt, fam, rep)
            back, dt = _timed(C.decode_admit, C.BitReader(bits.bits), pub, cbt, fam, rep)
            if back != list(admit):
                raise SystemExit(f"{t['name']}: admit under {fam} does not round-trip")
            entry["bits"][f"admit/{fam}"], entry["decode_seconds"][f"admit/{fam}"] = lens, dt
            bits, lens = C.encode_residual(res, pub, cbt, fam, rep)
            back, dt = _timed(C.decode_residual, C.BitReader(bits.bits), pub, cbt, fam, rep)
            if back != res:
                raise SystemExit(f"{t['name']}: the residual under {fam} does not round-trip")
            entry["bits"][f"residual/{fam}"], entry["decode_seconds"][f"residual/{fam}"] = lens, dt
        entry["symbols"] = {}
        for subj in SUBJECTS:
            for fam in sorted(set(DECOMPOSITION.values())):
                syms = _symbols(subj, fam, adv, pub, admit, res, rep, cbt)
                entry["symbols"][f"{subj}/{fam}"] = [sum(hi > lo for _, _, lo, hi in win) for win in syms]
                if (subj, fam) in FIELDS:
                    entry["fields"][f"{subj}/{fam}"] = by_field(syms, cbt, fam)
        if A is not None:
            by = A.bits_by_window(to_adv_spec(A, adv), pub.T)
            entry["reference"] = {"v0/adv-spec": [by.get(w, 0) for w in range(pub.W)]}
        rows.append(entry)
    return {"traces": rows}


def _counted(t: dict, key: str) -> list[int]:
    """The windows a row counts: those with an arrival or any bits under `key`."""
    return [w for w, b in enumerate(t["bits"].get(key) or t["reference"][key]) if b or t["arrivals_per_window"][w]]


def summarize(result: dict, cb, T: int) -> dict:
    """One row per subject and code over every test window with an arrival or any bits; the conditional and static
    codes' bits per field; the advice module's own layout where it is installed."""
    rows = []
    caps_arrivals = cb.caps.arrivals_per_bucket * T
    tail = {f: 2 if f not in ("fixed", "gamma") else 0 for f in C.FAMILIES}
    keys = [(s, f) for s in SUBJECTS for f in C.FAMILIES]
    if all("reference" in t for t in result["traces"]):
        keys.append(("v0", "adv-spec"))
    for subj, fam in keys:
        key = f"{subj}/{fam}"
        vals, arr, worst_pub, syms = [], [], [], []
        for t in result["traces"]:
            lens = t["bits"].get(key) or t["reference"][key]
            for w in _counted(t, key):
                a, m = t["arrivals_per_window"][w], t["resumes_per_window"][w]
                vals.append(lens[w])
                arr.append(a)
                if fam in DECOMPOSITION:
                    syms.append(t["symbols"][f"{subj}/{DECOMPOSITION[fam]}"][w])
                if fam != "adv-spec":
                    worst_pub.append(a * worst_admit_arrival(cb, T, fam) + worst_admit_flag(cb, fam) + tail[fam]
                                     if subj == "admit" else
                                     worst_residual_window(cb, T, fam, a, m) if subj == "residual" else
                                     worst_v0_window(cb, T, fam, a))
        caps_worst = None if fam == "adv-spec" else (
            caps_arrivals * worst_admit_arrival(cb, T, fam) + worst_admit_flag(cb, fam) + tail[fam] if subj == "admit" else
            worst_residual_window(cb, T, fam, caps_arrivals, 0) if subj == "residual" else
            worst_v0_window(cb, T, fam, caps_arrivals))
        decode = [t["decode_seconds"][key] for t in result["traces"] if key in t.get("decode_seconds", {})]
        rows.append({
            "subject": subj, "code": fam, "windows": len(vals),
            "mean_bits": statistics.fmean(vals), "p50_bits": statistics.median(vals), "max_bits": max(vals),
            "mean_bits_per_arrival": sum(vals) / max(sum(arr), 1),
            "decode_ms_per_window": 1000 * sum(decode) / sum(t["windows"] for t in result["traces"]) if decode else None,
            "symbols_mean": statistics.fmean(syms) if syms else None, "symbols_max": max(syms) if syms else None,
            "worst_bits_public_counts_max": max(worst_pub) if worst_pub else None, "worst_bits_caps": caps_worst,
            "worst_bits_per_resume": _worst_hit(cb, fam, "rehit", cb.caps.prompt_max + cb.caps.output_max)
            if subj == "residual" else None})
    fields = {}
    for subj, fam in FIELDS:
        key = f"{subj}/{fam}"
        n = sum(len(_counted(t, key)) for t in result["traces"])
        tot: dict[str, float] = {}
        for t in result["traces"]:
            for name, b in t["fields"][key].items():
                tot[name] = tot.get(name, 0.0) + b
        fields[key] = {name: b / n for name, b in sorted(tot.items(), key=lambda kv: -kv[1])}
    return {"rows": rows, "fields_mean_bits_per_window": fields}


def _num(v) -> str:
    return "–" if v is None else f"{v:,.0f}" if v < 1e9 else f"{v:.2e}"


def render(summary: dict, meta: dict) -> str:
    folds = meta.get("folds")
    fitted = (f"Codebook `{meta['codebook_digest'][:16]}…` (SHA-512) fit on every trace, which bounds the worst cases; "
              f"each trace's bits are under the codebook of its fold, fit on the other folds "
              f"({'; '.join(f'`{f['codebook_digest'][:16]}…` measures {f['test']}' for f in folds)}). ") if folds else (
              f"Codebook `{meta['codebook_digest'][:16]}…` (SHA-512), fit on {meta['train']}; measured on {meta['test']}. ")
    lines = [f"# Advice bits per window ({meta['corpus']})", "",
             fitted + f"T = {meta['T']} buckets of {meta['tau_us'] // 1000} ms. Step-time model {meta['steptime']}.", "",
             "| subject | code | windows | mean bits/window | p50 | max | bits/arrival | worst, public counts (max) | "
             "worst, caps | coded symbols/window (mean, max) | decode ms/window (Python) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary["rows"]:
        ms, sm, sx = r.get("decode_ms_per_window"), r.get("symbols_mean"), r.get("symbols_max")
        lines.append(f"| {r['subject']} | {r['code']} | {r['windows']} | {r['mean_bits']:,.0f} | {r['p50_bits']:,.0f} | "
                     f"{r['max_bits']:,} | {r['mean_bits_per_arrival']:,.2f} | {_num(r['worst_bits_public_counts_max'])} | "
                     f"{_num(r['worst_bits_caps'])} | {'–' if sm is None else f'{sm:,.0f}, {sx:,}'} | "
                     f"{'–' if ms is None else f'{ms:,.1f}'} |")
    per_resume = {r["code"]: r["worst_bits_per_resume"] for r in summary["rows"] if r["subject"] == "residual"}
    lines += ["", "The residual's worst case at the caps counts no resume; each resume adds at most "
              + ", ".join(f"{_num(v)} ({k})" for k, v in per_resume.items()) + " bits.", "",
              "A coded symbol is one with more than one feasible value: one step of the arithmetic decoder (an escape or a "
              "bin's offset adds one more), one prefix read for gamma, and a slice of the row for fixed.", "",
              "Mean bits per window by field (the model's ideal lengths; each segment adds at most 2 bits):", "",
              "| field | " + " | ".join(summary["fields_mean_bits_per_window"]) + " |",
              "|---|" + "---|" * len(summary["fields_mean_bits_per_window"])]
    names = list(dict.fromkeys(n for f in summary["fields_mean_bits_per_window"].values() for n in f))
    for n in names:
        lines.append(f"| {n} | " + " | ".join(f"{f[n]:,.1f}" if n in f else "" for f in
                                             summary["fields_mean_bits_per_window"].values()) + " |")
    lines += ["", f"Reference: the step clock coded per bucket at the caps is T·log2(s_max + 1) = "
                  f"{meta['T'] * math.log2(meta['steps_per_bucket'] + 1):,.0f} bits per window; the clock sync is "
                  f"log2 #Σ_sync = 7 bits per window."]
    return "\n".join(lines) + "\n"


def run(train: list[dict], test: list[dict], caps, T: int, tau_us: int, out: Path, corpus: str, folds: int = 0) -> dict:
    """Fit on `train` and measure `test`; or with `folds`, measure each of `test`'s folds under a codebook fit on the
    others (written to `codebooks/`), with `codebook.json` fit on all of `train` and bounding the worst cases."""
    t0 = time.perf_counter()
    cb = fit(train, caps, T, tau_us)
    out.mkdir(parents=True, exist_ok=True)
    (out / "codebook.json").write_text(json.dumps(cb.to_json(), sort_keys=True) + "\n")
    held = []
    if folds:
        result = {"traces": []}
        (out / "codebooks").mkdir(exist_ok=True)
        for k in range(folds):
            ids = range(k, len(test), folds)
            fcb = fit([t for j, t in enumerate(test) if j not in ids], caps, T, tau_us)
            (out / "codebooks" / f"fold-{k}.json").write_text(json.dumps(fcb.to_json(), sort_keys=True) + "\n")
            held.append({"codebook_digest": fcb.digest(), "test": [test[j]["name"] for j in ids]})
            result["traces"] += measure([test[j] for j in ids], fcb)["traces"]
    else:
        result = measure(test, cb)
    summary = summarize(result, cb, T)
    meta = {"corpus": corpus, "codebook_digest": cb.digest(),
            "train": f"held out: each fold under a codebook fit on the other {folds - 1}" if folds else
            [t["name"] for t in train], "test": [t["name"] for t in test], "folds": held, "T": T, "tau_us": tau_us,
            "steptime": vars(cb.steptime), "steps_per_bucket": caps.steps_per_bucket, "caps": vars(caps),
            "trace_caps": {t["name"]: vars(t["caps"]) for t in test if t.get("caps")},
            "seconds": round(time.perf_counter() - t0, 1)}
    (out / "table.json").write_text(json.dumps({"meta": meta, "summary": summary, **result}, indent=1) + "\n")
    (out / "table.md").write_text(render(summary, meta))
    return {"meta": meta, "summary": summary}


def synth_corpus_params(params: list) -> list[dict]:
    S = _load("synth")
    out = []
    for p in params:
        adv, pub, admit, dur = S.generate(p, C)
        out.append({"name": f"synth-r{p.rate:g}-s{p.seed}", "advice": adv, "public": pub, "admit": admit,
                    "durations": dur, "tau_us": p.tau_us})
    return out


def synth_corpus(rates: list[float], windows: int, seeds: list[int]) -> list[dict]:
    S = _load("synth")
    return synth_corpus_params([S.Params(rate=r, windows=windows, seed=s) for r in rates for s in seeds])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--rates", default="0.5,2,6")
    s.add_argument("--windows", type=int, default=5)
    s.add_argument("--train", type=int, default=2)
    s.add_argument("--test", type=int, default=2)
    c = sub.add_parser("corpus", help="the trace corpus's traces (trace_corpus.py): each model directory's traces")
    c.add_argument("roots", type=Path, nargs="+", help="a corpus run's model directory, <out>/corpus/<model>")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--folds", type=int, default=2)
    c.add_argument("--label", help="the corpus as the table names it (its art: id, say); default the roots' names")
    a = ap.parse_args(argv)
    if a.cmd == "corpus":
        TC = _load("trace_corpus")
        traces = [t for r in a.roots for t in TC.load_corpus(C, r)]
        if len(traces) < max(a.folds, 2):
            raise SystemExit(f"{len(traces)} traces: need at least {max(a.folds, 2)} for {a.folds} folds")
        caps = dataclasses.replace(traces[0]["caps"], **{f.name: max(getattr(t["caps"], f.name) for t in traces)
                                                         for f in dataclasses.fields(traces[0]["caps"])})
        label = a.label or ", ".join(r.name for r in a.roots)
        res = run(traces, traces, caps, 600, 100_000, a.out, f"trace-corpus vLLM traces ({label})", folds=a.folds)
        print((a.out / "table.md").read_text())
        print(f"{res['meta']['seconds']} s")
    if a.cmd == "synth":
        S = _load("synth")
        rates = [float(x) for x in a.rates.split(",")]
        train = synth_corpus(rates, a.windows, [1000 + i for i in range(a.train)])
        test = synth_corpus(rates, a.windows, list(range(a.test)))
        p = S.Params()
        caps = C.Caps(prompt_max=p.prompt_max, output_max=p.max_tokens, seqs_max=p.max_num_seqs,
                      tokens_max=p.max_batched_tokens, steps_per_bucket=32, block_size=16)
        res = run(train, test, caps, p.T, p.tau_us, a.out, f"synthetic vLLM-V1-like engine, rates {a.rates} req/s")
        print((a.out / "table.md").read_text())
        print(f"{res['meta']['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
