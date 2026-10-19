"""Windows, routing, the dense-ladder fit and the decision rule of the routed-mixture probes, in the standard library.

``probe.py`` writes one record per window and model on a GPU (``prefix_bits``, ``span_bits``, ``span_bytes``); this
module turns those records into g, the routing gain in F3's units (the dense size that reaches a configuration's bits
per byte with unlimited data, over the weights one sequence uses), and g into D through ``benchmarks/nci/fit.py``. The
plan and the decision rule are the Notion page "Settling F3: routed-mixture experiments".
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import fit  # noqa: E402
import tiers  # noqa: E402

WINDOW_BYTES = 8192
PREFIX_BYTES = 1024

#: (alpha, beta) the ladder is fitted with at fixed exponents; ``free`` searches EXPONENT_GRID instead
EXPONENTS = {
    "besiroglu": (fit.FITS["besiroglu"]["alpha"], fit.FITS["besiroglu"]["beta"]),
    "hoffmann": (fit.FITS["hoffmann"]["alpha"], fit.FITS["hoffmann"]["beta"]),
    "kaplan-like": (0.30, 0.29),
}
EXPONENT_GRID = [round(0.10 + 0.01 * i, 2) for i in range(71)]

#: the dossier's line for F3 at N*/g: at or below it the composition gains nothing worth a claim change
NULL_GAIN = 1.2
#: the rung and premise the talk's claim is about, and the D it claims there
CLAIM = {"wipe_gb": 6.0, "b": 2.5, "D": 100.0}
RUNGS = ((6.0, 2.5), (6.0, 16.0), (13.0, 2.5), (13.0, 16.0))


def _boundary(data: bytes, i: int) -> int:
    """The largest UTF-8 character boundary at or below ``i``."""
    i = min(i, len(data))
    while 0 < i < len(data) and data[i] & 0xC0 == 0x80:
        i -= 1
    return i


def windows(doc_id: str, stratum: str, text: str, size: int = WINDOW_BYTES, prefix: int = PREFIX_BYTES) -> list[dict]:
    """A document's non-overlapping windows of at most ``size`` bytes, cut at character boundaries. Each window's first
    ``prefix`` bytes are context only: every model reads them, the prefix router chooses on them, and only the rest is
    scored. A tail shorter than half a window is dropped, so every scored span is at least ``size / 2 - prefix``
    bytes."""
    data = text.encode()
    out: list[dict] = []
    start = 0
    while len(data) - start >= size // 2:
        end = _boundary(data, start + size)
        cut = _boundary(data, start + prefix)
        out.append({"id": f"{doc_id}:{len(out)}", "doc": doc_id, "stratum": stratum, "text": data[start:end].decode(),
                    "prefix_chars": len(data[start:cut].decode())})
        start = end
    return out


def load_scores(path: Path) -> dict[str, dict]:
    """``probe.py score``'s records by window id."""
    with open(path) as f:
        return {r["window"]: r for r in map(json.loads, f)}


def compose(members: dict[str, dict[str, dict]], routing: str) -> dict[str, dict]:
    """Per-window records of a per-sequence composition: each window goes to one member.

    ``oracle`` picks the member with the fewest bits on the scored span and pays log2(members) bits for naming it, which
    makes the result a code length a decoder running one member can meet. ``prefix`` picks on the context-only prefix
    and pays nothing. Members must score the same span (one tokenizer)."""
    if routing not in ("oracle", "prefix"):
        raise ValueError(f"routing is 'oracle' or 'prefix', not {routing!r}")
    names = sorted(members)
    cost = math.log2(len(names)) if routing == "oracle" else 0.0
    field = "span_bits" if routing == "oracle" else "prefix_bits"
    out = {}
    for w in set.intersection(*(set(members[n]) for n in names)):
        recs = [members[n][w] for n in names]
        if len({r["span_bytes"] for r in recs}) != 1:
            raise ValueError(f"window {w}: the members score spans of different lengths; compose models that share a "
                             "tokenizer")
        best = min(range(len(recs)), key=lambda i: (recs[i][field], i))
        out[w] = {**recs[best], "span_bits": recs[best]["span_bits"] + cost, "choice": names[best]}
    return out


def bpb(records: dict[str, dict], ids: list[str]) -> float:
    """Bits per byte over the windows ``ids`` (repeats count, as a bootstrap sample needs)."""
    return sum(records[w]["span_bits"] for w in ids) / sum(records[w]["span_bytes"] for w in ids)


def _solve(M: list[list[float]], v: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting; None when ``M`` is singular."""
    n = len(v)
    a = [row[:] + [v[i]] for i, row in enumerate(M)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(a[r][c]))
        if abs(a[p][c]) < 1e-300:
            return None
        a[c], a[p] = a[p], a[c]
        for r in range(n):
            if r != c:
                k = a[r][c] / a[c][c]
                a[r] = [x - k * y for x, y in zip(a[r], a[c])]
    return [a[i][n] / a[i][i] for i in range(n)]


def _fit_fixed(points: list[tuple], alpha: float, beta: float, data_term: bool, offset: bool) -> dict | None:
    rows = [[1.0, p[0] ** -alpha] + ([p[1] ** -beta] if data_term else [])
            + ([float(p[3]) if len(p) > 3 else 0.0] if offset else []) for p in points]
    k = len(rows[0])
    M = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    v = [sum(r[i] * p[2] for r, p in zip(rows, points)) for i in range(k)]
    x = _solve(M, v)
    if x is None:
        return None
    sse = sum((sum(c * f for c, f in zip(x, r)) - p[2]) ** 2 for r, p in zip(rows, points))
    return {"E": x[0], "A": x[1], "B": x[2] if data_term else 0.0, "alpha": alpha, "beta": beta if data_term else None,
            "data_term": data_term, "anneal_offset": x[-1] if offset else 0.0, "rms": math.sqrt(sse / len(points))}


def fit_ladder(points: list[tuple], form: str, data_term: bool | None = None) -> dict:
    """Fit L = E + A N^-alpha + B D^-beta + c u (bits per byte) to dense ladder points (N weights, D tokens, L) or
    (N, D, L, u), where u = 1 marks a checkpoint taken before its learning-rate anneal.

    Unannealed checkpoints sit above the annealed curve, and fitting the data term through them alone would charge that
    offset to B D^-beta; with both kinds on the ladder, c absorbs it and the law describes annealed models (u = 0). The
    data term is fitted only when the ladder has at least two token counts (or ``data_term`` says so); otherwise
    B D^-beta folds into E and the law answers only at the ladder's own token count, which ``dense_size`` reports as
    unlimited data (flagged)."""
    if data_term is None:
        data_term = len({p[1] for p in points}) >= 2
    offset = len({bool(p[3]) if len(p) > 3 else False for p in points}) == 2
    if form == "free":
        if len({p[0] for p in points}) < 3 or (data_term and len({p[1] for p in points}) < 3):
            raise ValueError("free exponents need at least three sizes (and three token counts with a data term): "
                             "through two, every exponent fits exactly")
        fits = [f for a in EXPONENT_GRID for b in (EXPONENT_GRID if data_term else [None])
                if (f := _fit_fixed(points, a, b or 0.0, data_term, offset)) and f["A"] > 0 and f["B"] >= 0]
        if not fits:
            raise ValueError("no exponent pair on the grid gives A > 0 and B >= 0")
        law = min(fits, key=lambda f: f["rms"])
    else:
        law = _fit_fixed(points, *EXPONENTS[form], data_term, offset)
        if law is None:
            raise ValueError("the ladder's design matrix is singular: it needs at least two sizes")
    return {**law, "form": form}


def dense_size(law: dict, L: float, D: float = math.inf) -> float:
    """The dense size at which the law reaches ``L`` with ``D`` tokens; inf when no size does."""
    rest = L - law["E"] - (law["B"] * D ** -law["beta"] if law["data_term"] and math.isfinite(D) else 0.0)
    return (law["A"] / rest) ** (1 / law["alpha"]) if rest > 0 and law["A"] > 0 else math.inf


def gains(law: dict, L: float, P: float, D_train: float | None) -> dict:
    """g (unlimited-data dense equivalent over P) and G_same (dense on the configuration's own training tokens)."""
    same = dense_size(law, L, D_train) / P if D_train and law["data_term"] else None
    return {"g": dense_size(law, L) / P, "g_same": same}


def interval(vals: list[float]) -> tuple[float, float]:
    """The 95% percentile interval of bootstrap replicates."""
    vals = sorted(vals)
    return vals[int(0.025 * len(vals))], vals[max(int(0.975 * len(vals)) - 1, 0)]


def d_at_gain(g: float, wipe_gb: float, b: float, w_e: str = "central") -> dict:
    """D of the fit bound when one path holds N*/g weights (405B-class target, Besiroglu's law, T4 tier, m = 1), as the
    dossier's ``case()`` computes it; a void bound falls to PoUW's floor."""
    L = fit.target_loss("besiroglu", "405b")
    p = fit.params(wipe_gb)
    closed = tiers.evaluate(p, tiers.TIERS["T4"])
    R, declared = closed["wipe_state_bits"], closed["declared"]
    rate, periods = fit.fit_rate(p, R, fit.n_star("besiroglu", L) / g, fit.BOUNDARY[w_e], b)
    if not periods:
        return {"D": fit.dampener(p, math.inf, declared), "void": True}
    return {"D": fit.dampener(p, fit.share(p, rate), declared), "void": False}


def gain_at(target_D: float, wipe_gb: float, b: float) -> float:
    """The largest gain at which D still reaches ``target_D``."""
    lo, hi = 1.0, 1e3
    if d_at_gain(lo, wipe_gb, b)["D"] < target_D:
        return 0.0
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if d_at_gain(mid, wipe_gb, b)["D"] >= target_D else (lo, mid)
    return lo


def verdict(a: tuple[float, float] | None, b1: tuple[float, float] | None) -> dict:
    """The page's decision rule on A's and B1's 95% intervals of g.

    Only A's composition is trained in threads of P weights, so only A can drop the claim on its own; B1 above the line
    shows the room exists in a released model and hedges it. None is an experiment without a valid interval (missing,
    or a ladder that doesn't fall with size)."""
    line = gain_at(CLAIM["D"], CLAIM["wipe_gb"], CLAIM["b"])
    above = {k: iv is not None and iv[0] >= line for k, iv in (("A", a), ("B1", b1))}
    low = all(iv is not None and iv[1] <= NULL_GAIN for iv in (a, b1))
    missing = [k for k, iv in (("A", a), ("B1", b1)) if iv is None]
    if above["A"]:
        call = "drop"
        say = (f"drop 'D above {CLAIM['D']:.0f} at a {CLAIM['wipe_gb']:g} GB wipe' at {CLAIM['b']} bits; quote D at "
               "the measured gain and the 16-bit-premise numbers")
    elif above["B1"]:
        call = "hedge"
        say = ("F3 in words fails for a released MoE at B1's path sizes; training threads of that size are untested. "
                "Quote D at x2 and at B1's gain, and run the trend run with confinement-style paths")
    elif missing:
        call = "incomplete"
        say = f"no valid interval for {' and '.join(missing)}: nothing below the line can be concluded"
    elif low:
        call = "keep-margin"
        say = f"keep x2 as the stated margin: measured at most x{NULL_GAIN} at these path sizes"
    else:
        call = "keep-run-trend"
        say = "keep x2; run the mini trend run (C), then T1, to see whether g rises with path size"
    return {"call": call, "say": say, "claim_line": line, "null_gain": NULL_GAIN}


def analyze(manifest: dict, base: Path, resamples: int = 1000, seed: int = 0) -> dict:
    """Every study's configurations and the verdict over the configurations marked ``decides`` "A" or "B1" (the best
    lower end of g among each mark's)."""
    out = {"studies": {name: study(s, base, resamples, seed) for name, s in manifest["studies"].items()}}
    marked: dict[str, list[tuple[float, float]]] = {"A": [], "B1": []}
    for name, s in manifest["studies"].items():
        for kind in ("compositions", "singles"):
            for cname, c in s.get(kind, {}).items():
                if c.get("decides") in marked:
                    marked[c["decides"]].append(out["studies"][name]["configs"][cname]["g_interval"])
    best = {k: max((iv for iv in ivs if iv), key=lambda iv: iv[0], default=None) for k, ivs in marked.items()}
    out["verdict"] = {**verdict(best["A"], best["B1"]), "A": best["A"], "B1": best["B1"]}
    return out


def study(manifest: dict, base: Path, resamples: int, seed: int) -> dict:
    """One ladder's configurations: bits per byte overall and by stratum, g and G_same under each law form with a 95%
    interval, and D at g on each rung.

    ``manifest`` names score files (``scores``: names, each read from ``base/scores/<name>.jsonl``), the dense ladder
    (``ladder``: name -> {N, D, unannealed}), optionally ``data_term`` (see ``fit_ladder``), compositions
    (``compositions``: name -> {members, routing, P, D_train}) and single configurations (``singles``: name -> {P,
    D_train, code_bits, scores}). Only windows every file scores are used."""
    scores = {n: load_scores(base / "scores" / f"{n}.jsonl") for n in manifest["scores"]}
    configs: dict[str, tuple[dict, dict]] = {}
    for name, c in manifest.get("compositions", {}).items():
        configs[name] = (compose({m: scores[m] for m in c["members"]}, c["routing"]), c)
    for name, c in manifest.get("singles", {}).items():
        extra = c.get("code_bits", 0.0)
        recs = scores[c.get("scores", name)]
        configs[name] = ({w: {**r, "span_bits": r["span_bits"] + extra} for w, r in recs.items()}, c)
    ids = sorted(set.intersection(*(set(s) for s in scores.values()), *(set(r) for r, _ in configs.values())))
    if not ids:
        raise ValueError("no window is scored by every model in the manifest")
    ladder = manifest["ladder"]
    col = {n: ([scores[n][w]["span_bits"] for w in ids], [scores[n][w]["span_bytes"] for w in ids]) for n in ladder}
    cfg = {n: ([r[w]["span_bits"] for w in ids], [r[w]["span_bytes"] for w in ids]) for n, (r, _) in configs.items()}

    def at(column: tuple[list[float], list[int]], sample) -> float:
        bits, nbytes = column
        return sum(bits[i] for i in sample) / sum(nbytes[i] for i in sample)

    everything = range(len(ids))

    def law(form: str, sample) -> dict:
        return fit_ladder([(m["N"], m["D"], at(col[n], sample), m.get("unannealed", False)) for n, m in ladder.items()],
                          form, manifest.get("data_term"))

    laws = {form: law(form, everything) for form in EXPONENTS}
    try:
        laws["free"] = law("free", everything)
    except ValueError as e:
        free_error = str(e)
    else:
        free_error = None
    flat = [form for form in EXPONENTS if laws[form]["A"] <= 0]
    ladder_error = (f"the ladder's bits per byte don't fall with size under the {', '.join(flat)} exponents, so no "
                    "dense size can be read from it" if flat else None)
    # replicate-major: one resample of windows per replicate, shared by every model and law form, so differences pair
    reps: dict[str, dict[str, list[float]]] = {n: {form: [] for form in EXPONENTS} for n in configs}
    rng = random.Random(seed)
    for _ in range(0 if ladder_error else resamples):
        sample = [rng.randrange(len(ids)) for _ in ids]
        boot = {form: law(form, sample) for form in EXPONENTS}
        for n, (_, c) in configs.items():
            L = at(cfg[n], sample)
            for form, fitted in boot.items():
                reps[n][form].append(dense_size(fitted, L) / c["P"])

    out: dict = {"windows": len(ids), "resamples": resamples, "laws": laws, "free_error": free_error,
                 "ladder_error": ladder_error, "ladder": {n: at(col[n], everything) for n in ladder}, "configs": {}}
    strata = sorted({scores[next(iter(scores))][w]["stratum"] for w in ids})
    for name, (recs, c) in configs.items():
        L = bpb(recs, ids)
        row = {"bpb": L, "P": c["P"], "by_stratum": {s: bpb(recs, [w for w in ids if recs[w]["stratum"] == s])
                                                      for s in strata},
               "gains": {form: gains(fitted, L, c["P"], c.get("D_train")) for form, fitted in laws.items()}}
        if "choice" in next(iter(recs.values())):
            row["choices"] = {m: sum(recs[w]["choice"] == m for w in ids) for m in c["members"]}
        if ladder_error:
            row["g_interval"], row["D_at_g"] = None, None
            out["configs"][name] = row
            continue
        for form in EXPONENTS:
            row["gains"][form]["g_interval"] = interval(reps[name][form])
        row["g_interval"] = (min(row["gains"][f]["g_interval"][0] for f in EXPONENTS),
                             max(row["gains"][f]["g_interval"][1] for f in EXPONENTS))
        g = row["gains"]["besiroglu"]["g"]
        row["D_at_g"] = {f"{w:g} GB, b={b:g}": d_at_gain(g, w, b) if math.isfinite(g) else {"D": None, "void": True}
                         for w, b in RUNGS}
        out["configs"][name] = row
    return out
