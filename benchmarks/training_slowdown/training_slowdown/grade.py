"""The grade: a config's slowdown S = p·ρ, from its jobs' records and its honest config's.

    python -m training_slowdown.grade --config attacks/NAME.toml --starts starts.toml --records DIR --honest DIR \\
        [--also NAME=DIR ...] [--out OUT]

A config (an attack's TOML file) names its `start`, its `target.gain`, its `policy`, its `honest` config and its
`seeds` (`search` and `final`). DIR holds one config's records as its jobs wrote them, `DIR/<seed>/<split>/run.json`
with `split` `search` or `final`; a traced record may come as its rows instead, `rows.jsonl[.gz]` in the same place (one
row per checkpoint, as `rows_of` builds them). `--honest` is the honest config's DIR, and each `--also` another
config's at the same start and gain. The grade reads no file of a served run. It prints the block (`text`) and
writes `OUT/slowdown.json` (`SCHEMA`):

- **the gain**, through the fit of the start's data (`fit`): L(C) = E + A·C^(−γ) through the registry's starts on that
  data, each at its training compute C0 and its loss. A loss's effective compute is the C at which the fit reaches it,
  and the gain from the start's loss l0 to a loss is the ratio of the two (`gain_from`), so the target at gain g is
  E + (l0 − E)·g^(−γ) (`target`), l0 being the start's loss on the record's own split;
- **the seeds** (`seed_at`): a seed reaches the gain at its first scored checkpoint at or below its target, and its
  |C| is the gates counted through that checkpoint (`counted`: the parts of `formalism.of_record`; a checkpoint of an
  invalid record, or with no loss, is unscored, with the reason); else the seed is not reached, at its last. A
  config's C is the mean over its seeds that reach, with their spread (`spread`);
- **H**, the least counted |C| reaching the gain: the honest config's C, and the least C over every config given
  (the attack, the honest one and `--also`'s);
- **p** = C/H against each, with its range over the seeds (C's least over H's greatest, to C's greatest over H's
  least). C and H are read on one split (`basis`): the final set where the attack and the honest config each have a
  final record, else the search split, and the block says which;
- **ρ**, the attack's own cost at the config's policy (`policy.named`): `formalism.published` at (c, k) on the traced
  search record, at its first checkpoint reaching the gain on the search split, else at its last, which the block
  says. An untraced search record, or a traced step `formalism.measure` refuses, has no ρ, and then there is no S;
- **S = p·ρ** against each H, a lower bound where ρ is one (`partial`), with what each factor rests on;
- **the bound of record** at the same policy (`calculator.bound`): what every partition of the attack's traced
  circuit pays at the least, beside S, what this attack pays, and never mixed into it;
- **per period** (`per_period`): the run's AND gates at S, C·ρ, and their share of one datacenter's and of the
  world's compute over a window, a week, a month and a year (`calculator.fleets`, `calculator.periods`), with the
  measured, proved and scaled parts named.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import sys
import tomllib
from pathlib import Path

import numpy as np

from . import VERSION, calculator, gates, policy
from . import formalism as F

SCHEMA = "slowdown/v1"
SPLITS = {"search": "the search split", "final": "the final set"}
FORM = "L(C) = E + A*C^-gamma"
#: the fit: its points per decade of C, the fewest it takes and the least factor in C they span, the count under which
#: it is provisional, its parameters (on no more points its residuals bound nothing), and the digits E, A and γ keep
PER_DECADE, FIT_MIN, FIT_SPAN, FIT_TARGET, PARAMS, DIGITS = 20, 3, 10.0, 5, 3, 6
#: γ's grid, refined about its best by golden section
GAMMAS = np.linspace(0.01, 2.0, 399)
#: a checkpoint's facts about its output that `formalism.measure` reads, where the record has them
OUTPUT = ("e_bytes", "format", "rebuild_macs", "decoder_gates")
#: the theorem's inputs on a traced step, as `formalism.he_products` names them
WORK = ("N", "O", "beta_w", "beta_x", "b_y", "F")


# ---- the fit -----------------------------------------------------------------------------------------------------------

def _num(v: float, digits: int = DIGITS) -> float:
    return float(f"{v:.{digits}g}")


def fit(points: list[tuple[float, float]]) -> dict | None:
    """L(C) = E + A·C^(−γ) through `points` (C in gates, loss in bits per byte). It reads their frontier (by C, the loss
    strictly falling), one point per 1/`PER_DECADE` decade of C, and fits by least squares in relative loss: for each γ
    of `GAMMAS` and then by golden section about the best, (E, A) by weighted linear least squares, E held in [0, the
    least loss) and A > 0. E, A and γ are kept to `DIGITS` digits, the values every gain reads, and the residuals are
    theirs. None on fewer than `FIT_MIN` points or a span under `FIT_SPAN`; `provisional` on fewer than `FIT_TARGET`."""
    bins, least = {}, math.inf
    for c, l in sorted(points):
        if l < least:
            least, bins[math.floor(PER_DECADE * math.log10(c))] = l, (c, l)
    pts = sorted(bins.values())
    if len(pts) < FIT_MIN or pts[-1][0] < FIT_SPAN * pts[0][0]:
        return None
    c, l = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
    z, w, top = np.log(c / c.max()), 1 / l, l.min() * (1 - 1e-9)

    def solve(g: float) -> tuple[float, float, float, float] | None:
        x = np.exp(-g * z)
        (e, a), *_ = np.linalg.lstsq(np.stack([w, x * w], 1), np.ones_like(l), rcond=None)
        if not 0 <= e < top:
            e = min(max(float(e), 0.0), top)
            a = float(((l - e) * x * w * w).sum() / (x * x * w * w).sum())
        if a <= 0:
            return None
        r = (e + a * x) * w - 1
        return float(r @ r), g, float(e), float(a)

    tried = [t for g in GAMMAS if (t := solve(float(g)))]
    if not tried:
        return None
    step, phi = float(GAMMAS[1] - GAMMAS[0]), (math.sqrt(5) - 1) / 2
    lo, hi = max(1e-6, min(tried)[1] - step), min(tried)[1] + step
    for _ in range(60):
        m1, m2 = hi - phi * (hi - lo), lo + phi * (hi - lo)
        t1, t2 = solve(m1), solve(m2)
        tried += [t for t in (t1, t2) if t]
        if t2 is None or t1 is not None and t1[0] <= t2[0]:
            hi = m2
        else:
            lo = m1
    _, g, e, a = min(tried)
    out = {"form": FORM, "E": _num(e), "A": _num(a * float(c.max()) ** g), "gamma": _num(g), "points": len(pts),
           "C_lo": float(c.min()), "C_hi": float(c.max()), "provisional": len(pts) < FIT_TARGET}
    res = [100 * (y / (out["E"] + out["A"] * x ** -out["gamma"]) - 1) for x, y in pts]
    return out | {"rms_pct": _num(math.sqrt(sum(v * v for v in res) / len(res)), 3)}


def target(f: dict, l0: float, g: float) -> float | None:
    """The loss a gain of `g` from `l0` reaches on the fit, E + (l0 − E)·g^(−γ); None where `l0` is at or below E, which
    no compute reaches."""
    return f["E"] + (l0 - f["E"]) * g ** -f["gamma"] if l0 > f["E"] else None


def gain_from(f: dict, l0: float, loss: float) -> float:
    """The gain from `l0` (above E) to `loss` on the fit: the effective compute of `loss` over that of `l0`,
    ((l0 − E)/(loss − E))^(1/γ), infinite at or below E."""
    return ((l0 - f["E"]) / (loss - f["E"])) ** (1 / f["gamma"]) if loss > f["E"] else math.inf


def c_spread(f: dict, loss: float) -> float | None:
    """The fit's uncertainty in effective compute at `loss`, as a share: the change in C that moves the fit's loss by
    its rms residual, rms·loss/(γ·(loss − E)); None on no more points than its `PARAMS` parameters, whose residuals
    bound nothing."""
    return None if f["points"] <= PARAMS else f["rms_pct"] / 100 * loss / (f["gamma"] * (loss - f["E"]))


# ---- a job's checkpoints ---------------------------------------------------------------------------------------------

def macs_log(record: dict, through: int | None) -> dict:
    """A record's MACs by class (`by`) and by op and shape (`ops`: [count, MACs]) over its setup and its steps through
    `through` (None: the whole run, its tail too): what `formalism` checks a trace's products against."""
    by, ops = {}, {}
    steps = [record["setup"], *(s for s in record["steps"] if through is None or s["step"] <= through)]
    if through is None:
        steps.append(record.get("tail") or {})
    for s in steps:
        for k, v in s.get("by", {}).items():
            by[k] = by.get(k, 0) + v
        for row in (record["structures"].get(s.get("struct")) or {}).get("log", []):
            n, m = ops.get(key := " ".join(map(str, row[:4])), (0, 0))
            ops[key] = [n + row[4], m + row[5]]
    return {"by": by, "ops": ops}


def rows_of(record: dict) -> list[dict]:
    """One row per checkpoint of a run record, as `formalism.measure` reads one: its step, its loss (`val_bpb`, the
    checkpoint's `score.bpb`), the start's loss on the same split (`start_bpb`, its `score.start_bpb`), the split, the
    `formalism` block and op log of the run's prefix through it, its output's facts (`OUTPUT`) and the record's
    `invalid`."""
    last = record["steps"][-1]["step"] if record["steps"] else -1
    out = []
    for cp in record["checkpoints"]:
        through, score = None if cp["step"] >= last else cp["step"], cp.get("score") or {}
        out.append({"checkpoint_step": cp["step"], "val_bpb": score.get("bpb", cp.get("bpb", cp.get("val_bpb"))),
                    "start_bpb": score.get("start_bpb", record["start"].get("val_bpb")),
                    "split": score.get("split", record.get("split")), "start": record["start"].get("name"),
                    "formalism": F.of_record(record, through), "macs_log": macs_log(record, through),
                    "output": {k: cp[k] for k in OUTPUT if k in cp}, "invalid": record.get("invalid") or []})
    return out


def counted(row: dict) -> float | None:
    """A row's counted |C| in gates: its `formalism` block's parts' (`formalism.of_record`); None without them."""
    parts = (row.get("formalism") or {}).get("parts")
    return sum(p["gates"] for p in parts) if parts else None


def checkpoints(run: dict, start: str) -> list[dict]:
    """A job's checkpoints by step (`run`: its `seed`, its `split` and its `record`, or its `rows`): each one's loss,
    the start's loss on the job's split and its counted |C| in gates (`counted`), or why it has none. A job from
    another start, or scored on another split than its place says, is refused."""
    rows = rows_of(run["record"]) if run.get("rows") is None else run["rows"]
    where = f"seed {run['seed']}, {run['split']}"
    out = []
    for r in sorted(rows, key=lambda r: r["checkpoint_step"]):
        if r.get("start") not in (None, start):
            raise ValueError(f"grade: the record at {where} is from start {r['start']!r}, not the config's {start!r}")
        if r.get("split") not in (None, run["split"]):
            raise ValueError(f"grade: the record at {where} was scored on {r['split']!r}")
        why = ("invalid: " + "; ".join(map(str, r["invalid"])) if r.get("invalid") else
               "unscored" if r.get("val_bpb") is None else None if counted(r) else "uncounted")
        out.append({"step": r["checkpoint_step"], "bpb": r.get("val_bpb"), "l0": r.get("start_bpb"),
                    "gates": None if why else counted(r), "row": r} | ({"why": why} if why else {}))
    return out


# ---- seeds, H and p ----------------------------------------------------------------------------------------------------

def seed_at(cps: list[dict], f: dict, g: float) -> dict:
    """One seed at gain `g`: its l0 (the start's loss on its split), its target, and its first scored checkpoint at or
    below it (`reached`, with its step, loss, |C| and gain); else not reached, at its last scored checkpoint."""
    l0 = next((c["l0"] for c in cps if c.get("l0") is not None), None)
    if l0 is None:
        return {"reached": False, "why": "its record has no start loss" if cps else "no checkpoint"}
    t = target(f, l0, g)
    if t is None:
        return {"l0": l0, "reached": False, "why": f"its l0 {l0:g} is at or below the fit's E {f['E']:g}"}
    scored = [c for c in cps if c["gates"] is not None]
    if not scored:
        return {"l0": l0, "target": t, "reached": False,
                "why": "no scored checkpoint: " + "; ".join(sorted({c.get("why", "") for c in cps}))}
    hit = next((c for c in scored if c["bpb"] <= t), None)
    c = hit or scored[-1]
    return {"l0": l0, "target": t, "reached": hit is not None, "step": c["step"], "bpb": c["bpb"],
            "C_gates": c["gates"], "gain": gain_from(f, l0, c["bpb"])}


def spread(xs: list[float]) -> dict:
    return {"mean": statistics.fmean(xs), "sd": statistics.stdev(xs) if len(xs) > 1 else None, "min": min(xs),
            "max": max(xs), "n": len(xs)}


def side(name: str, runs: list[dict], f: dict, g: float, planned: dict | None = None) -> dict:
    """A config at gain `g`: per split, each seed (`seed_at`; a planned seed with no record says so), how many reach
    and C, the mean |C| over those that do with its spread (None where none does)."""
    out = {"config": name, "splits": {}}
    for split in SPLITS:
        seeds = {str(r["seed"]): seed_at(r["checkpoints"], f, g) for r in runs if r["split"] == split}
        for s in (planned or {}).get(split, ()):
            seeds.setdefault(str(s), {"reached": False, "why": "no record"})
        hit = [v["C_gates"] for v in seeds.values() if v["reached"]]
        out["splits"][split] = {"seeds": seeds, "reached": len(hit), "of": len(seeds), "C": spread(hit) if hit else None}
    return out


def ratio(c: dict | None, h: dict | None) -> dict | None:
    """p = C/H at their means, with its range over the seeds."""
    return c and h and {"p": c["mean"] / h["mean"], "range": [c["min"] / h["max"], c["max"] / h["min"]]}


# ---- ρ, the bound of record and the periods --------------------------------------------------------------------------

def rho_of(run: dict | None, f: dict, g: float, pol: dict, shape: dict) -> dict:
    """ρ at the policy `pol` from the config's search job `run`, a start of `shape`: `formalism.published` at (c, k) on
    its first scored checkpoint at or below its target, else on its last (`at` says which), with the theorem's inputs
    there (`work`, where its products term has them) and the capture that traced it. No ρ, with why, without a search
    record, without a scored checkpoint, on an untraced one, or on one `formalism.measure` refuses or can't measure."""
    if run is None:
        return {"rho": None, "why": "unmeasured: the config has no search record"}
    at = seed_at(run["checkpoints"], f, g)
    cp = next((c for c in run["checkpoints"] if c["gates"] is not None and c["step"] == at.get("step")), None)
    if cp is None:
        return {"rho": None, "why": f"unmeasured: the search record has no scored checkpoint ({at.get('why')})"}
    if (cp["row"].get("formalism") or {}).get("trace") is None:
        return {"rho": None, "why": "unmeasured: the search record is untraced, and ρ reads a traced step"}
    fwd = (run.get("record") or {}).get("forward_macs_per_token")
    try:
        m = F.measure(cp["row"], shape | ({"forward_macs_per_token": fwd} if fwd else {}))
    except F.Refused as e:
        return {"rho": None, "why": f"refused: {e}"}
    except ValueError as e:
        return {"rho": None, "why": f"unmeasured: {e}"}
    c, k = pol["c_gates"], pol["k_bits"]
    term, _, got = F.he_products(m, c, k)
    return F.published(m, c, k) | {
        "seed": run["seed"], "checkpoint_step": cp["step"], "C_gates": cp["gates"],
        "at": "its first checkpoint reaching the gain" if at["reached"] else
              f"its last checkpoint: the search record doesn't reach {g:g}x",
        "capture": (m.products or {}).get("capture"),
        "work": None if term is None else {x: got.get(x) for x in WORK} | {"C_gates": cp["gates"]}}


def per_period(run_gates: float, rho: dict) -> dict:
    """The run's gates at S (`run_gates`, C·ρ) in AND gates, at a served-precision GEMM's gates an AND, and their share
    of each fleet's compute over each period (`calculator.fleets`, `calculator.periods`), with what is measured, what
    is proved and what is scaled."""
    run_and = run_gates / gates.per_and(f"gemm/{policy.SERVED_PRECISION}")
    rows = [{"over": fl["over"], "fleet": fl["fleet"], "locations": fl["locations"], "period": name, "seconds": s,
             "fleet_and_gates": fl["and_per_s"] * s, "share": run_and / (fl["and_per_s"] * s)}
            for fl in calculator.fleets() for name, s in calculator.periods().items()]
    return {"run_and_gates": run_and, "window": calculator.window()[0], "rows": rows, "parts": {
        "measured": "|C| of each seed and H, the counted gates of the records read, so p; ρ's inputs, one traced step "
                    "of the search record",
        "proved": (f"the form of ρ's products term, {calculator.THEOREM}'s second conjunct (`HEShared`'s where its "
                   "products share arrays): a lower bound at the N, O, β_w, β_x and b_y the traced step gives, given "
                   "(H), (E) and the theorem's hypotheses on the circuit, which the trace doesn't check; p and the "
                   "shares are not proved" if rho["binding_kind"] == "products" else
                   f"nothing: ρ's {rho['binding_kind']} term is computed"),
        "scaled": f"every share: a fleet's AND gates are its registry's peak FLOP a second, two a MAC at the "
                  f"{calculator.FLEET_PRECISION} GEMM's AND gates, times its locations and the period, and the run's "
                  f"gates are AND gates at the {policy.SERVED_PRECISION} GEMM's ratio. The fleets' sizes and the "
                  f"window are stated parameters"}}


# ---- the grade ---------------------------------------------------------------------------------------------------------

def grade(config: dict, starts: dict, runs: list[dict], honest: list[dict],
          others: dict[str, list[dict]] | None = None) -> dict:
    """The slowdown of `config` (its `name`, `start`, `target.gain`, `policy`, `honest` and `seeds`) from its jobs
    `runs`, its honest config's (`honest`) and other configs' at the same start and gain (`others`, by name), each job
    its `seed`, its `split` and its `record` or `rows`; `starts` is the starts registry (module docstring)."""
    starts = starts.get("starts", starts)
    if config["start"] not in starts:
        raise ValueError(f"grade: no start {config['start']!r} in the starts registry ({', '.join(starts)})")
    entry, g = starts[config["start"]], float(config["target"]["gain"])
    pol = policy.named(config["policy"], entry["shape"])
    data = [n for n, e in starts.items() if e.get("data") == entry.get("data")]
    f = fit([(policy.c0(starts[n])[0], starts[n]["l0_bpb"]) for n in data])
    if f is None:
        raise ValueError(f"grade: the {len(data)} starts on data {entry.get('data')!r} give no fit: it takes "
                         f"{FIT_MIN} frontier points spanning a factor {FIT_SPAN:g} in C0")
    me, hon = config["name"], config["honest"]
    jobs = {n: [r | {"checkpoints": checkpoints(r, config["start"])} for r in rs]
            for n, rs in ({me: runs, hon: honest} | (others or {})).items()}
    sides = {n: side(n, rs, f, g, config.get("seeds") if n == me else None) for n, rs in jobs.items()}
    lacking = [n for n in (me, hon) if not any(r["split"] == "final" for r in jobs[n])]
    basis = "search" if lacking else "final"
    C, at = sides[me]["splits"][basis]["C"], {n: s["splits"][basis] for n, s in sides.items()}
    best = min((n for n in at if at[n]["C"]), key=lambda n: at[n]["C"]["mean"], default=None)
    H = {"honest": {"config": hon} | {k: at[hon][k] for k in ("C", "reached", "of")},
         "any": {"config": best, "over": list(at)} | {k: at[best][k] if best else None for k in ("C", "reached", "of")}}
    p = {k: ratio(C, h["C"]) for k, h in H.items()}
    wanted = (config.get("seeds") or {}).get("search") or [None]
    search = next((r for r in jobs[me] if r["split"] == "search" and wanted[0] in (None, r["seed"])), None)
    rho = rho_of(search, f, g, pol, entry["shape"])
    why = (rho.get("why") or "ρ is unmeasured") if rho["rho"] is None else \
        None if C else f"no seed of the config reaches {g:g}x on {SPLITS[basis]}"
    S = {k: None if why or not p[k] else p[k]["p"] * rho["rho"] for k in H} | {
        "why": why, "lower_bound": bool(rho.get("partial"))}
    return {
        "schema": SCHEMA, "training_slowdown": VERSION, "config": me, "start": config["start"], "gain": g,
        "policy": pol, "basis": {"split": basis, "why": (f"no final record of {' or '.join(lacking)}" if lacking else
                                                         "the attack and the honest config each have a final record")},
        "fit": f | {"data": entry.get("data"), "starts": data,
                    "c_spread": c_spread(f, target(f, entry["l0_bpb"], g))},
        "attack": sides[me], "honest": sides[hon], "others": [sides[n] for n in others or {}],
        "H": H, "p": p, "rho": rho, "S": S,
        "bound_of_record": calculator.bound(pol["c_gates"], pol["k_bits"], rho.get("work")),
        "rests_on": {
            "p": f"counted |C| ({gates.UNIT}) at each seed's first checkpoint at or below its target on "
                 f"{SPLITS[basis]}, the mean over the seeds that reach, over the same for H",
            "rho": rho.get("rests_on"), "capture": rho.get("capture"), "capture_code": rho.get("capture_code"),
            "gain": f"the fit {FORM} on {f['points']} of the {len(data)} starts on data {entry.get('data')}"
                    + (f", provisional under {FIT_TARGET}" if f["provisional"] else "")},
        "per_period": per_period(C["mean"] * rho["rho"], rho) if C and rho["rho"] is not None else None,
        "records": [_read(n, r) for n, rs in jobs.items() for r in rs]}


def _read(name: str, run: dict) -> dict:
    """What the grade read of one job: its place, its form, the label and the version of the code that wrote it, and
    how many of its checkpoints are scored."""
    wrote = run.get("record") or (run.get("rows") or [{}])[0]
    return {"config": name, "seed": run["seed"], "split": run["split"], "source": run.get("source"),
            "form": "rows" if "rows" in run else "record", "label": wrote.get("label"),
            "version": wrote.get("training_slowdown") or wrote.get("te_core"), "checkpoints": len(run["checkpoints"]),
            "scored": sum(c["gates"] is not None for c in run["checkpoints"])}


# ---- the block -------------------------------------------------------------------------------------------------------

def _seed(split: str, seed: str, s: dict) -> str:
    if "step" not in s:
        return f"{split} {seed}: {s['why']}"
    how = "reaches at" if s["reached"] else "not reached, at its last scored"
    return f"{split} {seed}: {how} step {s['step']}, gain {s['gain']:.4f}, |C| {s['C_gates']:.4g}"


def _h(h: dict) -> str:
    if not h["C"]:
        return f"{h['config'] or 'no config'}: none ({h['reached'] or 0} of {h['of'] or 0} seeds reach)"
    return f"{h['config']}: {h['C']['mean']:.4g} gates ({h['reached']} of {h['of']} seeds reach)"


def _p(p: dict | None) -> str:
    return "none" if p is None else f"{p['p']:.4g} [{p['range'][0]:.3g}, {p['range'][1]:.3g}]"


def text(s: dict) -> str:
    """The block the grade prints (module docstring)."""
    pol, f, rho, S, basis = s["policy"], s["fit"], s["rho"], s["S"], s["basis"]["split"]
    mine, per, bound = s["attack"]["splits"], s["per_period"], s["bound_of_record"]
    lines = [f"SLOWDOWN {s['config']}   start {s['start']}, gain {s['gain']:g}x, policy {pol['name']} ({pol['kind']}: "
             f"c = {pol['c_gates']:.4g} gates, k = {pol['k_bits']:,.0f} bits; {pol['why']})"]

    def add(head: str, *body: str) -> None:
        lines.extend(f"  {h:<17} {b}" for h, b in zip((head, *[""] * len(body)), body))

    if S["why"]:
        add("S = p x rho", f"none: {S['why']}")
    else:
        add("S = p x rho", *(f"against {name}: S {'>= ' if S['lower_bound'] else ''}{S[k]:,.6g} = p {s['p'][k]['p']:.4g}"
                             f" x rho {rho['rho']:,.6g}" if S[k] else f"against {name}: none (H: {_h(s['H'][k])})"
                             for k, name in (("any", "any route"), ("honest", "the honest route"))))
    spread_ = "the fit's residuals bound none of its uncertainty" if f["c_spread"] is None else \
        f"the fit's rms is {100 * f['c_spread']:.3g}% in effective compute at the target"
    add("gain", f"read on {SPLITS[basis]} ({s['basis']['why']}); {s['rests_on']['gain']}: E {f['E']:g}, A {f['A']:g}, "
                f"gamma {f['gamma']:g}; {spread_}")
    add("H", f"the honest route, {_h(s['H']['honest'])}",
        f"any route, over {len(s['H']['any']['over'])} configs ({', '.join(s['H']['any']['over'])}), {_h(s['H']['any'])}")
    C = mine[basis]["C"]
    add("p", f"against any route {_p(s['p']['any'])}; against the honest route {_p(s['p']['honest'])}; the attack's "
             + (f"|C| {C['mean']:.4g} gates" if C else "|C| none")
             + f" ({mine[basis]['reached']} of {mine[basis]['of']} seeds reach)")
    add("rho", f"none: {rho['why']}" if rho["rho"] is None else
        f"{rho['rho']:,.6g}{' (a lower bound: ' + str(rho['why'] or 'a kind is unmeasured') + ')' if rho['partial'] else ''}"
        f", binding {rho['binding_kind']}, on search seed {rho['seed']} at {rho['at']} (step {rho['checkpoint_step']})")
    seeds = [_seed(split, seed, v) for split in ("final", "search") for seed, v in mine[split]["seeds"].items()]
    theirs = [_seed(split, seed, v) for split in ("final", "search")
              for seed, v in s["honest"]["splits"][split]["seeds"].items()]
    add("seeds", *seeds, *(f"honest {x}" for x in theirs),
        "the spread of |C|: " + ("none" if not C else f"{C['min']:.4g} to {C['max']:.4g} gates over {C['n']} seeds"
                                  + (f", sd {C['sd']:.3g}" if C["sd"] is not None else "")))
    r = s["rests_on"]
    add("rests on", f"rho: {r['rho'] or 'nothing: it is unmeasured'}",
        f"capture: {r['capture'] or 'none'}, its code {r['capture_code'] or 'unrecorded'}", f"p: {r['p']}",
        f"gain: {r['gain']}")
    add("bound of record", *([bound["why"]] if not bound["available"] else [
        f"every partition of the attack's traced circuit into the policy's blocks pays at least "
        f"{bound['slowdown']:,.6g}x ({bound['theorem']}: at least {bound['blocks']:.4g} blocks of c gates over its |C|)",
        f"rests on: {bound['rests_on']}", "a lower bound for any attack with this circuit, beside S, which this attack "
                                          "pays; the two are never combined"]))
    if per is None:
        add("per period", "none: no S")
    else:
        add("per period", f"the run at S: {per['run_and_gates']:.4g} AND gates (|C| x rho); its share of the compute of",
            *(f"{over} ({rows[0]['fleet']}, {rows[0]['locations']:,.0f} locations): " + ", ".join(
                f"{x['period']} {x['share']:.3g}" for x in rows)
              for over in dict.fromkeys(x["over"] for x in per["rows"])
              for rows in [[x for x in per["rows"] if x["over"] == over]]),
            f"(the window is {per['window']}'s; each fleet at its peak rate)",
            *(f"{k}: {v}" for k, v in per["parts"].items()))
    add("records", *(f"{x['config']} {x['seed']}/{x['split']}: {x['form']}, version {x['version']}, "
                     f"{x['scored']} of {x['checkpoints']} checkpoints scored" + (f", {x['label']}" if x["label"] else "")
                     for x in s["records"]))
    return "\n".join(lines)


def _plain(v):
    """`v` as JSON, a non-finite number as a string."""
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return str(v) if isinstance(v, float) and not math.isfinite(v) else v


def read(root: Path) -> list[dict]:
    """A config's jobs under `root`: `<seed>/<split>/` with `rows.jsonl.gz` or `rows.jsonl` (a traced record's rows),
    else `run.json`."""
    out = []
    for d in sorted(p for p in root.glob("*/*") if p.name in SPLITS and p.parent.name.isdigit()):
        job = {"seed": int(d.parent.name), "split": d.name}
        rows = next((p for p in (d / "rows.jsonl.gz", d / "rows.jsonl") if p.exists()), None)
        if rows is not None:
            with (gzip.open if rows.suffix == ".gz" else open)(rows, "rt") as fh:
                out.append(job | {"rows": [json.loads(ln) for ln in fh if ln.strip()], "source": str(rows)})
        elif (d / "run.json").exists():
            out.append(job | {"record": json.loads((d / "run.json").read_text()), "source": str(d / "run.json")})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True, help="the attack's config; its stem is its name")
    ap.add_argument("--starts", type=Path, required=True, help="the starts registry, starts.toml")
    ap.add_argument("--records", type=Path, required=True, help="the config's records, DIR/<seed>/<split>/")
    ap.add_argument("--honest", type=Path, required=True, help="its honest config's records")
    ap.add_argument("--also", action="append", default=[], metavar="NAME=DIR",
                    help="another config's records at the same start and gain")
    ap.add_argument("--out", type=Path, default=None, help="where slowdown.json goes")
    a = ap.parse_args(argv)
    config = tomllib.loads(a.config.read_text()) | {"name": a.config.stem}
    others = {n: read(Path(d)) for n, _, d in (x.partition("=") for x in a.also)}
    s = grade(config, tomllib.loads(a.starts.read_text()), read(a.records), read(a.honest), others)
    print(text(s))
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "slowdown.json").write_text(json.dumps(_plain(s), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
