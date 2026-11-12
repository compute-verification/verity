"""The grade: a config's slowdown S, p times the record's slowdown, from its jobs' records and its honest config's.

    python -m training_slowdown.grade --config attacks/NAME.toml --starts starts.toml --records DIR --honest DIR \\
        [--also NAME=DIR ...] [--out OUT]

A config (an attack's TOML file) names its `start`, its `target.gain`, its `policy`, its `honest` config and its
`seeds` (`search` and `final`). DIR holds one config's records as its jobs wrote them, `DIR/<seed>/<split>/run.json`
with `split` `search` or `final`. `--honest` is the honest config's DIR, and each `--also` another config's at the
same start and gain. The grade reads no file of a served run. A record may say where it came from (`provenance`, which
no job writes): the grade never reads it and passes it on, beside each record it read (`records`) and, for the record
whose slowdown it read, in what S rests on (`rests_on`). It prints the block (`text`) and writes `OUT/slowdown.json` (`SCHEMA`):

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
- **the record's slowdown**, the attack's own cost at the config's policy (`policy.named`): `formalism.published` at (c, k) on the search
  record, at its first checkpoint reaching the gain on the search split, else at its last, which the block says. It
  reads the record's `trace`, the step the job traced itself, and prices its product rows here (`traced`,
  `work.priced`): the grade holds no price. Where the Fit fails it is the slowdown the benchmark's one entry point
  gives on those products (`calculator.slowdown`, which `formalism.bound` calls): the grade computes none. A search
  record with no trace, with one the capture refused or of a capture version this tree doesn't read, or with a step
  `formalism.measure` refuses, has no slowdown, and then there is no S; the reason names how the trace is taken again
  (`retrace`). So does a record whose trace was not taken (`trace.failed`: the process that traced the step raised, or
  ended), whose slowdown is unmeasured, not refused: nothing was found against the step;
- **the sensitivities** beside the record's slowdown, never in it (`formalism.SENSITIVITIES`' reduction and update), and the version of
  the code that took the trace (`capture_code`): from the trace's `circuit`, the measured step's own cut of its
  gradient and its structure, and its `capture_code`. A record whose trace has neither still grades, at the same slowdown
  where τ's stub changes no step count: the block says which figures are blank and why (`blank`);
- **S**, p times the record's slowdown, against each H: a lower bound where that is one (`partial`), with what each
  factor rests on;
- **the bound of record** at the same policy (the `bound` of `calculator.slowdown`'s result there): the least
  slowdown of any partition of the attack's traced circuit, by the theorem's count solved for the fewest blocks. The
  products term of the record's slowdown is this bound, so where products bind the record's slowdown is it and S is p times it (`BOUND_IS_SLOWDOWN`). The block prints
  it under S with its statement and what it rests on;
- **per period** (`per_period`): the run's AND gates at S, C times the record's slowdown, and their share of one
  datacenter's and of the world's compute over a window, a week, a month and a year (`calculator.fleets`,
  `calculator.periods`), with the measured, proved and scaled parts named.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tomllib
from pathlib import Path

import numpy as np

from . import VERSION, calculator, gates, policy, work
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
#: what a record's slowdown reads of its `trace` (`traced`) and has no figure without; the job writes more (`traced.trace_step`)
TRACE = ("capture", "tokens", "k", "rows", "p_match", "at", "data_dependent", "carried_scalars")
#: what the grade reads of a trace beside `TRACE`, where the trace has it, each with what it is: a record without one
#: still grades, the figures that read it blank (`blank`)
BESIDE = {"circuit": "the measured step's cut of its gradient and its structure, which the reduction and update "
                     "sensitivities read",
          "capture_code": "the version of the code that took the trace"}
#: how the bound of record stands to the record's slowdown, which the block says under it: `calculator.slowdown` scores each choice of the
#: work by the calculator's bound, so the products term is the bound of record on the work it chose
BOUND_IS_SLOWDOWN = ("the products term of the record's slowdown is this bound, the theorem's count solved for the fewest blocks, times c/|C|, on "
                "the same N, O, β, b and F: where products bind, the record's slowdown is it and S is p times it")
#: the job that takes a record's trace again (`training_slowdown.run`'s `retrace`), which the reason names where the record's slowdown
#: has no trace to read: one refused, or one that was not taken
RETRACE = "python -m training_slowdown.run --retrace RUN_JSON --out DIR --device cpu"


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


def prefix(record: dict, step: int) -> int | None:
    """The last step of the prefix a checkpoint at `step` closes: None at or past the record's last step, the whole
    run and its tail."""
    last = record["steps"][-1]["step"] if record["steps"] else -1
    return None if step >= last else step


def rows_of(record: dict) -> list[dict]:
    """One row per checkpoint of a run record, as `formalism.measure` reads one: its step, its loss (`val_bpb`, the
    checkpoint's `score.bpb`), the start's loss on the same split (`start_bpb`, its `score.start_bpb`), the split, the
    `formalism` block and op log of the run's prefix through it, its output's facts (`OUTPUT`) and the record's
    `invalid`. The block is the count's (`formalism.of_record`); what the record's `trace` gives it is added where the record's slowdown
    reads it (`traced`)."""
    out = []
    for cp in record["checkpoints"]:
        through, score = prefix(record, cp["step"]), cp.get("score") or {}
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
    """A job's checkpoints by step (`run`: its `seed`, its `split` and its `record`): each one's loss, the start's
    loss on the job's split and its counted |C| in gates (`counted`), or why it has none. A job from another start, or
    scored on another split than its place says, is refused."""
    where = f"seed {run['seed']}, {run['split']}"
    out = []
    for r in sorted(rows_of(run["record"]), key=lambda r: r["checkpoint_step"]):
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


# ---- the record's slowdown, the bound of record and the periods ------------------------------------------------------

def retrace(record: dict) -> str:
    """How a record's trace is taken again, which a record whose slowdown can't be read says: by the job that re-traces a record's
    own config (`RETRACE`), which reads its `config.text`, takes the trace with the capture of the tree it runs in
    (so this tree's, for a trace this grade is to read) and refuses a tree that reads the config as another program
    than the record's; a record without its `config.text` or its `commit` has no config to re-trace from."""
    lacks = [k for k, v in (("config.text", (record.get("config") or {}).get("text")), ("commit", record.get("commit")))
             if not v]
    if lacks:
        return f"the record has no {' and no '.join(lacks)}, so no config to re-trace from"
    return (f"re-trace it at the record's commit {record['commit']} and config.text with a CPU job, `{RETRACE}`, "
            f"RUN_JSON the record: the tree the job runs in takes the trace with its own capture, and refuses a "
            f"config it reads as another program than the record's")


def changed_steps(record: dict, through: int | None) -> list[dict]:
    """The record's steps through `through` (None: all) after the run's first update: each that read a weight the run
    had changed, as the counter lists a step's `weights` (`flops`). A weight of θ is listed only where its value
    differs from θ's, as is a buffer of the model under its name (a position table a step wrote), and a value
    derived from a gradient always; a tensor neither θ nor the model's buffers hold (`new/…`: an adapter, an
    optimizer's state; in a record of a harness that held no buffers, a table the model builds) is listed by every
    step that reads it, so it counts once a value it is read at isn't the first it was read at (a record without
    digests shows none as changed)."""
    first, out = {}, []
    for s in record["steps"]:
        changed = [not key.startswith("new/") or any(v != first.setdefault(key, values[0]) for v in values)
                   for key, values in (s.get("weights") or {}).items()]
        if any(changed) and (through is None or s["step"] <= through):
            out.append(s)
    return out


def with_circuit(record: dict, circuit: dict) -> dict:
    """The record with its trace's circuit on every step after the run's first update (`changed_steps`): each one's
    structure with the traced step's σ, affected share, token factor, Δ_grad and cuts (`formalism.TRACED_STRUCTURE`),
    as `formalism.of_record` reads a traced structure, so the traced step stands for each of them. A step before the
    first update keeps its structure: it reads θ alone and no step like it was traced, so in a record with a traced
    structure it has no affected gates and no Δ_grad."""
    changed = {s["step"] for s in changed_steps(record, None)}
    out = dict(record, structures=dict(record["structures"]), steps=[dict(s) for s in record["steps"]])
    for s in out["steps"]:
        if s["step"] in changed:
            was, s["struct"] = s["struct"], f"{s['struct']}+traced"
            out["structures"][s["struct"]] = record["structures"][was] | {k: circuit[k] for k in F.TRACED_STRUCTURE}
    return out


def traced(record: dict, row: dict) -> dict:
    """`row` (one of `rows_of(record)`) as the record's slowdown reads it: with the products of the record's `trace`, the step its job
    traced (`traced.trace_step`), in its `formalism` block, as `formalism.measure` takes a traced record's. Each raw
    product row is priced here (`work.priced`), at the k the job classified the rows at and at each
    other k of the console's dial (`at`), and the traced step stands for every step through the row's checkpoint
    after the run's first update (`changed_steps`), by their tokens over its own, a `stand_in` where they differ from
    it in structure or tokens. Of the trace it reads `TRACE`: the capture's version, the traced step's tokens, the
    rows with their k, their P-match check and the dial's, and the read-backs `formalism.refusal` refuses on.

    Beside them it reads what the trace has of `BESIDE`. With the trace's `circuit` the block's parts are the
    record's with the traced step's structure on those steps (`with_circuit`, `formalism.of_record`), its τ, a and
    update profile are the circuit's (`formalism.of_trace`), and the share `formalism.refusal` reads for being 0 is
    the circuit's affected share; without it the parts are the count's, τ, a and the profile are stubs, and that
    share is the traced step's product MACs an update reaches. With the trace's `capture_code` the products state
    it; without it they state none, since a record whose trace was taken again keeps its job's `training_slowdown`.

    Raises `formalism.Refused`, with how the trace is taken again (`retrace`), for a record with no trace, with one
    the capture refused, with one of a capture version this tree's `priced` doesn't read, or with one that lacks a
    key of `TRACE`; and for a checkpoint with a nonempty E that no step after the run's first update precedes. A
    record whose trace was not taken (`failed`, with why: `run.inside`, `run.outside`) raises a plain `ValueError`
    with the same way out: its step is unmeasured, and nothing refused it."""
    from .capture import products

    tr = record.get("trace")
    if not tr:
        raise F.Refused(f"the search record has no trace, and its slowdown reads a traced step; {retrace(record)}")
    if "failed" in tr:
        raise ValueError(f"the search record's trace was not taken ({tr['failed']}), and its slowdown reads a traced step; "
                         f"{retrace(record)}")
    if "refused" in tr:
        r = tr["refused"]
        raise F.Refused(f"the capture refused the search record's step ({r.get('reason')}, {r.get('op')}: "
                        f"{r.get('message')}), and its slowdown reads a traced step; {retrace(record)}")
    if tr.get("capture") != products.CAPTURE:
        raise F.Refused(f"the search record's trace is capture {tr.get('capture')}'s, and this grade reads capture "
                        f"{products.CAPTURE}'s; {retrace(record)}")
    lacks = [k for k in TRACE if k not in tr]
    if lacks:
        raise F.Refused(f"the search record's trace has no {', '.join(lacks)}, which its slowdown reads; {retrace(record)}")
    steps = changed_steps(record, prefix(record, row["checkpoint_step"]))
    if not steps and (row.get("output") or {}).get("e_bytes") != 0:
        raise F.Refused(f"update-free prefix: E is not empty, but no step through checkpoint {row['checkpoint_step']} "
                        f"reads a weight the run changed")
    block = work.priced(tr)
    got = work.products_of_trace(block, sum(s["tokens"] for s in steps) / tr["tokens"])
    label = F.TRACED_PRODUCTS
    structures, counts = sorted({s["struct"] for s in steps}), sorted({s["tokens"] for s in steps})
    if len(structures) > 1 or counts not in ([], [tr["tokens"]]):
        got["stand_in"] = {"structures": structures, "step_tokens": counts, "traced_tokens": tr["tokens"],
                           "why": F.STAND_IN}
        label = f"{label}; {F.STAND_IN}"
    f, circuit = row["formalism"], tr.get("circuit")
    if circuit is None:
        macs = [(r["count"] * r["macs"], any(r["affected"])) for r in block["rows"]]
        circuit = {"affected_share": sum(m for m, hit in macs if hit) / sum(m for m, _ in macs) if macs else 0.0}
    else:
        f, read = F.of_record(with_circuit(record, circuit), prefix(record, row["checkpoint_step"])), F.of_trace(circuit)
        f |= read | {"labels": f["labels"] | read["labels"]}
    return row | {"formalism": f | {
        "products": got, "labels": f.get("labels", {}) | {"products": label},
        "trace": circuit | {"products": block} | {k: tr[k] for k in F.READ_BACKS}}}


def blank(record: dict, got: dict, unmeasured: dict) -> dict:
    """The figures beside the record's slowdown (`got`, `formalism.published`'s) that have no value, each with why. On a record whose
    trace lacks a key of `BESIDE`: the reduction and update sensitivities without its `circuit`, the `capture_code`
    without its own, each with how the trace is taken again (`retrace`; this tree's job keeps both). On a trace that
    has its circuit, a sensitivity the bound leaves without a figure, with the bound's reason (`unmeasured`)."""
    tr = record.get("trace") or {}

    def lacks(key: str) -> str:
        return f"the search record's trace has no {key}, {BESIDE[key]}; {retrace(record)}"

    out = {name: unmeasured[name] if "circuit" in tr else lacks("circuit")
           for name in F.CELL_SENSITIVITIES if got["sensitivities"][name] is None}
    if got["capture_code"] is None:
        out["capture_code"] = lacks("capture_code")
    return out


def slowdown_of(run: dict | None, f: dict, g: float, pol: dict, shape: dict) -> tuple[dict, dict]:
    """The record's slowdown at the policy `pol` from the config's search job `run`, a start of `shape`, and the bound of record there:
    `formalism.published` at (c, k) on its first scored checkpoint at or below its target, else on its last (`at`
    says which), read with its record's trace (`traced`), with the theorem's inputs there (`work`, the bound of
    record's: none where the record gives no slowdown or its M is empty), the capture that traced it, the figures
    beside it that have no value (`blank`) and what the bound says of a sensitivity that sums only some of the
    record's parts (`unmeasured`). The bound of record is the `bound` of `calculator.slowdown`'s result on the
    record, which the products term of the record's slowdown is of too (`formalism.bound`'s `entry`). No record's slowdown, with why, and the bound not
    available, without a search record, without a scored checkpoint, on a record whose trace `traced` refuses, or on
    one `formalism.measure` refuses or can't measure."""
    none = calculator.slowdown(pol, None)["bound"]
    if run is None:
        return {"rho": None, "why": "unmeasured: the config has no search record"}, none
    at = seed_at(run["checkpoints"], f, g)
    cp = next((c for c in run["checkpoints"] if c["gates"] is not None and c["step"] == at.get("step")), None)
    if cp is None:
        return {"rho": None, "why": f"unmeasured: the search record has no scored checkpoint ({at.get('why')})"}, none
    fwd = run["record"].get("forward_macs_per_token")
    try:
        m = F.measure(traced(run["record"], cp["row"]), shape | ({"forward_macs_per_token": fwd} if fwd else {}))
    except F.Refused as e:
        return {"rho": None, "why": f"refused: {e}"}, none
    except ValueError as e:
        return {"rho": None, "why": f"unmeasured: {e}"}, none
    c, k = pol["c_gates"], pol["k_bits"]
    b, out = F.bound(m, c, k), F.published(m, c, k)
    of_record = b["entry"]["bound"]
    return out | {
        "blank": blank(run["record"], out, b["unmeasured"]),
        "unmeasured": {name: b["unmeasured"][name] for name in F.CELL_SENSITIVITIES if name in b["unmeasured"]},
        "seed": run["seed"], "checkpoint_step": cp["step"], "C_gates": cp["gates"],
        "at": "its first checkpoint reaching the gain" if at["reached"] else
              f"its last checkpoint: the search record doesn't reach {g:g}x",
        "capture": (m.products or {}).get("capture"), "provenance": run["record"].get("provenance"),
        "work": of_record.get("work")}, of_record


def per_period(run_gates: float, slowdown: dict) -> dict:
    """The run's gates at S (`run_gates`, C times the record's slowdown) in AND gates, at a served-precision GEMM's
    gates an AND, and their share of each fleet's compute over each period (`calculator.fleets`,
    `calculator.periods`), with what is measured, what is proved and what is scaled."""
    run_and = run_gates / gates.per_and(f"gemm/{policy.SERVED_PRECISION}")
    rows = [{"over": fl["over"], "fleet": fl["fleet"], "locations": fl["locations"], "period": name, "seconds": s,
             "fleet_and_gates": fl["and_per_s"] * s, "share": run_and / (fl["and_per_s"] * s)}
            for fl in calculator.fleets() for name, s in calculator.periods().items()]
    return {"run_and_gates": run_and, "window": calculator.window()[0], "rows": rows, "parts": {
        "measured": "|C| of each seed and H, the counted gates of the records read, so p; the inputs of the record's slowdown, one traced step "
                    "of the search record",
        "proved": (f"the count the products term of the record's slowdown solves for the fewest blocks, {calculator.theorem()}'s first "
                   "conjunct (`HEShared`'s where its products share arrays): a lower bound at the N, O, β_w, β_x and "
                   "b_y the traced step gives, given (H), (E) and the theorem's hypotheses on the circuit, which the "
                   "trace doesn't check; the solving is the calculator's arithmetic, and p and the shares are not "
                   "proved" if slowdown["binding_kind"] == "products" else
                   f"nothing: the {slowdown['binding_kind']} term of the record's slowdown is computed"),
        "scaled": f"every share: a fleet's AND gates are its registry's peak FLOP a second, two a MAC at the "
                  f"{calculator.FLEET_PRECISION} GEMM's AND gates, times its locations and the period, and the run's "
                  f"gates are AND gates at the {policy.SERVED_PRECISION} GEMM's ratio. The fleets' sizes and the "
                  f"window are stated parameters"}}


# ---- the grade ---------------------------------------------------------------------------------------------------------

def grade(config: dict, starts: dict, runs: list[dict], honest: list[dict],
          others: dict[str, list[dict]] | None = None) -> dict:
    """The slowdown of `config` (its `name`, `start`, `target.gain`, `policy`, `honest` and `seeds`) from its jobs
    `runs`, its honest config's (`honest`) and other configs' at the same start and gain (`others`, by name), each job
    its `seed`, its `split` and its `record`; `starts` is the starts registry (module docstring)."""
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
    slowdown, of_record = slowdown_of(search, f, g, pol, entry["shape"])
    why = (slowdown.get("why") or "the record's slowdown is unmeasured") if slowdown["rho"] is None else \
        None if C else f"no seed of the config reaches {g:g}x on {SPLITS[basis]}"
    S = {k: None if why or not p[k] else p[k]["p"] * slowdown["rho"] for k in H} | {
        "why": why, "lower_bound": bool(slowdown.get("partial"))}
    return {
        "schema": SCHEMA, "training_slowdown": VERSION, "config": me, "start": config["start"], "gain": g,
        "policy": pol, "basis": {"split": basis, "why": (f"no final record of {' or '.join(lacking)}" if lacking else
                                                         "the attack and the honest config each have a final record")},
        "fit": f | {"data": entry.get("data"), "starts": data,
                    "c_spread": c_spread(f, target(f, entry["l0_bpb"], g))},
        "attack": sides[me], "honest": sides[hon], "others": [sides[n] for n in others or {}],
        "H": H, "p": p, "rho": slowdown, "S": S,
        "bound_of_record": of_record,
        "rests_on": {
            "p": f"counted |C| ({gates.UNIT}) at each seed's first checkpoint at or below its target on "
                 f"{SPLITS[basis]}, the mean over the seeds that reach, over the same for H",
            "rho": slowdown.get("rests_on"), "capture": slowdown.get("capture"), "capture_code": slowdown.get("capture_code"),
            "provenance": slowdown.get("provenance"),
            "gain": f"the fit {FORM} on {f['points']} of the {len(data)} starts on data {entry.get('data')}"
                    + (f", provisional under {FIT_TARGET}" if f["provisional"] else "")},
        "per_period": per_period(C["mean"] * slowdown["rho"], slowdown) if C and slowdown["rho"] is not None else None,
        "records": [_read(n, r) for n, rs in jobs.items() for r in rs]}


def _read(name: str, run: dict) -> dict:
    """What the grade read of one job: its place, the label and the version of the code that wrote its record, how
    many of its checkpoints are scored, and the record's `provenance` where it has one, passed on as it is."""
    wrote = run["record"]
    return {"config": name, "seed": run["seed"], "split": run["split"], "source": run.get("source"),
            "label": wrote.get("label"), "version": wrote.get("training_slowdown") or wrote.get("te_core"),
            "checkpoints": len(run["checkpoints"]),
            "scored": sum(c["gates"] is not None for c in run["checkpoints"])} | (
        {"provenance": wrote["provenance"]} if "provenance" in wrote else {})


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


def _said(provenance) -> str:
    return provenance if isinstance(provenance, str) else json.dumps(provenance, sort_keys=True)


def text(s: dict) -> str:
    """The block the grade prints (module docstring)."""
    pol, f, slowdown, S, basis = s["policy"], s["fit"], s["rho"], s["S"], s["basis"]["split"]
    mine, per, bound = s["attack"]["splits"], s["per_period"], s["bound_of_record"]
    lines = [f"SLOWDOWN {s['config']}   start {s['start']}, gain {s['gain']:g}x, policy {pol['name']} ({pol['kind']}: "
             f"c = {pol['c_gates']:.4g} gates, k = {pol['k_bits']:,.0f} bits; {pol['why']})"]

    def add(head: str, *body: str) -> None:
        lines.extend(f"  {h:<17} {b}" for h, b in zip((head, *[""] * len(body)), body))

    if S["why"]:
        add("S = p x slowdown", f"none: {S['why']}")
    else:
        add("S = p x slowdown", *(f"against {name}: S {'>= ' if S['lower_bound'] else ''}{S[k]:,.6g} = p {s['p'][k]['p']:.4g}"
                             f" x the record's slowdown {slowdown['rho']:,.6g}" if S[k] else f"against {name}: none (H: {_h(s['H'][k])})"
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
    add("record's slowdown", f"none: {slowdown['why']}" if slowdown["rho"] is None else
        f"{slowdown['rho']:,.6g}{' (a lower bound: ' + str(slowdown['why'] or 'a kind is unmeasured') + ')' if slowdown['partial'] else ''}"
        f", binding {slowdown['binding_kind']}, on search seed {slowdown['seed']} at {slowdown['at']} (step {slowdown['checkpoint_step']})")
    if slowdown["rho"] is not None:
        gaps = slowdown["blank"]
        add("sensitivities", ", ".join(f"{name} " + ("blank" if name in gaps else f"{v:,.6g}")
                                       for name, v in slowdown["sensitivities"].items())
            + ", beside the record's slowdown and never in it: each holds only for partitions split by tokens",
            *(f"{name}: " + (f"blank: {why}" if name in gaps else f"partial: {why.removeprefix('unmeasured: ')}")
              for name, why in (slowdown["unmeasured"] | {n: w for n, w in gaps.items() if n in slowdown["sensitivities"]}).items()))
    seeds = [_seed(split, seed, v) for split in ("final", "search") for seed, v in mine[split]["seeds"].items()]
    theirs = [_seed(split, seed, v) for split in ("final", "search")
              for seed, v in s["honest"]["splits"][split]["seeds"].items()]
    add("seeds", *seeds, *(f"honest {x}" for x in theirs),
        "the spread of |C|: " + ("none" if not C else f"{C['min']:.4g} to {C['max']:.4g} gates over {C['n']} seeds"
                                  + (f", sd {C['sd']:.3g}" if C["sd"] is not None else "")))
    r = s["rests_on"]
    add("rests on", f"the record's slowdown: {r['rho'] or 'nothing: it is unmeasured'}",
        f"capture: {r['capture'] or 'none'}, its code "
        + (r["capture_code"] or f"blank: {slowdown.get('blank', {}).get('capture_code', 'the slowdown of the record is unmeasured')}")
        + (f"; the record's provenance: {_said(r['provenance'])}" if r["provenance"] is not None else ""),
        f"p: {r['p']}", f"gain: {r['gain']}")
    add("bound of record", *([bound["why"]] if not bound["available"] else [
        f"every partition of the attack's traced circuit into the policy's blocks pays at least "
        f"{bound['slowdown']:,.6g}x ({bound['theorem']}: at least {bound['blocks']:.4g} blocks of c gates over its |C|)",
        f"rests on: {bound['rests_on']}", BOUND_IS_SLOWDOWN]))
    if per is None:
        add("per period", "none: no S")
    else:
        add("per period", f"the run at S: {per['run_and_gates']:.4g} AND gates (|C| x the record's slowdown); its share of the compute of",
            *(f"{over} ({rows[0]['fleet']}, {rows[0]['locations']:,.0f} locations): " + ", ".join(
                f"{x['period']} {x['share']:.3g}" for x in rows)
              for over in dict.fromkeys(x["over"] for x in per["rows"])
              for rows in [[x for x in per["rows"] if x["over"] == over]]),
            f"(the window is {per['window']}'s; each fleet at its peak rate)",
            *(f"{k}: {v}" for k, v in per["parts"].items()))
    add("records", *(f"{x['config']} {x['seed']}/{x['split']}: version {x['version']}, "
                     f"{x['scored']} of {x['checkpoints']} checkpoints scored" + (f", {x['label']}" if x["label"] else "")
                     + (f", provenance: {_said(x['provenance'])}" if "provenance" in x else "")
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
    """A config's jobs under `root`: each `<seed>/<split>/run.json`, the record its job wrote."""
    return [{"seed": int(p.parent.parent.name), "split": p.parent.name, "record": json.loads(p.read_text()),
             "source": str(p)}
            for p in sorted(root.glob("*/*/run.json")) if p.parent.name in SPLITS and p.parent.parent.name.isdigit()]


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
