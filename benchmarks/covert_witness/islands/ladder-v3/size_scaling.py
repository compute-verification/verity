"""Plot 3's panel `training/size-scaling`: how the red-teaming study scales with model size. Per size (the ladder's v3
starts and the console's 1B reference size), gain target and policy, standard training's ρ beside the measured
frontier's least S = p·ρ, in the dashboard's columns (`COLUMNS`).

    python islands/ladder-v3/size_scaling.py trace --out DIR [--only NAME,...]   # on a node, with torch: the steps
    python islands/ladder-v3/size_scaling.py build --tables DIR --ledger DIR --steps DIR --out DIR \
        [--trace-runs RUN | NAME=RUN,...]
    python islands/ladder-v3/size_scaling.py put FILE [--dry]                    # on the VM: PUT a built body

`put --dry` PUTs nothing: it prints the body's sha256, bytes, rows and note length, and what te_core.console's `put`
would refuse it for.

Run from `benchmarks/covert_witness/`. c and |C| are Boolean gates (te_core.gates), k bits.

**Sizes** (`SIZES`): s-t20, m-t20, l16-t20 and xl-t20, each at its shape (te_core.policy's `SHAPES`) and the ladder's
step (ladder.py: its batch of 1,024-token sequences in micro-batches of `MICRO`), and the console's 1B reference size
(te_core.policy's `REFERENCE_SIZES`, vocabulary 50,304), which no start has, at xl's step. A start's bits per byte and
effective compute are the v3 fit's (te_core.frontier_fit: C_eff at l0, lo and hi at l0 plus and minus the fit's rms in
loss); 1B has neither.

**Standard training** of a size is the ladder's step, te_core.io_bound's `dense` recipe (Muon on the block matrices,
AdamW on the tied embedding and the norm gains) traced at full size under fake tensors (`trace`): its |C| and its
products at each k of the console's dial (te_core.capture.products' `rows_on_dial`) and at the serving policy's k. Its
quantities are the step's as te_core.formalism completes a record's (`step`), and its ρ at (c, k) is te_core.formalism's
`published` at reading A, as te_core.workloads prices its models: the products term where it exceeds 1, partial since
a traced step alone has no token entries or op log, else null with why. Its p is 1: it is the honest route.

**Policies** (`policies`: each ρ's (c, k) and the policy's input ids in bits, `k_bits`): `grown`, the headline, the
size's own serving session (te_core.policy's `session` of the console's 4,096 input and 1,024 output tokens on the
size's own model, the session te_core.algorithm_policy's headline is): c its gates, k_bits its input ids at the size's
vocabulary (61,440 bits at 32,000, 65,536 at 50,304) and k the console's, a 16-bit word an input id, as
te_core.rho_vectors' session dial prices the products term, at which every record's products are classified. ρ falls
as k grows, so where k_bits is less than k the row says ρ there is at least its ρ at k (`policy_label`). `console`,
te_core.policy's `CONSOLE`, one session of the 45M reference, a sensitivity; `serving`, k = k_bits = `SERVING_K`
(100,000 input ids at 16 bits) and c one session of the size's own model at 100,000 input and 1,024 output tokens.
Each serving row says it rests on te_core.formalism's `BETA_K_LABEL`: at that k the products term charges an
activation at most β(k), io-bound's red-team of (H), not its width.

**The measured frontier**: the scorer's uncut algorithm-policy table (te_core.algorithm_policy), each run's first
checkpoint reaching each gain (`reached` yes, p = C/H), every route but standard training's (`STANDARD`), each
checkpoint measured as the scorer measured it (`scored`: its ledger record with the formalism block in the scorer's
formalism.jsonl, te_core.formalism's `measure`) and priced at each policy by `published`. Where the table is this
te_core's, every row's ρ, binding and basis are the table's own at console (rho_console) and at grown (its headline
rho_own, at the same c) (`check`, `TABLE`). A record keeps its products at the dial's k alone, so at a policy's k off it
the products term is unmeasured. Each priced row has the table's basis (`basis`): a row that is unmeasured, rests on a
stub or is partial is never the least S. A run that reached with no trace yet (te_core.algorithm_policy's `UNTRACED`)
has the table's p and no ρ. The row is the eligible one with the least S; else, where only untraced runs reached, the
one with the least p, its ρ and S null and basis `UNTRACED` (reached, not yet traced); else null S with why: no attack
run at this size, none reached, or none with a full ρ at the policy, with the commonest reason the products term has no
figure (`unpriced`: products not classified at the policy's k, where it is off the dial).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from te_core import VERSION, formalism, policy  # noqa: E402
from te_core.gates import UNIT  # noqa: E402

# te_core.console, te_core.frontier_fit and te_core.algorithm_policy (through te_core.plots) read matplotlib, which the
# trace's torch environment hasn't, so `build` and `put` import them

PANEL = "training/size-scaling"
#: (size, its parameters in te_core.policy's SHAPES, the ladder's batch in sequences): the four v3 starts at ladder.py's
#: batches and the console's 1B reference size at xl's
SIZES = (("s-t20", 35_658_240, 128), ("m-t20", 81_212_160, 192), ("l16-t20", 238_322_688, 384),
         ("xl-t20", 656_146_944, 768), ("1B", 999_397_440, 768))
#: ladder.py's sequences a micro-step
MICRO = 32
#: the serving policy's k: 100,000 input ids at 16 bits, and its session
SERVING_K = 1_600_000
SERVING_SESSION = (100_000, policy.CONSOLE_SESSION[1])
GAINS = (1.1, 2, 11)
POLICIES = ("console", "grown", "serving")
#: the routes that are standard training: ref-optim, the start's own recipe continued (te_core.plots' method keys)
STANDARD = "ref-optim/"
#: the dashboard's columns, then the policy's (c, k), the products term, the row's basis (te_core.algorithm_policy's
#: basis_default) and what the products term read (`products_inputs`)
COLUMNS = ["start", "params_total", "params_nonembedding", "bits_per_byte", "start_effective_compute",
           "start_effective_compute_lo", "start_effective_compute_hi", "gain_target", "policy", "route", "p", "rho", "S",
           "binding_kind", "rests_on", "reading", "te_core", "traced_or_modelled", "run", "c_gates", "k_bits",
           "rho_products", "basis", "products_inputs"]
#: each policy whose ρ the scorer's table holds, its columns that `check` compares: ρ, binding kind (None: no column),
#: basis and c (None: the policy's c is the console's constant)
TABLE = {"console": ("rho_console", "binding_kind", "basis_default", None),
         "grown": ("rho_own", None, "basis_own", "own_c_gates")}
#: what the products term read that `products_inputs` shows, where te_core.formalism's term gives it
INPUTS = ("subset", "N", "O", "beta_w", "beta_x", "beta_k", "b_y", "F", "alpha", "K_thr", "macs_share", "k_classified")
#: a standard step's affected gates: full training updates every weight, so every gate depends on θ'
AFFECTED = "stub: every gate depends on theta' (full training updates every weight)"
NO_ATTACK = "no attack run at this size"


def shape(params: int) -> dict:
    s = policy.SHAPES[params]
    if policy.parameters(s) != params:
        raise SystemExit(f"size_scaling: te_core.policy's shape for {params:,} has {policy.parameters(s):,} parameters")
    return s


def policies(params: int) -> dict[str, tuple[float, float, float]]:
    """The three policies (module docstring) for the size of `params` parameters, each (c, k, k_bits)."""
    s = shape(params)
    own = policy.session(params, *policy.CONSOLE_SESSION, s)
    return {"console": (*policy.CONSOLE, policy.CONSOLE[1]),
            "grown": (own["gates"], policy.CONSOLE[1], float(own["k_bits"])),
            "serving": (policy.session_gates(params, *SERVING_SESSION, s), float(SERVING_K), float(SERVING_K))}


# ---- trace: standard training's steps ------------------------------------------------------------------------------

def trace(name: str, params: int, batch: int) -> dict:
    """The ladder's step at full size for `name` (module docstring), as te_core.capture.products' rows at each k of the
    console's dial and at `SERVING_K`, with its |C| (`gates`) and the step's settings."""
    import model
    from te_core import io_bound
    from te_core.capture import products

    s = shape(params)
    model.SIZES.setdefault(name, model.Size(name, s["d_model"], s["n_layer"], s["n_head"], s["d_ff"]))
    t0 = time.monotonic()
    graph = io_bound.capture("dense", name, s["vocab"], batch, model.CTX, batch // MICRO)
    t1 = time.monotonic()
    block = products.rows_on_dial(graph)
    serving = products.rows(graph, float(SERVING_K))
    block["at"][str(SERVING_K)] = {x: serving[x] for x in ("k", "rows", "p_match")}
    return block | {"name": name, "params": params, "shape": s, "batch": batch, "micro": MICRO, "ctx": model.CTX,
                    "recipe": "te_core.io_bound's dense: Muon on the block matrices, AdamW on the tied embedding and "
                              "the norm gains",
                    "te_core": VERSION, "seconds": {"capture": round(t1 - t0, 1),
                                                    "rows": round(time.monotonic() - t1, 1)}}


# ---- pricing ---------------------------------------------------------------------------------------------------------

def step(block: dict) -> formalism.Measured:
    """A traced step's quantities as te_core.formalism completes a record's block (`complete`): its |C| as one part
    with the step's tokens, every gate affected (`AFFECTED`), its products, and the rest stubbed from its shape, θ' a
    full fp32 E stored as is (no decoder), each labelled."""
    C = block["gates"]
    f = {"parts": [{"gates": C, "tokens": block["tokens"], "affected_gates": C}],
         "products": formalism.products_of_trace(block),
         "labels": {"products": formalism.TRACED_PRODUCTS, "affected_gates": AFFECTED}}
    return formalism.complete(f, {"output": {"e_bytes": 4 * block["params"], "format": "full"}}, block["shape"])


def basis(one: dict, stub: bool) -> str:
    """The basis te_core.algorithm_policy's `build` gives the published ρ `one`: unmeasured, stub (`stub`: its
    quantities rest on one, `algorithm_policy.estimated`), partial, else the binding kind (its `voided` marks
    nothing)."""
    from te_core import algorithm_policy as ap

    return (ap.UNMEASURED if one["rho"] is None else ap.STUB if stub else ap.PARTIAL if one.get("partial")
            else one["binding_kind"])


def eligible(b: str) -> bool:
    """Whether a row of basis `b` can be the least S (te_core.algorithm_policy: no decade holds the others)."""
    from te_core import algorithm_policy as ap

    return b not in (ap.UNMEASURED, ap.STUB, ap.PARTIAL)


def policy_label(pol: str, k: float, bits: float) -> str | None:
    """What a row at policy `pol`, priced at k for a session of `bits` input bits, rests on beyond its ρ: a serving row
    te_core.formalism's `BETA_K_LABEL`; a grown row whose ids pack into fewer bits than k, that its ρ at k is a lower
    bound on its ρ at `bits`."""
    if pol == "serving":
        return formalism.BETA_K_LABEL
    if pol == "grown" and bits < k:
        ids = policy.id_bits(policy.SHAPES[policy.REFERENCE_PARAMS]["vocab"])
        return f"k = {k:,.0f} ({ids}-bit ids); at the packed {bits:,.0f} rho is at least this"
    return None


def labelled(row: dict, label: str | None) -> dict:
    """`row` with its policy's `label` (`policy_label`) at the end of what it rests on, where it hasn't it already."""
    if label is None or label in (row.get("rests_on") or ""):
        return row
    return row | {"rests_on": "; ".join(filter(None, (row.get("rests_on"), label)))}


def products_inputs(m: formalism.Measured, c: float, k: float) -> str:
    """What the products term read at (c, k) (`INPUTS`, where it gives them) with |C|, as one JSON cell."""
    got = formalism.bound(m, c, k)["inputs"]["fresh_products"]
    vals = {"C": m.gates} | {x: got[x] for x in INPUTS if got.get(x) is not None}
    return json.dumps({x: finite(v) if isinstance(v, float) and x != "k_classified" else v for x, v in vals.items()},
                      separators=(",", ":"))


def rests(one: dict) -> str:
    """What a published ρ rests on, as te_core.algorithm_policy's rows say it (`rests_on`), with its `why` where it has
    one (an unmeasured kind)."""
    from te_core import algorithm_policy

    return algorithm_policy.rests_on(one) + (f": {one['why']}" if one["why"] else "")


# ---- build -----------------------------------------------------------------------------------------------------------

def table(path: Path) -> list[dict]:
    p = json.loads(path.read_text())
    return [dict(zip(p["columns"], r)) for r in p["rows"]]


def fit_v3(rows: list[dict]) -> tuple[dict, dict[str, dict]]:
    """The v3 fit (frontier_fit's form: E, A, gamma, rms_pct, points) and its starts by name, each {label, l0}."""
    f = next(r for r in rows if r["row"] == "fit" and r["data"] == "v3")
    fit = {"E": f["E_bpb"], "A": f["A"], "gamma": f["gamma"], "rms_pct": f["rms_pct"], "points": f["points"]}
    starts = {}
    for r in rows:
        if r["data"] == "v3" and r["row"] in ("start", "point") and r["source"]:
            words = r["source"].split(" ", 2)
            if words[0] == "start":
                starts.setdefault(words[1], {"l0": r["loss_bpb"], "c0": r["C_gates"]})
                if len(words) > 2:
                    starts[words[1]]["label"] = words[2].strip("()")
            elif words[0] == "ladder":
                starts.setdefault(words[1].rstrip(","), {"l0": r["loss_bpb"], "c0": r["C_gates"]})
    return fit, starts


def effective(fit: dict, l0: float | None) -> tuple[float | None, float | None, float | None]:
    """C_eff at l0 and at l0 plus and minus the fit's rms in loss (lo, hi); None where the fit's points are no more
    than its parameters (te_core.frontier_fit's `unbounded`)."""
    from te_core import frontier_fit

    if l0 is None:
        return None, None, None
    mid = frontier_fit.c_eff(fit, l0)
    if fit["points"] <= frontier_fit.PARAMS:
        return mid, None, None
    d = fit["rms_pct"] / 100 * l0
    return mid, frontier_fit.c_eff(fit, l0 + d), frontier_fit.c_eff(fit, l0 - d)


def scored(tables: Path, ledger: Path, wanted: set[tuple[str, int]]) -> dict[tuple[str, int], dict]:
    """The rows the scorer measured for the checkpoints `wanted` ((run, step)): each one's ledger record (te_core.ledger's
    records/<agent>/<id>.json) with the formalism block the scorer's formalism.jsonl keeps for it (te_core.rescore's,
    after `with_latents`)."""
    runs = {r for r, _ in wanted}
    out = {}
    with (tables / "formalism.jsonl").open() as fh:
        for line in fh:
            if not any(r in line[:300] for r in runs):
                continue
            d = json.loads(line)
            key = (d.get("run_id") or d["id"], d.get("checkpoint_step"))
            if key in wanted and d.get("formalism") and not d.get("refused"):
                path = ledger / "records" / d["id"].rsplit("-", 1)[0] / f"{d['id']}.json"
                out[key] = json.loads(path.read_text()) | {"formalism": d["formalism"]}
    return out


def checkpoint(rec: str) -> tuple[str, int]:
    run, at = rec.rsplit("@", 1)
    return run, int(at)


def finite(v: float | None) -> float | None:
    return None if v is None or math.isinf(v) else float(f"{v:.4g}")


def same(mine: float | None, theirs) -> bool:
    """Whether a published ρ is the table's (null, "inf" or a number; a number column's null is also an infinite ρ,
    which the basis beside it tells apart)."""
    if theirs is None:
        return mine is None or math.isinf(mine)
    if mine is None:
        return False
    if math.isinf(mine) or theirs == "inf":
        return math.isinf(mine) and theirs == "inf"
    return math.isclose(mine, float(theirs), rel_tol=1e-9)


def compare(x: dict, pol: str, one: dict, b: str, c: float) -> dict:
    """`check`'s entry for the table's row `x` at policy `pol`: the published ρ `one`, its basis `b` and c beside the
    table's columns (`TABLE`)."""
    rho, kind, base, gates = TABLE[pol]
    theirs = [x[rho], x[kind] if kind else one["binding_kind"], x[base]] + ([x[gates]] if gates else [])
    ok = (same(one["rho"], theirs[0]) and one["binding_kind"] == theirs[1] and b == theirs[2]
          and (gates is None or math.isclose(c, x[gates], rel_tol=1e-12)))
    return {"record": x["record"], "gain_target": x["gain_target"], "policy": pol, "rho": one["rho"],
            "binding_kind": one["binding_kind"], "basis": b, "c": c, "table": theirs, "ok": ok}


def standard(m: formalism.Measured | None, path: Path, c: float, k: float, run: str | None) -> dict:
    """Standard training's row at (c, k): p 1 and `published` at reading A on its traced step `m` (module docstring;
    None: no step at `path`). Its basis is published's: of the stubs `step` labels the products term reads only τ,
    over a run's steps (a step on its own has none before it), and the code size, whose 32 bits a weight cap no
    narrower weight."""
    row = {"route": "standard training", "p": 1.0, "traced_or_modelled": "traced", "run": run}
    if m is None:
        return row | {"binding_kind": "unmeasured", "rests_on": f"no traced step ({path.name})", "basis": "unmeasured",
                      "traced_or_modelled": None, "run": None}
    one = formalism.published(m, c, k, "A")
    return row | {"rho": finite(one["rho"]), "S": finite(one["rho"]), "binding_kind": one["binding_kind"],
                  "rests_on": rests(one), "reading": one["reading"],
                  "rho_products": finite(one["terms"].get("fresh_products")), "basis": basis(one, False),
                  "products_inputs": products_inputs(m, c, k)}


def unpriced(cands: list[dict], priced: dict, measured: dict, pol: str, c: float, k: float) -> tuple[str, int] | None:
    """The commonest reason the products term has no figure among the candidates `cands` at (c, k), and on how many;
    None where it has one on each. A record whose products were classified at another k alone (a record keeps them at
    the dial's k) says so; any other reason is te_core.formalism's, to its first clause."""
    off: dict[str, int] = {}
    for x in cands:
        key = checkpoint(x["record"])
        if key in measured and priced[(key, pol)][0]["terms"].get("fresh_products") is None:
            b = formalism.bound(measured[key], c, k)
            got = b["inputs"]["fresh_products"]
            at = got.get("k_classified")
            if at is not None and "other_program" not in got and not math.isclose(at, k, rel_tol=1e-12):
                why = f"products not classified at k = {f'{k:.3g}'.replace('e+0', 'e').replace('e+', 'e')}"
            else:
                why = "the products term is " + b["unmeasured"]["fresh_products"].split(";")[0]
            off[why] = off.get(why, 0) + 1
    return max(off.items(), key=lambda kv: kv[1]) if off else None


def trace_runs(text: str | None) -> dict[str, str]:
    """`--trace-runs`: each size's trace run, as NAME=RUN,... or one RUN for every size."""
    if not text:
        return {}
    if "=" not in text:
        return {name: text for name, _, _ in SIZES}
    return dict(x.split("=", 1) for x in text.split(","))


def build(tables: Path, ledger: Path, steps: Path, runs: dict[str, str], stamp: str,
          art: str | None = None) -> tuple[dict, dict]:
    """The panel and what it rests on: every candidate's ρ, binding and basis beside the table's at each policy it
    holds (`check`, `TABLE`) where the table is this te_core's (else `check_skipped` says whose it is), the candidates
    the scorer's rows don't give (`missing`) and the reached runs with no trace yet (`untraced`). The source names the
    tables by `art` where given (within console's `SOURCE_MAX`), else by their time: `art` is their art id, and where
    that art holds more than one set of tables, the id with the set's path in it (art:ID/PATH)."""
    from te_core import algorithm_policy

    src = tables / "algorithm-policy" / "tables" / "algorithm-policy.json"
    made = json.loads(src.read_text())["updated_at"]
    fit, starts = fit_v3(table(tables / "algorithm-policy" / "tables" / "frontier-fit.json"))
    by_label = {s["label"]: name for name, s in starts.items() if "label" in s}
    names = {name for name, _, _ in SIZES}
    attacks = [r | {"name": by_label.get(r["start"])} for r in table(src)
               if r["data"] == "v3" and not str(r["method_key"]).startswith(STANDARD)]
    reached = [r for r in attacks if r["reached"] == "yes" and r["name"] in names]
    untraced = [r for r in reached if r["basis_default"] == algorithm_policy.UNTRACED]
    reached = [r for r in reached if r["basis_default"] != algorithm_policy.UNTRACED]
    versions = sorted({str(r["te_core"]) for r in reached + untraced})
    compared = set(versions) <= {VERSION}
    rows_of = scored(tables, ledger, {checkpoint(r["record"]) for r in reached})
    shapes, measured, stub, why = policy.starts(), {}, {}, {}
    for key, row in rows_of.items():
        try:
            measured[key] = formalism.measure(row, shapes[row["start"]])
        except ValueError as e:
            why[key] = str(e)
            continue
        stub[key] = any(t.endswith(f":{algorithm_policy.STUB}")
                        for t in algorithm_policy.estimated(measured[key]).split())
    attempted = {r["name"] for r in attacks}
    rows, check, priced = [], [], {}
    for name, params, _ in SIZES:
        s = shape(params)
        l0 = starts.get(name, {}).get("l0")
        eff, lo, hi = effective(fit, l0)
        base = {"start": name, "params_total": params, "params_nonembedding": params - s["vocab"] * s["d_model"],
                "bits_per_byte": l0, "start_effective_compute": finite(eff), "start_effective_compute_lo": finite(lo),
                "start_effective_compute_hi": finite(hi), "te_core": VERSION}
        path = steps / f"{name}.products.json"
        own = step(json.loads(path.read_text())) if path.exists() else None
        for g in GAINS:
            cands = [r for r in reached if r["name"] == name and r["gain_target"] == g]
            waiting = [r for r in untraced if r["name"] == name and r["gain_target"] == g]
            for pol, (c, k, bits) in policies(params).items():
                at = base | {"gain_target": g, "policy": pol, "c_gates": finite(c), "k_bits": bits}
                label = policy_label(pol, k, bits)
                rows.append(labelled(at | standard(own, path, c, k, runs.get(name)), label))
                best, bases = None, {algorithm_policy.UNTRACED: len(waiting)} if waiting else {}
                for x in cands:
                    key = checkpoint(x["record"])
                    m = measured.get(key)
                    if m is None:
                        bases["unmeasured"] = bases.get("unmeasured", 0) + 1
                        continue
                    if (key, pol) not in priced:
                        one = formalism.published(m, c, k)
                        priced[(key, pol)] = one, basis(one, stub[key])
                    one, b = priced[(key, pol)]
                    bases[b] = bases.get(b, 0) + 1
                    if compared and pol in TABLE and TABLE[pol][0] in x:
                        check.append(compare(x, pol, one, b, c))
                    if eligible(b) and finite(one["rho"]) is not None and x["p"] is not None:
                        rank = (x["p"] * one["rho"], x["p"], x["C_gates"])
                        if best is None or rank < best[0]:
                            best = (rank, x, one, b, m)
                if best is not None:
                    (S, _, _), x, one, b, m = best
                    row = {"route": x["method_key"], "p": finite(x["p"]), "rho": finite(one["rho"]), "S": finite(S),
                           "binding_kind": one["binding_kind"], "rests_on": rests(one), "reading": one["reading"],
                           "traced_or_modelled": "traced", "run": x["record"],
                           "rho_products": finite(one["terms"].get("fresh_products")), "basis": b,
                           "products_inputs": products_inputs(m, c, k)}
                elif waiting and not cands:
                    x = min(waiting, key=lambda r: (math.inf if r["p"] is None else r["p"], r["C_gates"]))
                    row = {"route": x["method_key"], "p": finite(x["p"]), "rests_on": algorithm_policy.UNTRACED,
                           "run": x["record"], "basis": algorithm_policy.UNTRACED}
                elif name not in attempted or not cands:
                    row = {"route": NO_ATTACK if name not in attempted else "none reached"}
                else:
                    total = len(cands) + len(waiting)
                    count = ", ".join(f"{n} {b}" for b, n in sorted(bases.items(), key=lambda kv: -kv[1]))
                    off = unpriced(cands, priced, measured, pol, c, k)
                    row = {"route": "none with a full rho", "rests_on": (
                        (f"{off[0]} on {off[1]} of the {total} reached ({count})" if off
                         else f"{total} reached ({count})") +
                        ": a row that is unmeasured, rests on a stub or is partial is never the least S")}
                rows.append(labelled(at | row, label))
    p = {"id": PANEL, "title": "Red teaming by model size: standard training's rho and the frontier's least S = p*rho",
         "kind": "table", "updated_at": stamp,
         "source": f"python islands/ladder-v3/size_scaling.py build (te_core {VERSION}; the scorer's "
                   f"algorithm-policy table {art or f'of {made}'}; steps traced in "
                   f"{', '.join(sorted(set(runs.values()))) or 'no run'})",
         "note": note(), "columns": COLUMNS, "rows": [[r.get(c) for c in COLUMNS] for r in rows]}
    missing = sorted({r["record"] for r in reached if checkpoint(r["record"]) not in measured})
    skipped = None if compared else f"the table is training's slowdown code {', '.join(versions)}, not {VERSION}"
    return p, {"check": check, "check_skipped": skipped, "mismatches": [x for x in check if not x["ok"]],
               "missing": missing, "untraced": sorted({r["record"] for r in untraced}),
               "unmeasured": {f"{r}@{s}": w for (r, s), w in why.items()}}


#: rho's and S's least and greatest ratio, at 32-bit attention's products over 16-bit, over the panel's rows: the as-if
#: re-pricing's build/out/asif/ss/{control,asif-steps}/size-scaling/size-scaling.json
ATTENTION_32 = "0.43 to 0.62"


def note() -> str:
    return (f"Per size, gain, policy ({UNIT}): standard training (full-size trace, p = 1) and least S = p*rho of "
            "reached attacks, never partial, stub or unmeasured; untraced alone: least p, rho null. Grown, own 4,096-in "
            "1,024-out session, k 65,536 (at packed k_bits, rho is no less). Console (45M), serving (k 1.6e6, own "
            "100,000-in). 1B: no start. Attention's products are priced at 16 bits, as the card computes them; at 32 "
            f"bits rho and S are {ATTENTION_32} of this.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trace")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--only", default="")
    b = sub.add_parser("build")
    b.add_argument("--tables", type=Path, required=True, help="the scorer's refresh's out/tables")
    b.add_argument("--ledger", type=Path, required=True, help="the ledger the scorer read (its records/)")
    b.add_argument("--steps", type=Path, required=True, help="trace's DIR")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--trace-runs", default=None, help="RUN, or NAME=RUN,...: the run that traced each size")
    b.add_argument("--art", default=None, help="the tables' art id, for the source; art:ID/PATH where the art holds "
                                               "more than one set of tables")
    u = sub.add_parser("put")
    u.add_argument("file", type=Path)
    u.add_argument("--dry", action="store_true", help="PUT nothing: what the put would send")
    a = ap.parse_args(argv)
    if a.cmd == "trace":
        a.out.mkdir(parents=True, exist_ok=True)
        only = set(filter(None, a.only.split(",")))
        for name, params, batch in SIZES:
            if only and name not in only:
                continue
            block = trace(name, params, batch)
            (a.out / f"{name}.products.json").write_text(json.dumps(block, allow_nan=False) + "\n")
            print(json.dumps({"size": name, "gates": block["gates"], "tokens": block["tokens"],
                              "rows": len(block["rows"]), "seconds": block["seconds"]}), flush=True)
        return 0
    from te_core import console

    if a.cmd == "build":
        stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        p, info = build(a.tables, a.ledger, a.steps, trace_runs(a.trace_runs), stamp, a.art)
        refused = console.refused(p)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "size-scaling.json").write_bytes(console.body(p))
        (a.out / "check.json").write_text(json.dumps(info, indent=1, default=str) + "\n")
        print(f"{PANEL}: {len(p['rows'])} rows, {len(console.body(p))} bytes, note {len(p['note'])} characters; "
              + (f"check skipped: {info['check_skipped']}, " if info["check_skipped"] else
                 f"check {len(info['check'])} rows at {', '.join(TABLE)}, {len(info['mismatches'])} mismatched, ") +
              f"{len(info['missing'])} candidates not in the scorer's rows, {len(info['untraced'])} reached not yet "
              f"traced, {len(info['unmeasured'])} unmeasured" + (f"; REFUSED: {refused}" if refused else ""))
        return 1 if refused or info["mismatches"] or len(console.body(p)) > console.BODY_MAX or \
            len(p["note"]) > console.NOTE_MAX else 0
    p = json.loads(a.file.read_text())
    if console.refused(p) or len(console.body(p)) > console.BODY_MAX or len(p["note"]) > console.NOTE_MAX:
        raise SystemExit(f"size_scaling put: the route refuses {a.file}: {console.refused(p)}")
    if a.dry:
        body = console.body(p)
        print(f"dry: {p['id']}: sha256 {hashlib.sha256(body).hexdigest()}, {len(body)} bytes, {len(p['rows'])} rows, "
              f"note {len(p['note'])} characters; console.put would refuse: {console.refusal(p) or 'nothing'}; nothing "
              f"stored, nothing PUT")
        return 0
    print(f"PUT {console.PANELS_URL}/{p['id']}: {console.put(p)} at {datetime.now(UTC).strftime('%H:%M:%SZ')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
