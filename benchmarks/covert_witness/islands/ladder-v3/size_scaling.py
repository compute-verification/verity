"""Plot 3's panel `training/size-scaling`: how the red-teaming study scales with model size. Per size (the ladder's v3
starts and the console's 1B reference size), gain target and policy, standard training's ρ beside the measured
frontier's least S = p·ρ, in the dashboard's columns (`COLUMNS`).

    python islands/ladder-v3/size_scaling.py trace --out DIR [--only NAME,...]   # on a node, with torch: the steps
    python islands/ladder-v3/size_scaling.py build --tables DIR --steps DIR --out DIR [--trace-run RUN]
    python islands/ladder-v3/size_scaling.py put FILE                            # on the VM: PUT a built body

Run from `benchmarks/covert_witness/`. c and |C| are Boolean gates (te_core.gates), k bits.

**Sizes** (`SIZES`): s-t20, m-t20, l16-t20 and xl-t20, each at its shape (te_core.policy's `SHAPES`) and the ladder's
step (ladder.py: its batch of 1,024-token sequences in micro-batches of `MICRO`), and the console's 1B reference size
(te_core.policy's `REFERENCE_SIZES`, vocabulary 50,304), which no start has, at xl's step. A start's bits per byte and
effective compute are the v3 fit's (te_core.frontier_fit: C_eff at l0, lo and hi at l0 plus and minus the fit's rms in
loss); 1B has neither.

**Standard training** of a size is the ladder's step, te_core.io_bound's `dense` recipe (Muon on the block matrices,
AdamW on the tied embedding and the norm gains) traced at full size under fake tensors (`trace`): its |C| and its
products at each k of the console's dial (te_core.capture.products' `rows_on_dial`) and at the serving policy's k. Its
ρ at (c, k) is the products term on that step as one part of |C| gates (`rho`), as te_core.rho_vectors prices its
full-size steps, and its p is 1: it is the honest route.

**Policies** (`policies`): `console`, te_core.policy's `CONSOLE`; `grown`, one session of the size's own model at the
console's 4,096 input and 1,024 output tokens, at the console's k; `serving`, k = `SERVING_K` (100,000 input ids at 16
bits) and c one session of the size's own model at 100,000 input and 1,024 output tokens (te_core.policy's `session`).

**The measured frontier** (`frontier`): the scorer's uncut algorithm-policy table (te_core.algorithm_policy), each
run's first checkpoint reaching each gain (`reached` yes, p = C/H), every route but standard training's (`STANDARD`),
each checkpoint priced at each policy by the same products term on its quantities in the scorer's formalism.jsonl
(`record`), which at the console's policy is the table's own ρ wherever it binds on the products (`check`). The row is
the one with the least S, else null S with why: no attack run at this size, none reached, or no ρ at the policy.
"""
from __future__ import annotations

import argparse
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

# te_core.console and te_core.frontier_fit (through te_core.plots) read matplotlib, which the trace's torch environment
# hasn't, so `build` and `put` import them

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
COLUMNS = ["start", "params_total", "params_nonembedding", "bits_per_byte", "start_effective_compute",
           "start_effective_compute_lo", "start_effective_compute_hi", "gain_target", "policy", "route", "p", "rho", "S",
           "binding_kind", "rests_on", "reading", "te_core", "traced_or_modelled", "run", "c_gates", "k_bits"]
STEP_RESTS_ON = ("computed from the traced op graph, premises assumed: the products term on the ladder's step at full "
                 "size, one part of |C| gates")
NO_ATTACK = "no attack run at this size"


def shape(params: int) -> dict:
    s = policy.SHAPES[params]
    if policy.parameters(s) != params:
        raise SystemExit(f"size_scaling: te_core.policy's shape for {params:,} has {policy.parameters(s):,} parameters")
    return s


def policies(params: int) -> dict[str, tuple[float, float]]:
    """The three policies (module docstring) for the size of `params` parameters, each (c, k)."""
    s = shape(params)
    return {"console": policy.CONSOLE,
            "grown": (policy.session_gates(params, *policy.CONSOLE_SESSION, s), policy.CONSOLE[1]),
            "serving": (policy.session_gates(params, *SERVING_SESSION, s), float(SERVING_K))}


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

def measured(parts: tuple, products: dict | None) -> formalism.Measured:
    """A Measured that only the products term reads: `parts` (its |C|) and `products`."""
    flat = ((0.0, 0.0), (1.0, 0.0))
    return formalism.Measured(parts, 0.0, 0.0, flat, flat, None, products=products)


def step(block: dict) -> formalism.Measured:
    """A traced step as one part of its |C| gates with its products (te_core.rho_vectors' `grid`)."""
    C = block["gates"]
    return measured((formalism.Part(gates=C, tokens=1.0, token_gates=C, affected_gates=C),),
                    formalism.products_of_trace(block))


def record(f: dict) -> formalism.Measured:
    """A checkpoint's parts and products from its formalism block, the products taken again from its trace's rows by
    this te_core's rules (te_core.formalism's `complete`)."""
    parts = tuple(formalism._part(p, {}) for p in f["parts"])
    return measured(parts, formalism.traced_products(f.get("products"), (f.get("trace") or {}).get("products")))


def rho(m: formalism.Measured, c: float, k: float) -> dict:
    """ρ at (c, k): the products term, floored at 1 (te_core.formalism's `published` where only that kind binds), with
    why where it has no figure."""
    term, why, got = formalism.fresh_products(m, c, k)
    if term is None:
        return {"rho": None, "binding_kind": "unmeasured", "why": why}
    return {"rho": max(1.0, term), "binding_kind": "fresh_products" if term > 1 else "none", "why": why,
            "rests_on": formalism.RESTS_ON_CELL["fresh_products" if term > 1 else "none"]}


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


def formalism_blocks(path: Path, wanted: set[tuple[str, int]]) -> dict[tuple[str, int], dict]:
    """The formalism blocks of the checkpoints `wanted` ((run, step)), from the scorer's formalism.jsonl."""
    runs = {r for r, _ in wanted}
    out = {}
    with path.open() as fh:
        for line in fh:
            head = line[:300]
            if not any(r in head for r in runs):
                continue
            d = json.loads(line)
            key = (d.get("run_id"), d.get("checkpoint_step"))
            if key in wanted and d.get("formalism"):
                out[key] = d["formalism"]
    return out


def checkpoint(rec: str) -> tuple[str, int]:
    run, at = rec.rsplit("@", 1)
    return run, int(at)


def finite(v: float | None) -> float | None:
    return None if v is None or math.isinf(v) else float(f"{v:.4g}")


def build(tables: Path, steps: Path, trace_run: str | None, stamp: str) -> tuple[dict, dict]:
    """The panel and what it rests on (the console check, `check`)."""
    src = tables / "algorithm-policy" / "tables" / "algorithm-policy.json"
    made = json.loads(src.read_text())["updated_at"]
    ap = table(src)
    fit, starts = fit_v3(table(tables / "algorithm-policy" / "tables" / "frontier-fit.json"))
    by_label = {s["label"]: name for name, s in starts.items() if "label" in s}
    v3 = [r | {"name": by_label.get(r["start"])} for r in ap if r["data"] == "v3"]
    reached = [r for r in v3 if r["reached"] == "yes" and not str(r["method_key"]).startswith(STANDARD)]
    blocks = formalism_blocks(tables / "formalism.jsonl", {checkpoint(r["record"]) for r in reached})
    attempted = {r["name"] for r in v3 if not str(r["method_key"]).startswith(STANDARD)}
    rows, check, priced = [], [], {}
    for name, params, _ in SIZES:
        s = shape(params)
        nonemb = params - s["vocab"] * s["d_model"]
        st = starts.get(name, {})
        l0 = st.get("l0")
        eff, lo, hi = effective(fit, l0)
        head = [name, params, nonemb, l0, finite(eff), finite(lo), finite(hi)]
        path = steps / f"{name}.products.json"
        own = step(json.loads(path.read_text())) if path.exists() else None
        for g in GAINS:
            cands = [r for r in reached if r["name"] == name and r["gain_target"] == g]
            for pol, (c, k) in policies(params).items():
                tail = [finite(c), k]
                if own is None:
                    rows.append(head + [g, pol, "standard training", 1.0, None, None, "unmeasured",
                                        f"no traced step ({path.name})", None, VERSION, None, None] + tail)
                else:
                    r = rho(own, c, k)
                    rows.append(head + [g, pol, "standard training", 1.0, finite(r["rho"]), finite(r["rho"]),
                                        r["binding_kind"], f"{STEP_RESTS_ON}; {r.get('rests_on') or r['why']}", "A",
                                        VERSION, "traced", trace_run] + tail)
                best = None
                for x in cands:
                    key = checkpoint(x["record"])
                    if key not in blocks:
                        continue
                    if (key, pol) not in priced:
                        priced[(key, pol)] = rho(record(blocks[key]), c, k)
                    got = priced[(key, pol)]
                    if pol == "console" and x["binding_kind"] == "fresh_products":
                        check.append((x["record"], got["rho"], x["rho_console"]))
                    if got["rho"] is not None and x["p"] is not None:
                        S = x["p"] * got["rho"]
                        if best is None or S < best[0]:
                            best = (S, x, got)
                if best is not None:
                    S, x, got = best
                    rows.append(head + [g, pol, x["method_key"], finite(x["p"]), finite(got["rho"]), finite(S),
                                        got["binding_kind"], got.get("rests_on") or got["why"], "A", VERSION,
                                        "measured", x["record"]] + tail)
                else:
                    why = (NO_ATTACK if name not in attempted else "none reached" if not cands
                           else "no rho at this policy")
                    rows.append(head + [g, pol, why, None, None, None, None, None, None, VERSION, None, None] + tail)
    bad = [(rec, mine, theirs) for rec, mine, theirs in check
           if theirs is None or mine is None or not math.isclose(mine, float(theirs), rel_tol=1e-9)]
    p = {"id": PANEL, "title": "Red teaming by model size: standard training's rho and the frontier's least S = p*rho",
         "kind": "table", "updated_at": stamp,
         "source": f"python islands/ladder-v3/size_scaling.py build (training's slowdown code {VERSION}; frontier: the "
                   f"scorer's algorithm-policy table of {made}; steps traced in {trace_run or 'no run'})",
         "note": note(), "columns": COLUMNS, "rows": rows}
    return p, {"check": check, "mismatches": bad}


def note() -> str:
    return (f"Per size, gain target and policy ({UNIT}): standard training (the ladder's step traced at full size, p = 1)"
            " and the measured frontier's least S = p*rho among reached attack routes. console: the 45M session; grown:"
            " the size's own 4,096-in 1,024-out session at k 65,536; serving: k 1.6e6, its own 100,000-in session. rho:"
            " training's slowdown code's products term, floored at 1. 1B: the console's reference size, no start.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trace")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--only", default="")
    b = sub.add_parser("build")
    b.add_argument("--tables", type=Path, required=True, help="the scorer's refresh's out/tables")
    b.add_argument("--steps", type=Path, required=True, help="trace's DIR")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--trace-run", default=None)
    u = sub.add_parser("put")
    u.add_argument("file", type=Path)
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
        p, info = build(a.tables, a.steps, a.trace_run, stamp)
        refused = console.refused(p)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "size-scaling.json").write_bytes(console.body(p))
        (a.out / "check.json").write_text(json.dumps(info, indent=1) + "\n")
        print(f"{PANEL}: {len(p['rows'])} rows, {len(console.body(p))} bytes, note {len(p['note'])} characters; "
              f"console check {len(info['check'])} rows, {len(info['mismatches'])} mismatched"
              + (f"; REFUSED: {refused}" if refused else ""))
        return 1 if refused or len(console.body(p)) > console.BODY_MAX else 0
    p = json.loads(a.file.read_text())
    if console.refused(p) or len(console.body(p)) > console.BODY_MAX:
        raise SystemExit(f"size_scaling put: the route refuses {a.file}: {console.refused(p)}")
    print(f"PUT {console.PANELS_URL}/{p['id']}: {console.put(p)} at {datetime.now(UTC).strftime('%H:%M:%SZ')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
