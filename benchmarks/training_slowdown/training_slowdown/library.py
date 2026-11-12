"""The library of training algorithms, and each entry's slowdown under its policy.

    python -m training_slowdown.library [ENTRY ...] [--starts starts.toml] [--out DIR]

An entry is a config of `attacks/` (`training_slowdown.config`): a route and its dials at a start, with the policy it
is read under. There is one kind of entry, whatever its route computes: a new one is a config of a route `routes/`
has, or a new route and a config of it. ENTRY is a config's file or its name there; none is every entry.

For each entry the command builds its route's model at the start's shape, traces its step on a CPU as a job does
after its last step (`traced.trace_step`: on fake tensors, so it trains nothing and reads no weights and no data), and
reads the slowdown on that step from the benchmark's one entry point (`calculator.slowdown`): the least slowdown of
any partition of the traced step into the policy's blocks, by the slowdown calculator's bound of record on the work
the theorem charges (the products whose two operands both depend on the update). It prints one row an entry: the
share of the start's parameters its step trains (`trains`), its slowdown and what binds, and how much of the step's
gates the capture approximates where it does (`approximated`); then the entry's planned run at that slowdown over
each fleet and period (`periods`). With `--out` it writes the rows with what each read to `DIR/library.json`.

A row has no slowdown, and says why, where the capture refuses the step, where the step breaks a premise the rows'
rates take, where the trace doesn't classify the products at the policy's k, and where the entry's E keeps less than
the trained weights (`CODED`): a weight then costs at most E's bits a trained weight, which only a run's record has,
so the grade reads that entry (`training_slowdown.grade`).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from . import calculator, config, policy, starts, traced
from .codecs import Identity
from .grade import per_period
from .opnames import LINALG_NAMES

SCHEMA = "library/v1"
#: the entries' directory
ENTRIES = Path(__file__).resolve().parents[1] / "attacks"
CODED = ("needs a record: the entry's E keeps less than the trained weights, so a weight costs at most E's bits a "
         "trained weight, which a trace alone doesn't have")
FLOOR = "the floor: blocks of at most c gates cover the step's gates, and the theorem's count asks for no more"
EMPTY = "the floor: no product of the step has two operands the update reaches, so the theorem charges nothing"
TRAINS = ("trains: the share of the start's parameters the traced step updates. An entry that trains less charges "
          "fewer products, so its slowdown is not its optimizer's.")
PARTS = {
    "traced": "each entry's step, by the capture on fake tensors at the config's batch and the start's shape: its "
              "gates, its tokens, the parameters it updates and its products' MACs, output wires and operand "
              "widths. No entry was trained",
    "approximated": "a step's dense linear algebra (an eigendecomposition, an SVD, a QR or a Cholesky factor, an "
                    "inverse, a solve), which the capture counts at n³ MACs a matrix and not as the gates of the "
                    "algorithm that computes it. A row whose step has any says what share of the step's gates "
                    "that count is; the slowdown is over the step's gates, that count among them",
    "proved": "the bound's form: the statement named beside each slowdown, a lower bound on every partition of the "
              "traced step into the policy's blocks, given its two premises and its hypotheses on the circuit, "
              "which the trace doesn't check",
    "premise": "(H) and (E), conjectures that one set of gates refutes, at the bits a value the row states",
    "scaled": "the planned run: the config's steps times the traced step's gates, so the traced step stands for "
              "every step; and every share: a fleet's AND gates are its registry's peak FLOP a second, two a MAC "
              f"at the {calculator.FLEET_PRECISION} GEMM's AND gates, times its locations and the period, and the "
              f"run's gates are AND gates at the {policy.SERVED_PRECISION} GEMM's ratio. The fleets' sizes and the "
              "window are stated parameters",
}


def entries(names: list[str] = ()) -> list[Path]:
    """The entries' config files: each of `names` (a file, or a name in `ENTRIES`), or every entry by name."""
    if not names:
        return sorted(ENTRIES.glob("*.toml"), key=lambda p: p.stem)
    out = []
    for name in names:
        path = next((p for p in (Path(name), ENTRIES / f"{name}.toml") if p.is_file()), None)
        if path is None:
            have = ", ".join(p.stem for p in entries())
            raise SystemExit(f"library: no entry {name!r} (a config's file, or a name in {ENTRIES}: {have})")
        out.append(path)
    return out


def classified(trace: dict, k: float) -> dict | None:
    """The trace's product rows as classified at k: its own block, or its dial's block at k (`at`); None at a k the
    trace keeps no rows at."""
    return next((b for b in (trace, *trace["at"].values()) if math.isclose(b["k"], k, rel_tol=1e-12)), None)


def bound(trace: dict, c: float, k: float) -> dict:
    """The slowdown at (c, k) on a traced step (`traced.trace_step`'s), as `calculator.slowdown` gives it on the trace
    at the step's own gates: its `slowdown` and what `binds`, with the calculator's bound (`bound`) and the choice
    of the work it read (`figure`). No slowdown, and why in `binds`, on a step the capture refused, on one whose
    elementwise ops fan out into a product's operand (its `p_match`), at a k the trace has no rows at, and past the
    last k (H) states an activation's β at. The floor, 1, where M is empty or the theorem's count is under the
    blocks the gates fill."""
    if "refused" not in trace and classified(trace, k) is None:
        at = ", ".join(f"{b['k']:g}" for b in (trace, *trace["at"].values()))
        return {"slowdown": None, "binds": f"unmeasured: the trace classifies the step's products at k = {at}, and "
                                           f"the policy's k is {k:g}"}
    got = calculator.slowdown({"c_gates": c, "k_bits": k}, trace)
    if got["slowdown"] is None:
        return {"slowdown": None, "binds": got["why"]}
    figure, b = got["figure"], got["bound"]
    if figure["subset"] is None:
        return {"slowdown": got["slowdown"], "binds": EMPTY, "figure": figure}
    binds = FLOOR
    if got["binds"] == "products":
        through = f", an operand reached through a rank-{figure['r_step']} bottleneck" if figure["below_width"] else ""
        binds = (f"{b['theorem'].rsplit('.', 1)[-1]} on {figure['subset']}, {figure['macs_share']:.0%} of the MACs "
                 f"whose two operands the update reaches (β_w {figure['beta_w']:.3g}, β_x {figure['beta_x']:.3g}, "
                 f"b_y {figure['b_y']:g}, F {figure['F']:.4g}{through})")
    return {"slowdown": got["slowdown"], "binds": binds, "bound": b, "figure": figure}


def trains(graph, of: int) -> dict:
    """What a traced step trains, from the capture's graph (`traced.traced_step`'s): the parameters the step writes (a
    buffer it writes, such as an average of the weights, is none), and their share of the start's `of` parameters.
    Adapters a route adds beside the start's weights count over the start's."""
    n = sum(math.prod(p["shape"]) for p in graph.params if p["trained"])
    return {"parameters": n, "of": of, "share": n / of}


def approximated(graph) -> dict:
    """What the capture approximates of a traced step, from its graph: the step's dense linear algebra
    (`opnames.LINALG_NAMES`), counted at n³ MACs a matrix (`StepGraph.approx_macs`) and not as the gates of the
    algorithm that computes it. Those MACs, their gates as the step's circuit holds them, and their share of the
    step's gates."""
    c = graph.circuit
    ops = [i for i, n in enumerate(graph.graph.graph.nodes) if n.op == "call_function"
           and getattr(getattr(n.target, "overloadpacket", None), "__name__", "") in LINALG_NAMES]
    g = float(c.gates[np.isin(c.node, ops)].sum())
    return {"macs": graph.approx_macs, "gates": g, "share": g / float(c.gates.sum())}


def row(cfg: config.Config) -> dict:
    """An entry's row: its route's step traced at its start's shape and its budget's batch (`step`: its gates, its
    tokens, the capture that took it and what that capture approximates), the parameters the step trains (`trains`),
    and the bound of record on it at the entry's policy (`bound`). An entry whose plan keeps less than the weights
    isn't traced: `CODED` stands in place of its slowdown."""
    route = config.route(cfg.route)
    pol = policy.named(cfg.policy, cfg.start.shape)
    out = {"entry": cfg.name, "route": cfg.route, "dials": cfg.dials, "start": cfg.start.name, "budget": cfg.budget,
           "policy": pol}
    if not all(isinstance(c, Identity) for _, c in route.plan(cfg.dials).rules):
        return out | {"slowdown": None, "binds": CODED}
    model, optimizers = route.build(cfg.start.shape, cfg.dials)
    trace, graph = traced.traced_step(cfg, route, model, optimizers, "cpu")
    if graph is not None:
        out |= {"step": {"gates": trace["circuit"]["gates"], "tokens": trace["tokens"], "capture": trace["capture"],
                         "capture_code": trace["capture_code"], "approximated": approximated(graph)},
                "trains": trains(graph, cfg.start.shape["parameters"])}
    return out | bound(trace, pol["c_gates"], pol["k_bits"])


def periods(r: dict) -> dict | None:
    """An entry's planned run at its slowdown, over all compute: the config's steps times the traced step's gates
    times the slowdown, in AND gates, and their share of each fleet's compute over each period, as the grade
    presents a run (`grade.per_period`). None for a row with no slowdown."""
    if r.get("slowdown") is None:
        return None
    over = per_period(r["budget"]["steps"] * r["step"]["gates"] * r["slowdown"], {"binding_kind": "products"})
    return {"run_and_gates": over["run_and_gates"], "shares": over["rows"]}


def table(head: list[str], lines: list[list[str]]) -> str:
    width = [max(len(x[i]) for x in (head, *lines)) for i in range(len(head) - 1)]
    return "\n".join("  ".join([*(x.ljust(w) for x, w in zip(line, width)), line[-1]]).rstrip()
                     for line in (head, *lines))


def line(r: dict) -> list[str]:
    """A row's cells: `trains` as a share of the start's parameters, and after what binds, where the capture
    approximated any of the step, how much of its gates."""
    share = r.get("step", {}).get("approximated", {}).get("share")
    note = f"; approximated: {share:.2%} of the step's gates are linear algebra at n³ MACs a matrix" if share else ""
    return [r["entry"], r["route"], f"{100 * r['trains']['share']:.3g}%" if "trains" in r else "-",
            "-" if r["slowdown"] is None else f"{r['slowdown']:.4g}", r["binds"] + note]


def text(rows: list[dict]) -> str:
    """The rows as the command prints them: the table and what its `trains` says, each statement's full name, the
    planned runs over each fleet and period, and which parts are traced, approximated, proved, premises and
    scaled."""
    out = [table(["entry", "route", "trains", "slowdown", "what binds"], [line(r) for r in rows]), TRAINS]
    for b in sorted({(r["bound"]["theorem"], r["bound"]["rests_on"]) for r in rows if "bound" in r}):
        out.append(f"\n{b[0].rsplit('.', 1)[-1]}: {b[1]}")
    over = [(r, p) for r in rows if (p := periods(r))]
    if over:
        names = list(calculator.periods())
        for fleet in dict.fromkeys(s["over"] for s in over[0][1]["shares"]):
            out.append(f"\nthe planned run at its slowdown, as a share of {fleet}'s compute over each period")
            out.append(table(["entry", "run, AND gates", *names], [
                [r["entry"], f"{p['run_and_gates']:.3g}",
                 *(f"{s['share']:.3g}" for s in p["shares"] if s["over"] == fleet)] for r, p in over]))
    return "\n".join(out + ["", *(f"{k}: {v}." for k, v in PARTS.items())])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("entry", nargs="*", help="an entry: a config's file, or its name in attacks/ (default: every one)")
    ap.add_argument("--starts", type=Path, default=starts.REGISTRY, help="the starts registry")
    ap.add_argument("--out", type=Path, default=None, help="where library.json goes")
    a = ap.parse_args(argv)
    reg = starts.registry(a.starts)
    rows = [row(config.load(p, reg)) for p in entries(a.entry)]
    print(text(rows))
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "library.json").write_text(json.dumps(
            {"schema": SCHEMA, "rows": [r | {"periods": periods(r)} for r in rows], "parts": PARTS}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
