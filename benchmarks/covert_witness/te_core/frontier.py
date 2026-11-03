"""The search's frontier, read from the ledger's index and scored by the training formalism v2 (te_core.formalism).

    python -m te_core.frontier [--root DIR] [--json OUT] [--pin COMMIT ...] [--facts FACTS.json]

It reads only the submissions measured by the pinned measurement commit or an earlier commit its `pinned.json` names as
measuring the same version (`same_version`; covert_witness's `pin.py`; `--pin`, repeated, names others), and leaves
out, each marked with its reason (`status`): an invalid record, one whose read-outs `cw.readout_leaks` lists
(`read-out-leak`, invalid: the coordinator's 14:22Z rule), an unscored one, one measured by any other commit
(`other-version`), one whose E digest (`output.e_sha256`) equals an earlier record's (`duplicate-e`: the first to submit
an E is the one its compute is counted for), and one a later ingest of its run supersedes (`re-ingested`). A row
ingested before 3.13 carries no read-outs; `--facts` gives them by run id (`with_facts`). Under 4.0 an update of the
tied embedding alone is deep (its lookup's rows are update-dependent and reach every later layer), so the scorer's
tied-lookup rule (coordinator 13:32Z) is retired. `measured(r)` says whether a row's quantities were measured on its
record's traced step (te_core.rescore scores only those by default); any other row's are stubs, labelled. A row whose
quantities te_core.formalism refuses (a trace before 4.2's read-back lists, a data-dependent capture, a Python number
folded over steps, a nonempty E with no update in its trace or prefix: `formalism.refusal`) is never scored: it is
skipped with its reason (`refused: ...`).

Each valid submission is scored by the formalism (`formalism_rows`): |C| in Boolean gates, ρ at every (c, k) of its
start's policy grid and at its reference shapes (te_core.policy's `cells`; c/(c − d) where Fit holds for the training,
Slowdown's target elsewhere). Per start and l* = its bits per byte less 0.5%, 1%, 2% and 5% (`formalism_report`), by the
coordinator's 19:50Z ruling (`target`): honest H(l*) is the least |C| over every checkpoint of every valid submission,
of any class or direction, that reaches l*, W*(l*) the least |C| ρ over the same checkpoints, and Security's
S = W*/H ≥ 1, with W*'s route, its x0 and s* (the assumptions' bounds, and the attack side's best-known values) and
every quantity's source. The reference directions' H alone is kept beside it as context.

`candidates(rows, T)` are the frontier members (no other submission beats them on both |C| ρ and val bits per byte)
and the elites (per niche, mechanism tag x affected-share range x start, the least val bits per byte, the least |C| ρ
breaking ties) at each reference shape, among the submissions at tier T - 1 not yet promoted to T; which of them are
promoted is decided on the sealed promotion set, one bit each (covert_witness's `promote.py --tier T`, run as the
scorer).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from . import formalism, policy
from .ledger import Ledger, version

AFFECTED_EDGES = (0.1, 0.25, 0.5, 0.75)
TARGETS = (0.005, 0.01, 0.02, 0.05)
REF_DIRECTIONS = ("ref-optim", "ref-data")


def bin_of(x: float, edges: tuple[float, ...]) -> int:
    return sum(x >= e for e in edges)


#: the first te_core whose ingest measures the formalism's quantities from the traced step circuit (a row's `te_core`)
MEASURED_FROM = (4, 0)
#: a skipped row's reason when te_core.formalism refuses its quantities
REFUSED = "refused: "


def _run_key(r: dict):
    return (r["run_id"], r.get("measurement_commit"), r.get("checkpoint_step")) if r.get("run_id") else None


def with_facts(rows: list[dict], facts: dict[str, dict]) -> list[dict]:
    """The rows with what the rules read beyond the price, by run id (`facts`: run id -> any of `args`,
    `readout_leaks` and `changed`, as `ledger.facts` gives them), for rows ingested before te_core 3.13; and with each
    checkpoint's E facts the row's `output` lacks (`checkpoints`: step -> `rebuild_macs`, `decoder_gates`), for rows
    ingested before 4.2."""
    out = []
    for r in rows:
        f = dict(facts.get(r.get("run_id"), {}))
        cp = f.pop("checkpoints", {}).get(str(r.get("checkpoint_step")))
        out.append(r | f | ({"output": cp | (r.get("output") or {})} if cp else {}))
    return out


def pinned(pilot: bool = False) -> tuple[str, ...]:
    """The pin's commit, then the earlier commits that measure the same version (`same_version`), and with `pilot` the
    pilot's (`pilot`: an earlier version's commits, whose records te_core.panels shows beside the pin's, labelled by
    their data)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pin import pinned as read

    pin = read()
    return (pin["commit"], *pin.get("same_version", []), *(pin.get("pilot", []) if pilot else []))


def commits(pin: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return (pin,) if isinstance(pin, str) else tuple(pin)


def status(rows: list[dict], pin: str | tuple[str, ...]) -> dict[str, str]:
    """Each left-out record's reason: re-ingested, invalid, read-out-leak, unscored, other-version, duplicate-e.
    `pin` is a commit or the commits of one version. A later ingest of the same run's checkpoint (its run id; a row
    without one is never superseded) supersedes an earlier one (`re-ingested`), which then seeds no duplicate. An E
    repeats an earlier
    record's when that record was measured at the pin or is another agent's: an agent's own relaunch at a new pin of an
    E it had at an old one counts."""
    pins = set(commits(pin))
    out, first = {}, {}
    ordered = sorted(rows, key=lambda r: (r.get("written_ts", 0), r["id"]))
    latest = {_run_key(r): r["id"] for r in ordered}
    for r in ordered:
        e = (r.get("output") or {}).get("e_sha256")
        key = _run_key(r)
        if key is not None and latest[key] != r["id"]:
            out[r["id"]] = "re-ingested"
            continue
        if e and any(at_pin or agent != r.get("agent") for at_pin, agent in first.get(e, ())):
            out[r["id"]] = "duplicate-e"
        elif r.get("invalid"):
            out[r["id"]] = "invalid"
        elif r.get("readout_leaks"):
            out[r["id"]] = "read-out-leak"
        elif r.get("val_bpb") is None:
            out[r["id"]] = "unscored"
        elif r.get("measurement_commit") not in pins:
            out[r["id"]] = "other-version"
        if e and out.get(r["id"]) != "duplicate-e":
            first.setdefault(e, set()).add((r.get("measurement_commit") in pins, r.get("agent")))
    return out


def measured(r: dict) -> bool:
    """Whether a row's formalism quantities were measured on its record's traced step: ingested by te_core 4.0 or
    later with the step's traced structure (`ledger ingest --structures TRACED.json`, `annotate.with_structures`),
    every updated step traced (none `untraced`), its block carrying the trace. Any other row's φ, τ, a, u, t and Δ_grad
    are stubs; `python -m te_core.ledger --agent AGENT ingest RUN_DIR ... --run-id RUN --structures TRACED.json`
    measures it."""
    return (version(r.get("te_core")) >= MEASURED_FROM and not (r.get("annotation") or {}).get("untraced")
            and "trace" in (r.get("formalism") or {}))


def valid(rows: list[dict], pin: str | tuple[str, ...] | None = None) -> list[dict]:
    left_out = status(rows, pin or pinned())
    return [r for r in rows if r["id"] not in left_out]


def cost(r: dict, key: str, size: str = "gates") -> float:
    return r[size] * r["rho"][key]


def pareto(rows: list[dict], key: str) -> list[dict]:
    pts = sorted(rows, key=lambda r: (cost(r, key), r["val_bpb"]))
    out, best = [], math.inf
    for r in pts:
        if r["val_bpb"] < best:
            out.append(r)
            best = r["val_bpb"]
    return out


def affected_share(m: formalism.Measured) -> float:
    """|A ∩ ∪T| / |C|: the share of |C| in the token blocks' affected gates."""
    return sum(p.affected_gates for p in m.parts) / m.gates if m.gates else 0.0


def niche(r: dict) -> tuple:
    return (r.get("mechanism", ""), bin_of(affected_share(r["measured"]), AFFECTED_EDGES), r["start"])


def elites(rows: list[dict], key: str) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for r in rows:
        n = niche(r)
        if n not in out or (r["val_bpb"], cost(r, key)) < (out[n]["val_bpb"], cost(out[n], key)):
            out[n] = r
    return out


def _of(r: dict | None) -> dict | None:
    return r and {"id": r["id"], "checkpoint": r.get("checkpoint_step", r.get("cp")), "direction": r.get("direction"),
                  "mechanism": r.get("mechanism")}


def target(on: list[dict], l_star: float, keys, reference=None, size: str = "gates") -> dict:
    """The coordinator's 19:50Z ruling at one l*, over one start's valid submissions `on` (each a checkpoint, with
    `id`, its |C| in `size`, `val_bpb` and `rho`). honest(l*) is the least |C| over every checkpoint of every
    submission, of any class or direction, that reaches l* (a checkpoint is a model someone reached; the sealed final
    reruns guard against picking noise), and W*(l*) in each cell of `keys` the least |C| ρ over the same checkpoints,
    so W*/honest ≥ 1 (ρ ≥ 1) and is the effective slowdown; `fit` is whether Fit holds for the training at W*'s
    checkpoint (None: the row doesn't say). `honest_reference` is the least |C| over the reference directions'
    checkpoints alone, as context (`reference` picks them, by default a direction in `REF_DIRECTIONS`). |C| is the
    formalism's gates (`formalism_targets`)."""
    is_ref = reference or (lambda r: r.get("direction") in REF_DIRECTIONS)
    reach = [r for r in on if r.get("val_bpb") is not None and r["val_bpb"] <= l_star]
    h = min(reach, key=lambda r: r[size], default=None)
    ref = min((r for r in reach if is_ref(r)), key=lambda r: r[size], default=None)
    out = {"l_star": l_star, "honest": h and h[size], "honest_of": _of(h), "honest_reference": ref and ref[size],
           "honest_reference_of": _of(ref), "cells": {}}
    for key in keys:
        w = min(reach, key=lambda r: cost(r, key, size), default=None)
        out["cells"][key] = {"w_star": w and cost(w, key, size), "of": _of(w), "rho": w and w["rho"][key],
                             "fit": w and (w.get("fit") or {}).get(key),
                             "effective_slowdown": cost(w, key, size) / h[size] if w and h[size] else None}
    return out


def start_bpb(on: list[dict]) -> float | None:
    """A start's l0: the whole-pool bits per byte its rows carry (the median, should they differ in the last digits)."""
    v = sorted(r["start_bpb"] for r in on if r.get("start_bpb") is not None)
    return v[len(v) // 2] if v else None


def formalism_rows(rows: list[dict], shapes: dict[str, dict], ladder=None) -> tuple[list[dict], dict[str, str]]:
    """Each row whose start has a shape and that the scoring cap admits (te_core.policy's `eligible`, under the ladder's
    fit `ladder`), with its formalism quantities (`measured`), |C| in gates, ρ in every cell of its start (te_core.policy's
    `cells`: the grid's and the reference shapes'), whether Fit holds there for the training (`fit`) and Slowdown's terms
    there (`terms`: x0, s* and the attack side's); and each row left out, with why (`no-shape`; `pilot-start`, a record
    on data v3 from a start whose shape has another tokenizer, as the pilot's FineWeb-Edu starts do, since v3's branch
    points are v3's starts; `over-cap`; `unmeasured`; or `refused: ` and te_core.formalism's reason,
    `formalism.refusal`)."""
    out, skipped, cells = [], {}, {}
    for r in rows:
        shape = shapes.get(r["start"])
        if shape is None:
            skipped[r["id"]] = "no-shape"
            continue
        if r.get("data") == "v3" and shape.get("tokenizer") != policy.V3_TOKENIZER:
            skipped[r["id"]] = "pilot-start"
            continue
        if not policy.eligible((r.get("model") or {}).get("forward_gates_per_token"),
                               r.get("start_forward_gates_per_token") or policy.w_fwd(shape), r.get("val_bpb"), ladder):
            skipped[r["id"]] = "over-cap"
            continue
        if r["start"] not in cells:
            cells[r["start"]] = policy.cells(shape)
        try:
            m = formalism.measure(r, shape)
        except formalism.Refused as e:
            skipped[r["id"]] = f"{REFUSED}{e}"
            continue
        except ValueError:
            skipped[r["id"]] = "unmeasured"
            continue
        rho, fit, terms = formalism.rhos(m, cells[r["start"]])
        out.append(r | {"gates": m.gates, "rho": rho, "fit": fit, "terms": terms, "measured": m})
    return out, skipped


TERMS = ("x0", "s_star", "x0_known", "s_star_known")


def formalism_targets(rows: list[dict], shapes: dict[str, dict]) -> dict[str, dict]:
    """Per start (its rows `formalism_rows` gives): its reference shapes, and at each l* = l0 less each of `TARGETS`
    honest H(l*) (least gates over every valid checkpoint reaching l*, `target`), and per cell of te_core.policy's
    `cells` (the reference shapes by name) W*(l*) (least gates·ρ over the same checkpoints), the checkpoint that sets it
    (`of`), its ρ, S = W*/H, `basis` (`fit` where Fit holds for the training there, ρ = c/(c − d) by Fit; else `target`,
    Slowdown's bound, a target) and its Slowdown terms there (`TERMS`: the bounds on x0 and s*, and the attack side's
    best-known values)."""
    out = {}
    for start in sorted({r["start"] for r in rows}):
        on = [r for r in rows if r["start"] == start]
        by_id = {r["id"]: r for r in on}
        l0 = start_bpb(on)
        if l0 is None:
            continue
        keys = list(policy.cells(shapes[start]))
        out[start] = {"l0": l0, "shape": shapes[start], "marks": policy.shapes(shapes[start]), "targets": []}
        for f in TARGETS:
            t = target(on, l0 * (1 - f), keys, size="gates")
            cells = {}
            for k, v in t["cells"].items():
                w = v["of"] and by_id[v["of"]["id"]]
                cells[k] = {"w_star": v["w_star"], "of": v["of"], "rho": v["rho"], "S": v["effective_slowdown"],
                            "basis": None if w is None else "fit" if v["fit"] else "target",
                            **{n: w and w["terms"][k][n] for n in TERMS}}
            out[start]["targets"].append({"drop": f, "l_star": t["l_star"], "honest": t["honest"],
                                          "honest_of": t["honest_of"], "cells": cells})
    return out


def formalism_report(rows: list[dict], pin: str | tuple[str, ...] | None = None,
                     shapes: dict[str, dict] | None = None, keep=None, ladder=None) -> dict:
    """The formalism's frontier over the valid rows (`valid`) that `keep` keeps (default all): `formalism_targets`,
    with every quantity's source (the labels of the rows' measurements) and the rows it couldn't score."""
    shapes = policy.starts() if shapes is None else shapes
    scored, skipped = formalism_rows([r for r in valid(rows, pin) if keep is None or keep(r)], shapes, ladder)
    labels: dict[str, dict[str, int]] = {}
    for r in scored:
        for k, v in r["measured"].labels.items():
            labels.setdefault(k, {}).setdefault(v, 0)
            labels[k][v] += 1
    return {"starts": formalism_targets(scored, shapes), "sources": labels, "skipped": skipped, "scored": len(scored)}


def _route(of: dict | None) -> str:
    return f"{of['direction']}/{of['mechanism']} {of['id'][-12:]}@{of['checkpoint']}" if of else ""


def _num(x: float | None) -> str:
    return "—" if x is None else f"{x:.3g}"


def formalism_markdown(rep: dict) -> str:
    lines = ["Security's S = W*/H per start and target at each reference shape (te_core.policy), by the training "
             "formalism v2. H and W* read every checkpoint of every valid record (19:50Z), so S >= 1; ρ is c/(c − d) "
             "where Fit holds for the training (basis fit) and Slowdown's target elsewhere (basis target), with A1's and "
             "A1''s upper bounds on s* and x0; |C| in Boolean gates; each S with W*'s ρ, basis and route "
             "(direction/mechanism record@checkpoint):", "",
             "| start | target | l* | H (gates; route) | " + " | ".join(f"S at {m}" for m in policy.MARKS) + " |",
             "|---|---|---|---|" + "---|" * len(policy.MARKS)]
    for start, v in rep["starts"].items():
        for t in v["targets"]:
            h = "—" if t["honest"] is None else f"{t['honest']:.3g} ({_route(t['honest_of'])})"
            ss = []
            for m in policy.MARKS:
                cell = t["cells"].get(m, {})
                ss.append("—" if cell.get("S") is None else
                          f"{cell['S']:.3g} (ρ {cell['rho']:.3g}, {cell['basis']}; {_route(cell['of'])})")
            lines.append(f"| {start} | -{100 * t['drop']:g}% | {t['l_star']:.4f} | {h} | " + " | ".join(ss) + " |")
    ref = policy.REFERENCE
    lines += ["", f"At the {ref} policy, W*'s Slowdown terms: the claim's bounds x0 <= max{{x : u_in(x) <= k}} (A1') and "
              "s* <= max_x x(k − u(x))/a (A1), beside the attack side's best-known values from exhibited blocks "
              "(tightness, never the claim):", "",
              "| start | target | S | ρ | basis | route | x0 bound | x0 known | s* bound | s* known |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for start, v in rep["starts"].items():
        for t in v["targets"]:
            c = t["cells"].get(ref, {})
            lines.append(f"| {start} | -{100 * t['drop']:g}% | {_num(c.get('S'))} | {_num(c.get('rho'))} | "
                         f"{c.get('basis') or '—'} | {_route(c.get('of'))} | {_num(c.get('x0'))} | "
                         f"{_num(c.get('x0_known'))} | {_num(c.get('s_star'))} | {_num(c.get('s_star_known'))} |")
    lines += ["", "Sources:"] + [f"- {k}: " + "; ".join(f"{n} rows {s}" for s, n in sorted(v.items()))
                                  for k, v in sorted(rep["sources"].items())]
    return "\n".join(lines)


def candidates(rows: list[dict], tier: int, pin: str | tuple[str, ...] | None = None,
               shapes: dict[str, dict] | None = None) -> list[str]:
    """The frontier members and elites at each reference shape (policy.MARKS), among the valid
    submissions run at tier - 1 and not yet promoted to `tier` that the formalism scores (`formalism_rows`)."""
    shapes = policy.starts() if shapes is None else shapes
    pool = [r for r in valid(rows, pin) if r["tier"] == tier - 1 and r.get("promoted", 0) < tier]
    scored, _ = formalism_rows(pool, shapes)
    ids = set()
    for key in policy.MARKS:
        ids |= {r["id"] for r in pareto(scored, key)} | {r["id"] for r in elites(scored, key).values()}
    return sorted(ids)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--pin", action="append", default=None,
                    help="a measurement commit, repeated for one version's (default covert_witness's pin.py)")
    ap.add_argument("--facts", type=Path, default=None,
                    help="JSON, run id -> args, readout_leaks, changed, checkpoints (ledger.facts)")
    ap.add_argument("--formalism", action="store_true",
                    help="accepted for te_core 3.14-3.18's command lines: the formalism is the only score")
    ap.add_argument("--promote", type=int, choices=(1, 2), default=None, help=argparse.SUPPRESS)
    ap.add_argument("--as", dest="agent", default="", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.promote:
        print(f"promotions are decided on the sealed promotion set, one bit each: from benchmarks/covert_witness/, "
              f"`python promote.py --tier {a.promote}` (the scorer's)", file=sys.stderr)
        return 2
    rows = Ledger(a.root).index()
    if a.facts:
        rows = with_facts(rows, json.loads(a.facts.read_text()))
    pin = tuple(a.pin) if a.pin else pinned()
    rep = formalism_report(rows, pin)
    left_out = status(rows, pin)
    if a.json:
        a.json.write_text(json.dumps({"pin": list(pin), "left_out": left_out} | rep, indent=1, default=str) + "\n")
    print(formalism_markdown(rep))
    reasons = sorted(set(left_out.values()))
    refused = sum(v.startswith(REFUSED) for v in rep["skipped"].values())
    print(f"pin {' + '.join(c[:12] for c in pin)}: {rep['scored']} scored of {len(rows)} submissions, "
          f"{refused} refused, {len(rep['skipped']) - refused} without a start shape or measurement; left out: "
          + (", ".join(f"{sum(v == x for v in left_out.values())} {x}" for x in reasons) or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
