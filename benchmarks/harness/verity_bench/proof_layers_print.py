"""The table `verity bench prove layers` prints (`render`), its JSON (`document`) and the measurements a runner
writes (`measured`), from the benchmark results `proof_layers` reads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from verity_bench.proof_layers_result import (
    FIGURES,
    LAYERS,
    MEMBERS,
    NOT_AN_AUDIT,
    NOT_STATED,
    SCHEMA,
    Result,
)

# ---- printing and writing ------------------------------------------------------------------------------------------------


def _cell(x: float | None, unit: str, per: str, what: str) -> str:
    """A figure with its unit as its first words (seconds a sampled proof block, or a whole run's wall seconds), then how it was
    read."""

    if x is None:
        return "—"
    figure = f"{x:,.0f} bytes" if unit == "bytes" else f"{x:.4g} s"
    wall = ", wall" if unit == "s" and per == "a run" else ""
    return f"{figure} {per}{wall}" + (f" ({what})" if what else "")


def _statements(r: Result, key: str, what: str) -> str:
    """A layers.sh benchmark result's cell: each statement's figure, its unit first and a statement's, never a proof block's."""

    cells = [(f"{x[key]:,.0f} bytes" if key == "proof_bytes" else f"{x[key]:.3g} s") + f" a statement of {x['units']:,} proof blocks"
             for x in r.statements if x.get(key) is not None]
    return "; ".join(cells) + f" ({what})" if cells else "—"


def has_number(results: Sequence[Result]) -> bool:
    return any(x is not None for r in results for x in (r.prover_s, r.lean_s, r.proof_bytes)) or any(
        x.get(k) is not None for r in results for x in r.statements for k in ("prove_s", "lean_s", "proof_bytes"))


def render(results: Sequence[Result]) -> str:
    """The table as text: its title; `NOT_AN_AUDIT` unless every benchmark result ran on a deployed challenger's coins at deployed
    parameters (and the JSON's ``audit`` is false regardless); each benchmark result's ``ran with:`` line; the table; then why each
    benchmark result without a number has none. With no number in any benchmark result its first line says so (`run` exits 1)."""

    lines = [] if has_number(results) else ["prove layers: no number: " + "; ".join(f"{r.layer}: {r.why}" for r in results)]
    lines.append("The proof benchmark by layer: the benchmark result for each layer, read from its own run's output "
                 "(`verity bench prove layers`).")
    if not all(r.ran_with.startswith("ran with: coins deployed challenger; parameters deployed") for r in results):
        lines.append(NOT_AN_AUDIT)
    lines += [f"- {LAYERS[r.layer]}: {r.ran_with}" for r in results]
    lines.append("- commits: " + "; ".join(f"{LAYERS[r.layer]} " + (f"{r.commit[:11]} ({r.commit_from})" if r.commit else
                                                                     NOT_STATED if r.status == "read" else "—")
                                           for r in results)
                 + "; a benchmark result reruns from main only where its commit is on main (`git merge-base --is-ancestor COMMIT origin/main`)")
    lines += ["", "| layer | workload | proof | device | ε at δ | prover | Lean verifier | Lean | proof bytes | circuit build |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        read = r.status == "read"
        proof = r.proof or (NOT_STATED if read else "—")
        cells = ((_statements(r, "prove_s", "Flock's prove_total_s"), _statements(r, "lean_s", "Lean's verify"),
                  _statements(r, "proof_bytes", "one rep's proof")) if r.statements else
                 (_cell(r.prover_s, 's', r.per, r.prover_what), _cell(r.lean_s, 's', r.per, r.lean_what),
                  _cell(r.proof_bytes, 'bytes', r.per, r.bytes_what)))
        build = "—" if r.build_s is None else f"{r.build_s:.0f} s a build, wall ({r.build_what}; build cache: {r.build_cache})"
        lines.append(f"| {LAYERS[r.layer]} | {r.workload} | {proof} | {r.device if read else '—'} | "
                     f"{r.eps_delta if read else '—'} | {cells[0]} | {r.lean} | {cells[1]} | {cells[2]} | {build} |")
    lines += ["", *(f"- {LAYERS[r.layer]}: no number: {r.why}" for r in results if r.why)]
    lines += [f"- {LAYERS[r.layer]}: {n}" for r in results for n in r.notes]
    return "\n".join(lines).rstrip()


def document(results: Sequence[Result]) -> dict[str, Any]:
    """The table as JSON. ``audit`` is always false: the command checks no challenger's signature."""

    return {"schema": SCHEMA, "audit": False, "results": [{"title": LAYERS[r.layer], **asdict(r)} for r in results]}


def measured(results: Sequence[Result], metrics: Mapping[str, str]) -> list[dict[str, Any]]:
    """The writer's measurements: under each of `metrics`' ids (each naming one of `FIGURES`), one a layer of `MEMBERS`,
    in its order. A layer's value is its figure a proof block, or for ``build_s`` its build step's wall where its output
    states whether the build reused a cache, only where its Lean verdict is an accept; else the value is None, with why. Recursion's benchmark result is no member: its seconds are a
    whole recursive run's."""

    if bad := sorted(set(metrics.values()) - set(FIGURES)):
        raise ValueError(f"metric ids name {bad}, not one of {sorted(FIGURES)}")
    by_layer = {r.layer: r for r in results}
    out = []
    for name, figure in metrics.items():
        for layer, member in MEMBERS.items():
            r = by_layer[layer]
            build = figure == "build_s"
            value = getattr(r, figure) if r.accepted and (r.build_cache != NOT_STATED if build else r.per_block) else None
            why = None if value is not None else r.why or (
                "its output gives no build step's wall (steps.txt)" if build and r.build_s is None else
                "its steps.txt states no build cache: line, so not whether the build reused a cache" if build else
                f"{r.workload.split(':')[0]}: its figures are a statement's, not a proof block's" if r.statements else
                f"its output gives no {figure} a proof block" if r.per_block else "its output gives no figure a proof block")
            out.append({"name": name, "value": value, "unit": FIGURES[figure], "member": member, "why": why,
                        "inputs": [r.given] if r.given else [],
                        "extra": {"layer": LAYERS[layer], "status": r.status, "format": r.format, "workload": r.workload,
                                  "lean": r.lean, "ran_with": r.ran_with, "detail": r.detail, "proof": r.proof,
                                  "prover_batch": r.prover_batch,
                                  **({"build_cache": r.build_cache} if build else {})}})
    return out
