"""`verity bench prove layers`: the proof benchmark by layer, one table of its four benchmark results (Flock, Flock
with zero knowledge, with sampling, with recursion), each read from what its own command wrote
(benchmarks/prove/README.md, "The proof benchmark by layer").

    verity bench prove layers [--flock DIR|RUN] [--zk DIR|RUN] [--sampling DIR|RUN] [--recursion DIR|RUN]

DIR is a run's ``out/`` (or the run directory that holds it), read with no store; RUN is a run id, whose records are
fetched through the evidence store where `research` is installed. A layer not given is a benchmark result that says
"not run". Which reader a directory takes is decided by what it holds (`detect`):

- ``recursion_e2e.json``: `benchmarks/recursion_e2e`'s job (`read_recursion`);
- an arm's ``audit.log`` with its ``PROVE-PATH`` line, or its ``RECORDS`` line under ``PP_RECORDS=1``:
  `benchmarks/prove_path/run.sh` (`read_prove_path`);
- ``summary.txt``, ``a5.log`` or the audit's ``summary.json``: `benchmarks/one_stage/a5_llama1b_h200.sh`, or `a5.sh`'s
  out (`read_a5`).

A benchmark result prints its numbers only when the Lean verifier's verdict, as its output states it, is an accept;
any other prints none, with why. Its ``ran with: coins COINS; parameters PARAMETERS (DETAIL)`` line names only what its output
says outright (`COINS`, `PARAMETERS`, "not stated" for a half it doesn't say, ``ran with: not stated`` for both), and
the JSON keeps the output's own words as ``detail``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from verity_bench import kit
from verity_bench.proof_layers_readers import (
    READERS,
    per_unit,
    read_a5,
    read_layers_sh,
    read_prove_path,
    read_recursion,
)
from verity_bench.proof_layers_result import (
    A5,
    ACCEPTS,
    CLAUSE,
    COINS,
    FETCH,
    FIGURES,
    FIX,
    LAYERS,
    LAYERS_SH,
    LAYERS_SH_DIRS,
    MEMBERS,
    NOT_AN_AUDIT,
    NOT_STATED,
    PARAMETERS,
    PROOF,
    PROVE_PATH,
    RECURSION,
    REPORT,
    RUN_ID,
    Result,
    SCHEMA,
    STAND_IN,
    Unreadable,
    VALUES_NOT_STATED,
)

__all__ = [
    "A5",
    "ACCEPTS",
    "CLAUSE",
    "COINS",
    "FETCH",
    "FIGURES",
    "FIX",
    "LAYERS",
    "LAYERS_SH",
    "LAYERS_SH_DIRS",
    "MEMBERS",
    "NOT_AN_AUDIT",
    "NOT_STATED",
    "PARAMETERS",
    "PROOF",
    "PROVE_PATH",
    "READERS",
    "RECURSION",
    "REPORT",
    "RUN_ID",
    "Result",
    "SCHEMA",
    "STAND_IN",
    "Unreadable",
    "VALUES_NOT_STATED",
    "add_arguments",
    "detect",
    "document",
    "fetch",
    "has_number",
    "locate",
    "measured",
    "per_unit",
    "read",
    "read_a5",
    "read_layers_sh",
    "read_prove_path",
    "read_recursion",
    "render",
    "run",
]


# ---- finding a benchmark result's output --------------------------------------------------------------------------------


def fetch(run: str) -> Path:
    """The root of `run`'s record as the evidence store holds it, with `FETCH`'s files fetched."""

    try:
        from research.store import labels_sync
        from research.store.local import LocalStore, StoreError
    except ImportError as e:
        raise Unreadable(f"{run}: the evidence store's client isn't installed here ({e}): {FIX}") from e
    store = LocalStore()
    try:
        attempt = store.get_attempt(run)
        if attempt is None:
            raise Unreadable(f"{run} is not in the evidence store this machine reads: {FIX}")
        outputs = attempt.get("outputs") or {}
        art = outputs.get("run_record") or outputs.get("run_files")
        if not art:
            raise Unreadable(f"{run} kept no record")
        labels_sync.read_through(store, [run, art], lambda m: print(f"prove layers: {m}", file=sys.stderr))
        if why := store.withdrawn(art) or store.withdrawn(run):
            raise Unreadable(f"{run} was withdrawn ({why}): nothing of it is read")
        return Path(store.fetch(art, paths=list(FETCH)))
    except StoreError as e:
        raise Unreadable(f"{run}'s record can't be fetched: {e}: {FIX}") from e


def detect(d: Path) -> str | None:
    """Which command's output `d` holds, by its files; None where it holds none of them."""

    if (d / "recursion_e2e.json").is_file():
        return RECURSION
    if (d / "audit.log").is_file() or any(d.glob("*/audit.log")):
        return PROVE_PATH
    if (d / "layers.txt").is_file() and (d / "a6" / "summary.json").is_file():
        return LAYERS_SH
    if any((d / f).is_file() for f in ("summary.txt", "a5.log", "a5/a5.log", "a5/a5/summary.json", "a5/summary.json")):
        return A5
    return None


def locate(given: str, fetcher: Callable[[str], Path]) -> Path:
    """The directory `given` names: a directory (its ``out/`` where it holds no output itself), or a run id's fetched
    ``out/``."""

    p = Path(given)
    if p.is_dir():
        return p / "out" if detect(p) is None and (p / "out").is_dir() else p
    if RUN_ID.fullmatch(given):
        root = fetcher(given)
        return root / "out" if (root / "out").is_dir() else root
    raise Unreadable(f"{given} is neither a directory nor a run id")


# ---- reading -------------------------------------------------------------------------------------------------------------


def read(layer: str, given: str | None, fetcher: Callable[[str], Path] = fetch) -> Result:
    """The benchmark result for `layer` from the output `given` names (None: not run)."""

    result = Result(layer)
    if given is None:
        result.why = f"not run: no output given (--{layer} DIR|RUN)"
        return result
    result.given, result.status = given, "read"
    try:
        d = locate(given, fetcher)
        if detect(d) is None and layer in LAYERS_SH_DIRS and detect(own := d / LAYERS_SH_DIRS[layer] / "out"):
            d = own
        result.format = detect(d)
        if result.format is None:
            raise Unreadable(f"{d} holds none of the outputs this reads (recursion_e2e.json, an arm's audit.log, "
                             "summary.txt or a5.log, layers.sh's layers.txt)")
        if result.format not in ACCEPTS[layer]:
            raise Unreadable(f"--{layer} takes {' or '.join(ACCEPTS[layer])}'s output, and {d} holds {result.format}'s")
        READERS[result.format](d, result)
    except (Unreadable, OSError, ValueError, KeyError, TypeError) as e:
        result.status, result.lean, result.workload = "unreadable", "not read", "not read"
        result.withhold(f"unreadable: {e}")
        return result
    if layer in PROOF and result.proof != PROOF[layer]:
        result.withhold(f"--{layer} is the {PROOF[layer]} proof's benchmark result, and this output "
                     + (f"says its proof is {result.proof}" if result.proof else "states no proof"))
    if not result.accepted:
        result.withhold(f"the Lean verdict is not an accept ({result.lean})")
    return result


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
    lines += ["", "| layer | workload | proof | device | ε at δ | prover | Lean verifier | Lean | proof bytes |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        read = r.status == "read"
        proof = r.proof or (NOT_STATED if read else "—")
        cells = ((_statements(r, "prove_s", "Flock's prove_total_s"), _statements(r, "lean_s", "Lean's verify"),
                  _statements(r, "proof_bytes", "one rep's proof")) if r.statements else
                 (_cell(r.prover_s, 's', r.per, r.prover_what), _cell(r.lean_s, 's', r.per, r.lean_what),
                  _cell(r.proof_bytes, 'bytes', r.per, r.bytes_what)))
        lines.append(f"| {LAYERS[r.layer]} | {r.workload} | {proof} | {r.device if read else '—'} | "
                     f"{r.eps_delta if read else '—'} | {cells[0]} | {r.lean} | {cells[1]} | {cells[2]} |")
    lines += ["", *(f"- {LAYERS[r.layer]}: no number: {r.why}" for r in results if r.why)]
    lines += [f"- {LAYERS[r.layer]}: {n}" for r in results for n in r.notes]
    return "\n".join(lines).rstrip()


def document(results: Sequence[Result]) -> dict[str, Any]:
    """The table as JSON. ``audit`` is always false: the command checks no challenger's signature."""

    return {"schema": SCHEMA, "audit": False, "results": [{"title": LAYERS[r.layer], **asdict(r)} for r in results]}


def measured(results: Sequence[Result], metrics: Mapping[str, str]) -> list[dict[str, Any]]:
    """The writer's measurements: under each of `metrics`' ids (each naming one of `FIGURES`), one a layer of `MEMBERS`,
    in its order. A layer's value is its figure a proof block, only where its Lean verdict is an accept; else the value is
    None, with why. Recursion's benchmark result is no member: its seconds are a whole recursive run's."""

    if bad := sorted(set(metrics.values()) - set(FIGURES)):
        raise ValueError(f"metric ids name {bad}, not one of {sorted(FIGURES)}")
    by_layer = {r.layer: r for r in results}
    out = []
    for name, figure in metrics.items():
        for layer, member in MEMBERS.items():
            r = by_layer[layer]
            value = getattr(r, figure) if r.accepted and r.per_block else None
            why = None if value is not None else r.why or (
                f"{r.workload.split(':')[0]}: its figures are a statement's, not a proof block's" if r.statements else
                f"its output gives no {figure} a proof block" if r.per_block else "its output gives no figure a proof block")
            out.append({"name": name, "value": value, "unit": FIGURES[figure], "member": member, "why": why,
                        "inputs": [r.given] if r.given else [],
                        "extra": {"layer": LAYERS[layer], "status": r.status, "format": r.format, "workload": r.workload,
                                  "lean": r.lean, "ran_with": r.ran_with, "detail": r.detail, "proof": r.proof}})
    return out


def add_arguments(parser: argparse.ArgumentParser) -> None:
    named = {"flock": "the Flock layer", "zk": "the zero-knowledge layer", "sampling": "the sampling layer",
             "recursion": "the recursion layer"}
    for layer in LAYERS:
        parser.add_argument(f"--{layer}", metavar="DIR|RUN",
                            help=f"the output the benchmark result for {named[layer]} is read from: a run's out/ directory, "
                                 "or a run id fetched through the store")


def run(args: argparse.Namespace) -> int:
    results = [read(layer, getattr(args, layer)) for layer in LAYERS]
    kit.report(REPORT, render(results), document(results), measured(results, getattr(args, "metrics", None) or {}))
    return 0 if has_number(results) else 1
