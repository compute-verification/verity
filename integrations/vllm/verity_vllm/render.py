"""Render ``docs/fixtures/results.md`` from ``fixtures/results/*/table{1,2,3,4}.jsonl``.

Four Markdown tables (contract section 9), rows sorted by ``case`` then ``experiment_id``,
experiment ids rendered as links to the result directory.  Every row is validated with
:func:`rows.validate_row` and every Table 1 ``result`` is recomputed with :func:`rows.verdict`
(using the experiment's Table 2 rows); over-claims and invalid rows are listed in an *Issues*
section instead of being dropped, so a hand-assigned ``PASS`` is visible rather than silently
rendered.

Entry point::

    python -m verity_vllm.render [--results DIR] [--out FILE] [--link-base PREFIX] [--check]

``--check`` exits 1 when any issue was found (for CI).  Output is deterministic (no timestamps).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import rows as R

TABLE_FILES = {t: f"table{t}.jsonl" for t in (1, 2, 3, 4)}
TITLES = {
    1: "Table 1 - cases (one row per case x concrete profile)",
    2: "Table 2 - op implementations (one row per profile x op x shape family x decomposition)",
    3: "Table 3 - targeted negative tests",
    4: "Table 4 - cost (one row per case x partition x instrumentation mode)",
}
SIDE_FILES = ("manifest.json", "description.json", "commitments.json", "log.txt", "timings.json")
AUDIT_FILE = "audit.json"  # WB2: an independent verifier's receipt for the row (see collect)
CONFORMANCE_PREFIX = "CONF-"  # lane CH: `CONF-<port>-<utc>/` holds a profile port's conformance run, not table rows
CONFORMANCE_LEVEL_FILE = "level.json"
CONFORMANCE_LEVELS = (1, 2, 3, 4)


def repo_root() -> Path | None:
    """``<root>`` when this module lives at ``<root>/verity_vllm/render.py``."""
    here = Path(__file__).resolve()
    if len(here.parents) >= 3 and here.parents[1].name == "vllm-poc":
        return here.parents[2]
    return None


def default_results_dir() -> Path:
    root = repo_root()
    if root is not None and (root / "vllm-poc").is_dir():
        return root / "fixtures" / "results"
    return Path.cwd() / "results"


def default_out_path() -> Path:
    root = repo_root()
    if root is not None and (root / "docs" / "vllm-poc").is_dir():
        return root / "docs" / "vllm-poc" / "results.md"
    return Path.cwd() / "results.md"


# --- collection ----------------------------------------------------------------------------------


@dataclass(slots=True)
class Experiment:
    experiment_id: str
    path: Path
    tables: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    side_files: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Collected:
    experiments: list[Experiment]
    issues: list[str]
    # configuration-conformance runs (`CONF-<port>-<utc>/level.json`, lane CH): one record per run directory
    conformance: list[dict[str, Any]] = field(default_factory=list)

    def rows(self, table: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for e in self.experiments:
            out.extend(e.tables.get(table, []))
        return sorted(out, key=lambda r: sort_key(table, r))


def conformance_problems(level: Mapping[str, Any]) -> list[str]:
    """Structural checks on a conformance run's `level.json`: the headline is never above the evidence
    (it equals the highest level whose status is `met`), and every level 1-4 carries a status."""
    problems: list[str] = []
    levels = level.get("levels")
    if not isinstance(levels, Mapping):
        return ["field 'levels' missing"]
    met: list[int] = []
    for n in CONFORMANCE_LEVELS:
        rec = levels.get(str(n))
        if not isinstance(rec, Mapping) or "status" not in rec:
            problems.append(f"level {n}: no status")
            continue
        if rec.get("met"):
            met.append(n)
    headline = level.get("level")
    if headline != (max(met) if met else 0):
        problems.append(f"headline level {headline!r} is not the highest met level ({max(met) if met else 0})")
    if not isinstance(level.get("port"), str):
        problems.append("field 'port' missing")
    return problems


def conformance_rows(collected_conformance: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The latest run per port (by the run's `utc`), sorted by port."""
    latest: dict[str, Mapping[str, Any]] = {}
    for rec in collected_conformance:
        port = str(rec["level"].get("port"))
        if port not in latest or str(rec["level"].get("utc", "")) > str(latest[port]["level"].get("utc", "")):
            latest[port] = rec
    return [dict(latest[k]) for k in sorted(latest)]


def sort_key(table: int, row: Mapping[str, Any]) -> tuple[str, ...]:
    base = (str(row.get("case", "")), str(row.get("experiment_id", "")))
    if table == 2:
        return base + (
            str(row.get("op_id", "")),
            str(row.get("phase", "")),
            str(row.get("decomposition", "")),
        )
    if table == 3:
        return base + (str(row.get("negative_id", "")),)
    if table == 4:
        return base + (str(row.get("partition_id", "")), str(row.get("instrumentation_mode", "")))
    return base + (str(row.get("profile_id", "")),)


def collect(results_dir: str | Path) -> Collected:
    """Read every ``<results_dir>/*/table*.jsonl``; validate rows; recompute Table 1 verdicts."""
    results_dir = Path(results_dir)
    experiments: list[Experiment] = []
    issues: list[str] = []
    collected_conformance: list[dict[str, Any]] = []
    if not results_dir.is_dir():
        return Collected([], [f"results directory {results_dir} does not exist"])
    for exp_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        if exp_dir.name.startswith(CONFORMANCE_PREFIX):
            # a conformance run carries no table rows: its level.json is the record (docs/vllm-poc/config-conformance.md)
            level_path = exp_dir / CONFORMANCE_LEVEL_FILE
            if not level_path.exists():
                issues.append(f"{exp_dir}: no {CONFORMANCE_LEVEL_FILE} found")
                continue
            try:
                level = json.loads(level_path.read_text())
            except json.JSONDecodeError as error:
                issues.append(f"{level_path}: {error}")
                continue
            problems = conformance_problems(level)
            issues.extend(f"{level_path}: {p}" for p in problems)
            collected_conformance.append({"experiment_id": exp_dir.name, "path": exp_dir, "level": level})
            continue
        exp = Experiment(exp_dir.name, exp_dir)
        exp.side_files = [name for name in SIDE_FILES if (exp_dir / name).exists()]
        for table, filename in TABLE_FILES.items():
            path = exp_dir / filename
            if not path.exists():
                continue
            try:
                loaded = R.read_jsonl(path)
            except R.RowError as error:
                issues.append(str(error))
                continue
            kept: list[dict[str, Any]] = []
            for number, row in enumerate(loaded, 1):
                if not isinstance(row, dict):
                    issues.append(f"{path}:{number}: row is not an object")
                    continue
                problems = R.validate_row(table, row, check_verdict=False)
                if row.get("experiment_id") not in (None, exp_dir.name):
                    problems.append(
                        f"experiment_id {row.get('experiment_id')!r} does not match "
                        f"directory {exp_dir.name!r}"
                    )
                for p in problems:
                    issues.append(f"{path}:{number}: {p}")
                kept.append(row)
            exp.tables[table] = kept
        if not exp.tables:
            issues.append(f"{exp_dir}: no table*.jsonl found")
        # WB2 (GUIDANCE §15.2): an auditor's re-verification of the row from the archive with its own seed is
        # attached as <experiment>/audit.json and becomes the row's `verification_evidence` (independently
        # verified, citing the report); rows without one carry what the verifier wrote (or the derived default).
        audit_path = exp_dir / AUDIT_FILE
        audit = None
        if audit_path.exists():
            try:
                audit = json.loads(audit_path.read_text())
            except json.JSONDecodeError as error:
                issues.append(f"{audit_path}: {error}")
        for row in exp.tables.get(1, []):
            try:
                row["verification_evidence"] = R.verification_evidence(row, audit)
            except R.RowError as error:
                issues.append(f"{audit_path}: {error}")
        experiments.append(exp)
    for exp in experiments:
        for row in exp.tables.get(1, []):
            if R.validate_row(1, row, check_verdict=False):
                continue  # malformed; already reported
            try:
                recomputed = R.verdict(row, exp.tables.get(2, []))
            except R.RowError as error:
                issues.append(f"{exp.path / TABLE_FILES[1]}: {error}")
                continue
            if R.overclaims(row["result"], recomputed.result):
                row["_rule_result"] = recomputed.result
                row["_rule_reason"] = recomputed.first_blocker
                issues.append(
                    f"{exp.path / TABLE_FILES[1]}: case {row.get('case')} says {row['result']} "
                    f"but the section 10 rule (with this experiment's Table 2) says "
                    f"{recomputed.result}: "
                    f"{recomputed.first_blocker}"
                )
    return Collected(experiments, issues, collected_conformance)


# --- formatting ----------------------------------------------------------------------------------


def esc(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def frac(d: Mapping[str, Any] | None, num: str, den: str) -> str:
    if not isinstance(d, Mapping):
        return "-"
    return f"{d.get(num, '-')}/{d.get(den, '-')}"


def num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.{digits}g}" if abs(value) < 1e6 else f"{value:.3e}"
    return esc(value)


def shape_cell(value: Any) -> str:
    if isinstance(value, list | tuple):
        return "[" + ",".join(esc(v) for v in value) + "]"
    return esc(value) if value is not None else "-"


def compact(value: Any) -> str:
    """One-cell rendering of a list/object: ``k=v, k=v`` or ``a; b``."""
    if value is None:
        return "-"
    if isinstance(value, Mapping):
        return ", ".join(f"{k}={num(v)}" for k, v in value.items()) or "-"
    if isinstance(value, list | tuple):
        return (
            "; ".join(
                compact(v) if isinstance(v, Mapping | list | tuple) else num(v) for v in value
            )
            or "-"
        )
    return num(value)


def link(experiment_id: str, link_base: str) -> str:
    base = link_base.rstrip("/")
    return f"[{esc(experiment_id)}]({base}/{experiment_id}/)"


def markdown_table(header: Sequence[str], body: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for cells in body:
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def common_cells(row: Mapping[str, Any], link_base: str) -> list[str]:
    return [
        link(str(row.get("experiment_id", "")), link_base),
        esc(row.get("case")),
        esc(row.get("profile_id")),
        esc(row.get("worker")),
        esc(row.get("adapter_revision")),
    ]


COMMON_HEADER = ["experiment_id", "case", "profile_id", "worker", "adapter_revision"]

T1_HEADER = COMMON_HEADER + [
    "workload",
    "production_changes",
    "replay_restrictions",
    "token_selection",
    "boundary_noninterference",
    "required_coverage",
    "authentication",
    "relation_checks",
    "structural",
    "negative",
    "bindings",
    "verification_evidence",
    "result",
    "first_blocker",
]


def bindings_cell(row: Mapping[str, Any]) -> str:
    """WB's C3 bindings + WB2's logprob status: ``model ok, architecture ok, output ok; logprobs: unbound``."""
    ch = row.get("challenge") or {}
    b = ch.get("bindings") if isinstance(ch, Mapping) else None
    parts = []
    if isinstance(b, Mapping):
        for k in ("model", "architecture", "output"):
            if k in b:
                parts.append(f"{k} {'ok' if b[k] else 'FAILED'}")
    parts.append(f"logprobs: {R.logprobs_label(row)}")
    return esc("; ".join(parts) if len(parts) > 1 else "no C3 bindings; " + parts[0])


def evidence_cell(row: Mapping[str, Any]) -> str:
    ev = row.get("verification_evidence") or {}
    label = esc(ev.get("label") or "-")
    if ev.get("report"):
        label += f" ({esc(ev.get('verifier') or 'auditor')}: `{esc(ev['report'])}`)"
    return label


def t1_cells(row: Mapping[str, Any], link_base: str) -> list[str]:
    w = row.get("workload", {}) or {}
    bni = row.get("boundary_noninterference", {}) or {}
    rel = row.get("relation_checks", {}) or {}
    st = row.get("structural", {}) or {}
    # the verdict is unqualified only under security-mode challenge randomness (rows.result_label);
    # a raw public --seed (every row before the staged challenge) is a diagnostic challenge
    result = esc(row.get("result"))
    if "_rule_result" in row:
        result = f"**{result}** (rule: {esc(row['_rule_result'])})"
    label = R.result_label(row)
    if label != str(row.get("result")):
        result += " " + label[len(str(row.get("result"))):].strip()
    return common_cells(row, link_base) + [
        (
            f"req={num(w.get('requests'))}, pos={num(w.get('generated_positions'))}, "
            f"len={compact(w.get('input_len_range'))}, "
            f"conc={num(w.get('peak_concurrency_achieved'))}"
        ),
        esc(row.get("production_changes")),
        compact(row.get("replay_restrictions")),
        frac(row.get("token_selection"), "exact", "attempted"),
        (
            f"identical={num(bni.get('identical'))}, compared={num(bni.get('compared'))}, "
            f"mismatched={num(bni.get('mismatched_scalars'))}, "
            f"ref={esc(bni.get('reference_method'))}"
        ),
        frac(row.get("required_coverage"), "bound", "required"),
        frac(row.get("authentication"), "valid", "attempted"),
        (
            f"same_runtime {frac(rel.get('same_runtime'), 'exact', 'attempted')}; "
            f"independent {frac(rel.get('independent'), 'exact', 'attempted')}"
        ),
        f"{esc(st.get('result'))}"
        + (f" ({esc(st['failing_invariant'])})" if st.get("failing_invariant") else ""),
        frac(row.get("negative"), "rejected", "attempted"),
        bindings_cell(row),
        evidence_cell(row),
        result,
        esc(row.get("first_blocker")) or "-",
    ]


T2_HEADER = COMMON_HEADER + [
    "op_id",
    "phase",
    "shape",
    "dtype",
    "layout_or_quant",
    "runtime_impl",
    "primitive_semantics",
    "decomposition",
    "interface",
    "origin",
    "binding",
    "checker",
    "natural_checks",
    "stress_checks",
    "mutations",
    "checker_work",
    "status",
    "limitation",
]


def t2_cells(row: Mapping[str, Any], link_base: str) -> list[str]:
    iface = row.get("interface", {}) or {}
    stress = row.get("stress_checks", {}) or {}
    work = row.get("checker_work", {}) or {}
    return common_cells(row, link_base) + [
        esc(row.get("op_id")),
        esc(row.get("phase")),
        shape_cell(row.get("shape")),
        esc(row.get("dtype")),
        esc(row.get("layout_or_quant")) or "-",
        esc(row.get("runtime_impl")),
        esc(row.get("primitive_semantics")),
        esc(row.get("decomposition")),
        (
            f"in={num(iface.get('input_bytes'))} B, out={num(iface.get('output_bits'))} bit, "
            f"range={compact(iface.get('range'))}"
        ),
        esc(row.get("origin")),
        frac(row.get("binding"), "present", "unresolved").replace("/", " present / ")
        + " unresolved",
        esc(row.get("checker")),
        frac(row.get("natural_checks"), "exact", "attempted"),
        f"{frac(stress, 'exact', 'attempted')}"
        + (f" ({esc(stress['domain'])})" if stress.get("domain") else ""),
        frac(row.get("mutations"), "rejected", "attempted"),
        (
            f"{num(work.get('seconds'))} s, {num(work.get('flops_or_bytes'))}, "
            f"full_op={num(work.get('includes_full_op'))}"
        ),
        esc(row.get("status")),
        esc(row.get("limitation")) or "-",
    ]


T3_HEADER = COMMON_HEADER + [
    "negative_id",
    "attempted",
    "rejected",
    "unexpectedly_accepted",
    "reproducer",
]


def t3_cells(row: Mapping[str, Any], link_base: str) -> list[str]:
    accepted = row.get("unexpectedly_accepted", 0)
    return common_cells(row, link_base) + [
        esc(row.get("negative_id")),
        num(row.get("attempted")),
        num(row.get("rejected")),
        f"**{num(accepted)}**" if accepted else num(accepted),
        esc(row.get("reproducer")) or "-",
    ]


T4_HEADER = COMMON_HEADER + [
    "partition_id",
    "instrumentation_mode",
    "production_overhead_ratio",
    "replay_commit_overhead_ratio",
    "replay_vs_production_gpu_work",
    "transparent_check_cost",
    "peak_replay_memory",
    "committed_logical_bytes",
    "processed_bytes",
    "description_size_bytes",
    "constructor_inputs_bytes",
    "logical_size",
    "lookup_cost_s",
    "validation_cost_s",
    "moe_padding",
    "restrictions",
    "spend",
]


def t4_cells(row: Mapping[str, Any], link_base: str) -> list[str]:
    return common_cells(row, link_base) + [
        esc(row.get("partition_id")),
        esc(row.get("instrumentation_mode")),
        num(row.get("production_overhead_ratio")),
        num(row.get("replay_commit_overhead_ratio")),
        num(row.get("replay_vs_production_gpu_work")),
        compact(row.get("transparent_check_cost")),
        compact(row.get("peak_replay_memory")),
        num(row.get("committed_logical_bytes")),
        compact(row.get("processed_bytes")),
        num(row.get("description_size_bytes")),
        num(row.get("constructor_inputs_bytes")),
        compact(row.get("logical_size")),
        num(row.get("lookup_cost_s")),
        num(row.get("validation_cost_s")),
        compact(row.get("moe_padding")),
        compact(row.get("restrictions")),
        compact(row.get("spend")),
    ]


HEADERS = {1: T1_HEADER, 2: T2_HEADER, 3: T3_HEADER, 4: T4_HEADER}
CELLS = {1: t1_cells, 2: t2_cells, 3: t3_cells, 4: t4_cells}


def render_markdown(collected: Collected, link_base: str) -> str:
    """The whole ``results.md`` document."""
    lines: list[str] = [
        "# Verity x vLLM replay POC - results",
        "",
        "Generated by `python -m verity_vllm.render` from "
        "`fixtures/results/*/table{1,2,3,4}.jsonl`. Do not edit by hand. Field names follow "
        "`CONTRACT.md` section 9; Table 1 `result` is recomputed with the section 10 rule "
        "(`verity_vllm.rows.verdict`) and any over-claim is flagged below. A `result` is unqualified only "
        "when the row's challenge randomness was verifier-owned (`challenge.mode = security`, "
        "`docs/vllm-poc/binding.md`); rows whose sample came from a public `--seed` (every row before the "
        "staged challenge) read `PASS (diagnostic challenge)`: the checks ran, the security claim is not made. "
        "`bindings` names WB's C3 bindings and whether the committed output boundaries bind the chosen logprobs "
        "(`logprobs: bound` needs the 2026-09-07c capture format; older archives are `unbound`, never upgraded). "
        "`verification_evidence` (GUIDANCE §15.2): *transcript-verified (security-mode staging)* = the prover's own "
        "pod and seed, a standalone transcript proving correct derivation conditional on the verifier's interaction/order; "
        "*independently verified* = an auditor re-verified the row from the archive with its own seed "
        "(`results/<experiment>/audit.json` cites the report); *derived bundle* = `verity_vllm_adapter.derive_logprobs` "
        "output, CPU-only structural checks, no challenge.",
        "",
    ]
    n_rows = {t: len(collected.rows(t)) for t in (1, 2, 3, 4)}
    lines.append(
        f"Experiments: {len(collected.experiments)}. Rows: "
        + ", ".join(f"table {t}: {n}" for t, n in n_rows.items())
        + f". Conformance runs: {len(collected.conformance)}. Issues: {len(collected.issues)}."
    )
    lines.append("")
    if collected.experiments:
        lines.append("## Experiments")
        lines.append("")
        lines.extend(
            markdown_table(
                ["experiment_id", "tables", "artifacts present"],
                [
                    [
                        link(e.experiment_id, link_base),
                        ", ".join(str(t) for t in sorted(e.tables)) or "-",
                        ", ".join(e.side_files) or "-",
                    ]
                    for e in sorted(collected.experiments, key=lambda e: e.experiment_id)
                ],
            )
        )
        lines.append("")
    for table in (1, 2, 3, 4):
        lines.append(f"## {TITLES[table]}")
        lines.append("")
        table_rows = collected.rows(table)
        if not table_rows:
            lines.append("_no rows_")
        else:
            lines.extend(
                markdown_table(HEADERS[table], [CELLS[table](row, link_base) for row in table_rows])
            )
        lines.append("")
    if collected.conformance:
        lines.append("## Configuration conformance (profile ports)")
        lines.append("")
        lines.append(
            "Latest `python -m verity_vllm_adapter.conformance` run per port (`fixtures/results/CONF-<port>-<utc>/level.json`; "
            "`docs/vllm-poc/config-conformance.md`). `level` is the highest level met; a level is `met`, `not-met` "
            "(evidence contradicts it), `not-attempted` (no evidence supplied), `not-claimed` (the port declines it, with "
            "the reason) or `absent`. Never above the evidence: the matrix with pointers is `docs/vllm-poc/conformance-matrix.md`."
        )
        lines.append("")
        body = []
        for rec in conformance_rows(collected.conformance):
            lv = rec["level"]
            cells = [esc(lv.get("port")), f"L{lv.get('level')}" if lv.get("level") else "-"]
            for n in CONFORMANCE_LEVELS:
                r = (lv.get("levels") or {}).get(str(n)) or {}
                cells.append(esc(r.get("status", "-")))
            cells.append(link(rec["experiment_id"], link_base))
            body.append(cells)
        lines.extend(markdown_table(["port", "level", "L1 reproducible", "L2 typed", "L3 bundle", "L4 proof", "run"], body))
        lines.append("")
    lines.append("## Issues")
    lines.append("")
    if collected.issues:
        lines.extend(f"- {esc(issue)}" for issue in collected.issues)
    else:
        lines.append("_none_")
    lines.append("")
    return "\n".join(lines)


def render(
    results_dir: str | Path | None = None,
    out: str | Path | None = None,
    link_base: str | None = None,
) -> tuple[Path, Collected]:
    """Collect, render and write.  Returns the output path and what was collected."""
    results_dir = Path(results_dir) if results_dir is not None else default_results_dir()
    out_path = Path(out) if out is not None else default_out_path()
    if link_base is None:
        # relative from the document's directory to the results directory (../../fixtures/results)
        link_base = Path(
            os.path.relpath(results_dir.resolve(), out_path.resolve().parent)
        ).as_posix()
    collected = collect(results_dir)
    text = render_markdown(collected, link_base)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path, collected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m verity_vllm.render", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument(
        "--results", type=Path, default=None, help="results directory (fixtures/results)"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="output Markdown (docs/fixtures/results.md)"
    )
    parser.add_argument("--link-base", default=None, help="link prefix for experiment directories")
    parser.add_argument("--check", action="store_true", help="exit 1 when any issue was found")
    parser.add_argument("--json", action="store_true", help="also print the issue list as JSON")
    args = parser.parse_args(argv)
    out_path, collected = render(args.results, args.out, args.link_base)
    print(
        f"wrote {out_path} ({len(collected.experiments)} experiments, "
        f"{len(collected.issues)} issues)"
    )
    for issue in collected.issues:
        print(f"issue: {issue}", file=sys.stderr)
    if args.json:
        print(json.dumps({"out": str(out_path), "issues": collected.issues}, indent=1))
    return 1 if (args.check and collected.issues) else 0


if __name__ == "__main__":
    raise SystemExit(main())
