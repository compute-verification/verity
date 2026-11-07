"""The coverage table: one row per model, grouped by architecture and sorted by size, with four status columns, each
cell one glyph that its sidecar (COVERAGE.json) backs with the run or art ids behind it.

    coverage_table.py render COVERAGE.json [--ladder SUMMARY.json]... --out DIR

- Circuit: the served computation compiles to a Program whose Definitions pass circuit-check and that equals its traced
  Build, with `predict.batch.check_compiled`'s R1-R3 holding.
- Replay: replaying the recorded schedule on compiled vLLM at the pin, serving and replays alike with Inductor's
  reduction tiling pinned (`TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0`), reproduces every output row; the cell gives the share
  of rows exact in the worst replay.
- Replay, unpinned: the same without the tiling pinned, which says whether the model needs the pin.
- Proved: at least one sampled statement of each block type proved and accepted by the Lean verifier.

A cell's status is `measured` (filled), `representable` (hollow: representable, not run) or `unrepresentable` (a dash:
can't represent yet), and every cell names its evidence. `--ladder` takes `ladder.py summary`'s summary.json and fills the
Replay cells of each row whose `ladder` names one of its models, from its pinned and unpinned arms. DIR gets coverage.md
and the merged coverage.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "verity-bench/coverage/v0"
GLYPH = {"measured": "●", "representable": "○", "unrepresentable": "—"}
COLUMNS = ("circuit", "replay", "replay_unpinned", "proved")
HEADS = {"circuit": "Circuit", "replay": "Replay", "replay_unpinned": "Replay, unpinned", "proved": "Proved"}
#: each Replay column and the ladder arm that fills it
ARMS = {"replay": True, "replay_unpinned": False}
#: the architecture groups, in table order: a row's `arch` tags, joined
GROUPS = ("Dense", "MoE", "MoE, MLA", "MoE, DSA")


def check(cov: dict) -> None:
    """Refuse a table with a cell that has no status from GLYPH or names no evidence."""
    bad = []
    for row in cov["rows"]:
        if ", ".join(row["arch"]) not in GROUPS:
            bad.append(f"{row['model']}: architecture {row['arch']} is not one of {GROUPS}")
        for col in COLUMNS:
            c = row["cells"].get(col)
            if c is None or c.get("status") not in GLYPH:
                bad.append(f"{row['model']}.{col}: status must be one of {sorted(GLYPH)}")
            elif not c.get("evidence"):
                bad.append(f"{row['model']}.{col}: no run or art id behind its glyph (give `evidence`)")
    if bad:
        raise SystemExit("coverage: " + "; ".join(bad))


def pct(share: float) -> str:
    """A share of rows as a percentage with enough digits that a share below one never reads as 100%."""
    if share >= 1:
        return "100%"
    for d in range(1, 9):
        s = f"{100 * share:.{d}f}"
        if float(s) < 100:
            return s.rstrip("0").rstrip(".") + "%"
    return f"{100 * share:.9f}%"


def size(b: float) -> str:
    """Billions of parameters, as a model card gives them."""
    return f"{b / 1000:.2g}T" if b >= 1000 else f"{b:.0f}B" if b >= 100 else f"{b:.3g}B"


def ladder_cell(m: dict) -> dict | None:
    """The Replay cell of one `ladder.py summary` model, or None when none of its replays finished."""
    reps = [r for r in m["replays"] if "bit_exact" in r]
    if not reps:
        return None
    ids = sorted({r["run_id"] for r in reps if r.get("run_id")} | ({m["serve_run_id"]} if m.get("serve_run_id") else set()))
    cell: dict[str, Any] = {"status": "measured", "evidence": ids, "rows_exact": min(r["rows_exact_share"] for r in reps),
                            "replays_exact": f"{sum(r['bit_exact'] for r in reps)} of {len(reps)}",
                            "requests": reps[0]["requests"], "rows": reps[0]["tokens_served"],
                            "requests_differing": [r["requests_tokens_differing"] for r in reps],
                            "fresh_compile": [r["compile"] == "compile" for r in reps]}
    if m.get("alone"):
        cell["alone"] = m["alone"]
    if m.get("tilings"):
        cell["tilings"] = m["tilings"]
    return cell


def pin_verdicts(cov: dict) -> tuple[list[str], list[str]]:
    """The models whose unpinned replays kept different tilings or differed from serving (they need the pin), and those
    whose unpinned replays, with their tilings logged, all kept the same tilings and matched serving."""
    need, fine = [], []
    for row in cov["rows"]:
        c = row["cells"].get("replay_unpinned") or {}
        if c.get("status") != "measured":
            continue
        t = c.get("tilings")
        if (t and t["distinct"] > 1) or c.get("rows_exact", 1) < 1:
            need.append(row["model"])
        elif t:
            fine.append(row["model"])
    return need, fine


def merge(cov: dict, summaries: list[dict]) -> dict:
    by = {(m["model"], bool(m.get("pin"))): m for s in summaries for m in s["models"]}
    for row in cov["rows"]:
        for col, pin in ARMS.items():
            m = by.get((row.get("ladder") or "", pin))
            cell = ladder_cell(m) if m else None
            if cell:
                row["cells"][col] = cell
    return cov


def markdown(cov: dict) -> str:
    lines = ["| Model | Architecture | Params, total (active) | Precision | GPUs | " + " | ".join(HEADS[c] for c in COLUMNS)
             + " |", "|---|---|---|---|---|" + ":---:|" * len(COLUMNS)]
    order = {g: i for i, g in enumerate(GROUPS)}
    for row in sorted(cov["rows"], key=lambda r: (order[", ".join(r["arch"])], r["params_b"], r["model"])):
        params = size(row["params_b"]) + (f" ({size(row['active_b'])})" if row.get("active_b") else "")
        cells = []
        for col in COLUMNS:
            c = row["cells"][col]
            g = GLYPH[c["status"]]
            if col in ARMS and c["status"] == "measured" and c.get("rows_exact") is not None:
                g += " " + pct(c["rows_exact"])
            cells.append(g)
        lines.append(f"| {row['model']} | {', '.join(row['arch'])} | {params} | {' / '.join(row['precision'])} | "
                     f"{row['gpus']} | " + " | ".join(cells) + " |")
    need, fine = pin_verdicts(cov)
    verdict = ""
    if need or fine:
        verdict = (f" Unpinned, the replays kept different tilings or differed from serving: {', '.join(need) or 'none'}; "
                   f"they kept the same tilings and matched: {', '.join(fine) or 'none'}.")
    return ("\n".join(lines) + "\n\n" + f"● measured, ○ representable but not run, — can't represent yet. Replay: share of "
            "output rows exact in the worst replay, serving and replaying with Inductor's reduction tiling pinned "
            "(`TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0`); unpinned, a fused add + RMSNorm kernel picks one of two tilings "
            f"per process.{verdict} vLLM {cov['pin']}, {cov['date']}; each cell's run and art ids are in coverage.json.\n")


def render(a: argparse.Namespace) -> int:
    cov = json.loads(Path(a.coverage).read_text())
    cov = merge(cov, [json.loads(Path(s).read_text()) for s in a.ladder])
    check(cov)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "coverage.json").write_text(json.dumps(cov, indent=1, ensure_ascii=False) + "\n")
    md = markdown(cov)
    (out / "coverage.md").write_text(md)
    print(md, end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("coverage")
    r.add_argument("--ladder", action="append", default=[], help="a `ladder.py summary` summary.json (repeatable)")
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"render": render}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
