"""The FP8 served table at a scheme's cap: per model, its served pass's per-input measurement (`served_debit.py --per-input`'s
``inputs.json``), the verdicts of its reference verify (`verify_run.py`'s ``verify.json``) where one was run, and the ε that
`HonestTileCapRate`'s served citation takes: the largest of the m inputs' simultaneous 95% upper limits.

  python benchmarks/pouw/pearl_c_vllm/served_table.py --model qwen3-8b INPUTS_JSON [VERIFY_JSON ...] \\
      --model llama-3.1-8b INPUTS_JSON [--fresh qwen3-8b FRESH_INPUTS_JSON] --out DIR

Writes ``<out>/table.json`` and ``<out>/table.md``.  A model's row is refused when its inputs name more than one run, when an
input's openings failed, or when a verify given for it did not accept its pass (a negative control's, ``control*``, must
reject).  ``--fresh`` adds a model's ε from fresh-salt trials (``--fresh-draws``) beside the pass-salt one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def model_row(name: str, inputs: dict, verifies: list[dict], fresh: dict | None) -> dict:
    rows = inputs["inputs"]
    runs = sorted({r["run"] for r in rows})
    if len(runs) != 1:
        raise ValueError(f"{name}: inputs from {runs}; a model's ε is over one served pass's inputs")
    if any(r["openings_failed"] for r in rows):
        raise ValueError(f"{name}: an input's openings failed; the measurement is not of the committed pass")
    out = {"model": name, "run": runs[0], "passes": sorted({r["pass"] for r in rows}),
           "manifests": sorted({f"{r['pass']}:{r['manifest_sha256']}" for r in rows}), "m": inputs["m"],
           "delta_per_input": inputs["delta_per_input"], "tile_draws": sum(r["drawn"] for r in rows),
           "failures": sum(r["failures"] for r in rows), "inputs_with_failures": sum(r["failures"] > 0 for r in rows),
           "share_max": max((r["share_max"] for r in rows if r["share_max"] is not None), default=None),
           "epsilon": inputs["cited_upper"], "epsilon_input": inputs["cited_input"],
           "epsilon_method": max(rows, key=lambda r: r["upper"])["bound_method"],
           "inputs": [{k: r[k] for k in ("pass", "matmul", "k", "n", "NT", "method", "drawn", "failures", "share_max", "upper")}
                      for r in rows]}
    for v in verifies:
        label = Path(v["manifest"]).parent.name
        want = not label.startswith("control")
        if v.get("cross_check_failed") or bool(v["accepted"]) != want:
            raise ValueError(f"{name}: the verify of {label} says {v['line'][:120]}")
        if v["transcript"].split(":", 1)[0] != runs[0]:
            raise ValueError(f"{name}: the verify of {label} is of {v['transcript'].split(':', 1)[0]}, not {runs[0]}")
    out["verify"] = [{"pass": Path(v["manifest"]).parent.name, "accepted": v["accepted"], "replay": v.get("replay", "scalar"),
                      "tiles": len(v["tiles"]), "cross_checked": len(v.get("cross_checked", [])),
                      "transcript_sha256": v["transcript_sha256"], "line": v["line"]} for v in verifies]
    if fresh is not None:
        if {r["run"] for r in fresh["inputs"]} != {runs[0]} or any(r["openings_failed"] for r in fresh["inputs"]):
            raise ValueError(f"{name}: the fresh trials are not of {runs[0]}'s committed pass")
        out["fresh"] = {"draws": fresh["fresh_draws"], "source": fresh["fresh_source"], "m": fresh["m"],
                        "trials": sum(r["drawn"] for r in fresh["inputs"]), "failures": sum(r["failures"] for r in fresh["inputs"]),
                        "epsilon": fresh["cited_upper"], "epsilon_input": fresh["cited_input"]}
    return out


def markdown(table: dict) -> str:
    lines = [f"# FP8 served table at cap {table['cap']}", "",
             "| model | pass | m | tile draws | failures | worst share | ε (95% simultaneous) | bound | fresh-salt ε | verify |",
             "|---|---|---:|---:|---:|---:|---:|---|---:|---|"]
    for r in table["models"]:
        if "absent" in r:
            lines.append(f"| {r['model']} | — | — | — | — | — | none | {r['absent']} | — | — |")
            continue
        fresh = f"{r['fresh']['epsilon']:.4f} ({r['fresh']['trials']} trials, {r['fresh']['failures']} failed)" if "fresh" in r else "—"
        verify = ", ".join(f"{v['pass']} {'ACCEPT' if v['accepted'] else 'REJECT'}" for v in r["verify"]) or "—"
        lines.append(f"| {r['model']} | `{r['run']}` | {r['m']} | {r['tile_draws']} | {r['failures']} | "
                     f"{r['share_max']:.3g} | {r['epsilon']:.4f} | {r['epsilon_method']} | {fresh} | {verify} |")
    lines += ["", "ε is the largest of the m inputs' upper limits on the input's honest tile-rejection rate at the cap, each at "
              "δ = 0.05/m (served_debit.input_bound); an input is one matmul of the pass."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", nargs="+", action="append", required=True, metavar=("NAME", "JSON"),
                   help="a model's name, its inputs.json, then any verify.json of its pass")
    p.add_argument("--fresh", nargs=2, action="append", default=[], metavar=("NAME", "INPUTS_JSON"))
    p.add_argument("--absent", nargs=2, action="append", default=[], metavar=("NAME", "REASON"),
                   help="a model the table is for but cannot cite, and why")
    p.add_argument("--cap", default="1/1000")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    fresh = {name: json.loads(Path(path).read_text()) for name, path in a.fresh}
    models = []
    for spec in a.model:
        if len(spec) < 2:
            p.error("--model NAME INPUTS_JSON [VERIFY_JSON ...]")
        name, inputs, *verifies = spec
        models.append(model_row(name, json.loads(Path(inputs).read_text()), [json.loads(Path(v).read_text()) for v in verifies],
                                fresh.get(name)))
    models += [{"model": name, "absent": reason} for name, reason in a.absent]
    table = {"tool": "pouw_pearl_c_served_table", "cap": a.cap, "models": models}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "table.json").write_text(json.dumps(table, indent=1) + "\n")
    (a.out / "table.md").write_text(markdown(table))
    print(markdown(table), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
