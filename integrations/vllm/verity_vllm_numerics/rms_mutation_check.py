"""Mutation tracking and dataflow-variant sweep for the Triton rms_norm model (CPU only, W11).

    python -m verity_vllm_numerics.rms_mutation_check <stress_fixture> --tables <dir> [--natural <fixture>] [--json out]

* mutations: for every entry of ``mutations.json`` (one-bit flip of an x word / a w word, or an eps change,
  each re-run through the production kernel by ``rms_stress.py``), rebuild the mutated operands, run the model
  and check (a) the model's output on the mutated operands equals the production output on them (bit-exact),
  and (b) the model's output moved away from the base output (the mutation is detected by the model);
* variants: run every dataflow variant of ``rms_triton_model.VARIANTS`` on the natural and stress fixtures and
  count mismatching outputs (the model, variant 0, must be the only one with none).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verity_vllm_numerics import rms_triton_model as rm  # noqa: E402
from verity_vllm_numerics.rms_check import load_words, run as run_check  # noqa: E402


def mutations(fixture: Path, tables: rm.RmsTables, kernel: str = "triton") -> dict:
    idx = json.loads((fixture / "index.json").read_text())
    muts = json.loads((fixture / "mutations.json").read_text())
    calls = {c["seq"]: c for c in idx["calls"]}
    weights = {k: load_words(fixture / v["file"]) for k, v in idx["weights"].items()}
    results = []
    for m in muts["mutations"]:
        base = calls[m["base_seq"]]
        x = load_words(fixture / base["x_file"]).reshape(base["rows"], base["n_cols"]).copy()
        w = weights[base["weight"]].copy()
        eps = base["eps"]
        if m["kind"] == "x_bit_flip":
            x[m["row"], m["col"]] ^= np.uint16(1 << m["bit"])
            assert f"{int(x[m['row'], m['col']]):04x}" == m["x_after"]
        elif m["kind"] == "w_bit_flip":
            w[m["col"]] ^= np.uint16(1 << m["bit"])
            assert f"{int(w[m['col']]):04x}" == m["w_after"]
        elif m["kind"] == "eps_change":
            eps = m["eps"]
        else:
            raise ValueError(m["kind"])
        hw_mut = load_words(fixture / m["out_file"]).reshape(base["rows"], base["n_cols"])
        hw_base = load_words(fixture / base["out_file"]).reshape(base["rows"], base["n_cols"])
        md_mut = rm.rms_norm_rows(x, w, eps, tables, kernel=kernel)
        md_base = rm.rms_norm_rows(load_words(fixture / base["x_file"]).reshape(base["rows"], base["n_cols"]), weights[base["weight"]], base["eps"], tables, kernel=kernel)
        results.append({
            "kind": m["kind"], "bit": m.get("bit"), "eps": m.get("eps"),
            "production_output_words_changed": int((hw_mut != hw_base).sum()),
            "model_output_words_changed": int((md_mut != md_base).sum()),
            "model_vs_production_on_mutated": int((md_mut != hw_mut).sum()),
            "detected_by_model": bool((md_mut != md_base).any()),
        })
    return {
        "n": len(results), "detected_by_model": sum(r["detected_by_model"] for r in results),
        "model_exact_on_mutated_operands": sum(r["model_vs_production_on_mutated"] == 0 for r in results),
        "production_changed": sum(r["production_output_words_changed"] > 0 for r in results),
        "model_changed_equals_production_changed": sum(r["production_output_words_changed"] == r["model_output_words_changed"] for r in results),
        "entries": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stress")
    ap.add_argument("--tables", required=True)
    ap.add_argument("--natural")
    ap.add_argument("--json")
    ap.add_argument("--kernel", default="triton", choices=["triton", "cuda"])
    args = ap.parse_args()
    tables = rm.RmsTables.load(args.tables)
    rep = {"kernel": args.kernel, "mutations": mutations(Path(args.stress), tables, args.kernel), "variants": {}}
    for name, v in (rm.VARIANTS if args.kernel == "triton" else rm.CUDA_VARIANTS).items():
        ent = {}
        if args.natural:
            r = run_check(Path(args.natural), tables, v, nfirst=0, kernel=args.kernel)
            ent["natural"] = {"outputs": r["total"]["outputs"], "mismatch_outputs": r["total"]["mismatch_outputs"], "mismatch_rows": r["total"]["mismatch_rows"]}
        r = run_check(Path(args.stress), tables, v, nfirst=0, families=True, kernel=args.kernel)
        ent["stress"] = {"outputs": r["total"]["outputs"], "mismatch_outputs": r["total"]["mismatch_outputs"], "mismatch_rows": r["total"]["mismatch_rows"],
                         "by_family": {k: g["mismatch_outputs"] for k, g in r["groups"].items()}}
        rep["variants"][name] = ent
    s = json.dumps(rep, indent=1)
    if args.json:
        Path(args.json).write_text(s)
    print(json.dumps({k: v for k, v in rep["mutations"].items() if k != "entries"}, indent=1))
    for name, ent in rep["variants"].items():
        print(name, {k: (v["mismatch_outputs"], v["mismatch_rows"]) for k, v in ent.items()})


if __name__ == "__main__":
    main()
