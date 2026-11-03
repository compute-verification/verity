"""The one bound's test vectors (te_core.formalism's `bound`, and `at` under reading A), which the console's rho.ts
runs. null is unmeasured (a term, the max, or ρ where no kind is measured); JSON has no infinity, so ∞ is "inf".

    python -m te_core.rho_vectors [--write]

writes `rho_vectors.json` beside this file (prints it without --write); te_core's tests check the file is what this
module makes. Each case is a step of wildcard's program traced on the tiny start (`CASES`, the circuits under
te_core/capture/tests/fixtures), measured by te_core.capture.circuit.measure as one part, with Δ and d stated per case,
and its ops the same trace's (te_core.tiles.of_graph, the `.ops.json` beside the circuit). Each vector is a case at a
cell (c, k) of `CELLS`: each kind's term and the inputs it reads, the max over the kinds and the kind it binds on, the
sensitivities, Fit, ρ_fit, ρ and the label. A case's `measured` is te_core.formalism.from_json's form, reading A's
`entries` and `ops` in it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

from . import VERSION, formalism
from .capture.circuit import StepCircuit, measure
from .gates import gates, per_mac

SCHEMA = "te-core/rho-vectors/v2"
OUT = Path(__file__).with_name("rho_vectors.json")
FIXTURES = Path(__file__).parent / "capture" / "tests" / "fixtures"
#: model.py's tiny start: its forward MACs per token at ctx 1024 (te_core.flops.forward_macs_per_token), vocab and width
TINY = {"forward_macs_per_token": 3_327_488, "vocab": 50304, "d_model": 64}
W_FWD = TINY["forward_macs_per_token"] * per_mac("gemm/bf16")
PROGRAM = ("islands/wildcard/train.py at 9460fcd228 on the tiny start, --batch 2 --micro 2 --warmup 1, traced at step 1 "
           "(te_core.capture.script)")
#: the LoRA case's decoder: W + B A for its 14 matrices, r out in fp32 multiply-adds each
LORA_REBUILD_MACS = 4 * 2 * (4 * 64 * 64 + 3 * 64 * 192)
CASES = {
    "adapter": {"circuit": "wildcard_adapter.npz",
                "flags": r"--grow 1 --adapter 8 --adapter-z 2 "
                         r"--set 'blocks\.2\.(mlp_norm|w_gate\.weight|w_up\.weight|w_down\.weight)'",
                "delta_bits": None, "decoder_gates": 0.0,
                "delta": "the trace's update cut, theta' as the step reads it (the grown block's four tensors)",
                "decoder": "none: E stores the grown block's tensors (the model format)"},
    "adapter-live": {"circuit": "wildcard_adapter.npz",
                     "flags": r"--grow 1 --adapter 8 --adapter-z 2 "
                              r"--set 'blocks\.2\.(mlp_norm|w_gate\.weight|w_up\.weight|w_down\.weight)'",
                     "delta_bits": 3 * 8 * 64 * 16 + 64 * 32, "decoder_gates": 0.0,
                     "delta": "stated: an E of the live entries alone (w_gate's and w_up's 8 live rows and w_down's 8 "
                              "live columns at bf16, mlp_norm at fp32; wildcard's 05:46Z LOOPHOLE)",
                     "decoder": "none: the dead entries are zero"},
    "dense": {"circuit": "dense_blocks.npz", "flags": r"--set 'blocks\..*'", "delta_bits": None, "decoder_gates": 0.0,
              "delta": "the trace's update cut, theta' as the step reads it (every block tensor)",
              "decoder": "none: E stores the tensors (the full format)"},
    "lora": {"circuit": "lora_blocks.npz", "flags": r"--set 'blocks\..*' --lora-rank 4", "delta_bits": None,
             "decoder_gates": LORA_REBUILD_MACS * gates(formalism.DECODER_DEFINITION),
             "delta": "the trace's update cut, the factors and gains as the step reads them",
             "decoder": f"W + B A for the 14 matrices, {LORA_REBUILD_MACS} fp32 multiply-adds at "
                        f"{formalism.DECODER_DEFINITION}'s gates each"},
}
#: (tokens per block, k bits): c = tokens x the tiny start's forward (te_core.panels' dial), its default first
CELLS = ((4096, 80_000), (4096, 800_000), (4096, 8_000_000), (100, 8_000_000), (1, 80_000_000))
OPS = ("counted (trace): the traced step's GEMMs with their dims and operand classes, and the rest of each phase's "
       "gates (te_core.tiles.of_graph)")


def case_measured(case: dict) -> formalism.Measured:
    """A case's quantities: its traced step's (`measure`) as one part, Δ and d as the case states, and the step's ops
    (the `.ops.json` beside its circuit)."""
    circuit = FIXTURES / case["circuit"]
    f = measure(StepCircuit.load(circuit))
    part = {"gates": f["gates"], "tokens": f["tokens"], "token_gates": f["sigma"] * f["gates"],
            "affected_gates": f["affected_share"] * f["gates"], "steps": 1, "delta_grad_bits": f["delta_grad_bits"],
            "partial_bits": f["partial_cut_bits"], "grad_bits": f["grad_cut_bits"]}
    delta = case["delta_bits"] if case["delta_bits"] is not None else float(f["update_cut_bits"])
    ops = json.loads(circuit.with_suffix(".ops.json").read_text())
    block = formalism.of_trace(f) | {"parts": [part], "delta_bits": delta, "decoder_gates": case["decoder_gates"],
                                     "ops": ops}
    block["labels"] |= {"delta_bits": case["delta"], "decoder_gates": case["decoder"], "ops": OPS}
    return formalism.complete(block, {"output": {"e_bytes": delta / 8}}, TINY)


def finite(x):
    """`x` with every infinity "inf" (JSON has none)."""
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite(v) for v in x]
    return "inf" if isinstance(x, float) and math.isinf(x) else x


def build() -> dict:
    measured, vectors = {}, []
    for name, case in CASES.items():
        m = case_measured(case)
        measured[name] = asdict(m)
        for n, k in CELLS:
            c = n * W_FWD
            b, a = formalism.bound(m, c, k), formalism.at(m, c, k, "A")
            vectors.append({"case": name, "tokens_per_block": n, "c": c, "k": k, "terms": b["terms"],
                            "inputs": b["inputs"], "max": b["max"], "binding": b["binding"],
                            "unmeasured": b["unmeasured"], "sensitivities": b["sensitivities"], "fit": a["fit"],
                            "rho_fit": a["rho_fit"], "rho": a["rho"], "label": b["label"]})
    return finite({
        "schema": SCHEMA, "version": VERSION, "about": __doc__.split("\n\n")[0],
        "rule": "rho = rho_fit where Fit holds, else max(1, the max over the kinds' measured terms); null where no "
                "kind is measured. Sensitivities are never in the max.",
        "kinds": list(formalism.KINDS), "rests_on": formalism.RESTS_ON, "sensitivities": formalism.SENSITIVITIES,
        "w_fwd_gates": W_FWD, "start": TINY, "program": PROGRAM,
        "cases": {name: {k: v for k, v in case.items() if k not in ("delta_bits", "decoder_gates")}
                  for name, case in CASES.items()},
        "measured": measured, "vectors": vectors, "tolerance": {"relative": 1e-9}})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--write", action="store_true", help=f"write {OUT.name} beside this module")
    a = ap.parse_args(argv)
    text = json.dumps(build(), indent=1, sort_keys=True) + "\n"
    if a.write:
        OUT.write_text(text)
        print(f"{OUT}: {len(text):,} bytes", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
