"""Optional: a captured step's GEMMs lowered onto the vLLM frontend's reference Definitions, with each Definition's gate
count and the share of MACs lowered.

    lower(step, phases=None) -> dict          {"definitions": {id: {"calls", "gates", "macs"}}, "gates", "macs",
                                               "lowered_macs", "coverage", "c_coverage", "unlowered": {reason: MACs}, "error"}
    lower_forward(model, example_inputs) -> dict    the frontend's own derivation of the forward, by the same keys

A GEMM lowers as the frontend's reference rules lower it (`rules.reference.LinearRefRule`, `MatmulRefRule`): `linear`,
`addmm` (alpha = beta = 1, a [N] bias), `mm`, `bmm` and `matmul` with operands of one dtype in {f32, bf16} become one call
per (batch, row) of the Definition the reference vocabulary's `linear` kind binds (`LinearRef_v1{K,N,DT,BIAS}`). Every
other GEMM (convolutions, `_scaled_mm`, `mv`, `dot`, other dtypes) stays unlowered, by reason. `coverage` is the lowered
share of the GEMMs' MACs and `c_coverage` its share of the step's |C| (`StepGraph.c_macs`): the inference frontend has no
rule for the optimizer's elementwise work. Gates are the Definitions' gate counts, `Program.gates`'s unit. Neither function
raises: a failure is the result's `error`, and a capture never depends on lowering.
"""
from __future__ import annotations

from collections import Counter

import torch

from training_slowdown.capture.structure import StepGraph

__all__ = ["lower", "lower_forward"]

_DT = {torch.float32: "f32", torch.bfloat16: "bf16"}
_LOWERED = {"linear", "addmm", "mm", "bmm", "matmul"}


def _binding(node, mm) -> tuple[dict, None] | tuple[None, str]:
    """The `linear` kind's statics for one GEMM, or why the reference rules don't lower it."""
    name = node.target.overloadpacket.__name__
    if name not in _LOWERED or mm.part:
        return None, f"no reference rule for {name}{'.' + mm.part if mm.part else ''}"
    tensors = [a for a in node.args if isinstance(a, torch.fx.Node)]
    dts = {a.meta["val"].dtype for a in tensors}
    if len(dts) != 1 or next(iter(dts)) not in _DT:
        return None, f"{name} over {sorted(str(d).removeprefix('torch.') for d in dts)}: reference GEMMs are f32 or bf16"
    bias = None
    if name == "addmm":
        if node.kwargs.get("alpha", 1) != 1 or node.kwargs.get("beta", 1) != 1:
            return None, "addmm with alpha or beta != 1"
        bias = node.args[0]
    elif name == "linear" and len(node.args) > 2:
        bias = node.args[2]
    if bias is not None and tuple(bias.meta["val"].shape) != (mm.dims["n"],):
        return None, f"{name} with a {tuple(bias.meta['val'].shape)} accumulator, not an [N] bias"
    return {"K": mm.dims["k"], "N": mm.dims["n"], "DT": _DT[next(iter(dts))], "BIAS": bias is not None}, None


def lower(step: StepGraph, phases=None) -> dict:
    """The step's GEMMs (those of `phases`, default all) lowered onto the reference Definitions (module docstring)."""
    out = {"definitions": {}, "gates": 0, "macs": 0, "lowered_macs": 0, "coverage": 0.0, "c_coverage": 0.0,
           "unlowered": {}, "error": None}
    try:
        from verity_vllm.program.frontend.rules.vocab import vocabulary_for
        vocab = vocabulary_for("reference", None)
        nodes = {n.name: n for n in step.graph.graph.nodes}
        bound: dict[tuple, object] = {}
        defs, unlowered = out["definitions"], Counter()
        for mm in step.matmuls:
            if phases is not None and mm.phase not in phases:
                continue
            out["macs"] += mm.macs
            statics, why = _binding(nodes[mm.node], mm)
            key = tuple(sorted(statics.items())) if statics else None
            if statics and key not in bound:
                bound[key] = vocab.kind("linear", **statics)
            d = bound.get(key)
            if d is None:
                unlowered[why or f"the vocabulary binds no linear{statics}"] += mm.macs
                continue
            calls = mm.dims["batch"] * mm.dims["m"]
            e = defs.setdefault(d.id, {"calls": 0, "gates": 0, "macs": 0})
            e["calls"] += calls
            e["gates"] += calls * d.gates
            e["macs"] += mm.macs
            out["gates"] += calls * d.gates
            out["lowered_macs"] += mm.macs
        out["unlowered"] = dict(unlowered)
        out["coverage"] = out["lowered_macs"] / out["macs"] if out["macs"] else 0.0
        c = step.c_macs if phases is None else out["macs"]
        out["c_coverage"] = out["lowered_macs"] / c if c else 0.0
    except Exception as e:  # noqa: BLE001  lowering is optional: its failure is data
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def lower_forward(model: torch.nn.Module, example_inputs) -> dict:
    """The frontend's own reference-mode derivation of `model`'s forward (`derive_report`): each Definition the Program
    calls, with its calls and gates, or why the frontend refused (`error`)."""
    out = {"definitions": {}, "gates": 0, "error": None}
    try:
        from verity_vllm.program.frontend.derive import derive_report
        res = derive_report(model, tuple(example_inputs), mode="reference").result
        if not hasattr(res, "root"):
            out["error"] = f"unsupported: {res}"
            return out
        for n in res.root.body.nodes:
            e = out["definitions"].setdefault(n.fn.id, {"calls": 0, "gates": 0})
            e["calls"] += 1
            e["gates"] += n.gates
        out["gates"] = res.gates
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out
