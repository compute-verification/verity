"""Walk the eager and compiled captures of one request in topological order and report the FIRST
canonical value whose bits differ (CONTRACT §16.6). CPU only (numpy)::

    PYTHONPATH=. python -m verity_vllm_adapter.diagnostics.compare_modes \
        --eager /workspace/diag/eager --compiled /workspace/diag/compiled --out /workspace/diag/compare

Inputs are the directories written by ``capture_modes.py``. The compiled capture is a stream of
kernel launches (``K<i>|<kernel>`` events, one per plain buffer argument, in execution order); the
eager capture is a stream of ``nn.Module`` inputs/outputs. The mapping compiled-buffer -> eager
(module, port) is derived from ``kernel_calls.json`` (the Inductor buffer's ``origins`` = the
post-grad FX nodes fused into it, whose ``nn_module_stack`` is in the graph dump) and checked by
shape at run time; the layer counter advances at every ``fused_add_rms_norm`` kernel that produces
the *next* layer's normalised input.

For every compared tensor the report has: number of differing words, the first differing flat
index -> (position, coordinate), expected (eager) / actual (compiled) bit patterns and float
values, max |ulp| distance. The first entry with ``n_diff > 0`` in execution order is the answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--eager", required=True)
ap.add_argument("--compiled", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-report", type=int, default=400)
args = ap.parse_args()

E = Path(args.eager)
C = Path(args.compiled)
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

eidx = json.load(open(E / "index.json"))
cidx = json.load(open(C / "index.json"))
kcalls = {k["index"]: k for k in json.load(open(C / "kernel_calls.json"))}


def bf16_to_f32(w: np.ndarray) -> np.ndarray:
    return (w.astype(np.uint32) << 16).view(np.float32)


def load(d: Path, ev: dict) -> np.ndarray:
    return np.load(d / "values" / ev["file"])


# ------------------------------------------------------------------------------------------
# eager lookup: (node, port, kind) -> [events in order]
# ------------------------------------------------------------------------------------------
eager: dict[tuple, list[dict]] = {}
for ev in eidx["events"]:
    eager.setdefault((ev["node"], ev["port"], ev["kind"]), []).append(ev)


def eager_get(node: str, port: int, kind: str = "module_out", occurrence: int = 0) -> dict | None:
    lst = eager.get((node, port, kind))
    if not lst or occurrence >= len(lst):
        return None
    return lst[occurrence]


# ------------------------------------------------------------------------------------------
# compiled kernel roles. Derived from kernel_calls.json of this capture (SmolLM2-135M / Llama,
# piecewise compile split at attention): see docs/vllm-poc/numerics.md "divergence diagnosis".
# role(kernel_name, arg, shape) -> (eager node template, port, kind, reshape) or None (skip:
# duplicate input already compared as the producer's output).
# ------------------------------------------------------------------------------------------
KINDS = {
    "embed_rms": re.compile(r"triton_per_fused__to_copy_embedding_rms_norm"),
    "matmul": re.compile(r"matmul_kernel_persistent"),
    "rope": re.compile(r"^triton_poi_fused_\d+$"),
    "fused_add_rms": re.compile(r"triton_per_fused_fused_add_rms_norm"),
    "silu_mul": re.compile(r"triton_poi_fused_mul_silu"),
}


def kind_of(kernel: str) -> str:
    for k, rx in KINDS.items():
        if rx.search(kernel):
            return k
    return "other"


def outputs_of(kc: dict) -> list[str]:
    """Buffer args that this kernel writes: for Inductor kernels those whose origins are not a bare
    graph input / 'empty'; for the user-defined matmul the single 'empty' buffer it mutates."""
    outs = []
    for a, info in kc["tensor_args"].items():
        if a.startswith("arg"):
            continue
        origins = info.get("origins") or []
        if kind_of(kc["kernel"]) == "matmul":
            if all(o.startswith("empty") for o in origins):
                outs.append(a)
        else:
            if not all(o.startswith("empty") for o in origins):
                outs.append(a)
    return outs


report: list[dict] = []
layer = 0
first_div: dict | None = None
n_matmul_in_piece = 0
piece_kind = "first"  # first | middle | last


def compare(cev: dict, node: str, port: int, kind: str, occurrence: int = 0, reshape: bool = True, role: str = "") -> None:
    global first_div
    eev = eager_get(node, port, kind, occurrence)
    entry = {
        "seq_compiled": cev["seq"],
        "kernel_call": cev.get("kernel_call_index"),
        "kernel": cev["node"].split("|", 1)[1],
        "arg": cev.get("arg"),
        "role": role,
        "layer": layer,
        "eager_node": node,
        "eager_port": port,
        "eager_kind": kind,
        "shape_compiled": cev["shape"],
        "dtype": cev["dtype"],
    }
    if eev is None:
        entry["status"] = "no-eager-counterpart"
        report.append(entry)
        return
    a = load(C, cev)
    e = load(E, eev)
    entry["shape_eager"] = eev["shape"]
    entry["seq_eager"] = eev["seq"]
    if a.size != e.size:
        entry["status"] = "size-mismatch"
        report.append(entry)
        return
    af = a.reshape(-1)
    ef = e.reshape(-1)
    diff = np.nonzero(af != ef)[0]
    entry["n_words"] = int(af.size)
    entry["n_diff"] = int(diff.size)
    entry["sha256_eager"] = eev["sha256"]
    entry["sha256_compiled"] = cev["sha256"]
    if diff.size:
        i = int(diff[0])
        row_len = int(np.prod(eev["shape"][1:])) if len(eev["shape"]) > 1 else 1
        pos, coord = divmod(i, row_len)
        entry["first_diff"] = {"flat": i, "position": int(pos), "coordinate": int(coord), "eager_shape": eev["shape"]}
        if cev["dtype"] == "bfloat16":
            ev_, av_ = bf16_to_f32(ef[diff]), bf16_to_f32(af[diff])
            entry["first_diff"].update(
                {
                    "expected_bits": f"0x{int(ef[i]):04x}",
                    "actual_bits": f"0x{int(af[i]):04x}",
                    "expected_value": float(ev_[0]),
                    "actual_value": float(av_[0]),
                    "ulp_distance": int(abs(int(ef[i]) - int(af[i]))),
                }
            )
            ulps = np.abs(af[diff].astype(np.int64) - ef[diff].astype(np.int64))
            entry["ulp_hist"] = {str(k): int(np.sum(ulps == k)) for k in (1, 2, 3, 4) if np.any(ulps == k)}
            if np.any(ulps > 4):
                entry["ulp_hist"][">4"] = int(np.sum(ulps > 4))
            entry["ulp_max"] = int(ulps.max())
            entry["max_abs_diff"] = float(np.max(np.abs(ev_.astype(np.float64) - av_.astype(np.float64))))
            entry["positions_with_diff"] = int(len(np.unique(diff // row_len)))
        else:
            entry["first_diff"].update({"expected": int(ef[i]), "actual": int(af[i])})
        entry["status"] = "mismatch"
        if first_div is None:
            first_div = entry
    else:
        entry["status"] = "bit-equal"
    report.append(entry)


rope_nodes = "model.layers.0.self_attn.rotary_emb"  # shared module in eager; occurrence = layer


def reprocess() -> None:
    """Walk the compiled stream with explicit piece detection (first kernel after a rope kernel starts a piece)."""
    global report, layer, first_div, n_matmul_in_piece, piece_kind
    report = []
    layer = 0
    first_div = None
    n_matmul_in_piece = 0
    piece_kind = "first"
    last_kernel_kind = None
    last_kidx = None
    for cev in cidx["events"]:
        if cev["kind"] != "inductor":
            continue
        kidx = cev["kernel_call_index"]
        kc = kcalls[kidx]
        kname = kc["kernel"]
        kk = kind_of(kname)
        if kidx != last_kidx:
            if last_kernel_kind == "rope" and kk == "matmul":
                # new piece begins (o_proj of layer `layer`)
                piece_kind = "middle"
                n_matmul_in_piece = 0
            # the last piece has no rope and ends with a single-output fused_add_rms_norm
            last_kernel_kind = kk
            last_kidx = kidx
        arg = cev["arg"]
        outs = outputs_of(kc)
        shape = cev["shape"]
        if kk == "embed_rms":
            if arg == "arg0_1":
                continue
            if arg in outs:
                if len(outs) == 2 and arg == outs[0]:
                    compare(cev, "model.embed_tokens", 0, "module_out", role="embedding")
                else:
                    compare(cev, f"model.layers.{layer}.input_layernorm", 0, "module_out", role="rms_norm(normed)")
            continue
        if kk == "matmul":
            if arg not in outs:
                continue
            if piece_kind == "first":
                compare(cev, f"model.layers.{layer}.self_attn.qkv_proj", 0, "module_out", role="qkv_proj")
                continue
            n_matmul_in_piece += 1
            if n_matmul_in_piece == 1:
                compare(cev, f"model.layers.{layer}.self_attn.o_proj", 0, "module_out", role="o_proj")
            elif n_matmul_in_piece == 2:
                compare(cev, f"model.layers.{layer}.mlp.gate_up_proj", 0, "module_out", role="gate_up_proj")
            elif n_matmul_in_piece == 3:
                compare(cev, f"model.layers.{layer}.mlp.down_proj", 0, "module_out", role="down_proj")
            elif n_matmul_in_piece == 4:
                compare(cev, f"model.layers.{layer}.self_attn.qkv_proj", 0, "module_out", role="qkv_proj")
            continue
        if kk == "rope":
            if arg.startswith("arg"):
                continue
            if arg in outs:
                port = 0 if shape[1] * shape[2] == 576 else 1
                compare(cev, rope_nodes, port, "module_out", occurrence=layer, role="rotary(q)" if port == 0 else "rotary(k)")
            continue
        if kk == "silu_mul":
            if arg in outs:
                compare(cev, f"model.layers.{layer}.mlp.act_fn", 0, "module_out", role="silu_and_mul")
            continue
        if kk == "fused_add_rms":
            if arg.startswith("arg") and len(shape) == 2:
                compare(cev, f"model.layers.{layer}.post_attention_layernorm", 1, "module_in", role="residual_in(piece input)")
                continue
            if arg not in outs:
                continue
            if len(outs) == 1:
                if n_matmul_in_piece >= 3:
                    compare(cev, "model.norm", 0, "module_out", role="final_norm(normed)")
                else:
                    compare(cev, f"model.layers.{layer}.post_attention_layernorm", 0, "module_out", role="post_attention_layernorm(normed)")
                continue
            if arg == outs[0]:
                compare(cev, f"model.layers.{layer + 1}.input_layernorm", 1, "module_out", role="input_layernorm(residual)")
            else:
                compare(cev, f"model.layers.{layer + 1}.input_layernorm", 0, "module_out", role="input_layernorm(normed)")
                layer += 1
            continue
        report.append({"seq_compiled": cev["seq"], "kernel": kname, "arg": arg, "status": "unmapped", "layer": layer})


reprocess()

# ------------------------------------------------------------------------------------------
# kernel source of the first divergent kernel (from the copied wrapper file)
# ------------------------------------------------------------------------------------------
def kernel_source(kernel: str, wrapper_file: Path) -> dict | None:
    txt = wrapper_file.read_text()
    m = re.search(rf"{re.escape(kernel)} = async_compile\.triton\('{re.escape(kernel)}', '''(.*?)''', device_str=", txt, re.S)
    if not m:
        return None
    src = m.group(1)
    return {"kernel": kernel, "source": src, "sha256": hashlib.sha256(src.encode()).hexdigest(), "lines": src.count("\n")}


summary = {
    "request_id": cidx["result"]["request_id"],
    "prompt_len": cidx["result"]["prompt_len"],
    "eager": {"token": eidx["result"]["output_token_ids"], "logprob": eidx["result"]["chosen_logprobs"]},
    "compiled": {"token": cidx["result"]["output_token_ids"], "logprob": cidx["result"]["chosen_logprobs"]},
    "compared": sum(1 for r in report if r.get("status") in ("bit-equal", "mismatch")),
    "bit_equal": sum(1 for r in report if r.get("status") == "bit-equal"),
    "mismatch": sum(1 for r in report if r.get("status") == "mismatch"),
    "unmapped": sum(1 for r in report if r.get("status") not in ("bit-equal", "mismatch")),
    "first_divergence": first_div,
}
if first_div is not None:
    cev = next(e for e in cidx["events"] if e.get("seq") == first_div["seq_compiled"])
    wf = C / "kernels" / Path(cev["wrapper_file"]).name
    ks = kernel_source(first_div["kernel"], wf) if wf.exists() else None
    if ks:
        (OUT / f"{first_div['kernel']}.py").write_text(ks["source"])
        summary["first_divergence_kernel"] = {k: v for k, v in ks.items() if k != "source"}
        summary["first_divergence_kernel"]["wrapper_file"] = cev["wrapper_file"]
        summary["first_divergence_kernel"]["wrapper_sha256"] = hashlib.sha256(wf.read_bytes()).hexdigest()
    # the compiled kernel call that produced it
    summary["first_divergence_kernel_call"] = kcalls[first_div["kernel_call"]]

(OUT / "report.json").write_text(json.dumps({"summary": summary, "rows": report[: args.max_report]}, indent=1) + "\n")
lines = [
    f"# eager vs compiled: request {summary['request_id']} (prompt {summary['prompt_len']} tokens)",
    "",
    f"eager token {summary['eager']['token']} logprob {summary['eager']['logprob']}; compiled token {summary['compiled']['token']} logprob {summary['compiled']['logprob']}",
    f"compared {summary['compared']} tensors: {summary['bit_equal']} bit-equal, {summary['mismatch']} mismatching, {summary['unmapped']} unmapped/skipped",
    "",
    "| # | layer | role | kernel | eager node:port | words | n_diff | positions | first diff (pos,coord) | expected | actual | ulp hist |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|",
]
for r in report:
    if r.get("status") not in ("bit-equal", "mismatch"):
        continue
    fd = r.get("first_diff") or {}
    lines.append(
        f"| {r['seq_compiled']} | {r['layer']} | {r['role']} | {r['kernel']} | {r['eager_node']}:{r['eager_port']} | {r.get('n_words')} | {r.get('n_diff')} | {r.get('positions_with_diff', '')} | "
        f"{(fd.get('position'), fd.get('coordinate')) if fd else ''} | {fd.get('expected_bits', '')} {fd.get('expected_value', '')} | {fd.get('actual_bits', '')} {fd.get('actual_value', '')} | {json.dumps(r.get('ulp_hist')) if r.get('ulp_hist') else ''} |"
    )
(OUT / "report.md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, indent=1))
