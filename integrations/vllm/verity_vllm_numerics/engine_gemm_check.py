"""Run the independent GEMM model on the operands captured *inside the engine*
(``engine_gemm_capture.py``) -- Table 2 row class ``the same GEMM inside the replayed model``.

Two passes over every captured call ``out = matmul_kernel_persistent(x, weight.t())``:

1. **full GEMM** -- ``cpu_model.gemm_model`` (C++/OpenMP) on the whole ``[M, K] x [N, K]`` call,
   bitwise against the captured ``[M, N]`` output: exact coordinates / attempted, first mismatch
   with bit patterns, CPU seconds per full GEMM and per output row;
2. **row VUs through W2's registry** -- for a sample of rows per call, build W2's ``Node`` /
   ``NodeInstance`` / ``Value`` objects exactly as the runner would (``verity.bi_mm`` node with
   ``x`` = the opened row and ``w`` = the parameter leaf ``[N, K]``) and call
   ``verity_vllm.relations.REGISTRY["gemm.bf16.v1"].checker(inst, inputs)`` (``lm_head.bf16.v1`` for
   the lm_head), comparing the returned bytes with the captured row's bytes.  This is the
   interface the verifier uses; the numerics are the same as pass 1 and it proves the checker
   accepts W1's description conventions (``extra.runtime_op``, weight layout, ``arg_cols``).
   Without W2's package importable this pass runs the same functions on plain mappings.

    OMP_NUM_THREADS=64 PYTHONPATH=/path/to/vllm-poc:/path/to/w2/vllm-poc python -m verity_vllm_numerics.engine_gemm_check CAPDIR OUT.json [--rows-per-call 4]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from . import cpu_model as CM
from . import relations as REL
from .mma import AMPERE_BF16

try:
    from verity_vllm import relations as W2R
    from verity_vllm.capture import Value
    from verity_vllm.description import Arg, Node, NodeInstance

    HAVE_W2 = True
except ImportError:  # pragma: no cover - the pod may not have W2's package
    HAVE_W2 = False


def proj_of(weight_name: str | None) -> str:
    if weight_name is None:
        return "unknown"
    if "embed_tokens" in weight_name or "lm_head" in weight_name:
        return "lm_head"
    m = re.search(r"\.(qkv_proj|o_proj|gate_up_proj|down_proj)\.weight$", weight_name)
    return m.group(1) if m else weight_name


def layer_of(weight_name: str | None) -> str:
    m = re.search(r"layers\.(\d+)\.", weight_name or "")
    return f"layers.{m.group(1)}" if m else ("lm_head" if proj_of(weight_name) == "lm_head" else "?")


def bits(u16: int) -> str:
    return f"0x{u16:04x}"


def first_mismatch(gpu: np.ndarray, model: np.ndarray, acc_f32: np.ndarray | None, x: np.ndarray, w: np.ndarray) -> dict | None:
    bad = np.argwhere(gpu != model)
    if bad.size == 0:
        return None
    i, j = (int(v) for v in bad[0])
    d = {"i": i, "j": j, "gpu": bits(int(gpu[i, j])), "model": bits(int(model[i, j])), "n_mismatch": int(bad.shape[0])}
    if acc_f32 is not None:
        d["model_acc_f32"] = f"0x{int(acc_f32[i, j]):08x}"
    d["x_row_nonfinite"] = bool(np.any((x[i] & 0x7F80) == 0x7F80))
    d["w_row_nonfinite"] = bool(np.any((w[j] & 0x7F80) == 0x7F80))
    return d


def w2_node(call: dict, proj: str) -> "Node":
    N, K = call["N"], call["K"]
    layer = layer_of(call["weight"])
    op = f"{layer}.{proj}.gemm" if proj != "lm_head" else "lm_head.gemm"
    producer = {"qkv_proj": "input_layernorm.rmsnorm", "o_proj": "self_attn.attn", "gate_up_proj": "post_attention_layernorm.rmsnorm", "down_proj": "mlp.act_fn.silu_mul", "lm_head": "norm.rmsnorm"}[proj]
    return Node(
        op=op, layer=layer, op_type="lm_head" if proj == "lm_head" else "gemm", dtype="bfloat16", shape=(N,),
        relation="lm_head.bf16.v1" if proj == "lm_head" else "gemm.bf16.v1",
        args=(Arg(name="x", source="node", op=f"{layer}.{producer}", port=0), Arg(name="w", source="weight", param=call["weight"])),
        runtime_impl="triton:vllm.model_executor.determinism.batch_invariant.matmul_persistent (via linear_batch_invariant)",
        extra={"runtime_op": "verity.bi_mm", "consts": {}, "arg_cols": {}, "arg_shapes": {"x": [1, K], "w": [K, N]}},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capdir")
    ap.add_argument("out")
    ap.add_argument("--rows-per-call", type=int, default=4)
    ap.add_argument("--max-calls", type=int, default=0)
    args = ap.parse_args()
    cap = Path(args.capdir)
    idx = json.loads((cap / "index.json").read_text())
    calls = idx["calls"] if not args.max_calls else idx["calls"][: args.max_calls]
    weights: dict[str, np.ndarray] = {}
    spec = CM.GemmModelSpec(AMPERE_BF16.name, bias_mode=0, note="single ascending chain, no epilogue (verity.bi_mm)")

    if HAVE_W2:
        registered = REL.REGISTERED_WITH_W2 or REL.register_with_w2()
        checker = {rid: W2R.REGISTRY[rid].checker for rid in ("gemm.bf16.v1", "lm_head.bf16.v1")}
    else:
        registered = []
        checker = {"gemm.bf16.v1": REL.independent_gemm, "lm_head.bf16.v1": REL.independent_gemm}

    per_call = []
    agg: dict[str, dict] = {}
    rng = np.random.default_rng(0)
    t_all = time.perf_counter()
    for c in calls:
        proj = proj_of(c["weight"])
        if c["weight"] is None:
            per_call.append({**{k: c[k] for k in ("seq", "M", "K", "N")}, "proj": proj, "status": "skipped: b is not a parameter view"})
            continue
        if c["weight"] not in weights:
            weights[c["weight"]] = np.load(cap / idx["weights"][c["weight"]]["file"]).reshape(c["N"], c["K"])
        w = weights[c["weight"]]
        x = np.load(cap / c["x_file"]).reshape(c["M"], c["K"])
        gpu = np.load(cap / c["out_file"]).reshape(c["M"], c["N"])
        assert not c["bias"], "verity.bi_mm never carries a bias"
        t0 = time.perf_counter()
        model, acc = CM.gemm_model(AMPERE_BF16, x, w, None, spec, want_f32=True)
        dt = time.perf_counter() - t0
        model = model.reshape(c["M"], c["N"])
        exact = int(np.count_nonzero(gpu == model))
        rec = {
            "seq": c["seq"], "proj": proj, "layer": layer_of(c["weight"]), "weight": c["weight"], "M": c["M"], "K": c["K"], "N": c["N"],
            "phase": "prefill" if c["M"] > len(idx["prompts"]) else "decode",
            "exact": exact, "attempted": int(gpu.size), "rows_exact": int(np.count_nonzero(np.all(gpu == model, axis=1))),
            "cpu_seconds_full_gemm": dt, "cpu_seconds_per_row": dt / c["M"],
            "nonfinite_gpu": int(np.count_nonzero((gpu & 0x7F80) == 0x7F80)),
            "first_mismatch": first_mismatch(gpu, model, acc.reshape(c["M"], c["N"]) if acc is not None else None, x, w),
        }
        # pass 2: row VUs through the registry interface
        n_rows = min(args.rows_per_call, c["M"])
        rows = sorted(rng.choice(c["M"], size=n_rows, replace=False).tolist()) if n_rows < c["M"] else list(range(c["M"]))
        rid = "lm_head.bf16.v1" if proj == "lm_head" else "gemm.bf16.v1"
        vu_exact = 0
        vu_err = None
        t1 = time.perf_counter()
        for r in rows:
            xb = x[r].astype("<u2").tobytes()
            wb = w.astype("<u2").tobytes() if HAVE_W2 else w
            if HAVE_W2:
                node = w2_node(c, proj)
                inst = NodeInstance(node=node, request_id="req-0", t=int(r), phase=rec["phase"])
                inputs = {
                    "x": Value(canonical_id=f"a100-vllm-bi1/req-0/t={r}/{rec['phase']}/{node.layer}/{node.args[0].op}/port=0", dtype="bfloat16", shape=(c["K"],), data=xb),
                    "w": Value(canonical_id=f"a100-vllm-bi1/weights/{c['weight']}", dtype="bfloat16", shape=(c["N"], c["K"]), data=wb, origin="weight"),
                }
            else:
                inst = {"op": f"{proj}.gemm", "runtime_op": "verity.bi_mm", "args": {"x": "x", "w": "w"}}
                inputs = {"x": xb, "w": wb}
            try:
                pred = checker[rid](inst, inputs)
            except Exception as e:  # noqa: BLE001
                vu_err = repr(e)
                break
            if pred == gpu[r].astype("<u2").tobytes():
                vu_exact += 1
        rec["registry_row_vus"] = {"relation": rid, "attempted": len(rows), "exact": vu_exact, "seconds": time.perf_counter() - t1, "error": vu_err, "via": "verity_vllm.relations.REGISTRY" if HAVE_W2 else "plain mappings (W2 package absent)"}
        per_call.append(rec)
        a = agg.setdefault(f"{proj}|{rec['phase']}", {"proj": proj, "phase": rec["phase"], "K": c["K"], "N": c["N"], "M_values": set(), "calls": 0, "exact": 0, "attempted": 0, "rows_exact": 0, "rows": 0, "cpu_seconds": 0.0, "vu_exact": 0, "vu_attempted": 0, "first_mismatch": None})
        a["M_values"].add(c["M"]); a["calls"] += 1; a["exact"] += exact; a["attempted"] += int(gpu.size); a["rows_exact"] += rec["rows_exact"]; a["rows"] += c["M"]
        a["cpu_seconds"] += dt; a["vu_exact"] += vu_exact; a["vu_attempted"] += len(rows)
        if a["first_mismatch"] is None and rec["first_mismatch"] is not None:
            a["first_mismatch"] = {"seq": c["seq"], "layer": rec["layer"], **rec["first_mismatch"]}
        print(f"[{c['seq']:4d}] {rec['layer']:>10s} {proj:>13s} M={c['M']:4d} K={c['K']:5d} N={c['N']:6d} exact {exact}/{gpu.size} rows {rec['rows_exact']}/{c['M']} {dt:.3f}s  VUs {vu_exact}/{len(rows)}", flush=True)

    for a in agg.values():
        a["M_values"] = sorted(a["M_values"])
        a["cpu_seconds_per_row"] = a["cpu_seconds"] / max(a["rows"], 1)
    res = {
        "capture": str(cap), "capture_meta": {k: idx[k] for k in ("model", "vllm", "torch", "triton", "gpu", "env", "compilation", "interception", "prompt_lens", "max_tokens", "n_calls")},
        "w2_registry": {"available": HAVE_W2, "registered_by_w4": registered},
        "model": {"per_instruction": AMPERE_BF16.name, "spec": spec.describe()},
        "total_seconds": time.perf_counter() - t_all,
        "summary": sorted(agg.values(), key=lambda a: (a["phase"], a["proj"])),
        "calls": per_call,
    }
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    tot_e = sum(a["exact"] for a in agg.values()); tot_a = sum(a["attempted"] for a in agg.values())
    print(json.dumps({"calls": len(per_call), "exact": tot_e, "attempted": tot_a, "vu_exact": sum(a["vu_exact"] for a in agg.values()), "vu_attempted": sum(a["vu_attempted"] for a in agg.values()), "seconds": res["total_seconds"]}))


if __name__ == "__main__":
    main()
