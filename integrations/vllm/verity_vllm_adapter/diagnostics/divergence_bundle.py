"""Build the standalone reproducer bundle for the first compiled/eager divergence (CONTRACT §16.6,
§16.8) from the two captures, and (on a GPU host) run it.

    PYTHONPATH=. python -m verity_vllm_adapter.diagnostics.divergence_bundle build \
        --eager /workspace/diag/eager --compiled /workspace/diag/compiled \
        --compare /workspace/diag/compare --out /workspace/diag/bundle
    cd /workspace/diag/bundle && python repro.py            # GPU: runs both kernels; CPU: model only

Bundle contents (all raw words; BF16 as uint16):
  input_ids.npy        int32 [N]   token ids of the request REMAPPED into the compact table
  embed_rows.npy       uint16 [U, H] the U distinct embedding rows the request touches (gather is
                                     exact, so a compact table reproduces the kernel bit for bit
                                     and keeps the bundle far under 50 MB)
  norm_weight.npy      uint16 [H]  model.layers.0.input_layernorm.weight
  eager_out.npy        uint16 [N, H] captured eager output (vLLM Triton _rms_norm_kernel)
  compiled_out.npy     uint16 [N, H] captured compiled output (Inductor kernel below)
  inductor_kernel.py   the generated Triton source exactly as in the compile cache
  repro.py             the standalone driver (see its docstring)
  README.md, sha256sums.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPRO = r'''"""Standalone reproducer: first compiled/eager divergence on SmolLM2-135M, request 784c37e1.

    python repro.py            # GPU host with the pinned torch + vLLM: runs both kernels and the model
    python repro.py --cpu      # numpy only: the model + the captured outputs

Steps
  1. gather the compact embedding table with the (remapped) token ids -> x  (bf16 [N, 576])
  2. compiled op : the Inductor-generated persistent-reduction kernel `inductor_kernel.py`
                   (embedding gather + rms_norm fused; launched through the same
                   torch._inductor.async_compile path the vLLM wrapper uses)
  3. eager op    : vLLM's batch-invariant Triton `_rms_norm_kernel` (rms_norm_batch_invariant)
  4. bit-compare the two outputs and print the first differing word (position, coordinate, bits)
  5. CPU model   : both kernels are out = bf16((x * s) * w) with FP32 arithmetic and a single
                   rounding; they differ only in the per-row FP32 scalar s = rsqrt(mean(x^2)+eps):
                   Inductor: tl.sum over a [XBLOCK, 1024] block then libdevice.rsqrt;
                   vLLM Triton: tl.sum over a 1024 block then 1.0 / tl.sqrt.
                   For every row we report the ulp offset k (w.r.t. the correctly rounded IEEE
                   s0) that reproduces each kernel's row bit-exactly.
"""
import argparse, hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-5
KERNEL_NAME = "triton_per_fused__to_copy_embedding_rms_norm_t_0"


def bf16_to_f32(w):
    return (np.asarray(w, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)


def first_diff(a, b, H):
    a = a.reshape(-1); b = b.reshape(-1)
    d = np.nonzero(a != b)[0]
    if d.size == 0:
        return None
    i = int(d[0]); pos, coord = divmod(i, H)
    return {"n_diff": int(d.size), "position": pos, "coordinate": coord,
            "eager_bits": f"0x{int(a[i]):04x}", "compiled_bits": f"0x{int(b[i]):04x}",
            "eager_value": float(bf16_to_f32(a[i:i+1])[0]), "compiled_value": float(bf16_to_f32(b[i:i+1])[0])}


def rows_model(x_bf16, w_bf16, s):
    x = bf16_to_f32(x_bf16); w = bf16_to_f32(w_bf16)
    y = (x * s[:, None]).astype(np.float32)
    y = (y * w[None, :]).astype(np.float32)
    return f32_to_bf16(y)


def ieee_s0(x_bf16, eps):
    x = bf16_to_f32(x_bf16).astype(np.float64)
    mean = (x * x).sum(axis=1) / x.shape[1]
    return (1.0 / np.sqrt(mean + eps)).astype(np.float32)


def fit_k(gpu, x_bf16, w_bf16, s0, kmax=6):
    best = np.full(gpu.shape[0], 99, np.int32)
    for k in sorted(range(-kmax, kmax + 1), key=abs):
        s = (s0.view(np.int32) + np.int32(k)).view(np.float32)
        ok = np.all(rows_model(x_bf16, w_bf16, s) == gpu, axis=1) & (best == 99)
        best[ok] = k
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cpu", action="store_true"); a = ap.parse_args()
    ids = np.load(os.path.join(HERE, "input_ids.npy"))
    table = np.load(os.path.join(HERE, "embed_rows.npy"))
    w = np.load(os.path.join(HERE, "norm_weight.npy"))
    eager_cap = np.load(os.path.join(HERE, "eager_out.npy"))
    comp_cap = np.load(os.path.join(HERE, "compiled_out.npy"))
    N, H = eager_cap.shape
    x = table[ids]  # exact gather
    print(f"request rows N={N} hidden H={H} distinct embedding rows U={table.shape[0]}")
    print("captured eager vs compiled:", json.dumps(first_diff(eager_cap, comp_cap, H)))
    out = {"captured": first_diff(eager_cap, comp_cap, H)}
    if not a.cpu:
        import torch
        src = open(os.path.join(HERE, "inductor_kernel.py")).read()
        print("inductor kernel sha256", hashlib.sha256(src.encode()).hexdigest())
        from torch._inductor.async_compile import AsyncCompile
        ac = AsyncCompile()
        k = ac.triton(KERNEL_NAME, src, device_str="cuda")
        g = {KERNEL_NAME: k}
        ac.wait(g)
        k = g[KERNEL_NAME]
        dev = "cuda"
        t_ids = torch.from_numpy(ids.astype(np.int32)).to(dev)
        t_tab = torch.from_numpy(table.view(np.int16)).view(torch.bfloat16).to(dev)
        t_w = torch.from_numpy(w.view(np.int16)).view(torch.bfloat16).to(dev)
        buf0 = torch.empty((N, H), dtype=torch.bfloat16, device=dev)
        buf3 = torch.empty((N, H), dtype=torch.bfloat16, device=dev)
        # the kernel indexes the table as 576*id: patch the bounds assert by giving it a table with
        # the compact row count (the source asserts id < 49152, compact ids are < U <= N, fine)
        stream = torch.cuda.current_stream().cuda_stream
        k.run(t_ids, t_tab, t_w, buf0, buf3, N, H, stream=stream)
        torch.cuda.synchronize()
        comp = buf3.cpu().view(torch.int16).numpy().view(np.uint16)
        emb = buf0.cpu().view(torch.int16).numpy().view(np.uint16)
        print("compiled kernel re-run == captured compiled_out:", bool(np.array_equal(comp, comp_cap)))
        print("compiled kernel embedding output == gathered x   :", bool(np.array_equal(emb, x)))
        out["compiled_rerun_equals_capture"] = bool(np.array_equal(comp, comp_cap))
        try:
            from vllm.model_executor.determinism.batch_invariant import rms_norm_batch_invariant
        except ImportError:
            from vllm.model_executor.layers.batch_invariant import rms_norm_batch_invariant
        t_x = torch.from_numpy(x.view(np.int16)).view(torch.bfloat16).to(dev)
        y = rms_norm_batch_invariant(t_x, t_w, EPS)
        torch.cuda.synchronize()
        eag = y.cpu().view(torch.int16).numpy().view(np.uint16)
        print("eager kernel re-run   == captured eager_out   :", bool(np.array_equal(eag, eager_cap)))
        out["eager_rerun_equals_capture"] = bool(np.array_equal(eag, eager_cap))
        fd = first_diff(eag, comp, H)
        print("FIRST DIFFERING WORD (eager vs compiled, re-run):", json.dumps(fd))
        out["rerun"] = fd
    # CPU model: single-rounding row model with per-row scalar s; fit ulp offsets
    s0 = ieee_s0(x, EPS)
    ke = fit_k(eager_cap, x, w, s0); kc = fit_k(comp_cap, x, w, s0)
    def hist(k):
        u, c = np.unique(k, return_counts=True); return {("none" if int(a_) == 99 else str(int(a_))): int(b_) for a_, b_ in zip(u, c)}
    print("rows reproduced by bf16((x*s)*w) with s = s0 + k ulp; k histogram eager:", hist(ke), "compiled:", hist(kc))
    diff_rows = np.nonzero(np.any(eager_cap != comp_cap, axis=1))[0]
    for r in diff_rows[:5]:
        print(f"row {int(r)}: eager k={int(ke[r])} compiled k={int(kc[r])} s0={float(s0[r]):.9g} (mean(x^2)+eps={float(1/np.float64(s0[r])**2):.9g})")
    rows_same_s = int(np.sum((ke == kc) & (ke != 99)))
    print(f"rows where both kernels use the same s: {rows_same_s}/{N}; rows with different s but bit-equal output: {int(np.sum((ke != kc) & (ke != 99) & (kc != 99) & ~np.isin(np.arange(N), diff_rows)))}")
    out["k_hist"] = {"eager": hist(ke), "compiled": hist(kc)}
    out["rows_with_different_s"] = int(np.sum(ke != kc)); out["rows_with_output_diff"] = int(diff_rows.size)
    json.dump(out, open(os.path.join(HERE, "repro_result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
'''


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(a) -> None:
    E, C, CMP, OUT = Path(a.eager), Path(a.compiled), Path(a.compare), Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    eidx = json.load(open(E / "index.json"))
    cidx = json.load(open(C / "index.json"))
    rep = json.load(open(CMP / "report.json"))
    fd = rep["summary"]["first_divergence"]
    assert fd["role"] == "rms_norm(normed)" and fd["layer"] == 0, fd
    cev = next(e for e in cidx["events"] if e.get("seq") == fd["seq_compiled"])
    eev = next(e for e in eidx["events"] if e.get("seq") == fd["seq_eager"])
    ids_ev = next(e for e in cidx["events"] if e.get("kernel_call_index") == 0 and e.get("arg") == "arg0_1")
    ids = np.load(C / "values" / ids_ev["file"]).astype(np.int64)
    emb_ev = next(e for e in eidx["events"] if e["node"] == "model.embed_tokens" and e["kind"] == "module_out")
    emb = np.load(E / "values" / emb_ev["file"])  # [N, H] uint16 = table[ids]
    uniq, inv = np.unique(ids, return_inverse=True)
    table = np.zeros((len(uniq), emb.shape[1]), np.uint16)
    for j, tok in enumerate(uniq):
        rows = np.nonzero(ids == tok)[0]
        table[j] = emb[rows[0]]
        assert all(np.array_equal(emb[r], emb[rows[0]]) for r in rows)
    params = np.load(E / "small_params.npz")
    w = params["model.layers.0.input_layernorm.weight"]
    np.save(OUT / "input_ids.npy", inv.astype(np.int32))
    np.save(OUT / "embed_rows.npy", table)
    np.save(OUT / "norm_weight.npy", w)
    np.save(OUT / "eager_out.npy", np.load(E / "values" / eev["file"]))
    np.save(OUT / "compiled_out.npy", np.load(C / "values" / cev["file"]))
    np.save(OUT / "original_token_ids.npy", ids.astype(np.int32))
    ksrc = CMP / f"{fd['kernel']}.py"
    shutil.copy2(ksrc, OUT / "inductor_kernel.py")
    (OUT / "repro.py").write_text(REPRO)
    meta = {
        "request_id": cidx["result"]["request_id"],
        "prompt_len": cidx["result"]["prompt_len"],
        "first_divergence": fd,
        "kernel": rep["summary"].get("first_divergence_kernel"),
        "kernel_call": rep["summary"].get("first_divergence_kernel_call"),
        "eager_impl": "vllm.model_executor.determinism.batch_invariant._rms_norm_kernel (Triton, BLOCK_SIZE=1024, 1.0/tl.sqrt), via RMSNorm.forward_cuda under VLLM_BATCH_INVARIANT=1, enforce_eager",
        "compiled_impl": "Inductor persistent reduction fusing aten.embedding + vllm_ir.rms_norm (native decomposition), libdevice.rsqrt, XBLOCK/R0_BLOCK=1024",
        "eager_effective": json.load(open(E / "effective.json"))["compilation"],
        "compiled_effective": json.load(open(C / "effective.json"))["compilation"],
        "norm_weight_sha256": hashlib.sha256(w.tobytes()).hexdigest(),
        "embedding_rows": int(len(uniq)),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    lines = [
        "# divergence-B0-784c37e1: first compiled/eager divergence (SmolLM2-135M, request 784c37e1)",
        "",
        f"First differing canonical value: layer 0, `input_layernorm` (rms_norm) output, position {fd['first_diff']['position']}, "
        f"coordinate {fd['first_diff']['coordinate']}: eager `{fd['first_diff']['expected_bits']}` ({fd['first_diff']['expected_value']}) vs compiled "
        f"`{fd['first_diff']['actual_bits']}` ({fd['first_diff']['actual_value']}), {fd['n_diff']} of {fd['n_words']} words differ (1 ulp).",
        "",
        f"Compiled kernel: `{fd['kernel']}` (Inductor, source `inductor_kernel.py`, sha256 `{meta['kernel']['sha256'] if meta.get('kernel') else '?'}`).",
        "Eager kernel: vLLM batch-invariant Triton `_rms_norm_kernel`.",
        "",
        "Run `python repro.py` on a GPU host with the pinned torch/vLLM (both kernels are re-executed on the saved operands and",
        "bit-compared), or `python repro.py --cpu` anywhere (numpy model + the captured outputs). See `meta.json` for the",
        "capture identity and `docs/vllm-poc/numerics.md` (divergence diagnosis) for the analysis.",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    sums = []
    for p in sorted(OUT.iterdir()):
        if p.name != "sha256sums.txt":
            sums.append(f"{sha(p)}  {p.name}")
    (OUT / "sha256sums.txt").write_text("\n".join(sums) + "\n")
    total = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"bundle {OUT}: {total/1e6:.2f} MB, {len(sums)} files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--eager", required=True)
    ap.add_argument("--compiled", required=True)
    ap.add_argument("--compare", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a)
    else:
        sys.exit(2)
