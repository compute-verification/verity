"""R13 lane tp, P4 step 1 — offline analysis of `tp2_semantics.py` recordings: is BI TP2 the same exact computation as TP1?

For every recorded module call (rank-local input X_r, weight shard W_r, partial GEMM output P_r = argument of the collective,
collective output R_r, module output) this recomputes, on one GPU with vLLM's batch-invariant mode enabled (the same
`matmul_persistent` kernel + tile table the engine used):

  row-parallel (o_proj, down_proj; K sharded, all-reduce):
    local     P_r == BI_linear(X_r, W_r)                              (the rank-local GEMM is the registered Gemm semantics at shard shape K/2)
    ranks     R_0 == R_1                                              (both ranks receive the same reduced value)
    reduce    R   == bf16(P_0 + P_1)  [= bf16(fp32(P_0)+fp32(P_1))]  (the exact all-reduce semantics for 2 ranks)
    tp1       R   vs BI_linear(cat(X_0,X_1), cat(W_0,W_1))           (the TP1 Program's Gemm_v1{K} on the same logical inputs)
  column-parallel (qkv_proj, gate_up_proj; N sharded, no collective):
    inputs    X_0 == X_1
    local     out_r == BI_linear(X_r, W_r)
    tp1       out_r == BI_linear(X, W_tp1)[:, shard_r]               (N-shard vs full-N tile semantics, TP1 row layout [Q;K;V] / [gate;up])
  embed_tokens (vocab sharded, all-reduce of masked partials):  R_0 == R_1;  R == P_0 + P_1;  R == cat(W_0,W_1)[ids]
  logits_processor (lm_head N-sharded + all-gather):            gathered == cat(shards);  shard_r == BI_linear(h, Wlm_r);  vs BI_linear(h, Wlm_tp1)

and, when `--tp1-dir` is given (the same workload recorded at --tp 1), compares the TP2 values with the TP1 engine's values
for the same module call (layer 0 = identical inputs by construction; later layers show the accumulated drift).
Writes <out>/semantics.json and semantics.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def bi_enable() -> None:
    import os
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    from vllm.model_executor.determinism import batch_invariant as bi
    bi.init_batch_invariance()
    assert bi._batch_invariant_MODE, "batch-invariant mode not enabled"


def bi_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # what UnquantizedLinearMethod.apply does under BI: torch.nn.functional.linear -> aten::linear override -> matmul_persistent
    return torch.nn.functional.linear(x, w)


def cmp(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.reshape(-1); b = b.reshape(-1)
    if a.shape != b.shape:
        return {"equal": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    ne = a != b
    n = int(ne.sum())
    out = {"equal": n == 0, "n_mismatch": n, "n": int(a.numel()), "frac": round(n / max(1, a.numel()), 6)}
    if n:
        d = (a.float() - b.float()).abs()
        ai = a.view(torch.int16).int() if a.dtype == torch.bfloat16 else None
        out["max_abs_diff"] = float(d.max())
        out["max_rel_diff"] = float((d / b.float().abs().clamp_min(1e-6)).max())
        if ai is not None:
            bi_ = b.view(torch.int16).int()
            out["max_ulp_bf16"] = int((ai - bi_).abs()[ne].max())
            out["ulp_hist"] = {str(k): int(v) for k, v in zip(*torch.unique((ai - bi_).abs()[ne], return_counts=True))}
    return out


def load(d: Path, name: str, call: int) -> dict | None:
    p = d / f"{name}.call{call}.pt"
    return torch.load(p, weights_only=False) if p.exists() else None


def split_sizes(attrs: dict, tp: int) -> list[int] | None:
    s = attrs.get("output_sizes")
    return [int(v) // tp for v in s] if s else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp2-dir", required=True)
    ap.add_argument("--tp1-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    bi_enable()
    dev = torch.device("cuda")
    d2 = Path(args.tp2_dir)
    out = Path(args.out or d2)
    meta = [json.load(open(d2 / f"rank{r}" / "meta.json")) for r in range(2)]
    W = [torch.load(d2 / f"rank{r}" / "weights.pt", weights_only=False) for r in range(2)]
    recs = json.load(open(d2 / "rank0" / "records.json"))
    d1 = Path(args.tp1_dir) / "rank0" if args.tp1_dir else None
    W1 = torch.load(d1 / "weights.pt", weights_only=False) if d1 else None
    results: list[dict] = []
    G = lambda t: t.to(dev)
    for rec in recs:
        if rec.get("missing"):
            continue
        name, call = rec["name"], rec["call"]
        A = [load(d2 / f"rank{r}", name, call) for r in range(2)]
        if any(a is None for a in A):
            continue
        cls = A[0]["module_class"]
        attrs = A[0]["module_attrs"]
        res = {"module": name, "call": call, "class": cls, "in_shape": rec["in_shape"], "out_shape": rec["out_shape"], "checks": {}}
        C = res["checks"]
        if cls == "RowParallelLinear":
            X = [G(a["in"][0]) for a in A]; Wr = [G(W[r][f"{name}.weight"]) for r in range(2)]
            P = [G(a["coll"][0]["in"]) for a in A]; R = [G(a["coll"][0]["out"]) for a in A]
            for r in range(2):
                C[f"local_rank{r}: P_r == BI_linear(X_r, W_r)"] = cmp(P[r], bi_linear(X[r], Wr[r]))
                C[f"module_out_rank{r} == R_r"] = cmp(G(A[r]["out"]), R[r])
            C["ranks: R_0 == R_1"] = cmp(R[0], R[1])
            C["reduce: R == bf16(P_0 + P_1) (torch bf16 add)"] = cmp(R[0], P[0] + P[1])
            C["reduce: R == bf16(fp32(P_0) + fp32(P_1))"] = cmp(R[0], (P[0].float() + P[1].float()).to(torch.bfloat16))
            C["reduce: R == bf16(P_1 + P_0) (order swapped)"] = cmp(R[0], P[1] + P[0])
            Xf = torch.cat(X, dim=-1); Wf = torch.cat(Wr, dim=1)
            T = bi_linear(Xf, Wf)
            C["tp1: R vs BI_linear(cat X, cat W) [TP1 Gemm_v1 K=full]"] = cmp(R[0], T)
            C["tp1: fp32 ref vs R (sanity: both are roundings of the exact product)"] = cmp(R[0], (Xf.float() @ Wf.float().t()).to(torch.bfloat16))
            res["shapes"] = {"X_r": list(X[0].shape), "W_r": list(Wr[0].shape), "K_full": int(Xf.shape[-1])}
        elif cls in ("QKVParallelLinear", "MergedColumnParallelLinear", "ColumnParallelLinear"):
            X = [G(a["in"][0]) for a in A]; Wr = [G(W[r][f"{name}.weight"]) for r in range(2)]
            O = [G(a["out"]) for a in A]
            C["inputs: X_0 == X_1"] = cmp(X[0], X[1])
            for r in range(2):
                C[f"local_rank{r}: out_r == BI_linear(X_r, W_r)"] = cmp(O[r], bi_linear(X[r], Wr[r]))
            sizes = split_sizes(attrs, 2)
            if sizes:   # TP1 row layout: [part0 of both ranks; part1 of both ranks; ...]
                parts = [list(torch.split(Wr[r], sizes, dim=0)) for r in range(2)]
                Wtp1 = torch.cat([parts[r][i] for i in range(len(sizes)) for r in range(2)], dim=0)
                Tf = bi_linear(X[0], Wtp1)
                off = 0; sl = {0: [], 1: []}
                for i, s in enumerate(sizes):
                    for r in range(2):
                        sl[r].append(Tf[:, off:off + s]); off += s
                for r in range(2):
                    C[f"tp1_layout: out_r == BI_linear(X, W_tp1)[:, shard_r]  rank{r}"] = cmp(O[r], torch.cat(sl[r], dim=1))
            Wcat = torch.cat(Wr, dim=0); Tc = bi_linear(X[0], Wcat)
            n0 = Wr[0].shape[0]
            C["cat_layout: out_0 == BI_linear(X, cat W)[:, :N0]"] = cmp(O[0], Tc[:, :n0])
            C["cat_layout: out_1 == BI_linear(X, cat W)[:, N0:]"] = cmp(O[1], Tc[:, n0:])
            res["shapes"] = {"X": list(X[0].shape), "W_r": list(Wr[0].shape), "output_sizes": attrs.get("output_sizes")}
        elif cls == "VocabParallelEmbedding":
            ids = [G(a["in"][0]) for a in A]; Wr = [G(W[r][f"{name}.weight"]) for r in range(2)]
            P = [G(a["coll"][0]["in"]) for a in A]; R = [G(a["coll"][0]["out"]) for a in A]
            C["inputs: ids_0 == ids_1"] = cmp(ids[0], ids[1])
            C["ranks: R_0 == R_1"] = cmp(R[0], R[1])
            C["reduce: R == P_0 + P_1"] = cmp(R[0], P[0] + P[1])
            n0 = Wr[0].shape[0]
            own = [(ids[0] < n0), (ids[0] >= n0)]
            for r in range(2):
                zero_rows = (P[r].abs().sum(-1) == 0)
                C[f"partial_rank{r}: nonzero exactly on the rows this rank owns"] = {"equal": bool((zero_rows == ~own[r]).all()), "rows_owned": int(own[r].sum())}
            Wf = torch.cat(Wr, dim=0)
            C["tp1: R == cat(W_0,W_1)[ids]"] = cmp(R[0], Wf[ids[0].long()])
            res["shapes"] = {"ids": list(ids[0].shape), "W_r": list(Wr[0].shape)}
        elif cls == "LogitsProcessor":
            h = [G(a["in"][1]) if isinstance(a["in"][1], torch.Tensor) else None for a in A]
            Wl = [G(W[r]["lm_head.weight"]) for r in range(2)]
            S = [G(a["coll"][0]["in"]) for a in A]; Gt = [G(a["coll"][0]["out"]) for a in A]
            res["collective"] = A[0]["coll"][0]["kind"]
            if h[0] is not None:
                C["inputs: h_0 == h_1"] = cmp(h[0], h[1])
                for r in range(2):
                    C[f"local_rank{r}: shard_r == BI_linear(h, Wlm_r)"] = cmp(S[r], bi_linear(h[r], Wl[r]))
                C["gather: G == cat(S_0, S_1)"] = cmp(Gt[0], torch.cat(S, dim=-1))
                C["ranks: G_0 == G_1"] = cmp(Gt[0], Gt[1])
                Tf = bi_linear(h[0], torch.cat(Wl, dim=0))
                C["tp1: G vs BI_linear(h, W_lm full) [TP1 Gemm_v1 N=V]"] = cmp(Gt[0], Tf)
                O = [G(a["out"]) for a in A]
                C["module_out == G[..., :org_vocab]"] = cmp(O[0], Gt[0][..., :O[0].shape[-1]])
            res["shapes"] = {"h": list(h[0].shape) if h[0] is not None else None, "Wlm_r": list(Wl[0].shape), "gathered": list(Gt[0].shape)}
        else:
            res["skipped"] = cls
        # TP1 engine values for the same module call
        if d1 is not None:
            B = load(d1, name, call)
            if B is not None:
                if cls == "RowParallelLinear":
                    C["engine_tp1: cat(X_0,X_1) == X_tp1"] = cmp(torch.cat([G(a["in"][0]) for a in A], dim=-1), G(B["in"][0]))
                    C["engine_tp1: R (TP2) vs out (TP1 engine)"] = cmp(G(A[0]["coll"][0]["out"]), G(B["out"]))
                    C["engine_tp1: W_tp1 == cat(W_0,W_1)"] = cmp(torch.cat([G(W[r][f"{name}.weight"]) for r in range(2)], dim=1), G(W1[f"{name}.weight"]))
                elif cls in ("QKVParallelLinear", "MergedColumnParallelLinear", "ColumnParallelLinear"):
                    C["engine_tp1: X (TP2) == X (TP1 engine)"] = cmp(G(A[0]["in"][0]), G(B["in"][0]))
                    sizes = split_sizes(attrs, 2)
                    if sizes:
                        Ot = G(B["out"]); off = 0; parts = {0: [], 1: []}
                        for s in sizes:
                            for r in range(2):
                                parts[r].append(Ot[:, off:off + s]); off += s
                        for r in range(2):
                            C[f"engine_tp1: out_r (TP2) == out_tp1[:, shard_r]  rank{r}"] = cmp(G(A[r]["out"]), torch.cat(parts[r], dim=1))
                elif cls == "VocabParallelEmbedding":
                    C["engine_tp1: R (TP2) == out (TP1 engine)"] = cmp(G(A[0]["coll"][0]["out"]), G(B["out"]))
                elif cls == "LogitsProcessor":
                    C["engine_tp1: logits (TP2) vs logits (TP1 engine)"] = cmp(G(A[0]["out"]), G(B["out"]))
                    a2 = G(A[0]["out"]).float().argmax(-1); a1 = G(B["out"]).float().argmax(-1)
                    C["engine_tp1: argmax equal"] = {"equal": bool((a2 == a1).all()), "n": int(a2.numel())}
        results.append(res)
        print(f"[sem] {name}#{call} {cls}: " + "; ".join(f"{k.split(':')[0]}={'OK' if v.get('equal') else ('MISMATCH ' + str(v.get('n_mismatch', v)))}" for k, v in C.items()), flush=True)

    facts = meta[0]["facts"]
    doc = {"schema": "verity-r13/tp2-semantics/v1", "tp2_dir": str(d2), "tp1_dir": str(d1) if d1 else None,
           "collective_path": {"device_communicator": facts["tp"].get("device_communicator"), "pynccl_comm": facts["tp"].get("pynccl_comm"),
                               "ca_comm": facts["tp"].get("ca_comm"), "qr_comm": facts["tp"].get("qr_comm"), "symm_mem_comm": facts["tp"].get("symm_mem_comm"),
                               "use_custom_op_call": facts["tp"].get("use_custom_op_call"), "nccl_env": facts.get("nccl_env"),
                               "parallel_config": facts.get("parallel_config")},
           "tokens_tp2": json.load(open(d2 / "tokens.json"))["requests"][0]["generated_token_ids"],
           "tokens_tp1": json.load(open(Path(args.tp1_dir) / "tokens.json"))["requests"][0]["generated_token_ids"] if d1 else None,
           "results": results}
    doc["tokens_equal_tp1_tp2"] = (doc["tokens_tp1"] == doc["tokens_tp2"]) if d1 else None
    (out / "semantics.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    lines = ["# TP2 vs TP1 numerical semantics (Llama-3.2-1B, BI, L40S x2, pinned vLLM d9105ea80)", "",
             f"collective path: {doc['collective_path']['device_communicator']} / pynccl {doc['collective_path']['pynccl_comm']} / custom-allreduce {doc['collective_path']['ca_comm']} / "
             f"quickreduce {doc['collective_path']['qr_comm']} / symm_mem {doc['collective_path']['symm_mem_comm']}; NCCL env {doc['collective_path']['nccl_env']}", "",
             f"tokens TP1 == TP2: {doc['tokens_equal_tp1_tp2']}", "", "| module | call | check | result |", "|---|---|---|---|"]
    for res in results:
        for k, v in res["checks"].items():
            r = "**equal**" if v.get("equal") else f"MISMATCH {v.get('n_mismatch')}/{v.get('n')} ({v.get('frac')}), max ulp {v.get('max_ulp_bf16')}, max abs {v.get('max_abs_diff')}"
            lines.append(f"| {res['module']} | {res['call']} | {k} | {r} |")
    (out / "semantics.md").write_text("\n".join(lines) + "\n")
    print(f"[sem] -> {out / 'semantics.json'}")


if __name__ == "__main__":
    main()
