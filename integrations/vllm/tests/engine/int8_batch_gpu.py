"""The batched int8 attention's conformance on one GPU: `engine.int8_ops.paged_attention` and `window_attention` (one kv_quant_batch or
kv_quant_window_batch and one window_rows_batch, on CUDA the Triton kernels of `int8_attention_triton`) against
`int8_attention.reference_paged` (the per-request, per-KV-head reference) on random batches, word for word: the served output
on the GPU (from the paged caches, and from a credited read's packed rows) against the reference on the GPU and on the CPU,
and the kv ops' results byte for byte against the CPU torch body (padding included) and their per-request slices against
`kv_quant` on the CPU.  Batches mix decode rows, chunked-prefill chunks and whole prompts over varied windows, at the test
shapes (H 8 / G 2 / d 64, H 4 / G 1 / d 32, H 12 / G 4 / d 16, H 6 / G 3 / d 80) and Llama-3.1-8B's (H 32 / G 8 / d 128) up
to windows of 8192 keys; logit scales from 1e-3 to 300 (A saturating past 2^32 at the top), and values from float32-subnormal
(2^-130) to 2^100 in magnitude.

Writes ``<out>/int8_batch_gpu.json``: per case its shape, rows, windows and mismatched words (every count must be 0).

    cd integrations/vllm; . ../../benchmarks/harness/env.sh
    $VENV/bin/python -m tests.engine.int8_batch_gpu --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from verity_vllm.engine import int8_attention as A
from verity_vllm.engine import int8_ops

SCHEMA = "verity-vllm/int8-batch-gpu/v1"
#: (H, G, d, requests, longest window, page size, logit spread[, amplitude: q, k and v times it, a power of two])
CASES = [(8, 2, 64, 6, 200, 16, 1.0), (4, 1, 32, 9, 70, 8, 300.0), (12, 4, 16, 5, 40, 16, 1e-3), (32, 8, 128, 8, 600, 16, 1.0),
         (32, 8, 128, 8, 2100, 16, 1.0), (32, 8, 128, 4, 8192, 16, 1.0), (32, 8, 128, 32, 300, 16, 2.0), (8, 2, 64, 1, 1, 16, 1.0),
         (8, 2, 64, 6, 90, 16, 1.0, 2.0 ** -130), (8, 2, 64, 6, 90, 16, 1.0, 2.0 ** 100), (6, 3, 80, 7, 150, 32, 1.0)]


def batch(gen: torch.Generator, h: int, g: int, d: int, n_req: int, s_hi: int, bs: int, spread: float, amp: float = 1.0):
    """A paged batch: each request a decode row, a chunk or a whole prompt over a random window, under a shuffled page table;
    q, k and v are N(0, 4) times `amp` (2^-130: subnormal in float32, scales too)."""
    lens, rows = [], []
    for r in range(n_req):
        s = int(torch.randint(1, s_hi + 1, (1,), generator=gen)) if r else s_hi
        kind = int(torch.randint(0, 3, (1,), generator=gen))
        m = 1 if kind == 0 or s == 1 else s if kind == 1 and s <= 256 else int(torch.randint(1, min(s, 64) + 1, (1,), generator=gen))
        lens.append(s)
        rows.append(m)
    per = [-(-s // bs) for s in lens]
    pages = sum(per) + 3
    order = torch.randperm(pages, generator=gen)
    kc = torch.zeros(pages, bs, g, d, dtype=torch.bfloat16)
    vc = torch.zeros_like(kc)
    table = torch.full((n_req, max(per) + 2), -1, dtype=torch.int32)
    at = 0
    for r, s in enumerate(lens):
        k = (torch.randn(s, g, d, generator=gen) * 2 * amp).bfloat16()
        v = (torch.randn(s, g, d, generator=gen) * 2 * amp).bfloat16()
        for b in range(per[r]):
            blk = int(order[at])
            at += 1
            table[r, b] = blk
            lo, hi = b * bs, min((b + 1) * bs, s)
            kc[blk, :hi - lo], vc[blk, :hi - lo] = k[lo:hi], v[lo:hi]
    query = (torch.randn(sum(rows), h, d, generator=gen) * 2 * amp).bfloat16()
    qsl = torch.tensor([0, *torch.tensor(rows).cumsum(0).tolist()], dtype=torch.int32)
    kamax = torch.stack([kc[:, :, j].float().abs().amax() for j in range(g)])
    ks = (kamax * torch.rand(g, generator=gen).clamp(min=0.3) / torch.tensor(127.0)).float()
    return query, kc, vc, qsl, torch.tensor(lens, dtype=torch.int32), table, ks, d ** -0.5 * spread


def words(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16).cpu()


def run_case(seed: int, case: tuple) -> dict[str, Any]:
    gen = torch.Generator().manual_seed(seed)
    query, kc, vc, qsl, lens, table, ks, scale = batch(gen, *case)
    cpu_ref = torch.full_like(query, float("nan"))
    A.reference_paged(query, kc, vc, qsl, lens, table, ks, scale, cpu_ref)
    dev = [t.cuda() for t in (query, kc, vc, qsl, lens, table, ks)]
    gpu_ref, served = torch.full_like(dev[0], float("nan")), torch.full_like(dev[0], float("nan"))
    A.reference_paged(*dev[:6], dev[6], scale, gpu_ref)
    int8_ops.paged_attention(*dev[:6], dev[6], scale, served)
    torch.cuda.synchronize()
    ls = lens.tolist()
    kv = int8_ops.kv_quant_batch(dev[1], dev[2], dev[5], ls, dev[6])
    kv_bad = _bytes_differ(kv, int8_ops.kv_quant_batch(kc, vc, table, ls, ks))
    for r, s in enumerate(ls):
        want = A.kv_quant(*A.window_of(kc, vc, table[r], s), ks)
        kv_bad += _bytes_differ((kv[0][r, :, :s], kv[1][r, :, :s], kv[2][r], kv[3][r]), want)
    wins = [A.window_of(kc, vc, table[r], s) for r, s in enumerate(ls)]
    starts = [0, *torch.tensor(ls).cumsum(0).tolist()[:-1]]
    keys, values = (torch.cat([w[j] for w in wins]).cuda() for j in (0, 1))
    credited = torch.full_like(dev[0], float("nan"))
    int8_ops.window_attention(dev[0], keys, values, starts, dev[3], dev[4], dev[6], scale, credited)
    kvw = int8_ops.kv_quant_window_batch(keys, values, starts, ls, dev[6])
    kv_bad += _bytes_differ(kvw, kv)
    sw = words(served)
    return {"seed": seed, "H": case[0], "G": case[1], "d": case[2], "requests": case[3], "rows": int(qsl[-1]), "windows": ls,
            "row_counts": (qsl[1:] - qsl[:-1]).tolist(), "scale": scale, "words": int(sw.numel()),
            "vs_gpu_reference": int((sw != words(gpu_ref)).sum()), "vs_cpu_reference": int((sw != words(cpu_ref)).sum()),
            "gpu_vs_cpu_reference": int((words(gpu_ref) != words(cpu_ref)).sum()),
            "credited_vs_cpu_reference": int((words(credited) != words(cpu_ref)).sum()), "kv_mismatch": kv_bad}


def _bytes_differ(got, want) -> int:
    """Bytes that differ between two tuples of tensors (shapes equal, else every byte)."""
    bad = 0
    for x, y in zip(got, want, strict=True):
        x, y = x.cpu().contiguous().view(torch.uint8), y.cpu().contiguous().view(torch.uint8)
        bad += int((x != y).sum()) if x.shape == y.shape else max(x.numel(), y.numel(), 1)
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="batched int8 attention on the GPU against the per-request reference, word for word")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args(argv)
    rows = []
    for i, case in enumerate(CASES):
        for seed in range(a.seeds):
            rows.append(run_case(1000 * i + seed, case))
            r = rows[-1]
            print(f"[batch] H={r['H']} G={r['G']} d={r['d']} reqs={r['requests']} rows={r['rows']} max_s={max(r['windows'])} "
                  f"words={r['words']} vs_gpu_ref={r['vs_gpu_reference']} vs_cpu_ref={r['vs_cpu_reference']} "
                  f"credited={r['credited_vs_cpu_reference']} kv={r['kv_mismatch']}", flush=True)
    bad = sum(r["vs_gpu_reference"] + r["vs_cpu_reference"] + r["gpu_vs_cpu_reference"] + r["credited_vs_cpu_reference"] + r["kv_mismatch"]
              for r in rows)
    doc = {"schema": SCHEMA, "argv": sys.argv, "torch": torch.__version__, "device": torch.cuda.get_device_name(0), "cases": rows,
           "words": sum(r["words"] for r in rows), "mismatch": bad, "ok": bad == 0}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "int8_batch_gpu.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"INT8-BATCH-GPU ok={doc['ok']} cases={len(rows)} words={doc['words']} mismatch={bad}", flush=True)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
