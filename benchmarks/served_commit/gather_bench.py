"""The gather kernel alone on a prefill-sized step (no hashing, no serving): GB_LAYERS layers of [T, 2048], [T, 3072], [T, 2048] bf16
(Llama-3.2-1B at T = B32 x 512 tokens), flushed per layer as the gather flush points do; `odd` puts a 33 B tensor before each layer so
the 4 B packed stream's destinations walk through every offset mod 16 (the served stream's case), `aligned` leaves them 16 B aligned.
Per (layout, blocks cap): GB/s over GB_ITERS steps, the bytes checked against the reference layout once.  Measurement only.

    python gather_bench.py [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from verity_vllm.commit.committer.native_collect import ext


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    T, L, iters = int(os.environ.get("GB_TOKENS", "16384")), int(os.environ.get("GB_LAYERS", "16")), int(os.environ.get("GB_ITERS", "10"))
    g = torch.Generator(device="cuda").manual_seed(0)
    srcs = [torch.randint(-128, 128, (T * w * 2,), dtype=torch.int8, device="cuda", generator=g).view(torch.bfloat16).view(T, w)
            for w in (2048, 3072, 2048)]
    odd_t = torch.randint(-128, 128, (33,), dtype=torch.int8, device="cuda", generator=g)
    mod = ext()
    rows = []
    for layout in ("aligned", "odd"):
        step = [([odd_t] if layout == "odd" else []) + srcs for _ in range(L)]
        flat = [t for layer in step for t in layer]
        off, ref = 0, []
        for t in flat:
            nb = t.numel() * t.element_size()
            o = (off + 3) & ~3
            ref.append((o, nb))
            off = o + nb
        total = off
        arena = torch.empty(total + 4096, dtype=torch.uint8, device="cuda")
        nbytes = sum(nb for _, nb in ref)
        for cap in (32, 1024):
            c = mod.Collector()
            c.enable_gather(4096)
            c.use_gather = True
            c.blocks_per_desc, c.auto_blocks = cap, True

            def run_step() -> None:
                c.begin_step(1, arena.data_ptr(), arena.numel(), 1)
                i = 0
                for layer in step:
                    for t in layer:
                        c.stage(i, t)
                        i += 1
                    c.flush()
                c.end_step()

            for _ in range(3):
                run_step()
            torch.cuda.synchronize()
            ok = all(torch.equal(arena[o:o + nb], t.reshape(-1).view(torch.uint8)) for t, (o, nb) in zip(flat, ref))
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(iters):
                run_step()
            e1.record()
            torch.cuda.synchronize()
            ms = e0.elapsed_time(e1) / iters
            row = {"layout": layout, "cap": cap, "bytes": nbytes, "descs": len(flat), "flushes": L, "ms_per_step": round(ms, 3),
                   "GB_s": round(nbytes / ms / 1e6, 1), "bytes_ok": ok, "gpu": torch.cuda.get_device_name(0)}
            print(json.dumps(row), flush=True)
            rows.append(row)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
