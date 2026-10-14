"""`paged.cu`'s kernels timed alone on one GPU at the encoded stack's shapes, on a synthetic layer cache in vLLM's
layout ([blocks][8 KV heads][16 slots][K ‖ V] bf16, encoded at k): decode attention (`pg_attn`) of `--batches` sequences
each of every context of `--ctxs` at every split count of `--splits` (the stack picks one by `paged.splits_for`), the
deferred append of rows tokens over `--layers` layers in one launch (`pg_append_multi`) at each of `--rows`, and a
prefill chunk's gather of `--gather` tokens' blocks (`pg_gather`). Every split count's output is paged_gate.py's to
check (encoded = plain bit for bit); this only times them. Writes `paged-splits.json` under `--out`."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paged as P
from paged import KB

BS = 16


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lib", required=True, help="NAME=PATH, paged.cu built at field NAME")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--batches", default="32,64,128")
    ap.add_argument("--ctxs", default="1280,1536")
    ap.add_argument("--splits", default="1,2,3,4,6,8,12,16")
    ap.add_argument("--rows", default="32,64,128,512,2048")
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--gather", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import torch

    lib = P.PagedLib(*a.lib.split("=", 1))
    sh = lib.shape
    H, G, D, WB = sh["heads"], sh["kv_heads"], sh["head_dim"], sh["wb"]
    KB.set_key(lib, KB.lib_key(lib, a.k))
    batches, ctxs = [int(v) for v in a.batches.split(",")], [int(v) for v in a.ctxs.split(",")]
    blocks = max(max(batches) * -(-max(ctxs) // BS), a.gather // BS, max(int(r) for r in a.rows.split(","))) + 64
    plain = (torch.randn(blocks, G, BS, 2 * D, device="cuda") * 2).to(torch.bfloat16)
    lay = P.layout(plain, G, WB)
    cache = torch.empty_like(plain)
    slots_all = torch.arange(blocks * BS, dtype=torch.int64, device="cuda")
    kk = plain[:, :, :, :D].permute(0, 2, 1, 3).reshape(blocks * BS, G * D).contiguous()
    vv = plain[:, :, :, D:].permute(0, 2, 1, 3).reshape(blocks * BS, G * D).contiguous()
    P.append(lib, torch, cache, lay, 0, a.layers, 0, 1, a.k, kk, vv, slots_all)
    del plain
    rng = random.Random(0)
    perm = torch.tensor(rng.sample(range(blocks), blocks), dtype=torch.int32)

    def timed(fn, *args) -> float:
        fn(*args)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        best = float("inf")
        for _ in range(a.reps):
            ev[0].record()
            fn(*args)
            ev[1].record()
            ev[1].synchronize()
            best = min(best, ev[0].elapsed_time(ev[1]))
        return best

    res = {"lib": lib.name, "shape": sh, "k": a.k, "sms": torch.cuda.get_device_properties(0).multi_processor_count,
           "attention": [], "append": [], "gather": None}
    for b in batches:
        for ctx in ctxs:
            nb = -(-ctx // BS)
            bt = perm[: b * nb].view(b, nb).contiguous().cuda()
            lens = torch.full((b,), ctx, dtype=torch.int32, device="cuda")
            q = torch.randn(b, H * D, device="cuda").to(torch.bfloat16)
            o = torch.empty_like(q)
            row = {"batch": b, "ctx": ctx, "picked": P.splits_for(b, res["sms"], G), "ms": {}}
            for sp in (int(v) for v in a.splits.split(",")):
                part = torch.empty(P.part_floats(lib, b, sp), dtype=torch.float32, device="cuda")
                row["ms"][sp] = timed(P.attn, lib, torch, cache, lay, 0, a.layers, 0, 1, a.k,
                                      q, o, part, lens, bt, b, sp)
            row["best"] = min(row["ms"], key=row["ms"].get)
            res["attention"].append(row)
            print(f"attention B={b} ctx={ctx}: picked {row['picked']}, best {row['best']} "
                  f"{row['ms'][row['best']]:.3f} ms; " + ", ".join(f"{s}: {t:.3f}" for s, t in row["ms"].items()),
                  flush=True)
    for rows in (int(v) for v in a.rows.split(",")):
        kk = (torch.randn(rows, G * D, device="cuda")).to(torch.bfloat16)
        vv = torch.randn_like(kk)
        slots = torch.randperm(blocks * BS, device="cuda")[:rows].to(torch.int64)
        tab = P.table(torch, [(kk, vv, slots, rows, cache, 0, a.layers, l) for l in range(a.layers)]).cuda()
        t = timed(P.append_multi, lib, torch, tab, a.layers, rows, lay, 1, a.k)
        res["append"].append({"rows": rows, "layers": a.layers, "ms": t})
        print(f"append_multi {rows} rows x {a.layers} layers: {t:.3f} ms", flush=True)
    gb = torch.arange(a.gather // BS, dtype=torch.int32, device="cuda")
    go = torch.empty(gb.shape[0], *cache.shape[1:], dtype=torch.bfloat16, device="cuda")
    t = timed(P.gather, lib, torch, cache, lay, 0, a.layers, 0, 1, a.k, gb, go)
    res["gather"] = {"tokens": a.gather, "ms_a_layer": t, "ms_layers": t * a.layers}
    print(f"gather {a.gather} tokens: {t:.3f} ms a layer, {t * a.layers:.2f} ms over {a.layers} layers", flush=True)
    (out / "paged-splits.json").write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
