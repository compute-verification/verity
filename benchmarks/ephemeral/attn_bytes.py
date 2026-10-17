"""What the encoded-KV attention kernel reads from HBM besides the KV: ra-word-codec's `kv_attn` at Llama-3.1-70B's
attention (64 query heads on 8 KV heads of 128, bf16 KV, one query token a sequence), its tensor-core forms over the plain
cache and over its words decoded inside the kernel at k rounds (PAIRS: the warp-specialized form 5 against plain 3, and
form 7, at the standalone decode's occupancy, against plain 6), under Nsight Compute's DRAM counters.

``run``: for each batch × context, a cache of L layers (the layers before the profiled ones cover more than the L2, so a
profiled layer's KV is cold, as in a decode step).  Before anything is profiled, form 5's output equals form 3's bit for
bit at every split count, at layers 0 and L - 1, and each plain form's is within RTOL of float64 attention where the reference
fits (batch × context <= REF_TOKENS).  Then, per form and split count of SPLITS, one unprofiled pass over every layer, and a window
(opened by `kv_spin`) over the first P layers.  ``attribute``: the windows' DRAM bytes per layer, split into the attention
kernel's and the combine's, against the KV bytes the layer must read (plain or stored), and the excess beside the split
partials' size at most (a split count's whole-tile ranges may be fewer).
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import sys
from pathlib import Path

LAYER_COLD_BYTES = 384 << 20  # the unprofiled layers after the profiled ones: 3x the L2
PROFILED = 4
MAX_CELL_BYTES = 48 << 30
REF_TOKENS = 1 << 15
RTOL = 8e-3  # the tensor-core forms round P to bf16 (ra-word-codec's kv_exhaust)
PAIRS = {5: 3, 7: 6}
SPLITS = (1, 2, 4, 8, 16)
READ, WRITE = "dram__bytes_op_read.sum", "dram__bytes_op_write.sum"


def shape(lib) -> dict:
    a = (ctypes.c_int * 10)()
    lib.so.kv_attn_shape(ctypes.cast(a, ctypes.c_void_p))
    return dict(zip(("heads", "kv_heads", "head_dim", "tlw", "tle", "tt", "part", "mw", "mma_ok", "mt"), a))


def run(args) -> int:
    sys.path.insert(0, args.src)
    import kv_bench as B
    import torch

    lib = B.Lib("p512", args.lib)
    K, f = B.K, lib.field
    sh = shape(lib)
    if not sh["mma_ok"] or (sh["heads"], sh["kv_heads"], sh["head_dim"]) != (64, K.KV_HEADS, K.HEAD_DIM):
        raise SystemExit(f"kv_attn_shape {sh}: no tensor-core form at 70B's attention")
    k = args.k
    B.set_key(lib, K.key(f.name, B.SEED, k))
    st, sms = B.stream(torch), torch.cuda.get_device_properties(0).multi_processor_count
    tl, tle = K.TOKEN_LAYER_BYTES, sh["tle"]
    scale = math.log2(math.e) / math.sqrt(K.HEAD_DIM)
    res: dict = {"field": f.name, "k": k, "shape": sh, "gpu": torch.cuda.get_device_name(0), "sms": sms, "cells": [],
                 "windows": []}
    for b in [int(x) for x in args.batches.split(",")]:
        for ctx in [int(x) for x in args.contexts.split(",")]:
            per_layer = b * ctx * tl
            p = PROFILED if per_layer < LAYER_COLD_BYTES else 1
            layers = p + max(1, -(-LAYER_COLD_BYTES // per_layer))
            if layers * b * ctx * (tl + tle) > MAX_CELL_BYTES:
                layers = p + 1
            if layers * b * ctx * (tl + tle) > MAX_CELL_BYTES:
                res["cells"].append({"batch": b, "context": ctx, "skipped": "does not fit"})
                continue
            g = torch.Generator(device="cuda").manual_seed(b * 100003 + ctx)
            plain = torch.randn(b * ctx * layers * tl // 2, dtype=torch.float32, device="cuda", generator=g).to(
                torch.bfloat16)
            pad = torch.zeros(b * ctx * layers, tle, dtype=torch.uint8, device="cuda")
            pad[:, :tl] = plain.view(torch.uint8).view(-1, tl)
            enc = torch.empty_like(pad)
            lib.call("kv_encode", B.ptr(pad), B.ptr(enc), pad.numel() // f.word_bytes, 0, k, sms * 8, st)
            del pad
            q = (torch.randn(b, 64, K.HEAD_DIM, device="cuda", generator=g) * scale).to(torch.bfloat16).contiguous()
            part = torch.empty(b * max(SPLITS) * sh["part"] * sh["mw"], dtype=torch.float32, device="cuda")
            o = {fm: torch.empty(b, 64, K.HEAD_DIM, dtype=torch.float32, device="cuda") for fm in (*PAIRS, *PAIRS.values())}

            def call(form: int, layer: int, sp: int) -> None:
                store = enc if form in PAIRS else plain
                lib.call("kv_attn", B.ptr(store), form, B.ptr(q), B.ptr(o[form]), B.ptr(part), b, ctx, layers, layer, sp,
                         k, st)

            gate = {"equal": True, "rel_err": None}
            kvl = plain.view(b, ctx, layers, 2, K.KV_HEADS, K.HEAD_DIM)
            for layer in (0, layers - 1):
                ref = None
                if b * ctx <= REF_TOKENS:
                    kk, vv = kvl[:, :, layer, 0].double(), kvl[:, :, layer, 1].double()
                    qq = q.double().view(b, K.KV_HEADS, 64 // K.KV_HEADS, K.HEAD_DIM)
                    s = torch.einsum("bgqd,btgd->bgqt", qq, kk)
                    pr = torch.exp2(s - s.amax(-1, keepdim=True))
                    ref = torch.einsum("bgqt,btgd->bgqd", pr / pr.sum(-1, keepdim=True), vv).reshape(b, 64, K.HEAD_DIM)
                    del kk, vv, s, pr
                for sp in SPLITS:
                    for enc_form, plain_form in PAIRS.items():
                        call(plain_form, layer, sp)
                        o[enc_form].zero_()
                        call(enc_form, layer, sp)
                        torch.cuda.synchronize()
                        gate["equal"] &= bool(torch.equal(o[plain_form], o[enc_form]))
                        if ref is not None:
                            err = float((o[plain_form].double() - ref).abs().max() / ref.abs().max())
                            gate["rel_err"] = max(gate["rel_err"] or 0.0, err)
                del ref
            gate["ok"] = gate["equal"] and (gate["rel_err"] is None or gate["rel_err"] <= RTOL)
            cell = {"batch": b, "context": ctx, "layers": layers, "profiled_layers": p, "splits": list(SPLITS),
                    "plain_bytes": per_layer, "encoded_bytes": b * ctx * tle, "gate": gate}
            res["cells"].append(cell)
            print(f"B={b} ctx={ctx}: {layers} layers, gate {gate}", flush=True)
            if gate["ok"]:
                for form in (*PAIRS.values(), *PAIRS):
                    for sp in SPLITS:
                        for layer in range(layers):
                            call(form, layer, sp)
                        torch.cuda.synchronize()
                        torch.cuda.profiler.start()
                        lib.call("kv_spin", 1000, st)
                        for layer in range(p):
                            call(form, layer, sp)
                        torch.cuda.synchronize()
                        torch.cuda.profiler.stop()
                        mw = 1 if form >= 6 else sh["mw"]
                        res["windows"].append({"batch": b, "context": ctx, "form": form, "splits": sp, "layers": p,
                                               "partial_bytes_max": b * sp * sh["part"] * mw * 4})
            del plain, enc, q, part, o, kvl
            torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(res, indent=1))
    return 0 if all(c.get("gate", {}).get("ok", True) for c in res["cells"]) else 2


def attribute(args) -> int:
    res = json.loads(Path(args.out).read_text())
    lines = [ln for ln in Path(args.csv).read_text().splitlines() if ln.startswith('"')]
    r = list(csv.reader(lines))
    head = r[0]
    ws: list[list[dict]] = []
    for row in r[2:]:
        d = dict(zip(head, row))
        name = d["Kernel Name"]
        if "spin_kernel" in name:
            ws.append([])
        elif ws:
            ws[-1].append({"name": name, "read": float(d[READ].replace(",", "") or 0),
                           "write": float(d[WRITE].replace(",", "") or 0)})
    if len(ws) != len(res["windows"]):
        raise SystemExit(f"{len(ws)} windows in the profile for {len(res['windows'])} recorded")
    cells = {(c["batch"], c["context"]): c for c in res["cells"]}
    print(f"{'B':>3} {'ctx':>6} {'form':>4} {'split':>5} {'kv MB':>9} {'read MB':>9} {'excess':>8} {'comb rd':>8} "
          f"{'write MB':>8} {'part MB':>8}")
    for w, k in zip(res["windows"], ws):
        c = cells[(w["batch"], w["context"])]
        kv = c["encoded_bytes"] if w["form"] in PAIRS else c["plain_bytes"]
        L = w["layers"]
        attn = [x for x in k if "combine" not in x["name"]]
        comb = [x for x in k if "combine" in x["name"]]
        w.update({"kv_bytes": kv, "read": sum(x["read"] for x in k) / L, "write": sum(x["write"] for x in k) / L,
                  "attn_read": sum(x["read"] for x in attn) / L, "combine_read": sum(x["read"] for x in comb) / L,
                  "kernels": sorted({x["name"][:80] for x in k})})
        w["excess_read"] = w["read"] - kv
        w["excess_share"] = w["excess_read"] / w["read"]
        print(f"{w['batch']:3d} {w['context']:6d} {w['form']:4d} {w['splits']:5d} {kv / 1e6:9.2f} {w['read'] / 1e6:9.2f} "
              f"{w['excess_share']:8.2%} {w['combine_read'] / 1e6:8.3f} {w['write'] / 1e6:8.3f} "
              f"{w['partial_bytes_max'] / 1e6:8.3f}")
    Path(args.out).write_text(json.dumps(res, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("run", "attribute"))
    ap.add_argument("--src", help="the directory holding kv.py and kv_bench.py")
    ap.add_argument("--lib", help="kv.cu built at KV_FIELD = 512")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--batches", default="1,4,16,64")
    ap.add_argument("--contexts", default="2048,8192,32768")
    ap.add_argument("--csv", help="attribute: ncu --page raw --print-units base --csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    return run(args) if args.mode == "run" else attribute(args)


if __name__ == "__main__":
    sys.exit(main())
