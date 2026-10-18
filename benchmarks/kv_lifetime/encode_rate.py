"""kv.cu's encode rate in 4 KiB blocks (one Llama-3.1-8B token-layer of bf16 K and V) a second on one GPU, for the
honest cost of appending and re-keying KV: run by `benchmarks/word_codec/run.sh` (KV=1, KV_SCRIPT=
../kv_lifetime/encode_rate.py), which builds kv.cu at each of KV_FIELDS and passes `--lib NAME=PATH` for each.

For each library and each k of `--ks`:
  chain    the compute-only rate (`kv_bench.rates`: `kv_chain`), words a second, and the squarings it implies
  store    `kv_encode` over a `--store-gib` store, contiguous (an append at the write, under the seed's key)
  bound    `kv_encode_bound` over the same store, a random bound offset a token-layer (the appended block's key
           b_j(c_a), which a re-key draws afresh)
  rekey    `kv_encode_bound` in place over `--pages` scattered pages of `--page-blocks` token-layers each, one call a
           page (a vLLM block of 16 tokens is 16 token-layers a layer; the re-key encodes a page where it is)
each checked first: the bound encode decodes back under its offsets. Writes encode_rate.json under `--out`.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "word_codec"))

import kv as K  # noqa: E402
import kv_bench as B  # noqa: E402


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lib", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="2,3")
    ap.add_argument("--store-gib", type=float, default=4.0)
    ap.add_argument("--pages", type=int, default=2048)
    ap.add_argument("--page-blocks", type=int, default=16)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    import torch

    ks = [int(k) for k in a.ks.split(",")]
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    res = {"gpu": torch.cuda.get_device_name(), "sms": sms, "args": vars(a), "libs": {}}
    for spec in a.lib:
        name, path = spec.split("=", 1)
        lib = B.Lib(name, path)
        f = lib.field
        w = K.token_layer_words(f)
        block_bytes = w * f.word_bytes
        n_blocks = int(a.store_gib * (1 << 30)) // block_bytes
        n = n_blocks * w
        st = B.stream(torch)
        store = B.random_words(torch, n, f, 7)
        out = torch.empty_like(store)
        back = torch.empty_like(store)
        g = torch.Generator(device="cuda").manual_seed(11)
        offs = torch.randint(0, 256, (n_blocks, f.word_bytes), dtype=torch.uint8, device="cuda", generator=g)
        offs[:, -1] &= 0x3F
        offs = offs.view(-1)
        r = {"field": f.name, "mask": lib.mask, "word_bytes": f.word_bytes, "token_layer_words": w,
             "block_bytes": block_bytes, "store_blocks": n_blocks, "rates": B.rates(lib, torch, ks, sms, a.reps, log),
             "k": {}}
        for k in ks:
            B.set_key(lib, B.lib_key(lib, k))
            lib.call("kv_encode_bound", B.ptr(store), B.ptr(out), n, 0, k, B.ptr(offs), 0, st)
            lib.call("kv_decode_bound", B.ptr(out), B.ptr(back), n, 0, k, B.ptr(offs), 0, st)
            torch.cuda.synchronize()
            ok = bool(torch.equal(back, store))
            t_store = statistics.median(B.timed(torch, lambda: lib.call(
                "kv_encode", B.ptr(store), B.ptr(out), n, 0, k, 0, st), a.reps))
            t_bound = statistics.median(B.timed(torch, lambda: lib.call(
                "kv_encode_bound", B.ptr(store), B.ptr(out), n, 0, k, B.ptr(offs), 0, st), a.reps))
            pg = a.page_blocks
            stride = max(pg, n_blocks // a.pages)
            starts = [i * stride for i in range(min(a.pages, n_blocks // stride))]
            pw = pg * w

            def rekey() -> None:
                for s in starts:
                    p = B.ptr(out) + s * block_bytes
                    lib.call("kv_encode_bound", p, p, pw, s * w, k, B.ptr(offs) + s * f.word_bytes, 0, st)

            t_rekey = statistics.median(B.timed(torch, rekey, a.reps))
            rk_blocks = len(starts) * pg
            chain = r["rates"]["encode"][k]["per_s"]
            r["k"][k] = {"bound_round_trip": ok,
                         "chain_blocks_s": chain / w,
                         "store_blocks_s": n_blocks / t_store, "store_GBps": n_blocks * block_bytes / t_store / 1e9,
                         "bound_blocks_s": n_blocks / t_bound, "bound_GBps": n_blocks * block_bytes / t_bound / 1e9,
                         "rekey_blocks_s": rk_blocks / t_rekey, "rekey_GBps": rk_blocks * block_bytes / t_rekey / 1e9,
                         "rekey_pages": len(starts), "rekey_page_blocks": pg}
            log(f"{name} k={k}: bound round trip {'ok' if ok else 'FAIL'}; blocks/s chain {chain / w:.3e}, store "
                f"{n_blocks / t_store:.3e}, bound {n_blocks / t_bound:.3e}, re-key in place {rk_blocks / t_rekey:.3e} "
                f"({len(starts)} pages of {pg})")
        res["libs"][name] = r
        del store, out, back, offs
        torch.cuda.empty_cache()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "encode_rate.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
