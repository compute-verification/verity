"""The canonical top-k's device time a call on one card at the pin, by rows and keys, in the form `priced.units` reads:
each pass `benchmarks/dsa_topk` offers (present: every wrong fast selection bisected; cut, where its `Canon` takes
``refine``: refined from the candidate's window) wrapped around the pin's `cooperative_topk`, at decode shapes of
next_n = 6 on logits GLM-5.3's context wide. For each shape, distribution and pass: whether the output is
`reference_rows`', the counters a call, and the device time a call inside a CUDA graph (the median of 5 trials of 20
replays, each trial kept) with the fast kernel's alone beside it. Then each pass forced: a first guess that is the true
set with its 8 smallest keys swapped for the next 8 (missed), the pin's overflow (every key above the k-th key's fp16
coarse bin, then the first keys of that bin by index) and the K smallest keys (garbage), timed less the guess's copy.

    topk_units.py OUT_DIR [--quick]     (writes OUT_DIR/units.jsonl, one record a line)
"""
import inspect
import json
import sys
import time
import zlib
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dsa_topk"))
import dsa_topk

UNITS = "units.jsonl"
DEV = torch.device("cuda", 0)
K, W, NEXT_N, WS = 2048, 262144, 6, 1024 * 1024
ROWS = (6, 12, 24)
PASSES = ({"present": {"refine": False}, "cut": {}} if "refine" in inspect.signature(dsa_topk.Canon).parameters
          else {"present": {}})


def logits(dist: str, rows: int, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    x = torch.randn(rows, W, generator=g, device=DEV) * 1000.0
    if dist == "randn":
        v = torch.randn(rows, n, generator=g, device=DEV)
    elif dist == "shift8":
        v = 8.0 + torch.randn(rows, n, generator=g, device=DEV)
    elif dist == "shift32":
        v = 32.0 + torch.randn(rows, n, generator=g, device=DEV)
    elif dist == "indexer":
        q = torch.randn(rows, 64, 128, generator=g, device=DEV) / 128 ** 0.5
        k = torch.randn(n, 128, generator=g, device=DEV)
        w = torch.randn(rows, 64, generator=g, device=DEV) / 8.0
        v = torch.zeros(rows, n, device=DEV)
        for c in range(0, n, 32768):
            s = torch.relu(torch.einsum("rhd,nd->rhn", q, k[c:c + 32768]))
            v[:, c:c + 32768] = torch.einsum("rh,rhn->rn", w, s)
    else:
        raise ValueError(dist)
    x[:, :n] = v
    return x


def device_us(fn, iters: int = 20, trials: int = 5) -> tuple[float, list[float]]:
    for _ in range(2):
        fn()
    t = []
    for _ in range(trials):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(100_000_000)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        torch.cuda.synchronize()
        t.append(a.elapsed_time(b) * 1000 / iters)
    return sorted(t)[len(t) // 2], t


def graph_of(fn) -> torch.cuda.CUDAGraph:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def counts(canon) -> dict:
    return dict(zip(dsa_topk.STATS, canon.stats(DEV).tolist()))


def main(argv: list[str]) -> int:
    out, quick = Path(argv[1]), "--quick" in argv
    out.mkdir(parents=True, exist_ok=True)
    dsa_topk.install()
    real = SimpleNamespace(**{op: getattr(torch.ops._C, op).real for op in dsa_topk.OPS if hasattr(torch.ops._C, op)})
    wrapped = {}
    for name, kw in PASSES.items():
        ns = SimpleNamespace(**vars(real))
        wrapped[name] = (ns, dsa_topk.install(ns, dsa_topk.Canon(**kw)))
    log = open(out / UNITS, "a")

    def emit(rec: dict) -> None:
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    emit({"vllm": metadata.version("vllm"), "torch": torch.__version__, "device": torch.cuda.get_device_name(),
          "sms": torch.cuda.get_device_properties(0).multi_processor_count, "K": K, "width": W, "passes": PASSES,
          "how": "device time a call inside a CUDA graph: the median of 5 trials of 20 replays"})
    dists = ["randn", "shift32"] if quick else ["randn", "indexer", "shift8", "shift32"]
    lengths = [20000, 131072] if quick else [20000, 40000, 70000, 100000, 131072]
    t0 = time.time()
    for dist in dists:
        for rows in ROWS:
            for n in lengths:
                x = logits(dist, rows, n, zlib.crc32(f"{dist}/{rows}/{n}".encode()))
                sl = (n - NEXT_N + 1 + torch.arange(NEXT_N, device=DEV, dtype=torch.int32)).repeat(rows // NEXT_N, 1)
                sl = sl.contiguous()
                ln = sl.reshape(-1)
                ws = torch.zeros(WS, dtype=torch.uint8, device=DEV)
                msl = int(ln.max())
                ref = dsa_topk.reference_rows(x, torch.zeros_like(ln), ln, K)
                o_fast = torch.empty((rows, K), dtype=torch.int32, device=DEV)
                g = graph_of(lambda: real.cooperative_topk(x, sl, o_fast, ws, K, msl))
                fast, trials = device_us(g.replay)
                rec = {"dist": dist, "rows": rows, "keys": n, "fast_us": fast, "fast_trials_us": trials}
                del g
                for name, (ns, canon) in wrapped.items():
                    o = torch.empty((rows, K), dtype=torch.int32, device=DEV)
                    fn = lambda: ns.cooperative_topk(x, sl, o, ws, K, msl)  # noqa: E731
                    c0 = counts(canon)
                    g = graph_of(fn)
                    ok = bool(torch.equal(o, ref))
                    c1 = counts(canon)
                    us, trials = device_us(g.replay)
                    c2 = counts(canon)
                    calls = 2 + 5 * 20
                    rec[name] = {"ok": ok, "us": us, "trials_us": trials, "added_us": us - rec["fast_us"],
                                 "segments": canon.segments(DEV, rows, W),
                                 "per_call": {k: (c2[k] - c1[k]) / calls for k in c0}}
                    del g
                emit(rec)
                del x
                torch.cuda.empty_cache()
    for rows in ROWS:
        for n in lengths:
            x = logits("shift32", rows, n, 99)
            sl = (n - NEXT_N + 1 + torch.arange(NEXT_N, device=DEV, dtype=torch.int32)).repeat(rows // NEXT_N, 1)
            sl = sl.contiguous()
            ln = sl.reshape(-1)
            ref = dsa_topk.reference_rows(x, torch.zeros_like(ln), ln, K)
            key = dsa_topk.keys(x[:, :n - NEXT_N + 1])
            order = torch.sort(key, dim=1, descending=True, stable=True).indices
            h = x[:, :n - NEXT_N + 1].to(torch.float16).view(torch.int16).to(torch.int64)
            win = torch.where(h < 0, h ^ 0x7FFF, h) >> 6
            kwin = torch.gather(win, 1, order[:, K - 1:K])
            # above the k-th key's coarse bin first, then its bin by index, then the rest: the first K
            rank = torch.where(win > kwin, 0, torch.where(win == kwin, 1, 2)) * key.shape[1] \
                + torch.arange(key.shape[1], device=DEV)
            over = torch.sort(rank, dim=1).indices[:, :K]
            guesses = {"missed": torch.cat([order[:, :K - 8], order[:, K:K + 8]], dim=1).to(torch.int32).contiguous(),
                       "overflow": over.to(torch.int32).contiguous(),
                       "garbage": order[:, -K:].to(torch.int32).contiguous()}
            for how, guess in guesses.items():
                o = torch.empty((rows, K), dtype=torch.int32, device=DEV)
                g = graph_of(lambda: o.copy_(guess))
                t_copy, copy_trials = device_us(g.replay, 5)
                del g
                rec = {"forced": how, "rows": rows, "keys": n, "copy_us": t_copy, "copy_trials_us": copy_trials}
                for name, (_, canon) in wrapped.items():
                    run = lambda: (o.copy_(guess), canon.run("cooperative_topk", x, o, K, 1, sl, rows=rows,  # noqa: E731
                                                             ncols=W, ls0=x.stride(0), ls1=x.stride(1)))
                    c0 = counts(canon)
                    run()
                    torch.cuda.synchronize()
                    one = {k: v - c0[k] for k, v in counts(canon).items()}
                    ok = bool(torch.equal(o, ref))
                    g = graph_of(run)
                    us, trials = device_us(g.replay, 5)
                    rec[name] = {"ok": ok, "one_call": one, "us": us - t_copy, "trials_us": trials}
                    del g
                emit(rec)
    emit({"done_s": time.time() - t0})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
