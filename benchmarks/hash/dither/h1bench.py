"""H1 + fx1's noise floor on one H200 or B200: the per-unit SHA-512 noise every H1 unit derives (h1tile.TileShape's `blocks`
from seed_j and `lanes` key blocks from sigma_R), timed interleaved with the plain int8 GEMM of the same call, at
Llama-3.1-8B's linears and M = 1, 32, 256, 2048, 8192. Whatever the GEMM and fold cost, an H1 kernel on these cards costs
at least this noise, so 1 + noise / plain is a lower bound on its gamma.

    python h1bench.py --lib h1noise.so --out OUT [--reps 5] [--ms ...] [--linears ...] [--js 1,4]

Units are the call's full 32 x 32 tiles (`served_h1/receipts.units`); below 32 rows a call has none, so M = 1 is counted
as one padded row tile (an assumption: the padding rule for decode is open). Every unit is counted live (the
dead-unit share is the credit's to count). The 4 SHA-512 compressions a unit spends on seed_j and sigma_R themselves are
not timed.

Arms (each one CUDA graph; noise arms launch the seed_j stream then the sigma_R stream):
  ca-sink, ca-write     served_h1/noise.cu's kernel, hashing only (one XOR word a block) / writing the noise
  mid-sink-jJ, mid-write    h1noise.cu's midstate kernel, J blocks a lane
  int_mm, cutlass       the plain call: torch._int_mm (17-row padding below 17) and vLLM's cutlass_scaled_mm

Gates before any timing (a failed gate exits 4 and times nothing): ca-write's noise against hashlib on sampled blocks;
mid's write equal to ca-write's on every block; each sink equal to the XOR fold of ca-write's; 0xA5 poison before each run and a
no-write control (a launch over zero units) that must be rejected.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from hb_kbench import LINEARS, MS, POISON, Clock, capture, inputs, one_call_ms, w_seed, x_seed

from verity.core.protocols.pouw.circuit import h1tile as HT

NOISE_TAG = b"verity/pouw/noise/v1".ljust(24, b"\0")
KEY_BLOCKS = 32
WRITE_CAP = 4 << 30


class Lib:
    def __init__(self, path: str):
        h = self.h = ctypes.CDLL(path)
        h.kn_ca.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        h.kn_mid.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.c_int,
                             ctypes.c_void_p]
        h.kn_mid_items.restype = ctypes.c_ulonglong
        h.kn_mid_items.argtypes = [ctypes.c_uint] * 3

    def ca(self, seeds, S, nb, out, sink):
        if self.h.kn_ca(seeds.data_ptr(), S, nb, out.data_ptr(), int(sink), torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("kn_ca")

    def mid(self, seeds, S, nb, J, out, sink):
        if self.h.kn_mid(seeds.data_ptr(), S, nb, J, out.data_ptr(), int(sink), torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("kn_mid")

    def items(self, S, nb, J) -> int:
        return int(self.h.kn_mid_items(S, nb, J))


def rand_seeds(S: int, gen: torch.Generator) -> torch.Tensor:
    return torch.randint(0, 256, (S, 64), dtype=torch.uint8, device="cuda", generator=gen).view(torch.int64)


def seed_bytes(row) -> bytes:
    return b"".join((int(v) & ((1 << 64) - 1)).to_bytes(8, "big") for v in row)


def xor_halves(blocks: torch.Tensor) -> torch.Tensor:
    """(..., 16) int32 halves -> (...) their XOR."""
    x = blocks
    while x.shape[-1] > 1:
        h = x.shape[-1] // 2
        x = x[..., :h] ^ x[..., h:]
    return x[..., 0]


def mid_fold(per_block: torch.Tensor, S: int, nb: int, J: int) -> torch.Tensor:
    """The ca kernel's per-block XOR words (S, nb) folded as mid's sink: per (item, lane), the XOR over its J blocks."""
    per = 32 * J
    chunks = -(-nb // per)
    pad = torch.zeros((S, chunks * per), dtype=torch.int32, device=per_block.device)
    pad[:, :nb] = per_block
    v = pad.view(S, chunks, J, 32)
    acc = v[:, :, 0]
    for j in range(1, J):
        acc = acc ^ v[:, :, j]
    return acc.reshape(-1)


def gate(lib: Lib, nb: int, js: list[int], gen: torch.Generator, samples: int = 48) -> dict:
    S = 40
    seeds = rand_seeds(S, gen)
    out_ca = torch.empty((S, nb, 16), dtype=torch.int32, device="cuda")
    res: dict = {"S": S, "nb": nb}
    out_ca.fill_(POISON)
    lib.ca(seeds, S, nb, out_ca, False)
    torch.cuda.synchronize()
    host = seeds.cpu().numpy()
    g = np.random.default_rng(nb)
    bad = 0
    picks = [(0, 0), (S - 1, nb - 1)] + [(int(g.integers(S)), int(g.integers(nb))) for _ in range(samples)]
    for u, t in picks:
        d = hashlib.sha512(NOISE_TAG + seed_bytes(host[u]) + t.to_bytes(8, "big")).digest()
        want = np.frombuffer(d, dtype=">u4").astype(np.uint32)
        got = out_ca[u, t].cpu().numpy().view(np.uint32)
        bad += int((want != got).sum())
    res["ca_vs_hashlib_bad_words"] = bad
    per_block = xor_halves(out_ca)
    sink_ca = torch.full((S * nb,), POISON, dtype=torch.int32, device="cuda")
    lib.ca(seeds, S, nb, sink_ca, True)
    torch.cuda.synchronize()
    res["ca_sink_bad"] = int((sink_ca != per_block.reshape(-1)).sum())
    out_mid = torch.full_like(out_ca, POISON)
    lib.mid(seeds, S, nb, js[0], out_mid, False)
    torch.cuda.synchronize()
    res["mid_write_bad"] = int((out_mid != out_ca).sum())
    res["mid_sink_bad"] = {}
    for J in js:
        sink = torch.full((lib.items(S, nb, J) * 32,), POISON, dtype=torch.int32, device="cuda")
        lib.mid(seeds, S, nb, J, sink, True)
        torch.cuda.synchronize()
        res["mid_sink_bad"][J] = int((sink != mid_fold(per_block, S, nb, J)).sum())
    ctl = torch.full_like(out_ca, POISON)
    lib.mid(seeds, 0, nb, js[0], ctl, False)
    lib.ca(seeds, 0, nb, ctl, False)
    torch.cuda.synchronize()
    res["control_bad"] = int((ctl != out_ca).sum())
    res["control_rejected"] = res["control_bad"] > 0
    res["ok"] = (bad == 0 and res["ca_sink_bad"] == 0 and res["mid_write_bad"] == 0
                 and all(v == 0 for v in res["mid_sink_bad"].values()) and res["control_rejected"])
    return res


def plain_arms(name: str, n: int, k: int, m: int, Xh: np.ndarray, l2: int) -> tuple[dict, dict]:
    dev = torch.device("cuda")
    W = torch.from_numpy(inputs(w_seed(name), n, k)).to(dev)
    nbuf = max(1, min(32, math.ceil(2.5 * l2 / (n * k))))
    Ws = [W] + [W.clone() for _ in range(nbuf - 1)]
    X = torch.from_numpy(Xh[:m]).to(dev)
    Xmm = X if m >= 17 else torch.cat([X, X.new_zeros(17 - m, k)])
    Cmm = torch.empty((Xmm.shape[0], n), dtype=torch.int32, device=dev)
    arms, info = {}, {"weight_copies": nbuf}
    arms["int_mm"] = lambda i: torch._int_mm(Xmm, Ws[i % nbuf].t(), out=Cmm)
    try:
        import vllm._custom_ops  # noqa: F401
        sa = torch.full((m, 1), 1e-3, dtype=torch.float32, device=dev)
        sw = torch.full((n, 1), 1e-3, dtype=torch.float32, device=dev)
        O = torch.empty((m, n), dtype=torch.bfloat16, device=dev)

        def cut(i):
            torch.ops._C.cutlass_scaled_mm(O, X, Ws[i % nbuf].t(), sa, sw, None)
        cut(0)
        torch.cuda.synchronize()
        arms["cutlass"] = cut
    except Exception as e:  # noqa: BLE001 - a refusal is a result
        info["cutlass_refusal"] = "".join(traceback.format_exception_only(type(e), e)).strip()[-1500:]
    return arms, info


def auto_j(S: int, nb: int, sms: int) -> int:
    """The largest J that still leaves about 64 warps an SM."""
    return max(1, min(32, (S * nb) // (32 * sms * 64)))


def bench(lib: Lib, args, name: str, n: int, k: int, m: int, Xh: np.ndarray, clock: Clock, l2: int, sms: int,
          gen: torch.Generator) -> dict:
    lay = HT.TileShape(32, 32, k)
    S = max(1, m // 32) * (n // 32)
    nb = lay.blocks
    out: dict = {"linear": name, "m": m, "n": n, "k": k, "units": S, "noise_blocks": nb, "key_blocks": KEY_BLOCKS,
                 "padded_row_block": m < 32, "arms": {}}
    sj, sr = rand_seeds(S, gen), rand_seeds(S, gen)
    arms: dict = {}
    ca_sink = torch.empty((S * nb,), dtype=torch.int32, device="cuda")
    ca_ksink = torch.empty((S * KEY_BLOCKS,), dtype=torch.int32, device="cuda")
    arms["ca-sink"] = lambda i: (lib.ca(sj, S, nb, ca_sink, True), lib.ca(sr, S, KEY_BLOCKS, ca_ksink, True))
    js = sorted({*args.js, auto_j(S, nb, sms)})
    out["auto_j"] = auto_j(S, nb, sms)
    for J in js:
        snk = torch.empty((lib.items(S, nb, J) * 32,), dtype=torch.int32, device="cuda")
        ksnk = torch.empty((lib.items(S, KEY_BLOCKS, 1) * 32,), dtype=torch.int32, device="cuda")
        arms[f"mid-sink-j{J}"] = (lambda i, J=J, snk=snk, ksnk=ksnk:
                                  (lib.mid(sj, S, nb, J, snk, True), lib.mid(sr, S, KEY_BLOCKS, 1, ksnk, True)))
    wbytes = S * (nb + KEY_BLOCKS) * 64
    if wbytes <= WRITE_CAP:
        wo = torch.empty((S, nb, 16), dtype=torch.int32, device="cuda")
        wk = torch.empty((S, KEY_BLOCKS, 16), dtype=torch.int32, device="cuda")
        arms["ca-write"] = lambda i: (lib.ca(sj, S, nb, wo, False), lib.ca(sr, S, KEY_BLOCKS, wk, False))
        jw = out["auto_j"]
        arms["mid-write"] = lambda i: (lib.mid(sj, S, nb, jw, wo, False), lib.mid(sr, S, KEY_BLOCKS, 1, wk, False))
    else:
        out["write_skipped"] = f"{wbytes} bytes of noise > {WRITE_CAP}"
    parms, info = plain_arms(name, n, k, m, Xh, l2)
    out.update(info)
    arms.update(parms)
    graphs, calls = {}, {}
    for a, fn in arms.items():
        t = one_call_ms(fn)
        G = max(1, min(400, round(args.item_ms / max(t, 1e-3))))
        graphs[a], calls[a] = capture(fn, G), G
    for gr in graphs.values():
        gr.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    per = {a: [] for a in graphs}
    mhz = {a: [] for a in graphs}
    order = list(graphs)
    for rep in range(args.reps):
        for a in order[rep % len(order):] + order[:rep % len(order)]:
            e0.record()
            graphs[a].replay()
            e1.record()
            e1.synchronize()
            per[a].append(e0.elapsed_time(e1) * 1e3 / calls[a])
            mhz[a].append(clock.sm())
    for a in graphs:
        out["arms"][a] = {"calls_per_item": calls[a], "us": [round(v, 3) for v in per[a]],
                          "median_us": round(statistics.median(per[a]), 3), "sm_mhz": mhz[a]}
    med = {a: v["median_us"] for a, v in out["arms"].items()}
    plain = {b: med[b] for b in ("int_mm", "cutlass") if b in med}
    fp = min(plain, key=plain.get)
    sinks = {a: v for a, v in med.items() if "sink" in a}
    best = min(sinks, key=sinks.get)
    blocks = S * (nb + KEY_BLOCKS)
    out.update({"fastest_plain": fp, "plain_us": plain[fp], "best_noise": best, "noise_us": sinks[best],
                "noise_over_plain": round(sinks[best] / plain[fp], 4), "gamma_floor": round(1 + sinks[best] / plain[fp], 4),
                "gcomp_per_s": {a: round(blocks / (v * 1e3), 3) for a, v in sinks.items()},
                "ratios": {f"{a}/{b}": round(med[a] / v, 4) for a in med if a not in plain for b, v in plain.items()}})
    del graphs
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--item-ms", type=float, default=3.0)
    ap.add_argument("--ms", default=",".join(map(str, MS)))
    ap.add_argument("--linears", default=",".join(LINEARS))
    ap.add_argument("--js", default="1,4")
    args = ap.parse_args()
    args.js = [int(v) for v in args.js.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lib = Lib(args.lib)
    clock = Clock()
    props = torch.cuda.get_device_properties(0)
    l2 = int(getattr(props, "L2_cache_size", 50 << 20))
    gen = torch.Generator(device="cuda")
    gen.manual_seed(0x4831)
    doc = {"schema": "dither-hb/h1noise/v1", "device": clock.name, "sms": props.multi_processor_count, "l2_bytes": l2,
           "torch": torch.__version__, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gates": [],
           "shapes": []}
    for k in sorted({LINEARS[n][1] for n in args.linears.split(",")}):
        for nb in (HT.TileShape(32, 32, k).blocks, KEY_BLOCKS):
            g = gate(lib, nb, sorted({*args.js, 2, 9}), gen)
            doc["gates"].append(g)
            print(json.dumps(g), flush=True)
    (out / "h1noise.json").write_text(json.dumps(doc, indent=1))
    if not all(g["ok"] for g in doc["gates"]):
        print("h1bench: a gate failed; nothing timed", file=sys.stderr)
        return 4
    xs: dict = {}
    for name in args.linears.split(","):
        n, k = LINEARS[name]
        if k not in xs:
            xs[k] = inputs(x_seed(k), max(int(v) for v in args.ms.split(",")), k)
        for m in (int(v) for v in args.ms.split(",")):
            t0 = time.time()
            try:
                r = bench(lib, args, name, n, k, m, xs[k], clock, l2, props.multi_processor_count, gen)
            except Exception as e:  # noqa: BLE001
                r = {"linear": name, "m": m, "error": "".join(traceback.format_exception(e))[-3000:]}
            r["wall_s"] = round(time.time() - t0, 2)
            doc["shapes"].append(r)
            print(json.dumps({x: r.get(x) for x in ("linear", "m", "units", "noise_us", "plain_us", "gamma_floor",
                                                     "best_noise", "gcomp_per_s", "error", "wall_s")}), flush=True)
            (out / "h1noise.json").write_text(json.dumps(doc, indent=1))
    layer: dict = {}
    for r in doc["shapes"]:
        if "noise_us" in r:
            d = layer.setdefault(r["m"], {"noise_us": 0.0, "plain_us": 0.0, "linears": []})
            d["noise_us"] += r["noise_us"]
            d["plain_us"] += r["plain_us"]
            d["linears"].append(r["linear"])
    for d in layer.values():
        d["gamma_floor"] = round(1 + d["noise_us"] / d["plain_us"], 4)
    doc["per_layer"] = layer
    doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "h1noise.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({"per_layer": layer}))
    return 0 if all("error" not in r for r in doc["shapes"]) else 4


if __name__ == "__main__":
    sys.exit(main())
