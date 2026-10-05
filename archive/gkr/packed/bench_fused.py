"""Measure the round-fused packed sumcheck (lane a-fusion) against the a-packed2 per-round kernels on one GPU.

Whole sumchecks (all rounds, host round trip per round for the coin, coins drawn ahead) with ``round_fusion`` on and
off, CUDA events around the whole proof (``device_ms``) and wall time; the fused kernels alone in CUDA graphs
(U pass vs the three packed rounds; ``ext_fold_ip`` vs ``ext_ip + 2 fold_ext``); the HBM traffic floor of each variant
(every table element read once per pass and written once per fold, at the datasheet bandwidth) and the INT8-peak
fraction of the limb MACs.  Shapes: ``--log2`` table sizes for ``--ks``; the v1 / v2 layer models come from
:mod:`bench` (``layer_model``) and :mod:`logup_graph_bench` (``V2_SHAPES``).

Pod::

    /workspace/venv312/bin/python -m archive.gkr.packed.bench_fused --log2 26 28 --ks 3 --out <dir>
"""
from __future__ import annotations

# ruff: noqa: B023 F821  -- timing lambdas are called inside the iteration that builds them; the later ``del`` (memory)
# makes ruff misread the names as undefined.
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from packed import bench as B
from packed import reference as ref
from packed import sumcheck_packed as sp

INT4 = 4
EXT_BYTES = 24


def traffic_floor_bytes(n: int, k: int, fusion: bool) -> dict:
    """Mandatory HBM traffic of the algorithm as implemented (weights, partial sums and messages are negligible)."""
    t = 1 << n
    if fusion and k:
        packed = 2 * INT4 * t                                     # U pass: g, h read once
    else:
        packed = sum(2 * INT4 * t for _ in range(1, k + 1))       # each packed round re-reads g, h
    fold_k = (2 * INT4 * t + 2 * EXT_BYTES * (t >> k)) if k else 0
    ext = 0
    m0 = t >> k if k else t
    # extension rounds on tables of m elements (pairs m / 2): round k+1 reads 2 m ext elements
    m = m0
    first = True
    while m >= 2:
        if fusion and not first:
            # fold(read 2 m . 2 pre-fold) + write folded 2 m + Hadamard from registers
            ext += 2 * EXT_BYTES * (2 * m) + 2 * EXT_BYTES * m
        else:
            ext += 2 * EXT_BYTES * m                              # Hadamard pass reads both tables
            if m > 2:
                ext += 2 * EXT_BYTES * m + 2 * EXT_BYTES * (m // 2)   # fold: read both, write both halves
        m //= 2
        first = False
    return {"packed": packed, "fold_k": fold_k, "ext": ext, "total": packed + fold_k + ext}


def _whole_sumcheck(kernel: sp.Kernel, n: int, k: int, repeats: int, warm: int, seed: int) -> dict:
    import torch

    dev = kernel.device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    t = 1 << n
    g = torch.randint(0, ref.P, (t,), dtype=torch.int64, generator=gen).to(torch.int32).to(dev)
    h = torch.randint(0, ref.P, (t,), dtype=torch.int64, generator=gen).to(torch.int32).to(dev)
    rng = random.Random(seed + 1)
    gammas = [ref.random_ext(rng) for _ in range(n)]
    coins = [ref.random_ext(rng) for _ in range(n)]

    inputs = sp.prepare_inputs(kernel, gammas, coins, k)      # host work (eq tables, fold operands) outside the timing
    st = sp.Stats()
    torch.cuda.reset_peak_memory_stats(dev)
    sp.prove_device(kernel, gammas, coins, g, h, k, inputs, st)
    torch.cuda.synchronize()
    # eager: the whole device pipeline launched from Python (launch gaps included), CUDA events + wall clock
    for _ in range(warm):
        sp.prove_device(kernel, gammas, coins, g, h, k, inputs)
    torch.cuda.synchronize()
    dev_ms, wall_ms = [], []
    for _ in range(repeats):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        a.record()
        sp.prove_device(kernel, gammas, coins, g, h, k, inputs)
        b.record()
        torch.cuda.synchronize()
        wall_ms.append((time.perf_counter() - t0) * 1e3)
        dev_ms.append(a.elapsed_time(b))
    # graphed: the whole sumcheck as one CUDA graph (the number a-gpu's per-layer graph would see)
    tg, how = B.time_graphed(lambda: sp.prove_device(kernel, gammas, coins, g, h, k, inputs), repeats, warm)
    mem = torch.cuda.max_memory_allocated(dev)
    del g, h
    torch.cuda.empty_cache()
    labels = {}
    for lab, M, K, N in st.matmuls:
        labels[lab] = labels.get(lab, 0) + 1
    return {"device_ms": tg * 1e3, "timing": how, "eager_device_ms": statistics.median(dev_ms), "eager_wall_ms": statistics.median(wall_ms),
            "macs_performed": st.macs_performed, "hadamard_base": st.hadamard_base, "hadamard_ext": st.hadamard_ext, "kernels": labels,
            "mem.peak_device_bytes": mem}


def _kernel_pieces(kernel: sp.Kernel, n: int, k: int, repeats: int, warm: int, seed: int) -> dict:
    """The fused kernels alone (graphed) vs their a-packed2 equivalents at the same shapes."""
    import torch

    dev = kernel.device
    tk = kernel._tk
    gen = torch.Generator(device="cpu").manual_seed(seed)
    t = 1 << n
    g = torch.randint(0, ref.P, (t,), dtype=torch.int64, generator=gen).to(torch.int32).to(dev)
    h = torch.randint(0, ref.P, (t,), dtype=torch.int64, generator=gen).to(torch.int32).to(dev)
    rng = random.Random(seed + 1)
    gammas = [ref.random_ext(rng) for _ in range(n)]
    out = {}
    if k:
        inp_k = kernel.inputs(gammas, [], k, k)
        K1, K2 = kernel.split(t >> k)
        split = kernel.split_b1(K1, K2)
        inp_k.e_lo32 = torch.nn.functional.pad(inp_k.e_lo_limbs, (0, 32 - B.DL))
        tm, _ = B.time_graphed(lambda: tk.packed_u_inner_product_gen(g, h, inp_k.e_lo32, k, K1, K2, split), repeats, warm)
        out["u_pass_ms"] = tm * 1e3
        tot = 0.0
        rs: list = []
        for i in range(1, k + 1):
            inp = kernel.inputs(gammas, rs, i, k)
            K1i, K2i = kernel.split(t >> i)
            spl = kernel.split_b1(K1i, K2i)
            inp.e_lo32 = torch.nn.functional.pad(inp.e_lo_limbs, (0, 32 - B.DL))
            ti, _ = B.time_graphed(lambda: tk.packed_inner_product(g, h, inp.e_lo32, i, K1i, K2i, spl), repeats, warm)
            tot += ti
            rs.append(ref.random_ext(rng))
        out["packed_rounds_ms"] = tot * 1e3
        out["u_pass_bytes"] = 2 * INT4 * t
        out["u_pass_hbm_floor_ms"] = 2 * INT4 * t / 3.35e12 * 1e3
    # extension: pre-fold tables of 2 m pairs -> fold + Hadamard of m pairs
    m_pairs = (t >> (k + 2)) if k else (t >> 2)     # pairs of the round after the first extension round
    gE = torch.randint(0, ref.P, (4 * m_pairs, 6), dtype=torch.int64, generator=gen).to(torch.int32).to(dev)
    hE = torch.randint(0, ref.P, (4 * m_pairs, 6), dtype=torch.int64, generator=gen).to(torch.int32).to(dev)
    del g, h
    K1, K2 = kernel.split(m_pairs)
    split = kernel.split_b1(K1, K2)
    i = (k + 2) if k else 2
    inp = kernel.inputs(gammas, [ref.random_ext(rng) for _ in range(i - 1)], i, k, ref.random_ext(rng))
    inp.e_lo32 = torch.nn.functional.pad(inp.e_lo_limbs, (0, 32 - B.DL))
    r_R = inp.fold_r
    tm, _ = B.time_graphed(lambda: tk.ext_fold_inner_product(gE, hE, r_R, inp.e_lo32, K1, K2, split), repeats, warm)
    out["ext_fold_ip_ms"] = tm * 1e3
    gF = tk.fold_ext(gE.view(2 * m_pairs, 12), r_R)
    hF = tk.fold_ext(hE.view(2 * m_pairs, 12), r_R)
    t1, _ = B.time_graphed(lambda: (tk.fold_ext(gE.view(2 * m_pairs, 12), r_R), tk.fold_ext(hE.view(2 * m_pairs, 12), r_R)), repeats, warm)
    t2, _ = B.time_graphed(lambda: tk.ext_inner_product(gF, hF, inp.e_lo32, K1, K2, split), repeats, warm)
    out["fold_ext_x2_ms"] = t1 * 1e3
    out["ext_ip_ms"] = t2 * 1e3
    out["ext_pairs"] = m_pairs
    out["ext_fused_bytes"] = 2 * EXT_BYTES * (4 * m_pairs) + 2 * EXT_BYTES * (2 * m_pairs)
    out["ext_fused_hbm_floor_ms"] = out["ext_fused_bytes"] / 3.35e12 * 1e3
    out["ext_unfused_bytes"] = 2 * EXT_BYTES * (4 * m_pairs) + 2 * EXT_BYTES * (2 * m_pairs) + 2 * EXT_BYTES * (2 * m_pairs)
    del gE, hE, gF, hF
    torch.cuda.empty_cache()
    return out


def micro(n: int, k: int, repeats: int, warm: int, k1_log2: int) -> dict:
    """Configuration sweep of the fused kernels alone at the ``(2^n, k)`` shape (graphed medians, ms)."""
    import torch

    from packed import kernels_fused as kf

    kernel = sp.Kernel("cuda", k1_log2=k1_log2, mode="triton")
    tk = kernel._tk
    dev = kernel.device
    gen = torch.Generator(device="cpu").manual_seed(5)
    t = 1 << n
    g = torch.randint(0, ref.P, (t,), dtype=torch.int64, generator=gen).to(torch.int32).to(dev)
    h = torch.randint(0, ref.P, (t,), dtype=torch.int64, generator=gen).to(torch.int32).to(dev)
    rng = random.Random(6)
    gammas = [ref.random_ext(rng) for _ in range(n)]
    out: dict = {}
    inp_k = kernel.inputs(gammas, [], k, k)
    K1, K2 = kernel.split(t >> k)
    inp_k.e_lo32 = torch.nn.functional.pad(inp_k.e_lo_limbs, (0, 32 - B.DL))
    for name, fn in (("u_tensor", kf.packed_u_inner_product), ("u_gen", tk.packed_u_inner_product_gen)):
        for bk in (64, 128, 256):
            for w in (4, 8):
                split = 1
                while K2 * split < 264 and K1 % (split * 2 * bk) == 0:
                    split *= 2
                if K1 % (split * bk):
                    continue
                try:
                    tm, _ = B.time_graphed(lambda: fn(g, h, inp_k.e_lo32, k, K1, K2, split, BLOCK_K=bk, num_warps=w), repeats, warm)
                    out[f"{name}.bk{bk}.w{w}"] = tm * 1e3
                except Exception as e:  # noqa: BLE001
                    out[f"{name}.bk{bk}.w{w}"] = f"error {type(e).__name__}"
                print(name, bk, w, out[f"{name}.bk{bk}.w{w}"], flush=True)
    del g, h
    m_pairs = t >> (k + 2)
    gE = torch.randint(0, ref.P, (4 * m_pairs, 6), dtype=torch.int64, generator=gen).to(torch.int32).to(dev)
    hE = torch.randint(0, ref.P, (4 * m_pairs, 6), dtype=torch.int64, generator=gen).to(torch.int32).to(dev)
    K1, K2 = kernel.split(m_pairs)
    i = k + 2
    inp = kernel.inputs(gammas, [ref.random_ext(rng) for _ in range(i - 1)], i, k, ref.random_ext(rng))
    inp.e_lo32 = torch.nn.functional.pad(inp.e_lo_limbs, (0, 32 - B.DL))
    r_R = inp.fold_r
    for bk in (32, 64, 128):
        for w in (2, 4, 8):
            split = 1
            while K2 * split < 264 and K1 % (split * 2 * bk) == 0:
                split *= 2
            try:
                tm, _ = B.time_graphed(lambda: tk.ext_fold_inner_product(gE, hE, r_R, inp.e_lo32, K1, K2, split, BLOCK_K=bk, num_warps=w), repeats, warm)
                out[f"ext_fold_ip.bk{bk}.w{w}"] = tm * 1e3
            except Exception as e:  # noqa: BLE001
                out[f"ext_fold_ip.bk{bk}.w{w}"] = f"error {type(e).__name__}"
            print("ext_fold_ip", bk, w, out[f"ext_fold_ip.bk{bk}.w{w}"], flush=True)
    return out


def main(argv=None) -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--micro", action="store_true", help="configuration sweep of the fused kernels alone")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log2", type=int, nargs="*", default=[26])
    ap.add_argument("--ks", type=int, nargs="*", default=[3])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--k1-log2", type=int, default=sp.K1_LOG2_DEFAULT)
    ap.add_argument("--no-pieces", dest="pieces", action="store_false")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.micro:
        res = {f"2^{n}.k{k}": micro(n, k, a.repeats, a.warm, a.k1_log2) for n in a.log2 for k in a.ks}
        (out / "micro.json").write_text(json.dumps(res, indent=1) + "\n")
        return 0
    hw = B.probe_hardware()
    peaks = B.GPU_PEAKS[hw["gpu"]["name"]]
    peak_macs = peaks["int8_dense_tops"] * 1e12 / 2
    bw = peaks["hbm_bytes_per_s"]
    results = []
    t0 = time.time()
    for n in a.log2:
        for k in a.ks:
            row = {"log2_t": n, "k": k}
            for fusion in (False, True):
                ker = sp.Kernel("cuda", k1_log2=a.k1_log2, mode="triton", round_fusion=fusion)
                r = _whole_sumcheck(ker, n, k, a.repeats, a.warm, seed=11 + n)
                floor = traffic_floor_bytes(n, k, fusion)
                r["hbm_floor_ms"] = floor["total"] / bw * 1e3
                r["hbm_floor_bytes"] = floor
                r["x_hbm_floor"] = r["device_ms"] / r["hbm_floor_ms"]
                r["frac_int8_peak"] = r["macs_performed"] / (r["device_ms"] / 1e3) / peak_macs
                row["fused" if fusion else "baseline"] = r
                print(f"t=2^{n} k={k} fusion={fusion}: graphed {r['device_ms']:.3f} ms ({r['timing']}) eager dev {r['eager_device_ms']:.3f} "
                      f"wall {r['eager_wall_ms']:.3f} ms  "
                      f"floor {r['hbm_floor_ms']:.3f} ms ({r['x_hbm_floor']:.2f}x)  INT8 {r['frac_int8_peak']*100:.2f}%  "
                      f"mem {r['mem.peak_device_bytes']/1e9:.2f} GB [{time.time()-t0:.0f}s]", flush=True)
            if a.pieces:
                ker = sp.Kernel("cuda", k1_log2=a.k1_log2, mode="triton", round_fusion=True)
                row["pieces"] = _kernel_pieces(ker, n, k, a.repeats, a.warm, seed=11 + n)
                print(f"  pieces: {json.dumps(row['pieces'])}", flush=True)
            results.append(row)
            (out / "bench_fused.json").write_text(json.dumps({"hardware": hw, "peaks": peaks, "torch": torch.__version__,
                                                               "commit": B.git_commit(), "config": vars(a), "results": results},
                                                              indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
