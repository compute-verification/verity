"""M2-E's cheap test: are trained open weights, quantized per-channel symmetric int8, already far from a cheap exact multiply?

For each sampled projection q (n x K, int8 per output channel), it searches splits q = L + S_small + S_out:

- L real of rank r, from the SVD of q (and two robust-PCA refinements that refit L with S_out's entries masked);
- S_small the rounded residual's entries in [-a, a], priced by the cheapest dense, b1-plane, FP4 or 2:4 route;
- S_out the rest, priced on the ALU (IDP4A), on a 2:4 pass, or dense.

Prices are `prices.py`'s W1 units against a b-bit Z (x = 0 leaves the noise's b bits). The search is generous to the
cheater: L is real (an exact split needs an integer L, which only costs more), and its MACs are priced at 1. It also
prices r = 0 routes per 128 x 128 tile (a GEMM picks a route per tile), and states the Eckart-Young floor: any L of rank
r leaves ||q - L||_F >= the SVD tail, so a residual in [-a, a] everywhere needs rank >= r_min(a).

The low bits get: P(odd) by magnitude, neighbour correlations, lzma ratio, GF(2) rank, the +-1 plane's top singular
value against sqrt(n) + sqrt(K), and the low two bits' spread. A row-shuffled copy of q is the i.i.d. control.

Weights are range-read from Hugging Face (only the sampled tensors), nothing is written but `--out`.
"""

import argparse
import concurrent.futures
import json
import lzma
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prices import CHIPS, ZBITS, alu_part, dense_part, lowrank_part, sparse24_part  # noqa: E402

MODELS = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "llama3.2-1b": "unsloth/Llama-3.2-1B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "llama3.1-8b": "unsloth/Meta-Llama-3.1-8B",
}
PROJ = [("self_attn", p) for p in ("q_proj", "k_proj", "v_proj", "o_proj")] + [
    ("mlp", p) for p in ("gate_proj", "up_proj", "down_proj")
]
ALPHABET = (0, 1, 2, 3, 4, 7, 15, 36)
TILE = 128
EVAL_ROWS = 4096
HF = "https://huggingface.co"


def fetch(url, start=None, end=None):
    headers = {"Range": f"bytes={start}-{end - 1}"} if start is not None else {}
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as f:  # noqa: S310  (HF's https URL)
                return f.read()
        except urllib.error.HTTPError as e:
            if e.code == 404 or attempt == 4:
                raise
            time.sleep(2 ** attempt)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def safetensors_header(url):
    n = int.from_bytes(fetch(url, 0, 8), "little")
    return 8 + n, json.loads(fetch(url, 8, 8 + n))


def load_tensor(url, base, entry):
    b, e = entry["data_offsets"]
    raw = fetch(url, base + b, base + e)
    dt = entry["dtype"]
    if dt == "BF16":
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        a = u.view(np.float32)
    elif dt == "F16":
        a = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif dt == "F32":
        a = np.frombuffer(raw, dtype=np.float32).copy()
    else:
        raise ValueError(f"dtype {dt}")
    return a.reshape(entry["shape"])


def quantize(w):
    s = np.abs(w).max(axis=1, keepdims=True) / 127.0
    s[s == 0] = 1.0
    return np.clip(np.rint(w / s), -127, 127).astype(np.int16)


def gf2_rank(bits):
    rows = np.packbits(bits.astype(np.uint8), axis=1)
    m, ncol = bits.shape
    rank = 0
    for col in range(ncol):
        byte, mask = col >> 3, np.uint8(0x80 >> (col & 7))
        piv = np.nonzero(rows[rank:, byte] & mask)[0]
        if piv.size == 0:
            continue
        p = rank + piv[0]
        if p != rank:
            rows[[rank, p]] = rows[[p, rank]]
        hit = np.nonzero(rows[:, byte] & mask)[0]
        hit = hit[hit != rank]
        rows[hit] ^= rows[rank]
        rank += 1
        if rank == m:
            break
    return rank


def alphabet_stats(q):
    a = np.abs(q.astype(np.int32))
    hist = np.bincount((q.astype(np.int32) + 127).ravel(), minlength=255)
    p = hist / hist.sum()
    nz = p[p > 0]
    return {
        "sigma_q": float(q.astype(np.float64).std()),
        "mean_abs_q": float(a.mean()),
        "entropy_bits": float(-(nz * np.log2(nz)).sum()),
        "frac_abs_le": {str(k): float((a <= k).mean()) for k in ALPHABET},
        "row_sigma_q_quantiles": [float(x) for x in np.quantile(q.astype(np.float64).std(axis=1), [0.0, 0.1, 0.5, 0.9, 1.0])],
        "hist": hist.tolist(),
    }


def lsb_stats(q):
    qi = q.astype(np.int32)
    lsb = (qi & 1).astype(np.float64)
    a = np.abs(qi)
    out = {"p_odd": float(lsb.mean())}
    for lo, hi in ((1, 4), (4, 8), (8, 16), (16, 32), (32, 128)):
        m = (a >= lo) & (a < hi)
        out[f"p_odd_abs_{lo}_{hi}"] = float(lsb[m].mean()) if m.any() else None
    big = a >= 8
    for axis, name in ((1, "row"), (0, "col")):
        x = lsb[:, :-1] if axis == 1 else lsb[:-1, :]
        y = lsb[:, 1:] if axis == 1 else lsb[1:, :]
        out[f"corr_{name}_neighbour"] = float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
        bx = big[:, :-1] & big[:, 1:] if axis == 1 else big[:-1, :] & big[1:, :]
        out[f"corr_{name}_neighbour_abs_ge8"] = float(np.corrcoef(x[bx], y[bx])[0, 1]) if bx.sum() > 1000 else None
    flat = (qi & 1).astype(np.uint8).ravel()[: 1 << 23]
    packed = np.packbits(flat).tobytes()
    out["lzma_ratio"] = len(lzma.compress(packed, preset=6)) / len(packed)
    low2 = np.bincount((qi & 3).ravel(), minlength=4) / qi.size
    out["low2_freq"] = low2.tolist()
    out["low2_max_dev"] = float(np.abs(low2 - 0.25).max())
    m = min(1024, q.shape[0], q.shape[1])
    out["gf2_rank"] = [int(gf2_rank((qi[:m, :m] & 1))), m]
    pm = torch.from_numpy((2 * lsb - 1).astype(np.float32))
    s = torch.svd_lowrank(pm, q=6, niter=3)[1]
    out["pm1_top_sv_over_rand"] = float(s[0] / (math.sqrt(q.shape[0]) + math.sqrt(q.shape[1])))
    row_bias = lsb.mean(axis=1)
    out["row_p_odd_max_z"] = float(np.abs(row_bias - lsb.mean()).max() / math.sqrt(0.25 / q.shape[1]))
    return out


def rank_grid(m):
    return sorted({r for r in (0, 1, 8, 32, 128, m // 16, m // 8, m // 4, m // 2) if r < m})


def group_excess(nz, k):
    """Entries beyond two in each group of four along K, as a fraction of all entries; per row group counts."""
    cnt = nz.reshape(nz.shape[0], k // 4, 4).sum(axis=2)
    return np.maximum(cnt - 2, 0)


def split_costs(qs, L, n, k, r):
    """Every (a, chip, b) cost of qs = L + S_small + S_out on the evaluated rows."""
    ri = np.rint(qs - L).astype(np.int32)
    ab = np.abs(ri)
    res = []
    lr = lowrank_part(r, n, k)
    for a in ALPHABET:
        small = ab <= a
        out = ~small
        d_out = float(out.mean())
        ex_small = float(group_excess(small & (ri != 0), k).sum() / ri.size)
        ex_out = float(group_excess(out, k).sum() / ri.size)
        max_out = int(ab[out].max()) if d_out > 0 else 0
        for chip in CHIPS:
            for b in ZBITS:
                c_small = min(dense_part(chip, a, b), sparse24_part(chip, a, b, ex_small)) if a > 0 else 0.0
                if d_out == 0:
                    c_out = 0.0
                else:
                    c_out = min(alu_part(chip, d_out), dense_part(chip, max_out, b),
                                sparse24_part(chip, max_out, b, ex_out))
                res.append({"r": r, "a": a, "chip": chip, "b": b, "cost": lr + c_small + c_out, "lowrank": lr,
                            "small": c_small, "out": c_out, "d_out": d_out})
    return res


def tiled_r0(q):
    """r = 0 routes chosen per 128 x 128 tile: the mean over tiles of each tile's cheapest route."""
    n, k = q.shape
    qi = q.astype(np.int32)
    ab = np.abs(qi)
    nt, kt = n // TILE, k // TILE
    if nt == 0 or kt == 0:
        return {}
    ab = ab[: nt * TILE, : kt * TILE]
    qi = qi[: nt * TILE, : kt * TILE]
    out = {}
    for chip in CHIPS:
        for b in ZBITS:
            best = np.ones((nt, kt))
            arg = np.full((nt, kt), -1)
            for a in ALPHABET[1:]:
                small = ab <= a
                o = ~small
                d_out = o.reshape(nt, TILE, kt, TILE).mean(axis=(1, 3))
                ex_s = group_excess(small & (qi != 0), kt * TILE).reshape(nt, TILE, kt, TILE // 4).sum(axis=(1, 3)) / TILE**2
                ex_o = group_excess(o, kt * TILE).reshape(nt, TILE, kt, TILE // 4).sum(axis=(1, 3)) / TILE**2
                c_small = np.minimum(dense_part(chip, a, b), sparse24_part(chip, a, b, 0.0) + ex_s * CHIPS[chip]["alu_mac"])
                c_out = np.minimum(np.minimum(d_out * CHIPS[chip]["alu_mac"], 1.0),
                                   CHIPS[chip]["sparse24"] + ex_o * CHIPS[chip]["alu_mac"])
                c_out = np.where(d_out == 0, 0.0, c_out)
                c = c_small + c_out
                better = c < best
                best = np.where(better, c, best)
                arg = np.where(better, a, arg)
            out[f"{chip}_b{b}"] = {"cost": float(best.mean()), "frac_tiles_below_1": float((best < 1).mean()),
                                   "min_tile": float(best.min()),
                                   "alphabet_counts": {str(a): int((arg == a).sum()) for a in (-1,) + ALPHABET[1:]}}
    return out


def best_by(costs):
    best = {}
    for c in costs:
        key = f"{c['chip']}_b{c['b']}"
        if key not in best or c["cost"] < best[key]["cost"]:
            best[key] = c
    return best


def analyse(q, rng, rpca):
    n, k = q.shape
    m = min(n, k)
    t0 = time.time()
    rec = {"shape": [n, k], "alphabet": alphabet_stats(q), "lsb": lsb_stats(q)}
    qf = torch.from_numpy(q.astype(np.float32))
    U, S, Vh = torch.linalg.svd(qf, full_matrices=False)
    s2 = (S.double() ** 2).numpy()
    tail = np.concatenate([np.cumsum(s2[::-1])[::-1], [0.0]])
    tot = s2.sum()
    grid = rank_grid(m)
    rec["spectrum"] = {
        "energy_top": {str(r): float(1 - tail[r] / tot) for r in grid if r > 0},
        "tail_rms": {str(r): float(math.sqrt(tail[r] / (n * k))) for r in grid},
        "sv": [float(x) for x in S[: min(8, m)]],
    }
    rmin = {}
    for a in ALPHABET[1:]:
        ok = np.nonzero(np.sqrt(tail / (n * k)) <= a)[0]
        rmin[str(a)] = int(ok[0]) if ok.size else m
    rec["eckart_young_rmin"] = rmin
    floor = {}
    for chip in CHIPS:
        for b in ZBITS:
            floor[f"{chip}_b{b}"] = min(
                [1.0] + [lowrank_part(rmin[str(a)], n, k) + dense_part(chip, a, b) for a in ALPHABET[1:]])
    rec["no_outlier_floor"] = floor
    shuf = np.array([rng.permutation(row) for row in q])
    ss = torch.linalg.svdvals(torch.from_numpy(shuf.astype(np.float32))).double().numpy() ** 2
    stail = np.concatenate([np.cumsum(ss[::-1])[::-1], [0.0]])
    rec["shuffled_tail_rms"] = {str(r): float(math.sqrt(stail[r] / (n * k))) for r in grid}
    rec["shuffled_energy_top"] = {str(r): float(1 - stail[r] / ss.sum()) for r in grid if r > 0}

    blocks = max(1, min(n // TILE, EVAL_ROWS // TILE)) if n >= TILE else 1
    if n >= TILE:
        starts = np.sort(rng.choice(n // TILE, size=blocks, replace=False)) * TILE
        rows = np.concatenate([np.arange(s, s + TILE) for s in starts])
    else:
        rows = np.arange(n)
    qs = q[rows].astype(np.float32)
    costs = []
    for r in grid:
        L = (U[rows, :r] * S[:r]) @ Vh[:r] if r > 0 else torch.zeros(len(rows), k)
        costs += split_costs(qs, L.numpy(), n, k, r)
    rec["best_svd"] = best_by(costs)
    rec["best_r0"] = best_by([c for c in costs if c["r"] == 0])
    rec["by_rank"] = {str(r): best_by([c for c in costs if c["r"] == r]) for r in grid}
    if rpca:
        rp = []
        for r in sorted({32, m // 8} & set(grid) - {0}):
            for a in (3,):
                L = (U[:, :r] * S[:r]) @ Vh[:r]
                for _ in range(2):
                    out = torch.abs(torch.round(qf - L)) > a
                    M = torch.where(out, L, qf)
                    u, s, v = torch.svd_lowrank(M, q=min(m, r + 10), niter=2)
                    L = (u[:, :r] * s[:r]) @ v[:, :r].T
                rp += split_costs(qs, L[rows].numpy(), n, k, r)
        rec["best_rpca"] = best_by(rp)
    rec["tiled_r0"] = tiled_r0(q)
    best = {}
    for key in rec["best_svd"]:
        cands = [rec["best_svd"][key]["cost"]] + ([rec["best_rpca"][key]["cost"]] if rpca and key in rec["best_rpca"] else [])
        cands.append(rec["tiled_r0"].get(key, {}).get("cost", 1.0))
        best[key] = min(cands)
    rec["best_cost"] = best
    rec["seconds"] = time.time() - t0
    return rec


def model_tasks(name, layers_arg):
    repo = MODELS[name]
    cfg = json.loads(fetch(f"{HF}/{repo}/resolve/main/config.json"))
    nl = cfg["num_hidden_layers"]
    layers = sorted({0, nl // 2, nl - 1}) if layers_arg == "edges" else [int(x) for x in layers_arg.split(",")]
    try:
        idx = json.loads(fetch(f"{HF}/{repo}/resolve/main/model.safetensors.index.json"))["weight_map"]
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        print(f"{repo}: no safetensors index (404), reading the single model.safetensors", flush=True)
        idx = None
    wanted = [f"model.layers.{i}.{blk}.{p}.weight" for i in layers for blk, p in PROJ]
    files = sorted({idx[t] for t in wanted}) if idx else ["model.safetensors"]
    headers = {f: safetensors_header(f"{HF}/{repo}/resolve/main/{f}") for f in files}
    tasks = []
    for i, t in enumerate(wanted):
        f = idx[t] if idx else "model.safetensors"
        base, h = headers[f]
        tasks.append((name, repo, t, f"{HF}/{repo}/resolve/main/{f}", base, h[t], 20261030 + i))
    return {"model": name, "repo": repo, "config": cfg, "layers": layers}, tasks


def work(task, rpca):
    name, repo, t, url, base, entry, seed = task
    torch.set_num_threads(1)
    q = quantize(load_tensor(url, base, entry))
    rec = analyse(q, np.random.default_rng(seed), rpca)
    rec.update({"model": name, "repo": repo, "tensor": t})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--layers", default="edges")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-rpca", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta, tasks, results = {}, [], {}
    for name in args.models.split(","):
        meta[name], tk = model_tasks(name, args.layers)
        results[name] = []
        tasks += tk
    big_first = sorted(tasks, key=lambda t: -math.prod(t[5]["shape"]))
    with concurrent.futures.ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, t, not args.no_rpca) for t in big_first]
        for fut in concurrent.futures.as_completed(futs):
            rec = fut.result()
            name = rec["model"]
            results[name].append(rec)
            b = rec["best_cost"]
            print(f"{name} {rec['tensor']} {rec['shape']} sigma_q={rec['alphabet']['sigma_q']:.2f} "
                  f"best h200_b2={b['h200_b2']:.3f} sm120_b2={b['sm120_b2']:.3f} "
                  f"p_odd={rec['lsb']['p_odd']:.4f} lzma={rec['lsb']['lzma_ratio']:.4f} {rec['seconds']:.0f}s", flush=True)
            (out / f"{name}.json").write_text(json.dumps({**meta[name], "matrices": results[name]}))


if __name__ == "__main__":
    main()
