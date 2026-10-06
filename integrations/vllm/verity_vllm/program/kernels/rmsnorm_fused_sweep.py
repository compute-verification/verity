"""Sweep of the two `RMSNormFusedCuda` variance-summation trees over every fused-norm row of the B0 commit store.

The B0 store (`vllm-poc/bundles/B0-358bb6d66805-ta1eager-commit.tar.xz`: SmolLM2-135M, one request of 32 prompt
tokens + 8 decode steps, A100, vLLM `d9105ea80`, `VLLM_BATCH_INVARIANT=1`) commits every `rmsnorm` output of
every position, so all 2,340 `fused_add_rms_norm` rows (39 positions x (59 layer norms + the final norm)) can be
replayed on the CPU from committed operands: `x` = the producing GEMM's leaf, `res` = the previous norm's
residual output (layer 0: the embedding), `w` = the checkpoint norm weight (fixture
`fixtures/verity-ir/rmsnorm-fused-b0/smollm2_norm_weights.npz`, sha256-checked against the store's weights index).

Two candidate trees, as vectorised numpy twins of the registered composites (checked against the gate-by-gate
IR evaluation in `tests/program/test_rmsnorm_fused.py`):

* `v1` — `RMSNormFusedCuda_v1`: `fused_add_rms_norm_kernel<bf16, 8>` (N/8 threads x one 8-vector, pairs summed
  sequentially); this kernel is guarded by `!batch_invariant_launch` in `csrc/layernorm_kernels.cu` and is not
  the one production ran.
* `v2` — `RMSNormFusedCuda_v2`: the generic `fused_add_rms_norm_kernel<c10::BFloat16, 0, true>` launched under
  `VLLM_BATCH_INVARIANT=1`: `block = min(N, 1024)` threads, thread t accumulates `fma(z_j, z_j, .)` over
  `j = t, t+block, ...` from 0, then `cub::BlockReduce<float, 1024>` (per-warp shfl_down tree 1,2,4,8,16, warp
  aggregates added sequentially from warp 0).

Both share the epilogue `s = rsqrtf(var / N + eps)` (MUFU.RSQ), `out = bf16((z * s) * w)`.

    PYTHONPATH=. verity-vllm rmsnorm-fused-sweep [--bundle PATH] [--out sweep.json]

Result (2026-09-07, this machine): v1 exact on 2,327/2,340 rows (1,347,826/1,347,840 words); v2 exact on
2,340/2,340 rows; the variance differs between the trees in 1,218 rows, `s` in 689, the bf16 output in 13 —
every one of those 13 words is what production committed under v2.
"""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path

import numpy as np
from dataclasses import dataclass

from verity_vllm.config import ROOT as REPO
from verity_vllm.config import option
from verity_vllm.program.registry.b1 import B0_CONFIG, FUSED_RMS_BLOCK

BUNDLE = REPO / "vllm-poc/bundles/B0-358bb6d66805-ta1eager-commit.tar.xz"
BUNDLE_SHA256 = "7ba7df9b00699d10740a3c25c8c8bbd207bbe364d281451d3d8134ae8fc97dba"
NORM_WEIGHTS = REPO / "fixtures/verity-ir/rmsnorm-fused-b0/smollm2_norm_weights.npz"
REQUEST = "dc159809-b295-51b0-9148-a405bc703cc1"


# ---------------------------------------------------------------------------------------------------------
# numpy twins (f32 = IEEE RN-even in numpy; MUFU.RSQ via the registered primitive's table)
# ---------------------------------------------------------------------------------------------------------

def bf16_to_f32(w: np.ndarray) -> np.ndarray:
    return (np.asarray(w, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_f2fp(x: np.ndarray) -> np.ndarray:
    """`__float2bfloat16` / `cvt.rn.bf16.f32`: RNE, canonical NaN 0x7FFF."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    nan = ((u & 0x7F800000) == 0x7F800000) & ((u & 0x007FFFFF) != 0)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    r[nan] = 0x7FFF
    return r


def _warp_tree(P: np.ndarray) -> np.ndarray:
    """cub `WarpReduceShfl`: lane i adds lane i+off for off = 1,2,4,8,16 when that lane exists. P: [rows, <=32]."""
    lanes = {i: P[:, i] for i in range(P.shape[1])}
    for off in (1, 2, 4, 8, 16):
        lanes = {i: ((v + lanes[i + off]).astype(np.float32) if i + off in lanes else v)
                 for i, v in lanes.items() if i % (2 * off) == 0}
    return lanes[0]


def cub_block_reduce_sum(partials: np.ndarray) -> np.ndarray:
    """`BlockReduce<float,1024>.Reduce(v, plus, blockDim.x)` over `partials[:, 0:blockDim.x]`; warp aggregates
    added sequentially from warp 0 (thread 0's `ApplyWarpAggregates`)."""
    n = partials.shape[1]
    warps = [_warp_tree(partials[:, 32 * w:32 * w + 32]) for w in range((n + 31) // 32)]
    var = warps[0]
    for w in warps[1:]:
        var = (var + w).astype(np.float32)
    return var


def variance_v1(z: np.ndarray) -> np.ndarray:
    """`fused_add_rms_norm_kernel<bf16, 8>`: N/8 threads, `((z0²+z1²) + (z2²+z3²)) + ...` sequential over 4 pairs."""
    rows, N = z.shape
    v = z.reshape(rows, N // 8, 8)
    sq = (v * v).astype(np.float32)
    pair = (sq[:, :, 0::2] + sq[:, :, 1::2]).astype(np.float32)
    acc = pair[:, :, 0]
    for i in range(1, 4):
        acc = (acc + pair[:, :, i]).astype(np.float32)
    return cub_block_reduce_sum(acc)


def variance_v2(z: np.ndarray, block: int = FUSED_RMS_BLOCK) -> np.ndarray:
    """Generic `fused_add_rms_norm_kernel<c10::BFloat16, 0, true>`: `min(N, block)` threads, thread t owns
    `t, t+block, ...`, `variance += x*x` (fma; the bf16 square is exact in f32 so `RN(acc + x*x)` is the fma)."""
    rows, N = z.shape
    nthreads = min(N, block)
    acc = np.zeros((rows, nthreads), np.float32)
    for base in range(0, N, nthreads):
        cnt = min(nthreads, N - base)
        x = z[:, base:base + cnt]
        acc[:, :cnt] = (acc[:, :cnt] + (x * x).astype(np.float32)).astype(np.float32)
    return cub_block_reduce_sum(acc)


def rsqrt_approx(x: np.ndarray) -> np.ndarray:
    """`rsqrt.approx.f32` through the registered primitive's table (same model the IR gate evaluates)."""
    from verity_vllm.program.kernels.prims_twins import _rms
    rm, t = _rms()
    return np.asarray(rm.rsqrt_f32(np.ascontiguousarray(x, dtype=np.float32), t), dtype=np.float32)


def fused_rows(x: np.ndarray, res: np.ndarray, w: np.ndarray, eps: float, variance) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rows of (normed, res') for bf16 word arrays x, res [rows, N] and w [rows, N] (per-row weight) under the
    given variance tree. Returns (normed words, res' words, var f32, s f32)."""
    rows, N = x.shape
    rp = f32_to_bf16_f2fp((bf16_to_f32(x) + bf16_to_f32(res)).astype(np.float32))
    z = bf16_to_f32(rp)
    var = variance(z)
    mean = (var / np.float32(N)).astype(np.float32)
    s = rsqrt_approx((mean + np.float32(eps)).astype(np.float32))
    y = (z * s[:, None]).astype(np.float32)
    y = (y * bf16_to_f32(w)).astype(np.float32)
    return f32_to_bf16_f2fp(y), rp, var, s


# ---------------------------------------------------------------------------------------------------------
# the B0 store's fused-norm rows
# ---------------------------------------------------------------------------------------------------------

def extract_bundle(bundle: Path, dest: Path) -> Path:
    h = hashlib.sha256(bundle.read_bytes()).hexdigest()
    if h != BUNDLE_SHA256:
        raise ValueError(f"{bundle}: sha256 {h} != {BUNDLE_SHA256}")
    with tarfile.open(bundle, "r:xz") as tf:
        tf.extractall(dest, filter="data")
    return dest / "commit"


def load_store_leaves(commit: Path, request: str = REQUEST) -> dict[str, np.ndarray]:
    """Every committed replay-unit leaf of the request as raw words, keyed by `t/ph=../l=../op=../port=k`."""
    leaves: dict[str, np.ndarray] = {}
    root = commit / "ru" / request
    for step in sorted(os.listdir(root)):
        p = root / step
        if not p.is_dir():
            continue
        ix = json.loads((p / "index.json").read_text())
        vals = np.memmap(p / "values.bin", dtype=np.uint8, mode="r")
        for e in ix["entries"]:
            raw = vals[e["offset"]:e["offset"] + e["nbytes"]]
            dt = {"bfloat16": np.uint16, "int32": np.uint32, "float32": np.uint32}[e["dtype"]]
            key = e["canonical_id"].split("/t=", 1)[1]
            leaves[key] = np.frombuffer(raw.tobytes(), dtype=dt)
    return leaves


def load_norm_weights(commit: Path, path: Path = NORM_WEIGHTS) -> dict[str, np.ndarray]:
    """The 61 norm weights (bf16 words), each checked against the store's weights index sha256."""
    z = np.load(path)
    idx = json.loads((commit / "weights" / "index.json").read_text())
    ents = {e["canonical_id"]: e for e in idx["entries"]}
    out = {}
    for k in z.files:
        cid = "w=" + k.replace("model.", "", 1)
        words = np.ascontiguousarray(z[k], dtype=np.uint16)
        h = hashlib.sha256(words.tobytes()).hexdigest()
        if ents[cid]["sha256"] != h:
            raise ValueError(f"norm weight fixture {k} does not match the committed leaf {cid}")
        out[k] = words
    return out


def fused_norm_rows(leaves: dict[str, np.ndarray], norms: dict[str, np.ndarray], C: dict = B0_CONFIG):
    """All fused-norm call sites of the store: (label, x, res, w, normed leaf, residual leaf)."""
    NL = C["NL"]

    def leaf(t, ph, l, op, port):
        return leaves[f"{t}/ph={ph}/l={l}/op={op}/port={port}"]

    positions = sorted({int(k.split("/")[0]) for k in leaves})
    rows = []
    for t in positions:
        ph = "prefill" if t < 32 else "decode"
        for L in range(NL):
            x = leaf(t, ph, L, f"layers.{L}.self_attn.o_proj.gemm", 0)
            res = leaf(t, ph, "embed", "embed_tokens.embedding", 0) if L == 0 else leaf(t, ph, L, f"layers.{L}.input_layernorm.rmsnorm", 1)
            rows.append((f"t={t}/layers.{L}/post_attention_layernorm", x, res, norms[f"model.layers.{L}.post_attention_layernorm.weight"],
                         leaf(t, ph, L, f"layers.{L}.post_attention_layernorm.rmsnorm", 0), leaf(t, ph, L, f"layers.{L}.post_attention_layernorm.rmsnorm", 1)))
            if L >= 1:  # layer 0's input norm is the Triton kernel (no residual)
                x = leaf(t, ph, L - 1, f"layers.{L - 1}.mlp.down_proj.gemm", 0)
                res = leaf(t, ph, L - 1, f"layers.{L - 1}.post_attention_layernorm.rmsnorm", 1)
                rows.append((f"t={t}/layers.{L}/input_layernorm", x, res, norms[f"model.layers.{L}.input_layernorm.weight"],
                             leaf(t, ph, L, f"layers.{L}.input_layernorm.rmsnorm", 0), leaf(t, ph, L, f"layers.{L}.input_layernorm.rmsnorm", 1)))
        x = leaf(t, ph, NL - 1, f"layers.{NL - 1}.mlp.down_proj.gemm", 0)
        res = leaf(t, ph, NL - 1, f"layers.{NL - 1}.post_attention_layernorm.rmsnorm", 1)
        rows.append((f"t={t}/norm", x, res, norms["model.norm.weight"], leaf(t, ph, "norm", "norm.rmsnorm", 0), leaf(t, ph, "norm", "norm.rmsnorm", 1)))
    return rows


TREES = {"v1": variance_v1, "v2": variance_v2}


def sweep(commit: Path, C: dict = B0_CONFIG) -> dict:
    leaves = load_store_leaves(commit)
    norms = load_norm_weights(commit)
    rows = fused_norm_rows(leaves, norms, C)
    X = np.stack([r[1] for r in rows]); R = np.stack([r[2] for r in rows]); W = np.stack([r[3] for r in rows])
    O0 = np.stack([r[4] for r in rows]); O1 = np.stack([r[5] for r in rows])
    labels = [r[0] for r in rows]
    report: dict = {"store": str(commit), "request": REQUEST, "N": C["H"], "rows": len(rows), "words_per_row": C["H"], "trees": {}}
    per = {}
    for name, tree in TREES.items():
        out, rp, var, s = fused_rows(X, R, W, C["EPS"], tree)
        ok = np.all(out == O0, axis=1) & np.all(rp == O1, axis=1)
        per[name] = (out, var, s)
        report["trees"][name] = {"rows_exact": int(ok.sum()), "words_exact": int((out == O0).sum()) + int((rp == O1).sum()),
                                "words_compared": 2 * O0.size, "residual_rows_exact": int(np.all(rp == O1, axis=1).sum()),
                                "mismatching_rows": [labels[k] for k in np.nonzero(~ok)[0]]}
    o1, v1, s1 = per["v1"]; o2, v2, s2 = per["v2"]
    dout = np.any(o1 != o2, axis=1)
    report["v1_vs_v2"] = {"variance_differs_rows": int((v1 != v2).sum()), "s_differs_rows": int((s1 != s2).sum()),
                          "output_differs_rows": int(dout.sum()),
                          "rows": [{"row": labels[k], "words": np.nonzero(o1[k] != o2[k])[0].tolist(),
                                    "v1": [f"0x{v:04x}" for v in o1[k][o1[k] != o2[k]]], "v2": [f"0x{v:04x}" for v in o2[k][o1[k] != o2[k]]],
                                    "captured": [f"0x{v:04x}" for v in O0[k][o1[k] != o2[k]]],
                                    "var_v1": f"0x{v1[k].view(np.uint32):08x}", "var_v2": f"0x{v2[k].view(np.uint32):08x}",
                                    "s_v1": f"0x{s1[k].view(np.uint32):08x}", "s_v2": f"0x{s2[k].view(np.uint32):08x}"}
                                   for k in np.nonzero(dout)[0]]}
    return report


@dataclass(frozen=True, kw_only=True)
class Options:
    bundle: Path = option("--bundle", type=Path, default=BUNDLE)
    commit: Path | None = option("--commit", type=Path, default=None,
                                 help="an already extracted commit store (skips the bundle)")
    out: Path | None = option("--out", type=Path, default=None)


def main(a: Options) -> int:
    with tempfile.TemporaryDirectory() as td:
        commit = a.commit or extract_bundle(a.bundle, Path(td))
        rep = sweep(commit)
    for name, r in rep["trees"].items():
        print(f"{name}: rows exact {r['rows_exact']}/{rep['rows']}, words exact {r['words_exact']}/{r['words_compared']}; "
              f"mismatching rows: {r['mismatching_rows']}")
    d = rep["v1_vs_v2"]
    print(f"v1 vs v2: variance differs in {d['variance_differs_rows']} rows, s in {d['s_differs_rows']}, bf16 output in {d['output_differs_rows']}")
    for r in d["rows"]:
        print(f"  {r['row']}: words {r['words']} v1={r['v1']} v2={r['v2']} captured={r['captured']}")
    if a.out:
        a.out.write_text(json.dumps(rep, indent=1))
    return 0
