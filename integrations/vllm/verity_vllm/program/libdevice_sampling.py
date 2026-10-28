"""[ov-sampling] vLLM's Gumbel noise row, bit-exact and torch-free (`gumbel_noise_exact`: Philox4x32-10 and
`uint_to_uniform_float` from `sampling_rng`, `__nv_logf` and `__nv_log1pf` from the libdevice transcription, `libdevice`), and
its comparison against the pod probe's GPU dumps.

    verity-vllm libdevice-sampling --compare PROBE_DIR   # PROBE_DIR/g_<seed>_<pos>.npy from the pod probe
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from dataclasses import dataclass

from verity_vllm.config import option
from verity_vllm.program import sampling_rng as R
from verity.ml.definitions.numerics.libdevice import F32, nv_log1pf, nv_logf


def gumbel_noise_exact(seed: int, pos: int, vocab_size: int, *, is_drafting: bool = False) -> np.ndarray:
    """vLLM `gumbel_noised_argmax`'s fp32 noise row, bit-exact: u = max(tl.rand(randint(seed, pos), v), 2^-31ish);
    g = -logf(-log1pf(-u))."""
    gs = R.gumbel_seed_of(seed, pos, is_drafting=is_drafting)
    keys = np.arange(int(vocab_size), dtype=np.int32)
    u = np.maximum(R.tl_rand(gs, keys, seed_bits=32), R.TL_RAND_MIN_F32).astype(F32)
    return (-nv_logf((-nv_log1pf((-u).astype(F32))).astype(F32))).astype(F32)


def compare_dir(probe_dir: Path) -> dict:
    """PROBE_DIR/g_<seed>_<pos>_<V>.npy (GPU) vs the transcription; also the GPU `u`/`l1` dumps when present."""
    out: dict = {"schema": "verity-gen/ov-sampling-libdevice-compare/v1", "probes": []}
    for p in sorted(probe_dir.glob("g_*.npy")):
        _, seed, pos, V = p.stem.split("_")
        seed, pos, V = int(seed), int(pos), int(V)
        g_gpu = np.load(p)
        g_cpu = gumbel_noise_exact(seed, pos, V)
        neq = int((g_gpu.view(np.uint32) != g_cpu.view(np.uint32)).sum())
        rec = {"seed": seed, "pos": pos, "V": V, "n_neq_g": neq, "bit_exact_g": neq == 0}
        l1 = probe_dir / f"l1_{seed}_{pos}_{V}.npy"
        if l1.exists():
            gs = R.gumbel_seed_of(seed, pos)
            u = np.maximum(R.tl_rand(gs, np.arange(V, dtype=np.int32), seed_bits=32), R.TL_RAND_MIN_F32).astype(F32)
            l1_cpu = nv_log1pf((-u).astype(F32))
            rec["n_neq_log1pf"] = int((np.load(l1).view(np.uint32) != l1_cpu.view(np.uint32)).sum())
        if neq:
            idx = np.flatnonzero(g_gpu.view(np.uint32) != g_cpu.view(np.uint32))[:5]
            rec["examples"] = [{"v": int(i), "gpu": float(g_gpu[i]), "cpu": float(g_cpu[i])} for i in idx]
        out["probes"].append(rec)
    out["all_bit_exact"] = bool(out["probes"]) and all(r["bit_exact_g"] for r in out["probes"])
    return out


@dataclass(frozen=True, kw_only=True)
class Options:
    compare: str = option("--compare", help="directory with the pod probe's g_<seed>_<pos>_<V>.npy dumps")
    out: str | None = option("--out", default=None)


def main(a: Options) -> int:
    rep = compare_dir(Path(a.compare))
    txt = json.dumps(rep, indent=1)
    if a.out:
        Path(a.out).write_text(txt + "\n")
    print(txt)
    return 0 if rep["all_bit_exact"] else 1
