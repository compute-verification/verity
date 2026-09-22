"""[ov-sampling] POD-ONLY: dump the GPU's Gumbel noise rows `g(seed, pos)[v]` (and the `log1pf` intermediate) exactly as vLLM's
`_gumbel_sample_kernel` computes them (Triton `tl.randint` / `tl.rand`, `tl.log`, `libdevice.log1p`), for a grid of
(seed, pos) pairs at the B1 vocabulary, so `gen_ov_sampling_libdevice.py --compare` can check its transcription bit for bit.

    PYTHONPATH=. /workspace/venv-cu129/bin/python -m verity_capture.bench.gen_ov_sampling_probe --out /workspace/cp/probe
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _noise_kernel(seed, pos, out_u_ptr, out_l1_ptr, out_g_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    gumbel_seed = tl.randint(seed, pos)              # scalar stream key, int64 seed / int64 pos as vLLM passes them
    u = tl.rand(gumbel_seed, offs)                   # int32 keys = vocab index
    u = tl.maximum(u, 4.6566127342e-10)
    tl.store(out_u_ptr + offs, u, mask=mask)
    l1 = tl.extra.cuda.libdevice.log1p(-u)
    tl.store(out_l1_ptr + offs, l1, mask=mask)
    g = -tl.log(-l1)
    tl.store(out_g_ptr + offs, g, mask=mask)


def dump(seed: int, pos: int, V: int, out: Path) -> dict:
    dev = "cuda"
    u = torch.empty(V, dtype=torch.float32, device=dev)
    l1 = torch.empty(V, dtype=torch.float32, device=dev)
    g = torch.empty(V, dtype=torch.float32, device=dev)
    BLOCK = 1024
    _noise_kernel[(triton.cdiv(V, BLOCK),)](seed, pos, u, l1, g, V, BLOCK=BLOCK)
    torch.cuda.synchronize()
    np.save(out / f"u_{seed}_{pos}_{V}.npy", u.cpu().numpy())
    np.save(out / f"l1_{seed}_{pos}_{V}.npy", l1.cpu().numpy())
    np.save(out / f"g_{seed}_{pos}_{V}.npy", g.cpu().numpy())
    gg = g.cpu().numpy()
    return {"seed": seed, "pos": pos, "V": V, "g_min": float(gg.min()), "g_max": float(gg.max()), "n_nonfinite": int((~np.isfinite(gg)).sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", type=int, default=151936)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = [(0, 32), (0, 47), (1, 32), (12345, 40), (-1, 33), (2**40 + 7, 2**33 + 5), (7, 0), (0, 2**30 + 32)]
    recs = [dump(s, p, a.vocab, out) for s, p in pairs]
    meta = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "triton": triton.__version__, "python": platform.python_version(),
            "probes": recs}
    (out / "probe_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
