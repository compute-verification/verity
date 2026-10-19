"""Plain attention's time on one GPU, fattn.cu's baseline: torch's scaled_dot_product_attention in bf16, causal, with
GQA (H query heads on kvh KV heads, head dimension 128), under each backend that runs it (FlashAttention-2, cuDNN,
memory-efficient). GQA through enable_gqa where the backend takes it, else K and V expanded to H heads before timing. Hot
items interleaved across the backends, CUDA events around enough calls for about 20 ms; one JSON line a shape.
    python3 flash_ref.py REPS S:H:KVH...
"""
from __future__ import annotations

import json
import statistics
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

BACKENDS = {"flash": SDPBackend.FLASH_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION}


def runner(name: str, q, k, v):
    """A call of backend `name`, or None if it refuses the shape: GQA native first, then K and V expanded."""
    rep = q.shape[1] // k.shape[1]
    ke, ve = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    for mode, call in (("gqa", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)),
                       ("expanded", lambda: F.scaled_dot_product_attention(q, ke, ve, is_causal=True))):
        try:
            with sdpa_kernel([BACKENDS[name]]):
                out = call()
            torch.cuda.synchronize()
        except RuntimeError as exc:
            last = str(exc).splitlines()[0][:200]
            continue
        return mode, call, out
    return last, None, None


def main() -> int:
    reps = int(sys.argv[1])
    dev = torch.cuda.get_device_properties(0)
    for shape in sys.argv[2:]:
        s, H, kvh = (int(x) for x in shape.split(":"))
        g = torch.Generator(device="cuda").manual_seed(s + H)
        q = torch.randn(1, H, s, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        k = torch.randn(1, kvh, s, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, kvh, s, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        arms, info, ref = {}, {}, None
        for name in BACKENDS:
            mode, f, out = runner(name, q, k, v)
            info[name] = mode
            if f is None:
                continue
            if ref is None:
                ref = out
            info[name + "_maxdiff"] = float((out.float() - ref.float()).abs().max())
            arms[name] = f
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        def time_arm(n, calls):
            with sdpa_kernel([BACKENDS[n]]):
                e0.record()
                for _ in range(calls):
                    arms[n]()
                e1.record()
            e1.synchronize()
            return e0.elapsed_time(e1) / calls

        names = list(arms)
        calls = {}
        for n in names:
            time_arm(n, 1)
            t = time_arm(n, 2)
            calls[n] = max(2, min(400, int(20.0 / max(t, 1e-4))))
        ms = {n: [] for n in names}
        for r in range(reps):
            for j in range(len(names)):
                n = names[(j + r) % len(names)]
                ms[n].append(time_arm(n, calls[n]))
        print(json.dumps({"s": s, "H": H, "kvh": kvh, "torch": torch.__version__, "device": dev.name,
                          "cudnn": torch.backends.cudnn.version(), "modes": info,
                          "arms": {n: {"ms": statistics.median(ms[n]), "calls": calls[n], "ms_all": ms[n]} for n in names}}),
              flush=True)
        del q, k, v, ref
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
