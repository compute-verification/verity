"""tree_allreduce's mnnvl fused all-reduce + RMSNorm emulation on one GPU: it runs at every token count vLLM's fusion takes
(1 to 42 rows of 6144 at TP8 on H200), under a CUDA graph as in eager, and stays within an ulp or two of an f64 RMSNorm.
Whether it is mnnvl's bits only a replay against the serve shows.

    python fused_smoke.py      (one GPU)
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tree_allreduce as ta

DIM, EPS, W = 6144, 1e-5, 8
dev = torch.device("cuda")
sms = torch.cuda.get_device_properties(dev).multi_processor_count
scale = ta.mufu_scale()
print(f"device {torch.cuda.get_device_name(dev)} SMs {sms}")
g = torch.Generator(device=dev).manual_seed(0)
gamma = (torch.rand(DIM, generator=g, device=dev) + 0.5).to(torch.bfloat16)
worst, launches = 0.0, set()
for t in range(1, 43):
    oneshot = t * DIM * W * 2 <= 64 * 1024 * 8 * 2
    launches.add((t, oneshot, ta.mnnvl_grid(t, DIM, sms)))
    ranks = (torch.randn(W, t, DIM, generator=g, device=dev) * 0.3).to(torch.bfloat16)
    res = torch.randn(t, DIM, generator=g, device=dev).to(torch.bfloat16)
    acc = torch.zeros(t, DIM, device=dev)
    for r in range(W):
        acc += ranks[r].float()
    ar = acc.to(torch.bfloat16)
    normed, h = ta.mnnvl_add_rmsnorm(ar, res, gamma, EPS, 0.0, oneshot, sms, scale)
    hv = h.double()
    ref = (hv * torch.rsqrt((hv * hv).mean(1, keepdim=True) + EPS) * gamma.double())
    ulp = ((normed.double() - ref).abs() / ref.abs().clamp_min(1e-30) * 128).max().item()
    worst = max(worst, ulp)
    s = torch.rand(t, generator=g, device=dev) * 1e4
    mine, approx = scale(s, DIM, EPS), torch.rsqrt(s / DIM + EPS)
    assert torch.allclose(mine, approx, rtol=1e-6, atol=0), (t, (mine - approx).abs().max())
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ta.mnnvl_add_rmsnorm(ar, res, gamma, EPS, 0.0, oneshot, sms, scale)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gn, gh = ta.mnnvl_add_rmsnorm(ar, res, gamma, EPS, 0.0, oneshot, sms, scale)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(gn, normed) and torch.equal(gh, h), f"{t} rows: the graph's replay differs from eager"
print("launches (rows, one-shot, (threads, blocks)):", sorted(launches))
print(f"worst error against an f64 RMSNorm of h: {worst:.2f} bf16 ulp; graph replay equals eager at all 42 row counts")
assert worst <= 2.0, worst
print("fused_smoke: ok")
