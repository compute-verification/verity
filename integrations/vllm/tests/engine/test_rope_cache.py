"""`engine.rope_cache`: the fused rope and KV-cache write against vLLM's `_C.rotary_embedding` then
`_C_cache_ops.reshape_and_cache_flash`, bit for bit over the whole qkv buffer and the whole cache (bytes it must not write
included), on random bit patterns (NaN payloads, infinities, -0 and subnormals, in the rows and in the cos/sin table),
served-like rows on a vLLM-computed table, position 0 and the table's last position, partial blocks, padded and negative
slots, and both cache layouts; and its install on layers running vLLM's `LlamaAttention.forward` over its own rope and
attention layer, which stands aside wherever the Commit or an observer would see a difference."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

#: (query heads, KV heads, head dimension): Llama-3.2-1B's, Llama-3.1-8B's, and one whose heads and halves the kernel pads.
L1B, L8B, ODD = (32, 8, 64), (32, 8, 128), (24, 6, 96)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.stride() == b.stride() and torch.equal(_bits(a), _bits(b))


def _random_bits(shape, g) -> torch.Tensor:
    return torch.randint(-32768, 32768, shape, generator=g, device="cuda", dtype=torch.int32).to(torch.int16).view(torch.bfloat16)


def _table(max_pos: int, d: int, base: float = 500000.0) -> torch.Tensor:
    """vLLM's `RotaryEmbedding._compute_cos_sin_cache` (fp32, then bf16) on the GPU."""
    inv = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float, device="cuda") / d))
    f = torch.einsum("i,j -> ij", torch.arange(max_pos, dtype=torch.float, device="cuda"), inv)
    return torch.cat((f.cos(), f.sin()), dim=-1).bfloat16()


def _cache(blocks: int, block_size: int, layout: str, g, dims) -> torch.Tensor:
    """FA's cache [blocks, KVH, block_size, 2D] (`FlashAttentionImpl.forward`), its bytes random: HND is that shape
    contiguous, NHD the same view of [blocks, block_size, KVH, 2D]."""
    _, kvh, d = dims
    if layout == "HND":
        return _random_bits((blocks, kvh, block_size, 2 * d), g)
    return _random_bits((blocks, block_size, kvh, 2 * d), g).permute(0, 2, 1, 3)


def _slots(t: int, n: int, blocks: int, block_size: int, start: int, holes: int, g) -> torch.Tensor:
    """n slots for t tokens: consecutive token slots from `start` through a random block order (partial first and last
    blocks), `holes` of them -1."""
    order = torch.randperm(blocks, generator=g, device="cuda")
    i = torch.arange(start, start + n, device="cuda")
    s = order[i // block_size] * block_size + i % block_size
    if holes:
        s[torch.randperm(n, generator=g, device="cuda")[:holes]] = -1
    return s.to(torch.int64).contiguous()


def _vllm(positions, qkv, table, kv_cache, slots, dims):
    import vllm._custom_ops as ops
    h, kvh, d = dims
    q, k, v = qkv.split([h * d, kvh * d, kvh * d], dim=-1)
    ops.rotary_embedding(positions, q, k, d, table, True)
    kc, vc = kv_cache.transpose(1, 2).split(d, dim=-1)
    one = torch.ones(1, dtype=torch.float32, device="cuda")
    if slots.numel():  # its grid is one block a slot
        ops.reshape_and_cache_flash(k.view(-1, kvh, d), v.view(-1, kvh, d), kc, vc, slots, "auto", one, one)


def _fused(positions, qkv, table, kv_cache, slots, dims, **kw):
    from verity_vllm.engine import rope_cache as R
    h, kvh, d = dims
    q, k, v = qkv.split([h * d, kvh * d, kvh * d], dim=-1)
    kc, vc = kv_cache.transpose(1, 2).split(d, dim=-1)
    return R.rope_cache(positions, q, k, v, table, kc, vc, slots, **kw)


def _case(t, n, block_size, layout, rows, table, seed, dims, holes=0, start=5):
    h, kvh, d = dims
    g = torch.Generator(device="cuda").manual_seed(seed)
    max_pos = table.shape[0]
    positions = torch.randint(0, max_pos, (t,), generator=g, device="cuda", dtype=torch.int64)
    positions[0] = 0
    if t > 1:
        positions[-1] = max_pos - 1
    if rows == "bits":
        qkv = _random_bits((t, (h + 2 * kvh) * d), g)
    else:
        qkv = (torch.randn(t, (h + 2 * kvh) * d, generator=g, device="cuda") * 2.0).bfloat16()
    blocks = (start + n) // block_size + 3
    return positions, qkv, _cache(blocks, block_size, layout, g, dims), _slots(t, n, blocks, block_size, start, holes, g)


@cuda
@pytest.mark.parametrize("t,n,block_size,layout,rows,holes,block_t,warps,dims", [
    (1, 1, 16, "NHD", "bits", 0, None, None, L1B),
    (37, 37, 16, "NHD", "bits", 0, None, None, L1B),
    (37, 37, 16, "HND", "served", 0, None, None, L1B),
    (300, 291, 16, "NHD", "bits", 7, None, None, L1B),
    (300, 0, 16, "HND", "bits", 0, None, None, L1B),
    (513, 500, 32, "HND", "bits", 3, 4, 2, L1B),
    (1088, 1088, 16, "NHD", "served", 0, 1, 1, L1B),
    (1088, 1088, 16, "NHD", "bits", 0, 8, 8, L1B),
    (4099, 4096, 16, "NHD", "served", 0, 16, 4, L1B),
    (37, 37, 16, "NHD", "bits", 0, None, None, L8B),
    (1088, 1000, 16, "NHD", "served", 5, None, None, L8B),
    (300, 291, 16, "HND", "bits", 7, 2, 4, L8B),
    (37, 37, 16, "NHD", "bits", 0, None, None, ODD),
    (300, 291, 32, "HND", "served", 7, 4, 1, ODD),
])
def test_rope_cache_is_rotary_embedding_then_reshape_and_cache_flash(t, n, block_size, layout, rows, holes, block_t, warps, dims):
    """Every byte of the qkv buffer and the cache is what vLLM's two ops leave: random bf16 bit patterns (rows and table)
    and served-like rows on vLLM's table at base 500,000, position 0 and the table's last row, first and last blocks
    partial, padded tokens past the slot mapping and -1 slots, at Llama-3.2-1B's and Llama-3.1-8B's heads and at a head
    count and width the kernel pads; the launch's PTX passes `ptx_check`."""
    pytest.importorskip("vllm")
    from verity_vllm.engine import rope_cache as R
    g = torch.Generator(device="cuda").manual_seed(t + n)
    d = dims[2]
    table = _random_bits((2048, d), g) if rows == "bits" else _table(131072, d)
    positions, qkv, kv_cache, slots = _case(t, n, block_size, layout, rows, table, seed=t + 1, dims=dims, holes=holes)
    want_qkv, want_cache = qkv.clone(), kv_cache.clone()
    _vllm(positions, want_qkv, table, want_cache, slots, dims)
    kw = {k: v for k, v in (("block_t", block_t), ("num_warps", warps)) if v is not None}
    launch = _fused(positions, qkv, table, kv_cache, slots, dims, **kw)
    torch.cuda.synchronize()
    assert _same(qkv, want_qkv)
    assert _same(kv_cache, want_cache)
    doc = R.ptx_check(launch)
    assert doc["ok"], doc


@cuda
def test_rope_cache_on_bf16_extremes():
    """Rows and table drawn from bf16's extremes (NaN, both infinities, both zeros, the smallest subnormal and normal, the
    largest finite, and products that underflow), each against vLLM's two ops bit for bit."""
    pytest.importorskip("vllm")
    edge = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 9.2e-41, -9.2e-41, 1.18e-38, -1.18e-38,
                         3.3895e38, -3.3895e38, 1e-20, -1e-20, 1e-19, 1.0, -1.0, 0.5, 0.70703125, -0.70703125, 1e-30,
                         2.0 ** -70, 3.0], device="cuda")
    g = torch.Generator(device="cuda").manual_seed(3)
    for seed, dims in enumerate((L1B, L1B, L8B, ODD)):
        h, kvh, d = dims
        t = 64
        table = edge[torch.randint(0, len(edge), (256, d), generator=g, device="cuda")].bfloat16()
        positions, _, kv_cache, slots = _case(t, t, 16, "NHD", "bits", table, seed=seed, dims=dims)
        qkv = edge[torch.randint(0, len(edge), (t, (h + 2 * kvh) * d), generator=g, device="cuda")].bfloat16()
        want_qkv, want_cache = qkv.clone(), kv_cache.clone()
        _vllm(positions, want_qkv, table, want_cache, slots, dims)
        _fused(positions, qkv, table, kv_cache, slots, dims)
        torch.cuda.synchronize()
        assert _same(qkv, want_qkv) and _same(kv_cache, want_cache), seed


WIRING = textwrap.dedent('''
    import json
    import torch
    import torch.nn as nn
    import vllm._custom_ops as ops
    from torch.utils._python_dispatch import TorchDispatchMode
    from vllm.config import CacheConfig, CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.models.llama import LlamaAttention

    from verity_vllm.engine import hooks, rope_cache as R

    H, KVH, D, T, N, BS, BLOCKS, LAYERS = 32, 8, 64, 300, 291, 16, 40, 2
    g = torch.Generator(device="cuda").manual_seed(7)
    last = {}

    class Qkv(nn.Module):
        def __init__(self, i):
            super().__init__()
            self.i = i
            self.w = (torch.randn((H + 2 * KVH) * D, H * D, generator=g, device="cuda") * 0.03).bfloat16()

        def forward(self, x):
            y = x @ self.w.t()
            last[self.i] = y
            return y, None

    class Out(nn.Module):
        def forward(self, x):
            return x.clone(), None

    class Attn(nn.Module):
        """vLLM's LlamaAttention.forward over its rope and attention layer."""
        forward = LlamaAttention.forward

        def __init__(self, i, vc):
            super().__init__()
            self.num_heads, self.num_kv_heads, self.head_dim = H, KVH, D
            self.q_size, self.kv_size = H * D, KVH * D
            self.qkv_proj, self.o_proj = Qkv(i), Out()
            self.rotary_emb = get_rope(D, max_position=131072, is_neox_style=True,
                                       rope_parameters={"rope_type": "llama3", "rope_theta": 500000.0, "factor": 32.0,
                                                        "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                                        "original_max_position_embeddings": 8192})
            self.attn = Attention(H, D, D ** -0.5, num_kv_heads=KVH, cache_config=vc.cache_config,
                                  prefix=f"model.layers.{i}.self_attn.attn")

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    vc = VllmConfig(cache_config=CacheConfig(block_size=BS), compilation_config=CompilationConfig(custom_ops=["all"]))
    with set_current_vllm_config(vc):
        model = nn.ModuleList([Attn(i, vc) for i in range(LAYERS)]).cuda()
    torch.set_default_dtype(prev)
    for m in model:
        m.attn.kv_cache = torch.randint(-32768, 32768, (BLOCKS, BS, KVH, 2 * D), generator=g, device="cuda",
                                        dtype=torch.int32).to(torch.int16).view(torch.bfloat16).permute(0, 2, 1, 3)
    caches0 = [m.attn.kv_cache.clone() for m in model]
    x0 = (torch.randn(T, H * D, generator=g, device="cuda")).bfloat16()
    positions = torch.randint(0, 1088, (T,), generator=g, device="cuda", dtype=torch.int64)
    positions[0], positions[-1] = 0, 131071
    i = torch.arange(5, 5 + N, device="cuda")
    slots = (torch.randperm(BLOCKS, generator=g, device="cuda")[i // BS] * BS + i % BS).to(torch.int64)
    slots[3] = -1
    names = {m.attn.layer_name: slots for m in model}

    def kernels(prof):
        return sorted({e.name.split("<")[0].split("(")[0].replace("void ", "") for e in prof.events()
                       if e.device_type == torch.autograd.DeviceType.CUDA
                       and ("rope" in e.name or "rotary" in e.name or "reshape_and_cache" in e.name)})

    def run(mode=None):
        for m, c0 in zip(model, caches0):
            m.attn.kv_cache.copy_(c0)
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with torch.inference_mode(), set_forward_context(None, vc, slot_mapping=names):
                if mode is not None:
                    with mode:
                        outs = [m(positions, x0) for m in model]
                else:
                    outs = [m(positions, x0) for m in model]
            torch.cuda.synchronize()
        state = outs + [last[i] for i in range(LAYERS)] + [m.attn.kv_cache for m in model]
        return [t.clone() for t in state], kernels(prof)

    def same(a, b):
        return len(a) == len(b) and all(u.shape == v.shape and torch.equal(u.view(torch.int16), v.view(torch.int16))
                                        for u, v in zip(a, b))

    class Ops(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    out = {"impl": type(model[0].attn.impl).__name__, "rope": type(model[0].rotary_emb).__name__}
    ref, out["kernels_unfused"] = run()
    out["changed"] = not same(ref[-LAYERS:], caches0)
    out["installed"] = R.install(model)
    out["live"] = len(hooks.live(R.TAG))
    got, out["kernels_fused"] = run()
    out["fused_equal"] = same(got, ref)
    got, out["kernels_dispatch"] = run(Ops())
    out["dispatch_equal"] = same(got, ref)
    h = model[0].rotary_emb.register_forward_hook(lambda mod, args, res: None)
    got, out["kernels_rope_hook"] = run()
    h.remove()
    out["rope_hook_equal"] = same(got, ref)
    taps = hooks.Hooks()
    taps.patch(model[1].attn, "forward", (lambda f: lambda *a, **k: f(*a, **k))(model[1].attn.forward))
    got, out["kernels_attn_wrap"] = run()
    taps.uninstall()
    out["attn_wrap_equal"] = same(got, ref)
    taps.patch(ops, "reshape_and_cache_flash", (lambda f: lambda *a: f(*a))(ops.reshape_and_cache_flash))
    got, out["kernels_cache_tap"] = run()
    taps.uninstall()
    out["cache_tap_equal"] = same(got, ref)
    got, out["kernels_again"] = run()
    out["again_equal"] = same(got, ref)
    site = model[0].attn.impl.do_kv_cache_update.__self__
    site.pending = (0, 0, 0, 0, 0, 0)
    try:
        model[0].attn.impl.do_kv_cache_update(model[0].attn, last[0][:, :KVH * D], last[0][:, :KVH * D],
                                              model[0].attn.kv_cache, slots)
        out["mismatch_raised"] = False
    except RuntimeError:
        out["mismatch_raised"] = True
    R.uninstall()
    out["live_after"] = len(hooks.live(R.TAG))
    got, out["kernels_after"] = run()
    out["after_equal"] = same(got, ref)
    print(json.dumps(out))
''')


@cuda
def test_the_layers_run_one_launch_and_stand_aside_under_taps():
    """Two layers running vLLM's LlamaAttention.forward over vLLM's llama3 rope and its attention layer on FA (no
    attention metadata, so FA's forward writes zeros, but the cache update runs), in a child process: installed, each layer
    launches `rope_cache_kernel` alone and leaves every output, the qkv buffer and the cache bit for bit as vLLM's two
    kernels do; under a dispatch mode, a forward hook on the rope, a wrapped attention forward or a patched
    `reshape_and_cache_flash` the layer runs vLLM's two kernels; a cache update that isn't the one just written raises;
    uninstalled, nothing is left."""
    pytest.importorskip("vllm")
    r = subprocess.run([sys.executable, "-c", WIRING], check=False, capture_output=True, text=True,
                       env={**os.environ, "VLLM_BATCH_INVARIANT": "1", "VERITY_ROPE_CACHE_FUSED": "1"})
    assert r.returncode == 0, r.stderr[-4000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    stock = ["vllm::reshape_and_cache_flash_kernel", "vllm::rotary_embedding_kernel"]
    assert out["impl"] == "FlashAttentionImpl" and out["changed"], out
    assert out["installed"] is not None and out["installed"]["layers"] == 2 and out["live"] == 4 and out["live_after"] == 0, out
    assert out["kernels_fused"] == out["kernels_again"] == ["rope_cache_kernel"], out
    for k in ("unfused", "dispatch", "rope_hook", "cache_tap", "after"):
        assert out[f"kernels_{k}"] == stock, (k, out)
    assert out["kernels_attn_wrap"] == sorted(stock + ["rope_cache_kernel"]), out
    assert all(out[k] for k in ("fused_equal", "dispatch_equal", "rope_hook_equal", "attn_wrap_equal", "cache_tap_equal",
                                "again_equal", "mismatch_raised", "after_equal")), out


def test_the_pinned_body_is_vllms():
    """`rope_cache._BODY` is the pinned vLLM's `LlamaAttention.forward`, and a body that differs is refused."""
    pytest.importorskip("vllm")
    from vllm.model_executor.models.llama import LlamaAttention

    from verity_vllm.engine import rope_cache as R
    assert R.body_matches(LlamaAttention.forward)

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output
    assert not R.body_matches(forward)
