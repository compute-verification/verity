"""Gather-staging flush points beyond Llama's lifetime table.

Without them the C2 oracle sees committed identities != the Program's value: `q_norm/out` and `k_norm/out` (RoPE rotates q/k IN
PLACE right after the QK norms, before an o_proj flush) and `post_attention_layernorm/1` (the residual, overwritten in place by the
next layer's fused add-RMSNorm; a fused-MoE block has no down_proj hook to flush it).  So: post-hook flush at q_norm/k_norm,
pre-forward flush at input_layernorm / final norm.

Rule tests need only torch importable (the collector module imports it); the byte-level tests drive the committer's own hook bodies
(`_make_wrapper`) over the C++ collector on a CUDA device and are skipped without one.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.commit.committer import native_collect as NC  # noqa: E402


def _bare(fault: str = "", *, gather: bool = True, nc=None) -> NC.NativeCollectCommitter:
    """A committer instance with only what the hook-body factories read (no engine, no extension build)."""
    obj = NC.NativeCollectCommitter.__new__(NC.NativeCollectCommitter)
    obj.fault, obj.gather, obj.profile = fault, gather, False
    obj._names, obj._name_ids, obj.errors = [], {}, []
    obj._cur = object()                      # "a step is open": the wrappers collect
    obj._nc = nc
    return obj


# ---------------------------------------------------------------- rule level (torch-free apart from the import)

def test_flush_leaves_cover_the_two_new_zero_window_overwriters():
    o = _bare()
    for leaf in ("qkv_proj", "o_proj", "down_proj"):                              # Llama's table (unchanged)
        assert o._flush_leaf(f"model.layers.3.self_attn.{leaf}") or o._flush_leaf(f"model.layers.3.mlp.{leaf}")
    assert o._flush_leaf("model.layers.3.self_attn.q_norm") and o._flush_leaf("model.layers.3.self_attn.k_norm")   # before RoPE (in place)
    assert o._pre_flush_leaf("model.layers.4.input_layernorm") and o._pre_flush_leaf("model.norm")               # before the fused add
    assert not o._pre_flush_leaf("model.layers.4.post_attention_layernorm")       # its /1 is consumed one block later, not by itself
    assert not o._pre_flush_leaf("model.layers.3.self_attn.q_norm")               # `norm` is the leaf `norm`, not a suffix match
    assert not o._flush_leaf("model.layers.4.input_layernorm") and not o._flush_leaf("model.layers.3.mlp.experts")


def test_negative_faults_drop_exactly_the_new_flush_points():
    qk = _bare("late_flush_qk")
    assert not qk._flush_leaf("model.layers.0.self_attn.q_norm") and not qk._flush_leaf("model.layers.0.self_attn.k_norm")
    assert qk._flush_leaf("model.layers.0.self_attn.qkv_proj") and qk._pre_flush_leaf("model.layers.1.input_layernorm")
    nm = _bare("late_flush_norm")
    assert not nm._pre_flush_leaf("model.layers.1.input_layernorm") and not nm._pre_flush_leaf("model.norm")
    assert nm._flush_leaf("model.layers.0.self_attn.q_norm")
    old = _bare("late_flush")                                                       # the late_flush negative is unchanged
    assert not old._flush_leaf("model.layers.0.self_attn.qkv_proj") and old._flush_leaf("model.layers.0.self_attn.q_norm")


def test_flush_03_the_moe_up_plane_leaves_right_after_staging_and_only_that_plane():
    """The fused-MoE `up` plane is overwritten by the block's own w2 launch (cache1/cache3 alias): the source's
    `flush_after_plane("up")` must issue the gather launch on the gather collector; the other planes are flushed by the next layer's
    pre-flush; the arena committer stages nothing; the `late_flush_moe_up` negative drops exactly this flush."""
    calls: list[str] = []

    class FakeNC:
        def flush(self):
            calls.append("flush")

    o = _bare(nc=FakeNC())
    assert o._flush_plane("up") and not any(o._flush_plane(p) for p in ("router_w", "router_ids", "act", "down", "sum"))
    assert o.flush_after_plane("up") is True and calls == ["flush"]
    assert o.flush_after_plane("down") is False and o.flush_after_plane("act") is False and calls == ["flush"]
    assert _bare(nc=FakeNC(), gather=False).flush_after_plane("up") is False        # non-gather: copied at the hook, nothing staged
    neg = _bare("late_flush_moe_up", nc=FakeNC())
    calls.clear()
    assert neg.flush_after_plane("up") is False and calls == []
    assert neg._pre_flush_leaf("model.layers.1.input_layernorm") and neg._flush_leaf("model.layers.0.self_attn.q_norm")   # the norm flush points intact
    assert _bare("late_flush_norm", nc=FakeNC()).flush_after_plane("up") is True   # and the norm negative leaves the up-plane flush alone


def test_moe_source_asks_the_committer_to_flush_the_up_plane_only():
    """The MoE source binds `up` then calls `flush_after_plane("up")`; `act`/`down`/`sum`/router are bound without a flush; a committer
    without the method (an older tree) is tolerated."""
    from verity_vllm.acquire.sources import moe_source as MS

    log: list[tuple] = []

    class Committer:
        _cur = object()
        errors: list = []

        def _acquire(self, name, t, job):
            log.append(("acquire", name))

        def flush_after_plane(self, what):
            log.append(("flush", what))
            return what == "up"

    class Model:
        def named_modules(self):
            return []

    class CudaLike:                                           # what `_bind` reads of a device tensor (no CUDA device needed for the rule)
        device = type("D", (), {"type": "cuda"})()
        shape, dtype = (4, 2), "torch.bfloat16"

        def numel(self):
            return 8

        def element_size(self):
            return 2

    t = CudaLike()
    src = MS.MoeBlockSource(Committer(), Model())
    src.cur_layer = 3
    for what in ("router_w", "router_ids", "up", "act", "down", "sum"):
        src._bind(what, t)
    assert log == [("acquire", "moe/L3/router_w"), ("acquire", "moe/L3/router_ids"), ("acquire", "moe/L3/up"), ("flush", "up"),
                   ("acquire", "moe/L3/act"), ("acquire", "moe/L3/down"), ("acquire", "moe/L3/sum")]
    assert src.stats["plane_flushes"] == 1 and src.declaration_classes()[0]["plane_flushes"] == 1

    class OldCommitter(Committer):
        flush_after_plane = None                              # attribute absent / None: no flush, no error
    log.clear()
    old = MS.MoeBlockSource(OldCommitter(), Model())
    old.cur_layer = 0
    old._bind("up", t)
    assert log == [("acquire", "moe/L0/up")] and old.stats["plane_flushes"] == 0


def test_pre_flush_wraps_only_the_consumers_and_flushes_before_forward():
    calls: list[str] = []

    class FakeNC:
        def flush(self):
            calls.append("flush")

    o = _bare(nc=FakeNC())
    body = lambda *a, **k: calls.append("forward") or "out"                     # noqa: E731

    assert o._with_pre_flush(body, "model.layers.2.self_attn.o_proj") is body    # not a consumer: the body is returned as is
    w = o._with_pre_flush(body, "model.layers.3.input_layernorm")
    assert w is not body and w() == "out" and calls == ["flush", "forward"]
    calls.clear()
    assert _bare(nc=FakeNC(), gather=False)._with_pre_flush(body, "model.norm") is body   # non-gather collectors copy at the hook: no staging
    assert _bare("late_flush_norm", nc=FakeNC())._with_pre_flush(body, "model.norm") is body


# ---------------------------------------------------------------- byte level: the committer's hook bodies over the C++ collector

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().reshape(-1).view(torch.uint8)


class _FusedAddNorm(torch.nn.Module):
    """vLLM's fused_add_rms_norm contract: residual += x and x <- norm(residual), BOTH in place; returns (x, residual) -- the same objects."""

    def forward(self, x, residual):
        residual.add_(x)
        x.copy_(residual * 0.5)
        return x, residual


class _NormThenRope(torch.nn.Module):
    def forward(self, q):
        return q * 0.25


@cuda
@pytest.mark.parametrize("fault", ["", "late_flush_norm"])
def test_residual_of_post_attention_norm_is_committed_before_the_next_fused_add(fault):
    """A fused-MoE decoder layer: post_attention_layernorm -> (MoE, no down_proj hook) -> next layer's input_layernorm.  With the pre-flush
    the committed `/1` is the residual as produced; with `late_flush_norm` (no pre-flush) the read happens at the next
    layer's qkv_proj flush, after the fused add overwrote it -> the committed bytes are the NEXT layer's residual."""
    mod = NC.ext(verbose=False)
    c = mod.Collector()
    c.enable_gather(64)
    o = _bare(fault, nc=c)
    arena = torch.zeros(1 << 14, dtype=torch.uint8, device="cuda")
    post = _FusedAddNorm().cuda()
    nxt = _FusedAddNorm().cuda()
    w_post = o._make_wrapper(post.forward, "model.layers.0.post_attention_layernorm", False)
    w_next = o._make_wrapper(nxt.forward, "model.layers.1.input_layernorm", False)
    x = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    res = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    c.begin_step(plan_key=1, arena_ptr=arena.data_ptr(), arena_bytes=arena.numel(), use_plan=True)
    h, res = w_post(x, res)                                  # stages /0 (h) and /1 (res): arena [0,512) and [512,1024)
    produced_res = res.clone()
    y = h * 3.0                                              # the MoE block: a NEW tensor, nothing hooked, nothing flushed
    h2, res2 = w_next(y, res)                                # pre-flush here (fix) -> then residual += y in place
    c.flush()                                                # the next layer's qkv_proj flush (always there)
    c.end_step()
    torch.cuda.synchronize()
    committed_res = arena[512:1024]
    if fault:
        assert not torch.equal(committed_res, _as_bytes(produced_res)), "the late read must not reproduce the produced residual"
        assert torch.equal(committed_res, _as_bytes(res2)), "late read = the NEXT layer's residual"
    else:
        assert torch.equal(committed_res, _as_bytes(produced_res))
        assert not torch.equal(committed_res, _as_bytes(res2))


@cuda
@pytest.mark.parametrize("fault", ["", "late_flush_moe_up"])
def test_moe_up_plane_is_committed_before_the_blocks_w2_launch_overwrites_its_workspace(fault):
    """vLLM's TritonExperts: cache1 (up) and cache3 (down) are `_resize_cache(workspace2, ...)` views of ONE buffer.
    The MoE source stages `up` after the w13 launch; the block's own w2 launch then writes `down` into the same bytes; the next hook (the
    next layer's `input_layernorm` pre-flush) is the first flush.  With the up-plane flush the committed `up` is the w13 output;
    with `late_flush_moe_up` (no up-plane flush) it is the same layer's `down`, byte for byte."""
    from verity_vllm.acquire.sources import moe_source as MS

    mod = NC.ext(verbose=False)
    c = mod.Collector()
    c.enable_gather(64)
    o = _bare(fault, nc=c)
    o._py_committed = 0
    arena = torch.zeros(1 << 14, dtype=torch.uint8, device="cuda")

    class Model:
        def named_modules(self):
            return []

    src = MS.MoeBlockSource(o, Model())
    src.cur_layer = 0
    workspace2 = torch.zeros(8 * 256, dtype=torch.bfloat16, device="cuda")          # the block's shared workspace (2I == H: full alias)
    cache1 = workspace2.flatten()[: 8 * 256].view(8, 256)                           # up   = _resize_cache(workspace2, (M*topk, 2I))
    cache3 = workspace2.flatten()[: 8 * 256].view(8, 256)                           # down = _resize_cache(workspace2, (M*topk, H))
    x = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
    c.begin_step(plan_key=11, arena_ptr=arena.data_ptr(), arena_bytes=arena.numel(), use_plan=True)
    cache1.copy_(x * 2.0)                                     # the w13 launch writes up
    src._bind("up", cache1)                                   # staged at [0, 4096) (+ the up-plane flush)
    produced_up = cache1.clone()
    act = torch.nn.functional.silu(cache1[:, :128]) * cache1[:, 128:]                 # silu_and_mul -> cache2 (another workspace)
    src._bind("act", act)                                     # [4096, 6144)
    cache3.copy_(torch.cat([act, act], dim=1) * 0.5)          # the w2 launch writes down INTO THE SAME BYTES as up
    src._bind("down", cache3)                                 # [6144, 10240)
    c.flush()                                                 # the next layer's input_layernorm pre-flush (always there)
    c.end_step()
    torch.cuda.synchronize()
    committed_up, committed_down = arena[0:4096], arena[6144:10240]
    assert torch.equal(committed_down, _as_bytes(cache3))                            # down is right either way
    if fault:
        assert src.stats["plane_flushes"] == 0
        assert torch.equal(committed_up, committed_down) and not torch.equal(committed_up, _as_bytes(produced_up)), \
            "the late read must commit the same layer's down bytes under the up name"
    else:
        assert src.stats["plane_flushes"] == 1
        assert torch.equal(committed_up, _as_bytes(produced_up)) and not torch.equal(committed_up, committed_down)


@cuda
@pytest.mark.parametrize("fault", ["", "late_flush_qk"])
def test_qk_norm_output_is_committed_before_rope_rotates_it_in_place(fault):
    """QK-norm attention (OLMoE / Qwen3): q = q_norm(q); rotary_emb(positions, q, k) IN PLACE; ... o_proj (flush).  With the q_norm/k_norm
    flush the committed `q_norm/out` is the norm's output; with `late_flush_qk` it is the rotated q (first
    differing element = the first token at a non-zero position)."""
    mod = NC.ext(verbose=False)
    c = mod.Collector()
    c.enable_gather(64)
    o = _bare(fault, nc=c)
    arena = torch.zeros(1 << 14, dtype=torch.uint8, device="cuda")
    qn = _NormThenRope().cuda()
    w_q = o._make_wrapper(qn.forward, "model.layers.0.self_attn.q_norm", False)
    q = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    c.begin_step(plan_key=7, arena_ptr=arena.data_ptr(), arena_bytes=arena.numel(), use_plan=True)
    qn_out = w_q(q)                                          # stages q_norm/out at [0,512) (+ flush with the fix)
    produced = qn_out.clone()
    qn_out.mul_(-1.0)                                        # RoPE stand-in: in place on the norm's output
    c.flush()                                                # the o_proj flush
    c.end_step()
    torch.cuda.synchronize()
    got = arena[:512]
    if fault:
        assert torch.equal(got, _as_bytes(qn_out)) and not torch.equal(got, _as_bytes(produced))
    else:
        assert torch.equal(got, _as_bytes(produced))
