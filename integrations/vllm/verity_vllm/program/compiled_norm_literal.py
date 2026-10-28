"""Literal Triton transcriptions of Inductor's generated RMSNorm kernels -- the compiled rows' norm relation of record.

The torch-modelled relation (compiled_relations._triton_sum_models: hand-modelled tl.sum order, torch
IEEE division, torch.rsqrt) matched 221 of 224 add1_norm launches and left 3 launches at ONE element / one bf16 ulp each (55-row prefill,
one row per launch, on Qwen).  Probe of record (r16c_sumprobe.py, 8192 rows each width): the model's SUM TREE is exact (0 rows differ)
and its epilogue is exact; the residue is the statement pair `tmp15 = tmp7 / 1536.0; tmp17 = tmp15 + 1e-06`, which the compiler (fp
fusion on) lowers to ONE fma(tmp7, 1/1536, eps) -- reciprocal multiply, add contracted, a single rounding -- while the torch model rounds
twice (`s / N + eps`): kernel rstd == torch.rsqrt(fma(s, 1/N, eps)) on 8192/8192 rows, == torch.rsqrt(s/N + eps) on 7452.  For N = 2048
(Llama) 1/N is a power of two, s*(1/N) is exact and both agree on every row -- which is why the Llama rows never show it.  Rather than model each
such lowering decision (the reduction tree, the contraction, libdevice), as for the SiLU kernel (compiled_relations
`_silu_mul_triton`), the relation of record is therefore the generated source itself, transcribed statement for statement with the two
model constants (row width N, epsilon EPS) and the vocabulary bound (embedding form) as constexpr parameters, and LAUNCHED WITH THE
EXECUTED CONFIG (XBLOCK, R0_BLOCK, num_warps, num_stages) read from the process's own Inductor launchers -- so the reduction tree is the
same lowering of the same tile, not a model of it.  The four forms (vLLM RMSNorm forward_native as Inductor fuses it, Llama-3.2 and
Qwen2.5 alike -- their sources differ only in N, EPS and the vocabulary):

  add1_norm   triton_red_fused_fused_add_rms_norm_t_0    [x, res, w]        -> normed              (x + res in f32; normed = bf16((sum*rstd)*w))
  add2_norm   triton_red_fused_fused_add_rms_norm_t_2    [x, r1, r2, w]     -> res_out, normed     (x + (r1 + r2); res_out = bf16(sum))
  final_norm  triton_red_fused_fused_add_rms_norm_2      [x, r1, r2, w]     -> normed
  embed_norm  triton_red_fused__to_copy_embedding_rms_norm_t_0 [ids, table, w] -> rows, normed     (rows = table[ids] bf16)

The torch variants stay in the record as witnesses (their label names the order that matched); a launch is exact when ANY variant
reproduces every output bit for bit, and the variant that did is named in by_kind.variants_exact."""
from __future__ import annotations

from typing import Any

_K: dict[str, Any] | None = None
_EXEC: dict[str, list[dict[str, Any]]] | None = None
DEFAULT_CONFIG = {"kwargs": {"XBLOCK": 2, "R0_BLOCK": 2048}, "num_warps": 16, "num_stages": 1}   # Inductor's reduction heuristic for size_hints r0_ 2048 (Llama 2048-wide and Qwen 1536-wide rows both, as executed_configs() reports)


def _build() -> dict[str, Any] | None:
    global _K
    if _K is not None:
        return _K or None
    try:
        import triton
        import triton.language as tl
        from triton.language.extra import libdevice
    except Exception:  # noqa: BLE001
        _K = {}
        return None

    @triton.jit
    def add1_norm(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, N: tl.constexpr, EPS: tl.constexpr, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
        r0_numel = N
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        x0 = xindex
        _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_1 + N * x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp2 = tl.load(in_ptr1 + (r0_1 + N * x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp1 = tmp0.to(tl.float32)
            tmp3 = tmp2.to(tl.float32)
            tmp4 = tmp1 + tmp3
            tmp5 = tmp4 * tmp4
            tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
            tmp8 = _tmp7 + tmp6
            _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
        tmp7 = tl.sum(_tmp7, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            r0_1 = r0_index
            tmp9 = tl.load(in_ptr0 + (r0_1 + N * x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp11 = tl.load(in_ptr1 + (r0_1 + N * x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp21 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp10 = tmp9.to(tl.float32)
            tmp12 = tmp11.to(tl.float32)
            tmp13 = tmp10 + tmp12
            tmp14 = tl.full([1, 1], N, tl.float32)
            tmp15 = (tmp7 / tmp14)
            tmp16 = tl.full([1, 1], EPS, tl.float32)
            tmp17 = tmp15 + tmp16
            tmp18 = libdevice.rsqrt(tmp17)
            tmp19 = tmp13 * tmp18
            tmp20 = tmp19.to(tl.float32)
            tmp22 = tmp20 * tmp21
            tl.store(out_ptr1 + (r0_1 + N * x0), tmp22, r0_mask & xmask)

    @triton.jit
    def add2_norm(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, r0_numel, N: tl.constexpr, EPS: tl.constexpr, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
        r0_numel = N
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        x0 = xindex
        _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_1 + N * x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp2 = tl.load(in_ptr1 + (r0_1 + N * x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp4 = tl.load(in_ptr2 + (r0_1 + N * x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp1 = tmp0.to(tl.float32)
            tmp3 = tmp2.to(tl.float32)
            tmp5 = tmp4.to(tl.float32)
            tmp6 = tmp3 + tmp5
            tmp7 = tmp6.to(tl.float32)
            tmp8 = tmp7.to(tl.float32)
            tmp9 = tmp1 + tmp8
            tmp10 = tmp9 * tmp9
            tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
            tmp13 = _tmp12 + tmp11
            _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
            tmp14 = tmp9.to(tl.float32)
            tl.store(out_ptr1 + (r0_1 + N * x0), tmp14, r0_mask & xmask)
        tmp12 = tl.sum(_tmp12, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            r0_1 = r0_index
            tmp15 = tl.load(in_ptr0 + (r0_1 + N * x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp17 = tl.load(in_ptr1 + (r0_1 + N * x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp19 = tl.load(in_ptr2 + (r0_1 + N * x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp32 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp16 = tmp15.to(tl.float32)
            tmp18 = tmp17.to(tl.float32)
            tmp20 = tmp19.to(tl.float32)
            tmp21 = tmp18 + tmp20
            tmp22 = tmp21.to(tl.float32)
            tmp23 = tmp22.to(tl.float32)
            tmp24 = tmp16 + tmp23
            tmp25 = tl.full([1, 1], N, tl.float32)
            tmp26 = (tmp12 / tmp25)
            tmp27 = tl.full([1, 1], EPS, tl.float32)
            tmp28 = tmp26 + tmp27
            tmp29 = libdevice.rsqrt(tmp28)
            tmp30 = tmp24 * tmp29
            tmp31 = tmp30.to(tl.float32)
            tmp33 = tmp31 * tmp32
            tl.store(out_ptr2 + (r0_1 + N * x0), tmp33, r0_mask & xmask)

    @triton.jit
    def final_norm(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, N: tl.constexpr, EPS: tl.constexpr, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
        r0_numel = N
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        x0 = xindex
        _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            r0_1 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_1 + N * x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp2 = tl.load(in_ptr1 + (r0_1 + N * x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp4 = tl.load(in_ptr2 + (r0_1 + N * x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp1 = tmp0.to(tl.float32)
            tmp3 = tmp2.to(tl.float32)
            tmp5 = tmp4.to(tl.float32)
            tmp6 = tmp3 + tmp5
            tmp7 = tmp6.to(tl.float32)
            tmp8 = tmp7.to(tl.float32)
            tmp9 = tmp1 + tmp8
            tmp10 = tmp9 * tmp9
            tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
            tmp13 = _tmp12 + tmp11
            _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tmp12 = tl.sum(_tmp12, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            r0_1 = r0_index
            tmp14 = tl.load(in_ptr0 + (r0_1 + N * x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp16 = tl.load(in_ptr1 + (r0_1 + N * x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp18 = tl.load(in_ptr2 + (r0_1 + N * x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp31 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp15 = tmp14.to(tl.float32)
            tmp17 = tmp16.to(tl.float32)
            tmp19 = tmp18.to(tl.float32)
            tmp20 = tmp17 + tmp19
            tmp21 = tmp20.to(tl.float32)
            tmp22 = tmp21.to(tl.float32)
            tmp23 = tmp15 + tmp22
            tmp24 = tl.full([1, 1], N, tl.float32)
            tmp25 = (tmp12 / tmp24)
            tmp26 = tl.full([1, 1], EPS, tl.float32)
            tmp27 = tmp25 + tmp26
            tmp28 = libdevice.rsqrt(tmp27)
            tmp29 = tmp23 * tmp28
            tmp30 = tmp29.to(tl.float32)
            tmp32 = tmp30 * tmp31
            tl.store(out_ptr1 + (r0_1 + N * x0), tmp32, r0_mask & xmask)

    @triton.jit
    def embed_norm(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, xnumel, r0_numel, N: tl.constexpr, EPS: tl.constexpr, V: tl.constexpr, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
        r0_numel = N
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
        _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            r0_1 = r0_index
            tmp1 = tmp0.to(tl.int64)
            tl.device_assert(((0 <= tmp1) & (tmp1 < V)) | ~(xmask), "index out of bounds: 0 <= tmp1 < V")
            tmp3 = tl.load(in_ptr1 + (r0_1 + N * tmp1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp4 = tmp3.to(tl.float32)
            tmp5 = tmp4 * tmp4
            tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
            tmp8 = _tmp7 + tmp6
            _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
            tl.store(out_ptr0 + (r0_1 + N * x0), tmp3, r0_mask & xmask)
        tmp7 = tl.sum(_tmp7, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            r0_1 = r0_index
            tmp9 = tl.load(out_ptr0 + (r0_1 + N * x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
            tmp18 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp10 = tmp9.to(tl.float32)
            tmp11 = tl.full([1, 1], N, tl.float32)
            tmp12 = (tmp7 / tmp11)
            tmp13 = tl.full([1, 1], EPS, tl.float32)
            tmp14 = tmp12 + tmp13
            tmp15 = libdevice.rsqrt(tmp14)
            tmp16 = tmp10 * tmp15
            tmp17 = tmp16.to(tl.float32)
            tmp19 = tmp17 * tmp18
            tl.store(out_ptr2 + (r0_1 + N * x0), tmp19, r0_mask & xmask)

    _K = {"add1_norm": add1_norm, "add2_norm": add2_norm, "final_norm": final_norm, "embed_norm": embed_norm, "triton": triton}
    return _K


def _scan_executed() -> dict[str, list[dict[str, Any]]]:
    """Executed launch configs of every Inductor kernel bound in THIS process (CachingAutotuner.launchers[*].config, the one(s) left after
    autotuning -- what the CUDA graphs replay), keyed by kernel name.  Same registry loaded_code_objects() reads; no hashing."""
    out: dict[str, list[dict[str, Any]]] = {}
    try:
        from torch._inductor.codecache import PyCodeCache
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    except Exception:  # noqa: BLE001
        return out
    seen: set[int] = set()
    for src_name in ("modules", "modules_no_attr", "cache"):
        coll = getattr(PyCodeCache, src_name, None)
        items = list(coll.values()) if isinstance(coll, dict) else list(coll or [])
        for m in items:
            if id(m) in seen:
                continue
            seen.add(id(m))
            for name, obj in list(vars(m).items()):
                if not isinstance(obj, CachingAutotuner):
                    continue
                kname = getattr(getattr(obj, "fn", None), "__name__", None) or name
                for l in list(getattr(obj, "launchers", None) or []):
                    c = getattr(l, "config", None)
                    if c is None:
                        continue
                    d = {"kwargs": {str(a): int(b) for a, b in (getattr(c, "kwargs", None) or {}).items() if isinstance(b, int)},
                         "num_warps": int(getattr(c, "num_warps", 0) or 0), "num_stages": int(getattr(c, "num_stages", 0) or 0)}
                    lst = out.setdefault(kname, [])
                    if d not in lst:
                        lst.append(d)
    return out


def executed_configs(kernel: str) -> list[dict[str, Any]]:
    """The executed config(s) of `kernel` in this process ([] when unknown: no Inductor registry, or the kernel not bound here)."""
    global _EXEC
    if _EXEC is None or kernel not in _EXEC:
        _EXEC = _scan_executed()
    return list(_EXEC.get(kernel, []))


def _cfg_label(cfg: dict[str, Any], source: str) -> str:
    kw = cfg.get("kwargs") or {}
    return f"triton-literal XBLOCK={kw.get('XBLOCK')} R0_BLOCK={kw.get('R0_BLOCK')} warps={cfg.get('num_warps')} ({source})"


def literal_variants(kind: str, kernel: str, ins: list, eps: float, vocab: int | None = None) -> dict[str, list]:
    """{label: [outputs...]} of the literal transcription of `kind` on the launch's inputs, one per executed config of `kernel` in this
    process (falls back to DEFAULT_CONFIG, labelled 'assumed', when the process has no launcher for it -- offline replays).  {} when Triton
    or CUDA is unavailable or the launch cannot be transcribed (the torch variants then stand alone)."""
    import torch

    K = _build()
    if K is None or kind not in K:
        return {}
    tens = [t for t in ins if t is not None]
    if not tens or not all(t.is_cuda for t in tens):
        return {}
    dev = tens[0].device
    try:
        if kind == "embed_norm":
            ids, table, w = ins[0], ins[1], ins[2]
            M, N = int(ids.numel()), int(table.shape[1])
            V = int(vocab if vocab is not None else table.shape[0])
            xs: list = [ids.reshape(-1).contiguous(), table.contiguous(), w.contiguous()]
        else:
            w = [t for t in tens if t.dim() == 1][-1]
            xs = [t.contiguous() for t in tens if t.dim() != 1]
            M, N = int(xs[0].shape[0]), int(xs[0].shape[1])
            xs = xs + [w.contiguous()]
            V = 0
    except Exception:  # noqa: BLE001
        return {}
    cfgs = executed_configs(kernel)
    src = "executed"
    if not cfgs:
        cfgs, src = [DEFAULT_CONFIG], "assumed"
    out: dict[str, list] = {}
    for cfg in cfgs:
        kw = cfg.get("kwargs") or {}
        xb, r0b = int(kw.get("XBLOCK") or 0), int(kw.get("R0_BLOCK") or 0)
        nw, ns = int(cfg.get("num_warps") or 0), int(cfg.get("num_stages") or 1)
        if xb <= 0 or r0b <= 0 or nw <= 0:
            continue
        grid = (K["triton"].cdiv(M, xb),)
        try:
            if kind == "add1_norm":
                o = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
                K[kind][grid](xs[0], xs[1], xs[2], o, M, N, N=N, EPS=float(eps), XBLOCK=xb, R0_BLOCK=r0b, num_warps=nw, num_stages=ns)
                outs = [o]
            elif kind == "add2_norm":
                o1 = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
                o2 = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
                K[kind][grid](xs[0], xs[1], xs[2], xs[3], o1, o2, M, N, N=N, EPS=float(eps), XBLOCK=xb, R0_BLOCK=r0b, num_warps=nw, num_stages=ns)
                outs = [o1, o2]
            elif kind == "final_norm":
                o = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
                K[kind][grid](xs[0], xs[1], xs[2], xs[3], o, M, N, N=N, EPS=float(eps), XBLOCK=xb, R0_BLOCK=r0b, num_warps=nw, num_stages=ns)
                outs = [o]
            else:
                o0 = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
                o2 = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
                K[kind][grid](xs[0], xs[1], xs[2], o0, o2, M, N, N=N, EPS=float(eps), V=V, XBLOCK=xb, R0_BLOCK=r0b, num_warps=nw, num_stages=ns)
                outs = [o0, o2]
        except Exception:  # noqa: BLE001
            continue
        out[_cfg_label(cfg, src)] = outs
    return out
