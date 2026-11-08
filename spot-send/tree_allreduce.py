"""One node's TP all-reduce arithmetic on another: the all-reduces the serving node's NVSwitch reduced are summed in its multimem
reduction tree, in software, so a replay on a node whose NVSwitch reduces in another order computes the served bits.

On the served H200 node, `CudaCommunicator.all_reduce` hands a bf16 all-reduce FlashInfer's 0.5 MiB doesn't take to torch's
symmetric memory (``multimem.ld_reduce ... acc::f32``, to 64 MiB) and past that to NCCL's NVLS; the node's capture found both reduce
every element in one tree of f32 adds, then round to bf16 (`AllReduceMultimem_v1` at the node's TREES, FTZ off). This plugin keeps
FlashInfer's route, the custom all-reduce and the fused all-reduce + RMSNorm as they are (their order is in software, the same on every node),
and replaces the other two with an NCCL all-gather (a copy) and that tree's f32 adds on every rank, rounded once to bf16. On a
replay node without NVLink multicast, three more routes differ from the serving node's, and the plugin restores them:
- vLLM leaves the custom all-reduce its full size, so it gets the serving node's size cap back (`served_custom_cap`).
- FlashInfer's workspace falls back from ``mnnvl`` to ``trtllm``, which vLLM calls with bf16 accumulation. So each call
  FlashInfer takes is summed as ``mnnvl`` sums it (`reduce_from_zero`).
- vLLM's all-reduce + RMSNorm fusion (``fuse_allreduce_rms``, steps of at most 0.5 MiB a rank) runs FlashInfer's fused kernel
  on that workspace too, with trtllm's sums and norm. So each fused call is computed as ``mnnvl``'s one-shot or two-shot
  kernels compute it (`mnnvl_add_rmsnorm`).
It changes what the replay node computes, never the served record: a replay under it says so.

    python tree_allreduce.py site DIR --tree '(((((((0,2),3),1),5),6),4),7)' --source WHERE
        # DIR on PYTHONPATH registers the plugin (vLLM general plugin ``verity_tree_allreduce``) with that tree
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

CONFIG = "tree_allreduce.json"
Tree = int | tuple


def parse_tree(s: str) -> Tree:
    """A tree as the probe prints it, ``(((0,2),3),1)``: nested pairs of ranks."""
    t = ast.literal_eval(s)

    def check(x: Any) -> Tree:
        if isinstance(x, int):
            return x
        if isinstance(x, tuple) and len(x) == 2:
            return (check(x[0]), check(x[1]))
        raise ValueError(f"{s!r}: a tree is a rank or a pair of trees")
    return check(t)


def leaves(t: Tree) -> list[int]:
    return [t] if isinstance(t, int) else leaves(t[0]) + leaves(t[1])


def buffers(t: Tree) -> int:
    """The f32 rows `reduce_gathered` sums ``t`` in: one, and one more for each nesting of a right subtree that isn't a rank."""
    if isinstance(t, int):
        return 1
    return max(buffers(t[0]), 1 + buffers(t[1]) if isinstance(t[1], tuple) else 1)


def reduce_gathered(gathered: Any, world: int, t: Tree, out: Any = None, f32: Any = None) -> Any:
    """A rank-major all-gather of a bf16 tensor (``world * n`` words) reduced in ``t``, a node its two subtrees' f32 sum, then
    rounded once to bf16, into ``out`` when given.  The sums are in place in ``f32``'s rows (`buffers` of at least ``n`` words; an
    f32 ``add_`` of a bf16 row adds its exact f32 value), so with ``out`` and ``f32`` given a call allocates nothing, which a
    CUDA graph's capture needs: its allocations come from the graph's own pool, past what the serve's KV cache leaves."""
    import torch
    rows = gathered.view(world, -1)
    n = rows.shape[1]
    if f32 is None:
        f32 = torch.empty(buffers(t), n, dtype=torch.float32, device=gathered.device)

    def go(x: Tree, k: int) -> Any:
        acc = f32[k, :n]
        if isinstance(x, int):
            return acc.copy_(rows[x])
        go(x[0], k)
        return acc.add_(rows[x[1]] if isinstance(x[1], int) else go(x[1], k + 1))
    r = go(t, 0)
    return r.to(torch.bfloat16) if out is None else out.copy_(r)


def reduce_from_zero(gathered: Any, world: int, out: Any = None, f32: Any = None) -> Any:
    """A rank-major all-gather reduced as FlashInfer's ``mnnvl`` all-reduce does at up to 16 ranks (`reduceOneshotDeterministic`,
    and the two-shot `reduceLamportRanksChunked` in one chunk): an f32 accumulator from +0, each rank's bf16 value added in rank
    order, rounded once. Starting from +0 rather than rank 0's value matters only for the sign of an all-(-0) sum."""
    import torch
    rows = gathered.view(world, -1)
    n = rows.shape[1]
    acc = (torch.empty(n, dtype=torch.float32, device=gathered.device) if f32 is None else f32[0, :n]).zero_()
    for r in range(world):
        acc.add_(rows[r])
    return acc.to(torch.bfloat16) if out is None else out.copy_(acc)


def mnnvl_grid(tokens: int, dim: int, sms: int, elts: int = 8) -> tuple[int, int]:
    """FlashInfer's `adjustGridConfig` with clusters, as its ``mnnvl`` fused kernels launch over ``tokens`` rows of ``dim``:
    (threads a block, blocks in a row's cluster). Thread ``p`` of a row's cluster holds elements ``[elts*p, elts*p + elts)``.
    A shape whose launch the reductions below don't model (a partial warp, padding, several loads a thread) is refused."""
    need = -(-dim // elts)
    cluster = 8
    while need % cluster and cluster > 1:
        cluster //= 2
    most = cluster
    block = -(-need // cluster)
    while block < 128 and cluster >= 2:
        block, cluster = block * 2, cluster // 2
    while tokens * cluster > sms and cluster > 1 and block <= 512:
        block, cluster = block * 2, cluster // 2
    while cluster < most:
        b = -(-need // (cluster * 2))
        if b < 64 or tokens * cluster * 2 > sms:
            break
        block, cluster = b, cluster * 2
    if block > 1024 or block % 32 or block * cluster * elts != dim:
        raise NotImplementedError(f"tree-allreduce: mnnvl's launch for {tokens} rows of {dim} ({block} threads x {cluster} "
                                  "blocks) is one the emulation doesn't model")
    return block, cluster


def butterfly(x: Any) -> Any:
    """A warp's ``__shfl_xor_sync`` sum over the last axis (32 lanes): every lane ends with the same tree of f32 adds."""
    for h in (16, 8, 4, 2, 1):
        x = x[..., :h] + x[..., h:2 * h]
    return x[..., 0]


def mnnvl_add_rmsnorm(ar: Any, residual: Any, gamma: Any, eps: float, weight_bias: float, oneshot: bool, sms: int,
                      scale: Any) -> tuple[Any, Any]:
    """``(normed, h)`` as FlashInfer's ``mnnvl`` fused all-reduce + residual + RMSNorm computes them from the bf16 sum ``ar``
    (`oneshotAllreduceFusionKernel` with RMSNorm, or two-shot's `rmsNormLamport`), read from the source and its SASS:
    ``h = ar + residual`` rounded to bf16; each thread's 8 squares summed in order from +0 (one-shot squares in bf16, two-shot
    in f32 by FFMA, exact for a bf16 value); a warp butterfly, a butterfly over the block's warp sums, then the cluster's block
    sums in order; ``scale(sum, dim, eps)`` for ``rsqrtf(sum / dim + eps)``; then one-shot ``(h * r) * (wb + g)`` and two-shot
    ``r * (h * (wb + g))``, each product f32, rounded to bf16. Subnormals, which the kernels flush, are taken not to occur."""
    import torch
    t, dim = ar.shape
    block, cluster = mnnvl_grid(t, dim, sms)
    h = (ar.float() + residual.float()).to(torch.bfloat16)
    v = h.float()
    sq = ((v * v).to(torch.bfloat16).float() if oneshot else v * v).view(t, dim // 8, 8)
    ts = sq[..., 0].clone()
    for j in range(1, 8):
        ts += sq[..., j]
    warps = butterfly(ts.view(t, cluster, block // 32, 32))
    blocks = butterfly(torch.nn.functional.pad(warps, (0, 32 - block // 32)))
    full = blocks[:, 0].clone()
    for i in range(1, cluster):
        full += blocks[:, i]
    r = scale(full, dim, eps)[:, None]
    g = gamma.float() + weight_bias
    out = (v * r) * g if oneshot else r * (v * g)
    return out.to(torch.bfloat16), h


def device_sms(device: Any) -> int:
    """The multiprocessors FlashInfer's `GetCudaMultiProcessorCount` sizes the launch by; the serving GPU's, as this node has the
    same (H200: 132)."""
    import torch
    return torch.cuda.get_device_properties(device).multi_processor_count


_SCALE: Any = None


def mufu_scale() -> Any:
    """``scale(sum, dim, eps)``: FlashInfer's ``rsqrtf(sum / dim + eps)`` under ``-use_fast_math``, which its SASS computes as
    ``MUFU.RSQ(FFMA(MUFU.RCP(dim), sum, eps))``; each step is the PTX instruction that compiles to it."""
    global _SCALE, tl
    if _SCALE is not None:
        return _SCALE
    import torch
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["n", "dim"])
    def kernel(s_ptr, o_ptr, n, dim, eps, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        s = tl.load(s_ptr + i, mask=m, other=1.0)
        d = tl.full([BLOCK], dim, tl.int32).to(tl.float32)
        e = tl.full([BLOCK], eps, tl.float32)
        r = tl.inline_asm_elementwise(
            "{ .reg .f32 t; rcp.approx.ftz.f32 t, $2; fma.rn.ftz.f32 t, t, $1, $3; rsqrt.approx.ftz.f32 $0, t; }",
            "=f,f,f,f", [s, d, e], dtype=tl.float32, is_pure=True, pack=1)
        tl.store(o_ptr + i, r, mask=m)

    def scale(s, dim, eps):
        s = s.contiguous()
        o = torch.empty_like(s)
        kernel[(triton.cdiv(s.numel(), 256),)](s, o, s.numel(), dim, eps, BLOCK=256)
        return o
    _SCALE = scale
    return scale


def install_fused(t: Tree, device: Any) -> None:
    """Wrap ``flashinfer.comm.allreduce_fusion``, which vLLM's fused all-reduce + RMSNorm op looks up at each call, and compile
    the scale kernel now: this runs in vLLM's eager profiling forward, before any CUDA graph is captured."""
    import torch
    if device.type != "cuda":
        return
    try:
        import flashinfer.comm as fc
    except ImportError:
        return
    if getattr(fc.allreduce_fusion, "__wrapped__", None) is None:
        fc.allreduce_fusion = fused_from_mnnvl(fc.allreduce_fusion, t)
    mufu_scale()(torch.ones(1, dtype=torch.float32, device=device), 6144, 1e-5)


def fused_from_mnnvl(orig: Any, t: Tree) -> Any:
    """``flashinfer.comm.allreduce_fusion`` with the all-reduce + residual + RMSNorm calls a workspace other than the serving
    node's ``mnnvl`` gets computed as ``mnnvl`` computes them: an all-gather summed from +0 in rank order, rounded once (two-shot
    then clears a -0), and `mnnvl_add_rmsnorm` with the strategy ``mnnvl``'s AUTO picks for the shape. Any other pattern on such
    a workspace is refused."""
    def allreduce_fusion(*args, **kw):
        ws = kw.get("workspace")
        if args or ws is None or getattr(ws, "backend", None) == SERVED_FI_BACKEND:
            return orig(*args, **kw)
        import flashinfer.comm as fc
        from flashinfer.comm.trtllm_mnnvl_ar import MNNVLAllreduceFusionStrategy as S
        from vllm.distributed.parallel_state import get_tp_group
        if kw.get("pattern") != fc.AllReduceFusionPattern.kARResidualRMSNorm or any(
                kw.get(k) is not None for k in ("quant_out", "scale_out", "scale_factor")):
            raise RuntimeError(f"tree-allreduce: FlashInfer fusion pattern {kw.get('pattern')} on a {ws.backend} workspace, "
                               "which the mnnvl emulation doesn't cover")
        x, res, gamma, eps = kw["input"], kw["residual_in"], kw["rms_gamma"], kw["rms_eps"]
        norm_out, res_out = kw["norm_out"], kw["residual_out"]
        if eps is None or norm_out is None or res_out is None or x.dim() != 2:
            raise RuntimeError("tree-allreduce: a fused all-reduce + RMSNorm call without vLLM's outputs, eps or 2-D input")
        comm = get_tp_group().device_communicator
        if not getattr(comm, "_tree_fused_said", False):
            comm._tree_fused_said = True
            print(f"[tree-allreduce] FlashInfer's fused all-reduce + RMSNorm runs on a {ws.backend} workspace on this node: "
                  f"computed as the serving node's {SERVED_FI_BACKEND} kernels compute it", file=sys.stderr, flush=True)
        ar = gather_reduce(comm, x, t, lambda g, w, out, f32: reduce_from_zero(g, w, out, f32))
        oneshot = S.select_strategy(comm.world_size, x.shape[0], x.shape[1], x.dtype) == S.ONESHOT
        if not oneshot:
            ar = ar + 0.0
        normed, h = mnnvl_add_rmsnorm(ar, res, gamma, eps, kw.get("weight_bias") or 0.0, oneshot, device_sms(x.device),
                                      mufu_scale())
        res_out.copy_(h)
        norm_out.copy_(normed)
        return norm_out
    allreduce_fusion.__wrapped__ = orig
    return allreduce_fusion


def config(d: Path | None = None) -> dict | None:
    p = (d or Path(__file__).resolve().parent) / CONFIG
    return json.loads(p.read_text()) if p.is_file() else None


_TREE: Tree | None = None


def register() -> None:
    """The ``vllm.general_plugins`` entry: wraps `CudaCommunicator.all_reduce` with the configured tree."""
    global _TREE
    cfg = config()
    if cfg is None or _TREE is not None:
        return
    _TREE = parse_tree(cfg["tree"])
    from vllm.distributed.device_communicators import cuda_communicator as cc
    cc.CudaCommunicator.all_reduce = wrap(cc.CudaCommunicator.all_reduce, _TREE)
    print(f"[tree-allreduce] symmetric-memory and NCCL all-reduces sum in {cfg['tree']} ({cfg.get('source', '')})",
          file=sys.stderr, flush=True)


CHUNK = 1 << 22  # words a rank gathers at once: elementwise, so only the workspace's size depends on it
UNMEASURED = ("qr_comm", "fi_pcie_ipc_ar_comm", "aiter_ar_comm")


def served_custom_cap(comm: Any) -> int | None:
    """Give the custom all-reduce the size cap it had on the serving node, which had torch's symmetric memory: vLLM then caps
    it at `CUSTOM_ALL_REDUCE_MAX_SIZES` (256 KB on H200 at TP8, under FlashInfer's 0.5 MiB). On a replay node without multicast
    vLLM disables symmetric memory and leaves the custom all-reduce its full size (8 MiB), so it would take calls the serving
    node summed in the tree. Returns the cap set, or None when the cap already matches."""
    ca, sm = comm.ca_comm, getattr(comm, "symm_mem_comm", None)
    if ca is None or ca.disabled or (sm is not None and not sm.disabled):
        return None
    from vllm.distributed.device_communicators.all_reduce_utils import CUSTOM_ALL_REDUCE_MAX_SIZES
    from vllm.platforms import current_platform
    cap = current_platform.get_device_capability()
    m = CUSTOM_ALL_REDUCE_MAX_SIZES.get(cap.as_version_str() if cap is not None else "", {}).get(comm.world_size)
    if m is None or ca.max_size <= m:
        return None
    ca.max_size = m
    return m


SERVED_FI_BACKEND = "mnnvl"


def fi_backend() -> str | None:
    """The backend of vLLM's FlashInfer all-reduce workspace in this process, once a call has created it."""
    fam = sys.modules.get("vllm.distributed.device_communicators.flashinfer_all_reduce")
    ws = getattr(fam, "_fi_ar_workspace", None)
    return None if ws is None else ws.backend


def gather_reduce(comm: Any, input_: Any, t: Tree, reduce: Any) -> Any:
    """``input_`` all-gathered across ``comm``'s ranks in CHUNK-word pieces and each piece reduced by ``reduce(gathered, world,
    out, f32)``, in the communicator's one workspace."""
    import torch
    nc, w = comm.pynccl_comm, comm.world_size
    flat = input_.contiguous().view(-1)
    if getattr(comm, "_tree_ws", None) is None:
        # vLLM's profiling forward is eager and precedes every graph capture, so this is allocated outside the graphs' pools
        comm._tree_ws = (torch.empty(w * CHUNK, dtype=flat.dtype, device=flat.device),
                         torch.empty(buffers(t), CHUNK, dtype=torch.float32, device=flat.device))
    gws, f32 = comm._tree_ws
    out = torch.empty_like(flat)
    for i in range(0, flat.numel(), CHUNK):
        x = flat[i:i + CHUNK]
        g = gws[:w * x.numel()]
        if nc is not None and not nc.disabled:
            nc.all_gather(g, x)
        else:
            torch.distributed.all_gather_into_tensor(g, x, group=comm.device_group)
        reduce(g, w, out[i:i + x.numel()], f32)
    return out.view(input_.shape)


def wrap(orig: Any, t: Tree) -> Any:
    """`CudaCommunicator.all_reduce` with the calls vLLM hands torch's symmetric memory or NCCL summed in ``t``. The custom
    all-reduce's calls (an order in software, the same on every node) and other dtypes go to vLLM, and so do FlashInfer's while
    its workspace is the serving node's ``mnnvl``; under another backend they are summed as ``mnnvl`` sums them. A route the
    tree wasn't measured with (NCCL's symmetric memory, quick reduce, FlashInfer's PCIe IPC, aiter) is refused."""
    def all_reduce(self, input_):
        import torch
        if not getattr(self, "_tree_cap_set", False):
            self._tree_cap_set = True
            m = served_custom_cap(self)
            if m is not None:
                print(f"[tree-allreduce] no symmetric memory on this node: the custom all-reduce is capped at {m} bytes, as on the "
                      "serving node", file=sys.stderr, flush=True)
            install_fused(t, input_.device)
        if input_.dtype != torch.bfloat16:
            return orig(self, input_)
        fi, ca = self.fi_ar_comm, self.ca_comm
        if fi is not None and not fi.disabled and fi.should_use_fi_ar(input_):
            b = fi_backend()
            if b == SERVED_FI_BACKEND:
                return orig(self, input_)
            if b is None:
                raise RuntimeError("tree-allreduce: FlashInfer took a call with no all-reduce workspace to read its backend from")
            if not getattr(self, "_tree_fi_said", False):
                self._tree_fi_said = True
                print(f"[tree-allreduce] FlashInfer's all-reduce workspace is {b} on this node: its calls are summed as the "
                      f"serving node's {SERVED_FI_BACKEND} sums them", file=sys.stderr, flush=True)
            return gather_reduce(self, input_, t, lambda g, w, out, f32: reduce_from_zero(g, w, out, f32))
        if ca is not None and not ca.disabled and ca.should_custom_ar(input_):
            return orig(self, input_)
        from vllm.distributed.device_communicators.pynccl_allocator import is_symmetric_memory_enabled
        on = [k for k in UNMEASURED if getattr(self, k, None) is not None and not getattr(getattr(self, k), "disabled", False)]
        if is_symmetric_memory_enabled():
            on.append("NCCL's symmetric memory")
        if on:
            raise RuntimeError(f"tree-allreduce: {', '.join(on)} on, a route the tree wasn't measured with")
        if sorted(leaves(t)) != list(range(self.world_size)):
            raise RuntimeError(f"tree-allreduce: tree {t} is not over the group's {self.world_size} ranks")
        return gather_reduce(self, input_, t, lambda g, w, out, f32: reduce_gathered(g, w, t, out, f32))
    all_reduce.__wrapped__ = orig
    return all_reduce


def site(d: Path, tree: str, source: str) -> Path:
    """A directory that, on PYTHONPATH, registers this file as the ``vllm.general_plugins`` entry ``verity_tree_allreduce``."""
    import shutil
    parse_tree(tree)
    info = d / "verity_tree_allreduce-0.0.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: verity-tree-allreduce\nVersion: 0.0\n")
    (info / "entry_points.txt").write_text("[vllm.general_plugins]\nverity_tree_allreduce = tree_allreduce:register\n")
    shutil.copyfile(__file__, d / "tree_allreduce.py")
    (d / CONFIG).write_text(json.dumps({"tree": tree, "source": source}, indent=1))
    return d


def main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="tree_allreduce.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("site", help="write the plugin's site directory with its tree")
    s.add_argument("dir", type=Path)
    s.add_argument("--tree", required=True, help="the serving node's multimem reduction tree, as its probe printed it")
    s.add_argument("--source", required=True, help="where the tree was read (the probe's record)")
    a = ap.parse_args(argv)
    print(site(a.dir, a.tree, a.source))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
