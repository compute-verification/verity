"""Export-compatibility shims for idioms in the pinned vLLM source that non-strict `torch.export` cannot trace.
Each shim is generic (names no model, no layer, no op sequence) and semantics-preserving for inference; the REPORT
counts them as structural scaffolding.

1. `Tensor.data` reads inside forward (`self.weight.data` in `RMSNorm.forward_cuda`, `layernorm.py:104-113`): under
   non-strict export `.data` yields an untracked FakeTensor that export then tries to lift as a constant
   ("We found a fake tensor in the exported program constant's list"). For fake tensors the shim returns `detach()`
   — the same value, recorded as `aten.detach` of the parameter placeholder.
2. The declared target applied (`apply_target_profile`).  3. vLLM's MoE dispatcher ops traced through (`moe_ops_traced`).
4. DeepSeek sparse attention's three kernels as leaf ops (`dsa_ops_leaf`).  5. The MoE gate's cuteDSL GEMM as a leaf op
   (`moe_gate_leaf`).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager

import torch

from verity_vllm.engine import env as engine_env, hooks

_ORIG_DATA = torch._C.TensorBase.data


def _is_fake(t) -> bool:
    try:
        return torch._subclasses.fake_tensor.is_fake(t) or t.device.type == "meta"
    except Exception:  # noqa: BLE001
        return False


@contextmanager
def tensor_data_as_detach():
    def fget(self):
        if _is_fake(self):
            return self.detach()
        return _ORIG_DATA.__get__(self, type(self))

    def fset(self, v):
        _ORIG_DATA.__set__(self, v)

    with hooks.patched(torch.Tensor, "data", property(fget, fset)):
        yield


_TUNED_TABLE_MODULES = ("vllm.model_executor.determinism.batch_invariant_configs", "vllm.model_executor.determinism.batch_invariant")
_NUM_CU_MODULES = ("vllm.utils.platform_utils", "vllm.model_executor.determinism.batch_invariant")


def _reset_tuned_table_cache(into: hooks.Hooks) -> None:
    """vLLM caches the tuned-GEMM table per process (`_TUNED_MATMUL_CONFIGS_RESOLVED/_FOR_DEVICE`); force re-resolution under the
    values now applied, with the previous state restored when `into` is uninstalled."""
    import importlib
    for modname in _TUNED_TABLE_MODULES:
        try:
            m = importlib.import_module(modname)
        except Exception:  # noqa: BLE001
            continue
        if hasattr(m, "_TUNED_MATMUL_CONFIGS_RESOLVED"):
            into.patch(m, "_TUNED_MATMUL_CONFIGS_RESOLVED", False)
            into.patch(m, "_TUNED_MATMUL_CONFIGS_FOR_DEVICE", None)


@contextmanager
def apply_target_profile(profile):
    """2. Apply the declared `TargetProfile` to the pinned vLLM for the whole construction (instantiation, export, derive):

    * `current_platform.get_device_capability()` answers `profile.compute_capability` — this selects the FA version
      (`get_flash_attn_version`), the batch-invariant GEMM override family and the tuned-GEMM table (`resolve_tuned_matmul_configs`);
    * `num_compute_units()` answers `profile.num_sms` — vLLM's batch-invariant GEMM sizes its persistent grid by it
      (`matmul_persistent`), and the GEMM rule checks every launch against the same profile value.

    The builder constructs for the DECLARED target on any host: the patch is applied whether or not a GPU is visible (a visible
    GPU's own answers are recorded by the driver under `target.host`, never used). There is no other place these two values are set;
    `read_back_applied()` inside this context returns what vLLM now sees, and the driver refuses if that differs from the profile.
    Nesting is safe (LIFO restore)."""
    import importlib
    from vllm.platforms import current_platform
    from vllm.platforms.interface import DeviceCapability
    major, minor = profile.compute_capability
    with hooks.Hooks() as h:
        h.patch(type(current_platform), "get_device_capability",
                classmethod(lambda c, device_id=0, _M=major, _m=minor: DeviceCapability(major=_M, minor=_m)))
        _reset_tuned_table_cache(h)
        for modname in _NUM_CU_MODULES:
            try:
                m = importlib.import_module(modname)
            except Exception:  # noqa: BLE001
                continue
            if hasattr(m, "num_compute_units"):
                h.patch(m, "num_compute_units", lambda device_id=None, _n=profile.num_sms: _n)
        yield


_CUDA_PLATFORM = "vllm.platforms.cuda.CudaPlatform"


def _answer_cuda(platforms) -> None:
    hooks.patch(platforms, "resolve_current_platform_cls_qualname", lambda: _CUDA_PLATFORM)


class _AnswerCudaLoader:
    """`vllm.platforms`' own loader, then `_answer_cuda` on the executed module: before anything can read `current_platform`."""

    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module) -> None:
        self.inner.exec_module(module)
        _answer_cuda(module)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class _AnswerCudaFinder:
    def find_spec(self, name, path, target=None):
        if name != "vllm.platforms":
            return None
        import importlib.machinery
        sys.meta_path.remove(self)
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _AnswerCudaLoader(spec.loader)
        return spec


def select_cuda_platform(world: int = 1) -> str:
    """A declared target is a CUDA device: vLLM's platform resolution answers `vllm.platforms.cuda.CudaPlatform` (the qualname its CUDA
    plugin returns on a GPU host) instead of probing NVML for it.  On a GPU-free Build host NVML is absent and that probe resolves a
    non-CUDA platform.  `import vllm` itself resolves `current_platform` (`torch_utils.PIN_MEMORY`), so before vLLM is imported the answer
    is installed as `vllm.platforms` finishes executing; the class vLLM then instantiates is the GPU host's (`CudaPlatform`: the
    NVML-backed class when NVML answers, the torch-backed one when it does not), and `apply_target_profile` answers the target's
    capability on either.  The target is `world` such devices: the platform's `device_count()` answers `world`, not the host's visible
    count, so `ParallelConfig`'s "world size larger than the available GPUs" check passes on a GPU-free host and the executor backend
    resolves to `mp`, as on a host with the `world` GPUs the Commit runs on.  Build path only: serving resolves its platform and counts
    its devices as before.  Raises when `current_platform` already resolved to a non-CUDA platform."""
    # vLLM freezes export knobs at import (`VLLM_USE_LAYERNAME` -> `torch_utils._USE_LAYERNAME`): pin them before importing vLLM here
    engine_env.pin(engine_env.EXPORT)
    loaded = sys.modules.get("vllm.platforms")
    if loaded is None:
        sys.meta_path.insert(0, _AnswerCudaFinder())
    elif loaded._current_platform is None:
        _answer_cuda(loaded)
    from vllm.platforms import current_platform
    if not current_platform.is_cuda():
        raise RuntimeError(f"vLLM resolved {type(current_platform).__qualname__} before the declared CUDA target could select its platform")
    hooks.patch(type(current_platform), "device_count", classmethod(lambda cls, _n=int(world): _n))
    return type(current_platform).__qualname__


def read_back_applied() -> dict:
    """What the pinned vLLM answers RIGHT NOW for the target facts the construction consults — the manifest's `target.applied` block
    is this record, taken inside `apply_target_profile`, never a literal. Absent answers are recorded as None (a mismatch)."""
    import importlib
    out: dict = {"compute_capability": None, "num_sms": None, "batch_invariant": None, "vllm_version": None, "torch_version": torch.__version__,
                 "triton_version": None, "tuned_gemm_arch_family": None}
    try:
        from vllm.platforms import current_platform
        cap = current_platform.get_device_capability()
        out["compute_capability"] = [int(cap.major), int(cap.minor)] if cap is not None else None
    except Exception as e:  # noqa: BLE001
        out["compute_capability_error"] = f"{type(e).__name__}: {e}"
    for modname in _NUM_CU_MODULES:
        try:
            m = importlib.import_module(modname)
            if hasattr(m, "num_compute_units"):
                out["num_sms"] = int(m.num_compute_units(0))
                break
        except Exception as e:  # noqa: BLE001
            out["num_sms_error"] = f"{type(e).__name__}: {e}"
    try:
        import vllm
        import vllm.envs as envs
        out["vllm_version"] = str(vllm.__version__)
        out["batch_invariant"] = bool(envs.VLLM_BATCH_INVARIANT)
    except Exception as e:  # noqa: BLE001
        out["vllm_error"] = f"{type(e).__name__}: {e}"
    try:
        import triton
        out["triton_version"] = str(triton.__version__)
    except Exception:  # noqa: BLE001
        pass
    try:
        bic = importlib.import_module("vllm.model_executor.determinism.batch_invariant_configs")
        fam = getattr(bic, "_get_tuned_matmul_arch_family", None)
        if fam is not None and out["compute_capability"] is not None:
            from vllm.platforms.interface import DeviceCapability
            out["tuned_gemm_arch_family"] = fam(DeviceCapability(*out["compute_capability"]))
    except Exception as e:  # noqa: BLE001
        out["tuned_gemm_arch_family_error"] = f"{type(e).__name__}: {e}"
    return out


def host_facts() -> dict:
    """The construction HOST (informational; not part of any identity): hostname, CUDA visibility, the host GPU's name."""
    import os
    import socket
    h = {"hostname": socket.gethostname(), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "cuda_available": bool(torch.cuda.is_available()),
         "host_gpu": None}
    if h["cuda_available"]:
        try:
            h["host_gpu"] = torch.cuda.get_device_name(0)
            h["host_sm_count"] = torch.cuda.get_device_properties(0).multi_processor_count
            cc = torch.cuda.get_device_capability(0)
            h["host_compute_capability"] = [int(cc[0]), int(cc[1])]
        except Exception as e:  # noqa: BLE001
            h["host_gpu_error"] = f"{type(e).__name__}: {e}"
    return h


#: [R13 moe] the vLLM dispatcher custom ops whose body is PURE PYTHON DISPATCH to the computation (no kernel of their own):
#: `vllm.moe_forward[_shared](hidden, router_logits, ..., layer_name)` -> `MoERunner._forward_impl`, and inside it
#: `vllm.fused_experts(...)` -> `fused_moe.fused_experts_op` -> `fused_experts_impl` (the router top-k, the block schedule, the two
#: grouped-GEMM Triton launches, the activation and the routed sum).  Exported as opaque nodes they hide the whole MoE block behind
#: ONE mutating op no rule can bind (R12 F-moe-02 / B-MOE); traced through their bodies, the block is the leaf kernels the rules bind
#: (`_moe_C.topk_softmax`, `_moe_C.moe_align_block_size`, `fused_moe_kernel` x2 via `verity_cba::triton_launch`, `_C.silu_and_mul`,
#: `_moe_C.moe_sum`) -- the same launches the eager capture records.  Generic: names the dispatcher ops, not a model/layer.
_MOE_DISPATCH_OPS = ("moe_forward", "moe_forward_shared", "fused_experts")


def _moe_c_fake_impls() -> list:
    """Fake (meta) kernels for the `_moe_C` / `_C` leaf ops the traced MoE body launches, when the pinned vLLM registers none: all but
    `grouped_topk` write into caller-allocated outputs and return nothing, so their fake is `None` (shapes are the allocations').  Returns
    the ops registered (nothing to undo: registration is per process and is only ever the same fake)."""
    done = []
    for ns, name in (("_moe_C", "topk_softmax"), ("_moe_C", "moe_align_block_size"), ("_moe_C", "moe_sum"), ("_C", "silu_and_mul"),
                     ("_C", "silu_and_mul_per_block_quant")):
        try:
            op = getattr(getattr(torch.ops, ns), name)
        except (AttributeError, RuntimeError):
            continue
        try:
            torch.library.register_fake(f"{ns}::{name}")(lambda *a, **k: None)
            done.append(f"{ns}::{name}")
        except RuntimeError:  # already has a fake / meta kernel
            pass
    # `grouped_topk(scores [M, E], n_group, topk_group, topk, ...)` allocates its outputs: f32 weights and int32 ids, [M, topk]
    try:
        torch.ops._moe_C.grouped_topk
        torch.library.register_fake("_moe_C::grouped_topk")(
            lambda scores, n_group, topk_group, topk, *a, **k: (scores.new_empty((scores.shape[0], topk), dtype=torch.float32),
                                                                scores.new_empty((scores.shape[0], topk), dtype=torch.int32)))
        done.append("_moe_C::grouped_topk")
    except (AttributeError, RuntimeError):
        pass
    return done


@contextmanager
def moe_ops_traced(model=None):
    """3. Trace THROUGH vLLM's MoE dispatcher custom ops (`_MOE_DISPATCH_OPS`): every module of `model` whose `_forward_entry` is the
    `vllm.moe_forward[_shared]` op packet gets, for the duration, a Python entry that calls its own `_forward_impl` (what the op's
    real kernel does: `moe_runner._moe_forward`), and `torch.ops.vllm.fused_experts` is the Python `fused_experts_op` (what the op's
    real kernel does).  Semantics-preserving by construction (the op bodies ARE these calls); only the export sees a difference.
    Without vLLM (or a model without MoE layers) it is a no-op."""
    try:
        from vllm.model_executor.layers.fused_moe import fused_moe as _fm
    except Exception:  # noqa: BLE001
        yield
        return
    _moe_c_fake_impls()
    ns = torch.ops.vllm
    packets = {}
    for name in _MOE_DISPATCH_OPS:
        try:
            packets[name] = getattr(ns, name)
        except (AttributeError, RuntimeError):
            pass
    h = hooks.Hooks()
    if model is not None:
        for m in model.modules():
            # the engine builds each MoE layer's kernel object (prepare/finalize + experts, `make_unquantized_moe_kernel`) in
            # `process_weights_after_loading`, which a meta instantiation never runs: build it here the same way (pure Python
            # object construction; the TRITON backend neither pads nor shuffles the weights) so `_forward_impl` has its kernel.
            qm = getattr(m, "quant_method", None)
            if qm is not None and hasattr(qm, "_init_moe_kernel") and getattr(qm, "moe_kernel", None) is None and hasattr(m, "w13_weight"):
                qm._init_moe_kernel(m)
            fe = getattr(m, "_forward_entry", None)
            if fe is None or not hasattr(m, "_forward_impl"):
                continue
            if fe is packets.get("moe_forward") or fe is packets.get("moe_forward_shared"):
                h.patch(m, "_forward_entry", (lambda hs, rl, sei, ids, _name, _hd, _l=m: _l._forward_impl(hs, rl, sei, ids)))
        _router_bias_through_owner(model, h)
    if "fused_experts" in packets:
        # `_OpNamespace.__getattr__` caches resolved packets as instance attributes; a plain function in its place is what callers
        # of `torch.ops.vllm.fused_experts(**kw)` now reach, and the packet is restored below.
        h.patch(ns, "fused_experts", _fm.fused_experts_op)
    # the engine's MoE workspace (`v1/worker/workspace.py`): ONE persistent device buffer per worker that every layer's
    # intermediate caches are views of (`get_simultaneous`).  Under export a buffer that outlives the trace is hidden state (a lifted
    # constant); a fresh allocation per request is the same computation (every workspace word is written before it is read within the
    # layer), so the export-time manager hands out `torch.empty` views -- ordinary `Alloc`s the in-place rules rebind.
    try:
        from vllm.v1.worker import workspace as _ws
    except Exception:  # noqa: BLE001
        _ws = None
    if _ws is not None and hasattr(_ws, "_manager"):
        h.patch(_ws, "_manager", _ExportWorkspaceManager())
    with h:
        yield


def _router_bias_through_owner(model, h) -> None:
    """A MoE router is a plain object (not a module) holding the layer's `e_score_correction_bias` Parameter (`GroupedTopKRouter`,
    `FusedTopKBiasRouter`).  The export swaps each module's parameters for its own tensors while it traces, so a reference held
    outside the module tree is the original tensor, which the export lifts as a hidden constant.  For the duration, each such
    router reads the bias from the module that registers it, at every `select_experts` (the same Parameter: the bias the engine
    loads into it)."""
    owners = {id(p): (mod, n) for mod in model.modules() for n, p in mod._parameters.items() if p is not None}
    for m in model.modules():
        r = getattr(m, "router", None)
        b = getattr(r, "e_score_correction_bias", None)
        if r is None or isinstance(r, torch.nn.Module) or not isinstance(b, torch.nn.Parameter) or id(b) not in owners:
            continue
        owner, name = owners[id(b)]

        def select(*a, _r=r, _o=owner, _n=name, _sel=r.select_experts, **k):
            _r.e_score_correction_bias = getattr(_o, _n)
            return _sel(*a, **k)

        h.patch(r, "e_score_correction_bias", b)
        h.patch(r, "select_experts", select)


class _ExportWorkspaceManager:
    """`WorkspaceManager` for the export: fresh allocations instead of views of one persistent buffer (see `moe_ops_traced`)."""

    def __init__(self) -> None:
        self._locked = False

    def get_simultaneous(self, *shapes_and_dtypes):
        return [torch.empty(tuple(s), dtype=d) for s, d in shapes_and_dtypes]

    def lock(self) -> None:
        self._locked = True

    def unlock(self) -> None:
        self._locked = False

    def is_locked(self) -> bool:
        return self._locked


def _dsa_layers(model) -> list:
    return [m for m in (model.modules() if model is not None else ())
            if hasattr(m, "_sparse_indexer_and_attn") and hasattr(m, "fused_qkv_a_proj")]


@contextmanager
def dsa_ops_leaf(model=None):
    """4. DeepSeek sparse attention (`DeepseekV32Attention.forward`, GLM-5.3, DeepSeek-V3.2) as three leaf ops.  Its two Triton kernels
    run outside the MLA attention op and write the MLA and indexer caches through the forward context's slot mapping, which the meta
    export has no cache for (`fused_norm_rope` asserts `slot_mapping is not None`); its attention reads those caches back through the
    same context.  The kernels are patched in the module the layers' `forward` is defined in, the globals it reads them from.  See
    `dsa_leaves`.  A model without a sparse attention layer is unchanged."""
    sites = {sys.modules[type(m).forward.__module__] for m in _dsa_layers(model)}
    if len(sites) > 1:
        raise NotImplementedError(f"dsa_ops_leaf: sparse attention layers whose forward reads its kernels from {len(sites)} modules")
    if not sites:
        yield
        return
    with dsa_leaves(sites.pop(), model):
        yield


@contextmanager
def dsa_leaves(site, model=None):
    """For the duration, ``site`` (the module `DeepseekV32Attention.forward` reads its kernels from) calls `vllm_iface.dsa_norm_rope`
    for `fused_norm_rope` and `vllm_iface.dsa_q` for `fused_q`, and every module of ``model`` with a `_sparse_indexer_and_attn` gets
    one that calls `vllm_iface.dsa_indexer_attn` and then the absorbed value projection its body ends with, `bmm(attn, W_UV)` into
    `output`.  The cache rows the kernels write are the first op's outputs (`kv_c_out`, `k_pe_out`, the index K and its scale) and
    the third op's operands, so each layer's cache is the derive's state (`rules/vllm_dsa.py`).  What the shim does not model is
    refused at export: prefill context parallelism, decode context parallelism, an fp8 query or KV cache, a caller-supplied `q_c_out`
    or `index_k_out`.  `W_UK_T` and `W_UV` are Parameters the engine registers after loading; a meta model gets them from
    `vllm_meta.mla_absorbed_weights_as_the_loader_does`."""
    from verity_vllm.program.frontend.rules import vllm_iface  # noqa: F401  (registers the ops)
    iface = torch.ops.vllm_iface
    pending: dict[int, tuple] = {}
    h = hooks.Hooks()

    def fused_norm_rope(positions, q_c, q_w, q_eps, kv_c, kv_w, kv_eps, k_pe, k_cs, index_k, index_k_w, index_k_b, index_k_eps, index_cs,
                        topk_indices_buffer, slot_mapping=None, indexer_k_cache=None, mla_kv_cache=None, mla_kv_cache_dtype="auto",
                        mla_k_scale=None, has_indexer=True, index_rope_interleave=False, q_c_out=None, kv_c_out=None, k_pe_out=None,
                        index_k_out=None):
        if q_c_out is not None or index_k_out is not None or kv_c_out is None or k_pe_out is None:
            raise NotImplementedError("dsa_leaves: fused_norm_rope with q_c_out / index_k_out, or without kv_c_out / k_pe_out")
        if mla_kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(f"dsa_leaves: MLA KV cache dtype {mla_kv_cache_dtype!r} (only a bf16 cache is modelled)")
        ix = (index_k, index_k_w, index_k_b, index_cs) if has_indexer else (None, None, None, None)
        q, ik, iks = iface.dsa_norm_rope(positions, q_c, q_w, float(q_eps), kv_c, kv_w, float(kv_eps), k_pe, k_cs, ix[0], ix[1], ix[2],
                                         float(index_k_eps), ix[3], bool(index_rope_interleave), kv_c_out, k_pe_out)
        pending[id(kv_c_out)] = (ik, iks) if has_indexer else (None, None)
        return q

    def fused_q(positions, q_pe, q_pe_cs, index_q, index_q_cs, ql_nope, q_scale, index_weights, softmax_scale, head_scale,
                has_indexer=True, index_rope_interleave=False, quantize_mqa=True):
        if quantize_mqa:
            raise NotImplementedError("dsa_leaves: fused_q with quantize_mqa (an fp8 MQA query)")
        ix = (index_q, index_q_cs, index_weights) if has_indexer else (None, None, None)
        q_pe_out, iq, iw = iface.dsa_q(positions, q_pe, q_pe_cs, ix[0], ix[1], ix[2], float(softmax_scale), float(head_scale),
                                       bool(index_rope_interleave))
        return iq, iw, q_pe_out

    def indexer_and_attn(layer):
        def run(positions, q_c, q_nope, q_pe, index_q_fp8, index_k, index_weights_out, kv_c, k_pe, ql_nope, mqa_q, output):
            if layer.use_pcp or layer.impl.dcp_world_size > 1 or layer._fp8_query or layer._fp8_kv_needs_view:
                raise NotImplementedError(f"dsa_leaves: {layer.layer_name} under context parallelism or an fp8 KV cache")
            ik, iks = pending.pop(id(kv_c))
            has = ik is not None
            a = iface.dsa_indexer_attn(positions, ql_nope, mqa_q, kv_c, k_pe, index_q_fp8 if has else None, index_weights_out if has else None,
                                       ik, iks, layer.layer_name, int(layer.topk_indices_buffer.shape[-1]), float(layer.scale))
            n, nh, lat, v = output.shape[0], layer.num_heads, layer.kv_lora_rank, layer.v_head_dim
            y = torch.bmm(a.view(n, nh, lat).transpose(0, 1), layer.W_UV)
            output.copy_(y.transpose(0, 1).reshape(n, nh * v))
        return run

    h.patch(site, "fused_norm_rope", fused_norm_rope)
    h.patch(site, "fused_q", fused_q)
    for m in _dsa_layers(model):
        h.patch(m, "_sparse_indexer_and_attn", indexer_and_attn(m))
    with h:
        yield


_LL_BF16_SITE = "vllm.model_executor.kernels.linear.cute_dsl.ll_bf16"


@contextmanager
def moe_gate_leaf(model=None, site=None):
    """5. `GateLinear.forward`'s tier 1 (router/gate_linear.py:181-187, M <= 16 on sm_90+ wherever CUTLASS DSL imports) calls
    `ll_bf16_gemm`, a CUTLASS DSL kernel that checks its operands are on a CUDA device and leaves no dispatcher op behind.  When a
    module of ``model`` would take that tier (`allow_ll_bf16_gemm`), for the duration ``site``'s (default vLLM's `cute_dsl.ll_bf16`,
    which `forward` imports it from at every call) is `vllm_iface.ll_bf16_gemm(x, w)`, so the tier the declared target takes is the
    node the rules bind; any output dtype but f32 refuses at export.  A model without such a gate is unchanged."""
    if not any(getattr(m, "allow_ll_bf16_gemm", False) for m in (model.modules() if model is not None else ())):
        yield
        return
    if site is None:
        import importlib
        site = importlib.import_module(_LL_BF16_SITE)
    from verity_vllm.program.frontend.rules import vllm_iface  # noqa: F401  (registers the op)

    def ll_bf16_gemm(hidden_states, router_weight, output_dtype=torch.float32):
        if output_dtype != torch.float32:
            raise NotImplementedError(f"moe_gate_leaf: ll_bf16_gemm with output dtype {output_dtype} (only f32 is modelled)")
        return torch.ops.vllm_iface.ll_bf16_gemm(hidden_states, router_weight)

    with hooks.patched(site, "ll_bf16_gemm", ll_bf16_gemm):
        yield


@contextmanager
def export_compat(profile=None, model=None):
    """The export shims: `.data` -> `detach()` on fake tensors + the declared target applied (default profile if None) + the MoE
    dispatcher ops traced through their bodies (`moe_ops_traced`) + DeepSeek sparse attention's kernels and the MoE gate's cuteDSL GEMM
    as leaf ops (`dsa_ops_leaf`, `moe_gate_leaf`); `model` = the instantiated module, for the per-layer entries."""
    if profile is None:
        from verity_vllm.program.frontend.target_profile import TargetProfile
        profile = TargetProfile.default()
    with tensor_data_as_detach(), apply_target_profile(profile), moe_ops_traced(model), dsa_ops_leaf(model), moe_gate_leaf(model):
        yield


__all__ = ["export_compat", "tensor_data_as_detach", "apply_target_profile", "select_cuda_platform", "moe_ops_traced", "dsa_ops_leaf", "dsa_leaves",
           "moe_gate_leaf", "read_back_applied", "host_facts"]
