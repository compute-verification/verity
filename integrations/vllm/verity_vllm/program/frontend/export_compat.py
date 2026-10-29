"""Export-compatibility shims for idioms in the pinned vLLM source that non-strict `torch.export` cannot trace.
Each shim is generic (names no model, no layer, no op sequence) and semantics-preserving for inference; the REPORT
counts them as structural scaffolding.

1. `Tensor.data` reads inside forward (`self.weight.data` in `RMSNorm.forward_cuda`, `layernorm.py:104-113`): under
   non-strict export `.data` yields an untracked FakeTensor that export then tries to lift as a constant
   ("We found a fake tensor in the exported program constant's list"). For fake tensors the shim returns `detach()`
   — the same value, recorded as `aten.detach` of the parameter placeholder.
2. The declared target applied (`apply_target_profile`).  3. vLLM's MoE dispatcher ops traced through (`moe_ops_traced`).
4. DeepSeek sparse attention's three kernels as leaf ops (`dsa_ops_leaf`).  5. The MoE gate's cuteDSL GEMM as a leaf op
   (`moe_gate_leaf`).  6. The default Hopper block-FP8 linear's dynamic custom op as a leaf
   (`default_fp8_linear_leaf`).  7. CUDA streams and events as inert stand-ins (`cuda_streams_inert`).  8. A registered op's
   Python body, called directly by a model module, as the op (`registered_op_bodies_as_ops`).  9. DeepSeek-V4's layer-0 mHC
   broadcast pre as a leaf op (`mhc_pre_broadcast_leaf`).  10. DeepSeek-V4's sparse MLA kernels as leaf ops
   (`dsv4_leaves.dsv4_attention_leaves`).  11. DeepGEMM's `fp8_einsum` as a leaf op (`deep_gemm_fp8_einsum_leaf`).  12. The MXFP4
   Marlin MoE's expert GEMMs as leaf ops over the logical weights (`marlin_moe_leaf.mxfp4_marlin_moe_leaf`).
"""

from __future__ import annotations

import importlib
import inspect
import sys
from contextlib import contextmanager

import torch

from verity_vllm.engine import env as engine_env, hooks
from verity_vllm.program.frontend.dsv4_leaves import dsv4_attention_leaves
from verity_vllm.program.frontend.marlin_moe_leaf import mxfp4_marlin_moe_leaf

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
_FLASHINFER_FP8_MODULE = "vllm.model_executor.kernels.linear.scaled_mm.flashinfer"


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
        if (major, minor) == (9, 0):
            # The standard Hopper vLLM environment carries FlashInfer.  A GPU-free Build cannot load its CUDA extension, so its
            # availability probe would otherwise choose plain DeepGEMM while the served target selects the earlier dynamic
            # FlashInfer/DeepGEMM kernel.  Answer the declared target's installed capability, as we do for its cc and SM count.
            flashinfer_fp8 = importlib.import_module(_FLASHINFER_FP8_MODULE)
            h.patch(flashinfer_fp8, "is_flashinfer_fp8_blockscale_gemm_supported", lambda: True)
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


def select_cuda_platform(world: int = 1, device_name: str | None = None) -> str:
    """A declared target is a CUDA device: vLLM's platform resolution answers `vllm.platforms.cuda.CudaPlatform` (the qualname its CUDA
    plugin returns on a GPU host) instead of probing NVML for it.  On a GPU-free Build host NVML is absent and that probe resolves a
    non-CUDA platform.  `import vllm` itself resolves `current_platform` (`torch_utils.PIN_MEMORY`), so before vLLM is imported the answer
    is installed as `vllm.platforms` finishes executing; the class vLLM then instantiates is the GPU host's (`CudaPlatform`: the
    NVML-backed class when NVML answers, the torch-backed one when it does not), and `apply_target_profile` answers the target's
    capability on either.  The target is `world` such devices: the platform's `device_count()` answers `world`, not the host's visible
    count, so `ParallelConfig`'s "world size larger than the available GPUs" check passes on a GPU-free host and the executor backend
    resolves to `mp`, as on a host with the `world` GPUs the Commit runs on.  On a host with no visible device the platform's
    `get_device_name()` answers `device_name` (the row's class's, `engine.profiles.devices`), the key vLLM's tuned fused-MoE configs
    are read by, so the Build reads the config file the class's GPU does; a GPU host answers its own.  Build path only: serving resolves
    its platform and counts its devices as before.  Raises when `current_platform` already resolved to a non-CUDA platform."""
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
    if device_name is not None and not torch.cuda.is_available():
        hooks.patch(type(current_platform), "get_device_name", classmethod(lambda cls, device_id=0, _n=device_name: _n))
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
    import socket
    from verity_vllm import settings
    h = {"hostname": socket.gethostname(), "CUDA_VISIBLE_DEVICES": settings.raw("CUDA_VISIBLE_DEVICES"), "cuda_available": bool(torch.cuda.is_available()),
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


#: the vLLM dispatcher custom ops whose body is PURE PYTHON DISPATCH to the computation (no kernel of their own):
#: `vllm.moe_forward[_shared](hidden, router_logits, ..., layer_name)` -> `MoERunner._forward_impl`, and inside it
#: `vllm.fused_experts(...)` -> `fused_moe.fused_experts_op` -> `fused_experts_impl` (the router top-k, the block schedule, the two
#: grouped-GEMM Triton launches, the activation and the routed sum).  Exported as opaque nodes they hide the whole MoE block behind
#: ONE mutating op no rule can bind; traced through their bodies, the block is the leaf kernels the rules bind
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
            _expert_weight_scales_through_owner(m, qm, h)
            fe = getattr(m, "_forward_entry", None)
            if fe is None or not hasattr(m, "_forward_impl"):
                continue
            if fe is packets.get("moe_forward") or fe is packets.get("moe_forward_shared"):
                h.patch(m, "_forward_entry", (lambda hs, rl, sei, ids, _name, _hd, _l=m: _l._forward_impl(hs, rl, sei, ids)))
        _router_params_through_owner(model, h)
    if "fused_experts" in packets:
        # `_OpNamespace.__getattr__` caches resolved packets as instance attributes; a plain function in its place is what callers
        # of `torch.ops.vllm.fused_experts(**kw)` now reach, and the packet is restored below.
        h.patch(ns, "fused_experts", _fm.fused_experts_op)
    _moe_config_at_launch(_fm, h)
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


def _moe_config_at_launch(fused_moe_module, h) -> None:
    """Under a declared launch batch, the Triton experts' tile config (`try_get_optimal_moe_config`, read at trace time from the
    token count) is the one for the launch's tokens, every module that imported it by name included."""
    from verity_vllm.program.frontend.rules.vllm_bindings.observations import launch_batch_declared, traced_launch_tokens
    if not launch_batch_declared():
        return
    original = fused_moe_module.try_get_optimal_moe_config

    def config_at_launch(w1_shape, w2_shape, top_k, dtype, M, block_shape=None):
        return original(w1_shape, w2_shape, top_k, dtype, traced_launch_tokens(M), block_shape=block_shape)

    for name, module in list(sys.modules.items()):
        if name.startswith(fused_moe_module.__package__) and getattr(module, "try_get_optimal_moe_config", None) is original:
            h.patch(module, "try_get_optimal_moe_config", config_at_launch)


def _expert_weight_scales_through_owner(layer, quant_method, h) -> None:
    """Make a plain MoE-kernel object read its registered expert-scale Parameters through their owning module during export.

    vLLM's FP8 ``_init_moe_kernel`` copies the ``RoutedExperts.w13_weight_scale_inv`` / ``w2_weight_scale_inv`` objects into
    ``moe_kernel.fused_experts.quant_config._w1.scale`` / ``_w2.scale``; the experts' ``w1_scale`` / ``w2_scale`` are read-only
    properties over those descriptors.  Neither object is an ``nn.Module``.  Torch export swaps Parameters on modules for graph values
    while tracing, so the stale descriptor references otherwise become lifted tensor constants.  Refresh them immediately before
    ``fused_experts.apply`` from the owning module, exactly as ``_router_params_through_owner`` does for the router's Parameters.
    Identity is checked before patching: a backend-owned derived scale is not silently relabelled as a Parameter."""
    kernel = getattr(quant_method, "moe_kernel", None)
    experts = getattr(kernel, "fused_experts", None)
    apply = getattr(experts, "apply", None)
    if experts is None or not callable(apply):
        return
    owners = {id(p): (mod, name) for mod in layer.modules() for name, p in mod._parameters.items() if p is not None}
    bindings = []
    quant_config = getattr(experts, "quant_config", None)
    for attr, descriptor_name in (("w1_scale", "_w1"), ("w2_scale", "_w2")):
        scale = getattr(experts, attr, None)
        descriptor = getattr(quant_config, descriptor_name, None)
        if descriptor is not None and getattr(descriptor, "scale", None) is scale and id(scale) in owners:
            bindings.append((descriptor, *owners[id(scale)]))
    if not bindings:
        return

    def apply_with_owned_scales(*args, _apply=apply, _bindings=tuple(bindings), **kwargs):
        for descriptor, owner, name in _bindings:
            descriptor.scale = getattr(owner, name)
        return _apply(*args, **kwargs)

    for descriptor, _owner, _name in bindings:
        h.patch(descriptor, "scale", descriptor.scale)
    h.patch(experts, "apply", apply_with_owned_scales)


def _router_params_through_owner(model, h) -> None:
    """A MoE router is a plain object (not a module) holding Parameters of the layer: its `e_score_correction_bias`
    (`GroupedTopKRouter`, `FusedTopKBiasRouter`) and `FusedTopKBiasRouter`'s hash table (`_hash_indices_table`, DeepSeek-V4's
    `gate.tid2eid`).  The export swaps each module's parameters for its own tensors while it traces, so a reference held outside
    the module tree is the original tensor, which the export lifts as a hidden constant.  For the duration, each such router reads
    each of them from the module that registers it, at every `select_experts` (the same Parameter: the one the engine loads)."""
    owners = {id(p): (mod, n) for mod in model.modules() for n, p in mod._parameters.items() if p is not None}
    for m in model.modules():
        r = getattr(m, "router", None)
        if r is None or isinstance(r, torch.nn.Module):
            continue
        held = tuple((a, *owners[id(p)]) for a, p in vars(r).items() if isinstance(p, torch.nn.Parameter) and id(p) in owners)
        if not held:
            continue

        def select(*a, _r=r, _held=held, _sel=r.select_experts, **k):
            for attr, owner, name in _held:
                setattr(_r, attr, getattr(owner, name))
            return _sel(*a, **k)

        for attr, _owner, _name in held:
            h.patch(r, attr, getattr(r, attr))
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
    `output`.  A call vLLM serves by dense MHA (`vllm_dsa_dense.takes_dense_mha`) runs `forward_mha`'s body: `kv_b_proj` up projects
    the latent to k_nope and v, then `vllm_iface.dsa_dense_mha` over q's and k's parts (`_concat_k_nope_k_pe`'s copies, which its rule
    assembles), its rows flattened into `output`.  The cache rows the kernels write are the first op's outputs (`kv_c_out`,
    `k_pe_out`, the index K and its scale) and the attention op's operands, so each layer's cache is the derive's state
    (`rules/vllm_dsa.py`).  What the shim does not model is refused at export: prefill context parallelism, decode context
    parallelism, an fp8 query or KV cache, a caller-supplied `q_c_out` or `index_k_out`.  `W_UK_T` and `W_UV` are Parameters the
    engine registers after loading; a meta model gets them from `vllm_meta.mla_absorbed_weights_as_the_loader_does`."""
    from verity_vllm.program.frontend.rules import vllm_dsa_dense, vllm_iface  # noqa: F401  (vllm_iface registers the ops)
    iface = torch.ops.vllm_iface
    pending: dict[int, tuple] = {}
    seen: dict[str, int] = {}
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
            n, topk = output.shape[0], int(layer.topk_indices_buffer.shape[-1])
            rows = seen[layer.layer_name] = seen.get(layer.layer_name, 0) + n
            if vllm_dsa_dense.takes_dense_mha(layer, n, rows, topk):
                nh, dn, dv = layer.num_heads, layer.qk_nope_head_dim, layer.v_head_dim
                k_nope, v = layer.kv_b_proj(kv_c)[0].view(-1, nh, dn + dv).split([dn, dv], dim=-1)
                o = iface.dsa_dense_mha(positions, q_nope, mqa_q, k_nope, v, kv_c, k_pe, ik, iks, layer.layer_name, topk, float(layer.scale))
                output.copy_(o.flatten(start_dim=-2))
                return
            a = iface.dsa_indexer_attn(positions, ql_nope, mqa_q, kv_c, k_pe, index_q_fp8 if has else None, index_weights_out if has else None,
                                       ik, iks, layer.layer_name, topk, float(layer.scale))
            nh, lat, v = layer.num_heads, layer.kv_lora_rank, layer.v_head_dim
            y = torch.bmm(a.view(n, nh, lat).transpose(0, 1), layer.W_UV)
            output.copy_(y.transpose(0, 1).reshape(n, nh * v))
        return run

    h.patch(site, "fused_norm_rope", fused_norm_rope)
    h.patch(site, "fused_q", fused_q)
    for m in _dsa_layers(model):
        h.patch(m, "_sparse_indexer_and_attn", indexer_and_attn(m))
    with h:
        yield


def _model_modules(model) -> list:
    """The modules ``model``'s classes are defined in: the globals their methods read."""
    return [sys.modules[n] for n in sorted({type(m).__module__ for m in model.modules()}) if n in sys.modules]


_FUSED_AR_NORM_MODULE = "vllm.models.common.ops.fused_allreduce_rms_norm"
_FUSED_AR_NORM_NAME = "fused_allreduce_rms_norm"


@contextmanager
def fused_allreduce_rms_norm_leaf(model=None):
    """Author the leaf that meta execution otherwise misses when the served call takes FlashInfer's fused TP8 path.

    `_can_use_flashinfer` rejects every meta tensor because it checks ``is_cuda``.  The other dispatch facts are static:
    world 8, bf16 ``[M,6144]`` with M <= 42 (M the launch's tokens, `traced_launch_tokens`), and FlashInfer all-reduce enabled.
    The construction's ``TPStub`` carries those workload settings and authors the rank-ordered peer operands through its
    ``CollectiveBus``.  Calls outside that path keep vLLM's original fallback (ordinary all-reduce followed by the norm).
    """
    from verity_vllm.program.frontend.rules.vllm_bindings.observations import traced_launch_tokens
    home = sys.modules.get(_FUSED_AR_NORM_MODULE)
    original = getattr(home, _FUSED_AR_NORM_NAME, None) if home is not None else None
    if original is None or model is None:
        yield
        return
    users = [m for m in _model_modules(model) if m is not home and getattr(m, _FUSED_AR_NORM_NAME, None) is original]
    if not users:
        yield
        return

    def shim(hidden_states, residual, norm, *args, **kwargs):
        M = traced_launch_tokens(hidden_states.shape[0])
        try:
            from vllm.distributed.parallel_state import get_tp_group
            group = get_tp_group()
            bus = getattr(group, "bus", None)
            shape = tuple(hidden_states.shape)
            fused = (bus is not None and int(getattr(group, "world_size", 1)) == 8
                     and bool(bus.settings.get("flashinfer")) and hidden_states.dtype == torch.bfloat16
                     and len(shape) == 2 and shape[1] == 6144 and M <= 42
                     and tuple(residual.shape) == shape and tuple(norm.weight.shape) == (6144,))
            if fused:
                return bus.fused_allreduce_rms_norm(hidden_states, residual, norm.weight, float(norm.variance_epsilon))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        return original(hidden_states, residual, norm, *args, **kwargs)

    with hooks.Hooks() as h:
        h.patch(home, _FUSED_AR_NORM_NAME, shim)
        for module in users:
            h.patch(module, _FUSED_AR_NORM_NAME, shim)
        yield


_LL_BF16_SITE = "vllm.model_executor.kernels.linear.cute_dsl.ll_bf16"


@contextmanager
def moe_gate_leaf(model=None, site=None):
    """5. `GateLinear.forward`'s tier 1 (router/gate_linear.py:181-187, M <= 16 on sm_90+ wherever CUTLASS DSL imports) calls
    `ll_bf16_gemm`, a CUTLASS DSL kernel that checks its operands are on a CUDA device and leaves no dispatcher op behind.  When a
    module of ``model`` would take that tier (`allow_ll_bf16_gemm`), for the duration ``site``'s (default vLLM's `cute_dsl.ll_bf16`,
    which `forward` imports it from at every call) is `vllm_iface.ll_bf16_gemm(x, w)`, so the tier the declared target takes is the
    node the rules bind; any output dtype but f32 refuses at export.  A model without such a gate is unchanged.  Past 16 launch rows
    (`traced_launch_tokens`: the request's own rows sit in a larger batch) vLLM takes the cuBLAS tier, ``torch.mm(x, W.T,
    out_dtype=float32)``, which the call then authors; a gate that would take another tier there refuses."""
    gates = [m for m in (model.modules() if model is not None else ()) if getattr(m, "allow_ll_bf16_gemm", False)]
    if not gates:
        yield
        return
    if site is None:
        site = importlib.import_module(_LL_BF16_SITE)
    from verity_vllm.program.frontend.rules import vllm_iface  # noqa: F401  (registers the op)
    from verity_vllm.program.frontend.rules.vllm_bindings.observations import traced_launch_tokens
    from verity.ml.definitions.registry.bf16_gemm_check import LL_MAX_M
    cublas_tier = all(getattr(g, "allow_cublas_router_gemm", False) and not getattr(g, "allow_fp32_router_gemm", False)
                      and not getattr(g, "allow_bf16x3_router_gemm", False) for g in gates)

    def ll_bf16_gemm(hidden_states, router_weight, output_dtype=torch.float32):
        if output_dtype != torch.float32:
            raise NotImplementedError(f"moe_gate_leaf: ll_bf16_gemm with output dtype {output_dtype} (only f32 is modelled)")
        if traced_launch_tokens(hidden_states.shape[0]) > LL_MAX_M:
            if not cublas_tier:
                raise NotImplementedError("moe_gate_leaf: past 16 launch rows a gate without the cuBLAS tier alone takes another tier")
            return torch.mm(hidden_states, router_weight.T, out_dtype=torch.float32)
        return torch.ops.vllm_iface.ll_bf16_gemm(hidden_states, router_weight)

    with hooks.patched(site, "ll_bf16_gemm", ll_bf16_gemm):
        yield


@contextmanager
def default_fp8_linear_leaf(model=None, kernel_class=None):
    """6. Keep vLLM's default Hopper block-FP8 linear as its served custom op.

    The pinned kernel's ``apply_block_scaled_mm`` calls
    ``vllm.dynamic_flashinfer_deepgemm_blockscale_gemm``.  Its runtime body
    contains ``torch.cond`` and, under non-strict meta export, is traced
    through to the M-specific branch.  That exposes mutating implementation
    details (notably ``per_token_group_fp8_quant(use_ue8m0=True)``) instead of
    the custom-op boundary the served engine dispatches.  During export,
    replace that kernel class's ``apply_block_scaled_mm`` with the equivalent
    ``vllm_iface`` leaf.  Another selected kernel is unchanged.
    """
    if model is None:
        yield
        return
    if kernel_class is None:
        from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
            FlashInferFp8DeepGEMMDynamicBlockScaledKernel as kernel_class,
        )
    from verity_vllm.program.frontend.rules.vllm_iface import iface

    def apply_block_scaled_mm(self, A, B, As, Bs):
        return iface.dynamic_flashinfer_deepgemm_blockscale_gemm(
            A, B, Bs, int(self.weight_group_shape.col), bool(self.fallback.use_deep_gemm_e8m0)
        )

    with hooks.patched(kernel_class, "apply_block_scaled_mm", apply_block_scaled_mm):
        yield


class InertStream:
    """A CUDA stream of a meta model (`cuda_streams_inert`)."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def wait_stream(self, stream) -> None:
        pass

    def wait_event(self, event) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def query(self) -> bool:
        return True


class InertEvent:
    """A CUDA event of a meta model (`cuda_streams_inert`)."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def record(self, stream=None) -> None:
        pass

    def wait(self, stream=None) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def query(self) -> bool:
        return True


@contextmanager
def cuda_streams_inert():
    """7. CUDA streams and events as inert stand-ins, for the meta construction and the export alike.  A model may make aux streams
    and events in `__init__` (DeepSeek-V4's attention overlaps its input GEMMs through `multi_stream_utils.execute_in_parallel`), and
    making one needs a driver a GPU-free Build host lacks.  A stream only says which queue a kernel joins and an event only orders
    queues, so on meta both are nothing: `torch.cuda.stream` of a stand-in is torch's no-op `stream(None)`, and `record` / `wait`
    do nothing.  The forward still calls its functions in the order the engine launches them (`execute_in_parallel`: the aux
    functions, then the default one)."""
    stream = torch.cuda.stream
    with hooks.Hooks() as h:
        h.patch(torch.cuda, "Stream", InertStream)
        h.patch(torch.cuda, "Event", InertEvent)
        h.patch(torch.cuda, "stream", lambda s: stream(None if isinstance(s, InertStream) else s))
        yield


def _registered_from(ns, f):
    """The op packet of ``ns`` that was registered from function ``f``: of f's name, with the schema `infer_schema` gives f (the
    arguments the op writes declared as mutated), else None."""
    from torch._library.infer_schema import infer_schema
    if not inspect.isfunction(f):
        return None
    try:
        packet = getattr(ns, f.__name__)
        schema = packet.default._schema
    except (AttributeError, RuntimeError):
        return None
    written = [a.name for a in schema.arguments if a.alias_info is not None and a.alias_info.is_write]
    try:
        mine = torch._C.parse_schema(f"{schema.name}{infer_schema(f, mutates_args=written)}")
    except (ValueError, RuntimeError):
        return None
    return packet if mine == schema else None


@contextmanager
def registered_op_bodies_as_ops(model=None, ns=None):
    """8. A model module may import the Python body of a vLLM custom op and call it directly: DeepSeek-V4's decoder layer calls
    `mhc_fused_post_pre_tilelang`, which `direct_register_custom_op` also registers as `vllm::mhc_fused_post_pre_tilelang`.  Traced,
    the body launches kernels a meta export cannot run.  For the duration, each global of a module ``model``'s classes are defined in
    that is the function an op of ``ns`` (default `torch.ops.vllm`) was registered from (`_registered_from`) is that op: the same call
    through the dispatcher, the op's fake giving the shapes."""
    if model is None:
        yield
        return
    ns = torch.ops.vllm if ns is None else ns
    with hooks.Hooks() as h:
        for module in _model_modules(model):
            for name, f in list(vars(module).items()):
                if (packet := _registered_from(ns, f)) is not None:
                    h.patch(module, name, packet)
        yield


_MHC_SITE = "vllm.model_executor.kernels.mhc.tilelang"


@contextmanager
def mhc_pre_broadcast_leaf(model=None, original=None):
    """9. Layer 0's mHC pre on the embedding broadcast to the residual streams (`mhc_pre_broadcast_tilelang`, DeepSeek-V4), which
    vLLM does not register: its body sizes DeepGEMM's split-K from the device's SM count and launches TileLang.  For the duration,
    each global of a module ``model``'s classes are defined in that is ``original`` (default vLLM's function, loaded by the model
    module that imports it) is `vllm_iface.mhc_pre_broadcast_tilelang`, an op of the schema `infer_schema` gives the function.  A
    model that does not call it is unchanged."""
    if original is None:
        original = getattr(sys.modules.get(_MHC_SITE), "mhc_pre_broadcast_tilelang", None)
    if model is None or original is None:
        yield
        return
    from verity_vllm.program.frontend.rules.vllm_iface import iface
    with hooks.Hooks() as h:
        for module in _model_modules(model):
            for name, f in list(vars(module).items()):
                if f is original:
                    h.patch(module, name, iface.mhc_pre_broadcast_tilelang)
        yield


_DEEP_GEMM = "vllm.utils.deep_gemm"


@contextmanager
def deep_gemm_fp8_einsum_leaf(model=None):
    """11. DeepGEMM's `fp8_einsum` (vLLM's `utils.deep_gemm` entry; DeepSeek-V4's o-projection `bhr,hdr->bhd`) JIT-compiles its
    kernel through the CUDA driver and leaves no dispatcher op behind.  For the duration, that entry and every vLLM module's global
    bound to it is `vllm_iface.deep_gemm_fp8_einsum`, which writes `d` as DeepGEMM does; an accumulator `c` refuses at export."""
    original = getattr(sys.modules.get(_DEEP_GEMM), "fp8_einsum", None)
    if model is None or original is None:
        yield
        return
    from verity_vllm.program.frontend.rules.vllm_iface import iface

    def fp8_einsum(expr, a, b, d, c=None, recipe=(1, 128, 128)):
        if c is not None:
            raise NotImplementedError("deep_gemm_fp8_einsum_leaf: fp8_einsum with an accumulator c")
        iface.deep_gemm_fp8_einsum(expr, a[0], a[1], b[0], b[1], d, [int(r) for r in recipe])

    users = [m for n, m in list(sys.modules.items()) if n.startswith("vllm.") and getattr(m, "fp8_einsum", None) is original]
    with hooks.Hooks() as h:
        for module in users:
            h.patch(module, "fp8_einsum", fp8_einsum)
        yield


@contextmanager
def export_compat(profile=None, model=None):
    """The export shims: `.data` -> `detach()` on fake tensors + the declared target applied (default profile if None) + the MoE
    dispatcher ops traced through their bodies (`moe_ops_traced`) + DeepSeek sparse attention's kernels, the MoE gate's cuteDSL GEMM
    and the default Hopper FP8 linear as leaf ops (`dsa_ops_leaf`, `moe_gate_leaf`, `default_fp8_linear_leaf`) + inert CUDA streams
    (`cuda_streams_inert`) + registered op bodies called as their ops (`registered_op_bodies_as_ops`), layer 0's mHC broadcast
    pre, DeepSeek-V4's sparse MLA, DeepGEMM's einsum and the MXFP4 Marlin MoE's GEMMs as leaves (`mhc_pre_broadcast_leaf`,
    `dsv4_attention_leaves`, `deep_gemm_fp8_einsum_leaf`, `mxfp4_marlin_moe_leaf`, which builds the kernel objects `moe_ops_traced`
    then reads); `model` = the instantiated module, for the per-layer entries."""
    if profile is None:
        from verity_vllm.program.frontend.target_profile import TargetProfile
        profile = TargetProfile.default()
    with (tensor_data_as_detach(), apply_target_profile(profile), mxfp4_marlin_moe_leaf(model), moe_ops_traced(model), dsa_ops_leaf(model),
          fused_allreduce_rms_norm_leaf(model), moe_gate_leaf(model), default_fp8_linear_leaf(model), cuda_streams_inert(),
          registered_op_bodies_as_ops(model), mhc_pre_broadcast_leaf(model), dsv4_attention_leaves(model),
          deep_gemm_fp8_einsum_leaf(model)):
        yield


__all__ = ["export_compat", "tensor_data_as_detach", "apply_target_profile", "select_cuda_platform", "moe_ops_traced", "dsa_ops_leaf", "dsa_leaves",
           "fused_allreduce_rms_norm_leaf", "moe_gate_leaf", "default_fp8_linear_leaf", "cuda_streams_inert", "InertStream", "InertEvent",
           "registered_op_bodies_as_ops", "mhc_pre_broadcast_leaf", "deep_gemm_fp8_einsum_leaf", "read_back_applied", "host_facts"]
