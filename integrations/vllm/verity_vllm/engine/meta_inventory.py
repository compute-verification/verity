"""Instantiate a checkpoint's vLLM model class on the meta device, without its weights or a GPU, and list its modules.

What a capture map is checked against (`verity_vllm.check.capture_map`): every module the pinned vLLM builds for a
`config.json`, with its class, its parameters' and buffers' per-rank shapes and dtypes, and the attributes that select its
kernels (`quant_method`, an attention layer's backend, a fused-MoE layer's router), at one tensor-parallel rank.

No GPU is touched: no CUDA device is visible to the process (`engine.env.hide_devices`), every tensor vLLM places on `cuda`
is made on `meta`, and vLLM's platform is the CUDA platform answering the declared `TargetProfile` (`--target`: compute
capability, SM count, switches) and the target device's name and memory, so the model code takes the branches it takes on
the target (an H200 for a model our nodes cannot serve).  The host's toolchain is the target's too: vLLM counts FlashInfer as
present only with its cubins or `nvcc` on PATH, as on the pods, so run it with the CUDA toolkit's `bin` on PATH (`selections`
records `has_flashinfer`).  Each rank's tensor-parallel group is the Build's stand-in
(`export_ops.tp_group_override`); no collective runs.  The same `inventory(model)` lists a live model's modules on a serving
rank (the go/no-go check of a capture map on the target): `live_document(llm.collective_rpc(rank_inventory))`.

    verity-vllm meta-inventory --repo zai-org/GLM-5.3 --revision <sha> --tp 8 --ranks 0 1 2 3 4 5 6 7 \\
        --target '{"compute_capability": [9, 0], "num_sms": 132, "batch_invariant": false, "fp8_block_gemm": "cutlass"}' \\
        --device-name "NVIDIA H200" --device-memory-gb 141 --out inv.json
"""
from __future__ import annotations

import hashlib
import inspect
import json
import re
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity_vllm.config import option
from verity_vllm.engine import env as engine_env, hooks

SCHEMA = "verity-vllm/meta-inventory/v1"

#: attributes whose value says which kernel a module runs; recorded by type name (objects) or value (scalars)
KERNEL_ATTRS = ("quant_method", "attn_backend", "impl", "router", "runner", "quant_config", "kernel", "linear_method",
                "fp8_linear", "w8a8_block_fp8_linear", "moe_kernel", "fused_experts", "indexer_op", "prefill_backend")
SCALAR_ATTRS = ("is_neox_style", "rotary_dim", "head_size", "max_position_embeddings", "base", "kv_cache_dtype", "scoring_func",
                "top_k", "use_grouped_topk", "num_expert_group", "topk_group", "routed_scaling_factor", "renormalize", "out_dtype",
                "variance_epsilon", "eps", "skip_topk", "topk_tokens", "n_head", "head_dim", "reduce_results", "tp_size", "tp_rank",
                "input_size", "output_size", "input_size_per_partition", "output_size_per_partition", "output_partition_sizes",
                "local_num_experts", "global_num_experts", "intermediate_size_per_partition", "num_heads", "num_local_heads",
                "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "q_lora_rank", "scale", "num_embeddings",
                "num_embeddings_per_partition", "embedding_dim", "shard_indices", "router_dtype", "apply_routed_scale_to_output",
                "use_sparse", "layer_name", "softmax_scale", "quant_block_size", "scale_fmt", "return_bias", "skip_bias_add",
                "disable_tp", "is_sequence_parallel", "use_mha", "use_pcp", "logits_as_input", "soft_cap", "vocab_size",
                "org_vocab_size", "_use_min_latency_gemm", "_index_rope_interleave", "_fp8_query", "dtype",
                "supports_dense_mha_prefill", "_dense_mha_metadata_layer_name")
_LAYER = re.compile(r"\.(\d+)(?=\.|$)")


def _type_name(obj: Any) -> str:
    t = type(obj)
    return f"{t.__module__}.{t.__qualname__}"


def _scalar(v: Any) -> Any:
    import torch

    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, torch.dtype):
        return str(v).replace("torch.", "")
    if isinstance(v, (list, tuple)) and len(v) <= 16 and all(isinstance(x, (bool, int, float, str)) for x in v):
        return list(v)
    return f"<{_type_name(v)}>"


def describe(obj: Any, depth: int = 1) -> dict[str, Any]:
    """An object's type and, `depth` levels down, its kernel-selecting attributes (objects by type, scalars by value)."""
    out: dict[str, Any] = {"type": _type_name(obj)}
    if depth <= 0:
        return out
    for k, v in sorted(getattr(obj, "__dict__", {}).items()):
        if k.startswith("__") or callable(v) and not hasattr(v, "__dict__"):
            continue
        if isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
        elif k in KERNEL_ATTRS or k.endswith(("kernel", "_method", "_backend", "_impl", "_op")):
            out[k] = describe(v, depth - 1)
    return out


def _source(cls: type) -> dict[str, Any] | None:
    fwd = cls.__dict__.get("forward") or next((c.__dict__["forward"] for c in cls.__mro__ if "forward" in c.__dict__), None)
    if fwd is None:
        return None
    try:
        lines, start = inspect.getsourcelines(fwd)
        path = inspect.getsourcefile(fwd) or "?"
    except (OSError, TypeError):
        return None
    rel = path.split("site-packages/", 1)[-1]
    return {"forward": f"{rel}:{start}", "forward_sha256": hashlib.sha256("".join(lines).encode()).hexdigest()[:16],
            "forward_owner": next(c.__qualname__ for c in cls.__mro__ if "forward" in c.__dict__)}


def _tensors(mod: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind, items in (("param", mod.named_parameters(recurse=False)), ("buffer", mod.named_buffers(recurse=False))):
        for n, t in items:
            out[n] = {"kind": kind, "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}
    for n, v in vars(mod).items():  # tensors held as plain attributes (an MLA layer's W_UK_T / W_UV, a top-k buffer)
        import torch

        if isinstance(v, torch.Tensor) and n not in out and not n.startswith("_parameters"):
            out[n] = {"kind": "attr", "shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")}
    return out


def inventory(model: Any) -> dict[str, Any]:
    """Every module of `model` (path, class, defining module, forward source, per-rank tensors, kernel attributes), and the
    same grouped by its path with layer indices replaced by `{i}`."""
    mods = []
    for path, m in model.named_modules():
        cls = type(m)
        rec: dict[str, Any] = {"path": path, "type": cls.__qualname__, "module": cls.__module__, "tensors": _tensors(m)}
        attrs = {k: _scalar(getattr(m, k)) for k in SCALAR_ATTRS if k in vars(m) or any(k in c.__dict__ for c in cls.__mro__)}
        attrs = {k: v for k, v in attrs.items() if v is not None and not (isinstance(v, str) and v.startswith("<bound"))}
        if attrs:
            rec["attrs"] = attrs
        flags = {k: v for k, v in sorted(vars(m).items()) if isinstance(v, bool) and k != "training" and not k.startswith("__")}
        if flags:  # the booleans a forward branches on (a router GEMM's tiers, a fused path's switch), recorded whole
            rec["flags"] = flags
        kern = {k: describe(getattr(m, k), 2) for k in KERNEL_ATTRS if getattr(m, k, None) is not None and not callable(getattr(m, k))}
        if kern:
            rec["kernels"] = kern
        src = _source(cls)
        if src:
            rec["source"] = src
        mods.append(rec)
    groups: dict[str, dict[str, Any]] = {}
    for r in mods:
        layers = [int(x) for x in _LAYER.findall(r["path"])]
        pattern = _LAYER.sub(".{i}", r["path"], count=1)
        g = groups.setdefault(pattern + "|" + r["type"], {"pattern": pattern, "type": r["type"], "module": r["module"], "layers": [],
                                                         "shapes": {}, "example": r["path"]})
        if layers:
            g["layers"].append(layers[0])
        sig = json.dumps(r["tensors"], sort_keys=True)
        g["shapes"].setdefault(sig, []).append(layers[0] if layers else -1)
        for k in ("attrs", "flags", "kernels", "source"):
            if k in r and k not in g:
                g[k] = r[k]
    for g in groups.values():
        g["shapes"] = [{"tensors": json.loads(s), "layers": ls} for s, ls in g["shapes"].items()]
    types: dict[str, int] = {}
    for r in mods:
        types[r["type"]] = types.get(r["type"], 0) + 1
    out = {"modules": len(mods), "module_types": dict(sorted(types.items())), "groups": sorted(groups.values(), key=lambda g: g["pattern"])}
    out["signature"] = signature(out)
    return out


def signature(inv: dict[str, Any]) -> str:
    """sha256 of what a capture map is checked against: each group's path pattern, class, layers and per-rank tensor shapes
    (not the rank-dependent attribute values), so the ranks of one build compare by one digest.  A loaded model holds tensors
    a meta build does not (an MLA layer's W_UK_T and W_UV are made from kv_b_proj after loading), so a live rank's digest is
    compared with the other live ranks', not with the meta build's."""
    rows = [[g["pattern"], g["type"], g["layers"], g["shapes"]] for g in inv["groups"]]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rank_inventory(worker: Any) -> dict[str, Any]:
    """On a serving rank (`LLM.collective_rpc(rank_inventory)`): `inventory` of its loaded model, its tensor-parallel rank, the
    device facts vLLM answers there (`applied`) and the switches `_selections` reads."""
    from vllm.distributed import get_tensor_model_parallel_rank

    from verity_vllm.program.frontend.export_compat import read_back_applied

    return {"rank": get_tensor_model_parallel_rank(), "applied": read_back_applied(), "selections": _selections(),
            **inventory(worker.get_model())}


def live_document(per_rank: list[dict[str, Any]], **fields: Any) -> dict[str, Any]:
    """`rank_inventory`'s records from every rank as one document in a meta build's layout (`capture-map check --inventory`)."""
    return {"schema": SCHEMA, **fields, "ranks": {str(r["rank"]): r for r in sorted(per_rank, key=lambda r: r["rank"])}}


def write_live(llm: Any, path: Path, *, timeout: float | None = None, **fields: Any) -> dict[str, Any]:
    """`live_document` of every rank of a loaded `llm` (`LLM.collective_rpc(rank_inventory)`), with `fields`, written to `path`."""
    doc = live_document(llm.collective_rpc(rank_inventory, timeout=timeout), source="live", **fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return doc


# ----------------------------------------------------------------------------------------------------- the meta instantiation


class _Stream:
    """A stream or event a module makes while it is built (a shared expert's aux stream); never run."""

    def __init__(self, *a: Any, **k: Any) -> None:
        pass

    def __enter__(self) -> "_Stream":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return lambda *a, **k: None


@contextmanager
def target_device(target: Any, device_name: str, memory_gb: int):
    """For the model's construction: every tensor vLLM makes on `cuda` is made on `meta`, host tensors are not pinned, and the
    device queries of vLLM's platform and of `torch.cuda` answer the target device (`target`, a `TargetProfile`, and its name
    and memory) without a CUDA driver.  Inside `export_compat.apply_target_profile(target)`, which answers the platform's
    capability and SM count."""
    import torch
    from torch.overrides import TorchFunctionMode
    from vllm.platforms import current_platform

    meta = torch.device("meta")

    class Mode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            dev = kwargs.get("device")
            if dev is not None and str(dev).startswith("cuda"):
                kwargs["device"] = meta
            if kwargs.get("pin_memory"):
                kwargs["pin_memory"] = False
            if func is torch.Tensor.cuda or func is torch.Tensor.pin_memory:
                return args[0]
            if func is torch.Tensor.to and len(args) > 1 and isinstance(args[1], (str, torch.device)) \
                    and str(args[1]).startswith("cuda"):
                args = (args[0], meta, *args[2:])
            return func(*args, **kwargs)

    cap, memory = tuple(target.compute_capability), memory_gb << 30
    props = SimpleNamespace(name=device_name, major=cap[0], minor=cap[1], total_memory=memory,
                            multi_processor_count=target.num_sms)
    cuda = {"Stream": _Stream, "Event": _Stream, "current_stream": lambda *a, **k: _Stream(),
            "default_stream": lambda *a, **k: _Stream(), "set_stream": lambda *a, **k: None,
            "stream": lambda *a, **k: _Stream(), "synchronize": lambda *a, **k: None,
            "is_current_stream_capturing": lambda: False, "current_device": lambda: 0, "set_device": lambda *a, **k: None,
            "get_device_properties": lambda *a, **k: props, "get_device_name": lambda *a, **k: device_name,
            "get_device_capability": lambda *a, **k: cap}
    with hooks.Hooks() as h:
        for name, value in cuda.items():
            h.patch(torch.cuda, name, value)
        platform = type(current_platform)
        h.patch(platform, "get_device_name", classmethod(lambda cls, device_id=0: device_name))
        h.patch(platform, "get_device_total_memory", classmethod(lambda cls, device_id=0: memory))
        h.patch(platform, "is_fully_connected", classmethod(lambda cls, physical_device_ids: True))
        with meta, Mode():
            yield


def fetch_config(repo: str, revision: str, into: Path) -> Path:
    """`config.json` alone from the hub at `revision` (never a weight file); the directory vLLM reads it from."""
    from huggingface_hub import hf_hub_download

    p = Path(hf_hub_download(repo_id=repo, filename="config.json", revision=revision, cache_dir=str(into)))
    return p.parent


def build(model_dir: Path, target: Any, *, tp: int, rank: int, device_name: str, memory_gb: int, max_model_len: int,
          kv_cache_dtype: str, num_layers: int | None = None) -> tuple[Any, Any, float]:
    """Rank `rank` of `tp` of the model `model_dir/config.json` describes, on `meta` (no weights), with its vLLM config and
    the seconds it took.  Call inside `export_compat.apply_target_profile(target)`, once `prepare` has run."""
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.utils.torch_utils import set_default_torch_dtype

    from verity_vllm.program.frontend import vllm_meta as vm
    from verity_vllm.program.frontend.export_ops import CollectiveBus, tp_group_override

    overrides = {} if num_layers is None else {"num_hidden_layers": num_layers}
    cfg = vm.make_vllm_config(model=str(model_dir), revision=None, dtype="auto", max_model_len=max_model_len,
                              tensor_parallel_size=tp, disable_custom_all_reduce=False, load_format="dummy",
                              kv_cache_dtype=kv_cache_dtype, seed=0, hf_overrides=overrides, generation_config="vllm",
                              **target.engine_args(execution=False))
    vm.ensure_distributed(cfg, "gloo")
    t0 = time.perf_counter()
    # the loader's entry, not `vm.instantiate_meta`: it gives the quant config the model's packed-module mapping, which decides
    # which fused linears a checkpoint's `modules_to_not_convert` leaves unquantized
    with tp_group_override(rank, tp, CollectiveBus(rank=rank, world=tp)), set_default_torch_dtype(cfg.model_config.dtype), \
            target_device(target, device_name, memory_gb):
        model = initialize_model(cfg)
    return model, cfg, time.perf_counter() - t0


def prepare(target: Any, tp: int) -> str:
    """The process before vLLM is imported: no CUDA device visible, the target's switches and the meta build's pins in the
    environment, and vLLM's platform the CUDA platform of `tp` devices.  Returns the platform class vLLM resolved."""
    engine_env.hide_devices()
    engine_env.apply_target(target.env())
    from verity_vllm.program.frontend.export_compat import select_cuda_platform

    return select_cuda_platform(tp)


def _selections() -> dict[str, Any]:
    """Process-wide switches that change a kernel's numbers without changing a module (read after the model is built)."""
    out: dict[str, Any] = {}
    try:
        from vllm.utils import deep_gemm

        out["deep_gemm_supported"] = bool(deep_gemm.is_deep_gemm_supported())
        out["deep_gemm_e8m0"] = bool(deep_gemm.is_deep_gemm_e8m0_used())
    except Exception as e:  # recorded, so a missing switch is visible rather than defaulted
        out["deep_gemm_error"] = repr(e)
    try:
        # FlashInfer counts as present only with its cubins or `nvcc` on PATH (`has_flashinfer`), and its block-scale FP8 GEMM
        # leads the sm_90 FP8 linear's kernel list (`FlashInferFp8DeepGEMMDynamicBlockScaledKernel`)
        from vllm.utils import flashinfer

        out["flashinfer"] = bool(flashinfer.has_flashinfer())
        out["flashinfer_fp8_blockscale_gemm"] = bool(flashinfer.is_flashinfer_fp8_blockscale_gemm_supported())
    except Exception as e:
        out["flashinfer_error"] = repr(e)
    import vllm.envs as envs

    for k in ("VLLM_BATCH_INVARIANT", "VLLM_USE_DEEP_GEMM", "VLLM_MOE_USE_DEEP_GEMM", "VLLM_USE_DEEP_GEMM_E8M0",
              "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER", "VLLM_USE_FUSED_MOE_GROUPED_TOPK", "VLLM_ALLREDUCE_USE_FLASHINFER",
              "VLLM_ALLREDUCE_USE_SYMM_MEM"):
        try:
            out[k] = getattr(envs, k)
        except AttributeError:
            out[k] = None
    return out


@dataclass(frozen=True, kw_only=True)
class Options:
    "a checkpoint's modules, built on the meta device from its config.json alone, at each tensor-parallel rank"

    repo: str = option("--repo")
    revision: str = option("--revision", help="the checkpoint's full commit sha")
    config_dir: str | None = option("--config-dir", default=None,
                                    help="a directory holding the checkpoint's config.json (default: fetch config.json alone)")
    target: str = option("--target", help="the target's TargetProfile: JSON overrides of the default profile, or a JSON file")
    device_name: str = option("--device-name", help="the target device's name, as vLLM's platform reports it")
    device_memory_gb: int = option("--device-memory-gb", type=int, help="the target device's memory (GiB)")
    tp: int = option("--tp", type=int, default=1)
    ranks: list[int] = option("--ranks", type=int, nargs="+", default=[0])
    num_layers: int | None = option("--num-layers", type=int, default=None, help="num_hidden_layers (default: the config's)")
    max_model_len: int = option("--max-model-len", type=int, default=4096)
    kv_cache_dtype: str = option("--kv-cache-dtype", default="auto")
    out: str = option("--out")


def main(a: Options) -> int:
    from verity_vllm.program.frontend.target_profile import TargetProfile, parse_target_arg

    target = parse_target_arg(a.target, base=TargetProfile.default())
    model_dir = Path(a.config_dir) if a.config_dir else fetch_config(a.repo, a.revision,
                                                                     Path(tempfile.mkdtemp(prefix="meta-inventory-")))
    platform = prepare(target, a.tp)
    import torch
    import vllm

    from verity_vllm.program.frontend.export_compat import apply_target_profile, read_back_applied

    doc: dict[str, Any] = {"schema": SCHEMA, "repo": a.repo, "revision": a.revision,
                           "config_sha256": hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest(),
                           "vllm": vllm.__version__, "torch": torch.__version__, "platform": platform,
                           "target": target.to_json(), "device": {"name": a.device_name, "memory_gb": a.device_memory_gb},
                           "env": target.env(), "tp": a.tp, "num_layers": a.num_layers, "max_model_len": a.max_model_len,
                           "kv_cache_dtype": a.kv_cache_dtype, "ranks": {}}
    with apply_target_profile(target):
        # inside the target: vLLM's kernel modules read the device capability when imported (`kernels.oink_ops`)
        from vllm.model_executor.model_loader.utils import get_model_architecture

        for r in a.ranks:
            model, cfg, secs = build(model_dir, target, tp=a.tp, rank=r, device_name=a.device_name,
                                     memory_gb=a.device_memory_gb, max_model_len=a.max_model_len,
                                     kv_cache_dtype=a.kv_cache_dtype, num_layers=a.num_layers)
            cls, arch = get_model_architecture(cfg.model_config)
            doc.update(architecture=arch, model_class=f"{cls.__module__}.{cls.__qualname__}",
                       quantization=cfg.model_config.quantization, cache_dtype=cfg.cache_config.cache_dtype)
            inv = inventory(model)
            doc["ranks"][str(r)] = {"seconds": round(secs, 2), "applied": read_back_applied(), "selections": _selections(),
                                    **inv}
            print(f"[meta-inventory] rank {r}/{a.tp}: {cls.__qualname__} {inv['modules']} modules, "
                  f"{len(inv['module_types'])} types, {secs:.1f}s", flush=True)
            del model
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=False))
    print(f"[meta-inventory] wrote {out}", flush=True)
    return 0
