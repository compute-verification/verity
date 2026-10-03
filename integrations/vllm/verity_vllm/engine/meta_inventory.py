"""Instantiate a checkpoint's vLLM model class on the meta device, without its weights or a GPU, and list its modules.

What a capture map is checked against (`verity_vllm.correspondence.capture_map`): every module the pinned vLLM builds for a
`config.json`, with its class, its parameters' and buffers' per-rank shapes and dtypes, and the attributes that select its
kernels (`quant_method`, an attention layer's backend, a fused-MoE layer's router), at one tensor-parallel rank.

No GPU is touched: `CUDA_VISIBLE_DEVICES` is emptied before torch loads, every tensor vLLM places on `cuda` is made on
`meta`, and vLLM's platform is the CUDA platform reporting the `--capability` and `--device-name` of the target device (an
H200 for a model our nodes cannot serve), so the model code takes the branches it takes there.  The tensor-parallel group
is a stand-in of size `--tp` at `--rank`; no collective runs.  The same `inventory(model)` lists a live model's modules on a
serving rank (the go/no-go check of a capture map on the target): `live_document(llm.collective_rpc(rank_inventory))`.

Run with the pinned vLLM's interpreter; stdlib, torch and vLLM only, so the file runs on its own:

    python meta_inventory.py --repo zai-org/GLM-5.3 --revision <sha> --tp 8 --ranks 0 7 --capability 9.0 --out inv.json
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

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


def _hide_gpus() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "1")


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
    """On a serving rank (`LLM.collective_rpc(rank_inventory)`): `inventory` of its loaded model, its tensor-parallel rank and the
    switches `_selections` reads."""
    from vllm.distributed import get_tensor_model_parallel_rank

    return {"rank": get_tensor_model_parallel_rank(), "selections": _selections(), **inventory(worker.get_model())}


def live_document(per_rank: list[dict[str, Any]], **fields: Any) -> dict[str, Any]:
    """`rank_inventory`'s records from every rank as one document in a meta build's layout (`capture_map check --inventory`)."""
    return {"schema": SCHEMA, **fields, "selections": per_rank[0]["selections"],
            "ranks": {str(r["rank"]): r for r in sorted(per_rank, key=lambda r: r["rank"])}}


# ----------------------------------------------------------------------------------------------------- the meta instantiation


def _target_platform(capability: tuple[int, int], name: str, memory_gb: int, world: int):
    from vllm.platforms.cuda import CudaPlatform
    from vllm.platforms.interface import DeviceCapability

    class Target(CudaPlatform):  # the CUDA platform of the target device, read from flags, not from this host's GPUs
        @classmethod
        def get_device_capability(cls, device_id: int = 0):
            return DeviceCapability(major=capability[0], minor=capability[1])

        @classmethod
        def get_device_name(cls, device_id: int = 0) -> str:
            return name

        @classmethod
        def get_device_total_memory(cls, device_id: int = 0) -> int:
            return memory_gb << 30

        @classmethod
        def device_count(cls) -> int:
            return world

        @classmethod
        def is_fully_connected(cls, physical_device_ids) -> bool:
            return True

    Target.__name__ = Target.__qualname__ = f"{CudaPlatform.__name__}AsTarget"
    return Target()


class _Group:
    """A stand-in for vLLM's GroupCoordinator: its size and this rank, for the code that reads them while building a model."""

    def __init__(self, name: str, world: int, rank: int) -> None:
        self.unique_name, self.world_size, self.rank_in_group, self.rank, self.local_rank = f"{name}:0", world, rank, rank, rank
        self.ranks = list(range(world))
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == world - 1
        self.use_all2all = False
        self.device_group = self.cpu_group = self
        self.device_communicator = None

    def size(self) -> int:
        return self.world_size

    def __getattr__(self, name: str):
        raise AttributeError(f"stand-in {self.unique_name} group has no {name!r} (a collective or a communicator attribute)")


def _fake_parallel(world: int, rank: int) -> None:
    import vllm.distributed.parallel_state as ps

    ps._WORLD = _Group("world", world, rank)
    ps._TP = _Group("tp", world, rank)
    ps._EP = _Group("ep", world, rank)
    for g in ("_PP", "_DP", "_DCP", "_PCP"):
        if hasattr(ps, g):
            setattr(ps, g, _Group(g[1:].lower(), 1, 0))
    if hasattr(ps, "_EPLB"):
        ps._EPLB = None


class _MetaDevices:
    """Every tensor made on `cuda` is made on `meta`; host tensors are not pinned (no CUDA driver is loaded)."""

    def __init__(self) -> None:
        import torch
        from torch.overrides import TorchFunctionMode

        class Mode(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                kwargs = dict(kwargs or {})
                dev = kwargs.get("device")
                if dev is not None and str(dev).startswith("cuda"):
                    kwargs["device"] = torch.device("meta")
                if kwargs.get("pin_memory"):
                    kwargs["pin_memory"] = False
                if func in (torch.Tensor.cuda,):
                    return args[0]
                if func in (torch.Tensor.to,) and len(args) > 1 and (isinstance(args[1], (str, torch.device)) and str(args[1]).startswith("cuda")):
                    args = (args[0], torch.device("meta"), *args[2:])
                if func is torch.Tensor.pin_memory:
                    return args[0]
                return func(*args, **kwargs)

        class Stream:  # streams and events a module makes while it is built (a shared expert's aux stream); never run
            def __init__(self, *a, **k) -> None:
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def __getattr__(self, name):
                return lambda *a, **k: None

        class Props:
            def __init__(self, name: str, cap: tuple[int, int], memory: int, sms: int) -> None:
                self.name, (self.major, self.minor), self.total_memory, self.multi_processor_count = name, cap, memory, sms

        cuda = torch.cuda
        self.cuda_stubs = {"Stream": Stream, "Event": Stream, "current_stream": lambda *a, **k: Stream(),
                           "default_stream": lambda *a, **k: Stream(), "set_stream": lambda *a, **k: None,
                           "stream": lambda *a, **k: Stream(), "synchronize": lambda *a, **k: None,
                           "is_current_stream_capturing": lambda: False, "current_device": lambda: 0,
                           "set_device": lambda *a, **k: None}
        self.props = Props
        self.saved = {k: getattr(cuda, k) for k in self.cuda_stubs}
        self.mode, self.meta = Mode(), torch.device("meta")

    def target(self, name: str, capability: tuple[int, int], memory_gb: int, sms: int) -> "_MetaDevices":
        p = self.props(name, capability, memory_gb << 30, sms)
        self.cuda_stubs.update({"get_device_properties": lambda *a, **k: p, "get_device_name": lambda *a, **k: name,
                                "get_device_capability": lambda *a, **k: capability})
        import torch

        self.saved.update({k: getattr(torch.cuda, k) for k in self.cuda_stubs if k not in self.saved})
        return self

    def __enter__(self):
        import torch

        for k, v in self.cuda_stubs.items():
            setattr(torch.cuda, k, v)
        self.meta.__enter__()
        self.mode.__enter__()
        return self

    def __exit__(self, *exc):
        import torch

        self.mode.__exit__(*exc)
        self.meta.__exit__(*exc)
        for k, v in self.saved.items():
            setattr(torch.cuda, k, v)


def fetch_config(repo: str, revision: str, into: Path) -> Path:
    """`config.json` alone from the hub at `revision` (never a weight file); the directory vLLM reads it from."""
    from huggingface_hub import hf_hub_download

    p = Path(hf_hub_download(repo_id=repo, filename="config.json", revision=revision, cache_dir=str(into)))
    return p.parent


def build(model_dir: Path, *, tp: int, rank: int, capability: tuple[int, int], device_name: str, memory_gb: int,
          num_layers: int | None, max_model_len: int, kv_cache_dtype: str, attention_backend: str | None, env: dict[str, str]):
    import torch
    import vllm.platforms

    for k, v in env.items():
        os.environ[k] = v
    if not type(vllm.platforms.__dict__.get("current_platform")).__name__.endswith("AsTarget"):
        vllm.platforms.current_platform = _target_platform(capability, device_name, memory_gb, tp)
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.utils import get_model_architecture, initialize_model

    overrides: dict[str, Any] = {}
    if num_layers is not None:
        overrides["num_hidden_layers"] = num_layers
    kw: dict[str, Any] = dict(model=str(model_dir), skip_tokenizer_init=True, tensor_parallel_size=tp, enforce_eager=True,
                              max_model_len=max_model_len, load_format="dummy", kv_cache_dtype=kv_cache_dtype, seed=0,
                              hf_overrides=overrides, generation_config="vllm")
    if attention_backend:
        kw["attention_backend"] = attention_backend
    args = EngineArgs(**kw)
    vcfg = args.create_engine_config()
    _fake_parallel(tp, rank)
    cls, arch = get_model_architecture(vcfg.model_config)
    torch.set_default_dtype(vcfg.model_config.dtype)
    t0 = time.time()
    with _MetaDevices().target(device_name, capability, memory_gb, 132):
        model = initialize_model(vcfg)
    return model, cls, arch, vcfg, time.time() - t0


def _selections() -> dict[str, Any]:
    """Process-wide switches that change a kernel's numbers without changing a module (read after the model is built)."""
    out: dict[str, Any] = {}
    try:
        from vllm.utils import deep_gemm

        out["deep_gemm_supported"] = bool(deep_gemm.is_deep_gemm_supported())
        out["deep_gemm_e8m0"] = bool(deep_gemm.is_deep_gemm_e8m0_used())
    except Exception as e:  # recorded, so a missing switch is visible rather than defaulted
        out["deep_gemm_error"] = repr(e)
    import vllm.envs as envs

    for k in ("VLLM_BATCH_INVARIANT", "VLLM_USE_DEEP_GEMM", "VLLM_MOE_USE_DEEP_GEMM", "VLLM_USE_DEEP_GEMM_E8M0",
              "VLLM_USE_FUSED_MOE_GROUPED_TOPK", "VLLM_ALLREDUCE_USE_FLASHINFER", "VLLM_ALLREDUCE_USE_SYMM_MEM"):
        try:
            out[k] = getattr(envs, k)
        except AttributeError:
            out[k] = None
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True, help="the full commit sha of the checkpoint")
    ap.add_argument("--config-dir", type=Path, default=None, help="a directory holding config.json (else it is fetched)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--ranks", type=int, nargs="+", default=[0])
    ap.add_argument("--capability", default="9.0", help="the target's compute capability, e.g. 9.0 for an H200")
    ap.add_argument("--device-name", default="NVIDIA H200")
    ap.add_argument("--memory-gb", type=int, default=141)
    ap.add_argument("--num-layers", type=int, default=None, help="override num_hidden_layers (default: the config's)")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--attention-backend", default=None)
    ap.add_argument("--env", action="append", default=[], help="K=V set before vLLM reads its environment (e.g. VLLM_BATCH_INVARIANT=1)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    _hide_gpus()
    env = dict(kv.split("=", 1) for kv in a.env)
    cap = tuple(int(x) for x in a.capability.split("."))
    scratch = Path(tempfile.mkdtemp(prefix="meta-inventory-"))
    model_dir = a.config_dir or fetch_config(a.repo, a.revision, scratch)
    os.environ["HF_HUB_OFFLINE"] = "1"
    cfg_bytes = (model_dir / "config.json").read_bytes()
    import torch
    import vllm

    doc: dict[str, Any] = {"schema": SCHEMA, "repo": a.repo, "revision": a.revision,
                           "config_sha256": hashlib.sha256(cfg_bytes).hexdigest(), "vllm": vllm.__version__,
                           "torch": torch.__version__, "target": {"compute_capability": list(cap), "device_name": a.device_name,
                                                                  "memory_gb": a.memory_gb},
                           "tp": a.tp, "num_layers": a.num_layers, "max_model_len": a.max_model_len, "kv_cache_dtype": a.kv_cache_dtype,
                           "attention_backend": a.attention_backend, "env": env, "ranks": {}}
    for r in a.ranks:
        model, cls, arch, vcfg, secs = build(model_dir, tp=a.tp, rank=r, capability=cap, device_name=a.device_name,
                                             memory_gb=a.memory_gb, num_layers=a.num_layers, max_model_len=a.max_model_len,
                                             kv_cache_dtype=a.kv_cache_dtype, attention_backend=a.attention_backend, env=env)
        inv = inventory(model)
        doc["architecture"] = arch
        doc["model_class"] = f"{cls.__module__}.{cls.__qualname__}"
        doc["quantization"] = vcfg.model_config.quantization
        doc["cache_dtype"] = vcfg.cache_config.cache_dtype
        doc["selections"] = _selections()
        doc["ranks"][str(r)] = {"seconds": round(secs, 2), **inv}
        print(f"[meta-inventory] rank {r}/{a.tp}: {cls.__qualname__} {inv['modules']} modules, {len(inv['module_types'])} types, "
              f"{secs:.1f}s", flush=True)
        del model
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=False))
    print(f"[meta-inventory] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
