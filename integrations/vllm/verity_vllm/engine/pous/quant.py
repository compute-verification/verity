"""`quantization="pous"`: vLLM serves the model from its P3 encoding `C`, decoded on the GPU at every use.

Loading is stock: `create_weights` makes plain parameters, so vLLM's loaders do the merges (qkv, gate_up).  At the first
`process_weights_after_loading` the store reads `W` (the checkpoint's data section, `layout.py`), encodes it on the GPU under the
verifier's salt (`codec.encode`) and keeps only `C`.  Each module then checks that decoding its checkpoint tensors from `C`
reproduces the loaded parameter bit for bit, and drops the parameter.  At run time a module's `apply` (or `embedding`) asks the
store for its weight, which is decoded from `C` for its group (`layers_per_decode` transformer layers, default one) at the group's
first use in a forward and released after its last; nothing decoded outlives the group's use within the pass.  The GEMM is
`UnquantizedLinearMethod.apply`'s, so the served bits are the plain model's.

The run's settings are the JSON file vLLM passes as `hf_overrides["quantization_config_file"]`: `safetensors` (the checkpoint
file), `build_dir` (the native build, `native.load`), `salt` (hex, 24 bytes, drawn by the verifier after the checkpoint
commitment), `layers_per_decode`, `thread_rows` (`codec.THREAD_ROWS`), and `drop_fraction` / `drop_source` (the negative control: that fraction of `C`'s blocks, drawn
through `verity.randomness` from `drop_source`, is zeroed after encoding, and the load-time check is skipped).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from verity.randomness import derive
from verity_pous.verifier import leaf_hash
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod, VocabParallelEmbedding

from verity_vllm.engine.pous import codec, layout, native

DROP_DOMAIN = "verity/pous-vllm/negative-control-drop/v1"
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
_LAYER = re.compile(r"^(.*\.layers\.)(\d+)\.")
_STORE: list[Store] = []


def current_store() -> Store:
    if not _STORE:
        raise RuntimeError("no POUS store in this process (was the engine built with quantization='pous'?)")
    return _STORE[0]


class _Weight:
    """What `UnquantizedLinearMethod.apply` reads off a layer: its weight."""

    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight


class Store:
    """`C` on the device, the layout of `W`, and the per-layer decode."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.checkpoint = layout.read(spec["safetensors"])
        self.gpu_encode, self.gpu_decode = codec.twin(codec.REFERENCE)
        codec.THREAD_ROWS[0] = int(spec.get("thread_rows") or 0)
        self.salt = bytes.fromhex(spec["salt"])
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.c: torch.Tensor | None = None
        self.consumers: dict[str, dict[str, list[str]]] = {}
        self.cache: dict[str, dict[str, Any]] = {}
        self.dropped: list[int] = []
        self.stats: dict[str, Any] = {"decodes": 0, "decode_s": 0.0, "decoded_bytes": 0, "checks": {}, "digests": {}}

    @property
    def segments(self) -> int:
        return self.checkpoint.params.segments

    @property
    def blocks(self) -> int:
        return self.checkpoint.params.blocks

    def encode(self) -> None:
        t0 = time.perf_counter()
        w = torch.frombuffer(bytearray(self.checkpoint.read_w()), dtype=torch.uint8).to(self.device)
        w = w.view(self.segments, codec.N, codec.M)
        tags = codec.segment_tags(self.salt, list(range(self.segments)), self.device)
        self.c = self.gpu_encode(w, tags)
        del w
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.stats["encode_s"] = time.perf_counter() - t0
        print(f"[pous] encoded {self.segments} segments in {self.stats['encode_s']:.1f} s, allocated "
              f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB", flush=True)
        frac = float(self.spec.get("drop_fraction") or 0.0)
        if frac > 0:
            key = derive(bytes.fromhex(self.spec["drop_source"]), DROP_DOMAIN, {"salt": self.salt})
            self.dropped = key.subset(int(round(frac * self.blocks)), self.blocks, ())
            idx = torch.tensor(self.dropped, dtype=torch.long, device=self.device)
            self.c.view(self.blocks, codec.M)[idx] = 0

    def decode(self, names: list[str]) -> dict[str, torch.Tensor]:
        """The checkpoint tensors `names`, decoded from the segments that hold them (in one batch)."""
        t0 = time.perf_counter()
        lay = self.checkpoint.layout
        segs = sorted({s for n in names for s in lay.segments(n)})
        w = self.gpu_decode(self.c[torch.tensor(segs, device=self.device)], codec.segment_tags(self.salt, segs, self.device))
        flat = w.view(len(segs), codec.SEGMENT_BYTES)
        at = {s: i for i, s in enumerate(segs)}
        out = {}
        for n in names:
            p, r = lay.placement(n), lay.segments(n)
            start = p.offset - r.start * codec.SEGMENT_BYTES
            raw = flat[at[r.start]:at[r.start] + len(r)].view(-1)[start:start + p.size]
            out[n] = raw.view(DTYPES[self.checkpoint.dtypes[n]]).view(self.checkpoint.shapes[n])
        torch.cuda.synchronize()
        self.stats["decodes"] += 1
        self.stats["decode_s"] += time.perf_counter() - t0
        self.stats["decoded_bytes"] += len(segs) * codec.SEGMENT_BYTES
        return out

    def group_of(self, prefix: str) -> str:
        """The modules decoded together: `layers_per_decode` consecutive transformer layers, or the module alone."""
        m = _LAYER.match(prefix)
        if not m:
            return prefix
        return f"{m.group(1)}{int(m.group(2)) // int(self.spec.get('layers_per_decode') or 1)}"

    def register(self, prefix: str, names: list[str], loaded: torch.Tensor) -> None:
        """Module `prefix` computes with the concatenation (dim 0) of `names`; unless blocks were dropped, decoding them must
        reproduce the loaded parameter bit for bit."""
        self.consumers.setdefault(self.group_of(prefix), {})[prefix] = names
        if self.dropped:
            return
        parts = self.decode(names)
        got = self._join(parts, names)
        same = got.shape == loaded.shape and got.dtype == loaded.dtype and torch.equal(got.view(torch.uint8), loaded.view(torch.uint8))
        self.stats["checks"][prefix] = bool(same)
        for n in names:
            self.stats["digests"][n] = hashlib.sha256(parts[n].contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        del parts, got
        torch.cuda.empty_cache()
        print(f"[pous] {prefix} checked={same} allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB "
              f"reserved={torch.cuda.memory_reserved() / 2**30:.2f} GiB", flush=True)
        if not same:
            raise RuntimeError(f"POUS: decoding {names} from C does not reproduce the loaded {prefix} weight")

    @staticmethod
    def _join(parts: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor:
        return parts[names[0]] if len(names) == 1 else torch.cat([parts[n] for n in names], 0)

    def take(self, prefix: str) -> torch.Tensor:
        """Module `prefix`'s weight for this use: its group is decoded at its first use in a pass and dropped after its last."""
        group = self.group_of(prefix)
        members = self.consumers[group]
        entry = self.cache.get(group)
        if entry is None or prefix in entry["taken"]:
            names = sorted({n for ns in members.values() for n in ns})
            entry = {"parts": self.decode(names), "taken": set()}
            self.cache[group] = entry
        entry["taken"].add(prefix)
        w = self._join(entry["parts"], members[prefix])
        if entry["taken"] == set(members):
            del self.cache[group]
        return w

    def info(self) -> dict[str, Any]:
        return {"segments": self.segments, "blocks": self.blocks, "block_bytes": codec.M, "codec": codec.REFERENCE.name,
                "gpu_twin": codec.NAME, "key_mode": codec.MODE,
                "w_bytes": self.checkpoint.params.w_bytes, "data_bytes": self.checkpoint.data_bytes, "c_bytes": int(self.c.numel()),
                "dropped": len(self.dropped), "cached_groups": len(self.cache), **{k: v for k, v in self.stats.items()}}

    def leaves(self) -> list[bytes]:
        """The reference's leaf digests (`verity_pous.verifier.leaf_hash`) of every block of `C`."""
        flat = self.c.view(self.blocks, codec.M)
        params = self.checkpoint.params
        out = []
        for i in range(0, self.blocks, 1024):
            for j, row in enumerate(flat[i:i + 1024].cpu().numpy()):
                out.append(leaf_hash(params, self.salt, i + j, row.tobytes()))
        return out


def _checkpoint_names(cfg: PousConfig, prefix: str, tensors: dict) -> list[str]:
    """The checkpoint tensors a module's weight is made of, in the order vLLM's loader stacks them."""
    if prefix + ".weight" in tensors:
        return [prefix + ".weight"]
    head, _, leaf = prefix.rpartition(".")
    parts = cfg.packed_modules_mapping.get(leaf)
    if parts:
        names = [f"{head}.{p}.weight" for p in parts]
        if all(n in tensors for n in names):
            return names
    if leaf == "lm_head" and "model.embed_tokens.weight" in tensors:
        return ["model.embed_tokens.weight"]
    raise KeyError(f"POUS: no checkpoint tensors for module {prefix!r}")


class PousLinearMethod(UnquantizedLinearMethod):
    def __init__(self, cfg: PousConfig, prefix: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.prefix = prefix

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        store = self.cfg.store()
        store.register(self.prefix, _checkpoint_names(self.cfg, self.prefix, store.checkpoint.dtypes), layer.weight.data)
        layer.weight = torch.nn.Parameter(torch.empty(0, dtype=layer.weight.dtype, device=layer.weight.device), requires_grad=False)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return super().apply(_Weight(self.cfg.store().take(self.prefix)), x, bias)


class PousEmbeddingMethod(UnquantizedEmbeddingMethod):
    def __init__(self, cfg: PousConfig, prefix: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.prefix = prefix

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        store = self.cfg.store()
        store.register(self.prefix, _checkpoint_names(self.cfg, self.prefix, store.checkpoint.dtypes), layer.weight.data)
        layer.weight = torch.nn.Parameter(torch.empty(0, dtype=layer.weight.dtype, device=layer.weight.device), requires_grad=False)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return super().apply(_Weight(self.cfg.store().take(self.prefix)), x, bias)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return super().embedding(_Weight(self.cfg.store().take(self.prefix)), input_)


@register_quantization_config("pous")
class PousConfig(QuantizationConfig):
    def __init__(self, spec: dict[str, Any]) -> None:
        super().__init__()
        self.spec = spec

    def store(self) -> Store:
        if not _STORE:
            native.load(self.spec["build_dir"])
            s = Store(self.spec)
            s.encode()
            _STORE.append(s)
        return _STORE[0]

    def get_name(self) -> str:
        return "pous"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> PousConfig:
        return cls(config)

    @classmethod
    def from_config_file(cls, path: str) -> PousConfig:
        return cls(json.loads(Path(path).read_text()))

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            return PousLinearMethod(self, prefix)
        if isinstance(layer, VocabParallelEmbedding):
            return PousEmbeddingMethod(self, prefix)
        return None
