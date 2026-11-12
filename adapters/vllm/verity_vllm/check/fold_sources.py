"""Where the fold compare's values come from: the profile and the folded log (`load_profile`, `fold_log`), the snapshots
of a capture (`SnapshotStore`) and the checkpoint's words by leaf interval of the weights Record (`WeightSource`, with the tables
that say which checkpoint tensors a field is).  `fold_compare` hands every name on; other code reads them through it.
Everything here is torch-free."""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from verity.circuits.types import Type

from verity_vllm.check.checkpoint_words import (
    MLA_LAYER_FIELD,
    load_safetensors_words,
    mla_field_words,
    mla_moe_sources,
)
from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.fold import FoldResult, fold
from verity_vllm.observe.fold.resolver import Profile
from verity_vllm.observe.log import read_log
from verity_vllm.program.frontend.rules.family import _field_intervals

DEFAULT_COS_SIN = Path(str(resources.files(__package__) / "tables" / "B0-divergence-20260907T1604Z" / "cos_sin_cache.npy"))
HF_GLOB = str(Path.home() / ".cache" / "huggingface" / "hub" / "models--HuggingFaceTB--SmolLM2-135M" / "snapshots" / "*" / "model.safetensors")


def hf_safetensors_glob(repo: str | None) -> str:
    """`model*.safetensors` of a checkpoint repo in the HF cache (`$HF_HOME/hub` or `~/.cache/huggingface/hub`); the B0
    default when no repo is known."""
    if not repo:
        return HF_GLOB
    from verity_vllm import settings
    home = Path(settings.HF_HOME or (Path.home() / ".cache" / "huggingface"))
    return str(home / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / "*" / "model*.safetensors")

_NP = {"bfloat16": np.uint16, "float16": np.uint16, "float32": np.uint32, "int32": np.uint32, "uint32": np.uint32,
       "int64": np.uint64, "uint64": np.uint64, "int8": np.uint8, "uint8": np.uint8, "bool": np.uint8, "int16": np.uint16}


# --- loaders ----------------------------------------------------------------------------------------------

def load_profile(name: str) -> Profile:
    """`verity_vllm.observe.fold.profiles.<name>`: its `PROFILE` / `profile` object or `build_profile()` / `make_profile()`."""
    import importlib
    mod = importlib.import_module(f"verity_vllm.observe.fold.profiles.{name}")
    for attr in ("PROFILE", "profile"):
        p = getattr(mod, attr, None)
        if isinstance(p, Profile):
            return p
    for attr in ("build_profile", "make_profile", "get_profile"):
        f = getattr(mod, attr, None)
        if callable(f):
            return f()
    raise ImportError(f"verity_vllm.observe.fold.profiles.{name} exposes no Profile (PROFILE / profile / build_profile())")


def fold_log(log_path: str | Path, profile: Profile, *, strict: bool | None = None, with_program: bool = True) -> FoldResult:
    return fold(list(read_log(log_path)), profile, strict=strict, with_program=with_program)


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


# --- snapshots --------------------------------------------------------------------------------------------

@dataclass
class Snapshot:
    event: int
    arg: str
    view: ViewDesc
    file: Path
    sha256: str | None = None
    step: int | None = None
    spanned: bool = True          # L1's layout: raw bytes of the storage span the view covers (else logical row-major)
    lo: int = 0                   # first storage element of the file (spanned layout)
    _words: np.ndarray | None = field(default=None, repr=False)
    _index: dict[int, int] | None = field(default=None, repr=False)   # storage element -> logical position (logical layout)

    def contiguous(self) -> bool:
        exp = 1
        for n, s in zip(reversed(self.view.shape), reversed(self.view.strides)):
            if n != 1 and s != exp:
                return False
            exp *= n
        return True


class SnapshotStore:
    """The `snapshots.json` index of one capture (`verity-capture/snapshots/v1`: `entries` with event, path, view,
    byte_offset/nbytes of the spanned storage bytes, file, step) and the element words of its files."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        idx = self.root / "snapshots.json"
        if not idx.exists():
            raise FileNotFoundError(idx)
        raw = _load_json(idx)
        self.meta = {k: v for k, v in raw.items() if k != "entries"} if isinstance(raw, dict) else {}
        entries: Any = raw.get("entries", raw.get("snapshots", raw)) if isinstance(raw, dict) else raw
        if isinstance(entries, dict):   # {event_id: [entry...]} or {event_id: {arg: entry}}
            flat = []
            for k, v in entries.items():
                items = v.values() if isinstance(v, dict) and not ("view" in v or "$v" in v) else (v if isinstance(v, list) else [v])
                for e in items:
                    e = dict(e)
                    e.setdefault("event", int(k))
                    flat.append(e)
            entries = flat
        default_layout = self.meta.get("layout", "spanned-storage-bytes")
        # `skipped` entries (a tensor of more bytes than the capture's `snapshot_max_bytes_op`, D87) describe a view that was
        # NOT dumped: no `file`.  They must not become Snapshots (the empty file path resolved to the capture root and
        # `_raw` raised IsADirectoryError the first time a boundary value lived in a skipped alloc — OLMoE B8-static's
        # 1300-token step 0); they are kept aside for the report.
        self.skipped: list[dict] = [e for e in entries if isinstance(e, dict) and e.get("skipped")]
        self.entries: list[Snapshot] = [self._entry(e, default_layout) for e in entries if not (isinstance(e, dict) and e.get("skipped"))]
        self.by_event: dict[int, list[Snapshot]] = defaultdict(list)
        self.by_alloc: dict[int, list[Snapshot]] = defaultdict(list)
        for s in self.entries:
            self.by_event[s.event].append(s)
            self.by_alloc[s.view.alloc].append(s)
        for lst in self.by_alloc.values():
            lst.sort(key=lambda s: s.event)
        self.steps = sorted({s.step for s in self.entries if s.step is not None})

    def _entry(self, e: dict, default_layout: str) -> Snapshot:
        ev: Any = e.get("event", e.get("event_id", e.get("id")))
        arg = e.get("arg", e.get("arg_path", e.get("path", "")))
        v = e.get("view", e)
        view = ViewDesc.from_json(v) if isinstance(v, dict) and "$v" in v else ViewDesc.from_json({"$v": v}) if isinstance(v, list) else ViewDesc(**v)
        f = Path(e.get("file", e.get("path_file", "")))
        if not f.is_absolute():
            f = self.root / f
        layout = e.get("layout", default_layout)
        spanned = layout == "spanned-storage-bytes"
        lo = _extent(view)[0]
        if spanned and "byte_offset" in e:
            lo = int(e["byte_offset"]) // view.itemsize
        return Snapshot(int(ev), str(arg), view, f, e.get("sha256"), e.get("step"), spanned, lo)

    def _raw(self, s: Snapshot) -> np.ndarray:
        """The file's words: storage order from element `s.lo` (spanned layout) or logical row-major order."""
        if s._words is None:
            raw = np.fromfile(s.file, dtype=np.uint8)
            dt = _NP[s.view.dtype]
            isz = s.view.itemsize
            lo, hi = _extent(s.view)
            if s.spanned:
                if raw.size < (hi - s.lo) * isz or s.lo > lo:
                    raise ValueError(f"snapshot {s.file}: {raw.size} bytes from element {s.lo} do not span {s.view!r}")
            elif raw.size != s.view.numel * isz:
                raise ValueError(f"snapshot {s.file}: {raw.size} bytes for {s.view!r} ({s.view.numel} elements)")
            s._words = raw.view(dt).reshape(-1)
        return s._words

    def words(self, s: Snapshot) -> np.ndarray:
        """Element words of the snapshot in row-major logical order of its view."""
        return self.extract(s, s.view)

    def covers(self, s: Snapshot, view: ViewDesc) -> bool:
        if view.alloc != s.view.alloc or view.numel == 0 or view.itemsize != s.view.itemsize:
            return False
        lo, hi = _extent(view)
        slo, shi = _extent(s.view)
        if s.spanned:   # every byte of the span was dumped
            return lo >= min(slo, s.lo) and hi <= shi
        if lo < slo or hi > shi:
            return False
        if s.contiguous():
            return True
        idx = self._index(s)
        return all(int(e) in idx for e in _storage_indices(view).reshape(-1))

    def _index(self, s: Snapshot) -> dict[int, int]:
        if s._index is None:
            s._index = {int(e): i for i, e in enumerate(_storage_indices(s.view).reshape(-1))}
        return s._index

    def extract(self, s: Snapshot, view: ViewDesc) -> np.ndarray:
        """Words of `view` (inside `s`, same element width), in row-major order of `view`."""
        w = self._raw(s)
        el = _storage_indices(view).reshape(-1)
        if s.spanned:
            return w[el - s.lo]
        if s.contiguous():
            return w[el - s.view.offset]
        idx = self._index(s)
        return w[np.fromiter((idx[int(e)] for e in el), dtype=np.int64, count=el.size)]

    def find(self, view: ViewDesc, *, at: int | None = None, before: int | None = None) -> Snapshot | None:
        """The snapshot covering `view`: of event `at`, or the latest with event id < `before`."""
        cands = self.by_event.get(at, []) if at is not None else self.by_alloc.get(view.alloc, [])
        best = None
        for s in cands:   # by_alloc lists are sorted by event id
            if before is not None and s.event >= before:
                if at is None:
                    break
                continue
            if self.covers(s, view):
                best = s
                if at is not None:
                    break
        return best


def _storage_indices(v: ViewDesc) -> np.ndarray:
    idx = np.array(v.offset, dtype=np.int64)
    for n, s in zip(v.shape, v.strides):
        idx = np.add.outer(idx, np.arange(n, dtype=np.int64) * s)
    return idx.reshape(v.shape) if v.shape else idx.reshape(())


def _extent(v: ViewDesc) -> tuple[int, int]:
    lo = hi = v.offset
    for n, s in zip(v.shape, v.strides):
        if n > 1:
            if s >= 0:
                hi += (n - 1) * s
            else:
                lo += (n - 1) * s
    return lo, hi + 1


# --- weights (parameter words by leaf interval of the weights Record) ----------------------------------------

# circuit weights record field -> (capture canonical id, checkpoint tensors concatenated along dim 0: none for the fixture's cos_sin)
_WEIGHT_FIELDS: dict[str, tuple[str, list[str]]] = {
    "embed": ("w=embed_tokens.weight", ["model.embed_tokens.weight"]),
    "norm": ("w=norm.weight", ["model.norm.weight"]),
    "cos_sin": ("w=layers.0.self_attn.rotary_emb.cos_sin_cache", []),
}
_LAYER_FIELDS = {
    "ln1": ("w=layers.{L}.input_layernorm.weight", ["model.layers.{L}.input_layernorm.weight"]),
    "qkv_w": ("w=layers.{L}.self_attn.qkv_proj.weight", [f"model.layers.{{L}}.self_attn.{p}_proj.weight" for p in "qkv"]),
    "qkv_b": ("w=layers.{L}.self_attn.qkv_proj.bias", [f"model.layers.{{L}}.self_attn.{p}_proj.bias" for p in "qkv"]),
    "o_w": ("w=layers.{L}.self_attn.o_proj.weight", ["model.layers.{L}.self_attn.o_proj.weight"]),
    "ln2": ("w=layers.{L}.post_attention_layernorm.weight", ["model.layers.{L}.post_attention_layernorm.weight"]),
    "gate_up_w": ("w=layers.{L}.mlp.gate_up_proj.weight", ["model.layers.{L}.mlp.gate_proj.weight", "model.layers.{L}.mlp.up_proj.weight"]),
    "down_w": ("w=layers.{L}.mlp.down_proj.weight", ["model.layers.{L}.mlp.down_proj.weight"]),
}


def _moe_layer_fields(weights_type: Type) -> dict[str, tuple[str, list[str]]]:
    """Checkpoint sources of the fused-MoE / qk-norm layer fields that
    `analytic.moe_layer_weights_type` adds, read off the weights Record itself (E = the expert axis of `w13`).  vLLM
    d9105ea80 stacks `experts.{e}.gate_proj` | `up_proj` into `w13[e]` ([2I][H], gate rows first) and `experts.{e}.down_proj`
    into `w2[e]`; the router is `mlp.gate.weight`, the qk-norms `self_attn.{q,k}_norm.weight`.  Dense Records add nothing."""
    try:
        layer = weights_type.fields[weights_type.index("layers")][1].elem
        names = {n for n, _ in layer.fields}
    except (AttributeError, KeyError, ValueError, IndexError):
        return {}
    out: dict[str, tuple[str, list[str]]] = {}
    for f, hf in (("q_norm", "self_attn.q_norm.weight"), ("k_norm", "self_attn.k_norm.weight"), ("gate_w", "mlp.gate.weight")):
        if f in names:
            out[f] = (f"w=layers.{{L}}.{hf}", [f"model.layers.{{L}}.{hf}"])
    if "w13" in names and "w2" in names:
        E = layer.fields[layer.index("w13")][1].n
        out["w13"] = ("w=layers.{L}.mlp.experts.w13_weight",
                      [f"model.layers.{{L}}.mlp.experts.{e}.{p}_proj.weight" for e in range(E) for p in ("gate", "up")])
        out["w2"] = ("w=layers.{L}.mlp.experts.w2_weight", [f"model.layers.{{L}}.mlp.experts.{e}.down_proj.weight" for e in range(E)])
    return out


def _f16_words_to_bf16(w: np.ndarray) -> np.ndarray:
    """The bf16 word vLLM's loader produces from an fp16 checkpoint word: `param.data.copy_(loaded)`
    into a bf16 parameter on the GPU is `__float2bfloat16((float)half)` — fp16 -> f32 exact, f32 -> bf16 round-to-nearest-even
    (`analytic.f32_to_bf16_words`; NaN payloads are not preserved by the CUDA convert, but a checkpoint has no NaN words)."""
    from verity_vllm.program.model import f32_to_bf16_words
    return f32_to_bf16_words(np.ascontiguousarray(w, dtype=np.uint16).view(np.float16).astype(np.float32))


def _layernorm_family_fields(C: dict | None) -> tuple[dict[str, tuple[str, list[str]]], dict[str, tuple[str, list[str]]]] | None:
    """Checkpoint sources of the `analytic.layernorm_layer_weights_type` fields (GPT-NeoX / Pythia,
    `C["NORM"] == "layernorm"`): `(top_fields, layer_fields)` in the `_WEIGHT_FIELDS` / `_LAYER_FIELDS` shape, with vLLM's
    GPT-NeoX parameter names (= the HF checkpoint names: `gpt_neox.layers.{L}.attention.query_key_value.weight`, ...).  The fused
    QKV tensor is stored `[NH, 3, D]`-major in the checkpoint and `[3, NH, D]`-major in vLLM's parameter (`GPTNeoXForCausalLM.
    load_weights` transposes); `WeightSource` applies that permutation (`_qkv_neox_permute`).  None for other families."""
    if not C or C.get("NORM") != "layernorm":
        return None
    top = {"embed": ("w=embed_in.weight", ["gpt_neox.embed_in.weight"]),
           "norm": ("w=final_layer_norm.weight", ["gpt_neox.final_layer_norm.weight"]),
           "norm_b": ("w=final_layer_norm.bias", ["gpt_neox.final_layer_norm.bias"]),
           "lm_head": ("w=embed_out.weight", ["embed_out.weight"]),
           "cos_sin": ("w=layers.0.attention.rotary_emb.cos_sin_cache", [])}
    layer = {f: (f"w=layers.{{L}}.{hf}", [f"gpt_neox.layers.{{L}}.{hf}"]) for f, hf in (
        ("ln1", "input_layernorm.weight"), ("ln1_b", "input_layernorm.bias"),
        ("qkv_w", "attention.query_key_value.weight"), ("qkv_b", "attention.query_key_value.bias"),
        ("o_w", "attention.dense.weight"), ("o_b", "attention.dense.bias"),
        ("ln2", "post_attention_layernorm.weight"), ("ln2_b", "post_attention_layernorm.bias"),
        ("up_w", "mlp.dense_h_to_4h.weight"), ("up_b", "mlp.dense_h_to_4h.bias"),
        ("down_w", "mlp.dense_4h_to_h.weight"), ("down_b", "mlp.dense_4h_to_h.bias"))}
    return top, layer


def _qkv_neox_permute(w: np.ndarray, NH: int, D: int) -> np.ndarray:
    """vLLM `GPTNeoXForCausalLM.load_weights`: the checkpoint's `query_key_value` output dim is
    `(NH, 3, D)`; the parameter's is `(3, NH, D)` (`loaded_weight.view(NH, 3, D, *rest).transpose(0, 1).reshape(shape)`)."""
    rest = w.shape[1:]
    return np.ascontiguousarray(w.reshape(NH, 3, D, *rest).transpose(1, 0, 2, *range(3, 3 + len(rest))).reshape(w.shape))
#: checkpoint sources of the extra per-layer RMSNorm fields `analytic.weights_type` adds
#: from `C["extra_layer_norms"]` (Gemma-2: `pre_feedforward_layernorm` before the MLP, `post_feedforward_layernorm` after it;
#: `_LAYER_FIELDS` already has `ln1` / `ln2`).  Read off the Record like `_moe_layer_fields`; Llama/Qwen/Phi Records add nothing.
_EXTRA_NORM_SOURCES = {"ln_pre_mlp": "pre_feedforward_layernorm.weight", "ln_post_mlp": "post_feedforward_layernorm.weight"}


def _extra_norm_layer_fields(weights_type: Type) -> dict[str, tuple[str, list[str]]]:
    try:
        layer = weights_type.fields[weights_type.index("layers")][1].elem
        names = {n for n, _ in layer.fields}
    except (AttributeError, KeyError, ValueError, IndexError):
        return {}
    return {f: (f"w=layers.{{L}}.{hf}", [f"model.layers.{{L}}.{hf}"]) for f, hf in _EXTRA_NORM_SOURCES.items() if f in names}


class WeightSource:
    """Words of the `weights` Record leaves (bf16 bit patterns) from the checkpoint safetensors + the cos/sin fixture."""

    def __init__(self, weights_type: Type, safetensors: Iterable[Path] | None = None, cos_sin: Path | None = None,
                 repo: str | None = None, config: dict | None = None) -> None:
        paths = list(safetensors) if safetensors else [Path(p) for p in sorted(glob.glob(hf_safetensors_glob(repo)))]
        self.tensors, self.tensor_dtypes = load_safetensors_words(paths) if paths else ({}, {})
        self.paths = paths
        self.cos_sin_path = Path(cos_sin) if cos_sin else DEFAULT_COS_SIN
        self.weights_type = weights_type
        self.intervals = _field_intervals(weights_type, "")
        self.config = dict(config or {})
        self._layer_fields = {**_LAYER_FIELDS, **_moe_layer_fields(weights_type),   # + MoE / qk-norm fields
                              **_extra_norm_layer_fields(weights_type)}             # + Gemma-2's pre/post-MLP norms
        self._weight_fields = {**_WEIGHT_FIELDS, "lm_head": ("w=lm_head.weight", ["lm_head.weight"])}   # untied checkpoints
        fam = _layernorm_family_fields(self.config)                                   # GPT-NeoX names
        if fam is not None:
            self._weight_fields, self._layer_fields = dict(fam[0]), dict(fam[1])
        self._mla = mla_moe_sources(weights_type)                                     # dense_layers + layers
        self._cache: dict[str, np.ndarray] = {}

    def available(self) -> bool:
        return bool(self.tensors) and self.cos_sin_path.exists()

    def _tensor_words(self, name: str, field: str) -> np.ndarray:
        """Runtime (bf16) words of checkpoint tensor `name` for Record field `field`: F16 tensors converted as vLLM's loader
        does (); the GPT-NeoX fused QKV permuted into the parameter's `[3, NH, D]` order."""
        t = self.tensors[name]
        if self.tensor_dtypes.get(name) == "F16":
            t = _f16_words_to_bf16(t).reshape(t.shape)
        if field in ("qkv_w", "qkv_b") and self.config.get("NORM") == "layernorm":
            t = _qkv_neox_permute(np.asarray(t), int(self.config["NH"]), int(self.config["D"]))
        return np.asarray(t).reshape(-1)

    def _field_words(self, name: str) -> np.ndarray:
        if name in self._cache:
            return self._cache[name]
        if name == "cos_sin":
            w = np.load(self.cos_sin_path)
            if w.dtype != np.uint16:
                raise ValueError(f"{self.cos_sin_path}: expected uint16 bf16 bit patterns, got {w.dtype}")
            w = w.reshape(-1)
        elif name == "normalizer":
            # Gemma-2's `model.normalizer` (analytic-table root, `analytic.weights_type`): one bf16 word,
            # bf16(sqrt(H)); H is the Record's own hidden width (the `norm` field's length), so no configuration is needed here.
            from verity_vllm.program import model as analytic
            H = self.weights_type.fields[self.weights_type.index("norm")][1].n
            w = np.array([analytic.normalizer_word(int(H))], dtype=np.uint16)
        elif name in self._weight_fields:
            w = np.concatenate([self._tensor_words(t, name) for t in self._weight_fields[name][1]])
        elif self._mla is not None and MLA_LAYER_FIELD.match(name):
            w = mla_field_words(self._mla, name, self._tensor_words)
        elif name.startswith("layers["):
            L = int(name[len("layers["):name.index("]")])
            f = name.split(".", 1)[1]
            cid, srcs = self._layer_fields[f]
            names = [t.format(L=L) for t in srcs]
            if any(t not in self.tensors for t in names):
                # [PROPOSED GEN dense2] fused checkpoints (Phi-3: `qkv_proj.weight`, `gate_up_proj.weight`) store the projection
                # already concatenated under vLLM's fused parameter name, i.e. the field's canonical id (`w=layers.{L}.<fused>`)
                fused = "model." + cid[len("w="):].format(L=L)
                if fused in self.tensors:
                    names = [fused]
            w = np.concatenate([self._tensor_words(t, f) for t in names])
        else:
            raise KeyError(name)
        self._cache[name] = w
        return w

    def words(self, lo: int, hi: int) -> np.ndarray:
        out = []
        for flo, fhi, name in self.intervals:
            if fhi <= lo or flo >= hi:
                continue
            a, b = max(lo, flo), min(hi, fhi)
            w = self._field_words(name)
            if w.size != fhi - flo:
                raise ValueError(f"weights field {name}: {w.size} checkpoint words for {fhi - flo} leaves")
            out.append(w[a - flo:b - flo])
        w = np.concatenate(out) if out else np.zeros(0, dtype=np.uint16)
        if w.size != hi - lo:
            raise ValueError(f"weights leaves [{lo}, {hi}) not covered by the checkpoint fields")
        return w
