"""verity_vllm/check/fold_compare.py — M4: the four comparisons of a folded observation log against the evidence and Serve_v4.

    uv run verity-vllm fold-compare --log data/logs/m1.jsonl.gz --snapshots data/logs/m1 \
        --profile vllm_d9105ea80_sm89_eager --out out/capture/reports/compare_m1.json

Each comparison is a function returning a JSON-able dict with a `status` in {"pass", "fail", "skipped", "error"}:

1. `compare_tokens`     — the sampled token ids the fold's TokenSelect instances stand for (`SampleEnd`) vs the
                          observer's `tokens.json` vs the accepted record's `output_token_ids`.
2. `compare_boundary`   — provenance: where the fold says an instance's operand holds the references of some
                          earlier instance's output (or a parameter), the bytes L1 snapshotted for the operand's
                          view (state right before the consumer) are bit-identical to the bytes of the producer's
                          output view (or the checkpoint / prompt).  Covers the KV slot rows read by attention
                          (RoPE k outputs, the qkv v slice), the logits-row gather and the decode input token.
3. `compare_subcircuits`— a handful of derived instances are evaluated with the reference evaluator on their snapshot
                          operand values (fed as encodings through `evaluate_call`) and compared bit-exactly with the
                          snapshot of their outputs.
4. `compare_structure`  — `match_program(flat_body, Serve_v4, allow_extra=True)`: matched / superset, the extra nodes
                          grouped into named CLASSES (never forced to isomorphism), and node counts per specialization
                          id in the derived body vs the family's expanded leaves.

Everything here is torch-free.  Resource discipline (DESIGN §11): the M1 run measured 0.93 GB max RSS / 8 s on the
laptop (fold + Program() + digest alone: 0.21 GB / 2.4 s); comparison 3 deliberately excludes Embedding and lm_head
instances, whose `FunctionCall.args` would materialise 28M-element tuples (~1 GB each).  Snapshots follow DESIGN §3: `<snapshots>/snapshots.json` is an index of entries
`{event, arg, view: {"$v": [...]}, file, sha256, step?}` and each file holds the raw element bytes of the view in
row-major logical order (a storage-order dump of the view's byte extent is also accepted).  Weights come from the
SmolLM2-135M safetensors in the HF cache (`verity_vllm.correspondence.capture_identities._LAYER_FIELDS`) and the cos/sin cache from the
tracked fixture the interior producer uses.
"""
from __future__ import annotations

import glob
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from verity.ml.circuits.defs import bind
from verity_vllm.check.checkpoint_words import MLA_LAYER_FIELD, load_safetensors_words, mla_field_words, mla_moe_sources
from verity.ml.circuits.evaluation.reference import evaluate_call
from verity_vllm.config import option
from verity_vllm.program.frontend.rules.family import _field_intervals, expand_leaves, match_program, node_leaf_id
from verity.ml.circuits.layout import FunctionCall
from verity.ml.circuits.refs import Refs, runs
from verity.ml.definitions.registry import b1
from verity.ml.circuits.types import Array, Tuple as TupleT, Type

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.fold import FoldResult, fold, summarize
from verity_vllm.observe.log import read_log
from verity_vllm.observe.fold.resolver import Profile

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
        entries = raw.get("entries", raw.get("snapshots", raw)) if isinstance(raw, dict) else raw
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
        # `skipped` entries (a tensor over the capture's snapshot cap, D87 / `snapshot_max_bytes_op`) describe a view that was
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
        ev = e.get("event", e.get("event_id", e.get("id")))
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
           "cos_sin": ("w=layers.0.attention.rotary_emb.cos_sin_cache", None)}
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
        from verity_vllm.correspondence.capture_identities import _LAYER_FIELDS, _WEIGHT_FIELDS
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


# --- values by provenance ----------------------------------------------------------------------------------

class Values:
    """Words of a reference sequence resolved through the fold: parameter leaves from the prompt / checkpoint, node
    output leaves from the snapshot of the producing event's output view."""

    def __init__(self, result: FoldResult, store: SnapshotStore, weights: WeightSource | None) -> None:
        self.r, self.store, self.weights = result, store, weights
        self.by_node = {m["node"]: m for m in result.instances}
        self.W = len(result.params) - 1

    def of_refs(self, refs: Refs) -> np.ndarray:
        parts = [self.of_run(space, idx, lo, hi) for space, idx, lo, hi in runs(refs)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint32)

    def of_run(self, space: str, idx: int, lo: int, hi: int) -> np.ndarray:
        if space == "p":
            if idx == self.W:
                if self.weights is None or not self.weights.available():
                    raise LookupError("no weight source (checkpoint safetensors / cos_sin fixture)")
                return self.weights.words(lo, hi)
            toks = self.r.requests[idx]["prompt_token_ids"]
            return np.array([int(t) & 0xFFFFFFFF for t in toks[lo:hi]], dtype=np.uint32)
        if space != "n":
            raise LookupError(f"reference space {space!r}")
        m = self.by_node.get(idx)
        if m is None:
            raise LookupError(f"node {idx} has no instance record")
        out = []
        for k, (clo, chi) in enumerate(_component_ranges(m)):
            if chi <= lo or clo >= hi:
                continue
            view = m["output_views"][k]
            if view is None:
                raise LookupError(f"node {idx} component {k}: output view unknown")
            s = self.store.find(view, at=m["event"])
            if s is None:
                raise LookupError(f"node {idx} ({m['name']}): no snapshot of event {m['event']} covers {view!r}")
            words = self.store.extract(s, view)
            out.append(words[max(lo, clo) - clo:min(hi, chi) - clo])
        w = np.concatenate(out)
        if w.size != hi - lo:
            raise LookupError(f"node {idx}: leaves [{lo}, {hi}) outside its {sum(chi - clo for clo, chi in _component_ranges(m))} outputs")
        return w


def _component_ranges(meta: dict) -> list[tuple[int, int]]:
    """Leaf ranges of the return components of the instance's specialization."""
    spec = meta.get("_spec")
    if spec is None:
        raise LookupError("instance metadata carries no specialization object")
    _, ret = spec.signature()
    comps = ret.fields if isinstance(ret, TupleT) else (ret,)
    out, off = [], 0
    for t in comps:
        out.append((off, off + t.leaves))
        off += t.leaves
    return out


def _attach_specs(result: FoldResult) -> None:
    """The instance records carry `spec` labels; comparisons need the specialization object of each node."""
    for m in result.instances:
        m["_spec"] = result.body.nodes[m["node"]].fn


def _encodings(words: np.ndarray) -> list[int]:
    return [int(x) for x in words.tolist()]


# --- 1. tokens ---------------------------------------------------------------------------------------------

def compare_tokens(result: FoldResult, tokens_json: Path | None = None, record: Path | None = None) -> dict:
    per_req: dict[str, list[int]] = {req: [int(tok) for _, tok in seq] for req, seq in result.sampled.items()}
    per_req_steps: dict[str, list[int]] = {req: [int(step) for step, _ in seq] for req, seq in result.sampled.items()}
    token_select_nodes = {}
    for req, seq in result.outputs.items():
        for j, refs in enumerate(seq):
            node = None
            for space, idx, lo, hi in runs(refs):
                node = (space, idx)
            token_select_nodes[f"{req}@{j}"] = None if node is None or node[0] != "n" else result.body.nodes[node[1]].fn.id
    out: dict[str, Any] = {"status": "pass", "fold": per_req, "fold_steps": per_req_steps,
                           "token_select_specs": Counter(v for v in token_select_nodes.values() if v), "mismatches": []}
    if any(v is None or not v.startswith("TokenSelect") for v in token_select_nodes.values()):
        out["mismatches"].append("some sampler outputs are not TokenSelect instances: "
                                 + ", ".join(f"{k}: {v}" for k, v in token_select_nodes.items() if not (v or "").startswith("TokenSelect")))
    if tokens_json is not None and Path(tokens_json).exists():
        obs = _tokens_from_json(_load_json(Path(tokens_json)))
        out["observer"] = obs
        for req, toks in per_req.items():
            ref = obs.get(req) if req in obs else (next(iter(obs.values())) if len(obs) == 1 else None)
            if ref is None:
                out["mismatches"].append(f"observer tokens.json has no entry for request {req!r}")
            elif [int(t) for t in ref] != toks:
                out["mismatches"].append(f"observer tokens for {req!r} differ: {ref} vs fold {toks}")
    else:
        out["observer"] = None
    # the accepted tokens the capture wrote next to its own (tokens.json `accepted_record_token_ids`, from the workload file)
    if tokens_json is not None and Path(tokens_json).exists():
        d = _load_json(Path(tokens_json))
        acc = {r["request_id"]: [int(t) for t in r["accepted_record_token_ids"]] for r in (d.get("requests") or [])
               if isinstance(r, dict) and r.get("accepted_record_token_ids")}
        out["accepted"] = acc or None
        for req, toks in per_req.items():
            a = acc.get(req)
            if a is not None and a != toks:
                out["mismatches"].append(f"accepted tokens for {req!r} differ: {a} vs fold {toks}")
    rec = _record_tokens(record) if record is not None else None
    out["record"] = rec
    if rec:
        for req, toks in per_req.items():
            r = rec.get(req) or (next(iter(rec.values())) if len(rec) == 1 else None)
            if r is None:
                out["mismatches"].append(f"record has no request {req!r}")
            else:
                if r["output_token_ids"] != toks:
                    out["mismatches"].append(f"record output_token_ids for {req!r} differ: {r['output_token_ids']} vs fold {toks}")
                pr = next((q["prompt_token_ids"] for q in result.requests if q["request_id"] == req), None)
                if pr is not None and [int(t) for t in pr] != r["prompt_token_ids"]:
                    out["mismatches"].append(f"record prompt_token_ids for {req!r} differ from the log's")
    if out["mismatches"]:
        out["status"] = "fail"
    return out


def _tokens_from_json(d: Any) -> dict[str, list[int]]:
    if isinstance(d, list) and all(isinstance(x, int) for x in d):
        return {"": d}
    if isinstance(d, dict):
        if all(isinstance(v, list) and all(isinstance(x, int) for x in v) for v in d.values()) and d:
            return {str(k): v for k, v in d.items()}
        for key in ("sampled", "sampled_token_ids", "output_token_ids", "tokens", "generated", "requests"):
            if key in d:
                v = d[key]
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    return {str(r.get("request_id", r.get("id", i))):
                            [int(t) for t in r.get("generated_token_ids", r.get("output_token_ids", r.get("tokens", [])))]
                            for i, r in enumerate(v)}
                return _tokens_from_json(v)
    return {}


def _record_tokens(record: Path) -> dict[str, dict[str, list[int]]]:
    out = {}
    for p in sorted(record.glob("adapter/*/ru/*/boundary.json")):
        d = _load_json(p)
        out[d.get("request_id", p.parent.name)] = {"prompt_token_ids": [int(t) for t in d["prompt_token_ids"]],
                                                    "output_token_ids": [int(t) for t in d["output_token_ids"]], "file": str(p)}
    return out


# --- 2. boundary values / provenance ------------------------------------------------------------------------

def _in_place(meta: dict) -> bool:
    outs = [v for v in meta.get("output_views", []) if v is not None]
    for op in meta.get("operand_views", []):
        if op is None:
            continue
        for ov in outs:
            if op.alloc == ov.alloc and _extent(op)[0] < _extent(ov)[1] and _extent(ov)[0] < _extent(op)[1]:
                return True
    return False


def lm_head_nodes(result: FoldResult) -> set[int]:
    """The Gemm instances whose output a TokenSelect instance reads (the lm_head over the gathered rows)."""
    out: set[int] = set()
    for n in result.body.nodes:
        if n.fn.id.startswith("TokenSelect"):
            for space, idx, _, _ in runs(n.args[0].refs):
                if space == "n" and result.body.nodes[idx].fn.id.startswith("Gemm"):
                    out.add(idx)
    return out


def compare_boundary(result: FoldResult, store: SnapshotStore, weights: WeightSource | None = None,
                     max_per_class: int = 8) -> dict:
    """For consumer instances whose step has snapshots: operand words by provenance (through the fold's references)
    vs the words L1 snapshotted for the operand view right before the consumer ran."""
    _attach_specs(result)
    vals = Values(result, store, weights)
    steps_with = set(store.steps) or None
    lm = lm_head_nodes(result)
    # class name -> (predicate on the instance record, operand index whose provenance is checked)
    classes: dict[str, tuple] = {
        "attention K rows (KV slots <- RoPE k outputs)": (lambda m: m["spec"].startswith("Attention"), 1),
        "attention V rows (KV slots <- qkv v slice)": (lambda m: m["spec"].startswith("Attention"), 2),
        "lm_head Gemm x <- gathered final-norm row": (lambda m: m["node"] in lm, 0),
        "decode Embedding token <- previous TokenSelect": (lambda m: m["spec"].startswith("Embedding") and m["step"] > 0, 0),
        "prefill Embedding token <- prompt": (lambda m: m["spec"].startswith("Embedding") and m["step"] == 0, 0),
        "Gemm x <- producer output": (lambda m: m["spec"].startswith("Gemm") and m["node"] not in lm, 0),
        "RMSNorm residual <- producer output": (lambda m: m["spec"].startswith("RMSNormFused"), 1),
        "SiluMul x <- gate_up Gemm output": (lambda m: m["spec"].startswith("SiluMul"), 0),
    }
    report: dict[str, Any] = {"status": "pass", "checks": [], "classes": {}}
    counts: Counter = Counter()
    for m in result.instances:
        if steps_with is not None and m["step"] not in steps_with:
            continue
        for cname, (pred, k) in classes.items():
            if not pred(m) or counts[cname] >= max_per_class:
                continue
            node = result.body.nodes[m["node"]]
            if k >= len(node.args):
                continue
            view = m["operand_views"][k] if k < len(m["operand_views"]) else None
            check = {"class": cname, "node": m["node"], "name": m["name"], "operand": k}
            try:
                if view is None:
                    raise LookupError("operand view unknown")
                if view.numel != node.args[k].refs.count:
                    raise LookupError(f"operand view has {view.numel} elements for {node.args[k].refs.count} references (gathered rows are not one run)")
                expect = vals.of_refs(node.args[k].refs)
                before = m["event"] if _in_place(m) else m["event"] + 1
                s = store.find(view, before=before)
                if s is None:
                    raise LookupError(f"no snapshot before event {before} covers {view!r}")
                got = store.extract(s, view)
                eq = got.size == expect.size and bool(np.array_equal(got.astype(np.uint64), expect.astype(np.uint64)))
                check.update(status="pass" if eq else "fail", n=int(got.size), snapshot_event=s.event, snapshot_arg=s.arg,
                             first_diff=None if eq else int(np.argmax(got.astype(np.uint64) != expect.astype(np.uint64))))
            except LookupError as e:
                check.update(status="skipped", reason=str(e))
            counts[cname] += 1
            report["checks"].append(check)
    for cname in classes:
        cs = [c for c in report["checks"] if c["class"] == cname]
        report["classes"][cname] = Counter(c["status"] for c in cs)
    st = Counter(c["status"] for c in report["checks"])
    report["summary"] = dict(st)
    report["status"] = "fail" if st.get("fail") else ("pass" if st.get("pass") else "skipped")
    return report


# --- 3. expanded small subcircuits --------------------------------------------------------------------------

def select_instances(result: FoldResult, attention_T: Iterable[int] = (1, 2, 3)) -> list[dict]:
    """One Gemm row of layer 0 qkv, one RMSNormFusedCuda row, one RoPE row, one SiluMul row, the TokenSelect of the
    first step and the attention rows with T in `attention_T` of the first step (no lm_head Gemm / Embedding: their
    weight operands are tens of millions of leaves)."""
    picks: list[dict] = []
    first_step = min((m["step"] for m in result.instances), default=0)
    lm = lm_head_nodes(result)
    want = {"Gemm": lambda m: "layers.0/" in m["name"] and m["node"] not in lm and "qkv" in (m["name"] + (m["hint"] or "")),
            "RMSNormFusedCuda": lambda m: True, "RoPE": lambda m: True, "SiluMul": lambda m: True,
            "TokenSelect": lambda m: m["step"] == first_step}
    seen = set()
    for m in result.instances:
        base = m["spec"].split("{")[0].split("_v")[0]
        if base in want and base not in seen and want[base](m):
            picks.append(m)
            seen.add(base)
    Ts = set(attention_T)
    for m in result.instances:
        if m["step"] == first_step and m["spec"].startswith("Attention"):
            T = int(m["spec"].split("T=")[1].split(",")[0].split("}")[0]) if "T=" in m["spec"] else None
            if T in Ts:
                picks.append(m)
                Ts.discard(T)
    return picks


def compare_subcircuits(result: FoldResult, store: SnapshotStore, weights: WeightSource | None = None,
                        picks: list[dict] | None = None) -> dict:
    _attach_specs(result)
    if result.program is None:
        return {"status": "skipped", "reason": "fold ran without Program()"}
    vals = Values(result, store, weights)
    circuit = result.program.circuit
    root = result.program.root
    n_in = len(root.body.nodes) - len(result.body.nodes)
    report: dict[str, Any] = {"status": "pass", "instances": []}
    for m in picks if picks is not None else select_instances(result):
        entry = {"node": m["node"], "name": m["name"], "spec": m["spec"], "step": m["step"]}
        node = result.body.nodes[m["node"]]
        try:
            t0 = time.time()
            args: list[int] = []
            for a in node.args:
                args += _encodings(vals.of_refs(a.refs))
            expect: list[int] = []
            for k, view in enumerate(m["output_views"]):
                if view is None:
                    raise LookupError(f"output {k} view unknown")
                s = store.find(view, at=m["event"])
                if s is None:
                    raise LookupError(f"no snapshot of event {m['event']} covers output {k} {view!r}")
                expect += _encodings(store.extract(s, view))
            call = FunctionCall(circuit, circuit.root_scope, n_in + m["node"])
            got = list(evaluate_call(circuit, call, args))
            if len(got) != len(expect):
                raise LookupError(f"{len(got)} return encodings vs {len(expect)} snapshot words")
            diff = [i for i in range(len(got)) if got[i] != expect[i]]
            entry.update(status="pass" if not diff else "fail", n_args=len(args), n_returns=len(got), n_diff=len(diff),
                         first_diff=diff[:4], gates=call.hi - call.lo, seconds=round(time.time() - t0, 3))
        except LookupError as e:
            entry.update(status="skipped", reason=str(e))
        except Exception as e:  # evaluator / MUFU table problems are reported, not fatal
            entry.update(status="error", reason=f"{type(e).__name__}: {e}")
        report["instances"].append(entry)
    st = Counter(e["status"] for e in report["instances"])
    report["summary"] = dict(st)
    report["status"] = "fail" if st.get("fail") or st.get("error") else ("pass" if st.get("pass") else "skipped")
    return report


# --- 4. program structure ------------------------------------------------------------------------------------

def _consumed_nodes(result: FoldResult) -> set[int]:
    used: set[int] = set()
    for n in result.body.nodes:
        for a in n.args:
            for space, idx, _, _ in runs(a.refs):
                if space == "n":
                    used.add(idx)
    for space, idx, _, _ in runs(result.body.ret):
        if space == "n":
            used.add(idx)
    return used


def classify_extra(result: FoldResult, extra: list[dict]) -> dict[str, dict]:
    """Group `match_program`'s extra nodes into named classes (spec base, phase, hint tail, live/dead)."""
    by_node = {m["node"]: m for m in result.instances}
    used = _consumed_nodes(result)
    lm = lm_head_nodes(result)
    classes: dict[str, dict] = {}
    for e in extra:
        k = e["index"]
        m = by_node.get(k, {})
        base = e["id"].split("{")[0]
        step = m.get("step")
        phase = "prefill" if step == 0 else "decode" if step is not None else "?"
        hint = (m.get("hint") or "")
        tail = hint.rsplit(".", 1)[-1] if hint else ""
        live = "live" if k in used else "dead"
        if base.startswith("RMSNormFusedCuda") and phase == "prefill" and "layers." not in (m.get("name") or "") and live == "dead":
            name = "final norm applied to every prefill row in vLLM; Serve_v4 normalises only the last position (dead extra instances)"
        elif base.startswith("Gemm") and k in lm:
            name = f"lm_head Gemm over extra gathered rows ({phase}, {live})"
        elif base.startswith("Embedding"):
            name = f"extra Embedding rows ({phase}, {live})"
        else:
            name = f"unclassified: {base} ({phase}, hint tail {tail!r}, {live})"
        c = classes.setdefault(name, {"count": 0, "specs": Counter(), "steps": Counter(), "nodes": []})
        c["count"] += 1
        c["specs"][e["id"]] += 1
        c["steps"][str(step)] += 1
        if len(c["nodes"]) < 8:
            c["nodes"].append({"index": k, "name": e.get("name")})
    return classes


def compare_structure(result: FoldResult, *, C=b1.B0_CONFIG, LP: int = 32, STEPS: int = 15, BN: int | None = None,
                      family=None) -> dict:
    """`match_program` against the hand-authored family `Serve_v4{C, LP, STEPS, BN}`; `BN` defaults to the FA2 rule for
    `C["D"]` (128 at head dim 64, 64 at head dim 128).  `C` carries no `BN` key in `b1.*_CONFIG`; a profile config does."""
    if BN is None:
        BN = C.get("BN") or b1.fa2_kblock_n(C["D"])
    C = {k: v for k, v in C.items() if k != "BN"}
    fam = family if family is not None else bind(b1.ServeV4, C=C, LP=LP, STEPS=STEPS, BN=BN)
    W = len(result.params) - 1
    t0 = time.time()
    try:
        rep = match_program(result.body, fam, flat_weights_param=W, allow_extra=True, with_map=False)
    except Exception as e:
        rep = {"matched": False, "reason": f"{type(e).__name__}: {e}"}
    out: dict[str, Any] = {"status": "pass" if rep.get("matched") or rep.get("superset") else "fail",
                           "family": fam.id, "matched": bool(rep.get("matched")), "superset": bool(rep.get("superset")),
                           "reason": rep.get("reason"), "seconds": round(time.time() - t0, 3)}
    extra = rep.get("extra_nodes") or []
    out["n_extra"] = len(extra)
    out["extra_by_spec"] = dict(Counter(e["id"] for e in extra))
    out["classes"] = {k: {"count": v["count"], "specs": dict(v["specs"]), "steps": dict(v["steps"]), "examples": v["nodes"]}
                      for k, v in classify_extra(result, extra).items()}
    # gate I8 (generalization plan): a superset passes only when every extra instance falls in a named, explained class;
    # an `unclassified:` class is unexplained extra computation and fails (the M4 rule let any superset pass).
    unclassified = {k: v["count"] for k, v in out["classes"].items() if k.startswith("unclassified:")}
    out["unclassified"] = unclassified
    if out["status"] == "pass" and unclassified:
        out["status"] = "fail"
        out["reason"] = f"{sum(unclassified.values())} extra instances in unclassified classes: " + "; ".join(sorted(unclassified))
    derived = Counter(node_leaf_id(n) for n in result.body.nodes)
    leaf_bases = {n.fn.base_id for n in result.body.nodes if hasattr(n.fn, "base_id")}
    try:
        leaves, _, _, _ = expand_leaves(fam, leaf_bases)
        family_hist = Counter(l.id for l in leaves)
    except Exception as e:
        family_hist = Counter()
        out["family_expand_error"] = f"{type(e).__name__}: {e}"
    ids = sorted(set(derived) | set(family_hist))
    out["histogram"] = {i: {"derived": derived.get(i, 0), "family": family_hist.get(i, 0), "delta": derived.get(i, 0) - family_hist.get(i, 0)}
                        for i in ids}
    out["n_derived_nodes"] = len(result.body.nodes)
    out["n_family_leaves"] = sum(family_hist.values())
    if rep.get("flat_node_ids") or rep.get("family_leaf_ids"):
        out["match_program_ids"] = {"flat": rep.get("flat_node_ids"), "family": rep.get("family_leaf_ids")}
    return out


# --- CLI ------------------------------------------------------------------------------------------------------

def _max_rss_gb() -> float:
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 30 if sys.platform == "darwin" else r / 2 ** 20


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Counter):
        return {str(k): v for k, v in x.items()}
    if isinstance(x, ViewDesc):
        return repr(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, Path):
        return str(x)
    return x


@dataclass(frozen=True, kw_only=True)
class Options:
    log: str = option("--log")
    snapshots: str | None = option("--snapshots", default=None, help="directory with snapshots.json, values/, tokens.json")
    profile: str = option("--profile")
    record: str | None = option("--record", default=None, help="accepted record dir with adapter/*/ru/*/boundary.json (default: none, no record comparison)")
    out: str = option("--out")
    strict: bool = option("--strict", default=False, action="store_true")
    no_program: bool = option("--no-program", default=False, action="store_true", help="skip Program() (comparison 3 is then skipped)")
    lp: int = option("--lp", type=int, default=32)
    steps: int = option("--steps", type=int, default=15)
    bn: int | None = option("--bn", type=int, default=None, help="Serve_v4 BN (default: the profile config's / the FA2 rule at D)")
    safetensors: str | None = option("--safetensors", default=None, help="glob of the checkpoint's model*.safetensors (default: the HF cache of the profile's model)")
    cos_sin: str | None = option("--cos-sin", default=None, help="uint16 .npy of the rotary cos_sin table (default: <snapshots>/cos_sin_cache.npy, else the B0 fixture)")


def main(a: Options) -> int:
    t0 = time.time()
    profile = load_profile(a.profile)
    case = (profile.notes or {}).get("case", "B0")
    repo = (profile.config or {}).get("model")
    record = Path(a.record) if a.record else None
    result = fold_log(a.log, profile, strict=True if a.strict else None, with_program=not a.no_program)
    report: dict[str, Any] = {"log": a.log, "profile": a.profile, "case": case, "model": repo, "fold": summarize(result),
                              "digest": result.digest, "fold_seconds": round(time.time() - t0, 3)}
    snaps = Path(a.snapshots) if a.snapshots else None
    report["tokens"] = compare_tokens(result, snaps / "tokens.json" if snaps else None, record)
    store = None
    if snaps is not None and (snaps / "snapshots.json").exists():
        store = SnapshotStore(snaps)
        cos_sin = Path(a.cos_sin) if a.cos_sin else ((snaps / "cos_sin_cache.npy") if (snaps / "cos_sin_cache.npy").exists() else None)
        weights = WeightSource(result.params[-1][1], [Path(p) for p in sorted(glob.glob(a.safetensors))] if a.safetensors else None,
                               cos_sin, repo=repo, config=dict(profile.config or {}))   # family / dtype facts
        report["weights_available"] = weights.available()
        report["weights_source"] = {"safetensors": [str(p) for p in weights.paths], "cos_sin": str(weights.cos_sin_path)}
        report["boundary"] = compare_boundary(result, store, weights)
        report["subcircuits"] = compare_subcircuits(result, store, weights) if not a.no_program else {"status": "skipped", "reason": "--no-program"}
    else:
        report["boundary"] = {"status": "skipped", "reason": "no snapshots.json"}
        report["subcircuits"] = {"status": "skipped", "reason": "no snapshots.json"}
    report["structure"] = compare_structure(result, C=dict(profile.config), LP=a.lp, STEPS=a.steps, BN=a.bn)
    report["seconds"] = round(time.time() - t0, 3)
    report["max_rss_gb"] = round(_max_rss_gb(), 3)   # DESIGN §11: every full-size run documents its footprint
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(report), indent=1, sort_keys=False))
    for k in ("tokens", "boundary", "subcircuits", "structure"):
        print(f"{k:12s} {report[k]['status']}", file=sys.stderr)
    print(str(out))
    return 0 if all(report[k]["status"] in ("pass", "skipped") for k in ("tokens", "boundary", "subcircuits", "structure")) else 1
