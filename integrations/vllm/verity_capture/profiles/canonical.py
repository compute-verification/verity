"""[P6] The canonical, JSON-serialisable form of a κ profile: what two constructions of "the same profile" must agree on.

`canonical(profile, module)` covers exactly the load-bearing surface of a profile object -- the things the resolver / fold / gates
read -- and nothing of its free text:
  * `patterns`: the ordered pattern list (the resolver is first-match, `resolver.resolve_node`), each as its class name, `name` and
    every dataclass / instance field, recursively (nested patterns, pinned constexpr dicts, tables, callables by qualified name);
  * `config`: `C`;
  * `pins`: `PROFILE.notes["pins"]` (κ constexpr / geometry values);
  * `weights_type`: the leaf structure of `analytic.weights_type(C)`;
  * `weight_field` / `root_class`: probed over every root name any family's map knows (`PROBE_NAMES(C)`), so the whole map is compared
    by extension, not by identity of the closure;
  * `check_header`: probed over the profile's own HF config (must be consistent) and one perturbed copy per checked key;
  * `id`, `accepted`, `case`, the kernel allowlist's length and sha.
The per-role fixtures under `profiles/expected/<derived module name>.json` pin this form (`tests/test_profiles_generic.py`);
`python -m verity_capture.profiles.canonical --write` regenerates them from the generic constructor (a deliberate re-pin, reviewed in the diff).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import types
from typing import Any

from .. import analytic

#: every per-layer parameter suffix any family map knows (base Llama vocabulary, Qwen3 head norms, FP8 scales, Gemma-2 extra norms,
#: fused-MoE experts / router, OLMoE all-heads norms, GPT-NeoX LayerNorm-with-bias family) plus names none should bind
LAYER_SUFFIXES: tuple[str, ...] = (
    "input_layernorm.weight", "self_attn.qkv_proj.weight", "self_attn.qkv_proj.bias", "self_attn.o_proj.weight", "post_attention_layernorm.weight",
    "mlp.gate_up_proj.weight", "mlp.down_proj.weight", "self_attn.rotary_emb.cos_sin_cache",
    "self_attn.q_norm.weight", "self_attn.k_norm.weight",
    "self_attn.qkv_proj.weight_scale_inv", "self_attn.o_proj.weight_scale_inv", "mlp.gate_up_proj.weight_scale_inv", "mlp.down_proj.weight_scale_inv",
    "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
    "mlp.gate.weight", "mlp.experts.routed_experts.w13_weight", "mlp.experts.routed_experts.w2_weight",
    "mlp.shared_expert.gate_up_proj.weight", "mlp.gate.e_score_correction_bias", "mlp.experts.expert_map",
    "self_attn.attn.k_scale", "self_attn.attn._v_scale", "self_attn.attn.q_scale", "self_attn.attn.prob_scale",
    "input_layernorm.bias", "attention.query_key_value.weight", "attention.query_key_value.bias", "attention.dense.weight", "attention.dense.bias",
    "post_attention_layernorm.bias", "mlp.dense_h_to_4h.weight", "mlp.dense_h_to_4h.bias", "mlp.dense_4h_to_h.weight", "mlp.dense_4h_to_h.bias",
    "attention.rotary_emb.cos_sin_cache", "attention.attn.k_scale", "attention.attn._prob_scale",
    "self_attn.q_proj.weight", "mlp.up_proj.weight", "nonsense.weight",
)
TOP_NAMES: tuple[str, ...] = ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight", "model.normalizer",
                              "gpt_neox.embed_in.weight", "gpt_neox.final_layer_norm.weight", "gpt_neox.final_layer_norm.bias", "embed_out.weight",
                              "model.rotary_emb.cos_sin_cache", "nonsense")
LAYER_PREFIXES: tuple[str, ...] = ("model.layers", "gpt_neox.layers")


def probe_names(C: dict[str, Any]) -> list[str]:
    NL = int(C["NL"])
    layers = sorted({0, 1, NL - 1})
    out = list(TOP_NAMES)
    for pre in LAYER_PREFIXES:
        for L in layers:
            out += [f"{pre}.{L}.{s}" for s in LAYER_SUFFIXES]
    return out


def _jsonable(x: Any, depth: int = 0) -> Any:
    if depth > 12:
        return f"<depth>{type(x).__name__}"
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (list, tuple, set, frozenset)):
        items = list(x)
        if isinstance(x, (set, frozenset)):
            items = sorted(items, key=repr)
        return [_jsonable(v, depth + 1) for v in items]
    if isinstance(x, dict):
        return {str(k) if not isinstance(k, tuple) else repr(k): _jsonable(v, depth + 1) for k, v in sorted(x.items(), key=lambda kv: repr(kv[0]))}
    if isinstance(x, re.Pattern):
        return {"__re__": x.pattern}
    if isinstance(x, type):
        return {"__class__": f"{x.__module__}.{x.__qualname__}"}
    if isinstance(x, (types.FunctionType, types.MethodType, types.BuiltinFunctionType)):
        return {"__callable__": f"{getattr(x, '__module__', '?')}.{getattr(x, '__qualname__', repr(x))}"}
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {"__type__": type(x).__name__, **{f.name: _jsonable(getattr(x, f.name), depth + 1) for f in dataclasses.fields(x)}}
    if hasattr(x, "__dict__"):
        d = {k: v for k, v in vars(x).items() if not k.startswith("_")}
        cls_name = getattr(type(x), "name", None)
        if "name" not in d and isinstance(cls_name, str):
            d["name"] = cls_name
        return {"__type__": type(x).__name__, **{k: _jsonable(v, depth + 1) for k, v in sorted(d.items())}}
    return repr(x)


def pattern_entry(p: Any) -> dict[str, Any]:
    d = _jsonable(p)
    if not isinstance(d, dict):
        d = {"__type__": type(p).__name__, "repr": d}
    d.setdefault("name", getattr(p, "name", None))
    return d


def _type_tree(t: Any) -> Any:
    fields = getattr(t, "fields", None)
    if fields is not None:                                  # Record
        return {"Record": [(n, _type_tree(ft)) for n, ft in fields]}
    if hasattr(t, "elem") and hasattr(t, "n"):              # Array
        return {"Array": [int(t.n), _type_tree(t.elem)]}
    return repr(t)


def _header(cfg: dict[str, Any]):
    from ..events import RunHeader
    fields = {f.name for f in dataclasses.fields(RunHeader)}
    kw: dict[str, Any] = {"model": {"config": cfg}}
    for f in dataclasses.fields(RunHeader):
        if f.name != "model" and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:  # type: ignore[attr-defined]
            kw[f.name] = None
    return RunHeader(**{k: v for k, v in kw.items() if k in fields})


CHECKED_HF_KEYS = ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "intermediate_size", "vocab_size", "rms_norm_eps",
                   "layer_norm_eps", "rope_theta", "rotary_emb_base", "max_position_embeddings", "head_dim", "attention_bias", "tie_word_embeddings",
                   "attn_logit_softcapping", "final_logit_softcapping", "query_pre_attn_scalar", "sliding_window", "hidden_act", "rotary_pct",
                   "use_parallel_residual", "num_experts", "num_experts_per_tok", "norm_topk_prob", "clip_qkv", "moe_intermediate_size", "mlp_only_layers")


def check_header_probe(check_header, hf_config: dict[str, Any] | None) -> dict[str, Any]:
    if check_header is None or not hf_config:
        return {"available": check_header is not None, "hf_config": bool(hf_config)}
    out: dict[str, Any] = {"self": sorted(check_header(_header(dict(hf_config))))}
    for k in CHECKED_HF_KEYS:
        if k not in hf_config:
            continue
        v = hf_config[k]
        bad = (not v) if isinstance(v, bool) else (v + 1 if isinstance(v, (int, float)) else (v + "_x" if isinstance(v, str) else None))
        if bad is None:
            continue
        out[f"perturb:{k}"] = sorted(check_header(_header({**hf_config, k: bad})))
    return out


def canonical(profile: Any, module: Any = None, *, hf_config: dict[str, Any] | None = None) -> dict[str, Any]:
    C = dict(profile.config)
    wf = profile.weight_field
    rc = getattr(module, "root_class", None) if module is not None else None
    ch = getattr(module, "check_header", None) if module is not None else None
    if hf_config is None and module is not None:
        hf_config = getattr(module, "HF_CONFIG", None)
    names = probe_names(C)
    allow = tuple(profile.kernel_allowlist or ())
    return {
        "id": profile.id,
        "case": (profile.notes or {}).get("case"),
        "accepted": bool(profile.accepted),
        "kernel_allowlist": {"n": len(allow), "sha256": hashlib.sha256("\n".join(allow).encode()).hexdigest()[:16]},
        "config": _jsonable(C),
        "pins": _jsonable((profile.notes or {}).get("pins")),
        "patterns": [pattern_entry(p) for p in profile.patterns],
        "weights_type": _type_tree(analytic.weights_type(C)),
        # by extension over the probe set; the None / "unknown" answers are implied by absence (the set itself is `probe_names(C)`)
        "probe": {"n": len(names)},
        "weight_field": {n: _jsonable(p) for n in names if (p := wf(n)) is not None},
        "root_class": {n: c for n in names if (c := rc(n)) != "unknown"} if rc is not None else None,
        "check_header": check_header_probe(ch, hf_config),
    }


def dumps(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=1, sort_keys=True, default=repr) + "\n"


EXPECTED_DIR = __import__("pathlib").Path(__file__).resolve().parent / "expected"

#: the tensor-parallel worlds with recorded runs (the two `*_tp2rank` hand modules and `gen_ov_moe_qwen3`'s TP2 half, retired in P6)
TP_WORLDS: dict[str, tuple[int, ...]] = {"B1": (2,), "LLAMA32_1B": (2,), "QWEN3_30B_A3B": (2,)}


def fixture_roles() -> list[tuple[str, int]]:
    """Every role with a shipped HF config.json (`profiles/hf_configs/<ROLE>.config.json`), at TP1 and at the recorded TP worlds."""
    from . import dense_generic as _dg
    roles = sorted(p.name[: -len(".config.json")] for p in _dg.HF_CONFIG_DIR.glob("*.config.json"))
    return [(r, w) for r in roles for w in (1, *TP_WORLDS.get(r, ()))]


def canonical_of_role(role: str, world: int = 1) -> dict[str, Any]:
    from . import generic
    mod = generic.make_module(f"verity_capture.profiles.{generic.module_name(role, world)}")
    return canonical(mod.PROFILE, mod, hf_config=mod.HF_CONFIG)


def write_fixtures(roles: list[tuple[str, int]] | None = None) -> list[str]:
    from . import generic
    EXPECTED_DIR.mkdir(exist_ok=True)
    written = []
    for role, world in roles or fixture_roles():
        try:
            doc = canonical_of_role(role, world)
        except ImportError as e:
            print(f"skip {role} tp{world}: {e}")
            continue
        path = EXPECTED_DIR / f"{generic.module_name(role, world)}.json"
        path.write_text(dumps(doc))
        written.append(path.name)
    return written


if __name__ == "__main__":                                              # pragma: no cover - the re-pin CLI
    import sys
    if "--write" in sys.argv:
        for name in write_fixtures():
            print("wrote", name)
    else:
        for role, world in fixture_roles():
            print(json.dumps({"role": role, "world": world, "id": canonical_of_role(role, world)["id"]}))
