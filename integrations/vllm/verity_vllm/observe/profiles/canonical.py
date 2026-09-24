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
`python -m verity_vllm.observe.profiles.canonical --write` regenerates them from the generic constructor (a deliberate re-pin, reviewed in the diff).
"""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import re
import types
from typing import Any

from verity_vllm.input_provenance import analytic

#: the names no map should bind (a probe of the maps' refusals), and the quantization / attention-scale placeholders the base map knows
#: by shape rather than by family: the family and base vocabularies themselves are read off the maps (`probe_vocabulary`)
_UNBOUND_PROBES: tuple[str, ...] = ("self_attn.q_proj.weight", "mlp.up_proj.weight", "nonsense.weight")
#: one name per MoE variant `generic._UNMODELLED_MOE` refuses (shared experts, router bias, expert-parallel map)
_MOE_VARIANT_PROBES: tuple[str, ...] = ("mlp.shared_expert.gate_up_proj.weight", "mlp.gate.e_score_correction_bias", "mlp.experts.expert_map")
_SCALE_SUFFIX = "_scale_inv"
_ATTN_SCALE_NAMES: tuple[str, ...] = ("k_scale", "_v_scale", "q_scale", "prob_scale", "_prob_scale")
_TOP_PROBES: tuple[str, ...] = ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight", "model.rotary_emb.cos_sin_cache", "nonsense")


def probe_vocabulary() -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """(layer prefixes, per-layer suffixes, top-level names): the union over the base Llama map and every family's declared parameter names
    (`family_facts.FAMILY_FACTS[*]["params"]`), plus the FP8 block scales of the base linears, the attention-scale placeholders and the
    unbound probes.  Derived from the maps, so a new family's names enter the probe set with its facts entry."""
    from . import family_facts as FF, vllm_d9105ea80_sm89_eager as base
    prefixes: list[str] = ["model.layers"]
    suffixes: list[str] = [*base._LAYER_FIELDS, base.COS_SIN_BUFFER]
    suffixes += [f"{k}{_SCALE_SUFFIX}" for k in base._LAYER_FIELDS if k.endswith(".weight") and "norm" not in k]
    tops: list[str] = list(_TOP_PROBES)
    attn_scales: list[str] = ["self_attn.attn"]
    for F in FF.FAMILY_FACTS.values():
        P = F.get("params") or {}
        if P.get("layer_prefix"):
            prefixes.append(P["layer_prefix"])
        suffixes += list(P.get("layer") or {})
        tops += list(P.get("top") or {})
        if P.get("cos_sin"):
            suffixes.append(P["cos_sin"])
        if P.get("attn_scales"):
            attn_scales.append(P["attn_scales"])
    suffixes += [f"{a}.{s}" for a in attn_scales for s in _ATTN_SCALE_NAMES]
    suffixes += [*_MOE_VARIANT_PROBES, *_UNBOUND_PROBES]
    return tuple(dict.fromkeys(prefixes)), tuple(dict.fromkeys(suffixes)), tuple(dict.fromkeys(tops))


def probe_names(C: dict[str, Any]) -> list[str]:
    NL = int(C["NL"])
    layers = sorted({0, 1, NL - 1})
    prefixes, suffixes, tops = probe_vocabulary()
    out = list(tops)
    for pre in prefixes:
        for L in layers:
            out += [f"{pre}.{L}.{s}" for s in suffixes]
    return out


def _field(name: str, v: Any) -> Any:
    # a definition's `doc` is `fn.__doc__`, which Python >= 3.13 stores dedented and 3.12 does not: pin the dedented form on both
    return inspect.cleandoc(v) if name == "doc" and isinstance(v, str) else v


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
        return {"__type__": type(x).__name__, **{f.name: _jsonable(_field(f.name, getattr(x, f.name)), depth + 1) for f in dataclasses.fields(x)}}
    if hasattr(x, "__dict__"):
        d = {k: v for k, v in vars(x).items() if not k.startswith("_")}
        cls_name = getattr(type(x), "name", None)
        if "name" not in d and isinstance(cls_name, str):
            d["name"] = cls_name
        return {"__type__": type(x).__name__, **{k: _jsonable(_field(k, v), depth + 1) for k, v in sorted(d.items())}}
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

def fixture_roles() -> list[tuple[str, int]]:
    """Every role with a shipped HF config.json (`profiles/hf_configs/<ROLE>.config.json`) at TP1, plus every (role, world) a fixture
    already pins (the tensor-parallel worlds with recorded runs are data -- the fixture files -- not a table here; pin a new world with
    `python -m verity_vllm.observe.profiles.canonical --write ROLE:W`)."""
    from . import dense_generic as _dg, generic
    roles = sorted(p.name[: -len(".config.json")] for p in _dg.HF_CONFIG_DIR.glob("*.config.json"))
    out = {(r, 1) for r in roles}
    for p in EXPECTED_DIR.glob("*.json"):
        parsed = generic.parse_module(f"verity_vllm.observe.profiles.{p.stem}")
        if parsed is not None:
            out.add(parsed[:2])
    return sorted(out)


def canonical_of_role(role: str, world: int = 1) -> dict[str, Any]:
    from . import generic
    mod = generic.make_module(f"verity_vllm.observe.profiles.{generic.module_name(role, world)}")
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
