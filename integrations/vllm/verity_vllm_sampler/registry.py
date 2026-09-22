"""Plug `sampler.select` into W2's relation registry (`verity_vllm.relations`, branch harness).

W2 reserved the slot ``sampler.select.v1`` (class ``independent``, op_type ``sampler``, owner W3)
and raises ``RelationNotImplemented`` until somebody fills it.  Importing this module fills it
with the numpy re-derivation from :mod:`verity_vllm_sampler.select` and additionally registers
``sampler.select.same_runtime.v1`` (class ``same_runtime``; needs torch + vllm + a GPU at check
time, so it is only *usable* on the pod but is always *registered* so the runner reports it as
``error`` rather than silently skipping it).

Usage (integration owner, before ``relations.check``):

    import verity_vllm_sampler.registry as sampler_registry
    sampler_registry.install()

Description node this checker expects (W2 description schema, `positions: "sampled"`):

    {"op": "sampler.select", "layer": "sampler", "op_type": "sampler",
     "dtype": "int32", "shape": [], "positions": "sampled",
     "args": [{"name": "logits",          "source": "node",     "op": "lm_head.gemm", "port": 0},
              {"name": "sampling_params", "source": "boundary", "field": "sampling_params"},
              {"name": "seed",            "source": "boundary", "field": "seed"}],
     "relation": "sampler.select.v1"}

``inst.t`` is the absolute position of the input token whose logits are sampled
(= prompt_len - 1 + k for output k), which is exactly the ``pos`` vLLM's V2 sampler feeds to
``tl.randint(seed, pos)`` — so no prompt_len is needed.  ``sampling_params`` and ``seed`` are the
boundary tree's JSON leaves (authenticated by the runner before this checker sees them).

Certification: the independent checker returns bytes only when its prediction is *certified*
(no other token can overtake within the documented libdevice log/log1p and top-p boundary error
budgets).  An uncertified prediction raises ``RelationError`` and the runner records ``error``
(never ``exact``), so a lucky coincidence at an ambiguous position cannot become a PASS.  Set
``STRICT_CERTIFICATION = False`` to compare uncertified predictions too (they are then indistinguishable
from certified ones in the runner's counts; W3's own ``analyze.py`` keeps the two apart).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .select import independent_select, same_runtime_select

STRICT_CERTIFICATION = True

try:  # W2's package is on another branch; keep this module importable without it
    from verity_vllm import relations as _rel
    from verity_vllm.capture import Value as _Value
except Exception:  # pragma: no cover - exercised only when verity_vllm is absent
    _rel = None
    _Value = None

RELATION_INDEPENDENT = "sampler.select.v1"
RELATION_SAME_RUNTIME = "sampler.select.same_runtime.v1"


def _json_arg(inputs: dict[str, Any], name: str) -> Any:
    """A boundary JSON leaf (Value dtype json) or a const arg."""
    v = inputs.get(name)
    if v is None:
        raise _rel.RelationError(f"missing input {name!r}")
    if _Value is not None and isinstance(v, _Value):
        if v.dtype != "json":
            raise _rel.RelationError(f"{name}: expected a json boundary leaf, got {v.dtype}")
        return v.json()
    if isinstance(v, dict) and "const" in v:
        return v["const"]
    return v


def _params(inputs: dict[str, Any]) -> tuple[float, int, float, int | None]:
    # W2's toy description names the boundary arg `params`; ours says `sampling_params`.
    sp = _json_arg(inputs, "sampling_params" if "sampling_params" in inputs else "params") or {}
    if not isinstance(sp, dict):
        raise _rel.RelationError("sampling_params must be a JSON object")
    seed: Any = inputs.get("seed")
    seed = _json_arg(inputs, "seed") if seed is not None else sp.get("seed")
    for unsupported in ("min_p", "repetition_penalty", "frequency_penalty", "presence_penalty", "logit_bias",
                        "bad_words", "allowed_token_ids"):
        val = sp.get(unsupported)
        neutral = {"min_p": (0, 0.0), "repetition_penalty": (1, 1.0), "frequency_penalty": (0, 0.0),
                   "presence_penalty": (0, 0.0)}.get(unsupported, (None, [], {}))
        if val is not None and val not in neutral:
            raise _rel.RelationError(f"sampler.select: {unsupported}={val!r} is outside the frozen profile")
    T = float(sp.get("temperature", 1.0))
    k = int(sp.get("top_k", -1) if sp.get("top_k") is not None else -1)
    if k == 0:
        k = -1  # vLLM 0.28: top_k=0 means disabled (SamplingParams normalises 0 -> -1)
    p = float(sp.get("top_p", 1.0))
    return T, k, p, (None if seed is None else int(seed))


def _logits(inputs: dict[str, Any]) -> np.ndarray:
    v = _rel.need(inputs, "logits")
    if v.dtype not in ("bfloat16", "float32", "float16"):
        raise _rel.RelationError(f"logits dtype {v.dtype} not supported by sampler.select")
    x = v.float32().reshape(-1)
    if x.size == 0 or np.isnan(x).any():
        raise _rel.RelationError("empty or NaN logits")
    return x


def _encode(inst: Any, token: int) -> bytes:
    if inst.dtype not in ("int32", "int64"):
        raise _rel.RelationError(f"sampler output dtype {inst.dtype} must be int32 or int64")
    return _rel.encode_scalar(inst.dtype, int(token))


def _topk_ties(inputs: dict[str, Any]) -> str | None:
    """Which top-k tie rule the profile pins (module default TOPK_TIES; const arg `topk_ties` overrides).

    "exact_k" = the Triton kernel (POC pin 0.28.1rc1.dev472 at every batch size; 0.28.0 at >= 8 rows);
    "keep_all" = the torch sort path (0.28.0 below 8 rows); None = not pinned: the checker certifies
    only positions where both rules select the same token."""
    v = inputs.get("topk_ties")
    if v is None:
        return TOPK_TIES
    v = v["const"] if isinstance(v, dict) and "const" in v else v
    return None if v in (None, "unpinned") else str(v)


TOPK_TIES: str | None = "exact_k"


def sampler_select_independent(inst: Any, inputs: dict[str, Any]) -> bytes:
    T, k, p, seed = _params(inputs)
    x = _logits(inputs)
    res = independent_select(x, T, k, p, seed, int(inst.t), topk_ties=_topk_ties(inputs))
    if res.predicted < 0:
        raise _rel.RelationError(res.reason or "sampler.select: no prediction")
    if STRICT_CERTIFICATION and not res.certified:
        raise _rel.RelationError(f"sampler.select: uncertified prediction {res.predicted} "
                                 f"(margin={res.margin}): {res.reason}")
    return _encode(inst, res.predicted)


def sampler_select_same_runtime(inst: Any, inputs: dict[str, Any]) -> bytes:
    T, k, p, seed = _params(inputs)
    x = _logits(inputs)
    try:
        res = same_runtime_select(x, T, k, p, seed, int(inst.t))
    except ImportError as e:  # torch / vllm absent on the verifier host
        raise _rel.RelationError(f"same_runtime sampler needs torch+vllm+GPU: {e}") from e
    if res.predicted < 0:
        raise _rel.RelationError(res.reason or "sampler.select: no prediction")
    return _encode(inst, res.predicted)


_SAVED: dict[str, Any] = {}


def install() -> list[str]:
    """Register both checkers (idempotent).  Returns the relation ids registered.

    Explicit rather than an import-time side effect so W2's registry tests (which assert the slot
    is unfilled) are unaffected by having this module on the path; the integration owner calls
    ``verity_vllm_sampler.registry.install()`` once before ``relations.check``.
    """
    if _rel is None:
        raise ImportError("verity_vllm (W2 harness) is not importable; nothing to plug into")
    for rid in (RELATION_INDEPENDENT, RELATION_SAME_RUNTIME):
        _SAVED.setdefault(rid, _rel.REGISTRY.get(rid))
    existing = _rel.REGISTRY.get(RELATION_INDEPENDENT)
    _rel.register(RELATION_INDEPENDENT, klass=_rel.INDEPENDENT, op_type="sampler", includes_full_op=False,
                  owner="W3", replace=bool(existing and existing.implemented),
                  doc="vLLM 0.28.0 V2 sampler re-derivation in numpy: fp32 temperature, top-k/top-p masks, "
                      "Philox4x32-10 Gumbel noise keyed by (seed, pos=t), first-index argmax; greedy = first argmax. "
                      "Returns bytes only for certified predictions (error otherwise).")(sampler_select_independent)
    existing = _rel.REGISTRY.get(RELATION_SAME_RUNTIME)
    _rel.register(RELATION_SAME_RUNTIME, klass=_rel.SAME_RUNTIME, op_type="sampler", includes_full_op=True,
                  owner="W3", replace=bool(existing and existing.implemented),
                  doc="vLLM's own apply_top_k_top_p + gumbel_sample on the opened logits row (GPU, fresh process).")(
        sampler_select_same_runtime)
    return [RELATION_INDEPENDENT, RELATION_SAME_RUNTIME]


def uninstall() -> None:
    """Restore whatever the registry held before :func:`install` (the unimplemented W3 slot)."""
    if _rel is None:
        return
    for rid, prev in _SAVED.items():
        if prev is None:
            _rel.REGISTRY.pop(rid, None)
        else:
            _rel.REGISTRY[rid] = prev
    _SAVED.clear()
