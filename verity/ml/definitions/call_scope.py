"""What `query.word` and `query.cross_call` share beyond core's `verity.ml.circuits.cut`: the registered Definition a spec names
(`specialization`), a spec id's parts (`spec_base`, `spec_statics`) and the violation both raise (`QueryRuleViolation`)."""
from __future__ import annotations

import json
from typing import Any, Callable

class QueryRuleViolation(ValueError):
    """A Program whose units cannot satisfy `Q_word_v1{X,W}`: named per Definition and module, never widened."""


def specialization(fam: str, statics: dict[str, Any]):
    """The registered Definition `fam` bound to `statics` (a primitive as it is); function-valued statics (`DOT`, rendered
    `{"fn": id}` or as the bare id) resolve to the registered Definition they name."""
    from verity.ml.circuits.codec import lookup_primitive
    from verity.ml.definitions.registry.catalog import load
    reg, _errors = load()
    d = reg.defs.get(fam)
    if d is None:
        return lookup_primitive(reg, fam)

    def fn_value(v: Any) -> Any:
        fid = v.get("fn") if isinstance(v, dict) else (json.loads(v).get("fn") if isinstance(v, str) and v[:5] == '{"fn"' else v)
        if not isinstance(fid, str):
            return v
        try:
            return lookup_primitive(reg, fid)
        except KeyError:
            return reg.defs[fid] if fid in reg.defs else v

    return d.bind(**{k: fn_value(v) for k, v in statics.items()}) if hasattr(d, "bind") else d


def spec_base(spec: str) -> str:
    """`Gemm_v1{K=2304,N=4096}` -> `Gemm_v1` (a fold's `prefix:` before the record dropped; a `:` inside the record is JSON)."""
    head = spec.split("{", 1)[0]
    return head.rsplit(":", 1)[-1]


def spec_record_id(spec: str) -> str:
    head, sep, rec = spec.partition("{")
    return head.rsplit(":", 1)[-1] + sep + rec


def static_value(v: Any) -> Any:
    if isinstance(v, dict):
        if set(v) == {"f64"}:
            return float.fromhex(v["f64"])
        if set(v) == {"fn"}:
            return v["fn"]
        if set(v) == {"dict"}:
            return {k: static_value(x) for k, x in v["dict"].items()}
        return {k: static_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return tuple(static_value(x) for x in v)
    return v


def spec_statics(spec: str) -> dict[str, Any]:
    """The binding record of a spec id as Python values (`{"f64": hex}` -> float, lists -> tuples), in id order."""
    from verity.ml.circuits.codec import _split_record
    out: dict[str, Any] = {}
    for k, v in _split_record(spec_record_id(spec)):
        try:
            out[k] = static_value(json.loads(v))
        except ValueError:
            out[k] = v
    return out


def bits_gates(group: dict[str, Any], specialize: Callable[[str, dict], Any], violations: list[dict[str, Any]]) -> int:
    """The Boolean gates per Call of a `query.word.check_calls` group's `bits` Definition (a Boolean Program's word link), 0 for a
    group without one; a `bits` that does not resolve is an `unresolved` violation, appended to ``violations``."""
    bits = group.get("bits")
    if not bits:
        return 0
    try:
        return int(specialize(spec_base(bits), spec_statics(bits)).gates)
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        violations.append({"definition": bits, "class": "unresolved", "why": f"{type(e).__name__}: {str(e)[:200]}"})
        return 0


__all__ = ["QueryRuleViolation", "bits_gates", "spec_base", "spec_statics", "specialization"]
